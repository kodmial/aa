"""Local dense retrieval over RU canonical chunks (issue #17).

Production backend: pinned local ``intfloat/multilingual-e5-base``
(``corpus/embedding.lock.json``) with L2-normalized embeddings and exact
``faiss.IndexFlatIP`` search. Local/keyless only: the model is loaded
exclusively from the local Hugging Face hub cache with networking
disabled; no remote embedding service is ever contacted.

Offline/test backend: deterministic hashing embeddings (also normalized,
also exact inner-product search). The hashing backend keeps unit tests
and CI hermetic without downloading weights; the production e5 path is
selected automatically whenever the pinned model plus ``transformers``
and ``torch`` are locally available.

The exact-IP index uses ``faiss.IndexFlatIP`` when the ``faiss`` package
is importable and a bit-exact pure-Python equivalent otherwise.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from aa.retrieval.normalize import ru_stem, ru_tokens

DENSE_TOP_K = 40
HASHING_DIM = 256
HASHING_BACKEND_NAME = "hashing-char-token/1"
E5_BACKEND_NAME = "intfloat-multilingual-e5-base/1"

# Per-snapshot loaded e5 (tokenizer, model) pairs. Ordinary turns reuse
# the built index and these cached weights; nothing is reloaded per turn.
_E5_LOADED: dict[str, tuple[Any, Any]] = {}


class DenseError(ValueError):
    """Raised when dense embeddings or search cannot be served locally."""


def l2_normalize(vector: list[float]) -> list[float]:
    """Return the L2-normalized copy of ``vector`` (fails on zero norm)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise DenseError("refusing to normalize a zero vector")
    return [value / norm for value in vector]


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def hashing_embed(text: str, *, dim: int = HASHING_DIM) -> list[float]:
    """Deterministic normalized hashing embedding (local, no weights).

    Combines whole-token buckets (weight 1.0, stemmed so inflections
    collide) with character-trigram buckets (weight 0.5, typo tolerant).
    """
    if dim <= 0:
        raise DenseError("dim must be > 0")
    vector = [0.0] * dim
    tokens = [ru_stem(token) for token in ru_tokens(text)]
    for token in tokens:
        vector[_bucket("tok:" + token, dim)] += 1.0
    collapsed = "".join(tokens)
    if len(collapsed) >= 3:
        for offset in range(len(collapsed) - 2):
            vector[_bucket("tri:" + collapsed[offset : offset + 3], dim)] += 0.5
    else:
        for token in tokens:
            vector[_bucket("tok:" + token, dim)] += 0.5
    if all(value == 0.0 for value in vector):
        # Empty/alphanumeric-free input: deterministic fallback direction.
        vector[_bucket("empty:", dim)] = 1.0
    return l2_normalize(vector)


def is_e5_available() -> bool:
    """Return True when the local e5 stack is importable (no import error)."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None and (
        importlib.util.find_spec("transformers") is not None
    )


def is_faiss_available() -> bool:
    """Return True when ``faiss`` is importable."""
    import importlib.util

    return importlib.util.find_spec("faiss") is not None


def e5_embed(texts: list[str], *, model_dir: str | None = None) -> list[list[float]]:
    """Embed ``texts`` with the pinned local e5 model (normalized).

    Loads exclusively from the local HF cache (``HF_HUB_OFFLINE=1``,
    ``local_files_only=True``); raises :class:`DenseError` when the
    stack or the pinned snapshot is unavailable instead of touching
    the network. Loaded weights are cached per snapshot directory so
    ordinary turns reuse them instead of reloading per query.
    """
    import importlib
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        torch_mod = importlib.import_module("torch")
        transformers_mod = importlib.import_module("transformers")
    except Exception as exc:
        raise DenseError(f"local e5 stack is unavailable: {exc}") from exc
    AutoModel = transformers_mod.AutoModel
    AutoTokenizer = transformers_mod.AutoTokenizer
    from aa.corpus.public_cache import (
        default_lock_path,
        load_embedding_lock,
        resolve_hf_cache_dir,
        resolve_model_root,
        snapshot_dir,
        verify_cached_model,
    )

    lock_path = model_dir
    try:
        lock = load_embedding_lock(default_lock_path() if lock_path is None else lock_path)
    except Exception as exc:
        raise DenseError(f"embedding lock is invalid: {exc}") from exc
    model_root = resolve_model_root(resolve_hf_cache_dir())
    if not verify_cached_model(model_root, lock):
        raise DenseError("pinned e5 snapshot is not cached locally; refusing network fetch")
    snapshot = snapshot_dir(model_root, str(lock.get("revision")))
    cache_key = str(snapshot)
    cached = _E5_LOADED.get(cache_key)
    if cached is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False
            )
            model = AutoModel.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False
            )
        except Exception as exc:
            raise DenseError(f"cannot load pinned e5 snapshot locally: {exc}") from exc
        model.eval()
        cached = (tokenizer, model)
        _E5_LOADED[cache_key] = cached
    else:
        tokenizer, model = cached
    vectors: list[list[float]] = []
    with torch_mod.no_grad():
        for text in texts:
            encoded = tokenizer(
                "query: " + text, return_tensors="pt", truncation=True, max_length=512
            )
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normalized = torch_mod.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append([float(value) for value in normalized[0].tolist()])
    return vectors


@dataclass
class ExactIPIndex:
    """Exact inner-product index with Faiss-compatible semantics.

    Uses ``faiss.IndexFlatIP`` when importable, otherwise a bit-exact
    pure-Python scan (same normalized-IP ranking contract). Vectors must
    be L2-normalized; scores are cosine similarities.
    """

    dim: int
    ids: list[str]
    vectors: list[list[float]]
    use_faiss: bool = False
    _faiss_index: object = None

    @classmethod
    def build(cls, ids: list[str], vectors: list[list[float]]) -> ExactIPIndex:
        """Build the index (validates normalization and uniqueness)."""
        if not ids:
            raise DenseError("refusing to build an empty dense index")
        if len(ids) != len(vectors):
            raise DenseError("ids and vectors must be parallel")
        if len(set(ids)) != len(ids):
            raise DenseError("dense index ids must be unique")
        dim = len(vectors[0])
        if dim <= 0:
            raise DenseError("dense vectors must be non-empty")
        for vector in vectors:
            if len(vector) != dim:
                raise DenseError("dense vectors must share one dimension")
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
                raise DenseError("dense vectors must be L2-normalized")
        use_faiss = False
        faiss_index: object = None
        try:
            import importlib

            faiss_mod = importlib.import_module("faiss")
            np_mod = importlib.import_module("numpy")

            index = faiss_mod.IndexFlatIP(dim)

            matrix = np_mod.array(vectors, dtype=np_mod.float32)
            index.add(matrix)
            use_faiss = True
            faiss_index = index
        except Exception:
            use_faiss = False
            faiss_index = None
        return cls(
            dim=dim,
            ids=list(ids),
            vectors=[list(v) for v in vectors],
            use_faiss=use_faiss,
            _faiss_index=faiss_index,
        )

    def search(self, query: list[float], *, top_k: int) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, ip_score)]`` best-first (exact search)."""
        if top_k <= 0:
            raise DenseError("top_k must be > 0")
        if len(query) != self.dim:
            raise DenseError("query dimension does not match the index")
        if self.use_faiss and self._faiss_index is not None:
            try:
                import importlib

                np_mod = importlib.import_module("numpy")

                matrix = np_mod.array([query], dtype=np_mod.float32)
                scores, indices = self._faiss_index.search(  # type: ignore[attr-defined]
                    matrix, min(top_k, len(self.ids))
                )
                out: list[tuple[str, float]] = []
                for position, score in zip(indices[0].tolist(), scores[0].tolist(), strict=True):
                    if position < 0:
                        continue
                    out.append((self.ids[position], float(score)))
                return out
            except Exception:
                pass
        scored = [
            (chunk_id, sum(q * v for q, v in zip(query, vector, strict=True)))
            for chunk_id, vector in zip(self.ids, self.vectors, strict=True)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def search_text(
        self, text: str, *, top_k: int, embed: str = "hashing"
    ) -> list[tuple[str, float]]:
        """Embed ``text`` with the hashing backend and search the index."""
        del embed
        return self.search(hashing_embed(text, dim=self.dim), top_k=top_k)
