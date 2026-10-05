"""RU-first hybrid retrieval index with aligned EN control (issue #115).

Fixed production baseline per planner aspect:

1. preserve/use the original Russian query;
2. add same-language Russian rewrites from #46;
3. lexical retrieval: RAM-resident SQLite FTS5/BM25 over RU canonical chunks;
4. semantic retrieval: pinned local ``intfloat/multilingual-e5-base`` over
   RU canonical chunks, normalized embeddings + exact in-memory
   ``faiss.IndexFlatIP``;
5. fuse with RRF (``k=60``);
6. deduplicate overlaps and retain section/chapter diversity.

Fixed search parameters: lexical top 40, dense top 40, RRF ``k=60``,
max 12 compact candidates per aspect. Ordinary turns reuse the RAM-resident
index; search never rebuilds or re-embeds the corpus and never reads
``lexical.db`` / canonical text from disk.

Canonical substrate: RU source -> standardized razdel sentence segmentation
-> E5-token child chunks (256-token baseline) -> exact child Documents ->
RAM SQLite FTS5 + RAM E5/FAISS -> structured ``RetrievalHit`` with exact
canonical Russian text plus provenance.

EN is reference/control only: the index carries aligned EN section
metadata so #47 can benchmark an optional EN secondary discovery
branch, but the Russian path never routes RU queries through EN.

On-disk layout (all under ``corpus/generated/retrieval/``, ignored by
Git, never stored in Actions cache; startup/persistence only):

- ``index.json`` — version metadata, RU chunk records with exact text,
  and EN control metadata;
- ``lexical.db`` — persisted SQLite FTS5/BM25 table (copied to ``:memory:``
  at startup via the backup mechanism);
- ``dense.json`` — normalized dense vectors parallel to chunk ids (loaded
  into one long-lived FAISS ``IndexFlatIP`` at startup).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aa.retrieval.dense import (
    DENSE_TOP_K,
    E5_BACKEND_NAME,
    HASHING_BACKEND_NAME,
    HASHING_DIM,
    DenseError,
    ExactIPIndex,
    e5_embed,
    hashing_embed,
    l2_normalize,
)
from aa.retrieval.fusion import (
    MAX_CANDIDATES_PER_ASPECT,
    MAX_PER_SECTION,
    RRF_K,
    deduplicate_overlaps,
    enforce_diversity,
    rrf_fuse,
)
from aa.retrieval.lexical import (
    FTS_TABLE,
    LEXICAL_TOP_K,
    build_lexical_db,
    lexical_search_conn,
    load_lexical_into_memory,
)
from aa.retrieval.planner import QueryPlan, aspect_search_queries

INDEX_FORMAT = "aa-hybrid-index/2"
INDEX_BUILDER_VERSION = 2
LEGACY_INDEX_FORMAT = "aa-hybrid-index/1"
LEGACY_BUILDER_VERSION = 1
PREVIEW_CHARS = 240

LEXICAL_DB_NAME = "lexical.db"
DENSE_NAME = "dense.json"
INDEX_NAME = "index.json"


class HybridIndexError(ValueError):
    """Raised when the hybrid index cannot be built, opened or searched."""


class StaleIndexError(HybridIndexError):
    """Raised when the index no longer matches pinned corpus/model versions."""


@dataclass(frozen=True)
class ChunkRecord:
    """One RU canonical child document with exact text and provenance."""

    chunk_id: str
    logical_chunk_id: str
    section: str
    book: str
    parent: str
    prev: str | None
    next: str | None
    source_id: str
    source_file: str
    source_sha256: str
    char_start: int
    char_end: int
    text_sha256: str
    text: str
    corpus_version: str
    tokens: int | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """One fused child-document hit with exact canonical Russian text.

    The exact ``text`` plus provenance is the evidence substrate; ranking
    fields are internal metadata. ``preview`` is retained as navigation-only
    text and must never substitute for ``text`` in generation.
    """

    logical_chunk_id: str
    chunk_id: str
    section: str
    source_id: str
    source_file: str
    char_start: int
    char_end: int
    text_sha256: str
    text: str
    lexical_rank: int | None
    dense_rank: int | None
    lexical_score: float | None
    dense_score: float | None
    fused_rank: int
    fused_score: float
    preview: str
    parent: str
    prev: str | None
    next: str | None
    index_version: int
    ru_corpus_version: str
    embedding_model: str
    en_control_section: str
    en_control_title: str

    @property
    def section_id(self) -> str:
        """Return the section id (contract alias)."""
        return self.section

    @property
    def parent_id(self) -> str:
        """Return the parent paragraph id (contract alias)."""
        return self.parent

    @property
    def prev_id(self) -> str | None:
        """Return the previous chunk id (contract alias)."""
        return self.prev

    @property
    def next_id(self) -> str | None:
        """Return the next chunk id (contract alias)."""
        return self.next

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable hit contract (exact text included)."""
        return {
            "logical_chunk_id": self.logical_chunk_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "section_id": self.section,
            "text": self.text,
            "ru_locator": {
                "source_id": self.source_id,
                "source_file": self.source_file,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "text_sha256": self.text_sha256,
            },
            "component_ranks": {
                "lexical_rank": self.lexical_rank,
                "dense_rank": self.dense_rank,
                "lexical_score": self.lexical_score,
                "dense_score": self.dense_score,
            },
            "fused_rank": self.fused_rank,
            "fused_score": self.fused_score,
            "preview": self.preview,
            "neighbors": {
                "parent": self.parent,
                "parent_id": self.parent,
                "prev": self.prev,
                "prev_id": self.prev,
                "next": self.next,
                "next_id": self.next,
            },
            "versions": {
                "index_version": self.index_version,
                "ru_corpus_version": self.ru_corpus_version,
                "embedding_model": self.embedding_model,
            },
            "en_control": {
                "section": self.en_control_section,
                "title": self.en_control_title,
                "role": "reference-control",
            },
        }


@dataclass
class HybridIndex:
    """Opened RAM-resident RU-first hybrid index (reused across turns).

    After :func:`open_hybrid_index` returns, all hot-path data lives in
    process memory: canonical child text plus metadata/neighbor maps in
    ``chunks``, the FTS5 table in ``lexical_conn`` (``:memory:``), dense
    vectors in ``dense`` (long-lived FAISS ``IndexFlatIP``), and the cached
    E5 tokenizer/model reused by queries. Ordinary search/read/neighbor
    lookup performs no filesystem reads.
    """

    directory: Path
    metadata: dict[str, Any]
    chunks: dict[str, ChunkRecord]
    dense: ExactIPIndex
    lexical_conn: sqlite3.Connection | None = None
    ram_resident: bool = False
    en_control: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed RU chunks."""
        return len(self.chunks)


def logical_chunk_id(physical_ru_id: str) -> str:
    """Return the language-neutral logical id for a RU physical chunk id."""
    # Physical: "<section>:ru:c0007" -> logical "<section>:c0007".
    parts = physical_ru_id.split(":")
    if len(parts) == 3 and parts[1] == "ru":
        return f"{parts[0]}:{parts[2]}"
    return physical_ru_id


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HybridIndexError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HybridIndexError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HybridIndexError(f"{label} must be a JSON object: {path}")
    return payload


def extract_ru_chunks(
    full_structure: dict[str, Any],
) -> tuple[list[ChunkRecord], dict[str, dict[str, Any]]]:
    """Extract RU chunk records plus EN section control from full structure."""
    sections = full_structure.get("sections")
    if not isinstance(sections, list) or not sections:
        raise HybridIndexError("full structure has no sections list")
    records: list[ChunkRecord] = []
    en_control: dict[str, dict[str, Any]] = {}
    for entry in sections:
        if not isinstance(entry, dict):
            raise HybridIndexError("full structure has a malformed section")
        section_id = str(entry.get("id"))
        ru_branch = entry.get("ru")
        en_branch = entry.get("en")
        if not isinstance(ru_branch, dict) or not isinstance(en_branch, dict):
            raise HybridIndexError(f"section {section_id!r} is missing ru/en branches")
        titles = entry.get("titles")
        en_title = ""
        if isinstance(titles, dict) and isinstance(titles.get("en"), str):
            en_title = str(titles["en"])
        en_control[section_id] = {
            "section": section_id,
            "title": en_title or str(en_branch.get("title", "")),
            "text_sha256": str(en_branch.get("text_sha256", "")),
            "source_id": str(en_branch.get("source_id", "")),
            "role": "reference-control",
            "chunk_count": len(en_branch.get("chunks", []))
            if isinstance(en_branch.get("chunks"), list)
            else 0,
        }
        ru_chunks = ru_branch.get("chunks")
        if not isinstance(ru_chunks, list) or not ru_chunks:
            raise HybridIndexError(f"section {section_id!r} has no RU chunks")
        for node in ru_chunks:
            if not isinstance(node, dict):
                raise HybridIndexError("RU chunk node must be an object")
            for key in (
                "id",
                "parent",
                "section",
                "source_id",
                "source_file",
                "source_sha256",
                "text",
                "text_sha256",
                "corpus_version",
                "char_start",
                "char_end",
            ):
                if node.get(key) is None:
                    raise HybridIndexError(f"RU chunk is missing {key}")
            text = str(node["text"])
            if _sha256_text(text) != str(node["text_sha256"]):
                raise HybridIndexError(f"RU chunk checksum mismatch: {node.get('id')!r}")
            physical = str(node["id"])
            tokens_raw = node.get("tokens")
            records.append(
                ChunkRecord(
                    chunk_id=physical,
                    logical_chunk_id=logical_chunk_id(physical),
                    section=str(node["section"]),
                    book=str(node.get("book", "aa-big-book")),
                    parent=str(node["parent"]),
                    prev=str(node["prev"]) if node.get("prev") is not None else None,
                    next=str(node["next"]) if node.get("next") is not None else None,
                    source_id=str(node["source_id"]),
                    source_file=str(node["source_file"]),
                    source_sha256=str(node["source_sha256"]),
                    char_start=int(str(node["char_start"])),
                    char_end=int(str(node["char_end"])),
                    text_sha256=str(node["text_sha256"]),
                    text=text,
                    corpus_version=str(node["corpus_version"]),
                    tokens=int(str(tokens_raw)) if tokens_raw is not None else None,
                )
            )
    if not records:
        raise HybridIndexError("no RU chunks extracted")
    return records, en_control


def _embed_backend_vectors(
    texts: list[str], *, backend: str, dim: int
) -> tuple[list[list[float]], str, int]:
    """Embed chunk texts with the selected local backend (normalized)."""
    if backend == "hashing":
        return [hashing_embed(text, dim=dim) for text in texts], HASHING_BACKEND_NAME, dim
    if backend == "e5":
        try:
            vectors = e5_embed(["passage: " + text for text in texts])
        except DenseError as exc:
            raise HybridIndexError(str(exc)) from exc
        normalized = [l2_normalize(vector) for vector in vectors]
        return normalized, E5_BACKEND_NAME, len(normalized[0])
    if backend == "auto":
        try:
            vectors = e5_embed(["passage: " + text for text in texts])
            normalized = [l2_normalize(vector) for vector in vectors]
            return normalized, E5_BACKEND_NAME, len(normalized[0])
        except DenseError:
            return (
                [hashing_embed(text, dim=dim) for text in texts],
                HASHING_BACKEND_NAME,
                dim,
            )
    raise HybridIndexError(f"unknown embedding backend: {backend!r}")


def _chunking_proof_from_structure(full_structure: dict[str, Any]) -> dict[str, Any]:
    """Extract chunker/tokenizer proof fields from the full structure."""
    proof: dict[str, Any] = {
        "sentence_segmenter": str(
            full_structure.get("sentence_segmenter", full_structure.get("sentence_rule", ""))
        ),
        "sentence_rule": str(full_structure.get("sentence_rule", "")),
        "chunker": str(full_structure.get("chunker", "")),
        "chunker_version": full_structure.get("chunker_version"),
        "tokenizer": str(full_structure.get("tokenizer", "")),
        "chunk_policy": str(full_structure.get("chunk_policy", "")),
        "chunk_max_tokens": full_structure.get("chunk_max_tokens"),
        "e5_hard_input_tokens": full_structure.get("e5_hard_input_tokens"),
    }
    # Fail closed on missing proof: no pinned-default backfill here, so a
    # structure without segmenter/chunker/tokenizer/chunk-limit provenance
    # reaches the build_hybrid_index checks below and raises instead of
    # building with claimed provenance.
    return proof


def build_hybrid_index(
    full_structure: dict[str, Any],
    *,
    ru_manifest: dict[str, Any],
    en_manifest: dict[str, Any],
    embedding_lock: dict[str, Any],
    out_dir: str | Path,
    backend: str = "hashing",
    dim: int = HASHING_DIM,
) -> HybridIndex:
    """Build the RAM-resident RU-first hybrid index (generated artifact)."""
    records, en_control = extract_ru_chunks(full_structure)
    ru_artifact = str(ru_manifest.get("artifact_sha256", ""))
    en_artifact = str(en_manifest.get("artifact_sha256", ""))
    if not ru_artifact or not en_artifact:
        raise HybridIndexError("manifests must carry artifact_sha256")
    model_id = str(embedding_lock.get("model_id", ""))
    revision = str(embedding_lock.get("revision", ""))
    if not model_id or not revision:
        raise HybridIndexError("embedding lock must carry model_id + revision")
    proof = _chunking_proof_from_structure(full_structure)
    if not proof.get("sentence_segmenter") or not proof.get("chunker"):
        raise HybridIndexError("full structure must carry chunker/segmenter proof")
    if not proof.get("tokenizer"):
        raise HybridIndexError("full structure must carry tokenizer identity")
    if (
        proof.get("chunker_version") is None
        or proof.get("chunk_max_tokens") is None
        or proof.get("e5_hard_input_tokens") is None
    ):
        raise HybridIndexError("full structure must carry chunk-limit proof")
    if not proof.get("chunk_policy"):
        raise HybridIndexError("full structure must carry chunk-policy proof")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    chunk_ids = [record.chunk_id for record in records]
    sections = [record.section for record in records]
    texts = [record.text for record in records]
    build_lexical_db(
        directory / LEXICAL_DB_NAME, chunk_ids=chunk_ids, sections=sections, texts=texts
    )

    vectors, backend_name, used_dim = _embed_backend_vectors(texts, backend=backend, dim=dim)
    dense_index = ExactIPIndex.build(chunk_ids, vectors, backend=backend_name)
    dense_payload = {
        "format": "aa-dense-vectors/1",
        "backend": backend_name,
        "dim": used_dim,
        "ids": chunk_ids,
        "vectors": vectors,
    }
    (directory / DENSE_NAME).write_text(
        json.dumps(dense_payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    try:
        from aa.qualification.sentence_qualification import FIXTURE_VERSION as _fixture_version
    except Exception:
        _fixture_version = "aa-sentence-boundary-fixture/1"
    metadata: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "builder_version": INDEX_BUILDER_VERSION,
        "ru_artifact_sha256": ru_artifact,
        "ru_manifest_format": str(ru_manifest.get("format", "")),
        "en_artifact_sha256": en_artifact,
        "en_manifest_format": str(en_manifest.get("format", "")),
        "structure_format": str(full_structure.get("format", "")),
        "structure_builder_version": full_structure.get("builder_version"),
        "embedding_model_id": model_id,
        "embedding_revision": revision,
        "embedding_backend": backend_name,
        "embedding_dim": used_dim,
        "sentence_segmenter": proof.get("sentence_segmenter"),
        "sentence_rule": proof.get("sentence_rule"),
        "sentence_fixture_version": _fixture_version,
        "chunker": proof.get("chunker"),
        "chunker_version": proof.get("chunker_version"),
        "tokenizer": proof.get("tokenizer"),
        "tokenizer_model": model_id,
        "tokenizer_revision": revision,
        "chunk_policy": proof.get("chunk_policy"),
        "chunk_max_tokens": proof.get("chunk_max_tokens"),
        "e5_hard_input_tokens": proof.get("e5_hard_input_tokens"),
        "ram_resident": True,
        "search_params": {
            "lexical_top_k": LEXICAL_TOP_K,
            "dense_top_k": DENSE_TOP_K,
            "rrf_k": RRF_K,
            "max_per_aspect": MAX_CANDIDATES_PER_ASPECT,
            "max_per_section": MAX_PER_SECTION,
        },
        "chunk_count": len(records),
        "en_control": en_control,
    }
    index_payload = {
        "metadata": metadata,
        "chunks": [
            {
                "chunk_id": record.chunk_id,
                "logical_chunk_id": record.logical_chunk_id,
                "section": record.section,
                "book": record.book,
                "parent": record.parent,
                "prev": record.prev,
                "next": record.next,
                "source_id": record.source_id,
                "source_file": record.source_file,
                "source_sha256": record.source_sha256,
                "char_start": record.char_start,
                "char_end": record.char_end,
                "text_sha256": record.text_sha256,
                "text": record.text,
                "corpus_version": record.corpus_version,
                "tokens": record.tokens,
            }
            for record in records
        ],
    }
    (directory / INDEX_NAME).write_text(
        json.dumps(index_payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chunks = {record.chunk_id: record for record in records}
    lexical_conn = load_lexical_into_memory(directory / LEXICAL_DB_NAME)
    return HybridIndex(
        directory=directory,
        metadata=metadata,
        chunks=chunks,
        dense=dense_index,
        lexical_conn=lexical_conn,
        ram_resident=True,
        en_control=en_control,
    )


def _expected_from_live(
    *,
    ru_manifest_path: Path | None,
    en_manifest_path: Path | None,
    lock_path: Path | None,
) -> dict[str, str]:
    expected: dict[str, str] = {}
    if ru_manifest_path is not None:
        manifest = _load_json(ru_manifest_path, "RU manifest")
        expected["ru_artifact_sha256"] = str(manifest.get("artifact_sha256", ""))
    if en_manifest_path is not None:
        manifest = _load_json(en_manifest_path, "EN manifest")
        expected["en_artifact_sha256"] = str(manifest.get("artifact_sha256", ""))
    if lock_path is not None:
        lock = _load_json(lock_path, "embedding lock")
        expected["embedding_model_id"] = str(lock.get("model_id", ""))
        expected["embedding_revision"] = str(lock.get("revision", ""))
    return expected


def close_hybrid_index(index: HybridIndex) -> None:
    """Release the RAM-resident lexical connection (persistence files stay)."""
    conn = index.lexical_conn
    index.lexical_conn = None
    index.ram_resident = False
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def open_hybrid_index(
    directory: str | Path,
    *,
    ru_manifest_path: str | Path | None = None,
    en_manifest_path: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> HybridIndex:
    """Open a built index into RAM, rejecting stale corpus/model bindings.

    Startup loads the persisted FTS5 into ``:memory:`` via the backup
    mechanism, dense vectors into one long-lived FAISS ``IndexFlatIP``,
    and canonical child text/metadata into process memory. Disk remains
    startup/persistence storage only.
    """
    directory_path = Path(directory)
    payload = _load_json(directory_path / INDEX_NAME, "hybrid index")
    metadata_raw = payload.get("metadata")
    chunks_raw = payload.get("chunks")
    if not isinstance(metadata_raw, dict) or not isinstance(chunks_raw, list):
        raise HybridIndexError("hybrid index payload is malformed")
    metadata: dict[str, Any] = dict(metadata_raw)
    if metadata.get("format") == LEGACY_INDEX_FORMAT:
        raise StaleIndexError(
            "hybrid index format aa-hybrid-index/1 is stale; rebuild via the "
            "trusted canonical-artifact bootstrap"
        )
    if metadata.get("format") != INDEX_FORMAT:
        raise HybridIndexError(f"unsupported hybrid index format: {metadata.get('format')!r}")
    if metadata.get("builder_version") != INDEX_BUILDER_VERSION:
        raise StaleIndexError("hybrid index builder version is stale")
    for required in (
        "sentence_segmenter",
        "chunker",
        "chunker_version",
        "tokenizer",
        "chunk_policy",
        "chunk_max_tokens",
    ):
        if metadata.get(required) in (None, ""):
            raise StaleIndexError(f"hybrid index is stale: missing {required}")

    expected = _expected_from_live(
        ru_manifest_path=Path(ru_manifest_path) if ru_manifest_path is not None else None,
        en_manifest_path=Path(en_manifest_path) if en_manifest_path is not None else None,
        lock_path=Path(lock_path) if lock_path is not None else None,
    )
    for key, live_value in expected.items():
        if live_value and metadata.get(key) != live_value:
            raise StaleIndexError(f"hybrid index is stale: {key} changed")

    chunks: dict[str, ChunkRecord] = {}
    for node in chunks_raw:
        if not isinstance(node, dict):
            raise HybridIndexError("hybrid index holds a malformed chunk")
        text = str(node.get("text", ""))
        if _sha256_text(text) != str(node.get("text_sha256", "")):
            raise HybridIndexError(f"index chunk checksum mismatch: {node.get('chunk_id')!r}")
        physical = str(node.get("chunk_id", ""))
        tokens_raw = node.get("tokens")
        record = ChunkRecord(
            chunk_id=physical,
            logical_chunk_id=str(node.get("logical_chunk_id", logical_chunk_id(physical))),
            section=str(node.get("section", "")),
            book=str(node.get("book", "aa-big-book")),
            parent=str(node.get("parent", "")),
            prev=str(node["prev"]) if node.get("prev") is not None else None,
            next=str(node["next"]) if node.get("next") is not None else None,
            source_id=str(node.get("source_id", "")),
            source_file=str(node.get("source_file", "")),
            source_sha256=str(node.get("source_sha256", "")),
            char_start=int(str(node.get("char_start", 0))),
            char_end=int(str(node.get("char_end", 0))),
            text_sha256=str(node.get("text_sha256", "")),
            text=text,
            corpus_version=str(node.get("corpus_version", "")),
            tokens=int(str(tokens_raw)) if tokens_raw is not None else None,
        )
        chunks[physical] = record
    if len(chunks) != int(metadata.get("chunk_count", len(chunks))):
        raise HybridIndexError("hybrid index chunk count does not match metadata")

    dense_payload = _load_json(directory_path / DENSE_NAME, "dense vectors")
    dense_ids = dense_payload.get("ids")
    dense_vectors = dense_payload.get("vectors")
    if not isinstance(dense_ids, list) or not isinstance(dense_vectors, list):
        raise HybridIndexError("dense vectors payload is malformed")
    if [str(item) for item in dense_ids] != list(chunks):
        raise HybridIndexError("dense vector ids do not match indexed chunks")
    vectors: list[list[float]] = [[float(value) for value in row] for row in dense_vectors]
    dense_backend = str(
        dense_payload.get("backend", metadata.get("embedding_backend", HASHING_BACKEND_NAME))
    )
    if dense_backend != str(metadata.get("embedding_backend", dense_backend)):
        raise HybridIndexError("dense backend does not match index metadata")
    try:
        dense_index = ExactIPIndex.build(
            [str(item) for item in dense_ids], vectors, backend=dense_backend
        )
    except DenseError as exc:
        raise HybridIndexError(str(exc)) from exc

    lexical_path = directory_path / LEXICAL_DB_NAME
    try:
        lexical_conn = load_lexical_into_memory(lexical_path)
    except ValueError as exc:
        raise HybridIndexError(str(exc)) from exc
    try:
        count = lexical_conn.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()
    except sqlite3.Error as exc:
        lexical_conn.close()
        raise HybridIndexError(f"lexical memory index is unreadable: {exc}") from exc
    if count is None or int(count[0]) != len(chunks):
        lexical_conn.close()
        raise HybridIndexError("lexical database chunk count does not match index")

    en_control_raw = metadata.get("en_control", {})
    en_control: dict[str, dict[str, Any]] = (
        {str(key): dict(value) for key, value in en_control_raw.items()}
        if isinstance(en_control_raw, dict)
        else {}
    )
    return HybridIndex(
        directory=directory_path,
        metadata=metadata,
        chunks=chunks,
        dense=dense_index,
        lexical_conn=lexical_conn,
        ram_resident=True,
        en_control=en_control,
    )


def _preview(text: str, *, limit: int = PREVIEW_CHARS) -> str:
    snippet = " ".join(text.split())[:limit]
    return snippet + " … [preview — navigation only, never evidence]"


def _embed_query(index: HybridIndex, query: str) -> list[float]:
    backend = str(index.metadata.get("embedding_backend", HASHING_BACKEND_NAME))
    dim = int(index.metadata.get("embedding_dim", HASHING_DIM))
    if backend == HASHING_BACKEND_NAME:
        return hashing_embed(query, dim=dim)
    if backend == E5_BACKEND_NAME:
        try:
            vectors = e5_embed(["query: " + query])
        except DenseError as exc:
            raise HybridIndexError(str(exc)) from exc
        return l2_normalize(vectors[0])
    raise HybridIndexError(f"unsupported index embedding backend: {backend!r}")


def search_aspect(
    index: HybridIndex,
    queries: list[str],
    *,
    lexical_top_k: int = LEXICAL_TOP_K,
    dense_top_k: int = DENSE_TOP_K,
    rrf_k: int = RRF_K,
    max_n: int = MAX_CANDIDATES_PER_ASPECT,
    max_per_section: int = MAX_PER_SECTION,
) -> list[RetrievalHit]:
    """Search one planner aspect (original + rewrites) and fuse to <= max_n hits.

    The opened RAM-resident index is reused as-is; corpus embedding is never
    repeated and no filesystem read occurs on this hot path.
    """
    if not queries:
        raise HybridIndexError("aspect queries must be non-empty")
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise HybridIndexError("aspect queries must be non-empty strings")
    if index.lexical_conn is None or not index.ram_resident:
        raise HybridIndexError("index is not RAM-resident; open it via open_hybrid_index")
    ranked_lists: list[list[tuple[str, float]]] = []
    for query in queries:
        ranked_lists.append(lexical_search_conn(index.lexical_conn, query, top_k=lexical_top_k))
        ranked_lists.append(
            index.dense.search(
                _embed_query(index, query), top_k=min(dense_top_k, len(index.chunks))
            )
        )
    fused = rrf_fuse(ranked_lists, k=rrf_k)
    if not fused:
        return []
    spans = {
        chunk_id: (record.section, record.char_start, record.char_end)
        for chunk_id, record in index.chunks.items()
    }
    sections = {chunk_id: record.section for chunk_id, record in index.chunks.items()}
    deduped = deduplicate_overlaps(list(fused.values()), spans=spans)
    diverse = enforce_diversity(
        deduped, sections=sections, max_n=max_n, max_per_section=max_per_section
    )
    hits: list[RetrievalHit] = []
    ru_version = str(index.metadata.get("ru_artifact_sha256", ""))
    embedding_model = (
        f"{index.metadata.get('embedding_model_id')}"
        f"@{str(index.metadata.get('embedding_revision', ''))[:12]}"
    )
    builder = int(index.metadata.get("builder_version", INDEX_BUILDER_VERSION))
    for rank, candidate in enumerate(
        sorted(diverse, key=lambda item: item.fused_score, reverse=True), start=1
    ):
        record = index.chunks[candidate.chunk_id]
        if _sha256_text(record.text) != record.text_sha256:
            raise HybridIndexError(f"RAM chunk checksum mismatch: {record.chunk_id!r}")
        control = index.en_control.get(record.section, {})
        hits.append(
            RetrievalHit(
                logical_chunk_id=record.logical_chunk_id,
                chunk_id=record.chunk_id,
                section=record.section,
                source_id=record.source_id,
                source_file=record.source_file,
                char_start=record.char_start,
                char_end=record.char_end,
                text_sha256=record.text_sha256,
                text=record.text,
                lexical_rank=candidate.lexical_rank,
                dense_rank=candidate.dense_rank,
                lexical_score=candidate.lexical_score,
                dense_score=candidate.dense_score,
                fused_rank=rank,
                fused_score=candidate.fused_score,
                preview=_preview(record.text),
                parent=record.parent,
                prev=record.prev,
                next=record.next,
                index_version=builder,
                ru_corpus_version=ru_version,
                embedding_model=embedding_model,
                en_control_section=record.section,
                en_control_title=str(control.get("title", "")),
            )
        )
    return hits


def search_plan(index: HybridIndex, plan: QueryPlan) -> dict[str, list[RetrievalHit]]:
    """Search every planner aspect; return hits keyed by ``aspect_id``."""
    results: dict[str, list[RetrievalHit]] = {}
    for aspect in plan.aspects:
        queries = aspect_search_queries(aspect, original_query=plan.original_query)
        results[aspect.aspect_id] = search_aspect(index, queries)
    return results
