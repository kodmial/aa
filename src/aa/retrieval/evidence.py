"""Multi-query hybrid retrieval with Evidence Packs (issues #116, #295).

Target retrieval pipeline for the new conversation graph, built on the
planner state (``QueryPlan.queries``: 0 or 1..16 context-resolved
Russian queries; #295 flexible meaning-driven count) and the #115
RAM-resident canonical index:

```text
QueryPlan(1-16) -> BM25 + E5/FAISS per query -> global RRF(k=60)
  -> per-query retention -> overlap dedup/diversity
  -> model-driven semantic selection over broad candidates
  -> small-to-big expansion -> compact Evidence Pack (16k token budget)
```

Contract notes:

- ``queries == []`` skips book retrieval and yields an empty pack for a
  purely conversational/glue turn (no index access at all);
- every non-empty planner query runs both branches at top 40
  (SQLite FTS5/BM25 against the in-memory lexical DB plus exact E5 +
  FAISS ``IndexFlatIP`` over the same canonical child chunks);
- raw BM25 and dense scores are never compared directly; fusion uses
  only the standard RRF primitive;
- selection after fusion is RRF-ordered with an optional bounded
  model-driven semantic promotion over genuinely broad candidates
  (see :mod:`aa.conversation.semantic_selection`); canonical text and
  provenance are never rewritten; deep fused ranks (>16) stay reachable
  and are never irreversibly pruned before semantic inspection;
- small-to-big expansion stays within one canonical source/section
  unless an explicit neighbor link crosses a valid boundary, merges
  overlapping/adjacent windows, and keeps exact text plus provenance;
- the Evidence Pack carries only generation-relevant evidence; ranking
  metadata (RRF/BM25/dense scores, embeddings, planner reasoning,
  search previews) is internal-only and never enters ``<book_evidence>``.

This path has zero dependency on legacy semantic logic: no handwritten
query-expansion dictionaries and no legacy planner structures. It is
RAM-only (BM25 + E5/FAISS plus RRF) with zero network/download
dependency on the hot path and no unconditional second-stage
cross-encoder/BGE reranker.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape as _xml_escape_text
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS, estimate_text_tokens
from aa.retrieval.dense import (
    E5_BACKEND_NAME,
    HASHING_BACKEND_NAME,
    HASHING_DIM,
    DenseError,
    e5_embed,
    hashing_embed,
    l2_normalize,
)
from aa.retrieval.fusion import (
    RRF_K,
    FusedCandidate,
    deduplicate_overlaps,
    rrf_fuse,
)
from aa.retrieval.index import HybridIndex
from aa.retrieval.lexical import lexical_search_conn

logger = logging.getLogger("aa.retrieval.evidence")

MIN_PLANNER_QUERIES = 1
MAX_PLANNER_QUERIES = 16
BRANCH_TOP_K = 40
# Broad-candidate recall (issue #295): the fused pool stays wide enough
# that decisive passages at fused rank >16 remain inspectable by the
# model-driven semantic selection layer before any top-N budgeting.
# All caps stay bounded by real RAM/latency/token budgets; nothing here
# claims infinite context fits one model call.
POOL_CAP = 128
TOP_CHILD_CAP = 32
MAX_PER_SECTION = 6
NEIGHBOR_WINDOW = 2
# Interactive Telegram budget for one warm retrieval turn. The RRF-only
# RAM path is expected to serve well within this budget.
INTERACTIVE_LATENCY_BUDGET_MS = 5000.0

# Cross-turn E5 query-vector cache (latency optimization for the 5s
# interactive budget). Planner queries repeat across turns (terse
# follow-ups, overlapping paraphrases); reusing the exact L2-normalized
# vector skips a padded transformer forward per repeated query. Keys bind
# the index backend plus dimension plus the exact query string. Bounded
# FIFO (1024 entries) with its own lock; vectors are copied on store and
# on return so callers cannot mutate cached state.
_QUERY_VECTOR_CACHE_MAX = 1024
_QUERY_VECTOR_CACHE: dict[tuple[str, int, str], list[float]] = {}
_QUERY_VECTOR_CACHE_LOCK = threading.Lock()


def clear_query_vector_cache() -> None:
    """Drop cached cross-turn query vectors (tests/tooling only)."""
    with _QUERY_VECTOR_CACHE_LOCK:
        _QUERY_VECTOR_CACHE.clear()


def query_vector_cache_info() -> dict[str, int]:
    """Return the current query-vector cache size (observability)."""
    with _QUERY_VECTOR_CACHE_LOCK:
        return {"size": len(_QUERY_VECTOR_CACHE), "max": _QUERY_VECTOR_CACHE_MAX}


class EvidenceError(ValueError):
    """Raised when the evidence pipeline cannot serve a request (fails closed)."""


class EvidenceIntegrityError(EvidenceError):
    """Same stable passage id maps to different bytes/content (fails closed)."""


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable retrieval/evidence parameters (recorded in metadata/evals)."""

    branch_top_k: int = BRANCH_TOP_K
    rrf_k: int = RRF_K
    pool_cap: int = POOL_CAP
    top_child_cap: int = TOP_CHILD_CAP
    max_per_section: int = MAX_PER_SECTION
    neighbor_window: int = NEIGHBOR_WINDOW
    budget_tokens: int = RETRIEVED_PASSAGES_BUDGET_TOKENS


@dataclass(frozen=True)
class EvidencePassageData:
    """One coherent expanded passage of exact canonical Russian text."""

    passage_id: str
    exact_text: str
    source_id: str
    section_id: str
    child_chunk_ids: tuple[str, ...]
    char_start: int
    char_end: int
    text_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class EvidencePack:
    """Framework-independent generation evidence (exact text + provenance)."""

    passages: tuple[EvidencePassageData, ...]
    total_tokens: int
    corpus_version: str
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _short_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stable_passage_id(
    *,
    source_sha256: str,
    section_id: str,
    char_start: int,
    char_end: int,
    text_sha256: str,
) -> str:
    """Return the stable identity for one exact passage (never positional).

    Derived only from source provenance plus the exact range/content
    (``source_sha256``, ``section_id``, ``char_start``, ``char_end``,
    ``text_sha256``), so two searches over the same canonical source
    reproduce the same id while two different ranges never collide.
    Request-local ``p1..pN`` aliases may exist in prompts only and are
    never persistent identities.
    """
    section = str(section_id or "unknown-section")
    src = str(source_sha256 or "")
    src_short = src[:12] if src else "nosrc"
    txt = str(text_sha256 or "")
    txt_short = txt[:16] if txt else "notext"
    try:
        start = int(char_start)
    except (TypeError, ValueError):
        start = 0
    try:
        end = int(char_end)
    except (TypeError, ValueError):
        end = 0
    return f"{section}#{src_short}:{start}-{end}:{txt_short}"


def check_passage_id_consistency(
    passage_id: str,
    existing: dict[str, object],
    incoming: dict[str, object],
) -> None:
    """Fail closed when one stable id maps to different content.

    Compares exact text, ``text_sha256``, range and source provenance;
    any mismatch raises :class:`EvidenceIntegrityError` instead of
    silently keeping the older passage.
    """
    for key in ("text", "text_sha256", "char_start", "char_end", "source_sha256"):
        old = existing.get(key) if isinstance(existing, dict) else None
        new = incoming.get(key) if isinstance(incoming, dict) else None
        if old is None or new is None:
            continue
        if old != new:
            raise EvidenceIntegrityError(
                f"stable passage id collision with different content: {passage_id!r} ({key})"
            )


def enrich_pack_provenance(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Fill missing provenance on a deduplicated pack entry (in place).

    Only backfills empty provenance slots from the duplicate; never
    overwrites present values and never invents content.
    """
    for provenance_key in ("source_sha256", "corpus_version"):
        try:
            if not existing.get(provenance_key) and incoming.get(provenance_key):
                existing[provenance_key] = incoming.get(provenance_key)
        except AttributeError:
            continue


def validate_planner_queries(queries: object) -> list[str]:
    """Validate the planner query list (fails closed).

    Issue #295: 1..16 distinct useful queries; the planner itself is
    responsible for emitting a meaning-driven count without padding.
    """
    if not isinstance(queries, list):
        raise EvidenceError("planner queries must be a list of strings")
    if not queries:
        return []
    cleaned: list[str] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise EvidenceError("planner queries must be non-empty strings")
        cleaned.append(query.strip())
    if not MIN_PLANNER_QUERIES <= len(cleaned) <= MAX_PLANNER_QUERIES:
        raise EvidenceError(
            "non-empty planner query lists require "
            f"{MIN_PLANNER_QUERIES}-{MAX_PLANNER_QUERIES} queries, got {len(cleaned)}"
        )
    return cleaned


def validate_recovery_queries(queries: object) -> list[str]:
    """Validate one bounded empty-pack recovery query list (fails closed).

    Recovery uses the live turn plus conversation context directly (never
    canned generic queries) and carries 1..6 queries. This validator
    accepts 1..16 non-empty strings; the flexible planner contract
    (1..16 useful queries) stays unchanged.
    """
    if not isinstance(queries, list):
        raise EvidenceError("recovery queries must be a list of strings")
    if not queries:
        return []
    cleaned: list[str] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise EvidenceError("recovery queries must be non-empty strings")
        cleaned.append(query.strip())
    if not 1 <= len(cleaned) <= MAX_PLANNER_QUERIES:
        raise EvidenceError(
            f"recovery query lists require 1-{MAX_PLANNER_QUERIES} queries, got {len(cleaned)}"
        )
    return cleaned


def empty_evidence_pack(*, corpus_version: str = "") -> EvidencePack:
    """Return the empty pack for a purely conversational turn (no retrieval)."""
    return EvidencePack(
        passages=(),
        total_tokens=0,
        corpus_version=corpus_version,
        retrieval_metadata={"planner_query_count": 0, "retrieval_skipped": True},
    )


def _embed_query_vector(index: HybridIndex, query: str) -> list[float]:
    """Embed one query with the index backend (mirrors the RAM substrate)."""
    backend = str(index.metadata.get("embedding_backend", HASHING_BACKEND_NAME))
    dim = int(index.metadata.get("embedding_dim", HASHING_DIM))
    if backend == HASHING_BACKEND_NAME:
        return hashing_embed(query, dim=dim)
    if backend == E5_BACKEND_NAME:
        try:
            vectors = e5_embed(["query: " + query])
        except DenseError as exc:
            raise EvidenceError(str(exc)) from exc
        return l2_normalize(vectors[0])
    raise EvidenceError(f"unsupported index embedding backend: {backend!r}")


def _require_ram_index(index: HybridIndex) -> None:
    if index.lexical_conn is None or not index.ram_resident:
        raise EvidenceError("index is not RAM-resident; open it via open_hybrid_index")


def _embed_query_vectors(index: HybridIndex, queries: list[str]) -> list[list[float]]:
    """Embed all planner queries in one batched call (hot-path optimization).

    The hashing backend is cheap and stays per-query; the E5 backend goes
    through a single :func:`e5_embed` forward with padded batching instead
    of one lock/model call per query. Repeated E5 queries reuse the exact
    cached L2-normalized vector across turns (bounded FIFO) so a repeated
    follow-up skips its transformer forward. Returned vectors are
    L2-normalized and order-preserving; ranking is unchanged.
    """
    backend = str(index.metadata.get("embedding_backend", HASHING_BACKEND_NAME))
    dim = int(index.metadata.get("embedding_dim", HASHING_DIM))
    if backend == HASHING_BACKEND_NAME:
        return [hashing_embed(query, dim=dim) for query in queries]
    if backend == E5_BACKEND_NAME:
        keys = [(backend, dim, query) for query in queries]
        with _QUERY_VECTOR_CACHE_LOCK:
            cached = [_QUERY_VECTOR_CACHE.get(key) for key in keys]
        if all(vector is not None for vector in cached):
            return [list(vector) for vector in cached if vector is not None]
        miss_queries: list[str] = []
        miss_keys: list[tuple[str, int, str]] = []
        seen_miss: set[tuple[str, int, str]] = set()
        for key, query, vector in zip(keys, queries, cached, strict=True):
            if vector is None and key not in seen_miss:
                seen_miss.add(key)
                miss_keys.append(key)
                miss_queries.append("query: " + query)
        try:
            vectors = e5_embed(miss_queries)
        except DenseError as exc:
            raise EvidenceError(str(exc)) from exc
        normalized = [l2_normalize(vector) for vector in vectors]
        with _QUERY_VECTOR_CACHE_LOCK:
            for key, vector in zip(miss_keys, normalized, strict=True):
                if key not in _QUERY_VECTOR_CACHE:
                    while len(_QUERY_VECTOR_CACHE) >= _QUERY_VECTOR_CACHE_MAX:
                        _QUERY_VECTOR_CACHE.pop(next(iter(_QUERY_VECTOR_CACHE)))
                    _QUERY_VECTOR_CACHE[key] = list(vector)
            out: list[list[float]] = []
            for key in keys:
                stored = _QUERY_VECTOR_CACHE.get(key)
                if stored is None:
                    raise EvidenceError("query vector cache missed after embed")
                out.append(list(stored))
            return out
    raise EvidenceError(f"unsupported index embedding backend: {backend!r}")


def run_branch_searches(
    index: HybridIndex,
    queries: list[str],
    *,
    branch_top_k: int = BRANCH_TOP_K,
) -> tuple[list[list[tuple[str, float]]], list[list[str]]]:
    """Run lexical + dense branches for every planner query (hot path, no disk)."""
    _require_ram_index(index)
    if branch_top_k <= 0:
        raise EvidenceError("branch_top_k must be > 0")
    assert index.lexical_conn is not None
    query_vectors = _embed_query_vectors(index, queries)
    ranked_lists: list[list[tuple[str, float]]] = []
    per_query_ids: list[list[str]] = []
    for query, query_vector in zip(queries, query_vectors, strict=True):
        lexical_ranked = lexical_search_conn(index.lexical_conn, query, top_k=branch_top_k)
        dense_ranked = index.dense.search(
            query_vector,
            top_k=min(branch_top_k, len(index.chunks)),
        )
        ranked_lists.append(lexical_ranked)
        ranked_lists.append(dense_ranked)
        contributed: list[str] = []
        seen: set[str] = set()
        for chunk_id, _ in (*lexical_ranked, *dense_ranked):
            if chunk_id not in seen:
                seen.add(chunk_id)
                contributed.append(chunk_id)
        per_query_ids.append(contributed)
    return ranked_lists, per_query_ids


def fuse_query_pool(
    ranked_lists: list[list[tuple[str, float]]],
    per_query_ids: list[list[str]],
    *,
    rrf_k: int = RRF_K,
    pool_cap: int = POOL_CAP,
) -> tuple[dict[str, FusedCandidate], list[str]]:
    """Fuse branch rankings with RRF, retaining each query's best candidate.

    Per-query retention keeps multi-query recall: every query's best
    unique candidate is retained first (sorted by fused score and
    truncated only when distinct bests exceed ``pool_cap``), then the
    rest of the pool fills by global RRF rank. With 1..16 planner
    queries and the broad pool this guarantees the ticket
    rule (at least the best unique candidate per query) while leaving
    ample slots for global RRF depth, including fused ranks >16.
    """
    if rrf_k <= 0:
        raise EvidenceError("rrf_k must be > 0")
    if pool_cap <= 0:
        raise EvidenceError("pool_cap must be > 0")
    fused = rrf_fuse(ranked_lists, k=rrf_k)
    ordered = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
    pool: list[FusedCandidate] = []
    seen: set[str] = set()
    # Preserve multi-query recall: keep each query's best unique candidate.
    # Every distinct per-query best is retained first (up to pool_cap), so
    # no required best is discarded before the global RRF fill; only when
    # distinct bests exceed the pool is the tail truncated by fused score.
    retained: list[FusedCandidate] = []
    for contributed in per_query_ids:
        best: FusedCandidate | None = None
        for chunk_id in contributed:
            candidate = fused.get(chunk_id)
            if candidate is None or chunk_id in seen:
                continue
            if best is None or candidate.fused_score > best.fused_score:
                best = candidate
        if best is not None:
            retained.append(best)
            seen.add(best.chunk_id)
    retained.sort(key=lambda item: item.fused_score, reverse=True)
    pool.extend(retained[:pool_cap])
    for candidate in ordered:
        if len(pool) >= pool_cap:
            break
        if candidate.chunk_id not in seen:
            pool.append(candidate)
            seen.add(candidate.chunk_id)
    return fused, [item.chunk_id for item in pool[:pool_cap]]


def candidate_query_provenance(
    per_query_ids: list[list[str]],
) -> dict[str, list[str]]:
    """Map each candidate to the planner query ids that retrieved it (#311).

    Query ids are stable ``q1``..``qN`` in plan order. A candidate hit by
    several queries retains all associations (many-to-many); nothing is
    fabricated and no lexical inference is applied.
    """
    provenance: dict[str, list[str]] = {}
    for pos, contributed in enumerate(list(per_query_ids or [])):
        query_id = f"q{pos + 1}"
        seen_in_query: set[str] = set()
        for chunk_id in list(contributed or []):
            cid = str(chunk_id or "").strip()
            if not cid or cid in seen_in_query:
                continue
            seen_in_query.add(cid)
            bucket = provenance.setdefault(cid, [])
            if query_id not in bucket:
                bucket.append(query_id)
    for bucket in provenance.values():
        bucket.sort(key=lambda qid: int(qid[1:]) if qid[1:].isdigit() else 0)
    return provenance


def candidate_need_provenance(
    candidate_query_map: dict[str, list[str]],
    query_need_map: list[dict[str, Any]] | list[Any],
) -> dict[str, list[str]]:
    """Map each candidate to semantic need ids via its query ids (#311).

    Many-to-many links are preserved: a cross-query shared hit retains
    all need associations. Queries mapped to no need (unknown/unmapped)
    contribute no need; candidates with no mapped need stay unmapped
    (``[]``) and must trigger conservative discovery, never fabricated
    quota coverage.
    """
    query_to_needs: dict[str, list[str]] = {}
    for entry in list(query_need_map or []):
        try:
            if isinstance(entry, dict):
                qid = str(entry.get("query_id", "") or "").strip()
                raw_needs = entry.get("need_ids", []) or []
            else:
                qid = str(getattr(entry, "query_id", "") or "").strip()
                raw_needs = list(getattr(entry, "need_ids", []) or [])
            if not qid:
                continue
            kept: list[str] = []
            seen: set[str] = set()
            items = list(raw_needs) if isinstance(raw_needs, list) else []
            for raw in items:
                nid = str(raw or "").strip()
                if nid and nid not in seen:
                    seen.add(nid)
                    kept.append(nid)
            query_to_needs[qid] = kept
        except Exception:
            continue
    out: dict[str, list[str]] = {}
    for chunk_id, query_ids in (candidate_query_map or {}).items():
        cid = str(chunk_id or "").strip()
        if not cid:
            continue
        merged: list[str] = []
        seen_needs: set[str] = set()
        for qid in list(query_ids or []):
            for nid in query_to_needs.get(str(qid), []):
                if nid not in seen_needs:
                    seen_needs.add(nid)
                    merged.append(nid)
        merged.sort()
        out[cid] = merged
    return out


def dedup_and_diversify(
    index: HybridIndex,
    pool_ids: list[str],
    fused: dict[str, FusedCandidate],
    *,
    pool_cap: int = POOL_CAP,
    max_per_section: int = MAX_PER_SECTION,
) -> list[FusedCandidate]:
    """Collapse overlaps while preserving bounded cross-section recall.

    The normal RRF pool remains the primary candidate set. In addition,
    the best RRF-scored candidate from every section seen anywhere in
    the branch union receives a section ticket before the bounded global
    fill. This prevents repeated hits from popular sections from erasing
    a lower-frequency but still retrieved section before Evidence Pack
    construction.
    """
    if pool_cap <= 0 or max_per_section <= 0:
        raise EvidenceError("pool_cap and max_per_section must be > 0")

    sections = {chunk_id: record.section for chunk_id, record in index.chunks.items()}
    by_id = {chunk_id: fused[chunk_id] for chunk_id in pool_ids if chunk_id in fused}

    best_by_section: dict[str, FusedCandidate] = {}
    for candidate in fused.values():
        section = sections.get(candidate.chunk_id)
        if section is None:
            continue
        current = best_by_section.get(section)
        if current is None or candidate.fused_score > current.fused_score:
            best_by_section[section] = candidate
    for candidate in best_by_section.values():
        by_id.setdefault(candidate.chunk_id, candidate)

    spans = {
        chunk_id: (record.section, record.char_start, record.char_end)
        for chunk_id, record in index.chunks.items()
    }
    deduped = deduplicate_overlaps(list(by_id.values()), spans=spans)
    ordered = sorted(deduped, key=lambda item: item.fused_score, reverse=True)

    picked: list[FusedCandidate] = []
    picked_ids: set[str] = set()
    seen_sections: set[str] = set()
    counts: dict[str, int] = {}

    for candidate in ordered:
        section = sections.get(candidate.chunk_id, "?")
        if section in seen_sections:
            continue
        picked.append(candidate)
        picked_ids.add(candidate.chunk_id)
        seen_sections.add(section)
        counts[section] = 1
        if len(picked) >= pool_cap:
            return picked

    for candidate in ordered:
        if candidate.chunk_id in picked_ids:
            continue
        section = sections.get(candidate.chunk_id, "?")
        if counts.get(section, 0) >= max_per_section:
            continue
        picked.append(candidate)
        picked_ids.add(candidate.chunk_id)
        counts[section] = counts.get(section, 0) + 1
        if len(picked) >= pool_cap:
            break
    return picked


def select_top_candidates(
    candidates: list[FusedCandidate],
    *,
    top_cap: int = TOP_CHILD_CAP,
    sections: dict[str, str] | None = None,
) -> list[FusedCandidate]:
    """Select bounded RRF winners with an optional section-coverage floor."""
    if top_cap <= 0:
        raise EvidenceError("top_cap must be > 0")
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
    if sections is None:
        return ranked[:top_cap]

    picked: list[FusedCandidate] = []
    picked_ids: set[str] = set()
    seen_sections: set[str] = set()
    for candidate in ranked:
        section = sections.get(candidate.chunk_id, "?")
        if section in seen_sections:
            continue
        picked.append(candidate)
        picked_ids.add(candidate.chunk_id)
        seen_sections.add(section)
        if len(picked) >= top_cap:
            return picked

    for candidate in ranked:
        if candidate.chunk_id in picked_ids:
            continue
        picked.append(candidate)
        if len(picked) >= top_cap:
            break
    return picked


def _ordered_section_chunks(index: HybridIndex, section_id: str) -> list[str]:
    ordered = sorted(
        (record for record in index.chunks.values() if record.section == section_id),
        key=lambda item: (item.char_start, item.char_end, item.chunk_id),
    )
    return [record.chunk_id for record in ordered]


def expand_small_to_big(
    index: HybridIndex,
    winners: list[FusedCandidate],
    *,
    neighbor_window: int = NEIGHBOR_WINDOW,
) -> list[EvidencePassageData]:
    """Expand RRF-selected child hits to coherent parent/neighbor passages.

    Each winning child expands to its full canonical parent paragraph
    (all sibling child chunks) plus a bounded adjacent-child window on
    each side. Expansion never leaves the child source/section unless an
    explicit ``prev``/``next`` link crosses a valid boundary into the
    same source. Overlapping/adjacent expanded windows merge. Text and
    provenance stay exact; whole chapters are never loaded blindly.
    """
    if neighbor_window < 0:
        raise EvidenceError("neighbor_window must be >= 0")
    if not winners:
        return []
    rank_of = {candidate.chunk_id: pos for pos, candidate in enumerate(winners)}
    section_order: dict[str, list[str]] = {}
    section_position: dict[str, dict[str, int]] = {}
    for chunk_id in rank_of:
        record = index.chunks.get(chunk_id)
        if record is None:
            raise EvidenceError(f"selected winner is not indexed: {chunk_id!r}")
        if record.section not in section_order:
            ordered_ids = _ordered_section_chunks(index, record.section)
            section_order[record.section] = ordered_ids
            section_position[record.section] = {item: pos for pos, item in enumerate(ordered_ids)}
    raw_windows: list[tuple[str, set[str]]] = []
    for chunk_id in rank_of:
        record = index.chunks[chunk_id]
        siblings = {
            item.chunk_id
            for item in index.chunks.values()
            if item.parent == record.parent and item.section == record.section
        }
        siblings.add(chunk_id)
        ordered_ids = section_order[record.section]
        positions = section_position[record.section]
        sibling_positions = sorted(positions[item] for item in siblings if item in positions)
        window = set(siblings)
        if sibling_positions:
            before = max(0, sibling_positions[0] - neighbor_window)
            after = min(len(ordered_ids), sibling_positions[-1] + neighbor_window + 1)
            for pos in range(before, sibling_positions[0]):
                cand_id = ordered_ids[pos]
                cand = index.chunks[cand_id]
                if cand.section == record.section and cand.source_id == record.source_id:
                    window.add(cand_id)
            for pos in range(sibling_positions[-1] + 1, after):
                cand_id = ordered_ids[pos]
                cand = index.chunks[cand_id]
                if cand.section == record.section and cand.source_id == record.source_id:
                    window.add(cand_id)
        for link in (record.prev, record.next):
            if link is None:
                continue
            neighbor = index.chunks.get(link)
            if (
                neighbor is not None
                and neighbor.section == record.section
                and neighbor.source_id == record.source_id
            ):
                window.add(link)
        raw_windows.append((record.section, window))

    # Merge overlapping/adjacent windows within each section.
    def _windows_touch(section_id: str, first: set[str], second: set[str]) -> bool:
        if first & second:
            return True
        positions = section_position.get(section_id)
        if not positions:
            return False
        try:
            first_pos = sorted(positions[cid] for cid in first)
            second_pos = sorted(positions[cid] for cid in second)
        except KeyError:
            return False
        if not first_pos or not second_pos:
            return False
        # Contiguous but disjoint windows coalesce: any positions adjacent
        # or intervals touching/overlapping in section order.
        if first_pos[-1] + 1 >= second_pos[0] and second_pos[-1] + 1 >= first_pos[0]:
            # Guard the gap case (e.g. {0,5} vs {2,3} overlap the span but
            # share no adjacency): require overlap or a +/-1 edge.
            pos_set = set(second_pos)
            return any(
                pos in pos_set or (pos + 1) in pos_set or (pos - 1) in pos_set for pos in first_pos
            )
        return False

    merged: list[tuple[str, set[str]]] = []
    for section_id, window in raw_windows:
        placed = False
        for pos, (kept_section, kept) in enumerate(merged):
            if kept_section != section_id or not _windows_touch(section_id, kept, window):
                continue
            merged[pos] = (kept_section, kept | window)
            placed = True
            break
        if not placed:
            merged.append((section_id, set(window)))
    changed = True
    while changed:
        changed = False
        collapsed: list[tuple[str, set[str]]] = []
        for section_id, window in merged:
            absorbed = False
            for pos, (kept_section, kept) in enumerate(collapsed):
                if kept_section == section_id and _windows_touch(section_id, kept, window):
                    collapsed[pos] = (kept_section, kept | window)
                    absorbed = True
                    changed = True
                    break
            if not absorbed:
                collapsed.append((section_id, window))
        merged = collapsed
    passages: list[EvidencePassageData] = []
    passage_ranks: list[int] = []
    for section_id, window in merged:
        members = sorted(
            window,
            key=lambda item: (
                index.chunks[item].char_start,
                index.chunks[item].char_end,
                item,
            ),
        )
        texts = [index.chunks[item].text for item in members]
        for item, text in zip(members, texts, strict=True):
            record = index.chunks[item]
            if _sha256_text(text) != record.text_sha256 or text != record.text:
                raise EvidenceError(f"expansion checksum mismatch: {item!r}")
        # Partition into exactly contiguous runs. Chunks tile their
        # paragraph exactly (next.char_start == prev.char_end), so only
        # adjacent runs concatenate to the exact source slice with no
        # synthetic glue. Any gap or overlap becomes separate passages.
        runs: list[list[str]] = []
        for item in members:
            record = index.chunks[item]
            if runs:
                prev = index.chunks[runs[-1][-1]]
                if (
                    record.source_id == prev.source_id
                    and record.section == prev.section
                    and record.char_start == prev.char_end
                ):
                    runs[-1].append(item)
                    continue
            runs.append([item])
        for run in runs:
            run_texts = [index.chunks[item].text for item in run]
            exact_text = "".join(run_texts)
            first = index.chunks[run[0]]
            starts = [index.chunks[item].char_start for item in run]
            ends = [index.chunks[item].char_end for item in run]
            window_ranks = [rank_of[cid] for cid in window if cid in rank_of]
            run_best = min(window_ranks) if window_ranks else max(rank_of.values(), default=0) + 1
            start = min(starts)
            end = max(ends)
            text_hash = _sha256_text(exact_text)
            stable_id = stable_passage_id(
                source_sha256=first.source_sha256,
                section_id=section_id,
                char_start=start,
                char_end=end,
                text_sha256=text_hash,
            )
            passages.append(
                EvidencePassageData(
                    passage_id=stable_id,
                    exact_text=exact_text,
                    source_id=first.source_id,
                    section_id=section_id,
                    child_chunk_ids=tuple(run),
                    char_start=start,
                    char_end=end,
                    text_sha256=text_hash,
                    source_sha256=first.source_sha256,
                )
            )
            passage_ranks.append(run_best)
    # Highest-value (best fused order) passages first. Each run inherits
    # the best fused position of its merged window, so neighbor-only runs
    # (no direct winner) still order deterministically.
    passages = [
        passage
        for passage, _ in sorted(
            zip(passages, passage_ranks, strict=True),
            key=lambda pair: (pair[1], pair[0].char_start, pair[0].passage_id),
        )
    ]
    return passages


def select_passages_under_budget(
    passages: list[EvidencePassageData],
    *,
    budget_tokens: int = RETRIEVED_PASSAGES_BUDGET_TOKENS,
    index: HybridIndex | None = None,
    priority_child_ids: tuple[str, ...] = (),
) -> tuple[list[EvidencePassageData], int]:
    """Select coherent passages atomically under the source-token budget.

    Expanded passages are preferred whenever they fit. If a passage
    cannot fit, directly selected RRF winners inside that passage degrade
    immediately to exact child chunks before any lower-priority passage
    can consume the remaining budget. This preserves fused-rank priority
    without truncating canonical text.

    Direct callers that do not provide ``priority_child_ids`` keep the
    legacy fallback semantics.
    """
    if budget_tokens <= 0:
        raise EvidenceError("budget_tokens must be > 0")

    def _atom(cid: str) -> EvidencePassageData | None:
        if index is None:
            return None
        record = index.chunks.get(cid)
        if record is None:
            return None
        return EvidencePassageData(
            passage_id=stable_passage_id(
                source_sha256=record.source_sha256,
                section_id=record.section,
                char_start=record.char_start,
                char_end=record.char_end,
                text_sha256=record.text_sha256,
            ),
            exact_text=record.text,
            source_id=record.source_id,
            section_id=record.section,
            child_chunk_ids=(cid,),
            char_start=record.char_start,
            char_end=record.char_end,
            text_sha256=record.text_sha256,
            source_sha256=record.source_sha256,
        )

    selected: list[EvidencePassageData] = []
    covered: set[str] = set()
    total = 0

    if priority_child_ids:
        priority = {cid: rank for rank, cid in enumerate(priority_child_ids)}
        atoms = {cid: _atom(cid) for cid in priority_child_ids}
        atom_costs = {
            cid: estimate_text_tokens(atom.exact_text)
            for cid, atom in atoms.items()
            if atom is not None
        }
        remaining_priority = {cid for cid in priority_child_ids if cid in atom_costs}
        remaining_reserve = sum(atom_costs[cid] for cid in remaining_priority)
        reserve_all = remaining_reserve <= budget_tokens

        for passage in passages:
            passage_priority = sorted(
                (cid for cid in passage.child_chunk_ids if cid in remaining_priority),
                key=priority.__getitem__,
            )
            covered_by_passage = set(passage_priority)
            reserve_after = remaining_reserve - sum(atom_costs[cid] for cid in covered_by_passage)
            need = estimate_text_tokens(passage.exact_text)

            if total + need <= budget_tokens and (
                not reserve_all or total + need + reserve_after <= budget_tokens
            ):
                selected.append(passage)
                covered.update(passage.child_chunk_ids)
                for cid in covered_by_passage:
                    remaining_priority.discard(cid)
                remaining_reserve = reserve_after
                total += need
                continue

            for cid in passage_priority:
                atom = atoms.get(cid)
                if atom is None:
                    continue
                atom_need = atom_costs[cid]
                if atom_need <= 0 or total + atom_need > budget_tokens:
                    continue
                selected.append(atom)
                covered.add(cid)
                remaining_priority.discard(cid)
                remaining_reserve -= atom_need
                total += atom_need

        if reserve_all and remaining_priority:
            raise EvidenceError("priority winner reservation was not materialized")
        return selected, total

    for passage in passages:
        need = estimate_text_tokens(passage.exact_text)
        if total + need <= budget_tokens:
            selected.append(passage)
            covered.update(passage.child_chunk_ids)
            total += need

    if index is not None and passages:
        top = passages[0]
        top_covered = any(cid in covered for cid in top.child_chunk_ids)
        if not selected or not top_covered:
            seen: set[str] = set(covered)
            fallbacks: list[EvidencePassageData] = []
            for passage in passages:
                for cid in passage.child_chunk_ids:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    atom = _atom(cid)
                    if atom is None:
                        continue
                    need = estimate_text_tokens(atom.exact_text)
                    if need <= 0 or total + need > budget_tokens:
                        continue
                    fallbacks.append(atom)
                    total += need
                    if total >= budget_tokens:
                        break
                if total >= budget_tokens:
                    break
            if fallbacks:
                if not selected:
                    selected = fallbacks
                elif not top_covered:
                    top_ids = set(top.child_chunk_ids)
                    head = [item for item in fallbacks if item.child_chunk_ids[0] in top_ids]
                    tail = [item for item in fallbacks if item.child_chunk_ids[0] not in top_ids]
                    selected = [*head, *selected, *tail]
    return selected, total


def broad_fused_ranking(
    index: HybridIndex,
    queries: object,
    *,
    config: RetrievalConfig | None = None,
) -> tuple[dict[str, Any], list[tuple[str, float]]]:
    """Expose the genuinely broad fused ranking for semantic selection.

    Returns ``(fused_by_id, ordered)`` where ``ordered`` is
    ``[(chunk_id, fused_score)]`` best-first over the full fused pool
    (ranks >16 included). Used by the model-driven selection layer for
    discovery previews; the Evidence Pack path below stays budgeted.
    """
    active = config if config is not None else RetrievalConfig()
    cleaned = validate_planner_queries(queries)
    if not cleaned:
        return {}, []
    ranked_lists, per_query_ids = run_branch_searches(
        index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists, per_query_ids, rrf_k=active.rrf_k, pool_cap=active.pool_cap
    )
    ordered = sorted(
        ((chunk_id, fused[chunk_id].fused_score) for chunk_id in pool_ids if chunk_id in fused),
        key=lambda pair: pair[1],
        reverse=True,
    )
    # Include any remaining fused candidates beyond the pool cap tail so
    # deep ranks stay inspectable when the pool itself is the bound.
    if len(ordered) < len(fused):
        seen = {chunk_id for chunk_id, _ in ordered}
        rest = sorted(
            (
                (chunk_id, candidate.fused_score)
                for chunk_id, candidate in fused.items()
                if chunk_id not in seen
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        ordered.extend(rest)
    return fused, ordered


def retrieve_evidence(
    index: HybridIndex,
    queries: object,
    *,
    config: RetrievalConfig | None = None,
    resolved_intent: str = "",
    conversation_context: str = "",
) -> EvidencePack:
    """Run the full RRF-only pipeline for one planner query list.

    ``queries`` is the ``QueryPlan.queries`` (0 or 1..16 flexible
    meaning-driven queries per issue #295). An empty
    list performs no retrieval and returns an empty pack. The answering model receives only the
    returned exact passages; ``retrieval_metadata`` stays internal.

    Pipeline: ``QueryPlan -> BM25+E5 -> RRF -> dedup/diversity ->
    semantic promotion (optional, bounded) -> small-to-big`` over the
    RAM-resident index with zero network/download dependency and no
    unconditional second-stage reranker. Per-query retention keeps each
    query's best unique candidate first, then fills the remainder by
    global RRF rank. When ``resolved_intent`` is supplied, a bounded
    generic semantic promotion reorders winners so intent-relevant deep
    candidates (fused rank >16) surface without dropping any candidate
    before budgeting.
    """
    active = config if config is not None else RetrievalConfig()
    if active.branch_top_k <= 0 or active.rrf_k <= 0:
        raise EvidenceError("branch_top_k and rrf_k must be > 0")
    if active.pool_cap <= 0 or active.top_child_cap <= 0:
        raise EvidenceError("pool caps must be > 0")
    if active.max_per_section <= 0 or active.neighbor_window < 0:
        raise EvidenceError("diversity/neighbor parameters are invalid")
    if active.budget_tokens <= 0:
        raise EvidenceError("budget_tokens must be > 0")
    cleaned = validate_planner_queries(queries)
    corpus_version = str(index.metadata.get("ru_artifact_sha256", ""))
    if not cleaned:
        return empty_evidence_pack(corpus_version=corpus_version)
    started = time.perf_counter()
    ranked_lists, per_query_ids = run_branch_searches(
        index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists,
        per_query_ids,
        rrf_k=active.rrf_k,
        pool_cap=active.pool_cap,
    )
    diverse = dedup_and_diversify(
        index,
        pool_ids,
        fused,
        pool_cap=active.pool_cap,
        max_per_section=active.max_per_section,
    )
    sections = {chunk_id: record.section for chunk_id, record in index.chunks.items()}
    winners = select_top_candidates(
        diverse,
        top_cap=active.top_child_cap,
        sections=sections,
    )
    # Bounded semantic promotion (issue #295): when the planner resolved
    # a real intent, reorder winners by generic intent relevance over
    # short previews so a decisive deep-ranked candidate surfaces before
    # expansion/budgeting. Stable: RRF order breaks ties, nothing is
    # dropped here, and the empty-intent path behaves byte-identically.
    semantic_promoted = False
    if resolved_intent.strip():
        try:
            from aa.conversation.semantic_selection import (
                PREVIEW_CHARS as _PREVIEW_CHARS,
            )
            from aa.conversation.semantic_selection import (
                heuristic_relevance_score as _score,
            )

            _context = f"{resolved_intent.strip()} {conversation_context.strip()}".strip()
            _order = {candidate.chunk_id: pos for pos, candidate in enumerate(winners)}
            _scored = []
            for candidate in winners:
                record = index.chunks.get(candidate.chunk_id)
                preview = str(getattr(record, "text", ""))[: _PREVIEW_CHARS * 2] if record else ""
                _scored.append((_score(preview, _context), _order[candidate.chunk_id], candidate))
            _scored.sort(key=lambda triple: (-triple[0], triple[1]))
            winners = [candidate for _, _, candidate in _scored]
            semantic_promoted = any(score > 0 for score, _, _ in _scored)
        except Exception as exc:
            logger.warning(
                "v2 evidence semantic promotion failed; using RRF order",
                extra={"category": type(exc).__name__},
            )
            semantic_promoted = False
    expanded = expand_small_to_big(index, winners, neighbor_window=active.neighbor_window)
    selected, total = select_passages_under_budget(
        expanded,
        budget_tokens=active.budget_tokens,
        index=index,
        priority_child_ids=tuple(candidate.chunk_id for candidate in winners),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metadata: dict[str, Any] = {
        "planner_query_count": len(cleaned),
        "primary_query_digest": _short_digest(cleaned[0]),
        "branch_lists": len(ranked_lists),
        "branch_top_k": active.branch_top_k,
        "rrf_k": active.rrf_k,
        "pool_cap": active.pool_cap,
        "fused_unique": len(fused),
        "pool_unique": len(pool_ids),
        "diverse_unique": len(diverse),
        "top_child_cap": active.top_child_cap,
        "selected_winners": len(winners),
        "retrieval_backend": "rrf-only/1",
        "semantic_promoted": semantic_promoted,
        "neighbor_window": active.neighbor_window,
        "expanded_passages": len(expanded),
        "selected_passages": len(selected),
        "budget_tokens": active.budget_tokens,
        "total_tokens": total,
        "latency_ms": elapsed_ms,
        "latency_budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
        "latency_over_budget": elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS,
    }
    logger.info(
        "v2 evidence queries=%d pool=%d winners=%d passages=%d tokens=%d",
        len(cleaned),
        len(diverse),
        len(winners),
        len(selected),
        total,
    )
    return EvidencePack(
        passages=tuple(selected),
        total_tokens=total,
        corpus_version=corpus_version,
        retrieval_metadata=metadata,
    )


def retrieve_evidence_for_recovery(
    index: HybridIndex,
    queries: object,
    *,
    config: RetrievalConfig | None = None,
) -> EvidencePack:
    """Run the RRF-only pipeline for one bounded empty-pack recovery list.

    Recovery carries the live turn plus conversation context directly
    (1..6 queries, never canned generics). Ranking, dedup/diversity,
    small-to-big expansion and budget selection are identical to
    :func:`retrieve_evidence`; only the query-count validator allows
    short recovery lists.
    """
    active = config if config is not None else RetrievalConfig()
    if active.branch_top_k <= 0 or active.rrf_k <= 0:
        raise EvidenceError("branch_top_k and rrf_k must be > 0")
    if active.pool_cap <= 0 or active.top_child_cap <= 0:
        raise EvidenceError("pool caps must be > 0")
    if active.max_per_section <= 0 or active.neighbor_window < 0:
        raise EvidenceError("diversity/neighbor parameters are invalid")
    if active.budget_tokens <= 0:
        raise EvidenceError("budget_tokens must be > 0")
    cleaned = validate_recovery_queries(queries)
    corpus_version = str(index.metadata.get("ru_artifact_sha256", ""))
    if not cleaned:
        return empty_evidence_pack(corpus_version=corpus_version)
    started = time.perf_counter()
    ranked_lists, per_query_ids = run_branch_searches(
        index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists,
        per_query_ids,
        rrf_k=active.rrf_k,
        pool_cap=active.pool_cap,
    )
    diverse = dedup_and_diversify(
        index,
        pool_ids,
        fused,
        pool_cap=active.pool_cap,
        max_per_section=active.max_per_section,
    )
    sections = {chunk_id: record.section for chunk_id, record in index.chunks.items()}
    winners = select_top_candidates(
        diverse,
        top_cap=active.top_child_cap,
        sections=sections,
    )
    expanded = expand_small_to_big(index, winners, neighbor_window=active.neighbor_window)
    selected, total = select_passages_under_budget(
        expanded,
        budget_tokens=active.budget_tokens,
        index=index,
        priority_child_ids=tuple(candidate.chunk_id for candidate in winners),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metadata: dict[str, Any] = {
        "planner_query_count": len(cleaned),
        "primary_query_digest": _short_digest(cleaned[0]),
        "branch_lists": len(ranked_lists),
        "branch_top_k": active.branch_top_k,
        "rrf_k": active.rrf_k,
        "pool_cap": active.pool_cap,
        "fused_unique": len(fused),
        "pool_unique": len(pool_ids),
        "diverse_unique": len(diverse),
        "top_child_cap": active.top_child_cap,
        "selected_winners": len(winners),
        "retrieval_backend": "rrf-only/1",
        "recovery": True,
        "neighbor_window": active.neighbor_window,
        "expanded_passages": len(expanded),
        "selected_passages": len(selected),
        "budget_tokens": active.budget_tokens,
        "total_tokens": total,
        "latency_ms": elapsed_ms,
        "latency_budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
        "latency_over_budget": elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS,
    }
    logger.info(
        "v2 recovery evidence queries=%d pool=%d winners=%d passages=%d tokens=%d",
        len(cleaned),
        len(diverse),
        len(winners),
        len(selected),
        total,
    )
    return EvidencePack(
        passages=tuple(selected),
        total_tokens=total,
        corpus_version=corpus_version,
        retrieval_metadata=metadata,
    )


def to_prompt_passages(pack: EvidencePack) -> list[Any]:
    """Map pack passages to prompt-builder passages (exact text + provenance)."""
    from aa.conversation.prompt_builder import EvidencePassage

    return [
        EvidencePassage(
            passage_id=item.passage_id,
            source=item.source_id,
            section=item.section_id,
            text=item.exact_text,
        )
        for item in pack.passages
    ]


def _escape_text(value: str) -> str:
    return _xml_escape_text(value, {"'": "&apos;", '"': "&quot;"})


def render_book_evidence(pack: EvidencePack) -> str:
    """Render ``<book_evidence>`` carrying only exact text + provenance.

    Ranking metadata (RRF/BM25/dense scores, embeddings, planner
    reasoning, search previews) can never enter this block.
    """
    lines: list[str] = ["<book_evidence>"]
    if pack.passages:
        for item in pack.passages:
            lines.append(
                f"<passage id={_xml_quoteattr(item.passage_id)} "
                f"source={_xml_quoteattr(item.source_id)} "
                f"section={_xml_quoteattr(item.section_id)}>"
                f"{_escape_text(item.exact_text)}</passage>"
            )
    else:
        lines.append("(no book evidence supplied for this turn)")
    lines.append("</book_evidence>")
    return "\n".join(lines)


__all__ = [
    "BRANCH_TOP_K",
    "INTERACTIVE_LATENCY_BUDGET_MS",
    "MAX_PER_SECTION",
    "MAX_PLANNER_QUERIES",
    "MIN_PLANNER_QUERIES",
    "NEIGHBOR_WINDOW",
    "POOL_CAP",
    "TOP_CHILD_CAP",
    "broad_fused_ranking",
    "candidate_need_provenance",
    "candidate_query_provenance",
    "clear_query_vector_cache",
    "query_vector_cache_info",
    "EvidenceError",
    "EvidenceIntegrityError",
    "EvidencePack",
    "EvidencePassageData",
    "RetrievalConfig",
    "check_passage_id_consistency",
    "enrich_pack_provenance",
    "stable_passage_id",
    "dedup_and_diversify",
    "empty_evidence_pack",
    "expand_small_to_big",
    "fuse_query_pool",
    "render_book_evidence",
    "retrieve_evidence",
    "retrieve_evidence_for_recovery",
    "run_branch_searches",
    "select_passages_under_budget",
    "select_top_candidates",
    "to_prompt_passages",
    "validate_planner_queries",
    "validate_recovery_queries",
]
