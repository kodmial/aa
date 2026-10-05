"""Multi-query hybrid retrieval with Evidence Packs (issue #116).

Target retrieval pipeline for the new conversation graph, built on the
#113 planner state (``QueryPlan.queries``: 0 or 10..16 context-resolved
Russian queries) and the #115 RAM-resident canonical index:

```text
QueryPlan(10-16) -> BM25 + E5/FAISS per query -> global RRF(k=60)
  -> per-query retention -> overlap dedup/diversity
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
- selection after fusion is RRF-only (fused-score order); canonical
  text and provenance are never rewritten;
- small-to-big expansion stays within one canonical source/section
  unless an explicit neighbor link crosses a valid boundary, merges
  overlapping/adjacent windows, and keeps exact text plus provenance;
- the Evidence Pack carries only generation-relevant evidence; ranking
  metadata (RRF/BM25/dense scores, embeddings, planner reasoning,
  search previews) is internal-only and never enters ``<book_evidence>``.

This path has zero dependency on legacy semantic logic: no handwritten
query-expansion dictionaries and no legacy planner structures. It is
RAM-only (BM25 + E5/FAISS plus RRF) with zero network/download
dependency on the hot path and no second-stage cross-encoder/BGE
reranker.
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
    enforce_diversity,
    rrf_fuse,
)
from aa.retrieval.index import HybridIndex
from aa.retrieval.lexical import lexical_search_conn

logger = logging.getLogger("aa.retrieval.evidence")

MIN_PLANNER_QUERIES = 10
MAX_PLANNER_QUERIES = 16
BRANCH_TOP_K = 40
POOL_CAP = 64
TOP_CHILD_CAP = 16
MAX_PER_SECTION = 4
NEIGHBOR_WINDOW = 1
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


def validate_planner_queries(queries: object) -> list[str]:
    """Validate the minimal #113 planner query list (fails closed)."""
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
    rest of the pool fills by global RRF rank. With 10..16 planner
    queries and the 64-cap pool this guarantees the ticket
    rule (at least the best unique candidate per query) while leaving
    ample slots for global RRF depth.
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


def dedup_and_diversify(
    index: HybridIndex,
    pool_ids: list[str],
    fused: dict[str, FusedCandidate],
    *,
    pool_cap: int = POOL_CAP,
    max_per_section: int = MAX_PER_SECTION,
) -> list[FusedCandidate]:
    """Collapse overlapping child spans and apply bounded section diversity."""
    if pool_cap <= 0 or max_per_section <= 0:
        raise EvidenceError("pool_cap and max_per_section must be > 0")
    candidates = [fused[chunk_id] for chunk_id in pool_ids if chunk_id in fused]
    spans = {
        chunk_id: (record.section, record.char_start, record.char_end)
        for chunk_id, record in index.chunks.items()
    }
    deduped = deduplicate_overlaps(candidates, spans=spans)
    sections = {chunk_id: record.section for chunk_id, record in index.chunks.items()}
    return enforce_diversity(
        deduped, sections=sections, max_n=pool_cap, max_per_section=max_per_section
    )


def select_top_candidates(
    candidates: list[FusedCandidate],
    *,
    top_cap: int = TOP_CHILD_CAP,
) -> list[FusedCandidate]:
    """Select the top RRF-ordered candidates (no second-stage reranker).

    Ordering is pure fused-score order; canonical text and provenance
    are never rewritten.
    """
    if top_cap <= 0:
        raise EvidenceError("top_cap must be > 0")
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
    return ranked[:top_cap]


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
    seq = 0
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
            passages.append(
                EvidencePassageData(
                    passage_id=f"{section_id}#exp{seq:04d}",
                    exact_text=exact_text,
                    source_id=first.source_id,
                    section_id=section_id,
                    child_chunk_ids=tuple(run),
                    char_start=min(starts),
                    char_end=max(ends),
                    text_sha256=_sha256_text(exact_text),
                    source_sha256=first.source_sha256,
                )
            )
            passage_ranks.append(run_best)
            seq += 1
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
            passage_id=f"{record.section}#atom-{cid.split(':')[-1]}",
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
        for passage in passages:
            need = estimate_text_tokens(passage.exact_text)
            if total + need <= budget_tokens:
                selected.append(passage)
                covered.update(passage.child_chunk_ids)
                total += need
                continue

            direct_winners = sorted(
                (
                    cid
                    for cid in passage.child_chunk_ids
                    if cid in priority and cid not in covered
                ),
                key=priority.__getitem__,
            )
            for cid in direct_winners:
                atom = _atom(cid)
                if atom is None:
                    continue
                atom_need = estimate_text_tokens(atom.exact_text)
                if atom_need <= 0 or total + atom_need > budget_tokens:
                    continue
                selected.append(atom)
                covered.add(cid)
                total += atom_need
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
                    head = [
                        item for item in fallbacks if item.child_chunk_ids[0] in top_ids
                    ]
                    tail = [
                        item for item in fallbacks if item.child_chunk_ids[0] not in top_ids
                    ]
                    selected = [*head, *selected, *tail]
    return selected, total


def retrieve_evidence(
    index: HybridIndex,
    queries: object,
    *,
    config: RetrievalConfig | None = None,
) -> EvidencePack:
    """Run the full RRF-only pipeline for one planner query list.

    ``queries`` is the minimal ``QueryPlan.queries`` from #113. An empty
    list performs no retrieval and returns an empty pack. Otherwise the
    count must be 10..16. The answering model receives only the
    returned exact passages; ``retrieval_metadata`` stays internal.

    Pipeline: ``QueryPlan -> BM25+E5 -> RRF -> dedup/diversity ->
    small-to-big`` over the RAM-resident index with zero
    network/download dependency and no second-stage reranker.
    Per-query retention keeps each query's best unique candidate
    first, then fills the remainder by global RRF rank.
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
    winners = select_top_candidates(diverse, top_cap=active.top_child_cap)
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
    "clear_query_vector_cache",
    "query_vector_cache_info",
    "EvidenceError",
    "EvidencePack",
    "EvidencePassageData",
    "RetrievalConfig",
    "dedup_and_diversify",
    "empty_evidence_pack",
    "expand_small_to_big",
    "fuse_query_pool",
    "render_book_evidence",
    "retrieve_evidence",
    "run_branch_searches",
    "select_passages_under_budget",
    "select_top_candidates",
    "to_prompt_passages",
    "validate_planner_queries",
]
