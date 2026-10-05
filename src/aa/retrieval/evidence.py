"""Multi-query hybrid retrieval with BGE reranking and Evidence Packs (issue #116).

Target retrieval pipeline for the new conversation graph, built on the
#113 planner state (``QueryPlan.queries``: 0 or 10..16 context-resolved
Russian queries) and the #115 RAM-resident canonical index:

```text
QueryPlan(10-16) -> BM25 + E5/FAISS per query -> global RRF(k=60)
  -> per-query retention -> overlap dedup/diversity -> BGE rerank
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
- ``queries[0]`` (the direct context-resolved formulation) is the only
  reranker query, scored as ``queries[0] <-> exact Russian child text``;
- the reranker only reorders/selects; canonical text and provenance are
  never rewritten;
- small-to-big expansion stays within one canonical source/section
  unless an explicit neighbor link crosses a valid boundary, merges
  overlapping/adjacent windows, and keeps exact text plus provenance;
- the Evidence Pack carries only generation-relevant evidence; ranking
  metadata (RRF/BM25/dense/rerank scores, embeddings, planner reasoning,
  search previews) is internal-only and never enters ``<book_evidence>``.

This path has zero dependency on legacy semantic logic: no handwritten
query-expansion dictionaries and no legacy planner structures.
"""

from __future__ import annotations

import hashlib
import logging
import math
import numbers
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
from aa.retrieval.reranker import CrossEncoderReranker, RerankerError, get_reranker

logger = logging.getLogger("aa.retrieval.evidence")

MIN_PLANNER_QUERIES = 10
MAX_PLANNER_QUERIES = 16
BRANCH_TOP_K = 40
RERANKER_POOL_CAP = 64
POST_RERANK_CHILD_CAP = 16
MAX_PER_SECTION = 4
NEIGHBOR_WINDOW = 1
# Interactive Telegram budget for one warm retrieval turn (mirrors the
# frozen v2 qualification gate). The qualified BGE validation records
# ~20-22s p50/p95 with per-turn RRF plus 64-candidate CPU BGE rerank,
# far above this budget: production cutover stays blocked until explicit
# performance acceptance or optimization lands.
INTERACTIVE_LATENCY_BUDGET_MS = 5000.0

# Optimized interactive pool for the BGE rerank hot path. CPU BGE cost
# scales linearly with scored pairs, so 64 -> 16 cuts BGE forwards ~4x
# (frozen ~21s warm p50 suggests ~5-6s on the same CPU before other
# overheads). Branch recall (top-40 RRF union) is untouched; only the
# reranked prefix shrinks. Requires re-validation of recall/rerank lift
# before cutover: use interactive_retrieval_config() explicitly.
INTERACTIVE_RERANKER_POOL_CAP = 16
INTERACTIVE_POST_RERANK_CHILD_CAP = 16


class EvidenceError(ValueError):
    """Raised when the evidence pipeline cannot serve a request (fails closed)."""


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable retrieval/evidence parameters (recorded in metadata/evals)."""

    branch_top_k: int = BRANCH_TOP_K
    rrf_k: int = RRF_K
    reranker_pool_cap: int = RERANKER_POOL_CAP
    post_rerank_child_cap: int = POST_RERANK_CHILD_CAP
    max_per_section: int = MAX_PER_SECTION
    neighbor_window: int = NEIGHBOR_WINDOW
    budget_tokens: int = RETRIEVED_PASSAGES_BUDGET_TOKENS


def interactive_retrieval_config() -> RetrievalConfig:
    """Return the optimized interactive config for the BGE hot path.

    Keeps branch recall (top-40 per branch, RRF k=60) and the evidence
    budget identical to the full-quality default while capping the CPU
    BGE rerank to ``INTERACTIVE_RERANKER_POOL_CAP`` candidates. This is
    the documented optimization for the frozen ~21-22s warm p50/p95 vs
    the 5s budget: ~4x fewer BGE forwards on the hot path. Cutover still
    requires re-validation plus explicit performance acceptance via
    ``aa.qualification.v2_retrieval.require_v2_cutover_acceptance``.
    """
    return RetrievalConfig(
        branch_top_k=BRANCH_TOP_K,
        rrf_k=RRF_K,
        reranker_pool_cap=INTERACTIVE_RERANKER_POOL_CAP,
        post_rerank_child_cap=INTERACTIVE_POST_RERANK_CHILD_CAP,
        max_per_section=MAX_PER_SECTION,
        neighbor_window=NEIGHBOR_WINDOW,
        budget_tokens=RETRIEVED_PASSAGES_BUDGET_TOKENS,
    )


def is_interactive_config(config: RetrievalConfig) -> bool:
    """Return True when ``config`` respects the interactive BGE pool cap.

    The frozen 64-candidate validation exceeds the 5s budget, so only
    configs at or below ``INTERACTIVE_RERANKER_POOL_CAP`` qualify as the
    optimized production path. Branch recall parameters are intentionally
    not constrained here: the interactive optimization only shrinks the
    reranked prefix.
    """
    return config.reranker_pool_cap <= INTERACTIVE_RERANKER_POOL_CAP


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
    of one lock/model call per query. Returned vectors are L2-normalized
    and order-preserving; ranking is unchanged.
    """
    backend = str(index.metadata.get("embedding_backend", HASHING_BACKEND_NAME))
    dim = int(index.metadata.get("embedding_dim", HASHING_DIM))
    if backend == HASHING_BACKEND_NAME:
        return [hashing_embed(query, dim=dim) for query in queries]
    if backend == E5_BACKEND_NAME:
        try:
            vectors = e5_embed(["query: " + query for query in queries])
        except DenseError as exc:
            raise EvidenceError(str(exc)) from exc
        return [l2_normalize(vector) for vector in vectors]
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
    pool_cap: int = RERANKER_POOL_CAP,
) -> tuple[dict[str, FusedCandidate], list[str]]:
    """Fuse branch rankings with RRF, retaining each query's best candidate."""
    if rrf_k <= 0:
        raise EvidenceError("rrf_k must be > 0")
    if pool_cap <= 0:
        raise EvidenceError("pool_cap must be > 0")
    fused = rrf_fuse(ranked_lists, k=rrf_k)
    ordered = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
    pool: list[FusedCandidate] = []
    seen: set[str] = set()
    # Preserve multi-query recall: keep each query's best unique candidate
    # before filling the pool by global RRF rank.
    for contributed in per_query_ids:
        best: FusedCandidate | None = None
        for chunk_id in contributed:
            candidate = fused.get(chunk_id)
            if candidate is None or chunk_id in seen:
                continue
            if best is None or candidate.fused_score > best.fused_score:
                best = candidate
        if best is not None:
            pool.append(best)
            seen.add(best.chunk_id)
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
    pool_cap: int = RERANKER_POOL_CAP,
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


def rerank_candidates(
    index: HybridIndex,
    candidates: list[FusedCandidate],
    *,
    reranker_query: str,
    reranker: CrossEncoderReranker | None = None,
    post_rerank_cap: int = POST_RERANK_CHILD_CAP,
) -> tuple[list[FusedCandidate], list[float]]:
    """Rerank fused candidates with ``queries[0] <-> exact child text``.

    Batched scoring through one long-lived reranker instance. Ordering
    and selection may change; canonical text and provenance never do.
    """
    if post_rerank_cap <= 0:
        raise EvidenceError("post_rerank_cap must be > 0")
    if not candidates:
        return [], []
    if not reranker_query.strip():
        raise EvidenceError("reranker query must be a non-empty string")
    active = reranker if reranker is not None else get_reranker()
    texts: list[str] = []
    for candidate in candidates:
        record = index.chunks.get(candidate.chunk_id)
        if record is None:
            raise EvidenceError(f"fused candidate is not indexed: {candidate.chunk_id!r}")
        if _sha256_text(record.text) != record.text_sha256:
            raise EvidenceError(f"RAM chunk checksum mismatch: {record.chunk_id!r}")
        texts.append(record.text)
    try:
        scores = active.score(reranker_query, texts)
    except RerankerError as exc:
        raise EvidenceError(str(exc)) from exc
    if len(scores) != len(candidates):
        raise EvidenceError("reranker must return one score per candidate")
    clean_scores: list[float] = []
    for score in scores:
        if isinstance(score, bool) or isinstance(score, (str, bytes, bytearray)):
            raise EvidenceError("reranker scores must be floats")
        if not isinstance(score, numbers.Real):
            try:
                value = float(score)  # numpy scalars without Real registration
            except (TypeError, ValueError, ArithmeticError) as exc:
                raise EvidenceError("reranker scores must be floats") from exc
            if not math.isfinite(value):
                raise EvidenceError("reranker scores must be finite floats")
            clean_scores.append(value)
            continue
        value = float(score)
        if not math.isfinite(value):
            raise EvidenceError("reranker scores must be finite floats")
        clean_scores.append(value)
    ranked = sorted(
        zip(candidates, clean_scores, strict=True),
        key=lambda pair: (pair[1], pair[0].fused_score),
        reverse=True,
    )
    winners = [candidate for candidate, _ in ranked[:post_rerank_cap]]
    ordered_scores = [score for _, score in ranked[:post_rerank_cap]]
    return winners, ordered_scores


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
    """Expand reranked child hits to coherent parent/neighbor passages.

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
            raise EvidenceError(f"reranked winner is not indexed: {chunk_id!r}")
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
    for seq, (section_id, window) in enumerate(merged):
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
        exact_text = "\n".join(texts)
        first = index.chunks[members[0]]
        starts = [index.chunks[item].char_start for item in members]
        ends = [index.chunks[item].char_end for item in members]
        passages.append(
            EvidencePassageData(
                passage_id=f"{section_id}#exp{seq:04d}",
                exact_text=exact_text,
                source_id=first.source_id,
                section_id=section_id,
                child_chunk_ids=tuple(members),
                char_start=min(starts),
                char_end=max(ends),
                text_sha256=_sha256_text(exact_text),
                source_sha256=first.source_sha256,
            )
        )
    # Highest-value (best rerank order) passages first.
    passages.sort(
        key=lambda item: min(rank_of[cid] for cid in item.child_chunk_ids if cid in rank_of)
    )
    return passages


def select_passages_under_budget(
    passages: list[EvidencePassageData],
    *,
    budget_tokens: int = RETRIEVED_PASSAGES_BUDGET_TOKENS,
    index: HybridIndex | None = None,
) -> tuple[list[EvidencePassageData], int]:
    """Select coherent passages atomically under the source-token budget.

    No passage is silently character-truncated: a passage either fits in
    full or is skipped in the first pass. When ``index`` is provided and
    the first pass leaves no passage selected, or drops the top-ranked
    passage while one of its constituent child chunks would fit, an
    atomic single-child fallback is emitted: the highest-priority
    fitting child chunks become one-child passages with exact text and
    provenance taken from ``index``. Without ``index`` no fallback text
    can be mapped to an exact child id and char span, so oversized
    passages are skipped atomically instead of truncating.
    """
    if budget_tokens <= 0:
        raise EvidenceError("budget_tokens must be > 0")
    selected: list[EvidencePassageData] = []
    total = 0
    for passage in passages:
        need = estimate_text_tokens(passage.exact_text)
        if total + need <= budget_tokens:
            selected.append(passage)
            total += need
            continue
        # Atomic: skip oversized passages instead of truncating.
        continue
    if index is not None and passages:
        covered = {cid for item in selected for cid in item.child_chunk_ids}
        top = passages[0]
        top_covered = any(cid in covered for cid in top.child_chunk_ids)
        if not selected or not top_covered:
            remaining = budget_tokens - total
            seen: set[str] = set(covered)
            fallbacks: list[EvidencePassageData] = []
            for passage in passages:
                for cid in passage.child_chunk_ids:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    record = index.chunks.get(cid)
                    if record is None:
                        continue
                    need = estimate_text_tokens(record.text)
                    if need <= 0 or need > remaining:
                        continue
                    fallbacks.append(
                        EvidencePassageData(
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
                    )
                    remaining -= need
                    total += need
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            if fallbacks:
                if not selected:
                    selected = fallbacks
                elif not top_covered:
                    # Keep rerank priority: top-winner atoms come first.
                    top_ids = set(top.child_chunk_ids)
                    head = [item for item in fallbacks if item.child_chunk_ids[0] in top_ids]
                    tail = [item for item in fallbacks if item.child_chunk_ids[0] not in top_ids]
                    selected = [*head, *selected, *tail]
    return selected, total


def retrieve_evidence(
    index: HybridIndex,
    queries: object,
    *,
    config: RetrievalConfig | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> EvidencePack:
    """Run the full target pipeline for one planner query list.

    ``queries`` is the minimal ``QueryPlan.queries`` from #113. An empty
    list performs no retrieval and returns an empty pack. Otherwise the
    count must be 10..16 and ``queries[0]`` is the canonical reranker
    query. The answering model receives only the returned exact
    passages; ``retrieval_metadata`` stays internal.

    When ``config`` is omitted, ordinary turns use the optimized
    interactive pool (``interactive_retrieval_config``, 16 BGE
    candidates) instead of the frozen full-quality 64-candidate default:
    CPU BGE cost scales linearly, so the default hot path issues ~4x
    fewer forwards than the ~21-22s warm p50/p95 validation recorded
    against the 5s budget. The frozen BGE validation must pass an
    explicit full-quality ``RetrievalConfig()`` so its numbers stay
    comparable.
    """
    active = config if config is not None else interactive_retrieval_config()
    if active.branch_top_k <= 0 or active.rrf_k <= 0:
        raise EvidenceError("branch_top_k and rrf_k must be > 0")
    if active.reranker_pool_cap <= 0 or active.post_rerank_child_cap <= 0:
        raise EvidenceError("reranker caps must be > 0")
    if active.max_per_section <= 0 or active.neighbor_window < 0:
        raise EvidenceError("diversity/neighbor parameters are invalid")
    if active.budget_tokens <= 0:
        raise EvidenceError("budget_tokens must be > 0")
    cleaned = validate_planner_queries(queries)
    corpus_version = str(index.metadata.get("ru_artifact_sha256", ""))
    if not cleaned:
        return empty_evidence_pack(corpus_version=corpus_version)
    started = time.perf_counter()
    # Resolve the long-lived reranker once per turn and reuse the same
    # instance for scoring and metadata: a second get_reranker() call
    # would repeat validation work on the hot path.
    active_reranker = reranker if reranker is not None else get_reranker()
    ranked_lists, per_query_ids = run_branch_searches(
        index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists,
        per_query_ids,
        rrf_k=active.rrf_k,
        pool_cap=active.reranker_pool_cap,
    )
    diverse = dedup_and_diversify(
        index,
        pool_ids,
        fused,
        pool_cap=active.reranker_pool_cap,
        max_per_section=active.max_per_section,
    )
    winners, rerank_scores = rerank_candidates(
        index,
        diverse,
        reranker_query=cleaned[0],
        reranker=active_reranker,
        post_rerank_cap=active.post_rerank_child_cap,
    )
    expanded = expand_small_to_big(index, winners, neighbor_window=active.neighbor_window)
    selected, total = select_passages_under_budget(
        expanded, budget_tokens=active.budget_tokens, index=index
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metadata: dict[str, Any] = {
        "planner_query_count": len(cleaned),
        "reranker_query_digest": _short_digest(cleaned[0]),
        "branch_lists": len(ranked_lists),
        "branch_top_k": active.branch_top_k,
        "rrf_k": active.rrf_k,
        "reranker_pool_cap": active.reranker_pool_cap,
        "fused_unique": len(fused),
        "pool_unique": len(pool_ids),
        "diverse_unique": len(diverse),
        "post_rerank_child_cap": active.post_rerank_child_cap,
        "reranked_winners": len(winners),
        "reranker_model": active_reranker.model_id,
        "reranker_revision": active_reranker.revision,
        "reranker_backend": active_reranker.backend,
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

    Ranking metadata (RRF/BM25/dense/rerank scores, embeddings, planner
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
    "INTERACTIVE_POST_RERANK_CHILD_CAP",
    "INTERACTIVE_RERANKER_POOL_CAP",
    "MAX_PER_SECTION",
    "MAX_PLANNER_QUERIES",
    "MIN_PLANNER_QUERIES",
    "NEIGHBOR_WINDOW",
    "POST_RERANK_CHILD_CAP",
    "RERANKER_POOL_CAP",
    "interactive_retrieval_config",
    "is_interactive_config",
    "EvidenceError",
    "EvidencePack",
    "EvidencePassageData",
    "RetrievalConfig",
    "dedup_and_diversify",
    "empty_evidence_pack",
    "expand_small_to_big",
    "fuse_query_pool",
    "render_book_evidence",
    "rerank_candidates",
    "retrieve_evidence",
    "run_branch_searches",
    "select_passages_under_budget",
    "to_prompt_passages",
    "validate_planner_queries",
]
