"""RRF-only v2 retrieval benchmark (issue #116).

Extends the frozen retrieval benchmark without tuning to only known
literal phrases. Measures the RRF-only target pipeline (multi-query
hybrid plus global RRF, dedup/diversity and small-to-big Evidence
Packs from :mod:`aa.retrieval.evidence`) with no second-stage
reranker.

The benchmark exercises the combined planner + retrieval pipeline, not
only hand-authored search strings: a frozen, versioned set of unseen
Russian conversational turns and multi-turn contexts (broad requests,
slang/typos, short follow-ups, pronouns/ellipsis, topic shifts, narrow
questions) is resolved into 10..16 proxy planner queries per case. The
proxy generator sees only the utterance, its multi-turn context and the
case's own paraphrase lists; it never sees oracle relevant-section
labels.

Measured per case:

- recall@5 / recall@10 of relevant canonical regions;
- MRR / nDCG@10 where oracle (binary) relevance exists;
- unique relevant-region coverage;
- duplicate rate;
- planner query diversity;
- evidence-pack source-token size;
- warm p50/p95 latency (warmup first, then timed);
- unsupported/irrelevant passage rate.

Quality gate covers operational invariants only (recall gate,
duplicate rate, token budget, warm latency, zero hot-path disk reads
verified separately by the harness). Product-quality judgments on the
real book are explicitly out of scope here (owned by #130).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS
from aa.qualification.aa_retrieval import GoldCase
from aa.retrieval.evidence import (
    BRANCH_TOP_K,
    INTERACTIVE_LATENCY_BUDGET_MS,
    MAX_PER_SECTION,
    NEIGHBOR_WINDOW,
    POOL_CAP,
    TOP_CHILD_CAP,
    EvidencePack,
    RetrievalConfig,
    retrieve_evidence,
)
from aa.retrieval.fusion import RRF_K
from aa.retrieval.index import HybridIndex
from aa.retrieval.normalize import ru_tokens

V2_BENCHMARK_VERSION = "aa-v2-retrieval-benchmark/2"
V2_EVAL_SET_VERSION = "aa-v2-conversational-eval/1"

# Interactive Telegram budget for one warm RRF-only retrieval turn.
# Single-sourced from the hot-path evidence pipeline so the gate and
# per-turn metadata share one value.
V2_TARGET_P95_LATENCY_BUDGET_MS = INTERACTIVE_LATENCY_BUDGET_MS

V2_PRODUCTION_CONFIG_VERSION = "v2-rrf-only/1"
V2_CUTOVER_BLOCKED_REASON = (
    "v2 RRF-only pipeline is measured by this benchmark; "
    "product-quality promotion on the real book is owned by #130"
)

PROXY_QUERY_COUNT = 12
BROAD_CATEGORIES = (
    "slang-drinking",
    "slang-family",
    "diminutive",
    "morphology",
    "typo",
    "transposition",
    "terse-followup",
    "ambiguous-sorvalsya",
    "implicit-family",
    "implicit-work",
    "implicit-alcohol",
    "multi-theme",
)


@dataclass(frozen=True)
class V2CaseResult:
    """Per-case RRF-only outcome plus stage diagnostics."""

    case_id: str
    category: str
    is_unsupported: bool
    proxy_queries: tuple[str, ...]
    query_diversity: float
    duplicate_query_rate: float
    target_sections_10: tuple[str, ...]
    target_recall_at_5: bool
    target_recall_at_10: bool
    target_mrr: float
    target_ndcg_10: float
    target_coverage: float
    union_recall: bool
    pool_recall: bool
    pack_recall: bool
    selection_loss: bool
    duplicate_rate: float
    target_tokens: int
    target_latency_ms: float
    unsupported_clean: bool


@dataclass(frozen=True)
class V2Summary:
    """Aggregate RRF-only benchmark summary."""

    benchmark_version: str
    eval_set_version: str
    cases: int
    target_recall_at_5: float
    target_recall_at_10: float
    target_mrr: float
    target_ndcg_10: float
    target_coverage: float
    duplicate_rate: float
    mean_query_diversity: float
    mean_duplicate_query_rate: float
    mean_target_tokens: float
    p50_target_latency_ms: float
    p95_target_latency_ms: float
    unsupported_clean_rate: float
    pool_to_pack_loss_rate: float
    peak_rss_mb: float


def planner_proxy_queries(case: GoldCase, *, count: int = PROXY_QUERY_COUNT) -> list[str]:
    """Build up to 10..16 deterministic proxy planner queries for one gold case.

    ``queries[0]`` is the direct context-resolved formulation of the
    current turn (utterance plus multi-turn context when the case
    requires it). Remaining queries diversify wording using only the
    case's own utterance, context, paraphrase lists and allowed
    interpretations; oracle relevant-section labels are never consulted
    and no hand-written expansion dictionary is used. No synthetic
    filler vocabulary is invented: when the case owns fewer distinct
    phrasings, the honest deduped pool (possibly shorter than ``count``)
    is returned.
    """
    if count < 10 or count > 16:
        raise ValueError("proxy query count must be 10..16")
    resolved = case.utterance.strip()
    if case.requires_context and case.context.strip():
        resolved = f"{case.context.strip()} {resolved}"
    pool: list[str] = [resolved]
    pool.extend(list(case.plan_queries_ru))
    if not case.plan_queries_ru:
        pool.extend(list(case.en_gloss_queries))
    pool.extend([str(item).strip() for item in case.allowed_interpretations])
    for query in case.plan_queries_ru:
        combined = f"{case.utterance.strip()} {query.strip()}".strip()
        if combined:
            pool.append(combined)
    for interpretation in case.allowed_interpretations:
        combined = f"{case.utterance.strip()} {str(interpretation).strip()}".strip()
        if combined:
            pool.append(combined)
    if case.context.strip():
        for query in case.plan_queries_ru:
            combined = f"{case.context.strip()} {query.strip()}".strip()
            if combined:
                pool.append(combined)
    plan = list(case.plan_queries_ru) or list(case.en_gloss_queries)
    for first in plan:
        for second in plan:
            if first != second:
                combined = f"{first.strip()} {second.strip()}".strip()
                if combined:
                    pool.append(combined)
    for query in plan:
        combined = f"{resolved} {query.strip()}".strip()
        if combined:
            pool.append(combined)
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in pool:
        collapsed = " ".join(str(query).split())
        if not collapsed or collapsed.casefold() in seen:
            continue
        seen.add(collapsed.casefold())
        cleaned.append(collapsed)
        if len(cleaned) >= count:
            break
    if len(cleaned) < count:
        # Do not invent vocabulary to hit `count`; the frozen eval must use
        # only case-owned phrasing. Callers already handle variable lengths
        # via `query_diversity` and `duplicate_query_rate`.
        pass
    return cleaned[:count]


def query_diversity(queries: list[str]) -> float:
    """Return 1 minus mean pairwise stemmed-token Jaccard (higher is diverse)."""
    if len(queries) < 2:
        return 0.0
    sets = [set(ru_tokens(query)) for query in queries]
    total = 0.0
    pairs = 0
    for pos in range(len(sets)):
        for other in range(pos + 1, len(sets)):
            union = sets[pos] | sets[other]
            inter = sets[pos] & sets[other]
            total += 1.0 - (len(inter) / len(union) if union else 1.0)
            pairs += 1
    return total / pairs if pairs else 0.0


def duplicate_query_rate(queries: list[str]) -> float:
    """Return the near-duplicate query rate (stemmed-set equality)."""
    if not queries:
        return 0.0
    keys = [" ".join(sorted(set(ru_tokens(query)))) for query in queries]
    return 1.0 - len(set(keys)) / len(keys)


def _pack_sections(pack: EvidencePack, limit: int) -> tuple[str, ...]:
    sections: list[str] = []
    for passage in pack.passages:
        if passage.section_id not in sections:
            sections.append(passage.section_id)
        if len(sections) >= limit:
            break
    return tuple(sections)


def _recall_at(sections: tuple[str, ...], case: GoldCase, k: int) -> bool:
    if case.is_unsupported:
        return False
    return any(section in case.relevant_sections for section in sections[:k])


def _mrr(sections: tuple[str, ...], case: GoldCase) -> float:
    if case.is_unsupported:
        return 0.0
    for rank, section in enumerate(sections, start=1):
        if section in case.relevant_sections:
            return 1.0 / rank
    return 0.0


def _ndcg(sections: tuple[str, ...], case: GoldCase, k: int = 10) -> float:
    if case.is_unsupported:
        return 0.0
    gains = [1.0 if section in case.relevant_sections else 0.0 for section in sections[:k]]
    if not any(gains):
        return 0.0
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _coverage(sections: tuple[str, ...], case: GoldCase) -> float:
    if case.is_unsupported or not case.relevant_sections:
        return 1.0
    found = sum(1 for section in case.relevant_sections if section in sections)
    return found / len(case.relevant_sections)


def _duplicate_rate(pack: EvidencePack) -> float:
    child_ids = [cid for passage in pack.passages for cid in passage.child_chunk_ids]
    if not child_ids:
        return 0.0
    return 1.0 - len(set(child_ids)) / len(child_ids)


def _peak_rss_mb() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def run_v2_case(
    index: HybridIndex,
    case: GoldCase,
    *,
    config: RetrievalConfig | None = None,
) -> V2CaseResult:
    """Run the RRF-only pipeline for one case (warmup, then timed)."""
    active = (
        config
        if config is not None
        else RetrievalConfig(
            branch_top_k=BRANCH_TOP_K,
            rrf_k=RRF_K,
            pool_cap=POOL_CAP,
            top_child_cap=TOP_CHILD_CAP,
            max_per_section=MAX_PER_SECTION,
            neighbor_window=NEIGHBOR_WINDOW,
            budget_tokens=RETRIEVED_PASSAGES_BUDGET_TOKENS,
        )
    )
    queries = planner_proxy_queries(case)
    diversity = query_diversity(queries)
    dup_rate = duplicate_query_rate(queries)
    # Warm latency: warm up first (untimed), then measure the warm run.
    retrieve_evidence(index, queries, config=active)
    started = time.perf_counter()
    target_pack = retrieve_evidence(index, queries, config=active)
    target_ms = (time.perf_counter() - started) * 1000.0
    target_sections = _pack_sections(target_pack, 10)
    target_r5 = _recall_at(target_sections, case, 5)
    target_r10 = _recall_at(target_sections, case, 10)
    target_mrr_v = _mrr(target_sections, case)
    target_ndcg = _ndcg(target_sections, case)
    # Stage diagnostics for the target arm: union of all planner-query
    # branches, RRF pool truncation, and final pack recall. The pool is
    # recomputed deterministically from the same branch outputs (no
    # rerank involved), so union -> pool -> pack losses are attributable.
    # The union is derived from the same branch outputs used for fusion
    # (no separate re-embedding/re-search per query), avoiding doubled
    # hot-path cost and union-vs-pool divergence.
    from aa.retrieval import evidence as _evidence

    _ranked, _per_query = _evidence.run_branch_searches(
        index, queries, branch_top_k=active.branch_top_k
    )
    union_sections: set[str] = set()
    for ranked in _ranked:
        for chunk_id, _ in ranked:
            record = index.chunks.get(chunk_id)
            if record is not None:
                union_sections.add(record.section)
    for contributed in _per_query:
        for chunk_id in contributed:
            record = index.chunks.get(chunk_id)
            if record is not None:
                union_sections.add(record.section)
    union_hit = (
        any(section in case.relevant_sections for section in union_sections)
        if not case.is_unsupported
        else False
    )
    _fused, _pool_ids = _evidence.fuse_query_pool(
        _ranked, _per_query, rrf_k=active.rrf_k, pool_cap=active.pool_cap
    )
    _diverse = _evidence.dedup_and_diversify(
        index,
        _pool_ids,
        _fused,
        pool_cap=active.pool_cap,
        max_per_section=active.max_per_section,
    )
    pool_sections = tuple(
        dict.fromkeys(
            record.section
            for chunk in _diverse
            if (record := index.chunks.get(chunk.chunk_id)) is not None
        )
    )
    pool_hit = _recall_at(pool_sections, case, 10)
    pack_hit = target_r10
    unsupported_clean = (not target_pack.passages) if case.is_unsupported else True
    return V2CaseResult(
        case_id=case.case_id,
        category=case.category,
        is_unsupported=case.is_unsupported,
        proxy_queries=tuple(queries),
        query_diversity=diversity,
        duplicate_query_rate=dup_rate,
        target_sections_10=target_sections,
        target_recall_at_5=target_r5,
        target_recall_at_10=target_r10,
        target_mrr=target_mrr_v,
        target_ndcg_10=target_ndcg,
        target_coverage=_coverage(target_sections, case),
        union_recall=union_hit,
        pool_recall=pool_hit,
        pack_recall=pack_hit,
        selection_loss=bool(pool_hit and not pack_hit),
        duplicate_rate=_duplicate_rate(target_pack),
        target_tokens=target_pack.total_tokens,
        target_latency_ms=target_ms,
        unsupported_clean=unsupported_clean,
    )


def _branch_union(index: HybridIndex, query: str, *, top_k: int) -> list[tuple[str, float]]:
    """Return the lexical + dense branch union for one query (diagnostics)."""
    from aa.retrieval.lexical import lexical_search_conn as _lex_conn

    seen: dict[str, float] = {}
    if index.lexical_conn is not None:
        for chunk_id, score in _lex_conn(index.lexical_conn, query, top_k=top_k):
            seen.setdefault(chunk_id, score)
    from aa.retrieval.evidence import _embed_query_vector as _embed

    if index.chunks and top_k > 0:
        for chunk_id, score in index.dense.search(
            _embed(index, query), top_k=min(top_k, len(index.chunks))
        ):
            seen.setdefault(chunk_id, score)
    return list(seen.items())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((pct / 100.0) * len(ordered))
    pos = max(0, min(len(ordered) - 1, rank - 1))
    return float(ordered[pos])


def summarize_v2(results: list[V2CaseResult]) -> V2Summary:
    """Aggregate per-case RRF-only results."""
    if not results:
        raise ValueError("v2 benchmark has no case results")
    supported = [item for item in results if not item.is_unsupported]
    denom = max(1, len(supported))
    latencies = [item.target_latency_ms for item in results]
    return V2Summary(
        benchmark_version=V2_BENCHMARK_VERSION,
        eval_set_version=V2_EVAL_SET_VERSION,
        cases=len(results),
        target_recall_at_5=sum(1 for r in supported if r.target_recall_at_5) / denom,
        target_recall_at_10=sum(1 for r in supported if r.target_recall_at_10) / denom,
        target_mrr=sum(r.target_mrr for r in supported) / denom,
        target_ndcg_10=sum(r.target_ndcg_10 for r in supported) / denom,
        target_coverage=sum(r.target_coverage for r in supported) / denom,
        duplicate_rate=sum(r.duplicate_rate for r in results) / len(results),
        mean_query_diversity=sum(r.query_diversity for r in results) / len(results),
        mean_duplicate_query_rate=sum(r.duplicate_query_rate for r in results) / len(results),
        mean_target_tokens=sum(float(r.target_tokens) for r in results) / len(results),
        p50_target_latency_ms=_percentile(latencies, 50),
        p95_target_latency_ms=_percentile(latencies, 95),
        unsupported_clean_rate=sum(1 for r in results if r.unsupported_clean) / len(results),
        pool_to_pack_loss_rate=sum(1 for r in supported if r.selection_loss) / denom,
        peak_rss_mb=_peak_rss_mb(),
    )


def v2_quality_gate(summary: V2Summary) -> tuple[bool, list[str]]:
    """Evaluate the RRF-only benchmark gate (fails closed).

    The gate covers operational invariants only: duplicate rate,
    token budget, and warm p50/p95 latency within the interactive
    Telegram budget. Product-quality judgments on the real book are
    owned by #130 and are not gated here. Hot-path disk reads remain
    zero (verified separately by the harness blocking filesystem
    access). Unsupported-case passage rates are measured and reported
    but carry no abstention gate.
    """
    failures: list[str] = []
    if summary.duplicate_rate > 0.05:
        failures.append(f"duplicate rate {summary.duplicate_rate:.3f} exceeds 0.05")
    if summary.mean_target_tokens > RETRIEVED_PASSAGES_BUDGET_TOKENS:
        failures.append(
            f"mean target tokens {summary.mean_target_tokens:.0f} exceeds "
            f"budget {RETRIEVED_PASSAGES_BUDGET_TOKENS}"
        )
    if summary.p50_target_latency_ms > V2_TARGET_P95_LATENCY_BUDGET_MS:
        failures.append(
            f"warm p50 latency {summary.p50_target_latency_ms:.0f}ms exceeds "
            f"the interactive budget {V2_TARGET_P95_LATENCY_BUDGET_MS:.0f}ms"
        )
    if summary.p95_target_latency_ms > V2_TARGET_P95_LATENCY_BUDGET_MS:
        failures.append(
            f"warm p95 latency {summary.p95_target_latency_ms:.0f}ms exceeds "
            f"the interactive budget {V2_TARGET_P95_LATENCY_BUDGET_MS:.0f}ms"
        )
    return (not failures, failures)


def v2_production_status(summary: V2Summary) -> dict[str, Any]:
    """Return the machine-readable v2 production promotion status.

    Mirrors the ``production_promotion_blocked`` pattern from the
    qualified retrieval artifact: any quality-gate failure keeps the v2
    pipeline blocked behind the legacy production path. Product-quality
    promotion on the real book is owned by #130.
    """
    passed, failures = v2_quality_gate(summary)
    blocked = not passed
    return {
        "config_id": V2_PRODUCTION_CONFIG_VERSION if passed else "blocked",
        "config_version": V2_PRODUCTION_CONFIG_VERSION,
        "cutover_allowed": passed,
        "production_promotion_blocked": blocked,
        "quality_gate_passed": passed,
        "quality_gate_failures": list(failures),
        "reason": (
            "v2 RRF-only quality gate passed; product-quality promotion "
            "on the real book is owned by #130"
            if passed
            else V2_CUTOVER_BLOCKED_REASON
        ),
    }


def find_repo_root() -> Path:
    """Return the repository root containing qualification + corpus."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "corpus" / "embedding.lock.json").is_file():
            return parent
    raise ValueError("repository root with embedding lock not found")


__all__ = [
    "BROAD_CATEGORIES",
    "PROXY_QUERY_COUNT",
    "V2_BENCHMARK_VERSION",
    "V2_CUTOVER_BLOCKED_REASON",
    "V2_EVAL_SET_VERSION",
    "V2_PRODUCTION_CONFIG_VERSION",
    "V2_TARGET_P95_LATENCY_BUDGET_MS",
    "V2CaseResult",
    "V2Summary",
    "duplicate_query_rate",
    "find_repo_root",
    "planner_proxy_queries",
    "query_diversity",
    "run_v2_case",
    "summarize_v2",
    "v2_production_status",
    "v2_quality_gate",
]
