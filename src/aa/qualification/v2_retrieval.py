"""Frozen v2 retrieval benchmark: baseline vs target pipeline (issue #116).

Extends the frozen retrieval benchmark without tuning to only known
literal phrases. Compares the qualified legacy baseline (per-aspect
RRF-only hybrid from ``aa.qualification.aa_retrieval``) against the
target pipeline (multi-query hybrid plus global RRF, pinned local BGE
reranking and small-to-big Evidence Packs from
:mod:`aa.retrieval.evidence`).

The benchmark exercises the combined planner + retrieval pipeline, not
only hand-authored search strings: a frozen, versioned set of unseen
Russian conversational turns and multi-turn contexts (broad requests,
slang/typos, short follow-ups, pronouns/ellipsis, topic shifts, narrow
questions) is resolved into 10..16 proxy planner queries per case. The
proxy generator sees only the utterance, its multi-turn context and the
case's own paraphrase lists; it never sees oracle relevant-section
labels.

Measured per case and arm:

- recall@5 / recall@10 of relevant canonical regions;
- MRR / nDCG@10 where oracle (binary) relevance exists;
- unique relevant-region coverage;
- duplicate rate;
- reranker lift over RRF-only candidates;
- planner query diversity;
- evidence-pack source-token size;
- warm p50/p95 latency;
- unsupported/irrelevant passage rate.

Quality gate:

- no regression in the existing qualified retrieval recall gate;
- the reranker must show measurable ranking/relevance benefit over
  RRF-only on the versioned evaluation set;
- broad/colloquial/multi-turn cases improve or remain non-regressive;
- hot-path disk reads remain zero (verified by the test harness).

If the reranker cannot beat RRF-only on the frozen eval, this module
reports failure rather than removing the mandatory reranker.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.qualification.aa_retrieval import GoldCase
from aa.retrieval.evidence import (
    INTERACTIVE_LATENCY_BUDGET_MS,
    EvidencePack,
    RetrievalConfig,
    retrieve_evidence,
)
from aa.retrieval.index import HybridIndex
from aa.retrieval.normalize import ru_tokens
from aa.retrieval.reranker import CrossEncoderReranker

V2_BENCHMARK_VERSION = "aa-v2-retrieval-benchmark/1"
V2_EVAL_SET_VERSION = "aa-v2-conversational-eval/1"

# Interactive Telegram budget for one warm retrieval turn (RRF plus
# 64-candidate CPU BGE reranking). Single-sourced from the hot-path
# evidence pipeline so the gate and per-turn metadata share one value.
# The frozen BGE validation records ~20-22s p50/p95 per case, far above
# interactive expectations: cutover requires explicit performance
# acceptance or optimization (aa.retrieval.evidence
# .interactive_retrieval_config with a 16-candidate BGE pool, exact
# text-dedup in BGE scoring, quantized/GPU serving, or a documented
# higher budget) before this gate can pass.
V2_TARGET_P95_LATENCY_BUDGET_MS = INTERACTIVE_LATENCY_BUDGET_MS

# Production cutover is explicitly out of scope for #116. The committed
# BGE validation fails the latency gate (warm p95 ~22s vs the 5s
# interactive budget from per-turn RRF plus 64-candidate CPU BGE
# rerank), so the v2 pipeline must stay behind the legacy production
# path until explicit performance acceptance or optimization lands.
V2_PRODUCTION_CONFIG_VERSION = "v2-blocked-latency/1"
V2_CUTOVER_BLOCKED_REASON = (
    "v2 retrieval warm p95 ~21-22s exceeds the 5s interactive budget "
    "(per-turn RRF plus 64-candidate CPU BGE rerank); production "
    "cutover is blocked until explicit performance acceptance or "
    "optimization lands"
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
    """Per-case baseline-vs-target outcome plus stage diagnostics."""

    case_id: str
    category: str
    is_unsupported: bool
    proxy_queries: tuple[str, ...]
    query_diversity: float
    duplicate_query_rate: float
    baseline_sections_10: tuple[str, ...]
    target_sections_10: tuple[str, ...]
    baseline_recall_at_5: bool
    target_recall_at_5: bool
    baseline_recall_at_10: bool
    target_recall_at_10: bool
    baseline_mrr: float
    target_mrr: float
    baseline_ndcg_10: float
    target_ndcg_10: float
    baseline_coverage: float
    target_coverage: float
    union_recall: bool
    pool_recall: bool
    pack_recall: bool
    reranker_false_negative_loss: bool
    reranker_lift_recall: float
    reranker_lift_mrr: float
    reranker_lift_ndcg: float
    duplicate_rate: float
    baseline_tokens: int
    target_tokens: int
    baseline_latency_ms: float
    target_latency_ms: float
    unsupported_clean: bool


@dataclass(frozen=True)
class V2Summary:
    """Aggregate baseline-vs-target benchmark summary."""

    benchmark_version: str
    eval_set_version: str
    cases: int
    baseline_recall_at_5: float
    target_recall_at_5: float
    baseline_recall_at_10: float
    target_recall_at_10: float
    baseline_mrr: float
    target_mrr: float
    baseline_ndcg_10: float
    target_ndcg_10: float
    target_coverage: float
    duplicate_rate: float
    mean_reranker_lift_recall: float
    mean_reranker_lift_mrr: float
    mean_reranker_lift_ndcg: float
    mean_query_diversity: float
    mean_duplicate_query_rate: float
    mean_target_tokens: float
    p50_target_latency_ms: float
    p95_target_latency_ms: float
    unsupported_clean_rate: float
    broad_non_regressive: bool
    pool_to_pack_loss_rate: float
    peak_rss_mb: float


def planner_proxy_queries(case: GoldCase, *, count: int = PROXY_QUERY_COUNT) -> list[str]:
    """Build 10..16 deterministic proxy planner queries for one gold case.

    ``queries[0]`` is the direct context-resolved formulation of the
    current turn (utterance plus multi-turn context when the case
    requires it). Remaining queries diversify wording using only the
    case's own utterance, context, paraphrase lists and allowed
    interpretations; oracle relevant-section labels are never consulted
    and no hand-written expansion dictionary is used.
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
        base = cleaned[0] if cleaned else (resolved if resolved.strip() else "вопрос")
        base = " ".join(str(base).split()) or "вопрос"
        suffix = 1
        guard = 0
        while len(cleaned) < count and guard < 10 * count + 50:
            guard += 1
            candidate = f"{base} уточнение {suffix}"
            suffix += 1
            collapsed = " ".join(candidate.split())
            if not collapsed or collapsed.casefold() in seen:
                continue
            seen.add(collapsed.casefold())
            cleaned.append(collapsed)
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
    reranker: CrossEncoderReranker,
    config: RetrievalConfig | None = None,
) -> V2CaseResult:
    """Run baseline (RRF-only) vs target (RRF + rerank + expansion) for one case."""
    active = config if config is not None else RetrievalConfig()
    queries = planner_proxy_queries(case)
    diversity = query_diversity(queries)
    dup_rate = duplicate_query_rate(queries)
    # Warm latency: measure the second (warm) execution per arm.
    retrieve_evidence(index, queries, config=active, reranker=reranker)
    started = time.perf_counter()
    target_pack = retrieve_evidence(index, queries, config=active, reranker=reranker)
    target_ms = (time.perf_counter() - started) * 1000.0
    from aa.retrieval.reranker import CrossEncoderReranker as _Reranker
    from aa.retrieval.reranker import RerankerError as _RerankerError

    class _IdentityReranker(_Reranker):
        """RRF-only baseline: preserve fused order instead of reranking."""

        def score(self, query: str, texts: list[str]) -> list[float]:
            if not query.strip():
                raise _RerankerError("reranker query must be non-empty")
            # Descending scores keep the fused candidate order intact.
            return [float(len(texts) - pos) for pos in range(len(texts))]

    baseline_reranker = _IdentityReranker(lock=dict(reranker.lock), backend="rrf-only/1")
    retrieve_evidence(index, queries, config=active, reranker=baseline_reranker)
    started = time.perf_counter()
    baseline_pack = retrieve_evidence(index, queries, config=active, reranker=baseline_reranker)
    baseline_ms = (time.perf_counter() - started) * 1000.0
    baseline_sections = _pack_sections(baseline_pack, 10)
    target_sections = _pack_sections(target_pack, 10)
    baseline_r5 = _recall_at(baseline_sections, case, 5)
    target_r5 = _recall_at(target_sections, case, 5)
    baseline_r10 = _recall_at(baseline_sections, case, 10)
    target_r10 = _recall_at(target_sections, case, 10)
    baseline_mrr_v = _mrr(baseline_sections, case)
    target_mrr_v = _mrr(target_sections, case)
    baseline_ndcg = _ndcg(baseline_sections, case)
    target_ndcg = _ndcg(target_sections, case)
    # Stage diagnostics for the target arm: union of all planner-query
    # branches, RRF pool truncation, and final pack recall. The pool is
    # recomputed deterministically from the same branch outputs (no
    # rerank involved), so union -> pool -> pack losses are attributable.
    union_sections: set[str] = set()
    for query in queries:
        for chunk_id, _ in _branch_union(index, query, top_k=active.branch_top_k):
            record = index.chunks.get(chunk_id)
            if record is not None:
                union_sections.add(record.section)
    union_hit = (
        any(section in case.relevant_sections for section in union_sections)
        if not case.is_unsupported
        else False
    )
    from aa.retrieval import evidence as _evidence

    _ranked, _per_query = _evidence.run_branch_searches(
        index, queries, branch_top_k=active.branch_top_k
    )
    _fused, _pool_ids = _evidence.fuse_query_pool(
        _ranked, _per_query, rrf_k=active.rrf_k, pool_cap=active.reranker_pool_cap
    )
    _diverse = _evidence.dedup_and_diversify(
        index,
        _pool_ids,
        _fused,
        pool_cap=active.reranker_pool_cap,
        max_per_section=active.max_per_section,
    )
    pool_sections = tuple(dict.fromkeys(index.chunks[c.chunk_id].section for c in _diverse))
    pool_hit = _recall_at(pool_sections, case, len(pool_sections))
    pack_hit = target_r10
    unsupported_clean = (not target_pack.passages) if case.is_unsupported else True
    return V2CaseResult(
        case_id=case.case_id,
        category=case.category,
        is_unsupported=case.is_unsupported,
        proxy_queries=tuple(queries),
        query_diversity=diversity,
        duplicate_query_rate=dup_rate,
        baseline_sections_10=baseline_sections,
        target_sections_10=target_sections,
        baseline_recall_at_5=baseline_r5,
        target_recall_at_5=target_r5,
        baseline_recall_at_10=baseline_r10,
        target_recall_at_10=target_r10,
        baseline_mrr=baseline_mrr_v,
        target_mrr=target_mrr_v,
        baseline_ndcg_10=baseline_ndcg,
        target_ndcg_10=target_ndcg,
        baseline_coverage=_coverage(baseline_sections, case),
        target_coverage=_coverage(target_sections, case),
        union_recall=union_hit,
        pool_recall=pool_hit,
        pack_recall=pack_hit,
        reranker_false_negative_loss=bool(pool_hit and not pack_hit),
        reranker_lift_recall=float(target_r5) - float(baseline_r5),
        reranker_lift_mrr=target_mrr_v - baseline_mrr_v,
        reranker_lift_ndcg=target_ndcg - baseline_ndcg,
        duplicate_rate=_duplicate_rate(target_pack),
        baseline_tokens=baseline_pack.total_tokens,
        target_tokens=target_pack.total_tokens,
        baseline_latency_ms=baseline_ms,
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
    """Aggregate per-case baseline-vs-target results."""
    if not results:
        raise ValueError("v2 benchmark has no case results")
    supported = [item for item in results if not item.is_unsupported]
    denom = max(1, len(supported))
    broad = [item for item in supported if item.category in BROAD_CATEGORIES]
    broad_ok = all(item.target_recall_at_5 >= item.baseline_recall_at_5 for item in broad)
    latencies = [item.target_latency_ms for item in results]
    return V2Summary(
        benchmark_version=V2_BENCHMARK_VERSION,
        eval_set_version=V2_EVAL_SET_VERSION,
        cases=len(results),
        baseline_recall_at_5=sum(1 for r in supported if r.baseline_recall_at_5) / denom,
        target_recall_at_5=sum(1 for r in supported if r.target_recall_at_5) / denom,
        baseline_recall_at_10=sum(1 for r in supported if r.baseline_recall_at_10) / denom,
        target_recall_at_10=sum(1 for r in supported if r.target_recall_at_10) / denom,
        baseline_mrr=sum(r.baseline_mrr for r in supported) / denom,
        target_mrr=sum(r.target_mrr for r in supported) / denom,
        baseline_ndcg_10=sum(r.baseline_ndcg_10 for r in supported) / denom,
        target_ndcg_10=sum(r.target_ndcg_10 for r in supported) / denom,
        target_coverage=sum(r.target_coverage for r in supported) / denom,
        duplicate_rate=sum(r.duplicate_rate for r in results) / len(results),
        mean_reranker_lift_recall=sum(r.reranker_lift_recall for r in supported) / denom,
        mean_reranker_lift_mrr=sum(r.reranker_lift_mrr for r in supported) / denom,
        mean_reranker_lift_ndcg=sum(r.reranker_lift_ndcg for r in supported) / denom,
        mean_query_diversity=sum(r.query_diversity for r in results) / len(results),
        mean_duplicate_query_rate=sum(r.duplicate_query_rate for r in results) / len(results),
        mean_target_tokens=sum(float(r.target_tokens) for r in results) / len(results),
        p50_target_latency_ms=_percentile(latencies, 50),
        p95_target_latency_ms=_percentile(latencies, 95),
        unsupported_clean_rate=sum(1 for r in results if r.unsupported_clean) / len(results),
        broad_non_regressive=broad_ok,
        pool_to_pack_loss_rate=sum(1 for r in supported if r.reranker_false_negative_loss) / denom,
        peak_rss_mb=_peak_rss_mb(),
    )


def v2_quality_gate(summary: V2Summary) -> tuple[bool, list[str]]:
    """Evaluate the #116 benchmark gate (fails closed, never drops the reranker).

    The gate mirrors the issue: no regression in the qualified recall
    gate, measurable reranker benefit over RRF-only, non-regressive
    broad/colloquial/multi-turn cases, warm p95 latency within the
    interactive Telegram budget, and zero hot-path disk reads
    (verified separately by the harness blocking filesystem access).
    Unsupported-case passage rates are measured and reported but carry
    no abstention gate: #116 defines no grounding abstention mechanism.
    """
    failures: list[str] = []
    if summary.target_recall_at_5 < summary.baseline_recall_at_5:
        failures.append(
            f"target recall@5 {summary.target_recall_at_5:.3f} regressed "
            f"vs baseline {summary.baseline_recall_at_5:.3f}"
        )
    if summary.target_recall_at_10 < summary.baseline_recall_at_10:
        failures.append(
            f"target recall@10 {summary.target_recall_at_10:.3f} regressed "
            f"vs baseline {summary.baseline_recall_at_10:.3f}"
        )
    lift = (
        summary.mean_reranker_lift_recall,
        summary.mean_reranker_lift_mrr,
        summary.mean_reranker_lift_ndcg,
    )
    if not any(value > 0 for value in lift):
        failures.append(
            "reranker shows no measurable benefit over RRF-only "
            f"(recall {lift[0]:+.3f}, mrr {lift[1]:+.3f}, ndcg {lift[2]:+.3f}); "
            "failing rather than removing the mandatory reranker"
        )
    if not summary.broad_non_regressive:
        failures.append("broad/colloquial/multi-turn cases regressed vs baseline")
    if summary.duplicate_rate > 0.05:
        failures.append(f"duplicate rate {summary.duplicate_rate:.3f} exceeds 0.05")
    if summary.p95_target_latency_ms > V2_TARGET_P95_LATENCY_BUDGET_MS:
        failures.append(
            f"warm p95 latency {summary.p95_target_latency_ms:.0f}ms exceeds "
            f"the interactive budget {V2_TARGET_P95_LATENCY_BUDGET_MS:.0f}ms "
            "(per-turn RRF plus 64-candidate CPU BGE rerank); "
            "explicit performance acceptance or optimization is required "
            "before cutover"
        )
    return (not failures, failures)


def v2_production_status(summary: V2Summary) -> dict[str, Any]:
    """Return the machine-readable v2 production promotion status.

    Mirrors the ``production_promotion_blocked`` pattern from the
    qualified retrieval artifact: any quality-gate failure (currently
    the warm-p95 interactive-latency breach) keeps the v2 pipeline
    blocked behind the legacy production path. Cutover also stays
    blocked when the gate passes because #116 explicitly leaves cutover
    out of scope; a later cutover task must re-evaluate the gate with
    explicit performance acceptance or optimization.
    """
    passed, failures = v2_quality_gate(summary)
    blocked = True
    config_id = "blocked"
    reason = V2_CUTOVER_BLOCKED_REASON
    if passed:
        reason = (
            "v2 quality gate passed but production cutover remains out of "
            "scope for #116; a later cutover task must re-evaluate with "
            "explicit performance acceptance"
        )
    return {
        "config_id": config_id,
        "config_version": V2_PRODUCTION_CONFIG_VERSION,
        "cutover_allowed": not blocked,
        "production_promotion_blocked": blocked,
        "quality_gate_passed": passed,
        "quality_gate_failures": list(failures),
        "reason": reason,
    }


def require_v2_cutover_acceptance(*, performance_accepted: bool) -> None:
    """Fail closed unless explicit v2 performance acceptance is recorded.

    The frozen BGE validation exceeds the interactive budget by ~4x, so
    any production-cutover caller must pass ``performance_accepted=True``
    after documenting acceptance or landing an optimization. The default
    (``False``) raises instead of silently promoting the slow path.
    """
    if not performance_accepted:
        raise ValueError(V2_CUTOVER_BLOCKED_REASON)


def find_repo_root() -> Path:
    """Return the repository root containing qualification + corpus."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "corpus" / "reranker.lock.json").is_file():
            return parent
    raise ValueError("repository root with reranker lock not found")


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
    "require_v2_cutover_acceptance",
    "run_v2_case",
    "summarize_v2",
    "v2_production_status",
    "v2_quality_gate",
]
