"""V2 retrieval node: planner queries to compact Evidence Pack (issues #116, #295).

The node consumes the ``search_queries`` produced by the hidden
planner (0 or 1..16 context-resolved useful Russian queries) and runs
the RRF-only target pipeline from :mod:`aa.retrieval.evidence` over the
#115 RAM-resident canonical index:

- ``search_queries == []`` performs no retrieval and yields an empty
  pack for a purely conversational/glue turn;
- otherwise every query runs BM25 + E5/FAISS branches, global RRF,
  dedup/diversity, small-to-big expansion and atomic budget selection.

Only orchestration state is written; ``messages`` is left untouched.
State carries exact passage text plus minimal provenance. Ranking
metadata (RRF/BM25/dense scores, embeddings, planner reasoning,
search previews) stays in internal retrieval metadata and never enters
the user-facing prompt. Logs carry only routes, counts and token
lengths, never prompts or user text.

The hot path is RAM-only with zero network/download dependency and no
second-stage reranker: ``QueryPlan -> BM25+E5 -> RRF -> dedup ->
expansion -> pack``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from typing import Any

from aa.conversation.graph_state import TurnState
from aa.conversation.prompt_builder import EvidencePassage
from aa.opencode.errors import OpenCodeRateLimitError
from aa.retrieval.evidence import (
    INTERACTIVE_LATENCY_BUDGET_MS,
    EvidencePack,
    RetrievalConfig,
)
from aa.retrieval.index import HybridIndex, logical_chunk_id

logger = logging.getLogger("aa.conversation.retrieval_node")

# Bounded per-model-call slice of the interactive retrieval budget
# (kodmial/aa#331): one slow selector/coverage provider call must not
# consume the whole turn. Each call is bounded by the smaller of this
# cap and the remaining interactive budget; expiry yields a typed
# ``exhausted/latency-budget`` outcome, never a retry loop or fake
# coverage. Reuses the existing selection-attempt scale (8s), not a new
# product threshold.
SELECTION_PER_CALL_BUDGET_S = 8.0
COVERAGE_PER_CALL_BUDGET_S = 8.0

# Bounded canonical read cache (kodmial/aa#331): exact expanded
# passages keyed by stable child id + neighbor window. Content is
# content-addressed (checksum validated on hit); the cache only avoids
# re-expanding the same canonical range within/between turns. Bounded
# FIFO so memory stays flat; keys never carry user text.
_READ_CACHE_MAX = 512
_EXPANDED_READ_CACHE: dict[tuple[str, int], Any] = {}


class _InteractiveBudgetTimeout(TimeoutError):
    """One bounded model/retriever await exceeded the interactive deadline."""


def _remaining_interactive_ms(started: float) -> float:
    """Remaining interactive retrieval budget in milliseconds."""
    try:
        from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS as _budget

        return float(_budget) - (time.perf_counter() - started) * 1000.0
    except Exception:
        return 0.0


async def _await_under_interactive_deadline(
    coro_factory: Any, started: float, *, per_call_cap_s: float
) -> Any:
    """Await one provider/retriever coroutine under the interactive deadline.

    ``coro_factory`` is a zero-argument callable producing the awaitable,
    invoked only after the budget check so an already-spent budget never
    creates an un-awaited coroutine. The timeout is the smaller of
    ``per_call_cap_s`` and the remaining interactive budget, so a slow
    provider tail fails fast to a typed exhausted outcome instead of
    grinding sequential calls. The underlying await is cancelled on
    expiry so no provider work continues after the turn is irrecoverably
    timed out. ``Cancelled`` and provider 429 always propagate for runner
    retire/resume and are never converted to exhaustion.
    """
    remaining_ms = _remaining_interactive_ms(started)
    if remaining_ms <= 0:
        raise _InteractiveBudgetTimeout("interactive budget already spent")
    timeout_s = min(float(per_call_cap_s), remaining_ms / 1000.0)
    if timeout_s <= 0:
        raise _InteractiveBudgetTimeout("interactive budget already spent")
    coro = coro_factory() if callable(coro_factory) else coro_factory
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError as exc:
        raise _InteractiveBudgetTimeout("interactive deadline exceeded") from exc


def _preview_identity(previews: Any) -> str:
    """Stable identity for one preview set (candidates + ranks + need seam).

    Covers chunk ids, fused ranks and the trusted need seam only. The
    planner query ids (``q1..qN`` positions) are deliberately excluded:
    re-attributing the same candidates to a new query id without any
    new candidate, rank change, or need association is not progress and
    must not justify another sequential selector call. A genuinely new
    deep-ranked candidate, a re-rank, or a changed need seam changes the
    identity and continues.
    """
    try:
        parts: list[str] = []
        for preview in list(previews or []):
            cid = str(getattr(preview, "chunk_id", "") or "")
            rank = str(getattr(preview, "fused_rank", "") or "")
            nids = ",".join(sorted(str(n) for n in list(getattr(preview, "need_ids", ()) or ())))
            if cid:
                parts.append(f"{cid}@{rank}[{nids}]")
        parts.sort()
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _cached_expand_small_to_big(index: Any, winners: Any, *, neighbor_window: int) -> list[Any]:
    """Expand winners reusing validated cached canonical ranges where safe.

    Cache key is the stable child chunk id plus the neighbor window; a
    hit is returned only after the cached exact text still checksums
    against the live index record, so a changed corpus never serves a
    stale range. Misses delegate to the canonical expansion and are
    stored under a bounded FIFO. Provenance and exact text stay
    canonical; this only avoids recomputing the same range twice.
    """
    from aa.retrieval.evidence import expand_small_to_big as _expand

    window = int(neighbor_window)
    winners_list = list(winners or [])
    if not winners_list:
        return []
    uncached: list[Any] = []
    uncached_pos: list[int] = []
    out: list[Any | None] = [None] * len(winners_list)
    hits = 0
    for pos, winner in enumerate(winners_list):
        try:
            cid = str(getattr(winner, "chunk_id", "") or "")
        except Exception:
            cid = ""
        if not cid:
            uncached.append(winner)
            uncached_pos.append(pos)
            continue
        key = (cid, window)
        cached = _EXPANDED_READ_CACHE.get(key)
        if cached is not None:
            try:
                record = index.chunks.get(cid)
                if record is not None and str(getattr(record, "text", "")) in str(
                    getattr(cached, "exact_text", "")
                ):
                    out[pos] = cached
                    hits += 1
                    continue
            except Exception:
                pass
        uncached.append(winner)
        uncached_pos.append(pos)
    _ = hits
    if uncached:
        fresh = _expand(index, uncached, neighbor_window=window)
        fresh_by_child: dict[str, list[Any]] = {}
        for passage in fresh:
            try:
                for cid in list(getattr(passage, "child_chunk_ids", ()) or ()):
                    fresh_by_child.setdefault(str(cid), []).append(passage)
            except Exception:
                continue
        for winner, pos in zip(uncached, uncached_pos, strict=True):
            try:
                cid = str(getattr(winner, "chunk_id", "") or "")
            except Exception:
                cid = ""
            options = fresh_by_child.get(cid, [])
            chosen = options[0] if options else None
            if chosen is None and fresh:
                chosen = fresh[0]
            out[pos] = chosen
            if chosen is not None and cid:
                try:
                    key = (cid, window)
                    if key not in _EXPANDED_READ_CACHE:
                        while len(_EXPANDED_READ_CACHE) >= _READ_CACHE_MAX:
                            _EXPANDED_READ_CACHE.pop(next(iter(_EXPANDED_READ_CACHE)))
                        _EXPANDED_READ_CACHE[key] = chosen
                except Exception:
                    pass
    merged: dict[str, Any] = {}
    order: list[str] = []
    for item in list(out):
        if item is None:
            continue
        try:
            pid = str(getattr(item, "passage_id", "") or "")
        except Exception:
            continue
        if pid and pid not in merged:
            merged[pid] = item
            order.append(pid)
    # Preserve canonical expansion order for genuinely new ranges while
    # reusing cached identities above; fall back to fresh order when the
    # cache path cannot reconstruct it.
    try:
        fresh_ids = [str(getattr(p, "passage_id", "") or "") for p in fresh]  # noqa: F821
        ordered = [merged[pid] for pid in fresh_ids if pid in merged]
        for pid in order:
            if pid not in fresh_ids:
                ordered.append(merged[pid])
        return ordered
    except Exception:
        return [merged[pid] for pid in order]


def clear_canonical_read_cache() -> None:
    """Drop cached canonical expansions (tests/tooling only)."""
    _EXPANDED_READ_CACHE.clear()


def _provenance_lists(
    child_ids: list[str],
    *,
    child_query_map: dict[str, list[str]] | None = None,
    child_need_map: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Union stable query/need ids for one passage's children (#311)."""
    queries: list[str] = []
    needs: list[str] = []
    seen_q: set[str] = set()
    seen_n: set[str] = set()
    for cid in child_ids:
        for qid in list((child_query_map or {}).get(cid, []) or []):
            clean = str(qid or "").strip()
            if clean and clean not in seen_q:
                seen_q.add(clean)
                queries.append(clean)
        for nid in list((child_need_map or {}).get(cid, []) or []):
            clean = str(nid or "").strip()
            if clean and clean not in seen_n:
                seen_n.add(clean)
                needs.append(clean)
    queries.sort(key=lambda q: int(q[1:]) if len(q) > 1 and q[1:].isdigit() else 0)
    needs.sort()
    return queries, needs


def pack_to_state(
    pack: EvidencePack,
    *,
    child_query_map: dict[str, list[str]] | None = None,
    child_need_map: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map an Evidence Pack to graph state payloads (exact text + provenance)."""
    hits: list[dict[str, Any]] = []
    pack_dicts: list[dict[str, Any]] = []
    for passage in pack.passages:
        child_ids = list(passage.child_chunk_ids)
        query_ids, need_ids = _provenance_lists(
            child_ids, child_query_map=child_query_map, child_need_map=child_need_map
        )
        entry: dict[str, Any] = {
            "passage_id": passage.passage_id,
            "text": passage.exact_text,
            "source_id": passage.source_id,
            "section_id": passage.section_id,
            "child_chunk_ids": child_ids,
            "char_start": passage.char_start,
            "char_end": passage.char_end,
            "text_sha256": passage.text_sha256,
            "source_sha256": passage.source_sha256,
            "corpus_version": pack.corpus_version,
        }
        if query_ids:
            entry["query_ids"] = query_ids
        if need_ids:
            entry["need_ids"] = need_ids
        pack_dicts.append(entry)
        for chunk_id in passage.child_chunk_ids:
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "logical_chunk_id": logical_chunk_id(chunk_id),
                    "section_id": passage.section_id,
                    "source_id": passage.source_id,
                    "passage_id": passage.passage_id,
                    "char_start": passage.char_start,
                    "char_end": passage.char_end,
                    "text_sha256": passage.text_sha256,
                    "source_sha256": passage.source_sha256,
                    "corpus_version": pack.corpus_version,
                }
            )
    return hits, pack_dicts


def state_passages_to_prompt(pack_dicts: list[dict[str, Any]]) -> list[EvidencePassage]:
    """Map stored state passages back to prompt-builder passages."""
    passages: list[EvidencePassage] = []
    for item in pack_dicts:
        text = item.get("text")
        passage_id = item.get("passage_id")
        source_id = item.get("source_id")
        section_id = item.get("section_id")
        if not isinstance(text, str) or not text:
            continue
        if not isinstance(passage_id, str) or not passage_id:
            continue
        # Never coerce a missing id to the literal "None": skip passages
        # without real grounding provenance instead of misattributing them.
        if not isinstance(source_id, str) or not source_id:
            continue
        if not isinstance(section_id, str) or not section_id:
            continue
        passages.append(
            EvidencePassage(
                passage_id=passage_id,
                source=source_id,
                section=section_id,
                text=text,
            )
        )
    return passages


def _selection_context_from_state(state: TurnState) -> tuple[str, str, str]:
    """Extract resolved intent, conversation context and user message.

    Consumes the single canonical ``ResolvedTurn``/``ConversationContext``
    when present so the selector proves the same bytes and digest as the
    planner, generator and verifier. Legacy ad-hoc assembly remains only
    for direct calls without canonical state. Callers never log the
    returned text.
    """
    try:
        from aa.conversation.conversation_context import assert_same_context_digest as _assert
        from aa.conversation.conversation_context import canonical_model_view as _view
        from aa.conversation.conversation_context import conversation_context_from_state as _cc
        from aa.conversation.conversation_context import resolved_turn_from_state as _rt

        _assert(state, stage="selector")
        resolved = _rt(state)
        canonical = _cc(state)
        if resolved is not None or canonical is not None:
            view_source = (
                resolved.conversation_context
                if resolved is not None and resolved.conversation_context
                else (canonical.model_dump(mode="json") if canonical is not None else {})
            )
            view = _view(view_source) if view_source else {"combined": "", "user_message": ""}
            intent = ""
            if resolved is not None and str(resolved.resolved_intent or "").strip():
                intent = str(resolved.resolved_intent)
            else:
                try:
                    retry = state.get("retry_state", {})
                    retry_d = dict(retry) if isinstance(retry, dict) else {}
                    intent = str(
                        retry_d.get("resolved_intent", state.get("resolved_intent", "")) or ""
                    )
                except Exception:
                    intent = str(state.get("resolved_intent", "") or "")
            if not intent.strip():
                intent = str(state.get("current_user_message", "") or "")
            user_message = ""
            if resolved is not None and str(resolved.user_message or "").strip():
                user_message = str(resolved.user_message)
            else:
                user_message = str(state.get("current_user_message", "") or "")
            context = str(view.get("combined", "") or "")
            return intent, context, user_message
    except ValueError:
        raise
    except Exception:
        pass
    try:
        retry = state.get("retry_state", {})
        retry_d = dict(retry) if isinstance(retry, dict) else {}
        intent = str(retry_d.get("resolved_intent", state.get("resolved_intent", "")) or "")
    except Exception:
        intent = str(state.get("resolved_intent", "") or "")
    if not intent.strip():
        intent = str(state.get("current_user_message", "") or "")
    user_message = str(state.get("current_user_message", "") or "")
    try:
        summary = str(state.get("conversation_summary", "") or "")
    except Exception:
        summary = ""
    try:
        from langchain_core.messages import BaseMessage as _BM

        from aa.conversation.conversation_context import truncate_preserving_tail as _tail

        recent_texts: list[str] = []
        for item in list(state.get("messages", []) or [])[-6:]:
            if isinstance(item, _BM):
                content = getattr(item, "content", "")
                if isinstance(content, str) and content.strip():
                    recent_texts.append(_tail(content.strip(), 1200))
        context = _tail(" ".join([summary.strip(), *recent_texts[-6:]]).strip(), 4000)
    except Exception:
        context = str(summary or "")[:4000]
    return intent, context, user_message


def selection_context_digest(state: TurnState) -> str:
    """Return the canonical digest the selector must prove (empty if legacy)."""
    try:
        from aa.conversation.conversation_context import context_digest_from_state as _digest

        return str(_digest(state) or "")
    except Exception:
        return ""


async def _coerce_needs_for_selection(
    information_needs: Any | None,
) -> list[Any]:
    """Normalize information needs to a list of models/dicts (no fabrication)."""
    if not information_needs:
        return []
    try:
        from aa.conversation.conversation_context import InformationNeed as _Need
    except Exception:
        return []
    out: list[Any] = []
    for entry in list(information_needs):
        try:
            if isinstance(entry, _Need):
                if str(entry.need_id or "").strip() and str(entry.text or "").strip():
                    out.append(entry)
            elif isinstance(entry, dict) and str(entry.get("need_id", "") or "").strip():
                validated = _Need.model_validate(entry)
                if str(validated.text or "").strip():
                    out.append(validated)
        except Exception:
            continue
    return out


async def aretrieve_with_semantic_selection(
    index: HybridIndex,
    queries: list[str],
    *,
    config: RetrievalConfig | None = None,
    resolved_intent: str = "",
    conversation_context: str = "",
    user_message: str = "",
    selection_model: Any | None = None,
    context_digest: str = "",
    information_needs: Any | None = None,
    query_need_map: Any | None = None,
    coverage_model: Any | None = None,
    repair_hint: str = "",
    max_iterations: int = 3,
) -> EvidencePack:
    """Run broad hybrid retrieval as one read/coverage loop (#306).

    Issue #295: the genuinely broad BM25+E5/RRF pool (including fused rank
    >16) is exposed as discovery previews; an LLM selects pertinent
    candidates against the real context-resolved intent; exact complete
    canonical passages (+neighbors, provenance preserved) are fetched only
    for the validated selection.

    Issue #311: previews preserve per-query and per-information-need
    exposure through a bounded need-aware strategy. Provenance
    ``candidate -> query_ids -> need_ids`` survives fusion, previews,
    selection, follow-ups and state serialization.

    Issue #306: preview selection decides **what to read**, never whether
    the book suffices. Every iteration runs ``select_for_reading`` over
    the combined discovered pool, ``read_exact`` of full canonical
    passages, and ``assess_coverage`` of every information need against
    the full exact text (conditions/qualifiers preserved, supporting
    spans verified as substrings). Uncovered needs trigger ``expand``
    around already-read ranges or ``search_more`` followed by fresh
    semantic selection (never rank-only appends). Progress is semantic
    (newly read relevant context or a newly closed need); novel
    irrelevant ids alone never count. The loop is bounded with
    repeated-state detection; exhaustion keeps the best-effort pack
    with typed ``coverage_status=exhausted`` and uncovered needs and
    never converts insufficient evidence into a supported answer.
    Lexical overlap guides discovery only and never marks sufficiency.
    """
    from aa.conversation.coverage_loop import (
        MAX_COVERAGE_EXPANSIONS,
        MAX_COVERAGE_SEARCHES,
        STATE_ASSESS_COVERAGE,
        STATE_READ_EXACT,
        STATE_SELECT_FOR_READING,
        aassess_coverage,
        conservative_uncovered_verdict,
        expand_read_ranges,
        loop_fingerprint,
        sanitize_followup_queries,
    )
    from aa.retrieval.evidence import (
        candidate_need_provenance,
        candidate_query_provenance,
        fuse_query_pool,
        run_branch_searches,
        select_passages_under_budget,
        select_top_candidates,
        validate_planner_queries,
        validate_recovery_queries,
    )

    active = config if config is not None else RetrievalConfig()
    cleaned = validate_planner_queries(queries)
    if not cleaned:
        from aa.retrieval.evidence import empty_evidence_pack

        corpus = str(index.metadata.get("ru_artifact_sha256", ""))
        return empty_evidence_pack(corpus_version=corpus)
    needs_for_selection = await _coerce_needs_for_selection(information_needs)
    try:
        from aa.conversation.conversation_context import build_query_need_map as _build_map
        from aa.conversation.conversation_context import validate_query_need_map as _validate_map

        if query_need_map is not None:
            try:
                query_map_models = _validate_map(query_need_map)
            except Exception:
                query_map_models = _build_map(cleaned, needs_for_selection)
        else:
            query_map_models = _build_map(cleaned, needs_for_selection)
    except Exception:
        query_map_models = []
    query_map_dicts: list[dict[str, Any]] = []
    for entry in list(query_map_models or []):
        try:
            query_map_dicts.append(
                {"query_id": str(entry.query_id), "need_ids": list(entry.need_ids or [])}
            )
        except Exception:
            continue
    # The original request stays primary and isolated: repair/safety
    # control text travels only as a supplementary hint to coverage
    # assessment, never as intent or as a search query.
    original_request = " ".join(str(resolved_intent or "").split()).strip()
    if not original_request:
        original_request = " ".join(str(user_message or "").split()).strip()
    repair_hint_clean = " ".join(str(repair_hint or "").split()).strip()
    coverage_assessor = coverage_model if coverage_model is not None else selection_model
    try:
        need_id_order = [
            str(getattr(item, "need_id", "") or "").strip()
            for item in list(needs_for_selection or [])
            if str(getattr(item, "need_id", "") or "").strip()
        ]
    except Exception:
        need_id_order = []
    if not need_id_order:
        try:
            need_id_order = [
                str(item.get("need_id", "") or "").strip()
                for item in list(needs_for_selection or [])
                if isinstance(item, dict) and str(item.get("need_id", "") or "").strip()
            ]
        except Exception:
            need_id_order = []

    from aa.conversation.semantic_selection import (
        MAX_FOLLOWUP_QUERIES,
        MAX_SELECTED_CHUNKS,
        MAX_SELECTION_CANDIDATES,
        aselect_semantic_candidates,
        assess_need_preview_coverage,
        select_need_aware_previews,
        serialize_preview_coverage,
    )

    started = time.perf_counter()
    # ---- discover (initial): broad hybrid pool, no truncation before selection.
    ranked_lists, per_query_ids = await asyncio.to_thread(
        run_branch_searches, index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists, per_query_ids, rrf_k=active.rrf_k, pool_cap=active.pool_cap
    )
    candidate_query_map = candidate_query_provenance(per_query_ids)
    candidate_need_map = candidate_need_provenance(candidate_query_map, query_map_dicts)
    all_queries: list[str] = list(cleaned)
    sent_query_fingerprints: set[str] = set()
    for query_text in all_queries:
        sent_query_fingerprints.add(
            __import__("hashlib")
            .sha256(" ".join(query_text.split()).casefold().encode("utf-8"))
            .hexdigest()[:16]
        )

    # ---- bounded read/coverage loop state.
    read_child_set: set[str] = set()
    read_child_order: list[str] = []
    cumulative_expanded: dict[str, Any] = {}
    cumulative_expanded_order: list[str] = []
    covered_need_ids: set[str] = set()
    cited_passage_ids: set[str] = set()
    progress_events: list[str] = []
    fingerprints: list[str] = []
    fingerprint_set: set[str] = set()
    preview_identities_seen: set[str] = set()
    deadline_stops = 0
    preview_map_merged: dict[str, dict[str, Any]] = {}
    discovered_first: list[str] = []
    try:
        broad_sorted_first = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
        discovered_first = [item.chunk_id for item in broad_sorted_first if item.chunk_id in fused]
    except Exception:
        discovered_first = []
    last_previews: list[Any] = []
    last_selection: Any = None
    last_preview_statuses: list[Any] = []
    last_verdict: Any = None
    sel_ms_total = 0.0
    model_calls = 0
    expansions = 0
    searches = 0
    followup_added_total = 0
    followup_need_ids: list[str] = []
    coverage_status = "exhausted"
    exhaustion_reason = "no-iterations"
    read_ids: list[str] = []
    previewed_ids_all: list[str] = []
    previewed_set_all: set[str] = set()
    selected: list[Any] = []
    total = 0
    expanded: list[Any] = []

    bounded_iterations = max(1, min(int(max_iterations or 1), 5))
    forbidden_echoes = [repair_hint_clean] if repair_hint_clean else []

    def _interactive_budget_exceeded() -> bool:
        """Whether the retrieval loop already spent the interactive budget.

        Gate E root repair: the shared read/coverage loop previously ran
        up to three iterations (≈8 model calls) regardless of wall-clock
        cost, producing the observed retrieval p95 ≈76s on the slow free
        provider. The first discover/select/read/assess pair always runs
        (model-driven semantic coverage preserved); further
        expand/search_more iterations stop once the interactive budget
        is spent and return a typed ``exhausted`` outcome with the
        best-effort pack instead of grinding. No fake coverage, no
        truncation, no skipped semantic checks on the fast path.
        """
        try:
            from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS as _budget

            return (time.perf_counter() - started) * 1000.0 > float(_budget)
        except Exception:
            return False

    for iteration in range(bounded_iterations):
        if iteration > 0 and _interactive_budget_exceeded():
            coverage_status = "exhausted"
            exhaustion_reason = "latency-budget"
            progress_events.append("interactive-budget-stop")
            break
        # ---- select_for_reading over the combined discovered pool.
        broad_sorted = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
        fused_ordered = [(item.chunk_id, item.fused_score) for item in broad_sorted]
        in_index = [cid for cid, _ in fused_ordered if cid in index.chunks]
        texts = {cid: getattr(index.chunks[cid], "text", "") for cid in in_index}
        sections = {cid: getattr(index.chunks[cid], "section", "") for cid in in_index}
        sources = {cid: getattr(index.chunks[cid], "source_id", "") for cid in in_index}
        previews = select_need_aware_previews(
            fused_ordered=fused_ordered,
            texts=texts,
            sections=sections,
            sources=sources,
            candidate_query_map=candidate_query_map,
            candidate_need_map=candidate_need_map,
            information_needs=needs_for_selection,
            limit=MAX_SELECTION_CANDIDATES,
        )
        last_previews = list(previews)
        for seen_preview in previews:
            if seen_preview.chunk_id not in previewed_set_all:
                previewed_set_all.add(seen_preview.chunk_id)
                previewed_ids_all.append(seen_preview.chunk_id)
        try:
            preview_statuses, _unmapped = assess_need_preview_coverage(
                previews, needs_for_selection
            )
        except Exception:
            preview_statuses = []
        last_preview_statuses = list(preview_statuses)
        # Bounded progression (#331): an identical preview set (same
        # candidate ids, ranks and need seam) cannot produce new reading
        # progress. Issuing another selector model call over byte-identical
        # previews is demonstrably redundant sequential work, so stop
        # without a call; a new deep-ranked candidate or a changed seam
        # changes the identity and continues. Never a fixed-count cap.
        preview_identity = _preview_identity(previews)
        if preview_identity and preview_identity in preview_identities_seen:
            coverage_status = "exhausted"
            exhaustion_reason = "repeated-state"
            progress_events.append("preview-unchanged-skip")
            break
        if preview_identity:
            preview_identities_seen.add(preview_identity)
        sel_started = time.perf_counter()
        known_chunk_ids = {cid for cid in fused if cid in index.chunks}
        known_need_ids = {s.need_id for s in preview_statuses} or None
        try:
            selection = await _await_under_interactive_deadline(
                functools.partial(
                    aselect_semantic_candidates,
                    previews,
                    resolved_intent=original_request,
                    conversation_context=conversation_context,
                    user_message=user_message,
                    model=selection_model,
                    known_chunk_ids=known_chunk_ids,
                    context_digest=context_digest,
                    information_needs=needs_for_selection,
                    known_need_ids=known_need_ids,
                ),
                started,
                per_call_cap_s=SELECTION_PER_CALL_BUDGET_S,
            )
        except _InteractiveBudgetTimeout:
            coverage_status = "exhausted"
            exhaustion_reason = "latency-budget"
            progress_events.append("selection-deadline-stop")
            break
        model_calls += 1
        # Conservative close-out (#311): unrepresented/unselected needs
        # report uncovered; only full-read assessment can mark sufficiency.
        try:
            preview_by_id = {p.chunk_id: p for p in previews}
            selected_need_ids: set[str] = set()
            for cid in list(selection.selected_chunk_ids or []):
                preview = preview_by_id.get(cid)
                if preview is not None:
                    for nid in list(preview.need_ids or ()):
                        selected_need_ids.add(str(nid))
            uncovered_set = {str(nid) for nid in list(selection.uncovered_need_ids or [])}
            augmented = False
            for status in preview_statuses:
                if not status.represented or status.need_id not in selected_need_ids:
                    if status.need_id not in uncovered_set:
                        uncovered_set.add(status.need_id)
                        augmented = True
            if augmented:
                used_model = bool(getattr(selection, "_used_model", False))
                selection = type(selection)(
                    selected_chunk_ids=list(selection.selected_chunk_ids or []),
                    need_more_detail=True,
                    followup_queries=list(selection.followup_queries or []),
                    uncovered_need_ids=sorted(uncovered_set),
                )
                try:
                    selection._used_model = used_model
                except Exception:
                    pass
        except Exception:
            pass
        last_selection = selection
        sel_ms_total += (time.perf_counter() - sel_started) * 1000.0
        try:
            for map_entry in serialize_preview_coverage(previews, selection):
                cid = str(map_entry.get("chunk_id", ""))
                if cid and cid not in preview_map_merged:
                    preview_map_merged[cid] = dict(map_entry)
                elif cid:
                    merged_entry = dict(preview_map_merged[cid])
                    if bool(map_entry.get("selected", False)):
                        merged_entry["selected"] = True
                    preview_map_merged[cid] = merged_entry
        except Exception:
            pass
        followup_need_ids = sorted(
            {str(nid) for nid in list(selection.uncovered_need_ids or []) if str(nid).strip()}
        )
        winners = [fused[cid] for cid in selection.selected_chunk_ids if cid in fused]
        explicit_empty_followup = bool(
            getattr(selection, "_used_model", False)
            and not selection.selected_chunk_ids
            and bool(selection.need_more_detail)
        )
        if not winners and not explicit_empty_followup:
            # Fail-safe bounded fallback (never empty silent success).
            # Diversity caps apply here only, never to discovery previews.
            from aa.retrieval.evidence import dedup_and_diversify as _dedup

            _diverse_fallback = _dedup(
                index,
                pool_ids,
                fused,
                pool_cap=active.pool_cap,
                max_per_section=active.max_per_section,
            )
            _ordered_fallback = sorted(
                _diverse_fallback, key=lambda item: item.fused_score, reverse=True
            )
            sections_map: dict[str, str] = {}
            for _item in _ordered_fallback:
                _record = index.chunks.get(_item.chunk_id)
                if _record is not None:
                    sections_map[_item.chunk_id] = str(getattr(_record, "section", "") or "")
            winners = select_top_candidates(
                _ordered_fallback,
                top_cap=min(active.top_child_cap, MAX_SELECTED_CHUNKS),
                sections=sections_map,
            )
        # ---- read_exact: full canonical passages for newly selected ids.
        # Canonical reads reuse the validated read cache where the same
        # stable range was already expanded; a new range or wider window
        # is genuine progress and is read fully.
        new_winners = [w for w in winners if w.chunk_id not in read_child_set]
        if new_winners:
            try:
                newly_expanded = await asyncio.to_thread(
                    _cached_expand_small_to_big,
                    index,
                    new_winners,
                    neighbor_window=active.neighbor_window,
                )
            except Exception:
                from aa.retrieval.evidence import expand_small_to_big as _expand_direct

                newly_expanded = await asyncio.to_thread(
                    _expand_direct, index, new_winners, neighbor_window=active.neighbor_window
                )
            for passage in newly_expanded:
                pid = str(getattr(passage, "passage_id", "") or "")
                if pid and pid not in cumulative_expanded:
                    cumulative_expanded[pid] = passage
                    cumulative_expanded_order.append(pid)
            for w in new_winners:
                if w.chunk_id not in read_child_set:
                    read_child_set.add(w.chunk_id)
                    read_child_order.append(w.chunk_id)
        # Budget the cumulative exact reads (atomic passages; priority
        # winners degrade to exact child chunks rather than truncating).
        ordered_cumulative = [cumulative_expanded[pid] for pid in cumulative_expanded_order]
        if ordered_cumulative:
            selected, total = select_passages_under_budget(
                ordered_cumulative,
                budget_tokens=active.budget_tokens,
                index=index,
                priority_child_ids=tuple(read_child_order),
            )
            expanded = list(ordered_cumulative)
        else:
            selected, total = [], 0
            expanded = []
        read_ids = list(read_child_order)
        # ---- assess_coverage over the full exact budgeted pack.
        # need_more_detail=false never proves sufficiency: only this
        # full-read verdict can mark ready. Bounded by the remaining
        # interactive deadline so a slow coverage tail fails fast to a
        # typed exhausted outcome instead of grinding; 429/cancel
        # propagate, never swallowed.
        if coverage_assessor is not None:
            try:
                verdict = await _await_under_interactive_deadline(
                    functools.partial(
                        aassess_coverage,
                        list(selected),
                        needs_for_selection,
                        original_request=original_request,
                        model=coverage_assessor,
                        repair_hint=repair_hint_clean,
                    ),
                    started,
                    per_call_cap_s=COVERAGE_PER_CALL_BUDGET_S,
                )
            except _InteractiveBudgetTimeout:
                coverage_status = "exhausted"
                exhaustion_reason = "latency-budget"
                progress_events.append("coverage-deadline-stop")
                deadline_stops += 1
                break
            model_calls += 1
        else:
            verdict = conservative_uncovered_verdict(needs_for_selection)
        last_verdict = verdict
        # Semantic progress: newly closed needs or newly cited relevant
        # passages count; novel irrelevant ids alone never do.
        newly_covered = [
            item.need_id
            for item in list(verdict.needs or [])
            if item.covered and item.need_id not in covered_need_ids
        ]
        newly_cited = [
            pid
            for item in list(verdict.needs or [])
            for pid in list(item.passage_ids or [])
            if pid and pid not in cited_passage_ids
        ]
        progressed = False
        if newly_covered:
            progressed = True
            for nid in newly_covered:
                covered_need_ids.add(nid)
                progress_events.append(f"need-covered:{nid}")
        if newly_cited:
            progressed = True
            for pid in newly_cited:
                cited_passage_ids.add(pid)
            progress_events.append(f"cited:{len(newly_cited)}-passages")
        # Repeated action/state detection over semantic state (unmet
        # needs, passages/ranges read, action and candidate identity).
        try:
            read_ranges = [
                f"{str(getattr(cumulative_expanded[pid], 'char_start', 0))}-"
                f"{str(getattr(cumulative_expanded[pid], 'char_end', 0))}"
                for pid in cumulative_expanded_order
            ]
        except Exception:
            read_ranges = []
        try:
            candidate_identity = "|".join(list(selection.selected_chunk_ids or []))
        except Exception:
            candidate_identity = ""
        fingerprint = loop_fingerprint(
            read_passage_ids=sorted(cumulative_expanded_order),
            read_ranges=read_ranges,
            uncovered_need_ids=sorted(str(n) for n in list(verdict.missing_need_ids or [])),
            action=f"{STATE_SELECT_FOR_READING}/{STATE_READ_EXACT}/{STATE_ASSESS_COVERAGE}",
            candidate_identity=candidate_identity,
        )
        fingerprints.append(fingerprint)
        repeated = fingerprint in fingerprint_set
        fingerprint_set.add(fingerprint)
        # Vacuous coverage (no typed needs, model-free assessment) with
        # an explicit selection follow-up request honors a bounded
        # discovery round (#303 follow-up semantics) instead of closing
        # ready; exhaustion below stays typed.
        honor_selection_followup = False
        try:
            _more_requested = bool(getattr(selection, "need_more_detail", False))
            _more_raw = list(getattr(selection, "followup_queries", []) or [])
        except Exception:
            _more_requested, _more_raw = False, []
        if (
            _more_requested
            and _more_raw
            and not need_id_order
            and not verdict.used_model
            and searches < MAX_COVERAGE_SEARCHES
        ):
            _more_sanitized = sanitize_followup_queries(
                _more_raw,
                seen_fingerprints=set(sent_query_fingerprints),
                forbidden_texts=forbidden_echoes,
                max_queries=MAX_FOLLOWUP_QUERIES,
            )
            honor_selection_followup = bool(_more_sanitized)
        coverage_ready = bool(
            verdict.all_covered
            and (verdict.used_model or not need_id_order)
            and not honor_selection_followup
        )
        if coverage_ready:
            coverage_status = "ready"
            exhaustion_reason = ""
            break
        if repeated and not progressed:
            coverage_status = "exhausted"
            exhaustion_reason = "repeated-state"
            break
        if iteration >= bounded_iterations - 1:
            coverage_status = "exhausted"
            exhaustion_reason = "iteration-budget"
            break
        # Decide expand vs search_more for the next round.
        uncovered_now = [str(n) for n in list(verdict.missing_need_ids or [])]
        if not uncovered_now and not honor_selection_followup:
            coverage_status = "ready"
            exhaustion_reason = ""
            break
        expanded_this_round = False
        if expansions < MAX_COVERAGE_EXPANSIONS and read_child_order:
            try:
                grown = await asyncio.to_thread(
                    expand_read_ranges,
                    index,
                    list(read_child_order),
                    neighbor_window=active.neighbor_window,
                    extra_step=1,
                )
            except Exception:
                grown = []
            added = 0
            for passage in grown or []:
                pid = str(getattr(passage, "passage_id", "") or "")
                if pid and pid not in cumulative_expanded:
                    cumulative_expanded[pid] = passage
                    cumulative_expanded_order.append(pid)
                    added += 1
            if added:
                expansions += 1
                expanded_this_round = True
                # Re-budget and re-assess the grown pack before any new
                # search: adjacent context may close coverage with no new ID.
                regrown_ordered = [cumulative_expanded[pid] for pid in cumulative_expanded_order]
                regrown_selected, regrown_total = select_passages_under_budget(
                    regrown_ordered,
                    budget_tokens=active.budget_tokens,
                    index=index,
                    priority_child_ids=tuple(read_child_order),
                )
                if coverage_assessor is not None and not _interactive_budget_exceeded():
                    try:
                        regrown_verdict = await _await_under_interactive_deadline(
                            functools.partial(
                                aassess_coverage,
                                list(regrown_selected),
                                needs_for_selection,
                                original_request=original_request,
                                model=coverage_assessor,
                                repair_hint=repair_hint_clean,
                            ),
                            started,
                            per_call_cap_s=COVERAGE_PER_CALL_BUDGET_S,
                        )
                    except _InteractiveBudgetTimeout:
                        regrown_verdict = conservative_uncovered_verdict(needs_for_selection)
                        progress_events.append("interactive-budget-skip-reassess")
                        deadline_stops += 1
                    else:
                        model_calls += 1
                elif coverage_assessor is not None:
                    # Interactive budget spent: keep the grown exact
                    # passages but do not issue another model call.
                    # Coverage stays conservatively uncovered (typed
                    # exhausted downstream), never fake-covered.
                    regrown_verdict = conservative_uncovered_verdict(needs_for_selection)
                    progress_events.append("interactive-budget-skip-reassess")
                else:
                    regrown_verdict = conservative_uncovered_verdict(needs_for_selection)
                regrown_covered = [
                    item.need_id
                    for item in list(regrown_verdict.needs or [])
                    if item.covered and item.need_id not in covered_need_ids
                ]
                regrown_cited = [
                    pid
                    for item in list(regrown_verdict.needs or [])
                    for pid in list(item.passage_ids or [])
                    if pid and pid not in cited_passage_ids
                ]
                if regrown_covered or regrown_cited:
                    progressed = True
                    for nid in regrown_covered:
                        covered_need_ids.add(nid)
                        progress_events.append(f"expand-need-covered:{nid}")
                    for pid in regrown_cited:
                        cited_passage_ids.add(pid)
                    if regrown_cited:
                        progress_events.append(f"expand-cited:{len(regrown_cited)}-passages")
                selected, total = regrown_selected, regrown_total
                expanded = list(regrown_ordered)
                last_verdict = regrown_verdict
                if regrown_verdict.all_covered and (
                    regrown_verdict.used_model or not need_id_order
                ):
                    coverage_status = "ready"
                    exhaustion_reason = ""
                    break
                # Useful expansion without new IDs still counts as
                # progress: continue rather than stagnating.
                if regrown_covered or regrown_cited:
                    continue
        if expanded_this_round and not progressed:
            # Expansion added ranges but no coverage gain; fall through
            # to search_more below rather than stalling.
            pass
        if searches >= MAX_COVERAGE_SEARCHES or model_calls >= 8:
            coverage_status = "exhausted"
            exhaustion_reason = "search-budget"
            break
        if _interactive_budget_exceeded():
            coverage_status = "exhausted"
            exhaustion_reason = "latency-budget"
            progress_events.append("interactive-budget-stop-search")
            break
        # ---- search_more: bounded follow-up, then semantic
        # reconsideration next iteration (never rank-only append).
        raw_followups = list(selection.followup_queries[:MAX_FOLLOWUP_QUERIES])
        if not raw_followups and uncovered_now:
            # Coverage-driven fallback (#306): the selector was
            # confidently wrong (need_more_detail=false, no follow-ups)
            # yet full-read coverage still misses needs. Derive one
            # bounded follow-up per uncovered need from the typed need
            # texts (never from draft claims or policy wording), marked
            # for deeper detail so the normalized string differs from
            # the already-sent plan.
            for nid in uncovered_now:
                need_text = ""
                try:
                    for entry in list(needs_for_selection or []):
                        if isinstance(entry, dict):
                            cand_id = str(entry.get("need_id", "") or "").strip()
                            cand_text = str(entry.get("text", "") or "")
                        else:
                            cand_id = str(getattr(entry, "need_id", "") or "").strip()
                            cand_text = str(getattr(entry, "text", "") or "")
                        if cand_id == nid and cand_text.strip():
                            need_text = " ".join(cand_text.split())
                            break
                except Exception:
                    need_text = ""
                if not need_text:
                    need_text = original_request
                candidate = " ".join(f"{need_text} подробнее".split()).strip()
                if candidate:
                    raw_followups.append(candidate[:500])
                if len(raw_followups) >= MAX_FOLLOWUP_QUERIES:
                    break
        try:
            validated_followups = validate_recovery_queries(raw_followups)
        except Exception:
            validated_followups = []
        followups = sanitize_followup_queries(
            validated_followups,
            seen_fingerprints=set(sent_query_fingerprints),
            forbidden_texts=forbidden_echoes,
            max_queries=MAX_FOLLOWUP_QUERIES,
        )
        if not followups:
            coverage_status = "exhausted"
            exhaustion_reason = "no-followup-queries" if not progressed else "iteration-budget"
            if progressed and exhaustion_reason == "no-followup-queries":
                exhaustion_reason = "iteration-budget"
            break
        for query_text in followups:
            sent_query_fingerprints.add(
                __import__("hashlib")
                .sha256(" ".join(query_text.split()).casefold().encode("utf-8"))
                .hexdigest()[:16]
            )
            all_queries.append(query_text)
        try:
            remaining_ms = _remaining_interactive_ms(started)
            if remaining_ms <= 0:
                raise _InteractiveBudgetTimeout("interactive budget already spent")
            f_ranked, f_per_ids = await asyncio.wait_for(
                asyncio.to_thread(
                    run_branch_searches, index, followups, branch_top_k=active.branch_top_k
                ),
                timeout=max(0.05, remaining_ms / 1000.0),
            )
            f_fused, _f_pool = fuse_query_pool(
                f_ranked, f_per_ids, rrf_k=active.rrf_k, pool_cap=active.pool_cap
            )
        except _InteractiveBudgetTimeout:
            coverage_status = "exhausted"
            exhaustion_reason = "latency-budget"
            progress_events.append("interactive-budget-stop-search")
            deadline_stops += 1
            break
        except Exception as exc:
            if isinstance(exc, (OpenCodeRateLimitError, asyncio.CancelledError)):
                raise
            coverage_status = "exhausted"
            exhaustion_reason = "search-failed"
            break
        # Merge follow-up discoveries into the combined pool with the
        # same typed query->need seam (attributed to the uncovered needs
        # that triggered them, never via lexical guessing).
        new_query_base = len(query_map_dicts)
        for pos, _query_text in enumerate(followups):
            query_map_dicts.append(
                {"query_id": f"q{new_query_base + pos + 1}", "need_ids": list(uncovered_now)}
            )
        fresh_new = 0
        for cid, cand in f_fused.items():
            if cid not in index.chunks:
                continue
            # Follow-up evidence re-ranks genuinely: a candidate the new
            # search scores higher is promoted monotonically (never
            # demoted), so the next semantic selection reconsiders it on
            # fresh evidence rather than a stale first-round rank. The
            # selector still decides what to read; rank never appends.
            # Follow-up evidence re-ranks genuinely: a candidate the new
            # search scores higher is promoted monotonically (never
            # demoted), so the next semantic selection reconsiders it on
            # fresh evidence rather than a stale first-round rank. The
            # selector still decides what to read; rank never appends.
            previous = fused.get(cid)
            if previous is None:
                fused[cid] = cand
                promoted = True
            elif cand.fused_score > previous.fused_score:
                fused[cid] = cand
                promoted = True
            else:
                promoted = False
            # A follow-up surfaces a candidate with genuinely new or
            # up-ranked evidence that is still unread -- including one
            # discovered earlier but outside the preview window
            # (rank-66), now promoted for semantic reconsideration.
            # Merely re-listing an already-read id never counts.
            if promoted and cid not in read_child_set:
                fresh_new += 1
            # Provenance for newly discovered candidates follows the new
            # query ids through the trusted map.
            new_qids = [f"q{new_query_base + pos + 1}" for pos in range(len(followups))]
            bucket_q = list(candidate_query_map.get(cid, []))
            for qid in new_qids:
                if qid not in bucket_q:
                    bucket_q.append(qid)
            candidate_query_map[cid] = bucket_q
        candidate_need_map = candidate_need_provenance(candidate_query_map, query_map_dicts)
        searches += 1
        if fresh_new:
            followup_added_total += fresh_new
        # Many new IDs with no improved coverage is not progress: the
        # next iteration's selection/read/assess decides; merely seeing
        # them never counts here.
        continue

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        from aa.conversation.semantic_selection import selection_telemetry as _telemetry

        telemetry = _telemetry(
            previews=last_previews,
            selection=last_selection
            if last_selection is not None
            else type(
                "Empty",
                (),
                {
                    "selected_chunk_ids": [],
                    "need_more_detail": True,
                    "followup_queries": [],
                    "uncovered_need_ids": [],
                    "_used_model": False,
                },
            )(),
            latency_ms=sel_ms_total,
            information_needs=needs_for_selection,
        )
    except Exception:
        telemetry = {}
    try:
        preview_map_serialized = list(preview_map_merged.values())
    except Exception:
        preview_map_serialized = []
    from aa.retrieval.evidence import _short_digest as _digest
    from aa.retrieval.evidence import dedup_and_diversify as _dedup_meta

    try:
        _diverse_meta = _dedup_meta(
            index,
            pool_ids,
            fused,
            pool_cap=active.pool_cap,
            max_per_section=active.max_per_section,
        )
        _diverse_unique = len(_diverse_meta)
    except Exception:
        _diverse_unique = 0
    try:
        need_ids_ordered = [s.need_id for s in last_preview_statuses]
    except Exception:
        need_ids_ordered = list(need_id_order)
    query_need_serialized: list[dict[str, Any]] = []
    for map_dict in list(query_map_dicts or []):
        try:
            query_need_serialized.append(
                {
                    "query_id": str(map_dict.get("query_id", "")),
                    "need_ids": list(map_dict.get("need_ids", []) or []),
                }
            )
        except Exception:
            continue
    # Final coverage snapshot over the served pack (narrowing-aware: the
    # verdict below was assessed on the budgeted pack, so a condition
    # lost to child-chunk degradation already reads as uncovered).
    final_missing: list[str] = []
    final_per_need: list[dict[str, Any]] = []
    coverage_used_model = False
    coverage_all_covered = False
    try:
        final_verdict = last_verdict
        if final_verdict is not None:
            coverage_used_model = bool(getattr(final_verdict, "used_model", False))
            coverage_all_covered = bool(getattr(final_verdict, "all_covered", False))
            missing_raw = list(getattr(final_verdict, "missing_need_ids", []) or [])
            final_missing = [str(n) for n in missing_raw]
            for item in list(getattr(final_verdict, "needs", []) or []):
                try:
                    final_per_need.append(
                        {
                            "need_id": str(getattr(item, "need_id", "")),
                            "covered": bool(getattr(item, "covered", False)),
                            "passage_ids": [
                                str(p) for p in list(getattr(item, "passage_ids", []) or [])
                            ],
                            "anchors": [
                                {
                                    "passage_id": str(getattr(a, "passage_id", "")),
                                    "source_id": str(getattr(a, "source_id", "")),
                                    "section_id": str(getattr(a, "section_id", "")),
                                    "char_start": int(getattr(a, "char_start", 0) or 0),
                                    "char_end": int(getattr(a, "char_end", 0) or 0),
                                }
                                for a in list(getattr(item, "anchors", []) or [])
                            ],
                            "supporting_spans": [
                                str(s)[:200]
                                for s in list(getattr(item, "supporting_spans", []) or [])
                            ],
                            "missing": str(getattr(item, "missing", "") or "")[:300],
                        }
                    )
                except Exception:
                    continue
    except Exception:
        pass
    if coverage_status == "ready" and not coverage_all_covered:
        coverage_status = "exhausted"
        exhaustion_reason = "coverage-incomplete"
    if coverage_status == "ready":
        exhaustion_reason = ""
    try:
        from aa.corpus.budget import estimate_text_tokens as _estimate

        token_estimate = sum(_estimate(str(getattr(p, "exact_text", "") or "")) for p in selected)
    except Exception:
        token_estimate = int(total)
    metadata: dict[str, Any] = {
        "planner_query_count": len(cleaned),
        "primary_query_digest": _digest(cleaned[0]),
        "branch_lists": len(ranked_lists),
        "branch_top_k": active.branch_top_k,
        "rrf_k": active.rrf_k,
        "pool_cap": active.pool_cap,
        "fused_unique": len(fused),
        "pool_unique": len(pool_ids),
        "diverse_unique": _diverse_unique,
        "top_child_cap": active.top_child_cap,
        "selected_winners": len(read_child_order),
        "retrieval_backend": "rrf-only/1+semantic-selection+coverage-loop",
        "retrieval_loop": "discover/select_for_reading/read_exact/assess_coverage",
        "semantic_selection_model": selection_model is not None,
        "semantic_promoted": True,
        "followup_added": followup_added_total,
        "followup_need_ids": list(followup_need_ids),
        "discovered_ids": list(discovered_first),
        "previewed_ids": list(previewed_ids_all),
        "read_ids": list(read_ids),
        "information_need_ids": list(need_ids_ordered),
        "query_need_map": query_need_serialized,
        "preview_coverage": [
            {
                "need_id": s.need_id,
                "previewed_count": s.previewed_count,
                "best_fused_rank": s.best_fused_rank,
                "represented": s.represented,
            }
            for s in last_preview_statuses
        ],
        "selection_preview_map": preview_map_serialized,
        "uncovered_need_ids": list(
            dict.fromkeys(
                [
                    str(n)
                    for n in list(
                        (last_selection.uncovered_need_ids if last_selection is not None else [])
                        or []
                    )
                ]
                + list(final_missing)
            )
        ),
        "neighbor_window": active.neighbor_window,
        "expanded_passages": len(expanded),
        "selected_passages": len(selected),
        "budget_tokens": active.budget_tokens,
        "total_tokens": total,
        "latency_ms": elapsed_ms,
        "latency_budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
        "latency_over_budget": elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS,
        "coverage_status": coverage_status,
        "coverage_exhausted": bool(coverage_status == "exhausted"),
        "coverage_all_covered": bool(coverage_all_covered),
        "coverage_used_model": bool(coverage_used_model),
        "coverage_missing_need_ids": list(final_missing),
        "coverage_per_need": final_per_need,
        "coverage_exhaustion_reason": exhaustion_reason,
        "loop_iterations": len(fingerprints),
        "loop_expansions": expansions,
        "loop_searches": searches,
        "loop_model_calls": model_calls,
        "loop_token_estimate": token_estimate,
        "loop_progress_events": list(progress_events),
        "loop_fingerprints": list(fingerprints),
        "loop_deadline_stops": int(deadline_stops),
        "loop_preview_dedup": len(preview_identities_seen),
    }
    metadata.update({f"selection_{k}": v for k, v in telemetry.items()})
    logger.info(
        "v2 evidence queries=%d pool=%d winners=%d passages=%d tokens=%d coverage=%s",
        len(cleaned),
        len(last_previews),
        len(read_child_order),
        len(selected),
        total,
        coverage_status,
    )
    return EvidencePack(
        passages=tuple(selected),
        total_tokens=total,
        corpus_version=str(index.metadata.get("ru_artifact_sha256", "")),
        retrieval_metadata=metadata,
    )


def sanitize_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a privacy-safe selection/coverage snapshot (no text).

    Carries only the executed-path event records: stable chunk/passage
    ids, fused ranks, per-need ids, digests, counts, latencies and the
    truthful ``selection_route``. Any embedded preview/excerpt text is
    stripped so telemetry never persists book or user content.
    """
    safe: dict[str, Any] = {}
    try:
        raw = dict(metadata or {})
    except Exception:
        return {}
    for key in (
        "planner_query_count",
        "branch_lists",
        "branch_top_k",
        "rrf_k",
        "pool_cap",
        "fused_unique",
        "pool_unique",
        "diverse_unique",
        "top_child_cap",
        "selected_winners",
        "retrieval_backend",
        "retrieval_loop",
        "semantic_selection_model",
        "semantic_promoted",
        "followup_added",
        "neighbor_window",
        "expanded_passages",
        "selected_passages",
        "budget_tokens",
        "total_tokens",
        "latency_ms",
        "latency_budget_ms",
        "latency_over_budget",
        "coverage_status",
        "coverage_exhausted",
        "coverage_all_covered",
        "coverage_used_model",
        "coverage_exhaustion_reason",
        "loop_iterations",
        "loop_expansions",
        "loop_searches",
        "loop_model_calls",
        "loop_token_estimate",
    ):
        if key in raw and isinstance(raw[key], (str, int, float, bool)):
            safe[key] = raw[key]
    for key in (
        "discovered_ids",
        "previewed_ids",
        "read_ids",
        "information_need_ids",
        "followup_need_ids",
        "uncovered_need_ids",
        "coverage_missing_need_ids",
        "loop_progress_events",
        "loop_fingerprints",
    ):
        value = raw.get(key, [])
        if isinstance(value, list):
            safe[key] = [str(item) for item in value if str(item).strip()][:128]
    query_map = raw.get("query_need_map", [])
    if isinstance(query_map, list):
        safe["query_need_map"] = [
            {
                "query_id": str(entry.get("query_id", "")),
                "need_ids": [str(nid) for nid in list(entry.get("need_ids", []) or [])],
            }
            for entry in query_map
            if isinstance(entry, dict)
        ][:32]
    preview_coverage = raw.get("preview_coverage", [])
    if isinstance(preview_coverage, list):
        safe["preview_coverage"] = [
            {
                "need_id": str(entry.get("need_id", "")),
                "previewed_count": int(entry.get("previewed_count", 0) or 0),
                "represented": bool(entry.get("represented", False)),
            }
            for entry in preview_coverage
            if isinstance(entry, dict)
        ][:32]
    preview_map = raw.get("selection_preview_map", [])
    if isinstance(preview_map, list):
        safe["selection_preview_map"] = [
            {
                "chunk_id": str(entry.get("chunk_id", "")),
                "fused_rank": int(entry.get("fused_rank", 0) or 0),
                "query_ids": [str(q) for q in list(entry.get("query_ids", []) or [])],
                "need_ids": [str(n) for n in list(entry.get("need_ids", []) or [])],
                "selected": bool(entry.get("selected", False)),
            }
            for entry in preview_map
            if isinstance(entry, dict)
        ][:64]
    coverage_per_need = raw.get("coverage_per_need", [])
    if isinstance(coverage_per_need, list):
        safe["coverage_per_need"] = [
            {
                "need_id": str(entry.get("need_id", "")),
                "covered": bool(entry.get("covered", False)),
                "passage_ids": [
                    str(pid) for pid in list(entry.get("passage_ids", []) or []) if str(pid)
                ][:12],
            }
            for entry in coverage_per_need
            if isinstance(entry, dict)
        ][:16]
    for key in (
        "selection_candidates",
        "selection_selected",
        "selection_max_rank",
        "selection_need_more",
        "selection_followups",
        "selection_latency_ms",
        "selection_model_used",
        "selection_fallback_used",
        "selection_need_count",
        "selection_need_represented",
        "selection_unmapped_previews",
        "selection_unmapped_selected",
    ):
        for candidate in (key, f"selection_{key}"):
            if candidate in raw and isinstance(raw[candidate], (int, float, bool)):
                safe[key] = raw[candidate]
                break
    for key in (
        "selection_digest",
        "selection_deep_rank_gt5",
        "selection_deep_rank_gt16",
        "selection_need_unrepresented",
        "selection_uncovered_need_ids",
    ):
        for candidate in (key, f"selection_{key}"):
            if candidate in raw:
                value = raw[candidate]
                if isinstance(value, (str, bool, list)):
                    safe[key] = value
                break
    per_need = raw.get(
        "selection_per_need_selected", raw.get("selection_selection_per_need_selected")
    )
    if isinstance(per_need, dict):
        safe["selection_per_need_selected"] = {
            str(key): int(value or 0) for key, value in per_need.items() if str(key).strip()
        }
    safe["selection_route"] = selection_route_from_metadata(raw)
    return safe


def selection_route_from_metadata(metadata: dict[str, Any]) -> str:
    """Derive the truthful selection route from executed-path events.

    ``model_selection`` only when the selector model actually ran
    (``selection_model_used`` true); ``lexical_fallback`` when previews
    were discovered but the heuristic path served them;
    ``selection_unavailable`` when retrieval ran but produced no usable
    selection; ``not_requested`` when no retrieval was requested.
    Never inferred from pack size or heuristic ordering.
    """
    try:
        raw = dict(metadata or {})
    except Exception:
        return "selection_unavailable"
    if str(raw.get("coverage_status", "") or "") == "not_requested":
        return "not_requested"
    model_used = raw.get("selection_model_used", raw.get("selection_selection_model_used", False))
    fallback_used = raw.get(
        "selection_fallback_used", raw.get("selection_selection_fallback_used", False)
    )
    try:
        if bool(model_used):
            return "model_selection"
    except Exception:
        pass
    try:
        if bool(fallback_used):
            return "lexical_fallback"
    except Exception:
        pass
    discovered = raw.get("discovered_ids", [])
    previewed = raw.get("previewed_ids", [])
    has_discovery = (isinstance(discovered, list) and len(discovered) > 0) or (
        isinstance(previewed, list) and len(previewed) > 0
    )
    if has_discovery:
        return "lexical_fallback"
    if str(raw.get("coverage_status", "") or "") == "not_requested":
        return "not_requested"
    return "selection_unavailable"


async def retrieval_node(
    state: TurnState,
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
    selection_model: Any | None = None,
    coverage_model: Any | None = None,
) -> dict[str, Any]:
    """LangGraph retrieval node: queries to hits plus Evidence Pack.

    Runs the shared read/coverage loop on the RAM-resident index. The
    per-turn wall-clock latency against
    ``INTERACTIVE_LATENCY_BUDGET_MS`` is measured and propagated in
    state (``retrieval_latency_ms``/``retrieval_over_budget``) for
    observability. Only counts and latencies are logged, never prompts
    or user text.

    Model-driven semantic selection runs over genuinely broad hybrid
    candidates (including fused rank >16) BEFORE winner/pack budgeting;
    without a model the bounded heuristic discovery applies and coverage
    stays conservatively uncovered (lexical signals never decide
    sufficiency).
    """
    raw_queries = state.get("search_queries", [])
    if isinstance(raw_queries, (list, tuple)):
        queries: list[str] = []
        seen: set[str] = set()
        for raw in raw_queries:
            if not isinstance(raw, str):
                continue
            cleaned = " ".join(raw.split())
            if not cleaned or cleaned.casefold() in seen:
                continue
            seen.add(cleaned.casefold())
            queries.append(cleaned)
            if len(queries) >= 16:
                break
    else:
        queries = []
    if not queries:
        logger.info("v2 retrieval skipped", extra={"queries": 0})
        return {
            "retrieval_hits": [],
            "evidence_pack": [],
            "retrieval_latency_ms": 0.0,
            "retrieval_over_budget": False,
            "retrieval_metadata": sanitize_retrieval_metadata(
                {
                    "retrieval_backend": "rrf-only/1+semantic-selection+coverage-loop",
                    "selection_route": "not_requested",
                    "discovered_ids": [],
                    "previewed_ids": [],
                    "read_ids": [],
                    "coverage_status": "not_requested",
                }
            ),
        }
    active_config = config if config is not None else RetrievalConfig()
    started = time.perf_counter()
    resolved_intent, conversation_context, user_message = _selection_context_from_state(state)
    canonical_digest = selection_context_digest(state)
    # Recover the typed information needs + trusted query->need map (#311)
    # from the canonical resolved turn; unresolvable stays unmapped and
    # triggers conservative discovery downstream, never fake coverage.
    state_needs: Any | None = None
    state_query_map: Any | None = None
    try:
        from aa.conversation.conversation_context import query_need_map_from_state as _qmap_state
        from aa.conversation.conversation_context import resolved_turn_from_state as _rt_state

        _rt = _rt_state(state)
        if _rt is not None:
            state_needs = [
                item.model_dump(mode="json") for item in list(_rt.information_needs or [])
            ]
            state_query_map = [
                item.model_dump(mode="json") for item in list(_rt.query_need_map or [])
            ]
        else:
            raw_needs = state.get("information_needs", [])
            state_needs = list(raw_needs) if isinstance(raw_needs, list) else []
            state_query_map = list(_qmap_state(state)) or None
            if state_query_map is not None:
                try:
                    state_query_map = [item.model_dump(mode="json") for item in state_query_map]
                except Exception:
                    state_query_map = None
    except Exception:
        state_needs = None
        state_query_map = None
    # One shared async discover/select/read/assess/expand/search_more
    # mechanism for every path (normal, repair, safety, no-evidence).
    # Provider 429 propagates for checkpoint recovery; any other failure
    # inside the loop already degrades to conservative uncovered rather
    # than an empty silent success, so no legacy rank-only fallback runs
    # here.
    pack = await aretrieve_with_semantic_selection(
        index,
        queries,
        config=active_config,
        resolved_intent=resolved_intent,
        conversation_context=conversation_context,
        user_message=user_message,
        selection_model=selection_model,
        coverage_model=coverage_model if coverage_model is not None else selection_model,
        context_digest=canonical_digest,
        information_needs=state_needs,
        query_need_map=state_query_map,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    over_budget = elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS
    try:
        _meta = dict(pack.retrieval_metadata or {})
        _preview_map = _meta.get("selection_preview_map", [])
        _child_query: dict[str, list[str]] = {}
        _child_need: dict[str, list[str]] = {}
        if isinstance(_preview_map, list):
            for entry in _preview_map:
                if not isinstance(entry, dict):
                    continue
                cid = str(entry.get("chunk_id", "") or "").strip()
                if not cid:
                    continue
                qids = [str(q) for q in list(entry.get("query_ids", []) or []) if str(q).strip()]
                nids = [str(n) for n in list(entry.get("need_ids", []) or []) if str(n).strip()]
                if qids:
                    _child_query[cid] = qids
                if nids:
                    _child_need[cid] = nids
        # Follow-up winners carry need attribution via the same typed map.
        for f_need in list(_meta.get("followup_need_ids", []) or []):
            _ = f_need
        hits, pack_dicts = pack_to_state(
            pack, child_query_map=_child_query or None, child_need_map=_child_need or None
        )
    except Exception:
        hits, pack_dicts = pack_to_state(pack)
    if over_budget:
        logger.warning(
            "v2 retrieval over interactive budget",
            extra={
                "queries": len(queries),
                "latency_ms": round(elapsed_ms, 1),
                "budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
            },
        )
    logger.info(
        "v2 retrieval done",
        extra={
            "queries": len(queries),
            "hits": len(hits),
            "passages": len(pack_dicts),
            "tokens": pack.total_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "over_budget": over_budget,
        },
    )
    try:
        retrieval_metadata = sanitize_retrieval_metadata(dict(pack.retrieval_metadata or {}))
    except Exception:
        retrieval_metadata = {}
    return {
        "retrieval_hits": hits,
        "evidence_pack": pack_dicts,
        "retrieval_latency_ms": elapsed_ms,
        "retrieval_over_budget": over_budget,
        "retrieval_metadata": retrieval_metadata,
    }


def make_retrieval_node(
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
    selection_model: Any | None = None,
    coverage_model: Any | None = None,
) -> Any:
    """Build the evidence retrieval node bound to one RAM-resident index."""
    active_config = config if config is not None else RetrievalConfig()

    async def run_evidence_retrieval(state: TurnState) -> dict[str, Any]:
        return await retrieval_node(
            state,
            index=index,
            config=active_config,
            selection_model=selection_model,
            coverage_model=coverage_model if coverage_model is not None else selection_model,
        )

    return run_evidence_retrieval


__all__ = [
    "COVERAGE_PER_CALL_BUDGET_S",
    "SELECTION_PER_CALL_BUDGET_S",
    "aretrieve_with_semantic_selection",
    "clear_canonical_read_cache",
    "make_retrieval_node",
    "pack_to_state",
    "retrieval_node",
    "sanitize_retrieval_metadata",
    "selection_context_digest",
    "selection_route_from_metadata",
    "state_passages_to_prompt",
]
