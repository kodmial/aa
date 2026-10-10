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
import logging
import time
from typing import Any

from aa.conversation.graph_state import TurnState
from aa.conversation.prompt_builder import EvidencePassage
from aa.retrieval.evidence import (
    INTERACTIVE_LATENCY_BUDGET_MS,
    EvidencePack,
    RetrievalConfig,
    retrieve_evidence,
)
from aa.retrieval.index import HybridIndex, logical_chunk_id

logger = logging.getLogger("aa.conversation.retrieval_node")


def pack_to_state(pack: EvidencePack) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map an Evidence Pack to graph state payloads (exact text + provenance)."""
    hits: list[dict[str, Any]] = []
    pack_dicts: list[dict[str, Any]] = []
    for passage in pack.passages:
        pack_dicts.append(
            {
                "passage_id": passage.passage_id,
                "text": passage.exact_text,
                "source_id": passage.source_id,
                "section_id": passage.section_id,
                "child_chunk_ids": list(passage.child_chunk_ids),
                "char_start": passage.char_start,
                "char_end": passage.char_end,
                "text_sha256": passage.text_sha256,
                "source_sha256": passage.source_sha256,
                "corpus_version": pack.corpus_version,
            }
        )
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
) -> EvidencePack:
    """Run broad hybrid retrieval with model-driven selection before budgeting.

    Issue #295: the genuinely broad BM25+E5/RRF pool (including fused rank
    >16) is exposed as discovery previews; an LLM selects pertinent
    candidates against the real context-resolved intent; exact complete
    canonical passages (+neighbors, provenance preserved) are fetched only
    for the validated selection; one bounded targeted follow-up search
    genuinely changes the evidence when relevance is weak/unknown. Earlier
    top-N caps never irreversibly hide candidates before selection. No
    second-stage BGE; generic token-overlap is fallback/discovery ordering
    only, never semantic authority when a model is bound.
    """
    from aa.retrieval.evidence import (
        expand_small_to_big,
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
    started = time.perf_counter()
    ranked_lists, per_query_ids = await asyncio.to_thread(
        run_branch_searches, index, cleaned, branch_top_k=active.branch_top_k
    )
    fused, pool_ids = fuse_query_pool(
        ranked_lists, per_query_ids, rrf_k=active.rrf_k, pool_cap=active.pool_cap
    )
    # Discovery previews use the broad fused pool directly; per-section
    # diversity caps are applied only to fallback/budgeting after selection.
    # Broad discovery previews BEFORE any per-section diversity truncation
    # or top-N winner budgeting (issue #295). The genuinely broad fused
    # pool (global RRF order, exact-ID dedup only, ranks >16 included) is
    # shown to the LLM; diversity caps apply only to fallback/budgeting
    # after the model has selected.
    from aa.conversation.semantic_selection import (
        MAX_FOLLOWUP_QUERIES,
        MAX_SELECTED_CHUNKS,
        MAX_SELECTION_CANDIDATES,
        aselect_semantic_candidates,
        preview_candidates,
        selection_telemetry,
    )

    broad_sorted = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
    fused_ordered = [(item.chunk_id, item.fused_score) for item in broad_sorted]
    in_index = [cid for cid, _ in fused_ordered if cid in index.chunks]
    texts = {cid: getattr(index.chunks[cid], "text", "") for cid in in_index}
    sections = {cid: getattr(index.chunks[cid], "section", "") for cid in in_index}
    sources = {cid: getattr(index.chunks[cid], "source_id", "") for cid in in_index}
    previews = preview_candidates(
        fused_ordered=fused_ordered,
        texts=texts,
        sections=sections,
        sources=sources,
        limit=MAX_SELECTION_CANDIDATES,
    )
    sel_started = time.perf_counter()
    # Provider 429 propagates (runner retire/checkpoint resume); other
    # model failures fall back to the bounded heuristic inside the selector.
    # Pre-model preview gate verifies against this independent fused/index
    # ground truth, never against the previews themselves.
    known_chunk_ids = {cid for cid in fused if cid in index.chunks}
    selection = await aselect_semantic_candidates(
        previews,
        resolved_intent=resolved_intent,
        conversation_context=conversation_context,
        user_message=user_message,
        model=selection_model,
        known_chunk_ids=known_chunk_ids,
        context_digest=context_digest,
    )
    sel_ms = (time.perf_counter() - sel_started) * 1000.0
    winners = [fused[cid] for cid in selection.selected_chunk_ids if cid in fused]
    # Observable retrieval stages: discovery alone never blocks later
    # consideration. Only ids the model actually previewed/read are
    # excluded from follow-up promotion; a rank-66 candidate that was
    # discovered but never previewed stays eligible.
    discovered_ids = [item.chunk_id for item in broad_sorted if item.chunk_id in fused]
    previewed_ids = [preview.chunk_id for preview in previews]
    previewed_set = set(previewed_ids)
    # One bounded targeted follow-up when the selector reports weak/unknown
    # relevance: a focused additional search that must add genuinely new
    # chunk ids (never a silent regeneration on the same top prefixes).
    followup_added = 0
    if selection.need_more_detail:
        followups = list(selection.followup_queries[:MAX_FOLLOWUP_QUERIES])
        if not followups and resolved_intent.strip():
            followups = [" ".join(resolved_intent.split())[:500]]
        try:
            followups = validate_recovery_queries(followups)
        except Exception:
            followups = []
        if followups:
            try:
                f_ranked, f_per_ids = await asyncio.to_thread(
                    run_branch_searches, index, followups, branch_top_k=active.branch_top_k
                )
                f_fused, f_pool = fuse_query_pool(
                    f_ranked, f_per_ids, rrf_k=active.rrf_k, pool_cap=active.pool_cap
                )
                seen = {w.chunk_id for w in winners} | previewed_set
                # Genuinely new evidence first: ids the model never
                # previewed/read, best-first, bounded. Discovered-only ids
                # (returned by the first retrieval but outside the preview
                # window) remain promotable here.
                fresh = sorted(
                    (
                        cand
                        for cid, cand in f_fused.items()
                        if cid not in seen and cid in index.chunks
                    ),
                    key=lambda item: item.fused_score,
                    reverse=True,
                )[:8]
                followup_added = 0
                for cand in fresh:
                    if len(winners) >= MAX_SELECTED_CHUNKS + 8:
                        break
                    winners.append(cand)
                    fused[cand.chunk_id] = cand
                    followup_added += 1
            except Exception as exc:
                from aa.opencode.errors import OpenCodeRateLimitError as _FollowupRateLimit

                if isinstance(exc, (_FollowupRateLimit, asyncio.CancelledError)):
                    raise
                followup_added = 0
    explicit_empty_followup = bool(
        getattr(selection, "_used_model", False)
        and not selection.selected_chunk_ids
        and bool(selection.need_more_detail)
    )
    if not winners and not explicit_empty_followup:
        # Fail-safe bounded fallback (never empty silent success): top RRF
        # winners in fused order. Reached only when selection yields nothing
        # usable; deep ranks stay reachable via the previews above on retry.
        # Diversity caps apply here only, never to discovery previews above.
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
        # Section lookup stays bounded to the fallback candidates instead
        # of iterating the whole index on every fallback turn.
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
    expanded = await asyncio.to_thread(
        expand_small_to_big, index, winners, neighbor_window=active.neighbor_window
    )
    selected, total = select_passages_under_budget(
        expanded,
        budget_tokens=active.budget_tokens,
        index=index,
        priority_child_ids=tuple(w.chunk_id for w in winners),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        telemetry = selection_telemetry(previews=previews, selection=selection, latency_ms=sel_ms)
    except Exception:
        telemetry = {}
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

    read_ids = [w.chunk_id for w in winners]
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
        "selected_winners": len(winners),
        "retrieval_backend": "rrf-only/1+semantic-selection",
        "semantic_selection_model": selection_model is not None,
        "semantic_promoted": True,
        "followup_added": followup_added,
        "discovered_ids": list(discovered_ids),
        "previewed_ids": list(previewed_ids),
        "read_ids": list(read_ids),
        "neighbor_window": active.neighbor_window,
        "expanded_passages": len(expanded),
        "selected_passages": len(selected),
        "budget_tokens": active.budget_tokens,
        "total_tokens": total,
        "latency_ms": elapsed_ms,
        "latency_budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
        "latency_over_budget": elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS,
    }
    metadata.update({f"selection_{k}": v for k, v in telemetry.items()})
    logger.info(
        "v2 evidence queries=%d pool=%d winners=%d passages=%d tokens=%d",
        len(cleaned),
        len(previews),
        len(winners),
        len(selected),
        total,
    )
    return EvidencePack(
        passages=tuple(selected),
        total_tokens=total,
        corpus_version=str(index.metadata.get("ru_artifact_sha256", "")),
        retrieval_metadata=metadata,
    )


async def retrieval_node(
    state: TurnState,
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
    selection_model: Any | None = None,
) -> dict[str, Any]:
    """LangGraph retrieval node: queries to hits plus Evidence Pack.

    Runs the RRF-only pipeline in a worker thread on the RAM-resident index.
    The per-turn wall-clock latency against
    ``INTERACTIVE_LATENCY_BUDGET_MS`` is measured and propagated in
    state (``retrieval_latency_ms``/``retrieval_over_budget``) for
    observability. Only counts and latencies are logged, never prompts
    or user text.

    When ``selection_model`` is bound, model-driven semantic selection runs
    over genuinely broad hybrid candidates (including fused rank >16)
    BEFORE winner/pack budgeting; otherwise the bounded heuristic promotion
    inside :func:`retrieve_evidence` applies.
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
        }
    active_config = config if config is not None else RetrievalConfig()
    started = time.perf_counter()
    resolved_intent, conversation_context, user_message = _selection_context_from_state(state)
    canonical_digest = selection_context_digest(state)
    if selection_model is not None:
        # Model-driven path: broad candidates -> LLM selection -> exact
        # full fetch -> budgeting. Provider 429 propagates for checkpoint
        # recovery; any other selection failure falls back to the bounded
        # heuristic pipeline below (never an empty silent success).
        try:
            pack = await aretrieve_with_semantic_selection(
                index,
                queries,
                config=active_config,
                resolved_intent=resolved_intent,
                conversation_context=conversation_context,
                user_message=user_message,
                selection_model=selection_model,
                context_digest=canonical_digest,
            )
        except Exception as exc:
            from aa.opencode.errors import OpenCodeRateLimitError as _SelRateLimit

            if isinstance(exc, (_SelRateLimit, asyncio.CancelledError)):
                raise
            logger.info(
                "v2 retrieval selection failed; heuristic pipeline used",
                extra={"category": type(exc).__name__},
            )
            pack = await asyncio.to_thread(
                retrieve_evidence,
                index,
                queries,
                config=active_config,
                resolved_intent=resolved_intent,
                conversation_context=conversation_context,
            )
    else:
        pack = await asyncio.to_thread(
            retrieve_evidence,
            index,
            queries,
            config=active_config,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    over_budget = elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS
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
    return {
        "retrieval_hits": hits,
        "evidence_pack": pack_dicts,
        "retrieval_latency_ms": elapsed_ms,
        "retrieval_over_budget": over_budget,
    }


def make_retrieval_node(
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
    selection_model: Any | None = None,
) -> Any:
    """Build the evidence retrieval node bound to one RAM-resident index."""
    active_config = config if config is not None else RetrievalConfig()

    async def run_evidence_retrieval(state: TurnState) -> dict[str, Any]:
        return await retrieval_node(
            state, index=index, config=active_config, selection_model=selection_model
        )

    return run_evidence_retrieval


__all__ = [
    "aretrieve_with_semantic_selection",
    "make_retrieval_node",
    "pack_to_state",
    "retrieval_node",
    "selection_context_digest",
    "state_passages_to_prompt",
]
