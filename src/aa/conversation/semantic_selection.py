"""Model-driven semantic candidate selection over broad book candidates.

Issue #295: BM25+E5/FAISS/RRF discovers a genuinely broad ranked
candidate set, but fixed top-5/top-16 caps previously discarded
potentially decisive passages before the answer model could evaluate
them. This module implements the bounded selection layer between
fusion and full-text fetch:

- candidate *previews* (short discovery excerpts + provenance + fused
  rank) are used only for discovery, never as final supporting text;
- an LLM assesses the previews against the user's real context-resolved
  intent plus dialogue context, selects pertinent candidates, and may
  request targeted re-read (neighbor expansion) or another search;
- once selected, callers fetch exact complete canonical passages via
  :func:`fetch_full_passages_for_selection` (small-to-big expansion,
  provenance preserved);
- without a model (tests, outage) a deterministic generic heuristic
  promotion applies: generic token-overlap relevance over the same
  previews, stable by fused rank. No domain-specific regex, intent
  whitelist, keyword table, or fixed qualification routing lives here.

Bounds (never an unbounded dump, never unconditional slow BGE):

- at most ``MAX_SELECTION_CANDIDATES`` previews enter one selection;
- previews carry at most ``PREVIEW_CHARS`` characters each;
- at most ``MAX_SELECTED_CHUNKS`` chunk ids are selected per round;
- follow-up queries are bounded to ``MAX_FOLLOWUP_QUERIES``.

Privacy: previews contain short book excerpts for the selecting model
only; telemetry carries counts/ranks/ids-hashes, never text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("aa.conversation.semantic_selection")

MAX_SELECTION_CANDIDATES = 64
PREVIEW_CHARS = 400
MAX_SELECTED_CHUNKS = 12
MAX_FOLLOWUP_QUERIES = 6
SELECTION_ATTEMPT_BUDGET_S = 8.0

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


@dataclass(frozen=True)
class CandidatePreview:
    """One broad candidate for discovery-only inspection."""

    chunk_id: str
    fused_rank: int
    fused_score: float
    section: str
    source_id: str
    preview_text: str
    full_text: str = ""


class SemanticSelection(BaseModel):
    """Bounded model decision over candidate previews (transport only)."""

    selected_chunk_ids: list[str] = Field(default_factory=list)
    need_more_detail: bool = Field(default=False)
    followup_queries: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class SemanticSelectionError(ValueError):
    """Semantic selection output failed validation."""


def _tokenize(text: str) -> list[str]:
    return [tok.casefold() for tok in _WORD_RE.findall(text or "") if len(tok) >= 3]


def preview_candidates(
    *,
    fused_ordered: Sequence[tuple[str, float]],
    texts: dict[str, str],
    sections: dict[str, str] | None = None,
    sources: dict[str, str] | None = None,
    limit: int = MAX_SELECTION_CANDIDATES,
) -> list[CandidatePreview]:
    """Build bounded discovery previews in fused-rank order.

    ``fused_ordered`` is ``[(chunk_id, fused_score)]`` best-first over the
    genuinely broad pool (ranks >16 included). Preview text is truncated
    for discovery; :func:`fetch_full_passages_for_selection` later
    supplies the exact complete text for selected ids only.
    """
    out: list[CandidatePreview] = []
    sections = sections or {}
    sources = sources or {}
    for rank, (chunk_id, score) in enumerate(fused_ordered[: max(1, limit)]):
        full = texts.get(chunk_id, "")
        preview = full[:PREVIEW_CHARS] if isinstance(full, str) else ""
        out.append(
            CandidatePreview(
                chunk_id=chunk_id,
                fused_rank=rank,
                fused_score=float(score),
                section=str(sections.get(chunk_id, "")),
                source_id=str(sources.get(chunk_id, "")),
                preview_text=preview,
                full_text=full if isinstance(full, str) else "",
            )
        )
    return out


def heuristic_relevance_score(preview_text: str, resolved_intent: str) -> float:
    """Generic token-overlap relevance (no domain tables, no regex intents).

    Counts distinct intent tokens (length >=3, casefolded) present in the
    preview, normalized by intent size. Purely generic lexical signal used
    only when no selection model is available, or to order previews
    deterministically before the model call. Never decides support.
    """
    intent_tokens = {tok for tok in _tokenize(resolved_intent)}
    if not intent_tokens:
        return 0.0
    preview_tokens = set(_tokenize(preview_text))
    if not preview_tokens:
        return 0.0
    return len(intent_tokens & preview_tokens) / max(1, len(intent_tokens))


def rerank_previews_semantically(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
) -> list[CandidatePreview]:
    """Promote intent-relevant previews without dropping deep ranks.

    Stable sort: heuristic relevance first, fused rank second. Deep fused
    ranks (>5, >16) with genuine lexical overlap therefore surface while
    every candidate stays inspectable. No content is rewritten.
    """
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    scored = [(heuristic_relevance_score(p.preview_text, context), p) for p in previews]
    scored.sort(key=lambda pair: (-pair[0], pair[1].fused_rank))
    return [preview for _, preview in scored]


def select_by_fused_fallback(
    previews: Sequence[CandidatePreview],
    *,
    limit: int = MAX_SELECTED_CHUNKS,
) -> SemanticSelection:
    """Deterministic fallback: keep fused order (never hide deep ranks)."""
    selected = [p.chunk_id for p in list(previews)[: max(1, limit)]]
    return SemanticSelection(
        selected_chunk_ids=selected, need_more_detail=False, followup_queries=[]
    )


def heuristic_select(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
    limit: int = MAX_SELECTED_CHUNKS,
) -> SemanticSelection:
    """Deterministic generic selection promoting relevant deep candidates."""
    if not previews:
        return SemanticSelection(selected_chunk_ids=[], need_more_detail=True, followup_queries=[])
    reranked = rerank_previews_semantically(
        previews, resolved_intent=resolved_intent, conversation_context=conversation_context
    )
    selected = [p.chunk_id for p in reranked[: max(1, limit)]]
    # Signal targeted re-read when nothing lexically matches: the caller
    # should run one focused additional search instead of regenerating on
    # the same misleading top prefixes.
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    best = heuristic_relevance_score(reranked[0].preview_text, context) if reranked else 0.0
    return SemanticSelection(
        selected_chunk_ids=selected,
        need_more_detail=bool(best <= 0.0 and context),
        followup_queries=[],
    )


def selection_prompt(
    *,
    previews: Sequence[CandidatePreview],
    resolved_intent: str,
    conversation_context: str = "",
    user_message: str = "",
) -> tuple[str, str]:
    """Render the bounded selection prompt (system, user)."""
    system = (
        "You are a book-evidence selector. Read the context-resolved intent "
        "and the candidate passage previews below. Select the candidate ids "
        "that are materially relevant for answering the intent. Use only "
        "semantic relevance to the actual intent, never keyword counting "
        "alone. Set need_more_detail true when no preview looks sufficient "
        "and a focused follow-up search is needed. Return only the required "
        "JSON decision."
    )
    lines: list[str] = [
        "<resolved_intent>",
        (resolved_intent or user_message or "").strip()[:2000] or "(no intent)",
        "</resolved_intent>",
    ]
    if user_message.strip():
        lines += ["<current_user_message>", user_message.strip()[:2000], "</current_user_message>"]
    if conversation_context.strip():
        lines += [
            "<conversation_context>",
            conversation_context.strip()[:2000],
            "</conversation_context>",
        ]
    lines.append("<candidates>")
    for preview in list(previews)[:MAX_SELECTION_CANDIDATES]:
        lines.append(
            f'<candidate id="{preview.chunk_id}" rank="{preview.fused_rank}" '
            f'section="{preview.section}">{preview.preview_text}</candidate>'
        )
    if not previews:
        lines.append("(no candidates)")
    lines.append("</candidates>")
    lines.append(
        "Return ONLY a JSON object with exactly these keys: "
        '{"selected_chunk_ids": array of strings, '
        '"need_more_detail": boolean, "followup_queries": array of strings}. '
        f"Select at most {MAX_SELECTED_CHUNKS} ids from <candidates>. "
        f"Provide at most {MAX_FOLLOWUP_QUERIES} follow-up queries."
    )
    return system, "\n".join(lines)


def semantic_selection_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_chunk_ids": {"type": "array", "items": {"type": "string"}},
            "need_more_detail": {"type": "boolean"},
            "followup_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["selected_chunk_ids", "need_more_detail", "followup_queries"],
    }


def validate_semantic_selection(
    data: object, *, known_chunk_ids: set[str] | None = None
) -> SemanticSelection:
    """Strictly validate one selection decision (fail-closed)."""
    if isinstance(data, SemanticSelection):
        selection = data
    elif isinstance(data, dict):
        try:
            selection = SemanticSelection.model_validate(data)
        except Exception as exc:
            raise SemanticSelectionError(f"semantic selection invalid: {exc}") from exc
    else:
        raise SemanticSelectionError("semantic selection is not an object")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in selection.selected_chunk_ids:
        chunk_id = str(raw or "").strip()
        if not chunk_id or chunk_id in seen:
            continue
        if known_chunk_ids is not None and chunk_id not in known_chunk_ids:
            raise SemanticSelectionError(f"selection cites unknown candidate {chunk_id!r}")
        seen.add(chunk_id)
        cleaned.append(chunk_id)
    if not cleaned:
        raise SemanticSelectionError("semantic selection selected no candidates")
    cleaned = cleaned[:MAX_SELECTED_CHUNKS]
    followups: list[str] = []
    for raw in selection.followup_queries:
        query = " ".join(str(raw or "").split()).strip()
        if not query:
            continue
        key = query.casefold()
        if key in {q.casefold() for q in followups}:
            continue
        followups.append(query)
        if len(followups) >= MAX_FOLLOWUP_QUERIES:
            break
    return SemanticSelection(
        selected_chunk_ids=cleaned,
        need_more_detail=bool(selection.need_more_detail),
        followup_queries=followups,
    )


async def aselect_semantic_candidates(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
    user_message: str = "",
    model: Any | None = None,
) -> SemanticSelection:
    """Model-driven selection with deterministic fallback (bounded).

    When ``model`` is None or the model call fails, falls back to
    :func:`heuristic_select` so deep relevant candidates still surface
    without inventing ids. Unknown cited ids fail closed to the
    heuristic. Provider 429 propagates.
    """
    preview_list = list(previews)[:MAX_SELECTION_CANDIDATES]
    if not preview_list:
        return SemanticSelection(selected_chunk_ids=[], need_more_detail=True, followup_queries=[])
    known = {p.chunk_id for p in preview_list}
    if model is None:
        return heuristic_select(
            preview_list,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
        )
    system, user_text = selection_prompt(
        previews=preview_list,
        resolved_intent=resolved_intent,
        conversation_context=conversation_context,
        user_message=user_message,
    )
    try:
        structured = getattr(model, "ainvoke_structured", None)
        if callable(structured):
            raw = await structured(
                user_text,
                system=system,
                schema=semantic_selection_schema(),
                retry_count=1,
            )
            return validate_semantic_selection(raw, known_chunk_ids=known)
        text_invoke = getattr(model, "_ainvoke_text", None) or getattr(model, "ainvoke", None)
        if not callable(text_invoke):
            raise SemanticSelectionError("selection model has no invocation path")
        if getattr(text_invoke, "__name__", "") == "_ainvoke_text":
            reply = await text_invoke(user_text, system=system)
            text = reply if isinstance(reply, str) else str(reply)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            message = await text_invoke(
                [SystemMessage(content=system), HumanMessage(content=user_text)]
            )
            content = getattr(message, "content", "")
            text = content if isinstance(text, str) else str(content)
        import json as _json

        cleaned = text.strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        payload = cleaned[start : end + 1] if start >= 0 and end > start else cleaned
        try:
            data = _json.loads(payload)
        except Exception as exc:
            raise SemanticSelectionError(f"selection text is not JSON: {exc}") from exc
        return validate_semantic_selection(data, known_chunk_ids=known)
    except Exception as exc:
        # 429 must escape this selector: the workflow retires the runner
        # and resumes from its last verified checkpoint. An inner broad
        # except previously swallowed the re-raised rate-limit error.
        from aa.opencode.errors import OpenCodeRateLimitError

        if isinstance(exc, OpenCodeRateLimitError):
            raise
        if isinstance(exc, SemanticSelectionError):
            logger.info("semantic selection invalid; heuristic fallback used")
        else:
            logger.info(
                "semantic selection unavailable; heuristic fallback used",
                extra={"category": type(exc).__name__},
            )
        return heuristic_select(
            preview_list,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
        )


def fetch_full_passages_for_selection(
    index: Any,
    selected_chunk_ids: Sequence[str],
    *,
    neighbor_window: int | None = None,
) -> list[Any]:
    """Fetch exact complete passages for selected chunk ids (provenance kept).

    Uses small-to-big expansion (full parent paragraph + bounded neighbor
    window) over the canonical index. Returns
    ``EvidencePassageData`` items in selection order; unknown ids raise.
    No truncation is applied here: the caller budgets the pack.
    """
    from aa.retrieval.evidence import NEIGHBOR_WINDOW, expand_small_to_big
    from aa.retrieval.fusion import FusedCandidate

    window = NEIGHBOR_WINDOW if neighbor_window is None else int(neighbor_window)
    winners = [
        FusedCandidate(
            chunk_id=str(chunk_id),
            fused_score=float(len(selected_chunk_ids) - pos),
            lexical_rank=None,
            dense_rank=None,
            lexical_score=None,
            dense_score=None,
        )
        for pos, chunk_id in enumerate(selected_chunk_ids)
    ]
    return expand_small_to_big(index, winners, neighbor_window=window)


def order_pack_semantically(
    pack_dicts: Sequence[dict[str, Any]],
    *,
    resolved_intent: str,
    conversation_context: str = "",
) -> list[dict[str, Any]]:
    """Reorder an Evidence Pack by generic intent relevance (stable).

    No passage is dropped and no text is altered: relevant deep-rank
    passages move forward while the full pack stays available to the
    generator and verifier. Deterministic and model-free so hermetic
    tests and outage paths behave identically.
    """
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    if not context:
        return [dict(item) for item in pack_dicts]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for pos, item in enumerate(pack_dicts):
        text = str(item.get("text", "")) if isinstance(item, dict) else ""
        score = heuristic_relevance_score(text[: PREVIEW_CHARS * 2], context)
        scored.append((score, pos, dict(item)))
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [item for _, _, item in scored]


def selection_telemetry(
    *,
    previews: Sequence[CandidatePreview],
    selection: SemanticSelection,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Privacy-safe selection telemetry (ranks/counts only, never text)."""
    by_id = {p.chunk_id: p for p in previews}
    ranks = [by_id[cid].fused_rank for cid in selection.selected_chunk_ids if cid in by_id]
    digest = hashlib.sha256("|".join(selection.selected_chunk_ids).encode("utf-8")).hexdigest()[:16]
    return {
        "selection_candidates": len(previews),
        "selection_selected": len(selection.selected_chunk_ids),
        "selection_max_rank": max(ranks) if ranks else -1,
        "selection_deep_rank_gt5": any(rank > 5 for rank in ranks),
        "selection_deep_rank_gt16": any(rank > 16 for rank in ranks),
        "selection_need_more": bool(selection.need_more_detail),
        "selection_followups": len(selection.followup_queries),
        "selection_digest": digest,
        "selection_latency_ms": round(float(latency_ms), 1),
    }


__all__ = [
    "MAX_FOLLOWUP_QUERIES",
    "MAX_SELECTED_CHUNKS",
    "MAX_SELECTION_CANDIDATES",
    "PREVIEW_CHARS",
    "SELECTION_ATTEMPT_BUDGET_S",
    "CandidatePreview",
    "SemanticSelection",
    "SemanticSelectionError",
    "aselect_semantic_candidates",
    "fetch_full_passages_for_selection",
    "heuristic_relevance_score",
    "heuristic_select",
    "order_pack_semantically",
    "preview_candidates",
    "rerank_previews_semantically",
    "select_by_fused_fallback",
    "selection_prompt",
    "selection_telemetry",
    "semantic_selection_schema",
    "validate_semantic_selection",
]
