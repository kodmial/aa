"""Canonical conversation context and resolved turn for audit P0-3 (kodmial/aa#305).

One serializable ``ConversationContext`` is assembled exactly once after
managed memory and before the planner; the planner consumes it and returns
a typed ``ResolvedTurn`` (raw user message, resolved intent, semantic
information needs and deterministic context digest). Only downstream stages
(selector, generator, repair/recovery, final verifier and the #304 delivery
gate) consume that completed resolved turn.

Provenance is explicit: user statements, previous assistant responses,
unresolved/ambiguous references and continuity summaries travel with roles.
Conversation memory is never book evidence and summaries can never
authenticate book claims.

When token budgeting is unavoidable, a single common relevance-preserving
view is derived from the canonical context with traceable omissions BEFORE
the stages. No stage secretly truncates first-N characters while claiming
the same ``context_digest``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

RoleName = Literal["user", "assistant"]
ProvenanceName = Literal["human", "assistant", "summary", "ambiguous"]
ReferenceStatus = Literal["resolved", "unresolved", "ambiguous"]

SNAPSHOT_VERSION = "canonical-conversation-context/1"

# Common relevance-preserving model view budget. All stages share this
# exact view (same bytes) so selector/generator/verifier prove the same
# digest. Large enough that ordinary turns travel verbatim; long inputs
# use tail-preserving truncation with an explicit marker, never silent
# first-N clipping.
CANONICAL_VIEW_CHARS = 8000
CANONICAL_MESSAGE_CHARS = 2000

# Usable history window shared by the generator display and the proactive
# summary trigger. Older continuity stays reachable via the running summary
# plus relevance promotion, never via silent first-N truncation.
USABLE_HISTORY_MESSAGES = 12

TRUNCATION_SUFFIX_FORMAT = "... [truncated {omitted} chars omitted; see original]"

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


class ConversationMessage(BaseModel):
    """One canonical history entry with role and provenance."""

    message_id: str = ""
    role: RoleName = "user"
    provenance: ProvenanceName = "human"
    text: str = ""
    order: int = 0
    reference_status: ReferenceStatus = "resolved"

    model_config = {"extra": "forbid"}


class ContextOmission(BaseModel):
    """Traceable record of one relevance/budget omission."""

    source_message_id: str = ""
    omitted_chars: int = 0
    reason: str = ""

    model_config = {"extra": "forbid"}


class ConversationContext(BaseModel):
    """Serializable canonical conversation representation (single source)."""

    user_message: str = ""
    messages: list[ConversationMessage] = Field(default_factory=list)
    summary: str = ""
    summary_provenance: str = "continuity-only-not-evidence"
    omissions: list[ContextOmission] = Field(default_factory=list)
    snapshot_version: str = SNAPSHOT_VERSION

    model_config = {"extra": "forbid"}


class InformationNeed(BaseModel):
    """One typed semantic sub-need of the current request (#311 seam).

    A planner query may map to multiple needs later; IDs are stable for the
    current resolved turn (``need-1``..``need-N`` in plan order) so graph
    replay and follow-up searches can reference them. Text/purpose are
    semantic sub-needs of the request, never prewritten advice or answers,
    and carrying an ID never claims the need is fulfilled.
    """

    need_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    purpose: str = Field(default="semantic sub-need of the current request")

    model_config = {"extra": "forbid"}


class ResolvedTurn(BaseModel):
    """Serializable planner output bound to one canonical context."""

    user_message: str = ""
    resolved_intent: str = ""
    conversation_context: dict[str, Any] = Field(default_factory=dict)
    information_needs: list[InformationNeed] = Field(default_factory=list)
    context_digest: str = Field(min_length=1)
    planner_mode: str = ""
    search_queries: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def truncate_preserving_tail(text: str, limit: int) -> str:
    """Truncate with head+tail preservation and an explicit marker.

    Unlike blind ``text[:N]``, a trailing condition/referent survives:
    the head keeps roughly two thirds and the tail keeps the final third
    so an end-of-message condition is never silently dropped. Short texts
    travel byte-identical.
    """
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    head_len = (limit * 2) // 3
    tail_len = limit - head_len
    if head_len <= 0 or tail_len <= 0:
        return text[:limit] + TRUNCATION_SUFFIX_FORMAT.format(omitted=omitted)
    return (
        text[:head_len]
        + TRUNCATION_SUFFIX_FORMAT.format(omitted=omitted)
        + text[len(text) - tail_len :]
    )


def _tokenize_generic(text: str) -> set[str]:
    return {tok.casefold() for tok in _WORD_RE.findall(text or "") if len(tok) >= 4}


def canonical_snapshot(context: ConversationContext) -> dict[str, Any]:
    """Return the deterministic snapshot covered by the context digest."""
    return {
        "user_message": str(context.user_message or ""),
        "messages": [
            {
                "message_id": str(item.message_id or ""),
                "role": str(item.role or ""),
                "provenance": str(item.provenance or ""),
                "text": str(item.text or ""),
                "order": int(item.order or 0),
                "reference_status": str(item.reference_status or "resolved"),
            }
            for item in list(context.messages or [])
        ],
        "summary": str(context.summary or ""),
        "summary_provenance": str(context.summary_provenance or ""),
        "snapshot_version": str(context.snapshot_version or SNAPSHOT_VERSION),
    }


def context_digest_for_canonical(context: ConversationContext | dict[str, Any]) -> str:
    """Return the deterministic digest of one canonical context."""
    if isinstance(context, dict):
        try:
            parsed = ConversationContext.model_validate(context)
        except Exception:
            payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        snapshot = canonical_snapshot(parsed)
    else:
        snapshot = canonical_snapshot(context)
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_conversation_context(
    messages: list[BaseMessage] | list[Any],
    summary: str,
    current_user_message: str,
) -> ConversationContext:
    """Assemble the single canonical context (full texts, no clipping).

    Full message texts are preserved so trailing conditions, long-range
    referents and topic returns stay resolvable. Budgeting happens only in
    the shared model view below, with traceable omissions.
    """
    ordered: list[ConversationMessage] = []
    clean_summary = str(summary or "")
    live = str(current_user_message or "")
    items = [item for item in (messages or []) if isinstance(item, BaseMessage)]
    # Drop the trailing duplicate of the live turn: it travels as the
    # structured live request, not duplicated from history.
    if items and live and items[-1].type == "human":
        last_content = getattr(items[-1], "content", "")
        if isinstance(last_content, str) and last_content == live:
            items = items[:-1]
    for pos, item in enumerate(items):
        content = getattr(item, "content", "")
        text = content if isinstance(content, str) else str(content)
        if item.type == "human":
            role: RoleName = "user"
            provenance: ProvenanceName = "human"
        else:
            role = "assistant"
            provenance = "assistant"
        message_id = str(getattr(item, "id", "") or f"msg-{pos}")
        ordered.append(
            ConversationMessage(
                message_id=message_id,
                role=role,
                provenance=provenance,
                text=text,
                order=pos,
                reference_status="resolved",
            )
        )
    return ConversationContext(
        user_message=live,
        messages=ordered,
        summary=clean_summary,
        summary_provenance="continuity-only-not-evidence",
        omissions=[],
        snapshot_version=SNAPSHOT_VERSION,
    )


def needs_from_plan(resolved_intent: str, queries: list[str]) -> list[InformationNeed]:
    """Derive stable typed information needs from one validated plan.

    Each distinct query becomes one semantic sub-need with a stable
    ``need-N`` id in plan order; the standalone resolved intent is always
    ``need-1`` when present so follow-up searches can reference it. IDs are
    stable through graph replay because they derive from plan order, not
    from random values. An empty (conversational) plan carries no needs.
    """
    intent = " ".join(str(resolved_intent or "").split()).strip()
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in list(queries or []):
        text = " ".join(str(raw or "").split()).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cleaned.append(text)
    needs: list[InformationNeed] = []
    if intent:
        needs.append(
            InformationNeed(
                need_id="need-1",
                text=intent,
                purpose="standalone resolved intent of the current request",
            )
        )
        for query in cleaned:
            if query.casefold() == intent.casefold():
                continue
            needs.append(
                InformationNeed(
                    need_id=f"need-{len(needs) + 1}",
                    text=query,
                    purpose="semantic sub-need of the current request",
                )
            )
            if len(needs) >= 16:
                break
    return needs


def build_resolved_turn(
    *,
    user_message: str,
    resolved_intent: str,
    context: ConversationContext,
    information_needs: list[InformationNeed] | None = None,
    planner_mode: str = "",
    search_queries: list[str] | None = None,
) -> ResolvedTurn:
    """Bind one planner decision to its canonical context with a digest."""
    digest = context_digest_for_canonical(context)
    return ResolvedTurn(
        user_message=str(user_message or ""),
        resolved_intent=" ".join(str(resolved_intent or "").split()),
        conversation_context=context.model_dump(mode="json"),
        information_needs=list(information_needs or []),
        context_digest=digest,
        planner_mode=str(planner_mode or ""),
        search_queries=list(search_queries or []),
    )


def resolved_turn_from_state(state: Any) -> ResolvedTurn | None:
    """Recover the canonical resolved turn from graph state, if present."""
    try:
        raw = state.get("resolved_turn", None)
    except Exception:
        return None
    if isinstance(raw, ResolvedTurn):
        return raw
    if isinstance(raw, dict) and raw.get("context_digest"):
        try:
            return ResolvedTurn.model_validate(raw)
        except Exception:
            return None
    return None


def conversation_context_from_state(state: Any) -> ConversationContext | None:
    """Recover the canonical context from graph state, if present."""
    try:
        raw = state.get("conversation_context", None)
    except Exception:
        return None
    if isinstance(raw, ConversationContext):
        return raw
    if isinstance(raw, dict) and raw.get("snapshot_version"):
        try:
            return ConversationContext.model_validate(raw)
        except Exception:
            return None
    return None


def context_digest_from_state(state: Any) -> str:
    """Return the canonical digest carried by graph state, if present."""
    try:
        direct = str(state.get("context_digest", "") or "").strip()
    except Exception:
        direct = ""
    if direct:
        return direct
    resolved = resolved_turn_from_state(state)
    if resolved is not None:
        return str(resolved.context_digest or "")
    context = conversation_context_from_state(state)
    if context is not None:
        return context_digest_for_canonical(context)
    return ""


def canonical_model_view(context: ConversationContext | dict[str, Any]) -> dict[str, str]:
    """Render the single shared bounded view consumed by every stage.

    Returns ``user_message`` (verbatim, never truncated), ``summary`` and
    ``conversation`` strings with identical bytes for planner, selector,
    generator, repair and verifier. Long inputs use tail-preserving
    truncation with an explicit marker; omissions stay traceable through
    the canonical ``omissions`` list rather than secret per-stage clips.
    """
    if isinstance(context, dict):
        try:
            parsed = ConversationContext.model_validate(context)
        except Exception:
            return {"user_message": "", "summary": "", "conversation": ""}
    else:
        parsed = context
    user_message = str(parsed.user_message or "")
    summary = truncate_preserving_tail(str(parsed.summary or "").strip(), CANONICAL_MESSAGE_CHARS)
    parts: list[str] = []
    for item in list(parsed.messages or []):
        label = "user" if item.role == "user" else "assistant"
        parts.append(f"{label}: {truncate_preserving_tail(item.text, CANONICAL_MESSAGE_CHARS)}")
    conversation = "\n".join(parts)
    if len(conversation) > CANONICAL_VIEW_CHARS:
        conversation = truncate_preserving_tail(conversation, CANONICAL_VIEW_CHARS)
    combined = " ".join([summary, conversation]).strip()
    if len(combined) > CANONICAL_VIEW_CHARS:
        combined = truncate_preserving_tail(combined, CANONICAL_VIEW_CHARS)
    return {
        "user_message": user_message,
        "summary": summary,
        "conversation": conversation,
        "combined": combined,
    }


def select_relevant_history(
    context: ConversationContext | dict[str, Any],
    *,
    resolved_intent: str = "",
    max_messages: int = USABLE_HISTORY_MESSAGES,
) -> list[ConversationMessage]:
    """Select the relevance-preserving history window from canonical context.

    The most recent window is always kept in order; older messages sharing
    generic lexical overlap (length >=4, casefolded, no domain tables) with
    the live request/intent are promoted so topic returns beyond the window
    stay resolvable. No message text is altered here.
    """
    if isinstance(context, dict):
        try:
            parsed = ConversationContext.model_validate(context)
        except Exception:
            return []
    else:
        parsed = context
    ordered = list(parsed.messages or [])
    if len(ordered) <= max_messages:
        return ordered
    recent = ordered[-max_messages:]
    recent_ids = {id(item) for item in recent}
    older = [item for item in ordered if id(item) not in recent_ids]
    query_text = f"{parsed.user_message or ''} {resolved_intent or ''}"
    query_tokens = _tokenize_generic(query_text)
    promoted: list[ConversationMessage] = []
    if query_tokens:
        for item in older:
            tokens = _tokenize_generic(item.text)
            if tokens & query_tokens:
                promoted.append(item)
    # Keep order, bound promotion so the window stays bounded.
    promoted = promoted[-(max_messages // 2) :] if promoted else []
    combined = promoted + recent
    # Preserve original order.
    order = {item.message_id: item.order for item in ordered}
    combined.sort(key=lambda item: order.get(item.message_id, item.order))
    return combined


def needs_proactive_summary(
    messages: list[BaseMessage] | list[Any],
    *,
    summary: str = "",
    keep_tokens: int = 40_000,
    max_messages: int = USABLE_HISTORY_MESSAGES * 2,
) -> bool:
    """Whether continuity must be summarized before history falls out.

    Triggers on model-context pressure (token estimate above the keep
    window) or on message-count pressure (history about to exceed twice the
    usable window), never solely on the legacy 120k global threshold.
    """
    items = [item for item in (messages or []) if isinstance(item, BaseMessage)]
    if len(items) >= max_messages and not str(summary or "").strip():
        return True
    if len(items) >= max_messages * 2:
        return True
    try:
        from langchain_core.messages.utils import count_tokens_approximately as _count

        tokens = int(_count(items)) if items else 0
    except Exception:
        tokens = sum(len(str(getattr(item, "content", ""))) // 4 for item in items)
    return bool(tokens >= keep_tokens)


def assert_same_context_digest(state: Any, *, stage: str) -> str:
    """Prove one stage consumes the canonical digest (fail-closed).

    Returns the canonical digest when the stage snapshot matches; raises
    ``ValueError`` when state carries conflicting context snapshots so the
    #304 finalizer boundary can fail closed instead of certifying another
    context.
    """
    canonical = context_digest_from_state(state)
    if not canonical:
        return ""
    try:
        stored_turn = state.get("resolved_turn", None)
    except Exception:
        stored_turn = None
    if isinstance(stored_turn, dict):
        turn_digest = str(stored_turn.get("context_digest", "") or "")
        if turn_digest and turn_digest != canonical:
            raise ValueError(f"context digest mismatch at {stage}")
    try:
        stored_direct = str(state.get("context_digest", "") or "")
    except Exception:
        stored_direct = ""
    if stored_direct and stored_direct != canonical:
        raise ValueError(f"context digest mismatch at {stage}")
    return canonical


__all__ = [
    "CANONICAL_MESSAGE_CHARS",
    "CANONICAL_VIEW_CHARS",
    "SNAPSHOT_VERSION",
    "TRUNCATION_SUFFIX_FORMAT",
    "USABLE_HISTORY_MESSAGES",
    "ContextOmission",
    "ConversationContext",
    "ConversationMessage",
    "InformationNeed",
    "ReferenceStatus",
    "ResolvedTurn",
    "assert_same_context_digest",
    "build_conversation_context",
    "build_resolved_turn",
    "canonical_model_view",
    "canonical_snapshot",
    "context_digest_for_canonical",
    "context_digest_from_state",
    "conversation_context_from_state",
    "needs_from_plan",
    "needs_proactive_summary",
    "resolved_turn_from_state",
    "select_relevant_history",
    "truncate_preserving_tail",
]
