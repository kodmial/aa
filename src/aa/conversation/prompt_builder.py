"""Prompt assembly contract for the future v2 answer node (issue #113).

The answer model must never infer which text is the user message and which
came from retrieval. Every answer invocation assembles input in one fixed
order:

1. the stable English system prompt;
2. recent real user/assistant messages with correct roles;
3. one application-built current-turn context payload in the final
   user-role input, with explicit XML boundaries:
   ``<conversation_memory>``, ``<book_evidence>`` (exact ``<passage>``
   elements), ``<response_budget>`` (bounded efficiency guard) and
   ``<user_message>`` last.

Backend retrieval data travels as structured context in that payload. It
is never fabricated as a native tool message when the model issued no
tool call. This module builds (but does not yet send) that contract; the
later generation task owns the answer node itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from aa.conversation.prompt_safety import escape_xml_text, quote_xml_attr
from aa.conversation.v2_prompts import load_aa_agent_system_v2


@dataclass(frozen=True)
class EvidencePassage:
    """One exact canonical passage selected for generation."""

    passage_id: str
    source: str
    section: str
    text: str


# Bounded answer-generation display (Gate C+E live repair, kodmial/aa#217
# recurrence 2 on exact main a7d76f1 run 37670332968: p50 25.2s / p95
# 33.6s / max 44.7s over the 30s budget with planner p50 3.6s / p95
# 9.2s, retrieval p50 0.4s, repair_turns=0). The prior repair bounded the
# planner per-message display and the answer passage COUNT (6-window) and
# saved ~4s p50 (29.6s -> 25.2s), but the persistent remainder
# (total - planner - retrieval ~= 21s) is the initial answer draft plus
# per-unit verifier round-trips: the answer prompt still carries full
# per-passage text plus full per-message history, so every ordinary turn
# still pays the largest provider input on the answer call, generates
# long multi-unit drafts, and then pays one verifier round-trip per
# unit. This change bounds a different dimension at the responsible
# answer-assembly boundary: per-passage characters and per-message
# characters with explicit markers. Message/passage count and order, the
# live user message and the running summary travel untruncated; the
# stored Evidence Pack, checksum/quote/cite gates and verifier verdicts
# still use full exact text, so grounding strictness is unchanged.
# Turn-independent, never an exact-question special case, Product
# Contract #110 unchanged.
#
# Gate C+E live repair, kodmial/aa#217 recurrence 7 on exact main
# 58f943c run 37709271567: answer input averages ~9k tokens per request
# and the answer stage (p50 8.9s / p95 12.5s) is the slowest single model
# call on the critical path. 600 chars keep several sentences of
# decisive context per passage with the explicit marker while cutting
# ~25% of display tokens per passage; the stored pack and all gates
# still use full exact text.
#
# kodmial/aa#244 on exact main a0d377a run 37753553708
# (C:live-book-grounding-substantive-drinking-2 plus E p50 18.9s / p95
# 24.3s, answer p50 6.6s / p95 10.0s pinned at its 10s wall over
# message-text p50 4.5s / p95 10.0s): 500 chars keeps several sentences
# of decisive context per passage with the explicit marker while cutting
# a further ~17% of display tokens per answer call. The stored pack,
# checksum/quote/cite gates and verifier verdicts still use full exact
# text, so grounding strictness is unchanged. Turn-independent, never an
# exact-question special case.
# Source passages must reach OpenCode in their entirety.
ANSWER_MAX_PASSAGE_CHARS = 0  # Deprecated compatibility constant; passage text is never clipped.

ANSWER_MAX_MESSAGE_CHARS = 1500

# Answer history window (issue #295: preserved multi-turn context).
# The last N messages travel with a wide per-message bound so short
# follow-ups, referents, topic shifts and sufficient original dialogue
# stay available for pronoun/ellipsis resolution; older continuity stays
# via the running summary and the live user message travels untruncated.
# Verification, checksum, quote and cite gates still use the full stored
# pack. Turn-independent, never an exact-question special case. Voice
# and text share this exact pipeline per #294 (voice ASR/TTS is
# transport-only, never a separate content limit).
ANSWER_MAX_HISTORY_MESSAGES = 12

ANSWER_TRUNCATION_SUFFIX_FORMAT = "... [truncated {omitted} chars omitted]"


def _display_answer_text(value: object, *, limit: int) -> str:
    """Bound one display string for answer generation with a marker.

    Uses the shared tail-preserving truncation so trailing conditions and
    referents survive; short texts travel byte-identical.
    """
    from aa.conversation.conversation_context import truncate_preserving_tail as _tail

    text = value if isinstance(value, str) else ""
    return _tail(text, limit)


def _escape_text(value: str) -> str:
    """Escape dynamic text so XML block boundaries stay literal data."""
    return escape_xml_text(value)


def render_turn_context(
    *,
    summary: str,
    passages: list[EvidencePassage],
    user_message: str,
    safety_policy: str = "",
    resolved_intent: str = "",
) -> str:
    """Render the current-turn structured context payload (user role).

    ``<resolved_intent>`` travels explicitly alongside the verbatim
    ``<user_message>`` (which stays unchanged and last) so the generator
    resolves pronouns/corrections from the planner's canonical intent
    without reinterpreting the request. ``safety_policy`` travels as a
    separate ``<safety_policy>`` control block (kodmial/aa#300), never
    merged into ``<user_message>`` or any retrieval query.
    """
    from aa.conversation.output_limits import (
        DEFAULT_GENERATION_BUDGET_TOKENS,
        generation_budget_instruction,
    )

    lines: list[str] = ["<conversation_memory>"]
    stripped = summary.strip()
    lines.append(_escape_text(stripped) if stripped else "(no prior conversation)")
    lines.append("</conversation_memory>")
    lines.append("<book_evidence>")
    if passages:
        for passage in passages:
            display = passage.text  # Preserve entire canonical evidence passage.
            lines.append(
                f"<passage id={quote_xml_attr(passage.passage_id)} "
                f"source={quote_xml_attr(passage.source)} "
                f"section={quote_xml_attr(passage.section)}>"
                f"{_escape_text(display)}</passage>"
            )
    else:
        lines.append("(no book evidence supplied for this turn)")
    lines.append("</book_evidence>")
    # Bounded generation budget (Gate C live repair, run 37561542378:
    # 14/14 ordinary turns collapsed to generic clarification with the
    # verifier never served and max 51.4s over the 30s budget. The v2
    # answer path omitted the #83 efficiency guard that the legacy path
    # carries, so weak fallback drafts ran long (many razdel units per
    # draft), making the verifier batch large, slow and flaky on
    # structured output while planner/answer served. The budget hint keeps
    # drafts to 2-5 short sentences, so verifier batches stay small and
    # fast. Turn-independent, never an exact-question special case; the
    # hard character/word/quote envelope stays authoritative).
    lines.append("<response_budget>")
    lines.append(generation_budget_instruction(DEFAULT_GENERATION_BUDGET_TOKENS))
    lines.append("</response_budget>")
    if safety_policy.strip():
        lines.append("<safety_policy>")
        lines.append(_escape_text(safety_policy.strip()))
        lines.append("</safety_policy>")
    if str(resolved_intent or "").strip():
        lines.append("<resolved_intent>")
        lines.append(_escape_text(str(resolved_intent).strip()))
        lines.append("</resolved_intent>")
    lines.append("<user_message>")
    lines.append(_escape_text(user_message))
    lines.append("</user_message>")
    return "\n".join(lines)


def build_answer_messages(
    *,
    recent: list[BaseMessage],
    summary: str,
    passages: list[EvidencePassage],
    user_message: str,
    system_prompt: str | None = None,
    safety_policy: str = "",
    resolved_intent: str = "",
    conversation_context: dict[str, Any] | None = None,
) -> list[BaseMessage]:
    """Assemble the full answer-node model input in contract order.

    History uses the shared canonical relevance-preserving window: the
    most recent messages in order plus older topic-relevant messages
    promoted by generic lexical overlap (no domain tables), so topic
    returns beyond the window stay resolvable. Per-message characters use
    tail-preserving truncation with an explicit marker. The live user
    message travels untruncated inside the structured payload alongside
    the explicit resolved intent.
    """
    system_text = system_prompt or load_aa_agent_system_v2()
    assembled: list[BaseMessage] = [SystemMessage(content=system_text)]
    # Canonical relevance-preserving window (#305): recent window plus
    # promoted older topic-relevant messages from the shared context when
    # available; legacy most-recent slicing otherwise.
    windowed: list[BaseMessage] = list(recent or [])
    if conversation_context is not None:
        try:
            from aa.conversation.conversation_context import select_relevant_history as _select

            selected = _select(
                conversation_context,
                resolved_intent=str(resolved_intent or ""),
                max_messages=ANSWER_MAX_HISTORY_MESSAGES,
            )
            if selected:
                by_id: dict[str, BaseMessage] = {}
                for message in windowed:
                    mid = str(getattr(message, "id", "") or "")
                    if mid:
                        by_id[mid] = message
                rebuilt: list[BaseMessage] = []
                for entry in selected:
                    match = by_id.get(str(entry.message_id or ""))
                    if match is not None:
                        rebuilt.append(match)
                    else:
                        # Canonical entry without a live message object:
                        # reconstruct a role-correct message so the bytes
                        # match the shared canonical view.
                        if entry.role == "user":
                            rebuilt.append(HumanMessage(content=entry.text))
                        else:
                            rebuilt.append(AIMessage(content=entry.text))
                if rebuilt:
                    windowed = rebuilt
            elif len(windowed) > ANSWER_MAX_HISTORY_MESSAGES:
                windowed = windowed[-ANSWER_MAX_HISTORY_MESSAGES:]
        except Exception:
            if len(windowed) > ANSWER_MAX_HISTORY_MESSAGES:
                windowed = windowed[-ANSWER_MAX_HISTORY_MESSAGES:]
    elif len(windowed) > ANSWER_MAX_HISTORY_MESSAGES:
        windowed = windowed[-ANSWER_MAX_HISTORY_MESSAGES:]
    for message in windowed:
        if message.type == "human":
            content = message.content
            text = content if isinstance(content, str) else str(content)
            assembled.append(
                HumanMessage(content=_display_answer_text(text, limit=ANSWER_MAX_MESSAGE_CHARS))
            )
        elif message.type == "ai":
            content = message.content
            text = content if isinstance(content, str) else str(content)
            assembled.append(
                AIMessage(content=_display_answer_text(text, limit=ANSWER_MAX_MESSAGE_CHARS))
            )
    assembled.append(
        HumanMessage(
            content=render_turn_context(
                summary=summary,
                passages=passages,
                user_message=user_message,
                safety_policy=safety_policy,
                resolved_intent=resolved_intent,
            )
        )
    )
    return assembled


__all__ = [
    "ANSWER_MAX_HISTORY_MESSAGES",
    "ANSWER_MAX_MESSAGE_CHARS",
    "ANSWER_MAX_PASSAGE_CHARS",
    "ANSWER_TRUNCATION_SUFFIX_FORMAT",
    "EvidencePassage",
    "build_answer_messages",
    "render_turn_context",
]
