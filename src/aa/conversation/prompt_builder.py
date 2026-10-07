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
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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
ANSWER_MAX_PASSAGE_CHARS = 800

ANSWER_MAX_MESSAGE_CHARS = 500

ANSWER_TRUNCATION_SUFFIX_FORMAT = "... [truncated {omitted} chars omitted]"


def _display_answer_text(value: object, *, limit: int) -> str:
    """Bound one display string for answer generation with a marker."""
    text = value if isinstance(value, str) else ""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + ANSWER_TRUNCATION_SUFFIX_FORMAT.format(omitted=omitted)


def _escape_text(value: str) -> str:
    """Escape dynamic text so XML block boundaries stay literal data."""
    return _xml_escape(value, {"'": "&apos;", '"': "&quot;"})


def render_turn_context(
    *,
    summary: str,
    passages: list[EvidencePassage],
    user_message: str,
) -> str:
    """Render the current-turn structured context payload (user role).

    ``<user_message>`` is always last so the real request stays
    structurally unambiguous.
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
            display = _display_answer_text(passage.text, limit=ANSWER_MAX_PASSAGE_CHARS)
            lines.append(
                f"<passage id={_xml_quoteattr(passage.passage_id)} "
                f"source={_xml_quoteattr(passage.source)} "
                f"section={_xml_quoteattr(passage.section)}>"
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
) -> list[BaseMessage]:
    """Assemble the full answer-node model input in contract order.

    Recent history keeps message count and order (full history still sent
    for pronoun/ellipsis resolution); only per-message characters are
    display-bounded with an explicit marker. The live user message travels
    untruncated inside the structured payload.
    """
    system_text = system_prompt or load_aa_agent_system_v2()
    assembled: list[BaseMessage] = [SystemMessage(content=system_text)]
    for message in recent:
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
                summary=summary, passages=passages, user_message=user_message
            )
        )
    )
    return assembled


__all__ = [
    "ANSWER_MAX_MESSAGE_CHARS",
    "ANSWER_MAX_PASSAGE_CHARS",
    "ANSWER_TRUNCATION_SUFFIX_FORMAT",
    "EvidencePassage",
    "build_answer_messages",
    "render_turn_context",
]
