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
            lines.append(
                f"<passage id={_xml_quoteattr(passage.passage_id)} "
                f"source={_xml_quoteattr(passage.source)} "
                f"section={_xml_quoteattr(passage.section)}>"
                f"{_escape_text(passage.text)}</passage>"
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
    """Assemble the full answer-node model input in contract order."""
    system_text = system_prompt or load_aa_agent_system_v2()
    assembled: list[BaseMessage] = [SystemMessage(content=system_text)]
    for message in recent:
        if message.type == "human":
            content = message.content
            assembled.append(
                HumanMessage(content=content if isinstance(content, str) else str(content))
            )
        elif message.type == "ai":
            content = message.content
            assembled.append(
                AIMessage(content=content if isinstance(content, str) else str(content))
            )
    assembled.append(
        HumanMessage(
            content=render_turn_context(
                summary=summary, passages=passages, user_message=user_message
            )
        )
    )
    return assembled


__all__ = ["EvidencePassage", "build_answer_messages", "render_turn_context"]
