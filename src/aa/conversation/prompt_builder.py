"""Prompt assembly contract for the future v2 answer node (issue #113).

The answer model must never infer which text is the user message and which
came from retrieval. Every answer invocation assembles input in one fixed
order:

1. the stable English system prompt;
2. recent real user/assistant messages with correct roles;
3. one application-built current-turn context payload in the final
   user-role input, with explicit XML boundaries:
   ``<conversation_memory>``, ``<book_evidence>`` (exact ``<passage>``
   elements) and ``<user_message>`` last.

Backend retrieval data travels as structured context in that payload. It
is never fabricated as a native tool message when the model issued no
tool call. This module builds (but does not yet send) that contract; the
later generation task owns the answer node itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from aa.conversation.v2_prompts import load_aa_agent_system_v2


@dataclass(frozen=True)
class EvidencePassage:
    """One exact canonical passage selected for generation."""

    passage_id: str
    source: str
    section: str
    text: str


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
    lines: list[str] = ["<conversation_memory>"]
    lines.append(summary.strip() if summary.strip() else "(no prior conversation)")
    lines.append("</conversation_memory>")
    lines.append("<book_evidence>")
    if passages:
        for passage in passages:
            lines.append(
                f'<passage id="{passage.passage_id}" '
                f'source="{passage.source}" section="{passage.section}">'
                f"{passage.text}</passage>"
            )
    else:
        lines.append("(no book evidence supplied for this turn)")
    lines.append("</book_evidence>")
    lines.append("<user_message>")
    lines.append(user_message)
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
