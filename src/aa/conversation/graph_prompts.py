"""Prompt assembly contract for the future answer node (issue #113).

Order is fixed:

1. stable English system prompt;
2. recent real user/assistant messages with correct roles;
3. one application-built current-turn context payload in the final
   user-role input with ``<conversation_memory>``, ``<book_evidence>``,
   and ``<user_message>`` last.

Retrieval output travels as structured context, never as a fabricated native
tool message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

AA_AGENT_PROMPT_VERSION = "aa-agent-system-v2"


def _prompts_dir() -> Path:
    """Return the repository ``prompts/`` directory."""
    return Path(__file__).resolve().parents[3] / "prompts"


def load_aa_agent_prompt() -> str:
    """Load the versioned English production AA Agent prompt (#112)."""
    return (_prompts_dir() / f"{AA_AGENT_PROMPT_VERSION}.md").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class EvidencePassage:
    """One exact canonical passage selected for generation."""

    passage_id: str
    source: str
    section: str
    text: str


def render_evidence_block(passages: list[EvidencePassage]) -> str:
    """Render ``<book_evidence>`` with one exact element per passage."""
    if not passages:
        return "<book_evidence>\n</book_evidence>"
    lines = ["<book_evidence>"]
    for passage in passages:
        lines.append(
            f'<passage id="{passage.passage_id}" '
            f'source="{passage.source}" section="{passage.section}">'
            f"{passage.text}</passage>"
        )
    lines.append("</book_evidence>")
    return "\n".join(lines)


def render_context_payload(
    *,
    conversation_memory: str,
    evidence: list[EvidencePassage],
    user_message: str,
) -> str:
    """Render the final user-role payload with the message strictly last."""
    if not user_message.strip():
        raise ValueError("user_message must not be empty")
    memory_block = f"<conversation_memory>\n{conversation_memory.strip()}\n</conversation_memory>"
    evidence_block = render_evidence_block(evidence)
    return (
        f"{memory_block}\n\n{evidence_block}\n\n"
        f"<user_message>\n{user_message.strip()}\n</user_message>"
    )


def build_answer_messages(
    *,
    system_prompt: str | None = None,
    recent_messages: list[BaseMessage] | None = None,
    conversation_memory: str = "",
    evidence: list[EvidencePassage] | None = None,
    user_message: str,
) -> list[BaseMessage]:
    """Assemble answer-node input with unambiguous structure and roles."""
    system_text = system_prompt if system_prompt is not None else load_aa_agent_prompt()
    if not system_text.strip():
        raise ValueError("system prompt must not be empty")
    payload = render_context_payload(
        conversation_memory=conversation_memory,
        evidence=list(evidence) if evidence else [],
        user_message=user_message,
    )
    assembled: list[BaseMessage] = [SystemMessage(content=system_text)]
    for message in list(recent_messages) if recent_messages else []:
        assembled.append(message)
    # Structured backend context travels as the final human input, never as
    # a fabricated tool message.
    assembled.append(HumanMessage(content=payload))
    return assembled


__all__ = [
    "AA_AGENT_PROMPT_VERSION",
    "EvidencePassage",
    "build_answer_messages",
    "load_aa_agent_prompt",
    "render_context_payload",
    "render_evidence_block",
]
