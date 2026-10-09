"""Natural AA answer generation for the v2 turn pipeline.

Orchestration has already supplied the Evidence Pack. The user-facing AA
Agent never calls book-search tools to make primary retrieval happen: it
receives the stable English system prompt, real recent user/assistant
messages with correct roles, and one structured current-turn payload
(``<conversation_memory>``, ``<book_evidence>``, ``<user_message>``
last) assembled by the prompt-assembly contract.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from aa.conversation.evidence_integrity import validate_book_pack_for_model_use
from aa.conversation.prompt_builder import EvidencePassage, build_answer_messages
from aa.conversation.retrieval_node import state_passages_to_prompt
from aa.conversation.v2_prompts import load_aa_agent_system_v2

logger = logging.getLogger("aa.conversation.answer_node")


def recent_history(
    messages: Sequence[BaseMessage], *, current_user_message: str
) -> list[BaseMessage]:
    """Return prior real messages without duplicating the live turn."""
    recent = [item for item in messages if isinstance(item, BaseMessage)]
    if recent and recent[-1].type == "human" and current_user_message:
        last_text = recent[-1].content
        if isinstance(last_text, str) and last_text == current_user_message:
            recent = recent[:-1]
    return recent


def _reply_text(reply: object) -> str:
    if isinstance(reply, BaseMessage):
        content = reply.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(str(item.get("text")))
            return "\n".join(parts)
        return str(content)
    text = getattr(reply, "content", reply)
    return text if isinstance(text, str) else str(text)


async def generate_draft(
    *,
    model: Any,
    recent: Sequence[BaseMessage],
    summary: str,
    passages: list[EvidencePassage],
    user_message: str,
    safety_policy: str = "",
) -> str:
    """Generate one natural grounded Russian draft from the Evidence Pack.

    ``safety_policy`` is a separate control block (kodmial/aa#300), never
    merged into ``user_message``; empty for ordinary drafts.
    """
    if not user_message.strip():
        raise ValueError("refusing to generate an answer without a user message")
    # Structural integrity before any answer model call (kodmial/aa#310):
    # relied-upon passages must carry real identities and text. Full
    # checksum/source/version/range validation runs on the stored pack
    # dicts upstream (generate_draft_from_state / turn pipeline); an
    # empty list is the legitimate no-book path and passes.
    for passage in list(passages or []):
        if not isinstance(passage.passage_id, str) or not passage.passage_id.strip():
            raise ValueError("answer evidence has missing passage_id")
        if not isinstance(passage.text, str) or not passage.text:
            raise ValueError("answer evidence has missing text")
        if not isinstance(passage.source, str) or not passage.source.strip():
            raise ValueError("answer evidence has missing source")
        if not isinstance(passage.section, str) or not passage.section.strip():
            raise ValueError("answer evidence has missing section")
    system_text = load_aa_agent_system_v2()
    messages = build_answer_messages(
        recent=list(recent),
        summary=summary,
        passages=passages,
        user_message=user_message,
        system_prompt=system_text,
        safety_policy=safety_policy,
    )
    reply = await model.ainvoke(messages)
    text = _reply_text(reply).strip()
    if not text:
        raise ValueError("answer model returned an empty draft")
    logger.info("v2 answer draft generated")
    return text


async def generate_draft_from_state(
    state: Any,
    *,
    model: Any,
    user_message_override: str | None = None,
) -> str:
    """Generate one draft directly from graph state passages."""
    current = str(user_message_override or state.get("current_user_message", ""))
    summary = str(state.get("conversation_summary", ""))
    messages = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
    pack_dicts = [item for item in state.get("evidence_pack", []) if isinstance(item, dict)]
    # Authoritative pack integrity before any answer model call: missing
    # checksum/source/version/section/range or wrong hash fails here,
    # never as a skipped passage. Empty pack (glue) passes.
    validate_book_pack_for_model_use(pack_dicts)
    passages = state_passages_to_prompt(pack_dicts)
    return await generate_draft(
        model=model,
        recent=recent_history(
            messages, current_user_message=str(state.get("current_user_message", ""))
        ),
        summary=summary,
        passages=passages,
        user_message=current,
    )


__all__ = ["generate_draft", "generate_draft_from_state", "recent_history"]
