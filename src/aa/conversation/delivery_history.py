"""Keep certified but unconfirmed answers out of subsequent dialogue context."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage

DELIVERY_METADATA = "aa_delivery"


def provisional_answer(text: str, certificate_id: str) -> AIMessage:
    """Give the provisional history entry a stable, certificate-bound identity."""
    return AIMessage(
        content=text,
        id=f"answer-{certificate_id}",
        additional_kwargs={
            DELIVERY_METADATA: {"certificate_id": certificate_id, "status": "pending"}
        },
    )


def _fully_confirmed(
    message: AIMessage, receipts: Sequence[Mapping[str, Any]], thread: str
) -> bool:
    if not isinstance(message.content, str) or not message.content:
        return False
    metadata = message.additional_kwargs.get(DELIVERY_METADATA)
    if not isinstance(metadata, dict) or not metadata.get("certificate_id"):
        return False
    text = message.content
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    attempts: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        retry_id = receipt.get("retry_id")
        channel = receipt.get("channel")
        if (
            receipt.get("turn_id") == thread
            and receipt.get("certificate_id") == metadata["certificate_id"]
            and receipt.get("final_sha256") == digest
            and isinstance(retry_id, str)
            and retry_id
            and channel in ("sendMessage", "sendVoice")
        ):
            attempts.setdefault((retry_id, str(channel)), []).append(receipt)
    for attempt in attempts.values():
        # Never assemble a successful delivery from different retries, or
        # accept a confirmed prefix as confirmation of the complete answer.
        count = len(attempt)
        fields = ("segment_index", "segment_count", "char_start", "char_end")
        if any(
            receipt.get("status") != "confirmed"
            or any(type(receipt.get(field)) is not int for field in fields)
            or receipt.get("segment_count") != count
            for receipt in attempt
        ):
            continue
        ordered = sorted(attempt, key=lambda receipt: receipt["segment_index"])
        cursor = 0
        complete = True
        for index, receipt in enumerate(ordered):
            start, end = receipt["char_start"], receipt["char_end"]
            if (
                receipt["segment_index"] != index
                or not cursor <= start < end <= len(text)
                or text[cursor:start].strip()
                or receipt.get("utf8_start") != len(text[:start].encode("utf-8"))
                or receipt.get("utf8_end") != len(text[:end].encode("utf-8"))
            ):
                complete = False
                break
            cursor = end
        if complete and cursor == len(text):
            return True
    return False


def delivery_history_updates(
    messages: Sequence[BaseMessage],
    receipts: Sequence[Mapping[str, Any]],
    *,
    thread: str,
    drop_unconfirmed: bool,
) -> list[BaseMessage]:
    """Return idempotent LangGraph message updates, without exposing any text.

    Delivery commits promote confirmed answers, including answers with no
    book quotes. Immediately before the next turn, unconfirmed answers are
    removed before either summarization or context assembly can consume them.
    Partial/unknown delivery conservatively excludes the whole answer; its
    receipts and quote-exposure accounting remain independent and unchanged.
    Legacy messages without delivery metadata are preserved.
    """
    updates: list[BaseMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        metadata = message.additional_kwargs.get(DELIVERY_METADATA)
        if not isinstance(metadata, dict) or metadata.get("status") == "confirmed":
            continue
        if _fully_confirmed(message, receipts, thread):
            updates.append(
                message.model_copy(
                    update={
                        "additional_kwargs": {
                            **message.additional_kwargs,
                            DELIVERY_METADATA: {**metadata, "status": "confirmed"},
                        }
                    }
                )
            )
        elif drop_unconfirmed:
            if not message.id:
                raise ValueError("provisional history entry has no message id")
            updates.append(RemoveMessage(id=message.id))
    return updates
