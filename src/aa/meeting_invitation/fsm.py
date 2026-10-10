"""Typed meeting-invitation FSM states and deterministic transitions.

Zero LLM calls: all navigation, calendar, and locality logic is ordinary
code. One accepted incoming event produces at most one business-state
transition. Terminal states never self-trigger another transition.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

MeetingStateName = Literal[
    "IDLE",
    "OFFERED",
    "CHOOSE_FORMAT",
    "AWAIT_CITY",
    "CLARIFY_CITY",
    "SHOW_RESULTS",
    "DONE",
    "DECLINED",
    "CANCELLED",
    "EXPIRED",
    "FAILED",
]

ACTIVE_STATES: frozenset[str] = frozenset(
    {"OFFERED", "CHOOSE_FORMAT", "AWAIT_CITY", "CLARIFY_CITY", "SHOW_RESULTS"}
)
TERMINAL_STATES: frozenset[str] = frozenset({"DONE", "DECLINED", "CANCELLED", "EXPIRED", "FAILED"})

MeetingFormat = Literal["unknown", "online", "in_person"]

MAX_CITY_ATTEMPTS = 2
MAX_CLARIFY_CHOICES = 4
OFFER_TTL_SECONDS = 24 * 3600
RESULTS_PAGE_SIZE = 3


class MeetingState(StrEnum):
    """Serializable invitation flow states."""

    IDLE = "IDLE"
    OFFERED = "OFFERED"
    CHOOSE_FORMAT = "CHOOSE_FORMAT"
    AWAIT_CITY = "AWAIT_CITY"
    CLARIFY_CITY = "CLARIFY_CITY"
    SHOW_RESULTS = "SHOW_RESULTS"
    DONE = "DONE"
    DECLINED = "DECLINED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class CallbackAction(StrEnum):
    """Bounded protocol actions carried by signed callback tokens."""

    ACCEPT = "accept"
    DECLINE = "decline"
    ONLINE = "online"
    IN_PERSON = "in_person"
    BACK = "back"
    CANCEL = "cancel"
    MORE = "more"
    OTHER_CITY = "other_city"
    CHOOSE_PLACE = "choose_place"
    NEW_SEARCH = "new_search"


@dataclass
class ResultsSnapshot:
    """Bounded deterministic results cursor for SHOW_RESULTS paging."""

    place_id: str | None = None
    region_id: str | None = None
    meeting_format: str = "online"
    now_iso: str = ""
    cursor: str | None = None
    directory_version: str = ""
    directory_digest: str = ""
    occurrence_keys: tuple[str, ...] = ()
    exhausted: bool = False
    fallback: str | None = None
    directory_link: str | None = None


@dataclass
class MeetingFlow:
    """One per-chat invitation flow (serializable, minimum stored)."""

    chat_hash: str = ""
    flow_id: str = ""
    flow_seq: int = 0
    generation: int = 0
    offer_id: str = ""
    state: MeetingState = MeetingState.IDLE
    meeting_format: MeetingFormat = "unknown"
    place_id: str | None = None
    city_attempts: int = 0
    region_clarifications: int = 0
    results: ResultsSnapshot = field(default_factory=ResultsSnapshot)
    candidates: tuple[dict[str, str], ...] = ()
    bound_message_id: int | None = None
    pending_delivery: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    consumed_actions: tuple[str, ...] = ()
    version: int = 1
    nonce: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe snapshot of the flow."""
        return {
            "flow_id": self.flow_id,
            "flow_seq": self.flow_seq,
            "generation": self.generation,
            "offer_id": self.offer_id,
            "state": self.state.value,
            "meeting_format": self.meeting_format,
            "place_id": self.place_id,
            "city_attempts": self.city_attempts,
            "region_clarifications": self.region_clarifications,
            "results": {
                "place_id": self.results.place_id,
                "region_id": self.results.region_id,
                "meeting_format": self.results.meeting_format,
                "now_iso": self.results.now_iso,
                "cursor": self.results.cursor,
                "directory_version": self.results.directory_version,
                "directory_digest": self.results.directory_digest,
                "occurrence_keys": list(self.results.occurrence_keys),
                "exhausted": self.results.exhausted,
                "fallback": self.results.fallback,
                "directory_link": self.results.directory_link,
            },
            "candidates": [dict(item) for item in self.candidates],
            "bound_message_id": self.bound_message_id,
            "pending_delivery": self.pending_delivery,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "consumed_actions": list(self.consumed_actions),
            "version": self.version,
            "nonce": self.nonce,
        }


def new_flow(
    chat_hash: str, generation: int, flow_seq: int, now: float | None = None
) -> MeetingFlow:
    """Create a fresh IDLE flow for one chat generation."""
    moment = float(now if now is not None else time.time())
    return MeetingFlow(
        chat_hash=chat_hash,
        flow_id=uuid.uuid4().hex,
        flow_seq=int(flow_seq),
        generation=int(generation),
        offer_id="",
        state=MeetingState.IDLE,
        created_at=moment,
        updated_at=moment,
        nonce=uuid.uuid4().hex[:16],
    )


def is_terminal(state: MeetingState) -> bool:
    """Whether the state ends automatic wizard work."""
    return state.value in TERMINAL_STATES


def is_active(state: MeetingState) -> bool:
    """Whether the state still accepts wizard navigation."""
    return state.value in ACTIVE_STATES


def is_expired(flow: MeetingFlow, now: float) -> bool:
    """Lazy 24h inactivity expiry for a non-terminal flow."""
    if is_terminal(flow.state) or flow.state == MeetingState.IDLE:
        return False
    return (float(now) - float(flow.updated_at)) >= float(OFFER_TTL_SECONDS)


def back_target(state: MeetingState) -> MeetingState | None:
    """One well-defined previous step for the back action."""
    mapping: dict[MeetingState, MeetingState] = {
        MeetingState.SHOW_RESULTS: MeetingState.CHOOSE_FORMAT,
        MeetingState.CLARIFY_CITY: MeetingState.AWAIT_CITY,
        MeetingState.AWAIT_CITY: MeetingState.CHOOSE_FORMAT,
        MeetingState.CHOOSE_FORMAT: MeetingState.OFFERED,
    }
    return mapping.get(state)


def mark_consumed(flow: MeetingFlow, token: str) -> MeetingFlow:
    """Record one consumed callback token (idempotent replay guard)."""
    if token in flow.consumed_actions:
        return flow
    kept = (*flow.consumed_actions, token)[-64:]
    flow.consumed_actions = kept
    return flow


__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "MAX_CITY_ATTEMPTS",
    "MAX_CLARIFY_CHOICES",
    "OFFER_TTL_SECONDS",
    "RESULTS_PAGE_SIZE",
    "CallbackAction",
    "MeetingFlow",
    "MeetingFormat",
    "MeetingState",
    "MeetingStateName",
    "ResultsSnapshot",
    "back_target",
    "is_active",
    "is_expired",
    "is_terminal",
    "mark_consumed",
    "new_flow",
]
