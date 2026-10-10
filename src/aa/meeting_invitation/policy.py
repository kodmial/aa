"""Deterministic meeting policy after semantic interpretation.

The model determines when an invitation is relevant; this module enforces
valid transitions, prior decline, pending-action lifecycle, and
safety/transport constraints in ordinary code. It never parses user text
with keyword lists: callers pass a typed observation already interpreted
by the planner/model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MeetingObservationEvent = Literal[
    "none",
    "accepts_offer",
    "declines_offer",
    "postpones",
    "asks_to_find",
    "expresses_concern",
    "reports_attendance",
]

MeetingAction = Literal[
    "none",
    "offer_meeting",
    "request_format",
    "provide_resource",
    "ask_locality",
    "acknowledge_decline",
    "address_concern",
]


@dataclass(frozen=True)
class MeetingObservation:
    """Typed semantic observation for one turn (model-produced)."""

    event: MeetingObservationEvent = "none"
    related_offer_turn_id: str = ""
    meeting_format: str = ""
    place_query: str = ""


@dataclass(frozen=True)
class PolicyContext:
    """Code-owned constraints for one policy decision."""

    invitation_state: str = "IDLE"
    has_pending_offer: bool = False
    declined_this_session: bool = False
    safety_blocked: bool = False
    transport_failed: bool = False
    directory_available: bool = True
    already_attends: bool = False


def decide_action(observation: MeetingObservation, context: PolicyContext) -> MeetingAction:
    """Select one deterministic meeting action for a typed observation."""
    if context.safety_blocked or context.transport_failed:
        return "none"
    event = observation.event
    if event == "none":
        return "none"
    if event in ("declines_offer", "postpones"):
        return "acknowledge_decline"
    if event == "reports_attendance":
        return "none"
    if event == "expresses_concern":
        return "address_concern"
    if event == "asks_to_find":
        if context.has_pending_offer:
            return "request_format"
        if context.declined_this_session:
            return "none"
        if not context.directory_available:
            return "none"
        return "provide_resource"
    if event == "accepts_offer":
        if context.has_pending_offer:
            return "request_format"
        if context.declined_this_session:
            return "none"
        return "none"
    return "none"


def may_offer(context: PolicyContext) -> bool:
    """Whether a fresh proactive offer is permitted by code constraints."""
    if context.safety_blocked or context.transport_failed:
        return False
    if context.has_pending_offer:
        return False
    if context.declined_this_session:
        return False
    if context.already_attends:
        return False
    if not context.directory_available:
        return False
    return context.invitation_state in ("IDLE", "DONE", "CANCELLED", "EXPIRED")


__all__ = [
    "MeetingAction",
    "MeetingObservation",
    "MeetingObservationEvent",
    "PolicyContext",
    "decide_action",
    "may_offer",
]
