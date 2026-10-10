"""Meeting invitation FSM and callback contracts (issue #327, supersedes #315).

Typed durable-when-available per-chat meeting-invitation state machine with
deterministic protocol transitions and zero LLM calls on the hot path.
"""

from __future__ import annotations

from aa.meeting_invitation.callbacks import CallbackCodec, CallbackData, StaleCallbackError
from aa.meeting_invitation.fsm import CallbackAction, MeetingFlow, MeetingState
from aa.meeting_invitation.policy import MeetingObservation, PolicyContext, decide_action, may_offer
from aa.meeting_invitation.service import CallbackOutcome, MeetingService, OfferResult, TextOutcome

__all__ = [
    "CallbackAction",
    "CallbackCodec",
    "CallbackData",
    "CallbackOutcome",
    "MeetingFlow",
    "MeetingObservation",
    "MeetingService",
    "MeetingState",
    "OfferResult",
    "PolicyContext",
    "StaleCallbackError",
    "TextOutcome",
    "decide_action",
    "may_offer",
]
