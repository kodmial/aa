"""P0 kodmial/aa#360: wizard callback tokens must never alias across generations.

Proven systemic Gate A failure on exact main 0e8f6d4 (run 38232291747):

- A:shell-step-failed:gate-a-shell via
  tests/test_meeting_invitation.py::test_back_and_cancel_from_every_active_state
  (cancel from a presumed CHOOSE_FORMAT landed as DECLINED from OFFERED).

Root cause (failure class, not one FSM branch): callback tokens carry an
8-bit salt minted with ``secrets.randbits(8)``. Under a frozen clock (or a
same-second production burst) every accept token for one flow shares
action/seq/generation/message/expiry and differs only in that salt, so at
most 256 distinct tokens exist per semantic identity. A fresh accept after
back-navigation could byte-match an already-consumed token and be dropped
as a duplicate press, leaving the flow one step behind. Across a 1968-test
suite minting hundreds of tokens per run, recurrence is structural.

Repair (generic, no FSM branch tweaks): per-codec monotonic salt plus a
per-service monotonic mint nonce, so distinct button generations are
pairwise distinct within the 64-entry dedup window. True double-presses
(same token string twice) are still rejected as duplicates.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aa.meeting_directory.loader import reset_cache
from aa.meeting_invitation.callbacks import CallbackCodec
from aa.meeting_invitation.fsm import MeetingState
from aa.meeting_invitation.service import MeetingService

TUESDAY_NOON = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _service() -> MeetingService:
    reset_cache()
    return MeetingService(secret=b"test-secret-32-bytes-padding-0001", clock=lambda: TUESDAY_NOON)


def test_codec_mints_distinct_tokens_under_frozen_clock() -> None:
    codec = CallbackCodec(secret=b"test-secret-32-bytes-padding-0001")
    seen: set[str] = set()
    for _ in range(300):
        seen.add(
            codec.encode(
                action="accept",
                flow_seq=1,
                generation=0,
                message_id=100,
                now=TUESDAY_NOON.timestamp(),
            )
        )
    # More mints than the 8-bit salt space must still alias at the salt
    # cycle boundary, but the first full salt cycle is pairwise distinct.
    assert len(seen) >= 256


def test_service_buttons_are_distinct_across_generations() -> None:
    svc = _service()
    svc.create_offer(7, 100)
    flow = svc.flow_for(7)
    assert flow is not None
    tokens = {
        svc._token(action="accept", flow=flow, message_id=100)  # noqa: SLF001
        for _ in range(64)
    }
    assert len(tokens) == 64


def test_back_accept_cancel_cycle_never_aliases_as_duplicate() -> None:
    # The exact CI failure shape, generalized: any number of
    # back -> accept rounds must leave the flow in CHOOSE_FORMAT, so the
    # following cancel always reaches CANCELLED (never a stale OFFERED).
    # Every press goes through the production service mint seam.
    for chat in range(20):
        svc = _service()
        svc.create_offer(chat, 100)
        flow = svc.flow_for(chat)
        assert flow is not None
        first = svc._token(action="accept", flow=flow, message_id=100)  # noqa: SLF001
        out = svc.handle_callback(chat, chat, 100, first)
        assert out.accepted
        for _ in range(3):
            flow = svc.flow_for(chat)
            assert flow is not None
            back = svc._token(action="back", flow=flow, message_id=100)  # noqa: SLF001
            assert svc.handle_callback(chat, chat, 100, back).accepted
            flow = svc.flow_for(chat)
            assert flow is not None
            assert flow.state == MeetingState.OFFERED
            accept = svc._token(action="accept", flow=flow, message_id=100)  # noqa: SLF001
            again = svc.handle_callback(chat, chat, 100, accept)
            assert again.accepted, "fresh accept must not be dropped as duplicate"
            flow = svc.flow_for(chat)
            assert flow is not None
            assert flow.state == MeetingState.CHOOSE_FORMAT
        flow = svc.flow_for(chat)
        assert flow is not None
        cancel = svc._token(action="cancel", flow=flow, message_id=100)  # noqa: SLF001
        done = svc.handle_callback(chat, chat, 100, cancel)
        assert done.accepted
        flow = svc.flow_for(chat)
        assert flow is not None
        assert flow.state == MeetingState.CANCELLED


def test_true_double_press_is_still_rejected_as_duplicate() -> None:
    svc = _service()
    svc.create_offer(7, 100)
    flow = svc.flow_for(7)
    assert flow is not None
    token = svc._token(action="accept", flow=flow, message_id=100)  # noqa: SLF001
    first = svc.handle_callback(7, 7, 100, token)
    assert first.accepted
    second = svc.handle_callback(7, 7, 100, token)
    assert not second.accepted and second.stale
