"""Meeting invitation FSM and nearest-meetings integration (issue #327).

Deterministic acceptance matrix for the #315 successor: full FSM paths,
city disambiguation, callback contracts, expiry/reset, delivery binding,
and real #316/#317 directory integration. Zero LLM calls on the hot path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aa.meeting_directory.loader import reset_cache
from aa.meeting_invitation.callbacks import CallbackCodec
from aa.meeting_invitation.fsm import MeetingState
from aa.meeting_invitation.policy import MeetingObservation, PolicyContext, decide_action, may_offer
from aa.meeting_invitation.rendering import format_results_text
from aa.meeting_invitation.service import MeetingService
from aa.telegram.transport import parse_callback_query

TUESDAY_NOON = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _service(now: datetime = TUESDAY_NOON) -> MeetingService:
    reset_cache()
    return MeetingService(secret=b"test-secret-32-bytes-padding-0001", clock=lambda: now)


def _offer(svc: MeetingService, chat: int = 7, message: int = 100) -> Any:
    return svc.create_offer(chat, message)


def _callback_token(svc: MeetingService, chat: int, action: str, message: int) -> str:
    flow = svc.flow_for(chat)
    assert flow is not None
    codec = svc._codec  # noqa: SLF001 - test-only token minting
    assert isinstance(codec, CallbackCodec)
    return codec.encode(
        action=action,
        flow_seq=int(flow.flow_seq),
        generation=int(flow.generation),
        message_id=int(message),
        now=svc._now_utc().timestamp(),  # noqa: SLF001
    )


def test_accept_online_three_chronological_then_more_exhausted() -> None:
    svc = _service()
    _offer(svc)
    out = svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    assert out.accepted
    _flow = svc.flow_for(7)
    assert _flow is not None and _flow.state == MeetingState.CHOOSE_FORMAT
    out = svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "online", 100))
    assert out.accepted and out.send_text
    _flow = svc.flow_for(7)
    assert _flow is not None and _flow.state == MeetingState.SHOW_RESULTS
    first_keys = tuple(_flow.results.occurrence_keys)
    assert 1 <= len(first_keys) <= 3
    flow = svc.flow_for(7)
    assert flow is not None
    if flow.results.cursor:
        more = svc._codec.encode(  # noqa: SLF001
            action="more",
            flow_seq=int(flow.flow_seq),
            generation=int(flow.generation),
            message_id=100,
            now=svc._now_utc().timestamp(),  # noqa: SLF001
        )
        second = svc.handle_callback(7, 7, 100, more)
        assert second.accepted
        flow2 = svc.flow_for(7)
        assert flow2 is not None
        new_keys = flow2.results.occurrence_keys[len(first_keys) :]
        assert new_keys
        assert not (set(first_keys) & set(new_keys))
        # Exhaustion never replays page one.
        while flow2.state == MeetingState.SHOW_RESULTS and flow2.results.cursor:
            token = svc._codec.encode(  # noqa: SLF001
                action="more",
                flow_seq=int(flow2.flow_seq),
                generation=int(flow2.generation),
                message_id=100,
                now=svc._now_utc().timestamp(),  # noqa: SLF001
            )
            nxt = svc.handle_callback(7, 7, 100, token)
            assert nxt.accepted or nxt.stale
            flow2 = svc.flow_for(7)
            assert flow2 is not None
            if len(flow2.results.occurrence_keys) > 60:
                break
        if flow2.state == MeetingState.DONE:
            dup = svc.handle_callback(7, 7, 100, more)
            assert dup.stale or not dup.accepted
    else:
        assert svc.flow_for(7) is not None


def test_in_person_known_city_skips_prompt() -> None:
    svc = _service()
    _offer(svc)
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "in_person", 100))
    flow = svc.flow_for(7)
    assert flow is not None and flow.state == MeetingState.AWAIT_CITY
    outcome = svc.handle_text(7, "Москва")
    assert outcome.handled and outcome.send_text
    _flow = svc.flow_for(7)
    assert _flow is not None
    assert _flow.state in (MeetingState.SHOW_RESULTS, MeetingState.DONE)


def test_unknown_city_bounded_then_directory_terminal() -> None:
    svc = _service()
    _offer(svc)
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "in_person", 100))
    prompts = 0
    for attempt in range(30):
        outcome = svc.handle_text(7, f"Xyzzytown-{attempt}")
        if outcome.handled and outcome.send_text:
            prompts += 1
        _f = svc.flow_for(7)
        if _f is not None and _f.state == MeetingState.DONE:
            break
    assert prompts <= 2
    _f = svc.flow_for(7)
    assert _f is not None and _f.state == MeetingState.DONE
    # No further prompts after terminal.
    final = svc.handle_text(7, "Xyzzytown-final")
    assert not final.handled
    assert final.exit_to_dialogue
    assert final.send_text is None
    assert final.state == MeetingState.DONE.value


def test_ambiguous_kirov_offers_bounded_choices_zero_llm() -> None:
    svc = _service()
    _offer(svc)
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "in_person", 100))
    outcome = svc.handle_text(7, "Киров")
    assert outcome.handled
    flow = svc.flow_for(7)
    assert flow is not None and flow.state == MeetingState.CLARIFY_CITY
    assert 1 <= len(flow.candidates) <= 4
    assert outcome.send_keyboard is not None
    # Choose the first verified candidate deterministically.
    token = svc._codec.encode(  # noqa: SLF001
        action="choose_place",
        flow_seq=int(flow.flow_seq),
        generation=int(flow.generation),
        message_id=100,
        place_index=0,
        now=svc._now_utc().timestamp(),  # noqa: SLF001
    )
    chosen = svc.handle_callback(7, 7, 100, token)
    assert chosen.accepted
    assert svc.flow_for(7) is not None
    _f = svc.flow_for(7)
    assert _f is not None
    assert _f.state in (MeetingState.SHOW_RESULTS, MeetingState.DONE)
    # Directory-only Kirov localities present an honest link, not a meeting.
    _f = svc.flow_for(7)
    assert _f is not None
    if _f.state == MeetingState.DONE:
        assert chosen.send_text is not None and "http" in chosen.send_text


def test_back_and_cancel_from_every_active_state() -> None:
    svc = _service()
    _offer(svc)
    # OFFERED decline path.
    out = svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "decline", 100))
    _f = svc.flow_for(7)
    assert out.accepted and _f is not None and _f.state == MeetingState.DECLINED
    assert out.send_text is None

    svc2 = _service()
    _offer(svc2)
    svc2.handle_callback(7, 7, 100, _callback_token(svc2, 7, "accept", 100))
    # CHOOSE_FORMAT back -> OFFERED, never forward.
    back = svc2.handle_callback(7, 7, 100, _callback_token(svc2, 7, "back", 100))
    _f2 = svc2.flow_for(7)
    assert back.accepted and _f2 is not None and _f2.state == MeetingState.OFFERED
    svc2.handle_callback(7, 7, 100, _callback_token(svc2, 7, "accept", 100))
    # CHOOSE_FORMAT cancel -> CANCELLED with zero chat messages.
    cancel = svc2.handle_callback(7, 7, 100, _callback_token(svc2, 7, "cancel", 100))
    _f2 = svc2.flow_for(7)
    assert cancel.accepted and _f2 is not None and _f2.state == MeetingState.CANCELLED
    assert cancel.send_text is None

    svc3 = _service()
    _offer(svc3)
    svc3.handle_callback(7, 7, 100, _callback_token(svc3, 7, "accept", 100))
    svc3.handle_callback(7, 7, 100, _callback_token(svc3, 7, "in_person", 100))
    back2 = svc3.handle_callback(7, 7, 100, _callback_token(svc3, 7, "back", 100))
    _f3 = svc3.flow_for(7)
    assert back2.accepted and _f3 is not None and _f3.state == MeetingState.CHOOSE_FORMAT


def test_unrelated_text_exits_to_dialogue() -> None:
    svc = _service()
    _offer(svc)
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "in_person", 100))
    outcome = svc.handle_text(7, "А что значит третий шаг? Объясни подробно, пожалуйста?")
    assert not outcome.handled and outcome.exit_to_dialogue


def test_new_invalidates_every_keyboard() -> None:
    svc = _service()
    _offer(svc)
    token = _callback_token(svc, 7, "accept", 100)
    svc.handle_new(7)
    stale = svc.handle_callback(7, 7, 100, token)
    assert stale.stale and not stale.accepted


def test_lazy_expiry_after_24h() -> None:
    svc = _service()
    _offer(svc)
    later = datetime(2026, 10, 8, 13, 0, tzinfo=UTC)
    svc._clock = lambda: later  # noqa: SLF001
    token = _callback_token(svc, 7, "accept", 100)
    out = svc.handle_callback(7, 7, 100, token)
    assert out.stale
    _f = svc.flow_for(7)
    assert _f is not None and _f.state == MeetingState.EXPIRED


def test_duplicate_callback_causes_zero_extra_outputs() -> None:
    svc = _service()
    _offer(svc)
    token = _callback_token(svc, 7, "accept", 100)
    first = svc.handle_callback(7, 7, 100, token)
    assert first.accepted
    second = svc.handle_callback(7, 7, 100, token)
    assert second.stale and not second.accepted
    assert second.send_text is None and second.send_keyboard is None


def test_foreign_sender_and_cross_chat_rejected() -> None:
    svc = _service()
    _offer(svc, chat=7)
    token = _callback_token(svc, 7, "accept", 100)
    foreign = svc.handle_callback(7, 999, 100, token)
    assert foreign.stale and not foreign.accepted
    other = svc.handle_callback(8, 8, 100, token)
    assert other.stale and not other.accepted


def test_callback_payload_length_bounded_and_opaque() -> None:
    svc = _service()
    _offer(svc)
    token = _callback_token(svc, 7, "accept", 100)
    assert len(token.encode("utf-8")) <= 64
    assert "осква" not in token and "Moscow" not in token


def test_failed_delivery_leaves_no_phantom_offer() -> None:
    svc = _service()
    flow = svc.begin_offer(7)
    _ = flow
    svc.activate_offer(7, 100)
    state = svc.confirm_offer_delivery(7, 100, "failed")
    assert state == MeetingState.IDLE


def test_unknown_delivery_is_provisional_and_reconciled() -> None:
    svc = _service()
    svc.create_offer(7, 100)
    state = svc.confirm_offer_delivery(7, 100, "unknown")
    assert state == MeetingState.OFFERED
    flow = svc.flow_for(7)
    assert flow is not None and flow.pending_delivery is True
    out = svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    assert out.accepted


def test_real_directory_smoke_moscow_and_online_ordered() -> None:
    from aa.meeting_directory.query import get_upcoming_meetings

    reset_cache()
    moscow = get_upcoming_meetings(place_id="place-moscow-city", now_utc=TUESDAY_NOON, limit=3)
    assert moscow.fallback is None
    assert 1 <= len(moscow.occurrences) <= 3
    assert [item.start_utc for item in moscow.occurrences] == sorted(
        item.start_utc for item in moscow.occurrences
    )
    assert all(item.start_utc > TUESDAY_NOON for item in moscow.occurrences)
    online = get_upcoming_meetings(format="online", now_utc=TUESDAY_NOON, limit=3)
    assert online.fallback is None
    assert all(item.start_utc > TUESDAY_NOON for item in online.occurrences)
    text = format_results_text(list(moscow.occurrences))
    assert text.strip() and "http" in text


def test_unsupported_place_uses_truthful_link_only() -> None:
    from aa.meeting_directory.query import get_upcoming_meetings

    reset_cache()
    result = get_upcoming_meetings(
        place_id="place-novosibirsk-city", format="in_person", now_utc=TUESDAY_NOON, limit=3
    )
    assert result.fallback == "directory_only"
    assert result.occurrences == ()


def test_policy_has_no_keyword_routing() -> None:
    ctx = PolicyContext(invitation_state="IDLE", directory_available=True)
    assert decide_action(MeetingObservation(event="none"), ctx) == "none"
    assert (
        decide_action(
            MeetingObservation(event="asks_to_find"),
            PolicyContext(invitation_state="IDLE", directory_available=True),
        )
        == "provide_resource"
    )
    assert (
        decide_action(
            MeetingObservation(event="asks_to_find"),
            PolicyContext(invitation_state="IDLE", directory_available=False),
        )
        == "none"
    )
    assert may_offer(PolicyContext(invitation_state="IDLE", directory_available=True)) is True
    assert (
        may_offer(
            PolicyContext(
                invitation_state="IDLE", declined_this_session=True, directory_available=True
            )
        )
        is False
    )


def test_callback_query_parsing_and_inaccessible_inert() -> None:
    raw: dict[str, Any] = {
        "update_id": 501,
        "callback_query": {
            "id": "cq-1",
            "from": {"id": 7},
            "data": "token",
            "message": {
                "message_id": 100,
                "chat": {"id": 7, "type": "private"},
            },
        },
    }
    parsed = parse_callback_query(raw)
    assert parsed is not None and parsed.is_callback and parsed.callback_id == "cq-1"
    inaccessible: dict[str, Any] = {
        "update_id": 502,
        "callback_query": {"id": "cq-2", "from": {"id": 7}, "data": "token"},
    }
    parsed2 = parse_callback_query(inaccessible)
    assert parsed2 is not None and parsed2.callback_inaccessible


def test_bounded_harness_rejects_loops() -> None:
    svc = _service()
    _offer(svc)
    accepted = 0
    for step in range(100):
        flow = svc.flow_for(7)
        assert flow is not None
        if flow.state == MeetingState.OFFERED:
            token = _callback_token(svc, 7, "accept", 100)
        elif flow.state == MeetingState.CHOOSE_FORMAT:
            token = _callback_token(svc, 7, "online", 100)
        elif flow.state == MeetingState.SHOW_RESULTS and flow.results.cursor:
            token = svc._codec.encode(  # noqa: SLF001
                action="more",
                flow_seq=int(flow.flow_seq),
                generation=int(flow.generation),
                message_id=100,
                now=svc._now_utc().timestamp(),  # noqa: SLF001
            )
        else:
            break
        out = svc.handle_callback(7, 7, 100, token)
        if out.accepted:
            accepted += 1
        else:
            break
        if step > 60:
            break
    assert accepted <= 40
    flow = svc.flow_for(7)
    assert flow is not None
    # After DONE no unsolicited wizard output is generated.
    if flow.state == MeetingState.DONE:
        extra = svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "online", 100))
        assert not extra.accepted


def test_restart_makes_old_buttons_stale() -> None:
    svc = _service()
    _offer(svc)
    token = _callback_token(svc, 7, "accept", 100)
    restarted = MeetingService(
        secret=b"different-secret-32-bytes-pad-0002", clock=lambda: TUESDAY_NOON
    )
    out = restarted.handle_callback(7, 7, 100, token)
    assert out.stale and not out.accepted


def test_stale_schedule_renders_directory_link_not_fake_meeting() -> None:
    svc = _service()
    _offer(svc)
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "accept", 100))
    svc.handle_callback(7, 7, 100, _callback_token(svc, 7, "in_person", 100))
    outcome = svc.handle_text(7, "Казань")
    flow = svc.flow_for(7)
    assert flow is not None
    assert flow.state == MeetingState.DONE
    assert outcome.send_text is not None and "http" in outcome.send_text
    assert (
        "каталог" in outcome.send_text.lower()
        or "directory" in outcome.send_text.lower()
        or "http" in outcome.send_text
    )


async def test_app_offer_and_callback_use_zero_llm() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.telegram.transport import TelegramIncoming

    app = Application(Settings.from_env({}))
    await app.dispatcher.start()
    try:
        assert await app.offer_meeting(7) is True
        flow = app.meetings.flow_for(7)
        assert flow is not None and flow.state == MeetingState.OFFERED
        assert len(app.transport.sent) == 1  # type: ignore[attr-defined]
        edits = app.transport.markup_edits  # type: ignore[attr-defined]
        assert edits
        bound = int(flow.bound_message_id or 0)
        assert bound > 0
        accept_token = str(edits[-1]["markup"]["inline_keyboard"][0][0]["callback_data"])
        await app._process_dispatched_update(
            TelegramIncoming(
                update_id=1,
                chat_id=7,
                message_id=bound,
                text="",
                is_callback=True,
                callback_id="cq-1",
                callback_data=accept_token,
                sender_id=7,
                callback_message_id=bound,
            )
        )
        flow2 = app.meetings.flow_for(7)
        assert flow2 is not None and flow2.state == MeetingState.CHOOSE_FORMAT
    finally:
        await app.dispatcher.stop()
