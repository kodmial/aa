"""Gate C/D isolation (issue #340).

Proves Gate C genuine model/book/conversation behavior is separated from
Gate D real Telegram transport and runtime readiness, and that simulated
outbound delivery is never claimed as live Telegram delivery.
"""

from __future__ import annotations

from typing import Any

import pytest

from aa.qualification.gate_cd_boundary import (
    EXTERNAL_SEND_CONFIRMED,
    EXTERNAL_SEND_UNVERIFIED,
    GateBoundary,
    GateBoundaryError,
    assert_no_external_claim_from_simulated,
    classify_optional_send_status,
    coverage_gates,
    gate_b_boundary,
    gate_c_boundary,
    gate_d_boundary,
    owner_gate_for_check,
    validate_gate_boundary,
)


def test_boundary_rejects_unknown_and_contradictions() -> None:
    with pytest.raises(GateBoundaryError):
        validate_gate_boundary(GateBoundary(telegram_ingress_mode="nope"))
    with pytest.raises(GateBoundaryError):
        validate_gate_boundary(GateBoundary(telegram_egress_mode="nope"))
    with pytest.raises(GateBoundaryError):
        validate_gate_boundary(GateBoundary(delivery_receipt_kind="nope"))
    # In-memory id mislabeled external fails closed.
    with pytest.raises(GateBoundaryError):
        validate_gate_boundary(
            GateBoundary(telegram_egress_mode="stubbed", delivery_receipt_kind="external-confirmed")
        )
    with pytest.raises(GateBoundaryError):
        assert_no_external_claim_from_simulated(
            telegram_egress_mode="stubbed",
            delivery_receipt_kind="external-confirmed",
            message_id=3,
        )
    # Real provider/book without the real app path is contradictory.
    with pytest.raises(GateBoundaryError):
        validate_gate_boundary(GateBoundary(real_provider=True, real_app=False))


def test_canonical_boundaries_are_valid() -> None:
    gate_c = gate_c_boundary()
    assert gate_c.telegram_egress_mode == "stubbed"
    assert gate_c.delivery_receipt_kind == "simulated-in-process"
    assert gate_c.real_provider and gate_c.real_book and gate_c.real_app
    gate_b = gate_b_boundary()
    assert gate_b.telegram_egress_mode == "stubbed"
    assert gate_b.real_provider is False
    gate_d = gate_d_boundary(egress_real=True, consent_opt_in=True)
    assert gate_d.telegram_egress_mode == "real"
    assert gate_d.delivery_receipt_kind == "external-confirmed"


def test_optional_send_status_never_invents_confirmation() -> None:
    assert classify_optional_send_status(peer_configured=True, consent_opt_in=True) == (
        EXTERNAL_SEND_CONFIRMED
    )
    assert classify_optional_send_status(peer_configured=False, consent_opt_in=True) == (
        EXTERNAL_SEND_UNVERIFIED
    )
    assert classify_optional_send_status(peer_configured=True, consent_opt_in=False) == (
        EXTERNAL_SEND_UNVERIFIED
    )


def test_coverage_mapping_preserves_all_checks() -> None:
    grouped = coverage_gates()
    assert set(grouped) == {"B", "C", "D"}
    assert owner_gate_for_check("01-greeting-natural") == "B"
    assert owner_gate_for_check("25-typing-heartbeat-live") == "B"
    assert owner_gate_for_check("33-single-start-path") == "B"
    assert owner_gate_for_check("44-ready-only-after-polling-live") == "D"
    assert owner_gate_for_check("live-substantive-grounded-book-answer") == "C"
    assert owner_gate_for_check("real-telegram-startup-ready") == "D"
    assert owner_gate_for_check("V01-voice-turn-boundary-present") == "B"
    with pytest.raises(GateBoundaryError):
        owner_gate_for_check("totally-unknown-check-xyz")


async def test_transport_lane_deterministic_pass_without_credentials(monkeypatch: Any) -> None:
    from aa.qualification.product_contract_live import run_transport_lane

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_LIVE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("AA_TEST_TELEGRAM_BOT_TOKEN", raising=False)
    result = await run_transport_lane()
    assert result.lane == "telegram-transport-25-32"
    assert result.status == "PASS", result.failed
    assert result.metrics["gate_boundary"]["telegram_egress_mode"] == "stubbed"
    assert result.metrics["live_network_owned_by"] == "telegram-readiness-D"


async def test_readiness_lane_without_identity_is_typed_blocker(monkeypatch: Any) -> None:
    from aa.qualification.product_contract_live import run_telegram_readiness_lane

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_TEST_TELEGRAM_BOT_TOKEN", raising=False)
    result = await run_telegram_readiness_lane()
    assert result.lane == "telegram-readiness-D"
    assert result.status == "INCOMPLETE"
    assert "EXTERNAL_TEST_BOT_UNAVAILABLE" in result.metrics["live_external_dependency"]
    assert result.metrics["external_send_status"] == EXTERNAL_SEND_UNVERIFIED


async def test_readiness_ready_without_peer_stays_mandatory_pass(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_telegram_readiness_lane

    async def _fake_ready(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(
            status="ready",
            category="ready",
            bot_id_present=True,
            username_len=3,
            polling_live=False,
            bootstrap_verified=True,
        )

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("AA_TEST_TELEGRAM_PEER_CHAT_ID", raising=False)
    monkeypatch.delenv("AA_LIVE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("AA_TEST_TELEGRAM_PROBE_SEND", raising=False)
    monkeypatch.delenv("AA_LIVE_TELEGRAM_PROBE_SEND", raising=False)
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_ready)
    result = await run_telegram_readiness_lane()
    assert result.status == "PASS", (result.failed, result.incomplete)
    assert "real-telegram-startup-ready" in result.passed
    assert result.metrics["external_send_status"] == EXTERNAL_SEND_UNVERIFIED
    assert result.metrics["live_transport_bootstrap_verified"] is True
    assert "delivery-confirmed" not in " ".join(result.passed)


async def test_readiness_opt_in_peer_confirms_genuine_ack(monkeypatch: Any) -> None:
    import aa.qualification.product_contract_live as live_mod
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_telegram_readiness_lane

    async def _fake_ready(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(
            status="ready",
            category="ready",
            bot_id_present=True,
            username_len=3,
            polling_live=False,
            bootstrap_verified=True,
        )

    async def _fake_send(token: str, peer: int) -> str:
        _ = (token, peer)
        return "confirmed"

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("AA_TEST_TELEGRAM_PEER_CHAT_ID", "123456")
    monkeypatch.setenv("AA_TEST_TELEGRAM_PROBE_SEND", "1")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_ready)
    monkeypatch.setattr(live_mod, "_probe_live_delivery", _fake_send)
    result = await run_telegram_readiness_lane()
    assert result.status == "PASS", (result.failed, result.incomplete)
    assert "real-telegram-delivery-confirmed" in result.passed
    assert result.metrics["external_send_status"] == EXTERNAL_SEND_CONFIRMED
    assert result.metrics["gate_boundary"]["delivery_receipt_kind"] == "external-confirmed"


async def test_readiness_conflict_is_blocked_without_destruction(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_telegram_readiness_lane

    async def _fake_blocked(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(status="blocked", category="polling-webhook-conflict")

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_blocked)
    result = await run_telegram_readiness_lane()
    assert result.status == "INCOMPLETE"
    assert result.metrics["live_external_dependency"] == "EXTERNAL_POLLER_CONFLICT"
    assert not result.failed


async def test_readiness_auth_failure_is_fail(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_telegram_readiness_lane

    async def _fake_auth(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(status="failed", category="telegram-auth")

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "bad-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_auth)
    result = await run_telegram_readiness_lane()
    assert result.status == "FAIL"
    assert "real-telegram-startup-telegram-auth" in result.failed


async def test_readiness_timeout_and_403_are_typed(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_telegram_readiness_lane
    from aa.telegram.startup_probe import classify_probe_http_error

    assert classify_probe_http_error(code=409, mandatory=True) == "EXTERNAL_POLLER_CONFLICT"
    assert classify_probe_http_error(code=403, mandatory=True) == "EXTERNAL_MANDATORY_FORBIDDEN"
    assert classify_probe_http_error(code=403, mandatory=False) == "EXTERNAL_OPTIONAL_FORBIDDEN"

    async def _boom(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        raise TimeoutError("probe timeout")

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _boom)
    result = await run_telegram_readiness_lane()
    assert result.status == "FAIL"
    assert "real-telegram-startup-timeout" in result.failed


def test_safe_identity_resolution_prefers_test_bot(monkeypatch: Any) -> None:
    from aa.telegram.startup_probe import (
        resolve_probe_token,
        verify_bot_identity_scope,
    )

    monkeypatch.setenv("AA_TEST_TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "prod-token")
    token, kind = resolve_probe_token(production_token="prod-token")
    assert token == "test-token" and kind == "isolated-test-bot"
    monkeypatch.delenv("AA_TEST_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE", raising=False)
    with pytest.raises(ValueError):
        resolve_probe_token(production_token="prod-token")
    monkeypatch.setenv("AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE", "1")
    token, kind = resolve_probe_token(production_token="prod-token")
    assert token == "prod-token" and kind == "production-lease"
    assert verify_bot_identity_scope(bot_info={"id": 7}, expected_bot_id=7) is True
    assert verify_bot_identity_scope(bot_info={"id": 8}, expected_bot_id=7) is False


def test_trusted_marker_rejects_stale_or_copied() -> None:
    from aa.telegram.startup_probe import validate_trusted_ready_marker

    sha = "a" * 40
    ok, _ = validate_trusted_ready_marker(
        marker_sha=sha,
        marker_run_id="42",
        marker_timestamp_s=1000.0,
        expected_sha=sha,
        expected_run_id="42",
        now_s=1000.0 + 60.0,
        poll_task_live=True,
    )
    assert ok is True
    ok, _ = validate_trusted_ready_marker(
        marker_sha=sha,
        marker_run_id="42",
        marker_timestamp_s=1000.0,
        expected_sha=sha,
        expected_run_id="42",
        now_s=1000.0 + 5 * 3600.0,
        poll_task_live=True,
    )
    assert ok is False
    ok, _ = validate_trusted_ready_marker(
        marker_sha=sha,
        marker_run_id="43",
        marker_timestamp_s=1000.0,
        expected_sha=sha,
        expected_run_id="42",
        now_s=1060.0,
        poll_task_live=True,
    )
    assert ok is False
    ok, _ = validate_trusted_ready_marker(
        marker_sha=sha,
        marker_run_id="42",
        marker_timestamp_s=1000.0,
        expected_sha=sha,
        expected_run_id="42",
        now_s=1060.0,
        poll_task_live=False,
    )
    assert ok is False


def test_grounding_split_c_fails_while_d_healthy() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    ungrounded = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 0,
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    assert _is_grounded_substantive_reply(ungrounded, "Some ungrounded text.") is False
    grounded = dict(ungrounded, verified_book_units=2)
    assert _is_grounded_substantive_reply(grounded, "Some grounded text.") is True


def test_partial_unknown_and_failed_receipts_never_confirm() -> None:
    from aa.conversation.finalization import normalize_answer_text, sha256_text
    from aa.telegram.startup_probe import confirm_full_delivery

    certified = "Понял вас. Давайте разберём спокойно."
    normalized = normalize_answer_text(certified)
    digest = sha256_text(normalized)
    total = len(normalized)

    def _receipt(status: str, start: int, end: int) -> dict[str, Any]:
        return {"status": status, "final_sha256": digest, "char_start": start, "char_end": end}

    ok, _ = confirm_full_delivery(certified, [_receipt("confirmed", 0, total)])
    assert ok is True
    for status in ("split", "partial", "unknown", "failed"):
        ok, _ = confirm_full_delivery(certified, [_receipt(status, 0, total)])
        assert ok is False
    ok, _ = confirm_full_delivery(certified, [_receipt("confirmed", 0, total // 2)])
    assert ok is False


async def test_failed_send_commits_no_history_or_session_state() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.finalization import normalize_answer_text, sha256_text
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.startup_probe import confirm_full_delivery
    from aa.telegram.transport import (
        StubTelegramTransport,
        TelegramApiError,
        TelegramIncoming,
        TelegramReply,
    )

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    class _FlakyOnce(StubTelegramTransport):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def send(self, reply: TelegramReply) -> int:
            self.calls += 1
            raise TelegramApiError("transient")

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"})
    flaky = _FlakyOnce()
    app = Application(
        settings,
        transport=flaky,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        await app._process_dispatched_update(
            TelegramIncoming(update_id=34001, chat_id=34001, message_id=1, text="привет")
        )
        assert len(flaky.sent) == 0
        certified = "Понял вас. Давайте разберём спокойно."
        normalized = normalize_answer_text(certified)
        digest = sha256_text(normalized)
        ok, _ = confirm_full_delivery(
            certified,
            [
                {
                    "status": "failed",
                    "final_sha256": digest,
                    "char_start": 0,
                    "char_end": len(normalized),
                }
            ],
        )
        assert ok is False
    finally:
        await app.stop()


async def test_unsafe_cross_user_callback_stays_inert() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно."

    transport = StubTelegramTransport()
    app = Application(
        Settings.from_env(),
        transport=transport,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        chat_id = 34011
        assert await app.offer_meeting(chat_id) is True
        flow = app.meetings.flow_for(chat_id)
        assert flow is not None and flow.bound_message_id is not None
        bound = int(flow.bound_message_id)
        sent_before = len(transport.sent)
        await app._handle_callback_update(
            TelegramIncoming(
                update_id=34012,
                chat_id=chat_id,
                message_id=bound,
                text="",
                is_callback=True,
                callback_id="x",
                callback_data="forged-data",
                sender_id=99999,
                callback_message_id=bound,
            )
        )
        assert len(transport.sent) == sent_before
    finally:
        await app.stop()
