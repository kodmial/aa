"""Gate C focus mode unit tests (kodmial/aa#431).

Covers: the campaign mode resolver, the typed c-focus result and Gate E
diagnostic builders, the attempt ledger's proven-quiescence semantics, the
upgrade of the stalled-turn quiesce cleanup, and the reducer script's
fail-closed C-focus guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pytest")

SHA = "b" * 40
PRODUCT = "product"
RUNTIME = "opencode-runtime"


# ---------------------------------------------------------------------------
# campaign mode resolver
# ---------------------------------------------------------------------------


def test_campaign_mode_dispatch_input_wins_over_repo_variable() -> None:
    from aa.qualification.campaign_mode import (
        MODE_FULL,
        SOURCE_DISPATCH_INPUT,
        resolve_campaign_mode,
    )

    decision = resolve_campaign_mode(dispatch_input=MODE_FULL, repo_variable="gate-c-focus")
    assert decision.mode == MODE_FULL
    assert decision.source == SOURCE_DISPATCH_INPUT


def test_campaign_mode_repo_variable_wins_over_automatic_default() -> None:
    from aa.qualification.campaign_mode import (
        MODE_GATE_C_FOCUS,
        SOURCE_REPO_VARIABLE,
        resolve_campaign_mode,
    )

    decision = resolve_campaign_mode(repo_variable=MODE_GATE_C_FOCUS)
    assert decision.mode == MODE_GATE_C_FOCUS
    assert decision.source == SOURCE_REPO_VARIABLE


def test_campaign_mode_invalid_explicit_value_fails_closed() -> None:
    from aa.qualification.campaign_mode import CampaignModeError, resolve_campaign_mode

    with pytest.raises(CampaignModeError):
        resolve_campaign_mode(dispatch_input="ab-mode")
    with pytest.raises(CampaignModeError):
        resolve_campaign_mode(repo_variable="full-at-any-cost")


def test_campaign_mode_automatic_trigger_never_hidden_full() -> None:
    from aa.qualification.campaign_mode import (
        AUTOMATIC_DEFAULT_MODE,
        automatic_trigger_uses_full_mode,
        resolve_campaign_mode,
    )

    automatic = resolve_campaign_mode()
    assert automatic.mode == AUTOMATIC_DEFAULT_MODE == "gate-c-focus"
    assert not automatic_trigger_uses_full_mode(automatic)
    owner_full = resolve_campaign_mode(repo_variable="full")
    assert automatic_trigger_uses_full_mode(owner_full)


# ---------------------------------------------------------------------------
# c-focus result building
# ---------------------------------------------------------------------------


def _live_summary_with_c_lane(status: str = "PASS") -> dict[str, object]:
    return {
        "main_sha": SHA,
        "run_id": "run-1",
        "status": status,
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "status": status,
                "passed": [],
                "failed": [],
                "incomplete": [],
                "metrics": {
                    "turns_executed": 2,
                    "turn_latency_samples": [
                        {"outcome": "served", "latency_ms": 12000.0},
                        {"outcome": "timeout", "latency_ms": 90000.0, "censored": True},
                    ],
                    "latency_series": "c-series-simulated-handoff",
                    "latency_provider_id": "opencode/production-model",
                    "scenario_diagnostics": [
                        {
                            "family": "snapshot-22-33",
                            "last_active_stage": "verifier",
                            "timeout_category": "turn-timeout",
                            "timeout_reason": "no-real-progress",
                            "request_state": "http-in-flight",
                            "stall_kind": "http-in-flight",
                        }
                    ],
                },
            }
        ],
    }


def test_c_focus_pass_is_never_a_trusted_product_qualification() -> None:
    from aa.qualification.c_focus import build_c_focus_result

    result = build_c_focus_result(
        main_sha=SHA,
        run_id="run-1",
        gate_c_evidence={
            "schema_version": "aa-gate-evidence/2",
            "gate": "C",
            "status": "PASS",
            "sha": SHA,
            "run_id": "run-1",
        },
        live_summary=_live_summary_with_c_lane("PASS"),
    )
    assert result["schema_version"] == "aa-gate-c-focus-result/1"
    assert result["status"] == "PASS"
    assert result["verdict"] == "C_PASS"
    diagnostics = result["privacy_safe_diagnostics"]
    assert diagnostics["product_qualified"] is False
    assert diagnostics["a_b_executed"] is False
    assert result["not_qualified"]


def test_c_focus_fail_keeps_stage_of_failure() -> None:
    from aa.qualification.c_focus import build_c_focus_result

    result = build_c_focus_result(
        main_sha=SHA,
        run_id="run-1",
        gate_c_evidence={"gate": "C", "status": "FAIL", "sha": SHA, "run_id": "run-1"},
        live_summary=_live_summary_with_c_lane("FAIL"),
    )
    assert result["verdict"] == "C_FAIL"
    diagnostics = result["privacy_safe_diagnostics"]
    assert diagnostics["stage_of_failure"] == "verifier"
    assert diagnostics["stall_kind"] == "http-in-flight"


def test_c_focus_missing_or_mismatched_evidence_never_passes() -> None:
    from aa.qualification.c_focus import build_c_focus_result

    blocked = build_c_focus_result(main_sha=SHA, run_id="run-1", gate_c_evidence=None)
    assert blocked["status"] == "BLOCKED"
    assert blocked["verdict"] == "C_BLOCKED"

    mismatch = build_c_focus_result(
        main_sha=SHA,
        run_id="run-1",
        gate_c_evidence={"gate": "C", "status": "PASS", "sha": "a" * 40, "run_id": "run-1"},
    )
    assert mismatch["status"] == "BLOCKED"
    assert mismatch["verdict"] == "C_BLOCKED"


def test_gate_e_diagnostic_has_no_slo_verdict() -> None:
    from aa.qualification.c_focus import build_gate_e_diagnostic

    payload = build_gate_e_diagnostic(
        main_sha=SHA, run_id="run-1", live_summary=_live_summary_with_c_lane("PASS")
    )
    assert payload["schema_version"] == "aa-gate-e-diagnostic/1"
    assert payload["slo_thresholds_applied"] is False
    assert payload["latency_sample_count"] == 2
    assert payload["latency_p50_ms"] == 12000.0
    assert payload["latency_max_ms"] == 90000.0
    assert payload["censored_turn_count"] == 1
    assert payload["verdicts_applicable"].startswith("none")
    assert "verdict:PASS" not in json.dumps(payload).lower()


def test_gate_e_diagnostic_without_samples_is_not_applicable() -> None:
    from aa.qualification.c_focus import build_gate_e_diagnostic

    payload = build_gate_e_diagnostic(main_sha=SHA, run_id="run-1", live_summary=None)
    assert payload["latency_sample_count"] == 0
    assert payload["latency_p50_ms"] is None
    assert payload["non_applicable_reason"] == "no-c-latency-samples"


# ---------------------------------------------------------------------------
# attempt ledger proven quiescence
# ---------------------------------------------------------------------------


def test_attempt_ledger_proven_quiescence_lifecycle() -> None:
    from aa.opencode.attempt_audit import AttemptLedger

    ledger = AttemptLedger()
    attempt_id = ledger.register_start(operation="chat", agent="main")
    assert not ledger.proven_quiesced()
    assert ledger.has_unterminated()
    ledger.mark_error_kind(attempt_id, "timeout")
    ledger.finish(attempt_id, state="completed", latency_ms=10.0, failure_category="timeout")
    assert not ledger.has_unterminated()
    assert ledger.proven_quiesced()

    # A "cancelled" attempt proves nothing about the HTTP thread until the
    # worker future itself is corroborated by `mark_worker_done`.
    causality = "causality-for-cancel"
    cancel_id = ledger.register_start(operation="chat", agent="main", causality_id=causality)
    ledger.finish(cancel_id, state="cancelled", latency_ms=5.0)
    assert not ledger.proven_quiesced(causality)
    ledger.mark_worker_done(cancel_id)
    assert ledger.proven_quiesced(causality)


def test_attempt_ledger_late_outcome_never_overrides_terminal_state() -> None:
    from aa.opencode.attempt_audit import AttemptLedger

    ledger = AttemptLedger()
    attempt_id = ledger.register_start(operation="chat", agent="main")
    ledger.finish(attempt_id, state="cancelled", latency_ms=1.0)
    ledger.mark_worker_done(attempt_id)
    accepted = ledger.finish(
        attempt_id, state="failed", latency_ms=30.0, failure_category="timeout", late=True
    )
    assert accepted
    recorded = ledger.entries()[-1]
    assert recorded["state"] == "cancelled"
    assert recorded["late_outcome"] == "failed"
    assert recorded["late_latency_ms"] == 30.0


def test_client_quiescence_proven_endpoint() -> None:
    from aa.opencode.client import HttpOpenCodeClient

    client = HttpOpenCodeClient("http://127.0.0.1:1")
    causality = "add-stone-causality"
    attempt_id = client._attempts.register_start(
        operation="chat", agent="main", causality_id=causality
    )
    assert client.quiescence_proven(causality) is False
    client._attempts.finish(attempt_id, state="completed", latency_ms=1.0)
    client._attempts.mark_worker_done(attempt_id)
    assert client.quiescence_proven(causality) is True


# ---------------------------------------------------------------------------
# stalled turn cleanup upgrade
# ---------------------------------------------------------------------------


class _Seam:
    def __init__(self) -> None:
        self.begun: list[str] = []

    def begin_quiescence(self, reason: str) -> None:
        self.begun.append(reason)

    def end_quiescence(self) -> None:
        pass


class _App:
    def __init__(self, *, in_flight: bool) -> None:
        self._in_flight = in_flight
        self.restart_count = 0

    def cancel_pending_turns(self, chat_id: int, grace_s: float) -> dict[str, object]:
        return {"had_worker": True, "stopped_within_grace": True}

    def turn_in_flight(self, chat_id: int) -> bool:
        return self._in_flight

    async def stop(self) -> None:
        self.restart_count += 1

    async def start(self) -> None:
        pass


@pytest.mark.asyncio
async def test_quiesce_without_probe_isolates_runtime_fail_closed() -> None:
    from aa.qualification.live_turn_repair import quiesce_and_abort_stalled_turn

    app = _App(in_flight=False)
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=int("1001"),
        seam=_Seam(),
        grace_s=0.01,
        reason="no-real-progress",
        restart=app.stop,
    )
    # The caller cannot prove termination without the probe: the runtime
    # must be isolated so no HTTP thread can deliver stale output.
    assert cleanup["orphan_http_attempt_isolated"] is True
    assert cleanup["non_cooperative"] is True
    assert cleanup["http_quiescence_proven"] is False
    assert app.restart_count == 1


@pytest.mark.asyncio
async def test_quiesce_with_proven_probe_skips_restart() -> None:
    from aa.qualification.live_turn_repair import quiesce_and_abort_stalled_turn

    app = _App(in_flight=False)
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=int("1001"),
        seam=_Seam(),
        grace_s=0.01,
        reason="no-real-progress",
        restart=app.stop,
        quiescence_probe=lambda: True,
    )
    assert cleanup["http_quiescence_proven"] is True
    assert cleanup.get("orphan_http_attempt_isolated", False) is False
    assert app.restart_count == 0


@pytest.mark.asyncio
async def test_quiesce_probe_failure_degrades_to_isolation() -> None:
    from aa.qualification.live_turn_repair import quiesce_and_abort_stalled_turn

    def _failing_probe() -> bool:
        raise RuntimeError("probe unavailable")

    app = _App(in_flight=False)
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=int("1001"),
        seam=_Seam(),
        grace_s=0.01,
        reason="no-real-progress",
        restart=app.stop,
        quiescence_probe=_failing_probe,
    )
    assert cleanup["http_quiescence_proven"] is False
    assert cleanup["non_cooperative"] is True
    assert app.restart_count == 1


@pytest.mark.asyncio
async def test_quiesce_probe_in_flight_wins_over_clean_cancel() -> None:
    from aa.qualification.live_turn_repair import quiesce_and_abort_stalled_turn

    app = _App(in_flight=True)
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=int("1001"),
        seam=_Seam(),
        grace_s=0.01,
        reason="no-real-progress",
        restart=app.stop,
        quiescence_probe=lambda: True,
    )
    assert cleanup["non_cooperative"] is True
    assert app.restart_count == 1


# ---------------------------------------------------------------------------
# reducer script fail-closed contract
# ---------------------------------------------------------------------------


def _write_live_summary(tmp_path: Path, *, status: str = "FAIL") -> None:
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(
            {
                "main_sha": SHA,
                "run_id": "run-1",
                "status": status,
                "lanes": [
                    {
                        "lane": "live-telegram-evidence",
                        "status": status,
                        "passed": [],
                        "failed": ["live-answer-no-generic-collapse"],
                        "incomplete": [],
                        "metrics": {
                            "turn_latency_samples": [
                                {"outcome": "served", "latency_ms": 12000.0},
                                {
                                    "outcome": "timeout",
                                    "latency_ms": 90000.0,
                                    "censored": True,
                                },
                            ],
                            "latency_series": "c-series-simulated-handoff",
                            "gate_boundary": {
                                "telegram_egress_mode": "stubbed",
                                "delivery_receipt_kind": "simulated-in-process",
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_reducer_gate_c_focus_publishes_typed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.delenv("SELF_PROVING_LIVE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    _write_live_summary(tmp_path, status="FAIL")
    out_dir = tmp_path / "self-proving-out"
    exit_code = runner.main(
        [
            "--main-sha",
            SHA,
            "--run-id",
            "run-1",
            "--out-dir",
            str(out_dir),
            "--emit-gate",
            "C",
            "--qualification-mode",
            "gate-c-focus",
            "--emit-c-focus-result",
            "--emit-e-diagnostic",
        ]
    )
    assert exit_code == 1  # real Gate C FAIL maps to C_FAIL, never BLOCKED
    result = json.loads((out_dir / "c-focus-result.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == "aa-gate-c-focus-result/1"
    assert result["verdict"] == "C_FAIL"
    assert result["status"] == "FAIL"
    assert result["sha"] == SHA
    diagnostics = result["privacy_safe_diagnostics"]
    assert diagnostics["a_b_executed"] is False
    assert diagnostics["product_qualified"] is False
    diagnostic = json.loads((out_dir / "gate-E-diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["schema_version"] == "aa-gate-e-diagnostic/1"
    assert diagnostic["slo_thresholds_applied"] is False
    assert diagnostic["latency_sample_count"] == 2


def test_reducer_gate_c_focus_never_runs_gate_a_or_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for gate in ("A", "B"):
        exit_code = runner.main(
            [
                "--main-sha",
                SHA,
                "--run-id",
                "run-1",
                "--out-dir",
                str(tmp_path / "blocked-out"),
                "--emit-gate",
                gate,
                "--qualification-mode",
                "gate-c-focus",
            ]
        )
        assert exit_code == 2
    captured = capsys.readouterr().err
    assert "never executes Gate A" in captured
    assert "never executes Gate B" in captured


def test_reducer_gate_c_focus_blocks_trusted_full_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    exit_code = runner.main(
        [
            "--main-sha",
            SHA,
            "--run-id",
            "run-1",
            "--out-dir",
            str(tmp_path / "blocked-full"),
            "--final-only",
            "--qualification-mode",
            "gate-c-focus",
        ]
    )
    assert exit_code == 2
    assert "never a product qualification" in capsys.readouterr().err


def test_reducer_focus_artifacts_reject_non_c_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    exit_code = runner.main(
        [
            "--main-sha",
            SHA,
            "--run-id",
            "run-1",
            "--out-dir",
            str(tmp_path / "guard-out"),
            "--emit-gate",
            "E",
            "--emit-c-focus-result",
        ]
    )
    assert exit_code == 2
    assert "only valid with --emit-gate C" in capsys.readouterr().err


def test_reducer_gate_c_focus_blocked_without_live_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.delenv("SELF_PROVING_LIVE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "blocked-evidence"
    exit_code = runner.main(
        [
            "--main-sha",
            SHA,
            "--run-id",
            "run-1",
            "--out-dir",
            str(out_dir),
            "--emit-gate",
            "C",
            "--qualification-mode",
            "gate-c-focus",
            "--emit-c-focus-result",
        ]
    )
    # Without any live lane the emit itself is BLOCKED; the focus result
    # must exist and carry the typed C_BLOCKED verdict.
    assert exit_code == 2
    result = json.loads((out_dir / "c-focus-result.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "C_BLOCKED"
    assert result["status"] == "BLOCKED"
