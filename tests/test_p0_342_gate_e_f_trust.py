"""Gate E real latency evidence and Gate F trustworthy verdict (kodmial/aa#342).

Gate E is an independent evidence consumer and latency calculator: it never
repeats Gate C, the A/B bootstrap, provider calls, or Gate D network
operations. Failed/timeout ordinary turns are counted as bounded-censored
evidence, never dropped. Gate F is a pure result reducer over trusted
versioned A-E artifacts. Thresholds are unchanged (p95 <= 60s, ordinary max
< 120s); the optional external send never becomes a mandatory condition.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_self_proving_qualification as runner  # noqa: E402
from aa.qualification.gate_evidence import (  # noqa: E402
    GATE_EVIDENCE_FILES,
    REQUIRED_A_CHECKS,
    REQUIRED_B_CHECKS,
    SubcheckResult,
    build_checkpoint_body,
    build_evidence_body,
    checkpoint_checksum,
    load_and_validate_checkpoint,
    load_and_validate_evidence,
    sign_checkpoint_body,
    sign_evidence_body,
    write_checkpoint_atomic,
)
from aa.qualification.gate_latency import (  # noqa: E402
    C_SERIES,
    EXTERNAL_CONFIRMED,
    EXTERNAL_UNVERIFIED,
    GateLatencyError,
    TurnSample,
    build_c_series_from_live_summary,
    build_gate_e_evidence,
    classify_optional_external_send,
    coerce_samples,
    evaluate_latency_report,
    new_causality_id,
    validate_samples,
)
from aa.qualification.self_proving import (  # noqa: E402
    ORDINARY_TURN_BUDGET_MS,
    P95_TARGET_MS,
    SelfProvingError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER_SHA = "b" * 40
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _sample(
    latency_ms: float,
    *,
    outcome: str = "served",
    family: str = "ordinary-text",
    censored: bool | None = None,
) -> TurnSample:
    return TurnSample(
        causality_id=new_causality_id(),
        family=family,
        mode="text",
        outcome=outcome,
        latency_ms=float(latency_ms),
        series=C_SERIES,
        censored=True if outcome in ("timeout", "failed") else bool(censored or False),
    )


def _subs(checks: tuple[str, ...], owner: str) -> list[SubcheckResult]:
    return [SubcheckResult(check_id=name, status="PASS", component=owner) for name in checks]


def _signed(
    gate: str,
    status: str,
    subs: list[SubcheckResult],
    *,
    sha: str = SHA,
    run_id: str = "run-9",
    component: str = "",
    category: str = "",
    live: bool = False,
) -> dict[str, object]:
    body = build_evidence_body(
        gate=gate,
        status=status,
        sha=sha,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        component=component,
        failure_category=category,
        run_id=run_id,
        subchecks=subs,
        live_trusted=live,
    )
    return sign_evidence_body(body)


def _live_out_payload(
    *,
    sha: str = SHA,
    run_id: str = "run-9",
    served_ms: list[float] | None = None,
    censored_families: list[str] | None = None,
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for value in served_ms or []:
        samples.append(
            {
                "causality_id": new_causality_id(),
                "family": "ordinary-text",
                "mode": "text",
                "outcome": "served",
                "latency_ms": float(value),
                "monotonic": True,
                "clock": "monotonic-perf_counter",
                "series": C_SERIES,
                "boundary": "production-update-intake-to-simulated-handoff",
                "receipt_kind": "simulated-in-process",
                "censored": False,
            }
        )
    for family in censored_families or []:
        samples.append(
            {
                "causality_id": new_causality_id(),
                "family": family,
                "mode": "text",
                "outcome": "timeout",
                "latency_ms": float(ORDINARY_TURN_BUDGET_MS),
                "monotonic": True,
                "clock": "monotonic-perf_counter",
                "series": C_SERIES,
                "boundary": "production-update-intake-to-simulated-handoff",
                "receipt_kind": "simulated-in-process",
                "censored": True,
            }
        )
    diagnostics: list[dict[str, object]] = []
    for family in censored_families or []:
        diagnostics.append({"family": family, "delivery_timeout": True})
    return {
        "main_sha": sha,
        "run_id": run_id,
        "status": "PASS",
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "status": "PASS",
                "passed": ["live-text-latency-measured"],
                "failed": [],
                "incomplete": [],
                "metrics": {
                    "turns_executed": len(samples),
                    "turn_latencies_ms": [float(v) for v in (served_ms or [])],
                    "turn_latency_samples": samples,
                    "latency_provider_id": "opencode/test-model",
                    "latency_boundary": "production-update-intake-to-simulated-handoff",
                    "latency_source_proof_kind": "c-series-simulated-handoff",
                    "external_send_status": EXTERNAL_UNVERIFIED,
                    "scenario_diagnostics": diagnostics,
                    "gate_boundary": {
                        "telegram_egress_mode": "stubbed",
                        "delivery_receipt_kind": "simulated-in-process",
                    },
                },
            }
        ],
    }


def test_slo_thresholds_unchanged() -> None:
    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000


def test_gate_e_blocked_without_samples() -> None:
    evidence = runner._gate_e(SHA, "run-1", PRODUCT, RUNTIME, [], gate_c_live=True)
    assert evidence.status == "BLOCKED"
    assert evidence.failure_category == "missing-live-samples"
    evidence = runner._gate_e(SHA, "run-1", PRODUCT, RUNTIME, [1000.0], gate_c_live=False)
    assert evidence.status == "BLOCKED"
    assert evidence.failure_category == "missing-live-samples"


def test_censored_timeout_fails_reliability_and_is_counted() -> None:
    samples = [_sample(2000.0), _sample(3000.0), _sample(120000.0, outcome="timeout")]
    report = evaluate_latency_report(samples)
    assert report.status == "FAIL"
    assert report.censored_count == 1
    assert report.served_count == 2
    assert report.sample_count == 3
    assert report.max_ms == 120000.0
    evidence = build_gate_e_evidence(
        report,
        sha=SHA,
        run_id="run-1",
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
    )
    assert evidence.status == "FAIL"
    assert evidence.live_trusted is True


def test_p95_and_budget_boundary_cases() -> None:
    # p95 exactly at the unchanged target passes; one ms over fails.
    assert evaluate_latency_report([_sample(60000.0)]).status == "PASS"
    assert evaluate_latency_report([_sample(60001.0)]).status == "FAIL"
    # Twenty samples: p95 lands on ordered[18]; max stays under the budget.
    rising = [_sample(float(1000 + index * 1000)) for index in range(17)]
    boundary_pass = rising + [_sample(60000.0), _sample(60000.0), _sample(119999.0)]
    passing = evaluate_latency_report(boundary_pass)
    assert passing.status == "PASS"
    assert passing.p95_ms == 60000.0
    assert passing.max_ms == 119999.0
    boundary_fail = rising + [_sample(60000.0), _sample(60001.0), _sample(119999.0)]
    assert evaluate_latency_report(boundary_fail).status == "FAIL"
    failed = evaluate_latency_report([_sample(120000.0)])
    assert failed.status == "FAIL"


def test_samples_require_unique_causality_and_monotonic_clock() -> None:
    first = _sample(1000.0)
    duplicate = TurnSample(
        causality_id=first.causality_id,
        family="ordinary-text",
        mode="text",
        outcome="served",
        latency_ms=1100.0,
        series=C_SERIES,
    )
    with pytest.raises(GateLatencyError):
        validate_samples([first, duplicate])
    with pytest.raises(GateLatencyError):
        TurnSample.from_dict(
            {
                "causality_id": new_causality_id(),
                "family": "ordinary-text",
                "mode": "text",
                "outcome": "served",
                "latency_ms": 1000.0,
                "monotonic": False,
                "clock": "wall-clock",
                "series": C_SERIES,
            }
        )


def test_c_series_never_relabels_simulated_ack_as_external() -> None:
    with pytest.raises(GateLatencyError):
        TurnSample.from_dict(
            {
                "causality_id": new_causality_id(),
                "family": "ordinary-text",
                "mode": "text",
                "outcome": "served",
                "latency_ms": 1000.0,
                "monotonic": True,
                "clock": "monotonic-perf_counter",
                "series": C_SERIES,
                "receipt_kind": "external-confirmed",
            }
        )


def test_optional_external_send_never_mandatory() -> None:
    assert classify_optional_external_send(peer_configured=False, consent_opt_in=False) == (
        EXTERNAL_UNVERIFIED
    )
    assert classify_optional_external_send(peer_configured=True, consent_opt_in=False) == (
        EXTERNAL_UNVERIFIED
    )
    assert (
        classify_optional_external_send(
            peer_configured=True, consent_opt_in=True, message_id=None, delivery_confirmed=False
        )
        == EXTERNAL_UNVERIFIED
    )
    assert (
        classify_optional_external_send(
            peer_configured=True, consent_opt_in=True, message_id=0, delivery_confirmed=True
        )
        == EXTERNAL_UNVERIFIED
    )
    assert (
        classify_optional_external_send(
            peer_configured=True, consent_opt_in=True, message_id=42, delivery_confirmed=True
        )
        == EXTERNAL_CONFIRMED
    )


def test_c_series_builder_counts_timeouts_from_diagnostics() -> None:
    payload = _live_out_payload(served_ms=[1000.0, 2000.0], censored_families=["topic-shift"])
    samples = build_c_series_from_live_summary(payload)
    assert len(samples) == 3
    assert sum(1 for item in samples if item.censored) == 1
    report = evaluate_latency_report(samples)
    assert report.status == "FAIL"


def test_c_series_builder_blocked_without_samples() -> None:
    with pytest.raises(GateLatencyError):
        build_c_series_from_live_summary({"lanes": []})


def test_emit_e_evaluates_despite_c_semantic_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B PASS / C simulated-egress PASS-but-semantic-FAIL / E measured FAIL."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    # C failed semantically but its lane is live-trusted with real samples.
    c_body = build_evidence_body(
        gate="C",
        status="FAIL",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        component="live-production-path",
        failure_category="live-path-failed",
        run_id="run-9",
        subchecks=[],
        live_trusted=True,
    )
    (out_dir / GATE_EVIDENCE_FILES["C"]).write_text(
        json.dumps(sign_evidence_body(c_body)) + "\n", encoding="utf-8"
    )
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(_live_out_payload(served_ms=[1000.0, 2000.0, 90000.0])) + "\n",
        encoding="utf-8",
    )
    evidence, product, runtime = runner._emit_gate_e_from_evidence(
        expected=SHA, run_id="run-9", out_dir=out_dir
    )
    assert product == PRODUCT
    assert runtime == RUNTIME
    assert evidence.status == "FAIL"
    assert evidence.live_trusted is True


def test_emit_e_passes_on_healthy_c_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    c_body = build_evidence_body(
        gate="C",
        status="PASS",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        component="live-production-path",
        failure_category="",
        run_id="run-9",
        subchecks=[],
        live_trusted=True,
    )
    (out_dir / GATE_EVIDENCE_FILES["C"]).write_text(
        json.dumps(sign_evidence_body(c_body)) + "\n", encoding="utf-8"
    )
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(_live_out_payload(served_ms=[1000.0, 2000.0, 3000.0])) + "\n",
        encoding="utf-8",
    )
    evidence, _, _ = runner._emit_gate_e_from_evidence(
        expected=SHA, run_id="run-9", out_dir=out_dir
    )
    assert evidence.status == "PASS"
    assert evidence.live_trusted is True
    assert evidence.latency_p95_ms > 0


def test_emit_e_blocked_when_c_not_live_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    c_body = build_evidence_body(
        gate="C",
        status="BLOCKED",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        component="live-production-path",
        failure_category="live-evidence-required",
        run_id="run-9",
        subchecks=[],
        live_trusted=False,
    )
    (out_dir / GATE_EVIDENCE_FILES["C"]).write_text(
        json.dumps(sign_evidence_body(c_body)) + "\n", encoding="utf-8"
    )
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(_live_out_payload(served_ms=[1000.0])) + "\n", encoding="utf-8"
    )
    evidence, _, _ = runner._emit_gate_e_from_evidence(
        expected=SHA, run_id="run-9", out_dir=out_dir
    )
    assert evidence.status == "BLOCKED"
    assert evidence.failure_category == "missing-live-samples"


def test_gate_d_passes_on_fresh_stopped_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aa.control.runtime_status import RuntimeMarker, write_marker

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_marker(
        out_dir / "runtime-status.json",
        RuntimeMarker(run_id="run-9", sha=SHA, phase="STOPPED", timestamp_s=time.time()),
    )
    (out_dir / "runtime-timings.json").write_text(
        json.dumps(
            {
                "sha": SHA,
                "run_id": "run-9",
                "identity_kind": "isolated-test-bot",
                "startup_ms": 1200.0,
                "run_ms": 5000.0,
                "polling_verified": True,
                "stop_clean": True,
                "external_send_status": EXTERNAL_UNVERIFIED,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = runner._gate_d(SHA, "run-9", PRODUCT, RUNTIME)
    assert evidence.status == "PASS"
    assert evidence.live_trusted is True
    assert evidence.latency_p50_ms == 1200.0


def test_gate_d_rejects_stale_sha_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aa.control.runtime_status import RuntimeMarker, write_marker

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.delenv("SELF_PROVING_LIVE", raising=False)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_marker(
        out_dir / "runtime-status.json",
        RuntimeMarker(run_id="run-9", sha=OTHER_SHA, phase="STOPPED", timestamp_s=time.time()),
    )
    evidence = runner._gate_d(SHA, "run-9", PRODUCT, RUNTIME)
    assert evidence.status == "BLOCKED"


def test_gate_d_failed_marker_is_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aa.control.runtime_status import RuntimeMarker, write_marker

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_marker(
        out_dir / "runtime-status.json",
        RuntimeMarker(run_id="run-9", sha=SHA, phase="FAILED", timestamp_s=time.time()),
    )
    evidence = runner._gate_d(SHA, "run-9", PRODUCT, RUNTIME)
    assert evidence.status == "FAIL"


def _full_evidence_dir(tmp_path: Path, *, run_id: str = "run-9") -> Path:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _signed("A", "PASS", _subs(REQUIRED_A_CHECKS, "gate-a"), run_id=run_id),
        _signed("B", "PASS", _subs(REQUIRED_B_CHECKS, "gate-b"), run_id=run_id),
        _signed(
            "C",
            "FAIL",
            [],
            run_id=run_id,
            component="live-production-path",
            category="live-path-failed",
            live=True,
        ),
        _signed(
            "D",
            "PASS",
            [],
            run_id=run_id,
            component="telegram-readiness",
            live=True,
        ),
        _signed(
            "E",
            "FAIL",
            [],
            run_id=run_id,
            component="slo",
            category="latency-budget-exceeded",
            live=True,
        ),
    ]
    for signed in fixtures:
        gate = str(signed["gate"])
        (evidence_dir / f"gate-{gate}-evidence.json").write_text(
            json.dumps(signed) + "\n", encoding="utf-8"
        )
    return evidence_dir


def test_final_verdict_names_all_failed_gates_without_b_blame(tmp_path: Path) -> None:
    evidence_dir = _full_evidence_dir(tmp_path)
    out_dir = tmp_path / "out"
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
    )
    assert rc == 1
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["blocking_gate"] == "C"
    assert result["failed_gates"] == ["C", "E"]
    assert "B" not in result["failed_gates"]
    assert "D" not in result["failed_gates"]


def test_optional_unverified_send_does_not_block_final(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for signed in [
        _signed("A", "PASS", _subs(REQUIRED_A_CHECKS, "gate-a")),
        _signed("B", "PASS", _subs(REQUIRED_B_CHECKS, "gate-b")),
        _signed("C", "PASS", [], component="live-production-path", live=True),
        _signed("D", "PASS", [], component="telegram-readiness", live=True),
        _signed("E", "PASS", [], component="slo", live=True),
    ]:
        gate = str(signed["gate"])
        (evidence_dir / f"gate-{gate}-evidence.json").write_text(
            json.dumps(signed) + "\n", encoding="utf-8"
        )
    out_dir = tmp_path / "out"
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
    )
    assert rc == 0
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["result"] == "PASS"
    assert result["failed_gates"] == []


def test_typed_checkpoint_round_trip_and_stale_rejection(tmp_path: Path) -> None:
    body = build_checkpoint_body(
        gate="E",
        sha=SHA,
        run_id="run-9",
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        attempt="2",
        evidence_checksum="e" * 64,
    )
    signed = sign_checkpoint_body(body)
    assert checkpoint_checksum(body) == signed["checkpoint_checksum"]
    target = write_checkpoint_atomic(tmp_path, signed)
    loaded = load_and_validate_checkpoint(
        target,
        expected_gate="E",
        expected_sha=SHA,
        expected_run_id="run-9",
        expected_product=PRODUCT,
        expected_runtime=RUNTIME,
    )
    assert loaded["gate"] == "E"
    with pytest.raises(SelfProvingError):
        load_and_validate_checkpoint(
            target, expected_gate="E", expected_sha=OTHER_SHA, expected_run_id="run-9"
        )
    with pytest.raises(SelfProvingError):
        load_and_validate_checkpoint(
            target, expected_gate="E", expected_sha=SHA, expected_run_id="other-run"
        )


def test_checkpoint_skip_binds_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    assert (
        runner.checkpoint_allows_skip(
            checkpoint_dir=ckpt_dir,
            out_dir=out_dir,
            gate="C",
            expected_sha=SHA,
            run_id="run-9",
        )
        is False
    )
    c_body = build_evidence_body(
        gate="C",
        status="PASS",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        component="live-production-path",
        failure_category="",
        run_id="run-9",
        subchecks=[],
        live_trusted=True,
    )
    signed_evidence = sign_evidence_body(c_body)
    (out_dir / GATE_EVIDENCE_FILES["C"]).write_text(
        json.dumps(signed_evidence) + "\n", encoding="utf-8"
    )
    checkpoint = sign_checkpoint_body(
        build_checkpoint_body(
            gate="C",
            sha=SHA,
            run_id="run-9",
            product_fingerprint=PRODUCT,
            runtime_fingerprint=RUNTIME,
            attempt="1",
            evidence_checksum=str(signed_evidence["evidence_checksum"]),
        )
    )
    write_checkpoint_atomic(ckpt_dir, checkpoint)
    assert (
        runner.checkpoint_allows_skip(
            checkpoint_dir=ckpt_dir,
            out_dir=out_dir,
            gate="C",
            expected_sha=SHA,
            run_id="run-9",
        )
        is True
    )
    assert (
        runner.checkpoint_allows_skip(
            checkpoint_dir=ckpt_dir,
            out_dir=out_dir,
            gate="C",
            expected_sha=OTHER_SHA,
            run_id="run-9",
        )
        is False
    )


def test_emit_c_and_d_publish_single_owner_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as _time

    from aa.control.runtime_status import RuntimeMarker, write_marker

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("AA_GATE_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    out_dir = tmp_path / "self-proving-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(
            {
                "main_sha": SHA,
                "run_id": "run-9",
                "status": "FAIL",
                "lanes": [
                    {
                        "lane": "live-telegram-evidence",
                        "status": "FAIL",
                        "passed": [],
                        "failed": ["live-answer-no-generic-collapse"],
                        "incomplete": [],
                        "metrics": {
                            "turn_latencies_ms": [1000.0],
                            "gate_boundary": {
                                "telegram_egress_mode": "stubbed",
                                "delivery_receipt_kind": "simulated-in-process",
                            },
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc = runner._run_emit_gate(
        gate="C",
        expected=SHA,
        run_id="run-9",
        out_dir=out_dir,
        shell_status="missing",
        shell_detail="",
    )
    assert rc == 1
    c_loaded = load_and_validate_evidence(
        out_dir / GATE_EVIDENCE_FILES["C"],
        expected_gate="C",
        expected_sha=SHA,
        expected_run_id="run-9",
    )
    assert c_loaded["status"] == "FAIL"
    write_marker(
        out_dir / "runtime-status.json",
        RuntimeMarker(run_id="run-9", sha=SHA, phase="STOPPED", timestamp_s=_time.time()),
    )
    rc = runner._run_emit_gate(
        gate="D",
        expected=SHA,
        run_id="run-9",
        out_dir=out_dir,
        shell_status="missing",
        shell_detail="",
    )
    assert rc == 0
    d_loaded = load_and_validate_evidence(
        out_dir / GATE_EVIDENCE_FILES["D"],
        expected_gate="D",
        expected_sha=SHA,
        expected_run_id="run-9",
    )
    assert d_loaded["status"] == "PASS"


def test_coerce_samples_rejects_mixed_series() -> None:
    raw = [
        {
            "causality_id": new_causality_id(),
            "family": "ordinary-text",
            "mode": "text",
            "outcome": "served",
            "latency_ms": 1000.0,
            "monotonic": True,
            "clock": "monotonic-perf_counter",
            "series": "d-runtime-readiness",
            "boundary": "telegram-startup",
            "receipt_kind": "simulated-in-process",
            "censored": False,
        }
    ]
    with pytest.raises(GateLatencyError):
        coerce_samples(raw, series=C_SERIES)
