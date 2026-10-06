"""Live Product Contract lane tests (issue #142).

Proves qualification #7 can execute every live lane through the exact
production boundary. External live prerequisites (bot token, encrypted
snapshot identity, voice models) stay fail-closed INCOMPLETE, never a
fake PASS.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aa.qualification.product_contract_live import (
    EXIT_BY_STATUS,
    ProductContractLiveError,
    assert_no_text_leak,
    build_result_marker,
    collect_static_gates,
    decide_status,
    run_control_lane,
    run_voice_lane,
    validate_exact_sha,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_exact_sha_rejects_malformed() -> None:
    with pytest.raises(ProductContractLiveError):
        validate_exact_sha("not-a-sha")
    assert validate_exact_sha("a" * 40) == "a" * 40


def test_decide_status_is_fail_closed() -> None:
    assert decide_status(["PASS", "PASS", "PASS", "PASS", "PASS"]) == "PASS"
    assert decide_status(["PASS", "INCOMPLETE", "PASS", "PASS", "PASS"]) == "INCOMPLETE"
    assert decide_status(["PASS", "FAIL", "INCOMPLETE", "PASS", "PASS"]) == "FAIL"


def test_result_marker_format() -> None:
    marker = build_result_marker(sha="a" * 40, status="PASS", run="123")
    assert "issue=7" in marker
    assert ("a" * 40) in marker
    with pytest.raises(ProductContractLiveError):
        build_result_marker(sha="short", status="PASS", run="123")


def test_static_gates_carry_digests_only() -> None:
    gates = collect_static_gates(REPO_ROOT)
    assert len(gates["product_fingerprint"]) == 64
    assert gates["benchmark_total_substantive"] == 82
    assert_no_text_leak(gates)
    for key in ("utterance", "answer", "evidence_text", "transcript"):
        assert key not in gates


async def test_message_lane_executes_production_boundary() -> None:
    from aa.qualification.product_contract_live import run_message_lane

    result = await run_message_lane()
    assert result.lane == "product-contract-1-24"
    # Offline deterministic lanes must fully pass: the production
    # boundary is live-executed with stubbed externals.
    assert result.status == "PASS", result.failed
    assert len(result.passed) >= 20
    assert result.metrics["scenarios_executed"] == 24


async def test_transport_lane_reports_live_or_incomplete() -> None:
    from aa.qualification.product_contract_live import run_transport_lane

    result = await run_transport_lane()
    assert result.lane == "telegram-transport-25-32"
    # Without a live bot token the real-stream sublane is INCOMPLETE
    # (fail-closed); deterministic semantics still execute live.
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["scenarios_executed"] == 8
    if result.status == "INCOMPLETE":
        assert result.incomplete


def test_control_lane_bounds_and_restore() -> None:
    result = run_control_lane(REPO_ROOT)
    assert result.lane == "runtime-control-33-41"
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["max_starts"] == 4
    assert result.metrics["runtime_seconds"] == 18000


def test_voice_lane_policy_and_gates() -> None:
    result = run_voice_lane(REPO_ROOT)
    assert result.lane == "voice-1-16"
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["scenarios_executed"] == 16
    assert result.metrics["peak_rss_mb"] <= 12288
    assert_no_text_leak(result.to_dict())


def test_live_runner_stale_on_sha_mismatch(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            "scripts/run_product_contract_live_qualification.py",
            "--main-sha",
            "b" * 40,
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == EXIT_BY_STATUS["STALE"]
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] == "STALE"


def test_live_runner_executes_all_lanes(tmp_path: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    ).stdout.strip()
    proc = subprocess.run(
        [
            "python3",
            "scripts/run_product_contract_live_qualification.py",
            "--main-sha",
            head,
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # Offline working tree may be dirty (new untracked files) before the
    # repair merges; the runner stays fail-closed INCOMPLETE then. The key
    # assertion: either all five lanes executed with structured evidence,
    # or the runner refused the dirty tree without faking a result.
    assert proc.returncode in (EXIT_BY_STATUS["PASS"], EXIT_BY_STATUS["INCOMPLETE"])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] in ("PASS", "INCOMPLETE")
    if "lanes" not in payload:
        assert "clean" in str(payload.get("reason", "")).casefold()
        return
    assert len(payload["lanes"]) == 5
    assert {lane["lane"] for lane in payload["lanes"]} == {
        "product-contract-1-24",
        "telegram-transport-25-32",
        "runtime-control-33-41",
        "voice-1-16",
        "live-telegram-evidence",
    }
    summary = json.loads(
        (tmp_path / "product-contract-live-summary.json").read_text(encoding="utf-8")
    )
    assert_no_text_leak(summary)
