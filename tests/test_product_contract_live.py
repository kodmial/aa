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
    working_tree_clean,
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


def test_working_tree_clean_ignores_ephemeral_outputs(tmp_path: Path) -> None:
    # Regression for kodmial/aa#151 (Gate unknown): Gate C created live-out/
    # before the clean check, and Gate B downloads models/, so a pristine
    # checkout always reported "working tree is not clean". Ephemeral outputs
    # must never count as dirty; real source edits still fail closed.
    import subprocess

    repo = tmp_path / "probe-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True)
    (repo / "tracked.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)
    assert working_tree_clean(repo) is True
    for name in (
        "live-out",
        "self-proving-out",
        "eval-self-proving-out",
        "aa-self-proving-checkpoints",
        "models",
    ):
        target = repo / name
        target.mkdir(exist_ok=True)
        (target / "output.json").write_text("{}", encoding="utf-8")
    assert working_tree_clean(repo) is True
    (repo / "runtime-status.json").write_text("{}", encoding="utf-8")
    assert working_tree_clean(repo) is True
    # Real untracked sources still fail closed.
    (repo / "new-source.py").write_text("x=1", encoding="utf-8")
    assert working_tree_clean(repo) is False
    (repo / "new-source.py").unlink()
    assert working_tree_clean(repo) is True
    # Tracked modifications still fail closed.
    (repo / "tracked.txt").write_text("edited", encoding="utf-8")
    assert working_tree_clean(repo) is False


def test_async_lanes_attribute_harness_crash_instead_of_aborting() -> None:
    # Regression for kodmial/aa#150: a missing third-party module
    # (langchain_core on a minimal runner) raised an unhandled
    # ModuleNotFoundError from the production import chain, so the live
    # runner produced no result.json and no lane evidence. Each lane must
    # degrade to an attributable FAIL instead of aborting the evaluation.
    import asyncio

    import aa.qualification.product_contract_live as live_mod
    from aa.qualification.product_contract_live import LaneResult

    async def _boom() -> LaneResult:
        raise ModuleNotFoundError("No module named 'langchain_core'")

    async def _ok_transport() -> LaneResult:
        return LaneResult(lane="telegram-transport-25-32", status="PASS", passed=("ok",))

    async def _ok_live() -> LaneResult:
        return LaneResult(
            lane="live-telegram-evidence", status="INCOMPLETE", incomplete=("deferred",)
        )

    original_message = live_mod.run_message_lane
    original_transport = live_mod.run_transport_lane
    original_live = live_mod.run_live_telegram_evidence_lane
    live_mod.run_message_lane = _boom  # type: ignore[assignment]
    live_mod.run_transport_lane = _ok_transport  # type: ignore[assignment]
    live_mod.run_live_telegram_evidence_lane = _ok_live  # type: ignore[assignment]
    try:
        message, transport, live_evidence = asyncio.run(live_mod._run_async_lanes())
    finally:
        live_mod.run_message_lane = original_message  # type: ignore[assignment]
        live_mod.run_transport_lane = original_transport  # type: ignore[assignment]
        live_mod.run_live_telegram_evidence_lane = original_live  # type: ignore[assignment]
    assert message.lane == "product-contract-1-24"
    assert message.status == "FAIL"
    assert any("ModuleNotFoundError" in item for item in message.failed)
    assert_no_text_leak(message.to_dict())
    assert transport.status == "PASS"
    assert live_evidence.status == "INCOMPLETE"


def test_ephemeral_gate_checkpoints_stay_untracked() -> None:
    # Regression for kodmial/aa#150: root gate-*.done resume markers were
    # committed as tracked files although the workflow stages checkpoints
    # under aa-self-proving-checkpoints/. They must stay gitignored.
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "gate-*.done" in gitignore
    assert not (REPO_ROOT / "gate-A.done").exists()
    assert not (REPO_ROOT / "gate-B.done").exists()


def test_self_proving_gate_f_always_emits_verdict() -> None:
    # Regression for kodmial/aa#151 unknown-gate-failure: Gate F must run even
    # after a Gate C failure so the repair publisher maps a concrete blocking
    # gate instead of "unknown".
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    gate_f_index = workflow.index("Gate F - exact-main final verdict")
    gate_f_block = workflow[gate_f_index : gate_f_index + 800]
    assert "if: always()" in gate_f_block


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
