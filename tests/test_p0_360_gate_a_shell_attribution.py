"""P0 kodmial/aa#360: Gate A failures must retain per-check identity.

Systemic failure class covered here: distinct static/build defects collapsed
into one ``A:shell-step-failed:gate-a-shell`` fingerprint, so the repair loop
could not distinguish an entrypoint failure from a unit, lint, format, type,
contract, or runtime fixture failure. This is generic attribution coverage,
not a literal qualification prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_self_proving_qualification as runner  # noqa: E402
from aa.qualification.gate_a_shell import (  # noqa: E402
    GATE_A_SHELL_CHECKS,
    SHELL_CHECK_IDS,
    GateAShellResult,
    build_shell_results_body,
    load_shell_results,
    write_shell_results,
)
from aa.qualification.gate_evidence import (  # noqa: E402
    GATE_A_COVERAGE,
    SubcheckResult,
)
from aa.qualification.self_proving import (  # noqa: E402
    GateEvidence,
    SelfProvingError,
    failure_report_for_gate,
    repair_fingerprint,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
RUN_ID = "run-360"
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _python_evidence() -> GateEvidence:
    return GateEvidence(
        gate="A",
        status="PASS",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id=RUN_ID,
        live_trusted=True,
    )


def _passing_results() -> dict[str, GateAShellResult]:
    return {check_id: GateAShellResult(check_id, "PASS", 0, 1) for check_id in SHELL_CHECK_IDS}


def _failed_verdict(check_id: str) -> tuple[GateEvidence, list[SubcheckResult]]:
    results = _passing_results()
    results[check_id] = GateAShellResult(check_id, "FAIL", 1, 5, "exit-code-1")
    subchecks = runner._gate_a_subchecks(
        _python_evidence(),
        "fail",
        "workflow-gate-a-shell-fail",
        results,
    )
    verdict = runner._combine_ab_verdict(_python_evidence(), subchecks)
    return verdict, subchecks


def _write_results(
    tmp_path: Path,
    results: dict[str, GateAShellResult],
    *,
    sha: str = SHA,
    run_id: str = RUN_ID,
) -> Path:
    path = tmp_path / "gate-a-shell-results.json"
    write_shell_results(path, main_sha=sha, run_id=run_id, results=results.values())
    return path


def test_canonical_and_live_gate_a_use_one_shell_inventory() -> None:
    root = Path(__file__).resolve().parents[1]
    verify = (root / "scripts" / "verify.sh").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/run_gate_a_shell.py" in verify
    assert "--profile canonical" in verify
    assert "--profile gate-a" in workflow
    assert "--shell-a-results" in workflow


def test_shell_inventory_matches_every_gate_a_shell_coverage_row() -> None:
    covered = tuple(row["check_id"] for row in GATE_A_COVERAGE if row["owner"] == "gate-a-shell")
    assert SHELL_CHECK_IDS == covered
    assert len({check.check_id for check in GATE_A_SHELL_CHECKS}) == len(GATE_A_SHELL_CHECKS)


def test_distinct_shell_failures_keep_distinct_components_and_fingerprints() -> None:
    pytest_verdict, pytest_subchecks = _failed_verdict("a-shell-pytest")
    ruff_verdict, ruff_subchecks = _failed_verdict("a-shell-ruff-check")

    assert pytest_verdict.status == ruff_verdict.status == "FAIL"
    assert pytest_verdict.failure_category == ruff_verdict.failure_category == ("shell-step-failed")
    assert pytest_verdict.component == "a-shell-pytest"
    assert ruff_verdict.component == "a-shell-ruff-check"
    assert [item.check_id for item in pytest_subchecks if item.status == "FAIL"] == [
        "a-shell-pytest"
    ]
    assert [item.check_id for item in ruff_subchecks if item.status == "FAIL"] == [
        "a-shell-ruff-check"
    ]
    assert repair_fingerprint(failure_report_for_gate(pytest_verdict)) != (
        repair_fingerprint(failure_report_for_gate(ruff_verdict))
    )


def test_shell_pass_status_cannot_mask_a_failed_check() -> None:
    results = _passing_results()
    results["a-shell-mypy"] = GateAShellResult("a-shell-mypy", "FAIL", 1, 1, "exit-code-1")
    subchecks = runner._gate_a_subchecks(_python_evidence(), "pass", "", results)
    verdict = runner._combine_ab_verdict(_python_evidence(), subchecks)
    assert verdict.status == "BLOCKED"


def test_invalid_attribution_fails_closed_for_pass_status() -> None:
    subchecks = runner._gate_a_subchecks(
        _python_evidence(),
        "pass",
        "",
        None,
        "shell-results-invalid:SelfProvingError",
    )
    verdict = runner._combine_ab_verdict(_python_evidence(), subchecks)
    assert verdict.status == "BLOCKED"


def test_invalid_attribution_keeps_real_shell_failure_blocking() -> None:
    subchecks = runner._gate_a_subchecks(
        _python_evidence(),
        "fail",
        "workflow-gate-a-shell-fail",
        None,
        "shell-results-invalid:SelfProvingError",
    )
    verdict = runner._combine_ab_verdict(_python_evidence(), subchecks)
    assert verdict.status == "FAIL"
    assert verdict.failure_category == "shell-step-failed"


def test_shell_result_artifact_is_bound_to_exact_sha_and_run(tmp_path: Path) -> None:
    path = _write_results(tmp_path, _passing_results())
    loaded = load_shell_results(path, expected_sha=SHA, expected_run_id=RUN_ID)
    assert set(loaded) == set(SHELL_CHECK_IDS)
    with pytest.raises(SelfProvingError):
        load_shell_results(path, expected_sha=OTHER_SHA, expected_run_id=RUN_ID)
    with pytest.raises(SelfProvingError):
        load_shell_results(path, expected_sha=SHA, expected_run_id="other-run")


def test_shell_result_artifact_rejects_omitted_checks(tmp_path: Path) -> None:
    results = _passing_results()
    del results["a-shell-mypy"]
    path = _write_results(tmp_path, results)
    with pytest.raises(SelfProvingError):
        load_shell_results(path, expected_sha=SHA, expected_run_id=RUN_ID)


def test_shell_result_artifact_rejects_inconsistent_exit_evidence(
    tmp_path: Path,
) -> None:
    results = _passing_results()
    results["a-shell-pytest"] = GateAShellResult("a-shell-pytest", "PASS", 1, 1)
    path = _write_results(tmp_path, results)
    with pytest.raises(SelfProvingError):
        load_shell_results(path, expected_sha=SHA, expected_run_id=RUN_ID)


def test_shell_result_body_is_privacy_safe_and_complete() -> None:
    payload = build_shell_results_body(
        main_sha=SHA, run_id=RUN_ID, results=_passing_results().values()
    )
    assert payload["schema_version"] == "aa-gate-a-shell-results/1"
    assert payload["main_sha"] == SHA
    assert payload["run_id"] == RUN_ID
    assert len(payload["checks"]) == len(SHELL_CHECK_IDS)
