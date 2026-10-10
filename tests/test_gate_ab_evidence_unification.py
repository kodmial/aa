"""Unified Gate A/B evidence and read-only Gate F (issue #339).

Acceptance: one authoritative verdict per gate; A/B execute once and publish
a signed artifact each; Gate F is a pure deterministic reducer over validated
evidence; blocking_gate (human summary) differs from failed_gates (all);
every fail-closed negative (old SHA, stale run, digest mismatch, omitted
check, missing/forged artifact) holds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_self_proving_qualification as runner  # noqa: E402
from aa.qualification.gate_evidence import (  # noqa: E402
    COVERAGE_SCHEMA_VERSION,
    GATE_A_COVERAGE,
    GATE_B_COVERAGE,
    REQUIRED_A_CHECKS,
    REQUIRED_B_CHECKS,
    SubcheckResult,
    build_evidence_body,
    coverage_table,
    evidence_to_gate_evidence,
    failed_gates_for_verdict,
    load_and_validate_evidence,
    root_components_for_verdict,
    sign_evidence_body,
    verdict_summary_v2,
    write_evidence_atomic,
)
from aa.qualification.self_proving import (  # noqa: E402
    GATE_B_COMPONENTS,
    GateEvidence,
    SelfProvingError,
    decide_final_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER_SHA = "b" * 40
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _subs(checks: tuple[str, ...], owner: str) -> list[SubcheckResult]:
    return [SubcheckResult(check_id=name, status="PASS", component=owner) for name in checks]


def _signed(
    gate: str,
    status: str,
    subs: list[SubcheckResult],
    *,
    sha: str = SHA,
    run_id: str = "run-9",
    product: str = PRODUCT,
    runtime: str = RUNTIME,
    component: str = "",
    category: str = "",
    live: bool = False,
) -> dict[str, Any]:
    body = build_evidence_body(
        gate=gate,
        status=status,
        sha=sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component=component,
        failure_category=category,
        run_id=run_id,
        subchecks=subs,
        live_trusted=live,
    )
    return sign_evidence_body(body)


def _write_evidence(tmp_path: Path, signed: dict[str, object]) -> Path:
    gate = str(signed["gate"])
    target = tmp_path / f"gate-{gate}-evidence.json"
    target.write_text(json.dumps(signed, indent=2) + "\n", encoding="utf-8")
    return target


def _evidence_dir(
    tmp_path: Path,
    *,
    b_status: str = "PASS",
    b_component: str = "all-components",
    b_category: str = "",
    c_status: str = "PASS",
    e_status: str = "PASS",
    run_id: str = "run-9",
    sha: str = SHA,
) -> Path:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _signed("A", "PASS", _subs(REQUIRED_A_CHECKS, "gate-a"), sha=sha, run_id=run_id),
        _signed(
            "B",
            b_status,
            _subs(REQUIRED_B_CHECKS, "gate-b"),
            sha=sha,
            run_id=run_id,
            component=b_component,
            category=b_category,
        ),
        _signed(
            "C",
            c_status,
            [],
            sha=sha,
            run_id=run_id,
            component="live-production-path",
            category="" if c_status == "PASS" else "live-path-failed",
            live=True,
        ),
        _signed("D", "PASS", [], sha=sha, run_id=run_id, component="telegram-readiness", live=True),
        _signed(
            "E",
            e_status,
            [],
            sha=sha,
            run_id=run_id,
            component="slo",
            category="" if e_status == "PASS" else "latency-budget-exceeded",
            live=True,
        ),
    ]
    for signed in fixtures:
        gate = str(signed["gate"])
        (evidence_dir / f"gate-{gate}-evidence.json").write_text(
            json.dumps(signed, indent=2) + "\n", encoding="utf-8"
        )
    return evidence_dir


# -- coverage mapping -------------------------------------------------------


def test_coverage_mapping_has_owner_input_check_artifact_status() -> None:
    table = coverage_table()
    assert table["schema_version"] == COVERAGE_SCHEMA_VERSION
    assert len(table["gate_a"]) == len(GATE_A_COVERAGE) == len(REQUIRED_A_CHECKS)
    assert len(table["gate_b"]) == len(GATE_B_COVERAGE) == len(REQUIRED_B_CHECKS)
    for row in (*table["gate_a"], *table["gate_b"]):
        for key in ("check_id", "owner", "input", "executable_check", "proof_artifact", "status"):
            assert row[key], f"coverage row missing {key}: {row.get('check_id')}"
        assert row["status"] == "active"


def test_coverage_committed_file_matches_module() -> None:
    committed = json.loads((REPO_ROOT / "qualification" / "gate_ab_coverage.json").read_text())
    assert committed == coverage_table()


def test_no_gate_b_component_silently_dropped() -> None:
    covered = {row["check_id"] for row in GATE_B_COVERAGE}
    for component in GATE_B_COMPONENTS:
        assert f"b-{component}" in covered or component in (
            "retrieval-dedup",
            "retrieval-diversity",
        ), f"component {component} has no coverage check"


def test_no_gate_a_assertion_silently_dropped() -> None:
    covered = {row["check_id"] for row in GATE_A_COVERAGE}
    for required in (
        "a-exact-sha",
        "a-clean-tree",
        "a-product-fingerprint",
        "a-runtime-fingerprint",
        "a-model-policy",
        "a-shell-aa-check",
        "a-shell-pytest",
        "a-shell-ruff-check",
        "a-shell-ruff-format",
        "a-shell-mypy",
        "a-shell-contract-qualification",
        "a-shell-runtime-qualification",
    ):
        assert required in covered


# -- V2 evidence sign/validate ----------------------------------------------


def test_evidence_round_trip_and_checksum(tmp_path: Path) -> None:
    signed = _signed("A", "PASS", _subs(REQUIRED_A_CHECKS, "gate-a"))
    path = _write_evidence(tmp_path, signed)
    loaded = load_and_validate_evidence(
        path, expected_gate="A", expected_sha=SHA, expected_run_id="run-9"
    )
    assert loaded["evidence_checksum"] == signed["evidence_checksum"]
    assert evidence_to_gate_evidence(loaded).status == "PASS"


def test_evidence_tampering_fails_closed(tmp_path: Path) -> None:
    signed = _signed("B", "PASS", _subs(REQUIRED_B_CHECKS, "gate-b"))
    forged = dict(signed)
    forged["status"] = "PASS"
    raw_subchecks = forged["subchecks"]
    assert isinstance(raw_subchecks, list)
    subchecks = [dict(item) for item in raw_subchecks]
    subchecks[0]["status"] = "FAIL"
    forged["subchecks"] = subchecks
    path = _write_evidence(tmp_path, forged)
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="B", expected_sha=SHA, expected_run_id="run-9"
        )


def test_evidence_legacy_rejected(tmp_path: Path) -> None:
    legacy = {
        "gate": "A",
        "status": "PASS",
        "sha": SHA,
        "run_id": "run-9",
        "subchecks": [],
    }
    path = tmp_path / "gate-A-evidence.json"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="A", expected_sha=SHA, expected_run_id="run-9"
        )


def test_evidence_old_sha_stale_run_fingerprint_mismatch_fail_closed(tmp_path: Path) -> None:
    signed = _signed("A", "PASS", _subs(REQUIRED_A_CHECKS, "gate-a"))
    path = _write_evidence(tmp_path, signed)
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="A", expected_sha=OTHER_SHA, expected_run_id="run-9"
        )
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="A", expected_sha=SHA, expected_run_id="other-run"
        )
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path,
            expected_gate="A",
            expected_sha=SHA,
            expected_run_id="run-9",
            expected_product="f" * 64,
        )
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="B", expected_sha=SHA, expected_run_id="run-9"
        )


def test_evidence_omitted_check_fails_closed(tmp_path: Path) -> None:
    partial = _subs(REQUIRED_A_CHECKS, "gate-a")[:-1]
    signed = _signed("A", "PASS", partial)
    path = _write_evidence(tmp_path, signed)
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            path, expected_gate="A", expected_sha=SHA, expected_run_id="run-9"
        )


def test_evidence_missing_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SelfProvingError):
        load_and_validate_evidence(
            tmp_path / "gate-A-evidence.json",
            expected_gate="A",
            expected_sha=SHA,
            expected_run_id="run-9",
        )


def test_evidence_atomic_write_no_partial(tmp_path: Path) -> None:
    signed = _signed("B", "PASS", _subs(REQUIRED_B_CHECKS, "gate-b"))
    target = write_evidence_atomic(tmp_path, signed)
    assert target.is_file()
    assert list(tmp_path.glob(".*.tmp")) == []
    load_and_validate_evidence(target, expected_gate="B", expected_sha=SHA, expected_run_id="run-9")


# -- single authoritative A/B verdict ----------------------------------------


def _python_b(status: str, component: str, detail: str = "") -> GateEvidence:
    return GateEvidence(
        gate="B",
        status=status,
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        failure_category="component-fail" if status == "FAIL" else "",
        component=component,
        run_id="run-9",
        detail=detail,
    )


def test_b_shell_pass_python_pass_is_single_pass() -> None:
    verdict = runner._combine_ab_verdict(
        _python_b("PASS", "all-components"),
        runner._gate_b_subchecks(_python_b("PASS", "all-components"), "pass", "pytest-ok"),
    )
    assert verdict.status == "PASS"


def test_b_shell_pass_python_subcheck_fail_is_single_fail() -> None:
    python_evidence = _python_b("FAIL", "grounding", "unsupported claim accepted")
    verdict = runner._combine_ab_verdict(
        python_evidence, runner._gate_b_subchecks(python_evidence, "pass", "pytest-ok")
    )
    assert verdict.status == "FAIL"
    assert verdict.component == "grounding"


def test_b_shell_failed_python_pass_is_still_fail() -> None:
    python_evidence = _python_b("PASS", "all-components")
    verdict = runner._combine_ab_verdict(
        python_evidence, runner._gate_b_subchecks(python_evidence, "fail", "pytest-exit-1")
    )
    assert verdict.status == "FAIL"
    assert verdict.failure_category == "shell-step-failed"


def test_b_shell_missing_python_pass_is_blocked_never_pass() -> None:
    python_evidence = _python_b("PASS", "all-components")
    verdict = runner._combine_ab_verdict(
        python_evidence, runner._gate_b_subchecks(python_evidence, "missing", "")
    )
    assert verdict.status == "BLOCKED"


def test_b_missing_assets_is_blocked_not_semantic_fail() -> None:
    python_evidence = _python_b(
        "FAIL", "corpus-restore", "missing production artifacts: lexical.db"
    )
    verdict = runner._combine_ab_verdict(
        python_evidence, runner._gate_b_subchecks(python_evidence, "fail", "bootstrap-exit-1")
    )
    assert verdict.status == "BLOCKED"
    assert verdict.failure_category == "proof-unavailable"


def test_a_shell_failed_python_pass_is_fail() -> None:
    python_evidence = GateEvidence(
        gate="A",
        status="PASS",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-9",
        live_trusted=True,
    )
    verdict = runner._combine_ab_verdict(
        python_evidence, runner._gate_a_subchecks(python_evidence, "fail", "verify-sh-fail")
    )
    assert verdict.status == "FAIL"


# -- read-only Gate F --------------------------------------------------------


def test_gate_f_pure_reducer_no_ab_reevaluation(tmp_path: Path) -> None:
    evidence_dir = _evidence_dir(tmp_path)
    out_dir = tmp_path / "out"
    calls: list[str] = []

    def _bomb(*args: object, **kwargs: object) -> object:
        calls.append("invoked")
        raise AssertionError("Gate F must not re-evaluate gates")

    for name in (
        "_gate_a",
        "_gate_b",
        "_gate_c",
        "_gate_d",
        "_gate_e",
        "_product_fingerprint",
        "_runtime_fingerprint",
    ):
        setattr(runner, name, _bomb)
    try:
        rc = runner.reduce_final_from_evidence_dir(
            evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
        )
    finally:
        import importlib

        importlib.reload(runner)
    assert rc == 0
    assert calls == []
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["result"] == "PASS"


def test_gate_f_c_and_e_fail_b_stays_pass(tmp_path: Path) -> None:
    evidence_dir = _evidence_dir(tmp_path, c_status="FAIL", e_status="FAIL")
    out_dir = tmp_path / "out"
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
    )
    assert rc == 1
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    gates = {item["gate"]: item["status"] for item in result["gates"]}
    assert gates["B"] == "PASS"
    assert result["blocking_gate"] == "C"
    assert result["failed_gates"] == ["C", "E"]
    assert "B" not in result["failed_gates"]
    assert result["root_components"] == [
        "C:live-path-failed:live-production-path",
        "E:latency-budget-exceeded:slo",
    ]
    matrix = json.loads((out_dir / "failure-matrix.json").read_text(encoding="utf-8"))
    assert matrix["failed_gates"] == ["C", "E"]
    assert len(matrix["failures"]) == 2


def test_gate_f_verified_component_never_flips_without_new_evidence(tmp_path: Path) -> None:
    evidence_dir = _evidence_dir(tmp_path)
    out_dir = tmp_path / "out"
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
    )
    assert rc == 0
    summary = json.loads((out_dir / "self-proving-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["failed_gates"] == []
    assert all(item["status"] == "PASS" for item in summary["gates"])


def test_gate_f_fail_closed_negatives(tmp_path: Path) -> None:
    # Old SHA evidence.
    stale_dir = _evidence_dir(tmp_path / "stale", sha=OTHER_SHA)
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=stale_dir,
        out_dir=tmp_path / "stale-out",
        expected_sha=SHA,
        run_id="run-9",
    )
    assert rc == 2
    # Stale run identity.
    run_dir = _evidence_dir(tmp_path / "run", run_id="old-run")
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=run_dir, out_dir=tmp_path / "run-out", expected_sha=SHA, run_id="run-9"
    )
    assert rc == 2
    # Inconsistent fingerprints across gates.
    mixed = _evidence_dir(tmp_path / "mixed")
    payload = json.loads((mixed / "gate-E-evidence.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "evidence_checksum"}
    body["product_fingerprint"] = "f" * 64
    from aa.qualification.gate_evidence import evidence_checksum

    payload = dict(body, evidence_checksum=evidence_checksum(body))
    (mixed / "gate-E-evidence.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=mixed, out_dir=tmp_path / "mixed-out", expected_sha=SHA, run_id="run-9"
    )
    assert rc == 2
    # Missing artifact names the missing gate.
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=empty, out_dir=tmp_path / "empty-out", expected_sha=SHA, run_id="run-9"
    )
    assert rc == 2
    missing = json.loads((tmp_path / "empty-out" / "result.json").read_text(encoding="utf-8"))
    assert missing["result"] == "BLOCKED"
    assert missing["blocking_gate"] == "A"
    assert missing["failed_gates"] == ["A"]


def test_gate_f_forged_artifact_fails_closed(tmp_path: Path) -> None:
    evidence_dir = _evidence_dir(tmp_path)
    path = evidence_dir / "gate-B-evidence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir,
        out_dir=tmp_path / "out",
        expected_sha=SHA,
        run_id="run-9",
    )
    assert rc == 2


def test_gate_f_contradictory_b_fail_preserved_with_subcomponent(tmp_path: Path) -> None:
    evidence_dir = _evidence_dir(
        tmp_path,
        b_status="FAIL",
        b_component="grounding",
        b_category="component-fail",
    )
    out_dir = tmp_path / "out"
    rc = runner.reduce_final_from_evidence_dir(
        evidence_dir=evidence_dir, out_dir=out_dir, expected_sha=SHA, run_id="run-9"
    )
    assert rc == 1
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result["failed_gates"] == ["B"]
    assert result["root_components"] == ["B:component-fail:grounding"]


# -- blocking vs failed gates -------------------------------------------------


def test_blocking_gate_is_human_summary_failed_gates_is_complete() -> None:
    evidences = [
        GateEvidence(
            gate=gate,
            status=status,
            sha=SHA,
            product_fingerprint=PRODUCT,
            runtime_fingerprint=RUNTIME,
            run_id="run-9",
            failure_category="" if status == "PASS" else f"{gate.lower()}-failed",
            component="test-component" if status != "PASS" else "",
            live_trusted=True,
            mocked_only=False,
        )
        for gate, status in (
            ("A", "PASS"),
            ("B", "PASS"),
            ("C", "FAIL"),
            ("D", "PASS"),
            ("E", "FAIL"),
        )
    ]
    verdict = decide_final_verdict(
        evidences,
        current_sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-9",
    )
    assert verdict.blocking_gate == "C"
    assert failed_gates_for_verdict(verdict) == ["C", "E"]
    roots = root_components_for_verdict(verdict)
    assert len(roots) == 2
    assert all(root.split(":")[0] in ("C", "E") for root in roots)
    summary = verdict_summary_v2(verdict, failures=[])
    assert summary["blocking_gate"] == "C"
    assert summary["failed_gates"] == ["C", "E"]
    assert len(summary["root_components"]) == 2


# -- workflow wiring ----------------------------------------------------------


def test_workflow_single_execution_ab_and_readonly_f() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    gate_a = workflow.split("Gate A - static/build", 1)[1].split(
        "Gate B - deterministic component integration", 1
    )[0]
    assert "--emit-gate A" in gate_a
    assert "--shell-a-status" in gate_a
    gate_b = workflow.split("Gate B - deterministic component integration", 1)[1].split(
        "Gate C - LIVE exact production conversation path", 1
    )[0]
    assert "--emit-gate B" in gate_b
    assert "--shell-b-status" in gate_b
    gate_f = workflow.split("Gate F - exact-main final verdict", 1)[1].split(
        "Upload per-gate evidence and failure matrix", 1
    )[0]
    assert "--final-only" in gate_f
    assert "--emit-gate" not in gate_f
    assert "Upload per-gate evidence and failure matrix" in workflow
    assert "failure-matrix.json" in workflow
    assert "Failed gates (typed)" in workflow
    assert "Root components (typed)" in workflow
    # Resume preserves single-execution evidence across 429 runner retire.
    assert "gate-[A-E]-evidence.json" in workflow
