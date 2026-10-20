"""Gate A shell inventory, execution results, and fail-closed validation.

Gate A's static/build proof used to collapse seven independent commands into a
single shell boolean. Every failure then became the same
``A:shell-step-failed:gate-a-shell`` fingerprint even when the real causes were
unrelated (entrypoint, unit suite, lint, formatting, typing, contract fixtures,
runtime fixtures). This module keeps one executable inventory, runs every check
independently, and publishes a privacy-safe result artifact so evidence
attribution names the exact failed check.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.qualification.gate_evidence import GATE_A_COVERAGE
from aa.qualification.self_proving import (
    SelfProvingError,
    assert_no_text_leak,
    validate_exact_sha,
)

SCHEMA_VERSION = "aa-gate-a-shell-results/1"
SHELL_OWNER = "gate-a-shell"
RESULT_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "BLOCKED")


@dataclass(frozen=True)
class GateAShellCheck:
    """One executable Gate A shell command and its evidence identity."""

    check_id: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class GateAShellResult:
    """Privacy-safe outcome of one Gate A shell command."""

    check_id: str
    status: str
    exit_code: int | None
    duration_ms: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


def _python_command(*args: str) -> tuple[str, ...]:
    return (sys.executable or "python3", *args)


CANONICAL_SHELL_CHECKS: tuple[GateAShellCheck, ...] = (
    GateAShellCheck("a-shell-aa-check", _python_command("-m", "aa", "--check")),
    GateAShellCheck("a-shell-pytest", _python_command("-m", "pytest", "-q")),
    GateAShellCheck("a-shell-ruff-check", _python_command("-m", "ruff", "check", ".")),
    GateAShellCheck("a-shell-ruff-format", _python_command("-m", "ruff", "format", "--check", ".")),
    GateAShellCheck("a-shell-mypy", _python_command("-m", "mypy", "src", "tests")),
)

QUALIFICATION_SHELL_CHECKS: tuple[GateAShellCheck, ...] = (
    GateAShellCheck(
        "a-shell-contract-qualification",
        _python_command("scripts/verify_product_contract_qualification.py", "--require-active"),
    ),
    GateAShellCheck(
        "a-shell-runtime-qualification",
        _python_command("scripts/verify_runtime_qualification.py"),
    ),
)

GATE_A_SHELL_CHECKS: tuple[GateAShellCheck, ...] = (
    CANONICAL_SHELL_CHECKS + QUALIFICATION_SHELL_CHECKS
)
SHELL_CHECK_IDS: tuple[str, ...] = tuple(check.check_id for check in GATE_A_SHELL_CHECKS)

PROFILE_CHECKS: dict[str, tuple[GateAShellCheck, ...]] = {
    "canonical": CANONICAL_SHELL_CHECKS,
    "gate-a": GATE_A_SHELL_CHECKS,
}

_COVERAGE_SHELL_IDS = tuple(
    row["check_id"] for row in GATE_A_COVERAGE if row["owner"] == SHELL_OWNER
)
if SHELL_CHECK_IDS != _COVERAGE_SHELL_IDS:
    raise RuntimeError("Gate A shell inventory drifted from the coverage table")


def _normalize_status(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in RESULT_STATUSES:
        raise SelfProvingError(f"unknown Gate A shell status {normalized!r}")
    return normalized


def _normalized_run_id(value: Any) -> str:
    run_id = str(value or "").strip()
    if not run_id or any(char.isspace() for char in run_id):
        raise SelfProvingError("Gate A shell run id must be a non-empty token")
    return run_id


def _validate_result(payload: Any, expected_ids: Iterable[str]) -> GateAShellResult:
    if not isinstance(payload, dict):
        raise SelfProvingError("Gate A shell check result must be an object")
    check_id = str(payload.get("check_id", ""))
    if check_id not in set(expected_ids):
        raise SelfProvingError(f"unknown Gate A shell check id {check_id!r}")
    status = _normalize_status(payload.get("status"))
    exit_code_raw = payload.get("exit_code")
    if exit_code_raw is not None and (
        not isinstance(exit_code_raw, int) or isinstance(exit_code_raw, bool)
    ):
        raise SelfProvingError("Gate A shell exit code must be an integer or null")
    exit_code = int(exit_code_raw) if exit_code_raw is not None else None
    duration_raw = payload.get("duration_ms", 0)
    if not isinstance(duration_raw, int) or isinstance(duration_raw, bool) or duration_raw < 0:
        raise SelfProvingError("Gate A shell duration must be a non-negative integer")
    detail = str(payload.get("detail", "") or "")
    if len(detail) > 160:
        raise SelfProvingError("Gate A shell detail too long")
    if status == "PASS" and exit_code != 0:
        raise SelfProvingError("Gate A shell PASS result requires exit code 0")
    if status == "FAIL" and exit_code == 0:
        raise SelfProvingError("Gate A shell FAIL result cannot carry exit code 0")
    if status == "BLOCKED" and exit_code is not None and exit_code == 0:
        raise SelfProvingError("Gate A shell BLOCKED result cannot carry exit code 0")
    return GateAShellResult(
        check_id=check_id,
        status=status,
        exit_code=exit_code,
        duration_ms=int(duration_raw),
        detail=detail,
    )


def build_shell_results_body(
    *,
    main_sha: str,
    run_id: str,
    results: Iterable[GateAShellResult],
) -> dict[str, Any]:
    """Build the SHA/run-bound privacy-safe result payload."""
    ordered = list(results)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "main_sha": validate_exact_sha(main_sha),
        "run_id": _normalized_run_id(run_id),
        "checks": [result.to_dict() for result in ordered],
    }
    assert_no_text_leak(payload, owner="gate-a-shell-results")
    return payload


def write_shell_results(
    path: Path,
    *,
    main_sha: str,
    run_id: str,
    results: Iterable[GateAShellResult],
) -> dict[str, Any]:
    """Atomically write one Gate A shell result artifact."""
    payload = build_shell_results_body(main_sha=main_sha, run_id=run_id, results=results)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass
    return payload


def load_shell_results(
    path: Path,
    *,
    expected_sha: str,
    expected_run_id: str,
) -> dict[str, GateAShellResult]:
    """Load and validate exactly one complete Gate A shell result set."""
    if not path.is_file():
        raise SelfProvingError("missing Gate A shell results")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SelfProvingError("Gate A shell results unreadable") from exc
    if not isinstance(payload, dict):
        raise SelfProvingError("Gate A shell results invalid shape")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SelfProvingError("Gate A shell results schema mismatch")
    if str(payload.get("main_sha", "")).lower() != validate_exact_sha(expected_sha):
        raise SelfProvingError("Gate A shell results stale SHA")
    if str(payload.get("run_id", "")) != _normalized_run_id(expected_run_id):
        raise SelfProvingError("Gate A shell results stale run")
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list):
        raise SelfProvingError("Gate A shell results missing checks")
    by_id: dict[str, GateAShellResult] = {}
    for raw in raw_checks:
        result = _validate_result(raw, SHELL_CHECK_IDS)
        if result.check_id in by_id:
            raise SelfProvingError(f"duplicate Gate A shell check {result.check_id!r}")
        by_id[result.check_id] = result
    missing = [check_id for check_id in SHELL_CHECK_IDS if check_id not in by_id]
    if missing:
        raise SelfProvingError("Gate A shell results omitted checks: " + ",".join(missing[:4]))
    assert_no_text_leak(payload, owner="gate-a-shell-results")
    return by_id


__all__ = [
    "CANONICAL_SHELL_CHECKS",
    "GATE_A_SHELL_CHECKS",
    "PROFILE_CHECKS",
    "QUALIFICATION_SHELL_CHECKS",
    "RESULT_STATUSES",
    "SCHEMA_VERSION",
    "SHELL_CHECK_IDS",
    "SHELL_OWNER",
    "GateAShellCheck",
    "GateAShellResult",
    "build_shell_results_body",
    "load_shell_results",
    "write_shell_results",
]
