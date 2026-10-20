#!/usr/bin/env python3
"""Run Gate A shell checks independently and publish attributed results.

The canonical verification profile is the same inventory used by the live
Gate A workflow. Every check runs even when an earlier check fails, so the
result artifact names every failing command instead of one coarse
``gate-a-shell`` boolean.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.gate_a_shell import (  # noqa: E402
    PROFILE_CHECKS,
    GateAShellCheck,
    GateAShellResult,
    write_shell_results,
)


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("unable to resolve HEAD for Gate A shell results")
    return proc.stdout.strip()


def _run_check(check: GateAShellCheck) -> GateAShellResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(list(check.command), cwd=str(ROOT), check=False)
    except OSError as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        return GateAShellResult(
            check_id=check.check_id,
            status="FAIL",
            exit_code=None,
            duration_ms=duration_ms,
            detail=f"execution-error:{type(exc).__name__}"[:160],
        )
    duration_ms = int((time.monotonic() - started) * 1000)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    detail = "" if status == "PASS" else f"exit-code-{completed.returncode}"
    return GateAShellResult(
        check_id=check.check_id,
        status=status,
        exit_code=int(completed.returncode),
        duration_ms=duration_ms,
        detail=detail,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gate A shell checks independently.")
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(PROFILE_CHECKS)),
        default="gate-a",
        help="Check inventory to execute.",
    )
    parser.add_argument("--main-sha", default="", help="Exact SHA bound to the result artifact.")
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", "local"),
        help="Run id bound to the result artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "self-proving-out" / "gate-a-shell-results.json",
        help="Result artifact path.",
    )
    args = parser.parse_args(argv)

    main_sha = str(args.main_sha or "").strip() or _git_head()
    checks = PROFILE_CHECKS[str(args.profile)]
    results: list[GateAShellResult] = []
    for check in checks:
        print(f"[gate-a-shell] {check.check_id}", file=sys.stderr, flush=True)
        result = _run_check(check)
        results.append(result)
        print(
            f"[gate-a-shell] {check.check_id} {result.status} "
            f"exit={result.exit_code} duration_ms={result.duration_ms}",
            file=sys.stderr,
            flush=True,
        )
    write_shell_results(args.out, main_sha=main_sha, run_id=str(args.run_id), results=results)
    failed = [result.check_id for result in results if result.status != "PASS"]
    print(
        json.dumps(
            {
                "profile": args.profile,
                "checks": len(results),
                "failed_checks": failed,
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
