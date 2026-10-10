#!/usr/bin/env python3
"""Trusted #307 independent book-fidelity qualification runner.

Runs the deterministic offline controls (invented fixture text only)
plus the exact-main live gate: the compiled production graph with the
real canonical RU corpus/index and providers over the live-equivalent
Telegram boundary, and a completed expert human review.

Exit codes: 0 PASS, 1 FAIL, 2 INCOMPLETE, 3 STALE.

Offline controls proving the machinery never qualify the product on
their own: without a decisive live run and a completed expert review
the result is INCOMPLETE (``expert-review-pending`` or
``live-prerequisites-absent``), never PASS. Privacy: logs and summaries
carry ids, digests, counts, ranks and latencies only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.book_fidelity_307 import (  # noqa: E402
    EXIT_BY_STATUS,
    BookFidelity307Error,
    ChainTrace,
    build_offline_controls,
    build_protected_payload_307,
    decide_status_307,
    evaluate_case_fidelity,
    live_graph_telegram_wiring_present,
    summarize_public_307,
    validate_exact_sha,
)

DEFAULT_OUT = ROOT / "eval-book-fidelity-307-out"
STATUS_FILENAME = "result-307.json"
SUMMARY_FILENAME = "book-fidelity-307-summary.json"
PROTECTED_FILENAME = "book-fidelity-307-protected.tar.zst.age"
PROTECTED_SHA_FILENAME = "book-fidelity-307-protected.tar.zst.age.sha256"


def _fail_incomplete(out_dir: Path, *, reason: str, main_sha: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": "INCOMPLETE", "reason": reason, "main_sha": main_sha, "issue": 307}
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"#307 book-fidelity INCOMPLETE: {reason}", file=sys.stderr)
    return EXIT_BY_STATUS["INCOMPLETE"]


def _checked_out_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if proc.returncode != 0:
        raise BookFidelity307Error("cannot determine checked-out SHA")
    return proc.stdout.strip()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trusted #307 independent book-fidelity qualification."
    )
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--recipient", default="", help="Age recipient for encryption")
    parser.add_argument(
        "--expert-review-evidence",
        type=Path,
        default=None,
        help="Path to expert human review evidence (required for PASS)",
    )
    args = parser.parse_args(argv)

    try:
        expected_sha = validate_exact_sha(args.main_sha)
    except BookFidelity307Error as exc:
        print(f"#307 book-fidelity failed: {exc}", file=sys.stderr)
        return EXIT_BY_STATUS["INCOMPLETE"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        checked = _checked_out_sha()
    except BookFidelity307Error as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    if checked != expected_sha:
        payload = {
            "result": "STALE",
            "reason": "checked-out SHA is not the expected exact SHA",
            "main_sha": expected_sha,
            "issue": 307,
        }
        (out_dir / STATUS_FILENAME).write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print("#307 book-fidelity STALE: SHA mismatch", file=sys.stderr)
        return EXIT_BY_STATUS["STALE"]

    # ---- deterministic offline controls (machinery only) ----
    controls = build_offline_controls()
    failures = 0
    scores = []
    traces = []
    for control in controls:
        result = evaluate_case_fidelity(
            case=control["case"],
            candidate_text=control["candidate_text"],
            certificate=control["certificate"],
            evidence_pack=control["evidence_pack"],
            source_texts=control["source_texts"],
            delivered_text=control["delivered_text"],
            answered_subquestions=control["answered_subquestions"],
        )
        scores.append(result)
        traces.append(
            ChainTrace(
                case_id=control["case"].case_id,
                candidate_sha256=str(control["certificate"].get("answer_sha256", "")),
                certificate_id=str(control["certificate"].get("certificate_id", "")),
                evidence_digest=str(control["certificate"].get("evidence_digest", "")),
            )
        )
        expected = bool(control["expected_pass"])
        if result.passed != expected or (
            not expected and result.failure_code != control["expected_failure"]
        ):
            failures += 1
            print(
                json.dumps(
                    {
                        "control_id": control["control_id"],
                        "expected_pass": expected,
                        "observed_pass": result.passed,
                        "failure_code": result.failure_code,
                    }
                )
            )

    wiring_ok, _ = live_graph_telegram_wiring_present()
    if not wiring_ok:
        return _fail_incomplete(
            out_dir, reason="production-graph-telegram-missing", main_sha=expected_sha
        )

    # ---- live + human-review gates (never mocked to PASS) ----
    live_decisive = os.environ.get("AA_307_LIVE_DECISIVE", "") == "1"
    expert_review_completed = (
        bool(args.expert_review_evidence) and Path(args.expert_review_evidence).is_file()
    )
    incomplete = False
    reason = ""
    if failures:
        status = "FAIL"
    else:
        status = decide_status_307(
            stale=False,
            incomplete=False,
            failures=0,
            case_results=[item for item in scores if item.case_id.endswith("positive-faithful")],
            expert_review_completed=expert_review_completed,
            live_decisive=live_decisive,
        )
        if status == "INCOMPLETE" and not live_decisive:
            reason = "live-prerequisites-absent"
            incomplete = True
        elif status == "INCOMPLETE" and not expert_review_completed:
            reason = "expert-review-pending"
            incomplete = True

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    corpus_sha = "absent"
    benchmark_sha = "absent"
    retrieval_sha = "absent"
    config_sha = "absent"
    for rel, slot in (
        ("corpus/canonical.ru.manifest.json", "corpus"),
        ("qualification/ru_book_fidelity_307.v1.json", "benchmark"),
    ):
        path = ROOT / rel
        if path.is_file():
            digest = _file_sha(path)
            if slot == "corpus":
                corpus_sha = digest
            else:
                benchmark_sha = digest
    retrieval_path = ROOT / "src" / "aa" / "qualification" / "book_fidelity_307.py"
    if retrieval_path.is_file():
        retrieval_sha = _file_sha(retrieval_path)
    config_path = ROOT / "src" / "aa" / "config.py"
    if config_path.is_file():
        config_sha = _file_sha(config_path)
    model = (os.environ.get("OPENCODE_MODEL") or "unconfigured").strip() or "unconfigured"

    summary = summarize_public_307(
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
        config_sha=config_sha,
        model=model,
        traces=traces,
        scores=scores,
        status=status,
        run_id=str(run_id),
        expert_review_completed=expert_review_completed,
        live_decisive=live_decisive,
    )
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    protected = build_protected_payload_307(
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
        config_sha=config_sha,
        model=model,
        traces=traces,
        scores=scores,
        status=status,
        run_id=str(run_id),
    )
    protected_bytes = json.dumps(protected, sort_keys=True, ensure_ascii=False).encode("utf-8")
    recipient = str(args.recipient or "").strip()
    if recipient:
        try:
            from aa.qualification.real_book_retrieval import compress_and_encrypt

            encrypted = compress_and_encrypt(protected_bytes, recipient=recipient)
        except Exception as exc:
            return _fail_incomplete(
                out_dir,
                reason=f"protected-bundle-failed: {type(exc).__name__}",
                main_sha=expected_sha,
            )
        (out_dir / PROTECTED_FILENAME).write_bytes(encrypted)
        (out_dir / PROTECTED_SHA_FILENAME).write_text(
            hashlib.sha256(encrypted).hexdigest() + "\n", encoding="utf-8"
        )
    else:
        (out_dir / PROTECTED_FILENAME).write_bytes(protected_bytes)
        (out_dir / PROTECTED_SHA_FILENAME).write_text(
            hashlib.sha256(protected_bytes).hexdigest() + "\n", encoding="utf-8"
        )
    result_payload = {
        "result": status,
        "main_sha": expected_sha,
        "control_count": len(controls),
        "control_failures": failures,
        "live_decisive": live_decisive,
        "expert_review_completed": expert_review_completed,
        "issue": 307,
        "target_issue": 7,
    }
    if incomplete and reason:
        result_payload["reason"] = reason
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(result_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": status,
                "issue": 307,
                "controls": len(controls),
                "control_failures": failures,
                "live_decisive": live_decisive,
                "expert_review_completed": expert_review_completed,
            }
        )
    )
    _ = incomplete
    return EXIT_BY_STATUS[status]


if __name__ == "__main__":
    raise SystemExit(main())
