"""Answer-quality evaluator runner for the frozen RU rubric (#73).

Two modes (evaluation infrastructure only, never product behavior):

- ``--calibration``: grade the frozen synthetic calibration set and check
  every fixture meets its expected hard verdict and soft band. No secrets,
  no network, no Telegram bodies.
- ``--transcript PATH``: grade one decrypted benchmark transcript (JSON)
  against the frozen rubric and write a machine-readable result plus
  root-cause clusters. The caller decrypts the trusted #62 artifact first;
  this script never touches age identities and never logs answer bodies.

Both modes verify the rubric checksum binding before grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa.qualification.answer_quality import (  # noqa: E402
    AnswerQualityError,
    BenchmarkTuple,
    HardCheckContext,
    grade_batch,
    grade_turn,
    load_calibration,
    load_rubric,
    tuple_key,
    verify_benchmark_artifact,
    verify_rubric_bound,
)


def _fixture_context(
    fixture: dict[str, str | object], calibration: dict[str, object]
) -> HardCheckContext:
    sources = calibration["sources"]
    assert isinstance(sources, dict)
    oracle = fixture["oracle"]
    assert isinstance(oracle, dict)
    claimed = fixture.get("claimed_quotes", [])
    locators = fixture.get("evidence_locators", [])
    spans = fixture.get("forbidden_spans", [])
    history = fixture.get("history", [])
    assert isinstance(claimed, list) and isinstance(locators, list)
    assert isinstance(spans, list) and isinstance(history, list)
    expected_decision = str(oracle.get("expected_safety_decision", "allow"))
    captured = fixture.get("captured_safety_decision", expected_decision)
    return HardCheckContext(
        synthetic_input=str(fixture["synthetic_input"]),
        history=tuple(str(item) for item in history),
        generated_answer=str(fixture["generated_answer"]),
        captured_safety_decision=str(captured),
        claimed_quotes=tuple(str(item) for item in claimed),
        evidence_locators=tuple(str(item) for item in locators),
        evidence_book_claim=bool(fixture.get("evidence_book_claim", False)),
        forbidden_spans=tuple(str(item) for item in spans),
        source_texts={str(key): str(value) for key, value in sources.items()},
        valid_locators=frozenset(str(item) for item in calibration["valid_locators"]),
        oracle=dict(oracle),
    )


def run_calibration(*, generator_model: str, evaluator_id: str) -> dict[str, object]:
    """Grade every frozen calibration fixture; raise on any mismatch."""
    rubric_sha = verify_rubric_bound()
    rubric = load_rubric()
    calibration = load_calibration()
    checked = 0
    for fixture in calibration["fixtures"]:
        assert isinstance(fixture, dict)
        ctx = _fixture_context(fixture, calibration)
        grade = grade_turn(
            ctx,
            rubric,
            case_id=str(fixture["id"]),
            generator_model=generator_model,
            evaluator_id=evaluator_id,
        )
        expected = fixture["expected"]
        assert isinstance(expected, dict)
        if bool(expected["hard_pass"]) != grade.hard_passed:
            raise AnswerQualityError(
                f"{fixture['id']}: expected hard_pass={expected['hard_pass']} "
                f"got {grade.hard_passed} "
                f"({[(r.check_id, r.detail) for r in grade.hard_results if not r.passed]})"
            )
        for check_id in expected.get("failed_checks", []):
            assert isinstance(check_id, str)
            failed = [result.check_id for result in grade.hard_results if not result.passed]
            if check_id not in failed:
                raise AnswerQualityError(
                    f"{fixture['id']}: expected hard failure {check_id!r} missing (failed={failed})"
                )
        soft_min = float(expected.get("soft_min", 0.0))
        if grade.soft_mean < soft_min - 1e-9:
            raise AnswerQualityError(
                f"{fixture['id']}: soft mean {grade.soft_mean:.2f} below {soft_min}"
            )
        if "soft_max" in expected and grade.soft_mean > float(expected["soft_max"]) + 1e-9:
            raise AnswerQualityError(
                f"{fixture['id']}: soft mean {grade.soft_mean:.2f} above "
                f"{float(expected['soft_max'])}"
            )
        checked += 1
    return {"rubric_sha": rubric_sha, "fixtures_checked": checked}


def run_transcript(
    transcript_path: Path,
    *,
    generator_model: str,
    evaluator_id: str,
    out_dir: Path,
    tested_sha: str,
    current_main_sha: str,
    trusted_pass_sha: str,
    corpus_sha: str,
    artifact_sha: str,
    expected_corpus_sha: str = "",
    expected_artifact_sha: str = "",
    completeness_ok: bool = False,
    completeness_reason: str = "completeness not verified",
    expected_generator_model: str = "",
    expected_runtime_version: str = "",
) -> dict[str, object]:
    """Grade a decrypted transcript and write result + clusters (no bodies on stdout)."""
    rubric_sha = verify_rubric_bound()
    rubric = load_rubric()
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    turns = payload.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise AnswerQualityError("transcript holds no turns; refusing to grade")
    sources = payload.get("source_texts", {})
    if not isinstance(sources, dict):
        raise AnswerQualityError("transcript source_texts must be an object")
    # Locator authenticity is judged against the qualified RU source index
    # shipped inside the benchmark transcript itself (the same index the
    # workflow grading job restores from the canonical RU corpus), never the
    # synthetic calibration index: genuine book locators must resolve here.
    raw_valid = payload.get("valid_locators", None)
    if isinstance(raw_valid, list) and raw_valid:
        valid_locators = frozenset(str(item) for item in raw_valid)
    elif isinstance(raw_valid, (set, tuple, frozenset)):
        valid_locators = frozenset(str(item) for item in raw_valid)
    else:
        valid_locators = frozenset(str(key) for key in sources.keys()) if sources else frozenset()
    # Generator/runtime identity recorded by the benchmark artifact itself.
    identity = payload.get("eval_identity", {})
    if not isinstance(identity, dict):
        identity = {}
    actual_generator = str(
        identity.get("primary_model")
        or identity.get("generator_model")
        or payload.get("generator_model")
        or ""
    ).strip()
    actual_runtime = str(
        identity.get("runtime_version") or payload.get("runtime_version") or ""
    ).strip()
    if not actual_generator:
        observed = {
            str(entry.get("primary_model", "")).strip()
            for entry in turns
            if isinstance(entry, dict) and str(entry.get("primary_model", "")).strip()
        }
        if len(observed) > 1:
            raise AnswerQualityError(
                "transcript mixes generator models; refusing to grade unattributed tuple"
            )
        if len(observed) == 1:
            actual_generator = next(iter(observed))
    grades = []
    for entry in turns:
        if not isinstance(entry, dict):
            raise AnswerQualityError("transcript turn must be an object")
        oracle = entry.get("oracle", {})
        if not isinstance(oracle, dict):
            raise AnswerQualityError("transcript turn lacks oracle metadata")
        history = entry.get("history", [])
        claimed = entry.get("claimed_quotes", [])
        locators = entry.get("evidence_locators", [])
        spans = entry.get("forbidden_spans", [])
        if not isinstance(history, list) or not isinstance(claimed, list):
            raise AnswerQualityError("transcript turn lists must be lists")
        if not isinstance(locators, list) or not isinstance(spans, list):
            raise AnswerQualityError("transcript turn lists must be lists")
        if not all(isinstance(item, str) for item in locators + claimed + spans + history):
            raise AnswerQualityError("transcript turn lists must hold strings")
        ctx = HardCheckContext(
            synthetic_input=str(entry.get("synthetic_input", "")),
            history=tuple(str(item) for item in history),
            generated_answer=str(entry.get("generated_answer", "")),
            captured_safety_decision=str(entry.get("safety_decision", "")),
            claimed_quotes=tuple(str(item) for item in claimed),
            evidence_locators=tuple(str(item) for item in locators),
            evidence_book_claim=bool(entry.get("evidence_book_claim", False)),
            forbidden_spans=tuple(str(item) for item in spans),
            source_texts={str(k): str(v) for k, v in sources.items()},
            valid_locators=valid_locators,
            oracle=dict(oracle),
        )
        grades.append(
            grade_turn(
                ctx,
                rubric,
                case_id=str(entry.get("case_id", "?")),
                generator_model=generator_model,
                evaluator_id=evaluator_id,
                is_multi_turn=bool(entry.get("journey_id", "")),
            )
        )
    batch = grade_batch(grades, generator_model=generator_model, evaluator_id=evaluator_id)
    verdict = verify_benchmark_artifact(
        tested_sha=tested_sha,
        current_main_sha=current_main_sha,
        trusted_pass_sha=trusted_pass_sha,
        corpus_sha=corpus_sha,
        expected_corpus_sha=expected_corpus_sha,
        artifact_sha=artifact_sha,
        expected_artifact_sha=expected_artifact_sha,
        completeness_ok=completeness_ok,
        completeness_reason=completeness_reason,
        generator_model=actual_generator,
        expected_generator_model=expected_generator_model,
        runtime_version=actual_runtime,
        expected_runtime_version=expected_runtime_version,
    )
    if verdict.verdict == "rejected":
        raise AnswerQualityError(f"refusing to grade rejected artifact: {verdict.reasons}")
    entry = BenchmarkTuple(
        main_sha=tested_sha,
        corpus_sha=corpus_sha,
        artifact_sha=artifact_sha,
        rubric_sha=rubric_sha,
    )
    result = {
        "schema": "ru-answer-quality-result/1",
        "rubric_version": "ru-answer-quality-rubric-v1",
        "rubric_sha256": rubric_sha,
        "tuple_key": tuple_key(entry),
        "artifact_verdict": verdict.verdict,
        "artifact_reasons": list(verdict.reasons),
        "batch_result": batch.result,
        "graded_turns": batch.graded_turns,
        "failed_turns": batch.failed_turns,
        "hard_fail_turns": batch.hard_fail_turns,
        "batch_soft_mean": batch.batch_soft_mean,
        "provisional": batch.provisional,
        "limitation": batch.limitation,
        "clusters": [cluster.to_dict() for cluster in batch.clusters],
        "turns": [
            {
                "case_id": grade.case_id,
                "turn_pass": grade.turn_pass,
                "hard_passed": grade.hard_passed,
                "failed_checks": [
                    {"check_id": r.check_id, "root_cause": r.root_cause, "detail": r.detail}
                    for r in grade.hard_results
                    if not r.passed
                ],
                "soft_mean": grade.soft_mean,
                "soft_scores": grade.soft_scores,
                "evaluator_id": grade.evaluator_id,
            }
            for grade in grades
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality-result.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (out_dir / "quality-result.sha256").write_text(bundle + "\n", encoding="utf-8")
    print(
        f"graded {batch.graded_turns} turns: result={batch.result} "
        f"failed={batch.failed_turns} hard_fail={batch.hard_fail_turns} "
        f"mean={batch.batch_soft_mean:.2f} artifact={verdict.verdict}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the calibration set or grade one decrypted transcript."""
    parser = argparse.ArgumentParser(description="Run the RU answer-quality evaluator")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--out-dir", default="eval-quality-out")
    parser.add_argument("--generator-model", default="opencode/muse-spark-1.3-contributor-free")
    parser.add_argument("--evaluator-id", default="provisional-heuristic/1")
    parser.add_argument("--main-sha", default="0" * 40)
    parser.add_argument("--current-main-sha", default="")
    parser.add_argument("--trusted-pass-sha", default="")
    parser.add_argument("--corpus-sha", default="0" * 64)
    parser.add_argument("--artifact-sha", default="0" * 64)
    parser.add_argument("--expected-corpus-sha", default="")
    parser.add_argument("--expected-artifact-sha", default="")
    parser.add_argument("--completeness-ok", action="store_true")
    parser.add_argument("--completeness-reason", default="completeness not verified")
    parser.add_argument("--expected-generator-model", default="")
    parser.add_argument("--expected-runtime-version", default="")
    args = parser.parse_args(argv)
    if args.calibration:
        summary = run_calibration(
            generator_model=args.generator_model, evaluator_id=args.evaluator_id
        )
        checked = summary["fixtures_checked"]
        digest = str(summary["rubric_sha"])[:16]
        print(f"calibration ok: {checked} fixtures rubric={digest}")
        return 0
    if args.transcript:
        run_transcript(
            Path(args.transcript),
            generator_model=args.generator_model,
            evaluator_id=args.evaluator_id,
            out_dir=Path(args.out_dir),
            tested_sha=args.main_sha,
            current_main_sha=args.current_main_sha,
            trusted_pass_sha=args.trusted_pass_sha,
            corpus_sha=args.corpus_sha,
            artifact_sha=args.artifact_sha,
            expected_corpus_sha=args.expected_corpus_sha,
            expected_artifact_sha=args.expected_artifact_sha,
            completeness_ok=bool(args.completeness_ok),
            completeness_reason=args.completeness_reason,
            expected_generator_model=args.expected_generator_model,
            expected_runtime_version=args.expected_runtime_version,
        )
        return 0
    print("error: pass --calibration or --transcript PATH", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
