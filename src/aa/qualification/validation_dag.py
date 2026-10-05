"""Executable autonomous validation DAG (issues #106, #123).

Validation/result state is the source of truth, never issue open/closed
state:

- capability -> qualification is one-way metadata; a qualification tracker
  must not declare its capability as a normal blocker;
- validation trackers (7/62/63, plus historical 40) are never normal
  implementation work and must not enter the generic coding queue;
- a qualification tracker may stay open/reusable across many SHAs;
- current validity is a trusted machine-readable result tuple:
  qualification id + exact product revision/fingerprint + pass/fail + run;
- FAIL/INCOMPLETE creates or reuses one repair issue per root cause,
  waits for its merge, invalidates stale evidence and reruns automatically;
- #62 requires the current trusted #7 PASS plus the frozen vNext benchmark
  checksum plus the migrated harness version; #63 requires the trusted #62
  COMPLETE tuple plus the rubric/evaluator vNext checksum.

This module keeps the DAG decision logic pure (no network) so workflows
and unit tests share one implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Active validation trackers plus the historical product qualification
# tracker, which remains excluded from coding work as immutable evidence
# but is never authoritative for new #62/#63 cycles.
VALIDATION_TRACKERS: frozenset[int] = frozenset({7, 62, 63, 40})

# Active qualification authority for new benchmark cycles.
ACTIVE_VALIDATION_TRACKERS: frozenset[int] = frozenset({7, 62, 63})

# Historical tracker kept out of the coding queue as evidence only.
HISTORICAL_VALIDATION_TRACKERS: frozenset[int] = frozenset({40})

BENCHMARK_VERSION = "ru_product_contract.v1_2"
HARNESS_VERSION = "aa-conversation-eval-harness/2"
RUBRIC_VERSION = "ru-answer-quality-rubric-v2"
EVALUATOR_VERSION = "aa-answer-quality-evaluator/2"

# Capability qualified by each validation tracker (None = derived stage).
CAPABILITY_FOR_QUALIFICATION: dict[int, int | None] = {
    7: 6,
    62: None,
    63: None,
    40: 9,
}

PASS_MARKER = "continuum-qualification-result"
EVAL_MARKER = "aa-conversation-eval-result"
QUALITY_MARKER = "aa-answer-quality-result"

_BLOCKED_BY_RE = re.compile(r"<!--\s*automation-blocked-by:(.*?)-->", re.DOTALL)
_ISSUE_REF_RE = re.compile(r"#(\d+)")


@dataclass(frozen=True)
class QualificationTuple:
    """Trusted machine-readable validation result tuple."""

    issue: int
    sha: str
    product_fingerprint: str
    result: str
    run_id: str


@dataclass(frozen=True)
class EvalTuple:
    """Trusted #62 benchmark completion tuple."""

    sha: str
    corpus: str
    result: str
    run_id: str


@dataclass(frozen=True)
class QualityTuple:
    """Trusted #63 grading tuple."""

    sha: str
    corpus: str
    artifact: str
    rubric: str
    result: str
    run_id: str
    currency: str


def is_validation_tracker(issue_number: int) -> bool:
    """Return whether an issue is a validation tracker, never coding work."""
    return issue_number in VALIDATION_TRACKERS


def admit_to_coding_queue(issue_number: int) -> bool:
    """Return whether an issue may enter the generic coding queue."""
    return not is_validation_tracker(issue_number)


def has_qualification_blocker_cycle(qualification_body: str, capability: int) -> bool:
    """Return whether a qualification body blocks on its own capability.

    A capability -> qualification edge is one-way metadata
    (``automation-qualification`` on the capability). The reverse machine
    edge (``automation-blocked-by: #<capability>`` inside the qualification
    tracker) deadlocks the ``automation:qualifying`` lifecycle and must not
    exist.
    """
    match = _BLOCKED_BY_RE.search(qualification_body)
    if match is None:
        return False
    refs = {int(item) for item in _ISSUE_REF_RE.findall(match.group(1))}
    return capability in refs


def qualification_must_not_block(issue_number: int) -> int | None:
    """Return the capability a tracker must not machine-block, if any."""
    return CAPABILITY_FOR_QUALIFICATION.get(issue_number)


def parse_pass_markers(bodies: list[str], *, issue: int = 7) -> list[QualificationTuple]:
    """Parse trusted #7 PASS/FAIL markers from comment bodies in order."""
    pattern = re.compile(
        r"<!--\s*continuum-qualification-result\s+"
        r"issue=(?P<issue>\d+)\s+"
        r"sha=(?P<sha>[0-9a-f]{40})\s+"
        r"(?:product=(?P<product>[0-9a-f]{64})\s+)?"
        r"result=(?P<result>pass|fail)\s+"
        r"(?:run=(?P<run>\S+?)\s*)?-->"
    )
    out: list[QualificationTuple] = []
    for body in bodies:
        for match in pattern.finditer(body or ""):
            if int(match.group("issue")) != issue:
                continue
            out.append(
                QualificationTuple(
                    issue=issue,
                    sha=match.group("sha"),
                    product_fingerprint=match.group("product") or "",
                    result=match.group("result"),
                    run_id=match.group("run") or "",
                )
            )
    return out


def latest_pass_tuple(markers: list[QualificationTuple]) -> QualificationTuple | None:
    """Return the latest PASS tuple, where a later FAIL clears earlier PASS."""
    current: QualificationTuple | None = None
    for marker in markers:
        if marker.result == "pass":
            current = marker
        else:
            current = None
    return current


def is_pass_current(
    pass_tuple: QualificationTuple | None,
    *,
    current_fingerprint: str,
    current_sha: str = "",
) -> bool:
    """Return whether a trusted #7 PASS tuple authorizes downstream work.

    A fingerprint-bound PASS stays current across unrelated repository
    commits (scheduler-only changes do not rotate the product inputs). A
    legacy marker without a fingerprint falls back to exact-SHA equality,
    which callers must requalify on every new main SHA. Only issue #7
    markers are authoritative; legacy trackers never authorize new cycles.
    """
    if pass_tuple is None or pass_tuple.result != "pass":
        return False
    if pass_tuple.issue != 7:
        return False
    if pass_tuple.product_fingerprint:
        if pass_tuple.product_fingerprint != current_fingerprint:
            return False
        # Fingerprint equality alone is not enough: the PASS SHA must also
        # be the exact current main SHA for an authoritative #62 run.
        if current_sha and pass_tuple.sha != current_sha:
            return False
        return True
    if not current_sha:
        return False
    return pass_tuple.sha == current_sha


def readiness_62(
    pass_tuple: QualificationTuple | None,
    *,
    current_fingerprint: str,
    corpus_ready: bool,
    harness_ready: bool,
    current_sha: str = "",
    benchmark_version: str = BENCHMARK_VERSION,
    benchmark_sha: str = "",
    expected_benchmark_sha: str = "",
    harness_version: str = HARNESS_VERSION,
    expected_harness_version: str = HARNESS_VERSION,
) -> tuple[bool, str]:
    """Decide #62 readiness from trusted tuple/checksum state.

    Requires the current trusted #7 PASS on the exact same current main
    SHA plus the frozen benchmark vNext checksum plus the migrated harness
    version. Readiness is trusted tuple/checksum state, never issue closure
    alone.
    """
    if not corpus_ready:
        return False, "no-op: vNext benchmark is not frozen/ready"
    if not harness_ready:
        return False, "no-op: migrated #72 harness is not ready"
    if benchmark_version != BENCHMARK_VERSION:
        return False, f"no-op: unexpected benchmark version {benchmark_version!r}"
    if harness_version != expected_harness_version:
        return False, "no-op: harness version mismatch"
    if expected_benchmark_sha and benchmark_sha != expected_benchmark_sha:
        return False, "no-op: benchmark vNext checksum mismatch"
    if not is_pass_current(
        pass_tuple, current_fingerprint=current_fingerprint, current_sha=current_sha
    ):
        return False, "no-op: no current trusted #7 PASS tuple on the exact main SHA"
    assert pass_tuple is not None
    return True, f"ready: trusted #7 PASS sha={pass_tuple.sha}"


def parse_eval_markers(bodies: list[str]) -> list[EvalTuple]:
    """Parse trusted #62 benchmark markers from comment bodies in order."""
    pattern = re.compile(
        r"aa-conversation-eval-result\s+"
        r"issue=62\s+"
        r"sha=(?P<sha>[0-9a-f]{40})\s+"
        r"corpus=(?P<corpus>[0-9a-f]{64})\s+"
        r"result=(?P<result>complete|incomplete|stale)\s+"
        r"run=(?P<run>\S+)"
    )
    out: list[EvalTuple] = []
    for body in bodies:
        for match in pattern.finditer(body or ""):
            out.append(
                EvalTuple(
                    sha=match.group("sha"),
                    corpus=match.group("corpus"),
                    result=match.group("result"),
                    run_id=match.group("run"),
                )
            )
    return out


def latest_complete_tuple(markers: list[EvalTuple]) -> EvalTuple | None:
    """Return the latest COMPLETE tuple, where later INCOMPLETE/STALE clears it."""
    current: EvalTuple | None = None
    for marker in markers:
        if marker.result == "complete":
            current = marker
        else:
            current = None
    return current


def parse_quality_markers(bodies: list[str]) -> list[QualityTuple]:
    """Parse trusted #63 grading markers from comment bodies in order."""
    pattern = re.compile(
        r"aa-answer-quality-result\s+"
        r"issue=63\s+"
        r"sha=(?P<sha>[0-9a-f]{40})\s+"
        r"corpus=(?P<corpus>[0-9a-f]{64})\s+"
        r"artifact=(?P<artifact>[0-9a-f]{64})\s+"
        r"rubric=(?P<rubric>[0-9a-f]{64})\s+"
        r"result=(?P<result>pass|fail)\s+"
        r"run=(?P<run>\S+?)\s+"
        r"currency=(?P<currency>current|stale)"
    )
    out: list[QualityTuple] = []
    for body in bodies:
        for match in pattern.finditer(body or ""):
            out.append(
                QualityTuple(
                    sha=match.group("sha"),
                    corpus=match.group("corpus"),
                    artifact=match.group("artifact"),
                    rubric=match.group("rubric"),
                    result=match.group("result"),
                    run_id=match.group("run"),
                    currency=match.group("currency"),
                )
            )
    return out


def readiness_63(
    complete: EvalTuple | None,
    *,
    rubric_ready: bool,
    already_graded: bool,
    rubric_sha: str = "",
    expected_rubric_sha: str = "",
    evaluator_version: str = EVALUATOR_VERSION,
    expected_evaluator_version: str = EVALUATOR_VERSION,
) -> tuple[bool, str]:
    """Decide #63 readiness from the trusted #62 COMPLETE tuple + rubric.

    Requires the trusted #62 COMPLETE tuple plus the rubric/evaluator vNext
    checksum. Readiness is trusted tuple/checksum state, never issue closure
    alone.
    """
    if complete is None:
        return False, "no-op: no trusted #62 COMPLETE tuple to grade"
    if not rubric_ready:
        return False, "no-op: frozen rubric/evaluator vNext is not bound"
    if evaluator_version != expected_evaluator_version:
        return False, "no-op: evaluator version mismatch"
    if expected_rubric_sha and rubric_sha != expected_rubric_sha:
        return False, "no-op: rubric vNext checksum mismatch"
    if already_graded:
        return False, "no-op: identical tuple already has a #63 result"
    return True, f"ready: #62 COMPLETE sha={complete.sha}"


def should_schedule_fresh_cycle(
    *,
    latest_pass_sha: str,
    graded_sha: str,
) -> bool:
    """Whether a newer trusted #7 PASS invalidates stale authority.

    The same exact SHA + fixture + rubric tuple is idempotent (no rerun),
    while a newer #7 PASS schedules exactly one fresh evaluation cycle.
    """
    if not latest_pass_sha.strip() or not graded_sha.strip():
        return True
    return latest_pass_sha.strip() != graded_sha.strip()


def should_requalify_product(last_pass_fingerprint: str, *, current_fingerprint: str) -> bool:
    """Return whether a product change invalidates the stored PASS."""
    if not last_pass_fingerprint.strip() or not current_fingerprint.strip():
        return True
    return last_pass_fingerprint.strip() != current_fingerprint.strip()


def repair_fingerprint(category: str, *, sha: str, corpus: str, rubric: str) -> str:
    """Return the stable idempotency fingerprint for one repair root cause."""
    payload = "\x00".join((category.strip(), sha.strip(), corpus.strip(), rubric.strip()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def find_reusable_repair(open_repairs: list[tuple[int, str]], *, fingerprint: str) -> int | None:
    """Return the open repair issue already carrying a fingerprint, if any."""
    for number, existing in open_repairs:
        if existing == fingerprint:
            return number
    return None


__all__ = [
    "ACTIVE_VALIDATION_TRACKERS",
    "BENCHMARK_VERSION",
    "CAPABILITY_FOR_QUALIFICATION",
    "EVAL_MARKER",
    "EVALUATOR_VERSION",
    "HARNESS_VERSION",
    "HISTORICAL_VALIDATION_TRACKERS",
    "PASS_MARKER",
    "QUALITY_MARKER",
    "RUBRIC_VERSION",
    "VALIDATION_TRACKERS",
    "EvalTuple",
    "QualificationTuple",
    "QualityTuple",
    "admit_to_coding_queue",
    "find_reusable_repair",
    "has_qualification_blocker_cycle",
    "is_pass_current",
    "is_validation_tracker",
    "latest_complete_tuple",
    "latest_pass_tuple",
    "parse_eval_markers",
    "parse_pass_markers",
    "parse_quality_markers",
    "qualification_must_not_block",
    "readiness_62",
    "readiness_63",
    "repair_fingerprint",
    "should_requalify_product",
    "should_schedule_fresh_cycle",
]
