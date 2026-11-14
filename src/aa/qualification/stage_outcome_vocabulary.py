"""Shared closed vocabulary for privacy-safe stage outcome histograms.

Both Gate C (``self_proving``) and the live Product Contract lanes
(``product_contract_live``) validate the same ``stage_outcome_counts``
telemetry. Keep the vocabulary and validator in one place so the two
privacy guards cannot drift: a label accepted by one gate must be accepted
by the other.

Partial snapshots use bare stage names; completed summaries use the
corresponding ``*_outcome`` fields. Outcome labels are emitted by the
production turn pipeline (including failure-path telemetry) and are never
arbitrary user/model text. The stage name ``answer`` is structural here,
not an answer payload.
"""

from __future__ import annotations

from typing import Any

STAGE_OUTCOME_STAGES: frozenset[str] = frozenset({"planner", "retrieval", "answer", "verifier"})
STAGE_OUTCOME_FIELDS: frozenset[str] = frozenset(
    {f"{stage}_outcome" for stage in STAGE_OUTCOME_STAGES}
)
STAGE_OUTCOME_VALUES: dict[str, frozenset[str]] = {
    "planner": frozenset(
        {
            "unknown",
            "skipped-initial",
            "invoked",
            "ok",
            "empty",
            "invalid",
            "glue",
            "timeout",
            "replanned",
            "empty-replan",
            "failed",
            "skipped-glue",
            "skipped-no-planner",
        }
    ),
    "retrieval": frozenset(
        {
            "unknown",
            "skipped-initial",
            "preloaded",
            "recovered",
            "empty-after-recovery",
            "recovery-failed",
            "skipped-glue",
            "empty-pack",
            "skipped-duplicate-repair",
            "repaired",
            "empty",
            "failed",
            "evidence-ready",
        }
    ),
    "answer": frozenset(
        {
            "unknown",
            "failed",
            "draft-ok",
            "served",
            "narrowed-supported",
            "narrowed-compacted",
            "narrowed-adequacy",
            "narrowed-adequacy-regen",
            "safety-blocked",
            "adequacy-failed",
            "served-split",
            "envelope-exceeded",
            "quote-budget-exceeded",
            "language-guard",
            "retry-turn-budget",
            "unavailable-substantive-no-evidence",
            "conversational-generated",
            "conversational-generation-failed",
            "clarification-unavailable",
            "paging-guard",
            "adequacy-repair-failed",
        }
    ),
    "verifier": frozenset(
        {
            "unknown",
            "skipped",
            "skipped-turn-budget",
            "passed",
            "local-guard",
            "partial-unavailable",
            "unsupported",
            "passed-after-repair-existing-pack",
            "passed-after-repair",
            "unavailable",
            "passed-after-adequacy-repair",
        }
    ),
}


def is_structural_stage_outcome_counts(value: Any) -> bool:
    """Validate the closed-vocabulary, numeric-only stage outcome histogram."""
    if not isinstance(value, dict):
        return False
    if not value:
        return True

    # Incremental partial-evidence shape: stage -> outcome -> count.
    if set(value).issubset(STAGE_OUTCOME_STAGES):
        for stage, outcomes in value.items():
            if not isinstance(outcomes, dict):
                return False
            allowed = STAGE_OUTCOME_VALUES[stage]
            if any(
                outcome not in allowed or type(count) is not int or count < 0
                for outcome, count in outcomes.items()
            ):
                return False
        return True

    # Completed live-summary shape: *_outcome -> outcome -> count.
    # Partial snapshots may legitimately carry only the stages observed so far.
    for outcome_field, outcomes in value.items():
        if outcome_field not in STAGE_OUTCOME_FIELDS:
            return False
        stage = outcome_field.removesuffix("_outcome")
        if not isinstance(outcomes, dict):
            return False
        allowed = STAGE_OUTCOME_VALUES[stage]
        if any(
            outcome not in allowed or type(count) is not int or count < 0
            for outcome, count in outcomes.items()
        ):
            return False
    return True


__all__ = [
    "STAGE_OUTCOME_FIELDS",
    "STAGE_OUTCOME_STAGES",
    "STAGE_OUTCOME_VALUES",
    "is_structural_stage_outcome_counts",
]
