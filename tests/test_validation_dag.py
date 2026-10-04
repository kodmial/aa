"""Executable validation DAG tests (issue #106)."""

from __future__ import annotations

from aa.qualification.product_fingerprint import fingerprint_of_files, is_product_path
from aa.qualification.validation_dag import (
    admit_to_coding_queue,
    find_reusable_repair,
    has_qualification_blocker_cycle,
    is_pass_current,
    is_validation_tracker,
    latest_complete_tuple,
    latest_pass_tuple,
    parse_eval_markers,
    parse_pass_markers,
    parse_quality_markers,
    readiness_62,
    readiness_63,
    repair_fingerprint,
    should_requalify_product,
)


def _product_files(extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    base: dict[str, bytes] = {
        "src/aa/conversation/orchestrator.py": b"runtime",
        "prompts/aa-agent-system.md": b"prompt",
        "src/aa/telegram/transport.py": b"adapter",
    }
    if extra:
        base.update(extra)
    return base


def test_validation_trackers_never_enter_coding_queue() -> None:
    for tracker in (40, 62, 63, 7):
        assert is_validation_tracker(tracker) is True
        assert admit_to_coding_queue(tracker) is False
    assert admit_to_coding_queue(9) is True
    assert admit_to_coding_queue(106) is True


def test_qualification_must_not_machine_block_capability() -> None:
    assert has_qualification_blocker_cycle("<!-- automation-blocked-by: #9 -->", 9) is True
    assert has_qualification_blocker_cycle("<!-- automation-blocked-by: #6 -->", 6) is True
    assert has_qualification_blocker_cycle("no markers here", 9) is False
    assert has_qualification_blocker_cycle("<!-- automation-blocked-by: #12 -->", 9) is False


def test_reusable_tracker_uses_pass_tuple_not_closed_state() -> None:
    bodies = ["<!-- continuum-qualification-result issue=40 sha=" + "a" * 40 + " result=pass -->"]
    markers = parse_pass_markers(bodies)
    current = latest_pass_tuple(markers)
    assert current is not None
    # Legacy exact-SHA marker stays current only at the same SHA.
    assert is_pass_current(current, current_fingerprint="x" * 64, current_sha="a" * 40) is True
    assert is_pass_current(current, current_fingerprint="x" * 64, current_sha="b" * 40) is False


def test_fingerprint_bound_pass_survives_scheduler_only_change() -> None:
    before = fingerprint_of_files(_product_files())
    after = fingerprint_of_files(
        _product_files({".github/workflows/continuum-issue-scheduler.yml": b"scheduler"})
    )
    assert before == after
    assert is_product_path(".github/workflows/continuum-issue-scheduler.yml") is False
    assert is_product_path("docs/plan.md") is False
    bodies = [
        "<!-- continuum-qualification-result issue=40 sha="
        + "a" * 40
        + " product="
        + before
        + " result=pass -->"
    ]
    current = latest_pass_tuple(parse_pass_markers(bodies))
    assert current is not None
    assert is_pass_current(current, current_fingerprint=after) is True


def test_product_change_invalidates_and_reruns() -> None:
    before = fingerprint_of_files(_product_files())
    after = fingerprint_of_files(
        _product_files({"src/aa/conversation/orchestrator.py": b"runtime-changed"})
    )
    assert before != after
    assert should_requalify_product(before, current_fingerprint=after) is True
    assert should_requalify_product(after, current_fingerprint=after) is False
    bodies = [
        "<!-- continuum-qualification-result issue=40 sha="
        + "a" * 40
        + " product="
        + before
        + " result=pass -->"
    ]
    stale = latest_pass_tuple(parse_pass_markers(bodies))
    assert stale is not None
    assert is_pass_current(stale, current_fingerprint=after) is False


def test_62_readiness_uses_pass_tuple_and_harness_readiness() -> None:
    fingerprint = "f" * 64
    bodies = [
        "<!-- continuum-qualification-result issue=40 sha="
        + "a" * 40
        + " product="
        + fingerprint
        + " result=pass -->"
    ]
    current = latest_pass_tuple(parse_pass_markers(bodies))
    ready, _ = readiness_62(
        current, current_fingerprint=fingerprint, corpus_ready=True, harness_ready=True
    )
    assert ready is True
    not_ready, _ = readiness_62(
        current, current_fingerprint=fingerprint, corpus_ready=True, harness_ready=False
    )
    assert not_ready is False
    stale, _ = readiness_62(
        current, current_fingerprint="0" * 64, corpus_ready=True, harness_ready=True
    )
    assert stale is False


def test_63_readiness_uses_complete_tuple_and_rubric() -> None:
    bodies = [
        "<!-- aa-conversation-eval-result issue=62 sha="
        + "a" * 40
        + " corpus="
        + "c" * 64
        + " result=complete run=123 -->"
    ]
    complete = latest_complete_tuple(parse_eval_markers(bodies))
    assert complete is not None
    ready, _ = readiness_63(complete, rubric_ready=True, already_graded=False)
    assert ready is True
    assert readiness_63(None, rubric_ready=True, already_graded=False)[0] is False
    assert readiness_63(complete, rubric_ready=False, already_graded=False)[0] is False
    assert readiness_63(complete, rubric_ready=True, already_graded=True)[0] is False


def test_fail_reuses_one_repair_issue_and_reruns() -> None:
    sha = "a" * 40
    corpus = "c" * 64
    rubric = "r" * 64
    first = repair_fingerprint("grounding", sha=sha, corpus=corpus, rubric=rubric)
    second = repair_fingerprint("grounding", sha=sha, corpus=corpus, rubric=rubric)
    assert first == second
    assert repair_fingerprint("retrieval", sha=sha, corpus=corpus, rubric=rubric) != first
    assert find_reusable_repair([(11, first)], fingerprint=first) == 11
    assert find_reusable_repair([(11, first)], fingerprint="deadbeef") is None


def test_quality_markers_parse_with_currency() -> None:
    bodies = [
        "<!-- aa-answer-quality-result issue=63 sha="
        + "a" * 40
        + " corpus="
        + "c" * 64
        + " artifact="
        + "d" * 64
        + " rubric="
        + "e" * 64
        + " result=fail run=9 currency=current -->"
    ]
    parsed = parse_quality_markers(bodies)
    assert len(parsed) == 1
    assert parsed[0].result == "fail"
    assert parsed[0].currency == "current"
