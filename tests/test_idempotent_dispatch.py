"""Deterministic tests for idempotent /run dispatch (issue #119).

Covers the bounded AA runtime control-plane repair: the owner `/run`
command is parsed from the immutable event payload, pre-existing campaign
resolution excludes the current event comment id, re-delivery is idempotent
by source comment identity, the initial handler and scheduled reconciler
share one transition, dispatch failure semantics are deterministic, and
hard bounds (4 starts / 5h each / 20h aggregate / 24h wall clock) hold.
"""

from __future__ import annotations

from pathlib import Path

from aa.control.campaign import (
    CAMPAIGN_LIFETIME_SECONDS,
    CONTROL_ISSUE_NUMBER,
    MAX_STARTS,
    ControlComment,
    decide_initial_dispatch,
    decide_reconcile,
    effective_starts_for_generation,
    effective_starts_used,
    format_dispatch_marker,
    generation_id_for_comment,
    has_marker_for_source,
    is_authorized_control_event,
    markers_for_generation,
    parse_dispatch_marker,
    resolve_active_generation,
    status_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

OWNER = "owner"


def _comment(
    comment_id: int, body: str, created_at: float, *, owner: bool = True
) -> ControlComment:
    return ControlComment(comment_id=comment_id, body=body, created_at=created_at, is_owner=owner)


def _bodies(comments: list[ControlComment]) -> list[str]:
    return [c.body for c in comments]


def test_first_fresh_run_dispatches_start_1_of_4() -> None:
    now = 1_000_000.0
    event_id = 101
    event_at = now - 5.0
    comments = [_comment(event_id, "/run", event_at)]
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=event_id,
        event_created_at=event_at,
        comments=comments,
        comment_bodies=_bodies(comments),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert decision.action == "dispatch_new"
    assert decision.generation is not None
    assert decision.generation.generation_id == generation_id_for_comment(event_id)
    assert decision.generation.accepted_at == event_at
    assert decision.next_seq == 1
    # After commit, the durable marker plus the listed run mean start 1/4.
    marker_body = format_dispatch_marker(decision.generation.generation_id, 1, event_id)
    bodies = ["/run", marker_body]
    assert has_marker_for_source(bodies, event_id)
    gen = decision.generation
    assert effective_starts_for_generation(gen, bodies, []) == 1
    assert effective_starts_for_generation(gen, bodies, [event_at + 10.0]) == 1


def test_initiating_comment_never_self_matches() -> None:
    now = 2_000_000.0
    event_id = 201
    event_at = now - 5.0
    comments = [_comment(event_id, "/run", event_at)]
    # Without exclusion the fresh comment would self-match (the defect).
    buggy = resolve_active_generation(comments, [], now)
    assert buggy is not None
    # With the repair the pre-existing resolution excludes the event id.
    fixed = resolve_active_generation(comments, [], now, exclude_comment_id=event_id)
    assert fixed is None


def test_repeated_run_during_active_campaign_is_noop() -> None:
    now = 3_000_000.0
    first_id = 301
    first_at = now - 3600.0
    second_id = 302
    second_at = now - 60.0
    comments = [
        _comment(first_id, "/run", first_at),
        _comment(second_id, "/run", second_at),
        _comment(901, format_dispatch_marker("c301", 1, first_id), first_at + 5.0),
    ]
    bodies = _bodies(comments)
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=second_id,
        event_created_at=second_at,
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[first_at + 10.0],
        runtime_active=True,
        now=now,
    )
    assert decision.action == "noop_active"
    assert decision.generation is not None
    assert decision.generation.source_comment_id == first_id
    assert decision.starts_used == 1


def test_duplicate_delivery_of_same_event_is_idempotent() -> None:
    now = 4_000_000.0
    event_id = 401
    event_at = now - 120.0
    marker = format_dispatch_marker("c401", 1, event_id)
    comments = [
        _comment(event_id, "/run", event_at),
        _comment(902, marker, event_at + 5.0),
    ]
    bodies = _bodies(comments)
    # Re-delivery carries the same immutable payload body and comment id.
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=event_id,
        event_created_at=event_at,
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[event_at + 10.0],
        runtime_active=True,
        now=now,
    )
    assert decision.action == "noop_duplicate"
    # Even when the run listing lags, the marker alone prevents a second start.
    decision_lag = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=event_id,
        event_created_at=event_at,
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert decision_lag.action == "noop_duplicate"


def test_two_distinct_runs_close_together_single_dispatch() -> None:
    now = 5_000_000.0
    first_id = 501
    second_id = 502
    first_at = now - 10.0
    second_at = now - 5.0
    # First handler sees only its own comment and dispatches.
    first_comments = [_comment(first_id, "/run", first_at)]
    first = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=first_id,
        event_created_at=first_at,
        comments=first_comments,
        comment_bodies=_bodies(first_comments),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert first.action == "dispatch_new"
    # Second handler runs serialized after the first commit and re-reads.
    marker = format_dispatch_marker("c501", 1, first_id)
    reread = [
        _comment(first_id, "/run", first_at),
        _comment(second_id, "/run", second_at),
        _comment(903, marker, first_at + 1.0),
    ]
    second = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=second_id,
        event_created_at=second_at,
        comments=reread,
        comment_bodies=_bodies(reread),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert second.action == "noop_active"
    assert second.generation is not None
    assert second.generation.source_comment_id == first_id


def test_comment_edited_after_delivery_uses_payload() -> None:
    now = 6_000_000.0
    event_id = 601
    event_at = now - 30.0
    # Payload said /run but the stored body was edited afterwards.
    stored_edited = [
        _comment(event_id, "hello edited", event_at),
    ]
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=event_id,
        event_created_at=event_at,
        comments=stored_edited,
        comment_bodies=_bodies(stored_edited),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert decision.action == "dispatch_new"
    # Reverse: payload was not a command, even though the stored body now
    # looks like /run after an edit. No dispatch may follow.
    stored_run = [_comment(event_id, "/run", event_at)]
    invalid = decide_initial_dispatch(
        event_body="hello",
        event_comment_id=event_id,
        event_created_at=event_at,
        comments=stored_run,
        comment_bodies=_bodies(stored_run),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert invalid.action == "noop_invalid"


def test_reconciler_waits_while_initial_dispatch_pending() -> None:
    now = 7_000_000.0
    event_id = 701
    event_at = now - 60.0
    marker = format_dispatch_marker("c701", 1, event_id)
    comments = [
        _comment(event_id, "/run", event_at),
        _comment(904, marker, event_at + 2.0),
    ]
    bodies = _bodies(comments)
    # Run not yet listed: pending marker bridges the gap, no second poller.
    pending = decide_reconcile(
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert pending.action == "noop_pending"
    assert pending.starts_used == 1
    assert pending.next_seq == 2
    # Once the runtime is listed active, the reconciler also no-ops.
    active = decide_reconcile(
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[event_at + 10.0],
        runtime_active=True,
        now=now,
    )
    assert active.action == "noop_runtime_active"
    # After the first runtime ends cleanly, the reconciler may continue.
    continued = decide_reconcile(
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[event_at + 10.0],
        runtime_active=False,
        now=now,
    )
    assert continued.action == "dispatch"
    assert continued.next_seq == 2


def test_dispatch_failure_before_commit_consumes_no_start() -> None:
    now = 8_000_000.0
    event_id = 801
    event_at = now - 30.0
    comments = [_comment(event_id, "/run", event_at)]
    bodies = _bodies(comments)
    # No marker was posted because the API raised: nothing committed.
    assert not has_marker_for_source(bodies, event_id)
    assert effective_starts_used(0, 0) == 0
    # The campaign is still resolvable via the /run comment, so a later
    # serialized reconciler tick may retry exactly once after re-reading.
    retry = decide_reconcile(
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert retry.action == "dispatch"
    assert retry.next_seq == 1


def test_failure_after_commit_still_consumes_start() -> None:
    now = 9_000_000.0
    event_id = 901
    event_at = now - 3600.0
    marker = format_dispatch_marker("c901", 1, event_id)
    comments = [
        _comment(event_id, "/run", event_at),
        _comment(905, marker, event_at + 2.0),
    ]
    bodies = _bodies(comments)
    # The run was committed then failed: it still counts toward the bound.
    failed_run_at = event_at + 10.0
    gen = resolve_active_generation(comments, [failed_run_at], now)
    assert gen is not None
    assert effective_starts_for_generation(gen, bodies, [failed_run_at]) == 1
    followup = decide_reconcile(
        comments=comments,
        comment_bodies=bodies,
        run_created_at=[failed_run_at],
        runtime_active=False,
        now=now,
    )
    assert followup.action == "dispatch"
    assert followup.starts_used == 1
    assert followup.next_seq == 2


def test_stale_expired_campaign_allows_fresh_run() -> None:
    now = 10_000_000.0
    old_id = 1001
    old_at = now - (CAMPAIGN_LIFETIME_SECONDS + 3600.0)
    comments = [
        _comment(old_id, "/run", old_at),
        _comment(906, format_dispatch_marker("c1001", 1, old_id), old_at + 5.0),
    ]
    assert resolve_active_generation(comments, [old_at + 10.0], now) is None
    fresh_id = 1002
    fresh_at = now - 5.0
    with_fresh = [*comments, _comment(fresh_id, "/run", fresh_at)]
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=fresh_id,
        event_created_at=fresh_at,
        comments=with_fresh,
        comment_bodies=_bodies(with_fresh),
        run_created_at=[old_at + 10.0],
        runtime_active=False,
        now=now,
    )
    assert decision.action == "dispatch_new"
    assert decision.generation is not None
    assert decision.generation.source_comment_id == fresh_id


def test_status_while_initial_dispatch_pending_reports_1_of_4() -> None:
    now = 11_000_000.0
    event_id = 1101
    event_at = now - 30.0
    marker = format_dispatch_marker("c1101", 1, event_id)
    comments = [
        _comment(event_id, "/run", event_at),
        _comment(907, marker, event_at + 2.0),
    ]
    snapshot = status_snapshot(
        comments=comments,
        comment_bodies=_bodies(comments),
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert snapshot["active"] is True
    assert snapshot["starts_used"] == 1
    assert snapshot["pending_dispatch"] is True
    assert snapshot["generation"] == "c1101"
    # The defect state (active with 0/4 from self-match) cannot occur: the
    # committed marker already represents start 1/4 before the run appears.


def test_start_counter_atomicity_serialized_reread() -> None:
    now = 12_000_000.0
    first_id = 1201
    second_id = 1202
    first_at = now - 20.0
    second_at = now - 10.0
    stale_comments = [
        _comment(first_id, "/run", first_at),
        _comment(second_id, "/run", second_at),
    ]
    # A stale read without the first commit would dispatch twice; the repair
    # requires re-reading after each commit, so the second sees the marker.
    marker = format_dispatch_marker("c1201", 1, first_id)
    fresh_comments = [
        *_comment_list(stale_comments),
        _comment(908, marker, first_at + 1.0),
    ]
    bodies = _bodies(fresh_comments)
    second = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=second_id,
        event_created_at=second_at,
        comments=fresh_comments,
        comment_bodies=bodies,
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert second.action == "noop_active"
    # Marker and run never double count the same start.
    gen = resolve_active_generation(fresh_comments, [first_at + 5.0], now)
    assert gen is not None
    assert effective_starts_for_generation(gen, bodies, [first_at + 5.0]) == 1


def _comment_list(comments: list[ControlComment]) -> list[ControlComment]:
    return list(comments)


def test_stop_then_clean_new_campaign() -> None:
    now = 13_000_000.0
    first_id = 1301
    first_at = now - 7200.0
    stop_at = now - 3600.0
    comments = [
        _comment(first_id, "/run", first_at),
        _comment(909, format_dispatch_marker("c1301", 1, first_id), first_at + 5.0),
        _comment(1302, "/bot stop", stop_at),
    ]
    assert resolve_active_generation(comments, [first_at + 10.0], now) is None
    fresh_id = 1303
    fresh_at = now - 5.0
    with_fresh = [*comments, _comment(fresh_id, "/run", fresh_at)]
    decision = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=fresh_id,
        event_created_at=fresh_at,
        comments=with_fresh,
        comment_bodies=_bodies(with_fresh),
        run_created_at=[first_at + 10.0],
        runtime_active=False,
        now=now,
    )
    assert decision.action == "dispatch_new"
    assert decision.generation is not None
    assert decision.generation.source_comment_id == fresh_id


def test_hard_bounds_and_no_overlapping_pollers() -> None:
    now = 14_000_000.0
    event_id = 1401
    event_at = now - 3600.0
    comments = [_comment(event_id, "/run", event_at)]
    bodies = _bodies(comments)
    assert CONTROL_ISSUE_NUMBER == 31
    assert MAX_STARTS == 4
    assert effective_starts_used(4, 4) == 4
    # Active poller blocks every path; no queued backlog is ever created.
    blocked_initial = decide_initial_dispatch(
        event_body="/run",
        event_comment_id=9999,
        event_created_at=now - 5.0,
        comments=[*_comment_list(comments), _comment(9999, "/run", now - 5.0)],
        comment_bodies=[*bodies, "/run"],
        run_created_at=[event_at + 10.0],
        runtime_active=True,
        now=now,
    )
    assert blocked_initial.action in {"noop_active", "noop_runtime_active"}
    blocked_reconcile = decide_reconcile(
        comments=comments,
        comment_bodies=[*bodies, format_dispatch_marker("c1401", 1, event_id)],
        run_created_at=[event_at + 10.0],
        runtime_active=True,
        now=now,
    )
    assert blocked_reconcile.action == "noop_runtime_active"
    exhausted_comments = [_comment(event_id, "/run", event_at)]
    exhausted_bodies = [*[format_dispatch_marker("c1401", seq, None) for seq in (1, 2, 3, 4)]]
    exhausted = decide_reconcile(
        comments=exhausted_comments,
        comment_bodies=exhausted_bodies,
        run_created_at=[event_at + float(seq) for seq in (1, 2, 3, 4)],
        runtime_active=False,
        now=now,
    )
    # Fully consumed via committed runs: no active campaign remains, so the
    # reconciler stays a permanent no-op until a fresh owner /run.
    assert exhausted.action in {"noop_no_campaign", "noop_exhausted"}
    marker_exhausted = decide_reconcile(
        comments=exhausted_comments,
        comment_bodies=exhausted_bodies,
        run_created_at=[],
        runtime_active=False,
        now=now,
    )
    assert marker_exhausted.action == "noop_exhausted"
    assert marker_exhausted.starts_used == 4


def test_owner_and_issue_authentication() -> None:
    assert is_authorized_control_event("owner", "owner", 31)
    assert not is_authorized_control_event("intruder", "owner", 31)
    assert not is_authorized_control_event("owner", "owner", 32)
    assert not is_authorized_control_event("", "owner", 31)


def test_dispatch_marker_round_trip_is_privacy_safe() -> None:
    marker = format_dispatch_marker("c1501", 1, 1501)
    parsed = parse_dispatch_marker(f"human line\n{marker}\n")
    assert parsed is not None
    assert parsed["generation"] == "c1501"
    assert parsed["seq"] == 1
    assert parsed["source_comment"] == "1501"
    assert parse_dispatch_marker("no marker here") is None
    assert markers_for_generation(["a", marker], "c1501") == [1]
    assert markers_for_generation(["a", marker], "c9999") == []
    raw = f"{marker} {parsed}"
    for probe in ("TELEGRAM", "SECRET", "prompt", "corpus"):
        assert probe not in raw


def _read_workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_control_workflow_idempotent_dispatch_contract() -> None:
    text = _read_workflow("aa-runtime-control.yml")
    # Immutable payload plus current-id exclusion (the self-match repair).
    assert "payload.comment" in text
    assert "currentCommentId" in text or "current_comment" in text.lower()
    assert "exclude" in text.lower()
    # Durable explicit campaign identity and idempotency key.
    assert "aa-campaign-dispatch" in text
    assert "source-comment" in text
    assert "generation" in text
    # Exactly one dispatch attempt with failure-before/after semantics.
    assert "createWorkflowDispatch" in text
    assert "Failure-before-commit" in text or "failure-before-commit" in text.lower()
    assert "Failure-after-commit" in text or "failure-after-commit" in text.lower()
    assert text.count("createWorkflowDispatch") == 1
    # No second poller to repair uncertainty; bounds preserved.
    assert "never queue a second poller" in text
    assert "already active" in text
    assert "idempotent" in text
    assert "1/${MAX_STARTS}" in text or "Start 1" in text
    assert "cancel-in-progress: false" in text
    assert "aa-runtime-control" in text


def test_reconciler_shares_transition_and_cannot_race() -> None:
    control = _read_workflow("aa-runtime-control.yml")
    reconciler = _read_workflow("aa-runtime-reconciler.yml")
    # Same concurrency group serializes control vs reconciler.
    assert "group: aa-runtime-control" in control
    assert "group: aa-runtime-control" in reconciler
    # Same resolver shape and durable markers in both workflows.
    for snippet in (
        "aa-campaign-dispatch",
        "markersForGeneration",
        "resolveCampaign",
        "Math.max(runsAfter, markers)",
        "never queue a second poller",
    ):
        assert snippet in control, snippet
        assert snippet in reconciler, snippet
    # Pending-commit guard: wait instead of double-dispatching.
    assert "markers > runsAfter" in reconciler
    assert "no-op" in reconciler.lower() or "no-op" in reconciler
    assert "No renew" in reconciler or "no renew" in reconciler.lower()
    assert "*/30 * * * *" in reconciler
