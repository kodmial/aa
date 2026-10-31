"""Incremental privacy-safe partial diagnostics for the live Gate C lane
(kodmial/aa#453).

The watchdog incident (run 39923515643) proved the sequencing defect: the
child publishes its full per-attempt/stage/model/outcome metrics only at
the very end of the 18-scenario campaign (``product_contract_live.py``
metric assembly), while the outer watchdog preempts the child on
``total-deadline`` or ``no-real-progress`` long before that. The timeout
lane therefore lost every measured fact a parent could already have had.

This module owns the *durable partial evidence* contract:

- incrementally, on every meaningful transition (attempt start/change/end,
  stage transition, scenario completion), the child assembles a strictly
  privacy-safe, size-bounded snapshot of already-measured qualification
  metrics and merges it into the same watchdog progress file via
  :func:`aa.qualification.live_watchdog.record_live_progress` (atomic
  replace, no new external calls, no duplicated observation);
- the parent's timeout lane reads the persisted ``partial_evidence``
  marker, validates its shape fail-closed and projects the metrics into
  the typed FAIL evidence with explicit per-source coverage
  (``observed`` / ``missing`` / ``unobserved`` / unsupported server
  states). Nothing is resurrected as a fabricated zero or success: the
  verdict vocabularies and timeout semantics are unchanged.

Every fact carries its source classification so a consumer can never
mistake one vantage point for another:

- ``client-ledger``: the OpenCode HTTP client's own attempt/served-model/
  token/request-latency audit (local HTTP transport evidence);
- ``server-event``: bounded passive server-side session facts probed on
  the pinned ``opencode serve`` (never token/provider progress);
- ``provider-response-confirmed``: a provider fact that arrived in an
  actually received HTTP response/event payload (for example the
  documented numeric ``statusCode`` of a server ``session.error``).

Bounded size and privacy: fixed-vocabulary labels, numeric counters,
attempt allow-list projections and latency aggregates only. Never a
prompt, user message, answer, book passage, raw LLM reply, token count
per requirement, session id, credential, or arbitrary server line.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from aa.opencode.observer import attempt_view
from aa.opencode.server_events import unavailable_evidence

SCHEMA_VERSION = "aa-live-partial-evidence/1"

# Explicit typed marker carried by the timeout evidence when NOTHING was
# recorded for a source (or for the whole child) — the parent publishes
# coverage instead of substituting zeros or success.
DIAGNOSTICS_UNAVAILABLE = "diagnostics-unavailable"

# Closed coverage vocabulary per evidence source.
COVERAGE_OBSERVED = "observed"
COVERAGE_MISSING = "missing"
COVERAGE_UNOBSERVED = "unobserved"
_COVERAGE_VOCAB = (
    COVERAGE_OBSERVED,
    COVERAGE_MISSING,
    COVERAGE_UNOBSERVED,
    "unsupported",
    "disconnected",
    "unavailable",
)

# Evidence sources and the bounded meaning of each coverage value.
SOURCES: tuple[str, ...] = (
    "client_attempt_ledger",
    "served_model_audit",
    "token_usage_audit",
    "request_latency_audit",
    "runtime_process_state",
    "server_events",
    "stage_telemetry",
    "scenario_accounting",
)

_MAX_ATTEMPT_ENTRIES = 48
_MAX_LATENCY_SAMPLES_REF = 256
_MAX_BYTES_SOFT = 1 << 20

# Event identity of the two distinct timeout records for one scenario:
# the child's own scenario abort vs the parent's campaign termination.
EVENT_KIND_CHILD_TIMEOUT = "child-scenario-timeout"
EVENT_KIND_WATCHDOG_TERMINATION = "watchdog-campaign-termination"

# Source-of-fact classification constants used by consumers.
SOURCE_CLIENT_LEDGER = "client-ledger"
SOURCE_SERVER_EVENT = "server-event"
SOURCE_PROVIDER_CONFIRMED = "provider-response-confirmed"


def _bounded_text(value: object, limit: int = 64) -> str:
    return str(value or "")[:limit]


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value and value != float("inf"):
        return float(value)
    return None


def _coverage(state: str) -> dict[str, Any]:
    return {"coverage": state if state in _COVERAGE_VOCAB else COVERAGE_MISSING}


def _observed_or_missing(entries: Sequence[Mapping[str, Any]] | None) -> str:
    if entries is None:
        return COVERAGE_UNOBSERVED
    return COVERAGE_OBSERVED if any(True for _ in entries) else COVERAGE_MISSING


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        rank = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return round(ordered[rank], 1)

    return {"count": len(ordered), "p50": pct(50), "p95": pct(95), "max": round(max(ordered), 1)}


def _request_latency_audit_view(
    audit: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bounded aggregates over the client's local HTTP timing audit."""
    if audit is None:
        return {}, _coverage(COVERAGE_UNOBSERVED)
    buckets: dict[str, list[float]] = {}
    for item in audit:
        if not isinstance(item, Mapping):
            continue
        operation = _bounded_text(item.get("operation"), 24).replace(" ", "-") or "other"
        latency = _number(item.get("latency_ms"))
        # Only completed attempts carry a positive measured latency; the
        # in-flight 0.0 placeholder of an attempt-start row is never a
        # measured zero-second request (kodmial/aa#431 semantics).
        if latency is None or latency <= 0:
            continue
        state = str(item.get("state", "in-flight"))[:24]
        if state == "in-flight":
            continue
        buckets.setdefault(operation, []).append(latency)
    if not buckets:
        return {}, _coverage(COVERAGE_MISSING)
    aggregates = {operation: _aggregate(values) for operation, values in sorted(buckets.items())}
    return aggregates, _coverage(COVERAGE_OBSERVED)


def _token_usage_view(
    audit: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    """Bounded per-agent token usage counters (plain numeric facts).

    Mirrors the normal end-of-campaign ``opencode_token_usage_by_agent``
    shape: one bucket per logical agent carrying the request count and
    token-kind totals. The watchdog-FAIL lane must not swap the same metric
    key for a flat token-kind dict, or per-agent attribution is lost.
    """
    if audit is None:
        return {}, _coverage(COVERAGE_UNOBSERVED)
    usage: dict[str, dict[str, int]] = {}
    for item in audit:
        if not isinstance(item, Mapping):
            continue
        agent = _bounded_text(item.get("agent"), 128) or "unknown"
        bucket = usage.setdefault(
            agent,
            {
                "requests": 0,
                "input_total": 0,
                "output_total": 0,
                "reasoning_total": 0,
                "cache_read_total": 0,
                "cache_write_total": 0,
            },
        )
        bucket["requests"] += 1
        for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
            value = item.get(key)
            if isinstance(value, (int, float)) and float(value) >= 0:
                bucket[f"{key}_total"] += int(value)
    if not usage:
        return {}, _coverage(COVERAGE_MISSING)
    return usage, _coverage(COVERAGE_OBSERVED)


def _served_models_view(
    audit: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if audit is None:
        return {}, _coverage(COVERAGE_UNOBSERVED)
    served: dict[str, list[str]] = {}
    for item in audit:
        if not isinstance(item, Mapping):
            continue
        agent = _bounded_text(item.get("agent"), 128)
        model = _bounded_text(item.get("served"), 128)
        if not agent or not model:
            continue
        models = served.setdefault(agent, [])
        if model not in models:
            models.append(model)
    if not served:
        return {}, _coverage(COVERAGE_MISSING)
    return served, _coverage(COVERAGE_OBSERVED)


def _process_state_view(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload is None:
        return {}, _coverage(COVERAGE_UNOBSERVED)
    bounded: dict[str, Any] = {}
    for key in ("owned_by_runtime", "alive", "exit_code", "spawned", "started"):
        if key in payload:
            bounded[key] = payload[key]
    if "runtime_state_error" in payload:
        return {}, _coverage(COVERAGE_UNOBSERVED)
    return bounded, _coverage(COVERAGE_OBSERVED)


def _stage_outcome_counts(
    snapshots: Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if not snapshots:
        return counts
    fields = (
        ("planner", "planner_outcome"),
        ("retrieval", "retrieval_outcome"),
        ("answer", "answer_outcome"),
        ("verifier", "verifier_outcome"),
    )
    for stage, field in fields:
        outcomes: dict[str, int] = {}
        for snapshot in snapshots:
            if not isinstance(snapshot, Mapping):
                continue
            outcome = _bounded_text(snapshot.get(field), 48).replace(" ", "-")
            if not outcome:
                continue
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcomes:
            counts[stage] = outcomes
    return counts


def stage_latency_coverage(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Honest coverage verdict for one telemetry snapshot's stage latencies.

    Closed vocabulary per stage (never a fabricated substitute latency):

    - ``measured``: the recorded latency field is a strictly positive
      number (a real measurement exists, regardless of later aggregation);
    - ``zero-unproven``: the recorded field is exactly ``0`` *and* the
      same stage recorded an outcome — the pipeline did stamp this stage,
      but a true 0.0 ms measurement cannot be distinguished from a
      skipped-stamp zero at this projection and no latency is invented;
    - ``unobserved``: the latency field is ``0``/absent and the stage
      outcome is empty — the stage never even reported an outcome;
    - ``not-measured``: the latency field itself is absent.
    """
    verdicts: dict[str, str] = {}
    for stage, field, outcome_field in (
        ("planner", "planner_latency_ms", "planner_outcome"),
        ("retrieval", "retrieval_latency_ms", "retrieval_outcome"),
        ("answer", "answer_latency_ms", "answer_outcome"),
        ("verifier", "verifier_latency_ms", "verifier_outcome"),
    ):
        latency = snapshot.get(field)
        outcome = _bounded_text(snapshot.get(outcome_field), 48).replace(" ", "-")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            verdicts[stage] = "not-measured"
            continue
        if float(latency) > 0:
            verdicts[stage] = "measured"
            continue
        verdicts[stage] = "zero-unproven" if outcome else "unobserved"
    return verdicts


def stage_latency_view(
    snapshots: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measured-only per-stage latency aggregates plus honest coverage.

    Mirrors the normal-end ``stage_latency_ms`` contract (a stage appears
    only when a positive measurement exists) and adds the per-stage
    coverage counters, including ``unobserved`` turns where the zero in
    the telemetry stamp is not provable as a real measurement.
    """
    if snapshots is None:
        return {}, {
            "coverage": COVERAGE_UNOBSERVED,
            "stage_count": 0,
            "turn_count": 0,
        }
    measured: dict[str, list[float]] = {}
    coverage_counts: dict[str, dict[str, int]] = {}
    turn_count = 0
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        turn_count += 1
        verdicts = stage_latency_coverage(snapshot)
        for stage, verdict in verdicts.items():
            counts = coverage_counts.setdefault(stage, {})
            counts[verdict] = counts.get(verdict, 0) + 1
            latency = snapshot.get(f"{stage}_latency_ms")
            if (
                verdict == "measured"
                and isinstance(latency, (int, float))
                and not isinstance(latency, bool)
            ):
                measured.setdefault(stage, []).append(float(latency))
    aggregates = {stage: _aggregate(values) for stage, values in sorted(measured.items())}
    coverage: dict[str, Any] = {
        "coverage": COVERAGE_OBSERVED if turn_count else COVERAGE_MISSING,
        "stage_count": len(coverage_counts),
        "turn_count": turn_count,
    }
    if coverage_counts:
        coverage["stages"] = coverage_counts
    return aggregates, coverage


def stage_telemetry_count(snapshots: Sequence[Mapping[str, Any]] | None) -> int:
    return sum(1 for snapshot in (snapshots or ()) if isinstance(snapshot, Mapping))


def build_partial_evidence_snapshot(
    *,
    phase: str,
    scenario_index: int | None = None,
    family: str = "",
    causality_id: str = "",
    current_request_state: str = "",
    scenario_total: int | None = None,
    completed_scenario_count: int | None = None,
    scenarios_started: int | None = None,
    scenarios_interrupted: int | None = None,
    product_outcome_counts: Mapping[str, int] | None = None,
    delivered_service_error_count: int | None = None,
    stage_snapshots: Sequence[Mapping[str, Any]] | None = None,
    attempt_entries: Sequence[Mapping[str, Any]] | None = None,
    served_audit: Sequence[Mapping[str, Any]] | None = None,
    token_audit: Sequence[Mapping[str, Any]] | None = None,
    request_audit: Sequence[Mapping[str, Any]] | None = None,
    process_state: Mapping[str, Any] | None = None,
    server_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one bounded partial-evidence snapshot from measured facts.

    Every argument is an already-privacy-safe projection measured by the
    lane (never generated here). Absent sources are explicitly recorded
    in ``coverage`` — the parent never receives a fabricated zero, and
    the snapshot is small enough to travel inside every progress write.
    """
    attempts_view: list[dict[str, Any]] = []
    attempt_coverage = _coverage(COVERAGE_MISSING)
    in_flight = 0
    late_outcomes = 0
    if attempt_entries is None:
        attempt_coverage = _coverage(COVERAGE_UNOBSERVED)
    else:
        for entry in attempt_entries:
            if not isinstance(entry, Mapping):
                continue
            state = str(entry.get("state", ""))[:24]
            if state == "in-flight":
                in_flight += 1
            if str(entry.get("late_outcome", ""))[:24]:
                late_outcomes += 1
            attempts_view.append(attempt_view(entry))
        attempts_view = attempts_view[-_MAX_ATTEMPT_ENTRIES:]
        attempt_coverage = _coverage(COVERAGE_OBSERVED if attempts_view else COVERAGE_MISSING)

    served_view, served_coverage = _served_models_view(served_audit)
    token_view, token_coverage = _token_usage_view(token_audit)
    request_view, request_coverage = _request_latency_audit_view(request_audit)
    process_view, process_coverage = _process_state_view(process_state)

    if server_evidence is None:
        server_events = unavailable_evidence("no-server-vantage-configured")
        server_coverage = _coverage(COVERAGE_UNOBSERVED)
        server_state = "unobserved"
    elif isinstance(server_evidence, Mapping):
        server_events = dict(server_evidence)
        server_state = _bounded_text(server_events.get("state"), 24)
        # Only the typed server-evidence states decide coverage. A real
        # observation of the documented surface is ``observed``; the typed
        # absence states (unsupported/offline/never-probed) carry THEMSELVES
        # as coverage so a stale or absent vantage can never be read as a
        # seen server fact — while an undocumented state drift stays
        # ``missing``, never a pretended observation.
        if server_state == "supported":
            server_coverage = _coverage(COVERAGE_OBSERVED)
        elif server_state in ("unsupported", "disconnected", "unobserved", "unavailable"):
            server_coverage = _coverage(server_state)
        else:
            server_coverage = _coverage(COVERAGE_MISSING)
    else:
        server_events = unavailable_evidence("invalid-server-evidence-shape")
        server_coverage = _coverage(COVERAGE_MISSING)
        server_state = "unobserved"

    stage_latency, stage_coverage = stage_latency_view(stage_snapshots)
    stage_outcomes = _stage_outcome_counts(stage_snapshots)

    coverage: dict[str, Any] = {
        "client_attempt_ledger": attempt_coverage["coverage"],
        "served_model_audit": served_coverage["coverage"],
        "token_usage_audit": token_coverage["coverage"],
        "request_latency_audit": request_coverage["coverage"],
        "runtime_process_state": process_coverage["coverage"],
        "server_events": server_coverage["coverage"] if server_state else COVERAGE_MISSING,
        "stage_telemetry": stage_coverage["coverage"],
        "scenario_accounting": (
            COVERAGE_OBSERVED if product_outcome_counts is not None else COVERAGE_MISSING
        ),
    }

    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "partial": True,
        "phase": _bounded_text(phase),
        "updated_at": round(time.time(), 3),
        "updated_monotonic": round(time.monotonic(), 3),
        "source": "child-lane",
        "coverage": coverage,
        "coverage_vocab": list(_COVERAGE_VOCAB),
        "source_types": {
            "client_attempt_ledger": SOURCE_CLIENT_LEDGER,
            "served_model_audit": SOURCE_CLIENT_LEDGER,
            "token_usage_audit": SOURCE_CLIENT_LEDGER,
            "request_latency_audit": SOURCE_CLIENT_LEDGER,
            "runtime_process_state": SOURCE_CLIENT_LEDGER,
            "server_events": (
                SOURCE_PROVIDER_CONFIRMED
                if isinstance(server_evidence, Mapping) and "server_error_kind" in server_evidence
                else SOURCE_SERVER_EVENT
            ),
            "stage_telemetry": SOURCE_CLIENT_LEDGER,
            "scenario_accounting": SOURCE_CLIENT_LEDGER,
        },
        "attempt_audit": attempts_view,
        "attempt_counts": {
            "rows": len(attempts_view),
            "in_flight": in_flight,
            "late_outcome": late_outcomes,
        },
        "served_models_by_agent": served_view,
        "opencode_token_usage_by_agent": token_view,
        "opencode_request_latency_ms": request_view,
        "runtime_process": process_view,
        "server_events": server_events,
        "server_evidence_state": server_state or "unobserved",
        "stage_latency_ms": stage_latency,
        "stage_latency_coverage": stage_coverage,
        "stage_outcome_counts": stage_outcomes,
        "scenario_totals": {
            "completed_scenario_count": completed_scenario_count,
            "started": scenarios_started,
            "interrupted": scenarios_interrupted,
            "total": scenario_total,
        },
        "current_scenario": {
            "index": scenario_index,
            "family": _bounded_text(family) or "unknown",
            "causality_id": _bounded_text(causality_id, 64),
            "request_state": _bounded_text(current_request_state, 48) or "unobserved",
        },
        "product_outcome_counts": {
            _bounded_text(key, 48).replace(" ", "-") or "unknown": int(value)
            for key, value in (product_outcome_counts or {}).items()
            if isinstance(value, (int, float))
        },
        "delivered_service_error_count": delivered_service_error_count,
        "fact_sources": {
            "attempt_audit_max_entries": _MAX_ATTEMPT_ENTRIES,
            "latency_samples_reference": _MAX_LATENCY_SAMPLES_REF,
        },
    }
    return snapshot


def validated_partial_evidence(progress: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Fail-closed validation of the persisted ``partial_evidence`` marker.

    Returns the bounded snapshot when the marker exists with the expected
    schema, or ``None`` (missing) when nothing valid was recorded — never
    a partially trusted mixture.
    """
    if not isinstance(progress, Mapping):
        return None
    payload = progress.get("partial_evidence")
    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("schema_version", "")) != SCHEMA_VERSION:
        return None
    if payload.get("partial") is not True:
        return None
    return dict(payload)


def coverage_from_partial(partial: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-source observed/missing/unobserved coverage (plus key absence)."""
    coverage: dict[str, Any] = {}
    for source in SOURCES:
        if partial is None:
            coverage[source] = COVERAGE_MISSING
            continue
        raw = partial.get("coverage")
        state = raw.get(source) if isinstance(raw, Mapping) else None
        coverage[source] = state if state in _COVERAGE_VOCAB else COVERAGE_MISSING
    return coverage


def diagnostics_unavailable_coverage() -> dict[str, Any]:
    """Typed coverage block proving NOTHING was recorded for any source."""
    return {source: COVERAGE_MISSING for source in SOURCES}


def merge_partial_into_metrics(
    metrics: dict[str, Any],
    partial: Mapping[str, Any] | None,
    *,
    exact_current_fields: bool = False,
) -> dict[str, Any]:
    """Project validated partial evidence into the timeout-lane metrics.

    Only measured facts travel, under the same top-level metric names the
    normal end-of-campaign lane publishes. Sources whose data never got
    recorded stay out of the metric namespace and are reported explicitly
    in ``partial_evidence_coverage``; when every source is absent (or no
    partial marker exists at all) the typed
    ``watchdog_diagnostics_status = diagnostics-unavailable`` marker is
    carried instead of any invented zero/turn/success.
    """
    metrics["partial_evidence_reported"] = bool(partial)
    metrics["partial_evidence_coverage"] = coverage_from_partial(partial)
    if partial is None:
        metrics["watchdog_diagnostics_status"] = DIAGNOSTICS_UNAVAILABLE
        metrics["diagnostics_unavailable_sources"] = sorted(
            source
            for source, state in metrics["partial_evidence_coverage"].items()
            if state != COVERAGE_OBSERVED
        )
        return metrics

    coverage = coverage_from_partial(partial)
    metrics["watchdog_diagnostics_status"] = (
        "observed-partial"
        if any(state == COVERAGE_OBSERVED for state in coverage.values())
        else DIAGNOSTICS_UNAVAILABLE
    )

    # Projection per source, keyed on the coverage verdict per source.
    if coverage["client_attempt_ledger"] == COVERAGE_OBSERVED:
        metrics["opencode_attempt_audit"] = partial.get("attempt_audit", [])
        counts = partial.get("attempt_counts", {})
        if isinstance(counts, Mapping):

            def _bounded_count(value: object) -> int | None:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return None
                if value != value or value == float("inf") or float(value) < 0:
                    return None
                return int(value)

            in_flight = _bounded_count(counts.get("in_flight"))
            late_outcome = _bounded_count(counts.get("late_outcome"))
            if in_flight is not None:
                metrics["opencode_attempt_in_flight_count"] = in_flight
            if late_outcome is not None:
                metrics["opencode_attempt_late_count"] = late_outcome
    if coverage["served_model_audit"] == COVERAGE_OBSERVED:
        metrics["served_models_by_agent"] = partial.get("served_models_by_agent", {})
    if coverage["token_usage_audit"] == COVERAGE_OBSERVED:
        metrics["opencode_token_usage_by_agent"] = partial.get("opencode_token_usage_by_agent", {})
    if coverage["request_latency_audit"] == COVERAGE_OBSERVED:
        metrics["opencode_request_latency_ms"] = partial.get("opencode_request_latency_ms", {})
    if coverage["runtime_process_state"] == COVERAGE_OBSERVED:
        metrics["opencode_runtime_process"] = partial.get("runtime_process", {})
    if coverage["stage_telemetry"] == COVERAGE_OBSERVED:
        metrics["stage_outcome_counts"] = partial.get("stage_outcome_counts", {})
        metrics["stage_latency_ms"] = partial.get("stage_latency_ms", {})
        metrics["stage_latency_coverage"] = partial.get("stage_latency_coverage", {})
    if coverage["scenario_accounting"] == COVERAGE_OBSERVED:
        totals = partial.get("scenario_totals")
        if isinstance(totals, Mapping):
            for target, key in (
                ("scenarios_started", "started"),
                ("scenarios_interrupted", "interrupted"),
                ("completed_scenario_count", "completed_scenario_count"),
                ("scenario_total", "total"),
            ):
                value = totals.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) >= 0
                ):
                    metrics.setdefault(target, int(value))
        outcome_counts = partial.get("product_outcome_counts")
        metrics["product_outcome_counts"] = (
            dict(outcome_counts) if isinstance(outcome_counts, Mapping) else {}
        )
        delivered = partial.get("delivered_service_error_count")
        if isinstance(delivered, int):
            metrics["delivered_service_error_count"] = delivered
    if coverage["server_events"] != COVERAGE_MISSING:
        # Observed AND typed absent states both travel: a documented
        # unsupported/disconnected/unobserved verdict inside the FAIL
        # artifact is the honest coverage the consumer needs, while a
        # silently missing vantage stays out of the metric namespace.
        metrics["opencode_server_events"] = partial.get("server_events", {})
    return metrics


__all__ = [
    "COVERAGE_MISSING",
    "COVERAGE_OBSERVED",
    "COVERAGE_UNOBSERVED",
    "DIAGNOSTICS_UNAVAILABLE",
    "EVENT_KIND_CHILD_TIMEOUT",
    "EVENT_KIND_WATCHDOG_TERMINATION",
    "SCHEMA_VERSION",
    "SOURCE_CLIENT_LEDGER",
    "SOURCE_PROVIDER_CONFIRMED",
    "SOURCE_SERVER_EVENT",
    "SOURCES",
    "build_partial_evidence_snapshot",
    "coverage_from_partial",
    "diagnostics_unavailable_coverage",
    "merge_partial_into_metrics",
    "stage_latency_coverage",
    "stage_latency_view",
    "validated_partial_evidence",
]
