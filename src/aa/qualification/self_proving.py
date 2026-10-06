"""Self-proving layered qualification DAG (issue #146).

Authoritative Gates A-F for the exact current ``main`` SHA:

- Gate A -- static/build: exact SHA checkout + clean tree, canonical
  restore metadata/fingerprints, unit/type/lint, prompt/config/model/
  retrieval fingerprint consistency.
- Gate B -- deterministic component integration: real RU corpus restore +
  real BM25/E5/FAISS index, planner cardinality/shape,
  retrieval/RRF/dedup/diversity/small-to-big, grounding/verifier and output
  envelope, memory/FIFO/concurrency/safety, voice fixtures/cache/resource
  checks. Failures attribute to a concrete component, never a generic
  PASS/INCOMPLETE.
- Gate C -- LIVE exact production conversation path: repository-owned
  workflow with required secrets/assets, real local OpenCode process and
  real configured runtime provider/model policy, real decrypted RU corpus +
  real production index, synthetic Telegram Updates injected only at the
  production Telegram adapter boundary, then the exact same
  ``Application -> dispatcher -> LangGraph -> planner -> retrieval ->
  answer -> verifier -> delivery`` code used by the bot. No stubbing of
  planner/answer/verifier/provider/retrieval. Scenario families (never
  exact-question whitelists), response-diversity assertion, privacy-safe
  per-stage outcomes and latency only.
- Gate D -- real Telegram network/runtime readiness: real bot token,
  OpenCode health, ``getMe``, webhook cleanup/config, commands setup, long
  polling started, durable ``STARTING -> READY -> STOPPED|FAILED`` marker
  with exact run id/SHA. ``/bot status`` reports that marker.
- Gate E -- live performance/SLO: end-to-end and per-stage latency,
  p50/p95, hard guard (no ordinary turn >= 30s), typing heartbeat
  continuity verified independently.
- Gate F -- exact-main final verdict: PASS only if A-E PASS for the exact
  same current main SHA and matching product/runtime fingerprints. No stale
  reuse, no manual override.

Non-negotiable invariant: owner ``/run`` for manual testing stays available
regardless of qualification state. Qualification state controls only whether
the system may claim the product is READY/qualified, never whether the owner
may launch it. ``INCOMPLETE``, missing secret/model, stale SHA, mocked-only
evidence, or unknown state all mean FAIL/BLOCKED for qualification, never a
restriction on owner-initiated manual testing. The launched runtime is
labeled ``UNQUALIFIED`` without fresh exact-main PASS and publishes
``READY (UNQUALIFIED)`` or ``READY (QUALIFIED)`` once the poller is live.

All decision helpers here are pure (no network) so repository workflows
and unit tests share one implementation. Live execution lives in
``scripts/run_self_proving_qualification.py`` and the repository-owned
``aa-self-proving-qualification.yml`` workflow.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

ISSUE_NUMBER = 146
CONTROL_ISSUE_NUMBER = 31
QUALIFICATION_ISSUE_NUMBER = 7

GATE_IDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
MANDATORY_GATES: tuple[str, ...] = ("A", "B", "C", "D", "E")

GATE_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "INCOMPLETE", "STALE", "BLOCKED")
FINAL_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "BLOCKED")

# Gate C scenario families (behavioral families, never exact questions).
SCENARIO_FAMILIES: tuple[str, ...] = (
    "meta-capability",
    "substantive-drinking",
    "family-relationship",
    "followup-ellipsis",
    "topic-shift",
    "unsupported-out-of-book",
    "emergency",
    "long-conversation",
)

# Gate B attributable components (failures name one of these).
GATE_B_COMPONENTS: tuple[str, ...] = (
    "corpus-restore",
    "bm25-index",
    "e5-faiss-index",
    "planner-cardinality",
    "planner-shape",
    "retrieval-rrf",
    "retrieval-dedup",
    "retrieval-diversity",
    "small-to-big",
    "grounding",
    "verifier",
    "output-envelope",
    "memory-fifo",
    "concurrency",
    "safety",
    "voice-fixture",
    "voice-cache",
    "resource",
)

# Gate C pipeline stages exercised through the exact production boundary.
GATE_C_STAGES: tuple[str, ...] = (
    "application",
    "dispatcher",
    "langgraph",
    "planner",
    "retrieval",
    "answer",
    "verifier",
    "delivery",
)

# Automatic repair loop bound (Gate failure -> repair -> requalify cycles).
MAX_REPAIR_CYCLES = 3

# Live performance SLO (Gate E): no ordinary qualification turn may take
# >= 30s. p95 must be materially below that; the target tightens from the
# observed baseline after repair.
ORDINARY_TURN_BUDGET_MS = 30_000
P95_TARGET_MS = 15_000

# Provider/infrastructure policy carried by the DAG (never product PASS).
HTTP_429_RESTART_MARKER = "OPENCODE_429_RESTART_REQUIRED"
MIN_403_RETRY_DELAY_S = 5.0
MAX_403_RETRIES = 5

_CANARY_EVERY_HOURS = 4

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")

FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "text",
        "exact_text",
        "utterance",
        "answer",
        "generated_answer",
        "evidence_text",
        "source_text",
        "passage_text",
        "book_text",
        "content",
        "synthetic_input",
        "transcript",
        "summary",
        "chain_of_thought",
        "hidden_reasoning",
        "secret",
        "identity",
        "private_key",
        "token",
        "user_text",
        "corpus_text",
    }
)


class SelfProvingError(ValueError):
    """Raised when self-proving qualification invariants fail (fail-closed)."""


@dataclass(frozen=True)
class GateEvidence:
    """Privacy-safe evidence for one gate on one exact SHA."""

    gate: str
    status: str
    sha: str
    product_fingerprint: str = ""
    runtime_fingerprint: str = ""
    failure_category: str = ""
    component: str = ""
    run_id: str = ""
    live_trusted: bool = False
    mocked_only: bool = False
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    max_turn_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "sha": self.sha,
            "product_fingerprint": self.product_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "failure_category": self.failure_category,
            "component": self.component,
            "run_id": self.run_id,
            "live_trusted": self.live_trusted,
            "mocked_only": self.mocked_only,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "max_turn_ms": self.max_turn_ms,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalVerdict:
    """Exact-main final verdict (Gate F)."""

    sha: str
    status: str
    product_fingerprint: str
    runtime_fingerprint: str
    gates: tuple[GateEvidence, ...]
    blocking_gate: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aa-self-proving-verdict/1",
            "issue": ISSUE_NUMBER,
            "sha": self.sha,
            "status": self.status,
            "product_fingerprint": self.product_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "blocking_gate": self.blocking_gate,
            "run_id": self.run_id,
            "gates": [gate.to_dict() for gate in self.gates],
        }


@dataclass(frozen=True)
class FailureReport:
    """Machine-readable gate failure payload for the repair loop."""

    gate: str
    category: str
    component: str
    sha: str
    run_id: str
    latency_p50_ms: float
    latency_p95_ms: float
    max_turn_ms: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aa-self-proving-failure/1",
            "gate": self.gate,
            "category": self.category,
            "component": self.component,
            "sha": self.sha,
            "run_id": self.run_id,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "max_turn_ms": self.max_turn_ms,
            "detail": self.detail,
        }


def validate_exact_sha(value: str) -> str:
    """Normalize an exact 40-hex SHA or raise fail-closed."""
    normalized = (value or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise SelfProvingError("sha must be an exact 40-hex main SHA")
    return normalized


def validate_hex64(value: str) -> str:
    """Normalize a 64-hex fingerprint (empty means unknown)."""
    normalized = (value or "").strip().lower()
    if not normalized:
        return ""
    if not _HEX64_RE.fullmatch(normalized):
        raise SelfProvingError("fingerprint must be 64-hex or empty")
    return normalized


def assert_no_text_leak(payload: Any, owner: str = "self-proving") -> None:
    """Reject user/corpus-text-carrying keys in public evidence."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_EVIDENCE_KEYS:
                raise SelfProvingError(f"{owner}: forbidden key {key!r}")
            assert_no_text_leak(value, f"{owner}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            assert_no_text_leak(item, f"{owner}[{index}]")


def percentile_ms(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` (ms units)."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    rank = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


def normalize_gate_status(status: str) -> str:
    """Normalize a gate status or raise fail-closed on unknown state."""
    normalized = (status or "").strip().upper()
    if normalized not in GATE_STATUSES:
        raise SelfProvingError(f"unknown gate status {status!r}")
    return normalized


def effective_gate_status(evidence: GateEvidence, *, current_sha: str) -> str:
    """Apply the non-negotiable invariant to one gate's evidence.

    ``INCOMPLETE``, missing secret/model (``live_trusted`` False for live
    gates), stale SHA, mocked-only evidence, or unknown state all collapse
    to BLOCKED. This function never returns a warning grade.
    """
    current = validate_exact_sha(current_sha)
    status = normalize_gate_status(evidence.status)
    if evidence.sha != current:
        return "STALE"
    if status in ("STALE", "BLOCKED"):
        return "BLOCKED" if status == "BLOCKED" else "STALE"
    if status == "INCOMPLETE":
        return "BLOCKED"
    if status == "FAIL":
        return "FAIL"
    # status == PASS from here: still fail closed on trust markers.
    if evidence.mocked_only:
        return "BLOCKED"
    if evidence.gate in ("C", "D", "E") and not evidence.live_trusted:
        return "BLOCKED"
    return "PASS"


def decide_final_verdict(
    evidences: list[GateEvidence],
    *,
    current_sha: str,
    product_fingerprint: str,
    runtime_fingerprint: str,
    run_id: str,
) -> FinalVerdict:
    """Compute the Gate F exact-main final verdict (no stale reuse)."""
    current = validate_exact_sha(current_sha)
    product = validate_hex64(product_fingerprint)
    runtime = validate_hex64(runtime_fingerprint)
    if not product or not runtime:
        raise SelfProvingError("product/runtime fingerprints must be known 64-hex")
    if not run_id.strip() or any(c.isspace() for c in run_id):
        raise SelfProvingError("run_id must be a non-empty token")
    by_gate = {item.gate: item for item in evidences}
    ordered: list[GateEvidence] = []
    blocking = ""
    overall = "PASS"
    for gate in MANDATORY_GATES:
        evidence = by_gate.get(gate)
        if evidence is None:
            blocking = blocking or gate
            overall = "BLOCKED"
            ordered.append(
                GateEvidence(
                    gate=gate,
                    status="BLOCKED",
                    sha=current,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category="missing-evidence",
                    run_id=run_id,
                )
            )
            continue
        if evidence.product_fingerprint != product:
            blocking = blocking or gate
            overall = "BLOCKED"
            ordered.append(
                GateEvidence(
                    gate=gate,
                    status="BLOCKED",
                    sha=evidence.sha,
                    product_fingerprint=evidence.product_fingerprint,
                    runtime_fingerprint=evidence.runtime_fingerprint,
                    failure_category="fingerprint-mismatch",
                    component=evidence.component,
                    run_id=evidence.run_id or run_id,
                    live_trusted=evidence.live_trusted,
                    mocked_only=evidence.mocked_only,
                )
            )
            continue
        if evidence.runtime_fingerprint != runtime:
            blocking = blocking or gate
            overall = "BLOCKED"
            ordered.append(
                GateEvidence(
                    gate=gate,
                    status="BLOCKED",
                    sha=evidence.sha,
                    product_fingerprint=evidence.product_fingerprint,
                    runtime_fingerprint=evidence.runtime_fingerprint,
                    failure_category="runtime-mismatch",
                    component=evidence.component,
                    run_id=evidence.run_id or run_id,
                    live_trusted=evidence.live_trusted,
                    mocked_only=evidence.mocked_only,
                )
            )
            continue
        resolved = effective_gate_status(evidence, current_sha=current)
        if resolved == "PASS":
            ordered.append(evidence)
            continue
        if resolved in ("STALE", "BLOCKED"):
            overall = "BLOCKED"
        elif resolved == "FAIL" and overall != "BLOCKED":
            overall = "FAIL"
        blocking = blocking or gate
        ordered.append(evidence)
    if overall == "PASS":
        # Any mismatch above already flipped overall; re-assert unanimity.
        if any(effective_gate_status(item, current_sha=current) != "PASS" for item in ordered):
            overall = "BLOCKED"
            if not blocking:
                blocking = next(
                    (
                        item.gate
                        for item in ordered
                        if effective_gate_status(item, current_sha=current) != "PASS"
                    ),
                    "A",
                )
    status = "PASS" if overall == "PASS" else ("FAIL" if overall == "FAIL" else "BLOCKED")
    verdict = FinalVerdict(
        sha=current,
        status=status,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        gates=tuple(ordered),
        blocking_gate=blocking,
        run_id=run_id,
    )
    assert_no_text_leak(verdict.to_dict())
    return verdict


def is_run_allowed(verdict: FinalVerdict | None, *, current_sha: str) -> bool:
    """Return whether owner ``/run`` may start.

    The non-negotiable invariant (issue #146, control issue #31) is that
    owner/manual ``/run`` stays available regardless of qualification state.
    Qualification controls only the QUALIFIED/UNQUALIFIED label, never the
    launch itself. This helper therefore always returns ``True``; use
    :func:`decide_final_verdict` / :func:`effective_gate_status` for the
    separate qualification verdict (FAIL/BLOCKED vs PASS).
    """
    return True


def refusal_explanation(verdict: FinalVerdict | None, *, current_sha: str) -> str:
    """Describe the qualification label for an owner ``/run`` (advisory only).

    Never refuses the launch: the owner may always start manual testing. The
    returned text labels the runtime ``UNQUALIFIED`` (with the blocking gate
    named) or ``READY (QUALIFIED)`` when exact-main PASS holds. Privacy-safe,
    no user/corpus text.
    """
    try:
        current = validate_exact_sha(current_sha)
    except SelfProvingError:
        return "UNQUALIFIED: current main SHA is unknown; /run allowed as UNQUALIFIED."
    if verdict is None:
        return (
            "UNQUALIFIED: no self-proving qualification verdict for current main; "
            "/run allowed as UNQUALIFIED."
        )
    if verdict.sha != current:
        return (
            f"UNQUALIFIED: qualification verdict is for stale SHA {verdict.sha[:12]}; "
            f"current main is {current[:12]}; /run allowed as UNQUALIFIED pending requalification."
        )
    if verdict.status == "PASS" and not verdict.blocking_gate:
        return f"READY (QUALIFIED): exact main {current[:12]} holds fresh trusted PASS."
    blocking = verdict.blocking_gate or next(
        (
            gate.gate
            for gate in verdict.gates
            if effective_gate_status(gate, current_sha=current) != "PASS"
        ),
        "unknown",
    )
    detail = next(
        (gate for gate in verdict.gates if gate.gate == blocking),
        None,
    )
    if detail and detail.failure_category:
        category = detail.failure_category.strip()
    elif detail is not None:
        try:
            category = effective_gate_status(detail, current_sha=current)
        except SelfProvingError:
            category = "unknown"
    else:
        category = "unknown"
    return (
        f"UNQUALIFIED: Gate {blocking} blocks qualification (state={category}, "
        f"verdict={verdict.status}, sha={current[:12]}, run={verdict.run_id}); "
        "/run allowed as UNQUALIFIED; repair/requalification proceeds automatically."
    )


def failure_report_for_gate(evidence: GateEvidence) -> FailureReport:
    """Build the machine-readable failure payload for one gate failure."""
    status = normalize_gate_status(evidence.status)
    if status == "PASS":
        raise SelfProvingError("PASS gates carry no failure report")
    category = evidence.failure_category.strip() or (
        "component-fail" if status == "FAIL" else "incomplete-evidence"
    )
    report = FailureReport(
        gate=evidence.gate,
        category=category,
        component=evidence.component.strip(),
        sha=evidence.sha,
        run_id=evidence.run_id,
        latency_p50_ms=float(evidence.latency_p50_ms),
        latency_p95_ms=float(evidence.latency_p95_ms),
        max_turn_ms=float(evidence.max_turn_ms),
        detail=evidence.detail,
    )
    assert_no_text_leak(report.to_dict())
    return report


def repair_fingerprint(report: FailureReport) -> str:
    """Return the stable idempotency fingerprint for one repair root cause."""
    payload = "\x00".join(
        (
            report.gate.strip(),
            report.category.strip(),
            report.component.strip(),
            report.sha.strip(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def find_reusable_repair(open_repairs: list[tuple[int, str]], *, fingerprint: str) -> int | None:
    """Return the open repair issue already carrying a fingerprint, if any."""
    wanted = (fingerprint or "").strip()
    if not wanted:
        return None
    for number, existing in open_repairs:
        if str(existing).strip() == wanted:
            return int(number)
    return None


def rerun_plan(failed_gate: str) -> list[str]:
    """Return the rerun order: failed gate first, then downstream, then F."""
    gate = (failed_gate or "").strip().upper()
    if gate not in MANDATORY_GATES:
        raise SelfProvingError(f"unknown failed gate {failed_gate!r}")
    index = MANDATORY_GATES.index(gate)
    downstream = list(MANDATORY_GATES[index:])
    if "F" not in downstream:
        downstream.append("F")
    return downstream


def slo_guards(
    latencies_ms: list[float], *, p95_target_ms: float = float(P95_TARGET_MS)
) -> tuple[bool, str, dict[str, float]]:
    """Evaluate Gate E SLO guards (fail-closed on pathological latency)."""
    metrics = {
        "p50_ms": percentile_ms(latencies_ms, 50),
        "p95_ms": percentile_ms(latencies_ms, 95),
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        "turns": float(len(latencies_ms)),
        "budget_ms": float(ORDINARY_TURN_BUDGET_MS),
        "p95_target_ms": float(p95_target_ms),
    }
    if not latencies_ms:
        return False, "no latency samples; INCOMPLETE", metrics
    if metrics["max_ms"] >= float(ORDINARY_TURN_BUDGET_MS):
        return (
            False,
            (f"ordinary turn took {metrics['max_ms']:.0f}ms >= {ORDINARY_TURN_BUDGET_MS}ms budget"),
            metrics,
        )
    if metrics["p95_ms"] > float(p95_target_ms):
        return (
            False,
            (f"p95 {metrics['p95_ms']:.0f}ms exceeds target {p95_target_ms:.0f}ms"),
            metrics,
        )
    return True, "slo within guard", metrics


def diversity_passes(reply_signatures: list[str]) -> tuple[bool, str]:
    """Assert unrelated valid inputs did not collapse to one fallback.

    ``reply_signatures`` are privacy-safe reply hashes (never text). Two or
    more distinct scenario families sharing one generic signature means the
    production path collapsed unrelated turns to one fallback (the exact
    runtime #37422302821 failure Gate C must catch).
    """
    cleaned = [str(item).strip() for item in reply_signatures if str(item).strip()]
    if len(cleaned) < 2:
        return False, "diversity needs at least two unrelated family replies"
    distinct = set(cleaned)
    if len(distinct) <= 1:
        return False, "unrelated inputs collapsed to one generic fallback"
    return True, f"{len(distinct)}/{len(cleaned)} distinct replies"


def heartbeat_continuity_ok(
    *, sends: int, duration_ms: float, interval_ms: float
) -> tuple[bool, str]:
    """Verify typing heartbeat continuity independently (Gate E)."""
    if interval_ms <= 0:
        return False, "heartbeat interval must be > 0"
    if sends < 1:
        return False, "heartbeat never fired"
    expected = max(1, int(duration_ms // interval_ms) + 1)
    # Allow one missed refresh for scheduling jitter, never a dead gap.
    if sends + 1 < expected and duration_ms > interval_ms * 2:
        return False, f"heartbeat gap: {sends} sends for {duration_ms:.0f}ms"
    return True, f"heartbeat continuity: {sends} sends"


def retry_delay_403(attempt: int) -> float:
    """Bounded exponential retry delay for HTTP 403 (>= 5s base)."""
    bounded = max(0, min(int(attempt), MAX_403_RETRIES))
    return float(MIN_403_RETRY_DELAY_S) * float(2**bounded)


def is_429_restart(reason: str) -> bool:
    """Whether a failure reason requires runner retire/restart (resume)."""
    return HTTP_429_RESTART_MARKER in (reason or "")


def canary_scope() -> dict[str, Any]:
    """Return the bounded current-main canary scope (every 4h).

    The canary never blocks owner ``/run``; on failure it marks current main
    ``UNQUALIFIED`` and schedules automatic P0 repair while manual testing
    stays available labeled ``UNQUALIFIED``.
    """
    return {
        "schedule_hours": _CANARY_EVERY_HOURS,
        "gates": ["C-smoke", "D-readiness", "E-latency-guard"],
        "blocks_new_run_until_green": False,
        "never_blocks_owner_run": True,
        "auto_repair": True,
    }


def build_result_marker(*, sha: str, status: str, run: str) -> str:
    """Build the canonical self-proving result marker fragment."""
    normalized_sha = validate_exact_sha(sha)
    normalized_status = (status or "").strip().upper()
    if normalized_status not in FINAL_STATUSES:
        raise SelfProvingError("marker status must be PASS|FAIL|BLOCKED")
    if not run.strip() or any(c.isspace() for c in run):
        raise SelfProvingError("marker run id must be a non-empty token")
    return (
        f"<!-- aa-self-proving-qualification issue={ISSUE_NUMBER} "
        f"sha={normalized_sha} result={normalized_status.lower()} run={run} -->"
    )


def current_timestamp_s() -> float:
    """Return the current epoch seconds (single seam for tests)."""
    return time.time()


__all__ = [
    "CONTROL_ISSUE_NUMBER",
    "GATE_B_COMPONENTS",
    "GATE_C_STAGES",
    "GATE_IDS",
    "GATE_STATUSES",
    "ISSUE_NUMBER",
    "MANDATORY_GATES",
    "MAX_REPAIR_CYCLES",
    "MIN_403_RETRY_DELAY_S",
    "ORDINARY_TURN_BUDGET_MS",
    "P95_TARGET_MS",
    "QUALIFICATION_ISSUE_NUMBER",
    "SCENARIO_FAMILIES",
    "FinalVerdict",
    "FailureReport",
    "GateEvidence",
    "SelfProvingError",
    "assert_no_text_leak",
    "build_result_marker",
    "canary_scope",
    "current_timestamp_s",
    "decide_final_verdict",
    "diversity_passes",
    "effective_gate_status",
    "failure_report_for_gate",
    "find_reusable_repair",
    "heartbeat_continuity_ok",
    "is_429_restart",
    "is_run_allowed",
    "normalize_gate_status",
    "percentile_ms",
    "refusal_explanation",
    "repair_fingerprint",
    "rerun_plan",
    "retry_delay_403",
    "slo_guards",
    "validate_exact_sha",
    "validate_hex64",
]
