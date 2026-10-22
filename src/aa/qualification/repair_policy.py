"""Pure AA qualification-failure admission policy.

Product quality verdicts and permission to mutate code are distinct decisions.
This module NEVER changes a FAIL/BLOCKED qualification result into PASS.
Only independently identifiable product defects may launch a code repair.
Transient provider, model, infrastructure, harness and ambiguous faults do not.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_PROVIDERS = re.compile(
    r"opencodeprovideraccesserror|opencoderatelimiterror|opencodetimeouterror|"
    r"opencodetransienterror|provider[-_ ](?:access|"
    r"unavailable|error|down|timeout|rate)|model[-_ ](?:unavailable|route|probe|policy[-_ ]"
    r"unavailable)|http[-_ ]?(?:403|429|500|502|503|504)|rate[-_ ]?limit|"
    r"external[-_ ](?:model|provider|service)|transport[-_ ](?:timeout|network)|"
    r"connection(?:error|refused|reset)|network[-_ ](?:failure|error)|"
    r"service[-_ ]unavailable|quota[-_ ]exhausted",
    re.IGNORECASE,
)

_HARNESS = re.compile(
    r"workflow[-_ ](?:step|gate[-_ ][ab][- _]?shell)[-_ ]fail(?:ed)?|"
    r"evidence[-_ ](?:invalid|incomplete|missing)|"
    r"missing[-_ ](?:live|secret|fixture|artifact|checkpoint)|"
    r"harness|collector|modulenotfounderror|importerror|"
    r"runtime[-_ ](?:not[-_ ]ready|unavailable)|"
    r"live[-_ ]execution[-_ ]not[-_ ]enabled|stale[-_ ]main|"
    r"synthetic[-_ ]only|heldout[-_ ]fixture[-_ ]unavailable|"
    r"index[-_ ](?:missing|unavailable)",
    re.IGNORECASE,
)

_PRODUCT_C = re.compile(
    r"grounding|semantic|substantive|relevance|adequacy|"
    r"unsupported[-_ ]claim|book[-_ ]fidelity|quote[-_ ]origin|"
    r"followup|topic[-_ ]shift|conversation[-_ ]continuity|"
    r"response[-_ ]diversity|privacy[-_ ]leak|"
    r"safety[-_ ]violation|output[-_ ]guard|"
    r"source[-_ ]provenance|intent[-_ ]resolution|"
    r"live[-_ ]book[-_ ]retrieval|"
    r"retrieval[-_ ](?:coverage|wrong[-_ ]passage)",
    re.IGNORECASE,
)
_PERFORMANCE_E = re.compile(
    r"slo|latency|p95|turn[-_ ]budget|ordinary[-_ ]turn[-_ ]failed", re.IGNORECASE
)

# kodmial/aa#417: typed stalled-turn infrastructure evidence. Unlike a
# provider outage or a harness failure, this is what the run's own
# watchdog/scenario lifecycle recorded as dead air in a specific turn.
# It is never a product-semantic verdict.
_INFRA_TIMEOUT = re.compile(
    r"live[-_ ]scenario[-_ ]watchdog[-_ ]timeout|"
    r"live[-_ ]delivery[-_ ].*[-_ ]timeout|"
    r"watchdog[-_ ]timeout|delivery[-_ ]timeout",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Admission:
    actionable: tuple[int, ...]
    held: tuple[tuple[str, str], ...]
    # Typed failures that require a bounded diagnostic follow-up, not a
    # product-content code patch (kodmial/aa#417). Each entry carries the
    # failure index and the closed-vocabulary diagnostic class.
    diagnostics: tuple[tuple[int, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionable": list(self.actionable),
            "held": [{"gate": gate, "reason": reason} for gate, reason in self.held],
            "diagnostics": [
                {"index": index, "diagnostic_class": kind} for index, kind in self.diagnostics
            ],
        }


def _classify(failure: Mapping[str, Any]) -> str:
    gate = str(failure.get("gate", "")).upper()
    category = str(failure.get("failure_category", "") or "")
    component = str(failure.get("component", "") or "")
    signal = category + " " + component

    if str(failure.get("status", "")).upper() != "FAIL":
        return "not-a-failed-gate"
    if gate not in {"A", "B", "C", "E"}:
        return "not-mandatory"
    if _INFRA_TIMEOUT.search(signal):
        # A typed stalled-turn timeout is infrastructure evidence: it may
        # justify a diagnostic/action follow-up but never a speculative
        # semantic code patch.
        return "infrastructure-timeout"
    if _PROVIDERS.search(signal):
        return "provider-or-infrastructure"
    if _HARNESS.search(signal):
        return "harness-or-evidence"
    if gate in {"A", "B"}:
        # A and B own deterministic configuration, build, and integration
        # failures, unless specifically classified as infrastructure/harness.
        return "product"
    if gate == "C" and _PRODUCT_C.search(category):
        return "product"
    if gate == "E" and _PERFORMANCE_E.search(category):
        return "product"
    return "unattributed"


def plan_repair(
    failures: Sequence[Mapping[str, Any]],
    *,
    gate_c_breaker: bool = False,
    in_flight_repair: bool = False,
) -> Admission:
    """Return only proven code-mutation candidates, with typed held reasons.

    A global C breaker does not suppress an independent A/B defect.
    A live C provider fault makes latency E inconclusive; C must be healthy
    before latency can justify a separate source-code change.
    An active code repair serializes the entire AA qualification repair lane.
    """
    decisions = [_classify(failure) for failure in failures]
    c_unavailable = any(
        str(f.get("gate", "")).upper() == "C" and decision != "product"
        for f, decision in zip(failures, decisions, strict=True)
        if str(f.get("status", "")).upper() == "FAIL"
    )
    # A known-exhausted C failure can poison E timing samples. Do not
    # reopen the same C convergence loop under the alternative E fingerprint.
    c_exhausted = gate_c_breaker and any(
        str(f.get("gate", "")).upper() == "C" and str(f.get("status", "")).upper() == "FAIL"
        for f in failures
    )
    actionable: list[int] = []
    held: list[tuple[str, str]] = []
    diagnostics: list[tuple[int, str]] = []
    for index, (failure, reason) in enumerate(zip(failures, decisions, strict=True)):
        gate = str(failure.get("gate", "")).upper()
        if reason == "infrastructure-timeout":
            # Held as a code-mutation candidate but admitted as a typed
            # diagnostic follow-up; admission dedupe is enforced by the
            # caller/workflow fingerprint, never here.
            held.append((gate, reason))
            diagnostics.append((index, "infrastructure-timeout"))
            continue
        if reason == "provider-or-infrastructure":
            held.append((gate, reason))
            diagnostics.append((index, "provider-or-infrastructure"))
            continue
        if reason == "harness-or-evidence":
            held.append((gate, reason))
            diagnostics.append((index, "harness-or-evidence"))
            continue
        if reason != "product":
            held.append((gate, reason))
        elif gate == "C" and gate_c_breaker:
            held.append((gate, "global-gate-c-breaker"))
        elif gate == "E" and c_unavailable:
            held.append((gate, "latency-inconclusive-while-c-unavailable"))
        elif gate == "E" and c_exhausted:
            held.append((gate, "latency-dependent-on-exhausted-c"))
        elif in_flight_repair:
            held.append((gate, "repair-already-in-flight"))
        else:
            actionable.append(index)
    return Admission(actionable=tuple(actionable), held=tuple(held), diagnostics=tuple(diagnostics))
