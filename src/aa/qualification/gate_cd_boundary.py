"""Gate C/D isolation boundary (issue #340).

Separates Gate C genuine model/book/conversation behavior from Gate D real
Telegram transport and runtime readiness. Simulated outbound delivery is
never live Telegram delivery.

Gate responsibilities (authoritative #340):

- B: deterministic application/transport semantics and voice fixtures with
  isolated offline stubs. No live token/peer requirement.
- C: real production application, graph, configured LLM provider,
  canonical decrypted AA book, agent and verifier. Incoming raw Telegram
  adapter is exercised; outgoing Telegram network is simulated and labeled
  SIMULATED with trusted in-process receipt only. C PASS proves
  model/book/user-turn behavior, never external transport delivery.
- D: real Telegram Bot API identity, startup/health, polling/webhook
  ownership, callback/button readiness and state transitions. Optional
  owner-authorized real text/voice probes to an isolated test peer record
  genuine network ids/acknowledgments separately as
  ``external_send_confirmed`` or ``external_send_unverified``.

All helpers are pure (no network) so unit tests share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TELEGRAM_INGRESS_MODES = ("production-adapter", "stubbed", "unknown")
TELEGRAM_EGRESS_MODES = ("stubbed", "real")
DELIVERY_RECEIPT_KINDS = (
    "simulated-in-process",
    "external-confirmed",
    "external-unverified",
    "none",
)
NETWORK_PROBE_CONSENTS = ("opt-in", "opt-out", "absent")

EXTERNAL_SEND_CONFIRMED = "external_send_confirmed"
EXTERNAL_SEND_UNVERIFIED = "external_send_unverified"

# Typed Gate D external blockers (fail-closed, never PASS).
EXTERNAL_POLLER_CONFLICT = "EXTERNAL_POLLER_CONFLICT"
EXTERNAL_TEST_BOT_UNAVAILABLE = "EXTERNAL_TEST_BOT_UNAVAILABLE"
EXTERNAL_TELEGRAM_TOKEN_UNAVAILABLE = "EXTERNAL_TELEGRAM_TOKEN_UNAVAILABLE"
EXTERNAL_TELEGRAM_PEER_UNAVAILABLE = "EXTERNAL_TELEGRAM_PEER_UNAVAILABLE"
EXTERNAL_MANDATORY_NETWORK_FAILURE = "EXTERNAL_MANDATORY_NETWORK_FAILURE"
EXTERNAL_MANDATORY_FORBIDDEN = "EXTERNAL_MANDATORY_FORBIDDEN"

# Freshness window for a trusted read-only deployed-runtime READY marker.
# A marker older than this never proves a currently live poller.
TRUSTED_MARKER_FRESHNESS_S = 4 * 3600.0

# Explicit old-check to new-check ownership mapping. Preserves the original
# 1-41 plus voice 1-16 coverage: every historic check prefix maps to exactly
# one owning gate, so no assertion is silently dropped during the split.
OLD_CHECK_TO_GATE: dict[str, str] = {}

for _prefix in (
    "01-",
    "02-",
    "03-04-",
    "05-",
    "06-",
    "07-",
    "08-",
    "08b-",
    "09-11-",
    "12-",
    "13-",
    "14-15-",
    "16-",
    "16b-",
    "17-",
    "18-",
    "19-",
    "20-",
    "21-",
    "22-",
    "23-",
    "24-",
    "25b-",
):
    OLD_CHECK_TO_GATE[_prefix] = "B"
for _prefix in (
    "25-",
    "26-",
    "27-",
    "28-",
    "29-",
    "30-",
    "31-",
    "32-",
):
    OLD_CHECK_TO_GATE[_prefix] = "B"
for _prefix in (
    "33",
    "34-",
    "35-",
    "36-",
    "37-",
    "38-",
    "39-",
    "40-",
    "41-bootstrap",
    "41b-",
):
    OLD_CHECK_TO_GATE[_prefix] = "B"
for _prefix in (
    "42-",
    "43-",
    "43b-",
    "43c-",
    "44-",
    "45-",
    "46-",
    "47-",
    "47b-",
):
    OLD_CHECK_TO_GATE[_prefix] = "D"
for _prefix in ("V01-", "V02-", "V03-", "V04-", "V05-", "V05b-", "V06-08-", "V09-11-"):
    OLD_CHECK_TO_GATE[_prefix] = "B"
for _prefix in ("V09-cache", "V12-", "V13-", "V14-", "V15-", "V16-", "GATE-"):
    OLD_CHECK_TO_GATE[_prefix] = "B"
for _prefix in (
    "live-raw-telegram-transport-boundary",
    "live-real-opencode-runtime-ready",
    "live-real-retrieval-index-loaded",
    "live-transport-",
    "live-delivery-",
    "live-book-grounding-",
    "live-answer-relevance-",
    "live-independent-helpfulness-",
    "live-meta-direct-",
    "live-continuation-helpful-",
    "live-step-",
    "live-short-admission-",
    "live-typo-variant-",
    "live-context-switch-",
    "live-negative-control-",
    "live-independent-judge-",
    "live-answer-no-generic-collapse",
    "live-substantive-grounded-book-answer",
    "live-answer-diversity",
    "live-typing-heartbeat-",
    "live-delivery-sendmessage-observed",
    "live-actual-served-model-identity",
    "live-independent-judge-primary-only",
    "live-verifier-",
    "live-planner-retrieval-answer-verifier-telemetry",
    "live-voice-",
    "live-text-latency-measured",
):
    OLD_CHECK_TO_GATE[_prefix] = "C"
for _prefix in (
    "real-telegram-startup-",
    "real-telegram-delivery-",
    "real-telegram-typing-stream-",
    "live-telegram-startup-",
    "live-telegram-delivery-",
    "live-telegram-send-",
):
    OLD_CHECK_TO_GATE[_prefix] = "D"


class GateBoundaryError(ValueError):
    """Raised when gate boundary metadata is unknown or contradictory."""


@dataclass(frozen=True)
class GateBoundary:
    """Explicit per-gate execution boundary (fail-closed metadata)."""

    real_provider: bool = False
    real_book: bool = False
    real_app: bool = False
    telegram_ingress_mode: str = "unknown"
    telegram_egress_mode: str = "stubbed"
    delivery_receipt_kind: str = "none"
    network_probe_consent: str = "absent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "real_provider": bool(self.real_provider),
            "real_book": bool(self.real_book),
            "real_app": bool(self.real_app),
            "telegram_ingress_mode": self.telegram_ingress_mode,
            "telegram_egress_mode": self.telegram_egress_mode,
            "delivery_receipt_kind": self.delivery_receipt_kind,
            "network_probe_consent": self.network_probe_consent,
        }


def validate_gate_boundary(boundary: GateBoundary) -> GateBoundary:
    """Validate boundary metadata, failing closed on unknown/contradiction."""
    if boundary.telegram_ingress_mode not in TELEGRAM_INGRESS_MODES:
        raise GateBoundaryError(f"unknown telegram ingress mode {boundary.telegram_ingress_mode!r}")
    if boundary.telegram_egress_mode not in TELEGRAM_EGRESS_MODES:
        raise GateBoundaryError(f"unknown telegram egress mode {boundary.telegram_egress_mode!r}")
    if boundary.delivery_receipt_kind not in DELIVERY_RECEIPT_KINDS:
        raise GateBoundaryError(f"unknown delivery receipt kind {boundary.delivery_receipt_kind!r}")
    if boundary.network_probe_consent not in NETWORK_PROBE_CONSENTS:
        raise GateBoundaryError(f"unknown network probe consent {boundary.network_probe_consent!r}")
    # Contradiction: stubbed egress can never yield an external receipt.
    if boundary.telegram_egress_mode == "stubbed" and boundary.delivery_receipt_kind in (
        "external-confirmed",
    ):
        raise GateBoundaryError("in-memory message id mislabeled as external delivery")
    # Contradiction: external confirmation requires real egress + opt-in.
    if boundary.delivery_receipt_kind == "external-confirmed":
        if boundary.telegram_egress_mode != "real":
            raise GateBoundaryError("external-confirmed receipt requires real egress")
        if boundary.network_probe_consent != "opt-in":
            raise GateBoundaryError("external-confirmed receipt requires opt-in consent")
    # Contradiction: real provider/book claims without a real app path.
    if (boundary.real_provider or boundary.real_book) and not boundary.real_app:
        raise GateBoundaryError("real provider/book requires the real application path")
    return boundary


def assert_no_external_claim_from_simulated(
    *, telegram_egress_mode: str, delivery_receipt_kind: str, message_id: Any
) -> None:
    """Fail closed when a simulated ack is presented as external delivery.

    In-memory ``sendMessage`` stubs return sequential synthetic ids. Those
    ids prove in-process serialization only and must never satisfy an
    ``external_send_confirmed`` acceptance criterion.
    """
    _ = message_id
    if telegram_egress_mode == "stubbed" and delivery_receipt_kind == "external-confirmed":
        raise GateBoundaryError("simulated outbound ack cannot prove external delivery")


def gate_c_boundary() -> GateBoundary:
    """Canonical Gate C boundary: real behavior, simulated egress."""
    return validate_gate_boundary(
        GateBoundary(
            real_provider=True,
            real_book=True,
            real_app=True,
            telegram_ingress_mode="production-adapter",
            telegram_egress_mode="stubbed",
            delivery_receipt_kind="simulated-in-process",
            network_probe_consent="absent",
        )
    )


def gate_b_boundary() -> GateBoundary:
    """Canonical Gate B boundary: deterministic offline stubs only."""
    return validate_gate_boundary(
        GateBoundary(
            real_provider=False,
            real_book=False,
            real_app=True,
            telegram_ingress_mode="stubbed",
            telegram_egress_mode="stubbed",
            delivery_receipt_kind="none",
            network_probe_consent="absent",
        )
    )


def gate_d_boundary(*, egress_real: bool, consent_opt_in: bool) -> GateBoundary:
    """Canonical Gate D boundary for a real-network readiness evaluation."""
    return validate_gate_boundary(
        GateBoundary(
            real_provider=False,
            real_book=False,
            real_app=True,
            telegram_ingress_mode="production-adapter",
            telegram_egress_mode="real" if egress_real else "stubbed",
            delivery_receipt_kind=(
                "external-confirmed"
                if (egress_real and consent_opt_in)
                else ("external-unverified" if egress_real else "none")
            ),
            network_probe_consent="opt-in" if consent_opt_in else "absent",
        )
    )


def owner_gate_for_check(check_id: str) -> str:
    """Return the owning gate for one historic check id (fail-closed)."""
    name = (check_id or "").strip()
    for prefix, gate in OLD_CHECK_TO_GATE.items():
        if name == prefix.rstrip("-") or name.startswith(prefix):
            return gate
    # Live lane check families without a numeric prefix keep their lane owner.
    if name.startswith("live-"):
        return "C"
    if name.startswith("real-telegram-") or name.startswith("live-telegram-"):
        return "D"
    raise GateBoundaryError(f"unknown check ownership for {check_id!r}")


def coverage_gates() -> dict[str, list[str]]:
    """Return the preserved old-check coverage grouped by owning gate."""
    grouped: dict[str, list[str]] = {"B": [], "C": [], "D": []}
    for prefix, gate in OLD_CHECK_TO_GATE.items():
        grouped.setdefault(gate, []).append(prefix)
    for gate in grouped:
        grouped[gate] = sorted(grouped[gate])
    return grouped


def classify_optional_send_status(
    *,
    peer_configured: bool,
    consent_opt_in: bool,
    message_id: int | None = None,
    delivery_confirmed: bool = False,
) -> str:
    """Return the optional external-delivery status (never a mandatory verdict).

    Configuration alone (peer plus opt-in) never proves external delivery:
    ``external_send_confirmed`` additionally requires a genuine Bot API
    integer message id and a ``confirm_full_delivery`` acknowledgment. Any
    missing/invalid proof stays ``external_send_unverified`` (fail-closed,
    no synthetic proof).
    """
    if not (peer_configured and consent_opt_in):
        return EXTERNAL_SEND_UNVERIFIED
    if not delivery_confirmed:
        return EXTERNAL_SEND_UNVERIFIED
    if not isinstance(message_id, bool) and isinstance(message_id, int) and message_id > 0:
        return EXTERNAL_SEND_CONFIRMED
    return EXTERNAL_SEND_UNVERIFIED


__all__ = [
    "DELIVERY_RECEIPT_KINDS",
    "EXTERNAL_MANDATORY_FORBIDDEN",
    "EXTERNAL_MANDATORY_NETWORK_FAILURE",
    "EXTERNAL_POLLER_CONFLICT",
    "EXTERNAL_SEND_CONFIRMED",
    "EXTERNAL_SEND_UNVERIFIED",
    "EXTERNAL_TELEGRAM_PEER_UNAVAILABLE",
    "EXTERNAL_TELEGRAM_TOKEN_UNAVAILABLE",
    "EXTERNAL_TEST_BOT_UNAVAILABLE",
    "NETWORK_PROBE_CONSENTS",
    "OLD_CHECK_TO_GATE",
    "TELEGRAM_EGRESS_MODES",
    "TELEGRAM_INGRESS_MODES",
    "TRUSTED_MARKER_FRESHNESS_S",
    "GateBoundary",
    "GateBoundaryError",
    "assert_no_external_claim_from_simulated",
    "classify_optional_send_status",
    "coverage_gates",
    "gate_b_boundary",
    "gate_c_boundary",
    "gate_d_boundary",
    "owner_gate_for_check",
    "validate_gate_boundary",
]
