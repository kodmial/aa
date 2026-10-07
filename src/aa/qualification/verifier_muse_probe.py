"""Muse Spark verifier access probe (kodmial/aa#202).

Reproduces the Muse verifier failure through the same local official
``opencode serve`` boundary used by production (ephemeral sessions via
:class:`aa.opencode.client.OpenCodeClient`), isolating whether a
rejection is tied to the custom ``aa-verifier-v2`` transport agent
selector versus Muse itself.

The verifier system prompt always travels through the native ``system``
field; only the transport ``agent`` selector varies (custom selector vs
omitted). Requested model stays pinned to Muse Spark on both variants.
Every result is privacy-safe: only error categories, numeric HTTP
statuses, model identities and wire/logical agent names are recorded.
No prompt text, system text, or model output ever enters the result or
the logs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL
from aa.conversation.model_adapter import VERIFIER_AGENT_V2
from aa.opencode.errors import OpenCodeError

logger = logging.getLogger("aa.qualification.verifier_muse_probe")

# Exact live failure vocabulary (kodmial/aa#202): the absence of a served
# verifier is a specific Muse-access failure, never a generic Gate C
# failure, and Space Bunny must never serve the verifier.
MUSE_ACCESS_FAILURE = "live-verifier-muse-access-failure"
SERVED_MISMATCH_FAILURE = "live-verifier-served-model-mismatch"
REQUESTED_PIN_FAILURE = "live-verifier-muse-requested-pinned"
SPACE_BUNNY_FORBIDDEN_FAILURE = "live-verifier-space-bunny-forbidden"

VERIFIER_EXPECTED_MODEL = DEFAULT_PRIMARY_MODEL
VERIFIER_FORBIDDEN_MODEL = DEFAULT_FALLBACK_MODEL

# Fixed neutral probe utterance. It carries no evaluation content and is
# never logged or returned; both wire variants use the identical text so
# the only variable is the transport agent selector.
_PROBE_TEXT = "Neutral connectivity probe."
_HTTP_STATUS_RE = re.compile(r"http=(\d{3})")


def _http_status_from(error: BaseException) -> int:
    """Extract the numeric HTTP status from a classified error (0 if none)."""
    match = _HTTP_STATUS_RE.search(str(error))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _error_category(error: BaseException) -> str:
    """Return the privacy-safe error category for ``error``."""
    kind = getattr(error, "kind", "")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()[:64]
    return type(error).__name__[:64]


async def _probe_variant(
    client: Any,
    *,
    model: str,
    system: str,
    wire_agent: str,
) -> dict[str, Any]:
    """Run one verifier-shaped probe variant over ephemeral sessions.

    Uses the plain-text path (already proven to serve on healthy routes)
    with the verifier system prompt in the native ``system`` field. Only
    the transport agent selector varies. Returns a privacy-safe outcome;
    never records prompt, system, or reply text.
    """
    session_id = ""
    try:
        session = await client.create_session(title="")
        session_id = str(getattr(session, "id", ""))
        reply = await client.send_message(
            session_id,
            _PROBE_TEXT,
            timeout=60.0,
            agent=wire_agent,
            model=model,
            system=system,
            audit_agent=VERIFIER_AGENT_V2,
        )
        _ = reply
        audit = getattr(client, "served_model_audit", ())
        served = ""
        try:
            entries = list(audit) if not callable(audit) else list(audit())
        except Exception:
            entries = []
        for item in reversed(entries):
            if isinstance(item, dict) and str(item.get("agent", "")) == VERIFIER_AGENT_V2:
                served = str(item.get("served", ""))
                break
        logger.info(
            "verifier muse probe served",
            extra={"wire_agent_set": bool(wire_agent), "served_present": bool(served)},
        )
        return {
            "outcome": "served",
            "category": "",
            "http_status": 0,
            "requested": model,
            "served": served,
            "wire_agent_set": bool(wire_agent),
        }
    except Exception as exc:
        category = _error_category(exc)
        status = _http_status_from(exc) if isinstance(exc, OpenCodeError) else 0
        logger.info(
            "verifier muse probe rejected",
            extra={"category": category, "http_status": status},
        )
        return {
            "outcome": "rejected",
            "category": category,
            "http_status": status,
            "requested": model,
            "served": "",
            "wire_agent_set": bool(wire_agent),
        }
    finally:
        if session_id:
            try:
                await client.delete_session(session_id)
            except Exception:
                logger.info("verifier muse probe cleanup failed")


async def probe_verifier_muse_access(
    client: Any,
    *,
    model: str = VERIFIER_EXPECTED_MODEL,
    system: str = "",
) -> dict[str, Any]:
    """Probe Muse access with and without the custom transport selector.

    Both variants use the same pinned model and the same verifier system
    prompt through the native ``system`` field; only the wire transport
    agent selector differs. The result carries no prompt, system, or
    output text -- only categories, numeric statuses, and identities.
    """
    system_text = system
    if not system_text.strip():
        try:
            from aa.conversation.v2_prompts import load_verifier_system_v2

            system_text = load_verifier_system_v2()
        except Exception:
            system_text = "verifier-system"
    custom = await _probe_variant(
        client, model=model, system=system_text, wire_agent=VERIFIER_AGENT_V2
    )
    omitted = await _probe_variant(client, model=model, system=system_text, wire_agent="")
    if custom.get("outcome") == "served" and omitted.get("outcome") == "served":
        diagnosis = "served-both"
    elif custom.get("outcome") != "served" and omitted.get("outcome") == "served":
        diagnosis = "agent-selector-rejection"
    elif custom.get("outcome") == "served" and omitted.get("outcome") != "served":
        diagnosis = "omitted-selector-rejection"
    elif custom.get("category") != omitted.get("category") or custom.get(
        "http_status"
    ) != omitted.get("http_status"):
        diagnosis = "selector-sensitive-rejection"
    else:
        diagnosis = "model-rejection"
    logger.info("verifier muse probe completed", extra={"diagnosis": diagnosis})
    return {
        "requested": model,
        "custom_agent": custom,
        "omitted_agent": omitted,
        "diagnosis": diagnosis,
    }


def verifier_audit_entries(audit: Any) -> list[dict[str, str]]:
    """Return privacy-safe verifier audit entries (logical identity only)."""
    entries: list[dict[str, str]] = []
    try:
        items = list(audit)
    except Exception:
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("agent", "")) != VERIFIER_AGENT_V2:
            continue
        entries.append(
            {
                "agent": str(item.get("agent", ""))[:128],
                "requested": str(item.get("requested", ""))[:128],
                "served": str(item.get("served", ""))[:128],
            }
        )
    return entries


def check_verifier_served_exact(
    audit: Any,
    *,
    expected: str = VERIFIER_EXPECTED_MODEL,
    forbidden: str = VERIFIER_FORBIDDEN_MODEL,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check requested == served == Muse for the verifier (fail-closed).

    Returns ``(passed, failed)`` check names using the exact live failure
    vocabulary. An absent served verifier is the specific Muse-access
    failure, never a generic Gate C failure. Any Space Bunny service for
    the verifier fails with the dedicated forbidden check.
    """
    passed: list[str] = []
    failed: list[str] = []
    entries = verifier_audit_entries(audit)
    if not entries:
        failed.append(MUSE_ACCESS_FAILURE)
        return tuple(passed), tuple(failed)
    for entry in entries:
        if entry["served"] == forbidden or entry["requested"] == forbidden:
            failed.append(SPACE_BUNNY_FORBIDDEN_FAILURE)
    if SPACE_BUNNY_FORBIDDEN_FAILURE in failed:
        return tuple(passed), tuple(failed)
    exact = [
        entry for entry in entries if entry["requested"] == expected and entry["served"] == expected
    ]
    if not exact:
        failed.append(SERVED_MISMATCH_FAILURE)
        return tuple(passed), tuple(failed)
    passed.append(REQUESTED_PIN_FAILURE)
    if all(entry["requested"] == expected and entry["served"] == expected for entry in entries):
        passed.append("live-verifier-muse-served-exact")
    else:
        failed.append(SERVED_MISMATCH_FAILURE)
    return tuple(passed), tuple(failed)


__all__ = [
    "MUSE_ACCESS_FAILURE",
    "REQUESTED_PIN_FAILURE",
    "SERVED_MISMATCH_FAILURE",
    "SPACE_BUNNY_FORBIDDEN_FAILURE",
    "VERIFIER_EXPECTED_MODEL",
    "VERIFIER_FORBIDDEN_MODEL",
    "check_verifier_served_exact",
    "probe_verifier_muse_access",
    "verifier_audit_entries",
]
