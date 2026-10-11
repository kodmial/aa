"""Honest Telegram startup/delivery readiness probe (issue #332).

The product-contract transport lane historically reported INCOMPLETE with
``real-telegram-typing-stream-not-dialed-in-qualification`` even when a bot
token was configured: it constructed a transport but never dialled the Bot
API, so Gate B transport acceptance could never reach PASS with verifiable
live proof. This module owns the bounded, privacy-safe classification:

- ``environment-secrets``: no bot token (or no live peer for delivery);
- ``telegram-auth``: invalid/revoked token (401);
- ``telegram-getme``: ``getMe`` failed or returned an unexpected identity;
- ``telegram-bootstrap``: ``deleteWebhook``/``setMyCommands`` failed;
- ``polling-webhook-conflict``: HTTP 409, another poller/webhook owns updates;
- ``transport-network-failure``: unreachable/timeout/OSError;
- ``transport-rate-limited``: HTTP 429;
- ``receipt-certificate-mismatch``: confirmed bytes do not cover the certificate;
- ``true-message-delivery-failure``: send failed after bounded retries;
- ``qualifier-artifact-collector``: evidence collection itself failed.

Logs and outcomes carry only categories, counts and lengths: never the bot
token, chat identifiers, message bodies, prompts, or book passages.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger("aa.telegram.startup_probe")

# Live delivery peer: a Telegram chat id owned by the operator used only to
# prove real end-to-end delivery. Absent by default; without it the lane
# stays fail-closed INCOMPLETE with a typed external dependency instead of
# inventing verified live evidence.
LIVE_PEER_ENV_VAR = "AA_LIVE_TELEGRAM_CHAT_ID"
# Explicit opt-in for actually sending a live probe message to the peer.
# Qualification never sends without this flag, even when a peer is set.
LIVE_SEND_ENV_VAR = "AA_LIVE_TELEGRAM_PROBE_SEND"
# Isolated Gate D test identity (issue #340): a dedicated bot token/bot id
# used only for qualification probes. When present, probes use this identity
# and never touch the deployed production poller. Production-token use
# additionally requires AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE=1.
TEST_BOT_TOKEN_ENV_VAR = "AA_TEST_TELEGRAM_BOT_TOKEN"
TEST_BOT_ID_ENV_VAR = "AA_TEST_TELEGRAM_BOT_ID"
TEST_PEER_ENV_VAR = "AA_TEST_TELEGRAM_PEER_CHAT_ID"
TEST_SEND_ENV_VAR = "AA_TEST_TELEGRAM_PROBE_SEND"
PRODUCTION_LEASE_ENV_VAR = "AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE"

EXTERNAL_PEER_DEPENDENCY = "EXTERNAL_TELEGRAM_DELIVERY_PEER_UNAVAILABLE"

StartupStatus = Literal["ready", "blocked", "failed"]


@dataclass(frozen=True)
class StartupOutcome:
    """Typed startup probe result (privacy-safe)."""

    status: StartupStatus
    category: str
    bot_id_present: bool = False
    username_len: int = 0
    polling_live: bool = False
    bootstrap_verified: bool = False


def classify_http_status(code: int | None, description: str = "") -> str:
    """Map a Telegram HTTP status to a bounded probe category."""
    lowered = (description or "").lower()
    if code == 401 or "unauthorized" in lowered:
        return "telegram-auth"
    if code == 409 or "conflict" in lowered:
        return "polling-webhook-conflict"
    if code == 429 or "too many requests" in lowered or "rate limit" in lowered:
        return "transport-rate-limited"
    return "transport-network-failure"


def categorize_startup_exception(exc: BaseException) -> str:
    """Map a startup exception to a bounded privacy-safe category."""
    from aa.telegram.transport import (
        TelegramApiError,
        TelegramAuthError,
        TelegramConflictError,
        TelegramRateLimitedError,
    )

    if isinstance(exc, TelegramAuthError):
        return "telegram-auth"
    if isinstance(exc, TelegramConflictError):
        return "polling-webhook-conflict"
    if isinstance(exc, TelegramRateLimitedError):
        return "transport-rate-limited"
    if isinstance(exc, TelegramApiError):
        message = str(exc).lower()
        if "getme" in message:
            return "telegram-getme"
        if "network" in message or "unreachable" in message or "timeout" in message:
            return "transport-network-failure"
        return "telegram-bootstrap"
    if isinstance(exc, (TimeoutError, OSError)):
        return "transport-network-failure"
    if isinstance(exc, ValueError):
        message = str(exc).lower()
        if "token" in message:
            return "environment-secrets"
        return "app-startup"
    return "app-startup"


def live_delivery_peer_from_env() -> int | None:
    """Return the configured live delivery peer chat id, if any.

    Only digits (optionally leading ``-``) are accepted; anything else is
    treated as absent so a malformed value can never become a live send
    target. The value itself is never logged.
    """
    raw = (os.environ.get(LIVE_PEER_ENV_VAR, "") or "").strip()
    if not raw:
        return None
    candidate = raw[1:] if raw.startswith("-") else raw
    if not candidate.isdigit():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def live_send_authorized() -> bool:
    """Whether a live probe send to the peer is explicitly authorized."""
    return (os.environ.get(LIVE_SEND_ENV_VAR, "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or (os.environ.get(TEST_SEND_ENV_VAR, "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def isolated_test_bot_token() -> str:
    """Return the dedicated Gate D test bot token, if configured."""
    return (os.environ.get(TEST_BOT_TOKEN_ENV_VAR, "") or "").strip()


def isolated_test_bot_id() -> int | None:
    """Return the dedicated Gate D test bot id, if configured."""
    raw = (os.environ.get(TEST_BOT_ID_ENV_VAR, "") or "").strip()
    if not raw or not raw.isdigit():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def isolated_test_peer_from_env() -> int | None:
    """Return the consented isolated test peer chat id, if configured.

    Only digits (optionally leading ``-``) are accepted. Sending to an
    arbitrary chat id taken from environment logs or unrelated settings is
    forbidden: only the dedicated test-peer variable plus explicit opt-in
    authorizes a real egress attempt.
    """
    for var in (TEST_PEER_ENV_VAR, LIVE_PEER_ENV_VAR):
        raw = (os.environ.get(var, "") or "").strip()
        if not raw:
            continue
        candidate = raw[1:] if raw.startswith("-") else raw
        if not candidate.isdigit():
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def isolated_test_send_authorized() -> bool:
    """Whether a real egress attempt to the isolated test peer is allowed."""
    return live_send_authorized()


def production_lease_authorized() -> bool:
    """Whether the operator explicitly leased the production poller identity."""
    return (os.environ.get(PRODUCTION_LEASE_ENV_VAR, "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def resolve_probe_token(*, production_token: str = "") -> tuple[str, str]:
    """Resolve which bot identity a Gate D probe may use (fail-closed).

    Returns ``(token, identity_kind)`` where ``identity_kind`` is
    ``"isolated-test-bot"`` or ``"production-lease"``. Raises
    :class:`ValueError` with a typed blocker when no safely usable identity
    exists: an isolated test token is preferred; the production token is
    used only with an explicit exclusive lease flag.
    """
    test_token = isolated_test_bot_token()
    if test_token:
        return test_token, "isolated-test-bot"
    cleaned = (production_token or "").strip()
    if cleaned and production_lease_authorized():
        return cleaned, "production-lease"
    raise ValueError("EXTERNAL_TEST_BOT_UNAVAILABLE")


def verify_bot_identity_scope(*, bot_info: dict[str, Any], expected_bot_id: int | None) -> bool:
    """Verify a real ``getMe`` identity matches the isolated test scope.

    With no expected id configured the scope check passes (the caller still
    owns poller-conflict/lease handling); with an expected id any mismatch
    fails closed so a production identity is never mistaken for the test bot.
    """
    if expected_bot_id is None:
        return True
    try:
        actual = int((bot_info or {}).get("id", 0) or 0)
    except (TypeError, ValueError):
        return False
    return actual == int(expected_bot_id)


def classify_probe_http_error(*, code: int | None, mandatory: bool) -> str:
    """Classify a real 403/409/network outcome for one attempted operation.

    Mandatory #7 operations (getMe/bootstrap/poller) keep D FAIL/BLOCKED on
    403/409/network failures; optional peer-send attempts record a separate
    optional-send category and must never downgrade mandatory readiness.
    """
    if code == 409:
        return "EXTERNAL_POLLER_CONFLICT"
    if code == 403:
        return "EXTERNAL_MANDATORY_FORBIDDEN" if mandatory else "EXTERNAL_OPTIONAL_FORBIDDEN"
    if code == 429:
        if mandatory:
            return "EXTERNAL_MANDATORY_RATE_LIMITED"
        return "EXTERNAL_OPTIONAL_RATE_LIMITED"
    if mandatory:
        return "EXTERNAL_MANDATORY_NETWORK_FAILURE"
    return "EXTERNAL_OPTIONAL_NETWORK_FAILURE"


def validate_trusted_ready_marker(
    *,
    marker_sha: str,
    marker_run_id: str,
    marker_timestamp_s: float,
    expected_sha: str,
    expected_run_id: str,
    now_s: float,
    poll_task_live: bool,
) -> tuple[bool, str]:
    """Validate a read-only deployed-runtime READY marker (fail-closed).

    A marker alone never proves a network send or a currently live poller:
    it must carry an independently trusted origin (exact run id/SHA match
    here), the exact runtime SHA/identity, a freshness window, and direct
    liveness evidence for the active poll task. Stale, unsigned/self-declared
    or copied markers fail closed.
    """
    from aa.qualification.gate_cd_boundary import TRUSTED_MARKER_FRESHNESS_S

    if (marker_sha or "").strip().lower() != (expected_sha or "").strip().lower():
        return False, "marker-sha-mismatch"
    if (marker_run_id or "").strip() != (expected_run_id or "").strip():
        return False, "marker-run-mismatch"
    try:
        age = float(now_s) - float(marker_timestamp_s)
    except (TypeError, ValueError):
        return False, "marker-timestamp-invalid"
    if not (0 <= age <= TRUSTED_MARKER_FRESHNESS_S):
        return False, "marker-stale"
    if not poll_task_live:
        return False, "marker-without-live-poller"
    return True, "marker-trusted"


def confirm_full_delivery(
    certificate_text: str, receipts: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Require confirmed full-certificate delivery (no phantom receipts).

    Delegates to the authoritative #304/#312 receipt check: only
    ``confirmed`` intervals that jointly cover the whole certified text
    count. Split/partial/unknown/failed receipts never prove delivery.
    """
    from aa.qualification.book_fidelity_307 import check_delivery_receipts

    try:
        return check_delivery_receipts(
            certificate_text=certificate_text, receipts=list(receipts or [])
        )
    except Exception:
        return False, "receipt-certificate-mismatch"


async def probe_telegram_startup(
    token: str,
    *,
    api: Any | None = None,
    poll_timeout_seconds: int = 0,
    start_polling: bool = True,
) -> StartupOutcome:
    """Prove real Telegram startup through the production transport.

    Runs the exact production bootstrap (``getMe`` -> ``deleteWebhook`` ->
    ``setMyCommands``) via :class:`PollingTelegramTransport`. With
    ``start_polling=True`` (default) a live poll task is additionally
    started and verified, then stopped cleanly with no leaked poll task.
    With ``start_polling=False`` only the bootstrap handshake runs, so the
    probe never issues ``getUpdates`` and cannot steal updates from a
    concurrently running production poller; live-poll evidence then stays
    owned by Gate D. Returns a typed outcome; raises nothing for transport
    failures (they become ``failed``/``blocked`` outcomes instead), except
    programming errors.
    """
    from aa.telegram.transport import PollingTelegramTransport

    cleaned = (token or "").strip()
    if not cleaned:
        logger.info("telegram startup probe blocked: no token")
        return StartupOutcome(status="blocked", category="environment-secrets")
    transport = PollingTelegramTransport(
        token=cleaned,
        api=api,
        poll_timeout_seconds=poll_timeout_seconds,
        retry_base_delay_seconds=0.05,
        retry_max_delay_seconds=0.5,
        max_bootstrap_retries=2,
    )
    if not start_polling:
        try:
            await transport._bootstrap()
        except Exception as exc:
            category = categorize_startup_exception(exc)
            logger.warning("telegram startup probe failed", extra={"category": category})
            status: StartupStatus = (
                "blocked" if category == "polling-webhook-conflict" else "failed"
            )
            return StartupOutcome(status=status, category=category)
        bot_info = transport.bot_info or {}
        bot_id_present = bool(bot_info.get("id"))
        username = bot_info.get("username", "")
        username_len = len(username) if isinstance(username, str) else 0
        if not bot_id_present:
            logger.warning("telegram startup probe failed", extra={"category": "telegram-getme"})
            return StartupOutcome(status="failed", category="telegram-getme")
        logger.info("telegram startup probe bootstrap verified")
        return StartupOutcome(
            status="ready",
            category="ready",
            bot_id_present=True,
            username_len=username_len,
            polling_live=False,
            bootstrap_verified=True,
        )
    try:
        await transport.start()
    except Exception as exc:
        category = categorize_startup_exception(exc)
        logger.warning("telegram startup probe failed", extra={"category": category})
        try:
            await transport.stop()
        except Exception:
            pass
        status = "blocked" if category == "polling-webhook-conflict" else "failed"
        return StartupOutcome(status=status, category=category)
    try:
        bot_info = transport.bot_info or {}
        bot_id_present = bool(bot_info.get("id"))
        username = bot_info.get("username", "")
        username_len = len(username) if isinstance(username, str) else 0
        poll_task = getattr(transport, "_poll_task", None)
        polling_live = bool(
            transport.running and poll_task is not None and not bool(poll_task.done())
        )
        if not bot_id_present:
            logger.warning("telegram startup probe failed", extra={"category": "telegram-getme"})
            return StartupOutcome(status="failed", category="telegram-getme")
        if not polling_live:
            logger.warning(
                "telegram startup probe failed", extra={"category": "telegram-bootstrap"}
            )
            return StartupOutcome(status="failed", category="telegram-bootstrap")
        logger.info("telegram startup probe ready")
        return StartupOutcome(
            status="ready",
            category="ready",
            bot_id_present=True,
            username_len=username_len,
            polling_live=True,
            bootstrap_verified=True,
        )
    finally:
        try:
            await transport.stop()
        except Exception:
            pass
