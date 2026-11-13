"""OpenCode failure taxonomy.

Failures are classified so callers can react deterministically:

- :class:`OpenCodeStartupError`: the local ``opencode serve`` process could
  not be started or never became healthy. Not retryable within this process;
  fix the environment (binary, port, workdir) and restart.
- :class:`OpenCodeNotReadyError`: an OpenCode operation was attempted before
  readiness was verified. Deterministic caller bug.
- :class:`OpenCodeTimeoutError`: a readiness probe or request exceeded its
  deadline. Treated as transient.
- :class:`OpenCodeRateLimitError`: HTTP/provider 429. Escalates to hosted-runner
  lifecycle recovery; it is never retried locally inside a worker process.
- :class:`OpenCodeTransientError`: network errors, HTTP 408/425/5xx and
  provider errors flagged retryable by OpenCode. Safe to retry with backoff.
- :class:`OpenCodeDeterministicError`: HTTP 4xx (other than 408/425/429),
  malformed requests and provider errors flagged non-retryable by OpenCode.
  Retrying without changing the request will not help.
- :class:`OpenCodeSessionNotFoundError`: the OpenCode session id is unknown
  (HTTP 404 or a deleted session). Recover by creating a fresh session.

Error messages never contain user prompt text or credentials; they carry only
kinds, HTTP status codes and session-id hints safe for structured logs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class OpenCodeError(Exception):
    """Base class for all OpenCode integration failures."""

    kind = "opencode-error"
    transient = False


class OpenCodeStartupError(OpenCodeError):
    """The local OpenCode runtime could not be started or readied."""

    kind = "startup"
    transient = False


class OpenCodeNotReadyError(OpenCodeError):
    """An operation was attempted before the runtime was ready."""

    kind = "not-ready"
    transient = False


class OpenCodeTimeoutError(OpenCodeError):
    """A probe or request exceeded its deadline (transient)."""

    kind = "timeout"
    transient = True


class OpenCodeTransientError(OpenCodeError):
    """A retryable failure (network, 5xx, retryable provider error)."""

    kind = "transient"
    transient = True


class OpenCodeRateLimitError(OpenCodeTransientError):
    """HTTP/provider 429: retire the current runner and retry on a fresh one."""

    kind = "rate-limit"
    transient = True


class OpenCodeDeterministicError(OpenCodeError):
    """A non-retryable failure (bad request, non-retryable provider error)."""

    kind = "deterministic"
    transient = False


class OpenCodeProviderAccessError(OpenCodeDeterministicError):
    """Provider/model access rejection where trying a configured fallback is valid."""

    kind = "provider-access"
    transient = False


class OpenCodeSessionNotFoundError(OpenCodeDeterministicError):
    """The referenced OpenCode session id does not exist."""

    kind = "session-not-found"
    transient = False


# Statuses that are retryable even though they live in the 4xx range
# (rate limiting / concurrency control).
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429})


def _short_session_hint(session_id: str) -> str:
    """Return a log-safe session hint (prefix only, ids are opaque)."""
    if not session_id:
        return "<none>"
    return f"{session_id[:12]}..." if len(session_id) > 12 else session_id


def classify_http_status(status: int, *, session_id: str = "") -> OpenCodeError:
    """Build the classified error for an HTTP failure status.

    Only the numeric status and a truncated session hint are embedded in the
    message so no response body, prompt text or credential can leak into logs.
    """
    hint = _short_session_hint(session_id)
    if status == 404:
        return OpenCodeSessionNotFoundError(f"opencode session not found: {hint}")
    if status == 429:
        return OpenCodeRateLimitError("opencode request rate-limited: http=429")
    if status >= 500 or status in _RETRYABLE_HTTP_STATUSES:
        return OpenCodeTransientError(f"opencode request failed transiently: http={status}")
    if status == 403:
        return OpenCodeProviderAccessError("opencode provider access rejected: http=403")
    return OpenCodeDeterministicError(f"opencode request rejected: http={status}")


def classify_provider_error(error: Mapping[str, Any] | None) -> OpenCodeError:
    """Classify an OpenCode assistant-message ``error`` payload.

    Two observed shapes are supported:

    - v1: ``{"name": ..., "data": {"statusCode": int,
      "isRetryable": bool, ...}}``;
    - v2: ``{"type": ..., "message": ..., "status": int,
      "retryAfterMs": int, ...}``.

    The human-readable provider message is deliberately *not* propagated:
    it may echo request content, so only the status code and retryability
    flags drive the classification.
    """
    if not isinstance(error, Mapping):
        return OpenCodeDeterministicError("opencode returned an unsuccessful result")
    retryable = False
    status_code = 0
    retry_after = error.get("retryAfterMs")
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
        retryable = retry_after >= 0
    data = error.get("data")
    if isinstance(data, Mapping):
        retryable = retryable or data.get("isRetryable") is True
        raw_status = data.get("statusCode")
        status_code = _coerce_status(raw_status)
    raw_status = error.get("status")
    if status_code == 0:
        status_code = _coerce_status(raw_status)
    if status_code == 429:
        return OpenCodeRateLimitError("opencode provider rate-limited: http=429")
    if retryable or (status_code >= 500) or status_code in _RETRYABLE_HTTP_STATUSES:
        return OpenCodeTransientError(
            f"opencode provider error is retryable: http={status_code or 'unknown'}"
        )
    if status_code == 403:
        # A model/provider access rejection can be SKU-specific (for example
        # an unavailable/free-tier route) while a separately configured
        # fallback model remains usable. The request shape itself is valid,
        # so let the model adapter try that fallback exactly once.
        return OpenCodeProviderAccessError("opencode provider access rejected: http=403")
    return OpenCodeDeterministicError(
        f"opencode provider error is deterministic: http={status_code or 'unknown'}"
    )


def classify_structured_error(error: Mapping[str, Any] | None) -> OpenCodeError:
    """Classify an OpenCode v2 ``Session.StructuredError`` payload.

    v2 carries provider failures as ``{"type", "message", "status",
    "retryAfterMs", "response"}`` on completed sessions and idle messages.
    Only the numeric status and retry hint drive the taxonomy; the provider
    message/body are never propagated because they may echo request content.
    """
    if not isinstance(error, Mapping):
        return OpenCodeDeterministicError("opencode session failed without an error payload")
    status_code = _coerce_status(error.get("status"))
    retry_after = error.get("retryAfterMs")
    retryable = isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool)
    if status_code == 429:
        return OpenCodeRateLimitError("opencode provider rate-limited: http=429")
    if retryable or status_code >= 500 or status_code in _RETRYABLE_HTTP_STATUSES:
        return OpenCodeTransientError(
            f"opencode session failed transiently: http={status_code or 'unknown'}"
        )
    if status_code == 403:
        return OpenCodeProviderAccessError("opencode provider access rejected: http=403")
    return OpenCodeDeterministicError(
        f"opencode session failed deterministically: http={status_code or 'unknown'}"
    )


def _coerce_status(value: object) -> int:
    """Return a bounded integer HTTP status, never a bool/other value."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0
