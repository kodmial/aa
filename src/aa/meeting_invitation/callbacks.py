"""Opaque length-bounded signed callback tokens for the meeting wizard.

Bot API callback_data is limited to 1..64 bytes. Tokens carry no PII,
no conversation text, and no precise identifiers: only version, action,
flow sequence, generation, message binding, and expiry, authenticated
with HMAC-SHA256 truncated to a bounded suffix.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from aa.meeting_invitation.fsm import CallbackAction

CALLBACK_VERSION = 1
CALLBACK_TTL_SECONDS = 24 * 3600
MAX_CALLBACK_BYTES = 64

_ACTION_TO_CODE: dict[str, int] = {
    CallbackAction.ACCEPT.value: 1,
    CallbackAction.DECLINE.value: 2,
    CallbackAction.ONLINE.value: 3,
    CallbackAction.IN_PERSON.value: 4,
    CallbackAction.BACK.value: 5,
    CallbackAction.CANCEL.value: 6,
    CallbackAction.MORE.value: 7,
    CallbackAction.OTHER_CITY.value: 8,
    CallbackAction.CHOOSE_PLACE.value: 9,
    CallbackAction.NEW_SEARCH.value: 10,
}
_CODE_TO_ACTION: dict[int, str] = {code: action for action, code in _ACTION_TO_CODE.items()}


class StaleCallbackError(ValueError):
    """A callback token is expired, foreign, replayed, or unauthenticated."""


@dataclass(frozen=True)
class CallbackData:
    """Validated callback identity (never carries user text)."""

    action: str
    flow_seq: int
    generation: int
    message_id: int
    place_index: int = 0


class CallbackCodec:
    """Issue and validate opaque wizard callback tokens."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = bytes(secret) if secret else secrets.token_bytes(32)

    @property
    def secret(self) -> bytes:
        """Return the codec secret (tests only; never logged)."""
        return bytes(self._secret)

    def encode(
        self,
        *,
        action: str,
        flow_seq: int,
        generation: int,
        message_id: int,
        place_index: int = 0,
        nonce: int | None = None,
        now: float | None = None,
        ttl_seconds: int = CALLBACK_TTL_SECONDS,
    ) -> str:
        """Create one bounded opaque token for a wizard button."""
        if action not in _ACTION_TO_CODE:
            raise ValueError(f"unknown callback action: {action}")
        moment = int(now if now is not None else time.time())
        expiry = moment + max(1, int(ttl_seconds))
        salt = int(nonce) % 256 if nonce is not None else secrets.randbits(8)
        payload = (
            int(CALLBACK_VERSION).to_bytes(1, "big")
            + int(_ACTION_TO_CODE[action]).to_bytes(1, "big")
            + int(flow_seq).to_bytes(4, "big", signed=False)
            + int(generation).to_bytes(4, "big", signed=False)
            + int(message_id).to_bytes(4, "big", signed=False)
            + int(place_index).to_bytes(1, "big", signed=False)
            + int(salt).to_bytes(1, "big", signed=False)
            + int(expiry).to_bytes(4, "big", signed=False)
        )
        body = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        mac = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).hexdigest()[:16]
        token = f"{body}.{mac}"
        if len(token.encode("utf-8")) > MAX_CALLBACK_BYTES:
            raise ValueError("callback token exceeds the Bot API bound")
        return token

    def decode(
        self,
        token: str,
        *,
        now: float | None = None,
    ) -> CallbackData:
        """Validate one token and return its identity or raise stale."""
        raw = str(token or "")
        if not raw or len(raw.encode("utf-8")) > MAX_CALLBACK_BYTES:
            raise StaleCallbackError("callback token has an invalid length")
        try:
            body, mac = raw.split(".", 1)
        except ValueError as exc:
            raise StaleCallbackError("callback token is malformed") from exc
        expected = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(expected, mac):
            raise StaleCallbackError("callback token signature mismatch")
        padded = body + "=" * (-len(body) % 4)
        try:
            payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise StaleCallbackError("callback token payload is malformed") from exc
        if len(payload) != 20:
            raise StaleCallbackError("callback token payload has an invalid shape")
        version = int.from_bytes(payload[0:1], "big")
        if version != CALLBACK_VERSION:
            raise StaleCallbackError("callback token version is stale")
        code = int.from_bytes(payload[1:2], "big")
        action = _CODE_TO_ACTION.get(code)
        if action is None:
            raise StaleCallbackError("callback token action is unknown")
        flow_seq = int.from_bytes(payload[2:6], "big", signed=False)
        generation = int.from_bytes(payload[6:10], "big", signed=False)
        message_id = int.from_bytes(payload[10:14], "big", signed=False)
        place_index = int.from_bytes(payload[14:15], "big", signed=False)
        expiry = int.from_bytes(payload[16:20], "big", signed=False)
        moment = int(now if now is not None else time.time())
        if expiry < moment:
            raise StaleCallbackError("callback token expired")
        return CallbackData(
            action=action,
            flow_seq=int(flow_seq),
            generation=int(generation),
            message_id=int(message_id),
            place_index=int(place_index),
        )


__all__ = [
    "CALLBACK_TTL_SECONDS",
    "CALLBACK_VERSION",
    "MAX_CALLBACK_BYTES",
    "CallbackCodec",
    "CallbackData",
    "StaleCallbackError",
]
