"""OpenCode session/runtime integration boundary (local only, no LLM client)."""

from __future__ import annotations

from aa.opencode.client import (
    ChatMessage,
    FakeOpenCodeClient,
    HealthInfo,
    HttpOpenCodeClient,
    OpenCodeClient,
    SessionInfo,
    model_payload,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeError,
    OpenCodeNotReadyError,
    OpenCodeSessionNotFoundError,
    OpenCodeStartupError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
    classify_http_status,
    classify_provider_error,
)
from aa.opencode.runtime import (
    LocalOpenCodeRuntime,
    OpenCodeConfig,
    OpenCodeRuntime,
    StubOpenCodeRuntime,
    split_base_url,
)

__all__ = [
    "ChatMessage",
    "FakeOpenCodeClient",
    "HealthInfo",
    "HttpOpenCodeClient",
    "LocalOpenCodeRuntime",
    "OpenCodeClient",
    "OpenCodeConfig",
    "OpenCodeDeterministicError",
    "OpenCodeError",
    "OpenCodeNotReadyError",
    "OpenCodeRuntime",
    "OpenCodeSessionNotFoundError",
    "OpenCodeStartupError",
    "OpenCodeTimeoutError",
    "OpenCodeTransientError",
    "SessionInfo",
    "StubOpenCodeRuntime",
    "classify_http_status",
    "classify_provider_error",
    "model_payload",
    "split_base_url",
]
