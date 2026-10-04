"""Environment-backed configuration for the AA Telegram worker.

The worker intentionally owns only transport/session/corpus settings plus
pointer settings owned by the OpenCode runtime (endpoint, model and limit
hints). It must not own Zen credentials and must not implement a second
LLM client; therefore no ``ZEN_*``, ``ANTHROPIC_*`` or ``OPENAI_*`` style
settings may be added here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_AA_AGENT = "aa"
DEFAULT_PRIMARY_MODEL = "opencode/muse-spark-1.3-contributor-free"
DEFAULT_FALLBACK_MODEL = "opencode/space-bunny-free"

# Conservative concurrency defaults for the single local ``opencode serve``
# process per worker (issue #5). Qualification determines the safe value
# against the pinned OpenCode/provider/runtime; these defaults stay small
# so the worker cannot overload OpenCode/provider/runtime resources.
DEFAULT_MAX_CONCURRENT_TURNS = 4
DEFAULT_PER_CHAT_QUEUE_SIZE = 8


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved worker configuration.

    All values come from the environment with safe local defaults so the
    worker can boot (and tests can run) without secrets or network access.
    """

    # Telegram transport.
    telegram_bot_token: str = ""

    # OpenCode local endpoint/process configuration (pointers only; the
    # OpenCode runtime owns credentials, model execution and limits).
    opencode_base_url: str = "http://127.0.0.1:4096"
    opencode_command: str = "opencode"
    opencode_workdir: str = "."

    # Requested bot session duration in seconds. ``0`` means "run until
    # stopped" (used by tests); positive values set a shutdown deadline.
    bot_session_duration_seconds: float = 0.0

    # AA corpus location/version pointer.
    aa_corpus_path: str = "./corpus"
    aa_corpus_version: str = "local"

    # Model/context/output limits where owned by the OpenCode runtime.
    # These are hints passed through to OpenCode, not a second LLM client.
    # ``opencode_max_output_tokens`` is a bounded prompt-advertised
    # generation budget (issue #83; default 256 when unset): the pinned
    # OpenCode message API exposes no per-message max-tokens field, so the
    # deterministic character validator stays authoritative.
    opencode_agent: str = DEFAULT_AA_AGENT
    opencode_model: str = DEFAULT_PRIMARY_MODEL
    opencode_fallback_model: str = DEFAULT_FALLBACK_MODEL
    opencode_context_limit_tokens: int = 0
    opencode_max_output_tokens: int = 0

    # Concurrency architecture (issue #5): one authoritative Telegram poller
    # and one local ``opencode serve`` process per worker. Different chats
    # may execute concurrently up to ``max_concurrent_turns``; turns for the
    # same chat stay strict FIFO with at most one active turn. Each per-chat
    # pending queue is bounded by ``per_chat_queue_size`` so the worker
    # backpressures safely instead of growing memory without bound.
    max_concurrent_turns: int = DEFAULT_MAX_CONCURRENT_TURNS
    per_chat_queue_size: int = DEFAULT_PER_CHAT_QUEUE_SIZE

    # Logging.
    log_level: str = "INFO"

    # Reserved names (authoritative contract). Any rename must stay in sync
    # with ``.env.example`` and the README.
    RESERVED_ENV_NAMES: tuple[str, ...] = field(
        default=(
            "TELEGRAM_BOT_TOKEN",
            "OPENCODE_BASE_URL",
            "OPENCODE_COMMAND",
            "OPENCODE_WORKDIR",
            "BOT_SESSION_DURATION_SECONDS",
            "AA_CORPUS_PATH",
            "AA_CORPUS_VERSION",
            "OPENCODE_AGENT",
            "OPENCODE_MODEL",
            "OPENCODE_FALLBACK_MODEL",
            "OPENCODE_CONTEXT_LIMIT_TOKENS",
            "OPENCODE_MAX_OUTPUT_TOKENS",
            "MAX_CONCURRENT_TURNS",
            "PER_CHAT_QUEUE_SIZE",
            "LOG_LEVEL",
        ),
        compare=False,
        repr=False,
    )

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from the process environment (or a mapping)."""
        source = os.environ if environ is None else environ
        if environ is None:
            return cls(
                telegram_bot_token=_get_str("TELEGRAM_BOT_TOKEN", ""),
                opencode_base_url=_get_str("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
                opencode_command=_get_str("OPENCODE_COMMAND", "opencode"),
                opencode_workdir=_get_str("OPENCODE_WORKDIR", "."),
                bot_session_duration_seconds=_get_float("BOT_SESSION_DURATION_SECONDS", 0.0),
                aa_corpus_path=_get_str("AA_CORPUS_PATH", "./corpus"),
                aa_corpus_version=_get_str("AA_CORPUS_VERSION", "local"),
                opencode_agent=_get_str("OPENCODE_AGENT", DEFAULT_AA_AGENT),
                opencode_model=_get_str("OPENCODE_MODEL", DEFAULT_PRIMARY_MODEL),
                opencode_fallback_model=_get_str("OPENCODE_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
                opencode_context_limit_tokens=int(
                    source.get("OPENCODE_CONTEXT_LIMIT_TOKENS", "") or 0
                ),
                opencode_max_output_tokens=int(source.get("OPENCODE_MAX_OUTPUT_TOKENS", "") or 0),
                max_concurrent_turns=int(
                    source.get("MAX_CONCURRENT_TURNS", "") or DEFAULT_MAX_CONCURRENT_TURNS
                ),
                per_chat_queue_size=int(
                    source.get("PER_CHAT_QUEUE_SIZE", "") or DEFAULT_PER_CHAT_QUEUE_SIZE
                ),
                log_level=_get_str("LOG_LEVEL", "INFO").upper(),
            )
        return cls(
            telegram_bot_token=source.get("TELEGRAM_BOT_TOKEN", ""),
            opencode_base_url=source.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
            opencode_command=source.get("OPENCODE_COMMAND", "opencode"),
            opencode_workdir=source.get("OPENCODE_WORKDIR", "."),
            bot_session_duration_seconds=float(
                source.get("BOT_SESSION_DURATION_SECONDS", "") or 0.0
            ),
            aa_corpus_path=source.get("AA_CORPUS_PATH", "./corpus"),
            aa_corpus_version=source.get("AA_CORPUS_VERSION", "local"),
            opencode_agent=source.get("OPENCODE_AGENT", DEFAULT_AA_AGENT),
            opencode_model=source.get("OPENCODE_MODEL", DEFAULT_PRIMARY_MODEL),
            opencode_fallback_model=source.get("OPENCODE_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
            opencode_context_limit_tokens=int(source.get("OPENCODE_CONTEXT_LIMIT_TOKENS", "") or 0),
            opencode_max_output_tokens=int(source.get("OPENCODE_MAX_OUTPUT_TOKENS", "") or 0),
            max_concurrent_turns=int(
                source.get("MAX_CONCURRENT_TURNS", "") or DEFAULT_MAX_CONCURRENT_TURNS
            ),
            per_chat_queue_size=int(
                source.get("PER_CHAT_QUEUE_SIZE", "") or DEFAULT_PER_CHAT_QUEUE_SIZE
            ),
            log_level=source.get("LOG_LEVEL", "INFO").upper(),
        )

    @property
    def has_bot_token(self) -> bool:
        """Whether a Telegram bot token is configured."""
        return bool(self.telegram_bot_token)

    def validate(self, *, require_bot_token: bool = False) -> None:
        """Validate settings, raising ``ValueError`` on misuse."""
        from aa.control.campaign import RUNTIME_SECONDS

        if self.bot_session_duration_seconds < 0:
            raise ValueError("BOT_SESSION_DURATION_SECONDS must be >= 0")
        if self.bot_session_duration_seconds > float(RUNTIME_SECONDS):
            raise ValueError(
                f"BOT_SESSION_DURATION_SECONDS must be <= {RUNTIME_SECONDS} (5h campaign max)"
            )
        if not self.opencode_agent.strip():
            raise ValueError("OPENCODE_AGENT must not be empty")
        if not self.opencode_model.strip():
            raise ValueError("OPENCODE_MODEL must not be empty")
        if not self.opencode_fallback_model.strip():
            raise ValueError("OPENCODE_FALLBACK_MODEL must not be empty")
        if self.opencode_context_limit_tokens < 0:
            raise ValueError("OPENCODE_CONTEXT_LIMIT_TOKENS must be >= 0")
        if self.opencode_max_output_tokens < 0:
            raise ValueError("OPENCODE_MAX_OUTPUT_TOKENS must be >= 0")
        if self.max_concurrent_turns <= 0:
            raise ValueError("MAX_CONCURRENT_TURNS must be > 0")
        if self.per_chat_queue_size <= 0:
            raise ValueError("PER_CHAT_QUEUE_SIZE must be > 0")
        if require_bot_token and not self.has_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required but missing or empty")

    def to_safe_dict(self) -> dict[str, object]:
        """Return a log-safe snapshot with secrets redacted."""
        from aa.logging import redact_secret

        return {
            "telegram_bot_token": redact_secret(self.telegram_bot_token),
            "opencode_base_url": self.opencode_base_url,
            "opencode_command": self.opencode_command,
            "opencode_workdir": self.opencode_workdir,
            "bot_session_duration_seconds": self.bot_session_duration_seconds,
            "aa_corpus_path": self.aa_corpus_path,
            "aa_corpus_version": self.aa_corpus_version,
            "opencode_agent": self.opencode_agent,
            "opencode_model": self.opencode_model,
            "opencode_fallback_model": self.opencode_fallback_model,
            "opencode_context_limit_tokens": self.opencode_context_limit_tokens,
            "opencode_max_output_tokens": self.opencode_max_output_tokens,
            "max_concurrent_turns": self.max_concurrent_turns,
            "per_chat_queue_size": self.per_chat_queue_size,
            "log_level": self.log_level,
        }
