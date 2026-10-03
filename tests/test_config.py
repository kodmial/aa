"""Configuration contract tests."""

from __future__ import annotations

import pathlib

from aa.config import Settings


def test_from_env_defaults_without_secrets() -> None:
    settings = Settings.from_env({})
    assert settings.telegram_bot_token == ""
    assert settings.opencode_base_url == "http://127.0.0.1:4096"
    assert settings.opencode_command == "opencode"
    assert settings.bot_session_duration_seconds == 0.0
    assert settings.aa_corpus_path == "./corpus"
    assert settings.aa_corpus_version == "local"
    assert not settings.has_bot_token
    settings.validate()


def test_from_env_reads_reserved_names() -> None:
    settings = Settings.from_env(
        {
            "TELEGRAM_BOT_TOKEN": "123456:ABCDEF-test-token",
            "OPENCODE_BASE_URL": "http://127.0.0.1:9999",
            "OPENCODE_COMMAND": "opencode-test",
            "OPENCODE_WORKDIR": "/tmp/work",
            "BOT_SESSION_DURATION_SECONDS": "30",
            "AA_CORPUS_PATH": "./corpus-test",
            "AA_CORPUS_VERSION": "v1",
            "OPENCODE_MODEL": "test-model",
            "OPENCODE_CONTEXT_LIMIT_TOKENS": "1000",
            "OPENCODE_MAX_OUTPUT_TOKENS": "200",
            "LOG_LEVEL": "debug",
        }
    )
    assert settings.telegram_bot_token == "123456:ABCDEF-test-token"
    assert settings.opencode_base_url == "http://127.0.0.1:9999"
    assert settings.opencode_command == "opencode-test"
    assert settings.opencode_workdir == "/tmp/work"
    assert settings.bot_session_duration_seconds == 30.0
    assert settings.aa_corpus_path == "./corpus-test"
    assert settings.aa_corpus_version == "v1"
    assert settings.opencode_model == "test-model"
    assert settings.opencode_context_limit_tokens == 1000
    assert settings.opencode_max_output_tokens == 200
    assert settings.log_level == "DEBUG"
    assert settings.has_bot_token
    settings.validate(require_bot_token=True)


def test_reserved_names_cover_contract() -> None:
    reserved = set(Settings.RESERVED_ENV_NAMES)
    assert "TELEGRAM_BOT_TOKEN" in reserved
    assert "OPENCODE_BASE_URL" in reserved
    assert "OPENCODE_COMMAND" in reserved
    assert "OPENCODE_WORKDIR" in reserved
    assert "BOT_SESSION_DURATION_SECONDS" in reserved
    assert "AA_CORPUS_PATH" in reserved
    assert "AA_CORPUS_VERSION" in reserved
    assert "OPENCODE_MODEL" in reserved
    assert "OPENCODE_CONTEXT_LIMIT_TOKENS" in reserved
    assert "OPENCODE_MAX_OUTPUT_TOKENS" in reserved


def test_worker_must_not_own_zen_or_second_llm_client() -> None:
    reserved = set(Settings.RESERVED_ENV_NAMES)
    forbidden_prefixes = ("ZEN_", "ANTHROPIC_", "OPENAI_", "GOOGLE_", "LLM_API_KEY")
    for name in reserved:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), f"forbidden credential setting: {name}"
    fields = set(Settings.__dataclass_fields__)
    for field_name in fields:
        lowered = field_name.lower()
        assert "zen" not in lowered, f"forbidden zen field: {field_name}"
    # No second LLM client vendored into the runtime dependencies.
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8").lower()
    for vendor in ("openai", "anthropic", "google-generativeai", "httpx", "aiohttp"):
        # Dev tooling aside, runtime project dependencies must stay empty.
        assert vendor not in text.split("[project.optional-dependencies]")[0], vendor


def test_safe_dict_redacts_token() -> None:
    settings = Settings.from_env({"TELEGRAM_BOT_TOKEN": "123456:ABCDEF-secret"})
    safe = settings.to_safe_dict()
    assert safe["telegram_bot_token"] != "123456:ABCDEF-secret"
    assert "123456" not in str(safe["telegram_bot_token"])
    # Non-secret pointers remain visible for operability.
    assert safe["opencode_base_url"] == settings.opencode_base_url
    assert safe["aa_corpus_version"] == settings.aa_corpus_version
