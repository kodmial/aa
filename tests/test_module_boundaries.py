"""Module boundary tests (no vendor coupling)."""

from __future__ import annotations

import pathlib


def _src_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "src" / "aa"


def test_module_boundaries_exist() -> None:
    root = _src_root()
    expected = [
        "config.py",
        "logging.py",
        "app.py",
        "__main__.py",
        "telegram/transport.py",
        "opencode/runtime.py",
        "corpus/context.py",
        "sessions/coordinator.py",
        "safety/router.py",
        "control/runtime_control.py",
    ]
    for relative in expected:
        assert (root / relative).exists(), f"missing module boundary: {relative}"


def test_no_vendor_coupling_in_runtime_sources() -> None:
    root = _src_root()
    forbidden = (
        "import telegram",
        "from telegram",
        "import openai",
        "from openai",
        "import anthropic",
        "from anthropic",
        "import httpx",
        "from httpx",
        "import aiohttp",
        "from aiohttp",
        "import aiogram",
        "from aiogram",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text, f"{path}: vendor coupling {snippet!r}"


async def test_boundaries_wire_without_network() -> None:
    from aa.config import Settings
    from aa.control.runtime_control import RuntimeController
    from aa.corpus.context import CorpusContext
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.safety.router import SafetyDecision, SafetyRouter
    from aa.sessions.coordinator import SessionCoordinator
    from aa.telegram.transport import StubTelegramTransport, TelegramReply

    settings = Settings.from_env({})
    transport = StubTelegramTransport()
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(
            base_url=settings.opencode_base_url,
            command=settings.opencode_command,
            workdir=settings.opencode_workdir,
        )
    )
    corpus = CorpusContext(path=settings.aa_corpus_path, version=settings.aa_corpus_version)
    sessions = SessionCoordinator()
    safety = SafetyRouter()
    controller = RuntimeController(session_duration_seconds=0)

    await controller.start()
    await corpus.load()
    await runtime.start()
    await sessions.start()
    await safety.start()
    await transport.start()

    session = sessions.record_message(123)
    assert session.chat_id == 123
    assert safety.check("hello").decision is SafetyDecision.ALLOW
    assert safety.check("   ").decision is SafetyDecision.BLOCK
    await transport.send(TelegramReply(chat_id=123, text="hi"))

    await transport.stop()
    await safety.stop()
    await sessions.stop()
    await runtime.stop()
    await corpus.unload()
    await controller.stop()
