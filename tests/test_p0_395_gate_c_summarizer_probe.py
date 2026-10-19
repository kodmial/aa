"""P0 kodmial/aa#395: the live summarizer probe must use production policy.

Proven product failure (exact main 813f49a run 39299511659):

- C:live-summarizer-probe-OpenCodeProviderAccessError on live-production-path
  (plus E:ordinary-turn-failed downstream SLO).

The live qualification lane explicitly exercised the production summarizer
once, but constructed the probe as ``OpenCodeChatModel(agent=..., )`` with a
custom provider selector. Production builds the summarizer through the hidden
family helper, whose transport selector is omitted. The provider rejected the
custom selector with 403 and the probe recorded a production-path failure even
though the production summarizer path never uses that selector.

Repair (generic, no exact-question special cases, Product Contract #110
unchanged): build the production summarizer and the qualification probe
through the same ``build_summarizer_model`` helper so the probe cannot drift
from the real production wire policy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from aa.conversation.model_adapter import (
    SUMMARIZER_AGENT_V2,
    SUMMARIZER_TRANSPORT_AGENT_V2,
    OpenCodeChatModel,
    build_summarizer_model,
    clear_primary_circuit,
)
from aa.opencode.client import SessionInfo
from aa.opencode.errors import OpenCodeProviderAccessError

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "opencode/nemotron-3-ultra-free"


@pytest.fixture(autouse=True)
def _clear_shared_primary_circuit() -> Any:
    clear_primary_circuit()
    yield
    clear_primary_circuit()


class _SelectorAwareClient:
    """Serves the omitted selector; rejects any custom provider selector."""

    def __init__(self) -> None:
        self.wire_agents: list[str] = []
        self.audit_agents: list[str] = []
        self.models: list[str] = []
        self._sessions = 0

    async def create_session(self, title: str = "") -> SessionInfo:
        self._sessions += 1
        return SessionInfo(id=f"ses_{self._sessions}", title=title)

    async def delete_session(self, session_id: str) -> bool:
        _ = session_id
        return True

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        agent: str = "",
        model: str = "",
        audit_agent: str = "",
        system: str = "",
        **_kwargs: Any,
    ) -> str:
        _ = (session_id, text, system)
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        if agent:
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
        return "суммировано"


async def test_production_summarizer_builder_omits_custom_selector() -> None:
    client = _SelectorAwareClient()
    model = build_summarizer_model(
        client,  # type: ignore[arg-type]
        primary_model=MODEL,
        fallback_model=MODEL,
    )
    assert model.agent == SUMMARIZER_AGENT_V2
    assert model.wire_agent == SUMMARIZER_TRANSPORT_AGENT_V2 == ""
    assert model.primary_model == MODEL
    assert model.fallback_model == MODEL

    reply = await model.ainvoke([HumanMessage(content="Кратко суммируй фразу.")])
    assert str(reply.content).strip()

    # The wire omits the selector while the logical audit identity survives,
    # so the summarizer still appears in the served-model audit.
    assert client.wire_agents == [""]
    assert client.audit_agents == [SUMMARIZER_AGENT_V2]
    assert client.models == [MODEL]


async def test_custom_selector_probe_was_the_concrete_broken_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old probe construction reproduces the proven provider rejection."""
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    legacy_probe = OpenCodeChatModel(
        _SelectorAwareClient(),  # type: ignore[arg-type]
        agent=SUMMARIZER_AGENT_V2,
        primary_model=MODEL,
        fallback_model=MODEL,
    )
    assert legacy_probe.wire_agent == SUMMARIZER_AGENT_V2
    with pytest.raises(OpenCodeProviderAccessError):
        await legacy_probe.ainvoke([HumanMessage(content="Кратко суммируй фразу.")])
    assert sleeps and sleeps[0] >= 5.0

    # The production builder cannot enter that rejected path.
    probe = build_summarizer_model(
        _SelectorAwareClient(),  # type: ignore[arg-type]
        primary_model=MODEL,
        fallback_model=MODEL,
    )
    reply = await probe.ainvoke([HumanMessage(content="Кратко суммируй фразу.")])
    assert str(reply.content).strip()


def test_production_graph_and_live_probe_share_the_summarizer_builder() -> None:
    runtime_source = (REPO_ROOT / "src" / "aa" / "conversation" / "graph_runtime.py").read_text(
        encoding="utf-8"
    )
    live_source = (
        REPO_ROOT / "src" / "aa" / "qualification" / "product_contract_live.py"
    ).read_text(encoding="utf-8")

    assert "build_summarizer_model(" in runtime_source
    assert "with_agent(SUMMARIZER_AGENT_V2)" not in runtime_source

    probe_window = live_source.split("explicitly exercise the production summarizer")[1][:1200]
    assert "build_summarizer_model(" in probe_window
    assert "agent=SUMMARIZER_AGENT_V2" not in probe_window
