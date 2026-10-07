"""P0 kodmial/aa#202 regression: Muse Spark actually serves aa-verifier-v2.

Exact main already pins ``aa-verifier-v2`` to
``opencode/muse-spark-1.3-contributor-free`` with no fallback, but the
repository-owned exact-main run proves the verifier never reaches a
served-model identity (14 unavailable / 0 pass) while planner, answer
and summarizer serve through the Space Bunny fallback. Space Bunny is
forbidden for the verifier, so the repair must be Muse access, not
verifier routing.

These tests lock the repair at the adapter boundary:

- logical audit identity (``aa-verifier-v2``) is decoupled from the
  OpenCode transport agent selector (omitted on the wire) while the
  verifier system prompt still travels through the native ``system``
  field and the requested model stays pinned to Muse Spark;
- requested == served == Muse Spark is enforced exactly; a served-model
  mismatch fails closed;
- Space Bunny can never be recorded as serving ``aa-verifier-v2``;
- an absent served verifier is the specific Muse-access failure, never
  a generic Gate C failure;
- the 403 bounded retry (>=5s) and the 429 runner restart/resume
  semantics are preserved for the fallback-free verifier.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aa.conversation.model_adapter import (
    PLANNER_AGENT_V2,
    PLANNER_TRANSPORT_AGENT_V2,
    VERIFIER_AGENT_V2,
    VERIFIER_TRANSPORT_AGENT_V2,
    OpenCodeChatModel,
    build_planner_model,
    build_verifier_model,
    clear_primary_circuit,
)
from aa.conversation.verifier import clear_verifier_capability_cache
from aa.opencode.client import SessionInfo
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
)
from aa.qualification.verifier_muse_probe import (
    MUSE_ACCESS_FAILURE,
    SPACE_BUNNY_FORBIDDEN_FAILURE,
    VERIFIER_EXPECTED_MODEL,
    VERIFIER_FORBIDDEN_MODEL,
    check_verifier_served_exact,
    probe_verifier_muse_access,
    verifier_audit_entries,
)

MUSE = "opencode/muse-spark-1.3-contributor-free"
SPACE_BUNNY = "opencode/space-bunny-free"


@pytest.fixture(autouse=True)
def _clear_state() -> Any:
    clear_primary_circuit()
    clear_verifier_capability_cache()
    yield
    clear_primary_circuit()
    clear_verifier_capability_cache()


class _DecouplingClient:
    """Records wire transport agent vs logical audit identity."""

    def __init__(self, *, served_model: str = MUSE) -> None:
        self.wire_agents: list[str] = []
        self.audit_agents: list[str] = []
        self.models: list[str] = []
        self.systems: list[str] = []
        self._served_model = served_model
        self._sessions = 0

    async def create_session(self, title: str = "") -> SessionInfo:
        self._sessions += 1
        return SessionInfo(id=f"ses_{self._sessions}", title=title)

    async def delete_session(self, session_id: str) -> bool:
        return True

    def _record(self, *, agent: str, audit_agent: str, model: str, system: str) -> None:
        self.wire_agents.append(agent)
        self.audit_agents.append(audit_agent)
        self.models.append(model)
        self.systems.append(system)

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        system: str = "",
        audit_agent: str = "",
        **_: Any,
    ) -> str:
        self._record(agent=agent, audit_agent=audit_agent, model=model, system=system)
        return "probe-reply"

    async def send_structured_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str = "",
        agent: str = "",
        system: str = "",
        audit_agent: str = "",
        **_: Any,
    ) -> dict[str, object]:
        self._record(agent=agent, audit_agent=audit_agent, model=model, system=system)
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
        }


def test_verifier_factory_is_muse_only_with_omitted_transport() -> None:
    """The factory pins Muse, empties fallback, and omits the wire selector."""
    verifier = build_verifier_model(object(), primary_model=MUSE)  # type: ignore[arg-type]
    assert verifier.agent == VERIFIER_AGENT_V2 == "aa-verifier-v2"
    assert verifier.primary_model == MUSE
    assert verifier.fallback_model == ""
    assert verifier.wire_agent == VERIFIER_TRANSPORT_AGENT_V2 == ""


async def test_verifier_wire_omits_selector_but_keeps_logical_audit() -> None:
    """The wire omits the custom selector; audit keeps the logical identity."""
    client = _DecouplingClient()
    verifier = build_verifier_model(client, primary_model=MUSE)  # type: ignore[arg-type]
    await verifier._ainvoke_text("hello", system="verifier-system")
    assert client.wire_agents == [""]
    assert client.audit_agents == ["aa-verifier-v2"]
    assert client.models == [MUSE]
    assert client.systems == ["verifier-system"]


async def test_verifier_structured_wire_omits_selector_but_keeps_logical_audit() -> None:
    """Structured verifier calls decouple the same way as text calls."""
    client = _DecouplingClient()
    verifier = build_verifier_model(client, primary_model=MUSE)  # type: ignore[arg-type]
    await verifier.ainvoke_structured("hello", system="verifier-system", schema={"type": "object"})
    assert client.wire_agents == [""]
    assert client.audit_agents == ["aa-verifier-v2"]
    assert client.models == [MUSE]


async def test_hidden_model_family_omits_selector_and_keeps_logical_audit() -> None:
    """Planner plus derived answer/summarizer omit only the wire selector."""
    client = _DecouplingClient()
    planner = build_planner_model(
        client,  # type: ignore[arg-type]
        primary_model=MUSE,
        fallback_model=SPACE_BUNNY,
    )
    answer = planner.with_agent("aa-v2")
    summarizer = planner.with_agent("aa-summarizer-v2")
    await planner._ainvoke_text("hello", system="planner-system")
    await answer._ainvoke_text("hello", system="answer-system")
    await summarizer._ainvoke_text("hello", system="summary-system")
    assert planner.agent == PLANNER_AGENT_V2
    assert planner.wire_agent == PLANNER_TRANSPORT_AGENT_V2 == ""
    assert answer.wire_agent == ""
    assert summarizer.wire_agent == ""
    assert client.wire_agents == ["", "", ""]
    assert client.audit_agents == ["aa-planner-v2", "aa-v2", "aa-summarizer-v2"]


async def test_verifier_403_retries_bounded_then_raises_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback-free verifier: one >=5s 403 retry, then the rejection stands."""

    class _RejectingClient(_DecouplingClient):
        async def send_message(self, session_id: str, text: str, **kwargs: Any) -> str:
            self._record(
                agent=str(kwargs.get("agent", "")),
                audit_agent=str(kwargs.get("audit_agent", "")),
                model=str(kwargs.get("model", "")),
                system=str(kwargs.get("system", "")),
            )
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")

    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    verifier = build_verifier_model(
        _RejectingClient(),  # type: ignore[arg-type]
        primary_model=MUSE,
    )
    with pytest.raises(OpenCodeProviderAccessError):
        await verifier._ainvoke_text("hello", system="verifier-system")
    assert sleeps == [5.0]
    assert sleeps[0] >= 5.0


async def test_verifier_429_propagates_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 429 never retries locally; the runner restarts/resumes."""

    class _LimitedClient(_DecouplingClient):
        async def send_message(self, session_id: str, text: str, **kwargs: Any) -> str:
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    verifier = build_verifier_model(
        _LimitedClient(),  # type: ignore[arg-type]
        primary_model=MUSE,
    )
    with pytest.raises(OpenCodeRateLimitError):
        await verifier._ainvoke_text("hello", system="verifier-system")
    assert sleeps == []


def test_exact_verifier_pin_passes_for_muse() -> None:
    """Requested == served == Muse Spark passes the exact live assertion."""
    audit = [{"agent": "aa-verifier-v2", "requested": MUSE, "served": MUSE}]
    passed, failed = check_verifier_served_exact(audit)
    assert failed == ()
    assert "live-verifier-muse-requested-pinned" in passed
    assert "live-verifier-muse-served-exact" in passed
    assert VERIFIER_EXPECTED_MODEL == MUSE


def test_absent_verifier_is_specific_muse_access_failure() -> None:
    """No served verifier is the specific Muse-access failure, not Gate C."""
    passed, failed = check_verifier_served_exact([])
    assert failed == (MUSE_ACCESS_FAILURE,)
    assert passed == ()
    assert MUSE_ACCESS_FAILURE == "live-verifier-muse-access-failure"
    assert "gate" not in MUSE_ACCESS_FAILURE


def test_space_bunny_can_never_serve_verifier() -> None:
    """Any Space Bunny record for the verifier fails the forbidden check."""
    assert VERIFIER_FORBIDDEN_MODEL == SPACE_BUNNY
    served_fallback = [{"agent": "aa-verifier-v2", "requested": MUSE, "served": SPACE_BUNNY}]
    _, failed = check_verifier_served_exact(served_fallback)
    assert failed == (SPACE_BUNNY_FORBIDDEN_FAILURE,)
    requested_fallback = [
        {"agent": "aa-verifier-v2", "requested": SPACE_BUNNY, "served": SPACE_BUNNY}
    ]
    _, failed_requested = check_verifier_served_exact(requested_fallback)
    assert failed_requested == (SPACE_BUNNY_FORBIDDEN_FAILURE,)


def test_verifier_mismatch_fails_closed() -> None:
    """A non-Muse served verifier fails the exact pin, never passes."""
    audit = [{"agent": "aa-verifier-v2", "requested": MUSE, "served": "opencode/other-free"}]
    passed, failed = check_verifier_served_exact(audit)
    assert passed == ()
    assert failed == ("live-verifier-served-model-mismatch",)


def test_verifier_audit_entries_ignore_other_agents() -> None:
    """Only the logical verifier identity feeds the exact assertion."""
    audit = [
        {"agent": "aa-v2", "requested": MUSE, "served": MUSE},
        {"agent": "aa-verifier-v2", "requested": MUSE, "served": MUSE},
    ]
    entries = verifier_audit_entries(audit)
    assert entries == [{"agent": "aa-verifier-v2", "requested": MUSE, "served": MUSE}]


async def test_probe_isolates_selector_vs_model_and_stays_private() -> None:
    """The probe varies only the selector and never carries prompt/output."""

    class _ProbeClient:
        def __init__(self) -> None:
            self.wire_agents: list[str] = []
            self._sessions = 0

        async def create_session(self, title: str = "") -> SessionInfo:
            self._sessions += 1
            return SessionInfo(id=f"ses_{self._sessions}", title=title)

        async def delete_session(self, session_id: str) -> bool:
            return True

        @property
        def served_model_audit(self) -> tuple[dict[str, str], ...]:
            return ({"agent": "aa-verifier-v2", "requested": MUSE, "served": MUSE},)

        async def send_message(
            self,
            session_id: str,
            text: str,
            *,
            agent: str = "",
            model: str = "",
            system: str = "",
            audit_agent: str = "",
            **_: Any,
        ) -> str:
            assert model == MUSE
            assert audit_agent == "aa-verifier-v2"
            assert system.strip()
            self.wire_agents.append(agent)
            return "ok"

    client = _ProbeClient()
    result = await probe_verifier_muse_access(client, model=MUSE, system="verifier-system")
    assert result["diagnosis"] == "served-both"
    assert result["requested"] == MUSE
    assert result["custom_agent"]["served"] == MUSE
    assert result["omitted_agent"]["served"] == MUSE
    # Both wire variants were exercised with the same pinned model.
    assert client.wire_agents == ["aa-verifier-v2", ""]
    # Privacy: no prompt, system, or reply text in the probe result.
    serialized = repr(result)
    assert "verifier-system" not in serialized
    assert "Neutral connectivity probe." not in serialized
    assert "probe-reply" not in serialized
    assert "'ok'" not in serialized


async def test_probe_distinguishes_selector_rejection_from_model_rejection() -> None:
    """A custom-selector 403 with an omitted-selector serve isolates the agent."""

    class _SelectorRejectingClient:
        def __init__(self) -> None:
            self._sessions = 0

        async def create_session(self, title: str = "") -> SessionInfo:
            self._sessions += 1
            return SessionInfo(id=f"ses_{self._sessions}", title=title)

        async def delete_session(self, session_id: str) -> bool:
            return True

        async def send_message(
            self,
            session_id: str,
            text: str,
            *,
            agent: str = "",
            model: str = "",
            system: str = "",
            audit_agent: str = "",
            **_: Any,
        ) -> str:
            if agent:
                raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")
            return "ok"

    result = await probe_verifier_muse_access(
        _SelectorRejectingClient(), model=MUSE, system="verifier-system"
    )
    assert result["diagnosis"] == "agent-selector-rejection"
    assert result["custom_agent"]["outcome"] == "rejected"
    assert result["custom_agent"]["category"] == "provider-access"
    assert result["custom_agent"]["http_status"] == 403
    assert result["omitted_agent"]["outcome"] == "served"


def test_production_verifier_uses_decoupled_helper_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production wiring keeps Muse-only routing with an omitted wire selector."""
    import aa.conversation.graph as graph_module
    import aa.conversation.model_adapter as adapter_module
    from aa.config import DEFAULT_PRIMARY_MODEL
    from aa.conversation.graph_runtime import _ProductionGraphRuntime

    real_model = adapter_module.OpenCodeChatModel

    class _DummyModel:
        def __init__(
            self,
            client: object,
            *,
            agent: str,
            primary_model: str,
            fallback_model: str = "",
            request_timeout: float = 120.0,
            transport_agent: str | None = None,
        ) -> None:
            self.client = client
            self.agent = agent
            self.primary_model = primary_model
            self.fallback_model = fallback_model
            self.request_timeout = request_timeout
            self.transport_agent = transport_agent

        def with_agent(self, agent: str) -> _DummyModel:
            return _DummyModel(
                self.client,
                agent=agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                request_timeout=self.request_timeout,
                transport_agent=self.transport_agent,
            )

        @property
        def wire_agent(self) -> str:
            if self.transport_agent is None:
                return self.agent
            return self.transport_agent

    def _fake_build_planner(
        client: object,
        *,
        primary_model: str,
        fallback_model: str,
        request_timeout: float = 120.0,
    ) -> _DummyModel:
        return _DummyModel(
            client,
            agent="aa-planner-v2",
            primary_model=primary_model,
            fallback_model=fallback_model,
            request_timeout=request_timeout,
            transport_agent="",
        )

    def _fake_build_verifier(
        client: object, *, primary_model: str, request_timeout: float = 120.0
    ) -> _DummyModel:
        return _DummyModel(
            client,
            agent="aa-verifier-v2",
            primary_model=primary_model,
            fallback_model="",
            request_timeout=request_timeout,
            transport_agent="",
        )

    monkeypatch.setattr(adapter_module, "OpenCodeChatModel", _DummyModel)
    monkeypatch.setattr(adapter_module, "build_planner_model", _fake_build_planner)
    monkeypatch.setattr(adapter_module, "build_verifier_model", _fake_build_verifier)
    monkeypatch.setattr(graph_module, "build_turn_graph", lambda **kwargs: kwargs)

    runtime = _ProductionGraphRuntime(
        client=object(),
        settings=SimpleNamespace(
            opencode_model="opencode/muse-spark-1.3-contributor-free",
            opencode_fallback_model="opencode/space-bunny-free",
        ),
        index=None,
    )
    wired = runtime._build_graph(object())
    planner = wired["planner_model"]
    verifier = wired["verifier_model"]

    assert planner.fallback_model == "opencode/space-bunny-free"
    assert planner.wire_agent == ""
    assert wired["answer_model"].wire_agent == ""
    assert wired["summary_model"].wire_agent == ""
    assert verifier.agent == "aa-verifier-v2"
    assert verifier.primary_model == DEFAULT_PRIMARY_MODEL == MUSE
    assert verifier.fallback_model == ""
    assert verifier.wire_agent == ""
    assert real_model is not None


def test_production_verifier_never_uses_with_agent_fallback() -> None:
    """The verifier must never be rebuilt via planner.with_agent(...)."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "aa" / "conversation" / "graph_runtime.py"
    ).read_text(encoding="utf-8")
    assert "build_verifier_model" in source
    assert "with_agent(VERIFIER" not in source
    assert 'with_agent("aa-verifier' not in source
    assert "with_agent('aa-verifier" not in source


def test_live_lane_locks_exact_verifier_pin() -> None:
    """The live lane asserts requested == served == Muse for the verifier."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "qualification"
        / "product_contract_live.py"
    ).read_text(encoding="utf-8")
    assert "check_verifier_served_exact" in source
    assert "verifier_failed" in source
    assert "verifier_muse_probe" in source
