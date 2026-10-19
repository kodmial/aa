"""A/B and C are distinct exact-SHA evidence owners; no scenario is waived."""

from pathlib import Path

import pytest

from aa.qualification import product_contract_live as live


SHA = "a" * 40


def _fake_lane(lane: str, *, status: str = "PASS") -> live.LaneResult:
    return live.LaneResult(lane=lane, status=status)


def _setup(monkeypatch):
    monkeypatch.setattr(live, "checked_out_sha", lambda _: SHA)
    monkeypatch.setattr(live, "collect_static_gates", lambda _: {})
    monkeypatch.setattr(live, "assert_no_text_leak", lambda _: None)


def test_gate_c_only_calls_real_conversation_once_and_no_deterministic_lanes(monkeypatch):
    _setup(monkeypatch)
    calls = []

    async def model_lane(run_id):
        calls.append(("real", run_id))
        return _fake_lane("live-telegram-evidence")

    async def forbidden():
        raise AssertionError("Gate C duplicated Gate B deterministic product tests")

    monkeypatch.setattr(live, "run_live_telegram_evidence_lane", model_lane)
    monkeypatch.setattr(live, "run_message_lane", forbidden)
    monkeypatch.setattr(live, "run_transport_lane", forbidden)
    monkeypatch.setattr(live, "run_control_lane", lambda root: forbidden())
    monkeypatch.setattr(live, "run_voice_lane", lambda root: forbidden())
    result = live.evaluate_live(SHA, repo_root=Path("."), run_id="r1", gate_c_only=True)
    assert result.status == "PASS"
    assert [l.lane for l in result.lanes] == ["live-telegram-evidence"]
    assert calls == [("real", "r1")]


def test_gate_b_runs_all_product_1_to_41_and_voice_without_real_provider(monkeypatch):
    _setup(monkeypatch)
    calls = []

    async def messages():
        calls.append("message")
        return _fake_lane("product-contract-1-24")

    async def transport():
        calls.append("transport")
        return _fake_lane("telegram-transport-25-32")

    def control(root):
        calls.append("control")
        return _fake_lane("runtime-control-33-41")

    def voice(root):
        calls.append("voice")
        return _fake_lane("voice-1-16")

    async def no_real_provider(run_id):
        raise AssertionError("Gate B unexpectedly called live provider")

    monkeypatch.setattr(live, "run_message_lane", messages)
    monkeypatch.setattr(live, "run_transport_lane", transport)
    monkeypatch.setattr(live, "run_control_lane", control)
    monkeypatch.setattr(live, "run_voice_lane", voice)
    monkeypatch.setattr(live, "run_live_telegram_evidence_lane", no_real_provider)
    result = live.evaluate_live(SHA, repo_root=Path("."), deterministic_only=True)
    assert result.status == "PASS"
    assert calls == ["message", "transport", "control", "voice"]
    assert {l.lane for l in result.lanes} == {
        "product-contract-1-24",
        "telegram-transport-25-32",
        "runtime-control-33-41",
        "voice-1-16",
    }


def test_b_failure_preserved_not_flattened_to_pass(monkeypatch):
    _setup(monkeypatch)

    async def messages():
        return _fake_lane("product-contract-1-24", status="FAIL")

    async def transport():
        return _fake_lane("telegram-transport-25-32")

    monkeypatch.setattr(live, "run_message_lane", messages)
    monkeypatch.setattr(live, "run_transport_lane", transport)
    monkeypatch.setattr(live, "run_control_lane", lambda root: _fake_lane("runtime-control-33-41"))
    monkeypatch.setattr(live, "run_voice_lane", lambda root: _fake_lane("voice-1-16"))
    result = live.evaluate_live(SHA, repo_root=Path("."), deterministic_only=True)
    assert result.status == "FAIL"


def test_gate_c_provider_failure_cannot_be_reported_as_pass(monkeypatch):
    _setup(monkeypatch)

    async def provider_fails(run_id):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(live, "run_live_telegram_evidence_lane", provider_fails)
    result = live.evaluate_live(SHA, repo_root=Path("."), gate_c_only=True)
    assert result.status == "FAIL"
    assert result.lanes[0].failed == ("live-telegram-evidence:RuntimeError",)


def test_c_and_deterministic_flags_are_mutually_exclusive(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(live.ProductContractLiveError):
        live.evaluate_live(SHA, repo_root=Path("."), gate_c_only=True, deterministic_only=True)


def test_workflow_uses_two_bounded_jobs_and_exact_run_evidence_handoff():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/aa-self-proving-qualification.yml").read_text()
    assert "timeout-minutes: 120" not in workflow
    assert workflow.count("timeout-minutes: 45") == 2
    assert "qualify-live:" in workflow
    assert "needs: qualify" in workflow
    assert "aa-preflight-evidence" in workflow
    assert "--deterministic-only" in workflow
    assert "--gate-c-only" in workflow
    assert "self-proving-out/gate-A-evidence.json" in workflow
    assert "self-proving-out/gate-B-evidence.json" in workflow
    assert "Gate F - exact-main final verdict" in workflow
    assert "Publish authoritative Product Contract result" in workflow
    assert "continuum-qualification-result" in workflow


def test_standalone_full_diagnostic_still_invokes_all_lanes(monkeypatch):
    _setup(monkeypatch)
    async def async_all(run_id):
        return (_fake_lane("product-contract-1-24"),
                _fake_lane("telegram-transport-25-32"),
                _fake_lane("live-telegram-evidence"))

    monkeypatch.setattr(live, "_run_async_lanes", async_all)
    monkeypatch.setattr(live, "run_control_lane", lambda root: _fake_lane("runtime-control-33-41"))
    monkeypatch.setattr(live, "run_voice_lane", lambda root: _fake_lane("voice-1-16"))
    result = live.evaluate_live(SHA, repo_root=Path("."))
    assert result.status == "PASS"
    assert len(result.lanes) == 5
