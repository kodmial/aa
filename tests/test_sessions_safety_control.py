"""Session, safety and runtime-control unit tests."""

from __future__ import annotations

import pytest

from aa.control.runtime_control import RuntimeController
from aa.safety.router import SafetyDecision, SafetyRouter
from aa.sessions.coordinator import SessionCoordinator


async def test_session_coordinator_tracks_chats() -> None:
    coordinator = SessionCoordinator()
    await coordinator.start()
    first = coordinator.get_or_create(1)
    second = coordinator.get_or_create(1)
    assert first is second
    coordinator.record_message(1)
    coordinator.record_message(2)
    assert coordinator.session_count() == 2
    await coordinator.stop()


async def test_safety_router_blocks_empty() -> None:
    router = SafetyRouter()
    await router.start()
    assert router.check("").decision is SafetyDecision.BLOCK
    assert router.check("   ").decision is SafetyDecision.BLOCK
    allowed = router.check("hello bot")
    assert allowed.decision is SafetyDecision.ALLOW
    await router.stop()


async def test_runtime_controller_deadline() -> None:
    controller = RuntimeController(session_duration_seconds=1000)
    await controller.start()
    assert controller.deadline is not None
    assert controller.time_remaining() is not None
    assert not controller.should_stop()
    await controller.stop()
    assert controller.should_stop()


async def test_runtime_controller_unbounded_until_stopped() -> None:
    controller = RuntimeController(session_duration_seconds=0)
    await controller.start()
    assert controller.deadline is None
    assert controller.time_remaining() is None
    assert not controller.should_stop()
    await controller.stop()
    assert controller.should_stop()


def test_runtime_controller_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        RuntimeController(session_duration_seconds=-1)


def test_runtime_controller_rejects_duration_over_three_hours() -> None:
    with pytest.raises(ValueError):
        RuntimeController(session_duration_seconds=10801)


async def test_runtime_controller_accepts_three_hour_maximum() -> None:
    controller = RuntimeController(session_duration_seconds=10800)
    await controller.start()
    assert controller.deadline is not None
    assert not controller.should_stop()
    await controller.stop()
