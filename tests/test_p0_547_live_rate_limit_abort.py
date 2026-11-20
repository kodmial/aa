"""P0 kodmial/aa#547: provider 429 aborts a live scenario promptly.

Proven failure on exact main ``d23287a758d2775749823981cc51a9e5545bf24d``
(run 44611952510, qualification #7): Gate C scenario 4
(``family=followup-ellipsis``) received an ``OpenCodeRateLimitError`` from a
verifier ``message-structured`` call. The application recorded the fatal
worker-lifecycle failure, but the per-chat dispatcher swallowed the re-raised
exception and the live lane only watched for a delivered reply. The scenario
therefore waited out the full 1800-second inactivity watchdog and was
misclassified as ``stuck-provider`` instead of retiring the runner for
fresh-VM checkpoint recovery.

The lane now observes the application's recorded fatal error in every live
delivery wait and re-raises it, so the existing 429 runner-retire contract
(exit 75 / ``OPENCODE_429_RESTART_REQUIRED``) is preserved.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from aa.app import Application
from aa.config import Settings
from aa.opencode.errors import OpenCodeRateLimitError
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import TelegramIncoming


def _app() -> Application:
    return Application(
        Settings.from_env({}),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
    )


async def test_rate_limit_turn_records_application_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 turn must surface the lifecycle signal the run loop re-raises."""
    app = _app()
    error = OpenCodeRateLimitError("opencode request rate-limited: http=429")

    async def _boom(incoming: TelegramIncoming) -> None:
        _ = incoming
        raise error

    monkeypatch.setattr(app, "_handle_telegram_update", _boom)
    incoming = TelegramIncoming(
        update_id=1,
        chat_id=920002,
        message_id=1,
        text="synthetic scenario text",
    )
    with pytest.raises(OpenCodeRateLimitError) as raised:
        await app._process_dispatched_update(incoming)
    assert raised.value is error
    assert app.fatal_error is error
    assert app.controller.should_stop() is True


def test_live_lane_reraises_recorded_fatal_error() -> None:
    from aa.qualification.product_contract_live import _raise_if_runtime_fatal

    error = OpenCodeRateLimitError("opencode request rate-limited: http=429")
    with pytest.raises(OpenCodeRateLimitError) as raised:
        _raise_if_runtime_fatal(SimpleNamespace(fatal_error=error))
    assert raised.value is error


def test_live_lane_fatal_check_is_a_noop_without_failure() -> None:
    from aa.qualification.product_contract_live import _raise_if_runtime_fatal

    _raise_if_runtime_fatal(SimpleNamespace(fatal_error=None))


def test_live_lane_fatal_check_fails_fast_without_signal() -> None:
    from aa.qualification.product_contract_live import _raise_if_runtime_fatal

    with pytest.raises(AttributeError, match="fatal_error"):
        _raise_if_runtime_fatal(SimpleNamespace())


def _is_fatal_check(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "_raise_if_runtime_fatal"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
    )


def _contains_sleep(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "sleep"
        for child in ast.walk(node)
    )


def test_live_delivery_waits_observe_fatal_error_before_inactivity() -> None:
    """Every live wait checks fatal before sleeping and again after it exits."""
    from aa.qualification.product_contract_live import run_live_telegram_evidence_lane

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_live_telegram_evidence_lane)))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    wait_loops = [
        node for node in ast.walk(tree) if isinstance(node, ast.While) and _contains_sleep(node)
    ]
    assert wait_loops
    for loop in wait_loops:
        assert _is_fatal_check(loop.body[0])
        parent = parents[loop]
        siblings: list[ast.stmt] | None = None
        for field in ("body", "orelse", "finalbody"):
            candidate = getattr(parent, field, None)
            if isinstance(candidate, list) and loop in candidate:
                siblings = candidate
                break
        assert siblings is not None
        index = siblings.index(loop)
        assert index + 1 < len(siblings), "wait loop must re-check fatal after exit"
        assert _is_fatal_check(siblings[index + 1])
