"""Production-boundary routing regression tests (issue #106)."""

from __future__ import annotations

from aa.app import Application
from aa.config import Settings
from aa.conversation.meta import META_CAPABILITY_REPLY, is_meta_capability_request
from aa.conversation.orchestrator import (
    FAIL_CLOSED_REPLY,
    contains_english_fallback,
    is_substantive,
    meets_russian_only,
)
from aa.conversation.output_limits import envelope_passes
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.qualification.production_boundary import (
    REQUIRED_CLASSES,
    REQUIRED_UTTERANCES,
    evaluate_all,
    evaluate_case,
    fail_closed_reply_is_valid,
    fixture_summary,
    load_fixture,
    observe_path,
)
from aa.qualification.validation_dag import admit_to_coding_queue
from aa.safety.router import SafetyRouter


def test_fixture_covers_required_classes_and_known_failures() -> None:
    cases = load_fixture()
    assert {case.routing_class for case in cases} >= set(REQUIRED_CLASSES)
    assert {case.utterance for case in cases} >= set(REQUIRED_UTTERANCES)
    summary = fixture_summary(cases)
    assert summary["cases"] == len(cases)
    assert fail_closed_reply_is_valid() is True


def test_known_meta_failures_are_conversational_not_grounded() -> None:
    for text in ("А что ты можешь?", "Тогда зачем ты?"):
        assert is_meta_capability_request(text) is True
        assert is_substantive(text) is False
        assert observe_path(next(c for c in load_fixture() if c.utterance == text)) == (
            "conversational"
        )


def test_known_broad_failure_takes_grounded_path() -> None:
    assert is_meta_capability_request("как бросить пить") is False
    assert is_substantive("как бросить пить") is True
    assert observe_path(next(c for c in load_fixture() if c.utterance == "как бросить пить")) == (
        "grounded"
    )


def test_meta_capability_reply_is_deterministic_russian() -> None:
    assert meets_russian_only(META_CAPABILITY_REPLY) is True
    assert not contains_english_fallback(META_CAPABILITY_REPLY)
    assert envelope_passes(META_CAPABILITY_REPLY) is True
    assert "[" not in META_CAPABILITY_REPLY


async def test_app_serves_meta_without_grounded_or_model_work() -> None:
    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:

        async def _forbidden_grounded(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("meta turn must not call the grounded pipeline")

        async def _forbidden_trivial(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("meta turn must not call the model path")

        app._run_grounded_turn = _forbidden_grounded  # type: ignore[method-assign]
        app._run_trivial_turn = _forbidden_trivial  # type: ignore[method-assign]
        for text in ("А что ты можешь?", "Тогда зачем ты?"):
            reply = await app.respond(1001, text)
            assert reply == META_CAPABILITY_REPLY
            assert reply != FAIL_CLOSED_REPLY
            assert meets_russian_only(reply) is True
    finally:
        await app.stop()


async def test_session_reset_and_isolation_are_control_plane() -> None:
    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        first = await app.sessions.ensure_opencode_session(2001, runtime.client)
        second = await app.sessions.ensure_opencode_session(2002, runtime.client)
        assert first != second
        reset = await app.sessions.reset_opencode_session(2001, runtime.client)
        assert reset != first
        assert (await app.sessions.ensure_opencode_session(2002, runtime.client)) == second
    finally:
        await app.stop()


def test_boundary_evaluation_passes_and_failing_fixture_repairs() -> None:
    cases = load_fixture()
    result, verdicts = evaluate_all(cases, router=SafetyRouter())
    assert result == "pass"
    assert verdicts and all(item.passed for item in verdicts)
    # Deliberately failing fixture: meta misrouted to grounded yields FAIL.
    broken = next(case for case in cases if case.utterance == "А что ты можешь?")
    wrong = broken.__class__(
        id=broken.id,
        routing_class=broken.routing_class,
        utterance=broken.utterance,
        expected_safety=broken.expected_safety,
        expected_path="grounded",
        requires_context=broken.requires_context,
    )
    failing = evaluate_case(wrong, router=SafetyRouter())
    assert failing.passed is False
    # Repaired current revision reruns to PASS without human intervention.
    repaired = evaluate_case(broken, router=SafetyRouter())
    assert repaired.passed is True


def test_validation_trackers_stay_out_of_coding_queue() -> None:
    for tracker in (40, 62, 63, 7):
        assert admit_to_coding_queue(tracker) is False
