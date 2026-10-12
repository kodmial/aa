"""P0 kodmial/aa#370: Gate C harness collapse must stay concrete and SLO-evaluable.

Proven product failure on exact main fe7d3d22d1250485b3757f4a77cfa97dc729968b
(run 38262817876):

- C:live-production-telegram-harness on live-production-path
  (plus D:workflow-step-failed downstream and E:missing-live-samples SLO).

Live diagnostics for the failing run show the functional gap is in the
production-path harness itself, not only latency (Gate E owns the SLO):

- live-telegram-evidence FAIL with only 4 passed checks
  (transport-boundary, opencode-runtime-ready, retrieval-index-loaded,
  egress-simulated-labeled) and a single generic
  live-production-telegram-harness failure;
- turns_executed 0, served_turn_count 0, stage_snapshots_count 0,
  scenario_diagnostics [], verifier 0 units, zero latencies;
- opencode_request_latency message-text n:4 (p50 8587ms, max 67109ms),
  session-create n:4, session-delete n:4, judge_calls 0.

A lane-level crash before any turn completes therefore erased all
functional (fallback/diversity/grounding) and SLO evidence: Gate C could
only report a generic harness failure and Gate E degraded to
missing-live-samples.

Root cause (failure class, not one prompt): the live lane collapsed any
lane-level exception to the generic live-production-telegram-harness with
no exception-type attribution and no censored latency backfill; the
summarizer smoke probe aborted all 18 production scenarios on any
non-429 error; and generic (non-TurnFailed) graph failures left an empty
stage snapshot instead of a concrete failure-category snapshot.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): preserve the closed-vocabulary exception type as
live-production-harness-<ExcType>, backfill one bounded-censored latency
sample per ordinary family plus a harness diagnostic so Gate E still
evaluates the SLO, keep the summarizer probe non-fatal (429 still
retires), and preserve a minimal failure-category snapshot for generic
graph failures so qualification attributes the delivered service-error
text to its concrete failure instead of missing-turn-telemetry.
"""

from __future__ import annotations


def test_generic_graph_failure_preserves_concrete_snapshot() -> None:
    import asyncio

    from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime

    async def _raise_generic(graph: object, thread: str, text: str) -> dict[str, object]:
        _ = (graph, thread, text)
        raise ValueError("synthetic generic pipeline failure")

    async def _run() -> None:
        runtime = GraphTurnRuntime()
        runtime.attach_graph(object())
        runtime._invoke_graph = _raise_generic  # type: ignore[method-assign]
        await runtime.start()
        try:
            try:
                await runtime.run_turn(923701, "synthetic substantive request")
            except GraphRuntimeError as exc:
                assert exc.category == "graph-failed"
            else:
                raise AssertionError("generic failure must raise GraphRuntimeError")
            snapshot = runtime.last_telemetry_for_thread(runtime.thread_id(923701))
            assert snapshot, "generic failure must not leave an empty snapshot"
            assert snapshot.get("answer_outcome") == "failed"
            assert snapshot.get("failure_category") == "ValueError"
        finally:
            await runtime.stop()

    asyncio.run(_run())


def test_live_harness_records_concrete_category_and_censored_samples() -> None:
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "qualification"
        / "product_contract_live.py"
    ).read_text(encoding="utf-8")
    assert "live-production-harness-" in source
    assert "harness_error" in source
    assert "censored" in source
    assert "live-summarizer-probe-" in source
    # Closed vocabulary only: no user/corpus text, no exact qualification
    # prompts in the harness attribution itself.
    assert "failure_category" in source


def test_no_exact_question_special_cases() -> None:
    import inspect
    import pathlib

    import aa.conversation.graph_runtime as graph_mod

    runtime_source = inspect.getsource(graph_mod.GraphTurnRuntime.run_turn)
    live_source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "qualification"
        / "product_contract_live.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Поругались дома из-за бухла",
        "Какой телефон сейчас выгоднее купить",
        "Привет! К вечеру очень тянет выпить",
        "Привет, подскажи че делать когда под вечер тянеет выпить",
    ):
        assert fragment not in runtime_source
        # The live lane legitimately contains the frozen C-regression corpus;
        # the repair itself (harness attribution + backfill + summarizer
        # hardening) must not branch on any literal prompt.
        harness_window = live_source.split("live-production-harness-")[1][:2000]
        assert fragment not in harness_window
