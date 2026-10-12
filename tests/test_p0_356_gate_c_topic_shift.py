"""P0 kodmial/aa#356: Gate C topic-shift continuation stays on the prior subject.

Proven product failure on exact main 683d1aa (run 38219806839):

- C:live-continuation-helpful-topic-shift-5 on live-production-path
  (plus D:workflow-step-failed downstream and E:ordinary-turn-failed SLO).

Root cause: the Gate C topic-shift turn shares its Telegram chat with a
prior different-subject turn. The AA agent system prompt told the
generator to keep answering about the same subject on follow-ups that
rely on it, but stated no explicit rule for the opposite case: when the
current message explicitly introduces a different subject, the generator
must answer the new subject from the current evidence instead of
continuing the prior one. The planner prompt likewise asked for topic
shifts to be resolved without stating that an explicit change of
subject supersedes prior context. A continuity-biased generation
therefore serves prior-subject guidance on the new-subject turn, and
the continuation check (production telemetry plus the independent
whole-turn judge) fails it as unhelpful.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): state the explicit topic-shift switching rule in the answer
stage system prompt and the supersession rule in the planner stage
system prompt, both turn-independent.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def test_answer_prompt_switches_on_explicit_subject_change() -> None:
    agent = _read_prompt("aa-agent-system-v2.md")
    lowered = agent.casefold()
    assert "different subject" in lowered
    assert "do not continue the prior subject" in lowered
    assert "<book_evidence>" in agent


def test_planner_prompt_supersedes_prior_context_on_subject_change() -> None:
    planner = _read_prompt("aa-planner-system-v2.md")
    lowered = planner.casefold()
    assert "topic shifts" in lowered
    assert "supersedes prior context" in lowered
    assert "resolved intent" in lowered


def test_answer_assembly_carries_resolved_intent_with_user_message_last() -> None:
    from langchain_core.messages import BaseMessage, HumanMessage

    from aa.conversation.prompt_builder import build_answer_messages, render_turn_context

    recent: list[BaseMessage] = [
        HumanMessage(content="prior different-subject message"),
    ]
    messages = build_answer_messages(
        recent=recent,
        summary="",
        passages=[],
        user_message="current new-subject message",
        resolved_intent="resolved new-subject intent",
    )
    payload = str(messages[-1].content)
    assert "<resolved_intent>" in payload
    assert "<user_message>" in payload
    assert payload.rindex("<user_message>") > payload.rindex("<resolved_intent>")
    assert "current new-subject message" in payload
    assert "resolved new-subject intent" in payload
    # The standalone renderer keeps the same order contract.
    rendered = render_turn_context(
        summary="",
        passages=[],
        user_message="current new-subject message",
        resolved_intent="resolved new-subject intent",
    )
    assert rendered.rindex("<user_message>") > rendered.rindex("<resolved_intent>")


def test_planner_fallback_keeps_standalone_current_turn_first() -> None:
    from aa.conversation.planner_node import build_generic_fallback_queries

    queries = build_generic_fallback_queries(
        "current new-subject request",
        summary="prior different-subject summary",
        recent=["prior different-subject recent message"],
    )
    assert queries
    assert queries[0] == "current new-subject request"


def test_no_exact_question_special_cases() -> None:
    agent = _read_prompt("aa-agent-system-v2.md")
    planner = _read_prompt("aa-planner-system-v2.md")
    for fragment in (
        "ночью не могу успокоиться",
        "мысли крутятся",
        "ссора из-за моей выпивки",
        "поругались дома",
        "тянеет выпить",
    ):
        assert fragment not in agent
        assert fragment not in planner
