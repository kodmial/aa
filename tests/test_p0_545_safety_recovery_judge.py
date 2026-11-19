"""P0 kodmial/aa#545: a judge-negative certified safety recovery is still served.

Proven failure on exact main 8bfcf7909f850199214380560a0bbf6976b6189e
(run 44494594832, canary qualification #7): the ``substantive-drinking``
canary turn asked "К вечеру очень тянет выпить, как с этим обходиться?".
The original draft was blocked by the mandatory outbound safety gate, and
the bounded safety recovery produced a candidate that passed outbound
safety, verifier grounding and whole-turn adequacy. The independent
whole-turn judge then rejected that recovered text, and the fall-through
converted the turn into ``answer_outcome="safety-blocked"`` -- a marked
service error delivered to the user. The canary failed exactly there with
``status=FAIL reason=service-error-delivered``.

The recovery pipeline has already exhausted its bounded alternative search
at that point, so a judge-only negative cannot manufacture a safer answer:
it can only replace a certified safe answer with a service error. Recovery
now carries the same practical-answer focus as the adequacy repair (so the
candidate is less likely to substitute a generic difficulty description for
guidance), and a judge-negative certified recovery is delivered with the
independent verdict recorded instead of collapsing to a typed service error.
Ordinary substantive delivery keeps the kodmial/aa#372 judge gate; that
contract is covered by ``test_p0_372_gate_c_substantive_judge``.

All fixtures below are synthetic, never the literal qualification prompt.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_HARMFUL_DRAFT = (
    "Многие пытались доказать себе, что они исключение, с помощью разных уловок. "
    "Честно разобраться помогает такой опыт: попробовать начать пить "
    "и резко прекратить, несколько раз."
)

_SAFE_RECOVERY = (
    "Поддержка рядом помогает пережить тягу сегодня, "
    "а честный разговор проясняет ближайшие трезвые шаги."
)


def _pack_entry(passage_id: str, text: str) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _HarmfulThenSafeAnswer:
    """Serve one unsafe draft, then the safe recovery candidate.

    The recorded human payloads let the regression prove that recovery
    generation uses the practical-answer focus while the safety policy
    remains a separate control block.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.user_payloads: list[str] = []

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, BaseMessage) and item.type == "human":
                    content = item.content
                    if isinstance(content, str) and "<user_message>" in content:
                        self.user_payloads.append(content)
        if self.calls == 1:
            return AIMessage(content=_HARMFUL_DRAFT)
        return AIMessage(content=_SAFE_RECOVERY)


class _SupportingVerifier:
    """Support every unit with the pack's display passage."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
            "addresses_intent": True,
        }


class _ScriptedJudge:
    """Separately-identified whole-turn judge double with scripted verdicts.

    The first verdict approves the original harmful draft so the turn
    reaches the outbound safety boundary; later verdicts decide the safe
    recovery candidate. This reproduces the live ordering without any
    text-dependent branching.
    """

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = list(verdicts)
        self.calls = 0
        self.agent = "stub-judge-545"

    async def ainvoke_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = 1,
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        index = min(self.calls, len(self._verdicts) - 1)
        helpful = bool(self._verdicts[index]) if self._verdicts else False
        self.calls += 1
        return {
            "helpful": helpful,
            "addresses_intent": helpful,
            "contains_substantive_claim": True,
        }


def _fresh_pack() -> Any:
    from aa.retrieval.evidence import EvidencePack, EvidencePassageData

    text = "Честный разговор проясняет ближайшие трезвые шаги."
    passage = EvidencePassageData(
        passage_id="chapter-3#exp0001",
        exact_text=text,
        source_id="ru-fourth-edition-txt",
        section_id="chapter-3",
        child_chunk_ids=("chapter-3#exp0001",),
        char_start=0,
        char_end=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(b"src").hexdigest(),
    )
    return EvidencePack(
        passages=(passage,),
        total_tokens=10,
        corpus_version="test",
        retrieval_metadata={},
    )


def _install_retrieval(monkeypatch: Any) -> None:
    import aa.retrieval.evidence as evidence

    def _fake_retrieve(  # noqa: E501
        index: object, queries: object, *, config: object = None, **kwargs: Any
    ) -> Any:
        _ = (index, queries, config, kwargs)
        return _fresh_pack()

    monkeypatch.setattr(evidence, "retrieve_evidence", _fake_retrieve)


def _turn_kwargs(judge: Any, answer_model: Any) -> dict[str, Any]:
    return {
        "user_message": "Синтетический запрос о вечернем напряжении",
        "summary": "",
        "recent": [HumanMessage(content="Здравствуйте")],
        "evidence_pack": [
            _pack_entry(
                "chapter-3#exp0000",
                "Поддержка рядом помогает пережить тягу сегодня.",
            )
        ],
        "answer_model": answer_model,
        "verifier_model": _SupportingVerifier(),
        "planner_model": object(),
        "retrieval_index": object(),
        "initial_query_count": 12,
        "upstream_latency_ms": 0.0,
        "planner_mode": "retrieval",
        "resolved_intent": "Синтетическое намерение справиться с вечерним напряжением",
        "whole_turn_judge_model": judge,
    }


async def test_judge_negative_certified_recovery_is_served_not_service_error(
    monkeypatch: Any,
) -> None:
    """A certified safe recovery is never collapsed into a service error."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_retrieval(monkeypatch)
    judge = _ScriptedJudge([True, False])
    answer_model = _HarmfulThenSafeAnswer()

    outcome = await run_v2_answer_turn(**_turn_kwargs(judge, answer_model))

    telemetry = dict(outcome.get("telemetry", {}))
    assert outcome["text"] == _SAFE_RECOVERY
    assert telemetry.get("answer_outcome") == "served"
    assert telemetry.get("answer_outcome") != "safety-blocked"
    assert telemetry.get("outbound_safety") == "repaired"
    assert telemetry.get("adequacy_verdict") == "pass"
    assert telemetry.get("answers_request") is True
    assert telemetry.get("technically_grounded") is True
    assert telemetry.get("failure_category") == ""
    assert telemetry.get("answer_failure_cause") is None
    assert telemetry.get("safety_recovery_judge_rejected") is True
    assert telemetry.get("independent_judge_helpful") is False
    # The original harmful draft was never delivered.
    assert _HARMFUL_DRAFT not in outcome["text"]
    assert judge.calls == 2


async def test_judge_positive_certified_recovery_is_served_without_rejection_marker(
    monkeypatch: Any,
) -> None:
    """The ordinary judge-approved recovery path is unchanged."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_retrieval(monkeypatch)
    judge = _ScriptedJudge([True, True])
    answer_model = _HarmfulThenSafeAnswer()

    outcome = await run_v2_answer_turn(**_turn_kwargs(judge, answer_model))

    telemetry = dict(outcome.get("telemetry", {}))
    assert outcome["text"] == _SAFE_RECOVERY
    assert telemetry.get("answer_outcome") == "served"
    assert telemetry.get("outbound_safety") == "repaired"
    assert telemetry.get("adequacy_verdict") == "pass"
    assert telemetry.get("independent_judge_helpful") is True
    assert telemetry.get("safety_recovery_judge_rejected") is not True


async def test_recovery_generation_uses_practical_focus_and_separate_safety_policy(
    monkeypatch: Any,
) -> None:
    """Recovery asks for practical guidance without merging the safety policy."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.safety.outbound import SAFE_RECOVERY_INSTRUCTION

    _install_retrieval(monkeypatch)
    judge = _ScriptedJudge([True, True])
    answer_model = _HarmfulThenSafeAnswer()

    await run_v2_answer_turn(**_turn_kwargs(judge, answer_model))

    assert len(answer_model.user_payloads) >= 2
    recovery_payload = answer_model.user_payloads[1]
    assert "практичный ответ по книге" in recovery_payload
    assert "<safety_policy>" in recovery_payload
    user_block = recovery_payload.split("<user_message>")[1].split("</user_message>")[0]
    assert SAFE_RECOVERY_INSTRUCTION not in user_block
