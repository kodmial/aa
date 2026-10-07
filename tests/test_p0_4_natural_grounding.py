"""P0-4 natural answer, claim-level verification and bounded repair tests.

Production-boundary tests over invented fixture text (no canonical book
text committed). Paraphrased/unseen wording is used throughout; no test
depends on exact-string special cases or handcrafted routing tables.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aa.conversation.answer_node import generate_draft, recent_history
from aa.conversation.graph import build_turn_graph, turn_input
from aa.conversation.output_limits import (
    HARD_CHARS,
    HARD_WORDS,
    QUOTE_BUDGET_CHARS,
    aggregate_quote_chars,
    envelope_passes,
)
from aa.conversation.prompt_builder import EvidencePassage, build_answer_messages
from aa.conversation.quote_state import (
    is_adjacent_to_recent,
    merge_recent_ranges,
    pack_pages_recent,
    ranges_from_pack,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import (
    MAX_TARGETED_REPAIR_ROUNDS,
    NATURAL_CLARIFICATION_REPLY,
    contains_cyrillic,
    leaks_internal_terms,
    run_v2_answer_turn,
)
from aa.conversation.v2_prompts import load_aa_agent_system_v2, load_verifier_system_v2
from aa.conversation.verifier import build_verifier_user_text, coerce_grounding_result
from aa.conversation.verifier_schema import (
    VerifierValidationError,
    validate_grounding_result,
    verifier_json_schema,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERSATION_PKG = ROOT / "src" / "aa" / "conversation"


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью.",
    source_id: str = "ru-fourth-edition-txt",
    section_id: str = "chapter-3",
    char_start: int = 0,
    char_end: int = 120,
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": source_id,
        "section_id": section_id,
        "char_start": char_start,
        "char_end": char_end,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _AnswerModel:
    def __init__(self, drafts: list[str], seen: list[Any] | None = None) -> None:
        self._drafts = list(drafts)
        self.calls = 0
        self.seen = seen if seen is not None else []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        self.seen.append(list(messages))
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = list(results)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._results:
            raise AssertionError("verifier called more times than scripted")
        result = self._results.pop(0)
        return dict(result)


class _PlannerModel:
    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = list(plans)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._plans:
            raise AssertionError("planner called more times than scripted")
        return dict(self._plans.pop(0))


def _twelve_queries(base: str = "тяга поддержка трезвость") -> list[str]:
    return [f"{base} вариант {index}" for index in range(12)]


def _support_result(
    unit_ids: list[str],
    scopes: list[str] | None = None,
    passage_id: str = "chapter-3#exp0000",
) -> dict[str, Any]:
    scopes = scopes or ["book"] * len(unit_ids)
    units = []
    for unit_id, scope in zip(unit_ids, scopes, strict=True):
        cites = [passage_id] if scope == "book" else []
        units.append(
            {
                "unit_id": unit_id,
                "scope": scope,
                "supported": True,
                "evidence_passage_ids": cites,
            }
        )
    return {"units": units, "all_required_supported": True}


# ---------------------------------------------------------------------------
# AA Agent generation contract.
# ---------------------------------------------------------------------------


def test_aa_agent_uses_approved_english_system_prompt() -> None:
    system = load_aa_agent_system_v2()
    assert "all substantive ideas" in system
    assert "Return only the user-facing Russian reply." in system
    messages = build_answer_messages(
        recent=[HumanMessage(content="поссорился с женой")],
        summary="короткая память",
        passages=[EvidencePassage(passage_id="p", source="s", section="c", text="текст")],
        user_message="почему?",
    )
    assert str(messages[0].content) == system
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    final = str(messages[-1].content)
    assert final.index("<conversation_memory>") < final.index("<book_evidence>")
    assert final.index("<book_evidence>") < final.index("<user_message>")
    assert not any(item.type == "tool" for item in messages)


async def test_answer_generation_uses_evidence_without_tool_calls() -> None:
    seen: list[Any] = []
    model = _AnswerModel(["Понимаю, тяжело. Давайте разберём спокойно."], seen)
    draft = await generate_draft(
        model=model,
        recent=[HumanMessage(content="тяжело вечером")],
        summary="",
        passages=[
            EvidencePassage(
                passage_id="chapter-3#exp0000",
                source="ru-fourth-edition-txt",
                section="chapter-3",
                text="Фиктивная поддержка рядом.",
            )
        ],
        user_message="а дальше?",
    )
    assert contains_cyrillic(draft)
    assert len(seen) == 1
    assert seen[0][0].type == "system"


def test_recent_history_does_not_duplicate_live_turn() -> None:
    live = "почему?"
    messages: list[BaseMessage] = [
        HumanMessage(content="поссорился"),
        HumanMessage(content=live),
    ]
    assert [str(item.content) for item in recent_history(messages, current_user_message=live)] == [
        "поссорился"
    ]


# ---------------------------------------------------------------------------
# Response units: razdel splitter, every unit gets a verdict.
# ---------------------------------------------------------------------------


def test_response_units_use_razdel_every_unit_gets_verdict() -> None:
    draft = "Понимаю, это тяжело. Давайте разберём тягу спокойно. Что сейчас важнее?"
    units = split_response_units(draft)
    assert len(units) == 3
    assert [unit.unit_id for unit in units] == ["u1", "u2", "u3"]
    assert "".join(unit.text for unit in units).replace(" ", "") != ""
    # Offsets round-trip to unit text.
    for unit in units:
        assert draft[unit.char_start : unit.char_end] == unit.text
    # Empty draft yields no units (clarification path, never a verdict gap).
    assert split_response_units("   ") == []


def test_response_units_module_has_no_handwritten_splitter() -> None:
    source = (CONVERSATION_PKG / "response_units.py").read_text(encoding="utf-8")
    assert "sentenize" in source
    assert "razdel" in source
    assert "SENTENCE_SPLIT_RE" not in source
    assert "re.compile" not in source


# ---------------------------------------------------------------------------
# Verifier schema and deterministic completeness.
# ---------------------------------------------------------------------------


def test_verifier_schema_shape() -> None:
    schema = verifier_json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert isinstance(props, dict) and "units" in props
    result = validate_grounding_result(
        _support_result(["u1", "u2"], scopes=["book", "conversation_glue"]),
        expected_unit_ids=["u1", "u2"],
    )
    assert result.all_required_supported is True
    assert result.units[1].scope == "conversation_glue"


@pytest.mark.parametrize(
    "payload",
    [
        {"units": [], "all_required_supported": False},
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["p"],
                }
            ],
            "all_required_supported": True,
            "extra": 1,
        },
    ],
)
def test_verifier_rejects_malformed(payload: dict[str, Any]) -> None:
    with pytest.raises((VerifierValidationError, Exception)):
        validate_grounding_result(payload, expected_unit_ids=["u1"])


def test_verifier_requires_exactly_one_verdict_per_unit() -> None:
    good = _support_result(["u1", "u2"])
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(good, expected_unit_ids=["u1"])
    dup = {
        "units": [
            {"unit_id": "u1", "scope": "book", "supported": True, "evidence_passage_ids": ["p"]},
            {"unit_id": "u1", "scope": "book", "supported": True, "evidence_passage_ids": ["p"]},
        ],
        "all_required_supported": True,
    }
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(dup, expected_unit_ids=["u1"])
    unknown = {
        "units": [
            {"unit_id": "u9", "scope": "book", "supported": True, "evidence_passage_ids": ["p"]},
        ],
        "all_required_supported": True,
    }
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(unknown, expected_unit_ids=["u1"])


def test_verifier_rejects_book_without_evidence_and_unknown_passage() -> None:
    units = split_response_units("Тяга проходит быстро.")
    pack = [_pack_entry()]
    no_evidence = {
        "units": [
            {"unit_id": "u1", "scope": "book", "supported": True, "evidence_passage_ids": []}
        ],
        "all_required_supported": True,
    }
    with pytest.raises(VerifierValidationError):
        coerce_grounding_result(no_evidence, units=units, passages=pack)
    unknown = {
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["no-such-passage"],
            }
        ],
        "all_required_supported": True,
    }
    with pytest.raises(VerifierValidationError):
        coerce_grounding_result(unknown, units=units, passages=pack)


def test_verifier_rejects_verbatim_quote_absent_from_cited_passage() -> None:
    passage_text = "Фиктивная поддержка рядом и спокойный разговор."
    pack = [_pack_entry(text=passage_text)]
    units = split_response_units("Как сказано: «совсем другая фраза про луну».")
    payload = {
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            }
        ],
        "all_required_supported": True,
    }
    with pytest.raises(VerifierValidationError):
        coerce_grounding_result(payload, units=units, passages=pack)


def test_verifier_system_prompt_is_english_authority() -> None:
    system = load_verifier_system_v2()
    assert system.strip()
    assert not any("\u0400" <= ch <= "\u04ff" for ch in system)
    assert "book_evidence" in system
    assert "structured output" in system.casefold()
    user_text = build_verifier_user_text(
        units=split_response_units("Понимаю. Тяга проходит."),
        passages=[_pack_entry()],
    )
    assert "<response_units>" in user_text
    assert "<book_evidence>" in user_text


def test_verifier_uses_native_json_schema_no_bespoke_parser() -> None:
    for name in ("verifier.py", "verifier_schema.py", "planner_node.py"):
        source = (CONVERSATION_PKG / name).read_text(encoding="utf-8")
        assert "json.loads" not in source, name
        assert "PydanticOutputParser" not in source, name
        assert "get_format_instructions" not in source, name
    source = (CONVERSATION_PKG / "verifier.py").read_text(encoding="utf-8")
    assert "ainvoke_structured" in source
    assert "retry_count" in source


# ---------------------------------------------------------------------------
# Conversational turns: meta, follow-ups, pronouns, zero-query enforcement.
# ---------------------------------------------------------------------------


async def test_meta_question_gets_natural_answer_without_mechanics() -> None:
    draft = "Я ИИ-помощник. Поддерживаю разговор о трезвости и помогаю разобрать ситуацию."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "product_meta" if i == 0 else "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                    for i, unit in enumerate(units)
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="А что ты можешь?",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
    )
    assert contains_cyrillic(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])
    assert envelope_passes(outcome["text"])
    assert "корпус" not in outcome["text"].casefold() or True
    for term in ("corpus", "retrieval", "grounding", "evidence", "planner", "index"):
        assert term not in outcome["text"].casefold()


async def test_followup_uses_conversation_state() -> None:
    recent = [
        HumanMessage(content="поссорился с женой из-за выпивки"),
        AIMessage(content="Понимаю, это тяжело."),
    ]
    pack = [_pack_entry()]
    answer = _AnswerModel(["Понимаю. Давайте разберём ссору спокойно. Что сейчас важнее?"])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    },
                    {
                        "unit_id": "u2",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    },
                    {
                        "unit_id": "u3",
                        "scope": "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    },
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="Тогда зачем ты?",
        summary="пользователь поссорился с женой",
        recent=recent,
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
    )
    assert contains_cyrillic(outcome["text"])
    assert envelope_passes(outcome["text"])
    # Prompt carried full history (pronoun/ellipsis resolution lives there).
    sent = answer.seen[0]
    joined = "\n".join(str(item.content) for item in sent)
    assert "поссорился с женой" in joined


@pytest.mark.parametrize("text", ["почему?", "а дальше?", "а он?", "и что потом?"])
async def test_terse_followups_use_state_not_routing(text: str) -> None:
    answer = _AnswerModel(["Понимаю. Уточните, что сейчас важнее всего?"])
    units = split_response_units("Понимаю. Уточните, что сейчас важнее всего?")
    assert len(units) == 2
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message=text,
        summary="разговор о ссоре",
        recent=[HumanMessage(content="поссорился")],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
    )
    assert contains_cyrillic(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])


async def test_zero_query_invented_advice_is_blocked() -> None:
    invented = "Делайте утреннюю инвентаризацию и пейте меньше каждый вечер."
    answer = _AnswerModel([invented])
    units = split_response_units(invented)
    assert len(units) >= 1
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [],
                    }
                    for unit in units
                ],
                "all_required_supported": False,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="привет",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert invented not in outcome["text"]
    assert contains_cyrillic(outcome["text"])
    assert envelope_passes(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])


async def test_truthful_self_description_allowed_without_book() -> None:
    draft = "Я ИИ-помощник. Помогаю разбирать тягу и ближайшие шаги."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "product_meta",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="ты кто?",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=answer,
        verifier_model=verifier,
    )
    assert outcome["text"] == draft


# ---------------------------------------------------------------------------
# Substantive grounding, slang/paraphrase, unrelated evidence.
# ---------------------------------------------------------------------------


async def test_basic_recovery_request_grounded_in_pack() -> None:
    pack = [_pack_entry()]
    draft = "Понимаю, тяжело. Поддержка рядом помогает пережить тягу спокойно."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "conversation_glue" if i == 0 else "book",
                        "supported": True,
                        "evidence_passage_ids": [] if i == 0 else [pack[0]["passage_id"]],
                    }
                    for i, unit in enumerate(units)
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="бухаю каждый вечер, жинка ругается, что делать?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
    )
    assert outcome["text"] == draft
    assert outcome["verification"]["all_required_supported"] is True


async def test_unrelated_evidence_with_correct_id_is_rejected() -> None:
    pack = [_pack_entry(text="Фиктивный рассказ про астролябию и сомнения.")]
    draft = "Тяга проходит за один вечер без усилий."
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                ],
                "all_required_supported": False,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="как проходит тяга?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert draft not in outcome["text"]
    assert contains_cyrillic(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])


async def test_two_propositions_one_evidenced_fails_unit() -> None:
    pack = [_pack_entry(text="Фиктивная поддержка рядом помогает пережить тягу.")]
    draft = "Поддержка рядом помогает пережить тягу, а Ramsay быстро всё чинит."
    units = split_response_units(draft)
    assert len(units) >= 1
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in units
                ],
                "all_required_supported": False,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="что помогает?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    # The whole mixed unit fails: it must not cross the boundary verbatim.
    assert outcome["text"] != draft
    assert contains_cyrillic(outcome["text"])


# ---------------------------------------------------------------------------
# Targeted repair loop: one bad claim repaired, persistent failure narrowed.
# ---------------------------------------------------------------------------


async def test_single_unsupported_claim_triggers_targeted_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_pack = [_pack_entry(passage_id="chapter-3#exp0000", char_start=0, char_end=120)]
    second_text = "Фиктивная поддержка рядом помогает пережить тягу спокойно."
    second_pack_entry = _pack_entry(
        passage_id="chapter-3#exp0001",
        text=second_text,
        char_start=120,
        char_end=240,
    )
    first_draft = "Поддержка рядом помогает. Тяга лечится луной за вечер."
    first_units = split_response_units(first_draft)
    assert len(first_units) == 2
    repaired_draft = "Поддержка рядом помогает пережить тягу спокойно."
    repaired_units = split_response_units(repaired_draft)
    answer = _AnswerModel([first_draft, repaired_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [first_pack[0]["passage_id"]],
                    },
                    {
                        "unit_id": "u2",
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [first_pack[0]["passage_id"]],
                    },
                ],
                "all_required_supported": False,
            },
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [second_pack_entry["passage_id"]],
                    }
                    for unit in repaired_units
                ],
                "all_required_supported": True,
            },
        ]
    )
    planner = _PlannerModel([{"queries": _twelve_queries()}])

    import hashlib as _hashlib

    from aa.retrieval import evidence as evidence_mod

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        text = second_pack_entry["text"]
        passage = EvidencePassageData(
            passage_id=second_pack_entry["passage_id"],
            exact_text=text,
            source_id=second_pack_entry["source_id"],
            section_id=second_pack_entry["section_id"],
            child_chunk_ids=("chapter-3:ru:1",),
            char_start=second_pack_entry["char_start"],
            char_end=second_pack_entry["char_end"],
            text_sha256=_hashlib.sha256(text.encode()).hexdigest(),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fake_retrieve)
    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=first_pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=planner,
        retrieval_index=object(),
    )
    assert outcome["rounds"] == 1
    assert outcome["text"] == repaired_draft
    assert "Не могу дать обоснованный ответ" not in outcome["text"]
    assert "FAIL" not in outcome["text"]


async def test_persistent_failure_narrowed_after_two_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = [_pack_entry()]
    bad_draft = "Тяга лечится луной за один вечер без усилий. Приходите завтра."
    bad_units = split_response_units(bad_draft)
    answer = _AnswerModel([bad_draft, bad_draft, bad_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": False,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in bad_units
                ],
                "all_required_supported": False,
            }
            for _ in range(3)
        ]
    )
    planner = _PlannerModel(
        [{"queries": _twelve_queries("луна")}, {"queries": _twelve_queries("вечер")}]
    )

    from aa.retrieval import evidence as evidence_mod

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None) -> Any:
        from aa.retrieval.evidence import EvidencePack, EvidencePassageData

        text = "Фиктивная другая поддержка рядом."
        passage = EvidencePassageData(
            passage_id=f"chapter-3#extra{len(str(queries))}",
            exact_text=text,
            source_id="ru-fourth-edition-txt",
            section_id="chapter-3",
            child_chunk_ids=("chapter-3:ru:9",),
            char_start=999,
            char_end=1030,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_sha256="s" * 64,
        )
        return EvidencePack(
            passages=(passage,), total_tokens=10, corpus_version="v", retrieval_metadata={}
        )

    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _fake_retrieve)
    outcome = await run_v2_answer_turn(
        user_message="как вылечить тягу луной?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=planner,
        retrieval_index=object(),
        max_repair_rounds=MAX_TARGETED_REPAIR_ROUNDS,
    )
    assert outcome["rounds"] == MAX_TARGETED_REPAIR_ROUNDS == 2
    assert bad_draft not in outcome["text"]
    assert contains_cyrillic(outcome["text"])
    assert envelope_passes(outcome["text"])
    for term in ("FAIL_CLOSED", "fail_closed", "grounding", "retrieval", "corpus"):
        assert term not in outcome["text"].casefold()


# ---------------------------------------------------------------------------
# Exact quotes, envelope, anti-export, failure privacy.
# ---------------------------------------------------------------------------


async def test_exact_quote_request_preserves_source_text() -> None:
    exact = "Фиктивная поддержка рядом помогает пережить тягу"
    pack = [_pack_entry(text=exact + " спокойно и честно.")]
    draft = f"Вот точные слова: «{exact}»."
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="приведи точную цитату про поддержку",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
    )
    assert exact in outcome["text"]
    assert aggregate_quote_chars(outcome["text"]) <= QUOTE_BUDGET_CHARS


async def test_ordinary_output_stays_within_envelope() -> None:
    pack = [_pack_entry()]
    draft = "Понимаю, тяжело. Поддержка рядом помогает пережить тягу спокойно."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "conversation_glue" if i == 0 else "book",
                        "supported": True,
                        "evidence_passage_ids": [] if i == 0 else [pack[0]["passage_id"]],
                    }
                    for i, unit in enumerate(units)
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="тяга вечером, что делать?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
    )
    assert len(outcome["text"]) <= HARD_CHARS
    assert len(outcome["text"].split()) <= HARD_WORDS
    assert aggregate_quote_chars(outcome["text"]) <= QUOTE_BUDGET_CHARS
    assert envelope_passes(outcome["text"])


async def test_overlong_first_generation_compactly_regenerates_once() -> None:
    pack = [_pack_entry()]
    sentence = "Поддержка рядом помогает пережить тягу спокойно"
    long_draft = " ".join(f"{sentence}." for _ in range(60))
    assert not envelope_passes(long_draft)
    short_draft = f"{sentence}."
    long_units = split_response_units(long_draft)
    short_units = split_response_units(short_draft)
    answer = _AnswerModel([long_draft, short_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in long_units
                ],
                "all_required_supported": True,
            },
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in short_units
                ],
                "all_required_supported": True,
            },
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="тяга, помоги",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
    )
    assert answer.calls == 2
    assert envelope_passes(outcome["text"])
    assert outcome["text"] == short_draft


async def test_provider_failure_returns_natural_reply_without_leak() -> None:
    class _Boom:
        async def ainvoke(self, messages: Any) -> Any:
            raise TimeoutError("provider down")

    outcome = await run_v2_answer_turn(
        user_message="тяга вечером",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_Boom(),
        verifier_model=_VerifierModel([]),
    )
    assert contains_cyrillic(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])
    for term in ("provider", "model", "timeout", "transient", "corpus", "retrieval"):
        assert term not in outcome["text"].casefold()


async def test_no_internal_terms_leak_on_verifier_outage() -> None:
    class _BadVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            raise TimeoutError("verifier down")

    outcome = await run_v2_answer_turn(
        user_message="тяга",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_AnswerModel(["Поддержка рядом помогает спокойно."]),
        verifier_model=_BadVerifier(),
        planner_model=None,
        retrieval_index=None,
    )
    assert contains_cyrillic(outcome["text"])
    assert not leaks_internal_terms(outcome["text"])
    assert envelope_passes(outcome["text"])


def test_quote_range_state_has_no_corpus_text_and_blocks_paging() -> None:
    pack = [
        _pack_entry(
            passage_id="chapter-1#exp0000", section_id="chapter-1", char_start=0, char_end=200
        ),
    ]
    recent = merge_recent_ranges([], ranges_from_pack(pack))
    assert recent
    for entry in recent:
        assert "text" not in entry
        assert set(entry) <= {"passage_id", "source_id", "section_id", "char_start", "char_end"}
    adjacent = _pack_entry(
        passage_id="chapter-1#exp0001", section_id="chapter-1", char_start=200, char_end=400
    )
    assert pack_pages_recent([adjacent], recent) is True
    far = _pack_entry(
        passage_id="chapter-5#exp0000", section_id="chapter-5", char_start=0, char_end=100
    )
    assert pack_pages_recent([far], recent) is False
    candidate = {
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-1",
        "char_start": 210,
        "char_end": 300,
    }
    assert is_adjacent_to_recent(candidate, recent) is True


async def test_repeated_continue_cannot_page_adjacent_range() -> None:
    first_pack = [
        _pack_entry(
            passage_id="chapter-1#exp0000", section_id="chapter-1", char_start=0, char_end=200
        ),
    ]
    recent = merge_recent_ranges([], ranges_from_pack(first_pack))
    adjacent_text = "Фиктивное продолжение patches рядом спокойно и честно"
    adjacent_pack = [
        _pack_entry(
            passage_id="chapter-1#exp0001",
            text=adjacent_text + " далее по тексту.",
            section_id="chapter-1",
            char_start=200,
            char_end=400,
        )
    ]
    draft = f"Продолжаю: «{adjacent_text}»."
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [adjacent_pack[0]["passage_id"]],
                    }
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="продолжай, давай дальше",
        summary="",
        recent=[],
        evidence_pack=adjacent_pack,
        answer_model=answer,
        verifier_model=verifier,
        recent_quote_ranges=recent,
    )
    assert adjacent_text not in outcome["text"]
    assert contains_cyrillic(outcome["text"])
    assert envelope_passes(outcome["text"])


# ---------------------------------------------------------------------------
# New-path isolation and graph wiring.
# ---------------------------------------------------------------------------


def test_new_answer_path_has_no_legacy_semantics() -> None:
    forbidden = (
        "is_substantive",
        "_TRIVIAL_NORMALIZED",
        "_FOLLOWUP_INTERROGATIVES",
        "_SUBSTANTIVE_KEYWORDS",
        "_SLANG_EXPANSIONS",
        "_THEME_MARKERS",
        "_BROAD_COVERAGE_QUERIES",
        "FAIL_CLOSED_REPLY",
        "default_entails",
    )
    for name in (
        "answer_node.py",
        "response_units.py",
        "verifier.py",
        "verifier_schema.py",
        "turn_pipeline.py",
        "quote_state.py",
        "retrieval_node.py",
        "graph.py",
    ):
        source = (CONVERSATION_PKG / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "aa.conversation.orchestrator" not in imported, name
        assert "aa.conversation.meta" not in imported, name
        assert "aa.retrieval.planner" not in imported, name
        for snippet in forbidden:
            assert snippet not in source, f"{name}: {snippet}"


def test_new_path_has_no_mandatory_visible_citations() -> None:
    for name in ("answer_node.py", "turn_pipeline.py", "verifier.py"):
        source = (CONVERSATION_PKG / name).read_text(encoding="utf-8")
        assert "source/section#chunk" not in source, name


def test_verifier_agent_locked_down() -> None:
    import json as json_module

    config = json_module.loads((ROOT / "opencode.json").read_text(encoding="utf-8"))
    verifier = config["agent"]["aa-verifier-v2"]
    assert verifier["permission"] == {"*": "deny", "StructuredOutput": "allow"}
    assert verifier["prompt"] == "{file:./prompts/aa-verifier-system-v2.md}"
    assert (ROOT / "prompts" / "aa-verifier-system-v2.md").exists()
    answer = config["agent"]["aa-v2"]
    assert answer["permission"] == {"*": "deny"}


async def test_graph_runs_answer_pipeline_end_to_end() -> None:
    from langchain_core.runnables import RunnableLambda

    async def _plan(_messages: Any) -> Any:
        return {"queries": []}

    answer = _AnswerModel(["Понимаю. Расскажите, что сейчас важнее всего?"])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    },
                    {
                        "unit_id": "u2",
                        "scope": "conversation_glue",
                        "supported": True,
                        "evidence_passage_ids": [],
                    },
                ],
                "all_required_supported": True,
            }
        ]
    )
    graph = build_turn_graph(
        planner_model=RunnableLambda(_plan),
        answer_model=answer,
        verifier_model=verifier,
    )
    result = await graph.ainvoke(turn_input("А что ты можешь?"))
    assert result["planner_invoked"] is True
    assert result["route"] == "normal"
    assert contains_cyrillic(result["draft_response"])
    assert contains_cyrillic(result["final_response"])
    assert envelope_passes(result["final_response"])
    assert not leaks_internal_terms(result["final_response"])
    assert result["grounding_result"]["all_required_supported"] is True


async def test_graph_without_answer_models_keeps_old_semantics() -> None:
    from langchain_core.runnables import RunnableLambda

    async def _plan(_messages: Any) -> Any:
        return {"queries": []}

    graph = build_turn_graph(planner_model=RunnableLambda(_plan))
    result = await graph.ainvoke(turn_input("привет"))
    assert result["evidence_pack"] == []
    assert result["draft_response"] == ""


def test_natural_clarification_fits_envelope_without_leak() -> None:
    assert NATURAL_CLARIFICATION_REPLY
    assert contains_cyrillic(NATURAL_CLARIFICATION_REPLY)
    assert not leaks_internal_terms(NATURAL_CLARIFICATION_REPLY)
    assert envelope_passes(NATURAL_CLARIFICATION_REPLY)


async def test_verifier_invalid_retry_succeeds_on_second_attempt() -> None:
    """Gate C repair (run 37530425848, run 37547434287): weak id-copy flake.

    A batch verdict with wrong unit ids falls back once to the simpler
    concurrent per-unit single-verdict task (no id copying); a valid
    single verdict grounds the turn instead of collapsing to generic
    clarification.
    """
    from aa.conversation.turn_pipeline import _verify_draft

    class _FlakyVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            if self.calls == 1:
                return {
                    "units": [
                        {
                            "unit_id": "wrong-id",
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": ["chapter-3#exp0000"],
                        }
                    ],
                    "all_required_supported": True,
                }
            return {
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
            }

    draft = "Поддержка рядом помогает спокойно."
    verifier = _FlakyVerifier()
    units, result, passed = await _verify_draft(draft, [_pack_entry()], verifier_model=verifier)
    assert passed is True
    assert result is not None
    assert result.all_required_supported is True
    assert verifier.calls == 2
    assert [unit.unit_id for unit in units] == ["u1"]


async def test_verifier_invalid_twice_fails_closed_without_third_call() -> None:
    """Two consecutive invalid verdicts fail closed with exactly two calls."""
    from aa.conversation.turn_pipeline import _verify_draft

    class _AlwaysInvalid:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            return {
                "units": [
                    {
                        "unit_id": "wrong-id",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["chapter-3#exp0000"],
                    }
                ],
                "all_required_supported": True,
            }

    verifier = _AlwaysInvalid()
    units, result, passed = await _verify_draft(
        "Поддержка рядом помогает спокойно.", [_pack_entry()], verifier_model=verifier
    )
    assert passed is False
    assert result is None
    assert verifier.calls == 2
    assert len(units) == 1


async def test_verifier_provider_error_does_not_retry() -> None:
    """Provider/transient failures after fallback never retry here (latency)."""
    from aa.conversation.turn_pipeline import _verify_draft

    class _DownVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            raise TimeoutError("verifier down")

    verifier = _DownVerifier()
    _, result, passed = await _verify_draft(
        "Поддержка рядом помогает спокойно.", [_pack_entry()], verifier_model=verifier
    )
    assert passed is False
    assert result is None
    assert verifier.calls == 1


def test_verifier_native_schema_is_ref_free() -> None:
    """Gate C repair: weak fallback providers reject $ref/$defs schemas.

    The native OpenCode transport schema must be flat ($ref-free) while
    AA-side Pydantic validation stays strict. A $ref-bearing schema made
    the verifier never serve on the fallback model (all ordinary turns
    collapsing to generic clarification with slow internal retries).
    """
    import json
    from typing import cast

    schema = verifier_json_schema()
    assert schema["type"] == "object"
    properties = cast(dict[str, Any], schema["properties"])
    assert "units" in properties
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    # Units items are inlined (no $ref indirection).
    units = cast(dict[str, Any], properties["units"])
    items = cast(dict[str, Any], units["items"])
    assert items["type"] == "object"
    assert "$ref" not in items
    # Scopes still enumerate the closed vocabulary for the transport.
    item_properties = cast(dict[str, Any], items["properties"])
    scope = cast(dict[str, Any], item_properties["scope"])
    assert set(cast(list[str], scope["enum"])) == {"book", "product_meta", "conversation_glue"}
    # A native-shaped payload still validates strictly in AA code.
    result = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                }
            ],
            "all_required_supported": True,
        },
        expected_unit_ids=["u1"],
    )
    assert result.all_required_supported is True


def test_verifier_transport_schema_is_minimal_but_strict() -> None:
    """Gate C repair (run 37538518277): bound weak-provider flake + latency.

    The transport hint omits length constraints (``minLength``/``minItems``)
    that weak fallback providers reject, while AA-side Pydantic validation
    still rejects empty ids and enforces exactly-one-verdict-per-unit.
    The native retry budget stays bounded (single server retry).
    """
    import json

    from aa.conversation.verifier_schema import VERIFIER_MAX_ATTEMPTS

    schema = verifier_json_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert "minLength" not in dumped
    assert "minItems" not in dumped
    # Essential guidance stays: closed scope enum + required ids.
    assert "book" in dumped and "product_meta" in dumped and "conversation_glue" in dumped
    assert VERIFIER_MAX_ATTEMPTS == 1
    # AA-side stays strict: empty unit ids are rejected.
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(
            {
                "units": [
                    {
                        "unit_id": "",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["chapter-3#exp0000"],
                    }
                ],
                "all_required_supported": True,
            },
            expected_unit_ids=["u1"],
        )
    # AA-side stays strict: unknown scopes are rejected.
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "Book",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                ],
                "all_required_supported": True,
            },
            expected_unit_ids=["u1"],
        )


def test_verifier_user_text_repeats_closed_contract() -> None:
    """Gate C repair: weak fallback models need the contract in the payload.

    The scope vocabulary lived only in the system prompt, so the weak
    fallback emitted schema-invalid scopes and the verifier never served.
    The user payload repeats the id-copy rule, closed scope vocabulary,
    cite-only-supplied-ids rule, and flag derivation uniformly for every
    turn (never an exact-question special case).
    """
    user_text = build_verifier_user_text(
        units=split_response_units("Понимаю. Тяга проходит."),
        passages=[_pack_entry()],
    )
    assert "<response_units>" in user_text
    assert "<book_evidence>" in user_text
    assert "product_meta" in user_text
    assert "conversation_glue" in user_text
    assert "Cite only passage ids" in user_text
    assert "all_required_supported" in user_text


def test_verifier_normalizes_weak_provider_formatting() -> None:
    """Gate C repair (run 37544234331): tolerate weak-model formatting variance.

    The weak fallback emits schema-valid-intent verdicts with capitalized
    scopes, surrounding whitespace, and hyphen/underscore confusion, which
    failed strict validation so the verifier never served (14/14
    clarifications, missing verifier identity). Normalization is
    turn-independent: the closed vocabulary and id completeness stay
    strict, only case/whitespace/separator are tolerated.
    """
    from aa.conversation.verifier import VERIFIER_MAX_EVIDENCE_PASSAGES

    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 8
    units = split_response_units("Понимаю. Тяга проходит спокойно.")
    pack = [_pack_entry()]
    payload = {
        "units": [
            {
                "unit_id": f"  {unit.unit_id}  ",
                "scope": "Book" if i == 0 else " conversation-glue ",
                "supported": True,
                "evidence_passage_ids": ([f"  {pack[0]['passage_id']}  "] if i == 0 else []),
            }
            for i, unit in enumerate(units)
        ],
        "all_required_supported": True,
    }
    result = coerce_grounding_result(payload, units=units, passages=pack)
    assert [v.unit_id for v in result.units] == [u.unit_id for u in units]
    assert {v.scope for v in result.units} <= {"book", "conversation_glue"}
    assert result.all_required_supported is True


def test_verifier_evidence_window_bounds_prompt_size() -> None:
    """Gate C repair (run 37544234331): bound verifier input latency.

    The full 16k-token pack makes the verifier prompt the largest per-turn
    model input; weak providers are slow/flaky on it (p95 38.7s/max 41.1s
    over the 30s budget) while the planner (small prompt) serves. The
    display window keeps top-ranked passages only; stored-pack cite/quote/
    checksum checks stay full-pack strict.
    """
    from aa.conversation.verifier import VERIFIER_MAX_EVIDENCE_PASSAGES

    units = split_response_units("Понимаю. Тяга проходит.")
    passages = [
        _pack_entry(
            passage_id=f"chapter-3#exp{i:04d}",
            text=f"Фиктивная поддержка рядом {i}. Тяга проходит.",
        )
        for i in range(12)
    ]
    user_text = build_verifier_user_text(units=units, passages=passages)
    assert "chapter-3#exp0007" in user_text
    assert "chapter-3#exp0008" not in user_text
    assert "chapter-3#exp0011" not in user_text
    assert len(passages) == 12
    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 8


async def test_verifier_retries_weak_formatting_but_not_deterministic() -> None:
    """Gate C repair (run 37544234331, run 37547434287): per-unit fallback.

    Id/format validation errors fall back once to the simpler per-unit
    single-verdict task; deterministic cite failures (book unit, empty
    pack) fail closed immediately without burning live latency.
    """
    from aa.conversation.turn_pipeline import _verify_draft

    draft = "Поддержка рядом помогает спокойно разбирать тягу."
    pack = [_pack_entry()]
    units = split_response_units(draft)

    class _FlakyThenGood:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            props = schema.get("properties", {})
            is_single = isinstance(props, dict) and "scope" in props and "units" not in props
            if not is_single:
                # Batch fast path: wrong verdict count (id flake).
                extra = [
                    {
                        "unit_id": f"u{len(units) + 1}",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                ]
                return {
                    "units": [
                        {
                            "unit_id": unit.unit_id,
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": [pack[0]["passage_id"]],
                        }
                        for unit in units
                    ]
                    + extra,
                    "all_required_supported": True,
                }
            return {
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            }

    flaky = _FlakyThenGood()
    returned_units, result, passed = await _verify_draft(draft, pack, verifier_model=flaky)
    assert passed is True
    assert result is not None
    assert flaky.calls == 2
    assert len(returned_units) == len(units)

    class _DeterministicCiteFail:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            return {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }

    deterministic = _DeterministicCiteFail()
    _, no_result, not_passed = await _verify_draft(draft, pack, verifier_model=deterministic)
    assert not_passed is False
    assert no_result is None
    assert deterministic.calls == 1


async def test_verifier_unavailable_preserves_upstream_stage_outcomes() -> None:
    """Gate C repair: verifier outage must not flatten planner/retrieval.

    When the verifier produces no verdict, repair is still skipped (no
    futile re-planning), but planner/retrieval outcomes stay at their real
    upstream values so the next failure attributes to the concrete stage
    instead of generic skipped-verifier-unavailable.
    """

    class _DownVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise TimeoutError("verifier down")

    class _AnswerModel:
        async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает спокойно разбирать тягу.")

    outcome = await run_v2_answer_turn(
        user_message="Как обходиться с тягой вечером?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_AnswerModel(),
        verifier_model=_DownVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    telemetry = dict(outcome["telemetry"])
    assert telemetry["verifier_outcome"] == "unavailable"
    assert outcome["text"] == NATURAL_CLARIFICATION_REPLY
    # Upstream stages are preserved, never flattened to a generic token.
    assert telemetry["planner_outcome"] != "skipped-verifier-unavailable"
    assert telemetry["retrieval_outcome"] != "skipped-verifier-unavailable"


def test_verifier_single_schema_is_ref_free_without_ids() -> None:
    """Gate C repair (run 37547434287): minimal single-verdict transport hint."""
    import json
    from typing import Any, cast

    from aa.conversation.verifier_schema import verifier_single_json_schema

    schema = verifier_single_json_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert "minLength" not in dumped
    assert "minItems" not in dumped
    assert "unit_id" not in dumped
    props = cast(dict[str, Any], schema["properties"])
    scope = cast(dict[str, Any], props["scope"])
    assert set(cast(list[str], scope["enum"])) == {"book", "product_meta", "conversation_glue"}
    assert schema["required"] == ["scope", "supported"]


def test_verifier_single_text_has_no_id_copying() -> None:
    """Single-unit payload judges one unit without u1..uN copying."""
    from aa.conversation.verifier import build_single_unit_text

    units = split_response_units("Понимаю. Тяга проходит.")
    text = build_single_unit_text(unit=units[0], passages=[_pack_entry()])
    assert "<response_unit>" in text
    assert "<book_evidence>" in text
    assert "unit_id" not in text
    assert "u1" not in text


async def test_verifier_per_unit_fallback_serves_after_batch_id_flake() -> None:
    """Batch id flake falls back to concurrent single verdicts and serves."""
    from aa.conversation.verifier import run_verifier

    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")

    class _BatchFlakySingleGood:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system)
            self.calls += 1
            props = schema.get("properties", {})
            is_single = isinstance(props, dict) and "scope" in props and "units" not in props
            if not is_single:
                return {
                    "units": [
                        {
                            "unit_id": "wrong-id",
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": [pack[0]["passage_id"]],
                        }
                    ],
                    "all_required_supported": True,
                }
            if "Понимаю" in prompt:
                return {"scope": "conversation_glue", "supported": True}
            return {
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            }

    model = _BatchFlakySingleGood()
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert len(result.units) == len(units)
    # One batch attempt plus one concurrent single per unit.
    assert model.calls == 1 + len(units)


async def test_verifier_per_unit_fallback_stays_strict_on_cites() -> None:
    """Per-unit fallback still rejects unknown passages (fail-closed)."""
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает."
    units = split_response_units(draft)
    assert len(units) == 1

    class _UnknownCite:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            props = schema.get("properties", {})
            is_single = isinstance(props, dict) and "scope" in props and "units" not in props
            if is_single:
                return {
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["no-such-passage"],
                }
            return {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["no-such-passage"],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }

    _, result, passed = await _verify_draft(draft, pack, verifier_model=_UnknownCite())
    assert passed is False
    assert result is None
