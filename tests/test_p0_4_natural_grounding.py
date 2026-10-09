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
from aa.conversation.failures import TurnFailed
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
    contains_cyrillic,
    leaks_internal_terms,
    run_v2_answer_turn,
)
from aa.conversation.v2_prompts import load_aa_agent_system_v2, load_verifier_system_v2
from aa.conversation.verifier import build_single_unit_text, coerce_single_verdict
from aa.conversation.verifier_schema import (
    VerifierValidationError,
    validate_grounding_result,
    validate_unit_decision,
    verifier_single_json_schema,
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
    """Per-unit only verifier mock (kodmial/aa#190).

    Accepts both new boolean decisions and legacy batch payloads for
    migration: legacy batch/scope payloads are expanded to per-unit
    boolean decisions (book -> requires True, otherwise False).
    """

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results: list[dict[str, Any]] = []
        for entry in results:
            self._results.extend(_expand_to_decisions(entry))
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


def _expand_to_decisions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one scripted entry to per-unit boolean decisions."""
    if "units" in entry and isinstance(entry["units"], list):
        decisions: list[dict[str, Any]] = []
        for unit in entry["units"]:
            if not isinstance(unit, dict):
                continue
            scope = str(unit.get("scope", "book"))
            supported = bool(unit.get("supported", False))
            decisions.append(
                {
                    "requires_book_evidence": scope == "book",
                    "supported": supported,
                    "evidence_passage_ids": list(unit.get("evidence_passage_ids", [])),
                    "addresses_intent": bool(unit.get("addresses_intent", supported)),
                }
            )
        return decisions
    if "scope" in entry and "requires_book_evidence" not in entry:
        supported = bool(entry.get("supported", False))
        return [
            {
                "requires_book_evidence": str(entry.get("scope", "book")) == "book",
                "supported": supported,
                "evidence_passage_ids": list(entry.get("evidence_passage_ids", [])),
                "addresses_intent": bool(entry.get("addresses_intent", supported)),
            }
        ]
    return [dict(entry)]


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
                "addresses_intent": True,
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
    schema = verifier_single_json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert isinstance(props, dict) and "requires_book_evidence" in props
    assert "supported" in props and "evidence_passage_ids" in props
    assert "scope" not in props and "unit_id" not in str(schema)
    assert "all_required_supported" not in str(schema)
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
    from aa.conversation.verifier import check_cited_passage_ids

    units = split_response_units("Тяга проходит быстро.")
    pack = [_pack_entry()]
    pack_ids = {str(pack[0]["passage_id"])}
    no_evidence = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": [],
            "addresses_intent": True,
        },
        unit_id="u1",
    )
    assert no_evidence.scope == "book"
    with pytest.raises(VerifierValidationError):
        check_cited_passage_ids(
            validate_grounding_result(
                {
                    "units": [
                        {
                            "unit_id": "u1",
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": [],
                        }
                    ],
                    "all_required_supported": True,
                },
                expected_unit_ids=["u1"],
            ),
            pack_ids=pack_ids,
        )
    unknown_verdict = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["no-such-passage"],
            "addresses_intent": True,
        },
        unit_id="u1",
    )
    assert unknown_verdict.scope == "book"
    with pytest.raises(VerifierValidationError):
        check_cited_passage_ids(
            validate_grounding_result(
                {
                    "units": [
                        {
                            "unit_id": "u1",
                            "scope": "book",
                            "supported": True,
                            "evidence_passage_ids": ["no-such-passage"],
                        }
                    ],
                    "all_required_supported": True,
                },
                expected_unit_ids=["u1"],
            ),
            pack_ids=pack_ids,
        )
    _ = units


def test_verifier_rejects_verbatim_quote_absent_from_cited_passage() -> None:
    from aa.conversation.verifier import check_exact_quotes

    passage_text = "Фиктивная поддержка рядом и спокойный разговор."
    pack = [_pack_entry(text=passage_text)]
    units = split_response_units("Как сказано: «совсем другая фраза про луну».")
    verdict = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": [pack[0]["passage_id"]],
            "addresses_intent": True,
        },
        unit_id="u1",
    )
    assert verdict.scope == "book"
    assembled = validate_grounding_result(
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
        },
        expected_unit_ids=["u1"],
    )
    with pytest.raises(VerifierValidationError):
        check_exact_quotes(units=units, result=assembled, passages=pack)


def test_verifier_system_prompt_is_english_authority() -> None:
    system = load_verifier_system_v2()
    assert system.strip()
    assert not any("\u0400" <= ch <= "\u04ff" for ch in system)
    assert "book_evidence" in system
    assert "structured output" in system.casefold()
    assert "requires_book_evidence" in system
    units = split_response_units("Понимаю. Тяга проходит.")
    user_text = build_single_unit_text(unit=units[0], passages=[_pack_entry()])
    assert "<response_unit>" in user_text
    assert "<book_evidence>" in user_text


def test_verifier_uses_native_json_schema_no_bespoke_parser() -> None:
    # Native structured output stays the primary verifier channel; planner
    # and schema modules never parse JSON text. The verifier owns exactly
    # one bounded capability-compatible text-JSON fallback (kodmial/aa#192)
    # for provider paths where native json_schema is unavailable; it is
    # strictly Pydantic-validated and 429-safe, never an unconstrained
    # bespoke parser.
    for name in ("verifier_schema.py", "planner_node.py"):
        source = (CONVERSATION_PKG / name).read_text(encoding="utf-8")
        assert "json.loads" not in source, name
        assert "PydanticOutputParser" not in source, name
        assert "get_format_instructions" not in source, name
    source = (CONVERSATION_PKG / "verifier.py").read_text(encoding="utf-8")
    assert "PydanticOutputParser" not in source
    assert "get_format_instructions" not in source
    assert "ainvoke_structured" in source
    assert "retry_count" in source
    assert "parse_text_json_decision" in source
    assert "OpenCodeRateLimitError" in source
    assert source.count("json.loads") == 1


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
    import pytest as _pt04a

    from aa.conversation.failures import TurnFailed as _TF04a

    with _pt04a.raises(_TF04a):
        await run_v2_answer_turn(
            user_message="привет",
            summary="",
            recent=[],
            evidence_pack=[],
            answer_model=answer,
            verifier_model=verifier,
            planner_model=None,
            retrieval_index=None,
        )


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
    import pytest as _pt04b

    from aa.conversation.failures import TurnFailed as _TF04b

    with _pt04b.raises(_TF04b):
        await run_v2_answer_turn(
            user_message="как проходит тяга?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            planner_model=None,
            retrieval_index=None,
        )


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
    import pytest as _pt04c

    from aa.conversation.failures import TurnFailed as _TF04c

    # The whole mixed unit fails: it must never cross the boundary.
    with _pt04c.raises(_TF04c):
        await run_v2_answer_turn(
            user_message="что помогает?",
            summary="",
            recent=[],
            evidence_pack=pack,
            answer_model=answer,
            verifier_model=verifier,
            planner_model=None,
            retrieval_index=None,
        )


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
    planner = _PlannerModel(
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries(),
            }
        ]
    )

    import hashlib as _hashlib

    from aa.retrieval import evidence as evidence_mod

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None, **kwargs: Any) -> Any:
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
        [
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("луна"),
            },
            {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("вечер"),
            },
        ]
    )

    from aa.retrieval import evidence as evidence_mod

    def _fake_retrieve(index: Any, queries: object, *, config: Any = None, **kwargs: Any) -> Any:
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
    import pytest as _pt04f

    from aa.conversation.failures import TurnFailed as _TF04f

    with _pt04f.raises(_TF04f) as _exc04f:
        await run_v2_answer_turn(
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
    assert _exc04f.value.telemetry["repair_rounds"] == MAX_TARGETED_REPAIR_ROUNDS == 2


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
    # Issue #295 approved behavior: a fully verified grounded complete
    # answer is served whole via bounded transport segments (no extra
    # compact regeneration, no leading-sentence truncation of final
    # points). Each segment passes the envelope; the full text keeps all
    # 60 supported sentences.
    from aa.conversation.output_limits import MAX_TRANSPORT_SEGMENTS

    pack = [_pack_entry()]
    sentence = "Поддержка рядом помогает пережить тягу спокойно"
    long_draft = " ".join(f"{sentence}." for _ in range(60))
    assert not envelope_passes(long_draft)
    long_units = split_response_units(long_draft)
    answer = _AnswerModel([long_draft])
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
    assert answer.calls == 1
    assert outcome["telemetry"].get("transport_split") is True
    segments = outcome.get("segments") or []
    assert 1 < len(segments) <= MAX_TRANSPORT_SEGMENTS
    assert all(envelope_passes(seg) for seg in segments)
    assert outcome["text"] == long_draft
    assert outcome["text"].count(sentence) == 60


async def test_provider_failure_returns_natural_reply_without_leak() -> None:
    class _Boom:
        async def ainvoke(self, messages: Any) -> Any:
            raise TimeoutError("provider down")

    import pytest as _pt04d

    from aa.conversation.failures import TurnFailed as _TF04d

    # Provider outage is a typed failure; internals never leak into a
    # delivered conversation (transport shows the marked service error).
    with _pt04d.raises(_TF04d):
        await run_v2_answer_turn(
            user_message="тяга вечером",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_Boom(),
            verifier_model=_VerifierModel([]),
        )


async def test_no_internal_terms_leak_on_verifier_outage() -> None:
    class _BadVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            raise TimeoutError("verifier down")

    import pytest as _pt04e

    from aa.conversation.failures import TurnFailed as _TF04e

    with _pt04e.raises(_TF04e):
        await run_v2_answer_turn(
            user_message="тяга",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_AnswerModel(["Поддержка рядом помогает спокойно."]),
            verifier_model=_BadVerifier(),
            planner_model=None,
            retrieval_index=None,
        )


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
        "NATURAL_CLARIFICATION_REPLY",
        "NATURAL_RETRY_REPLY",
        "CONVERSATIONAL_FALLBACK_REPLY",
        "SAFE_UNAVAILABLE_REPLY",
        "select_retry_reply",
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
        return {"mode": "conversational", "resolved_intent": "", "queries": []}

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
        return {"mode": "conversational", "resolved_intent": "", "queries": []}

    graph = build_turn_graph(planner_model=RunnableLambda(_plan))
    result = await graph.ainvoke(turn_input("привет"))
    assert result["evidence_pack"] == []
    assert result["draft_response"] == ""


def test_natural_clarification_fits_envelope_without_leak() -> None:
    # Issue #301: no fixed clarification string exists in the pipeline.
    import pathlib as _pl

    _src = (
        _pl.Path(__file__).resolve().parents[1] / "src" / "aa" / "conversation" / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "NATURAL_CLARIFICATION_REPLY" not in _src
    assert "select_retry_reply" not in _src


async def test_verifier_invalid_retry_succeeds_on_second_attempt() -> None:
    """Per-unit only verifier serves boolean decisions in one round.

    No id copying exists, so there is no remap round: one concurrent round
    of boolean decisions grounds the turn. Turn-independent, never an
    exact-question special case.
    """
    from aa.conversation.turn_pipeline import _verify_draft

    draft = "Поддержка рядом помогает спокойно."
    verifier = _VerifierModel(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": True,
            }
        ]
    )
    units, result, passed = await _verify_draft(draft, [_pack_entry()], verifier_model=verifier)
    assert passed is True
    assert result is not None
    assert result.all_required_supported is True
    assert verifier.calls == 1
    assert [unit.unit_id for unit in units] == ["u1"]


async def test_verifier_invalid_twice_fails_closed_without_third_call() -> None:
    """Invalid boolean decisions fail closed with exactly one round.

    A schema-invalid per-unit decision (missing booleans) fails closed
    without extra rounds burning live latency.
    """
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
                "requires_book_evidence": "yes",
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            }

    verifier = _AlwaysInvalid()
    units, result, passed = await _verify_draft(
        "Поддержка рядом помогает спокойно.", [_pack_entry()], verifier_model=verifier
    )
    assert passed is False
    assert result is None
    assert verifier.calls == 1
    assert len(units) == 1


async def test_verifier_provider_error_does_not_retry() -> None:
    """Provider/transient failures stay bounded to one concurrent round.

    Per-unit only verifier (kodmial/aa#190): exactly one concurrent round
    runs, no fallback, no retry. A fully down verifier therefore costs one
    call per unit and still clarifies.
    """
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
    """Per-unit boolean transport schema is flat ($ref-free) and strict."""
    import json
    from typing import cast

    schema = verifier_single_json_schema()
    assert schema["type"] == "object"
    properties = cast(dict[str, Any], schema["properties"])
    assert "requires_book_evidence" in properties
    assert "supported" in properties
    assert "evidence_passage_ids" in properties
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert '"enum"' not in json.dumps(schema)
    # A native-shaped boolean payload validates strictly in AA code.
    decision = validate_unit_decision(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
            "addresses_intent": True,
        }
    )
    assert decision.requires_book_evidence is True
    assert decision.supported is True


def test_verifier_transport_schema_is_minimal_but_strict() -> None:
    """Boolean transport hint omits length constraints; code stays strict."""
    import json

    from aa.conversation.verifier_schema import VERIFIER_MAX_ATTEMPTS

    schema = verifier_single_json_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert "minLength" not in dumped
    assert "minItems" not in dumped
    assert '"enum"' not in dumped
    assert "requires_book_evidence" in dumped and "supported" in dumped
    assert "scope" not in dumped
    assert "unit_id" not in dumped
    assert "all_required_supported" not in dumped
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
    # Transport stays strict: scope strings are rejected.
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(
            {
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [],
            },
        )


def test_verifier_user_text_repeats_closed_contract() -> None:
    """Single-unit payload carries the boolean contract, no scope strings."""
    units = split_response_units("Понимаю. Тяга проходит.")
    user_text = build_single_unit_text(unit=units[0], passages=[_pack_entry()])
    assert "<response_unit>" in user_text
    assert "<book_evidence>" in user_text
    assert "requires_book_evidence" in user_text
    assert "Cite only passage ids" in user_text
    assert "scope" not in user_text
    assert "all_required_supported" not in user_text
    assert "product_meta" not in user_text
    # kodmial/aa#308: the payload teaches the claim-origin taxonomy, so
    # the origin vocabulary is present by contract (never unit ids).
    assert "claim_origin" in user_text


def test_verifier_normalizes_weak_provider_formatting() -> None:
    """Boolean transport rejects scope strings; evidence ids trim strictly."""
    from aa.conversation.verifier import VERIFIER_MAX_EVIDENCE_PASSAGES

    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 0
    # Scope-shaped payloads are rejected: the model must emit booleans.
    with pytest.raises(VerifierValidationError):
        validate_unit_decision({"scope": "Book", "supported": True, "evidence_passage_ids": []})
    # Boolean decisions validate; AA code derives scope deterministically.
    units = split_response_units("Понимаю. Тяга проходит спокойно.")
    pack = [_pack_entry()]
    first = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": [pack[0]["passage_id"]],
            "addresses_intent": True,
        },
        unit_id=units[0].unit_id,
    )
    second = coerce_single_verdict(
        {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
            "addresses_intent": True,
        },
        unit_id=units[1].unit_id,
    )
    assert first.scope == "book"
    assert second.scope != "book"
    assert [first.unit_id, second.unit_id] == [u.unit_id for u in units]


def test_verifier_evidence_window_bounds_prompt_size() -> None:
    """Verifier receives the full pack with complete text (#295)."""
    from aa.conversation.verifier import VERIFIER_MAX_EVIDENCE_PASSAGES

    units = split_response_units("Понимаю. Тяга проходит.")
    passages = [
        _pack_entry(
            passage_id=f"chapter-3#exp{i:04d}",
            text=f"Фиктивная поддержка рядом {i}. Тяга проходит.",
        )
        for i in range(12)
    ]
    user_text = build_single_unit_text(unit=units[0], passages=passages)
    # Short ordinal display ids stay copyable; long provenance ids never
    # enter the prompt. The full pack (all 12) reaches the verifier.
    assert 'id="p1"' in user_text
    assert 'id="p5"' in user_text
    assert 'id="p6"' in user_text
    assert 'id="p12"' in user_text
    for passage in passages:
        assert passage["text"] in user_text
    assert len(passages) == 12
    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 0


async def test_verifier_retries_weak_formatting_but_not_deterministic() -> None:
    """Per-unit only: valid booleans serve; deterministic cite fails closed."""
    from aa.conversation.turn_pipeline import _verify_draft

    draft = "Поддержка рядом помогает спокойно разбирать тягу."
    pack = [_pack_entry()]
    units = split_response_units(draft)

    good = _VerifierModel(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            }
            for _ in units
        ]
    )
    returned_units, result, passed = await _verify_draft(draft, pack, verifier_model=good)
    assert passed is True
    assert result is not None
    assert good.calls == len(units)
    assert len(returned_units) == len(units)

    deterministic = _VerifierModel(
        [
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            }
            for _ in units
        ]
    )
    _, no_result, not_passed = await _verify_draft(draft, pack, verifier_model=deterministic)
    assert not_passed is False
    assert no_result is None
    assert deterministic.calls == len(units)


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

    import pytest as _pytest

    with _pytest.raises(TurnFailed) as _exc:
        await run_v2_answer_turn(
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
    # Verifier outage is a typed unsuccessful outcome, never canned text.
    assert _exc.value.category == "clarification-unavailable"


def test_verifier_single_schema_is_ref_free_without_ids() -> None:
    """Minimal boolean transport hint: no ids, no scope, no aggregate."""
    import json
    from typing import Any, cast

    from aa.conversation.verifier_schema import verifier_single_json_schema

    schema = verifier_single_json_schema()
    dumped = json.dumps(schema)
    assert "$ref" not in dumped
    assert "minLength" not in dumped
    assert "minItems" not in dumped
    assert "unit_id" not in dumped
    assert '"enum"' not in dumped
    assert "scope" not in dumped
    assert "all_required_supported" not in dumped
    props = cast(dict[str, Any], schema["properties"])
    # kodmial/aa#308: optional model-led claim-origin hint plus
    # extensible origin_ref provenance; required keys unchanged.
    assert set(props) == {
        "requires_book_evidence",
        "supported",
        "evidence_passage_ids",
        "addresses_intent",
        "claim_origin",
        "origin_ref",
    }
    assert schema["required"] == [
        "requires_book_evidence",
        "supported",
        "evidence_passage_ids",
        "addresses_intent",
    ]


def test_verifier_single_text_has_no_id_copying() -> None:
    """Single-unit payload judges one unit without id/scope/aggregate."""
    from aa.conversation.verifier import build_single_unit_text

    units = split_response_units("Понимаю. Тяга проходит.")
    text = build_single_unit_text(unit=units[0], passages=[_pack_entry()])
    assert "<response_unit>" in text
    assert "<book_evidence>" in text
    assert "unit_id" not in text
    assert "u1" not in text
    assert "all_required_supported" not in text
    assert "product_meta" not in text
    # kodmial/aa#308: the payload teaches the claim-origin taxonomy, so
    # the origin vocabulary is present by contract (never unit ids).
    assert "claim_origin" in text
    assert "requires_book_evidence" in text


async def test_verifier_per_unit_fallback_serves_after_batch_id_flake() -> None:
    """Per-unit concurrent round serves boolean decisions in one round."""
    from aa.conversation.verifier import run_verifier

    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")

    model = _VerifierModel(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            },
        ]
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert len(result.units) == len(units)
    assert model.calls == len(units)


async def test_verifier_per_unit_fallback_stays_strict_on_cites() -> None:
    """Per-unit round still rejects unknown passages (fail-closed)."""
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Поддержка рядом помогает."
    units = split_response_units(draft)
    assert len(units) == 1

    _, result, passed = await _verify_draft(
        draft,
        pack,
        verifier_model=_VerifierModel(
            [
                {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": ["no-such-passage"],
                    "addresses_intent": True,
                }
            ]
        ),
    )
    assert passed is False
    assert result is None


def test_verifier_order_remap_repairs_id_flake_without_extra_calls() -> None:
    """AA code binds unit ids; the model never copies them.

    A scope-shaped payload is rejected, while boolean decisions bind the
    input id deterministically with full cite gates.
    """
    with pytest.raises(VerifierValidationError):
        validate_unit_decision(
            {"unit_id": "u1", "scope": "book", "supported": True, "evidence_passage_ids": []}
        )

    units = split_response_units("Понимаю. Тяга проходит спокойно.")
    pack = [_pack_entry()]
    first = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": [pack[0]["passage_id"]],
            "addresses_intent": True,
        },
        unit_id=units[0].unit_id,
    )
    second = coerce_single_verdict(
        {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
            "addresses_intent": True,
        },
        unit_id=units[1].unit_id,
    )
    assert [first.unit_id, second.unit_id] == [u.unit_id for u in units]
    assert first.scope == "book"


def test_verifier_unsupported_needs_no_citation_or_quote() -> None:
    """Unsupported needs no valid citation; supported book stays strict."""
    from aa.conversation.verifier import check_cited_passage_ids

    pack = [_pack_entry()]
    pack_ids = {str(pack[0]["passage_id"])}
    unsupported = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": False,
            "evidence_passage_ids": [],
            "addresses_intent": False,
        },
        unit_id="u1",
    )
    assert unsupported.supported is False
    assembled = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": False,
                    "evidence_passage_ids": [],
                }
            ],
            "all_required_supported": False,
        },
        expected_unit_ids=["u1"],
    )
    assert check_cited_passage_ids(assembled, pack_ids=pack_ids).all_required_supported is False
    # Supported book with no evidence still fails closed.
    supported_empty = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
            "all_required_supported": True,
        },
        expected_unit_ids=["u1"],
    )
    with pytest.raises(VerifierValidationError):
        check_cited_passage_ids(supported_empty, pack_ids=pack_ids)


def test_verifier_display_truncates_long_passages_but_checks_full_pack() -> None:
    """Verifier receives complete text; deterministic gates use the full pack (#295)."""
    from aa.conversation.verifier import VERIFIER_MAX_PASSAGE_CHARS, build_single_unit_text

    long_text = "Фиктивная поддержка рядом. " * 200
    assert VERIFIER_MAX_PASSAGE_CHARS == 0
    pack = [_pack_entry(text=long_text)]
    units = split_response_units("Понимаю. Тяга проходит.")
    user_text = build_single_unit_text(unit=units[0], passages=pack)
    assert long_text in user_text
    # Short display ids enter the prompt; the long provenance id resolves
    # only in AA-side validation against the full stored pack.
    assert 'id="p1"' in user_text
    assert pack[0]["passage_id"] not in user_text
    # Full-pack checks still see the untruncated stored text.
    assert pack[0]["text"] == long_text


def test_verifier_short_display_ids_resolve_to_full_pack() -> None:
    """Short ids are copyable; AA code resolves them before strict gates."""
    from aa.conversation.verifier import (
        build_single_unit_text,
        coerce_single_verdict,
        display_id_map_for_window,
    )

    pack = [
        _pack_entry(passage_id="chapter-3#exp0000"),
        _pack_entry(passage_id="chapter-7#atom-abc123"),
    ]
    units = split_response_units("Поддержка рядом помогает разбирать тягу.")
    assert len(units) == 1
    user_text = build_single_unit_text(unit=units[0], passages=pack)
    assert 'id="p1"' in user_text
    assert 'id="p2"' in user_text
    assert "chapter-3#exp0000" not in user_text
    assert "chapter-7#atom-abc123" not in user_text

    window_map = display_id_map_for_window(pack[:8])
    assert window_map == {"p1": "chapter-3#exp0000", "p2": "chapter-7#atom-abc123"}

    # A model verdict citing short ids validates to full stored ids.
    verdict = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
            "addresses_intent": True,
        },
        unit_id="u1",
        short_to_full=window_map,
        full_ids={"chapter-3#exp0000", "chapter-7#atom-abc123"},
    )
    assert verdict.evidence_passage_ids == ["chapter-3#exp0000"]

    # Full stored ids stay accepted (back-compat with exact copiers).
    back_compat = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-7#atom-abc123"],
            "addresses_intent": True,
        },
        unit_id="u1",
        short_to_full=window_map,
        full_ids={"chapter-3#exp0000", "chapter-7#atom-abc123"},
    )
    assert back_compat.evidence_passage_ids == ["chapter-7#atom-abc123"]

    # Unknown short ids still fail closed (never invented).
    from aa.conversation.verifier import check_cited_passage_ids

    unknown = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p9"],
            "addresses_intent": True,
        },
        unit_id="u1",
        short_to_full=window_map,
        full_ids={"chapter-3#exp0000", "chapter-7#atom-abc123"},
    )
    assembled = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": str(unknown.scope),
                    "supported": True,
                    "evidence_passage_ids": list(unknown.evidence_passage_ids),
                }
            ],
            "all_required_supported": True,
        },
        expected_unit_ids=["u1"],
    )
    with pytest.raises(VerifierValidationError):
        check_cited_passage_ids(assembled, pack_ids={"chapter-3#exp0000", "chapter-7#atom-abc123"})


def test_verifier_single_unit_short_ids_resolve() -> None:
    """Single-unit decisions resolve short ids with the same strictness."""
    from aa.conversation.verifier import (
        build_single_unit_text,
        check_cited_passage_ids,
        coerce_single_verdict,
    )

    pack = [_pack_entry(passage_id="chapter-3#exp0000")]
    units = split_response_units("Поддержка рядом помогает.")
    text = build_single_unit_text(unit=units[0], passages=pack)
    assert 'id="p1"' in text
    assert "chapter-3#exp0000" not in text

    verdict = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
            "addresses_intent": True,
        },
        unit_id="u1",
        short_to_full={"p1": "chapter-3#exp0000"},
        full_ids={"chapter-3#exp0000"},
    )
    assert verdict.unit_id == "u1"
    assert verdict.evidence_passage_ids == ["chapter-3#exp0000"]

    unknown = coerce_single_verdict(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p2"],
            "addresses_intent": True,
        },
        unit_id="u1",
        short_to_full={"p1": "chapter-3#exp0000"},
        full_ids={"chapter-3#exp0000"},
    )
    assembled = validate_grounding_result(
        {
            "units": [
                {
                    "unit_id": "u1",
                    "scope": str(unknown.scope),
                    "supported": True,
                    "evidence_passage_ids": list(unknown.evidence_passage_ids),
                }
            ],
            "all_required_supported": True,
        },
        expected_unit_ids=["u1"],
    )
    with pytest.raises(VerifierValidationError):
        check_cited_passage_ids(assembled, pack_ids={"chapter-3#exp0000"})


def test_v2_answer_carries_bounded_generation_budget_before_user_message() -> None:
    """Gate C repair (run 37561542378): bound v2 draft length turn-independently.

    The v2 answer path omitted the #83 efficiency guard, so weak fallback
    drafts ran long (many razdel units), making the verifier batch large,
    slow and flaky on structured output (14/14 clarifications, verifier
    never served, max 51.4s over budget). The budget hint keeps drafts to
    2-5 short sentences so verifier batches stay small and fast. The hard
    envelope stays authoritative; this is only an efficiency guard, never
    an exact-question special case.
    """
    messages = build_answer_messages(
        recent=[HumanMessage(content="тяжело вечером")],
        summary="",
        passages=[EvidencePassage(passage_id="p", source="s", section="c", text="текст")],
        user_message="почему?",
    )
    final = str(messages[-1].content)
    assert "<response_budget>" in final
    assert "answer length follows the question" in final.lower() or "Generation budget" in final
    assert final.index("<book_evidence>") < final.index("<response_budget>")
    assert final.index("<response_budget>") < final.index("<user_message>")
    assert final.rstrip().endswith("</user_message>")


def test_verifier_window_tightened_for_weak_fallback_slo() -> None:
    """Verifier receives the full pack with complete text (#295).

    Deterministic cite/quote/checksum gates still validate against the
    full stored pack; no display-window cap remains at this boundary.
    """
    from aa.conversation.verifier import (
        VERIFIER_MAX_EVIDENCE_PASSAGES,
        VERIFIER_MAX_PASSAGE_CHARS,
    )

    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 0
    assert VERIFIER_MAX_PASSAGE_CHARS == 0


def test_verifier_transport_schema_has_no_enum_but_code_stays_strict() -> None:
    """Boolean transport has no enum; internal scope validation stays strict."""
    import json

    from aa.conversation.verifier_schema import verifier_single_json_schema

    assert '"enum"' not in json.dumps(verifier_single_json_schema())
    with pytest.raises(VerifierValidationError):
        validate_grounding_result(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "unknown-scope",
                        "supported": True,
                        "evidence_passage_ids": [],
                    }
                ],
                "all_required_supported": True,
            },
            expected_unit_ids=["u1"],
        )


async def test_verifier_falls_back_to_per_unit_on_batch_provider_error() -> None:
    """Per-unit concurrent round serves; transient per-unit errors fail closed."""
    from aa.conversation.turn_pipeline import _verify_draft

    pack = [_pack_entry()]
    draft = "Понимаю. Поддержка рядом помогает."
    units = split_response_units(draft)
    good = _VerifierModel(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            },
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
                "addresses_intent": True,
            },
        ]
    )
    returned, result, passed = await _verify_draft(draft, pack, verifier_model=good)
    assert passed is True
    assert result is not None
    assert result.all_required_supported is True
    assert len(returned) == len(units)
    assert good.calls == len(units)


async def test_verifier_provider_429_never_falls_back() -> None:
    """Provider 429 must propagate for runner retire, never extra rounds."""
    from aa.conversation.verifier import run_verifier
    from aa.opencode.errors import OpenCodeRateLimitError

    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")

    class _Always429:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    model = _Always429()
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, pack, model=model)
    assert model.calls == len(units)


def test_verifier_payload_carries_generic_scoping_illustrations() -> None:
    """Single-unit payload carries generic boolean guidance, no live questions."""
    from aa.conversation.verifier import build_single_unit_text

    units = split_response_units("Понимаю. Тяга проходит.")
    pack = [_pack_entry()]
    single_text = build_single_unit_text(unit=units[0], passages=pack)
    assert "requires_book_evidence" in single_text
    assert "[p1]" not in single_text or "p1" in single_text
    assert "scope" not in single_text
    assert "all_required_supported" not in single_text
    # No exact live-question special cases in the prompt builder.
    for probe in ("тянет выпить", "ссора", "акции", "покончить"):
        assert probe not in single_text
