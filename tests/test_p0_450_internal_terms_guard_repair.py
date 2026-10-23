"""P0 kodmial/aa#450: internal-terms guard audit and safe same-pack recovery.

Historical fault (live Telegram runs #39972781629, job 120479539082): a
locally ``guard-internal-terms``-rejected draft collapsed into the generic
``verifier_outcome=unavailable`` path, skipped every repair, and the user
received a ``[service-error]`` after a full planner+retrieval+answer chain
while the already-fetched, integrity-verified Evidence Pack could still
support a clean Russian answer.

These tests pin, with synthetic fixtures only (never real Telegram input,
never canonical excerpts, never credentials):

1. the closed internal-terms vocabulary: allowed benign samples pass the
   guard, genuine mechanics leaks (RU and EN, mixed case and hyphen
   variants) stay blocked, and no rule was removed or loosened;
2. typed classification: a local guard rejection is distinct from
   ``verifier-unavailable``/schema/verdict/safety/429/budget failures and
   the rejected original draft is never sent to the semantic verifier;
3. bounded same-pack recovery: at most one regeneration from exactly the
   same verified pack, the original resolved intent and unchanged history
   (zero planner/retrieval calls, one regeneration, unchanged evidence
   fingerprint, one delivery), every candidate reverified under the
   unchanged verifier and final gates;
4. fail-closed paths: repeated leaks, unsupported regen, verifier outage,
   provider 429, cancellation, exhausted budget -- the rejected draft is
   never delivered, never typed as success, and the earliest typed cause
   is retained.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_LEAKY_SEARCH_QUERY_DRAFT = (
    "Я переписал ваш вопрос как запрос и запустил поиск по корпусу текстов, "
    "но эмбеддинги не совпали, поэтому спросите иначе."
)
_LEAKY_RETRY_DRAFT = "Риранк по корпусу текстов не дал ответа: векторная база не открылась."
_NON_RUSSIAN_DRAFT = "Plain artificial English filler sentence. Another filler sentence."
_SAFE_HELP_1 = (
    "Поддержка рядом помогает пережить тягу сегодня. Расскажите, что сейчас важнее всего?"
)
_SAFE_HELP_2 = "Честный разговор о трудностях и ближайшие трезвые шаги помогают сегодня."
_SAFE_HELP_3 = "Спокойный вечерний распорядок и поддержка рядом помогают сегодня."

# Audit table A1: allowed samples. Ordinary Russian book-grounded /
# supportive phrasing, benign English book and proper names, and harmless
# near-miss wording that merely shares a general concept with a mechanics
# rule. None of these may trigger the internal-terms guard.
_GUARD_ALLOWED_SAMPLES: tuple[str, ...] = (
    _SAFE_HELP_1,
    _SAFE_HELP_2,
    _SAFE_HELP_3,
    "Хорошо, что вы написали. Расскажите, чего вы ожидаете от этого дня?",
    "Поговорите со спонсором до того, как примете решение, — это опора группы.",
    "Большая Книга — верный спутник; приходите на собрание и расскажите о трудном дне.",
    "Alcoholics Anonymous — история сообщества, где каждый может начать заново.",
    "Завтра — день поиска поддержки; собрание в семь вечера.",
    "Корпус здания виден от входа в общину, подходят любые одежды.",
    "Ваш запрос важен; приходите на собрание и спрашивайте о прошлом опыте.",
)

# Review follow-up: token-bounded matching must not treat a rule embedded
# inside a longer ordinary word as a mechanics disclosure.
_GUARD_CONTAINING_WORD_SAMPLES: tuple[str, ...] = (
    "The indexing of the group notes is complete.",
    "Providers of support meet on Tuesdays.",
    "Planners and coordinators share the schedule.",
    "Qualifications for the volunteer role are simple.",
    "Evidenced progress matters more than perfection.",
)

# Audit table A2: forbidden samples. Genuine technical disclosure in
# English, Russian, mixed case, and hyphen/format variants; source-ID,
# status and provider error echoes; mixed RU+EN disclosures. Each one
# MUST keep the guard blocked.
_GUARD_FORBIDDEN_SAMPLES: tuple[str, ...] = (
    "Мы переписали ваш вопрос в поисковый запрос и проверяем эмбеддинги.",
    "Система запустила поиск по корпусу текстов, но индекс не откликнулся.",
    "The retrieval pipeline failed while re-ranking; bm25 index unavailable.",
    "Fail-Closed: ответ сервера не прошёл json-schema проверку.",
    "Модель использует векторную базу из Faiss и Bm-25.",
    "Structured-Output parsing failed, повторите запрос позже.",
    "Диагностика: rerank внутри FAISS упал в выдаче.",
    "Dense embeddings store the passages.",
    "code=ru-fourth-edition-txt, status unavailable, provider returned 429",
)

# Unmodified historical rules keep exact blocking coverage.
_HISTORICAL_TERMS_KEEP_COVERAGE: tuple[str, ...] = (
    "evidence",
    "corpus",
    "provider",
    "qualification",
    "grounding",
    "index",
    "embedding",
    "planner",
    "retrieval",
    "json_schema",
    "bm25",
    "faiss",
    "rerank",
)


def _pack_entry(passage_id: str = "chapter-3#exp0000", text: str = _SAFE_HELP_3) -> dict[str, Any]:
    """Synthetic integrity-valid book passage (never real corpus text)."""
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


def _unit_count(draft: str) -> int:
    """Expected per-unit verifier calls for one scripted draft."""
    from aa.conversation.response_units import split_response_units

    return len(split_response_units(draft))


class _ScriptedAnswer:
    """Return scripted drafts in order, recording prompt shape only."""

    def __init__(self, texts: list[Any]) -> None:
        self._texts = list(texts)
        self.calls = 0
        self.seen_user_messages: list[str] = []
        self.seen_safety_blocks: list[str] = []

    async def ainvoke(self, messages: Any) -> AIMessage:
        from langchain_core.messages import BaseMessage

        self.calls += 1
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, BaseMessage) and item.type == "human":
                    content = item.content
                    if isinstance(content, str):
                        if "<user_message>" in content:
                            self.seen_user_messages.append(content)
                        if "<safety_policy>" in content:
                            start = content.index("<safety_policy>")
                            end = content.index("</safety_policy>")
                            self.seen_safety_blocks.append(content[start:end])
        scripted = self._texts[min(self.calls - 1, len(self._texts) - 1)]
        if isinstance(scripted, BaseException):
            raise scripted
        return AIMessage(content=str(scripted))


class _ScriptedVerifier:
    """Structured verifier double with per-call scripted verdict payloads."""

    def __init__(self, payloads: list[dict[str, Any] | BaseException]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        """Plain-text fallback channel: mirror the scripted transport state."""
        _ = messages
        self.calls += 1
        last = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        if isinstance(last, BaseException):
            raise last
        raise AssertionError("unexpected verifier text fallback in this test")

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        if isinstance(payload, BaseException):
            raise payload
        return dict(payload)


def _supported_verdict() -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": ["p1"],
        "addresses_intent": True,
    }


def _unsupported_verdict() -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": False,
        "evidence_passage_ids": [],
        "addresses_intent": False,
    }


def _forbid_planner_and_retrieval(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Counters that raise the moment any planner/retrieval call fires."""
    import aa.conversation.planner_node as planner_node
    import aa.retrieval.evidence as evidence_mod

    counters = {"planner": 0, "retrieval": 0}

    async def _never_planner(*args: Any, **kwargs: Any) -> Any:
        counters["planner"] += 1
        raise AssertionError("planner must not run during local-guard repair")

    def _never_retrieve(*args: Any, **kwargs: Any) -> Any:
        counters["retrieval"] += 1
        raise AssertionError("retrieval must not run during local-guard repair")

    monkeypatch.setattr(planner_node, "run_planner", _never_planner)
    monkeypatch.setattr(evidence_mod, "retrieve_evidence", _never_retrieve)
    return counters


def _base_turn(
    *,
    answer: _ScriptedAnswer,
    verifier: _ScriptedVerifier,
    user_message: str = "Вечером тяжело без выпивки, как справляться?",
) -> dict[str, Any]:
    return {
        "user_message": user_message,
        "summary": "",
        "recent": [HumanMessage(content="Здравствуйте")],
        "evidence_pack": [_pack_entry()],
        "answer_model": answer,
        "verifier_model": verifier,
        "planner_model": object(),
        "retrieval_index": object(),
        "initial_query_count": 12,
        "upstream_latency_ms": 0.0,
        "planner_mode": "retrieval",
        "resolved_intent": user_message,
    }


# ---------------------------------------------------------------------------
# Internal-terms matcher audit (kodmial/aa#450 parts A+B).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("allowed", _GUARD_ALLOWED_SAMPLES)
def test_guard_allowed_samples_pass(allowed: str) -> None:
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(allowed) is False


@pytest.mark.parametrize("forbidden", _GUARD_FORBIDDEN_SAMPLES)
def test_guard_forbidden_samples_blocked(forbidden: str) -> None:
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(forbidden) is True


@pytest.mark.parametrize("allowed", _GUARD_CONTAINING_WORD_SAMPLES)
def test_guard_allows_containing_word_collisions(allowed: str) -> None:
    """A rule embedded in a longer ordinary word is not a standalone leak."""
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(allowed) is False


@pytest.mark.parametrize(
    "blocked",
    (
        "эмбеддинги",  # plural inflection of guarded "эмбеддинг"
        "риранка",  # genitive of "риранк"
        "фаиссе",  # prepositional of "фаисс"
        "векторной базе",  # inflected first word of "векторная база"
        "векторном хранилище",  # inflected both words of "векторное хранилище"
        "поиска по корпусу",  # inflected first word of "поиск по корпусу"
        "корпуса текстов",  # inflected first word of "корпус текстов"
        "векторною базою",  # instrumental case ending on both words
        "векторного хранилища",  # genitive ending on both words
        "поисками по корпусам",  # plural instrumental endings
        "корпусами текстами",  # plural instrumental endings
    ),
)
def test_cyrillic_inflected_forms_blocked(blocked: str) -> None:
    """Russian case endings on guarded terms are intentionally blocked."""
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(blocked) is True


@pytest.mark.parametrize(
    ("word", "stem"),
    (
        ("базою", "баз"),
        ("векторною", "векторн"),
        ("корпусами", "корпус"),
        ("текстами", "текст"),
        ("хранилищами", "хранилищ"),
    ),
)
def test_cyrillic_stem_reduces_stacked_endings(word: str, stem: str) -> None:
    """Iterative stripping reduces stacked/less common endings to the root."""
    from aa.conversation.turn_pipeline import _cyrillic_stem

    assert _cyrillic_stem(word) == stem


def test_cyrillic_word_roots_cover_every_guarded_phrase_word() -> None:
    """Every closed-vocabulary word has an explicit audited stable root."""
    from aa.conversation.turn_pipeline import (
        _CYRILLIC_RE,
        _CYRILLIC_WORD_ROOTS,
        _INTERNAL_TERMS,
    )

    missing = {
        word
        for term in _INTERNAL_TERMS
        if _CYRILLIC_RE.search(term)
        for word in term.split()
        if len(word) > 3 and word not in _CYRILLIC_WORD_ROOTS
    }
    assert missing == set()


@pytest.mark.parametrize(
    "blocked",
    (
        "векторненькая база",  # diminutive ending absent from the suffix table
        "векторненькой базой",  # stacked diminutive/instrumental endings
        "корпусище текстов",  # augmentative ending absent from the suffix table
        "корпусищем текстов",  # stacked augmentative/instrumental endings
    ),
)
def test_cyrillic_endings_outside_suffix_table_blocked(blocked: str) -> None:
    """Matching accepts any continuation after the audited stable root."""
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(blocked) is True


@pytest.mark.parametrize(
    "allowed",
    (
        "корпус здания",  # unrelated word sharing a prefix with "корпус текстов"
        "векторный анализ",  # unrelated word sharing a prefix with "векторная база"
        "Поиск смысла помогает восстановиться.",  # no mechanics corpus phrase
        "Корпусная мебель стоит в комнате.",  # no mechanics corpus phrase
    ),
)
def test_cyrillic_unrelated_prefix_allowed(allowed: str) -> None:
    """Unrelated Russian words sharing a prefix with a guarded phrase are allowed."""
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(allowed) is False


@pytest.mark.parametrize("term", _HISTORICAL_TERMS_KEEP_COVERAGE)
def test_unmodified_terms_keep_blocking(term: str) -> None:
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(term) is True


@pytest.mark.parametrize("term", ["BM25", "FAISS", "Json-Schema", "FAIL CLOSED", "Rerank"])
def test_mixed_case_and_format_variants_stay_blocked(term: str) -> None:
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert leaks_internal_terms(term) is True


def test_added_rules_each_have_negative_and_positive_case() -> None:
    """Each additive rule blocks a direct mixed-case leak with Cyrillic.

    The additions stay narrow: ordinary book-grounded wording that only
    shares a general concept with a mechanics rule stays allowed (no
    semantic or word-family loosening).
    """
    from aa.conversation.turn_pipeline import leaks_internal_terms

    added_rules = (
        "эмбеддинг",
        "риранк",
        "фаисс",
        "бм25",
        "векторная база",
        "векторное хранилище",
        "поиск по корпусу",
        "корпус текстов",
        "fail-closed",
        "search-failure",
        "structured-output",
        "json-schema",
        "bm-25",
    )
    from aa.conversation import turn_pipeline as tp

    for rule in added_rules:
        assert rule in tp._INTERNAL_TERMS, rule
        assert leaks_internal_terms(f"Ответ недоступен: {rule.capitalize()} закрыт.") is True

    for near_miss in _GUARD_ALLOWED_SAMPLES:
        assert leaks_internal_terms(near_miss) is False


async def test_guard_cause_records_closed_rule_id_only() -> None:
    """Typed cause detail carries the closed rule id, never draft text."""
    from aa.conversation.answer_causality import GUARD_INTERNAL_TERMS, TELEMETRY_CAUSE_KEY
    from aa.conversation.turn_pipeline import _INTERNAL_TERMS as rule_set
    from aa.conversation.turn_pipeline import _matched_internal_rule, _verify_draft

    class _StubModel:
        pass

    telemetry: dict[str, Any] = {}
    units, result, ok = await _verify_draft(
        _LEAKY_SEARCH_QUERY_DRAFT,
        [_pack_entry()],
        verifier_model=_StubModel(),
        telemetry=telemetry,
    )
    assert not ok
    assert units
    assert result is None
    cause = telemetry[TELEMETRY_CAUSE_KEY]
    assert cause["reason"] == GUARD_INTERNAL_TERMS
    assert cause["source"] == "local"
    detail = cause["detail"]
    assert detail.startswith("rule:")
    rule_id = detail[len("rule:") :]
    assert rule_id in rule_set
    assert "_LEAKY_SEARCH_QUERY_DRAFT" not in detail
    assert "переписал" not in detail
    assert len(detail) <= 64
    assert _matched_internal_rule(_LEAKY_SEARCH_QUERY_DRAFT) == rule_id


def test_typed_local_guard_reason_keeps_the_immutable_first_cause() -> None:
    """The repair admission reads the first cause only while ``_attach``
    guarantees later stages never overwrite it."""
    from aa.conversation.answer_causality import (
        GUARD_INTERNAL_TERMS,
        TELEMETRY_CAUSE_KEY,
        TELEMETRY_CHAIN_KEY,
        AnswerFailureCause,
        attach_answer_failure_cause,
    )
    from aa.conversation.turn_pipeline import _typed_local_guard_reason

    telemetry: dict[str, Any] = {}
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded(GUARD_INTERNAL_TERMS, "verification", "local"),
    )
    attach_answer_failure_cause(
        telemetry,
        AnswerFailureCause.bounded("verdict-unsupported", "verification", "verification"),
    )
    assert _typed_local_guard_reason(telemetry) == GUARD_INTERNAL_TERMS
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_INTERNAL_TERMS
    assert telemetry[TELEMETRY_CHAIN_KEY][-1]["reason"] == "verdict-unsupported"


def test_language_gate_single_cyrillic_limitation_is_documented() -> None:
    """Known separate defect (unmodified): one Cyrillic symbol passes the
    Russian-output language check. Documented in the #450 audit only; no
    behavior change in this issue."""
    from aa.conversation.turn_pipeline import contains_cyrillic

    assert contains_cyrillic("Please say something. Раз.") is True


# ---------------------------------------------------------------------------
# Typed rejection classification and bounded same-pack recovery.
# ---------------------------------------------------------------------------


async def test_local_guard_rejection_recovered_by_one_same_pack_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaky draft: local guard, verifier not invoked, one same-pack regen.

    The safe grounded Russian candidate reaches the real verifier and all
    final gates and is delivered. Planner/retrieval calls stay at zero,
    one regeneration, evidence fingerprint and citations unchanged, and
    exactly one delivery.
    """
    from aa.conversation.answer_causality import (
        GUARD_INTERNAL_TERMS,
        TELEMETRY_CAUSE_KEY,
    )
    from aa.conversation.failures import is_service_error
    from aa.conversation.finalization import evidence_digest_for_pack
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    counters = _forbid_planner_and_retrieval(monkeypatch)
    pack = [_pack_entry()]
    input_digest = evidence_digest_for_pack(pack)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _SAFE_HELP_1])
    verifier = _ScriptedVerifier([_supported_verdict()])
    outcome = await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    # Original leaky draft never reached the semantic verifier. The
    # per-unit concurrent fallback verifies each response unit of the
    # regenerated draft, so derive the count from the splitter rather
    # than pinning today's unit count.
    assert verifier.calls == _unit_count(_SAFE_HELP_1)
    assert answer.calls == 2
    assert outcome["text"] == _SAFE_HELP_1
    assert not is_service_error(outcome["text"])
    from aa.conversation.turn_pipeline import leaks_internal_terms

    assert not leaks_internal_terms(outcome["text"])
    # One delivery only.
    assert outcome.get("segments", [outcome["text"]]) == [outcome["text"]]
    telemetry = dict(outcome["telemetry"])
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_INTERNAL_TERMS
    assert telemetry["verifier_outcome"] == "passed-after-repair-existing-pack"
    assert telemetry["answer_outcome"] == "served"
    assert telemetry["qualified"] is True
    assert telemetry["local_guard_initial_rejection"] is True
    assert telemetry["local_guard_regen_attempts"] == 1
    assert telemetry["local_guard_repair_status"] == "served"
    # No re-planning, no new retrieval, unchanged evidence fingerprint.
    assert telemetry["planner_query_count"] == 0
    assert telemetry["planner_outcome"] == "skipped-initial"
    assert telemetry["retrieval_outcome"] == "preloaded"
    assert telemetry["retrieval_passages"] == 1
    assert telemetry["pack_digest"] == input_digest
    assert telemetry["answer_rounds"] == 2
    # Scoped control instruction travels only in the <safety_policy>
    # block of the regeneration prompt, never in the user request.
    assert len(answer.seen_safety_blocks) == 1
    assert answer.seen_user_messages, "regeneration prompt missing"
    last = answer.seen_user_messages[-1]
    assert "<user_message>" in last and "<safety_policy>" in last


async def test_repeated_leaky_regeneration_never_delivered_stays_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.answer_causality import (
        GUARD_INTERNAL_TERMS,
        TELEMETRY_CAUSE_KEY,
    )
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _LEAKY_RETRY_DRAFT])
    verifier = _ScriptedVerifier([_supported_verdict()])
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    assert verifier.calls == 0
    assert answer.calls == 2
    assert exc.value.category == "clarification-unavailable"
    telemetry = dict(exc.value.telemetry)
    # Verifier was never invoked; the status stays distinct from
    # ``verifier-unavailable``.
    assert telemetry["verifier_outcome"] == "local-guard"
    assert telemetry["verifier_latency_ms"] >= 0.0
    assert telemetry["answer_outcome"] == "clarification-unavailable"
    assert telemetry["qualified"] is False
    # Typed original reason retained as the first cause.
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_INTERNAL_TERMS
    assert telemetry["local_guard_initial_rejection"] is True
    assert telemetry["local_guard_regen_attempts"] == 1
    assert telemetry["local_guard_repair_status"] == "failed"


async def test_no_budget_blocks_regeneration_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausted per-turn regress budget keeps the typed local-guard
    failure (never a fake delivery, never ``verifier-unavailable``)."""
    import aa.conversation.turn_pipeline as turn_pipeline

    # Deterministic stage-bound budget: allow the initial draft and its
    # verifier round, then spend the turn before the regeneration call.
    state = {"calls": 0}

    def _stage_budget() -> float:
        state["calls"] += 1
        return 105.0 if state["calls"] <= 2 else -1.0

    monkeypatch.setattr(turn_pipeline, "_effective_turn_budget_s", _stage_budget)
    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _SAFE_HELP_2])
    verifier = _ScriptedVerifier([_supported_verdict(), _supported_verdict()])
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    assert verifier.calls == 0
    telemetry = dict(exc.value.telemetry)
    assert exc.value.category == "turn-budget-exceeded"
    assert telemetry["verifier_outcome"] == "local-guard"
    assert telemetry["local_guard_repair_status"] == "budget-blocked"
    assert telemetry["local_guard_regen_attempts"] == 0
    assert telemetry["turn_budget_exceeded"] is True


async def test_budget_lost_during_regeneration_reports_budget_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission succeeds but the re-verify slice is gone after drafting.

    The repair must report a budget boundary (``budget-blocked``), not a
    content failure, and must not attempt a doomed verifier call.
    """
    import aa.conversation.turn_pipeline as turn_pipeline
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    state = {"budget": 105.0}

    def _budget() -> float:
        return state["budget"]

    monkeypatch.setattr(turn_pipeline, "_effective_turn_budget_s", _budget)
    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _SAFE_HELP_1])
    original_ainvoke = answer.ainvoke

    async def _ainvoke(messages: Any) -> AIMessage:
        result = await original_ainvoke(messages)
        if answer.calls == 2:
            # Regeneration succeeded, but it consumed the remaining
            # end-to-end budget before re-verification.
            state["budget"] = -1.0
        return result

    monkeypatch.setattr(answer, "ainvoke", _ainvoke)
    verifier = _ScriptedVerifier([_supported_verdict()])
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    assert verifier.calls == 0
    telemetry = dict(exc.value.telemetry)
    assert telemetry["local_guard_regen_attempts"] == 1
    assert telemetry["local_guard_repair_status"] == "budget-blocked"
    assert telemetry["turn_budget_exceeded"] is True


async def test_unsupported_regeneration_fails_closed_without_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe-ish regeneration the verifier rejects: typed failure retained,
    full re-planning stays disabled, nothing delivered as success."""
    from aa.conversation.answer_causality import (
        GUARD_INTERNAL_TERMS,
        TELEMETRY_CAUSE_KEY,
    )
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _SAFE_HELP_2])
    verifier = _ScriptedVerifier([_unsupported_verdict()])
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    assert verifier.calls == _unit_count(_SAFE_HELP_2)
    assert exc.value.category == "clarification-unavailable"
    telemetry = dict(exc.value.telemetry)
    assert telemetry["verifier_outcome"] == "unsupported"
    assert telemetry["answer_outcome"] == "clarification-unavailable"
    assert telemetry["qualified"] is False
    reasons = [item["reason"] for item in telemetry["answer_failure_cause_chain"]]
    assert reasons[0] == GUARD_INTERNAL_TERMS
    # A fully-verified-but-unsupported repair draft maps to the
    # pre-existing required-support cause (per-unit vocabulary, #450).
    assert "verifier-incomplete-verdict" in reasons
    for item in telemetry["answer_failure_cause_chain"]:
        if item["reason"] == "verifier-incomplete-verdict":
            assert item["detail"] == "required-support"
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_INTERNAL_TERMS


async def test_verifier_outage_during_regeneration_says_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real transport outage of the verifier remains ``unavailable``,
    distinct from the local-guard status, and remains fail-closed."""
    from aa.conversation.answer_causality import (
        GUARD_INTERNAL_TERMS,
        TELEMETRY_CAUSE_KEY,
    )
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.opencode.errors import OpenCodeTransientError

    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, _SAFE_HELP_2])
    verifier = _ScriptedVerifier([OpenCodeTransientError("stream dropped")])
    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    # One structured attempt plus the bounded text fallback (which the
    # double keeps failing) per response unit.
    assert verifier.calls == _unit_count(_SAFE_HELP_2) * 2
    assert exc.value.category == "clarification-unavailable"
    telemetry = dict(exc.value.telemetry)
    assert telemetry["verifier_outcome"] == "unavailable"
    assert telemetry["qualified"] is False
    reasons = [item["reason"] for item in telemetry["answer_failure_cause_chain"]]
    assert reasons[0] == GUARD_INTERNAL_TERMS
    assert "verifier-unavailable" in reasons
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_INTERNAL_TERMS


async def test_provider_429_propagates_from_guard_repair() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.opencode.errors import OpenCodeRateLimitError

    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, OpenCodeRateLimitError("429")])
    verifier = _ScriptedVerifier([_supported_verdict()])
    with pytest.raises(OpenCodeRateLimitError):
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))


async def test_cancellation_propagates_from_guard_repair() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    answer = _ScriptedAnswer([_LEAKY_SEARCH_QUERY_DRAFT, asyncio.CancelledError()])
    verifier = _ScriptedVerifier([_supported_verdict()])
    with pytest.raises(asyncio.CancelledError):
        await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))


async def test_guard_language_rejection_also_gets_same_pack_regen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Cyrillic initial draft: same bounded repair admission."""
    from aa.conversation.answer_causality import GUARD_LANGUAGE, TELEMETRY_CAUSE_KEY
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_NON_RUSSIAN_DRAFT, _SAFE_HELP_1])
    verifier = _ScriptedVerifier([_supported_verdict()])
    outcome = await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    # Per-unit concurrent double: each response unit of the regenerated
    # draft gets one unit verdict call.
    assert verifier.calls == _unit_count(_SAFE_HELP_1)
    assert outcome["text"] == _SAFE_HELP_1
    telemetry = dict(outcome["telemetry"])
    assert telemetry[TELEMETRY_CAUSE_KEY]["reason"] == GUARD_LANGUAGE
    assert telemetry["verifier_outcome"] == "passed-after-repair-existing-pack"
    assert telemetry["answer_outcome"] == "served"
    assert telemetry["local_guard_repair_status"] == "served"
    assert telemetry["local_guard_regen_attempts"] == 1


async def test_normal_served_turn_costs_zero_extra_guard_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path is unchanged: no repair fields fire, one model call."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    counters = _forbid_planner_and_retrieval(monkeypatch)
    answer = _ScriptedAnswer([_SAFE_HELP_1])
    verifier = _ScriptedVerifier([_supported_verdict()])
    outcome = await run_v2_answer_turn(**_base_turn(answer=answer, verifier=verifier))

    assert counters == {"planner": 0, "retrieval": 0}
    assert answer.calls == 1
    # Per-unit concurrent double: one unit-verdict call per response
    # unit of the served draft (fast path has no extra guard rounds).
    assert verifier.calls == _unit_count(_SAFE_HELP_1)
    assert outcome["text"] == _SAFE_HELP_1
    telemetry = dict(outcome["telemetry"])
    assert telemetry["local_guard_initial_rejection"] is False
    assert telemetry["local_guard_regen_attempts"] == 0
    assert telemetry["local_guard_repair_status"] == "none"
    assert telemetry["answer_rounds"] == 1
    assert telemetry["answer_outcome"] == "served"
    assert telemetry["qualified"] is True
