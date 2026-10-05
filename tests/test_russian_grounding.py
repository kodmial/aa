"""Russian quotation and multilingual grounding policy tests (issue #48).

Uses invented fixture sentences only; no canonical book text is committed.
"""

from __future__ import annotations

import pathlib

import pytest

from aa.app import Application
from aa.config import Settings
from aa.grounding import (
    TRANSLATION_MARKER_RU,
    EvidenceKind,
    EvidenceUnit,
    GroundingGate,
    Provenance,
    QuoteKind,
    check_grounding,
    contains_translation_label,
    format_russian_quotation,
)

RU_VERSION = "ru-test-2014-pin"
EN_VERSION = "en-test-1-pin"

RU_PASSAGE = "Фиктивная фраза про трезвость поддержку друзей утренние собрания"
RU_QUOTE = "поддержку друзей утренние собрания"
RU_CLAIM = "Фиктивная фраза про трезвость поддержку друзей утренние собрания"

EN_PASSAGE = "Fixture sentence about morning meetings and steady support habits"


def _ru_provenance() -> Provenance:
    return Provenance(
        corpus_version=RU_VERSION,
        source_id="ru-test-source",
        section_id="chapter-5",
        chunk_id="c1",
        char_start=0,
        char_end=len(RU_PASSAGE),
        source_checksum="0" * 64,
        source_language="ru",
    )


def _en_provenance() -> Provenance:
    return Provenance(
        corpus_version=EN_VERSION,
        source_id="core-pages-1-164",
        section_id="chapter-5",
        chunk_id="c1",
        char_start=0,
        char_end=len(EN_PASSAGE),
        source_checksum="1" * 64,
        source_language="en",
    )


def _ru_source_unit() -> EvidenceUnit:
    return EvidenceUnit(
        kind=EvidenceKind.SOURCE_TEXT,
        language="ru",
        text=RU_PASSAGE,
        provenance=_ru_provenance(),
    )


def _en_source_unit() -> EvidenceUnit:
    return EvidenceUnit(
        kind=EvidenceKind.SOURCE_TEXT,
        language="en",
        text=EN_PASSAGE,
        provenance=_en_provenance(),
    )


def test_exact_russian_quote_passes_with_pinned_ru_source() -> None:
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[_ru_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert verdict.passed
    assert verdict.source_exact
    assert verdict.code == "ok-exact-source"


def test_exact_quote_without_ru_corpus_fails_closed() -> None:
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[_ru_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=True,
    )
    assert not verdict.passed
    assert verdict.code == "ru-corpus-unavailable"


def test_exact_quote_requires_ru_source_text() -> None:
    verdict = check_grounding(
        russian_claim="Fixture sentence about morning meetings",
        quoted_text="morning meetings",
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=False,
        entails=lambda _claim, _source: True,
    )
    assert not verdict.passed
    assert verdict.code == "no-ru-source-text"


def test_translation_presented_as_exact_fails() -> None:
    labeled = f"{TRANSLATION_MARKER_RU} {RU_QUOTE}"
    verdict = check_grounding(
        russian_claim=labeled,
        quoted_text=labeled,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[_ru_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=True,
    )
    assert not verdict.passed
    assert verdict.code == "translation-as-exact"


def test_non_verbatim_exact_quote_fails() -> None:
    verdict = check_grounding(
        russian_claim="Фиктивная фраза про изобретенные несуществующие обстоятельства",
        quoted_text="изобретенные несуществующие обстоятельства",
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[_ru_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert not verdict.passed
    assert verdict.code in ("verbatim-mismatch", "unsupported-interpretation")


def test_translation_requires_explicit_label() -> None:
    verdict = check_grounding(
        russian_claim="Фиктивная фраза про трезвость поддержку друзей",
        quoted_text="Фиктивная фраза про трезвость",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=True,
        entails=lambda _claim, _source: True,
    )
    assert not verdict.passed
    assert verdict.code == "translation-unlabeled"


def test_translation_requires_allowed_fallback() -> None:
    claim = f"{TRANSLATION_MARKER_RU} Фиктивная фраза про трезвость поддержку"
    verdict = check_grounding(
        russian_claim=claim,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=False,
        entails=lambda _claim, _source: True,
    )
    assert not verdict.passed
    assert verdict.code == "translation-fallback-disabled"


def test_labeled_translation_passes_only_as_non_exact() -> None:
    claim = f"{TRANSLATION_MARKER_RU} Фиктивное утверждение про утренние собрания"
    verdict = check_grounding(
        russian_claim=claim,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=True,
        entails=lambda _claim, _source: True,
    )
    assert verdict.passed
    assert not verdict.source_exact
    assert verdict.code == "ok-translation"


def test_source_id_alone_cannot_pass() -> None:
    navigation = EvidenceUnit(
        kind=EvidenceKind.NAVIGATION,
        language="ru",
        text=RU_PASSAGE,
        provenance=_ru_provenance(),
    )
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[navigation],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert not verdict.passed
    assert verdict.code == "no-source-text"


def test_cited_provenance_must_resolve_to_evidence_text() -> None:
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert not verdict.passed
    assert verdict.code == "no-source-text"

    other = Provenance(
        corpus_version=RU_VERSION,
        source_id="ru-test-source",
        section_id="chapter-9",
        chunk_id="c9",
        source_language="ru",
    )
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[other],
        evidence=[_ru_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert not verdict.passed
    assert verdict.code == "unknown-provenance"


def test_normalization_and_query_rewrites_are_never_evidence() -> None:
    query = EvidenceUnit(
        kind=EvidenceKind.QUERY,
        language="ru",
        text=RU_CLAIM,
        provenance=_ru_provenance(),
    )
    draft = EvidenceUnit(
        kind=EvidenceKind.TRANSLATION_DRAFT,
        language="ru",
        text=RU_CLAIM,
        provenance=_ru_provenance(),
    )
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[query, draft],
        ru_corpus_available=True,
        allow_translation_fallback=False,
    )
    assert not verdict.passed
    assert verdict.code == "no-source-text"


def test_unrelated_passage_with_matching_id_fails_semantics() -> None:
    unrelated = EvidenceUnit(
        kind=EvidenceKind.SOURCE_TEXT,
        language="ru",
        text="Совершенно посторонняя заметка про огородные работы зимой",
        provenance=_ru_provenance(),
    )
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_ru_provenance()],
        evidence=[unrelated],
        ru_corpus_available=True,
        allow_translation_fallback=True,
        entails=None,
    )
    # The claim carries no translation label, so it fails labeling first;
    # with a label it must still fail semantic support by default.
    assert not verdict.passed
    labeled = f"{TRANSLATION_MARKER_RU} {RU_CLAIM}"
    verdict = check_grounding(
        russian_claim=labeled,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_ru_provenance()],
        evidence=[unrelated],
        ru_corpus_available=True,
        allow_translation_fallback=True,
        entails=None,
    )
    assert not verdict.passed
    assert verdict.code == "unsupported-interpretation"


def test_cross_language_requires_measured_entailment() -> None:
    claim = f"{TRANSLATION_MARKER_RU} Фиктивное утверждение про утренние собрания"
    defaulted = check_grounding(
        russian_claim=claim,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=True,
        entails=None,
    )
    assert not defaulted.passed
    assert defaulted.code == "unsupported-interpretation"

    measured = check_grounding(
        russian_claim=claim,
        quoted_text="",
        quote_kind=QuoteKind.TRANSLATION,
        cited=[_en_provenance()],
        evidence=[_en_source_unit()],
        ru_corpus_available=False,
        allow_translation_fallback=True,
        entails=lambda _claim, _source: True,
    )
    assert measured.passed


def test_rejected_entailment_fails_despite_verbatim_quote() -> None:
    verdict = check_grounding(
        russian_claim=RU_CLAIM,
        quoted_text=RU_QUOTE,
        quote_kind=QuoteKind.EXACT_SOURCE,
        cited=[_ru_provenance()],
        evidence=[_ru_source_unit()],
        ru_corpus_available=True,
        allow_translation_fallback=False,
        entails=lambda _claim, _source: False,
    )
    assert not verdict.passed
    assert verdict.code == "unsupported-interpretation"


def test_provenance_must_be_pinned() -> None:
    with pytest.raises(ValueError):
        Provenance(corpus_version="  ", source_id="s", section_id="sec").validate()
    gate = GroundingGate(corpus_version="  ")
    verdict = gate.judge(russian_claim=RU_CLAIM)
    assert not verdict.passed


def test_format_russian_quotation_marks_kind_unambiguously() -> None:
    exact = format_russian_quotation(RU_QUOTE, QuoteKind.EXACT_SOURCE, _ru_provenance())
    assert RU_QUOTE in exact
    assert not contains_translation_label(exact)

    translated = format_russian_quotation(RU_QUOTE, QuoteKind.TRANSLATION, _en_provenance())
    assert TRANSLATION_MARKER_RU in translated
    assert contains_translation_label(translated)

    with pytest.raises(ValueError):
        format_russian_quotation("   ", QuoteKind.EXACT_SOURCE, _ru_provenance())
    with pytest.raises(ValueError):
        format_russian_quotation(
            f"{TRANSLATION_MARKER_RU} {RU_QUOTE}",
            QuoteKind.EXACT_SOURCE,
            _ru_provenance(),
        )


def test_application_wires_fail_closed_grounding_gate() -> None:
    # Cutover #118: the LangGraph verifier owns grounding; the retired
    # orchestrator gate handle stays None for constructor compatibility.
    settings = Settings.from_env({})
    app = Application(settings)
    assert app.grounding is None


def _prompt_text() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "prompts" / "aa-agent-system.md").read_text(encoding="utf-8")


def test_prompt_states_russian_policy_and_defers_to_orchestrator() -> None:
    prompt = _prompt_text()
    assert "RUSSIAN QUOTATION AND MULTILINGUAL GROUNDING" in prompt
    assert "Russian conversations display quotations in Russian" in prompt
    assert "aa.grounding" in prompt
    assert TRANSLATION_MARKER_RU in prompt


def test_prompt_does_not_duplicate_orchestrator_mechanics() -> None:
    prompt = _prompt_text()
    assert "They are never evidence." not in prompt
    assert "Map summaries, embedding metadata, scores and reranker output" not in prompt
    assert "Every substantive claim must remain traceable" not in prompt


def test_prompt_policy_stays_in_english() -> None:
    prompt = _prompt_text()
    without_allowed_ru = prompt.replace(TRANSLATION_MARKER_RU, "")
    without_allowed_ru.encode("ascii")
