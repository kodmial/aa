"""P0 kodmial/aa#246: real Telegram answers must be substantive/book-grounded.

Proven regression (manual Telegram dialogue 2026-10-08): an ordinary
arbitrary question about stopping drinking plus follow-up requests for
recommendations ended in long repeated empty clarifications without
practically useful book information; a meta question about the bot role
got an evasive template. Closing #240/#244 and a stale Gate C PASS did
not prove the fix.

Hardening (turn-independent, Product Contract #110 unchanged, no
exact-question whitelist/special-case in runtime):

- substantive turns require a helpful answer on the substance (not only
  ``verified_book_units > 0``): a concrete recommendation/explanation
  supported by relevant exact canonical RU passages, briefly human-like,
  with every substantive claim confirmed by the independent verifier;
  side facts, general sympathy, verification excuses, avoiding
  clarifications, template retries and hash-variant filler fail;
- meta turns require a direct natural RU answer without false identity;
- negative controls (forced generic retry/clarification, irrelevant
  quote-only, service-link-only) fail even with positive verifier counts;
  true brief grounded answers pass;
- the live lane judges the real sent Telegram message plus this turn's
  actual telemetry snapshot (never an intermediate draft) and covers
  freely worded RU families: concrete anti-drinking actions, a
  "what should I do?" continuation, short contextual follow-ups,
  colloquial/typo variants and meta role questions (frozen core plus
  held-out rephrasings).

Gate E SLO stays strict; quality has priority over latency.
"""

from __future__ import annotations

import pathlib

from aa.conversation.turn_pipeline import (
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_REPLY,
)
from aa.qualification.product_contract_live import (
    _has_declarative_substance,
    _is_avoiding_clarification_text,
    _is_direct_meta_reply,
    _is_grounded_substantive_reply,
    _is_quote_only_text,
    _is_service_link_only_text,
)


def _grounded_snapshot(**overrides: object) -> dict[str, object]:
    from typing import Any as _Any

    snapshot: dict[str, _Any] = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 3,
        "verified_book_units": 1,
        "response_units": 1,
    }
    snapshot.update(overrides)
    return snapshot


def test_brief_grounded_answer_passes() -> None:
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    assert _is_grounded_substantive_reply(_grounded_snapshot(), reply) is True


def test_narrowed_brief_grounded_answer_passes() -> None:
    reply = "Спокойный вечер и режим помогают уснуть без выпивки."
    snapshot = _grounded_snapshot(answer_outcome="narrowed-supported")
    assert _is_grounded_substantive_reply(snapshot, reply) is True


def test_forced_generic_retry_fails_despite_counts() -> None:
    snapshot = _grounded_snapshot()
    assert _is_grounded_substantive_reply(snapshot, NATURAL_RETRY_REPLY) is False


def test_forced_clarification_fails_despite_counts() -> None:
    snapshot = _grounded_snapshot()
    assert _is_grounded_substantive_reply(snapshot, NATURAL_CLARIFICATION_REPLY) is False


def test_avoiding_clarification_fails() -> None:
    snapshot = _grounded_snapshot()
    avoiding = "Расскажите чуть подробнее, что сейчас важнее всего?"
    assert _is_avoiding_clarification_text(avoiding) is True
    assert _is_grounded_substantive_reply(snapshot, avoiding) is False


def test_service_link_only_fails() -> None:
    snapshot = _grounded_snapshot()
    link_only = "Подробнее смотрите здесь: https://example.com/book-chapter ."
    assert _is_service_link_only_text(link_only) is True
    assert _is_grounded_substantive_reply(snapshot, link_only) is False


def test_quote_only_fails() -> None:
    snapshot = _grounded_snapshot()
    quote_only = "«Поддержка рядом помогает пережить тягу сегодня и завтра»"
    assert _is_quote_only_text(quote_only) is True
    assert _is_grounded_substantive_reply(snapshot, quote_only) is False


def test_sympathy_only_without_substance_fails() -> None:
    snapshot = _grounded_snapshot()
    sympathy = "Понимаю, вам тяжело?"
    assert _has_declarative_substance(sympathy) is False
    assert _is_grounded_substantive_reply(snapshot, sympathy) is False


def test_unavailable_verifier_fails() -> None:
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    snapshot = _grounded_snapshot(verifier_unavailable_units=1)
    assert _is_grounded_substantive_reply(snapshot, reply) is False


def test_turn_budget_exceeded_fails() -> None:
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    snapshot = _grounded_snapshot(turn_budget_exceeded=True)
    assert _is_grounded_substantive_reply(snapshot, reply) is False


def test_zero_verified_units_fails() -> None:
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    snapshot = _grounded_snapshot(verified_book_units=0)
    assert _is_grounded_substantive_reply(snapshot, reply) is False


def test_direct_meta_reply_passes() -> None:
    reply = (
        "Я ИИ-помощник по материалам сообщества: поддерживаю разговор "
        "о трезвости, помогаю разобрать тягу и ближайшие шаги. "
        "Спросите о конкретной ситуации."
    )
    assert _is_direct_meta_reply(reply) is True


def test_evasive_meta_template_fails() -> None:
    assert _is_direct_meta_reply(NATURAL_CLARIFICATION_REPLY) is False
    assert _is_direct_meta_reply("Расскажите чуть подробнее?") is False


def test_false_identity_meta_fails() -> None:
    reply = (
        "Я человек с многолетним опытом трезвости, я ваш спонсор "
        "и помогу разобрать тягу. Расскажите о своей ситуации подробнее."
    )
    assert _is_direct_meta_reply(reply) is False


def test_live_lane_covers_required_families_and_sent_message_checks() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    live_source = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "held_out_scenarios" in live_source
    assert "book_grounded_families" in live_source
    assert "live-substantive-grounded-book-answer" in live_source
    assert "live-meta-direct-" in live_source
    assert "live-continuation-helpful-" in live_source
    assert "_is_direct_meta_reply" in live_source
    assert "_is_avoiding_clarification_text" in live_source
    assert "_is_service_link_only_text" in live_source
    assert "_is_quote_only_text" in live_source
    assert "_has_declarative_substance" in live_source
    # Sent Telegram messages plus this turn's telemetry, never a draft.
    assert "api.sent_texts[-1]" in live_source
    assert "last_telemetry_for_thread" in live_source
    # Held-out colloquial/typo plus an explicit what-to-do continuation.
    assert "тянеет" in live_source
    assert "что мне делать" in live_source


def test_no_exact_live_question_special_cases_in_product() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    frozen_fragments = (
        "тянет выпить",
        "тянеет выпить",
        "что мне делать-то",
        "ссора из-за моей выпивки",
        "Поругались дома",
        "не могу успокоиться и уснуть",
        "мысли крутятся",
        "покупать акции",
        "выгоднее купить",
        "покончить с собой",
        "Не хочу жить",
        "одному не получается",
        "чем помочь можешь",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source


def test_gate_e_slo_stays_strict() -> None:
    from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS

    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000
