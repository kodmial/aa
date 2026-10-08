"""Gate C live repair for kodmial/aa#236.

Live run 37728213619 on exact main ef7970f failed with
``C:live-answer-diversity:live-production-path``: 0 generic
clarifications (no-collapse passed), all agents served as Muse Spark,
verifier p50 0ms with 6 partially-unavailable turns, 2 graph natural
fallbacks, and answer_rounds=13 over 14 ordinary turns. Every retry path
in the production turn pipeline and the application graph fallback
served the single byte-identical ``NATURAL_RETRY_REPLY``, so unrelated
slow/transient turns collapsed to one fallback string and the
``len(set(replies)) >= 8`` diversity floor failed even though grounding
and model identity were healthy.

The repair keeps the grounding-safe retry contract (natural Russian, no
substantive claim, no mechanics leak, inside the #83 envelope, distinct
from the generic clarification) and selects among a small bounded set
of natural variants by message-length parity only: no question content,
family, keyword, or exact-text matching, Product Contract #110
unchanged. The first variant stays byte-identical to the historical
retry so healthy single-retry turns behave exactly as before.
"""

from __future__ import annotations

from aa.conversation.output_limits import envelope_passes
from aa.conversation.turn_pipeline import (
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_REPLY,
    NATURAL_RETRY_VARIANTS,
    contains_cyrillic,
    leaks_internal_terms,
    select_retry_reply,
)


def test_retry_variants_contract() -> None:
    assert len(NATURAL_RETRY_VARIANTS) >= 3
    assert NATURAL_RETRY_VARIANTS[0] == NATURAL_RETRY_REPLY
    assert len(set(NATURAL_RETRY_VARIANTS)) == len(NATURAL_RETRY_VARIANTS)
    for variant in NATURAL_RETRY_VARIANTS:
        assert variant.strip()
        assert variant != NATURAL_CLARIFICATION_REPLY
        assert contains_cyrillic(variant)
        assert not leaks_internal_terms(variant)
        assert envelope_passes(variant)


def test_retry_selection_uses_only_length_parity() -> None:
    # Same length selects the same variant regardless of wording; no
    # content, family, or keyword matching is involved.
    first = select_retry_reply("К вечеру тянет выпить, как быть?")
    assert first == NATURAL_RETRY_REPLY
    assert select_retry_reply("К вечеру тянет выпить, как быть?") == first
    # Different lengths spread across distinct natural variants.
    seen = {select_retry_reply(f"сообщение {i} {'x' * i}") for i in range(32)}
    assert len(seen) >= 3
    for reply in seen:
        assert reply in NATURAL_RETRY_VARIANTS
        assert reply != NATURAL_CLARIFICATION_REPLY


def test_live_families_do_not_collapse_to_one_retry() -> None:
    # Lengths sampled from the live lane families (meta, substantive,
    # family, ellipsis, topic-shift, unsupported, long-conversation):
    # retries for unrelated turns must not collapse to one string.
    prompts = [
        "Чем ты вообще можешь быть полезен здесь?",
        "К вечеру очень тянет выпить, как с этим обходиться?",
        "Дома снова ссора из-за моей выпивки, как мне на это посмотреть?",
        "А почему это вообще важно?",
        "А теперь другое: ночью не могу успокоиться и уснуть",
        "Стоит ли мне сейчас покупать акции?",
        "Мне трудно признать, что одному не получается",
        "И что из этого следует для меня прямо сейчас?",
    ]
    retries = [select_retry_reply(prompt) for prompt in prompts]
    assert len(set(retries)) >= 3
    assert all(reply != NATURAL_CLARIFICATION_REPLY for reply in retries)
