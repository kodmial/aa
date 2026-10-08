"""P0 kodmial/aa#263 recurrence 2: repair persistent Gate C drinking-11.

Proven product failure on exact main a19c148 (run 37835226373):

- C:live-book-grounding-substantive-drinking-11 on live-production-path
  (a held-out colloquial/typo craving turn fails grounding while its
  correctly spelled sibling passes).

Recurrence analysis against the prior repair (c34dd9e): the previous
cycle fixed empty-pack recovery validation and the four-token
continuation bound, which converged the long-conversation-17 family,
but drinking-11 persists unchanged. The dominant persistent cause is
therefore not recovery/continuation: it is craving-verb and colloquial
fragmentation at the retrieval/domain boundary, turn-independent:

- lexical stems fragmented the craving verb (correct and typo forms
  never met the noun root), so the typo turn depended on a single
  shared token while its sibling kept more overlap;
- the recovery-domain vocabularies missed the verb inflection entirely,
  so domain fallbacks in adequacy and live relevance hinged on one
  token instead of the craving root;
- the colloquial spoken form stayed a distinct FTS term from its
  literary base.

Strategy change at the responsible boundary (not another
recovery/continuation retune): typo-tolerant retrieval normalization
(repeated-vowel collapse, colloquial mapping, craving-root unification)
plus matching domain stems in production adequacy and the live
qualification lane. Generic signals only, never an exact-question
whitelist; Product Contract #110 unchanged.
"""

from __future__ import annotations

import pathlib


def test_repeated_vowel_typo_collapses() -> None:
    from aa.retrieval.normalize import normalize_ru

    assert normalize_ru("тянеет") == "тянет"
    # Double consonants are meaningful in Russian and stay distinct.
    assert normalize_ru("ссора") == "ссора"
    assert normalize_ru("поддержка") == "поддержка"


def test_colloquial_tokens_map_to_literary_base() -> None:
    from aa.retrieval.normalize import ru_tokens

    assert "что" in ru_tokens("че делать")
    assert "че" not in ru_tokens("че делать")


def test_craving_verb_and_noun_share_retrieval_root() -> None:
    from aa.retrieval.normalize import ru_stem, stemmed_norm_text

    assert ru_stem("тянет") == "тяг"
    assert ru_stem("тянеет") == "тяг"
    assert ru_stem("тяга") == "тяг"
    assert ru_stem("тяну") == "тяг"
    typo_norm = stemmed_norm_text("опять тянеет выпить")
    core_norm = stemmed_norm_text("очень тянет выпить")
    assert "тяг" in typo_norm.split()
    assert "тяг" in core_norm.split()
    # Unrelated double-consonant stems are not merged by the vowel rule.
    assert ru_stem("поддержка") == "поддержк"
    assert ru_stem("ссора") == "ссор"


def test_craving_verb_counts_as_recovery_domain() -> None:
    from aa.conversation.answer_adequacy import _has_recovery_domain

    assert _has_recovery_domain("тянет") is True
    assert _has_recovery_domain("тянеет") is True
    assert _has_recovery_domain("тяну") is True
    assert _has_recovery_domain("вечер спокойно") is False


def test_live_relevance_domain_covers_craving_verb() -> None:
    from aa.qualification.product_contract_live import _assess_prompt_reply_relevance

    support_reply = "Стоит обратиться к спонсору и прийти на собрание сообщества."
    assert _assess_prompt_reply_relevance("вечером тянет", support_reply) is True
    assert _assess_prompt_reply_relevance("вечером тянеет", support_reply) is True
    assert (
        _assess_prompt_reply_relevance(
            "Вечером тяжело пережить тягу",
            "Ведите финансовый бюджет спокойно.",
        )
        is False
    )


def test_typo_and_core_queries_share_lexical_terms() -> None:
    from aa.retrieval.normalize import stemmed_norm_text

    typo_terms = set(stemmed_norm_text("Под вечер опять тянеет выпить че делать").split())
    core_terms = set(stemmed_norm_text("К вечеру очень тянет выпить как обходиться").split())
    assert "тяг" in typo_terms & core_terms
    assert "выпит" in typo_terms & core_terms
    assert "что" in typo_terms


def test_no_exact_live_question_special_cases() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "retrieval" / "normalize.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "retrieval" / "evidence.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "тянет выпить",
            "тянеет выпить",
            "Поругались дома",
            "покупать акции",
            "покончить с собой",
            "чем помочь можешь",
            "одному не получается",
        ):
            assert fragment not in source
