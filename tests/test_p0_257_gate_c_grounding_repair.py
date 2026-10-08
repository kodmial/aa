"""P0 kodmial/aa#257: repair Gate C live-book-grounding failure.

Proven product failure on exact main 8a3971423a9ae9cbf835ea40602f5e8e581b26b9
(run 37784607712):

- C:live-book-grounding-substantive-drinking-2 on live-production-path
  (all 8 book-grounded turns failed while every relevance check passed;
  verifier passed 12/16 with 4 unsupported, answer served 10 plus
  narrowed-supported 3, planner ok 12 / replanned 4, retrieval
  evidence-ready 12 / repaired 4, response_units_total=39 max=3).

Diagnosis: the qualification grounding predicate required strict
``verified_book_units == response_units``. A natural helpful reply mixes
one book-supported guidance unit with brief conversational glue
(empathy/acknowledgement scoped as conversation_glue, supported=true),
so verified < response_units for every multi-sentence answer while
relevance still passes. The same equality also contradicts the
narrowing contract: a narrowed-supported turn serves only the supported
subset, so verified < response_units is expected there by design.

Fix at the grounding-predicate boundary (turn-independent, no
exact-question special cases, Product Contract #110 unchanged): a
served turn must hold a passing verifier outcome (no unsupported
material delivered) with at least one book unit and sane counts;
narrowed turns keep the supported-subset contract with at least one
book unit. All other fail-closed gates (adequacy, failure category,
planner reason, split statuses, substance, relevance, safety) stay
strict.
"""

from __future__ import annotations


def _served_snapshot(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 3,
        "verified_book_units": 1,
        "response_units": 2,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "planner_reason": "substantive-with-queries",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    snapshot.update(overrides)
    return snapshot


_MIXED_REPLY = "Понимаю, вам сейчас тяжело. Поддержка рядом помогает пережить тягу сегодня."


def test_mixed_glue_plus_book_served_passes() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    assert _is_grounded_substantive_reply(_served_snapshot(), _MIXED_REPLY) is True


def test_single_book_unit_served_still_passes() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(verified_book_units=1, response_units=1)
    assert (
        _is_grounded_substantive_reply(snapshot, "Поддержка рядом помогает пережить тягу сегодня.")
        is True
    )


def test_served_with_unsupported_verifier_fails() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(verifier_outcome="unsupported")
    assert _is_grounded_substantive_reply(snapshot, _MIXED_REPLY) is False


def test_narrowed_supported_subset_passes() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(
        answer_outcome="narrowed-supported",
        verifier_outcome="unsupported",
        verified_book_units=1,
        response_units=3,
    )
    assert _is_grounded_substantive_reply(snapshot, _MIXED_REPLY) is True


def test_insane_counts_fail_closed() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(verified_book_units=3, response_units=2)
    assert _is_grounded_substantive_reply(snapshot, _MIXED_REPLY) is False


def test_zero_verified_units_still_fails() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(verified_book_units=0)
    assert _is_grounded_substantive_reply(snapshot, _MIXED_REPLY) is False


def test_legacy_snapshot_without_verifier_outcome_keeps_passing() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _served_snapshot(verified_book_units=1, response_units=1)
    del snapshot["verifier_outcome"]
    assert (
        _is_grounded_substantive_reply(snapshot, "Поддержка рядом помогает пережить тягу сегодня.")
        is True
    )


def test_no_exact_live_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "тянет выпить",
            "тянеет выпить",
            "Поругались дома",
            "одному не получается",
            "покупать акции",
            "покончить с собой",
        ):
            assert fragment not in source
