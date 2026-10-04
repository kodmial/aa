"""Qualification tests for the Russian real-world alcohol-help v1_1 corpus (#61).

The v1 seed prompts stay byte-stable; v1_1 freezes generator-visible input
and evaluation-only oracle projections.  These tests fail on the forbidden
contract: ``clarify``/``ordinary`` as router states, ``/new`` as a normal
model turn, label leakage into input, or oracle/input divergence.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from aa.qualification.ru_realworld import (
    ALLOWED_EMERGENCY_CATEGORIES,
    ALLOWED_RESPONSE_MODES,
    ALLOWED_SAFETY_DECISIONS,
    CONTROL_JOURNEY_ID,
    CORPUS_REL,
    EXPECTED_CONTROL_EVENTS,
    EXPECTED_JOURNEYS,
    EXPECTED_SINGLE_TURNS,
    EXPECTED_SUBSTANTIVE_JOURNEY_TURNS,
    EXPECTED_TOTAL_SUBSTANTIVE,
    INPUT_REL,
    ORACLE_REL,
    ORACLE_SINGLE_KEYS,
    ORACLE_SINGLE_OPTIONAL_KEYS,
    ORACLE_TURN_KEYS,
    ORACLE_TURN_OPTIONAL_KEYS,
    SOURCES_REL,
    VERSION_REL,
    RuRealWorldCorpusError,
    _is_verbatim_copy,
    _source_theme_texts,
    build_version_payload,
    find_repo_root,
    load_input,
    load_oracle,
    load_sources,
    sha256_file,
    validate,
)
from aa.safety.emergency import classify_emergency


def _root() -> pathlib.Path:
    return find_repo_root()


def test_corpus_validates_mechanically_with_fixed_counts() -> None:
    summary = validate(_root())
    assert summary.single_turn == EXPECTED_SINGLE_TURNS
    assert summary.journeys == EXPECTED_JOURNEYS
    assert summary.substantive_journey_turns == EXPECTED_SUBSTANTIVE_JOURNEY_TURNS
    assert summary.control_events == EXPECTED_CONTROL_EVENTS
    assert summary.total_substantive == EXPECTED_TOTAL_SUBSTANTIVE


def test_corpus_files_are_strict_utf8() -> None:
    root = _root()
    for relative in (CORPUS_REL, INPUT_REL, ORACLE_REL, SOURCES_REL, VERSION_REL):
        raw = (root / relative).read_bytes()
        raw.decode("utf-8")  # raises on invalid UTF-8
        assert raw, f"{relative} must not be empty"


def test_ids_cover_exact_sequences_without_duplicates() -> None:
    _, input_singles, input_journeys = load_input(_root() / INPUT_REL)
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    assert [str(i["id"]) for i in input_singles] == [f"RU-S-{n:03d}" for n in range(1, 201)]
    assert [str(i["id"]) for i in oracle_singles] == [f"RU-S-{n:03d}" for n in range(1, 201)]
    assert [str(i["id"]) for i in input_journeys] == [f"RU-J-{n:03d}" for n in range(1, 31)]
    assert [str(i["id"]) for i in oracle_journeys] == [f"RU-J-{n:03d}" for n in range(1, 31)]
    input_ids = {str(i["id"]) for i in input_singles} | {str(i["id"]) for i in input_journeys}
    assert len(input_ids) == 230


def test_safety_decisions_use_only_allow_emergency_block() -> None:
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    seen: set[str] = set()
    for record in oracle_singles:
        decision = str(record["expected_safety_decision"])
        assert decision in ALLOWED_SAFETY_DECISIONS
        assert decision not in ("ordinary", "clarify")
        assert "expected_route" not in record
        seen.add(decision)
    for record in oracle_journeys:
        for turn in record["turns"]:
            decision = str(turn["expected_safety_decision"])
            assert decision in ALLOWED_SAFETY_DECISIONS
            assert decision not in ("ordinary", "clarify")
            assert "expected_route" not in turn
            seen.add(decision)
    assert "allow" in seen
    assert "emergency" in seen
    assert "block" in seen
    assert set(ALLOWED_SAFETY_DECISIONS) == {"allow", "emergency", "block"}


def test_clarify_is_response_mode_never_router_state() -> None:
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    for record in oracle_singles:
        assert str(record["expected_response_mode"]) in ALLOWED_RESPONSE_MODES
    for record in oracle_journeys:
        for turn in record["turns"]:
            assert str(turn["expected_response_mode"]) in ALLOWED_RESPONSE_MODES
    by_single = {str(i["id"]): i for i in oracle_singles}
    for number in range(61, 71):
        record = by_single[f"RU-S-{number:03d}"]
        assert record["expected_safety_decision"] == "allow"
        assert record["expected_response_mode"] == "medical_boundary_clarification"
    by_journey = {str(i["id"]): i for i in oracle_journeys}
    ambiguous = {int(t["turn"]): t for t in by_journey["RU-J-013"]["turns"]}
    assert ambiguous[1]["expected_safety_decision"] == "allow"
    assert ambiguous[1]["expected_response_mode"] == "ambiguity_clarification"
    secret = by_journey["RU-J-017"]["turns"]
    assert "medical_refusal_boundary" in [str(t["expected_response_mode"]) for t in secret]
    for turn in secret:
        assert str(turn["expected_safety_decision"]) in ("allow", "block")
    for turn in secret:
        if str(turn["expected_response_mode"]) == "medical_refusal_boundary":
            assert str(turn["expected_safety_decision"]) == "block"
    refusal_singles = [
        record
        for record in oracle_singles
        if str(record["expected_response_mode"]) == "medical_refusal_boundary"
    ]
    assert refusal_singles, "oracle must carry medical refusal fixtures"
    for record in refusal_singles:
        assert str(record["expected_safety_decision"]) == "block"


def test_new_is_control_event_never_model_turn() -> None:
    _, input_singles, input_journeys = load_input(_root() / INPUT_REL)
    for record in input_singles:
        assert str(record["utterance"]).strip() != "/new"
    by_journey = {str(i["id"]): i for i in input_journeys}
    reset = by_journey[CONTROL_JOURNEY_ID]
    entries = list(reset["turns"])
    assert [e["turn"] for e in entries] == [1, 2, 3, 4, 5]
    control = entries[2]
    assert control["kind"] == "control"
    assert control["control"] == "session_reset"
    assert set(control.keys()) == {"control", "kind", "turn"}
    for entry in entries:
        if entry.get("kind") == "user":
            assert str(entry["utterance"]).strip() != "/new"
    _, _, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    oracle_by_id = {str(i["id"]): i for i in oracle_journeys}
    oracle_turns = {int(t["turn"]): t for t in oracle_by_id[CONTROL_JOURNEY_ID]["turns"]}
    assert sorted(oracle_turns) == [1, 2, 4, 5]
    assert oracle_turns[4]["requires_context"] is False
    assert oracle_turns[1]["requires_context"] is False


def test_malformed_control_event_and_new_as_turn_fail() -> None:
    _, input_singles, input_journeys = load_input(_root() / INPUT_REL)
    bad: dict[str, object] = {
        "type": "multi_turn_journey",
        "id": "RU-J-027",
        "journey": "new_session_reset",
        "turns": [
            {"kind": "user", "turn": 1, "utterance": "a"},
            {"kind": "user", "turn": 2, "utterance": "b"},
            {"kind": "user", "turn": 3, "utterance": "/new"},
            {"kind": "user", "turn": 4, "utterance": "c"},
            {"kind": "user", "turn": 5, "utterance": "d"},
        ],
    }
    _ = (input_singles, input_journeys)
    from aa.qualification.ru_realworld import _check_input_journey

    with pytest.raises(RuRealWorldCorpusError):
        _check_input_journey(bad, 26)


def test_input_oracle_separation_without_generator_leak() -> None:
    root = _root()
    _, input_singles, input_journeys = load_input(root / INPUT_REL)
    _, oracle_singles, oracle_journeys = load_oracle(root / ORACLE_REL)
    oracle_only = {
        "expected_safety_decision",
        "expected_response_mode",
        "topic",
        "provenance_ids",
        "rubric_tags",
        "book_relevance",
        "stage",
        "requires_context",
    }
    for record in input_singles:
        assert not (set(record.keys()) & oracle_only)
        assert "expected_route" not in record
    for record in input_journeys:
        for turn in record["turns"]:
            assert not (set(turn.keys()) & oracle_only)
            assert "expected_route" not in turn
    for record in oracle_singles:
        assert "utterance" not in record
        for forbidden in (
            "expected_answer",
            "assistant_response",
            "desired_response",
            "ideal_answer",
        ):
            assert forbidden not in record
    for record in oracle_journeys:
        assert "utterance" not in record
        for turn in record["turns"]:
            assert "utterance" not in turn
    assert [str(r["id"]) for r in input_singles] == [str(r["id"]) for r in oracle_singles]
    assert [str(r["id"]) for r in input_journeys] == [str(r["id"]) for r in oracle_journeys]
    for input_record, oracle_record in zip(input_journeys, oracle_journeys, strict=True):
        input_turns = sorted(
            int(t["turn"]) for t in input_record["turns"] if t.get("kind") == "user"
        )
        oracle_turns = sorted(int(t["turn"]) for t in oracle_record["turns"])
        assert input_turns == oracle_turns


def test_oracle_carries_all_required_fields() -> None:
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    base_single = tuple(sorted(ORACLE_SINGLE_KEYS))
    extended_single = tuple(sorted((*ORACLE_SINGLE_KEYS, *ORACLE_SINGLE_OPTIONAL_KEYS)))
    base_turn = tuple(sorted(ORACLE_TURN_KEYS))
    extended_turn = tuple(sorted((*ORACLE_TURN_KEYS, *ORACLE_TURN_OPTIONAL_KEYS)))
    for record in oracle_singles:
        assert tuple(sorted(record.keys())) in (base_single, extended_single)
        assert record["book_relevance"] in ("required", "optional", "not-applicable")
        assert isinstance(record["requires_context"], bool)
        assert isinstance(record["requests_exact_quote"], bool)
        assert isinstance(record["forbidden_inferences"], list)
        assert record["forbidden_inferences"]
        assert isinstance(record["safety_boundary_tags"], list)
        assert isinstance(record["provenance_ids"], list) and record["provenance_ids"]
        assert "russian_realworld" in record["rubric_tags"]
        if str(record["expected_safety_decision"]) == "emergency":
            categories = record.get("expected_emergency_categories")
            assert isinstance(categories, list) and categories
            for item in categories:
                assert item in ALLOWED_EMERGENCY_CATEGORIES
        elif "expected_emergency_categories" in record:
            assert record["expected_emergency_categories"] == []
    for record in oracle_journeys:
        for turn in record["turns"]:
            assert tuple(sorted(turn.keys())) in (base_turn, extended_turn)
            assert turn["book_relevance"] in ("required", "optional", "not-applicable")
            assert isinstance(turn["requires_context"], bool)
            assert isinstance(turn["requests_exact_quote"], bool)
            assert isinstance(turn["forbidden_inferences"], list)
            assert turn["forbidden_inferences"]
            if str(turn["expected_safety_decision"]) == "emergency":
                categories = turn.get("expected_emergency_categories")
                assert isinstance(categories, list) and categories
                for item in categories:
                    assert item in ALLOWED_EMERGENCY_CATEGORIES
            elif "expected_emergency_categories" in turn:
                assert turn["expected_emergency_categories"] == []
    by_journey = {str(i["id"]): i for i in oracle_journeys}
    quote = {int(t["turn"]): t for t in by_journey["RU-J-023"]["turns"]}
    assert quote[2]["requests_exact_quote"] is True
    assert quote[2]["book_relevance"] == "required"


def test_journeys_hold_five_entries_with_post_reset_context() -> None:
    _, _, input_journeys = load_input(_root() / INPUT_REL)
    total_user = 0
    total_control = 0
    for record in input_journeys:
        turns = record["turns"]
        assert len(turns) == 5
        assert [t["turn"] for t in turns] == [1, 2, 3, 4, 5]
        for turn in turns:
            if turn["kind"] == "user":
                total_user += 1
            else:
                total_control += 1
    assert total_user == EXPECTED_SUBSTANTIVE_JOURNEY_TURNS
    assert total_control == EXPECTED_CONTROL_EVENTS


def test_seed_new_reclassification_is_explicit() -> None:
    from aa.qualification.ru_realworld import (
        CONTROL_TURN_NUMBER,
        EXPECTED_SEED_JOURNEY_USER_TURNS,
        load_records,
    )

    root = _root()
    _, seed_singles, seed_journeys = load_records(root / CORPUS_REL)
    assert len(seed_singles) == EXPECTED_SINGLE_TURNS
    seed_journey_turns = sum(len(record["turns"]) for record in seed_journeys)
    assert seed_journey_turns == EXPECTED_SEED_JOURNEY_USER_TURNS
    _, input_singles, input_journeys = load_input(root / INPUT_REL)
    assert len(input_singles) == EXPECTED_SINGLE_TURNS
    by_journey = {str(item["id"]): item for item in input_journeys}
    reset = by_journey[CONTROL_JOURNEY_ID]
    assert reset["turns"][CONTROL_TURN_NUMBER - 1]["kind"] == "control"
    substantive = sum(
        len([t for t in item["turns"] if t.get("kind") == "user"]) for item in input_journeys
    )
    assert substantive == EXPECTED_SUBSTANTIVE_JOURNEY_TURNS
    assert substantive + EXPECTED_CONTROL_EVENTS == EXPECTED_SEED_JOURNEY_USER_TURNS
    assert len(input_singles) + substantive == EXPECTED_TOTAL_SUBSTANTIVE


def test_utterances_are_deduplicated() -> None:
    _, input_singles, input_journeys = load_input(_root() / INPUT_REL)

    def _norm(text: str) -> str:
        return " ".join(text.strip().casefold().split())

    seen: dict[str, str] = {}
    for record in input_singles:
        norm = _norm(str(record["utterance"]))
        assert norm not in seen, f"duplicate utterance in {record['id']}"
        seen[norm] = str(record["id"])
    for record in input_journeys:
        for turn in record["turns"]:
            if turn.get("kind") != "user":
                continue
            norm = _norm(str(turn["utterance"]))
            owner = f"{record['id']} turn {turn['turn']}"
            assert norm not in seen, f"duplicate utterance: {owner}"
            seen[norm] = owner
    assert len(seen) == EXPECTED_TOTAL_SUBSTANTIVE


def test_every_provenance_id_resolves_and_sources_are_used() -> None:
    from aa.qualification.ru_realworld import load_sources

    root = _root()
    _, oracle_singles, oracle_journeys = load_oracle(root / ORACLE_REL)
    payload = load_sources(root / SOURCES_REL)
    sources = payload["sources"]
    assert len(sources) == 15
    valid = {str(entry["id"]) for entry in sources}
    referenced: set[str] = set()
    for record in oracle_singles:
        for provenance_id in record["provenance_ids"]:
            assert provenance_id in valid, f"unresolved {provenance_id}"
            referenced.add(str(provenance_id))
    for record in oracle_journeys:
        for provenance_id in record["provenance_ids"]:
            assert provenance_id in valid, f"unresolved {provenance_id}"
            referenced.add(str(provenance_id))
    assert referenced == valid


def test_no_verbatim_copy_search_stats_or_encoded_answers() -> None:
    root = _root()
    _, input_singles, input_journeys = load_input(root / INPUT_REL)
    _, oracle_singles, oracle_journeys = load_oracle(root / ORACLE_REL)
    payload = load_sources(root / SOURCES_REL)
    sources = payload.get("sources", [])
    by_source_id = {str(e["id"]): e for e in sources if isinstance(e, dict)}
    theme_texts = _source_theme_texts(by_source_id)
    for record in input_singles:
        assert not _is_verbatim_copy(str(record["utterance"]), theme_texts)
        for key in record:
            assert key not in (
                "expected_answer",
                "assistant_response",
                "desired_response",
                "ideal_answer",
            )
    for record in input_journeys:
        for turn in record["turns"]:
            if turn.get("kind") != "user":
                continue
            assert not _is_verbatim_copy(str(turn["utterance"]), theme_texts)
    for record in (*oracle_singles, *oracle_journeys):
        for key in record:
            assert key not in (
                "expected_answer",
                "assistant_response",
                "desired_response",
                "ideal_answer",
            )


def test_allow_inputs_stay_off_emergency_path() -> None:
    root = _root()
    _, input_singles, input_journeys = load_input(root / INPUT_REL)
    _, oracle_singles, oracle_journeys = load_oracle(root / ORACLE_REL)
    utterances: dict[str, str] = {str(r["id"]): str(r["utterance"]) for r in input_singles}
    for record in input_journeys:
        for turn in record["turns"]:
            if turn.get("kind") == "user":
                utterances[f"{record['id']}#{turn['turn']}"] = str(turn["utterance"])
    checked = 0
    for record in oracle_singles:
        if str(record["expected_safety_decision"]) == "allow":
            assert not classify_emergency(utterances[str(record["id"])]).is_emergency
            checked += 1
    for record in oracle_journeys:
        for turn in record["turns"]:
            if str(turn["expected_safety_decision"]) == "allow":
                key = f"{record['id']}#{turn['turn']}"
                assert not classify_emergency(utterances[key]).is_emergency
                checked += 1
    assert checked > 0


def test_emergency_labels_stay_on_reviewed_escalations() -> None:
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    for record in oracle_singles:
        if str(record["expected_safety_decision"]) == "emergency":
            assert str(record["topic"]) in ("withdrawal_emergency", "hopeless_selfharm")
    for record in oracle_journeys:
        for turn in record["turns"]:
            if str(turn["expected_safety_decision"]) == "emergency":
                assert str(record["journey"]) in (
                    "sleep_to_withdrawal_boundary",
                    "withdrawal_escalation",
                    "selfharm_escalation",
                    "family_member_emergency",
                )


def test_slang_ambiguity_and_session_fixtures_preserved() -> None:
    _, _, input_journeys = load_input(_root() / INPUT_REL)
    _, oracle_singles, oracle_journeys = load_oracle(_root() / ORACLE_REL)
    by_single = {str(item["id"]): item for item in oracle_singles}
    for number in range(191, 201):
        record = by_single[f"RU-S-{number:03d}"]
        assert "slang" in record["rubric_tags"]
        assert "meaning_preservation" in record["rubric_tags"]
        assert record["expected_safety_decision"] == "allow"
        assert record["expected_response_mode"] == "ordinary_support"
    by_journey = {str(item["id"]): item for item in oracle_journeys}
    ambiguous = {int(t["turn"]): t for t in by_journey["RU-J-013"]["turns"]}
    assert ambiguous[1]["expected_response_mode"] == "ambiguity_clarification"
    assert ambiguous[1]["expected_safety_decision"] == "allow"
    assert input_journeys is not None
    med_modes = [str(t["expected_response_mode"]) for t in by_journey["RU-J-017"]["turns"]]
    assert "medical_refusal_boundary" in med_modes


def test_stable_version_checksum_matches_validated_corpus() -> None:
    root = _root()
    summary = validate(root)
    assert summary.input_sha256 == sha256_file(root / INPUT_REL)
    assert summary.oracle_sha256 == sha256_file(root / ORACLE_REL)
    assert summary.sources_sha256 == sha256_file(root / SOURCES_REL)
    recorded = json.loads((root / VERSION_REL).read_text(encoding="utf-8"))
    assert recorded == build_version_payload(summary)
    assert recorded["counts"]["total_substantive_utterances"] == 349
    assert recorded["safety_contract"]["decisions"] == ["allow", "emergency", "block"]
    assert recorded["safety_contract"]["clarify_is_router_state"] is False


def test_validation_is_deterministic() -> None:
    assert validate(_root()) == validate(_root())


def test_near_duplicate_ids_fail() -> None:
    from aa.qualification.ru_realworld import _check_near_duplicate_ids

    with pytest.raises(RuRealWorldCorpusError):
        _check_near_duplicate_ids(["RU-S-001", "ru-s-001"], "input")
    with pytest.raises(RuRealWorldCorpusError):
        _check_near_duplicate_ids(["RU-S-001", "RU-S-01"], "oracle")
    # Sequential ids remain distinct.
    _check_near_duplicate_ids(["RU-S-001", "RU-S-002", "RU-J-001"], "input")


def test_provenance_diversity_beyond_single_qna_source() -> None:
    from aa.qualification.ru_realworld import _check_provenance_diversity, load_sources

    root = _root()
    payload = load_sources(root / SOURCES_REL)
    by_id = {str(e["id"]): e for e in payload["sources"]}
    kinds = {str(e["kind"]) for e in payload["sources"]}
    assert len(payload["sources"]) == 15
    assert len(kinds) >= 3
    _, oracle_singles, oracle_journeys = load_oracle(root / ORACLE_REL)
    referenced = {i for r in oracle_singles for i in r["provenance_ids"]}
    for r in oracle_journeys:
        referenced.update(r["provenance_ids"])
    assert referenced == set(by_id)
    _check_provenance_diversity(by_id, referenced)
