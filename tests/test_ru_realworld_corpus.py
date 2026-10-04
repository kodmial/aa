"""Qualification tests for the Russian real-world alcohol-help v1 corpus (#61).

The JSONL fixture is authoritative and must stay byte-stable: these tests
fail on accidental deletion, duplicate ids, count drift, malformed journeys,
unresolved provenance, verbatim copying, fabricated frequency statistics,
encoded answers, or routing-review regressions against #21/#46.
"""

from __future__ import annotations

import json
import pathlib

from aa.qualification.ru_realworld import (
    CORPUS_REL,
    EXPECTED_JOURNEY_TURNS,
    EXPECTED_JOURNEYS,
    EXPECTED_SINGLE_TURNS,
    EXPECTED_TOTAL_UTTERANCES,
    SOURCES_REL,
    VERSION_REL,
    build_version_payload,
    find_repo_root,
    load_records,
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
    assert summary.journey_turns == EXPECTED_JOURNEY_TURNS
    assert summary.total_utterances == EXPECTED_TOTAL_UTTERANCES


def test_corpus_files_are_strict_utf8() -> None:
    root = _root()
    for relative in (CORPUS_REL, SOURCES_REL, VERSION_REL):
        raw = (root / relative).read_bytes()
        raw.decode("utf-8")  # raises on invalid UTF-8
        assert raw, f"{relative} must not be empty"


def test_ids_cover_exact_sequences_without_duplicates() -> None:
    _, singles, journeys = load_records(_root() / CORPUS_REL)
    single_ids = [str(item["id"]) for item in singles]
    assert single_ids == [f"RU-S-{number:03d}" for number in range(1, 201)]
    journey_ids = [str(item["id"]) for item in journeys]
    assert journey_ids == [f"RU-J-{number:03d}" for number in range(1, 31)]
    assert len(set(single_ids + journey_ids)) == 230


def test_journeys_hold_five_sequential_turns_each() -> None:
    _, _, journeys = load_records(_root() / CORPUS_REL)
    total = 0
    for record in journeys:
        turns = record["turns"]
        assert len(turns) == 5
        assert [turn["turn"] for turn in turns] == [1, 2, 3, 4, 5]
        assert turns[0]["context_dependent"] is False
        total += len(turns)
    assert total == 150


def test_utterances_are_deduplicated() -> None:
    _, singles, journeys = load_records(_root() / CORPUS_REL)

    def _norm(text: str) -> str:
        return " ".join(text.strip().casefold().split())

    seen: dict[str, str] = {}
    for record in singles:
        norm = _norm(str(record["utterance"]))
        assert norm not in seen, f"duplicate utterance in {record['id']}"
        seen[norm] = str(record["id"])
    for record in journeys:
        for turn in record["turns"]:
            norm = _norm(str(turn["utterance"]))
            owner = f"{record['id']} turn {turn['turn']}"
            assert norm not in seen, f"duplicate utterance: {owner}"
            seen[norm] = owner
    assert len(seen) == EXPECTED_TOTAL_UTTERANCES


def test_every_provenance_id_resolves_and_sources_are_used() -> None:
    root = _root()
    _, singles, journeys = load_records(root / CORPUS_REL)
    payload = load_sources(root / SOURCES_REL)
    sources = payload["sources"]
    assert len(sources) == 15
    valid = {str(entry["id"]) for entry in sources}
    referenced: set[str] = set()
    for record in (*singles, *journeys):
        for provenance_id in record["provenance_ids"]:
            assert provenance_id in valid, f"unresolved {provenance_id}"
            referenced.add(str(provenance_id))
    assert referenced == valid


def test_no_verbatim_copy_search_stats_or_encoded_answers() -> None:
    root = _root()
    _, singles, journeys = load_records(root / CORPUS_REL)
    sources_text = (root / SOURCES_REL).read_text(encoding="utf-8")
    for record in singles:
        assert str(record["utterance"]).strip() not in sources_text
        for key in record:
            assert key not in (
                "expected_answer",
                "assistant_response",
                "desired_response",
                "ideal_answer",
            )
    for record in journeys:
        for turn in record["turns"]:
            assert str(turn["utterance"]).strip() not in sources_text
        for key in record:
            assert key not in (
                "expected_answer",
                "assistant_response",
                "desired_response",
                "ideal_answer",
            )


def test_ordinary_and_clarify_inputs_stay_off_emergency_path() -> None:
    _, singles, journeys = load_records(_root() / CORPUS_REL)
    checked = 0
    for record in singles:
        if str(record["expected_route"]) in ("ordinary", "clarify"):
            assert not classify_emergency(str(record["utterance"])).is_emergency
            checked += 1
    for record in journeys:
        for turn in record["turns"]:
            if str(turn["expected_route"]) in ("ordinary", "clarify"):
                assert not classify_emergency(str(turn["utterance"])).is_emergency
                checked += 1
    assert checked == (160 + 20) + (123 + 18)


def test_emergency_labels_stay_on_reviewed_escalations() -> None:
    _, singles, journeys = load_records(_root() / CORPUS_REL)
    for record in singles:
        if str(record["expected_route"]) == "emergency":
            assert str(record["topic"]) in ("withdrawal_emergency", "hopeless_selfharm")
    for record in journeys:
        for turn in record["turns"]:
            if str(turn["expected_route"]) == "emergency":
                assert str(record["journey"]) in (
                    "sleep_to_withdrawal_boundary",
                    "withdrawal_escalation",
                    "selfharm_escalation",
                    "family_member_emergency",
                )


def test_slang_ambiguity_and_session_fixtures_preserved() -> None:
    _, singles, journeys = load_records(_root() / CORPUS_REL)
    by_single = {str(item["id"]): item for item in singles}
    for number in range(191, 201):
        record = by_single[f"RU-S-{number:03d}"]
        assert "slang" in record["rubric_tags"]
        assert "meaning_preservation" in record["rubric_tags"]
        assert record["expected_route"] == "ordinary"
    by_journey = {str(item["id"]): item for item in journeys}
    ambiguous = by_journey["RU-J-013"]["turns"][0]
    assert ambiguous["utterance"] == "Я сорвался."
    assert ambiguous["expected_route"] == "clarify"
    reset_turns = [turn["utterance"] for turn in by_journey["RU-J-027"]["turns"]]
    assert "/new" in reset_turns
    med_routes = [turn["expected_route"] for turn in by_journey["RU-J-017"]["turns"]]
    assert "clarify" in med_routes


def test_stable_version_checksum_matches_validated_corpus() -> None:
    root = _root()
    summary = validate(root)
    assert summary.corpus_sha256 == sha256_file(root / CORPUS_REL)
    assert summary.sources_sha256 == sha256_file(root / SOURCES_REL)
    recorded = json.loads((root / VERSION_REL).read_text(encoding="utf-8"))
    assert recorded == build_version_payload(summary)
    assert recorded["counts"]["total_user_utterances"] == 350


def test_validation_is_deterministic() -> None:
    assert validate(_root()) == validate(_root())
