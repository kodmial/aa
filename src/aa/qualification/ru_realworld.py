"""Machine validation for the authoritative Russian real-world corpus (issue #61).

The JSONL fixture at ``qualification/ru_realworld_alcohol_help.v1.jsonl`` is
the authoritative v1 seed. This module validates it mechanically without
changing production runtime behavior:

- strict UTF-8 decoding, one JSON object per non-empty line;
- exact schema, id sequences, counts and journey structure;
- global ID uniqueness and normalized-utterance deduplication;
- every ``provenance_ids`` entry resolves to the sources file;
- no identifiable narrative is copied verbatim and no search-volume
  statistics or encoded assistant answers are present;
- ``ordinary``/``clarify`` utterances never trigger the deterministic
  emergency classifier from issue #21 (false-positive bound);
- ``emergency`` labels appear only on reviewed withdrawal/self-harm
  escalation fixtures (semantic review, not classifier equality);
- slang/ambiguity fixtures required by issue #46 stay present and keep
  their meaning-preservation routing (``sorvalsya`` ambiguity, ``/new``
  reset, typo chain, secret-medication boundary).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.safety.emergency import classify_emergency

SCHEMA_VERSION = "ru-realworld-alcohol-help-v1"
PROVENANCE_SCHEMA_VERSION = "ru-realworld-provenance-v1"
CORPUS_REL = "qualification/ru_realworld_alcohol_help.v1.jsonl"
SOURCES_REL = "qualification/ru_realworld_alcohol_help.v1.sources.json"
VERSION_REL = "qualification/ru_realworld_alcohol_help.v1.version.json"
CORPUS_VERSION = "ru_realworld_alcohol_help.v1"

EXPECTED_SINGLE_TURNS = 200
EXPECTED_JOURNEYS = 30
EXPECTED_JOURNEY_TURNS = 150
EXPECTED_TOTAL_UTTERANCES = 350
EXPECTED_SOURCES = 15
EXPECTED_JOURNEY_LENGTH = 5

ALLOWED_ROUTES = ("ordinary", "clarify", "emergency")
SINGLE_REQUIRED_KEYS = (
    "expected_route",
    "id",
    "provenance_ids",
    "rubric_tags",
    "topic",
    "type",
    "utterance",
)
JOURNEY_REQUIRED_KEYS = ("id", "journey", "provenance_ids", "turns", "type")
TURN_REQUIRED_KEYS = ("context_dependent", "expected_route", "turn", "utterance")

EMERGENCY_SINGLE_TOPICS = ("withdrawal_emergency", "hopeless_selfharm")
EMERGENCY_JOURNEYS = (
    "sleep_to_withdrawal_boundary",
    "withdrawal_escalation",
    "selfharm_escalation",
    "family_member_emergency",
)
CLARIFY_SINGLE_TOPICS = ("anxiety_sleep", "medical_requests")

# Substrings that would indicate a fabricated search-volume/frequency claim
# inside a test utterance (the manifest/sources disclaimers are exempt).
FORBIDDEN_UTTERANCE_PATTERNS = (
    re.compile(r"google", re.IGNORECASE),
    re.compile(r"search.?volume", re.IGNORECASE),
    re.compile(r"запросов в месяц", re.IGNORECASE),
    re.compile(r"частота запросов", re.IGNORECASE),
    re.compile(r"\b\d+\s*%\s*(пользовател|запросов|людей)", re.IGNORECASE),
)


class RuRealWorldCorpusError(ValueError):
    """Raised when the v1 corpus fails mechanical validation."""


@dataclass(frozen=True)
class CorpusSummary:
    """Validated corpus identity and counts for downstream issues (#62)."""

    corpus_sha256: str
    sources_sha256: str
    single_turn: int
    journeys: int
    journey_turns: int
    total_utterances: int


def find_repo_root() -> Path:
    """Return the repository root containing the qualification directory."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / CORPUS_REL).exists() and (parent / SOURCES_REL).exists():
            return parent
    raise RuRealWorldCorpusError("repository root with qualification corpus not found")


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's raw bytes."""
    return sha256_bytes(path.read_bytes())


def _decode_utf8_strict(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuRealWorldCorpusError(f"{path.name}: not strict UTF-8: {exc}") from exc


def _norm_utterance(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def load_records(corpus_path: Path) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Load manifest, single-turn records and journeys from the JSONL file."""
    text = _decode_utf8_strict(corpus_path)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuRealWorldCorpusError("corpus file is empty")
    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuRealWorldCorpusError(f"manifest line is not valid JSON: {exc}") from exc
    if not isinstance(first, dict) or first.get("type") != "manifest":
        raise RuRealWorldCorpusError("first JSONL line must be the manifest record")
    singles: list[Any] = []
    journeys: list[Any] = []
    for lineno, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuRealWorldCorpusError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise RuRealWorldCorpusError(f"line {lineno}: record must be an object")
        record_type = record.get("type")
        if record_type == "single_turn":
            singles.append(record)
        elif record_type == "multi_turn_journey":
            journeys.append(record)
        else:
            raise RuRealWorldCorpusError(f"line {lineno}: unknown type {record_type!r}")
    return first, singles, journeys


def load_sources(sources_path: Path) -> dict[str, Any]:
    """Load and minimally shape-check the provenance sources file."""
    text = _decode_utf8_strict(sources_path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuRealWorldCorpusError(f"sources file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuRealWorldCorpusError("sources file must contain a JSON object")
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise RuRealWorldCorpusError(
            f"sources schema_version must be {PROVENANCE_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuRealWorldCorpusError("sources file must contain a non-empty 'sources' list")
    return payload


def _check_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuRealWorldCorpusError(
            f"manifest schema_version must be {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("language") != "ru":
        raise RuRealWorldCorpusError("manifest language must be 'ru'")
    counts = manifest.get("counts")
    expected_counts = {
        "single_turn": EXPECTED_SINGLE_TURNS,
        "multi_turn_journeys": EXPECTED_JOURNEYS,
        "multi_turn_user_turns": EXPECTED_JOURNEY_TURNS,
        "total_user_utterances": EXPECTED_TOTAL_UTTERANCES,
    }
    if (
        not isinstance(counts, dict)
        or {key: counts.get(key) for key in expected_counts} != expected_counts
    ):
        raise RuRealWorldCorpusError(f"manifest counts must equal {expected_counts}")
    if manifest.get("provenance_file") != SOURCES_REL:
        raise RuRealWorldCorpusError(f"manifest provenance_file must be {SOURCES_REL!r}")


def _check_single(record: dict[str, Any], index: int) -> None:
    if tuple(sorted(record.keys())) != SINGLE_REQUIRED_KEYS:
        raise RuRealWorldCorpusError(
            f"single_turn index {index}: keys must be exactly {list(SINGLE_REQUIRED_KEYS)}"
        )
    expected_id = f"RU-S-{index + 1:03d}"
    if record["id"] != expected_id:
        raise RuRealWorldCorpusError(
            f"single_turn index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    utterance = record["utterance"]
    if not isinstance(utterance, str) or not utterance.strip():
        raise RuRealWorldCorpusError(f"{record['id']}: utterance must be a non-empty string")
    if record["expected_route"] not in ALLOWED_ROUTES:
        raise RuRealWorldCorpusError(f"{record['id']}: unknown expected_route")
    if not isinstance(record["topic"], str) or not record["topic"].strip():
        raise RuRealWorldCorpusError(f"{record['id']}: topic must be a non-empty string")
    provenance_ids = record["provenance_ids"]
    if not isinstance(provenance_ids, list) or not provenance_ids:
        raise RuRealWorldCorpusError(f"{record['id']}: provenance_ids must be non-empty")
    for item in provenance_ids:
        if not isinstance(item, str) or not item.strip():
            raise RuRealWorldCorpusError(f"{record['id']}: provenance id must be a string")
    tags = record["rubric_tags"]
    if not isinstance(tags, list) or "russian_realworld" not in tags:
        raise RuRealWorldCorpusError(
            f"{record['id']}: rubric_tags must include 'russian_realworld'"
        )
    for pattern in FORBIDDEN_UTTERANCE_PATTERNS:
        if pattern.search(utterance):
            raise RuRealWorldCorpusError(
                f"{record['id']}: utterance carries a forbidden frequency/statistic claim"
            )


def _check_journey(record: dict[str, Any], index: int) -> None:
    if tuple(sorted(record.keys())) != JOURNEY_REQUIRED_KEYS:
        raise RuRealWorldCorpusError(
            f"journey index {index}: keys must be exactly {list(JOURNEY_REQUIRED_KEYS)}"
        )
    expected_id = f"RU-J-{index + 1:03d}"
    if record["id"] != expected_id:
        raise RuRealWorldCorpusError(
            f"journey index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    if not isinstance(record["journey"], str) or not record["journey"].strip():
        raise RuRealWorldCorpusError(f"{record['id']}: journey slug must be non-empty")
    provenance_ids = record["provenance_ids"]
    if not isinstance(provenance_ids, list) or not provenance_ids:
        raise RuRealWorldCorpusError(f"{record['id']}: provenance_ids must be non-empty")
    turns = record["turns"]
    if not isinstance(turns, list) or len(turns) != EXPECTED_JOURNEY_LENGTH:
        raise RuRealWorldCorpusError(
            f"{record['id']}: each journey must hold exactly {EXPECTED_JOURNEY_LENGTH} turns"
        )
    for position, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict) or tuple(sorted(turn.keys())) != TURN_REQUIRED_KEYS:
            raise RuRealWorldCorpusError(
                f"{record['id']} turn {position}: keys must be {list(TURN_REQUIRED_KEYS)}"
            )
        if turn["turn"] != position:
            raise RuRealWorldCorpusError(
                f"{record['id']}: turn numbers must run 1..{EXPECTED_JOURNEY_LENGTH}"
            )
        utterance = turn["utterance"]
        if not isinstance(utterance, str) or not utterance.strip():
            raise RuRealWorldCorpusError(f"{record['id']} turn {position}: empty utterance")
        if turn["expected_route"] not in ALLOWED_ROUTES:
            raise RuRealWorldCorpusError(f"{record['id']} turn {position}: bad expected_route")
        if not isinstance(turn["context_dependent"], bool):
            raise RuRealWorldCorpusError(
                f"{record['id']} turn {position}: context_dependent must be a bool"
            )
        for pattern in FORBIDDEN_UTTERANCE_PATTERNS:
            if pattern.search(utterance):
                raise RuRealWorldCorpusError(
                    f"{record['id']} turn {position}: forbidden frequency/statistic claim"
                )
    first = turns[0]
    if first["context_dependent"] is not False:
        raise RuRealWorldCorpusError(f"{record['id']}: first turn must open a new context")


def _check_sources_shape(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = payload["sources"]
    assert isinstance(sources, list)
    if len(sources) != EXPECTED_SOURCES:
        raise RuRealWorldCorpusError(f"expected {EXPECTED_SOURCES} sources, got {len(sources)}")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            raise RuRealWorldCorpusError("each source entry must be an object")
        for key in ("id", "kind", "url", "themes"):
            if key not in entry:
                raise RuRealWorldCorpusError(f"source entry missing key {key!r}")
        source_id = entry["id"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise RuRealWorldCorpusError("source id must be a non-empty string")
        if source_id in by_id:
            raise RuRealWorldCorpusError(f"duplicate source id {source_id!r}")
        themes = entry["themes"]
        if (
            not isinstance(themes, list)
            or not themes
            or not all(isinstance(item, str) and item.strip() for item in themes)
        ):
            raise RuRealWorldCorpusError(f"source {source_id}: themes must be non-empty strings")
        for theme in themes:
            if len(theme) > 120:
                raise RuRealWorldCorpusError(
                    f"source {source_id}: theme {theme!r} is too long for a theme label"
                )
        for forbidden in ("story", "narrative", "message_body", "verbatim"):
            if forbidden in entry:
                raise RuRealWorldCorpusError(
                    f"source {source_id}: key {forbidden!r} must not be stored"
                )
        by_id[source_id] = entry
    return by_id


def _check_review_semantics(
    singles: list[Any],
    journeys: list[Any],
    by_source_id: dict[str, dict[str, Any]],
) -> None:
    valid_ids = set(by_source_id)
    for record in singles:
        assert isinstance(record, dict)
        for provenance_id in record["provenance_ids"]:
            if provenance_id not in valid_ids:
                raise RuRealWorldCorpusError(
                    f"{record['id']}: provenance id {provenance_id!r} does not resolve"
                )
        route = record["expected_route"]
        if route == "emergency" and record["topic"] not in EMERGENCY_SINGLE_TOPICS:
            raise RuRealWorldCorpusError(
                f"{record['id']}: emergency label outside reviewed withdrawal/self-harm topics"
            )
        if route == "clarify" and record["topic"] not in CLARIFY_SINGLE_TOPICS:
            raise RuRealWorldCorpusError(
                f"{record['id']}: clarify label outside reviewed boundary topics"
            )
        if route in ("ordinary", "clarify"):
            classification = classify_emergency(str(record["utterance"]))
            if classification.is_emergency:
                raise RuRealWorldCorpusError(
                    f"{record['id']}: {route} utterance triggers the emergency "
                    f"classifier ({classifiers(classification)})"
                )
    for record in journeys:
        assert isinstance(record, dict)
        for provenance_id in record["provenance_ids"]:
            if provenance_id not in valid_ids:
                raise RuRealWorldCorpusError(
                    f"{record['id']}: provenance id {provenance_id!r} does not resolve"
                )
        for turn in record["turns"]:
            assert isinstance(turn, dict)
            if turn["expected_route"] == "emergency" and record["journey"] not in (
                EMERGENCY_JOURNEYS
            ):
                raise RuRealWorldCorpusError(
                    f"{record['id']} turn {turn['turn']}: emergency label outside "
                    "reviewed escalation journeys"
                )
            if turn["expected_route"] in ("ordinary", "clarify"):
                classification = classify_emergency(str(turn["utterance"]))
                if classification.is_emergency:
                    raise RuRealWorldCorpusError(
                        f"{record['id']} turn {turn['turn']}: {turn['expected_route']} "
                        f"utterance triggers the emergency classifier "
                        f"({classifiers(classification)})"
                    )


def classifiers(classification: Any) -> str:
    """Render matched emergency categories for an error message."""
    categories = getattr(classification, "categories", ())
    return ",".join(getattr(item, "value", str(item)) for item in categories)


def _check_dedup(singles: list[Any], journeys: list[Any]) -> None:
    seen_ids: set[str] = set()
    for record in (*singles, *journeys):
        assert isinstance(record, dict)
        record_id = str(record["id"])
        if record_id in seen_ids:
            raise RuRealWorldCorpusError(f"duplicate id {record_id!r}")
        seen_ids.add(record_id)
    seen_utterances: dict[str, str] = {}
    single_norms: set[str] = set()
    for record in singles:
        assert isinstance(record, dict)
        norm = _norm_utterance(str(record["utterance"]))
        if norm in single_norms:
            raise RuRealWorldCorpusError(f"duplicate single-turn utterance in {record['id']}")
        single_norms.add(norm)
        seen_utterances[norm] = str(record["id"])
    for record in journeys:
        assert isinstance(record, dict)
        for turn in record["turns"]:
            assert isinstance(turn, dict)
            norm = _norm_utterance(str(turn["utterance"]))
            owner = f"{record['id']} turn {turn['turn']}"
            if norm in seen_utterances:
                raise RuRealWorldCorpusError(
                    f"duplicate utterance: {owner} repeats {seen_utterances[norm]}"
                )
            seen_utterances[norm] = owner


def _check_meaning_preservation_anchors(
    singles: list[Any], journeys: list[Any], sources_text: str
) -> None:
    by_id = {str(item["id"]): item for item in singles if isinstance(item, dict)}
    slang_ids = [f"RU-S-{number:03d}" for number in range(191, 201)]
    for slang_id in slang_ids:
        record = by_id.get(slang_id)
        if record is None:
            raise RuRealWorldCorpusError(f"missing slang fixture {slang_id}")
        tags = record.get("rubric_tags", [])
        assert isinstance(tags, list)
        if "slang" not in tags or "meaning_preservation" not in tags:
            raise RuRealWorldCorpusError(f"{slang_id}: slang fixtures must keep both tags")
        if record.get("expected_route") != "ordinary":
            raise RuRealWorldCorpusError(f"{slang_id}: slang fixtures must stay ordinary")
    journeys_by_id = {str(item["id"]): item for item in journeys if isinstance(item, dict)}
    required_journeys = (
        "RU-J-012",
        "RU-J-013",
        "RU-J-017",
        "RU-J-023",
        "RU-J-027",
        "RU-J-028",
        "RU-J-030",
    )
    for journey_id in required_journeys:
        if journey_id not in journeys_by_id:
            raise RuRealWorldCorpusError(f"missing journey fixture {journey_id}")
    ambiguous = journeys_by_id["RU-J-013"]
    assert isinstance(ambiguous, dict)
    turns = ambiguous["turns"]
    assert isinstance(turns, list)
    first = turns[0]
    assert isinstance(first, dict)
    if first["utterance"] != "Я сорвался." or first["expected_route"] != "clarify":
        raise RuRealWorldCorpusError("RU-J-013 must keep the bare ambiguous sorvalsya as clarify")
    new_session = journeys_by_id["RU-J-027"]
    assert isinstance(new_session, dict)
    new_turns = [str(item["utterance"]) for item in new_session["turns"]]
    if "/new" not in new_turns:
        raise RuRealWorldCorpusError("RU-J-027 must keep the /new session-reset turn")
    secret_med = journeys_by_id["RU-J-017"]
    assert isinstance(secret_med, dict)
    secret_routes = [str(item["expected_route"]) for item in secret_med["turns"]]
    if "clarify" not in secret_routes:
        raise RuRealWorldCorpusError("RU-J-017 must keep clarify turns for secret medication")
    for record in singles:
        assert isinstance(record, dict)
        if str(record["utterance"]).strip() in sources_text:
            raise RuRealWorldCorpusError(
                f"{record['id']}: utterance is copied verbatim from the sources file"
            )


def validate(root: Path | None = None) -> CorpusSummary:
    """Validate the whole v1 corpus and return its stable identity."""
    base = root if root is not None else find_repo_root()
    corpus_path = base / CORPUS_REL
    sources_path = base / SOURCES_REL
    if not corpus_path.exists():
        raise RuRealWorldCorpusError(f"missing corpus file {CORPUS_REL}")
    if not sources_path.exists():
        raise RuRealWorldCorpusError(f"missing sources file {SOURCES_REL}")
    manifest, singles, journeys = load_records(corpus_path)
    payload = load_sources(sources_path)
    _check_manifest(manifest)
    if len(singles) != EXPECTED_SINGLE_TURNS:
        raise RuRealWorldCorpusError(
            f"expected {EXPECTED_SINGLE_TURNS} single turns, got {len(singles)}"
        )
    if len(journeys) != EXPECTED_JOURNEYS:
        raise RuRealWorldCorpusError(f"expected {EXPECTED_JOURNEYS} journeys, got {len(journeys)}")
    for index, record in enumerate(singles):
        if not isinstance(record, dict):
            raise RuRealWorldCorpusError(f"single_turn index {index} must be an object")
        _check_single(record, index)
    for index, record in enumerate(journeys):
        if not isinstance(record, dict):
            raise RuRealWorldCorpusError(f"journey index {index} must be an object")
        _check_journey(record, index)
    by_source_id = _check_sources_shape(payload)
    journey_turns = sum(len(item["turns"]) for item in journeys if isinstance(item, dict))
    if journey_turns != EXPECTED_JOURNEY_TURNS:
        raise RuRealWorldCorpusError(
            f"expected {EXPECTED_JOURNEY_TURNS} journey turns, got {journey_turns}"
        )
    _check_dedup(singles, journeys)
    _check_review_semantics(singles, journeys, by_source_id)
    sources_text = _decode_utf8_strict(sources_path)
    _check_meaning_preservation_anchors(singles, journeys, sources_text)
    referenced = {item for record in (*singles, *journeys) for item in record["provenance_ids"]}
    unreferenced = sorted(set(by_source_id) - referenced)
    if unreferenced:
        raise RuRealWorldCorpusError(f"unreferenced source ids: {unreferenced}")
    return CorpusSummary(
        corpus_sha256=sha256_file(corpus_path),
        sources_sha256=sha256_file(sources_path),
        single_turn=len(singles),
        journeys=len(journeys),
        journey_turns=journey_turns,
        total_utterances=len(singles) + journey_turns,
    )


def build_version_payload(summary: CorpusSummary) -> dict[str, Any]:
    """Build the stable version record consumed by downstream issue #62."""
    return {
        "corpus_version": CORPUS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "files": {"corpus": CORPUS_REL, "sources": SOURCES_REL, "version": VERSION_REL},
        "sha256": {"corpus": summary.corpus_sha256, "sources": summary.sources_sha256},
        "counts": {
            "single_turn": summary.single_turn,
            "multi_turn_journeys": summary.journeys,
            "multi_turn_user_turns": summary.journey_turns,
            "total_user_utterances": summary.total_utterances,
        },
        "provenance_sources": EXPECTED_SOURCES,
        "seed_commits": {
            "corpus": "3070a929865f8374601625734b08e2447f7f4ed3",
            "provenance": "0fd52e2711ff7184eb7c90e30d5f5dddfecb97d6",
        },
        "review": {
            "emergency_classifier_false_positives": 0,
            "fixture_corrections": 0,
            "note": "Emergency/clarify labels reviewed against #21; slang and "
            "ambiguity fixtures reviewed against #46. No fixture rewrite was "
            "required: emergency labels stay on reviewed withdrawal/self-harm "
            "escalations and ordinary/clarify inputs stay off the deterministic "
            "emergency path.",
        },
    }


__all__ = [
    "CORPUS_REL",
    "CORPUS_VERSION",
    "EXPECTED_JOURNEYS",
    "EXPECTED_JOURNEY_TURNS",
    "EXPECTED_SINGLE_TURNS",
    "EXPECTED_TOTAL_UTTERANCES",
    "RuRealWorldCorpusError",
    "CorpusSummary",
    "SCHEMA_VERSION",
    "SOURCES_REL",
    "VERSION_REL",
    "build_version_payload",
    "find_repo_root",
    "load_records",
    "load_sources",
    "sha256_bytes",
    "sha256_file",
    "validate",
]
