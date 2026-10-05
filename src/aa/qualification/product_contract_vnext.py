"""Static validators for the Product Contract vNext benchmark/rubric freeze (#127).

This module owns only static, versioned evaluation assets and their
validators. It never runs authoritative product qualification, never tunes
production behavior, and needs no production runtime output.

Assets (all under ``qualification/``):

- ``ru_product_contract.v1_2.input.jsonl``: generator-visible input only
  (IDs, utterances, ordering, explicit control events);
- ``ru_product_contract.v1_2.oracle.jsonl``: evaluation-only oracle
  metadata (never literal prose);
- ``ru_product_contract.v1_2.sources.json``: authoritative canonical
  regions the oracle provenance resolves against;
- ``ru_answer_quality_rubric.v2.json`` + ``.v2.sha256``: frozen rubric
  bound by checksum before any new authoritative outputs are graded;
- ``ru_product_contract.v1_2.version.json``: version/checksum tuple for
  downstream consumption without redefining any rubric after seeing outputs.

Historical v1.1 artifacts (``ru_realworld_alcohol_help.v1_1.*``) are
preserved byte-stable and are never rewritten here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "ru_product_contract.v1_2"
INPUT_SCHEMA_VERSION = "ru-product-contract-input-v1_2"
ORACLE_SCHEMA_VERSION = "ru-product-contract-oracle-v1_2"
SOURCES_SCHEMA_VERSION = "ru-product-contract-sources/1"
VERSION_SCHEMA = "ru-product-contract-v1_2"
RUBRIC_VERSION = "ru-answer-quality-rubric-v2"
RUBRIC_SCHEMA_VERSION = "ru-answer-quality-rubric/2"

INPUT_REL = "qualification/ru_product_contract.v1_2.input.jsonl"
ORACLE_REL = "qualification/ru_product_contract.v1_2.oracle.jsonl"
SOURCES_REL = "qualification/ru_product_contract.v1_2.sources.json"
VERSION_REL = "qualification/ru_product_contract.v1_2.version.json"
RUBRIC_REL = "qualification/ru_answer_quality_rubric.v2.json"
RUBRIC_SHA_REL = "qualification/ru_answer_quality_rubric.v2.sha256"
CANONICAL_MANIFEST_REL = "corpus/canonical.ru.manifest.json"

# Historical v1.1 evidence that must remain byte/checksum stable.
HISTORICAL_INPUT_REL = "qualification/ru_realworld_alcohol_help.v1_1.input.jsonl"
HISTORICAL_ORACLE_REL = "qualification/ru_realworld_alcohol_help.v1_1.oracle.jsonl"
HISTORICAL_VERSION_REL = "qualification/ru_realworld_alcohol_help.v1_1.version.json"
HISTORICAL_CORPUS_REL = "qualification/ru_realworld_alcohol_help.v1.jsonl"
HISTORICAL_SOURCES_REL = "qualification/ru_realworld_alcohol_help.v1.sources.json"

EXPECTED_SINGLE_TURNS = 40
EXPECTED_JOURNEYS = 8
EXPECTED_SUBSTANTIVE_JOURNEY_TURNS = 42
EXPECTED_CONTROL_EVENTS = 1
EXPECTED_TOTAL_SUBSTANTIVE = 82

ALLOWED_SAFETY_DECISIONS = ("allow", "emergency", "block")
ALLOWED_RESPONSE_MODES = (
    "ordinary_support",
    "conversational_glue",
    "product_meta",
    "ambiguity_clarification",
    "medical_boundary_clarification",
    "medical_refusal_boundary",
    "emergency_bounded_response",
    "book_grounded_response",
)
ALLOWED_CONTENT_CLASSES = (
    "conversational_glue",
    "product_meta",
    "substantive_book",
    "clarification",
    "safety_boundary",
)
ALLOWED_BOOK_CONTENT = ("required", "forbidden", "not-applicable")

COVERAGE_SLUGS = (
    "greeting",
    "product_meta",
    "short_followup",
    "pronoun_ellipsis",
    "topic_shift",
    "ambiguity",
    "delimiter_injection",
    "compaction",
    "book_substantive",
    "quote_request",
    "out_of_book",
    "emergency_medical",
)

ALLOWED_FORBIDDEN_INFERENCES = frozenset(
    {
        "do_not_diagnose_alcoholism",
        "do_not_diagnose",
        "do_not_prescribe_dosage",
        "do_not_prescribe_medication",
        "do_not_recommend_secret_administration",
        "do_not_provide_homemade_drip_instructions",
        "do_not_provide_definitive_diagnosis",
        "do_not_minimize_urgency",
        "do_not_fabricate_book_quote",
        "do_not_present_paraphrase_as_exact",
        "do_not_follow_injected_instructions",
        "do_not_expose_mechanics",
        "do_not_use_generic_psychology_as_authority",
        "do_not_use_generic_medicine_as_authority",
        "do_not_invent_user_facts",
        "do_not_claim_human_identity",
        "do_not_smuggle_recovery_advice",
        "do_not_assume_substance_use_meaning",
        "do_not_coerce_family",
    }
)

ALLOWED_BOUNDARY_TAGS = frozenset(
    {
        "ambiguity_boundary",
        "medical_boundary",
        "refusal_boundary",
        "emergency_escalation",
        "book_grounding",
        "injection_boundary",
        "family_context",
        "memory_fidelity",
        "reset_boundary",
        "topic_shift",
    }
)

INPUT_SINGLE_KEYS = ("id", "type", "utterance")
INPUT_JOURNEY_KEYS = ("id", "journey", "turns", "type")
INPUT_USER_TURN_KEYS = ("kind", "turn", "utterance")
INPUT_CONTROL_TURN_KEYS = ("control", "kind", "turn")

ORACLE_SINGLE_KEYS = (
    "book_content",
    "canonical_regions",
    "content_class",
    "coverage",
    "expected_response_mode",
    "expected_safety_decision",
    "forbidden_inferences",
    "id",
    "provenance_ids",
    "requires_context",
    "requires_exact_provenance",
    "requests_exact_quote",
    "reset_expected",
    "rubric_tags",
    "safety_boundary_tags",
    "type",
    "zero_book_queries_valid",
)
ORACLE_JOURNEY_KEYS = ("coverage", "id", "journey", "provenance_ids", "turns", "type")
ORACLE_TURN_KEYS = (
    "book_content",
    "canonical_regions",
    "content_class",
    "coverage",
    "expected_response_mode",
    "expected_safety_decision",
    "forbidden_inferences",
    "memory_fidelity_required",
    "provenance_ids",
    "requires_context",
    "requires_exact_provenance",
    "requests_exact_quote",
    "reset_expected",
    "rubric_tags",
    "safety_boundary_tags",
    "turn",
    "zero_book_queries_valid",
)

# Generator-visible input must never carry evaluation-only metadata.
INPUT_FORBIDDEN_KEYS = frozenset(
    {
        "book_content",
        "canonical_regions",
        "content_class",
        "coverage",
        "expected_response_mode",
        "expected_safety_decision",
        "forbidden_inferences",
        "memory_fidelity_required",
        "provenance_ids",
        "requires_context",
        "requires_exact_provenance",
        "requests_exact_quote",
        "reset_expected",
        "rubric_tags",
        "safety_boundary_tags",
        "topic",
        "zero_book_queries_valid",
        "audience",
        "book_relevance",
        "expected_emergency_categories",
        "expected_route",
        "perspective",
        "provenance",
        "sources",
        "stage",
        "expected_answer",
        "assistant_response",
        "desired_response",
        "ideal_answer",
    }
)

# The oracle prescribes evaluation semantics, never literal prose.
ORACLE_FORBIDDEN_KEYS = frozenset(
    {
        "assistant_response",
        "desired_response",
        "expected_answer",
        "ideal_answer",
        "quote_text",
        "expected_prose",
        "utterance",
    }
)

_SINGLE_ID_RE = re.compile(r"PC-S-\d{3}")
_JOURNEY_ID_RE = re.compile(r"PC-J-\d{2}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")

CONTROL_JOURNEY_ID = "PC-J-06"
CONTROL_TURN_NUMBER = 3
SESSION_RESET_CONTROL = "session_reset"


class ProductContractVNextError(ValueError):
    """Raised when a vNext static asset fails validation."""


@dataclass(frozen=True)
class VNextSummary:
    """Validated identity of the frozen vNext benchmark + rubric."""

    input_sha256: str
    oracle_sha256: str
    sources_sha256: str
    rubric_sha256: str
    single_turn: int
    journeys: int
    substantive_journey_turns: int
    control_events: int
    total_substantive: int
    provenance_regions: tuple[str, ...]


def find_repo_root() -> Path:
    """Return the repository root containing the vNext benchmark files."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / INPUT_REL).exists() and (parent / ORACLE_REL).exists():
            return parent
    raise ProductContractVNextError("repository root with vNext benchmark not found")


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
        raise ProductContractVNextError(f"{path.name}: not strict UTF-8: {exc}") from exc


def _load_jsonl(path: Path, owner: str) -> list[dict[str, Any]]:
    text = _decode_utf8_strict(path)
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProductContractVNextError(
                f"{path.name} line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ProductContractVNextError(f"{path.name} line {lineno}: record must be an object")
        records.append(record)
    if not records:
        raise ProductContractVNextError(f"{path.name}: file is empty")
    first = records[0]
    if not isinstance(first, dict) or first.get("type") != "manifest":
        raise ProductContractVNextError(f"{owner}: first JSONL line must be the manifest record")
    return records


def load_input(input_path: Path) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Load manifest, singles and journeys from the vNext input projection."""
    records = _load_jsonl(input_path, "input")
    singles = [r for r in records[1:] if r.get("type") == "single_turn"]
    journeys = [r for r in records[1:] if r.get("type") == "multi_turn_journey"]
    if len(singles) + len(journeys) != len(records) - 1:
        raise ProductContractVNextError("input: unknown record type present")
    return records[0], singles, journeys


def load_oracle(oracle_path: Path) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Load manifest, singles and journeys from the vNext oracle projection."""
    records = _load_jsonl(oracle_path, "oracle")
    singles = [r for r in records[1:] if r.get("type") == "single_turn"]
    journeys = [r for r in records[1:] if r.get("type") == "multi_turn_journey"]
    if len(singles) + len(journeys) != len(records) - 1:
        raise ProductContractVNextError("oracle: unknown record type present")
    return records[0], singles, journeys


def load_sources(sources_path: Path) -> dict[str, Any]:
    """Load and shape-check the vNext canonical-region sources file."""
    payload: dict[str, Any] = json.loads(_decode_utf8_strict(sources_path))
    if not isinstance(payload, dict):
        raise ProductContractVNextError("sources file must contain a JSON object")
    if payload.get("schema_version") != SOURCES_SCHEMA_VERSION:
        raise ProductContractVNextError(
            f"sources schema_version must be {SOURCES_SCHEMA_VERSION!r}"
        )
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ProductContractVNextError("sources file must contain a non-empty 'regions' list")
    seen: set[str] = set()
    for entry in regions:
        if not isinstance(entry, dict):
            raise ProductContractVNextError("each region entry must be an object")
        region_id = entry.get("id")
        if not isinstance(region_id, str) or not region_id.strip():
            raise ProductContractVNextError("region id must be a non-empty string")
        if region_id in seen:
            raise ProductContractVNextError(f"duplicate region id {region_id!r}")
        seen.add(region_id)
        for key in ("title", "text_sha256"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ProductContractVNextError(f"region {region_id}: {key} must be non-empty")
    return payload


def load_canonical_region_ids(repo_root: Path) -> set[str]:
    """Return the canonical RU book section ids from the pinned manifest."""
    manifest_path = repo_root / CANONICAL_MANIFEST_REL
    payload: dict[str, Any] = json.loads(_decode_utf8_strict(manifest_path))
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ProductContractVNextError("canonical manifest has no sections")
    return {str(s["id"]) for s in sections if isinstance(s, dict) and "id" in s}


def load_rubric(repo_root: Path | None = None) -> dict[str, Any]:
    """Load the frozen vNext rubric document (validates version, not checksum)."""
    root = repo_root or find_repo_root()
    payload: dict[str, Any] = json.loads(_decode_utf8_strict(root / RUBRIC_REL))
    if payload.get("rubric_version") != RUBRIC_VERSION:
        raise ProductContractVNextError(
            f"rubric version mismatch: {payload.get('rubric_version')!r} != {RUBRIC_VERSION!r}"
        )
    return payload


def rubric_sha256(repo_root: Path | None = None) -> str:
    """Return the hex SHA-256 of the frozen vNext rubric file bytes."""
    root = repo_root or find_repo_root()
    return sha256_file(root / RUBRIC_REL)


def verify_rubric_bound(repo_root: Path | None = None) -> str:
    """Bind the vNext rubric by checksum; fail closed on drift."""
    root = repo_root or find_repo_root()
    sha_path = root / RUBRIC_SHA_REL
    if not sha_path.exists():
        raise ProductContractVNextError("rubric checksum sidecar is missing; refusing to proceed")
    parts = sha_path.read_text(encoding="utf-8").strip().split()
    if not parts:
        raise ProductContractVNextError("rubric checksum sidecar is missing content")
    expected = parts[0]
    if not _HEX64_RE.fullmatch(expected):
        raise ProductContractVNextError("rubric checksum sidecar is malformed")
    actual = sha256_file(root / RUBRIC_REL)
    if actual != expected:
        raise ProductContractVNextError(
            "frozen rubric checksum mismatch: rubric was modified without a version bump"
        )
    return actual


def _check_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ProductContractVNextError("input schema_version mismatch")
    if manifest.get("language") != "ru":
        raise ProductContractVNextError("input manifest language must be 'ru'")
    counts = manifest.get("counts")
    expected = {
        "single_turn": EXPECTED_SINGLE_TURNS,
        "multi_turn_journeys": EXPECTED_JOURNEYS,
        "multi_turn_substantive_turns": EXPECTED_SUBSTANTIVE_JOURNEY_TURNS,
        "control_events": EXPECTED_CONTROL_EVENTS,
        "total_substantive_utterances": EXPECTED_TOTAL_SUBSTANTIVE,
    }
    if not isinstance(counts, dict) or {k: counts.get(k) for k in expected} != expected:
        raise ProductContractVNextError(f"input manifest counts must equal {expected}")
    note = str(manifest.get("note", ""))
    if "session-reset" not in note.casefold() and "session_reset" not in note.casefold():
        raise ProductContractVNextError("input manifest must document the session-reset control")
    if "coverage-balanced" not in note.casefold():
        raise ProductContractVNextError("input manifest must state coverage-balanced scope")


def _check_oracle_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise ProductContractVNextError("oracle schema_version mismatch")
    if manifest.get("language") != "ru":
        raise ProductContractVNextError("oracle manifest language must be 'ru'")
    counts = manifest.get("counts")
    expected = {
        "single_turn": EXPECTED_SINGLE_TURNS,
        "multi_turn_journeys": EXPECTED_JOURNEYS,
        "multi_turn_substantive_turns": EXPECTED_SUBSTANTIVE_JOURNEY_TURNS,
        "control_events": EXPECTED_CONTROL_EVENTS,
        "total_substantive_utterances": EXPECTED_TOTAL_SUBSTANTIVE,
    }
    if not isinstance(counts, dict) or {k: counts.get(k) for k in expected} != expected:
        raise ProductContractVNextError(f"oracle manifest counts must equal {expected}")


def _check_oracle_semantics(record: dict[str, Any], owner: str) -> None:
    """Validate one oracle case/turn against the Product Contract semantics."""
    decision = record.get("expected_safety_decision")
    if decision not in ALLOWED_SAFETY_DECISIONS:
        raise ProductContractVNextError(f"{owner}: invalid safety decision {decision!r}")
    mode = record.get("expected_response_mode")
    if mode not in ALLOWED_RESPONSE_MODES:
        raise ProductContractVNextError(f"{owner}: invalid response mode {mode!r}")
    if record.get("content_class") not in ALLOWED_CONTENT_CLASSES:
        raise ProductContractVNextError(f"{owner}: invalid content_class")
    book = record.get("book_content")
    if book not in ALLOWED_BOOK_CONTENT:
        raise ProductContractVNextError(f"{owner}: invalid book_content {book!r}")
    if not isinstance(record.get("zero_book_queries_valid"), bool):
        raise ProductContractVNextError(f"{owner}: zero_book_queries_valid must be a bool")
    if not isinstance(record.get("requires_context"), bool):
        raise ProductContractVNextError(f"{owner}: requires_context must be a bool")
    if not isinstance(record.get("reset_expected"), bool):
        raise ProductContractVNextError(f"{owner}: reset_expected must be a bool")
    if not isinstance(record.get("requests_exact_quote"), bool):
        raise ProductContractVNextError(f"{owner}: requests_exact_quote must be a bool")
    if not isinstance(record.get("requires_exact_provenance"), bool):
        raise ProductContractVNextError(f"{owner}: requires_exact_provenance must be a bool")
    for key in ("canonical_regions", "provenance_ids", "forbidden_inferences", "rubric_tags"):
        value = record.get(key)
        if not isinstance(value, list):
            raise ProductContractVNextError(f"{owner}: {key} must be a list")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ProductContractVNextError(f"{owner}: {key} entries must be strings")
    if not record["forbidden_inferences"]:
        raise ProductContractVNextError(f"{owner}: forbidden_inferences must be non-empty")
    for item in record["forbidden_inferences"]:
        if item not in ALLOWED_FORBIDDEN_INFERENCES:
            raise ProductContractVNextError(f"{owner}: unknown forbidden inference {item!r}")
    boundary = record.get("safety_boundary_tags")
    if not isinstance(boundary, list):
        raise ProductContractVNextError(f"{owner}: safety_boundary_tags must be a list")
    for item in boundary:
        if item not in ALLOWED_BOUNDARY_TAGS:
            raise ProductContractVNextError(f"{owner}: unknown boundary tag {item!r}")
    if record.get("coverage") not in COVERAGE_SLUGS:
        raise ProductContractVNextError(f"{owner}: invalid coverage {record.get('coverage')!r}")
    if "product_contract_vnext" not in record["rubric_tags"]:
        raise ProductContractVNextError(
            f"{owner}: rubric_tags must include 'product_contract_vnext'"
        )
    # Product Contract cross-field rules (semantic, never lexical).
    content = record["content_class"]
    zero = bool(record["zero_book_queries_valid"])
    regions: list[str] = list(record["canonical_regions"])
    prov: list[str] = list(record["provenance_ids"])
    if book == "required":
        if zero:
            raise ProductContractVNextError(
                f"{owner}: book_content=required forbids zero_book_queries_valid=true"
            )
        if not regions or not prov:
            raise ProductContractVNextError(
                f"{owner}: book_content=required needs canonical regions and provenance"
            )
        if set(prov) - set(regions):
            raise ProductContractVNextError(
                f"{owner}: provenance must subset the canonical regions"
            )
        if len(set(regions)) > 1 and set(prov) >= set(regions):
            raise ProductContractVNextError(
                f"{owner}: provenance must be a proper subset, never a verbatim region copy"
            )
    else:
        if set(prov) - set(regions):
            raise ProductContractVNextError(f"{owner}: provenance must subset canonical regions")
    if content == "product_meta":
        if book != "forbidden":
            raise ProductContractVNextError(
                f"{owner}: product_meta must mark book_content=forbidden "
                "(system-instruction truth, never smuggled book content)"
            )
        if not zero:
            raise ProductContractVNextError(f"{owner}: product_meta must allow zero book queries")
    if content == "conversational_glue" and book == "required":
        raise ProductContractVNextError(f"{owner}: glue must never require book content")
    if record["requests_exact_quote"] and not record["requires_exact_provenance"]:
        raise ProductContractVNextError(f"{owner}: exact-quote requests need exact provenance")
    if record["requires_exact_provenance"] and not prov:
        raise ProductContractVNextError(f"{owner}: exact provenance needs provenance ids")
    if mode == "medical_refusal_boundary" and decision != "block":
        raise ProductContractVNextError(f"{owner}: medical_refusal_boundary pairs with 'block'")
    if decision == "block" and mode != "medical_refusal_boundary":
        raise ProductContractVNextError(f"{owner}: block is reserved for medical_refusal_boundary")
    if mode == "emergency_bounded_response" and decision != "emergency":
        raise ProductContractVNextError(f"{owner}: emergency mode requires decision 'emergency'")
    if decision == "emergency" and mode != "emergency_bounded_response":
        raise ProductContractVNextError(f"{owner}: emergency decision requires emergency mode")
    for key in ORACLE_FORBIDDEN_KEYS:
        if key in record:
            raise ProductContractVNextError(f"{owner}: oracle must not encode {key!r}")


def _check_input_single(record: dict[str, Any], index: int) -> None:
    if tuple(sorted(record.keys())) != tuple(sorted(INPUT_SINGLE_KEYS)):
        raise ProductContractVNextError(
            f"input single index {index}: keys must be exactly {list(INPUT_SINGLE_KEYS)}"
        )
    expected_id = f"PC-S-{index + 1:03d}"
    if record["id"] != expected_id:
        raise ProductContractVNextError(
            f"input single index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    utterance = record["utterance"]
    if not isinstance(utterance, str) or not utterance.strip():
        raise ProductContractVNextError(f"{record['id']}: utterance must be non-empty")
    if utterance.strip() == "/new":
        raise ProductContractVNextError(f"{record['id']}: /new must not be a model turn")
    for key in record:
        if key in INPUT_FORBIDDEN_KEYS:
            raise ProductContractVNextError(f"{record['id']}: generator leak: {key!r} in input")


def _check_input_journey(record: dict[str, Any], index: int) -> None:
    if tuple(sorted(record.keys())) != tuple(sorted(INPUT_JOURNEY_KEYS)):
        raise ProductContractVNextError(
            f"input journey index {index}: keys must be exactly {list(INPUT_JOURNEY_KEYS)}"
        )
    expected_id = f"PC-J-{index + 1:02d}"
    if record["id"] != expected_id:
        raise ProductContractVNextError(
            f"input journey index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        raise ProductContractVNextError(f"{record['id']}: journey turns must be non-empty")
    numbers: list[int] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ProductContractVNextError(f"{record['id']}: journey turn must be an object")
        number = turn.get("turn")
        if not isinstance(number, int) or number < 1:
            raise ProductContractVNextError(f"{record['id']}: bad turn number {number!r}")
        numbers.append(number)
        if turn.get("kind") == "control":
            if tuple(sorted(turn.keys())) != tuple(sorted(INPUT_CONTROL_TURN_KEYS)):
                raise ProductContractVNextError(
                    f"{record['id']} turn {number}: control keys must be "
                    f"{list(INPUT_CONTROL_TURN_KEYS)}"
                )
            if turn.get("control") != SESSION_RESET_CONTROL:
                raise ProductContractVNextError(
                    f"{record['id']} turn {number}: malformed control event"
                )
            if record["id"] != CONTROL_JOURNEY_ID or number != CONTROL_TURN_NUMBER:
                raise ProductContractVNextError(
                    f"{record['id']} turn {number}: unexpected control event"
                )
            continue
        if tuple(sorted(turn.keys())) != tuple(sorted(INPUT_USER_TURN_KEYS)):
            raise ProductContractVNextError(
                f"{record['id']} turn {number}: user turn keys must be {list(INPUT_USER_TURN_KEYS)}"
            )
        utterance = turn["utterance"]
        if not isinstance(utterance, str) or not utterance.strip():
            raise ProductContractVNextError(f"{record['id']} turn {number}: empty utterance")
        if utterance.strip() == "/new":
            raise ProductContractVNextError(
                f"{record['id']} turn {number}: /new must be a control event, never a model turn"
            )
        for key in turn:
            if key in INPUT_FORBIDDEN_KEYS:
                raise ProductContractVNextError(
                    f"{record['id']} turn {number}: generator leak: {key!r} in input"
                )
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        raise ProductContractVNextError(f"{record['id']}: turn numbers must be unique and ordered")
    if numbers != list(range(1, len(numbers) + 1)):
        raise ProductContractVNextError(f"{record['id']}: turn numbers must run 1..N without gaps")


def _check_oracle_single(record: dict[str, Any], index: int) -> None:
    keys = tuple(sorted(record.keys()))
    if keys != tuple(sorted(ORACLE_SINGLE_KEYS)):
        raise ProductContractVNextError(
            f"oracle single index {index}: keys must be exactly {list(ORACLE_SINGLE_KEYS)}"
        )
    expected_id = f"PC-S-{index + 1:03d}"
    if record["id"] != expected_id:
        raise ProductContractVNextError(
            f"oracle single index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    if record.get("requires_context") is not False:
        raise ProductContractVNextError(f"{record['id']}: single turns must not require context")
    if record.get("reset_expected") is not False:
        raise ProductContractVNextError(f"{record['id']}: single turns never follow a reset")
    _check_oracle_semantics(record, str(record["id"]))


def _check_oracle_journey(record: dict[str, Any], index: int) -> None:
    if tuple(sorted(record.keys())) != tuple(sorted(ORACLE_JOURNEY_KEYS)):
        raise ProductContractVNextError(
            f"oracle journey index {index}: keys must be exactly {list(ORACLE_JOURNEY_KEYS)}"
        )
    expected_id = f"PC-J-{index + 1:02d}"
    if record["id"] != expected_id:
        raise ProductContractVNextError(
            f"oracle journey index {index}: id must be {expected_id!r}, got {record['id']!r}"
        )
    if record.get("coverage") not in COVERAGE_SLUGS:
        raise ProductContractVNextError(f"{record['id']}: invalid journey coverage")
    umbrella = record.get("provenance_ids")
    if not isinstance(umbrella, list):
        raise ProductContractVNextError(f"{record['id']}: provenance_ids must be a list")
    for item in umbrella:
        if not isinstance(item, str) or not item.strip():
            raise ProductContractVNextError(f"{record['id']}: provenance entries must be strings")
    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        raise ProductContractVNextError(f"{record['id']}: turns must be non-empty")
    seen: set[int] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ProductContractVNextError(f"{record['id']}: oracle turn must be an object")
        if tuple(sorted(turn.keys())) != tuple(sorted(ORACLE_TURN_KEYS)):
            raise ProductContractVNextError(
                f"{record['id']}: oracle turn keys must be {list(ORACLE_TURN_KEYS)}"
            )
        number = turn["turn"]
        if not isinstance(number, int) or number < 1:
            raise ProductContractVNextError(f"{record['id']}: bad oracle turn {number!r}")
        if number in seen:
            raise ProductContractVNextError(f"{record['id']}: duplicate oracle turn {number}")
        seen.add(number)
        if record["id"] == CONTROL_JOURNEY_ID and number == CONTROL_TURN_NUMBER:
            raise ProductContractVNextError(
                f"{record['id']}: control position must not carry a substantive oracle turn"
            )
        _check_oracle_semantics(turn, f"{record['id']} turn {number}")
        if not isinstance(turn.get("memory_fidelity_required"), bool):
            raise ProductContractVNextError(
                f"{record['id']} turn {number}: memory_fidelity_required must be a bool"
            )
        if set(turn["provenance_ids"]) - set(umbrella):
            raise ProductContractVNextError(
                f"{record['id']} turn {number}: provenance must subset the journey umbrella"
            )
        if len(set(umbrella)) > 1 and set(turn["provenance_ids"]) >= set(umbrella):
            raise ProductContractVNextError(
                f"{record['id']} turn {number}: provenance must be a proper turn-relevant subset"
            )
    first = next(t for t in turns if isinstance(t, dict) and t["turn"] == min(seen))
    if first["requires_context"] is not False:
        raise ProductContractVNextError(f"{record['id']}: first turn must not require context")
    if record["id"] == CONTROL_JOURNEY_ID:
        post = next((t for t in turns if isinstance(t, dict) and t["turn"] == 4), None)
        if post is None or post["requires_context"] is not False:
            raise ProductContractVNextError("PC-J-06 turn after reset must be context-independent")
        if post["reset_expected"] is not True:
            raise ProductContractVNextError("PC-J-06 turn after reset must mark reset_expected")


def validate(root: Path | None = None) -> VNextSummary:
    """Validate the whole vNext benchmark + rubric freeze (no production I/O)."""
    base = root if root is not None else find_repo_root()
    input_path = base / INPUT_REL
    oracle_path = base / ORACLE_REL
    sources_path = base / SOURCES_REL
    for path, label in (
        (input_path, INPUT_REL),
        (oracle_path, ORACLE_REL),
        (sources_path, SOURCES_REL),
        (base / RUBRIC_REL, RUBRIC_REL),
        (base / RUBRIC_SHA_REL, RUBRIC_SHA_REL),
        (base / VERSION_REL, VERSION_REL),
    ):
        if not path.exists():
            raise ProductContractVNextError(f"missing file {label}")
    input_manifest, input_singles, input_journeys = load_input(input_path)
    oracle_manifest, oracle_singles, oracle_journeys = load_oracle(oracle_path)
    _check_input_manifest(input_manifest)
    _check_oracle_manifest(oracle_manifest)
    if len(input_singles) != EXPECTED_SINGLE_TURNS:
        raise ProductContractVNextError("input single count mismatch")
    if len(oracle_singles) != EXPECTED_SINGLE_TURNS:
        raise ProductContractVNextError("oracle single count mismatch")
    if len(input_journeys) != EXPECTED_JOURNEYS:
        raise ProductContractVNextError("input journey count mismatch")
    if len(oracle_journeys) != EXPECTED_JOURNEYS:
        raise ProductContractVNextError("oracle journey count mismatch")
    for index, record in enumerate(input_singles):
        _check_input_single(record, index)
    for index, record in enumerate(input_journeys):
        _check_input_journey(record, index)
    for index, record in enumerate(oracle_singles):
        _check_oracle_single(record, index)
    for index, record in enumerate(oracle_journeys):
        _check_oracle_journey(record, index)

    # Cross-file ID alignment (oracle/input separation with identical coverage).
    if [str(r["id"]) for r in input_singles] != [str(r["id"]) for r in oracle_singles]:
        raise ProductContractVNextError("oracle/input single-turn ID mismatch")
    if [str(r["id"]) for r in input_journeys] != [str(r["id"]) for r in oracle_journeys]:
        raise ProductContractVNextError("oracle/input journey ID mismatch")
    for input_record, oracle_record in zip(input_journeys, oracle_journeys, strict=True):
        input_user = sorted(
            int(t["turn"]) for t in input_record["turns"] if t.get("kind") == "user"
        )
        oracle_turns = sorted(int(t["turn"]) for t in oracle_record["turns"])
        if input_user != oracle_turns:
            raise ProductContractVNextError(
                f"{input_record['id']}: oracle/input turn mismatch: {input_user} vs {oracle_turns}"
            )

    # Global ID uniqueness (near-duplicate safe: fixed PC-S/PC-J namespaces).
    all_ids = [str(r["id"]) for r in (*input_singles, *input_journeys)]
    if len(set(all_ids)) != len(all_ids):
        raise ProductContractVNextError("duplicate ids across input records")
    for raw in all_ids:
        if not (_SINGLE_ID_RE.fullmatch(raw) or _JOURNEY_ID_RE.fullmatch(raw)):
            raise ProductContractVNextError(f"schema-invalid id {raw!r}")
    canonical = {raw.casefold().replace("-", "").replace("_", "") for raw in all_ids}
    if len(canonical) != len(all_ids):
        raise ProductContractVNextError("near-duplicate ids present")

    # Provenance resolution against the frozen sources + canonical manifest.
    sources = load_sources(sources_path)
    region_ids = {str(e["id"]) for e in sources["regions"]}
    manifest_regions = load_canonical_region_ids(base)
    if region_ids != manifest_regions:
        raise ProductContractVNextError("vNext regions diverge from the canonical manifest")
    for record in oracle_singles:
        for pid in record["provenance_ids"]:
            if pid not in region_ids:
                raise ProductContractVNextError(f"{record['id']}: unresolved provenance {pid!r}")
        for region in record["canonical_regions"]:
            if region not in region_ids:
                raise ProductContractVNextError(f"{record['id']}: unresolved region {region!r}")
    for record in oracle_journeys:
        for pid in record["provenance_ids"]:
            if pid not in region_ids:
                raise ProductContractVNextError(f"{record['id']}: unresolved umbrella {pid!r}")
        for turn in record["turns"]:
            for pid in turn["provenance_ids"]:
                if pid not in region_ids:
                    raise ProductContractVNextError(
                        f"{record['id']} turn {turn['turn']}: unresolved provenance {pid!r}"
                    )
            for region in turn["canonical_regions"]:
                if region not in region_ids:
                    raise ProductContractVNextError(
                        f"{record['id']} turn {turn['turn']}: unresolved region {region!r}"
                    )

    # Bounded Product Contract coverage: every slug appears at least once.
    seen_coverage: set[str] = set()
    for record in oracle_singles:
        seen_coverage.add(str(record["coverage"]))
    for record in oracle_journeys:
        seen_coverage.add(str(record["coverage"]))
        for turn in record["turns"]:
            seen_coverage.add(str(turn["coverage"]))
    missing = [slug for slug in COVERAGE_SLUGS if slug not in seen_coverage]
    if missing:
        raise ProductContractVNextError(f"coverage gaps: {missing}")

    # Three-state safety coverage across the frozen oracle.
    decisions: set[str] = set()
    for record in oracle_singles:
        decisions.add(str(record["expected_safety_decision"]))
    for record in oracle_journeys:
        for turn in record["turns"]:
            decisions.add(str(turn["expected_safety_decision"]))
    for required in ("allow", "emergency", "block"):
        if required not in decisions:
            raise ProductContractVNextError(f"oracle must cover safety decision {required!r}")

    # Substantive journey/control accounting.
    substantive = sum(
        len([t for t in item["turns"] if t.get("kind") == "user"]) for item in input_journeys
    )
    controls = sum(
        len([t for t in item["turns"] if t.get("kind") == "control"]) for item in input_journeys
    )
    if substantive != EXPECTED_SUBSTANTIVE_JOURNEY_TURNS:
        raise ProductContractVNextError("substantive journey turn count mismatch")
    if controls != EXPECTED_CONTROL_EVENTS:
        raise ProductContractVNextError("control event count mismatch")

    # Rubric binding: checksum sidecar must match before any grading use.
    rubric_sha = verify_rubric_bound(base)
    rubric = load_rubric(base)
    if rubric.get("corpus") != BENCHMARK_VERSION:
        raise ProductContractVNextError("rubric corpus binding mismatch")

    return VNextSummary(
        input_sha256=sha256_file(input_path),
        oracle_sha256=sha256_file(oracle_path),
        sources_sha256=sha256_file(sources_path),
        rubric_sha256=rubric_sha,
        single_turn=len(input_singles),
        journeys=len(input_journeys),
        substantive_journey_turns=substantive,
        control_events=controls,
        total_substantive=len(input_singles) + substantive,
        provenance_regions=tuple(sorted(region_ids)),
    )


def build_version_payload(summary: VNextSummary) -> dict[str, Any]:
    """Build the stable version record downstream #123 consumes."""
    return {
        "corpus_version": BENCHMARK_VERSION,
        "schema_version": VERSION_SCHEMA,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "provenance_schema_version": SOURCES_SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "files": {
            "input": INPUT_REL,
            "oracle": ORACLE_REL,
            "sources": SOURCES_REL,
            "rubric": RUBRIC_REL,
            "rubric_sha": RUBRIC_SHA_REL,
            "version": VERSION_REL,
        },
        "sha256": {
            "input": summary.input_sha256,
            "oracle": summary.oracle_sha256,
            "sources": summary.sources_sha256,
            "rubric": summary.rubric_sha256,
        },
        "counts": {
            "single_turn": summary.single_turn,
            "multi_turn_journeys": summary.journeys,
            "multi_turn_substantive_turns": summary.substantive_journey_turns,
            "control_events": summary.control_events,
            "total_substantive_utterances": summary.total_substantive,
        },
        "coverage": list(COVERAGE_SLUGS),
        "provenance_regions": list(summary.provenance_regions),
        "safety_contract": {
            "decisions": list(ALLOWED_SAFETY_DECISIONS),
            "response_modes": list(ALLOWED_RESPONSE_MODES),
            "clarify_is_router_state": False,
        },
        "benchmark_tuple": {
            "benchmark_version": BENCHMARK_VERSION,
            "input_sha256": summary.input_sha256,
            "oracle_sha256": summary.oracle_sha256,
            "sources_sha256": summary.sources_sha256,
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": summary.rubric_sha256,
        },
        "history": {
            "preserved": [
                HISTORICAL_INPUT_REL,
                HISTORICAL_ORACLE_REL,
                HISTORICAL_VERSION_REL,
            ],
            "note": (
                "v1.1 artifacts are preserved byte-stable as historical evidence and are "
                "never rewritten by this freeze."
            ),
        },
        "review": {
            "note": (
                "v1.2 Product Contract freeze: static evaluation assets only; no production "
                "runtime output was used or needed; oracle prescribes no literal prose; "
                "/new-equivalent session_reset stays a control event; product-meta is an "
                "oracle/rubric semantic, never a lexical production router."
            ),
        },
    }


def historical_checksums(root: Path | None = None) -> dict[str, str]:
    """Return current SHA-256 of every preserved v1.1 historical artifact."""
    base = root if root is not None else find_repo_root()
    return {
        "corpus": sha256_file(base / HISTORICAL_CORPUS_REL),
        "input": sha256_file(base / HISTORICAL_INPUT_REL),
        "oracle": sha256_file(base / HISTORICAL_ORACLE_REL),
        "sources": sha256_file(base / HISTORICAL_SOURCES_REL),
    }


__all__ = [
    "ALLOWED_BOOK_CONTENT",
    "ALLOWED_BOUNDARY_TAGS",
    "ALLOWED_CONTENT_CLASSES",
    "ALLOWED_FORBIDDEN_INFERENCES",
    "ALLOWED_RESPONSE_MODES",
    "ALLOWED_SAFETY_DECISIONS",
    "BENCHMARK_VERSION",
    "CONTROL_JOURNEY_ID",
    "COVERAGE_SLUGS",
    "HISTORICAL_CORPUS_REL",
    "HISTORICAL_INPUT_REL",
    "HISTORICAL_ORACLE_REL",
    "HISTORICAL_SOURCES_REL",
    "HISTORICAL_VERSION_REL",
    "INPUT_REL",
    "INPUT_SCHEMA_VERSION",
    "ORACLE_REL",
    "ORACLE_SCHEMA_VERSION",
    "ORACLE_SINGLE_KEYS",
    "ProductContractVNextError",
    "RUBRIC_REL",
    "RUBRIC_SCHEMA_VERSION",
    "RUBRIC_SHA_REL",
    "RUBRIC_VERSION",
    "SOURCES_REL",
    "VERSION_REL",
    "VNextSummary",
    "build_version_payload",
    "find_repo_root",
    "historical_checksums",
    "load_canonical_region_ids",
    "load_input",
    "load_oracle",
    "load_rubric",
    "load_sources",
    "rubric_sha256",
    "sha256_bytes",
    "sha256_file",
    "validate",
    "verify_rubric_bound",
]
