"""Final RU-first retrieval/tool qualification tests (issue #19).

Hermetic benchmark uses invented fixture text only; no canonical book
text is committed. The versioned artifact ``qualification/aa-retrieval.json``
stays bound to the live RU source checksum, EN reference checksum,
aligned structure version, index configuration, planner schema and
gold-set version. The #9 runtime gate must reject stale artifacts.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

from aa.corpus.structure import SECTION_IDS
from aa.qualification.aa_retrieval import (
    ARTIFACT_FORMAT,
    ARTIFACT_REL,
    BENCHMARK_VERSION,
    GOLD_REL,
    GOLD_VERSION,
    PRODUCTION_CONFIG_VERSION,
    RECALL_GATE_AT_5,
    find_repo_root,
    load_gold,
    run_qualification,
    validate_artifact_payload,
    validate_gold_payload,
)
from aa.retrieval.book_tools import book_read
from aa.retrieval.dense import DENSE_TOP_K
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT, RRF_K
from aa.retrieval.index import build_hybrid_index
from aa.retrieval.lexical import LEXICAL_TOP_K
from aa.retrieval.planner import SCHEMA_VERSION as PLANNER_SCHEMA


def _root() -> pathlib.Path:
    return find_repo_root()


def _artifact() -> dict[str, Any]:
    payload = json.loads((_root() / ARTIFACT_REL).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_gold_covers_full_theme_matrix() -> None:
    cases = load_gold(_root() / GOLD_REL)
    assert len(cases) == 52
    categories = {case.category for case in cases}
    for required in (
        "craving",
        "drinking",
        "relapse",
        "powerlessness",
        "resentment",
        "fear",
        "inventory",
        "amends",
        "prayer",
        "meditation",
        "higher-power",
        "agnosticism",
        "helping-another",
        "family",
        "family-afterward",
        "work",
        "employer",
        "membership",
        "fellowship",
        "book-purpose",
        "book-history",
        "exact-phrase",
        "exact-fact",
        "slang-drinking",
        "slang-family",
        "diminutive",
        "morphology",
        "typo",
        "transposition",
        "terse-followup",
        "ambiguous-sorvalsya",
        "implicit-family",
        "implicit-work",
        "implicit-alcohol",
        "multi-theme",
        "unsupported-out-of-corpus",
        "en-control",
    ):
        assert required in categories
    covered: set[str] = set()
    for case in cases:
        covered.update(case.relevant_sections)
    for section in SECTION_IDS:
        assert section in covered
    lowered = " ".join(c.utterance.casefold() for c in cases if c.language == "ru")
    for token in ("бухаю", "нажрался", "тяпнул", "жинка", "женушка", "сорвался"):
        assert token in lowered
    ru_count = sum(1 for c in cases if c.language == "ru")
    en_count = sum(1 for c in cases if c.language == "en")
    assert ru_count == 46
    assert en_count == 6
    assert en_count < ru_count
    unsupported = [c for c in cases if c.is_unsupported]
    assert len(unsupported) == 3
    for case in unsupported:
        assert case.relevant_sections == ()
    terse = [c for c in cases if c.category == "terse-followup"]
    assert len(terse) >= 2 and all(c.requires_context for c in terse)
    multi = [c for c in cases if c.category == "multi-theme"]
    assert len(multi) >= 2
    for case in multi:
        assert len(case.relevant_sections) >= 2
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids)


def test_gold_rejects_forbidden_strengthening() -> None:
    cases = load_gold(_root() / GOLD_REL)
    for case in cases:
        haystack = f"{case.planner_meaning} {' '.join(case.plan_queries_ru)}".casefold()
        for forbidden in case.forbidden_inferences:
            assert forbidden.casefold() not in haystack


def test_gold_validation_fails_closed() -> None:
    cases = load_gold(_root() / GOLD_REL)
    assert cases
    bad: dict[str, Any] = {"gold_version": GOLD_VERSION, "cases": []}
    try:
        validate_gold_payload(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("empty gold cases must fail closed")


def test_quality_coverage_gates_pass() -> None:
    artifact = _artifact()
    retrieval = artifact["retrieval"]
    assert retrieval["recall_at_5"] >= RECALL_GATE_AT_5
    assert retrieval["recall_at_5"] >= 0.95
    assert retrieval["coverage"] >= 0.90
    assert retrieval["missed_relevant_region_rate"] <= 0.10
    assert retrieval["duplicate_rate"] <= 0.05
    assert retrieval["source_support_success_rate"] >= 0.95
    assert retrieval["false_strengthening_count"] == 0
    gate = artifact["quality_gate"]
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_slang_colloquial_cases_pass() -> None:
    artifact = _artifact()
    assert artifact["quality_gate"]["slang_pass_rate_measured"] == 1.0
    cases = load_gold(_root() / GOLD_REL)
    slang_ids = {c.case_id for c in cases if c.is_slang}
    assert len(slang_ids) >= 6
    rows = {row["case_id"]: row for row in artifact["per_case"]}
    for case_id in slang_ids:
        assert rows[case_id]["recall_at_5"] is True


def test_exact_ru_fidelity_is_complete() -> None:
    artifact = _artifact()
    fidelity = artifact["fidelity"]
    assert fidelity["rate"] == 1.0
    assert fidelity["passed"] == fidelity["total"] and fidelity["total"] > 0


def test_stale_index_rejection_is_complete() -> None:
    artifact = _artifact()
    stale = artifact["stale_index"]
    assert stale["rate"] == 1.0
    assert stale["passed"] == stale["total"] and stale["total"] >= 4


def test_context_tool_cost_recorded() -> None:
    artifact = _artifact()
    retrieval = artifact["retrieval"]
    for key in (
        "mean_tool_calls",
        "p95_tool_calls",
        "mean_source_tokens",
        "p95_source_tokens",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_cold_latency_ms",
        "mean_planner_cost_tokens",
        "mean_evidence_tokens",
    ):
        assert key in retrieval
        assert retrieval[key] >= 0
    assert retrieval["mean_tool_calls"] > 0
    assert retrieval["mean_source_tokens"] > 0
    assert retrieval["mean_latency_ms"] > 0
    assert retrieval["mean_cold_latency_ms"] > 0
    assert retrieval["p95_tool_calls"] >= retrieval["mean_tool_calls"]
    assert retrieval["p95_source_tokens"] >= retrieval["mean_source_tokens"]


def test_branch_contribution_and_second_pass_recorded() -> None:
    artifact = _artifact()
    retrieval = artifact["retrieval"]
    assert retrieval["lexical_recall_at_5"] > 0
    assert retrieval["dense_recall_at_5"] > 0
    assert "second_pass_yield" in retrieval
    assert retrieval["second_pass_yield"] >= 0
    assert retrieval["corpus_section_coverage"] >= 0.9
    assert retrieval["mean_diversity"] > 0
    assert artifact["en_control"]["recall_at_5"] >= 0.95
    assert artifact["en_control"]["role"] == "reference-control"


def test_simplest_passing_config_versioned() -> None:
    artifact = _artifact()
    production = artifact["production"]
    assert production["config_id"] == "ru-first-only"
    assert production["config_version"] == PRODUCTION_CONFIG_VERSION
    assert production["ru_only"] is True
    assert production["evidence_language"] == "ru"
    assert "exact RU canonical text" in production["evidence_rule"]
    assert artifact["format"] == ARTIFACT_FORMAT
    assert artifact["benchmark_version"] == BENCHMARK_VERSION
    assert artifact["gold_version"] == GOLD_VERSION
    assert artifact["planner_schema"] == PLANNER_SCHEMA
    index_config = artifact["index_config"]
    assert index_config["lexical_top_k"] == LEXICAL_TOP_K == 40
    assert index_config["dense_top_k"] == DENSE_TOP_K == 40
    assert index_config["rrf_k"] == RRF_K == 60
    assert index_config["max_per_aspect"] == MAX_CANDIDATES_PER_ASPECT == 12
    assert index_config["chunk_max_chars"] == 1500
    assert artifact["tool_config"]["expand_before"] == 1
    assert artifact["tool_config"]["expand_after"] == 1
    assert artifact["tuning"]
    assert any(t["candidate"].startswith("baseline") and t["passed"] for t in artifact["tuning"])
    assert artifact["chunking"]["selected_max_chars"] == 1500
    assert artifact["expansion"]["selected_before"] == 1


def test_artifact_bound_to_live_versions() -> None:
    artifact = _artifact()
    validate_artifact_payload(artifact, repo_root=_root())
    gold_sha = hashlib.sha256((_root() / GOLD_REL).read_bytes()).hexdigest()
    assert artifact["gold_sha256"] == gold_sha
    ru_manifest = json.loads(
        (_root() / "corpus" / "canonical.ru.manifest.json").read_text(encoding="utf-8")
    )
    assert artifact["bindings"]["ru_artifact_sha256"] == ru_manifest["artifact_sha256"]
    en_manifest = json.loads(
        (_root() / "corpus" / "canonical.manifest.json").read_text(encoding="utf-8")
    )
    assert artifact["bindings"]["en_artifact_sha256"] == en_manifest["artifact_sha256"]
    lock = json.loads((_root() / "corpus" / "embedding.lock.json").read_text(encoding="utf-8"))
    assert artifact["bindings"]["embedding_model_id"] == lock["model_id"]
    assert artifact["bindings"]["embedding_revision"] == lock["revision"]


def test_artifact_rejects_stale_bindings() -> None:
    artifact = _artifact()
    tampered = json.loads(json.dumps(artifact))
    assert isinstance(tampered, dict)
    bindings = tampered["bindings"]
    assert isinstance(bindings, dict)
    bindings["ru_artifact_sha256"] = "0" * 64
    try:
        validate_artifact_payload(tampered, repo_root=_root())
    except ValueError:
        pass
    else:
        raise AssertionError("rotated RU artifact must fail closed")
    tampered2 = json.loads(json.dumps(artifact))
    assert isinstance(tampered2, dict)
    tampered2["gold_sha256"] = "0" * 64
    try:
        validate_artifact_payload(tampered2, repo_root=_root())
    except ValueError:
        pass
    else:
        raise AssertionError("tampered gold sha must fail closed")
    tampered3 = json.loads(json.dumps(artifact))
    assert isinstance(tampered3, dict)
    prod = tampered3["production"]
    assert isinstance(prod, dict)
    prod["config_id"] = "legacy-ru-to-en-only"
    try:
        validate_artifact_payload(tampered3, repo_root=_root())
    except ValueError:
        pass
    else:
        raise AssertionError("legacy production config must fail closed")


def test_all_evidence_resolves_to_exact_ru(tmp_path: pathlib.Path) -> None:
    from aa.qualification.aa_retrieval import build_en_side, build_fixture_full

    cases = load_gold(_root() / GOLD_REL)
    ru_cases = [c for c in cases if c.language == "ru" and not c.is_unsupported]
    full = build_fixture_full()
    ru_manifest = {
        "format": "aa-canonical-manifest-ru/1",
        "artifact_sha256": "r" * 64,
        "edition": "ru-edition",
    }
    en_manifest = {
        "format": "aa-canonical-manifest/1",
        "artifact_sha256": "e" * 64,
        "edition": "en-edition",
    }
    lock = json.loads((_root() / "corpus" / "embedding.lock.json").read_text())
    index = build_hybrid_index(
        full,
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=lock,
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )
    _ = build_en_side(full, workdir=tmp_path / "en")
    by_logical = {r.logical_chunk_id: r for r in index.chunks.values()}
    checked = 0
    for case in ru_cases[:6]:
        from aa.qualification.aa_retrieval import run_case

        result = run_case(index, case)
        for logical_id in result.hit_logical_ids[:2]:
            record = by_logical[logical_id]
            read = book_read(index, logical_id)
            assert read["text"] == record.text
            assert (
                hashlib.sha256(read["text"].encode("utf-8")).hexdigest()
                == read["ru_locator"]["text_sha256"]
            )
            assert ":ru:" in read["chunk_id"]
            assert ":en:" not in read["chunk_id"]
            checked += 1
    assert checked > 0


def test_per_case_rows_cover_every_fixture() -> None:
    artifact = _artifact()
    cases = load_gold(_root() / GOLD_REL)
    rows = artifact["per_case"]
    assert len(rows) == len(cases)
    for row in rows:
        assert set(row) >= {
            "case_id",
            "category",
            "recall_at_5",
            "hit_sections_5",
            "tool_calls",
            "source_tokens",
            "latency_ms",
        }


def test_qualification_benchmark_reproduces_gates() -> None:
    payload = run_qualification(repo_root=_root())
    assert payload["retrieval"]["recall_at_5"] >= 0.95
    assert payload["quality_gate"]["passed"] is True
    assert payload["fidelity"]["rate"] == 1.0
    assert payload["stale_index"]["rate"] == 1.0
    validate_artifact_payload(payload, repo_root=_root())


def test_no_remote_service_in_qualification() -> None:
    root = _root() / "src" / "aa" / "qualification" / "aa_retrieval.py"
    text = root.read_text(encoding="utf-8")
    for snippet in (
        "import openai",
        "from openai",
        "import anthropic",
        "from anthropic",
        "import httpx",
        "from httpx",
        "api.openai.com",
    ):
        assert snippet not in text


def test_runtime_gate_scripts_pass_and_reject_stale() -> None:
    gate = _root() / "scripts" / "verify_runtime_qualification.py"
    assert gate.is_file()
    proc = subprocess.run([sys.executable, str(gate)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    qualifier = _root() / "scripts" / "qualify_aa_retrieval.py"
    assert qualifier.is_file()
