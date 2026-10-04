"""RU-first retrieval qualification tests (issue #47).

The hermetic benchmark uses invented fixture text only; no canonical
book text is committed. The versioned decision artifact stays bound to
the live RU source checksum, structure version, index config, planner
schema and gold-set version.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from aa.corpus.structure import SECTION_IDS
from aa.qualification.ru_first import (
    BENCHMARK_VERSION,
    DECISION_FORMAT,
    DECISION_REL,
    GOLD_REL,
    GOLD_VERSION,
    PRODUCTION_CONFIG_VERSION,
    QUALITY_GATE_RECALL_AT_5,
    REQUIRED_GOLD_CATEGORIES,
    build_en_side,
    build_fixture_full,
    find_repo_root,
    load_gold,
    run_benchmark,
    run_fixture_a,
    run_fixture_b,
    run_fixture_c,
    summarize,
    validate_decision_payload,
    validate_gold_payload,
)
from aa.retrieval.dense import DENSE_TOP_K
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT, RRF_K
from aa.retrieval.index import build_hybrid_index
from aa.retrieval.lexical import LEXICAL_TOP_K
from aa.retrieval.planner import SCHEMA_VERSION as PLANNER_SCHEMA


def _root() -> pathlib.Path:
    return find_repo_root()


def _decision() -> dict[str, Any]:
    payload = json.loads((_root() / DECISION_REL).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_gold_set_validates_and_covers_required_categories() -> None:
    cases = load_gold(_root() / GOLD_REL)
    assert len(cases) == 20
    categories = {case.category for case in cases}
    for required in REQUIRED_GOLD_CATEGORIES:
        assert required in categories
    lowered = " ".join(case.utterance.casefold() for case in cases)
    for token in ("бухаю", "нажрался", "тяпнул", "жинка", "женушка", "сорвался"):
        assert token in lowered
    for case in cases:
        assert case.relevant_sections
        assert set(case.relevant_sections) <= set(SECTION_IDS)
        assert case.allowed_interpretations
        assert case.forbidden_inferences
        assert case.plan_queries_ru
        assert case.en_gloss_queries
        assert case.planner_meaning
    ids = [case.case_id for case in cases]
    assert len(set(ids)) == len(ids)


def test_gold_rejects_forbidden_strengthening() -> None:
    cases = load_gold(_root() / GOLD_REL)
    for case in cases:
        haystack = f"{case.planner_meaning} {' '.join(case.plan_queries_ru)}".casefold()
        for forbidden in case.forbidden_inferences:
            assert forbidden.casefold() not in haystack


def test_gold_has_terse_and_ambiguous_and_multi_theme() -> None:
    cases = load_gold(_root() / GOLD_REL)
    by_id = {case.case_id: case for case in cases}
    assert by_id["RU-RF-009"].requires_context is True
    assert by_id["RU-RF-019"].requires_context is True
    assert by_id["RU-RF-010"].category == "ambiguous-sorvalsya"
    assert len(by_id["RU-RF-014"].relevant_sections) >= 3
    assert by_id["RU-RF-015"].category == "exact-phrase"
    assert by_id["RU-RF-016"].category == "exact-fact"


def test_benchmark_ru_first_meets_quality_gate() -> None:
    payload = run_benchmark(repo_root=_root())
    gate = payload["quality_gate"]
    assert isinstance(gate, dict)
    assert gate["passed"] is True
    assert gate["failures"] == []
    assert payload["configs"]["a_ru_first"]["recall_at_5"] >= QUALITY_GATE_RECALL_AT_5
    assert payload["configs"]["a_ru_first"]["false_strengthening_count"] == 0
    assert gate["slang_pass_rate_measured"] == 1.0


def test_benchmark_measures_en_secondary_incremental_yield() -> None:
    payload = run_benchmark(repo_root=_root())
    en = payload["en_secondary"]
    assert "incremental_recall_at_5" in en
    assert "new_relevant_sections" in en
    assert "latency_ratio_b_over_a" in en
    assert en["latency_ratio_b_over_a"] > 0
    # Materiality rule: RU-only stays unless the EN branch shows a real gain.
    assert payload["production"]["config_id"] == "ru-first-only"
    assert en["decision"] == "disabled"
    assert en["incremental_recall_at_5"] < 0.05


def test_benchmark_records_legacy_control() -> None:
    payload = run_benchmark(repo_root=_root())
    assert "c_legacy_ru_to_en_only" in payload["configs"]
    legacy = payload["configs"]["c_legacy_ru_to_en_only"]
    assert legacy["fixtures"] == payload["configs"]["a_ru_first"]["fixtures"]
    assert legacy["recall_at_5"] <= payload["configs"]["a_ru_first"]["recall_at_5"]
    assert payload["production"]["config_id"] != "legacy-ru-to-en-only"
    assert "cannot become the production default" in payload["notes"].lower() or (
        "cannot become production default" in payload["notes"].lower()
    )


def test_slang_cases_pass_without_strengthening() -> None:
    payload = run_benchmark(repo_root=_root())
    rows = payload["per_fixture"]
    assert isinstance(rows, list) and rows
    cases = load_gold(_root() / GOLD_REL)
    slang_ids = {case.case_id for case in cases if case.is_slang}
    assert slang_ids
    for row in rows:
        assert isinstance(row, dict)
        if str(row["case_id"]) in slang_ids:
            assert row["a_recall_at_5"] is True
    assert payload["configs"]["a_ru_first"]["false_strengthening_count"] == 0


def test_production_configuration_is_versioned() -> None:
    decision = _decision()
    production = decision["production"]
    assert production["config_id"] == "ru-first-only"
    assert production["config_version"] == PRODUCTION_CONFIG_VERSION
    assert production["ru_only"] is True
    assert production["evidence_language"] == "ru"
    assert "exact RU canonical text" in production["evidence_rule"]
    assert decision["format"] == DECISION_FORMAT
    assert decision["benchmark_version"] == BENCHMARK_VERSION
    assert decision["gold_version"] == GOLD_VERSION
    assert decision["planner_schema"] == PLANNER_SCHEMA
    index_config = decision["index_config"]
    assert index_config["lexical_top_k"] == LEXICAL_TOP_K == 40
    assert index_config["dense_top_k"] == DENSE_TOP_K == 40
    assert index_config["rrf_k"] == RRF_K == 60
    assert index_config["max_per_aspect"] == MAX_CANDIDATES_PER_ASPECT == 12


def test_decision_artifact_is_bound_to_live_versions() -> None:
    decision = _decision()
    validate_decision_payload(decision, repo_root=_root())
    gold_sha = hashlib.sha256((_root() / GOLD_REL).read_bytes()).hexdigest()
    assert decision["gold_sha256"] == gold_sha
    ru_manifest = json.loads(
        (_root() / "corpus" / "canonical.ru.manifest.json").read_text(encoding="utf-8")
    )
    assert decision["bindings"]["ru_artifact_sha256"] == ru_manifest["artifact_sha256"]
    lock = json.loads((_root() / "corpus" / "embedding.lock.json").read_text(encoding="utf-8"))
    assert decision["bindings"]["embedding_model_id"] == lock["model_id"]
    assert decision["bindings"]["embedding_revision"] == lock["revision"]


def test_all_final_evidence_resolves_to_exact_ru_text(tmp_path: pathlib.Path) -> None:
    cases = load_gold(_root() / GOLD_REL)
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
    en_side = build_en_side(full, workdir=tmp_path / "en")
    by_logical = {record.logical_chunk_id: record for record in index.chunks.values()}
    for case in cases:
        result_b = run_fixture_b(index, en_side, case)
        for logical_id in result_b.hit_logical_ids:
            record = by_logical[logical_id]
            assert record.text
            assert hashlib.sha256(record.text.encode("utf-8")).hexdigest() == record.text_sha256
            assert record.chunk_id.startswith(f"{record.section}:ru:")
            assert ":en:" not in record.chunk_id


def test_branch_contribution_and_budgets_are_measured() -> None:
    payload = run_benchmark(repo_root=_root())
    summary_a = payload["configs"]["a_ru_first"]
    for key in (
        "lexical_hit_rate",
        "dense_hit_rate",
        "both_branch_rate",
        "duplicate_rate",
        "mean_latency_ms",
        "mean_planner_cost_tokens",
        "mean_evidence_tokens",
        "coverage",
        "missed_relevant_region_rate",
    ):
        assert key in summary_a
        assert summary_a[key] >= 0
    assert summary_a["mean_latency_ms"] > 0
    assert summary_a["mean_planner_cost_tokens"] > 0
    assert summary_a["mean_evidence_tokens"] > 0
    assert summary_a["lexical_hit_rate"] > 0
    assert summary_a["dense_hit_rate"] > 0


def test_configs_cover_every_fixture_with_per_fixture_rows() -> None:
    payload = run_benchmark(repo_root=_root())
    cases = load_gold(_root() / GOLD_REL)
    assert len(payload["per_fixture"]) == len(cases)
    for row in payload["per_fixture"]:
        assert set(row) >= {
            "case_id",
            "relevant_sections",
            "a_hit_sections",
            "a_recall_at_5",
            "b_hit_sections",
            "b_recall_at_5",
            "c_hit_sections",
            "c_recall_at_5",
        }


def test_fixture_results_bound_candidate_counts(tmp_path: pathlib.Path) -> None:
    cases = load_gold(_root() / GOLD_REL)
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
    en_side = build_en_side(full, workdir=tmp_path / "en")
    results_a = [run_fixture_a(index, case) for case in cases]
    results_b = [run_fixture_b(index, en_side, case) for case in cases]
    results_c = [run_fixture_c(index, en_side, case) for case in cases]
    for result in (*results_a, *results_b, *results_c):
        assert len(result.hit_sections) <= 12
        assert len(result.hit_logical_ids) <= 12
    summary_a = summarize(results_a, config_id="a_ru_first")
    assert summary_a.fixtures == len(cases)


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


def test_no_remote_retrieval_service_in_qualification() -> None:
    root = _root() / "src" / "aa" / "qualification" / "ru_first.py"
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
