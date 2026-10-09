"""#295 premerge proof: trusted real-book check uses the production selector path.

Regression ensuring the #295 diagnostic path exercises
``aretrieve_with_semantic_selection`` / ``retrieval_node(selection_model=...)``
and never silently falls back to the frozen RRF-only ``_run_turn`` for #295,
fails closed without inventing coverage, and uses true full canonical
passages on success.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from aa.qualification.real_book_production_selection_295 import (
    ProductionSelectionDiagnostics,
    assert_full_evidence_fidelity,
    decide_status_295,
    run_production_selection_turn,
    summarize_public_295,
)
from aa.qualification.real_book_retrieval import RealBookRetrievalError

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-real-book-retrieval-qualification.yml"
FROZEN_RUNNER = ROOT / "scripts" / "run_real_book_retrieval_qualification.py"
PROOF_RUNNER = ROOT / "scripts" / "run_real_book_production_selection_295.py"
PROOF_MODULE = ROOT / "src" / "aa" / "qualification" / "real_book_production_selection_295.py"


def _demo_diag(**over: Any) -> ProductionSelectionDiagnostics:
    base: dict[str, Any] = {
        "case_id": "RU-S-001",
        "planner_query_count": 3,
        "fused_pool_unique": 64,
        "preview_count": 64,
        "selected_count": 4,
        "selection_max_rank": 20,
        "deep_rank_gt5": True,
        "deep_rank_gt16": True,
        "full_chars_total": 4000,
        "full_max_chars": 1200,
        "beyond_500_chars": True,
        "source_sections_covered": 3,
        "fidelity_ok": True,
        "selector_available": True,
        "fallback_used": False,
        "need_more_detail": False,
        "followup_added": 0,
        "latency_ms": 42.0,
        "selection_latency_ms": 9.0,
        "had_429": False,
        "oracle_hit": True,
    }
    base.update(over)
    return ProductionSelectionDiagnostics(**base)


def test_proof_runner_uses_production_selection_path_not_frozen_turn() -> None:
    text = PROOF_RUNNER.read_text(encoding="utf-8")
    assert "aretrieve_with_semantic_selection" in text
    assert "run_production_selection_turn" in text
    assert "OpenCodeChatModel" in text
    assert "selection_model" in text
    # The #295 runner must not silently call the frozen RRF-only helper.
    assert "_run_turn" not in text
    assert "run_branch_searches" not in text
    assert "fuse_query_pool" not in text
    assert "dedup_and_diversify" not in text
    # Real provider + intent/context before budget.
    assert "resolved_intent" in text
    assert "conversation_context" in text


def test_proof_module_calls_production_path() -> None:
    text = PROOF_MODULE.read_text(encoding="utf-8")
    assert "aretrieve_with_semantic_selection" in text
    assert "selection_max_rank" in text
    assert "selection_deep_rank_gt16" in text or "deep_rank_gt16" in text


def test_frozen_runner_untouched_for_130() -> None:
    text = FROZEN_RUNNER.read_text(encoding="utf-8")
    # Frozen #130 path stays RRF-only; it must not claim production selection.
    assert "def _run_turn" in text
    assert "aretrieve_with_semantic_selection" not in text
    assert "result.json" in text


def test_workflow_keeps_frozen_section_and_adds_separated_295() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Execute frozen benchmark via real planner + production retrieval" in text
    assert "run_real_book_retrieval_qualification.py" in text
    assert "Execute #295 production-selector diagnostics (separate premerge proof)" in text
    assert "run_real_book_production_selection_295.py" in text
    assert "result-295.json" in text
    assert "eval-real-book-295-out" in text
    assert "aa-real-book-production-selection-295-result" in text
    assert "issue_number: 295" in text or "issue_number:295" in text
    # Non-main SHA is premerge evidence only, never exact-main PASS.
    assert "premerge" in text.lower()


def test_no_deep_rank_decisive_case_is_incomplete_never_pass() -> None:
    shallow = _demo_diag(selection_max_rank=2, deep_rank_gt5=False, deep_rank_gt16=False)
    assert (
        decide_status_295(stale=False, incomplete=False, failures=0, turns=[shallow])
        == "INCOMPLETE"
    )
    no_beyond = _demo_diag(beyond_500_chars=False, full_max_chars=400)
    assert (
        decide_status_295(stale=False, incomplete=False, failures=0, turns=[no_beyond])
        == "INCOMPLETE"
    )
    assert decide_status_295(stale=False, incomplete=False, failures=0, turns=[]) == "INCOMPLETE"
    # The model object may be bound but fail internally and select lexically:
    # it is not acceptable evidence that the actual OpenCode selector worked.
    assert (
        decide_status_295(
            stale=False,
            incomplete=False,
            failures=0,
            turns=[_demo_diag(selector_available=False, fallback_used=True)],
        )
        == "INCOMPLETE"
    )
    assert (
        decide_status_295(stale=False, incomplete=False, failures=0, turns=[_demo_diag()]) == "PASS"
    )
    assert (
        decide_status_295(stale=True, incomplete=False, failures=0, turns=[_demo_diag()]) == "STALE"
    )
    assert (
        decide_status_295(stale=False, incomplete=False, failures=1, turns=[_demo_diag()]) == "FAIL"
    )


def test_public_summary_is_metrics_only() -> None:
    summary = summarize_public_295(
        main_sha="a" * 40,
        corpus_sha="b" * 64,
        benchmark_sha="c" * 64,
        retrieval_sha="d" * 64,
        turns=[_demo_diag()],
        status="PASS",
        run_id="9",
        selector_available=True,
    )
    assert summary["result"] == "PASS"
    assert summary["decisive_count"] == 1
    blob = json.dumps(summary, ensure_ascii=False)
    assert "exact_text" not in blob
    assert "utterance" not in blob


def _fixture_index(tmp_path: Any) -> Any:
    from aa.corpus.structure import SECTION_IDS, build_full_structure
    from aa.retrieval.index import build_hybrid_index

    # Ensure the decisive section text is long with a unique marker past 500.
    decisive = (
        ("Вступление о поддержке. " * 30)
        + "РЕШАЮЩИЙ-ПРИЗНАК-ТЯГА-УТРОМ "
        + ("Продолжение смысла. " * 40)
    )
    assert len(decisive) > 900
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for idx, sid in enumerate(SECTION_IDS):
        text = (
            decisive
            if sid == "chapter-5"
            else f"Фоновый отрывок {idx}. Общие слова без признака. Второй абзац."
        )
        en_sections.append(
            {
                "id": sid,
                "title": f"EN {sid}",
                "text": f"Fixture EN {sid} opening. Second sentence.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": sid,
                "title": f"RU {sid}",
                "text": text,
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        max_tokens=10,
        token_counter=lambda t: max(1, len(t.split())),
    )
    lock = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
    return build_hybrid_index(
        full,
        ru_manifest={
            "format": "aa-canonical-manifest-ru/1",
            "artifact_sha256": "r" * 64,
            "edition": "ru-edition",
        },
        en_manifest={
            "format": "aa-canonical-manifest/1",
            "artifact_sha256": "e" * 64,
            "edition": "en-edition",
        },
        embedding_lock=lock,
        out_dir=pathlib.Path(str(tmp_path)) / "retrieval",
        backend="hashing",
    )


async def _run_deep_case(tmp_path: Any) -> Any:
    from aa.retrieval.evidence import broad_fused_ranking
    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index(tmp_path)
    try:
        queries = ["поддержка утром тяга разбор"]
        fused, ordered = broad_fused_ranking(index, queries)
        assert len(ordered) > 5
        deep_pos = 16 if len(ordered) > 16 else 5
        deep_chunk = ordered[deep_pos][0]

        class _DeepModel:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                assert deep_chunk in prompt
                return {
                    "selected_chunk_ids": [deep_chunk],
                    "need_more_detail": False,
                    "followup_queries": [],
                }

        diag = await run_production_selection_turn(
            index=index,
            case_id="RU-S-001",
            planner_queries=queries,
            resolved_intent="утром разбирать тягу с поддержкой",
            conversation_context="прошлый вопрос о вечере",
            user_message="Как разбирать тягу утром?",
            selection_model=_DeepModel(),
            oracle_sections=set(),
            config=None,
        )
        return index, diag, deep_chunk
    except Exception:
        close_hybrid_index(index)
        raise


def test_production_turn_selects_deep_rank_with_full_fidelity(tmp_path: Any) -> None:
    import asyncio

    from aa.retrieval.index import close_hybrid_index

    index, diag, deep_chunk = asyncio.run(_run_deep_case(tmp_path))
    try:
        assert diag.selector_available is True
        assert diag.fallback_used is False
        assert diag.fidelity_ok is True
        assert diag.selection_max_rank >= 5
        assert diag.preview_count >= diag.selected_count > 0
        assert diag.full_max_chars > 0
        assert diag.planner_query_count == 1
        assert diag.had_429 is False
    finally:
        close_hybrid_index(index)


def test_production_turn_fails_closed_on_unknown_ids(tmp_path: Any) -> None:
    import asyncio

    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index(tmp_path)
    try:

        class _BadModel:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                return {
                    "selected_chunk_ids": ["chapter-9#c9999-unknown"],
                    "need_more_detail": False,
                    "followup_queries": [],
                }

        # Unknown ids fail closed inside the selector to the heuristic, which
        # still yields real indexed evidence (never invented coverage).
        diag = asyncio.run(
            run_production_selection_turn(
                index=index,
                case_id="RU-S-002",
                planner_queries=["поддержка утром"],
                resolved_intent="утром поддержка",
                selection_model=_BadModel(),
            )
        )
        assert diag.fidelity_ok is True
    finally:
        close_hybrid_index(index)


def test_production_turn_429_propagates_for_checkpoint(tmp_path: Any) -> None:
    import asyncio

    from aa.opencode.errors import OpenCodeRateLimitError
    from aa.retrieval.index import close_hybrid_index

    index = _fixture_index(tmp_path)
    try:

        class _Limited:
            async def ainvoke_structured(
                self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int
            ) -> object:
                _ = (prompt, system, schema, retry_count)
                raise OpenCodeRateLimitError("http=429")

        with pytest.raises(OpenCodeRateLimitError):
            asyncio.run(
                run_production_selection_turn(
                    index=index,
                    case_id="RU-S-003",
                    planner_queries=["поддержка утром"],
                    resolved_intent="утром поддержка",
                    selection_model=_Limited(),
                )
            )
    finally:
        close_hybrid_index(index)


def test_full_fidelity_rejects_invented_or_truncated_pack() -> None:
    from aa.retrieval.evidence import EvidencePack

    with pytest.raises(RealBookRetrievalError):
        assert_full_evidence_fidelity(
            EvidencePack(passages=(), total_tokens=0, corpus_version="x"), object()
        )
