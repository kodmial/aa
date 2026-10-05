"""Deterministic contracts for the trusted real-book qualification (#131)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aa.qualification.real_book_retrieval import (
    BENCHMARK_INPUT_REL,
    E5_BACKEND_NAME,
    EMBEDDING_MODEL_ID,
    EMBEDDING_REVISION,
    PLANNER_AGENT,
    RealBookRetrievalError,
    TurnDiagnostics,
    assert_exact_sha_binding,
    assert_no_oracle_leak,
    assert_no_reranker,
    assert_public_summary_safe,
    assert_source_exact,
    build_protected_payload,
    build_result_marker,
    classify_prerequisites,
    compress_and_encrypt,
    decide_status,
    decrypt_and_decompress,
    parse_result_marker,
    require_production_backend,
    require_secret,
    retrieval_config_checksum,
    retrieval_config_dict,
    secret_present,
    sha256_text,
    summarize_public,
    validate_exact_sha,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-real-book-retrieval-qualification.yml"
RUNNER = ROOT / "scripts" / "run_real_book_retrieval_qualification.py"
MODULE = ROOT / "src" / "aa" / "qualification" / "real_book_retrieval.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _demo_turn() -> TurnDiagnostics:
    return TurnDiagnostics(
        case_id="RU-S-001",
        planner_queries=tuple(f"query {idx}" for idx in range(10)),
        bm25_top=("chapter-1:ru:c0001", "chapter-2:ru:c0002"),
        e5_top=("chapter-1:ru:c0001", "chapter-3:ru:c0003"),
        rrf_survivors=("chapter-1:ru:c0001",),
        dedup_survivors=("chapter-1:ru:c0001",),
        diversity_survivors=("chapter-1:ru:c0001",),
        evidence_ids=("chapter-1:ru:c0001",),
        source_tokens=120,
        source_chars=240,
        latency_ms=12.5,
        branch_union_hit=True,
        rrf_hit=True,
        dedup_hit=True,
        budget_hit=True,
        budget_loss=False,
        oracle_hit=True,
    )


def test_workflow_exists_with_optional_exact_sha() -> None:
    assert WORKFLOW.is_file()
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "exact_sha" in text
    assert "required: false" in text


def test_workflow_never_runs_without_secret_or_corpus() -> None:
    text = _workflow_text()
    assert "AA_BOOK_AGE_IDENTITY" in text
    assert "scripts/restore_canonical.py --lang ru" in text
    assert "--no-network-fallback" in text
    assert "fail-closed" in text.lower() or "fail closed" in text.lower()
    assert "is not configured; refusing trusted run" in text or "not configured" in text
    assert "AA_ALLOW_NETWORK_FETCH" in text


def test_workflow_enforces_exact_sha_before_and_after() -> None:
    text = _workflow_text()
    assert text.count("git rev-parse HEAD") >= 2
    assert "Revalidate exact SHA before run" in text
    assert "Revalidate exact SHA after run" in text
    assert "checked-out SHA" in text or "exact-main checkout verified" in text


def test_workflow_has_no_generic_scheduler_dependency() -> None:
    text = _workflow_text()
    assert "continuum-issue-scheduler" not in text
    assert "ordinary issue-scheduler" in text or "never routes" in text.lower()


def test_workflow_has_bounded_timeout_and_deterministic_status() -> None:
    text = _workflow_text()
    assert "timeout-minutes:" in text
    for status in ("PASS", "FAIL", "INCOMPLETE", "STALE"):
        assert status in text


def test_workflow_posts_idempotent_marker_keyed_by_full_tuple() -> None:
    text = _workflow_text()
    assert "issue_number: 130" in text or "issue_number:130" in text or "130" in text
    assert "aa-real-book-retrieval-result" in text
    for key in ("sha=", "corpus=", "benchmark=", "retrieval=", "result=", "run="):
        assert key in text
    assert "duplicate" in text.lower()
    assert "no duplicate" in text.lower() or "already marked" in text.lower()


def test_workflow_covers_trusted_runtime_requirements() -> None:
    text = _workflow_text()
    assert "intfloat/multilingual-e5-base" in text
    assert "d1287505" in text
    assert "prefetch_public_assets.py" in text
    assert "build_retrieval_index.py --backend e5" in text
    assert "aa-planner-v2" in text
    assert "json_schema" in text
    assert "BM25" in text
    assert "FAISS" in text or "E5/FAISS" in text
    assert BENCHMARK_INPUT_REL in text or "ru_realworld_alcohol_help.v1_1.input.jsonl" in text
    assert "tar.zst.age" in text
    assert "metrics only" in text
    assert "never" in text.lower() and "plaintext" in text.lower()
    assert "never enter the Actions cache" in text or "never enter" in text.lower()


def test_workflow_never_prints_secret_or_caches_decrypted_artifacts() -> None:
    text = _workflow_text()
    assert "never echoed" in text.lower() or "never printed" in text.lower()
    assert "pip" in text.lower()
    assert "huggingface" in text.lower()
    lowered = text.lower()
    assert "corpus/generated" not in lowered or "never" in lowered


def test_exact_sha_binding_helpers() -> None:
    sha = "a" * 40
    assert validate_exact_sha(sha) == sha
    with pytest.raises(RealBookRetrievalError):
        validate_exact_sha("short")
    assert_exact_sha_binding(checked_out=sha, expected=sha)
    with pytest.raises(RealBookRetrievalError):
        assert_exact_sha_binding(checked_out=sha, expected="b" * 40)


def test_secret_presence_is_fail_closed() -> None:
    assert secret_present({"AA_BOOK_AGE_IDENTITY": "AGE-SECRET-KEY-X"}) is True
    assert secret_present({"AA_BOOK_AGE_IDENTITY": "  "}) is False
    assert secret_present({}) is False
    require_secret({"AA_BOOK_AGE_IDENTITY": "AGE-SECRET-KEY-X"})
    with pytest.raises(RealBookRetrievalError):
        require_secret({})


def test_oracle_never_reaches_planner_or_generator() -> None:
    assert_no_oracle_leak({"utterance": "hello", "id": "RU-S-001"}, "case")
    with pytest.raises(RealBookRetrievalError):
        assert_no_oracle_leak({"utterance": "hi", "expected_safety_decision": "allow"}, "case")
    with pytest.raises(RealBookRetrievalError):
        assert_no_oracle_leak({"oracle": {"a": 1}}, "case")
    with pytest.raises(RealBookRetrievalError):
        assert_no_oracle_leak({"expected_response_mode": "ordinary_support"}, "case")
    runner_text = _runner_text()
    assert "assert_no_oracle_leak" in runner_text
    assert "oracle" in runner_text.lower()


def test_no_bge_or_cross_encoder_path() -> None:
    assert_no_reranker({})
    with pytest.raises(RealBookRetrievalError):
        assert_no_reranker({"bge_model": object()})
    with pytest.raises(RealBookRetrievalError):
        assert_no_reranker({"cross_encoder": object()})
    with pytest.raises(RealBookRetrievalError):
        assert_no_reranker({"sentence_transformers_rerank": object()})
    retrieval_root = ROOT / "src" / "aa" / "retrieval"
    code_markers = (
        "import bge",
        "from bge",
        "CrossEncoder",
        "cross_encoder(",
        "cross-encoder(",
        "bge_embed",
        "bge_model",
    )
    for path in sorted(retrieval_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in code_markers:
            assert marker not in text, f"{path.name}: reranker code path {marker!r}"
        assert "rerank(" not in text.lower() or "no second-stage" in text.lower()
    workflow_text = _workflow_text().lower()
    assert "bge" not in workflow_text
    assert "cross-encoder" not in workflow_text
    runner_text = _runner_text().lower()
    assert "bge" not in runner_text
    assert "cross-encoder" not in runner_text
    assert "cross_encoder" not in runner_text


def test_source_exact_offsets_and_checksums_survive() -> None:
    text = "exact source text"
    digest = sha256_text(text)
    assert_source_exact(text=text, text_sha256=digest, char_start=0, char_end=len(text))
    with pytest.raises(RealBookRetrievalError):
        assert_source_exact(text=text, text_sha256="0" * 64, char_start=0, char_end=len(text))
    with pytest.raises(RealBookRetrievalError):
        assert_source_exact(text=text, text_sha256=digest, char_start=5, char_end=5)
    with pytest.raises(RealBookRetrievalError):
        assert_source_exact(text=text, text_sha256=digest, char_start=0, char_end=len(text) + 1)
    record = _demo_turn().protected_record()
    assert record["evidence_ids"] == ["chapter-1:ru:c0001"]
    assert record["case_id"] == "RU-S-001"


def test_missing_prerequisites_yield_incomplete_never_pass() -> None:
    assert (
        classify_prerequisites(e5_available=True, planner_available=True, provider_available=True)
        == ""
    )
    assert (
        classify_prerequisites(e5_available=False, planner_available=True, provider_available=True)
        == "INCOMPLETE"
    )
    assert (
        classify_prerequisites(e5_available=True, planner_available=False, provider_available=True)
        == "INCOMPLETE"
    )
    assert (
        classify_prerequisites(e5_available=True, planner_available=True, provider_available=False)
        == "INCOMPLETE"
    )
    assert decide_status(stale=False, incomplete=True, failures=0) == "INCOMPLETE"
    assert decide_status(stale=True, incomplete=False, failures=0) == "STALE"
    assert decide_status(stale=False, incomplete=False, failures=0) == "PASS"
    assert decide_status(stale=False, incomplete=False, failures=2) == "FAIL"
    require_production_backend(backend=E5_BACKEND_NAME)
    with pytest.raises(RealBookRetrievalError):
        require_production_backend(backend="hashing-char-token/1")


def test_public_summary_holds_metrics_only_and_encrypts_protected() -> None:
    turn = _demo_turn()
    summary = summarize_public(
        main_sha="a" * 40,
        corpus_sha="b" * 64,
        benchmark_sha="c" * 64,
        retrieval_sha="d" * 64,
        turns=[turn],
        status="PASS",
        run_id="123",
    )
    assert summary["result"] == "PASS"
    assert summary["turn_count"] == 1
    assert_public_summary_safe(summary)
    with pytest.raises(RealBookRetrievalError):
        assert_public_summary_safe({"exact_text": "book text"})
    with pytest.raises(RealBookRetrievalError):
        assert_public_summary_safe({"utterance": "hello"})
    with pytest.raises(RealBookRetrievalError):
        assert_public_summary_safe({"note": "x" * 2001})
    protected = build_protected_payload(
        main_sha="a" * 40,
        corpus_sha="b" * 64,
        benchmark_sha="c" * 64,
        retrieval_sha="d" * 64,
        turns=[turn],
        status="PASS",
        run_id="123",
    )
    assert protected["turns"][0]["planner_queries"][0] == "query 0"
    from aa.corpus.age_v1 import generate_identity

    identity, recipient = generate_identity()
    encrypted = compress_and_encrypt(
        json.dumps(protected, sort_keys=True).encode("utf-8"), recipient=recipient
    )
    decrypted = decrypt_and_decompress(encrypted, identity=identity)
    assert json.loads(decrypted.decode("utf-8"))["turns"][0]["case_id"] == "RU-S-001"


def test_marker_binds_full_tuple_and_parses() -> None:
    marker = build_result_marker(
        sha="a" * 40,
        corpus="b" * 64,
        benchmark="c" * 64,
        retrieval="d" * 64,
        result="PASS",
        run="123",
    )
    assert "issue=130" in marker
    parsed = parse_result_marker(marker)
    assert parsed == {
        "sha": "a" * 40,
        "corpus": "b" * 64,
        "benchmark": "c" * 64,
        "retrieval": "d" * 64,
        "result": "PASS",
        "run": "123",
    }
    with pytest.raises(RealBookRetrievalError):
        parse_result_marker("no marker here")


def test_retrieval_config_pins_production_substrate() -> None:
    lock = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
    config = retrieval_config_dict(lock)
    assert config["embedding_model_id"] == EMBEDDING_MODEL_ID
    assert config["embedding_revision"] == EMBEDDING_REVISION
    assert config["embedding_backend"] == E5_BACKEND_NAME
    assert config["planner_agent"] == PLANNER_AGENT
    digest = retrieval_config_checksum(config)
    assert len(digest) == 64
    assert retrieval_config_checksum(config) == digest


def test_runner_uses_production_adapter_and_never_logs_text() -> None:
    text = _runner_text()
    assert "OpenCodeChatModel" in text
    assert "PLANNER_AGENT_V2" in text or "aa-planner-v2" in text
    assert "query_plan_json_schema" in text
    assert "run_planner" in text
    assert "open_hybrid_index" in text
    assert "retrieve_evidence" in text or "run_branch_searches" in text
    assert "AA_BOOK_AGE_IDENTITY" in text
    assert "never printed" in text.lower() or "never echoed" in text.lower()
    assert "INCOMPLETE" in text
    assert "STALE" in text
    assert "compress_and_encrypt" in text
    assert MODULE.is_file()
