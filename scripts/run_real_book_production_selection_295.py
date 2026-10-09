#!/usr/bin/env python3
"""Trusted #295 production-selector real-book diagnostics (issue #295).

Clearly separated from the frozen #130 RRF-only benchmark in
``scripts/run_real_book_retrieval_qualification.py`` (which stays
unchanged): this runner exercises THE ACTUAL PRODUCTION
``aretrieve_with_semantic_selection`` / ``retrieval_node(selection_model=...)``
path with the real OpenCode provider, resolved intent + carried
multi-turn context, and model selection BEFORE book budget, over the
restored canonical RU E5/BM25/FAISS substrate.

Privacy: logs carry IDs/counts/ranks/latencies only; exact book text
lives only in the compressed + age-encrypted bundle.
Exit codes: 0 PASS, 1 FAIL, 2 INCOMPLETE, 3 STALE.
A run without a deep-rank decisive case is INCOMPLETE, never PASS.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Reuse the frozen benchmark projection helpers without mutating that script.
sys.path.insert(0, str(ROOT / "scripts"))

import run_real_book_retrieval_qualification as frozen  # noqa: E402

from aa.corpus.public_cache import (  # noqa: E402
    default_lock_path,
    load_embedding_lock,
    resolve_hf_cache_dir,
    resolve_model_root,
    verify_cached_model,
)
from aa.opencode.errors import OpenCodeRateLimitError  # noqa: E402
from aa.qualification.real_book_production_selection_295 import (  # noqa: E402
    ProductionSelectionDiagnostics,
    build_protected_payload_295,
    decide_status_295,
    run_production_selection_turn,
    summarize_public_295,
)
from aa.qualification.real_book_retrieval import (  # noqa: E402
    EXIT_BY_STATUS,
    RealBookRetrievalError,
    assert_no_oracle_leak,
    assert_no_reranker,
    benchmark_binding_checksum,
    corpus_checksum,
    production_retrieval_checksum,
    secret_present,
    validate_exact_sha,
)
from aa.retrieval.evidence import RetrievalConfig  # noqa: E402
from aa.retrieval.index import open_hybrid_index  # noqa: E402

DEFAULT_OUT = ROOT / "eval-real-book-295-out"
STATUS_FILENAME = "result-295.json"
SUMMARY_FILENAME = "real-book-production-selection-295-summary.json"
PROTECTED_FILENAME = "real-book-production-selection-295-protected.tar.zst.age"
PROTECTED_SHA_FILENAME = "real-book-production-selection-295-protected.tar.zst.age.sha256"
EXIT_RUNNER_RESTART_REQUIRED = 75


def _fail_incomplete(out_dir: Path, *, reason: str, main_sha: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": "INCOMPLETE", "reason": reason, "main_sha": main_sha, "issue": 295}
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"#295 production-selection INCOMPLETE: {reason}", file=sys.stderr)
    return EXIT_BY_STATUS["INCOMPLETE"]


def _checked_out_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    )
    if proc.returncode != 0:
        raise RealBookRetrievalError("cannot determine checked-out SHA")
    return proc.stdout.strip()


async def _plan_with_production_adapter(
    utterance: str,
    *,
    primary_model: str,
    fallback_model: str,
    recent_user_turns: tuple[str, ...] = (),
) -> tuple[list[str], str]:
    """Real mandatory planner; returns (queries, resolved_intent)."""
    from langchain_core.messages import HumanMessage

    from aa.conversation.model_adapter import PLANNER_AGENT_V2, OpenCodeChatModel
    from aa.conversation.planner_node import query_plan_json_schema, run_planner
    from aa.opencode.client import HttpOpenCodeClient

    base_url = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    client = HttpOpenCodeClient(base_url)
    health = await client.health()
    if not health.healthy:
        raise RealBookRetrievalError("opencode runtime is not healthy")
    model = OpenCodeChatModel(
        client, agent=PLANNER_AGENT_V2, primary_model=primary_model, fallback_model=fallback_model
    )
    recent = [HumanMessage(content=item) for item in recent_user_turns]
    plan = await run_planner(utterance, model=model, recent=recent)
    _ = query_plan_json_schema()
    return list(plan.queries), str(plan.resolved_intent or utterance)


def _selector_model(*, primary_model: str, fallback_model: str) -> Any:
    """Real OpenCode selector bound exactly like production graph."""
    from aa.conversation.model_adapter import PLANNER_AGENT_V2, OpenCodeChatModel
    from aa.opencode.client import HttpOpenCodeClient

    base_url = os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    client = HttpOpenCodeClient(base_url)
    # Same agent/profile binding as build_turn_graph selection_model.
    return OpenCodeChatModel(
        client, agent=PLANNER_AGENT_V2, primary_model=primary_model, fallback_model=fallback_model
    )


from typing import Any  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trusted #295 production-selector diagnostics.")
    parser.add_argument("--main-sha", required=True, help="Exact PR-head SHA under test")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--recipient", required=True, help="Age recipient for encryption")
    parser.add_argument("--limit", type=int, default=0, help="Optional turn cap (0 = all)")
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        expected_sha = validate_exact_sha(args.main_sha)
    except RealBookRetrievalError as exc:
        print(f"#295 diagnostics failed: {exc}", file=sys.stderr)
        return EXIT_BY_STATUS["INCOMPLETE"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        checked = _checked_out_sha()
        if checked != expected_sha:
            payload = {
                "result": "STALE",
                "reason": "checked-out SHA is not the expected exact SHA",
                "main_sha": expected_sha,
                "issue": 295,
            }
            (out_dir / STATUS_FILENAME).write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            print("#295 diagnostics STALE: SHA mismatch", file=sys.stderr)
            return EXIT_BY_STATUS["STALE"]
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)

    if not secret_present():
        return _fail_incomplete(
            out_dir, reason="AA_BOOK_AGE_IDENTITY is not configured", main_sha=expected_sha
        )
    try:
        assert_no_reranker()
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    try:
        corpus_sha = corpus_checksum(ROOT)
        benchmark_sha = benchmark_binding_checksum(ROOT)
        retrieval_sha = production_retrieval_checksum(ROOT)
    except (RealBookRetrievalError, OSError) as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    try:
        lock = load_embedding_lock(default_lock_path())
    except Exception as exc:
        return _fail_incomplete(
            out_dir, reason=f"embedding lock invalid: {exc}", main_sha=expected_sha
        )
    model_root = resolve_model_root(resolve_hf_cache_dir())
    if not verify_cached_model(model_root, lock):
        return _fail_incomplete(
            out_dir,
            reason="pinned E5 snapshot is not cached locally; refusing network fetch",
            main_sha=expected_sha,
        )

    from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL

    primary_model = (os.environ.get("OPENCODE_MODEL") or DEFAULT_PRIMARY_MODEL).strip()
    fallback_model = (os.environ.get("OPENCODE_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL).strip()
    if not primary_model:
        return _fail_incomplete(
            out_dir, reason="OPENCODE_MODEL is not configured", main_sha=expected_sha
        )

    index_dir = (
        Path(args.index_dir) if args.index_dir else ROOT / "corpus" / "generated" / "retrieval"
    )
    try:
        from aa.qualification.real_book_retrieval import RU_MANIFEST_REL

        index = open_hybrid_index(index_dir, ru_manifest_path=ROOT / RU_MANIFEST_REL)
    except Exception as exc:
        return _fail_incomplete(
            out_dir, reason=f"hybrid index unavailable: {type(exc).__name__}", main_sha=expected_sha
        )
    try:
        from aa.qualification.real_book_retrieval import require_production_backend

        require_production_backend(backend=str(index.metadata.get("embedding_backend", "")))
    except RealBookRetrievalError as exc:
        try:
            from aa.retrieval.index import close_hybrid_index

            close_hybrid_index(index)
        except Exception:
            pass
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    if str(index.metadata.get("ru_artifact_sha256", "")) != corpus_sha:
        try:
            from aa.retrieval.index import close_hybrid_index

            close_hybrid_index(index)
        except Exception:
            pass
        return _fail_incomplete(
            out_dir, reason="index corpus binding is stale", main_sha=expected_sha
        )

    try:
        input_records = frozen._load_jsonl(ROOT / frozen.BENCHMARK_INPUT_REL)
        oracle_records = frozen._load_jsonl(ROOT / frozen.BENCHMARK_ORACLE_REL)
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    oracle_by_case = frozen._oracle_by_case(oracle_records)
    turns = frozen._input_turns(input_records, oracle_by_case)
    if args.limit and args.limit > 0:
        turns = turns[: args.limit]
    if not turns:
        return _fail_incomplete(
            out_dir, reason="benchmark has no book-required turns", main_sha=expected_sha
        )
    oracle_regions = frozen._oracle_regions(oracle_by_case)

    from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig

    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(
            base_url=os.environ.get("OPENCODE_BASE_URL", "http://127.0.0.1:4096"),
            command=os.environ.get("OPENCODE_COMMAND", "opencode"),
            workdir=os.environ.get("OPENCODE_WORKDIR", str(ROOT)),
            model=primary_model,
        )
    )
    try:
        asyncio.run(runtime.start())
    except Exception as exc:
        return _fail_incomplete(
            out_dir,
            reason=f"opencode runtime startup failed: {type(exc).__name__}",
            main_sha=expected_sha,
        )

    config = RetrievalConfig()
    diagnostics: list[ProductionSelectionDiagnostics] = []
    infra_failures = 0
    selector = _selector_model(primary_model=primary_model, fallback_model=fallback_model)
    try:
        for case_id, utterance, recent_user_turns in turns:
            if not utterance.strip():
                infra_failures += 1
                continue
            try:
                queries, resolved_intent = asyncio.run(
                    _plan_with_production_adapter(
                        utterance,
                        primary_model=primary_model,
                        fallback_model=fallback_model,
                        recent_user_turns=recent_user_turns,
                    )
                )
            except OpenCodeRateLimitError as exc:
                payload = {
                    "result": "INCOMPLETE",
                    "reason": "OpenCode 429 requires fresh runner recovery",
                    "reason_code": "OPENCODE_429_RESTART_REQUIRED",
                    "resume_after_case_count": len(diagnostics),
                    "main_sha": expected_sha,
                    "issue": 295,
                }
                (out_dir / STATUS_FILENAME).write_text(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "planner_error": type(exc).__name__,
                            "runner_restart_required": True,
                        }
                    )
                )
                return EXIT_RUNNER_RESTART_REQUIRED
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "planner_error": type(exc).__name__,
                            "utterance_chars": len(utterance),
                        }
                    )
                )
                infra_failures += 1
                continue
            if not queries:
                infra_failures += 1
                continue
            assert_no_oracle_leak({"queries": queries}, case_id)
            conversation_context = " ".join(recent_user_turns[-6:])[:4000]
            try:
                diag = asyncio.run(
                    run_production_selection_turn(
                        index=index,
                        case_id=case_id,
                        planner_queries=queries,
                        resolved_intent=resolved_intent,
                        conversation_context=conversation_context,
                        user_message=utterance,
                        selection_model=selector,
                        oracle_sections=oracle_regions.get(case_id, set()),
                        config=config,
                    )
                )
            except OpenCodeRateLimitError as exc:
                payload = {
                    "result": "INCOMPLETE",
                    "reason": "OpenCode 429 requires fresh runner recovery",
                    "reason_code": "OPENCODE_429_RESTART_REQUIRED",
                    "resume_after_case_count": len(diagnostics),
                    "main_sha": expected_sha,
                    "issue": 295,
                }
                (out_dir / STATUS_FILENAME).write_text(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "selection_error": type(exc).__name__,
                            "runner_restart_required": True,
                        }
                    )
                )
                return EXIT_RUNNER_RESTART_REQUIRED
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "selection_error": type(exc).__name__,
                            "utterance_chars": len(utterance),
                        }
                    )
                )
                infra_failures += 1
                continue
            diagnostics.append(diag)
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "planner_queries": len(queries),
                        "pool_unique": diag.fused_pool_unique,
                        "selected": diag.selected_count,
                        "max_rank": diag.selection_max_rank,
                        "deep_gt5": diag.deep_rank_gt5,
                        "deep_gt16": diag.deep_rank_gt16,
                        "beyond_500": diag.beyond_500_chars,
                        "fidelity_ok": diag.fidelity_ok,
                        "selector": diag.selector_available,
                        "latency_ms": round(diag.latency_ms, 1),
                        "oracle_hit": diag.oracle_hit,
                    }
                )
            )
    finally:
        try:
            from aa.retrieval.index import close_hybrid_index

            close_hybrid_index(index)
        except Exception:
            pass
        try:
            asyncio.run(runtime.stop())
        except Exception:
            pass

    try:
        final_sha = _checked_out_sha()
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    if final_sha != expected_sha:
        status = "STALE"
    elif infra_failures:
        status = "INCOMPLETE"
    else:
        misses = sum(1 for item in diagnostics if item.oracle_hit is False)
        status = decide_status_295(
            stale=False, incomplete=False, failures=misses, turns=diagnostics
        )

    from aa.qualification.real_book_retrieval import compress_and_encrypt

    protected = build_protected_payload_295(
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
        turns=diagnostics,
        status=status,
        run_id=str(run_id),
    )
    protected_bytes = json.dumps(protected, sort_keys=True, ensure_ascii=False).encode("utf-8")
    try:
        encrypted = compress_and_encrypt(protected_bytes, recipient=args.recipient)
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    (out_dir / PROTECTED_FILENAME).write_bytes(encrypted)
    (out_dir / PROTECTED_SHA_FILENAME).write_text(
        hashlib.sha256(encrypted).hexdigest() + "\n", encoding="utf-8"
    )
    selector_available = bool(diagnostics) and all(
        t.selector_available and not t.fallback_used for t in diagnostics
    )
    summary = summarize_public_295(
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
        turns=diagnostics,
        status=status,
        run_id=str(run_id),
        selector_available=selector_available,
    )
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(
            {
                "result": status,
                "main_sha": expected_sha,
                "corpus_sha256": corpus_sha,
                "benchmark_sha256": benchmark_sha,
                "retrieval_config_sha256": retrieval_sha,
                "turn_count": len(diagnostics),
                "infra_failures": infra_failures,
                "issue": 295,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": status,
                "issue": 295,
                "turns": len(diagnostics),
                "infra_failures": infra_failures,
                "deep_gt16": sum(1 for t in diagnostics if t.deep_rank_gt16),
                "beyond_500": sum(1 for t in diagnostics if t.beyond_500_chars),
            }
        )
    )
    # Keep the 429-resume sentinel distinct from STALE (3).
    _ = time.perf_counter
    return EXIT_BY_STATUS[status]


if __name__ == "__main__":
    raise SystemExit(main())
