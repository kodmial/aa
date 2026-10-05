#!/usr/bin/env python3
"""Trusted real-book retrieval qualification runner (issue #131).

Executes the frozen real-world benchmark input against the production
retrieval substrate inside the dedicated trusted workflow:

1. exact-SHA binding (``git rev-parse HEAD`` must equal ``--main-sha``);
2. ``AA_BOOK_AGE_IDENTITY`` presence validated fail-closed;
3. canonical RU corpus verified against the committed manifest;
4. pinned ``intfloat/multilingual-e5-base`` snapshot validated via the
   public model-cache mechanism (missing snapshot yields INCOMPLETE);
5. real mandatory planner invoked through the production OpenCode
   structured-output adapter/profile (``aa-planner-v2``);
6. RAM-resident BM25 + E5/FAISS index opened (production backend only);
7. frozen benchmark input executed turn by turn;
8. per-turn diagnostics captured (planner queries, BM25/E5 tops,
   RRF/dedup/diversity survivors, Evidence Pack IDs/locators, exact
   source token/character counts, latency, loss diagnostics,
   oracle-region hit/miss);
9. protected per-case payload compressed + age-encrypted before upload;
10. compact non-sensitive public summary/manifest written.

Logs carry IDs, checksums, counts and latencies only. The age identity
is never printed and plaintext book text is never logged or uploaded.
Exit codes: 0 PASS, 1 FAIL, 2 INCOMPLETE, 3 STALE, 75 RUNNER_RESTART_REQUIRED.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.budget import estimate_text_tokens  # noqa: E402
from aa.corpus.public_cache import (  # noqa: E402
    default_lock_path,
    load_embedding_lock,
    resolve_hf_cache_dir,
    resolve_model_root,
    verify_cached_model,
)
from aa.opencode.errors import OpenCodeRateLimitError  # noqa: E402
from aa.qualification.real_book_retrieval import (  # noqa: E402
    BENCHMARK_INPUT_REL,
    BENCHMARK_ORACLE_REL,
    EXIT_BY_STATUS,
    RU_MANIFEST_REL,
    RealBookRetrievalError,
    TurnDiagnostics,
    assert_no_oracle_leak,
    assert_no_reranker,
    assert_source_exact,
    benchmark_binding_checksum,
    build_protected_payload,
    build_result_marker,
    corpus_checksum,
    decide_status,
    production_retrieval_checksum,
    require_production_backend,
    secret_present,
    summarize_public,
    validate_exact_sha,
)
from aa.retrieval.evidence import RetrievalConfig  # noqa: E402
from aa.retrieval.index import open_hybrid_index  # noqa: E402

DEFAULT_OUT = ROOT / "eval-real-book-out"
STATUS_FILENAME = "result.json"
SUMMARY_FILENAME = "real-book-retrieval-summary.json"
PROTECTED_FILENAME = "real-book-retrieval-protected.tar.zst.age"
PROTECTED_SHA_FILENAME = "real-book-retrieval-protected.tar.zst.age.sha256"
EXIT_RUNNER_RESTART_REQUIRED = 75
CHECKPOINT_FILENAME = "checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_TUPLE_FIELDS = (
    "planner_queries",
    "bm25_top",
    "e5_top",
    "rrf_survivors",
    "dedup_survivors",
    "diversity_survivors",
    "evidence_ids",
)


def _write_checkpoint(
    out_dir: Path,
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    processed_case_ids: list[str],
    diagnostics: list[TurnDiagnostics],
    infra_failures: int,
) -> None:
    """Persist restart-safe benchmark progress without source/book text."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "main_sha": main_sha,
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "processed_case_ids": processed_case_ids,
        "infra_failures": infra_failures,
        "diagnostics": [asdict(item) for item in diagnostics],
    }
    (out_dir / CHECKPOINT_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(
    out_dir: Path,
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
) -> tuple[list[str], list[TurnDiagnostics], int]:
    """Load a checkpoint only when every immutable qualification binding matches."""
    path = out_dir / CHECKPOINT_FILENAME
    if not path.is_file():
        return [], [], 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], [], 0
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "main_sha": main_sha,
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        print("resume checkpoint ignored: immutable qualification binding changed", flush=True)
        return [], [], 0
    processed_raw = payload.get("processed_case_ids", [])
    diagnostics_raw = payload.get("diagnostics", [])
    infra_raw = payload.get("infra_failures", 0)
    if not isinstance(processed_raw, list) or not isinstance(diagnostics_raw, list):
        return [], [], 0
    processed = [str(item) for item in processed_raw]
    diagnostics: list[TurnDiagnostics] = []
    try:
        for item in diagnostics_raw:
            if not isinstance(item, dict):
                raise ValueError("invalid diagnostic checkpoint entry")
            restored = dict(item)
            for field in _CHECKPOINT_TUPLE_FIELDS:
                restored[field] = tuple(restored.get(field, ()))
            diagnostics.append(TurnDiagnostics(**restored))
        infra_failures = int(infra_raw)
    except (TypeError, ValueError, KeyError):
        return [], [], 0
    print(
        f"resume checkpoint accepted: processed={len(processed)} diagnostics={len(diagnostics)}",
        flush=True,
    )
    return processed, diagnostics, infra_failures



def _fail_incomplete(out_dir: Path, *, reason: str, main_sha: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": "INCOMPLETE", "reason": reason, "main_sha": main_sha}
    (out_dir / STATUS_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"real-book qualification INCOMPLETE: {reason}", file=sys.stderr)
    return EXIT_BY_STATUS["INCOMPLETE"]


def _checked_out_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise RealBookRetrievalError("cannot determine checked-out SHA")
    return proc.stdout.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RealBookRetrievalError(f"{path.name} line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise RealBookRetrievalError(f"{path.name} line {lineno}: not an object")
        records.append(record)
    if not records:
        raise RealBookRetrievalError(f"{path.name}: file is empty")
    return records


def _oracle_by_case(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index evaluation-only oracle records by case id.

    Oracle data is used only to select book-required retrieval cases and to
    score canonical regions. It is never included in planner/model inputs.
    """
    indexed: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        rtype = record.get("type")
        if rtype == "single_turn":
            indexed[str(record["id"])] = dict(record)
        elif rtype == "multi_turn_journey":
            journey = str(record["id"])
            for entry in record.get("turns", []):
                if not isinstance(entry, dict):
                    continue
                indexed[f"{journey}#{entry['turn']}"] = dict(entry)
    return indexed


def _input_turns(
    records: list[dict[str, Any]],
    oracle_by_case: dict[str, dict[str, Any]],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Project Product Contract v1.2 to book-required retrieval turns.

    Journey context contains only prior generator-visible user utterances.
    A session_reset control clears that context. No oracle fields are copied
    into planner inputs.
    """
    turns: list[tuple[str, str, tuple[str, ...]]] = []
    for record in records[1:]:
        rtype = record.get("type")
        if rtype == "single_turn":
            assert_no_oracle_leak(record, str(record.get("id", "single")))
            case_id = str(record["id"])
            oracle = oracle_by_case.get(case_id, {})
            if oracle.get("book_content") == "required":
                turns.append((case_id, str(record["utterance"]), ()))
        elif rtype == "multi_turn_journey":
            assert_no_oracle_leak(record, str(record.get("id", "journey")))
            history: list[str] = []
            for entry in record.get("turns", []):
                if not isinstance(entry, dict):
                    raise RealBookRetrievalError("journey entry must be an object")
                assert_no_oracle_leak(entry, str(record.get("id", "journey")))
                if entry.get("kind") == "control":
                    if entry.get("control") == "session_reset":
                        history.clear()
                    continue
                if entry.get("kind") != "user":
                    continue
                case_id = f"{record['id']}#{entry['turn']}"
                utterance = str(entry["utterance"])
                oracle = oracle_by_case.get(case_id, {})
                if oracle.get("book_content") == "required":
                    turns.append((case_id, utterance, tuple(history)))
                history.append(utterance)
    return turns


def _oracle_regions(oracle_by_case: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    regions: dict[str, set[str]] = {}
    for case_id, record in oracle_by_case.items():
        provenances = record.get("provenance_ids", [])
        regions[case_id] = {str(item) for item in provenances if str(item)}
    return regions


async def _plan_with_production_adapter(
    utterance: str,
    *,
    primary_model: str,
    fallback_model: str,
    recent_user_turns: tuple[str, ...] = (),
) -> list[str]:
    """Invoke the real mandatory planner via the production adapter/profile."""
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
        client,
        agent=PLANNER_AGENT_V2,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )
    recent = [HumanMessage(content=item) for item in recent_user_turns]
    plan = await run_planner(utterance, model=model, recent=recent)
    _ = query_plan_json_schema()
    return list(plan.queries)


def _run_turn(
    *,
    index: Any,
    case_id: str,
    utterance: str,
    planner_queries: list[str],
    oracle_sections: set[str],
    config: RetrievalConfig,
) -> TurnDiagnostics:
    from aa.retrieval import evidence as evidence_mod

    started = time.perf_counter()
    ranked_lists, per_query_ids = evidence_mod.run_branch_searches(
        index, planner_queries, branch_top_k=config.branch_top_k
    )
    bm25_ids: list[str] = []
    e5_ids: list[str] = []
    for position, ranked in enumerate(ranked_lists):
        branch_ids = [chunk_id for chunk_id, _ in ranked[:10]]
        if position % 2 == 0:
            bm25_ids.extend(branch_ids)
        else:
            e5_ids.extend(branch_ids)
    fused, pool_ids = evidence_mod.fuse_query_pool(
        ranked_lists, per_query_ids, rrf_k=config.rrf_k, pool_cap=config.pool_cap
    )
    rrf_ids = list(pool_ids)
    diverse = evidence_mod.dedup_and_diversify(
        index,
        pool_ids,
        fused,
        pool_cap=config.pool_cap,
        max_per_section=config.max_per_section,
    )
    dedup_ids = [item.chunk_id for item in diverse]
    winners = evidence_mod.select_top_candidates(diverse, top_cap=config.top_child_cap)
    diversity_ids = [item.chunk_id for item in winners]
    expanded = evidence_mod.expand_small_to_big(
        index, winners, neighbor_window=config.neighbor_window
    )
    selected, _ = evidence_mod.select_passages_under_budget(
        expanded, budget_tokens=config.budget_tokens, index=index
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    evidence_ids: list[str] = []
    source_chars = 0
    for passage in selected:
        for chunk_id in passage.child_chunk_ids:
            record = index.chunks.get(chunk_id)
            if record is None:
                raise RealBookRetrievalError(f"selected chunk is not indexed: {chunk_id}")
            assert_source_exact(
                text=record.text,
                text_sha256=record.text_sha256,
                char_start=record.char_start,
                char_end=record.char_end,
            )
            evidence_ids.append(chunk_id)
            source_chars += len(record.text)
    source_tokens = sum(
        estimate_text_tokens(index.chunks[cid].text) for cid in evidence_ids if cid in index.chunks
    )

    def _sections(ids: list[str]) -> set[str]:
        out: set[str] = set()
        for chunk_id in ids:
            record = index.chunks.get(chunk_id)
            if record is not None:
                out.add(record.section)
        return out

    pack_sections = _sections(evidence_ids)
    oracle_hit: bool | None
    if not oracle_sections:
        oracle_hit = None
        branch_hit = rrf_hit = dedup_hit = budget_hit = False
    else:
        branch_sections = _sections([cid for contributed in per_query_ids for cid in contributed])
        branch_hit = bool(branch_sections & oracle_sections)
        rrf_hit = bool(_sections(rrf_ids) & oracle_sections)
        dedup_hit = bool(_sections(dedup_ids) & oracle_sections)
        budget_hit = bool(pack_sections & oracle_sections)
        oracle_hit = budget_hit
    budget_loss = bool(dedup_hit and not budget_hit)
    return TurnDiagnostics(
        case_id=case_id,
        planner_queries=tuple(planner_queries),
        bm25_top=tuple(list(dict.fromkeys(bm25_ids))[:20]),
        e5_top=tuple(list(dict.fromkeys(e5_ids))[:20]),
        rrf_survivors=tuple(rrf_ids[:20]),
        dedup_survivors=tuple(dedup_ids[:20]),
        diversity_survivors=tuple(diversity_ids[:16]),
        evidence_ids=tuple(evidence_ids),
        source_tokens=source_tokens,
        source_chars=source_chars,
        latency_ms=elapsed_ms,
        branch_union_hit=branch_hit if oracle_sections else False,
        rrf_hit=rrf_hit if oracle_sections else False,
        dedup_hit=dedup_hit if oracle_sections else False,
        budget_hit=budget_hit if oracle_sections else False,
        budget_loss=budget_loss,
        oracle_hit=oracle_hit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trusted real-book retrieval qualification runner."
    )
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--recipient", required=True, help="Age recipient for encryption")
    parser.add_argument("--limit", type=int, default=0, help="Optional turn cap (0 = all)")
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        expected_sha = validate_exact_sha(args.main_sha)
    except RealBookRetrievalError as exc:
        print(f"real-book qualification failed: {exc}", file=sys.stderr)
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
            }
            (out_dir / STATUS_FILENAME).write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            print("real-book qualification STALE: SHA mismatch", file=sys.stderr)
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
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    except OSError as exc:
        return _fail_incomplete(out_dir, reason=f"checksum failed: {exc}", main_sha=expected_sha)

    try:
        lock = load_embedding_lock(default_lock_path())
    except Exception as exc:
        return _fail_incomplete(
            out_dir, reason=f"embedding lock is invalid: {exc}", main_sha=expected_sha
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
        ru_manifest = ROOT / RU_MANIFEST_REL
        index = open_hybrid_index(index_dir, ru_manifest_path=ru_manifest)
    except Exception as exc:
        return _fail_incomplete(
            out_dir,
            reason=f"hybrid index is unavailable: {type(exc).__name__}",
            main_sha=expected_sha,
        )
    try:
        backend = str(index.metadata.get("embedding_backend", ""))
        require_production_backend(backend=backend)
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
        input_records = _load_jsonl(ROOT / BENCHMARK_INPUT_REL)
        oracle_records = _load_jsonl(ROOT / BENCHMARK_ORACLE_REL)
    except RealBookRetrievalError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)

    oracle_by_case = _oracle_by_case(oracle_records)
    turns = _input_turns(input_records, oracle_by_case)
    if args.limit and args.limit > 0:
        turns = turns[: args.limit]
    if not turns:
        return _fail_incomplete(
            out_dir, reason="benchmark has no book-required retrieval turns", main_sha=expected_sha
        )
    oracle_regions = _oracle_regions(oracle_by_case)

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
    processed_case_ids, diagnostics, infra_failures = _load_checkpoint(
        out_dir,
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
    )
    processed = set(processed_case_ids)
    try:
        for case_id, utterance, recent_user_turns in turns:
            if case_id in processed:
                continue
            if not utterance.strip():
                infra_failures += 1
                processed.add(case_id)
                processed_case_ids.append(case_id)
                _write_checkpoint(
                    out_dir,
                    main_sha=expected_sha,
                    corpus_sha=corpus_sha,
                    benchmark_sha=benchmark_sha,
                    retrieval_sha=retrieval_sha,
                    processed_case_ids=processed_case_ids,
                    diagnostics=diagnostics,
                    infra_failures=infra_failures,
                )
                continue
            try:
                queries = asyncio.run(
                    _plan_with_production_adapter(
                        utterance,
                        primary_model=primary_model,
                        fallback_model=fallback_model,
                        recent_user_turns=recent_user_turns,
                    )
                )
            except OpenCodeRateLimitError as exc:
                _write_checkpoint(
                    out_dir,
                    main_sha=expected_sha,
                    corpus_sha=corpus_sha,
                    benchmark_sha=benchmark_sha,
                    retrieval_sha=retrieval_sha,
                    processed_case_ids=processed_case_ids,
                    diagnostics=diagnostics,
                    infra_failures=infra_failures,
                )
                payload = {
                    "result": "INCOMPLETE",
                    "reason": "OpenCode 429 requires fresh runner recovery",
                    "reason_code": "OPENCODE_429_RESTART_REQUIRED",
                    "resume_after_case_count": len(processed_case_ids),
                    "main_sha": expected_sha,
                }
                (out_dir / STATUS_FILENAME).write_text(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "planner_error": type(exc).__name__,
                            "planner_error_detail": str(exc),
                            "runner_restart_required": True,
                        }
                    )
                )
                return EXIT_RUNNER_RESTART_REQUIRED
            except Exception as exc:
                error_payload = {
                    "case_id": case_id,
                    "planner_error": type(exc).__name__,
                    "utterance_chars": len(utterance),
                }
                # aa.opencode.errors messages are deliberately sanitized and
                # contain only status/retryability/session hints, never prompt
                # text or provider response bodies. Surface that safe detail
                # so trusted qualification can distinguish a bad request from
                # a provider/model rejection instead of reporting only a type.
                if exc.__class__.__module__ == "aa.opencode.errors":
                    error_payload["planner_error_detail"] = str(exc)
                print(json.dumps(error_payload))
                infra_failures += 1
                processed.add(case_id)
                processed_case_ids.append(case_id)
                _write_checkpoint(
                    out_dir,
                    main_sha=expected_sha,
                    corpus_sha=corpus_sha,
                    benchmark_sha=benchmark_sha,
                    retrieval_sha=retrieval_sha,
                    processed_case_ids=processed_case_ids,
                    diagnostics=diagnostics,
                    infra_failures=infra_failures,
                )
                continue
            if not queries:
                infra_failures += 1
                processed.add(case_id)
                processed_case_ids.append(case_id)
                _write_checkpoint(
                    out_dir,
                    main_sha=expected_sha,
                    corpus_sha=corpus_sha,
                    benchmark_sha=benchmark_sha,
                    retrieval_sha=retrieval_sha,
                    processed_case_ids=processed_case_ids,
                    diagnostics=diagnostics,
                    infra_failures=infra_failures,
                )
                continue
            assert_no_oracle_leak({"queries": queries}, case_id)
            try:
                diagnostics.append(
                    _run_turn(
                        index=index,
                        case_id=case_id,
                        utterance=utterance,
                        planner_queries=queries,
                        oracle_sections=oracle_regions.get(case_id, set()),
                        config=config,
                    )
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "retrieval_error": type(exc).__name__,
                            "utterance_chars": len(utterance),
                        }
                    )
                )
                infra_failures += 1
                processed.add(case_id)
                processed_case_ids.append(case_id)
                _write_checkpoint(
                    out_dir,
                    main_sha=expected_sha,
                    corpus_sha=corpus_sha,
                    benchmark_sha=benchmark_sha,
                    retrieval_sha=retrieval_sha,
                    processed_case_ids=processed_case_ids,
                    diagnostics=diagnostics,
                    infra_failures=infra_failures,
                )
                continue
            processed.add(case_id)
            processed_case_ids.append(case_id)
            _write_checkpoint(
                out_dir,
                main_sha=expected_sha,
                corpus_sha=corpus_sha,
                benchmark_sha=benchmark_sha,
                retrieval_sha=retrieval_sha,
                processed_case_ids=processed_case_ids,
                diagnostics=diagnostics,
                infra_failures=infra_failures,
            )
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "planner_queries": len(queries),
                        "evidence_ids": len(diagnostics[-1].evidence_ids),
                        "source_tokens": diagnostics[-1].source_tokens,
                        "source_chars": diagnostics[-1].source_chars,
                        "latency_ms": round(diagnostics[-1].latency_ms, 1),
                        "oracle_hit": diagnostics[-1].oracle_hit,
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
        status = decide_status(stale=False, incomplete=False, failures=misses)

    from aa.qualification.real_book_retrieval import compress_and_encrypt

    protected = build_protected_payload(
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

    summary = summarize_public(
        main_sha=expected_sha,
        corpus_sha=corpus_sha,
        benchmark_sha=benchmark_sha,
        retrieval_sha=retrieval_sha,
        turns=diagnostics,
        status=status,
        run_id=str(run_id),
    )
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = build_result_marker(
        sha=expected_sha,
        corpus=corpus_sha,
        benchmark=benchmark_sha,
        retrieval=retrieval_sha,
        result=status,
        run=str(run_id),
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
                "marker": marker,
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
                "turns": len(diagnostics),
                "infra_failures": infra_failures,
                "corpus_sha256": corpus_sha[:16],
                "benchmark_sha256": benchmark_sha[:16],
                "retrieval_sha256": retrieval_sha[:16],
            }
        )
    )
    return EXIT_BY_STATUS[status]


if __name__ == "__main__":
    raise SystemExit(main())
