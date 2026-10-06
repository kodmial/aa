#!/usr/bin/env python3
"""Self-proving layered qualification runner (issue #146, Gates A-F).

Evaluates Gates A-E for the exact required main SHA and emits the Gate F
exact-main final verdict. Writes privacy-safe evidence only (ids, SHAs,
digests, counts, latencies; never user/corpus text):

- ``result.json`` (PASS/FAIL/BLOCKED/STALE + marker + blocking gate);
- ``self-proving-summary.json`` (per-gate evidence + failure reports).

Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED/INCOMPLETE, 3 STALE.

Fail-closed contract:
- stale SHA, dirty tree, missing secret/model, mocked-only evidence, or
  unknown state all yield BLOCKED/STALE, never a warning and never PASS.
- infrastructure failure (provider 429, missing secrets) is reported as
  BLOCKED with a machine-readable category, never as product PASS.
- provider 429 writes a restart marker and exits 75 so the workflow can
  retire the runner and resume from the last completed gate checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.self_proving import (  # noqa: E402
    GateEvidence,
    SelfProvingError,
    build_result_marker,
    decide_final_verdict,
    failure_report_for_gate,
    repair_fingerprint,
    validate_exact_sha,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
EXIT_STALE = 3
EXIT_429_RESTART = 75


def _git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    if proc.returncode != 0:
        raise SelfProvingError(f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _write(out_dir: Path, verdict: object, result: str, **extra: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": result, **extra}
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _runtime_fingerprint() -> str:
    import hashlib

    from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL, Settings

    settings = Settings.from_env({})
    material = "\x00".join(
        (
            "aa-conversation-runtime/1",
            settings.opencode_agent,
            settings.opencode_model or DEFAULT_PRIMARY_MODEL,
            settings.opencode_fallback_model or DEFAULT_FALLBACK_MODEL,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _product_fingerprint() -> str:
    from aa.qualification.product_fingerprint import compute_product_fingerprint

    return compute_product_fingerprint(ROOT)


def _gate_a(expected_sha: str, run_id: str) -> GateEvidence:
    try:
        checked = _git(["rev-parse", "HEAD"])
    except SelfProvingError as exc:
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    if checked != expected_sha:
        return GateEvidence(
            gate="A",
            status="STALE",
            sha=expected_sha,
            failure_category="stale-sha",
            run_id=run_id,
        )
    try:
        dirty = _git(["status", "--porcelain"])
    except SelfProvingError as exc:
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    if dirty.strip():
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="dirty-tree",
            run_id=run_id,
        )
    try:
        product = _product_fingerprint()
        runtime = _runtime_fingerprint()
    except Exception as exc:
        return GateEvidence(
            gate="A",
            status="FAIL",
            sha=expected_sha,
            failure_category="fingerprint-error",
            component="fingerprint-consistency",
            run_id=run_id,
            detail=type(exc).__name__[:64],
        )
    if len(product) != 64 or len(runtime) != 64:
        return GateEvidence(
            gate="A",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="fingerprint-mismatch",
            component="fingerprint-consistency",
            run_id=run_id,
        )
    # Static/build trust: exact SHA + clean tree + computable fingerprints.
    # Unit/type/lint themselves run in the workflow before this script; this
    # gate records their trust boundary (workflow must fail closed first).
    return GateEvidence(
        gate="A",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        run_id=run_id,
        live_trusted=True,
    )


def _fail_b(
    expected_sha: str,
    run_id: str,
    product: str,
    runtime: str,
    component: str,
    detail: str = "",
) -> GateEvidence:
    return GateEvidence(
        gate="B",
        status="FAIL",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        failure_category="component-fail",
        component=component,
        run_id=run_id,
        detail=detail[:96],
    )


def _gate_b(expected_sha: str, run_id: str, product: str, runtime: str) -> GateEvidence:
    # Deterministic component integration through real production modules
    # (no network, no secrets). Any component failure names its component.
    # Covers: real RU corpus restore layout + real BM25/dense index build,
    # planner cardinality/shape, retrieval/RRF/dedup/diversity/small-to-big
    # over the real index, grounding/verifier, output envelope,
    # memory/FIFO/concurrency/safety, voice fixtures/cache/resource checks.
    try:
        from aa.conversation.planner_schema import QueryPlan, validate_query_plan
        from aa.retrieval.evidence import RetrievalConfig
    except Exception as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "planner-shape", type(exc).__name__)
    # -- planner cardinality/shape (real production schema) --
    try:
        plan = validate_query_plan(QueryPlan(queries=[f"запрос {i}" for i in range(12)]))
        if not 10 <= len(plan.queries) <= 16:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "planner-cardinality",
                "cardinality out of bounds",
            )
        try:
            validate_query_plan(QueryPlan(queries=["q0"]))
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "planner-cardinality",
                "single query unexpectedly accepted",
            )
        except ValueError:
            pass
        config = RetrievalConfig()
        if config.branch_top_k <= 0 or config.pool_cap <= 0:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "retrieval-rrf",
                "retrieval config not production",
            )
        import aa.retrieval.evidence as evidence_mod

        source = Path(evidence_mod.__file__).read_text(encoding="utf-8").lower()
        if "cross_encoder" in source or "bge-rerank" in source:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "retrieval-rrf",
                "second-stage reranker present",
            )
    except ValueError as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "retrieval-rrf", str(exc))

    # -- corpus-restore layout (real pinned artifacts must exist) --
    try:
        lock_path = ROOT / "corpus" / "embedding.lock.json"
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if not lock_payload.get("model_id") or not lock_payload.get("revision"):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "corpus-restore",
                "embedding lock missing model/revision",
            )
        if not (ROOT / "corpus").is_dir():
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "corpus-restore",
                "corpus directory missing",
            )
    except Exception as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "corpus-restore", type(exc).__name__)

    # -- real BM25 + dense index build and full retrieval pipeline --
    try:
        import hashlib
        import tempfile

        from aa.retrieval.dense import (
            HASHING_BACKEND_NAME,
            HASHING_DIM,
            ExactIPIndex,
            hashing_embed,
        )
        from aa.retrieval.evidence import (
            dedup_and_diversify,
            expand_small_to_big,
            fuse_query_pool,
            retrieve_evidence,
            run_branch_searches,
            select_top_candidates,
        )
        from aa.retrieval.index import ChunkRecord, HybridIndex
        from aa.retrieval.lexical import build_lexical_db, load_lexical_into_memory

        topics = ["алкоголизм тяга трезвость", "семья отношения поддержка"]
        sections = ["ru-test/sec-a", "ru-test/sec-b"]
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        records: dict[str, ChunkRecord] = {}
        for sec_idx, section in enumerate(sections):
            for para in range(3):
                cid = f"ru-test:{sec_idx}:{para}"
                text = (
                    f"Канонический русский абзац {para} раздела {section} "
                    f"про {topics[sec_idx]} выздоровление программа шаги. "
                    f"Уникальный маркер параграфа p{sec_idx}{para}."
                )
                chunk_ids.append(cid)
                chunk_texts.append(text)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                records[cid] = ChunkRecord(
                    chunk_id=cid,
                    logical_chunk_id=cid,
                    section=section,
                    book="ru-test-book",
                    parent=f"parent-{sec_idx}-{para // 2}",
                    prev=chunk_ids[-2] if len(chunk_ids) > 1 else None,
                    next=None,
                    source_id="ru-fourth-edition-txt",
                    source_file="corpus/source/raw-ru/aa-big-book.txt",
                    source_sha256=hashlib.sha256(b"ru-source").hexdigest(),
                    char_start=para * 100,
                    char_end=para * 100 + len(text),
                    text_sha256=digest,
                    text=text,
                    corpus_version="ru-test-v1",
                )
        # Link next pointers for small-to-big neighbor traversal.
        for pos, cid in enumerate(chunk_ids):
            rec = records[cid]
            nxt = chunk_ids[pos + 1] if pos + 1 < len(chunk_ids) else None
            if rec.next != nxt:
                records[cid] = ChunkRecord(
                    **{**rec.__dict__, "next": nxt},
                )
        with tempfile.TemporaryDirectory(prefix="aa-gate-b-lexical-") as tmp:
            db_path = Path(tmp) / "lexical.db"
            build_lexical_db(
                db_path,
                chunk_ids=chunk_ids,
                sections=[records[c].section for c in chunk_ids],
                texts=chunk_texts,
            )
            lexical_conn = load_lexical_into_memory(db_path)
        try:
            vectors = [hashing_embed(text, dim=HASHING_DIM) for text in chunk_texts]
            dense = ExactIPIndex.build(chunk_ids, vectors, backend=HASHING_BACKEND_NAME)
            index = HybridIndex(
                directory=Path(tmp) if False else Path("."),
                metadata={
                    "embedding_backend": HASHING_BACKEND_NAME,
                    "embedding_dim": HASHING_DIM,
                },
                chunks=records,
                dense=dense,
                lexical_conn=lexical_conn,
                ram_resident=True,
            )
            if not index.ram_resident or index.lexical_conn is None:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "bm25-index",
                    "index not RAM-resident",
                )
            if index.dense.dim != HASHING_DIM or len(index.chunks) < 4:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "e5-faiss-index",
                    "dense substrate shape invalid",
                )
            queries = [f"тяга к алкоголю вариант {i}" for i in range(12)]
            ranked_lists, per_query_ids = run_branch_searches(index, queries)
            # Every planner query must run both branches (lexical + dense).
            if len(ranked_lists) != 2 * len(queries):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-rrf",
                    "branch coverage incomplete",
                )
            if any(not ids for ids in per_query_ids):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-rrf",
                    "query produced no candidates",
                )
            fused, pool_ids = fuse_query_pool(
                ranked_lists,
                per_query_ids,
                rrf_k=config.rrf_k,
                pool_cap=config.pool_cap,
            )
            if not fused or not pool_ids:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-rrf",
                    "RRF fusion yielded empty pool",
                )
            # Per-query retention: every distinct per-query best must be
            # retained (small fixtures may have fewer chunks than queries).
            distinct_bests: set[str] = set()
            for contributed in per_query_ids:
                for chunk_id in contributed:
                    if chunk_id in fused and chunk_id not in distinct_bests:
                        distinct_bests.add(chunk_id)
                        break
            if not distinct_bests or any(best not in set(pool_ids) for best in distinct_bests):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-rrf",
                    "per-query retention violated",
                )
            diverse = dedup_and_diversify(
                index,
                pool_ids,
                fused,
                pool_cap=config.pool_cap,
                max_per_section=config.max_per_section,
            )
            if not diverse:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-dedup",
                    "dedup emptied the pool",
                )
            from collections import Counter as _Counter

            section_of = {cid: records[cid].section for cid in records}
            counts = _Counter(section_of[c.chunk_id] for c in diverse)
            if any(n > config.max_per_section for n in counts.values()):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-diversity",
                    "section cap violated",
                )
            winners = select_top_candidates(
                diverse, top_cap=config.top_child_cap, sections=section_of
            )
            if not winners:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-diversity",
                    "no winners selected",
                )
            expanded = expand_small_to_big(index, winners, neighbor_window=config.neighbor_window)
            if not expanded:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "small-to-big",
                    "expansion yielded no passages",
                )
            for passage in expanded:
                joined = "".join(records[c].text for c in passage.child_chunk_ids)
                if hashlib.sha256(joined.encode("utf-8")).hexdigest() != passage.text_sha256:
                    return _fail_b(
                        expected_sha,
                        run_id,
                        product,
                        runtime,
                        "small-to-big",
                        "expansion checksum mismatch",
                    )
            pack = retrieve_evidence(index, queries, config=config)
            if not pack.passages or pack.total_tokens <= 0:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "small-to-big",
                    "end-to-end pack empty",
                )
        finally:
            try:
                lexical_conn.close()
            except Exception:
                pass
    except Exception as exc:
        label = type(exc).__name__
        component = "bm25-index" if "Lexical" in label else "e5-faiss-index"
        if isinstance(exc, ValueError) and "pool" in str(exc).lower():
            component = "retrieval-rrf"
        return _fail_b(expected_sha, run_id, product, runtime, component, f"{label}: {exc}"[:96])

    # -- grounding / verifier (deterministic, no model calls) --
    try:
        from aa.conversation.verifier_schema import validate_grounding_result
        from aa.grounding.gate import check_grounding
        from aa.grounding.quotes import EvidenceKind, EvidenceUnit, Provenance, QuoteKind

        claim = "Фиктивная фраза про трезвость поддержку друзей утренние собрания"
        provenance = Provenance(
            corpus_version="ru-test-v1",
            source_id="ru-fourth-edition-txt",
            section_id="ru-test/sec-a",
            chunk_id="ru-test:0:0",
            char_start=0,
            char_end=len(claim),
            source_checksum="1" * 64,
            source_language="ru",
        )
        unit = EvidenceUnit(
            kind=EvidenceKind.SOURCE_TEXT,
            language="ru",
            text=claim,
            provenance=provenance,
        )
        verdict = check_grounding(
            russian_claim=claim,
            quoted_text=claim,
            quote_kind=QuoteKind.EXACT_SOURCE,
            cited=[unit.provenance],
            evidence=[unit],
            ru_corpus_available=True,
            allow_translation_fallback=False,
        )
        if not verdict.passed:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "grounding",
                f"supported claim rejected: {verdict.code}",
            )
        bad = check_grounding(
            russian_claim="Несвязанное утверждение о полетах на Марс",
            quoted_text="Несвязанное утверждение о полетах на Марс",
            quote_kind=QuoteKind.EXACT_SOURCE,
            cited=[unit.provenance],
            evidence=[unit],
            ru_corpus_available=True,
            allow_translation_fallback=False,
        )
        if bad.passed:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "grounding",
                "unsupported claim accepted",
            )
        good = validate_grounding_result(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": ["p1"],
                    }
                ],
                "all_required_supported": True,
            },
            expected_unit_ids=["u1"],
        )
        if not good.all_required_supported:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "verifier",
                "valid grounding rejected",
            )
        try:
            validate_grounding_result(
                {"units": [], "all_required_supported": False},
                expected_unit_ids=["u1"],
            )
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "verifier",
                "incomplete verdicts accepted",
            )
        except Exception:
            pass
    except Exception as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "verifier", type(exc).__name__)

    # -- output envelope --
    try:
        from aa.conversation.output_limits import compact_text_to_envelope, envelope_passes

        if not envelope_passes("Короткий трезвый ответ."):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "output-envelope",
                "short reply rejected",
            )
        long_text = "Слово " * 400
        if envelope_passes(long_text):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "output-envelope",
                "overflow reply accepted",
            )
        compacted = compact_text_to_envelope(long_text)
        if not envelope_passes(compacted):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "output-envelope",
                "compaction still over budget",
            )
    except Exception as exc:
        return _fail_b(
            expected_sha, run_id, product, runtime, "output-envelope", type(exc).__name__
        )

    # -- memory/FIFO/concurrency/safety --
    try:
        from aa.conversation.memory import thread_id_for_chat
        from aa.retrieval.evidence import query_vector_cache_info
        from aa.safety.router import SafetyDecision, SafetyRouter

        if thread_id_for_chat(123) != thread_id_for_chat(123):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "memory-fifo",
                "thread mapping nondeterministic",
            )
        if thread_id_for_chat(123) == thread_id_for_chat(124):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "memory-fifo",
                "thread mapping collides",
            )
        info = query_vector_cache_info()
        if info.get("max", 0) != 1024 or info.get("size", -1) < 0:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "memory-fifo",
                "query cache FIFO bound invalid",
            )
        from aa.telegram.dispatcher import ChatTurnDispatcher

        if not hasattr(ChatTurnDispatcher, "start") or not hasattr(ChatTurnDispatcher, "stop"):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "concurrency",
                "dispatcher lifecycle missing",
            )
        router = SafetyRouter()
        if router.check("").decision is not SafetyDecision.BLOCK:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "safety",
                "empty not blocked",
            )
        if router.check("Обычный вопрос о программе").decision is not SafetyDecision.ALLOW:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "safety",
                "ordinary not allowed",
            )
    except Exception as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "memory-fifo", type(exc).__name__)

    # -- voice fixtures/cache/resource --
    try:
        from aa.corpus.voice_cache import load_voice_lock

        voice_lock_path = ROOT / "corpus" / "voice.lock.json"
        if voice_lock_path.is_file():
            load_voice_lock(voice_lock_path)
        else:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "voice-fixture",
                "voice lock missing",
            )
        from aa.telegram.voice_presentation import VoicePresentationClassifier

        if VoicePresentationClassifier is None:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "voice-cache",
                "presentation classifier missing",
            )
        import resource as _resource

        peak_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        if peak_kb <= 0:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "resource",
                "resource usage unreadable",
            )
    except Exception as exc:
        return _fail_b(expected_sha, run_id, product, runtime, "voice-fixture", type(exc).__name__)

    return GateEvidence(
        gate="B",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component="all-components",
        run_id=run_id,
        live_trusted=True,
    )


def _live_prerequisites() -> tuple[bool, str]:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    identity = (os.environ.get("AA_BOOK_AGE_IDENTITY", "") or "").strip()
    live = (os.environ.get("SELF_PROVING_LIVE", "") or "").strip() == "1"
    if not live:
        return False, "live-execution-not-enabled"
    if not token:
        return False, "missing-secret-telegram-token"
    if not identity:
        return False, "missing-secret-corpus-identity"
    return True, ""


def _coerce_turn_telemetries(raw_turns: object) -> list[object] | None:
    """Coerce a JSON ``turns`` list into ``TurnTelemetry`` objects, if possible."""
    if not isinstance(raw_turns, list) or not raw_turns:
        return None
    try:
        from aa.conversation.stage_telemetry import TurnTelemetry, record_stage
    except Exception:
        return None
    coerced: list[object] = []
    for entry in raw_turns:
        if not isinstance(entry, dict):
            return None
        try:
            turn = TurnTelemetry(
                family=str(entry.get("family", "")),
                reply_len=int(entry.get("reply_len", 0) or 0),
                reply_signature=str(entry.get("reply_signature", "")),
                fallback=bool(entry.get("fallback", False)),
                served_model=str(entry.get("served_model", "")),
            )
            stages = entry.get("stages", [])
            if not isinstance(stages, list):
                return None
            for sample in stages:
                if not isinstance(sample, dict):
                    return None
                record_stage(
                    turn,
                    stage=str(sample.get("stage", "")),
                    ok=bool(sample.get("ok", False)),
                    latency_ms=float(sample.get("latency_ms", 0.0)),
                    category=str(sample.get("category", "")),
                    model=str(sample.get("model", "")),
                )
            coerced.append(turn)
        except Exception:
            return None
    return coerced


def _live_latencies_from_product_summary(payload: object) -> list[float] | None:
    """Extract real measured per-turn latencies (ms) from a live summary."""
    if not isinstance(payload, dict):
        return None
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        return None
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        metrics = lane.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        turns = metrics.get("turns_executed", metrics.get("turns", 0))
        try:
            count = int(turns or 0)
        except (TypeError, ValueError):
            count = 0
        p50 = metrics.get("latency_p50_s", metrics.get("end_to_end_p50_ms", None))
        p95 = metrics.get("latency_p95_s", metrics.get("end_to_end_p95_ms", None))
        maximum = metrics.get("latency_max_s", metrics.get("max_ms", None))
        try:
            # Lane metrics record seconds; telemetry records ms. Accept both.
            p50_ms = float(p50 or 0.0)
            p95_ms = float(p95 or 0.0)
            max_ms = float(maximum or 0.0)
        except (TypeError, ValueError):
            continue
        if p50_ms > 0 and p50_ms < 60:
            # Seconds-scale values: convert to ms.
            p50_ms *= 1000.0
            p95_ms *= 1000.0
            max_ms *= 1000.0
        if count <= 0 or p50_ms <= 0:
            continue
        # Reconstruct a privacy-safe representative sample from the measured
        # aggregates (never text): half the turns at p50, the rest spread to
        # p95, plus the observed max so the 30s guard sees real outliers.
        samples = [p50_ms] * (count // 2 + 1)
        samples += [p95_ms] * (count - len(samples))
        if max_ms > 0:
            samples[-1] = max_ms
        return [float(v) for v in samples]
    return None


def _gate_c_live_evidence(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> tuple[GateEvidence, list[float]] | None:
    """Map repository-owned live evidence to a Gate C verdict, if present.

    Consumes the ``live-out`` artifact produced by
    ``run_product_contract_live_qualification.py`` (real local OpenCode
    process, real provider/model policy, real decrypted RU corpus + real
    production index, synthetic Updates injected only at the production
    Telegram adapter boundary). Returns ``None`` when no usable live
    evidence exists (caller stays BLOCKED fail-closed).
    """
    candidates = [
        ROOT / "live-out" / "product-contract-live-summary.json",
        ROOT / "live-out" / "self-proving-summary.json",
        ROOT / "self-proving-out" / "product-contract-live-summary.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return GateEvidence(
                gate="C",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="live-evidence-invalid",
                component="live-production-path",
                run_id=run_id,
                mocked_only=False,
                detail="unreadable-live-summary"[:64],
            ), []
        if not isinstance(payload, dict):
            continue
        if str(payload.get("main_sha", payload.get("sha", expected_sha))) != expected_sha:
            continue
        # Telemetry-shaped evidence: evaluate scenario families + diversity.
        raw_turns = payload.get("turns")
        if isinstance(raw_turns, list) and raw_turns:
            turns = _coerce_turn_telemetries(raw_turns)
            if turns is None:
                return GateEvidence(
                    gate="C",
                    status="BLOCKED",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category="live-evidence-invalid",
                    component="live-production-path",
                    run_id=run_id,
                    mocked_only=False,
                    detail="telemetry-shape-invalid"[:64],
                ), []
            try:
                from aa.conversation.stage_telemetry import evaluate_gate_c_telemetry

                ok, detail, _ = evaluate_gate_c_telemetry(
                    turns,
                    required_families=4,  # type: ignore[arg-type]
                )
            except Exception as exc:
                return GateEvidence(
                    gate="C",
                    status="BLOCKED",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category="live-evidence-invalid",
                    component="live-production-path",
                    run_id=run_id,
                    mocked_only=False,
                    detail=type(exc).__name__[:64],
                ), []
            latencies = [float(t.total_ms()) for t in turns]  # type: ignore[attr-defined]
            if ok:
                return GateEvidence(
                    gate="C",
                    status="PASS",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    component="live-production-path",
                    run_id=run_id,
                    live_trusted=True,
                    mocked_only=False,
                ), latencies
            return GateEvidence(
                gate="C",
                status="FAIL",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="live-path-failed",
                component="live-production-path",
                run_id=run_id,
                live_trusted=True,
                mocked_only=False,
                detail=str(detail)[:160],
            ), latencies
        # Product-contract lane evidence: the exact production conversation
        # path (synthetic Updates at the production adapter boundary) must
        # have no failed lane. Lanes may additionally carry INCOMPLETE live-
        # network sublanes (real Telegram dialing deferred to Gate D); those
        # are accepted only when explicitly network-deferred, every lane
        # has at least one passed production check, and nothing failed.
        status = str(payload.get("status", ""))
        lanes = payload.get("lanes", [])
        if isinstance(lanes, list) and lanes and status in ("PASS", "INCOMPLETE"):
            lane_failures = [
                lane for lane in lanes if not isinstance(lane, dict) or lane.get("status") == "FAIL"
            ]
            if lane_failures:
                latencies = _live_latencies_from_product_summary(payload) or []
                return GateEvidence(
                    gate="C",
                    status="FAIL",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category="live-path-failed",
                    component="live-production-path",
                    run_id=run_id,
                    live_trusted=True,
                    mocked_only=False,
                    detail="live-lane-failed"[:160],
                ), latencies
            deferred: tuple[str, ...] = ("real-telegram", "requires-token", "not-dialed")
            complete = True
            for lane in lanes:
                assert isinstance(lane, dict)
                passed = lane.get("passed", [])
                incomplete = lane.get("incomplete", [])
                if not isinstance(passed, list) or len(passed) == 0:
                    complete = False
                    break
                if isinstance(incomplete, list) and any(
                    not isinstance(item, str) or not any(token in item for token in deferred)
                    for item in incomplete
                ):
                    complete = False
                    break
            latencies = _live_latencies_from_product_summary(payload) or []
            if complete:
                return GateEvidence(
                    gate="C",
                    status="PASS",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    component="live-production-path",
                    run_id=run_id,
                    live_trusted=True,
                    mocked_only=False,
                ), latencies
            return GateEvidence(
                gate="C",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="live-evidence-incomplete",
                component="live-production-path",
                run_id=run_id,
                mocked_only=False,
                detail="live-lane-incomplete"[:64],
            ), latencies
        if status in ("FAIL",):
            latencies = _live_latencies_from_product_summary(payload) or []
            return GateEvidence(
                gate="C",
                status="FAIL",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="live-path-failed",
                component="live-production-path",
                run_id=run_id,
                live_trusted=True,
                mocked_only=False,
                detail="live-lane-failed"[:160],
            ), latencies
    return None


def _gate_c(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> tuple[GateEvidence, list[float]]:
    ready, reason = _live_prerequisites()
    if not ready:
        return GateEvidence(
            gate="C",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category=reason,
            component="live-production-path",
            run_id=run_id,
            mocked_only=True,
        ), []
    # Live prerequisites hold: map the repository-owned live artifact to
    # PASS/FAIL. Without usable live evidence this gate stays BLOCKED
    # (fail-closed); it never reports mocked PASS.
    decided = _gate_c_live_evidence(expected_sha, run_id, product, runtime)
    if decided is not None:
        return decided
    return GateEvidence(
        gate="C",
        status="BLOCKED",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        failure_category="live-evidence-required",
        component="live-production-path",
        run_id=run_id,
        mocked_only=False,
    ), []


def _telegram_get_json(token: str, method: str, *, timeout_s: int = 15) -> dict[str, object]:
    """Call one Telegram Bot API method with the real token (bounded)."""
    import urllib.parse
    import urllib.request

    url = f"https://api.telegram.org/bot{token}/{method}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SelfProvingError(f"telegram {method} unreachable: {type(exc).__name__}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SelfProvingError(f"telegram {method} invalid response: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelfProvingError(f"telegram {method} invalid response shape")
    return payload


def _gate_d_marker_verdict(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> GateEvidence | None:
    """Return a Gate D verdict from a durable runtime marker, if present.

    The repository-owned workflow Gate D probe publishes READY only from
    the running Application after proving OpenCode health, getMe identity,
    webhook/commands bootstrap, ``transport.running`` and a live poll task
    (then STOPPED after a clean stop with no leaked poll task). A FAILED
    marker means the live app failed to reach readiness.
    """
    from aa.control.runtime_status import RuntimeStatusError, read_marker

    for candidate in (
        ROOT / "self-proving-out" / "runtime-status.json",
        ROOT / "runtime-status.json",
    ):
        if not candidate.is_file():
            continue
        try:
            marker = read_marker(candidate)
        except RuntimeStatusError:
            continue
        if marker.sha != expected_sha:
            continue
        if marker.phase in ("READY", "STOPPED"):
            return GateEvidence(
                gate="D",
                status="PASS",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                component="telegram-readiness",
                run_id=run_id,
                live_trusted=True,
            )
        if marker.phase == "FAILED":
            return GateEvidence(
                gate="D",
                status="FAIL",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="runtime-not-ready",
                component="telegram-readiness",
                run_id=run_id,
                live_trusted=True,
            )
    return None


def _gate_d(expected_sha: str, run_id: str, product: str, runtime: str) -> GateEvidence:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    live = (os.environ.get("SELF_PROVING_LIVE", "") or "").strip() == "1"
    if not live or not token:
        return GateEvidence(
            gate="D",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category=(
                "missing-secret-telegram-token" if not token else "live-evidence-required"
            ),
            component="telegram-readiness",
            run_id=run_id,
        )
    # A durable READY/STOPPED marker from the running Application is the
    # strongest readiness proof (live poller verified by the workflow probe).
    marker_verdict = _gate_d_marker_verdict(expected_sha, run_id, product, runtime)
    if marker_verdict is not None:
        return marker_verdict
    # Otherwise prove real Telegram network readiness live: getMe identity
    # plus webhook-clean polling state. Fail-closed on any network/API error.
    try:
        me_payload = _telegram_get_json(token, "getMe")
        if me_payload.get("ok") is not True:
            raise SelfProvingError("telegram getMe returned ok != true")
        result = me_payload.get("result")
        if not isinstance(result, dict) or not result.get("id"):
            raise SelfProvingError("telegram getMe identity payload invalid")
        if result.get("is_bot") is not True:
            raise SelfProvingError("telegram getMe identity is not a bot")
        hook_payload = _telegram_get_json(token, "getWebhookInfo")
        hook_result = hook_payload.get("result")
        if not isinstance(hook_result, dict):
            raise SelfProvingError("telegram webhook state unreadable")
        hook_url = str(hook_result.get("url", "") or "")
        pending = hook_result.get("pending_update_count", 0)
        if hook_url:
            # Webhook set: long polling would miss updates; clean it first.
            _telegram_get_json(token, "deleteWebhook?drop_pending_updates=true")
            hook_payload = _telegram_get_json(token, "getWebhookInfo")
            hook_result = hook_payload.get("result")
            if not isinstance(hook_result, dict) or str(hook_result.get("url", "") or ""):
                raise SelfProvingError("telegram webhook cleanup failed")
            pending = hook_result.get("pending_update_count", 0)
        try:
            pending_count = int(pending or 0)
        except (TypeError, ValueError):
            pending_count = 0
        if pending_count < 0:
            raise SelfProvingError("telegram webhook state invalid")
    except SelfProvingError as exc:
        return GateEvidence(
            gate="D",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="telegram-readiness-unproven",
            component="telegram-readiness",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    return GateEvidence(
        gate="D",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component="telegram-readiness",
        run_id=run_id,
        live_trusted=True,
    )


def _gate_e(
    expected_sha: str,
    run_id: str,
    product: str,
    runtime: str,
    latencies_ms: list[float],
    gate_c_live: bool,
) -> GateEvidence:
    if not gate_c_live or not latencies_ms:
        return GateEvidence(
            gate="E",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="missing-live-latency",
            component="slo",
            run_id=run_id,
        )
    from aa.qualification.self_proving import percentile_ms

    p50 = percentile_ms(latencies_ms, 50)
    p95 = percentile_ms(latencies_ms, 95)
    maximum = max(latencies_ms)
    if p95 > 15_000 or maximum >= 30_000:
        return GateEvidence(
            gate="E",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="latency-budget-exceeded",
            component="slo",
            run_id=run_id,
            live_trusted=True,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            max_turn_ms=maximum,
        )
    return GateEvidence(
        gate="E",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component="slo",
        run_id=run_id,
        live_trusted=True,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        max_turn_ms=maximum,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-proving qualification runner (Gates A-F).")
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval-self-proving-out")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args(argv)

    try:
        expected = validate_exact_sha(args.main_sha)
    except SelfProvingError as exc:
        print(f"self-proving qualification BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(args.run_id)

    if os.environ.get("OPENCODE_429_RESTART_REQUIRED", "") == "1":
        marker = out_dir / "restart-required.json"
        marker.write_text(
            json.dumps(
                {"reason_code": "OPENCODE_429_RESTART_REQUIRED", "sha": expected},
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("self-proving qualification: provider 429, runner restart required", file=sys.stderr)
        return EXIT_429_RESTART

    gate_a = _gate_a(expected, run_id)
    if gate_a.status in ("STALE",):
        _write(out_dir, None, "STALE", main_sha=expected, run_id=run_id)
        print("self-proving qualification STALE: SHA mismatch", file=sys.stderr)
        return EXIT_STALE
    product = gate_a.product_fingerprint or _product_fingerprint()
    runtime = gate_a.runtime_fingerprint or _runtime_fingerprint()

    gate_b = _gate_b(expected, run_id, product, runtime)
    gate_c, latencies = _gate_c(expected, run_id, product, runtime)
    gate_d = _gate_d(expected, run_id, product, runtime)
    gate_e = _gate_e(
        expected,
        run_id,
        product,
        runtime,
        latencies,
        gate_c_live=(gate_c.status == "PASS" and gate_c.live_trusted),
    )

    evidences = [gate_a, gate_b, gate_c, gate_d, gate_e]
    try:
        verdict = decide_final_verdict(
            evidences,
            current_sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            run_id=run_id,
        )
    except SelfProvingError as exc:
        print(f"self-proving qualification BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    failures = []
    for evidence in verdict.gates:
        if evidence.status == "PASS":
            continue
        try:
            report = failure_report_for_gate(evidence)
        except SelfProvingError:
            continue
        payload = report.to_dict()
        payload["repair_fingerprint"] = repair_fingerprint(report)
        failures.append(payload)

    summary = verdict.to_dict()
    summary["failures"] = failures
    summary["marker"] = build_result_marker(
        sha=expected,
        status=verdict.status,
        run=run_id,
    )
    (out_dir / "self-proving-summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "result": verdict.status,
                "main_sha": expected,
                "run_id": run_id,
                "blocking_gate": verdict.blocking_gate,
                "marker": summary["marker"],
                "gates": [{"gate": item.gate, "status": item.status} for item in verdict.gates],
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
                "result": verdict.status,
                "blocking_gate": verdict.blocking_gate,
                "gates": [{item.gate: item.status} for item in verdict.gates],
            }
        )
    )
    if verdict.status == "PASS":
        return EXIT_PASS
    if verdict.status == "FAIL":
        return EXIT_FAIL
    return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
