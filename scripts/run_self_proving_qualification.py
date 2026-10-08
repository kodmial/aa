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
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aa.conversation.stage_telemetry import TurnTelemetry

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
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=str(ROOT), check=False
        )
    except OSError as exc:
        raise SelfProvingError(f"git {' '.join(args)} unavailable: {exc}") from exc
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

    settings = Settings.from_env(None)
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
    except (SelfProvingError, OSError) as exc:
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
        dirty_raw = _git(["status", "--porcelain"])
    except (SelfProvingError, OSError) as exc:
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    ignored_prefixes = (
        "eval-self-proving-out/",
        "self-proving-out/",
        "live-out/",
        "runtime-status.json",
    )
    dirty_lines = []
    for line in dirty_raw.splitlines():
        if not line.strip():
            continue
        raw_path = line[3:].strip()
        candidates = [p.strip().strip('"') for p in raw_path.split(" -> ") if p.strip()]
        if not candidates:
            dirty_lines.append(line)
            continue

        def _is_ignored(path: str) -> bool:
            for prefix in ignored_prefixes:
                if prefix.endswith("/"):
                    if path.startswith(prefix):
                        return True
                elif path == prefix:
                    return True
            return False

        if all(_is_ignored(path) for path in candidates):
            continue
        dirty_lines.append(line)
    if dirty_lines:
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
    try:
        from aa.config import (
            DEFAULT_AA_AGENT,
            DEFAULT_FALLBACK_MODEL,
            DEFAULT_PRIMARY_MODEL,
            Settings,
        )

        _policy = Settings.from_env(None)
        _policy_ok = (
            _policy.opencode_agent == DEFAULT_AA_AGENT
            and _policy.opencode_model == DEFAULT_PRIMARY_MODEL
            and _policy.opencode_fallback_model == DEFAULT_FALLBACK_MODEL
            and _policy.opencode_model != _policy.opencode_fallback_model
        )
    except Exception:
        _policy_ok = False
    if not _policy_ok:
        return GateEvidence(
            gate="A",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="model-policy-mismatch",
            component="model-policy",
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


def _evidence_has_second_stage_reranker(evidence_mod: object) -> bool:
    """Detect real second-stage reranker code (ignores comments/strings).

    Substring search over source text false-positives on docstrings,
    comments, or log strings and false-negatives on renamed wiring.
    This inspects executable code only: AST import/name/attribute nodes
    plus live module attributes of the evidence module itself. String
    constants (docstrings, log messages) never trigger. Globally loaded
    modules (``sys.modules``) are never consulted: unrelated transitive
    imports must not fail the evidence module's own RRF-only check.
    """

    import ast

    code_fragments = (
        "bge",
        "cross_encoder",
        "crossencoder",
        "sentence_transformers",
        "flagembedding",
        "rerank",
    )

    def _code_hit(name: str) -> bool:
        lowered = name.lower()
        if "cross-encoder" in lowered:
            return True
        return any(fragment in lowered for fragment in code_fragments)

    try:
        module_file = getattr(evidence_mod, "__file__", "")
        tree = ast.parse(Path(str(module_file)).read_text(encoding="utf-8"))
    except Exception:
        raise
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _code_hit(alias.name or ""):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if _code_hit(node.module or ""):
                return True
            for alias in node.names:
                if _code_hit(alias.name or ""):
                    return True
        elif isinstance(node, ast.Name):
            # Skip the local binding of the imported RRF primitive itself;
            # presence of ``rrf_fuse`` is the expected RRF-only path.
            if node.id == "rrf_fuse":
                continue
            if _code_hit(node.id or ""):
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr == "rrf_fuse":
                continue
            if _code_hit(node.attr or ""):
                return True
    for attr_name in dir(evidence_mod):
        if attr_name in ("rrf_fuse",):
            continue
        if _code_hit(attr_name):
            return True
    return False


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

        if _evidence_has_second_stage_reranker(evidence_mod):
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "retrieval-rrf",
                "second-stage reranker present",
            )
    except Exception as exc:
        return _fail_b(
            expected_sha,
            run_id,
            product,
            runtime,
            "retrieval-rrf",
            type(exc).__name__[:96],
        )

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

    # -- canonical RU BM25 + pinned E5/FAISS production index proof --
    # Gate B must validate the same generated retrieval substrate the runtime
    # consumes. Synthetic chunks or the hermetic hashing backend are not
    # acceptable evidence for Product Contract qualification.
    try:
        from aa.retrieval.dense import E5_BACKEND_NAME
        from aa.retrieval.index import (
            close_hybrid_index,
            open_hybrid_index,
            search_aspect,
        )

        retrieval_dir = ROOT / "corpus" / "generated" / "retrieval"
        required = (
            ROOT / "corpus" / "generated" / "canonical.ru.json",
            ROOT / "corpus" / "generated" / "corpus_structure.json",
            retrieval_dir / "index.json",
            retrieval_dir / "dense.json",
            retrieval_dir / "lexical.db",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return _fail_b(
                expected_sha,
                run_id,
                product,
                runtime,
                "corpus-restore",
                "missing production artifacts: " + ",".join(missing),
            )

        index = open_hybrid_index(
            retrieval_dir,
            ru_manifest_path=ROOT / "corpus" / "canonical.ru.manifest.json",
            en_manifest_path=ROOT / "corpus" / "canonical.manifest.json",
            lock_path=ROOT / "corpus" / "embedding.lock.json",
        )
        try:
            if not index.ram_resident or index.lexical_conn is None:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "bm25-index",
                    "production lexical index is not RAM-resident",
                )
            if index.metadata.get("embedding_backend") != E5_BACKEND_NAME:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "e5-faiss-index",
                    "production index is not pinned E5",
                )
            if not getattr(index.dense, "use_faiss", False):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "e5-faiss-index",
                    "production dense index is not FAISS IndexFlatIP",
                )
            if index.metadata.get("embedding_model_id") != "intfloat/multilingual-e5-base":
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "e5-faiss-index",
                    "production embedding model identity mismatch",
                )
            queries = [
                "как оставаться трезвым сегодня",
                "что книга говорит о тяге к алкоголю",
                "как признать бессилие перед алкоголем",
                "отношения с семьёй и выздоровление",
                "что делать после срыва",
                "страх и честность в выздоровлении",
                "помощь другим алкоголикам",
                "духовные принципы без религиозного давления",
                "обиды и инвентаризация",
                "ежедневная практика трезвости",
                "как просить о помощи",
                "надежда на изменение жизни",
            ]
            hits = search_aspect(index, queries)
            if not hits:
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "retrieval-rrf",
                    "production BM25+E5/FAISS search returned no fused hits",
                )
            if any(
                not hit.source_id
                or not hit.text_sha256
                or not hit.section
                or not hit.logical_chunk_id
                for hit in hits
            ):
                return _fail_b(
                    expected_sha,
                    run_id,
                    product,
                    runtime,
                    "small-to-big",
                    "production retrieval hit lacks canonical provenance",
                )
        finally:
            close_hybrid_index(index)
    except Exception as exc:
        label = type(exc).__name__
        component = "e5-faiss-index"
        message = str(exc).lower()
        if "lexical" in message or "sqlite" in message or "fts" in message:
            component = "bm25-index"
        elif "stale" in message or "manifest" in message or "canonical" in message:
            component = "corpus-restore"
        return _fail_b(
            expected_sha,
            run_id,
            product,
            runtime,
            component,
            f"{label}: {exc}"[:96],
        )

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


def _coerce_turn_telemetries(raw_turns: object) -> list[TurnTelemetry] | None:
    """Coerce a JSON ``turns`` list into ``TurnTelemetry`` objects, if possible."""
    if not isinstance(raw_turns, list) or not raw_turns:
        return None
    try:
        from aa.conversation.stage_telemetry import TurnTelemetry, record_stage
    except Exception:
        return None
    coerced: list[TurnTelemetry] = []
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


def _live_real_latencies_from_product_summary(payload: object) -> list[float] | None:
    """Return explicit per-turn latencies (ms) when the payload carries them.

    Only real measured per-turn samples are returned. Aggregate-only
    summaries (p50/p95/max without a per-turn array) yield ``None`` so the
    caller evaluates the aggregate SLO directly instead of synthesising a
    representative sample (synthesis distorts recomputed p95/max).
    """
    if not isinstance(payload, dict):
        return None
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        return None
    collected: list[float] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        metrics = lane.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key in (
            "turn_latencies_ms",
            "per_turn_latencies_ms",
            "per_turn_ms",
            "latencies_ms",
            "turn_ms",
        ):
            raw = metrics.get(key, None)
            if isinstance(raw, list) and raw:
                try:
                    values = [float(v) for v in raw]
                except (TypeError, ValueError):
                    continue
                values = [v for v in values if v > 0]
                if values:
                    collected.extend(values)
        turns = lane.get("turns", None)
        if isinstance(turns, list) and turns:
            turn_values: list[float] = []
            valid = True
            for entry in turns:
                if not isinstance(entry, dict):
                    valid = False
                    break
                found: float | None = None
                for key in ("latency_ms", "total_ms", "end_to_end_ms"):
                    if key in entry:
                        try:
                            found = float(entry[key])
                        except (TypeError, ValueError):
                            found = None
                        break
                if found is None or not found > 0:
                    valid = False
                    break
                turn_values.append(found)
            if valid and turn_values:
                collected.extend(turn_values)
    if collected:
        return collected
    return None


def _live_aggregate_slo_from_product_summary(payload: object) -> dict[str, float] | None:
    """Extract the measured aggregate SLO (ms) directly, without synthesis.

    Units come from key names only (``*_s`` seconds convert to ms,
    ``*_ms`` are already ms). Returns ``{"p50_ms","p95_ms","max_ms",
    "turns"}`` with the conservative cross-lane maximum, or ``None`` when
    no usable aggregate exists.
    """
    if not isinstance(payload, dict):
        return None
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        return None
    best_p50 = 0.0
    best_p95 = 0.0
    best_max = 0.0
    total_turns = 0
    seen = False
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
        p50_s = metrics.get("latency_p50_s", None)
        p95_s = metrics.get("latency_p95_s", None)
        max_s = metrics.get("latency_max_s", None)
        p50_ms_raw = metrics.get("end_to_end_p50_ms", None)
        p95_ms_raw = metrics.get("end_to_end_p95_ms", None)
        max_ms_raw = metrics.get("max_ms", None)
        try:
            # Units are determined by key name, never by magnitude: `*_s`
            # keys are seconds and convert to ms; `*_ms` keys are already
            # ms and must not be scaled (a magnitude heuristic cannot
            # distinguish seconds-scale from small-ms-scale values).
            if p50_s is not None:
                p50_ms = float(p50_s or 0.0) * 1000.0
            else:
                p50_ms = float(p50_ms_raw or 0.0)
            if p95_s is not None:
                p95_ms = float(p95_s or 0.0) * 1000.0
            else:
                p95_ms = float(p95_ms_raw or 0.0)
            if max_s is not None:
                max_ms = float(max_s or 0.0) * 1000.0
            else:
                max_ms = float(max_ms_raw or 0.0)
        except (TypeError, ValueError):
            continue
        if count <= 0 or p50_ms <= 0:
            continue
        if max_ms <= 0:
            continue
        lane_max = max_ms
        best_p50 = max(best_p50, p50_ms)
        best_p95 = max(best_p95, p95_ms)
        best_max = max(best_max, lane_max)
        total_turns += count
        seen = True
    if not seen or total_turns <= 0 or best_p50 <= 0:
        return None
    return {
        "p50_ms": float(best_p50),
        "p95_ms": float(best_p95),
        "max_ms": float(best_max),
        "turns": float(total_turns),
    }


def _live_latencies_from_product_summary(payload: object) -> list[float] | None:
    """Back-compat wrapper: return only real per-turn samples, never synthesis."""
    return _live_real_latencies_from_product_summary(payload)


# Gate C lane-evidence strictness (fail-closed, no exact-question whitelist).
#
# A lane with ``passed=["trivial"]`` plus a network-deferred ``incomplete``
# must never become Gate C PASS. Every lane must carry multiple explicit
# production checks, and the only accepted ``incomplete`` entries are the
# exact real-Telegram dialing deferrals owned by Gate D. Any other
# incomplete (missing identity/models/runtime, harness gaps) stays BLOCKED.
_ALLOWED_GATE_C_DEFERRED: frozenset[str] = frozenset(
    {
        "real-telegram-typing-stream-not-dialed-in-qualification",
        "real-telegram-typing-stream-requires-token",
    }
)

_PRODUCTION_CHECK_TOKENS: tuple[str, ...] = (
    "planner",
    "queries",
    "retrieval",
    "rrf",
    "grounding",
    "verifier",
    "answer",
    "safety",
    "envelope",
    "transport",
    "typing",
    "heartbeat",
    "concurrency",
    "fifo",
    "controller",
    "ready",
    "delivery",
    "greeting",
    "capability",
    "memory",
    "quote",
    "citation",
    "leak",
)


def _lane_has_production_checks(passed: object) -> bool:
    if not isinstance(passed, list) or len(passed) < 4:
        return False
    lowered = [str(item).lower() for item in passed if isinstance(item, str) and str(item).strip()]
    if len(lowered) < 4:
        return False
    matched: set[str] = set()
    for name in lowered:
        for token in _PRODUCTION_CHECK_TOKENS:
            if token in name:
                matched.add(token)
                break
    return len(matched) >= 4


_STAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("planner", "queries", "greeting", "capability"),
    ("retrieval", "rrf", "memory", "quote", "citation"),
    ("verifier", "grounding", "safety", "answer", "leak"),
    (
        "transport",
        "typing",
        "concurrency",
        "fifo",
        "controller",
        "ready",
        "delivery",
        "envelope",
        "heartbeat",
    ),
)


def _stage_group_covers(name: str, group: tuple[str, ...]) -> bool:
    return any(token in name for token in group)


def _union_has_distinct_stage_cover(names: list[str]) -> bool:
    """Whether distinct passed checks cover every pipeline stage group.

    Each group must be satisfied by a different check name, so one token
    (e.g. ``rrf``) cannot satisfy two stages. Groups are disjoint by
    construction.
    """
    used: set[int] = set()

    def _assign(group_idx: int) -> bool:
        if group_idx == len(_STAGE_GROUPS):
            return True
        group = _STAGE_GROUPS[group_idx]
        for name_idx, name in enumerate(names):
            if name_idx in used:
                continue
            if _stage_group_covers(name, group):
                used.add(name_idx)
                if _assign(group_idx + 1):
                    return True
                used.remove(name_idx)
        return False

    return _assign(0)


def _gate_c_live_evidence(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> tuple[GateEvidence, list[float], dict[str, float] | None] | None:
    """Map repository-owned live evidence to a Gate C verdict, if present.

    Consumes the ``live-out`` artifact produced by
    ``run_product_contract_live_qualification.py`` (real local OpenCode
    process, real provider/model policy, real decrypted RU corpus + real
    production index, synthetic Updates injected only at the production
    Telegram adapter boundary). Returns ``None`` when no usable live
    evidence exists (caller stays BLOCKED fail-closed).

    The third tuple element carries the measured aggregate SLO
    (p50/p95/max_ms + turns) evaluated directly from ``*_s``/``*_ms``
    keys. Gate E consumes real per-turn samples when present, otherwise
    the aggregate SLO directly; it never consumes synthesised samples.
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
            return (
                GateEvidence(
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
                ),
                [],
                None,
            )
        if not isinstance(payload, dict):
            continue
        main_sha = payload.get("main_sha", payload.get("sha"))
        if str(main_sha or "") != expected_sha:
            continue
        payload_run = str(payload.get("run_id", payload.get("run", "")) or "")
        if payload_run != run_id:
            continue
        # Telemetry-shaped evidence: evaluate scenario families + diversity.
        raw_turns = payload.get("turns")
        if isinstance(raw_turns, list) and raw_turns:
            turns = _coerce_turn_telemetries(raw_turns)
            if turns is None:
                return (
                    GateEvidence(
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
                    ),
                    [],
                    None,
                )
            try:
                from aa.conversation.stage_telemetry import evaluate_gate_c_telemetry
                from aa.qualification.self_proving import SCENARIO_FAMILIES

                ok, detail, _ = evaluate_gate_c_telemetry(
                    turns,
                    required_families=len(SCENARIO_FAMILIES),
                )
                try:
                    latencies = [float(t.total_ms()) for t in turns]
                except Exception as exc:
                    return (
                        GateEvidence(
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
                        ),
                        [],
                        None,
                    )
            except Exception as exc:
                return (
                    GateEvidence(
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
                    ),
                    [],
                    None,
                )
            if ok:
                return (
                    GateEvidence(
                        gate="C",
                        status="PASS",
                        sha=expected_sha,
                        product_fingerprint=product,
                        runtime_fingerprint=runtime,
                        component="live-production-path",
                        run_id=run_id,
                        live_trusted=True,
                        mocked_only=False,
                    ),
                    latencies,
                    None,
                )
            return (
                GateEvidence(
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
                ),
                latencies,
                None,
            )
        # LiveSummary evidence: Gate C owns only the exact live production
        # conversation lane. The aggregate summary also contains deterministic
        # transport, runtime-control and voice/resource lanes whose failures
        # belong to B/D/E. Never let a foreign lane poison Gate C and start
        # another misclassified repair loop.
        lanes = payload.get("lanes", [])
        real_latencies = _live_real_latencies_from_product_summary(payload) or []
        aggregate = _live_aggregate_slo_from_product_summary(payload)
        if not isinstance(lanes, list):
            lanes = []
        live_lane = next(
            (
                lane
                for lane in lanes
                if isinstance(lane, dict) and str(lane.get("lane", "")) == "live-telegram-evidence"
            ),
            None,
        )
        if live_lane is None:
            return (
                GateEvidence(
                    gate="C",
                    status="BLOCKED",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category="live-evidence-incomplete",
                    component="live-production-path",
                    run_id=run_id,
                    mocked_only=False,
                    detail="missing-live-telegram-evidence-lane",
                ),
                real_latencies,
                aggregate,
            )

        lane_status = str(live_lane.get("status", "")).upper()
        passed = [
            str(item) for item in (live_lane.get("passed", []) or []) if isinstance(item, str)
        ]
        failed = [
            str(item) for item in (live_lane.get("failed", []) or []) if isinstance(item, str)
        ]
        incomplete = [
            str(item) for item in (live_lane.get("incomplete", []) or []) if isinstance(item, str)
        ]
        required_checks = {
            "live-raw-telegram-transport-boundary",
            "live-answer-no-generic-collapse",
            "live-substantive-grounded-book-answer",
            "live-answer-diversity",
            "live-actual-served-model-identity",
            "live-planner-retrieval-answer-verifier-telemetry",
        }

        if lane_status == "PASS" and required_checks.issubset(set(passed)):
            return (
                GateEvidence(
                    gate="C",
                    status="PASS",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    component="live-production-path",
                    run_id=run_id,
                    live_trusted=True,
                    mocked_only=False,
                ),
                real_latencies,
                aggregate,
            )
        if lane_status == "FAIL":
            category = failed[0] if failed else "live-path-failed"
            return (
                GateEvidence(
                    gate="C",
                    status="FAIL",
                    sha=expected_sha,
                    product_fingerprint=product,
                    runtime_fingerprint=runtime,
                    failure_category=category[:96],
                    component="live-production-path",
                    run_id=run_id,
                    live_trusted=True,
                    mocked_only=False,
                    detail="live-telegram-evidence-failed",
                ),
                real_latencies,
                aggregate,
            )
        detail = incomplete[0] if incomplete else "live-evidence-incomplete"
        return (
            GateEvidence(
                gate="C",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="live-evidence-incomplete",
                component="live-production-path",
                run_id=run_id,
                mocked_only=False,
                detail=detail[:160],
            ),
            real_latencies,
            aggregate,
        )
    return None


def _gate_c(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> tuple[GateEvidence, list[float], dict[str, float] | None]:
    # Preserve repository-owned live evidence first (issue #153): when Gate C
    # already produced a matching-SHA/run live artifact (PASS or FAIL), that
    # verdict is authoritative even if the current process lacks the live
    # execution context (Gate E/F re-evaluation must still preserve it, and
    # this runner-level ordering is the backstop). This prevents Gate F from
    # replacing a real live-path-failed with live-execution-not-enabled and
    # sending the repair loop at the wrong category. Only when no usable live
    # evidence exists does the prerequisite check apply (fail-closed, never
    # mocked PASS).
    decided = _gate_c_live_evidence(expected_sha, run_id, product, runtime)
    if decided is not None:
        return decided
    ready, reason = _live_prerequisites()
    if not ready:
        return (
            GateEvidence(
                gate="C",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category=reason,
                component="live-production-path",
                run_id=run_id,
                mocked_only=True,
            ),
            [],
            None,
        )
    # Live prerequisites hold but no usable live artifact exists: stay BLOCKED
    # (fail-closed); never report mocked PASS.
    return (
        GateEvidence(
            gate="C",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="live-evidence-required",
            component="live-production-path",
            run_id=run_id,
            mocked_only=False,
        ),
        [],
        None,
    )


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
        if marker.sha != expected_sha or marker.run_id != run_id:
            continue
        if marker.phase == "STOPPED":
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
        if marker.phase == "READY":
            return GateEvidence(
                gate="D",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="telegram-readiness-unproven",
                component="telegram-readiness",
                run_id=run_id,
                detail="runtime-ready-requires-clean-stop",
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
    # Otherwise prove real Telegram network reachability live, but never PASS
    # on getMe/webhook alone: PASS requires the durable READY/STOPPED marker
    # published from the running Application after proving OpenCode health,
    # getMe identity, webhook/commands bootstrap, ``transport.running`` with
    # a live poll task, and a clean stop with no leaked poll task. A bare
    # getMe-only script is forbidden and stays BLOCKED here.
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
            return GateEvidence(
                gate="D",
                status="BLOCKED",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="telegram-readiness-unproven",
                component="telegram-readiness",
                run_id=run_id,
                detail="webhook-configured-polling-blocked",
            )
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
        status="BLOCKED",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        failure_category="telegram-readiness-unproven",
        component="telegram-readiness",
        run_id=run_id,
        detail="live-application-proof-required",
    )


def _gate_e(
    expected_sha: str,
    run_id: str,
    product: str,
    runtime: str,
    latencies_ms: list[float],
    gate_c_live: bool,
    aggregate_slo_ms: dict[str, float] | None = None,
) -> GateEvidence:
    from aa.qualification.self_proving import (
        ORDINARY_TURN_BUDGET_MS,
        P95_TARGET_MS,
        slo_guards,
    )

    def _blocked() -> GateEvidence:
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

    def _fail(p50: float, p95: float, maximum: float, detail: str = "") -> GateEvidence:
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
            detail=detail[:160],
        )

    def _verdict(p50: float, p95: float, maximum: float) -> GateEvidence:
        if p95 > float(P95_TARGET_MS) or maximum >= float(ORDINARY_TURN_BUDGET_MS):
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

    if not gate_c_live:
        return _blocked()
    # Prefer real measured per-turn samples; they carry the true variance.
    if latencies_ms:
        ok, detail, metrics = slo_guards(latencies_ms)
        p50 = float(metrics["p50_ms"])
        p95 = float(metrics["p95_ms"])
        maximum = float(metrics["max_ms"])
        if not ok:
            return _fail(p50, p95, maximum, detail)
        return _verdict(p50, p95, maximum)
    # Aggregate-only live summaries: evaluate the measured aggregates
    # directly against the canonical SLO (no sample synthesis, which
    # distorts recomputed p95/max).
    if aggregate_slo_ms is not None:
        try:
            p50 = float(aggregate_slo_ms.get("p50_ms", 0.0))
            p95 = float(aggregate_slo_ms.get("p95_ms", 0.0))
            maximum = float(aggregate_slo_ms.get("max_ms", 0.0))
            turns = float(aggregate_slo_ms.get("turns", 0.0))
        except (TypeError, ValueError):
            return _blocked()
        if turns <= 0 or p50 <= 0 or p95 <= 0 or maximum <= 0:
            return _blocked()
        if not all(math.isfinite(v) for v in (p50, p95, maximum, turns)):
            return _blocked()
        if p95 > float(P95_TARGET_MS) or maximum >= float(ORDINARY_TURN_BUDGET_MS):
            return _fail(p50, p95, maximum, "aggregate p95/budget exceeds canonical guard")
        return _verdict(p50, p95, maximum)
    return _blocked()


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
        restart_marker = out_dir / "restart-required.json"
        restart_marker.write_text(
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

    try:
        gate_a = _gate_a(expected, run_id)
    except (SelfProvingError, OSError) as exc:
        gate_a = GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=f"{type(exc).__name__}: {exc}"[:160],
        )
    except Exception as exc:
        gate_a = GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected,
            failure_category="runner-error",
            run_id=run_id,
            detail=f"{type(exc).__name__}: {exc}"[:160],
        )
    if gate_a.status in ("STALE",):
        _write(out_dir, None, "STALE", main_sha=expected, run_id=run_id)
        print("self-proving qualification STALE: SHA mismatch", file=sys.stderr)
        return EXIT_STALE
    # Fail-closed fingerprint backfill: Gate A may return empty fingerprints
    # on fingerprint-error. Recomputing here must never crash the runner
    # with no result.json/summary (that would violate the BLOCKED contract).
    fingerprint_recompute_failed = ""
    try:
        product = gate_a.product_fingerprint or _product_fingerprint()
    except Exception as exc:
        product = ""
        fingerprint_recompute_failed = type(exc).__name__
    try:
        runtime = gate_a.runtime_fingerprint or _runtime_fingerprint()
    except Exception as exc:
        runtime = ""
        fingerprint_recompute_failed = fingerprint_recompute_failed or type(exc).__name__
    if fingerprint_recompute_failed:
        gate_a = GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="fingerprint-error",
            component="fingerprint-consistency",
            run_id=run_id,
            detail=fingerprint_recompute_failed[:64],
        )
    elif not gate_a.product_fingerprint or not gate_a.runtime_fingerprint:
        # Preserve the original Gate A failure category while attaching the
        # successfully recomputed fingerprints. Otherwise the final exact-SHA
        # consistency check can misclassify dirty-tree/git-unavailable as a
        # fingerprint mismatch and send the repair loop at the wrong component.
        gate_a = GateEvidence(
            gate=gate_a.gate,
            status=gate_a.status,
            sha=gate_a.sha,
            product_fingerprint=gate_a.product_fingerprint or product,
            runtime_fingerprint=gate_a.runtime_fingerprint or runtime,
            failure_category=gate_a.failure_category,
            component=gate_a.component,
            run_id=gate_a.run_id,
            live_trusted=gate_a.live_trusted,
            mocked_only=gate_a.mocked_only,
            latency_p50_ms=gate_a.latency_p50_ms,
            latency_p95_ms=gate_a.latency_p95_ms,
            max_turn_ms=gate_a.max_turn_ms,
            detail=gate_a.detail,
        )

    try:
        gate_b = _gate_b(expected, run_id, product, runtime)
    except Exception as exc:
        gate_b = _fail_b(
            expected,
            run_id,
            product,
            runtime,
            "planner-shape",
            f"{type(exc).__name__}: {exc}"[:96],
        )
    try:
        gate_c, latencies, aggregate_slo = _gate_c(expected, run_id, product, runtime)
    except Exception as exc:
        gate_c = GateEvidence(
            gate="C",
            status="BLOCKED",
            sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="runner-error",
            component="live-production-path",
            run_id=run_id,
            detail=f"{type(exc).__name__}: {exc}"[:64],
        )
        latencies, aggregate_slo = [], None
    try:
        gate_d = _gate_d(expected, run_id, product, runtime)
    except Exception as exc:
        gate_d = GateEvidence(
            gate="D",
            status="BLOCKED",
            sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="runner-error",
            component="telegram-readiness",
            run_id=run_id,
            detail=f"{type(exc).__name__}: {exc}"[:64],
        )
    try:
        gate_e = _gate_e(
            expected,
            run_id,
            product,
            runtime,
            latencies,
            gate_c_live=gate_c.live_trusted,
            aggregate_slo_ms=aggregate_slo,
        )
    except Exception as exc:
        gate_e = GateEvidence(
            gate="E",
            status="BLOCKED",
            sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="runner-error",
            component="slo",
            run_id=run_id,
            detail=f"{type(exc).__name__}: {exc}"[:64],
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
        marker = build_result_marker(sha=expected, status="BLOCKED", run=run_id)
        blocking_gate = next((item.gate for item in evidences if item.status != "PASS"), "A")
        blocked_failures: list[object] = []
        for evidence in evidences:
            if evidence.status == "PASS":
                continue
            try:
                report = failure_report_for_gate(evidence)
            except SelfProvingError:
                continue
            payload = report.to_dict()
            payload["repair_fingerprint"] = repair_fingerprint(report)
            blocked_failures.append(payload)
        (out_dir / "self-proving-summary.json").write_text(
            json.dumps(
                {
                    "sha": expected,
                    "status": "BLOCKED",
                    "product_fingerprint": product,
                    "runtime_fingerprint": runtime,
                    "blocking_gate": blocking_gate,
                    "run_id": run_id,
                    "gates": [item.to_dict() for item in evidences],
                    "failures": blocked_failures,
                    "marker": marker,
                    "error": str(exc)[:160],
                },
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "result.json").write_text(
            json.dumps(
                {
                    "result": "BLOCKED",
                    "main_sha": expected,
                    "run_id": run_id,
                    "blocking_gate": blocking_gate,
                    "marker": marker,
                    "gates": [{"gate": item.gate, "status": item.status} for item in evidences],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
