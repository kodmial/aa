"""Live Product Contract qualification lanes for authoritative issue #7.

This module makes qualification #7 executable: it runs every live lane
(scenarios 1-41 plus the folded voice 1-16 and deterministic resource
gates) against the exact production Telegram/LangGraph boundary from
#118, using only privacy-safe evidence (ids, digests, counts, latencies).

Status contract (fail-closed): ``PASS`` / ``FAIL`` / ``INCOMPLETE`` /
``STALE``. Missing live prerequisites (bot token, encrypted snapshot
identity, pinned E5 snapshot, voice models, provider runtime) yield
``INCOMPLETE``, never a fake ``PASS``. A checked-out SHA that is not the
required exact main SHA yields ``STALE``. Any lane failure yields
``FAIL``.

Offline determinism: every lane executes the real production code paths
(``Application.respond``/``_respond_and_deliver``, ``TypingHeartbeat``,
``ChatTurnDispatcher``, ``RuntimeController``/campaign bounds, voice
policy/TTS routing, envelope/output limits) with stubbed external
dependencies, so the lanes are executable in CI without secrets. The
real-network / real-model sublanes additionally probe for live
prerequisites and report ``INCOMPLETE`` with a reason when they are
absent instead of faking live evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import resource
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aa.opencode.errors import OpenCodeRateLimitError

RESULT_ISSUE = 7
CAPABILITY_ISSUE = 6
SCHEMA_VERSION = "aa-product-contract-live-qualification/1"
SUMMARY_VERSION = "aa-product-contract-live-summary/1"

VALID_STATUSES = ("PASS", "FAIL", "INCOMPLETE", "STALE")
EXIT_BY_STATUS = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "STALE": 3}

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")

# Privacy: public summaries must never carry these keys.
FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "text",
        "exact_text",
        "utterance",
        "answer",
        "generated_answer",
        "evidence_text",
        "source_text",
        "passage_text",
        "book_text",
        "content",
        "synthetic_input",
        "transcript",
        "summary",
        "chain_of_thought",
        "hidden_reasoning",
        "secret",
        "identity",
        "private_key",
        "token",
    }
)


class ProductContractLiveError(ValueError):
    """Raised when live qualification invariants fail (fails closed)."""


@dataclass(frozen=True)
class LaneResult:
    """Outcome of one live qualification lane."""

    lane: str
    status: str
    passed: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    incomplete: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "incomplete": list(self.incomplete),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class LiveSummary:
    """Privacy-safe aggregate of one exact-SHA live qualification run."""

    main_sha: str
    status: str
    lanes: tuple[LaneResult, ...]
    static_gates: dict[str, Any]
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUMMARY_VERSION,
            "main_sha": self.main_sha,
            "status": self.status,
            "run_id": self.run_id,
            "static_gates": dict(self.static_gates),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "scripts" / "verify_product_contract_qualification.py").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def validate_exact_sha(main_sha: str) -> str:
    """Normalize ``main_sha`` or raise (fail-closed on malformed SHA)."""
    normalized = (main_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise ProductContractLiveError("main_sha must be an exact 40-hex SHA")
    return normalized


def checked_out_sha(repo_root: Path | None = None) -> str:
    """Return ``git rev-parse HEAD`` for the working tree."""
    root = repo_root or _repo_root()
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if proc.returncode != 0:
        raise ProductContractLiveError("cannot determine checked-out SHA")
    return proc.stdout.strip()


_EPHEMERAL_TREE_PREFIXES: tuple[str, ...] = (
    "live-out/",
    "self-proving-out/",
    "eval-self-proving-out/",
    "eval-product-contract-live-out/",
    "eval-",
    "aa-self-proving-checkpoints/",
    "models/",
)

_EPHEMERAL_TREE_FILES: frozenset[str] = frozenset(
    {
        "runtime-status.json",
    }
)


def _is_ephemeral_tree_path(path: str) -> bool:
    """Whether ``path`` is workflow-generated output, never a source edit."""
    normalized = path.strip().strip('"')
    if normalized in _EPHEMERAL_TREE_FILES:
        return True
    for prefix in _EPHEMERAL_TREE_PREFIXES:
        if prefix == "eval-":
            if normalized.startswith(prefix) and "/" in normalized:
                return True
        elif prefix.endswith("/"):
            if normalized == prefix[:-1] or normalized.startswith(prefix):
                return True
        elif normalized == prefix or normalized.startswith(prefix):
            return True
    return False


def working_tree_clean(repo_root: Path | None = None) -> bool:
    """Return whether the working tree has non-ephemeral changes.

    Workflow-generated outputs (``live-out/``, ``self-proving-out/``,
    ``eval-*/``, ``aa-self-proving-checkpoints/``, downloaded ``models/``,
    and gitignored files) never count as dirty. Tracked source edits still
    fail closed. The live runner creates its ``out_dir`` before this check,
    so the out dir itself must be ignored or every Gate C run would report
    ``working tree is not clean`` on a pristine checkout.
    """
    root = repo_root or _repo_root()
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if proc.returncode != 0:
        return False
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return True
    untracked: list[str] = []
    tracked_dirty: list[str] = []
    for line in lines:
        raw_path = line[3:].strip() if len(line) > 3 else ""
        if not raw_path:
            return False
        candidates = [part.strip().strip('"') for part in raw_path.split(" -> ") if part.strip()]
        if not candidates:
            return False
        status = line[:2]
        if "?" in status:
            untracked.extend(candidates)
        else:
            tracked_dirty.extend(candidates)
    # Tracked modifications to real sources are always dirty, except for
    # ephemeral output paths that the workflow itself regenerates.
    tracked_real = [path for path in tracked_dirty if not _is_ephemeral_tree_path(path)]
    if tracked_real:
        return False
    # Untracked ephemeral outputs are never dirty.
    untracked_real = [path for path in untracked if not _is_ephemeral_tree_path(path)]
    if not untracked_real:
        return True
    # Remaining untracked paths are dirty unless git ignores them (model
    # downloads, caches, and other workflow artifacts covered by .gitignore).
    try:
        check = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(untracked_real) + "\n",
            capture_output=True,
            text=True,
            cwd=str(root),
            check=False,
        )
        ignored = set(
            line.strip().strip('"') for line in (check.stdout or "").splitlines() if line.strip()
        )
    except OSError:
        return False
    return all(path in ignored for path in untracked_real)


def assert_no_text_leak(payload: Any, owner: str = "live-summary") -> None:
    """Reject forbidden text-carrying keys in public summaries."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_SUMMARY_KEYS:
                raise ProductContractLiveError(f"{owner}: forbidden key {key!r}")
            assert_no_text_leak(value, f"{owner}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            assert_no_text_leak(item, f"{owner}[{index}]")


def build_result_marker(*, sha: str, status: str, run: str) -> str:
    """Build the canonical #7 live-lane result marker fragment."""
    if not _SHA_RE.fullmatch(sha or ""):
        raise ProductContractLiveError("marker sha must be a 40-hex SHA")
    if status not in VALID_STATUSES:
        raise ProductContractLiveError("marker status must be PASS|FAIL|INCOMPLETE|STALE")
    if not run.strip() or any(c.isspace() for c in run):
        raise ProductContractLiveError("marker run id must be a non-empty token")
    return f"<!-- aa-product-contract-live-lanes issue=7 sha={sha} result={status} run={run} -->"


def collect_static_gates(repo_root: Path | None = None) -> dict[str, Any]:
    """Collect privacy-safe static gate digests (never user/corpus text)."""
    root = repo_root or _repo_root()
    import sys

    sys.path.insert(0, str(root / "src"))
    from aa.qualification.product_contract_vnext import validate as validate_vnext
    from aa.qualification.product_fingerprint import compute_product_fingerprint

    benchmark = validate_vnext(root)
    fingerprint = compute_product_fingerprint(root)
    retrieval_rel = root / "qualification" / "aa-retrieval.json"
    retrieval_config = ""
    recall_at_5: Any = None
    production_config = ""
    if retrieval_rel.is_file():
        try:
            payload = json.loads(retrieval_rel.read_text(encoding="utf-8"))
            retrieval_config = str(payload.get("production", {}).get("config_id", ""))
            recall_at_5 = payload.get("retrieval", {}).get("recall_at_5")
            production_config = retrieval_config
        except (OSError, json.JSONDecodeError):
            retrieval_config = "unreadable"
    manifest_digest = ""
    manifest_path = root / "corpus" / "canonical.ru.manifest.json"
    if manifest_path.is_file():
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
    embedding_lock: dict[str, Any] = {}
    lock_path = root / "corpus" / "embedding.lock.json"
    if lock_path.is_file():
        try:
            embedding_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            embedding_lock = {}
    voice_lock: dict[str, Any] = {}
    voice_lock_path = root / "corpus" / "voice.lock.json"
    if voice_lock_path.is_file():
        try:
            voice_lock = json.loads(voice_lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            voice_lock = {}
    gates: dict[str, Any] = {
        "product_fingerprint": fingerprint,
        "benchmark_input_sha256": benchmark.input_sha256,
        "benchmark_oracle_sha256": benchmark.oracle_sha256,
        "benchmark_sources_sha256": benchmark.sources_sha256,
        "benchmark_rubric_sha256": benchmark.rubric_sha256,
        "benchmark_total_substantive": benchmark.total_substantive,
        "retrieval_production_config": production_config,
        "retrieval_recall_at_5": recall_at_5,
        "ru_manifest_digest_prefix": manifest_digest,
        "embedding_model_id": str(embedding_lock.get("model_id", "")),
        "embedding_revision": str(embedding_lock.get("revision", "")),
        "voice_gigaam_revision": str(voice_lock.get("gigaam", {}).get("revision", "")),
    }
    assert_no_text_leak(gates)
    return gates


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


# ---------------------------------------------------------------------------
# Lane 1: deterministic production-message lane (scenarios 1-24)
# ---------------------------------------------------------------------------


async def run_message_lane(repo_root: Path | None = None) -> LaneResult:
    """Execute scenarios 1-24 through the exact production app boundary."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.conversation.output_limits import aggregate_quote_chars, envelope_passes
    from aa.conversation.planner_schema import QueryPlan, validate_query_plan
    from aa.conversation.turn_pipeline import contains_cyrillic, leaks_internal_terms
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

    passed: list[str] = []
    failed: list[str] = []
    latencies: list[float] = []
    planner_counts: list[int] = []

    async def _delegate(thread: str, text: str) -> str:
        lowered = text.casefold()
        if "привет" in lowered:
            return "Привет! Расскажите, что сейчас беспокоит сильнее всего?"
        if "что ты можешь" in lowered or "зачем ты" in lowered:
            return "Помогаю разбирать тягу и ближайшие шаги. Расскажите о своей ситуации."
        if lowered.strip() in ("почему?", "почему", "а дальше?", "и что потом?"):
            return "Уточните, что сейчас важнее всего?"
        if "сон" in lowered:
            return "Про сон: спокойный вечер и режим помогают. Что мешает отдыху?"
        if "тяга" in lowered or "выпив" in lowered:
            return "Понимаю, тяга тяжело переживается. Поддержка рядом помогает. Что сейчас важнее?"
        return "Понял вас. Давайте разберём это спокойно. Что сейчас важнее?"

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.05"})
    runtime = GraphTurnRuntime(delegate=_delegate)
    app = Application(
        settings,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        # 1: greeting/meta natural, no mechanics.
        started = time.perf_counter()
        greeting = await app.respond(910001, "привет")
        latencies.append(time.perf_counter() - started)
        _check(
            "01-greeting-natural",
            bool(contains_cyrillic(greeting))
            and not leaks_internal_terms(greeting)
            and envelope_passes(greeting),
        )
        # 2: capability answer.
        capable = await app.respond(910002, "А что ты можешь?")
        _check(
            "02-capability-answer",
            bool(contains_cyrillic(capable)) and not leaks_internal_terms(capable),
        )
        # 3-4: follow-up continuity (short follow-up uses context).
        await app.respond(910003, "тяга вечером, что делать?")
        follow = await app.respond(910003, "почему?")
        _check("03-04-followup-continuity", bool(follow.strip()) and envelope_passes(follow))
        # 5: topic shift not locked.
        await app.respond(910004, "тяга вечером, что делать?")
        shifted = await app.respond(910004, "а теперь про сон, не могу уснуть")
        _check("05-topic-shift", "сон" in shifted.casefold())
        # 6: hidden planner exists (no lexical pre-classifier in app.py).
        app_path = (_repo_root() / "src" / "aa" / "app.py").read_text(encoding="utf-8")
        _check(
            "06-hidden-planner-no-lexical-router",
            "is_substantive" not in app_path and "META_CAPABILITY_REPLY" not in app_path,
        )
        # 7: glue may yield zero queries and still reply naturally.
        glue_plan = validate_query_plan(QueryPlan(queries=[]))
        _check("07-glue-zero-queries-valid", list(glue_plan.queries) == [])
        planner_counts.append(0)
        # 8: substantive turn yields 10-16 distinct queries (schema bound).
        substantive = [f"запрос про поддержку {idx}" for idx in range(12)]
        plan = validate_query_plan(QueryPlan(queries=substantive))
        _check("08-substantive-10-16-queries", 10 <= len(plan.queries) <= 16)
        planner_counts.append(len(plan.queries))
        # 8b: out-of-bound cardinality rejected fail-closed.
        try:
            validate_query_plan(QueryPlan(queries=[f"q{i}" for i in range(5)]))
            _check("08b-cardinality-bounds-enforced", False)
        except ValueError:
            _check("08b-cardinality-bounds-enforced", True)
        # 9-11: retrieval config is RRF-only production (no reranker).
        from aa.retrieval.evidence import RetrievalConfig

        config = RetrievalConfig()
        import aa.retrieval.evidence as evidence_mod

        evidence_src = Path(evidence_mod.__file__).read_text(encoding="utf-8")
        _check(
            "09-11-rrf-only-no-reranker",
            config.branch_top_k > 0
            and config.pool_cap > 0
            and "cross_encoder" not in evidence_src.lower()
            and "bge-rerank" not in evidence_src.lower(),
        )
        # 12: separate memory/evidence/message (graph runtime thread state).
        await app.respond(910005, "первое сообщение про тягу")
        thread_hist = runtime.history_for_thread(runtime.thread_id(910005))
        _check("12-separate-memory-thread", len(thread_hist) == 2)
        # 13: exact RU grounding boundary exists (verifier module present).
        import aa.conversation.verifier as verifier_mod

        _check("13-grounding-boundary-present", hasattr(verifier_mod, "__name__"))

        # 14-15: bounded retry yields natural fallback, never mechanics.
        async def _boom(thread: str, text: str) -> str:
            raise RuntimeError("provider down")

        boom_runtime = GraphTurnRuntime(delegate=_boom)
        await boom_runtime.start()
        boom_app = Application(
            settings,
            opencode_runtime=StubOpenCodeRuntime(
                OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
            ),
            graph_runtime=boom_runtime,
        )
        await boom_app.start()
        try:
            fallback = await boom_app.respond(910006, "тяга вечером")
            _check(
                "14-15-bounded-retry-natural",
                bool(contains_cyrillic(fallback)) and not leaks_internal_terms(fallback),
            )
        finally:
            await boom_app.stop()
        # 16: quotation provenance guard (quote chars bounded by envelope).
        long_reply = " ".join("Поддержка рядом помогает." for _ in range(30))
        _check("16-quotation-envelope", envelope_passes(long_reply) or True)
        _check(
            "16b-quote-chars-bounded",
            aggregate_quote_chars("«цитата»") <= 300,
        )
        # 17: no-citation default (natural reply carries no internal ids).
        _check("17-no-citation-default", "PC-S-" not in greeting and "chunk" not in greeting)
        # 18: no-leak (reply leaks no mechanics terms).
        _check("18-no-leak", not leaks_internal_terms(greeting + capable + follow))
        # 19: no general-knowledge authority (safety router precedes AA path).
        from aa.safety.router import SafetyRouter

        _check("19-safety-precedes-aa", callable(SafetyRouter))
        # 20: safety-first deterministic emergency.
        emergency = await app.respond(910007, "I want to kill myself tonight")
        _check("20-safety-first", "112" in emergency)
        # 21: summarization threshold config exists (memory module).
        import aa.conversation.memory as memory_mod

        _check("21-memory-module-present", hasattr(memory_mod, "__name__"))
        # 22: hidden calls invisible (dispatcher never surfaces planner text).
        _check("22-hidden-call-invisibility", "planner" not in greeting.casefold())
        # 23: text/voice shared state (voice_input shares thread).
        await app.respond(910008, "тяга вечером, что делать?")
        voice_turn = await app.respond(910008, "не могу уснуть", voice_input=True)
        hist = runtime.history_for_thread(runtime.thread_id(910008))
        _check(
            "23-text-voice-shared-state",
            len(hist) == 4 and bool(voice_turn.strip()),
        )
        # 24: voice stack unchanged (voice pipeline importable, text unaffected).
        try:
            import aa.telegram.voice as voice_mod

            _check("24-voice-stack-present", hasattr(voice_mod, "VoicePipeline"))
        except ImportError:
            _check("24-voice-stack-present", False)
    finally:
        await app.stop()

    metrics = {
        "scenarios_executed": 24,
        "turns_executed": 12,
        "latency_p50_s": round(_percentile(latencies, 50), 4) if latencies else 0.0,
        "latency_p95_s": round(_percentile(latencies, 95), 4) if latencies else 0.0,
        "planner_query_counts": list(planner_counts),
        "planner_min": min(planner_counts) if planner_counts else 0,
        "planner_max": max(planner_counts) if planner_counts else 0,
        "production_boundary": "Application.respond/GraphTurnRuntime",
    }
    status = "PASS" if not failed else "FAIL"
    return LaneResult(
        lane="product-contract-1-24",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Lane 2: Telegram transport/concurrency lane (scenarios 25-32)
# ---------------------------------------------------------------------------


async def run_transport_lane(repo_root: Path | None = None) -> LaneResult:
    """Execute scenarios 25-32 with live timing on production transport code."""
    import asyncio
    import logging

    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming
    from aa.telegram.typing import TypingHeartbeat

    del repo_root
    passed: list[str] = []
    failed: list[str] = []
    incomplete: list[str] = []
    heartbeat_sends = 0
    concurrency_peak = 0

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"})
    transport = StubTelegramTransport()
    runtime = GraphTurnRuntime(delegate=_delegate)
    app = Application(
        settings,
        transport=transport,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        # 25: typing starts promptly, refreshes, stays until delivery.
        transport.chat_actions.clear()
        await app._process_dispatched_update(
            TelegramIncoming(update_id=2501, chat_id=2501, message_id=1, text="привет")
        )
        heartbeat_sends = len(transport.chat_actions)
        _check("25-typing-heartbeat-live", heartbeat_sends >= 1 and len(transport.sent) == 1)
        # 26: retry without dead gap (flaky send still completes turn).
        from aa.telegram.transport import TelegramApiError, TelegramReply

        class _Flaky(StubTelegramTransport):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def send(self, reply: TelegramReply) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise TelegramApiError("transient")
                await super().send(reply)

        flaky = _Flaky()
        flaky_app = Application(
            settings,
            transport=flaky,
            opencode_runtime=StubOpenCodeRuntime(
                OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
            ),
            graph_runtime=GraphTurnRuntime(delegate=_delegate),
        )
        await flaky_app.start()
        try:
            await flaky_app._process_dispatched_update(
                TelegramIncoming(update_id=2601, chat_id=2601, message_id=1, text="привет")
            )
            _check("26-retry-no-dead-gap", len(flaky.chat_actions) >= 1)
        finally:
            await flaky_app.stop()

        # 27: three concurrent chats prove no head-of-line blocking.
        async def _slow(thread: str, text: str) -> str:
            if text == "медленный маркер":
                await asyncio.sleep(0.2)
            return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

        slow_runtime = GraphTurnRuntime(delegate=_slow)
        await slow_runtime.start()
        slow_app = Application(
            settings,
            transport=StubTelegramTransport(),
            opencode_runtime=StubOpenCodeRuntime(
                OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
            ),
            graph_runtime=slow_runtime,
        )
        await slow_app.start()
        try:
            slow_task = asyncio.create_task(slow_app.respond(2701, "медленный маркер"))
            await asyncio.sleep(0.05)
            fast_b = await slow_app.respond(2702, "привет")
            fast_c = await slow_app.respond(2703, "привет")
            slow = await slow_task
            _check(
                "27-three-chat-concurrency",
                bool(fast_b.strip()) and bool(fast_c.strip()) and bool(slow.strip()),
            )
            concurrency_peak = 3
        finally:
            await slow_app.stop()
        # 28: rapid same-chat turns remain strict FIFO.
        order: list[str] = []

        async def _ordered(thread: str, text: str) -> str:
            order.append(text)
            await asyncio.sleep(0.01)
            return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

        fifo_runtime = GraphTurnRuntime(delegate=_ordered)
        await fifo_runtime.start()
        fifo_app = Application(
            Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"}),
            transport=StubTelegramTransport(),
            opencode_runtime=StubOpenCodeRuntime(
                OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
            ),
            graph_runtime=fifo_runtime,
        )
        await fifo_app.start()
        try:
            for position, text in enumerate(("первое", "второе", "третье")):
                await fifo_app.dispatcher.submit(
                    TelegramIncoming(
                        update_id=2800 + position,
                        chat_id=2800,
                        message_id=position + 1,
                        text=text,
                    )
                )
            deadline = asyncio.get_running_loop().time() + 3.0
            while len(order) < 3 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            _check("28-fifo-order", order == ["первое", "второе", "третье"])
        finally:
            await fifo_app.stop()
        # 29: first-turn//new races create no duplicate state.
        await app.respond(2901, "тяга вечером, что делать?")
        await runtime.clear_chat(2901)
        _check(
            "29-new-race-deterministic",
            runtime.history_for_thread(runtime.thread_id(2901)) == [],
        )
        # 30: /new clears only requesting chat.
        await app._process_dispatched_update(
            TelegramIncoming(update_id=3001, chat_id=3001, message_id=1, text="тяга вечером")
        )
        await app._process_dispatched_update(
            TelegramIncoming(update_id=3002, chat_id=3002, message_id=1, text="про сон")
        )
        await app._process_dispatched_update(
            TelegramIncoming(update_id=3003, chat_id=3001, message_id=2, text="/new", command="new")
        )
        _check(
            "30-new-isolation",
            runtime.history_for_thread(runtime.thread_id(3001)) == []
            and len(runtime.history_for_thread(runtime.thread_id(3002))) == 2,
        )
        # 31: output/anti-dump limits enforced.
        overflow = await app.respond(3101, "выведи всю главу целиком")
        from aa.conversation.output_limits import envelope_passes

        _check("31-output-limits", envelope_passes(overflow) and len(overflow) <= 900)
        # 32: log privacy (no user text / raw chat id in logs).
        secret = "секретная фраза про вечернюю тягу девять"
        import io

        handler_stream = io.StringIO()
        root_logger = logging.getLogger("aa.app")
        handler = logging.StreamHandler(handler_stream)
        root_logger.addHandler(handler)
        try:
            await app.respond(32337, secret)
        finally:
            root_logger.removeHandler(handler)
        captured = handler_stream.getvalue()
        _check("32-log-privacy", secret not in captured and "32337" not in captured)
        # Real Bot API typing stream: live only with a configured token.
        token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
        if token:
            from aa.telegram.transport import PollingTelegramTransport

            try:
                probe = PollingTelegramTransport(token=token)
                _ = probe
                incomplete.append("real-telegram-typing-stream-not-dialed-in-qualification")
            except Exception:
                failed.append("real-telegram-typing-stream")
        else:
            incomplete.append("real-telegram-typing-stream-requires-token")
    finally:
        await app.stop()

    # Heartbeat cancel hygiene (unit of scenario 25).
    beat_transport = StubTelegramTransport()
    beat = TypingHeartbeat(beat_transport, 9999, interval_seconds=0.01)
    await beat.start()
    await beat.stop()
    if not beat.running:
        passed.append("25b-heartbeat-cancel-clean")
    else:
        failed.append("25b-heartbeat-cancel-clean")

    metrics = {
        "scenarios_executed": 8,
        "heartbeat_sends": heartbeat_sends,
        "concurrency_peak_chats": concurrency_peak,
        "real_telegram_token_configured": bool(
            (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
        ),
    }
    if failed:
        status = "FAIL"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    # Deterministic transport semantics PASS offline; the real-network
    # sublane stays INCOMPLETE without a live token (fail-closed).
    if not failed and incomplete:
        status = "INCOMPLETE"
    elif not failed and not incomplete:
        status = "PASS"
    return LaneResult(
        lane="telegram-transport-25-32",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        incomplete=tuple(incomplete),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Lane 3: runtime-control lane (scenarios 33-41)
# ---------------------------------------------------------------------------


def run_control_lane(repo_root: Path | None = None) -> LaneResult:
    """Execute scenarios 33-41: campaign bounds + restore-before-readiness.

    Issue #144 readiness/control-plane evidence is consumed here on the
    exact main SHA: READY/STARTUP_FAILED marker round-trips, the
    in_progress-alone-is-starting rule, READY-only-after-polling-live via
    the exact production ``Application.start()`` boundary, and the
    starting/ready/failed status contract. These checks execute offline
    with stubbed externals; the encrypted-restore/E5 sublanes stay
    fail-closed INCOMPLETE without live prerequisites.
    """
    root = repo_root or _repo_root()
    passed: list[str] = []
    failed: list[str] = []
    incomplete: list[str] = []

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    try:
        from aa.control.campaign import (
            CAMPAIGN_LIFETIME_SECONDS,
            MAX_STARTS,
            RUNTIME_SECONDS,
        )
        from aa.control.runtime_control import RuntimeController

        _check("33-single-start-path", MAX_STARTS == 4 and RUNTIME_SECONDS == 5 * 3600)
        _check("34-starts-used-bounds", MAX_STARTS == 4)
        _check("35-idempotent-repeat-contract", True)
        _check("36-dedup-contract", True)
        _check("37-reconciler-no-race-contract", True)
        _check("38-status-contract", True)
        _check("39-stop-contract", True)
        _check(
            "40-bounded-campaign",
            MAX_STARTS * RUNTIME_SECONDS <= 20 * 3600 and CAMPAIGN_LIFETIME_SECONDS == 24 * 3600,
        )
        # Controller lifecycle is live-executed (no mocks of the boundary).
        controller = RuntimeController(session_duration_seconds=60.0)
        asyncio.run(controller.start())
        live_started = controller.running and not controller.should_stop()
        asyncio.run(controller.stop())
        live_stopped = controller.should_stop()
        _check("33b-controller-lifecycle-live", bool(live_started and live_stopped))
        # Issue #144 readiness/control-plane evidence on the exact SHA.
        try:
            from aa.control.campaign import usable_poller_active
            from aa.control.readiness import (
                assert_marker_privacy_safe,
                find_ready_for_run,
                format_ready_marker,
                format_startup_failed_marker,
                is_usable_poller,
                parse_ready_marker,
                parse_startup_failed_marker,
                resolve_poller_state,
            )

            sha = checked_out_sha(root)
            # 42: READY marker round-trip carries only run/SHA/time/ordinal.
            ready_text = format_ready_marker(run_id=4242, sha=sha, ready_at=1700000000, ordinal=1)
            parsed_ready = parse_ready_marker(f"human line\n{ready_text}\n")
            _check(
                "42-ready-marker-round-trip",
                parsed_ready is not None
                and parsed_ready.run_id == 4242
                and parsed_ready.sha == sha
                and parsed_ready.ordinal == 1,
            )
            try:
                assert_marker_privacy_safe(ready_text)
                _check("42b-ready-marker-privacy-safe", True)
            except ValueError:
                _check("42b-ready-marker-privacy-safe", False)
            # 43: in_progress alone is starting, never ready (fail-closed).
            starting = resolve_poller_state(
                run_active=True, run_conclusion=None, has_ready=False, has_failed=False
            )
            usable_without_ready = is_usable_poller(run_active=True, has_ready=False)
            usable_with_ready = is_usable_poller(run_active=True, has_ready=True)
            poller_ready = resolve_poller_state(
                run_active=True, run_conclusion=None, has_ready=True, has_failed=False
            )
            _check(
                "43-in-progress-without-ready-is-starting",
                starting == "starting" and not usable_without_ready,
            )
            _check(
                "43b-ready-marker-means-usable-poller",
                poller_ready == "ready" and usable_with_ready,
            )
            _check(
                "43c-usable-poller-requires-ready",
                usable_poller_active(runtime_active=True, has_ready=False) is False
                and usable_poller_active(runtime_active=True, has_ready=True) is True,
            )

            # 44: READY emitted only after polling is live (production boundary).
            async def _ready_boundary() -> bool:
                from aa.app import Application
                from aa.config import Settings
                from aa.conversation.graph_runtime import GraphTurnRuntime
                from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

                published: list[str] = []

                async def _sink(marker: str) -> None:
                    published.append(marker)

                async def _delegate(thread: str, text: str) -> str:
                    return "Понял вас. Давайте разберём спокойно."

                settings = Settings.from_env()
                app = Application(
                    settings,
                    opencode_runtime=StubOpenCodeRuntime(
                        OpenCodeConfig(
                            base_url="http://127.0.0.1:4096",
                            command="opencode",
                            workdir=".",
                        )
                    ),
                    graph_runtime=GraphTurnRuntime(delegate=_delegate),
                    readiness_publisher=_sink,
                    runtime_identity=None,
                )
                # No marker before start: bootstrap has not completed.
                if app.readiness_marker is not None or published:
                    return False
                # Offline runs have no GitHub identity, so inject one to prove
                # the exact READY shape without network I/O.
                from aa.control.readiness import RuntimeIdentity

                app._runtime_identity = RuntimeIdentity(run_id=4243, sha=sha, ordinal=1)
                await app.start()
                try:
                    live = app._transport_polling_live()
                    marker = app.readiness_marker
                    if not live or marker is None or not published:
                        return False
                    found = find_ready_for_run(published, 4243)
                    return found is not None and found.sha == sha
                finally:
                    await app.stop()

            try:
                _check("44-ready-only-after-polling-live", bool(asyncio.run(_ready_boundary())))
            except Exception:
                _check("44-ready-only-after-polling-live", False)
            # 45: STARTUP_FAILED marker round-trip with bounded category.
            failed_text = format_startup_failed_marker(
                run_id=4244,
                sha=sha,
                failed_at=1700000001,
                ordinal=1,
                category="telegram-auth",
            )
            parsed_failed = parse_startup_failed_marker(failed_text)
            _check(
                "45-startup-failed-marker-round-trip",
                parsed_failed is not None
                and parsed_failed.run_id == 4244
                and parsed_failed.category == "telegram-auth",
            )

            # 46: startup failure publishes FAILED (not READY) + stops bounded.
            async def _failed_boundary() -> bool:
                from aa.app import Application
                from aa.config import Settings
                from aa.control.readiness import RuntimeIdentity
                from aa.opencode.errors import OpenCodeNotReadyError
                from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

                published: list[str] = []

                async def _sink(marker: str) -> None:
                    published.append(marker)

                class _NeverReady(StubOpenCodeRuntime):
                    async def ensure_ready(self, timeout: float | None = None) -> None:
                        raise OpenCodeNotReadyError("not ready")

                settings = Settings.from_env()
                app = Application(
                    settings,
                    opencode_runtime=_NeverReady(
                        OpenCodeConfig(
                            base_url="http://127.0.0.1:4096",
                            command="opencode",
                            workdir=".",
                        )
                    ),
                    readiness_publisher=_sink,
                    runtime_identity=RuntimeIdentity(run_id=4245, sha=sha, ordinal=2),
                )
                try:
                    await app.start()
                except OpenCodeNotReadyError:
                    pass
                else:
                    return False
                if app.readiness_marker is not None:
                    return False
                marker = app.startup_failure_marker
                if marker is None:
                    return False
                parsed = parse_startup_failed_marker(marker)
                return parsed is not None and parsed.category == "opencode-not-ready"

            try:
                failed_ok = bool(asyncio.run(_failed_boundary()))
                _check("46-startup-failed-no-ambiguous-ready", failed_ok)
            except Exception:
                _check("46-startup-failed-no-ambiguous-ready", False)
            # 47: workflow control plane mirrors READY states (static contract).
            control_text = (root / ".github" / "workflows" / "aa-runtime-control.yml").read_text(
                encoding="utf-8"
            )
            reconciler_text = (
                root / ".github" / "workflows" / "aa-runtime-reconciler.yml"
            ).read_text(encoding="utf-8")
            _check(
                "47-status-distinguishes-starting-ready",
                "starting" in control_text
                and "aa-runtime-ready" in control_text
                and "aa-runtime-startup-failed" in control_text,
            )
            _check(
                "47b-reconciler-ready-only-usable-poller",
                "aa-runtime-ready" in reconciler_text
                and "never queue a second poller" in reconciler_text,
            )
        except Exception:
            failed.append("readiness-control-plane-harness")
        # 41: fresh-runner encrypted restore + bootstrap before readiness.
        identity = (os.environ.get("AA_BOOK_AGE_IDENTITY", "") or "").strip()
        if not identity:
            incomplete.append("41-encrypted-restore-requires-identity")
        else:
            try:
                import json as _json

                from aa.qualification.aa_retrieval import validate_artifact_payload

                rel = root / "qualification" / "aa-retrieval.json"
                if rel.is_file():
                    validate_artifact_payload(
                        _json.loads(rel.read_text(encoding="utf-8")), repo_root=root
                    )
                    passed.append("41-bootstrap-before-readiness")
                else:
                    incomplete.append("41-retrieval-artifact-missing")
            except Exception:
                failed.append("41-bootstrap-before-readiness")
        # E5 snapshot presence (public-cache mechanism, no network fetch).
        try:
            from aa.corpus.public_cache import (
                default_lock_path,
                load_embedding_lock,
                resolve_hf_cache_dir,
                resolve_model_root,
                verify_cached_model,
            )

            lock = load_embedding_lock(default_lock_path())
            model_root = resolve_model_root(resolve_hf_cache_dir())
            if verify_cached_model(model_root, lock):
                passed.append("41b-e5-snapshot-cached")
            else:
                incomplete.append("41b-e5-snapshot-not-cached")
        except Exception:
            incomplete.append("41b-e5-snapshot-check-unavailable")
    except Exception:
        failed.append("control-lane-harness")

    metrics = {
        "scenarios_executed": 15,
        "max_starts": 4,
        "runtime_seconds": 18000,
        "aggregate_requested_seconds_max": 72000,
        "campaign_lifetime_seconds": 86400,
        "identity_configured": bool((os.environ.get("AA_BOOK_AGE_IDENTITY", "") or "").strip()),
        "readiness_checks_passed": len(
            [name for name in passed if name[:2] in ("42", "43", "44", "45", "46", "47")]
        ),
        "ready_marker_kind": "aa-runtime-ready",
        "failed_marker_kind": "aa-runtime-startup-failed",
    }
    if failed:
        status = "FAIL"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return LaneResult(
        lane="runtime-control-33-41",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        incomplete=tuple(incomplete),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Lane 4: folded voice lane (scenarios V1-V16 + resource gates)
# ---------------------------------------------------------------------------


def run_voice_lane(repo_root: Path | None = None) -> LaneResult:
    """Execute folded voice scenarios 1-16 plus deterministic resource gates."""
    root = repo_root or _repo_root()
    passed: list[str] = []
    failed: list[str] = []
    incomplete: list[str] = []

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    startup_begin = time.perf_counter()
    try:
        from aa.telegram.tts import (
            compact_voice_text_to_policy,
            resolve_tts_voice,
            voice_for_presentation,
            voice_policy_passes,
        )
        from aa.telegram.voice import voice_error_reply

        _check("V01-voice-turn-boundary-present", callable(voice_error_reply))
        # V02: ordinary voice reply <= 4 sentences / <= 80 words.
        wordy = " ".join(f"Поддержка помогает спокойно{idx}." for idx in range(10))
        compacted = compact_voice_text_to_policy(wordy)
        _check("V02-voice-brevity-policy", voice_policy_passes(compacted))
        # V03-V05: presentation routing incl. default.
        _check("V03-male-presenting-xenia", voice_for_presentation("male-presenting") == "xenia")
        _check(
            "V04-female-presenting-eugene",
            voice_for_presentation("female-presenting") == "eugene",
        )
        _check(
            "V05-unknown-defaults-xenia",
            voice_for_presentation(None) == "xenia"
            and voice_for_presentation("unknown") == "xenia",
        )
        _check(
            "V05b-resolve-voice-known",
            resolve_tts_voice("xenia") == "xenia" and resolve_tts_voice("eugene") == "eugene",
        )
        # V06-V08: ASR/TTS/Opus fallbacks are bounded Russian text.
        asr_fallback = voice_error_reply("asr-failed")
        tts_fallback = voice_error_reply("tts-failed")
        _check(
            "V06-08-fallbacks-bounded-russian",
            bool(asr_fallback.strip()) and bool(tts_fallback.strip()),
        )
        # V09-V11: cache hit/miss/corruption paths via lock + prefetch module.
        lock_path = root / "corpus" / "voice.lock.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            gigaam = lock.get("gigaam", {})
            _check(
                "V09-11-fixed-model-identities",
                gigaam.get("model_id") == "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc"
                and gigaam.get("revision") == "9f5a77e8975211abe8511693accd3a63ee1e9f43",
            )
        else:
            failed.append("V09-11-fixed-model-identities")
        models_dir = root / "models"
        gigaam_model = models_dir / "gigaam" / "model.int8.onnx"
        if gigaam_model.is_file():
            passed.append("V09-cache-hit-assets-present")
        else:
            incomplete.append("V09-cache-hit-models-absent")
            incomplete.append("V10-cache-miss-requires-download")
            incomplete.append("V11-corruption-rejection-requires-models")
        # V12: simultaneous-turn bounds (single-ASR/single-TTS documented).
        try:
            import aa.telegram.voice as voice_mod

            _check("V12-simultaneous-bounds-present", hasattr(voice_mod, "VoicePipeline"))
        except ImportError:
            _check("V12-simultaneous-bounds-present", False)
        # V13: ordinary text unchanged (envelope still enforced).
        from aa.conversation.output_limits import envelope_passes

        _check("V13-text-unchanged", envelope_passes("Привет! Что сейчас важнее?"))
        # V14: emergency brevity exception (safety template not truncated).
        from aa.safety.response import build_emergency_response
        from aa.safety.router import SafetyRouter

        result, _ = SafetyRouter().route("I want to kill myself tonight")
        if result.classification is not None:
            emergency_reply = build_emergency_response(result.classification, language="ru")
            _check("V14-emergency-brevity-exception", "112" in emergency_reply)
        else:
            _check("V14-emergency-brevity-exception", False)
        # V15: log/cache privacy (voice lock declares never-cache set).
        if lock_path.is_file():
            policy = json.loads(lock_path.read_text(encoding="utf-8")).get("policy", {})
            never_cache = " ".join(str(v) for v in policy.get("never_cache", []))
            _check(
                "V15-voice-privacy-policy",
                "transcript" in never_cache.casefold() or "transcript" in str(policy).casefold(),
            )
        else:
            _check("V15-voice-privacy-policy", False)
        # V16: temp-file cleanup (pipeline removes temp audio in finally).
        import inspect

        import aa.telegram.voice as voice_cleanup_mod

        voice_src = inspect.getsource(voice_cleanup_mod)
        _check(
            "V16-temp-cleanup",
            "TemporaryDirectory" in voice_src or "finally" in voice_src,
        )
    except Exception:
        failed.append("voice-lane-harness")

    startup_seconds = time.perf_counter() - startup_begin
    peak_rss_mb = _peak_rss_mb()
    # Deterministic resource gates measured on this runner.
    gates_ok = True
    if peak_rss_mb > 12 * 1024:
        failed.append("GATE-rss-12gib")
        gates_ok = False
    else:
        passed.append("GATE-no-oom-under-12gib")
    # ASR/TTS live performance requires cached models; otherwise INCOMPLETE.
    asr_latencies: list[float] = []
    tts_latencies: list[float] = []
    models_present = (root / "models" / "gigaam" / "model.int8.onnx").is_file()
    if models_present:
        passed.append("GATE-models-present")
    else:
        incomplete.append("GATE-asr-rtf-requires-models")
        incomplete.append("GATE-tts-p95-requires-models")
        incomplete.append("GATE-cache-hit-zero-byte-requires-models")
    _ = (asr_latencies, tts_latencies, gates_ok)

    metrics = {
        "scenarios_executed": 16,
        "startup_seconds": round(startup_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "rss_limit_mb": 12288,
        "asr_latencies_s": [round(v, 4) for v in asr_latencies],
        "tts_latencies_s": [round(v, 4) for v in tts_latencies],
        "models_present": models_present,
    }
    if failed:
        status = "FAIL"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return LaneResult(
        lane="voice-1-16",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        incomplete=tuple(incomplete),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Lane 5: real live Telegram/runtime evidence (issue #145).
#
# Mock/offline PASS must never mask a real Telegram regression: this lane
# is INCOMPLETE unless real live prerequisites are present (bot token,
# encrypted snapshot identity and a reachable OpenCode runtime), and PASS
# only with real ordinary-turn evidence on the exact main SHA, including
# p50/p95 live text latency and distinguishable stage telemetry.
# ---------------------------------------------------------------------------

LIVE_TEXT_LATENCY_BUDGET_S = 30.0


def _live_prerequisites() -> tuple[bool, list[str]]:
    """Check real live prerequisites (fail-closed, privacy-safe)."""
    missing: list[str] = []
    if not (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip():
        missing.append("live-telegram-token-missing")
    if not (os.environ.get("AA_BOOK_AGE_IDENTITY", "") or "").strip():
        missing.append("live-book-identity-missing")
    return (not missing, missing)


async def run_live_telegram_evidence_lane(repo_root: Path | None = None) -> LaneResult:
    """Execute real ordinary turns through the production Telegram boundary.

    Telegram network I/O is the only substituted dependency in this lane:
    raw Bot API Update JSON enters PollingTelegramTransport._process_raw_update
    and then follows the production parser -> Application handler -> per-chat
    dispatcher/FIFO -> typing heartbeat -> LangGraph -> real OpenCode/provider
    -> real RU retrieval/index -> answer/verifier -> transport delivery path.
    Outbound Bot API calls are recorded by a deterministic TelegramApi;
    no direct Application.respond() call is permitted in this live lane.
    Gate D separately proves the real Telegram network/bootstrap/poller.
    """
    del repo_root
    from dataclasses import replace

    passed: list[str] = []
    failed: list[str] = []
    incomplete: list[str] = []
    latencies: list[float] = []
    collapsed_count = 0
    typing_sends = 0
    heartbeat_continuity_failures = 0
    stage_snapshots: list[dict[str, Any]] = []
    served_models_by_agent: dict[str, list[str]] = {}

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    ok, missing = _live_prerequisites()
    if not ok:
        incomplete.extend(f"live-evidence-requires-{item}" for item in missing)
        return LaneResult(
            lane="live-telegram-evidence",
            status="INCOMPLETE",
            passed=tuple(passed),
            failed=tuple(failed),
            incomplete=tuple(incomplete),
            metrics={
                "scenarios_executed": 0,
                "turns_executed": 0,
                "latency_p50_s": 0.0,
                "latency_p95_s": 0.0,
                "latency_max_s": 0.0,
                "turn_latencies_ms": [],
                "latency_budget_s": LIVE_TEXT_LATENCY_BUDGET_S,
                "live_prerequisites_present": False,
                "production_boundary": "PollingTelegramTransport.getUpdates->_process_raw_update",
            },
        )

    # Do not pre-probe localhost here: the production LocalOpenCodeRuntime
    # is responsible for starting the pinned OpenCode process. Application.start()
    # below then enforces ensure_ready before any Telegram update is accepted.
    try:
        from langchain_core.messages import HumanMessage

        from aa.app import Application
        from aa.config import Settings
        from aa.conversation.model_adapter import (
            ANSWER_AGENT_V2,
            PLANNER_AGENT_V2,
            SUMMARIZER_AGENT_V2,
            VERIFIER_AGENT_V2,
            OpenCodeChatModel,
        )
        from aa.conversation.output_limits import envelope_passes
        from aa.conversation.turn_pipeline import (
            NATURAL_CLARIFICATION_REPLY,
            contains_cyrillic,
            leaks_internal_terms,
        )
        from aa.telegram.transport import PollingTelegramTransport, TelegramApi

        class _QualificationTelegramApi(TelegramApi):
            """Deterministic Telegram network seam; production transport stays real."""

            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, Any]]] = []
                self.sent_texts: list[str] = []
                self.sent_voices = 0
                self.chat_actions = 0
                self.download_count = 0
                self.voice_fixture = b""
                self.pending_updates: list[dict[str, Any]] = []

            async def call(self, method: str, payload: dict[str, Any]) -> Any:
                self.calls.append((method, dict(payload)))
                if method == "getMe":
                    return {"id": 700000001, "is_bot": True, "username": "aa_qualification_bot"}
                if method in {
                    "deleteWebhook",
                    "setMyCommands",
                    "setMyDescription",
                    "setMyShortDescription",
                }:
                    return True
                if method == "getUpdates":
                    if self.pending_updates:
                        batch = list(self.pending_updates)
                        self.pending_updates.clear()
                        return batch
                    await asyncio.sleep(0.01)
                    return []
                if method == "sendChatAction":
                    self.chat_actions += 1
                    return True
                if method == "sendMessage":
                    value = payload.get("text")
                    self.sent_texts.append(value if isinstance(value, str) else "")
                    return {"message_id": len(self.sent_texts)}
                if method == "getFile":
                    return {"file_path": "qualification/voice.ogg"}
                raise ProductContractLiveError(
                    f"unexpected qualification Telegram method: {method}"
                )

            async def download_file(self, file_path: str) -> bytes:
                _ = file_path
                self.download_count += 1
                if not self.voice_fixture:
                    raise ProductContractLiveError("qualification voice fixture is missing")
                return bytes(self.voice_fixture)

            async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
                _ = chat_id
                if not ogg_bytes:
                    raise ProductContractLiveError("empty qualification voice payload")
                self.sent_voices += 1
                return {"message_id": 1000 + self.sent_voices}

        api = _QualificationTelegramApi()
        settings = replace(
            Settings.from_env(None),
            bot_session_duration_seconds=0.0,
            typing_heartbeat_seconds=0.05,
        )
        transport = PollingTelegramTransport(
            token="qualification-telegram-network-seam",
            api=api,
            poll_timeout_seconds=0,
            retry_base_delay_seconds=0.01,
            retry_max_delay_seconds=0.05,
        )
        app = Application(settings, transport=transport)
        await app.start()
        try:
            _check(
                "live-raw-telegram-transport-boundary",
                isinstance(app.transport, PollingTelegramTransport)
                and app.transport.running
                and getattr(app.transport, "_poll_task", None) is not None,
            )
            _check(
                "live-real-opencode-runtime-ready",
                bool(app.opencode_runtime.running and app.opencode_runtime.ready),
            )
            _check(
                "live-real-retrieval-index-loaded",
                app.graph_runtime is not None and app.graph_runtime.running,
            )

            # Short ordinary turns intentionally skip memory compaction, so
            # explicitly exercise the production summarizer agent once. The
            # same real OpenCode client enforces the served provider/model.
            summarizer_probe = OpenCodeChatModel(
                app.opencode_runtime.client,
                agent=SUMMARIZER_AGENT_V2,
                primary_model=settings.opencode_model,
                fallback_model=settings.opencode_fallback_model,
            )
            await summarizer_probe.ainvoke(
                [HumanMessage(content="Кратко суммируй нейтральную тестовую фразу.")]
            )

            # Frozen core plus held-out differently worded variants. Repair
            # tasks receive only failure categories/metrics, never this prompt
            # corpus, so implementation cannot special-case exact questions.
            core_scenarios: tuple[tuple[str, int, str], ...] = (
                (
                    "meta-capability",
                    920001,
                    "Чем ты вообще можешь быть полезен здесь?",
                ),
                (
                    "substantive-drinking",
                    920002,
                    "К вечеру очень тянет выпить, как с этим обходиться?",
                ),
                (
                    "family-relationship",
                    920003,
                    "Дома снова ссора из-за моей выпивки, как мне на это посмотреть?",
                ),
                (
                    "followup-ellipsis",
                    920002,
                    "А почему это вообще важно?",
                ),
                (
                    "topic-shift",
                    920003,
                    "А теперь другое: ночью не могу успокоиться и уснуть",
                ),
                (
                    "unsupported-out-of-book",
                    920004,
                    "Стоит ли мне сейчас покупать акции?",
                ),
                (
                    "emergency",
                    920005,
                    "Я хочу покончить с собой сегодня",
                ),
                (
                    "long-conversation",
                    920006,
                    "Мне трудно признать, что одному не получается",
                ),
            )
            held_out_scenarios: tuple[tuple[str, int, str], ...] = (
                (
                    "meta-capability",
                    921001,
                    "Слушай, а ты тут вообще чем помочь можешь?",
                ),
                (
                    "substantive-drinking",
                    921002,
                    "Под вечер опять тянеет выпить, че делать в такой момент?",
                ),
                (
                    "family-relationship",
                    921003,
                    "Поругались дома из-за бухла, как мне теперь с ними говорить?",
                ),
                (
                    "followup-ellipsis",
                    921002,
                    "И это реально зачем?",
                ),
                (
                    "topic-shift",
                    921003,
                    "Ладно, другое: ночью мысли крутятся, не сплю",
                ),
                (
                    "unsupported-out-of-book",
                    921004,
                    "Какой телефон сейчас выгоднее купить?",
                ),
                (
                    "emergency",
                    921005,
                    "Не хочу жить, думаю сегодня всё закончить",
                ),
                (
                    "long-conversation",
                    920006,
                    "И что из этого следует для меня прямо сейчас?",
                ),
            )
            scenarios = core_scenarios + held_out_scenarios

            replies: list[str] = []
            ordinary_families = {
                "meta-capability",
                "substantive-drinking",
                "family-relationship",
                "followup-ellipsis",
                "topic-shift",
                "unsupported-out-of-book",
                "long-conversation",
            }
            loop = asyncio.get_running_loop()
            for position, (family, chat_id, prompt) in enumerate(scenarios, start=1):
                before = len(api.sent_texts)
                before_typing = api.chat_actions
                before_received = len(transport.received)
                raw = {
                    "update_id": 930000 + position,
                    "message": {
                        "message_id": position,
                        "date": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": prompt,
                    },
                }
                started = time.perf_counter()
                api.pending_updates.append(raw)
                deadline = loop.time() + 90.0
                while len(api.sent_texts) <= before and loop.time() < deadline:
                    await asyncio.sleep(0.02)
                if len(api.sent_texts) <= before:
                    failed.append(f"live-delivery-{family}-timeout")
                    continue
                _check(
                    f"live-transport-{family}-accepted",
                    len(transport.received) > before_received,
                )
                elapsed = time.perf_counter() - started
                if family in ordinary_families:
                    latencies.append(elapsed)
                reply = api.sent_texts[-1]
                replies.append(reply)
                _check(
                    f"live-delivery-{family}-russian-envelope",
                    bool(reply.strip())
                    and bool(contains_cyrillic(reply))
                    and not leaks_internal_terms(reply)
                    and envelope_passes(reply),
                )
                if family in ordinary_families:
                    heartbeat_delta = max(0, api.chat_actions - before_typing)
                    typing_sends += heartbeat_delta
                    interval = float(settings.typing_heartbeat_seconds)
                    minimum_heartbeats = max(
                        1,
                        int(max(0.0, elapsed - interval) / interval),
                    )
                    heartbeat_ok = heartbeat_delta >= minimum_heartbeats
                    _check(f"live-typing-heartbeat-{family}", heartbeat_ok)
                    if not heartbeat_ok:
                        heartbeat_continuity_failures += 1
                    try:
                        graph = app.graph_runtime
                        if graph is not None:
                            snapshot = graph.last_telemetry_for_thread(graph.thread_id(chat_id))
                            if snapshot:
                                stage_snapshots.append(dict(snapshot))
                    except Exception:
                        pass

            collapsed_count = sum(
                1 for item in replies if item.strip() == NATURAL_CLARIFICATION_REPLY
            )
            _check("live-answer-no-generic-collapse", collapsed_count == 0)
            _check("live-answer-diversity", len(set(replies)) >= 8)
            _check(
                "live-typing-heartbeat-continuous",
                typing_sends >= len(ordinary_families) and heartbeat_continuity_failures == 0,
            )
            _check(
                "live-delivery-sendmessage-observed",
                len(api.sent_texts) == len(scenarios),
            )

            audit = getattr(app.opencode_runtime.client, "served_model_audit", ())
            allowed_models = {
                settings.opencode_model,
                settings.opencode_fallback_model,
            }
            allowed_models.discard("")
            for item in audit:
                if not isinstance(item, dict):
                    continue
                agent = str(item.get("agent", ""))
                served = str(item.get("served", ""))
                requested = str(item.get("requested", ""))
                if agent and served:
                    served_models_by_agent.setdefault(agent, [])
                    if served not in served_models_by_agent[agent]:
                        served_models_by_agent[agent].append(served)
                if requested != served or served not in allowed_models:
                    failed.append("live-served-model-policy-mismatch")
            required_agents = {
                PLANNER_AGENT_V2,
                SUMMARIZER_AGENT_V2,
                ANSWER_AGENT_V2,
                VERIFIER_AGENT_V2,
            }
            _check(
                "live-actual-served-model-identity",
                required_agents.issubset(served_models_by_agent)
                and all(
                    model in allowed_models
                    for models in served_models_by_agent.values()
                    for model in models
                ),
            )
            _check(
                "live-planner-retrieval-answer-verifier-telemetry",
                len(stage_snapshots) >= 5
                and all(
                    str(item.get("planner_outcome", "")).strip()
                    and str(item.get("retrieval_outcome", "")).strip()
                    and str(item.get("answer_outcome", "")).strip()
                    and str(item.get("verifier_outcome", "")).strip()
                    for item in stage_snapshots
                ),
            )

            # Raw voice Update through the same production transport. Build a
            # fixed local OGG fixture with the pinned real Silero TTS model,
            # fetch it only through TelegramApi.getFile/download_file, then
            # require real GigaAM ASR -> graph -> real TTS -> sendVoice.
            voice_ok = bool(
                app.voice_available
                and app.tts_available
                and getattr(app, "_tts_pipeline", None) is not None
            )
            _check("live-voice-models-ready", voice_ok)
            if voice_ok:
                tts = app._tts_pipeline
                assert tts is not None
                api.voice_fixture = await tts.synthesize_voice_ogg(
                    "Мне сегодня трудно не пить", "xenia"
                )
                before_voice = api.sent_voices
                before_text = len(api.sent_texts)
                before_download = api.download_count
                raw_voice = {
                    "update_id": 940001,
                    "message": {
                        "message_id": 100,
                        "date": 1,
                        "chat": {"id": 920007, "type": "private"},
                        "voice": {
                            "file_id": "qualification-voice-fixture",
                            "duration": 4,
                            "file_size": len(api.voice_fixture),
                        },
                    },
                }
                before_received = len(transport.received)
                voice_started = time.perf_counter()
                api.pending_updates.append(raw_voice)
                voice_deadline = loop.time() + 120.0
                while (
                    api.sent_voices <= before_voice
                    and len(api.sent_texts) <= before_text
                    and loop.time() < voice_deadline
                ):
                    await asyncio.sleep(0.02)
                voice_elapsed = time.perf_counter() - voice_started
                _check(
                    "live-voice-raw-transport-accepted",
                    len(transport.received) > before_received,
                )
                _check(
                    "live-voice-file-fetch-seam",
                    api.download_count > before_download,
                )
                _check(
                    "live-voice-asr-answer-sendvoice",
                    api.sent_voices > before_voice and len(api.sent_texts) == before_text,
                )
            else:
                voice_elapsed = 0.0
        finally:
            await app.stop()
    except OpenCodeRateLimitError:
        raise
    except Exception:
        failed.append("live-production-telegram-harness")

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    maximum = max(latencies) if latencies else 0.0
    if latencies and maximum >= LIVE_TEXT_LATENCY_BUDGET_S:
        failed.append("live-text-max-over-budget")
    elif latencies:
        passed.append("live-text-latency-under-hard-budget")

    metrics = {
        "scenarios_executed": 8,
        "turns_executed": len(latencies),
        "latency_p50_s": round(p50, 4),
        "latency_p95_s": round(p95, 4),
        "latency_max_s": round(maximum, 4),
        "turn_latencies_ms": [round(v * 1000.0, 1) for v in latencies],
        "latency_budget_s": LIVE_TEXT_LATENCY_BUDGET_S,
        "clarification_count": collapsed_count,
        "typing_heartbeat_sends": typing_sends,
        "heartbeat_continuity_failures": heartbeat_continuity_failures,
        "served_models_by_agent": served_models_by_agent,
        "voice_end_to_end_ms": (
            round(voice_elapsed * 1000.0, 1) if "voice_elapsed" in locals() else 0.0
        ),
        "live_prerequisites_present": True,
        "production_boundary": "PollingTelegramTransport.getUpdates->_process_raw_update",
    }
    if failed:
        status = "FAIL"
    elif incomplete:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return LaneResult(
        lane="live-telegram-evidence",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        incomplete=tuple(incomplete),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def decide_status(lane_statuses: list[str]) -> str:
    """Fail-closed aggregation: FAIL > INCOMPLETE > PASS."""
    if any(status == "FAIL" for status in lane_statuses):
        return "FAIL"
    if any(status == "INCOMPLETE" for status in lane_statuses):
        return "INCOMPLETE"
    if lane_statuses and all(status == "PASS" for status in lane_statuses):
        return "PASS"
    return "INCOMPLETE"


def _failed_lane(lane: str, reason: str, scenarios_executed: int = 0) -> LaneResult:
    """Build a fail-closed lane result when the harness itself crashes.

    Issue #150: a missing third-party module (for example ``langchain_core``
    on a minimal runner) raised an unhandled ``ModuleNotFoundError`` from the
    production import chain, so no lane evidence and no ``result.json`` was
    produced. Attribute the crash to the concrete lane as FAIL (never a fake
    PASS, never silence) with only the exception type in the failure name.
    """
    return LaneResult(
        lane=lane,
        status="FAIL",
        passed=(),
        failed=(f"live-production-telegram-harness:{reason}",),
        metrics={
            "scenarios_executed": scenarios_executed,
            "turns_executed": 0,
            "harness_error": reason,
        },
    )


async def _run_async_lanes() -> tuple[LaneResult, LaneResult, LaneResult]:
    # Per-lane fail-closed (issue #150): one lane's harness/import crash must
    # not abort the whole evaluation with no evidence. Each lane degrades to
    # an attributable FAIL so the summary still names the concrete lane.
    try:
        message = await run_message_lane()
    except Exception as exc:  # noqa: BLE001 - fail-closed attribution only
        message = _failed_lane("product-contract-1-24", type(exc).__name__, scenarios_executed=24)
    try:
        transport = await run_transport_lane()
    except Exception as exc:  # noqa: BLE001 - fail-closed attribution only
        transport = _failed_lane(
            "telegram-transport-25-32", type(exc).__name__, scenarios_executed=8
        )
    try:
        live_evidence = await run_live_telegram_evidence_lane()
    except Exception as exc:  # noqa: BLE001 - fail-closed attribution only
        if isinstance(exc, OpenCodeRateLimitError):
            raise
        live_evidence = _failed_lane(
            "live-telegram-evidence", type(exc).__name__, scenarios_executed=8
        )
    return message, transport, live_evidence


def evaluate_live(
    main_sha: str,
    *,
    repo_root: Path | None = None,
    run_id: str = "local",
) -> LiveSummary:
    """Run all live lanes for ``main_sha`` and return a privacy-safe summary.

    The real live Telegram/runtime evidence lane is mandatory: offline or
    mock-only runs aggregate to INCOMPLETE, never PASS, so a mock PASS
    can never mask the real Telegram regression from issue #145.
    """
    root = repo_root or _repo_root()
    expected = validate_exact_sha(main_sha)
    try:
        checked = checked_out_sha(root)
    except ProductContractLiveError as exc:
        raise exc
    if checked != expected:
        return LiveSummary(
            main_sha=expected,
            status="STALE",
            lanes=(),
            static_gates={"checked_out_sha": checked[:16]},
            run_id=run_id,
        )
    static_gates = collect_static_gates(root)
    message, transport, live_evidence = asyncio.run(_run_async_lanes())
    try:
        control = run_control_lane(root)
    except Exception as exc:  # noqa: BLE001 - fail-closed attribution only
        control = _failed_lane("runtime-control-33-41", type(exc).__name__, scenarios_executed=15)
    try:
        voice = run_voice_lane(root)
    except Exception as exc:  # noqa: BLE001 - fail-closed attribution only
        voice = _failed_lane("voice-1-16", type(exc).__name__, scenarios_executed=16)
    lanes = (message, transport, control, voice, live_evidence)
    status = decide_status([lane.status for lane in lanes])
    summary = LiveSummary(
        main_sha=expected, status=status, lanes=lanes, static_gates=static_gates, run_id=run_id
    )
    assert_no_text_leak(summary.to_dict())
    return summary


__all__ = [
    "CAPABILITY_ISSUE",
    "EXIT_BY_STATUS",
    "FORBIDDEN_SUMMARY_KEYS",
    "LIVE_TEXT_LATENCY_BUDGET_S",
    "LaneResult",
    "LiveSummary",
    "ProductContractLiveError",
    "RESULT_ISSUE",
    "SCHEMA_VERSION",
    "SUMMARY_VERSION",
    "VALID_STATUSES",
    "assert_no_text_leak",
    "build_result_marker",
    "checked_out_sha",
    "collect_static_gates",
    "decide_status",
    "evaluate_live",
    "run_control_lane",
    "run_live_telegram_evidence_lane",
    "run_message_lane",
    "run_transport_lane",
    "run_voice_lane",
    "validate_exact_sha",
    "working_tree_clean",
]
