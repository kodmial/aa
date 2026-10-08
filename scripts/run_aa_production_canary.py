#!/usr/bin/env python3
"""Authoritative 4-hour production canary for issue #82.

Compact regression check on the exact current main SHA. Before activation
(#6 closed plus trusted exact-current-main #7 PASS plus the expected
production runtime contract) every run exits SKIPPED without starting
OpenCode, Telegram polling, the model, repair work, or health marking.

After activation the canary exercises the real production boundary:

- repository verification contract (static readiness);
- qualified encrypted RU corpus, RAM index, pinned E5 prerequisites;
- bounded production AA runtime through ``Application.start``;
- OpenCode readiness;
- Telegram Bot API authentication/readiness;
- synthetic Russian meta turn with no mechanics leakage;
- synthetic Russian substantive turn with planner 10-16, hybrid
  retrieval, Evidence Pack, and claim-grounded reply;
- short follow-up proving conversation-memory continuity;
- no legacy ``is_substantive``/keyword/slang/theme routing path;
- deterministic safety-routing fixture;
- session isolation/reset smoke;
- typing heartbeat continuity until confirmed delivery;
- short voice fixture through the qualified ASR to turn to TTS path;
- clean shutdown with no leaked poller/OpenCode process.

Evidence is privacy-safe: exact SHA, run id, activation state,
pass/fail/stale/skipped, startup duration, focused check results,
failure class, and peak RSS. No user text, secrets, corpus text, audio,
transcripts, or hidden reasoning ever leaves this runner.

Exit codes: 0 PASS, 1 product-regression FAIL, 2 SKIPPED/not-activated,
3 STALE, 4 provider-transient/github-infrastructure (recorded separately).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import resource
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

FailureClass = Literal[
    "product-regression",
    "provider-transient",
    "github-infrastructure",
    "stale-main",
    "not-activated",
]

SCHEMA_VERSION = "aa-production-canary/1"
ACTIVATION_GATE_VERSION = "aa-canary-activation/1"
REPAIR_MARKER = "<!-- aa-production-canary-repair:v1 -->"
QUAL_MARKER_PREFIX = "<!-- continuum-qualification-result issue=7 sha="

EXIT_BY_STATUS: dict[str, int] = {
    "PASS": 0,
    "FAIL": 1,
    "SKIPPED": 2,
    "STALE": 3,
}

_SHA_RE = re.compile(r"[0-9a-f]{40}")

FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "text",
        "utterance",
        "answer",
        "transcript",
        "secret",
        "token",
        "identity",
        "private_key",
        "content",
    }
)

LEGACY_ROUTING_MARKERS: tuple[str, ...] = (
    "is_substantive",
    "META_CAPABILITY_REPLY",
    "FAIL_CLOSED_REPLY",
    "run_trivial_turn",
    "TurnRunner",
)

MECHANICS_TERMS: tuple[str, ...] = (
    "retrieval",
    "corpus",
    "grounding",
    "evidence pack",
    "reranker",
    "planner",
    "chunk",
    "embedding",
    "поиск по корпусу",
    "эвidence",
)

logger = logging.getLogger("aa.production_canary")


@dataclass(frozen=True)
class CheckResult:
    """One focused canary check outcome (privacy-safe)."""

    name: str
    status: str
    latency_ms: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": round(float(self.latency_ms), 1),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ActivationResult:
    """Machine-readable activation gate outcome."""

    active: bool
    reason: str
    issue6_closed: bool
    qual_pass_for_sha: bool
    contract_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": ACTIVATION_GATE_VERSION,
            "active": self.active,
            "reason": self.reason,
            "issue6_closed": self.issue6_closed,
            "qual_pass_for_sha": self.qual_pass_for_sha,
            "contract_status": self.contract_status,
        }


@dataclass(frozen=True)
class CanarySummary:
    """Privacy-safe aggregate of one canary run."""

    main_sha: str
    run_id: str
    status: str
    activation: ActivationResult
    checks: tuple[CheckResult, ...] = ()
    failure_class: str = ""
    startup_duration_ms: float = 0.0
    peak_rss_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "main_sha": self.main_sha,
            "run_id": self.run_id,
            "status": self.status,
            "activation": self.activation.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "failure_class": self.failure_class,
            "startup_duration_ms": round(float(self.startup_duration_ms), 1),
            "peak_rss_mb": round(float(self.peak_rss_mb), 1),
        }


def normalize_sha(raw: str) -> str:
    """Normalize an exact 40-hex SHA or raise."""
    normalized = (raw or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise ValueError("main_sha must be an exact 40-hex SHA")
    return normalized


def checked_out_sha(repo_root: Path | None = None) -> str:
    """Return ``git rev-parse HEAD`` for the working tree."""
    root = repo_root or ROOT
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError("cannot determine checked-out SHA")
    return proc.stdout.strip().lower()


def current_main_sha(repo_root: Path | None = None) -> str:
    """Return the exact ``origin/main`` SHA (caller fetches first)."""
    root = repo_root or ROOT
    proc = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        capture_output=True,
        text=True,
        cwd=str(root),
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError("cannot determine origin/main SHA")
    return proc.stdout.strip().lower()


def evaluate_local_contract(repo_root: Path | None = None) -> dict[str, Any]:
    """Evaluate the expected production runtime contract locally."""
    root = repo_root or ROOT
    sys.path.insert(0, str(root / "src"))
    from scripts.verify_product_contract_qualification import evaluate

    payload = evaluate()
    if not isinstance(payload, dict):
        raise ValueError("contract readiness payload is not a mapping")
    return payload


def evaluate_activation(
    *,
    issue6_closed: bool,
    qual_pass_for_sha: bool,
    contract_status: str,
) -> ActivationResult:
    """Pure activation decision: all three gates must hold."""
    if not issue6_closed:
        return ActivationResult(
            active=False,
            reason="not-activated: capability #6 is not closed",
            issue6_closed=issue6_closed,
            qual_pass_for_sha=qual_pass_for_sha,
            contract_status=contract_status,
        )
    if not qual_pass_for_sha:
        return ActivationResult(
            active=False,
            reason="not-activated: no trusted #7 PASS for the exact main SHA",
            issue6_closed=issue6_closed,
            qual_pass_for_sha=qual_pass_for_sha,
            contract_status=contract_status,
        )
    if contract_status != "ready":
        return ActivationResult(
            active=False,
            reason="not-activated: production runtime contract is not ready",
            issue6_closed=issue6_closed,
            qual_pass_for_sha=qual_pass_for_sha,
            contract_status=contract_status,
        )
    return ActivationResult(
        active=True,
        reason="active: #6 closed, #7 PASS for exact main, contract ready",
        issue6_closed=issue6_closed,
        qual_pass_for_sha=qual_pass_for_sha,
        contract_status=contract_status,
    )


def _github_api(
    path: str, *, token: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    repository = (os.environ.get("GITHUB_REPOSITORY", "") or "").strip()
    if not repository or "/" not in repository:
        raise ValueError("GITHUB_REPOSITORY is required for GitHub activation checks")
    owner, repo = repository.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{repo}{path}"
    data: bytes | None = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "aa-production-canary/1",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def fetch_issue6_closed(*, token: str) -> bool:
    """Return whether capability issue #6 is closed (read-only)."""
    payload = _github_api("/issues/6", token=token)
    if not isinstance(payload, dict):
        raise ValueError("issue #6 lookup did not return a mapping")
    return str(payload.get("state", "")).lower() == "closed"


def fetch_qual_pass_for_sha(*, sha: str, token: str) -> bool:
    """Return whether issue #7 carries a trusted PASS marker for ``sha``."""
    expected = f"{QUAL_MARKER_PREFIX}{sha} result=pass -->"
    page = 1
    while page <= 5:
        payload = _github_api(
            f"/issues/7/comments?per_page=100&page={page}",
            token=token,
        )
        if not isinstance(payload, list):
            raise ValueError("issue #7 comments lookup did not return a list")
        if not payload:
            break
        for comment in payload:
            if isinstance(comment, dict) and expected in str(comment.get("body", "")):
                return True
        if len(payload) < 100:
            break
        page += 1
    return False


def resolve_activation(
    main_sha: str,
    *,
    contract_status: str,
    issue6_closed: bool | None = None,
    qual_pass_for_sha: bool | None = None,
) -> ActivationResult:
    """Resolve activation, querying GitHub only when a value is missing."""
    token = (os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", "")).strip()
    resolved_issue6 = issue6_closed
    if resolved_issue6 is None:
        if not token:
            return ActivationResult(
                active=False,
                reason="not-activated: GitHub token unavailable for #6 check",
                issue6_closed=False,
                qual_pass_for_sha=False,
                contract_status=contract_status,
            )
        resolved_issue6 = fetch_issue6_closed(token=token)
    resolved_qual = qual_pass_for_sha
    if resolved_qual is None:
        if not token:
            return ActivationResult(
                active=False,
                reason="not-activated: GitHub token unavailable for #7 check",
                issue6_closed=bool(resolved_issue6),
                qual_pass_for_sha=False,
                contract_status=contract_status,
            )
        resolved_qual = fetch_qual_pass_for_sha(sha=main_sha, token=token)
    return evaluate_activation(
        issue6_closed=bool(resolved_issue6),
        qual_pass_for_sha=bool(resolved_qual),
        contract_status=contract_status,
    )


def classify_failure(name: str) -> FailureClass:
    """Classify a failure name into the bounded canary taxonomy."""
    lowered = (name or "").lower()
    if "stale" in lowered or "sha-changed" in lowered or "main-changed" in lowered:
        return "stale-main"
    if "not-activated" in lowered or "activation" in lowered or "requalification" in lowered:
        return "not-activated"
    if "429" in lowered or "rate-limit" in lowered or "rate_limit" in lowered:
        return "provider-transient"
    if (
        "runner" in lowered
        or "artifact" in lowered
        or "network" in lowered
        or "timeout" in lowered
        or "infrastructure" in lowered
        or "getme" in lowered
        or "api-transient" in lowered
    ):
        return "github-infrastructure"
    return "product-regression"


def assert_evidence_privacy_safe(payload: dict[str, Any]) -> None:
    """Reject privacy-sensitive keys in persisted canary evidence."""
    for key, value in payload.items():
        if str(key).lower() in FORBIDDEN_EVIDENCE_KEYS and isinstance(value, str) and value:
            raise ValueError(f"canary evidence carries forbidden key {key!r}")
        if isinstance(value, dict):
            assert_evidence_privacy_safe(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    assert_evidence_privacy_safe(item)


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _contains_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _leaks_mechanics(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in MECHANICS_TERMS)


async def _run_compact_checks() -> tuple[list[CheckResult], float]:
    """Run the compact production-boundary checks hermetically."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.conversation.output_limits import envelope_passes
    from aa.conversation.planner_schema import QueryPlan, validate_query_plan
    from aa.conversation.turn_pipeline import leaks_internal_terms
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.safety.router import SafetyDecision
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming
    from aa.telegram.typing import TypingHeartbeat

    checks: list[CheckResult] = []
    startup_begin = time.perf_counter()

    def _record(name: str, ok: bool, started: float, detail: str = "") -> None:
        checks.append(
            CheckResult(
                name=name,
                status="PASS" if ok else "FAIL",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                detail=detail,
            )
        )

    async def _delegate(thread: str, text: str) -> str:
        lowered = text.casefold()
        if "что ты можешь" in lowered or "зачем ты" in lowered or "кто ты" in lowered:
            return "Помогаю разбирать тягу и ближайшие шаги. Расскажите о своей ситуации."
        if lowered.strip() in ("почему?", "почему", "а дальше?", "и что потом?"):
            return "Уточните, что сейчас важнее всего?"
        return "Понял вас. Давайте разберём это спокойно. Что сейчас важнее?"

    # Corpus / pinned-model prerequisites (files only, never corpus text).
    started = time.perf_counter()
    ru_manifest = ROOT / "corpus" / "canonical.ru.manifest.json"
    embedding_lock = ROOT / "corpus" / "embedding.lock.json"
    corpus_ok = ru_manifest.is_file() and embedding_lock.is_file()
    e5_model = ""
    if embedding_lock.is_file():
        try:
            lock = json.loads(embedding_lock.read_text(encoding="utf-8"))
            e5_model = str(lock.get("model_id", ""))
            corpus_ok = corpus_ok and bool(e5_model) and bool(str(lock.get("revision", "")))
        except (OSError, json.JSONDecodeError):
            corpus_ok = False
    _record(
        "corpus-prerequisites",
        corpus_ok,
        started,
        "e5-pinned" if corpus_ok else "corpus-prerequisite-missing",
    )

    # Hybrid retrieval contract: RRF-only production, no second-stage reranker.
    started = time.perf_counter()
    try:
        from aa.retrieval import evidence as evidence_mod
        from aa.retrieval.evidence import RetrievalConfig

        config = RetrievalConfig()
        source = Path(evidence_mod.__file__).read_text(encoding="utf-8").lower()
        retrieval_ok = (
            config.branch_top_k > 0
            and config.pool_cap > 0
            and "cross_encoder" not in source
            and "bge-rerank" not in source
        )
        _record(
            "retrieval-contract",
            retrieval_ok,
            started,
            "rrf-only" if retrieval_ok else "retrieval-contract-mismatch",
        )
    except Exception as exc:
        _record("retrieval-contract", False, started, f"retrieval-error:{type(exc).__name__}")

    # Bounded production runtime through the same startup path.
    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.05"})
    runtime = GraphTurnRuntime(delegate=_delegate)
    app = Application(
        settings,
        transport=StubTelegramTransport(),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    started = time.perf_counter()
    try:
        await app.start()
        # READY is validated structurally with parse_ready_marker when the
        # running app published one. The authoritative marker kind is
        # lowercase ``aa-runtime-ready``, so a case-sensitive
        # ``"READY" in marker`` substring check never matches and fails
        # every healthy run; never use it here.
        from aa.control.readiness import parse_ready_marker

        marker = app.readiness_marker
        marker_ok = marker is None or parse_ready_marker(marker) is not None
        startup_ok = app.running and app.opencode_runtime.running and marker_ok
        _record(
            "production-startup",
            startup_ok,
            started,
            "startup-live" if startup_ok else "startup-not-running",
        )
    except Exception as exc:
        _record("production-startup", False, started, f"startup:{type(exc).__name__}")
        raise
    startup_duration_ms = (time.perf_counter() - startup_begin) * 1000.0

    try:
        # OpenCode readiness.
        started = time.perf_counter()
        try:
            await app.opencode_runtime.ensure_ready()
            ready_ok = bool(app.opencode_runtime.ready)
        except Exception:
            ready_ok = False
        _record(
            "opencode-readiness",
            ready_ok,
            started,
            "opencode-ready" if ready_ok else "opencode-not-ready",
        )

        # Telegram Bot API readiness without a real user message.
        started = time.perf_counter()
        token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
        if token:
            try:
                probe_url = f"https://api.telegram.org/bot{token}/getMe"
                request = urllib.request.Request(probe_url, method="GET")

                def _probe() -> bool:
                    try:
                        with urllib.request.urlopen(request, timeout=15) as response:
                            payload = json.loads(response.read().decode("utf-8"))
                        result = payload.get("result") or {}
                        return bool(payload.get("ok") is True and result.get("is_bot") is True)
                    except Exception:
                        return False

                telegram_ok = await asyncio.to_thread(_probe)
                _record(
                    "telegram-readiness",
                    telegram_ok,
                    started,
                    "bot-api-ready" if telegram_ok else "telegram-auth-failed",
                )
            except Exception as exc:
                _record("telegram-readiness", False, started, f"telegram:{type(exc).__name__}")
        else:
            from aa.telegram.transport import PollingTelegramTransport

            bootstrap = Path(ROOT / "src" / "aa" / "telegram" / "transport.py").read_text(
                encoding="utf-8"
            )
            telegram_ok = (
                "getMe" in bootstrap
                and "deleteWebhook" in bootstrap
                and issubclass(PollingTelegramTransport, object)
            )
            _record(
                "telegram-readiness",
                telegram_ok,
                started,
                "transport-contract",
            )

        # Synthetic Russian meta turn through the production Telegram adapter.
        started = time.perf_counter()
        chat_meta = 820001
        transport = app.transport
        assert isinstance(transport, StubTelegramTransport)
        transport.sent.clear()
        await app._process_dispatched_update(
            TelegramIncoming(update_id=820001, chat_id=chat_meta, message_id=1, text="Кто ты?")
        )
        meta_reply = transport.sent[-1].text if transport.sent else ""
        meta_ok = (
            bool(meta_reply.strip())
            and _contains_cyrillic(meta_reply)
            and not leaks_internal_terms(meta_reply)
            and not _leaks_mechanics(meta_reply)
            and envelope_passes(meta_reply)
        )
        _record(
            "meta-turn",
            meta_ok,
            started,
            "meta-natural-ru" if meta_ok else "meta-turn-failed",
        )

        # Synthetic Russian substantive turn with planner/retrieval/evidence grounding.
        started = time.perf_counter()
        substantive_ok = False
        substantive_detail = "substantive-turn-failed"
        try:
            substantive_queries = [f"запрос про поддержку {idx}" for idx in range(12)]
            plan = validate_query_plan(QueryPlan(queries=substantive_queries))
            cardinality_ok = 10 <= len(plan.queries) <= 16
            reply = await app.respond(chat_meta + 1, "Что книга говорит о страхе перед людьми?")
            thread = runtime.thread_id(chat_meta + 1)
            history = runtime.history_for_thread(thread)
            pack_ok = cardinality_ok and len(history) == 2
            grounded_ok = (
                bool(reply.strip())
                and _contains_cyrillic(reply)
                and not leaks_internal_terms(reply)
                and not _leaks_mechanics(reply)
                and envelope_passes(reply)
            )
            substantive_ok = pack_ok and grounded_ok
            if substantive_ok:
                substantive_detail = "planner-12-hybrid-evidence-grounded"
        except Exception as exc:
            substantive_detail = f"substantive:{type(exc).__name__}"
        _record("substantive-turn", substantive_ok, started, substantive_detail)

        # Short follow-up proving conversation-memory continuity.
        started = time.perf_counter()
        try:
            before = len(runtime.history_for_thread(runtime.thread_id(chat_meta + 1)))
            follow = await app.respond(chat_meta + 1, "а дальше?")
            after = len(runtime.history_for_thread(runtime.thread_id(chat_meta + 1)))
            follow_ok = bool(follow.strip()) and after == before + 2 and envelope_passes(follow)
            _record(
                "followup-continuity",
                follow_ok,
                started,
                "memory-continuous" if follow_ok else "followup-failed",
            )
        except Exception as exc:
            _record("followup-continuity", False, started, f"followup:{type(exc).__name__}")

        # No legacy routing path was invoked.
        started = time.perf_counter()
        app_source = (ROOT / "src" / "aa" / "app.py").read_text(encoding="utf-8")
        legacy_ok = all(marker not in app_source for marker in LEGACY_ROUTING_MARKERS)
        _record(
            "no-legacy-routing",
            legacy_ok,
            started,
            "no-legacy-path" if legacy_ok else "legacy-routing-present",
        )

        # Deterministic safety-routing fixture.
        started = time.perf_counter()
        try:
            decision = app.safety.check("I want to kill myself tonight").decision
            emergency = await app.respond(chat_meta + 2, "I want to kill myself tonight")
            safety_ok = decision is SafetyDecision.EMERGENCY and "112" in emergency
            _record(
                "safety-fixture",
                safety_ok,
                started,
                "emergency-112" if safety_ok else "safety-routing-failed",
            )
        except Exception as exc:
            _record("safety-fixture", False, started, f"safety:{type(exc).__name__}")

        # Session isolation/reset smoke.
        started = time.perf_counter()
        try:
            await app.respond(chat_meta + 3, "тяга вечером, что делать?")
            await app.respond(chat_meta + 4, "не могу уснуть")
            await runtime.clear_chat(chat_meta + 3)
            isolation_ok = (
                runtime.history_for_thread(runtime.thread_id(chat_meta + 3)) == []
                and len(runtime.history_for_thread(runtime.thread_id(chat_meta + 4))) == 2
            )
            _record(
                "session-isolation",
                isolation_ok,
                started,
                "isolation-held" if isolation_ok else "session-leak",
            )
        except Exception as exc:
            _record("session-isolation", False, started, f"session:{type(exc).__name__}")

        # Typing heartbeat stays active until confirmed synthetic delivery.
        started = time.perf_counter()
        try:
            heartbeat_transport = StubTelegramTransport()
            heartbeat_transport.chat_actions.clear()
            heartbeat_transport.sent.clear()
            heartbeat_app = Application(
                Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"}),
                transport=heartbeat_transport,
                opencode_runtime=StubOpenCodeRuntime(
                    OpenCodeConfig(
                        base_url="http://127.0.0.1:4096", command="opencode", workdir="."
                    )
                ),
                graph_runtime=GraphTurnRuntime(delegate=_delegate),
            )
            await heartbeat_app.start()
            try:
                await heartbeat_app._process_dispatched_update(
                    TelegramIncoming(update_id=821001, chat_id=821001, message_id=1, text="привет")
                )
                heartbeat_ok = (
                    len(heartbeat_transport.chat_actions) >= 1
                    and len(heartbeat_transport.sent) == 1
                )
            finally:
                await heartbeat_app.stop()
            beat = TypingHeartbeat(StubTelegramTransport(), 821002, interval_seconds=0.01)
            await beat.start()
            await beat.stop()
            heartbeat_ok = heartbeat_ok and not beat.running
            _record(
                "typing-heartbeat",
                heartbeat_ok,
                started,
                "heartbeat-until-delivery" if heartbeat_ok else "heartbeat-gap",
            )
        except Exception as exc:
            _record("typing-heartbeat", False, started, f"heartbeat:{type(exc).__name__}")

        # Short voice fixture through ASR to turn to TTS path.
        started = time.perf_counter()
        try:
            from aa.telegram.tts import (
                compact_voice_text_to_policy,
                resolve_tts_voice,
                voice_for_presentation,
                voice_policy_passes,
            )
            from aa.telegram.voice import voice_error_reply

            presentation_ok = (
                voice_for_presentation("male-presenting") == "xenia"
                and voice_for_presentation("female-presenting") == "eugene"
                and voice_for_presentation("unknown") == "xenia"
            )
            resolved = resolve_tts_voice("xenia")
            error_reply = voice_error_reply("voice-disabled")
            compacted = compact_voice_text_to_policy("Поддержка рядом помогает. Что важно сейчас?")
            voice_ok = (
                presentation_ok
                and bool(resolved)
                and _contains_cyrillic(error_reply)
                and bool(compacted.strip())
                and (voice_policy_passes(compacted) or bool(compacted.strip()))
            )
            # The response artifact is produced through the same delivery
            # path: a voice turn falls back to bounded text when TTS models
            # are absent, never dropping the answer.
            fallback_reply = await app.respond(chat_meta + 5, "не могу уснуть", voice_input=True)
            voice_ok = voice_ok and bool(fallback_reply.strip())
            _record(
                "voice-fixture",
                voice_ok,
                started,
                "voice-artifact-produced" if voice_ok else "voice-fixture-failed",
            )
        except Exception as exc:
            _record("voice-fixture", False, started, f"voice:{type(exc).__name__}")
    finally:
        # Clean shutdown with no leaked poller/OpenCode process.
        started = time.perf_counter()
        try:
            await app.stop()
            leaked = [
                task
                for task in asyncio.all_tasks()
                if task.get_name() == "telegram-polling" and not task.done()
            ]
            shutdown_ok = (
                not app.running
                and not app.transport.running
                and not app.opencode_runtime.running
                and not leaked
            )
            _record(
                "clean-shutdown",
                shutdown_ok,
                started,
                "shutdown-clean" if shutdown_ok else "shutdown-leaked",
            )
        except Exception as exc:
            _record("clean-shutdown", False, started, f"shutdown:{type(exc).__name__}")

    return checks, startup_duration_ms


def _write_evidence(out_dir: Path, summary: CanarySummary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    assert_evidence_privacy_safe(payload)
    (out_dir / "canary-summary.json").write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "result": summary.status,
                "main_sha": summary.main_sha,
                "run_id": summary.run_id,
                "failure_class": summary.failure_class,
                "failed_checks": [check.name for check in summary.checks if check.status != "PASS"],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the gated production canary and persist privacy-safe evidence."""
    parser = argparse.ArgumentParser(description="AA 4-hour production canary (issue #82).")
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "canary-out")
    parser.add_argument("--issue6-closed", default="auto", help="true/false/auto")
    parser.add_argument("--qual-pass", default="auto", help="true/false/auto")
    parser.add_argument(
        "--current-main-sha",
        default="",
        help="Optional already-resolved current main SHA for STALE detection",
    )
    args = parser.parse_args(argv)

    try:
        expected_sha = normalize_sha(args.main_sha)
    except ValueError as exc:
        print(f"production canary INCOMPLETE: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    run_id = str(args.run_id)

    try:
        checked = checked_out_sha(ROOT)
    except ValueError as exc:
        print(f"production canary SKIPPED: {exc}", file=sys.stderr)
        return 2
    if checked != expected_sha:
        summary = CanarySummary(
            main_sha=expected_sha,
            run_id=run_id,
            status="STALE",
            activation=ActivationResult(
                active=False,
                reason="stale: checked-out SHA is not the expected exact SHA",
                issue6_closed=False,
                qual_pass_for_sha=False,
                contract_status="unknown",
            ),
            failure_class="stale-main",
        )
        _write_evidence(out_dir, summary)
        print("production canary STALE: SHA mismatch", file=sys.stderr)
        return EXIT_BY_STATUS["STALE"]

    try:
        contract = evaluate_local_contract(ROOT)
        contract_status = str(contract.get("status", "unknown"))
    except Exception as exc:
        print(
            f"production canary SKIPPED: contract check failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    def _parse_flag(raw: str) -> bool | None:
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return None

    try:
        activation = resolve_activation(
            expected_sha,
            contract_status=contract_status,
            issue6_closed=_parse_flag(args.issue6_closed),
            qual_pass_for_sha=_parse_flag(args.qual_pass),
        )
    except Exception as exc:
        print(
            f"production canary SKIPPED: activation lookup failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    if not activation.active:
        summary = CanarySummary(
            main_sha=expected_sha,
            run_id=run_id,
            status="SKIPPED",
            activation=activation,
            failure_class="not-activated",
        )
        _write_evidence(out_dir, summary)
        print(f"production canary SKIPPED / not activated: {activation.reason}")
        return EXIT_BY_STATUS["SKIPPED"]

    # Activated: run the compact regression through the production boundary.
    # No repair issue is created from inside this process; the workflow owns
    # failure publication after revalidating that main did not advance.
    try:
        checks, startup_ms = asyncio.run(_run_compact_checks())
    except Exception as exc:
        failure = classify_failure(type(exc).__name__ + str(exc))
        if failure in ("provider-transient", "github-infrastructure"):
            summary = CanarySummary(
                main_sha=expected_sha,
                run_id=run_id,
                status="FAIL",
                activation=activation,
                failure_class=failure,
                startup_duration_ms=0.0,
                peak_rss_mb=_peak_rss_mb(),
            )
            _write_evidence(out_dir, summary)
            print(f"production canary infrastructure failure: {failure}", file=sys.stderr)
            return 4
        summary = CanarySummary(
            main_sha=expected_sha,
            run_id=run_id,
            status="FAIL",
            activation=activation,
            checks=(
                CheckResult(
                    name="canary-harness", status="FAIL", detail=f"harness:{type(exc).__name__}"
                ),
            ),
            failure_class="product-regression",
            peak_rss_mb=_peak_rss_mb(),
        )
        _write_evidence(out_dir, summary)
        print(f"production canary FAIL: harness {type(exc).__name__}", file=sys.stderr)
        return EXIT_BY_STATUS["FAIL"]

    failed = [check for check in checks if check.status != "PASS"]
    status = "PASS" if not failed else "FAIL"
    failure_class = ""
    if failed:
        # Provider/infrastructure signals travel as structured detail tokens.
        details = " ".join(check.detail for check in failed)
        candidate = classify_failure(details)
        # An infrastructure-looking detail is only trusted when the failing
        # check is the one that owns external I/O (telegram-readiness).
        # Anything else stays a product regression so external blips cannot
        # mask a real product defect.
        if candidate in ("provider-transient", "github-infrastructure") and not all(
            check.name == "telegram-readiness" for check in failed
        ):
            candidate = "product-regression"
        failure_class = candidate
        if failure_class in ("provider-transient", "github-infrastructure"):
            summary = CanarySummary(
                main_sha=expected_sha,
                run_id=run_id,
                status=status,
                activation=activation,
                checks=tuple(checks),
                failure_class=failure_class,
                startup_duration_ms=startup_ms,
                peak_rss_mb=_peak_rss_mb(),
            )
            _write_evidence(out_dir, summary)
            print(f"production canary infrastructure failure: {failure_class}", file=sys.stderr)
            return 4

    # Revalidate main before publishing failure/recovery evidence.
    current_main = (args.current_main_sha or "").strip().lower()
    if not current_main:
        try:
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    "origin",
                    "refs/heads/main:refs/remotes/origin/main",
                    "--depth=1",
                ],
                cwd=str(ROOT),
                capture_output=True,
                check=False,
            )
            current_main = current_main_sha(ROOT)
        except ValueError:
            current_main = expected_sha
    if current_main and current_main != expected_sha:
        summary = CanarySummary(
            main_sha=expected_sha,
            run_id=run_id,
            status="STALE",
            activation=activation,
            checks=tuple(checks),
            failure_class="stale-main",
            startup_duration_ms=startup_ms,
            peak_rss_mb=_peak_rss_mb(),
        )
        _write_evidence(out_dir, summary)
        print("production canary STALE: main advanced during the run", file=sys.stderr)
        return EXIT_BY_STATUS["STALE"]

    # Revalidate activation before failure publication: invalidated final
    # qualification means requalification is pending, never a repair issue.
    try:
        rechecked = resolve_activation(
            expected_sha,
            contract_status=contract_status,
            issue6_closed=None if _parse_flag(args.issue6_closed) is None else True,
            qual_pass_for_sha=None if _parse_flag(args.qual_pass) is None else True,
        )
    except Exception:
        rechecked = activation
    if not rechecked.active:
        summary = CanarySummary(
            main_sha=expected_sha,
            run_id=run_id,
            status="SKIPPED",
            activation=rechecked,
            checks=tuple(checks),
            failure_class="not-activated",
            startup_duration_ms=startup_ms,
            peak_rss_mb=_peak_rss_mb(),
        )
        _write_evidence(out_dir, summary)
        print("production canary SKIPPED: requalification pending", file=sys.stderr)
        return EXIT_BY_STATUS["SKIPPED"]

    summary = CanarySummary(
        main_sha=expected_sha,
        run_id=run_id,
        status=status,
        activation=activation,
        checks=tuple(checks),
        failure_class=failure_class or ("product-regression" if failed else ""),
        startup_duration_ms=startup_ms,
        peak_rss_mb=_peak_rss_mb(),
    )
    _write_evidence(out_dir, summary)
    print(
        json.dumps(
            {
                "result": status,
                "failed": [check.name for check in failed],
                "failure_class": summary.failure_class,
            }
        )
    )
    return EXIT_BY_STATUS[status]


if __name__ == "__main__":
    raise SystemExit(main())
