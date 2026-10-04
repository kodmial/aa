"""Reusable exact-main Russian conversation benchmark harness (issue #72).

Evaluation infrastructure only. This module never performs the authoritative
#62 benchmark run and never changes product behavior to make a benchmark pass.

Execution boundary: bulk evaluation must call the same transport-independent
production turn boundary used by Telegram -- deterministic safety gating
(:class:`aa.safety.router.SafetyRouter`), session coordination
(:class:`aa.sessions.coordinator.SessionCoordinator`) and the planner /
retrieval / evidence / synthesis / grounding path behind the OpenCode runtime
(:class:`aa.opencode.client.OpenCodeClient`). Telegram network transport
itself is excluded: no hundreds of cases go through the public Bot API.

Generator input isolation: the runner exposes to the generator only the
current synthetic utterance, bounded prior turns from the same synthetic
chat/session, and normal production runtime context. Oracle labels, expected
safety decisions, provenance metadata, rubric tags, forbidden-inference
annotations, expected book regions and evaluator instructions never reach the
generator; :func:`assert_no_oracle_leak` enforces this mechanically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import posixpath
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import zstandard as zstd

from aa.corpus import age_v1
from aa.opencode.errors import (
    OpenCodeError,
    OpenCodeSessionNotFoundError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)
from aa.qualification.ru_realworld import (
    INPUT_FORBIDDEN_KEYS,
    INPUT_REL,
    ORACLE_REL,
    SESSION_RESET_CONTROL,
    VERSION_REL,
    find_repo_root,
    load_input,
    sha256_file,
    validate,
)

EVAL_SCHEMA_VERSION = "aa-conversation-eval/1"
RESULT_MARKER = "aa-conversation-eval-result"
RESULT_ISSUE = 62

DEFAULT_SHARD_COUNT = 4
MAX_SHARD_COUNT = 8
DEFAULT_MAX_PARALLEL = 2
MAX_PARALLEL_SHARDS = 4

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 30.0

# Generator must never see these evaluation-only keys. Anchored on the corpus
# input-leak boundary plus evaluator/rubric/provenance aliases so a new label
# fails closed instead of leaking silently.
FORBIDDEN_GENERATOR_KEYS = frozenset(
    set(INPUT_FORBIDDEN_KEYS)
    | {
        "provenance",
        "sources",
        "expected_answer",
        "assistant_response",
        "desired_response",
        "ideal_answer",
        "oracle",
        "label",
        "labels",
        "evaluator_instructions",
        "eval_instructions",
        "evaluator_prompt",
        "score",
        "grade",
        "expected_book_region",
        "expected_book_regions",
        "book_region",
    }
)

# Infrastructure/provider outcomes (never answer-quality failures).
INFRASTRUCTURE_MATCHERS = (
    "429",
    "too many requests",
    "provider-unavailable",
    "provider unavailable",
    "model-unavailable",
    "model unavailable",
    "freeusagelimit",
    "free-usage-limit",
    "overloaded",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)

_MARKER_RE = re.compile(
    r"<!--\s*aa-conversation-eval-result\s+"
    r"issue=(?P<issue>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"corpus=(?P<corpus>[0-9a-f]{64})\s+"
    r"result=(?P<result>complete|incomplete|stale)\s+"
    r"run=(?P<run>\S+?)\s*-->"
)


class ConversationEvalError(ValueError):
    """Raised when harness invariants fail (fails closed)."""


# ---------------------------------------------------------------------------
# Generator input isolation
# ---------------------------------------------------------------------------


def assert_no_oracle_leak(payload: Any, owner: str = "generator-input") -> None:
    """Reject any oracle/rubric/provenance key in generator-visible payload."""
    if isinstance(payload, dict):
        for key in payload:
            if key in FORBIDDEN_GENERATOR_KEYS:
                raise ConversationEvalError(
                    f"{owner}: generator leak: {key!r} must not reach generator"
                )
            if key == "expected_route" or key.startswith("expected_"):
                raise ConversationEvalError(
                    f"{owner}: generator leak: {key!r} must not reach generator"
                )
        for key, value in payload.items():
            assert_no_oracle_leak(value, f"{owner}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            assert_no_oracle_leak(item, f"{owner}[{index}]")


@dataclass(frozen=True)
class SingleCaseView:
    """Generator-visible single-turn case: ID plus utterance only."""

    case_id: str
    utterance: str

    def to_generator_payload(self) -> dict[str, str]:
        """Return the only payload the generator may receive for this case."""
        payload = {"id": self.case_id, "utterance": self.utterance}
        assert_no_oracle_leak(payload, self.case_id)
        return {"utterance": self.utterance}


@dataclass(frozen=True)
class JourneyTurnView:
    """One generator-visible journey entry (user turn or reset control)."""

    turn: int
    kind: str  # "user" | "control"
    utterance: str = ""
    control: str = ""

    def is_control(self) -> bool:
        """Whether this entry is a session-reset control event."""
        return self.kind == "control"


@dataclass(frozen=True)
class JourneyView:
    """Generator-visible journey: ordered entries with no oracle labels."""

    journey_id: str
    slug: str
    turns: tuple[JourneyTurnView, ...]

    def substantive_utterances(self) -> list[tuple[int, str]]:
        """Return ``(turn, utterance)`` for substantive user turns only."""
        return [(t.turn, t.utterance) for t in self.turns if t.kind == "user"]


def load_generator_views(
    repo_root: Path | None = None,
) -> tuple[list[SingleCaseView], list[JourneyView]]:
    """Load generator-visible views from the frozen input projection only.

    The oracle fixture is never opened here by construction: only
    ``INPUT_REL`` is read, and every record is re-checked against
    :data:`FORBIDDEN_GENERATOR_KEYS` before it is returned.
    """
    root = repo_root or find_repo_root()
    # Mechanical corpus validation first (schema, counts, checksums, /new).
    validate(root)
    _, singles, journeys = load_input(root / INPUT_REL)
    single_views: list[SingleCaseView] = []
    for record in singles:
        if not isinstance(record, dict):
            raise ConversationEvalError("input single record must be an object")
        assert_no_oracle_leak(record, str(record.get("id", "single")))
        case_id = str(record["id"])
        utterance = str(record["utterance"])
        if not utterance.strip():
            raise ConversationEvalError(f"{case_id}: empty utterance")
        if utterance.strip() == "/new":
            raise ConversationEvalError(f"{case_id}: /new must not be a model turn")
        single_views.append(SingleCaseView(case_id=case_id, utterance=utterance))
    journey_views: list[JourneyView] = []
    for record in journeys:
        if not isinstance(record, dict):
            raise ConversationEvalError("input journey record must be an object")
        assert_no_oracle_leak(record, str(record.get("id", "journey")))
        journey_id = str(record["id"])
        slug = str(record["journey"])
        entries: list[JourneyTurnView] = []
        for entry in record["turns"]:
            if not isinstance(entry, dict):
                raise ConversationEvalError(f"{journey_id}: journey entry must be an object")
            assert_no_oracle_leak(entry, f"{journey_id} turn {entry.get('turn')}")
            kind = str(entry["kind"])
            number = int(entry["turn"])
            if kind == "control":
                if str(entry.get("control")) != SESSION_RESET_CONTROL:
                    raise ConversationEvalError(f"{journey_id}: malformed control event")
                if "/new" in str(entry.get("utterance", "")):
                    raise ConversationEvalError(f"{journey_id}: control must not carry /new prose")
                entries.append(
                    JourneyTurnView(turn=number, kind="control", control=SESSION_RESET_CONTROL)
                )
            else:
                utterance = str(entry["utterance"])
                if utterance.strip() == "/new":
                    raise ConversationEvalError(f"{journey_id}: /new must be a control event")
                entries.append(JourneyTurnView(turn=number, kind="user", utterance=utterance))
        journey_views.append(JourneyView(journey_id=journey_id, slug=slug, turns=tuple(entries)))
    return single_views, journey_views


def bounded_history(prior_turns: list[str], *, limit: int = 10) -> list[str]:
    """Return at most the last ``limit`` prior turns from the same session."""
    if limit <= 0:
        raise ConversationEvalError("history limit must be > 0")
    return list(prior_turns[-limit:])


# ---------------------------------------------------------------------------
# Session semantics
# ---------------------------------------------------------------------------


def allocate_chat_ids(journey_ids: list[str], *, base: int = 900000) -> dict[str, int]:
    """Allocate one unique synthetic chat/session id per journey id.

    Deterministic and collision-free: ``base + index`` over sorted journey
    ids, so the same corpus always maps to the same synthetic chats while
    distinct journeys never share a session.
    """
    if base <= 0:
        raise ConversationEvalError("chat base must be > 0")
    ordered = sorted(set(journey_ids))
    if len(ordered) != len(journey_ids):
        raise ConversationEvalError("duplicate journey ids cannot share a session")
    return {jid: base + index for index, jid in enumerate(ordered)}


def fresh_chat_ids(journey_ids: list[str], *, base: int, repeat: int) -> dict[str, int]:
    """Allocate fresh sessions for an independent repeat (no state reuse)."""
    if repeat < 0:
        raise ConversationEvalError("repeat index must be >= 0")
    # Each repeat shifts into a disjoint chat-id block so repeats never collide
    # with each other or with the primary allocation.
    stride = 100000
    return allocate_chat_ids(journey_ids, base=base + (repeat + 1) * stride)


# ---------------------------------------------------------------------------
# Production turn boundary (transport-independent)
# ---------------------------------------------------------------------------


class TurnSender(Protocol):
    """Narrow OpenCode send boundary behind the production turn pipeline."""

    async def ensure_session(self, chat_id: int) -> str: ...
    async def send(self, session_id: str, text: str, *, model: str) -> str: ...
    async def reset(self, chat_id: int) -> str: ...


@dataclass
class TurnObservation:
    """Raw observation of one production turn (pre-capture)."""

    answer: str
    safety_decision: str
    safety_categories: tuple[str, ...] = ()
    planner_diagnostics: dict[str, Any] = field(default_factory=dict)
    retrieval_source_ids: tuple[str, ...] = ()
    evidence_locators: tuple[str, ...] = ()
    evidence_checksums: tuple[str, ...] = ()
    grounding_passed: bool | None = None
    regeneration_count: int = 0
    primary_model: str = ""
    actual_model: str = ""
    fallback_used: bool = False
    latency_s: float = 0.0
    tool_call_count: int = 0
    evidence_token_count: int = 0
    error_category: str = "ok"
    retry_count: int = 0


def classify_error_category(exc: BaseException) -> str:
    """Classify a turn failure as infrastructure or quality-relevant.

    HTTP 429 / provider-unavailable / model-unavailable (and the transient
    taxonomy that carries them) are infrastructure/provider outcomes, never
    answer-quality failures.
    """
    if isinstance(exc, OpenCodeSessionNotFoundError):
        return "session-not-found"
    if isinstance(exc, (OpenCodeTransientError, OpenCodeTimeoutError)):
        text = str(exc).casefold()
        if any(m in text for m in ("429", "http=429")):
            return "provider-429"
        return "provider-transient"
    if isinstance(exc, OpenCodeError):
        text = str(exc).casefold()
        lowered = text
        if any(m in lowered for m in INFRASTRUCTURE_MATCHERS):
            if "429" in lowered or "too many requests" in lowered:
                return "provider-429"
            return "provider-unavailable"
        return "deterministic-error"
    text = str(exc).casefold()
    if any(m in text for m in INFRASTRUCTURE_MATCHERS):
        if "429" in text or "too many requests" in text:
            return "provider-429"
        return "provider-unavailable"
    return "deterministic-error"


def is_infrastructure_category(category: str) -> bool:
    """Whether an error category is infrastructure (not quality)."""
    return category in (
        "provider-429",
        "provider-transient",
        "provider-unavailable",
        "model-unavailable",
    )


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry/backoff; never rotates runners for quota."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_s: float = DEFAULT_BASE_DELAY_S
    max_delay_s: float = DEFAULT_MAX_DELAY_S

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 5:
            raise ConversationEvalError("max_attempts must be within 1..5 (bounded)")
        if self.base_delay_s <= 0 or self.max_delay_s <= 0:
            raise ConversationEvalError("retry delays must be > 0")
        if self.base_delay_s > self.max_delay_s:
            raise ConversationEvalError("base delay must not exceed max delay")

    def delay_for(self, attempt: int) -> float:
        """Exponential backoff for 1-based ``attempt`` (capped)."""
        delay: float = self.base_delay_s * (2.0 ** (attempt - 1))
        if delay > self.max_delay_s:
            return self.max_delay_s
        return delay


async def run_with_retry(
    sender: TurnSender,
    *,
    chat_id: int,
    utterance: str,
    primary_model: str,
    fallback_model: str,
    policy: RetryPolicy | None = None,
    sleep: Any = None,
) -> TurnObservation:
    """Execute one turn with bounded retry and a configured technical fallback.

    The production safety gate runs first, exactly as in the Telegram
    path: ``block``/``emergency`` turns never reach the model and are
    captured with their real safety decision and categories instead of a
    fabricated ``"allow"``. Only ``allow`` turns call the sender.

    Retries transient/provider failures only, up to ``policy.max_attempts``.
    Deterministic failures fail immediately. When the primary model is
    transiently unavailable, one bounded fallback attempt is made and
    segmented via ``fallback_used=True`` so fallback output never silently
    counts as a primary-model qualification result. Runner/IP rotation for
    quota evasion is never performed here.

    Planner/retrieval/evidence/grounding capture fields stay empty when the
    narrow :class:`TurnSender` boundary returns answer text only; they must
    be populated by the production capture stage when available and never
    fabricated here.
    """
    active = policy or RetryPolicy()
    sleeper = sleep or asyncio.sleep
    from aa.safety.router import SafetyDecision, SafetyRouter

    safety_result, emergency_reply = SafetyRouter().route(utterance)
    safety_categories = tuple(category.value for category in safety_result.categories)
    if safety_result.decision is SafetyDecision.BLOCK:
        return TurnObservation(
            answer="",
            safety_decision="block",
            safety_categories=safety_categories,
            primary_model=primary_model,
            actual_model=primary_model,
            fallback_used=False,
            latency_s=0.0,
            retry_count=0,
        )
    if safety_result.decision is SafetyDecision.EMERGENCY:
        return TurnObservation(
            answer=emergency_reply or "",
            safety_decision="emergency",
            safety_categories=safety_categories,
            primary_model=primary_model,
            actual_model=primary_model,
            fallback_used=False,
            latency_s=0.0,
            retry_count=0,
        )
    session_id = await sender.ensure_session(chat_id)
    last_error: BaseException | None = None
    for attempt in range(1, active.max_attempts + 1):
        started = time.monotonic()
        try:
            answer = await sender.send(session_id, utterance, model=primary_model)
            return TurnObservation(
                answer=answer,
                safety_decision="allow",
                safety_categories=safety_categories,
                primary_model=primary_model,
                actual_model=primary_model,
                fallback_used=False,
                latency_s=max(0.0, time.monotonic() - started),
                retry_count=attempt - 1,
            )
        except OpenCodeSessionNotFoundError as exc:
            # Rebind once per attempt chain: a stale local mapping recreates
            # a fresh remote session and retries against it.
            last_error = exc
            session_id = await sender.reset(chat_id)
            if attempt >= active.max_attempts:
                break
            await sleeper(active.delay_for(attempt))
        except (OpenCodeTransientError, OpenCodeTimeoutError) as exc:
            last_error = exc
            if attempt >= active.max_attempts:
                break
            await sleeper(active.delay_for(attempt))
        except OpenCodeError as exc:
            # Deterministic: retrying the identical request cannot help.
            raise exc
    # Bounded technical fallback: exactly one fallback-model attempt after the
    # primary path exhausted its bounded retries.
    if last_error is not None and fallback_model and fallback_model != primary_model:
        started = time.monotonic()
        try:
            answer = await sender.send(session_id, utterance, model=fallback_model)
            return TurnObservation(
                answer=answer,
                safety_decision="allow",
                safety_categories=safety_categories,
                primary_model=primary_model,
                actual_model=fallback_model,
                fallback_used=True,
                latency_s=max(0.0, time.monotonic() - started),
                retry_count=active.max_attempts,
                error_category="fallback-used",
            )
        except OpenCodeError as exc:
            category = classify_error_category(exc)
            raise ConversationEvalError(f"turn failed after fallback: {category}") from exc
    category = (
        classify_error_category(last_error) if last_error is not None else "deterministic-error"
    )
    raise ConversationEvalError(f"turn failed without fallback: {category}")


# ---------------------------------------------------------------------------
# Capture schema
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """Return hex SHA-256 of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class TurnCapture:
    """Per-substantive-turn capture (no hidden chain-of-thought)."""

    case_id: str
    journey_id: str
    turn: int
    synthetic_input: str
    generated_answer: str
    safety_decision: str
    safety_categories: tuple[str, ...] = ()
    planner_diagnostics: dict[str, Any] = field(default_factory=dict)
    retrieval_source_ids: tuple[str, ...] = ()
    evidence_locators: tuple[str, ...] = ()
    evidence_checksums: tuple[str, ...] = ()
    grounding_passed: bool | None = None
    regeneration_count: int = 0
    primary_model: str = ""
    actual_model: str = ""
    fallback_used: bool = False
    latency_s: float = 0.0
    tool_call_count: int = 0
    evidence_token_count: int = 0
    source_token_count: int = 0
    retry_count: int = 0
    error_category: str = "ok"
    answer_sha256: str = ""
    output_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ConversationEvalError("capture case_id must not be empty")
        if self.safety_decision not in ("allow", "emergency", "block"):
            raise ConversationEvalError(f"invalid safety decision {self.safety_decision!r}")
        if self.regeneration_count < 0 or self.retry_count < 0:
            raise ConversationEvalError("regeneration/retry counts must be >= 0")

    def with_hashes(self) -> TurnCapture:
        """Return a copy with answer/output hashes populated."""
        digest = sha256_text(self.generated_answer)
        self.answer_sha256 = digest
        self.output_sha256 = digest
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the capture (full transcript side, encrypted at rest)."""
        return {
            "case_id": self.case_id,
            "journey_id": self.journey_id,
            "turn": self.turn,
            "synthetic_input": self.synthetic_input,
            "generated_answer": self.generated_answer,
            "safety_decision": self.safety_decision,
            "safety_categories": list(self.safety_categories),
            "planner_diagnostics": dict(self.planner_diagnostics),
            "retrieval_source_ids": list(self.retrieval_source_ids),
            "evidence_locators": list(self.evidence_locators),
            "evidence_checksums": list(self.evidence_checksums),
            "grounding_passed": self.grounding_passed,
            "regeneration_count": self.regeneration_count,
            "primary_model": self.primary_model,
            "actual_model": self.actual_model,
            "fallback_used": self.fallback_used,
            "model_path": ("fallback" if self.fallback_used else "primary"),
            "latency_s": self.latency_s,
            "tool_call_count": self.tool_call_count,
            "evidence_token_count": self.evidence_token_count,
            "source_token_count": self.source_token_count,
            "retry_count": self.retry_count,
            "error_category": self.error_category,
            "answer_sha256": self.answer_sha256 or sha256_text(self.generated_answer),
            "output_sha256": self.output_sha256 or sha256_text(self.generated_answer),
        }

    def manifest_row(self) -> dict[str, Any]:
        """Privacy-safe manifest row: metrics/checksums, never answer text."""
        return {
            "case_id": self.case_id,
            "journey_id": self.journey_id,
            "turn": self.turn,
            "safety_decision": self.safety_decision,
            "safety_categories": list(self.safety_categories),
            "grounding_passed": self.grounding_passed,
            "regeneration_count": self.regeneration_count,
            "primary_model": self.primary_model,
            "actual_model": self.actual_model,
            "fallback_used": self.fallback_used,
            "model_path": ("fallback" if self.fallback_used else "primary"),
            "latency_s": self.latency_s,
            "tool_call_count": self.tool_call_count,
            "evidence_token_count": self.evidence_token_count,
            "source_token_count": self.source_token_count,
            "retry_count": self.retry_count,
            "error_category": self.error_category,
            "answer_sha256": self.answer_sha256 or sha256_text(self.generated_answer),
            "output_sha256": self.output_sha256 or sha256_text(self.generated_answer),
            "input_chars": len(self.synthetic_input),
        }


# ---------------------------------------------------------------------------
# Deterministic sharding + completeness merge
# ---------------------------------------------------------------------------


def stable_shard(stable_id: str, shard_count: int) -> int:
    """Assign ``stable_id`` to a shard in ``0..shard_count-1`` deterministically."""
    if not stable_id.strip():
        raise ConversationEvalError("stable id must not be empty")
    if not 1 <= shard_count <= MAX_SHARD_COUNT:
        raise ConversationEvalError(f"shard_count must be within 1..{MAX_SHARD_COUNT}")
    digest = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shard_count


@dataclass(frozen=True)
class ShardPlan:
    """One deterministic shard: atomic journeys plus sharded single cases."""

    shard_index: int
    shard_count: int
    single_ids: tuple[str, ...]
    journey_ids: tuple[str, ...]

    def case_ids(self) -> tuple[str, ...]:
        """All top-level case ids covered by this shard."""
        return (*self.single_ids, *self.journey_ids)


def plan_shards(
    single_ids: list[str],
    journey_ids: list[str],
    *,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> list[ShardPlan]:
    """Plan deterministic shards keeping each journey atomic in one shard."""
    if not 1 <= shard_count <= MAX_SHARD_COUNT:
        raise ConversationEvalError(f"shard_count must be within 1..{MAX_SHARD_COUNT}")
    if len(set(single_ids)) != len(single_ids):
        raise ConversationEvalError("duplicate single-turn ids")
    if len(set(journey_ids)) != len(journey_ids):
        raise ConversationEvalError("duplicate journey ids")
    if set(single_ids) & set(journey_ids):
        raise ConversationEvalError("single and journey ids must not overlap")
    buckets_single: list[list[str]] = [[] for _ in range(shard_count)]
    buckets_journey: list[list[str]] = [[] for _ in range(shard_count)]
    for case_id in sorted(single_ids):
        buckets_single[stable_shard(case_id, shard_count)].append(case_id)
    for journey_id in sorted(journey_ids):
        buckets_journey[stable_shard(journey_id, shard_count)].append(journey_id)
    plans: list[ShardPlan] = []
    for index in range(shard_count):
        plans.append(
            ShardPlan(
                shard_index=index,
                shard_count=shard_count,
                single_ids=tuple(sorted(buckets_single[index])),
                journey_ids=tuple(sorted(buckets_journey[index])),
            )
        )
    return plans


def check_parallelism(*, shard_count: int, max_parallel: int = DEFAULT_MAX_PARALLEL) -> None:
    """Enforce bounded parallelism (conservative default, no quota evasion)."""
    if max_parallel < 1:
        raise ConversationEvalError("max_parallel must be >= 1")
    if max_parallel > MAX_PARALLEL_SHARDS:
        raise ConversationEvalError(f"max_parallel must not exceed {MAX_PARALLEL_SHARDS}")
    if max_parallel > shard_count:
        raise ConversationEvalError("max_parallel must not exceed shard_count")


@dataclass(frozen=True)
class ShardManifest:
    """Privacy-safe per-shard manifest (no answer/evidence text)."""

    shard_index: int
    shard_count: int
    case_ids: tuple[str, ...]
    turn_rows: tuple[dict[str, Any], ...]
    bundle_sha256: str
    eval_identity: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shard manifest."""
        return {
            "schema_version": EVAL_SCHEMA_VERSION,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "case_ids": list(self.case_ids),
            "turn_rows": list(self.turn_rows),
            "bundle_sha256": self.bundle_sha256,
            "eval_identity": dict(self.eval_identity),
        }


def merge_manifests(
    manifests: list[ShardManifest],
    *,
    expected_case_ids: list[str],
) -> dict[str, Any]:
    """Merge shard manifests, rejecting missing/duplicate case ids.

    Raises :class:`ConversationEvalError` when any expected case is missing,
    any case appears twice, or any unexpected case appears.
    """
    expected = sorted(expected_case_ids)
    if len(set(expected)) != len(expected):
        raise ConversationEvalError("expected case ids contain duplicates")
    seen: dict[str, int] = {}
    for manifest in manifests:
        for case_id in manifest.case_ids:
            if case_id in seen:
                raise ConversationEvalError(f"duplicate case id {case_id!r} across shards")
            seen[case_id] = manifest.shard_index
    missing = [c for c in expected if c not in seen]
    if missing:
        raise ConversationEvalError(f"merge is missing {len(missing)} case(s): {missing[:5]}")
    unexpected = [c for c in seen if c not in set(expected)]
    if unexpected:
        raise ConversationEvalError(f"merge holds unexpected cases: {unexpected[:5]}")
    rows: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda m: m.shard_index):
        rows.extend(manifest.turn_rows)
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "shard_count": len(manifests),
        "case_ids": expected,
        "turn_rows": rows,
    }


def is_resumable(prior_identity: dict[str, str], current_identity: dict[str, str]) -> bool:
    """Whether a successful shard may be resumed under the current config.

    Resumable only when exact SHA + corpus checksums + runtime config + model
    config all match; anything else forces a fresh shard execution.
    """
    required = (
        "main_sha",
        "corpus_input_sha256",
        "corpus_oracle_sha256",
        "runtime_version",
        "prompt_version",
        "retrieval_version",
        "index_version",
        "primary_model",
        "fallback_model",
    )
    for key in required:
        if key not in prior_identity or key not in current_identity:
            return False
        if prior_identity[key] != current_identity[key]:
            return False
    return True


# ---------------------------------------------------------------------------
# Exact-main contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalIdentity:
    """Exact configuration identifying one authoritative benchmark tuple."""

    main_sha: str
    corpus_input_sha256: str
    corpus_oracle_sha256: str
    runtime_version: str
    prompt_version: str
    retrieval_version: str
    index_version: str
    primary_model: str
    fallback_model: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the identity."""
        return {
            "main_sha": self.main_sha,
            "corpus_input_sha256": self.corpus_input_sha256,
            "corpus_oracle_sha256": self.corpus_oracle_sha256,
            "runtime_version": self.runtime_version,
            "prompt_version": self.prompt_version,
            "retrieval_version": self.retrieval_version,
            "index_version": self.index_version,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
        }


def collect_eval_identity(
    *,
    main_sha: str,
    repo_root: Path | None = None,
    primary_model: str,
    fallback_model: str,
) -> EvalIdentity:
    """Collect the exact benchmark identity from the frozen corpus + runtime."""
    root = repo_root or find_repo_root()
    summary = validate(root)
    prompt_path = root / "prompts" / "aa-agent-system.md"
    prompt_sha = sha256_file(prompt_path) if prompt_path.exists() else "missing-prompt"
    return EvalIdentity(
        main_sha=main_sha,
        corpus_input_sha256=summary.input_sha256,
        corpus_oracle_sha256=summary.oracle_sha256,
        runtime_version=runtime_version(),
        prompt_version=prompt_sha,
        retrieval_version=retrieval_version(),
        index_version=index_version(),
        primary_model=primary_model,
        fallback_model=fallback_model,
    )


def runtime_version() -> str:
    """Return the production runtime version pointer."""
    from aa import __version__ as app_version

    return f"aa-worker/{app_version}"


def retrieval_version() -> str:
    """Return the retrieval/structure builder version pointer."""
    from aa.corpus.structure import BUILDER_VERSION, STRUCTURE_FORMAT

    return f"{STRUCTURE_FORMAT}/builder-{BUILDER_VERSION}"


def index_version() -> str:
    """Return the canonical index format pointer."""
    from aa.corpus.canonical import ARTIFACT_FORMAT, RU_ARTIFACT_FORMAT

    return f"{ARTIFACT_FORMAT}+{RU_ARTIFACT_FORMAT}"


def validate_exact_main(
    *,
    tested_sha: str,
    current_main_sha: str,
    trusted_pass_sha: str,
) -> None:
    """Refuse to run/publish unless the exact #40 PASS SHA is still current.

    All three SHAs must be identical: the SHA under test, the SHA main
    currently points at, and the latest trusted #40 PASS marker. A moved main
    or stale qualification evidence fails closed.
    """
    for name, value in (
        ("tested_sha", tested_sha),
        ("current_main_sha", current_main_sha),
        ("trusted_pass_sha", trusted_pass_sha),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", value or ""):
            raise ConversationEvalError(f"{name} must be a 40-hex SHA")
    if tested_sha != trusted_pass_sha:
        raise ConversationEvalError("tested SHA is not the current #40 PASS SHA")
    if current_main_sha != trusted_pass_sha:
        raise ConversationEvalError(
            "main advanced past the trusted #40 PASS SHA; refusing stale run"
        )


def revalidate_at_publication(*, tested_sha: str, current_main_sha: str) -> str:
    """Re-check main at publication time; returns ``complete`` or ``stale``.

    A moved main cannot produce a current COMPLETE result: the artifact is
    kept for diagnostics but the marker is ``stale`` and #62 stays open.
    """
    if current_main_sha != tested_sha:
        return "stale"
    return "complete"


def build_result_marker(*, sha: str, corpus: str, result: str, run: str) -> str:
    """Build the canonical #62 result marker comment."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise ConversationEvalError("marker sha must be a 40-hex SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", corpus or ""):
        raise ConversationEvalError("marker corpus must be a 64-hex SHA")
    if result not in ("complete", "incomplete", "stale"):
        raise ConversationEvalError("marker result must be complete|incomplete|stale")
    if not run.strip() or any(c.isspace() for c in run):
        raise ConversationEvalError("marker run id must be a non-empty token")
    prefix = "<!-- aa-conversation-eval-result"
    return f"{prefix} issue=62 sha={sha} corpus={corpus} result={result} run={run} -->"


def parse_result_marker(text: str) -> dict[str, str]:
    """Parse and validate a canonical result marker comment."""
    match = _MARKER_RE.search(text)
    if match is None:
        raise ConversationEvalError("no canonical result marker found")
    if match.group("issue") != str(RESULT_ISSUE):
        raise ConversationEvalError("result marker must target issue 62")
    return {
        "issue": match.group("issue"),
        "sha": match.group("sha"),
        "corpus": match.group("corpus"),
        "result": match.group("result"),
        "run": match.group("run"),
    }


def should_rerun(
    *,
    latest_complete: dict[str, str] | None,
    candidate_sha: str,
    candidate_corpus: str,
) -> bool:
    """Decide whether a newer #40 PASS tuple needs a fresh #62 run.

    Idempotent: the same exact SHA + corpus tuple never reruns. A
    newer/different tuple reruns exactly once per SHA+corpus pair.
    """
    if latest_complete is None:
        return True
    return not (
        latest_complete.get("sha") == candidate_sha
        and latest_complete.get("corpus") == candidate_corpus
    )


# ---------------------------------------------------------------------------
# Encryption + privacy-safe manifests
# ---------------------------------------------------------------------------


def compress_and_encrypt(payload: bytes, *, recipient: str) -> bytes:
    """Compress with zstd then age-encrypt to ``recipient`` (shard bundle)."""
    if not payload:
        raise ConversationEvalError("refusing to encrypt an empty payload")
    if not recipient.strip():
        raise ConversationEvalError("age recipient must not be empty")
    try:
        from aa.corpus.age_v1 import parse_recipient as _parse_recipient
    except ImportError as exc:
        raise ConversationEvalError("age recipient support is unavailable") from exc
    _parse_recipient(recipient)  # fail closed on malformed recipient
    compressed = zstd.ZstdCompressor(level=3, threads=1).compress(payload)
    return age_v1.encrypt_bytes(compressed, [recipient])


def decrypt_and_decompress(bundle: bytes, *, identity: str) -> bytes:
    """Decrypt a shard bundle and decompress the zstd payload (tests/tools)."""
    if not bundle:
        raise ConversationEvalError("refusing to decrypt an empty bundle")
    if not identity.strip():
        raise ConversationEvalError("age identity must not be empty")
    compressed = age_v1.decrypt_bytes(bundle, [identity])
    try:
        return zstd.ZstdDecompressor().decompress(compressed, max_output_size=64 * 1024 * 1024)
    except zstd.ZstdError as exc:
        raise ConversationEvalError(f"bundle decompression failed: {exc}") from exc


def build_shard_bundle(captures: list[TurnCapture], *, recipient: str) -> tuple[bytes, str]:
    """Build the already compressed + age-encrypted shard bundle.

    Returns ``(encrypted_bytes, sha256_of_encrypted_bytes)``. Plaintext
    answer/evidence bundles are never uploaded as intermediate artifacts;
    only this encrypted payload leaves the shard job.
    """
    if not captures:
        raise ConversationEvalError("refusing to bundle zero captures")
    payload = json.dumps(
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "turns": [c.with_hashes().to_dict() for c in captures],
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    encrypted = compress_and_encrypt(payload, recipient=recipient)
    return encrypted, hashlib.sha256(encrypted).hexdigest()


def build_compact_manifest(
    *,
    eval_identity: EvalIdentity,
    shard_manifests: list[ShardManifest],
    run_id: str,
    result: str,
) -> dict[str, Any]:
    """Build the compact privacy-safe manifest persisted outside main.

    Keyed by exact main SHA + corpus checksum + run id; holds IDs, metrics
    and checksums only, never large source dumps or answer text.
    """
    if result not in ("complete", "incomplete", "stale"):
        raise ConversationEvalError("manifest result must be complete|incomplete|stale")
    if not run_id.strip():
        raise ConversationEvalError("run id must not be empty")
    rows: list[dict[str, Any]] = []
    for manifest in shard_manifests:
        rows.extend(manifest.turn_rows)
    payload: dict[str, Any] = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "run_id": run_id,
        "result": result,
        "main_sha": eval_identity.main_sha,
        "corpus_input_sha256": eval_identity.corpus_input_sha256,
        "corpus_oracle_sha256": eval_identity.corpus_oracle_sha256,
        "runtime_version": eval_identity.runtime_version,
        "prompt_version": eval_identity.prompt_version,
        "retrieval_version": eval_identity.retrieval_version,
        "index_version": eval_identity.index_version,
        "primary_model": eval_identity.primary_model,
        "fallback_model": eval_identity.fallback_model,
        "turn_rows": rows,
        "bundle_sha256s": [
            m.bundle_sha256 for m in sorted(shard_manifests, key=lambda m: m.shard_index)
        ],
    }
    assert_manifest_privacy_safe(payload)
    return payload


_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "generated_answer",
        "synthetic_input",
        "utterance",
        "answer",
        "evidence_text",
        "source_text",
        "chain_of_thought",
        "hidden_reasoning",
        "secret",
        "identity",
        "private_key",
    }
)


def assert_manifest_privacy_safe(payload: Any, *, depth: int = 0) -> None:
    """Reject secrets, answer dumps or large source text in compact manifests."""
    if depth > 12:
        raise ConversationEvalError("manifest is nested too deeply")
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _FORBIDDEN_MANIFEST_KEYS:
                raise ConversationEvalError(f"manifest must not carry {key!r}")
            if isinstance(value, str) and len(value) > 2000:
                raise ConversationEvalError(f"manifest field {key!r} is a large source dump")
            assert_manifest_privacy_safe(value, depth=depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            assert_manifest_privacy_safe(item, depth=depth + 1)


def summarize_run(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute privacy-safe analytics over manifest rows (no text needed)."""
    total = len(turn_rows)
    infra = sum(
        1 for r in turn_rows if is_infrastructure_category(str(r.get("error_category", "ok")))
    )
    fallback = sum(1 for r in turn_rows if bool(r.get("fallback_used", False)))
    latencies = sorted(float(r.get("latency_s", 0.0)) for r in turn_rows)

    def _pct(p: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, int(p * len(latencies)))
        return latencies[index]

    by_decision: dict[str, int] = {}
    for row in turn_rows:
        decision = str(row.get("safety_decision", "unknown"))
        by_decision[decision] = by_decision.get(decision, 0) + 1
    completed = sum(1 for r in turn_rows if str(r.get("error_category", "ok")) == "ok")
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "total_turns": total,
        "completed_turns": completed,
        "completion_rate": (completed / total) if total else 0.0,
        "infrastructure_failures": infra,
        "provider_failure_rate": (infra / total) if total else 0.0,
        "fallback_turns": fallback,
        "safety_decisions": by_decision,
        "latency_p50_s": _pct(0.5),
        "latency_p95_s": _pct(0.95),
    }


def expected_case_ids(
    singles: list[SingleCaseView],
    journeys: list[JourneyView],
) -> list[str]:
    """Return every top-level case id the authoritative run must complete."""
    return [s.case_id for s in singles] + [j.journey_id for j in journeys]


def validate_files_do_not_mutate_main(paths: list[str]) -> None:
    """Guard that benchmark outputs never land on production main paths."""
    protected_dirs = (
        "corpus",
        "prompts",
        "src",
        "tests",
        "scripts",
        "qualification",
        ".github",
    )
    protected_files = ("pyproject.toml", "main")
    for path in paths:
        normalized = path.strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        normalized = posixpath.normpath(normalized)
        if normalized in (".", "") or normalized == ".." or normalized.startswith("../"):
            raise ConversationEvalError(f"benchmark output must not mutate main path {path!r}")
        for directory in protected_dirs:
            if normalized == directory or normalized.startswith(directory + "/"):
                raise ConversationEvalError(f"benchmark output must not mutate main path {path!r}")
        if normalized in protected_files:
            raise ConversationEvalError(f"benchmark output must not mutate main path {path!r}")


def corpus_checksums(repo_root: Path | None = None) -> dict[str, str]:
    """Return stable input/oracle checksums for the frozen corpus."""
    root = repo_root or find_repo_root()
    return {
        "input": sha256_file(root / INPUT_REL),
        "oracle": sha256_file(root / ORACLE_REL),
        "version": sha256_file(root / VERSION_REL),
    }


__all__ = [
    "DEFAULT_BASE_DELAY_S",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_DELAY_S",
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_SHARD_COUNT",
    "EVAL_SCHEMA_VERSION",
    "FORBIDDEN_GENERATOR_KEYS",
    "MAX_PARALLEL_SHARDS",
    "MAX_SHARD_COUNT",
    "RESULT_ISSUE",
    "RESULT_MARKER",
    "ConversationEvalError",
    "EvalIdentity",
    "JourneyTurnView",
    "JourneyView",
    "RetryPolicy",
    "ShardManifest",
    "ShardPlan",
    "SingleCaseView",
    "TurnCapture",
    "TurnObservation",
    "TurnSender",
    "allocate_chat_ids",
    "assert_manifest_privacy_safe",
    "assert_no_oracle_leak",
    "bounded_history",
    "build_compact_manifest",
    "build_result_marker",
    "build_shard_bundle",
    "check_parallelism",
    "classify_error_category",
    "collect_eval_identity",
    "compress_and_encrypt",
    "corpus_checksums",
    "decrypt_and_decompress",
    "expected_case_ids",
    "fresh_chat_ids",
    "index_version",
    "is_infrastructure_category",
    "is_resumable",
    "load_generator_views",
    "merge_manifests",
    "parse_result_marker",
    "plan_shards",
    "retrieval_version",
    "revalidate_at_publication",
    "run_with_retry",
    "runtime_version",
    "sha256_text",
    "should_rerun",
    "stable_shard",
    "summarize_run",
    "validate_exact_main",
    "validate_files_do_not_mutate_main",
]
