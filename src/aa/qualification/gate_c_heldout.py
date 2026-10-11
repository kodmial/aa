"""Gate C two-tier semantic evaluator (kodmial/aa#341).

Gate C proves the Product Contract as human-like Russian AA conversation
grounded exclusively in the canonical book. A visible fixed prompt list is
regression tooling, never independent unseen-conversation proof. This module
therefore separates two explicit tiers inside Gate C's single typed semantic
report:

- ``C-regression``: stable visible deterministic cases for fast failure
  localization. The 18 fixed literal Russian turns embedded in
  ``product_contract_live.run_live_telegram_evidence_lane`` plus the public
  ``qualification/held_out_v1.json`` descriptors and the public
  ``ru_answer_quality_rubric.v2.json`` calibration fixtures belong here.
- ``C-heldout``: sealed independent evaluation inputs unexposed to the
  implementer at code-change time. A visible checked-in fixture alone never
  makes held-out PASS.

Unseen-source policy (no new owner-provisioned secrets presumed):

1. A pre-existing approved protected fixture (a sealed JSON document that is
   never checked into the repository, supplied either through the
   ``AA_GATE_C_HELDOUT_FIXTURE`` path or through one of the protected
   repository paths) is preferred when present and schema-valid.
2. Otherwise a trusted post-code-freeze evaluator generates never-
   preannounced Russian multi-turn prompts/semantic variants with a
   run-unique unpredictable seed and the configured real provider. A
   repository-owned trusted workflow pins the merged evaluator source SHA
   plus the rubric digest as the protected evaluation baseline
   (``qualification/gate_c_evaluator_baseline.json``); the generated
   fixture is accepted only when that baseline exists and matches the
   running tree. Feature PRs may not substitute their own evaluator
   implementation or mutate the trusted baseline during qualification.
3. When neither viable unseen source can be obtained or validated, the
   held-out tier returns typed ``BLOCKED`` (never a silent reuse of
   exposed regression strings, never a product FAIL for a missing
   external prerequisite).

Independence is reported honestly: a separate session/agent on the same
provider family is ``procedural_only``. Family diversity is claimed only
when generator and judge model families provably differ.

Privacy: tier reports carry opaque scenario ids, digests, counts and
fixed-vocabulary categories only. Prompt text and seeds never enter
reports, logs or agent-visible evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_C_EVALUATOR_VERSION = "aa-gate-c-evaluator/1"
GATE_C_HELDOUT_SCHEMA_VERSION = "aa-gate-c-heldout-fixture/1"
GATE_C_BASELINE_SCHEMA_VERSION = "aa-gate-c-evaluator-baseline/1"
GATE_C_SEMANTIC_REPORT_VERSION = "aa-gate-c-semantic-report/1"

RUBRIC_PATH = "qualification/ru_answer_quality_rubric.v2.json"
RUBRIC_VERSION = "ru-answer-quality-rubric-v2"
BASELINE_PATH = "qualification/gate_c_evaluator_baseline.json"
SEALED_FIXTURE_PATHS: tuple[str, ...] = (
    "qualification/gate_c_heldout.sealed.json",
    "qualification/gate_c_heldout.generated.json",
)
ENV_FIXTURE_PATH = "AA_GATE_C_HELDOUT_FIXTURE"
ENV_BASELINE_PATH = "AA_GATE_C_HELDOUT_BASELINE"

HELDOUT_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "BLOCKED")

INDEPENDENCE_PROCEDURAL_ONLY = "procedural_only"
INDEPENDENCE_FAMILY_DIVERSE = "family_diverse"

# Typed external blockers. A missing unseen prerequisite is BLOCKED (an
# explicit external dependency), never a product FAIL and never a PASS.
BLOCKED_FIXTURE_UNAVAILABLE = "EXTERNAL_HELDOUT_FIXTURE_UNAVAILABLE"
BLOCKED_BASELINE_UNPINNED = "EXTERNAL_HELDOUT_BASELINE_UNPINNED"
BLOCKED_BASELINE_MISMATCH = "EXTERNAL_HELDOUT_BASELINE_MISMATCH"
BLOCKED_GENERATOR_UNAVAILABLE = "EXTERNAL_HELDOUT_GENERATOR_UNAVAILABLE"
BLOCKED_JUDGE_UNAVAILABLE = "EXTERNAL_JUDGE_UNAVAILABLE"
BLOCKED_CORPUS_UNREADABLE = "EXTERNAL_CORPUS_UNREADABLE"
BLOCKED_PROVIDER_UNAVAILABLE = "EXTERNAL_SEMANTIC_PROVIDER_UNAVAILABLE"
BLOCKED_PREREQUISITES_UNAVAILABLE = "EXTERNAL_LIVE_PREREQUISITES_UNAVAILABLE"

HELDOUT_SOURCES: tuple[str, ...] = (
    "protected_fixture",
    "post_freeze_generated",
    "unavailable",
)

# Coverage required of every sealed held-out fixture (fixed vocabulary).
HELDOUT_REQUIRED_CATEGORIES: tuple[str, ...] = (
    "same-thread-multi-turn-memory",
    "indirect-help",
    "meta-description",
    "book-grounded-alcohol-recovery",
    "family-sponsorship",
    "elliptical-followup",
    "topic-shift",
    "out-of-scope",
    "emergency",
    "quote-source-fidelity",
    "mixed-short-voice",
    "multi-need-canonical-evidence",
)

SAFETY_EXPECTATIONS: tuple[str, ...] = ("ordinary", "out_of_book", "emergency")

MIN_HELDOUT_SCENARIOS = 8
MIN_SEED_BYTES = 16

_UNAVAILABLE_VERIFIER_OUTCOMES: frozenset[str] = frozenset(
    {"unavailable", "partial-unavailable", "skipped-turn-budget"}
)

_KNOWN_RECEIPT_KINDS: frozenset[str] = frozenset(
    {"simulated-in-process", "external-confirmed", "external-unverified", "none"}
)

_SERVED_ANSWER_OUTCOMES: frozenset[str] = frozenset(
    {
        "served",
        "narrowed-supported",
        "narrowed-compacted",
        "narrowed-adequacy",
        "narrowed-adequacy-regen",
    }
)

_SERVED_VERIFIER_OUTCOMES: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "passed",
        "passed-after-repair",
        "passed-after-repair-existing-pack",
        "passed-after-adequacy-repair",
    }
)


class GateCHeldoutError(ValueError):
    """Raised when held-out fixture/baseline handling fails closed."""


class GateCHeldoutBlockedError(GateCHeldoutError):
    """Raised when no viable unseen source exists (typed BLOCKED reason)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "scripts" / "verify_product_contract_qualification.py").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def rubric_digest(repo_root: Path | None = None) -> str:
    """Return the SHA-256 hex digest of the frozen v2 rubric bytes."""
    root = repo_root or _repo_root()
    data = (root / RUBRIC_PATH).read_bytes()
    return hashlib.sha256(data).hexdigest()


def evaluator_source_sha(repo_root: Path | None = None) -> str:
    """Return the SHA-256 hex digest of this evaluator module's bytes.

    The trusted post-merge workflow pins this digest as the protected
    evaluation baseline so feature PRs cannot silently substitute their
    own evaluator implementation during qualification.
    """
    del repo_root
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def normalize_dialogue_text(text: str) -> str:
    """Normalize dialogue text for novelty comparison (mechanical only)."""
    lowered = (text or "").casefold().replace("ё", "е")
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


def fixture_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest of a fixture payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HeldoutSourceProvenance:
    """Where the held-out tier inputs came from (privacy-safe)."""

    source: str
    fixture_digest: str
    generation_success: bool
    evaluator_version: str
    evaluator_source_sha: str
    rubric_version: str
    rubric_digest: str
    run_id: str
    independence_level: str
    reproducibility: str
    blocked_reason: str
    scenario_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fixture_digest": self.fixture_digest,
            "generation_success": bool(self.generation_success),
            "evaluator_version": self.evaluator_version,
            "evaluator_source_sha": self.evaluator_source_sha,
            "rubric_version": self.rubric_version,
            "rubric_digest": self.rubric_digest,
            "run_id": self.run_id,
            "independence_level": self.independence_level,
            "reproducibility": self.reproducibility,
            "blocked_reason": self.blocked_reason,
            "scenario_count": int(self.scenario_count),
        }


@dataclass(frozen=True)
class HeldoutTurnVerdict:
    """Typed verdict for one held-out turn (booleans/counts/ids only)."""

    outcome: str
    scenario_id: str
    judge_available: bool
    judge_model: str
    citation_scope: dict[str, int] = field(default_factory=dict)
    unsupported_claim_count: int = 0
    relevance_to_resolved_intent: bool = False
    dialogue_continuity_ok: bool = False
    receipt_kind: str = "simulated-in-process"
    failure_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "scenario_id": self.scenario_id,
            "judge_available": bool(self.judge_available),
            "judge_model": self.judge_model,
            "citation_scope": dict(self.citation_scope),
            "unsupported_claim_count": int(self.unsupported_claim_count),
            "relevance_to_resolved_intent": bool(self.relevance_to_resolved_intent),
            "dialogue_continuity_ok": bool(self.dialogue_continuity_ok),
            "receipt_kind": self.receipt_kind,
            "failure_category": self.failure_category,
        }


@dataclass(frozen=True)
class HeldoutTierReport:
    """Privacy-safe report for one Gate C tier (opaque ids only)."""

    tier: str
    status: str
    scenario_ids: tuple[str, ...] = ()
    passed_ids: tuple[str, ...] = ()
    failed_ids: tuple[str, ...] = ()
    failure_categories: dict[str, int] = field(default_factory=dict)
    provenance: HeldoutSourceProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tier": self.tier,
            "status": self.status,
            "scenario_count": len(self.scenario_ids),
            "scenario_ids": list(self.scenario_ids),
            "passed_count": len(self.passed_ids),
            "failed_count": len(self.failed_ids),
            "failure_categories": dict(self.failure_categories),
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class GateCSemanticReport:
    """Single typed Gate C semantic report holding both tiers."""

    schema_version: str
    regression: HeldoutTierReport
    heldout: HeldoutTierReport
    overall: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regression": self.regression.to_dict(),
            "heldout": self.heldout.to_dict(),
            "overall": self.overall,
        }


def _baseline_path(repo_root: Path) -> Path:
    override = (os.environ.get(ENV_BASELINE_PATH, "") or "").strip()
    if override:
        return Path(override)
    return repo_root / BASELINE_PATH


def _candidate_fixture_paths(repo_root: Path) -> list[Path]:
    override = (os.environ.get(ENV_FIXTURE_PATH, "") or "").strip()
    if override:
        return [Path(override)]
    return [repo_root / rel for rel in SEALED_FIXTURE_PATHS]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateCHeldoutBlockedError(BLOCKED_CORPUS_UNREADABLE) from exc
    if not isinstance(payload, dict):
        raise GateCHeldoutBlockedError(BLOCKED_FIXTURE_UNAVAILABLE)
    return payload


def _blocked_provenance(
    *,
    reason: str,
    evaluator_sha: str,
    rubric_hex: str,
    run_id: str,
) -> HeldoutSourceProvenance:
    return HeldoutSourceProvenance(
        source="unavailable",
        fixture_digest="",
        generation_success=False,
        evaluator_version=GATE_C_EVALUATOR_VERSION,
        evaluator_source_sha=evaluator_sha,
        rubric_version=RUBRIC_VERSION,
        rubric_digest=rubric_hex,
        run_id=run_id,
        independence_level=INDEPENDENCE_PROCEDURAL_ONLY,
        reproducibility="heldout-unavailable-no-reproducible-fixture",
        blocked_reason=reason,
        scenario_count=0,
    )


def validate_heldout_fixture(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a sealed held-out fixture (schema + coverage, no text read).

    Returns ``(ok, missing_categories)`` using opaque counts only.
    """
    if not isinstance(payload, Mapping):
        return False, list(HELDOUT_REQUIRED_CATEGORIES)
    if payload.get("version") != GATE_C_HELDOUT_SCHEMA_VERSION:
        return False, list(HELDOUT_REQUIRED_CATEGORIES)
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < MIN_HELDOUT_SCENARIOS:
        return False, list(HELDOUT_REQUIRED_CATEGORIES)
    seen: set[str] = set()
    ids: set[str] = set()
    multi_need_evidence = 0
    multi_turn_threads: dict[str, int] = {}
    for entry in scenarios:
        if not isinstance(entry, dict):
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        scenario_id = str(entry.get("id", ""))
        category = str(entry.get("category", ""))
        thread = str(entry.get("thread", ""))
        prompt = str(entry.get("prompt", ""))
        safety = str(entry.get("safety_expectation", "ordinary"))
        if not scenario_id or not category or not thread or not prompt.strip():
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        if scenario_id in ids:
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        ids.add(scenario_id)
        if category not in HELDOUT_REQUIRED_CATEGORIES:
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        if safety not in SAFETY_EXPECTATIONS:
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        if not isinstance(entry.get("requires_book_evidence"), bool):
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        if not str(entry.get("gold_intent_category", "")).strip():
            return False, list(HELDOUT_REQUIRED_CATEGORIES)
        seen.add(category)
        multi_turn_threads[thread] = multi_turn_threads.get(thread, 0) + 1
        if category == "multi-need-canonical-evidence" and bool(
            entry.get("requires_book_evidence")
        ):
            multi_need_evidence += 1
    missing = [name for name in HELDOUT_REQUIRED_CATEGORIES if name not in seen]
    if missing:
        return False, missing
    # Genuine multi-turn progress: at least one thread holds 2+ turns, and
    # at least two information needs require full canonical evidence.
    if not any(count >= 2 for count in multi_turn_threads.values()):
        return False, ["same-thread-multi-turn-memory"]
    if multi_need_evidence < 2:
        return False, ["multi-need-canonical-evidence"]
    return True, []


def check_fixture_novelty(
    *,
    fixture_prompts: Sequence[str],
    regression_prompts: Sequence[str],
) -> dict[str, Any]:
    """Prove held-out prompts are novel echo-free dialogue (counts only).

    Any held-out prompt normalizing equal to a visible regression prompt
    is overt fixture leakage and fails novelty. Intra-fixture duplicates
    fail as well: repetition is not unseen coverage.
    """
    normalized_regression = {
        normalize_dialogue_text(item) for item in regression_prompts if item.strip()
    }
    normalized_regression.discard("")
    seen: set[str] = set()
    echoed = 0
    duplicated = 0
    for prompt in fixture_prompts:
        normalized = normalize_dialogue_text(prompt)
        if not normalized:
            continue
        if normalized in normalized_regression:
            echoed += 1
        if normalized in seen:
            duplicated += 1
        seen.add(normalized)
    return {
        "novel": bool(echoed == 0 and duplicated == 0 and seen),
        "echoed_count": int(echoed),
        "duplicated_count": int(duplicated),
        "fixture_prompt_count": int(len(seen)),
    }


def resolve_heldout_source(
    *,
    repo_root: Path | None = None,
    run_id: str = "local",
) -> tuple[HeldoutSourceProvenance, dict[str, Any] | None]:
    """Resolve the unseen held-out source without leaking prompt text.

    Returns ``(provenance, fixture_or_None)``. A visible checked-in
    fixture (the embedded regression strings or the public
    ``held_out_v1.json`` descriptors) is never returned as a sealed
    source: without a protected fixture or a validated post-freeze
    generated fixture the result is typed BLOCKED.
    """
    root = repo_root or _repo_root()
    try:
        rubric_hex = rubric_digest(root)
    except OSError:
        return (
            HeldoutSourceProvenance(
                source="unavailable",
                fixture_digest="",
                generation_success=False,
                evaluator_version=GATE_C_EVALUATOR_VERSION,
                evaluator_source_sha=evaluator_source_sha(root),
                rubric_version=RUBRIC_VERSION,
                rubric_digest="",
                run_id=run_id,
                independence_level=INDEPENDENCE_PROCEDURAL_ONLY,
                reproducibility="heldout-unavailable-no-reproducible-fixture",
                blocked_reason=BLOCKED_CORPUS_UNREADABLE,
                scenario_count=0,
            ),
            None,
        )
    evaluator_sha = evaluator_source_sha(root)
    baseline_file = _baseline_path(root)
    baseline: dict[str, Any] | None = None
    if baseline_file.is_file():
        try:
            candidate = _read_json(baseline_file)
        except GateCHeldoutBlockedError:
            return (
                _blocked_provenance(
                    reason=BLOCKED_CORPUS_UNREADABLE,
                    evaluator_sha=evaluator_sha,
                    rubric_hex=rubric_hex,
                    run_id=run_id,
                ),
                None,
            )
        if candidate.get("version") != GATE_C_BASELINE_SCHEMA_VERSION:
            return (
                _blocked_provenance(
                    reason=BLOCKED_BASELINE_MISMATCH,
                    evaluator_sha=evaluator_sha,
                    rubric_hex=rubric_hex,
                    run_id=run_id,
                ),
                None,
            )
        baseline = candidate
        # Anti-substitution: a pinned baseline that no longer matches the
        # running evaluator/rubric invalidates every unseen claim.
        if (
            str(baseline.get("evaluator_source_sha", "")) != evaluator_sha
            or str(baseline.get("rubric_digest", "")) != rubric_hex
        ):
            return (
                _blocked_provenance(
                    reason=BLOCKED_BASELINE_MISMATCH,
                    evaluator_sha=evaluator_sha,
                    rubric_hex=rubric_hex,
                    run_id=run_id,
                ),
                None,
            )
    for candidate_path in _candidate_fixture_paths(root):
        if not candidate_path.is_file():
            continue
        try:
            fixture = _read_json(candidate_path)
        except GateCHeldoutBlockedError as exc:
            return (
                _blocked_provenance(
                    reason=str(exc.reason),
                    evaluator_sha=evaluator_sha,
                    rubric_hex=rubric_hex,
                    run_id=run_id,
                ),
                None,
            )
        ok, _ = validate_heldout_fixture(fixture)
        if not ok:
            return (
                _blocked_provenance(
                    reason=BLOCKED_FIXTURE_UNAVAILABLE,
                    evaluator_sha=evaluator_sha,
                    rubric_hex=rubric_hex,
                    run_id=run_id,
                ),
                None,
            )
        scenarios = fixture["scenarios"]
        assert isinstance(scenarios, list)
        generated = str(fixture.get("source", "protected_fixture"))
        if generated == "post_freeze_generated":
            # Freshly generated fixtures are trusted only against the
            # pinned baseline; a missing baseline cannot validate them.
            if baseline is None:
                return (
                    _blocked_provenance(
                        reason=BLOCKED_BASELINE_UNPINNED,
                        evaluator_sha=evaluator_sha,
                        rubric_hex=rubric_hex,
                        run_id=run_id,
                    ),
                    None,
                )
            provenance = HeldoutSourceProvenance(
                source="post_freeze_generated",
                fixture_digest=fixture_digest(fixture),
                generation_success=True,
                evaluator_version=GATE_C_EVALUATOR_VERSION,
                evaluator_source_sha=evaluator_sha,
                rubric_version=RUBRIC_VERSION,
                rubric_digest=rubric_hex,
                run_id=run_id,
                independence_level=INDEPENDENCE_PROCEDURAL_ONLY,
                reproducibility="run-unique-seed-single-use-fixture",
                blocked_reason="",
                scenario_count=len(scenarios),
            )
        else:
            provenance = HeldoutSourceProvenance(
                source="protected_fixture",
                fixture_digest=fixture_digest(fixture),
                generation_success=False,
                evaluator_version=GATE_C_EVALUATOR_VERSION,
                evaluator_source_sha=evaluator_sha,
                rubric_version=RUBRIC_VERSION,
                rubric_digest=rubric_hex,
                run_id=run_id,
                independence_level=INDEPENDENCE_PROCEDURAL_ONLY,
                reproducibility="protected-fixture-versioned-digest",
                blocked_reason="",
                scenario_count=len(scenarios),
            )
        return provenance, dict(fixture)
    # Neither a protected fixture nor a validated generated fixture exists.
    if baseline is None:
        reason = BLOCKED_BASELINE_UNPINNED
    else:
        reason = BLOCKED_GENERATOR_UNAVAILABLE
    return (
        _blocked_provenance(
            reason=reason, evaluator_sha=evaluator_sha, rubric_hex=rubric_hex, run_id=run_id
        ),
        None,
    )


def _category_plan(scenario_count: int) -> list[str]:
    """Round-robin category plan covering every required category.

    ``scenario_count`` is a minimum: when the round-robin pass yields
    fewer than two multi-need canonical-evidence scenarios, coverage
    appends extras (at least two information needs must require full
    canonical evidence).
    """
    plan: list[str] = []
    index = 0
    while len(plan) < scenario_count:
        plan.append(HELDOUT_REQUIRED_CATEGORIES[index % len(HELDOUT_REQUIRED_CATEGORIES)])
        index += 1
    while sum(1 for item in plan if item == "multi-need-canonical-evidence") < 2:
        plan.append("multi-need-canonical-evidence")
    return plan


def generate_unseen_fixture(
    *,
    seed: bytes,
    rubric_hex: str,
    synthesizer: Callable[[str, str, int, str], Mapping[str, Any]] | None,
    regression_prompts: Sequence[str] = (),
    scenario_count: int = 12,
    run_id: str = "local",
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], HeldoutSourceProvenance]:
    """Generate a run-unique unseen fixture via the trusted provider path.

    ``synthesizer(category, thread, index, seed_hex)`` must be backed by
    the configured real provider: small substitution/keyword templates
    pretending to be unseen are rejected by the novelty and gold-label
    validation below. A missing synthesizer (offline CI, no provider)
    raises typed BLOCKED instead of fabricating a fixture. Only the seed
    digest (never the seed) is recorded.
    """
    root = repo_root or _repo_root()
    evaluator_sha = evaluator_source_sha(root)
    if synthesizer is None:
        raise GateCHeldoutBlockedError(BLOCKED_GENERATOR_UNAVAILABLE)
    if len(seed) < MIN_SEED_BYTES:
        raise GateCHeldoutBlockedError(BLOCKED_GENERATOR_UNAVAILABLE)
    if scenario_count < MIN_HELDOUT_SCENARIOS:
        raise GateCHeldoutError("scenario_count below the held-out minimum")
    seed_hex = hashlib.sha256(bytes(seed)).hexdigest()
    plan = _category_plan(scenario_count)
    threads = [f"c-heldout-T{(idx // 3) + 1}" for idx in range(len(plan))]
    scenarios: list[dict[str, Any]] = []
    prompts: list[str] = []
    for position, category in enumerate(plan):
        produced = synthesizer(category, threads[position], position, seed_hex)
        if not isinstance(produced, Mapping):
            raise GateCHeldoutError("synthesizer must return a mapping per scenario")
        prompt = str(produced.get("prompt", ""))
        context = str(produced.get("context", ""))
        gold = str(produced.get("gold_intent_category", ""))
        safety = str(produced.get("safety_expectation", "ordinary"))
        requires_book = produced.get("requires_book_evidence")
        if not prompt.strip() or not gold.strip():
            raise GateCHeldoutError("synthesizer omitted prompt or gold label")
        if safety not in SAFETY_EXPECTATIONS:
            raise GateCHeldoutError("synthesizer returned an unknown safety expectation")
        if not isinstance(requires_book, bool):
            raise GateCHeldoutError("synthesizer omitted requires_book_evidence")
        if category == "emergency" and safety != "emergency":
            raise GateCHeldoutError("emergency category requires emergency gold label")
        if category == "out-of-scope" and safety != "out_of_book":
            raise GateCHeldoutError("out-of-scope category requires out_of_book gold label")
        scenario_id = f"c-heldout-{seed_hex[:8]}-{position:03d}"
        scenarios.append(
            {
                "id": scenario_id,
                "category": category,
                "thread": threads[position],
                "turn": position,
                "requires_book_evidence": bool(requires_book),
                "safety_expectation": safety,
                "gold_intent_category": gold,
                "prompt": prompt,
                "context": context,
            }
        )
        prompts.append(prompt)
    novelty = check_fixture_novelty(
        fixture_prompts=prompts, regression_prompts=list(regression_prompts)
    )
    if not bool(novelty["novel"]):
        raise GateCHeldoutError("generated fixture echoes regression or repeats itself")
    fixture: dict[str, Any] = {
        "version": GATE_C_HELDOUT_SCHEMA_VERSION,
        "source": "post_freeze_generated",
        "evaluator_version": GATE_C_EVALUATOR_VERSION,
        "evaluator_source_sha": evaluator_sha,
        "rubric_version": RUBRIC_VERSION,
        "rubric_digest": rubric_hex,
        "seed_sha256": seed_hex,
        "run_id": run_id,
        "scenarios": scenarios,
    }
    ok, _ = validate_heldout_fixture(fixture)
    if not ok:
        raise GateCHeldoutError("generated fixture misses required coverage")
    provenance = HeldoutSourceProvenance(
        source="post_freeze_generated",
        fixture_digest=fixture_digest(fixture),
        generation_success=True,
        evaluator_version=GATE_C_EVALUATOR_VERSION,
        evaluator_source_sha=evaluator_sha,
        rubric_version=RUBRIC_VERSION,
        rubric_digest=rubric_hex,
        run_id=run_id,
        independence_level=INDEPENDENCE_PROCEDURAL_ONLY,
        reproducibility="run-unique-seed-single-use-fixture",
        blocked_reason="",
        scenario_count=len(scenarios),
    )
    return fixture, provenance


def new_unseen_seed() -> bytes:
    """Return a run-unique unpredictable seed (never logged or reported)."""
    return secrets.token_bytes(32)


def describe_independence(
    *,
    judge_models: Sequence[str] = (),
    generator_models: Sequence[str] = (),
) -> str:
    """Report the honest independence level (never over-claim).

    Separate sessions/agents on the same provider family remain
    ``procedural_only``. Family diversity is reported only when the
    observed judge and generator model families provably differ.
    """

    def _known_provider(name: str) -> str:
        token = (name or "").strip().lower()
        if not token:
            return ""
        for prefix, provider in (
            ("openai", "openai"),
            ("chatgpt", "openai"),
            ("gpt-", "openai"),
            ("gpt_", "openai"),
            ("anthropic", "anthropic"),
            ("claude", "anthropic"),
            ("gemini", "google"),
            ("palm", "google"),
            ("bard", "google"),
            ("google", "google"),
            ("vertex", "google"),
            ("llama", "meta"),
            ("mistral", "mistral"),
            ("mixtral", "mistral"),
            ("grok", "xai"),
            ("deepseek", "deepseek"),
            ("qwen", "alibaba"),
        ):
            if token == prefix.rstrip("-_") or token.startswith(prefix):
                return provider
        for bare, provider in (("o1", "openai"), ("o3", "openai"), ("o4", "openai")):
            if token == bare or token.startswith(bare + "-") or token.startswith(bare + "_"):
                return provider
        return ""

    def _family(model: str) -> str:
        name = (model or "").strip().lower()
        if not name:
            return ""
        known = _known_provider(name)
        if known:
            return known
        head = re.split(r"[/:]", name, maxsplit=1)[0]
        known_head = _known_provider(head)
        if known_head:
            return known_head
        return head or name

    def _has_explicit_family(model: str) -> bool:
        name = (model or "").strip().lower()
        if not name:
            return False
        if "/" in name or ":" in name:
            return True
        return bool(_known_provider(name))

    judge_families = {_family(item) for item in judge_models if _family(item)}
    generator_families = {_family(item) for item in generator_models if _family(item)}
    if not judge_families or not generator_families:
        return INDEPENDENCE_PROCEDURAL_ONLY
    models = [str(item) for item in list(judge_models) + list(generator_models)]
    if any(not _has_explicit_family(item) for item in models if str(item or "").strip()):
        # A bare model name that is neither provider-prefixed (a/b) nor a
        # recognized provider alias (e.g. gpt-4o vs o1 share provider
        # "openai" but share no text prefix) cannot prove provider
        # diversity: fail closed to procedural_only instead of
        # over-claiming family_diverse.
        return INDEPENDENCE_PROCEDURAL_ONLY
    if not (judge_families & generator_families):
        return INDEPENDENCE_FAMILY_DIVERSE
    return INDEPENDENCE_PROCEDURAL_ONLY


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value) >= 0:
        return int(value)
    return None


def evaluate_heldout_turn(
    *,
    scenario_id: str,
    reply: str,
    snapshot: Mapping[str, Any],
    judge_helpful: bool | None,
    judge_addresses_intent: bool = False,
    judge_contains_substantive_claim: bool = True,
    judge_model: str = "",
    requires_book_evidence: bool = True,
    safety_expectation: str = "ordinary",
    continuity_broken: bool = False,
    receipt_kind: str = "simulated-in-process",
) -> HeldoutTurnVerdict:
    """Judge one held-out turn against exact canonical source evidence.

    A required judge that is unavailable or invalid yields ``BLOCKED``
    (typed missing prerequisite); a negative semantic verdict yields
    ``FAIL``; positive telemetry alone never yields ``PASS``. Ordinary
    non-substantive glue without AA doctrine is not spuriously required
    to cite the book.
    """
    from aa.conversation.failures import is_service_error
    from aa.conversation.turn_pipeline import leaks_internal_terms
    from aa.safety.outbound import is_outbound_safe

    clean_id = (scenario_id or "c-heldout-unknown").strip()[:64] or "c-heldout-unknown"
    model_name = (judge_model or "").strip()[:64]
    if receipt_kind not in _KNOWN_RECEIPT_KINDS:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="receipt-unknown",
        )
    if receipt_kind == "external-confirmed":
        # Gate C never proves external delivery; Gate D owns it (#340).
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="external-receipt-mislabel",
        )
    cleaned = (reply or "").strip()
    if not cleaned:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="empty-reply",
        )
    if is_service_error(cleaned):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="service-error-no-help",
        )
    try:
        outbound_ok = bool(is_outbound_safe(cleaned))
    except Exception:
        outbound_ok = False
    if not outbound_ok:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="outbound-unsafe",
        )
    lowered = cleaned.casefold()
    if (
        "http://" in lowered
        or "https://" in lowered
        or "t.me" in lowered
        or "pc-s-" in lowered
        or "chunk" in lowered
    ):
        stripped = re.sub(r"https?://\S+", " ", cleaned)
        stripped = re.sub(r"t\.me\S*", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"pc-s-[\w-]+", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"chunk[\w-]*", " ", stripped, flags=re.IGNORECASE)
        if len("".join(ch for ch in stripped if ch.isalpha())) < 8:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                receipt_kind=receipt_kind,
                failure_category="link-only-no-help",
            )
    try:
        from aa.conversation.output_limits import aggregate_quote_chars
    except Exception:
        aggregate_quote_chars = None  # type: ignore[assignment]
    if aggregate_quote_chars is not None:
        try:
            quoted = int(aggregate_quote_chars(cleaned) or 0)
        except Exception:
            quoted = 0
        if quoted > 0 and quoted / max(1, len(cleaned)) > 0.5:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                receipt_kind=receipt_kind,
                failure_category="quote-only-no-guidance",
            )
    try:
        internal_leak = bool(leaks_internal_terms(cleaned))
    except Exception:
        internal_leak = False
    if internal_leak:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="internal-detail-leak",
        )
    if not isinstance(snapshot, Mapping) or not snapshot:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="missing-turn-telemetry",
        )
    if bool(snapshot.get("turn_budget_exceeded", False)):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="turn-budget-exceeded",
        )
    unavailable = _as_int(snapshot.get("verifier_unavailable_units", 0))
    if unavailable is None or unavailable != 0:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="verifier-unavailable",
        )
    if str(snapshot.get("verifier_outcome", "")).strip() in _UNAVAILABLE_VERIFIER_OUTCOMES:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="verifier-unavailable",
        )
    if str(snapshot.get("planner_reason", "") or "").strip() in (
        "provider-error",
        "timeout",
        "invalid",
    ):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="planner-invalid",
        )
    failure_category = str(snapshot.get("failure_category", "") or "").strip()
    if failure_category not in ("", "adequacy-repair-failed"):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="failure-mark-present",
        )
    if snapshot.get("answers_request") is not True:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="not-answering-request",
        )
    if snapshot.get("qualified") is not True:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            receipt_kind=receipt_kind,
            failure_category="not-qualified",
        )
    adequacy = str(snapshot.get("adequacy_verdict", "") or "").strip()
    explicit_relevant = snapshot.get("answer_relevant", None)
    if explicit_relevant is False or adequacy == "fail":
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=not continuity_broken,
            receipt_kind=receipt_kind,
            failure_category="telemetry-not-relevant",
        )
    planner_count = _as_int(snapshot.get("planner_query_count", 0))
    passages = _as_int(snapshot.get("retrieval_passages", 0))
    verified = _as_int(snapshot.get("verified_book_units", 0))
    response_units = _as_int(snapshot.get("response_units", 0))
    if safety_expectation in ("emergency", "out_of_book"):
        # Out-of-book and emergency turns must not fabricate book support.
        # Malformed count telemetry (booleans/negatives normalize to None
        # via _as_int) fails closed: only an explicit verified == 0 passes.
        if verified is None or verified != 0:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                citation_scope={
                    "verified_book_units": int(verified or 0),
                    "response_units": int(response_units or 0),
                },
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="fabricated-book-support",
            )
    elif requires_book_evidence:
        if snapshot.get("technically_grounded") is not True:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="not-technically-grounded",
            )
        if adequacy != "pass":
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="adequacy-not-pass",
            )
        if planner_count is None or planner_count <= 0:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="no-planner-evidence",
            )
        if passages is None or passages <= 0 or verified is None or verified <= 0:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="no-verified-book-units",
            )
    unsupported = 0
    answer_outcome = str(snapshot.get("answer_outcome", "") or "").strip()
    verifier_outcome = str(snapshot.get("verifier_outcome", "") or "").strip()
    if answer_outcome not in _SERVED_ANSWER_OUTCOMES:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=not continuity_broken,
            receipt_kind=receipt_kind,
            failure_category="answer-not-served",
        )
    if failure_category == "adequacy-repair-failed" and answer_outcome not in (
        "narrowed-adequacy",
        "narrowed-adequacy-regen",
    ):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=not continuity_broken,
            receipt_kind=receipt_kind,
            failure_category="failure-mark-present",
        )
    # Every served-ish answer outcome (including narrowed-supported and
    # narrowed-compacted) must carry a clean verifier verdict: a narrowed
    # turn was not cleanly verified when verifier_outcome failed/unknown.
    if verifier_outcome not in _SERVED_VERIFIER_OUTCOMES:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=not continuity_broken,
            receipt_kind=receipt_kind,
            failure_category="verifier-not-passed",
        )
    if response_units is not None and verified is not None and response_units > 0:
        if verified > response_units:
            return HeldoutTurnVerdict(
                outcome="FAIL",
                scenario_id=clean_id,
                judge_available=judge_helpful is not None,
                judge_model=model_name,
                citation_scope={
                    "verified_book_units": int(verified),
                    "response_units": int(response_units),
                },
                relevance_to_resolved_intent=False,
                dialogue_continuity_ok=not continuity_broken,
                receipt_kind=receipt_kind,
                failure_category="citation-scope-impossible",
            )
    # Unsupported substantive claims travel as explicit telemetry: the
    # production verifier scopes natural conversational glue as supported
    # without book passages, so ``response_units - verified_book_units``
    # arithmetic alone never proves an unsupported claim (a helpful mixed
    # reply validly has verified_book_units < response_units).
    unsupported_raw = snapshot.get("unsupported_claim_count", 0)
    unsupported = 0
    if isinstance(unsupported_raw, bool):
        unsupported = 0
    elif isinstance(unsupported_raw, (int, float)) and float(unsupported_raw) > 0:
        unsupported = int(unsupported_raw)
    if unsupported > 0:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            citation_scope={
                "verified_book_units": int(verified or 0),
                "response_units": int(response_units or 0),
            },
            unsupported_claim_count=int(unsupported),
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=not continuity_broken,
            receipt_kind=receipt_kind,
            failure_category="unsupported-claims-present",
        )
    if continuity_broken:
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=judge_helpful is not None,
            judge_model=model_name,
            citation_scope={
                "verified_book_units": int(verified or 0),
                "response_units": int(response_units or 0),
            },
            unsupported_claim_count=int(unsupported),
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=False,
            receipt_kind=receipt_kind,
            failure_category="dialogue-continuity-broken",
        )
    # The independent judge is required: unavailable/invalid is BLOCKED,
    # a negative semantic verdict is FAIL, and telemetry alone never PASS.
    if judge_helpful is None:
        return HeldoutTurnVerdict(
            outcome="BLOCKED",
            scenario_id=clean_id,
            judge_available=False,
            judge_model=model_name,
            citation_scope={
                "verified_book_units": int(verified or 0),
                "response_units": int(response_units or 0),
            },
            unsupported_claim_count=int(unsupported),
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=True,
            receipt_kind=receipt_kind,
            failure_category=BLOCKED_JUDGE_UNAVAILABLE,
        )
    if not bool(judge_helpful) or not bool(judge_addresses_intent):
        return HeldoutTurnVerdict(
            outcome="FAIL",
            scenario_id=clean_id,
            judge_available=True,
            judge_model=model_name,
            citation_scope={
                "verified_book_units": int(verified or 0),
                "response_units": int(response_units or 0),
            },
            unsupported_claim_count=int(unsupported),
            relevance_to_resolved_intent=False,
            dialogue_continuity_ok=True,
            receipt_kind=receipt_kind,
            failure_category="judge-negative",
        )
    _ = judge_contains_substantive_claim
    return HeldoutTurnVerdict(
        outcome="PASS",
        scenario_id=clean_id,
        judge_available=True,
        judge_model=model_name,
        citation_scope={
            "verified_book_units": int(verified or 0),
            "response_units": int(response_units or 0),
        },
        unsupported_claim_count=int(unsupported),
        relevance_to_resolved_intent=True,
        dialogue_continuity_ok=True,
        receipt_kind=receipt_kind,
        failure_category="",
    )


def build_heldout_tier_report(
    *,
    provenance: HeldoutSourceProvenance,
    verdicts: Sequence[HeldoutTurnVerdict],
) -> HeldoutTierReport:
    """Build the privacy-safe held-out tier report (opaque ids only)."""
    if provenance.blocked_reason:
        return HeldoutTierReport(
            tier="C-heldout",
            status="BLOCKED",
            scenario_ids=(),
            passed_ids=(),
            failed_ids=(),
            failure_categories={provenance.blocked_reason: 1},
            provenance=provenance,
        )
    passed = [item.scenario_id for item in verdicts if item.outcome == "PASS"]
    failed = [item.scenario_id for item in verdicts if item.outcome == "FAIL"]
    blocked = [item for item in verdicts if item.outcome == "BLOCKED"]
    categories: dict[str, int] = {}
    for item in verdicts:
        if item.failure_category:
            categories[item.failure_category] = categories.get(item.failure_category, 0) + 1
    # A proven negative semantic verdict is never hidden as an external
    # prerequisite: FAIL outranks BLOCKED (consistent with
    # decide_gate_c_overall, where any FAIL is overall FAIL).
    if failed:
        return HeldoutTierReport(
            tier="C-heldout",
            status="FAIL",
            scenario_ids=tuple(item.scenario_id for item in verdicts),
            passed_ids=tuple(passed),
            failed_ids=tuple(failed),
            failure_categories=categories or {"heldout-verdict-missing": 1},
            provenance=provenance,
        )
    if blocked:
        return HeldoutTierReport(
            tier="C-heldout",
            status="BLOCKED",
            scenario_ids=tuple(item.scenario_id for item in verdicts),
            passed_ids=tuple(passed),
            failed_ids=tuple(failed),
            failure_categories=categories or {BLOCKED_JUDGE_UNAVAILABLE: len(blocked)},
            provenance=provenance,
        )
    if len(passed) != len(list(verdicts)) or not verdicts:
        return HeldoutTierReport(
            tier="C-heldout",
            status="FAIL",
            scenario_ids=tuple(item.scenario_id for item in verdicts),
            passed_ids=tuple(passed),
            failed_ids=tuple(failed),
            failure_categories=categories or {"heldout-verdict-missing": 1},
            provenance=provenance,
        )
    return HeldoutTierReport(
        tier="C-heldout",
        status="PASS",
        scenario_ids=tuple(item.scenario_id for item in verdicts),
        passed_ids=tuple(passed),
        failed_ids=(),
        failure_categories={},
        provenance=provenance,
    )


def build_regression_tier_report(
    *,
    scenario_ids: Sequence[str],
    failed_ids: Sequence[str] = (),
    failure_categories: Mapping[str, int] | None = None,
) -> HeldoutTierReport:
    """Build the C-regression tier report (visible cases, never unseen proof)."""
    failed_set = set(failed_ids)
    passed = [item for item in scenario_ids if item not in failed_set]
    status = "PASS" if not failed_set else "FAIL"
    return HeldoutTierReport(
        tier="C-regression",
        status=status,
        scenario_ids=tuple(scenario_ids),
        passed_ids=tuple(passed),
        failed_ids=tuple(failed_ids),
        failure_categories=dict(failure_categories or {}),
        provenance=None,
    )


def decide_gate_c_overall(*, regression_status: str, heldout_status: str) -> str:
    """Combine both tiers: PASS needs regression PASS plus unseen held-out PASS.

    A negative semantic verdict anywhere is FAIL. A missing unseen source,
    missing judge, unreadable corpus or provider outage is BLOCKED (the
    live lane surfaces it as INCOMPLETE, never as product FAIL).
    """
    if regression_status == "FAIL" or heldout_status == "FAIL":
        return "FAIL"
    if regression_status == "PASS" and heldout_status == "PASS":
        return "PASS"
    return "BLOCKED"


def build_gate_c_semantic_report(
    *,
    regression: HeldoutTierReport,
    heldout: HeldoutTierReport,
) -> GateCSemanticReport:
    """Build Gate C's single typed semantic report holding both tiers."""
    overall = decide_gate_c_overall(
        regression_status=regression.status, heldout_status=heldout.status
    )
    return GateCSemanticReport(
        schema_version=GATE_C_SEMANTIC_REPORT_VERSION,
        regression=regression,
        heldout=heldout,
        overall=overall,
    )


def write_baseline_file(
    *,
    repo_root: Path | None = None,
    path: Path | None = None,
    run_id: str = "local",
) -> dict[str, Any]:
    """Pin the evaluator source SHA plus rubric digest (trusted workflow)."""
    root = repo_root or _repo_root()
    target = path or _baseline_path(root)
    payload: dict[str, Any] = {
        "version": GATE_C_BASELINE_SCHEMA_VERSION,
        "evaluator_version": GATE_C_EVALUATOR_VERSION,
        "evaluator_source_sha": evaluator_source_sha(root),
        "rubric_version": RUBRIC_VERSION,
        "rubric_digest": rubric_digest(root),
        "run_id": run_id,
    }
    target.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _passing_book_snapshot() -> dict[str, Any]:
    return {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 3,
        "retrieval_passages": 4,
        "verified_book_units": 2,
        "response_units": 3,
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
        "answer_relevant": True,
        "failure_category": "",
        "planner_reason": "",
    }


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of the deterministic evaluator calibration controls."""

    passed: tuple[str, ...]
    failed: tuple[str, ...]

    @property
    def all_passed(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": list(self.passed),
            "failed": list(self.failed),
            "controls_executed": len(self.passed) + len(self.failed),
        }


def run_calibration_controls(
    *,
    regression_prompts: Sequence[str] = (),
) -> CalibrationResult:
    """Run deterministic false-positive/false-negative evaluator controls.

    Pure offline mechanics (scripted telemetry plus scripted judge
    verdicts); no network, no provider, no book text. A calibrated
    evaluator must pass every control below.
    """
    passed: list[str] = []
    failed: list[str] = []

    def _check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    helpful_reply = "Steady support reply with grounded guidance."
    # 1. Fixture-hidden core: a genuinely relevant book-supported answer
    #    succeeds.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-core-supported",
        reply=helpful_reply,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_contains_substantive_claim=True,
        judge_model="probe-model",
    )
    _check("calibration-relevant-grounded-passes", verdict.outcome == "PASS")

    # 2. Self-affirming telemetry with unrelated text fails: telemetry says
    #    fail/not-answering while counts look positive.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-unrelated",
        reply=helpful_reply,
        snapshot=dict(
            _passing_book_snapshot(),
            adequacy_verdict="fail",
            answers_request=False,
        ),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    _check("calibration-unrelated-telemetry-fails", verdict.outcome == "FAIL")

    # 3. Polished but book-unsupported advice fails.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-unsupported",
        reply="Polished generic advice without any book evidence.",
        snapshot=dict(
            _passing_book_snapshot(),
            verified_book_units=0,
            retrieval_passages=0,
            planner_query_count=0,
        ),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    _check("calibration-polished-unsupported-fails", verdict.outcome == "FAIL")

    # 4. Short elliptical follow-up resolved from history passes.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-elliptical",
        reply=helpful_reply,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    _check("calibration-elliptical-continuation-passes", verdict.outcome == "PASS")

    # 5. Semantic topic confusion fails even with positive counts.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-topic-confusion",
        reply=helpful_reply,
        snapshot=dict(_passing_book_snapshot(), answer_relevant=False),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    _check("calibration-topic-confusion-fails", verdict.outcome == "FAIL")

    # 6. Quote-source boundary: verified units but a quote-only dump fails.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-quote-dump",
        reply="«Поддержка рядом помогает пережить тягу спокойно и уверенно»",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    _check("calibration-quote-boundary-false-positive-fails", verdict.outcome == "FAIL")

    # 7. Multi-need long conversation covering only part of the stated
    #    information needs fails: the independent judge finds the resolved
    #    intent unaddressed even though verifier counts look positive.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-multi-need-partial",
        reply=helpful_reply,
        snapshot=dict(_passing_book_snapshot(), response_units=5, verified_book_units=2),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    _check("calibration-multi-need-partial-fails", verdict.outcome == "FAIL")

    # 7b. Explicit unsupported-claim telemetry fails even with a helpful
    #     judge: every substantive proposition needs canonical support.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-explicit-unsupported",
        reply=helpful_reply,
        snapshot=dict(_passing_book_snapshot(), unsupported_claim_count=2),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    _check(
        "calibration-explicit-unsupported-fails",
        verdict.outcome == "FAIL" and verdict.unsupported_claim_count == 2,
    )

    # 8. Generic filler: telemetry PASS but the independent judge dissents.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-filler",
        reply=helpful_reply,
        snapshot=_passing_book_snapshot(),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    _check("calibration-filler-judge-override-fails", verdict.outcome == "FAIL")

    # 9. Unhelpful link-only response fails.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-link-only",
        reply="https://example.invalid/pc-s-0001",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    _check("calibration-link-only-fails", verdict.outcome == "FAIL")

    # 10. User-visible internal infrastructure detail fails.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-internal-detail",
        reply="Retrieval corpus evidence pack verifier outcome passed.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    internal_hit = verdict.outcome == "FAIL"
    _check("calibration-internal-detail-fails", internal_hit)

    # 11. Ordinary non-substantive glue without AA doctrine needs no book
    #     citation and passes.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-glue",
        reply="I hear you. What feels most important right now?",
        snapshot={
            "answer_outcome": "served",
            "verifier_outcome": "passed",
            "verifier_unavailable_units": 0,
            "turn_budget_exceeded": False,
            "planner_query_count": 0,
            "retrieval_passages": 0,
            "verified_book_units": 0,
            "response_units": 0,
            "adequacy_verdict": "pass",
            "answers_request": True,
            "technically_grounded": True,
            "qualified": True,
            "answer_relevant": True,
            "failure_category": "",
            "planner_reason": "",
        },
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_contains_substantive_claim=False,
        judge_model="probe-model",
        requires_book_evidence=False,
    )
    _check("calibration-glue-without-doctrine-passes", verdict.outcome == "PASS")

    # 12. Required judge unavailable or invalid is BLOCKED, never PASS/FAIL.
    verdict = evaluate_heldout_turn(
        scenario_id="cal-no-judge",
        reply=helpful_reply,
        snapshot=_passing_book_snapshot(),
        judge_helpful=None,
        judge_model="",
    )
    _check(
        "calibration-judge-unavailable-blocked",
        verdict.outcome == "BLOCKED" and verdict.judge_available is False,
    )

    # 13. A visible checked-in fixture alone never makes held-out PASS.
    visible_prompts = [item for item in regression_prompts if item.strip()]
    if visible_prompts:
        novelty = check_fixture_novelty(
            fixture_prompts=[visible_prompts[0]], regression_prompts=visible_prompts
        )
        _check("calibration-visible-fixture-never-heldout-pass", novelty["novel"] is False)
    else:
        novelty = check_fixture_novelty(
            fixture_prompts=["Some held-out style prompt."],
            regression_prompts=["Some held-out style prompt."],
        )
        _check("calibration-visible-fixture-never-heldout-pass", novelty["novel"] is False)

    # 14. Two independent input seeds across a stable rubric version give
    #     distinct fixture digests (run-unique unpredictability proof).
    def _scripted_synth(category: str, thread: str, index: int, seed_hex: str) -> Mapping[str, Any]:
        safety = "ordinary"
        if category == "emergency":
            safety = "emergency"
        elif category == "out-of-scope":
            safety = "out_of_book"
        return {
            "prompt": f"synth-{seed_hex[:8]}-{index}-{category}",
            "context": f"thread-{thread}",
            "gold_intent_category": category,
            "safety_expectation": safety,
            "requires_book_evidence": safety == "ordinary",
        }

    try:
        first, _ = generate_unseen_fixture(
            seed=b"0123456789abcdef-seed-one....",
            rubric_hex="r" * 64,
            synthesizer=_scripted_synth,
            regression_prompts=list(regression_prompts),
            scenario_count=12,
            run_id="calibration",
        )
        second, _ = generate_unseen_fixture(
            seed=b"fedcba9876543210-seed-two....",
            rubric_hex="r" * 64,
            synthesizer=_scripted_synth,
            regression_prompts=list(regression_prompts),
            scenario_count=12,
            run_id="calibration",
        )
        _check(
            "calibration-two-seeds-distinct-digests",
            fixture_digest(first) != fixture_digest(second),
        )
    except GateCHeldoutError:
        _check("calibration-two-seeds-distinct-digests", False)

    # 15. Missing generator (offline, no provider) is BLOCKED, never faked.
    try:
        generate_unseen_fixture(
            seed=b"0123456789abcdef-seed-one....",
            rubric_hex="r" * 64,
            synthesizer=None,
            scenario_count=12,
            run_id="calibration",
        )
        _check("calibration-missing-generator-blocked", False)
    except GateCHeldoutBlockedError as exc:
        _check(
            "calibration-missing-generator-blocked",
            exc.reason == BLOCKED_GENERATOR_UNAVAILABLE,
        )

    return CalibrationResult(passed=tuple(passed), failed=tuple(failed))


__all__ = [
    "BASELINE_PATH",
    "BLOCKED_BASELINE_MISMATCH",
    "BLOCKED_BASELINE_UNPINNED",
    "BLOCKED_CORPUS_UNREADABLE",
    "BLOCKED_FIXTURE_UNAVAILABLE",
    "BLOCKED_GENERATOR_UNAVAILABLE",
    "BLOCKED_JUDGE_UNAVAILABLE",
    "BLOCKED_PREREQUISITES_UNAVAILABLE",
    "BLOCKED_PROVIDER_UNAVAILABLE",
    "ENV_BASELINE_PATH",
    "ENV_FIXTURE_PATH",
    "GATE_C_BASELINE_SCHEMA_VERSION",
    "GATE_C_EVALUATOR_VERSION",
    "GATE_C_HELDOUT_SCHEMA_VERSION",
    "GATE_C_SEMANTIC_REPORT_VERSION",
    "HELDOUT_REQUIRED_CATEGORIES",
    "HELDOUT_SOURCES",
    "HELDOUT_STATUSES",
    "INDEPENDENCE_FAMILY_DIVERSE",
    "INDEPENDENCE_PROCEDURAL_ONLY",
    "MIN_HELDOUT_SCENARIOS",
    "MIN_SEED_BYTES",
    "RUBRIC_PATH",
    "RUBRIC_VERSION",
    "SAFETY_EXPECTATIONS",
    "SEALED_FIXTURE_PATHS",
    "CalibrationResult",
    "GateCHeldoutBlockedError",
    "GateCHeldoutError",
    "GateCSemanticReport",
    "HeldoutSourceProvenance",
    "HeldoutTierReport",
    "HeldoutTurnVerdict",
    "build_gate_c_semantic_report",
    "build_heldout_tier_report",
    "build_regression_tier_report",
    "check_fixture_novelty",
    "decide_gate_c_overall",
    "describe_independence",
    "evaluate_heldout_turn",
    "evaluator_source_sha",
    "fixture_digest",
    "generate_unseen_fixture",
    "new_unseen_seed",
    "normalize_dialogue_text",
    "resolve_heldout_source",
    "rubric_digest",
    "run_calibration_controls",
    "validate_heldout_fixture",
    "write_baseline_file",
]
