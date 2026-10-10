"""Unified Gate A/B evidence (V2) and read-only Gate F reduction (issue #339).

Single authoritative verdict per gate. Gates A and B each execute their
checks exactly once per workflow attempt and publish one atomic, durable,
validated evidence artifact. Gate F is a pure deterministic reducer over
already-produced, validated A-E evidence: it performs no corpus restore,
no provider calls, no index builds, and no component re-evaluation.

Fail-closed: legacy evidence (schema 1 / unversioned), absent artifacts,
invalid digests, old SHA, stale run identity, omitted checks, missing or
forged artifacts, contradictory proof, and tampering all yield
BLOCKED/FAIL, never PASS.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.qualification.self_proving import (
    GATE_STATUSES,
    MANDATORY_GATES,
    FinalVerdict,
    GateEvidence,
    SelfProvingError,
    assert_no_text_leak,
    effective_gate_status,
    validate_exact_sha,
    validate_hex64,
)

SCHEMA_VERSION = "aa-gate-evidence/2"
VERDICT_SCHEMA_VERSION = "aa-self-proving-verdict/2"
COVERAGE_SCHEMA_VERSION = "aa-gate-ab-coverage/1"

EVIDENCE_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "BLOCKED", "STALE")

GATE_A_EVIDENCE_FILE = "gate-A-evidence.json"
GATE_B_EVIDENCE_FILE = "gate-B-evidence.json"
GATE_EVIDENCE_FILES: dict[str, str] = {
    "A": GATE_A_EVIDENCE_FILE,
    "B": GATE_B_EVIDENCE_FILE,
    "C": "gate-C-evidence.json",
    "D": "gate-D-evidence.json",
    "E": "gate-E-evidence.json",
}

# Machine-readable inventory of every Gate A assertion. Each row maps one
# executable check to its owner, input, and proof artifact so no assertion
# can be silently dropped during unification.
GATE_A_COVERAGE: tuple[dict[str, str], ...] = (
    {
        "check_id": "a-exact-sha",
        "owner": "gate-a",
        "input": "git rev-parse HEAD + expected main SHA",
        "executable_check": "scripts/run_self_proving_qualification.py:_gate_a sha match",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-exact-sha]",
    },
    {
        "check_id": "a-clean-tree",
        "owner": "gate-a",
        "input": "git status --porcelain (ignored output dirs excluded)",
        "executable_check": "runner _gate_a dirty-tree scan + workflow sha step",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-clean-tree]",
    },
    {
        "check_id": "a-product-fingerprint",
        "owner": "gate-a",
        "input": "repository product fingerprint material",
        "executable_check": "product_fingerprint.compute_product_fingerprint 64-hex",
        "proof_artifact": "gate-A-evidence.json:product_fingerprint",
    },
    {
        "check_id": "a-runtime-fingerprint",
        "owner": "gate-a",
        "input": "runtime model/agent policy material",
        "executable_check": "scripts/run_self_proving_qualification.py:_runtime_fingerprint 64-hex",
        "proof_artifact": "gate-A-evidence.json:runtime_fingerprint",
    },
    {
        "check_id": "a-model-policy",
        "owner": "gate-a",
        "input": "Settings.from_env agent/model/fallback policy",
        "executable_check": "aa.config defaults identity (agent/model/fallback distinct)",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-model-policy]",
    },
    {
        "check_id": "a-shell-aa-check",
        "owner": "gate-a-shell",
        "input": "production entrypoint",
        "executable_check": "scripts/verify.sh: python3 -m aa --check",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-aa-check]",
    },
    {
        "check_id": "a-shell-pytest",
        "owner": "gate-a-shell",
        "input": "full unit suite",
        "executable_check": "scripts/verify.sh: python3 -m pytest -q",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-pytest]",
    },
    {
        "check_id": "a-shell-ruff-check",
        "owner": "gate-a-shell",
        "input": "repository tree",
        "executable_check": "scripts/verify.sh: python3 -m ruff check .",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-ruff-check]",
    },
    {
        "check_id": "a-shell-ruff-format",
        "owner": "gate-a-shell",
        "input": "repository tree",
        "executable_check": "scripts/verify.sh: python3 -m ruff format --check .",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-ruff-format]",
    },
    {
        "check_id": "a-shell-mypy",
        "owner": "gate-a-shell",
        "input": "src + tests",
        "executable_check": "scripts/verify.sh: python3 -m mypy src tests",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-mypy]",
    },
    {
        "check_id": "a-shell-contract-qualification",
        "owner": "gate-a-shell",
        "input": "product contract fixtures",
        "executable_check": "scripts/verify_product_contract_qualification.py --require-active",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-contract-qualification]",
    },
    {
        "check_id": "a-shell-runtime-qualification",
        "owner": "gate-a-shell",
        "input": "runtime fixtures",
        "executable_check": "scripts/verify_runtime_qualification.py",
        "proof_artifact": "gate-A-evidence.json:subchecks[a-shell-runtime-qualification]",
    },
)

# Machine-readable inventory of every Gate B assertion. Shell bootstrap /
# pytest checks and Python component checks are complementary families kept
# together under ONE B verdict (duplicates are recorded, never dropped).
GATE_B_COVERAGE: tuple[dict[str, str], ...] = (
    {
        "check_id": "b-shell-restore-canonical",
        "owner": "gate-b-shell",
        "input": "pinned RU/EN corpus artifacts",
        "executable_check": "scripts/restore_canonical.py --no-network-fallback (+ --lang ru)",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-restore-canonical]",
    },
    {
        "check_id": "b-shell-corpus-structure",
        "owner": "gate-b-shell",
        "input": "restored canonical corpus",
        "executable_check": "scripts/build_corpus_structure.py",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-corpus-structure]",
    },
    {
        "check_id": "b-shell-prefetch-public",
        "owner": "gate-b-shell",
        "input": "pinned public model assets",
        "executable_check": "scripts/prefetch_public_assets.py",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-prefetch-public]",
    },
    {
        "check_id": "b-shell-prefetch-voice",
        "owner": "gate-b-shell",
        "input": "pinned voice assets",
        "executable_check": "scripts/prefetch_voice_assets.py",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-prefetch-voice]",
    },
    {
        "check_id": "b-shell-build-index",
        "owner": "gate-b-shell",
        "input": "restored corpus + pinned E5 model",
        "executable_check": "scripts/build_retrieval_index.py --backend e5",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-build-index]",
    },
    {
        "check_id": "b-shell-pytest",
        "owner": "gate-b-shell",
        "input": "deterministic component suites",
        "executable_check": "pytest 8 deterministic component suites (workflow Gate B step)",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-shell-pytest]",
    },
    {
        "check_id": "b-planner-cardinality",
        "owner": "gate-b-python",
        "input": "QueryPlan 12-query valid + 17-query oversize",
        "executable_check": "scripts/run_self_proving_qualification.py:_gate_b planner cardinality",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-planner-cardinality]",
    },
    {
        "check_id": "b-planner-shape",
        "owner": "gate-b-python",
        "input": "RetrievalConfig + evidence module wiring",
        "executable_check": "runner _gate_b planner shape / RRF-only",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-planner-shape]",
    },
    {
        "check_id": "b-corpus-restore",
        "owner": "gate-b-python",
        "input": "corpus/embedding.lock.json + corpus/ + generated artifacts",
        "executable_check": "runner _gate_b corpus-restore layout",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-corpus-restore]",
    },
    {
        "check_id": "b-bm25-index",
        "owner": "gate-b-python",
        "input": "production lexical index (RAM-resident SQLite FTS)",
        "executable_check": "runner _gate_b bm25-index open/search",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-bm25-index]",
    },
    {
        "check_id": "b-e5-faiss-index",
        "owner": "gate-b-python",
        "input": "pinned E5 FAISS production dense index",
        "executable_check": "runner _gate_b e5-faiss-index identity/search",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-e5-faiss-index]",
    },
    {
        "check_id": "b-retrieval-rrf",
        "owner": "gate-b-python",
        "input": "12 production RU queries over hybrid index",
        "executable_check": "scripts/run_self_proving_qualification.py:_gate_b fused-hit evidence",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-retrieval-rrf]",
    },
    {
        "check_id": "b-small-to-big",
        "owner": "gate-b-python",
        "input": "fused retrieval hits",
        "executable_check": "runner _gate_b canonical provenance",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-small-to-big]",
    },
    {
        "check_id": "b-grounding",
        "owner": "gate-b-python",
        "input": "supported + unsupported exact-source claims",
        "executable_check": "runner _gate_b check_grounding accept/reject",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-grounding]",
    },
    {
        "check_id": "b-verifier",
        "owner": "gate-b-python",
        "input": "valid + incomplete grounding verdicts",
        "executable_check": "runner _gate_b validate_grounding_result",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-verifier]",
    },
    {
        "check_id": "b-output-envelope",
        "owner": "gate-b-python",
        "input": "short + overflow + compacted replies",
        "executable_check": "runner _gate_b envelope accept/reject/compact",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-output-envelope]",
    },
    {
        "check_id": "b-memory-fifo",
        "owner": "gate-b-python",
        "input": "thread mapping + query vector cache bounds",
        "executable_check": "runner _gate_b memory/FIFO determinism",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-memory-fifo]",
    },
    {
        "check_id": "b-concurrency",
        "owner": "gate-b-python",
        "input": "ChatTurnDispatcher lifecycle",
        "executable_check": "runner _gate_b dispatcher start/stop",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-concurrency]",
    },
    {
        "check_id": "b-safety",
        "owner": "gate-b-python",
        "input": "empty + ordinary utterances",
        "executable_check": "runner _gate_b SafetyRouter BLOCK/ALLOW",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-safety]",
    },
    {
        "check_id": "b-voice-fixture",
        "owner": "gate-b-python",
        "input": "corpus/voice.lock.json",
        "executable_check": "scripts/run_self_proving_qualification.py:_gate_b voice lock load",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-voice-fixture]",
    },
    {
        "check_id": "b-voice-cache",
        "owner": "gate-b-python",
        "input": "VoicePresentationClassifier + resource usage",
        "executable_check": "runner _gate_b voice-cache/resource",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-voice-cache]",
    },
    {
        "check_id": "b-resource",
        "owner": "gate-b-python",
        "input": "process resource usage",
        "executable_check": "scripts/run_self_proving_qualification.py:_gate_b ru_maxrss readable",
        "proof_artifact": "gate-B-evidence.json:subchecks[b-resource]",
    },
)

REQUIRED_A_CHECKS: tuple[str, ...] = tuple(row["check_id"] for row in GATE_A_COVERAGE)
REQUIRED_B_CHECKS: tuple[str, ...] = tuple(row["check_id"] for row in GATE_B_COVERAGE)


@dataclass(frozen=True)
class SubcheckResult:
    """One executed subcheck inside a Gate A/B evidence artifact."""

    check_id: str
    status: str
    component: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "component": self.component,
            "detail": self.detail,
        }

    @staticmethod
    def from_dict(payload: Any) -> SubcheckResult:
        if not isinstance(payload, dict):
            raise SelfProvingError("subcheck must be an object")
        check_id = str(payload.get("check_id", ""))
        status = str(payload.get("status", "")).upper()
        if not check_id:
            raise SelfProvingError("subcheck missing check_id")
        if status not in ("PASS", "FAIL", "BLOCKED"):
            raise SelfProvingError(f"unknown subcheck status {status!r}")
        component = str(payload.get("component", "") or "")
        detail = str(payload.get("detail", "") or "")
        if len(detail) > 160:
            raise SelfProvingError("subcheck detail too long")
        assert_no_text_leak(payload, owner="gate-evidence-subcheck")
        return SubcheckResult(
            check_id=check_id, status=status, component=component, detail=detail[:160]
        )


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def evidence_checksum(payload: dict[str, Any]) -> str:
    """Checksum over the immutable evidence body (excludes the checksum)."""
    return hashlib.sha256(_canonical_payload(payload)).hexdigest()


def build_evidence_body(
    *,
    gate: str,
    status: str,
    sha: str,
    product_fingerprint: str,
    runtime_fingerprint: str,
    component: str,
    failure_category: str,
    run_id: str,
    subchecks: list[SubcheckResult],
    detail: str = "",
    live_trusted: bool = False,
    mocked_only: bool = False,
    latency_p50_ms: float = 0.0,
    latency_p95_ms: float = 0.0,
    max_turn_ms: float = 0.0,
) -> dict[str, Any]:
    normalized_gate = (gate or "").strip().upper()
    if normalized_gate not in ("A", "B", "C", "D", "E"):
        raise SelfProvingError(f"unknown evidence gate {gate!r}")
    normalized_status = (status or "").strip().upper()
    if normalized_status not in EVIDENCE_STATUSES:
        raise SelfProvingError(f"unknown evidence status {status!r}")
    sha_n = validate_exact_sha(sha)
    product = validate_hex64(product_fingerprint)
    runtime = validate_hex64(runtime_fingerprint)
    if not run_id.strip() or any(c.isspace() for c in run_id):
        raise SelfProvingError("run_id must be a non-empty token")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": normalized_gate,
        "status": normalized_status,
        "sha": sha_n,
        "product_fingerprint": product,
        "runtime_fingerprint": runtime,
        "failure_category": (failure_category or "").strip()[:96],
        "component": (component or "").strip()[:96],
        "run_id": run_id.strip(),
        "live_trusted": bool(live_trusted),
        "mocked_only": bool(mocked_only),
        "latency_p50_ms": float(latency_p50_ms),
        "latency_p95_ms": float(latency_p95_ms),
        "max_turn_ms": float(max_turn_ms),
        "detail": (detail or "")[:160],
        "subchecks": [item.to_dict() for item in subchecks],
    }
    assert_no_text_leak(body, owner="gate-evidence")
    return body


def sign_evidence_body(body: dict[str, Any]) -> dict[str, Any]:
    """Attach the immutable-content checksum to an evidence body."""
    if body.get("schema_version") != SCHEMA_VERSION:
        raise SelfProvingError("refusing to sign non-V2 evidence body")
    digest = evidence_checksum(body)
    signed = dict(body)
    signed["evidence_checksum"] = digest
    assert_no_text_leak(signed, owner="gate-evidence")
    return signed


def write_evidence_atomic(out_dir: Path, signed: dict[str, Any]) -> Path:
    """Atomically publish one validated evidence artifact (durable)."""
    gate = str(signed.get("gate", ""))
    filename = GATE_EVIDENCE_FILES.get(gate)
    if filename is None:
        raise SelfProvingError(f"unknown evidence gate {gate!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=str(out_dir))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(signed, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass
    return target


def load_and_validate_evidence(
    path: Path,
    *,
    expected_gate: str,
    expected_sha: str,
    expected_run_id: str,
    expected_product: str = "",
    expected_runtime: str = "",
) -> dict[str, Any]:
    """Load one evidence artifact and validate it fail-closed.

    Legacy (unversioned / schema 1), absent, unreadable, digest-mismatched,
    wrong-gate, old-SHA, stale-run, fingerprint-mismatched, or
    coverage-incomplete artifacts all raise :class:`SelfProvingError`.
    """
    if not path.is_file():
        raise SelfProvingError(f"missing {expected_gate} evidence artifact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SelfProvingError(f"{expected_gate} evidence unreadable") from exc
    if not isinstance(payload, dict):
        raise SelfProvingError(f"{expected_gate} evidence invalid shape")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SelfProvingError(f"{expected_gate} legacy evidence rejected")
    digest = str(payload.get("evidence_checksum", "") or "")
    if len(digest) != 64:
        raise SelfProvingError(f"{expected_gate} evidence missing checksum")
    body = {key: value for key, value in payload.items() if key != "evidence_checksum"}
    if evidence_checksum(body) != digest.lower():
        raise SelfProvingError(f"{expected_gate} evidence digest mismatch")
    if str(payload.get("gate", "")).upper() != expected_gate.upper():
        raise SelfProvingError(f"{expected_gate} evidence gate mismatch")
    if str(payload.get("sha", "")).lower() != validate_exact_sha(expected_sha):
        raise SelfProvingError(f"{expected_gate} evidence stale SHA")
    if str(payload.get("run_id", "")) != expected_run_id:
        raise SelfProvingError(f"{expected_gate} evidence stale run")
    if expected_product and str(payload.get("product_fingerprint", "")).lower() != (
        expected_product.lower()
    ):
        raise SelfProvingError(f"{expected_gate} evidence product mismatch")
    if expected_runtime and str(payload.get("runtime_fingerprint", "")).lower() != (
        expected_runtime.lower()
    ):
        raise SelfProvingError(f"{expected_gate} evidence runtime mismatch")
    status = str(payload.get("status", "")).upper()
    if status not in EVIDENCE_STATUSES:
        raise SelfProvingError(f"{expected_gate} evidence unknown status")
    raw_subchecks = payload.get("subchecks", [])
    if expected_gate in ("A", "B") and not isinstance(raw_subchecks, list):
        raise SelfProvingError(f"{expected_gate} evidence missing subchecks")
    subchecks = [SubcheckResult.from_dict(item) for item in (raw_subchecks or [])]
    required = (
        REQUIRED_A_CHECKS
        if expected_gate == "A"
        else (REQUIRED_B_CHECKS if expected_gate == "B" else ())
    )
    if required:
        seen = {item.check_id for item in subchecks}
        missing = [name for name in required if name not in seen]
        if missing:
            raise SelfProvingError(
                f"{expected_gate} evidence omitted checks: {','.join(missing[:4])}"
            )
    assert_no_text_leak(payload, owner="gate-evidence")
    return payload


def evidence_to_gate_evidence(payload: dict[str, Any]) -> GateEvidence:
    """Convert validated V2 evidence into the contract GateEvidence."""
    return GateEvidence(
        gate=str(payload.get("gate", "")),
        status=str(payload.get("status", "")),
        sha=str(payload.get("sha", "")),
        product_fingerprint=str(payload.get("product_fingerprint", "") or ""),
        runtime_fingerprint=str(payload.get("runtime_fingerprint", "") or ""),
        failure_category=str(payload.get("failure_category", "") or ""),
        component=str(payload.get("component", "") or ""),
        run_id=str(payload.get("run_id", "") or ""),
        live_trusted=bool(payload.get("live_trusted", False)),
        mocked_only=bool(payload.get("mocked_only", False)),
        latency_p50_ms=float(payload.get("latency_p50_ms", 0.0) or 0.0),
        latency_p95_ms=float(payload.get("latency_p95_ms", 0.0) or 0.0),
        max_turn_ms=float(payload.get("max_turn_ms", 0.0) or 0.0),
        detail=str(payload.get("detail", "") or ""),
    )


def failed_gates_for_verdict(verdict: FinalVerdict) -> list[str]:
    """All independently failed gates (FAIL or BLOCKED/STALE), in gate order."""
    ordered = [gate for gate in MANDATORY_GATES]
    failed: list[str] = []
    by_gate = {item.gate: item for item in verdict.gates}
    for gate in ordered:
        item = by_gate.get(gate)
        if item is None:
            failed.append(gate)
            continue
        try:
            resolved = effective_gate_status(item, current_sha=verdict.sha)
        except SelfProvingError:
            failed.append(gate)
            continue
        if resolved != "PASS":
            failed.append(gate)
    return failed


def root_components_for_verdict(verdict: FinalVerdict) -> list[str]:
    """Typed repair roots: gate:category:component for every failed gate."""
    roots: list[str] = []
    by_gate = {item.gate: item for item in verdict.gates}
    for gate in failed_gates_for_verdict(verdict):
        item = by_gate.get(gate)
        if item is None:
            roots.append(f"{gate}:missing-evidence:unknown")
            continue
        category = (item.failure_category or item.status or "unknown").strip() or "unknown"
        component = (item.component or "unknown").strip() or "unknown"
        roots.append(f"{gate}:{category}:{component}")
    return roots


def verdict_summary_v2(verdict: FinalVerdict, *, failures: list[dict[str, Any]]) -> dict[str, Any]:
    """V2 summary: blocking_gate (human) + failed_gates (all) + roots."""
    payload = verdict.to_dict()
    payload["schema_version"] = VERDICT_SCHEMA_VERSION
    payload["failed_gates"] = failed_gates_for_verdict(verdict)
    payload["root_components"] = root_components_for_verdict(verdict)
    payload["failures"] = failures
    assert_no_text_leak(payload, owner="gate-verdict")
    return payload


def coverage_table() -> dict[str, Any]:
    """Machine-readable Gate A/B coverage mapping (no silent drops)."""
    gate_a = [dict(row, status="active") for row in GATE_A_COVERAGE]
    gate_b = [dict(row, status="active") for row in GATE_B_COVERAGE]
    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "gate_a": gate_a,
        "gate_b": gate_b,
    }
    assert_no_text_leak(payload, owner="gate-coverage")
    return payload


def validate_no_status_in_vocab(status: str) -> str:
    normalized = (status or "").strip().upper()
    if normalized not in GATE_STATUSES:
        raise SelfProvingError(f"unknown gate status {status!r}")
    return normalized


__all__ = [
    "COVERAGE_SCHEMA_VERSION",
    "EVIDENCE_STATUSES",
    "GATE_A_COVERAGE",
    "GATE_B_COVERAGE",
    "GATE_EVIDENCE_FILES",
    "REQUIRED_A_CHECKS",
    "REQUIRED_B_CHECKS",
    "SCHEMA_VERSION",
    "VERDICT_SCHEMA_VERSION",
    "SubcheckResult",
    "build_evidence_body",
    "coverage_table",
    "evidence_checksum",
    "evidence_to_gate_evidence",
    "failed_gates_for_verdict",
    "load_and_validate_evidence",
    "root_components_for_verdict",
    "sign_evidence_body",
    "verdict_summary_v2",
    "write_evidence_atomic",
]
