"""Trusted real-book retrieval qualification harness (issue #131).

Least-privilege validation-run support for #130: executes the frozen
real-world benchmark against the actual production retrieval substrate
(real RU canonical corpus plus RAM-resident BM25 + E5/FAISS index plus
the mandatory planner through the production OpenCode structured-output
adapter/profile) without exposing secrets or plaintext book text.

Privacy contract (fail-closed):

- exact evidence text exists only inside the trusted job and the
  compressed + age-encrypted result artifact;
- logs and the public summary carry IDs, checksums, counts and
  latencies only, never raw literary text;
- ``AA_BOOK_AGE_IDENTITY`` is never printed;
- decrypted corpus/index artifacts never enter the Actions cache;
- the frozen oracle is evaluation-only and never reaches
  planner/generator inputs.

Status contract: ``PASS`` / ``FAIL`` / ``INCOMPLETE`` / ``STALE``.
Missing E5/planner/provider prerequisites yield ``INCOMPLETE``, never a
fake ``PASS``. No BGE/cross-encoder reranking exists on this path.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd

from aa.corpus import age_v1
from aa.qualification.conversation_eval import FORBIDDEN_GENERATOR_KEYS
from aa.retrieval.evidence import (
    BRANCH_TOP_K,
    MAX_PER_SECTION,
    NEIGHBOR_WINDOW,
    POOL_CAP,
    TOP_CHILD_CAP,
    RetrievalConfig,
)
from aa.retrieval.fusion import RRF_K
from aa.retrieval.index import INDEX_FORMAT

WORKFLOW_NAME = "aa-real-book-retrieval-qualification.yml"
RESULT_ISSUE = 130
SCHEMA_VERSION = "aa-real-book-retrieval-qualification/1"
PUBLIC_SUMMARY_VERSION = "aa-real-book-retrieval-summary/1"
PROTECTED_ARTIFACT_VERSION = "aa-real-book-retrieval-protected/1"

BENCHMARK_INPUT_REL = "qualification/ru_realworld_alcohol_help.v1_1.input.jsonl"
BENCHMARK_ORACLE_REL = "qualification/ru_realworld_alcohol_help.v1_1.oracle.jsonl"
EMBEDDING_LOCK_REL = "corpus/embedding.lock.json"
RU_MANIFEST_REL = "corpus/canonical.ru.manifest.json"

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_BACKEND_NAME = "intfloat-multilingual-e5-base/1"

PLANNER_AGENT = "aa-planner-v2"
PLANNER_PROFILE = "opencode-structured-output/json_schema"
PLANNER_SCHEMA_NAME = "QueryPlan"

RETRIEVAL_CONFIG_VERSION = "v2-rrf-only/1"

VALID_STATUSES = ("PASS", "FAIL", "INCOMPLETE", "STALE")

EXIT_BY_STATUS = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "STALE": 3}

FORBIDDEN_RERANKER_FRAGMENTS = (
    "bge",
    "cross-encoder",
    "cross_encoder",
    "crossencoder",
    "rerank",
)

FORBIDDEN_PUBLIC_KEYS = frozenset(
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
        "chain_of_thought",
        "hidden_reasoning",
        "secret",
        "identity",
        "private_key",
    }
)

_FORBIDDEN_ORACLE_KEY_RE = re.compile(r"^expected_.*|^oracle$|^labels?$")

_MARKER_RE = re.compile(
    r"<!--\s*aa-real-book-retrieval-result\s+"
    r"issue=130\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"corpus=(?P<corpus>[0-9a-f]{64})\s+"
    r"benchmark=(?P<benchmark>[0-9a-f]{64})\s+"
    r"retrieval=(?P<retrieval>[0-9a-f]{64})\s+"
    r"result=(?P<result>PASS|FAIL|INCOMPLETE|STALE)\s+"
    r"run=(?P<run>\S+?)\s*-->"
)


class RealBookRetrievalError(ValueError):
    """Raised when trusted qualification invariants fail (fails closed)."""


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` (UTF-8)."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def secret_present(env: dict[str, str] | None = None) -> bool:
    """Return True when the trusted age identity is configured."""
    import os

    source = os.environ if env is None else env
    return bool(str(source.get("AA_BOOK_AGE_IDENTITY", "")).strip())


def require_secret(env: dict[str, str] | None = None) -> None:
    """Fail closed when the trusted age identity is missing."""
    if not secret_present(env):
        raise RealBookRetrievalError("AA_BOOK_AGE_IDENTITY is not configured; refusing trusted run")


def validate_exact_sha(value: str) -> str:
    """Validate an exact 40-hex main SHA (fails closed)."""
    cleaned = value.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", cleaned):
        raise RealBookRetrievalError("exact SHA must be a 40-hex digest")
    return cleaned


def assert_exact_sha_binding(*, checked_out: str, expected: str) -> None:
    """Fail closed unless the checked-out SHA equals the expected SHA."""
    if validate_exact_sha(checked_out) != validate_exact_sha(expected):
        raise RealBookRetrievalError("checked-out SHA does not match the trusted exact SHA")


def retrieval_config_dict(lock: dict[str, Any]) -> dict[str, Any]:
    """Return the pinned production retrieval config for checksum binding."""
    model_id = str(lock.get("model_id", ""))
    revision = str(lock.get("revision", ""))
    if model_id != EMBEDDING_MODEL_ID:
        raise RealBookRetrievalError(f"unexpected embedding model id: {model_id!r}")
    if revision != EMBEDDING_REVISION:
        raise RealBookRetrievalError("embedding revision is not the pinned snapshot")
    config = RetrievalConfig()
    return {
        "config_version": RETRIEVAL_CONFIG_VERSION,
        "branch_top_k": config.branch_top_k,
        "rrf_k": config.rrf_k,
        "pool_cap": config.pool_cap,
        "top_child_cap": config.top_child_cap,
        "max_per_section": config.max_per_section,
        "neighbor_window": config.neighbor_window,
        "budget_tokens": config.budget_tokens,
        "embedding_model_id": model_id,
        "embedding_revision": revision,
        "embedding_backend": E5_BACKEND_NAME,
        "planner_agent": PLANNER_AGENT,
        "planner_profile": PLANNER_PROFILE,
        "planner_schema": PLANNER_SCHEMA_NAME,
        "index_format": INDEX_FORMAT,
        "defaults": {
            "branch_top_k": BRANCH_TOP_K,
            "rrf_k": RRF_K,
            "pool_cap": POOL_CAP,
            "top_child_cap": TOP_CHILD_CAP,
            "max_per_section": MAX_PER_SECTION,
            "neighbor_window": NEIGHBOR_WINDOW,
        },
    }


def retrieval_config_checksum(config: dict[str, Any]) -> str:
    """Return the SHA-256 binding the exact retrieval configuration."""
    payload = (json.dumps(config, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return sha256_bytes(payload)


def production_retrieval_checksum(repo_root: Path | None = None) -> str:
    """Bind the production retrieval config at ``repo_root`` by checksum."""
    root = repo_root or find_repo_root()
    lock = json.loads((root / EMBEDDING_LOCK_REL).read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise RealBookRetrievalError("embedding lock must be a JSON object")
    return retrieval_config_checksum(retrieval_config_dict(lock))


def benchmark_checksums(repo_root: Path | None = None) -> dict[str, str]:
    """Return stable input/oracle checksums for the frozen benchmark."""
    root = repo_root or find_repo_root()
    return {
        "input": sha256_file(root / BENCHMARK_INPUT_REL),
        "oracle": sha256_file(root / BENCHMARK_ORACLE_REL),
    }


def corpus_checksum(repo_root: Path | None = None) -> str:
    """Return the canonical RU corpus checksum from the committed manifest."""
    root = repo_root or find_repo_root()
    manifest = json.loads((root / RU_MANIFEST_REL).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RealBookRetrievalError("RU manifest must be a JSON object")
    sha = manifest.get("artifact_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RealBookRetrievalError("RU manifest artifact_sha256 is invalid")
    return sha


def find_repo_root() -> Path:
    """Return the repository root containing the frozen benchmark."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / BENCHMARK_INPUT_REL).exists():
            return parent
    raise RealBookRetrievalError("repository root with frozen benchmark not found")


def assert_no_oracle_leak(payload: Any, owner: str = "planner-input") -> None:
    """Reject any oracle/evaluation-only key in planner/generator inputs."""
    if isinstance(payload, dict):
        for key in payload:
            if key in FORBIDDEN_GENERATOR_KEYS:
                raise RealBookRetrievalError(
                    f"{owner}: oracle leak: {key!r} must not reach planner/generator"
                )
            if isinstance(key, str) and _FORBIDDEN_ORACLE_KEY_RE.match(key):
                raise RealBookRetrievalError(
                    f"{owner}: oracle leak: {key!r} must not reach planner/generator"
                )
        for key, value in payload.items():
            assert_no_oracle_leak(value, f"{owner}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            assert_no_oracle_leak(item, f"{owner}[{index}]")


def assert_planner_input_isolated(*, utterance: str, oracle_record: dict[str, Any]) -> None:
    """Fail closed when oracle text/labels leak into the planner utterance."""
    assert_no_oracle_leak({"utterance": utterance}, "planner-utterance")
    lowered = utterance.lower()
    candidates: set[str] = set()
    for value in oracle_record.values():
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                candidates.add(cleaned)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item:
                        candidates.add(cleaned_item)
    for label in candidates:
        if len(label) < 4:
            continue
        if label.lower() in lowered:
            raise RealBookRetrievalError(
                f"planner-utterance: oracle leak: {label!r} must not reach planner/generator"
            )


def assert_no_reranker(modules: dict[str, Any] | None = None) -> None:
    """Fail closed when a BGE/cross-encoder path is loaded."""
    loaded = sys.modules if modules is None else modules
    for name in loaded:
        lowered = str(name).lower()
        for fragment in FORBIDDEN_RERANKER_FRAGMENTS:
            if fragment in lowered:
                raise RealBookRetrievalError(
                    f"reranker path is forbidden on real-book qualification: {name!r}"
                )


def assert_source_exact(*, text: str, text_sha256: str, char_start: int, char_end: int) -> None:
    """Fail closed unless exact text, checksum and char offsets agree."""
    if sha256_text(text) != text_sha256:
        raise RealBookRetrievalError("source chunk checksum mismatch")
    if char_start < 0 or char_end <= char_start:
        raise RealBookRetrievalError("source char offsets are invalid")
    if char_end - char_start != len(text):
        raise RealBookRetrievalError("source char span does not match exact text")


def require_production_backend(*, backend: str) -> None:
    """Fail closed unless the index uses the production E5 backend."""
    if backend != E5_BACKEND_NAME:
        raise RealBookRetrievalError(
            f"production E5 backend required, got {backend!r}; "
            "missing E5 yields INCOMPLETE, never PASS"
        )


def classify_prerequisites(
    *,
    e5_available: bool,
    planner_available: bool,
    provider_available: bool,
) -> str:
    """Return INCOMPLETE when prerequisites are missing, else an empty string."""
    if not e5_available or not planner_available or not provider_available:
        return "INCOMPLETE"
    return ""


def decide_status(
    *,
    stale: bool,
    incomplete: bool,
    failures: int,
) -> str:
    """Decide the deterministic PASS/FAIL/INCOMPLETE/STALE status."""
    if stale:
        return "STALE"
    if incomplete:
        return "INCOMPLETE"
    if failures < 0:
        raise RealBookRetrievalError("failure count must be >= 0")
    return "FAIL" if failures > 0 else "PASS"


def assert_public_summary_safe(payload: Any, *, depth: int = 0) -> None:
    """Reject raw text, secrets or large dumps in the public summary."""
    if depth > 12:
        raise RealBookRetrievalError("public summary is nested too deeply")
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise RealBookRetrievalError(
                    f"public summary must not carry {key!r} (metrics only)"
                )
            if isinstance(value, str) and len(value) > 2000:
                raise RealBookRetrievalError(f"public summary field {key!r} is a large source dump")
            assert_public_summary_safe(value, depth=depth + 1)
    elif isinstance(payload, list):
        for item in payload:
            assert_public_summary_safe(item, depth=depth + 1)


@dataclass(frozen=True)
class TurnDiagnostics:
    """Per-turn trusted diagnostics (protected side may hold locators)."""

    case_id: str
    planner_queries: tuple[str, ...]
    bm25_top: tuple[str, ...]
    e5_top: tuple[str, ...]
    rrf_survivors: tuple[str, ...]
    dedup_survivors: tuple[str, ...]
    diversity_survivors: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_tokens: int
    source_chars: int
    latency_ms: float
    branch_union_hit: bool
    rrf_hit: bool
    dedup_hit: bool
    budget_hit: bool
    budget_loss: bool
    oracle_hit: bool | None

    def public_row(self) -> dict[str, Any]:
        """Return the metrics-only public projection for this turn."""
        return {
            "case_id": self.case_id,
            "planner_query_count": len(self.planner_queries),
            "bm25_result_count": len(self.bm25_top),
            "e5_result_count": len(self.e5_top),
            "rrf_survivor_count": len(self.rrf_survivors),
            "dedup_survivor_count": len(self.dedup_survivors),
            "diversity_survivor_count": len(self.diversity_survivors),
            "evidence_count": len(self.evidence_ids),
            "source_tokens": self.source_tokens,
            "source_chars": self.source_chars,
            "latency_ms": round(self.latency_ms, 3),
            "branch_union_hit": self.branch_union_hit,
            "rrf_hit": self.rrf_hit,
            "dedup_hit": self.dedup_hit,
            "budget_hit": self.budget_hit,
            "budget_loss": self.budget_loss,
            "oracle_hit": self.oracle_hit,
        }

    def protected_record(self) -> dict[str, Any]:
        """Return the full protected record (encrypted before upload)."""
        return {
            "case_id": self.case_id,
            "planner_queries": list(self.planner_queries),
            "bm25_top_ids": list(self.bm25_top),
            "e5_top_ids": list(self.e5_top),
            "rrf_survivor_ids": list(self.rrf_survivors),
            "dedup_survivor_ids": list(self.dedup_survivors),
            "diversity_survivor_ids": list(self.diversity_survivors),
            "evidence_ids": list(self.evidence_ids),
            "source_tokens": self.source_tokens,
            "source_chars": self.source_chars,
            "latency_ms": self.latency_ms,
            "branch_union_hit": self.branch_union_hit,
            "rrf_hit": self.rrf_hit,
            "dedup_hit": self.dedup_hit,
            "budget_hit": self.budget_hit,
            "budget_loss": self.budget_loss,
            "oracle_hit": self.oracle_hit,
        }


def summarize_public(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    turns: list[TurnDiagnostics],
    status: str,
    run_id: str,
) -> dict[str, Any]:
    """Build the compact non-sensitive public summary (metrics only)."""
    if status not in VALID_STATUSES:
        raise RealBookRetrievalError(f"invalid status {status!r}")
    latencies = sorted(item.latency_ms for item in turns)

    def _percentile(pct: float) -> float:
        if not latencies:
            return 0.0
        index = min(len(latencies) - 1, int(pct * len(latencies)))
        return round(latencies[index], 3)

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_SUMMARY_VERSION,
        "issue": RESULT_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "embedding_model": f"{EMBEDDING_MODEL_ID}@{EMBEDDING_REVISION[:12]}",
        "planner_agent": PLANNER_AGENT,
        "planner_profile": PLANNER_PROFILE,
        "retrieval_backend": "bm25+e5-faiss/rrf-only",
        "result": status,
        "run_id": run_id,
        "turn_count": len(turns),
        "oracle_hit_count": sum(1 for item in turns if item.oracle_hit is True),
        "oracle_miss_count": sum(1 for item in turns if item.oracle_hit is False),
        "budget_loss_count": sum(1 for item in turns if item.budget_loss),
        "mean_source_tokens": (
            round(sum(item.source_tokens for item in turns) / len(turns), 3) if turns else 0.0
        ),
        "mean_source_chars": (
            round(sum(item.source_chars for item in turns) / len(turns), 3) if turns else 0.0
        ),
        "latency_p50_ms": _percentile(0.5),
        "latency_p95_ms": _percentile(0.95),
        "turns": [item.public_row() for item in turns],
    }
    assert_public_summary_safe(payload)
    return payload


def build_protected_payload(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    turns: list[TurnDiagnostics],
    status: str,
    run_id: str,
) -> dict[str, Any]:
    """Build the protected per-case payload (compressed + encrypted)."""
    return {
        "schema_version": PROTECTED_ARTIFACT_VERSION,
        "issue": RESULT_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "result": status,
        "run_id": run_id,
        "turns": [item.protected_record() for item in turns],
    }


def compress_and_encrypt(payload: bytes, *, recipient: str) -> bytes:
    """Compress with zstd then age-encrypt to ``recipient``."""
    if not payload:
        raise RealBookRetrievalError("refusing to encrypt an empty payload")
    if not recipient.strip():
        raise RealBookRetrievalError("age recipient must not be empty")
    age_v1.parse_recipient(recipient)
    compressed = zstd.ZstdCompressor(level=3, threads=1).compress(payload)
    return age_v1.encrypt_bytes(compressed, [recipient])


def decrypt_and_decompress(bundle: bytes, *, identity: str) -> bytes:
    """Decrypt a protected bundle and decompress the zstd payload."""
    if not bundle:
        raise RealBookRetrievalError("refusing to decrypt an empty bundle")
    if not identity.strip():
        raise RealBookRetrievalError("age identity must not be empty")
    compressed = age_v1.decrypt_bytes(bundle, [identity])
    try:
        return zstd.ZstdDecompressor().decompress(compressed, max_output_size=64 * 1024 * 1024)
    except zstd.ZstdError as exc:
        raise RealBookRetrievalError(f"bundle decompression failed: {exc}") from exc


def build_result_marker(
    *,
    sha: str,
    corpus: str,
    benchmark: str,
    retrieval: str,
    result: str,
    run: str,
) -> str:
    """Build the canonical idempotent #130 result marker."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise RealBookRetrievalError("marker sha must be a 40-hex SHA")
    for name, value in (
        ("corpus", corpus),
        ("benchmark", benchmark),
        ("retrieval", retrieval),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value or ""):
            raise RealBookRetrievalError(f"marker {name} must be a 64-hex SHA")
    if result not in VALID_STATUSES:
        raise RealBookRetrievalError("marker result must be PASS|FAIL|INCOMPLETE|STALE")
    if not run.strip() or any(item.isspace() for item in run):
        raise RealBookRetrievalError("marker run id must be a non-empty token")
    return (
        "<!-- aa-real-book-retrieval-result "
        f"issue=130 sha={sha} corpus={corpus} benchmark={benchmark} "
        f"retrieval={retrieval} result={result} run={run} -->"
    )


def parse_result_marker(text: str) -> dict[str, str]:
    """Parse and validate a canonical #130 result marker."""
    match = _MARKER_RE.search(text)
    if match is None:
        raise RealBookRetrievalError("no canonical #130 result marker found")
    return {
        "sha": match.group("sha"),
        "corpus": match.group("corpus"),
        "benchmark": match.group("benchmark"),
        "retrieval": match.group("retrieval"),
        "result": match.group("result"),
        "run": match.group("run"),
    }


__all__ = [
    "BENCHMARK_INPUT_REL",
    "BENCHMARK_ORACLE_REL",
    "E5_BACKEND_NAME",
    "EMBEDDING_LOCK_REL",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_REVISION",
    "EXIT_BY_STATUS",
    "FORBIDDEN_RERANKER_FRAGMENTS",
    "PLANNER_AGENT",
    "PLANNER_PROFILE",
    "PLANNER_SCHEMA_NAME",
    "PROTECTED_ARTIFACT_VERSION",
    "PUBLIC_SUMMARY_VERSION",
    "RESULT_ISSUE",
    "RETRIEVAL_CONFIG_VERSION",
    "RU_MANIFEST_REL",
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "WORKFLOW_NAME",
    "RealBookRetrievalError",
    "TurnDiagnostics",
    "assert_exact_sha_binding",
    "assert_no_oracle_leak",
    "assert_no_reranker",
    "assert_planner_input_isolated",
    "assert_public_summary_safe",
    "assert_source_exact",
    "benchmark_checksums",
    "build_protected_payload",
    "build_result_marker",
    "classify_prerequisites",
    "compress_and_encrypt",
    "corpus_checksum",
    "decide_status",
    "decrypt_and_decompress",
    "find_repo_root",
    "parse_result_marker",
    "production_retrieval_checksum",
    "require_production_backend",
    "require_secret",
    "retrieval_config_checksum",
    "retrieval_config_dict",
    "secret_present",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "summarize_public",
    "validate_exact_sha",
]
