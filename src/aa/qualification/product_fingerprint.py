"""Deterministic product-qualification input fingerprint (issues #106, #121).

The Product Contract qualification (#7 for capability #6) measures the
user-facing AA runtime, not repository plumbing. A change to the
deterministic product input set invalidates the current product PASS and
must rerun qualification; unrelated automation/docs changes must not stale
it merely because the repository HEAD moved.

The product input set covers runtime code, prompt, retrieval/grounding,
corpus/index bindings, the Telegram production adapter and relevant
config. Scheduler callers, docs and unrelated automation are explicitly
excluded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

# File prefixes whose content affects the qualified product behavior.
PRODUCT_INPUT_PREFIXES: tuple[str, ...] = (
    "src/aa/conversation/",
    "src/aa/retrieval/",
    "src/aa/grounding/",
    "src/aa/corpus/",
    "src/aa/safety/",
    "src/aa/sessions/",
    "src/aa/telegram/",
    "src/aa/qualification/",
)

# Exact product files outside the prefixed directories.
PRODUCT_INPUT_FILES: tuple[str, ...] = (
    "src/aa/app.py",
    "src/aa/config.py",
    "src/aa/__main__.py",
    "prompts/aa-agent-system-v2.md",
    "prompts/aa-planner-system-v2.md",
    "prompts/aa-summarizer-system-v2.md",
    "prompts/aa-verifier-system-v2.md",
    "opencode.json",
    "corpus/canonical.ru.manifest.json",
    "corpus/canonical.manifest.json",
    "corpus/structure.json",
    "corpus/source.lock.json",
    "corpus/source.ru.lock.json",
    "corpus/embedding.lock.json",
    "qualification/aa-retrieval.json",
    "qualification/ru_first_retrieval.v1.decision.json",
    "qualification/ru_answer_quality_rubric.v2.json",
    "qualification/ru_product_contract.v1_2.input.jsonl",
    "qualification/ru_product_contract.v1_2.oracle.jsonl",
    "qualification/ru_product_contract.v1_2.sources.json",
    "qualification/production_boundary.v1.json",
    "scripts/verify_product_contract_qualification.py",
    ".opencode/tools/book_search.ts",
    ".opencode/tools/book_read.ts",
    ".opencode/tools/book_expand.ts",
    ".opencode/tools/book_section.ts",
)

# Repository automation/docs paths that never affect product qualification.
# Kept as documentation for reviewers; matching is by exclusion (anything
# not in the product set is unrelated by construction).
UNRELATED_PREFIXES: tuple[str, ...] = (
    ".github/workflows/continuum-",
    "docs/",
    ".opencode/.git",
)

FINGERPRINT_VERSION = "product-fingerprint/1"

# Generated/cache artifacts that never affect qualified product behavior.
# A local test run or install materializing ``__pycache__``/``*.pyc``
# must not rotate the product fingerprint and spuriously invalidate PASS.
_GENERATED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".pyd")
_GENERATED_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


def _is_generated_artifact(relpath: str) -> bool:
    """Return whether ``relpath`` is a generated/cache artifact."""
    normalized = relpath.replace("\\", "/").lstrip("./")
    parts = normalized.split("/")
    if any(part in _GENERATED_DIRS for part in parts):
        return True
    basename = parts[-1] if parts else ""
    if basename == ".DS_Store":
        return True
    return basename.endswith(_GENERATED_SUFFIXES)


def is_product_path(relpath: str) -> bool:
    """Return whether a repo-relative path belongs to the product input set."""
    normalized = relpath.replace("\\", "/").lstrip("./")
    if _is_generated_artifact(normalized):
        return False
    for prefix in PRODUCT_INPUT_PREFIXES:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            return True
    return normalized in PRODUCT_INPUT_FILES


def fingerprint_of_files(files: Mapping[str, bytes]) -> str:
    """Compute the deterministic fingerprint of product file contents.

    Only product input paths contribute; unrelated paths are ignored so a
    scheduler-only change cannot stale product qualification. The digest
    covers sorted ``path + NUL + bytes`` records under a versioned domain
    separator, so renames and content changes both invalidate.
    """
    records: list[tuple[str, bytes]] = []
    for relpath, content in files.items():
        normalized = relpath.replace("\\", "/").lstrip("./")
        if is_product_path(normalized):
            records.append((normalized, bytes(content)))
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256(FINGERPRINT_VERSION.encode("utf-8"))
    for relpath, content in records:
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\x00")
    return digest.hexdigest()


def collect_product_files(repo_root: Path) -> dict[str, bytes]:
    """Read current product input file bytes from the working tree."""
    collected: dict[str, bytes] = {}
    for prefix in PRODUCT_INPUT_PREFIXES:
        base = repo_root / prefix
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                continue
            if _is_generated_artifact(rel) or not is_product_path(rel):
                continue
            try:
                collected[rel] = path.read_bytes()
            except OSError:
                continue
    for rel in PRODUCT_INPUT_FILES:
        path = repo_root / rel
        if path.is_file():
            try:
                collected[rel] = path.read_bytes()
            except OSError:
                continue
    return collected


def compute_product_fingerprint(repo_root: Path) -> str:
    """Compute the current product fingerprint for the working tree."""
    return fingerprint_of_files(collect_product_files(repo_root))


def fingerprint_is_current(stored: str, repo_root: Path) -> bool:
    """Return whether a stored PASS fingerprint matches the working tree."""
    if not stored.strip():
        return False
    return stored.strip() == compute_product_fingerprint(repo_root)


__all__ = [
    "FINGERPRINT_VERSION",
    "PRODUCT_INPUT_FILES",
    "PRODUCT_INPUT_PREFIXES",
    "UNRELATED_PREFIXES",
    "collect_product_files",
    "compute_product_fingerprint",
    "fingerprint_is_current",
    "fingerprint_of_files",
    "is_product_path",
]
