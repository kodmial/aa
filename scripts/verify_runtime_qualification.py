#!/usr/bin/env python3
"""Runtime qualification gate for the AA Telegram worker (issues #19/#9).

Telegram polling must not start unless the qualified retrieval artifacts
match the live corpus/model versions:

- RU canonical manifest plus the encrypted RU snapshot and its metadata
  are current and mutually consistent (#28);
- the aligned structure version matches the live corpus bindings;
- ``qualification/aa-retrieval.json`` (#19 final RU-first retrieval/tool
  policy: recall/coverage/slang/fidelity/stale gates, RU-only production);
- ``qualification/ru_first_retrieval.v1.decision.json`` (#47 RU-first
  architecture decision with EN-secondary disabled);
- the authoritative prompt/agent config matches (named ``aa`` primary
  agent, pinned primary model, exactly the four book tools,
  deny-by-default);
- the four narrow book tools exist with deny-by-default permissions.

Stale or mismatched artifacts fail closed. Logs carry only ids, digests
and counts, never user text or corpus text.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_STRUCTURE_FORMAT = "aa-aligned-structure/1"
EXPECTED_STRUCTURE_BUILDER = 1
EXPECTED_RU_MANIFEST_FORMAT = "aa-canonical-manifest-ru/1"
EXPECTED_RU_METADATA_VERSION = 2
EXPECTED_ARCHIVE_FORMAT = "canonical-tar-zst/1"

REQUIRED_PROMPT_MARKERS = (
    "RUSSIAN QUOTATION AND MULTILINGUAL GROUNDING",
    "Russian conversations display quotations in Russian",
    "aa.grounding",
)
REQUIRED_AGENT_MODEL = "opencode/muse-spark-1.3-contributor-free"


def _fail(message: str) -> int:
    print(f"runtime qualification failed: {message}", file=sys.stderr)
    return 1


def _read_json(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {owner}: {path.relative_to(ROOT)}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{owner} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_ru_snapshot() -> dict[str, str]:
    """Validate the RU canonical manifest + encrypted snapshot consistency."""
    ru_manifest = _read_json(
        ROOT / "corpus" / "canonical.ru.manifest.json", owner="RU canonical manifest"
    )
    if ru_manifest.get("format") != EXPECTED_RU_MANIFEST_FORMAT:
        raise ValueError("RU manifest format is not aa-canonical-manifest-ru/1")
    ru_artifact = str(ru_manifest.get("artifact_sha256", ""))
    if not ru_artifact:
        raise ValueError("RU manifest carries no artifact_sha256")
    metadata = _read_json(
        ROOT / "corpus" / "source" / "encrypted" / "metadata.ru.json",
        owner="RU encrypted snapshot metadata",
    )
    if metadata.get("metadata_version") != EXPECTED_RU_METADATA_VERSION:
        raise ValueError("RU snapshot metadata_version is not 2")
    if metadata.get("archive_format") != EXPECTED_ARCHIVE_FORMAT:
        raise ValueError("RU snapshot archive_format is stale")
    if str(metadata.get("manifest_artifact_sha256", "")) != ru_artifact:
        raise ValueError("RU snapshot metadata does not match the RU manifest artifact")
    if str(metadata.get("canonical_sha256", "")) != ru_artifact:
        raise ValueError("RU snapshot canonical digest does not match the RU manifest")
    archive_rel = str(
        metadata.get("encrypted_file", "corpus/source/encrypted/canonical.ru.tar.zst.age")
    )
    archive = ROOT / archive_rel
    if not archive.is_file():
        raise ValueError(f"missing RU encrypted snapshot: {archive_rel}")
    live_digest = _sha256_file(archive)
    if live_digest != str(metadata.get("encrypted_sha256", "")):
        raise ValueError("RU encrypted snapshot digest does not match its metadata")
    return {"ru_artifact": ru_artifact[:16], "snapshot": live_digest[:16]}


def _check_structure_bindings(live_en_artifact: str) -> dict[str, Any]:
    """Validate the aligned structure version against live corpus bindings."""
    structure = _read_json(ROOT / "corpus" / "structure.json", owner="aligned structure")
    if structure.get("format") != EXPECTED_STRUCTURE_FORMAT:
        raise ValueError("aligned structure format is stale")
    if structure.get("builder_version") != EXPECTED_STRUCTURE_BUILDER:
        raise ValueError("aligned structure builder_version is stale")
    ru_branch = structure.get("ru")
    en_branch = structure.get("en")
    if not isinstance(ru_branch, dict) or not isinstance(en_branch, dict):
        raise ValueError("aligned structure is missing ru/en branches")
    # The EN branch pins the exact live EN artifact; the RU branch pins its
    # manifest format/edition (the RU text artifact rotates independently of
    # the aligned section layout, whose staleness is owned by #19 bindings).
    if str(en_branch.get("artifact_sha256", "")) != live_en_artifact:
        raise ValueError("aligned structure EN artifact is stale")
    if str(ru_branch.get("manifest_format", "")) != EXPECTED_RU_MANIFEST_FORMAT:
        raise ValueError("aligned structure RU manifest format is stale")
    return {
        "format": str(structure.get("format")),
        "builder": structure.get("builder_version"),
    }


def live_artifacts() -> tuple[str, str]:
    """Return the live ``(ru_artifact, en_artifact)`` manifest digests."""
    ru_manifest = _read_json(
        ROOT / "corpus" / "canonical.ru.manifest.json", owner="RU canonical manifest"
    )
    en_manifest = _read_json(
        ROOT / "corpus" / "canonical.manifest.json", owner="EN canonical manifest"
    )
    return (
        str(ru_manifest.get("artifact_sha256", "")),
        str(en_manifest.get("artifact_sha256", "")),
    )


def _check_prompt_config() -> dict[str, str]:
    prompt_path = ROOT / "prompts" / "aa-agent-system.md"
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError("missing authoritative prompt: prompts/aa-agent-system.md") from exc
    for marker in REQUIRED_PROMPT_MARKERS:
        if marker not in prompt:
            raise ValueError(f"authoritative prompt is missing {marker!r}")
    config = _read_json(ROOT / "opencode.json", owner="agent config")
    try:
        agent = config["agent"]["aa"]
        permission = agent["permission"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"agent permission contract is unreadable: {exc}") from exc
    if agent.get("mode") != "primary":
        raise ValueError("aa agent must stay the primary agent")
    if agent.get("model") != REQUIRED_AGENT_MODEL:
        raise ValueError("aa agent primary model is not the pinned runtime model")
    if "prompts/aa-agent-system.md" not in str(agent.get("prompt", "")):
        raise ValueError("aa agent prompt binding is stale")
    if permission.get("*") != "deny":
        raise ValueError("aa agent must stay deny-by-default")
    allowed = {name for name, value in permission.items() if value == "allow"}
    if allowed != {"book_search", "book_read", "book_expand", "book_section"}:
        raise ValueError("aa agent must allow exactly the four book tools")
    return {"model": str(agent.get("model")), "prompt": "prompts/aa-agent-system.md"}


def main() -> int:
    """Validate live qualification bindings; exit 0 only when current."""
    try:
        from aa.qualification.aa_retrieval import (
            ARTIFACT_REL as AA_RETRIEVAL_REL,
        )
        from aa.qualification.aa_retrieval import (
            validate_artifact_payload as validate_aa_retrieval,
        )
        from aa.qualification.ru_first import (
            DECISION_REL as RU_FIRST_REL,
        )
        from aa.qualification.ru_first import (
            validate_decision_payload as validate_ru_first,
        )
    except Exception as exc:
        return _fail(f"qualification modules are not importable: {exc}")

    try:
        snapshot = _check_ru_snapshot()
    except ValueError as exc:
        return _fail(f"RU canonical/encrypted snapshot is not current: {exc}")
    try:
        ru_full, en_full = live_artifacts()
        if not ru_full or not en_full:
            raise ValueError("live manifests carry no artifact digests")
        structure = _check_structure_bindings(en_full)
    except ValueError as exc:
        return _fail(f"aligned structure/index bindings are stale: {exc}")
    try:
        prompt_info = _check_prompt_config()
    except ValueError as exc:
        return _fail(f"authoritative prompt/agent config mismatch: {exc}")

    try:
        aa_payload = json.loads((ROOT / AA_RETRIEVAL_REL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"missing #19 artifact: {AA_RETRIEVAL_REL}")
    except json.JSONDecodeError as exc:
        return _fail(f"#19 artifact is not valid JSON: {exc}")
    try:
        validate_aa_retrieval(aa_payload, repo_root=ROOT)
    except ValueError as exc:
        return _fail(f"#19 artifact is stale or invalid: {exc}")
    if aa_payload.get("production", {}).get("config_id") != "ru-first-only":
        return _fail("#19 production configuration is not ru-first-only")

    try:
        ru_first_payload = json.loads((ROOT / RU_FIRST_REL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"missing #47 decision artifact: {RU_FIRST_REL}")
    except json.JSONDecodeError as exc:
        return _fail(f"#47 decision artifact is not valid JSON: {exc}")
    try:
        validate_ru_first(ru_first_payload, repo_root=ROOT)
    except ValueError as exc:
        return _fail(f"#47 decision artifact is stale or invalid: {exc}")

    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        if not (ROOT / ".opencode" / "tools" / f"{tool}.ts").is_file():
            return _fail(f"missing OpenCode book tool wrapper: {tool}")

    try:
        from aa.conversation.orchestrator import RUNTIME_VERSION
        from aa.retrieval.planner import SCHEMA_VERSION as planner_schema
    except Exception as exc:
        return _fail(f"conversation runtime is not importable: {exc}")
    try:
        from aa.qualification.product_fingerprint import compute_product_fingerprint
    except Exception as exc:
        return _fail(f"product fingerprint module is not importable: {exc}")
    try:
        product_fingerprint = compute_product_fingerprint(ROOT)
    except Exception as exc:
        return _fail(f"product fingerprint is not computable: {exc}")
    bindings = aa_payload.get("bindings", {})
    if not isinstance(bindings, dict) or bindings.get("ru_artifact_sha256") != ru_full:
        return _fail("#19 RU bindings do not match the live RU manifest")
    if "en_artifact_sha256" in bindings and bindings.get("en_artifact_sha256") != en_full:
        return _fail("#19 EN bindings do not match the live EN manifest")

    retrieval = aa_payload.get("retrieval", {})
    print(
        json.dumps(
            {
                "aa_retrieval": AA_RETRIEVAL_REL,
                "ru_first": RU_FIRST_REL,
                "recall_at_5": retrieval.get("recall_at_5"),
                "production": aa_payload.get("production", {}).get("config_id"),
                "ru_snapshot": snapshot["snapshot"],
                "structure": structure,
                "agent_model": prompt_info["model"],
                "planner_schema": planner_schema,
                "runtime": RUNTIME_VERSION,
                "product_fingerprint": product_fingerprint,
                "status": "qualified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
