#!/usr/bin/env python3
"""Static exact-main readiness gate for Product Contract qualification #7.

This gate is validation infrastructure only. It deliberately distinguishes
"not activated yet" from "invalid": #121 may merge while the production
Telegram cutover (#118) is still in progress, but authoritative #7
qualification must never dispatch until the legacy production conversation
path is gone and the v2 LangGraph boundary is visibly wired into production.

The checks are privacy-safe and inspect only repository structure, versions
and digests. They never read user messages, prompts at runtime, decrypted
corpus text, evidence text, audio or transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA_VERSION = "aa-product-contract-qualification-readiness/1"

REQUIRED_V2_FILES: tuple[str, ...] = (
    "src/aa/conversation/graph.py",
    "src/aa/conversation/graph_state.py",
    "src/aa/conversation/memory.py",
    "src/aa/conversation/model_adapter.py",
    "src/aa/conversation/planner_node.py",
    "src/aa/conversation/planner_schema.py",
    "src/aa/conversation/retrieval_node.py",
    "src/aa/conversation/answer_node.py",
    "src/aa/conversation/turn_pipeline.py",
    "src/aa/conversation/verifier.py",
    "src/aa/conversation/verifier_schema.py",
    "prompts/aa-agent-system-v2.md",
    "prompts/aa-planner-system-v2.md",
    "prompts/aa-summarizer-system-v2.md",
    "prompts/aa-verifier-system-v2.md",
    "src/aa/control/runtime_control.py",
)

LEGACY_APPLICATION_MARKERS: tuple[str, ...] = (
    "from aa.conversation.meta import",
    "from aa.conversation.orchestrator import",
    "TurnRunner",
    "is_substantive",
    "FAIL_CLOSED_REPLY",
    "META_CAPABILITY_REPLY",
    "run_trivial_turn",
)

V2_BINDING_MARKERS: tuple[str, ...] = (
    "build_turn_graph(",
    "build_turn_graph,",
    "build_turn_graph as ",
)

TYPING_MARKER = "sendChatAction"


class ReadinessError(RuntimeError):
    """Repository is inconsistent and cannot be qualified."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_text(relpath: str) -> str:
    path = ROOT / relpath
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReadinessError(f"missing required file: {relpath}") from exc


def _require_v2_files() -> dict[str, str]:
    digests: dict[str, str] = {}
    for relpath in REQUIRED_V2_FILES:
        path = ROOT / relpath
        if not path.is_file():
            raise ReadinessError(f"missing v2 runtime asset: {relpath}")
        digests[relpath] = _sha256(path)[:16]
    return digests


def _production_python_sources() -> list[Path]:
    root = ROOT / "src" / "aa"
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and path.relative_to(ROOT).as_posix() != "src/aa/conversation/graph.py"
    )


def _has_v2_production_binding() -> bool:
    for path in _production_python_sources():
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in V2_BINDING_MARKERS):
            return True
    return False


def _has_typing_heartbeat_binding() -> bool:
    candidates = [
        ROOT / "src" / "aa" / "app.py",
        *(ROOT / "src" / "aa" / "telegram").glob("*.py"),
    ]
    return any(
        path.is_file() and TYPING_MARKER in path.read_text(encoding="utf-8")
        for path in candidates
    )


def _legacy_application_markers() -> list[str]:
    app = _read_text("src/aa/app.py")
    return [marker for marker in LEGACY_APPLICATION_MARKERS if marker in app]


def evaluate() -> dict[str, Any]:
    """Return one privacy-safe readiness payload for the checked-out tree."""
    v2_digests = _require_v2_files()

    try:
        from aa.qualification.product_contract_vnext import validate as validate_vnext
        from aa.qualification.product_fingerprint import compute_product_fingerprint
    except Exception as exc:
        raise ReadinessError(f"qualification modules are not importable: {exc}") from exc

    try:
        benchmark = validate_vnext(ROOT)
    except Exception as exc:
        raise ReadinessError(f"Product Contract benchmark assets are invalid: {exc}") from exc

    try:
        product_fingerprint = compute_product_fingerprint(ROOT)
    except Exception as exc:
        raise ReadinessError(f"product fingerprint is not computable: {exc}") from exc

    legacy = _legacy_application_markers()
    v2_bound = _has_v2_production_binding()
    typing_bound = _has_typing_heartbeat_binding()
    active = not legacy and v2_bound and typing_bound

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if active else "not-activated",
        "activation": {
            "legacy_application_markers": len(legacy),
            "v2_production_binding": v2_bound,
            "telegram_typing_binding": typing_bound,
        },
        "product_fingerprint": product_fingerprint,
        "benchmark": {
            "input_sha256": benchmark.input_sha256,
            "oracle_sha256": benchmark.oracle_sha256,
            "sources_sha256": benchmark.sources_sha256,
            "rubric_sha256": benchmark.rubric_sha256,
            "total_substantive": benchmark.total_substantive,
        },
        "v2_assets_digest": hashlib.sha256(
            json.dumps(v2_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-active",
        action="store_true",
        help="exit non-zero when the #118 production cutover is not active yet",
    )
    args = parser.parse_args()

    try:
        payload = evaluate()
    except ReadinessError as exc:
        print(f"product-contract qualification readiness failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, sort_keys=True))
    if args.require_active and payload["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
