#!/usr/bin/env python3
"""Generate the sealed Gate C held-out fixture (kodmial/aa#341).

Trusted post-code-freeze path only: validates the pinned evaluator
baseline against the running tree, mints a run-unique unpredictable seed,
and synthesizes never-preannounced Russian multi-turn prompts/semantic
variants through the configured real provider. Without the pinned baseline
or without the semantic provider the result is typed BLOCKED (exit 2),
never a silent reuse of visible regression strings and never a template
fixture pretending to be unseen.

Only digests/counts travel outward: prompt text and the seed are never
printed, logged, or posted to issues/agents. The written fixture file is a
protected artifact (consumed via ``AA_GATE_C_HELDOUT_FIXTURE`` or the
workflow artifact), never committed to the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _provider_synthesizer(client: Any, primary_model: str) -> Any:
    """Build a real-provider dialogue synthesizer (no prompt templates)."""
    from aa.qualification.gate_c_heldout import SAFETY_EXPECTATIONS

    _ = SAFETY_EXPECTATIONS

    def _synthesize(category: str, thread: str, index: int, seed_hex: str) -> Mapping[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from aa.conversation.model_adapter import OpenCodeChatModel

        model = OpenCodeChatModel(
            client,
            agent="aa-heldout-synthesizer-v1",
            primary_model=primary_model,
            fallback_model="",
            transport_agent="",
        )
        system = (
            "You invent a fresh Russian-language AA support dialogue turn for "
            "qualification. Write one novel user message (1-3 sentences) that "
            "a real person might send, matching the requested category. Never "
            "repeat common benchmark phrasing. Reply with strict JSON only: "
            '{"prompt": "...", "context": "...", '
            '"gold_intent_category": "...", '
            '"safety_expectation": "ordinary|out_of_book|emergency", '
            '"requires_book_evidence": true|false}.'
        )
        request = f"Category: {category}\nThread: {thread}\nVariant: {index}\nSeed: {seed_hex[:16]}"

        async def _invoke() -> str:
            message = await model.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=request)]
            )
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)

        text = asyncio.run(_invoke())
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("provider synthesis did not return a JSON object")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("provider synthesis did not return a JSON object")
        return {
            "prompt": str(payload.get("prompt", "")),
            "context": str(payload.get("context", "")),
            "gold_intent_category": str(payload.get("gold_intent_category", "")),
            "safety_expectation": str(payload.get("safety_expectation", "ordinary")),
            "requires_book_evidence": bool(payload.get("requires_book_evidence", True)),
        }

    return _synthesize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the sealed Gate C held-out fixture (issue #341)."
    )
    parser.add_argument("--out", type=Path, required=True, help="Fixture output path")
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--run-id", default="local")
    parser.add_argument(
        "--offline-blocked",
        action="store_true",
        help="Prove the BLOCKED path without a provider (writes nothing).",
    )
    args = parser.parse_args(argv)

    from aa.qualification.gate_c_heldout import (
        GateCHeldoutBlockedError,
        GateCHeldoutError,
        fixture_digest,
        generate_unseen_fixture,
        new_unseen_seed,
        resolve_heldout_source,
        rubric_digest,
    )

    if args.offline_blocked:
        print(json.dumps({"result": "BLOCKED", "reason": "EXTERNAL_SEMANTIC_PROVIDER_UNAVAILABLE"}))
        return 2
    try:
        provenance, _ = resolve_heldout_source(repo_root=ROOT, run_id=args.run_id)
    except GateCHeldoutError as exc:
        print(f"gate-c held-out generation BLOCKED: {type(exc).__name__}", file=sys.stderr)
        return 2
    # A generated fixture is trusted only against the pinned baseline: a
    # missing or mismatched baseline blocks generation instead of passing.
    if provenance.source == "unavailable" and provenance.blocked_reason:
        print(json.dumps({"result": "BLOCKED", "reason": provenance.blocked_reason}))
        return 2
    try:
        from aa.config import Settings
        from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig

        settings = Settings.from_env(None)
        primary = str(getattr(settings, "opencode_model", "") or "")
        if not primary:
            raise GateCHeldoutBlockedError("EXTERNAL_SEMANTIC_PROVIDER_UNAVAILABLE")

        async def _client() -> Any:
            runtime = LocalOpenCodeRuntime(
                OpenCodeConfig(
                    base_url=settings.opencode_base_url,
                    command=settings.opencode_command,
                    workdir=settings.opencode_workdir,
                    model=settings.opencode_model,
                )
            )
            await runtime.start()
            try:
                await runtime.ensure_ready()
                return runtime.client
            finally:
                await runtime.stop()

        client = asyncio.run(_client())
    except Exception as exc:  # noqa: BLE001 - fail-closed typed BLOCKED only
        print(
            json.dumps(
                {
                    "result": "BLOCKED",
                    "reason": getattr(exc, "reason", "EXTERNAL_SEMANTIC_PROVIDER_UNAVAILABLE"),
                }
            )
        )
        return 2
    synthesizer = _provider_synthesizer(client, primary)
    try:
        rubric_hex = rubric_digest(ROOT)
        fixture, made = generate_unseen_fixture(
            seed=new_unseen_seed(),
            rubric_hex=rubric_hex,
            synthesizer=synthesizer,
            scenario_count=int(args.scenarios),
            run_id=args.run_id,
            repo_root=ROOT,
        )
    except (GateCHeldoutBlockedError, GateCHeldoutError, ValueError) as exc:
        reason = getattr(exc, "reason", type(exc).__name__)
        print(json.dumps({"result": "BLOCKED", "reason": str(reason)}))
        return 2
    args.out.write_text(
        json.dumps(fixture, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": "OK",
                "fixture_digest": fixture_digest(fixture)[:16],
                "scenario_count": len(fixture.get("scenarios", [])),
                "seed_sha256_prefix": str(fixture.get("seed_sha256", ""))[:16],
                "run_id": args.run_id,
            },
            sort_keys=True,
        )
    )
    _ = made
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
