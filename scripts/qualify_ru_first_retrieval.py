#!/usr/bin/env python3
"""Qualify RU-first retrieval and decide the EN secondary branch (issue #47).

Runs the hermetic A/B/C benchmark over invented fixture text (no book
text committed), then writes the versioned benchmark/decision artifact
bound to the RU source checksum, structure version, index config,
planner schema and gold-set version:

- ``qualification/ru_first_retrieval.v1.decision.json``.

Logs carry only ids, sections, digests and counts, never book text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.ru_first import (  # noqa: E402
    DECISION_REL,
    find_repo_root,
    run_benchmark,
    validate_decision_payload,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run the benchmark and write the decision artifact."""
    parser = argparse.ArgumentParser(description="Qualify RU-first retrieval (issue #47).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Decision artifact path (default: qualification/ru_first_retrieval.v1.decision.json).",
    )
    args = parser.parse_args(argv)
    try:
        root = find_repo_root()
    except ValueError as exc:
        print(f"ru-first qualification failed: {exc}", file=sys.stderr)
        return 1
    out_path = args.out if args.out is not None else (root / DECISION_REL)
    try:
        payload = run_benchmark(repo_root=root)
        validate_decision_payload(payload, repo_root=root)
    except ValueError as exc:
        print(f"ru-first qualification failed: {exc}", file=sys.stderr)
        return 1
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = payload["quality_gate"]
    production = payload["production"]
    print(
        json.dumps(
            {
                "gold_fixtures": payload["gold_fixtures"],
                "recall_at_5_a": payload["configs"]["a_ru_first"]["recall_at_5"],
                "recall_at_5_b": payload["configs"]["b_ru_first_plus_en_secondary"]["recall_at_5"],
                "recall_at_5_c": payload["configs"]["c_legacy_ru_to_en_only"]["recall_at_5"],
                "incremental_recall_b_minus_a": payload["en_secondary"]["incremental_recall_at_5"],
                "gate_passed": gate["passed"],
                "production": production["config_id"],
                "out": str(out_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
