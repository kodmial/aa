#!/usr/bin/env python3
"""Qualify the final RU-first retrieval/tool layer (issue #19).

Runs the hermetic benchmark over invented fixture text (no book text
committed) and writes the versioned qualification artifact bound to the
RU source checksum, EN reference checksum, aligned structure version,
index configuration, planner schema and gold-set version:

- ``qualification/aa-retrieval.json``.

Logs carry only ids, sections, digests and counts, never book text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.aa_retrieval import (  # noqa: E402
    ARTIFACT_REL,
    AaRetrievalError,
    find_repo_root,
    run_qualification,
    validate_artifact_payload,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run the benchmark and write the artifact."""
    parser = argparse.ArgumentParser(description="Qualify RU-first retrieval (issue #19).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Artifact path (default: qualification/aa-retrieval.json).",
    )
    args = parser.parse_args(argv)
    try:
        root = find_repo_root()
    except AaRetrievalError as exc:
        print(f"aa-retrieval qualification failed: {exc}", file=sys.stderr)
        return 1
    out_path = args.out if args.out is not None else (root / ARTIFACT_REL)
    try:
        payload = run_qualification(repo_root=root)
        validate_artifact_payload(payload, repo_root=root)
    except AaRetrievalError as exc:
        print(f"aa-retrieval qualification failed: {exc}", file=sys.stderr)
        return 1
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    retrieval = payload["retrieval"]
    gate = payload["quality_gate"]
    print(
        json.dumps(
            {
                "gold_fixtures": payload["gold_fixtures"],
                "recall_at_5": retrieval["recall_at_5"],
                "coverage": retrieval["coverage"],
                "slang_pass_rate": gate["slang_pass_rate_measured"],
                "fidelity": payload["fidelity"]["rate"],
                "stale_rejection": payload["stale_index"]["rate"],
                "en_control_recall_at_5": payload["en_control"]["recall_at_5"],
                "gate_passed": gate["passed"],
                "production": payload["production"]["config_id"],
                "out": str(out_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
