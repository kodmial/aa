#!/usr/bin/env python3
"""Pin the Gate C evaluator baseline (kodmial/aa#341, trusted workflow only).

Computes the evaluator source SHA plus the frozen v2 rubric digest for the
checked-out tree and writes the protected evaluation baseline
(``qualification/gate_c_evaluator_baseline.json`` by default). The trusted
post-merge workflow runs this on main; feature PRs must not mutate the
pinned baseline during qualification (a mismatch makes the unseen held-out
proof BLOCKED, never passed).

Only digests/versions travel outward. Exit 0 on success, 2 when the rubric
or evaluator source is unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pin the Gate C held-out evaluator baseline (issue #341)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "qualification" / "gate_c_evaluator_baseline.json",
        help="Baseline output path",
    )
    parser.add_argument("--run-id", default="local")
    args = parser.parse_args(argv)

    from aa.qualification.gate_c_heldout import (
        GateCHeldoutBlockedError,
        write_baseline_file,
    )

    try:
        payload = write_baseline_file(repo_root=ROOT, path=args.out, run_id=args.run_id)
    except (OSError, GateCHeldoutBlockedError) as exc:
        print(f"gate-c baseline pinning BLOCKED: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "version": payload["version"],
                "evaluator_source_sha": str(payload["evaluator_source_sha"])[:16],
                "rubric_digest": str(payload["rubric_digest"])[:16],
                "run_id": payload["run_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
