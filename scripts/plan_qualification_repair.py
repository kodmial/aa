#!/usr/bin/env python3
"""Read privacy-safe typed gate failures on stdin; emit code-repair admission JSON.

Must not be used to calculate product PASS/FAIL. Exit nonzero on invalid input
rather than silently accepting empty/corrupt evidence as a product success.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aa.qualification.repair_policy import plan_repair  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict) or not isinstance(payload.get("failures"), list):
            raise ValueError("missing typed failure list")
        if not all(isinstance(value, dict) for value in payload["failures"]):
            raise ValueError("malformed failure entry")
        plan = plan_repair(
            payload["failures"],
            gate_c_breaker=payload.get("gate_c_breaker") is True,
            in_flight_repair=payload.get("in_flight_repair") is True,
        )
        print(json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True))
        return 0
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"repair-policy-invalid-input:{type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
