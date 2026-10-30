#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# The OpenCode checkout may not have the project installed as a package.
# Keep both the inventory import and subprocess checks on the same src/ tree.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from aa.qualification.gate_a_shell import CANONICAL_SHELL_CHECKS

expected = [
    "python3 -m aa --check",
    "python3 -m pytest -q",
    "python3 -m ruff check .",
    "python3 -m ruff format --check .",
    "python3 -m mypy src tests",
]
observed = ["python3 " + " ".join(check.command[1:]) for check in CANONICAL_SHELL_CHECKS]
if observed != expected:
    raise SystemExit("canonical static contract drifted from the Gate A shell inventory")
PY

exec python3 scripts/run_gate_a_shell.py \
  --profile canonical \
  --out "${TMPDIR:-/tmp}/aa-gate-a-shell-results-canonical.json"
