#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec python3 scripts/run_gate_a_shell.py \
  --profile canonical \
  --out "${TMPDIR:-/tmp}/aa-gate-a-shell-results-canonical.json"
