#!/usr/bin/env bash
set -euo pipefail

# CI and Continuum issue runs use one immutable bootstrap. The environment
# must exist before pre-change baseline, including isolated checkpoint replays.
cd "$(git rev-parse --show-toplevel)"

if [[ -z "${RUNNER_TEMP:-}" || -z "${GITHUB_PATH:-}" ]]; then
  echo "::error::AA verification preparation requires GitHub Actions runner paths." >&2
  exit 2
fi

# The canonical CI uses Python 3.12. Never fall back to a different system Python.
python312=""
for candidate in /opt/hostedtoolcache/Python/3.12.*/x64/bin/python3.12; do
  if [[ -x "$candidate" ]]; then
    python312="$candidate"
    break
  fi
done
if [[ -z "$python312" ]]; then
  python312="$(command -v python3.12 || true)"
fi
if [[ -z "$python312" || ! -x "$python312" ]]; then
  echo "::error::Python 3.12 unavailable; refusing to verify with another interpreter." >&2
  exit 1
fi
if [[ "$("$python312" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "::error::Verification interpreter is not Python 3.12." >&2
  exit 1
fi

venv="${RUNNER_TEMP}/aa-verification-py312"
"$python312" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -e ".[dev]"

# Abort before coding if the canonical static-check tools are missing.
"$venv/bin/python" - <<'PY'
import pytest
import ruff
import mypy
import aa
PY

# Later baseline and final steps inherit this venv, including isolated worktrees.
printf '%s\n' "$venv/bin" >> "$GITHUB_PATH"
echo "AA canonical verification environment prepared: Python 3.12 with dev tools."
