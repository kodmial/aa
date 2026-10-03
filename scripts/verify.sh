#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python3 -m aa --check
python3 -m pytest -q
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src tests
