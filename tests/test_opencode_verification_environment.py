"""Protect AA/Continuum issue verification from missing CI dependencies."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_opencode_caller_and_ci_share_canonical_bootstrap() -> None:
    caller = (ROOT / ".github/workflows/continuum-opencode.yml").read_text("utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    script = ROOT / "scripts/prepare_opencode_verification_env.sh"
    content = script.read_text("utf-8")
    command = "bash scripts/prepare_opencode_verification_env.sh"
    assert f'prepare_command: "{command}"' in caller
    assert f"run: {command}" in ci
    assert 'validation_command: "bash scripts/verify.sh"' in caller
    assert "run: bash scripts/verify.sh" in ci
    assert 'python-version: "3.12"' in ci
    assert "python3.12" in content
    assert '-m pip install -e ".[dev]"' in content
    assert '>> "$GITHUB_PATH"' in content
    assert "TELEGRAM_BOT_TOKEN" not in content
    subprocess.run(["bash", "-n", str(script)], check=True)
