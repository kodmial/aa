"""Contracts for the prebuilt AA runtime image (issue #298)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dockerfile_is_reproducible_and_secret_free() -> None:
    text = _read(ROOT / "docker" / "Dockerfile.aa-runtime")
    assert "python:3.12" in text
    assert "ffmpeg" in text
    assert "libopus" in text
    for pin in (
        "sherpa-onnx==1.13.8",
        "torch==2.14.1",
        "onnxruntime==1.30.0",
        "faiss-cpu==1.13.2",
        "transformers==4.57.3",
    ):
        assert pin in text
    assert "docker/opencode.version" in text
    assert "1.18.34" not in text  # version comes from the pin file, not drift
    assert "1.18.34" in _read(ROOT / "docker" / "opencode.version")
    for forbidden in ("TELEGRAM_BOT_TOKEN", "AGE-SECRET-KEY", "corpus/generated"):
        assert forbidden not in text
    assert "COPY docker/opencode.version" in text


def test_digest_file_and_runtime_pin_are_immutable_and_consistent() -> None:
    lines = _read(ROOT / "docker" / "aa-runtime.digest").strip().splitlines()
    digest = lines[-1].strip()
    assert re.fullmatch(r"ghcr\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}", digest)
    workflow = _read(WORKFLOWS / "aa-runtime.yml")
    assert "ghcr.io/kodmial/aa-runtime@sha256:" in workflow
    assert digest in workflow
    assert "container:" in workflow
    assert "credentials:" in workflow
    assert "packages: read" in workflow


def test_runtime_preserves_single_poller_and_fail_closed_preflight() -> None:
    workflow = _read(WORKFLOWS / "aa-runtime.yml")
    assert "group: aa-bot-runtime" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "Reject duplicate runtime" in workflow
    assert "verify_runtime_image.py" in workflow
    assert "git rev-parse HEAD" in workflow
    assert "opencode --version" in workflow
    assert "encoder=libopus" in workflow


def test_runtime_hot_path_skips_redundant_installs() -> None:
    workflow = _read(WORKFLOWS / "aa-runtime.yml")
    assert "pip install -e . --no-deps" in workflow
    assert "already present" in workflow
    assert "docker/opencode.version" in workflow
    # Cold-miss path is bounded: apt-get stays guarded, OpenCode retries capped.
    assert "if ! command -v ffmpeg" in workflow
    assert "for attempt in 1 2 3" in workflow


def test_image_workflow_publishes_only_from_trusted_events() -> None:
    text = _read(WORKFLOWS / "aa-runtime-image.yml")
    assert "push:" in text
    assert "branches: [main]" in text or "branches: [ main ]" in text or "main" in text
    assert "packages: write" in text
    assert "GITHUB_TOKEN" in text
    assert "type=registry" in text
    assert "provenance: true" in text
    assert "pull_request" in text
    assert "owner-only" in text or "repository_owner" in text
    assert "canary" in text.lower()
    # The only mention of secret names must be the build-context scan
    # pattern; the workflow itself never consumes secrets.
    assert "secrets.TELEGRAM_BOT_TOKEN" not in text
    assert "secrets.AA_BOOK_AGE_IDENTITY" not in text
    assert "TELEGRAM_BOT_TOKEN: ${{" not in text


def test_image_scripts_and_docs_exist() -> None:
    assert (ROOT / "scripts" / "verify_runtime_image.py").is_file()
    assert (ROOT / "scripts" / "promote_runtime_image.py").is_file()
    assert (ROOT / "docs" / "aa-runtime-image.md").is_file()
    helper = _read(ROOT / "scripts" / "verify_runtime_image.py")
    assert "--pins-only" in helper
    assert "--canary" in helper
    assert "libopus" in helper
    promoter = _read(ROOT / "scripts" / "promote_runtime_image.py")
    assert "--digest" in promoter
    assert "--apply" in promoter


def test_pins_helper_accepts_current_tree() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/verify_runtime_image.py", "--pins-only"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
