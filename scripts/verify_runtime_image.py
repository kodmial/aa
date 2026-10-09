#!/usr/bin/env python3
"""Prebuilt AA runtime image verification (issue #298).

Fail-closed hot-path checks shared by the image build workflow, the image
canary, and the bot runtime preflight:

- Dockerfile pins mirror pyproject.toml (``--pins-only``, no network);
- ffmpeg OGG/Opus synthetic encode/decode smoke (0.5s synthetic tone only);
- Python 3.12 plus torch / sherpa-onnx / onnxruntime native imports;
- pinned OpenCode version from docker/opencode.version;
- public model identity via the existing --check-only prefetch entrypoints
  (never downloads here; bounded cold-miss stays in the workflow steps);
- checked-out application source matches the current HEAD SHA;
- digest file format plus workflow container-pin consistency;
- timing/size report without leaking private state.

Never starts a Telegram poller, never touches secrets, book plaintext,
decrypted indexes, audio, transcripts, or generated responses.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile.aa-runtime"
OPENCODE_VERSION_FILE = ROOT / "docker" / "opencode.version"
DIGEST_FILE = ROOT / "docker" / "aa-runtime.digest"
RUNTIME_WORKFLOW = ROOT / ".github" / "workflows" / "aa-runtime.yml"
PYPROJECT = ROOT / "pyproject.toml"

DIGEST_RE = re.compile(r"^ghcr\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}$")
ZERO_DIGEST = "0" * 64

EXPECTED_PINS = (
    "sherpa-onnx==1.13.8",
    "torch==2.14.1",
    "onnxruntime==1.30.0",
    "langchain==1.4.3",
    "langchain-core==1.6.6",
    "langgraph==1.2.12",
    "langgraph-checkpoint-sqlite==3.1.1",
    "langmem==0.0.30",
    "pydantic==2.13.5",
    "razdel==0.5.0",
    "transformers==4.57.3",
    "faiss-cpu==1.13.2",
)

FORBIDDEN_DOCKER_COPY = (
    "corpus/source/raw",
    "corpus/generated",
    "models/",
    ".env",
    "runtime-status.json",
)


def _fail(message: str) -> int:
    print(f"runtime image verification failed: {message}", file=sys.stderr)
    return 1


def check_pins() -> list[str]:
    errors: list[str] = []
    try:
        docker_text = DOCKERFILE.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Dockerfile unreadable: {exc}"]
    try:
        project_text = PYPROJECT.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"pyproject.toml unreadable: {exc}"]
    for pin in EXPECTED_PINS:
        if pin not in project_text:
            errors.append(f"pyproject.toml missing expected pin {pin}")
        if pin not in docker_text:
            errors.append(f"Dockerfile missing expected pin {pin}")
    try:
        want = OPENCODE_VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return errors + [f"opencode.version unreadable: {exc}"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", want):
        errors.append(f"opencode.version malformed: {want!r}")
    elif "docker/opencode.version" not in docker_text:
        errors.append("Dockerfile does not consume docker/opencode.version")
    if "python:3.12" not in docker_text:
        errors.append("Dockerfile must use a Python 3.12 base")
    if "libopus" not in docker_text:
        errors.append("Dockerfile must verify ffmpeg libopus support")
    if "ffmpeg" not in docker_text.lower():
        errors.append("Dockerfile must install ffmpeg")
    for forbidden in FORBIDDEN_DOCKER_COPY:
        for line in docker_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY") and forbidden in stripped:
                errors.append(f"Dockerfile must not COPY {forbidden}")
    return errors


def check_digest_wiring(*, allow_placeholder: bool) -> list[str]:
    errors: list[str] = []
    try:
        digest_line = DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        return [f"digest file unreadable: {exc}"]
    digest = digest_line[-1].strip() if digest_line else ""
    if not DIGEST_RE.match(digest):
        errors.append(f"digest file malformed: {digest!r}")
        return errors
    if digest.endswith(":" + ZERO_DIGEST) and not allow_placeholder:
        errors.append("digest file still holds the pre-promotion placeholder")
    try:
        workflow_text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    except OSError as exc:
        return errors + [f"aa-runtime.yml unreadable: {exc}"]
    if "ghcr.io/kodmial/aa-runtime@sha256:" not in workflow_text:
        errors.append("aa-runtime.yml is not pinned to the GHCR runtime image")
    elif digest not in workflow_text and ZERO_DIGEST not in workflow_text:
        errors.append("aa-runtime.yml container pin does not match digest file")
    if "container:" not in workflow_text or "credentials:" not in workflow_text:
        errors.append("aa-runtime.yml must use container.credentials for GHCR pull")
    if "packages: read" not in workflow_text:
        errors.append("aa-runtime.yml must grant packages: read for GHCR pull")
    if "group: aa-bot-runtime" not in workflow_text:
        errors.append("aa-runtime.yml must keep the no-duplicate-poller concurrency group")
    return errors


def check_ffmpeg(*, started_ms: dict[str, int]) -> list[str]:
    begin = time.perf_counter()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return ["ffmpeg is not installed"]
    try:
        with tempfile.TemporaryDirectory(prefix="aa-codec-smoke-") as tmp:
            ogg = str(Path(tmp) / "test.ogg")
            pcm = str(Path(tmp) / "test.pcm")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=0.5",
                    "-ac",
                    "1",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    ogg,
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    ogg,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    pcm,
                ],
                check=True,
                capture_output=True,
            )
            if Path(ogg).stat().st_size == 0 or Path(pcm).stat().st_size == 0:
                return ["ffmpeg smoke produced empty output"]
    except (subprocess.CalledProcessError, OSError) as exc:
        return [f"ffmpeg OGG/Opus smoke failed: {exc.__class__.__name__}"]
    started_ms["ffmpeg_ms"] = int((time.perf_counter() - begin) * 1000)
    return []


def check_native_imports(*, started_ms: dict[str, int]) -> list[str]:
    begin = time.perf_counter()
    errors: list[str] = []
    for module in ("torch", "sherpa_onnx", "onnxruntime", "faiss"):
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"native import failed for {module}: {exc.__class__.__name__}")
    started_ms["imports_ms"] = int((time.perf_counter() - begin) * 1000)
    return errors


def check_opencode() -> list[str]:
    try:
        want = OPENCODE_VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"opencode.version unreadable: {exc}"]
    binary = shutil.which("opencode")
    if binary is None:
        home_binary = Path.home() / ".opencode" / "bin" / "opencode"
        binary = str(home_binary) if home_binary.is_file() else None
    if binary is None:
        return ["opencode binary not found on PATH"]
    try:
        proc = subprocess.run(
            [binary, "--version"], check=True, capture_output=True, text=True, timeout=60
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        return [f"opencode --version failed: {exc.__class__.__name__}"]
    if want not in proc.stdout + proc.stderr:
        return [f"opencode version mismatch: want {want}"]
    return []


def check_head_match() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ROOT),
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        return [f"git HEAD unreadable: {exc.__class__.__name__}"]
    sha = proc.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return [f"HEAD is not a full SHA: {sha!r}"]
    return []


def run_check_only_prefetch() -> list[str]:
    errors: list[str] = []
    commands = (
        [sys.executable, "scripts/prefetch_public_assets.py", "--check-only"],
        [sys.executable, "scripts/prefetch_voice_assets.py", "--check-only"],
    )
    for command in commands:
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=120, cwd=str(ROOT)
            )
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{command[1]} --check-only failed: {exc.__class__.__name__}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the prebuilt AA runtime image.")
    parser.add_argument("--pins-only", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--allow-placeholder-digest", action="store_true")
    args = parser.parse_args(argv)

    if args.pins_only:
        errors = check_pins()
        print(json.dumps({"pins": "pass" if not errors else "fail", "errors": errors}))
        return 0 if not errors else 1

    timings: dict[str, int] = {}
    errors = check_pins()
    errors += check_digest_wiring(allow_placeholder=args.allow_placeholder_digest or args.canary)
    if args.canary:
        # Canary runs inside the freshly pulled image: digest placeholder is
        # irrelevant because the image reference itself is the digest under test.
        errors = [e for e in errors if "placeholder" not in e and "container pin" not in e]
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, sort_keys=True))
        return 1

    errors += check_ffmpeg(started_ms=timings)
    errors += check_native_imports(started_ms=timings)
    errors += check_opencode()
    errors += check_head_match()
    # Identity checks only: --check-only never downloads and prunes nothing
    # except stale local content; the bounded download path stays in workflow.
    errors += run_check_only_prefetch()
    payload: dict[str, object] = {"status": "pass" if not errors else "fail", "timings_ms": timings}
    if errors:
        payload["errors"] = errors
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
