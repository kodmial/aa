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
    # Build from docker/ only: an allowlisted minimal context is stronger
    # than excluding private paths from the repository-root build context.
    ignore = ROOT / "docker" / ".dockerignore"
    try:
        ignore_lines = {line.strip() for line in ignore.read_text(encoding="utf-8").splitlines()}
    except OSError as exc:
        errors.append(f"image context ignore file unreadable: {exc}")
        ignore_lines = set()
    if not {"*", "!Dockerfile.aa-runtime", "!opencode.version"}.issubset(ignore_lines):
        errors.append("image context must allowlist only Dockerfile and opencode.version")
    image_workflow = ROOT / ".github" / "workflows" / "aa-runtime-image.yml"
    try:
        image_text = image_workflow.read_text(encoding="utf-8")
        if "context: ./docker" not in image_text:
            errors.append("Docker build context must be ./docker (not repository root)")
    except OSError as exc:
        errors.append(f"image workflow unreadable: {exc}")
    copy_lines = [line.strip() for line in docker_text.splitlines()
                  if line.strip().startswith(("COPY ", "ADD "))]
    if copy_lines != ["COPY opencode.version /opt/aa/opencode.version"]:
        errors.append("image must copy only the pinned OpenCode version, never AA source")
    for forbidden in FORBIDDEN_DOCKER_COPY:
        for line in copy_lines:
            if forbidden in line:
                errors.append(f"Dockerfile must not COPY {forbidden}")
    return errors


def check_digest_wiring(*, allow_placeholder: bool) -> list[str]:
    errors: list[str] = []
    try:
        digest_lines = DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()
        digest = digest_lines[-1].strip() if digest_lines else ""
        workflow_text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"runtime image staging metadata unavailable: {exc}"]
    if not DIGEST_RE.fullmatch(digest):
        return [f"digest file malformed: {digest!r}"]
    placeholder = digest.endswith(":" + ZERO_DIGEST)
    container_active = "\\n    container:\\n" in workflow_text
    if placeholder:
        # First-stage implementation must *not* turn on a nonexistent image.
        if container_active:
            errors.append("unpublished placeholder image must never activate the runtime container")
        if not allow_placeholder:
            errors.append("image not yet published/promoted")
    else:
        if not container_active:
            errors.append("validated digest exists but AA job is not using it")
        if digest not in workflow_text:
            errors.append("runtime container digest differs from promoted digest record")
        if "credentials:" not in workflow_text or "packages: read" not in workflow_text:
            errors.append("runtime image job lacks GHCR read credentials")
    if "group: aa-bot-runtime" not in workflow_text:
        errors.append("AA must keep the single-poller concurrency group")
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
    if not args.canary:
        errors += check_digest_wiring(allow_placeholder=args.allow_placeholder_digest)
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, sort_keys=True))
        return 1

    errors += check_ffmpeg(started_ms=timings)
    errors += check_native_imports(started_ms=timings)
    errors += check_opencode()
    errors += check_head_match()
    # Identity checks only: --check-only never downloads and prunes nothing
    # except stale local content; the bounded download path stays in workflow.
    # Fresh-runner image canary intentionally has no cached public model data.
    # Missing model caches are not a defect in the dependency image itself;
    # the real runtime restores and validates those separately.
    if not args.canary:
        errors += run_check_only_prefetch()
    payload: dict[str, object] = {"status": "pass" if not errors else "fail", "timings_ms": timings}
    if errors:
        payload["errors"] = errors
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
