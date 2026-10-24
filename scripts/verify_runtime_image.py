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

``--tools-only`` (kodmial/aa#441) checks the ACTUAL availability of the
prepared dev toolchain (pytest/ruff/mypy, spaCy RU model, pinned OpenCode
CLI) inside the currently selected container. It exists because a green
``--pins-only`` result only validates source/pin declarations and never
proved what binaries the selected image provides: run 39884474265 printed
``{"pins":"pass","errors":[]}`` and then failed `pytest: command not
found` with exit 127 before Gate C scenario 1.

Never starts a Telegram poller, never touches secrets, book plaintext,
decrypted indexes, audio, transcripts, or generated responses.
"""

from __future__ import annotations

import argparse
import json
import os
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
INDEX_DIGEST_FILE = ROOT / "docker" / "aa-index.digest"
RUNTIME_WORKFLOW = ROOT / ".github" / "workflows" / "aa-runtime.yml"
PYPROJECT = ROOT / "pyproject.toml"

DIGEST_RE = re.compile(r"^ghcr\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}$")
INDEX_REFERENCE_RE = re.compile(r"^ghcr\.io/kodmial/aa-index@sha256:[0-9a-f]{64}$")
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
    "pymorphy3==2.0.6",
    "pymorphy3-dicts-ru==2.4.417150.4580142",
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
    elif "COPY opencode.version /opt/aa/opencode.version" not in docker_text:
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
    copy_lines = [
        line.strip()
        for line in docker_text.splitlines()
        if line.strip().startswith(("COPY ", "ADD "))
    ]
    if copy_lines != ["COPY opencode.version /opt/aa/opencode.version"]:
        errors.append("image must copy only the pinned OpenCode version, never AA source")
    for forbidden in FORBIDDEN_DOCKER_COPY:
        for line in copy_lines:
            if forbidden in line:
                errors.append(f"Dockerfile must not COPY {forbidden}")
    errors += check_dev_pins(docker_text=docker_text, project_text=project_text)
    return errors


_DEV_PIN_RE = re.compile(r'"([A-Za-z0-9_.-]+)==([0-9a-zA-Z.]+)"')
_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*(.*)$")
_CONSTRAINT_RE = re.compile(r"(===|==|!=|>=|<=|~=|>|<)\s*([^\s,;\"']+)")


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in value.strip().split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers)


def _compare_version(left: str, right: str) -> int:
    left_tuple = list(_version_tuple(left)) or [0]
    right_tuple = list(_version_tuple(right)) or [0]
    width = max(len(left_tuple), len(right_tuple))
    left_tuple += [0] * (width - len(left_tuple))
    right_tuple += [0] * (width - len(right_tuple))
    return (tuple(left_tuple) > tuple(right_tuple)) - (tuple(left_tuple) < tuple(right_tuple))


def _compatible_release(want: str, got: str) -> bool:
    """PEP 440 compatible-release (~=) check for numeric versions."""
    if _compare_version(got, want) < 0:
        return False
    want_parts = want.split(".")
    got_parts = got.split(".")
    prefix_len = max(1, len(want_parts) - 1)
    want_prefix = _version_tuple(".".join(want_parts[:prefix_len]))
    got_prefix = _version_tuple(".".join(got_parts[:prefix_len]))
    return got_prefix == want_prefix


def _op_satisfied(op: str, want: str, got: str) -> bool:
    order = _compare_version(got, want)
    if op == "===":
        return got == want
    if op == "==":
        if want.endswith(".*"):
            prefix = want[:-2]
            return got == prefix or got.startswith(prefix + ".")
        return order == 0
    if op == "!=":
        if want.endswith(".*"):
            prefix = want[:-2]
            return not (got == prefix or got.startswith(prefix + "."))
        return order != 0
    if op == ">=":
        return order >= 0
    if op == "<=":
        return order <= 0
    if op == ">":
        return order > 0
    if op == "<":
        return order < 0
    if op == "~=":
        return _compatible_release(want, got)
    return False


def _parse_dev_selectors(dev_section: str) -> list[tuple[str, str, str]]:
    """Extract every version constraint, including extras and multi-clause ranges."""
    selectors: list[tuple[str, str, str]] = []
    for raw_line in dev_section.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("["):
            continue
        line = line.strip('"').strip("'").rstrip(",").strip()
        if " @ " in line:
            # Direct URL requirements carry no comparable version selector.
            continue
        match = _REQUIREMENT_RE.match(line)
        if match is None:
            continue
        name = match.group(1)
        rest = match.group(2)
        for op, version in _CONSTRAINT_RE.findall(rest):
            selectors.append((name, op, version))
    return selectors


def check_dev_pins(*, docker_text: str, project_text: str) -> list[str]:
    """Verify the baked dev/qualification pins stay within the pyproject ranges.

    The prepared environment serves Gate A/B (ruff/mypy/pytest) and Gate C
    (spacy RU model). Drift between an exact image pin and a relaxed
    selector range must be visible, never silently stale.
    """
    errors: list[str] = []
    if "[project.optional-dependencies]" not in project_text:
        return ["pyproject.toml optional-dependencies section is missing"]
    remainder = project_text.split("[project.optional-dependencies]", 1)[1]
    dev_lines: list[str] = []
    for line in remainder.splitlines():
        if line.strip().startswith("["):
            break
        dev_lines.append(line)
    dev_section = "\n".join(dev_lines)
    selectors = _parse_dev_selectors(dev_section)
    if not selectors:
        return ["pyproject.toml dev selectors are unreadable"]
    pins: dict[str, str] = {}
    for name, version in _DEV_PIN_RE.findall(docker_text):
        pins[name.lower()] = version
    for name, op, want in selectors:
        got = pins.get(name.lower())
        if got is None:
            errors.append(f"Dockerfile dev layer lacks pin for {name}")
            continue
        if not _op_satisfied(op, want, got):
            errors.append(f"Dockerfile dev pin {name}=={got} violates selector {name}{op}{want}")
    for required in ("pytest", "ruff", "mypy", "spacy"):
        if required not in pins:
            errors.append(f"Dockerfile dev layer lacks {required}")
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
    container_active = "\n    container:\n" in workflow_text
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


def check_index_digest_wiring() -> list[str]:
    """Fail closed on a malformed persistent derived-index L2 pin (issue #429)."""
    errors: list[str] = []
    if not INDEX_DIGEST_FILE.is_file():
        return ["docker/aa-index.digest metadata file is missing"]
    lines = INDEX_DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()
    reference = lines[-1].strip() if lines else ""
    if not INDEX_REFERENCE_RE.fullmatch(reference):
        return [f"index digest file malformed: {reference!r}"]
    if reference.endswith(":" + ZERO_DIGEST):
        # Placeholder is the honest pre-publication state; verify the
        # consumer client's actual behavior instead of grepping for a
        # fragile error-message string.
        errors += _check_placeholder_rejected()

    if not (ROOT / ".github" / "workflows" / "aa-derived-index.yml").is_file():
        errors.append("trusted aa-derived-index publisher workflow is missing")
    return errors


def _check_placeholder_rejected() -> list[str]:
    try:
        from aa.retrieval.oci_artifact import ArtifactError, parse_pinned_digest
    except ImportError:
        src = str(ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from aa.retrieval.oci_artifact import ArtifactError, parse_pinned_digest
        except ImportError:
            return ["cannot import oci_artifact module for placeholder check"]
    try:
        parse_pinned_digest(INDEX_DIGEST_FILE)
    except ArtifactError as exc:
        if exc.category != "l2-unpublished":
            return [f"placeholder pin rejected with wrong category: {exc.category}"]
        return []
    return ["consumer client does not reject placeholder L2 pin"]


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
        want = (
            os.environ.get("CONTINUUM_OPENCODE_VERSION")
            or os.environ.get("AA_OPENCODE_VERSION")
            or OPENCODE_VERSION_FILE.read_text(encoding="utf-8").strip()
        )
    except OSError as exc:
        return [f"opencode.version unreadable: {exc}"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", want):
        return ["invalid effective OpenCode version"]
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
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
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


def check_dev_toolchain() -> list[str]:
    """Prepared qualification toolchain works (issue #429): no reinstall."""
    errors: list[str] = []
    for tool in ("pytest", "ruff", "mypy"):
        try:
            subprocess.run(
                [sys.executable, "-m", tool, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"dev toolchain {tool} failed: {exc.__class__.__name__}")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import spacy; spacy.load('ru_core_news_sm'); print('ok')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if "ok" not in proc.stdout:
            errors.append("spacy RU model check produced no confirmation")
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"spacy RU model check failed: {exc.__class__.__name__}")
    return errors


def check_console_scripts_on_path() -> list[str]:
    """Console-script entries on PATH (kodmial/aa#441).

    The Gate C incident evidence was `pytest: command not found` (exit 127):
    the workflow runs the bare `pytest`/`ruff`/`mypy` console scripts and
    the pinned `opencode` CLI, so a missing PATH entry is a real prepared
    environment failure even when `-m<tool>` succeeds in some interpreter.
    Static --pins-only reports pass without validating any of this.
    """
    errors: list[str] = []
    for tool in ("pytest", "ruff", "mypy", "opencode"):
        if shutil.which(tool) is None:
            errors.append(f"console script {tool} not found on PATH")
    return errors


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
    parser.add_argument("--tools-only", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--allow-placeholder-digest", action="store_true")
    args = parser.parse_args(argv)

    if args.pins_only:
        errors = check_pins()
        print(json.dumps({"pins": "pass" if not errors else "fail", "errors": errors}))
        return 0 if not errors else 1

    if args.tools_only:
        # kodmial/aa#441: actual binary availability in the SELECTED
        # container, deliberately distinct from the static --pins-only
        # declaration check. No network, no installs, no hot-path actions.
        errors = check_console_scripts_on_path() + check_dev_toolchain() + check_opencode()
        print(json.dumps({"tools": "pass" if not errors else "fail", "errors": errors}))
        return 0 if not errors else 1

    timings: dict[str, int] = {}
    errors = check_pins()
    if not args.canary:
        errors += check_digest_wiring(allow_placeholder=args.allow_placeholder_digest)
    errors += check_index_digest_wiring()
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, sort_keys=True))
        return 1

    errors += check_ffmpeg(started_ms=timings)
    errors += check_native_imports(started_ms=timings)
    errors += check_opencode()
    errors += check_dev_toolchain()
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
