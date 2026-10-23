#!/usr/bin/env python3
"""Deterministic runtime-image promotion contract guard (kodmial/aa#441).

Gate C startup failed (run 39884474265) because promotion left consumers
stale: `docker/aa-runtime.digest` and `aa-runtime.yml` pointed at the
canary-verified promoted digest while the qualification jobs, the
derived-index publisher and the bootstrap-metrics consumer silently kept
the obsolete image whose prepared toolchain was missing, producing
`{"pins":"pass","errors":[]}` followed by `pytest: command not found`.

This guard enforces the promotion contract with no network and no Docker:

- every `ghcr.io/kodmial/aa-runtime@sha256:<digest>` occurrence in every
  AA workflow file must equal the single promoted digest recorded in
  docker/aa-runtime.digest (this also rejects newly added consumers that
  fall back to an obsolete digest);
- the qualification workflow must keep BOTH of its independent jobs
  (`qualify` and `qualify-live`) pinned to that promoted digest;
- the promotion script must update and retain exactly the consumer set it
  declares in CONSUMER_WORKFLOWS (a newly added consumer cannot silently
  fall back to an obsolete digest, and a stale registration is reported);
- every prepared-environment consumer must keep the STATIC `--pins-only`
  pin-declaration preflight AND the ACTUAL binary-availability
  `--tools-only` preflight wired, so the two are never confused again
  (run 39884474265 printed pins-pass and failed `pytest` with exit 127).

Exit code 0 on a consistent universe, non-zero on any mismatch. The
expected digest is always derived from the promotion source itself (never
a literal), so a fresh canary-verified promotion passes this guard too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

PROMOTION_SOURCE_REL = Path("docker") / "aa-runtime.digest"
WORKFLOW_DIR_REL = Path(".github") / "workflows"
PROMOTION_SCRIPT_REL = Path("scripts") / "promote_runtime_image.py"
PROMOTER_NEXT_TO_GUARD = Path(__file__).resolve().parent / "promote_runtime_image.py"

DIGEST_RE = re.compile(r"^ghcr\.io/kodmial/aa-runtime@sha256:([0-9a-f]{64})$")
IMAGE_OCCURRENCE_RE = re.compile(r"ghcr\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}")
CONSUMER_DECL_RE = re.compile(r'workflows"\s*/\s*"([A-Za-z0-9_.-]+\.yml)"')
ZERO = "0" * 64
RUNTIME_WORKFLOW = "aa-runtime.yml"
QUALIFICATION_WORKFLOW = "aa-self-proving-qualification.yml"
QUALIFICATION_JOBS = ("qualify", "qualify-live")
# Prepared-environment consumers that must keep BOTH preflights wired:
# the static pin-declaration check and the actual binary availability.
TOOLS_PREFLIGHT_WORKFLOWS = (
    QUALIFICATION_WORKFLOW,
    "aa-derived-index.yml",
    "aa-bootstrap-metrics.yml",
)


def _fail(message: str) -> None:
    print(f"runtime-image consumer guard failed: {message}", file=sys.stderr)


def _read_promoted_reference(root: Path) -> tuple[str, list[str]]:
    messages: list[str] = []
    try:
        lines = [
            line.strip()
            for line in (root / PROMOTION_SOURCE_REL).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        record = lines[-1] if lines else ""
    except OSError as exc:
        record = ""
        messages.append(f"promoted digest record unreadable: {exc.__class__.__name__}")
    match = DIGEST_RE.fullmatch(record)
    if match is None:
        messages.append(f"promoted digest record malformed: {record!r}")
        return "", messages
    if match.group(1) == ZERO:
        messages.append("promoted digest is the unpublished all-zero placeholder")
        return "", messages
    return record, messages


def _read_promoter_text(root: Path) -> tuple[str, list[str]]:
    messages: list[str] = []
    for candidate in (root / PROMOTION_SCRIPT_REL, PROMOTER_NEXT_TO_GUARD):
        try:
            return candidate.read_text(encoding="utf-8"), messages
        except OSError:
            continue
    messages.append("promotion contract source unreadable (promote_runtime_image.py)")
    return "", messages


def _read_workflows(root: Path) -> tuple[dict[str, str], list[str]]:
    messages: list[str] = []
    workflows: dict[str, str] = {}
    for path in sorted((root / WORKFLOW_DIR_REL).glob("*.yml")):
        try:
            workflows[path.name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            messages.append(
                f"{path.name}: workflow unreadable for the guard: {exc.__class__.__name__}"
            )
    if not workflows:
        messages.append("no workflow files found to guard")
    return workflows, messages


def _job_image_pins(workflow_text: str, name: str, messages: list[str]) -> dict[str, str]:
    """Job-name -> container image for every aa-runtime image mapping.

    Container pins are plain scalars in the workflows, so the literal
    digest values survive `yaml.safe_load`; every `container: image:` that
    references the runtime image is collected with its owning job name.
    Provenance-only occurrences (echo/summary lines or shell variables) do
    not create a job pin and are already covered by the literal equality
    check over every line.
    """
    try:
        data = yaml.safe_load(workflow_text)
    except yaml.YAMLError as exc:
        messages.append(f"{name}: workflow YAML invalid: {exc}")
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        messages.append(f"{name}: workflow has no jobs mapping")
        return {}
    pins: dict[str, str] = {}

    def walk(node: object, job_name: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "image" and isinstance(value, str) and "aa-runtime@" in value:
                    pins.setdefault(job_name, value)
                walk(value, job_name)
        elif isinstance(node, list):
            for item in node:
                walk(item, job_name)

    for job_name, job_body in data["jobs"].items():
        walk(job_body, str(job_name))
    return pins


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to guard (defaults to this checkout)",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    messages: list[str] = []

    promoted, new_messages = _read_promoted_reference(root)
    messages += new_messages
    promoter_text, new_messages = _read_promoter_text(root)
    messages += new_messages
    workflows, new_messages = _read_workflows(root)
    messages += new_messages

    consumer_files: dict[str, list[tuple[int, str]]] = {}
    for name, text in sorted(workflows.items()):
        for line_no, line in enumerate(text.splitlines(), start=1):
            for ref in IMAGE_OCCURRENCE_RE.findall(line):
                consumer_files.setdefault(name, []).append((line_no, ref))
                if promoted:
                    if ref != promoted:
                        messages.append(
                            f"{name}:{line_no} pins obsolete runtime image {ref}; "
                            f"promoted source is {promoted}"
                        )
                else:
                    messages.append(
                        f"{name}:{line_no} pins {ref} while the promoted "
                        "digest record is invalid; the promotion source must "
                        "be repaired, consumer equality is undecidable"
                    )
    if promoted and RUNTIME_WORKFLOW not in consumer_files:
        messages.append(
            f"{RUNTIME_WORKFLOW} no longer pins the promoted digest; "
            "the promotion source would be orphaned"
        )
    if consumer_files and promoted:
        # Every occurrence must be realized as a real job container pin
        # and the qualification workflow must keep BOTH jobs pinned.
        for name in consumer_files:
            pins = _job_image_pins(workflows[name], name, messages)
            if not pins:
                messages.append(
                    f"{name} references the runtime image but has no job "
                    "container pin; consumers must pin the promoted image "
                    f"{promoted} in their job images"
                )
            for job, image in pins.items():
                match = DIGEST_RE.fullmatch(image)
                if match is None or match.group(0) != promoted:
                    messages.append(
                        f"{name}: job '{job}' container image {image!r} differs "
                        f"from promoted {promoted}"
                    )
            if name == QUALIFICATION_WORKFLOW:
                missing = [job for job in QUALIFICATION_JOBS if job not in pins]
                if missing:
                    messages.append(
                        f"{name}: qualification jobs missing the promoted image pin: {missing}"
                    )
        # Promotion coverage: no unregistered consumer, no declared
        # consumer that silently stopped pinning.
        declared = set(CONSUMER_DECL_RE.findall(promoter_text)) - {RUNTIME_WORKFLOW}
        present = set(consumer_files) - {RUNTIME_WORKFLOW}
        if not declared:
            messages.append(
                "promotion script declares no aa-runtime consumer workflows "
                "(CONSUMER_WORKFLOWS contract drifted)"
            )
        for extra in sorted(present - declared):
            messages.append(
                f"{extra} pins a runtime-image consumer but is not registered "
                "in scripts/promote_runtime_image.py CONSUMER_WORKFLOWS; a "
                "future promotion would silently leave it stale"
            )
        for registered in sorted(declared - present):
            messages.append(
                f"promotion consumer {registered} is declared but no longer "
                "pins the runtime image; drop it from CONSUMER_WORKFLOWS or "
                "restore the pin"
            )
        # Static pin declarations vs actual binary availability must stay
        # wired together in every prepared-environment consumer.
        for name in TOOLS_PREFLIGHT_WORKFLOWS:
            if name not in present:
                continue
            text = workflows[name]
            if "--pins-only" not in text or "--tools-only" not in text:
                messages.append(
                    f"{name} must keep static --pins-only and actual "
                    "binary-availability --tools-only preflight"
                )

    consumer_names = (
        sorted(name for name in consumer_files if name != RUNTIME_WORKFLOW)
        if consumer_files
        else []
    )
    status = "fail" if messages else "pass"
    print(
        json.dumps(
            {
                "promoted": promoted or None,
                "consumers": consumer_names,
                "occurrences": sum(len(lines) for lines in consumer_files.values()),
                "status": status,
                "errors": messages,
            }
        )
    )
    if messages:
        for message in messages:
            _fail(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
