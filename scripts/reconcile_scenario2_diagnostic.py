#!/usr/bin/env python3
"""AA-owned scenario #2 diagnostic post-run reconciler (kodmial/aa#425).

Consumes the newest AA-owned diagnostic Actions run for the exact main
SHA (or its accepted machinery-identical ancestor), downloads that run's
artifact through the authenticated Actions API, validates identity/schema
/path safety, and publishes ONE deterministic privacy-safe triage comment
per (sha, run) on operations issue #419. On a proven harness/evidence gap
it plans exactly ONE minimal repair issue (fingerprint-deduplicated), and
when no diagnostic run exists for the SHA it dispatches the existing
single-scenario workflow (``target_sha`` input only) with bounded
idempotent retries. There is NO ChatGPT/external-LLM judge: every
decision comes from ``aa.control.scenario2_triage`` (pure, unit-tested).

Live GitHub access is isolated to small authenticated ``gh`` calls; tests
exercise the decision core directly through the pure module.

Exit codes:
- 0: reconciled (ANY typed outcome, incl. blocked evidence — truthful),
- 3: GitHub API access failed (named missing permissions surfaced),
- 2: malformed local inventory input failed closed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aa.control import scenario2_triage as triage  # noqa: E402

_TRIAGE_MODULE = Path(triage.__file__).resolve()
if not _TRIAGE_MODULE.is_relative_to(SRC_ROOT):
    raise ImportError(f"aa.control.scenario2_triage resolved outside repo: {_TRIAGE_MODULE}")

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_GITHUB_ERROR = 3

MISSING_PERMISSION_MARKERS = (
    "resource not accessible by integration",
    "http 403",
    "403 forbidden",
    "issues write access",
)

ARTIFACT_CACHE_DIR = Path(tempfile.gettempdir()) / f"aa-scenario2-triage-artifacts-{os.getpid()}"


# ---------------------------------------------------------------------------
# Small scoped gh calls (the only authenticated boundary)
# ---------------------------------------------------------------------------


class GhError(RuntimeError):
    """One failed gh call with a sanitized stderr detail."""


def _gh_text(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        snippet = lines[-1][:200] if lines else ""
        raise GhError(f"gh {' '.join(args[:2])} failed: {snippet}")
    return proc.stdout


def _gh_api_json(args: list[str], *, payload: Any = None) -> Any:
    """Small gh api call; payloads travel through a chmod-600 temp file."""
    if payload is None:
        return json.loads(_gh_text(["api", *args]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "payload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)
        return json.loads(_gh_text(["api", *args, "--input", str(path)]))


def _gh_api_bytes(endpoint: str) -> bytes:
    proc = subprocess.run(["gh", "api", endpoint], capture_output=True, check=False)
    if proc.returncode != 0:
        lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        snippet = lines[-1][:200] if lines else ""
        raise GhError(f"gh api {endpoint} failed: {snippet}")
    return proc.stdout


def collect_runs(owner: str, repo: str) -> list[triage.DiagnosticRun]:
    runs: list[triage.DiagnosticRun] = []
    page = 1
    while True:
        payload = _gh_api_json(
            [
                f"repos/{owner}/{repo}/actions/workflows/"
                f"{triage.DIAGNOSTIC_WORKFLOW_FILE}/runs"
                f"?branch=main&per_page=100&page={page}"
            ]
        )
        page_runs = triage.parse_runs(payload)
        runs.extend(page_runs)
        items = payload.get("workflow_runs") if isinstance(payload, dict) else payload
        if not isinstance(items, list) or len(items) < 100:
            break
        page += 1
    return runs


def collect_issue_comments(owner: str, repo: str, issue: int) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    page = 1
    while True:
        payload = _gh_api_json(
            [f"repos/{owner}/{repo}/issues/{issue}/comments?per_page=100&page={page}"]
        )
        items = payload if isinstance(payload, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                pairs.append((int(item["id"]), str(item.get("body", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        if len(items) < 100:
            break
        page += 1
    return pairs


def collect_open_issue_bodies(owner: str, repo: str) -> list[str]:
    payload = _gh_api_json([f"repos/{owner}/{repo}/issues?state=open&per_page=100"])
    bodies: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict) or "pull_request" in item:
                continue
            bodies.append(f"{item.get('title', '')}\n{item.get('body', '')}")
    return bodies


def collect_main_sha(owner: str, repo: str) -> str:
    payload = _gh_api_json([f"repos/{owner}/{repo}/branches/{MAIN_BRANCH}"])
    commit = payload.get("commit") if isinstance(payload, dict) else {}
    sha = str((commit or {}).get("sha", ""))
    return triage.normalize_sha(sha)


def collect_artifact(owner: str, repo: str, run_id: int) -> tuple[int | None, bool]:
    """(`artifact_id`, `expired`) for the AA-owned diagnostic artifact."""
    payload = _gh_api_json([f"repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"])
    items = payload.get("artifacts") if isinstance(payload, dict) else []
    expected = triage.artifact_name_for_run(run_id)
    for item in items or []:
        if not isinstance(item, dict) or str(item.get("name", "")) != expected:
            continue
        artifact_id = item.get("id")
        if isinstance(artifact_id, int) and artifact_id > 0:
            return artifact_id, bool(item.get("expired", False))
    return None, False


@lru_cache(maxsize=64)
def _machinery_unchanged_cached(owner: str, repo: str, older: str, newer: str) -> bool:
    if older == newer:
        return True
    payload = _gh_api_json([f"repos/{owner}/{repo}/compare/{older}...{newer}"])
    files = payload.get("files") if isinstance(payload, dict) else []
    for item in files or []:
        if isinstance(item, dict) and str(item.get("filename", "")) in triage.MACHINERY_FILES:
            return False
    return True


def machinery_unchanged_between(owner: str, repo: str, *, older: str, newer: str) -> bool:
    """True when no diagnostic machinery file differs between two SHAs."""
    return _machinery_unchanged_cached(owner, repo, older, newer)


# ---------------------------------------------------------------------------
# GitHub writes (single idempotent comment, one repair issue, one dispatch)
# ---------------------------------------------------------------------------


def post_triage_comment(
    owner: str, repo: str, issue: int, action: triage.CommentAction, body: str
) -> str:
    if action.kind == "update" and action.comment_id is not None:
        _gh_api_json(
            ["-X", "PATCH", f"repos/{owner}/{repo}/issues/comments/{action.comment_id}"],
            payload={"body": body},
        )
        return f"updated comment {action.comment_id} in place"
    _gh_api_json([f"repos/{owner}/{repo}/issues/{issue}/comments"], payload={"body": body})
    return f"created comment on #{issue}"


def create_repair_issue(owner: str, repo: str, plan: triage.RepairIssuePlan) -> int | None:
    created = _gh_api_json(
        [f"repos/{owner}/{repo}/issues"],
        payload={
            "title": plan.title,
            "body": plan.body,
            "labels": list(plan.labels),
        },
    )
    number = created.get("number") if isinstance(created, dict) else None
    return int(number) if isinstance(number, int) else None


def post_dispatch_marker(owner: str, repo: str, issue: int, marker_body: str) -> None:
    _gh_api_json([f"repos/{owner}/{repo}/issues/{issue}/comments"], payload={"body": marker_body})


def dispatch_diagnostic(owner: str, repo: str, target_sha: str) -> None:
    """Trigger the EXISTING diagnostic workflow (target_sha input only)."""
    deltas: list[str] = []
    for key, value in triage.build_dispatch_inputs(target_sha).items():
        deltas += ["-f", f"{key}={value}"]
    _gh_text(["workflow", "run", triage.DIAGNOSTIC_WORKFLOW_FILE, "--ref", "main", *deltas])


# ---------------------------------------------------------------------------
# Artifact evidence evaluation (never executed; read-only, safe paths)
# ---------------------------------------------------------------------------


def _zip_member_names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise zipfile.BadZipFile("artifact fails its own integrity check")
        return [info.filename for info in archive.infolist()]


def _read_zip_member(zip_path: Path, member: str) -> bytes | None:
    with zipfile.ZipFile(zip_path) as archive:
        wanted = member.lstrip("/")
        for info in archive.infolist():
            name = info.filename.lstrip("/")
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError(f"unsafe artifact member path: {info.filename!r}")
            if name != wanted:
                continue
            if info.flag_bits & 0x1:  # encrypted member never accepted
                raise ValueError("encrypted artifact member rejected")
            with archive.open(info) as stream:
                return stream.read()
    return None


def _diagnostic_banner(report_raw: bytes | None) -> str | None:
    """Sanitized one-line self-report banner from final-report.md."""
    if report_raw is None:
        return None
    text = report_raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("> **outcome:"):
            triage.assert_sanitized(stripped, owner="diagnostic-banner")
            return stripped[:120]
    return None


def evaluate_artifact(
    artifact_zip: Path, run: triage.DiagnosticRun
) -> tuple[
    triage.TriageOutcome, list[triage.TimelineEntry], list[str], tuple[str, ...], str | None
]:
    """Evaluate AA-owned artifact evidence for the selected run."""
    try:
        member_names = _zip_member_names(artifact_zip)
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        return (
            triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_CORRUPT),
            [],
            [f"artifact unreadable/corrupt: {type(exc).__name__}"],
            (),
            None,
        )
    if triage.OPTIONAL_REPORT_MEMBER in member_names:
        banner = _diagnostic_banner(_read_zip_member(artifact_zip, triage.OPTIONAL_REPORT_MEMBER))
    else:
        banner = None
    missing_required = [member for member in triage.REQUIRED_MEMBERS if member not in member_names]
    if missing_required:
        outcome = triage.map_typed_outcome(None, missing_files=missing_required)
        gaps = [
            "required artifact members missing: " + ", ".join(missing_required),
        ]
        if banner is not None:
            gaps.append(f"diagnostic self-report banner: {banner}")
        return outcome, [], gaps, (), banner
    result_raw = _read_zip_member(artifact_zip, triage.REQUIRED_RESULT_MEMBER)
    if result_raw is None:
        return (
            triage.map_typed_outcome(None, blocked_code=triage.BLOCK_RESULT_MISSING),
            [],
            [],
            (),
            banner,
        )
    try:
        result = json.loads(result_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return (
            triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_CORRUPT),
            [],
            [f"result.json unreadable: {type(exc).__name__}"],
            (),
            banner,
        )
    if not isinstance(result, dict):
        return (
            triage.map_typed_outcome(None, blocked_code=triage.BLOCK_SCHEMA_INVALID),
            [],
            [],
            (),
            banner,
        )
    identity_error = triage.validate_result_identity(result, run)
    if identity_error is not None:
        return (
            triage.map_typed_outcome(None, blocked_code=triage.BLOCK_IDENTITY_MISMATCH),
            [],
            [identity_error],
            (),
            banner,
        )
    timeline = triage.build_timeline(
        {
            "supervisor": triage.parse_jsonl(
                _read_zip_member(artifact_zip, triage.REQUIRED_SUPERVISOR_MEMBER) or b""
            ),
            "probe-direct-model": triage.parse_jsonl(
                _read_zip_member(artifact_zip, "probe-direct-model-events.jsonl") or b""
            ),
            "probe-scenario2": triage.parse_jsonl(
                _read_zip_member(artifact_zip, triage.REQUIRED_SCENARIO_EVENTS_MEMBER) or b""
            ),
        }
    )
    outcome = triage.map_typed_outcome(result)
    return outcome, timeline, [], triage.slow_work_crossings(result), banner


def download_artifact(owner: str, repo: str, artifact_id: int) -> Path:
    """Download one AA-owned artifact zip into the bounded cache dir."""
    ARTIFACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ARTIFACT_CACHE_DIR / f"{artifact_id}.zip"
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        raw = _gh_api_bytes(f"repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip")
        part = zip_path.with_suffix(".part")
        part.write_bytes(raw)
        part.chmod(0o600)
        part.replace(zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

MAIN_BRANCH = "main"


def _sorted_completed(runs: list[triage.DiagnosticRun]) -> list[triage.DiagnosticRun]:
    def sort_key(run: triage.DiagnosticRun) -> tuple[bool, float, int]:
        created_epoch = triage.parse_iso_epoch(run.created_at)
        return (
            created_epoch is not None,
            created_epoch if created_epoch is not None else 0.0,
            run.run_id,
        )

    return sorted(
        (run for run in runs if run.is_completed),
        key=sort_key,
        reverse=True,
    )


def select_with_ancestor_backfill(
    runs: list[triage.DiagnosticRun],
    *,
    owner: str,
    repo: str,
    allowed: tuple[str, ...],
    target_sha: str,
    resolve_ancestor: bool,
) -> tuple[triage.SelectionDecision, str]:
    """Selection, accepting a machinery-identical ancestor diagnostic run.

    Backfills the diagnostic produced before this reconciler was installed:
    a diagnostic run from an older main SHA is accepted evidence while the
    current tree differs in NO diagnostic machinery file.
    """
    accepted_ancestor = ""
    selection = triage.select_diagnostic_run(runs, allowed_shas=allowed)
    if selection.action != "missing" or not resolve_ancestor:
        return selection, accepted_ancestor
    newest_completed = next(iter(_sorted_completed(runs)), None)
    if newest_completed is None:
        return selection, accepted_ancestor
    if machinery_unchanged_between(owner, repo, older=newest_completed.head_sha, newer=target_sha):
        accepted_ancestor = newest_completed.head_sha
        selection = triage.select_diagnostic_run(
            runs, allowed_shas=triage.normalize_allowed_shas((*allowed, accepted_ancestor))
        )
    return selection, accepted_ancestor


def write_step_summary(_run_id: int | None, outcome: str) -> None:
    """Concise workflow job summary (never a product gate)."""
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not step_summary:
        return
    lines = [
        "## AA scenario2 diagnostic reconcile",
        f"- run: `{_run_id if _run_id is not None else '(none selected)'}`",
        f"- typed outcome: `{outcome}`",
        "- GitHub green is never treated as product success; the typed verdict "
        "comes from AA-owned artifact evidence.",
    ]
    try:
        with Path(step_summary).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _reconcile_selected_run(
    args: argparse.Namespace,
    run: triage.DiagnosticRun,
    summary: dict[str, Any],
) -> triage.TriageOutcome:
    """Evaluate + publish triage for one completed diagnostic run."""
    artifact_name = triage.artifact_name_for_run(run.run_id)
    if args.artifact_file:
        outcome, timeline, gaps, crossings, banner = evaluate_artifact(
            Path(args.artifact_file), run
        )
        summary["artifact"] = {"source": "local", "name": artifact_name}
    else:
        artifact_id, expired = collect_artifact(args.owner, args.repo, run.run_id)
        if artifact_id is not None and not expired:
            outcome, timeline, gaps, crossings, banner = evaluate_artifact(
                download_artifact(args.owner, args.repo, artifact_id), run
            )
            summary["artifact"] = {"source": "downloaded", "name": artifact_name}
        else:
            outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_MISSING)
            timeline = []
            crossings = ()
            banner = None
            gaps = [
                f"AA-owned artifact `{artifact_name}` "
                + ("expired (retention window)" if expired else "absent")
                + ": diagnostic evidence cannot be re-read for this run"
            ]
            summary["artifact"] = {
                "source": "expired" if expired else "absent",
                "name": artifact_name,
            }
    if crossings:
        gaps = [
            *gaps,
            "observed slow-but-completing work (recorded, never judged a hang)",
        ]
    if banner is not None:
        gaps.append(f"diagnostic self-report banner: {banner}")

    fingerprint_key = triage.fingerprint_key_for_triage(outcome, {})
    repair_plan = triage.build_repair_issue_plan(
        triage=outcome,
        run=run,
        artifact_name=artifact_name,
        fingerprint_key=fingerprint_key,
        existing_fingerprints=triage.parse_issue_fingerprints(_open_bodies(args)),
    )
    comment_body = triage.build_triage_comment(
        run=run,
        artifact_name=artifact_name,
        triage=outcome,
        timeline_lines=triage.render_timeline(timeline),
        crossings=crossings,
        evidence_gaps=gaps,
        repair_issue_note=repair_plan.reason if repair_plan.should_create else "",
    )
    comment_pairs = _comment_input(args)
    action = triage.decide_triage_comment(comment_pairs, sha=run.head_sha, run_id=run.run_id)
    summary["comment"] = {"action": action.kind, "comment_id": action.comment_id}
    if args.post:
        summary["posted"] = post_triage_comment(
            args.owner, args.repo, args.triage_issue, action, comment_body
        )
        if repair_plan.should_create:
            number = create_repair_issue(args.owner, args.repo, repair_plan)
            summary["repair_issue"] = {
                "created": number,
                "fingerprint": repair_plan.fingerprint,
                "labels": list(repair_plan.labels),
            }
        else:
            summary["repair_issue"] = {"created": None, "reason": repair_plan.reason}
    return outcome


def _comment_input(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.comments_file:
        return _load_comment_pairs(args.comments_file)
    return collect_issue_comments(args.owner, args.repo, args.triage_issue)


def _open_bodies(args: argparse.Namespace) -> list[str]:
    if args.open_issues_file:
        payload = json.loads(Path(args.open_issues_file).read_text(encoding="utf-8"))
        return [
            f"{item.get('title', '')}\n{item.get('body', '')}"
            for item in payload
            if isinstance(item, dict)
        ]
    return collect_open_issue_bodies(args.owner, args.repo)


def _load_comment_pairs(path: str) -> list[tuple[int, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("comments file must be a JSON list")
    pairs: list[tuple[int, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            pairs.append((int(item["id"]), str(item.get("body", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return pairs


def run(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = {}
    now_epoch: float = args.now_epoch if args.now_epoch > 0 else time.time()
    target_sha = (
        triage.normalize_sha(args.main_sha)
        if args.main_sha
        else collect_main_sha(args.owner, args.repo)
    )
    allowed = triage.normalize_allowed_shas((target_sha, *args.event_sha))
    runs = (
        triage.parse_runs(json.loads(Path(args.runs_file).read_text(encoding="utf-8")))
        if args.runs_file
        else collect_runs(args.owner, args.repo)
    )
    summary["target_sha"] = target_sha
    summary["main_sha"] = target_sha

    selection, accepted_ancestor = select_with_ancestor_backfill(
        runs,
        owner=args.owner,
        repo=args.repo,
        allowed=allowed,
        target_sha=target_sha,
        resolve_ancestor=not args.artifact_file,
    )
    if accepted_ancestor:
        summary["accepted_ancestor"] = accepted_ancestor
    summary["selection"] = {"action": selection.action, "reason": selection.reason}

    if selection.action == "wait":
        summary["phase"] = "wait"
        summary["run_id"] = selection.run.run_id if selection.run else None
        write_step_summary(selection.run.run_id if selection.run else None, "WAIT")
        print(json.dumps(summary, sort_keys=True))
        return EXIT_OK

    if selection.action == "missing":
        markers = triage.parse_dispatch_markers([body for _, body in _comment_input(args)])
        plan = triage.decide_dispatch(
            runs,
            target_sha=target_sha,
            markers=markers,
            now_epoch=now_epoch,
        )
        summary["dispatch"] = {
            "should_dispatch": plan.should_dispatch,
            "attempt": plan.attempt,
            "reason": plan.reason,
        }
        if plan.should_dispatch and args.post:
            post_dispatch_marker(args.owner, args.repo, args.triage_issue, plan.marker_body)
            dispatch_diagnostic(args.owner, args.repo, target_sha)
            summary["dispatched"] = True
        summary["phase"] = "dispatched" if plan.should_dispatch else "missing-run"
        write_step_summary(None, triage.OUTCOME_RUN_MISSING)
        print(json.dumps(summary, sort_keys=True))
        return EXIT_OK

    run = selection.run
    if run is None:  # type narrowing; selection guarantees a run here
        raise ValueError("selection resolved to reconcile without a selected run")
    summary["run_id"] = run.run_id
    summary["run_url"] = run.html_url
    outcome = _reconcile_selected_run(args, run, summary)
    summary["outcome"] = outcome.outcome
    summary["blocked_code"] = outcome.blocked_code
    summary["phase"] = "reconciled"
    write_step_summary(run.run_id, outcome.outcome)
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AA-owned post-run triage and reconciliation of the scenario #2 "
            "diagnostic (kodmial/aa#425)."
        )
    )
    parser.add_argument("--owner", default="kodmial")
    parser.add_argument("--repo", default="aa")
    parser.add_argument("--triage-issue", type=int, default=triage.TRIAGE_ISSUE_NUMBER)
    parser.add_argument("--main-sha", default="", help="exact target SHA (default: main)")
    parser.add_argument("--event-sha", action="append", default=[])
    parser.add_argument("--now-epoch", type=float, default=0.0)
    parser.add_argument("--runs-file", default="")
    parser.add_argument("--comments-file", default="")
    parser.add_argument("--open-issues-file", default="")
    parser.add_argument("--artifact-file", default="")
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except GhError as exc:
        message = str(exc)
        print(json.dumps({"error": message}))
        permission_missing = any(marker in message.lower() for marker in MISSING_PERMISSION_MARKERS)
        if permission_missing:
            print(
                "missing-permission: github.token needs actions:read, "
                "issues:write and contents:read; failing instead of skipping"
            )
        return EXIT_GITHUB_ERROR
    except (ValueError, zipfile.BadZipFile, KeyError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
