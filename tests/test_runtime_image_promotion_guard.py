"""Runtime-image promotion contract guard tests (kodmial/aa#441).

Covers the Gate C startup restoration regression class:

(a) the old, canary-retired digest must fail the consumer guard;
(b) on the real tree every consumer (runtime, A/B, Gate C live,
    derived-index, bootstrap metrics, container Git trust regression)
    pins the single promoted digest from docker/aa-runtime.digest;
(c) full-mode A/B evidence restore stays wired, C-only (gate-c-focus)
    never downloads the skipped `aa-preflight-evidence` artifact;
(d) full-mode trusted restore still downloads the A/B evidence and keeps
    Gate F as its fail-closed SHA/run/checksum validator;
(e) checkpoint restore requires `github.run_attempt > 1` for either A/B
    result because of explicit parentheses (the unparenthesized legacy
    condition is reproduced and shown to resurface the attempt-1 bug);
(f) missing preflight tooling (`pytest: command not found` class) is
    classified as an infrastructure BLOCK, never a forged PASS/FAIL.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PROMOTE_SCRIPT = ROOT / "scripts" / "promote_runtime_image.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_runtime_image.py"
DIGEST_FILE = ROOT / "docker" / "aa-runtime.digest"
QUALIFICATION = WORKFLOWS / "aa-self-proving-qualification.yml"

# Evidence from the incident (kodmial/aa#441): the promotion consumers
# silently kept this retired digest while docker/aa-runtime.digest moved.
RETIRED_DIGEST = (
    "ghcr.io/kodmial/aa-runtime@"
    "sha256:8d49898823e712076fe05bee961faa8812257306357ba0c2946058933740d409"
)
PROMOTED_DIGEST = DIGEST_FILE.read_text(encoding="utf-8").strip()
CONSUMER_DECL_RE = re.compile(r'workflows"\s*/\s*"([A-Za-z0-9_.-]+\.yml)"')
EXPECTED_CONSUMERS = frozenset(
    {
        "aa-self-proving-qualification.yml",
        "aa-derived-index.yml",
        "aa-bootstrap-metrics.yml",
        "aa-container-git-trust-regression.yml",
    }
)


def _run_guard(root: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_runtime_image_consumers.py",
            "--repo-root",
            str(root),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {}
    assert isinstance(payload, dict)
    return proc, payload


def _guard_hidden_tree(tmp_path: Path) -> Path:
    """Minimal repository copy with the promotion-relevant files."""
    hidden = tmp_path / "repo"
    (hidden / "docker").mkdir(parents=True)
    (hidden / "scripts").mkdir(parents=True)
    (hidden / ".github" / "workflows").mkdir(parents=True)
    shutil.copyfile(DIGEST_FILE, hidden / "docker" / "aa-runtime.digest")
    shutil.copyfile(PROMOTE_SCRIPT, hidden / "scripts" / "promote_runtime_image.py")
    for workflow in WORKFLOWS.glob("*.yml"):
        shutil.copyfile(workflow, hidden / ".github" / "workflows" / workflow.name)
    return hidden


_JOB_RE = re.compile(r"  [A-Za-z0-9_-]+:(\s*#.*)?$")
_STEP_IF_RE = re.compile(r"^        if: (.+)$")


def _job_block_lines(lines: list[str], job_name: str) -> list[str]:
    job_start = next(i for i, line in enumerate(lines) if line == f"  {job_name}:")
    try:
        # The qualifying `qualify-live` job is the last job in the file;
        # a missing following job key means the block runs to the end.
        job_end = next(i for i in range(job_start + 1, len(lines)) if _JOB_RE.fullmatch(lines[i]))
    except StopIteration:
        job_end = len(lines)
    return lines[job_start:job_end]


def _step_if(job_body: list[str], step_name: str) -> str:
    step_start = next(
        i for i, line in enumerate(job_body) if line.strip() == f"- name: {step_name}"
    )
    for index in range(step_start + 1, len(job_body)):
        line = job_body[index]
        if line.lstrip().startswith("- name:"):
            break
        condition = _STEP_IF_RE.match(line)
        if condition:
            return condition.group(1).strip()
    raise AssertionError(f"step {step_name!r} has no if: condition")


def _step_condition(step_name: str, job_name: str = "qualify-live") -> str:
    """The `if:` value of one named step in the qualification workflow.

    The restore conditions under test live in the `qualify-live` job; the
    `qualify` job has independent same-named steps with their own (already
    correct, unrelated) conditions, so the search is job-scoped. The
    workflow is scanned line-based instead of through a YAML library so
    the repository does not need an extra typing stub just for these
    contract tests.
    """
    lines = QUALIFICATION.read_text(encoding="utf-8").splitlines()
    try:
        return _step_if(_job_block_lines(lines, job_name), step_name)
    except (StopIteration, AssertionError) as exc:
        raise AssertionError(f"step {step_name!r} not found in job {job_name!r}: {exc}") from None


def _eval_condition(condition: str, qualify_result: str, run_attempt: int) -> bool:
    """Evaluate a repository-owned GH condition template (no traversal)."""
    expression = condition.strip()
    assert expression.startswith("${{")
    assert expression.endswith("}}")
    py = expression[3:-2].strip()
    py = py.replace("&&", " and ").replace("||", " or ")
    py = py.replace("github.run_attempt", "RUN_ATTEMPT")
    py = re.sub(r"needs\.qualify\.result", "QUALIFY_RESULT", py)
    return bool(
        eval(  # noqa: S307 - template text comes from this repository only
            py,
            {"__builtins__": {}},
            {"QUALIFY_RESULT": qualify_result, "RUN_ATTEMPT": run_attempt},
        )
    )


# (b) real-tree promotion contract ------------------------------------------------


def test_guard_passes_on_current_tree_and_covers_named_consumers() -> None:
    proc, payload = _run_guard(ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert payload["status"] == "pass"
    assert payload["promoted"] == PROMOTED_DIGEST
    assert set(payload["consumers"]) == set(EXPECTED_CONSUMERS)
    # 2 qualification jobs + derived-index publisher + bootstrap metrics +
    # 3 container Git trust references + the runtime job itself.
    assert payload["occurrences"] == 8
    assert payload["errors"] == []


def test_qualification_jobs_pin_the_promoted_digest() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    # Both container images in the qualification workflow (the `qualify`
    # A/B preflight job and the `qualify-live` Gate C job) are the exact
    # promoted digest; neither may retain the retired digest.
    assert text.count(f"\n      image: {PROMOTED_DIGEST}\n") == 2
    assert RETIRED_DIGEST not in text


def test_promotion_script_declares_and_retains_every_consumer() -> None:
    declared = set(CONSUMER_DECL_RE.findall(PROMOTE_SCRIPT.read_text(encoding="utf-8")))
    declared.discard("aa-runtime.yml")
    assert declared == set(EXPECTED_CONSUMERS)
    for consumer in EXPECTED_CONSUMERS:
        consumer_text = (WORKFLOWS / consumer).read_text(encoding="utf-8")
        assert PROMOTED_DIGEST in consumer_text, consumer
        assert RETIRED_DIGEST not in consumer_text, consumer


def test_activation_pr_stages_every_updated_runtime_consumer() -> None:
    release = (WORKFLOWS / "aa-runtime-image.yml").read_text(encoding="utf-8")
    assert release.count("activation_paths=(") == 1
    staged = {
        line.strip()
        for line in release.split("activation_paths=(", 1)[1]
        .split("\n          )", 1)[0]
        .splitlines()
        if line.strip()
    }
    expected = {"docker/aa-runtime.digest", ".github/workflows/aa-runtime.yml"}
    expected.update(f".github/workflows/{name}" for name in EXPECTED_CONSUMERS)
    assert staged == expected
    assert 'if git diff --quiet -- "${activation_paths[@]}"; then' in release
    assert 'git add -- "${activation_paths[@]}"' in release


# (a) stale consumer regression must fail the guard -------------------------------


def test_guard_fails_on_retired_consumer_digest(tmp_path: Path) -> None:
    hidden = _guard_hidden_tree(tmp_path)
    stale = hidden / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    stale.write_text(
        stale.read_text(encoding="utf-8").replace(PROMOTED_DIGEST, RETIRED_DIGEST),
        encoding="utf-8",
    )
    proc, payload = _run_guard(hidden)
    assert proc.returncode != 0
    assert payload["status"] == "fail"
    rendered = " ".join(str(error) for error in payload["errors"])
    assert "aa-self-proving-qualification.yml" in rendered
    assert RETIRED_DIGEST in rendered
    assert PROMOTED_DIGEST in rendered


def test_guard_fails_when_a_new_consumer_is_unregistered(tmp_path: Path) -> None:
    hidden = _guard_hidden_tree(tmp_path)
    extra = hidden / ".github" / "workflows" / "aa-fake-new-consumer.yml"
    extra.write_text(
        "\n".join(
            [
                "name: Fake new runtime consumer",
                "on: workflow_dispatch",
                "jobs:",
                "  fake:",
                "    runs-on: ubuntu-latest",
                "    container:",
                f"      image: {PROMOTED_DIGEST}",
                "    steps:",
                "      - run: 'noop'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    proc, payload = _run_guard(hidden)
    assert proc.returncode != 0
    rendered = " ".join(str(error) for error in payload["errors"])
    assert "aa-fake-new-consumer.yml" in rendered
    assert "not registered" in rendered


# (c)/(d) mode-aware A/B evidence restoration -------------------------------------


def test_c_only_mode_never_restores_skipped_ab_evidence() -> None:
    restoration = _step_condition(
        "Restore trusted exact-run A/B evidence (never recompute on live runner)"
    )
    assert restoration == (
        "${{ needs.qualify.result == 'success' || needs.qualify.result == 'failure' }}"
    )
    # Full mode: A/B evidence is always restored (a successful preflight
    # feeds the live lanes; a failed one feeds the fail-closed Gate F
    # reduction and stays a FAIL, never a hidden skip).
    assert _eval_condition(restoration, "success", 1) is True
    assert _eval_condition(restoration, "failure", 1) is True
    # Gate-c-focus mode: the preflight job is skipped entirely, so the
    # C-only lane must neither demand nor download the (never published)
    # aa-preflight-evidence artifact.
    assert _eval_condition(restoration, "skipped", 1) is False
    assert _eval_condition(restoration, "skipped", 2) is False


def test_workflow_no_longer_restores_ab_evidence_with_always() -> None:
    workflow = QUALIFICATION.read_text(encoding="utf-8")
    start = workflow.index(
        "Restore trusted exact-run A/B evidence (never recompute on live runner)"
    )
    end = workflow.index("- name: Bootstrap exact production book/index")
    chunk = workflow[start:end]
    assert not re.search(r"^\s*if: always\(\)\s*$", chunk, flags=re.MULTILINE)
    assert "name: aa-preflight-evidence" in chunk
    assert "path: self-proving-out" in chunk
    # The restore remains a bounded download; the trusted fail-closed
    # validation stays with the Gate F SHA/run/checksum reducer (item d).
    assert "continue-on-error: true" in chunk


# (e) checkpoint restore requires run_attempt > 1 ----------------------------------


@pytest.mark.parametrize(
    "step_name",
    [
        "Restore gate checkpoints from previous attempt",
        "Stage restored gate checkpoints",
    ],
)
def test_checkpoint_restore_guards_use_explicit_parentheses(step_name: str) -> None:
    condition = _step_condition(step_name)
    assert condition == (
        "${{ (needs.qualify.result == 'success' || needs.qualify.result == 'skipped') "
        "&& github.run_attempt > 1 }}"
    )
    # Retrying (run_attempt > 1) restores typed checkpoints in both
    # campaign modes; a FRESH attempt never does (item e).
    assert _eval_condition(condition, "success", 1) is False
    assert _eval_condition(condition, "success", 2) is True
    assert _eval_condition(condition, "skipped", 1) is False  # gate-c-focus attempt 1
    assert _eval_condition(condition, "skipped", 2) is True  # gate-c-focus retry
    assert _eval_condition(condition, "skipped", 3) is True


def test_unparenthesized_legacy_condition_resurfaces_attempt_one_bug() -> None:
    """kodmial/aa#441 root cause: && binds tighter than || in GH expressions.

    A `full` run with successful A/B evaluated the legacy condition TRUE on
    attempt 1 and could re-use previous-attempt checkpoints as if they
    belonged to the fresh run. The in-tree parentheses keep attempt 1
    fresh.
    """
    legacy = (
        "needs.qualify.result == 'success' || needs.qualify.result == 'skipped' "
        "&& github.run_attempt > 1"
    )
    assert _eval_condition("${{ " + legacy + " }}", "success", 1) is True
    condition = _step_condition("Restore gate checkpoints from previous attempt")
    fixed = condition.strip().removeprefix("${{").removesuffix("}}").strip()
    assert fixed.startswith("(")
    assert _eval_condition(condition, "success", 1) is False
    assert _eval_condition(condition, "success", 2) is True


def test_workflow_has_no_unparenthesized_checkpoint_condition() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    assert not re.search(
        r"needs\.qualify\.result == '(?:success|skipped)' && github\.run_attempt",
        text,
    ), "without parentheses attempt 1 would restore previous-attempt checkpoints"


# (f) missing preflight tooling is an infrastructure BLOCK -------------------------


def test_tools_only_fails_when_console_scripts_are_missing() -> None:
    """Reproduce the exact incident: bare `pytest` absent on PATH."""
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--tools-only"],
        capture_output=True,
        text=True,
        env={"PATH": "", "HOME": str(Path.home())},
        cwd=str(ROOT),
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["tools"] == "fail"
    rendered = " ".join(str(error) for error in payload["errors"])
    # The missing tool is classified per component; there is no silent
    # pass-over. `pass` never appears as a tools status in this path.
    for tool in ("pytest", "ruff", "mypy", "opencode"):
        assert f"console script {tool} not found on PATH" in rendered, tool
    assert '"tools": "pass"' not in proc.stdout


def test_tools_only_passes_in_the_prepared_dev_environment() -> None:
    if shutil.which("opencode") is None:
        pytest.skip("OpenCode CLI is unavailable in this environment")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--tools-only"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["tools"] == "pass"
    assert payload["errors"] == []


def test_qualification_wires_static_and_actual_preflight_distinctly() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    # Both prepared-environment steps run the static declaration check AND
    # the actual binary-availability check against the selected container.
    assert text.count("python scripts/verify_runtime_image.py --pins-only") == 2
    assert text.count("python scripts/verify_runtime_image.py --tools-only") == 2
    assert (
        text.count("--pins-only\n          python scripts/verify_runtime_image.py --tools-only")
        == 2
    )
    # The live lane's preflight step is addressable for the honest
    # infrastructure classification of a missing-tool failure.
    assert "id: preflight" in text
    assert "C_PREFLIGHT_OUTCOME: ${{ steps.preflight.outcome }}" in text


def _terminal_verdict_code() -> str:
    text = QUALIFICATION.read_text(encoding="utf-8")
    start = text.index("Gate C focus - typed terminal verdict")
    end = text.index("- name: Gate E - live performance/SLO")
    return text[start:end]


def test_missing_preflight_is_classified_infrastructure_not_product() -> None:
    code = _terminal_verdict_code()
    assert "C_PREFLIGHT_OUTCOME" in code
    assert "preflight tooling unavailable" in code
    assert "infrastructure, not a product verdict" in code
    # No missing/unreadable c-focus artifact can ever become a PASS:
    # both absence and corruption exit 2 (BLOCKED), never 0.
    assert "no typed c-focus-result.json artifact" in code
    assert "sys.exit(2)" in code


def test_live_lane_actually_invokes_gate_c_only_runner() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    assert (
        "python scripts/run_product_contract_live_qualification.py \\\n"
        '            --main-sha "${{ steps.sha.outputs.sha }}" \\\n'
        '            --out-dir live-out --run-id "${{ github.run_id }}" --gate-c-only'
    ) in text


def test_focus_publisher_never_fabricates_a_missing_verdict() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    publisher = text[text.index("Publish Gate C focus outcome") :]
    assert "No typed Gate C focus outcome available" in publisher
    assert "Gate C focus verdict is not typed" in publisher
