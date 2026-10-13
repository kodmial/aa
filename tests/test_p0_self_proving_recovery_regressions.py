"""Regressions for AA self-proving restart/repair and Gate B proof drift.

Uses invented Russian text only: no corpus, user message or secret is exposed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aa.conversation.output_limits import compact_text_to_envelope, envelope_passes

ROOT = Path(__file__).resolve().parents[1]

PUBLISHER_START = "- name: Publish machine-readable failure and repair marker"
PUBLISHER_END = "- name: Publish authoritative Product Contract result"
SUMMARY_READ = "const summary = JSON.parse("


def _publisher_block() -> str:
    workflow = (ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    return workflow.split(PUBLISHER_START, 1)[1].split(PUBLISHER_END, 1)[0]


def _extract_publisher_script() -> str:
    block = _publisher_block()
    script = block.split("script: |", 1)[1]
    end = script.find("\n      - name:")
    if end != -1:
        script = script[:end]
    lines = script.splitlines()
    dedented: list[str] = []
    for line in lines:
        if not line.strip():
            dedented.append("")
        elif line.startswith("            "):
            dedented.append(line[12:])
        else:
            dedented.append(line.lstrip())
    return "\n".join(dedented).strip() + "\n"


def test_gate_b_output_envelope_proves_both_safe_compaction_and_refusal() -> None:
    # Many COMPLETE short units may be compacted to a full-unit prefix.
    complete_units = "Слово. " * 400
    assert not envelope_passes(complete_units)
    compacted = compact_text_to_envelope(complete_units)
    assert envelope_passes(compacted)
    assert compacted and compacted.endswith(".")
    assert compacted.count("Слово.") < 400

    # One indivisible over-limit unit is never cut into a misleading claim.
    with pytest.raises(ValueError, match="no complete unit"):
        compact_text_to_envelope("Слово " * 400)

    # The actual Gate B check must test valid units and unsafe ones separately.
    code = (ROOT / "scripts" / "run_self_proving_qualification.py").read_text(encoding="utf-8")
    section = code.split("# -- output envelope --", 1)[1].split(
        "# -- memory/FIFO/concurrency/safety --", 1
    )[0]
    assert 'long_text = "Слово. " * 400' in section
    assert 'compact_text_to_envelope("Слово " * 400)' in section


def test_qualification_does_not_preempt_same_sha_or_forge_missing_gate_a() -> None:
    workflow = (ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    expected_group = "group: aa-self-proving-qualification-${{ inputs.required_sha || github.sha }}"
    assert expected_group in workflow
    assert "cancel-in-progress: false" in workflow
    gate_f = workflow.split("- name: Gate F - exact-main final verdict", 1)[1].split(
        "- name: Upload per-gate evidence", 1
    )[0]
    assert "steps.gate-a.outcome == 'success'" in gate_f
    assert "steps.gate-a.outcome == 'failure'" in gate_f
    publisher = workflow.split("- name: Publish authoritative Product Contract result", 1)[1]
    assert "steps.gate-a.outcome == 'success'" in publisher
    assert "steps.gate-a.outcome == 'failure'" in publisher


def test_qualification_repair_publisher_keeps_failure_set_in_shared_scope() -> None:
    publisher = _publisher_block()
    before_parser, inside_parser = publisher.split(SUMMARY_READ, 1)
    assert "let failGates = new Set();" in before_parser
    assert "failGates = new Set(productFailures.map(" in inside_parser
    assert "const failGates =" not in inside_parser
    # Typed Gate F details must also survive the parser try block so the
    # repair issue body can be built after the catch without ReferenceError.
    assert "let typedFailedGates = [];" in before_parser
    assert "let typedRoots = [];" in before_parser
    assert "const typedFailedGates" not in publisher
    assert "const typedRoots" not in publisher
    assert "typedFailedGates = Array.isArray(summary.failed_gates)" in inside_parser
    assert "typedRoots = Array.isArray(summary.root_components)" in inside_parser
    # After the parser fallback, missing typed details degrade to the single
    # blocking gate/component instead of crashing or reporting phantom gates.
    after_catch = publisher.split("using generic failure category", 1)[1]
    assert "typedFailedGates.length ? typedFailedGates : [blocking]" in after_catch
    assert "typedRoots.length ? typedRoots : [component]" in after_catch
    assert "failGates.has('C')" in publisher
    assert "steps.gate-a.outcome == 'success'" in publisher
    assert "steps.gate-a.outcome == 'failure'" in publisher
    assert "github.rest.repos.getBranch({" in publisher
    assert "is stale" in publisher
    assert "no repair issue is created" in publisher


def _run_publisher_scenario(
    tmp_path: Path,
    *,
    summary_text: str | None,
    gate_outcomes: dict[str, str],
    failed_sha: str,
    initial_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the repair publisher regression")
    script = _extract_publisher_script()
    work = tmp_path / "publisher-work"
    if work.exists():
        shutil.rmtree(work)
    out_dir = work / "self-proving-out"
    out_dir.mkdir(parents=True)
    if summary_text is not None:
        (out_dir / "self-proving-summary.json").write_text(summary_text, encoding="utf-8")
    issues_file = work / "issues.json"
    issues_file.write_text(json.dumps(initial_issues), encoding="utf-8")
    comments_file = work / "comments.json"
    comments_file.write_text(json.dumps([]), encoding="utf-8")
    result_file = work / "result.json"
    harness = work / "harness.js"
    prelude = (
        "const __fs = require('fs');\n"
        "const core = { warnings: [], notices: [], infos: [], failures: [],\n"
        "  warning(m) { this.warnings.push(String(m)); },\n"
        "  notice(m) { this.notices.push(String(m)); },\n"
        "  info(m) { this.infos.push(String(m)); },\n"
        "  setFailed(m) { this.failures.push(String(m)); } };\n"
        "const context = { repo: { owner: 'o', repo: 'r' }, "
        f"sha: '{failed_sha}', runId: 424242 }};\n"
        "const __issues = JSON.parse(__fs.readFileSync("
        f"'{issues_file.as_posix()}', 'utf8'));\n"
        "const __calls = { created: [], updated: [], dispatched: 0 };\n"
        "function __listForRepo() {}\n"
        "function __listComments() {}\n"
        "const github = { rest: { repos: { getBranch: async () => "
        "({ data: { commit: { sha: process.env.FAILED_SHA } } }) },\n"
        "  issues: { listForRepo: __listForRepo, listComments: __listComments,\n"
        "    create: async (args) => { const n = 2000 + __calls.created.length;\n"
        "      __calls.created.push(args);\n"
        "      __issues.push({ number: n, state: 'open', body: args.body, title: args.title,\n"
        "        labels: (args.labels || []).map((name) => ({ name })) });\n"
        "      return { data: { number: n } }; },\n"
        "    update: async (args) => { __calls.updated.push(args);\n"
        "      const found = __issues.find((i) => i.number === args.issue_number);\n"
        "      if (found) { if (args.body) found.body = args.body;\n"
        "        if (args.title) found.title = args.title;\n"
        "        if (args.state) found.state = args.state;\n"
        "        if (args.labels) found.labels = args.labels.map((name) => ({ name })); }\n"
        "      return { data: {} }; } },\n"
        "  actions: { createWorkflowDispatch: async () => "
        "{ __calls.dispatched += 1; return {}; } } },\n"
        "  paginate: async (fn) => { if (fn === __listForRepo) return __issues;\n"
        "    if (fn === __listComments) return []; return []; } };\n"
        "(async () => {\n"
        "async function __publisher() {\n"
    )
    epilogue = (
        "\n}\nawait __publisher();\n"
        f"__fs.writeFileSync('{result_file.as_posix()}', JSON.stringify("
        "{ created: __calls.created, updated: __calls.updated, "
        "dispatched: __calls.dispatched, issues: __issues, "
        "warnings: core.warnings, failures: core.failures }));\n"
        "})();\n"
    )
    harness.write_text(prelude + script + epilogue, encoding="utf-8")
    env = dict(gate_outcomes)
    env.update(
        {
            "FAILED_SHA": failed_sha,
            "MOCK_RESULT": result_file.as_posix(),
        }
    )
    proc = subprocess.run(
        [node, harness.as_posix()],
        capture_output=True,
        text=True,
        cwd=str(work),
        env={**os.environ, **env},
        check=False,
    )
    assert "ReferenceError" not in proc.stderr, f"publisher crashed: {proc.stderr}"
    assert "ReferenceError" not in proc.stdout, f"publisher crashed: {proc.stdout}"
    assert proc.returncode == 0, f"publisher failed: {proc.stdout}\n{proc.stderr}"
    assert result_file.exists(), f"harness did not report: {proc.stdout}\n{proc.stderr}"
    payload: dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
    return payload


def _synthetic_summary(
    gates: list[dict[str, Any]],
    failed_gates: list[str],
    roots: list[str],
    blocking: str,
) -> str:
    return json.dumps(
        {
            "gates": gates,
            "failed_gates": failed_gates,
            "root_components": roots,
            "blocking_gate": blocking,
        }
    )


def test_qualification_repair_publisher_recovers_failed_gate_a_without_reference_error(
    tmp_path: Path,
) -> None:
    failed_sha = "a" * 40
    base_outcomes = {
        "GATE_A_OUTCOME": "failure",
        "GATE_B_OUTCOME": "skipped",
        "GATE_C_OUTCOME": "skipped",
        "GATE_D_OUTCOME": "skipped",
        "GATE_E_OUTCOME": "skipped",
    }

    # Missing summary: the parser try block fails, so the publisher must fall
    # back to the generic unknown failure without ReferenceError, phantom
    # B/C/D/E product failures, or spurious SUCCESS.
    missing = _run_publisher_scenario(
        tmp_path,
        summary_text=None,
        gate_outcomes=dict(base_outcomes),
        failed_sha=failed_sha,
        initial_issues=[],
    )
    assert len(missing["created"]) == 1
    body = str(missing["created"][0]["body"])
    assert "unknown-gate-failure" in body
    assert "Failed gates (typed): `unknown`" in body
    assert "Repair fingerprint:" in body
    for phantom in ("B:", "C:", "D:", "E:"):
        assert f"All proven product failures: `{phantom}" not in body

    # Invalid summary: same generic fallback, still no ReferenceError.
    invalid = _run_publisher_scenario(
        tmp_path,
        summary_text="not-json{",
        gate_outcomes=dict(base_outcomes),
        failed_sha=failed_sha,
        initial_issues=[],
    )
    assert len(invalid["created"]) == 1
    invalid_body = str(invalid["created"][0]["body"])
    assert "unknown-gate-failure" in invalid_body
    assert "Failed gates (typed): `unknown`" in invalid_body

    # One failing Gate A summary with typed details recovers the publisher and
    # reports exactly the typed A failure without phantom gates.
    gate_a_summary = _synthetic_summary(
        [
            {
                "gate": "A",
                "status": "FAIL",
                "failure_category": "synthetic-static-format",
                "component": "synthetic-static-build",
                "latency_p50_ms": 0,
                "latency_p95_ms": 0,
                "max_turn_ms": 0,
            }
        ],
        ["A"],
        ["A:synthetic-static-format:synthetic-static-build"],
        "A",
    )
    single = _run_publisher_scenario(
        tmp_path,
        summary_text=gate_a_summary,
        gate_outcomes=dict(base_outcomes),
        failed_sha=failed_sha,
        initial_issues=[],
    )
    assert len(single["created"]) == 1
    single_body = str(single["created"][0]["body"])
    assert "Failed gates (typed): `A`" in single_body
    assert "A:synthetic-static-format:synthetic-static-build" in single_body
    assert "B:" not in single_body.split("Failed gates (typed):")[1].split("`")[1]

    # Independent failures A and C stay independent: both are proven, no
    # phantom B/D/E product failures and no spurious SUCCESS.
    both_summary = _synthetic_summary(
        [
            {
                "gate": "A",
                "status": "FAIL",
                "failure_category": "synthetic-static-format",
                "component": "synthetic-static-build",
                "latency_p50_ms": 0,
                "latency_p95_ms": 0,
                "max_turn_ms": 0,
            },
            {
                "gate": "C",
                "status": "FAIL",
                "failure_category": "synthetic-live-path",
                "component": "synthetic-answer",
                "latency_p50_ms": 0,
                "latency_p95_ms": 0,
                "max_turn_ms": 0,
            },
        ],
        ["A", "C"],
        [
            "A:synthetic-static-format:synthetic-static-build",
            "C:synthetic-live-path:synthetic-answer",
        ],
        "A",
    )
    both_outcomes = dict(base_outcomes)
    both_outcomes["GATE_C_OUTCOME"] = "failure"
    both = _run_publisher_scenario(
        tmp_path,
        summary_text=both_summary,
        gate_outcomes=both_outcomes,
        failed_sha=failed_sha,
        initial_issues=[],
    )
    assert len(both["created"]) == 1
    both_body = str(both["created"][0]["body"])
    assert "Failed gates (typed): `A, C`" in both_body
    assert "C:synthetic-live-path:synthetic-answer" in both_body
    proven = both_body.split("All proven product failures: `")[1].split("`")[0]
    assert "A:" in proven and "C:" in proven
    assert "B:" not in proven and "D:" not in proven and "E:" not in proven

    # Idempotence: repeating the same same-SHA failure reuses the single P0
    # repair issue instead of minting a second one.
    existing_issues = both["issues"]
    assert len(existing_issues) == 1
    repeat = _run_publisher_scenario(
        tmp_path,
        summary_text=both_summary,
        gate_outcomes=both_outcomes,
        failed_sha=failed_sha,
        initial_issues=existing_issues,
    )
    assert len(repeat["created"]) == 0
    assert len(repeat["updated"]) == 1
    assert len(repeat["issues"]) == 1


def test_external_telegram_probe_is_not_a_qualification_prerequisite() -> None:
    """Qualification must never require a second bot or poller lease."""
    wf = (ROOT / ".github/workflows/aa-self-proving-qualification.yml").read_text(encoding="utf-8")
    assert "- name: Gate D - real Telegram" not in wf
    assert "AA_TEST_TELEGRAM_BOT_TOKEN" not in wf
    assert "AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE" not in wf
    assert "EXTERNAL_TEST_BOT_UNAVAILABLE" not in wf
    assert "- name: Gate C - LIVE exact production conversation path" in wf
    assert "- name: Gate E - live performance/SLO" in wf


def _inline_github_script(block: str) -> str:
    import textwrap

    raw = block.split("script: |", 1)[1]
    return textwrap.dedent(raw).strip() + "\n"


def test_self_proving_recovery_is_bounded_and_javascript_parses(tmp_path: Path) -> None:
    """Production incident: 500 on rerun after 240s caused a frozen six-attempt series."""
    wf = (ROOT / ".github/workflows/aa-self-proving-429-recovery.yml").read_text(encoding="utf-8")
    assert '"startup_failure"' in wf
    assert "externalFound" in wf
    assert "if (attempt >= 3)" in wf
    assert "const delaySeconds = Math.min(15" in wf
    assert "getWorkflowRun" in wf
    assert "aa-startup-failure-recovery sha=" in wf
    assert "attempt < 10" not in wf
    script = _inline_github_script(wf)
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to verify github-script syntax")
    wrapped = "async function run() {\n" + script + "\n}\n"
    check = subprocess.run(
        [node, "--check"], input=wrapped, text=True, capture_output=True, check=False
    )
    assert check.returncode == 0, check.stderr


def test_lifecycle_does_not_wait_for_new_test_bot_credential() -> None:
    wf = (ROOT / ".github/workflows/aa-mandatory-qualification-lifecycle.yml").read_text(
        encoding="utf-8"
    )
    assert "Reconcile newly authorized isolated test bot" not in wf
    assert "AA_TEST_TELEGRAM_BOT_TOKEN" not in wf
