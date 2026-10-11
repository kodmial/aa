"""Regressions for AA self-proving restart/repair and Gate B proof drift.

Uses invented Russian text only: no corpus, user message or secret is exposed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aa.conversation.output_limits import compact_text_to_envelope, envelope_passes

ROOT = Path(__file__).resolve().parents[1]


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
    code = (ROOT / "scripts" / "run_self_proving_qualification.py").read_text(
        encoding="utf-8"
    )
    section = code.split("# -- output envelope --", 1)[1].split(
        "# -- memory/FIFO/concurrency/safety --", 1
    )[0]
    assert 'long_text = "Слово. " * 400' in section
    assert 'compact_text_to_envelope("Слово " * 400)' in section


def test_qualification_does_not_preempt_same_sha_or_forge_missing_gate_a() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
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
    workflow = (
        ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    start = "- name: Publish machine-readable failure and repair marker"
    end = "- name: Publish authoritative Product Contract result"
    publisher = workflow.split(start, 1)[1].split(end, 1)[0]
    before_parser, inside_parser = publisher.split(
        "const summary = JSON.parse(", 1
    )
    assert "let failGates = new Set();" in before_parser
    assert "failGates = new Set(productFailures.map(" in inside_parser
    assert "const failGates =" not in inside_parser
    assert "failGates.has('C')" in publisher
    assert "steps.gate-a.outcome == 'success'" in publisher
    assert "steps.gate-a.outcome == 'failure'" in publisher
    assert "github.rest.repos.getBranch({" in publisher
    assert "is stale" in publisher
    assert "no repair issue is created" in publisher
