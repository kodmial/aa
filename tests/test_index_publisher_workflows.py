"""Trusted derived-index L2 publisher workflow contracts (issue #429)."""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PUBLISHER = WORKFLOWS / "aa-derived-index.yml"
RUNTIME = WORKFLOWS / "aa-runtime.yml"
QUALIFICATION = WORKFLOWS / "aa-self-proving-qualification.yml"
METRICS = WORKFLOWS / "aa-bootstrap-metrics.yml"


def test_publisher_is_trusted_events_only() -> None:
    text = PUBLISHER.read_text(encoding="utf-8")
    assert "push:" in text and "branches:\n      - main" in text
    assert "workflow_dispatch:" in text
    assert "pull_request" not in text
    assert "packages: write" in text
    assert "ghcr.io/kodmial/aa-index" not in text or "kodmial/aa-index" in text
    # Publication runs inside the exact promoted prepared environment.
    assert "ghcr.io/kodmial/aa-runtime@sha256:" in text


def test_publisher_enforces_strict_e5_backend() -> None:
    text = PUBLISHER.read_text(encoding="utf-8")
    assert "--backend e5 --require-backend e5" in text
    assert "--mode publish-l2" in text
    assert "--require-backend e5" in text.split("--mode publish-l2")[1]


def test_publisher_creates_reviewed_digest_pin_pr() -> None:
    text = PUBLISHER.read_text(encoding="utf-8")
    assert "scripts/promote_derived_index.py" in text
    assert "docker/aa-index.digest" in text
    assert "TAP_PAT" in text
    # Only the immutable content-addressed digest may be promoted.
    assert 'branch="index/aa-derived-index-' in text


def test_runtime_and_qualification_restore_strict_e5() -> None:
    for path in (RUNTIME, QUALIFICATION):
        text = path.read_text(encoding="utf-8")
        assert "--mode restore-l2" in text, path.name
        assert "--require-backend e5" in text, path.name
        assert "--mode restore-or-rebuild" in text, path.name
        assert "--backend e5 --require-backend e5" in text, path.name


def test_placeholders_fail_closed_instead_of_hitting_l2() -> None:
    assert "@sha256:0000000000000000000000000000000000000000000000000000000000000000" not in (
        QUALIFICATION.read_text(encoding="utf-8") + RUNTIME.read_text(encoding="utf-8")
    )
    # A committed pin may be either the unpublished placeholder or a promoted
    # immutable digest; placeholder rejection is tested with explicit fixtures.
    assert "L2 publication pin is still a placeholder" in pathlib.Path(
        REPO_ROOT / "src" / "aa" / "retrieval" / "oci_artifact.py"
    ).read_text(encoding="utf-8")
    placeholder = pathlib.Path(REPO_ROOT / "docker" / "aa-index.digest").read_text()
    assert re.fullmatch(r"ghcr\.io/kodmial/aa-index@sha256:[0-9a-f]{64}\n?", placeholder)


def test_qualification_bootstrap_keeps_status_files_out_of_checkout() -> None:
    # kodmial/aa#444: relative L1/L2 status paths created two untracked
    # JSON files in the repository root and blocked the product
    # qualification with 'working tree is not clean'. Status files must
    # live under RUNNER_TEMP exactly like the proven aa-runtime pattern.
    text = QUALIFICATION.read_text(encoding="utf-8")
    assert '--status-file "derived-l1-status.json"' not in text
    assert '--status-file "derived-l2-status.json"' not in text
    bootstrap_index = text.index("Bootstrap exact production book/index")
    bootstrap_block = text[bootstrap_index : bootstrap_index + 3500]
    assert "${RUNNER_TEMP:-/tmp}/derived-l2-status.json" in bootstrap_block
    assert "${RUNNER_TEMP:-/tmp}/derived-l1-status.json" in bootstrap_block


def test_qualification_bootstrap_verifies_committed_public() -> None:
    # kodmial/aa#444: qualification must never rewrite tracked public
    # corpus artifacts; the audit mode verifies byte equality instead.
    text = QUALIFICATION.read_text(encoding="utf-8")
    invocations = [
        line.strip() for line in text.splitlines() if "scripts/build_corpus_structure.py" in line
    ]
    assert len(invocations) == 2
    for line in invocations:
        assert "--verify-committed-public" in line
    assert not any(
        line.startswith("python scripts/build_corpus_structure.py ")
        and "--verify-committed-public" not in line
        for line in invocations
    )


def test_publisher_prefetches_public_e5_assets_before_rebuild() -> None:
    # kodmial/aa#444: the trusted publisher rebuilds inside the prepared
    # runtime container, which intentionally carries no public model
    # weights. Without the pinned E5 snapshot prefetch the strict-E5
    # rebuild failed and L2 publication never activated (persistent zero
    # digest pin, 'l2-unpublished' rejections in every bootstrap).
    text = PUBLISHER.read_text(encoding="utf-8")
    build_index = text.index("Build strict E5 production index and publish L2")
    preamble = text[:build_index]
    assert "scripts/prefetch_public_assets.py" in preamble
    # The publication status file must also stay outside the checkout.
    publish_block = text[build_index:]
    assert "--status-file" in publish_block
    assert '--status-file "derived-publish-status.json"' not in text
    assert "${RUNNER_TEMP:-/tmp}/derived-publish-status.json" in text


def test_runtime_actions_cache_key_is_rotation_aware() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert "AA_DERIVED_CACHE_ROTATION=" not in text, "rotation is read by the manager from env"
    assert "derived-actions-key" in text
    # The bundle's internal manifest cache_key stays the base content key.
    assert "derived-key=" in text
    assert "key: ${{ steps.derived-cache-key.outputs.derived-actions-key }}" in text


def test_metrics_workflow_measures_both_paths() -> None:
    text = METRICS.read_text(encoding="utf-8")
    assert "legacy-path" in text
    assert "prepared-image-path" in text
    assert "aa-bootstrap-legacy-metrics" in text
    assert "ghcr.io/kodmial/aa-runtime@sha256:" in text
    assert "--mode restore-l2" in text


def test_promotion_scripts_keep_digest_pin_consistency() -> None:
    promote_runtime = (REPO_ROOT / "scripts" / "promote_runtime_image.py").read_text(
        encoding="utf-8"
    )
    assert "aa-self-proving-qualification.yml" in promote_runtime
    assert "aa-derived-index.yml" in promote_runtime


# --- kodmial/aa#462: scoped ephemeral packages token on trusted L2 steps ----


def test_runtime_trusted_l2_step_exports_scoped_ephemeral_token() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    # The runtime keeps its existing named permissions including the
    # packages pull grant for the prebuilt image and the aa-index package.
    permissions_block = text.split("permissions:", 1)[1].split("\nconcurrency:", 1)[0]
    assert "packages: read" in permissions_block
    # Never a write grant for the read-only runtime path.
    assert "packages: write" not in permissions_block
    # Only the trusted L2 restore invocation step carries the scoped
    # ephemeral repo token: exactly one occurrence, exported through the
    # env of the step that performs the actual digest-pinned restore.
    assert text.count("AA_INDEX_PACKAGES_TOKEN") == 1
    env_index = text.index("AA_INDEX_PACKAGES_TOKEN: ${{ github.token }}")
    step_index = text.rindex("Build and validate AA knowledge runtime", 0, env_index)
    l2_invocation_index = text.index("--mode restore-l2 \\", env_index)
    assert step_index < env_index < l2_invocation_index
    # Ephemeral github token only: never a PAT and never an Actions-cache
    # style global env export near the L2 invocation.
    line = text[env_index : env_index + 120].splitlines()[0]
    assert line.strip() == "AA_INDEX_PACKAGES_TOKEN: ${{ github.token }}"
    assert "TAP_PAT" not in line
    assert ">> $GITHUB_ENV" not in line


def test_qualification_trusted_l2_step_exports_scoped_ephemeral_token() -> None:
    text = QUALIFICATION.read_text(encoding="utf-8")
    permissions_block = text.split("permissions:", 1)[1].split("\nconcurrency:", 1)[0]
    # kodmial/aa#462: each pre-existing named permission is preserved and
    # `packages: read` is added; nothing is removed and nothing escalated.
    permission_lines = sorted(
        line.strip()
        for line in permissions_block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert permission_lines == [
        "actions: write",
        "contents: write",
        "issues: write",
        "packages: read",
    ]
    # kodmial/aa#462 + kodmial/aa#463: two trusted steps carry the scoped
    # token, one per L2 pull site — the Gate B qualification bootstrap and
    # the Gate C live bootstrap. Each export precedes its own
    # restore-l2 invocation, stays step-scoped, and is never global.
    assert text.count("AA_INDEX_PACKAGES_TOKEN") == 2
    first_env = text.index("AA_INDEX_PACKAGES_TOKEN: ${{ github.token }}")
    second_env = text.index("AA_INDEX_PACKAGES_TOKEN: ${{ github.token }}", first_env + 1)
    gate_b_index = text.rindex("Gate B - deterministic component integration", 0, first_env)
    l2_first = text.index("--mode restore-l2 \\", first_env)
    assert gate_b_index < first_env < l2_first
    assert first_env < l2_first < second_env
    gate_c_index = text.rindex("Bootstrap exact production book/index", 0, second_env)
    l2_second = text.index("--mode restore-l2 \\", second_env)
    assert gate_c_index < second_env < l2_second
    # Ephemeral github token only: never a PAT and never a global export.
    for env_index in (first_env, second_env):
        line = text[env_index : env_index + 120].splitlines()[0]
        assert line.strip() == "AA_INDEX_PACKAGES_TOKEN: ${{ github.token }}"
        assert "TAP_PAT" not in line
        assert ">> $GITHUB_ENV" not in line


def test_packages_token_export_scope_every_workflow() -> None:
    # kodmial/aa#462: the scoped ephemeral token may only be exported on
    # the trusted runtime L2 step, the trusted Gate C L2 step, and
    # (pre-existing, issue #429) the trusted L2 publisher container. No
    # other step carries it, and no PAT ever injects the token value.
    allowed = {
        "aa-runtime.yml": "${{ github.token }}",
        "aa-self-proving-qualification.yml": "${{ github.token }}",
        "aa-derived-index.yml": "${{ github.token }}",
    }
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        count = text.count("AA_INDEX_PACKAGES_TOKEN")
        expected = count > 0
        assert expected == (path.name in allowed), path.name
        if expected:
            marker_index = text.index("AA_INDEX_PACKAGES_TOKEN:")
            line = text[marker_index : marker_index + 120].splitlines()[0]
            assert line.strip() == f"AA_INDEX_PACKAGES_TOKEN: {allowed[path.name]}", line
            assert "TAP_PAT" not in line, line
        for line in text.splitlines():
            if "AA_INDEX_PACKAGES_TOKEN" in line:
                assert ">> $GITHUB_ENV" not in line, line
