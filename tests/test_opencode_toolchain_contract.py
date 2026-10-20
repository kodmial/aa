"""Contract: AA runners and image consume the same, exact OpenCode version.

No workflow may embed a second OpenCode installer or a hardcoded CLI version.
Docker builds use the same version file, since GitHub composite actions cannot
run within a Docker build context.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ACTION = ROOT / ".github" / "actions" / "aa-opencode-toolchain" / "action.yml"
PIN = ROOT / "docker" / "opencode.version"

OPEN_CODE_WORKFLOWS = (
    "ci.yml",
    "aa-gate-c-heldout-bootstrap.yml",
    "aa-self-proving-qualification.yml",
    "aa-canary.yml",
    "aa-real-book-retrieval-qualification.yml",
    "aa-runtime.yml",
    "aa-opencode-model-route-e2e.yml",
)


def test_aa_version_pin_is_exact_semver_and_consistent() -> None:
    value = PIN.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value)
    assert value != "1.18.34", "obsolete OpenCode version must not return"
    docker = (ROOT / "docker/Dockerfile.aa-runtime").read_text(encoding="utf-8")
    assert "COPY opencode.version /opt/aa/opencode.version" in docker
    assert "cat /opt/aa/opencode.version" in docker
    assert 'ARG AA_OPENCODE_VERSION=""' in docker
    assert '${AA_OPENCODE_VERSION:-$(cat /opt/aa/opencode.version)}' in docker
    image = (WORKFLOWS / "aa-runtime-image.yml").read_text(encoding="utf-8")
    assert "OPENCODE_VERSION_OVERRIDE: ${{ vars.CONTINUUM_OPENCODE_VERSION }}" in image
    assert "AA_OPENCODE_VERSION=${{ steps.meta.outputs.opencode }}" in image
    assert "opencode --version" in docker


def test_all_aa_live_runners_share_verified_toolchain() -> None:
    for name in OPEN_CODE_WORKFLOWS:
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "uses: ./.github/actions/aa-opencode-toolchain" in text, name
        assert "https://opencode.ai/install" not in text, name
        assert not re.search(r"bash -s -- --version\s+[0-9]", text), name


def test_aa_toolchain_reads_only_pin_and_verifies_real_binary() -> None:
    text = ACTION.read_text(encoding="utf-8")
    assert 'OPENCODE_VERSION_OVERRIDE: ${{ vars.CONTINUUM_OPENCODE_VERSION }}' in text
    assert 'version="${OPENCODE_VERSION_OVERRIDE:-$(cat docker/opencode.version)}"' in text
    assert '[[ "$version" =~' in text
    assert "kodmial/continuum/.github/actions/agent-toolchain@" in text
    assert "components: opencode" in text
    assert "opencode-version: ${{ steps.version.outputs.version }}" in text
    assert 'actual="$(opencode --version)"' in text
    assert '[[ "$actual" != "$PIN" || "$RESOLVED" != "$PIN" ]]' in text


def test_no_workflow_embeds_an_unreviewed_cli_installer() -> None:
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert "https://opencode.ai/install" not in text, path.name
        assert not re.search(r"bash -s -- --version\s+[0-9]", text), path.name
