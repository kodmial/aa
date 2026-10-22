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
    # The committed pin file itself is allowed to be the placeholder while
    # the artifact is unpublished; consumers never trust it (oci_artifact).
    assert "L2 publication pin is still a placeholder" in pathlib.Path(
        REPO_ROOT / "src" / "aa" / "retrieval" / "oci_artifact.py"
    ).read_text(encoding="utf-8")
    placeholder = pathlib.Path(REPO_ROOT / "docker" / "aa-index.digest").read_text()
    assert re.fullmatch(r"ghcr\.io/kodmial/aa-index@sha256:[0-9a-f]{64}\n?", placeholder)
    assert placeholder.startswith("ghcr.io/kodmial/aa-index@sha256:" + "0" * 64)


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
