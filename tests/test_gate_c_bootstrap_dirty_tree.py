"""Gate C bootstrap checkout-cleanliness regression (kodmial/aa#444).

Runs the exact deterministic bootstrap sequence outside a real runner:
temp Git repo (clean checkout) -> local/fake L2 HIT and L2 MISS/REJECTED
plus L1 miss, with every ``--status-file`` kept in RUNNER_TEMP-equivalent
storage -> committed-public audit -> ``git status --porcelain`` empty ->
``working_tree_clean()`` true. The former pollution (relative status
paths creating untracked JSON in the checkout root) is reproduced, and
plain dirty-tree cases still fail closed. No models, no LLM, no network.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from aa.qualification.product_contract_live import (
    _is_ephemeral_tree_path,
    working_tree_clean,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_corpus_structure import (  # type: ignore[import-not-found]  # noqa: E402
    _build_corpus_structure,
    _write_builder_fixture_inputs,
)
from test_derived_cache import (  # type: ignore[import-not-found]  # noqa: E402
    _build_fixture_index,
)
from test_derived_index_l2 import (  # type: ignore[import-not-found]  # noqa: E402
    _encrypted_fixture,
    _RegistryFixture,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
MANAGE_SCRIPT = _REPO / "scripts" / "manage_derived_cache.py"


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _manager(
    *args: str,
    cwd: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MANAGE_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        check=False,
    )


def _clean_checkout(repo: pathlib.Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True)
    (repo / ".gitignore").write_text("/corpus/generated/\n", encoding="utf-8")
    (repo / "corpus").mkdir(parents=True)
    subprocess.run(
        ["git", "add", ".gitignore", "corpus"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "commit", "-qm", "clean checkout"], cwd=str(repo), check=True)


def _porcelain(repo: pathlib.Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def test_gate_c_bootstrap_sequence_keeps_checkout_clean(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "gate-c-checkout"
    repo.mkdir()
    _clean_checkout(repo)
    inputs = _write_builder_fixture_inputs(tmp_path / "runner-inputs")
    status_dir = tmp_path / "runner-temp"
    status_dir.mkdir()
    l2_status_file = status_dir / "derived-l2-status.json"
    l1_status_file = status_dir / "derived-l1-status.json"

    assert working_tree_clean(repo) is True

    # The committed checkout already carries public corpus artifacts from
    # an explicit development run on main; qualification only audits them.
    seed = _build_corpus_structure(
        inputs=inputs,
        full_output=repo / "corpus" / "generated" / "corpus_structure.json",
        public_output=repo / "corpus" / "structure.json",
        map_output=repo / "corpus" / "book-map.md",
    )
    assert seed.returncode == 0, seed.stderr
    _git(repo, "add", "corpus/structure.json", "corpus/book-map.md")
    _git(repo, "commit", "-qm", "seed public corpus artifacts")
    assert working_tree_clean(repo) is True

    recheck = _build_corpus_structure(
        inputs=inputs,
        full_output=repo / "corpus" / "generated" / "corpus_structure.json",
        public_output=repo / "corpus" / "structure.json",
        map_output=repo / "corpus" / "book-map.md",
        extra_args=["--verify-committed-public"],
    )
    assert recheck.returncode == 0, recheck.stderr
    assert json.loads(recheck.stdout)["committed_public_match"] is True
    assert _porcelain(repo) == ""

    # L2 MISS/REJECTED with a zero placeholder pin: status lands in
    # RUNNER_TEMP-equivalent storage and the checkout stays clean.
    pin = tmp_path / "placeholder.digest"
    pin.write_text("ghcr.io/kodmial/aa-index@sha256:" + "0" * 64 + "\n", encoding="utf-8")
    rejected = _manager(
        "--mode",
        "restore-l2",
        "--cache-dir",
        str(tmp_path / "l2-cache"),
        "--digest-file",
        str(pin),
        "--require-backend",
        "e5",
        "--status-file",
        str(l2_status_file),
        cwd=repo,
    )
    assert rejected.returncode == 0, rejected.stderr
    payload = json.loads(rejected.stdout)
    assert payload["status"] == "rejected"
    assert payload["reason"] == "l2-unpublished"
    assert l2_status_file.is_file()
    assert _porcelain(repo) == ""

    # Local fake-registry L2 HIT (mock, digest-pinned): same tmp status.
    fixture = _RegistryFixture()
    try:
        identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
        from aa.retrieval.derived_cache import ENCRYPTED_NAME, META_NAME
        from aa.retrieval.oci_artifact import expected_l2_tag, push_artifact

        digest = push_artifact(
            fixture.registry,
            files={ENCRYPTED_NAME: encrypted, META_NAME: meta},
            tag=expected_l2_tag(cache_digest),
            session=fixture.session,
        )
        identity_file = tmp_path / "identity.txt"
        identity_file.write_text(identity, encoding="utf-8")
        local_pin = tmp_path / "local-pin.digest"
        local_pin.write_text(
            f"127.0.0.1:{fixture.port}/kodmial/aa-index@{digest}\n", encoding="utf-8"
        )
        hit = _manager(
            "--mode",
            "restore-l2",
            "--cache-dir",
            str(tmp_path / "l2-hit-cache"),
            "--identity-file",
            str(identity_file),
            "--digest-file",
            str(local_pin),
            "--oci-registry",
            f"127.0.0.1:{fixture.port}",
            "--oci-repository",
            "kodmial/aa-index",
            "--oci-scheme",
            "http",
            "--require-backend",
            "hashing",
            "--retrieval-dir",
            str(tmp_path / "retrieval-dir"),
            "--status-file",
            str(l2_status_file),
            cwd=repo,
        )
        assert hit.returncode == 0, hit.stderr
        hit_payload = json.loads(hit.stdout)
        assert hit_payload["status"] == "hit"
        assert hit_payload["source"] == "l2"
        assert hit_payload["l2_digest"] == digest
        assert _porcelain(repo) == ""
    finally:
        fixture.shutdown()

    # L1 exact-restore branch (miss with an empty tmp cache): tmp status.
    miss = _manager(
        "--mode",
        "restore",
        "--cache-dir",
        str(tmp_path / "empty-l1-cache"),
        "--backend",
        "hashing",
        "--require-backend",
        "e5",
        "--status-file",
        str(l1_status_file),
        cwd=repo,
    )
    assert miss.returncode == 0, miss.stderr
    miss_payload = json.loads(miss.stdout)
    assert miss_payload["status"] == "miss"
    assert _porcelain(repo) == ""

    assert working_tree_clean(repo) is True


def test_relative_status_files_reproduce_checkout_pollution(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "legacy-checkout"
    repo.mkdir()
    _clean_checkout(repo)
    pin = tmp_path / "placeholder.digest"
    pin.write_text("ghcr.io/kodmial/aa-index@sha256:" + "0" * 64 + "\n", encoding="utf-8")
    polluted = _manager(
        "--mode",
        "restore-l2",
        "--cache-dir",
        str(tmp_path / "l2-cache"),
        "--digest-file",
        str(pin),
        "--require-backend",
        "e5",
        "--status-file",
        "derived-l2-status.json",
        cwd=repo,
    )
    assert polluted.returncode == 0, polluted.stderr
    assert (repo / "derived-l2-status.json").is_file()
    assert _porcelain(repo).strip() != ""
    assert working_tree_clean(repo) is False
    (repo / "derived-l2-status.json").unlink()
    assert working_tree_clean(repo) is True


def test_p0_444_dirty_checkout_still_fails_closed(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "dirty-checkout"
    repo.mkdir()
    _clean_checkout(repo)
    (repo / "corpus" / "structure.json").write_text('{"tampered": true}\n', encoding="utf-8")
    assert working_tree_clean(repo) is False
    (repo / "corpus" / "structure.json").unlink()
    (repo / "untracked-source.py").write_text("x = 1\n", encoding="utf-8")
    assert working_tree_clean(repo) is False
    assert not _is_ephemeral_tree_path("derived-l1-status.json")
    assert not _is_ephemeral_tree_path("derived-l2-status.json")
    assert not _is_ephemeral_tree_path("derived-publish-status.json")
    (repo / "untracked-source.py").unlink()
    assert working_tree_clean(repo) is True


def test_committed_placeholder_pin_rejected_inside_clean_repo(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "pin-checkout"
    repo.mkdir()
    _clean_checkout(repo)
    committed_pin = _REPO / "docker" / "aa-index.digest"
    assert committed_pin.is_file(), "committed L2 pin file is part of the repository contract"
    proc = _manager(
        "--mode",
        "restore-l2",
        "--cache-dir",
        str(tmp_path / "l2-cache"),
        "--digest-file",
        str(committed_pin),
        "--require-backend",
        "e5",
        "--status-file",
        str(tmp_path / "derived-l2-status.json"),
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "rejected"
    assert payload["reason"] == "l2-unpublished"
    assert _porcelain(repo) == ""


def test_build_fixture_index_support_offline_harness(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    _build_fixture_index(src)
    assert (src / "index.json").is_file()
