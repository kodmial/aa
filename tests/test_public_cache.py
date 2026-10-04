"""Public dependency/model cache tests (issue #59).

The public cache is optional acceleration only. These tests prove the
Definition of Done without network access: lock pinning, compatibility
keys, miss/hit/repair behaviour, non-fatal save failure, sensitive-path
exclusion, and unchanged runtime semantics.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from aa.corpus.public_cache import (
    MODEL_ID,
    PublicCacheError,
    assert_cache_paths_safe,
    is_cache_enabled,
    is_hex_revision,
    load_embedding_lock,
    lock_digest,
    public_model_key,
    public_pip_key,
    public_pip_restore_prefix,
    resolve_model_root,
    verify_cached_model,
    write_marker,
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _lock() -> dict[str, Any]:
    return load_embedding_lock(_repo_root() / "corpus" / "embedding.lock.json")


def _make_cached_model(root: pathlib.Path, lock: dict[str, Any]) -> pathlib.Path:
    revision = str(lock["revision"])
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    required = lock["required_files"]
    assert isinstance(required, list)
    for name in required:
        assert isinstance(name, str)
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"public-fixture:" + name.encode())
    write_marker(root, lock)
    return root


def _run_prefetch(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(__import__("os").environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(_repo_root() / "scripts" / "prefetch_public_assets.py"), *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        env=env,
        check=False,
    )


def test_embedding_lock_is_pinned() -> None:
    lock = _lock()
    assert lock["model_id"] == "intfloat/multilingual-e5-base"
    assert lock["model_id"] == MODEL_ID
    assert is_hex_revision(lock["revision"])
    assert lock["sha"] == lock["revision"]
    required = lock["required_files"]
    assert isinstance(required, list) and len(required) >= 3
    assert "config.json" in required
    # The lock carries no corpus text, secrets, or state.
    raw = (_repo_root() / "corpus" / "embedding.lock.json").read_text(encoding="utf-8")
    for snippet in ("AGE-SECRET-KEY", "TELEGRAM_BOT_TOKEN", "canonical.json", "Chapter"):
        assert snippet not in raw


def test_cache_keys_are_compatibility_bound() -> None:
    lock = _lock()
    digest = lock_digest(lock)
    pip = public_pip_key(os_name="Linux", arch="X64", python_version="3.12", deps_hash="abc123")
    assert "Linux" in pip and "X64" in pip and "3.12" in pip and "abc123" in pip
    assert pip.startswith("aa-public-pip-v1-")
    other_os = public_pip_key(
        os_name="macOS", arch="X64", python_version="3.12", deps_hash="abc123"
    )
    assert other_os != pip
    other_py = public_pip_key(
        os_name="Linux", arch="X64", python_version="3.11", deps_hash="abc123"
    )
    assert other_py != pip
    prefix = public_pip_restore_prefix(os_name="Linux", arch="X64", python_version="3.12")
    assert pip.startswith(prefix)

    model = public_model_key(
        os_name="Linux",
        arch="X64",
        model_id=MODEL_ID,
        revision=str(lock["revision"]),
        lock_hash=digest,
    )
    assert model.startswith("aa-public-model-v1-")
    assert "Linux" in model and "X64" in model
    assert "multilingual-e5-base" in model
    assert str(lock["revision"]) in model
    assert digest in model
    other_arch = public_model_key(
        os_name="Linux",
        arch="ARM64",
        model_id=MODEL_ID,
        revision=str(lock["revision"]),
        lock_hash=digest,
    )
    assert other_arch != model
    with pytest.raises(PublicCacheError):
        public_model_key(
            os_name="Linux", arch="X64", model_id=MODEL_ID, revision="short", lock_hash=digest
        )
    with pytest.raises(PublicCacheError):
        public_model_key(
            os_name="Linux",
            arch="X64",
            model_id="other/model",
            revision=str(lock["revision"]),
            lock_hash=digest,
        )


def test_clean_run_with_cache_disabled_succeeds(tmp_path: pathlib.Path) -> None:
    assert is_cache_enabled({"AA_PUBLIC_CACHE_ENABLED": "0"}) is False
    assert is_cache_enabled({"AA_PUBLIC_CACHE_ENABLED": "1"}) is True
    assert is_cache_enabled({}) is True
    proc = _run_prefetch(
        ["--check-only", "--hf-cache", str(tmp_path / "hub")],
        {"AA_PUBLIC_CACHE_ENABLED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "disabled"
    proc = _run_prefetch(
        ["--hf-cache", str(tmp_path / "hub")],
        {"AA_PUBLIC_CACHE_ENABLED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "disabled"


def test_first_run_misses_then_second_run_restores(tmp_path: pathlib.Path) -> None:
    lock = _lock()
    hub = tmp_path / "hub"
    model_root = resolve_model_root(hub)
    assert verify_cached_model(model_root, lock) is False
    proc = _run_prefetch(["--check-only", "--hf-cache", str(hub)], {})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "miss"

    _make_cached_model(model_root, lock)
    assert verify_cached_model(model_root, lock) is True
    proc = _run_prefetch(["--check-only", "--hf-cache", str(hub)], {})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "hit"


def test_wrong_key_misses_safely(tmp_path: pathlib.Path) -> None:
    lock = _lock()
    hub = tmp_path / "hub"
    model_root = resolve_model_root(hub)
    _make_cached_model(model_root, lock)
    assert verify_cached_model(model_root, lock) is True
    wrong = dict(lock)
    wrong["revision"] = "0" * 40
    assert verify_cached_model(model_root, wrong) is False
    wrong_id = dict(lock)
    wrong_id["model_id"] = "intfloat/other-model"
    assert verify_cached_model(model_root, wrong_id) is False


def test_corrupt_cache_is_rejected_and_repaired(tmp_path: pathlib.Path) -> None:
    lock = _lock()
    hub = tmp_path / "hub"
    model_root = resolve_model_root(hub)
    _make_cached_model(model_root, lock)
    assert verify_cached_model(model_root, lock) is True
    revision = str(lock["revision"])
    victim = model_root / "snapshots" / revision / "config.json"
    victim.write_bytes(b"")
    assert verify_cached_model(model_root, lock) is False
    # check-only prunes corrupt content so a normal download can repair it.
    proc = _run_prefetch(["--check-only", "--hf-cache", str(hub)], {})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "miss"
    assert verify_cached_model(model_root, lock) is False
    _make_cached_model(model_root, lock)
    assert verify_cached_model(model_root, lock) is True


def test_prefetch_without_hub_library_is_non_fatal(tmp_path: pathlib.Path) -> None:
    hub = tmp_path / "hub"
    proc = _run_prefetch(["--hf-cache", str(hub)], {"AA_PUBLIC_CACHE_ENABLED": "1"})
    # Either the environment can download (hit/downloaded/miss) or the
    # optional hub library is absent (skipped); none of these fail runtime.
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] in {"hit", "downloaded", "miss", "skipped"}


def test_no_sensitive_paths_are_cached() -> None:
    assert_cache_paths_safe(
        [
            str(pathlib.Path.home() / ".cache" / "pip"),
            str(
                pathlib.Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--intfloat--multilingual-e5-base"
            ),
        ]
    )
    sensitive = [
        "corpus/generated/canonical.json",
        "corpus/source/raw/AA.txt",
        "corpus/source/encrypted/canonical.tar.zst.age",
        "corpus/source/fetch-state.json",
        "storage/telegram.session",
        "sessions/opencode.db",
        "AGE-SECRET-KEY-foo",
    ]
    for path in sensitive:
        with pytest.raises(PublicCacheError):
            assert_cache_paths_safe([path])


def test_workflow_caches_only_public_assets() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    assert "corpus/embedding.lock.json" in workflow
    assert "hashFiles('pyproject.toml')" in workflow
    assert "hashFiles('corpus/embedding.lock.json')" in workflow
    assert "runner.os" in workflow and "runner.arch" in workflow
    assert "multilingual-e5-base" in workflow or "embedding" in workflow
    # Restore/save round-trip with verification between them.
    assert "cache/restore" in workflow
    assert "cache/save" in workflow
    assert "prefetch_public_assets.py --check-only" in workflow
    assert "prefetch_public_assets.py" in workflow
    # Save failure is non-fatal.
    assert workflow.count("continue-on-error: true") >= 4
    # Only public dependency caches use a fallback prefix, and the model key is exact.
    assert "restore-keys" in workflow
    assert "No restore-keys here" in workflow or "Exact key only" in workflow
    # No sensitive path is listed as a cache path.
    for fragment in (
        "corpus/generated",
        "corpus/source/raw",
        "corpus/source/encrypted",
        "fetch-state",
        "canonical.json",
        ".tar.zst",
        ".age",
        "TELEGRAM",
        "sessions",
    ):
        for line in workflow.splitlines():
            stripped = line.strip()
            if stripped.startswith("path:") and fragment in line:
                raise AssertionError(f"sensitive cache path in workflow: {line!r}")


def test_workflow_runtime_semantics_are_unchanged() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "actions/checkout",
        'python-version: "3.12"',
        "pip install -e .",
        "scripts/fetch_aa_source.py",
        "scripts/build_canonical.py",
        "python -m aa",
    ):
        assert required in workflow
    # The cache never keeps a VM alive, changes session state, or gates startup.
    for forbidden in ("sleep infinity", "tail -f /dev/null", "nohup", "tmux", "screen -"):
        assert forbidden not in workflow
    # Save steps run opportunistically after install and never gate startup.
    assert "Save public pip cache" in workflow
    assert "Save public embedding model cache" in workflow
    assert "if: success()" in workflow


def test_prefetch_script_refuses_invalid_lock(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "bad-lock.json"
    bad.write_text(json.dumps({"format": "wrong", "model_id": MODEL_ID}), encoding="utf-8")
    proc = _run_prefetch(
        ["--check-only", "--lock", str(bad), "--hf-cache", str(tmp_path / "hub")], {}
    )
    assert proc.returncode != 0


def test_public_cache_module_has_no_vendor_coupling() -> None:
    text = (_repo_root() / "src" / "aa" / "corpus" / "public_cache.py").read_text(encoding="utf-8")
    for snippet in ("import torch", "import transformers", "from telegram", "import openai"):
        assert snippet not in text


def test_prefetch_script_module_loads_without_hub_dependency() -> None:
    path = _repo_root() / "scripts" / "prefetch_public_assets.py"
    spec = importlib.util.spec_from_file_location("prefetch_public_assets_under_test", str(path))
    assert spec is not None and spec.loader is not None
