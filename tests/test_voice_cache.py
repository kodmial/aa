"""Public voice-model cache tests (issue #79).

The voice cache is optional acceleration only for the fixed public
assets introduced by #76-#78. These tests prove the Definition of Done
without network access: lock pinning against the #76-#78 implementation
constants, exact compatibility keys, miss/hit/repair behaviour, non-fatal
failure, user-data exclusion, measurement recording, and reuse of the
#59 cache semantics in the runtime workflow.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from aa.corpus.public_cache import PublicCacheError
from aa.corpus.voice_cache import (
    assert_voice_cache_paths_safe,
    default_voice_lock_path,
    load_voice_lock,
    verify_gigaam_model,
    verify_presentation_model,
    verify_tts_model,
    voice_cache_keys,
    voice_gigaam_key,
    voice_lock_digest,
    voice_presentation_key,
    voice_silero_key,
    write_gigaam_marker,
    write_presentation_marker,
    write_tts_marker,
)

_PRESENTATION_SHA = "fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252"
_GIGAAM_REVISION = "9f5a77e8975211abe8511693accd3a63ee1e9f43"
_FIXTURE_ONNX = b"onnx-fixture:" + _PRESENTATION_SHA.encode()


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _lock() -> dict[str, Any]:
    return load_voice_lock(default_voice_lock_path())


def _fixture_lock(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a lock copy whose presentation SHA matches fixture bytes."""
    lock = _lock()
    presentation = dict(lock["presentation"])
    assert isinstance(presentation, dict)
    fixture_sha = hashlib.sha256(_FIXTURE_ONNX).hexdigest()
    presentation["sha256"] = fixture_sha
    lock = dict(lock)
    lock["presentation"] = presentation
    path = tmp_path / "voice.lock.json"
    path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _load_prefetch_module() -> Any:
    path = _repo_root() / "scripts" / "prefetch_voice_assets.py"
    spec = importlib.util.spec_from_file_location("prefetch_voice_assets_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prefetch_voice_assets_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _make_gigaam(model_dir: pathlib.Path, lock: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.int8.onnx").write_bytes(b"gigaam-fixture-model")
    (model_dir / "tokens.txt").write_bytes(b"gigaam-fixture-tokens")
    write_gigaam_marker(model_dir, lock)


def _make_silero(model_path: pathlib.Path, lock: dict[str, Any]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"silero-fixture-model")
    write_tts_marker(model_path, lock)


def _make_presentation(model_path: pathlib.Path, lock: dict[str, Any]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(_FIXTURE_ONNX)
    write_presentation_marker(model_path, lock)


def _run_prefetch(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(_repo_root() / "scripts" / "prefetch_voice_assets.py"), *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        env=env,
        check=False,
    )


def test_voice_lock_pins_fixed_assets() -> None:
    lock = _lock()
    gigaam = lock["gigaam"]
    assert gigaam["model_id"] == "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc"
    assert gigaam["revision"] == _GIGAAM_REVISION
    assert gigaam["model_file"] == "large/model.int8.onnx"
    assert gigaam["tokens_file"] == "large/tokens.txt"
    silero = lock["silero"]
    assert silero["model_id"] == "v5_5_ru"
    assert silero["url"] == "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    presentation = lock["presentation"]
    assert presentation["model_id"] == "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2"
    assert presentation["model_file"] == "onnx/model_quantized.onnx"
    assert presentation["sha256"] == _PRESENTATION_SHA
    # The lock carries no user-derived data or secrets (its policy list
    # names excluded categories; it contains no audio, text, or keys).
    raw = default_voice_lock_path().read_text(encoding="utf-8")
    for snippet in ("TELEGRAM_BOT_TOKEN", "AGE-SECRET-KEY", ".ogg", ".pcm", "Chapter"):
        assert snippet not in raw
    for required in ("Telegram voice", "transcript", "speaker", "session", "secret"):
        assert required.lower() in raw.lower()


def test_voice_lock_matches_implementation_constants() -> None:
    """The lock must agree with the #76-#78 fixed implementation contracts."""
    from aa.telegram.tts import TTS_MODEL_ID, TTS_MODEL_URL  # noqa: PLC0415
    from aa.telegram.voice import (  # noqa: PLC0415
        MODEL_FILE,
        MODEL_REPO,
        MODEL_REVISION,
        TOKENS_FILE,
    )
    from aa.telegram.voice_presentation import (  # noqa: PLC0415
        PRESENTATION_MODEL_FILE,
        PRESENTATION_MODEL_REPO,
        PRESENTATION_MODEL_SHA256,
    )

    lock = _lock()
    assert lock["gigaam"]["model_id"] == MODEL_REPO
    assert lock["gigaam"]["revision"] == MODEL_REVISION
    assert lock["gigaam"]["model_file"] == MODEL_FILE
    assert lock["gigaam"]["tokens_file"] == TOKENS_FILE
    assert lock["silero"]["model_id"] == TTS_MODEL_ID
    assert lock["silero"]["url"] == TTS_MODEL_URL
    assert lock["presentation"]["model_id"] == PRESENTATION_MODEL_REPO
    assert lock["presentation"]["model_file"] == PRESENTATION_MODEL_FILE
    assert lock["presentation"]["sha256"] == PRESENTATION_MODEL_SHA256


def test_cache_keys_are_compatibility_bound() -> None:
    lock = _lock()
    digest = voice_lock_digest(lock)
    gigaam = voice_gigaam_key(
        os_name="Linux",
        arch="X64",
        python_version="3.12",
        revision=_GIGAAM_REVISION,
        deps_hash="abc123",
    )
    assert gigaam.startswith("aa-voice-gigaam-v1-")
    for token in ("Linux", "X64", "3.12", _GIGAAM_REVISION, "abc123"):
        assert token in gigaam
    assert (
        voice_gigaam_key(
            os_name="macOS",
            arch="X64",
            python_version="3.12",
            revision=_GIGAAM_REVISION,
            deps_hash="abc123",
        )
        != gigaam
    )
    assert (
        voice_gigaam_key(
            os_name="Linux",
            arch="X64",
            python_version="3.11",
            revision=_GIGAAM_REVISION,
            deps_hash="abc123",
        )
        != gigaam
    )
    with pytest.raises(PublicCacheError):
        voice_gigaam_key(
            os_name="Linux", arch="X64", python_version="3.12", revision="short", deps_hash="abc123"
        )

    silero = voice_silero_key(
        os_name="Linux", arch="X64", python_version="3.12", model_id="v5_5_ru", deps_hash="abc123"
    )
    assert silero.startswith("aa-voice-silero-v1-")
    assert "v5_5_ru" in silero and "abc123" in silero
    with pytest.raises(PublicCacheError):
        voice_silero_key(
            os_name="Linux", arch="X64", python_version="3.12", model_id="v4_ru", deps_hash="abc123"
        )

    presentation = voice_presentation_key(
        os_name="Linux",
        arch="X64",
        python_version="3.12",
        sha256=_PRESENTATION_SHA,
        deps_hash="abc123",
    )
    assert presentation.startswith("aa-voice-presentation-v1-")
    assert _PRESENTATION_SHA in presentation and "abc123" in presentation
    with pytest.raises(PublicCacheError):
        voice_presentation_key(
            os_name="Linux", arch="X64", python_version="3.12", sha256="short", deps_hash="abc123"
        )

    keys = voice_cache_keys(
        os_name="Linux", arch="X64", python_version="3.12", lock=lock, deps_hash=digest
    )
    assert set(keys) == {"gigaam", "silero", "presentation"}
    assert keys["gigaam"] != keys["silero"] != keys["presentation"]
    # One exact key per family: no broad fallback restore prefix exists.
    import aa.corpus.voice_cache as voice_cache_module  # noqa: PLC0415

    for name in dir(voice_cache_module):
        assert "restore" not in name.lower()


def test_print_keys_matches_lock() -> None:
    proc = _run_prefetch(
        ["--print-keys", "--os", "Linux", "--arch", "X64", "--deps-hash", "abc123"], {}
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["keys"]["gigaam"].startswith("aa-voice-gigaam-v1-")
    assert _GIGAAM_REVISION in payload["keys"]["gigaam"]
    assert "v5_5_ru" in payload["keys"]["silero"]
    assert _PRESENTATION_SHA in payload["keys"]["presentation"]


def test_clean_run_with_cache_disabled_succeeds(tmp_path: pathlib.Path) -> None:
    proc = _run_prefetch(
        [
            "--check-only",
            "--lock",
            str(_fixture_lock(tmp_path)),
            "--gigaam-dir",
            str(tmp_path / "gigaam"),
            "--tts-path",
            str(tmp_path / "tts" / "v5_5_ru.pt"),
            "--presentation-path",
            str(tmp_path / "presentation" / "model_quantized.onnx"),
        ],
        {"AA_PUBLIC_CACHE_ENABLED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "disabled"


def test_first_run_misses_then_second_run_restores(tmp_path: pathlib.Path) -> None:
    lock_path = _fixture_lock(tmp_path)
    lock = load_voice_lock(lock_path)
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"
    args = [
        "--check-only",
        "--lock",
        str(lock_path),
        "--gigaam-dir",
        str(gigaam_dir),
        "--tts-path",
        str(tts_path),
        "--presentation-path",
        str(presentation_path),
    ]
    assert verify_gigaam_model(gigaam_dir, lock) is False
    proc = _run_prefetch(args, {})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "miss"
    assert all(item["category"] == "miss" for item in payload["families"].values())

    _make_gigaam(gigaam_dir, lock)
    _make_silero(tts_path, lock)
    _make_presentation(presentation_path, lock)
    assert verify_gigaam_model(gigaam_dir, lock) is True
    assert verify_tts_model(tts_path, lock) is True
    assert verify_presentation_model(presentation_path, lock) is True
    proc = _run_prefetch(args, {})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "hit"
    for family, item in payload["families"].items():
        assert item["status"] == "hit", family
        assert item["category"] == "hit", family
        assert item["downloaded_bytes"] == 0, family
        assert item["snapshot_bytes"] > 0, family


def test_wrong_key_misses_safely(tmp_path: pathlib.Path) -> None:
    lock = load_voice_lock(_fixture_lock(tmp_path))
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"
    _make_gigaam(gigaam_dir, lock)
    _make_silero(tts_path, lock)
    _make_presentation(presentation_path, lock)

    wrong_gigaam = json.loads(json.dumps(lock))
    wrong_gigaam["gigaam"]["revision"] = "0" * 40
    assert verify_gigaam_model(gigaam_dir, wrong_gigaam) is False
    wrong_silero = json.loads(json.dumps(lock))
    wrong_silero["silero"]["model_id"] = "v4_ru"
    assert verify_tts_model(tts_path, wrong_silero) is False
    wrong_presentation = json.loads(json.dumps(lock))
    wrong_presentation["presentation"]["sha256"] = "0" * 64
    assert verify_presentation_model(presentation_path, wrong_presentation) is False


def test_corrupt_cache_is_rejected_and_repaired(tmp_path: pathlib.Path) -> None:
    lock_path = _fixture_lock(tmp_path)
    lock = load_voice_lock(lock_path)
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"
    args = [
        "--check-only",
        "--lock",
        str(lock_path),
        "--gigaam-dir",
        str(gigaam_dir),
        "--tts-path",
        str(tts_path),
        "--presentation-path",
        str(presentation_path),
    ]
    _make_gigaam(gigaam_dir, lock)
    _make_silero(tts_path, lock)
    _make_presentation(presentation_path, lock)

    # Corrupt one byte of the checksum-guarded classifier and empty a
    # GigaAM file: both families must fail verification.
    presentation_path.write_bytes(b"X" + _FIXTURE_ONNX[1:])
    assert verify_presentation_model(presentation_path, lock) is False
    (gigaam_dir / "model.int8.onnx").write_bytes(b"")
    assert verify_gigaam_model(gigaam_dir, lock) is False
    assert verify_tts_model(tts_path, lock) is True

    proc = _run_prefetch(args, {})
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "miss"
    assert payload["families"]["presentation"]["category"] == "rejected"
    assert payload["families"]["gigaam"]["category"] == "rejected"
    assert payload["families"]["silero"]["category"] == "hit"
    # check-only prunes corrupt content so a normal download can repair it.
    assert verify_gigaam_model(gigaam_dir, lock) is False
    assert verify_presentation_model(presentation_path, lock) is False
    _make_gigaam(gigaam_dir, lock)
    _make_presentation(presentation_path, lock)
    assert verify_gigaam_model(gigaam_dir, lock) is True
    assert verify_presentation_model(presentation_path, lock) is True


def test_full_miss_and_failure_are_non_fatal(tmp_path: pathlib.Path) -> None:
    """A total miss and a backend failure behave like a clean install."""
    module = _load_prefetch_module()
    lock = load_voice_lock(_fixture_lock(tmp_path))
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"

    def _boom(
        family: str,
        *,
        gigaam_dir: pathlib.Path,
        tts_path: pathlib.Path,
        presentation_path: pathlib.Path,
    ) -> None:
        raise OSError(f"backend unavailable for {family}")

    module._download_family = _boom
    assert (
        module._prefetch(
            lock=lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        == 0
    )
    assert verify_gigaam_model(gigaam_dir, lock) is False


def test_failed_download_leaves_no_trusted_cache(tmp_path: pathlib.Path) -> None:
    """Garbage bytes that fail verification are pruned, never trusted."""
    module = _load_prefetch_module()
    lock = load_voice_lock(_fixture_lock(tmp_path))
    # Corrupt the expected classifier checksum so the stubbed download can
    # never verify; GigaAM/Silero stubs still succeed without network.
    presentation = dict(lock["presentation"])
    assert isinstance(presentation, dict)
    presentation["sha256"] = "1" * 64
    lock = dict(lock)
    lock["presentation"] = presentation
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"

    def _stub_download(
        family: str,
        *,
        gigaam_dir: pathlib.Path,
        tts_path: pathlib.Path,
        presentation_path: pathlib.Path,
    ) -> None:
        if family == "gigaam":
            gigaam_dir.mkdir(parents=True, exist_ok=True)
            (gigaam_dir / "model.int8.onnx").write_bytes(b"stub-model")
            (gigaam_dir / "tokens.txt").write_bytes(b"stub-tokens")
        elif family == "silero":
            tts_path.parent.mkdir(parents=True, exist_ok=True)
            tts_path.write_bytes(b"stub-silero")
        else:
            presentation_path.parent.mkdir(parents=True, exist_ok=True)
            presentation_path.write_bytes(b"stub-checksum-mismatch")

    module._download_family = _stub_download
    assert (
        module._prefetch(
            lock=lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        == 0
    )
    assert verify_gigaam_model(gigaam_dir, lock) is True
    assert verify_tts_model(tts_path, lock) is True
    # The checksum-guarded family is rejected and pruned, never trusted.
    assert verify_presentation_model(presentation_path, lock) is False
    assert not presentation_path.exists()


def test_measurements_are_recorded(tmp_path: pathlib.Path) -> None:
    lock_path = _fixture_lock(tmp_path)
    lock = load_voice_lock(lock_path)
    gigaam_dir = tmp_path / "gigaam"
    tts_path = tmp_path / "tts" / "v5_5_ru.pt"
    presentation_path = tmp_path / "presentation" / "model_quantized.onnx"
    _make_gigaam(gigaam_dir, lock)
    _make_silero(tts_path, lock)
    _make_presentation(presentation_path, lock)
    proc = _run_prefetch(
        [
            "--check-only",
            "--lock",
            str(lock_path),
            "--gigaam-dir",
            str(gigaam_dir),
            "--tts-path",
            str(tts_path),
            "--presentation-path",
            str(presentation_path),
        ],
        {},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["duration_ms"] >= 0
    assert isinstance(payload["peak_rss_mb"], (int, float))
    for family, item in payload["families"].items():
        assert item["duration_ms"] >= 0, family
        assert item["snapshot_bytes"] > 0, family
        assert item["downloaded_bytes"] == 0, family
        assert isinstance(item["peak_rss_mb"], (int, float)), family
        assert item["category"] in {"hit", "miss", "rejected"}, family


def test_no_user_derived_data_enters_cache(tmp_path: pathlib.Path) -> None:
    assert_voice_cache_paths_safe(
        [
            str(tmp_path / "gigaam"),
            str(tmp_path / "tts" / "v5_5_ru.pt"),
            str(tmp_path / "presentation" / "model_quantized.onnx"),
        ]
    )
    user_paths = [
        "models/voice/note.ogg",
        "models/tts/reply.pcm",
        "tmp/transcript.txt",
        "tmp/generated-answer.json",
        "tmp/speaker-probabilities.json",
        "tmp/aa-voice-note.opus",
        "storage/telegram.session",
        "sessions/opencode.db",
        "corpus/generated/canonical.json",
    ]
    for path in user_paths:
        with pytest.raises(PublicCacheError):
            assert_voice_cache_paths_safe([path])


def test_prefetch_script_refuses_invalid_lock(tmp_path: pathlib.Path) -> None:
    bad = tmp_path / "bad-lock.json"
    bad.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
    proc = _run_prefetch(
        ["--check-only", "--lock", str(bad), "--gigaam-dir", str(tmp_path / "gigaam")], {}
    )
    assert proc.returncode != 0


def test_voice_cache_reuses_public_cache_semantics() -> None:
    text = (_repo_root() / "src" / "aa" / "corpus" / "voice_cache.py").read_text(encoding="utf-8")
    # Reuses the #59 enable flag, path guard, and error type.
    assert "from aa.corpus.public_cache import" in text
    assert "is_cache_enabled" in text
    assert "assert_cache_paths_safe" in text
    assert "PublicCacheError" in text
    # No second cache subsystem: no GitHub cache action wrapper and no
    # provisioning/download of its own (downloads stay in the #76-#78
    # ensure_* paths reused by the prefetch script).
    assert "actions/cache" not in text
    assert "urlopen" not in text
    assert "urllib" not in text
    assert "snapshot_download" not in text
    script = (_repo_root() / "scripts" / "prefetch_voice_assets.py").read_text(encoding="utf-8")
    assert "ensure_model_files" in script
    assert "ensure_tts_model_file" in script
    assert "ensure_presentation_model_file" in script
    assert "huggingface.co" not in script
    assert "models.silero.ai" not in script


def test_workflow_caches_only_fixed_voice_assets() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "Compute voice cache keys",
        "Restore GigaAM voice model cache",
        "Restore Silero voice model cache",
        "Restore voice presentation model cache",
        "Verify pinned voice assets",
        "Prefetch pinned voice models",
        "Save GigaAM voice model cache",
        "Save Silero voice model cache",
        "Save voice presentation model cache",
        "prefetch_voice_assets.py --check-only",
        "prefetch_voice_assets.py --print-keys",
        "corpus/voice.lock.json",
        "hashFiles('pyproject.toml', 'corpus/voice.lock.json')",
    ):
        assert required in workflow, required
    # Exact keys only for model files: no restore-keys fallback appears in
    # any voice restore block (only the #59 public pip cache keeps one).
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("name: Restore ") and "voice" in line.lower():
            block = "\n".join(lines[index : index + 12])
            assert "actions/cache/restore" in block
            assert "restore-keys" not in block
            assert "Exact key only" in block
        if line.strip().startswith("name: Save ") and "voice" in line.lower():
            block = "\n".join(lines[index : index + 8])
            assert "actions/cache/save" in block
            assert "cache-hit != 'true'" in block
    # Save failure is non-fatal.
    assert workflow.count("continue-on-error: true") >= 10
    # Measurements recorded on the workflow (hit/miss/rejected category only).
    for required in ("downloaded_bytes", "peak_rss_mb", "duration_ms", "category"):
        assert required in workflow, required
    # No user-derived path is listed as a voice cache path.
    for fragment in (".ogg", ".pcm", "transcript", "sessions", "TELEGRAM"):
        for line in lines:
            if line.strip().startswith("path:") and fragment in line:
                raise AssertionError(f"sensitive cache path in workflow: {line!r}")


def test_workflow_reuses_pip_cache_for_speech_dependencies() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    # The pinned speech dependencies (sherpa-onnx, torch, onnxruntime) are
    # covered by the existing #59 pip cache keyed on pyproject.toml.
    assert "hashFiles('pyproject.toml')" in workflow
    assert workflow.count("Restore public pip cache") == 1
    assert workflow.count("Save public pip cache") == 1
    assert "sherpa-onnx" in (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")


def test_workflow_runtime_semantics_are_unchanged() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "python -m aa",
        "prefetch_public_assets.py --check-only",
        "if: success()",
    ):
        assert required in workflow
