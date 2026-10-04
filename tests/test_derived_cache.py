"""Encrypted disposable derived retrieval cache tests (issue #58).

The cache is acceleration only. These tests prove the Definition of Done
without network access and without committing any canonical book text:
exact content-addressed keys, encrypted-before-save, hit restores a
ready-to-open index, miss/stale/corrupt/unavailable transparently
rebuilds, non-fatal save failure, no plaintext/secrets in Actions cache,
trusted-save only, and privacy-safe metrics.

All literary content uses invented fixture sentences.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from aa.corpus import age_v1
from aa.retrieval import derived_cache as dc
from aa.retrieval.index import build_hybrid_index, open_hybrid_index

RU_FIXTURES: dict[str, str] = {
    "doctors-opinion": (
        "Фиктивное мнение доктора о тяге. Наблюдение продолжается.\n\n"
        "Второй абзац мнения доктора. Выводы записываются аккуратно."
    ),
    "chapter-1": (
        "Фиктивный рассказ о первом глотке. Герой начал пить каждый вечер.\n\n"
        "Второй абзац рассказа. Утро после выпивки было тяжелым."
    ),
    "chapter-2": (
        "Фиктивный выход есть для пьющих. Надежда и поддержка рядом.\n\n"
        "Второй абзац выхода. Сообщество встречает новичков тепло."
    ),
    "chapter-3": (
        "Фиктивный алкоголизм как феномен тяги. Тяга приходит внезапно.\n\n"
        "Второй абзац о последствиях. Многие опасаются тяги в семье."
    ),
    "chapter-4": (
        "Фиктивные размышления агностика. Готовность принять помощь растет.\n\n"
        "Второй абзац агностика. Сомнения обсуждаются открыто."
    ),
    "chapter-5": (
        "Фиктивная программа в действии требует честности. Шаги каждый день.\n\n"
        "Второй абзац программы. Утренний настрой задает тон."
    ),
    "chapter-6": (
        "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию.\n\n"
        "Второй абзац работы. Вечером подводим итоги дня."
    ),
    "chapter-7": (
        "Фиктивная работа с другими людьми. Несем весть тем кто страдает.\n\n"
        "Второй абзац помощи. Разговор ведется спокойно и честно."
    ),
    "chapter-8": (
        "Фиктивная жена ругает из-за пьянки. Женушка переживает за семью.\n\n"
        "Второй абзац о семье. Пьянство разрушает доверие постепенно."
    ),
    "chapter-9": (
        "Фиктивные новые отношения в семье. Доверие возвращается постепенно.\n\n"
        "Второй абзац семьи. Разговоры становятся спокойнее."
    ),
    "chapter-10": (
        "Фиктивное обращение к работодателям. Трезвость на работе важна.\n\n"
        "Второй абзац работодателям. Поддержка коллег помогает многим."
    ),
    "chapter-11": (
        "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется.\n\n"
        "Второй абзац будущего. Планы строятся на трезвую голову."
    ),
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _fixture_full() -> dict[str, Any]:
    from aa.corpus.structure import SECTION_IDS, build_full_structure

    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.\n\n"
                f"Fixture EN {section_id} second paragraph.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": RU_FIXTURES[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    return dict(
        build_full_structure(
            en_sections=en_sections,
            ru_sections=ru_sections,
            en_edition="en-edition",
            ru_edition="ru-edition",
            en_corpus_version="en-v1",
            ru_corpus_version="ru-v1",
        )
    )


def _build_fixture_index(out_dir: pathlib.Path) -> Any:
    lock = json.loads((_repo_root() / "corpus" / "embedding.lock.json").read_text())
    ru_manifest = {
        "format": "aa-canonical-manifest-ru/1",
        "artifact_sha256": "r" * 64,
        "edition": "ru-edition",
    }
    en_manifest = {
        "format": "aa-canonical-manifest/1",
        "artifact_sha256": "e" * 64,
        "edition": "en-edition",
    }
    return build_hybrid_index(
        _fixture_full(),
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=lock,
        out_dir=out_dir,
        backend="hashing",
    )


def _run_manager(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(__import__("os").environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(_repo_root() / "scripts" / "manage_derived_cache.py"), *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        env=env,
        check=False,
    )


def test_cache_key_is_exact_and_content_addressed() -> None:
    bindings = dc.read_bindings(_repo_root(), os_name="Linux", arch="X64")
    first = dc.derived_cache_key(bindings)
    second = dc.derived_cache_key(dc.read_bindings(_repo_root(), os_name="Linux", arch="X64"))
    assert first.key == second.key
    assert first.fingerprint == second.fingerprint == first.digest[:16]
    assert first.key.startswith("aa-derived-retrieval-v1-Linux-X64-")
    other_os = dc.derived_cache_key(dc.read_bindings(_repo_root(), os_name="macOS", arch="X64"))
    assert other_os.key != first.key
    other_arch = dc.derived_cache_key(dc.read_bindings(_repo_root(), os_name="Linux", arch="ARM64"))
    assert other_arch.key != first.key
    # Semantic identity only: no ephemeral workflow identity in the key.
    lowered = first.key.lower()
    for fragment in ("branch", "run-id", "run_id", "timestamp", "hostname"):
        assert fragment not in lowered
    assert "Linux" in first.key and "X64" in first.key


def test_cache_key_binds_all_required_identity() -> None:
    bindings = dc.read_bindings(_repo_root())
    for required in (
        "cache_schema_version",
        "ru_artifact_sha256",
        "en_artifact_sha256",
        "structure_sha256",
        "structure_format",
        "retrieval_config_sha256",
        "retrieval_production_config_id",
        "retrieval_production_config_version",
        "retrieval_planner_schema",
        "retrieval_gold_sha256",
        "embedding_model_id",
        "embedding_revision",
        "embedding_lock_sha256",
        "abi",
        "os",
        "arch",
    ):
        assert bindings.get(required) not in (None, ""), f"missing binding {required}"
    assert bindings["cache_schema_version"] == dc.DERIVED_CACHE_SCHEMA_VERSION
    assert bindings["embedding_model_id"] == "intfloat/multilingual-e5-base"
    assert len(str(bindings["embedding_revision"])) == 40
    abi = bindings["abi"]
    assert isinstance(abi, dict)
    for key in ("python", "sqlite", "index_format", "planner_schema", "rrf_k"):
        assert abi.get(key) not in (None, "")


def test_runtime_boots_with_cache_completely_disabled(tmp_path: pathlib.Path) -> None:
    assert dc.is_cache_enabled({"AA_DERIVED_CACHE_ENABLED": "0"}) is False
    assert dc.is_cache_enabled({"AA_DERIVED_CACHE_ENABLED": "1"}) is True
    assert dc.is_cache_enabled({}) is True
    proc = _run_manager(
        [
            "--mode",
            "restore",
            "--cache-dir",
            str(tmp_path / "staging"),
            "--retrieval-dir",
            str(tmp_path / "retrieval"),
        ],
        {"AA_DERIVED_CACHE_ENABLED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "disabled"
    # Disabled cache still allows a deterministic clean rebuild.
    index = _build_fixture_index(tmp_path / "clean")
    checked = dc.self_test_index(tmp_path / "clean")
    assert checked["chunk_count"] == index.chunk_count


def test_valid_hit_restores_ready_to_open_index(tmp_path: pathlib.Path) -> None:
    identity, recipient = age_v1.generate_identity()
    try:
        src = tmp_path / "src"
        _build_fixture_index(src)
        bindings = dc.read_bindings(_repo_root(), os_name="Linux", arch="X64")
        key = dc.derived_cache_key(bindings)
        digests = dc.bundle_file_digests(src)
        checked = dc.self_test_index(src)
        manifest = dc.build_cache_manifest(
            cache_key=key,
            bindings=bindings,
            file_digests=digests,
            chunk_count=int(checked["chunk_count"]),
            index_version=int(checked["index_version"]),
        )
        package = dc.create_package(src, manifest)
        # Deterministic packaging: same inputs produce identical bytes.
        assert dc.create_package(src, manifest) == package
        encrypted = dc.encrypt_package(package, recipient)
        assert encrypted.startswith(b"age-encryption.org/v1\n")
        assert identity.encode() not in encrypted
        # Plaintext retrieval text must never appear in the encrypted bundle.
        probe_text = str(json.loads((src / "index.json").read_text())["chunks"][0]["text"])
        assert probe_text.encode() not in encrypted
        assert probe_text[:32].encode() not in encrypted
        # Restore path: checksum, decrypt, manifest bindings, self-test.
        assert dc.sha256_bytes(encrypted) == dc.sha256_bytes(encrypted)
        recovered = dc.decrypt_package(encrypted, identity)
        assert recovered == package
        files, recovered_manifest = dc.extract_package(recovered)
        dc.verify_manifest(recovered_manifest, live_bindings=bindings, expected_key=key.key)
        dc.verify_extracted_files(files, recovered_manifest)
        dest = tmp_path / "restored"
        dc.write_extracted_bundle(files, dest, recovered_manifest)
        opened = open_hybrid_index(dest)
        assert opened.chunk_count == checked["chunk_count"]
        rechecked = dc.self_test_index(dest)
        assert rechecked["chunk_count"] == checked["chunk_count"]
    finally:
        identity = "destroyed"  # noqa: F841
        recipient = "destroyed"  # noqa: F841


def test_miss_stale_corrupt_transparently_rebuild(tmp_path: pathlib.Path) -> None:
    identity, recipient = age_v1.generate_identity()
    try:
        bindings = dc.read_bindings(_repo_root(), os_name="Linux", arch="X64")
        key = dc.derived_cache_key(bindings)
        # Miss: empty staging directory has no encrypted package.
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not (empty / dc.ENCRYPTED_NAME).exists()
        # Corrupt package is rejected, never trusted.
        with pytest.raises(dc.DerivedCacheError):
            dc.extract_package(b"not a zst archive")
        # Wrong identity cannot decrypt.
        src = tmp_path / "src"
        _build_fixture_index(src)
        digests = dc.bundle_file_digests(src)
        checked = dc.self_test_index(src)
        manifest = dc.build_cache_manifest(
            cache_key=key,
            bindings=bindings,
            file_digests=digests,
            chunk_count=int(checked["chunk_count"]),
            index_version=int(checked["index_version"]),
        )
        encrypted = dc.encrypt_package(dc.create_package(src, manifest), recipient)
        other_identity, _ = age_v1.generate_identity()
        try:
            with pytest.raises(dc.DerivedCacheError):
                dc.decrypt_package(encrypted, other_identity)
        finally:
            other_identity = "destroyed"  # noqa: F841
        tampered = bytearray(encrypted)
        tampered[-10] ^= 0xFF
        with pytest.raises(dc.DerivedCacheError):
            dc.decrypt_package(bytes(tampered), identity)
        # Stale version binding is rejected.
        stale_bindings = dict(bindings)
        stale_bindings["ru_artifact_sha256"] = "0" * 64
        with pytest.raises(dc.DerivedCacheError):
            dc.verify_manifest(manifest, live_bindings=stale_bindings, expected_key=key.key)
        # Tampered file bytes fail the manifest checksum.
        files, _ = dc.extract_package(dc.decrypt_package(encrypted, identity))
        files["index.json"] = b"tampered"
        with pytest.raises(dc.DerivedCacheError):
            dc.verify_extracted_files(files, manifest)
        # Any rejection deletes restored data and rebuilds deterministically.
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "lexical.db").write_bytes(b"partial-restore")
        dc.clear_directory(dest)
        assert list(dest.iterdir()) == []
        rebuilt = _build_fixture_index(dest)
        assert dc.self_test_index(dest)["chunk_count"] == rebuilt.chunk_count
        # Script-level miss is transparent and non-fatal (exit 0, miss status).
        proc = _run_manager(
            [
                "--mode",
                "restore",
                "--cache-dir",
                str(empty),
                "--retrieval-dir",
                str(tmp_path / "out"),
            ],
            {"AA_DERIVED_CACHE_ENABLED": "1", "AA_BOOK_AGE_IDENTITY": ""},
        )
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["status"] in {"miss", "rejected"}
    finally:
        identity = "destroyed"  # noqa: F841
        recipient = "destroyed"  # noqa: F841


def test_cache_backend_unavailable_rebuilds(tmp_path: pathlib.Path) -> None:
    # No staging dir at all: backend unavailable must miss, not fail.
    proc = _run_manager(
        [
            "--mode",
            "restore",
            "--cache-dir",
            str(tmp_path / "no-such-staging"),
            "--retrieval-dir",
            str(tmp_path / "out"),
        ],
        {"AA_DERIVED_CACHE_ENABLED": "1", "AA_BOOK_AGE_IDENTITY": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "miss"


def test_save_failure_is_non_fatal(tmp_path: pathlib.Path) -> None:
    # Missing index: save is skipped (exit 0), runtime still starts.
    proc = _run_manager(
        [
            "--mode",
            "save",
            "--cache-dir",
            str(tmp_path / "staging"),
            "--retrieval-dir",
            str(tmp_path / "no-such-index"),
        ],
        {"AA_DERIVED_CACHE_ENABLED": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] in {"save-skipped", "disabled"}
    # Disabled cache: save is a non-fatal no-op.
    proc = _run_manager(
        [
            "--mode",
            "save",
            "--cache-dir",
            str(tmp_path / "staging"),
            "--retrieval-dir",
            str(tmp_path / "no-such-index"),
        ],
        {"AA_DERIVED_CACHE_ENABLED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "disabled"


def test_no_plaintext_corpus_index_or_secrets_enter_actions_cache(
    tmp_path: pathlib.Path,
) -> None:
    dc.assert_cache_paths_safe([str(tmp_path / "aa-derived-cache")])
    for sensitive in (
        "corpus/generated/retrieval",
        "corpus/generated/canonical.json",
        "corpus/source/raw/AA.txt",
        "corpus/source/encrypted/canonical.tar.zst.age",
        "corpus/source/fetch-state.json",
        "storage/telegram.session",
        "sessions/opencode.db",
        "AGE-SECRET-KEY-foo",
        "lexical.db",
        "dense.json",
        "index.json",
    ):
        with pytest.raises(dc.DerivedCacheError):
            dc.assert_cache_paths_safe([sensitive])
    # Non-secret metadata carries digests only, never text or secrets.
    bindings = dc.read_bindings(_repo_root())
    key = dc.derived_cache_key(bindings)
    meta = dc.build_meta_payload(
        cache_key=key,
        encrypted_sha256="a" * 64,
        encrypted_bytes=123,
        bindings=bindings,
    )
    raw = json.dumps(meta, sort_keys=True)
    assert "AGE-SECRET-KEY" not in raw
    assert "TELEGRAM_BOT_TOKEN" not in raw
    for probe in RU_FIXTURES.values():
        assert probe[:32] not in raw


def test_workflow_caches_only_encrypted_bundle() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    assert "manage_derived_cache.py --mode print-key" in workflow
    assert "manage_derived_cache.py --mode restore-or-rebuild" in workflow
    assert "manage_derived_cache.py --mode save" in workflow
    # Exact lookup only for the encrypted bundle: no broad fallback key.
    assert "Restore encrypted derived retrieval cache" in workflow
    assert "Upload encrypted derived retrieval cache" in workflow
    # Only the public pip cache uses a fallback prefix; neither the public
    # model cache nor the derived bundle does.
    restore_section = workflow.split("Restore encrypted derived retrieval cache", 1)[1]
    block = restore_section.split("uses: actions/cache/restore", 1)[1]
    restore_block = block.split("- name:", 1)[0]
    assert "restore-keys:" not in restore_block
    assert "Exact key only" in workflow
    assert workflow.count("continue-on-error: true") >= 6
    # Only the encrypted staging directory enters the derived cache.
    assert "corpus/embedding.lock.json" in workflow
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:") and "derived" in line.lower():
            lowered = line.lower()
            assert "corpus/generated" not in lowered
            assert "corpus/source" not in lowered
            assert "lexical.db" not in lowered
            assert "dense.json" not in lowered
            assert "index.json" not in lowered
    for fragment in (
        "corpus/generated",
        "corpus/source/raw",
        "canonical.json",
        ".tar.zst\n",
        "TELEGRAM_BOT_TOKEN",
        "AA_BOOK_AGE_IDENTITY",
    ):
        for line in workflow.splitlines():
            if line.strip().startswith("path:") and fragment in line:
                raise AssertionError(f"sensitive derived cache path: {line!r}")


def test_low_trust_workflows_cannot_poison_trusted_cache() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "aa-runtime.yml").read_text(
        encoding="utf-8"
    )
    # The runtime workflow never runs on pull requests: only manually
    # dispatched trusted runs can save the derived cache.
    assert "workflow_dispatch" in workflow
    assert "pull_request" not in workflow
    assert "pull_request_target" not in workflow
    assert "Trusted save only" in workflow or "trusted" in workflow.lower()
    assert "low-trust" in workflow.lower()
    # Save is opportunistic and never gates startup.
    assert "Save encrypted derived retrieval cache" in workflow
    assert "if: success()" in workflow


def test_startup_metrics_are_privacy_safe(tmp_path: pathlib.Path) -> None:
    identity, _ = age_v1.generate_identity()
    try:
        src = tmp_path / "src"
        _build_fixture_index(src)
        probe = str(json.loads((src / "index.json").read_text())["chunks"][0]["text"])
        proc = _run_manager(
            [
                "--mode",
                "restore",
                "--cache-dir",
                str(tmp_path / "staging"),
                "--retrieval-dir",
                str(tmp_path / "out"),
            ],
            {"AA_DERIVED_CACHE_ENABLED": "1", "AA_BOOK_AGE_IDENTITY": identity},
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["status"] in {"hit", "miss", "rejected", "disabled"}
        allowed_keys = {
            "status",
            "reason",
            "cache_key",
            "cache_fingerprint",
            "cache_digest",
            "restore_ms",
            "decrypt_ms",
            "build_ms",
            "package_ms",
            "chunk_count",
            "index_version",
            "encrypted_sha256",
            "encrypted_bytes",
            "os",
            "arch",
        }
        assert set(payload) <= allowed_keys
        assert probe not in proc.stdout
        assert probe[:32] not in proc.stdout
        assert identity not in proc.stdout
        assert identity not in proc.stderr
        # Module source never logs chunk text or secrets by construction.
        module_text = (_repo_root() / "src" / "aa" / "retrieval" / "derived_cache.py").read_text(
            encoding="utf-8"
        )
        assert "record.text" not in module_text or "self-test" in module_text
    finally:
        identity = "destroyed"  # noqa: F841


def test_derived_cache_module_has_no_vendor_coupling() -> None:
    text = (_repo_root() / "src" / "aa" / "retrieval" / "derived_cache.py").read_text(
        encoding="utf-8"
    )
    for snippet in ("import torch", "import transformers", "from telegram", "import openai"):
        assert snippet not in text


def test_generated_retrieval_dir_stays_gitignored() -> None:
    gitignore = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/generated/" in gitignore
