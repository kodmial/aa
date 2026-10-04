#!/usr/bin/env python3
"""Prefetch the pinned public voice models (issue #79).

Optional acceleration only for the fixed public assets introduced by
#76-#78:

- GigaAM ``large/model.int8.onnx`` + ``large/tokens.txt`` at the pinned
  revision (issue #76);
- Silero TTS ``v5_5_ru.pt`` (issue #77);
- the pinned voice-presentation ONNX file with its required SHA-256
  (issue #78).

Cache semantics reuse the #59 implementation pattern
(``scripts/prefetch_public_assets.py`` + ``aa.corpus.public_cache``):

- ``--print-keys`` emits one exact content/compatibility key per model
  family (runner OS/arch, Python 3.12, fixed model identity/revision or
  checksum, dependency-lock hash). Model files never use a broad
  fallback restore key.
- ``--check-only`` verifies the pinned identity and prunes stale or
  corrupt content, then exits 0 without downloading;
- the default mode reuses a verified hit, otherwise downloads normally
  with the pinned production provisioning path and revalidates before
  use;
- any download/cache failure is non-fatal (exit 0 with a ``miss`` or
  ``skipped`` status) so a valid runtime is never failed by the cache;
- only the pinned public assets from ``corpus/voice.lock.json`` are
  handled. Telegram voice/PCM/OGG, transcripts, generated answers,
  speaker probabilities/classes/features, session/user data, and
  secrets are never touched here.

Each run prints one JSON document with per-family measurements: startup
duration, snapshot bytes, bytes downloaded on hit vs miss, peak RSS,
and the hit/miss/rejected category only. Logs contain only paths,
sizes, digests, and categories, never audio, transcripts, or secrets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.public_cache import PublicCacheError  # noqa: E402
from aa.corpus.voice_cache import (  # noqa: E402
    PYTHON_VERSION,
    assert_voice_cache_paths_safe,
    default_gigaam_dir,
    default_presentation_model_path,
    default_tts_model_path,
    default_voice_lock_path,
    invalidate_gigaam_model,
    invalidate_presentation_model,
    invalidate_tts_model,
    is_cache_enabled,
    load_voice_lock,
    verify_gigaam_model,
    verify_presentation_model,
    verify_tts_model,
    voice_cache_keys,
    write_gigaam_marker,
    write_presentation_marker,
    write_tts_marker,
)

FAMILIES = ("gigaam", "silero", "presentation")


def _peak_rss_mb() -> float:
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return -1.0
    # Linux ru_maxrss is KiB; macOS reports bytes.
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _snapshot_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            return -1
    return total


def _family_paths(
    family: str, *, gigaam_dir: Path, tts_path: Path, presentation_path: Path
) -> list[Path]:
    if family == "gigaam":
        return [gigaam_dir / "model.int8.onnx", gigaam_dir / "tokens.txt"]
    if family == "silero":
        return [tts_path]
    return [presentation_path]


def _verify_family(
    family: str,
    lock: dict[str, Any],
    *,
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> bool:
    if family == "gigaam":
        return verify_gigaam_model(gigaam_dir, lock)
    if family == "silero":
        return verify_tts_model(tts_path, lock)
    return verify_presentation_model(presentation_path, lock)


def _invalidate_family(
    family: str,
    *,
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> None:
    if family == "gigaam":
        invalidate_gigaam_model(gigaam_dir)
    elif family == "silero":
        invalidate_tts_model(tts_path)
    else:
        invalidate_presentation_model(presentation_path)


def _download_family(
    family: str,
    *,
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> None:
    if family == "gigaam":
        from aa.telegram.voice import ensure_model_files  # noqa: PLC0415

        ensure_model_files(gigaam_dir)
    elif family == "silero":
        from aa.telegram.tts import ensure_tts_model_file  # noqa: PLC0415

        ensure_tts_model_file(tts_path)
    else:
        from aa.telegram.voice_presentation import (  # noqa: PLC0415
            ensure_presentation_model_file,
        )

        ensure_presentation_model_file(presentation_path)


def _write_family_marker(
    family: str,
    lock: dict[str, Any],
    *,
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> None:
    if family == "gigaam":
        write_gigaam_marker(gigaam_dir, lock)
    elif family == "silero":
        write_tts_marker(tts_path, lock)
    else:
        write_presentation_marker(presentation_path, lock)


def _status(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


def _check_only(
    *,
    lock: dict[str, Any],
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> int:
    started = time.perf_counter()
    families: dict[str, Any] = {}
    for family in FAMILIES:
        family_started = time.perf_counter()
        paths = _family_paths(
            family,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        if _verify_family(
            family,
            lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        ):
            families[family] = {
                "status": "hit",
                "category": "hit",
                "duration_ms": int((time.perf_counter() - family_started) * 1000),
                "snapshot_bytes": _snapshot_bytes(paths),
                "downloaded_bytes": 0,
                "peak_rss_mb": _peak_rss_mb(),
            }
            continue
        # Miss, corruption, or incompatibility: prune so a normal
        # download can repair it.
        rejected = any(path.exists() for path in paths)
        _invalidate_family(
            family,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        families[family] = {
            "status": "miss",
            "category": "rejected" if rejected else "miss",
            "duration_ms": int((time.perf_counter() - family_started) * 1000),
            "snapshot_bytes": 0,
            "downloaded_bytes": 0,
            "peak_rss_mb": _peak_rss_mb(),
        }
    overall = "hit" if all(item["status"] == "hit" for item in families.values()) else "miss"
    return _status(
        {
            "status": overall,
            "families": families,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "peak_rss_mb": _peak_rss_mb(),
        }
    )


def _prefetch(
    *,
    lock: dict[str, Any],
    gigaam_dir: Path,
    tts_path: Path,
    presentation_path: Path,
) -> int:
    started = time.perf_counter()
    families: dict[str, Any] = {}
    for family in FAMILIES:
        family_started = time.perf_counter()
        paths = _family_paths(
            family,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        had_files = any(path.exists() for path in paths)
        if _verify_family(
            family,
            lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        ):
            families[family] = {
                "status": "hit",
                "category": "hit",
                "duration_ms": int((time.perf_counter() - family_started) * 1000),
                "snapshot_bytes": _snapshot_bytes(paths),
                "downloaded_bytes": 0,
                "peak_rss_mb": _peak_rss_mb(),
            }
            continue
        # Stale or corrupt content must never be trusted: clear it first.
        was_rejected = had_files
        _invalidate_family(
            family,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        try:
            _download_family(
                family,
                gigaam_dir=gigaam_dir,
                tts_path=tts_path,
                presentation_path=presentation_path,
            )
        except Exception as exc:  # noqa: BLE001 - cache backend failure is non-fatal
            _invalidate_family(
                family,
                gigaam_dir=gigaam_dir,
                tts_path=tts_path,
                presentation_path=presentation_path,
            )
            print(
                f"voice asset download failed, continuing without cache: {exc}",
                file=sys.stderr,
            )
            families[family] = {
                "status": "miss",
                "category": "rejected" if was_rejected else "miss",
                "reason": exc.__class__.__name__,
                "duration_ms": int((time.perf_counter() - family_started) * 1000),
                "snapshot_bytes": 0,
                "downloaded_bytes": 0,
                "peak_rss_mb": _peak_rss_mb(),
            }
            continue
        try:
            _write_family_marker(
                family,
                lock,
                gigaam_dir=gigaam_dir,
                tts_path=tts_path,
                presentation_path=presentation_path,
            )
        except OSError as exc:
            print(f"voice cache marker failed, continuing: {exc}", file=sys.stderr)
        if _verify_family(
            family,
            lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        ):
            downloaded = _snapshot_bytes(paths)
            families[family] = {
                "status": "downloaded",
                "category": "miss",
                "duration_ms": int((time.perf_counter() - family_started) * 1000),
                "snapshot_bytes": downloaded,
                "downloaded_bytes": downloaded,
                "peak_rss_mb": _peak_rss_mb(),
            }
            continue
        _invalidate_family(
            family,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
        print("downloaded voice asset failed pinned verification; cache pruned", file=sys.stderr)
        families[family] = {
            "status": "miss",
            "category": "rejected",
            "reason": "verification-failed",
            "duration_ms": int((time.perf_counter() - family_started) * 1000),
            "snapshot_bytes": 0,
            "downloaded_bytes": 0,
            "peak_rss_mb": _peak_rss_mb(),
        }
    overall = (
        "hit"
        if all(item["status"] == "hit" for item in families.values())
        else "downloaded"
        if all(item["status"] in ("hit", "downloaded") for item in families.values())
        else "miss"
    )
    return _status(
        {
            "status": overall,
            "families": families,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "peak_rss_mb": _peak_rss_mb(),
        }
    )


def _print_keys(
    *,
    lock: dict[str, Any],
    os_name: str,
    arch: str,
    deps_hash: str,
) -> int:
    keys = voice_cache_keys(
        os_name=os_name,
        arch=arch,
        python_version=PYTHON_VERSION,
        lock=lock,
        deps_hash=deps_hash,
    )
    return _status({"keys": keys, "python_version": PYTHON_VERSION})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch the pinned public voice models.")
    parser.add_argument("--lock", type=Path, default=default_voice_lock_path())
    parser.add_argument("--gigaam-dir", type=Path, default=None)
    parser.add_argument("--tts-path", type=Path, default=None)
    parser.add_argument("--presentation-path", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the pinned identity without downloading.",
    )
    parser.add_argument(
        "--print-keys",
        action="store_true",
        help="Print the exact per-family cache keys without touching models.",
    )
    parser.add_argument("--os", dest="os_name", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--deps-hash", default=None)
    args = parser.parse_args(argv)

    try:
        lock = load_voice_lock(args.lock)
    except PublicCacheError as exc:
        print(f"voice cache failed: {exc}", file=sys.stderr)
        return 1

    if args.print_keys:
        if not args.os_name or not args.arch or not args.deps_hash:
            print("voice cache failed: --os, --arch and --deps-hash are required", file=sys.stderr)
            return 1
        return _print_keys(
            lock=lock, os_name=args.os_name, arch=args.arch, deps_hash=args.deps_hash
        )

    gigaam_dir = Path(args.gigaam_dir) if args.gigaam_dir is not None else default_gigaam_dir()
    tts_path = Path(args.tts_path) if args.tts_path is not None else default_tts_model_path()
    presentation_path = (
        Path(args.presentation_path)
        if args.presentation_path is not None
        else default_presentation_model_path()
    )
    try:
        assert_voice_cache_paths_safe([str(gigaam_dir), str(tts_path), str(presentation_path)])
    except PublicCacheError as exc:
        print(f"voice cache failed: {exc}", file=sys.stderr)
        return 1
    if not is_cache_enabled():
        gigaam = lock["gigaam"]
        assert isinstance(gigaam, dict)
        return _status({"status": "disabled", "model_id": str(gigaam["model_id"])})

    # The workflow binds the dependency hash into the cache keys; the
    # voice lock digest is recorded inside each family marker instead.
    if args.check_only:
        return _check_only(
            lock=lock,
            gigaam_dir=gigaam_dir,
            tts_path=tts_path,
            presentation_path=presentation_path,
        )
    return _prefetch(
        lock=lock,
        gigaam_dir=gigaam_dir,
        tts_path=tts_path,
        presentation_path=presentation_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
