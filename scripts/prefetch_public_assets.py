#!/usr/bin/env python3
"""Prefetch the pinned public embedding model (issue #59).

Optional acceleration only. The AA runtime works identically when this
cache is missing or unavailable:

- ``--check-only`` verifies the pinned identity and prunes corrupt or
  incompatible content, then exits 0 without downloading;
- the default mode reuses a verified hit, otherwise downloads normally
  via ``huggingface_hub`` when available and revalidates before use;
- any download/cache failure is non-fatal (exit 0 with a ``miss`` or
  ``skipped`` status) so a valid runtime is never failed by the cache;
- only the pinned public model id + revision from
  ``corpus/embedding.lock.json`` is handled. Corpus text, retrieval
  indexes, secrets, Telegram/user state, and decrypted artifacts are
  never touched here.

Logs contain only paths, sizes, and digests, never model weights or
secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.public_cache import (  # noqa: E402
    PublicCacheError,
    default_lock_path,
    invalidate_cached_model,
    is_cache_enabled,
    load_embedding_lock,
    resolve_hf_cache_dir,
    resolve_model_root,
    snapshot_dir,
    verify_cached_model,
    write_marker,
)


def _status(payload: dict[str, object]) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


def _check_only(*, lock_path: Path, hf_cache: Path | None) -> int:
    try:
        lock = load_embedding_lock(lock_path)
    except PublicCacheError as exc:
        print(f"public cache check failed: {exc}", file=sys.stderr)
        return 1
    if not is_cache_enabled():
        return _status({"status": "disabled", "model_id": str(lock.get("model_id"))})
    hf_dir = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_dir()
    model_root = resolve_model_root(hf_dir)
    if verify_cached_model(model_root, lock):
        snapshot = snapshot_dir(model_root, str(lock.get("revision")))
        total = 0
        try:
            for child in snapshot.rglob("*"):
                if child.is_file():
                    total += child.stat().st_size
        except OSError:
            total = -1
        return _status(
            {
                "status": "hit",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
                "model_root": str(model_root),
                "snapshot_bytes": total,
            }
        )
    # Miss, corruption, or incompatibility: prune so a normal download repairs it.
    invalidate_cached_model(model_root)
    return _status(
        {
            "status": "miss",
            "model_id": str(lock.get("model_id")),
            "revision": str(lock.get("revision")),
            "model_root": str(model_root),
        }
    )


def _prefetch(*, lock_path: Path, hf_cache: Path | None) -> int:
    try:
        lock = load_embedding_lock(lock_path)
    except PublicCacheError as exc:
        print(f"public cache prefetch failed: {exc}", file=sys.stderr)
        return 1
    if not is_cache_enabled():
        return _status({"status": "disabled", "model_id": str(lock.get("model_id"))})
    hf_dir = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_dir()
    model_root = resolve_model_root(hf_dir)
    if verify_cached_model(model_root, lock):
        return _status(
            {
                "status": "hit",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
                "model_root": str(model_root),
            }
        )
    # Stale or corrupt content must never be trusted: clear it before download.
    invalidate_cached_model(model_root)
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional dependency, never fatal
        return _status(
            {
                "status": "skipped",
                "reason": f"huggingface_hub is unavailable: {exc.__class__.__name__}",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
            }
        )
    try:
        snapshot_download(
            repo_id=str(lock.get("model_id")),
            revision=str(lock.get("revision")),
            cache_dir=str(hf_dir),
        )
    except Exception as exc:  # noqa: BLE001 - cache backend failure is non-fatal
        invalidate_cached_model(model_root)
        print(f"public model download failed, continuing without cache: {exc}", file=sys.stderr)
        return _status(
            {
                "status": "miss",
                "reason": exc.__class__.__name__,
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
            }
        )
    snapshot = snapshot_dir(model_root, str(lock.get("revision")))
    required = lock.get("required_files")
    files_ok = isinstance(required, list) and bool(required)
    if files_ok:
        for name in required:
            assert isinstance(name, str)
            candidate = snapshot / name
            try:
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    files_ok = False
                    break
            except OSError:
                files_ok = False
                break
    if files_ok:
        write_marker(model_root, lock)
        return _status(
            {
                "status": "downloaded",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
                "model_root": str(model_root),
            }
        )
    invalidate_cached_model(model_root)
    print("downloaded model failed pinned verification; cache pruned", file=sys.stderr)
    return _status(
        {
            "status": "miss",
            "reason": "verification-failed",
            "model_id": str(lock.get("model_id")),
            "revision": str(lock.get("revision")),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch the pinned public embedding model.")
    parser.add_argument("--lock", type=Path, default=default_lock_path())
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the pinned identity without downloading.",
    )
    args = parser.parse_args(argv)
    _ = os.environ.get("AA_PUBLIC_CACHE_ENABLED", "")
    if args.check_only:
        return _check_only(lock_path=args.lock, hf_cache=args.hf_cache)
    return _prefetch(lock_path=args.lock, hf_cache=args.hf_cache)


if __name__ == "__main__":
    raise SystemExit(main())
