#!/usr/bin/env python3
"""Prefetch the pinned BGE reranker model (issue #116).

Optional acceleration only, mirroring ``prefetch_public_assets.py``. The
AA runtime works identically when this cache is missing or unavailable,
falling back to the deterministic offline scoring backend:

- ``--check-only`` verifies the pinned identity and prunes corrupt or
  incompatible content, then exits 0 without downloading;
- the default mode reuses a verified hit, otherwise downloads normally
  via ``huggingface_hub`` when available and revalidates before use;
- any download/cache failure is non-fatal (exit 0 with a ``miss`` or
  ``skipped`` status) so a valid runtime is never failed by the cache;
- only the pinned reranker id + revision from
  ``corpus/reranker.lock.json`` is handled. Corpus text, retrieval
  indexes, secrets, Telegram/user state, and decrypted artifacts are
  never touched here.

Runtime bootstrap may populate/validate this cache, but an ordinary
turn reuses the already-loaded model with zero downloads and zero
network access.

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

from aa.retrieval.reranker import (  # noqa: E402
    default_reranker_lock_path,
    invalidate_cached_reranker,
    load_reranker_lock,
    resolve_hf_cache_dir,
    resolve_reranker_root,
    snapshot_dir,
    verify_cached_reranker,
    write_reranker_marker,
)


def _status(payload: dict[str, object]) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


def _check_only(*, lock_path: Path, hf_cache: Path | None) -> int:
    try:
        lock = load_reranker_lock(lock_path)
    except ValueError as exc:
        print(f"reranker cache check failed: {exc}", file=sys.stderr)
        return 1
    hf_dir = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_dir()
    model_root = resolve_reranker_root(hf_dir)
    if verify_cached_reranker(model_root, lock):
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
    invalidate_cached_reranker(model_root)
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
        lock = load_reranker_lock(lock_path)
    except ValueError as exc:
        print(f"reranker prefetch failed: {exc}", file=sys.stderr)
        return 1
    hf_dir = Path(hf_cache) if hf_cache is not None else resolve_hf_cache_dir()
    model_root = resolve_reranker_root(hf_dir)
    if verify_cached_reranker(model_root, lock):
        return _status(
            {
                "status": "hit",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
                "model_root": str(model_root),
            }
        )
    invalidate_cached_reranker(model_root)
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
        invalidate_cached_reranker(model_root)
        print(f"reranker download failed, continuing without cache: {exc}", file=sys.stderr)
        return _status(
            {
                "status": "miss",
                "reason": exc.__class__.__name__,
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
            }
        )
    required = lock.get("required_files")
    snapshot = snapshot_dir(model_root, str(lock.get("revision")))
    files_ok = isinstance(required, list) and bool(required)
    if files_ok:
        for name in required:
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                files_ok = False
                break
            candidate = snapshot / name
            try:
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    files_ok = False
                    break
            except OSError:
                files_ok = False
                break
    if files_ok and verify_cached_reranker(model_root, lock):
        write_reranker_marker(model_root, lock)
        return _status(
            {
                "status": "downloaded",
                "model_id": str(lock.get("model_id")),
                "revision": str(lock.get("revision")),
                "model_root": str(model_root),
            }
        )
    invalidate_cached_reranker(model_root)
    print("downloaded reranker failed pinned verification; cache pruned", file=sys.stderr)
    return _status(
        {
            "status": "miss",
            "reason": "verification-failed",
            "model_id": str(lock.get("model_id")),
            "revision": str(lock.get("revision")),
        }
    )


def main(argv: list[str] | None = None) -> int:
    """Run the reranker prefetch CLI."""
    parser = argparse.ArgumentParser(description="Prefetch the pinned BGE reranker model.")
    parser.add_argument("--lock", type=Path, default=default_reranker_lock_path())
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
