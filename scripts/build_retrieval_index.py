#!/usr/bin/env python3
"""Build the RU-first hybrid retrieval index (issue #17).

Reads the generated full hierarchy (``corpus/generated/corpus_structure.json``,
issue #8) plus the committed manifests and embedding lock, then writes the
runtime-only text-bearing index to ``corpus/generated/retrieval/``:

- ``index.json`` — version metadata, RU chunk records, EN control metadata;
- ``lexical.db`` — SQLite FTS5/BM25 table over RU chunks;
- ``dense.json`` — normalized dense vectors (pinned e5 when locally cached,
  otherwise deterministic local hashing backend).

Ordinary turns reuse the built index; this script runs once per corpus or
model change. Plaintext indexes are never committed and never enter the
Actions cache. Logs carry only paths, sizes and digests, never book text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.retrieval.index import build_hybrid_index  # noqa: E402

DEFAULT_FULL = ROOT / "corpus" / "generated" / "corpus_structure.json"
DEFAULT_RU_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_EN_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_LOCK = ROOT / "corpus" / "embedding.lock.json"
DEFAULT_OUT = ROOT / "corpus" / "generated" / "retrieval"


def _fail(message: str) -> int:
    print(f"retrieval index build failed: {message}", file=sys.stderr)
    return 1


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{label} is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build the hybrid index from generated artifacts."""
    parser = argparse.ArgumentParser(description="Build the RU-first hybrid index.")
    parser.add_argument("--full-structure", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--ru-manifest", type=Path, default=DEFAULT_RU_MANIFEST)
    parser.add_argument("--en-manifest", type=Path, default=DEFAULT_EN_MANIFEST)
    parser.add_argument("--embedding-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--backend",
        choices=("auto", "hashing", "e5"),
        default="hashing",
        help="Dense backend: hashing is hermetic; e5 loads the pinned local "
        "model only (never network); auto prefers cached e5.",
    )
    args = parser.parse_args(argv)

    try:
        full = _load_json(args.full_structure, "full corpus structure")
        ru_manifest = _load_json(args.ru_manifest, "RU manifest")
        en_manifest = _load_json(args.en_manifest, "EN manifest")
        lock = _load_json(args.embedding_lock, "embedding lock")
    except ValueError as exc:
        return _fail(f"{exc}; build corpus structure first via scripts/build_corpus_structure.py")

    if lock.get("model_id") != "intfloat/multilingual-e5-base":
        return _fail("embedding lock must pin intfloat/multilingual-e5-base")

    try:
        index = build_hybrid_index(
            full,  # type: ignore[arg-type]
            ru_manifest=ru_manifest,  # type: ignore[arg-type]
            en_manifest=en_manifest,  # type: ignore[arg-type]
            embedding_lock=lock,  # type: ignore[arg-type]
            out_dir=args.out_dir,
            backend=args.backend,
        )
    except ValueError as exc:
        return _fail(str(exc))

    digest = hashlib.sha256((args.out_dir / "index.json").read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "chunks": index.chunk_count,
                "backend": index.metadata.get("embedding_backend"),
                "ru_artifact_sha256": index.metadata.get("ru_artifact_sha256"),
                "en_artifact_sha256": index.metadata.get("en_artifact_sha256"),
                "embedding_revision": index.metadata.get("embedding_revision"),
                "index_sha256": digest,
                "out_dir": str(args.out_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
