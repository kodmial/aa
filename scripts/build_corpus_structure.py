#!/usr/bin/env python3
"""Build the aligned RU/EN AA hierarchy and compact book map (issue #8).

Reads the validated canonical artifacts (English ``aa-canonical/1`` plus
Russian ``aa-canonical-ru/1`` from issue #50) and writes:

- ``corpus/generated/corpus_structure.json`` (private, text-bearing full
  hierarchy for runtime retrieval; ignored by Git);
- ``corpus/structure.json`` (public, metadata-only aligned structure);
- ``corpus/book-map.md`` (public, compact primarily-English routing map).

Section/chapter alignment is mandatory; paragraph/chunk alignment is never
forced across languages. Every chunk round-trips to exact source text and
no machine translation enters storage. The map is navigation only and must
stay within the 6000-token compact-map budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.budget import BOOK_MAP_BUDGET_TOKENS, estimate_text_tokens  # noqa: E402
from aa.corpus.structure import (  # noqa: E402
    SECTION_IDS,
    build_full_structure,
    build_public_structure,
    render_book_map,
)

DEFAULT_EN_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_RU_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_EN_CANONICAL = ROOT / "corpus" / "generated" / "canonical.json"
DEFAULT_RU_CANONICAL = ROOT / "corpus" / "generated" / "canonical.ru.json"
DEFAULT_FULL_OUTPUT = ROOT / "corpus" / "generated" / "corpus_structure.json"
DEFAULT_PUBLIC_OUTPUT = ROOT / "corpus" / "structure.json"
DEFAULT_MAP_OUTPUT = ROOT / "corpus" / "book-map.md"

# Longest verbatim run allowed to appear in a public artifact. Titles and
# topics are short navigation strings; anything longer proves a literary
# leak from the canonical text.
MAX_PUBLIC_VERBATIM_RUN = 60


def _fail(message: str) -> int:
    print(f"corpus structure build failed: {message}", file=sys.stderr)
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


def _section_inputs(artifact: dict[str, object], label: str) -> list[dict[str, object]]:
    sections = artifact.get("sections")
    if not isinstance(sections, list):
        raise ValueError(f"{label} has no sections list")
    inputs: list[dict[str, object]] = []
    for entry in sections:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} has a malformed section")
        for key in ("id", "title", "text", "source_id", "source_file", "source_sha256"):
            if not isinstance(entry.get(key), str) or not entry.get(key):
                raise ValueError(f"{label} section is missing {key}")
        inputs.append(
            {
                "id": str(entry["id"]),
                "title": str(entry["title"]),
                "text": str(entry["text"]),
                "source_id": str(entry["source_id"]),
                "source_file": str(entry["source_file"]),
                "source_sha256": str(entry["source_sha256"]),
            }
        )
    if [item["id"] for item in inputs] != list(SECTION_IDS):
        raise ValueError(f"{label} sections are not the canonical twelve in order")
    return inputs


def _check_no_text_leak(
    *,
    public_text: str,
    section_texts: list[str],
    label: str,
) -> None:
    lowered_public = public_text.lower()
    for section_text in section_texts:
        normalized = " ".join(section_text.split())
        if len(normalized) <= MAX_PUBLIC_VERBATIM_RUN:
            continue
        for start in range(0, len(normalized) - MAX_PUBLIC_VERBATIM_RUN, 400):
            probe = normalized[start : start + MAX_PUBLIC_VERBATIM_RUN].lower()
            if len(probe.strip()) < MAX_PUBLIC_VERBATIM_RUN:
                continue
            if probe in lowered_public:
                raise ValueError(f"{label} carries substantial literary text")
    if '"text"' in public_text or "'text'" in public_text:
        # A serialized ``text`` key would mean a text-bearing payload leaked
        # into a public metadata file.
        raise ValueError(f"{label} must not contain a text payload")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build hierarchy plus public routing artifacts."""
    parser = argparse.ArgumentParser(description="Build the aligned RU/EN hierarchy.")
    parser.add_argument("--en-manifest", type=Path, default=DEFAULT_EN_MANIFEST)
    parser.add_argument("--ru-manifest", type=Path, default=DEFAULT_RU_MANIFEST)
    parser.add_argument("--en-canonical", type=Path, default=DEFAULT_EN_CANONICAL)
    parser.add_argument("--ru-canonical", type=Path, default=DEFAULT_RU_CANONICAL)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC_OUTPUT)
    parser.add_argument("--map-output", type=Path, default=DEFAULT_MAP_OUTPUT)
    args = parser.parse_args(argv)

    try:
        en_manifest = _load_json(args.en_manifest, "EN manifest")
        ru_manifest = _load_json(args.ru_manifest, "RU manifest")
        en_artifact = _load_json(args.en_canonical, "EN canonical artifact")
        ru_artifact = _load_json(args.ru_canonical, "RU canonical artifact")
    except ValueError as exc:
        return _fail(f"{exc}; restore artifacts first via scripts/restore_canonical.py")

    if en_artifact.get("format") != "aa-canonical/1":
        return _fail("EN canonical artifact has an unsupported format")
    if ru_artifact.get("format") != "aa-canonical-ru/1":
        return _fail("RU canonical artifact has an unsupported format")

    try:
        en_sections = _section_inputs(en_artifact, "EN canonical artifact")
        ru_sections = _section_inputs(ru_artifact, "RU canonical artifact")
    except ValueError as exc:
        return _fail(str(exc))

    try:
        full = build_full_structure(
            en_sections=en_sections,
            ru_sections=ru_sections,
            en_edition=str(en_manifest.get("edition", "")),
            ru_edition=str(ru_manifest.get("edition", "")),
            en_corpus_version=str(en_manifest.get("artifact_sha256", "")),
            ru_corpus_version=str(ru_manifest.get("artifact_sha256", "")),
        )
        public = build_public_structure(en_manifest=en_manifest, ru_manifest=ru_manifest)
        book_map = render_book_map(public)
    except ValueError as exc:
        return _fail(str(exc))

    map_tokens = estimate_text_tokens(book_map)
    if map_tokens > BOOK_MAP_BUDGET_TOKENS:
        return _fail(
            f"book map needs {map_tokens} tokens "
            f"but the compact-map budget is {BOOK_MAP_BUDGET_TOKENS}"
        )

    full_payload = (json.dumps(full, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    public_payload = (
        json.dumps(public, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    map_payload = book_map.encode("utf-8")

    section_texts = [str(item["text"]) for item in en_sections + ru_sections]
    try:
        _check_no_text_leak(
            public_text=public_payload.decode("utf-8"),
            section_texts=section_texts,
            label="public structure",
        )
        _check_no_text_leak(public_text=book_map, section_texts=section_texts, label="book map")
    except ValueError as exc:
        return _fail(str(exc))

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.full_output.write_bytes(full_payload)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_bytes(public_payload)
    args.map_output.parent.mkdir(parents=True, exist_ok=True)
    args.map_output.write_bytes(map_payload)

    ru_chunks = sum(len(section["ru"]["chunks"]) for section in full["sections"])  # type: ignore[index]
    en_chunks = sum(len(section["en"]["chunks"]) for section in full["sections"])  # type: ignore[index]
    print(
        json.dumps(
            {
                "sections": len(SECTION_IDS),
                "ru_chunks": ru_chunks,
                "en_chunks": en_chunks,
                "map_tokens": map_tokens,
                "map_budget": BOOK_MAP_BUDGET_TOKENS,
                "full_sha256": _sha256_bytes(full_payload),
                "public_sha256": _sha256_bytes(public_payload),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
