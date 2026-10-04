#!/usr/bin/env python3
"""Validate the Russian real-world v1_1 corpus (issue #61).

Machine entry point for CI and for downstream issues #62/#72/#73/#63:
validates the successor allow/emergency/block oracle, the input/oracle
split, the session-reset control event, ids, counts, UTF-8, deduplication,
provenance resolution, #21 routing review and #46 meaning-preservation
anchors, then checks the stable version record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa.qualification.ru_realworld import (  # noqa: E402
    CORPUS_REL,
    INPUT_REL,
    ORACLE_REL,
    SOURCES_REL,
    VERSION_REL,
    RuRealWorldCorpusError,
    build_version_payload,
    find_repo_root,
    validate,
)


def main() -> int:
    try:
        root = find_repo_root()
        summary = validate(root)
    except (RuRealWorldCorpusError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ru-realworld corpus INVALID: {exc}", file=sys.stderr)
        return 1
    version_path = root / VERSION_REL
    if not version_path.exists():
        print(f"ru-realworld corpus INVALID: missing {VERSION_REL}", file=sys.stderr)
        return 1
    try:
        recorded = json.loads(version_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(
            f"ru-realworld corpus INVALID: {VERSION_REL} unreadable: {exc}",
            file=sys.stderr,
        )
        return 1
    expected = build_version_payload(summary)
    if recorded != expected:
        print(
            "ru-realworld corpus INVALID: "
            f"{VERSION_REL} does not match validated corpus; "
            "regenerate it from the validated files",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "corpus": CORPUS_REL,
                "input": INPUT_REL,
                "oracle": ORACLE_REL,
                "sources": SOURCES_REL,
                "corpus_sha256": summary.corpus_sha256,
                "input_sha256": summary.input_sha256,
                "oracle_sha256": summary.oracle_sha256,
                "sources_sha256": summary.sources_sha256,
                "counts": {
                    "single_turn": summary.single_turn,
                    "multi_turn_journeys": summary.journeys,
                    "multi_turn_substantive_turns": summary.substantive_journey_turns,
                    "control_events": summary.control_events,
                    "total_substantive_utterances": summary.total_substantive,
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
