#!/usr/bin/env python3
"""Runtime qualification gate for the AA Telegram worker (issues #19/#9).

Telegram polling must not start unless the qualified retrieval artifacts
match the live corpus/model versions:

- ``qualification/aa-retrieval.json`` (#19 final RU-first retrieval/tool
  policy: recall/coverage/slang/fidelity/stale gates, RU-only production);
- ``qualification/ru_first_retrieval.v1.decision.json`` (#47 RU-first
  architecture decision with EN-secondary disabled);
- the four narrow book tools exist with deny-by-default permissions.

Stale or mismatched artifacts fail closed. Logs carry only ids, digests
and counts, never user text or corpus text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _fail(message: str) -> int:
    print(f"runtime qualification failed: {message}", file=sys.stderr)
    return 1


def main() -> int:
    """Validate live qualification bindings; exit 0 only when current."""
    try:
        from aa.qualification.aa_retrieval import (
            ARTIFACT_REL as AA_RETRIEVAL_REL,
        )
        from aa.qualification.aa_retrieval import (
            validate_artifact_payload as validate_aa_retrieval,
        )
        from aa.qualification.ru_first import (
            DECISION_REL as RU_FIRST_REL,
        )
        from aa.qualification.ru_first import (
            validate_decision_payload as validate_ru_first,
        )
    except Exception as exc:
        return _fail(f"qualification modules are not importable: {exc}")

    try:
        aa_payload = json.loads((ROOT / AA_RETRIEVAL_REL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"missing #19 artifact: {AA_RETRIEVAL_REL}")
    except json.JSONDecodeError as exc:
        return _fail(f"#19 artifact is not valid JSON: {exc}")
    try:
        validate_aa_retrieval(aa_payload, repo_root=ROOT)
    except ValueError as exc:
        return _fail(f"#19 artifact is stale or invalid: {exc}")
    if aa_payload.get("production", {}).get("config_id") != "ru-first-only":
        return _fail("#19 production configuration is not ru-first-only")

    try:
        ru_first_payload = json.loads((ROOT / RU_FIRST_REL).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"missing #47 decision artifact: {RU_FIRST_REL}")
    except json.JSONDecodeError as exc:
        return _fail(f"#47 decision artifact is not valid JSON: {exc}")
    try:
        validate_ru_first(ru_first_payload, repo_root=ROOT)
    except ValueError as exc:
        return _fail(f"#47 decision artifact is stale or invalid: {exc}")

    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        if not (ROOT / ".opencode" / "tools" / f"{tool}.ts").is_file():
            return _fail(f"missing OpenCode book tool wrapper: {tool}")
    try:
        config = json.loads((ROOT / "opencode.json").read_text(encoding="utf-8"))
        permission = config["agent"]["aa"]["permission"]
    except Exception as exc:
        return _fail(f"agent permission contract is unreadable: {exc}")
    if permission.get("*") != "deny":
        return _fail("aa agent must stay deny-by-default")
    allowed = {name for name, value in permission.items() if value == "allow"}
    if allowed != {"book_search", "book_read", "book_expand", "book_section"}:
        return _fail("aa agent must allow exactly the four book tools")

    retrieval = aa_payload.get("retrieval", {})
    print(
        json.dumps(
            {
                "aa_retrieval": AA_RETRIEVAL_REL,
                "ru_first": RU_FIRST_REL,
                "recall_at_5": retrieval.get("recall_at_5"),
                "production": aa_payload.get("production", {}).get("config_id"),
                "status": "qualified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
