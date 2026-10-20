"""P0 BREAKER kodmial/aa#403: canonical static checks are a pytest invariant.

Convergence scope: ``fingerprint:A:shell-step-failed:gate-a-shell``
(blocking gate A, component gate-a-shell, main
``1edbe7394e02db60d19105f1f2b977349cc5585e``, run 39420278923, local
repair recurrence 3, systemic recurrence 3).

Reconstruction across recent exact-main qualification history: three
consecutive repair trees reached Gate A with the full unit suite passing,
then failed ``scripts/verify.sh`` on its static checks. Commit ``b425a49``
failed ``ruff check`` with E501 in ``scripts/verify_runtime_image.py``;
commits ``0499f4f`` and ``1edbe73`` failed ``ruff format --check`` on the
same toolchain contract test. Each repair had exercised only focused tests,
so the canonical static contract was never observed until exact-main
qualification. This is not a product, grounding, relevance, safety, or Gate
E defect; it is a verification-ownership gap.

Smallest architecture invariant: the canonical static contract is part of
the executable test contract, not a separate shell-only concern. A tree
that pytest accepts must already satisfy the same ruff check and ruff
format checks that Gate A runs, so a narrow repair cannot publish a tree
that only the later shell step rejects.

The repair keeps Product Contract #110, Gate semantics, grounding,
relevance, safety and the Gate E SLO unchanged. It adds no prompt branch,
keyword/stem table, numbered-step parsing, token-overlap relevance, or
handcrafted semantic predicate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify.sh"


def _ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_clean(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_canonical_ruff_check_passes_for_the_repository_tree() -> None:
    _assert_clean(_ruff("check", "."))


def test_canonical_ruff_format_is_stable_for_the_repository_tree() -> None:
    _assert_clean(_ruff("format", "--check", "."))


def test_canonical_verify_entrypoint_runs_the_static_contract() -> None:
    text = VERIFY.read_text(encoding="utf-8")
    assert "python3 -m ruff check ." in text
    assert "python3 -m ruff format --check ." in text
    assert "python3 -m mypy src tests" in text
