"""Regression: live AA agent provider access denial must retire its route."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aa-opencode-model-route-e2e.yml"


class ProviderDenialRouteTests(unittest.TestCase):
    def test_access_denial_is_classified_as_real_model_unavailable(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        anchor = (
            'elif grep -Eiq "OpenCode.s free tier can only be used from within OpenCode" '
            '"$log"; then'
        )
        self.assertIn(anchor, text)
        # The actual provider message has an apostrophe. Use the identical
        # POSIX extended regex shape as the workflow (without shell quoting).
        pattern = r"OpenCode.s free tier can only be used from within OpenCode"
        self.assertIsNotNone(
            re.search(
                pattern,
                "Error: OpenCode's free tier can only be used from within OpenCode",
                re.IGNORECASE,
            )
        )
        self.assertIn("kind=unavailable", text)
        self.assertIn("provider_failure=$kind", text)
        self.assertIn("steps.live_aa.outputs.provider_failure != ''", text)
        self.assertIn("operation: fail", text)
        self.assertIn("failure-reason: ${{ steps.live_aa.outputs.provider_failure }}", text)

    def test_denial_is_not_reclassified_as_qualified_success(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("AA agent live call failed; provider_error=", text)
        self.assertIn("exit 1", text)
        self.assertIn("Attest real AA task completion", text)
        self.assertNotIn("continue-on-error: true", text)


if __name__ == "__main__":
    unittest.main()
