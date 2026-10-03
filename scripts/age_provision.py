#!/usr/bin/env python3
"""One-time production age key provisioning for issue #24 (run locally only).

This tool generates a fresh X25519 age keypair for encrypting the canonical
book snapshot. It must be executed once by the repository owner on a trusted
local machine — never inside an ordinary repository workflow — because the
private identity must live only in the ``AA_BOOK_AGE_IDENTITY`` GitHub Actions
repository secret and must never be committed, cached, or printed into CI logs.

Procedure (production activation is tracked separately in #28):

1. Run locally: ``python3 scripts/age_provision.py --write-recipient
   corpus/source/encrypted/recipient.txt``.
2. Copy the printed ``AGE-SECRET-KEY-...`` value into the repository secret
   named ``AA_BOOK_AGE_IDENTITY`` (Settings -> Secrets -> Actions).
3. Commit only ``corpus/source/encrypted/recipient.txt`` (the public
   ``age1...`` recipient). Never commit the printed identity.
4. Run the manually dispatched ``encrypted-corpus-refresh`` workflow to
   publish the production encrypted snapshot.

The script prints the private identity to stdout exactly once so the operator
can store it in the secret. It never writes the identity to the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa.corpus.age_v1 import generate_identity, parse_identity, parse_recipient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-recipient",
        type=Path,
        default=None,
        help="Write the public recipient to this path "
        "(e.g. corpus/source/encrypted/recipient.txt).",
    )
    args = parser.parse_args(argv)

    identity, recipient = generate_identity()
    # Self-check: the pair must parse before anything is shown or written.
    parse_identity(identity)
    parse_recipient(recipient)

    if args.write_recipient is not None:
        args.write_recipient.parent.mkdir(parents=True, exist_ok=True)
        args.write_recipient.write_text(recipient + "\n", encoding="utf-8")

    print("Public recipient (safe to commit as corpus/source/encrypted/recipient.txt):")
    print(recipient)
    print()
    print("Private identity (store ONCE as the AA_BOOK_AGE_IDENTITY secret, then delete):")
    print(identity)
    print()
    print(
        "Next: set the AA_BOOK_AGE_IDENTITY repository secret to the identity above, "
        "commit only the recipient, and dispatch the encrypted-corpus-refresh workflow. "
        "Production activation is tracked in #28."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
