# AA project agent instructions

The GitHub issue and its Definition of Done are the task specification. Keep changes scoped to that task.

## Environment

The project uses Python 3.12+. If development dependencies are not installed, install them with:

```sh
python3 -m pip install -e ".[dev]"
```

The local OpenCode CLI is required by the boot check and is provided by the Continuum issue workflow.

## Mandatory verification

After the final repository edit, run the complete canonical verification suite from the repository root:

```sh
bash scripts/verify.sh
```

This command is the same verification entry point used by CI.

Do not substitute a narrower command. In particular, `mypy src` is not equivalent to the required `mypy src tests`.

Do not claim that verification passed unless `bash scripts/verify.sh` completed successfully after the final edit. If a required check cannot run, report the exact limitation instead of claiming success.

Focused checks may be used during development, but they do not replace the final full verification.

## Safety and privacy

Never log or commit Telegram message bodies, credentials, private keys, or plaintext book artifacts that repository policy keeps out of Git.
