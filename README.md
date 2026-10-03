# AA

Public repository for the AA Telegram bot.

Continuum core is installed from and intentionally tracks `kodmial/continuum@main`.
Application bootstrap is handled by Work Lock #47.

## AA Telegram worker foundation

Minimal Python 3.12+ asyncio worker intended to run in the same GitHub
Actions job as OpenCode. This foundation provides configuration, lifecycle,
privacy-safe logging, and module boundaries only — no live Telegram calls
and no live OpenCode calls yet.

### Layout

- `src/aa/config.py` — environment-backed configuration.
- `src/aa/logging.py` — privacy-safe structured (JSON) logging with secret redaction.
- `src/aa/app.py` — async application lifecycle (`start`/`stop`/`run`).
- `src/aa/__main__.py` — `python -m aa` entrypoint.
- `src/aa/telegram/` — Telegram transport boundary.
- `src/aa/opencode/` — OpenCode session/runtime integration boundary.
- `src/aa/corpus/` — AA corpus/context boundary.
- `src/aa/sessions/` — per-chat session coordination.
- `src/aa/safety/` — safety routing.
- `src/aa/control/` — runtime control (session duration/deadline).
- `tests/` — pytest harness (lifecycle, config, privacy, boundaries).
- `.github/workflows/ci.yml` — CI checks (boot, tests, ruff, mypy).

The worker must not own Zen credentials and must not implement a second LLM
client. Model/context/output limits are pointers owned by the OpenCode
runtime (`OPENCODE_MODEL`, `OPENCODE_CONTEXT_LIMIT_TOKENS`,
`OPENCODE_MAX_OUTPUT_TOKENS`).

### Configuration

See `.env.example` for the reserved names:

- `TELEGRAM_BOT_TOKEN`
- `OPENCODE_BASE_URL`, `OPENCODE_COMMAND`, `OPENCODE_WORKDIR`
- `BOT_SESSION_DURATION_SECONDS`
- `AA_CORPUS_PATH`, `AA_CORPUS_VERSION`
- `OPENCODE_MODEL`, `OPENCODE_CONTEXT_LIMIT_TOKENS`, `OPENCODE_MAX_OUTPUT_TOKENS`
- `LOG_LEVEL`

### Local development

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in local values; never commit secrets
python -m aa --check
```

Run the worker (requires `TELEGRAM_BOT_TOKEN`):

```bash
python -m aa
```

### Test commands

```bash
pytest -q
python -m aa --check
ruff check .
ruff format --check .
mypy src tests
```
