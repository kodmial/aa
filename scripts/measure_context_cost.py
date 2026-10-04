#!/usr/bin/env python3
"""Measure RU/EN context cost on the pinned OpenCode/Zen runtime (issue #49).

The harness sends controlled requests that differ only by language payload
(same wrappers, model, agent and reasoning settings) against a local
``opencode serve`` instance and captures provider/runtime-reported
input-token usage. Every probe uses a fresh session, so per-case deltas
against the shared baseline isolate the payload's token cost.

Method notes (fixed 2026-10-04):
- The stable prompt-side metric is ``input + cache.read + cache.write``:
  the provider splits identical prompt costs between ``input`` and cache
  fields nondeterministically, while the sum is exactly stable per
  payload. Median and p95 are reported across repeated runs.
- The default agent is the server default (``build`` path). The dedicated
  ``aa`` agent is denied-by-default (``"*": "deny"``) and the pinned
  Muse Spark free tier rejects that combination with a 403
  ``FreeTierError``; measurement therefore uses the default agent on the
  identical pinned model/tokenizer path and keeps the system prompt as a
  measured payload instead.
- No generic tokenizer is used anywhere: all numbers are runtime-reported.

Usage:
    opencode serve --port 4096 --hostname 127.0.0.1  # pinned 1.18.34
    python3 scripts/measure_context_cost.py --json-out /tmp/cost.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.token_measurement import (  # noqa: E402
    ALIGNED_PAIRS,
    PINNED_MODEL,
    RUSSIAN_HISTORIES,
    TURN_BROAD_MIXED,
    build_turn,
    collect_case,
    describe_case,
)

FALLBACK_MODEL = "opencode/space-bunny-free"
REQUEST_TIMEOUT = 180.0


def _request(base_url: str, method: str, path: str, body: dict[str, Any] | None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"runtime request failed: {method} {path} http={exc.code}") from exc
    except OSError as exc:
        raise RuntimeError(f"runtime is unreachable at {base_url}: {exc}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _split_model(model: str) -> dict[str, str]:
    provider, _, model_id = model.partition("/")
    if not provider or not model_id:
        raise ValueError(f"model must look like 'provider/model': {model!r}")
    return {"providerID": provider, "modelID": model_id}


class RuntimeMeasurer:
    """Fresh-session prober for the pinned runtime message boundary."""

    def __init__(self, base_url: str, *, model: str, agent: str) -> None:
        self._base_url = base_url
        self._model = _split_model(model)
        self._agent = agent
        self.version = ""

    def check_health(self) -> str:
        """Verify readiness and return the server version (fail closed)."""
        payload = _request(self._base_url, "GET", "/global/health", None)
        if not isinstance(payload, dict) or payload.get("healthy") is not True:
            raise RuntimeError("opencode runtime is not healthy")
        version = payload.get("version")
        self.version = version if isinstance(version, str) else ""
        return self.version

    def send_once(self, text: str, *, case: str) -> dict[str, Any]:
        """Send one probe on a fresh session; return raw ``info.tokens``."""
        created = _request(self._base_url, "POST", "/session", {"title": f"measure-{case}"})
        if not isinstance(created, dict) or not created.get("id"):
            raise RuntimeError("session creation returned no id")
        session_id = urllib.parse.quote(str(created["id"]), safe="")
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if self._agent:
            body["agent"] = self._agent
        body["model"] = self._model
        response = _request(self._base_url, "POST", f"/session/{session_id}/message", body)
        if not isinstance(response, dict) or not isinstance(response.get("info"), dict):
            raise RuntimeError("prompt returned an unexpected payload")
        info = response["info"]
        error = info.get("error")
        if error not in (None, False):
            detail = error.get("data", {}) if isinstance(error, dict) else {}
            raise RuntimeError(
                f"provider error for case {case!r}: "
                f"status={detail.get('statusCode', 'unknown')} "
                f"message={detail.get('message', 'withheld')!r}"
            )
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            raise RuntimeError("prompt returned no token accounting")
        return tokens


def _cases(system_full_en: str) -> list[tuple[str, str, str]]:
    history_short = RUSSIAN_HISTORIES["history_short"]
    history_medium = RUSSIAN_HISTORIES["history_medium"]
    history_long = RUSSIAN_HISTORIES["history_long"]
    evidence_en = ALIGNED_PAIRS["evidence_pack"]["en"]
    book_map_en = ALIGNED_PAIRS["book_map"]["en"]
    first_passage = "\n".join(evidence_en.splitlines()[:5])
    return [
        ("baseline", "mixed", build_turn("(none)")),
        ("system_full_en", "en", build_turn(system_full_en)),
        ("system_sample_en", "en", build_turn(ALIGNED_PAIRS["system_sample"]["en"])),
        ("system_sample_ru", "ru", build_turn(ALIGNED_PAIRS["system_sample"]["ru"])),
        ("book_map_en", "en", build_turn(ALIGNED_PAIRS["book_map"]["en"])),
        ("book_map_ru", "ru", build_turn(ALIGNED_PAIRS["book_map"]["ru"])),
        ("evidence_pack_en", "en", build_turn(ALIGNED_PAIRS["evidence_pack"]["en"])),
        ("evidence_pack_ru", "ru", build_turn(ALIGNED_PAIRS["evidence_pack"]["ru"])),
        ("history_short_ru", "ru", build_turn(history_short)),
        ("history_medium_ru", "ru", build_turn(history_medium)),
        ("history_long_ru", "ru", build_turn(history_long)),
        ("planner_wrapper_en", "en", build_turn(ALIGNED_PAIRS["planner_wrapper"]["en"])),
        ("planner_wrapper_ru", "ru", build_turn(ALIGNED_PAIRS["planner_wrapper"]["ru"])),
        ("turn_short", "mixed", build_turn(history_short, first_passage)),
        ("turn_medium", "mixed", build_turn(history_medium, book_map_en, evidence_en)),
        ("turn_broad", "mixed", build_turn(history_long, TURN_BROAD_MIXED)),
    ]


def _fallback_cases() -> list[tuple[str, str, str]]:
    return [
        ("baseline", "mixed", build_turn("(none)")),
        ("book_map_en", "en", build_turn(ALIGNED_PAIRS["book_map"]["en"])),
        ("book_map_ru", "ru", build_turn(ALIGNED_PAIRS["book_map"]["ru"])),
        ("evidence_pack_en", "en", build_turn(ALIGNED_PAIRS["evidence_pack"]["en"])),
        ("evidence_pack_ru", "ru", build_turn(ALIGNED_PAIRS["evidence_pack"]["ru"])),
    ]


def run_suite(
    measurer: RuntimeMeasurer,
    cases: list[tuple[str, str, str]],
    *,
    repeats: int,
    baseline_median: float | None = None,
    baseline_chars: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Measure every case; return (results, baseline median)."""
    if baseline_median is None:
        name, _, payload = cases[0]
        assert name == "baseline"
        costs = collect_case(
            lambda text: measurer.send_once(text, case=name), payload, repeats=repeats
        )
        baseline = describe_case(
            name=name,
            language="mixed",
            payload=payload,
            costs=costs,
            baseline_median=0.0,
            baseline_chars=len(payload),
        )
        results = [
            {
                "name": baseline.name,
                "language": baseline.language,
                "chars": baseline.chars,
                "runs": list(baseline.runs),
                "median": baseline.stats.median,
                "p95": baseline.stats.p95,
                "min": baseline.stats.minimum,
                "max": baseline.stats.maximum,
                "delta_tokens": 0.0,
                "chars_per_token": None,
            }
        ]
        baseline_median = baseline.stats.median
        baseline_chars = baseline.chars
        rest = cases[1:]
    else:
        results = []
        rest = cases
        assert baseline_chars is not None
    for name, language, payload in rest:
        costs = collect_case(
            lambda text, case=name: measurer.send_once(text, case=case),
            payload,
            repeats=repeats,
        )
        described = describe_case(
            name=name,
            language=language,
            payload=payload,
            costs=costs,
            baseline_median=baseline_median,
            baseline_chars=baseline_chars,
        )
        results.append(
            {
                "name": described.name,
                "language": described.language,
                "chars": described.chars,
                "runs": list(described.runs),
                "median": described.stats.median,
                "p95": described.stats.p95,
                "min": described.stats.minimum,
                "max": described.stats.maximum,
                "delta_tokens": described.delta_tokens,
                "chars_per_token": described.chars_per_token,
            }
        )
        print(
            f"measured {name}: chars={described.chars} "
            f"median={described.stats.median} p95={described.stats.p95}"
        )
    assert baseline_median is not None
    return results, baseline_median


def render_markdown(report: dict[str, Any]) -> str:
    """Render the measurement report as Markdown (no prompt text logged)."""
    lines = [
        f"# Context-cost measurement ({report['measured_at']})",
        "",
        f"Runtime: OpenCode {report['runtime']['version']} "
        f"(pinned {report['runtime']['pinned_version']})",
        f"Model: `{report['runtime']['model']}` "
        f"Agent: `{report['runtime']['agent'] or 'server default'}` "
        f"Repeats: {report['runtime']['repeats']}",
        f"Baseline effective input: {report['baseline_median']} tokens",
        "",
        "| Case | Lang | Chars | Median | p95 | Delta | Chars/token |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        per_token = f"{case['chars_per_token']:.2f}" if case["chars_per_token"] else "n/a"
        lines.append(
            f"| {case['name']} | {case['language']} | {case['chars']} "
            f"| {case['median']} | {case['p95']} | {case['delta_tokens']} | {per_token} |"
        )
    if report.get("fallback"):
        lines += [
            "",
            f"## Fallback spot-check (`{report['fallback']['model']}`)",
            "",
            "| Case | Lang | Chars | Median | p95 | Delta | Chars/token |",
            "|---|---|---|---|---|---|---|",
        ]
        for case in report["fallback"]["cases"]:
            per_token = f"{case['chars_per_token']:.2f}" if case["chars_per_token"] else "n/a"
            lines.append(
                f"| {case['name']} | {case['language']} | {case['chars']} "
                f"| {case['median']} | {case['p95']} | {case['delta_tokens']} "
                f"| {per_token} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: measure, then emit JSON and Markdown."""
    parser = argparse.ArgumentParser(description="Measure RU/EN context cost.")
    parser.add_argument("--base-url", default="http://127.0.0.1:4096")
    parser.add_argument("--model", default=PINNED_MODEL)
    parser.add_argument("--agent", default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        print("repeats must be >= 1", file=sys.stderr)
        return 2

    system_path = ROOT / "prompts" / "aa-agent-system.md"
    try:
        system_full_en = system_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        print(f"authoritative system prompt is missing: {exc}", file=sys.stderr)
        return 1

    measurer = RuntimeMeasurer(args.base_url, model=args.model, agent=args.agent)
    try:
        version = measurer.check_health()
    except RuntimeError as exc:
        print(f"runtime check failed: {exc}", file=sys.stderr)
        return 1
    print(f"runtime healthy: OpenCode {version}")

    try:
        cases = _cases(system_full_en)
        results, baseline = run_suite(measurer, cases, repeats=args.repeats)
        report: dict[str, Any] = {
            "measured_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "runtime": {
                "version": version,
                "pinned_version": "1.18.34",
                "model": args.model,
                "agent": args.agent,
                "repeats": args.repeats,
            },
            "baseline_median": baseline,
            "cases": results,
        }
        if not args.no_fallback:
            fallback = RuntimeMeasurer(args.base_url, model=args.fallback_model, agent=args.agent)
            fallback.check_health()
            fb_results, fb_baseline = run_suite(
                fallback, _fallback_cases(), repeats=min(args.repeats, 3)
            )
            report["fallback"] = {
                "model": args.fallback_model,
                "baseline_median": fb_baseline,
                "cases": fb_results,
            }
    except RuntimeError as exc:
        print(f"measurement failed: {exc}", file=sys.stderr)
        return 1

    markdown = render_markdown(report)
    print(markdown)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
