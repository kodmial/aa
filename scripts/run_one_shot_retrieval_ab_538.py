#!/usr/bin/env python3
"""Synthetic packing/coverage harness for kodmial/aa#538.

This is deliberately a harness, not a real-corpus retrieval A/B. It runs
the real ``aretrieve_with_semantic_selection`` orchestration over a frozen
synthetic fixture with the feature flag off and on, but local discovery is
replaced by the fixture's fused-candidate map and the provider transport is
an in-process scripted model. It therefore exercises canonical reading,
prompt packing, strict quote validation, fallback handling and scripted
call accounting; it does not measure BM25+E5/FAISS/RRF discovery recall or
real provider HTTP counts, tokens or latency. Those require the production
index/corpus and provider qualification lane.

The fixture contains synthetic text only. Report output carries IDs,
counts, scripted token estimates and harness latencies, never prompt or
passage text. Metrics are labelled synthetic: ``retrieval_recall`` is only
the recall of expected fixture chunk ids in the served pack, and
``scripted_http_calls`` mirrors the adapter counter through a scripted
provider, not real HTTP.

Usage:
    python3 scripts/run_one_shot_retrieval_ab_538.py \
        --json-out /tmp/one-shot-ab-538.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.conversation.conversation_context import InformationNeed  # noqa: E402
from aa.conversation.model_adapter import (  # noqa: E402
    clear_omitted_structured_cache,
    clear_primary_circuit,
    llm_http_call_count,
    note_llm_http_call,
    reset_llm_http_call_count,
)
from aa.conversation.retrieval_node import (  # noqa: E402
    aretrieve_with_semantic_selection,
    clear_canonical_read_cache,
)
from aa.corpus.budget import estimate_text_tokens  # noqa: E402
from aa.opencode.errors import OpenCodeRateLimitError  # noqa: E402
from aa.retrieval.evidence import RetrievalConfig  # noqa: E402
from aa.retrieval.fusion import FusedCandidate  # noqa: E402
from aa.retrieval.index import ChunkRecord  # noqa: E402

DEFAULT_FIXTURE = ROOT / "qualification" / "one_shot_retrieval_ab_538.v1.json"
DEFAULT_JSON_OUT = ROOT / "eval-one-shot-ab-538.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(chunk: dict[str, Any], corpus: dict[str, Any], start: int) -> ChunkRecord:
    text = str(chunk["text"])
    return ChunkRecord(
        chunk_id=str(chunk["chunk_id"]),
        logical_chunk_id=str(chunk["chunk_id"]),
        section=str(chunk.get("section", "chapter-1")),
        book="aa-big-book",
        parent="p-" + str(chunk.get("section", "chapter-1")),
        prev=None,
        next=None,
        source_id=str(corpus["source_id"]),
        source_file="synthetic/one-shot-ab-538.txt",
        source_sha256=str(corpus["source_sha256"]),
        char_start=start,
        char_end=start + len(text),
        text_sha256=_sha(text),
        text=text,
        corpus_version=str(corpus["corpus_version"]),
    )


def _build_index(case: dict[str, Any], corpus: dict[str, Any]) -> Any:
    records: list[ChunkRecord] = []
    start = 0
    for chunk in case["chunks"]:
        record = _record(chunk, corpus, start)
        records.append(record)
        start = record.char_end
    return SimpleNamespace(
        chunks={record.chunk_id: record for record in records},
        metadata={
            "ru_artifact_sha256": str(corpus["corpus_version"]),
            "embedding_backend": "hashing",
            "embedding_dim": 64,
        },
        dense=None,
        lexical_conn=None,
        ram_resident=True,
    )


def _fused(case: dict[str, Any], index: Any) -> dict[str, FusedCandidate]:
    scores = dict(case.get("fused_scores", {}))
    return {
        cid: FusedCandidate(
            chunk_id=cid,
            fused_score=float(scores.get(cid, 0.0)),
            lexical_rank=pos + 1,
            dense_rank=pos + 1,
            lexical_score=float(scores.get(cid, 0.0)),
            dense_score=float(scores.get(cid, 0.0)),
        )
        for pos, cid in enumerate(index.chunks)
    }


def _passage_texts(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in re.finditer(r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL):
        out[match.group(1)] = " ".join(match.group(2).split())
    return out


class _ScriptedProvider:
    """Deterministic in-process provider; calls are scripted, not HTTP.

    ``note_llm_http_call`` mirrors the production adapter boundary so the
    harness can exercise retrieval call accounting, but no network request
    or real token usage occurs.
    """

    def __init__(self, case: dict[str, Any]) -> None:
        self._case = case
        self._markers = {str(item["need_id"]): str(item["text"]) for item in case.get("needs", [])}
        self.logical_calls = 0
        self.input_tokens = 0
        self.selection_calls = 0
        self.coverage_calls = 0

    async def ainvoke_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = 1,
    ) -> dict[str, object]:
        _ = (schema, retry_count)
        note_llm_http_call()
        self.logical_calls += 1
        self.input_tokens += estimate_text_tokens(system) + estimate_text_tokens(prompt)
        if "<passage id=" in prompt and "<original_request>" in prompt:
            self.coverage_calls += 1
            if bool(self._case.get("provider_error")):
                raise OpenCodeRateLimitError("http=429")
            texts = _passage_texts(prompt)
            needs: list[dict[str, Any]] = []
            for need_id, marker in self._markers.items():
                supporting = [pid for pid, body in texts.items() if marker in body]
                needs.append(
                    {
                        "need_id": need_id,
                        "covered": bool(supporting),
                        "supporting_passage_ids": supporting,
                        "supporting_quote": marker if supporting else "",
                        "missing": "" if supporting else f"no passage states {marker}",
                    }
                )
            return {"needs": needs}
        if "<candidate id=" in prompt:
            self.selection_calls += 1
            return {
                "selected_chunk_ids": [str(chunk["chunk_id"]) for chunk in self._case["chunks"]],
                "need_more_detail": False,
                "followup_queries": [],
            }
        raise AssertionError("unexpected scripted prompt shape")


def _covered_ids(metadata: dict[str, Any]) -> list[str]:
    covered: list[str] = []
    for entry in list(metadata.get("coverage_per_need") or []):
        if bool(entry.get("covered")):
            covered.append(str(entry.get("need_id", "")))
    return sorted(item for item in covered if item)


async def _run_case(case: dict[str, Any], corpus: dict[str, Any], mode: str) -> dict[str, Any]:
    index = _build_index(case, corpus)
    fused = _fused(case, index)
    provider = _ScriptedProvider(case)
    needs = [
        InformationNeed(need_id=str(item["need_id"]), text=str(item["text"]))
        for item in case.get("needs", [])
    ]

    # Harness boundary: discovery is the fixture's frozen fused map. The
    # real local BM25+E5/FAISS/RRF stack is not executed here, so recall
    # and source metrics below are fixture assertions, not discovery proof.
    def _fixture_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(fused)])

    def _fixture_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, per_ids, kwargs)
        return dict(fused), list(fused)

    clear_canonical_read_cache()
    clear_primary_circuit()
    clear_omitted_structured_cache()
    reset_llm_http_call_count()
    started = time.perf_counter()
    metadata: dict[str, Any] = {}
    error_type = ""
    served_chunk_ids: list[str] = []
    with ExitStack() as stack:
        from aa.retrieval import evidence as evidence_mod

        stack.enter_context(patch.object(evidence_mod, "run_branch_searches", _fixture_branch))
        stack.enter_context(patch.object(evidence_mod, "fuse_query_pool", _fixture_fuse))
        try:
            pack = await aretrieve_with_semantic_selection(
                index,
                list(case.get("queries") or [case.get("resolved_intent", "")]),
                config=RetrievalConfig(one_shot_fast_path=(mode == "on")),
                resolved_intent=str(case.get("resolved_intent", "")),
                conversation_context="",
                user_message=str(case.get("user_message", "")),
                selection_model=provider,
                information_needs=needs,
                max_iterations=3,
            )
            metadata = dict(pack.retrieval_metadata or {})
            served_chunk_ids = sorted(
                {
                    str(cid)
                    for passage in list(pack.passages or [])
                    for cid in list(getattr(passage, "child_chunk_ids", ()) or ())
                    if str(cid)
                }
            )
        except OpenCodeRateLimitError as exc:
            error_type = type(exc).__name__
        except Exception as exc:  # pragma: no cover - diagnostics only
            error_type = type(exc).__name__
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    expected = sorted(str(item) for item in case.get("expected_covered", []))
    covered = _covered_ids(metadata)
    coverage_match = None if error_type else covered == expected
    expected_chunks = sorted(str(item) for item in case.get("expected_chunk_ids", []))
    served_set = set(served_chunk_ids)
    expected_set = set(expected_chunks)
    # Empty expected sets and provider-error rows are not recall evidence.
    # Keeping them out of the denominator prevents an error that retrieved
    # nothing from contributing a perfect score to the aggregate.
    source_match = expected_set.issubset(served_set) if expected_set else None
    retrieval_recall = (
        round(len(expected_set & served_set) / len(expected_set), 4)
        if expected_set and not error_type
        else None
    )
    provider_error_case = bool(case.get("provider_error"))
    safe_outcome = (
        bool(error_type) and str(metadata.get("coverage_status", "")) != "ready"
        if provider_error_case
        else (not error_type and coverage_match is not False)
    )
    return {
        "case_id": str(case["case_id"]),
        "split": str(case["split"]),
        "category": str(case["category"]),
        "mode": mode,
        "error_type": error_type,
        "retrieval_mode": str(metadata.get("retrieval_mode", "")),
        "fastpath_attempted": bool(metadata.get("fastpath_attempted", False)),
        "fallback_reason": str(metadata.get("fallback_reason", "")),
        "coverage_status": str(metadata.get("coverage_status", "")),
        "covered_need_ids": covered,
        "expected_covered_need_ids": expected,
        "coverage_match": coverage_match,
        "expected_chunk_ids": expected_chunks,
        "served_chunk_ids": served_chunk_ids,
        "source_match": source_match,
        "retrieval_recall": retrieval_recall,
        "actual_needs_coverage_match": coverage_match,
        "safe_outcome": safe_outcome,
        "logical_model_calls": int(provider.logical_calls),
        # Scripted logical calls at the adapter accounting boundary, not
        # real provider HTTP requests.
        "scripted_http_calls": int(llm_http_call_count()),
        "input_tokens": int(provider.input_tokens),
        "input_tokens_kind": "scripted-estimate",
        "latency_ms": round(
            float(metadata.get("total_ms", elapsed_ms) or elapsed_ms),
            3,
        ),
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    pos = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return round(float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight), 3)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row["error_type"]]
    compared = [row for row in rows if row["coverage_match"] is not None]
    recall_rows = [row for row in rows if row.get("retrieval_recall") is not None]
    source_rows = [row for row in rows if row.get("source_match") is not None]
    latencies = [float(row["latency_ms"]) for row in rows]
    fallback_rows = [row for row in rows if row["fallback_reason"] or row["error_type"]]
    return {
        "cases": len(rows),
        "errors": len(errors),
        "coverage_matches": sum(1 for row in compared if row["coverage_match"]),
        "coverage_compared": len(compared),
        "source_matches": sum(1 for row in source_rows if row["source_match"]),
        "source_compared": len(source_rows),
        "source_unavailable": len(rows) - len(source_rows),
        "retrieval_recall": round(
            sum(float(row["retrieval_recall"]) for row in recall_rows) / len(recall_rows), 4
        )
        if recall_rows
        else None,
        "retrieval_recall_compared": len(recall_rows),
        "retrieval_recall_skipped": len(rows) - len(recall_rows),
        "safe_outcomes": sum(1 for row in rows if row["safe_outcome"]),
        "ready_cases": sum(1 for row in rows if row["coverage_status"] == "ready"),
        "fallback_cases": len(fallback_rows),
        "fallback_scripted_http_calls": sum(
            int(row["scripted_http_calls"]) for row in fallback_rows
        ),
        "logical_model_calls": sum(int(row["logical_model_calls"]) for row in rows),
        "scripted_http_calls": sum(int(row["scripted_http_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "latency_ms": round(sum(latencies), 3),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
    }


def run_ab(fixture_path: Path, split: str = "all") -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    corpus = dict(fixture["corpus"])
    cases = [
        case
        for case in fixture.get("cases", [])
        if split == "all" or str(case.get("split")) == split
    ]
    if not cases:
        raise SystemExit(f"no cases for split {split!r}")

    os.environ.pop("AA_RETRIEVAL_ONE_SHOT_FASTPATH", None)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in ("off", "on"):
            rows.append(asyncio.run(_run_case(case, corpus, mode)))

    by_category: dict[str, dict[str, Any]] = {}
    by_split: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = row["category"]
        split_name = row["split"]
        by_category.setdefault(category, {}).setdefault(row["mode"], []).append(row)
        by_split.setdefault(split_name, {}).setdefault(row["mode"], []).append(row)

    def _summaries(bucket: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
        return {
            key: {mode: _aggregate(items) for mode, items in modes.items()}
            for key, modes in bucket.items()
        }

    return {
        "fixture": str(fixture_path),
        "fixture_version": str(fixture.get("version", "")),
        "split": split,
        "harness": "synthetic-packing-coverage",
        "real_corpus_index_model_config": False,
        "same_fixture_corpus_index_model_config": True,
        # Retained for existing consumers: both modes use the same
        # synthetic fixture index/model configuration, never a real corpus.
        "same_corpus_index_model_config": True,
        "discovery": "fixture-fused-candidates",
        "provider_transport": "scripted-in-process",
        "http_call_metric": "scripted-logical-calls",
        "recall_metric": "fixture-expected-chunk-recall",
        "results": rows,
        "summary": {
            "by_category": _summaries(by_category),
            "by_split": _summaries(by_split),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--split", choices=("all", "frozen", "held_out"), default="all")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args(argv)

    report = run_ab(args.fixture, args.split)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
