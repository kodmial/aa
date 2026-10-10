"""P0 turn-budget separation (kodmial/aa#335).

The 5s warm-local-RRF value is a diagnostic target for the local index
stage only. Semantic selector/coverage model awaits are bounded by the
graph-owned end-to-end turn budget instead:

- local 0.4s + selector 4s + coverage 4s (total far below the turn
  bound) must reach ``ready`` and never be forcibly exhausted by the
  local threshold; no fake adequate marks, no bypass of the final
  certificate path (coverage still model-verified over exact text);
- a model slower than the actual remaining turn deadline yields a typed
  ``provider_semantic_timeout`` with no delivery, no optimistic
  history, no extra retries and no leaked worker;
- an RRF stage over its 5s diagnostic target raises the measurable
  ``local_retrieval_slow`` flag without misclassifying adequate LLM
  evidence;
- budget constants stay pinned to the authoritative unchanged Gate E
  policies (hard max 120s, p95 target 60s, graph turn budget 105s).
"""

from __future__ import annotations

import asyncio
import hashlib
import time as _time
from types import SimpleNamespace
from typing import Any


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Текст про поддержку.",
    parent: str | None = None,
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=(parent or ("p-" + chunk_id)),
        prev=None,
        next=None,
        source_id="ru-fourth-edition-txt",
        source_file="corpus/source/raw-ru/aa-big-book.txt",
        source_sha256="s" * 64,
        char_start=start,
        char_end=end,
        text_sha256=_sha(text),
        text=text,
        corpus_version="ru-v1",
    )


def _fake_index(records: list[Any]) -> Any:
    return SimpleNamespace(
        chunks={record.chunk_id: record for record in records},
        metadata={
            "ru_artifact_sha256": "r" * 64,
            "embedding_backend": "hashing",
            "embedding_dim": 64,
        },
        dense=None,
        lexical_conn=None,
        ram_resident=True,
    )


def _fused(chunk_id: str, score: float) -> Any:
    from aa.retrieval.fusion import FusedCandidate

    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


class _DelayedScriptedModel:
    """Serve scripted selections; judge coverage over exact passages."""

    def __init__(
        self,
        selections: list[dict[str, Any]],
        markers: dict[str, str],
        *,
        selection_delay_s: float = 0.0,
        coverage_delay_s: float = 0.0,
    ) -> None:
        self._selections = list(selections)
        self._markers = dict(markers)
        self._selection_delay = float(selection_delay_s)
        self._coverage_delay = float(coverage_delay_s)
        self.selection_calls = 0
        self.coverage_calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        import re as _re

        _ = (system, schema, retry_count)
        if "<passages>" in prompt and "<original_request>" in prompt:
            self.coverage_calls += 1
            if self._coverage_delay > 0:
                await asyncio.sleep(self._coverage_delay)
            passage_texts: dict[str, str] = {}
            for match in _re.finditer(
                r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, _re.DOTALL
            ):
                pid, body = match.group(1), match.group(2)
                passage_texts[pid] = " ".join(body.split())
            needs: list[dict[str, object]] = []
            for nid, marker in self._markers.items():
                supporting = [pid for pid, body in passage_texts.items() if marker in body]
                if supporting:
                    needs.append(
                        {
                            "need_id": nid,
                            "covered": True,
                            "supporting_passage_ids": supporting,
                            "supporting_quote": marker,
                            "missing": "",
                        }
                    )
                else:
                    needs.append(
                        {
                            "need_id": nid,
                            "covered": False,
                            "supporting_passage_ids": [],
                            "supporting_quote": "",
                            "missing": f"no passage states {marker}",
                        }
                    )
            return {"needs": needs}
        self.selection_calls += 1
        if self._selection_delay > 0:
            await asyncio.sleep(self._selection_delay)
        if self._selections:
            return dict(self._selections.pop(0))
        return {
            "selected_chunk_ids": [],
            "need_more_detail": False,
            "followup_queries": [],
        }


def _install_branch(
    monkeypatch: Any,
    pools: list[dict[str, Any]],
    per_ids: list[list[str]],
    *,
    local_delay_s: float = 0.0,
) -> dict[str, Any]:
    import aa.retrieval.evidence as evidence_mod

    calls = {"count": 0}

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        if local_delay_s > 0:
            import time as _sync_time

            _sync_time.sleep(local_delay_s)
        return ([], [list(ids) for ids in per_ids])

    def _fake_fuse(ranked: Any, _per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, _per_ids, kwargs)
        pos = min(calls["count"], len(pools) - 1)
        calls["count"] += 1
        fused = dict(pools[pos])
        return fused, list(fused)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)
    return calls


def test_budget_constants_track_authoritative_policies() -> None:
    """Turn-budget constants mirror the unchanged Gate E/product policies."""
    from aa.conversation import turn_budget as budget_mod
    from aa.conversation import turn_pipeline as pipeline_mod
    from aa.qualification import self_proving as proving_mod
    from aa.retrieval import evidence as evidence_mod

    assert budget_mod.LOCAL_RRF_DIAGNOSTIC_BUDGET_MS == 5000.0
    assert evidence_mod.INTERACTIVE_LATENCY_BUDGET_MS == 5000.0
    assert budget_mod.LOCAL_RRF_DIAGNOSTIC_BUDGET_MS == float(
        evidence_mod.INTERACTIVE_LATENCY_BUDGET_MS
    )
    assert budget_mod.TURN_END_TO_END_BUDGET_S == float(pipeline_mod.TURN_END_TO_END_BUDGET_S)
    assert budget_mod.TURN_END_TO_END_BUDGET_S == 105.0
    assert budget_mod.TURN_HARD_SLO_MS == float(proving_mod.ORDINARY_TURN_BUDGET_MS)
    assert budget_mod.TURN_HARD_SLO_MS == 120_000.0
    assert budget_mod.TURN_P95_TARGET_MS == float(proving_mod.P95_TARGET_MS)
    assert budget_mod.TURN_P95_TARGET_MS == 60_000.0


def test_turn_budget_allocates_remaining_end_to_end() -> None:
    """One model await receives remaining turn budget minus reserve."""
    from aa.conversation.turn_budget import (
        DOWNSTREAM_MIN_RESERVE_S,
        TurnBudgetExpired,
        new_turn_budget,
    )

    budget = new_turn_budget()
    allocated = budget.allocate_semantic_timeout_s()
    assert allocated > 60.0
    assert allocated <= 105.0
    assert (
        allocated == budget.remaining_s() - DOWNSTREAM_MIN_RESERVE_S
        or abs(allocated - (budget.remaining_s() - DOWNSTREAM_MIN_RESERVE_S)) < 0.05
    )
    spent = new_turn_budget(deadline_s=0.2)
    try:
        spent.allocate_semantic_timeout_s()
    except TurnBudgetExpired:
        pass
    else:
        raise AssertionError("spent budget must raise TurnBudgetExpired")


async def test_local_plus_semantic_within_turn_bound_reaches_ready(
    monkeypatch: Any,
) -> None:
    """0.4s local + 4s selector + 4s coverage stays ready (never 5s-exhausted)."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Утром помогает разбор АЛЬФА-МАРКЕР рядом.")
    second = _record("chapter-2:ru:second", start=0, text="Вечером помогает шаг БЕТА-МАРКЕР рядом.")
    index = _fake_index([first, second])
    _install_branch(
        monkeypatch,
        [
            {
                first.chunk_id: _fused(first.chunk_id, 50.0),
                second.chunk_id: _fused(second.chunk_id, 49.0),
            }
        ],
        [[first.chunk_id, second.chunk_id]],
        local_delay_s=0.4,
    )
    model = _DelayedScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id, second.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={"need-1": "АЛЬФА-МАРКЕР", "need-2": "БЕТА-МАРКЕР"},
        selection_delay_s=4.0,
        coverage_delay_s=4.0,
    )
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги", "вечернее беспокойство шаги"],
        resolved_intent="как разбирать утреннюю тягу и вечернее беспокойство",
        conversation_context="",
        user_message="как разбирать утреннюю тягу и вечернее беспокойство",
        selection_model=model,
        information_needs=[
            {"need_id": "need-1", "text": "утренний разбор тяги"},
            {"need_id": "need-2", "text": "вечернее беспокойство"},
        ],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    assert metadata.get("coverage_exhaustion_reason") in ("", None)
    assert metadata.get("typed_stage") in ("", None)
    assert metadata.get("coverage_all_covered") is True
    assert metadata.get("coverage_used_model") is True
    # Genuine evidence over exact canonical text (no fake adequate marks).
    served = " ".join(str(item.exact_text) for item in pack.passages)
    assert "АЛЬФА-МАРКЕР" in served and "БЕТА-МАРКЕР" in served
    # Local diagnostic measured but not a deadline: 0.4s is not slow and
    # the turn still holds ample remaining budget after ~8.4s of genuine
    # semantic work.
    assert metadata.get("local_retrieval_slow") is False
    assert float(metadata.get("turn_remaining_ms", 0.0)) > 60_000.0
    # Provider histogram reflects the real sequential semantic workload.
    durations = list(metadata.get("provider_call_durations_ms", []) or [])
    assert len(durations) == 2
    assert float(metadata.get("provider_p50_ms", 0.0)) >= 3_000.0
    assert float(metadata.get("provider_max_ms", 0.0)) >= 3_000.0
    assert int(metadata.get("provider_call_count", 0)) == 2


async def test_slow_model_beyond_turn_deadline_is_typed_timeout(monkeypatch: Any) -> None:
    """A model slower than the remaining turn deadline never delivers."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.conversation.turn_budget import local_worker_info, new_turn_budget

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )
    budget = new_turn_budget(deadline_s=6.0)
    model = _DelayedScriptedModel(
        selections=[],
        markers={"need-1": "НЕДОСТИЖИМЫЙ-МАРКЕР"},
        selection_delay_s=30.0,
        coverage_delay_s=30.0,
    )
    started = _time.perf_counter()
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
        max_iterations=3,
        turn_budget=budget,
    )
    elapsed = _time.perf_counter() - started
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "exhausted"
    assert metadata.get("coverage_exhaustion_reason") == "latency-budget"
    assert metadata.get("typed_stage") == "provider_semantic_timeout"
    assert metadata.get("coverage_all_covered") is False
    assert metadata.get("coverage_used_model") is False
    # No delivery, no optimistic history: nothing read, nothing served.
    assert list(pack.passages) == []
    assert list(metadata.get("read_ids", []) or []) == []
    # Bounded wall time far below the 30s provider tail; single attempt.
    assert elapsed < 10.0
    assert model.selection_calls <= 1
    # No leaked background waiter work after the timeout.
    info = local_worker_info()
    assert int(info.get("waiters", 0)) == 0


async def test_slow_local_rrf_flags_diagnostic_without_misclassification(
    monkeypatch: Any,
) -> None:
    """RRF over its 5s diagnostic target flags slow, keeps adequate evidence."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.conversation.turn_budget import local_worker_info

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Разбор АЛЬФА-МАРКЕР рядом.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
        local_delay_s=0.35,
    )
    # Tighten only the diagnostic target: 0.35s of local work reads as
    # slow without changing the real turn deadline for model coverage.
    monkeypatch.setattr(retrieval_node_mod, "INTERACTIVE_LATENCY_BUDGET_MS", 100.0)
    import aa.retrieval.evidence as evidence_mod

    monkeypatch.setattr(evidence_mod, "INTERACTIVE_LATENCY_BUDGET_MS", 100.0)
    model = _DelayedScriptedModel(
        selections=[
            {
                "selected_chunk_ids": [first.chunk_id],
                "need_more_detail": False,
                "followup_queries": [],
            }
        ],
        markers={"need-1": "АЛЬФА-МАРКЕР"},
    )
    pack = await aretrieve_with_semantic_selection(
        index,
        ["утренний разбор тяги"],
        resolved_intent="утренний разбор тяги",
        conversation_context="",
        user_message="как разбирать тягу утром",
        selection_model=model,
        information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
        max_iterations=3,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata.get("coverage_status") == "ready"
    assert metadata.get("coverage_all_covered") is True
    assert metadata.get("local_retrieval_slow") is True
    assert metadata.get("typed_stage") in ("", None)
    served = " ".join(str(item.exact_text) for item in pack.passages)
    assert "АЛЬФА-МАРКЕР" in served
    info = local_worker_info()
    assert int(info.get("waiters", 0)) == 0
    assert int(info.get("worker_active", 0)) == 0


async def test_rate_limit_still_propagates_for_runner_recovery(monkeypatch: Any) -> None:
    """A real 429 escapes the loop (typed provider_429, never exhaustion)."""
    import aa.conversation.retrieval_node as retrieval_node_mod
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection
    from aa.opencode.errors import OpenCodeRateLimitError

    retrieval_node_mod.clear_canonical_read_cache()
    first = _record("chapter-1:ru:first", start=0, text="Начало про утренний разбор.")
    index = _fake_index([first])
    _install_branch(
        monkeypatch,
        [{first.chunk_id: _fused(first.chunk_id, 50.0)}],
        [[first.chunk_id]],
    )

    class _RateLimited:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeRateLimitError("429 slow down")

    try:
        await aretrieve_with_semantic_selection(
            index,
            ["утренний разбор тяги"],
            resolved_intent="утренний разбор тяги",
            conversation_context="",
            user_message="как разбирать тягу утром",
            selection_model=_RateLimited(),
            information_needs=[{"need_id": "need-1", "text": "утренний разбор"}],
            max_iterations=3,
        )
    except OpenCodeRateLimitError:
        return
    raise AssertionError("provider 429 must propagate, never convert to exhaustion")
