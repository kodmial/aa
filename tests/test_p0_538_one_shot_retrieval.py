"""P0 kodmial/aa#538: bounded one-shot retrieval coverage fast path.

The fast path is explicitly opt-in. These regressions prove that, when
enabled, one locally prepared canonical pack is judged by exactly one real
model HTTP request and only a fully verified per-need verdict (every need
covered, non-empty exact supporting quote per covered need) becomes
``coverage=ready``. Empty/false/unknown support, incomplete need lists,
oversized packs, invalid schema, missing candidates and provider errors
never produce an unproven answer: they fall back to the unchanged legacy
selector -> read_exact -> coverage loop or fail with a typed outcome.

Off mode must reproduce the legacy sequence with no additional model call.
All fixture text is synthetic; no canonical book or user text is stored.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aa.conversation.conversation_context import InformationNeed
from aa.retrieval.fusion import FusedCandidate


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    chunk_id: str,
    *,
    section: str = "chapter-1",
    start: int = 0,
    text: str = "Synthetic passage.",
    parent: str | None = None,
    prev: str | None = None,
    next_id: str | None = None,
) -> Any:
    from aa.retrieval.index import ChunkRecord

    end = start + len(text)
    return ChunkRecord(
        chunk_id=chunk_id,
        logical_chunk_id=chunk_id,
        section=section,
        book="aa-big-book",
        parent=parent or ("p-" + chunk_id),
        prev=prev,
        next=next_id,
        source_id="ru-fourth-edition-txt",
        source_file="corpus/source/raw-ru/aa-big-book.txt",
        source_sha256="s" * 64,
        char_start=start,
        char_end=end,
        text_sha256=_sha(text),
        text=text,
        corpus_version="ru-v1",
    )


def _index(records: list[Any]) -> Any:
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


def _fused(chunk_id: str, score: float) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_score=score,
        lexical_rank=1,
        dense_rank=1,
        lexical_score=score,
        dense_score=score,
    )


def _install_pool(monkeypatch: Any, fused: dict[str, FusedCandidate]) -> None:
    import aa.retrieval.evidence as evidence_mod

    def _fake_branch(index_arg: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index_arg, queries, kwargs)
        return ([], [list(fused)])

    def _fake_fuse(ranked: Any, per_ids: Any, **kwargs: Any) -> Any:
        _ = (ranked, per_ids, kwargs)
        return dict(fused), list(fused)

    monkeypatch.setattr(evidence_mod, "run_branch_searches", _fake_branch)
    monkeypatch.setattr(evidence_mod, "fuse_query_pool", _fake_fuse)


def _passage_texts(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in re.finditer(r'<passage id="([^"]+)"[^>]*>(.*?)</passage>', prompt, re.DOTALL):
        out[match.group(1)] = " ".join(match.group(2).split())
    return out


def _need_markers(markers: dict[str, str]) -> Any:
    def _script(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = one_shot
        texts = _passage_texts(prompt)
        needs: list[dict[str, Any]] = []
        for nid, marker in markers.items():
            supporting = [pid for pid, body in texts.items() if marker in body]
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

    return _script


class _ScriptedModel:
    """Counts one-shot / selector / legacy-coverage prompts separately."""

    def __init__(
        self,
        *,
        one_shot: Any | None = None,
        coverage: Any | None = None,
        selection: Any | None = None,
    ) -> None:
        self._one_shot = one_shot
        self._coverage = coverage
        self._selection = selection
        self.one_shot_prompts: list[str] = []
        self.coverage_prompts: list[str] = []
        self.selection_prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.one_shot_prompts) + len(self.coverage_prompts) + len(self.selection_prompts)

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (schema, retry_count)
        one_shot = "single retrieval coverage decision" in system
        if "<passages>" in prompt and "<original_request>" in prompt:
            if one_shot:
                self.one_shot_prompts.append(prompt)
                if self._one_shot is None:
                    return {"needs": []}
                return cast("dict[str, object]", self._one_shot(prompt, one_shot=True))
            self.coverage_prompts.append(prompt)
            if self._coverage is None:
                return {"needs": []}
            return cast("dict[str, object]", self._coverage(prompt, one_shot=False))
        self.selection_prompts.append(prompt)
        if self._selection is None:
            return {
                "selected_chunk_ids": [],
                "need_more_detail": True,
                "followup_queries": [],
            }
        return cast("dict[str, object]", self._selection(prompt, one_shot=False))


def _single_marker_fixture() -> tuple[Any, dict[str, FusedCandidate], list[Any]]:
    core_text = (
        "Вводный контекст разговора. " * 25
        + "BOOK_ANSWER: трезвая поддержка и спокойный разбор тяги."
    )
    assert len(core_text) > 500
    core = _record("chapter-1:core", text=core_text)
    neighbor = _record(
        "chapter-1:qualifier",
        start=core.char_end,
        text=" BOOK_QUALIFIER: сначала разговор с наставником.",
        prev=core.chunk_id,
    )
    return (
        _index([core, neighbor]),
        {
            core.chunk_id: _fused(core.chunk_id, 50.0),
            neighbor.chunk_id: _fused(neighbor.chunk_id, 49.0),
        },
        [core, neighbor],
    )


async def _run_retrieval(
    monkeypatch: Any,
    *,
    index: Any,
    model: Any,
    needs: list[InformationNeed],
    queries: list[str] | None = None,
    query_need_map: Any | None = None,
    budget_tokens: int | None = None,
    turn_budget: Any | None = None,
    conversation_context: str = "",
    resolved_intent: str = "разговор о тяге и поддержке",
    user_message: str = "как спокойно разобрать тягу",
) -> Any:
    from aa.conversation.retrieval_node import (
        aretrieve_with_semantic_selection,
        clear_canonical_read_cache,
    )
    from aa.retrieval.evidence import RetrievalConfig

    clear_canonical_read_cache()
    config = RetrievalConfig()
    if budget_tokens is not None:
        config = RetrievalConfig(budget_tokens=budget_tokens)
    return await aretrieve_with_semantic_selection(
        index,
        queries or ["разговор о тяге и поддержке"],
        config=config,
        resolved_intent=resolved_intent,
        conversation_context=conversation_context,
        user_message=user_message,
        selection_model=model,
        information_needs=needs,
        query_need_map=query_need_map,
        turn_budget=turn_budget,
        max_iterations=3,
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from aa.conversation.model_adapter import (
        clear_omitted_structured_cache,
        clear_primary_circuit,
        reset_llm_http_call_count,
    )

    clear_primary_circuit()
    clear_omitted_structured_cache()
    reset_llm_http_call_count()
    monkeypatch.delenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", raising=False)
    yield


async def test_one_shot_simple_covered_case_exactly_one_real_http_call(
    monkeypatch: Any,
) -> None:
    """Full local evidence -> ready with exactly one provider HTTP request."""
    from aa.conversation.model_adapter import (
        OpenCodeChatModel,
        llm_http_call_count,
        reset_llm_http_call_count,
    )

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    class _Client:
        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def create_session(self, title: str = "") -> Any:
            _ = title
            return SimpleNamespace(id="session-1")

        async def delete_session(self, session_id: str) -> bool:
            _ = session_id
            return True

        async def send_message(self, *args: Any, **kwargs: Any) -> str:
            _ = (args, kwargs)
            self.text_calls += 1
            raise AssertionError("text transport must not be used on the fast path")

        async def send_structured_message(self, session_id: str, prompt: str, **kwargs: Any) -> Any:
            _ = (session_id, kwargs)
            self.structured_calls += 1
            return _need_markers({"need-1": "BOOK_QUALIFIER"})(prompt, one_shot=True)

    client = _Client()
    model = OpenCodeChatModel(
        client,  # type: ignore[arg-type]
        agent="aa-planner-v2",
        primary_model="provider/model",
        fallback_model="",
        transport_agent="",
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    reset_llm_http_call_count()
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_QUALIFIER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["retrieval_mode"] == "one_shot"
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] == ""
    assert metadata["coverage_status"] == "ready"
    assert metadata["coverage_all_covered"] is True
    assert metadata["LLM_call_count"] == 1
    assert metadata["loop_model_calls"] == 1
    assert metadata["selector_ms"] == 0.0
    assert metadata["loop_searches"] == 0
    assert metadata["loop_expansions"] == 0
    assert client.structured_calls == 1
    assert client.text_calls == 0
    assert llm_http_call_count() == 1
    assert len(pack.passages) > 0
    per_need = {item["need_id"]: item for item in metadata["coverage_per_need"]}
    assert per_need["need-1"]["covered"] is True
    assert per_need["need-1"]["anchors"]
    # The decisive qualifier after character 500 survives untruncated.
    joined = " ".join(passage.exact_text for passage in pack.passages)
    assert "BOOK_QUALIFIER" in joined
    assert len(next(p.exact_text for p in pack.passages if "BOOK_QUALIFIER" in p.exact_text)) > 500


async def test_one_shot_off_reproduces_legacy_selector_and_coverage(monkeypatch: Any) -> None:
    """Default off executes the existing selector -> coverage path unchanged."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    script = _need_markers({"need-1": "BOOK_QUALIFIER"})
    model = _ScriptedModel(
        coverage=script,
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_QUALIFIER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["retrieval_mode"] == "legacy"
    assert metadata["fastpath_attempted"] is False
    assert metadata["fallback_reason"] == ""
    assert model.one_shot_prompts == []
    assert len(model.selection_prompts) == 1
    assert len(model.coverage_prompts) >= 1
    assert metadata["loop_model_calls"] >= 2


async def test_one_shot_multi_part_missing_need_never_ready(monkeypatch: Any) -> None:
    """One supported need plus one missing need is not a one-shot READY."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _one_shot(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = one_shot
        texts = _passage_texts(prompt)
        pid = next(pid for pid, body in texts.items() if "BOOK_ANSWER" in body)
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": True,
                    "supporting_passage_ids": [pid],
                    "supporting_quote": "BOOK_ANSWER",
                    "missing": "",
                },
                {
                    "need_id": "need-2",
                    "covered": False,
                    "supporting_passage_ids": [],
                    "supporting_quote": "",
                    "missing": "second part is absent",
                },
            ]
        }

    model = _ScriptedModel(
        one_shot=_one_shot,
        coverage=_need_markers({"need-1": "BOOK_ANSWER", "need-2": "NEVER_PRESENT"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[
            InformationNeed(need_id="need-1", text="BOOK_ANSWER"),
            InformationNeed(need_id="need-2", text="NEVER_PRESENT"),
        ],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] == "uncovered-needs"
    assert metadata["coverage_status"] != "ready"
    assert "need-2" in metadata["coverage_missing_need_ids"]
    assert len(model.one_shot_prompts) == 1
    assert len(model.selection_prompts) >= 1


async def test_sixteen_paraphrases_one_genuine_need_one_obligation(monkeypatch: Any) -> None:
    """Sixteen paraphrase queries to one genuine need cannot create 16 obligations."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    queries = [f"перефраз тяги номер {i}" for i in range(16)]
    query_map = [{"query_id": f"q{i + 1}", "need_ids": ["need-1"]} for i in range(16)]
    model = _ScriptedModel(one_shot=_need_markers({"need-1": "BOOK_QUALIFIER"}))
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_QUALIFIER")],
        queries=queries,
        query_need_map=query_map,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["coverage_status"] == "ready"
    assert metadata["genuine_need_count"] == 1
    assert metadata["LLM_call_count"] == 1
    assert len(model.one_shot_prompts) == 1


async def test_deep_ranked_hits_gt16_and_gt64_are_preserved(monkeypatch: Any) -> None:
    """Deep fused candidates stay reachable through the one-shot pack."""
    records: list[Any] = []
    fused: dict[str, FusedCandidate] = {}
    for pos in range(70):
        text = f"BACKGROUND {pos}"
        if pos == 66:
            text = f"BACKGROUND {pos} DEEP_ANSWER: спокойный разбор тяги."
        record = _record(f"chapter-{pos % 5}:c{pos:03d}", section=f"chapter-{pos % 5}", text=text)
        records.append(record)
        fused[record.chunk_id] = _fused(record.chunk_id, 100.0 - pos)
    index = _index(records)
    _install_pool(monkeypatch, fused)
    model = _ScriptedModel(one_shot=_need_markers({"need-1": "DEEP_ANSWER"}))
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="DEEP_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["coverage_status"] == "ready"
    deep_id = "chapter-1:c066"
    assert deep_id in metadata["discovered_ids"]
    assert deep_id in metadata["read_ids"]
    assert any(deep_id in passage.child_chunk_ids for passage in pack.passages)


async def test_unknown_passage_id_falls_back_after_typed_invalid_evidence(
    monkeypatch: Any,
) -> None:
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _bad_one_shot(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = (prompt, one_shot)
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": True,
                    "supporting_passage_ids": ["invented#passage"],
                    "supporting_quote": "BOOK_ANSWER",
                    "missing": "",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_bad_one_shot,
        coverage=_need_markers({"need-1": "BOOK_ANSWER"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] == "invalid-evidence"
    assert metadata["retrieval_mode"] == "legacy"
    # The legacy fallback may still reach ready with verified full evidence.
    assert metadata["coverage_status"] == "ready"
    assert len(model.one_shot_prompts) == 1


@pytest.mark.parametrize("bad_quote", ["", "НЕ СУЩЕСТВУЕТ В ТЕКСТЕ"])
async def test_empty_or_false_quote_never_becomes_ready(monkeypatch: Any, bad_quote: str) -> None:
    """A valid passage id alone is never enough on the fast path."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _bad_one_shot(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = one_shot
        texts = _passage_texts(prompt)
        pid = next(iter(texts))
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": True,
                    "supporting_passage_ids": [pid],
                    "supporting_quote": bad_quote,
                    "missing": "",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_bad_one_shot,
        coverage=_need_markers({"need-1": "BOOK_ANSWER"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] == "invalid-evidence"
    assert len(model.one_shot_prompts) == 1


async def test_quote_about_other_subject_is_not_covered(monkeypatch: Any) -> None:
    """A lexical quote that does not answer the need cannot support it."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _other_subject(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = (prompt, one_shot)
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": False,
                    "supporting_passage_ids": [],
                    "supporting_quote": "",
                    "missing": "the cited text is about another subject",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_other_subject,
        coverage=_need_markers({"need-1": "NEVER_PRESENT"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "uncovered-needs"
    assert metadata["coverage_status"] != "ready"


async def test_incomplete_need_list_falls_back(monkeypatch: Any) -> None:
    """A response that judges only one of two needs never becomes ready."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _incomplete(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = one_shot
        texts = _passage_texts(prompt)
        pid = next(pid for pid, body in texts.items() if "BOOK_ANSWER" in body)
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": True,
                    "supporting_passage_ids": [pid],
                    "supporting_quote": "BOOK_ANSWER",
                    "missing": "",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_incomplete,
        coverage=_need_markers({"need-1": "BOOK_ANSWER", "need-2": "NEVER_PRESENT"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[
            InformationNeed(need_id="need-1", text="BOOK_ANSWER"),
            InformationNeed(need_id="need-2", text="NEVER_PRESENT"),
        ],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] in {"uncovered-needs", "incomplete-needs"}
    assert metadata["coverage_status"] != "ready"


async def test_negation_or_exception_contradicting_conclusion_is_not_covered(
    monkeypatch: Any,
) -> None:
    contradiction = _record(
        "chapter-1:contradiction",
        text="BOOK_ANSWER не помогает, если человек один; нужен наставник.",
    )
    index = _index([contradiction])
    fused = {contradiction.chunk_id: _fused(contradiction.chunk_id, 40.0)}
    _install_pool(monkeypatch, fused)

    def _negated(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = (prompt, one_shot)
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": False,
                    "supporting_passage_ids": [],
                    "supporting_quote": "",
                    "missing": "the passage states the opposite condition",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_negated,
        coverage=_negated,
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [contradiction.chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "uncovered-needs"
    assert metadata["coverage_status"] != "ready"


async def test_oversized_first_pack_skips_one_shot_without_llm(monkeypatch: Any) -> None:
    """An oversized prompt skips the fast path before any model call."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    model = _ScriptedModel(
        one_shot=_need_markers({"need-1": "BOOK_ANSWER"}),
        coverage=_need_markers({"need-1": "BOOK_ANSWER"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
        budget_tokens=1,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] in {"prompt-oversize", "pack-oversize"}
    assert model.one_shot_prompts == []
    assert len(model.selection_prompts) >= 1


async def test_invalid_schema_falls_back_and_is_not_positive_coverage(
    monkeypatch: Any,
) -> None:
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _invalid(prompt: str, *, one_shot: bool) -> Any:
        _ = (prompt, one_shot)
        return {"needs": "not-a-list"}

    model = _ScriptedModel(
        one_shot=_invalid,
        coverage=_need_markers({"need-1": "BOOK_ANSWER"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "invalid-evidence"
    assert len(model.one_shot_prompts) == 1


async def test_provider_429_propagates_from_one_shot(monkeypatch: Any) -> None:
    from aa.opencode.errors import OpenCodeRateLimitError

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    class _RateLimited:
        async def ainvoke_structured(self, *args: Any, **kwargs: Any) -> Any:
            _ = (args, kwargs)
            raise OpenCodeRateLimitError("http=429")

    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    with pytest.raises(OpenCodeRateLimitError, match="429"):
        await _run_retrieval(
            monkeypatch,
            index=index,
            model=_RateLimited(),
            needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
        )


async def test_no_model_skips_fast_path_without_calls(monkeypatch: Any) -> None:
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=None,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is False
    assert metadata["fallback_reason"] == "no-model"
    from aa.conversation.model_adapter import llm_http_call_count

    assert llm_http_call_count() == 0
    assert metadata["coverage_status"] != "ready"


async def test_latency_budget_skips_one_shot_before_model(monkeypatch: Any) -> None:
    """A spent upstream stage budget is a typed exhaustion, never a call."""
    from aa.conversation.turn_budget import new_turn_budget

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    model = _ScriptedModel(one_shot=_need_markers({"need-1": "BOOK_ANSWER"}))
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")
    # Spend the 20s retrieval semantic headroom but keep the 105s turn
    # deadline open, so the fast path skips the model call itself.
    budget = new_turn_budget(upstream_latency_ms=25_000.0)
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
        turn_budget=budget,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "latency-budget"
    assert metadata["LLM_call_count"] == 0
    assert model.calls == 0
    assert metadata["coverage_status"] == "exhausted"


async def test_replay_is_deterministic_for_both_modes(monkeypatch: Any) -> None:
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    script = _need_markers({"need-1": "BOOK_ANSWER"})
    model = _ScriptedModel(
        one_shot=script,
        coverage=script,
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    needs = [InformationNeed(need_id="need-1", text="BOOK_ANSWER")]
    first = await _run_retrieval(monkeypatch, index=index, model=model, needs=needs)
    second = await _run_retrieval(monkeypatch, index=index, model=model, needs=needs)
    first_ids = [passage.passage_id for passage in first.passages]
    second_ids = [passage.passage_id for passage in second.passages]
    assert first_ids == second_ids
    assert first.retrieval_metadata["coverage_status"] == "ready"
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    third = await _run_retrieval(monkeypatch, index=index, model=model, needs=needs)
    fourth = await _run_retrieval(monkeypatch, index=index, model=model, needs=needs)
    third_ids = [passage.passage_id for passage in third.passages]
    fourth_ids = [passage.passage_id for passage in fourth.passages]
    assert third_ids == fourth_ids
    assert third.retrieval_metadata["retrieval_mode"] == "one_shot"
    assert "chapter-1:core" in {
        cid for passage in third.passages for cid in passage.child_chunk_ids
    }


def test_sanitized_metadata_carries_required_privacy_safe_counters() -> None:
    from aa.conversation.retrieval_node import sanitize_retrieval_metadata

    safe = sanitize_retrieval_metadata(
        {
            "retrieval_mode": "one_shot",
            "fastpath_attempted": True,
            "fallback_reason": "uncovered-needs",
            "LLM_call_count": 1,
            "selector_ms": 0.0,
            "coverage_ms": 12.5,
            "total_ms": 20.0,
            "genuine_need_count": 2,
            "uncovered_need_count": 1,
            "selected_passages": 3,
            "read_ids": ["chapter-1:core"],
            "prompt_text": "secret book text",
            "user_message": "secret user text",
        }
    )
    for key in (
        "retrieval_mode",
        "fastpath_attempted",
        "fallback_reason",
        "LLM_call_count",
        "selector_ms",
        "coverage_ms",
        "total_ms",
        "genuine_need_count",
        "uncovered_need_count",
    ):
        assert key in safe
    assert "prompt_text" not in safe
    assert "user_message" not in safe
    assert "secret" not in str(safe)


def test_coverage_prompt_is_not_a_product_pass() -> None:
    """The one-shot decision prompt cannot claim a product-level PASS."""
    from aa.conversation.coverage_loop import coverage_prompt

    system, user = coverage_prompt(
        passages=[],
        information_needs=[InformationNeed(need_id="need-1", text="synthetic")],
        original_request="synthetic request",
        one_shot=True,
    )
    assert "single retrieval coverage decision" in system
    assert "product-level PASS" in system
    assert "supporting_quote" in user
    assert "<passages>" in user


def test_legacy_coverage_prompt_does_not_receive_fast_path_text() -> None:
    """Flag-off keeps the existing system/user prompt additions gated."""
    from aa.conversation.coverage_loop import coverage_prompt

    system, user = coverage_prompt(
        passages=[],
        information_needs=[InformationNeed(need_id="need-1", text="synthetic")],
        original_request="synthetic request",
        conversation_context="previous synthetic turn",
        one_shot=False,
    )
    assert "single retrieval coverage decision" not in system
    assert "product-level PASS" not in system
    assert "exact non-empty quote copied from that passage" not in system
    assert "previous synthetic turn" not in user


async def test_llm_http_call_accounting_is_isolated_per_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping retrievals each report only their own provider calls."""
    import asyncio

    from aa.conversation import retrieval_node as retrieval_mod
    from aa.conversation.model_adapter import (
        llm_http_call_count,
        note_llm_http_call,
        reset_llm_http_call_count,
    )

    async def _fake(index: Any, queries: list[str], **kwargs: Any) -> Any:
        _ = (index, queries, kwargs)
        note_llm_http_call()
        await asyncio.sleep(0.01)
        note_llm_http_call()
        return llm_http_call_count()

    from aa.retrieval.index import HybridIndex

    monkeypatch.setattr(retrieval_mod, "_aretrieve_with_semantic_selection_impl", _fake)
    reset_llm_http_call_count()
    fake_index = cast("HybridIndex", SimpleNamespace())
    first, second = await asyncio.gather(
        retrieval_mod.aretrieve_with_semantic_selection(fake_index, []),
        retrieval_mod.aretrieve_with_semantic_selection(fake_index, []),
    )
    assert cast("int", first) == 2
    assert cast("int", second) == 2
    assert llm_http_call_count() == 4


async def test_one_shot_long_qualifier_and_short_continuation_keep_context(
    monkeypatch: Any,
) -> None:
    """A >4000-char atomic passage is never silently truncated for one-shot.

    The prompt also carries the previous turn's context so a short
    continuation such as "В начале" is judged against the resolved
    intent instead of the literal fragment.
    """
    long_text = ("Фоновый абзац для проверки длины. " * 160) + (
        "LONG_QUALIFIER: только вместе с наставником."
    )
    assert len(long_text) > 4500
    core = _record("chapter-1:long", text=long_text)
    index = _index([core])
    fused = {core.chunk_id: _fused(core.chunk_id, 50.0)}
    _install_pool(monkeypatch, fused)
    seen_prompts: list[str] = []

    def _answer(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = one_shot
        seen_prompts.append(prompt)
        assert "LONG_QUALIFIER" in prompt
        pid = next(iter(_passage_texts(prompt)))
        return {
            "needs": [
                {
                    "need_id": "need-1",
                    "covered": True,
                    "supporting_passage_ids": [pid],
                    "supporting_quote": "LONG_QUALIFIER",
                    "missing": "",
                }
            ]
        }

    model = _ScriptedModel(one_shot=_answer)
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="LONG_QUALIFIER")],
        queries=["В начале"],
        resolved_intent="В начале — что помогает при тяге",
        conversation_context="Предыдущий вопрос: как справиться с сильной тягой",
        user_message="В начале",
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["retrieval_mode"] == "one_shot"
    assert metadata["coverage_status"] == "ready"
    assert seen_prompts
    assert "<conversation_context>" in seen_prompts[0]
    assert "Предыдущий вопрос" in seen_prompts[0]
    assert "LONG_QUALIFIER" in seen_prompts[0]


async def test_full_graph_both_modes_keep_answer_and_final_verification(
    monkeypatch: Any,
) -> None:
    """Planner -> retrieval -> answer -> final verifier is unchanged by the mode.

    The fast path only replaces the retrieval coverage decision. The
    planner contract, the answer draft, the independent final
    verification certificate and the evidence digest stay identical;
    off mode executes the legacy selector/coverage sequence, on mode
    issues exactly one retrieval model call.
    """
    import json as _json

    from langchain_core.messages import AIMessage

    from aa.conversation.finalization import evidence_digest_for_pack
    from aa.conversation.graph import build_turn_graph, turn_input
    from aa.retrieval.evidence import RetrievalConfig

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)
    question = "как спокойно разобрать тягу вечером?"
    intent = "как спокойно разобрать тягу вечером"
    draft = "Спокойный разбор тяги опирается на трезвую поддержку рядом."

    class _GraphModel:
        def __init__(self) -> None:
            self.planner_prompts: list[str] = []
            self.selection_prompts: list[str] = []
            self.coverage_prompts: list[str] = []
            self.one_shot_prompts: list[str] = []

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (schema, retry_count)
            if "<passage id=" in prompt and "<original_request>" in prompt:
                one_shot = "single retrieval coverage decision" in system
                bucket = self.one_shot_prompts if one_shot else self.coverage_prompts
                bucket.append(prompt)
                return cast(
                    "dict[str, object]",
                    _need_markers({"need-1": "BOOK_ANSWER"})(prompt, one_shot=one_shot),
                )
            if "<candidate id=" in prompt:
                self.selection_prompts.append(prompt)
                return {
                    "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
                    "need_more_detail": False,
                    "followup_queries": [],
                }
            self.planner_prompts.append(prompt)
            return {
                "mode": "retrieval",
                "resolved_intent": intent,
                "queries": [intent],
                "information_needs": [{"need_id": "need-1", "text": "BOOK_ANSWER"}],
            }

        async def ainvoke(self, messages: Any) -> Any:
            _ = messages
            return AIMessage(
                content=_json.dumps(
                    {
                        "mode": "retrieval",
                        "resolved_intent": intent,
                        "queries": [intent],
                        "information_needs": [{"need_id": "need-1", "text": "BOOK_ANSWER"}],
                    }
                )
            )

    class _GraphAnswer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> Any:
            _ = messages
            self.calls += 1
            return AIMessage(content=draft)

    class _GraphVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 2,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
                "addresses_intent": True,
            }

    async def _run(flag: bool) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        from aa.conversation.retrieval_node import clear_canonical_read_cache

        clear_canonical_read_cache()
        model = _GraphModel()
        answer = _GraphAnswer()
        verifier = _GraphVerifier()
        graph = build_turn_graph(
            planner_model=model,  # type: ignore[arg-type]
            retrieval_index=index,
            retrieval_config=RetrievalConfig(one_shot_fast_path=flag),
            answer_model=answer,
            verifier_model=verifier,
        )
        result = await graph.ainvoke(turn_input(question))
        return model, answer, verifier, result, dict(result.get("retrieval_metadata") or {})

    off_model, off_answer, off_verifier, off_result, off_meta = await _run(False)
    assert off_result["final_response"] == draft
    assert off_result["delivery_status"] == "provisional-certified"
    assert off_meta["retrieval_mode"] == "legacy"
    assert off_meta["fastpath_attempted"] is False
    assert off_model.one_shot_prompts == []
    assert off_model.selection_prompts
    assert off_model.coverage_prompts
    assert off_answer.calls >= 1
    assert off_verifier.calls >= 1

    on_model, on_answer, on_verifier, on_result, on_meta = await _run(True)
    assert on_result["final_response"] == draft
    assert on_result["delivery_status"] == "provisional-certified"
    assert on_meta["retrieval_mode"] == "one_shot"
    assert on_meta["fastpath_attempted"] is True
    assert on_meta["fallback_reason"] == ""
    assert on_meta["coverage_status"] == "ready"
    assert on_meta["LLM_call_count"] == 1
    assert len(on_model.one_shot_prompts) == 1
    assert on_model.selection_prompts == []
    assert on_model.coverage_prompts == []
    assert on_answer.calls >= 1
    assert on_verifier.calls >= 1

    for result in (off_result, on_result):
        served_pack = [dict(item) for item in result["evidence_pack"]]
        assert served_pack
        assert any("BOOK_ANSWER" in str(item.get("text", "")) for item in served_pack)
        certificate = result["verification_certificate"]
        candidate = result["answer_candidate"]
        assert certificate["answer_sha256"] == _sha(draft)
        assert certificate["context_digest"] == candidate["context_digest"]
        assert certificate["evidence_digest"] == evidence_digest_for_pack(served_pack)
        assert certificate["whole_answer_verdict"]["supported"] is True


def test_hermetic_ab_reproducibly_reports_ground_truth_and_calls(tmp_path: Any) -> None:
    """Frozen + held-out A/B uses one corpus/index/model config in both modes."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "one-shot-ab-538.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_one_shot_retrieval_ab_538.py"),
            "--json-out",
            str(out),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["harness"] == "synthetic-packing-coverage"
    assert report["real_corpus_index_model_config"] is False
    assert report["same_corpus_index_model_config"] is True
    assert report["fixture_version"] == "one-shot-retrieval-ab-538.v1"
    assert set(report["summary"]["by_split"]) == {"frozen", "held_out"}
    assert set(report["summary"]["by_category"]) >= {"simple", "complex", "provider-error"}

    rows = {(row["case_id"], row["mode"]): row for row in report["results"]}
    simple_off = rows[("frozen-simple", "off")]
    simple_on = rows[("frozen-simple", "on")]
    assert simple_off["retrieval_mode"] == "legacy"
    assert simple_on["retrieval_mode"] == "one_shot"
    assert simple_off["coverage_match"] is True
    assert simple_on["coverage_match"] is True
    assert simple_on["scripted_http_calls"] == 1
    assert simple_off["scripted_http_calls"] > simple_on["scripted_http_calls"]

    complex_off = rows[("frozen-complex", "off")]
    complex_on = rows[("frozen-complex", "on")]
    assert complex_off["coverage_match"] is True
    assert complex_on["coverage_match"] is True
    assert complex_off["coverage_status"] != "ready"
    assert complex_on["coverage_status"] != "ready"
    assert complex_on["fallback_reason"] == "uncovered-needs"

    for mode in ("off", "on"):
        error = rows[("heldout-provider-error", mode)]
        assert error["error_type"] == "OpenCodeRateLimitError"
        assert error["retrieval_recall"] is None
        assert error["source_match"] is None

    error_summary = report["summary"]["by_category"]["provider-error"]
    for mode in ("off", "on"):
        assert error_summary[mode]["retrieval_recall"] is None
        assert error_summary[mode]["retrieval_recall_compared"] == 0
        assert error_summary[mode]["retrieval_recall_skipped"] == 1

    simple = report["summary"]["by_category"]["simple"]
    assert simple["on"]["logical_model_calls"] < simple["off"]["logical_model_calls"]
    assert simple["on"]["input_tokens"] < simple["off"]["input_tokens"]
    assert simple["on"]["source_matches"] == simple["on"]["source_compared"]
    assert simple["off"]["source_matches"] == simple["off"]["source_compared"]
    assert simple["on"]["retrieval_recall_compared"] == simple["on"]["cases"]
    assert simple["on"]["latency_p50_ms"] >= 0.0
    assert simple["on"]["latency_p95_ms"] >= simple["on"]["latency_p50_ms"]
    assert simple["on"]["safe_outcomes"] == simple["on"]["cases"]
    complex_summary = report["summary"]["by_category"]["complex"]
    assert complex_summary["on"]["fallback_cases"] == 1
    assert complex_summary["on"]["fallback_scripted_http_calls"] > 0


async def test_modern_cancellation_propagates_from_one_shot(monkeypatch: Any) -> None:
    import asyncio

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    class _Cancelled:
        async def ainvoke_structured(self, *args: Any, **kwargs: Any) -> Any:
            _ = (args, kwargs)
            raise asyncio.CancelledError()

    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    with pytest.raises(asyncio.CancelledError):
        await _run_retrieval(
            monkeypatch,
            index=index,
            model=_Cancelled(),
            needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
        )


async def test_slow_provider_one_shot_times_out_into_typed_exhaustion(
    monkeypatch: Any,
) -> None:
    """A slow provider is bounded by the turn stage headroom, never a fake PASS."""
    import asyncio

    from aa.conversation.turn_budget import new_turn_budget

    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    class _Slow:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(self, *args: Any, **kwargs: Any) -> Any:
            _ = (args, kwargs)
            self.calls += 1
            await asyncio.sleep(5)
            return {"needs": []}

    model = _Slow()
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")
    # Leave only ~0.4s of the 20s retrieval semantic headroom.
    budget = new_turn_budget(upstream_latency_ms=19_600.0)
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
        turn_budget=budget,
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "latency-budget"
    assert metadata["coverage_status"] == "exhausted"
    assert metadata["coverage_status"] != "ready"
    assert model.calls == 1


async def test_no_candidates_skips_fast_path_without_llm(monkeypatch: Any) -> None:
    """An empty discovered pool is a typed skip before any model request."""
    index, _fused_map, _records = _single_marker_fixture()
    _install_pool(monkeypatch, {})
    model = _ScriptedModel(one_shot=_need_markers({"need-1": "BOOK_ANSWER"}))
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fastpath_attempted"] is True
    assert metadata["fallback_reason"] == "no-candidates"
    assert model.one_shot_prompts == []
    assert model.calls == 0
    assert metadata["coverage_status"] != "ready"


async def test_unknown_need_id_falls_back_and_never_ready(monkeypatch: Any) -> None:
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    def _unknown_need(prompt: str, *, one_shot: bool) -> dict[str, Any]:
        _ = (prompt, one_shot)
        return {
            "needs": [
                {
                    "need_id": "invented-need",
                    "covered": True,
                    "supporting_passage_ids": [],
                    "supporting_quote": "BOOK_ANSWER",
                    "missing": "",
                }
            ]
        }

    model = _ScriptedModel(
        one_shot=_unknown_need,
        coverage=_need_markers({"need-1": "NEVER_PRESENT"}),
        selection=lambda prompt, one_shot: {
            "selected_chunk_ids": [index.chunks["chapter-1:core"].chunk_id],
            "need_more_detail": False,
            "followup_queries": [],
        },
    )
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "invalid-evidence"
    assert metadata["coverage_status"] != "ready"


async def test_invalid_json_text_fallback_fails_closed(monkeypatch: Any) -> None:
    """A malformed text answer is never parsed into positive coverage."""
    index, fused, _records = _single_marker_fixture()
    _install_pool(monkeypatch, fused)

    class _TextOnly:
        def __init__(self) -> None:
            self.calls = 0

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.calls += 1
            return "this is not json"

    model = _TextOnly()
    monkeypatch.setenv("AA_RETRIEVAL_ONE_SHOT_FASTPATH", "true")
    pack = await _run_retrieval(
        monkeypatch,
        index=index,
        model=model,
        needs=[InformationNeed(need_id="need-1", text="BOOK_ANSWER")],
    )
    metadata = dict(pack.retrieval_metadata or {})
    assert metadata["fallback_reason"] == "invalid-evidence"
    assert metadata["coverage_status"] != "ready"
    assert model.calls >= 1
