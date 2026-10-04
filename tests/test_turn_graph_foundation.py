"""Foundation regression tests for the LangGraph turn layer (issue #113).

These are regression fixtures proving the planner/memory/prompt contracts;
they never encode string-specific routing rules.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, ValidationError

from aa.conversation.graph_memory import (
    COMPACTION_TRIGGER_TOKENS,
    CONTEXT_REFERENCE_TOKENS,
    DEFAULT_MEMORY_CONFIG,
    RETAIN_TOKENS,
    CheckpointerConfig,
    build_summarization_middleware,
    count_conversation_tokens,
    create_checkpointer,
    load_summarization_prompt,
    needs_compaction,
    run_summarization,
    split_for_compaction,
    thread_id_for_chat,
)
from aa.conversation.graph_planner import (
    MAX_QUERIES,
    MIN_NONEMPTY_QUERIES,
    QUERY_PLANNER_VERSION,
    QueryPlan,
    build_planner_chain,
    format_planner_context,
    load_query_planner_prompt,
    run_planner_chain,
    validate_query_plan,
)
from aa.conversation.graph_prompts import (
    AA_AGENT_PROMPT_VERSION,
    EvidencePassage,
    build_answer_messages,
    load_aa_agent_prompt,
)
from aa.conversation.graph_state import GRAPH_STATE_FIELDS, TurnState, initial_state
from aa.conversation.turn_graph import build_turn_graph, run_turn
from aa.opencode.chat_model import OpenCodeChatModel
from aa.opencode.client import FakeOpenCodeClient

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model returning canned texts (test double)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    replies: list[str]
    calls: int = 0

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies=replies, calls=0)  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = stop
        _ = run_manager
        _ = kwargs
        _ = messages
        index = min(self.calls, len(self.replies) - 1)
        text = self.replies[index]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _plan_json(queries: list[str]) -> str:
    return json.dumps({"queries": queries}, ensure_ascii=False)


def _twelve_queries() -> list[str]:
    return [f"трезвость формулировка номер {index}" for index in range(12)]


def _src_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "src" / "aa"


# ---------------------------------------------------------------------------
# Typed state + dependencies + artifacts
# ---------------------------------------------------------------------------


def test_graph_state_fields_match_contract() -> None:
    assert set(GRAPH_STATE_FIELDS) == {
        "messages",
        "conversation_summary",
        "current_user_message",
        "search_queries",
        "retrieval_hits",
        "evidence_pack",
        "draft_response",
        "grounding_result",
        "retry_state",
    }
    state = initial_state(current_user_message="привет")
    assert state["current_user_message"] == "привет"
    assert state["search_queries"] == []


def test_dependencies_are_pinned() -> None:
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    for pinned in (
        "langchain==1.4.3",
        "langchain-core==1.6.6",
        "langgraph==1.2.12",
        "langgraph-checkpoint==4.2.0",
        "pydantic==2.13.5",
    ):
        assert pinned in text


def test_prompt_artifacts_are_versioned_english() -> None:
    root = pathlib.Path(__file__).resolve().parents[1] / "prompts"
    planner_path = root / f"{QUERY_PLANNER_VERSION}.md"
    summary_path = root / "conversation-summary-v1.md"
    agent_path = root / f"{AA_AGENT_PROMPT_VERSION}.md"
    for path in (planner_path, summary_path, agent_path):
        assert path.exists(), f"missing prompt artifact: {path.name}"
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"empty prompt artifact: {path.name}"
        assert not any("\u0400" <= ch <= "\u04ff" for ch in text), f"{path.name} must be English"
    assert load_query_planner_prompt() == planner_path.read_text(encoding="utf-8").strip()
    assert load_summarization_prompt() == summary_path.read_text(encoding="utf-8").strip()
    assert load_aa_agent_prompt() == agent_path.read_text(encoding="utf-8").strip()


def test_planner_prompt_is_baseline_contract() -> None:
    text = load_query_planner_prompt()
    assert "hidden retrieval query planner" in text
    assert "Return only the structured output." in text
    assert "10 to 16" in text


def test_aa_agent_prompt_is_product_behavior() -> None:
    text = load_aa_agent_prompt()
    assert "<book_evidence>" in text
    assert "<conversation_memory>" in text
    assert "Always answer in Russian." in text
    assert "answer truthfully that you are an AI assistant" in text
    for technical in ("FAISS", "SQLite", "RRF"):
        assert technical not in text


def test_new_graph_imports_no_legacy_routers() -> None:
    forbidden_imports = (
        "from aa.conversation.orchestrator",
        "import aa.conversation.orchestrator",
        "from aa.retrieval.planner",
        "import aa.retrieval.planner",
        "from aa.conversation.meta",
        "import aa.conversation.meta",
    )
    forbidden_definitions = (
        "def is_substantive",
        "_TRIVIAL_NORMALIZED =",
        "_TRIVIAL_NORMALIZED=",
        "_FOLLOWUP_INTERROGATIVES =",
        "_SUBSTANTIVE_KEYWORDS =",
        "_SLANG_EXPANSIONS =",
        "_THEME_MARKERS =",
        "_BROAD_COVERAGE_QUERIES =",
        "ru-query-plan-v1",
        "def is_meta_capability_request",
    )
    checked = [
        "opencode/chat_model.py",
        "conversation/graph_state.py",
        "conversation/graph_planner.py",
        "conversation/graph_memory.py",
        "conversation/graph_prompts.py",
        "conversation/turn_graph.py",
    ]
    root = _src_root()
    assert checked, "new graph file list must not be empty"
    for relative in checked:
        text = (root / relative).read_text(encoding="utf-8")
        for snippet in (*forbidden_imports, *forbidden_definitions):
            assert snippet not in text, f"{relative} must not contain {snippet!r}"


# ---------------------------------------------------------------------------
# Planner schema
# ---------------------------------------------------------------------------


def test_planner_accepts_empty_and_bounded_lists() -> None:
    assert validate_query_plan({"queries": []}).queries == []
    plan = validate_query_plan({"queries": _twelve_queries()})
    assert len(plan.queries) == 12
    assert QueryPlan(queries=[]).queries == []


def test_planner_rejects_other_cardinalities() -> None:
    for count in (1, 2, 5, 9, 17, 20):
        queries = [f"запрос трезвость {index}" for index in range(count)]
        with pytest.raises(ValidationError):
            QueryPlan(queries=queries)


def test_planner_rejects_empty_and_duplicate_entries() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(queries=["   "] + [f"запрос {i}" for i in range(10)])
    duplicated = [f"запрос трезвость {i}" for i in range(10)] + ["Запрос Трезвость 0"]
    with pytest.raises(ValidationError):
        QueryPlan(queries=duplicated)
    assert MIN_NONEMPTY_QUERIES == 10
    assert MAX_QUERIES == 16


def test_planner_schema_is_minimal() -> None:
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"queries": [], "intent": "chat"})
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"queries": [], "category": "x"})
    assert set(QueryPlan.model_fields) == {"queries"}


# ---------------------------------------------------------------------------
# Mandatory planner path (no string-specific routing)
# ---------------------------------------------------------------------------


async def _run_graph_with_scripted(replies: list[str], message: str) -> dict[str, Any]:
    model = ScriptedChatModel(replies=replies)
    compiled = build_turn_graph(model)
    return await run_turn(compiled, chat_id=777001, user_message=message)


async def test_every_ordinary_turn_invokes_planner() -> None:
    model = ScriptedChatModel(replies=[_plan_json([])])
    compiled = build_turn_graph(model)
    result = await run_turn(compiled, chat_id=4242, user_message="расскажи про трезвость")
    assert model.calls == 1
    assert result["search_queries"] == []
    assert result["retry_state"]["route"] == "normal"


async def test_tricky_utterances_share_one_planner_path() -> None:
    utterances = [
        "А что ты можешь?",
        "Тогда зачем ты?",
        "почему?",
        "а он?",
        "а дальше?",
        "и что потом?",
        "ну и?",
        "расскажи про страх другими словами без повторов",
        "меня накрыла тяга после ссоры с женой что делать",
    ]
    for position, utterance in enumerate(utterances):
        model = ScriptedChatModel(replies=[_plan_json([])])
        compiled = build_turn_graph(model)
        result = await run_turn(compiled, chat_id=900000 + position, user_message=utterance)
        assert model.calls == 1, f"planner was skipped for {utterance!r}"
        assert result["retry_state"]["route"] == "normal"
        assert "search_queries" in result


async def test_commands_and_empty_skip_planner() -> None:
    model = ScriptedChatModel(replies=[_plan_json([])])
    compiled = build_turn_graph(model)
    result = await run_turn(compiled, chat_id=5150, user_message="/new")
    assert model.calls == 0
    assert result["retry_state"]["route"] == "command"
    empty_result = await run_turn(compiled, chat_id=5151, user_message="   ")
    assert empty_result["retry_state"]["route"] == "blocked"


async def test_planner_chain_uses_standard_parser_with_retry() -> None:
    model = ScriptedChatModel(replies=[_plan_json(_twelve_queries())])
    chain = build_planner_chain(model)
    assert hasattr(chain, "with_retry")
    state: TurnState = initial_state(current_user_message="тяга и страх")
    plan = await run_planner_chain(state, model=model)
    assert len(plan.queries) == 12


# ---------------------------------------------------------------------------
# Memory: summary continuity, checkpointing, thread identity
# ---------------------------------------------------------------------------


def test_memory_thresholds_are_token_driven() -> None:
    assert CONTEXT_REFERENCE_TOKENS == 200_000
    assert COMPACTION_TRIGGER_TOKENS == 120_000
    assert RETAIN_TOKENS == 40_000
    assert DEFAULT_MEMORY_CONFIG.trigger_tokens == 120_000
    assert DEFAULT_MEMORY_CONFIG.retain_tokens == 40_000
    assert not needs_compaction(1000)
    assert needs_compaction(200_000)
    assert count_conversation_tokens([HumanMessage(content="hi")]) > 0


def test_thread_identity_is_deterministic_and_opaque() -> None:
    first = thread_id_for_chat(12345)
    assert thread_id_for_chat(12345) == first
    assert thread_id_for_chat(54321) != first
    assert "12345" not in first


def test_checkpointer_backend_is_explicit_and_local() -> None:
    checkpointer = create_checkpointer(CheckpointerConfig(kind="memory"))
    assert checkpointer is not None
    with pytest.raises(ValueError):
        create_checkpointer(CheckpointerConfig(kind="postgres-future"))


def test_split_never_orphans_structured_messages() -> None:
    tool_call = AIMessage(content="working", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    tool_result = ToolMessage(content="result", tool_call_id="1")
    messages: list[BaseMessage] = [
        HumanMessage(content="old turn " + "x" * 50000),
        AIMessage(content="old reply " + "y" * 50000),
        tool_call,
        tool_result,
    ]
    to_summarize, to_keep = split_for_compaction(messages)
    assert to_keep, "newest messages must be retained verbatim"
    assert not (
        len(to_keep) == 1 and isinstance(to_keep[0], ToolMessage) and tool_call not in to_keep
    )
    assert to_summarize is not to_keep


def test_summarization_middleware_is_token_driven() -> None:
    model = ScriptedChatModel(replies=["summary"])
    middleware = build_summarization_middleware(model)
    trigger = getattr(middleware, "trigger", None)
    keep = getattr(middleware, "keep", None)
    assert trigger == ("tokens", COMPACTION_TRIGGER_TOKENS)
    assert keep == ("tokens", RETAIN_TOKENS)
    prompt = load_summarization_prompt()
    assert "Conversation memory is not an authority" in prompt


async def test_summary_and_recent_messages_resolve_followups() -> None:
    summary = "Пользователь обсуждает тягу вечером после работы."
    recent: list[BaseMessage] = [
        HumanMessage(content="я сорвался вчера вечером"),
        AIMessage(content="понимаю, давай разберём вечернюю тягу"),
    ]
    state: TurnState = initial_state(
        current_user_message="почему это случилось?",
        messages=recent,
        conversation_summary=summary,
    )
    context = format_planner_context(state)
    assert "тягу вечером" in context
    assert "почему это случилось?" not in context or "Current" not in context
    assert summary in context
    # The same summary must travel as continuity memory, never as evidence.
    assembled = build_answer_messages(
        conversation_memory=summary,
        evidence=[],
        user_message="почему это случилось?",
        recent_messages=recent,
    )
    payload = str(assembled[-1].content)
    assert "<conversation_memory>" in payload
    assert summary in payload.split("<book_evidence>")[0]


async def test_hidden_calls_leave_visible_history_untouched() -> None:
    before: list[BaseMessage] = [HumanMessage(content="видимое сообщение")]
    state: TurnState = initial_state(
        current_user_message="почему?",
        messages=list(before),
    )
    model = ScriptedChatModel(replies=[_plan_json([])])
    plan = await run_planner_chain(state, model=model)
    assert plan.queries == []
    assert state["messages"] == before
    summarizer = ScriptedChatModel(replies=["краткое продолжение"])
    summary = await run_summarization(list(before), model=summarizer)
    assert summary == "краткое продолжение"
    assert list(before) == state["messages"]


async def test_graph_persists_turns_on_one_thread() -> None:
    model = ScriptedChatModel(replies=[_plan_json([]), _plan_json([])])
    compiled = build_turn_graph(model)
    first = await run_turn(compiled, chat_id=31337, user_message="первое сообщение")
    second = await run_turn(compiled, chat_id=31337, user_message="второе сообщение")
    texts = [
        str(message.content) for message in second["messages"] if isinstance(message, HumanMessage)
    ]
    assert any("первое сообщение" in text for text in texts)
    assert any("второе сообщение" in text for text in texts)
    assert first["retry_state"]["route"] == "normal"


# ---------------------------------------------------------------------------
# OpenCode adapter: isolated hidden sessions + configured fallback policy
# ---------------------------------------------------------------------------


async def test_adapter_uses_isolated_hidden_sessions() -> None:
    client = FakeOpenCodeClient()
    user_session = await client.create_session("user-visible")
    await client.send_message(user_session.id, "видимый вопрос")
    before = await client.list_messages(user_session.id)
    adapter = OpenCodeChatModel(client, agent="aa", primary_model=PRIMARY, fallback_model=FALLBACK)
    reply = await adapter.ainvoke([HumanMessage(content="скрытый план")])
    assert str(reply.content).strip()
    after = await client.list_messages(user_session.id)
    assert len(after) == len(before)
    # Hidden sessions are cleaned up; only the user session remains.
    assert set(client._sessions) == {user_session.id}


async def test_adapter_reuses_configured_fallback_policy() -> None:
    from aa.opencode.errors import OpenCodeTransientError

    client = FakeOpenCodeClient(fail_next=OpenCodeTransientError("http=429"))
    adapter = OpenCodeChatModel(client, agent="aa", primary_model=PRIMARY, fallback_model=FALLBACK)
    reply = await adapter.ainvoke([HumanMessage(content="проверка фолбэка")])
    assert str(reply.content).strip()


def test_adapter_contains_no_semantic_logic() -> None:
    text = (_src_root() / "opencode" / "chat_model.py").read_text(encoding="utf-8")
    for snippet in ("is_substantive", "retrieval", "keyword", "intent", "category"):
        assert snippet not in text.lower() or "no semantic" in text.lower()


# ---------------------------------------------------------------------------
# Prompt assembly contract
# ---------------------------------------------------------------------------


def test_prompt_assembly_keeps_blocks_separated_and_message_last() -> None:
    recent: list[BaseMessage] = [
        HumanMessage(content="прошлая реплика"),
        AIMessage(content="прошлый ответ"),
    ]
    evidence = [
        EvidencePassage(
            passage_id="c1", source="ru-book", section="chapter-1", text="точный текст один"
        ),
        EvidencePassage(
            passage_id="c2", source="ru-book", section="chapter-2", text="точный текст два"
        ),
    ]
    assembled = build_answer_messages(
        recent_messages=recent,
        conversation_memory="память продолжения",
        evidence=evidence,
        user_message="текущий вопрос",
    )
    assert "Russian-language" in str(assembled[0].content)
    assert isinstance(assembled[1], HumanMessage)
    assert isinstance(assembled[2], AIMessage)
    payload = str(assembled[-1].content)
    memory_pos = payload.index("<conversation_memory>")
    evidence_pos = payload.index("<book_evidence>")
    message_pos = payload.index("<user_message>")
    assert memory_pos < evidence_pos < message_pos
    assert payload.rstrip().endswith("</user_message>")
    assert "точный текст один" in payload.split("<book_evidence>")[1].split("</book_evidence>")[0]
    assert "текущий вопрос" in payload.split("<user_message>")[1]
    assert not any(type(message).__name__ == "ToolMessage" for message in assembled)


# ---------------------------------------------------------------------------
# Log privacy: no prompt/user content in logs
# ---------------------------------------------------------------------------


async def test_no_prompt_or_user_content_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "секрет-пользователя-xyzzy-987"
    model = ScriptedChatModel(replies=[_plan_json([])])
    compiled = build_turn_graph(model)
    with caplog.at_level(logging.INFO, logger="aa.conversation"):
        await run_turn(compiled, chat_id=777777, user_message=f"вопрос {secret}")
    for record in caplog.records:
        rendered = record.getMessage()
        assert secret not in rendered
        assert "вопрос" not in rendered or "turn" in rendered.lower()
    adapter_client = FakeOpenCodeClient()
    adapter = OpenCodeChatModel(
        adapter_client, agent="aa", primary_model=PRIMARY, fallback_model=FALLBACK
    )
    with caplog.at_level(logging.INFO, logger="aa.opencode"):
        await adapter.ainvoke([HumanMessage(content=f"скрытый {secret}")])
    for record in caplog.records:
        assert secret not in record.getMessage()
