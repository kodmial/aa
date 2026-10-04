"""Regression fixtures for the v2 LangGraph turn foundation (issue #113).

These tests pin the foundation contract; they are not special-case
routing rules. Every ordinary turn reaches the same hidden planner path
regardless of wording.
"""

from __future__ import annotations

import ast
import json
import logging
import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aa.conversation import graph as graph_module
from aa.conversation.graph import build_turn_graph, turn_input
from aa.conversation.graph_state import TurnState, initial_state
from aa.conversation.memory import (
    COMPACTION_TRIGGER_TOKENS,
    CONTEXT_REFERENCE_TOKENS,
    RECENT_KEEP_TOKENS,
    MemoryConfig,
    SqliteCheckpointerFactory,
    build_summarization_middleware,
    count_message_tokens,
    default_memory_config,
    maybe_compact_state,
    needs_compaction,
    split_keep_window,
    thread_id_for_chat,
)
from aa.conversation.model_adapter import (
    ANSWER_AGENT_V2,
    PLANNER_AGENT_V2,
    SUMMARIZER_AGENT_V2,
    OpenCodeChatModel,
)
from aa.conversation.planner_node import build_planner_messages, parse_plan_text, run_planner
from aa.conversation.planner_schema import (
    MAX_QUERIES,
    MIN_NONEMPTY_QUERIES,
    QueryPlan,
    QueryPlanValidationError,
    validate_query_plan,
)
from aa.conversation.prompt_builder import (
    EvidencePassage,
    build_answer_messages,
    render_turn_context,
)
from aa.conversation.v2_prompts import (
    load_aa_agent_system_v2,
    load_planner_system_v2,
    load_summarizer_system_v2,
)
from aa.opencode.client import FakeOpenCodeClient, OpenCodeClient
from aa.opencode.errors import OpenCodeTimeoutError, OpenCodeTransientError

CYRILLIC_START = 0x0400
CYRILLIC_END = 0x04FF


def _has_cyrillic(text: str) -> bool:
    return any(CYRILLIC_START <= ord(char) <= CYRILLIC_END for char in text)


def _twelve_queries() -> list[str]:
    return [f"трезвость поддержка вопрос {index}" for index in range(12)]


def _plan_json(queries: list[str]) -> str:
    return json.dumps({"queries": queries}, ensure_ascii=False)


def _script_model(
    replies: list[str], seen: list[list[BaseMessage]] | None = None
) -> Runnable[list[BaseMessage], BaseMessage]:
    queue = list(replies)

    def _reply(messages: list[BaseMessage]) -> BaseMessage:
        if seen is not None:
            seen.append(list(messages))
        if not queue:
            raise AssertionError("script model called more times than scripted")
        return AIMessage(content=queue.pop(0))

    return RunnableLambda(_reply)


# ---------------------------------------------------------------------------
# Planner schema: exactly 0 or 10..16 distinct queries.
# ---------------------------------------------------------------------------


def test_planner_accepts_empty_plan() -> None:
    assert validate_query_plan(QueryPlan(queries=[])).queries == []


def test_planner_accepts_ten_and_sixteen() -> None:
    assert len(validate_query_plan(QueryPlan(queries=_twelve_queries()[:10])).queries) == 10
    extended = _twelve_queries() + ["a", "b", "c", "d"]
    assert len(validate_query_plan(QueryPlan(queries=extended)).queries) == 16


@pytest.mark.parametrize("count", [1, 2, 5, 9, 17, 20])
def test_planner_rejects_other_cardinalities(count: int) -> None:
    queries = [f"запрос {index}" for index in range(count)]
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(QueryPlan(queries=queries))


def test_planner_dedupes_whitespace_case_duplicates() -> None:
    plan = validate_query_plan(
        QueryPlan(queries=["  Трезвость  "] + [f"запрос {index}" for index in range(9)])
    )
    assert len(plan.queries) == 10
    with pytest.raises(QueryPlanValidationError):
        # Nine distinct after collapsing one exact duplicate.
        duplicates = ["Трезвость", "трезвость "] + [f"запрос {index}" for index in range(8)]
        validate_query_plan(QueryPlan(queries=duplicates))


def test_planner_schema_has_only_queries_field() -> None:
    assert set(QueryPlan.model_fields) == {"queries"}
    assert MIN_NONEMPTY_QUERIES == 10
    assert MAX_QUERIES == 16


def test_parse_plan_text_uses_framework_parser() -> None:
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=QueryPlan)
    plan = parse_plan_text(_plan_json(_twelve_queries()), parser=parser)
    assert len(plan.queries) == 12
    with pytest.raises(QueryPlanValidationError):
        parse_plan_text(_plan_json(["один"]), parser=parser)
    with pytest.raises(QueryPlanValidationError):
        parse_plan_text("not json at all {{{", parser=parser)


# ---------------------------------------------------------------------------
# Mandatory planner: every ordinary turn reaches the same hidden path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "А что ты можешь?",
        "Тогда зачем ты?",
        "почему?",
        "а он?",
        "а дальше?",
        "и что потом?",
        "привет",
        "спасибо",
        "необычная перефразировка про вечернюю тревогу",
        "бухаю каждый вечер, жинка ругается",
    ],
)
async def test_every_ordinary_turn_invokes_planner(text: str) -> None:
    seen: list[list[BaseMessage]] = []
    model = _script_model([_plan_json([])], seen)
    graph = build_turn_graph(planner_model=model)
    result = await graph.ainvoke(turn_input(text))
    assert result["route"] == "normal"
    assert result["planner_invoked"] is True
    assert result["search_queries"] == []
    # The planner saw the live turn as structured input exactly once.
    assert len(seen) == 1
    prompt_text = "\n".join(str(item.content) for item in seen[0] if isinstance(item, BaseMessage))
    assert text in prompt_text


async def test_planner_output_lands_in_state_not_messages() -> None:
    seen: list[list[BaseMessage]] = []
    model = _script_model([_plan_json(_twelve_queries())], seen)
    graph = build_turn_graph(planner_model=model)
    result = await graph.ainvoke(turn_input("как справиться с тягой?"))
    assert result["search_queries"] == _twelve_queries()
    roles = [item.type for item in result["messages"]]
    assert roles == ["human"]
    assert all(isinstance(item, HumanMessage) for item in result["messages"])


async def test_planner_retry_repairs_invalid_first_output() -> None:
    model = _script_model(["не json", _plan_json(_twelve_queries())])
    plan = await run_planner("тяга вечером", model=model)
    assert len(plan.queries) == 12


async def test_planner_fails_closed_after_bounded_retries() -> None:
    model = _script_model(["мусор", "мусор"])
    graph = build_turn_graph(planner_model=model)
    result = await graph.ainvoke(turn_input("тяга"))
    assert result["planner_invoked"] is True
    assert result["search_queries"] == []
    assert "planner_error" in result["retry_state"]


async def test_application_commands_skip_planner() -> None:
    def _boom(messages: list[BaseMessage]) -> BaseMessage:
        raise AssertionError("planner must not run for application commands")

    graph = build_turn_graph(planner_model=RunnableLambda(_boom))
    for command in ("/start", "/new"):
        result = await graph.ainvoke(turn_input(command))
        assert result["route"] == "command"
        assert result["planner_invoked"] is False


# ---------------------------------------------------------------------------
# Memory: summary + recent resolve follow-ups without becoming evidence.
# ---------------------------------------------------------------------------


def test_planner_input_carries_summary_and_recent_roles() -> None:
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=QueryPlan)
    recent = [
        HumanMessage(content="поссорился с женой"),
        AIMessage(content="понимаю, это тяжело"),
    ]
    messages = build_planner_messages(
        user_message="почему?",
        summary="пользователь поссорился с женой",
        recent=recent,
        parser=parser,
    )
    assert messages[0].type == "system"
    assert "Return only the structured output" in str(messages[0].content)
    body = str(messages[1].content)
    assert "почему?" in body
    assert "поссорился с женой" in body
    assert "user: поссорился с женой" in body
    assert "assistant: понимаю, это тяжело" in body


def test_memory_defaults_are_token_driven() -> None:
    config = default_memory_config(checkpoint_dir=pathlib.Path("/tmp/aa-v2-test"))
    assert config.context_reference_tokens == CONTEXT_REFERENCE_TOKENS == 200_000
    assert config.trigger_tokens == COMPACTION_TRIGGER_TOKENS == 120_000
    assert config.keep_tokens == RECENT_KEEP_TOKENS == 40_000


def test_compaction_trigger_is_token_based_not_turn_count() -> None:
    short: list[BaseMessage] = [HumanMessage(content="да")]
    assert needs_compaction(short, trigger_tokens=COMPACTION_TRIGGER_TOKENS) is False
    long_text = "слово " * 100_000
    long: list[BaseMessage] = [HumanMessage(content=long_text)]
    assert count_message_tokens(long) >= COMPACTION_TRIGGER_TOKENS
    assert needs_compaction(long, trigger_tokens=COMPACTION_TRIGGER_TOKENS) is True


def test_split_keep_window_never_splits_tool_pairs() -> None:
    from langchain_core.messages import ToolMessage

    call = AIMessage(
        content="",
        tool_calls=[{"name": "book_read", "args": {}, "id": "call-1", "type": "tool_call"}],
    )
    tool = ToolMessage(content="text", tool_call_id="call-1")
    messages: list[BaseMessage] = [HumanMessage(content="q"), call, tool]
    split = split_keep_window(messages, keep_tokens=1)
    assert split <= 1
    retained = messages[split:]
    if any(item.type == "tool" for item in retained):
        assert retained[0].type != "tool" or split == 0


async def test_maybe_compact_summarizes_old_and_keeps_recent() -> None:
    long_text = "разговор о трезвости " * 8_000
    messages: list[BaseMessage] = [
        HumanMessage(content=long_text),
        AIMessage(content=long_text),
        HumanMessage(content="почему?"),
    ]
    summary_model = _script_model(["краткое резюме"])
    config = MemoryConfig(
        checkpoint_dir=pathlib.Path("/tmp/aa-v2-test"),
        trigger_tokens=10,
        keep_tokens=50,
    )
    summary, retained = await maybe_compact_state(
        messages=messages, previous_summary="", model=summary_model, config=config
    )
    assert summary == "краткое резюме"
    assert retained
    assert str(retained[-1].content) == "почему?"


async def test_maybe_compact_passthrough_below_trigger() -> None:
    messages: list[BaseMessage] = [HumanMessage(content="привет")]

    def _boom(batch: list[BaseMessage]) -> BaseMessage:
        raise AssertionError("summarizer must not run below the token trigger")

    summary, retained = await maybe_compact_state(
        messages=messages,
        previous_summary="старое",
        model=RunnableLambda(_boom),
        config=default_memory_config(checkpoint_dir=pathlib.Path("/tmp/aa-v2-test")),
    )
    assert summary == "старое"
    assert retained == messages


def test_summarization_middleware_uses_framework_trigger_keep() -> None:
    adapter = OpenCodeChatModel(
        FakeOpenCodeClient(),
        agent=SUMMARIZER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    middleware = build_summarization_middleware(
        adapter,
        config=MemoryConfig(checkpoint_dir=pathlib.Path("/tmp/aa-v2-test")),
    )
    assert middleware.trigger == ("tokens", COMPACTION_TRIGGER_TOKENS)
    assert middleware.keep == ("tokens", RECENT_KEEP_TOKENS)
    assert "Conversation memory is not an authority" in middleware.summary_prompt


def test_thread_mapping_is_deterministic_and_opaque() -> None:
    first = thread_id_for_chat(12345)
    assert first == thread_id_for_chat(12345)
    assert first != thread_id_for_chat(54321)
    assert "12345" not in first


# ---------------------------------------------------------------------------
# Checkpointer wiring behind a replaceable factory.
# ---------------------------------------------------------------------------


async def test_sqlite_factory_db_lives_in_private_dir(tmp_path: pathlib.Path) -> None:
    factory = SqliteCheckpointerFactory(MemoryConfig(checkpoint_dir=tmp_path / "job-1"))
    assert factory.db_path.parent == tmp_path / "job-1"
    async with factory.checkpointer() as saver:
        assert isinstance(saver, AsyncSqliteSaver)
    assert factory.db_path.exists()
    factory.cleanup()
    assert not (tmp_path / "job-1").exists()


async def test_graph_checkpointer_persists_thread_state(tmp_path: pathlib.Path) -> None:
    factory = SqliteCheckpointerFactory(MemoryConfig(checkpoint_dir=tmp_path))
    async with factory.checkpointer() as saver:
        graph = build_turn_graph(
            planner_model=_script_model([_plan_json([]), _plan_json([])]),
            checkpointer=saver,
        )
        config: RunnableConfig = {"configurable": {"thread_id": thread_id_for_chat(777)}}
        first = await graph.ainvoke(turn_input("первое сообщение"), config=config)
        assert first["planner_invoked"] is True
        second = await graph.ainvoke(turn_input("почему?", summary=None), config=config)
        human_texts = [str(item.content) for item in second["messages"] if item.type == "human"]
        assert "первое сообщение" in human_texts
        assert "почему?" in human_texts


# ---------------------------------------------------------------------------
# Prompt assembly contract for the future answer node.
# ---------------------------------------------------------------------------


def test_prompt_assembly_keeps_blocks_separated_user_last() -> None:
    recent = [
        HumanMessage(content="поссорился с женой"),
        AIMessage(content="понимаю"),
    ]
    passages = [
        EvidencePassage(passage_id="a/sec#1", source="a", section="sec", text="точный текст")
    ]
    messages = build_answer_messages(
        recent=recent,
        summary="пользователь поссорился",
        passages=passages,
        user_message="почему?",
    )
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert messages[2].type == "ai"
    final = str(messages[-1].content)
    memory_pos = final.index("<conversation_memory>")
    evidence_pos = final.index("<book_evidence>")
    user_pos = final.index("<user_message>")
    assert memory_pos < evidence_pos < user_pos
    assert final.rstrip().endswith("</user_message>")
    assert "почему?" in final.split("<user_message>")[1]
    assert "точный текст" in final.split("<book_evidence>")[1].split("</book_evidence>")[0]
    assert not any(item.type == "tool" for item in messages)


def test_turn_context_marks_empty_blocks_explicitly() -> None:
    rendered = render_turn_context(summary="", passages=[], user_message="привет")
    assert "(no prior conversation)" in rendered
    assert "(no book evidence supplied for this turn)" in rendered
    assert rendered.rstrip().endswith("</user_message>")


# ---------------------------------------------------------------------------
# Adapter: ephemeral transport only, fakeable, fallback policy reused.
# ---------------------------------------------------------------------------


class _RecordingClient(FakeOpenCodeClient):
    def __init__(self) -> None:
        super().__init__()
        self.created = 0
        self.deleted = 0
        self.agents: list[str] = []
        self.models: list[str] = []

    async def create_session(self, title: str = "") -> Any:
        self.created += 1
        return await super().create_session(title)

    async def delete_session(self, session_id: str) -> bool:
        self.deleted += 1
        return await super().delete_session(session_id)

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
    ) -> str:
        self.agents.append(agent)
        self.models.append(model)
        return await super().send_message(
            session_id, text, timeout=timeout, agent=agent, model=model
        )


async def test_adapter_uses_ephemeral_sessions() -> None:
    client = _RecordingClient()
    model = OpenCodeChatModel(
        client,
        agent=PLANNER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    reply = await model.ainvoke([HumanMessage(content="скрытый вызов")])
    assert isinstance(reply, AIMessage)
    assert client.created == 1
    assert client.deleted == 1
    assert client.agents == [PLANNER_AGENT_V2]
    assert client.models == ["opencode/muse-spark-1.3-contributor-free"]


class _FlakyClient(FakeOpenCodeClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
    ) -> str:
        self.calls += 1
        if self.calls == 1:
            raise OpenCodeTransientError("provider busy")
        return "резервный ответ"


async def test_adapter_falls_back_on_transient_error() -> None:
    client = _FlakyClient()
    model = OpenCodeChatModel(
        client,
        agent=SUMMARIZER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    reply = await model.ainvoke([HumanMessage(content="резюмируй")])
    assert str(reply.content) == "резервный ответ"
    assert client.calls == 2


async def test_adapter_hidden_calls_do_not_accumulate_history() -> None:
    client = FakeOpenCodeClient()
    model = OpenCodeChatModel(
        client,
        agent=ANSWER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    await model.ainvoke([HumanMessage(content="первый")])
    await model.ainvoke([HumanMessage(content="второй")])
    remaining: dict[str, Any] = client._sessions
    assert remaining == {}


def test_adapter_carries_no_semantic_policy() -> None:
    source_path = pathlib.Path(graph_module.__file__).parent / "model_adapter.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert all(not name.startswith("aa.retrieval") for name in imported)
    assert "aa.conversation.orchestrator" not in imported
    assert "aa.conversation.meta" not in imported
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert defined <= {
        "OpenCodeChatModel",
        "render_messages_text",
        "__init__",
        "_message_text",
        "_run_coro_sync",
        "opencode_client",
        "with_agent",
        "_invoke_ephemeral",
        "_ainvoke_text",
        "_generate",
        "_agenerate",
        "_llm_type",
        "_runner",
    }


# ---------------------------------------------------------------------------
# Privacy: no prompt or user content in logs.
# ---------------------------------------------------------------------------


async def test_no_user_content_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret = "секретная фраза про срыв семьсот"
    model = _script_model([_plan_json([])])
    graph = build_turn_graph(planner_model=model)
    with caplog.at_level(logging.INFO, logger="aa"):
        await graph.ainvoke(turn_input(secret))
    assert secret not in caplog.text


async def test_adapter_logs_nothing_with_content(caplog: pytest.LogCaptureFixture) -> None:
    secret = "тайный текст пользователя девять"
    client = FakeOpenCodeClient()
    model = OpenCodeChatModel(
        client,
        agent=PLANNER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    with caplog.at_level(logging.INFO, logger="aa"):
        await model.ainvoke([HumanMessage(content=secret)])
    assert secret not in caplog.text


# ---------------------------------------------------------------------------
# Boundaries: new graph imports nothing legacy-semantic; prompts versioned.
# ---------------------------------------------------------------------------


def test_new_graph_has_no_legacy_semantic_imports() -> None:
    package = pathlib.Path(graph_module.__file__).parent
    forbidden = (
        "is_substantive",
        "_TRIVIAL_NORMALIZED",
        "_FOLLOWUP_INTERROGATIVES",
        "_SUBSTANTIVE_KEYWORDS",
        "_SLANG_EXPANSIONS",
        "_THEME_MARKERS",
        "_BROAD_COVERAGE_QUERIES",
        "ru-query-plan-v1",
        "is_meta_capability_request",
    )
    modules = (
        "graph_state.py",
        "planner_schema.py",
        "model_adapter.py",
        "planner_node.py",
        "memory.py",
        "prompt_builder.py",
        "graph.py",
        "v2_prompts.py",
    )
    for name in modules:
        source = (package / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "aa.conversation.orchestrator" not in imported, name
        assert "aa.conversation.meta" not in imported, name
        assert "aa.retrieval.planner" not in imported, name
        for snippet in forbidden:
            assert snippet not in source, f"{name}: {snippet}"


def test_prompt_artifacts_are_versioned_english() -> None:
    agent = load_aa_agent_system_v2()
    planner = load_planner_system_v2()
    summarizer = load_summarizer_system_v2()
    assert "Return only the user-facing Russian reply." in agent
    assert "all substantive ideas" in agent
    assert "Return only the structured output." in planner
    assert "10 to 16 semantically distinct Russian search queries" in planner
    assert "Conversation memory is not an authority" in summarizer
    for text in (agent, planner, summarizer):
        assert not _has_cyrillic(text)


def test_opencode_agents_keep_legacy_and_add_locked_down_v2() -> None:
    import json as json_module

    root = pathlib.Path(graph_module.__file__).parents[3]
    config = json_module.loads((root / "opencode.json").read_text(encoding="utf-8"))
    agents = config["agent"]
    legacy = agents["aa"]
    assert legacy["mode"] == "primary"
    assert legacy["permission"]["book_search"] == "allow"
    for name, prompt_file in (
        ("aa-v2", "aa-agent-system-v2.md"),
        ("aa-planner-v2", "aa-planner-system-v2.md"),
        ("aa-summarizer-v2", "aa-summarizer-system-v2.md"),
    ):
        agent = agents[name]
        assert agent["permission"] == {"*": "deny"}
        assert agent["prompt"] == "{file:./prompts/" + prompt_file + "}"
        assert (root / "prompts" / prompt_file).exists()
        assert agent["model"] == "opencode/muse-spark-1.3-contributor-free"


def test_typed_state_contract_fields() -> None:
    state = initial_state("привет")
    for field in (
        "messages",
        "conversation_summary",
        "current_user_message",
        "search_queries",
        "retrieval_hits",
        "evidence_pack",
        "draft_response",
        "grounding_result",
        "retry_state",
    ):
        assert field in state
    assert isinstance(state["messages"][0], HumanMessage)
    assert TurnState.__total__ is False


def test_opencode_client_boundary_unchanged() -> None:
    assert issubclass(FakeOpenCodeClient, OpenCodeClient)
    assert PLANNER_AGENT_V2 == "aa-planner-v2"
    assert SUMMARIZER_AGENT_V2 == "aa-summarizer-v2"
    assert ANSWER_AGENT_V2 == "aa-v2"


async def test_timeouts_fall_back_without_user_content_leak() -> None:
    class _TimeoutClient(FakeOpenCodeClient):
        async def send_message(
            self,
            session_id: str,
            text: str,
            *,
            timeout: float | None = None,
            agent: str = "",
            model: str = "",
        ) -> str:
            if model == "opencode/muse-spark-1.3-contributor-free":
                raise OpenCodeTimeoutError("slow")
            return "ok"

    model = OpenCodeChatModel(
        _TimeoutClient(),
        agent=PLANNER_AGENT_V2,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    reply = await model.ainvoke([HumanMessage(content="план")])
    assert str(reply.content) == "ok"
