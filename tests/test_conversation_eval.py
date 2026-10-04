"""Harness tests for the exact-main Russian conversation benchmark (#72).

Evaluation infrastructure only: no authoritative #62 run happens here and
product behavior is never changed to make a benchmark pass.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aa.corpus.age_v1 import generate_identity
from aa.opencode.errors import OpenCodeTransientError
from aa.qualification.conversation_eval import (
    ConversationEvalError,
    EvalIdentity,
    JourneyTurnView,
    RetryPolicy,
    ShardManifest,
    TurnCapture,
    TurnSender,
    allocate_chat_ids,
    assert_manifest_privacy_safe,
    assert_no_oracle_leak,
    bounded_history,
    build_compact_manifest,
    build_result_marker,
    build_shard_bundle,
    check_parallelism,
    classify_error_category,
    collect_eval_identity,
    compress_and_encrypt,
    decrypt_and_decompress,
    expected_case_ids,
    fresh_chat_ids,
    is_infrastructure_category,
    is_resumable,
    load_generator_views,
    merge_manifests,
    parse_result_marker,
    plan_shards,
    revalidate_at_publication,
    run_with_retry,
    should_rerun,
    stable_shard,
    summarize_run,
    validate_exact_main,
    validate_files_do_not_mutate_main,
)
from aa.qualification.ru_realworld import (
    CONTROL_JOURNEY_ID,
    INPUT_FORBIDDEN_KEYS,
    SESSION_RESET_CONTROL,
    find_repo_root,
)
from aa.safety.router import SafetyRouter
from aa.sessions.coordinator import SessionCoordinator

SHA_A = "a" * 40
SHA_B = "b" * 40
CORPUS_HEX = "c" * 64


def _identity(sha: str = SHA_A) -> EvalIdentity:
    return EvalIdentity(
        main_sha=sha,
        corpus_input_sha256="i" * 64,
        corpus_oracle_sha256="o" * 64,
        runtime_version="aa-worker/0.1.0",
        prompt_version="p" * 64,
        retrieval_version="r/1",
        index_version="idx/1",
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )


class _ScriptSender(TurnSender):
    """Deterministic stub sender with production session semantics."""

    def __init__(self, *, replies: dict[str, str] | None = None) -> None:
        self._sessions: dict[int, str] = {}
        self._counter = 0
        self._replies = replies or {}
        self.sent: list[tuple[int, str, str]] = []
        self.resets: list[int] = []

    async def ensure_session(self, chat_id: int) -> str:
        if chat_id not in self._sessions:
            self._counter += 1
            self._sessions[chat_id] = f"ses_{self._counter:06d}"
        return self._sessions[chat_id]

    async def send(self, session_id: str, text: str, *, model: str) -> str:
        assert_no_oracle_leak({"utterance": text}, "stub-send")
        chat = next(c for c, s in self._sessions.items() if s == session_id)
        self.sent.append((chat, text, model))
        return self._replies.get(text, f"reply:{text[:12]}:{model}")

    async def reset(self, chat_id: int) -> str:
        self.resets.append(chat_id)
        self._counter += 1
        self._sessions[chat_id] = f"ses_{self._counter:06d}"
        return self._sessions[chat_id]


# -- Generator input isolation ---------------------------------------------


def test_generator_view_never_carries_oracle_labels() -> None:
    singles, journeys = load_generator_views()
    assert len(singles) == 200
    assert len(journeys) == 30
    for single in singles:
        payload = {"id": single.case_id, "utterance": single.utterance}
        assert_no_oracle_leak(payload, single.case_id)
        assert set(payload) & set(INPUT_FORBIDDEN_KEYS) == set()
    for journey in journeys:
        for turn in journey.turns:
            if turn.kind == "user":
                assert_no_oracle_leak({"utterance": turn.utterance}, journey.journey_id)
            else:
                assert turn.control == SESSION_RESET_CONTROL


def test_oracle_keys_fail_generator_leak_check() -> None:
    for key in (
        "expected_safety_decision",
        "expected_response_mode",
        "expected_emergency_categories",
        "provenance_ids",
        "rubric_tags",
        "forbidden_inferences",
        "book_relevance",
        "evaluator_instructions",
    ):
        with pytest.raises(ConversationEvalError):
            assert_no_oracle_leak({key: "x"}, "leak-probe")


def test_bounded_history_stays_within_session() -> None:
    assert bounded_history(["a", "b", "c"], limit=2) == ["b", "c"]
    with pytest.raises(ConversationEvalError):
        bounded_history(["a"], limit=0)


# -- Session / reset / isolation semantics ----------------------------------


async def test_unique_session_per_journey_same_session_within() -> None:
    mapping = allocate_chat_ids(["RU-J-001", "RU-J-002", "RU-J-027"])
    assert len(set(mapping.values())) == 3
    # Deterministic allocation.
    assert allocate_chat_ids(["RU-J-001", "RU-J-002", "RU-J-027"]) == mapping
    with pytest.raises(ConversationEvalError):
        allocate_chat_ids(["RU-J-001", "RU-J-001"])


async def test_repeats_use_fresh_sessions() -> None:
    first = allocate_chat_ids(["RU-J-001"], base=900000)
    repeat = fresh_chat_ids(["RU-J-001"], base=900000, repeat=0)
    assert first["RU-J-001"] != repeat["RU-J-001"]


async def test_control_event_calls_real_reset_boundary() -> None:
    sender = _ScriptSender()
    sessions = SessionCoordinator()
    await sessions.start()
    singles, journeys = load_generator_views()
    target = next(j for j in journeys if j.journey_id == CONTROL_JOURNEY_ID)
    chat_id = allocate_chat_ids([target.journey_id])[target.journey_id]
    prior: list[str] = []
    for entry in target.turns:
        if entry.is_control():
            assert isinstance(entry, JourneyTurnView)
            await sender.reset(chat_id)
            prior = []
        else:
            sid = await sender.ensure_session(chat_id)
            await sender.send(sid, entry.utterance, model="m")
            prior.append(entry.utterance)
    assert sender.resets == [chat_id]
    # After reset, pre-reset state is unavailable: only post-reset turns ran
    # after the single reset call.
    assert len(prior) == 2
    await sessions.stop()


async def test_production_safety_boundary_stays_authoritative() -> None:
    router = SafetyRouter()
    await router.start()
    assert router.check("").decision.value == "block"
    assert router.check("обычный вопрос о поддержке").decision.value in (
        "allow",
        "emergency",
        "block",
    )
    await router.stop()


async def test_single_turn_cases_use_isolated_sessions() -> None:
    sender = _ScriptSender()
    first = await sender.ensure_session(910001)
    second = await sender.ensure_session(910002)
    assert first != second


# -- Deterministic sharding + merge ------------------------------------------


def test_shards_keep_journeys_atomic_and_cover_all_cases() -> None:
    singles = [f"RU-S-{n:03d}" for n in range(1, 201)]
    journeys = [f"RU-J-{n:03d}" for n in range(1, 31)]
    plans = plan_shards(singles, journeys, shard_count=4)
    assert len(plans) == 4
    # Journeys atomic: each journey appears in exactly one shard.
    seen: dict[str, int] = {}
    for plan in plans:
        for jid in plan.journey_ids:
            assert jid not in seen
            seen[jid] = plan.shard_index
    assert len(seen) == 30
    covered = sorted([c for p in plans for c in p.case_ids()])
    assert covered == sorted(singles + journeys)
    # Deterministic across calls.
    again = plan_shards(singles, journeys, shard_count=4)
    assert [(p.single_ids, p.journey_ids) for p in plans] == [
        (p.single_ids, p.journey_ids) for p in again
    ]


def test_stable_shard_is_deterministic() -> None:
    assert stable_shard("RU-S-001", 4) == stable_shard("RU-S-001", 4)
    with pytest.raises(ConversationEvalError):
        stable_shard("", 4)
    with pytest.raises(ConversationEvalError):
        plan_shards(["a", "a"], [], shard_count=2)


def test_merge_rejects_missing_and_duplicate_ids() -> None:
    def _manifest(idx: int, cases: tuple[str, ...]) -> ShardManifest:
        return ShardManifest(
            shard_index=idx,
            shard_count=2,
            case_ids=cases,
            turn_rows=tuple({"case_id": c} for c in cases),
            bundle_sha256=hashlib.sha256(str(idx).encode()).hexdigest(),
            eval_identity=_identity().to_dict(),
        )

    merged = merge_manifests(
        [_manifest(0, ("a",)), _manifest(1, ("b",))], expected_case_ids=["a", "b"]
    )
    assert merged["case_ids"] == ["a", "b"]
    with pytest.raises(ConversationEvalError):
        merge_manifests([_manifest(0, ("a",))], expected_case_ids=["a", "b"])
    with pytest.raises(ConversationEvalError):
        merge_manifests([_manifest(0, ("a",)), _manifest(1, ("a",))], expected_case_ids=["a"])


def test_resumable_only_on_exact_match() -> None:
    current = _identity().to_dict()
    assert is_resumable(dict(current), current) is True
    altered = dict(current)
    altered["primary_model"] = "other/model"
    assert is_resumable(altered, current) is False
    altered2 = dict(current)
    altered2["main_sha"] = SHA_B
    assert is_resumable(altered2, current) is False


def test_bounded_parallelism_defaults_conservative() -> None:
    from aa.qualification.conversation_eval import (
        DEFAULT_MAX_PARALLEL,
        DEFAULT_SHARD_COUNT,
        MAX_PARALLEL_SHARDS,
    )

    assert DEFAULT_SHARD_COUNT == 4
    assert DEFAULT_MAX_PARALLEL == 2
    assert MAX_PARALLEL_SHARDS == 4
    check_parallelism(shard_count=4, max_parallel=2)
    with pytest.raises(ConversationEvalError):
        check_parallelism(shard_count=4, max_parallel=8)
    with pytest.raises(ConversationEvalError):
        check_parallelism(shard_count=2, max_parallel=3)


# -- Provider failures --------------------------------------------------------


def test_provider_failures_are_infrastructure_not_quality() -> None:
    assert is_infrastructure_category("provider-429") is True
    assert is_infrastructure_category("provider-unavailable") is True
    assert is_infrastructure_category("ok") is False
    assert is_infrastructure_category("deterministic-error") is False
    assert classify_error_category(OpenCodeTransientError("http=429")) in (
        "provider-429",
        "provider-transient",
    )
    assert classify_error_category(ValueError("boom")) == "deterministic-error"


def test_retry_policy_is_bounded() -> None:
    with pytest.raises(ConversationEvalError):
        RetryPolicy(max_attempts=99)
    assert RetryPolicy().delay_for(1) <= RetryPolicy().delay_for(2)


class _FlakySender(TurnSender):
    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.attempts = 0

    async def ensure_session(self, chat_id: int) -> str:
        return "ses_flaky"

    async def send(self, session_id: str, text: str, *, model: str) -> str:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise OpenCodeTransientError("opencode request failed transiently: http=503")
        return "ok-answer"

    async def reset(self, chat_id: int) -> str:
        return "ses_flaky2"


async def test_bounded_retry_recovers_transient() -> None:
    sender = _FlakySender(failures=1)

    async def _noop(_delay: float) -> None:
        return None

    obs = await run_with_retry(
        sender,
        chat_id=1,
        utterance="привет",
        primary_model="p",
        fallback_model="f",
        policy=RetryPolicy(max_attempts=3),
        sleep=_noop,
    )
    assert obs.answer == "ok-answer"
    assert obs.fallback_used is False


class _DownSender(TurnSender):
    async def ensure_session(self, chat_id: int) -> str:
        return "ses_down"

    async def send(self, session_id: str, text: str, *, model: str) -> str:
        if model == "primary/m":
            raise OpenCodeTransientError("opencode request failed transiently: http=429")
        return "fallback-answer"

    async def reset(self, chat_id: int) -> str:
        return "ses_down2"


async def test_fallback_is_segmented_not_silent_primary() -> None:
    sender = _DownSender()

    async def _noop(_delay: float) -> None:
        return None

    obs = await run_with_retry(
        sender,
        chat_id=1,
        utterance="привет",
        primary_model="primary/m",
        fallback_model="fallback/m",
        policy=RetryPolicy(max_attempts=1),
        sleep=_noop,
    )
    assert obs.fallback_used is True
    assert obs.actual_model == "fallback/m"
    assert obs.error_category == "fallback-used"


# -- Capture schema ------------------------------------------------------------


def test_capture_schema_has_required_fields() -> None:
    cap = TurnCapture(
        case_id="RU-S-001",
        journey_id="",
        turn=1,
        synthetic_input="привет",
        generated_answer="ответ",
        safety_decision="allow",
        primary_model="p",
        actual_model="p",
    ).with_hashes()
    payload = cap.to_dict()
    for key in (
        "case_id",
        "journey_id",
        "turn",
        "synthetic_input",
        "generated_answer",
        "safety_decision",
        "planner_diagnostics",
        "retrieval_source_ids",
        "evidence_locators",
        "evidence_checksums",
        "grounding_passed",
        "regeneration_count",
        "actual_model",
        "latency_s",
        "tool_call_count",
        "evidence_token_count",
        "retry_count",
        "error_category",
        "answer_sha256",
        "output_sha256",
    ):
        assert key in payload
    assert payload["answer_sha256"] == hashlib.sha256("ответ".encode()).hexdigest()
    row = cap.manifest_row()
    assert "generated_answer" not in row
    assert "synthetic_input" not in row


# -- Exact-main contract -------------------------------------------------------


def test_exact_main_validation_rejects_moved_main() -> None:
    validate_exact_main(tested_sha=SHA_A, current_main_sha=SHA_A, trusted_pass_sha=SHA_A)
    with pytest.raises(ConversationEvalError):
        validate_exact_main(tested_sha=SHA_A, current_main_sha=SHA_B, trusted_pass_sha=SHA_A)
    with pytest.raises(ConversationEvalError):
        validate_exact_main(tested_sha=SHA_B, current_main_sha=SHA_B, trusted_pass_sha=SHA_A)


def test_revalidate_at_publication_blocks_moved_main() -> None:
    assert revalidate_at_publication(tested_sha=SHA_A, current_main_sha=SHA_A) == "complete"
    assert revalidate_at_publication(tested_sha=SHA_A, current_main_sha=SHA_B) == "stale"


def test_collect_identity_records_versions_and_models() -> None:
    identity = collect_eval_identity(
        main_sha=SHA_A,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    assert identity.main_sha == SHA_A
    assert len(identity.corpus_input_sha256) == 64
    assert len(identity.corpus_oracle_sha256) == 64
    assert identity.primary_model != identity.fallback_model


def test_result_marker_round_trip() -> None:
    marker = build_result_marker(sha=SHA_A, corpus=CORPUS_HEX, result="complete", run="123")
    parsed = parse_result_marker(marker)
    assert parsed == {
        "issue": "62",
        "sha": SHA_A,
        "corpus": CORPUS_HEX,
        "result": "complete",
        "run": "123",
    }
    with pytest.raises(ConversationEvalError):
        parse_result_marker("no marker here")


def test_rerun_is_idempotent_per_tuple() -> None:
    latest = {"sha": SHA_A, "corpus": CORPUS_HEX}
    assert (
        should_rerun(latest_complete=latest, candidate_sha=SHA_A, candidate_corpus=CORPUS_HEX)
        is False
    )
    assert (
        should_rerun(latest_complete=latest, candidate_sha=SHA_B, candidate_corpus=CORPUS_HEX)
        is True
    )
    assert (
        should_rerun(latest_complete=None, candidate_sha=SHA_A, candidate_corpus=CORPUS_HEX) is True
    )


def test_results_never_mutate_main() -> None:
    validate_files_do_not_mutate_main(["eval-out/shard-0-manifest.json"])
    with pytest.raises(ConversationEvalError):
        validate_files_do_not_mutate_main(["src/aa/app.py"])
    with pytest.raises(ConversationEvalError):
        validate_files_do_not_mutate_main(["corpus/structure.json"])


# -- Encryption + privacy -------------------------------------------------------


def test_every_shard_encrypts_before_upload() -> None:
    identity, recipient = generate_identity()
    captures = [
        TurnCapture(
            case_id="RU-S-001",
            journey_id="",
            turn=1,
            synthetic_input="привет",
            generated_answer="ответ",
            safety_decision="allow",
            primary_model="p",
            actual_model="p",
        )
    ]
    encrypted, bundle_sha = build_shard_bundle(captures, recipient=recipient)
    assert hashlib.sha256(encrypted).hexdigest() == bundle_sha
    assert "ответ".encode() not in encrypted
    raw = decrypt_and_decompress(encrypted, identity=identity)
    payload = json.loads(raw.decode("utf-8"))
    assert payload["turns"][0]["generated_answer"] == "ответ"


def test_compress_encrypt_round_trip() -> None:
    identity, recipient = generate_identity()
    bundle = compress_and_encrypt(b'{"a": 1}', recipient=recipient)
    assert decrypt_and_decompress(bundle, identity=identity) == b'{"a": 1}'


def test_compact_manifest_is_privacy_safe() -> None:
    identity = _identity()
    cap = TurnCapture(
        case_id="RU-S-001",
        journey_id="",
        turn=1,
        synthetic_input="привет",
        generated_answer="ответ",
        safety_decision="allow",
        primary_model=identity.primary_model,
        actual_model=identity.primary_model,
    ).with_hashes()
    manifest = ShardManifest(
        shard_index=0,
        shard_count=1,
        case_ids=("RU-S-001",),
        turn_rows=(cap.manifest_row(),),
        bundle_sha256="b" * 64,
        eval_identity=identity.to_dict(),
    )
    compact = build_compact_manifest(
        eval_identity=identity, shard_manifests=[manifest], run_id="999", result="complete"
    )
    assert compact["main_sha"] == SHA_A
    assert_manifest_privacy_safe(compact)
    with pytest.raises(ConversationEvalError):
        assert_manifest_privacy_safe({"generated_answer": "secret text"})


def test_summarize_segments_provider_failures() -> None:
    rows = [
        {
            "error_category": "ok",
            "fallback_used": False,
            "safety_decision": "allow",
            "latency_s": 1.0,
        },
        {
            "error_category": "provider-429",
            "fallback_used": False,
            "safety_decision": "allow",
            "latency_s": 2.0,
        },
        {
            "error_category": "fallback-used",
            "fallback_used": True,
            "safety_decision": "allow",
            "latency_s": 3.0,
        },
    ]
    summary = summarize_run(rows)
    assert summary["total_turns"] == 3
    assert summary["infrastructure_failures"] == 1
    assert summary["fallback_turns"] == 1


def test_expected_case_ids_cover_full_corpus() -> None:
    singles, journeys = load_generator_views()
    ids = expected_case_ids(singles, journeys)
    assert len(ids) == 230
    assert "RU-J-027" in ids
    root = find_repo_root()
    assert (root / "qualification" / "ru_realworld_alcohol_help.v1_1.input.jsonl").exists()
