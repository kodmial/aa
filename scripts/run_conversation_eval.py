"""Deterministic shard runner for the Russian conversation benchmark (#72).

Reads only the frozen input projection, executes the shard's cases against
an injected production turn boundary, and writes exactly two files:

- ``<out-dir>/shard-<i>-manifest.json`` (privacy-safe, plaintext allowed);
- ``<out-dir>/shard-<i>.tar.zst.age`` (already compressed + age-encrypted).

Plaintext answer/evidence bundles are never written. Outputs never mutate
production main paths (see ``validate_files_do_not_mutate_main``).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa.qualification.conversation_eval import (  # noqa: E402
    ConversationEvalError,
    EvalIdentity,
    JourneyView,
    RetryPolicy,
    ShardManifest,
    SingleCaseView,
    TurnCapture,
    TurnSender,
    allocate_chat_ids,
    assert_no_oracle_leak,
    bounded_history,
    build_shard_bundle,
    check_parallelism,
    collect_eval_identity,
    expected_case_ids,
    load_generator_views,
    plan_shards,
    validate_files_do_not_mutate_main,
)
from aa.qualification.ru_realworld import SESSION_RESET_CONTROL  # noqa: E402


@dataclass
class DryRunSender(TurnSender):
    """Offline sender for ``--dry-run`` (no network, no Telegram)."""

    primary_model: str = "dry-run/primary"
    sessions: dict[int, str] = None  # type: ignore[assignment]
    counter: int = 0

    def __post_init__(self) -> None:
        self.sessions = {}

    async def ensure_session(self, chat_id: int) -> str:
        assert isinstance(self.sessions, dict)
        if chat_id not in self.sessions:
            self.counter += 1
            self.sessions[chat_id] = f"ses_dryrun{self.counter:06d}"
        sid: str = self.sessions[chat_id]
        return sid

    async def send(self, session_id: str, text: str, *, model: str) -> str:
        assert_no_oracle_leak({"utterance": text}, "dry-run-send")
        return f"dry-run reply to {text[:48]}"

    async def reset(self, chat_id: int) -> str:
        assert isinstance(self.sessions, dict)
        self.counter += 1
        self.sessions[chat_id] = f"ses_dryrun{self.counter:06d}"
        sid: str = self.sessions[chat_id]
        return sid


async def _execute_shard(
    *,
    shard_index: int,
    shard_count: int,
    singles: list[SingleCaseView],
    journeys: list[JourneyView],
    identity: EvalIdentity,
    sender: TurnSender,
    recipient: str,
    out_dir: Path,
    dry_run: bool = False,
) -> ShardManifest:
    plans = plan_shards(
        [s.case_id for s in singles],
        [j.journey_id for j in journeys],
        shard_count=shard_count,
    )
    plan = plans[shard_index]
    wanted_singles = {s.case_id: s for s in singles if s.case_id in set(plan.single_ids)}
    wanted_journeys = {j.journey_id: j for j in journeys if j.journey_id in set(plan.journey_ids)}
    all_journey_ids = [j.journey_id for j in journeys]
    global_chat_ids = allocate_chat_ids(all_journey_ids)
    chat_ids = {jid: global_chat_ids[jid] for jid in wanted_journeys}
    # Fresh-session semantics: every single-turn case gets its own synthetic
    # chat, disjoint from the journey block, so independent singles never
    # share session state within a shard.
    single_chat_ids = allocate_chat_ids(list(wanted_singles), base=810000 + shard_index * 10000)
    captures: list[TurnCapture] = []
    policy = RetryPolicy()
    # Production execution must use real exponential backoff; no-sleep is
    # only for dry-run/offline tests to avoid delaying unit tests.
    sleep_fn = _no_sleep if dry_run else asyncio.sleep
    for case_id in sorted(wanted_singles):
        view = wanted_singles[case_id]
        payload = view.to_generator_payload()
        history: list[str] = []
        _ = bounded_history(history)
        from aa.qualification.conversation_eval import run_with_retry

        obs = await run_with_retry(
            sender,
            chat_id=single_chat_ids[case_id],
            utterance=payload["utterance"],
            primary_model=identity.primary_model,
            fallback_model=identity.fallback_model,
            policy=policy,
            sleep=sleep_fn,
        )
        captures.append(
            TurnCapture(
                case_id=case_id,
                journey_id="",
                turn=1,
                synthetic_input=payload["utterance"],
                generated_answer=obs.answer,
                safety_decision=obs.safety_decision,
                safety_categories=obs.safety_categories,
                planner_diagnostics=dict(obs.planner_diagnostics),
                retrieval_source_ids=obs.retrieval_source_ids,
                evidence_locators=obs.evidence_locators,
                evidence_checksums=obs.evidence_checksums,
                grounding_passed=obs.grounding_passed,
                regeneration_count=obs.regeneration_count,
                primary_model=identity.primary_model,
                actual_model=obs.actual_model or identity.primary_model,
                fallback_used=obs.fallback_used,
                latency_s=obs.latency_s,
                error_category=obs.error_category,
                retry_count=obs.retry_count,
            ).with_hashes()
        )
    for journey_id in sorted(wanted_journeys):
        view = wanted_journeys[journey_id]
        chat_id = chat_ids[journey_id]
        prior: list[str] = []
        for entry in view.turns:
            if entry.is_control():
                if entry.control != SESSION_RESET_CONTROL:
                    raise ValueError(f"{journey_id}: malformed control event")
                await sender.reset(chat_id)
                prior = []
                continue
            from aa.qualification.conversation_eval import run_with_retry

            visible = bounded_history(prior)
            del visible
            obs = await run_with_retry(
                sender,
                chat_id=chat_id,
                utterance=entry.utterance,
                primary_model=identity.primary_model,
                fallback_model=identity.fallback_model,
                policy=policy,
                sleep=sleep_fn,
            )
            captures.append(
                TurnCapture(
                    case_id=f"{journey_id}#{entry.turn}",
                    journey_id=journey_id,
                    turn=entry.turn,
                    synthetic_input=entry.utterance,
                    generated_answer=obs.answer,
                    safety_decision=obs.safety_decision,
                    safety_categories=obs.safety_categories,
                    planner_diagnostics=dict(obs.planner_diagnostics),
                    retrieval_source_ids=obs.retrieval_source_ids,
                    evidence_locators=obs.evidence_locators,
                    evidence_checksums=obs.evidence_checksums,
                    grounding_passed=obs.grounding_passed,
                    regeneration_count=obs.regeneration_count,
                    primary_model=identity.primary_model,
                    actual_model=obs.actual_model or identity.primary_model,
                    fallback_used=obs.fallback_used,
                    latency_s=obs.latency_s,
                    error_category=obs.error_category,
                    retry_count=obs.retry_count,
                ).with_hashes()
            )
            prior.append(entry.utterance)
    encrypted, bundle_sha = build_shard_bundle(captures, recipient=recipient)
    manifest = ShardManifest(
        shard_index=shard_index,
        shard_count=shard_count,
        case_ids=tuple(sorted([*wanted_singles, *wanted_journeys])),
        turn_rows=tuple(c.manifest_row() for c in captures),
        bundle_sha256=bundle_sha,
        eval_identity=identity.to_dict(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"shard-{shard_index}-manifest.json"
    bundle_path = out_dir / f"shard-{shard_index}.tar.zst.age"
    validate_files_do_not_mutate_main([str(manifest_path), str(bundle_path)])
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    bundle_path.write_bytes(encrypted)
    print(f"shard {shard_index}: {len(captures)} turns bundle_sha={bundle_sha[:16]}")
    return manifest


async def _no_sleep(_delay: float) -> None:
    return None


def main(argv: list[str] | None = None) -> int:
    """Run one deterministic shard (or only plan shards with ``--plan-only``)."""
    parser = argparse.ArgumentParser(description="Run one deterministic eval shard")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--out-dir", default="eval-out")
    parser.add_argument("--recipient-file", default="")
    parser.add_argument("--recipient", default="")
    parser.add_argument("--primary-model", default="opencode/muse-spark-1.3-contributor-free")
    parser.add_argument("--fallback-model", default="opencode/space-bunny-free")
    parser.add_argument("--main-sha", default="0" * 40)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    check_parallelism(shard_count=args.shard_count, max_parallel=args.max_parallel)
    singles, journeys = load_generator_views()
    print(f"loaded {len(singles)} singles + {len(journeys)} journeys")
    print(f"expected cases: {len(expected_case_ids(singles, journeys))}")
    if args.plan_only:
        for plan in plan_shards(
            [s.case_id for s in singles],
            [j.journey_id for j in journeys],
            shard_count=args.shard_count,
        ):
            n_single = len(plan.single_ids)
            n_journey = len(plan.journey_ids)
            print(f"shard {plan.shard_index}: singles={n_single} journeys={n_journey}")
        return 0
    recipient = args.recipient
    if args.recipient_file:
        recipient = Path(args.recipient_file).read_text(encoding="utf-8").strip()
    if not recipient.strip():
        print(
            "error: an age recipient is required (already-encrypted bundles only)", file=sys.stderr
        )
        return 2
    identity = collect_eval_identity(
        main_sha=args.main_sha,
        primary_model=args.primary_model,
        fallback_model=args.fallback_model,
    )
    out_dir = Path(args.out_dir)
    if args.dry_run:
        sender: TurnSender = DryRunSender(primary_model=args.primary_model)
    else:
        raise ConversationEvalError(
            "production TurnSender must be injected for non-dry-run execution; "
            "refusing to publish dry-run output as authoritative"
        )
    manifest = asyncio.run(
        _execute_shard(
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            singles=singles,
            journeys=journeys,
            identity=identity,
            sender=sender,
            recipient=recipient,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
    )
    digest = hashlib.sha256(json.dumps(manifest.to_dict(), sort_keys=True).encode()).hexdigest()
    print(f"manifest digest {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
