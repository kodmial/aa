"""Deterministic per-chat meeting invitation service (zero LLM calls).

One live active offer per chat. All directory access uses the offline
#316/#317 API. All math, filtering, and paging is ordinary code.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from aa.meeting_directory.models import DirectorySnapshot
from aa.meeting_invitation.callbacks import CallbackCodec, StaleCallbackError
from aa.meeting_invitation.fsm import (
    MAX_CITY_ATTEMPTS,
    MAX_CLARIFY_CHOICES,
    RESULTS_PAGE_SIZE,
    CallbackAction,
    MeetingFlow,
    MeetingState,
    ResultsSnapshot,
    is_expired,
    is_terminal,
    new_flow,
)
from aa.meeting_invitation.rendering import (
    CITY_PROMPT_TEXT,
    DIRECTORY_ONLY_TEXT,
    FORMAT_TEXT,
    NO_COVERAGE_TEXT,
    OFFER_TEXT,
    ONLINE_CATALOG_URL,
    REGION_CLARIFY_TEXT,
    ROOT_CATALOG_URL,
    STALE_TEXT,
    back_cancel_keyboard,
    clarify_keyboard,
    format_keyboard,
    format_results_text,
    offer_keyboard,
    results_keyboard,
)

logger = logging.getLogger("aa.meeting_invitation")


def _hash_chat(chat_id: int) -> str:
    return hashlib.sha256(str(int(chat_id)).encode()).hexdigest()[:16]


ReceiptStatus = Literal["confirmed", "failed", "unknown"]


@dataclass
class OfferResult:
    """A pending offer awaiting delivery confirmation."""

    flow_id: str
    text: str
    keyboard: dict[str, Any]
    pending: bool = True


@dataclass
class CallbackOutcome:
    """Exactly one UI operation resulting from one accepted callback."""

    accepted: bool
    ack_text: str = ""
    send_text: str | None = None
    send_keyboard: dict[str, Any] | None = None
    edit_keyboard: dict[str, Any] | None = None
    remove_keyboard: bool = False
    exit_to_dialogue: bool = False
    state: str = "IDLE"
    stale: bool = False
    duplicate: bool = False


@dataclass
class TextOutcome:
    """Exactly one UI operation resulting from one city-pending text turn."""

    handled: bool
    send_text: str | None = None
    send_keyboard: dict[str, Any] | None = None
    edit_keyboard: dict[str, Any] | None = None
    exit_to_dialogue: bool = False
    state: str = "IDLE"


SnapshotProvider = Callable[[], DirectorySnapshot | None]


class MeetingService:
    """Per-chat invitation flows with delivery-aware commits."""

    def __init__(
        self,
        *,
        secret: bytes | None = None,
        snapshot_provider: SnapshotProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._codec = CallbackCodec(secret)
        self._flows: dict[int, MeetingFlow] = {}
        self._generations: dict[int, int] = {}
        self._seq: dict[int, int] = {}
        self._snapshot_provider = snapshot_provider
        self._clock = clock

    def _now_utc(self) -> datetime:
        if self._clock is not None:
            moment = self._clock()
            if moment.tzinfo is None:
                raise ValueError("service clock must be timezone-aware")
            return moment.astimezone(UTC)
        return datetime.now(UTC)

    def _now_epoch(self) -> float:
        return self._now_utc().timestamp()

    def _snapshot(self) -> DirectorySnapshot | None:
        if self._snapshot_provider is not None:
            try:
                return self._snapshot_provider()
            except Exception:
                return None
        try:
            from aa.meeting_directory.loader import get_cached_snapshot

            return get_cached_snapshot()
        except Exception:
            return None

    def generation_for(self, chat_id: int) -> int:
        """Return the current session generation for one chat."""
        return int(self._generations.get(int(chat_id), 0))

    def flow_for(self, chat_id: int) -> MeetingFlow | None:
        """Return the current flow for one chat, if any."""
        return self._flows.get(int(chat_id))

    def handle_new(self, chat_id: int) -> None:
        """Invalidate every keyboard via a new generation (``/new``)."""
        key = int(chat_id)
        self._generations[key] = int(self._generations.get(key, 0)) + 1
        self._flows.pop(key, None)

    def _next_seq(self, chat_id: int) -> int:
        key = int(chat_id)
        value = int(self._seq.get(key, 0)) + 1
        self._seq[key] = value
        return value

    def _token(
        self,
        *,
        action: str,
        flow: MeetingFlow,
        message_id: int,
        place_index: int = 0,
    ) -> str:
        return self._codec.encode(
            action=action,
            flow_seq=int(flow.flow_seq),
            generation=int(flow.generation),
            message_id=int(message_id),
            place_index=int(place_index),
            now=self._now_epoch(),
        )

    def begin_offer(self, chat_id: int) -> MeetingFlow:
        """Start a pending offer; delivery confirmation activates it."""
        key = int(chat_id)
        generation = int(self._generations.get(key, 0))
        flow = new_flow(_hash_chat(key), generation, self._next_seq(key), self._now_epoch())
        flow.offer_id = uuid.uuid4().hex[:16]
        flow.pending_delivery = True
        flow.state = MeetingState.OFFERED
        flow.updated_at = self._now_epoch()
        self._flows[key] = flow
        return flow

    def create_offer(self, chat_id: int, bound_message_id: int) -> OfferResult:
        """Create an offer bound to an already-sent message id (tests)."""
        self.begin_offer(chat_id)
        return self.activate_offer(chat_id, int(bound_message_id))

    def activate_offer(self, chat_id: int, bound_message_id: int) -> OfferResult:
        """Bind a pending offer to a real Telegram message and issue keys."""
        key = int(chat_id)
        flow = self._flows.get(key)
        if flow is None:
            raise ValueError("no pending offer for this chat")
        flow.bound_message_id = int(bound_message_id)
        flow.pending_delivery = True
        flow.updated_at = self._now_epoch()
        keyboard = offer_keyboard(
            self._token(
                action=CallbackAction.ACCEPT.value, flow=flow, message_id=int(bound_message_id)
            ),
            self._token(
                action=CallbackAction.DECLINE.value, flow=flow, message_id=int(bound_message_id)
            ),
        )
        return OfferResult(flow_id=flow.flow_id, text=OFFER_TEXT, keyboard=keyboard, pending=True)

    def confirm_offer_delivery(
        self, chat_id: int, bound_message_id: int, status: ReceiptStatus
    ) -> MeetingState:
        """Commit invitation state only on confirmed keyboard delivery."""
        key = int(chat_id)
        flow = self._flows.get(key)
        if flow is None:
            return MeetingState.IDLE
        if int(flow.bound_message_id or -1) != int(bound_message_id):
            return flow.state
        if status == "confirmed" or status == "unknown":
            flow.pending_delivery = status == "unknown"
            flow.state = MeetingState.OFFERED
            flow.updated_at = self._now_epoch()
            return flow.state
        # Failed send leaves no phantom invitation.
        fresh = new_flow(
            _hash_chat(key), int(flow.generation), int(flow.flow_seq), self._now_epoch()
        )
        self._flows[key] = fresh
        return fresh.state

    def _lazy_expiry(self, key: int, flow: MeetingFlow) -> CallbackOutcome | None:
        if is_expired(flow, self._now_epoch()):
            flow.state = MeetingState.EXPIRED
            flow.updated_at = self._now_epoch()
            return CallbackOutcome(
                accepted=False,
                ack_text="",
                remove_keyboard=True,
                state=flow.state.value,
                stale=True,
            )
        return None

    def _validate_callback(
        self, key: int, sender_id: int, message_id: int | None, token: str
    ) -> tuple[MeetingFlow, Any]:
        if int(sender_id) != int(key):
            raise StaleCallbackError("foreign callback sender")
        data = self._codec.decode(str(token), now=self._now_epoch())
        flow = self._flows.get(key)
        if flow is None:
            raise StaleCallbackError("no active flow")
        if int(data.generation) != int(flow.generation):
            raise StaleCallbackError("stale session generation")
        if int(data.flow_seq) != int(flow.flow_seq):
            raise StaleCallbackError("superseded flow")
        # Message binding: the token must belong to the message carrying
        # the pressed button. Multiple wizard messages may be live within
        # one flow, so the token is not required to equal the latest
        # bound message; cross-flow reuse is already blocked by the
        # flow sequence and generation above.
        if message_id is not None and int(message_id) > 0:
            if int(data.message_id) != int(message_id):
                raise StaleCallbackError("stale message binding")
        if getattr(data, "place_index", 0) < 0 or getattr(data, "place_index", 0) > 63:
            raise StaleCallbackError("callback token action is unknown")
        if is_terminal(flow.state):
            raise StaleCallbackError("terminal flow")
        expired = self._lazy_expiry(key, flow)
        if expired is not None:
            raise StaleCallbackError("offer expired")
        if str(token) in flow.consumed_actions:
            raise StaleCallbackError("duplicate callback")
        return flow, data

    def note_sent_message(self, chat_id: int, message_id: int) -> None:
        """Record one wizard message id for delivery-bound transitions."""
        flow = self._flows.get(int(chat_id))
        if flow is None:
            return
        flow.bound_message_id = int(message_id)
        flow.pending_delivery = False
        flow.updated_at = self._now_epoch()

    def handle_callback(
        self,
        chat_id: int,
        sender_id: int,
        message_id: int | None,
        token: str,
    ) -> CallbackOutcome:
        """Process exactly one callback token into at most one transition."""
        key = int(chat_id)
        try:
            flow, data = self._validate_callback(key, int(sender_id), message_id, str(token))
        except StaleCallbackError:
            existing = self._flows.get(key)
            state = existing.state.value if existing is not None else MeetingState.IDLE.value
            return CallbackOutcome(accepted=False, ack_text="", state=state, stale=True)
        action = str(data.action)
        if str(token) in flow.consumed_actions:
            return CallbackOutcome(
                accepted=False, ack_text="", state=flow.state.value, stale=True, duplicate=True
            )
        outcome = self._apply_action(key, flow, action, int(data.place_index))
        flow.consumed_actions = (*flow.consumed_actions, str(token))[-64:]
        flow.updated_at = self._now_epoch()
        outcome.state = flow.state.value
        return outcome

    def _apply_action(
        self, key: int, flow: MeetingFlow, action: str, place_index: int
    ) -> CallbackOutcome:
        state = flow.state
        if action in (CallbackAction.CANCEL.value,):
            return self._enter_cancelled(flow)
        if state == MeetingState.OFFERED:
            if action == CallbackAction.ACCEPT.value:
                return self._enter_format(flow)
            if action == CallbackAction.DECLINE.value:
                flow.state = MeetingState.DECLINED
                return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        if state == MeetingState.CHOOSE_FORMAT:
            if action == CallbackAction.ONLINE.value:
                return self._show_online(flow)
            if action == CallbackAction.IN_PERSON.value:
                return self._ask_city(flow)
            if action == CallbackAction.BACK.value:
                flow.state = MeetingState.OFFERED
                if flow.bound_message_id is not None:
                    keyboard = offer_keyboard(
                        self._token(
                            action=CallbackAction.ACCEPT.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                        self._token(
                            action=CallbackAction.DECLINE.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                    )
                    return CallbackOutcome(accepted=True, ack_text="", edit_keyboard=keyboard)
                return CallbackOutcome(accepted=True, ack_text="")
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        if state == MeetingState.AWAIT_CITY:
            if action == CallbackAction.BACK.value:
                return self._enter_format(flow, preserve_city=True)
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        if state == MeetingState.CLARIFY_CITY:
            if action == CallbackAction.BACK.value:
                flow.state = MeetingState.AWAIT_CITY
                if flow.bound_message_id is not None:
                    keyboard = back_cancel_keyboard(
                        self._token(
                            action=CallbackAction.BACK.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                        self._token(
                            action=CallbackAction.CANCEL.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                    )
                    return CallbackOutcome(
                        accepted=True,
                        ack_text="",
                        send_text=CITY_PROMPT_TEXT,
                        send_keyboard=keyboard,
                    )
                return CallbackOutcome(accepted=True, ack_text="", send_text=CITY_PROMPT_TEXT)
            if action == CallbackAction.OTHER_CITY.value:
                flow.state = MeetingState.AWAIT_CITY
                flow.city_attempts = 0
                flow.region_clarifications = 0
                if flow.bound_message_id is not None:
                    keyboard = back_cancel_keyboard(
                        self._token(
                            action=CallbackAction.BACK.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                        self._token(
                            action=CallbackAction.CANCEL.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                    )
                    return CallbackOutcome(
                        accepted=True,
                        ack_text="",
                        send_text=CITY_PROMPT_TEXT,
                        send_keyboard=keyboard,
                    )
                return CallbackOutcome(accepted=True, ack_text="", send_text=CITY_PROMPT_TEXT)
            if action == CallbackAction.CHOOSE_PLACE.value:
                return self._choose_place(flow, int(place_index))
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        if state == MeetingState.SHOW_RESULTS:
            if action == CallbackAction.MORE.value:
                return self._page_more(flow)
            if action == CallbackAction.BACK.value:
                flow.state = MeetingState.CHOOSE_FORMAT
                flow.results = ResultsSnapshot()
                return self._enter_format(flow, preserve_city=True)
            if action == CallbackAction.ONLINE.value:
                return self._show_online(flow)
            if action == CallbackAction.OTHER_CITY.value:
                flow.state = MeetingState.AWAIT_CITY
                flow.city_attempts = 0
                flow.region_clarifications = 0
                if flow.bound_message_id is not None:
                    keyboard = back_cancel_keyboard(
                        self._token(
                            action=CallbackAction.BACK.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                        self._token(
                            action=CallbackAction.CANCEL.value,
                            flow=flow,
                            message_id=int(flow.bound_message_id),
                        ),
                    )
                    return CallbackOutcome(
                        accepted=True,
                        ack_text="",
                        send_text=CITY_PROMPT_TEXT,
                        send_keyboard=keyboard,
                    )
                return CallbackOutcome(accepted=True, ack_text="", send_text=CITY_PROMPT_TEXT)
            if action == CallbackAction.NEW_SEARCH.value:
                flow.state = MeetingState.CHOOSE_FORMAT
                flow.results = ResultsSnapshot()
                return self._enter_format(flow, preserve_city=False)
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        return CallbackOutcome(accepted=False, ack_text="", stale=True)

    def _enter_cancelled(self, flow: MeetingFlow) -> CallbackOutcome:
        if flow.state == MeetingState.OFFERED:
            flow.state = MeetingState.DECLINED
        else:
            flow.state = MeetingState.CANCELLED
        return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)

    def _enter_format(self, flow: MeetingFlow, preserve_city: bool = False) -> CallbackOutcome:
        flow.state = MeetingState.CHOOSE_FORMAT
        if not preserve_city:
            flow.meeting_format = "unknown"
        if flow.bound_message_id is None:
            return CallbackOutcome(accepted=True, ack_text="", send_text=FORMAT_TEXT)
        keyboard = format_keyboard(
            self._token(
                action=CallbackAction.ONLINE.value, flow=flow, message_id=int(flow.bound_message_id)
            ),
            self._token(
                action=CallbackAction.IN_PERSON.value,
                flow=flow,
                message_id=int(flow.bound_message_id),
            ),
            self._token(
                action=CallbackAction.CANCEL.value, flow=flow, message_id=int(flow.bound_message_id)
            ),
        )
        return CallbackOutcome(accepted=True, ack_text="", edit_keyboard=keyboard)

    def _ask_city(self, flow: MeetingFlow) -> CallbackOutcome:
        if flow.place_id:
            return self._show_in_person(flow, str(flow.place_id))
        flow.state = MeetingState.AWAIT_CITY
        flow.city_attempts = 0
        flow.region_clarifications = 0
        if flow.bound_message_id is None:
            return CallbackOutcome(accepted=True, ack_text="", send_text=CITY_PROMPT_TEXT)
        keyboard = back_cancel_keyboard(
            self._token(
                action=CallbackAction.BACK.value, flow=flow, message_id=int(flow.bound_message_id)
            ),
            self._token(
                action=CallbackAction.CANCEL.value, flow=flow, message_id=int(flow.bound_message_id)
            ),
        )
        return CallbackOutcome(
            accepted=True, ack_text="", send_text=CITY_PROMPT_TEXT, send_keyboard=keyboard
        )

    def _directory(self) -> DirectorySnapshot | None:
        return self._snapshot()

    def _show_online(self, flow: MeetingFlow) -> CallbackOutcome:
        flow.meeting_format = "online"
        snapshot = self._directory()
        if snapshot is None:
            flow.state = MeetingState.DONE
            return CallbackOutcome(
                accepted=True,
                ack_text="",
                send_text=f"{NO_COVERAGE_TEXT}\n{ONLINE_CATALOG_URL}",
            )
        try:
            from aa.meeting_directory.query import get_upcoming_meetings
        except Exception:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        now = self._now_utc()
        try:
            result = get_upcoming_meetings(
                format="online", now_utc=now, limit=RESULTS_PAGE_SIZE, snapshot=snapshot
            )
        except Exception:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        return self._render_upcoming(
            flow, result, place_id=None, region_id=None, meeting_format="online", now=now
        )

    def _show_in_person(self, flow: MeetingFlow, place_id: str) -> CallbackOutcome:
        flow.meeting_format = "in_person"
        flow.place_id = place_id
        snapshot = self._directory()
        if snapshot is None:
            flow.state = MeetingState.DONE
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        try:
            from aa.meeting_directory.query import get_upcoming_meetings
        except Exception:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        now = self._now_utc()
        try:
            result = get_upcoming_meetings(
                place_id=place_id,
                format="in_person",
                now_utc=now,
                limit=RESULTS_PAGE_SIZE,
                snapshot=snapshot,
            )
        except Exception:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        return self._render_upcoming(
            flow, result, place_id=place_id, region_id=None, meeting_format="in_person", now=now
        )

    def _render_upcoming(
        self,
        flow: MeetingFlow,
        result: Any,
        *,
        place_id: str | None,
        region_id: str | None,
        meeting_format: str,
        now: datetime,
    ) -> CallbackOutcome:
        occurrences = list(getattr(result, "occurrences", ()) or ())
        fallback = getattr(result, "fallback", None)
        cursor = getattr(result, "next_cursor", None)
        version = str(getattr(result, "directory_version", "") or "")
        digest = ""
        snapshot = self._directory()
        if snapshot is not None:
            digest = str(getattr(snapshot, "digest", "") or "")
        if occurrences:
            flow.state = MeetingState.SHOW_RESULTS
            keys = tuple(
                f"{str(getattr(item, 'slot_id', ''))}|{str(getattr(item, 'start_utc', ''))}"
                for item in occurrences
            )
            flow.results = ResultsSnapshot(
                place_id=place_id,
                region_id=region_id,
                meeting_format=meeting_format,
                now_iso=now.isoformat(),
                cursor=cursor if isinstance(cursor, str) else None,
                directory_version=version,
                directory_digest=digest,
                occurrence_keys=keys,
                exhausted=not bool(cursor),
                fallback=None,
                directory_link=None,
            )
            text = format_results_text(list(occurrences))
            keyboard = self._results_keyboard(flow)
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=text, send_keyboard=keyboard
            )
        link = self._directory_link_for(result, place_id, meeting_format)
        note = DIRECTORY_ONLY_TEXT
        if fallback in ("schedule_stale", "stale_details"):
            note = STALE_TEXT
        elif fallback in ("no_coverage", "no_upcoming_in_window", None):
            note = NO_COVERAGE_TEXT
        flow.state = MeetingState.DONE
        flow.results = ResultsSnapshot(
            place_id=place_id,
            region_id=region_id,
            meeting_format=meeting_format,
            now_iso=now.isoformat(),
            cursor=None,
            directory_version=version,
            directory_digest=digest,
            occurrence_keys=(),
            exhausted=True,
            fallback=str(fallback) if fallback else "directory_only",
            directory_link=link,
        )
        text = f"{note}\n{link}" if link else note
        return CallbackOutcome(accepted=True, ack_text="", send_text=text)

    def _directory_link_for(self, result: Any, place_id: str | None, meeting_format: str) -> str:
        for attr in ("display_note",):
            _ = attr
        snapshot = self._directory()
        if snapshot is not None:
            try:
                from aa.meeting_directory.query import search as _search

                if place_id:
                    found = _search(place_id=place_id, snapshot=snapshot)
                    if found.region_link:
                        return str(found.region_link)
                    if found.source_urls:
                        return str(found.source_urls[0])
            except Exception:
                pass
        if meeting_format == "online":
            return ONLINE_CATALOG_URL
        return ROOT_CATALOG_URL

    def _results_keyboard(self, flow: MeetingFlow) -> dict[str, Any] | None:
        if flow.bound_message_id is None:
            bound = 0
        else:
            bound = int(flow.bound_message_id)
        more = None
        if flow.results.cursor and not flow.results.exhausted:
            more = self._token(action=CallbackAction.MORE.value, flow=flow, message_id=bound)
        other = self._token(action=CallbackAction.OTHER_CITY.value, flow=flow, message_id=bound)
        online = self._token(action=CallbackAction.ONLINE.value, flow=flow, message_id=bound)
        cancel = self._token(action=CallbackAction.CANCEL.value, flow=flow, message_id=bound)
        return results_keyboard(more, other, online, cancel)

    def _page_more(self, flow: MeetingFlow) -> CallbackOutcome:
        cursor = flow.results.cursor
        if not cursor or flow.results.exhausted:
            flow.state = MeetingState.DONE
            flow.results.exhausted = True
            flow.results.cursor = None
            return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)
        snapshot = self._directory()
        if snapshot is None:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        try:
            from aa.meeting_directory.query import get_upcoming_meetings
        except Exception:
            flow.state = MeetingState.FAILED
            return CallbackOutcome(
                accepted=True, ack_text="", send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}"
            )
        now_iso = flow.results.now_iso
        try:
            now = datetime.fromisoformat(now_iso) if now_iso else self._now_utc()
            if now.tzinfo is None:
                now = self._now_utc()
        except (ValueError, TypeError):
            now = self._now_utc()
        fmt: Any = (
            flow.results.meeting_format
            if flow.results.meeting_format in ("online", "in_person")
            else None
        )
        try:
            result = get_upcoming_meetings(
                place_id=flow.results.place_id,
                region_id=flow.results.region_id,
                format=fmt,
                now_utc=now,
                limit=RESULTS_PAGE_SIZE,
                cursor=cursor,
                snapshot=snapshot,
            )
        except Exception:
            flow.state = MeetingState.DONE
            flow.results.exhausted = True
            flow.results.cursor = None
            return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)
        occurrences = list(getattr(result, "occurrences", ()) or ())
        if not occurrences:
            flow.state = MeetingState.DONE
            flow.results.exhausted = True
            flow.results.cursor = None
            return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)
        new_keys = tuple(
            f"{str(getattr(item, 'slot_id', ''))}|{str(getattr(item, 'start_utc', ''))}"
            for item in occurrences
        )
        if set(new_keys) <= set(flow.results.occurrence_keys):
            flow.state = MeetingState.DONE
            flow.results.exhausted = True
            flow.results.cursor = None
            return CallbackOutcome(accepted=True, ack_text="", remove_keyboard=True)
        flow.results.occurrence_keys = (*flow.results.occurrence_keys, *new_keys)
        next_cursor = getattr(result, "next_cursor", None)
        flow.results.cursor = str(next_cursor) if isinstance(next_cursor, str) else None
        flow.results.exhausted = not bool(flow.results.cursor)
        text = format_results_text(list(occurrences))
        keyboard = self._results_keyboard(flow)
        return CallbackOutcome(accepted=True, ack_text="", send_text=text, send_keyboard=keyboard)

    def _choose_place(self, flow: MeetingFlow, place_index: int) -> CallbackOutcome:
        candidates = list(flow.candidates or [])
        if not candidates or place_index < 0 or place_index >= len(candidates):
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        place_id = str(candidates[int(place_index)].get("place_id", "") or "")
        if not place_id:
            return CallbackOutcome(accepted=False, ack_text="", stale=True)
        flow.candidates = ()
        return self._show_in_person(flow, place_id)

    def handle_text(self, chat_id: int, text: str) -> TextOutcome:
        """Route one inbound text turn while a city question may be pending."""
        key = int(chat_id)
        flow = self._flows.get(key)
        if flow is None:
            return TextOutcome(handled=False, exit_to_dialogue=True, state="IDLE")
        expired = is_expired(flow, self._now_epoch())
        if expired:
            flow.state = MeetingState.EXPIRED
            return TextOutcome(handled=False, exit_to_dialogue=True, state=flow.state.value)
        if flow.state not in (MeetingState.AWAIT_CITY, MeetingState.CLARIFY_CITY):
            return TextOutcome(handled=False, exit_to_dialogue=True, state=flow.state.value)
        cleaned = str(text or "").strip()
        if not cleaned:
            return TextOutcome(handled=False, exit_to_dialogue=True, state=flow.state.value)
        if self._looks_like_dialogue(cleaned):
            snapshot = self._try_resolve(cleaned, None)
            if snapshot is None or str(getattr(snapshot, "status", "unknown")) != "exact_unique":
                flow.state = MeetingState.IDLE
                flow.updated_at = self._now_epoch()
                return TextOutcome(handled=False, exit_to_dialogue=True, state="IDLE")
        return self._resolve_city_text(flow, cleaned)

    def _looks_like_dialogue(self, cleaned: str) -> bool:
        if "?" in cleaned:
            return True
        words = cleaned.split()
        if len(words) > 12 or len(cleaned) > 140:
            return True
        sentences = [
            part for part in cleaned.replace("!", ".").replace("?", ".").split(".") if part.strip()
        ]
        if len(sentences) > 1 and len(words) > 6:
            return True
        return False

    def _try_resolve(self, query: str, region_hint: str | None) -> Any | None:
        snapshot = self._directory()
        if snapshot is None:
            return None
        try:
            from aa.meeting_directory.query import resolve_locality as _resolve

            return _resolve(str(query), region_hint, snapshot)
        except Exception:
            return None

    def _resolve_city_text(self, flow: MeetingFlow, cleaned: str) -> TextOutcome:
        resolution = self._try_resolve(cleaned, None)
        if resolution is None:
            flow.state = MeetingState.DONE
            return TextOutcome(
                handled=True,
                send_text=f"{NO_COVERAGE_TEXT}\n{ROOT_CATALOG_URL}",
                state=flow.state.value,
            )
        status = str(getattr(resolution, "status", "unknown"))
        if status == "exact_unique":
            place_id = str(getattr(resolution, "place_id", "") or "")
            if not place_id:
                return self._register_city_failure(flow)
            outcome = self._show_in_person(flow, place_id)
            return TextOutcome(
                handled=True,
                send_text=outcome.send_text,
                send_keyboard=outcome.send_keyboard,
                edit_keyboard=outcome.edit_keyboard,
                state=flow.state.value,
            )
        if status in ("ambiguous", "needs_region"):
            candidates = getattr(resolution, "candidates", ()) or ()
            limited = [
                {
                    "place_id": str(getattr(item, "place_id", "")),
                    "display_name": str(getattr(item, "display_name", "")),
                    "region_hint": str(getattr(item, "region_hint", "") or ""),
                }
                for item in list(candidates)[:MAX_CLARIFY_CHOICES]
            ]
            flow.state = MeetingState.CLARIFY_CITY
            flow.candidates = tuple(limited)
            flow.updated_at = self._now_epoch()
            bound = int(flow.bound_message_id) if flow.bound_message_id is not None else 0
            tokens = [
                self._token(
                    action=CallbackAction.CHOOSE_PLACE.value,
                    flow=flow,
                    message_id=bound,
                    place_index=index,
                )
                for index in range(len(limited))
            ]
            other = self._token(action=CallbackAction.OTHER_CITY.value, flow=flow, message_id=bound)
            cancel = self._token(action=CallbackAction.CANCEL.value, flow=flow, message_id=bound)
            keyboard = clarify_keyboard(limited, tokens, other, cancel)
            prompt = REGION_CLARIFY_TEXT if status == "needs_region" else CITY_PROMPT_TEXT
            return TextOutcome(
                handled=True, send_text=prompt, send_keyboard=keyboard, state=flow.state.value
            )
        return self._register_city_failure(flow)

    def _register_city_failure(self, flow: MeetingFlow) -> TextOutcome:
        flow.city_attempts = int(flow.city_attempts) + 1
        flow.updated_at = self._now_epoch()
        if int(flow.city_attempts) < int(MAX_CITY_ATTEMPTS):
            bound = int(flow.bound_message_id) if flow.bound_message_id is not None else 0
            keyboard = back_cancel_keyboard(
                self._token(action=CallbackAction.BACK.value, flow=flow, message_id=bound),
                self._token(action=CallbackAction.CANCEL.value, flow=flow, message_id=bound),
            )
            return TextOutcome(
                handled=True,
                send_text=REGION_CLARIFY_TEXT,
                send_keyboard=keyboard,
                state=flow.state.value,
            )
        flow.state = MeetingState.DONE
        link = self._region_link_for_text() or ROOT_CATALOG_URL
        return TextOutcome(
            handled=True, send_text=f"{DIRECTORY_ONLY_TEXT}\n{link}", state=flow.state.value
        )

    def _region_link_for_text(self) -> str | None:
        snapshot = self._directory()
        if snapshot is None:
            return None
        try:
            links = list(getattr(snapshot, "region_links", ()) or ())
            if links:
                url = str(getattr(links[0], "url", "") or "")
                return url or None
        except Exception:
            return None
        return None

    def check_expiry(self, chat_id: int) -> MeetingState | None:
        """Lazily expire one idle wizard; returns the new state if changed."""
        flow = self._flows.get(int(chat_id))
        if flow is None:
            return None
        if is_expired(flow, self._now_epoch()):
            flow.state = MeetingState.EXPIRED
            flow.updated_at = self._now_epoch()
            return flow.state
        return None


__all__ = [
    "CallbackOutcome",
    "MeetingService",
    "OfferResult",
    "ReceiptStatus",
    "TextOutcome",
]
