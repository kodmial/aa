"""Allowed Russian UI/protocol strings and deterministic result rendering.

Only short control/navigation text lives here. No AA recovery advice,
no medical claims, and no canned substantive replies are constructed.
Directory facts are navigation metadata and never book evidence.
"""

from __future__ import annotations

from typing import Any

OFFER_TEXT = "Если хочешь, могу помочь найти собрание. Это добровольно."
FORMAT_TEXT = "Выбери формат собрания."
CITY_PROMPT_TEXT = "В каком городе искать очное собрание? Напиши название — можно голосом."
REGION_CLARIFY_TEXT = "Уточни регион или выбери вариант ниже."
NO_COVERAGE_TEXT = "В этом месте нет подтверждённых данных. Вот официальный каталог."
DIRECTORY_ONLY_TEXT = "Это официальный каталог, а не найденное собрание."
STALE_TEXT = "Расписание может быть устаревшим. Проверь через официальный каталог."
UNKNOWN_DELIVERY_TEXT = "Не удалось подтвердить доставку. Попробуй ещё раз."
UNCONFIRMED_NOTE = "Published schedule, unconfirmed: verify with organizers."

BTN_OFFER_ACCEPT = "Да, подобрать"
BTN_DECLINE = "Не сейчас"
BTN_ONLINE = "Онлайн"
BTN_IN_PERSON = "Очно"
BTN_BACK = "Назад"
BTN_MORE = "Показать ещё"
BTN_OTHER_CITY = "Другой город"
BTN_NEW_SEARCH = "Новый поиск"
BTN_YES_CITY = "Да"
BTN_CANCEL = "Не сейчас"

ROOT_CATALOG_URL = "https://aarussia.ru/aagroups/"
ONLINE_CATALOG_URL = "https://aarussia.ru/online-groups/"


def inline_keyboard(rows: list[list[dict[str, str]]]) -> dict[str, Any]:
    """Build one Telegram inline keyboard structure."""
    return {"inline_keyboard": [[dict(cell) for cell in row] for row in rows]}


def offer_keyboard(accept_token: str, decline_token: str) -> dict[str, Any]:
    """Two-button initial invitation keyboard."""
    return inline_keyboard(
        [
            [
                {"text": BTN_OFFER_ACCEPT, "callback_data": accept_token},
                {"text": BTN_DECLINE, "callback_data": decline_token},
            ]
        ]
    )


def format_keyboard(online_token: str, in_person_token: str, cancel_token: str) -> dict[str, Any]:
    """Format selection keyboard with an unobtrusive exit."""
    return inline_keyboard(
        [
            [
                {"text": BTN_ONLINE, "callback_data": online_token},
                {"text": BTN_IN_PERSON, "callback_data": in_person_token},
            ],
            [{"text": BTN_DECLINE, "callback_data": cancel_token}],
        ]
    )


def clarify_keyboard(
    candidates: list[dict[str, str]],
    choose_tokens: list[str],
    other_city_token: str,
    cancel_token: str,
) -> dict[str, Any]:
    """Bounded disambiguation keyboard for at most four verified choices."""
    rows: list[list[dict[str, str]]] = []
    for candidate, token in zip(candidates[:4], choose_tokens[:4], strict=False):
        label = str(candidate.get("display_name", ""))
        region = str(candidate.get("region_hint", "") or "")
        text = f"{label}, {region}" if region else label
        rows.append([{"text": text[:60] or "Выбрать", "callback_data": token}])
    rows.append([{"text": BTN_OTHER_CITY, "callback_data": other_city_token}])
    rows.append([{"text": BTN_DECLINE, "callback_data": cancel_token}])
    return inline_keyboard(rows)


def results_keyboard(
    more_token: str | None,
    other_city_token: str,
    online_token: str,
    cancel_token: str,
) -> dict[str, Any]:
    """Result navigation keyboard with a bounded forward-only pager."""
    rows: list[list[dict[str, str]]] = []
    if more_token is not None:
        rows.append([{"text": BTN_MORE, "callback_data": more_token}])
    rows.append(
        [
            {"text": BTN_OTHER_CITY, "callback_data": other_city_token},
            {"text": BTN_ONLINE, "callback_data": online_token},
        ]
    )
    rows.append([{"text": BTN_DECLINE, "callback_data": cancel_token}])
    return inline_keyboard(rows)


def back_cancel_keyboard(back_token: str, cancel_token: str) -> dict[str, Any]:
    """Minimal navigation keyboard for intermediate prompts."""
    return inline_keyboard(
        [
            [
                {"text": BTN_BACK, "callback_data": back_token},
                {"text": BTN_DECLINE, "callback_data": cancel_token},
            ]
        ]
    )


_RU_WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


def _local_headline(item: Any, start_local: str, day: str) -> str:
    """Build a Russian meeting-local headline without locale dependence."""
    prefix = {"today": "Сегодня", "tomorrow": "Завтра"}.get(day, "")
    weekday_ru = ""
    try:
        start_utc = getattr(item, "start_utc", None)
        zone_name = str(getattr(item, "timezone", "") or "")
        if start_utc is not None and zone_name and zone_name != "unknown":
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo as _Zone

            moment = start_utc
            if isinstance(moment, _dt):
                local = moment.astimezone(_Zone(zone_name))
                weekday_ru = _RU_WEEKDAYS[local.weekday()]
                stamp = local.strftime("%Y-%m-%d %H:%M")
                head = f"{weekday_ru} {stamp} ({zone_name})"
                return f"{prefix} {head}".strip() if prefix else head
    except Exception:
        weekday_ru = ""
    head = start_local.strip()
    if prefix:
        return f"{prefix} {head}".strip()
    return head


def format_occurrence(item: Any) -> str:
    """Render one verified occurrence as deterministic navigation text."""
    group = str(getattr(item, "group_name", "") or "")
    start_local = str(getattr(item, "start_local", "") or "")
    venue = str(getattr(item, "venue_or_url", "") or "")
    source = str(getattr(item, "source_url", "") or "")
    day = str(getattr(item, "day_label", "") or "")
    head = _local_headline(item, start_local, day)
    lines = [group.strip(), head.strip(), venue.strip(), source.strip(), UNCONFIRMED_NOTE]
    return "\n".join(line for line in lines if line)


def format_results_text(occurrences: list[Any], directory_link: str | None = None) -> str:
    """Render up to three occurrences plus an honest directory caveat."""
    parts = [format_occurrence(item) for item in occurrences[:3] if format_occurrence(item)]
    text = "\n\n".join(part for part in parts if part)
    if directory_link:
        suffix = f"{DIRECTORY_ONLY_TEXT}\n{directory_link}"
        text = f"{text}\n\n{suffix}" if text else suffix
    return text


__all__ = [
    "BTN_BACK",
    "BTN_DECLINE",
    "BTN_IN_PERSON",
    "BTN_MORE",
    "BTN_NEW_SEARCH",
    "BTN_OFFER_ACCEPT",
    "BTN_ONLINE",
    "BTN_OTHER_CITY",
    "BTN_YES_CITY",
    "BTN_CANCEL",
    "CITY_PROMPT_TEXT",
    "DIRECTORY_ONLY_TEXT",
    "FORMAT_TEXT",
    "NO_COVERAGE_TEXT",
    "OFFER_TEXT",
    "ONLINE_CATALOG_URL",
    "REGION_CLARIFY_TEXT",
    "ROOT_CATALOG_URL",
    "STALE_TEXT",
    "UNCONFIRMED_NOTE",
    "UNKNOWN_DELIVERY_TEXT",
    "back_cancel_keyboard",
    "clarify_keyboard",
    "format_keyboard",
    "format_occurrence",
    "format_results_text",
    "inline_keyboard",
    "offer_keyboard",
    "results_keyboard",
]
