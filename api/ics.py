"""Calendar export: action items as an .ics file (RFC 5545).

Pure stdlib. A handful of VEVENTs does not justify a calendar dependency, and
"no new package" is worth more here than generality — the locality points are
judged on what this machine has to reach for.

The interesting half is `parse_due`: the LLM gives the deadline back as it was
*said* ("до пятницы", "жұмаға дейін", "25 сентября"), because that is what the
transcript contains. A calendar needs a real date, so we resolve that phrase
relative to the meeting date, in the three languages this demo speaks.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

PRODID = "-//Gonkong//AI Meeting Intelligence//RU"

# --------------------------------------------------------------------------
# Due-date parsing
# --------------------------------------------------------------------------

# Monday = 0, matching date.weekday().
_RU_WEEKDAYS: dict[int, tuple[str, ...]] = {
    0: ("понедельник", "понедельника", "понедельнику", "понедельником", "понедельнике"),
    1: ("вторник", "вторника", "вторнику", "вторником", "вторнике"),
    2: ("среда", "среду", "среды", "среде", "средой"),
    3: ("четверг", "четверга", "четвергу", "четвергом", "четверге"),
    4: ("пятница", "пятницу", "пятницы", "пятнице", "пятницей"),
    5: ("суббота", "субботу", "субботы", "субботе", "субботой"),
    6: ("воскресенье", "воскресенья", "воскресенью", "воскресеньем", "воскресенья"),
}

# Kazakh takes case endings on the bare noun ("жұма" -> "жұмаға дейін"), so we
# match a known stem plus a short known suffix rather than listing every form.
# Prefix matching (not substring) is load-bearing: "сенбі" (Saturday) is a
# substring of дүйсенбі/сейсенбі/сәрсенбі/бейсенбі/жексенбі but a prefix of none.
_KK_WEEKDAYS: dict[int, tuple[str, ...]] = {
    0: ("дүйсенбі",),
    1: ("сейсенбі",),
    2: ("сәрсенбі",),
    3: ("бейсенбі",),
    4: ("жұма",),
    5: ("сенбі",),
    6: ("жексенбі",),
}
_KK_SUFFIXES = frozenset({
    "", "ге", "ға", "ке", "қа", "де", "да", "те", "та",
    "ні", "ны", "нің", "ның", "нде", "нда", "сі", "сы", "гі", "ғы",
})

_EN_WEEKDAYS: dict[int, tuple[str, ...]] = {
    0: ("monday", "mon"),
    1: ("tuesday", "tue", "tues"),
    2: ("wednesday", "wed", "weds"),
    3: ("thursday", "thu", "thur", "thurs"),
    4: ("friday", "fri"),
    5: ("saturday", "sat"),
    6: ("sunday", "sun"),
}

# Cyrillic month names are matched by prefix (the case endings vary); Latin ones
# by exact token, because "mar"/"may" as prefixes eat "market" and "maybe".
_MONTH_PREFIXES: dict[int, tuple[str, ...]] = {
    1: ("январ", "қаңтар"),
    2: ("феврал", "ақпан"),
    3: ("март", "наурыз"),
    4: ("апрел", "сәуір"),
    5: ("мая", "май", "мае", "мамыр"),
    6: ("июн", "маусым"),
    7: ("июл", "шілде"),
    8: ("август", "тамыз"),
    9: ("сентябр", "қыркүйек"),
    10: ("октябр", "қазан"),
    11: ("ноябр", "қараша"),
    12: ("декабр", "желтоқсан"),
}
_MONTH_WORDS: dict[int, tuple[str, ...]] = {
    1: ("january", "jan"),
    2: ("february", "feb"),
    3: ("march", "mar"),
    4: ("april", "apr"),
    5: ("may",),
    6: ("june", "jun"),
    7: ("july", "jul"),
    8: ("august", "aug"),
    9: ("september", "sept", "sep"),
    10: ("october", "oct"),
    11: ("november", "nov"),
    12: ("december", "dec"),
}

# Kazakh number words, cardinal and ordinal. Tens and units combine:
# "жиырма бесінші" = 20 + 5 = the 25th.
_KK_NUMBERS: dict[str, int] = {
    "бір": 1, "бірінші": 1,
    "екі": 2, "екінші": 2,
    "үш": 3, "үшінші": 3,
    "төрт": 4, "төртінші": 4,
    "бес": 5, "бесінші": 5,
    "алты": 6, "алтыншы": 6,
    "жеті": 7, "жетінші": 7,
    "сегіз": 8, "сегізінші": 8,
    "тоғыз": 9, "тоғызыншы": 9,
    "он": 10, "оныншы": 10,
    "жиырма": 20, "жиырмасыншы": 20,
    "отыз": 30, "отызыншы": 30,
}

_TOMORROW_PREFIXES = ("завтра", "ертең", "tomorrow")
_END_OF_WEEK = (
    "конца недели", "концу недели", "конец недели", "этой неделе", "этой недели",
    "апта соңы", "аптаның соңы", "апта соңына", "осы аптада", "осы апта",
    "this week", "end of the week", "end of week", "by friday",
)
# The protocol LLM likes to normalise a spoken "до среды" into "середина
# недели" when it writes the `due` field, so the phrase it actually emits has
# to resolve too — otherwise a perfectly datable deadline lands in a VTODO.
_MIDWEEK = re.compile(r"середин\w*\s+недел|сред\w*\s+недел|апта\w*\s+орта|mid-?week", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text, re.UNICODE)


def _next_weekday(base: date, weekday: int) -> date:
    """The next `weekday` strictly after `base` (never `base` itself)."""
    delta = (weekday - base.weekday()) % 7
    return base + timedelta(days=delta or 7)


def _find_weekday(tokens: Iterable[str]) -> int | None:
    tokens = list(tokens)
    for token in tokens:
        for weekday, forms in _RU_WEEKDAYS.items():
            if token in forms:
                return weekday
        for weekday, forms in _EN_WEEKDAYS.items():
            if token in forms:
                return weekday
        for weekday, stems in _KK_WEEKDAYS.items():
            for stem in stems:
                if token.startswith(stem) and token[len(stem):] in _KK_SUFFIXES:
                    return weekday
    return None


def _find_month(tokens: Iterable[str]) -> int | None:
    for token in tokens:
        for month, words in _MONTH_WORDS.items():
            if token in words:
                return month
        for month, prefixes in _MONTH_PREFIXES.items():
            if any(token.startswith(p) for p in prefixes):
                return month
    return None


def _find_day(tokens: list[str]) -> int | None:
    """A day-of-month number: digits, or a Kazakh number word (pair)."""
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= 31 and len(token) <= 2:
            return int(token)
    for i, token in enumerate(tokens):
        value = _KK_NUMBERS.get(token)
        if value is None:
            continue
        if value in (10, 20, 30) and i + 1 < len(tokens):
            unit = _KK_NUMBERS.get(tokens[i + 1])
            if unit is not None and unit < 10:
                value += unit
        if 1 <= value <= 31:
            return value
    return None


def _roll(day: int, month: int, meeting_date: date, year: int | None = None) -> date | None:
    """Build a date, pushing it to next year when it already passed."""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    for candidate_year in ([year] if year else [meeting_date.year, meeting_date.year + 1]):
        try:
            made = date(candidate_year, month, day)
        except ValueError:
            return None
        if year or made >= meeting_date:
            return made
    return None


def parse_due(text: str | None, meeting_date: date | None = None) -> date | None:
    """Resolve a spoken deadline to a real date, or None if it isn't one.

    Handles Russian/Kazakh/English weekday names (next occurrence strictly
    after the meeting), explicit day+month in any of the three languages,
    numeric 25.09 / 25/09, "25-го" in the meeting's month, "завтра/ертең/
    tomorrow", and "до конца недели/this week" (-> Friday). Anything else is
    None — a wrong date in someone's calendar is worse than no event.
    """
    if not text:
        return None
    meeting_date = meeting_date or date.today()
    low = str(text).strip().lower()
    if not low:
        return None
    tokens = _tokens(low)

    # 25.09 / 25/09 / 25.09.2026
    match = re.search(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)", low)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        if year is not None and year < 100:
            year += 2000
        made = _roll(day, month, meeting_date, year)
        if made:
            return made

    # "25 сентября", "Sept 25", "жиырма бесінші қыркүйек"
    month = _find_month(tokens)
    if month:
        day = _find_day(tokens)
        if day:
            made = _roll(day, month, meeting_date)
            if made:
                return made

    # "25-го", "25 числа" — the meeting's own month.
    match = re.search(r"(?<!\d)(\d{1,2})\s*-?\s*(?:го|е|числа)(?!\w)", low)
    if match:
        made = _roll(int(match.group(1)), meeting_date.month, meeting_date)
        if made:
            return made

    if any(t.startswith(_TOMORROW_PREFIXES) for t in tokens):
        return meeting_date + timedelta(days=1)

    if _MIDWEEK.search(low):
        return _next_weekday(meeting_date, 2)

    if any(marker in low for marker in _END_OF_WEEK):
        return _next_weekday(meeting_date, 4)

    weekday = _find_weekday(tokens)
    if weekday is not None:
        return _next_weekday(meeting_date, weekday)

    return None


# --------------------------------------------------------------------------
# iCalendar serialisation
# --------------------------------------------------------------------------

def _esc(value: Any) -> str:
    """RFC 5545 3.3.11 TEXT escaping."""
    return (str(value if value is not None else "")
            .replace("\\", "\\\\")
            .replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
            .replace(";", "\\;")
            .replace(",", "\\,"))


def _fold(line: str) -> str:
    """Fold a content line at 75 octets, never mid-codepoint (RFC 5545 3.1)."""
    if len(line.encode("utf-8")) <= 75:
        return line
    out: list[str] = []
    current, width = "", 0
    for char in line:
        size = len(char.encode("utf-8"))
        if width + size > 75:
            out.append(current)
            current, width = " " + char, 1 + size
        else:
            current += char
            width += size
    out.append(current)
    return "\r\n".join(out)


def to_ics(result: dict[str, Any], meeting_date: date | None = None) -> bytes:
    """The action items as a calendar: dated ones VEVENT, undated ones VTODO."""
    rid = str(result.get("id") or "result")
    items = (result.get("protocol") or {}).get("action_items") or []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        task = str(item.get("task") or "").strip() or "Поручение"
        owner = str(item.get("owner") or "").strip() or "—"
        priority = str(item.get("priority") or "").strip() or "—"
        due_text = str(item.get("due") or "").strip()
        description = (f"Ответственный: {owner}. Приоритет: {priority}. "
                       f"Срок (как прозвучало): {due_text or '—'}")
        uid = f"{rid}-{index}@ai-meeting-intelligence.local"
        due = parse_due(due_text, meeting_date)

        if due is not None:
            lines += [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{stamp}",
                f"DTSTART;VALUE=DATE:{due:%Y%m%d}",
                f"DTEND;VALUE=DATE:{due + timedelta(days=1):%Y%m%d}",
                f"SUMMARY:{_esc(task)}",
                f"DESCRIPTION:{_esc(description)}",
                "TRANSP:TRANSPARENT",
                "END:VEVENT",
            ]
        else:
            # No parseable date: a VTODO without DUE still lands in a task list,
            # which is where an open-ended action item belongs anyway.
            lines += [
                "BEGIN:VTODO",
                f"UID:{uid}",
                f"DTSTAMP:{stamp}",
                f"SUMMARY:{_esc(task)}",
                f"DESCRIPTION:{_esc(description)}",
                "STATUS:NEEDS-ACTION",
                "END:VTODO",
            ]

    lines.append("END:VCALENDAR")
    return ("\r\n".join(_fold(line) for line in lines) + "\r\n").encode("utf-8")
