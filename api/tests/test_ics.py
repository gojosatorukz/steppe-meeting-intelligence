"""Tests for the .ics export (api/ics.py) and its route.

All date assertions are relative to a fixed meeting date — Wednesday
2026-09-09 — so "до пятницы" has a single right answer that does not depend
on the day the suite happens to run.
"""

from __future__ import annotations

from datetime import date

import pytest

import ics

WEDNESDAY = date(2026, 9, 9)


# ---------------------------------------------------------------------------
# parse_due
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # Russian weekdays, various cases, with and without a preposition.
    ("до пятницы", date(2026, 9, 11)),
    ("в среду", date(2026, 9, 16)),        # strictly after the meeting day
    ("к четвергу", date(2026, 9, 10)),
    ("до понедельника", date(2026, 9, 14)),
    ("в субботу", date(2026, 9, 12)),
    ("к воскресенью", date(2026, 9, 13)),
    ("до вторника", date(2026, 9, 15)),
    # Kazakh weekdays, bare and with case endings.
    ("жұмаға дейін", date(2026, 9, 11)),
    ("дүйсенбі", date(2026, 9, 14)),
    ("сейсенбіге дейін", date(2026, 9, 15)),
    ("сәрсенбі", date(2026, 9, 16)),
    ("бейсенбіге", date(2026, 9, 10)),
    ("сенбі", date(2026, 9, 12)),          # not confused with дүй/сей/сәр/бей/жек-сенбі
    ("жексенбі", date(2026, 9, 13)),
    # English.
    ("by Wednesday", date(2026, 9, 16)),
    ("Friday", date(2026, 9, 11)),
    ("next Monday", date(2026, 9, 14)),
    ("by thu", date(2026, 9, 10)),
    # Explicit day + month.
    ("25 сентября", date(2026, 9, 25)),
    ("до 25 сентября", date(2026, 9, 25)),
    ("25-го", date(2026, 9, 25)),
    ("25.09", date(2026, 9, 25)),
    ("25/09", date(2026, 9, 25)),
    ("25.09.2027", date(2027, 9, 25)),
    ("Sept 25", date(2026, 9, 25)),
    ("September 25", date(2026, 9, 25)),
    ("25 қыркүйек", date(2026, 9, 25)),
    ("жиырма бесінші қыркүйек", date(2026, 9, 25)),
    # Relative shorthands.
    ("завтра", date(2026, 9, 10)),
    ("ертең", date(2026, 9, 10)),
    ("tomorrow", date(2026, 9, 10)),
    ("до конца недели", date(2026, 9, 11)),
    # What the protocol LLM actually writes when it hears "до среды".
    ("середина недели", date(2026, 9, 16)),
    ("к середине недели", date(2026, 9, 16)),
    ("mid-week", date(2026, 9, 16)),
    ("this week", date(2026, 9, 11)),
])
def test_parse_due_resolves(text, expected):
    assert ics.parse_due(text, WEDNESDAY) == expected


@pytest.mark.parametrize("text", [
    "", None, "по готовности", "ASAP", "как можно скорее", "не определено",
    "после релиза", "—", "белгісіз",
])
def test_parse_due_gives_up_rather_than_guessing(text):
    """A wrong date in someone's calendar is worse than no event at all."""
    assert ics.parse_due(text, WEDNESDAY) is None


def test_past_day_and_month_rolls_into_next_year():
    assert ics.parse_due("5 марта", WEDNESDAY) == date(2027, 3, 5)


def test_time_like_string_is_not_a_date():
    # 15.30 would be day 15 of month 30 — reject, don't clamp.
    assert ics.parse_due("в 15.30", WEDNESDAY) is None


# ---------------------------------------------------------------------------
# to_ics
# ---------------------------------------------------------------------------

SAMPLE = {
    "id": "abc123def456",
    "protocol": {
        "action_items": [
            {"owner": "Айгерим", "task": "Подготовить план, смету и график",
             "due": "до среды", "priority": "высокий"},
            {"owner": "Ержан", "task": "Уточнить условия у подрядчика",
             "due": "по готовности", "priority": "средний"},
        ]
    },
}


@pytest.fixture
def rendered() -> str:
    return ics.to_ics(SAMPLE, WEDNESDAY).decode("utf-8")


def test_calendar_envelope(rendered):
    assert rendered.startswith("BEGIN:VCALENDAR\r\n")
    assert rendered.rstrip("\r\n").endswith("END:VCALENDAR")
    assert "VERSION:2.0" in rendered
    assert "CALSCALE:GREGORIAN" in rendered
    assert "PRODID:" in rendered


def test_dated_item_becomes_an_all_day_vevent(rendered):
    assert rendered.count("BEGIN:VEVENT") == 1
    assert "DTSTART;VALUE=DATE:20260916" in rendered
    assert "DTEND;VALUE=DATE:20260917" in rendered          # exclusive end
    assert "DTSTAMP:" in rendered
    assert "UID:abc123def456-1@" in rendered


def test_undated_item_becomes_a_vtodo_without_due(rendered):
    assert rendered.count("BEGIN:VTODO") == 1
    assert "DUE" not in rendered
    assert "UID:abc123def456-2@" in rendered
    assert "Уточнить условия у подрядчика" in rendered


def test_description_carries_owner_priority_and_spoken_due(rendered):
    unfolded = rendered.replace("\r\n ", "")
    assert ("DESCRIPTION:Ответственный: Айгерим. Приоритет: высокий. "
            "Срок (как прозвучало): до среды") in unfolded


def test_text_is_escaped_and_lines_are_crlf_folded():
    raw = ics.to_ics(SAMPLE, WEDNESDAY)
    assert b"\r\n" in raw
    # No bare LF anywhere: every newline is part of a CRLF pair.
    assert raw.replace(b"\r\n", b"").count(b"\n") == 0
    assert all(len(line) <= 75 for line in raw.split(b"\r\n"))
    unfolded = raw.decode("utf-8").replace("\r\n ", "")
    assert "Подготовить план\\, смету и график" in unfolded


def test_special_characters_are_escaped():
    result = {"id": "x", "protocol": {"action_items": [
        {"owner": "A;B", "task": "a,b;c\\d\ne", "due": "", "priority": "low"},
    ]}}
    unfolded = ics.to_ics(result, WEDNESDAY).decode("utf-8").replace("\r\n ", "")
    assert "SUMMARY:a\\,b\\;c\\\\d\\ne" in unfolded


def test_empty_protocol_still_yields_a_valid_calendar():
    out = ics.to_ics({"id": "empty", "protocol": {"action_items": []}}).decode("utf-8")
    assert out.startswith("BEGIN:VCALENDAR")
    assert "BEGIN:VEVENT" not in out and "BEGIN:VTODO" not in out


def test_uid_is_deterministic():
    first = ics.to_ics(SAMPLE, WEDNESDAY).decode("utf-8")
    second = ics.to_ics(SAMPLE, WEDNESDAY).decode("utf-8")
    uids = lambda s: [l for l in s.split("\r\n") if l.startswith("UID:")]
    assert uids(first) == uids(second)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


def test_route_serves_the_calendar(client):
    import main
    main.RESULTS["ics-test"] = dict(SAMPLE, id="ics-test")
    try:
        response = client.get("/result/ics-test/export.ics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/calendar")
        assert "charset=utf-8" in response.headers["content-type"]
        assert response.headers["content-disposition"] == (
            'attachment; filename="action_items_ics-test.ics"'
        )
        assert response.content.startswith(b"BEGIN:VCALENDAR")
        assert b"BEGIN:VEVENT" in response.content
    finally:
        main.RESULTS.pop("ics-test", None)


def test_route_404s_for_an_unknown_id(client):
    assert client.get("/result/no-such-result/export.ics").status_code == 404
