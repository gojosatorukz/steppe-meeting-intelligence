"""Tests for RESULTS persistence (api/main.py) — weight-free, no models loaded.

Uses `TestClient(main.app)` *without* the `with` form so the lifespan (and its
model warm-up) never runs — these tests only exercise save_result/load_results
and the read routes against an in-memory RESULTS dict we control directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import offline_guard  # noqa: E402  # arms the guard for this process, as main does
import main  # noqa: E402


@pytest.fixture(autouse=True)
def clean_results():
    """Every test starts and ends with the RESULTS dict as it found it."""
    saved = dict(main.RESULTS)
    main.RESULTS.clear()
    yield
    main.RESULTS.clear()
    main.RESULTS.update(saved)


def fake_result(rid: str = "abc123def456") -> dict:
    return {
        "id": rid,
        "filename": "совещание.wav",
        "language": "ru",
        "timing": {"total_s": 1.23},
        "transcript": "Привет, коллеги! Обсудим протокол собрания.",
        "summary": "Договорились перенести встречу на среду.",
    }


def test_save_result_round_trips_cyrillic(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    result = fake_result()

    main.save_result(result)

    path = tmp_path / f"{result['id']}.json"
    assert path.is_file()
    raw = path.read_text(encoding="utf-8")
    # ensure_ascii=False: Cyrillic must be literal text, not \uXXXX escapes.
    assert "Привет" in raw
    assert "\\u" not in raw

    loaded = json.loads(raw)
    assert loaded["id"] == result["id"]
    assert loaded["transcript"] == result["transcript"]
    assert loaded["summary"] == result["summary"]
    assert loaded["filename"] == result["filename"]
    assert "created_at" in loaded  # stamped by save_result if absent


def test_load_results_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    result = fake_result("restartid001")
    main.save_result(result)

    # Simulate a process restart: the in-memory dict is gone, disk is not.
    main.RESULTS.clear()
    assert main.RESULTS == {}

    count = main.load_results()
    assert count == 1
    assert "restartid001" in main.RESULTS
    assert main.RESULTS["restartid001"]["transcript"] == result["transcript"]

    client = TestClient(main.app)  # no `with`: skips lifespan / model warm-up
    resp = client.get("/result/restartid001")
    assert resp.status_code == 200
    assert resp.json()["id"] == "restartid001"

    resp = client.get("/result/restartid001/export.json")
    assert resp.status_code == 200


def test_load_results_skips_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    good = fake_result("goodid0001")
    main.save_result(good)

    corrupt = tmp_path / "corrupt0001.json"
    corrupt.write_text("{not valid json at all", encoding="utf-8")

    count = main.load_results()

    assert count == 1
    assert "goodid0001" in main.RESULTS
    assert "corrupt0001" not in main.RESULTS
