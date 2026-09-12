"""Tests for the meeting chat (pipeline.answer_question and its route).

Retrieval and the failure paths are pure unit tests with a monkeypatched
`ollama.chat`. One live test at the bottom actually asks the local model a
question and is skipped — not failed — when Ollama is not reachable.
"""

from __future__ import annotations

import time

import pytest

import pipeline


def _ollama_ready() -> bool:
    try:
        available = {
            m.get("model") or m.get("name")
            for m in pipeline.ollama.list().get("models", [])
        }
    except Exception:
        return False
    return any(candidate in available for candidate in pipeline.LLM_MODEL_CANDIDATES)


requires_ollama = pytest.mark.skipif(
    not _ollama_ready(),
    reason="Ollama unreachable at localhost:11434 or no supported model pulled",
)

MEETING = {
    "id": "meet1",
    "protocol_language": "ru",
    "language": "ru",
    "transcript": "…",
    "speakers": [
        {"speaker": "Speaker 1", "start": 0.0, "end": 12.0,
         "text": "Коллеги, начнём с запуска. Сроки сдвигаются на неделю."},
        {"speaker": "Speaker 2", "start": 12.0, "end": 40.0,
         "text": "По найму: закрываем две вакансии в отделе поддержки до конца месяца."},
        {"speaker": "Speaker 1", "start": 56.0, "end": 78.0,
         "text": "Маркетинговый бюджет на квартал утверждаем в размере двух миллионов тенге."},
        {"speaker": "Speaker 2", "start": 78.0, "end": 95.0,
         "text": "Тогда Айгерим готовит план размещения до среды."},
    ],
    "protocol": {"action_items": []},
}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def test_budget_question_ranks_the_budget_passage_first():
    ranked = pipeline.rank_passages(MEETING, "Что решили по бюджету?")
    score, best = ranked[0]
    assert score > 0
    assert best["start"] == 56.0
    assert "бюджет" in best["text"].lower()


def test_inflected_forms_still_match():
    """"бюджету" vs "бюджет" — the 5-character prefix is the whole point."""
    ranked = pipeline.rank_passages(MEETING, "какой бюджет утвердили")
    assert ranked[0][1]["start"] == 56.0


def test_short_meetings_go_in_whole_and_in_order():
    chosen = pipeline.select_passages(MEETING, "Что решили по бюджету?")
    assert len(chosen) == len(MEETING["speakers"])
    assert [p["start"] for p in chosen] == sorted(p["start"] for p in chosen)


def test_long_meetings_are_trimmed_to_the_top_k_but_stay_chronological():
    big = {
        "id": "big",
        "segments": [
            {"speaker": "Speaker 1", "start": float(i * 10), "end": float(i * 10 + 9),
             "text": f"Обсуждение пункта номер {i} по повестке дня."}
            for i in range(60)
        ] + [
            {"speaker": "Speaker 2", "start": 700.0, "end": 710.0,
             "text": "Бюджет на маркетинг утверждён."},
        ],
    }
    chosen = pipeline.select_passages(big, "Что с бюджетом на маркетинг?")
    assert len(chosen) == pipeline.CHAT_TOP_K
    assert [p["start"] for p in chosen] == sorted(p["start"] for p in chosen)
    assert any(p["start"] == 700.0 for p in chosen), "the relevant passage must survive the cut"


def test_passages_prefer_speaker_blocks_over_raw_segments():
    result = {
        "speakers": [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "блок"}],
        "segments": [{"speaker": "", "start": 0.0, "end": 1.0, "text": "сегмент"}],
    }
    assert [p["text"] for p in pipeline.chat_passages(result)] == ["блок"]


TEXT_ONLY = {
    "id": "text1",
    "protocol_language": "ru",
    "language": "ru",
    "segments": [],
    "transcript": (
        "Жаныбек: Коллеги, начнём с маркетингового бюджета на четвёртый квартал. "
        "Айгерим: Предлагаю утвердить два миллиона тенге, половина на digital. "
        "Жаныбек: Хорошо, бюджет утверждаем в размере двух миллионов тенге. "
        "Айгерим готовит план размещения до среды."
    ),
}


def test_a_text_only_meeting_is_still_chattable():
    """/process-text stores no segments — the fast demo path must still answer."""
    passages = pipeline.chat_passages(TEXT_ONLY)
    assert passages, "a bare transcript must still produce passages"
    assert all(p["start"] == 0.0 for p in passages), "times are not invented"
    assert "бюджет" in " ".join(p["text"] for p in passages).lower()


def test_text_only_meeting_is_not_asked_to_cite_timestamps(monkeypatch):
    seen: list = []
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(pipeline.ollama, "chat", _fake_chat("Два миллиона тенге.", seen))
    out = pipeline.answer_question(TEXT_ONLY, "Что решили по бюджету?", "ru")
    system, user = (m["content"] for m in seen[0]["messages"])
    assert "[мм:сс]" not in system, "nothing to cite when there is no timeline"
    assert "[фрагмент 1]" in user
    assert out["answer"] == "Два миллиона тенге."
    assert out["sources"], "the excerpts actually read should still be shown"


def test_untimed_passages_keep_document_order():
    """Untimed passages all start at 0.0, so ordering has to come from the text."""
    meeting = {"transcript": " ".join(
        f"Пункт номер {i} обсуждали подробно, по существу и без лишних отступлений."
        for i in range(250)
    )}
    everything = [p["text"] for p in pipeline.chat_passages(meeting)]
    assert len(everything) > pipeline.CHAT_ALL_PASSAGES_UNDER, "need enough to force a cut"
    chosen = [p["text"] for p in pipeline.select_passages(meeting, "что по пункту 7")]
    assert len(chosen) == pipeline.CHAT_TOP_K
    assert chosen == [t for t in everything if t in chosen], "prompt order must follow the text"


def test_language_resolution():
    assert pipeline.chat_language(MEETING, "en") == "en"
    assert pipeline.chat_language(MEETING, "auto") == "ru"
    assert pipeline.chat_language({"protocol_language": "auto", "language": "kk"}, "auto") == "kk"
    assert pipeline.chat_language({}, None) == "ru"


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

def _fake_chat(answer: str, seen: list | None = None):
    def chat(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return {"message": {"content": answer}}
    return chat


def test_cited_timestamp_selects_the_source(monkeypatch):
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(
        pipeline.ollama, "chat",
        _fake_chat("Бюджет на квартал — два миллиона тенге [00:56]."),
    )
    out = pipeline.answer_question(MEETING, "Что решили по бюджету?", "ru")
    assert "два миллиона" in out["answer"]
    assert out["language"] == "ru"
    assert [s["start"] for s in out["sources"]] == [56.0]
    assert out["sources"][0]["speaker"] == "Speaker 1"
    assert set(out["sources"][0]) == {"start", "end", "speaker", "text"}


def test_uncited_answer_falls_back_to_the_top_matches(monkeypatch):
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(pipeline.ollama, "chat", _fake_chat("Два миллиона тенге."))
    out = pipeline.answer_question(MEETING, "Что решили по бюджету?", "ru")
    assert 1 <= len(out["sources"]) <= 3
    assert any(s["start"] == 56.0 for s in out["sources"])


def test_the_prompt_carries_timestamped_excerpts_and_the_question(monkeypatch):
    seen: list = []
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(pipeline.ollama, "chat", _fake_chat("ответ [00:56]", seen))
    pipeline.answer_question(MEETING, "Что решили по бюджету?", "ru")
    assert len(seen) == 1
    kwargs = seen[0]
    assert kwargs["think"] is False
    assert "format" not in kwargs
    assert kwargs["options"]["num_ctx"] == 16384
    assert kwargs["options"]["temperature"] == 0.2
    assert kwargs["options"]["num_predict"] == 512
    user = kwargs["messages"][1]["content"]
    assert "[00:56–01:18 Speaker 1]" in user
    assert "Что решили по бюджету?" in user


def test_llm_failure_is_reported_not_raised(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("ollama is down")
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(pipeline.ollama, "chat", boom)
    out = pipeline.answer_question(MEETING, "Что решили по бюджету?", "ru")
    assert out["answer"].startswith("Не удалось получить ответ от модели:")
    assert "ollama is down" in out["answer"]
    assert out["sources"] == []


def test_unreachable_ollama_does_not_raise(monkeypatch):
    def boom():
        raise RuntimeError("no model pulled")
    monkeypatch.setattr(pipeline, "resolved_llm_model", boom)
    out = pipeline.answer_question(MEETING, "Что решили по бюджету?", "ru")
    assert out["answer"].startswith("Не удалось получить ответ от модели:")
    assert out["sources"] == []


def test_meeting_without_a_transcript_says_so(monkeypatch):
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    out = pipeline.answer_question({"id": "empty"}, "Что решили?", "ru")
    assert out["sources"] == []
    assert "расшифровки" in out["answer"]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


def test_route_answers(client, monkeypatch):
    import main
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "fake-model")
    monkeypatch.setattr(pipeline.ollama, "chat", _fake_chat("Два миллиона тенге [00:56]."))
    main.RESULTS["chat-test"] = MEETING
    try:
        response = client.post("/result/chat-test/chat",
                               json={"question": "Что решили по бюджету?", "language": "ru"})
        assert response.status_code == 200
        body = response.json()
        assert "Два миллиона" in body["answer"]
        assert body["sources"][0]["start"] == 56.0
        assert body["language"] == "ru"
    finally:
        main.RESULTS.pop("chat-test", None)


def test_route_rejects_an_empty_question(client):
    import main
    main.RESULTS["chat-test"] = MEETING
    try:
        response = client.post("/result/chat-test/chat", json={"question": "   "})
        assert response.status_code == 400
    finally:
        main.RESULTS.pop("chat-test", None)


def test_route_404s_for_an_unknown_meeting(client):
    response = client.post("/result/nope/chat", json={"question": "Что решили?"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Live — the real local model, skipped when Ollama is not up
# ---------------------------------------------------------------------------

@requires_ollama
def test_live_answer_about_the_marketing_budget():
    started = time.perf_counter()
    out = pipeline.answer_question(MEETING, "Какой бюджет утвердили на маркетинг?", "ru")
    elapsed = time.perf_counter() - started
    print(f"\nlive chat answer in {elapsed:.1f}s: {out['answer']!r}")
    assert not out["answer"].startswith("Не удалось получить ответ"), out["answer"]
    assert len(out["answer"]) > 10
    assert any("а" <= ch.lower() <= "я" for ch in out["answer"]), out["answer"]
    assert out["sources"], "an answer should point at the excerpts it used"
