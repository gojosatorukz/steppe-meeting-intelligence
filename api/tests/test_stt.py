"""Tests for the STT half of pipeline.py — VAD-first, per-window language routing.

The model tests need the two CPU Whisper models (the turbo from the HF cache and
the Kazakh fine-tune in api/models/kk-turbo-ct2) plus the audio fixtures in
tests/audio/; they are skipped, not failed, when those are missing. The routing
and filtering unit tests are pure Python and always run.

Run with `-s` to see the per-window routing table and the full transcript.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline

AUDIO_DIR = Path(__file__).parent / "audio"
KAZAKH_LETTERS = set("әғқңөұүі")
PARTICIPANTS = ["Жаныбек", "Бекзат", "Айгерим"]

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _words(text: str) -> list[str]:
    """Lowercase, punctuation-stripped word list — the reference .txt convention."""
    return [w for w in _NON_WORD.split(text.lower()) if w]


def _recall(reference: str, hypothesis: str) -> float:
    """Share of reference words the transcript actually contains (multiset)."""
    ref, hyp = Counter(_words(reference)), Counter(_words(hypothesis))
    total = sum(ref.values())
    return sum((ref & hyp).values()) / total if total else 0.0


def _routing_table(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(
        f"  {r.getMessage()}"
        for r in caplog.records
        if r.getMessage().startswith(("group ", "window ", "utterance "))
    )


@pytest.fixture(scope="session")
def stt_ready() -> None:
    """Load the turbo model once; skip the whole model-backed suite if it can't."""
    try:
        pipeline.load_turbo()
    except Exception as exc:                       # pragma: no cover - env dependent
        pytest.skip(f"turbo STT model unavailable offline: {exc}")
    if not pipeline.is_kk_available():
        print(f"\n[warn] Kazakh model missing at {pipeline.KK_MODEL_DIR} — kk routed to turbo")


def _audio(name: str) -> Path:
    path = AUDIO_DIR / name
    if not path.exists():
        pytest.skip(f"missing audio fixture {path}")
    return path


# --------------------------------------------------------------------------
# Unit tests (no models, always run)
# --------------------------------------------------------------------------

def test_constrained_language_ignores_higher_scoring_tatar():
    """Whisper's unrestricted top-1 on Kazakh is often Tatar; kk must still win."""
    all_probs = [("tt", 0.62), ("kk", 0.31), ("ba", 0.04), ("ru", 0.02), ("en", 0.001)]
    assert pipeline.constrained_language(all_probs) == ("kk", 0.31)


def test_constrained_language_falls_back_when_unsure():
    all_probs = [("tt", 0.80), ("kk", 0.05), ("ru", 0.04), ("en", 0.01)]
    language, probability = pipeline.constrained_language(all_probs)
    assert language == pipeline.FALLBACK_LANGUAGE == "ru"
    assert probability < pipeline.MIN_LANGUAGE_PROB


def test_constrained_language_trusts_a_clear_leader_below_the_floor():
    """Real measured posterior: weak in absolute terms, but kk leads ru by 2.2x.

    Applying the < 0.3 floor literally would route genuine Kazakh to Russian,
    which is the bug this routing exists to fix.
    """
    all_probs = [("tt", 0.40), ("kk", 0.254), ("ru", 0.113), ("en", 0.05)]
    language, probability = pipeline.constrained_language(all_probs)
    assert language == "kk"
    assert probability == pytest.approx(0.254)
    assert probability < pipeline.MIN_LANGUAGE_PROB, "the floor must really be in play here"


def test_constrained_language_falls_back_when_nothing_leads():
    """Same floor, no lead: kk is top of the three but only 1.25x ru -> ru."""
    all_probs = [("tt", 0.50), ("kk", 0.15), ("ru", 0.12), ("en", 0.10)]
    language, _ = pipeline.constrained_language(all_probs)
    assert language == pipeline.FALLBACK_LANGUAGE == "ru"


def test_constrained_language_handles_empty_posterior():
    assert pipeline.constrained_language([]) == ("ru", 0.0)


def test_drop_hallucinations():
    def seg(text: str, no_speech_prob: float = 0.0) -> SimpleNamespace:
        return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)

    segments = [
        seg("Начинаем совещание."),
        seg("Начинаем совещание."),            # verbatim repeat
        seg("Продолжение следует..."),         # boilerplate
        seg("Субтитры сделал DimaTorzok"),     # boilerplate
        seg("Редактор субтитров А.Кулакова"),  # boilerplate
        seg("Thank you for watching!"),        # boilerplate
        seg("Подписывайтесь на канал"),        # boilerplate
        seg("Тишина", 0.95),                   # no speech
        seg("   "),                            # empty
        seg("Бюджет пока не утверждён."),
    ]
    kept = [s.text for s in pipeline.drop_hallucinations(segments)]
    assert kept == ["Начинаем совещание.", "Бюджет пока не утверждён."]


def test_lid_groups_glue_short_chunks_to_the_nearest_neighbour():
    """A 1.2 s fragment is too short to language-ID; it joins the closer turn."""
    assert pipeline.MIN_LID_SEC == 5.0, "this fixture is built around the 5 s minimum"
    sr = pipeline.SAMPLE_RATE
    chunks = [(int(s * sr), int(e * sr)) for s, e in [
        (0.00, 8.00),      # 8.0 s, stands alone
        (9.00, 15.00),     # 6.0 s, stands alone
        (15.50, 16.70),    # 1.2 s: gap 0.50 back, 0.20 forward -> joins forward
        (16.90, 23.00),    # 6.1 s
        (30.00, 31.00),    # 1.0 s, but 7 s of silence around it: left alone
    ]]
    groups = [(s / sr, e / sr) for s, e in pipeline._lid_groups(chunks)]
    assert groups == [(0.0, 8.0), (9.0, 15.0), (15.5, 23.0), (30.0, 31.0)]


def test_merge_by_language_never_spans_two_languages():
    sr = pipeline.SAMPLE_RATE
    routed = [(int(s * sr), int(e * sr), lang) for s, e, lang in [
        (0.0, 8.0, "ru"),
        (8.2, 12.0, "ru"),      # same language, small gap -> merged
        (12.3, 20.0, "kk"),     # different language -> new window
        (20.2, 24.0, "kk"),
        (26.0, 30.0, "kk"),     # 2 s silence -> new window
    ]]
    merged = [(s / sr, e / sr, lang) for s, e, lang in pipeline._merge_by_language(routed)]
    assert merged == [(0.0, 12.0, "ru"), (12.3, 24.0, "kk"), (26.0, 30.0, "kk")]


def test_silent_recording_returns_an_empty_transcript_without_loading_a_model(
    monkeypatch, tmp_path
):
    """A long silent upload must come back empty and fast, not decode silence."""
    silence = np.zeros(int(90 * pipeline.SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(pipeline, "decode_audio", lambda *a, **k: silence)

    def _never(*args, **kwargs):
        raise AssertionError("no STT model may be loaded for silent audio")

    monkeypatch.setattr(pipeline, "load_turbo", _never)
    monkeypatch.setattr(pipeline, "load_kk", _never)

    t0 = time.perf_counter()
    transcript = pipeline.transcribe(tmp_path / "silence.wav")
    elapsed = time.perf_counter() - t0

    assert transcript.text == ""
    assert transcript.segments == []
    assert transcript.languages == {}
    assert transcript.language == "unknown"
    assert transcript.seconds == pytest.approx(90.0)
    assert elapsed < 10.0, f"empty path took {elapsed:.1f}s"


def test_short_silent_input_still_falls_back_to_the_whole_recording(monkeypatch, tmp_path):
    """Below the cutoff the VAD gets second-guessed: one quiet utterance is cheap to try."""
    quiet = np.zeros(int(5 * pipeline.SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(pipeline, "decode_audio", lambda *a, **k: quiet)
    monkeypatch.setattr(pipeline, "load_turbo", lambda: "stub-model")
    monkeypatch.setattr(pipeline, "_detect_language", lambda model, pcm: ("ru", 0.9))

    windows: list[tuple[float, float, str]] = []

    def fake_transcribe_window(pcm, offset, language, participants):
        end = offset + pcm.size / pipeline.SAMPLE_RATE
        windows.append((round(offset, 2), round(end, 2), language))
        return [pipeline.Segment(offset, end, "тихая реплика", language=language)]

    monkeypatch.setattr(pipeline, "_transcribe_window", fake_transcribe_window)

    transcript = pipeline.transcribe(tmp_path / "quiet.wav")

    assert windows == [(0.0, 5.0, "ru")]
    assert transcript.text == "тихая реплика"
    assert transcript.languages == {"ru": 1.0}


def test_drop_hallucinations_keeps_our_segments_without_no_speech_prob():
    segments = [
        pipeline.Segment(0.0, 1.0, "Тема одна", language="ru"),
        pipeline.Segment(1.0, 2.0, "Тема одна", language="ru"),
        pipeline.Segment(2.0, 3.0, "Тема два", language="ru"),
    ]
    assert [s.text for s in pipeline.drop_hallucinations(segments)] == ["Тема одна", "Тема два"]


# --------------------------------------------------------------------------
# Per-language samples
# --------------------------------------------------------------------------

SAMPLES = [
    ("kk_1.wav", "kk", 0.5),
    ("kk_2.wav", "kk", 0.5),
    ("ru_1.wav", "ru", 0.7),
    ("ru_2.wav", "ru", 0.7),
    ("en_1.wav", "en", 0.7),
]


@pytest.mark.parametrize("name,expected_language,min_recall", SAMPLES)
def test_sample_routes_to_its_language(stt_ready, caplog, name, expected_language, min_recall):
    audio = _audio(name)
    reference = audio.with_suffix(".txt")
    if not reference.exists():
        pytest.skip(f"missing reference {reference}")

    with caplog.at_level(logging.INFO, logger="pipeline"):
        t0 = time.perf_counter()
        transcript = pipeline.transcribe(audio)
        elapsed = time.perf_counter() - t0

    recall = _recall(reference.read_text(encoding="utf-8"), transcript.text)
    print(
        f"\n[{name}] {transcript.seconds:.1f}s audio in {elapsed:.2f}s "
        f"({transcript.seconds / elapsed:.2f}x realtime) languages={transcript.languages} "
        f"recall={recall:.2f}\n{_routing_table(caplog)}\n  {transcript.text}"
    )

    assert transcript.language == expected_language, transcript.languages
    assert [s.language for s in transcript.segments] == [expected_language] * len(
        transcript.segments
    )
    assert recall >= min_recall, f"word recall {recall:.2f} < {min_recall}"
    for segment in transcript.segments:
        assert segment.words, f"no word timestamps for {segment.text!r}"
        assert segment.start <= segment.end
        assert segment.speaker is None


def test_transcribe_pcm_single_utterance(stt_ready, caplog):
    """The live path: one utterance, no VAD, must come back in a couple of seconds."""
    from faster_whisper import decode_audio

    pcm = decode_audio(str(_audio("ru_1.wav")), sampling_rate=pipeline.SAMPLE_RATE)
    utterance = pcm[: 6 * pipeline.SAMPLE_RATE]

    with caplog.at_level(logging.INFO, logger="pipeline"):
        t0 = time.perf_counter()
        transcript = pipeline.transcribe_pcm(utterance, pipeline.SAMPLE_RATE)
        elapsed = time.perf_counter() - t0

    print(
        f"\n[transcribe_pcm] {utterance.size / pipeline.SAMPLE_RATE:.1f}s utterance "
        f"in {elapsed:.2f}s -> {transcript.language}: {transcript.text}"
    )
    assert transcript.language == "ru"
    assert transcript.text
    assert elapsed < 20.0, f"live path too slow: {elapsed:.2f}s for a 6 s utterance"


# --------------------------------------------------------------------------
# The real recording — three languages, two speakers, one file
# --------------------------------------------------------------------------

def test_team_mixed_recording(stt_ready, caplog):
    audio = _audio("team_mixed_35.m4a")

    with caplog.at_level(logging.INFO, logger="pipeline"):
        t0 = time.perf_counter()
        transcript = pipeline.transcribe(audio, participants=PARTICIPANTS)
        elapsed = time.perf_counter() - t0

    lines = "\n".join(
        f"  [{s.start:6.2f}-{s.end:6.2f}] {s.language}: {s.text}" for s in transcript.segments
    )
    print(
        f"\n[team_mixed_35] {transcript.seconds:.1f}s audio in {elapsed:.2f}s "
        f"({transcript.seconds / elapsed:.2f}x realtime)\n"
        f"languages={transcript.languages} dominant={transcript.language}\n"
        f"{_routing_table(caplog)}\n{lines}\n\nFULL TRANSCRIPT:\n{transcript.text}"
    )

    # The participant prompt is what keeps this from being "Пикзат".
    assert "Бекзат" in transcript.text
    # Kazakh actually transcribed as Kazakh, not hallucinated into Russian.
    assert KAZAKH_LETTERS & set(transcript.text.lower()), "no Kazakh-specific letters"
    # English kept in English instead of being translated to Russian.
    assert "London" in transcript.text
    assert "Wednesday" in transcript.text
    # All three languages present in the routing.
    for language in ("kk", "ru", "en"):
        assert transcript.languages.get(language, 0.0) > 0.0, transcript.languages


# --------------------------------------------------------------------------
# Repetition-loop safety net (weight-free)
# --------------------------------------------------------------------------

def test_collapse_repeats_keeps_the_surrounding_words():
    assert pipeline.collapse_repeats("а Бекзат Бекзат Бекзат Бекзат б") == "а Бекзат б"


def test_collapse_repeats_is_case_and_punctuation_tolerant():
    assert pipeline.collapse_repeats("Бекзат, бекзат, Бекзат, Бекзат") == "Бекзат"
    # Two in a row is emphasis, not a loop — left alone.
    assert pipeline.collapse_repeats("да да") == "да да"


def test_collapse_repeats_rescues_the_live_kazakh_loop():
    looped = "жарайды жұмаға дейін дайындаймын " + "Бекзат " * 50 + "терверге қолжетімділік"
    collapsed = pipeline.collapse_repeats(looped)
    assert collapsed.count("Бекзат") == 1
    assert "жарайды жұмаға дейін дайындаймын" in collapsed
    assert "терверге қолжетімділік" in collapsed


def test_drop_hallucinations_drops_a_segment_that_is_only_a_loop():
    only_loop = pipeline.Segment(0.0, 3.0, "Бекзат, Бекзат, Бекзат, Бекзат", "kk")
    assert pipeline.drop_hallucinations([only_loop]) == []


def test_drop_hallucinations_collapses_and_rebuilds_the_word_list():
    words = [{"start": float(i), "end": i + 1.0, "text": t}
             for i, t in enumerate(["а", "Бекзат", "Бекзат", "Бекзат", "Бекзат", "б"])]
    segment = pipeline.Segment(0.0, 6.0, "а Бекзат Бекзат Бекзат Бекзат б", "kk", words)
    kept = pipeline.drop_hallucinations([segment])

    assert [s.text for s in kept] == ["а Бекзат б"]
    # The old words no longer describe the text, so one pseudo-word spans it.
    assert kept[0].words == [{"start": 0.0, "end": 6.0, "text": "а Бекзат б"}]


def test_drop_hallucinations_leaves_clean_speech_alone():
    clean = pipeline.Segment(0.0, 2.0, "Бюджет утверждён, Бекзат готовит план.", "ru")
    assert [s.text for s in pipeline.drop_hallucinations([clean])] == [clean.text]


def test_only_the_turbo_model_gets_the_participants_prompt(monkeypatch):
    """The name hint is what makes the Kazakh fine-tune loop — it must not see it."""
    calls: dict[str, dict] = {}

    def fake(name):
        def transcribe(pcm, **kwargs):
            calls[name] = kwargs
            return iter(()), None
        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setattr(pipeline, "load_turbo", lambda: fake("turbo"))
    monkeypatch.setattr(pipeline, "load_kk", lambda: fake("kk"))
    pcm = np.zeros(pipeline.SAMPLE_RATE, dtype=np.float32)

    pipeline._transcribe_window(pcm, 0.0, "kk", PARTICIPANTS)
    pipeline._transcribe_window(pcm, 0.0, "ru", PARTICIPANTS)

    kk, turbo = calls["kk"], calls["turbo"]
    assert "initial_prompt" not in kk and "hotwords" not in kk
    assert turbo["initial_prompt"] == "Жаныбек, Бекзат, Айгерим"
    # The Kazakh model gets the anti-loop decode knobs instead.
    assert kk["no_repeat_ngram_size"] == 3
    assert kk["repetition_penalty"] == 1.15
    # And both decode on a temperature ladder so the repetition guard can retry.
    for kwargs in (kk, turbo):
        assert kwargs["temperature"] == [0.0, 0.2, 0.4]
        assert kwargs["compression_ratio_threshold"] == 2.4
        assert kwargs["condition_on_previous_text"] is False


def test_resolve_protocol_language():
    assert pipeline.resolve_protocol_language(None, "kk") == "ru"      # default
    assert pipeline.resolve_protocol_language("kk", "ru") == "kk"
    assert pipeline.resolve_protocol_language("auto", "kk") == "kk"
    assert pipeline.resolve_protocol_language("auto", "unknown") is None
    assert pipeline.resolve_protocol_language("nonsense", "en") == "en"
