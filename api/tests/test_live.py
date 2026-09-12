"""Tests for the live mic path (api/live.py) — no microphone, no GPU.

Everything that touches PortAudio or the ONNX VAD is either injected as a fake
or guarded with ``skipif``. The state machine, the worker queue, the timestamp
arithmetic and the Finish sequencing are tested with fakes and run everywhere.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live  # noqa: E402
import pipeline  # noqa: E402

SR = live.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSegment:
    """Mirrors sherpa_onnx SpeechSegment: `.start` (sample index) + `.samples`."""

    def __init__(self, start: int, samples):
        self.start = start
        self.samples = list(samples)


class FakeVad:
    """Mirrors sherpa_onnx.VoiceActivityDetector's surface.

    `script` is a list of ``(after_n_frames, start_sample, n_samples)``: the
    fake emits that speech segment once it has been fed that many frames.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.frames = 0
        self._ready: list[FakeSegment] = []
        self.flushed = False

    def accept_waveform(self, frame):
        assert len(frame) == live.FRAME_SAMPLES, f"VAD fed {len(frame)} samples"
        self.frames += 1
        for at, start, n in self.script:
            if at == self.frames:
                self._ready.append(FakeSegment(start, np.zeros(n, dtype=np.float32)))

    def empty(self) -> bool:
        return not self._ready

    @property
    def front(self):
        return self._ready[0]

    def pop(self):
        self._ready.pop(0)

    def flush(self):
        self.flushed = True

    def is_speech_detected(self) -> bool:
        return False

    def reset(self):
        self._ready.clear()


class RecordingHub(live.Hub):
    """A Hub that keeps every event, including the ones not kept for replay."""

    def __init__(self):
        super().__init__()
        self.events: list[dict] = []
        self.at: list[float] = []

    def publish(self, event, keep=True):
        self.events.append(event)
        self.at.append(time.perf_counter())
        super().publish(event, keep=keep)

    def time_of(self, kind: str, **match) -> float:
        """When the first event of `kind` (optionally matching fields) went out."""
        for t, e in zip(self.at, self.events):
            if e["type"] == kind and all(e.get(k) == v for k, v in match.items()):
                return t
        raise AssertionError(f"no {kind} event {match} in {self.types()}")

    def types(self) -> list[str]:
        """Event types, minus the 5 Hz mic-level noise."""
        return [e["type"] for e in self.events if e["type"] != "level"]

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]


def fake_transcriber(segments, language="ru"):
    """Build a transcriber returning `segments` as (start, end, text, lang)."""

    def call(pcm, sr=SR, participants=None):
        call.calls.append((np.asarray(pcm).size, sr, tuple(participants or ())))
        segs = [
            pipeline.Segment(
                start=s, end=e, text=t, language=lang,
                words=[{"start": s, "end": e, "text": t}],
            )
            for s, e, t, lang in segments
        ]
        return pipeline.Transcript(
            text=" ".join(s.text for s in segs), language=language,
            segments=segs, seconds=np.asarray(pcm).size / SR,
            languages={language: 1.0},
        )

    call.calls = []
    return call


def blocks(seconds: float) -> np.ndarray:
    """A block of quiet noise, `seconds` long."""
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(SR * seconds)) * 0.01).astype(np.float32)


def wait_until(predicate, timeout=5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def make_session(**kw):
    kw.setdefault("hub", RecordingHub())
    kw.setdefault("transcriber", fake_transcriber([(0.0, 1.0, "привет", "ru")]))
    kw.setdefault("vad", FakeVad())
    return live.LiveSession(**kw)


# ---------------------------------------------------------------------------
# Capture / VAD framing
# ---------------------------------------------------------------------------

def test_feed_keeps_all_raw_pcm_not_only_speech():
    """Diarization runs on the whole recording, so every sample is retained."""
    s = make_session()
    s.feed(blocks(0.5))
    s.feed(blocks(0.25))
    assert s.raw_pcm().size == int(SR * 0.75)
    assert s.raw_pcm().dtype == np.float32


def test_feed_chunks_audio_into_512_sample_vad_frames():
    vad = FakeVad()
    s = make_session(vad=vad)
    s.feed(np.zeros(700, dtype=np.float32))      # 1 frame, 188 carried
    assert vad.frames == 1
    s.feed(np.zeros(324, dtype=np.float32))      # 188 + 324 = 512 -> 1 more
    assert vad.frames == 2
    assert s.raw_pcm().size == 1024


def test_level_events_are_emitted_while_capturing():
    s = make_session()
    for _ in range(20):
        s.feed(blocks(0.05))
    levels = [e for e in s.hub.events if e["type"] == "level"]
    assert levels, "expected mic level events"
    assert all(0.0 <= e["rms"] <= 1.0 for e in levels)


# ---------------------------------------------------------------------------
# Utterance queue + worker
# ---------------------------------------------------------------------------

def test_finalized_utterance_broadcasts_pending_then_segment():
    """The page must show a '…' row immediately, then have it replaced."""
    vad = FakeVad(script=[(1, SR * 2, SR)])       # 1 s of speech starting at t=2 s
    s = make_session(vad=vad)
    s.feed(np.zeros(512, dtype=np.float32))

    assert s.hub.types()[:2] == ["pending", "queue"]
    assert s.hub.of("pending")[0] == {"type": "pending", "utterance_id": 0,
                                      "start": 2.0, "end": 3.0}
    assert s.hub.of("queue")[0] == {"type": "queue", "pending": 1}

    assert s.process_queued() == 1
    types = s.hub.types()
    assert "segment" in types
    assert types.index("pending") < types.index("segment")
    assert s.hub.events[-1] == {"type": "queue", "pending": 0}


def test_segment_times_are_offset_into_session_time():
    """transcribe_pcm returns utterance-relative times; the session is absolute."""
    vad = FakeVad(script=[(1, SR * 10, SR * 2)])  # utterance at 10.0 s, 2 s long
    s = make_session(
        vad=vad,
        transcriber=fake_transcriber([(0.0, 0.8, "раз", "ru"), (0.9, 2.0, "два", "kk")]),
    )
    s.feed(np.zeros(512, dtype=np.float32))
    s.process_queued()

    segs = [e for e in s.hub.events if e["type"] == "segment"]
    assert [(e["start"], e["end"], e["text"]) for e in segs] == [
        (10.0, 10.8, "раз"), (10.9, 12.0, "два")
    ]
    assert [e["language"] for e in segs] == ["ru", "kk"]
    assert all(e["utterance_id"] == 0 for e in segs)
    assert [(x["start"], x["end"]) for x in s.session_words()] == [(10.0, 10.8), (10.9, 12.0)]


def test_participants_are_forwarded_to_the_transcriber():
    tr = fake_transcriber([(0.0, 1.0, "ok", "en")])
    vad = FakeVad(script=[(1, 0, SR)])
    s = make_session(vad=vad, transcriber=tr, participants=["Айгүл", "Бек"])
    s.feed(np.zeros(512, dtype=np.float32))
    s.process_queued()
    assert tr.calls[0][2] == ("Айгүл", "Бек")
    assert tr.calls[0][0] == SR          # the utterance's samples, not the whole stream


def test_empty_utterance_is_dropped_not_shown():
    vad = FakeVad(script=[(1, 0, SR)])
    s = make_session(vad=vad, transcriber=fake_transcriber([(0.0, 1.0, "   ", "ru")]))
    s.feed(np.zeros(512, dtype=np.float32))
    s.process_queued()

    assert [e["type"] for e in s.hub.events if e["type"] == "segment"] == []
    assert {"type": "drop", "utterance_id": 0} in s.hub.events
    assert s.transcript_text() == ""


def test_transcriber_failure_does_not_kill_the_session():
    def boom(pcm, sr=SR, participants=None):
        raise RuntimeError("model exploded")

    vad = FakeVad(script=[(1, 0, SR), (2, SR, SR)])
    s = make_session(vad=vad, transcriber=boom)
    s.feed(np.zeros(1024, dtype=np.float32))
    assert s.process_queued() == 2
    assert s.hub.events[-1] == {"type": "queue", "pending": 0}
    assert any(e["type"] == "drop" for e in s.hub.events)


def test_worker_thread_serialises_utterances_and_never_blocks_the_feed():
    """One worker, FIFO: transcription is ~8 s/utterance on this CPU."""
    order: list[int] = []
    gate = threading.Event()

    def slow(pcm, sr=SR, participants=None):
        gate.wait(2.0)
        order.append(len(order))
        return pipeline.Transcript(
            text="x", language="ru",
            segments=[pipeline.Segment(start=0.0, end=1.0, text="x", language="ru")],
            seconds=1.0, languages={"ru": 1.0},
        )

    vad = FakeVad(script=[(1, 0, SR), (2, SR, SR), (3, SR * 2, SR)])
    s = make_session(vad=vad, transcriber=slow)
    s.start_worker()
    t0 = time.perf_counter()
    s.feed(np.zeros(512 * 3, dtype=np.float32))   # feed returns immediately
    feed_time = time.perf_counter() - t0
    assert feed_time < 0.5, f"feed blocked for {feed_time:.2f}s"
    assert [e for e in s.hub.events if e["type"] == "queue"][-1]["pending"] == 3

    gate.set()
    s.stop_worker(timeout=10)
    assert order == [0, 1, 2]


def test_worker_survives_a_failure_after_transcription():
    """Everything past the transcribe call can raise; the loop must not end.

    A worker that dies takes every later utterance with it and leaves the
    pending count stuck, which then freezes Finish.
    """
    hub = ExplodingHub(fail_on="segment")
    vad = FakeVad(script=[(1, 0, SR), (2, SR, SR)])
    s = make_session(hub=hub, vad=vad)
    s.start_worker()
    s.feed(np.zeros(1024, dtype=np.float32))

    assert wait_until(lambda: s.pending_count() == 0), \
        f"queue stranded at {s.pending_count()}"
    assert hub.exploded
    assert s._worker.is_alive(), "the worker thread died on one bad utterance"
    assert {"type": "drop", "utterance_id": 0} in hub.events, \
        "the failed utterance left a '…' row on the page"
    assert [e for e in hub.events
            if e["type"] == "segment" and e.get("utterance_id") == 1], \
        "the utterance after the failure was never transcribed"
    assert hub.of("queue")[-1]["pending"] == 0
    s.stop_worker()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_finish_does_not_wait_out_the_timeout_when_the_worker_is_gone():
    """A worker that cannot answer must not freeze Finish for DRAIN_TIMEOUT."""

    def suicide(pcm, sr=SR, participants=None):
        raise SystemExit("worker thread died")   # BaseException: escapes the guard

    vad = FakeVad(script=[(1, 0, SR)])
    s = make_session(vad=vad, transcriber=suicide)
    s.start_worker()
    s.feed(np.zeros(512, dtype=np.float32))

    assert wait_until(lambda: not s._worker.is_alive()), "worker did not die as set up"
    assert s.pending_count() == 1, "a dead worker leaves the count stranded"

    t0 = time.perf_counter()
    s._drain(timeout=30)
    waited = time.perf_counter() - t0
    assert waited < 5, f"drain polled a dead worker for {waited:.1f}s"


# ---------------------------------------------------------------------------
# Hub / replay
# ---------------------------------------------------------------------------

def test_replay_drops_pending_rows_that_were_already_answered():
    hub = live.Hub()
    hub.publish({"type": "pending", "utterance_id": 0, "start": 0.0, "end": 1.0})
    hub.publish({"type": "pending", "utterance_id": 1, "start": 1.0, "end": 2.0})
    hub.publish({"type": "segment", "utterance_id": 0, "text": "a", "start": 0.0,
                 "end": 1.0, "language": "ru"})
    hub.publish({"type": "queue", "pending": 1}, keep=False)

    replay = hub.replay()
    assert [e["type"] for e in replay] == ["pending", "segment"]
    assert replay[0]["utterance_id"] == 1          # only the unanswered one survives


def test_replay_drops_pending_rows_for_dropped_utterances():
    hub = live.Hub()
    hub.publish({"type": "pending", "utterance_id": 0, "start": 0.0, "end": 1.0})
    hub.publish({"type": "drop", "utterance_id": 0}, keep=False)
    assert hub.replay() == []


# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------

class ExplodingHub(RecordingHub):
    """Fails once on a chosen event type — i.e. *after* the transcriber returned.

    Models the class of bug the worker guard exists for: everything in _handle
    past the transcribe call (the append, the dict union, the publish) can raise.
    """

    def __init__(self, fail_on="segment"):
        super().__init__()
        self.fail_on = fail_on
        self.exploded = False

    def publish(self, event, keep=True):
        if event.get("type") == self.fail_on and not self.exploded:
            self.exploded = True
            self.events.append(event)
            self.at.append(time.perf_counter())
            raise RuntimeError("hub exploded")
        super().publish(event, keep=keep)


def timed_structure(sleep=0.2, protocol=None):
    """A structure() stand-in that records when it ran."""

    def call(text, model=None, language_hint=None):
        t0 = time.perf_counter()
        time.sleep(sleep)
        call.interval = (t0, time.perf_counter())
        return protocol if protocol is not None else {"summary": ["итог"]}

    call.interval = None
    return call


class FakeDiarizer:
    def __init__(self, available=True, turns=None, blocks_=None, fail=False, sleep=0.0):
        self._available = available
        self.sleep = sleep
        self.interval = None
        self._turns = turns or [{"start": 0.0, "end": 1.0, "speaker": 0}]
        self._blocks = blocks_ if blocks_ is not None else [
            {"speaker": "Speaker 1", "start": 0.0, "end": 1.0, "text": "привет"}
        ]
        self.fail = fail
        self.num_speakers = "unset"
        self.audio_seconds = None

    def is_available(self):
        return self._available

    def diarize(self, audio, sr=SR, num_speakers=None, threshold=0.5):
        t0 = time.perf_counter()
        try:
            if self.fail:
                raise RuntimeError("diarization exploded")
            time.sleep(self.sleep)
            self.num_speakers = num_speakers
            self.audio_seconds = np.asarray(audio).size / sr
            return self._turns
        finally:
            self.interval = (t0, time.perf_counter())

    def assign_speakers(self, words, turns):
        return self._blocks if (list(words) and list(turns)) else []


def finished_session(tmp_path, **kw):
    """A session with one transcribed utterance, ready for finish()."""
    vad = FakeVad(script=[(1, 0, SR)])
    kw.setdefault("vad", vad)
    kw.setdefault("transcriber", fake_transcriber([(0.0, 1.0, "привет", "ru")]))
    s = make_session(num_speakers=2, **kw)
    s.wav_dir = Path(tmp_path)
    s.feed(np.zeros(512, dtype=np.float32))
    s.feed(blocks(1.0))
    s.process_queued()
    return s


def test_finish_builds_a_process_audio_shaped_result(tmp_path):
    structured = {"summary": ["итог"], "decisions": [], "topics": [],
                  "open_questions": [], "risks": [], "action_items": []}
    calls = []

    def fake_structure(text, model=None, language_hint=None):
        calls.append((text, language_hint))
        return structured

    s = finished_session(tmp_path, structure_fn=fake_structure, diarizer=FakeDiarizer())
    result = s.finish("abc123")

    for key in ("transcript", "language", "languages", "audio_seconds", "segments",
                "protocol", "timing", "participants", "id", "filename",
                "speakers", "audio_path"):
        assert key in result, f"missing {key}"
    assert result["id"] == "abc123"
    assert result["transcript"] == "привет"
    assert result["protocol"] is structured
    assert calls == [("привет", "ru")]            # dominant language is the hint
    assert result["speakers"][0]["speaker"] == "Speaker 1"
    assert Path(result["audio_path"]).is_file()
    assert result["segments"][0]["start"] == 0.0


def test_finish_writes_the_whole_recording_as_16k_mono_pcm16(tmp_path):
    import wave

    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {},
                         diarizer=FakeDiarizer(available=False))
    result = s.finish("wavtest")
    with wave.open(result["audio_path"], "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        assert w.getnframes() == 512 + SR          # every raw sample, not just speech


def test_finish_says_finalizing_before_doing_any_of_the_work(tmp_path):
    """The page must be told Finish is working before the slow part starts."""
    d = FakeDiarizer(sleep=0.2)
    st = timed_structure(sleep=0.2)
    s = finished_session(tmp_path, structure_fn=st, diarizer=d)

    stamps: dict[str, float] = {}
    original_save = s._save_wav
    s._save_wav = lambda rid, audio: (stamps.setdefault("wav", time.perf_counter()),
                                      original_save(rid, audio))[1]

    s.finish("rid7")
    said = s.hub.time_of("status", state="finalizing")
    assert said < stamps["wav"], "finalizing came after the WAV was written"
    assert said < d.interval[0] and said < st.interval[0]


def test_finish_runs_diarization_and_the_llm_concurrently(tmp_path):
    """They need different inputs; overlapping them saves ~20-40 s on stage."""
    d = FakeDiarizer(sleep=0.2)
    st = timed_structure(sleep=0.2)
    s = finished_session(tmp_path, structure_fn=st, diarizer=d)
    t0 = time.perf_counter()
    s.finish("rid7")
    elapsed = time.perf_counter() - t0

    assert d.interval and st.interval
    assert d.interval[0] < st.interval[1] and st.interval[0] < d.interval[1], (
        f"diarization {d.interval} and the LLM {st.interval} did not overlap"
    )
    assert elapsed < 0.35, f"ran them back to back: {elapsed:.2f}s for 2x0.2s"


def test_finish_broadcasts_done_last_and_only_after_both_finished(tmp_path):
    d = FakeDiarizer(sleep=0.2)
    st = timed_structure(sleep=0.2)
    s = finished_session(tmp_path, structure_fn=st, diarizer=d)
    s.finish("rid7")

    assert s.hub.types()[-1] == "done"
    assert s.hub.events[-1]["id"] == "rid7"
    assert s.hub.at[-1] > max(d.interval[1], st.interval[1])


def test_done_is_published_only_after_the_result_is_stored(tmp_path):
    """The page fetches /result/{id} on `done`; a gap there is a 404 on stage."""
    stored: list[tuple[float, str]] = []
    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {},
                         diarizer=FakeDiarizer())
    result = s.finish("stored1", lambda r: stored.append((time.perf_counter(), r["id"])))

    assert stored and stored[0][1] == "stored1"
    assert stored[0][0] < s.hub.time_of("done")
    assert result["id"] == "stored1"


def test_finish_passes_the_requested_speaker_count_and_full_audio(tmp_path):
    d = FakeDiarizer()
    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {}, diarizer=d)
    s.finish("rid")
    assert d.num_speakers == 2
    assert d.audio_seconds == pytest.approx((512 + SR) / SR, abs=1e-6)


def test_finish_without_diarization_falls_back_to_the_plain_transcript(tmp_path):
    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {},
                         diarizer=FakeDiarizer(available=False))
    result = s.finish("rid")
    assert result.get("speakers") in (None, [])
    assert result["transcript"] == "привет"


def test_finish_survives_a_diarization_failure(tmp_path):
    """Diarization is a bonus; the protocol must still come out."""
    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {"summary": ["x"]},
                         diarizer=FakeDiarizer(fail=True))
    result = s.finish("rid")
    assert result["protocol"] == {"summary": ["x"]}
    assert result.get("speakers") in (None, [])


def test_finish_survives_malformed_speaker_blocks_and_still_stores(tmp_path):
    """F2: _label_segments must not be able to lose a finished live protocol.

    A speaker block missing "start" makes _label_segments raise KeyError; that
    must not stop finish() from storing the result and publishing `done`.
    """
    bad_blocks = [{"speaker": "Speaker 1", "end": 1.0, "text": "привет"}]  # no "start"
    d = FakeDiarizer(blocks_=bad_blocks)
    stored: list[dict] = []
    s = finished_session(tmp_path, structure_fn=lambda *a, **k: {"summary": ["x"]}, diarizer=d)
    result = s.finish("ridbad", lambda r: stored.append(r))

    assert stored and stored[0]["id"] == "ridbad", "on_stored was never called"
    assert s.hub.types()[-1] == "done"
    assert s.hub.events[-1]["id"] == "ridbad"
    assert result["id"] == "ridbad"
    assert result["transcript"] == "привет"


def test_finish_survives_an_llm_failure(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("ollama down")

    s = finished_session(tmp_path, structure_fn=boom, diarizer=FakeDiarizer())
    result = s.finish("rid")
    assert result["protocol"] == {} or result["protocol"].get("summary") == []
    assert result["transcript"] == "привет"


def test_finish_drains_in_flight_utterances_first(tmp_path):
    """Finish must wait for the ~8 s/utterance queue, not truncate it."""
    vad = FakeVad(script=[(1, 0, SR), (2, SR, SR)])
    s = make_session(vad=vad, structure_fn=lambda *a, **k: {},
                     diarizer=FakeDiarizer(available=False))
    s.wav_dir = Path(tmp_path)
    s.start_worker()
    s.feed(np.zeros(1024, dtype=np.float32))
    result = s.finish("rid")
    assert result["transcript"] == "привет привет"
    assert len(result["segments"]) == 2


def test_dominant_language_is_by_word_share():
    counts = live.dominant_language([
        {"language": "ru", "text": "раз два"},
        {"language": "kk", "text": "бір екі үш төрт"},
    ])
    assert counts[0] == "kk"
    assert counts[1]["kk"] > counts[1]["ru"]
    assert sum(counts[1].values()) == pytest.approx(1.0)


def test_dominant_language_of_nothing_is_unknown():
    assert live.dominant_language([])[0] == "unknown"


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


def _fake_start(self):
    """Stand in for opening PortAudio: mark the session live and nothing else."""
    self.state = "running"


def test_second_start_is_rejected(monkeypatch):
    monkeypatch.setattr(live.LiveSession, "start", _fake_start)
    live.start_session()
    with pytest.raises(live.LiveError, match="already running"):
        live.start_session()


def test_start_after_a_finished_session_opens_a_fresh_one(monkeypatch):
    monkeypatch.setattr(live.LiveSession, "start", _fake_start)
    first = live.start_session()
    first.state = "done"
    second = live.start_session()
    assert second is not first


def test_stop_without_a_session_is_an_error():
    with pytest.raises(live.LiveError, match="No live session"):
        live.stop_session()


# ---------------------------------------------------------------------------
# HTTP / WebSocket
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


def test_live_page_renders(client):
    # `/live` serves the built React dashboard when frontend/dist exists and
    # falls back to this inline page otherwise; the inline page is always at
    # `/legacy-live`, so that is where its markup is asserted.
    r = client.get("/legacy-live")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    for marker in ("/live/start", "/live/stop", "/live/ws", "Финиш", "offline"):
        assert marker in body
    assert "charset=\"utf-8\"" in body      # Cyrillic-safe
    live_page = client.get("/live")
    assert live_page.status_code == 200
    assert "text/html" in live_page.headers["content-type"]


@pytest.mark.skipif(not live.AUDIO_AVAILABLE, reason="sounddevice/PortAudio not installed")
def test_start_with_an_impossible_device_returns_400(client):
    r = client.post("/live/start", json={"device": 99999, "num_speakers": None})
    assert r.status_code == 400
    assert r.json()["detail"]
    assert live.current_session() is None


def test_stop_without_a_session_returns_409(client):
    r = client.post("/live/stop")
    assert r.status_code == 409


def test_mic_test_reports_a_portaudio_error_as_400(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("PortAudio: no default input device")

    monkeypatch.setattr(live, "record", boom)
    r = client.get("/mic-test")
    assert r.status_code == 400
    assert "PortAudio" in r.json()["detail"]


def test_mic_test_reports_level(client, monkeypatch):
    monkeypatch.setattr(
        live, "record",
        lambda seconds=2.0, device=None: (np.full(SR, 0.5, dtype=np.float32), "Fake Mic"),
    )
    r = client.get("/mic-test")
    assert r.status_code == 200
    body = r.json()
    assert body["device"] == "Fake Mic"
    assert body["ok"] is True
    assert body["rms"] == pytest.approx(0.5, abs=1e-3)
    assert body["peak"] == pytest.approx(0.5, abs=1e-3)


def test_mic_test_flags_a_silent_mic(client, monkeypatch):
    monkeypatch.setattr(
        live, "record",
        lambda seconds=2.0, device=None: (np.zeros(SR, dtype=np.float32), "Fake Mic"),
    )
    assert client.get("/mic-test").json()["ok"] is False


def test_websocket_replays_the_transcript_then_streams(client):
    live.HUB.publish({"type": "status", "state": "running"})
    live.HUB.publish({"type": "segment", "utterance_id": 0, "start": 0.0, "end": 1.0,
                      "text": "привет", "language": "ru"})
    live.HUB.publish({"type": "pending", "utterance_id": 1, "start": 1.0, "end": 2.0})

    with client.websocket_connect("/live/ws") as ws:
        replay = [ws.receive_json() for _ in range(3)]
        assert [e["type"] for e in replay] == ["status", "segment", "pending"]
        assert replay[1]["text"] == "привет"

        live.HUB.publish({"type": "segment", "utterance_id": 1, "start": 1.0,
                          "end": 2.0, "text": "сәлем", "language": "kk"})
        assert ws.receive_json()["text"] == "сәлем"


# ---------------------------------------------------------------------------
# Real VAD wiring (no mic: synthetic frames through the real ONNX model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not live.vad_available(),
                    reason="silero_vad.onnx missing — run api/scripts/get_vad_model.sh")
def test_real_vad_config_matches_the_agreed_settings():
    cfg = live.vad_config()
    assert cfg.silero_vad.threshold == pytest.approx(0.5)
    assert cfg.silero_vad.min_silence_duration == pytest.approx(0.6)
    assert cfg.silero_vad.min_speech_duration == pytest.approx(0.25)
    assert cfg.silero_vad.max_speech_duration == pytest.approx(12.0)
    assert cfg.sample_rate == SR


@pytest.mark.skipif(not live.vad_available(),
                    reason="silero_vad.onnx missing — run api/scripts/get_vad_model.sh")
def test_real_vad_finds_the_utterance_in_a_recorded_file():
    """A file, not a mic: the real VAD must segment real speech."""
    wav = Path(__file__).resolve().parent / "audio" / "ru_1.wav"
    if not wav.is_file():
        pytest.skip("tests/audio/ru_1.wav missing")
    from faster_whisper import decode_audio

    pcm = decode_audio(str(wav), sampling_rate=SR).astype(np.float32)
    utterances: list[tuple[float, float]] = []
    s = make_session(vad=live.make_vad())
    s.on_utterance = lambda u: utterances.append((u.start, u.end))
    for i in range(0, pcm.size - live.FRAME_SAMPLES, live.FRAME_SAMPLES * 8):
        s.feed(pcm[i:i + live.FRAME_SAMPLES * 8])

    assert utterances, "real VAD found no speech in ru_1.wav"
    assert utterances[0][0] > 0.0
    assert all(e - s_ <= 13.0 for s_, e in utterances), "max_speech_duration not honoured"
