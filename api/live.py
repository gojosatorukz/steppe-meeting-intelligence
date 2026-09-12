"""Live meeting transcription: mic -> VAD -> routed STT -> WebSocket -> protocol.

The server owns the microphone; the browser only displays. That keeps the whole
thing offline (no getUserMedia upload, no codec round-trip) and lets the same
`pipeline.transcribe_pcm` that serves the file path serve the live path.

Shape of a session:

    PortAudio callback ──> block queue ──> capture thread
                                             │  every 512-sample frame -> Silero VAD
                                             │  raw PCM -> session buffer (for diarization)
                                             ▼
                                        utterance FIFO ──> ONE worker thread
                                                              │ transcribe_pcm (~8 s/utterance)
                                                              ▼
                                                         Hub ──> WebSocket clients

The single worker is deliberate: transcription costs ~8 s per utterance on this
CPU regardless of utterance length, so a second worker would only thrash the
same cores. The page is kept honest instead — a `pending` row appears the
instant an utterance is cut, a `queue` event says how many are waiting, and the
`segment` events replace the pending row when the text lands.

Timestamps: `transcribe_pcm` returns times relative to the utterance it was
given. Everything leaving this module is relative to the start of the session.
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import pipeline

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512           # Silero's window at 16 kHz; the VAD requires exactly this
LEVEL_INTERVAL = 0.2          # ~5 Hz mic meter
MAX_HISTORY = 4000            # replay cap, so a long meeting cannot grow without bound

VAD_MODEL = Path(
    os.getenv("LIVE_VAD_MODEL")
    or (Path(__file__).resolve().parent / "models" / "vad" / "silero_vad.onnx")
)

# Tuned for meetings, and for the fact that transcription costs ~8 s per
# utterance whatever its length: fewer, longer utterances are cheaper.
VAD_THRESHOLD = float(os.getenv("LIVE_VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE = float(os.getenv("LIVE_VAD_MIN_SILENCE", "0.6"))
VAD_MIN_SPEECH = float(os.getenv("LIVE_VAD_MIN_SPEECH", "0.25"))
VAD_MAX_SPEECH = float(os.getenv("LIVE_VAD_MAX_SPEECH", "12.0"))
DIARIZE_THRESHOLD = float(os.getenv("LIVE_DIARIZE_THRESHOLD", "0.7"))
DRAIN_TIMEOUT = float(os.getenv("LIVE_DRAIN_TIMEOUT", "600"))

# Diarization is a bonus on this path: if the models or the module are missing,
# Finish still produces a protocol from the plain transcript.
try:  # pragma: no cover - exercised by whichever branch this machine has
    import diarize as _diarize
except Exception:  # pragma: no cover
    _diarize = None

try:  # pragma: no cover
    import sounddevice as _sd

    AUDIO_AVAILABLE = True
except Exception:  # pragma: no cover
    _sd = None
    AUDIO_AVAILABLE = False


class LiveError(RuntimeError):
    """Anything the caller should see as a 4xx: no device, no session, busy."""


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def vad_available() -> bool:
    """True when the Silero VAD model is on disk (get_vad_model.sh has run)."""
    return VAD_MODEL.is_file()


def diarization_available() -> bool:
    """True when diarize.py is importable and its models are on disk."""
    try:
        return bool(_diarize is not None and _diarize.is_available())
    except Exception:
        return False


def vad_config():
    """The streaming Silero VAD shipped inside sherpa-onnx (no torch)."""
    import sherpa_onnx

    return sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(VAD_MODEL),
            threshold=VAD_THRESHOLD,
            min_silence_duration=VAD_MIN_SILENCE,
            min_speech_duration=VAD_MIN_SPEECH,
            max_speech_duration=VAD_MAX_SPEECH,
            window_size=FRAME_SAMPLES,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
    )


def make_vad():
    import sherpa_onnx

    if not vad_available():
        raise LiveError(
            f"VAD model missing at {VAD_MODEL} — run: bash api/scripts/get_vad_model.sh"
        )
    return sherpa_onnx.VoiceActivityDetector(vad_config(), buffer_size_in_seconds=60)


# ---------------------------------------------------------------------------
# Event hub
# ---------------------------------------------------------------------------

class Hub:
    """Fan-out of session events to WebSocket clients, plus a replay log.

    Publishers are plain threads (audio callback, worker); subscribers live on
    the asyncio loop, so every hand-off goes through `call_soon_threadsafe`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: set[Any] = set()
        self._history: list[dict] = []
        self._loop: Any = None

    def set_loop(self, loop) -> None:
        with self._lock:
            self._loop = loop

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    def publish(self, event: dict, keep: bool = True) -> None:
        with self._lock:
            if event.get("type") in ("segment", "drop"):
                # The '…' row this utterance produced has been answered.
                uid = event.get("utterance_id")
                self._history = [
                    e for e in self._history
                    if not (e.get("type") == "pending" and e.get("utterance_id") == uid)
                ]
            if keep:
                self._history.append(event)
                if len(self._history) > MAX_HISTORY:
                    del self._history[: len(self._history) - MAX_HISTORY]
            clients = list(self._clients)
            loop = self._loop

        if loop is None:
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:  # loop already closed
                pass

    def replay(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def subscribe(self, q) -> None:
        with self._lock:
            self._clients.add(q)

    def unsubscribe(self, q) -> None:
        with self._lock:
            self._clients.discard(q)


HUB = Hub()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    id: int
    start: float              # seconds since session start
    end: float
    samples: np.ndarray
    cut_at: float = field(default_factory=time.perf_counter)
    settled: bool = False     # its slot in the pending count has been given back


def dominant_language(segments: Sequence[dict]) -> tuple[str, dict[str, float]]:
    """The meeting's language by word share, plus the full share map.

    Word share, not segment count: one long Kazakh answer should outweigh three
    Russian interjections when the LLM is told which language to write in.
    """
    counts: dict[str, int] = {}
    for seg in segments:
        lang = (seg.get("language") or "").strip()
        if not lang:
            continue
        counts[lang] = counts.get(lang, 0) + max(1, len((seg.get("text") or "").split()))
    total = sum(counts.values())
    if not total:
        return "unknown", {}
    shares = {lang: round(n / total, 3) for lang, n in counts.items()}
    return max(counts, key=lambda k: counts[k]), shares


class LiveSession:
    """One microphone recording, from Start to Finish."""

    def __init__(
        self,
        device: int | None = None,
        participants: Sequence[str] = (),
        num_speakers: int | None = None,
        protocol_language: str | None = pipeline.DEFAULT_PROTOCOL_LANGUAGE,
        hub: Hub | None = None,
        transcriber: Callable[..., Any] | None = None,
        vad: Any = None,
        structure_fn: Callable[..., dict] | None = None,
        diarizer: Any = "default",
        samplerate: int = SAMPLE_RATE,
    ) -> None:
        self.device = device
        self.participants = [p for p in participants if p]
        self.num_speakers = num_speakers
        self.protocol_language = protocol_language or pipeline.DEFAULT_PROTOCOL_LANGUAGE
        self.hub = hub if hub is not None else Hub()
        self.samplerate = samplerate
        self.wav_dir = Path(tempfile.gettempdir()) / "ai-meeting-live"

        self._transcriber = transcriber or pipeline.transcribe_pcm
        self._structure = structure_fn or pipeline.structure
        self._diarizer = _diarize if diarizer == "default" else diarizer
        self._vad = vad

        self.state = "idle"                      # idle|running|finalizing|done|error
        self.device_name = ""
        self.started_at = 0.0
        self.stopped_at = 0.0

        self._stream = None
        self._blocks: queue.Queue = queue.Queue()
        self._utterances: queue.Queue = queue.Queue()
        self._capture_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None

        self._carry = np.empty(0, dtype=np.float32)
        self._raw: list[np.ndarray] = []
        self._raw_lock = threading.Lock()
        self._segments: list[dict] = []
        self._seg_lock = threading.Lock()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._level_at = 0.0
        self._transcribe_seconds = 0.0
        self.latencies: list[float] = []

        # Test seam: called with each finalized Utterance before it is queued.
        self.on_utterance: Callable[[Utterance], None] | None = None

    # -- capture ----------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and start the capture + worker threads."""
        if not AUDIO_AVAILABLE:
            raise LiveError("sounddevice/PortAudio is not available on this machine")
        if self._vad is None:
            self._vad = make_vad()

        try:
            self.device_name = device_name(self.device)
            self._stream = _sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="float32",
                blocksize=FRAME_SAMPLES, device=self.device, callback=self._callback,
            )
            self._stream.start()
        except Exception as e:  # PortAudio errors are plain exceptions
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    logger.exception("failed to close the input stream after a failed start")
            self._stream = None
            self.state = "error"
            raise LiveError(f"Could not open the microphone: {e}") from e

        self.started_at = time.perf_counter()
        self.state = "running"
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="live-capture", daemon=True
        )
        self._capture_thread.start()
        self.start_worker()
        self.hub.publish({"type": "status", "state": "running",
                          "device": self.device_name,
                          "participants": self.participants})
        logger.info("live session started on %s", self.device_name)

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover
        """PortAudio thread: copy and hand off. Never blocks, never transcribes."""
        if status:
            logger.debug("audio status: %s", status)
        self._blocks.put(np.array(indata[:, 0], dtype=np.float32, copy=True))

    def _capture_loop(self) -> None:  # pragma: no cover - needs a real stream
        while True:
            block = self._blocks.get()
            if block is None:
                break
            try:
                self.feed(block)
            except Exception:
                logger.exception("live capture failed on a block")

    def feed(self, block: np.ndarray) -> None:
        """Consume one block of mono float32 audio (any length).

        Keeps every sample for diarization, emits a mic level, feeds the VAD in
        exact 512-sample frames and queues whatever utterances come out.
        """
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        with self._raw_lock:
            self._raw.append(block)
        self._emit_level(block)

        data = np.concatenate((self._carry, block)) if self._carry.size else block
        n = (data.size // FRAME_SAMPLES) * FRAME_SAMPLES
        for i in range(0, n, FRAME_SAMPLES):
            self._vad.accept_waveform(data[i:i + FRAME_SAMPLES])
            self._drain_vad()
        self._carry = data[n:].copy()

    def _emit_level(self, block: np.ndarray) -> None:
        now = time.perf_counter()
        if now - self._level_at < LEVEL_INTERVAL:
            return
        self._level_at = now
        self.hub.publish(
            {"type": "level",
             "rms": round(float(np.sqrt(np.mean(np.square(block)))), 4),
             "peak": round(float(np.max(np.abs(block))), 4),
             "seconds": round(self.audio_seconds(), 1)},
            keep=False,
        )

    def _drain_vad(self) -> None:
        while not self._vad.empty():
            seg = self._vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            start = float(seg.start) / self.samplerate
            self._vad.pop()
            if samples.size:
                self._enqueue(start, samples)

    def _enqueue(self, start: float, samples: np.ndarray) -> None:
        utt = Utterance(
            id=self._next_id, start=round(start, 2),
            end=round(start + samples.size / self.samplerate, 2), samples=samples,
        )
        self._next_id += 1
        if self.on_utterance:
            self.on_utterance(utt)
        with self._pending_lock:
            self._pending += 1
            depth = self._pending
        # The '…' row goes out before the 8 s of transcription, not after it.
        self.hub.publish({"type": "pending", "utterance_id": utt.id,
                          "start": utt.start, "end": utt.end})
        self.hub.publish({"type": "queue", "pending": depth}, keep=False)
        self._utterances.put(utt)

    # -- transcription worker ---------------------------------------------

    def start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="live-stt", daemon=True
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            utt = self._utterances.get()
            if utt is None:
                break
            # Nothing may end this loop but the sentinel. A worker that dies
            # takes every later utterance with it and leaves Finish polling a
            # count that can no longer fall.
            try:
                self._handle(utt)
            except Exception:
                logger.exception("live worker failed on utterance %s", utt.id)
                self._abandon(utt)

    def stop_worker(self, timeout: float = 30.0) -> None:
        if self._worker and self._worker.is_alive():
            self._utterances.put(None)
            self._worker.join(timeout)
        self._worker = None

    def process_queued(self) -> int:
        """Transcribe everything queued, in this thread. Used by tests and Finish."""
        done = 0
        while True:
            try:
                utt = self._utterances.get_nowait()
            except queue.Empty:
                return done
            if utt is None:
                continue
            try:
                self._handle(utt)
            except Exception:
                logger.exception("live worker failed on utterance %s", utt.id)
                self._abandon(utt)
            done += 1

    def _settle(self, utt: Utterance) -> int:
        """Give this utterance's slot in the pending count back, exactly once."""
        with self._pending_lock:
            if not utt.settled:
                utt.settled = True
                self._pending = max(0, self._pending - 1)
            return self._pending

    def _abandon(self, utt: Utterance) -> None:
        """Last resort after _handle blew up: never strand the queue or the page."""
        depth = self._settle(utt)
        for event in ({"type": "drop", "utterance_id": utt.id},
                      {"type": "queue", "pending": depth}):
            try:
                self.hub.publish(event, keep=False)
            except Exception:
                logger.exception("could not publish %s after a worker failure",
                                 event["type"])

    def _handle(self, utt: Utterance) -> None:
        t0 = time.perf_counter()
        segments: list[dict] = []
        try:
            tr = self._transcriber(utt.samples, self.samplerate,
                                   self.participants or None)
            segments = self._to_session_time(utt, tr)
        except Exception:
            logger.exception("transcription failed for utterance %s", utt.id)
        self._transcribe_seconds += time.perf_counter() - t0

        # The segments are recorded BEFORE the queue depth drops: Finish waits
        # on that depth, and must never snapshot the transcript mid-append.
        if segments:
            with self._seg_lock:
                self._segments.extend(segments)
            for seg in segments:
                self.hub.publish({k: v for k, v in seg.items() if k != "words"}
                                 | {"type": "segment"})
            latency = time.perf_counter() - utt.cut_at
            self.latencies.append(latency)
        else:
            # Nothing said (or the model failed): take the '…' row back down.
            latency = time.perf_counter() - utt.cut_at
            self.hub.publish({"type": "drop", "utterance_id": utt.id}, keep=False)

        depth = self._settle(utt)
        self.hub.publish({"type": "queue", "pending": depth}, keep=False)
        logger.info(
            "utterance %s [%.2f-%.2f] %.1fs audio -> %d segment(s) in %.2fs "
            "(latency VAD-end->broadcast %.2fs, %d still queued)",
            utt.id, utt.start, utt.end, utt.end - utt.start, len(segments),
            time.perf_counter() - t0, latency, depth,
        )

    def _to_session_time(self, utt: Utterance, tr: Any) -> list[dict]:
        """Offset one utterance's transcript into session time."""
        out: list[dict] = []
        for seg in getattr(tr, "segments", []) or []:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append({
                "utterance_id": utt.id,
                "start": round(utt.start + float(seg.start), 2),
                "end": round(utt.start + float(seg.end), 2),
                "text": text,
                "language": seg.language or getattr(tr, "language", "") or "",
                "speaker": None,
                "words": [
                    {"start": round(utt.start + float(w["start"]), 3),
                     "end": round(utt.start + float(w["end"]), 3),
                     "text": w["text"]}
                    for w in (seg.words or [])
                ],
            })
        return out

    # -- accessors ---------------------------------------------------------

    def raw_pcm(self) -> np.ndarray:
        with self._raw_lock:
            if not self._raw:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(self._raw)

    def audio_seconds(self) -> float:
        with self._raw_lock:
            return sum(b.size for b in self._raw) / self.samplerate

    def segments(self) -> list[dict]:
        with self._seg_lock:
            return sorted(self._segments, key=lambda s: (s["start"], s["utterance_id"]))

    def transcript_text(self) -> str:
        return " ".join(s["text"] for s in self.segments()).strip()

    def session_words(self) -> list[dict]:
        words: list[dict] = []
        for seg in self.segments():
            words.extend(seg["words"] or [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            ])
        return words

    def pending_count(self) -> int:
        with self._pending_lock:
            return self._pending

    # -- finish ------------------------------------------------------------

    def stop_capture(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("closing the input stream failed")
            self._stream = None
        if self._capture_thread and self._capture_thread.is_alive():
            self._blocks.put(None)
            self._capture_thread.join(5)
        self._capture_thread = None
        if self._vad is not None:
            try:
                self._vad.flush()
                self._drain_vad()        # whatever speech was still open
            except Exception:
                logger.exception("flushing the VAD failed")
        self.stopped_at = time.perf_counter()

    def _drain(self, timeout: float = DRAIN_TIMEOUT) -> None:
        """Wait for in-flight utterances — they are the last few seconds said.

        The liveness of the worker is re-checked every pass, not just once: a
        worker that is gone will never move the count again, and waiting out
        the full timeout would freeze Finish for ten minutes on stage.
        """
        worker = self._worker
        deadline = time.perf_counter() + timeout
        while (worker is not None and worker.is_alive()
               and self.pending_count() > 0 and time.perf_counter() < deadline):
            time.sleep(0.1)

        if worker is None or not worker.is_alive():
            self.process_queued()      # nobody left to do it; do it here
        if self.pending_count():
            logger.warning(
                "finish: %d utterance(s) unaccounted for (worker alive=%s) — "
                "finishing with what we have",
                self.pending_count(), bool(worker and worker.is_alive()),
            )

    def finish(self, rid: str, on_stored: Callable[[dict], None] | None = None
               ) -> dict[str, Any]:
        """Stop, drain, save the WAV, diarize + structure, return the result.

        `on_stored` is called with the finished result *before* the `done` event
        goes out: the page reacts to `done` by fetching /result/{id}, so the
        result has to be in RESULTS by the time that event is on the wire.
        """
        t0 = time.perf_counter()
        self.state = "finalizing"
        self.hub.publish({"type": "status", "state": "finalizing",
                          "pending": self.pending_count()})
        self.stop_capture()
        self._drain()
        self.stop_worker()

        audio = self.raw_pcm()
        wav_path = self._save_wav(rid, audio)
        segments = self.segments()
        text = self.transcript_text()
        language, shares = dominant_language(segments)

        # Independent work: diarization needs the audio, the LLM needs the text.
        # Running them together saves ~20-40 s in front of an audience.
        timings: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-finish") as pool:
            fut_speakers = pool.submit(self._diarize_blocks, audio, timings)
            fut_protocol = pool.submit(self._structure_text, text, language, timings)
            speakers = fut_speakers.result()
            protocol = fut_protocol.result()

        if speakers:
            try:
                _label_segments(segments, speakers)
            except Exception:
                logger.exception("speaker labelling failed; segments stay unlabelled")

        total = time.perf_counter() - t0
        seconds = round(audio.size / self.samplerate, 1)
        result: dict[str, Any] = {
            "id": rid,
            "filename": "(live)",
            "transcript": text,
            "language": language,
            "languages": shares,
            "protocol_language": self.protocol_language,
            "participants": list(self.participants),
            "audio_seconds": seconds,
            "segments": segments,
            "protocol": protocol,
            "audio_path": str(wav_path),
            "timing": {
                "transcribe_sec": round(self._transcribe_seconds, 2),
                "diarize_sec": round(timings.get("diarize", 0.0), 2),
                "llm_sec": round(timings.get("llm", 0.0), 2),
                "finalize_sec": round(total, 2),
                "total_sec": round(total, 2),
                "realtime_factor": round(seconds / total, 2) if total > 0 and seconds else None,
                "utterances": self._next_id,
                "mean_latency_sec": (round(sum(self.latencies) / len(self.latencies), 2)
                                     if self.latencies else None),
            },
        }
        if speakers:
            result["speakers"] = speakers

        self.state = "done"
        if on_stored is not None:
            on_stored(result)
        self.hub.publish({"type": "done", "id": rid})
        logger.info("live session finished: %.1fs audio, %d segments, %d speakers, %.1fs",
                    seconds, len(segments), len({s["speaker"] for s in speakers}), total)
        return result

    def _save_wav(self, rid: str, audio: np.ndarray) -> Path:
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        path = self.wav_dir / f"live_{rid}.wav"
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(pcm16.tobytes())
        return path

    def _diarize_blocks(self, audio: np.ndarray, timings: dict) -> list[dict]:
        d = self._diarizer
        t0 = time.perf_counter()
        try:
            if d is None or not d.is_available() or audio.size < self.samplerate:
                return []
            words = self.session_words()
            if not words:
                return []
            turns = d.diarize(audio, self.samplerate, num_speakers=self.num_speakers,
                              threshold=DIARIZE_THRESHOLD)
            return d.assign_speakers(words, turns)
        except Exception:
            logger.exception("diarization failed; falling back to the plain transcript")
            return []
        finally:
            timings["diarize"] = time.perf_counter() - t0

    def _structure_text(self, text: str, language: str, timings: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if not text.strip():
                return {}
            hint = pipeline.resolve_protocol_language(self.protocol_language, language)
            return self._structure(text, None, hint)
        except Exception:
            logger.exception("structuring failed; returning an empty protocol")
            return {}
        finally:
            timings["llm"] = time.perf_counter() - t0


def _label_segments(segments: list[dict], blocks: Sequence[dict]) -> None:
    """Stamp each transcript segment with the speaker it overlaps most."""
    for seg in segments:
        best, best_overlap = None, 0.0
        for b in blocks:
            overlap = min(seg["end"], b["end"]) - max(seg["start"], b["start"])
            if overlap > best_overlap:
                best, best_overlap = b["speaker"], overlap
        seg["speaker"] = best


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def device_name(device: int | None = None) -> str:
    if not AUDIO_AVAILABLE:
        return "unavailable"
    try:
        if device is None:
            device = _sd.default.device[0]
        info = _sd.query_devices(device)
        return str(info.get("name", device))
    except Exception:
        return str(device)


def record(seconds: float = 2.0, device: int | None = None) -> tuple[np.ndarray, str]:
    """Block for `seconds` and return (mono float32 samples, device name)."""
    if not AUDIO_AVAILABLE:
        raise LiveError("sounddevice/PortAudio is not available on this machine")
    frames = int(seconds * SAMPLE_RATE)
    data = _sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    _sd.wait()
    return np.asarray(data, dtype=np.float32).reshape(-1), device_name(device)


def mic_test(seconds: float = 2.0, device: int | None = None) -> dict[str, Any]:
    try:
        pcm, name = record(seconds, device)
    except Exception as e:
        raise LiveError(str(e)) from e
    rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    return {"device": name, "rms": round(rms, 4), "peak": round(peak, 4),
            "ok": rms > 0.005}


# ---------------------------------------------------------------------------
# One session at a time
# ---------------------------------------------------------------------------

_current: LiveSession | None = None
_registry_lock = threading.Lock()


def current_session() -> LiveSession | None:
    with _registry_lock:
        return _current


def start_session(device: int | None = None, participants: Sequence[str] = (),
                  num_speakers: int | None = None,
                  protocol_language: str | None = pipeline.DEFAULT_PROTOCOL_LANGUAGE,
                  ) -> LiveSession:
    """Start the one live session. Raises LiveError if one is already running."""
    global _current
    with _registry_lock:
        if _current is not None and _current.state in ("running", "finalizing"):
            raise LiveError("A live session is already running — finish it first")
        session = LiveSession(device=device, participants=participants,
                              num_speakers=num_speakers,
                              protocol_language=protocol_language, hub=HUB)
        HUB.clear()
        session.start()
        _current = session
        return session


def stop_session() -> LiveSession:
    """Take the running session out of the registry so Finish can run unlocked."""
    global _current
    with _registry_lock:
        session = _current
        if session is None or session.state != "running":
            raise LiveError("No live session is running")
        session.state = "finalizing"
        return session


def reset_for_tests() -> None:
    global _current
    with _registry_lock:
        _current = None
    HUB.clear()


def new_result_id() -> str:
    return uuid.uuid4().hex[:12]
