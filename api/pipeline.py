"""Local meeting-protocol pipeline: audio -> transcript -> structured protocol.

Everything here runs on this machine. No network calls leave the box —
that is a hard requirement of the task (20 points), not a preference.
"""

from __future__ import annotations

# MUST stay above every other import. The turbo model is loaded by hub id, so
# huggingface_hub answers the first transcription with a revision check against
# huggingface.co — which, with the wifi off, hangs rather than failing fast. The
# hub reads this variable when it is imported, and `faster_whisper` below is what
# imports it. `main` gets the same thing (plus the socket guard) from
# `offline_guard`; this line is for the tests and scripts that import `pipeline`
# directly. setdefault: an explicit HF_HUB_OFFLINE=0 still wins.
import os  # isort: skip

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# MUST come before faster_whisper/ctranslate2: importing `gpu` is what pulls
# libcublas/libcudnn into the process, and CTranslate2 can only find them if
# they are already there (see gpu.py). Nothing below this line may be moved
# above it.
import gpu  # isort: skip
import numpy as np
import ollama
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.audio import pad_or_trim
from faster_whisper.vad import VadOptions, get_speech_timestamps

logger = logging.getLogger(__name__)

# Preference order: env override first, then locally pulled models, most
# capable/likely-available first. qwen2.5:7b-instruct stays last as a legacy
# fallback for machines that still have it pulled.
LLM_MODEL_CANDIDATES = ["qwen3.5:4b", "qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:7b-instruct"]
LLM_MODEL = os.getenv("LLM_MODEL") or LLM_MODEL_CANDIDATES[0]

CHUNK_CHARS = 9000           # transcript slice per LLM pass
CHUNK_OVERLAP = 400

# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
#
# One recording here contains Kazakh, Russian and English — often in adjacent
# sentences. A single Whisper pass over the whole file detects one language for
# everything (measured: ru 0.976 / kk 0.008 on team_mixed_35.m4a) and then
# hallucinates Russian over the Kazakh turns and translates the English.
#
# So: VAD first, then language ID *per speech window*, then the right model for
# that window.
#   * turbo — large-v3-turbo, handles RU/EN and is the language-ID head for
#     every window (the Kazakh fine-tune's own LID head is useless: it scores
#     kk at 0.18 on perfect Kazakh).
#   * kk — a Kazakh fine-tune of that same turbo, converted to CTranslate2. Its
#     model card is explicit that it does NOT retain RU/EN, so it may only ever
#     see windows that language ID routed to Kazakh.

SAMPLE_RATE = 16000

TURBO_MODEL_ID = os.getenv("STT_TURBO_MODEL") or "deepdml/faster-whisper-large-v3-turbo-ct2"
KK_MODEL_DIR = Path(
    os.getenv("STT_KK_MODEL") or (Path(__file__).resolve().parent / "models" / "kk-turbo-ct2")
)
WHISPER_SIZE = TURBO_MODEL_ID          # what /health reports as the STT model

# Device policy. STT_DEVICE=auto|cuda|cpu (default auto).
#
# The GPU is worth ~8x here: a speech window that costs ~8 s of CPU costs ~1 s on
# the 3060, which is the difference between live captions that lag a sentence
# behind and ones that arrive as the speaker stops. The CPU path stays fully
# intact and is one env var away, because it is the fallback if the card is busy,
# missing, or the CUDA libraries cannot be loaded on the machine we demo on.
STT_DEVICE_REQUEST = (os.getenv("STT_DEVICE") or "auto").strip().lower() or "auto"
STT_CPU_COMPUTE_TYPE = "int8"
# int8_float16 rather than float16: the two models together then cost ~1 GiB of
# the 5.7 GiB card instead of ~2.5 GiB, and the 3.4 GiB LLM has to fit beside
# them in RAM while the handoff is in flight.
STT_CUDA_COMPUTE_TYPE = "int8_float16"
STT_CPU_THREADS = 8
# A CUDA build that fails is usually losing a race for VRAM with something that
# is about to release it (observed: the LLM still tearing down). One retry after
# a short pause fixes that; two failures mean the card really has no room.
CUDA_LOAD_RETRY_DELAY = 1.0

# What a *new* model load will attempt. Not latched to "cpu" by a single failed
# build: a fallback is scoped to the model that failed (see `_new_model`), so one
# transient OOM cannot silently drag the whole service onto the CPU for good.
_stt_device: str = "cpu"
# label -> the device that model is actually resident on. This, not the request
# above, is what /health should be told about.
_model_devices: dict[str, str] = {}


def _resolve_stt_device() -> str:
    """Decide once, at import, whether STT runs on the GPU.

    At import and not lazily: `load_turbo()` has to know whether to take the GPU
    lock before it allocates VRAM, and the first load can happen on any thread.
    """
    if STT_DEVICE_REQUEST == "cpu":
        return "cpu"
    ready, reason = gpu.cuda_ready()
    if ready:
        logger.info("STT on the GPU (%s, %s)", reason, STT_CUDA_COMPUTE_TYPE)
        return "cuda"
    # An explicit STT_DEVICE=cuda that cannot be honoured is a louder problem
    # than `auto` quietly picking the path it was always allowed to pick.
    log = logger.error if STT_DEVICE_REQUEST == "cuda" else logger.info
    log("STT falling back to CPU: %s", reason)
    return "cpu"


def stt_devices() -> dict[str, str]:
    """Per-model resident device, e.g. `{"turbo": "cuda", "kk": "cpu"}`.

    The honest answer for /health. A single summary string cannot describe the
    state where one model made it onto the card and the other did not — and that
    is exactly the state worth seeing, because the model on the CPU is paying the
    ~8 s per window this task exists to delete.
    """
    return dict(_model_devices)


def stt_compute_types() -> dict[str, str]:
    """Per-model compute type, matching `stt_devices()`."""
    return {
        label: (STT_CUDA_COMPUTE_TYPE if device == "cuda" else STT_CPU_COMPUTE_TYPE)
        for label, device in _model_devices.items()
    }


def _summarise(values: dict[str, str], pending: str) -> str:
    distinct = set(values.values())
    if not distinct:
        return pending                      # nothing loaded yet
    return distinct.pop() if len(distinct) == 1 else "mixed"


def stt_device() -> str:
    """One-word summary of where STT is running: cuda | cpu | mixed | degraded.

    "degraded" wins over everything: if the models are registered on the card but
    could not be loaded back after an LLM phase, claiming "cuda" would be a lie
    that /health tells right up until the next transcription raises.
    """
    if gpu.degraded_reason():
        return "degraded"
    return _summarise(_model_devices, _stt_device)


def stt_compute_type() -> str:
    """Summary compute type, or what the next load will use if none is loaded."""
    return _summarise(
        stt_compute_types(),
        STT_CUDA_COMPUTE_TYPE if _stt_device == "cuda" else STT_CPU_COMPUTE_TYPE,
    )


_stt_device = _resolve_stt_device()
gpu.set_enabled(_stt_device == "cuda")

# Language ID is an argmax restricted to these three: unrestricted, Whisper
# happily routes Kazakh to Tatar/Bashkir/Uzbek, which then decodes as noise.
ROUTED_LANGUAGES = ("kk", "ru", "en")
FALLBACK_LANGUAGE = "ru"               # dominant demo language, used when LID is unsure
MIN_LANGUAGE_PROB = 0.3
# ...unless the leader still clearly beats the other two. Measured on real
# Kazakh speech: kk 0.254 / ru 0.113 / en 0.107 — weak in absolute terms, but a
# 2.2x lead, and forcing that window to Russian is exactly the bug we fix.
LANGUAGE_LEAD_RATIO = 2.0

VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200
# If the VAD hears nothing at all we only second-guess it on short input. Past
# this length, "no speech" is taken at its word and the transcript comes back
# empty instead of spending minutes decoding silence.
VAD_FALLBACK_MAX_SEC = 60.0
WINDOW_MAX_SEC = 30.0                  # Whisper's receptive field
WINDOW_GAP_SEC = 1.0                   # a pause this long usually *is* a speaker turn

# Language ID runs on the first LID_SECONDS of a group, as a mel padded to
# LID_MEL_FRAMES. Measured on this CPU: a 30 s mel costs 6.1 s per encoder pass,
# a 10 s mel costs 1.1 s — for an identical posterior (kk 0.983 vs 0.987). That
# 5.6x is what makes language ID per speech chunk affordable at all.
LID_SECONDS = 10.0
LID_MEL_FRAMES = 1000                  # 100 mel frames per second of audio
# Language ID needs evidence, and short speech is where it is confidently wrong:
# measured, 1.2 s of Kazakh scores en 0.744 and 3.9 s of Kazakh scores ru 0.813,
# while every chunk over 5 s in the test set is right with p > 0.97. So chunks
# shorter than this are glued to a neighbour before we ask. The ceiling on this
# value is real too: at 6 s a 5.7 s Kazakh turn gets absorbed into the English
# turn 0.2 s before it, which is the exact bug this routing exists to fix.
MIN_LID_SEC = 5.0

NO_SPEECH_MAX = 0.6
HALLUCINATION_PHRASES = (
    "субтитры",
    "продолжение следует",
    "редактор субтитров",
    "amara.org",
    "thank you for watching",
    "подписывайтесь",
)

TRANSCRIBE_KWARGS: dict[str, Any] = {
    "beam_size": 5,
    "word_timestamps": True,
    "condition_on_previous_text": False,   # each window stands alone; stops drift
    "vad_filter": False,                   # windows are already VAD-gated
    # A *ladder*, not a scalar. With a single temperature faster-whisper's
    # repetition guards (compression_ratio_threshold / log_prob_threshold) have
    # nowhere to retry to, so a degenerate window is returned as-is — which is
    # exactly how a live Kazakh turn came back as "Бекзат" fifty times.
    "temperature": [0.0, 0.2, 0.4],
    "compression_ratio_threshold": 2.4,    # default, pinned: it drives the retry
}

# The Kazakh fine-tune is a small, lowercase/unpunctuated model and it loops on
# a repeated token far more readily than turbo does. These two make the loop
# expensive inside the beam instead of only catching it afterwards.
KK_TRANSCRIBE_KWARGS: dict[str, Any] = {
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.15,
}

# faster-whisper >= 1.0 takes `hotwords`; older builds do not. Both the hotword
# and the initial prompt survive `condition_on_previous_text=False` for the
# first 30 s window, which is all any window of ours is.
_HOTWORDS_SUPPORTED = "hotwords" in inspect.signature(WhisperModel.transcribe).parameters

_turbo: WhisperModel | None = None
_kk: WhisperModel | None = None
_kk_missing_warned = False

# FastAPI runs the sync /process endpoint in a threadpool, so two uploads really
# can land in here at the same moment. Without this lock both would see `None`,
# both would load, and one 1.6 GB int8 model would be left resident and
# orphaned. One lock for both models on purpose: loading them concurrently would
# only have them fight over the same 8 CPU threads.
#
# Lock order: gpu.phase_lock() FIRST, then _model_lock — loading a model
# allocates VRAM, so it may not happen while the LLM holds the card.
_model_lock = threading.Lock()


def _new_model(label: str, name_or_path: str) -> WhisperModel:
    """Build one WhisperModel on the resolved device, falling back to CPU.

    The fallback is scoped to *this* model. A CUDA build that fails is retried
    once after `CUDA_LOAD_RETRY_DELAY` — the realistic cause is losing a race for
    VRAM with something that is about to free it — and only then does this model
    go to the CPU. What it must NOT do is latch the whole service onto the CPU,
    which is how one transient OOM used to cost every later load as well.
    """
    if _stt_device == "cuda":
        for attempt in (1, 2):
            try:
                model = WhisperModel(
                    name_or_path, device="cuda", compute_type=STT_CUDA_COMPUTE_TYPE
                )
            except Exception as exc:
                if attempt == 1:
                    logger.warning(
                        "STT model %s did not load on CUDA (%s); retrying in %.1fs",
                        label, exc, CUDA_LOAD_RETRY_DELAY,
                    )
                    time.sleep(CUDA_LOAD_RETRY_DELAY)
                    continue
                logger.error(
                    "STT model %s stays on the CPU: CUDA load failed twice (%s). "
                    "Other models are unaffected.", label, exc,
                )
            else:
                # Registered so the LLM phase can move it off the card and back.
                gpu.register_stt_model(label, model)
                _model_devices[label] = "cuda"
                return model
    model = WhisperModel(
        name_or_path,
        device="cpu",
        compute_type=STT_CPU_COMPUTE_TYPE,
        cpu_threads=STT_CPU_THREADS,
    )
    gpu.unregister_stt_model(label)
    _model_devices[label] = "cpu"
    return model


def load_turbo() -> WhisperModel:
    """RU/EN model and language-ID head. Loaded once; ~3 s we pay only at startup."""
    global _turbo
    if _turbo is None:                              # fast path: no lock once loaded
        with gpu.phase_lock(), _model_lock:         # GPU before _model_lock, always
            if _turbo is None:                      # another thread may have won the race
                t0 = time.perf_counter()
                model = _new_model("turbo", TURBO_MODEL_ID)
                logger.info(
                    "loaded turbo STT model %s on %s in %.1fs",
                    TURBO_MODEL_ID, _model_devices.get("turbo"), time.perf_counter() - t0,
                )
                _turbo = model                      # publish only once fully built
    return _turbo


def is_kk_available() -> bool:
    """Whether the Kazakh fine-tune is on disk (surfaced by /health)."""
    return (KK_MODEL_DIR / "model.bin").is_file()


def load_kk() -> WhisperModel | None:
    """Kazakh model, or None — in which case everything falls back to turbo."""
    global _kk, _kk_missing_warned
    if not is_kk_available():
        with _model_lock:
            if not _kk_missing_warned:
                logger.warning(
                    "Kazakh STT model not found at %s — routing every window to %s. "
                    "Kazakh quality will be poor until the model is in place.",
                    KK_MODEL_DIR, TURBO_MODEL_ID,
                )
                _kk_missing_warned = True
        return None
    if _kk is None:
        with gpu.phase_lock(), _model_lock:         # GPU before _model_lock, always
            if _kk is None:
                t0 = time.perf_counter()
                model = _new_model("kk", str(KK_MODEL_DIR))
                logger.info(
                    "loaded Kazakh STT model %s on %s in %.1fs",
                    KK_MODEL_DIR, _model_devices.get("kk"), time.perf_counter() - t0,
                )
                _kk = model
    return _kk


def warm_up() -> str:
    """Load both STT models on the resolved device, before the first request.

    Called from the app lifespan: the first upload (or the first live utterance,
    where it is most visible) should not be the one paying the model load.
    Idempotent — the loaders publish once, so calling this twice is free.

    Returns the device STT actually ended up on.

    Never raises: a model that will not load is a problem for the first request
    to report, not a reason for the server to refuse to start.
    """
    t0 = time.perf_counter()
    try:
        load_turbo()
        load_kk()
    except Exception as exc:                        # pragma: no cover - env dependent
        logger.error("STT warm-up failed (%s); models will load on first use", exc)
    else:
        logger.info(
            "STT warm-up on %s finished in %.1fs", stt_device(), time.perf_counter() - t0
        )
    return stt_device()


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class Transcript:
    text: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    seconds: float = 0.0
    languages: dict[str, float] = field(default_factory=dict)


def constrained_language(all_probs: Sequence[tuple[str, float]]) -> tuple[str, float]:
    """Argmax over ROUTED_LANGUAGES only, from Whisper's full language posterior.

    Whisper's unrestricted top-1 on Kazakh is frequently Tatar or Bashkir; those
    tokens decode as gibberish. Restricting the argmax to the three languages
    this meeting can actually contain is what makes routing reliable.
    """
    probs = {lang: float(prob) for lang, prob in (all_probs or [])}
    ranked = sorted(
        ((probs.get(lang, 0.0), lang) for lang in ROUTED_LANGUAGES), reverse=True
    )
    (best_prob, best), (runner_up, _) = ranked[0], ranked[1]
    unsure = best_prob <= 0.0 or (
        best_prob < MIN_LANGUAGE_PROB and best_prob < LANGUAGE_LEAD_RATIO * runner_up
    )
    if unsure:
        logger.debug(
            "weak language ID (best %s=%.3f, runner-up %.3f) — falling back to %s",
            best, best_prob, runner_up, FALLBACK_LANGUAGE,
        )
        return FALLBACK_LANGUAGE, best_prob
    return best, best_prob


def _speech_chunks(pcm: np.ndarray, sampling_rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """Speech spans from the VAD, in samples. The routing atom."""
    options = VadOptions(
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        # Split over-long speech at its deepest silence rather than mid-word;
        # the padding is added on both sides, so leave room for it.
        max_speech_duration_s=WINDOW_MAX_SEC - 2 * VAD_SPEECH_PAD_MS / 1000,
    )
    chunks = get_speech_timestamps(pcm, options, sampling_rate=sampling_rate)
    if not chunks:
        seconds = len(pcm) / sampling_rate
        if seconds > VAD_FALLBACK_MAX_SEC:
            # Decoding a long silent upload would mean minutes of ASR producing
            # nothing, which from the outside is indistinguishable from a hang.
            logger.warning(
                "VAD found no speech in %.1fs of audio — returning an empty transcript "
                "rather than decoding silence", seconds,
            )
            return []
        # Short input: the VAD may simply have been too strict (quiet mic, one
        # brief utterance), and one window is cheap enough to just try.
        logger.warning("VAD found no speech; falling back to the whole %.1fs input", seconds)
        chunks = [{"start": 0, "end": len(pcm)}]

    spans = [(int(c["start"]), int(c["end"])) for c in chunks]

    # Guard for the fallback above and for any pathologically long chunk:
    # nothing downstream may exceed Whisper's 30 s window.
    max_samples = int(WINDOW_MAX_SEC * sampling_rate)
    bounded: list[tuple[int, int]] = []
    for start, end in spans:
        span = end - start
        if span <= max_samples:
            bounded.append((start, end))
            continue
        pieces = -(-span // max_samples)                    # ceil
        step = -(-span // pieces)                           # even pieces, each <= 30 s
        bounded.extend(
            (offset, min(offset + step, end)) for offset in range(start, end, step)
        )
    return bounded


def _lid_groups(
    chunks: list[tuple[int, int]], sampling_rate: int = SAMPLE_RATE
) -> list[tuple[int, int]]:
    """Glue chunks too short to language-ID onto the neighbour they sit closest to.

    Language ID on 1.2 s of Kazakh comes back "en, 0.744" — confidently wrong.
    A chunk that short is nearly always a continuation of the turn next to it,
    so we attach it (nearest gap first) rather than asking about it. We never
    glue across a WINDOW_GAP_SEC silence: that gap is the one piece of evidence
    we have that the speaker changed.
    """
    groups = list(chunks)
    gap_samples = int(WINDOW_GAP_SEC * sampling_rate)
    max_samples = int(WINDOW_MAX_SEC * sampling_rate)
    min_samples = int(MIN_LID_SEC * sampling_rate)

    def neighbour(index: int) -> int | None:
        """Index to merge `index` into: the legal side with the smaller gap."""
        options: list[tuple[int, int]] = []
        if index > 0:
            gap = groups[index][0] - groups[index - 1][1]
            if gap < gap_samples and groups[index][1] - groups[index - 1][0] <= max_samples:
                options.append((gap, index - 1))
        if index < len(groups) - 1:
            gap = groups[index + 1][0] - groups[index][1]
            if gap < gap_samples and groups[index + 1][1] - groups[index][0] <= max_samples:
                options.append((gap, index + 1))
        return min(options)[1] if options else None

    while len(groups) > 1:
        short = sorted(
            (i for i, (s, e) in enumerate(groups) if e - s < min_samples),
            key=lambda i: groups[i][1] - groups[i][0],
        )
        for index in short:
            other = neighbour(index)
            if other is None:
                continue
            low, high = min(index, other), max(index, other)
            groups[low : high + 1] = [(groups[low][0], groups[high][1])]
            break
        else:
            break                                   # nothing left that can be merged
    return groups


def _merge_by_language(
    routed: Sequence[tuple[int, int, str]], sampling_rate: int = SAMPLE_RATE
) -> list[tuple[int, int, str]]:
    """Fuse neighbouring groups that agree on the language into one window.

    Fewer, longer windows means fewer encoder passes and more context for the
    decoder — but only where it is free: a window still never spans two
    languages, a silence of WINDOW_GAP_SEC, or 30 s of audio.
    """
    gap_samples = int(WINDOW_GAP_SEC * sampling_rate)
    max_samples = int(WINDOW_MAX_SEC * sampling_rate)

    windows: list[tuple[int, int, str]] = []
    for start, end, language in routed:
        if windows:
            w_start, w_end, w_language = windows[-1]
            if (
                language == w_language
                and start - w_end < gap_samples
                and end - w_start <= max_samples
            ):
                windows[-1] = (w_start, end, w_language)
                continue
        windows.append((start, end, language))
    return windows


def _detect_language(model: WhisperModel, pcm: np.ndarray) -> tuple[str, float]:
    """Constrained language ID for one span, using the turbo model's LID head.

    Whisper always encodes a fixed-size mel, so the cost is per *call*, not per
    second: we hand it a 10 s mel rather than the standard 30 s one, which on
    this CPU is 1.1 s instead of 6.1 s for the same answer. If that fast path
    ever breaks on a future faster-whisper, fall back to the public API.

    This is an encoder pass, so on the GPU path it holds the GPU lock: the
    weights it needs are exactly the ones the LLM phase moves off the card.
    """
    with gpu.phase_lock():
        try:
            features = model.feature_extractor(pcm[: int(LID_SECONDS * SAMPLE_RATE)])
            encoder_output = model.encode(pad_or_trim(features, LID_MEL_FRAMES))
            results = model.model.detect_language(encoder_output)[0]
            all_probs = [(token[2:-2], prob) for token, prob in results]
        except Exception as exc:                    # pragma: no cover - API drift
            logger.warning("fast language ID failed (%s); using the full 30 s path", exc)
            _, _, all_probs = model.detect_language(pcm)
    return constrained_language(all_probs)


def _as_float32(pcm: Sequence[float] | np.ndarray) -> np.ndarray:
    """Whisper wants mono float32 in [-1, 1]; callers may hand us int16 or a list."""
    audio = np.asarray(pcm)
    if audio.ndim > 1:
        audio = audio.mean(axis=tuple(range(1, audio.ndim)))
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Cheap linear resample to 16 kHz — the live path may arrive at any rate."""
    if sr == SAMPLE_RATE or pcm.size == 0:
        return pcm
    duration = pcm.size / sr
    target = np.linspace(0.0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    source = np.arange(pcm.size, dtype=np.float64) / sr
    return np.interp(target, source, pcm).astype(np.float32)


def _prompt_kwargs(participants: Sequence[str] | None) -> dict[str, Any]:
    """Names of the people in the room, fed to the decoder.

    Without this the demo recording gives "Пикзат" for "Бекзат" — and a
    misspelt name becomes a wrong action-item owner in the protocol.
    """
    if not participants:
        return {}
    names = ", ".join(p.strip() for p in participants if p and p.strip())
    if not names:
        return {}
    kwargs: dict[str, Any] = {"initial_prompt": names}
    if _HOTWORDS_SUPPORTED:
        kwargs["hotwords"] = names
    return kwargs


def _words_of(raw_segment: Any, offset: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for word in getattr(raw_segment, "words", None) or []:
        text = (getattr(word, "word", "") or "").strip()
        if not text:
            continue
        words.append({
            "start": round(float(word.start) + offset, 3),
            "end": round(float(word.end) + offset, 3),
            "text": text,
        })
    if not words:
        # Word timestamps can come back empty (very short segment, no alignment);
        # diarization still needs something with bounds to attach a speaker to.
        text = (getattr(raw_segment, "text", "") or "").strip()
        if text:
            words = [{
                "start": round(float(raw_segment.start) + offset, 3),
                "end": round(float(raw_segment.end) + offset, 3),
                "text": text,
            }]
    return words


# A run of the same word three or more times over is never speech; it is the
# decoder stuck in a loop. Case-insensitive and tolerant of the punctuation the
# turbo model sprinkles between the repeats ("Бекзат, Бекзат, Бекзат").
_REPEAT_RUN_RE = re.compile(
    r"\b(\w[\w'’\-]*)\b(?:[\s,.!?;:…—–\-]+\b\1\b){2,}",
    re.IGNORECASE | re.UNICODE,
)
_TOKEN_RE = re.compile(r"\w[\w'’\-]*", re.UNICODE)

# Share of the collapsed text a single token may occupy before we call the whole
# segment an artefact of the loop rather than a sentence that survived it.
REPEAT_ARTIFACT_SHARE = 0.6


def collapse_repeats(text: str) -> str:
    """Collapse every run of one token repeated >= 3 times to one occurrence."""
    if not text:
        return text
    previous = None
    while previous != text:                    # overlapping runs need a second pass
        previous = text
        text = _REPEAT_RUN_RE.sub(r"\1", text)
    return text


def _is_repeat_artifact(collapsed: str) -> bool:
    """Is what is left of a collapsed segment just the looped token itself?"""
    tokens = [t.lower() for t in _TOKEN_RE.findall(collapsed)]
    if not tokens:
        return True
    top = max(tokens.count(t) for t in set(tokens))
    return top / len(tokens) >= REPEAT_ARTIFACT_SHARE


def _decollapse(segment: Any) -> Any | None:
    """Strip repetition loops from one segment; None means drop it entirely.

    Works on raw faster-whisper segments and on our own :class:`Segment` — both
    are mutable dataclasses. When the text changes the word list no longer
    describes it, so it is cleared and rebuilt as one pseudo-word spanning the
    segment (the same fallback `_words_of` already uses).
    """
    text = (getattr(segment, "text", "") or "").strip()
    collapsed = collapse_repeats(text)
    if collapsed == text:
        return segment
    if _is_repeat_artifact(collapsed):
        logger.debug("dropped (repetition loop): %r", text)
        return None
    logger.debug("collapsed repetition loop: %r -> %r", text, collapsed)
    try:
        segment.text = collapsed
        words = getattr(segment, "words", None)
        if words:
            first, last = words[0], words[-1]
            if isinstance(first, dict):        # our Segment: words are dicts
                segment.words = [{
                    "start": first["start"], "end": last["end"], "text": collapsed,
                }]
            else:                              # raw: let _words_of rebuild it
                segment.words = None
    except Exception:                          # frozen/odd segment: text stands
        logger.debug("could not write back the collapsed text", exc_info=True)
    return segment


def drop_hallucinations(segments: Sequence[Any]) -> list[Any]:
    """Filter Whisper's stock inventions out of a segment list.

    Works on raw faster-whisper segments (which carry ``no_speech_prob``) and on
    our own :class:`Segment` (which does not — then only the repetition and
    boilerplate rules apply).
    """
    kept: list[Any] = []
    previous: str | None = None
    for segment in segments:
        segment = _decollapse(segment)
        if segment is None:
            continue
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        if no_speech > NO_SPEECH_MAX:
            logger.debug("dropped (no_speech_prob=%.2f): %r", no_speech, text)
            continue
        lowered = text.lower()
        if previous is not None and lowered == previous:
            logger.debug("dropped (repeat of previous segment): %r", text)
            continue
        if any(phrase in lowered for phrase in HALLUCINATION_PHRASES):
            logger.debug("dropped (subtitle boilerplate): %r", text)
            continue
        kept.append(segment)
        previous = lowered
    return kept


def _language_shares(segments: Sequence[Segment]) -> dict[str, float]:
    """Share of transcribed words per language, most spoken first."""
    counts: dict[str, int] = {}
    for segment in segments:
        n = len(segment.words) or len(segment.text.split())
        if not n:
            continue
        counts[segment.language] = counts.get(segment.language, 0) + n
    total = sum(counts.values())
    if not total:
        return {}
    shares = {lang: round(n / total, 3) for lang, n in counts.items()}
    return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))


def _transcribe_window(
    pcm: np.ndarray, offset: float, language: str, participants: Sequence[str] | None
) -> list[Segment]:
    """Transcribe one window with the model its language needs.

    Holds the GPU for exactly one window. Per-window rather than per-recording so
    a live utterance can slip between the windows of a 35-minute upload instead
    of queueing behind all of them.
    """
    with gpu.phase_lock():
        model, model_name = load_turbo(), "turbo"
        if language == "kk":
            kk_model = load_kk()
            if kk_model is not None:
                model, model_name = kk_model, "kk"

        # The participants hint is a turbo-only tool. Fed to the Kazakh
        # fine-tune it is the trigger for the repetition loop: the model latches
        # onto the prompt tokens and emits a name until the window runs out.
        extra: dict[str, Any] = (
            dict(KK_TRANSCRIBE_KWARGS) if model_name == "kk"
            else _prompt_kwargs(participants)
        )

        t0 = time.perf_counter()
        raw_segments, _ = model.transcribe(
            pcm,
            language=language,
            **TRANSCRIBE_KWARGS,
            **extra,
        )
        raw_segments = list(raw_segments)       # generator: force decode before timing
        asr_sec = time.perf_counter() - t0

    segments: list[Segment] = []
    for raw in drop_hallucinations(raw_segments):
        text = raw.text.strip()
        if model_name == "kk":
            # The fine-tune emits lowercase, unpunctuated text. Capitalising the
            # first letter is the whole of the cosmetic fix we apply — inventing
            # punctuation would invent meaning.
            text = text[:1].upper() + text[1:]
        segments.append(Segment(
            start=round(float(raw.start) + offset, 3),
            end=round(float(raw.end) + offset, 3),
            text=text,
            language=language,
            words=_words_of(raw, offset),
        ))

    logger.info(
        "window %.2f–%.2fs -> %s via %s (asr %.2fs, %d seg)",
        offset, offset + len(pcm) / SAMPLE_RATE, language, model_name,
        asr_sec, len(segments),
    )
    return segments


def _assemble(segments: list[Segment], seconds: float) -> Transcript:
    segments = drop_hallucinations(segments)   # catch repeats across window borders
    shares = _language_shares(segments)
    return Transcript(
        text=" ".join(s.text for s in segments).strip(),
        language=next(iter(shares), "unknown"),
        segments=segments,
        seconds=round(seconds, 3),
        languages=shares,
    )


def transcribe(
    audio_path: str | Path, participants: Sequence[str] | None = None
) -> Transcript:
    """Transcribe a recording, routing each speech window to the right model.

    Args:
        audio_path: any format ffmpeg/PyAV can decode.
        participants: names of the people in the meeting, fed to the decoder as
            a prompt so they come out spelled correctly.
    """
    pcm = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    seconds = len(pcm) / SAMPLE_RATE
    chunks = _speech_chunks(pcm)
    groups = _lid_groups(chunks)
    if not groups:
        # Nothing to route, so nothing to load either: a silent upload must not
        # cost 4 s of model load before it can return an empty transcript.
        logger.info("%s: %.1fs audio, no speech found", audio_path, seconds)
        return _assemble([], seconds)

    # Language ID every group, then merge neighbours that agree: the language is
    # what decides where a transcription window may end, so that no window is
    # ever handed to a model that cannot speak part of it.
    turbo = load_turbo()
    routed: list[tuple[int, int, str]] = []
    lid_total = 0.0
    for start, end in groups:
        t0 = time.perf_counter()
        language, probability = _detect_language(turbo, pcm[start:end])
        elapsed = time.perf_counter() - t0
        lid_total += elapsed
        logger.info(
            "group %.2f–%.2fs -> %s p=%.3f (lid %.2fs)",
            start / SAMPLE_RATE, end / SAMPLE_RATE, language, probability, elapsed,
        )
        routed.append((start, end, language))

    windows = _merge_by_language(routed)
    logger.info(
        "%s: %.1fs audio, %d speech chunk(s) -> %d language group(s) -> %d window(s), "
        "language ID %.1fs",
        audio_path, seconds, len(chunks), len(groups), len(windows), lid_total,
    )

    segments: list[Segment] = []
    for start, end, language in windows:
        segments.extend(
            _transcribe_window(pcm[start:end], start / SAMPLE_RATE, language, participants)
        )
    return _assemble(segments, seconds)


def transcribe_pcm(
    pcm: Sequence[float] | np.ndarray,
    sr: int = SAMPLE_RATE,
    participants: Sequence[str] | None = None,
) -> Transcript:
    """Transcribe one already-segmented utterance (the live path).

    No VAD step — the caller has already decided this is speech. Assumes a
    single utterance of at least ~0.5 s and at most ~30 s.
    """
    audio = _resample(_as_float32(pcm), sr)
    if audio.size == 0:
        return Transcript(text="", language="unknown", segments=[], seconds=0.0, languages={})

    # One utterance = one GPU hold: language ID and decode should not be split by
    # an LLM phase that would move the models off the card between them. The lock
    # is re-entrant, so the acquires inside both callees are free.
    with gpu.phase_lock():
        t0 = time.perf_counter()
        language, probability = _detect_language(load_turbo(), audio)
        logger.info(
            "utterance %.2fs -> %s p=%.3f (lid %.2fs)",
            audio.size / SAMPLE_RATE, language, probability, time.perf_counter() - t0,
        )
        segments = _transcribe_window(audio, 0.0, language, participants)
    return _assemble(segments, audio.size / SAMPLE_RATE)


# --------------------------------------------------------------------------
# Structuring
# --------------------------------------------------------------------------

PROTOCOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "points"],
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "task": {"type": "string"},
                    "due": {"type": "string"},
                    "priority": {"type": "string"},
                },
                "required": ["owner", "task", "due", "priority"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary", "decisions", "topics",
        "open_questions", "action_items", "risks",
    ],
}

SYSTEM = """Ты — протоколист деловых встреч. На входе стенограмма (русский, казахский
или английский, возможна смешанная речь). Верни СТРОГО JSON по схеме.

Правила:
- summary: 3–5 предложений для руководителя. Суть, а не пересказ.
- decisions: только то, о чём реально договорились. Обсуждение без вывода — не решение.
- topics: смысловые блоки встречи, каждый с тезисами.
- open_questions: вопросы, оставшиеся без ответа.
- action_items: поручения. owner — имя из стенограммы; если не назван, пиши "не назначен".
  due — срок только если он прозвучал, иначе "". priority — высокий / средний / низкий.
- risks: разногласия, спорные места, технические риски.

Ничего не выдумывай. Если чего-то в стенограмме нет — оставь пустой массив.
Отвечай на языке встречи."""

EMPTY: dict[str, list] = {
    "summary": [], "decisions": [], "topics": [],
    "open_questions": [], "action_items": [], "risks": [],
}

# language_hint -> line appended to SYSTEM so the protocol is written in the
# language the meeting was actually held in, instead of leaving it to the
# model's own "answer in the meeting's language" judgement.
_LANGUAGE_HINTS: dict[str, str] = {
    "ru": "русский",
    "kk": "қазақ тілі",
    "en": "English",
}

# What `protocol_language` may be. "auto" means "follow whatever was spoken
# most"; anything else is an explicit order and wins over the dominant language.
PROTOCOL_LANGUAGES = frozenset({"ru", "kk", "en", "auto"})
DEFAULT_PROTOCOL_LANGUAGE = "ru"


def resolve_protocol_language(choice: str | None, spoken: str | None) -> str | None:
    """The `language_hint` to give the LLM for this protocol.

    `choice` is the user's selection (default "ru"); `spoken` is the dominant
    transcript language, used only when the choice is "auto". An unknown choice
    falls back to "auto" rather than failing a demo over a typo.
    """
    choice = (choice or DEFAULT_PROTOCOL_LANGUAGE).strip().lower()
    if choice not in PROTOCOL_LANGUAGES:
        choice = "auto"
    if choice == "auto":
        return spoken if spoken and spoken != "unknown" else None
    return choice

_resolved_llm_model: str | None = None
# Models that reject the `think` kwarg outright get remembered here so we
# stop paying for a failed call + retry on every subsequent request.
_think_unsupported: set[str] = set()


def resolved_llm_model() -> str:
    """Resolve, once, the model name to actually use against local Ollama.

    `LLM_MODEL` env var wins outright if set. Otherwise the first of
    `LLM_MODEL_CANDIDATES` that is actually pulled (per `ollama.list()`) is
    used. The result is cached for the life of the process — call this again
    any time (e.g. from /health) and you get the same answer without
    re-hitting Ollama.

    Raises RuntimeError naming the pull command if nothing usable is found.
    """
    global _resolved_llm_model
    if _resolved_llm_model is not None:
        return _resolved_llm_model

    env_model = os.getenv("LLM_MODEL")
    if env_model:
        _resolved_llm_model = env_model
        return _resolved_llm_model

    try:
        available = {
            m.get("model") or m.get("name")
            for m in ollama.list().get("models", [])
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Ollama to resolve an LLM model "
            f"(is it running at localhost:11434?): {exc}. "
            f"Pull one with `ollama pull {LLM_MODEL_CANDIDATES[0]}`."
        ) from exc

    for candidate in LLM_MODEL_CANDIDATES:
        if candidate in available:
            _resolved_llm_model = candidate
            return _resolved_llm_model

    raise RuntimeError(
        "No supported LLM model is pulled in Ollama. Run "
        f"`ollama pull {LLM_MODEL_CANDIDATES[0]}` "
        f"(candidates tried: {LLM_MODEL_CANDIDATES})."
    )


def _chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return out


def _llm_keep_alive() -> str | int:
    """How long Ollama should hold the model in memory after a call.

    On the GPU path the LLM is a tenant, not the owner: long enough that a
    chunked transcript does not reload the model between chunks, short enough
    that a crash before our explicit unload still gives the card back. On the CPU
    path nothing is contended, so keep the original half-hour.
    """
    return "2m" if gpu.is_active() else "30m"


def _ask(transcript: str, model: str, language_hint: str | None = None) -> dict[str, Any]:
    system = SYSTEM
    hint = _LANGUAGE_HINTS.get(language_hint or "")
    if hint:
        # A bare language name is too weak for a 4B model when the meeting
        # transcript itself mixes languages — it tends to default back to
        # Russian. Spell out that every field must follow this language.
        system = (
            f"{system}\n\nЯзык протокола: {hint}. Весь текст протокола "
            f"(summary, decisions, topics, open_questions, action_items, risks) "
            f"должен быть написан на этом языке — не переводи на русский."
        )

    chat_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Стенограмма встречи:\n\n{transcript}"},
        ],
        # Structured output schema — Ollama enforces this at decode time.
        format=PROTOCOL_SCHEMA,
        keep_alive=_llm_keep_alive(),
        # Ollama's default context silently truncates the FRONT of long
        # inputs, so num_ctx is not optional here.
        options={
            "num_ctx": 16384,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.0,
            "num_predict": 2048,
            "seed": 0,
        },
    )

    if model in _think_unsupported:
        resp = ollama.chat(**chat_kwargs)
    else:
        try:
            resp = ollama.chat(think=False, **chat_kwargs)
        except ollama.ResponseError as exc:
            # Ollama answers an unrecognised/unsupported request field with a
            # 400 and a message naming the field — that's the only case we
            # treat as "this model rejects `think`". Anything else (a 5xx from
            # an OOM, a malformed-but-unrelated request, etc.) is a real error
            # and must propagate untouched rather than being silently retried
            # and mis-diagnosed as a think-support problem.
            rejected_think = exc.status_code == 400 and "think" in (exc.error or "").lower()
            if not rejected_think:
                raise
            logger.warning(
                "model %s rejected think=False (%s); retrying without it", model, exc
            )
            resp = ollama.chat(**chat_kwargs)
            # Only remember this model as think-unsupported once the
            # fallback call has actually succeeded — a transient failure on
            # the retry must not get mis-tagged as "think unsupported" (that
            # would silently disable think=False for the rest of the process
            # based on a connection blip, not a real capability gap).
            _think_unsupported.add(model)

    content = resp["message"]["content"]
    try:
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, TypeError):
        head = (content or "")[:300] if isinstance(content, str) else repr(content)[:300]
        logger.warning("LLM returned non-JSON content, falling back to EMPTY. Head: %r", head)
        return dict(EMPTY)


def _merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk protocols, dropping duplicates but keeping order."""
    out: dict[str, Any] = {k: [] for k in EMPTY}
    for part in parts:
        for key in out:
            for item in part.get(key) or []:
                fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if fingerprint not in {
                    json.dumps(x, ensure_ascii=False, sort_keys=True) for x in out[key]
                }:
                    out[key].append(item)
    out["summary"] = out["summary"][:5]
    return out


def structure(
    transcript: str, model: str | None = None, language_hint: str | None = None
) -> dict[str, Any]:
    """Transcript in, protocol out — borrowing the GPU from STT for the duration.

    On the GPU path the whole LLM phase sits inside `gpu.llm_phase()`: the STT
    weights move to RAM first, every chunk is answered while the card is the
    LLM's, and the card is handed back before this returns. That is one hold for
    all chunks, not one per chunk — 3.4 GiB reloaded per chunk would cost more
    than the structuring itself.
    """
    if not transcript.strip():
        return dict(EMPTY)
    resolved = model or resolved_llm_model()
    with gpu.llm_phase(resolved):
        parts = [_ask(c, resolved, language_hint) for c in _chunks(transcript)]
    return parts[0] if len(parts) == 1 else _merge(parts)


# --------------------------------------------------------------------------
# Question answering over a finished meeting ("RAG-чат")
# --------------------------------------------------------------------------
#
# Retrieval here is deliberately crude: lexical overlap on 5-character
# prefixes. Russian and Kazakh are heavily inflected ("бюджет" / "бюджету" /
# "бюджетті"), so a prefix is a poor man's stemmer that costs nothing and needs
# no extra model on a card that is already shared between STT and the LLM. A
# one-hour meeting is a few hundred passages — small enough that the ranking
# only has to beat "no ranking", and most meetings fit in the context whole.

CHAT_LANGUAGES = frozenset({"ru", "kk", "en"})
_CHAT_LANGUAGE_NAMES = {"ru": "русском", "kk": "казахском (қазақ тілінде)", "en": "английском"}
#: Below this many passages the whole meeting goes into the prompt; above it we
#: select. 40 short passages comfortably fit the 16k context we ask Ollama for.
CHAT_ALL_PASSAGES_UNDER = 40
CHAT_TOP_K = 12
_STEM_LEN = 5
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Question words carry no signal — every passage would match them, which turns
# the ranking into "whichever passage is longest".
_CHAT_STOPWORDS = frozenset({
    "что", "как", "кто", "где", "когда", "какой", "какая", "какие", "каких",
    "каком", "чем", "чём", "этот", "эта", "эти", "это", "был", "была", "было",
    "были", "для", "при", "про", "они", "она", "оно", "нас", "вам", "там",
    "тут", "или", "мы", "вы",
    "қандай", "қашан", "кім", "қай", "қайда", "бұл", "осы", "және",
    "the", "and", "for", "was", "were", "what", "who", "when", "where", "how",
    "which", "did", "does", "about", "with", "this", "that", "any",
})


def _mmss(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    return f"{total // 60:02d}:{total % 60:02d}"


def _stems(text: str) -> set[str]:
    """Content words of `text`, cut to a 5-character prefix."""
    return {
        word[:_STEM_LEN]
        for word in _WORD_RE.findall((text or "").lower())
        if len(word) >= 3 and word not in _CHAT_STOPWORDS
    }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _transcript_passages(transcript: str, target: int = 420) -> list[dict[str, Any]]:
    """Cut a bare transcript into passages when there are no timed segments.

    `/process-text` — the fast demo path — stores a transcript and nothing
    else, so without this the chat has nothing to retrieve over. The times are
    left at zero rather than invented; `answer_question` notices and asks for
    fragment numbers instead of [mm:ss] citations.
    """
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split((transcript or "").strip()):
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > target:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [{"start": 0.0, "end": 0.0, "speaker": "", "text": c} for c in chunks]


def chat_passages(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The meeting as retrievable passages: speaker blocks, else segments, else text."""
    blocks = result.get("speakers") or result.get("segments") or []
    passages: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(block.get("start") or 0.0)
            end = float(block.get("end") or start)
        except (TypeError, ValueError):
            start = end = 0.0
        passages.append({
            "start": start, "end": end,
            "speaker": str(block.get("speaker") or ""),
            "text": text,
        })
    return passages or _transcript_passages(str(result.get("transcript") or ""))


def rank_passages(result: dict[str, Any], question: str) -> list[tuple[int, dict[str, Any]]]:
    """Passages scored by question-token overlap, best first, ties by time."""
    wanted = _stems(question)
    scored = [
        (len(wanted & _stems(p["text"])), p)
        for p in chat_passages(result)
    ]
    return sorted(scored, key=lambda pair: (-pair[0], pair[1]["start"]))


def select_passages(result: dict[str, Any], question: str) -> list[dict[str, Any]]:
    """What actually goes in the prompt, in chronological order.

    A short meeting goes in whole — the model answers better when it can see
    the run of the conversation than when it sees twelve disconnected snippets.
    """
    order = {
        (p["start"], p["text"]): i for i, p in enumerate(chat_passages(result))
    }
    ranked = rank_passages(result, question)
    if len(ranked) > CHAT_ALL_PASSAGES_UNDER:
        ranked = ranked[:CHAT_TOP_K]
    return [
        p for _, p in
        sorted(ranked, key=lambda pair: order.get((pair[1]["start"], pair[1]["text"]), 0))
    ]


def chat_language(result: dict[str, Any], language: str | None) -> str:
    """ru/kk/en to answer in. "auto" defers to the protocol language, then the floor."""
    for candidate in (language, result.get("protocol_language"), result.get("language")):
        value = str(candidate or "").strip().lower()
        if value in CHAT_LANGUAGES:
            return value
    return DEFAULT_PROTOCOL_LANGUAGE


def _cited_seconds(answer: str) -> set[int]:
    """The [mm:ss] stamps the model actually used, as whole seconds."""
    out: set[int] = set()
    for bracketed in re.findall(r"\[([^\]]{0,40})\]", answer or ""):
        for minutes, seconds in re.findall(r"(\d{1,3}):(\d{2})", bracketed):
            out.add(int(minutes) * 60 + int(seconds))
    return out


_CHAT_SYSTEM = (
    "Ты отвечаешь на вопросы по расшифровке встречи. Правила:\n"
    "1. Отвечай ТОЛЬКО по приведённым фрагментам. Ничего не додумывай.\n"
    "2. Если ответа в фрагментах нет — прямо скажи, что этого во встрече не было.\n"
    "3. Ответ — 1–4 предложения, без вступлений и без списков.\n"
    "4. Укажи время фрагментов, на которые опираешься, в формате [мм:сс].\n"
    "5. Пиши ответ на {language} языке."
)


_CHAT_SYSTEM_UNTIMED = (
    "Ты отвечаешь на вопросы по расшифровке встречи. Правила:\n"
    "1. Отвечай ТОЛЬКО по приведённым фрагментам. Ничего не додумывай.\n"
    "2. Если ответа в фрагментах нет — прямо скажи, что этого во встрече не было.\n"
    "3. Ответ — 1–4 предложения, без вступлений и без списков.\n"
    "4. Пиши ответ на {language} языке."
)


def answer_question(
    result: dict[str, Any], question: str, language: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a question about one meeting from its own transcript.

    Retrieval is local and lexical, generation is the same local Ollama model
    that writes the protocol, and the GPU is borrowed the same way `structure`
    borrows it. Never raises: a chat box that throws in front of judges is
    worse than one that says the model is unavailable.
    """
    lang = chat_language(result, language)
    question = str(question or "").strip()
    if not question:
        return {"answer": "Вопрос пустой.", "sources": [], "language": lang}

    ranked = rank_passages(result, question)
    passages = select_passages(result, question)
    if not passages:
        return {
            "answer": "В этой встрече нет расшифровки, по которой можно ответить.",
            "sources": [], "language": lang,
        }

    # A text-only meeting has no timeline to cite, so don't ask for one: the
    # model would invent [00:00] stamps and the chips under the answer would be
    # a lie about where it came from.
    timed = any(p["start"] > 0 or p["end"] > 0 for p in passages)
    if timed:
        excerpts = "\n".join(
            f"[{_mmss(p['start'])}–{_mmss(p['end'])}"
            + (f" {p['speaker']}" if p["speaker"] else "")
            + f"] {p['text']}"
            for p in passages
        )
        template = _CHAT_SYSTEM
    else:
        excerpts = "\n".join(
            f"[фрагмент {i}] {p['text']}" for i, p in enumerate(passages, start=1)
        )
        template = _CHAT_SYSTEM_UNTIMED
    system = template.format(language=_CHAT_LANGUAGE_NAMES.get(lang, "русском"))
    user = f"Фрагменты встречи:\n\n{excerpts}\n\nВопрос: {question}"

    try:
        resolved = model or resolved_llm_model()
        chat_kwargs: dict[str, Any] = dict(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            keep_alive=_llm_keep_alive(),
            options={
                "num_ctx": 16384,
                "temperature": 0.2,
                "num_predict": 512,
            },
        )
        # Same GPU handoff as the protocol: STT parks in RAM, the LLM answers,
        # the card goes back before this returns.
        with gpu.llm_phase(resolved):
            if resolved in _think_unsupported:
                response = ollama.chat(**chat_kwargs)
            else:
                try:
                    response = ollama.chat(think=False, **chat_kwargs)
                except ollama.ResponseError as exc:
                    if not (exc.status_code == 400 and "think" in (exc.error or "").lower()):
                        raise
                    logger.warning("model %s rejected think=False (%s); retrying", resolved, exc)
                    response = ollama.chat(**chat_kwargs)
                    _think_unsupported.add(resolved)
        answer = str(response["message"]["content"] or "").strip()
    except Exception as exc:                                    # noqa: BLE001
        logger.exception("chat answer failed")
        return {"answer": f"Не удалось получить ответ от модели: {exc}",
                "sources": [], "language": lang}

    cited = _cited_seconds(answer) if timed else set()
    sources = [p for p in passages if int(p["start"]) in cited]
    if not sources:
        # The model answered but cited nothing usable — show what it was
        # actually given to read, so the answer is still checkable.
        sources = [p for _, p in ranked[:3]]
        sources.sort(key=lambda p: p["start"])
    return {"answer": answer, "sources": sources, "language": lang}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

# 0.5 over-segments: on the two-speaker team recordings it finds 5-6 speakers.
# 0.7 is the lowest value in {0.5..0.9} at which *both* come out at exactly 2
# (0.6 gives 3), so it is the default; the env var is the stage escape hatch.
DIARIZE_THRESHOLD_DEFAULT = 0.7


def process_audio(path: str | Path, whisper_size: str | None = None,
                  llm_model: str | None = None,
                  participants: Sequence[str] | None = None,
                  num_speakers: int | None = None,
                  diarize_threshold: float | None = None,
                  protocol_language: str | None = DEFAULT_PROTOCOL_LANGUAGE,
                  ) -> dict[str, Any]:
    """Audio in, transcript + structured protocol (+ speaker blocks) out.

    `whisper_size` is accepted and ignored: the STT model is no longer a single
    size, it is picked per speech window by language (see `transcribe`).

    `llm_model` defaults to None so `structure()` resolves it — a missing model
    then gives the friendly "run `ollama pull ...`" RuntimeError instead of a
    raw Ollama 404 for a model nobody has.

    Diarization runs *concurrently* with the LLM: one needs the audio, the other
    the text, and on a 77 s recording that overlap is ~10 s of wall clock in
    front of an audience. It is strictly best-effort — a diarizer that is
    missing, or that throws, leaves the plain transcript untouched.
    """
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    tr = transcribe(path, participants=participants)
    t1 = time.perf_counter()
    hint = resolve_protocol_language(protocol_language, tr.language)

    threshold = (diarize_threshold if diarize_threshold is not None
                 else float(os.getenv("DIARIZE_THRESHOLD") or DIARIZE_THRESHOLD_DEFAULT))
    timings: dict[str, float] = {"diarize": 0.0}
    protocol_error: str | None = None

    def diarize_blocks() -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            import diarize as diarization

            if not diarization.is_available():
                return []
            words = [w for s in tr.segments for w in s.words]
            if not words:
                return []
            turns = diarization.diarize(path, num_speakers=num_speakers, threshold=threshold)
            return diarization.assign_speakers(words, turns)
        except Exception:
            logger.exception("diarization failed; falling back to the plain transcript")
            return []
        finally:
            timings["diarize"] = time.perf_counter() - started

    def protocol_of_text() -> dict[str, Any]:
        nonlocal protocol_error
        started = time.perf_counter()
        try:
            return structure(tr.text, llm_model, language_hint=hint)
        except Exception as e:
            logger.exception("protocol structuring failed; returning empty protocol")
            protocol_error = str(e)
            return dict(EMPTY)
        finally:
            timings["llm"] = time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="process-audio") as pool:
        fut_speakers = pool.submit(diarize_blocks)
        fut_protocol = pool.submit(protocol_of_text)
        speakers = fut_speakers.result()
        protocol = fut_protocol.result()          # never raises: see protocol_of_text

    if speakers:
        try:
            _label_segments_by_words(tr.segments, speakers)
        except Exception:
            logger.exception("speaker labelling failed; segments stay unlabelled")
    t2 = time.perf_counter()

    result: dict[str, Any] = {
        "transcript": tr.text,
        "language": tr.language,
        "languages": tr.languages,
        "protocol_language": hint or "auto",
        "participants": list(participants or []),
        "audio_seconds": round(tr.seconds, 1),
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "language": s.language,
                "words": s.words,
                "speaker": s.speaker,
            }
            for s in tr.segments
        ],
        "protocol": protocol,
        "timing": {
            "transcribe_sec": round(t1 - t0, 2),
            "diarize_sec": round(timings["diarize"], 2),
            "llm_sec": round(timings.get("llm", t2 - t1), 2),
            "total_sec": round(t2 - t0, 2),
            "realtime_factor": round(tr.seconds / (t2 - t0), 2) if (t2 - t0) > 0 and tr.seconds else None,
        },
    }
    if speakers:
        result["speakers"] = speakers
        result["num_speakers"] = len({b["speaker"] for b in speakers})
    if protocol_error is not None:
        result["protocol_error"] = protocol_error
    return result


def _label_segments_by_words(segments: Sequence[Segment], blocks: Sequence[dict]) -> None:
    """Stamp each segment with the speaker most of its words belong to.

    Word-level rather than segment-overlap: a segment that straddles a turn
    boundary should be attributed to whoever said most of it, not to whichever
    block happens to overlap the segment's span by a millisecond more.
    """
    if not blocks:
        return
    for seg in segments:
        votes: dict[str, float] = {}
        for word in seg.words:
            start, end = word.get("start"), word.get("end")
            if start is None or end is None:
                continue
            mid = (float(start) + float(end)) / 2
            for b in blocks:
                if b["start"] <= mid <= b["end"]:
                    votes[b["speaker"]] = votes.get(b["speaker"], 0.0) + 1.0
                    break
        if not votes:                              # no words, or none inside a turn
            for b in blocks:
                overlap = min(seg.end, b["end"]) - max(seg.start, b["start"])
                if overlap > 0:
                    votes[b["speaker"]] = votes.get(b["speaker"], 0.0) + overlap
        if votes:
            seg.speaker = max(votes.items(), key=lambda kv: kv[1])[0]


def process_text(
    transcript: str, llm_model: str | None = None, language_hint: str | None = None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    protocol = structure(transcript, llm_model, language_hint)
    t1 = time.perf_counter()
    return {
        "transcript": transcript,
        "language": "n/a",
        "audio_seconds": 0.0,
        "segments": [],
        "protocol": protocol,
        "timing": {"transcribe_sec": 0.0, "llm_sec": round(t1 - t0, 2),
                   "total_sec": round(t1 - t0, 2), "realtime_factor": None},
    }
