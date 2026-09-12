"""Offline batch speaker diarization + word-level speaker assignment.

Two independent halves:

* :func:`diarize` runs sherpa-onnx' offline speaker diarization (pyannote
  segmentation + a 3D-Speaker ERes2Net embedding extractor, both ONNX, both on
  CPU) over a whole recording and returns speaker *turns*.
* :func:`assign_speakers` takes Whisper word timestamps plus those turns and
  produces speaker-labelled text blocks. It is pure Python and has no
  dependency on the models, so it is testable without them.

Everything here is 100% offline: the model files are fetched once by
``api/scripts/get_diarization_models.sh`` and live (uncommitted) in
``api/models/diarization/``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "MODELS_DIR",
    "SEGMENTATION_MODEL",
    "EMBEDDING_MODEL",
    "is_available",
    "diarize",
    "assign_speakers",
]

SAMPLE_RATE = 16000

MODELS_DIR = Path(__file__).resolve().parent / "models" / "diarization"
SEGMENTATION_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_MODEL = MODELS_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

NUM_THREADS = 4
MIN_DURATION_ON = 0.3
MIN_DURATION_OFF = 0.5
DEFAULT_THRESHOLD = 0.5

# One diarizer is reused across calls: loading ~44 MB of ONNX per request is
# pure waste. sherpa-onnx exposes set_config() precisely so the clustering
# parameters can be swapped without reloading the models. The lock keeps
# concurrent requests from racing on set_config()/process().
_diarizer: Any = None
_diarizer_lock = threading.Lock()


def is_available() -> bool:
    """True when both diarization model files are present on disk."""
    return SEGMENTATION_MODEL.is_file() and EMBEDDING_MODEL.is_file()


def _build_config(num_speakers: int | None, threshold: float):
    import sherpa_onnx

    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION_MODEL),
            ),
            num_threads=NUM_THREADS,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMBEDDING_MODEL),
            num_threads=NUM_THREADS,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            # num_clusters=-1 means "decide from the data", in which case
            # `threshold` is what actually stops the agglomeration.
            num_clusters=int(num_speakers) if num_speakers is not None else -1,
            threshold=float(threshold),
        ),
        min_duration_on=MIN_DURATION_ON,
        min_duration_off=MIN_DURATION_OFF,
    )


def _get_diarizer(num_speakers: int | None, threshold: float):
    """Return the shared diarizer, reconfigured for this call."""
    global _diarizer

    import sherpa_onnx

    config = _build_config(num_speakers, threshold)
    if not config.validate():
        raise RuntimeError(
            f"invalid sherpa-onnx diarization config (segmentation={SEGMENTATION_MODEL}, "
            f"embedding={EMBEDDING_MODEL})"
        )

    if _diarizer is None:
        _diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    else:
        _diarizer.set_config(config)
    return _diarizer


def _as_samples(audio: str | Path | np.ndarray | Sequence[float], sr: int) -> np.ndarray:
    """Coerce a path or array into float32 mono samples at SAMPLE_RATE."""
    if isinstance(audio, (str, Path)):
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(str(audio), sampling_rate=SAMPLE_RATE), dtype=np.float32)

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:  # (channels, n) or (n, channels) -> mono
        axis = 0 if samples.shape[0] < samples.shape[1] else 1
        samples = samples.mean(axis=axis)
    if samples.ndim != 1:
        raise ValueError(f"expected mono or 2-D audio, got shape {samples.shape}")
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"diarization needs {SAMPLE_RATE} Hz audio, got sr={sr}; "
            "resample first or pass a file path so it is decoded at 16 kHz"
        )
    return np.ascontiguousarray(samples)


def diarize(
    audio: str | Path | np.ndarray | Sequence[float],
    sr: int = SAMPLE_RATE,
    num_speakers: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Diarize a whole recording and return speaker turns.

    Args:
        audio: path to an audio file (decoded to 16 kHz mono via faster-whisper)
            or a float32 numpy array of mono samples.
        sr: sample rate of ``audio`` when it is an array. Must be 16000.
        num_speakers: exact number of speakers, when known. ``None`` lets the
            clustering decide using ``threshold``.
        threshold: clustering threshold used only when ``num_speakers`` is None.
            Lower finds more speakers.

    Returns:
        ``[{"start": float, "end": float, "speaker": int}, ...]`` sorted by
        start time. ``speaker`` is a 0-based cluster index.
    """
    if not is_available():
        raise RuntimeError(
            "diarization models are missing; run api/scripts/get_diarization_models.sh "
            f"(expected {SEGMENTATION_MODEL} and {EMBEDDING_MODEL})"
        )
    if num_speakers is not None and num_speakers < 1:
        raise ValueError(f"num_speakers must be >= 1 or None, got {num_speakers}")

    samples = _as_samples(audio, sr)
    if samples.size == 0:
        return []

    with _diarizer_lock:
        diarizer = _get_diarizer(num_speakers, threshold)
        result = diarizer.process(samples)

    return [
        {"start": float(seg.start), "end": float(seg.end), "speaker": int(seg.speaker)}
        for seg in result.sort_by_start_time()
    ]


# --------------------------------------------------------------------------
# word -> speaker assignment (pure python, no models needed)
# --------------------------------------------------------------------------


def _word_bounds(word: Any) -> tuple[float, float, str]:
    """Pull (start, end, text) out of a dict or a faster-whisper Word."""
    if isinstance(word, dict):
        text = word.get("word", word.get("text", ""))
        start = word.get("start")
        end = word.get("end")
    else:
        text = getattr(word, "word", getattr(word, "text", ""))
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
    if start is None or end is None:
        raise ValueError(f"word is missing start/end timestamps: {word!r}")
    return float(start), float(end), str(text).strip()


def _turn_bounds(turn: Any) -> tuple[float, float, int]:
    if isinstance(turn, dict):
        return float(turn["start"]), float(turn["end"]), int(turn["speaker"])
    return float(turn.start), float(turn.end), int(turn.speaker)


def assign_speakers(words: Iterable[Any], turns: Iterable[Any]) -> list[dict]:
    """Label each word with a speaker and group runs into blocks.

    Each word takes the speaker of the turn it overlaps most. A word that
    overlaps nothing inherits the previous word's speaker (the first word takes
    the nearest turn instead). Short stutters -- a run of fewer than 3 words
    whose neighbouring runs agree on a speaker -- are smoothed away.

    Args:
        words: dicts with ``word``/``text``, ``start``, ``end`` (faster-whisper
            ``Word`` objects also work).
        turns: the output of :func:`diarize` (or anything with start/end/speaker).

    Returns:
        ``[{"speaker": "Speaker 1", "start": float, "end": float, "text": str}, ...]``
        in time order. Returns ``[]`` if either input is empty -- callers should
        fall back to the plain transcript in that case.
    """
    parsed_words = [_word_bounds(w) for w in words]
    parsed_turns = [_turn_bounds(t) for t in turns]
    if not parsed_words or not parsed_turns:
        return []

    # 1. max-overlap assignment, with inheritance for words that overlap nothing
    speakers: list[int] = []
    for w_start, w_end, _ in parsed_words:
        best_speaker = None
        best_overlap = 0.0
        for t_start, t_end, t_speaker in parsed_turns:
            overlap = min(w_end, t_end) - max(w_start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t_speaker

        if best_speaker is None:
            if speakers:
                best_speaker = speakers[-1]  # inherit from the previous word
            else:
                # first word, no overlap anywhere: take the nearest turn
                best_speaker = min(
                    parsed_turns,
                    key=lambda t: max(t[0] - w_end, w_start - t[1], 0.0),
                )[2]
        speakers.append(best_speaker)

    # 2. smoothing: a short run flanked by two runs of the same speaker joins them.
    # Every decision is made against `original`, a snapshot of the run speakers
    # taken before the pass. Reading runs[i - 1] live instead would let a run that
    # was just relabelled masquerade as the left neighbour of the next run, so two
    # adjacent short runs (X X X X X | Y | X | Y Y Y Y Y) would smooth the first
    # and then wrongly leave the second alone.
    runs = _runs(speakers)
    original = [run[0] for run in runs]
    for i in range(1, len(runs) - 1):
        _, start_idx, end_idx = runs[i]
        if end_idx - start_idx < 3 and original[i - 1] == original[i + 1]:
            runs[i] = (original[i - 1], start_idx, end_idx)
    for speaker, start_idx, end_idx in runs:
        for idx in range(start_idx, end_idx):
            speakers[idx] = speaker

    # 3. group consecutive same-speaker words into blocks
    blocks: list[dict] = []
    for speaker, start_idx, end_idx in _runs(speakers):
        chunk = parsed_words[start_idx:end_idx]
        blocks.append(
            {
                "speaker": f"Speaker {speaker + 1}",
                "start": chunk[0][0],
                "end": chunk[-1][1],
                "text": " ".join(text for _, _, text in chunk if text),
            }
        )
    return blocks


def _runs(speakers: Sequence[int]) -> list[tuple[int, int, int]]:
    """Split a speaker-per-word list into (speaker, start, end) half-open runs."""
    runs: list[tuple[int, int, int]] = []
    for idx, speaker in enumerate(speakers):
        if runs and runs[-1][0] == speaker:
            prev_speaker, start_idx, _ = runs[-1]
            runs[-1] = (prev_speaker, start_idx, idx + 1)
        else:
            runs.append((speaker, idx, idx + 1))
    return runs
