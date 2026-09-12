"""Tests for api/diarize.py.

The assign_speakers tests are pure python and always run. The diarize() tests
need both the ONNX models (api/scripts/get_diarization_models.sh) and two
FLEURS sample files in tests/audio/, so they skip when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import diarize as d

AUDIO_DIR = Path(__file__).resolve().parent / "audio"
SILENCE_SECONDS = 0.5


def w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


# --------------------------------------------------------------------------
# assign_speakers
# --------------------------------------------------------------------------


def test_empty_inputs_give_empty_output():
    assert d.assign_speakers([], []) == []
    assert d.assign_speakers([], [{"start": 0.0, "end": 1.0, "speaker": 0}]) == []
    assert d.assign_speakers([w("hi", 0.0, 0.5)], []) == []


def test_word_takes_the_speaker_it_overlaps_most():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 1.0, "end": 3.0, "speaker": 1},
    ]
    # 0.2 s inside speaker 0, 0.8 s inside speaker 1 -> speaker 1
    blocks = d.assign_speakers([w("straddle", 0.8, 1.8)], turns)
    assert [b["speaker"] for b in blocks] == ["Speaker 2"]

    # ...and the other way round
    blocks = d.assign_speakers([w("straddle", 0.1, 1.1)], turns)
    assert [b["speaker"] for b in blocks] == ["Speaker 1"]


def test_word_in_a_gap_inherits_the_previous_word():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": 0},
        {"start": 5.0, "end": 6.0, "speaker": 1},
    ]
    words = [w("a", 0.0, 0.5), w("b", 2.0, 2.5), w("c", 5.0, 5.5)]
    blocks = d.assign_speakers(words, turns)
    # "b" overlaps nothing, so it stays with speaker 0 and merges into block 1
    assert [b["speaker"] for b in blocks] == ["Speaker 1", "Speaker 2"]
    assert blocks[0]["text"] == "a b"
    assert blocks[1]["text"] == "c"


def test_first_word_without_overlap_takes_the_nearest_turn():
    turns = [
        {"start": 10.0, "end": 12.0, "speaker": 0},
        {"start": 2.0, "end": 3.0, "speaker": 1},
    ]
    blocks = d.assign_speakers([w("early", 0.0, 0.5)], turns)
    assert blocks[0]["speaker"] == "Speaker 2"  # the turn at 2.0 s is nearer


def test_short_run_between_matching_neighbours_is_smoothed():
    # speaker 0 x3, speaker 1 x2 (a stutter), speaker 0 x3 -> all speaker 0
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": 0},
        {"start": 3.0, "end": 5.0, "speaker": 1},
        {"start": 5.0, "end": 8.0, "speaker": 0},
    ]
    words = [w(f"w{i}", float(i), float(i) + 0.9) for i in range(8)]
    blocks = d.assign_speakers(words, turns)
    assert len(blocks) == 1
    assert blocks[0]["speaker"] == "Speaker 1"
    assert blocks[0]["text"] == "w0 w1 w2 w3 w4 w5 w6 w7"


def test_run_of_three_is_not_smoothed():
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": 0},
        {"start": 3.0, "end": 6.0, "speaker": 1},
        {"start": 6.0, "end": 9.0, "speaker": 0},
    ]
    words = [w(f"w{i}", float(i), float(i) + 0.9) for i in range(9)]
    blocks = d.assign_speakers(words, turns)
    assert [b["speaker"] for b in blocks] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_smoothing_leaves_a_short_run_at_the_start_alone():
    # a short run at the very start has no left neighbour -> untouched
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": 1},
        {"start": 1.0, "end": 6.0, "speaker": 0},
    ]
    words = [w(f"w{i}", float(i), float(i) + 0.9) for i in range(6)]
    blocks = d.assign_speakers(words, turns)
    assert [b["speaker"] for b in blocks] == ["Speaker 2", "Speaker 1"]


def test_smoothing_leaves_a_short_run_at_the_end_alone():
    """Symmetric right-edge case: a short final run has no right neighbour."""
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 6.0, "speaker": 1},
    ]
    words = [w(f"w{i}", float(i), float(i) + 0.9) for i in range(6)]
    blocks = d.assign_speakers(words, turns)
    assert [b["speaker"] for b in blocks] == ["Speaker 1", "Speaker 2"]
    assert blocks[0]["text"] == "w0 w1 w2 w3 w4"
    assert blocks[1]["text"] == "w5"


def test_two_adjacent_short_runs_are_both_smoothed():
    """Run sequence X(5) Y(1) X(1) Y(5) -> exactly one X block then one Y block.

    Each short run is judged against the runs as they were *before* the pass:
    the lone Y is flanked by X/X so it becomes X, and the lone X is flanked by
    Y/Y so it becomes Y. Evaluating against already-relabelled neighbours would
    smooth the first and leave the second stranded as its own block.
    """
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": 0},  # X: w0..w4
        {"start": 5.0, "end": 6.0, "speaker": 1},  # Y: w5
        {"start": 6.0, "end": 7.0, "speaker": 0},  # X: w6
        {"start": 7.0, "end": 12.0, "speaker": 1},  # Y: w7..w11
    ]
    words = [w(f"w{i}", float(i), float(i) + 0.9) for i in range(12)]
    blocks = d.assign_speakers(words, turns)

    assert [b["speaker"] for b in blocks] == ["Speaker 1", "Speaker 2"]
    assert blocks[0]["text"] == "w0 w1 w2 w3 w4 w5"
    assert blocks[1]["text"] == "w6 w7 w8 w9 w10 w11"


def test_blocks_group_consecutive_words_and_carry_time_span():
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": 0},
        {"start": 4.0, "end": 8.0, "speaker": 1},
    ]
    words = [
        w("hello", 0.0, 0.5),
        w("there", 0.6, 1.2),
        w("friend", 1.3, 2.0),
        w("and", 4.0, 4.4),
        w("goodbye", 4.5, 5.1),
        w("now", 5.2, 5.8),
    ]
    blocks = d.assign_speakers(words, turns)
    assert blocks == [
        {"speaker": "Speaker 1", "start": 0.0, "end": 2.0, "text": "hello there friend"},
        {"speaker": "Speaker 2", "start": 4.0, "end": 5.8, "text": "and goodbye now"},
    ]


def test_faster_whisper_style_word_objects_are_accepted():
    class Word:
        def __init__(self, word, start, end):
            self.word = word
            self.start = start
            self.end = end

    turns = [{"start": 0.0, "end": 2.0, "speaker": 0}]
    blocks = d.assign_speakers([Word(" hello", 0.0, 0.5), Word(" world", 0.6, 1.2)], turns)
    assert blocks == [
        {"speaker": "Speaker 1", "start": 0.0, "end": 1.2, "text": "hello world"}
    ]


# --------------------------------------------------------------------------
# diarize
# --------------------------------------------------------------------------


LANG_PREFERENCE = ("kk", "ru", "en")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def _sample_pair() -> tuple[Path, Path] | None:
    """Two recordings from tests/audio/ that hold different speakers.

    Preferred (what the fixture plan produces): one FLEURS sample per language,
    e.g. kk_*.wav + ru_*.wav -- guaranteed different speakers. Falls back to any
    two distinct recordings in the directory so the test still runs if the
    fixtures are named differently.
    """
    if not AUDIO_DIR.is_dir():
        return None
    files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if len(files) < 2:
        return None

    by_lang: dict[str, Path] = {}
    for path in files:
        lang = path.name.split("_")[0]
        if lang in LANG_PREFERENCE:
            by_lang.setdefault(lang, path)
    picked = [by_lang[lang] for lang in LANG_PREFERENCE if lang in by_lang]
    if len(picked) >= 2:
        return picked[0], picked[1]
    return files[0], files[1]


needs_models = pytest.mark.skipif(
    not d.is_available(),
    reason="diarization models missing; run api/scripts/get_diarization_models.sh",
)
needs_audio = pytest.mark.skipif(
    _sample_pair() is None,
    reason="need at least two recordings in tests/audio/ (e.g. kk_*.wav and ru_*.wav)",
)


@pytest.fixture(scope="module")
def two_speaker_audio(tmp_path_factory) -> tuple[Path, float]:
    """Sample A + 0.5 s silence + sample B as a 16 kHz mono wav, and its duration."""
    import soundfile as sf
    from faster_whisper.audio import decode_audio

    pair = _sample_pair()
    assert pair is not None
    first = np.asarray(decode_audio(str(pair[0]), sampling_rate=16000), dtype=np.float32)
    second = np.asarray(decode_audio(str(pair[1]), sampling_rate=16000), dtype=np.float32)
    silence = np.zeros(int(SILENCE_SECONDS * 16000), dtype=np.float32)

    samples = np.concatenate([first, silence, second])
    path = tmp_path_factory.mktemp("diarize") / "two_speakers.wav"
    sf.write(str(path), samples, 16000, subtype="PCM_16")
    return path, samples.size / 16000


def _covered_seconds(turns: list[dict]) -> float:
    """Union of the turn intervals, so overlapping speech is not double counted."""
    covered = 0.0
    cursor = 0.0
    for turn in sorted(turns, key=lambda t: t["start"]):
        start = max(turn["start"], cursor)
        if turn["end"] > start:
            covered += turn["end"] - start
            cursor = turn["end"]
    return covered


@needs_models
@needs_audio
def test_diarize_finds_two_speakers(two_speaker_audio):
    path, duration = two_speaker_audio
    turns = d.diarize(str(path))

    assert turns, "expected at least one speaker turn"
    assert turns == sorted(turns, key=lambda t: t["start"])
    assert all(t["end"] > t["start"] for t in turns)
    assert len({t["speaker"] for t in turns}) >= 2

    assert _covered_seconds(turns) >= 0.6 * duration


@needs_models
@needs_audio
def test_diarize_honours_num_speakers(two_speaker_audio):
    turns = d.diarize(str(two_speaker_audio[0]), num_speakers=2)
    assert len({t["speaker"] for t in turns}) == 2


@needs_models
@needs_audio
def test_diarize_accepts_a_numpy_array(two_speaker_audio):
    from faster_whisper.audio import decode_audio

    samples = np.asarray(
        decode_audio(str(two_speaker_audio[0]), sampling_rate=16000), dtype=np.float32
    )
    turns = d.diarize(samples, sr=16000, num_speakers=2)
    assert len({t["speaker"] for t in turns}) == 2


@needs_models
def test_diarize_returns_nothing_for_empty_audio():
    assert d.diarize(np.zeros(0, dtype=np.float32)) == []


@needs_models
def test_diarize_rejects_a_wrong_sample_rate():
    with pytest.raises(ValueError, match="16000"):
        d.diarize(np.zeros(8000, dtype=np.float32), sr=8000)


def test_is_available_checks_both_model_files():
    assert d.is_available() == (
        d.SEGMENTATION_MODEL.is_file() and d.EMBEDDING_MODEL.is_file()
    )
