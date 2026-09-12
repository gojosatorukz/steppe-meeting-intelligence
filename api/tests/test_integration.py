"""End-to-end tests for the assembled service: upload -> protocol -> exports.

Three groups, with different costs:

* **Free everywhere** — the offline guard (pure socket monkeypatching) and the
  shape of `/health`. These run in CI.
* **Needs the STT weights** — the Hugging Face offline check, which is the whole
  point of `offline_guard`: it proves the first transcription makes no outbound
  request at all, rather than making one that fails quietly.
* **Needs Ollama and the models** — the full upload and the team recording with
  diarization. `skipif`-guarded so a laptop without the weights still gets a
  green suite, but on the demo machine they *do* run, and they are the ones
  that would catch a broken stage demo.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import offline_guard  # noqa: E402  # arms the guard for this process, as main does
import diarize  # noqa: E402
import exports  # noqa: E402
import pipeline  # noqa: E402

API_DIR = Path(__file__).resolve().parents[1]
AUDIO = API_DIR / "tests" / "audio"
RU_1 = AUDIO / "ru_1.wav"
RU_2 = AUDIO / "ru_2.wav"
TEAM = AUDIO / "team_mixed_35.m4a"
PARTICIPANTS = ["Жаныбек", "Бекзат", "Айгерим"]


def ollama_ready() -> bool:
    """True when Ollama is up *and* one of our candidate models is pulled."""
    try:
        pipeline.resolved_llm_model()
        return True
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(not ollama_ready(), reason="Ollama or an LLM model is missing")
def stt_ready() -> bool:
    """True when the configured STT model is a local path or cached snapshot."""
    configured = Path(pipeline.TURBO_MODEL_ID)
    if configured.is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=pipeline.TURBO_MODEL_ID, local_files_only=True)
        return True
    except Exception:
        return False


needs_stt = pytest.mark.skipif(
    not RU_2.is_file() or not stt_ready(),
    reason="test audio or local STT weights are missing",
)
needs_diarizer = pytest.mark.skipif(
    not diarize.is_available() or not TEAM.is_file(),
    reason="diarization models or the team recording are missing",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import main

    # `with` so the lifespan runs: that is what warms the models and sets `warm`.
    with TestClient(main.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Offline guard
# ---------------------------------------------------------------------------

def test_outbound_connect_raises_and_is_counted():
    """The requirement is "no request leaves this machine" — so prove it."""
    assert offline_guard.enabled(), "the guard must be armed by default"
    before = offline_guard.blocked_count()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(offline_guard.OfflineBlocked):
            sock.connect(("93.184.216.34", 80))     # example.com, a real route out
    finally:
        sock.close()

    assert offline_guard.blocked_count() == before + 1

    with pytest.raises(offline_guard.OfflineBlocked):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    assert offline_guard.blocked_count() == before + 2


def test_loopback_and_unix_sockets_stay_open():
    """Ollama lives on 127.0.0.1:11434 and the WebSocket on localhost."""
    for address in [("127.0.0.1", 11434), ("localhost", 11434), ("::1", 11434),
                    ("127.0.0.53", 53), "/run/some.sock", ("", 8000)]:
        assert offline_guard.is_loopback(address), address
    for address in [("8.8.8.8", 53), ("huggingface.co", 443), ("example.com", 80),
                    ("2001:4860:4860::8888", 53)]:
        assert not offline_guard.is_loopback(address), address

    before = offline_guard.blocked_count()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client_sock = socket.create_connection(listener.getsockname(), timeout=2)
    client_sock.close()
    listener.close()
    assert offline_guard.blocked_count() == before, "a loopback connect must not be blocked"


def test_the_guard_sets_the_hugging_face_offline_switches():
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


@needs_stt
def test_transcription_makes_no_outbound_request_at_all():
    """T3's must-fix: the turbo model is loaded by hub id.

    Without `HF_HUB_OFFLINE`, `huggingface_hub` answers that with a revision
    check against huggingface.co, which with the wifi off does not fail fast —
    it hangs. A guard that merely *blocks* the call is not enough; the call must
    not be attempted. So: transcribe, and assert the counter never moved.

    Run in a subprocess because the guard's counter is process-global and the
    STT models may already be loaded in this one, which would hide the call.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(API_DIR)!r})
        import offline_guard
        import pipeline
        tr = pipeline.transcribe({str(RU_2)!r})
        assert tr.text.strip(), "empty transcript"
        print("BLOCKED", offline_guard.blocked_count())
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=600, cwd=API_DIR)
    assert proc.returncode == 0, proc.stderr[-3000:]
    blocked = int(proc.stdout.strip().splitlines()[-1].split()[-1])
    assert blocked == 0, (
        f"{blocked} outbound connection(s) attempted during the first transcription — "
        "HF_HUB_OFFLINE is not taking effect early enough"
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_reports_every_moving_part(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()

    assert h["ok"] is True and h["offline"] is True
    assert set(h["stt_models"]) == {"turbo", "kk", "kk_available"}
    assert h["stt_models"]["turbo"] == pipeline.WHISPER_SIZE
    assert h["stt_device"] in {"cuda", "cpu", "mixed", "degraded"}
    assert isinstance(h["stt_devices"], dict) and isinstance(h["stt_compute_types"], dict)
    assert "gpu_degraded_reason" in h                        # None is the healthy answer
    assert h["diarization"]["available"] == diarize.is_available()
    assert h["diarization"]["threshold"] > 0
    assert isinstance(h["vad_model_present"], bool)
    assert h["offline_guard"] == {"enabled": True,
                                  "blocked_external_connections": offline_guard.blocked_count()}
    assert h["warm"] is True, "the lifespan must have warmed the models"
    assert isinstance(h["results_stored"], int) and h["results_stored"] >= 0
    # The resolved model, not the static constant: /health must not claim a
    # model nobody has pulled.
    if ollama_ready():
        assert h["llm_model"] == pipeline.resolved_llm_model()
        assert h["llm_error"] is None
    else:
        assert h["llm_model"] is None and h["llm_error"]


def test_health_llm_model_is_resolved_not_the_constant(client, monkeypatch):
    monkeypatch.setattr(pipeline, "resolved_llm_model", lambda: "some-other-model:1b")
    assert client.get("/health").json()["llm_model"] == "some-other-model:1b"


def test_health_survives_a_missing_llm(client, monkeypatch):
    def boom() -> str:
        raise RuntimeError("No supported LLM model is pulled in Ollama.")

    monkeypatch.setattr(pipeline, "resolved_llm_model", boom)
    h = client.get("/health").json()
    assert h["ok"] is True and h["llm_model"] is None
    assert "No supported LLM model" in h["llm_error"]


# ---------------------------------------------------------------------------
# Upload -> protocol -> exports
# ---------------------------------------------------------------------------

@needs_ollama
@needs_stt
def test_upload_returns_a_protocol_and_all_three_exports(client):
    with RU_1.open("rb") as fh:
        r = client.post("/process", files={"file": ("ru_1.wav", fh, "audio/wav")},
                        data={"participants": "Бекзат, Айгерим", "speakers": "auto"})
    assert r.status_code == 200, r.text
    result = r.json()

    assert result["transcript"].strip()
    assert result["language"] in {"ru", "kk", "en"}
    assert result["languages"] and abs(sum(result["languages"].values()) - 1.0) < 0.05
    assert result["participants"] == ["Бекзат", "Айгерим"]
    protocol = result["protocol"]
    for key in ("summary", "decisions", "topics", "open_questions", "risks", "action_items"):
        assert key in protocol, f"protocol is missing {key}"
    for key in ("transcribe_sec", "diarize_sec", "llm_sec", "total_sec", "realtime_factor"):
        assert key in result["timing"], f"timing is missing {key}"

    rid = result["id"]
    assert client.get(f"/result/{rid}").status_code == 200
    assert any(x["id"] == rid for x in client.get("/results").json())

    pdf = client.get(f"/result/{rid}/export.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    csv = client.get(f"/result/{rid}/export.csv")
    assert csv.status_code == 200 and "Ответственный" in csv.content.decode("utf-8-sig")
    js = client.get(f"/result/{rid}/export.json")
    assert js.status_code == 200 and js.json()["transcript"] == result["transcript"]

    print(f"\n[ru_1.wav] language={result['language']} languages={result['languages']} "
          f"timing={result['timing']}")


def test_upload_rejects_an_unsupported_format(client):
    r = client.post("/process", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "Unsupported format" in r.json()["detail"]


def test_allowed_formats_include_voice_note_and_phone_recording_extensions():
    """F1: Telegram/WhatsApp voice notes (.oga/.opus) and phone recordings
    (.mov/.amr/...) are decodable by PyAV and must not 400 on the extension
    check alone."""
    import main

    for ext in (".oga", ".opus", ".aac", ".mov", ".mkv", ".amr", ".wma", ".aiff"):
        assert ext in main.ALLOWED, ext


def test_upload_with_an_oga_extension_does_not_400_on_the_extension_check(client):
    """The extension check must pass for .oga; a 500 from decoding fake bytes
    (this is not a real Ogg/Opus file) is acceptable — only 400 is not."""
    r = client.post("/process", files={"file": ("voice.oga", b"not really audio", "audio/ogg")})
    assert r.status_code != 400, r.text


def test_upload_rejects_a_nonsense_speaker_count(client):
    with RU_1.open("rb") as fh:
        r = client.post("/process", files={"file": ("ru_1.wav", fh, "audio/wav")},
                        data={"speakers": "many"})
    assert r.status_code == 400 and "speakers" in r.json()["detail"]


# ---------------------------------------------------------------------------
# F3: a broken LLM must not lose a finished transcript/diarization
# ---------------------------------------------------------------------------

def test_process_audio_survives_a_broken_llm_and_keeps_the_transcript(monkeypatch):
    """structure() raising must not make process_audio raise: the transcript
    (and any diarization) is already finished work and must not be discarded.

    Weight-free: both `transcribe` and `structure` are monkeypatched, so this
    runs with no STT weights and no Ollama present.
    """
    fake_segment = pipeline.Segment(start=0.0, end=1.0, text="hello world",
                                    language="en", words=[])
    fake_tr = pipeline.Transcript(text="hello world", language="en",
                                  languages={"en": 1.0}, seconds=1.0,
                                  segments=[fake_segment])
    monkeypatch.setattr(pipeline, "transcribe", lambda path, participants=None: fake_tr)

    def boom(transcript, model=None, language_hint=None):
        raise RuntimeError("LLM is on fire")

    monkeypatch.setattr(pipeline, "structure", boom)

    result = pipeline.process_audio("unused.wav", participants=PARTICIPANTS)

    assert result["transcript"] == "hello world"
    assert result["protocol"] == pipeline.EMPTY
    assert result["protocol"] is not pipeline.EMPTY, "EMPTY must be copied, not shared"
    assert set(result["protocol"]) == {
        "summary", "decisions", "topics", "open_questions", "action_items", "risks"
    }
    assert all(v == [] for v in result["protocol"].values())
    assert result["protocol_error"] == "LLM is on fire"


# ---------------------------------------------------------------------------
# Protocol language selector
# ---------------------------------------------------------------------------

def test_process_text_sends_the_chosen_protocol_language_to_the_llm(client, monkeypatch):
    """The selector is an order, not a hint: whatever was spoken, ru means ru."""
    seen: dict = {}

    def fake_structure(transcript, model=None, language_hint=None):
        seen["hint"] = language_hint
        return dict(pipeline.EMPTY)

    monkeypatch.setattr(pipeline, "structure", fake_structure)

    r = client.post("/process-text", json={
        "transcript": "Бюджет утверждён. Айгерим готовит план до среды.",
        "protocol_language": "ru",
    })
    assert r.status_code == 200, r.text
    assert seen["hint"] == "ru"
    assert r.json()["protocol_language"] == "ru"


def test_process_text_defaults_to_russian_without_the_field(client, monkeypatch):
    seen: dict = {}

    def fake_structure(transcript, model=None, language_hint=None):
        seen["hint"] = language_hint
        return dict(pipeline.EMPTY)

    monkeypatch.setattr(pipeline, "structure", fake_structure)

    r = client.post("/process-text", json={"transcript": "Kazakh meeting text"})
    assert r.status_code == 200, r.text
    assert seen["hint"] == "ru"


def test_live_start_accepts_a_protocol_language(client, monkeypatch):
    import live

    started: dict = {}

    def fake_start(device=None, participants=(), num_speakers=None,
                   protocol_language="ru"):
        started.update(device=device, participants=list(participants),
                       num_speakers=num_speakers, protocol_language=protocol_language)
        return SimpleNamespace(device_name="fake mic", participants=list(participants),
                               num_speakers=num_speakers,
                               protocol_language=protocol_language)

    monkeypatch.setattr(live, "start_session", fake_start)

    r = client.post("/live/start", json={"participants": ["Бекзат"],
                                         "protocol_language": "kk"})
    assert r.status_code == 200, r.text
    assert started["protocol_language"] == "kk"
    assert r.json()["protocol_language"] == "kk"


# ---------------------------------------------------------------------------
# Diarization on the real two-speaker recording
# ---------------------------------------------------------------------------

@needs_ollama
@needs_diarizer
def test_team_recording_gets_two_speakers_and_names_the_owner():
    """The stage demo, in one test.

    77 s of RU/KK/EN with two people. With `speakers=2` the diarizer is told the
    answer, so the only question is whether the blocks and the protocol line up —
    and whether the LLM picks a real participant as an owner rather than a
    pronoun.
    """
    result = pipeline.process_audio(TEAM, participants=PARTICIPANTS, num_speakers=2)

    blocks = result.get("speakers") or []
    assert blocks, "diarization produced no speaker blocks"
    labels = {b["speaker"] for b in blocks}
    assert len(labels) == 2, f"expected 2 speakers, got {sorted(labels)}"
    assert all(b["text"].strip() for b in blocks)
    assert all(b["start"] <= b["end"] for b in blocks)
    assert result["num_speakers"] == 2
    assert any(s["speaker"] in labels for s in result["segments"]), \
        "segments were not stamped with a speaker"
    assert result["timing"]["diarize_sec"] > 0

    owners = " ".join(a.get("owner", "") for a in result["protocol"]["action_items"])
    assert "Бекзат" in owners, f"no action item owned by Бекзат; owners were: {owners!r}"

    # The lead asked to see these.
    print(f"\n[team_mixed_35.m4a] timing={result['timing']} "
          f"language={result['language']} languages={result['languages']}")
    for b in blocks:
        print(f"  [{b['start']:6.2f}-{b['end']:6.2f}] {b['speaker']}: {b['text'][:110]}")
    print("  action items:")
    for a in result["protocol"]["action_items"]:
        print(f"    {a.get('owner')} — {a.get('task')} (срок {a.get('due')}, {a.get('priority')})")

    # And the exports have to carry the speakers through.
    pdf = exports.to_pdf(result)
    assert pdf[:4] == b"%PDF"
    import json as _json
    assert _json.loads(exports.to_json(result))["speakers"] == blocks


# ---------------------------------------------------------------------------
# Exports with speakers, without touching a model
# ---------------------------------------------------------------------------

SPEAKER_RESULT = {
    "transcript": "Мы согласовали R&D бюджет. <неразборчиво>",
    "language": "ru",
    "audio_seconds": 12.0,
    "timing": {"total_sec": 3.0},
    "speakers": [
        {"speaker": "Speaker 1", "start": 0.0, "end": 4.2, "text": "Мы согласовали R&D бюджет."},
        {"speaker": "Speaker 2", "start": 4.2, "end": 9.9, "text": "<неразборчиво> до пятницы."},
    ],
    "protocol": {
        "summary": ["Бюджет R&D согласован"], "decisions": [], "topics": [],
        "open_questions": [], "risks": [],
        "action_items": [{"owner": "Бекзат", "task": "Свести смету & отправить",
                          "due": "пятница", "priority": "high"}],
    },
}


def test_pdf_renders_speaker_blocks_and_survives_markup_characters():
    pdf = exports.to_pdf(SPEAKER_RESULT)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500


def test_pdf_falls_back_to_the_plain_transcript_without_speakers():
    plain = {k: v for k, v in SPEAKER_RESULT.items() if k != "speakers"}
    assert exports.to_pdf(plain)[:4] == b"%PDF"


def test_json_export_carries_the_speaker_blocks():
    import json

    assert json.loads(exports.to_json(SPEAKER_RESULT))["speakers"] == SPEAKER_RESULT["speakers"]


def test_csv_export_is_unchanged_by_speakers():
    rows = exports.to_csv(SPEAKER_RESULT).decode("utf-8-sig").splitlines()
    assert rows[0] == "Ответственный,Задача,Срок,Приоритет"
    assert rows[1].startswith("Бекзат,")
