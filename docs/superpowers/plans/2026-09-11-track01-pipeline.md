# Track 01 — AI Meeting Intelligence: local pipeline plan

**Spec (binding authority):** the organizers' TZ, `TZ_AI_Meeting_Intelligence.pdf` (Must-Haves: local
transcription of MP3/WAV/M4A; executive summary 3–5 sentences; decisions / topics / open questions;
action-items table owner·task·due·priority; export csv/json/pdf; 100% local. Nice-to-have: diarization,
RU/KZ/EN incl. code-switching, RAG chat, integrations, risk detection. Scoring: functionality 30, AI
accuracy 20, locality 20, bonus features 15, UI/UX + speed on a 2-minute recording 10, presentation 5.)

**Goal of this plan:** batch pipeline with a Kazakh-capable STT router; live mic transcription that scrolls
in a browser and produces the protocol on "Finish"; batch diarization; LLM switched to `qwen3.5:4b`.

**Repo:** `/home/datamata/Projects/aisteppetechhack`, service in `api/` (FastAPI, managed by `uv`; run
everything as `uv run ...` from `api/`). Existing files: `api/pipeline.py` (transcribe + structure),
`api/exports.py` (csv/json/pdf, Cyrillic font registered), `api/main.py` (endpoints + single-page UI).

## Global Constraints (bind every task)

1. **100% offline at runtime.** No network call in any request path. No `anthropic`, `openai`, or any
   hosted-API client may be added. Model weights are downloaded once, before the demo.
2. **STT runs on CPU:** `WhisperModel(..., device="cpu", compute_type="int8", cpu_threads=8)`. CUDA for
   CTranslate2 is broken on this machine — do not try to fix it. The LLM runs on the GPU via Ollama
   (`localhost:11434`, already installed, v0.34).
3. **Languages:** Russian, Kazakh, English, mixed. All user-facing strings and exports must render
   Cyrillic correctly (the PDF font is already handled in `exports.py`).
4. **Tests** live in `api/tests/`, run with `uv run pytest -q` from `api/`. Tests must never require a
   microphone or GPU. Tests that need model weights or a running Ollama use `pytest.mark.skipif` with a
   clear reason, and the implementer's report must state whether they actually ran (not just passed by
   skipping).
5. **Commit only the files your task owns** (`git add <explicit paths>`; never `git add -A` — other
   uncommitted work may be present in the tree). Never commit anything under `api/models/` or audio
   over 2 MB. Commit messages: imperative subject, body says why.
6. **Do not edit** `docs/`, `README.md`, `CLAUDE.md`, `Makefile`, `scripts/` — the lead owns those.
7. **Python contingency:** if a required package has no wheel for the venv's Python, run
   `uv python pin 3.12 && uv sync` in `api/`, say so in the report, and continue.
8. Keep FastAPI; no new web frameworks. Keep the UI dependency-free (inline HTML/CSS/JS in `main.py`).

## Shared contracts (data shapes used across tasks)

```python
# word / turn / block are plain dicts so tasks stay decoupled:
word  = {"start": float, "end": float, "text": str}                      # produced by T1
turn  = {"start": float, "end": float, "speaker": int}                   # produced by T4 (0-based)
block = {"speaker": "Speaker 1", "start": float, "end": float, "text": str}   # produced by T4.assign_speakers

# pipeline.py after T1:
@dataclass
class Segment:  start: float; end: float; text: str; language: str; words: list[dict]; speaker: str | None = None
@dataclass
class Transcript: text: str; language: str; segments: list[Segment]; seconds: float; languages: dict[str, float]
def transcribe(audio_path) -> Transcript                       # batch, routed
def transcribe_pcm(pcm: np.ndarray, sr: int = 16000) -> Transcript   # one utterance, routed (live path)
def structure(transcript: str, model: str | None = None, language_hint: str | None = None) -> dict   # T2
def process_audio(path) -> dict   # result dict; keys: transcript, language, languages, audio_seconds,
                                  # segments (with words + speaker), protocol, timing, (speakers after T5)
# diarize.py (T4):
def is_available() -> bool
def diarize(audio: str | np.ndarray, sr: int = 16000, num_speakers: int | None = None) -> list[dict]  # turns
def assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]   # blocks
```

---

### Task 1: STT routing — Kazakh fine-tune + turbo, VAD-first, constrained language ID

**Owns:** `api/pipeline.py` (transcription half only — do NOT touch `structure()`/LLM code),
`api/tests/test_stt.py`.

**Models (CPU, int8, `cpu_threads=8`):**
- RU/EN: `deepdml/faster-whisper-large-v3-turbo-ct2` (HF hub id; faster-whisper downloads to the HF cache).
- KK: the CTranslate2 conversion of `shyngys879/kazakh-whisper-large-v3-turbo`, located at the path given
  in the dispatch context (expected `api/models/kk-turbo-ct2`). Its card says it does NOT retain RU/EN —
  it must only ever receive Kazakh audio.
- If the KK model directory is missing at import time: log a warning once and route everything to the
  turbo model (degraded but working). `is_kk_available()` must exist for `/health`.
- Load lazily, cache at module level (loading per request costs seconds we are scored on).

**Routing algorithm (batch `transcribe(path)`):**
1. Decode audio to 16 kHz mono float32 (`faster_whisper.audio.decode_audio`).
2. Speech chunks via `faster_whisper.vad.get_speech_timestamps` (VadOptions: `min_silence_duration_ms=500`,
   `speech_pad_ms=200`). Group consecutive chunks into windows of at most 30 s (never split a chunk).
3. Per window: `lang, prob, all_probs = turbo.detect_language(window_pcm)`; take the argmax **restricted to
   {kk, ru, en}** from `all_probs` (`all_language_probs` list of (lang, prob) — use whichever the installed
   faster-whisper returns; verify the signature). If kk → `kk_model.transcribe(window, language="kk", ...)`;
   else `turbo.transcribe(window, language=lang, ...)`.
4. transcribe kwargs everywhere: `beam_size=5, word_timestamps=True, condition_on_previous_text=False,
   vad_filter=False` (already VAD-gated), `temperature=0.0`.
5. Offset all segment/word timestamps by the window start. Collect `Segment` objects with `language`.
6. **Hallucination filter:** drop segments with `no_speech_prob > 0.6`; drop a segment whose text (stripped,
   lowercased) equals the previous one; drop segments containing any of: "субтитры", "продолжение следует",
   "редактор субтитров", "amara.org", "thank you for watching", "подписывайтесь".
7. `Transcript.languages` = share of words per language; `Transcript.language` = dominant.
8. `transcribe_pcm(pcm, sr)` — same steps 3–7 on a single utterance array (no VAD step; assume ≥ 0.5 s).
   Used by the live path. Must return within a few seconds for a 6 s utterance on this CPU — report the
   measured time.
9. **Participants hint:** `transcribe(path, participants: list[str] | None = None)` and
   `transcribe_pcm(..., participants=None)`: when given, pass `initial_prompt=", ".join(participants)` to every
   `transcribe()` call (and `hotwords=` too if the installed faster-whisper accepts it — check the signature).
   Baseline evidence: without it, "Бекзат" is transcribed as "Пикзат", which becomes a wrong action-item owner.
   `process_audio(path, participants=None)` forwards it and stores `result["participants"]`.
10. Keep `process_audio()` and `process_text()` working; `process_audio` now emits `languages` and per-
   segment `language`, `words`, `speaker: null`.

**Tests (`api/tests/test_stt.py`):** use `api/tests/audio/kk_*.wav`, `ru_*.wav`, `en_*.wav` with their
`.txt` references (created by the verification step; `skipif` missing). For each: routed language matches
the sample's language; word-level recall against the reference (lowercase, strip punctuation) ≥ 0.5 for
kk and ≥ 0.7 for ru/en; print timing. One unit test for the hallucination filter with synthetic segments.
One test for the constrained argmax helper with a fake `all_probs` where `tt` (Tatar) outranks `kk`.
Real-recording test (`skipif` missing): `api/tests/audio/team_mixed_35.m4a` (77 s; RU + KK + EN, two
speakers; script in the dispatch context) with `participants=["Жаныбек", "Бекзат", "Айгерим"]` → the
transcript contains "Бекзат" (not "Пикзат"), contains at least one Kazakh-specific letter (ә ғ қ ң ө ұ ү і),
contains the English words "London" and "Wednesday" untranslated, and `languages` has all of kk, ru, en > 0.
Print the full transcript and timing in the test output for the lead.

**Commit:** `Route STT by language: Kazakh fine-tune for kk, turbo for ru/en`.

---

### Task 2: LLM switch to qwen3.5:4b with safe fallbacks

**Owns:** `api/pipeline.py` (`structure()`/LLM half only — do NOT touch transcription code),
`api/tests/test_llm.py`, `api/tests/fixtures/transcript_ru.txt`, `api/tests/fixtures/transcript_kk.txt`.

**Requirements:**
1. Model selection: `LLM_MODEL = os.getenv("LLM_MODEL")` or, if unset, the first of
   `["qwen3.5:4b", "qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:7b-instruct"]` present in `ollama.list()`
   (resolve once, cache; expose `resolved_llm_model()` for `/health`). If none present, raise a clear
   `RuntimeError` naming the pull command.
2. Call: `ollama.chat(model=..., messages=..., format=PROTOCOL_SCHEMA, keep_alive="30m",
   options={"num_ctx": 16384, "temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.0,
   "num_predict": 2048, "seed": 0})` and `think=False`. Ollama's default context silently truncates the
   FRONT of long inputs — `num_ctx` is mandatory.
3. `think=False` may be rejected by non-thinking models: catch the error, retry once without `think`,
   remember per model. Verify actual behaviour against the running Ollama and state it in the report.
4. `structure(transcript, model=None, language_hint=None)`: if `language_hint` in {ru, kk, en}, append a
   system line "Язык протокола: <русский|қазақ тілі|English>" so output language follows the meeting.
   Otherwise keep the existing "answer in the meeting's language" rule.
5. Keep chunk+merge for long transcripts. Keep `EMPTY` fallback on JSON parse failure, but log the raw
   content head so failures are diagnosable.
6. Report measured wall-clock for the RU and KK fixtures on this GPU.

**Fixtures (write verbatim):**
`transcript_ru.txt`:
```
Алексей: Коллеги, начнём. Первый вопрос — запуск мобильного приложения. Мария, какой статус?
Мария: Бэкенд готов, фронт задерживается на неделю. Предлагаю перенести релиз на 25 сентября.
Алексей: Согласен, переносим на 25-е. Дамир, подготовь обновлённый план тестирования до пятницы.
Дамир: Сделаю. Но есть риск — тестовый стенд нестабилен, нужен доступ к продовой базе.
Алексей: Вопрос с доступом пока открытый, обсудим с безопасностью. Второе — бюджет на маркетинг. Мария, оцени стоимость кампании до среды, приоритет высокий.
Мария: Хорошо.
```
`transcript_kk.txt`:
```
Айгерім: Сәлеметсіздер ме, бастайық. Бірінші мәселе — жаңа сайтты іске қосу. Бекзат, жағдай қалай?
Бекзат: Дизайн дайын, бірақ төлем жүйесін қосу кешігіп жатыр. Іске қосуды 30 қыркүйекке ауыстыруды ұсынамын.
Айгерім: Келістік, 30 қыркүйекке ауыстырамыз. Бекзат, жұма күніне дейін жаңартылған жоспарды дайында.
Бекзат: Жарайды. Бір тәуекел бар — серверге қолжетімділік әлі шешілмеген.
Айгерім: Бұл сұрақ ашық қалады, IT бөлімімен талқылаймыз. Екінші — маркетинг бюджеті. Айдана, сәрсенбіге дейін науқанның құнын бағала, басымдығы жоғары.
Айдана: Жақсы.
```

**Tests (`test_llm.py`, `skipif` Ollama unreachable or no model):** for each fixture, `structure()` returns
all six keys; `summary` has 3–5 items; `action_items` non-empty and every `owner` appears in the fixture
text (no invented people); at least one decision mentions the moved date (25/30); output for the RU
fixture contains Cyrillic; KK fixture output contains at least one Kazakh-specific letter (ә, ғ, қ, ң, ө,
ұ, ү, і). Print wall-clock. Unit test for model resolution with a mocked `ollama.list`.

**Commit:** `Switch LLM to qwen3.5:4b with explicit context and model fallback`.

---

### Task 3: Live transcription — mic → VAD → routed STT → WebSocket → Finish

**Owns:** `api/live.py` (new), `api/main.py` (add endpoints + `/live` page; do not change existing
endpoints' behaviour), `api/tests/test_live.py`.

**Design (server-side capture; the browser only displays):**
- Dependencies: `sounddevice`, `silero-vad` (pip name `silero-vad`), `numpy`. PortAudio is installed.
- `LiveSession` (one at a time; a second `start` returns 409):
  - `sounddevice.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512, device=...)`.
    All raw PCM is also appended to an in-memory buffer and written to a WAV on stop.
  - Silero `VADIterator(model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=600,
    speech_pad_ms=200)`; utterance buffer; FINALIZE when ≥1 s of speech ended by 600 ms silence, or when
    the buffer reaches 7 s (forced; keep the last 300 ms as overlap for the next buffer).
  - Each finalized utterance → `pipeline.transcribe_pcm()` in a worker thread (never on the audio
    callback thread; never hold the loop) → segments appended to the session transcript → broadcast JSON
    `{"type":"segment","text","language","start","end"}` to all WebSocket clients. Also broadcast
    `{"type":"status", ...}` on start/stop and `{"type":"level","rms":...}` at ~5 Hz for a mic meter.
  - Segment timestamps are session-relative (seconds since start).
- Endpoints:
  - `POST /live/start` body `{"device": int|null, "num_speakers": int|null}` → `{"ok":true,"device":name}`.
    If no input device / PortAudio error → 400 with the error text.
  - `POST /live/stop` → stops capture, saves WAV to a temp dir, builds a result via the same shape as
    `process_audio` (transcript from the live segments — do NOT re-transcribe), runs
    `diarize.diarize()` + `assign_speakers()` if `diarize.is_available()` (see T4 contract; import
    guarded so T3 works if diarize.py is absent), then `pipeline.structure(text, language_hint=dominant)`;
    stores in `RESULTS`, returns the result (id included) so the page can render the protocol and export links.
  - `WS /live/ws` → replays the current transcript on connect, then streams events.
  - `GET /mic-test` → records 2 s from the default device, returns `{"device": name, "rms": float,
    "peak": float, "ok": rms > 0.005}`.
  - `GET /live` → HTML page: big scrolling transcript (newest at bottom, auto-scroll, language badge per
    line: RU/KK/EN), mic level bar, Start / Finish buttons, a "🔒 offline" badge, and after Finish the same
    protocol rendering as the index page plus export links. Cyrillic-safe.
- Hallucination handling is T1's; T3 only skips empty texts.
- Log per-utterance latency (utterance end → segment broadcast) at INFO; report the measured value.

**Tests (`test_live.py`, no mic):** unit-test the utterance state machine with synthetic frames (speech
flags injected via a stub VAD): 1 s speech + 700 ms silence → one finalize; 8 s continuous speech →
forced finalize at 7 s with 300 ms carried over. `TestClient`: `POST /live/start` with an invalid device
returns 400 with a message; `/live` returns 200 HTML. WebSocket replay test with a fake session.

**Commit:** `Add live mic transcription with VAD gating and WebSocket transcript`.

---

### Task 4: Batch speaker diarization (sherpa-onnx) + word-level speaker assignment

**Owns:** `api/diarize.py` (new), `api/tests/test_diarize.py`, `api/scripts/get_diarization_models.sh` (new).

**Requirements:**
1. `uv add sherpa-onnx`. Model files (NOT committed) under `api/models/diarization/`:
   - segmentation: `sherpa-onnx-pyannote-segmentation-3-0` from
     `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`
   - embedding: `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` from
     `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
     (note the upstream typo "recongition" in the URL). Write the download script; run it; report sizes.
2. `is_available()` → both model files present.
3. `diarize(audio, sr=16000, num_speakers=None)`: accept a file path (decode with
   `faster_whisper.audio.decode_audio`, 16 kHz mono) or a float32 numpy array. Build
   `sherpa_onnx.OfflineSpeakerDiarization` with `num_clusters=num_speakers` when given, else
   `threshold`-based clustering (default 0.5); `min_duration_on=0.3, min_duration_off=0.5`,
   `num_threads=4`. Return turns `[{"start","end","speaker"}]` sorted by start. Verify the exact config
   class/field names against the installed package (`python -c "import sherpa_onnx; help(...)"`) — the
   upstream example `python-api-examples/offline-speaker-diarization.py` is the reference.
4. `assign_speakers(words, turns)`: for each word choose the speaker with maximum time overlap; if no
   overlap, inherit the previous word's speaker (first word: nearest turn). Smooth: any run of < 3 words
   whose neighbours on both sides share a speaker takes that speaker. Group consecutive same-speaker
   words into blocks `{"speaker": f"Speaker {n+1}", "start", "end", "text"}` (text = words joined by
   space). Empty inputs → `[]`.
5. Measure and report wall-clock for the longest available test file on CPU.

**Tests (`test_diarize.py`):** `assign_speakers` unit tests with hand-built words/turns (overlap choice,
gap inheritance, smoothing, grouping). `diarize()` on a synthetic 2-speaker file built by concatenating two
different `api/tests/audio/*.wav` samples if present (`skipif` missing or models missing): returns ≥ 2
distinct speakers and turns cover most of the speech.

**Commit:** `Add offline speaker diarization and word-level speaker assignment`.

---

### Task 5: Integration, offline guard, health, exports with speakers

**Owns:** `api/main.py`, `api/exports.py`, `api/pipeline.py` (`process_audio` only), `api/offline_guard.py`
(new), `api/tests/test_integration.py`.

**Requirements:**
1. `process_audio()` and `/process`: after transcription, if `diarize.is_available()`, run diarization on
   the same decoded audio (`num_speakers` from an optional form field, default auto) and attach
   `result["speakers"] = blocks`; set each segment's `speaker` by majority of its words.
2. Exports: PDF transcript section renders "Speaker N:" blocks when present (else the plain transcript);
   JSON includes `speakers`; CSV unchanged.
3. Index page: show speaker-labelled transcript when present; add a "speakers: auto / 2 / 3 / 4" select
   on the upload form; show `languages` shares as badges.
4. `offline_guard.py`: when env `OFFLINE_GUARD=1` (default ON), monkeypatch `socket.getaddrinfo` /
   `socket.socket.connect` at import time in `main.py` so any connection to a non-loopback address raises
   and is counted; `/health` reports `{"offline_guard": true, "blocked_external_connections": n}`. Ollama
   on 127.0.0.1 and the WebSocket must keep working. Cover with a test that an outbound connect raises.
5. **Offline model loading (must-fix, found in T3):** the first transcription calls huggingface.co for a
   revision check because the turbo model is loaded by hub id. Set `os.environ.setdefault("HF_HUB_OFFLINE", "1")`
   (and `HF_HUB_DISABLE_TELEMETRY=1`) in `offline_guard.py` before any `faster_whisper`/`huggingface_hub`
   import, and make `main.py` import `offline_guard` first. Add a test: with the guard active, a
   `transcribe()` of `tests/audio/ru_2.wav` completes and `blocked_external_connections` stays 0 — i.e.
   no outbound attempt was even made. Also call `pipeline.warm_up()` (T6) and, if available, a diarizer
   warm-up in the FastAPI lifespan so the first stage request pays no model load.
6. `/health`: add `stt_models` (turbo path, kk path + availability), `llm_model` (resolved), `diarization`
   availability, `offline_guard` info.
7. End-to-end test with `TestClient`: upload `api/tests/audio/ru_1.wav` (skipif missing / no Ollama) →
   200, protocol keys present, `languages` present, exports return 200 for all three formats and the PDF
   bytes start with `%PDF`.

**Commit:** `Integrate diarization, offline guard and health reporting`.

---

### Task 6: STT on the GPU, with a VRAM handoff to the LLM

**Owns:** `api/pipeline.py` (all of it — T1 and T2 are closed), `api/gpu.py` (new), `api/tests/test_gpu.py`.
**Approved by the lead:** this task overrides Global Constraint 2 (STT on CPU). Everything else still binds.

**Why:** measured ~8 s fixed cost per utterance on CPU (live lines appear ~8–10 s after the speaker
stops; a 2-minute judge recording needs ~100 s of ASR). On the RTX 3060 the same work is ~1 s. The
card has 5.7 GiB: two STT models (~1.6–2 GB int8_float16) plus the LLM (3.4 GB + KV cache) do not
coexist, so the GPU is handed back and forth: STT owns it while transcribing; the LLM gets it only
during `structure()`.

**Requirements:**
1. CUDA libraries: `uv add nvidia-cublas-cu12 nvidia-cudnn-cu12` (CUDA 12 / cuDNN 9, matching the
   installed ctranslate2 wheel — verify with `uv run python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"` before and after). The libs must be
   found WITHOUT any manual env setup: `uv run uvicorn main:app` from `api/` must just work. Implement
   `gpu.py`: preload the `nvidia.cublas.lib` and `nvidia.cudnn.lib` shared objects via `ctypes.CDLL(...,
   mode=RTLD_GLOBAL)` in dependency order before `faster_whisper`/`ctranslate2` is imported (import
   `gpu` at the very top of `pipeline.py`); if that approach cannot make `get_cuda_device_count() > 0`,
   fall back to re-exec with `LD_LIBRARY_PATH` set — but prefer the preload. Record which worked.
2. Device policy in `pipeline.py`: `STT_DEVICE=auto|cuda|cpu` (env, default `auto`). `auto` → try
   `device="cuda", compute_type="int8_float16"`; on any failure log once and use `cpu`/`int8` (existing
   path). `stt_device()` reports the resolved device for `/health`. Keep `cpu_threads=8` for the CPU path.
3. Handoff: `gpu.py` exposes a phase manager used by `pipeline`:
   - `stt_release()` — move both loaded STT models off the GPU with ctranslate2's
     `model.model.unload_model(to_cpu=True)` (verify the method on the installed version; weights stay
     in RAM so `load_model()` is sub-second) and `stt_acquire()` — `load_model()` them back.
   - `structure()` wraps its LLM calls: acquire a module-level `_gpu_lock`, `stt_release()`, run the
     chat calls, then explicitly unload the LLM (a zero-keep-alive request to Ollama — verify with
     `ollama ps` that it actually frees VRAM), then `stt_acquire()`, release the lock — in a
     `try/finally` so STT always comes back. Every transcription window/utterance also holds `_gpu_lock`
     for its duration so a Finish and an upload cannot interleave phases.
   - LLM `keep_alive`: `"2m"` per call while `structure()` runs (chunked transcripts must not reload
     per chunk), then the explicit unload at the end. On the CPU path all of this is a no-op and
     `keep_alive` stays `"30m"`.
4. Startup warm-up helper `warm_up()` in `pipeline.py` that loads turbo + kk (on the resolved device) so
   the first request is not the one paying the load; T5 will call it from the app lifespan.
5. Measure and report: per-window transcription time on GPU for `kk_1.wav` and `team_mixed_35.m4a`
   (before/after vs the CPU numbers in the T1 report), `nvidia-smi --query-gpu=memory.used
   --format=csv` during the STT phase and during the LLM phase, the STT-release + LLM-load + STT-reload
   overhead added to a `structure()` call, and confirm `test_stt.py` still passes on GPU (recall
   thresholds unchanged).

**Tests (`test_gpu.py`):** weight-free: the phase manager calls release → unload-LLM → acquire in order
and re-acquires on exception (stub models with `unload_model`/`load_model` counters, stub ollama);
`_gpu_lock` serialises a transcription stub and an LLM phase stub. GPU-gated (`skipif` no CUDA):
`transcribe("tests/audio/kk_1.wav")` on cuda completes in < 4 s (excluding load) with recall ≥ 0.5;
after `structure("тест")` the STT models report a cuda device again.

**Commit:** `Run STT on the GPU and hand VRAM to the LLM during structuring`.
