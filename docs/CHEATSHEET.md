# Stage cheat sheet — Гонконг · AI Meeting Intelligence

## PRE-FLIGHT (in this order)
wifi OFF  →  `ollama ps` empty or 100% GPU  →  `/health`: cuda / null / true / true
→ `localhost:8000/live`, F5  →  participants: Жаныбек, Бекзат, Айгерим  →  speakers 2  →  Русский
Fallback 1: upload `api/tests/audio/team_mixed_35.m4a` on `/`     Fallback 2: backup video
Chat questions that land: бюджет · когда запуск · поручения Бекзата · риски

## THE NUMBERS
| What | Number |
|---|---|
| Kazakh WER, stock Whisper large-v3 / medium | 56% / 95% (hallucinated Russian) |
| Kazakh WER, our fine-tune (FLEURS) | 0% on test samples (11.8% published) |
| Transcribe 77 s recording | 3.4 s GPU (62.6 s CPU) — 22× realtime |
| Live utterance (6 s) | 0.35 s GPU (7.7 s CPU) |
| Upload → full protocol | 20–37 s |
| 2-min judge recording, expected | ~25–35 s |
| GPU handoff overhead | 1.5 s |
| VRAM: STT / LLM peak | 2.1 GB / 5.4 GB of 6 GB |
| Outbound connections at runtime | 0 (counted, shown in /health) |
| Tests | ~130, all local |

## THE STACK (say the names)
STT      faster-whisper (CTranslate2), GPU int8_float16
  RU/EN  deepdml/faster-whisper-large-v3-turbo-ct2  — also does language ID
  KK     shyngys879/kazakh-whisper-large-v3-turbo → CT2 (Apache-2.0, 1500 h Kazakh)
VAD      Silero VAD (sherpa-onnx), utterances ≤ 12 s
LLM      Qwen 3.5 4B via Ollama, JSON-schema output, thinking off, ctx 16k
Diariz.  sherpa-onnx: pyannote-segmentation-3.0 + 3D-Speaker ERes2Net embeddings
API/UI   FastAPI + React (Vite); exports csv / json / pdf / ics; chat with cited [mm:ss]

## HOW (one breath each)
Pipeline   VAD chunks → per-chunk language ID {kk,ru,en} → route to model → LLM → diarize → export
Why 2 STT  stock Whisper drops/invents Kazakh; fine-tune is exact but forgets RU/EN → route
Code-switch per chunk, not per file (whole-file LID said "ru 97%" and lost all Kazakh)
Offline    wifi off + in-process guard blocks & counts non-loopback connects + HF_HUB_OFFLINE
GPU trick  Whisper owns 6 GB while speaking; at Finish weights park in RAM, LLM loads, answers, unloads
Accuracy   temp 0.1, strict schema, "don't invent", participants list for owner names
Diariz.    batch at Finish (live IDs flip); word-level time-overlap alignment; count auto or set
Chat       retrieval by token overlap → LLM answers only from fragments, cites [mm:ss], refuses to infer
.ics       action items → all-day events; parses «до пятницы», «жұмаға дейін», «by Wednesday», «25 сентября»

## HONEST LIMITS (name first, reason second)
- Intra-sentence code-switching (Russian words inside Kazakh clause): unsolved by any model
- Sub-5 s switched turns may inherit the neighbour's language
- Kazakh model outputs lowercase, no punctuation
- Diarization is batch, not live — by design
- No formal accuracy eval on random meetings — only benchmark + our recording
- Single GPU, one meeting at a time; LLM cold-loads each protocol (handoff cost)
- Nothing is mocked. Everything runs.

## WITH A WEEK
streaming diarization · accuracy eval on real meetings · Jira/Notion push · smarter chat inference
