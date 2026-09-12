# Decisions

One line each, with the reason. The reason is the point — it's what stops the
same argument happening again at 3am when everyone is tired and nobody
remembers why.

Format: `HH:MM — decision — because reason`

## Log

- `00:00` — Repo scaffolded, stack left open — because the task isn't
  announced until 10:00.
- `00:00` — Feature freeze set at 16:00 — because the one-pager is mandatory
  and a 2-minute pitch needs rehearsing; 8 hours does not stretch.
- `10:xx` — Took Track 01 (AI Meeting Intelligence) over Track 02 — because
  the team's strengths (offline STT/LLM pipelines) map directly onto it and
  locality is worth 20 of the judged points, a criterion we can win outright
  by simply not making network calls.
- `10:xx` — Rejected stock Whisper for Kazakh — because measured on our own
  77s test recording it dropped both Kazakh utterances or replaced them with
  hallucinated Russian, and mistranslated an English sentence into Russian
  with the wrong meaning. Not a theoretical concern — reproduced on our own
  audio.
- `10:xx` — Built two-model STT routing (kazakh-whisper-large-v3-turbo for
  kk, faster-whisper-large-v3-turbo for ru/en + language ID) instead of one
  model for everything — because the measured WER gap (11.8% vs 31–56% on
  FLEURS-KZ) is too large to paper over with prompting, and per-utterance
  routing lets each language hit its best model at the cost of one extra
  language-ID pass.
- `10:xx` — Picked `qwen3.5:4b` over `qwen2.5:7b` for the protocol LLM —
  because 3.5:4b has materially better Kazakh coverage and fits comfortably
  in the VRAM left over after the STT handoff; 7B would have made the
  GPU handoff tighter for no accuracy win on this task.
- `10:xx` — Ruled out Gemma, Sherkala, and Parakeet — Gemma's footprint
  didn't fit the concurrent-with-STT VRAM budget, Sherkala has no GGUF/CT2
  build we could run offline in the time available, and Parakeet has no
  Kazakh support at all.
- `10:xx` — Diarization runs batch-only, after Finish/upload, via
  sherpa-onnx (pyannote-segmentation-3.0 ONNX + 3D-Speaker ERes2Net) —
  because sherpa-onnx's models are ungated and need no torch dependency,
  keeping the offline footprint small; live speaker IDs would flip mid-
  meeting anyway, so batch loses nothing a judge would notice in 2 minutes.
- `10:xx` — Live mic capture happens server-side via `sounddevice`, not the
  browser's `MediaRecorder` — because it feeds the same streaming VAD/STT
  pipeline as file upload with no browser-audio-format translation layer,
  and keeps the whole path — mic to model — in one process for the offline
  guard to watch.
- `10:xx` — STT runs on the GPU (int8_float16, `STT_DEVICE=auto` with CPU
  fallback) with the LLM sharing the same 6GB card via a load/unload handoff
  — because CPU-only STT measured at ~8s per utterance on this laptop, which
  is too slow for the live-mic UX we wanted; the handoff costs ~1.5s per
  protocol generation and is worth it.
- `10:xx` — Feature freeze held at 16:00 — because it was set at kickoff and
  the team stuck to it; final integration and whole-branch review happen in
  the two hours after, not new features.

- 2026-09-11 — Added React/TypeScript frontend in `frontend/`; FastAPI serves its build at `/` and `/live`, so the delivered app stays fully local and same-origin.
- 2026-09-11 — Frontend uses existing `/process`, `/process-text`, `/results`, `/result/*` and `/live/*` contracts. No mock meetings, invented source citations, RAG endpoints or date conversion.
- 2026-09-11 — Browser IndexedDB preserves actual results and uploaded audio; backend exports and model processing remain local. Original UI remains at `/legacy` and `/legacy-live`.

## Known fakes / shortcuts

Nothing in the demo path is faked, hardcoded, or stubbed as of freeze. The
honest list is the model's actual limits, not a workaround:

| What | Where | Real or faked? |
| ---- | ----- | -------------- |
| Sub-5s code-switched turns | STT language routing | Real limit — can inherit a neighbouring utterance's detected language, not faked around. |
| Intra-sentence code-switching (RU words inside a KK sentence) | STT | Real limit — unsolved by any model tried, including ours. Not corrected or hidden. |
| Kazakh transcript casing/punctuation | `kk-turbo-ct2` output | Real model output — lowercase, no punctuation. Passed through as-is, not post-processed. |
| Live speaker labels | Diarization | Real, but batch-only — diarization runs after Finish, not live, by design (see log above), not as a cut corner. |
| Protocol generation latency | GPU handoff | Real cost (~1.5s/generation) from sharing 6GB VRAM between STT and LLM. Not hidden with a spinner trick. |
- 2026-09-11 — Post-freeze bonus: `.ics` export of action items (`api/ics.py`, `/result/{id}/export.ics`). Due dates are parsed from the phrase as it was *said* (RU/KK/EN weekdays, day+month, "завтра/ертең/tomorrow") relative to the meeting date; anything unrecognised becomes a `VTODO` with no date rather than a guess — a wrong date in someone's calendar is worse than no event. This supersedes the earlier "no date conversion" line above, deliberately and with the fallback made explicit.
- 2026-09-11 — Post-freeze bonus: chat over a finished meeting (`pipeline.answer_question`, `POST /result/{id}/chat`). Retrieval is local lexical overlap on 5-character prefixes — a poor man's stemmer for RU/KK, no embedding model on a card already shared between STT and the LLM; meetings under 40 passages go into the prompt whole. Answers cite `[мм:сс]` and the UI shows those excerpts as chips, so every claim is checkable against the real transcript. Text-only meetings (`/process-text`) have no timeline, so the prompt asks for fragment numbers instead of inventing timestamps. This supersedes the earlier "no RAG endpoints / no invented source citations" line — the endpoint is real and the citations are the meeting's own excerpts.
- 2026-09-11 — Chat answers strictly from cited transcript fragments, never inferred beyond them — because the judged criterion is accuracy, not cleverness; a plausible-sounding wrong answer costs more than an honest "not in the meeting."
- 2026-09-11 — `.ics` chosen over Trello/Notion/Jira for action-item export — because it needs no network call and no account, and opens in any calendar app a judge already has; the alternatives are exactly what Track 01's locality criterion penalizes.
- 2026-09-11 — Results persisted to disk under `api/data/results/` instead of kept in memory — because a server restart in the minutes before the pitch must not kill an export link that's already in someone's hand.
