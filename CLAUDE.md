# Working in this repo

Hackathon project. Optimize for a working demo, not for a codebase that
survives a year.

## Context

- **Event:** AI Steppe Tech Hack, 11.09.2026, Alem.ai Astana. Team Гонконг (2 people).
- **Track:** 01, "AI Meeting Intelligence" — 100% offline meeting-protocol
  service. Locality (no external API calls) is worth 20 of the judged points.
- **Clock:** task at 10:00, build until 18:00, pitch 2 min + 2 min Q&A.
  Feature freeze 16:00 — the one-pager and rehearsal need those two hours.
- **Judged on:** functionality 30 · AI accuracy 20 · locality 20 · bonus
  features 15 · UI/UX + speed on a 2-minute recording 10 · presentation 5.
- **Stack:** decided — FastAPI (`api/`, Python 3.13, `uv`) + local models
  only. No Next.js, no hosted LLM API. Run: `cd api && uv run uvicorn
  main:app --port 8000`. Test: `cd api && uv run pytest -q`.
- **Model:** `claude-opus-5`. Do not swap to a cheaper model without asking —
  that is a cost decision the team makes, not a default.

## Conventions

- **Never add a hosted-API client.** No Anthropic/OpenAI/Google/Groq/etc.
  SDK, no API key in `.env`. STT is faster-whisper/CTranslate2, the protocol
  LLM is `qwen3.5:4b` via local Ollama (`localhost:11434`). An in-process
  offline guard blocks non-loopback network connections at runtime — do not
  work around it.
- Model weights live under `api/models/` (STT, diarization, VAD) and are
  gitignored — never commit them. Fetch scripts live in `api/scripts/`.
- Secrets live in `.env` at the repo root, but this product needs none —
  `.env` only holds local runtime knobs (device, model name, ports). Never
  write a key into a source file, a commit, or a log line regardless.
- Branch as `feat/<short-name>`; commit straight to `main` only when
  everyone is in the same room and knows.
- Keep `docs/DECISIONS.md` current. One line per decision, with the reason.
  A decision without a reason gets re-argued at 3am.

## What matters here, and what does not

Under time pressure, prefer the working thing over the correct thing, and say
plainly when you have made that trade so it can be revisited.

- **Worth doing:** the demo path working end to end; error states that fail
  visibly rather than silently; anything a judge will see.
- **Not worth doing now:** exhaustive test suites, abstraction layers for one
  implementation, performance work without a measurement, refactors that do
  not unblock the demo.
- **Never skip:** secret hygiene, and telling the team when something on the
  demo path is broken.

## Before saying something works

Run it. `make dev`, hit the endpoint, look at the output. "Should work" has
lost more demos than bad code has.
