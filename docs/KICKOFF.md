# Kickoff — 8 hours, 2 people

## Status 13:30

**Done:** Track 01 locked. FastAPI service in `api/` with `/`, `/live`,
`/health`, `/mic-test`. Two-model STT routing (kk-turbo-ct2 for Kazakh,
faster-whisper-large-v3-turbo for ru/en + language ID) with GPU/CPU
fallback. `qwen3.5:4b` protocol generation via Ollama, JSON-schema
constrained. GPU handoff between STT and LLM working and measured.
sherpa-onnx batch diarization wired. Live mic capture (server-side,
streaming VAD) working end to end. CSV/JSON/PDF export. Offline network
guard in place. Measured before/after numbers against stock Whisper on the
team's own test recording — see `docs/DECISIONS.md` and the README.

**Left:** final integration of the pieces above into one clean demo path,
a whole-branch review, `HF_HUB_OFFLINE=1` wiring, rehearsal of the 2-minute
script against a timer, the one-pager, and the backup video.

## The real clock

| Time | What |
| ---- | ---- |
| 09:00 | In-person registration. **Team composition locks here.** |
| 10:00 | Task announced to all teams simultaneously |
| 10:00–18:00 | Build (8 hours, and that includes the one-pager) |
| 18:00–19:00 | Pitches — **2 min + 2 min Q&A** |
| 19:30 | Results |

Venue: Alem.ai, Astana. Bring laptop + charger.

## Judged on exactly five things

Published in advance, so build toward them deliberately:

1. Соответствие заданию — does it answer the actual task
2. Полезность решения — is it useful
3. Качество и **работоспособность** прототипа — quality, and *it runs*
4. Оригинальность и обоснованность подхода — original, and you can justify it
5. Качество презентации — presentation quality

Two consequences. A rough thing that works beats a polished thing that doesn't
(#3). And presentation is a full fifth of the score — the one-pager and the
2-minute pitch are not overhead, they are the deliverable (#5).

## Hard deliverables

- Working prototype, demoed live
- **One-page presentation (mandatory, rule 5).** Template:
  https://docs.google.com/presentation/d/1McTJCkCnXoVbo9eDjBKN5wNMdJLD-PRDcwSG-YiN9tY/edit?usp=sharing
  Adaptable, but must stay one page. Two colours given.
- 2-minute pitch, rehearsed

---

## Schedule that fits 8 hours

**10:00–10:30 · Choose and lock scope.** Read the task twice. Write the literal
requirement into `README.md` in the organizers' words. Check whether the task
restricts tools — rule 3 allows own prior work, open libraries and AI tools
*unless the task says otherwise*.

Pick the smallest thing that demos. Three questions that kill bad ideas fast:

1. What does a judge **see** in 2 minutes?
2. What's the hardest part, and have we done something like it before?
3. If we lose 2 hours to one bug, is there still a demo?

With two people and eight hours, if question 3 has no answer, the idea is too
big. Write the choice and the reason into `docs/DECISIONS.md`.

**10:30–10:45 · Bootstrap.** `make preflight`, then `make next` or `make python`,
then `make dev`. Confirm the sample chat responds before building on it.

**10:45–13:00 · Skeleton end to end.** Not features — the whole path, ugly,
with hardcoded pieces where needed. Something demoable must exist by 13:00.

**13:00 · Checkpoint.** If nothing runs end to end, cut scope now, not at 16:00.

**13:00–16:00 · Make it good.** Real logic replacing the fakes, in the order a
judge would notice.

**16:00 · FEATURE FREEZE.** Non-negotiable. Two hours left and two deliverables
still outstanding.

**16:00–17:00 · One-pager + backup video.** Split it: one person on the slide,
one recording the demo and fixing whatever the recording exposes.

**17:00–18:00 · Rehearse and buffer.** Run the 2-minute pitch out loud at least
three times. The buffer is for the disaster that always happens.

---

## Standing rules

- **Commit every green state.** `git commit -am wip` costs nothing.
- **Demo path is sacred.** If it breaks, everything stops until it works.
- **Record the backup video as soon as anything works end to end**, then again
  when it looks better. Stage wifi fails; a video doesn't.
- **Track every fake in `DECISIONS.md`.** Judges ask what's real. Knowing your
  own answer is the difference between confident and caught out.

## Stuck for more than 20 minutes?

Say it out loud, then pick one:

1. **Cut it.** Does the demo survive without it? Usually yes.
2. **Fake it.** Hardcode the output, log it in `DECISIONS.md`, move on.
3. **Swap approach.** Twenty minutes in, you've lost almost nothing.

At two people and eight hours, sunk cost is the most expensive thing in the room.
