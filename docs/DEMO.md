# Demo — 2 minutes, then 2 minutes of questions

Owned from kickoff, not written at the end. Presentation quality is one of
the judged criteria (10 points for UI/UX + speed on a 2-minute recording,
5 for presentation), so this document is worth real time.

## The mandatory one-pager

Rule 5 requires a one-page presentation. Template:
https://docs.google.com/presentation/d/1McTJCkCnXoVbo9eDjBKN5wNMdJLD-PRDcwSG-YiN9tY/edit?usp=sharing

Adapt it freely, but **it must stay one page.**

Open the link early and confirm you can actually access and copy it.

## Pre-flight (do this before walking on stage)

- [ ] `make preflight` — all green
- [ ] `make dev` running, wait until `curl localhost:8000/health` reports
      models warm (not just "loading")
- [ ] `localhost:8000/live` open in a pre-opened tab, zoomed in
- [ ] **After any `make dev` restart, reload the dashboard tab (F5).** The app checks `/health` once at page load; a tab opened before the server finished warming shows «Нет связи с локальным сервером» until reloaded. Use `localhost:8000` only — never 5173.
- [ ] `curl localhost:8000/mic-test` — ok
- [ ] Participants field filled: `Жаныбек, Бекзат, Айгерим`
- [ ] Speakers = 2
- [ ] **`ollama ps` must be empty (or show 100% GPU) right before walking on.** If it shows the model split like `96%/4% CPU/GPU`, run `ollama stop qwen3.5:4b` — the next Finish would otherwise generate on the CPU (a minute instead of ~15 s). Cause seen today: a leftover test server holding VRAM when the model loaded. Also make sure no other `uvicorn` is running: `ss -ltnp | grep uvicorn`.
- [ ] **Turn wifi off, visibly, on stage** — locality is 20 of the 100
      judged points, and judges are watching for exactly this

## The script (read verbatim, ~85s)

```
Жаныбек (RU): Коллеги, начинаем. Сегодня два вопроса: запуск пилота с клиентом и подготовка к международной презентации. Бекзат, как дела по пилоту?
Бекзат (KK): Сәлеметсіз бе, Жаныбек. Пилот бойынша жағдай мынадай: жүйе дайын, бірақ клиенттің деректеріне қолжетімділік әлі ашылған жоқ. Сондықтан іске қосуды жиырма бесінші қыркүйекке ауыстыруды ұсынамын.
Жаныбек (RU): Согласен, переносим запуск на двадцать пятое сентября. Бекзат, подготовь обновлённый план тестирования до пятницы — это высокий приоритет.
Бекзат (KK): Жарайды, жұмаға дейін дайындаймын. Бір тәуекел бар: серверге қолжетімділік әлі шешілмеген, бұл мерзімді тағы кешіктіруі мүмкін.
Жаныбек (EN): Now, one more thing for our international partners. I'll ask Aigerim to prepare the English version of the pitch deck by Wednesday — medium priority. And we still need to decide who presents in London. That question stays open for now.
Бекзат (KK): Түсінікті. Соңғы сұрақ: маркетинг бюджеті бекітілді ме? Ол әлі белгісіз.
Жаныбек (RU): Бюджет пока не утверждён — оставляем вопрос открытым до встречи с финансами. Итого: запуск двадцать пятого, план тестирования до пятницы, английская презентация до среды. На этом всё, спасибо.
```

## Timing budget (2:00 total)

- Script read: ~85s
- Click **Finish** → protocol + speaker-labelled transcript: ~20s
- Show the protocol on screen: ~15s
- That's 2:00. Rehearse with a timer. If running over, the closing summary
  line ("На этом всё, спасибо") is the one to drop — it doesn't add
  information the protocol won't already show.

## Fallback ladder

1. **Mic fails on stage** → switch to `/`, upload
   `api/tests/audio/team_mixed_35.m4a` (the same recording used for the
   measurements in the README), same participants/speakers settings.
2. **App fails entirely** → play the backup video.

## The judges' own scenario

Judges may hand over their own random audio to test locality/speed instead
of the scripted one. Plan for it: upload on `/`, leave speakers on auto.
Expect roughly 20s of processing for 2 minutes of audio — say this number
out loud while it runs so the wait itself demonstrates the speed claim.

## The 2 minutes of questions

- **What's real and what's mocked?** — Everything demoed is real: STT,
  diarization, and the LLM protocol all run locally and produce their own
  output on whatever audio is given, not canned text. See
  `docs/DECISIONS.md` for the "known fakes" table (currently empty — this
  is the honest answer, not a hedge).

- **Why this approach over just running one bigger Whisper model?** — We
  measured it: stock Whisper `medium` dropped or mistranscribed every
  Kazakh utterance in our own test recording and mistranslated English into
  Russian. A Kazakh-specialized model plus per-utterance language routing
  fixed all of that; see the README's before/after table.

- **What breaks at scale / with real users?** — Sub-5-second code-switched
  turns can inherit a neighbouring utterance's detected language.
  Intra-sentence switching (Russian words inside a Kazakh sentence) isn't
  solved by any model we tried, ours included. Diarization is batch-only —
  live speaker IDs would flip mid-meeting, so we run it after Finish.
  Protocol generation pays a GPU reload every time because of the
  STT/LLM VRAM handoff (~1.5s, acceptable; would matter more at higher
  meeting volume).

- **What would you do with another week?** — Fix intra-sentence
  code-switching, make diarization incremental for live speaker labels,
  cut the GPU handoff cost with a larger card or a smaller concurrent LLM,
  and add punctuation/casing post-processing for the Kazakh model output
  (it currently emits lowercase, unpunctuated text).

## Known failure modes

- Kazakh model output is lowercase, no punctuation — the protocol LLM
  still parses it fine, but the raw transcript looks rougher for Kazakh
  lines than for Russian/English ones. If asked, say so plainly.
- A short code-switched turn (under ~5s) can be misrouted to the wrong
  STT model. If it happens live, it's the known limit above, not a bug
  you're discovering on stage.
- If diarization is still running when the protocol is ready, the protocol
  shows first and speaker labels fill in a beat later — expected, not a
  freeze.
