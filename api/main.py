"""AI Meeting Intelligence — 100% local meeting protocol service.

Track 01, AI Steppe Tech Hack. No request leaves this machine.

Run:  uv run uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

# FIRST, before anything that might pull in huggingface_hub or open a socket:
# importing this sets HF_HUB_OFFLINE (so the first transcription does not stall
# on a revision check against huggingface.co with the wifi off) and wraps the
# socket connect calls so nothing can reach off this machine. See offline_guard.
import offline_guard  # noqa: F401  # isort: skip

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import diarize
import exports
import gpu
import ics
import live
import pipeline

logger = logging.getLogger(__name__)

RESULTS: dict[str, dict[str, Any]] = {}
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def save_result(result: dict[str, Any]) -> None:
    """Persist one result to disk so it survives a server restart.

    Written atomically (temp file + os.replace) so a crash mid-write never
    leaves a half-written / corrupt JSON file behind.
    """
    result.setdefault("created_at", time.time())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['id']}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def load_results() -> int:
    """Reload every persisted result from disk into RESULTS. Returns the count."""
    if not RESULTS_DIR.is_dir():
        return 0
    count = 0
    for path in RESULTS_DIR.glob("*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("could not load persisted result %s; skipping", path)
            continue
        rid = result.get("id") or path.stem
        RESULTS[rid] = result
        count += 1
    return count


ALLOWED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm",
           ".opus", ".oga", ".aac", ".mov", ".mkv", ".amr", ".wma", ".aiff"}

#: Set by the lifespan once the models are resident; reported by /health so the
#: operator can tell "slow because it is loading" from "slow because it is slow".
WARM = False


def diarize_threshold() -> float:
    """Clustering threshold for uploads. 0.7 by default — see pipeline."""
    return float(os.getenv("DIARIZE_THRESHOLD") or pipeline.DIARIZE_THRESHOLD_DEFAULT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pay the model-load cost at startup, not in front of the audience.

    Both STT models and the diarizer are loaded here. Nothing in here may raise:
    a model that will not load is the first request's problem to report, not a
    reason for the server to refuse to start.
    """
    global WARM
    loaded = load_results()
    logger.info("loaded %d persisted result(s) from disk", loaded)
    warmed: list[str] = []
    try:
        device = await run_in_threadpool(pipeline.warm_up)
        warmed.append(f"STT on {device} ({pipeline.stt_devices()})")
    except Exception:                                       # pragma: no cover
        logger.exception("STT warm-up failed; models will load on first use")

    if diarize.is_available():
        try:
            # One second of silence: enough to build the ONNX sessions, too
            # short to cost anything.
            await run_in_threadpool(
                diarize.diarize, np.zeros(diarize.SAMPLE_RATE, dtype=np.float32),
                diarize.SAMPLE_RATE, None, diarize_threshold(),
            )
            warmed.append(f"diarizer (threshold {diarize_threshold()})")
        except Exception:                                   # pragma: no cover
            logger.exception("diarizer warm-up failed; it will load on first use")
    else:
        logger.warning("diarization models missing — uploads will have no speakers")

    WARM = bool(warmed)
    logger.info("warm-up complete: %s", "; ".join(warmed) or "nothing")
    if offline_guard.enabled():
        logger.info("offline guard armed: only loopback connections will be allowed")
    yield


app = FastAPI(title="AI Meeting Intelligence", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness — every claim here is checked, none is hard-coded.

    In particular `llm_model` is the *resolved* model (what a request would
    actually use), not the first candidate in the list, and `stt_device` is
    where the models really ended up rather than where they were asked to go.
    """
    ollama_up, models = False, []
    try:
        import ollama

        models = [m.get("model", "") for m in ollama.list().get("models", [])]
        ollama_up = True
    except Exception:
        pass

    try:
        llm_model: str | None = pipeline.resolved_llm_model()
        llm_error = None
    except Exception as e:
        llm_model, llm_error = None, str(e)

    return {
        "ok": True,
        "offline": True,
        "warm": WARM,
        "stt_models": {
            "turbo": pipeline.WHISPER_SIZE,
            "kk": str(pipeline.KK_MODEL_DIR),
            "kk_available": pipeline.is_kk_available(),
        },
        "stt_device": pipeline.stt_device(),
        "stt_devices": pipeline.stt_devices(),
        "stt_compute_types": pipeline.stt_compute_types(),
        "gpu_degraded_reason": gpu.degraded_reason(),
        "llm_model": llm_model,
        "llm_error": llm_error,
        "ollama_running": ollama_up,
        "ollama_models": models,
        "llm_ready": bool(llm_model) and any(llm_model.split(":")[0] in m for m in models),
        "diarization": {
            "available": diarize.is_available(),
            "threshold": diarize_threshold(),
        },
        "vad_model_present": live.vad_available(),
        "offline_guard": offline_guard.status(),
        "results_stored": len(RESULTS),
    }


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    participants: str = Form(""),
    speakers: str = Form("auto"),
    protocol_language: str = Form(pipeline.DEFAULT_PROTOCOL_LANGUAGE),
) -> dict[str, Any]:
    """Audio in, full structured protocol out.

    `participants` is a comma-separated list of names, fed to the decoder so
    they come out spelled right. `speakers` is "auto" or a count — telling the
    diarizer how many people are in the room beats any clustering threshold.
    """
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported format {suffix}. Allowed: {sorted(ALLOWED)}")

    names = [n.strip() for n in participants.split(",") if n.strip()]
    num_speakers: int | None = None
    if speakers and speakers.strip().lower() not in {"", "auto"}:
        try:
            num_speakers = int(speakers)
        except ValueError:
            raise HTTPException(400, f"speakers must be 'auto' or a number, got {speakers!r}")
        if num_speakers < 1:
            raise HTTPException(400, "speakers must be at least 1")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # In a threadpool, not inline: transcription is minutes of blocking CPU
        # work and the event loop still has a live WebSocket to serve.
        result = await run_in_threadpool(
            pipeline.process_audio, tmp_path,
            participants=names, num_speakers=num_speakers,
            diarize_threshold=diarize_threshold(),
            protocol_language=protocol_language,
        )
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    rid = uuid.uuid4().hex[:12]
    result["id"] = rid
    result["filename"] = file.filename
    RESULTS[rid] = result
    save_result(result)
    return result


class TextRequest(BaseModel):
    transcript: str
    protocol_language: str = pipeline.DEFAULT_PROTOCOL_LANGUAGE


@app.post("/process-text")
def process_text(req: TextRequest) -> dict[str, Any]:
    """The spec also accepts a ready transcript — and it's the fast demo path."""
    hint = pipeline.resolve_protocol_language(req.protocol_language, None)
    result = pipeline.process_text(req.transcript, language_hint=hint)
    result["protocol_language"] = hint or "auto"
    rid = uuid.uuid4().hex[:12]
    result["id"] = rid
    result["filename"] = "(text input)"
    RESULTS[rid] = result
    save_result(result)
    return result


@app.get("/results")
def list_results() -> list[dict[str, Any]]:
    ordered = sorted(RESULTS.values(), key=lambda r: r.get("created_at") or 0, reverse=True)
    return [
        {"id": r["id"], "filename": r.get("filename"),
         "language": r.get("language"), "timing": r.get("timing")}
        for r in ordered
    ]


@app.get("/result/{rid}")
def get_result(rid: str) -> dict[str, Any]:
    if rid not in RESULTS:
        raise HTTPException(404, "No such result")
    return RESULTS[rid]


@app.get("/result/{rid}/export.{fmt}")
def export(rid: str, fmt: str) -> Response:
    if rid not in RESULTS:
        raise HTTPException(404, "No such result")
    result = RESULTS[rid]

    if fmt == "json":
        return Response(exports.to_json(result), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="protocol_{rid}.json"'})
    if fmt == "csv":
        return Response(exports.to_csv(result), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="action_items_{rid}.csv"'})
    if fmt == "pdf":
        return Response(exports.to_pdf(result), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="protocol_{rid}.pdf"'})
    if fmt == "ics":
        # Handled here rather than as its own route: the `{fmt}` path above is
        # registered first, so a separate /export.ics route would never match.
        return Response(ics.to_ics(result), media_type="text/calendar; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="action_items_{rid}.ics"'})
    raise HTTPException(400, "Format must be json, csv, pdf or ics")


class ChatRequest(BaseModel):
    question: str
    language: str = "auto"


@app.post("/result/{rid}/chat")
async def chat(rid: str, req: ChatRequest) -> dict[str, Any]:
    """Ask a question about one finished meeting; answered from its transcript.

    Threadpooled like `/process`: this is a blocking Ollama call that also takes
    the GPU for its duration, and the event loop still has a live WebSocket to
    serve while it runs.
    """
    if rid not in RESULTS:
        raise HTTPException(404, "No such result")
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Question must not be empty")
    return await run_in_threadpool(
        pipeline.answer_question, RESULTS[rid], question, req.language
    )


# --------------------------------------------------------------------------
# Live transcription (server-side mic -> VAD -> STT -> WebSocket)
# --------------------------------------------------------------------------

class LiveStartRequest(BaseModel):
    device: int | None = None
    num_speakers: int | None = None
    participants: list[str] = []
    protocol_language: str = pipeline.DEFAULT_PROTOCOL_LANGUAGE


@app.post("/live/start")
async def live_start(req: LiveStartRequest | None = None) -> dict[str, Any]:
    """Open the microphone and start streaming a transcript over /live/ws."""
    req = req or LiveStartRequest()
    live.HUB.set_loop(asyncio.get_running_loop())
    try:
        session = await run_in_threadpool(
            live.start_session, req.device, req.participants, req.num_speakers,
            req.protocol_language,
        )
    except live.LiveError as e:
        raise HTTPException(400 if "already running" not in str(e) else 409, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"Could not start the microphone: {e}") from e
    return {"ok": True, "device": session.device_name,
            "participants": session.participants,
            "num_speakers": session.num_speakers,
            "protocol_language": session.protocol_language}


@app.post("/live/stop")
async def live_stop() -> dict[str, Any]:
    """Finish: drain the queue, diarize + structure, store and return the result."""
    try:
        session = live.stop_session()
    except live.LiveError as e:
        raise HTTPException(409, str(e)) from e

    rid = uuid.uuid4().hex[:12]

    def store(result: dict[str, Any]) -> None:
        # Runs inside finish(), before the "done" event: the page fetches
        # /result/{id} the moment it sees that event and must not 404.
        RESULTS[rid] = result
        save_result(result)

    try:
        result = await run_in_threadpool(session.finish, rid, store)
    except Exception as e:
        session.state = "error"
        raise HTTPException(500, f"Finishing the live session failed: {e}") from e
    return result


@app.get("/live/status")
def live_status() -> dict[str, Any]:
    session = live.current_session()
    return {
        "state": session.state if session else "idle",
        "device": session.device_name if session else None,
        "pending": session.pending_count() if session else 0,
        "segments": len(session.segments()) if session else 0,
        "audio_seconds": round(session.audio_seconds(), 1) if session else 0.0,
        "audio_available": live.AUDIO_AVAILABLE,
        "vad_available": live.vad_available(),
        "diarization_available": live.diarization_available(),
    }


@app.get("/mic-test")
async def mic_test(seconds: float = 2.0, device: int | None = None) -> dict[str, Any]:
    """Record a couple of seconds and report the level — the pre-demo sanity check."""
    try:
        return await run_in_threadpool(live.mic_test, seconds, device)
    except live.LiveError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.websocket("/live/ws")
async def live_ws(ws: WebSocket) -> None:
    """Replay the transcript so far, then stream every event as it happens."""
    await ws.accept()
    loop = asyncio.get_running_loop()
    live.HUB.set_loop(loop)
    queue: asyncio.Queue = asyncio.Queue()
    live.HUB.subscribe(queue)
    try:
        for event in live.HUB.replay():
            await ws.send_json(event)
        while True:
            await ws.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        live.HUB.unsubscribe(queue)


@app.get("/live", response_class=HTMLResponse)
def live_page():
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return LIVE_HTML


@app.get("/", response_class=HTMLResponse)
def index():
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return INDEX_HTML


# Same-origin local frontend. Build with: cd frontend && npm ci && npm run build.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

@app.get("/favicon.svg", include_in_schema=False)
def frontend_icon():
    icon = FRONTEND_DIST / "favicon.svg"
    if not icon.is_file():
        raise HTTPException(404, "Frontend has not been built")
    return FileResponse(icon, media_type="image/svg+xml")

@app.get("/legacy", response_class=HTMLResponse, include_in_schema=False)
def legacy_page() -> str:
    return INDEX_HTML

@app.get("/legacy-live", response_class=HTMLResponse, include_in_schema=False)
def legacy_live_page() -> str:
    return LIVE_HTML

@app.get("/result/{rid}/audio")
def result_audio(rid: str):
    value = RESULTS.get(rid)
    if value is None:
        raise HTTPException(404, "No such result")
    audio_path = value.get("audio_path")
    if not audio_path:
        raise HTTPException(404, "This recording is not stored on the server")
    audio = Path(audio_path).resolve()
    allowed = (Path(tempfile.gettempdir()) / "ai-meeting-live").resolve()
    if not audio.is_relative_to(allowed) or not audio.is_file():
        raise HTTPException(404, "Recording is no longer available")
    return FileResponse(audio, media_type="audio/wav", filename=f"meeting_{rid}.wav")


INDEX_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Meeting Intelligence</title>
<style>
:root{--bg:#0b0f17;--card:#141a24;--line:#263041;--fg:#e6edf7;--dim:#8b9bb4;--acc:#4f9cf9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--dim);margin-bottom:24px;font-size:14px}
.badge{display:inline-block;background:#0f2a1a;color:#4ade80;border:1px solid #1f5133;
 padding:3px 10px;border-radius:99px;font-size:12px;margin-left:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--dim);margin-bottom:8px}
input[type=file],textarea{width:100%;background:#0b1017;color:var(--fg);
 border:1px solid var(--line);border-radius:7px;padding:10px;font:inherit}
textarea{min-height:110px;resize:vertical}
button{background:var(--acc);color:#04121f;border:0;border-radius:7px;padding:10px 18px;
 font:600 14px system-ui;cursor:pointer;margin-top:10px}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.row>div{flex:1 1 180px}
input[type=text],select{width:100%;background:#0b1017;color:var(--fg);
 border:1px solid var(--line);border-radius:7px;padding:9px 10px;font:inherit}
.stat{background:#0b1017;border:1px solid var(--line);border-radius:7px;padding:8px 12px;font-size:13px}
.stat b{color:var(--acc)}
h2{font-size:15px;margin:20px 0 8px;color:var(--acc)}
ul{margin:0;padding-left:20px}li{margin-bottom:5px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#0b1017;color:var(--dim);font-weight:600}
a.dl{color:var(--acc);margin-right:14px;font-size:13px}
.err{color:#f87171}
pre{white-space:pre-wrap;color:var(--dim);font-size:12.5px;max-height:220px;overflow:auto}
.lang{display:inline-block;font:11px/1.6 ui-monospace,monospace;border:1px solid var(--line);
 border-radius:4px;padding:1px 7px;margin-right:6px;color:var(--dim)}
.lang.ru{color:#93c5fd;border-color:#1e3a5f}.lang.kk{color:#86efac;border-color:#1f5133}
.lang.en{color:#fcd34d;border-color:#5c4813}
.ln{margin-bottom:6px}
.ts{color:#5b6b84;font:12.5px ui-monospace,monospace;margin-right:8px}
.spk{color:var(--acc);font-weight:600;margin-right:6px}
#spkbox{max-height:300px;overflow:auto;background:#0b1017;border:1px solid var(--line);
 border-radius:8px;padding:12px 14px;margin-top:6px}
</style></head><body><div class="wrap">
<h1>AI Meeting Intelligence <span class="badge">100% offline</span></h1>
<div class="sub">Аудио встречи → протокол, решения и поручения. Всё локально.
 &nbsp;<a href="/live" style="color:var(--acc)">запись с микрофона →</a></div>

<div class="card">
  <label>Аудиозапись (MP3, WAV, M4A)</label>
  <input type="file" id="f" accept=".mp3,.wav,.m4a,.ogg,.flac,.mp4,.webm,.opus,.oga,.aac,.mov,.mkv,.amr,.wma,.aiff">
  <div class="row" style="margin-top:12px">
    <div><label>Участники (через запятую)</label>
      <input type="text" id="parts" placeholder="Жаныбек, Бекзат, Айгерим"></div>
    <div><label>Спикеров</label>
      <select id="spk"><option value="auto">авто</option><option>2</option><option>3</option>
      <option>4</option><option>5</option></select></div>
    <div><label>Язык протокола</label>
      <select id="plang"><option value="ru">Русский</option><option value="kk">Қазақша</option><option value="en">English</option><option value="auto">авто</option></select></div>
  </div>
  <button id="go">Обработать запись</button>
</div>

<div class="card">
  <label>…или вставьте готовую стенограмму</label>
  <textarea id="t" placeholder="Текст встречи"></textarea>
  <button id="goT">Обработать текст</button>
</div>

<div id="out"></div>

<script>
const $ = s => document.querySelector(s);
const esc = s => (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const hms = t => new Date(Math.max(0,t)*1000).toISOString().substr(11,8);

function render(r){
  const p = r.protocol||{}, t = r.timing||{};
  const list = (h,a)=> (a&&a.length)?`<h2>${h}</h2><ul>${a.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:'';
  const topics = (p.topics||[]).map(x=>
    `<li><b>${esc(x.title)}</b><ul>${(x.points||[]).map(q=>`<li>${esc(q)}</li>`).join('')}</ul></li>`).join('');
  const acts = (p.action_items||[]).map(a=>
    `<tr><td>${esc(a.owner)}</td><td>${esc(a.task)}</td><td>${esc(a.due)}</td><td>${esc(a.priority)}</td></tr>`).join('');
  // Language shares as badges: a mixed RU/KK/EN meeting is the point of the demo,
  // and "ru 62% · kk 31% · en 7%" says that better than a single winner does.
  const langs = Object.entries(r.languages||{}).sort((a,b)=>b[1]-a[1]).map(([l,s])=>
    `<span class="lang ${esc(l)}">${esc(l.toUpperCase())} ${Math.round(s*100)}%</span>`).join('');
  const spk = (r.speakers||[]).map(b=>
    `<div class="ln"><span class="ts">[${hms(b.start)}]</span>` +
    `<span class="spk">${esc(b.speaker)}:</span>${esc(b.text)}</div>`).join('');
  $('#out').innerHTML = `<div class="card">
    <div class="row">
      <div class="stat">Язык: <b>${esc(r.language)}</b>${langs?' &nbsp;'+langs:''}</div>
      <div class="stat">Аудио: <b>${r.audio_seconds}s</b></div>
      ${r.num_speakers?`<div class="stat">Спикеров: <b>${r.num_speakers}</b></div>`:''}
    </div>
    <div class="row" style="margin-top:8px">
      <div class="stat">Распознавание: <b>${t.transcribe_sec}s</b></div>
      ${t.diarize_sec?`<div class="stat">Диаризация: <b>${t.diarize_sec}s</b></div>`:''}
      <div class="stat">LLM: <b>${t.llm_sec}s</b></div>
      <div class="stat">Всего: <b>${t.total_sec}s</b></div>
      ${t.realtime_factor?`<div class="stat">Быстрее реального времени в <b>${t.realtime_factor}×</b></div>`:''}
    </div>
    ${list('Executive Summary', p.summary)}
    ${list('Принятые решения', p.decisions)}
    ${topics?`<h2>Темы и тезисы</h2><ul>${topics}</ul>`:''}
    ${list('Открытые вопросы', p.open_questions)}
    ${list('Риски и блокеры', p.risks)}
    ${acts?`<h2>Поручения</h2><table><tr><th>Ответственный</th><th>Задача</th><th>Срок</th><th>Приоритет</th></tr>${acts}</table>`:''}
    <h2>Экспорт</h2>
    <a class="dl" href="/result/${r.id}/export.pdf">PDF</a>
    <a class="dl" href="/result/${r.id}/export.csv">CSV</a>
    <a class="dl" href="/result/${r.id}/export.json">JSON</a>
    <a class="dl" href="/result/${r.id}/export.ics">ICS</a>
    <h2>Вопрос по встрече</h2>
    <div id="chat"></div>
    <div class="row">
      <div><input type="text" id="q" placeholder="Например: что решили по бюджету?"></div>
      <button id="ask" style="margin-top:0">Спросить</button>
    </div>
    ${spk?`<h2>Стенограмма по спикерам</h2><div id="spkbox">${spk}</div>`:''}
    ${r.transcript?`<h2>Стенограмма</h2><pre>${esc(r.transcript)}</pre>`:''}
  </div>`;

  // Chat over this meeting: answered locally from its own transcript.
  $('#ask').onclick = async () => {
    const q = $('#q'), btn = $('#ask'), thread = $('#chat');
    const question = q.value.trim();
    if(!question) return;
    const label = btn.textContent;
    btn.disabled = q.disabled = true; btn.textContent = 'Отвечаю…';
    try{
      const res = await fetch(`/result/${r.id}/chat`, {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question, language:'auto'})});
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || res.statusText);
      const src = (data.sources||[]).map(x=>
        `<span class="lang" title="${esc(x.text)}">${hms(x.start)} · ${esc(x.speaker||'фрагмент')}</span>`).join('');
      thread.innerHTML += `<div class="ln"><b>${esc(question)}</b></div>` +
        `<div class="ln">${esc(data.answer)}</div><div class="ln">${src}</div>`;
      q.value = '';
    }catch(e){ thread.innerHTML += `<div class="ln err">Ошибка: ${esc(e.message)}</div>`; }
    finally{ btn.disabled = q.disabled = false; btn.textContent = label; }
  };
}

async function run(btn, fn){
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = 'Обрабатываю…';
  $('#out').innerHTML = '<div class="card">Идёт обработка. Транскрибация и LLM работают локально…</div>';
  try { render(await fn()); }
  catch(e){ $('#out').innerHTML = `<div class="card err">Ошибка: ${esc(e.message)}</div>`; }
  finally { btn.disabled = false; btn.textContent = label; }
}

$('#go').onclick = () => run($('#go'), async () => {
  const f = $('#f').files[0];
  if(!f) throw new Error('Выберите файл');
  const fd = new FormData();
  fd.append('file', f);
  fd.append('participants', $('#parts').value.trim());
  fd.append('speakers', $('#spk').value);
  fd.append('protocol_language', $('#plang').value);
  const res = await fetch('/process', {method:'POST', body:fd});
  if(!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
});

$('#goT').onclick = () => run($('#goT'), async () => {
  const transcript = $('#t').value.trim();
  if(!transcript) throw new Error('Вставьте текст');
  const res = await fetch('/process-text', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({transcript, protocol_language: $('#plang').value})});
  if(!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
});
</script></div></body></html>"""


LIVE_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live — AI Meeting Intelligence</title>
<style>
:root{--bg:#0b0f17;--card:#141a24;--line:#263041;--fg:#e6edf7;--dim:#8b9bb4;--acc:#4f9cf9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--dim);margin-bottom:20px;font-size:14px}
.sub a{color:var(--acc)}
.badge{display:inline-block;background:#0f2a1a;color:#4ade80;border:1px solid #1f5133;
 padding:3px 10px;border-radius:99px;font-size:12px;margin-left:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--dim);margin-bottom:6px}
input,select,textarea{width:100%;background:#0b1017;color:var(--fg);
 border:1px solid var(--line);border-radius:7px;padding:9px 10px;font:inherit}
button{background:var(--acc);color:#04121f;border:0;border-radius:7px;padding:10px 18px;
 font:600 14px system-ui;cursor:pointer;margin-top:10px;margin-right:8px}
button.ghost{background:#1d2635;color:var(--fg)}
button.stop{background:#f87171;color:#2a0707}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.row>div{flex:1 1 160px}
.stat{background:#0b1017;border:1px solid var(--line);border-radius:7px;padding:8px 12px;font-size:13px}
.stat b{color:var(--acc)}
.bar{height:10px;background:#0b1017;border:1px solid var(--line);border-radius:99px;overflow:hidden;margin-top:12px}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#4ade80,#facc15,#f87171);
 transition:width .12s linear}
/* No scroll-behavior:smooth here: an animated scroll fires scroll events at
   intermediate positions, which would read as "the user scrolled up" and kill
   the auto-scroll for the rest of the meeting. */
#tape{height:46vh;min-height:240px;overflow-y:auto;background:#0b1017;border:1px solid var(--line);
 border-radius:8px;padding:12px 14px;font-size:15px;line-height:1.7}
.ln{margin-bottom:6px}
.ts{color:#5b6b84;font:12.5px ui-monospace,monospace;margin-right:8px}
.lang{display:inline-block;font:11px/1.5 ui-monospace,monospace;border:1px solid var(--line);
 border-radius:4px;padding:0 5px;margin-right:8px;color:var(--dim)}
.lang.ru{color:#93c5fd;border-color:#1e3a5f}.lang.kk{color:#86efac;border-color:#1f5133}
.lang.en{color:#fcd34d;border-color:#5c4813}
.spk{color:var(--acc);font-weight:600;margin-right:6px}
.wait{color:#5b6b84;font-style:italic}
h2{font-size:15px;margin:20px 0 8px;color:var(--acc)}
ul{margin:0;padding-left:20px}li{margin-bottom:5px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#0b1017;color:var(--dim);font-weight:600}
a.dl{color:var(--acc);margin-right:14px;font-size:13px}
.err{color:#f87171}
pre{white-space:pre-wrap;color:var(--dim);font-size:12.5px;max-height:260px;overflow:auto}
.muted{color:var(--dim);font-size:13px}
</style></head><body><div class="wrap">
<h1>Live-протокол встречи <span class="badge">&#128274; 100% offline</span></h1>
<div class="sub">Микрофон &rarr; распознавание &rarr; протокол. Ничего не уходит с этой машины.
 &nbsp;<a href="/">&larr; загрузить файл</a></div>

<div class="card">
  <div class="row">
    <div><label>Участники (через запятую)</label>
      <input id="parts" placeholder="Айгүл, Бекзат, Anna"></div>
    <div><label>Спикеров</label>
      <select id="spk"><option value="">авто</option><option>2</option><option>3</option>
      <option>4</option><option>5</option><option>6</option></select></div>
    <div><label>Язык протокола</label>
      <select id="plang"><option value="ru">Русский</option><option value="kk">Қазақша</option><option value="en">English</option><option value="auto">авто</option></select></div>
    <div><label>Устройство (номер, пусто = по умолчанию)</label>
      <input id="dev" placeholder=""></div>
  </div>
  <button id="start">&#127908; Начать запись</button>
  <button id="stop" class="stop" disabled>&#9209; Финиш и протокол</button>
  <button id="mic" class="ghost">Проверить микрофон</button>
  <div class="bar"><i id="lvl"></i></div>
  <div class="row" style="margin-top:12px">
    <div class="stat" id="state">Статус: <b>ожидание</b></div>
    <div class="stat" id="qd">В очереди: <b>0</b></div>
    <div class="stat" id="dur">Записано: <b>0.0s</b></div>
  </div>
  <div id="msg" class="muted" style="margin-top:10px"></div>
</div>

<div class="card">
  <label>Стенограмма в реальном времени</label>
  <div id="tape"></div>
</div>

<div id="out"></div>

<script>
const $ = s => document.querySelector(s);
const esc = s => (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const hms = t => new Date(Math.max(0,t)*1000).toISOString().substr(11,8);
const tape = $('#tape');
let stuck = true, autoAt = 0;   // auto-scroll unless the user scrolled up

// Our own scroll must not be mistaken for the user's: the scroll event lands
// after the assignment, by which time a burst of new rows may have moved the
// bottom again. Ignore scroll events for a beat after we scrolled ourselves.
tape.addEventListener('scroll', () => {
  if(performance.now() - autoAt < 150) return;
  stuck = tape.scrollHeight - tape.scrollTop - tape.clientHeight < 40;
});
function follow(){
  if(!stuck) return;
  autoAt = performance.now();
  tape.scrollTop = tape.scrollHeight;
}

function utt(id){
  let el = tape.querySelector(`[data-u="${id}"]`);
  if(!el){ el = document.createElement('div'); el.dataset.u = id; tape.appendChild(el); }
  return el;
}
function pending(e){
  utt(e.utterance_id).innerHTML =
    `<div class="ln wait"><span class="ts">[${hms(e.start)}]</span>…</div>`;
  follow();
}
function segment(e){
  const el = utt(e.utterance_id);
  if(el.querySelector('.wait')) el.innerHTML = '';
  const lang = (e.language||'').toLowerCase();
  el.insertAdjacentHTML('beforeend',
    `<div class="ln"><span class="ts">[${hms(e.start)}]</span>` +
    `<span class="lang ${esc(lang)}">${esc((e.language||'??').toUpperCase())}</span>` +
    `${esc(e.text)}</div>`);
  follow();
}
function drop(e){ const el = tape.querySelector(`[data-u="${e.utterance_id}"]`); if(el) el.remove(); }

const STATE = {idle:'ожидание', running:'идёт запись', finalizing:'формирую протокол…',
               done:'готово', error:'ошибка'};
function setState(s){ $('#state').innerHTML = `Статус: <b>${esc(STATE[s]||s)}</b>`; }

let ws = null;
function connect(){
  ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/live/ws');
  ws.onopen = () => { tape.innerHTML = ''; };
  ws.onmessage = ev => {
    const e = JSON.parse(ev.data);
    if(e.type==='segment') segment(e);
    else if(e.type==='pending') pending(e);
    else if(e.type==='drop') drop(e);
    else if(e.type==='queue') $('#qd').innerHTML = `В очереди: <b>${e.pending}</b>`;
    else if(e.type==='level'){
      $('#lvl').style.width = Math.min(100, e.rms*600).toFixed(0)+'%';
      $('#dur').innerHTML = `Записано: <b>${e.seconds}s</b>`;
    }
    else if(e.type==='status'){
      setState(e.state);
      if(e.state==='running'){ $('#start').disabled = true; $('#stop').disabled = false; }
      else if(e.state==='finalizing'){ $('#start').disabled = true; $('#stop').disabled = true; }
      else if(e.state==='done' || e.state==='error'){ $('#start').disabled = false; $('#stop').disabled = true; }
    }
    else if(e.type==='done') setState('done');
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
connect();

function render(r){
  const p = r.protocol||{}, t = r.timing||{};
  const list = (h,a)=> (a&&a.length)?`<h2>${h}</h2><ul>${a.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:'';
  const topics = (p.topics||[]).map(x=>
    `<li><b>${esc(x.title)}</b><ul>${(x.points||[]).map(q=>`<li>${esc(q)}</li>`).join('')}</ul></li>`).join('');
  const acts = (p.action_items||[]).map(a=>
    `<tr><td>${esc(a.owner)}</td><td>${esc(a.task)}</td><td>${esc(a.due)}</td><td>${esc(a.priority)}</td></tr>`).join('');
  const spk = (r.speakers||[]).map(b=>
    `<div class="ln"><span class="ts">[${hms(b.start)}]</span>` +
    `<span class="spk">${esc(b.speaker)}:</span>${esc(b.text)}</div>`).join('');
  $('#out').innerHTML = `<div class="card">
    <div class="row">
      <div class="stat">Язык: <b>${esc(r.language)}</b></div>
      <div class="stat">Аудио: <b>${r.audio_seconds}s</b></div>
      <div class="stat">Финализация: <b>${t.finalize_sec??t.total_sec}s</b></div>
      ${t.mean_latency_sec?`<div class="stat">Средняя задержка: <b>${t.mean_latency_sec}s</b></div>`:''}
    </div>
    ${list('Executive Summary', p.summary)}
    ${list('Принятые решения', p.decisions)}
    ${topics?`<h2>Темы и тезисы</h2><ul>${topics}</ul>`:''}
    ${list('Открытые вопросы', p.open_questions)}
    ${list('Риски и блокеры', p.risks)}
    ${acts?`<h2>Поручения</h2><table><tr><th>Ответственный</th><th>Задача</th><th>Срок</th><th>Приоритет</th></tr>${acts}</table>`:''}
    <h2>Экспорт</h2>
    <a class="dl" href="/result/${r.id}/export.pdf">PDF</a>
    <a class="dl" href="/result/${r.id}/export.csv">CSV</a>
    <a class="dl" href="/result/${r.id}/export.json">JSON</a>
    <a class="dl" href="/result/${r.id}/export.ics">ICS</a>
    ${spk?`<h2>Стенограмма по спикерам</h2>${spk}`:
      (r.transcript?`<h2>Стенограмма</h2><pre>${esc(r.transcript)}</pre>`:'')}
  </div>`;
}

async function post(url, body){
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body||{})});
  const data = await res.json();
  if(!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

$('#start').onclick = async () => {
  $('#msg').textContent = ''; $('#out').innerHTML = ''; tape.innerHTML = '';
  const dev = $('#dev').value.trim();
  const spk = $('#spk').value;
  try{
    const r = await post('/live/start', {
      device: dev===''? null : parseInt(dev,10),
      num_speakers: spk===''? null : parseInt(spk,10),
      participants: $('#parts').value.split(',').map(s=>s.trim()).filter(Boolean),
      protocol_language: $('#plang').value
    });
    $('#msg').innerHTML = `Микрофон: <b>${esc(r.device)}</b>`;
    $('#start').disabled = true; $('#stop').disabled = false;
    setState('running');
  }catch(e){ $('#msg').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
};

$('#stop').onclick = async () => {
  $('#stop').disabled = true; setState('finalizing');
  $('#out').innerHTML = '<div class="card">Дописываю последние реплики, затем диаризация и LLM…</div>';
  try{ render(await post('/live/stop')); }
  catch(e){ $('#out').innerHTML = `<div class="card err">Ошибка: ${esc(e.message)}</div>`; }
  finally{ $('#start').disabled = false; }
};

$('#mic').onclick = async () => {
  $('#msg').textContent = 'Пишу 2 секунды…';
  try{
    const res = await fetch('/mic-test');
    const d = await res.json();
    if(!res.ok) throw new Error(d.detail || res.statusText);
    $('#msg').innerHTML = `${d.ok?'&#9989;':'&#9888;&#65039;'} <b>${esc(d.device)}</b> — ` +
      `RMS ${d.rms}, пик ${d.peak}${d.ok?'':' — слишком тихо, проверьте вход'}`;
  }catch(e){ $('#msg').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
};
</script></div></body></html>"""
