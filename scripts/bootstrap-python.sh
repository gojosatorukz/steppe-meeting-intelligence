#!/usr/bin/env bash
# FastAPI + Claude, booting and streaming, in one command.
# Run ONLY after you know the task:  ./scripts/bootstrap-python.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d api ] && { echo "api/ already exists — delete it first if you mean to start over."; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv required: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "==> Scaffolding FastAPI into api/ ..."
mkdir -p api && cd api
uv init --no-workspace --name api >/dev/null 2>&1 || uv init --name api >/dev/null
rm -f hello.py main.py
uv add anthropic fastapi "uvicorn[standard]" python-dotenv pydantic

cat > main.py <<'PY'
"""Minimal Claude chat API. Run: uv run uvicorn main:app --reload --port 8000"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Secrets live in the repo-root .env — one file, not one per service.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "claude-opus-5"
SYSTEM = "You are a helpful assistant."

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
app = FastAPI(title="hack api")

# Wide open on purpose: this is a hackathon, not production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model": MODEL, "key_set": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def generate():
        try:
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=8000,
                system=SYSTEM,
                messages=[m.model_dump() for m in req.messages],
                # If a safety classifier declines, the server retries on a fallback
                # model instead of handing you an empty response mid-demo.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIError as e:
            yield f"\n\n[error] {e}"

    return StreamingResponse(generate(), media_type="text/plain")
PY

echo
echo "==> Done. Next:"
echo "    make dev     (or: cd api && uv run uvicorn main:app --reload --port 8000)"
echo "    curl -N localhost:8000/chat -H 'Content-Type: application/json' \\"
echo "         -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
