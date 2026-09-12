#!/usr/bin/env bash
# Next.js + Claude, booting and streaming, in one command.
# Run ONLY after you know the task:  ./scripts/bootstrap-next.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d app ] && { echo "app/ already exists — delete it first if you mean to start over."; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "pnpm required: npm i -g pnpm"; exit 1; }

echo "==> Scaffolding Next.js into app/ ..."
pnpm create next-app@latest app \
  --ts --tailwind --eslint --app --src-dir --import-alias "@/*" --use-pnpm --yes

cd app
echo "==> Installing Anthropic SDK ..."
pnpm add @anthropic-ai/sdk

# Next.js reads .env.local from the app dir. Link to the repo-root .env so there is
# exactly one place secrets live.
ln -sf ../.env .env.local

mkdir -p src/app/api/chat

cat > src/app/api/chat/route.ts <<'TS'
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from the environment

export const runtime = "nodejs";
export const maxDuration = 60;

const SYSTEM = "You are a helpful assistant.";

export async function POST(req: Request) {
  const { messages } = (await req.json()) as {
    messages: Anthropic.MessageParam[];
  };

  const stream = client.beta.messages.stream({
    model: "claude-opus-5",
    max_tokens: 8000,
    system: SYSTEM,
    messages,
    // If a safety classifier declines the request, the server retries on a
    // fallback model instead of handing you an empty response mid-demo.
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
  });

  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      async start(controller) {
        try {
          for await (const event of stream) {
            if (
              event.type === "content_block_delta" &&
              event.delta.type === "text_delta"
            ) {
              controller.enqueue(encoder.encode(event.delta.text));
            }
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          controller.enqueue(encoder.encode(`\n\n[error] ${msg}`));
        } finally {
          controller.close();
        }
      },
    }),
    { headers: { "Content-Type": "text/plain; charset=utf-8" } },
  );
}
TS

cat > src/app/page.tsx <<'TSX'
"use client";

import { useState } from "react";

type Msg = { role: "user" | "assistant"; content: string };

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;

    const next: Msg[] = [...messages, { role: "user", content: input }];
    setMessages([...next, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next }),
      });
      if (!res.body) throw new Error("no response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages([...next, { role: "assistant", content: acc }]);
      }
    } catch (err) {
      setMessages([
        ...next,
        { role: "assistant", content: `[error] ${String(err)}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex h-dvh max-w-2xl flex-col gap-4 p-6">
      <h1 className="text-lg font-semibold">Chat</h1>

      <div className="flex-1 space-y-4 overflow-y-auto">
        {messages.map((m, i) => (
          <div key={i} className="space-y-1">
            <div className="text-xs uppercase tracking-wide opacity-50">
              {m.role}
            </div>
            <div className="whitespace-pre-wrap">
              {m.content || <span className="opacity-40">…</span>}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={send} className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Say something"
          className="flex-1 rounded border border-black/20 px-3 py-2 dark:border-white/20"
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-foreground px-4 py-2 text-background disabled:opacity-40"
        >
          {busy ? "…" : "Send"}
        </button>
      </form>
    </main>
  );
}
TSX

echo
echo "==> Done. Next:"
echo "    make dev     (or: cd app && pnpm dev)"
echo "    open http://localhost:3000"
