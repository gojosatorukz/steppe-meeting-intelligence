#!/usr/bin/env bash
# Run this BEFORE the hackathon demo, and again any time something feels broken.
# It is read-only: it checks, it never installs, converts, or changes anything
# (network state included — it warns about wifi, it never touches it).
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; WARN=0; FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

head_ "Toolchain"
command -v uv     >/dev/null 2>&1 && ok "uv $(uv --version | awk '{print $2}')" || bad "uv missing — 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')" || bad "ffmpeg missing — audio decode will fail"

head_ "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    ok "GPU visible: $GPU_NAME, ${FREE_MB} MiB free"
    [ "${FREE_MB:-0}" -lt 3000 ] 2>/dev/null && warn "less than 3 GiB free VRAM — STT+LLM handoff needs headroom, close other GPU apps"
  else
    bad "nvidia-smi present but no GPU responding — STT will fall back to CPU (slow)"
  fi
else
  warn "nvidia-smi not found — STT_DEVICE=auto will fall back to CPU (slow)"
fi

head_ "Ollama + LLM"
if curl -sS -m 3 -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
  ok "Ollama service reachable on :11434"
  if ollama list 2>/dev/null | grep -q 'qwen3.5:4b'; then
    ok "qwen3.5:4b present"
  else
    bad "qwen3.5:4b not pulled — 'ollama pull qwen3.5:4b'"
  fi
else
  bad "Ollama not reachable on localhost:11434 — is the service running?"
fi

head_ "Model weights"
[ -f api/models/kk-turbo-ct2/model.bin ] && ok "Kazakh STT model present (kk-turbo-ct2)" \
  || bad "api/models/kk-turbo-ct2/model.bin missing — run the ct2-transformers-converter command in README.md"

if ls api/models/diarization/**/*.onnx >/dev/null 2>&1 || find api/models/diarization -name '*.onnx' 2>/dev/null | grep -q .; then
  ok "diarization ONNX models present"
else
  bad "diarization models missing — run api/scripts/get_diarization_models.sh"
fi

[ -f api/models/vad/silero_vad.onnx ] && ok "VAD model present" \
  || bad "api/models/vad/silero_vad.onnx missing — run api/scripts/get_vad_model.sh"

head_ "Ports"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
  warn "port 8000 already in use — stop the old server or it will fail to bind"
else
  ok "port 8000 free"
fi

head_ "Microphone"
MIC_OUT=$(cd api && uv run python -c "import sounddevice as sd; print(sd.query_devices(kind='input'))" 2>&1)
if [ $? -eq 0 ] && [ -n "$MIC_OUT" ]; then
  ok "input device found: $(printf '%s' "$MIC_OUT" | head -1)"
else
  warn "could not confirm a microphone input device — check hardware/permissions before relying on /live (upload path still works)"
fi

head_ "Git identity"
GN=$(git config --get user.name  || true)
GE=$(git config --get user.email || true)
[ -n "$GN" ] && ok "user.name = $GN"   || bad "git user.name unset — 'git config --global user.name \"You\"'"
[ -n "$GE" ] && ok "user.email = $GE"  || bad "git user.email unset — 'git config --global user.email you@example.com'"
if git remote -v 2>/dev/null | grep -q .; then
  ok "remote: $(git remote get-url origin 2>/dev/null || echo '(non-origin remote)')"
else
  warn "no git remote — push will fail until you add one"
fi

head_ "Network (this product must run offline)"
NET_UP=0
if command -v nmcli >/dev/null 2>&1 && nmcli -t -f TYPE,STATE device 2>/dev/null | grep -qE '^wifi:connected$'; then
  NET_UP=1
elif command -v ip >/dev/null 2>&1 && ip route 2>/dev/null | grep -q '^default'; then
  NET_UP=1
fi
if [ "$NET_UP" -eq 1 ]; then
  warn "a network connection looks up (wifi/default route present) — switch it off visibly before the demo; locality is worth 20 judged points"
else
  ok "no active network route detected"
fi

printf '\n\033[1m%d passed, %d warnings, %d failures\033[0m\n' "$PASS" "$WARN" "$FAIL"
[ "$FAIL" -gt 0 ] && { echo "Fix the ✗ items before the clock starts."; exit 1; }
echo "Ready."
