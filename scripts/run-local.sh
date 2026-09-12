#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../api"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
elif command -v uv >/dev/null 2>&1; then
  exec uv run --offline uvicorn main:app --host 127.0.0.1 --port 8000
else
  echo "Нет Python-окружения backend. См. README.md."
  exit 1
fi
