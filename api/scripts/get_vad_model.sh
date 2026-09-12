#!/usr/bin/env bash
# Download the Silero VAD model used by api/live.py (live mic transcription).
#
# This is the ONLY network access in the live feature: run it once, after that
# everything works fully offline. The model lands in api/models/vad/ which is
# gitignored -- never commit it.
#
# Usage:  bash api/scripts/get_vad_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(dirname "$SCRIPT_DIR")"
DEST="$API_DIR/models/vad"

VAD_NAME="silero_vad.onnx"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$VAD_NAME"

mkdir -p "$DEST"

fetch() {
    # fetch <url> <output-path>
    local url="$1" out="$2"
    if [ -f "$out" ]; then
        echo "already present: $out"
        return 0
    fi
    echo "downloading $url"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$out.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$out.part" "$url"
    else
        echo "error: need curl or wget" >&2
        return 1
    fi
    mv "$out.part" "$out"
}

fetch "$VAD_URL" "$DEST/$VAD_NAME"

echo
echo "VAD model in $DEST:"
ls -lh "$DEST/$VAD_NAME"
