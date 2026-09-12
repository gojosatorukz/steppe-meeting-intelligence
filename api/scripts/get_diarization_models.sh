#!/usr/bin/env bash
# Download the speaker-diarization models used by api/diarize.py.
#
# This is the ONLY network access in the diarization feature: run it once,
# after that everything works fully offline. Model files land in
# api/models/diarization/ which is gitignored -- never commit them.
#
# Usage:  bash api/scripts/get_diarization_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(dirname "$SCRIPT_DIR")"
DEST="$API_DIR/models/diarization"

SEG_NAME="sherpa-onnx-pyannote-segmentation-3-0"
SEG_TARBALL="$SEG_NAME.tar.bz2"
SEG_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/$SEG_TARBALL"

# NOTE: the upstream release tag really is spelled "recongition" (typo upstream).
EMB_NAME="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$EMB_NAME"

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

# 1. segmentation model (tarball -> directory containing model.onnx)
if [ -f "$DEST/$SEG_NAME/model.onnx" ]; then
    echo "already present: $DEST/$SEG_NAME/model.onnx"
else
    fetch "$SEG_URL" "$DEST/$SEG_TARBALL"
    echo "extracting $SEG_TARBALL"
    tar -xjf "$DEST/$SEG_TARBALL" -C "$DEST"
    rm -f "$DEST/$SEG_TARBALL"
fi

# 2. speaker-embedding model (plain .onnx)
fetch "$EMB_URL" "$DEST/$EMB_NAME"

echo
echo "diarization models in $DEST:"
ls -lh "$DEST/$SEG_NAME/model.onnx" "$DEST/$EMB_NAME"
