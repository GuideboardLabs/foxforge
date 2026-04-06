#!/usr/bin/env bash
# start_comfyui.sh — Launch ComfyUI for Foxforge image generation
#
# RX 5700 XT (gfx1010) needs HSA_OVERRIDE_GFX_VERSION so ROCm
# treats it as a supported gfx1030-class device.

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

COMFYUI_DIR="${COMFYUI_INSTALL_DIR:-$HOME/ComfyUI}"
source "$COMFYUI_DIR/.venv/bin/activate"
COMFYUI_EXTRA_ARGS="${COMFYUI_EXTRA_ARGS:---lowvram --disable-async-offload --use-split-cross-attention}"

echo "[ComfyUI] Starting on http://127.0.0.1:8188 ..."
echo "[ComfyUI] Press Ctrl+C to stop."
exec python "$COMFYUI_DIR/main.py" --listen 0.0.0.0 --port 8188 $COMFYUI_EXTRA_ARGS
