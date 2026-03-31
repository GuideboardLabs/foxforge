#!/usr/bin/env bash
# setup_comfyui.sh — Install ComfyUI + FLUX.1 Schnell GGUF for Foxforge
# Target hardware: AMD RX 5700 XT (gfx1010), ROCm
#
# What this script does:
#   1. Clone ComfyUI to ~/ComfyUI
#   2. Install ROCm PyTorch + ComfyUI Python deps
#   3. Install ComfyUI-GGUF custom node
#   4. Download FLUX.1 Schnell GGUF Q5_K_S + CLIP/T5/VAE models
#   5. Fix model filenames in Foxforge model_routing.json
#   6. Create start_comfyui.sh convenience script

set -euo pipefail

COMFYUI_DIR="${COMFYUI_INSTALL_DIR:-$HOME/ComfyUI}"
FOXFORGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTING_JSON="$FOXFORGE_DIR/SourceCode/configs/model_routing.json"

# ── Actual Hugging Face filenames (case-sensitive) ──────────────────────────
UNET_FILENAME="flux1-schnell-Q5_K_S.gguf"
CLIP_L_FILENAME="clip_l.safetensors"
T5_FILENAME="t5xxl_fp8_e4m3fn.safetensors"
VAE_FILENAME="ae.safetensors"

# ── Hugging Face token (required for gated models) ──────────────────────────
# black-forest-labs/FLUX.1-schnell is gated — you must:
#   1. Create a HF account at https://huggingface.co
#   2. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-schnell
#   3. Generate a token at https://huggingface.co/settings/tokens (read access is enough)
#   4. Pass it: HF_TOKEN=hf_xxx ./setup_comfyui.sh
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is not set."
    echo ""
    echo "The FLUX VAE (ae.safetensors) is in a gated HF repo and requires authentication."
    echo ""
    echo "Steps:"
    echo "  1. Sign in at https://huggingface.co"
    echo "  2. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-schnell"
    echo "  3. Create a token at https://huggingface.co/settings/tokens (read scope)"
    echo "  4. Re-run: HF_TOKEN=hf_yourtoken ./setup_comfyui.sh"
    exit 1
fi

# ── Colours ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
die()   { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# ── Prereq check ─────────────────────────────────────────────────────────────
for cmd in git python3 wget; do
    command -v "$cmd" &>/dev/null || die "Required command not found: $cmd"
done

# ── 1. Clone ComfyUI ─────────────────────────────────────────────────────────
if [[ -d "$COMFYUI_DIR/.git" ]]; then
    info "ComfyUI already cloned at $COMFYUI_DIR — pulling latest"
    git -C "$COMFYUI_DIR" pull --ff-only
else
    info "Cloning ComfyUI → $COMFYUI_DIR"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

# ── 2. Python environment ────────────────────────────────────────────────────
VENV="$COMFYUI_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
    info "Creating Python venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

info "Installing ROCm PyTorch for gfx1010 (RX 5700 XT) ..."
# ROCm 6.x wheel — compatible with gfx1010 via HSA_OVERRIDE_GFX_VERSION=10.3.0
pip install --upgrade pip --quiet
pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2 \
    --quiet

info "Installing ComfyUI requirements ..."
pip install -r "$COMFYUI_DIR/requirements.txt" --quiet

# ── 3. ComfyUI-GGUF custom node ──────────────────────────────────────────────
GGUF_NODE_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
if [[ -d "$GGUF_NODE_DIR/.git" ]]; then
    info "ComfyUI-GGUF already installed — pulling latest"
    git -C "$GGUF_NODE_DIR" pull --ff-only
else
    info "Installing ComfyUI-GGUF custom node ..."
    git clone https://github.com/city96/ComfyUI-GGUF.git "$GGUF_NODE_DIR"
fi
pip install -r "$GGUF_NODE_DIR/requirements.txt" --quiet

# ── 4. Model directories ─────────────────────────────────────────────────────
mkdir -p \
    "$COMFYUI_DIR/models/unet" \
    "$COMFYUI_DIR/models/clip" \
    "$COMFYUI_DIR/models/vae"

# Helper: download only if file is missing or zero-length
# Usage: download <dest> <url> <label> [hf_token]
download() {
    local dest="$1" url="$2" label="$3" token="${4:-}"
    if [[ -s "$dest" ]]; then
        info "$label already present — skipping"
    else
        info "Downloading $label ..."
        warn "  → $url"
        local wget_args=(--continue --show-progress -O "$dest")
        if [[ -n "$token" ]]; then
            wget_args+=(--header="Authorization: Bearer $token")
        fi
        wget "${wget_args[@]}" "$url" \
            || { rm -f "$dest"; die "Download failed: $label"; }
        info "$label download complete"
    fi
}

# ── 4a. FLUX.1 Schnell GGUF Q5_K_S (~6.5 GB) ────────────────────────────────
download \
    "$COMFYUI_DIR/models/unet/$UNET_FILENAME" \
    "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/$UNET_FILENAME" \
    "FLUX.1 Schnell Q5_K_S GGUF"

# ── 4b. CLIP-L (~246 MB) ────────────────────────────────────────────────────
download \
    "$COMFYUI_DIR/models/clip/$CLIP_L_FILENAME" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/$CLIP_L_FILENAME" \
    "CLIP-L"

# ── 4c. T5XXL FP8 (~4.9 GB) ─────────────────────────────────────────────────
download \
    "$COMFYUI_DIR/models/clip/$T5_FILENAME" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/$T5_FILENAME" \
    "T5XXL FP8"

# ── 4d. FLUX VAE (~335 MB) — gated repo, requires HF token ─────────────────
download \
    "$COMFYUI_DIR/models/vae/$VAE_FILENAME" \
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/$VAE_FILENAME" \
    "FLUX VAE (ae.safetensors)" \
    "$HF_TOKEN"

# ── 5. Update model_routing.json ─────────────────────────────────────────────
info "Updating model filenames in model_routing.json ..."
python3 - <<PYEOF
import json, pathlib, sys

path = pathlib.Path("$ROUTING_JSON")
try:
    data = json.loads(path.read_text())
except Exception as e:
    print(f"ERROR reading {path}: {e}", file=sys.stderr)
    sys.exit(1)

ig = data.get("image_generation", {})
updates = {
    "unet_name":   "$UNET_FILENAME",
    "clip_l_name": "$CLIP_L_FILENAME",
    "t5_name":     "$T5_FILENAME",
    "vae_name":    "$VAE_FILENAME",
}
changed = False
for key, val in updates.items():
    if ig.get(key) != val:
        print(f"  {key}: {ig.get(key)!r} → {val!r}")
        ig[key] = val
        changed = True

if changed:
    data["image_generation"] = ig
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("model_routing.json updated.")
else:
    print("model_routing.json already correct — no changes needed.")
PYEOF

# ── 6. Create start_comfyui.sh ───────────────────────────────────────────────
START_SCRIPT="$FOXFORGE_DIR/start_comfyui.sh"
cat > "$START_SCRIPT" <<STARTEOF
#!/usr/bin/env bash
# start_comfyui.sh — Launch ComfyUI for Foxforge image generation
#
# RX 5700 XT (gfx1010) needs HSA_OVERRIDE_GFX_VERSION so ROCm
# treats it as a supported gfx1030-class device.

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

COMFYUI_DIR="\${COMFYUI_INSTALL_DIR:-\$HOME/ComfyUI}"
source "\$COMFYUI_DIR/.venv/bin/activate"

echo "[ComfyUI] Starting on http://127.0.0.1:8188 ..."
echo "[ComfyUI] Press Ctrl+C to stop."
exec python "\$COMFYUI_DIR/main.py" --listen 0.0.0.0 --port 8188
STARTEOF
chmod +x "$START_SCRIPT"
info "Created $START_SCRIPT"

deactivate

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo ""
echo "  ComfyUI installed at : $COMFYUI_DIR"
echo "  Models:"
echo "    unet/ $UNET_FILENAME"
echo "    clip/ $CLIP_L_FILENAME"
echo "    clip/ $T5_FILENAME"
echo "    vae/  $VAE_FILENAME"
echo ""
echo "  To start ComfyUI:"
echo "    ./start_comfyui.sh"
echo ""
echo "  Then send a message like:"
echo "    \"draw a futuristic city at night\""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
