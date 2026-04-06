# ComfyUI Image + Video Setup

This guide contains the full setup details for Foxforge image generation and image-to-video workflows.

## Overview

Foxforge connects to a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance for visual generation tasks. ComfyUI can run on any machine on your network and does not need to be on the same host as Foxforge.

Set the ComfyUI address in `SourceCode/configs/model_routing.json` under `base_url` in the image generation lanes.

## Image generation setup

**ComfyUI custom nodes required:**
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (required for GGUF model loading)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (required for video output via `VHS_VideoCombine`)

Install via ComfyUI Manager or clone into `ComfyUI/custom_nodes/`.

## Style presets and required models

Foxforge ships with two tiers of image style presets:

- **Pony XL** (higher quality, ~8 GB VRAM)
- **Classic SD** (faster, lower VRAM)

You only need to download models for the styles you plan to use.

### Pony XL presets (recommended)

Base checkpoint:

| File | Destination |
|---|---|
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | `models/checkpoints/` |
| `sdxl_vae.safetensors` | `models/vae/` |

Each LoRA preset also needs its own file in `models/loras/`.
Download from [Civitai](https://civitai.com) and search by preset name.

| Preset | LoRA file |
|---|---|
| Borderfox | `borderfox_v2.safetensors` *(or variant — check in-app install hint)* |
| Painterly | *(see in-app install hint)* |
| UwU Figurine | *(see in-app install hint)* |
| & The Hound | *(see in-app install hint)* |
| Realism | *(see in-app install hint)* |
| Fixel | `sakuemonq-000011.safetensors` |

If a LoRA is missing, the style button shows as disabled with an install hint.

Background enhancement (BG+) uses a second LoRA:

| File | Destination |
|---|---|
| `zy_Detailed_Backgrounds_v1.safetensors` | `models/loras/` |

### Classic SD presets (SD1.5 / SD3.5)

| File | Destination |
|---|---|
| `v1-5-pruned-emaonly.safetensors` | `models/checkpoints/` |
| `sd3.5_medium-Q4_K_M.gguf` | `models/diffusion_models/` |
| `clip_l.safetensors` | `models/text_encoders/` |
| `clip_g.safetensors` | `models/text_encoders/` |
| `t5xxl_fp8_e4m3fn.safetensors` | `models/text_encoders/` |
| `sd3.5_vae.safetensors` | `models/vae/` |

## Image-to-video setup (Wan2.2)

Foxforge can animate a still image into a short clip. Recommended pipeline: **Wan2.2 I2V** via ComfyUI.

### Hardware requirements

| Variant | VRAM | Notes |
|---|---|---|
| Wan2.2 5B Q4_K_M (~3.4 GB model) | 8 GB+ | Minimum viable; expect ~5-10 min per clip |
| Wan2.2 14B Q4_K_M (~9.6 GB model) | 16 GB+ | Recommended for quality |
| Wan2.2 14B Q8_0 (~15.4 GB model) | 24 GB+ | Best quality |

### Model downloads

**5B variant** — [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)

| File | Destination |
|---|---|
| `Wan2.2-TI2V-5B-Q4_K_M.gguf` *(or your preferred quant)* | `models/diffusion_models/` |
| `wan2.2_vae.safetensors` | `models/vae/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` |

**14B variant** — [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)

The 14B uses two separate expert models (HighNoise + LowNoise):

| File | Destination |
|---|---|
| `Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf` | `models/diffusion_models/` |
| `Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf` | `models/diffusion_models/` |
| `wan_2.1_vae.safetensors` | `models/vae/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` |

### ComfyUI workflow setup

Wan2.2 uses `Wan22ImageToVideoLatent` instead of `WanImageToVideo`.
Export the workflow template from ComfyUI:

1. Open ComfyUI in your browser.
2. Load the official Wan2.2 I2V example from [docs.comfy.org/tutorials/video/wan/wan2_2](https://docs.comfy.org/tutorials/video/wan/wan2_2).
3. Run a test generation to confirm models load.
4. Go to **Settings -> Export (API format)** and save as `wan22_i2v_480p.json`.
5. Place the file at `SourceCode/configs/comfyui_workflows/wan22_i2v_480p.json`.

### Activating Wan2.2 in Foxforge

Update `SourceCode/configs/model_routing.json` and set `video_generation.model_name` to your downloaded model file.

Then activate Wan2.2 in `SourceCode/orchestrator/services/agent_registry.py`:

```python
# Change this:
registry.register("video_gen", StableVideoAgent())        # SVD XT
registry.register("video_gen_wan", ImageToVideoAgent())   # Wan2.2 (dormant)

# To this:
registry.register("video_gen", ImageToVideoAgent())       # Wan2.2 (active)
registry.register("video_gen_svd", StableVideoAgent())    # SVD XT (dormant)
```

Restart Foxforge after changes.

### Low-VRAM fallback (SVD XT)

For less than 8 GB VRAM, use **Stable Video Diffusion XT 1.1**.

| File | Destination |
|---|---|
| `svd_xt_1_1.safetensors` | `models/checkpoints/` |

SVD XT is the default active pipeline in shipped config and uses `SourceCode/configs/comfyui_workflows/svd_xt_i2v.json`.
