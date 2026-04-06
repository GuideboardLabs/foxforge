![Foxforge Banner](docs/images/banner.png)

# Foxforge

[![CI](https://github.com/GuideboardLabs/Foxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/GuideboardLabs/Foxforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Runtime](https://img.shields.io/badge/runtime-local--first-orange)
![LLM](https://img.shields.io/badge/LLM-Ollama-black)
![Status](https://img.shields.io/badge/status-experimental-yellow)
![License](https://img.shields.io/badge/license-Service--Only%20Source--Available-orange)

**Self-hosted AI workspace. No API keys. No cloud. No subscriptions.**

Foxforge is a local-first AI workspace for planning, research, and execution.
It combines conversational orchestration, structured planning, memory systems, and maker workflows in one self-hosted application.

## Why Foxforge

| | Foxforge | Cloud AI assistants |
|---|---|---|
| **Runs on** | Your own hardware | Provider's servers |
| **AI model** | Any Ollama-compatible model | Locked to provider |
| **Your data** | Stays on your machine | Sent to vendor |
| **Cost** | Free after hardware setup | Ongoing subscription |
| **Offline** | Works without internet | Requires connectivity |
| **Customizable** | Full source — fork and modify | Black box |

## Start Here (10 minutes)

### Fresh clone

```bash
git clone https://github.com/GuideboardLabs/Foxforge.git
cd Foxforge
```

### Linux (Ubuntu 24.04 / 22.04 LTS)

```bash
chmod +x install_foxforge_linux.sh
./install_foxforge_linux.sh
```

Then start the app:

```bash
sudo systemctl start foxforge
# or
./start_foxforge.sh
```

### Windows

```powershell
git clone https://github.com/GuideboardLabs/Foxforge.git
cd Foxforge
powershell -ExecutionPolicy Bypass -File .\install_foxforge.ps1
powershell -ExecutionPolicy Bypass -File .\start_foxforge_web.ps1
```

Open: `http://127.0.0.1:5050`

For recipient-friendly install steps, see [INSTALL_GUIDE.md](INSTALL_GUIDE.md).

## Feature Status

| Feature | Status | Notes |
|---|---|---|
| Discovery lane (research) | Available | Core workflow for web and local synthesis |
| Make lane (build/create) | Available | Code, docs, and artifact generation workflows |
| Topic system + memory | Available | Persistent context across sessions |
| Watchtower / briefing flows | Experimental | Active and evolving during experimental phase |
| Bot integrations (Discord/Slack/Telegram) | Experimental | Optional setup and environment dependent |
| Local image generation (ComfyUI) | Experimental | Optional external service, model-dependent |
| Image-to-video (Wan2.2/SVD XT) | Experimental | Optional, VRAM-dependent |
| Personal lane / Life Admin | Temporarily unavailable | Disabled in this hard-cutover build |

## Platform Support

| Platform | Status | Notes |
|---|---|---|
| Ubuntu 24.04 LTS | Tested (primary) | Preferred for GPU inference |
| Ubuntu 22.04 LTS | Tested | Installer supports this target |
| Windows 11 | Tested | Installer + web launcher supported |
| Other Linux distros | Experimental | May work, but not part of tested matrix |
| macOS | Untested | No official support commitment yet |

## Security Notes

- Foxforge is local-first and stores runtime state in your local `Runtime/` and outputs in `Projects/`.
- Startup scripts can bind the app to all interfaces (`0.0.0.0`) depending on launch mode, which can make it reachable from LAN/Tailscale.
- Keep local-only access by using loopback host (`127.0.0.1`) when desired.
- Change web host/port via `FOXFORGE_WEB_HOST` and `FOXFORGE_WEB_PORT` (or launch script flags).
- Set `FOXFORGE_WEB_PASSWORD` in environments where you expose beyond localhost.

## Docs Index

- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) — recipient-focused install guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution workflow and standards
- [ComfyUI image + video setup](docs/comfyui_image_video_setup.md) — full model, workflow, and VRAM guidance
- [Workspace tools](docs/workspace_tools.md) — utility scripts and tooling notes
- [Phase changelogs](docs/changelogs/) — milestone-level updates

## Core capabilities

- **Orchestrated chat** with intent routing across work lanes.
- **Second Brain memory** for persistent personal/project context with review controls.
- **Topic system** for long-lived research domains and memory enrichment.
- **Research lane** for evidence collection and synthesis.
- **Make lane** for building apps, scripts, docs, and other artifacts.
- **Watchtower/briefing flows** for ongoing situational awareness.
- **Bot integrations** for Discord, Slack, and Telegram.
- **Local image generation** and **image-to-video** via ComfyUI.

## Architecture

```text
                        ┌─────────────────────────────┐
                        │       Flask API (app.py)    │
                        │  auth · sessions · REST API │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │    Foxforge Orchestrator    │
                        │    intent routing · context  │
                        └──┬──────┬──────┬────────┬───┘
                           │      │      │        │
              ┌────────────▼┐ ┌───▼───┐ ┌▼──────┐ ┌▼──────────┐
              │  Discovery  │ │ Make  │ │Personal│ │ Watchtower │
              │  (research) │ │(build)│ │(life)  │ │ (briefings)│
              └────────────┘ └───────┘ └────────┘ └────────────┘
                           │
              ┌────────────▼──────────────────────────┐
              │            Memory systems             │
              │  Project · Topic · Personal (2nd Brain) │
              └────────────────────────────────────────┘
                           │
              ┌────────────▼──────────────────────────┐
              │          External services            │
              │  Ollama · SearXNG · Crawl4AI · ComfyUI │
              └────────────────────────────────────────┘
```

## Local image and video generation

Foxforge connects to [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation, enhancement, and image-to-video.
This is optional and can run on a separate machine from Foxforge.

You can use:

- Pony XL style presets (higher quality, ~8 GB VRAM target)
- Classic SD presets (faster/lower VRAM)
- Wan2.2 image-to-video (8+ GB VRAM, recommended 16+ GB)
- SVD XT fallback (4-6 GB VRAM)

For full setup details (required custom nodes, exact model files, workflow export, Wan2.2 activation, and fallback paths), see [ComfyUI image + video setup](docs/comfyui_image_video_setup.md).

## Requirements

- **Python 3.10+**
- **Ollama** running locally
- **Docker** (optional, for web-foraging stack — SearXNG + Crawl4AI)
- **ComfyUI** (optional, for image generation and image-to-video)
- **Optional extras**
  - `requirements-optional-docs.txt` for PDF / DOCX / OCR helpers
  - `requirements-optional-bots.txt` for Discord bot support
- **GPU drivers** (optional but recommended for performance)
  - AMD: ROCm 6.x — RX 5000 series and newer
  - NVIDIA: CUDA toolkit — GTX 10xx and newer, any RTX series

## Optional web-foraging stack

Powers the web research (Fieldbook) lane. Requires Docker.

Linux:

```bash
docker start searxng crawl4ai
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_web_foraging_stack.ps1
```

Default service ports:

| Service | Port |
|---|---:|
| SearXNG | 8080 |
| Crawl4AI | 11235 |

## Repository layout

| Path | Purpose |
|---|---|
| `SourceCode/orchestrator/` | Core orchestrator logic and routing |
| `SourceCode/web_gui/` | Flask app, API routes, frontend templates/static assets |
| `SourceCode/shared_tools/` | Shared services (memory, research, persistence, execution helpers) |
| `SourceCode/bots/` | Discord, Slack, and Telegram bot adapters |
| `SourceCode/configs/model_routing.json` | Model routing, fallback, and context config |
| `tests/` | Test suite |
| `docs/` | Architecture notes, release docs, planning artifacts |
| `tools/` | Utility scripts (health checks and developer tooling) |
| `Runtime/` | Local runtime state (generated at runtime; user-owned data) |
| `Projects/` | Generated project outputs and artifacts |

## Configuration

Primary runtime model config:

- `SourceCode/configs/model_routing.json`

Useful startup scripts:

- `start_foxforge_web.sh` (Linux launcher with host/port flags)
- `start_foxforge_web.ps1` (Windows launcher)

## Data ownership and runtime state

Foxforge stores local runtime state inside this repo folder (or derived runtime paths), including:

- memory/state under `Runtime/`
- generated artifacts under `Projects/`

The clean packaging script intentionally excludes user runtime/project outputs by default.

## Development workflow

Provision a dev environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

Run the standard check suite:

```bash
make check
```

Run checks individually:

```bash
python3 smoke_test.py
python3 run_integration_tests.py
python3 tools/ui_phase_smoke.py
python3 tools/repo_health_check.py
```

Optional feature installs:

```bash
pip install -r requirements-optional-docs.txt
pip install -r requirements-optional-bots.txt
```

## Changelog and release notes

- [docs/changelogs/phase19_accuracy_semantic_ui.md](docs/changelogs/phase19_accuracy_semantic_ui.md)
- [docs/changelogs/phase18c_confidence_and_memory.md](docs/changelogs/phase18c_confidence_and_memory.md)
- [docs/changelogs/phase18b_research_speed.md](docs/changelogs/phase18b_research_speed.md)
- [docs/changelogs/phase18a_query_routing.md](docs/changelogs/phase18a_query_routing.md)
- [docs/release_notes_phase18_optimization.md](docs/release_notes_phase18_optimization.md)
- [docs/release_notes_phase17_research_quality.md](docs/release_notes_phase17_research_quality.md)

## Packaging and distribution

Create a clean distributable ZIP:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_clean_zip.ps1
```

Create a GitHub-friendly ZIP (include docs/images, exclude installer EXE):

```powershell
powershell -ExecutionPolicy Bypass -File .\create_clean_zip.ps1 -IncludeDocsAndImages -IncludeInstallerExe:$false
```

Build installer EXE separately:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_installer_exe.ps1
```

## Troubleshooting

### Ollama not responding

Linux:

```bash
sudo systemctl restart ollama
sudo journalctl -u ollama -n 50
```

Windows:

```powershell
ollama serve
```

### Foxforge not starting

Linux:

```bash
sudo journalctl -u foxforge -n 50
```

Windows: re-run the start script and check the terminal output.

### GPU not being used by Ollama

AMD (Linux):

```bash
rocm-smi
groups $USER
```

If groups are missing, run `sudo usermod -aG render,video $USER` and log out / back in.

NVIDIA (Linux):

```bash
nvidia-smi
```

If not found, reboot and check again.

### First-owner setup incomplete

Linux:

```bash
./install_foxforge_linux.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_foxforge.ps1
```

### Port conflict on web startup

Linux:

```bash
sudo systemctl edit foxforge
# Add: Environment="FOXFORGE_WEB_PORT=5051"
sudo systemctl restart foxforge
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_foxforge_web.ps1 -WebPort 5051
```

## Project status

Foxforge is functional and actively used. It is in an **experimental** phase.
APIs and config formats may change between releases.

- CI runs on Python 3.10 and 3.12 on every push/PR
- Tested on Ubuntu 24.04 LTS (primary), Ubuntu 22.04 LTS, and Windows 11
- GPU acceleration via AMD ROCm or NVIDIA CUDA; CPU-only also works

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Foxforge is released under the [Guideboard Service-Only License 1.0](LICENSE).

- Commercial services around the software are allowed (consulting, integration, support).
- Selling the software product itself is not allowed.
- This is source-available and not an OSI open source license.

Dependency license notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
