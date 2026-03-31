![Foxforge Banner](docs/images/banner.png)

# Foxforge

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

## What Foxforge is for

Foxforge is built for people who want one practical system to:

- think through ideas in chat,
- run research and synthesis,
- organize family and life operations,
- build tools and content,
- and keep long-lived context in a usable "Second Brain."

The design philosophy is local-first and user-owned:

- local execution by default,
- inspectable code and prompts,
- composable components,
- and no required cloud lock-in.

## Core capabilities

- **Orchestrated chat** with intent routing across work lanes.
- **Life Admin + Second Brain** for durable personal context and memory-aware workflows.
- **Second Brain memory** for persistent personal/project context with review controls.
- **Topic system** for long-lived research domains and memory enrichment.
- **Research lane** for evidence collection and synthesis.
- **Make lane** for building apps, scripts, docs, and other artifacts.
- **Watchtower/briefing flows** for ongoing situational awareness.
- **Bot integrations** for Discord, Slack, and Telegram — bring Foxforge into your existing chat workflows.

## Architecture

```text
                        ┌─────────────────────────────┐
                        │       Flask API (app.py)     │
                        │  auth · sessions · REST API  │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │   Foxforge Orchestrator    │
                        │   intent routing · context   │
                        └──┬──────┬──────┬────────┬───┘
                           │      │      │        │
              ┌────────────▼┐ ┌───▼───┐ ┌▼──────┐ ┌▼──────────┐
              │  Discovery  │ │ Make  │ │Personal│ │ Watchtower │
              │  (research) │ │(build)│ │(life)  │ │ (briefings)│
              └────────────┘ └───────┘ └────────┘ └────────────┘
                           │
              ┌────────────▼──────────────────────────┐
              │            Memory systems              │
              │  Project · Topic · Personal (2nd Brain)│
              └────────────────────────────────────────┘
                           │
              ┌────────────▼──────────────────────────┐
              │          External services             │
              │   Ollama (LLM) · SearXNG · Crawl4AI   │
              └────────────────────────────────────────┘
```

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

## Requirements

- **Python 3.10+**
- **Ollama** running locally
- **Docker** (optional, for web-foraging stack — SearXNG + Crawl4AI)
- **Optional extras** (install only if you want those features)
  - `requirements-optional-docs.txt` for PDF / DOCX / OCR helpers
  - `requirements-optional-bots.txt` for Discord bot support
- **GPU drivers** (optional but recommended for performance)
  - AMD: ROCm 6.x — RX 5000 series and newer
  - NVIDIA: CUDA toolkit — GTX 10xx and newer, any RTX series
  - CPU-only mode works without any GPU drivers, just slower

## Quick start

### Linux (Ubuntu 24.04 / 22.04 LTS) — recommended for GPU inference

A single script handles everything: system packages, Ollama, GPU drivers, Python deps, model pulls, owner account, Docker containers, and systemd auto-start.

```bash
chmod +x install_foxforge_linux.sh
./install_foxforge_linux.sh
```

The script will ask which GPU you have (AMD / NVIDIA / None) and install only what is relevant for your hardware. All other steps are automatic.

> **AMD users:** ROCm (~2 GB) is installed to enable GPU inference via ROCm/HIP.
> A log out / log back in is required after install for GPU group membership to take effect.
> Verify with `rocm-smi` after relogging.

> **NVIDIA users:** The NVIDIA driver and CUDA toolkit (~3–4 GB) are installed.
> A reboot may be required for CUDA to activate.
> Verify with `nvidia-smi` after rebooting.

> **CPU-only users:** Skip the GPU step. Everything works the same, inference is just slower.

After the script finishes:

```bash
sudo systemctl start foxforge
# or
./start_foxforge.sh
```

Open: `http://127.0.0.1:5050`

---

### Windows — installer flow

```powershell
powershell -ExecutionPolicy Bypass -File .\install_foxforge.ps1
```

Then launch:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_foxforge_web.ps1
```

Open: `http://127.0.0.1:5050`

For full recipient-friendly install details, see [INSTALL_GUIDE.md](INSTALL_GUIDE.md).

### Manual model pull (if needed)

```powershell
ollama pull dolphin3:8b
ollama pull deepseek-r1:8b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Optional web-foraging stack

Powers the web research (Fieldbook) lane. Requires Docker.

**Linux** — handled automatically by `install_foxforge_linux.sh`. To start manually:

```bash
docker start searxng crawl4ai
```

**Windows** — run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_web_foraging_stack.ps1
```

Default service ports:

| Service | Port |
|---|---:|
| SearXNG | 8080 |
| Crawl4AI | 11235 |

## Everyday usage model

Foxforge centers around three working lanes:

- **Discovery**: research, compare, and synthesize.
- **Make**: generate and iterate on implementations/artifacts.
- **Personal**: currently unavailable in this hard-cutover build.

The orchestrator coordinates these with shared context from memory and project/topic systems.

## Topics and knowledge domains

Topics are persistent domains used for research organization and memory context.
Foxforge supports broad topic typing, including technical, medical, finance, current events, and `animal_care`, among others.

## Configuration

Primary runtime model config:

- `SourceCode/configs/model_routing.json`

Useful startup script:

- `start_foxforge_web.ps1` (starts Ollama service checks and web app launch flow)

## Data ownership and runtime state

Foxforge stores local runtime state inside this repo folder (or derived runtime paths), including:

- memory/state under `Runtime/`,
- generated artifacts under `Projects/`.

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

Compile-check major modules:

```bash
python3 -m compileall SourceCode tests smoke_test.py run_integration_tests.py tools
```

Manual browser verification:

```bash
python3 tools/browser_headless_smoke.py
```

Maintainers should refresh the pinned lockfile from a clean environment with:

```bash
python3 tools/refresh_requirements_lock.py
```

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

**Linux:**
```bash
sudo systemctl restart ollama
sudo journalctl -u ollama -n 50
```

**Windows:**
```powershell
ollama serve
```

### Foxforge not starting

**Linux:**
```bash
sudo journalctl -u foxforge -n 50
```

**Windows:** re-run the start script and check the terminal output.

### GPU not being used by Ollama

**AMD (Linux):** Verify ROCm is working and your user is in the `render` and `video` groups:
```bash
rocm-smi
groups $USER   # should include render and video
```
If groups are missing, run `sudo usermod -aG render,video $USER` and log out / back in.

**NVIDIA (Linux):** Verify the driver and CUDA are active:
```bash
nvidia-smi
```
If not found, reboot — NVIDIA drivers often require a full reboot to load.

### First-owner setup incomplete

**Linux:**
```bash
./install_foxforge_linux.sh   # re-running is safe, skips completed steps
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File .\install_foxforge.ps1
```

### Port conflict on web startup

**Linux** — edit the systemd service or set the environment variable before starting:
```bash
sudo systemctl edit foxforge
# Add: Environment="FOXFORGE_WEB_PORT=5051"
sudo systemctl restart foxforge
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File .\start_foxforge_web.ps1 -WebPort 5051
```

### App behavior seems stale or inconsistent

```bash
python3 smoke_test.py
python3 run_integration_tests.py
python3 tools/ui_phase_smoke.py
python3 tools/repo_health_check.py
```

## Project status

Foxforge is functional and actively used. It is in an **experimental** phase — APIs and config formats may change between releases.

- CI runs on Python 3.10 and 3.12 on every push
- Tested on Ubuntu 24.04 LTS (primary platform) and Windows 11
- GPU-accelerated inference via AMD ROCm 6.x or NVIDIA CUDA; CPU-only also works

Issues and pull requests welcome. Large feature additions should open a discussion first.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Foxforge is released under the [Guideboard Service-Only License 1.0](LICENSE).

- Commercial services around the software are allowed (consulting, integration, support).
- Selling the software product itself is not allowed.
- This is source-available and not an OSI open source license.

Dependency license notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
