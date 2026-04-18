![Foxforge Banner](docs/images/banner.png)

# Foxforge

[![CI](https://github.com/GuideboardLabs/Foxforge/actions/workflows/ci.yml/badge.svg)](https://github.com/GuideboardLabs/Foxforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Runtime](https://img.shields.io/badge/runtime-local--only-darkgreen)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20llama.cpp-black)
![Status](https://img.shields.io/badge/status-experimental-yellow)
![License](https://img.shields.io/badge/license-Service--Only%20Source--Available-orange)

**Self-hosted AI workspace. No API keys. No cloud. No subscriptions. No frontier model calls. Ever.**

Foxforge is a local-only AI workspace for research, writing, and software generation.
It routes requests through specialized multi-agent pipelines — each lane a coordinated team of models working in defined stages toward a quality-controlled output. Everything runs on your own hardware, on models you control, with data that never leaves your machine.

There is no external API integration and there never will be. The architecture is deliberately closed to frontier providers.

## Why Foxforge

| | Foxforge | Cloud AI assistants |
|---|---|---|
| **Runs on** | Your own hardware | Provider's servers |
| **AI models** | Any Ollama-compatible or llama.cpp model | Locked to provider |
| **Your data** | Stays on your machine — always | Sent to vendor |
| **API keys** | None required, none accepted | Required |
| **Cost** | Free after hardware setup | Ongoing subscription |
| **Offline** | Fully functional without internet | Requires connectivity |
| **Customizable** | Full source — fork and modify | Black box |

---

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

---

## The Lanes

Foxforge routes every request through one of three top-level lanes. Each lane is a pipeline of specialized agents running local models in sequence — not a single prompt, not a single model.

---

### Research Lane

Web and local evidence gathering, synthesis, and analysis. Powers the Fieldbook (web research) workflow.

**Deep Researcher** — 4 agents run in parallel (2 concurrent by default), each with a distinct research persona:

| Agent | Role | Model |
|---|---|---|
| Market Analyst | Market dynamics, alternatives, competitive positioning | qwen3:8b |
| Technical Researcher | Feasibility, bottlenecks, implementation constraints | deepseek-r1:8b (`think=True`) |
| Risk Researcher | Failure modes, mitigations, systemic constraints | deepseek-r1:8b (`think=True`) |
| Execution Planner | Practical sequencing, milestones, resources needed | qwen3:8b |

Each agent applies **evidence discipline**: findings are labeled `[E]` (evidence-backed), `[I]` (inferred), or `[S]` (speculative). A self-check rates quality (1–5) before output. A gap assessment identifies what's missing. A final skeptic pass (deepseek-r1:8b) validates the full picture before synthesis.

**Synthesizer** — unifies all four research streams into a coherent narrative with cross-persona consistency validation.

**Web foraging stack** (optional, Docker): SearXNG + Crawl4AI for live web research.

---

### Make Lane

Artifact generation. The Make lane covers seven distinct pools, each a purpose-built multi-agent pipeline for a specific class of deliverable.

All pools run entirely locally. No request touches a remote API.

---

#### Essay Pool

Short-to-medium documents: essays, reports, briefs.

**Pipeline — 6 stages:**

```
Outliner → Writers (≤3 parallel) → Critic → Revisor → Compositor → Proofreader
```

| Stage | Agent | Model | Role |
|---|---|---|---|
| 1 | Outliner | qwen2.5:7b | Thesis and per-section structure |
| 2 | Writers | qwen3:8b | ~400-word sections, parallel |
| 3 | Critic | deepseek-r1:8b | Flags gaps, repetition, drift |
| 4 | Revisor | qwen3:8b | Applies critic notes to flagged sections only |
| 5 | Compositor | qwen2.5:7b | Title, transitions, conclusion |
| 6 | Proofreader | deepseek-r1:8b | Fact contradictions, truncation, tense drift |

**Topic-aware templates** adjust section structure automatically:

- `history` — Background → Key events → Historiographical debate
- `science` — Evidence review (RCT > observational > anecdotal) → Implications
- `finance` — Market context → Risk factors → Thesis & recommendation *(not financial advice)*
- `medical` — Clinical summary → Evidence tiers → Safety profile → Disclaimers *(not medical advice)*
- `animal_care` — Vet-reviewed evidence → Safety profile → Owner considerations
- `politics` — Policy context → Stakeholder analysis → Counter-arguments
- `sports` — Statistical analysis → Risk & uncertainty → Analysis & outlook
- `underground` — No restrictions; all agents route to unrestricted model
- `technical`, `math`, `parenting`, `general` — Domain-specific variants

**Output targets:** `essay` (full treatment), `brief` (skips critic/revision/proofreader), `blog`, `social_post`

Underground topics route every agent to `huihui_ai/qwen3-abliterated:8b-Q4_K_M`.

---

#### Longform Pool

Extended structured outputs: long-form essays, guides, tutorials, video scripts, newsletters, press releases.

**Pipeline — 6 stages:**

```
Planner → Writers (parallel) → Critic (think=True) → Revisor → Compositor → Quality Gate
```

**Type-specific targets with word count enforcement:**

| Type | Word Range | Structure |
|---|---|---|
| `essay_long` | 1,800–3,500 | Hook → Argument pillars (3–5) → Steelman counterpoint → Synthesis |
| `essay_short` | 400–900 | Hook → Argument → Counterpoint → Close |
| `guide` | 1,000–2,500 | Prerequisites → Steps → Verification → Next steps |
| `tutorial` | 1,200–3,000 | Goal → Setup → Core logic → Integration → Troubleshooting |
| `video_script` | 1,500–4,000 | Hook (0–15s) → Premise → Beats → Turn/Reveal → CTA |
| `newsletter` | 600–1,200 | This Week → Worth Your Time → One Idea → Dessert |
| `press_release` | 400–700 | Headline → Dateline → Lede → Body → Boilerplate → Contact |

Video scripts include `[SEGMENT: name]` and `[B-ROLL: description]` markers for production use.

**Models:** `qwen3:8b` (planner/writer/compositor), `deepseek-r1:8b` (critic, `think=True`). Upgrades to `qwen2.5:32b` + `deepseek-r1:14b` automatically when available.

---

#### Content Pool

Short-form, high-velocity content: blog posts, social posts, emails.

**Pipeline — 6 stages:**

```
Planner → Writers (≤3 parallel) → Critic (think=True) → Revisor → Compositor → Quality Gate
```

| Type | Word Range | Notes |
|---|---|---|
| `blog` | 600–800 | Hook & headline → Context → Core (subheadings, examples) → CTA |
| `social_post` | 80–220 | Stop-scrolling hook → Body (2–3 lines) → CTA. Platform-aware voice. |
| `email` | 200–400 | Subject (<60 chars) → Front-loaded ask → Short body → Sign-off |

Drafter and Polish agents use `huihui_ai/qwen3-abliterated:8b-Q4_K_M` for creative latitude. Critic uses `deepseek-r1:8b`. Integrates learned feedback from the FeedbackLearningEngine across prior Make runs.

---

#### Specialist Pool

Domain-expert deliverables requiring specialized validation with enforced quality gates.

**Pipeline — 7 stages:**

```
Outliner → Writers → Domain Critic (think=True) → Revisor → Compositor → Quality Gate
```

**Supported domains:**

| Domain | Enforced Requirements |
|---|---|
| `medical` | Evidence tiers (RCT → observational → case study → opinion); safety profile; "not medical advice" |
| `finance` | Risk disclosures; assumption clarity; "not financial advice" |
| `sports` | Statistical claims with dates; injury/roster freshness notes |
| `history` | Source quality notes; historiographical balance; date/actor specificity |
| `game_design_doc` | Core loop clarity → Systems interlock → Scope feasibility → MVP vs. full vision |

Quality Gate enforces minimum 1,500 character outputs for medical/finance/history, required disclaimer presence, and truncation rejection.

---

#### Creative Pool

Long-form creative writing: novels, memoirs, books, screenplays.

**Pipeline — 5 stages:**

```
Story Planner → Scene Writers (sequential, continuity-aware) → Voice Critic → Revision → Compositor
```

Each scene writer receives the last 1,500 characters of the prior scene to maintain continuity. The Voice Critic checks tense, POV, pacing, and dialogue quality.

**Kind-specific formatting enforced:**

| Kind | Format Rules |
|---|---|
| `novel` | Scene headers, dialogue, interior monologue (italics), sensory anchoring, hooks |
| `memoir` | First-person intimate voice, time/place anchoring, reflective passages |
| `book` | Authority tone, thesis-driven, smooth evidence integration, reader address, subheadings |
| `screenplay` | INT./EXT. headings, action (present tense, ≤3 lines), character cues, sparse parentheticals, transitions (CUT TO / DISSOLVE TO) |

---

#### Web App Pool

Full-stack web applications: Flask backend + Vue 3 frontend + SQLite database.

**Pipeline — 8 sequential stages:**

```
DB Architect → API Implementer (+ py_compile fix loop) → Vue Architect → Vue Implementer
→ Integration Check → Integration Fixer → CSS Writer → README Writer
```

| Stage | Output |
|---|---|
| DB Architect | SQLite schema + Flask `db.py` helpers (parameterized queries, `sqlite3.Row`, PRAGMA foreign_keys) |
| API Implementer | Complete `app.py` (CRUD routes, CORS, error handling) — syntax-checked, auto-fixed up to 2 cycles |
| Vue Architect | Component/store plan derived from Flask routes |
| Vue Implementer | `index.html` (Vue 3 CDN, unpkg) + `app.js` (Composition API, fetch-based, no axios) |
| Integration Check | Flags route/fetch mismatches, CORS issues, JSON field name divergence |
| Integration Fixer | Applies fixes to both Flask and JS |
| CSS Writer | Full `styles.css` generated from HTML selectors (or extends existing) |
| README Writer | Setup instructions, DB init, API endpoint list, file structure |

**Extend Mode** — detects existing builds automatically; incremental updates preserve working code rather than regenerating from scratch.

**Output structure:**
```
Projects/{slug}/implementation/{timestamp}_app/
├── schema.sql
├── db.py
├── app.py
├── templates/index.html
├── static/app.js
├── static/styles.css
├── README.md
├── BUILD_SUMMARY.md
└── INTEGRATION_NOTES.md   (if integration issues were found and fixed)
```

**Model:** `qwen2.5-coder:7b` (or `qwen2.5-coder:14b` when available, all stages)

---

#### Desktop App Pool

Desktop applications: .NET 8 + Avalonia UI, MVVM, Windows-first with Linux portability.

**Stack:** Avalonia 11.x UI framework + ReactiveUI ViewModels + SQLite data layer

**Pipeline — 7 sequential stages:**

```
Specifier → Architect → ViewModel Impl → View Impl → Services Impl → Build Check → README Writer
```

| Stage | Output |
|---|---|
| Specifier | App name, features, state model, data layer, UI layout, external dependencies |
| Architect | Full project scaffold: `.sln`, `.csproj`, `Program.cs`, `App.axaml` |
| ViewModel Impl | ReactiveUI ViewModels with `[Reactive]` properties and `ReactiveCommand`s |
| View Impl | AXAML Views (data-bound, no code-behind logic) + minimal code-behind files |
| Services Impl | `IService` interfaces + implementations (repositories, file I/O, etc.) |
| Build Check | Project structure validation, dotnet syntax check |
| README Writer | Windows build steps, Linux port notes, MVVM architecture overview |

**Output structure:**
```
Projects/{slug}/desktop_apps/{AppName}/
├── README.md
├── .gitignore
├── {AppName}.sln
└── src/{AppName}/
    ├── {AppName}.csproj
    ├── App.axaml / App.axaml.cs
    ├── Program.cs
    ├── ViewModels/
    ├── Views/
    ├── Models/
    └── Services/
```

**Models:** `qwen2.5-coder:14b` (architect/implementation stages), `qwen3:8b` (spec/readme)

---

### Talk Lane

Conversational orchestration. Requests that aren't research or build tasks route here — the Reynard layer handles multi-turn dialogue, memory retrieval, and personal context via `dolphin3:8b`. The intent confirmer (`gemma3:4b`, <2s) gates every incoming request before any expensive pipeline fires.

---

## Model Distribution

| Task | Model | Context |
|---|---|---|
| Orchestration / reasoning | deepseek-r1:8b | 12,288 |
| Research & synthesis | qwen3:8b | 12,288 |
| Conversation (Reynard) | dolphin3:8b | 8,192 |
| Creative writing | qwen3:8b | 12,288 |
| Content (unrestricted topics) | huihui_ai/qwen3-abliterated:8b-Q4_K_M | 8,192 |
| Specialist / longform | qwen2.5:32b / deepseek-r1:14b (if available) | 24,576 |
| Code (web apps) | qwen2.5-coder:7b / :14b | 12,288 |
| Desktop app scaffold | qwen2.5-coder:14b | 16,384 |
| Intent gate | gemma3:4b | 4,096 |
| Embeddings / RAG | qwen3-embedding:4b | — |

All models run locally via Ollama or llama.cpp. Model assignments are configurable in `SourceCode/configs/model_routing.json`.

---

## Inference Backends

Foxforge supports two local inference backends:

- **Ollama** — default backend; handles most models via the Ollama API
- **llama.cpp** (OpenAI-compatible endpoint) — for TurboQuant and custom quantized models; configured per-model in `model_routing.json` under `llama_cpp_servers`

The inference router automatically falls back to Ollama if a configured llama.cpp server is unreachable. Server backoff is 180s after failure.

---

## Architecture

```
                     ┌─────────────────────────────────┐
                     │         Flask Web GUI            │
                     │   auth · REST API · job queue    │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │       Intent Confirmer           │
                     │   gemma3:4b gate · <2s latency  │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │      Foxforge Orchestrator       │
                     │   intent routing · turn planner  │
                     └───┬─────────┬──────────┬────────┘
                         │         │          │
          ┌──────────────▼┐  ┌─────▼──────┐  ┌▼──────────────┐
          │  Research Lane │  │  Make Lane │  │   Talk Lane   │
          │                │  │            │  │               │
          │ Deep Researcher│  │ essay      │  │ Reynard layer │
          │ 4 agents       │  │ longform   │  │ dolphin3:8b   │
          │ parallel pairs │  │ content    │  │               │
          │                │  │ specialist │  │               │
          │ Synthesizer    │  │ creative   │  │               │
          └────────────────┘  │ web app    │  └───────────────┘
                              │ desktop    │
                              └─────┬──────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │          Memory Systems          │
                     │  Topic · Project · Second Brain  │
                     │  Conversation · Research store   │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │       Local Inference            │
                     │    Ollama · llama.cpp            │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │      Optional External Services  │
                     │  SearXNG · Crawl4AI · ComfyUI   │
                     └─────────────────────────────────┘
```

---

## Feature Status

| Feature | Status | Notes |
|---|---|---|
| Research lane (deep researcher + synthesizer) | Available | 4-persona parallel research with evidence discipline |
| Essay pool | Available | 6-stage pipeline, 10+ topic templates |
| Longform pool | Available | 7 output types with word-count enforcement |
| Content pool | Available | Blog, social, email with feedback learning |
| Specialist pool | Available | Medical, finance, history, sports, game design |
| Creative pool | Available | Novel, memoir, book, screenplay with continuity |
| Web app pool | Available | Flask + Vue 3 + SQLite, Extend Mode |
| Desktop app pool | Available | .NET 8 + Avalonia, MVVM scaffold |
| Topic system + Second Brain memory | Available | Persistent context across sessions |
| Intent confirmer | Available | Fast gate prevents accidental pool activation |
| Feedback learning engine | Available | Learns from successful Make outputs |
| Watchtower / briefing flows | Experimental | Active and evolving |
| Bot integrations (Discord / Slack / Telegram) | Experimental | Optional, environment-dependent |
| Local image generation (ComfyUI) | Experimental | Optional external service, model-dependent |
| Image-to-video (Wan2.2 / SVD XT) | Experimental | Optional, VRAM-dependent |

---

## Platform Support

| Platform | Status | Notes |
|---|---|---|
| Ubuntu 24.04 LTS | Tested (primary) | Preferred for GPU inference |
| Ubuntu 22.04 LTS | Tested | Installer supports this target |
| Windows 11 | Tested | Installer + web launcher supported |
| Other Linux distros | Experimental | May work, not in tested matrix |
| macOS | Untested | No official support commitment |

---

## Requirements

- **Python 3.10+**
- **Ollama** running locally (required)
- **Docker** (optional — for web-foraging stack: SearXNG + Crawl4AI)
- **ComfyUI** (optional — for local image generation and image-to-video)
- **Optional extras**
  - `requirements-optional-docs.txt` — PDF / DOCX / OCR helpers
  - `requirements-optional-bots.txt` — Discord bot support
- **GPU drivers** (optional but strongly recommended)
  - AMD: ROCm 6.x — RX 5000 series and newer
  - NVIDIA: CUDA toolkit — GTX 10xx and newer, any RTX series

---

## Optional Web-Foraging Stack

Powers the Research lane's live web foraging. Requires Docker.

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

---

## Local Image and Video Generation

Foxforge connects to [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation, enhancement, and image-to-video. This is optional and can run on a separate machine.

Supported configurations:

- Pony XL style presets (~8 GB VRAM)
- Classic SD presets (lower VRAM)
- Wan2.2 image-to-video (8+ GB VRAM, recommended 16+ GB)
- SVD XT fallback (4–6 GB VRAM)

For full setup details — required custom nodes, model files, workflow export, Wan2.2 activation, and fallback paths — see [ComfyUI image + video setup](docs/comfyui_image_video_setup.md).

---

## Security Notes

- Foxforge is local-only. No data is ever transmitted to an external AI provider.
- Startup scripts can bind to all interfaces (`0.0.0.0`) for LAN/Tailscale access.
- Use loopback (`127.0.0.1`) to restrict to local access only.
- Configure host/port via `FOXFORGE_WEB_HOST` and `FOXFORGE_WEB_PORT`.
- Set `FOXFORGE_WEB_PASSWORD` when exposing beyond localhost.

---

## Repository Layout

| Path | Purpose |
|---|---|
| `SourceCode/orchestrator/` | Orchestrator, intent routing, turn planner, Make catalog |
| `SourceCode/agents_make/` | All Make lane pools (essay, longform, content, specialist, creative, web app, desktop) |
| `SourceCode/agents_research/` | Deep researcher and synthesizer |
| `SourceCode/web_gui/` | Flask app, API routes, frontend templates and static assets |
| `SourceCode/shared_tools/` | Inference router, memory systems, research tools, activity bus |
| `SourceCode/bots/` | Discord, Slack, and Telegram bot adapters |
| `SourceCode/configs/model_routing.json` | Model assignments, inference servers, fallback config |
| `tests/` | Test suite |
| `docs/` | Architecture notes, changelogs, planning artifacts |
| `tools/` | Utility scripts: health checks, developer tooling |
| `Runtime/` | Local runtime state (generated at runtime; user-owned) |
| `Projects/` | Generated outputs and artifacts |

---

## Configuration

Primary model and routing config:

- `SourceCode/configs/model_routing.json` — model assignments per lane, llama.cpp server entries, context sizes

Useful startup scripts:

- `start_foxforge_web.sh` (Linux, host/port flags)
- `start_foxforge_web.ps1` (Windows)

---

## Development Workflow

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

---

## Packaging and Distribution

Create a clean distributable ZIP:

```powershell
powershell -ExecutionPolicy Bypass -File .\create_clean_zip.ps1
```

GitHub-friendly ZIP (include docs/images, exclude installer EXE):

```powershell
powershell -ExecutionPolicy Bypass -File .\create_clean_zip.ps1 -IncludeDocsAndImages -IncludeInstallerExe:$false
```

Build installer EXE:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_installer_exe.ps1
```

---

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

Windows: re-run the start script and check terminal output.

### GPU not used by Ollama

AMD (Linux):
```bash
rocm-smi
groups $USER
# If render/video groups missing:
sudo usermod -aG render,video $USER
# Log out and back in
```

NVIDIA (Linux):
```bash
nvidia-smi
# If not found, reboot and check again
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

---

## Changelog and Release Notes

- [docs/changelogs/phase19_accuracy_semantic_ui.md](docs/changelogs/phase19_accuracy_semantic_ui.md)
- [docs/changelogs/phase18c_confidence_and_memory.md](docs/changelogs/phase18c_confidence_and_memory.md)
- [docs/changelogs/phase18b_research_speed.md](docs/changelogs/phase18b_research_speed.md)
- [docs/changelogs/phase18a_query_routing.md](docs/changelogs/phase18a_query_routing.md)
- [docs/release_notes_phase18_optimization.md](docs/release_notes_phase18_optimization.md)
- [docs/release_notes_phase17_research_quality.md](docs/release_notes_phase17_research_quality.md)

---

## Docs Index

- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) — recipient-focused install guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution workflow and standards
- [ComfyUI image + video setup](docs/comfyui_image_video_setup.md) — model, workflow, and VRAM guidance
- [Workspace tools](docs/workspace_tools.md) — utility scripts and tooling notes
- [Phase changelogs](docs/changelogs/) — milestone-level updates

---

## Project Status

Foxforge is functional and actively used. It is in an **experimental** phase — APIs and config formats may change between releases.

- CI runs on Python 3.10 and 3.12 on every push/PR
- Tested on Ubuntu 24.04 LTS (primary), Ubuntu 22.04 LTS, and Windows 11
- GPU acceleration via AMD ROCm or NVIDIA CUDA; CPU-only also works

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Foxforge is released under the [Guideboard Service-Only License 1.0](LICENSE).

- Commercial services around the software are allowed (consulting, integration, support).
- Selling the software product itself is not allowed.
- This is source-available, not an OSI open source license.

Dependency license notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
