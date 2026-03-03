# slicer-agent-engine

Minimal "Viewer-as-Engine" adapter for **3D Slicer WebServer**.

Design goals:

- **Do not modify 3D Slicer** (no fork, no patch).
- Keep all custom logic in an **external Python project**.
- Prefer stable WebServer endpoints for deterministic rendering:
  - `/slicer/slice` for 2D viewport PNG
  - `/slicer/screenshot`, `/slicer/threeD`, `/slicer/timeimage`
  - `/slicer/volumes`, `/slicer/mrml/*`, `/slicer/system/version`, etc.
- Use `/slicer/exec` **sparingly** for operations not covered by REST (DICOM import, ROI stats, export SEG, …)
  - Exec is routed through a **fixed-entry bridge module** (`slicer_side/slicer_agent_bridge.py`) to avoid
    copy-pasting long python strings and to keep the callable surface auditable.

This repo provides:

- `SlicerClient` (HTTP)
- `VideoRenderer` (ffmpeg)
- `SessionManager` (trace/state)
- `ToolContext` (agent-friendly functions)
- **MCP server** (stdio) exposing the tool surface
- Optional OpenAI + Gemini test harness (`scripts/open_ai_test.py`)

---

## Prerequisites

### 1) 3D Slicer with WebServer enabled

In Slicer:

- Open `WebServer` module
- Start server (default port `2016`)
- Advanced:
  - enable **Slicer API**
  - enable **Slicer API exec** (required for `load_dicom`, `roi_stats_ijk`, etc.)

Security note: do **NOT** expose the exec endpoint to untrusted networks.

### 2) Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (OpenAI + Gemini demo):

```bash
pip install -r requirements-agent.txt
```

### 3) ffmpeg

On macOS:

```bash
brew install ffmpeg
```

---

## Auto-starting Slicer WebServer (best effort)

Both `scripts/manual_test.py` and `scripts/open_ai_test.py` will:

1) try to ping `http://localhost:2016/slicer/system/version`
2) if unreachable, attempt to launch Slicer and run `slicer_side/bootstrap_webserver.py`

If Slicer is not discoverable, set:

```bash
export SLICER_EXECUTABLE="/Applications/Slicer.app/Contents/MacOS/Slicer"
```

If your Slicer/WebServer build differs, the bootstrap script may fail; in that case start WebServer manually.

---

## Quick manual test

Edit paths in `scripts/manual_test.py` if needed, then:

```bash
python scripts/manual_test.py
```

It will:

- load a DICOM folder
- list volumes
- render axial/sagittal/coronal mid-slice PNGs
- render a cine MP4 (scroll across the volume)
- compute ROI stats for a demo IJK box
- write `trace.jsonl` and `state.json` into the output folder

---

## OpenAI tool-calling demo + Gemini video reading

Set keys:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

Run:

```bash
python scripts/open_ai_test.py
```

Notes:

- The script uses **OpenAI Responses API function calling**.
- Slice PNGs are attached back to the model as `input_image` data URLs.
- GPT tool calls do not include video inputs here; if an MP4 is produced, the model can call:
  `gemini_analyze_video(video_path, prompt)`.
- Default Gemini model is `gemini-3-pro-preview` (requested). Gemini docs indicate this model is deprecated;
  prefer `gemini-3.1-pro-preview` by setting:

```bash
export GEMINI_MODEL="gemini-3.1-pro-preview"
```

---

## Run MCP server (stdio)

The MCP server exposes tools for:

- all documented `/slicer/*` endpoints (wrapped)
- the bridge-based exec tools (load_dicom, roi_stats_ijk, …)
- optional Gemini video analysis tool

Run:

```bash
python -m slicer_agent_engine.mcp_server \
  --base-url http://localhost:2016 \
  --bridge-dir ./slicer_side \
  --out-dir ./runs/mcp
```

Or after editable install:

```bash
pip install -e .
slicer-agent-mcp --base-url http://localhost:2016 --bridge-dir ./slicer_side --out-dir ./runs/mcp
```

---

## Project layout

- `slicer_agent_engine/`
  - `slicer_client.py`    : REST client for Slicer WebServer
  - `slicer_launcher.py`  : best-effort helper to auto-start Slicer WebServer
  - `video_renderer.py`   : ffmpeg wrapper to encode MP4 from PNG frames
  - `session.py`          : trace + state
  - `tools.py`            : agent tool functions
  - `mcp_server.py`       : MCP stdio server exposing tools
  - `gemini_video.py`     : Gemini Files API + video understanding wrapper
  - `server_http.py`      : (optional) FastAPI debug tool server
- `slicer_side/`
  - `slicer_agent_bridge.py`   : fixed-entry dispatcher called via `/slicer/exec`
  - `bootstrap_webserver.py`   : best-effort WebServer bootstrap for auto-start
- `scripts/`
  - `manual_test.py`: end-to-end test without any LLM
  - `open_ai_test.py`: OpenAI tool-calling demo + Gemini video analysis tool
