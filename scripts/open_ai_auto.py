#!/usr/bin/env python3
"""OpenAI auto-viewer demo.

This entrypoint is intentionally thin. The reusable logic lives in
`slicer_agent_engine.agents.openai_auto_agent` so benchmark scripts can import it.
"""

from __future__ import annotations


import sys
from pathlib import Path


def _ensure_imports() -> None:
    try:
        import slicer_agent_engine  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


_ensure_imports()


import logging
import os
from pathlib import Path

from slicer_agent_engine.agents.auto_agent import run_auto_task
from slicer_agent_engine.gemini_video import GeminiVideoAnalyzer
from slicer_agent_engine.llm.registry import build_builtin_model_tools, build_model_client, default_model_from_env, resolve_provider_name
from slicer_agent_engine.session import SessionManager
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.slicer_launcher import ensure_webserver
from slicer_agent_engine.tools import ToolContext
from slicer_agent_engine.video_renderer import VideoRenderer


def main() -> None:
    base_url = os.environ.get("SLICER_BASE_URL", "http://localhost:2016")
    bridge_dir = Path(__file__).resolve().parents[1] / "slicer_side"

    case_path = os.environ.get("CASE_PATH") or os.environ.get("DICOM_DIR")
    if not case_path:
        raise SystemExit("Please set CASE_PATH (or DICOM_DIR) to a DICOM folder or NIfTI file/folder.")
    case_path_p = Path(case_path).expanduser().resolve()

    out_dir = Path(os.environ.get("OUT_DIR", "./runs/openai_auto")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "open_ai_auto.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    llm_provider = os.environ.get("LLM_PROVIDER", "auto")
    llm_model = default_model_from_env()
    max_rounds = int(os.environ.get("MAX_ROUNDS", "12"))
    enable_code_interpreter = os.environ.get("OPENAI_ENABLE_CODE_INTERPRETER", "").strip().lower() in {"1", "true", "yes", "on"}
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT") or None

    # Gemini (optional)
    gemini = None
    if os.environ.get("GEMINI_API_KEY"):
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
        gemini = GeminiVideoAnalyzer(model=gemini_model)

    # Start/ensure Slicer
    ensure_webserver(
        base_url=base_url,
        slicer_executable=Path(os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer")),
        bridge_dir=bridge_dir,
        require_exec=True,
        require_slice_render=True,
    )

    # Connect
    client = SlicerClient(base_url=base_url)
    session = SessionManager(out_dir=out_dir)
    session.log_run_event("script_start", script="open_ai_auto.py", case_path=str(case_path_p), cli_env={"LLM_PROVIDER": llm_provider, "LLM_MODEL": llm_model, "MAX_ROUNDS": max_rounds})
    video = VideoRenderer()
    ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video, gemini=gemini)

    resolved_provider = resolve_provider_name(llm_provider, llm_model)
    model_client = build_model_client(resolved_provider, model=llm_model)
    extra_model_tools = build_builtin_model_tools(resolved_provider, model=llm_model, enable_code_interpreter=enable_code_interpreter)

    instructions = (
        "You are a radiology assistant using a remote multimodal medical image viewer (3D Slicer) for MRI/CT/PET. "
        "The case is opened before you start, and the current scene inventory is provided in the prompt. "
        "Use the provided core viewer tools first. When you adjust the view (WL/zoom/scroll), use tools that return updated observations. "
        "If scrolling pushes a slice view to a blank or off-target position, inspect get_viewer_state and use get_slice_offset_range plus set_slice_offset to recover the correct S/A/L slice position; use fit if the field of view is wrong. "
        "Activate optional packs only when the task clearly requires them. "
        "If you need to analyze an MP4 cine, call gemini_analyze_video. "
        "Finally, write a concise radiology-style report."
    )
    if extra_model_tools:
        instructions += " A provider-native code execution tool is enabled. Use it for multi-pass visual inspection and precise 0..999 bbox/point localization when needed."
    user_message = (
        "Please inspect the case and produce a short report (findings + impression). Choose the modality/sequence/series that best answers the task. "
        f"Case path: {case_path_p}"
    )

    result = run_auto_task(
        provider=resolved_provider,
        model_client=model_client,
        model=llm_model,
        ctx=ctx,
        case_path=case_path_p,
        instructions=instructions,
        user_message=user_message,
        max_rounds=max_rounds,
        extra_model_tools=extra_model_tools,
        reasoning_effort=reasoning_effort,
    )

    (out_dir / "report.txt").write_text(result.text, encoding="utf-8")
    session.log_run_event("script_finished", script="open_ai_auto.py", final_text=result.text)
    print("\n===== REPORT =====\n")
    print(result.text)
    print("\nOutputs:")
    print(f"  out_dir: {out_dir}")
    print(f"  log: {log_path}")
    print(f"  trace: {session.trace_path}")
    print(f"  evidence: {session.evidence_path}")
    print(f"  runlog: {session.runlog_path}")


if __name__ == "__main__":
    main()
