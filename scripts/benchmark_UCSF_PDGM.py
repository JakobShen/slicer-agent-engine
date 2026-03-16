#!/usr/bin/env python3
"""Benchmark: UCSF-PDGM 3-way classification.

Task (3-way multiple choice):
  A) Glioblastoma
  B) Oligodendroglioma / Astrocytoma
  C) No tumor

Ground truth is read from UCSF-PDGM-metadata_v5.csv.

This benchmark is *rule-based* (no LLM-as-judge): we extract the final letter
from model output and compare to the label derived from pathology.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_imports() -> None:
    """Allow running `python scripts/benchmark_UCSF_PDGM.py` without installing the package."""

    try:
        import slicer_agent_engine  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


_ensure_imports()

from slicer_agent_engine.agents.auto_agent import run_auto_task
from slicer_agent_engine.benchmarking.judge import extract_choice
from slicer_agent_engine.benchmarking.ucsf_pdgm import diagnosis_subtype, iter_cases
from slicer_agent_engine.gemini_video import GeminiVideoAnalyzer
from slicer_agent_engine.llm.registry import build_builtin_model_tools, build_model_client, default_model_from_env, resolve_provider_name
from slicer_agent_engine.session import SessionManager
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.slicer_launcher import ensure_webserver
from slicer_agent_engine.tools import ToolContext
from slicer_agent_engine.video_renderer import VideoRenderer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UCSF-PDGM benchmark (NIfTI).")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing UCSF-PDGM-*_nifti folders")
    p.add_argument("--csv_path", type=str, default="UCSF-PDGM-metadata_v5.csv", help="Metadata CSV path")
    p.add_argument("--provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), help="LLM provider name. Use auto to infer from --model (gemini -> Google, claude -> Anthropic, otherwise OpenAI).")
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model name")
    p.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all)")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_ucsf_pdgm", help="Directory for logs and outputs")
    p.add_argument("--max_rounds", type=int, default=12, help="Max tool-calling rounds")
    p.add_argument("--enable_code_interpreter", action="store_true", default=os.environ.get("OPENAI_ENABLE_CODE_INTERPRETER", "").strip().lower() in {"1", "true", "yes", "on"}, help="Expose a provider-native code execution tool during the run when supported (OpenAI Code Interpreter or Gemini code execution).")
    p.add_argument("--reasoning_effort", type=str, choices=["none", "minimal", "low", "medium", "high", "xhigh"], default=os.environ.get("LLM_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT") or None, help="Optional provider reasoning-effort hint.")
    p.add_argument("--base_url", type=str, default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"), help="Slicer WebServer base URL")
    p.add_argument(
        "--slicer_executable",
        type=str,
        default=os.environ.get("SLICER_EXECUTABLE", "/Applications/Slicer.app/Contents/MacOS/Slicer"),
        help="Path to Slicer executable",
    )
    p.add_argument("--include_extra", action="store_true", help="Load all non-seg NIfTI volumes (not just T1/T1c/T2/FLAIR)")
    p.add_argument("--include_segmentations", action="store_true", help="Also load segmentation masks (NOT recommended for classification benchmarks)")
    p.add_argument("--no_prefer_bias", action="store_true", help="Do not prefer *_bias.nii.gz when selecting modalities")
    p.add_argument("--enable_gemini", action="store_true", help="Enable Gemini video analysis tools (requires GEMINI_API_KEY)")
    p.add_argument("--force_gemini", action="store_true", help="Force prompt to require both keyframes and Gemini video analysis when using scroll")
    p.add_argument("--disturb", action="store_true", help="Add distractor options D/E/F to the multiple-choice prompt")
    p.add_argument("--disable_tool_packs", action="store_true", help="Disable optional tool packs and expose only the baseline/core tools.")
    p.add_argument("--balance_diagnosis_subtypes", action="store_true", help="Balance UCSF-PDGM sampling across Glioblastoma, Oligodendroglioma, and Astrocytoma as evenly as possible.")
    return p


def build_instructions(disturb: bool = False, force_gemini: bool = False, enable_code_execution_hint: bool = False, disable_tool_packs: bool = False) -> str:
    answer_set = "A|B|C|D|E|F" if disturb else "A|B|C"
    force_gemini_text = ""
    if force_gemini:
        force_gemini_text = "When you use scroll, you must do both keyframe extraction and Gemini analysis of the scroll video to avoid missing subtle details. "
    parts = [
        "You are an AI radiology assistant using a remote MRI viewer (3D Slicer). ",
        "The case has already been opened by the system before you start. ",
        "Do NOT call open_case unless the viewer is clearly empty/broken. ",
        "If you must call open_case, use the exact absolute path provided in the user message. ",
        "The case is already opened with a core multi-sequence MRI subset, and the current scene inventory is provided in the prompt. ",
        ("Use only the baseline/core tools already exposed in this run. " if disable_tool_packs else "Use the provided tools to inspect images and switch among the already loaded sequences. "),
        force_gemini_text,
        "If Gemini video analysis conflicts with keyframe evidence, prioritize the keyframe evidence. ",
        "Your task is a 3-way multiple choice classification based on MRI appearance. ",
        "If you are unsure, still choose the most likely option and explain why. ",
        f"Output must contain a final line exactly like: 'ANSWER: <{answer_set}>'.",
    ]
    base = "".join(parts)
    if enable_code_execution_hint:
        base += " A provider-native code execution tool is enabled. You may use it for multi-pass visual inspection or precise 0..999 image-space localization when the task benefits from that level of visual reasoning."
    return base


def build_user_message(case_id: str, case_path: Path, disturb: bool = False) -> str:
    case_path = case_path.expanduser().resolve()
    extra_options = ""
    answer_set = "A|B|C"
    if disturb:
        extra_options = (
            "  D) Dysembryoplastic neuroepithelial tumor (DNET) / Ganglioglioma\n"
            "  E) Pilocytic Astrocytoma\n"
            "  F) Central Neurocytoma / Ependymoma\n"
        )
        answer_set = "A|B|C|D|E|F"
    return (
        f"Case: {case_id}\n\n"
        f"Absolute case path (already opened): {case_path}\n\n"
        "Task: classify the case into ONE option:\n"
        "  A) Glioblastoma\n"
        "  B) Oligodendroglioma / Astrocytoma\n"
        "  C) No tumor\n"
        f"{extra_options}\n"
        "Steps you should follow:\n"
        "1) Review baseline views.\n"
        "2) Use the prompt-provided scene inventory to choose among the loaded sequences, then use select_volume to switch sequences (e.g., T1, T1c, T2, FLAIR).\n"
        "3) Use observe/scroll/zoom/wl as needed. Prefer scroll output='keyframes' unless you truly need video.\n\n"
        f"Write a brief rationale, then end with a single final line: ANSWER: <{answer_set}>."
    )


def extract_choice_with_disturb(text: str, *, disturb: bool = False) -> Optional[str]:
    pred = extract_choice(text)
    if pred is not None or not disturb:
        return pred

    m = re.search(r"\b(?:ANSWER|CHOICE|OPTION)\s*[:\-]?\s*([DEF])\b", text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper()

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines[-10:]):
        if re.fullmatch(r"[DEF]", ln, re.IGNORECASE):
            return ln.upper()
    return None




def _collect_tool_usage(runlog_path: Path) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    if runlog_path.exists():
        with runlog_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("kind")) != "run_event":
                    continue
                if str(rec.get("event_type")) != "tool_call_start":
                    continue
                tool_name = str(rec.get("tool_name") or "").strip()
                if not tool_name:
                    continue
                counts[tool_name] = counts.get(tool_name, 0) + 1
    return {
        "n_tool_calls": int(sum(counts.values())),
        "n_unique_tools": int(len(counts)),
        "tools_called": sorted(counts.keys()),
        "tool_call_counts": {k: counts[k] for k in sorted(counts.keys())},
    }


def _merge_tool_counts(dest: Dict[str, int], src: Dict[str, int]) -> None:
    for name, count in src.items():
        dest[str(name)] = int(dest.get(str(name), 0)) + int(count)


def _print_progress(message: str) -> None:
    print(message, flush=True)

def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    os.environ["SLICER_CASE_ROOT"] = str(data_root)
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = (data_root / csv_path).resolve()

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_dir / "benchmark.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("benchmark_ucsf_pdgm")

    bridge_dir = Path(__file__).resolve().parents[1] / "slicer_side"
    bootstrap_script = bridge_dir / "bootstrap_webserver.py"

    # Ensure Slicer once for the whole run.
    os.environ["SLICER_EXECUTABLE"] = str(Path(args.slicer_executable).expanduser())
    ensure_webserver(
        base_url=args.base_url,
        bootstrap_script=bootstrap_script,
        start_if_not_running=True,
        port=2016,
        require_exec=True,
        require_slice=True,
    )

    client = SlicerClient(base_url=args.base_url)
    video = VideoRenderer()

    gemini = None
    if args.enable_gemini and os.environ.get("GEMINI_API_KEY"):
        gemini = GeminiVideoAnalyzer(model=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"))

    resolved_provider = resolve_provider_name(args.provider, args.model)
    model_client = build_model_client(resolved_provider, model=args.model)
    extra_model_tools = build_builtin_model_tools(resolved_provider, model=args.model, enable_code_interpreter=bool(args.enable_code_interpreter))

    # Load dataset cases.
    limit = args.limit if args.limit and args.limit > 0 else None
    cases = list(
        iter_cases(
            data_root,
            csv_path,
            limit=limit,
            include_extra=bool(args.include_extra),
            include_segmentations=bool(args.include_segmentations),
            prefer_bias=not bool(args.no_prefer_bias),
            balance_diagnosis_subtypes=bool(args.balance_diagnosis_subtypes),
        )
    )
    log.info("Found %d cases under %s", len(cases), data_root)

    instructions = build_instructions(bool(args.disturb), bool(args.force_gemini), enable_code_execution_hint=bool(extra_model_tools), disable_tool_packs=bool(args.disable_tool_packs))

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_failed = 0
    aggregate_tool_call_counts: Dict[str, int] = {}
    total_tool_calls = 0

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)

        # Per-case session (trace/state/artifacts)
        session = SessionManager(out_dir=case_out)
        session.log_run_event("case_script_start", script="benchmark_UCSF_PDGM.py", case_id=case.case_id, model=args.model, max_rounds=int(args.max_rounds))
        ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video, gemini=gemini)

        user_msg = build_user_message(case.case_id, case.case_dir, bool(args.disturb))

        # Use explicit file list to avoid loading unwanted volumes.
        open_case_kwargs = {
            "preset": "mri_brain",
            "layout": "four_up",
            "clear_scene_first": True,
            "nifti_files": [str(p) for p in case.nifti_files],
            "include_segmentations": bool(args.include_segmentations),
        }

        core_sequences = ", ".join([p.name for p in case.nifti_files]) if case.nifti_files else "none"
        case_brief = (
            f"Case brief for {case.case_id}:\n"
            f"- CSV ID: {case.csv_id}\n"
            f"- Loaded core NIfTI files: {core_sequences}\n"
            "- The current scene inventory will also be shown after open_case so you do not need a separate listing tool.\n"
            "- Treat the loaded T1/T1c/T2/FLAIR-style sequences as your main evidence sources for this benchmark."
        )

        try:
            agent_res = run_auto_task(
                provider=resolved_provider,
                model_client=model_client,
                model=args.model,
                ctx=ctx,
                case_path=case.case_dir,
                instructions=instructions,
                user_message=user_msg,
                max_rounds=int(args.max_rounds),
                extra_model_tools=extra_model_tools,
                reasoning_effort=args.reasoning_effort,
                disable_tool_packs=bool(args.disable_tool_packs),
                model_request_timeout_sec=120.0,
                model_request_retry_on_timeout=1,
                open_case_kwargs=open_case_kwargs,
                case_brief_text=case_brief,
            )
            raw_text = agent_res.text
            (case_out / "agent_output.txt").write_text(raw_text, encoding="utf-8")

            pred = extract_choice_with_disturb(raw_text, disturb=bool(args.disturb))
            gt = case.label

            scored = gt in {"A", "B", "C"}
            correct = bool(scored and pred == gt)
            if scored:
                n_scored += 1
                n_correct += int(correct)

            tool_usage = _collect_tool_usage(session.runlog_path)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "csv_id": case.csv_id,
                "diagnosis": case.diagnosis,
                "diagnosis_subtype": diagnosis_subtype(case.diagnosis),
                "gt": gt,
                "pred": pred,
                "runlog_path": str(session.runlog_path),
                "agent_output_path": str(case_out / "agent_output.txt"),
                "correct": correct,
                "scored": scored,
                "nifti_files": [p.name for p in case.nifti_files],
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script="benchmark_UCSF_PDGM.py",
                case_id=case.case_id,
                ok=True,
                prediction=pred,
                ground_truth=gt,
                correct=correct,
                scored=scored,
                elapsed_sec=rec["seconds"],
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

            case_log.info("Completed case %d/%d: %s | pred=%s | gt=%s | correct=%s | tool_calls=%d", idx, len(cases), case.case_id, pred, gt, correct, tool_usage["n_tool_calls"])
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(f"Completed case {idx}/{len(cases)}: {case.case_id} | pred={pred} | gt={gt} | tool_calls={tool_usage['n_tool_calls']}")
        except Exception as e:
            n_failed += 1
            gt = case.label
            scored = gt in {"A", "B", "C"}
            if scored:
                n_scored += 1
            tool_usage = _collect_tool_usage(session.runlog_path)
            total_tool_calls += int(tool_usage["n_tool_calls"])
            _merge_tool_counts(aggregate_tool_call_counts, tool_usage["tool_call_counts"])
            rec = {
                "case_id": case.case_id,
                "csv_id": case.csv_id,
                "diagnosis": case.diagnosis,
                "diagnosis_subtype": diagnosis_subtype(case.diagnosis),
                "gt": gt,
                "pred": None,
                "runlog_path": str(session.runlog_path),
                "correct": False,
                "scored": scored,
                "error": str(e),
                "seconds": round(time.time() - t0, 3),
                **tool_usage,
            }
            session.log_run_event(
                "case_script_finished",
                script="benchmark_UCSF_PDGM.py",
                case_id=case.case_id,
                ok=False,
                error=str(e),
                elapsed_sec=rec["seconds"],
                scored=scored,
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(f"Completed case {idx}/{len(cases)}: {case.case_id} | pred=<error> | gt={gt} | tool_calls={tool_usage['n_tool_calls']}")

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    average_tool_calls_per_case = (total_tool_calls / len(results)) if results else 0.0
    summary = {
        "model": args.model,
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "disturb": bool(args.disturb),
        "n_cases": len(cases),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "n_failed": n_failed,
        "total_tool_calls": total_tool_calls,
        "average_tool_calls_per_case": average_tool_calls_per_case,
        "tool_call_counts": {k: aggregate_tool_call_counts[k] for k in sorted(aggregate_tool_call_counts.keys())},
        "tools_called": sorted(aggregate_tool_call_counts.keys()),
        "n_unique_tools_used": len(aggregate_tool_call_counts),
        "disable_tool_packs": bool(args.disable_tool_packs),
        "balance_diagnosis_subtypes": bool(args.balance_diagnosis_subtypes),
        "timestamp": int(time.time()),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (log_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    log.info("DONE. accuracy=%.4f (%d/%d) failed=%d total_tool_calls=%d avg_tool_calls_per_case=%.2f tools=%s", accuracy, n_correct, n_scored, n_failed, total_tool_calls, average_tool_calls_per_case, sorted(aggregate_tool_call_counts.keys()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
