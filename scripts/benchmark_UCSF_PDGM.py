#!/usr/bin/env python3
"""Benchmark: UCSF-PDGM 3-way classification with optional segment-grounded evidence.

Task (3-way multiple choice):
  A) Glioblastoma
  B) Oligodendroglioma / Astrocytoma
  C) No tumor

Ground truth is read from UCSF-PDGM-metadata_v5.csv.

When a hidden NIfTI tumor segmentation is available for the case, the benchmark also
requires grounded evidence output:
  - KEY_SLICE_AXIAL_MM
  - KEY_SLICE_SAGITTAL_MM
  - KEY_SLICE_CORONAL_MM
  - RSA: [R, A, S]

Evaluation remains deterministic and rule-based (no LLM-as-judge).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from _script_support import (
    build_context_bundle,
    build_model_runtime,
    configure_logging,
    default_model_from_env,
    ensure_repo_imports,
    ensure_slicer_runtime,
    resolve_run_output_dir,
    resolve_bridge_dir,
)

ensure_repo_imports()

from slicer_agent_engine.agents.auto_agent import run_auto_task
from slicer_agent_engine.benchmarking.check_segment import (
    SegmentGroundTruth,
    evaluate_segment_prediction,
    prepare_nifti_segment_ground_truth,
)
from slicer_agent_engine.benchmarking.judge import extract_choice
from slicer_agent_engine.benchmarking.ucsf_pdgm import diagnosis_subtype, iter_cases


EVIDENCE_VIEWS = ["axial", "sagittal", "coronal"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UCSF-PDGM benchmark (NIfTI).")
    p.add_argument("--data_root", type=str, default=".", help="Root directory containing UCSF-PDGM-*_nifti folders")
    p.add_argument("--csv_path", type=str, default="UCSF-PDGM-metadata_v5.csv", help="Metadata CSV path")
    p.add_argument("--provider", type=str, default=os.environ.get("LLM_PROVIDER", "auto"), help="LLM provider name. Use auto to infer from --model (gemini -> Google, claude -> Anthropic, otherwise OpenAI).")
    p.add_argument("--model", type=str, default=default_model_from_env(), help="LLM model name")
    p.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all)")
    p.add_argument("--log_dir", type=str, default="./runs/benchmark_ucsf_pdgm", help="Directory for logs and outputs")
    p.add_argument("--max_rounds", type=int, default=20, help="Max tool-calling rounds")
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
    p.add_argument("--include_segmentations", action="store_true", help="Ignored by this benchmark to avoid leaking hidden GT segmentations into the agent scene.")
    p.add_argument("--no_prefer_bias", action="store_true", help="Do not prefer *_bias.nii.gz when selecting modalities")
    p.add_argument("--enable_gemini", action="store_true", help="Enable Gemini video analysis tools (requires GEMINI_API_KEY)")
    p.add_argument("--force_gemini", action="store_true", help="Force prompt to require both keyframes and Gemini video analysis when using scroll")
    p.add_argument(
        "--disturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add distractor options D/E/F to the multiple-choice prompt.",
    )
    p.add_argument(
        "--disable_tool_packs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable optional tool packs and expose only the baseline/core tools.",
    )
    p.add_argument(
        "--balance_diagnosis_subtypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance UCSF-PDGM sampling across Glioblastoma, Oligodendroglioma, and Astrocytoma as evenly as possible.",
    )
    return p


def build_instructions(
    disturb: bool = False,
    force_gemini: bool = False,
    *,
    enable_code_execution_hint: bool = False,
    disable_tool_packs: bool = False,
) -> str:
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
        "If the task requests KEY_SLICE_*_MM evidence, report canonical RAS-axis mm from the image corner overlay: axial=S, sagittal=R, coronal=A. Do not use slice indices. ",
        "When calling image_coords_to_ras, send exactly one of bbox_1000 or point_1000; never send both in the same call. Use bbox_1000 if you want the box center, otherwise use point_1000 for a single clicked point. ",
        "Your task is a 3-way multiple choice classification based on MRI appearance. ",
        "If you are unsure, still choose the most likely option and explain why. ",
        "Do not output JSON. If the task requests an evidence block, include it exactly once before the final answer line. ",
        f"Output must contain a final line exactly like: 'ANSWER: <{answer_set}>'.",
    ]
    base = "".join(parts)
    if enable_code_execution_hint:
        base += " A provider-native code execution tool is enabled. You may use it for multi-pass visual inspection or precise 0..999 image-space localization when the task benefits from that level of visual reasoning."
    return base


def _format_segment_requirement(segment_evidence: Dict[str, Any]) -> str:
    source_name = str(segment_evidence.get("source_volume_name") or "").strip()
    source_line = ""
    if source_name:
        source_line = f"A geometry-matched loaded reference volume is '{source_name}'. You may use that as a convenient anchor, but the benchmark accepts any loaded anatomical MRI sequence in the shared co-registered space.\n"
    return (
        "Additional evidence output is required for this case.\n"
        "The benchmark has a hidden tumor segmentation in a shared co-registered MRI space. Unlike DICOM-SEG-on-a-single-series workflows, this NIfTI mask is judged in the shared RAS space, so you may localize the tumor on any loaded anatomical MRI sequence where it is clearly visible. T1c is often useful for enhancing tumor; FLAIR/T2 are often useful for non-enhancing abnormality.\n"
        + source_line
        + "If there are multiple disconnected tumor foci, localize the largest connected tumor component.\n"
        + "After your brief rationale, output the following lines exactly once before the final answer line:\n"
        + "EVIDENCE:\n"
        + "KEY_SLICE_AXIAL_MM: <float>\n"
        + "KEY_SLICE_SAGITTAL_MM: <float>\n"
        + "KEY_SLICE_CORONAL_MM: <float>\n"
        + "RSA: [<R>, <A>, <S>]\n"
        + "Definitions:\n"
        + "- Each KEY_SLICE_*_MM value must use canonical RAS-axis mm from the image corner overlay: axial=S, sagittal=R, coronal=A.\n"
        + "- RSA must be a single point strictly inside the visible tumor mask, not on background and not outside the lesion extent.\n"
        + "- Use one loaded MRI sequence consistently while localizing the tumor and producing the evidence block.\n"
        + "- If you use image_coords_to_ras, pass exactly one of bbox_1000 or point_1000. Never send both in the same call.\n"
    )


def build_user_message(
    case_id: str,
    case_path: Path,
    disturb: bool = False,
    *,
    segment_evidence: Optional[Dict[str, Any]] = None,
) -> str:
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

    evidence_block = ""
    if segment_evidence and bool(segment_evidence.get("has_segment")):
        evidence_block = _format_segment_requirement(segment_evidence) + "\n"

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
        + evidence_block
        + f"Write a brief rationale, then end with a single final line: ANSWER: <{answer_set}>."
        + " Finally, leave the viewer on the slice where the lesion is most obvious.\n\n"
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


def _safe_prepare_segment_gt(
    *,
    base_url: str,
    bridge_dir: Path,
    out_dir: Path,
    case_path: Path,
    segment_path: Optional[Path],
    reference_volume_paths: Optional[Sequence[Path]],
) -> Tuple[Optional[SegmentGroundTruth], Optional[str]]:
    if segment_path is None:
        return None, None
    try:
        hidden_bundle = build_context_bundle(
            base_url=base_url,
            out_dir=out_dir,
            bridge_dir=bridge_dir,
            enable_gemini_video=False,
        )
        gt = prepare_nifti_segment_ground_truth(
            hidden_bundle.ctx,
            case_path=case_path,
            out_dir=out_dir,
            segment_path=segment_path,
            reference_volume_paths=reference_volume_paths,
            source_modality="MR",
            provenance_policy="not_applicable",
            clear_scene_after=True,
        )
        return gt, None
    except Exception as e:
        return None, str(e)


def _segment_gt_summary(gt: Optional[SegmentGroundTruth]) -> Optional[Dict[str, Any]]:
    if gt is None:
        return None
    return {
        "has_segment": bool(gt.has_segment),
        "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
        "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
        "segmentation_name": gt.segmentation_name,
        "segment_name": gt.segment_name,
        "source_volume_name": gt.source_volume_name,
        "source_modality": gt.source_modality,
        "provenance_policy": gt.provenance_policy,
        "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in EVIDENCE_VIEWS},
        "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in EVIDENCE_VIEWS},
        "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
        "voxel_count": gt.voxel_count,
        "segment_total_voxel_count": gt.segment_total_voxel_count,
        "component_count": gt.component_count,
        "notes": list(gt.notes),
    }


def _emit_segment_debug(case_log: logging.Logger, *, case_id: str, segment_check: Dict[str, Any]) -> None:
    if not isinstance(segment_check, dict):
        return
    for line in segment_check.get("debug_lines") or []:
        case_log.info("[segment-check][%s] %s", case_id, line)


def _update_segment_metrics(
    *,
    gt: Optional[SegmentGroundTruth],
    segment_check: Dict[str, Any],
    key_slice_scored: Dict[str, int],
    key_slice_correct: Dict[str, int],
    rsa_metrics: Dict[str, int],
    provenance_metrics: Dict[str, int],
    evidence_case_metrics: Dict[str, int],
) -> None:
    if gt is None or not gt.has_segment:
        return
    evidence_case_metrics["n_cases_with_segment_gt"] += 1
    if bool(segment_check.get("geometry_exact_correct", False)):
        evidence_case_metrics["n_cases_with_geometry_exact_correct"] += 1
    if bool(segment_check.get("evidence_exact_correct", False)):
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] += 1

    key_slice_payload = segment_check.get("key_slice") or {}
    for view in EVIDENCE_VIEWS:
        key_slice_scored[view] += 1
        if bool((key_slice_payload.get(view) or {}).get("correct", False)):
            key_slice_correct[view] += 1

    rsa_metrics["n_scored"] += 1
    if bool((segment_check.get("rsa") or {}).get("correct", False)):
        rsa_metrics["n_correct"] += 1

    provenance_payload = segment_check.get("provenance") or {}
    if bool(provenance_payload.get("scored", False)):
        provenance_metrics["n_scored"] += 1
        if bool(provenance_payload.get("ok", False)):
            provenance_metrics["n_correct"] += 1


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    os.environ["SLICER_CASE_ROOT"] = str(data_root)
    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.is_absolute():
        csv_path = (data_root / csv_path).resolve()

    log_dir = resolve_run_output_dir(args.log_dir, model=args.model)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir, log_name="benchmark.log")
    log = logging.getLogger("benchmark_ucsf_pdgm")

    bridge_dir = resolve_bridge_dir()
    os.environ["SLICER_EXECUTABLE"] = str(Path(args.slicer_executable).expanduser())
    ensure_slicer_runtime(
        base_url=args.base_url,
        bridge_dir=bridge_dir,
        slicer_executable=Path(args.slicer_executable).expanduser(),
        require_exec=True,
        require_slice_render=True,
    )

    model_runtime = build_model_runtime(
        provider=args.provider,
        model=args.model,
        enable_code_interpreter=bool(args.enable_code_interpreter),
    )
    resolved_provider = model_runtime.resolved_provider
    model_client = model_runtime.model_client
    extra_model_tools = model_runtime.extra_model_tools

    if args.include_segmentations:
        log.warning("Ignoring --include_segmentations to avoid leaking hidden GT segmentations into the agent scene.")

    limit = args.limit if args.limit and args.limit > 0 else None
    cases = list(
        iter_cases(
            data_root,
            csv_path,
            limit=limit,
            include_extra=bool(args.include_extra),
            include_segmentations=False,
            prefer_bias=not bool(args.no_prefer_bias),
            balance_diagnosis_subtypes=bool(args.balance_diagnosis_subtypes),
        )
    )
    log.info("Found %d cases under %s", len(cases), data_root)

    instructions = build_instructions(
        bool(args.disturb),
        bool(args.force_gemini),
        enable_code_execution_hint=bool(extra_model_tools),
        disable_tool_packs=bool(args.disable_tool_packs),
    )

    results: List[Dict[str, Any]] = []
    n_scored = 0
    n_correct = 0
    n_correct_with_evidence = 0
    n_localization_correct = 0
    n_failed = 0
    aggregate_tool_call_counts: Dict[str, int] = {}
    total_tool_calls = 0

    key_slice_scored = {view: 0 for view in EVIDENCE_VIEWS}
    key_slice_correct = {view: 0 for view in EVIDENCE_VIEWS}
    rsa_metrics = {"n_scored": 0, "n_correct": 0}
    provenance_metrics = {"n_scored": 0, "n_correct": 0}
    evidence_case_metrics = {
        "n_cases_with_segment_gt": 0,
        "n_cases_with_geometry_exact_correct": 0,
        "n_cases_with_evidence_exact_correct": 0,
    }

    for idx, case in enumerate(cases, start=1):
        t0 = time.time()
        case_out = log_dir / case.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        case_log = logging.getLogger(f"case.{case.case_id}")
        _print_progress(f"Processing case {idx}/{len(cases)}: {case.case_id}")
        case_log.info("Processing case %d/%d: %s", idx, len(cases), case.case_id)

        segment_gt_dir = case_out / "segment_gt"
        segment_gt, segment_gt_error = _safe_prepare_segment_gt(
            base_url=args.base_url,
            bridge_dir=bridge_dir,
            out_dir=segment_gt_dir,
            case_path=case.case_dir,
            segment_path=case.segmentation_file,
            reference_volume_paths=case.nifti_files,
        )
        if segment_gt_error:
            case_log.warning("Segment GT discovery failed for %s: %s", case.case_id, segment_gt_error)

        context_bundle = build_context_bundle(
            base_url=args.base_url,
            out_dir=case_out,
            bridge_dir=bridge_dir,
            enable_gemini_video=bool(args.enable_gemini),
        )
        session = context_bundle.session
        session.log_run_event("case_script_start", script="benchmark_UCSF_PDGM.py", case_id=case.case_id, model=args.model, max_rounds=int(args.max_rounds))
        ctx = context_bundle.ctx

        if segment_gt is not None and segment_gt.gt_json_path and segment_gt.gt_json_path.exists():
            try:
                payload = json.loads(segment_gt.gt_json_path.read_text(encoding="utf-8"))
                (case_out / "segment_gt_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass

        user_msg = build_user_message(
            case.case_id,
            case.case_dir,
            bool(args.disturb),
            segment_evidence=segment_gt.to_prompt_spec() if segment_gt is not None and segment_gt.has_segment else None,
        )

        open_case_kwargs: Dict[str, Any] = {
            "preset": "mri_brain",
            "layout": "four_up",
            "clear_scene_first": True,
            "nifti_files": [str(p) for p in case.nifti_files],
            "include_segmentations": False,
        }
        active_prefer: List[str] = []
        if segment_gt is not None and segment_gt.has_segment and segment_gt.source_volume_name:
            active_prefer.append(str(segment_gt.source_volume_name))
        if case.nifti_files:
            active_prefer.append(case.nifti_files[0].stem)
            active_prefer.append(case.nifti_files[0].name)
        if active_prefer:
            open_case_kwargs["active_prefer"] = active_prefer

        core_sequences = ", ".join([p.name for p in case.nifti_files]) if case.nifti_files else "none"
        evidence_note = ""
        if segment_gt is not None and segment_gt.has_segment:
            if segment_gt.source_volume_name:
                evidence_note = (
                    f"- Hidden tumor mask is evaluated in a shared co-registered MRI space. A geometry-matched loaded reference volume is {segment_gt.source_volume_name}.\n"
                    "- Evidence may be measured on any loaded anatomical MRI sequence where the tumor is clearly visible; provenance to one specific MRI series is not enforced for this dataset.\n"
                )
            else:
                evidence_note = (
                    "- Hidden tumor mask is evaluated in a shared co-registered MRI space.\n"
                    "- Evidence may be measured on any loaded anatomical MRI sequence where the tumor is clearly visible; provenance to one specific MRI series is not enforced for this dataset.\n"
                )

        case_brief = (
            f"Case brief for {case.case_id}:\n"
            f"- CSV ID: {case.csv_id}\n"
            f"- Loaded core NIfTI files: {core_sequences}\n"
            + evidence_note
            + "- The current scene inventory will also be shown after open_case so you do not need a separate listing tool.\n"
            + "- Treat the loaded T1/T1c/T2/FLAIR-style sequences as your main evidence sources for this benchmark."
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

            segment_check = evaluate_segment_prediction(prediction=raw_text, gt=segment_gt, trace_path=session.trace_path) if segment_gt is not None else {
                "available": False,
                "has_segment_gt": False,
                "segment_gt_error": segment_gt_error,
            }
            _emit_segment_debug(case_log, case_id=case.case_id, segment_check=segment_check)
            _update_segment_metrics(
                gt=segment_gt,
                segment_check=segment_check,
                key_slice_scored=key_slice_scored,
                key_slice_correct=key_slice_correct,
                rsa_metrics=rsa_metrics,
                provenance_metrics=provenance_metrics,
                evidence_case_metrics=evidence_case_metrics,
            )
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_payload = segment_check.get("key_slice") or {}
            key_slice_all_correct = (
                bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))
                if has_segment_gt
                else None
            )
            localization_correct = (
                bool((segment_check.get("rsa") or {}).get("correct", False))
                if has_segment_gt
                else None
            )
            correct_with_evidence = (
                bool(correct and key_slice_all_correct)
                if has_segment_gt
                else None
            )
            if bool(correct_with_evidence):
                n_correct_with_evidence += 1
            if bool(localization_correct):
                n_localization_correct += 1

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
                "trace_path": str(session.trace_path),
                "agent_output_path": str(case_out / "agent_output.txt"),
                "correct": correct,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "scored": scored,
                "nifti_files": [p.name for p in case.nifti_files],
                "segmentation_file": str(case.segmentation_file) if case.segmentation_file else None,
                "has_segment_gt": has_segment_gt,
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
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
                correct_with_evidence=correct_with_evidence,
                key_slice_all_correct=key_slice_all_correct,
                localization_correct=localization_correct,
                scored=scored,
                has_segment_gt=has_segment_gt,
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                elapsed_sec=rec["seconds"],
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

            case_log.info(
                "Completed case %d/%d: %s | pred=%s | gt=%s | correct=%s | correct_with_evidence=%s | localization_correct=%s | tool_calls=%d",
                idx,
                len(cases),
                case.case_id,
                pred,
                gt,
                correct,
                correct_with_evidence,
                localization_correct,
                tool_usage["n_tool_calls"],
            )
            case_log.info("Model answer for %s:\n%s", case.case_id, raw_text)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | pred={pred} | gt={gt} | "
                f"evidence_correct={correct_with_evidence} | localization_correct={localization_correct} | "
                f"tool_calls={tool_usage['n_tool_calls']}"
            )
        except Exception as e:
            n_failed += 1
            gt = case.label
            scored = gt in {"A", "B", "C"}
            if scored:
                n_scored += 1

            segment_check = evaluate_segment_prediction(prediction="", gt=segment_gt, trace_path=session.trace_path) if segment_gt is not None else {
                "available": False,
                "has_segment_gt": False,
                "segment_gt_error": segment_gt_error,
            }
            _emit_segment_debug(case_log, case_id=case.case_id, segment_check=segment_check)
            _update_segment_metrics(
                gt=segment_gt,
                segment_check=segment_check,
                key_slice_scored=key_slice_scored,
                key_slice_correct=key_slice_correct,
                rsa_metrics=rsa_metrics,
                provenance_metrics=provenance_metrics,
                evidence_case_metrics=evidence_case_metrics,
            )
            has_segment_gt = bool(segment_gt is not None and segment_gt.has_segment)
            key_slice_payload = segment_check.get("key_slice") or {}
            key_slice_all_correct = (
                bool(all(bool((key_slice_payload.get(view) or {}).get("correct", False)) for view in EVIDENCE_VIEWS))
                if has_segment_gt
                else None
            )
            localization_correct = (
                bool((segment_check.get("rsa") or {}).get("correct", False))
                if has_segment_gt
                else None
            )
            correct_with_evidence = False if has_segment_gt else None
            if bool(localization_correct):
                n_localization_correct += 1

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
                "trace_path": str(session.trace_path),
                "correct": False,
                "correct_with_evidence": correct_with_evidence,
                "key_slice_all_correct": key_slice_all_correct,
                "localization_correct": localization_correct,
                "scored": scored,
                "nifti_files": [p.name for p in case.nifti_files],
                "segmentation_file": str(case.segmentation_file) if case.segmentation_file else None,
                "has_segment_gt": has_segment_gt,
                "segment_gt_error": segment_gt_error,
                "segment_gt": _segment_gt_summary(segment_gt),
                "segment_check": segment_check,
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
                has_segment_gt=has_segment_gt,
                correct_with_evidence=correct_with_evidence,
                key_slice_all_correct=key_slice_all_correct,
                localization_correct=localization_correct,
                evidence_exact_correct=segment_check.get("evidence_exact_correct") if isinstance(segment_check, dict) else None,
                n_tool_calls=tool_usage["n_tool_calls"],
                tools_called=tool_usage["tools_called"],
                tool_call_counts=tool_usage["tool_call_counts"],
            )
            results.append(rec)
            (case_out / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            case_log.exception("Case failed: %s", e)
            _print_progress(
                f"Completed case {idx}/{len(cases)}: {case.case_id} | pred=<error> | gt={gt} | "
                f"evidence_correct={correct_with_evidence} | localization_correct={localization_correct} | "
                f"tool_calls={tool_usage['n_tool_calls']}"
            )

    accuracy = (n_correct / n_scored) if n_scored else 0.0
    accuracy_with_evidence = (
        n_correct_with_evidence / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
    average_tool_calls_per_case = (total_tool_calls / len(results)) if results else 0.0

    key_slice_metrics = {
        view: {
            "n_scored": key_slice_scored[view],
            "n_correct": key_slice_correct[view],
            "accuracy": (key_slice_correct[view] / key_slice_scored[view]) if key_slice_scored[view] else 0.0,
        }
        for view in EVIDENCE_VIEWS
    }
    n_key_slice_items_scored = sum(key_slice_scored.values())
    n_key_slice_items_correct = sum(key_slice_correct.values())
    avg_evidence_correct = (n_key_slice_items_correct / n_key_slice_items_scored) if n_key_slice_items_scored else 0.0
    localization_acc = (rsa_metrics["n_correct"] / rsa_metrics["n_scored"]) if rsa_metrics["n_scored"] else 0.0
    n_evidence_items_scored = sum(key_slice_scored.values()) + rsa_metrics["n_scored"]
    n_evidence_items_correct = sum(key_slice_correct.values()) + rsa_metrics["n_correct"]
    evidence_item_accuracy = (n_evidence_items_correct / n_evidence_items_scored) if n_evidence_items_scored else 0.0
    geometry_case_accuracy = (
        evidence_case_metrics["n_cases_with_geometry_exact_correct"] / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
    evidence_case_accuracy = (
        evidence_case_metrics["n_cases_with_evidence_exact_correct"] / evidence_case_metrics["n_cases_with_segment_gt"]
        if evidence_case_metrics["n_cases_with_segment_gt"]
        else 0.0
    )
    provenance_accuracy = (
        provenance_metrics["n_correct"] / provenance_metrics["n_scored"]
        if provenance_metrics["n_scored"]
        else 0.0
    )

    summary = {
        "model": args.model,
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "disturb": bool(args.disturb),
        "n_cases": len(cases),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "avg_evidence_correct": avg_evidence_correct,
        "Localization_acc": localization_acc,
        "n_localization_correct": n_localization_correct,
        "n_correct_with_evidence": n_correct_with_evidence,
        "accuracy_with_evidence": accuracy_with_evidence,
        "segment_metrics": {
            "n_cases_with_segment_gt": evidence_case_metrics["n_cases_with_segment_gt"],
            "n_cases_with_geometry_exact_correct": evidence_case_metrics["n_cases_with_geometry_exact_correct"],
            "geometry_case_accuracy": geometry_case_accuracy,
            "n_cases_with_evidence_exact_correct": evidence_case_metrics["n_cases_with_evidence_exact_correct"],
            "evidence_case_accuracy": evidence_case_accuracy,
            "n_evidence_items_scored": n_evidence_items_scored,
            "n_evidence_items_correct": n_evidence_items_correct,
            "evidence_item_accuracy": evidence_item_accuracy,
            "key_slice_metrics": key_slice_metrics,
            "rsa_metrics": {
                "n_scored": rsa_metrics["n_scored"],
                "n_correct": rsa_metrics["n_correct"],
                "accuracy": localization_acc,
            },
            "provenance_metrics": {
                "n_scored": provenance_metrics["n_scored"],
                "n_correct": provenance_metrics["n_correct"],
                "accuracy": provenance_accuracy,
            },
        },
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

    log.info(
        "DONE. accuracy=%.4f (%d/%d) accuracy_with_evidence=%.4f (%d/%d segment-gt) localization_acc=%.4f (%d/%d segment-gt) failed=%d total_tool_calls=%d avg_tool_calls_per_case=%.2f tools=%s",
        accuracy,
        n_correct,
        n_scored,
        accuracy_with_evidence,
        n_correct_with_evidence,
        evidence_case_metrics["n_cases_with_segment_gt"],
        localization_acc,
        n_localization_correct,
        rsa_metrics["n_scored"],
        n_failed,
        total_tool_calls,
        average_tool_calls_per_case,
        sorted(aggregate_tool_call_counts.keys()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
