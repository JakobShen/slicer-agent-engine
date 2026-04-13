from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from _script_support import ensure_repo_imports

ensure_repo_imports()

from slicer_agent_engine.benchmarking.check_segment import (
    build_default_negative_prediction,
    build_default_positive_prediction,
    evaluate_segment_prediction,
    prepare_nifti_segment_ground_truth,
    prepare_segment_ground_truth,
)
from slicer_agent_engine.load_helper import build_nsclc_case_catalog
from slicer_agent_engine.tools import ToolContext
from _script_support import bootstrap_runtime


_SKIP_ERROR_HINTS = (
    "segmenteditorextraeffects",
    "local threshold",
    "install segmenteditorextraeffects",
    "quantitative reporting",
    "dicomsegmentationplugin",
)


def _skip_due_to_error(error: Any) -> bool:
    text = str(error or "").lower()
    return any(hint in text for hint in _SKIP_ERROR_HINTS)


class StepRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(self, name: str, fn: Callable[[], Any], *, required: bool = True) -> Any:
        try:
            res = fn()
        except Exception as e:
            logging.exception("Step failed: %s", name)
            res = {"ok": False, "error": str(e)}
        status = "passed"
        reason: Optional[str] = None
        if isinstance(res, dict) and res.get("skipped"):
            status = "skipped"
            reason = str(res.get("reason") or res.get("error") or "")
        elif isinstance(res, dict) and res.get("ok") is False:
            reason = str(res.get("error") or res)
            if (not required) or _skip_due_to_error(reason):
                status = "skipped"
                res = {**res, "skipped": True, "reason": reason}
            else:
                status = "failed"
        print(f"\n[{name}]\n{res}")
        self.records.append({"name": name, "status": status, "reason": reason, "required": required})
        return res

    def skip(self, name: str, reason: str) -> dict[str, Any]:
        res = {"ok": False, "skipped": True, "reason": reason}
        print(f"\n[{name}]\n{res}")
        self.records.append({"name": name, "status": "skipped", "reason": reason, "required": False})
        return res

    def summary(self) -> Dict[str, Any]:
        total = len(self.records)
        passed = sum(1 for r in self.records if r["status"] == "passed")
        failed = [r for r in self.records if r["status"] == "failed"]
        skipped = [r for r in self.records if r["status"] == "skipped"]
        return {"total": total, "passed": passed, "failed": failed, "skipped": skipped}

    def print_summary(self) -> Dict[str, Any]:
        s = self.summary()
        print("\n===== MANUAL TEST SUMMARY =====")
        print(f"total={s['total']} passed={s['passed']} failed={len(s['failed'])} skipped={len(s['skipped'])}")
        if s["failed"]:
            print("FAILED:")
            for r in s["failed"]:
                print(f"  - {r['name']}: {r.get('reason') or ''}")
        if s["skipped"]:
            print("SKIPPED:")
            for r in s["skipped"]:
                print(f"  - {r['name']}: {r.get('reason') or ''}")
        return s


def _pick_existing(*cands: str) -> Optional[Path]:
    for c in cands:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.exists():
            return p.resolve()
    return None


def _best_volume_name(ctx: ToolContext, volumes: Any, query: str) -> Optional[str]:
    if not isinstance(volumes, list):
        return None
    ranked: list[tuple[float, str]] = []
    for item in volumes:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        ranked.append((ctx._volume_match_score(query=query, candidate_name=name), name))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if ranked and ranked[0][0] > 250.0:
        return ranked[0][1]
    return None


def _move_axial_to_mid(ctx: ToolContext) -> Any:
    rng = ctx.get_slice_offset_range(view="axial")
    lo = rng.get("min_offset") if isinstance(rng, dict) else None
    hi = rng.get("max_offset") if isinstance(rng, dict) else None
    if lo is None or hi is None:
        return {"ok": False, "range": rng, "error": "min/max offset unavailable"}
    mid = (float(lo) + float(hi)) / 2.0
    set_res = ctx.set_slice_offset(view="axial", offset=mid)
    obs = ctx.observe_view(view="axial", size=512, out_name="axial_mid_offset.png")
    return {"ok": True, "range": rng, "mid_offset": mid, "set": set_res, "observation": obs, "png_path": obs.get("png_path")}


def _segmentation_smoke(ctx: ToolContext, *, out_dir: Path, prefix: str = "segdemo") -> Any:
    bone_name = f"{prefix}_bone"
    local_name = f"{prefix}_local"
    result: dict[str, Any] = {"ok": True, "failed_substeps": [], "skipped_substeps": []}

    def _sub(name: str, fn: Callable[[], Any], *, optional: bool = False) -> Any:
        try:
            sub = fn()
        except Exception as e:
            sub = {"ok": False, "error": str(e)}
        if isinstance(sub, dict) and sub.get("ok") is False:
            err = str(sub.get("error") or sub)
            if optional or _skip_due_to_error(err):
                sub = {**sub, "skipped": True, "reason": err}
                result["skipped_substeps"].append({"name": name, "reason": err})
            else:
                result["ok"] = False
                result["failed_substeps"].append({"name": name, "reason": err})
        result[name] = sub
        return sub

    _sub("list_segmentations", lambda: ctx.list_segmentations())
    _sub("threshold", lambda: ctx.segment_from_threshold(segment_name=bone_name, minimum_threshold=200.0, maximum_threshold=3000.0))
    _sub("keep_largest", lambda: ctx.segment_islands(segment_name=bone_name, operation="KEEP_LARGEST_ISLAND", minimum_size=5000))
    _sub("smoothing", lambda: ctx.segment_smoothing(segment_name=bone_name, smoothing_method="MEDIAN", kernel_size_mm=2.0))
    stats = _sub("stats", lambda: ctx.segment_statistics(segment_name=bone_name))
    if isinstance(stats, dict) and stats.get("center_ras"):
        _sub("sphere_edit", lambda: ctx.segment_edit_sphere(segment_name=bone_name, ras_center=stats["center_ras"], radius_mm=5.0, action="add"))
    _sub("center", lambda: ctx.center_on_segment(segment_name=bone_name))
    _sub("recover", lambda: ctx.recover_standard_views())
    _sub(
        "local_threshold_bbox_center_seed",
        lambda: ctx.segment_local_threshold(
            segment_name=local_name,
            minimum_threshold=200.0,
            maximum_threshold=3000.0,
            segmentation_algorithm="GrowCut",
            seed_view="axial",
            seed_bbox_1000=[440, 440, 560, 560],
        ),
    )
    seg_id = ctx.session.state.get("active_segmentation_id")
    ref_vol_id = ctx.session.state.get("active_volume_id")
    if seg_id and ref_vol_id:
        export_dir = out_dir / f"{prefix}_dicom_seg"
        export_dir.mkdir(parents=True, exist_ok=True)
        _sub(
            "dicom_export",
            lambda: ctx.export_segmentation_dicom(
                segmentation_id=str(seg_id),
                reference_volume_id=str(ref_vol_id),
                output_folder=export_dir,
            ),
            optional=True,
        )
    return result




def _segment_check_smoke(ctx: ToolContext, *, case_dir: Path, out_dir: Path) -> Any:
    gt_out_dir = out_dir / "segment_check_gt"
    gt = prepare_segment_ground_truth(ctx, case_path=case_dir, out_dir=gt_out_dir, clear_scene_after=True)
    if not gt.has_segment:
        return {
            "ok": False,
            "error": f"No non-empty segmentation found under {case_dir}",
            "gt": {
                "has_segment": False,
                "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
            },
        }

    positive = build_default_positive_prediction(gt)
    negative = build_default_negative_prediction(gt)
    positive_eval = evaluate_segment_prediction(prediction=positive, gt=gt, trace_path=None)
    negative_eval = evaluate_segment_prediction(prediction=negative, gt=gt, trace_path=None)

    negative_key_slice_all_wrong = all(not bool((negative_eval.get("key_slice") or {}).get(view, {}).get("correct", False)) for view in ("axial", "sagittal", "coronal"))
    ok = bool(positive_eval.get("geometry_exact_correct", False) and not bool(negative_eval.get("geometry_exact_correct", False)) and negative_key_slice_all_wrong and not bool((negative_eval.get("rsa") or {}).get("correct", False)))
    return {
        "ok": ok,
        "case_dir": str(case_dir),
        "gt": {
            "has_segment": True,
            "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
            "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
            "segmentation_name": gt.segmentation_name,
            "segment_name": gt.segment_name,
            "source_volume_name": gt.source_volume_name,
            "source_modality": gt.source_modality,
            "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in ("axial", "sagittal", "coronal")},
            "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in ("axial", "sagittal", "coronal")},
            "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in ("axial", "sagittal", "coronal")},
            "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
            "voxel_count": gt.voxel_count,
            "segment_total_voxel_count": gt.segment_total_voxel_count,
            "component_count": gt.component_count,
            "notes": list(gt.notes),
        },
        "positive_prediction": {
            "key_slice_mm": positive.key_slice_mm,
            "point_ras": list(positive.point_ras) if positive.point_ras is not None else None,
        },
        "positive_eval": positive_eval,
        "negative_prediction": {
            "key_slice_mm": negative.key_slice_mm,
            "point_ras": list(negative.point_ras) if negative.point_ras is not None else None,
        },
        "negative_eval": negative_eval,
        "note": "geometry_exact_correct is the relevant assertion here; provenance is intentionally not scored in this standalone smoke test.",
    }


def _nifti_segment_check_smoke(ctx: ToolContext, *, case_dir: Path, out_dir: Path) -> Any:
    gt_out_dir = out_dir / "nifti_segment_check_gt"
    gt = prepare_nifti_segment_ground_truth(
        ctx,
        case_path=case_dir,
        out_dir=gt_out_dir,
        source_modality="MR",
        provenance_policy="not_applicable",
        clear_scene_after=True,
    )
    if not gt.has_segment:
        return {
            "ok": False,
            "error": f"No non-empty NIfTI segmentation found under {case_dir}",
            "gt": {
                "has_segment": False,
                "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
                "notes": list(gt.notes),
            },
        }

    positive = build_default_positive_prediction(gt)
    negative = build_default_negative_prediction(gt)
    positive_eval = evaluate_segment_prediction(prediction=positive, gt=gt, trace_path=None)
    negative_eval = evaluate_segment_prediction(prediction=negative, gt=gt, trace_path=None)

    negative_key_slice_all_wrong = all(
        not bool((negative_eval.get("key_slice") or {}).get(view, {}).get("correct", False))
        for view in ("axial", "sagittal", "coronal")
    )
    ok = bool(
        positive_eval.get("geometry_exact_correct", False)
        and not bool(negative_eval.get("geometry_exact_correct", False))
        and negative_key_slice_all_wrong
        and not bool((negative_eval.get("rsa") or {}).get("correct", False))
    )
    return {
        "ok": ok,
        "case_dir": str(case_dir),
        "gt": {
            "has_segment": True,
            "gt_json_path": str(gt.gt_json_path) if gt.gt_json_path else None,
            "mask_npz_path": str(gt.mask_npz_path) if gt.mask_npz_path else None,
            "segmentation_name": gt.segmentation_name,
            "segment_name": gt.segment_name,
            "source_volume_name": gt.source_volume_name,
            "source_modality": gt.source_modality,
            "provenance_policy": gt.provenance_policy,
            "key_slice_ranges_mm": {view: list(gt.key_slice_ranges_mm.get(view) or []) for view in ("axial", "sagittal", "coronal")},
            "key_slice_axis_labels": {view: gt.key_slice_axis_labels.get(view) for view in ("axial", "sagittal", "coronal")},
            "ras_axis_ranges_mm": {view: list(gt.ras_axis_ranges_mm.get(view) or []) for view in ("axial", "sagittal", "coronal")},
            "representative_point_ras": list(gt.representative_point_ras) if gt.representative_point_ras is not None else None,
            "voxel_count": gt.voxel_count,
            "segment_total_voxel_count": gt.segment_total_voxel_count,
            "component_count": gt.component_count,
            "notes": list(gt.notes),
        },
        "positive_prediction": {
            "key_slice_mm": positive.key_slice_mm,
            "point_ras": list(positive.point_ras) if positive.point_ras is not None else None,
        },
        "positive_eval": positive_eval,
        "negative_prediction": {
            "key_slice_mm": negative.key_slice_mm,
            "point_ras": list(negative.point_ras) if negative.point_ras is not None else None,
        },
        "negative_eval": negative_eval,
        "note": "geometry_exact_correct is the relevant assertion here; source-series provenance is intentionally not scored for UCSF-PDGM because the NIfTI MRI volumes share a co-registered space.",
    }


def main() -> None:
    recorder = StepRecorder()
    base_url = os.environ.get("SLICER_BASE_URL", "http://localhost:2016")
    dicom_dir = Path(os.environ.get("DICOM_DIR", "/Users/weixiangshen/downloads/dicom/MRHead_DICOM"))
    out_dir = Path(os.environ.get("OUT_DIR", "/Users/weixiangshen/downloads/dicom/test_tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = bootstrap_runtime(
        out_dir=out_dir,
        log_name="manual_test.log",
        base_url=base_url,
        provider="openai",
        model=os.environ.get("MODEL", "gpt-5"),
        enable_gemini_video=bool(os.environ.get("GEMINI_API_KEY")),
        ensure_slicer_ready=True,
        require_exec=True,
        require_slice_render=True,
    )
    recorder.run("ensure_webserver", lambda: {"ok": True, "base_url": base_url})

    session = runtime.context.session
    session.log_run_event("script_start", script="manual_test.py", out_dir=str(out_dir), base_url=base_url)
    ctx = runtime.context.ctx

    recorder.run("ping", ctx.ping)
    recorder.run("open_case", lambda: ctx.open_case(dicom_dir, baseline=True, size=512))
    recorder.run("resample_isotropic", lambda: ctx.resample_volume_isotropic(output_spacing_mm=1.0, output_name="smoke_iso_1mm", make_active=False), required=False)
    vols_initial = recorder.run("list_volumes", lambda: ctx.list_volumes())
    recorder.run("observe_standard_views", lambda: ctx.observe_standard_views(size=512, out_prefix="manual_std"))
    recorder.run("observe_view_axial", lambda: ctx.observe_view(view="axial", size=512, out_name="manual_obs_axial.png"))
    recorder.run("wl_auto", lambda: ctx.set_window_level(auto=True))
    recorder.run("zoom", lambda: ctx.zoom_slice_relative(view="axial", factor=1.8))
    recorder.run("observe_after_zoom", lambda: ctx.observe_view(view="axial", size=512, out_name="manual_after_zoom.png"))
    recorder.run("scroll_keyframes", lambda: ctx.scroll_sweep(view="axial", orientation="axial", start=0.0, end=1.0, n_frames=12, size=384, output="keyframes", out_prefix="manual_scroll"))
    recorder.run("scroll_video", lambda: ctx.scroll_sweep(view="axial", orientation="axial", start=0.0, end=1.0, n_frames=60, size=384, output="video", fps=12, analyze=False, out_prefix="manual_cine"), required=False)
    recorder.run("scroll_video_analyze", lambda: ctx.scroll_sweep(view="axial", orientation="axial", start=0.0, end=1.0, n_frames=60, size=384, output="video", fps=12, analyze=True, out_prefix="manual_cine_analyze"), required=False)
    recorder.run("thick_slab_mip", lambda: ctx.set_thick_slab(view="axial", enabled=True, thickness_mm=12.0, mode="mip"))
    recorder.run("observe_after_slab", lambda: ctx.observe_view(view="axial", size=512, out_name="manual_after_slab.png"))
    recorder.run("axial_offset_mid_recovery", lambda: _move_axial_to_mid(ctx))
    recorder.run("recover_standard_views", lambda: ctx.recover_standard_views())

    nifti_case_dir = _pick_existing(
        os.environ.get("NIFTI_CASE_DIR", ""),
        "/Users/weixiangshen/UCSF-PDGM/UCSF-PDGM-0004_nifti",
        "/Users/weixiangshen/Downloads/UCSF-PDGM/UCSF-PDGM-0004_nifti",
        "/Users/weixiangshen/downloads/dicom/UCSF-PDGM/UCSF-PDGM-0004_nifti",
        "./UCSF-PDGM-0004_nifti",
        "../UCSF-PDGM-0004_nifti",
    )
    if nifti_case_dir is None:
        recorder.skip("nifti_smoke", "NIfTI case dir not found")
    else:
        recorder.run("open_case_nifti", lambda: ctx.open_case(nifti_case_dir, baseline=True, size=512, clear_scene_first=True))
        vols2 = recorder.run("list_volumes_nifti", lambda: ctx.list_volumes())
        if isinstance(vols2, list):
            for w in ["T1c", "FLAIR", "T2", "T1"]:
                match = next((v.get("name") for v in vols2 if isinstance(v, dict) and w.lower() in str(v.get("name", "")).lower()), None)
                if not match:
                    continue
                recorder.run(f"select_volume_{w}", lambda m=match: ctx.select_volume(volume_name=m))
                recorder.run(f"observe_{w}", lambda w=w: ctx.observe_standard_views(size=512, out_prefix=f"nifti_{w.lower()}"))
        recorder.run("ucsf_pdgm_segment_check", lambda: _nifti_segment_check_smoke(ctx, case_dir=Path(nifti_case_dir), out_dir=out_dir))

    nsclc_root = _pick_existing(os.environ.get("NSCLC_ROOT", ""), "/Users/weixiangshen/downloads/dicom/manifest-1772811064422")
    nsclc_meta = _pick_existing(os.environ.get("NSCLC_METADATA_CSV", ""), str(Path("/Users/weixiangshen/downloads/dicom/manifest-1772811064422") / "metadata.csv"))
    if nsclc_root is None or nsclc_meta is None:
        recorder.skip("nsclc_smoke", "NSCLC root/metadata not found")
    else:
        catalog = build_nsclc_case_catalog(data_root=nsclc_root, metadata_csv=nsclc_meta, subject_id="AMC-003")
        recorder.run("nsclc_catalog", lambda: catalog.to_summary())
        preload_paths = catalog.all_series_paths(prefer_latest=True)
        preload_display_names = catalog.all_series_display_names(prefer_latest=True)
        default_alias = catalog.default_aliases[0] if catalog.default_aliases else "diagnostic_ct"
        recorder.run(
            "open_case_nsclc_all_series",
            lambda: ctx.open_case(
                nsclc_root,
                preset="ct_pet_chest",
                layout="four_up",
                clear_scene_first=True,
                baseline=True,
                size=512,
                dicom_series_dirs=preload_paths,
                series_display_names=preload_display_names,
                active_prefer=catalog.alias_active_prefer(default_alias),
                window_preset=catalog.alias_window_preset(default_alias),
            ),
        )
        vols_petct = recorder.run("list_volumes_nsclc", lambda: ctx.list_volumes())
        recorder.run("observe_nsclc_all_series", lambda: ctx.observe_standard_views(size=512, out_prefix="nsclc_all_series"))
        recorder.run("ct_lung_preset", lambda: ctx.apply_window_preset(preset="ct_lung"))
        recorder.run("observe_ct_lung", lambda: ctx.observe_view(view="axial", size=512, out_name="nsclc_ct_lung.png"))
        recorder.run("ct_mediastinal_preset", lambda: ctx.apply_window_preset(preset="ct_mediastinal"))
        recorder.run("observe_ct_mediastinal", lambda: ctx.observe_view(view="axial", size=512, out_name="nsclc_ct_mediastinal.png"))
        recorder.run("nsclc_segmentation_smoke", lambda: _segmentation_smoke(ctx, out_dir=out_dir, prefix="nsclc"))

        ct_name = _best_volume_name(ctx, vols_petct, "ctac ct thorax lung")
        pet_name = _best_volume_name(ctx, vols_petct, "suvbw pet uptake body weight")
        if ct_name and pet_name:
            recorder.run("fusion_ct_pet", lambda: ctx.fuse_volumes(background_volume_name=ct_name, foreground_volume_name=pet_name, opacity=0.35))
            recorder.run("observe_fusion_ct_pet", lambda: ctx.observe_standard_views(size=512, out_prefix="nsclc_fusion"))
            recorder.run("clear_fusion", lambda: ctx.clear_fusion())
        else:
            recorder.skip("fusion_ct_pet", f"Could not resolve CT/PET volumes from loaded names: ct={ct_name!r} pet={pet_name!r}")

    nsclc_segment_case = _pick_existing(
        os.environ.get("NSCLC_SEGMENT_CASE_DIR", ""),
        "/Users/weixiangshen/Downloads/dicom/manifest-1772841278501/NSCLC Radiogenomics/R01-001",
    )
    if nsclc_segment_case is None:
        recorder.skip("nsclc_segment_check", "NSCLC segment case dir not found")
    else:
        recorder.run("nsclc_segment_check", lambda: _segment_check_smoke(ctx, case_dir=nsclc_segment_case, out_dir=out_dir))

    summary = recorder.print_summary()
    print("\nDone. Outputs are in:", out_dir)
    print("Trace:", session.trace_path)
    print("State:", session.state_path)
    print("Evidence:", session.evidence_path)
    print("Runlog:", session.runlog_path)
    session.log_run_event("script_finished", script="manual_test.py", out_dir=str(out_dir), summary=summary)


if __name__ == "__main__":
    main()
