from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from _script_support import ensure_repo_imports

ensure_repo_imports()

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
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        }

    def print_summary(self) -> Dict[str, Any]:
        s = self.summary()
        failed_names = [r["name"] for r in s["failed"]]
        skipped_names = [r["name"] for r in s["skipped"]]
        print("\n===== SMOKE TEST SUMMARY =====")
        print(f"total={s['total']} passed={s['passed']} failed={len(s['failed'])} skipped={len(s['skipped'])}")
        if failed_names:
            print("FAILED:")
            for r in s["failed"]:
                print(f"  - {r['name']}: {r.get('reason') or ''}")
        if skipped_names:
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
        score = ctx._volume_match_score(query=query, candidate_name=name)  # internal helper reused for test stability
        ranked.append((score, name))
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
    bone_copy_name = f"{prefix}_bone_copy"
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

    _sub("initial_segmentations", lambda: ctx.list_segmentations())
    _sub("threshold", lambda: ctx.segment_from_threshold(segment_name=bone_name, minimum_threshold=200.0, maximum_threshold=3000.0))
    _sub("segments_after_threshold", lambda: ctx.list_segments())
    _sub("keep_largest", lambda: ctx.segment_islands(segment_name=bone_name, operation="KEEP_LARGEST_ISLAND", minimum_size=5000))
    stats = _sub("stats_after_islands", lambda: ctx.segment_statistics(segment_name=bone_name))
    center_ras = stats.get("center_ras") if isinstance(stats, dict) else None
    _sub("median_smooth", lambda: ctx.segment_smoothing(segment_name=bone_name, smoothing_method="MEDIAN", kernel_size_mm=2.0))
    _sub("margin", lambda: ctx.segment_margin(segment_name=bone_name, margin_size_mm=1.0))
    stats2 = _sub("stats", lambda: ctx.segment_statistics(segment_name=bone_name))
    if isinstance(stats2, dict) and stats2.get("center_ras"):
        center_ras = stats2.get("center_ras")
    if center_ras:
        _sub("sphere_edit", lambda: ctx.segment_edit_sphere(segment_name=bone_name, ras_center=center_ras, radius_mm=5.0, action="add"))
    _sub("center", lambda: ctx.center_on_segment(segment_name=bone_name))
    _sub("recover_after_center", lambda: ctx.recover_standard_views())
    _sub(
        "local_threshold_bbox_center_seed",
        lambda: ctx.segment_local_threshold(
            segment_name=local_name,
            minimum_threshold=200.0,
            maximum_threshold=3000.0,
            minimum_diameter_mm=3.0,
            feature_size_mm=3.0,
            segmentation_algorithm="GrowCut",
            seed_view="axial",
            seed_bbox_1000=[440, 440, 560, 560],
        ),
    )
    _sub("logical_copy", lambda: ctx.segment_logical(operation="COPY", segment_name=bone_copy_name, modifier_segment_name=bone_name))
    _sub("segments_final", lambda: ctx.list_segments())
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


def main() -> None:
    recorder = StepRecorder()
    base_url = os.environ.get("SLICER_BASE_URL", "http://localhost:2016")
    dicom_dir = Path(os.environ.get("DICOM_DIR", "/Users/weixiangshen/downloads/dicom/MRHead_DICOM"))
    out_dir = Path(os.environ.get("OUT_DIR", "/Users/weixiangshen/downloads/dicom/test_tmp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = bootstrap_runtime(
        out_dir=out_dir,
        log_name="l0l1_smoke_test.log",
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
    session.log_run_event("script_start", script="l0l1_smoke_test.py", out_dir=str(out_dir), base_url=base_url)
    ctx = runtime.context.ctx

    # MRI / L0-L1 smoke
    recorder.run("ping", lambda: ctx.ping())
    recorder.run("open_case", lambda: ctx.open_case(dicom_dir, baseline=True, size=512))
    recorder.run("resample_isotropic", lambda: ctx.resample_volume_isotropic(output_spacing_mm=1.0, output_name="smoke_iso_1mm", make_active=False), required=False)
    recorder.run("baseline_axial", lambda: ctx.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=512, out_name="l0_baseline_axial.png"))
    recorder.run("wl_auto", lambda: ctx.set_window_level(auto=True))
    recorder.run("l0_auto_wl_img", lambda: ctx.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=512, out_name="l0_auto_wl.png"))
    recorder.run("wl_manual", lambda: ctx.set_window_level(window=200.0, level=80.0, auto=False))
    recorder.run("l0_manual_wl_img", lambda: ctx.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=512, out_name="l0_manual_wl.png"))
    recorder.run("fit", lambda: ctx.fit_slice(view="axial"))
    recorder.run("zoom", lambda: ctx.zoom_slice_relative(view="axial", factor=1.8))
    recorder.run("l0_zoomed_img", lambda: ctx.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=512, out_name="l0_zoomed.png"))
    recorder.run("slab_mip", lambda: ctx.set_thick_slab(view="axial", enabled=True, thickness_mm=12.0, mode="mip"))
    recorder.run("l1_thick_slab_mip_img", lambda: ctx.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=512, out_name="l1_thick_slab_mip.png"))
    recorder.run("slab_off", lambda: ctx.set_thick_slab(view="axial", enabled=False, thickness_mm=0.0, mode="mip"))
    recorder.run("axial_offset_mid_recovery", lambda: _move_axial_to_mid(ctx))
    recorder.run("recover_standard_views", lambda: ctx.recover_standard_views())
    vols = recorder.run("list_volumes_initial", lambda: ctx.list_volumes())
    vol0 = vols[0]["id"] if isinstance(vols, list) and vols else None
    recorder.run("roi_stats", lambda: ctx.roi_stats_ijk(ijk_min=[80, 80, 40], ijk_max=[160, 160, 90], volume_id=vol0))
    recorder.run("bookmark_add", lambda: ctx.bookmark_add(name="mid_axial", view="axial", orientation="axial", scroll_to=0.5, size=512))
    recorder.run("bookmark_list", lambda: ctx.bookmark_list())
    recorder.run("scroll_keyframes", lambda: ctx.scroll_sweep(view="axial", orientation="axial", n_frames=12, size=384, output="keyframes", out_prefix="l1_scroll"))

    # NIfTI multi-sequence smoke
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
        recorder.run("open_case_nifti", lambda: ctx.open_case(nifti_case_dir, preset="mri_brain", baseline=True, size=512, clear_scene_first=True))
        vols2 = recorder.run("list_volumes_nifti", lambda: ctx.list_volumes())
        if isinstance(vols2, list):
            for w in ["T1c", "FLAIR", "T2", "T1"]:
                match = next((v.get("name") for v in vols2 if isinstance(v, dict) and w.lower() in str(v.get("name", "")).lower()), None)
                if not match:
                    continue
                recorder.run(f"select_volume_{w}", lambda m=match: ctx.select_volume(volume_name=m))
                recorder.run(f"observe_{w}", lambda w=w: ctx.observe_standard_views(size=512, out_prefix=f"nifti_{w.lower()}"))

    # CT/PET NSCLC smoke
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
        vols_nsclc = recorder.run("list_volumes_nsclc", lambda: ctx.list_volumes())
        recorder.run("window_ct_lung", lambda: ctx.apply_window_preset(preset="ct_lung"))
        recorder.run("observe_ct_lung", lambda: ctx.observe_view(view="axial", size=512, out_name="nsclc_ct_lung_smoke.png"))
        recorder.run("window_ct_mediastinal", lambda: ctx.apply_window_preset(preset="ct_mediastinal"))
        recorder.run("observe_ct_mediastinal", lambda: ctx.observe_view(view="axial", size=512, out_name="nsclc_ct_mediastinal_smoke.png"))
        recorder.run("nsclc_segmentation_smoke", lambda: _segmentation_smoke(ctx, out_dir=out_dir, prefix="nsclc"))

        ct_name = _best_volume_name(ctx, vols_nsclc, "ctac ct thorax lung")
        pet_name = _best_volume_name(ctx, vols_nsclc, "suvbw pet uptake body weight")
        if ct_name and pet_name:
            recorder.run("fusion_ct_pet", lambda: ctx.fuse_volumes(background_volume_name=ct_name, foreground_volume_name=pet_name, opacity=0.35))
            recorder.run("observe_ct_pet_fusion", lambda: ctx.observe_standard_views(size=512, out_prefix="nsclc_fusion_smoke"))
            recorder.run("clear_fusion", lambda: ctx.clear_fusion())
        else:
            recorder.skip("fusion_ct_pet", f"Could not resolve CT/PET volumes from loaded names: ct={ct_name!r} pet={pet_name!r}")

    summary = recorder.print_summary()
    print("\nDone. Outputs:", out_dir)
    print("Trace:", session.trace_path)
    print("State:", session.state_path)
    print("Evidence:", session.evidence_path)
    print("Runlog:", session.runlog_path)
    session.log_run_event("script_finished", script="l0l1_smoke_test.py", out_dir=str(out_dir), summary=summary)


if __name__ == "__main__":
    main()
