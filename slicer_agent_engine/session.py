from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_MEASUREMENT_KEYS = {
    "distance_mm",
    "angle_deg",
    "area_mm2",
    "area_cm2",
}

_ARTIFACT_RESULT_KEYS = {
    "png_path",
    "contact_sheet_png_path",
    "mp4_path",
    "nrrd_path",
    "saved_path",
}


@dataclass
class SessionManager:
    """Holds viewer state and writes append-only trace + evidence logs.

    `trace.jsonl` is the low-level action stream.
    `evidence.jsonl` is the review-oriented artifact/measurement stream.
    """

    out_dir: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: Dict[str, Any] = field(default_factory=dict)
    _seq: int = 0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.out_dir / "trace.jsonl"
        self.evidence_path = self.out_dir / "evidence.jsonl"
        self.runlog_path = self.out_dir / "runlog.jsonl"
        self.state_path = self.out_dir / "state.json"
        self.log_run_event(
            "session_created",
            out_dir=str(self.out_dir),
            trace_path=str(self.trace_path),
            evidence_path=str(self.evidence_path),
            runlog_path=str(self.runlog_path),
            state_path=str(self.state_path),
            pid=os.getpid(),
            cwd=os.getcwd(),
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python_version=sys.version.split()[0],
        )

    def update_state(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        self._flush_state()

    def _flush_state(self) -> None:
        tmp = dict(self.state)
        tmp["session_id"] = self.session_id
        tmp["updated_ts"] = time.time()
        self.state_path.write_text(json.dumps(tmp, indent=2, ensure_ascii=False), encoding="utf-8")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compact(self, value: Any, *, max_str: int = 800, max_list: int = 20, depth: int = 0) -> Any:
        if depth >= 6:
            return "<max-depth>"
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= 60:
                    out["__truncated__"] = f"{len(value) - 60} more keys"
                    break
                out[str(k)] = self._compact(v, max_str=max_str, max_list=max_list, depth=depth + 1)
            return out
        if isinstance(value, list):
            items = [self._compact(v, max_str=max_str, max_list=max_list, depth=depth + 1) for v in value[:max_list]]
            if len(value) > max_list:
                items.append(f"<... {len(value) - max_list} more items>")
            return items
        if isinstance(value, tuple):
            return self._compact(list(value), max_str=max_str, max_list=max_list, depth=depth + 1)
        if isinstance(value, str):
            if len(value) > max_str:
                return value[:max_str] + f" ... <{len(value) - max_str} more chars>"
            return value
        return value

    def debug_state_summary(self) -> Dict[str, Any]:
        st = dict(self.state)
        viewer_state = st.get("viewer_state") if isinstance(st.get("viewer_state"), dict) else {}
        views_summary: Dict[str, Any] = {}
        if isinstance(viewer_state.get("views"), dict):
            for view_name, payload in viewer_state["views"].items():
                if not isinstance(payload, dict):
                    views_summary[str(view_name)] = payload
                    continue
                views_summary[str(view_name)] = {
                    "orientation": payload.get("orientation"),
                    "slice_offset": payload.get("slice_offset"),
                    "min_offset": payload.get("min_offset"),
                    "max_offset": payload.get("max_offset"),
                    "background_volume_id": payload.get("background_volume_id"),
                    "background_volume_name": payload.get("background_volume_name"),
                    "foreground_volume_id": payload.get("foreground_volume_id"),
                    "foreground_volume_name": payload.get("foreground_volume_name"),
                    "foreground_opacity": payload.get("foreground_opacity"),
                    "slab_enabled": payload.get("slab_enabled"),
                    "slab_thickness_mm": payload.get("slab_thickness_mm"),
                    "slab_type": payload.get("slab_type"),
                }

        inventory_names: List[str] = []
        scene_inventory = st.get("scene_inventory")
        if isinstance(scene_inventory, list):
            for item in scene_inventory[:12]:
                if isinstance(item, dict):
                    inventory_names.append(str(item.get("name") or item.get("id") or ""))

        loaded_series_display_names = st.get("loaded_series_display_names")
        loaded_series_dirs = st.get("loaded_series_dirs")
        return {
            "case_preset": st.get("case_preset"),
            "dicom_dir": st.get("dicom_dir"),
            "loaded_series_display_names": list(loaded_series_display_names[:12]) if isinstance(loaded_series_display_names, list) else loaded_series_display_names,
            "loaded_series_dirs": list(loaded_series_dirs[:12]) if isinstance(loaded_series_dirs, list) else loaded_series_dirs,
            "active_volume_id": st.get("active_volume_id"),
            "active_volume_name": st.get("active_volume_name"),
            "active_tool_packs": st.get("active_tool_packs"),
            "active_segmentation_id": st.get("active_segmentation_id"),
            "active_segmentation_name": st.get("active_segmentation_name"),
            "active_segment_id": st.get("active_segment_id"),
            "active_segment_name": st.get("active_segment_name"),
            "scene_inventory_names": inventory_names,
            "views": views_summary,
        }

    def log_run_event(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "kind": "run_event",
            "event_type": str(event_type),
            "seq": self._next_seq(),
            "ts": time.time(),
            "session_id": self.session_id,
            **payload,
            "state_summary": self.debug_state_summary(),
        }
        self._write_jsonl(self.runlog_path, self._compact(record, max_str=1200, max_list=40))
        return record

    def summarize_tool_result(self, result: Any) -> Any:
        if isinstance(result, dict):
            summary: Dict[str, Any] = {}
            important_keys = [
                "ok",
                "error",
                "active_volume_id",
                "active_volume_name",
                "background_volume_id",
                "background_volume_name",
                "foreground_volume_id",
                "foreground_volume_name",
                "loaded_alias",
                "loaded_paths",
                "loaded_display_names",
                "segmentation_id",
                "segmentation_name",
                "segment_id",
                "segment_name",
                "n_segments",
                "n_segmentations",
                "distance_mm",
                "angle_deg",
                "area_mm2",
                "area_cm2",
                "volume_mm3",
                "volume_cm3",
                "voxel_count",
                "mean",
                "median",
                "min",
                "max",
                "std",
                "center_ras",
                "slice_offset",
                "min_offset",
                "max_offset",
                "view",
                "mode",
                "operation",
                "minimum_threshold",
                "maximum_threshold",
                "minimum_diameter_mm",
                "feature_size_mm",
                "segmentation_algorithm",
                "seed_source",
                "seed_view",
                "seed_bbox_1000",
                "seed_center_norm",
                "seed_ras",
                "seed_ijk",
                "margin_size_mm",
                "smoothing_method",
                "kernel_size_mm",
                "radius_mm",
                "action",
                "opacity",
                "text",
                "analysis_text",
            ]
            for key in important_keys:
                if key in result:
                    summary[key] = result.get(key)

            if "png_path" in result:
                summary["png_path"] = result.get("png_path")
            if "contact_sheet_png_path" in result:
                summary["contact_sheet_png_path"] = result.get("contact_sheet_png_path")
            if "mp4_path" in result:
                summary["mp4_path"] = result.get("mp4_path")
            if "png_paths" in result:
                png_paths = result.get("png_paths")
                if isinstance(png_paths, dict):
                    summary["png_paths"] = dict(list(png_paths.items())[:6])
                elif isinstance(png_paths, list):
                    summary["png_paths"] = png_paths[:6]

            for key in ("viewer_state", "scene_inventory_text", "available_packs", "slice_offset_range"):
                if key in result:
                    summary[key] = self._compact(result.get(key), max_str=400, max_list=10)

            nested_keys = [
                "segmentation",
                "center",
                "set_slice_offset",
                "fit",
                "wl",
                "window_preset",
                "jump",
                "fusion",
                "slab",
                "scroll",
                "recover",
                "analysis",
                "observation",
            ]
            for key in nested_keys:
                value = result.get(key)
                if value is None:
                    continue
                summary[key] = self._compact(value, max_str=400, max_list=10)

            if not summary:
                return self._compact(result, max_str=400, max_list=12)
            return self._compact(summary, max_str=400, max_list=12)
        return self._compact(result, max_str=400, max_list=12)

    def log_event(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        ok: bool = True,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        event_id = uuid.uuid4().hex
        record: Dict[str, Any] = {
            "kind": "action_event",
            "event_id": event_id,
            "seq": self._next_seq(),
            "ts": time.time(),
            "session_id": self.session_id,
            "tool_name": tool,
            "args": args,
            "ok": ok,
            "error": error,
            "result": result,
            "artifacts": artifacts or {},
            "state": dict(self.state),
        }
        self._write_jsonl(self.trace_path, record)

        self.log_run_event(
            "tool_trace",
            tool_name=tool,
            ok=ok,
            error=error,
            args=self._compact(args, max_str=500, max_list=20),
            result_summary=self.summarize_tool_result(result),
            artifacts=self._compact(artifacts or {}, max_str=500, max_list=20),
            trace_event_id=event_id,
        )

        for evidence in self._derive_evidence(record):
            self._write_jsonl(self.evidence_path, evidence)

    def _derive_evidence(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not record.get("ok"):
            return []

        tool = str(record.get("tool_name") or "")
        result = record.get("result")
        files = self._collect_files(result=result, artifacts=record.get("artifacts") or {})
        payload = self._structured_payload(result)

        evidence_type = self._infer_evidence_type(tool=tool, result=result, files=files)
        if evidence_type is None:
            return []

        evidence = {
            "kind": "evidence_object",
            "evidence_id": uuid.uuid4().hex,
            "ts": time.time(),
            "session_id": self.session_id,
            "source_event_id": record.get("event_id"),
            "tool_name": tool,
            "type": evidence_type,
            "files": files,
            "structured_payload": payload,
        }
        return [evidence]

    def _collect_files(self, *, result: Any, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        files: Dict[str, Any] = {}
        for key, value in (artifacts or {}).items():
            if value:
                files[key] = value

        if isinstance(result, dict):
            for key in _ARTIFACT_RESULT_KEYS:
                value = result.get(key)
                if value:
                    files[key] = value
            png_paths = result.get("png_paths")
            if isinstance(png_paths, dict) and png_paths:
                files["png_paths"] = png_paths
            elif isinstance(png_paths, list) and png_paths:
                files["png_paths"] = png_paths

            baseline = result.get("baseline")
            if isinstance(baseline, dict) and baseline:
                files["baseline"] = baseline

            if result.get("frames_dir"):
                files["frames_dir"] = result["frames_dir"]

        return files

    def _structured_payload(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return result

        payload: Dict[str, Any] = {}
        for key, value in result.items():
            if key in _ARTIFACT_RESULT_KEYS or key in {"png_paths", "baseline", "frames_dir", "contact_sheet_png_path"}:
                continue
            payload[key] = value
        return payload

    def _infer_evidence_type(self, *, tool: str, result: Any, files: Dict[str, Any]) -> Optional[str]:
        if tool == "open_case":
            return "scene_open"
        if tool in {"scroll", "scroll_sweep"}:
            if isinstance(result, dict) and str(result.get("mode") or "") == "video":
                return "scroll_video"
            return "keyframe_sheet"
        if tool.startswith("measure_"):
            return "measurement"
        if tool.startswith("roi_stats"):
            return "roi_stats"
        if tool == "sample_intensity_ras":
            return "intensity_sample"
        if tool in {"fusion", "set_fusion", "clear_fusion"}:
            return "fusion_state"
        if tool == "compute_subtraction":
            return "derived_volume"
        if tool == "segment_statistics":
            return "segment_stats"
        if tool in {"segment_from_threshold", "segment_local_threshold", "segment_edit_sphere", "segment_margin", "segment_smoothing", "segment_islands", "segment_logical", "center_on_segment", "list_segmentations", "list_segments", "export_segmentation_dicom"}:
            return "segmentation_state"
        if tool in {"observe", "select_volume", "wl", "window_preset", "zoom", "fit", "jump", "recover_standard_views", "thick_slab", "set_layout", "get_slice_png", "get_screenshot_png", "get_threeD_png", "get_timeimage_png", "set_slice_offset"}:
            return "viewport_capture"
        if files:
            return "artifact"
        if isinstance(result, dict):
            for key in _MEASUREMENT_KEYS:
                if key in result:
                    return "measurement"
        return None
