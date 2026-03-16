from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from .click_contract import (
    annotate_png_with_bbox,
    annotate_png_with_point,
    bbox_center_norm,
    normalize_bbox_1000,
    normalize_point_1000,
    point_norm_from_1000,
)
from .context_briefs import format_scene_inventory_for_prompt
from .slicer_client import SlicerClient
from .session import SessionManager
from .video_renderer import VideoRenderer

if TYPE_CHECKING:
    from .gemini_video import GeminiVideoAnalyzer

logger = logging.getLogger(__name__)


def _as_path(p: Union[str, Path]) -> Path:
    return Path(p).expanduser().resolve()


_VIEW_ALIASES = {
    "axial": "axial",
    "red": "axial",
    "r": "axial",
    "sagittal": "sagittal",
    "yellow": "sagittal",
    "y": "sagittal",
    "coronal": "coronal",
    "green": "coronal",
    "g": "coronal",
}
_STANDARD_VIEWS = ("axial", "sagittal", "coronal")

_VIEWER_MUTATION_TOOLS = {
    "select_volume",
    "set_window_level",
    "apply_window_preset",
    "set_interpolation",
    "set_slice_orientation",
    "set_slice_offset",
    "set_slice_scroll_to",
    "fit_slice",
    "zoom_slice_relative",
    "set_field_of_view",
    "set_xyz_origin",
    "jump_to_ras",
    "recover_standard_views",
    "set_linked_slices",
    "set_layout",
    "set_thick_slab",
    "set_fusion",
    "clear_fusion",
    "compute_subtraction",
    "center_on_segment",
}

_SCENE_MUTATION_TOOLS = {
    "clear_scene",
    "load_dicom",
    "load_dicom_series",
    "load_nifti",
    "open_case",
    "compute_subtraction",
}


def _canonical_view(view: Optional[str], *, default: str = "axial") -> str:
    v = (view or default).strip().lower()
    if v in _VIEW_ALIASES:
        return _VIEW_ALIASES[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _normalize_public_view_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    view = normalized.get("view")
    if isinstance(view, str):
        try:
            normalized["view"] = _canonical_view(view)
        except ValueError:
            pass

    views = normalized.get("views")
    if isinstance(views, dict):
        normalized_views: Dict[str, Any] = {}
        for key, value in views.items():
            if isinstance(key, str):
                try:
                    normalized_views[_canonical_view(key)] = value
                    continue
                except ValueError:
                    pass
            normalized_views[key] = value
        normalized["views"] = normalized_views

    return normalized


@dataclass
class ToolContext:
    """High-level tool surface used by agents.

    This wraps:
    - SlicerClient: raw WebServer HTTP calls
    - SessionManager: state + trace logging
    - VideoRenderer: ffmpeg for cine MP4 encoding

    Design goal: tools return JSON-serializable dicts that point to artifacts on disk.
    """

    client: SlicerClient
    session: SessionManager
    bridge_dir: Path
    video: Optional[VideoRenderer] = None
    gemini: Optional["GeminiVideoAnalyzer"] = None

    # -------------------------
    # Internal helpers
    # -------------------------

    def _log(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        ok: bool = True,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        self._maybe_refresh_state_before_log(tool)
        self.session.log_event(tool=tool, args=args, result=result, ok=ok, error=error, artifacts=artifacts)

    def _artifact_path(self, name: str) -> Path:
        p = self.session.out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _estimate_load_timeout_s(self, *, n_series: int = 1) -> float:
        n = max(int(n_series), 1)
        return float(max(self.client.timeout_s, min(900.0, 90.0 + 20.0 * n)))

    def _annotate_bbox_preview(self, *, png_path: str, bbox_1000: Sequence[int | float], out_name: str, label: Optional[str] = None) -> str:
        out_path = self._artifact_path(out_name)
        annotate_png_with_bbox(png_path, bbox_1000=bbox_1000, out_path=out_path, label=label)
        return str(out_path)

    def _annotate_point_preview(self, *, png_path: str, point_1000: Sequence[int | float], out_name: str, label: Optional[str] = None) -> str:
        out_path = self._artifact_path(out_name)
        annotate_png_with_point(png_path, point_1000=point_1000, out_path=out_path, label=label)
        return str(out_path)

    def capture_slice_view_png(self, *, view: str = "axial", out_name: Optional[str] = None, include_controller: bool = False) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        args = {"view": canonical_view, "include_controller": bool(include_controller), "out_name": out_name}
        try:
            if out_name is None:
                out_name = f"sliceview_{canonical_view}.png"
            out_path = self._artifact_path(out_name)
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="capture_slice_view_png",
                args={"view": canonical_view, "output_path": str(out_path), "include_controller": bool(include_controller)},
                session_id=self.session.session_id,
            )
            result = {**(result or {}), "png_path": str(out_path)}
            self._log(tool="capture_slice_view_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="capture_slice_view_png", args=args, result=None, ok=False, error=str(e))
            raise

    def _refresh_viewer_state_quiet(self) -> None:
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_viewer_state",
                args={},
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if isinstance(result, dict) and result.get("ok"):
                self.session.update_state(viewer_state=result)
        except Exception:
            logger.debug("viewer state refresh failed", exc_info=True)

    def _refresh_scene_inventory_quiet(self) -> None:
        try:
            scene_inventory = self.client.list_volumes()
            self.session.update_state(
                scene_inventory=scene_inventory,
                scene_inventory_text=format_scene_inventory_for_prompt(
                    scene_inventory,
                    active_volume_id=self.session.state.get("active_volume_id"),
                    active_volume_name=self.session.state.get("active_volume_name"),
                ),
            )
        except Exception:
            logger.debug("scene inventory refresh failed", exc_info=True)

    def _maybe_refresh_state_before_log(self, tool: str) -> None:
        if tool in _VIEWER_MUTATION_TOOLS:
            self._refresh_viewer_state_quiet()
        if tool in _SCENE_MUTATION_TOOLS:
            self._refresh_scene_inventory_quiet()

    # -------------------------
    # Basic / system
    # -------------------------

    def ping(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.get_system_version()
            self._log(tool="ping", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="ping", args=args, result=None, ok=False, error=str(e))
            raise

    def shutdown_slicer(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.shutdown_system()
            self._log(tool="shutdown_slicer", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="shutdown_slicer", args=args, result=None, ok=False, error=str(e))
            raise

    def clear_scene(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            # Prefer bridge (stable) if available, else fall back to client.clear_scene
            try:
                result = self.client.exec_bridge(bridge_dir=self.bridge_dir, tool="clear_scene", args={}, session_id=self.session.session_id)
            except Exception:
                result = self.client.clear_scene()
            self.session.update_state(active_volume_id=None, active_volume_name=None)
            self._log(tool="clear_scene", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="clear_scene", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Data loading / volume selection (bridge)
    # -------------------------

    def load_dicom(
        self,
        dicom_dir: Union[str, Path],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        dicom_dir_p = _as_path(dicom_dir)
        args = {
            "dicom_dir": str(dicom_dir_p),
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_dicom",
                args={"dicom_dir": str(dicom_dir_p), "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=1),
            )
            if result.get("ok"):
                self.session.update_state(
                    dicom_dir=str(dicom_dir_p),
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_dicom", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_dicom", args=args, result=None, ok=False, error=str(e))
            raise

    def load_dicom_series(
        self,
        dicom_dirs: Sequence[Union[str, Path]],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        dicom_dirs_p = [str(_as_path(d)) for d in dicom_dirs]
        args = {
            "dicom_dirs": dicom_dirs_p,
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_dicom_series",
                args={"dicom_dirs": dicom_dirs_p, "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=len(dicom_dirs_p)),
            )
            if result.get("ok"):
                self.session.update_state(
                    dicom_dir=dicom_dirs_p[0] if len(dicom_dirs_p) == 1 else None,
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_dicom_series", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_dicom_series", args=args, result=None, ok=False, error=str(e))
            raise

    def load_nifti(
        self,
        nifti: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load NIfTI volume(s) from a file, a directory, or an explicit file list."""

        nifti_dir: Optional[str] = None
        nifti_files: Optional[List[str]] = None
        if isinstance(nifti, (str, Path)):
            p_in = _as_path(nifti)
            if p_in.is_dir():
                nifti_dir = str(p_in)
            else:
                nifti_files = [str(p_in)]
        else:
            nifti_files = [str(_as_path(x)) for x in nifti]

        args = {
            "nifti_dir": nifti_dir,
            "nifti_files": nifti_files,
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_nifti",
                args={"nifti_dir": nifti_dir, "nifti_files": nifti_files, "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=len(nifti_files or ([nifti_dir] if nifti_dir else []))),
            )
            if result.get("ok"):
                self.session.update_state(
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_nifti", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_nifti", args=args, result=None, ok=False, error=str(e))
            raise

    def list_volumes(self) -> List[Dict[str, str]]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_volumes()
            self._log(tool="list_volumes", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_volumes", args=args, result=None, ok=False, error=str(e))
            raise

    def select_volume(self, *, volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"volume_id": volume_id, "volume_name": volume_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="select_volume",
                args=args,
                session_id=self.session.session_id,
            )
            if result.get("ok"):
                self.session.update_state(
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="select_volume", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="select_volume", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # L0: Viewer actuation (bridge-backed)
    # -------------------------

    def get_viewer_state(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_viewer_state",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if result.get("ok"):
                # Store the last known full viewer state for debugging/replay.
                self.session.update_state(viewer_state=result)
            self._log(tool="get_viewer_state", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_viewer_state", args=args, result=None, ok=False, error=str(e))
            raise

    def set_window_level(
        self,
        *,
        window: Optional[float] = None,
        level: Optional[float] = None,
        auto: Optional[bool] = None,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "window": window,
            "level": level,
            "auto": auto,
            "volume_id": volume_id,
            "volume_name": volume_name,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_window_level",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_window_level", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_window_level", args=args, result=None, ok=False, error=str(e))
            raise

    def apply_window_preset(
        self,
        *,
        preset: str,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a clinically common window preset.

        Supported presets:
          - ct_lung / lung:          W=1500, L=-600
          - ct_mediastinal / mediastinal / soft_tissue: W=400, L=40
          - ct_bone / bone:          W=2000, L=300
          - pet / pet_auto / auto:   auto window/level
        """
        key = str(preset or '').strip().lower()
        mapping = {
            'ct_lung': (1500.0, -600.0),
            'lung': (1500.0, -600.0),
            'ct_mediastinal': (400.0, 40.0),
            'mediastinal': (400.0, 40.0),
            'soft_tissue': (400.0, 40.0),
            'ct_soft_tissue': (400.0, 40.0),
            'ct_bone': (2000.0, 300.0),
            'bone': (2000.0, 300.0),
        }
        if key in {'pet', 'pet_auto', 'auto', 'pet_default'}:
            result = self.set_window_level(auto=True, volume_id=volume_id, volume_name=volume_name)
            self._log(tool='apply_window_preset', args={'preset': preset, 'volume_id': volume_id, 'volume_name': volume_name}, result=result)
            return result
        if key not in mapping:
            raise ValueError(f'Unsupported window preset: {preset}')
        window, level = mapping[key]
        result = self.set_window_level(window=window, level=level, auto=False, volume_id=volume_id, volume_name=volume_name)
        self._log(tool='apply_window_preset', args={'preset': preset, 'volume_id': volume_id, 'volume_name': volume_name}, result=result)
        return result

    def set_interpolation(
        self,
        *,
        interpolate: bool,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {"interpolate": bool(interpolate), "volume_id": volume_id, "volume_name": volume_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_interpolation",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_interpolation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_interpolation", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_orientation(self, *, view: str, orientation: str) -> Dict[str, Any]:
        args = {"view": view, "orientation": orientation}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_orientation",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_orientation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_orientation", args=args, result=None, ok=False, error=str(e))
            raise

    def get_slice_offset_range(self, *, view: str) -> Dict[str, Any]:
        args = {"view": view}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_slice_offset_range",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="get_slice_offset_range", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_slice_offset_range", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_offset(self, *, view: str, offset: float) -> Dict[str, Any]:
        args = {"view": view, "offset": float(offset)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_offset",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_offset", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_offset", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_scroll_to(self, *, view: str, scroll_to: float) -> Dict[str, Any]:
        args = {"view": view, "scroll_to": float(scroll_to)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_scroll_to",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_scroll_to", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_scroll_to", args=args, result=None, ok=False, error=str(e))
            raise

    def fit_slice(self, *, view: str) -> Dict[str, Any]:
        args = {"view": view}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="fit_slice",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="fit_slice", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="fit_slice", args=args, result=None, ok=False, error=str(e))
            raise

    def zoom_slice_relative(self, *, view: str, factor: float) -> Dict[str, Any]:
        args = {"view": view, "factor": float(factor)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="zoom_slice_relative",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="zoom_slice_relative", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="zoom_slice_relative", args=args, result=None, ok=False, error=str(e))
            raise

    def set_field_of_view(self, *, view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
        args = {"view": view, "field_of_view": [float(v) for v in field_of_view]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_field_of_view",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_field_of_view", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_field_of_view", args=args, result=None, ok=False, error=str(e))
            raise

    def set_xyz_origin(self, *, view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
        args = {"view": view, "xyz_origin": [float(v) for v in xyz_origin]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_xyz_origin",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_xyz_origin", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_xyz_origin", args=args, result=None, ok=False, error=str(e))
            raise

    def jump_to_ras(self, *, ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
        args = {"ras": [float(v) for v in ras], "centered": bool(centered)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="jump_to_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="jump_to_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="jump_to_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def recover_standard_views(
        self,
        *,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
        centered: bool = True,
        fit: bool = True,
    ) -> Dict[str, Any]:
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "centered": bool(centered),
            "fit": bool(fit),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="recover_standard_views",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="recover_standard_views", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="recover_standard_views", args=args, result=None, ok=False, error=str(e))
            raise

    def set_linked_slices(self, *, enabled: bool) -> Dict[str, Any]:
        args = {"enabled": bool(enabled)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_linked_slices",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_linked_slices", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_linked_slices", args=args, result=None, ok=False, error=str(e))
            raise

    def set_layout(self, *, layout: str) -> Dict[str, Any]:
        args = {"layout": layout}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_layout",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_layout", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_layout", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Rendering (PNG) + cine (MP4)
    # -------------------------

    def get_slice_png(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
        out_name: Optional[str] = None,
        offset: Optional[float] = None,
        copy_slice_geometry_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        args = {
            "view": canonical_view,
            "orientation": orientation,
            "scroll_to": float(scroll_to),
            "size": int(size),
            "offset": offset,
            "copy_slice_geometry_from": copy_slice_geometry_from,
        }
        try:
            if out_name is None:
                out_name = f"slice_{canonical_view}_{orientation}_{scroll_to:.3f}_{int(size)}.png"
            out_path = self._artifact_path(out_name)
            self.client.save_slice_png(
                out_path,
                view=canonical_view,
                orientation=orientation,
                scroll_to=scroll_to,
                size=size,
                offset=offset,
                copy_slice_geometry_from=copy_slice_geometry_from,
            )
            result = {"png_path": str(out_path)}
            self._log(tool="get_slice_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_slice_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_screenshot_png(self, *, out_name: str = "screenshot.png") -> Dict[str, Any]:
        args = {"out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_screenshot_png(out_path)
            result = {"png_path": str(out_path)}
            self._log(tool="get_screenshot_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_screenshot_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_threeD_png(self, *, look_from_axis: Optional[str] = None, out_name: str = "threeD.png") -> Dict[str, Any]:
        args = {"look_from_axis": look_from_axis, "out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_threeD_png(out_path, look_from_axis=look_from_axis)
            result = {"png_path": str(out_path)}
            self._log(tool="get_threeD_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_threeD_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_timeimage_png(self, *, color: str = "pink", out_name: str = "timeimage.png") -> Dict[str, Any]:
        args = {"color": color, "out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_timeimage_png(out_path, color=color)
            result = {"png_path": str(out_path)}
            self._log(tool="get_timeimage_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_timeimage_png", args=args, result=None, ok=False, error=str(e))
            raise

    def capture_cine(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        start: float = 0.0,
        end: float = 1.0,
        n_frames: int = 60,
        size: int = 512,
        fps: int = 12,
        frames_dir_name: Optional[str] = None,
        out_mp4_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.video is None:
            raise RuntimeError("VideoRenderer is not configured (ffmpeg not available)")

        canonical_view = _canonical_view(view)

        args = {
            "view": canonical_view,
            "orientation": orientation,
            "start": float(start),
            "end": float(end),
            "n_frames": int(n_frames),
            "size": int(size),
            "fps": int(fps),
        }

        try:
            if frames_dir_name is None:
                frames_dir_name = f"frames_{canonical_view}_{orientation}_{n_frames}f"
            frames_dir = self._artifact_path(frames_dir_name)
            frames_dir.mkdir(parents=True, exist_ok=True)

            if out_mp4_name is None:
                out_mp4_name = f"cine_{canonical_view}_{orientation}_{n_frames}f_{fps}fps.mp4"
            out_mp4 = self._artifact_path(out_mp4_name)

            if n_frames <= 1:
                raise ValueError("n_frames must be >= 2")

            for i in range(n_frames):
                t = start + (end - start) * (i / (n_frames - 1))
                frame_path = frames_dir / f"frame_{i:05d}.png"
                self.client.save_slice_png(frame_path, view=canonical_view, orientation=orientation, scroll_to=float(t), size=size)

            mp4_path = self.video.encode_mp4_from_pattern(
                input_pattern=frames_dir / "frame_%05d.png",
                output_path=out_mp4,
                fps=fps,
            )

            result = {"mp4_path": str(mp4_path), "frames_dir": str(frames_dir)}
            self._log(tool="capture_cine", args=args, result=result, artifacts={"mp4": str(mp4_path)})
            return result

        except Exception as e:
            self._log(tool="capture_cine", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # L1: deterministic quantification / processing (bridge)
    # -------------------------

    def roi_stats_ijk(
        self,
        *,
        ijk_min: Sequence[int],
        ijk_max: Sequence[int],
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Default to the currently selected (active) volume if not provided.
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")

        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ijk_min": list(map(int, ijk_min)),
            "ijk_max": list(map(int, ijk_max)),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="roi_stats_ijk",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="roi_stats_ijk", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="roi_stats_ijk", args=args, result=None, ok=False, error=str(e))
            raise

    def roi_stats_ras_box(
        self,
        *,
        ras_min: Sequence[float],
        ras_max: Sequence[float],
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ras_min": [float(v) for v in ras_min],
            "ras_max": [float(v) for v in ras_max],
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="roi_stats_ras_box",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="roi_stats_ras_box", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="roi_stats_ras_box", args=args, result=None, ok=False, error=str(e))
            raise

    def sample_intensity_ras(
        self,
        *,
        ras: Sequence[float],
        method: str = "nearest",
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ras": [float(v) for v in ras],
            "method": str(method),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="sample_intensity_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="sample_intensity_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="sample_intensity_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_distance_ras(self, *, p1: Sequence[float], p2: Sequence[float]) -> Dict[str, Any]:
        args = {"p1": [float(v) for v in p1], "p2": [float(v) for v in p2]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_distance_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_distance_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_distance_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_angle_ras(self, *, p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> Dict[str, Any]:
        args = {"p1": [float(v) for v in p1], "p2": [float(v) for v in p2], "p3": [float(v) for v in p3]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_angle_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_angle_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_angle_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_area_polygon_ras(self, *, points: Sequence[Sequence[float]]) -> Dict[str, Any]:
        args = {"points": [[float(v) for v in p] for p in points]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_area_polygon_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_area_polygon_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_area_polygon_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def set_thick_slab(self, *, view: str, enabled: bool, thickness_mm: float = 0.0, mode: str = "mip") -> Dict[str, Any]:
        args = {"view": view, "enabled": bool(enabled), "thickness_mm": float(thickness_mm), "mode": str(mode)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_thick_slab",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_thick_slab", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_thick_slab", args=args, result=None, ok=False, error=str(e))
            raise

    @staticmethod
    def _normalize_volume_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    def _volume_match_score(self, *, query: str, candidate_name: str) -> float:
        q_raw = str(query or "").strip()
        c_raw = str(candidate_name or "").strip()
        q_norm = self._normalize_volume_key(q_raw)
        c_norm = self._normalize_volume_key(c_raw)
        if not q_norm or not c_norm:
            return float("-inf")
        if q_raw == c_raw:
            return 10_000.0
        if q_norm == c_norm:
            return 9_000.0

        score = 0.0
        if q_norm in c_norm:
            score += 6_000.0 + min(len(q_norm), 200) / 10.0

        q_tokens = [tok for tok in q_norm.split() if tok]
        c_tokens = [tok for tok in c_norm.split() if tok]
        overlap = len(set(q_tokens).intersection(c_tokens))
        score += overlap * 300.0

        ratio = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
        score += ratio * 400.0

        q_is_pet = any(tok in q_norm for tok in ["pet", "suv", "uptake", "mac", "nac", "wb"])
        q_is_ct = any(tok in q_norm for tok in ["ct", "ctac", "thorax", "lung", "chest", "b70", "b40", "b45"])
        c_is_pet = any(tok in c_norm for tok in ["pet", "suv", "uptake", "body weight", "suvbw", "suvlbm", "suvbsa", "suvibw"])
        c_is_ct = any(tok in c_norm for tok in ["ct", "ctac", "thorax", "lung", "chest", "b70", "b40", "b45"])

        if q_is_pet:
            if c_is_pet:
                score += 700.0
            if "suvbw" in c_norm or "body weight" in c_norm:
                score += 450.0
            if "standardized uptake value body weight" in c_norm:
                score += 350.0
            if c_is_ct:
                score -= 600.0
        if q_is_ct:
            if c_is_ct:
                score += 700.0
            if "ctac" in c_norm:
                score += 450.0
            if c_is_pet:
                score -= 700.0

        if q_tokens and all(tok in c_norm for tok in q_tokens[:2]):
            score += 250.0
        return score

    def _resolve_volume_id(self, *, volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> str:
        if volume_id:
            return str(volume_id)
        vols = self.list_volumes()
        if volume_name:
            exact = [v for v in vols if str(v.get("name")) == str(volume_name)]
            if exact:
                return str(exact[0]["id"])

            query = str(volume_name)
            ranked = sorted(
                (
                    (self._volume_match_score(query=query, candidate_name=str(v.get("name", ""))), v)
                    for v in vols
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if ranked and ranked[0][0] > 250.0:
                return str(ranked[0][1]["id"])
        raise ValueError(f"Could not resolve volume reference id={volume_id!r} name={volume_name!r}")

    def fuse_volumes(
        self,
        *,
        background_volume_id: Optional[str] = None,
        background_volume_name: Optional[str] = None,
        foreground_volume_id: Optional[str] = None,
        foreground_volume_name: Optional[str] = None,
        opacity: float = 0.35,
    ) -> Dict[str, Any]:
        bg_id = self._resolve_volume_id(volume_id=background_volume_id, volume_name=background_volume_name)
        fg_id = self._resolve_volume_id(volume_id=foreground_volume_id, volume_name=foreground_volume_name)
        return self.set_fusion(background_volume_id=bg_id, foreground_volume_id=fg_id, opacity=opacity)

    def set_fusion(self, *, background_volume_id: str, foreground_volume_id: str, opacity: float = 0.5) -> Dict[str, Any]:
        args = {"background_volume_id": background_volume_id, "foreground_volume_id": foreground_volume_id, "opacity": float(opacity)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_fusion",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_fusion", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_fusion", args=args, result=None, ok=False, error=str(e))
            raise

    def clear_fusion(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="clear_fusion",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="clear_fusion", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="clear_fusion", args=args, result=None, ok=False, error=str(e))
            raise

    def resample_volume_isotropic(
        self,
        *,
        output_spacing_mm: float = 1.0,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        output_name: Optional[str] = None,
        interpolation_type: str = "linear",
        make_active: bool = True,
    ) -> Dict[str, Any]:
        args = {
            "output_spacing_mm": float(output_spacing_mm),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "output_name": output_name,
            "interpolation_type": interpolation_type,
            "make_active": bool(make_active),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="resample_volume_isotropic",
                args=args,
                session_id=self.session.session_id,
                timeout_s=max(self.client.timeout_s, 180.0),
            )
            if isinstance(result, dict) and result.get("made_active"):
                self.session.update_state(
                    active_volume_id=result.get("output_volume_id"),
                    active_volume_name=result.get("output_volume_name"),
                )
            self._log(tool="resample_volume_isotropic", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="resample_volume_isotropic", args=args, result=None, ok=False, error=str(e))
            raise

    def compute_subtraction(self, *, volume_a_id: str, volume_b_id: str, output_name: str = "Subtraction") -> Dict[str, Any]:
        args = {"volume_a_id": volume_a_id, "volume_b_id": volume_b_id, "output_name": output_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="compute_subtraction",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="compute_subtraction", args=args, result=result)
            # Optionally update active volume in session (bridge sets it active)
            if result.get("ok"):
                out_id = result.get("output_volume_id")
                out_name = result.get("output_volume_name")
                if out_id:
                    self.session.update_state(active_volume_id=out_id, active_volume_name=out_name)
            return result
        except Exception as e:
            self._log(tool="compute_subtraction", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Export (bridge)
    # -------------------------

    def export_segmentation_dicom(
        self,
        *,
        segmentation_id: str,
        reference_volume_id: str,
        output_folder: Union[str, Path],
    ) -> Dict[str, Any]:
        output_folder_p = _as_path(output_folder)
        args = {
            "segmentation_id": segmentation_id,
            "reference_volume_id": reference_volume_id,
            "output_folder": str(output_folder_p),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="export_segmentation_dicom",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="export_segmentation_dicom", args=args, result=result, artifacts={"folder": str(output_folder_p)})
            return result
        except Exception as e:
            self._log(tool="export_segmentation_dicom", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Segmentation pack (curated stable subset)
    # -------------------------

    def _apply_segmentation_state_from_result(self, result: Dict[str, Any]) -> None:
        updates: Dict[str, Any] = {}
        if result.get("segmentation_id") is not None:
            updates["active_segmentation_id"] = result.get("segmentation_id")
        if result.get("segmentation_name") is not None:
            updates["active_segmentation_name"] = result.get("segmentation_name")
        if result.get("segment_id") is not None:
            updates["active_segment_id"] = result.get("segment_id")
        if result.get("segment_name") is not None:
            updates["active_segment_name"] = result.get("segment_name")
        if result.get("source_volume_id") is not None:
            updates["active_volume_id"] = result.get("source_volume_id")
        if result.get("source_volume_name") is not None:
            updates["active_volume_name"] = result.get("source_volume_name")
        if updates:
            self.session.update_state(**updates)

    def list_segmentations(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="list_segmentations",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict) and result.get("segmentations"):
                active_seg = result.get("active_segmentation") or {}
                if isinstance(active_seg, dict):
                    self._apply_segmentation_state_from_result(
                        {
                            "segmentation_id": active_seg.get("id"),
                            "segmentation_name": active_seg.get("name"),
                        }
                    )
            self._log(tool="list_segmentations", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_segmentations", args=args, result=None, ok=False, error=str(e))
            raise

    def list_segments(
        self,
        *,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="list_segments",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="list_segments", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_segments", args=args, result=None, ok=False, error=str(e))
            raise

    def image_coords_to_ras(
        self,
        *,
        view: str,
        bbox_1000: Optional[Sequence[int | float]] = None,
        point_1000: Optional[Sequence[int | float]] = None,
        size: int = 512,
    ) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        if (bbox_1000 is None) == (point_1000 is None):
            raise ValueError("Provide exactly one of bbox_1000 or point_1000.")

        norm_bbox: Optional[List[int]] = None
        norm_point: Optional[List[int]] = None
        point_norm: Optional[List[float]] = None
        preview_png_path: Optional[str] = None

        try:
            pre_obs = self.observe_view(view=canonical_view, size=int(size), out_name=f"coords_{canonical_view}.png")
            base_png = str(pre_obs.get("png_path")) if isinstance(pre_obs, dict) and pre_obs.get("png_path") else None
        except Exception:
            base_png = None

        if bbox_1000 is not None:
            norm_bbox = normalize_bbox_1000(bbox_1000)
            cx_norm, cy_norm = bbox_center_norm(norm_bbox)
            point_norm = [float(cx_norm), float(cy_norm)]
            if base_png:
                try:
                    preview_png_path = self._annotate_bbox_preview(
                        png_path=base_png,
                        bbox_1000=norm_bbox,
                        out_name=f"coords_{canonical_view}_bbox_annotated.png",
                        label="bbox1000",
                    )
                except Exception:
                    logger.exception("Failed annotating bbox preview", exc_info=True)
            bridge_args = {"view": canonical_view, "bbox_norm": [float(norm_bbox[0]) / 999.0, float(norm_bbox[1]) / 999.0, float(norm_bbox[2]) / 999.0, float(norm_bbox[3]) / 999.0]}
        else:
            norm_point = normalize_point_1000(point_1000 or [])
            px_norm, py_norm = point_norm_from_1000(norm_point)
            point_norm = [float(px_norm), float(py_norm)]
            if base_png:
                try:
                    preview_png_path = self._annotate_point_preview(
                        png_path=base_png,
                        point_1000=norm_point,
                        out_name=f"coords_{canonical_view}_point_annotated.png",
                        label="point1000",
                    )
                except Exception:
                    logger.exception("Failed annotating point preview", exc_info=True)
            bridge_args = {"view": canonical_view, "point_norm": [float(px_norm), float(py_norm)]}

        args = {
            "view": canonical_view,
            "bbox_1000": norm_bbox,
            "point_1000": norm_point,
            "size": int(size),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="image_coords_to_ras",
                args=bridge_args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                if norm_bbox is not None:
                    result["bbox_1000"] = norm_bbox
                if norm_point is not None:
                    result["point_1000"] = norm_point
                if point_norm is not None:
                    result["point_norm"] = point_norm
                if preview_png_path:
                    result["png_path"] = preview_png_path
            self._log(tool="image_coords_to_ras", args=args, result=result, artifacts={"png": preview_png_path} if preview_png_path else None)
            return result
        except Exception as e:
            self._log(tool="image_coords_to_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_from_threshold(
        self,
        *,
        minimum_threshold: float,
        maximum_threshold: float,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "minimum_threshold": float(minimum_threshold),
            "maximum_threshold": float(maximum_threshold),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_from_threshold",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_from_threshold", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_from_threshold", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_local_threshold(
        self,
        *,
        minimum_threshold: float,
        maximum_threshold: float,
        minimum_diameter_mm: float = 3.0,
        feature_size_mm: float = 3.0,
        segmentation_algorithm: str = "GrowCut",
        seed_view: Optional[str] = None,
        seed_bbox_1000: Optional[Sequence[int | float]] = None,
        seed_x_norm: Optional[float] = None,
        seed_y_norm: Optional[float] = None,
        ras_seed: Optional[Sequence[float]] = None,
        seed_segment_id: Optional[str] = None,
        seed_segment_name: Optional[str] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
        size: int = 512,
    ) -> Dict[str, Any]:
        bbox_norm: Optional[List[int]] = None
        seed_preview_png_path: Optional[str] = None
        if seed_bbox_1000 is None and (seed_x_norm is None or seed_y_norm is None) and ras_seed is None and not (seed_segment_id or seed_segment_name):
            raise ValueError("Provide seed_bbox_1000 + seed_view (preferred), or seed_x_norm/seed_y_norm + seed_view, or ras_seed, or seed_segment_id/seed_segment_name.")
        if seed_bbox_1000 is not None:
            bbox_norm = normalize_bbox_1000(seed_bbox_1000)
            seed_x_norm, seed_y_norm = bbox_center_norm(bbox_norm)
            if seed_view is None:
                raise ValueError("seed_view is required when using seed_bbox_1000")
            try:
                pre_obs = self.observe_view(view=str(seed_view), size=int(size), out_name=f"segment_local_threshold_seed_{_canonical_view(seed_view)}.png")
                if isinstance(pre_obs, dict) and pre_obs.get("png_path"):
                    seed_preview_png_path = self._annotate_bbox_preview(
                        png_path=str(pre_obs["png_path"]),
                        bbox_1000=bbox_norm,
                        out_name=f"segment_local_threshold_seed_{_canonical_view(seed_view)}_annotated.png",
                        label=f"seed {str(seed_view)} bbox1000",
                    )
            except Exception:
                logger.exception("Failed to create local-threshold seed preview", exc_info=True)

        args = {
            "minimum_threshold": float(minimum_threshold),
            "maximum_threshold": float(maximum_threshold),
            "minimum_diameter_mm": float(minimum_diameter_mm),
            "feature_size_mm": float(feature_size_mm),
            "segmentation_algorithm": str(segmentation_algorithm),
            "seed_view": seed_view,
            "seed_bbox_1000": bbox_norm,
            "seed_x_norm": (float(seed_x_norm) if seed_x_norm is not None else None),
            "seed_y_norm": (float(seed_y_norm) if seed_y_norm is not None else None),
            "ras_seed": ([float(v) for v in ras_seed] if ras_seed is not None else None),
            "seed_segment_id": seed_segment_id,
            "seed_segment_name": seed_segment_name,
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
            "size": int(size),
        }
        bridge_args = {k: v for k, v in args.items() if k not in {"seed_bbox_1000", "size"}}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_local_threshold",
                args=bridge_args,
                session_id=self.session.session_id,
            )
            png_paths: Dict[str, str] = {}
            if seed_preview_png_path:
                png_paths["seed_preview"] = seed_preview_png_path
            try:
                for v in _STANDARD_VIEWS:
                    cap = self.capture_slice_view_png(view=v, out_name=f"segment_local_threshold_{v}_overlay.png")
                    if isinstance(cap, dict) and cap.get("png_path"):
                        png_paths[v] = str(cap["png_path"])
                if bbox_norm is not None and seed_view is not None and seed_view in png_paths:
                    annotated_overlay = self._annotate_bbox_preview(
                        png_path=png_paths[seed_view],
                        bbox_1000=bbox_norm,
                        out_name=f"segment_local_threshold_{_canonical_view(seed_view)}_overlay_annotated.png",
                        label="seed center",
                    )
                    png_paths[seed_view] = annotated_overlay
            except Exception:
                logger.exception("Failed to capture local-threshold overlay previews", exc_info=True)

            if isinstance(result, dict):
                result["seed_bbox_1000"] = bbox_norm
                result["seed_center_norm"] = [float(seed_x_norm), float(seed_y_norm)] if seed_x_norm is not None and seed_y_norm is not None else None
                result["seed_preview_png_path"] = seed_preview_png_path
                if png_paths:
                    result["png_paths"] = png_paths
                    preferred_png = None
                    if seed_view is not None and str(seed_view) in png_paths:
                        preferred_png = png_paths[str(seed_view)]
                    elif "axial" in png_paths:
                        preferred_png = png_paths["axial"]
                    else:
                        preferred_png = next(iter(png_paths.values()), None)
                    result["png_path"] = preferred_png
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_local_threshold", args=args, result=result, artifacts={"png": next(iter(png_paths.values()), seed_preview_png_path or "")})
            return result
        except Exception as e:
            self._log(tool="segment_local_threshold", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_edit_sphere(
        self,
        *,
        ras_center: Sequence[float],
        radius_mm: float,
        action: str = "add",
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "ras_center": [float(v) for v in ras_center],
            "radius_mm": float(radius_mm),
            "action": str(action),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_edit_sphere",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_edit_sphere", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_edit_sphere", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_margin(
        self,
        *,
        margin_size_mm: float,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "margin_size_mm": float(margin_size_mm),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_margin",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_margin", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_margin", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_smoothing(
        self,
        *,
        smoothing_method: str = "MEDIAN",
        kernel_size_mm: float = 3.0,
        gaussian_std_mm: Optional[float] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "smoothing_method": str(smoothing_method),
            "kernel_size_mm": float(kernel_size_mm),
            "gaussian_std_mm": (float(gaussian_std_mm) if gaussian_std_mm is not None else None),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_smoothing",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_smoothing", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_smoothing", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_islands(
        self,
        *,
        operation: str = "KEEP_LARGEST_ISLAND",
        minimum_size: int = 1000,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "operation": str(operation),
            "minimum_size": int(minimum_size),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_islands",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_islands", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_islands", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_logical(
        self,
        *,
        operation: str,
        modifier_segment_id: Optional[str] = None,
        modifier_segment_name: Optional[str] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "operation": str(operation),
            "modifier_segment_id": modifier_segment_id,
            "modifier_segment_name": modifier_segment_name,
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_logical",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_logical", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_logical", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_statistics(
        self,
        *,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_statistics",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_statistics", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_statistics", args=args, result=None, ok=False, error=str(e))
            raise

    def center_on_segment(
        self,
        *,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="center_on_segment",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="center_on_segment", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="center_on_segment", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Bookmarks (host-side convenience; no Slicer dependency)
    # -------------------------

    def bookmark_add(
        self,
        *,
        name: str,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
    ) -> Dict[str, Any]:
        """Create a named bookmark: stores viewer_state + a representative slice PNG."""
        canonical_view = _canonical_view(view)
        args = {"name": name, "view": canonical_view, "orientation": orientation, "scroll_to": float(scroll_to), "size": int(size)}
        try:
            st = self.get_viewer_state()
            img = self.get_slice_png(view=canonical_view, orientation=orientation, scroll_to=scroll_to, size=size, out_name=f"bookmark_{name}.png")

            bookmarks = dict(self.session.state.get("bookmarks") or {})
            bookmarks[name] = {"viewer_state": st, "png_path": img.get("png_path")}
            self.session.update_state(bookmarks=bookmarks)

            result = {"name": name, "png_path": img.get("png_path")}
            self._log(tool="bookmark_add", args=args, result=result, artifacts={"png": img.get("png_path", "")})
            return result
        except Exception as e:
            self._log(tool="bookmark_add", args=args, result=None, ok=False, error=str(e))
            raise

    def bookmark_list(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            bookmarks = self.session.state.get("bookmarks") or {}
            result = {"names": sorted(list(bookmarks.keys()))}
            self._log(tool="bookmark_list", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="bookmark_list", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # GUI / layout (rarely needed)
    # -------------------------

    def set_gui(self, *, contents: Optional[str] = None, viewers_layout: Optional[str] = None) -> Dict[str, Any]:
        args = {"contents": contents, "viewers_layout": viewers_layout}
        try:
            result = self.client.put_gui(contents=contents, viewers_layout=viewers_layout)
            self._log(tool="set_gui", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_gui", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # MRML endpoints (raw WebServer)
    # -------------------------

    def mrml_list_nodes(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_list_nodes(classname=classname)
            self._log(tool="mrml_list_nodes", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_list_nodes", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_names(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_names(classname=classname)
            self._log(tool="mrml_names", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_names", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_ids(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_ids(classname=classname)
            self._log(tool="mrml_ids", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_ids", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_properties(self, *, node_id: str) -> Any:
        args = {"node_id": node_id}
        try:
            result = self.client.mrml_properties(node_id=node_id)
            self._log(tool="mrml_properties", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_properties", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_set_properties(self, *, node_id: str, properties: Dict[str, Any]) -> Any:
        args = {"node_id": node_id, "properties": properties}
        try:
            result = self.client.mrml_set_properties(node_id=node_id, properties=properties)
            self._log(tool="mrml_set_properties", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_set_properties", args=args, result=None, ok=False, error=str(e))
            raise


    def mrml_file_get(self, *, node_id: str, file_name: str) -> Any:
        args = {"node_id": node_id, "file_name": file_name}
        try:
            result = self.client.mrml_file_get(node_id=node_id, file_name=file_name)
            self._log(tool="mrml_file_get", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_get", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_file_post(self, *, file_name: str) -> Any:
        args = {"file_name": file_name}
        try:
            result = self.client.mrml_file_post(file_name=file_name)
            self._log(tool="mrml_file_post", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_post", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_file_delete(self, *, node_id: Optional[str] = None, file_name: Optional[str] = None) -> Any:
        args = {"node_id": node_id, "file_name": file_name}
        try:
            result = self.client.mrml_file_delete(node_id=node_id, file_name=file_name)
            self._log(tool="mrml_file_delete", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_delete", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Volume selection / transforms
    # -------------------------

    def volume_selection(self, *, cmd: str) -> Any:
        args = {"cmd": cmd}
        try:
            result = self.client.volume_selection(cmd=cmd)
            self._log(tool="volume_selection", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="volume_selection", args=args, result=None, ok=False, error=str(e))
            raise

    def list_gridtransforms(self) -> Any:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_gridtransforms()
            self._log(tool="list_gridtransforms", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_gridtransforms", args=args, result=None, ok=False, error=str(e))
            raise

    def download_volume_nrrd(self, *, node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"node_id": node_id, "out_name": out_name}
        try:
            if out_name is None:
                out_name = f"volume_{node_id}.nrrd"
            out_path = self._artifact_path(out_name)
            self.client.download_volume_nrrd(node_id=node_id, out_path=out_path)
            result = {"nrrd_path": str(out_path)}
            self._log(tool="download_volume_nrrd", args=args, result=result, artifacts={"nrrd": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="download_volume_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    def upload_volume_nrrd(self, *, nrrd_path: Union[str, Path]) -> Any:
        nrrd_path_p = _as_path(nrrd_path)
        args = {"nrrd_path": str(nrrd_path_p)}
        try:
            result = self.client.upload_volume_nrrd(nrrd_path=nrrd_path_p)
            self._log(tool="upload_volume_nrrd", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="upload_volume_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    def download_gridtransform_nrrd(self, *, node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"node_id": node_id, "out_name": out_name}
        try:
            if out_name is None:
                out_name = f"gridtransform_{node_id}.nrrd"
            out_path = self._artifact_path(out_name)
            self.client.download_gridtransform_nrrd(node_id=node_id, out_path=out_path)
            result = {"nrrd_path": str(out_path)}
            self._log(tool="download_gridtransform_nrrd", args=args, result=result, artifacts={"nrrd": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="download_gridtransform_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Fiducials
    # -------------------------

    def list_fiducials(self) -> Any:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_fiducials()
            self._log(tool="list_fiducials", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_fiducials", args=args, result=None, ok=False, error=str(e))
            raise

    def set_fiducial_ras(self, *, node_id: str, r: float, a: float, s: float) -> Any:
        args = {"node_id": node_id, "r": r, "a": a, "s": s}
        try:
            result = self.client.put_fiducial(node_id=node_id, r=r, a=a, s=s)
            self._log(tool="set_fiducial_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_fiducial_ras", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Tracking
    # -------------------------

    def get_tracking(self, *, m: Optional[str] = None, q: Optional[str] = None, p: Optional[str] = None) -> Any:
        args = {"m": m, "q": q, "p": p}
        try:
            result = self.client.get_tracking(m=m, q=q, p=p)
            self._log(tool="get_tracking", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_tracking", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Sample data
    # -------------------------

    def load_sampledata(self, *, name: str) -> Any:
        args = {"name": name}
        try:
            result = self.client.load_sampledata(name=name)
            self._log(tool="load_sampledata", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_sampledata", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # DICOMweb (Slicer endpoint)
    # -------------------------

    def access_dicomweb_study(self, payload: Dict[str, Any]) -> Any:
        args = {"payload": payload}
        try:
            result = self.client.access_dicomweb_study(payload)
            self._log(tool="access_dicomweb_study", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="access_dicomweb_study", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Raw requests for completeness
    # -------------------------

    def raw_slicer_request(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data_text: Optional[str] = None,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generic HTTP call to the `/slicer` subtree.

        Use this to access newly-added WebServer endpoints without changing this repo.
        """
        args = {"method": method, "path": path, "params": params, "json_body": json_body, "save_as": save_as}
        try:
            data: Optional[bytes] = None
            if data_text is not None:
                data = data_text.encode("utf-8")
            resp = self.client.raw_request(
                method,
                path,
                params=params,
                json_body=json_body,
                data=data,
                stream=bool(save_as),
            )
            headers = dict(resp.headers)
            ct = headers.get("Content-Type", "")

            if save_as:
                out_path = self._artifact_path(save_as)
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                result = {
                    "status_code": resp.status_code,
                    "content_type": ct,
                    "headers": headers,
                    "saved_path": str(out_path),
                }
                self._log(tool="raw_slicer_request", args=args, result=result, artifacts={"file": str(out_path)})
                return result

            # Try JSON first
            try:
                payload = resp.json()
                result = {"status_code": resp.status_code, "content_type": ct, "headers": headers, "json": payload}
            except Exception:
                result = {"status_code": resp.status_code, "content_type": ct, "headers": headers, "text": resp.text}

            self._log(tool="raw_slicer_request", args=args, result=result)
            return result

        except Exception as e:
            self._log(tool="raw_slicer_request", args=args, result=None, ok=False, error=str(e))
            raise

    def raw_dicomweb_request(
        self,
        *,
        method: str,
        dicom_path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data_text: Optional[str] = None,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generic HTTP call to the `/dicom` subtree (DICOMweb).

        Slicer can expose its internal DICOM database as DICOMweb.
        If you did not enable DICOMweb in WebServer, this will 404.
        """
        args = {
            "method": method,
            "dicom_path": dicom_path,
            "params": params,
            "headers": headers,
            "save_as": save_as,
        }
        try:
            data: Optional[bytes] = None
            if data_text is not None:
                data = data_text.encode("utf-8")
            resp = self.client.dicomweb_request(
                method,
                dicom_path,
                params=params,
                headers=headers,
                data=data,
                stream=bool(save_as),
            )
            resp_headers = dict(resp.headers)
            ct = resp_headers.get("Content-Type", "")

            if save_as:
                out_path = self._artifact_path(save_as)
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                result = {
                    "status_code": resp.status_code,
                    "content_type": ct,
                    "headers": resp_headers,
                    "saved_path": str(out_path),
                }
                self._log(tool="raw_dicomweb_request", args=args, result=result, artifacts={"file": str(out_path)})
                return result

            try:
                payload = resp.json()
                result = {"status_code": resp.status_code, "content_type": ct, "headers": resp_headers, "json": payload}
            except Exception:
                result = {"status_code": resp.status_code, "content_type": ct, "headers": resp_headers, "text": resp.text}

            self._log(tool="raw_dicomweb_request", args=args, result=result)
            return result

        except Exception as e:
            self._log(tool="raw_dicomweb_request", args=args, result=None, ok=False, error=str(e))
            raise



    # -------------------------
    # Viewer-friendly macros (agent UX helpers)
    # -------------------------

    def observe_view(self, *, view: str = "axial", size: int = 512, out_name: Optional[str] = None) -> Dict[str, Any]:
        """Capture the *current* slice view as PNG.

        Unlike `get_slice_png(scroll_to=...)`, this tries to preserve the current interactive state
        (slice offset + current orientation) by querying `get_viewer_state()` and rendering at that offset.

        This is the building block for "auto" viewer mode: after actions like zoom/WL/scroll, you want
        an updated observation without forcing the agent to call `get_slice_png` manually.
        """
        canonical_view = _canonical_view(view)
        args = {"view": canonical_view, "size": int(size), "out_name": out_name}
        try:
            st = self.get_viewer_state()
            v = canonical_view
            vs = (st.get("views") or {}).get(v) or {}
            orientation_raw = str(vs.get("orientation") or "axial")
            orientation = orientation_raw.strip().lower()
            # Slicer may return "Axial"/"Sagittal"/"Coronal".
            if "ax" in orientation:
                orientation = "axial"
            elif "sag" in orientation:
                orientation = "sagittal"
            elif "cor" in orientation:
                orientation = "coronal"
            else:
                orientation = "axial"

            offset = vs.get("slice_offset")
            if offset is None:
                # Fall back to a reasonable mid-slice render.
                return self.get_slice_png(view=v, orientation=orientation, scroll_to=0.5, size=size, out_name=out_name)

            if out_name is None:
                out_name = f"obs_{v}_{orientation}_{float(offset):.3f}_{int(size)}.png"
            out_path = self._artifact_path(out_name)

            self.client.save_slice_png(
                out_path,
                view=v,
                orientation=orientation,
                scroll_to=0.5,
                size=size,
                offset=float(offset),
            )

            result = {
                "ok": True,
                "png_path": str(out_path),
                "view": v,
                "orientation": orientation,
                "offset": float(offset),
            }
            self._log(tool="observe_view", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="observe_view", args=args, result=None, ok=False, error=str(e))
            raise

    def observe_standard_views(self, *, size: int = 512, out_prefix: str = "std") -> Dict[str, Any]:
        """Capture a standard 3-plane observation (axial/sagittal/coronal).

        Returns PNG paths for the current slice offsets.
        """
        args = {"size": int(size), "out_prefix": out_prefix}
        try:
            imgs: Dict[str, str] = {}
            for v in _STANDARD_VIEWS:
                res = self.observe_view(view=v, size=size, out_name=f"{out_prefix}_{v}.png")
                if isinstance(res, dict) and res.get("png_path"):
                    imgs[v] = res["png_path"]
            result = {"ok": True, "png_paths": imgs}
            self._log(tool="observe_standard_views", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="observe_standard_views", args=args, result=None, ok=False, error=str(e))
            raise

    def open_case(
        self,
        dicom_dir: Union[str, Path],
        *,
        preset: str = "mri_brain",
        layout: str = "four_up",
        linked_slices: bool = True,
        interpolation: bool = True,
        wl_auto: bool = True,
        baseline: bool = True,
        size: int = 512,
        clear_scene_first: bool = True,
        nifti_files: Optional[Sequence[Union[str, Path]]] = None,
        include_segmentations: bool = False,
        dicom_series_dirs: Optional[Sequence[Union[str, Path]]] = None,
        series_display_names: Optional[Sequence[str]] = None,
        active_prefer: Optional[Sequence[str]] = None,
        window_preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Open a case (DICOM folder/series or NIfTI file/folder) with a sensible default preset."""

        case_path = _as_path(dicom_dir)
        args = {
            "case_path": str(case_path),
            "preset": preset,
            "layout": layout,
            "linked_slices": bool(linked_slices),
            "interpolation": bool(interpolation),
            "wl_auto": bool(wl_auto),
            "baseline": bool(baseline),
            "size": int(size),
            "clear_scene_first": bool(clear_scene_first),
            "nifti_files": [str(_as_path(x)) for x in nifti_files] if nifti_files else None,
            "include_segmentations": bool(include_segmentations),
            "dicom_series_dirs": [str(_as_path(x)) for x in dicom_series_dirs] if dicom_series_dirs else None,
            "series_display_names": [str(x) for x in series_display_names] if series_display_names else None,
            "active_prefer": list(active_prefer) if active_prefer else None,
            "window_preset": window_preset,
        }

        summary: Dict[str, Any] = {"ok": True, "steps": {}, "baseline": None, "warnings": []}
        fatal_steps = {"load_dicom", "load_dicom_series", "load_nifti"}

        def _step(name: str, fn):
            try:
                r = fn()
                summary["steps"][name] = r
                if isinstance(r, dict) and r.get("ok") is False and name in fatal_steps:
                    summary["ok"] = False
                elif isinstance(r, dict) and r.get("ok") is False:
                    summary["warnings"].append({"step": name, "error": r.get("error") or r})
                return r
            except Exception as e:
                if name in fatal_steps:
                    summary["ok"] = False
                summary["steps"][name] = {"ok": False, "error": str(e)}
                summary["warnings"].append({"step": name, "error": str(e)})
                return None

        if clear_scene_first:
            _step("clear_scene", lambda: self.clear_scene())

        def _looks_like_nifti_dir(p: Path) -> bool:
            if not p.exists():
                return False
            if p.is_file():
                pl = str(p).lower()
                return pl.endswith(".nii") or pl.endswith(".nii.gz")
            if p.is_dir():
                for c in p.iterdir():
                    cl = str(c).lower()
                    if c.is_file() and (cl.endswith(".nii") or cl.endswith(".nii.gz")):
                        return True
            return False

        def _discover_nifti_files(dir_path: Path) -> List[Path]:
            all_files = sorted([p for p in dir_path.iterdir() if p.is_file() and str(p).lower().endswith((".nii", ".nii.gz"))])
            if not include_segmentations:
                all_files = [p for p in all_files if "segmentation" not in p.name.lower()]

            if preset != "mri_brain":
                return all_files

            def pick(modality: str, *, prefer_bias: bool = True) -> Optional[Path]:
                mod_l = modality.lower()
                cands = [p for p in all_files if mod_l in p.name.lower()]
                if modality.lower() == "t1":
                    cands = [p for p in cands if "t1c" not in p.name.lower()]
                if not cands:
                    return None
                if prefer_bias:
                    bias = [p for p in cands if "bias" in p.name.lower()]
                    if bias:
                        return sorted(bias)[0]
                return sorted(cands)[0]

            chosen: List[Path] = []
            for mod in ("t1c", "flair", "t2", "t1"):
                p = pick(mod)
                if p is not None:
                    chosen.append(p)
            return chosen if chosen else all_files

        load_res = None
        if dicom_series_dirs is not None:
            ds = [_as_path(x) for x in dicom_series_dirs]
            load_res = _step(
                "load_dicom_series",
                lambda: self.load_dicom_series(
                    [str(p) for p in ds],
                    clear_scene_first=False,
                    active_prefer=active_prefer,
                ),
            )
        elif nifti_files is not None:
            nf = [_as_path(x) for x in nifti_files]
            load_res = _step(
                "load_nifti",
                lambda: self.load_nifti(
                    [str(p) for p in nf],
                    clear_scene_first=False,
                    active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                ),
            )
        elif _looks_like_nifti_dir(case_path):
            if case_path.is_file():
                load_res = _step(
                    "load_nifti",
                    lambda: self.load_nifti(
                        str(case_path),
                        clear_scene_first=False,
                        active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                    ),
                )
            else:
                discovered = _discover_nifti_files(case_path)
                load_res = _step(
                    "load_nifti",
                    lambda: self.load_nifti(
                        [str(p) for p in discovered],
                        clear_scene_first=False,
                        active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                    ),
                )
        else:
            load_res = _step(
                "load_dicom",
                lambda: self.load_dicom(case_path, clear_scene_first=False, active_prefer=active_prefer),
            )

        _step("set_layout", lambda: self.set_layout(layout=layout))
        _step("set_linked_slices", lambda: self.set_linked_slices(enabled=linked_slices))
        _step("set_interpolation", lambda: self.set_interpolation(interpolate=interpolation))

        if window_preset:
            _step("apply_window_preset", lambda: self.apply_window_preset(preset=window_preset))
        elif wl_auto:
            _step("set_window_level_auto", lambda: self.set_window_level(auto=True))

        for v in _STANDARD_VIEWS:
            _step(f"fit_{v}", lambda v=v: self.fit_slice(view=v))

        if isinstance(load_res, dict) and load_res.get("ok"):
            self.session.update_state(case_preset=preset)

        if baseline:
            try:
                b = {
                    "axial": self.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=size, out_name="baseline_axial.png")["png_path"],
                    "sagittal": self.get_slice_png(view="sagittal", orientation="sagittal", scroll_to=0.5, size=size, out_name="baseline_sagittal.png")["png_path"],
                    "coronal": self.get_slice_png(view="coronal", orientation="coronal", scroll_to=0.5, size=size, out_name="baseline_coronal.png")["png_path"],
                }
                summary["baseline"] = b
            except Exception as e:
                summary["ok"] = False
                summary["baseline"] = {"ok": False, "error": str(e)}

        try:
            summary["viewer_state"] = self.get_viewer_state()
        except Exception as e:
            summary["ok"] = False
            summary["viewer_state"] = {"ok": False, "error": str(e)}

        try:
            scene_inventory = self.client.list_volumes()
            summary["scene_inventory"] = scene_inventory
            summary["scene_inventory_text"] = format_scene_inventory_for_prompt(
                scene_inventory,
                active_volume_id=self.session.state.get("active_volume_id"),
                active_volume_name=self.session.state.get("active_volume_name"),
            )
            self.session.update_state(
                scene_inventory=scene_inventory,
                scene_inventory_text=summary["scene_inventory_text"],
                loaded_series_dirs=args.get("dicom_series_dirs"),
                loaded_series_display_names=args.get("series_display_names"),
            )
        except Exception as e:
            summary["warnings"].append({"step": "scene_inventory", "error": str(e)})
            summary["scene_inventory"] = []
            summary["scene_inventory_text"] = f"Currently loaded scene volumes: unavailable ({e})"

        if args.get("dicom_series_dirs") is not None or args.get("series_display_names") is not None:
            self.session.update_state(
                loaded_series_dirs=args.get("dicom_series_dirs"),
                loaded_series_display_names=args.get("series_display_names"),
            )
        self._log(tool="open_case", args=args, result=summary)
        return summary

    def scroll_sweep(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        start: float = 0.0,
        end: float = 1.0,
        n_frames: int = 12,
        size: int = 512,
        output: str = "keyframes",
        fps: int = 12,
        analyze: bool = False,
        prompt: Optional[str] = None,
        auto_prompt: bool = True,
        out_prefix: str = "scroll",
    ) -> Dict[str, Any]:
        """Scroll/sweep through a stack and return observations.

        Keyframe contact sheets are annotated with the physical S/A/L position so the model can
        jump back to a specific slice later using set_slice_offset.
        """

        args = {
            "view": view,
            "orientation": orientation,
            "start": float(start),
            "end": float(end),
            "n_frames": int(n_frames),
            "size": int(size),
            "output": output,
            "fps": int(fps),
            "analyze": bool(analyze),
            "prompt": prompt,
            "auto_prompt": bool(auto_prompt),
            "out_prefix": out_prefix,
        }

        try:
            v = _canonical_view(view)
            ori = (orientation or "axial").lower()
            mode = (output or "keyframes").lower()

            if mode not in {"keyframes", "frames", "video"}:
                raise ValueError("output must be one of: keyframes, video")
            if n_frames < 2:
                raise ValueError("n_frames must be >= 2")

            axis_label = {"axial": "S", "sagittal": "L", "coronal": "A"}.get(v, "mm")
            offset_range = None
            try:
                offset_range = self.get_slice_offset_range(view=v)
            except Exception:
                offset_range = None

            if mode in {"keyframes", "frames"}:
                frames_dir = self._artifact_path(f"{out_prefix}_{v}_{ori}_{n_frames}f")
                frames_dir.mkdir(parents=True, exist_ok=True)
                png_paths: List[str] = []
                ts: List[float] = []
                frame_offsets_mm: List[Optional[float]] = []
                lo = hi = None
                if isinstance(offset_range, dict):
                    lo = offset_range.get("min_offset")
                    hi = offset_range.get("max_offset")
                for i in range(n_frames):
                    t = float(start + (end - start) * (i / (n_frames - 1)))
                    ts.append(t)
                    out_path = frames_dir / f"frame_{i:05d}.png"
                    self.client.save_slice_png(out_path, view=v, orientation=ori, scroll_to=t, size=size)
                    if lo is not None and hi is not None:
                        off = float(lo) + t * (float(hi) - float(lo))
                        frame_offsets_mm.append(off)
                        try:
                            from PIL import Image, ImageDraw, ImageFont  # type: ignore

                            with Image.open(out_path) as im:
                                img = im.convert("RGB")
                                draw = ImageDraw.Draw(img)
                                font = ImageFont.load_default()
                                label = f"{axis_label}: {off:.4f}mm"
                                l, t0, r, b = draw.textbbox((0, 0), label, font=font)
                                pad = 4
                                draw.rectangle([0, 0, (r - l) + pad * 2, (b - t0) + pad * 2], fill=(0, 0, 0))
                                draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
                                img.save(out_path)
                        except Exception:
                            logger.exception("Failed annotating scroll keyframe", exc_info=True)
                    else:
                        frame_offsets_mm.append(None)
                    png_paths.append(str(out_path))

                contact_sheet_path: Optional[Path] = None
                try:
                    from PIL import Image  # type: ignore

                    imgs = [Image.open(p) for p in png_paths]
                    w, h = imgs[0].size
                    n_cols = 4
                    n_rows = (len(imgs) + n_cols - 1) // n_cols
                    sheet = Image.new("RGB", (w * n_cols, h * n_rows))
                    for idx, im in enumerate(imgs):
                        r = idx // n_cols
                        c = idx % n_cols
                        sheet.paste(im.convert("RGB"), (c * w, r * h))
                    contact_sheet_path = self._artifact_path(f"{out_prefix}_{v}_{ori}_{n_frames}f_contact.png")
                    sheet.save(contact_sheet_path)
                except Exception:
                    contact_sheet_path = None

                result = {
                    "ok": True,
                    "mode": "keyframes",
                    "frames_dir": str(frames_dir),
                    "png_paths": png_paths,
                    "t_values": ts,
                    "frame_offsets_mm": frame_offsets_mm,
                    "offset_axis": axis_label,
                    "offset_range": offset_range,
                    "contact_sheet_png_path": str(contact_sheet_path) if contact_sheet_path else None,
                }
                self._log(tool="scroll_sweep", args=args, result=result, artifacts={"frames_dir": str(frames_dir)})
                return result

            if self.video is None:
                raise RuntimeError("VideoRenderer is not configured (ffmpeg not available)")

            cine = self.capture_cine(
                view=v,
                orientation=ori,
                start=float(start),
                end=float(end),
                n_frames=int(n_frames),
                size=int(size),
                fps=int(fps),
                frames_dir_name=f"{out_prefix}_{v}_{ori}_{n_frames}f_frames",
                out_mp4_name=f"{out_prefix}_{v}_{ori}_{n_frames}f_{fps}fps.mp4",
            )
            result: Dict[str, Any] = {"ok": True, "mode": "video", **cine}

            if analyze:
                if self.gemini is None:
                    raise RuntimeError("Gemini analyzer is not configured (set GEMINI_API_KEY).")
                if prompt is None:
                    if not auto_prompt:
                        raise ValueError("prompt must be provided when analyze=True and auto_prompt=False")
                    prompt = (
                        "You are reading a radiology scroll cine. Summarize where the most suspicious lesion appears, "
                        "which frames or slice positions are most relevant, and any obvious artifacts or pitfalls. Keep it concise."
                    )
                text = self.gemini.analyze_mp4(video_path=result["mp4_path"], prompt=prompt)
                result["analysis_text"] = text
                result["analysis_model"] = getattr(self.gemini, "model", None)

            self._log(tool="scroll_sweep", args=args, result=result, artifacts={"mp4": str(result.get("mp4_path", ""))})
            return result
        except Exception as e:
            self._log(tool="scroll_sweep", args=args, result=None, ok=False, error=str(e))
            raise
    # -------------------------
    # Exec escape hatches (dangerous)
    # -------------------------

    def exec_python(self, *, source: str) -> Dict[str, Any]:
        """Run raw Python in Slicer via `/slicer/exec`.

        **Unsafe**: do not expose to untrusted callers. Prefer the fixed-entry bridge tools.
        """
        args = {"source": source}
        try:
            result = self.client.exec_python(source)
            self._log(tool="exec_python", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="exec_python", args=args, result=None, ok=False, error=str(e))
            raise

    def exec_bridge(self, *, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call any bridge tool by name.

        This is a forward-compatibility hook: you can add new bridge tools on the Slicer side
        without updating the MCP schema immediately.
        """
        call_args = {"tool": tool, "args": args}
        try:
            result = self.client.exec_bridge(bridge_dir=self.bridge_dir, tool=tool, args=args, session_id=self.session.session_id)
            self._log(tool="exec_bridge", args=call_args, result=result)
            return result
        except Exception as e:
            self._log(tool="exec_bridge", args=call_args, result=None, ok=False, error=str(e))
            raise
