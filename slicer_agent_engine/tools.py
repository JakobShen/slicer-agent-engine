from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .slicer_client import SlicerClient
from .session import SessionManager
from .video_renderer import VideoRenderer

logger = logging.getLogger(__name__)


def _as_path(p: Union[str, Path]) -> Path:
    return Path(p).expanduser().resolve()


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
        self.session.log_event(tool=tool, args=args, result=result, ok=ok, error=error, artifacts=artifacts)

    def _artifact_path(self, name: str) -> Path:
        p = self.session.out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

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
    # DICOM loading / volume selection (bridge)
    # -------------------------

    def load_dicom(self, dicom_dir: Union[str, Path], *, clear_scene_first: bool = True) -> Dict[str, Any]:
        dicom_dir_p = _as_path(dicom_dir)
        args = {"dicom_dir": str(dicom_dir_p), "clear_scene_first": bool(clear_scene_first)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_dicom",
                args=args,
                session_id=self.session.session_id,
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

    def set_linked_slices(self, *, enabled: bool) -> Dict[str, Any]:
        args = {"enabled": bool(enabled)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_linked_slices",
                args=args,
                session_id=self.session.session_id,
            )
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
        view: str = "red",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
        out_name: Optional[str] = None,
        offset: Optional[float] = None,
        copy_slice_geometry_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "view": view,
            "orientation": orientation,
            "scroll_to": float(scroll_to),
            "size": int(size),
            "offset": offset,
            "copy_slice_geometry_from": copy_slice_geometry_from,
        }
        try:
            if out_name is None:
                out_name = f"slice_{view}_{orientation}_{scroll_to:.3f}_{int(size)}.png"
            out_path = self._artifact_path(out_name)
            self.client.save_slice_png(
                out_path,
                view=view,
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
        view: str = "red",
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

        args = {
            "view": view,
            "orientation": orientation,
            "start": float(start),
            "end": float(end),
            "n_frames": int(n_frames),
            "size": int(size),
            "fps": int(fps),
        }

        try:
            if frames_dir_name is None:
                frames_dir_name = f"frames_{view}_{orientation}_{n_frames}f"
            frames_dir = self._artifact_path(frames_dir_name)
            frames_dir.mkdir(parents=True, exist_ok=True)

            if out_mp4_name is None:
                out_mp4_name = f"cine_{view}_{orientation}_{n_frames}f_{fps}fps.mp4"
            out_mp4 = self._artifact_path(out_mp4_name)

            if n_frames <= 1:
                raise ValueError("n_frames must be >= 2")

            for i in range(n_frames):
                t = start + (end - start) * (i / (n_frames - 1))
                frame_path = frames_dir / f"frame_{i:05d}.png"
                self.client.save_slice_png(frame_path, view=view, orientation=orientation, scroll_to=float(t), size=size)

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
            self._log(tool="set_thick_slab", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_thick_slab", args=args, result=None, ok=False, error=str(e))
            raise

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
    # Bookmarks (host-side convenience; no Slicer dependency)
    # -------------------------

    def bookmark_add(
        self,
        *,
        name: str,
        view: str = "red",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
    ) -> Dict[str, Any]:
        """Create a named bookmark: stores viewer_state + a representative slice PNG."""
        args = {"name": name, "view": view, "orientation": orientation, "scroll_to": float(scroll_to), "size": int(size)}
        try:
            st = self.get_viewer_state()
            img = self.get_slice_png(view=view, orientation=orientation, scroll_to=scroll_to, size=size, out_name=f"bookmark_{name}.png")

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
