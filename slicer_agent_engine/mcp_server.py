from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mcp.server.fastmcp import FastMCP

from .session import SessionManager
from .slicer_client import SlicerClient
from .tools import ToolContext
from .video_renderer import VideoRenderer


logger = logging.getLogger(__name__)

mcp = FastMCP("slicer-agent-engine")

_CTX: Optional[ToolContext] = None


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("ToolContext not initialized. Call main().")
    return _CTX


# -------------------------
# Core tools (high-level)
# -------------------------


@mcp.tool()
def ping() -> Dict[str, Any]:
    """Ping Slicer WebServer (GET /slicer/system/version)."""

    return _ctx().ping()


@mcp.tool()
def shutdown_slicer() -> Dict[str, Any]:
    """Ask Slicer to exit (DELETE /slicer/system)."""

    return _ctx().shutdown_slicer()


@mcp.tool()
def clear_scene() -> Dict[str, Any]:
    """Clear Slicer's MRML scene."""

    return _ctx().clear_scene()


@mcp.tool()
def load_dicom(dicom_dir: str, clear_scene_first: bool = True) -> Dict[str, Any]:
    """Load a DICOM folder into Slicer using the fixed-entry bridge."""

    return _ctx().load_dicom(dicom_dir, clear_scene_first=clear_scene_first)


@mcp.tool()
def list_volumes() -> Any:
    """List volumes in the current scene (GET /slicer/volumes)."""

    return _ctx().list_volumes()


@mcp.tool()
def select_volume(volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
    """Select the active volume by ID or name (bridge)."""

    return _ctx().select_volume(volume_id=volume_id, volume_name=volume_name)




# -------------------------
# L0: Viewer actuation (bridge-backed)
# -------------------------


@mcp.tool()
def get_viewer_state() -> Dict[str, Any]:
    """Return the current viewer state snapshot (active volume, per-view slice state, window/level, etc.)."""

    return _ctx().get_viewer_state()


@mcp.tool()
def set_window_level(
    window: Optional[float] = None,
    level: Optional[float] = None,
    auto: Optional[bool] = None,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Set (or auto) window/level for a volume (bridge)."""

    return _ctx().set_window_level(window=window, level=level, auto=auto, volume_id=volume_id, volume_name=volume_name)


@mcp.tool()
def set_interpolation(
    interpolate: bool,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Toggle interpolation for a volume (bridge)."""

    return _ctx().set_interpolation(interpolate=interpolate, volume_id=volume_id, volume_name=volume_name)


@mcp.tool()
def set_slice_orientation(view: str, orientation: str) -> Dict[str, Any]:
    """Set a slice view's orientation (axial/sagittal/coronal or custom) (bridge)."""

    return _ctx().set_slice_orientation(view=view, orientation=orientation)


@mcp.tool()
def get_slice_offset_range(view: str) -> Dict[str, Any]:
    """Get min/max slice offsets for the current volume in a given view (bridge)."""

    return _ctx().get_slice_offset_range(view=view)


@mcp.tool()
def set_slice_offset(view: str, offset: float) -> Dict[str, Any]:
    """Set slice offset (mm) for a given view (bridge)."""

    return _ctx().set_slice_offset(view=view, offset=offset)


@mcp.tool()
def set_slice_scroll_to(view: str, scroll_to: float) -> Dict[str, Any]:
    """Set slice position using normalized [0,1] scroll_to for a given view (bridge)."""

    return _ctx().set_slice_scroll_to(view=view, scroll_to=scroll_to)


@mcp.tool()
def fit_slice(view: str) -> Dict[str, Any]:
    """Fit slice view to show the full volume (bridge)."""

    return _ctx().fit_slice(view=view)


@mcp.tool()
def zoom_slice_relative(view: str, factor: float) -> Dict[str, Any]:
    """Zoom slice view relative by a factor (>1 zooms in, <1 zooms out) (bridge)."""

    return _ctx().zoom_slice_relative(view=view, factor=factor)


@mcp.tool()
def set_field_of_view(view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
    """Set slice view field-of-view directly (bridge)."""

    return _ctx().set_field_of_view(view=view, field_of_view=field_of_view)


@mcp.tool()
def set_xyz_origin(view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
    """Set slice view XYZ origin directly (bridge)."""

    return _ctx().set_xyz_origin(view=view, xyz_origin=xyz_origin)


@mcp.tool()
def jump_to_ras(ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
    """Jump (and optionally center) slices to a RAS point in mm (bridge)."""

    return _ctx().jump_to_ras(ras=ras, centered=centered)


@mcp.tool()
def set_linked_slices(enabled: bool) -> Dict[str, Any]:
    """Enable/disable linked slice controllers (bridge)."""

    return _ctx().set_linked_slices(enabled=enabled)


@mcp.tool()
def set_layout(layout: str) -> Dict[str, Any]:
    """Set a standard layout (four_up, three_up, one_up_red, etc.) (bridge)."""

    return _ctx().set_layout(layout=layout)


# -------------------------
# L1: Deterministic quantification / processing (bridge-backed)
# -------------------------


@mcp.tool()
def roi_stats_ras_box(
    ras_min: Sequence[float],
    ras_max: Sequence[float],
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute ROI stats for a RAS-aligned box (implemented as IJK bounding box of corners)."""

    return _ctx().roi_stats_ras_box(ras_min=ras_min, ras_max=ras_max, volume_id=volume_id, volume_name=volume_name)


@mcp.tool()
def sample_intensity_ras(
    ras: Sequence[float],
    method: str = "nearest",
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Sample intensity at a RAS point (nearest or trilinear)."""

    return _ctx().sample_intensity_ras(ras=ras, method=method, volume_id=volume_id, volume_name=volume_name)


@mcp.tool()
def measure_distance_ras(p1: Sequence[float], p2: Sequence[float]) -> Dict[str, Any]:
    """Measure Euclidean distance (mm) between two RAS points."""

    return _ctx().measure_distance_ras(p1=p1, p2=p2)


@mcp.tool()
def measure_angle_ras(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> Dict[str, Any]:
    """Measure angle (deg) at vertex p2 between (p1-p2) and (p3-p2)."""

    return _ctx().measure_angle_ras(p1=p1, p2=p2, p3=p3)


@mcp.tool()
def measure_area_polygon_ras(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Measure planar polygon area (mm^2) from a list of RAS points (assumes approximately planar)."""

    return _ctx().measure_area_polygon_ras(points=points)


@mcp.tool()
def set_thick_slab(view: str, enabled: bool, thickness_mm: float = 0.0, mode: str = "mip") -> Dict[str, Any]:
    """Enable thick slab reconstruction on a slice view (MIP/min/mean)."""

    return _ctx().set_thick_slab(view=view, enabled=enabled, thickness_mm=thickness_mm, mode=mode)


@mcp.tool()
def set_fusion(background_volume_id: str, foreground_volume_id: str, opacity: float = 0.5) -> Dict[str, Any]:
    """Set simple slice fusion (background + foreground + opacity)."""

    return _ctx().set_fusion(background_volume_id=background_volume_id, foreground_volume_id=foreground_volume_id, opacity=opacity)


@mcp.tool()
def compute_subtraction(volume_a_id: str, volume_b_id: str, output_name: str = "Subtraction") -> Dict[str, Any]:
    """Compute voxel-wise subtraction A - B into a new volume (bridge)."""

    return _ctx().compute_subtraction(volume_a_id=volume_a_id, volume_b_id=volume_b_id, output_name=output_name)


# -------------------------
# Bookmarks (host-side convenience)
# -------------------------


@mcp.tool()
def bookmark_add(
    name: str,
    view: str = "red",
    orientation: str = "axial",
    scroll_to: float = 0.5,
    size: int = 512,
) -> Dict[str, Any]:
    """Create a bookmark (viewer_state + representative slice PNG)."""

    return _ctx().bookmark_add(name=name, view=view, orientation=orientation, scroll_to=scroll_to, size=size)


@mcp.tool()
def bookmark_list() -> Dict[str, Any]:
    """List bookmark names stored in the current session state."""

    return _ctx().bookmark_list()

@mcp.tool()
def get_slice_png(
    view: str = "red",
    orientation: str = "axial",
    scroll_to: float = 0.5,
    size: int = 512,
    out_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a slice view to PNG via `/slicer/slice`. Returns a local png_path."""

    return _ctx().get_slice_png(view=view, orientation=orientation, scroll_to=scroll_to, size=size, out_name=out_name)


@mcp.tool()
def capture_cine(
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
    """Capture a cine loop as MP4 (renders frames via /slicer/slice and encodes with ffmpeg)."""

    return _ctx().capture_cine(
        view=view,
        orientation=orientation,
        start=start,
        end=end,
        n_frames=n_frames,
        size=size,
        fps=fps,
        frames_dir_name=frames_dir_name,
        out_mp4_name=out_mp4_name,
    )


@mcp.tool()
def roi_stats_ijk(
    ijk_min: Sequence[int],
    ijk_max: Sequence[int],
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute ROI intensity stats in IJK coordinates (bridge)."""

    return _ctx().roi_stats_ijk(ijk_min=ijk_min, ijk_max=ijk_max, volume_id=volume_id, volume_name=volume_name)


@mcp.tool()
def export_segmentation_dicom(segmentation_id: str, reference_volume_id: str, output_folder: str) -> Dict[str, Any]:
    """Export a segmentation node as DICOM SEG (bridge)."""

    return _ctx().export_segmentation_dicom(
        segmentation_id=segmentation_id,
        reference_volume_id=reference_volume_id,
        output_folder=output_folder,
    )


# -------------------------
# Direct WebServer endpoint wrappers
# -------------------------


@mcp.tool()
def get_screenshot_png(out_name: str = "screenshot.png") -> Dict[str, Any]:
    """Get an application screenshot PNG via `/slicer/screenshot`."""

    return _ctx().get_screenshot_png(out_name=out_name)


@mcp.tool()
def get_threeD_png(look_from_axis: Optional[str] = None, out_name: str = "threeD.png") -> Dict[str, Any]:
    """Render the 3D view PNG via `/slicer/threeD`."""

    return _ctx().get_threeD_png(look_from_axis=look_from_axis, out_name=out_name)


@mcp.tool()
def get_timeimage_png(color: str = "pink", out_name: str = "timeimage.png") -> Dict[str, Any]:
    """Get timeimage PNG via `/slicer/timeimage`."""

    return _ctx().get_timeimage_png(color=color, out_name=out_name)


@mcp.tool()
def set_gui(contents: Optional[str] = None, viewers_layout: Optional[str] = None) -> Dict[str, Any]:
    """Set GUI settings via `PUT /slicer/gui`."""

    return _ctx().set_gui(contents=contents, viewers_layout=viewers_layout)


@mcp.tool()
def mrml_list_nodes(classname: Optional[str] = None) -> Any:
    """List MRML nodes via `GET /slicer/mrml`."""

    return _ctx().mrml_list_nodes(classname=classname)


@mcp.tool()
def mrml_names(classname: Optional[str] = None) -> Any:
    """List MRML node names via `GET /slicer/mrml/names`."""

    return _ctx().mrml_names(classname=classname)


@mcp.tool()
def mrml_ids(classname: Optional[str] = None) -> Any:
    """List MRML node ids via `GET /slicer/mrml/ids`."""

    return _ctx().mrml_ids(classname=classname)


@mcp.tool()
def mrml_properties(node_id: str) -> Any:
    """Get MRML node properties via `GET /slicer/mrml/properties?id=...`."""

    return _ctx().mrml_properties(node_id=node_id)


@mcp.tool()
def mrml_set_properties(node_id: str, properties: Dict[str, Any]) -> Any:
    """Set MRML node properties via `PUT /slicer/mrml?id=...`."""

    return _ctx().mrml_set_properties(node_id=node_id, properties=properties)


@mcp.tool()
def mrml_file_get(node_id: str, file_name: str) -> Any:
    """Trigger saving an MRML node to a file on the Slicer host via `GET /slicer/mrml/file`."""

    return _ctx().mrml_file_get(node_id=node_id, file_name=file_name)


@mcp.tool()
def mrml_file_post(file_name: str) -> Any:
    """Load MRML node from a file on the Slicer host via `POST /slicer/mrml/file`."""

    return _ctx().mrml_file_post(file_name=file_name)


@mcp.tool()
def mrml_file_delete(node_id: Optional[str] = None, file_name: Optional[str] = None) -> Any:
    """Delete an MRML node/file via `DELETE /slicer/mrml/file`."""

    return _ctx().mrml_file_delete(node_id=node_id, file_name=file_name)


@mcp.tool()
def volume_selection(cmd: str) -> Any:
    """Move volume selection forward/back via `GET /slicer/volumeSelection?cmd=...`."""

    return _ctx().volume_selection(cmd=cmd)


@mcp.tool()
def list_gridtransforms() -> Any:
    """List grid transforms via `GET /slicer/gridtransforms`."""

    return _ctx().list_gridtransforms()


@mcp.tool()
def download_volume_nrrd(node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
    """Download a volume as NRRD via `GET /slicer/volume?id=...`."""

    return _ctx().download_volume_nrrd(node_id=node_id, out_name=out_name)


@mcp.tool()
def upload_volume_nrrd(nrrd_path: str) -> Any:
    """Upload a volume NRRD via `POST /slicer/volume`."""

    return _ctx().upload_volume_nrrd(nrrd_path=nrrd_path)


@mcp.tool()
def download_gridtransform_nrrd(node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
    """Download a grid transform as NRRD via `GET /slicer/gridtransform?id=...`."""

    return _ctx().download_gridtransform_nrrd(node_id=node_id, out_name=out_name)


@mcp.tool()
def list_fiducials() -> Any:
    """List fiducial nodes via `GET /slicer/fiducials`."""

    return _ctx().list_fiducials()


@mcp.tool()
def set_fiducial_ras(node_id: str, r: float, a: float, s: float) -> Any:
    """Set fiducial RAS via `PUT /slicer/fiducial` (WebServer API)."""

    return _ctx().set_fiducial_ras(node_id=node_id, r=r, a=a, s=s)


@mcp.tool()
def get_tracking(m: Optional[str] = None, q: Optional[str] = None, p: Optional[str] = None) -> Any:
    """Get tracking data via `GET /slicer/tracking`."""

    return _ctx().get_tracking(m=m, q=q, p=p)


@mcp.tool()
def load_sampledata(name: str) -> Any:
    """Load sample data via `GET /slicer/sampledata?name=...`."""

    return _ctx().load_sampledata(name=name)


@mcp.tool()
def access_dicomweb_study(payload_json: str) -> Any:
    """Proxy `POST /slicer/accessDICOMwebStudy`.

    Args:
        payload_json: JSON string accepted by Slicer WebServer.
    """

    payload = json.loads(payload_json)
    return _ctx().access_dicomweb_study(payload)


@mcp.tool()
def raw_slicer_request(
    method: str,
    path: str,
    params_json: Optional[str] = None,
    body_json: Optional[str] = None,
    data_text: Optional[str] = None,
    save_as: Optional[str] = None,
) -> Dict[str, Any]:
    """Raw HTTP call to `/slicer/*` for forward compatibility."""

    params = json.loads(params_json) if params_json else None
    body = json.loads(body_json) if body_json else None
    return _ctx().raw_slicer_request(method=method, path=path, params=params, json_body=body, data_text=data_text, save_as=save_as)


@mcp.tool()
def raw_dicomweb_request(
    method: str,
    dicom_path: str,
    params_json: Optional[str] = None,
    headers_json: Optional[str] = None,
    data_text: Optional[str] = None,
    save_as: Optional[str] = None,
) -> Dict[str, Any]:
    """Raw HTTP call to `/dicom/*` (DICOMweb)."""

    params = json.loads(params_json) if params_json else None
    headers = json.loads(headers_json) if headers_json else None
    return _ctx().raw_dicomweb_request(
        method=method,
        dicom_path=dicom_path,
        params=params,
        headers=headers,
        data_text=data_text,
        save_as=save_as,
    )




# -------------------------
# Exec escape hatches (dangerous)
# -------------------------


@mcp.tool()
def exec_python(source: str) -> Dict[str, Any]:
    """Run raw Python in Slicer via `/slicer/exec`.

    **Unsafe**: do not expose to untrusted callers.
    """

    return _ctx().exec_python(source=source)


@mcp.tool()
def exec_bridge(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Call an arbitrary bridge tool by name."""

    return _ctx().exec_bridge(tool=tool, args=args)

# -------------------------
# Optional: Gemini video understanding as a tool
# -------------------------


@mcp.tool()
def gemini_analyze_video(video_path: str, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Analyze an MP4 video using Gemini (Files API + generate_content).

    Requires:
      - google-genai installed
      - GEMINI_API_KEY env var set

    Model default: gemini-3-pro-preview (requested) unless overridden.
    """

    from .gemini_video import GeminiVideoAnalyzer

    analyzer = GeminiVideoAnalyzer(model=model or os.environ.get("GEMINI_MODEL", "gemini-3-pro-preview"))
    text = analyzer.analyze_mp4(video_path=video_path, prompt=prompt)
    return {"text": text}


# -------------------------
# Entrypoint
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server exposing 3D Slicer WebServer as tools")
    parser.add_argument("--base-url", default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"))
    parser.add_argument(
        "--bridge-dir",
        default=os.environ.get("SLICER_BRIDGE_DIR"),
        help="Directory that contains slicer_agent_bridge.py (typically ./slicer_side)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SLICER_AGENT_OUT_DIR", "./runs/mcp"),
        help="Where to write artifacts + traces",
    )
    parser.add_argument("--session-id", default=os.environ.get("SLICER_AGENT_SESSION", None))
    parser.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"), choices=["stdio", "sse"]) 

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bridge_dir:
        bridge_dir = Path(args.bridge_dir).expanduser().resolve()
    else:
        # Best effort: assume repo layout where slicer_side sits next to this file's parent directory
        bridge_dir = (Path(__file__).resolve().parents[2] / "slicer_side").resolve()

    if not bridge_dir.exists():
        raise FileNotFoundError(
            f"bridge_dir not found: {bridge_dir}. Provide --bridge-dir or set SLICER_BRIDGE_DIR."
        )

    ffmpeg = shutil.which("ffmpeg")
    video = VideoRenderer(ffmpeg_path=ffmpeg) if ffmpeg else None

    client = SlicerClient(base_url=args.base_url)
    session = SessionManager(out_dir=out_dir, session_id=args.session_id)

    global _CTX
    _CTX = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video)

    # Server logs must go to stderr (MCP stdio contract)
    logging.basicConfig(level=logging.INFO)

    mcp.run(transport=args.transport)
