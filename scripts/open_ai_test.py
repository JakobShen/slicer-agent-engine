from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_imports() -> None:
    """Allow running `python scripts/open_ai_test.py` without installing the package."""

    try:
        import slicer_agent_engine  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


_ensure_imports()

from slicer_agent_engine.gemini_video import GeminiVideoAnalyzer
from slicer_agent_engine.session import SessionManager
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.slicer_launcher import ensure_webserver
from slicer_agent_engine.tools import ToolContext
from slicer_agent_engine.video_renderer import VideoRenderer


def _file_to_data_url(path: Path, mime: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"



def build_tools_schema() -> List[Dict[str, Any]]:
    """OpenAI function tools schema list.

    Format follows OpenAI Responses API function-calling guide.

    Keep this list focused on:
      - L0 viewer controls
      - L1 deterministic quantification
      - raw_slicer_request / raw_dicomweb_request as escape hatches
      - gemini_analyze_video for MP4 understanding (OpenAI models currently can't read video)
    """

    return [
        # --- Basics ---
        {
            "type": "function",
            "name": "ping",
            "description": "Ping Slicer WebServer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "clear_scene",
            "description": "Clear the current Slicer scene.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "load_dicom",
            "description": "Load a DICOM folder into Slicer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dicom_dir": {"type": "string"},
                    "clear_scene_first": {"type": "boolean", "default": True},
                },
                "required": ["dicom_dir"],
            },
        },
        {
            "type": "function",
            "name": "list_volumes",
            "description": "List volumes in Slicer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "select_volume",
            "description": "Select active volume by id or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": [],
            },
        },

        # --- L0: viewer actuation ---
        {
            "type": "function",
            "name": "get_viewer_state",
            "description": "Get current viewer state snapshot (active volume, per-view slice state, window/level, etc.).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "set_window_level",
            "description": "Set window/level. To auto-compute, set 'auto': true and OMIT window/level parameters entirely. DO NOT set window=0.",            "parameters": {
                "type": "object",
                "properties": {
                    "window": {"type": "number"},
                    "level": {"type": "number"},
                    "auto": {"type": "boolean"},
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "set_interpolation",
            "description": "Enable/disable slice interpolation for the active (or specified) volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interpolate": {"type": "boolean"},
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": ["interpolate"],
            },
        },
        {
            "type": "function",
            "name": "set_slice_orientation",
            "description": "Set slice view orientation: axial/sagittal/coronal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "orientation": {"type": "string", "default": "axial"},
                },
                "required": ["view", "orientation"],
            },
        },
        {
            "type": "function",
            "name": "set_slice_scroll_to",
            "description": "Set slice position using normalized scroll_to in [0,1].",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "scroll_to": {"type": "number", "default": 0.5},
                },
                "required": ["view", "scroll_to"],
            },
        },
        {
            "type": "function",
            "name": "get_slice_offset_range",
            "description": "Get min/max slice offset range (mm) for the current volume in a view.",
            "parameters": {
                "type": "object",
                "properties": {"view": {"type": "string", "default": "red"}},
                "required": ["view"],
            },
        },
        {
            "type": "function",
            "name": "set_slice_offset",
            "description": "Set slice offset in mm for a view.",
            "parameters": {
                "type": "object",
                "properties": {"view": {"type": "string", "default": "red"}, "offset": {"type": "number"}},
                "required": ["view", "offset"],
            },
        },
        {
            "type": "function",
            "name": "fit_slice",
            "description": "Fit slice view to the full volume.",
            "parameters": {
                "type": "object",
                "properties": {"view": {"type": "string", "default": "red"}},
                "required": ["view"],
            },
        },
        {
            "type": "function",
            "name": "zoom_slice_relative",
            "description": "Zoom slice relative by a factor (>1 zooms in).",
            "parameters": {
                "type": "object",
                "properties": {"view": {"type": "string", "default": "red"}, "factor": {"type": "number", "default": 1.5}},
                "required": ["view", "factor"],
            },
        },
        {
            "type": "function",
            "name": "set_field_of_view",
            "description": "Set slice view field-of-view directly (mm).",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "field_of_view": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["view", "field_of_view"],
            },
        },
        {
            "type": "function",
            "name": "set_xyz_origin",
            "description": "Set slice view XYZ origin directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "xyz_origin": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["view", "xyz_origin"],
            },
        },
        {
            "type": "function",
            "name": "jump_to_ras",
            "description": "Jump slices to a RAS point [x,y,z] in mm (centers if requested).",
            "parameters": {
                "type": "object",
                "properties": {"ras": {"type": "array", "items": {"type": "number"}}, "centered": {"type": "boolean", "default": True}},
                "required": ["ras"],
            },
        },
        {
            "type": "function",
            "name": "set_linked_slices",
            "description": "Enable/disable linked slice controllers.",
            "parameters": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean", "default": True}},
                "required": ["enabled"],
            },
        },
        {
            "type": "function",
            "name": "set_layout",
            "description": "Set a standard layout (four_up, three_up, one_up_red, etc.).",
            "parameters": {
                "type": "object",
                "properties": {"layout": {"type": "string", "default": "four_up"}},
                "required": ["layout"],
            },
        },

        # --- Rendering artifacts ---
        {
            "type": "function",
            "name": "get_slice_png",
            "description": "Render a slice view to PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "orientation": {"type": "string", "default": "axial"},
                    "scroll_to": {"type": "number", "default": 0.5},
                    "size": {"type": "integer", "default": 512},
                    "out_name": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "capture_cine",
            "description": "Render multiple slices and encode an MP4 cine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "orientation": {"type": "string", "default": "axial"},
                    "start": {"type": "number", "default": 0.0},
                    "end": {"type": "number", "default": 1.0},
                    "n_frames": {"type": "integer", "default": 60},
                    "size": {"type": "integer", "default": 512},
                    "fps": {"type": "integer", "default": 12},
                    "frames_dir_name": {"type": "string"},
                    "out_mp4_name": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "get_screenshot_png",
            "description": "Capture full screenshot PNG.",
            "parameters": {
                "type": "object",
                "properties": {"out_name": {"type": "string", "default": "screenshot.png"}},
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "get_threeD_png",
            "description": "Render the 3D view to PNG (if available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "look_from_axis": {"type": "string"},
                    "out_name": {"type": "string", "default": "threeD.png"},
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "get_timeimage_png",
            "description": "Render a small timeimage PNG (latency test image).",
            "parameters": {
                "type": "object",
                "properties": {"color": {"type": "string", "default": "pink"}, "out_name": {"type": "string", "default": "timeimage.png"}},
                "required": [],
            },
        },

        # --- L1: deterministic quantification/processing ---
        {
            "type": "function",
            "name": "roi_stats_ijk",
            "description": "Compute ROI intensity statistics for an IJK box. Must provide node_id or node_name",
            "parameters": {
                "type": "object",
                "properties": {
                    "ijk_min": {"type": "array", "items": {"type": "integer"}},
                    "ijk_max": {"type": "array", "items": {"type": "integer"}},
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": ["ijk_min", "ijk_max"],
            },
        },
        {
            "type": "function",
            "name": "roi_stats_ras_box",
            "description": "Compute ROI stats for a RAS box (implemented as IJK bounding box of corners).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ras_min": {"type": "array", "items": {"type": "number"}},
                    "ras_max": {"type": "array", "items": {"type": "number"}},
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": ["ras_min", "ras_max"],
            },
        },
        {
            "type": "function",
            "name": "sample_intensity_ras",
            "description": "Sample intensity at RAS point (nearest or trilinear).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ras": {"type": "array", "items": {"type": "number"}},
                    "method": {"type": "string", "default": "nearest"},
                    "volume_id": {"type": "string"},
                    "volume_name": {"type": "string"},
                },
                "required": ["ras"],
            },
        },
        {
            "type": "function",
            "name": "measure_distance_ras",
            "description": "Measure Euclidean distance in mm between two RAS points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p1": {"type": "array", "items": {"type": "number"}},
                    "p2": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["p1", "p2"],
            },
        },
        {
            "type": "function",
            "name": "measure_angle_ras",
            "description": "Measure angle in degrees at p2 between (p1-p2) and (p3-p2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "p1": {"type": "array", "items": {"type": "number"}},
                    "p2": {"type": "array", "items": {"type": "number"}},
                    "p3": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["p1", "p2", "p3"],
            },
        },
        {
            "type": "function",
            "name": "measure_area_polygon_ras",
            "description": "Measure planar polygon area in mm^2 from a list of RAS points.",
            "parameters": {
                "type": "object",
                "properties": {"points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}},
                "required": ["points"],
            },
        },
        {
            "type": "function",
            "name": "set_thick_slab",
            "description": "Enable thick slab reconstruction (MIP/min/mean) on a slice view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "default": "red"},
                    "enabled": {"type": "boolean", "default": True},
                    "thickness_mm": {"type": "number", "default": 10.0},
                    "mode": {"type": "string", "default": "mip"},
                },
                "required": ["view"],
            },
        },
        {
            "type": "function",
            "name": "set_fusion",
            "description": "Set slice fusion: background + foreground + opacity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "background_volume_id": {"type": "string"},
                    "foreground_volume_id": {"type": "string"},
                    "opacity": {"type": "number", "default": 0.5},
                },
                "required": ["background_volume_id", "foreground_volume_id"],
            },
        },
        {
            "type": "function",
            "name": "compute_subtraction",
            "description": "Compute subtraction volume_a - volume_b into a new volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "volume_a_id": {"type": "string"},
                    "volume_b_id": {"type": "string"},
                    "output_name": {"type": "string", "default": "Subtraction"},
                },
                "required": ["volume_a_id", "volume_b_id"],
            },
        },

        # --- Bookmarks (host-side convenience) ---
        {
            "type": "function",
            "name": "bookmark_add",
            "description": "Create a bookmark (viewer_state + representative slice PNG).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "view": {"type": "string", "default": "red"},
                    "orientation": {"type": "string", "default": "axial"},
                    "scroll_to": {"type": "number", "default": 0.5},
                    "size": {"type": "integer", "default": 512},
                },
                "required": ["name"],
            },
        },
        {
            "type": "function",
            "name": "bookmark_list",
            "description": "List bookmark names stored in the current session.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },

        # --- Gemini for MP4 understanding ---
        {
            "type": "function",
            "name": "gemini_analyze_video",
            "description": "Analyze an MP4 video with Gemini and return a text summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["video_path", "prompt"],
            },
        },

        # --- Raw escape hatches ---
        {
            "type": "function",
            "name": "raw_slicer_request",
            "description": "Raw HTTP call to Slicer WebServer /slicer/* endpoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "path": {"type": "string"},
                    "params_json": {"type": "string"},
                    "body_json": {"type": "string"},
                    "data_text": {"type": "string"},
                    "save_as": {"type": "string"},
                },
                "required": ["method", "path"],
            },
        },
        {
            "type": "function",
            "name": "raw_dicomweb_request",
            "description": "Raw HTTP call to Slicer DICOMweb /dicom/* endpoints (if enabled).",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "dicom_path": {"type": "string"},
                    "params_json": {"type": "string"},
                    "headers_json": {"type": "string"},
                    "data_text": {"type": "string"},
                    "save_as": {"type": "string"},
                },
                "required": ["method", "dicom_path"],
            },
        },
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # --- Config ---
    base_url = os.environ.get("SLICER_BASE_URL", "http://localhost:2016")
    openai_model = "gpt-5.2"

    # Requested sample path
    dicom_dir = Path("/Users/weixiangshen/downloads/dicom/MRHead_DICOM")
    out_dir = Path("/Users/weixiangshen/downloads/dicom/test_tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    bridge_dir = repo_root / "slicer_side"
    bootstrap_script = bridge_dir / "bootstrap_webserver.py"

    # Auto-start Slicer WebServer if needed
    ensure_webserver(
        base_url=base_url,
        bootstrap_script=bootstrap_script,
        start_if_not_running=True,
        port=2016,
        require_exec=True,
        require_slice=True,
    )

    # Tool context
    client = SlicerClient(base_url=base_url)
    session = SessionManager(out_dir=out_dir)
    video = VideoRenderer()
    ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video)

    # Gemini tool (optional)
    gemini_model =  "gemini-3-flash-preview"
    gemini_analyzer: Optional[GeminiVideoAnalyzer] = None
    try:
        # Initializes and checks API key
        gemini_analyzer = GeminiVideoAnalyzer(model=gemini_model)
    except Exception as e:
        logging.warning("Gemini tool disabled: %s", e)

    # OpenAI client
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai python package is not installed. Install requirements-agent.txt or `pip install -r requirements-agent.txt`."
        ) from e

    oa = OpenAI()

    tools = build_tools_schema()

    # Tool dispatcher
    def run_tool(name: str, arguments_json: str) -> Dict[str, Any]:
        args = json.loads(arguments_json) if arguments_json else {}

        # Simple stdout trace so you can see what the model is actually calling.
        logging.info("[TOOL] %s %s", name, args)

        if name == "ping":
            return ctx.ping()
        if name == "clear_scene":
            return ctx.clear_scene()
        if name == "load_dicom":
            return ctx.load_dicom(args["dicom_dir"], clear_scene_first=args.get("clear_scene_first", True))
        if name == "list_volumes":
            return {"volumes": ctx.list_volumes()}
        if name == "select_volume":
            return ctx.select_volume(volume_id=args.get("volume_id"), volume_name=args.get("volume_name"))

        if name == "get_viewer_state":
            return ctx.get_viewer_state()
        if name == "set_window_level":
            return ctx.set_window_level(
                window=args.get("window"),
                level=args.get("level"),
                auto=args.get("auto"),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
        if name == "set_interpolation":
            return ctx.set_interpolation(
                interpolate=bool(args.get("interpolate")),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
        if name == "set_slice_orientation":
            return ctx.set_slice_orientation(view=args["view"], orientation=args["orientation"])
        if name == "set_slice_scroll_to":
            return ctx.set_slice_scroll_to(view=args["view"], scroll_to=float(args["scroll_to"]))
        if name == "get_slice_offset_range":
            return ctx.get_slice_offset_range(view=args["view"])
        if name == "set_slice_offset":
            return ctx.set_slice_offset(view=args["view"], offset=float(args["offset"]))
        if name == "fit_slice":
            return ctx.fit_slice(view=args["view"])
        if name == "zoom_slice_relative":
            return ctx.zoom_slice_relative(view=args["view"], factor=float(args["factor"]))
        if name == "set_field_of_view":
            return ctx.set_field_of_view(view=args["view"], field_of_view=args["field_of_view"])
        if name == "set_xyz_origin":
            return ctx.set_xyz_origin(view=args["view"], xyz_origin=args["xyz_origin"])
        if name == "jump_to_ras":
            return ctx.jump_to_ras(ras=args["ras"], centered=bool(args.get("centered", True)))
        if name == "set_linked_slices":
            return ctx.set_linked_slices(enabled=bool(args.get("enabled", True)))
        if name == "set_layout":
            return ctx.set_layout(layout=args.get("layout", "four_up"))
        if name == "roi_stats_ras_box":
            return ctx.roi_stats_ras_box(
                ras_min=args["ras_min"],
                ras_max=args["ras_max"],
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
        if name == "sample_intensity_ras":
            return ctx.sample_intensity_ras(
                ras=args["ras"],
                method=args.get("method", "nearest"),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
        if name == "measure_distance_ras":
            return ctx.measure_distance_ras(p1=args["p1"], p2=args["p2"])
        if name == "measure_angle_ras":
            return ctx.measure_angle_ras(p1=args["p1"], p2=args["p2"], p3=args["p3"])
        if name == "measure_area_polygon_ras":
            return ctx.measure_area_polygon_ras(points=args["points"])
        if name == "set_thick_slab":
            return ctx.set_thick_slab(
                view=args.get("view", "red"),
                enabled=bool(args.get("enabled", True)),
                thickness_mm=float(args.get("thickness_mm", 0.0)),
                mode=args.get("mode", "mip"),
            )
        if name == "set_fusion":
            return ctx.set_fusion(
                background_volume_id=args["background_volume_id"],
                foreground_volume_id=args["foreground_volume_id"],
                opacity=float(args.get("opacity", 0.5)),
            )
        if name == "compute_subtraction":
            return ctx.compute_subtraction(
                volume_a_id=args["volume_a_id"],
                volume_b_id=args["volume_b_id"],
                output_name=args.get("output_name", "Subtraction"),
            )
        if name == "get_slice_png":
            return ctx.get_slice_png(
                view=args.get("view", "red"),
                orientation=args.get("orientation", "axial"),
                scroll_to=float(args.get("scroll_to", 0.5)),
                size=int(args.get("size", 512)),
                out_name=args.get("out_name"),
            )
        if name == "capture_cine":
            return ctx.capture_cine(
                view=args.get("view", "red"),
                orientation=args.get("orientation", "axial"),
                start=float(args.get("start", 0.0)),
                end=float(args.get("end", 1.0)),
                n_frames=int(args.get("n_frames", 60)),
                size=int(args.get("size", 512)),
                fps=int(args.get("fps", 12)),
                frames_dir_name=args.get("frames_dir_name"),
                out_mp4_name=args.get("out_mp4_name"),
            )
        if name == "roi_stats_ijk":
            return ctx.roi_stats_ijk(
                ijk_min=args["ijk_min"],
                ijk_max=args["ijk_max"],
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
        if name == "get_screenshot_png":
            return ctx.get_screenshot_png(out_name=args.get("out_name", "screenshot.png"))
        if name == "get_threeD_png":
            return ctx.get_threeD_png(look_from_axis=args.get("look_from_axis"), out_name=args.get("out_name", "threeD.png"))
        if name == "get_timeimage_png":
            return ctx.get_timeimage_png(color=args.get("color", "pink"), out_name=args.get("out_name", "timeimage.png"))
        if name == "bookmark_add":
            return ctx.bookmark_add(
                name=args["name"],
                view=args.get("view", "red"),
                orientation=args.get("orientation", "axial"),
                scroll_to=float(args.get("scroll_to", 0.5)),
                size=int(args.get("size", 512)),
            )
        if name == "bookmark_list":
            return ctx.bookmark_list()
        if name == "gemini_analyze_video":
            if gemini_analyzer is None:
                raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY and install google-genai.")
            model = args.get("model") or gemini_model
            prompt = args["prompt"]
            video_path = args["video_path"]
            # Use analyzer.model override per call
            analyzer = gemini_analyzer
            analyzer.model = model
            text = analyzer.analyze_mp4(video_path=video_path, prompt=prompt)
            return {"text": text, "model": model}
        if name == "raw_slicer_request":
            return ctx.raw_slicer_request(
                method=args["method"],
                path=args["path"],
                params=json.loads(args["params_json"]) if args.get("params_json") else None,
                json_body=json.loads(args["body_json"]) if args.get("body_json") else None,
                data_text=args.get("data_text"),
                save_as=args.get("save_as"),
            )
        if name == "raw_dicomweb_request":
            return ctx.raw_dicomweb_request(
                method=args["method"],
                dicom_path=args["dicom_path"],
                params=json.loads(args["params_json"]) if args.get("params_json") else None,
                headers=json.loads(args["headers_json"]) if args.get("headers_json") else None,
                data_text=args.get("data_text"),
                save_as=args.get("save_as"),
            )

        raise ValueError(f"Unknown tool: {name}")

    # --- Prompt ---
    instructions = (
        "You are a radiology benchmark agent. You MUST use the provided tools to inspect the MRI. "
        "Before writing the report, you MUST at minimum: "
        "You have to use gemini_analyze_video to scroll the MRI."
        "(1) call load_dicom, "
        "(2) render mid-slice PNGs in axial, sagittal, and coronal, "
        "(3) run auto window/level and re-render axial. "
        "CRITICAL INSTRUCTION FOR WINDOW/LEVEL: When calling `set_window_level` to use auto mode, ONLY pass `auto: true`. DO NOT explicitly set `window: 0` and `level: 0`, as a window of 0 will make the entire image completely white/saturated. "
        "If the image is still washed out, first call `roi_stats_ijk` to find the volume's min and max intensity, then manually call `set_window_level` with `window = max - min` and `level = (max + min) / 2`. "
        "(4) ENSURE the image is visually clear (not saturated) before enabling thick slab MIP (~12mm) and re-rendering axial. "
        "(5) compute an ROI stats box (ijk or ras), "
        "(6) capture a cine MP4 and call gemini_analyze_video to summarize it. DO NOT capture the cine if the current window/level is broken (pure white). "
        "If you cannot confirm something from the available images/stats, say so. "
        "This is for research/benchmarking, not clinical use. "
        "You can try all of these function to help the diagnosis."

    )

    input_list: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Load this DICOM folder in Slicer, then inspect the brain MRI and produce a short report (Findings + Impression).\n"
                f"DICOM folder: {dicom_dir}\n"
                f"When you render a slice PNG, I can attach it back to you as an image input.\n"
                "If you generate an MP4 cine, you cannot view it directly; call gemini_analyze_video(video_path, prompt) to get a text description."
            ),
        }
    ]

    # --- Tool calling loop (Responses API) ---
    response = oa.responses.create(model=openai_model, tools=tools, input=input_list, instructions=instructions)
    input_list += response.output

    max_rounds = 12
    round_idx = 0

    while round_idx < max_rounds:
        round_idx += 1

        function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not function_calls:
            break

        for call in function_calls:
            tool_name = call.name
            tool_args_json = call.arguments
            result = run_tool(tool_name, tool_args_json)

            # Append tool output
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

            # If a PNG was created, attach it back as an image input
            png_path = result.get("png_path") if isinstance(result, dict) else None
            if png_path:
                p = Path(png_path)
                if p.exists():
                    input_list.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": f"Rendered image from tool {tool_name}: {p.name}"},
                                {"type": "input_image", "image_url": _file_to_data_url(p, "image/png")},
                            ],
                        }
                    )

        # Next model turn
        response = oa.responses.create(model=openai_model, tools=tools, input=input_list, instructions=instructions)
        input_list += response.output

    # Final text
    report_text = response.output_text
    print("\n===== MODEL REPORT =====\n")
    print(report_text)

    report_path = out_dir / "openai_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print("\nSaved report:", report_path)
    print("Artifacts dir:", out_dir)
    print("Trace:", session.trace_path)


if __name__ == "__main__":
    main()
