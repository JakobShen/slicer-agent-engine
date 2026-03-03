from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .slicer_client import SlicerClient
from .session import SessionManager
from .tools import ToolContext
from .video_renderer import VideoRenderer, VideoRenderError


logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict)


def build_app(
    *,
    slicer_base_url: str,
    bridge_dir: Path,
    output_dir: Path,
    enable_video: bool = True,
) -> FastAPI:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = SessionManager(out_dir=output_dir)
    client = SlicerClient(base_url=slicer_base_url)

    video = None
    if enable_video:
        try:
            video = VideoRenderer()
            video.ensure_available()
        except VideoRenderError as e:
            logger.warning("Video disabled: %s", e)
            video = None

    ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video)

    app = FastAPI(title="slicer-agent-engine (tool server)", version="0.1.0")

    @app.get("/v1/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time()}

    @app.get("/v1/tools")
    def tools() -> Dict[str, Any]:
        # Keep this explicit and stable.
        return {
            "tools": [
                {"name": "ping", "desc": "Get Slicer version info"},
                {"name": "clear_scene", "desc": "Clear Slicer MRML scene"},
                {"name": "load_dicom", "desc": "Load DICOM folder into Slicer", "args": {"dicom_dir": "str", "clear_scene_first": "bool"}},
                {"name": "list_volumes", "desc": "List loaded volume nodes"},
                {"name": "select_volume", "desc": "Select active volume", "args": {"volume_id": "str?", "volume_name": "str?"}},
                {"name": "get_slice_png", "desc": "Render slice PNG", "args": {"view": "str", "orientation": "str", "scroll_to": "float", "size": "int", "out_path": "str?"}},
                {"name": "capture_cine", "desc": "Render cine MP4 by scrolling", "args": {"view": "str", "orientation": "str", "start": "float", "end": "float", "n_frames": "int", "size": "int", "fps": "int", "out_path": "str?"}},
                {"name": "roi_stats_ijk", "desc": "ROI intensity stats (IJK bounding box)", "args": {"volume_id": "str?", "volume_name": "str?", "ijk_min": "[i,j,k]", "ijk_max": "[i,j,k]"}},
                {"name": "export_segmentation_dicom", "desc": "Export segmentation node as DICOM SEG", "args": {"segmentation_id": "str", "reference_volume_id": "str", "output_folder": "str"}},
            ],
            "session": {"session_id": session.session_id, "out_dir": str(session.out_dir)},
        }

    def _call_tool(name: str, args: Dict[str, Any]) -> Any:
        if name == "ping":
            return ctx.ping()
        if name == "clear_scene":
            return ctx.clear_scene()
        if name == "load_dicom":
            return ctx.load_dicom(args["dicom_dir"], clear_scene_first=args.get("clear_scene_first", True))
        if name == "list_volumes":
            return ctx.list_volumes()
        if name == "select_volume":
            return ctx.select_volume(volume_id=args.get("volume_id"), volume_name=args.get("volume_name"))
        if name == "get_slice_png":
            return ctx.get_slice_png(
                view=args.get("view", "red"),
                orientation=args.get("orientation", "axial"),
                scroll_to=float(args.get("scroll_to", 0.5)),
                size=int(args.get("size", 512)),
                out_path=args.get("out_path"),
            )
        if name == "capture_cine":
            return ctx.capture_cine(
                view=args.get("view", "red"),
                orientation=args.get("orientation", "axial"),
                start=float(args.get("start", 0.0)),
                end=float(args.get("end", 1.0)),
                n_frames=int(args.get("n_frames", 60)),
                size=int(args.get("size", 512)),
                fps=int(args.get("fps", 10)),
                out_path=args.get("out_path"),
            )
        if name == "roi_stats_ijk":
            ijk_min = tuple(args["ijk_min"])
            ijk_max = tuple(args["ijk_max"])
            return ctx.roi_stats_ijk(volume_id=args.get("volume_id"), volume_name=args.get("volume_name"), ijk_min=ijk_min, ijk_max=ijk_max)
        if name == "export_segmentation_dicom":
            return ctx.export_segmentation_dicom(
                segmentation_id=args["segmentation_id"],
                reference_volume_id=args["reference_volume_id"],
                output_folder=args["output_folder"],
            )
        raise KeyError(f"Unknown tool: {name}")

    @app.post("/v1/tools/{tool_name}")
    def call_tool(tool_name: str, call: ToolCall) -> Dict[str, Any]:
        try:
            result = _call_tool(tool_name, call.args)
            return {"ok": True, "result": result, "session_id": session.session_id}
        except Exception as e:
            # Keep errors short but useful.
            raise HTTPException(status_code=400, detail=str(e))

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--slicer-base-url", default="http://localhost:2016")
    parser.add_argument("--bridge-dir", required=True, help="Path to folder containing slicer_agent_bridge.py")
    parser.add_argument("--output-dir", default="./runs/http_server")
    parser.add_argument("--no-video", action="store_true", help="Disable ffmpeg-based video rendering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = build_app(
        slicer_base_url=args.slicer_base_url,
        bridge_dir=Path(args.bridge_dir),
        output_dir=Path(args.output_dir),
        enable_video=not args.no_video,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
