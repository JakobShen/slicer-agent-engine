
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .capability_registry import ToolSpec, build_default_tool_surface
from .session import SessionManager
from .slicer_client import SlicerClient
from .tools import ToolContext
from .video_renderer import VideoRenderer

logger = logging.getLogger(__name__)

mcp = None
_CTX: Optional[ToolContext] = None
_SURFACE = None


def _load_fastmcp():
    try:
        from fastmcp import FastMCP  # type: ignore
        return FastMCP
    except Exception:
        from mcp.server.fastmcp import FastMCP  # type: ignore
        return FastMCP


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("ToolContext not initialized. Call main().")
    return _CTX


def _surface():
    if _SURFACE is None:
        raise RuntimeError("Tool surface not initialized. Call main().")
    return _SURFACE


def _safe_identifier(name: str) -> str:
    ident = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"tool_{ident}"
    return ident


def _build_dynamic_tool(spec: ToolSpec) -> Callable[..., Dict[str, Any]]:
    properties = dict(spec.parameters.get("properties") or {})
    required = set(spec.parameters.get("required") or [])
    arg_defs = []
    arg_to_prop: Dict[str, str] = {}

    for prop_name in properties:
        py_name = _safe_identifier(prop_name)
        arg_to_prop[py_name] = prop_name
        if prop_name in required:
            arg_defs.append(f"{py_name}: Any")
        else:
            arg_defs.append(f"{py_name}: Any = None")

    fn_name = _safe_identifier(spec.name)
    params_src = ", ".join(arg_defs)
    doc = spec.description.replace('"""', '\"\"\"')

    lines = [f"def {fn_name}({params_src}) -> Dict[str, Any]:"]
    lines.append(f'    """{doc}"""')
    lines.append("    args: Dict[str, Any] = {}")
    for py_name, prop_name in arg_to_prop.items():
        lines.append(f"    if {py_name} is not None:")
        lines.append(f'        args["{prop_name}"] = {py_name}')
    lines.append(f'    return _surface().execute_named_tool("{spec.name}", args)')
    src = "\n".join(lines)

    glb = {"Any": Any, "Dict": Dict, "_surface": _surface}
    loc: Dict[str, Any] = {}
    exec(src, glb, loc)
    fn = loc[fn_name]
    fn.__name__ = fn_name
    fn.__qualname__ = fn_name
    fn.__doc__ = spec.description
    fn.__module__ = __name__
    return fn


def _register_registry_tools() -> None:
    if mcp is None:
        raise RuntimeError("FastMCP server not initialized.")
    for spec in _surface().mcp_tool_specs():
        fn = _build_dynamic_tool(spec)
        mcp.tool()(fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server exposing stable 3D Slicer tools")
    parser.add_argument("--base-url", default=os.environ.get("SLICER_BASE_URL", "http://localhost:2016"))
    parser.add_argument(
        "--bridge-dir",
        default=os.environ.get("SLICER_BRIDGE_DIR"),
        help="Directory that contains slicer_agent_bridge.py (typically ./slicer_side)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SLICER_AGENT_OUT_DIR", "./runs/mcp"),
        help="Where to write artifacts, traces, and evidence logs",
    )
    parser.add_argument("--session-id", default=os.environ.get("SLICER_AGENT_SESSION", None))
    parser.add_argument("--transport", default=os.environ.get("MCP_TRANSPORT", "stdio"), choices=["stdio", "sse"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bridge_dir:
        bridge_dir = Path(args.bridge_dir).expanduser().resolve()
    else:
        bridge_dir = (Path(__file__).resolve().parents[1] / "slicer_side").resolve()

    if not bridge_dir.exists():
        raise FileNotFoundError(
            f"bridge_dir not found: {bridge_dir}. Provide --bridge-dir or set SLICER_BRIDGE_DIR."
        )

    ffmpeg = shutil.which("ffmpeg")
    video = VideoRenderer(ffmpeg_path=ffmpeg) if ffmpeg else None

    client = SlicerClient(base_url=args.base_url)
    session = SessionManager(out_dir=out_dir, session_id=args.session_id)

    global _CTX, _SURFACE, mcp
    _CTX = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video)
    _SURFACE = build_default_tool_surface(ctx=_CTX)

    FastMCP = _load_fastmcp()
    mcp = FastMCP("slicer-agent-engine")
    _register_registry_tools()

    # MCP stdio transports require logs on stderr.
    logging.basicConfig(level=logging.INFO)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
