#!/usr/bin/env python3
"""
Quick smoke test for the expanded L0/L1 tool surface.

Run:
  python scripts/l0l1_smoke_test.py

Prereqs:
  - 3D Slicer installed
  - WebServer enabled (this script will try to start Slicer+WebServer automatically)
  - A DICOM folder (edit DICOM_DIR below or pass via env DICOM_DIR)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_imports() -> None:
    """Allow running without `pip install -e .`."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_imports()

from slicer_agent_engine.session import SessionManager  # noqa: E402
from slicer_agent_engine.slicer_client import SlicerClient  # noqa: E402
from slicer_agent_engine.slicer_launcher import ensure_webserver  # noqa: E402
from slicer_agent_engine.tools import ToolContext  # noqa: E402
from slicer_agent_engine.video_renderer import VideoRenderer  # noqa: E402


DEFAULT_DICOM_DIR = "/Users/weixiangshen/downloads/dicom/MRHead_DICOM"
DEFAULT_OUT_DIR = "/Users/weixiangshen/downloads/dicom/test_tmp"


def main() -> None:
    dicom_dir = Path(os.environ.get("DICOM_DIR", DEFAULT_DICOM_DIR)).expanduser()
    out_dir = Path(os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("SLICER_BASE_URL", "http://localhost:2016")
    repo_root = Path(__file__).resolve().parents[1]
    bridge_dir = repo_root / "slicer_side"
    bootstrap_script = bridge_dir / "bootstrap_webserver.py"

    # Boot Slicer WebServer if needed.
    # NOTE: `ensure_webserver` intentionally does not accept a `client=` argument.
    # It creates a short-lived internal client for readiness checks.
    ensure_webserver(
        base_url=base_url,
        bootstrap_script=bootstrap_script,
        start_if_not_running=True,
        port=2016,
        require_exec=True,
        require_slice=True,
    )

    client = SlicerClient(base_url=base_url)

    session = SessionManager(out_dir=out_dir)
    ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=VideoRenderer())

    print("Ping:", ctx.ping())
    print("Load:", ctx.load_dicom(dicom_dir))

    # Baseline render
    print(ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512, out_name="l0_baseline_axial.png"))

    # Window/level: auto then manual
    print("WL auto:", ctx.set_window_level(auto=True))
    print(ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512, out_name="l0_auto_wl.png"))
    print("WL manual:", ctx.set_window_level(window=200.0, level=80.0))
    print(ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512, out_name="l0_manual_wl.png"))

    # Zoom
    print("Fit:", ctx.fit_slice(view="red"))
    print("Zoom:", ctx.zoom_slice_relative(view="red", factor=1.8))
    print(ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512, out_name="l0_zoomed.png"))

    # Thick slab MIP
    print("Slab:", ctx.set_thick_slab(view="red", enabled=True, thickness_mm=12.0, mode="mip"))
    print(ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512, out_name="l1_thick_slab_mip.png"))
    print("Slab off:", ctx.set_thick_slab(view="red", enabled=False, thickness_mm=0.0, mode="mip"))

    # ROI stats (IJK)
    st = ctx.get_viewer_state()
    vid = st.get("active_volume_id")
    if vid:
        print("ROI:", ctx.roi_stats_ijk(volume_id=vid, ijk_min=[80, 80, 40], ijk_max=[160, 160, 90]))

    # Bookmark
    print("Bookmark:", ctx.bookmark_add(name="mid_axial", view="red", orientation="axial", scroll_to=0.5, size=512))
    print("Bookmarks:", ctx.bookmark_list())

    print("Done. Outputs:", out_dir)
    print("Trace:", session.trace_path)
    print("State:", session.state_path)


if __name__ == "__main__":
    main()
