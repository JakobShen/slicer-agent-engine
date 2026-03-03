from __future__ import annotations

import logging
import sys
from pathlib import Path


def _ensure_imports() -> None:
    """Allow running `python scripts/manual_test.py` without installing the package."""

    try:
        import slicer_agent_engine  # noqa: F401
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


_ensure_imports()

from slicer_agent_engine.session import SessionManager
from slicer_agent_engine.slicer_client import SlicerClient
from slicer_agent_engine.slicer_launcher import ensure_webserver
from slicer_agent_engine.tools import ToolContext
from slicer_agent_engine.video_renderer import VideoRenderer


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    base_url = "http://localhost:2016"
    dicom_dir = Path("/Users/weixiangshen/downloads/dicom/MRHead_DICOM")
    out_dir = Path("/Users/weixiangshen/downloads/dicom/test_tmp")

    # Repo paths
    repo_root = Path(__file__).resolve().parents[1]
    bridge_dir = repo_root / "slicer_side"
    bootstrap_script = bridge_dir / "bootstrap_webserver.py"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to auto-start WebServer if it isn't already running.
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
    video = VideoRenderer()
    ctx = ToolContext(client=client, session=session, bridge_dir=bridge_dir, video=video)

    print("Ping Slicer...")
    print(ctx.ping())

    print(f"Loading DICOM from: {dicom_dir}")
    res = ctx.load_dicom(dicom_dir)
    print(res)

    vols = ctx.list_volumes()
    print(f"Found {len(vols)} volumes:")
    for v in vols:
        print("  -", v)

    # Render three orthogonal mid-slices
    png1 = ctx.get_slice_png(view="red", orientation="axial", scroll_to=0.5, size=512)
    png2 = ctx.get_slice_png(view="red", orientation="sagittal", scroll_to=0.5, size=512)
    png3 = ctx.get_slice_png(view="red", orientation="coronal", scroll_to=0.5, size=512)
    print("Saved PNGs:", png1, png2, png3)

    # Cine
    cine = ctx.capture_cine(view="red", orientation="axial", start=0.0, end=1.0, n_frames=60, size=512, fps=12)
    print("Saved cine:", cine)

    # ROI stats demo
    stats = ctx.roi_stats_ijk(ijk_min=[80, 80, 40], ijk_max=[160, 160, 90])
    print("ROI stats:", stats)

    print("Done. Outputs are in:", out_dir)
    print("Trace:", session.trace_path)
    print("State:", session.state_path)


if __name__ == "__main__":
    main()
