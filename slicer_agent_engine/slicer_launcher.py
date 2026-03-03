from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .slicer_client import SlicerClient, SlicerRequestError


@dataclass
class LaunchedSlicer:
    """Handle for a Slicer process launched by this library."""

    popen: subprocess.Popen
    executable: Path
    port: int

    def terminate(self) -> None:
        try:
            self.popen.terminate()
        except Exception:
            pass


def find_slicer_executable() -> Path:
    """Best-effort discovery of the Slicer executable.

    Order:
    1) $SLICER_EXECUTABLE
    2) `which Slicer`
    3) Common macOS install paths
    """

    env_path = os.environ.get("SLICER_EXECUTABLE")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    which = shutil.which("Slicer")
    if which:
        p = Path(which)
        if p.exists():
            return p

    mac_candidates = [
        "/Applications/Slicer.app/Contents/MacOS/Slicer",
        "/Applications/3D Slicer.app/Contents/MacOS/Slicer",
    ]
    for c in mac_candidates:
        p = Path(c)
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find Slicer executable. Set environment variable SLICER_EXECUTABLE to the full path, "
        "e.g. /Applications/Slicer.app/Contents/MacOS/Slicer"
    )


def launch_slicer_with_webserver(
    *,
    port: int,
    bootstrap_script: Path,
    enable_slicer_api: bool = True,
    enable_exec: bool = True,
    enable_dicomweb: bool = False,
    slicer_executable: Optional[Path] = None,
    extra_args: Optional[list[str]] = None,
    env: Optional[Dict[str, str]] = None,
    stdout_to_devnull: bool = True,
) -> LaunchedSlicer:
    """Launch Slicer and run a bootstrap script that starts the WebServer."""

    slicer_executable = slicer_executable or find_slicer_executable()
    bootstrap_script = Path(bootstrap_script).expanduser().resolve()
    if not bootstrap_script.exists():
        raise FileNotFoundError(f"bootstrap_script not found: {bootstrap_script}")

    # NOTE: Do NOT use --no-main-window by default.
    # WebServer endpoints such as /slicer/slice rely on the Qt layout manager
    # (slicer.app.layoutManager()) and slice widgets. When Slicer is started
    # with --no-main-window, layoutManager() is None and /slicer/slice fails
    # with: "'NoneType' object has no attribute 'sliceWidget'".
    #
    # For headless Linux nodes, run Slicer under a virtual display (e.g. Xvfb)
    # instead of disabling the main window.
    cmd = [
        str(slicer_executable),
        "--no-splash",
        "--python-script",
        str(bootstrap_script),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    proc_env["SLICER_WEBSERVER_PORT"] = str(int(port))
    proc_env["SLICER_WEBSERVER_ENABLE_SLICER"] = "1" if enable_slicer_api else "0"
    proc_env["SLICER_WEBSERVER_ENABLE_EXEC"] = "1" if enable_exec else "0"
    proc_env["SLICER_WEBSERVER_ENABLE_DICOM"] = "1" if enable_dicomweb else "0"

    stdout = subprocess.DEVNULL if stdout_to_devnull else None
    stderr = subprocess.DEVNULL if stdout_to_devnull else None

    popen = subprocess.Popen(cmd, env=proc_env, stdout=stdout, stderr=stderr)
    return LaunchedSlicer(popen=popen, executable=slicer_executable, port=port)


def wait_for_webserver(
    client: SlicerClient,
    *,
    timeout_s: float = 120.0,
    poll_s: float = 1.0,
) -> None:
    """Wait until `GET /slicer/system/version` succeeds."""

    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        try:
            client.get_system_version()
            return
        except Exception as e:
            last_err = e
            time.sleep(poll_s)

    raise SlicerRequestError(f"Slicer WebServer did not become ready within timeout. Last error: {last_err}")


def ensure_webserver(
    *,
    base_url: str,
    bootstrap_script: Path,
    start_if_not_running: bool = True,
    port: int = 2016,
    timeout_s: float = 180.0,
    require_exec: bool = False,
    require_slice: bool = False,
) -> Optional[LaunchedSlicer]:
    """Ensure Slicer WebServer is reachable.

    If not reachable and `start_if_not_running` is True, attempt to launch Slicer and start WebServer.

    Returns:
        LaunchedSlicer if a new process was started, else None.
    """

    client = SlicerClient(base_url=base_url, timeout_s=10.0)

    # 1) If a server is already running on this base_url but is missing required capabilities
    #    (e.g., exec disabled), DO NOT auto-launch another Slicer.
    #    It would likely start on a different port (WebServer finds a free port), while the client
    #    would still be talking to the original base_url.
    try:
        client.get_system_version()
        if require_exec:
            client.assert_exec_enabled()
        if require_slice:
            # IMPORTANT:
            # `/slicer/slice` may return HTTP 500 before any volume is loaded
            # (e.g. "GetDataType" on None image data). For startup readiness we
            # only need to know that the Qt slice widgets exist.
            if require_exec:
                client.assert_slice_widgets_available()
            else:
                # Best-effort check without exec: a full screenshot should only
                # work once the main window/layout exists.
                _ = client.get_screenshot_png()
        return None
    except SlicerRequestError as e:
        msg = str(e)
        # Only auto-launch if the server is truly unreachable (connection error, DNS, etc.).
        # If we got an HTTP error response, then a server is running but misconfigured.
        if "Failed to call Slicer WebServer" not in msg:
            raise

        if not start_if_not_running:
            raise

    launched = launch_slicer_with_webserver(port=port, bootstrap_script=bootstrap_script)

    # Wait for readiness
    client = SlicerClient(base_url=base_url, timeout_s=10.0)
    wait_for_webserver(client, timeout_s=timeout_s)
    if require_exec:
        client.assert_exec_enabled()
    if require_slice:
        # Slice widgets may initialize slightly after the system/version endpoint becomes ready.
        # Retry a bit to avoid flaky startup.
        deadline = time.time() + min(timeout_s, 90.0)
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                if require_exec:
                    client.assert_slice_widgets_available()
                else:
                    _ = client.get_screenshot_png()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        if last_err is not None:
            raise SlicerRequestError(f"Slicer slice widgets did not become ready. Last error: {last_err}")
    return launched
