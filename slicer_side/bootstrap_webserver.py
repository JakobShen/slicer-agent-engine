"""Start the 3D Slicer WebServer from within Slicer.

This script is meant to be run via:

  Slicer --no-splash --python-script /path/to/bootstrap_webserver.py

Configuration is taken from environment variables:

- SLICER_WEBSERVER_PORT (default: 2016)
- SLICER_WEBSERVER_ENABLE_SLICER (default: 1)
- SLICER_WEBSERVER_ENABLE_EXEC (default: 1)
- SLICER_WEBSERVER_ENABLE_DICOM (default: 0)

This is a "best effort" bootstrap because WebServer internals may differ across Slicer versions.
If it fails, start WebServer manually in the GUI and enable the required APIs.
"""

from __future__ import annotations

import os
import sys
import time

import slicer


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    port = int(os.environ.get("SLICER_WEBSERVER_PORT", "2016"))
    enable_slicer = _env_flag("SLICER_WEBSERVER_ENABLE_SLICER", True)
    enable_exec = _env_flag("SLICER_WEBSERVER_ENABLE_EXEC", True)
    enable_dicom = _env_flag("SLICER_WEBSERVER_ENABLE_DICOM", False)

    # IMPORTANT:
    # Do NOT instantiate WebServerLogic() with defaults and then call addDefaultRequestHandlers again.
    # WebServerLogic() creates default handlers immediately (enableExec defaults to False), and the first
    # SlicerRequestHandler may intercept /slicer/exec and respond with "unknown command b'/exec'".
    #
    # Correct pattern (mirrors WebServerWidget.startServer):
    #   logic = WebServerLogic(requestHandlers=[])
    #   logic.requestHandlers = []
    #   logic.addDefaultRequestHandlers(enableSlicer=..., enableExec=..., ...)
    #   logic.start()
    try:
        import WebServer

        # Create logic with *no* default handlers.
        logic = WebServer.WebServerLogic(
            port=port,
            enableSlicer=False,
            enableExec=False,
            enableDICOM=False,
            enableStaticPages=False,
            requestHandlers=[],
        )

        # (Re)register handlers with desired flags.
        logic.requestHandlers = []
        logic.addDefaultRequestHandlers(
            enableSlicer=enable_slicer,
            enableExec=enable_exec,
            enableDICOM=enable_dicom,
            enableStaticPages=True,
        )

        # Ensure requested port is used as the starting port.
        logic.port = port

        logic.start()
        slicer.app.processEvents()

        sys.stderr.write(
            f"[bootstrap_webserver] started=True port={logic.port} "
            f"enable_slicer={enable_slicer} enable_exec={enable_exec} enable_dicom={enable_dicom}\n"
        )
        sys.stderr.flush()

    except Exception as e:
        sys.stderr.write(f"[bootstrap_webserver] Failed to start WebServer: {e}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
