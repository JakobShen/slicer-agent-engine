from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EngineConfig:
    """Runtime configuration for slicer-agent-engine.

    All paths should be **absolute** so that they work regardless of current working directory.

    Attributes:
        slicer_base_url: Base URL of Slicer WebServer, e.g. http://localhost:2016
        slicer_bridge_dir: Directory that contains `slicer_agent_bridge.py` (imported inside Slicer via /slicer/exec)
        request_timeout_s: HTTP timeout for calls to Slicer.
    """

    slicer_base_url: str = "http://localhost:2016"
    slicer_bridge_dir: Optional[Path] = None
    request_timeout_s: float = 60.0

    def require_bridge_dir(self) -> Path:
        if self.slicer_bridge_dir is None:
            raise ValueError("slicer_bridge_dir is not set. It must point to the folder containing slicer_agent_bridge.py")
        return self.slicer_bridge_dir
