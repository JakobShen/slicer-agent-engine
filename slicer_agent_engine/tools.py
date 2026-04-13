from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .slicer_client import SlicerClient
from .session import SessionManager
from .video_renderer import VideoRenderer
from .runtime_tools import (
    ToolContextBaseMixin,
    ToolContextSceneMixin,
    ToolContextViewerMixin,
    ToolContextQuantMixin,
    ToolContextSegmentationMixin,
    ToolContextRegistrationMixin,
    ToolContextRawMixin,
    ToolContextWorkflowMixin,
)

if TYPE_CHECKING:
    from .algorithm_runtime import AlgorithmRuntime
    from .gemini_video import GeminiVideoAnalyzer


@dataclass
class ToolContext(
    ToolContextBaseMixin,
    ToolContextSceneMixin,
    ToolContextViewerMixin,
    ToolContextQuantMixin,
    ToolContextSegmentationMixin,
    ToolContextRegistrationMixin,
    ToolContextRawMixin,
    ToolContextWorkflowMixin,
):
    """High-level host-side runtime facade used by agents.

    The public API intentionally matches the pre-refactor ToolContext so existing
    benchmark/demo/manual scripts continue to behave the same, but the
    implementation now lives in focused mixin modules under
    ``slicer_agent_engine.runtime_tools``.
    """

    client: SlicerClient
    session: SessionManager
    bridge_dir: Path
    video: Optional[VideoRenderer] = None
    gemini: Optional["GeminiVideoAnalyzer"] = None
    algorithm_runtime: Optional["AlgorithmRuntime"] = None
