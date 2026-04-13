"""Slicer-side fixed entry bridge for `/slicer/exec`.

This module intentionally stays tiny. The operational code lives in
`bridge_runtime/` so the bridge surface can grow without turning this file back
into a multi-thousand-line monolith.
"""

from __future__ import annotations

from typing import Any, Dict

from bridge_runtime.core import CORE_BRIDGE_HANDLERS
from bridge_runtime.registration import REGISTRATION_BRIDGE_HANDLERS
from bridge_runtime.segmentation import SEGMENTATION_BRIDGE_HANDLERS


_BRIDGE_HANDLERS: Dict[str, Any] = {}
_BRIDGE_HANDLERS.update(CORE_BRIDGE_HANDLERS)
_BRIDGE_HANDLERS.update(REGISTRATION_BRIDGE_HANDLERS)
_BRIDGE_HANDLERS.update(SEGMENTATION_BRIDGE_HANDLERS)


def dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Single stable entry point called by external process via `/slicer/exec`."""

    tool = payload.get("tool")
    args = payload.get("args") or {}

    try:
        handler = _BRIDGE_HANDLERS.get(tool)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool}")
        result = handler(args)
        return {"ok": True, **result}
    except Exception as e:
        return {
            "ok": False,
            "tool": tool,
            "error": str(e),
        }
