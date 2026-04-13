"""Deprecated OpenAI-specific helpers.

This module remains as a compatibility shim only. The maintained implementation
now lives under ``slicer_agent_engine.legacy.openai_common`` and new code should
use the provider-neutral adapters in ``slicer_agent_engine.llm``.
"""

from __future__ import annotations

import warnings
from typing import Any

from .legacy.openai_common import *  # noqa: F401,F403
from .legacy import openai_common as _legacy

__all__ = [name for name in dir(_legacy) if not name.startswith("_")]


def __getattr__(name: str) -> Any:  # pragma: no cover - compatibility path
    warnings.warn(
        "slicer_agent_engine.openai_common is deprecated; use provider-neutral llm helpers instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_legacy, name)
