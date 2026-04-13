"""Deprecated OpenAI-only auto-agent entry point.

Use ``slicer_agent_engine.agents.auto_agent.run_auto_task`` for new code.
This module stays as a compatibility shim so existing local scripts keep
working during the transition.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..legacy.openai_auto_agent import *  # noqa: F401,F403
from ..legacy import openai_auto_agent as _legacy

__all__ = [name for name in dir(_legacy) if not name.startswith("_")]


def __getattr__(name: str) -> Any:  # pragma: no cover - compatibility path
    warnings.warn(
        "slicer_agent_engine.agents.openai_auto_agent is deprecated; use run_auto_task instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_legacy, name)
