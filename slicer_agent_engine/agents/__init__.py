"""Agent runners and task-specific wrappers.

The main entrypoint is :func:`run_auto_task`. The legacy OpenAI-specific wrapper
is kept as a lazy compatibility export so importing ``slicer_agent_engine.agents``
does not eagerly trigger deprecation warnings.
"""

from __future__ import annotations

from .auto_agent import AutoAgentResult, run_auto_task

__all__ = ["AutoAgentResult", "run_auto_task", "run_openai_auto_task"]


def __getattr__(name: str):
    if name == "run_openai_auto_task":
        from .openai_auto_agent import run_openai_auto_task as legacy_run_openai_auto_task

        return legacy_run_openai_auto_task
    raise AttributeError(name)
