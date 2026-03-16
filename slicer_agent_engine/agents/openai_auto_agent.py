from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Union

from .auto_agent import AutoAgentResult, run_auto_task
from ..llm.registry import build_builtin_model_tools
from ..tools import ToolContext

if TYPE_CHECKING:
    from ..load_helper import CaseSeriesCatalog


def run_openai_auto_task(
    *,
    oa_client: Any,
    model: str,
    ctx: ToolContext,
    case_path: Union[str, Path],
    instructions: str,
    user_message: str,
    max_rounds: int = 12,
    attach_images: bool = True,
    max_images_per_tool: int = 4,
    open_case_kwargs: Optional[Dict[str, Any]] = None,
    pre_messages: Optional[list[Any]] = None,
    case_brief_text: Optional[str] = None,
    case_series_catalog: Optional["CaseSeriesCatalog"] = None,
    initial_active_packs: Optional[Sequence[str]] = None,
    extra_tools_schema: Optional[Sequence[Dict[str, Any]]] = None,
    extra_tool_executor: Optional[Any] = None,
    reuse_current_scene: bool = False,
    enable_code_interpreter: bool = False,
    reasoning_effort: Optional[str] = None,
    disable_tool_packs: bool = False,
    model_request_timeout_sec: Optional[float] = None,
    model_request_retry_on_timeout: int = 0,
) -> AutoAgentResult:
    """Compatibility wrapper preserving the pre-refactor OpenAI entrypoint."""

    return run_auto_task(
        provider="openai",
        model_client=oa_client,
        model=model,
        ctx=ctx,
        case_path=case_path,
        instructions=instructions,
        user_message=user_message,
        max_rounds=max_rounds,
        attach_images=attach_images,
        max_images_per_tool=max_images_per_tool,
        open_case_kwargs=open_case_kwargs,
        pre_messages=pre_messages,
        case_brief_text=case_brief_text,
        case_series_catalog=case_series_catalog,
        initial_active_packs=initial_active_packs,
        extra_tools_schema=extra_tools_schema,
        extra_tool_executor=extra_tool_executor,
        reuse_current_scene=reuse_current_scene,
        extra_model_tools=build_builtin_model_tools("openai", enable_code_interpreter=bool(enable_code_interpreter)),
        reasoning_effort=reasoning_effort,
        disable_tool_packs=disable_tool_packs,
        model_request_timeout_sec=model_request_timeout_sec,
        model_request_retry_on_timeout=model_request_retry_on_timeout,
    )
