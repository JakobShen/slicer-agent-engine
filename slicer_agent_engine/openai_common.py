from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


logger = logging.getLogger(__name__)


def file_to_data_url(path: Union[str, Path], mime: str) -> str:
    p = Path(path).expanduser().resolve()
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _gather_png_paths(result: Any) -> List[Path]:
    """Best-effort extraction of PNG artifacts from a tool result."""

    paths: List[Path] = []
    if not isinstance(result, dict):
        return paths

    # Single image
    p = result.get("png_path")
    if isinstance(p, str) and p:
        paths.append(Path(p))

    # Multiple images (list)
    pp = result.get("png_paths")
    if isinstance(pp, list):
        for x in pp:
            if isinstance(x, str) and x:
                paths.append(Path(x))

    # Multiple images (dict)
    if isinstance(pp, dict):
        for x in pp.values():
            if isinstance(x, str) and x:
                paths.append(Path(x))

    # Contact sheet
    cs = result.get("contact_sheet_png_path")
    if isinstance(cs, str) and cs:
        paths.insert(0, Path(cs))

    return paths


def append_artifacts_as_user_messages(
    *,
    input_list: List[Any],
    tool_name: str,
    result: Any,
    max_images: int = 4,
) -> None:
    """Attach rendered images back to the model as `input_image` items.

    For large keyframe lists, attach the contact sheet (if present) or the first few images.
    """

    pngs = [p for p in _gather_png_paths(result) if p.exists()]
    if not pngs:
        return

    # If a contact sheet exists, prefer just that.
    contact = None
    if isinstance(result, dict):
        cs = result.get("contact_sheet_png_path")
        if isinstance(cs, str) and cs:
            p = Path(cs)
            if p.exists():
                contact = p

    if contact is not None:
        pngs = [contact]
    else:
        pngs = pngs[: max_images]

    for p in pngs:
        input_list.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Rendered image from tool {tool_name}: {p.name}"},
                    {"type": "input_image", "image_url": file_to_data_url(p, "image/png")},
                ],
            }
        )


ToolExecutor = Callable[[str, str], Dict[str, Any]]
ToolSchemaSource = Union[Sequence[Dict[str, Any]], Callable[[], Sequence[Dict[str, Any]]]]
BuiltInToolsSource = Union[Sequence[Dict[str, Any]], Callable[[], Sequence[Dict[str, Any]]]]


def run_responses_tool_loop(
    *,
    oa_client: Any,
    model: str,
    tools_schema: ToolSchemaSource,
    input_list: List[Any],
    instructions: str,
    tool_executor: ToolExecutor,
    max_rounds: int = 12,
    attach_images: bool = True,
    max_images_per_tool: int = 4,
    session: Optional[Any] = None,
    extra_openai_tools: Optional[BuiltInToolsSource] = None,
    reasoning_effort: Optional[str] = None,
    model_request_timeout_sec: Optional[float] = None,
    model_request_retry_on_timeout: int = 0,
) -> Tuple[str, List[Any]]:
    """Generic OpenAI Responses API tool-calling loop.

    - Never raises on tool execution errors: converts to {ok:false,error:...} so the run can continue.
    - Optionally attaches produced PNG artifacts back to the model.

    Returns (final_text, final_input_list).
    """

    def _current_function_tools() -> List[Dict[str, Any]]:
        if callable(tools_schema):
            return list(tools_schema())
        return list(tools_schema)

    def _current_extra_tools() -> List[Dict[str, Any]]:
        if extra_openai_tools is None:
            return []
        if callable(extra_openai_tools):
            return list(extra_openai_tools())
        return list(extra_openai_tools)

    def _current_tools() -> List[Dict[str, Any]]:
        return _current_function_tools() + _current_extra_tools()

    def _tool_names() -> List[str]:
        names: List[str] = []
        for tool in _current_tools():
            if not isinstance(tool, dict):
                continue
            if tool.get("name"):
                names.append(str(tool.get("name")))
            elif tool.get("type"):
                names.append(f"[{tool.get('type')}]")
        return names

    def _log_run_event(event_type: str, **payload: Any) -> None:
        if session is not None and hasattr(session, "log_run_event"):
            try:
                session.log_run_event(event_type, **payload)
            except Exception:
                logger.exception("Failed logging run event: %s", event_type)

    def _compact(value: Any) -> Any:
        if session is not None and hasattr(session, "_compact"):
            try:
                return session._compact(value, max_str=500, max_list=15)  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(value, dict):
            return {k: _compact(v) for k, v in list(value.items())[:20]}
        if isinstance(value, list):
            return [_compact(v) for v in value[:15]]
        if isinstance(value, str) and len(value) > 500:
            return value[:500] + f" ... <{len(value) - 500} more chars>"
        return value

    class _ModelRequestTimedOut(Exception):
        pass

    def _responses_create_with_optional_timeout(create_kwargs: Dict[str, Any]) -> Any:
        timeout_sec = float(model_request_timeout_sec) if model_request_timeout_sec else 0.0
        if timeout_sec <= 0:
            return oa_client.responses.create(**create_kwargs)

        q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                resp = oa_client.responses.create(**create_kwargs)
            except Exception as e:
                q.put(("error", e))
                return
            q.put(("ok", resp))

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout_sec)
        if th.is_alive():
            raise _ModelRequestTimedOut(f"responses.create exceeded {timeout_sec:.1f}s timeout")
        try:
            status, payload = q.get_nowait()
        except queue.Empty:
            raise RuntimeError("responses.create worker returned no result")
        if status == "error":
            raise payload
        return payload

    def _create_with_timing(stage: str, *, round_idx: int) -> Any:
        visible_tools = _tool_names()
        _log_run_event(
            "model_request_start",
            stage=stage,
            round_idx=round_idx,
            model=model,
            visible_tools=visible_tools,
            visible_tool_count=len(visible_tools),
            input_item_count=len(input_list),
        )
        logger.info("[OPENAI] waiting start: %s", stage)
        t0 = time.monotonic()
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "tools": _current_tools(),
            "input": input_list,
            "instructions": instructions,
        }
        if reasoning_effort:
            create_kwargs["reasoning"] = {"effort": str(reasoning_effort)}
        if any(isinstance(tool, dict) and tool.get("type") == "code_interpreter" for tool in _current_extra_tools()):
            create_kwargs["include"] = ["code_interpreter_call.outputs"]
        attempt = 0
        max_attempts = max(1, int(model_request_retry_on_timeout or 0) + 1)
        while True:
            attempt += 1
            try:
                resp = _responses_create_with_optional_timeout(create_kwargs)
                break
            except _ModelRequestTimedOut as e:
                dt = time.monotonic() - t0
                _log_run_event(
                    "model_request_timeout",
                    stage=stage,
                    round_idx=round_idx,
                    model=model,
                    duration_ms=round(dt * 1000.0, 1),
                    attempt=attempt,
                    timeout_sec=model_request_timeout_sec,
                    error=str(e),
                )
                logger.warning("[OPENAI] waiting timed out after %.2fs: %s (attempt %d/%d)", dt, stage, attempt, max_attempts)
                if attempt >= max_attempts:
                    _log_run_event(
                        "model_request_failed",
                        stage=stage,
                        round_idx=round_idx,
                        model=model,
                        duration_ms=round(dt * 1000.0, 1),
                        attempt=attempt,
                        error=str(e),
                    )
                    raise TimeoutError(str(e))
                continue
            except Exception:
                dt = time.monotonic() - t0
                _log_run_event(
                    "model_request_failed",
                    stage=stage,
                    round_idx=round_idx,
                    model=model,
                    duration_ms=round(dt * 1000.0, 1),
                    attempt=attempt,
                )
                logger.exception("[OPENAI] waiting failed after %.2fs: %s", dt, stage)
                raise
        dt = time.monotonic() - t0
        function_calls = [
            {
                "call_id": getattr(item, "call_id", None),
                "name": getattr(item, "name", None),
            }
            for item in getattr(resp, "output", [])
            if getattr(item, "type", None) == "function_call"
        ]
        _log_run_event(
            "model_request_done",
            stage=stage,
            round_idx=round_idx,
            model=model,
            duration_ms=round(dt * 1000.0, 1),
            response_id=getattr(resp, "id", None),
            output_text_preview=_compact(getattr(resp, "output_text", None)),
            function_calls=function_calls,
            output_item_types=[getattr(item, "type", None) for item in getattr(resp, "output", [])],
        )
        logger.info("[OPENAI] waiting done in %.2fs: %s", dt, stage)
        return resp

    response = _create_with_timing("initial", round_idx=0)
    input_list += response.output

    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1
        function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not function_calls:
            break

        for call in function_calls:
            tool_name = call.name
            tool_args_json = call.arguments
            try:
                parsed_args = json.loads(tool_args_json or "{}")
            except Exception:
                parsed_args = tool_args_json

            _log_run_event(
                "tool_call_start",
                round_idx=round_idx,
                call_id=getattr(call, "call_id", None),
                tool_name=tool_name,
                arguments=_compact(parsed_args),
            )
            tool_t0 = time.monotonic()

            try:
                result = tool_executor(tool_name, tool_args_json)
            except Exception as e:
                logger.exception("Tool executor crashed: %s", tool_name)
                result = {"ok": False, "tool": tool_name, "error": str(e)}
            tool_dt = time.monotonic() - tool_t0

            summary = result
            if session is not None and hasattr(session, "summarize_tool_result"):
                try:
                    summary = session.summarize_tool_result(result)
                except Exception:
                    summary = _compact(result)
            else:
                summary = _compact(result)

            _log_run_event(
                "tool_call_end",
                round_idx=round_idx,
                call_id=getattr(call, "call_id", None),
                tool_name=tool_name,
                duration_ms=round(tool_dt * 1000.0, 1),
                ok=(bool(result.get("ok", True)) if isinstance(result, dict) else True),
                result_summary=summary,
            )

            # Always append tool output
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

            # Attach PNG artifacts
            if attach_images:
                try:
                    append_artifacts_as_user_messages(
                        input_list=input_list,
                        tool_name=tool_name,
                        result=result,
                        max_images=max_images_per_tool,
                    )
                    png_count = len(_gather_png_paths(result))
                    if png_count:
                        _log_run_event(
                            "tool_artifacts_attached",
                            round_idx=round_idx,
                            call_id=getattr(call, "call_id", None),
                            tool_name=tool_name,
                            image_count=png_count,
                        )
                except Exception:
                    logger.exception("Failed attaching images for tool %s", tool_name)

        response = _create_with_timing(f"round_{round_idx}", round_idx=round_idx)
        input_list += response.output

    _log_run_event(
        "model_loop_finished",
        model=model,
        rounds_executed=round_idx,
        final_text_preview=_compact(getattr(response, "output_text", None)),
    )

    return response.output_text, input_list
