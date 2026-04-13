from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeminiVideoAnalyzer:
    """Thin wrapper around the Gemini API for video understanding.

    Implementation follows the official Gemini API Python client patterns.

    - Uses the Files API (`client.files.upload`) to upload the MP4.
    - Sends the uploaded file in the `contents` list to `client.models.generate_content`.

    Environment
    -----------
    - GEMINI_API_KEY must be set (unless you pass api_key explicitly).

    Model
    -----
    Default is `gemini-3.1-pro-preview` (recommended). You can override via the `model` argument or GEMINI_MODEL env var.
    """

    api_key: Optional[str] = None
    model: str = "gemini-3.1-pro-preview"
    file_activation_timeout_s: float = 120.0
    file_poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        # Lazy import so base env doesn't need google-genai unless used.
        from google import genai  # type: ignore

        if self.api_key is None:
            # google-genai can also read GEMINI_API_KEY implicitly, but we keep it explicit.
            self.api_key = os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export GEMINI_API_KEY or pass api_key=..."
            )

        self._genai = genai
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _normalize_state(state_obj: object) -> str:
        if state_obj is None:
            return ""
        state_value = getattr(state_obj, "value", None) or str(state_obj)
        state_text = str(state_value).upper()
        if "." in state_text:
            state_text = state_text.rsplit(".", maxsplit=1)[-1]
        return state_text

    def _wait_for_file_active(self, *, file_name: str):
        deadline = time.monotonic() + self.file_activation_timeout_s
        last_state = "UNKNOWN"

        while True:
            remote_file = self.client.files.get(name=file_name)
            state = self._normalize_state(getattr(remote_file, "state", None))
            if state:
                last_state = state

            if state in {"ACTIVE", "STATE_ACTIVE"}:
                return remote_file
            if state in {"FAILED", "STATE_FAILED"}:
                file_error = getattr(remote_file, "error", None)
                raise RuntimeError(
                    f"Gemini file processing failed for {file_name}: {file_error}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Gemini file {file_name} to become ACTIVE "
                    f"(last_state={last_state})."
                )
            time.sleep(self.file_poll_interval_s)

    def analyze_mp4(self, *, video_path: str, prompt: str) -> str:
        video_p = Path(video_path).expanduser().resolve()
        if not video_p.exists():
            raise FileNotFoundError(f"video not found: {video_p}")

        uploaded = self.client.files.upload(file=str(video_p))
        if not uploaded.name:
            raise RuntimeError("Gemini upload did not return a file name.")
        ready_file = self._wait_for_file_active(file_name=uploaded.name)
        # Good practice: file first in the contents list, then the instruction.
        response = self.client.models.generate_content(
            model=self.model,
            contents=[ready_file, prompt],
            config={"automatic_function_calling": {"disable": True}},
        )
        return response.text
