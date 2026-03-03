from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class VideoRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoRenderer:
    """Thin wrapper around ffmpeg.

    We deliberately keep video encoding *outside* of Slicer for portability and determinism.
    """

    ffmpeg_path: str = "ffmpeg"

    def ensure_available(self) -> None:
        if shutil.which(self.ffmpeg_path) is None:
            raise VideoRenderError(
                f"ffmpeg not found in PATH (ffmpeg_path={self.ffmpeg_path}).\n"
                "On macOS: brew install ffmpeg\n"
            )

    def encode_mp4_from_pattern(
        self,
        *,
        input_pattern: Path,
        output_path: Path,
        fps: int = 10,
        overwrite: bool = True,
        pix_fmt: str = "yuv420p",
    ) -> Path:
        """Encode MP4 from an image sequence like frame_%05d.png."""
        self.ensure_available()
        input_pattern = Path(input_pattern)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(input_pattern),
            "-pix_fmt",
            pix_fmt,
        ]
        if overwrite:
            cmd.insert(1, "-y")
        else:
            cmd.insert(1, "-n")

        cmd.append(str(output_path))

        logger.info("Running ffmpeg: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise VideoRenderError(f"ffmpeg failed with exit code {e.returncode}") from e

        return output_path
