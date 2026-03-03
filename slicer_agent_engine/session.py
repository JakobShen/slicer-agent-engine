from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SessionManager:
    """Holds viewer state and writes an append-only trace (JSONL).

    The trace is intended to be:
    - replayable (tool name + args + selected state)
    - auditable (timestamps, errors, artifacts)
    """

    out_dir: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.out_dir / "trace.jsonl"
        self.state_path = self.out_dir / "state.json"

    def update_state(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        self._flush_state()

    def _flush_state(self) -> None:
        tmp = dict(self.state)
        tmp["session_id"] = self.session_id
        tmp["updated_ts"] = time.time()
        self.state_path.write_text(json.dumps(tmp, indent=2, ensure_ascii=False), encoding="utf-8")

    def log_event(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        ok: bool = True,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "ts": time.time(),
            "session_id": self.session_id,
            "tool": tool,
            "args": args,
            "ok": ok,
            "error": error,
            "result": result,
            "artifacts": artifacts or {},
            "state": dict(self.state),
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
