from __future__ import annotations

import json
import re
from typing import Dict, Optional, Sequence, Set


_GENERIC_ANSWER_RE = re.compile(r"\b(?:ANSWER|CHOICE|OPTION)\b\s*[:\-]?\s*([A-Z])\b", re.IGNORECASE)
_NUMBERED_TOKEN_RE = re.compile(r"([1-9])\s*[:=\-\)]?\s*([A-Z])", re.IGNORECASE)


def extract_choice_from_set(text: str, *, valid_choices: Sequence[str]) -> Optional[str]:
    """Extract a multiple-choice answer from free-form model text.

    Supported patterns:
    - JSON containing {"choice": "A"} or {"answer": "B"}
    - Lines like "ANSWER: C" or "Option D"
    - A final line that is just one valid choice token
    """

    if not text:
        return None

    allowed: Set[str] = {str(choice).strip().upper() for choice in valid_choices if str(choice).strip()}
    if not allowed:
        return None

    # 1) Try JSON (common for structured prompting).
    try:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                for k in ("choice", "answer", "label"):
                    v = obj.get(k)
                    if isinstance(v, str):
                        token = v.strip().upper()
                        if token in allowed:
                            return token
    except Exception:
        pass

    # 2) Regex like "ANSWER: A"
    m = _GENERIC_ANSWER_RE.search(text)
    if m:
        token = m.group(1).upper()
        if token in allowed:
            return token

    # 3) Final-line fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines[-10:]):
        token = ln.strip().upper()
        if token in allowed:
            return token

    return None


def extract_choice(text: str) -> Optional[str]:
    """Backward-compatible A/B/C extractor."""

    return extract_choice_from_set(text, valid_choices=("A", "B", "C"))


def extract_numbered_answers(text: str, *, question_ids: Sequence[int]) -> Optional[str]:
    """Extract compact numbered answers like ``1C2D3B4A``.

    Accepts loose formats such as:
      - 1C2D3B4A
      - 1:C 2:D 3:B 4:A
      - 1) C, 2) D, 3) B, 4) A
    """

    if not text:
        return None

    qset = {str(int(q)) for q in question_ids}
    found: Dict[str, str] = {}

    for q, ans in _NUMBERED_TOKEN_RE.findall(text.upper()):
        if q in qset and ans.isalpha():
            found[q] = ans

    if not found:
        compact = re.sub(r"\s+", "", text.upper())
        for q, ans in _NUMBERED_TOKEN_RE.findall(compact):
            if q in qset and ans.isalpha():
                found[q] = ans

    if not found:
        return None

    try:
        ordered = ''.join(f"{q}{found[str(q)]}" for q in question_ids if str(q) in found)
    except Exception:
        return None

    return ordered if ordered else None
