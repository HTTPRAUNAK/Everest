"""Small helpers shared by the agent nodes."""
from __future__ import annotations

import re
import uuid
from datetime import date

NOT_FOUND_MESSAGE = "I could not find this information in the authorised university sources."

_STUDENT_ID = re.compile(r"\bS\d{4}\b", re.IGNORECASE)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def today() -> date:
    return date.today()


def find_student_ids(text: str) -> list[str]:
    """Student IDs mentioned inside message text (used by guard to refuse other-student requests)."""
    return sorted({m.upper() for m in _STUDENT_ID.findall(text or "")})


def clean_untrusted(text: str, max_chars: int = 1500) -> str:
    """Document text is DATA (R8). Strip control/zero-width chars and angle brackets so it
    can't forge prompt delimiters like </doc> or role tags, then cap its length."""
    text = re.sub(r"[\u0000-\u0008\u000b-\u001f\u200b-\u200f\u2028-\u202e\ufeff]", "", text or "")
    text = text.replace("<", "(").replace(">", ")")
    return " ".join(text.split())[:max_chars]
