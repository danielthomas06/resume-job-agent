import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("URL", re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)\S+", re.I)),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,16}\d(?![\w])")),
]


def redact_pii(text: str) -> str:
    """Mask contact details before text leaves the machine. Name stays; extraction needs it little."""
    for label, rx in _PATTERNS:
        text = rx.sub(f"[{label}]", text)
    return text
