"""Input, content and output guardrails.

- Resume input:  type/size/sanity validation before anything is parsed or sent to a model.
- Job content:   untrusted third-party text. Injection attempts are detected, stripped, and the
                 job is flagged and demoted rather than silently trusted.
- Output:        only http(s) URLs reach the user.
PII redaction for remote models lives in resume/redact.py.
"""
import re
from pathlib import Path
from urllib.parse import urlparse

MAX_RESUME_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
MIN_RESUME_CHARS, MAX_RESUME_CHARS = 150, 60_000
INJECTION_SCORE_CAP = 0.3  # final score ceiling for postings that tried to steer the model


class GuardrailError(ValueError):
    pass


def validate_resume_file(path: str | Path, size: int | None = None) -> None:
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise GuardrailError(f"Unsupported file type {p.suffix!r}; use {sorted(ALLOWED_SUFFIXES)}")
    size = size if size is not None else p.stat().st_size
    if size > MAX_RESUME_BYTES:
        raise GuardrailError(f"Resume too large ({size / 1e6:.1f} MB > 5 MB)")


def validate_resume_text(text: str) -> str:
    text = text.replace("\x00", "").strip()
    if len(text) < MIN_RESUME_CHARS:
        raise GuardrailError("Resume text is too short; scanned PDFs without a text layer are not supported")
    if sum(c.isalpha() for c in text) / len(text) < 0.4:
        raise GuardrailError("Resume text looks like binary or garbled content")
    return text[:MAX_RESUME_CHARS]


_INJECTION = [re.compile(p, re.I) for p in [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above) ",
    r"(reveal|print|show|repeat) (your |the )?(system )?(prompt|instructions)",
    r"you are now\b",
    r"\bnew instructions?\s*:",
    r"(give|assign|rate|score) (this|the) (job|role|candidate|posting)\b.{0,40}\b(10|ten|maximum|highest|perfect)",
    r"(note|message|instructions?) (to|for) (the )?(ai|llm|assistant|model|screening)",
    r"you must (recommend|rank|select|choose)\b.{0,40}\b(this|first|best)",
    r"</?(system|assistant|candidate|job)>",
]]


def scan_injection(text: str) -> list[str]:
    return [m.group(0)[:60] for rx in _INJECTION for m in [rx.search(text)] if m]


def sanitize_job_text(text: str) -> tuple[str, list[str]]:
    """Drop sentences containing injection phrases. Returns (clean_text, matched_snippets)."""
    hits = scan_injection(text)
    if not hits:
        return text, []
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return " ".join(p for p in parts if not scan_injection(p)), hits


def safe_url(url: str) -> bool:
    u = urlparse(url or "")
    return u.scheme in {"http", "https"} and bool(u.netloc)
