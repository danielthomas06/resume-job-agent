import re
from html import unescape

from pydantic import BaseModel


class Job(BaseModel):
    id: str  # "<source>:<native id>"
    source: str
    title: str
    company: str = ""
    location: str = ""
    remote: bool | None = None
    url: str
    description: str = ""
    posted_at: str = ""
    salary_min: float | None = None
    salary_max: float | None = None
    currency: str = ""
    tags: list[str] = []


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(text: str, limit: int = 4000) -> str:
    return _WS.sub(" ", _TAG.sub(" ", unescape(unescape(text or "")))).strip()[:limit]


def matches(job: Job, query: str) -> bool:
    """Keyword filter for feeds with no server-side search: every query token must appear."""
    hay = f"{job.title} {job.company} {' '.join(job.tags)} {job.description}".lower()
    return all(tok in hay for tok in query.lower().split())
