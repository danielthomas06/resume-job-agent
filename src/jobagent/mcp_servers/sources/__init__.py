from jobagent.config import get_settings

from .adzuna import AdzunaSource
from .base import JobSource
from .boards import GreenhouseSource, LeverSource
from .feeds import ArbeitnowSource, RemoteOKSource, RemotiveSource


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_sources() -> list[JobSource]:
    s = get_settings()
    sources = [
        AdzunaSource(s.adzuna_app_id, s.adzuna_app_key),
        RemotiveSource(), RemoteOKSource(), ArbeitnowSource(),
        GreenhouseSource(_split(s.greenhouse_boards)), LeverSource(_split(s.lever_boards)),
    ]
    return [x for x in sources if x.enabled()]
