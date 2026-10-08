"""Company career boards: Greenhouse and Lever public APIs (best source for real tech roles)."""
import asyncio
from datetime import UTC, datetime

import httpx

from jobagent.mcp_servers.schema import Job, clean_html, matches

from .base import UA, JobSource


class GreenhouseSource(JobSource):
    name = "greenhouse"

    def __init__(self, boards: list[str]):
        self.boards = boards

    def enabled(self):
        return bool(self.boards)

    async def _board(self, client, token) -> list[Job]:
        try:
            r = await client.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
                                 params={"content": "true"}, headers=UA)
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            Job(id=f"greenhouse:{token}:{j['id']}", source=self.name, title=j["title"], company=token,
                location=j.get("location", {}).get("name", ""), url=j["absolute_url"],
                description=clean_html(j.get("content", "")), posted_at=j.get("updated_at", ""))
            for j in r.json().get("jobs", [])
        ]

    async def search(self, client, query, location, limit):
        jobs = [j for b in await asyncio.gather(*(self._board(client, t) for t in self.boards)) for j in b]
        return [j for j in jobs if matches(j, query)][:limit]


class LeverSource(JobSource):
    name = "lever"

    def __init__(self, boards: list[str]):
        self.boards = boards

    def enabled(self):
        return bool(self.boards)

    async def _board(self, client, company) -> list[Job]:
        try:
            r = await client.get(f"https://api.lever.co/v0/postings/{company}",
                                 params={"mode": "json"}, headers=UA)
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            Job(id=f"lever:{company}:{j['id']}", source=self.name, title=j["text"], company=company,
                location=j.get("categories", {}).get("location", ""), url=j["hostedUrl"],
                description=clean_html(j.get("descriptionPlain", "")),
                posted_at=datetime.fromtimestamp(j["createdAt"] / 1000, UTC).isoformat()
                if j.get("createdAt") else "")
            for j in r.json()
        ]

    async def search(self, client, query, location, limit):
        jobs = [j for b in await asyncio.gather(*(self._board(client, t) for t in self.boards)) for j in b]
        return [j for j in jobs if matches(j, query)][:limit]
