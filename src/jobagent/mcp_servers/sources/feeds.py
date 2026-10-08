"""Keyless public feeds: Remotive, RemoteOK, Arbeitnow."""
from datetime import UTC, datetime

from jobagent.mcp_servers.schema import Job, clean_html, matches

from .base import UA, JobSource


class RemotiveSource(JobSource):
    name = "remotive"

    async def search(self, client, query, location, limit):
        r = await client.get("https://remotive.com/api/remote-jobs",
                             params={"search": query, "limit": limit}, headers=UA)
        r.raise_for_status()
        return [
            Job(id=f"remotive:{j['id']}", source=self.name, title=j["title"],
                company=j.get("company_name", ""), location=j.get("candidate_required_location", ""),
                remote=True, url=j["url"], description=clean_html(j.get("description", "")),
                posted_at=j.get("publication_date", ""), tags=j.get("tags", []))
            for j in r.json().get("jobs", [])
        ]


class RemoteOKSource(JobSource):
    name = "remoteok"

    async def search(self, client, query, location, limit):
        r = await client.get("https://remoteok.com/api", headers=UA)
        r.raise_for_status()
        out = []
        for j in r.json():
            if "position" not in j:  # first element is a legal notice
                continue
            job = Job(id=f"remoteok:{j['id']}", source=self.name, title=j["position"],
                      company=j.get("company", ""), location=j.get("location", "Remote"),
                      remote=True, url=j.get("url", ""), description=clean_html(j.get("description", "")),
                      posted_at=j.get("date", ""), tags=j.get("tags", []),
                      salary_min=j.get("salary_min") or None, salary_max=j.get("salary_max") or None,
                      currency="USD")
            if matches(job, query):
                out.append(job)
        return out[:limit]


class ArbeitnowSource(JobSource):
    name = "arbeitnow"

    async def search(self, client, query, location, limit):
        r = await client.get("https://www.arbeitnow.com/api/job-board-api", headers=UA)
        r.raise_for_status()
        out = []
        for j in r.json().get("data", []):
            ts = j.get("created_at")
            job = Job(id=f"arbeitnow:{j['slug']}", source=self.name, title=j["title"],
                      company=j.get("company_name", ""), location=j.get("location", ""),
                      remote=j.get("remote"), url=j["url"], description=clean_html(j.get("description", "")),
                      posted_at=datetime.fromtimestamp(ts, UTC).isoformat() if ts else "",
                      tags=j.get("tags", []))
            if matches(job, query):
                out.append(job)
        return out[:limit]
