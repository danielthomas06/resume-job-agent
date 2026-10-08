import asyncio

from jobagent.mcp_servers.schema import Job, clean_html

from .base import JobSource

# Adzuna country codes we map location hints to; default India.
COUNTRIES = {"india": "in", "us": "us", "usa": "us", "united states": "us", "uk": "gb",
             "united kingdom": "gb", "canada": "ca", "australia": "au", "germany": "de",
             "singapore": "sg"}


class AdzunaSource(JobSource):
    name = "adzuna"
    _sem: asyncio.Semaphore | None = None  # free tier rate-limits bursts; created lazily per loop

    def __init__(self, app_id: str, app_key: str):
        self.app_id, self.app_key = app_id, app_key

    def enabled(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def search(self, client, query, location, limit) -> list[Job]:
        loc = location.lower()
        country = next((c for k, c in COUNTRIES.items() if k in loc), "in")
        where = "" if loc in COUNTRIES or loc == "remote" else location
        params = {"app_id": self.app_id, "app_key": self.app_key, "what": query,
                  "results_per_page": min(limit, 50), "content-type": "application/json"}
        if where:
            params["where"] = where
        AdzunaSource._sem = AdzunaSource._sem or asyncio.Semaphore(2)
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        async with AdzunaSource._sem:
            for attempt in range(3):
                r = await client.get(url, params=params)
                if r.status_code != 429:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
        r.raise_for_status()
        return [
            Job(id=f"adzuna:{j['id']}", source=self.name, title=clean_html(j["title"], 200),
                company=j.get("company", {}).get("display_name", ""),
                location=j.get("location", {}).get("display_name", ""), url=j["redirect_url"],
                description=clean_html(j.get("description", "")), posted_at=j.get("created", ""),
                salary_min=j.get("salary_min"), salary_max=j.get("salary_max"),
                currency="INR" if country == "in" else "")
            for j in r.json().get("results", [])
        ]
