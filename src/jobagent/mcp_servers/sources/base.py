from abc import ABC, abstractmethod

import httpx

from jobagent.mcp_servers.schema import Job

UA = {"User-Agent": "jobagent-capstone/0.1 (+personal project)"}


class JobSource(ABC):
    name: str

    def enabled(self) -> bool:
        return True

    @abstractmethod
    async def search(
        self, client: httpx.AsyncClient, query: str, location: str, limit: int
    ) -> list[Job]: ...
