import httpx
import pytest
import respx

from jobagent.mcp_servers.schema import Job, clean_html, matches
from jobagent.mcp_servers.sources.adzuna import AdzunaSource
from jobagent.mcp_servers.sources.feeds import ArbeitnowSource


def test_clean_html_and_matches():
    assert clean_html("&lt;p&gt;Hi   <b>there</b>&lt;/p&gt;") == "Hi there"
    job = Job(id="x:1", source="x", title="ML Engineer", url="u", description="Python and PyTorch")
    assert matches(job, "ml pytorch") and not matches(job, "rust")


@pytest.mark.asyncio
@respx.mock
async def test_adzuna_maps_india():
    route = respx.get(url__regex=r"https://api\.adzuna\.com/v1/api/jobs/in/search/1.*").mock(
        return_value=httpx.Response(200, json={"results": [{
            "id": "7", "title": "ML Engineer", "company": {"display_name": "Acme"},
            "location": {"display_name": "Hyderabad"}, "redirect_url": "http://a/7",
            "description": "<b>Build</b> models", "created": "2026-10-01", "salary_min": 1.0}]}))
    async with httpx.AsyncClient() as c:
        jobs = await AdzunaSource("id", "key").search(c, "ml", "india", 5)
    assert route.called and jobs[0].id == "adzuna:7" and jobs[0].currency == "INR"
    assert jobs[0].description == "Build models"


@pytest.mark.asyncio
@respx.mock
async def test_arbeitnow_filters_client_side():
    respx.get("https://www.arbeitnow.com/api/job-board-api").mock(return_value=httpx.Response(
        200, json={"data": [
            {"slug": "a", "title": "ML Engineer", "url": "u", "created_at": 1700000000},
            {"slug": "b", "title": "Baker", "url": "u"}]}))
    async with httpx.AsyncClient() as c:
        jobs = await ArbeitnowSource().search(c, "ml engineer", "", 10)
    assert [j.id for j in jobs] == ["arbeitnow:a"]
