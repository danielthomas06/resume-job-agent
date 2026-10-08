"""jobs-mcp: MCP server exposing job search over many sources behind one normalized schema.

Runs as a subprocess of the agent. The client sends W3C trace context in MCP `_meta`, so server-side
spans (tool call, per-source HTTP) join the agent's trace instead of starting an orphan one.
"""
import asyncio
import logging
import time

import httpx
from mcp.server.fastmcp import Context, FastMCP

from jobagent.config import get_settings
from jobagent.mcp_servers.schema import Job
from jobagent.mcp_servers.sources import build_sources
from jobagent.observability import extract_context, flush, setup_tracing, tracer

log = logging.getLogger("jobs-mcp")
mcp = FastMCP("jobs-mcp")
_cache: dict[str, Job] = {}  # job id -> Job, populated by searches for get_job_details


def _parent(ctx: Context | None):
    """Trace context the client put in the request's _meta (None if absent)."""
    try:
        meta = ctx.request_context.meta if ctx else None
        carrier = (meta.model_extra or {}) if meta else {}
        return extract_context({k: v for k, v in carrier.items() if isinstance(v, str)})
    except Exception:  # tracing must never break a tool call
        return None


@mcp.tool()
def list_sources(ctx: Context) -> list[str]:
    """List the job sources that are currently enabled."""
    with tracer().start_as_current_span("mcp.server.list_sources", context=_parent(ctx)):
        return [s.name for s in build_sources()]


async def _run_source(s, client, query, location, limit) -> list[Job] | Exception:
    with tracer().start_as_current_span(f"source.{s.name}") as span:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("source.name", s.name)
        span.set_attribute("input.value", f"{query} @ {location}")
        t0 = time.perf_counter()
        try:
            jobs = await s.search(client, query, location, limit)
            span.set_attribute("source.results", len(jobs))
            return jobs
        except Exception as e:  # one failing source must not break the others
            span.record_exception(e)
            status = getattr(getattr(e, "response", None), "status_code", "")
            span.set_attribute("source.error", f"{type(e).__name__} {status}".strip())
            log.warning("source %s failed: %s %s", s.name, type(e).__name__, status)  # no str(e): URL has key
            return e
        finally:
            span.set_attribute("source.latency_s", round(time.perf_counter() - t0, 3))


@mcp.tool()
async def search_jobs(
    ctx: Context, query: str, location: str = "", sources: list[str] | None = None, limit: int = 10
) -> list[dict]:
    """Search jobs across sources.

    query: keywords, e.g. 'machine learning engineer'. location: country/city/'remote'
    (India is the default for Adzuna). sources: optional subset of list_sources().
    Returns normalized jobs (descriptions truncated to 600 chars; use get_job_details for full).
    """
    limit = max(1, min(limit, 30))
    with tracer().start_as_current_span("mcp.server.search_jobs", context=_parent(ctx)) as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        active = [s for s in build_sources() if not sources or s.name in sources]
        span.set_attribute("sources", ",".join(s.name for s in active))
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            results = await asyncio.gather(*(_run_source(s, client, query, location, limit) for s in active))
        jobs = [j for res in results if not isinstance(res, Exception) for j in res]
        span.set_attribute("jobs.returned", len(jobs))
        span.set_attribute("sources.failed", sum(isinstance(r, Exception) for r in results))
        for j in jobs:
            _cache[j.id] = j
        out = [{**j.model_dump(exclude={"description"}), "description": j.description[:600]} for j in jobs]
    flush()  # the process may be killed when the client disconnects; don't lose spans
    return out


@mcp.tool()
def get_job_details(ctx: Context, job_id: str) -> dict:
    """Full details for a job id returned by search_jobs in this server session."""
    with tracer().start_as_current_span("mcp.server.get_job_details", context=_parent(ctx)):
        job = _cache.get(job_id)
        return job.model_dump() if job else {"error": f"unknown job id {job_id}; run search_jobs first"}


def main():
    s = get_settings()
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s [trace=%(trace_id)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # URLs carry the Adzuna key
    setup_tracing(s.otel_exporter_otlp_endpoint, service_name="jobs-mcp", project=s.otel_project, public_url=s.phoenix_public_url)
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
