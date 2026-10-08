"""OpenTelemetry setup + helpers for run-level traceability.

Spans use OpenInference attributes so Phoenix renders them as LLM/tool/agent spans. Trace context is
propagated into the jobs-mcp subprocess (MCP `_meta`), so one run is ONE trace across both processes.
"""
import logging
from functools import lru_cache

import httpx
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider: TracerProvider | None = None
_endpoint = ""
_project = "jobagent"
_public = ""  # URL used for links shown to people


def setup_tracing(endpoint: str = "", service_name: str = "jobagent", project: str = "jobagent",
                  public_url: str = "") -> None:
    global _provider, _endpoint, _project, _public
    if _provider:
        return
    _endpoint, _project = endpoint.rstrip("/"), project
    _public = (public_url or endpoint).rstrip("/")
    _provider = TracerProvider(resource=Resource.create({
        "service.name": service_name,
        "openinference.project.name": project,  # Phoenix groups traces into projects by this attribute
    }))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_endpoint + "/v1/traces")))
    trace.set_tracer_provider(_provider)
    _install_log_correlation()


def _install_log_correlation() -> None:
    """Every log record carries the active trace id, so logs can be joined to traces."""
    old = logging.getLogRecordFactory()

    def factory(*a, **kw):
        rec = old(*a, **kw)
        ctx = trace.get_current_span().get_span_context()
        rec.trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else "-"
        return rec

    logging.setLogRecordFactory(factory)


def tracer() -> trace.Tracer:
    return trace.get_tracer("jobagent")


def flush(timeout_ms: int = 5000) -> None:
    if _provider:
        _provider.force_flush(timeout_ms)


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""


def inject_context() -> dict[str, str]:
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_context(carrier: dict | None) -> otel_context.Context | None:
    return extract(carrier) if carrier else None


@lru_cache(maxsize=8)
def _phoenix_project_id(endpoint: str, name: str) -> str | None:
    try:
        r = httpx.post(f"{endpoint}/graphql", timeout=1.5, json={
            "query": "{projects{edges{node{id name}}}}"})
        return next((e["node"]["id"] for e in r.json()["data"]["projects"]["edges"]
                     if e["node"]["name"] == name), None)
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def trace_url(trace_id: str, project: str | None = None) -> str:
    """Deep link to a trace in Phoenix; falls back to the project page if the id can't be resolved."""
    if not _endpoint or not trace_id:
        return ""
    pid = _phoenix_project_id(_endpoint, project or _project)
    if not pid:
        return f"{_public}/projects"
    return f"{_public}/projects/{pid}/traces/{trace_id}"
