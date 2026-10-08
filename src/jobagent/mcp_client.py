"""Thin MCP client: spawns jobs-mcp over stdio and calls its tools (traced)."""
import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jobagent.observability import inject_context, tracer


class JobsClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def call(self, tool: str, **args):
        with tracer().start_as_current_span(f"mcp.{tool}") as span:
            span.set_attribute("openinference.span.kind", "TOOL")
            span.set_attribute("tool.name", tool)
            span.set_attribute("input.value", json.dumps(args))
            res = await self.session.call_tool(tool, args, meta=inject_context())  # propagate trace context
            if res.isError:
                raise RuntimeError(f"MCP tool {tool} failed: {res.content}")
            data = res.structuredContent
            if data is None:
                data = [json.loads(c.text) for c in res.content if c.type == "text"]
            else:
                data = data.get("result", data)
            span.set_attribute("output.value", json.dumps(data)[:4000])
            return data


@asynccontextmanager
async def jobs_client():
    params = StdioServerParameters(command=sys.executable, args=["-m", "jobagent.mcp_servers.jobs_server"],
                                   env=dict(os.environ))  # default is a minimal env: API keys from the environment would be lost
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        yield JobsClient(session)
