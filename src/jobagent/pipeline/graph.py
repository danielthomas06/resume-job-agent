"""LangGraph pipeline: extract -> plan -> search -> dedupe -> (broaden?) -> prelim -> rerank -> critic -> report."""
import asyncio
import functools
import hashlib
import json
import operator
import re
import time
import uuid
from collections import Counter
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from jobagent import __version__
from jobagent.config import get_settings
from jobagent.guardrails import validate_resume_text
from jobagent.llm import LLMClient
from jobagent.llm.budget import Budget
from jobagent.mcp_client import jobs_client
from jobagent.mcp_servers.schema import Job
from jobagent.models import Profile
from jobagent.observability import current_trace_id, trace_url, tracer
from jobagent.pipeline import ranking as rk
from jobagent.pipeline.rank_stage import RERANK_TOP_N, llm_rerank, prelim_rank
from jobagent.resume import extract_profile
from jobagent.runs import save_run

_listener = None  # per-run progress callback, set by run_pipeline


def _emit(event: dict) -> None:
    if _listener:
        try:
            _listener(event)
        except Exception:  # progress display must never break the run
            pass


MIN_POOL = 15       # below this after dedupe, broaden the queries once


class State(TypedDict, total=False):
    resume_text: str
    locations: list[str]
    profile: Profile
    queries: list[str]
    attempt: int
    pool: list[Job]
    ranked: list[rk.Ranked]
    report: dict
    plan_rationale: str
    steps: Annotated[list[dict], operator.add]  # agent timeline: one entry per node execution


_SENIORITY = re.compile(r"\b(mid|junior|senior|sr|jr|lead|principal|staff|entry[- ]level)\b[- ]?", re.I)


def _strip_seniority(q: str) -> str:
    return _SENIORITY.sub("", q).strip()


class Plan(BaseModel):
    queries: list[str]
    rationale: str = ""  # why these titles: shown to the user as the agent's stated reasoning


def _state_view(state) -> dict:
    """Small, serialisable view of what a node was given (full state is too large for a span)."""
    return {"attempt": state.get("attempt"), "queries": state.get("queries"),
            "pool_size": len(state["pool"]) if "pool" in state else None,
            "ranked": len(state["ranked"]) if "ranked" in state else None}


def traced(name, note=None):  # emits live events through state["_emit"] when present
    """Node span + timeline entry. `note(out, state) -> str` is the one-line 'what happened'."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state, *a, **kw):
            with tracer().start_as_current_span(f"node.{name}") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("input.value", json.dumps(_state_view(state), default=str))
                t0 = time.perf_counter()
                _emit({"type": "node_start", "node": name, "attempt": state.get("attempt")})
                out = fn(state, *a, **kw)
                out = await out if asyncio.iscoroutine(out) else out
                text = note(out, state) if note else ""
                span.set_attribute("output.value", text)
                step = {"node": name, "seconds": round(time.perf_counter() - t0, 2), "note": text,
                        "attempt": state.get("attempt")}
                _emit({"type": "node_end", "step": step})
                return {**out, "steps": [step]}
        return wrapper
    return deco


def build_graph(llm: LLMClient, redact_remote: bool = True):
    @traced("extract_profile", note=lambda o, s: (
        f"{o['profile'].seniority}, {o['profile'].total_years_experience}y, {len(o['profile'].skills)} skills; "
        f"target titles {o['profile'].target_titles}"))
    def n_extract(s: State):
        return {"profile": extract_profile(s["resume_text"], llm, redact_remote), "attempt": 0}

    @traced("plan_queries", note=lambda o, s: f"queries={o['queries']}; {o['plan_rationale']}")
    def n_plan(s: State):
        p, attempt = s["profile"], s.get("attempt", 0)
        broaden = ("Previous searches returned too few jobs. Use broader, more generic job-board "
                   "keywords (1-2 words each).") if attempt else ""
        try:
            plan = llm.generate_structured(
                "You write job-board search queries: JOB TITLES only, 2-4 words each (e.g. 'Machine Learning Engineer'). "
                "Never include seniority words (mid, senior, junior) or tool lists. "
                "Return 3-4 diverse titles covering the candidate's best-fit roles, and a one-or-two sentence "
                "rationale for the choice. " + broaden,
                f"Seniority: {p.seniority}\nTarget titles: {p.target_titles}\nSkills: {p.skills[:12]}\n"
                f"Domains: {p.domains}", Plan, purpose="plan_queries")
            queries = [_strip_seniority(q) for q in plan.queries if q.strip()][:4]
            rationale = plan.rationale or "(no rationale given)"
        except Exception:
            queries, rationale = [], "LLM planning failed; fell back to profile titles/skills"
        queries = queries or (p.target_titles[:3] if not attempt else p.skills[:3]) or ["software engineer"]
        return {"queries": queries, "attempt": attempt + 1, "plan_rationale": rationale + (" [broadened]" if attempt else "")}

    @traced("search", note=lambda o, s: f"{len(o['pool']) - len(s.get('pool', []))} jobs from {len(s['queries'])} queries")
    async def n_search(s: State):
        locs = s.get("locations") or ["india"]
        # all sources once per query (feeds are global/remote); extra locations only matter to Adzuna
        calls = [dict(query=q, location=locs[0], limit=15) for q in s["queries"]]
        calls += [dict(query=q, location=loc, limit=15, sources=["adzuna"])
                  for q in s["queries"] for loc in locs[1:]]
        async with jobs_client() as c:
            batches = await asyncio.gather(*(c.call("search_jobs", **a) for a in calls),
                                           return_exceptions=True)
        jobs = [Job(**j) for b in batches if not isinstance(b, Exception) for j in b]
        return {"pool": s.get("pool", []) + jobs}

    @traced("dedupe", note=lambda o, s: (
        f"{len(o['pool'])} unique of {len(s['pool'])}; next = "
        + ("broaden queries" if len(o["pool"]) < MIN_POOL and s["attempt"] < 2 else "rank")))
    def n_dedupe(s: State):
        return {"pool": rk.dedupe(s["pool"])}

    def route_after_dedupe(s: State):
        return "plan" if len(s["pool"]) < MIN_POOL and s["attempt"] < 2 else "rank"

    @traced("prelim_rank", note=lambda o, s: f"scored {len(o['ranked'])}; top: " + "; ".join(r.job.title for r in o["ranked"][:3]))
    def n_prelim(s: State):
        return {"ranked": prelim_rank(s["profile"], s["pool"])}

    @traced("llm_rerank_and_critic", note=lambda o, s: (
        f"LLM judged {sum(r.llm_score is not None for r in o['ranked'])}; "
        f"critic flags on {sum(any(f.startswith(('ungrounded', 'dropped')) for f in r.flags) for r in o['ranked'])} jobs"))
    def n_rerank(s: State):
        def progress(i, total, r):
            _emit({"type": "job_judged", "i": i, "total": total, "title": r.job.title, "company": r.job.company,
                   "llm_score": r.llm_score, "final": round(r.final, 2),
                   "flags": len(r.flags)})
        return {"ranked": llm_rerank(s["profile"], s["ranked"], llm, RERANK_TOP_N, progress)}

    @traced("report")
    def n_report(s: State):
        top = s["ranked"][:RERANK_TOP_N]
        gaps = Counter(m for r in top if r.judgement for m in r.judgement.missing_skills)
        return {"report": {"profile": s["profile"].model_dump(), "queries": s["queries"],
                           "pool_size": len(s["pool"]),
                           "top_skill_gaps": gaps.most_common(8),
                           "plan_rationale": s.get("plan_rationale", ""), "steps": s.get("steps", []),
                           "jobs": [r.model_dump() for r in top]}}

    g = StateGraph(State)
    for name, fn in [("extract", n_extract), ("plan", n_plan), ("search", n_search),
                     ("dedupe", n_dedupe), ("rank", n_prelim), ("rerank", n_rerank), ("report", n_report)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract")
    g.add_edge("extract", "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "dedupe")
    g.add_conditional_edges("dedupe", route_after_dedupe, {"plan": "plan", "rank": "rank"})
    g.add_edge("rank", "rerank")
    g.add_edge("rerank", "report")
    g.add_edge("report", END)
    return g.compile()


async def run_pipeline(resume_text: str, llm: LLMClient, locations: list[str] | None = None,
                       redact_remote: bool = True, save: bool = True, on_event=None) -> dict:
    resume_text = validate_resume_text(resume_text)
    global _listener
    llm.budget = Budget(on_event=on_event)  # fresh per run: caps calls/tokens and meters usage
    _listener = on_event
    run_id = uuid.uuid4().hex[:12]
    cfg = get_settings()
    meta = {
        "run_id": run_id, "provider": llm.provider, "model": llm.model, "embed_model": cfg.embed_model,
        "redact_remote": redact_remote, "locations": locations or ["india"],
        "resume_sha256": hashlib.sha256(resume_text.encode()).hexdigest()[:16],  # identity without storing content
        "weights": {"emb": rk.W_EMB, "skill": rk.W_SKILL, "title": rk.W_TITLE, "prelim": rk.W_PRELIM, "llm": rk.W_LLM},
        "rerank_top_n": RERANK_TOP_N, "app_version": __version__,
    }
    with tracer().start_as_current_span("pipeline.run") as span:
        span.set_attribute("openinference.span.kind", "AGENT")
        span.set_attribute("session.id", run_id)
        for k, v in meta.items():
            span.set_attribute(f"run.{k}", json.dumps(v) if isinstance(v, (dict, list)) else v)
        trace_id = current_trace_id()
        try:
            out = await build_graph(llm, redact_remote).ainvoke(
                {"resume_text": resume_text, "locations": locations or []})
        finally:
            _listener = None
        rep = out["report"]
        rep["usage"] = llm.budget.summary()
        span.set_attribute("run.peak_context_utilization", rep["usage"]["peak_context_utilization"] or 0)
        span.set_attribute("run.truncated_calls", rep["usage"]["truncated_calls"])
        span.set_attribute("run.llm_calls", llm.budget.calls)
        span.set_attribute("run.tokens", llm.budget.input_tokens + llm.budget.output_tokens)
        span.set_attribute("run.pool_size", rep["pool_size"])
        span.set_attribute("output.value", json.dumps([r["job"]["title"] for r in rep["jobs"][:5]]))
    rep |= {"run_id": run_id, "trace_id": trace_id, "trace_url": trace_url(trace_id), "meta": meta}
    if save:
        save_run(rep)
    return rep
