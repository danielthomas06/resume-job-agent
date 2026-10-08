import json

import numpy as np
from opentelemetry import trace

from jobagent import runs
from jobagent.llm.base import LLMClient, LLMResult
from jobagent.mcp_servers.schema import Job
from jobagent.models import Profile
from jobagent.observability import (
    current_trace_id,
    extract_context,
    inject_context,
    setup_tracing,
    tracer,
)
from jobagent.pipeline import rank_stage


def test_trace_context_survives_process_boundary():
    setup_tracing("")  # global provider, no exporter
    with tracer().start_as_current_span("agent") as parent:
        carrier = inject_context()  # what the MCP client puts in _meta
        tid = current_trace_id()
    assert "traceparent" in carrier and tid
    # server side: rebuild the context from _meta and start a child span
    with tracer().start_as_current_span("server", context=extract_context(carrier)) as child:
        assert child.get_span_context().trace_id == parent.get_span_context().trace_id
        assert trace.format_trace_id(child.get_span_context().trace_id) == tid


def test_run_records_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path)
    rep = {"run_id": "abc", "trace_id": "t" * 32, "profile": {"name": "A"}, "jobs": [{"job": {"title": "ML"}}],
           "meta": {"provider": "ollama", "model": "m"}}
    runs.save_run(rep)
    assert runs.load_run("abc")["trace_id"] == "t" * 32
    assert runs.list_runs()[0]["top"] == "ML"


class FakeLLM(LLMClient):
    provider, model = "fake", "fake"

    def _generate(self, system, user, schema, temperature):
        return LLMResult(text=json.dumps({"fit_score": 8, "seniority_fit": "good", "reason": "Uses Python.",
                                          "matched_skills": ["Python"], "missing_skills": []}),
                         model="f", provider="f")


def test_ranked_jobs_carry_feature_breakdown_and_score_trail(monkeypatch):
    monkeypatch.setattr(rank_stage, "embed", lambda texts, kind="document": np.tile(
        np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)))
    jobs = [Job(id=f"s:{i}", source="s", title="ML Engineer", company=f"C{i}", url="https://x.io/1",
                description="Python role") for i in range(2)]
    p = Profile(skills=["Python"], target_titles=["ML Engineer"])
    ranked = rank_stage.llm_rerank(p, rank_stage.prelim_rank(p, jobs), FakeLLM(), top_n=2)
    r = ranked[0]
    assert set(r.features) >= {"cosine", "emb_norm", "skill_cov", "title_match", "score"}
    assert [t.split()[0] for t in r.trail] == ["prelim", "llm", "blend", "final"]


def test_budget_logs_context_use_and_flags_truncation():
    class Tiny(LLMClient):
        provider, model = "fake", "fake"

        def _generate(self, system, user, schema, temperature):
            return LLMResult(text="{}", model="f", provider="f", input_tokens=99, output_tokens=2,
                             context_window=100, finish_reason="stop")

    from jobagent.llm.budget import Budget
    llm = Tiny()
    llm.budget = Budget()
    llm.generate("sys", "user", purpose="judge_fit")
    row = llm.budget.summary()["calls"][0]
    assert row["purpose"] == "judge_fit" and row["truncated"] and row["context_utilization"] == 1.01
    assert llm.budget.summary()["truncated_calls"] == 1
