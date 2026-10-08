import asyncio

from jobagent.llm.base import LLMClient, LLMResult
from jobagent.llm.budget import Budget
from jobagent.ui.live import Timeline, render


def step(node, secs=1.0, note="n", attempt=1):
    return {"type": "node_end", "step": {"node": node, "seconds": secs, "note": note, "attempt": attempt}}


def test_budget_emits_start_and_end_events():
    class Fake(LLMClient):
        provider, model = "fake", "m"

        def _generate(self, system, user, schema, temperature):
            return LLMResult(text="x", model="m", provider="fake", input_tokens=10, output_tokens=5, context_window=100)

    events = []
    llm = Fake()
    llm.budget = Budget(on_event=events.append)
    llm.generate("s", "u", purpose="judge_fit")
    assert [e["type"] for e in events] == ["llm_start", "llm_end"]
    assert events[1]["totals"] == {"calls": 1, "prompt": 10, "completion": 5}


def test_broken_listener_never_breaks_a_call():
    def boom(_):
        raise RuntimeError("ui died")

    b = Budget(on_event=boom)
    b.started("x", 1, "m")  # must not raise


def test_timeline_progress_and_render():
    tl = Timeline()
    assert tl.fraction == 0
    tl.handle({"type": "node_start", "node": "extract_profile", "attempt": 0})
    tl.handle({"type": "llm_start", "purpose": "extract_profile", "attempt": 1, "model": "qwen"})
    _, md = render(tl)
    assert "waiting on qwen" in md and "⏳" in md
    tl.handle(step("extract_profile"))
    assert 0 < tl.fraction < 0.3
    tl.handle({"type": "node_start", "node": "llm_rerank_and_critic", "attempt": 1})
    tl.handle({"type": "job_judged", "i": 5, "total": 10, "title": "ML Engineer", "company": "A",
               "llm_score": 8, "final": 0.8, "flags": 1})
    frac, md = render(tl)
    assert frac > 0.4 and "5/10 judged" in md and "ML Engineer" in md and "1 flag" in md


def test_repeated_node_does_not_double_count_progress():
    tl = Timeline()
    for _ in range(2):
        tl.handle(step("plan_queries"))
    assert abs(tl.fraction - 0.05 / sum(__import__("jobagent.ui.live", fromlist=["WEIGHTS"]).WEIGHTS.values())) < 1e-9


def test_pipeline_listener_reset_after_run():
    import jobagent.pipeline.graph as g
    assert g._listener is None  # no leak between runs
    assert asyncio.iscoroutinefunction(g.run_pipeline)
