"""Live view of a running pipeline. `Timeline` folds pipeline events into state; `render` is pure
(state -> progress fraction + markdown) so it can be tested without Streamlit."""
from dataclasses import dataclass, field

# rough share of wall-clock each stage takes locally; rerank dominates and reports sub-progress
WEIGHTS = {"extract_profile": 0.15, "plan_queries": 0.05, "search": 0.08, "dedupe": 0.01,
           "prelim_rank": 0.05, "llm_rerank_and_critic": 0.62, "report": 0.0}
LABELS = {"extract_profile": "Read resume → profile", "plan_queries": "Plan searches",
          "search": "Search job sources (MCP)", "dedupe": "Deduplicate", "prelim_rank": "Embed + first-pass rank",
          "llm_rerank_and_critic": "LLM judge + critic", "report": "Build report"}


@dataclass
class Timeline:
    done: list[dict] = field(default_factory=list)
    current: str | None = None
    attempt: int | None = None
    llm: dict | None = None  # in-flight LLM request
    judged: list[dict] = field(default_factory=list)
    totals: dict = field(default_factory=lambda: {"calls": 0, "prompt": 0, "completion": 0})
    peak_ctx: float = 0.0
    truncated: int = 0

    def handle(self, e: dict) -> None:
        t = e["type"]
        if t == "node_start":
            self.current, self.attempt = e["node"], e.get("attempt")
        elif t == "node_end":
            self.done.append(e["step"])
            self.current, self.llm = None, None
        elif t == "llm_start":
            self.llm = e
        elif t == "llm_end":
            self.llm = None
            self.totals = e["totals"]
            u = e["row"]["context_utilization"]
            self.peak_ctx = max(self.peak_ctx, u or 0)
            self.truncated += bool(e["row"]["truncated"])
        elif t == "job_judged":
            self.judged.append(e)

    @property
    def fraction(self) -> float:
        total = sum(WEIGHTS.values())
        finished = sum(WEIGHTS.get(n, 0) for n in {s["node"] for s in self.done})
        if self.current == "llm_rerank_and_critic" and self.judged:
            finished += WEIGHTS["llm_rerank_and_critic"] * self.judged[-1]["i"] / self.judged[-1]["total"]
        return min(finished / total, 1.0)


def render(tl: Timeline) -> tuple[float, str]:
    lines = []
    for s in tl.done:
        pass_ = f" (pass {s['attempt']})" if s["node"] == "plan_queries" and s.get("attempt", 1) > 1 else ""
        lines.append(f"✅ **{LABELS.get(s['node'], s['node'])}**{pass_} · {s['seconds']:.1f}s  \n"
                     f"<small>{s['note'][:260]}</small>")
    if tl.current:
        extra = ""
        if tl.llm:
            extra = (f"  \n<small>↳ waiting on {tl.llm['model']} for `{tl.llm['purpose']}`"
                     f"{' (retry ' + str(tl.llm['attempt']) + ')' if tl.llm['attempt'] > 1 else ''}…</small>")
        elif tl.current == "search":
            extra = "  \n<small>↳ querying Adzuna, Remotive, RemoteOK, Arbeitnow, Greenhouse, Lever in parallel…</small>"
        prog = ""
        if tl.current == "llm_rerank_and_critic" and tl.judged:
            j = tl.judged[-1]
            prog = f" · {j['i']}/{j['total']} judged"
        lines.append(f"⏳ **{LABELS.get(tl.current, tl.current)}**{prog}{extra}")
    if tl.judged:
        lines.append("\n**Judged so far**")
        for j in tl.judged:
            warn = f" ⚠️ {j['flags']} flag(s)" if j["flags"] else ""
            score = "n/a" if j["llm_score"] is None else f"{j['llm_score']}/10"
            lines.append(f"- {j['title']} @ {j['company']} → LLM fit **{score}**, score {j['final']:.2f}{warn}")
    t = tl.totals
    lines.append(f"\n<small>{t['calls']} LLM calls · {t['prompt']:,} prompt + {t['completion']:,} completion tokens · "
                 f"peak context use {tl.peak_ctx:.0%}" + (f" · ⚠️ {tl.truncated} truncated" if tl.truncated else "") + "</small>")
    return tl.fraction, "\n\n".join(lines)
