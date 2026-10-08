"""Offline eval harness: golden resumes + a fixed labeled job pool (no live job APIs, so runs are
comparable across models and over time).

  extraction : skill recall, ungrounded-skill (hallucination) rate, seniority, title hit
  ranking    : nDCG@5/@10 for embedding-only vs blended prelim vs prelim+LLM+critic
  safety     : prompt-injection posting must not reach the top-5 and must be flagged
  grounding  : share of LLM skill claims that survive the critic
  cost       : LLM calls, tokens, latency
"""
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from jobagent.config import get_settings
from jobagent.evals.judge import CORRUPTION, judge_reason
from jobagent.evals.metrics import extraction_metrics, ndcg_at_k
from jobagent.llm import LLMClient
from jobagent.llm.budget import Budget
from jobagent.mcp_servers.schema import Job
from jobagent.models import Profile
from jobagent.observability import current_trace_id, trace_url, tracer
from jobagent.pipeline.rank_stage import llm_rerank, prelim_rank
from jobagent.resume import extract_profile, parse_resume

ROOT = Path(__file__).resolve().parents[3] / "evals"
DATA = ROOT / "data"


def load_data():
    gold = json.loads((DATA / "profiles.json").read_text())
    raw = json.loads((DATA / "jobs.json").read_text())
    meta_keys = {"labels", "tier", "adversarial", "benign_trap"}
    jobs = [Job(**{k: v for k, v in r.items() if k not in meta_keys}) for r in raw]
    labels = {r["id"]: r["labels"] for r in raw}
    meta = {r["id"]: {k: r.get(k) for k in ("tier", "adversarial", "benign_trap")} for r in raw}
    return gold, jobs, labels, meta


def _grades(ranked, labels, persona):
    return [labels[r.job.id].get(persona, 0) for r in ranked]


def run_eval(llm: LLMClient, personas: list[str] | None = None, llm_rank: bool = True,
             pool: str = "all", judge: LLMClient | None = None, judge_per_persona: int = 3) -> dict:
    s = get_settings()
    gold, jobs, labels, meta = load_data()
    if pool != "all":
        jobs = [j for j in jobs if meta[j.id]["tier"] == pool]
    adversarial = {j.id for j in jobs if meta[j.id]["adversarial"]}
    traps = {j.id for j in jobs if meta[j.id]["benign_trap"]}
    verdicts: list[tuple[bool, bool, int]] = []  # (original faithful, corrupted flagged, helpfulness)
    names = [n for n in gold if not personas or n in personas]
    llm.budget = Budget(max_calls=500, max_tokens=2_000_000)
    t0 = time.perf_counter()
    root = tracer().start_as_current_span("eval.run")
    root_span = root.__enter__()
    root_span.set_attribute("openinference.span.kind", "AGENT")
    root_span.set_attribute("eval.pool", pool)
    root_span.set_attribute("eval.model", f"{llm.provider}:{llm.model}")
    tid = current_trace_id()
    report = {"trace_id": tid, "trace_url": trace_url(tid, "jobagent-evals"), "provider": llm.provider, "model": llm.model, "timestamp": datetime.now().isoformat(timespec="seconds"),
              "pool": pool, "n_jobs": len(jobs), "extraction": {}, "ranking": {}, "safety": {}, "grounding": {}, "judge": {}}

    for n in names:
        with tracer().start_as_current_span(f"eval.extract.{n}") as _sp:
            _sp.set_attribute("eval.persona", n)
            text = parse_resume(DATA / "resumes" / f"{n}.txt")
            try:
                prof = extract_profile(text, llm, s.redact_before_remote)
                report["extraction"][n] = extraction_metrics(prof, gold[n], text)
            except Exception as e:
                report["extraction"][n] = {"error": f"{type(e).__name__}: {e}"}

    kept = dropped = 0
    for n in names:
        with tracer().start_as_current_span(f"eval.rank.{n}") as _sp:
            _sp.set_attribute("eval.persona", n)
            g = gold[n]
            profile = Profile(skills=g["skills"], seniority=g["seniority_ok"][0], domains=g["domains"],
                              target_titles=g["target_titles"], headline=g["headline"],
                              total_years_experience=g["total_years_experience"])
            all_grades = [labels[j.id].get(n, 0) for j in jobs]
            row = {}
            emb = prelim_rank(profile, jobs, mode="embedding")
            pre = prelim_rank(profile, jobs)
            variants = {"embedding_only": emb, "prelim": pre}
            if llm_rank:
                variants["full"] = llm_rerank(profile, prelim_rank(profile, jobs), llm)
            for name, ranked in variants.items():
                gr = _grades(ranked, labels, n)
                row[name] = {"ndcg@5": ndcg_at_k(gr, all_grades, 5), "ndcg@10": ndcg_at_k(gr, all_grades, 10)}
            report["ranking"][n] = row
            if llm_rank:
                full = variants["full"]
                ids = [r.job.id for r in full]
                flagged = {r.job.id for r in full if any("injection" in f for f in r.flags)}
                report["safety"][n] = {
                    "adversarial_in_top5": sum(ids.index(a) < 5 for a in adversarial),
                    "adversarial_flagged": len(adversarial & flagged), "adversarial_total": len(adversarial),
                    "benign_flagged": sorted(traps & flagged)}
                if judge:
                    for r in [r for r in full if r.judgement][:judge_per_persona]:
                        jt = f"{r.job.title}\n{r.job.description}"
                        ok = judge_reason(judge, profile, jt, r.judgement.reason)
                        bad = judge_reason(judge, profile, jt, r.judgement.reason + CORRUPTION)
                        verdicts.append((ok.faithful, not bad.faithful, ok.helpfulness))
                for r in full:
                    if r.judgement:
                        k = len(r.judgement.matched_skills) + len(r.judgement.missing_skills)
                        d = sum(f.startswith(("ungrounded", "dropped")) for f in r.flags)
                        kept, dropped = kept + k, dropped + d

    if llm_rank and kept + dropped:
        report["grounding"] = {"claims": kept + dropped, "grounded_rate": kept / (kept + dropped)}
    if verdicts:
        report["judge"] = {"n": len(verdicts), "judge_model": f"{judge.provider}:{judge.model}",
                           "faithful_rate": mean(v[0] for v in verdicts),
                           "corruption_detected_rate": mean(v[1] for v in verdicts),
                           "mean_helpfulness": mean(v[2] for v in verdicts)}
    report["summary"] = _summarize(report)
    root_span.set_attribute("eval.summary", json.dumps(report["summary"], default=str))
    root.__exit__(None, None, None)
    report["usage"] = {**{k: v for k, v in llm.budget.summary().items() if k != "calls"},
                       "wall_s": round(time.perf_counter() - t0, 1)}
    return report


def _summarize(r: dict) -> dict:
    ex = [v for v in r["extraction"].values() if "error" not in v]
    out = {}
    if ex:
        out |= {"skill_recall": mean(v["skill_recall"] for v in ex),
                "ungrounded_skill_rate": mean(v["ungrounded_skill_rate"] for v in ex),
                "seniority_acc": mean(v["seniority_ok"] for v in ex),
                "title_hit_rate": mean(v["title_hit"] for v in ex),
                "extraction_failures": len(r["extraction"]) - len(ex)}
    for variant in ("embedding_only", "prelim", "full"):
        rows = [v[variant] for v in r["ranking"].values() if variant in v]
        if rows:
            out[f"ndcg@5[{variant}]"] = mean(x["ndcg@5"] for x in rows)
            out[f"ndcg@10[{variant}]"] = mean(x["ndcg@10"] for x in rows)
    if r["safety"]:
        tot = sum(v["adversarial_total"] for v in r["safety"].values())
        out["adversarial_in_top5"] = sum(v["adversarial_in_top5"] for v in r["safety"].values())
        out["adversarial_flagged_rate"] = sum(v["adversarial_flagged"] for v in r["safety"].values()) / max(tot, 1)
        out["benign_false_flags"] = sum(len(v["benign_flagged"]) for v in r["safety"].values())
    if r.get("judge"):
        out |= {"judge_faithful_rate": r["judge"]["faithful_rate"],
                "judge_corruption_detected": r["judge"]["corruption_detected_rate"],
                "judge_helpfulness": r["judge"]["mean_helpfulness"]}
    if r["grounding"]:
        out["claim_grounded_rate"] = r["grounding"]["grounded_rate"]
    return out


def save(report: dict) -> Path:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    safe = report["model"].replace(":", "-").replace("/", "-")
    p = out / f"{report['provider']}_{safe}_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    return p
