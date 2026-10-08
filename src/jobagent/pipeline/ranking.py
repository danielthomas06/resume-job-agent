"""Ranking logic. Pure functions (no I/O) where possible so they are unit-testable.

Stage 1 (cheap, whole pool):  prelim = 0.5*embedding_sim + 0.3*skill_coverage + 0.2*title_match
Stage 2 (LLM, top-N only):    pointwise fit score 0-10 with matched/missing skills + reason
Critic (deterministic):       drop skills the LLM claims that are not grounded in profile / job text
Final:                        0.4*prelim + 0.6*(critic-adjusted llm score / 10)
"""
import re

import numpy as np
from pydantic import BaseModel, Field

from jobagent.mcp_servers.schema import Job
from jobagent.models import Profile

W_EMB, W_SKILL, W_TITLE = 0.5, 0.3, 0.2
W_PRELIM, W_LLM = 0.4, 0.6


class LLMJudgement(BaseModel):
    fit_score: int = Field(ge=0, le=10, description="0=irrelevant, 10=ideal match")
    seniority_fit: str = Field(description="too_junior | good | too_senior")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list, description="Required by the job, absent from candidate")
    reason: str = Field(description="1-2 sentences, cite concrete evidence")


class Ranked(BaseModel):
    job: Job
    prelim: float
    llm_score: float | None = None
    final: float
    judgement: LLMJudgement | None = None
    flags: list[str] = []
    features: dict[str, float] = {}  # prelim components: why it scored what it did
    trail: list[str] = []  # human-readable score history, in order


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#. ]", " ", s.lower())


def contains_term(text: str, term: str) -> bool:
    """Word-boundary match so 'R' or 'Go' do not match inside other words."""
    t = _norm(term).strip()
    return bool(t) and re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", _norm(text)) is not None


def skill_coverage(profile: Profile, job: Job) -> float:
    """Fraction of the candidate's skills the job text mentions (capped: 5 hits = full marks)."""
    text = f"{job.title} {job.description} {' '.join(job.tags)}"
    hits = sum(contains_term(text, sk) for sk in profile.skills)
    return min(hits / 5, 1.0)


def title_match(profile: Profile, job: Job) -> float:
    jt = set(_norm(job.title).split())
    best = 0.0
    for t in profile.target_titles:
        tt = set(_norm(t).split())
        if tt:
            best = max(best, len(tt & jt) / len(tt))
    return best


def prelim_breakdown(profile: Profile, jobs: list[Job], profile_vec: np.ndarray, job_vecs: np.ndarray) -> list[dict]:
    sims = job_vecs @ profile_vec  # cosine (rows normalized)
    # min-max within the pool: raw cosine for one embedding model sits in a narrow band
    span = sims.max() - sims.min()
    emb = (sims - sims.min()) / span if span > 1e-6 else np.full_like(sims, 0.5)
    rows = []
    for raw, e, j in zip(sims, emb, jobs):
        sk, ti = skill_coverage(profile, j), title_match(profile, j)
        rows.append({"cosine": float(raw), "emb_norm": float(e), "skill_cov": sk, "title_match": ti,
                     "score": W_EMB * float(e) + W_SKILL * sk + W_TITLE * ti})
    return rows


def prelim_scores(profile: Profile, jobs: list[Job], profile_vec: np.ndarray, job_vecs: np.ndarray):
    return [r["score"] for r in prelim_breakdown(profile, jobs, profile_vec, job_vecs)]


def critic(profile: Profile, job: Job, j: LLMJudgement) -> tuple[LLMJudgement, list[str]]:
    """Grounding check. matched must be in profile AND job text; missing must be in the job text
    and NOT in the profile. Ungrounded claims are removed and the score is penalized."""
    text = f"{job.title} {job.description}"
    flags, matched, missing = [], [], []
    for s in j.matched_skills:
        if any(contains_term(p, s) or contains_term(s, p) for p in profile.skills) and contains_term(text, s):
            matched.append(s)
        else:
            flags.append(f"ungrounded matched skill: {s}")
    for s in j.missing_skills:
        if contains_term(text, s) and not any(contains_term(p, s) for p in profile.skills):
            missing.append(s)
        else:
            flags.append(f"dropped missing skill: {s}")
    for s in j.matched_skills:
        if s not in matched and contains_term(j.reason, s):
            flags.append(f"reason cites ungrounded skill: {s}")
    score = j.fit_score
    if j.fit_score >= 7 and not matched:
        score = min(score, 5)
        flags.append("high score without any grounded evidence; capped at 5")
    penalty = min(sum(f.startswith("ungrounded") for f in flags), 3)  # "reason cites" flags are informational
    score = max(0, score - penalty)
    return j.model_copy(update={"matched_skills": matched, "missing_skills": missing, "fit_score": score}), flags


def final_score(prelim: float, llm_score: float | None) -> float:
    return prelim if llm_score is None else W_PRELIM * prelim + W_LLM * (llm_score / 10)


def dedupe(jobs: list[Job]) -> list[Job]:
    seen, out = set(), []
    for j in jobs:
        key = (_norm(j.title).strip(), _norm(j.company).strip())
        if key not in seen and j.id not in seen:
            seen.update([key, j.id])
            out.append(j)
    return out
