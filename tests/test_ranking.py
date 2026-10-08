import numpy as np

from jobagent.mcp_servers.schema import Job
from jobagent.models import Profile
from jobagent.pipeline import ranking as rk

P = Profile(skills=["Python", "PyTorch", "R"], target_titles=["ML Engineer"])


def job(title="ML Engineer", desc="", company="A", id="s:1"):
    return Job(id=id, source="s", title=title, company=company, url="u", description=desc)


def test_term_boundaries():
    assert rk.contains_term("we use R and Python", "R")
    assert not rk.contains_term("our teams grow rapidly", "R")


def test_skill_and_title_features():
    j = job(desc="Python PyTorch experience")
    assert rk.skill_coverage(P, j) == 2 / 5
    assert rk.title_match(P, j) == 1.0 and rk.title_match(P, job(title="Chef")) == 0.0


def test_prelim_prefers_closer_vector():
    jobs = [job(id="a"), job(id="b")]
    s = rk.prelim_scores(P, jobs, np.array([1.0, 0.0]), np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert s[0] > s[1]


def test_critic_removes_hallucinations_and_caps():
    j = job(desc="Needs Python and Kubernetes")
    judged = rk.LLMJudgement(fit_score=9, seniority_fit="good", reason="r",
                             matched_skills=["Python", "Java"], missing_skills=["Kubernetes", "Rust"])
    out, flags = rk.critic(P, j, judged)
    assert out.matched_skills == ["Python"] and out.missing_skills == ["Kubernetes"]
    assert out.fit_score == 8 and len(flags) == 2  # 1 ungrounded-matched penalty, 1 dropped missing


def test_critic_caps_unevidenced_high_score():
    out, _ = rk.critic(P, job(desc="Cooking"), rk.LLMJudgement(
        fit_score=9, seniority_fit="good", reason="r", matched_skills=["Python"]))
    assert out.fit_score <= 5


def test_dedupe_same_title_company():
    assert len(rk.dedupe([job(id="a"), job(id="b"), job(id="c", company="B")])) == 2


def test_critic_flags_reason_citing_dropped_skill():
    judged = rk.LLMJudgement(fit_score=6, seniority_fit="good", matched_skills=["Python", "Java"],
                             reason="Strong in Python and Java.")
    _, flags = rk.critic(P, job(desc="Python role"), judged)
    assert "reason cites ungrounded skill: Java" in flags
