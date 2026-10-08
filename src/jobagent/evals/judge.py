"""LLM-as-judge: is a ranking explanation faithful to the job text and the candidate profile?

Judges are fallible, so the harness also calibrates the judge: every explanation is re-judged after
a fabricated requirement is appended. A trustworthy judge should pass the original and fail the corrupted one.
"""
from pydantic import BaseModel, Field

from jobagent.llm import LLMClient
from jobagent.models import Profile

CORRUPTION = " The role also requires 8 years of Rust systems programming experience and a PhD."

SYSTEM = (
    "You are a strict fact-checker for job-match explanations. Given a candidate, a job posting and an "
    "explanation, decide whether EVERY factual claim in the explanation about the job or the candidate "
    "is supported by the texts. Opinions about fit are fine; invented requirements, skills, employers or "
    "years are not. List each unsupported claim verbatim."
)


class Verdict(BaseModel):
    faithful: bool = Field(description="true only if no claim is unsupported")
    unsupported_claims: list[str] = Field(default_factory=list)
    helpfulness: int = Field(ge=1, le=5, description="1-5: does it give concrete, decision-useful evidence?")


def judge_reason(judge: LLMClient, profile: Profile, job_text: str, reason: str) -> Verdict:
    return judge.generate_structured(
        SYSTEM,
        f"<candidate>\nSkills: {profile.skills}\nSeniority: {profile.seniority}, "
        f"{profile.total_years_experience} yrs\nHeadline: {profile.headline}\n</candidate>\n"
        f"<job>\n{job_text[:2500]}\n</job>\n<explanation>\n{reason}\n</explanation>", Verdict, purpose="eval_judge")
