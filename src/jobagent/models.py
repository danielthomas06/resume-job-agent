from typing import Literal

from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str
    company: str = ""
    start: str = ""
    end: str = ""
    summary: str = ""


class Education(BaseModel):
    degree: str
    institution: str = ""
    year: str = ""


class Profile(BaseModel):
    """Structured candidate profile extracted from a resume."""

    name: str = ""
    headline: str = Field("", description="One-line professional summary")
    total_years_experience: float = Field(0, description="Total professional years, best estimate")
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "principal"] = "mid"
    skills: list[str] = Field(default_factory=list, description="Technical skills, normalized names")
    domains: list[str] = Field(default_factory=list, description="e.g. NLP, fintech, MLOps")
    target_titles: list[str] = Field(
        default_factory=list, description="Job titles this person should search for"
    )
    locations: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
