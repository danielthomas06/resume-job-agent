import math

from jobagent.models import Profile
from jobagent.pipeline.ranking import contains_term


def ndcg_at_k(ranked_grades: list[int], all_grades: list[int], k: int) -> float:
    """Graded nDCG. ranked_grades: relevance of docs in system order; all_grades: every doc's grade."""
    def dcg(gs):
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    ideal = dcg(sorted(all_grades, reverse=True))
    return dcg(ranked_grades) / ideal if ideal else 0.0


def extraction_metrics(profile: Profile, gold: dict, resume_text: str) -> dict:
    got = profile.skills
    recall_hits = [g for g in gold["skills"] if any(contains_term(s, g) or contains_term(g, s) for s in got)]
    ungrounded = [s for s in got if not contains_term(resume_text, s)]
    title_words = {w.lower() for t in gold["target_titles"] for w in t.split()} - {"engineer", "developer"}
    title_hit = any(w in " ".join(profile.target_titles).lower() for w in title_words)
    return {
        "skill_recall": len(recall_hits) / len(gold["skills"]),
        "ungrounded_skill_rate": len(ungrounded) / len(got) if got else 0.0,
        "ungrounded_skills": ungrounded,
        "seniority_ok": profile.seniority in gold["seniority_ok"],
        "title_hit": title_hit,
        "missed_skills": [g for g in gold["skills"] if g not in recall_hits],
    }
