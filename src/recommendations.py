"""
recommendations.py — Learning resource recommendations for missing skills.

Maps canonical skill names to curated learning resources defined in config.py.
Builds a prioritized learning plan from the ranked skill gap results.

Recommendations are transparent:
- The priority reason (dataset demand frequency) is shown.
- Resources are clearly labelled by type (docs, course, tutorial).
- No claim is made that a particular resource is objectively best.
"""

import logging
from urllib.parse import quote

import pandas as pd

from src.config import DEFAULT_RESOURCE, LEARNING_RESOURCES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource lookup
# ---------------------------------------------------------------------------

def get_resources_for_skill(skill: str, max_resources: int = 3) -> list[dict]:
    """
    Return learning resources for a single canonical skill.

    If the skill is not in the resource map, a search-based fallback is returned.

    Parameters
    ----------
    skill         : canonical skill name (lowercase)
    max_resources : maximum number of resources to return

    Returns
    -------
    List of resource dicts: {title, url, type}
    """
    skill_key = skill.lower().strip()

    resources = LEARNING_RESOURCES.get(skill_key, None)

    if resources:
        return resources[:max_resources]

    # Generic fallback — link to a Coursera search
    fallback = {
        "title": f"Search '{skill.title()}' on Coursera",
        "url": DEFAULT_RESOURCE["url"].format(quote(skill)),
        "type": "search",
    }
    return [fallback]


# ---------------------------------------------------------------------------
# Learning plan builder
# ---------------------------------------------------------------------------

def build_learning_plan(
    ranked_skills: pd.DataFrame,
    max_skills: int = 8,
    max_resources_per_skill: int = 3,
) -> list[dict]:
    """
    Build an ordered learning plan from the ranked missing-skills DataFrame.

    Each entry in the plan contains:
    - skill          : canonical skill name
    - priority_rank  : rank (1 = highest priority)
    - priority_reason: why this skill is prioritized (demand-based)
    - resources      : list of learning resource dicts

    Parameters
    ----------
    ranked_skills         : output of rank_missing_skills()
    max_skills            : how many skills to include in the plan
    max_resources_per_skill : resources to attach per skill

    Returns
    -------
    List of plan entry dicts, ordered by priority_rank ascending.
    """
    if ranked_skills.empty:
        return []

    plan = []
    top_skills = ranked_skills.head(max_skills)

    for _, row in top_skills.iterrows():
        skill = row["skill"]
        resources = get_resources_for_skill(skill, max_resources=max_resources_per_skill)
        plan.append({
            "skill": skill,
            "priority_rank": int(row.get("priority_rank", 0)),
            "priority_reason": str(row.get("priority_reason", "")),
            "frequency": int(row.get("frequency", 0)),
            "pct_of_jobs": float(row.get("pct_of_jobs", 0.0)),
            "resources": resources,
        })

    return plan


def get_skill_display_name(skill: str) -> str:
    """Return a title-cased display name for a canonical skill."""
    # Handle special cases
    special_cases = {
        "sql": "SQL",
        "html": "HTML",
        "css": "CSS",
        "aws": "AWS",
        "ci/cd": "CI/CD",
        "mlops": "MLOps",
        "node.js": "Node.js",
        "next.js": "Next.js",
        "restful api": "RESTful API",
        "graphql": "GraphQL",
        "c++": "C++",
        "c#": "C#",
        "jwt": "JWT",
        "a/b testing": "A/B Testing",
        "grpc": "gRPC",
        "power bi": "Power BI",
    }
    return special_cases.get(skill.lower(), skill.title())
