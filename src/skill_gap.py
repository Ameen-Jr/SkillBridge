"""
skill_gap.py — Skill gap analysis between user profile and career roles.

Given a selected career role (category), computes:
- Skills the user already possesses that are relevant to the role
- Skills required by the role that the user lacks
- Frequency of each missing skill in the role's dataset records
- A transparent priority ranking for learning

All analysis is based on the actual dataset — no hardcoded job market claims.
"""

import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core gap analysis
# ---------------------------------------------------------------------------

def compute_skill_gap(
    user_skills: list[str],
    role_skills: list[str],
) -> dict:
    """
    Compare user skills against a role's skill set.

    Parameters
    ----------
    user_skills : canonical skills the user possesses
    role_skills : canonical skills associated with the target role

    Returns
    -------
    dict with keys:
    - possessed    : skills user has that appear in this role
    - missing      : skills the role requires that user lacks
    - extra        : user skills not appearing in the role's required set
    - overlap_pct  : percentage of role skills the user has
    - user_count   : total recognized user skills
    - role_count   : total skills in the role profile
    """
    user_set = set(user_skills)
    role_set = set(role_skills)

    possessed = sorted(user_set & role_set)
    missing = sorted(role_set - user_set)
    extra = sorted(user_set - role_set)
    overlap_pct = round(100 * len(possessed) / len(role_set), 1) if role_set else 0.0

    return {
        "possessed": possessed,
        "missing": missing,
        "extra": extra,
        "overlap_pct": overlap_pct,
        "user_count": len(user_set),
        "role_count": len(role_set),
    }


def get_role_skills(
    df: pd.DataFrame,
    category: str,
    top_n: int = 30,
) -> list[str]:
    """
    Collect the most common skills associated with a career category.

    Uses the 'extracted_skills' column if available, otherwise falls back
    to 'job_skill_set'.

    Parameters
    ----------
    df       : job DataFrame
    category : career category name to filter on
    top_n    : maximum number of skills to include in the role profile

    Returns
    -------
    List of canonical skill names (sorted by frequency descending)
    """
    cat_df = df[df["category"].str.lower() == category.lower()]

    if cat_df.empty:
        logger.warning("No records found for category '%s'.", category)
        return []

    counter: Counter = Counter()

    if "extracted_skills" in cat_df.columns:
        for skill_list in cat_df["extracted_skills"]:
            if isinstance(skill_list, list):
                counter.update(skill_list)
    elif "job_skill_set" in cat_df.columns:
        for skill_string in cat_df["job_skill_set"]:
            skills = [s.strip().lower() for s in str(skill_string).split(",") if s.strip()]
            counter.update(skills)

    return [skill for skill, _ in counter.most_common(top_n)]


def compute_skill_demand(
    missing_skills: list[str],
    category_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each missing skill, count how often it appears in the category's job records.

    This provides data-driven demand frequency based on the loaded dataset,
    not claims about the broader job market.

    Parameters
    ----------
    missing_skills : list of skills the user lacks
    category_df    : subset of the DataFrame for the selected category

    Returns
    -------
    DataFrame with columns: skill, frequency, pct_of_jobs
    (sorted by frequency descending)
    """
    total_jobs = len(category_df)
    if total_jobs == 0:
        return pd.DataFrame(columns=["skill", "frequency", "pct_of_jobs"])

    records = []
    for skill in missing_skills:
        count = 0
        for skill_list in category_df.get("extracted_skills", pd.Series(dtype=object)):
            if isinstance(skill_list, list) and skill in skill_list:
                count += 1
        pct = round(100 * count / total_jobs, 1)
        records.append({"skill": skill, "frequency": count, "pct_of_jobs": pct})

    if not records:
        return pd.DataFrame(columns=["skill", "frequency", "pct_of_jobs"])

    return (
        pd.DataFrame(records)
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )


def rank_missing_skills(
    gap_result: dict,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank missing skills by their demand frequency in the selected category.

    Priority score = frequency (count of job records containing this skill).
    Ties are broken alphabetically.

    Returns a DataFrame with columns:
    skill, frequency, pct_of_jobs, priority_rank, priority_reason
    """
    if demand_df.empty:
        return pd.DataFrame(columns=["skill", "frequency", "pct_of_jobs", "priority_rank", "priority_reason"])

    df = demand_df.copy()
    df = df.sort_values(["frequency", "skill"], ascending=[False, True]).reset_index(drop=True)
    df["priority_rank"] = df.index + 1
    df["priority_reason"] = df.apply(
        lambda row: (
            f"Appears in {row['frequency']} job records "
            f"({row['pct_of_jobs']}% of {row['skill'].title()} roles in this dataset)"
        ),
        axis=1,
    )
    return df


def full_gap_analysis(
    user_skills: list[str],
    category: str,
    df: pd.DataFrame,
) -> dict:
    """
    Run the complete skill gap analysis for a user and career category.

    Convenience wrapper that combines get_role_skills, compute_skill_gap,
    compute_skill_demand, and rank_missing_skills.

    Returns
    -------
    dict with keys:
    - gap          : result from compute_skill_gap
    - demand       : DataFrame from compute_skill_demand
    - ranked       : DataFrame from rank_missing_skills
    - role_skills  : full list of role skills used
    - category     : the selected category
    - n_jobs_in_cat: number of jobs in the dataset for this category
    """
    category_df = df[df["category"].str.lower() == category.lower()]
    role_skills = get_role_skills(df, category)
    gap = compute_skill_gap(user_skills, role_skills)
    demand = compute_skill_demand(gap["missing"], category_df)
    ranked = rank_missing_skills(gap, demand)

    return {
        "gap": gap,
        "demand": demand,
        "ranked": ranked,
        "role_skills": role_skills,
        "category": category,
        "n_jobs_in_cat": len(category_df),
    }
