"""
reporting.py — CSV export utilities for SkillBridge analysis results.

Provides functions that convert analysis results into CSV bytes
suitable for Streamlit's st.download_button.
"""

import io

import pandas as pd


def export_career_matches_csv(
    matches_df: pd.DataFrame,
    user_skills: list[str],
) -> bytes:
    """
    Export career match results as CSV bytes.

    Includes: rank, category, score, matching skills, missing skills.
    """
    if matches_df.empty:
        # Return a header-only CSV
        return pd.DataFrame(columns=["rank", "category", "score"]).to_csv(index=False).encode("utf-8")

    rows = []
    for i, row in matches_df.iterrows():
        job_skills = row.get("extracted_skills", [])
        if not isinstance(job_skills, list):
            job_skills = []

        user_set = set(user_skills)
        job_set = set(job_skills)
        matching = sorted(user_set & job_set)
        missing = sorted(job_set - user_set)

        score_col = "cosine_score" if "cosine_score" in row.index else "overlap_score"
        rows.append({
            "rank": i + 1,
            "job_title": row.get("job_title", ""),
            "category": row.get("category", ""),
            "similarity_score": round(row.get(score_col, 0.0), 4),
            "matching_skills": ", ".join(matching),
            "missing_skills": ", ".join(missing),
        })

    df_out = pd.DataFrame(rows)
    return df_out.to_csv(index=False).encode("utf-8")


def export_skill_gap_csv(
    gap_result: dict,
    learning_plan: list[dict],
) -> bytes:
    """
    Export skill gap analysis and learning plan as CSV bytes.
    """
    rows = []

    gap = gap_result.get("gap", {})
    category = gap_result.get("category", "")
    n_jobs = gap_result.get("n_jobs_in_cat", 0)

    # Possessed skills
    for skill in gap.get("possessed", []):
        rows.append({
            "skill": skill,
            "status": "Possessed",
            "career_category": category,
            "demand_frequency": "",
            "pct_of_jobs": "",
            "priority_rank": "",
            "recommended_resources": "",
        })

    # Missing skills (with demand data from the learning plan)
    plan_lookup = {item["skill"]: item for item in learning_plan}

    for skill in gap.get("missing", []):
        plan_item = plan_lookup.get(skill, {})
        resources = plan_item.get("resources", [])
        resource_str = " | ".join(f"{r['title']} ({r['url']})" for r in resources)
        rows.append({
            "skill": skill,
            "status": "Missing",
            "career_category": category,
            "demand_frequency": plan_item.get("frequency", ""),
            "pct_of_jobs": plan_item.get("pct_of_jobs", ""),
            "priority_rank": plan_item.get("priority_rank", ""),
            "recommended_resources": resource_str,
        })

    if not rows:
        return pd.DataFrame(columns=["skill", "status", "career_category"]).to_csv(index=False).encode("utf-8")

    df_out = pd.DataFrame(rows)
    return df_out.to_csv(index=False).encode("utf-8")


def export_evaluation_csv(skill_eval: dict) -> bytes:
    """Export skill extraction evaluation results as CSV bytes."""
    per_example = skill_eval.get("per_example", [])
    if not per_example:
        return b""

    rows = []
    for item in per_example:
        rows.append({
            "text_snippet": item.get("text", ""),
            "expected_skills": ", ".join(item.get("expected", [])),
            "predicted_skills": ", ".join(item.get("predicted", [])),
            "true_positives": item.get("tp", 0),
            "false_positives": item.get("fp", 0),
            "false_negatives": item.get("fn", 0),
            "precision": item.get("precision", 0),
            "recall": item.get("recall", 0),
            "f1": item.get("f1", 0),
        })

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
