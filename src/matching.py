"""
matching.py — Career matching using cosine similarity and skill-overlap baseline.

Two matching methods are implemented and can be compared:

1. Cosine Similarity (primary)
   - Transforms the user's skill list and each career-category profile into TF-IDF vectors.
   - Ranks profiles by cosine similarity to the user vector.
   - More sensitive to vocabulary in job descriptions.

2. Skill Overlap (baseline)
   - Computes the fraction of a career profile's skills that the user possesses.
   - Entirely interpretable: it is just a ratio.
   - Useful as a simple sanity check against the primary method.

Important: similarity scores are similarity measures, NOT employment probabilities.
"""

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TOP_K_MATCHES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity matching
# ---------------------------------------------------------------------------

def cosine_match(
    user_vector: csr_matrix,
    tfidf_matrix: csr_matrix,
    df: pd.DataFrame,
    top_k: int = TOP_K_MATCHES,
) -> pd.DataFrame:
    """
    Rank job records by cosine similarity to the user's TF-IDF vector.

    Parameters
    ----------
    user_vector  : (1, n_features) sparse TF-IDF vector for the user's skills
    tfidf_matrix : (n_docs, n_features) TF-IDF matrix for all job descriptions
    df           : the job DataFrame (must be aligned with tfidf_matrix rows)
    top_k        : number of top results to return

    Returns
    -------
    DataFrame with columns:
    job_id, job_title, category, cosine_score, extracted_skills
    (sorted by cosine_score descending)
    """
    if user_vector.nnz == 0:
        # Zero vector — no skills entered
        logger.warning("User vector is all-zero (no recognized skills). Returning empty matches.")
        return pd.DataFrame(columns=["job_id", "job_title", "category", "cosine_score", "extracted_skills"])

    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    result = df.iloc[top_indices][["job_id", "job_title", "category"]].copy()
    result["cosine_score"] = scores[top_indices].round(4)
    if "extracted_skills" in df.columns:
        result["extracted_skills"] = df.iloc[top_indices]["extracted_skills"].values
    else:
        result["extracted_skills"] = [[] for _ in top_indices]

    return result.reset_index(drop=True)


def aggregate_matches_by_category(
    match_df: pd.DataFrame,
    score_col: str = "cosine_score",
) -> pd.DataFrame:
    """
    Average match scores by career category.

    Returns a DataFrame sorted by average score descending, with columns:
    category, avg_score, top_score, match_count, sample_skills
    """
    if match_df.empty:
        return pd.DataFrame(columns=["category", "avg_score", "top_score", "match_count", "sample_skills"])

    agg = (
        match_df.groupby("category")
        .agg(
            avg_score=(score_col, "mean"),
            top_score=(score_col, "max"),
            match_count=(score_col, "count"),
        )
        .reset_index()
    )

    # Attach sample skills from the highest-scoring record in each category
    sample_skills_map: dict[str, list[str]] = {}
    for cat in agg["category"]:
        cat_df = match_df[match_df["category"] == cat]
        if not cat_df.empty:
            best_row = cat_df.loc[cat_df[score_col].idxmax()]
            skills = best_row.get("extracted_skills", [])
            sample_skills_map[cat] = skills if isinstance(skills, list) else []
        else:
            sample_skills_map[cat] = []

    agg["sample_skills"] = agg["category"].map(sample_skills_map)
    return agg.sort_values("avg_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Skill-overlap baseline
# ---------------------------------------------------------------------------

def skill_overlap_match(
    user_skills: list[str],
    df: pd.DataFrame,
    top_k: int = TOP_K_MATCHES,
) -> pd.DataFrame:
    """
    Rank job records by skill overlap fraction (simple baseline).

    Overlap fraction = |user_skills ∩ job_skills| / |job_skills|

    If a job has no recognized skills, its overlap score is 0.

    Parameters
    ----------
    user_skills : list of canonical skill names the user possesses
    df          : DataFrame with 'extracted_skills' column
    top_k       : number of results to return

    Returns
    -------
    DataFrame with columns: job_id, job_title, category, overlap_score, extracted_skills
    sorted by overlap_score descending.
    """
    if "extracted_skills" not in df.columns:
        raise ValueError("DataFrame must have an 'extracted_skills' column for overlap matching.")

    user_set = set(user_skills)
    scores: list[float] = []

    for skill_list in df["extracted_skills"]:
        job_set = set(skill_list) if isinstance(skill_list, list) else set()
        if not job_set:
            scores.append(0.0)
        else:
            scores.append(len(user_set & job_set) / len(job_set))

    scores_arr = np.array(scores)
    top_indices = scores_arr.argsort()[::-1][:top_k]

    result = df.iloc[top_indices][["job_id", "job_title", "category", "extracted_skills"]].copy()
    result["overlap_score"] = scores_arr[top_indices].round(4)
    return result.reset_index(drop=True)


def aggregate_overlap_by_category(
    overlap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate overlap scores by category (same shape as aggregate_matches_by_category)."""
    return aggregate_matches_by_category(overlap_df, score_col="overlap_score")


# ---------------------------------------------------------------------------
# Match explanation
# ---------------------------------------------------------------------------

def build_match_explanation(
    job_skills: list[str],
    user_skills: list[str],
    score: float,
    method: str = "cosine",
) -> dict:
    """
    Build a human-readable explanation for a career match result.

    Returns a dict with:
    - matching_skills : skills the user has that appear in this job
    - missing_skills  : skills in the job that the user lacks
    - overlap_pct     : percentage of job skills the user already has
    - explanation     : one-sentence explanation string
    """
    user_set = set(user_skills)
    job_set = set(job_skills) if isinstance(job_skills, list) else set()

    matching = sorted(user_set & job_set)
    missing = sorted(job_set - user_set)
    overlap_pct = round(100 * len(matching) / len(job_set), 1) if job_set else 0.0

    if method == "cosine":
        explanation = (
            f"Cosine similarity score: {score:.3f} (higher = more similar vocabulary). "
            f"You already have {len(matching)} of {len(job_set)} recognized skills for this role "
            f"({overlap_pct}% overlap)."
        )
    else:
        explanation = (
            f"Skill overlap: {overlap_pct}% ({len(matching)} of {len(job_set)} skills matched). "
            "This is a direct count of shared skills."
        )

    return {
        "matching_skills": matching,
        "missing_skills": missing,
        "overlap_pct": overlap_pct,
        "explanation": explanation,
    }
