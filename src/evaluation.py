"""
evaluation.py — Honest evaluation of SkillBridge ML components.

Contains:
1. A manually labelled evaluation set for skill extraction (30 examples).
2. A manually labelled evaluation set for career matching (20 examples).
3. Functions to compute precision, recall, F1 for skill extraction.
4. Functions to compute Precision@K for career matching.
5. A baseline comparison function.

All metrics are computed from actual test data.
No scores are fabricated. Limitations of the evaluation data are documented.

IMPORTANT: The evaluation set is small (30 and 20 examples) and hand-crafted.
Results should be interpreted as indicative, not as definitive benchmarks.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.skill_extractor import extract_skills

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill extraction evaluation set
# 30 manually reviewed examples with expected skill outputs.
# Each expected list contains only skills that a reasonable reviewer
# would expect to be extracted from the given text.
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_EVAL: list[dict] = [
    {"text": "Experience with Python and SQL required", "expected": ["python", "sql"]},
    {"text": "Must know React and TypeScript for frontend development", "expected": ["react", "typescript"]},
    {"text": "Docker and Kubernetes for container orchestration", "expected": ["docker", "kubernetes"]},
    {"text": "Machine learning with scikit-learn and pandas", "expected": ["machine learning", "pandas", "scikit-learn"]},
    {"text": "Deep learning using TensorFlow and PyTorch", "expected": ["deep learning", "pytorch", "tensorflow"]},
    {"text": "Data analysis using Excel and Tableau", "expected": ["data analysis", "excel", "tableau"]},
    {"text": "Build RESTful APIs with Node.js and Express", "expected": ["node.js", "restful api"]},
    {"text": "PostgreSQL for database management and Redis for caching", "expected": ["postgresql", "redis"]},
    {"text": "AWS cloud infrastructure and CI/CD pipelines with Jenkins", "expected": ["aws", "ci/cd", "jenkins"]},
    {"text": "Natural language processing and computer vision projects", "expected": ["computer vision", "natural language processing"]},
    {"text": "Git version control and Linux system administration", "expected": ["git", "linux"]},
    {"text": "Angular framework with TypeScript and HTML/CSS", "expected": ["angular", "css", "html", "typescript"]},
    {"text": "MongoDB and Elasticsearch for data storage", "expected": ["elasticsearch", "mongodb"]},
    {"text": "Java Spring Boot for enterprise applications", "expected": ["java", "spring boot"]},
    {"text": "Agile and Scrum project management", "expected": ["agile", "scrum"]},
    {"text": "Power BI and Tableau for business intelligence", "expected": ["power bi", "tableau"]},
    {"text": "Apache Spark and Hadoop for big data processing", "expected": ["hadoop", "spark"]},
    {"text": "Go programming language for high-performance services", "expected": ["go"]},
    {"text": "Vue.js for reactive web interfaces", "expected": ["vue"]},
    {"text": "Statistical analysis using R and Python", "expected": ["python", "r", "statistical analysis"]},
    {"text": "FastAPI with async Python and PostgreSQL", "expected": ["fastapi", "postgresql", "python"]},
    {"text": "Terraform and Ansible for infrastructure as code", "expected": ["ansible", "terraform"]},
    {"text": "GraphQL API development with Apollo Server", "expected": ["graphql"]},
    {"text": "C++ and Rust for systems programming", "expected": ["c++", "rust"]},
    {"text": "Airflow for orchestrating data pipelines", "expected": ["airflow"]},
    {"text": "NumPy and Matplotlib for scientific computing", "expected": ["matplotlib", "numpy"]},
    {"text": "Microservices architecture and system design patterns", "expected": ["microservices", "system design"]},
    {"text": "Penetration testing and cybersecurity audits", "expected": ["cybersecurity", "penetration testing"]},
    {"text": "Next.js server-side rendering and Tailwind CSS styling", "expected": ["next.js", "tailwind css"]},
    {"text": "Time series analysis and forecasting models", "expected": ["time series analysis"]},
]


# ---------------------------------------------------------------------------
# Career matching evaluation set
# 20 manually curated examples. Each entry has a skill list and the
# expected career category it should best match.
# ---------------------------------------------------------------------------

CAREER_MATCH_EVAL: list[dict] = [
    {"skills": ["python", "machine learning", "scikit-learn", "pandas", "numpy"], "expected_category": "Machine Learning Engineer"},
    {"skills": ["javascript", "react", "html", "css", "node.js"], "expected_category": "Full-Stack Developer"},
    {"skills": ["sql", "tableau", "excel", "data analysis", "statistical analysis"], "expected_category": "Data Analyst"},
    {"skills": ["python", "django", "postgresql", "docker", "aws"], "expected_category": "Python Developer"},
    {"skills": ["html", "css", "javascript", "react", "typescript"], "expected_category": "Frontend Developer"},
    {"skills": ["java", "spring boot", "postgresql", "docker", "microservices"], "expected_category": "Backend Developer"},
    {"skills": ["python", "tensorflow", "deep learning", "computer vision", "pytorch"], "expected_category": "Machine Learning Engineer"},
    {"skills": ["typescript", "react", "node.js", "postgresql", "aws"], "expected_category": "Full-Stack Developer"},
    {"skills": ["sql", "python", "power bi", "statistical analysis", "data visualization"], "expected_category": "Data Analyst"},
    {"skills": ["python", "flask", "redis", "celery", "linux"], "expected_category": "Python Developer"},
    {"skills": ["javascript", "vue", "css", "html", "webpack"], "expected_category": "Frontend Developer"},
    {"skills": ["go", "postgresql", "kubernetes", "docker", "microservices"], "expected_category": "Backend Developer"},
    {"skills": ["python", "natural language processing", "pytorch", "scikit-learn", "deep learning"], "expected_category": "Machine Learning Engineer"},
    {"skills": ["python", "react", "django", "postgresql", "javascript"], "expected_category": "Full-Stack Developer"},
    {"skills": ["r", "statistical analysis", "data analysis", "tableau", "sql"], "expected_category": "Data Analyst"},
    {"skills": ["python", "pandas", "numpy", "matplotlib", "git"], "expected_category": "Python Developer"},
    {"skills": ["typescript", "angular", "css", "html", "restful api"], "expected_category": "Frontend Developer"},
    {"skills": ["node.js", "express.js", "mongodb", "docker", "aws"], "expected_category": "Backend Developer"},
    {"skills": ["python", "mlops", "docker", "kubernetes", "aws"], "expected_category": "Machine Learning Engineer"},
    {"skills": ["javascript", "next.js", "react", "postgresql", "graphql"], "expected_category": "Full-Stack Developer"},
]


# ---------------------------------------------------------------------------
# Skill extraction metrics
# ---------------------------------------------------------------------------

def evaluate_skill_extraction(
    eval_set: Optional[list[dict]] = None,
) -> dict:
    """
    Compute precision, recall, and F1 for the skill extraction module.

    Uses the embedded SKILL_EXTRACTION_EVAL set by default.

    Precision = TP / (TP + FP)  — of the skills we extracted, how many are correct?
    Recall    = TP / (TP + FN)  — of the true skills, how many did we find?
    F1        = 2 * P * R / (P + R)

    Returns a dict with aggregated and per-example metrics.
    """
    if eval_set is None:
        eval_set = SKILL_EXTRACTION_EVAL

    per_example = []
    total_tp = total_fp = total_fn = 0

    for item in eval_set:
        text = item["text"]
        expected = set(item["expected"])
        predicted = set(extract_skills(text))

        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_example.append({
            "text": text[:60] + "…" if len(text) > 60 else text,
            "expected": sorted(expected),
            "predicted": sorted(predicted),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    # Micro-averaged metrics across all examples
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "n_examples": len(eval_set),
        "per_example": per_example,
        "limitation": (
            "This evaluation used a small hand-crafted set of 30 examples. "
            "Results are indicative and may not generalize to all job descriptions."
        ),
    }


# ---------------------------------------------------------------------------
# Career matching metrics
# ---------------------------------------------------------------------------

def evaluate_career_matching(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    df: pd.DataFrame,
    eval_set: Optional[list[dict]] = None,
    top_k: int = 3,
) -> dict:
    """
    Compute Precision@K for the cosine-similarity career matching model.

    Precision@K = (number of eval examples where the expected category
                   appears in the top-K matched categories) / total examples

    Parameters
    ----------
    vectorizer   : fitted TF-IDF vectorizer
    tfidf_matrix : TF-IDF matrix for all job records
    df           : job DataFrame (aligned with tfidf_matrix)
    eval_set     : list of {"skills": [...], "expected_category": "..."} dicts
    top_k        : how many top categories to consider a hit

    Returns
    -------
    dict with precision@1, precision@3, per_example results
    """
    from src.feature_engineering import transform_user_query
    from src.matching import cosine_match, aggregate_matches_by_category

    if eval_set is None:
        eval_set = CAREER_MATCH_EVAL

    hits_at_1 = 0
    hits_at_k = 0
    per_example = []

    for item in eval_set:
        user_skills = item["skills"]
        expected = item["expected_category"]

        user_vec = transform_user_query(vectorizer, user_skills)
        matches = cosine_match(user_vec, tfidf_matrix, df, top_k=20)
        cat_agg = aggregate_matches_by_category(matches)

        top_cats = cat_agg["category"].tolist()

        hit_at_1 = (len(top_cats) > 0 and top_cats[0] == expected)
        hit_at_k = expected in top_cats[:top_k]

        if hit_at_1:
            hits_at_1 += 1
        if hit_at_k:
            hits_at_k += 1

        per_example.append({
            "skills": user_skills,
            "expected": expected,
            "top_predictions": top_cats[:top_k],
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
        })

    n = len(eval_set)
    return {
        "precision_at_1": round(hits_at_1 / n, 3) if n > 0 else 0.0,
        f"precision_at_{top_k}": round(hits_at_k / n, 3) if n > 0 else 0.0,
        "n_examples": n,
        "top_k": top_k,
        "per_example": per_example,
        "limitation": (
            f"Evaluated on {n} hand-crafted examples. "
            "The evaluation set was designed to be representative but is not exhaustive. "
            "Real-world accuracy may differ."
        ),
    }


def evaluate_overlap_matching(
    df: pd.DataFrame,
    eval_set: Optional[list[dict]] = None,
    top_k: int = 3,
) -> dict:
    """
    Compute Precision@K for the skill-overlap baseline method.
    Same structure as evaluate_career_matching for direct comparison.
    """
    from src.matching import skill_overlap_match, aggregate_overlap_by_category

    if eval_set is None:
        eval_set = CAREER_MATCH_EVAL

    hits_at_1 = 0
    hits_at_k = 0
    per_example = []

    for item in eval_set:
        user_skills = item["skills"]
        expected = item["expected_category"]

        matches = skill_overlap_match(user_skills, df, top_k=20)
        cat_agg = aggregate_overlap_by_category(matches)
        top_cats = cat_agg["category"].tolist()

        hit_at_1 = (len(top_cats) > 0 and top_cats[0] == expected)
        hit_at_k = expected in top_cats[:top_k]

        if hit_at_1:
            hits_at_1 += 1
        if hit_at_k:
            hits_at_k += 1

        per_example.append({
            "skills": user_skills,
            "expected": expected,
            "top_predictions": top_cats[:top_k],
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
        })

    n = len(eval_set)
    return {
        "precision_at_1": round(hits_at_1 / n, 3) if n > 0 else 0.0,
        f"precision_at_{top_k}": round(hits_at_k / n, 3) if n > 0 else 0.0,
        "n_examples": n,
        "top_k": top_k,
        "per_example": per_example,
    }
