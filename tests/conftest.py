"""
conftest.py — Shared pytest fixtures for SkillBridge tests.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the required schema."""
    return pd.DataFrame({
        "job_id": ["1", "2", "3", "4", "5", "6"],
        "category": [
            "Python Developer", "Data Analyst", "Machine Learning Engineer",
            "Full-Stack Developer", "Backend Developer", "Frontend Developer",
        ],
        "job_title": [
            "Python Dev", "Data Analyst", "ML Engineer",
            "Full Stack Dev", "Backend Dev", "Frontend Dev",
        ],
        "job_description": [
            "Python developer with Django and PostgreSQL experience. Git required.",
            "Data analysis using SQL, Python, pandas, and Tableau visualization.",
            "Machine learning engineer using scikit-learn, TensorFlow, and PyTorch.",
            "Full-stack developer with React, Node.js, JavaScript, and PostgreSQL.",
            "Backend developer with Java, Spring Boot, Docker, and Kubernetes.",
            "Frontend developer with HTML, CSS, JavaScript, and React skills.",
        ],
        "job_skill_set": [
            "Python, Django, PostgreSQL, Git",
            "SQL, Python, Pandas, Tableau, Data Analysis",
            "Machine Learning, scikit-learn, TensorFlow, PyTorch",
            "React, Node.js, JavaScript, PostgreSQL, HTML, CSS",
            "Java, Spring Boot, Docker, Kubernetes",
            "HTML, CSS, JavaScript, React",
        ],
    })


@pytest.fixture
def preprocessed_df(sample_df):
    """DataFrame with preprocessing applied."""
    from src.preprocessing import preprocess_dataframe
    from src.skill_extractor import extract_skills_from_df
    df = preprocess_dataframe(sample_df)
    df["extracted_skills"] = extract_skills_from_df(df)
    return df


@pytest.fixture
def tfidf_artifacts(preprocessed_df):
    """Fitted TF-IDF vectorizer and matrix."""
    from src.feature_engineering import build_tfidf_matrix
    texts = preprocessed_df["clean_description"].tolist()
    vectorizer, matrix = build_tfidf_matrix(texts, max_features=200)
    return vectorizer, matrix


@pytest.fixture
def user_skills_sample():
    """A realistic set of user skills for testing."""
    return ["python", "machine learning", "pandas", "scikit-learn", "sql"]
