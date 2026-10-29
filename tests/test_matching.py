"""
test_matching.py — Tests for cosine similarity and skill-overlap career matching.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching import (
    aggregate_matches_by_category,
    aggregate_overlap_by_category,
    build_match_explanation,
    cosine_match,
    skill_overlap_match,
)
from src.feature_engineering import transform_user_query


class TestCosineMatch:
    def test_returns_dataframe_with_correct_columns(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python", "sql"])
        result = cosine_match(user_vec, matrix, preprocessed_df, top_k=3)
        for col in ["job_id", "job_title", "category", "cosine_score"]:
            assert col in result.columns

    def test_top_k_limits_results(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python"])
        result = cosine_match(user_vec, matrix, preprocessed_df, top_k=2)
        assert len(result) <= 2

    def test_scores_are_between_0_and_1(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python", "machine learning"])
        result = cosine_match(user_vec, matrix, preprocessed_df, top_k=5)
        if not result.empty:
            assert (result["cosine_score"] >= 0).all()
            assert (result["cosine_score"] <= 1).all()

    def test_empty_user_skills_returns_empty(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, [])
        result = cosine_match(user_vec, matrix, preprocessed_df)
        assert result.empty

    def test_sorted_by_score_descending(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python", "sql"])
        result = cosine_match(user_vec, matrix, preprocessed_df, top_k=5)
        if len(result) > 1:
            scores = result["cosine_score"].tolist()
            assert scores == sorted(scores, reverse=True)


class TestSkillOverlapMatch:
    def test_returns_dataframe_with_overlap_score(self, preprocessed_df):
        result = skill_overlap_match(["python", "sql"], preprocessed_df, top_k=3)
        assert "overlap_score" in result.columns

    def test_top_k_limits_results(self, preprocessed_df):
        result = skill_overlap_match(["python"], preprocessed_df, top_k=2)
        assert len(result) <= 2

    def test_overlap_score_between_0_and_1(self, preprocessed_df):
        result = skill_overlap_match(["python", "machine learning"], preprocessed_df, top_k=5)
        if not result.empty:
            assert (result["overlap_score"] >= 0).all()
            assert (result["overlap_score"] <= 1).all()

    def test_empty_user_skills_all_zero(self, preprocessed_df):
        result = skill_overlap_match([], preprocessed_df, top_k=3)
        if not result.empty:
            assert (result["overlap_score"] == 0).all()

    def test_raises_without_extracted_skills(self, sample_df):
        import pytest
        with pytest.raises(ValueError, match="extracted_skills"):
            skill_overlap_match(["python"], sample_df)


class TestAggregateByCategory:
    def test_returns_dataframe_with_category_column(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python"])
        matches = cosine_match(user_vec, matrix, preprocessed_df, top_k=10)
        agg = aggregate_matches_by_category(matches)
        assert "category" in agg.columns
        assert "avg_score" in agg.columns

    def test_sorted_by_avg_score_descending(self, preprocessed_df, tfidf_artifacts):
        vectorizer, matrix = tfidf_artifacts
        user_vec = transform_user_query(vectorizer, ["python", "machine learning"])
        matches = cosine_match(user_vec, matrix, preprocessed_df, top_k=10)
        agg = aggregate_matches_by_category(matches)
        if len(agg) > 1:
            scores = agg["avg_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_empty_match_df_returns_empty(self):
        import pandas as pd
        empty_df = pd.DataFrame(columns=["category", "cosine_score"])
        result = aggregate_matches_by_category(empty_df)
        assert result.empty


class TestBuildMatchExplanation:
    def test_returns_expected_keys(self):
        result = build_match_explanation(
            job_skills=["python", "sql", "docker"],
            user_skills=["python", "react"],
            score=0.75,
        )
        assert "matching_skills" in result
        assert "missing_skills" in result
        assert "overlap_pct" in result
        assert "explanation" in result

    def test_matching_skills_are_intersection(self):
        result = build_match_explanation(
            job_skills=["python", "sql", "docker"],
            user_skills=["python", "react"],
            score=0.5,
        )
        assert result["matching_skills"] == ["python"]

    def test_missing_skills_are_difference(self):
        result = build_match_explanation(
            job_skills=["python", "sql", "docker"],
            user_skills=["python", "react"],
            score=0.5,
        )
        assert "sql" in result["missing_skills"]
        assert "docker" in result["missing_skills"]

    def test_overlap_pct_calculation(self):
        result = build_match_explanation(
            job_skills=["python", "sql", "docker"],
            user_skills=["python"],
            score=0.3,
        )
        assert result["overlap_pct"] == pytest.approx(33.3, rel=0.1)

    def test_empty_job_skills(self):
        result = build_match_explanation(
            job_skills=[],
            user_skills=["python"],
            score=0.0,
        )
        assert result["overlap_pct"] == 0.0
