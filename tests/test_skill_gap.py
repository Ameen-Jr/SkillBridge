"""
test_skill_gap.py — Tests for skill gap analysis and ranking.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skill_gap import (
    compute_skill_gap,
    compute_skill_demand,
    get_role_skills,
    rank_missing_skills,
    full_gap_analysis,
)


class TestComputeSkillGap:
    def test_all_possessed(self):
        result = compute_skill_gap(
            user_skills=["python", "sql", "docker"],
            role_skills=["python", "sql", "docker"],
        )
        assert result["possessed"] == ["docker", "python", "sql"]
        assert result["missing"] == []
        assert result["overlap_pct"] == 100.0

    def test_none_possessed(self):
        result = compute_skill_gap(
            user_skills=["react"],
            role_skills=["python", "sql", "docker"],
        )
        assert result["possessed"] == []
        assert sorted(result["missing"]) == ["docker", "python", "sql"]
        assert result["overlap_pct"] == 0.0

    def test_partial_overlap(self):
        result = compute_skill_gap(
            user_skills=["python", "react"],
            role_skills=["python", "sql", "docker"],
        )
        assert "python" in result["possessed"]
        assert "sql" in result["missing"]
        assert "docker" in result["missing"]
        assert result["overlap_pct"] == pytest.approx(33.3, rel=0.1)

    def test_extra_user_skills(self):
        result = compute_skill_gap(
            user_skills=["python", "react", "kubernetes"],
            role_skills=["python", "sql"],
        )
        assert "react" in result["extra"]
        assert "kubernetes" in result["extra"]

    def test_empty_role_skills(self):
        result = compute_skill_gap(
            user_skills=["python"],
            role_skills=[],
        )
        assert result["overlap_pct"] == 0.0
        assert result["missing"] == []

    def test_empty_user_skills(self):
        result = compute_skill_gap(
            user_skills=[],
            role_skills=["python", "sql"],
        )
        assert result["possessed"] == []
        assert sorted(result["missing"]) == ["python", "sql"]


class TestGetRoleSkills:
    def test_returns_skills_for_category(self, preprocessed_df):
        skills = get_role_skills(preprocessed_df, "Python Developer")
        assert isinstance(skills, list)
        assert "python" in skills

    def test_empty_for_unknown_category(self, preprocessed_df):
        skills = get_role_skills(preprocessed_df, "Nonexistent Category 999")
        assert skills == []

    def test_case_insensitive(self, preprocessed_df):
        skills_lower = get_role_skills(preprocessed_df, "python developer")
        skills_normal = get_role_skills(preprocessed_df, "Python Developer")
        assert set(skills_lower) == set(skills_normal)

    def test_respects_top_n(self, preprocessed_df):
        skills = get_role_skills(preprocessed_df, "Data Analyst", top_n=3)
        assert len(skills) <= 3


class TestComputeSkillDemand:
    def test_frequency_is_correct(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Python Developer"]
        demand = compute_skill_demand(["python"], category_df)
        assert not demand.empty
        assert "frequency" in demand.columns
        assert demand.iloc[0]["frequency"] >= 0

    def test_empty_missing_skills(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Data Analyst"]
        demand = compute_skill_demand([], category_df)
        assert demand.empty

    def test_returns_pct_of_jobs(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Data Analyst"]
        demand = compute_skill_demand(["sql"], category_df)
        if not demand.empty:
            assert 0 <= demand.iloc[0]["pct_of_jobs"] <= 100

    def test_unknown_skill_has_zero_frequency(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Python Developer"]
        demand = compute_skill_demand(["superraredummyskillxyz"], category_df)
        if not demand.empty:
            assert demand[demand["skill"] == "superraredummyskillxyz"]["frequency"].iloc[0] == 0


class TestRankMissingSkills:
    def test_sorted_by_frequency_descending(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Data Analyst"]
        gap = compute_skill_gap(["react"], ["sql", "python", "tableau"])
        demand = compute_skill_demand(gap["missing"], category_df)
        ranked = rank_missing_skills(gap, demand)
        if len(ranked) > 1:
            freqs = ranked["frequency"].tolist()
            assert freqs == sorted(freqs, reverse=True)

    def test_empty_demand_returns_empty(self):
        gap = compute_skill_gap([], ["python"])
        empty_demand = pd.DataFrame(columns=["skill", "frequency", "pct_of_jobs"])
        ranked = rank_missing_skills(gap, empty_demand)
        assert ranked.empty

    def test_priority_rank_starts_at_1(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Data Analyst"]
        gap = compute_skill_gap([], ["sql", "python"])
        demand = compute_skill_demand(gap["missing"], category_df)
        ranked = rank_missing_skills(gap, demand)
        if not ranked.empty:
            assert ranked.iloc[0]["priority_rank"] == 1

    def test_has_priority_reason_column(self, preprocessed_df):
        category_df = preprocessed_df[preprocessed_df["category"] == "Data Analyst"]
        gap = compute_skill_gap([], ["sql"])
        demand = compute_skill_demand(gap["missing"], category_df)
        ranked = rank_missing_skills(gap, demand)
        if not ranked.empty:
            assert "priority_reason" in ranked.columns


class TestFullGapAnalysis:
    def test_returns_expected_keys(self, preprocessed_df):
        result = full_gap_analysis(["python"], "Python Developer", preprocessed_df)
        assert "gap" in result
        assert "demand" in result
        assert "ranked" in result
        assert "role_skills" in result
        assert "n_jobs_in_cat" in result

    def test_n_jobs_matches_category(self, preprocessed_df):
        result = full_gap_analysis(["python"], "Data Analyst", preprocessed_df)
        expected = len(preprocessed_df[preprocessed_df["category"] == "Data Analyst"])
        assert result["n_jobs_in_cat"] == expected
