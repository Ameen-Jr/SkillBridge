"""
test_skill_extractor.py — Tests for skill extraction and normalization.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skill_extractor import (
    extract_skills,
    extract_skills_from_row,
    normalize_user_skills,
)


class TestExtractSkills:
    def test_extracts_python(self):
        result = extract_skills("Experience with Python programming required.")
        assert "python" in result

    def test_extracts_multiple_skills(self):
        result = extract_skills("Must know Python, SQL, and Docker for this role.")
        assert "python" in result
        assert "sql" in result
        assert "docker" in result

    def test_case_insensitive(self):
        result = extract_skills("PYTHON and MACHINE LEARNING experience.")
        assert "python" in result
        assert "machine learning" in result

    def test_empty_text_returns_empty(self):
        assert extract_skills("") == []

    def test_none_text_returns_empty(self):
        assert extract_skills(None) == []

    def test_no_skills_in_text(self):
        result = extract_skills("We are looking for a motivated team player.")
        assert result == []

    def test_extracts_compound_skill(self):
        result = extract_skills("Natural language processing and computer vision required.")
        assert "natural language processing" in result
        assert "computer vision" in result

    def test_no_false_positive_py_in_copy(self):
        """'py' should not match inside 'copy'."""
        result = extract_skills("Please copy all files to the server.")
        assert "python" not in result

    def test_result_is_sorted_list(self):
        result = extract_skills("SQL, Python, Docker experience.")
        assert result == sorted(result)
        assert isinstance(result, list)

    def test_deduplicates_results(self):
        result = extract_skills("Python python PYTHON developer.")
        assert result.count("python") == 1

    def test_extracts_csharp(self):
        result = extract_skills("C# and ASP.NET development.")
        assert "c#" in result

    def test_extracts_cplusplus(self):
        result = extract_skills("C++ system programming role.")
        assert "c++" in result


class TestExtractSkillsFromRow:
    def test_combined_from_both_sources(self):
        result = extract_skills_from_row(
            skill_string="Python, SQL",
            description="Docker and Kubernetes deployment"
        )
        assert "python" in result
        assert "sql" in result
        assert "docker" in result
        assert "kubernetes" in result

    def test_empty_skill_string(self):
        result = extract_skills_from_row(skill_string="", description="Python developer needed.")
        assert "python" in result

    def test_empty_description(self):
        result = extract_skills_from_row(skill_string="Python, React", description="")
        assert "python" in result
        assert "react" in result

    def test_both_empty(self):
        result = extract_skills_from_row(skill_string="", description="")
        assert result == []


class TestNormalizeUserSkills:
    def test_recognized_skills(self):
        recognized, unrecognized = normalize_user_skills(["python", "sql", "docker"])
        assert "python" in recognized
        assert "sql" in recognized
        assert "docker" in recognized
        assert unrecognized == []

    def test_alias_resolution_ml(self):
        recognized, unrecognized = normalize_user_skills(["ml"])
        assert "machine learning" in recognized
        assert unrecognized == []

    def test_unrecognized_skills(self):
        recognized, unrecognized = normalize_user_skills(["superraredummyskill999"])
        assert len(recognized) == 0
        assert "superraredummyskill999" in unrecognized

    def test_mixed_recognized_unrecognized(self):
        recognized, unrecognized = normalize_user_skills(["python", "unknownskill"])
        assert "python" in recognized
        assert "unknownskill" in unrecognized

    def test_empty_input(self):
        recognized, unrecognized = normalize_user_skills([])
        assert recognized == []
        assert unrecognized == []

    def test_deduplicates_recognized(self):
        recognized, _ = normalize_user_skills(["python", "python", "PYTHON"])
        assert recognized.count("python") == 1
