"""
test_preprocessing.py — Tests for text cleaning and normalization.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    clean_text,
    normalize_skill_name,
    normalize_skill_list,
    parse_skill_string,
    skills_from_user_input,
)


class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("PYTHON Developer") == "python developer"

    def test_removes_newlines(self):
        result = clean_text("Python\nDeveloper\r\nNeeded")
        assert "\n" not in result
        assert "\r" not in result

    def test_collapses_whitespace(self):
        result = clean_text("  python   developer  ")
        assert "  " not in result
        assert result == "python developer"

    def test_preserves_csharp(self):
        result = clean_text("C# and C++ experience")
        assert "c#" in result
        assert "c++" in result

    def test_preserves_node_js(self):
        result = clean_text("Node.js experience required")
        assert "node.js" in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_removes_special_chars(self):
        result = clean_text("Python! @#$% Developer")
        assert "!" not in result
        assert "@" not in result


class TestNormalizeSkillName:
    def test_lowercase(self):
        assert normalize_skill_name("PYTHON") == "python"

    def test_strips_whitespace(self):
        assert normalize_skill_name("  python  ") == "python"

    def test_alias_ml_to_machine_learning(self):
        assert normalize_skill_name("ml") == "machine learning"

    def test_alias_js_to_javascript(self):
        assert normalize_skill_name("js") == "javascript"

    def test_alias_nodejs(self):
        assert normalize_skill_name("nodejs") == "node.js"

    def test_alias_sklearn(self):
        assert normalize_skill_name("sklearn") == "scikit-learn"

    def test_known_skill_unchanged(self):
        assert normalize_skill_name("python") == "python"

    def test_unknown_skill_unchanged(self):
        assert normalize_skill_name("unknownskill123") == "unknownskill123"


class TestNormalizeSkillList:
    def test_deduplicates(self):
        result = normalize_skill_list(["python", "Python", "PYTHON"])
        assert result.count("python") == 1

    def test_applies_aliases(self):
        result = normalize_skill_list(["ml", "js"])
        assert "machine learning" in result
        assert "javascript" in result

    def test_returns_sorted(self):
        result = normalize_skill_list(["python", "docker", "aws"])
        assert result == sorted(result)

    def test_empty_list(self):
        assert normalize_skill_list([]) == []


class TestParseSkillString:
    def test_comma_separated(self):
        result = parse_skill_string("Python, SQL, Docker")
        assert "python" in result
        assert "sql" in result
        assert "docker" in result

    def test_empty_string(self):
        assert parse_skill_string("") == []

    def test_none_input(self):
        assert parse_skill_string(None) == []

    def test_handles_extra_spaces(self):
        result = parse_skill_string("  Python  ,  SQL  ")
        assert "python" in result
        assert "sql" in result


class TestSkillsFromUserInput:
    def test_comma_separated(self):
        result = skills_from_user_input("Python, SQL, React")
        assert "python" in result

    def test_newline_separated(self):
        result = skills_from_user_input("Python\nSQL\nReact")
        assert "python" in result
        assert "sql" in result

    def test_alias_resolution(self):
        result = skills_from_user_input("ml, js")
        assert "machine learning" in result
        assert "javascript" in result

    def test_empty_input(self):
        assert skills_from_user_input("") == []
