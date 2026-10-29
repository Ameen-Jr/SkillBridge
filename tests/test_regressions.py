"""
test_regressions.py — Regression tests for defects found during project handoff.

Covers:
- AWS must have one canonical name (previously split into "aws" / "amazon web services").
- Skills present in the bundled dataset's tags must be in the dictionary.
- compute_skill_demand([]) must keep its columns.
- src.pipeline must be importable without executing the dashboard (pages used to
  `from app import load_pipeline`, which re-ran app.py and crashed every page).
- Every Streamlit page must run without an uncaught exception (Streamlit AppTest),
  including with an empty profile and a profile of unrecognized skills.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SKILL_ALIASES, SKILL_DICTIONARY
from src.preprocessing import normalize_skill_name
from src.skill_extractor import extract_skills, normalize_user_skills


class TestAwsCanonicalName:
    def test_alias_maps_long_name_to_aws(self):
        assert normalize_skill_name("amazon web services") == "aws"

    def test_aws_is_not_aliased_away(self):
        assert SKILL_ALIASES.get("aws") is None
        assert normalize_skill_name("AWS") == "aws"

    def test_user_input_aws_stays_aws(self):
        recognized, unrecognized = normalize_user_skills(["AWS"])
        assert recognized == ["aws"]
        assert unrecognized == []

    def test_dictionary_has_single_aws_entry(self):
        assert "aws" in SKILL_DICTIONARY
        assert "amazon web services" not in SKILL_DICTIONARY

    def test_description_text_and_tag_agree(self):
        from src.skill_extractor import extract_skills_from_row
        assert extract_skills_from_row("AWS", "Deploy on AWS") == ["aws"]


class TestDatasetSkillsInDictionary:
    @pytest.mark.parametrize("skill", ["jwt", "redux", "celery", "sqlalchemy", "etl"])
    def test_extracted_from_text(self, skill):
        assert skill in extract_skills(f"Experience with {skill.upper()} is required")


class TestSkillDemandEmpty:
    def test_empty_missing_skills_keeps_columns(self, preprocessed_df):
        from src.skill_gap import compute_skill_demand
        out = compute_skill_demand([], preprocessed_df)
        assert out.empty
        assert list(out.columns) == ["skill", "frequency", "pct_of_jobs"]


class TestPipelineModule:
    def test_importable_without_running_dashboard(self):
        import importlib
        mod = importlib.import_module("src.pipeline")
        assert callable(mod.load_pipeline)

    def test_pages_do_not_import_app(self):
        for page in (ROOT / "pages").glob("*.py"):
            assert "from app import" not in page.read_text(), page.name


PAGES = [
    "app.py",
    "pages/1_Profile.py",
    "pages/2_Career_Matches.py",
    "pages/3_Skill_Gap.py",
    "pages/4_Skill_Network.py",
    "pages/5_Model_Evaluation.py",
    "pages/6_About.py",
]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize(
    "skills",
    [[], ["python", "sql", "pandas", "docker"], ["zzzunknown"]],
    ids=["empty-profile", "typical-profile", "unrecognized-skill"],
)
def test_page_runs_without_exception(page, skills):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / page), default_timeout=180)
    at.session_state["user_skills"] = skills
    at.run()
    assert not at.exception, [e.value for e in at.exception]
