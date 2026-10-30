"""
test_dataset_selection.py — Dataset opt-in, list-format skill tags, and
evaluation behaviour when expected categories are absent from the data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import use_real_data
from src.evaluation import evaluate_career_matching, evaluate_overlap_matching
from src.feature_engineering import build_tfidf_matrix
from src.preprocessing import parse_skill_string, split_skill_tags
from src.skill_extractor import extract_skills_from_row


class TestUseRealData:
    def test_default_is_bundled_dataset(self, monkeypatch):
        monkeypatch.delenv("SKILLBRIDGE_USE_REAL_DATA", raising=False)
        assert use_real_data() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_opt_in_values(self, monkeypatch, value):
        monkeypatch.setenv("SKILLBRIDGE_USE_REAL_DATA", value)
        assert use_real_data() is True

    @pytest.mark.parametrize("value", ["", "0", "no", "false"])
    def test_other_values_stay_off(self, monkeypatch, value):
        monkeypatch.setenv("SKILLBRIDGE_USE_REAL_DATA", value)
        assert use_real_data() is False


class TestSplitSkillTags:
    def test_comma_format(self):
        assert split_skill_tags("Python, SQL, Docker") == ["Python", "SQL", "Docker"]

    def test_python_list_format(self):
        raw = "['employee relations', 'talent acquisition', 'FMLA']"
        assert split_skill_tags(raw) == ["employee relations", "talent acquisition", "FMLA"]

    def test_list_with_double_quotes(self):
        assert split_skill_tags('["Python", "SQL"]') == ["Python", "SQL"]

    def test_malformed_list_falls_back(self):
        assert split_skill_tags("['Python', 'SQL'") == ["Python", "SQL"]

    @pytest.mark.parametrize("value", ["", "   ", None, float("nan"), 5])
    def test_empty_or_non_string(self, value):
        assert split_skill_tags(value) == []

    def test_parse_skill_string_handles_list_format(self):
        assert parse_skill_string("['Python', 'ml', 'docker']") == ["docker", "machine learning", "python"]

    def test_extract_from_row_with_list_tags(self):
        assert extract_skills_from_row("['Python', 'Docker', 'payroll processing']", "") == ["docker", "python"]


class TestEvaluationSkipsMissingCategories:
    def test_examples_for_missing_categories_are_skipped(self, preprocessed_df, tfidf_artifacts):
        vec, mat = tfidf_artifacts
        df = preprocessed_df.copy()
        df["category"] = ["HR"] * len(df)  # none of the six tech roles exist
        cosine = evaluate_career_matching(vec, mat, df, top_k=3)
        overlap = evaluate_overlap_matching(df, top_k=3)
        assert cosine["n_examples"] == 0 and cosine["n_skipped"] == 20
        assert overlap["n_examples"] == 0 and overlap["n_skipped"] == 20
        assert cosine["precision_at_1"] == 0.0

    def test_nothing_skipped_when_categories_present(self, preprocessed_df, tfidf_artifacts):
        vec, mat = tfidf_artifacts
        cosine = evaluate_career_matching(vec, mat, preprocessed_df, top_k=3)
        assert cosine["n_skipped"] + cosine["n_examples"] == 20
        assert cosine["n_examples"] > 0

    def test_custom_eval_set_unaffected_when_category_exists(self, preprocessed_df, tfidf_artifacts):
        vec, mat = tfidf_artifacts
        custom = [{"skills": ["python", "django"], "expected_category": "Python Developer"}]
        result = evaluate_career_matching(vec, mat, preprocessed_df, eval_set=custom, top_k=3)
        assert result["n_examples"] == 1 and result["n_skipped"] == 0


class TestRealDataFormatLoads:
    def test_real_style_row_goes_through_pipeline(self, tmp_path):
        """A row shaped like the HuggingFace data (list-format tags, ALL-CAPS text)."""
        from src.data_loader import _load_and_clean
        from src.preprocessing import preprocess_dataframe
        from src.skill_extractor import extract_skills_from_df

        df = pd.DataFrame({
            "job_id": ["1"], "category": ["INFORMATION-TECHNOLOGY"], "job_title": ["IT Specialist"],
            "job_description": ["MANAGES DOCKER AND LINUX SERVERS USING PYTHON."],
            "job_skill_set": ["['Docker', 'Linux', 'communication']"],
        })
        path = tmp_path / "real.csv"
        df.to_csv(path, index=False)
        out = preprocess_dataframe(_load_and_clean(path))
        out["extracted_skills"] = extract_skills_from_df(out)
        assert out.loc[0, "extracted_skills"] == ["docker", "linux", "python"]
