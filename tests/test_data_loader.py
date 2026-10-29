"""
test_data_loader.py — Tests for dataset loading, schema validation, and cleaning.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import _load_and_clean, validate_schema, get_dataset_stats
from src.config import REQUIRED_COLUMNS


class TestSchemaValidation:
    def test_valid_schema_passes(self, sample_df):
        is_valid, missing = validate_schema(sample_df)
        assert is_valid is True
        assert missing == []

    def test_missing_one_column_detected(self, sample_df):
        df_bad = sample_df.drop(columns=["job_skill_set"])
        is_valid, missing = validate_schema(df_bad)
        assert is_valid is False
        assert "job_skill_set" in missing

    def test_missing_multiple_columns(self, sample_df):
        df_bad = sample_df.drop(columns=["job_description", "category"])
        is_valid, missing = validate_schema(df_bad)
        assert is_valid is False
        assert len(missing) == 2

    def test_empty_dataframe_with_correct_columns(self):
        df_empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        is_valid, missing = validate_schema(df_empty)
        assert is_valid is True
        assert missing == []

    def test_extra_columns_are_ok(self, sample_df):
        df_extra = sample_df.copy()
        df_extra["extra_col"] = "x"
        is_valid, missing = validate_schema(df_extra)
        assert is_valid is True


class TestLoadAndClean:
    def test_clean_removes_duplicates(self, tmp_path, sample_df):
        # Create duplicate
        dup_df = pd.concat([sample_df, sample_df.iloc[:2]], ignore_index=True)
        csv_path = tmp_path / "test_jobs.csv"
        dup_df.to_csv(csv_path, index=False)
        cleaned = _load_and_clean(csv_path)
        # Should have original 6 records, not 8
        assert len(cleaned) == 6

    def test_clean_fills_nan_descriptions(self, tmp_path, sample_df):
        df_nan = sample_df.copy()
        df_nan.loc[0, "job_description"] = None
        csv_path = tmp_path / "test_nan.csv"
        df_nan.to_csv(csv_path, index=False)
        cleaned = _load_and_clean(csv_path)
        assert cleaned["job_description"].isna().sum() == 0

    def test_clean_raises_on_missing_column(self, tmp_path, sample_df):
        df_bad = sample_df.drop(columns=["job_skill_set"])
        csv_path = tmp_path / "test_bad.csv"
        df_bad.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            _load_and_clean(csv_path)

    def test_load_actual_sample_csv(self):
        """Test that the bundled sample dataset loads successfully."""
        from src.config import SAMPLE_CSV
        if not SAMPLE_CSV.exists():
            pytest.skip("sample_jobs.csv not found")
        df = _load_and_clean(SAMPLE_CSV)
        assert len(df) > 0
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_load_drops_fully_empty_rows(self, tmp_path, sample_df):
        df_empty = sample_df.copy()
        df_empty.loc[0, "job_description"] = None
        df_empty.loc[0, "job_skill_set"] = None
        csv_path = tmp_path / "test_empty.csv"
        df_empty.to_csv(csv_path, index=False)
        cleaned = _load_and_clean(csv_path)
        # Row 0 should be dropped
        assert len(cleaned) == len(sample_df) - 1


class TestDatasetStats:
    def test_stats_returns_expected_keys(self, sample_df):
        stats = get_dataset_stats(sample_df)
        assert "total_records" in stats
        assert "categories" in stats
        assert "num_categories" in stats
        assert "avg_skills_per_job" in stats

    def test_stats_total_records(self, sample_df):
        stats = get_dataset_stats(sample_df)
        assert stats["total_records"] == len(sample_df)

    def test_stats_num_categories(self, sample_df):
        stats = get_dataset_stats(sample_df)
        assert stats["num_categories"] == 6
