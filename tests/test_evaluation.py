"""
test_evaluation.py — Tests for evaluation metrics computation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import evaluate_skill_extraction
from src.clustering import calculate_silhouette, fit_kmeans
from src.feature_engineering import build_tfidf_matrix
import numpy as np


class TestEvaluateSkillExtraction:
    def test_returns_expected_keys(self):
        result = evaluate_skill_extraction()
        assert "micro_precision" in result
        assert "micro_recall" in result
        assert "micro_f1" in result
        assert "n_examples" in result
        assert "per_example" in result

    def test_precision_recall_between_0_and_1(self):
        result = evaluate_skill_extraction()
        assert 0.0 <= result["micro_precision"] <= 1.0
        assert 0.0 <= result["micro_recall"] <= 1.0
        assert 0.0 <= result["micro_f1"] <= 1.0

    def test_n_examples_matches_eval_set(self):
        from src.evaluation import SKILL_EXTRACTION_EVAL
        result = evaluate_skill_extraction()
        assert result["n_examples"] == len(SKILL_EXTRACTION_EVAL)

    def test_custom_eval_set_perfect_extraction(self):
        """A custom eval set where the extractor should score perfectly."""
        eval_set = [
            {"text": "Python developer needed", "expected": ["python"]},
        ]
        result = evaluate_skill_extraction(eval_set=eval_set)
        # Python should be extracted — precision and recall should both be 1.0
        assert result["micro_precision"] == pytest.approx(1.0)
        assert result["micro_recall"] == pytest.approx(1.0)

    def test_custom_eval_set_zero_extraction(self):
        """Custom eval set with a skill that will NOT be extracted."""
        eval_set = [
            {"text": "We need expertise in a completely unknown skill", "expected": ["superraredummyskillxyz"]},
        ]
        result = evaluate_skill_extraction(eval_set=eval_set)
        # Should produce false negative — recall = 0
        assert result["micro_recall"] == 0.0

    def test_per_example_has_correct_structure(self):
        result = evaluate_skill_extraction()
        for item in result["per_example"]:
            assert "precision" in item
            assert "recall" in item
            assert "f1" in item
            assert "tp" in item
            assert "fp" in item
            assert "fn" in item

    def test_tp_fp_fn_are_non_negative(self):
        result = evaluate_skill_extraction()
        assert result["total_tp"] >= 0
        assert result["total_fp"] >= 0
        assert result["total_fn"] >= 0


class TestCalculateSilhouette:
    def test_returns_float_for_valid_config(self, preprocessed_df, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        n_samples = matrix.shape[0]
        if n_samples < 4:
            pytest.skip("Need at least 4 samples for clustering test")
        _, labels = fit_kmeans(matrix, n_clusters=2)
        score = calculate_silhouette(matrix, labels)
        assert score is None or isinstance(score, float)

    def test_returns_none_for_single_cluster(self, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        labels = np.zeros(matrix.shape[0], dtype=int)
        score = calculate_silhouette(matrix, labels)
        assert score is None

    def test_valid_score_in_range(self, preprocessed_df, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        if matrix.shape[0] < 4:
            pytest.skip("Not enough samples")
        _, labels = fit_kmeans(matrix, n_clusters=2)
        score = calculate_silhouette(matrix, labels)
        if score is not None:
            assert -1.0 <= score <= 1.0


class TestFitKmeans:
    def test_basic_clustering(self, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        kmeans, labels = fit_kmeans(matrix, n_clusters=2)
        assert len(labels) == matrix.shape[0]
        assert set(labels).issubset({0, 1})

    def test_n_clusters_clamped_to_max_valid(self, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        n_samples = matrix.shape[0]
        # Request more clusters than samples-1
        kmeans, labels = fit_kmeans(matrix, n_clusters=1000)
        # Should be clamped
        assert kmeans.n_clusters <= n_samples - 1

    def test_raises_for_single_sample(self):
        texts = ["python developer needed"]
        _, matrix = build_tfidf_matrix(texts)
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            fit_kmeans(matrix, n_clusters=2)

    def test_reproducible_with_same_seed(self, tfidf_artifacts):
        _, matrix = tfidf_artifacts
        _, labels1 = fit_kmeans(matrix, n_clusters=2, random_state=42)
        _, labels2 = fit_kmeans(matrix, n_clusters=2, random_state=42)
        assert np.array_equal(labels1, labels2)
