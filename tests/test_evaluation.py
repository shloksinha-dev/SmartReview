"""
Tests for src/evaluation.py extensions (baseline comparison and cross-validation).
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from src.data_loader import load_sample_dataset, split_dataset
from src.train_model import train
from src.evaluation import (
    evaluate_baseline,
    compute_cross_validation,
    evaluate_pipeline,
)


@pytest.fixture(scope="module")
def sample_data_and_pipeline():
    df = load_sample_dataset()
    train_df, test_df = split_dataset(df, test_size=0.2, random_state=42)
    pipeline = train(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        save=False,
    )
    return {
        "pipeline": pipeline,
        "train_texts": train_df["text"].tolist(),
        "train_labels": train_df["label"].tolist(),
        "test_texts": test_df["text"].tolist(),
        "test_labels": test_df["label"].tolist(),
    }


class TestEvaluateBaseline:
    def test_baseline_majority_class(self):
        train_labels = ["positive"] * 10 + ["negative"] * 40
        test_labels = ["positive"] * 3 + ["negative"] * 7
        res = evaluate_baseline(train_labels, test_labels)
        assert res["majority_class"] == "negative"
        assert res["strategy"] == "most_frequent"
        # 7 out of 10 test samples are negative -> accuracy 0.70
        assert res["accuracy"] == 0.70
        assert 0.0 <= res["macro_f1"] <= 1.0

    def test_baseline_empty_returns_empty_dict(self):
        assert evaluate_baseline([], ["positive"]) == {}
        assert evaluate_baseline(["positive"], []) == {}

    def test_baseline_keys(self):
        train_labels = ["positive", "positive", "negative"]
        test_labels = ["positive", "negative"]
        res = evaluate_baseline(train_labels, test_labels)
        for key in ["strategy", "majority_class", "accuracy", "macro_f1", "weighted_f1"]:
            assert key in res


class TestComputeCrossValidation:
    def test_cv_returns_scores(self, sample_data_and_pipeline):
        data = sample_data_and_pipeline
        cv_res = compute_cross_validation(
            data["pipeline"],
            data["train_texts"],
            data["train_labels"],
            n_splits=5,
        )
        assert isinstance(cv_res, dict)
        assert cv_res["folds"] == 5
        assert 0.0 <= cv_res["accuracy_mean"] <= 1.0
        assert 0.0 <= cv_res["f1_macro_mean"] <= 1.0
        assert cv_res["accuracy_std"] >= 0.0

    def test_cv_empty_input(self, sample_data_and_pipeline):
        data = sample_data_and_pipeline
        assert compute_cross_validation(data["pipeline"], [], []) == {}

    def test_cv_insufficient_samples(self, sample_data_and_pipeline):
        data = sample_data_and_pipeline
        # Only 1 sample per class -> cannot split into >= 2 folds
        res = compute_cross_validation(
            data["pipeline"],
            ["good", "bad"],
            ["positive", "negative"],
            n_splits=5,
        )
        assert res == {}


class TestEvaluatePipelineExtensions:
    def test_macro_metrics_present(self, sample_data_and_pipeline):
        data = sample_data_and_pipeline
        results = evaluate_pipeline(
            data["pipeline"],
            data["test_texts"],
            data["test_labels"],
            save=False,
        )
        for key in ["macro_precision", "macro_recall", "macro_f1"]:
            assert key in results
            assert 0.0 <= results[key] <= 1.0

    def test_evaluate_pipeline_with_baseline_and_cv(self, sample_data_and_pipeline):
        data = sample_data_and_pipeline
        results = evaluate_pipeline(
            data["pipeline"],
            data["test_texts"],
            data["test_labels"],
            save=False,
            train_texts=data["train_texts"],
            train_labels=data["train_labels"],
            cv_folds=3,
        )
        assert "baseline" in results
        assert "cross_validation" in results
        assert results["baseline"]["majority_class"] in {"positive", "negative"}
        assert results["cross_validation"]["folds"] == 3
