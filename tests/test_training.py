"""
Tests for training pipeline integration.
"""
import pytest
from pathlib import Path
from src.data_loader import load_sample_dataset, split_dataset
from src.train_model import build_pipeline, train
from src.evaluation import evaluate_pipeline


@pytest.fixture(scope="module")
def trained_pipeline_and_test():
    """Train once for the module; return (pipeline, test_df)."""
    df = load_sample_dataset()
    train_df, test_df = split_dataset(df, test_size=0.2, random_state=42)
    pipeline = train(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        save=False,  # Don't write to disk during tests
    )
    return pipeline, test_df


class TestBuildPipeline:
    def test_pipeline_has_two_steps(self):
        p = build_pipeline()
        assert len(p.steps) == 2

    def test_first_step_is_tfidf(self):
        p = build_pipeline()
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert isinstance(p.named_steps["tfidf"], TfidfVectorizer)

    def test_second_step_is_lr(self):
        p = build_pipeline()
        from sklearn.linear_model import LogisticRegression
        assert isinstance(p.named_steps["clf"], LogisticRegression)


class TestTrain:
    def test_pipeline_can_predict_after_training(self, trained_pipeline_and_test):
        pipeline, _ = trained_pipeline_and_test
        result = pipeline.predict(["This product is fantastic!"])
        assert len(result) == 1
        assert result[0] in {"positive", "negative", "neutral"}

    def test_pipeline_has_classes(self, trained_pipeline_and_test):
        pipeline, _ = trained_pipeline_and_test
        assert hasattr(pipeline, "classes_")
        assert len(pipeline.classes_) >= 2

    def test_predict_proba_sums_to_one(self, trained_pipeline_and_test):
        pipeline, _ = trained_pipeline_and_test
        proba = pipeline.predict_proba(["Excellent quality!"])[0]
        assert abs(sum(proba) - 1.0) < 1e-6


class TestEvaluatePipeline:
    def test_returns_dict(self, trained_pipeline_and_test):
        pipeline, test_df = trained_pipeline_and_test
        results = evaluate_pipeline(
            pipeline,
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            save=False,
        )
        assert isinstance(results, dict)

    def test_accuracy_in_range(self, trained_pipeline_and_test):
        pipeline, test_df = trained_pipeline_and_test
        results = evaluate_pipeline(
            pipeline,
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            save=False,
        )
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_required_keys_present(self, trained_pipeline_and_test):
        pipeline, test_df = trained_pipeline_and_test
        results = evaluate_pipeline(
            pipeline,
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            save=False,
        )
        for key in ["accuracy", "precision", "recall", "f1", "classification_report"]:
            assert key in results

    def test_confusion_matrix_shape(self, trained_pipeline_and_test):
        pipeline, test_df = trained_pipeline_and_test
        results = evaluate_pipeline(
            pipeline,
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            save=False,
        )
        n_classes = len(results["classes"])
        cm = results["confusion_matrix"]
        assert len(cm) == n_classes
        assert len(cm[0]) == n_classes

    def test_empty_test_set_returns_empty_dict(self, trained_pipeline_and_test):
        pipeline, _ = trained_pipeline_and_test
        results = evaluate_pipeline(pipeline, [], [], save=False)
        assert results == {}
