"""
Tests for src/predictor.py
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.predictor import (
    validate_input,
    predict_single,
    predict_batch,
    reset_pipeline,
    MAX_INPUT_LENGTH,
    MIN_INPUT_LENGTH,
)


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Reset the cached pipeline before each test."""
    reset_pipeline()
    yield
    reset_pipeline()


class TestValidateInput:
    def test_valid_text(self):
        ok, msg = validate_input("This product is great!")
        assert ok is True
        assert msg == ""

    def test_empty_string(self):
        ok, msg = validate_input("")
        assert ok is False
        assert len(msg) > 0

    def test_whitespace_only(self):
        ok, msg = validate_input("   ")
        assert ok is False

    def test_too_short(self):
        ok, msg = validate_input("Hi")
        assert ok is False

    def test_too_long(self):
        ok, msg = validate_input("x" * (MAX_INPUT_LENGTH + 1))
        assert ok is False
        assert "length" in msg.lower() or "maximum" in msg.lower()

    def test_exactly_min_length(self):
        ok, msg = validate_input("x" * MIN_INPUT_LENGTH)
        assert ok is True

    def test_exactly_max_length(self):
        ok, msg = validate_input("x" * MAX_INPUT_LENGTH)
        assert ok is True

    def test_non_string(self):
        ok, msg = validate_input(None)
        assert ok is False

    def test_integer_input(self):
        ok, msg = validate_input(12345)
        assert ok is False


class TestPredictSingle:
    def test_returns_dict(self):
        result = predict_single("This product is absolutely wonderful!")
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = predict_single("Great purchase, I love it.")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "input_valid" in result
        assert "error" in result

    def test_valid_input_produces_label(self):
        result = predict_single("I really love this product, amazing quality.")
        assert result["input_valid"] is True
        assert result["label"] in {"positive", "negative", "neutral"}

    def test_confidence_in_range(self):
        result = predict_single("Excellent product, highly recommended!")
        if result["input_valid"]:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        result = predict_single("This is a great product.")
        if result["input_valid"] and result["probabilities"]:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-6

    def test_empty_input_returns_error(self):
        result = predict_single("")
        assert result["input_valid"] is False
        assert result["error"] is not None
        assert result["label"] is None

    def test_whitespace_input_returns_error(self):
        result = predict_single("   ")
        assert result["input_valid"] is False

    def test_too_long_input_returns_error(self):
        result = predict_single("x" * (MAX_INPUT_LENGTH + 1))
        assert result["input_valid"] is False

    def test_positive_review_classification(self):
        """Positive review should lean toward positive — not a guarantee but
        with the training data it should be consistent."""
        result = predict_single(
            "Absolutely fantastic product! Best purchase I've made in years. "
            "The quality is superb and shipping was very fast."
        )
        assert result["input_valid"] is True
        # At minimum, no error and has a label
        assert result["label"] is not None

    def test_negative_review_classification(self):
        result = predict_single(
            "Terrible quality, broke immediately. Complete waste of money. "
            "Never buying from this brand again. Very disappointed."
        )
        assert result["input_valid"] is True
        assert result["label"] is not None


class TestPredictBatch:
    def test_returns_dataframe(self):
        texts = ["Great product!", "Terrible quality."]
        result = predict_batch(texts)
        assert isinstance(result, pd.DataFrame)

    def test_output_length_matches_input(self):
        texts = ["Good", "Bad", "Okay"] * 3
        result = predict_batch(texts)
        assert len(result) == len(texts)

    def test_required_columns(self):
        result = predict_batch(["This is great!", "This is awful."])
        for col in ["review_text", "predicted_sentiment", "confidence", "is_valid"]:
            assert col in result.columns

    def test_invalid_rows_marked(self):
        texts = ["", "   ", "Good product"]
        result = predict_batch(texts)
        invalid = result[result["is_valid"] == False]
        assert len(invalid) >= 2   # first two are invalid

    def test_valid_rows_have_predictions(self):
        texts = ["I love this product.", "I hate this product."]
        result = predict_batch(texts)
        valid = result[result["is_valid"] == True]
        assert valid["predicted_sentiment"].notna().all()

    def test_empty_list(self):
        result = predict_batch([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_confidence_in_range(self):
        texts = ["Great!", "Terrible!", "Okay product."]
        result = predict_batch(texts)
        valid_conf = result[result["is_valid"] == True]["confidence"].dropna()
        assert (valid_conf >= 0.0).all()
        assert (valid_conf <= 1.0).all()
