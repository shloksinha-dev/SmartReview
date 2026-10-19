"""
Tests for src/data_loader.py
"""

import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_dataset,
    load_sample_dataset,
    get_class_distribution,
    can_stratify,
    split_dataset,
)


@pytest.fixture
def tmp_valid_csv(tmp_path):
    """Create a valid temporary CSV with required columns."""
    df = pd.DataFrame({
        "review_text": [
            "Great product!",
            "Terrible quality.",
            "Amazing experience.",
            "Worst purchase ever.",
            "Love it!",
            "Hate it.",
            "Very good item.",
            "Not happy at all.",
            "Excellent!",
            "Dreadful.",
        ],
        "sentiment": [
            "positive", "negative", "positive", "negative", "positive",
            "negative", "positive", "negative", "positive", "negative",
        ]
    })
    path = tmp_path / "test_reviews.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_missing_col_csv(tmp_path):
    """CSV missing the label column."""
    df = pd.DataFrame({"review_text": ["Hello"], "wrong_col": ["positive"]})
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_bad_labels_csv(tmp_path):
    """CSV with invalid labels."""
    df = pd.DataFrame({
        "review_text": ["Great", "Awful", "Okay", "Good", "Bad"],
        "sentiment": ["pos", "neg", "meh", "positive", "negative"]
    })
    path = tmp_path / "bad_labels.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadDataset:
    def test_loads_valid_csv(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/file.csv")

    def test_raises_missing_column(self, tmp_missing_col_csv):
        with pytest.raises(ValueError, match="Required column"):
            load_dataset(tmp_missing_col_csv)

    def test_invalid_labels_dropped(self, tmp_bad_labels_csv):
        df = load_dataset(tmp_bad_labels_csv)
        # Only "positive" and "negative" should survive
        assert set(df["label"].unique()).issubset({"positive", "negative", "neutral"})

    def test_label_normalised_to_lowercase(self, tmp_path):
        df_raw = pd.DataFrame({
            "review_text": ["Good", "Bad"],
            "sentiment": ["Positive", "NEGATIVE"]
        })
        p = tmp_path / "case.csv"
        df_raw.to_csv(p, index=False)
        df = load_dataset(p)
        assert all(l == l.lower() for l in df["label"])

    def test_empty_rows_dropped(self, tmp_path):
        df_raw = pd.DataFrame({
            "review_text": ["", "   ", "Good product", None],
            "sentiment": ["positive", "negative", "positive", "positive"]
        })
        p = tmp_path / "empty.csv"
        df_raw.to_csv(p, index=False)
        df = load_dataset(p)
        assert all(len(t.strip()) > 0 for t in df["text"])

    def test_duplicates_dropped(self, tmp_path):
        df_raw = pd.DataFrame({
            "review_text": ["Great", "Great", "Great", "Awful"],
            "sentiment": ["positive", "positive", "positive", "negative"]
        })
        p = tmp_path / "dup.csv"
        df_raw.to_csv(p, index=False)
        df = load_dataset(p)
        assert df["text"].nunique() == len(df)


class TestLoadSampleDataset:
    def test_loads_without_error(self):
        df = load_sample_dataset()
        assert len(df) > 0
        assert "text" in df.columns
        assert "label" in df.columns

    def test_valid_labels_only(self):
        df = load_sample_dataset()
        valid = {"positive", "negative", "neutral"}
        assert set(df["label"].unique()).issubset(valid)


class TestClassDistribution:
    def test_returns_dict(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        dist = get_class_distribution(df)
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(df)


class TestCanStratify:
    def test_returns_true_for_large_classes(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        # Each class has 5 rows — enough for MIN_ROWS_PER_CLASS=5
        assert can_stratify(df) is True

    def test_returns_false_for_tiny_class(self, tmp_path):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e", "f"],
            "label": ["positive"] * 5 + ["negative"]  # only 1 negative
        })
        assert can_stratify(df) is False


class TestSplitDataset:
    def test_split_sizes(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        train, test = split_dataset(df, test_size=0.2)
        assert len(train) + len(test) == len(df)

    def test_no_overlap(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        train, test = split_dataset(df, test_size=0.2)
        train_texts = set(train["text"].tolist())
        test_texts = set(test["text"].tolist())
        assert len(train_texts & test_texts) == 0

    def test_reproducible_with_same_seed(self, tmp_valid_csv):
        df = load_dataset(tmp_valid_csv)
        train1, test1 = split_dataset(df, random_state=42)
        train2, test2 = split_dataset(df, random_state=42)
        assert train1["text"].tolist() == train2["text"].tolist()
