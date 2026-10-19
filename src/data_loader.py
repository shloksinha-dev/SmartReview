"""
Dataset loading and validation for SmartReview.

Responsibilities:
- Load CSV datasets.
- Validate schema (required columns, data types).
- Report class distribution.
- Handle duplicates, missing values, and unsupported labels.
- Return a clean DataFrame ready for train/test splitting.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Accepted label values (lowercase)
VALID_LABELS = {"positive", "negative", "neutral"}

# Minimum number of rows required per class for stratified splitting
MIN_ROWS_PER_CLASS = 5


def load_dataset(
    path: str | Path,
    text_col: str = "review_text",
    label_col: str = "sentiment",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load and validate a sentiment dataset from CSV.

    Parameters
    ----------
    path : str or Path
    text_col : str
        Column name containing the review text.
    label_col : str
        Column name containing the sentiment label.
    encoding : str

    Returns
    -------
    pd.DataFrame with columns [text_col, label_col], cleaned and validated.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing or no valid rows remain.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Try UTF-8 first, fall back to latin-1
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed; retrying with latin-1.")
        df = pd.read_csv(path, encoding="latin-1")

    logger.info("Loaded %d rows from %s", len(df), path)

    # ── Column validation ────────────────────────────────────────────────────
    missing_cols = []
    if text_col not in df.columns:
        missing_cols.append(text_col)
    if label_col not in df.columns:
        missing_cols.append(label_col)

    if missing_cols:
        raise ValueError(
            f"Required column(s) not found: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the relevant columns
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # ── Missing value handling ───────────────────────────────────────────────
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    dropped_missing = initial_count - len(df)
    if dropped_missing:
        logger.warning("Dropped %d rows with missing/empty text or label.", dropped_missing)

    # ── Duplicate handling ───────────────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"])
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes:
        logger.warning("Dropped %d duplicate review rows.", dropped_dupes)

    # ── Label normalisation and validation ──────────────────────────────────
    df["label"] = df["label"].str.strip().str.lower()
    invalid_mask = ~df["label"].isin(VALID_LABELS)
    invalid_count = invalid_mask.sum()
    if invalid_count:
        invalid_samples = df[invalid_mask]["label"].unique().tolist()
        logger.warning(
            "Dropping %d rows with unrecognised labels: %s",
            invalid_count,
            invalid_samples,
        )
        df = df[~invalid_mask]

    if len(df) == 0:
        raise ValueError("No valid rows remain after cleaning. Check your dataset.")

    # ── Class distribution report ────────────────────────────────────────────
    dist = df["label"].value_counts().to_dict()
    logger.info("Class distribution: %s", dist)

    return df.reset_index(drop=True)


def get_class_distribution(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Return {label: count} dict for the given DataFrame."""
    return df[label_col].value_counts().to_dict()


def can_stratify(df: pd.DataFrame, label_col: str = "label") -> bool:
    """
    Return True if every class has enough samples for stratified splitting.
    """
    counts = df[label_col].value_counts()
    return bool((counts >= MIN_ROWS_PER_CLASS).all())


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a validated DataFrame into train and test subsets.

    Uses stratification when class sizes allow it; falls back to a plain
    shuffle split with a warning.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'text' and 'label' columns)
    test_size : float
    random_state : int

    Returns
    -------
    (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    stratify = df["label"] if can_stratify(df) else None
    if stratify is None:
        logger.warning(
            "Some classes have fewer than %d samples. "
            "Falling back to non-stratified split.",
            MIN_ROWS_PER_CLASS,
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    logger.info(
        "Split → train: %d rows, test: %d rows", len(train_df), len(test_df)
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_sample_dataset() -> pd.DataFrame:
    """
    Load the bundled demonstration dataset.

    This is a small curated dataset for portfolio demonstration purposes.
    It is NOT a production-grade benchmark.
    """
    sample_path = Path(__file__).parent.parent / "data" / "sample_reviews.csv"
    return load_dataset(sample_path)
