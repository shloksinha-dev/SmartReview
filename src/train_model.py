"""
Model training pipeline for SmartReview.

Pipeline:
  TF-IDF Vectorizer  →  Logistic Regression

Design choices:
- A scikit-learn Pipeline ensures the vectorizer is fitted ONLY on training
  data, preventing data leakage.
- sublinear_tf=True in TF-IDF dampens the effect of very high term counts.
- ngram_range=(1,2) captures bigrams ("not good") for better negation handling.
- max_features caps memory usage at 50 000 features.
- Logistic Regression with lbfgs solver handles multi-class natively.
- max_iter=1000 avoids convergence warnings for most review datasets.
- random_state=42 makes training reproducible.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """
    Construct the TF-IDF + Logistic Regression scikit-learn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,   # apply preprocessing inside the pipeline
        ngram_range=(1, 2),        # unigrams + bigrams
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,                  # ignore terms in fewer than 2 docs
        strip_accents="unicode",
    )
    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        C=1.0,
        class_weight="balanced",   # handles class imbalance
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train(
    train_texts,
    train_labels,
    save: bool = True,
    extra_metadata: Optional[dict] = None,
) -> Pipeline:
    """
    Fit the pipeline on training data and optionally save it.

    Parameters
    ----------
    train_texts : array-like of str
    train_labels : array-like of str  (e.g. "positive", "negative")
    save : bool
        Persist the fitted pipeline and metadata to disk.
    extra_metadata : dict, optional
        Additional key-value pairs to include in the metadata JSON.

    Returns
    -------
    Fitted Pipeline object.
    """
    pipeline = build_pipeline()
    logger.info("Fitting pipeline on %d training samples…", len(train_texts))
    pipeline.fit(train_texts, train_labels)
    logger.info("Training complete.")

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info("Pipeline saved to %s", MODEL_PATH)

        metadata = {
            "training_samples": int(len(train_texts)),
            "classes": list(pipeline.classes_.tolist()),
            "random_state": RANDOM_STATE,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(METADATA_PATH, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Metadata saved to %s", METADATA_PATH)

    return pipeline


def load_pipeline() -> Pipeline:
    """
    Load a previously saved pipeline from disk.

    Raises
    ------
    FileNotFoundError
        If no saved pipeline exists.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_model.py first."
        )
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded from %s", MODEL_PATH)
    return pipeline


def load_metadata() -> dict:
    """
    Load training metadata if available.
    """
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def model_exists() -> bool:
    """Return True if a trained model file is present on disk."""
    return MODEL_PATH.exists()
