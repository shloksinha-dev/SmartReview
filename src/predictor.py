"""
Prediction interface for SmartReview.

Provides:
- Single-review prediction with confidence scores.
- Batch prediction for DataFrames.
- Lazy model loading with a module-level cache.
- Input validation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.train_model import load_pipeline, model_exists, train, load_metadata
from src.data_loader import load_sample_dataset, split_dataset

logger = logging.getLogger(__name__)

# Module-level cache so the model is only loaded once per process
_pipeline = None

MAX_INPUT_LENGTH = 10_000   # characters
MIN_INPUT_LENGTH = 3        # characters


def get_pipeline():
    """
    Return the fitted pipeline, loading from disk or training automatically.

    On first call the model is loaded (or trained if absent) and cached.
    Subsequent calls return the cached object.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if model_exists():
        _pipeline = load_pipeline()
    else:
        logger.info("No saved model found — training on sample dataset…")
        df = load_sample_dataset()
        train_df, _ = split_dataset(df)
        _pipeline = train(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            save=True,
        )

    return _pipeline


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate a single review input string.

    Returns
    -------
    (is_valid, error_message)
    """
    if not isinstance(text, str):
        return False, "Input must be a text string."
    stripped = text.strip()
    if len(stripped) == 0:
        return False, "Please enter a review before analysing."
    if len(stripped) < MIN_INPUT_LENGTH:
        return False, f"Review is too short (minimum {MIN_INPUT_LENGTH} characters)."
    if len(stripped) > MAX_INPUT_LENGTH:
        return False, (
            f"Review exceeds the maximum length of {MAX_INPUT_LENGTH:,} characters. "
            "Please shorten it."
        )
    return True, ""


def predict_single(text: str) -> Dict:
    """
    Predict the sentiment of a single review.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict with keys:
        label       : str   — predicted sentiment class
        confidence  : float — probability for the predicted class  (0–1)
        probabilities : dict  — {class: probability} for all classes
        input_valid : bool
        error       : str | None
    """
    is_valid, error = validate_input(text)
    if not is_valid:
        return {
            "label": None,
            "confidence": None,
            "probabilities": {},
            "input_valid": False,
            "error": error,
        }

    pipeline = get_pipeline()
    proba_array = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    probabilities = {cls: float(p) for cls, p in zip(classes, proba_array)}
    predicted_label = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "input_valid": True,
        "error": None,
    }


def predict_batch(
    texts: List[str],
) -> pd.DataFrame:
    """
    Predict sentiment for a list of review strings.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    pd.DataFrame with columns:
        review_text, predicted_sentiment, confidence, <class>_probability, is_valid, error
    """
    pipeline = get_pipeline()
    classes = list(pipeline.classes_)

    results = []
    for text in texts:
        is_valid, error_msg = validate_input(text)
        if not is_valid:
            row = {
                "review_text": text,
                "predicted_sentiment": None,
                "confidence": None,
                "is_valid": False,
                "error": error_msg,
            }
            for cls in classes:
                row[f"{cls}_probability"] = None
            results.append(row)
            continue

        proba_array = pipeline.predict_proba([text])[0]
        probabilities = {cls: float(p) for cls, p in zip(classes, proba_array)}
        predicted = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted]

        row = {
            "review_text": text,
            "predicted_sentiment": predicted,
            "confidence": round(confidence, 4),
            "is_valid": True,
            "error": None,
        }
        for cls in classes:
            row[f"{cls}_probability"] = round(probabilities[cls], 4)
        results.append(row)

    return pd.DataFrame(results)


def reset_pipeline():
    """
    Clear the module-level model cache (mainly useful in tests).
    """
    global _pipeline
    _pipeline = None
