"""
Model evaluation utilities for SmartReview.

Provides:
- Comprehensive evaluation on a test set.
- Classification report.
- Confusion matrix.
- Persisting evaluation results to JSON.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

EVAL_PATH = Path(__file__).parent.parent / "models" / "evaluation_results.json"


def evaluate_pipeline(
    pipeline: Pipeline,
    test_texts: List[str],
    test_labels: List[str],
    save: bool = True,
) -> Dict:
    """
    Evaluate a fitted pipeline on a held-out test set.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    test_texts : list of str
    test_labels : list of str
    save : bool
        Persist results to models/evaluation_results.json

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, classification_report,
        confusion_matrix, classes, test_samples
    """
    if len(test_texts) == 0:
        logger.warning("Empty test set — skipping evaluation.")
        return {}

    predictions = pipeline.predict(test_texts)
    classes = sorted(set(test_labels) | set(predictions))
    labels_for_avg = [c for c in classes if c in set(test_labels)]

    # Handle the case where test set has only one class
    avg_method = "weighted" if len(labels_for_avg) > 1 else "micro"

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(
        test_labels, predictions, average=avg_method, zero_division=0, labels=labels_for_avg
    )
    recall = recall_score(
        test_labels, predictions, average=avg_method, zero_division=0, labels=labels_for_avg
    )
    f1 = f1_score(
        test_labels, predictions, average=avg_method, zero_division=0, labels=labels_for_avg
    )

    report = classification_report(
        test_labels,
        predictions,
        zero_division=0,
        output_dict=False,
    )

    cm = confusion_matrix(test_labels, predictions, labels=classes)

    results = {
        "test_samples": int(len(test_texts)),
        "classes": classes,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
    }

    logger.info(
        "Evaluation — accuracy: %.4f | F1: %.4f | samples: %d",
        accuracy,
        f1,
        len(test_texts),
    )

    if save:
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Save a JSON-serialisable copy (without the text report)
        save_data = {k: v for k, v in results.items() if k != "classification_report"}
        save_data["classification_report_text"] = report
        with open(EVAL_PATH, "w", encoding="utf-8") as fh:
            json.dump(save_data, fh, indent=2)
        logger.info("Evaluation results saved to %s", EVAL_PATH)

    return results


def load_evaluation_results() -> Optional[Dict]:
    """Load previously saved evaluation results, or return None."""
    if not EVAL_PATH.exists():
        return None
    with open(EVAL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def format_confusion_matrix(results: Dict) -> pd.DataFrame:
    """
    Convert raw confusion matrix data from evaluation results into a DataFrame.
    """
    cm = results.get("confusion_matrix")
    labels = results.get("confusion_matrix_labels")
    if cm is None or labels is None:
        return pd.DataFrame()
    return pd.DataFrame(cm, index=labels, columns=labels)
