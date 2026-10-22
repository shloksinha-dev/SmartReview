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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

EVAL_PATH = Path(__file__).parent.parent / "models" / "evaluation_results.json"


def evaluate_baseline(
    train_labels: List[str],
    test_labels: List[str],
    strategy: str = "most_frequent",
) -> Dict:
    """
    Evaluate a dummy baseline classifier on the test set.

    Parameters
    ----------
    train_labels : list of str
    test_labels : list of str
    strategy : str, default 'most_frequent'

    Returns
    -------
    dict with baseline metrics (accuracy, macro_f1, weighted_f1, majority_class)
    """
    if not train_labels or not test_labels:
        return {}

    dummy = DummyClassifier(strategy=strategy)
    dummy.fit([""] * len(train_labels), train_labels)
    preds = dummy.predict([""] * len(test_labels))

    acc = accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_labels, preds, average="weighted", zero_division=0)

    majority_class = (
        str(dummy.classes_[np.argmax(dummy.class_prior_)])
        if hasattr(dummy, "class_prior_")
        else str(preds[0])
    )

    return {
        "strategy": strategy,
        "majority_class": majority_class,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


def compute_cross_validation(
    pipeline: Pipeline,
    train_texts: List[str],
    train_labels: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Perform Stratified K-Fold Cross Validation on the training data.

    Parameters
    ----------
    pipeline : fitted or unfitted sklearn Pipeline
    train_texts : list of str
    train_labels : list of str
    n_splits : int
    random_state : int

    Returns
    -------
    dict with mean and std for accuracy, macro F1, and weighted F1 across folds.
    """
    if not train_texts or not train_labels:
        return {}

    counts = pd.Series(train_labels).value_counts()
    min_count = counts.min() if len(counts) > 0 else 0
    actual_splits = min(n_splits, min_count)
    if actual_splits < 2:
        logger.warning("Not enough samples per class for cross-validation.")
        return {}

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(
        pipeline,
        train_texts,
        train_labels,
        cv=skf,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=False,
    )
    return {
        "folds": int(actual_splits),
        "accuracy_mean": round(float(cv_scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(cv_scores["test_accuracy"].std()), 4),
        "f1_macro_mean": round(float(cv_scores["test_f1_macro"].mean()), 4),
        "f1_macro_std": round(float(cv_scores["test_f1_macro"].std()), 4),
        "f1_weighted_mean": round(float(cv_scores["test_f1_weighted"].mean()), 4),
        "f1_weighted_std": round(float(cv_scores["test_f1_weighted"].std()), 4),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    test_texts: List[str],
    test_labels: List[str],
    save: bool = True,
    train_texts: Optional[List[str]] = None,
    train_labels: Optional[List[str]] = None,
    cv_folds: int = 5,
) -> Dict:
    """
    Evaluate a fitted pipeline on a held-out test set, with optional baseline and CV.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    test_texts : list of str
    test_labels : list of str
    save : bool
        Persist results to models/evaluation_results.json
    train_texts : list of str, optional
        Used for computing Stratified K-Fold cross validation
    train_labels : list of str, optional
        Used for dummy baseline comparison and cross validation
    cv_folds : int
        Number of cross-validation splits (default 5)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, macro_precision, macro_recall, macro_f1,
        classification_report, confusion_matrix, classes, test_samples,
        and optionally baseline, cross_validation.
    """
    if len(test_texts) == 0:
        logger.warning("Empty test set — skipping evaluation.")
        return {}

    predictions = pipeline.predict(test_texts)
    classes = sorted(set(test_labels) | set(predictions))
    labels_for_avg = [c for c in classes if c in set(test_labels)]

    # Weighted metrics
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

    # Macro metrics (unweighted, critical for imbalanced classes)
    precision_macro = precision_score(
        test_labels, predictions, average="macro", zero_division=0, labels=labels_for_avg
    )
    recall_macro = recall_score(
        test_labels, predictions, average="macro", zero_division=0, labels=labels_for_avg
    )
    f1_macro = f1_score(
        test_labels, predictions, average="macro", zero_division=0, labels=labels_for_avg
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
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
    }

    if train_labels is not None and len(train_labels) > 0:
        results["baseline"] = evaluate_baseline(train_labels, test_labels)

    if train_texts is not None and train_labels is not None:
        cv_res = compute_cross_validation(
            pipeline, train_texts, train_labels, n_splits=cv_folds
        )
        if cv_res:
            results["cross_validation"] = cv_res

    logger.info(
        "Evaluation — accuracy: %.4f | F1 (weighted): %.4f | F1 (macro): %.4f | samples: %d",
        accuracy,
        f1,
        f1_macro,
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
