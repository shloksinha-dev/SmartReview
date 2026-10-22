"""
Standalone training script for SmartReview.

Run from the project root:

    python train_pipeline.py

This script:
1. Loads the sample dataset (or a custom path via --data).
2. Validates and splits the data (80/20 train/test, stratified where possible).
3. Trains the TF-IDF + Logistic Regression pipeline.
4. Evaluates on the held-out test set.
5. Saves the fitted pipeline to models/sentiment_pipeline.joblib.
6. Saves evaluation results to models/evaluation_results.json.
7. Prints a summary to stdout.

The script is intentionally independent of Streamlit so it can be run in any
terminal, in CI, or as a scheduled job.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, load_sample_dataset, split_dataset
from src.evaluation import evaluate_pipeline
from src.train_model import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SmartReview sentiment pipeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a custom CSV dataset (default: bundled sample).",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="review_text",
        help="Column name for review text.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="sentiment",
        help="Column name for sentiment labels.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data to hold out for testing (default: 0.20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.data:
        logger.info("Loading custom dataset: %s", args.data)
        df = load_dataset(
            args.data,
            text_col=args.text_col,
            label_col=args.label_col,
        )
    else:
        logger.info("Loading bundled sample dataset…")
        df = load_sample_dataset()

    print(f"\n" + "-"*50)
    print(f"  Dataset loaded: {len(df)} rows")
    dist = df["label"].value_counts().to_dict()
    for label, count in sorted(dist.items()):
        print(f"    {label:10s}: {count:4d} rows ({count/len(df)*100:.1f}%)")
    print("-"*50 + "\n")

    # ── Train/test split ─────────────────────────────────────────────────────
    train_df, test_df = split_dataset(df, test_size=args.test_size)
    print(f"  Training samples : {len(train_df)}")
    print(f"  Test samples     : {len(test_df)}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    print("  Training pipeline...")
    pipeline = train(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        save=True,
        extra_metadata={
            "test_size": args.test_size,
            "dataset": args.data or "bundled sample",
        },
    )
    print("  [OK] Pipeline trained and saved.\n")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if len(test_df) == 0:
        print("  [WARN] No test samples available - skipping evaluation.")
        return

    print("  Evaluating on held-out test set and cross-validating...")
    results = evaluate_pipeline(
        pipeline,
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        save=True,
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        cv_folds=5,
    )

    print("\n" + "="*58)
    print(f"  HELD-OUT TEST SET EVALUATION  (test n={results['test_samples']})")
    print("="*58)
    print(f"  Accuracy (Test)       : {results['accuracy']:.4f}")
    print(f"  F1 Score (Weighted)   : {results['f1']:.4f}")
    print(f"  F1 Score (Macro)      : {results['macro_f1']:.4f}")
    print(f"  Precision (Weighted)  : {results['precision']:.4f}")
    print(f"  Recall (Weighted)     : {results['recall']:.4f}")

    if "baseline" in results:
        base = results["baseline"]
        print("\n" + "-"*58)
        print("  BASELINE MODEL COMPARISON (Dummy Classifier)")
        print("-"*58)
        print(f"  Strategy              : {base['strategy']}")
        print(f"  Majority Class        : {base['majority_class']}")
        print(f"  Baseline Accuracy     : {base['accuracy']:.4f}")
        print(f"  Baseline Macro F1     : {base['macro_f1']:.4f}")
        diff_acc = results['accuracy'] - base['accuracy']
        diff_f1 = results['macro_f1'] - base['macro_f1']
        print(f"  Diff Accuracy vs Base : {diff_acc:+.4f}")
        print(f"  Diff Macro F1 vs Base : {diff_f1:+.4f}")

    if "cross_validation" in results:
        cv = results["cross_validation"]
        print("\n" + "-"*58)
        print(f"  STRATIFIED {cv['folds']}-FOLD CROSS-VALIDATION (Training Set)")
        print("-"*58)
        print(f"  CV Accuracy           : {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}")
        print(f"  CV Macro F1           : {cv['f1_macro_mean']:.4f} +/- {cv['f1_macro_std']:.4f}")
        print(f"  CV Weighted F1        : {cv['f1_weighted_mean']:.4f} +/- {cv['f1_weighted_std']:.4f}")

    print("\n" + "-"*58)
    print("  PER-CLASS CLASSIFICATION REPORT")
    print("-"*58)
    print(results["classification_report"])
    print("="*58)
    print("\n  METHODOLOGICAL NOTE:")
    print("  * These metrics reflect the bundled sample demonstration dataset (87 rows).")
    print("  * The dataset is imbalanced (approx. 72% negative / 28% positive).")
    print("  * Macro F1 provides a more honest view than accuracy alone because a naive")
    print("    majority-class guesser matches raw test accuracy on this split.")
    print("  * Stratified cross-validation guards against single-split variance.\n")


if __name__ == "__main__":
    main()
