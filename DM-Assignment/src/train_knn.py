"""
k-Nearest Neighbors baseline model training script.

This script trains k-NN with default parameters and evaluates using
5-fold cross-validation to establish baseline performance.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import sys

from .utils.config import (
    RANDOM_SEED,
    TRAIN_FILE,
    CV_FOLDS,
    CV_SHUFFLE,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)
from .preprocessing import DataPreprocessor, load_data


def train_knn_baseline():
    """
    Train k-NN with default parameters and evaluate using cross-validation.

    Returns:
        dict: Training results including model, scores, and metadata
    """
    print("\n" + "=" * 70)
    print("K-NEAREST NEIGHBORS BASELINE TRAINING")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/4] Loading training data...")
    X_train, y_train = load_data(TRAIN_FILE)
    print(f"  ✓ Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"  ✓ Class distribution: {dict(y_train.value_counts())}")

    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"  ✓ Preprocessing complete")
    print(
        f"  ✓ Missing values after preprocessing: {X_train_processed.isnull().sum().sum()}"
    )

    # Step 3: Initialize model
    print("\n[3/4] Initializing k-NN with default parameters...")
    model = KNeighborsClassifier()
    print(f"  ✓ Model: {model.__class__.__name__}")
    print(f"  ✓ Key parameters:")
    print(f"      - n_neighbors: {model.n_neighbors}")
    print(f"      - weights: {model.weights}")
    print(f"      - metric: {model.metric}")
    print(f"      - algorithm: {model.algorithm}")

    # Step 4: Cross-validation evaluation
    print(f"\n[4/4] Performing {CV_FOLDS}-fold cross-validation...")
    print("  (This may take some time as k-NN is instance-based...)")

    # 1. Create a StratifiedKFold object with CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    # 2. Use cross_validate to evaluate the model with PRIMARY_METRIC and SECONDARY_METRIC
    scores = cross_validate(
        model,
        X_train_processed,
        y_train,
        cv=cv,
        scoring=[PRIMARY_METRIC, SECONDARY_METRIC],
        return_train_score=False,
    )

    # Aggregate cross-validation results
    cv_scores = {
        "f1_mean": np.mean(scores["test_f1"]),
        "f1_std": np.std(scores["test_f1"]),
        "accuracy_mean": np.mean(scores["test_accuracy"]),
        "accuracy_std": np.std(scores["test_accuracy"]),
        "cv_results": scores,
    }

    # Display results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(
        f"Accuracy: {cv_scores.get('accuracy_mean', 0):.4f} ± {cv_scores.get('accuracy_std', 0):.4f}"
    )
    print(
        f"F1 Score: {cv_scores.get('f1_mean', 0):.4f} ± {cv_scores.get('f1_std', 0):.4f}"
    )
    print("=" * 70)

    # Prepare return results
    results = {
        "model_name": "KNN",
        "model": model,
        "preprocessor": preprocessor,
        "cv_scores": cv_scores,
        "parameters": model.get_params(),
    }

    return results


def main():
    """Main execution function."""
    try:
        results = train_knn_baseline()

        print("\n✓ k-NN baseline training complete!")
        print(f"\nNext steps:")
        print(f"  1. Review the results above")
        print(f"  2. Compare with Decision Tree and Random Forest")
        print(f"  3. Train final model (Naïve Bayes)")

    except Exception as e:
        print(f"\n✗ Error during training: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
