"""
Baseline model training script for binary classification project.

This script trains Decision Tree with default parameters and evaluates using
5-fold cross-validation to establish baseline performance.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from pathlib import Path
import sys

from .config import (
    RANDOM_SEED,
    TRAIN_FILE,
    CV_FOLDS,
    CV_SHUFFLE,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)
from .preprocessing import DataPreprocessor, load_data


def train_decision_tree_baseline():
    """
    Train Decision Tree with default parameters and evaluate using cross-validation.

    Returns:
        dict: Training results including model, scores, and metadata
    """
    print("\n" + "=" * 70)
    print("DECISION TREE BASELINE TRAINING")
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
    print("\n[3/4] Initializing Decision Tree with default parameters...")
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    print(f"  ✓ Model: {model.__class__.__name__}")
    print(f"  ✓ Parameters: {model.get_params()}")

    # Step 4: Cross-validation evaluation
    print(f"\n[4/4] Performing {CV_FOLDS}-fold cross-validation...")

    # 1. Create a StratifiedKFold object with CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    # 2. Use cross_validate to evaluate the model on X_train_processed and y_train
    #    - Specify scoring metrics: 'accuracy' and 'f1'
    #    - Pass the cv (cross-validation) object
    #    - Set return_train_score=False to save computation
    scores = cross_validate(
        model,
        X_train_processed,
        y_train,
        scoring=[PRIMARY_METRIC, SECONDARY_METRIC],
        cv=cv,
        return_train_score=False,
    )

    # 3. Extract accuracy and f1 scores from cv_results
    accuracy_scores = scores["test_accuracy"]
    f1_scores = scores["test_f1"]

    # 4. Calculate mean and std for both metrics
    accuracy_mean = np.mean(accuracy_scores)
    accuracy_std = np.std(accuracy_scores)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    # 5. Store results in a dictionary with keys:
    #    - 'accuracy_mean', 'accuracy_std'
    #    - 'f1_mean', 'f1_std'
    #    - 'cv_results' (the full cross_validate output)
    cv_scores = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
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
        "model_name": "DecisionTree",
        "model": model,
        "preprocessor": preprocessor,
        "cv_scores": cv_scores,
        "parameters": model.get_params(),
    }

    return results


def main():
    """Main execution function."""
    try:
        results = train_decision_tree_baseline()

        print("\n✓ Baseline training complete!")
        print(f"\nNext steps:")
        print(f"  1. Review the results above")
        print(
            f"  2. Train remaining baseline models (Random Forest, k-NN, Naïve Bayes)"
        )
        print(f"  3. Compare performance and select best models for tuning")

    except Exception as e:
        print(f"\n✗ Error during training: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
