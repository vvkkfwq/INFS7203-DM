"""
Metrics utilities for model evaluation and reporting.

This module provides functions for:
- Cross-validation setup
- Results aggregation and display
- Performance comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from typing import Dict, Any, Optional

from .config import (
    RANDOM_SEED,
    CV_FOLDS,
    CV_SHUFFLE,
    PRIMARY_METRIC,
    SECONDARY_METRIC,
)


def setup_cross_validation(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """
    Create a StratifiedKFold cross-validation object with project settings.

    Args:
        n_splits (int): Number of folds (default: CV_FOLDS from config)

    Returns:
        StratifiedKFold: Configured cross-validation object
    """
    return StratifiedKFold(
        n_splits=n_splits, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )


def evaluate_model_cv(
    model,
    X,
    y,
    cv=None,
    scoring: Optional[list] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Evaluate a model using cross-validation and return aggregated metrics.

    Args:
        model: sklearn model instance
        X: Feature matrix
        y: Target vector
        cv: Cross-validation strategy (default: None, will use setup_cross_validation())
        scoring: List of scoring metrics (default: [PRIMARY_METRIC, SECONDARY_METRIC])
        n_jobs: Number of parallel jobs (default: -1, all CPUs)

    Returns:
        dict: Dictionary containing:
            - cv_results: Raw cross_validate output
            - accuracy_mean, accuracy_std: Accuracy statistics
            - f1_mean, f1_std: F1 score statistics
            - fold_scores: Individual fold scores
    """
    if cv is None:
        cv = setup_cross_validation()

    if scoring is None:
        scoring = [PRIMARY_METRIC, SECONDARY_METRIC]

    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=n_jobs
    )

    # Aggregate results
    results = {
        "cv_results": cv_results,
        "accuracy_mean": np.mean(cv_results["test_accuracy"]),
        "accuracy_std": np.std(cv_results["test_accuracy"]),
        "f1_mean": np.mean(cv_results["test_f1"]),
        "f1_std": np.std(cv_results["test_f1"]),
        "fold_scores": {
            "accuracy": cv_results["test_accuracy"],
            "f1": cv_results["test_f1"],
        },
    }

    return results


def display_cv_results(
    results: Dict[str, Any],
    model_name: str,
    show_header: bool = True,
    show_fold_details: bool = False,
) -> None:
    """
    Display cross-validation results in a formatted manner.

    Args:
        results (dict): Results dictionary from evaluate_model_cv()
        model_name (str): Name of the model for display
        show_header (bool): Whether to show header separator
        show_fold_details (bool): Whether to show individual fold scores
    """
    if show_header:
        print("\n" + "=" * 70)
        print(f"{model_name.upper()} - CROSS-VALIDATION RESULTS")
        print("=" * 70)

    print(
        f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}"
    )
    print(f"F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    if show_fold_details:
        print("\nIndividual Fold Scores:")
        for i, (acc, f1) in enumerate(
            zip(
                results["fold_scores"]["accuracy"],
                results["fold_scores"]["f1"],
            ),
            1,
        ):
            print(f"  Fold {i}: Accuracy={acc:.4f}, F1={f1:.4f}")

    if show_header:
        print("=" * 70)


def compare_with_baseline(
    current_score: float,
    baseline_score: float,
    metric_name: str = "F1 Score",
) -> None:
    """
    Display comparison between current and baseline performance.

    Args:
        current_score (float): Current model score
        baseline_score (float): Baseline score to compare against
        metric_name (str): Name of the metric being compared
    """
    improvement = current_score - baseline_score
    percent_change = (improvement / baseline_score) * 100

    print(f"\nImprovement over Baseline:")
    print(
        f"  {metric_name}: {baseline_score:.4f} → {current_score:.4f} ({improvement:+.4f})"
    )
    print(f"  Relative change: {percent_change:+.2f}%")


def display_training_progress(
    step: int, total_steps: int, step_name: str, details: Optional[str] = None
) -> None:
    """
    Display training progress information.

    Args:
        step (int): Current step number
        total_steps (int): Total number of steps
        step_name (str): Name of the current step
        details (str, optional): Additional details to display
    """
    print(f"\n[{step}/{total_steps}] {step_name}...")
    if details:
        print(f"  {details}")


def create_comparison_table(results_list: list) -> pd.DataFrame:
    """
    Create a comparison table from multiple model results.

    Args:
        results_list (list): List of result dictionaries

    Returns:
        pd.DataFrame: Comparison table sorted by F1 score
    """
    df = pd.DataFrame(results_list)

    # Sort by F1 score (descending)
    if "f1_mean" in df.columns:
        df = df.sort_values("f1_mean", ascending=False)

    return df


def calculate_target_gap(
    current_score: float, target_score: float = 0.65
) -> Dict[str, Any]:
    """
    Calculate gap to target F1 score.

    Args:
        current_score (float): Current F1 score
        target_score (float): Target F1 score (default: 0.65)

    Returns:
        dict: Dictionary containing:
            - gap: Absolute gap to target
            - achieved: Boolean indicating if target is met
            - improvement_needed_pct: Percentage improvement needed
    """
    gap = target_score - current_score
    achieved = current_score >= target_score
    improvement_needed_pct = (gap / current_score) * 100 if gap > 0 else 0

    return {
        "gap": gap,
        "achieved": achieved,
        "improvement_needed_pct": improvement_needed_pct,
        "current_score": current_score,
        "target_score": target_score,
    }
