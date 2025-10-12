"""
Voting Classifier Tuning - Version 3

This script tunes a Voting Classifier that combines the best performing models:
- Random Forest (best F1: 0.6419)
- Decision Tree (best F1: 0.6092) without class_weight
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.preprocessing import DataPreprocessor, load_data
from src.utils.config import (
    RANDOM_SEED,
    TRAIN_FILE,
    PRIMARY_METRIC,
    CV_FOLDS,
    CV_SHUFFLE,
)
from src.models.configs import get_param_grid


def create_base_estimators():
    """
    Create base estimators with best hyperparameters from previous tuning.

    Returns:
        list: List of (name, estimator) tuples
    """
    # Random Forest v3 best params (F1: 0.6419)
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # Decision Tree v3 best params (F1: 0.6092)
    dt = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=10,
        min_samples_split=40,
        min_samples_leaf=4,
        max_features=None,
        random_state=RANDOM_SEED,
    )

    return [
        ("rf", rf),
        ("dt", dt),
    ]


def main():
    print("\n" + "=" * 70)
    print("VOTING CLASSIFIER TUNING - VERSION 3")
    print("=" * 70)
    print("\nCombining best models: RF (0.6419) + DT (0.6092)")

    # Load and preprocess data
    print("\n📊 Loading training data...")
    X_train, y_train = load_data(TRAIN_FILE)

    print("🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    # Create base estimators
    estimators = create_base_estimators()

    # Create Voting Classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
    )

    # Parameter grid
    param_grid = get_param_grid("voting", "v3")

    print(
        f"\n🔍 Parameter grid: {len(param_grid['voting']) * len(param_grid['weights'])} combinations"
    )

    # Cross-validation setup
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    # Grid search
    print("\n⏳ Running GridSearchCV (this may take a while)...")
    grid_search = GridSearchCV(
        estimator=voting_clf,
        param_grid=param_grid,
        cv=cv,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    grid_search.fit(X_train_processed, y_train)

    # Results
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n🏆 Best Parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")

    print(f"\n📊 Best F1 Score: {best_score:.4f}")
    print(f"   Baseline (RF v3): 0.6419")
    improvement = ((best_score - 0.6419) / 0.6419) * 100
    print(f"   Improvement: {improvement:+.2f}%")

    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    output_file = "results/voting_tuning_v3_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to: {output_file}")

    # Summary results
    summary = {
        "model_name": "voting",
        "version": "v3",
        "best_params": best_params,
        "best_f1_score": best_score,
        "baseline_f1": 0.6419,
        "improvement_pct": improvement,
    }

    summary_file = "results/voting_tuning_v3_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"✅ Summary saved to: {summary_file}")

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
