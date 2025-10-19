"""
Hyperparameter tuning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import VotingTuner
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.utils.io import save_tuning_results

from src.utils.config import RANDOM_SEED


def create_base_estimators():
    """
    Create base estimators with best hyperparameters from previous tuning.

    Returns:
        list: List of (name, estimator) tuples
    """
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=10,
        min_samples_split=40,
        min_samples_leaf=4,
        max_features=None,
        # class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    return [
        ("rf", rf),
        ("dt", dt),
    ]


def tune_model(estimators_list, param_grid_version, outlier):

    print("\n\n" + "=" * 70)
    print("MODEL TUNING")
    print("=" * 70)

    # Tune model
    tuner = VotingTuner(
        estimators=estimators_list,
        param_grid_version=param_grid_version,
        imputationModel=imputationModel,
        outlier=outlier,
    )

    results = tuner.train()

    # Save results
    save_tuning_results(
        results,
        output_file=f"results/{tuner.model_name}_tuning_{param_grid_version}.csv",
    )

    return results


def main():

    estimators_list = create_base_estimators()

    tuning_results = tune_model(estimators_list, "v3", False)

    # Summary
    print("\n\n" + "=" * 70)
    print("TUNING COMPLETE - SUMMARY")
    print("=" * 70)

    print(
        f"  Best Accuracy: {tuning_results['best_accuracy']:.4f} ± {tuning_results['accuracy_std']:.4f}"
    )
    print(
        f"  Best F1: {tuning_results['best_f1_score']:.4f} ± {tuning_results['f1_std']:.4f}"
    )
    print(f"  Best params: {tuning_results['best_params']}")


if __name__ == "__main__":
    main()
