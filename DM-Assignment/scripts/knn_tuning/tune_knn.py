"""
k-NN hyperparameter tuning script.

This script performs GridSearchCV to find optimal hyperparameters for k-NN
to maximize F1 Score on the binary classification task.

IMPORTANT: k-NN requires feature scaling (StandardScaler) for optimal performance.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import (
    RANDOM_SEED,
    TRAIN_FILE,
    CV_FOLDS,
    CV_SHUFFLE,
    PRIMARY_METRIC,
)
from src.preprocessing import DataPreprocessor, load_data


def tune_knn():
    """
    Perform hyperparameter tuning for k-NN using GridSearchCV.

    NOTE: k-NN requires feature scaling. This function applies StandardScaler
    after categorical encoding for optimal distance-based computation.

    Returns:
        dict: Tuning results including best model, best parameters, and scores
    """
    print("\n" + "=" * 70)
    print("k-NN HYPERPARAMETER TUNING")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/6] Loading training data...")
    X_train, y_train = load_data(TRAIN_FILE)
    print(f"  ✓ Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"  ✓ Class distribution: {dict(y_train.value_counts())}")

    # Step 2: Preprocess data (imputation + encoding)
    print("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"  ✓ Preprocessing complete")
    print(
        f"  ✓ Missing values after preprocessing: {X_train_processed.isnull().sum().sum()}"
    )

    # Step 3: Apply StandardScaler (CRITICAL for k-NN)
    print("\n[3/6] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)

    # Print scaling statistics
    print(f"  ✓ StandardScaler fitted")
    print(f"  ✓ Scaled data shape: {X_train_scaled.shape}")
    print(f"  ✓ Mean after scaling: {X_train_scaled.mean():.6f} (should be ~0)")
    print(f"  ✓ Std after scaling: {X_train_scaled.std():.6f} (should be ~1)")

    # Step 4: Define parameter grid
    print("\n[4/6] Defining parameter grid...")
    # Define parameter search space for k-NN
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    print(f"  ✓ Parameter grid defined:")
    for param, values in param_grid.items():
        print(f"      - {param}: {values}")
    print(f"  ✓ Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

    # Step 5: Initialize GridSearchCV
    print("\n[5/6] Setting up GridSearchCV...")

    # Create StratifiedKFold for cross-validation
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    # Initialize base model
    base_model = KNeighborsClassifier()

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=PRIMARY_METRIC,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    print(f"  ✓ GridSearchCV initialized")
    print(f"  ✓ Scoring metric: {PRIMARY_METRIC}")
    print(f"  ✓ Cross-validation: {CV_FOLDS}-fold StratifiedKFold")
    print(f"  ✓ Parallel jobs: -1 (all CPUs)")

    # Step 6: Run grid search
    print("\n[6/6] Running grid search...")
    print("  (This may take 2-5 minutes depending on your hardware...)")
    print("  Progress will be shown below:\n")

    grid_search.fit(X_train_scaled, y_train)

    # Extract results
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    print(f"\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")

    print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

    # Evaluate best model on all metrics
    best_model = grid_search.best_estimator_
    best_idx = grid_search.best_index_
    f1_mean = grid_search.cv_results_["mean_test_score"][best_idx]
    f1_std = grid_search.cv_results_["std_test_score"][best_idx]

    print(f"\nBest Model Performance:")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")

    # Compare with baseline
    baseline_f1 = 0.2003  # From baseline results (WITHOUT StandardScaler)

    print(f"\nImprovement over Baseline:")
    print(
        f"  F1 Score: {baseline_f1:.4f} → {f1_mean:.4f} ({f1_mean - baseline_f1:+.4f})"
    )

    # Show top 5 parameter combinations
    print(f"\nTop 5 Parameter Combinations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")

    for idx, row in results_df.head(5).iterrows():
        print(f"\n  Rank {int(row['rank_test_score'])}:")
        print(
            f"    F1 Score: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}"
        )
        print(f"    Parameters: {row['params']}")

    print("\n" + "=" * 70)

    # Prepare return results
    results = {
        "model_name": "k-NN_Tuned",
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_f1_score": f1_mean,
        "f1_std": f1_std,
        "baseline_f1": baseline_f1,
        "f1_improvement": f1_mean - baseline_f1,
        "preprocessor": preprocessor,
        "scaler": scaler,  # IMPORTANT: Save scaler for test set
        "grid_search": grid_search,
        "cv_results": results_df,
    }

    return results


def save_results(results, output_file="results/knn_tuning_results.csv"):
    """
    Save tuning results to CSV file.

    Args:
        results (dict): Tuning results dictionary
        output_file (str): Output file path
    """
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save summary results
    summary = {
        "model": results["model_name"],
        "f1_mean": results["best_f1_score"],
        "f1_std": results["f1_std"],
        "baseline_f1": results["baseline_f1"],
        "f1_improvement": results["f1_improvement"],
        **{f"param_{k}": v for k, v in results["best_params"].items()},
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Save detailed CV results
    detailed_file = output_file.replace(".csv", "_detailed.csv")
    results["cv_results"].to_csv(detailed_file, index=False)
    print(f"✓ Detailed CV results saved to: {detailed_file}")


def main():
    """Main execution function."""
    try:
        results = tune_knn()

        # Save results
        save_results(results)

        print("\n✓ k-NN hyperparameter tuning complete!")
        print(f"\nNext steps:")
        print(f"  1. Review the best parameters above")
        print(
            f"  2. Compare with Decision Tree (F1=0.6039) and Random Forest (F1=0.5995)"
        )
        print(f"  3. Select the best model for final submission")

    except Exception as e:
        print(f"\n✗ Error during tuning: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
