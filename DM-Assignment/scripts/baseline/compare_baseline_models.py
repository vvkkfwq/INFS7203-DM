"""
Baseline models performance comparison script.

This script trains all 4 baseline models with 5-fold cross-validation
and generates a unified performance comparison table.

Models:
- Decision Tree
- Random Forest
- k-Nearest Neighbor (k-NN)
- Naïve Bayes
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.utils.config import (
    PRIMARY_METRIC,
    RANDOM_SEED,
    SECONDARY_METRIC,
    DATA_DIR,
    TARGET_COL,
    CV_FOLDS,
)
from src.preprocessing import DataPreprocessor, load_data


def train_and_evaluate_model(model, model_name, X, y, cv_splits=CV_FOLDS):
    """
    Train a model with cross-validation and return performance metrics.

    Args:
        model: sklearn model instance
        model_name (str): Name of the model for display
        X (pd.DataFrame): Features
        y (pd.Series): Target
        cv_splits (int): Number of CV folds

    Returns:
        dict: Performance metrics including mean and std
    """
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    # Perform cross-validation
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[PRIMARY_METRIC, SECONDARY_METRIC],  # Use config-based scoring names
        return_train_score=False,
        n_jobs=-1,  # Use all CPU cores
    )

    # Calculate statistics
    results = {
        "model": model_name,
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "cv_folds": cv_splits,
    }

    # Print results
    print(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    return results


def main():
    """
    Main function to compare all baseline models.
    """
    print("\n" + "=" * 60)
    print("BASELINE MODELS PERFORMANCE COMPARISON")
    print("=" * 60)

    # Load training data
    train_path = DATA_DIR / "train.csv"
    print(f"\nLoading data from: {train_path}")

    X_train, y_train = load_data(train_path)
    print(f"Training data shape: {X_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")

    # Preprocess data
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    print(f"Processed data shape: {X_train_processed.shape}")
    print(f"Missing values: {X_train_processed.isnull().sum().sum()}")

    # Baseline Model Configuration:
    # - Use default parameters to establish performance baseline without optimization
    # - random_state=RANDOM_SEED ensures reproducibility (project requirement)
    # - Focus on comparing model types, not parameter tuning (done in next phase)
    # - Tree models (DT, RF) expected to perform well on mixed feature types
    # - k-NN may underperform without feature scaling (will test with StandardScaler later)

    # Define baseline models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Naïve Bayes": GaussianNB(),
    }

    # Train and evaluate each model
    all_results = []

    for model_name, model in models.items():
        results = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            X=X_train_processed,
            y=y_train,
            cv_splits=CV_FOLDS,
        )
        all_results.append(results)

    # Create comparison table
    results_df = pd.DataFrame(all_results)

    # Sort by F1 score (descending)
    results_df = results_df.sort_values("f1_mean", ascending=False)

    # Display comparison table
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    print("\nRanked by F1 Score (Primary Metric):\n")

    # Format display
    for idx, row in results_df.iterrows():
        print(f"{row['model']}")
        print(f"  Accuracy: {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"  F1 Score: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        print()

    # Save results to CSV
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "baseline_models_comparison.csv"

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    best_model = results_df.iloc[0]
    worst_model = results_df.iloc[-1]

    print(f"\n✓ Best Model: {best_model['model']} (F1 = {best_model['f1_mean']:.4f})")
    print(f"✗ Worst Model: {worst_model['model']} (F1 = {worst_model['f1_mean']:.4f})")
    print(f"\nPerformance Gap: {(best_model['f1_mean'] - worst_model['f1_mean']):.4f}")

    # Check if we meet the target
    target_f1 = 0.65
    current_best_f1 = best_model["f1_mean"]
    gap_to_target = target_f1 - current_best_f1

    print(f"\nTarget F1 Score: {target_f1:.2f} (for full marks)")
    print(f"Current Best F1: {current_best_f1:.4f}")

    if current_best_f1 >= target_f1:
        print("✓ Target achieved!")
    else:
        print(f"✗ Gap to target: {gap_to_target:.4f}")
        print(f"  Improvement needed: {(gap_to_target/current_best_f1)*100:.1f}%")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Focus hyperparameter tuning on top 2 models")
    print("2. Consider feature scaling for k-NN if it shows potential")
    print("3. Explore ensemble methods if single models plateau")
    print()


if __name__ == "__main__":
    main()
