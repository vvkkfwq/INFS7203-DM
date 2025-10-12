"""
Bagging Tuning - Version 1

Bagging (Bootstrap Aggregating) trains multiple models on random subsets
of the training data and averages their predictions.

Strategy:
- Use Decision Tree as base estimator (default)
- Tune number of estimators
- Tune max_samples (fraction of samples to use)
- Tune max_features (fraction of features to use)
- Test bootstrap strategies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import HyperparameterTuner


def main():
    print("\n" + "=" * 70)
    print("BAGGING TUNING - VERSION 1")
    print("=" * 70)
    print("\nBootstrap Aggregating with Decision Tree base estimator")

    # Use the modularized framework
    tuner = HyperparameterTuner(
        model_name="bagging",
        param_grid_version="v1",
        baseline_score=0.6419,  # Best current model (RF v3)
    )

    results = tuner.train()

    # Display summary
    print("\n" + "=" * 70)
    print("TUNING COMPLETE - SUMMARY")
    print("=" * 70)
    print(
        f"\n🏆 Best F1 Score: {results['best_f1_score']:.4f} ± {results['f1_std']:.4f}"
    )
    print(f"   Baseline (RF v3): {results['baseline_score']:.4f}")
    print(f"   Improvement: {results['improvement_pct']:+.2f}%")
    print(f"\n📊 Best Parameters:")
    for param, value in results["best_params"].items():
        print(f"   {param}: {value}")

    print(f"\n✅ Results saved to:")
    print(f"   - results/bagging_tuning_v1_results.csv")
    print(f"   - results/bagging_tuning_v1_results_detailed.csv")

    print("\n💡 Note: Bagging is similar to Random Forest but more flexible.")
    print("   Consider using Random Forest if Bagging doesn't improve results.")


if __name__ == "__main__":
    main()
