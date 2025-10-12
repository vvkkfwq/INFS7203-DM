"""
AdaBoost Tuning - Version 1

AdaBoost (Adaptive Boosting) sequentially trains weak learners,
each focusing on samples misclassified by previous learners.

Strategy:
- Use Decision Tree as base estimator (default)
- Tune number of estimators
- Tune learning rate
- Compare SAMME vs SAMME.R algorithms
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import HyperparameterTuner


def main():
    print("\n" + "=" * 70)
    print("ADABOOST TUNING")
    print("=" * 70)
    print("\nAdaptive Boosting with Decision Tree base estimator")

    # Use the modularized framework
    tuner = HyperparameterTuner(
        model_name="adaboost",
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
    print(f"   - results/adaboost_tuning_v1_results.csv")
    print(f"   - results/adaboost_tuning_v1_results_detailed.csv")


if __name__ == "__main__":
    main()
