"""
Hyperparameter tuning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import HyperparameterTuner
from src.utils.io import save_tuning_results


def tune_model(model_name, param_grid_version):

    print("\n\n" + "=" * 70)
    print("MODEL TUNING")
    print("=" * 70)

    # Tune model
    tuner = HyperparameterTuner(
        model_name=model_name,
        param_grid_version=param_grid_version,
    )

    results = tuner.train()

    # Save results
    save_tuning_results(
        results, output_file=f"results/{model_name}_tuning_{param_grid_version}.csv"
    )

    return results


def main():
    """Main function to run all tuning examples."""

    tuning_results = tune_model("naive_bayes", "v0")

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
