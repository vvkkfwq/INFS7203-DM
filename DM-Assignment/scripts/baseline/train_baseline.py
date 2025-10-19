"""
Training baseline models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import BaselineTrainer
from src.utils.io import save_baseline_results


def main():

    print("\n" + "=" * 70)
    print("BASELINE MODELS TRAINING")
    print("=" * 70)

    # List of models to train
    models = ["decision_tree", "random_forest", "knn", "naive_bayes"]

    all_results = []

    for model_name in models:
        print(f"\n\nTraining {model_name}...")

        # Create trainer
        trainer = BaselineTrainer(model_name=model_name)
        results = trainer.train()

        all_results.append(results)

    # Summary
    print("\n\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    for result in all_results:
        cv_scores = result["cv_scores"]
        print(f"\n{result['model_name']}:")
        print(f"  F1 Score: {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
        print(
            f"  Accuracy: {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f}"
        )

    # Save results
    save_baseline_results(all_results)


if __name__ == "__main__":
    main()
