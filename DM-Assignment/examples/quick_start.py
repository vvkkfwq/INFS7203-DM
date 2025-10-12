"""
Quick Start Guide - Ultra-simple model training examples.

This demonstrates the SIMPLEST way to use the new framework.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# EXAMPLE 1: Train a baseline model in 2 lines
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 1: Train Random Forest Baseline (2 lines!)")
print("=" * 70)

from src.models import BaselineTrainer

trainer = BaselineTrainer(model_name="random_forest")
results = trainer.train()

print(f"\n✓ Done! F1 Score: {results['cv_scores']['f1_mean']:.4f}")


# ============================================================================
# EXAMPLE 2: Tune hyperparameters in 2 lines
# ============================================================================
print("\n\n" + "=" * 70)
print("EXAMPLE 2: Tune Random Forest (2 lines!)")
print("=" * 70)

from src.models import HyperparameterTuner

tuner = HyperparameterTuner(model_name="random_forest", param_grid_version="v3")
results = tuner.train()

print(f"\n✓ Done! Best F1 Score: {results['best_f1_score']:.4f}")
print(f"✓ Best params: {results['best_params']}")


# ============================================================================
# EXAMPLE 3: Even simpler - use convenience functions
# ============================================================================
print("\n\n" + "=" * 70)
print("EXAMPLE 3: Using Convenience Functions (1 line!)")
print("=" * 70)

from src.models.baseline_trainer import train_baseline_model
from src.models.hyperparameter_tuner import tune_model

# Train baseline
baseline_results = train_baseline_model("decision_tree")

# Tune model
tuning_results = tune_model("decision_tree", param_grid_version="v3", baseline_score=0.5344)

print("\n✓ All done!")


# ============================================================================
# EXAMPLE 4: Compare multiple models quickly
# ============================================================================
print("\n\n" + "=" * 70)
print("EXAMPLE 4: Compare Multiple Models")
print("=" * 70)

models = ["decision_tree", "random_forest", "knn", "naive_bayes"]

for model_name in models:
    results = train_baseline_model(model_name, verbose=False)
    f1 = results["cv_scores"]["f1_mean"]
    print(f"{model_name:15s}: F1 = {f1:.4f}")

print("\n✓ Comparison complete!")
