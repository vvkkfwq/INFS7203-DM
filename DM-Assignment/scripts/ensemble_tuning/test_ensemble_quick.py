"""
Quick test script for ensemble models.

This script quickly verifies that ensemble models are properly configured
and can run without errors (uses minimal parameter grid for speed).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.configs.model_configs import display_available_models, ModelFactory
from src.models.configs.param_grids import get_available_versions, get_param_grid


def test_model_registry():
    """Test that all ensemble models are registered."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Registry")
    print("=" * 70)

    ensemble_models = ['voting', 'adaboost', 'bagging']

    for model_name in ensemble_models:
        try:
            # This will raise error if model not registered
            model = ModelFactory.create_model(model_name)
            print(f"✅ {model_name}: Registered successfully")
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")


def test_param_grids():
    """Test that parameter grids are available."""
    print("\n" + "=" * 70)
    print("TEST 2: Parameter Grids")
    print("=" * 70)

    test_cases = [
        ('adaboost', 'v1'),
        ('bagging', 'v1'),
        ('voting', 'v1'),
    ]

    for model_name, version in test_cases:
        try:
            param_grid = get_param_grid(model_name, version)
            num_params = len(param_grid)
            print(f"✅ {model_name} ({version}): {num_params} parameters defined")

            # Show parameter names
            params = list(param_grid.keys())
            print(f"   Parameters: {', '.join(params)}")
        except Exception as e:
            print(f"❌ {model_name} ({version}): Error - {e}")


def test_available_versions():
    """Test getting available versions for each model."""
    print("\n" + "=" * 70)
    print("TEST 3: Available Versions")
    print("=" * 70)

    ensemble_models = ['voting', 'adaboost', 'bagging']

    for model_name in ensemble_models:
        versions = get_available_versions(model_name)
        if versions:
            print(f"✅ {model_name}: {', '.join(versions)}")
        else:
            print(f"❌ {model_name}: No versions found")


def main():
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL CONFIGURATION TEST")
    print("=" * 70)
    print("\nThis script verifies that ensemble models are properly configured.")

    # Run tests
    test_model_registry()
    test_param_grids()
    test_available_versions()

    # Display all available models
    print("\n" + "=" * 70)
    print("ALL AVAILABLE MODELS")
    print("=" * 70)
    display_available_models()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n✅ If all tests passed, you're ready to run ensemble tuning!")
    print("\nNext steps:")
    print("  1. python scripts/ensemble_tuning/tune_adaboost_v1.py")
    print("  2. python scripts/ensemble_tuning/tune_bagging_v1.py")
    print("  3. python scripts/ensemble_tuning/tune_voting_v1.py")


if __name__ == "__main__":
    main()
