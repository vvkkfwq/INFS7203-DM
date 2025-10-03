"""
Test script for preprocessing pipeline.

This script verifies that the preprocessing pipeline works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path to access src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import DataPreprocessor, load_data, verify_preprocessing
from src.config import TRAIN_FILE, TEST_FILE, RANDOM_SEED

import numpy as np

np.random.seed(RANDOM_SEED)


def test_preprocessing_pipeline():
    """
    Test the complete preprocessing pipeline.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    X_train, y_train, X_test = load_data(TRAIN_FILE, TEST_FILE)

    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Target distribution:\n{y_train.value_counts()}")

    # Check missing values before preprocessing
    print(f"\n[Before Preprocessing]")
    print(f"  Training missing values: {X_train.isnull().sum().sum()}")
    print(f"  Test missing values: {X_test.isnull().sum().sum()}")

    # Step 2: Initialize and fit preprocessor
    print("\n[Step 2] Fitting preprocessor on training data...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train, y_train)

    # Display feature info
    feature_info = preprocessor.get_feature_info()
    print(f"✓ Total features: {feature_info['total_features']}")
    print(f"✓ Numerical features: {feature_info['numerical_features']}")
    print(f"✓ Categorical features: {feature_info['categorical_features']}")

    # Step 3: Transform training data
    print("\n[Step 3] Transforming training data...")
    X_train_processed = preprocessor.transform(X_train)
    verify_preprocessing(X_train_processed, "Training Data")

    # Verify no missing values
    assert (
        X_train_processed.isnull().sum().sum() == 0
    ), "❌ Training data still has missing values!"
    print("✓ Training data: No missing values")

    # Step 4: Transform test data
    print("\n[Step 4] Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    verify_preprocessing(X_test_processed, "Test Data")

    # Verify no missing values
    assert (
        X_test_processed.isnull().sum().sum() == 0
    ), "❌ Test data still has missing values!"
    print("✓ Test data: No missing values")

    # Step 5: Verify shapes
    print("\n[Step 5] Verifying shapes...")
    assert X_train_processed.shape == X_train.shape, "❌ Training shape mismatch!"
    assert X_test_processed.shape == X_test.shape, "❌ Test shape mismatch!"
    print(f"✓ Training shape: {X_train_processed.shape}")
    print(f"✓ Test shape: {X_test_processed.shape}")

    # Step 6: Check data types
    print("\n[Step 6] Checking data types...")
    print(f"Training data types:\n{X_train_processed.dtypes.value_counts()}")
    print(f"\nTest data types:\n{X_test_processed.dtypes.value_counts()}")

    # Step 7: Sample transformed data
    print("\n[Step 7] Sample of transformed data:")
    print("\nFirst 3 rows of training data:")
    print(X_train_processed.head(3))

    print("\nFirst 3 rows of test data:")
    print(X_test_processed.head(3))

    # Final summary
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nPreprocessing pipeline is ready for model training.")
    print(
        f"\nNext step: Train baseline models on {len(X_train_processed)} training samples"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_preprocessing_pipeline()
