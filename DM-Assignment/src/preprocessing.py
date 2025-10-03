"""
Data preprocessing module for binary classification project.

This module implements the preprocessing pipeline based on experimental results:
- Missing value imputation: Global Median for numerical, Mode for categorical
- Categorical encoding: OrdinalEncoder
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

from .config import (
    NUM_IMPUTATION_STRATEGY,
    CAT_IMPUTATION_STRATEGY,
    UNKNOWN_VALUE,
    TARGET_COL,
)


class DataPreprocessor:
    """
    Data preprocessing pipeline for binary classification.

    This class handles:
    1. Missing value imputation (median for numerical, mode for categorical)
    2. Categorical encoding (OrdinalEncoder)

    Usage:
        preprocessor = DataPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    """

    def __init__(self):
        """
        Initialize the preprocessor with imputers and encoders.
        """
        # Missing value imputers
        self.num_imputer = SimpleImputer(strategy=NUM_IMPUTATION_STRATEGY)
        self.cat_imputer = SimpleImputer(strategy=CAT_IMPUTATION_STRATEGY)

        # Categorical encoder
        self.cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=UNKNOWN_VALUE
        )

        # Feature column names (will be set during fit)
        self.num_cols = None
        self.cat_cols = None
        self.feature_cols = None

        # Flag to track if preprocessor has been fitted
        self._is_fitted = False

    def _identify_feature_types(self, X):
        """
        Identify numerical and categorical feature columns.

        Args:
            X (pd.DataFrame): Input features

        Returns:
            tuple: (numerical_columns, categorical_columns)
        """
        num_cols = [col for col in X.columns if col.startswith("Num_")]
        cat_cols = [col for col in X.columns if col.startswith("Nom_")]

        return num_cols, cat_cols

    def fit(self, X, y=None):
        """
        Fit the preprocessor on training data.

        This method:
        1. Identifies feature types (numerical vs categorical)
        2. Fits imputers on training data
        3. Fits encoder on training data

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series, optional): Target variable (not used, for sklearn compatibility)

        Returns:
            self: Fitted preprocessor
        """
        # Identify feature types
        self.num_cols, self.cat_cols = self._identify_feature_types(X)
        self.feature_cols = self.num_cols + self.cat_cols

        # Fit numerical imputer
        if self.num_cols:
            self.num_imputer.fit(X[self.num_cols])

        # Fit categorical imputer
        if self.cat_cols:
            self.cat_imputer.fit(X[self.cat_cols])

        # Fit categorical encoder on imputed data
        if self.cat_cols:
            # First impute missing values, then fit encoder
            # Convert to DataFrame to preserve feature names
            X_cat_imputed = self.cat_imputer.transform(X[self.cat_cols])
            X_cat_imputed_df = pd.DataFrame(X_cat_imputed, columns=self.cat_cols)
            self.cat_encoder.fit(X_cat_imputed_df)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform features using fitted preprocessor.

        This method:
        1. Imputes missing values
        2. Encodes categorical features

        Args:
            X (pd.DataFrame): Features to transform

        Returns:
            pd.DataFrame: Transformed features
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )

        # Create a copy to avoid modifying original data
        X_processed = X.copy()

        # Impute numerical features
        if self.num_cols:
            X_processed[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        # Impute categorical features
        if self.cat_cols:
            X_processed[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

        # Encode categorical features
        if self.cat_cols:
            X_processed[self.cat_cols] = self.cat_encoder.transform(
                X_processed[self.cat_cols]
            )

        return X_processed

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series, optional): Target variable

        Returns:
            pd.DataFrame: Transformed features
        """
        return self.fit(X, y).transform(X)

    def get_feature_info(self):
        """
        Get information about features.

        Returns:
            dict: Dictionary containing feature information
        """
        if not self._is_fitted:
            return {"error": "Preprocessor not fitted yet"}

        return {
            "total_features": len(self.feature_cols),
            "numerical_features": len(self.num_cols),
            "categorical_features": len(self.cat_cols),
            "numerical_columns": self.num_cols,
            "categorical_columns": self.cat_cols,
        }


def load_data(train_path, test_path=None):
    """
    Load training and optionally test data.

    Args:
        train_path (str): Path to training CSV file
        test_path (str, optional): Path to test CSV file

    Returns:
        tuple: (X_train, y_train, X_test) or (X_train, y_train) if test_path is None
    """
    # Load training data
    train_df = pd.read_csv(train_path)

    # Separate features and target
    X_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]

    # Load test data if provided
    if test_path:
        X_test = pd.read_csv(test_path)
        return X_train, y_train, X_test

    return X_train, y_train


def verify_preprocessing(X, description="Preprocessed Data"):
    """
    Verify that preprocessing was successful.

    Args:
        X (pd.DataFrame): Processed features
        description (str): Description for logging

    Returns:
        dict: Verification results
    """
    results = {
        "description": description,
        "shape": X.shape,
        "missing_values": X.isnull().sum().sum(),
        "dtypes": X.dtypes.value_counts().to_dict(),
    }

    # Print verification
    print(f"\n{'='*60}")
    print(f"{description.upper()} VERIFICATION")
    print(f"{'='*60}")
    print(f"Shape: {results['shape']}")
    print(f"Missing values: {results['missing_values']}")
    print(f"Data types: {results['dtypes']}")
    print(f"{'='*60}\n")

    return results
