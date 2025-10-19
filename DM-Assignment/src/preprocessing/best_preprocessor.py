"""
Data preprocessing module for binary classification project.

This module implements the preprocessing pipeline based on experimental results:
- Missing value imputation: Global Mean for numerical, Mode for categorical
- Outlier Detection: NoOutlier
- Categorical encoding: TargetEncoder
"""

import pandas as pd
from .imputation import GlobalMeanModeImputer
from sklearn.preprocessing import TargetEncoder
from ..utils.config import TARGET_COL, RANDOM_SEED


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the preprocessor with imputers and encoders.
        """
        # Missing value imputers
        self.imputer = None
        self.X_processed = None

        # Categorical encoder
        self.cat_encoder = TargetEncoder(random_state=RANDOM_SEED)

        # Feature column names (will be set during fit)
        self.num_cols = None
        self.cat_cols = None
        self.feature_cols = None

        # Flag to track if preprocessor has been fitted
        self._is_fitted = False

    def _identify_feature_types(self, X):
        """
        Identify numerical and categorical feature columns.
        """
        num_cols = [col for col in X.columns if col.startswith("Num_")]
        cat_cols = [col for col in X.columns if col.startswith("Nom_")]

        return num_cols, cat_cols

    def fit(self, X, y=None):
        """
        Fit the preprocessor on training data.
        """
        # Identify feature types
        self.num_cols, self.cat_cols = self._identify_feature_types(X)
        self.feature_cols = self.num_cols + self.cat_cols

        self.imputer = GlobalMeanModeImputer(self.num_cols, self.cat_cols)
        self.imputer.fit(X)

        X_imputed = self.imputer.transform(X)

        if self.cat_cols:

            # TargetEncoder requires y for fitting
            if y is None:
                raise ValueError(
                    "TargetEncoder requires target variable 'y' for fitting. "
                    "Please provide y when calling fit()."
                )

            self.cat_encoder.fit(X_imputed[self.cat_cols], y)

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform features using fitted preprocessor.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )

        X_imputed = self.imputer.transform(X)
        if self.cat_cols:
            X_cat_encoded = self.cat_encoder.transform(X_imputed[self.cat_cols])
            X_cat_encoded = pd.DataFrame(
                X_cat_encoded, columns=self.cat_cols, index=X_imputed.index
            )
            self.X_processed = pd.concat(
                [X_imputed[self.num_cols], X_cat_encoded], axis=1
            )
        else:
            self.X_processed = X_imputed

        return self.X_processed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def verify_preprocessing(X, description="Preprocessed Data"):
        """
        Verify that preprocessing was successful.
        """
        results = {
            "description": description,
            "shape": X.shape,
            "missing_values": X.isnull().sum().sum(),
            "dtypes": X.dtypes.value_counts().to_dict(),
        }

        # Print verification
        print(f"\n{'='*70}")
        print(f"{description.upper()} VERIFICATION")
        print(f"{'='*70}")
        print(f"Shape: {results['shape']}")
        print(f"Missing values: {results['missing_values']}")
        print(f"Data types: {results['dtypes']}")
        print(f"{'='*70}\n")


def load_data(train_path, test_path=None):
    """
    Load training and optionally test data.
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
