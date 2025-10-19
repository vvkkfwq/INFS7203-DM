"""
Data preprocessing module for binary classification project.

This module implements the preprocessing pipeline based on experimental results:
- Missing value imputation: Global Mean for numerical, Mode for categorical
- Outlier Detection: LOF
- Categorical encoding: OrdinalEncoding
"""

import pandas as pd
from .imputation import GlobalMeanModeImputer
from sklearn.preprocessing import OrdinalEncoder
from .outlier_detection import LOFDetector
from ..utils.config import TARGET_COL


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the preprocessor with imputers and encoders.
        """
        # Missing value imputers
        self.imputer = None
        self.X_processed = None
        self.y_processed = None

        # Categorical encoder
        self.cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
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

        # Outlier detector
        self.outlier_detector = LOFDetector(
            self.num_cols, n_neighbors=20, contamination=0.01
        )

        if self.outlier_detector is not None:
            self.outlier_detector.fit(X_imputed)
            self.y_processed = y

        if self.cat_cols:

            self.cat_encoder.fit(X_imputed[self.cat_cols])

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
        print(f"\n   Doing Imputation: Global Mean / Mode")
        X_imputed = self.imputer.transform(X)

        print(f"\n   Doing outlier detection: LOF_k20_c0.01")
        n_before = len(X_imputed)
        outlier_counts = []
        if self.outlier_detector is not None:
            inlier_mask = self.outlier_detector.get_inlier_mask()
            X_imputed = X_imputed[inlier_mask].reset_index(drop=True)
            self.y_processed = self.y_processed[inlier_mask].reset_index(drop=True)

            n_removed = n_before - len(X_imputed)
            outlier_counts.append(n_removed)
        else:
            outlier_counts.append(0)
        print(
            f"\n   Removed={outlier_counts[-1]} ({outlier_counts[-1]/n_before*100:.1f}%)"
        )

        print(f"\n   Doing feature encoding: OrdinalEncoder")
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

        return self.X_processed, self.y_processed

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
