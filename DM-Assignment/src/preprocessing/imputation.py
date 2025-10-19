"""
Imputation module
"""

from .base import BasePreprocessor
from sklearn.impute import SimpleImputer
import pandas as pd


class BaseImputer(BasePreprocessor):

    def __init__(self, num_cols, cat_cols):

        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X):

        self._validate_features(X)
        self.num_imputer.fit(X[self.num_cols])
        self.cat_imputer.fit(X[self.cat_cols])

        return self

    def transform(self, X):

        self._validate_features(X)
        X_filled = X.copy()
        X_filled[self.num_cols] = self.num_imputer.transform(X[self.num_cols])
        X_filled[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

        assert (
            X_filled[self.num_cols].isnull().sum().sum() == 0
        ), "Numerical features still have missing values after imputation!"
        assert (
            X_filled[self.cat_cols].isnull().sum().sum() == 0
        ), "Categorical features still have missing values after imputation!"

        return X_filled

    def _validate_features(self, X):

        missing_cols = set(self.num_cols + self.cat_cols) - set(X.columns)

        if missing_cols:
            raise ValueError(
                f"Missing columns in data: {sorted(missing_cols)}\n"
                f"Expected columns: {sorted(self.num_cols + self.cat_cols)}\n"
                f"Actual columns: {sorted(X.columns.tolist())}"
            )


class GlobalMedianModeImputer(BaseImputer):

    def __init__(self, num_cols, cat_cols):

        super().__init__(num_cols, cat_cols)

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")


class GlobalMeanModeImputer(BaseImputer):

    def __init__(self, num_cols, cat_cols):
        super().__init__(num_cols, cat_cols)

        self.num_imputer = SimpleImputer(strategy="mean")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")


class ClassSpecificImputer(BaseImputer):

    def __init__(self, num_cols, cat_cols):

        super().__init__(num_cols, cat_cols)

        # imputer for class 0
        self.class_0_num_imputer = SimpleImputer(strategy="mean")
        self.class_0_cat_imputer = SimpleImputer(strategy="most_frequent")

        # imputer for class 1
        self.class_1_num_imputer = SimpleImputer(strategy="mean")
        self.class_1_cat_imputer = SimpleImputer(strategy="most_frequent")

        # imputer for validation set
        self.num_imputer = SimpleImputer(strategy="mean")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

    def fit_with_label(self, X, y):

        if y is None:
            raise ValueError("ClassSpecificImputer requires y parameter for fitting")

        self._validate_features(X)

        # fit for class 0
        mask_0 = y == 0
        self.class_0_num_imputer.fit(X.loc[mask_0, self.num_cols])
        self.class_0_cat_imputer.fit(X.loc[mask_0, self.cat_cols])

        # fit for class 1
        mask_1 = y == 1
        self.class_1_num_imputer.fit(X.loc[mask_1, self.num_cols])
        self.class_1_cat_imputer.fit(X.loc[mask_1, self.cat_cols])

        return self

    def transform_with_label(self, X, y):

        self._validate_features(X)
        X_filled = X.copy()

        # imputation for class 0
        mask_0 = y == 0
        if mask_0.sum() > 0:
            X_filled.loc[mask_0, self.num_cols] = self.class_0_num_imputer.transform(
                X.loc[mask_0, self.num_cols]
            )
            X_filled.loc[mask_0, self.cat_cols] = self.class_0_cat_imputer.transform(
                X.loc[mask_0, self.cat_cols]
            )

        # imputation for class 1
        mask_1 = y == 1
        if mask_1.sum() > 0:
            X_filled.loc[mask_1, self.num_cols] = self.class_1_num_imputer.transform(
                X.loc[mask_1, self.num_cols]
            )
            X_filled.loc[mask_1, self.cat_cols] = self.class_1_cat_imputer.transform(
                X.loc[mask_1, self.cat_cols]
            )

        assert (
            X_filled[self.num_cols].isnull().sum().sum() == 0
        ), "Numerical features still have missing values after imputation!"
        assert (
            X_filled[self.cat_cols].isnull().sum().sum() == 0
        ), "Categorical features still have missing values after imputation!"

        return X_filled


def get_imputer(strategy="median_mode", num_cols=None, cat_cols=None):

    # Policy Mapping Table
    strategies = {
        "median_mode": GlobalMedianModeImputer,
        "mean_mode": GlobalMeanModeImputer,
        "class_specific": ClassSpecificImputer,
    }

    # Verification Policy Name
    if strategy not in strategies:
        raise ValueError(
            f"Unknown imputation strategy: '{strategy}'. "
            f"Available strategies: {list(strategies.keys())}"
        )

    # Verification parameters
    if num_cols is None or cat_cols is None:
        raise ValueError(
            "Both num_cols and cat_cols must be provided. " "Cannot be None."
        )

    # Creates and returns an interpolator instance.
    imputer_class = strategies[strategy]
    return imputer_class(num_cols=num_cols, cat_cols=cat_cols)
