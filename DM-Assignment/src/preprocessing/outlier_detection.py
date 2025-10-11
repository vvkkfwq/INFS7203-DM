"""
Outlier Detection Module

Provides outlier detection methods for preprocessing pipeline.
Supports Isolation Forest and Local Outlier Factor (LOF) methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.utils.config import RANDOM_SEED


class BaseOutlierDetector:
    """
    Base class for outlier detectors.
    """

    def __init__(self, num_cols):
        """
        Initialize base outlier detector.
        """
        self.num_cols = num_cols
        self.inlier_mask = None

    def fit(self, X):
        """
        Fit the outlier detector on the training data.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def get_inlier_mask(self):
        """
        Get the inlier mask after fitting.
        """
        if self.inlier_mask is None:
            raise ValueError("Must call fit() before get_inlier_mask()")
        return self.inlier_mask

    def fit_transform(self, X):
        """
        Fit the detector and return filtered data.
        """
        self.fit(X)
        mask = self.get_inlier_mask()
        return X[mask].reset_index(drop=True)


class IsolationForestDetector(BaseOutlierDetector):
    """
    Outlier detection using Isolation Forest algorithm.
    """

    def __init__(
        self, num_cols, contamination=0.1, random_state=RANDOM_SEED, n_estimators=100
    ):
        super().__init__(num_cols)
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.detector = None

    def fit(self, X):
        """
        Fit Isolation Forest on numerical features.
        """

        # Extract numerical features only
        X_num = X[self.num_cols].values

        # Initialize and fit Isolation Forest
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1,
            verbose=1,
        )

        # fit_predict returns: 1 for inliers, -1 for outliers
        predictions = self.detector.fit_predict(X_num)

        # Convert to boolean mask: True for inliers, False for outliers
        self.inlier_mask = predictions == 1

        return self

    def __repr__(self):
        return (
            f"IsolationForestDetector(contamination={self.contamination}, "
            f"n_estimators={self.n_estimators})"
        )


class LOFDetector(BaseOutlierDetector):
    """
    Outlier detection using Local Outlier Factor (LOF) algorithm.
    """

    def __init__(self, num_cols, n_neighbors=20, contamination=0.1, metric="euclidean"):
        super().__init__(num_cols)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.detector = None
        self.scaler = None

    def fit(self, X):
        """
        Fit LOF on numerical features.
        """

        # Extract numerical features only
        X_num = X[self.num_cols].values

        # LOF requires feature scaling - use internal scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_num)

        # Initialize and fit LOF
        self.detector = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=False,  # fit_predict mode for training set
            n_jobs=-1,
        )

        # fit_predict returns: 1 for inliers, -1 for outliers
        predictions = self.detector.fit_predict(X_scaled)

        # Convert to boolean mask
        self.inlier_mask = predictions == 1

        return self

    def __repr__(self):
        return (
            f"LOFDetector(n_neighbors={self.n_neighbors}, "
            f"contamination={self.contamination})"
        )


# Convenience function to create outlier detector
def get_outlier_detector(method="isolation_forest", num_cols=None, **kwargs):
    """
    Factory function to create outlier detector.
    """
    if num_cols is None:
        raise ValueError("num_cols must be provided")

    if method == "isolation_forest":
        return IsolationForestDetector(num_cols, **kwargs)
    elif method == "lof":
        return LOFDetector(num_cols, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'isolation_forest' or 'lof'"
        )
