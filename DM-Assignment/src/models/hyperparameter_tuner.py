"""
Hyperparameter tuning module for model optimization.

This module provides a trainer for performing GridSearchCV-based
hyperparameter tuning with predefined parameter grids.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.model_selection import GridSearchCV

from .base_trainer import BaseTrainer
from .configs import ModelFactory, get_model_config, get_param_grid, needs_class_weight
from src.utils.metrics import (
    setup_cross_validation,
    compare_with_baseline,
    display_training_progress,
)
from src.utils.config import PRIMARY_METRIC, CV_FOLDS


class HyperparameterTuner(BaseTrainer):
    """
    Trainer for hyperparameter tuning using GridSearchCV.

    This trainer:
    1. Loads predefined parameter grids
    2. Performs exhaustive grid search with cross-validation
    3. Reports best parameters and performance
    4. Compares with baseline performance
    """

    def __init__(
        self,
        model_name: str,
        param_grid_version: str = "v1",
        baseline_score: Optional[float] = None,
        custom_param_grid: Optional[Dict[str, Any]] = None,
        base_model_params: Optional[Dict[str, Any]] = None,
        data_file: Optional[str] = None,
        preprocessor=None,
        verbose: bool = True,
        n_jobs: int = -1,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            model_name (str): Name of the model (e.g., 'random_forest', 'decision_tree')
            param_grid_version (str): Version of parameter grid to use (e.g., 'v1', 'v2')
            baseline_score (float, optional): Baseline F1 score for comparison
            custom_param_grid (dict, optional): Custom parameter grid (overrides version)
            base_model_params (dict, optional): Base parameters for model (e.g., class_weight)
            data_file (str, optional): Path to training data
            preprocessor (DataPreprocessor, optional): Fitted preprocessor
            verbose (bool): Whether to print progress
            n_jobs (int): Number of parallel jobs for GridSearchCV
        """
        super().__init__(data_file=data_file, preprocessor=preprocessor, verbose=verbose)

        self.model_name = model_name.lower()
        self.param_grid_version = param_grid_version
        self.baseline_score = baseline_score
        self.custom_param_grid = custom_param_grid
        self.base_model_params = base_model_params or {}
        self.n_jobs = n_jobs

        # Get model configuration
        self.model_config = get_model_config(self.model_name)

        # Grid search results
        self.grid_search = None
        self.best_params = None
        self.best_score = None

    def _get_model_name(self) -> str:
        """Get display name for the model."""
        return f"{self.model_config['metadata']['display_name']} (Tuned {self.param_grid_version})"

    def _get_param_grid(self) -> Dict[str, Any]:
        """Get parameter grid for tuning."""
        if self.custom_param_grid:
            return self.custom_param_grid
        else:
            return get_param_grid(self.model_name, self.param_grid_version)

    def _initialize_model(self) -> None:
        """Initialize base model and parameter grid."""
        if self.verbose:
            display_training_progress(3, 5, "Defining parameter grid")

        # Get parameter grid
        param_grid = self._get_param_grid()

        if self.verbose:
            print(f"  ✓ Parameter grid defined:")
            for param, values in param_grid.items():
                print(f"      - {param}: {values}")
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"  ✓ Total combinations: {total_combinations}")

        # Check if this version needs class_weight
        if needs_class_weight(self.model_name, self.param_grid_version):
            self.base_model_params["class_weight"] = "balanced"

        # Create base model
        self.model = ModelFactory.create_model(self.model_name, self.base_model_params)

        # Store parameter grid
        self.param_grid = param_grid

    def _train_model(self) -> None:
        """Perform grid search for hyperparameter tuning."""
        if self.verbose:
            display_training_progress(4, 5, "Setting up GridSearchCV")

        # Setup cross-validation
        cv = setup_cross_validation(n_splits=CV_FOLDS)

        # Create GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=PRIMARY_METRIC,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            return_train_score=False,
        )

        if self.verbose:
            print(f"  ✓ GridSearchCV initialized")
            print(f"  ✓ Scoring metric: {PRIMARY_METRIC}")
            print(f"  ✓ Cross-validation: {CV_FOLDS}-fold StratifiedKFold")
            print(f"  ✓ Parallel jobs: {self.n_jobs}")

        # Run grid search
        if self.verbose:
            display_training_progress(
                5,
                5,
                "Running grid search",
                "(This may take 10-30 minutes depending on your hardware...)",
            )
            print()

        self.grid_search.fit(self.X_train_processed, self.y_train)

        # Extract results
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        self.model = self.grid_search.best_estimator_

        # Get best model statistics
        best_idx = self.grid_search.best_index_
        f1_mean = self.grid_search.cv_results_["mean_test_score"][best_idx]
        f1_std = self.grid_search.cv_results_["std_test_score"][best_idx]

        # Store results
        self.results = {
            "cv_scores": {
                "f1_mean": f1_mean,
                "f1_std": f1_std,
            },
            "best_params": self.best_params,
            "best_f1_score": f1_mean,
            "f1_std": f1_std,
        }

    def _display_results(self) -> None:
        """Display grid search results."""
        if not self.results:
            return

        print("\n" + "=" * 70)
        print("GRID SEARCH RESULTS")
        print("=" * 70)

        # Best parameters
        print(f"\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  - {param}: {value}")

        # Best score
        print(f"\nBest Cross-Validation F1 Score: {self.best_score:.4f}")
        print(
            f"  F1 Score: {self.results['best_f1_score']:.4f} ± "
            f"{self.results['f1_std']:.4f}"
        )

        # Compare with baseline if provided
        if self.baseline_score:
            compare_with_baseline(
                current_score=self.results["best_f1_score"],
                baseline_score=self.baseline_score,
                metric_name="F1 Score",
            )

        # Show top 5 parameter combinations
        self._display_top_combinations(top_n=5)

        print("\n" + "=" * 70)

    def _display_top_combinations(self, top_n: int = 5) -> None:
        """Display top N parameter combinations."""
        print(f"\nTop {top_n} Parameter Combinations:")

        results_df = pd.DataFrame(self.grid_search.cv_results_)
        results_df = results_df.sort_values("rank_test_score")

        for idx, row in results_df.head(top_n).iterrows():
            print(f"\n  Rank {int(row['rank_test_score'])}:")
            print(
                f"    F1 Score: {row['mean_test_score']:.4f} ± "
                f"{row['std_test_score']:.4f}"
            )
            print(f"    Parameters: {row['params']}")

    def _prepare_results(self) -> None:
        """Prepare results dictionary."""
        super()._prepare_results()

        # Add tuning-specific results
        if self.baseline_score:
            self.results["baseline_f1"] = self.baseline_score
            self.results["f1_improvement"] = (
                self.results["best_f1_score"] - self.baseline_score
            )

        # Add CV results DataFrame
        if self.grid_search:
            self.results["cv_results"] = pd.DataFrame(self.grid_search.cv_results_)
            self.results["grid_search"] = self.grid_search

    def get_cv_results_df(self) -> pd.DataFrame:
        """
        Get detailed cross-validation results as DataFrame.

        Returns:
            pd.DataFrame: CV results with all parameter combinations
        """
        if self.grid_search is None:
            raise ValueError("No tuning results. Call train() first.")

        return pd.DataFrame(self.grid_search.cv_results_)


def tune_model(
    model_name: str,
    param_grid_version: str = "v1",
    baseline_score: Optional[float] = None,
    custom_param_grid: Optional[Dict[str, Any]] = None,
    data_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to tune a model's hyperparameters.

    Args:
        model_name (str): Name of the model
        param_grid_version (str): Version of parameter grid
        baseline_score (float, optional): Baseline F1 score for comparison
        custom_param_grid (dict, optional): Custom parameter grid
        data_file (str, optional): Path to training data
        verbose (bool): Whether to print progress

    Returns:
        dict: Tuning results
    """
    tuner = HyperparameterTuner(
        model_name=model_name,
        param_grid_version=param_grid_version,
        baseline_score=baseline_score,
        custom_param_grid=custom_param_grid,
        data_file=data_file,
        verbose=verbose,
    )

    return tuner.train()
