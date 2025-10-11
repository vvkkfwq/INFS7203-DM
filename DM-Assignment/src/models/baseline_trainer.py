"""
Baseline model trainer for quick model evaluation.

This module provides a trainer for evaluating baseline models
with default parameters using cross-validation.
"""

from typing import Optional, Dict, Any

from .base_trainer import BaseTrainer
from .configs import ModelFactory, get_model_config
from src.utils.metrics import (
    evaluate_model_cv,
    display_cv_results,
    display_training_progress,
)
from src.utils.config import CV_FOLDS


class BaselineTrainer(BaseTrainer):
    """
    Trainer for baseline models with default parameters.

    This trainer:
    1. Creates a model with default parameters
    2. Evaluates using cross-validation
    3. Reports performance metrics
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
        data_file: Optional[str] = None,
        preprocessor=None,
        verbose: bool = True,
    ):
        """
        Initialize baseline trainer.
        """
        super().__init__(
            data_file=data_file, preprocessor=preprocessor, verbose=verbose
        )

        self.model_name = model_name.lower()
        self.model_params = model_params or {}

        # Get model configuration
        self.model_config = get_model_config(self.model_name)

    def _get_model_name(self) -> str:
        """Get display name for the model."""
        return self.model_config["metadata"]["display_name"]

    def _initialize_model(self) -> None:
        """Initialize the model with default parameters."""
        if self.verbose:
            display_training_progress(
                3, 4, "Initializing model with default parameters"
            )

        # Create model using factory
        self.model = ModelFactory.create_model(self.model_name, self.model_params)

        if self.verbose:
            print(f"  ✓ Model: {self._get_model_name()}")
            print(f"  ✓ Parameters: {self.model.get_params()}")

    def _train_model(self) -> None:
        """Train and evaluate model using cross-validation."""
        if self.verbose:
            display_training_progress(
                4,
                4,
                f"Performing {CV_FOLDS}-fold cross-validation",
                "(This may take a few minutes...)",
            )

        # Evaluate using cross-validation
        cv_scores = evaluate_model_cv(
            model=self.model,
            X=self.X_train_processed,
            y=self.y_train,
        )

        # Store results
        self.results = {"cv_scores": cv_scores}

    def _display_results(self) -> None:
        """Display cross-validation results."""
        if self.results and "cv_scores" in self.results:
            display_cv_results(
                results=self.results["cv_scores"],
                model_name=self._get_model_name(),
                show_header=True,
                show_fold_details=False,
            )

    def _prepare_results(self) -> None:
        """Prepare results dictionary."""
        super()._prepare_results()

        # Add parameters to results
        if self.model is not None:
            self.results["parameters"] = self.model.get_params()


def train_baseline_model(
    model_name: str,
    model_params: Optional[Dict[str, Any]] = None,
    data_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to train a baseline model.

    Args:
        model_name (str): Name of the model
        model_params (dict, optional): Model parameters
        data_file (str, optional): Path to training data
        verbose (bool): Whether to print progress

    Returns:
        dict: Training results
    """
    trainer = BaselineTrainer(
        model_name=model_name,
        model_params=model_params,
        data_file=data_file,
        verbose=verbose,
    )

    return trainer.train()
