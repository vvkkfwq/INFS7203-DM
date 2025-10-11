"""
Base trainer abstract class for model training.

This module provides the abstract base class that defines the
common training workflow for all models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from src.preprocessing.data_preprocessor import DataPreprocessor, load_data
from src.utils.config import TRAIN_FILE
from src.utils.metrics import display_training_progress


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class defines the common workflow for training models:
    1. Load data
    2. Preprocess data
    3. Initialize model
    4. Train/evaluate model
    5. Display results

    Subclasses must implement:
    - _initialize_model()
    - _train_model()
    - _get_model_name()
    """

    def __init__(
        self,
        data_file: Optional[str] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        verbose: bool = True,
    ):
        """
        Initialize the trainer.
        """
        self.data_file = data_file or TRAIN_FILE
        self.preprocessor = preprocessor
        self.verbose = verbose

        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_train_processed = None

        # Model containers
        self.model = None
        self.results = None

    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        """
        if self.verbose:
            self._print_header()

        # Step 1: Load data
        self._load_data()

        # Step 2: Preprocess data
        self._preprocess_data()

        # Step 3: Initialize model
        self._initialize_model()

        # Step 4: Train/evaluate model
        self._train_model()

        # Step 5: Display results
        if self.verbose:
            self._display_results()

        # Step 6: Prepare return results
        self._prepare_results()

        return self.results

    def _load_data(self) -> None:
        """Load training data from file."""
        if self.verbose:
            display_training_progress(1, 4, "Loading training data")

        self.X_train, self.y_train = load_data(self.data_file)

        if self.verbose:
            print(
                f"  ✓ Loaded {self.X_train.shape[0]} samples with {self.X_train.shape[1]} features"
            )
            print(f"  ✓ Class distribution: {dict(self.y_train.value_counts())}")

    def _preprocess_data(self) -> None:
        """Preprocess the training data."""
        if self.verbose:
            display_training_progress(2, 4, "Preprocessing data")

        # Create new preprocessor if not provided
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
            self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        else:
            # Use provided fitted preprocessor
            self.X_train_processed = self.preprocessor.transform(self.X_train)

        if self.verbose:
            print(f"  ✓ Preprocessing complete")
            missing_values = self.X_train_processed.isnull().sum().sum()
            print(f"  ✓ Missing values after preprocessing: {missing_values}")

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the model.

        Subclasses must implement this method to create their specific model.
        """
        pass

    @abstractmethod
    def _train_model(self) -> None:
        """
        Train/evaluate the model.

        Subclasses must implement this method to define their training logic.
        """
        pass

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        Get the model name for display.

        Returns:
            str: Model name
        """
        pass

    def _print_header(self) -> None:
        """Print training header."""
        print("\n" + "=" * 70)
        print(f"{self._get_model_name().upper()} TRAINING")
        print("=" * 70)

    def _display_results(self) -> None:
        """
        Display training results.

        Override this method in subclasses to customize result display.
        """
        if self.results and "cv_scores" in self.results:
            print("\n" + "=" * 70)
            print("TRAINING RESULTS")
            print("=" * 70)
            cv_scores = self.results["cv_scores"]
            print(
                f"Accuracy: {cv_scores.get('accuracy_mean', 0):.4f} ± "
                f"{cv_scores.get('accuracy_std', 0):.4f}"
            )
            print(
                f"F1 Score: {cv_scores.get('f1_mean', 0):.4f} ± "
                f"{cv_scores.get('f1_std', 0):.4f}"
            )
            print("=" * 70)

    def _prepare_results(self) -> None:
        """
        Prepare results dictionary.

        Override this method in subclasses to add additional results.
        """
        if self.results is None:
            self.results = {}

        # Add common results
        self.results.update(
            {
                "model_name": self._get_model_name(),
                "model": self.model,
                "preprocessor": self.preprocessor,
            }
        )

    def get_model(self):
        """
        Get the trained model.

        Returns:
            Trained sklearn model instance
        """
        return self.model

    def get_preprocessor(self):
        """
        Get the fitted preprocessor.

        Returns:
            Fitted DataPreprocessor instance
        """
        return self.preprocessor

    def save_model(self, output_file: str) -> None:
        """
        Save the trained model and preprocessor.

        Args:
            output_file (str): Output file path
        """
        from src.utils.io import save_model

        if self.model is None or self.preprocessor is None:
            raise ValueError("No trained model to save. Call train() first.")

        save_model(self.model, self.preprocessor, output_file)
