"""
Model configuration and factory module.

This module provides:
- Model factory for creating sklearn models
- Model metadata (display names, descriptions)
- Default model configurations
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Dict, Any, Optional

from src.utils.config import RANDOM_SEED


# Model registry - maps model names to classes
MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
}


# Model metadata
MODEL_METADATA = {
    "decision_tree": {
        "display_name": "Decision Tree",
        "description": "Single decision tree classifier",
        "needs_scaling": False,
        "supports_parallelization": False,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": "Ensemble of decision trees",
        "needs_scaling": False,
        "supports_parallelization": True,
    },
    "knn": {
        "display_name": "k-Nearest Neighbors",
        "description": "Instance-based learning algorithm",
        "needs_scaling": True,  # k-NN requires feature scaling
        "supports_parallelization": True,
    },
    "naive_bayes": {
        "display_name": "Naïve Bayes",
        "description": "Gaussian Naïve Bayes classifier",
        "needs_scaling": False,
        "supports_parallelization": False,
    },
}


# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "decision_tree": {
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "random_state": RANDOM_SEED,
        "n_jobs": -1,  # Use all CPU cores
    },
    "knn": {
        # k-NN has no random_state parameter
    },
    "naive_bayes": {
        # Naive Bayes has no random_state parameter
    },
}


class ModelFactory:
    """
    Factory class for creating sklearn models with consistent configurations.
    """

    @staticmethod
    def create_model(
        model_name: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a model instance with specified parameters.

        Args:
            model_name (str): Name of the model (e.g., 'decision_tree', 'random_forest')
            params (dict, optional): Model parameters to override defaults

        Returns:
            sklearn model instance

        Raises:
            ValueError: If model_name is not recognized
        """
        model_name = model_name.lower()

        if model_name not in MODEL_REGISTRY:
            available_models = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available_models}"
            )

        # Get model class
        model_class = MODEL_REGISTRY[model_name]

        # Get default parameters
        default_params = DEFAULT_MODEL_PARAMS.get(model_name, {}).copy()

        # Override with provided parameters
        if params:
            default_params.update(params)

        # Create and return model instance
        return model_class(**default_params)

    @staticmethod
    def get_available_models() -> list:
        """
        Get list of available model names.

        Returns:
            list: List of model names
        """
        return list(MODEL_REGISTRY.keys())

    @staticmethod
    def needs_scaling(model_name: str) -> bool:
        """
        Check if a model requires feature scaling.

        Args:
            model_name (str): Name of the model

        Returns:
            bool: True if model needs scaling
        """
        return MODEL_METADATA.get(model_name, {}).get("needs_scaling", False)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get full configuration for a model.

    Args:
        model_name (str): Name of the model

    Returns:
        dict: Dictionary containing:
            - model_class: sklearn model class
            - default_params: Default parameters
            - metadata: Model metadata
    """
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    return {
        "model_class": MODEL_REGISTRY[model_name],
        "default_params": DEFAULT_MODEL_PARAMS.get(model_name, {}),
        "metadata": MODEL_METADATA.get(model_name, {}),
    }


def display_available_models() -> None:
    """
    Display all available models with their metadata.
    """
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)

    for model_name in MODEL_REGISTRY.keys():
        metadata = MODEL_METADATA.get(model_name, {})
        print(f"\n{metadata.get('display_name', model_name)}")
        print(f"  Key: {model_name}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        print(f"  Needs Scaling: {metadata.get('needs_scaling', False)}")
        print(f"  Parallelization: {metadata.get('supports_parallelization', False)}")

    print("\n" + "=" * 70)
