"""
Parameter grids for hyperparameter tuning.

This module defines parameter search spaces for different models
and tuning strategies (v1, v2, v3, etc.).
"""

from typing import Dict, Any


# Decision Tree parameter grids
DECISION_TREE_GRIDS = {
    "v1": {
        # Initial broad search
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "max_features": [None, "sqrt", "log2"],
    },
    "v2": {
        # Refined search based on v1 results
        "criterion": ["gini", "entropy"],
        "splitter": ["best"],
        "max_depth": [10, 15, 20, 25],
        "min_samples_split": [10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None],
    },
    "v3": {
        # Further refinement
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [5, 10, 15],
        "min_samples_split": [20, 30, 40],
        "min_samples_leaf": [2, 4, 6],
        "max_features": [None],
    },
}


# Random Forest parameter grids
RANDOM_FOREST_GRIDS = {
    "v1": {
        # Initial broad search
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True],
    },
    "v2": {
        # Refined search with class_weight
        "n_estimators": [100, 150, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None],
    },
    "v3": {
        # Focused search with balanced class weights
        "n_estimators": [50, 80, 100],
        "max_depth": [5, 8, 10],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3],
        "max_features": [None],
    },
    "v4": {
        # Fine-tuning around best results
        "n_estimators": [80, 100, 120],
        "max_depth": [8, 10, 12],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [1, 2],
        "max_features": [None],
    },
}


# k-NN parameter grids
KNN_GRIDS = {
    "v1": {
        # Initial search
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2],  # 1=manhattan, 2=euclidean
    },
    "v2": {
        # Refined search
        "n_neighbors": [5, 7, 9, 11, 13],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        "p": [2],
    },
}


# Naïve Bayes parameter grids (limited tuning options)
NAIVE_BAYES_GRIDS = {
    "v1": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    },
}


# Additional model-specific configurations
MODEL_SPECIFIC_CONFIGS = {
    "random_forest": {
        # Use class_weight='balanced' for versions that need it
        "class_weight_versions": ["v3", "v4"],
    },
}


def get_param_grid(model_name: str, version: str = "v1") -> Dict[str, Any]:
    """
    Get parameter grid for a specific model and version.

    Args:
        model_name (str): Name of the model ('decision_tree', 'random_forest', 'knn', 'naive_bayes')
        version (str): Version of parameter grid (default: 'v1')

    Returns:
        dict: Parameter grid for GridSearchCV

    Raises:
        ValueError: If model_name or version is not recognized
    """
    model_name = model_name.lower()
    version = version.lower()

    # Map model names to their grid dictionaries
    grid_registry = {
        "decision_tree": DECISION_TREE_GRIDS,
        "random_forest": RANDOM_FOREST_GRIDS,
        "knn": KNN_GRIDS,
        "naive_bayes": NAIVE_BAYES_GRIDS,
    }

    if model_name not in grid_registry:
        available_models = ", ".join(grid_registry.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available_models}"
        )

    grids = grid_registry[model_name]

    if version not in grids:
        available_versions = ", ".join(grids.keys())
        raise ValueError(
            f"Unknown version '{version}' for {model_name}. "
            f"Available versions: {available_versions}"
        )

    return grids[version]


def get_available_versions(model_name: str) -> list:
    """
    Get available parameter grid versions for a model.

    Args:
        model_name (str): Name of the model

    Returns:
        list: List of available version names
    """
    model_name = model_name.lower()

    grid_registry = {
        "decision_tree": DECISION_TREE_GRIDS,
        "random_forest": RANDOM_FOREST_GRIDS,
        "knn": KNN_GRIDS,
        "naive_bayes": NAIVE_BAYES_GRIDS,
    }

    if model_name not in grid_registry:
        return []

    return list(grid_registry[model_name].keys())


def needs_class_weight(model_name: str, version: str) -> bool:
    """
    Check if a specific model version should use class_weight='balanced'.

    Args:
        model_name (str): Name of the model
        version (str): Version identifier

    Returns:
        bool: True if class_weight should be set to 'balanced'
    """
    if model_name not in MODEL_SPECIFIC_CONFIGS:
        return False

    config = MODEL_SPECIFIC_CONFIGS[model_name]
    class_weight_versions = config.get("class_weight_versions", [])

    return version in class_weight_versions


def display_param_grid(model_name: str, version: str = "v1") -> None:
    """
    Display parameter grid information.

    Args:
        model_name (str): Name of the model
        version (str): Version identifier
    """
    try:
        param_grid = get_param_grid(model_name, version)

        print(f"\nParameter Grid for {model_name.upper()} ({version}):")
        print("=" * 60)

        total_combinations = 1
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
            total_combinations *= len(values)

        print(f"\nTotal combinations: {total_combinations}")
        print("=" * 60)

    except ValueError as e:
        print(f"Error: {e}")
