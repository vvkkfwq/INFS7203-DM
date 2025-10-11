"""
Model configurations module.

This module contains:
- Model factory for creating sklearn models
- Parameter grids for hyperparameter tuning
- Model metadata and configurations
"""

from .model_configs import ModelFactory, get_model_config
from .param_grids import get_param_grid, needs_class_weight

__all__ = ["ModelFactory", "get_model_config", "get_param_grid", "needs_class_weight"]
