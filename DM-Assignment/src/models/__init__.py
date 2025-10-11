"""
Models module for binary classification project.

This module provides a modular training framework including:
- Base trainer abstract class
- Baseline model trainer
- Hyperparameter tuning
- Model configurations
"""

from .base_trainer import BaseTrainer
from .baseline_trainer import BaselineTrainer
from .hyperparameter_tuner import HyperparameterTuner

__all__ = ["BaseTrainer", "BaselineTrainer", "HyperparameterTuner"]
