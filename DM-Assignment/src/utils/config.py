"""
Configuration file for the binary classification project.
Contains all constants and hyperparameters.
"""

import random
import numpy as np
from pathlib import Path

# ==================== REPRODUCIBILITY ====================
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==================== FILE PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


# ==================== STUDENT INFO ====================
STUDENT_ID = "s4860387"
SUBMISSION_EXTENSION = "infs4203"
RESULT_FILE = RESULTS_DIR / f"{STUDENT_ID}.{SUBMISSION_EXTENSION}"


# ==================== FEATURE COLUMNS ====================
TARGET_COL = "Target (Col44)"


# ==================== PREPROCESSING CONFIG ====================
# Missing value imputation strategy
NUM_IMPUTATION_STRATEGY = "median"  # Best from experiments: Global Median
CAT_IMPUTATION_STRATEGY = "most_frequent"  # Best from experiments: Mode

# Categorical encoding
CAT_ENCODING_METHOD = "ordinal"  # OrdinalEncoder used in experiments
UNKNOWN_VALUE = -1  # For unseen categories in test set


# ==================== MODEL TRAINING CONFIG ====================
# Cross-validation
CV_FOLDS = 5
CV_SHUFFLE = True
CV_STRATIFIED = True  # Use StratifiedKFold for class imbalance


# ==================== EVALUATION METRICS ====================
PRIMARY_METRIC = "f1"  # F1 Score (primary metric for project)
SECONDARY_METRIC = "accuracy"  # Accuracy (secondary metric)


# ==================== OUTPUT FORMAT ====================
DECIMAL_PLACES = 3  # For CV results in submission file
