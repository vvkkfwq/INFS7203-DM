"""
Configuration file for the binary classification project.
Contains all constants and hyperparameters.
"""

import random
import numpy as np

# ==================== REPRODUCIBILITY ====================
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==================== FILE PATHS ====================
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.csv"
TEST_FILE = f"{DATA_DIR}/test_data.csv"
RESULTS_DIR = "results"


# ==================== STUDENT INFO ====================
STUDENT_ID = "s4860387"
SUBMISSION_EXTENSION = "infs4203"
RESULT_FILE = f"{RESULTS_DIR}/{STUDENT_ID}.{SUBMISSION_EXTENSION}"


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
