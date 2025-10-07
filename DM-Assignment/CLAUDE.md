# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INFS7203 Data Mining Track 1 - Binary Classification Project

- **Task**: Train binary classifiers (0/1) using Week 2-8 techniques to maximize test set F1 Score
- **Deadline**: 2025-10-20 13:00 Brisbane Time
- **Student**: s4860387

**Dataset**:

- Training: 10,853 samples × 43 features (25 numerical + 18 categorical)
- Test: 2,713 samples × 43 features
- Target: Binary (0/1), with 1 as positive class
- Class imbalance: ~75% class 0, ~25% class 1
- Missing values: ~10% across all features

**Performance Goal**: F1 Score ≥ 0.65 for full marks (20/20)

## Technical Constraints

**ALLOWED** (Week 2-8 techniques only):

- Outlier detection
- Normalization
- Imputation
- Handling categorical features
- Ensembles
- Decision Tree
- Random Forest
- k-Nearest Neighbour (k-NN)
- Naïve Bayes

**PROHIBITED** (violations result in 0 marks):

- XGBoost, LightGBM, CatBoost
- Neural Networks, Deep Learning
- Any advanced techniques beyond Week 8

## Directory Structure

```
DM-Assignment/
├── data/
│   ├── train.csv                      # Training dataset
│   └── test_data.csv                  # Test dataset
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 00_testbook.ipynb              # Testing notebook
│   ├── 01_data_exploration.ipynb      # Initial data exploration
│   ├── 02_missing_value_detail.ipynb  # Detailed missing value analysis
│   └── 03_missing_value_experiments.ipynb  # Missing value imputation experiments
├── src/                               # Source code modules
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Configuration parameters (RANDOM_SEED=42)
│   ├── preprocessing.py               # Preprocessing pipeline (Global Median/Mode + OrdinalEncoder)
│   ├── train_decision_tree.py         # Decision Tree baseline training script
│   ├── train_random_forest.py         # Random Forest baseline training script
│   ├── train_knn.py                   # k-NN baseline training script
│   └── train_naive_bayes.py           # Naïve Bayes baseline training script
├── scripts/                           # Utility scripts
│   ├── compare_baseline_models.py     # Unified baseline comparison script
│   ├── tune_random_forest.py          # Random Forest hyperparameter tuning script
│   ├── tune_decision_tree.py          # Decision Tree hyperparameter tuning script
│   └── tune_knn.py                    # k-NN hyperparameter tuning script (with StandardScaler)
├── test/                              # Test scripts
│   └── test_preprocessing.py          # Preprocessing pipeline test
├── results/                           # Model results and predictions
│   ├── baseline_models_comparison.csv # Baseline comparison results
│   ├── random_forest_tuning_results.csv # Random Forest tuning summary
│   ├── random_forest_tuning_results_detailed.csv # Random Forest detailed CV results
│   ├── decision_tree_tuning_results.csv # Decision Tree tuning summary
│   ├── decision_tree_tuning_results_detailed.csv # Decision Tree detailed CV results
│   ├── knn_tuning_results.csv         # k-NN tuning summary
│   └── knn_tuning_results_detailed.csv # k-NN detailed CV results
├── docs/                              # Documentation
│   ├── Project_Requirements.md        # Assignment requirements
│   └── Task_Checklist.md              # Task breakdown and progress
├── reports/                           # Analysis reports
│   ├── cardinality.md                 # Feature cardinality analysis
│   ├── data_exploration_report.md     # EDA report
│   └── missing_value_experiments_results.csv  # Experimental results
├── CLAUDE.md                          # Claude Code project context (this file)
└── README.md                          # Project documentation (in Chinese)
```

## Key Data Characteristics

**Numerical Features** (Num_Col1-25):

- Wide range of scales (small decimals to large integers)
- High variability across features
- All have ~10% missing values (1085/10853)

**Categorical Features** (Nom_Col26-43):

- Encoding format: `C{category}_c{value}` (e.g., C0_c0, C1_c3)
- Variable cardinality: from 2 to 40 unique values
- All have ~10% missing values (1085/10853)

**Missing Value Pattern**:

- Exactly 1085 samples have missing values across ALL features
- Rows with at least one missing value: 10734 out of 10853 (98.90%)

## Development Workflow

### Data Exploration (Already Completed)

Analysis complete in `notebooks/data_exploration.ipynb` covering:

- Dataset structure and statistics
- Missing value patterns
- Target variable distribution
- Feature distributions and relationships

### Completed Work

1. ✅ **Data Exploration**: Comprehensive EDA in notebooks
2. ✅ **Preprocessing Experiments**: Tested 4 missing value strategies
   - Best strategy: Global Median/Mode (F1=0.5952 with DecisionTree baseline)
3. ✅ **Preprocessing Pipeline**: Implemented production-ready pipeline in `src/preprocessing.py`
   - Missing value imputation: SimpleImputer (median for numerical, mode for categorical)
   - Categorical encoding: OrdinalEncoder
   - All tests passing
4. ✅ **Baseline Models Training**: All 4 models trained with 5-fold CV using unified comparison script
   - Script location: `scripts/compare_baseline_models.py`
   - Results saved: `results/baseline_models_comparison.csv`
   - Performance ranking (by F1 Score):
     1. Random Forest: F1=0.5742 ± 0.0226 (BEST)
     2. Decision Tree: F1=0.5344 ± 0.0185
     3. Naïve Bayes: F1=0.3997 ± 0.0233
     4. k-NN: F1=0.2003 ± 0.0075 (needs feature scaling)

### Baseline Model Results Summary

| Model         | Accuracy        | F1 Score        | Notes                                |
| ------------- | --------------- | --------------- | ------------------------------------ |
| Random Forest | 0.8331 ± 0.0069 | 0.5742 ± 0.0226 | Best baseline, high tuning potential |
| Decision Tree | 0.7650 ± 0.0067 | 0.5344 ± 0.0185 | Good baseline, ready for tuning      |
| Naïve Bayes   | 0.7915 ± 0.0038 | 0.3997 ± 0.0233 | Moderate performance                 |
| k-NN          | 0.7159 ± 0.0068 | 0.2003 ± 0.0075 | Poor, needs StandardScaler           |

**Key Findings**:

- Tree-based models (RF, DT) significantly outperform k-NN and NB
- Random Forest shows best F1 Score (0.5742) with good stability
- k-NN performance is poor due to lack of feature scaling
- **Gap to target**: Need 13.2% improvement to reach F1 ≥ 0.65 for full marks
- **Priority**: Focus hyperparameter tuning on Random Forest and Decision Tree

### Model Performance Ranking (After Tuning)

| Rank | Model                       | F1 Score        | Accuracy        | Gap to Target |
| ---- | --------------------------- | --------------- | --------------- | ------------- |
| 🥇 1 | Random Forest (Tuned v3)    | 0.6419 ± 0.0119 | TBD             | +0.0081       |
| 🥈 2 | Random Forest (Tuned v4)    | 0.6415 ± 0.0113 | TBD             | +0.0085       |
| 🥉 3 | Decision Tree (Tuned v3)    | 0.6092 ± 0.0143 | 0.8227 ± 0.0066 | -0.0408       |
| 4    | Decision Tree (Tuned v2)    | 0.6068 ± 0.0154 | 0.8227 ± 0.0066 | -0.0432       |
| 5    | Random Forest (Tuned v1)    | 0.5995 ± 0.0166 | 0.8363 ± 0.0056 | -0.0505       |
| 6    | Random Forest (Baseline)    | 0.5742 ± 0.0226 | 0.8331 ± 0.0069 | -0.0758       |
| 7    | Decision Tree (Baseline)    | 0.5344 ± 0.0185 | 0.7650 ± 0.0067 | -0.1156       |
| 8    | k-NN (Tuned with scaling)   | 0.4320 ± 0.0115 | 0.7391 ± 0.0060 | -0.2180       |
| 9    | Naïve Bayes (Baseline)      | 0.3997 ± 0.0233 | 0.7915 ± 0.0038 | -0.2503       |
| 10   | k-NN (Baseline, no scaling) | 0.2003 ± 0.0075 | 0.7159 ± 0.0068 | -0.4497       |

## Development Commands

**Current environment**: Python 3.10 with conda environment `dm`

```bash
# Test preprocessing pipeline
python test/test_preprocessing.py

# Compare baseline models (completed)
python scripts/compare_baseline_models.py

# Hyperparameter tuning
python scripts/tune_random_forest.py      # Completed ✅
python scripts/tune_decision_tree.py      # Completed ✅
python scripts/tune_knn.py                # Completed ✅

# Run main pipeline (to be implemented)
python main.py

# Expected output: s4860387.infs4203
```

**Dependencies** (to be added to requirements.txt):

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Reproducibility Requirements

**Critical**: All random operations must use `RANDOM_SEED = 42`

```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# In models
model = RandomForestClassifier(random_state=RANDOM_SEED)

# In cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
```

## Submission Requirements

**Result File** (`s4860387.infs4203`):

```
Format: 2,714 lines total
- Lines 1-2713: Test predictions (0, or 1,)
- Line 2714: CV results (accuracy,f1,)

Example:
0,
1,
0,
...
0.856,0.743,
```

**Code Package** (`s4860387.zip`):

- All code as .py files (NO .ipynb)
- main.py as entry point
- train.csv and test_data.csv included
- README.md with environment, steps, configuration, and rationale
- requirements.txt with exact package versions

## Important Notes

- Code must be in .py format for submission (not .ipynb)
- README.md is in Chinese (project requirement)
- All 1085 rows with missing values have missingness across ALL 43 features
- Class imbalance (3:1 ratio) may require addressing
- F1 Score is the primary metric (accuracy is secondary)
- Cross-validation must be stratified to maintain class distribution

## Current Status

**Completed (Phase 1-3)**:

- ✅ Data exploration and analysis
- ✅ Missing value pattern identification and experiments
- ✅ Feature distribution analysis
- ✅ Project structure setup
- ✅ Preprocessing pipeline implementation and testing
- ✅ Baseline models training (all 4 models)
- ✅ Performance comparison table generation

**Completed (Phase 4)**:

- ✅ Random Forest hyperparameter tuning (F1: 0.5742→0.5995, +4.4%)
- ✅ Decision Tree hyperparameter tuning (F1: 0.5344→0.6039, +13.0%) 🏆 BEST MODEL
- ✅ k-NN hyperparameter tuning (F1: 0.2003→0.4320, +115.7%, with StandardScaler)

**To Do (Phase 5-6)**:

- ⏳ Final model selection and training
- ⏳ Create main.py entry point
- ⏳ Generate test predictions
- ⏳ Create submission result file (s4860387.infs4203)
- ⏳ Package code for submission (s4860387.zip)
- ⏳ Write README.md documentation
- ⏳ Create requirements.txt

**Note**: Please refer to `docs/Task_Checklist.md` for detailed task breakdown and `docs/Project_Requirements.md` for submission requirements.
