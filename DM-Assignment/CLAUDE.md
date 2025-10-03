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
│   ├── 01_data_exploration.ipynb      # Initial data exploration
│   ├── 02_missing_analysis.ipynb      # Missing value analysis
│   └── 03_missing_value_experiments.ipynb  # Missing value imputation experiments
├── src/                               # Source code modules
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Configuration parameters (RANDOM_SEED=42)
│   └── preprocessing.py               # Preprocessing pipeline (Global Median/Mode + OrdinalEncoder)
├── test/                              # Test scripts
│   └── test_preprocessing.py          # Preprocessing pipeline test
├── results/                           # Model results and predictions
├── docs/                              # Documentation
│   ├── Project_Requirements.md
│   └── Task_Checklist.md
├── reports/                           # Analysis reports
│   ├── cardinality.md
│   ├── data_exploration_report.md
│   └── missing_value_experiments_results.csv  # Experimental results
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
4. ✅ **Baseline Models Training**: All 4 models trained with 5-fold CV
   - Decision Tree: F1=0.5952 (BEST)
   - Random Forest: F1=0.5742 ± 0.0226
   - Naïve Bayes: F1=0.3997 ± 0.0233
   - k-NN: F1=0.2003 ± 0.0075 (needs feature scaling)

### Baseline Model Results Summary

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Decision Tree | ~0.856 | 0.5952 | Best baseline, ready for tuning |
| Random Forest | 0.8331 ± 0.0069 | 0.5742 ± 0.0226 | Good stability, high tuning potential |
| Naïve Bayes | 0.7915 ± 0.0038 | 0.3997 ± 0.0233 | Moderate performance |
| k-NN | 0.7159 ± 0.0068 | 0.2003 ± 0.0075 | Poor, needs StandardScaler |

**Key Findings**:
- Tree-based models (DT, RF) significantly outperform k-NN and NB
- k-NN performance is poor due to lack of feature scaling
- Focus tuning efforts on Random Forest and Decision Tree
- Current best F1=0.5952, need to reach ≥0.65 for full marks

### Next Steps (From Task_Checklist.md)

1. **Performance Comparison Table**: Create unified comparison script
2. **Hyperparameter Tuning**: GridSearchCV on Random Forest and Decision Tree
3. **k-NN with Scaling**: Optional - test k-NN with StandardScaler
4. **Ensemble Methods**: Optionally try Voting Classifier
5. **Final Model**: Train on full dataset with fixed seed (42)
6. **Code Conversion**: Convert all .ipynb to .py files

## Development Commands

**Current environment**: Python 3.10 with conda environment `dm`

```bash
# Test preprocessing pipeline
python test/test_preprocessing.py

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

**Completed**:

- ✅ Data exploration and analysis
- ✅ Missing value pattern identification
- ✅ Feature distribution analysis
- ✅ Project structure setup
- ✅ Preprocessing pipeline implementation

**To Do**:

- Baseline model training
- Hyperparameter tuning
- Final model selection
- Code conversion (.ipynb → .py)
- Result file generation
- Documentation completion
