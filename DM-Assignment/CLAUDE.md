# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

INFS7203 Data Mining Track 1 - Binary Classification Project

- **Task**: Binary classification (0/1) using Week 2-8 techniques, maximize F1 Score
- **Deadline**: 2025-10-20 13:00 Brisbane Time
- **Student ID**: s4860387
- **Dataset**: 10,853 train samples, 2,713 test samples, 43 features (25 numerical + 18 categorical)
- **Performance Goal**: F1 Score ≥ 0.65 for full marks

## Technical Constraints

**ALLOWED**: Outlier detection, Normalization, Imputation, Categorical encoding, Decision Tree, Random Forest, k-NN, Naïve Bayes
**PROHIBITED**: XGBoost, LightGBM, CatBoost, Neural Networks, Deep Learning

## Directory Structure

```
DM-Assignment/
├── data/                    # Training and test datasets
├── src/                     # Production code (modularized framework)
│   ├── models/              # Training framework (BaselineTrainer, HyperparameterTuner)
│   │   └── configs/         # Model configs and parameter grids
│   ├── preprocessing/       # DataPreprocessor and transformers
│   └── utils/               # Metrics, I/O, config
├── examples/                # Usage examples (quick_start.py, tune_model.py)
├── scripts/                 # Experimental scripts (baseline/, *_tuning/)
├── notebooks/               # EDA and experiments (*.ipynb)
├── results/                 # Model outputs (*.csv)
├── logs/                    # Training logs
└── docs/                    # Requirements and task checklist
```

**Key Modules**:

- `src/models/{baseline_trainer,hyperparameter_tuner}.py`: Unified training interface
- `src/models/configs/{model_configs,param_grids}.py`: Model registry and parameter grids
- `src/preprocessing/data_preprocessor.py`: Data preprocessing pipeline
- `src/utils/{metrics,io,config}.py`: Shared utilities

## File Placement Guidelines

**Follow these rules for organizing new code:**

| Code Type                                      | Location                            | Examples                                             |
| ---------------------------------------------- | ----------------------------------- | ---------------------------------------------------- |
| **Production modules** (reusable)              | `src/`                              | BaselineTrainer, DataPreprocessor, utility functions |
| **Exploratory tuning experiments** (versioned) | `scripts/model_name_tuning/`        | tune_rf_v1.py, tune_dt_v2.py                         |
| **Baseline comparisons**                       | `scripts/baseline/`                 | train_baseline.py, compare_baseline_models.py        |
| **Interactive exploration** (temporary)        | `notebooks/`                        | EDA, quick prototyping (\*.ipynb)                    |
| **Parameter grids** (centralized)              | `src/models/configs/param_grids.py` | All model parameter grids                            |
| **Usage demonstrations**                       | `examples/`                         | quick_start.py (show how to use src modules)         |

**Key Principles**:

- ✅ **Experiments** → `scripts/model_name_tuning/` (keep versioned history: v1, v2, v3...)
- ✅ **Reusable code** → `src/` (modularized, tested, production-ready)
- ✅ **Parameter configs** → `src/models/configs/param_grids.py` (centralized management)
- ✅ **Quick tests** → `notebooks/` (interactive exploration, then migrate to scripts/)
- ❌ **Don't mix**: Keep experiments (scripts/) separate from production code (src/)

## Quick Start (Modularized Framework)

```python
# Train baseline model (2 lines)
from src.models import BaselineTrainer
results = BaselineTrainer(model_name="random_forest").train()

# Hyperparameter tuning (2 lines)
from src.models import HyperparameterTuner
results = HyperparameterTuner(model_name="random_forest", param_grid_version="v3").train()

# See examples/README.md for detailed usage
```

## Model Performance (Best Results)

| Rank | Model               | F1 Score (Mean ± Std) | Status            | Key Params                                       |
| ---- | ------------------- | --------------------- | ----------------- | ------------------------------------------------ |
| 🥇   | **Voting v3**       | **0.6505** ± 0.0164   | **🎯 BEST MODEL** | soft, weights=[2,1] (RF+DT without class_weight) |
| 🥈   | Random Forest v3    | 0.6419 ± 0.0119       | **TARGET MET**    | n_estimators=200, max_depth=20                   |
| 🥉   | Random Forest v4    | 0.6415 ± 0.0113       | **TARGET MET**    | n_estimators=300, max_depth=25                   |
| 4    | Voting v2           | 0.6401 ± 0.0073       | Target Met        | soft, weights=[2,1] (RF+DT)                      |
| 5    | Voting v1           | 0.6237 ± 0.0202       | Below Target      | soft, weights=[3,2,1] (RF+DT+NB)                 |
| 6    | AdaBoost v1         | 0.6147 ± 0.0127       | Below Target      | n_estimators=200, learning_rate=1.0              |
| 7    | Decision Tree v3    | 0.6092 ± 0.0143       | Close             | max_depth=10, min_samples_split=5                |
| 8    | k-NN v1 (no scaler) | 0.3027 ± 0.0104       | Poor              | n_neighbors=3, weights=uniform, p=1              |

**Baseline Comparison** (Baseline → Best Tuned):

- **Voting Ensemble**: 0.6419 → **0.6505** (+1.3% improvement) ✨ **NEW BEST**
- Random Forest: 0.5742 → 0.6419 (+11.8% improvement)
- Decision Tree: 0.5344 → 0.6092 (+14.0% improvement)
- AdaBoost: N/A → 0.6147 (new model tested)
- k-NN: 0.2003 → 0.3027 (still poor without StandardScaler)
- Naïve Bayes: 0.3997 (baseline only)

## Development Commands

```bash
# Environment: Python 3.10, conda env `dm`

# Quick examples
python examples/quick_start.py        # Demonstrate framework usage
python examples/tune_model.py         # Tuning examples

# Legacy scripts (completed experiments)
python scripts/baseline/train_baseline.py
python scripts/random_forest_tuning/tune_rf_v3.py
python scripts/decision_tree_tuning/tune_dt_v3.py

# Main pipeline (to be implemented)
python main.py  # Generate s4860387.infs4203
```

## Key Implementation Details

**Preprocessing**:

- Imputation: SimpleImputer (median for numerical, mode for categorical)
- Encoding: OrdinalEncoder for categorical features
- Scaling: StandardScaler for k-NN only

**Cross-Validation**:

- StratifiedKFold (5 folds) to maintain class balance
- RANDOM_SEED = 42 for all operations

**Missing Values**:

- 1,085 samples (10%) have missing values across ALL 43 features

**Class Imbalance**:

- 75% class 0, 25% class 1 (handled via class_weight='balanced' in best models)

## Submission Requirements

**Result File** (`s4860387.infs4203`):

- 2,714 lines: first 2,713 lines = test predictions (0, or 1,), last line = CV scores (accuracy,f1,)

**Code Package** (`s4860387.zip`):

- main.py entry point, all .py files (NO .ipynb), data files, README.md (Chinese), requirements.txt

## Current Status

**Completed**:

- ✅ EDA and preprocessing experiments
- ✅ Modularized training framework (BaselineTrainer, HyperparameterTuner)
- ✅ Baseline training (all 4 basic models: RF, DT, k-NN, NB)
- ✅ Hyperparameter tuning (RF v1-v4, DT v1-v3, k-NN v1)
- ✅ Ensemble model tuning (Voting v1-v3, AdaBoost v1)
- ✅ **Target F1 ≥ 0.65 achieved and exceeded** (Voting v3: **0.6505**)
- ✅ **New best model identified**: Voting Classifier (soft voting, RF+DT with weights [2,1])

**Tuning Summary**:

- Random Forest: 4 versions tested → v3 best (0.6419)
- Decision Tree: 3 versions tested → v3 best (0.6092)
- Voting Ensemble: 3 versions tested → v3 best (0.6505) 🎯
- AdaBoost: 1 version tested (0.6147, below target)
- k-NN: 1 version tested (0.3027, needs StandardScaler improvement)

**Remaining** (see `docs/Task_Checklist.md` for details):

- ⏳ Implement main.py using best model (Voting v3)
- ⏳ Generate submission files (s4860387.infs4203, s4860387.zip)
- ⏳ Finalize README.md (Chinese) and requirements.txt
- 🔄 Optional: Try k-NN v2 with StandardScaler, explore Bagging/Stacking ensembles
