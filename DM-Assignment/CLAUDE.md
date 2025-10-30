# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INFS7203 Data Mining Binary Classification Project - Academic coursework with strict technical constraints.

- **Task**: Binary classification (0/1), maximize F1 Score
- **Dataset**: 10,853 train samples, 2,713 test samples, 43 features (25 numerical + 18 categorical)
- **Target**: F1 Score ≥ 0.65 for full marks (**ACHIEVED**: 0.6505 with Voting Classifier)
- **Technical Constraints**: ONLY allowed Week 2-8 techniques (Decision Tree, Random Forest, k-NN, Naïve Bayes). NO XGBoost, Neural Networks, LightGBM, CatBoost.

## Architecture Pattern

This codebase follows a **Template Method + Factory** pattern:

```
BaseTrainer (abstract)          # Template method for train() workflow
├── BaselineTrainer            # Default parameters + CV evaluation
├── HyperparameterTuner        # GridSearchCV + parameter optimization
└── VotingTuner                # Ensemble voting with custom param grids

ModelFactory                   # Creates sklearn models with configs
├── model_configs.py          # Model registry + metadata + baseline scores
└── param_grids.py            # Versioned parameter grids (v0, v1, v2...)

DataPreprocessor              # Stateful preprocessing pipeline
├── fit()                     # Learn imputation/encoding/outliers from train
├── transform()               # Apply to train (with outlier removal)
└── test_transform()          # Apply to test (NO outlier removal)
```

**Key Design Principles**:
1. **Separation of Concerns**: Production code (`src/`) vs experimental scripts (`scripts/`)
2. **Versioned Experiments**: Parameter grids use v0, v1, v2... for iterative tuning
3. **Reproducibility**: RANDOM_SEED=42 everywhere, StratifiedKFold for class balance
4. **Preprocessing State Management**: Preprocessor fits once, transforms train/test differently

## Development Commands

```bash
# Environment: Python 3.10, conda env `dm`
# Install dependencies
pip install -r requirements.txt

# Main pipeline (generate submission file)
python src/main.py  # Outputs: s4860387.infs4203

# Quick API examples
python examples/quick_start.py    # BaselineTrainer + HyperparameterTuner usage
python examples/tune_model.py     # Custom parameter grid example

# Experimental scripts (historical tuning runs)
python scripts/baseline/train_baseline.py
python scripts/ensemble_tuning/tune_voting_v3.py  # Best model (F1=0.6505)
python scripts/random_forest_tuning/tune_rf_v3.py
python scripts/decision_tree_tuning/tune_dt_v3.py
```

## File Organization Rules

| Code Type | Location | When to Use |
|-----------|----------|-------------|
| Reusable modules | `src/` | Production code, shared utilities, trainers |
| Versioned experiments | `scripts/{model}_tuning/` | Hyperparameter tuning iterations (v1, v2...) |
| Parameter grids | `src/models/configs/param_grids.py` | ALL model parameter grids (centralized) |
| Exploratory analysis | `notebooks/` | EDA, quick prototyping (migrate to scripts/ later) |

**Critical Rule**: DO NOT add parameters to experimental scripts. ALL parameter grids MUST go in `src/models/configs/param_grids.py` for version control and reusability.

## Key Implementation Details

**Preprocessing Pipeline** (`DataPreprocessor`):
- Imputation: GlobalMeanMode (numerical mean, categorical mode)
- Outlier Detection: LOF (LocalOutlierFactor, k=20, contamination=0.01) - removes ~1% train samples
- Encoding: OrdinalEncoder (handles unknown categories with -1)
- Scaling: NONE (tree-based models don't need it; k-NN requires separate handling)

**Critical Preprocessing Difference**:
```python
# Training: Fit + Transform (with outlier removal)
preprocessor.fit_transform(X_train, y_train)  # Removes ~109 outliers

# Test: Transform only (NO outlier removal, NO y)
preprocessor.test_transform(X_test)  # Keep all test samples
```

**Cross-Validation Strategy**:
- StratifiedKFold (5 folds, shuffle=True, random_state=42)
- Handles 75:25 class imbalance (Class 0: 75%, Class 1: 25%)
- Scoring: PRIMARY_METRIC='f1', SECONDARY_METRIC='accuracy'

**Model Factory Usage**:
```python
from src.models.configs import ModelFactory, get_model_config, get_param_grid

# Create model
model = ModelFactory.create_model("random_forest", params={...})

# Get baseline score for comparison
baseline = get_model_baseline_score("random_forest")  # {'f1_mean': '0.5664', ...}

# Get parameter grid for tuning
param_grid = get_param_grid("random_forest", version="v3")
```

## Best Model Configuration

**Current Best: Voting Classifier v3 (F1=0.6505 ± 0.0164)**
```python
# Components
RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=8,
    min_samples_leaf=2, class_weight='balanced', random_state=42
)
DecisionTreeClassifier(
    criterion='gini', splitter='random', max_depth=10,
    min_samples_split=30, min_samples_leaf=2,
    class_weight='balanced', random_state=42
)

# Ensemble
VotingClassifier(
    estimators=[('rf', rf), ('dt', dt)],
    voting='soft',    # Uses predict_proba
    weights=[2, 1]    # Favor RF over DT
)
```

**Model Performance Ranking**:
1. 🥇 Voting v3: 0.6505 (soft, [2,1], no class_weight on voting)
2. 🥈 RF v3: 0.6419 (n_estimators=200, max_depth=20)
3. 🥉 RF v4: 0.6415 (n_estimators=300, max_depth=25)
4. Voting v2: 0.6401 (soft, [2,1])
5. DT v3: 0.6092 (max_depth=10)
6. k-NN v1: 0.3027 (poor, needs StandardScaler)

## Common Workflows

### 1. Train a Baseline Model
```python
from src.models import BaselineTrainer

trainer = BaselineTrainer(model_name="random_forest")
results = trainer.train()  # Returns CV scores + trained model
```

### 2. Hyperparameter Tuning
```python
from src.models import HyperparameterTuner

tuner = HyperparameterTuner(
    model_name="random_forest",
    param_grid_version="v3"  # Uses predefined grid from param_grids.py
)
results = tuner.train()  # GridSearchCV + best params
```

### 3. Custom Parameter Grid
```python
tuner = HyperparameterTuner(
    model_name="random_forest",
    custom_param_grid={
        "n_estimators": [100, 200],
        "max_depth": [10, 20]
    }
)
```

### 4. Voting Ensemble Tuning
```python
from src.models import VotingTuner
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ("rf", RandomForestClassifier(**rf_params)),
    ("dt", DecisionTreeClassifier(**dt_params))
]

tuner = VotingTuner(
    estimators=estimators,
    param_grid_version="v3"
)
results = tuner.train()
```

## Submission Format

**Result File** (`s4860387.infs4203`):
```
0,         # Line 1-2713: predictions (0, or 1,)
1,
...
1,
0.795, 0.650,  # Line 2714: accuracy,f1, (3 decimals)
```

**Code Package** (`s4860387.zip`):
- Must contain: main.py, all .py files (NO .ipynb), data/, README.md, requirements.txt
- Must be runnable: `python src/main.py` should reproduce results

## Data Characteristics

**Missing Values**: 1,085 samples (10%) have missing values across ALL 43 features (not just a few).

**Class Imbalance**: 75:25 (Class 0:Class 1) - use `class_weight='balanced'` in tree models for best results.

**Feature Types**:
- 25 numerical features (prefix: `Num_`)
- 18 categorical features (prefix: `Nom_`)

**Outliers**: LOF detector identifies ~109 outliers (1%) removed during training.

## Critical Gotchas

1. **Preprocessor State**: Always call `fit_transform()` on train, then `test_transform()` (NOT `transform()`) on test.
2. **Random Seeds**: Set `RANDOM_SEED=42` in model, CV, and numpy for reproducibility.
3. **Parameter Grids**: Add to `param_grids.py`, NOT in individual scripts.
4. **Class Weight**: Best models use `class_weight='balanced'` due to 75:25 imbalance.
5. **Voting Weights**: Soft voting with `weights=[2,1]` (favor RF) beats equal weights.

## Future Exploration Ideas

**Potential Improvements** (if time allows):
- k-NN with StandardScaler (current F1=0.3027 → expected ~0.50+)
- Stacking ensemble (meta-learner on top of RF+DT)
- Feature engineering (polynomial features, interaction terms)
- Alternative imputation (KNNImputer, IterativeImputer)
- OneHotEncoder vs OrdinalEncoder comparison

**Note**: Current model (F1=0.6505) already exceeds target (0.65), so focus on submission preparation.
