# INFS4203/7203 Data Mining Project

## Project Information

- Student ID: s4860387
- Submission Date: 20th October 2025
- Final Test F1 Score: 0.650

## Environment Setup

### System Information

- Operating System: macOS 26
- Python Version: v3.10.18
- Hardware: Processor 2.6 GHz 6-Core Intel Core i7

### Required Packages

- numpy==2.2.6
- pandas==2.3.2
- scikit-learn==1.7.2

### Installation

```bash
pip install -r requirements.txt
```

## Final Model Configuration

### Preprocessing Methods

1. **Missing Value Imputation**

   - Numerical features: [mean imputation]
   - Categorical features: [mode imputation]

2. **Feature Normalization**

   - Method: StandardScaler
   - Justification: only used it for LOF outlier detection (not for
     final training)

3. **Outlier Detection and Handling**

   - Detection method: Local Outlier Factor
   - Handling strategy: Removal (109 samples, 1.0% of training data)
   - Justification: Based on preprocessing experiment results (reference
     [`scripts/preprocessing/preprocessing_experiment.py`](scripts/preprocessing/preprocessing_experiment.py))

4. **Categorical Feature Encoding**

   - Method: OrdinalEncoder
   - Justification: Universal, concise, easy to understand.

### Classification Model(s)

- **Primary Model**: Random Forest / Decision Tree
- **Model Rationale**: After grid search, their F1 scores were higher and closer to the target. Additionally, the score improvement was not significant after tuning other parameters.
- **Ensemble Strategy**:
  - Component models: Random Forest + Decision Tree
  - Combination method: hard voting

### Hyperparameters

#### **Model 1: Random Forest**

- "n_estimators": 100,
- "max_depth": 10,
- "min_samples_split": 8,
- "min_samples_leaf": 2,
- "max_features": None,
- "class_weight": "balanced",
- "random_state": 42,

#### **Model 2: Decision Tree**

- "criterion": "gini",
- "splitter": "random",
- "max_depth": 10,
- "min_samples_split": 30,
- "min_samples_leaf": 2,
- "max_features": None,
- "class_weight": "balanced",
- "random_state": 42,

#### **Model 3: Voting Classifier**

- "voting": "hard",
- "weights": [1, 1],

### Cross-Validation Results on Training Data

- Accuracy: 0.795 ± 0.009
- F1 Score: 0.6500 ± 0.0120
- CV Strategy: 5-fold StratifiedKFold cross-validation
- Random Seed: 42

## Reproduction Instructions

### File Structure

```
DM-Assignment/
├── src/                           # Source code modules
│   ├── main.py                   # Main execution file
│   ├── models/                   # Model training framework
│   │   ├── __init__.py
│   │   ├── base_trainer.py      # Abstract base trainer
│   │   ├── baseline_trainer.py  # Baseline model trainer
│   │   ├── hyperparameter_tuner.py  # Hyperparameter tuning
│   │   ├── voting_trainer.py    # Ensemble voting trainer
│   │   └── configs/             # Model configurations
│   │       ├── model_configs.py # Model registry & metadata
│   │       └── param_grids.py   # Parameter grids for tuning
│   ├── preprocessing/            # Data preprocessing
│   │   ├── __init__.py
│   │   ├── best_preprocessor.py # Final preprocessing pipeline
│   │   ├── imputation.py        # Missing value imputation
│   │   └── outlier_detection.py # Outlier detection methods
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── config.py            # Global configurations
│       ├── metrics.py           # Evaluation metrics
│       └── io.py                # File I/O operations
├── scripts/                      # Experimental scripts
│   ├── baseline/                # Baseline model training
│   │   └── train_baseline.py
│   ├── decision_tree_tuning/    # DT hyperparameter tuning
│   ├── random_forest_tuning/    # RF hyperparameter tuning
│   ├── knn_tuning/              # KNN hyperparameter tuning
│   ├── navie_nayes_tuning/      # NB hyperparameter tuning
│   ├── ensemble_tuning/         # Ensemble model tuning
│   └── preprocessing/           # Preprocessing experiments
│       └── preprocessing_experiment.py
├── data/                         # Dataset files
│   ├── train.csv                # Training data
│   └── test_data.csv            # Test data
├── results/                      # Model outputs
│   ├── baseline_results.csv     # Baseline comparison results
│   └── tuning_results/          # Hyperparameter tuning results
├── s4860387.infs4203            # Final result report
├── requirements.txt             # Package dependencies
└── README.md                    # This file
```

### Step-by-Step Execution

#### Step 1: Environment Preparation

```bash
pip install -r requirements.txt
```

#### Step 2: Run Complete Pipeline

```bash
python -u main.py
```

This will:

1. Load and preprocess training data
2. Perform cross-validation on training data
3. Train final model on full training data
4. Generate predictions on test data
5. Create result report file: s4860387.infs4203

## Model Selection and Hyperparameter Tuning

### Baseline Model Comparison

Four baseline models were evaluated:

1. Decision Tree
2. Random Forest
3. k-Nearest Neighbors
4. Naive Bayes

**Evaluation Results:**

| Model               | Accuracy        | F1 Score        |
| ------------------- | --------------- | --------------- |
| Random Forest       | 0.8313 ± 0.0071 | 0.5664 ± 0.0289 |
| Decision Tree       | 0.7720 ± 0.0062 | 0.5425 ± 0.0144 |
| Naïve Bayes         | 0.7916 ± 0.0055 | 0.4014 ± 0.0253 |
| k-Nearest Neighbors | 0.7188 ± 0.0051 | 0.2023 ± 0.0094 |

**Model Selection:**

Based on these results, **Random Forest** and **Decision Tree** were selected for further hyperparameter tuning due to their superior F1 scores. The final ensemble combines these two models to leverage their complementary strengths.

### Preprocessing Technique Selection

Evaluated combinations:

- Missing value strategies: [GlobalMeanMode / GlobalMedianMode / ClassSpecificImputation(Mean/Mode)]
- Normalization methods: StandardScaler
- Outlier handling: [IsolationForest / LOF]
- Encoding schemes: OrdinalEncoding

**Selection Process:**

- Compared each technique using 5-fold CV on training data
- Selected configuration that maximized F1 score from the result [logs/preprocessing_experiment.log)](logs/preprocessing_experiment.log)

### Hyperparameter Tuning Process

**Search Strategy:** Grid Search

**Parameter Ranges Explored:**

#### Decision Tree:

**Version v0** (Broad exploration):

- `criterion`: ["gini", "entropy"]
- `splitter`: ["best", "random"]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [10, 20, 30, 40, 50]
- `min_samples_leaf`: [2, 4, 6, 8, 10]
- `max_features`: [None, "sqrt", "log2"]
- `class_weight`: [None, "balanced"]

**Rationale:**

- Started with balanced class weights due to imbalanced dataset
- Focused on controlling tree depth and minimum samples to reduce overfitting
- Explored `splitter="random"` for additional regularization

- **Best Parameters:**

```python
{
    "criterion": "gini",
    "splitter": "random",
    "max_depth": 10,
    "min_samples_split": 30,
    "min_samples_leaf": 2,
    "max_features": None,
    "class_weight": "balanced"
}
```

- **Improvement over Baseline**:
  - F1 Score: 0.5425 → 0.6124 (+0.0699)
- **Training Scripts:**[`scripts/decision_tree_tuning/tune_dt_v0.py`](scripts/decision_tree_tuning/tune_dt_v0.py)

#### Random Forest

**Version v0**:

- `n_estimators`: [50, 80, 100]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 4, 6, 8]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: [None]
- `class_weight`: ["balanced"]

**Rationale:**

- Started with balanced class weights due to imbalanced dataset
- Focused on tree complexity (`max_depth`, `min_samples_split`) to prevent overfitting
- Limited `n_estimators` range for computational efficiency

**Tuning Results:**

- **Best Parameters:**

```python
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 8,
    "min_samples_leaf": 2,
    "max_features": None,
    "class_weight": "balanced"
}
```

- **Performance Improvement:**
  - F1 Score: 0.5664 → 0.6470 (+0.0806)
- **Training Scripts:** [`scripts/random_forest_tuning/tune_rf_v0.py`](scripts/random_forest_tuning/tune_rf_v0.py)

#### Navie Bayes:

**Version v0**:

- `var_smoothing`: [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

**Rationale:**

- Gaussian Naïve Bayes has limited hyperparameters
- `var_smoothing` adds portion of largest variance to all features for numerical stability

**Tuning Results:**

- **Best Parameters:**

```python
{
    "var_smoothing": 1e-09
}
```

- **Performance Improvement:**
  - F1 Score: 0.4014 → 0.4014 (+0.0000)
- **Training Script:** [`scripts/navie_nayes_tuning/tune_nb_v0.py`](scripts/navie_nayes_tuning/tune_nb_v0.py)

#### KNN:

**Version v0**:

- `n_neighbors`: [3, 5, 7, 9, 11, 13, 15, 20, 30, 40]
- `weights`: ["uniform", "distance"]
- `algorithm`: ["auto"]
- `p`: [1, 2] # Manhattan vs Euclidean distance

**Rationale:**

- Explored wide range of k values to find optimal neighborhood size
- Tested distance weighting to give closer neighbors more influence
- Compared Manhattan (p=1) and Euclidean (p=2) distance metrics

**Tuning Results:**

- **Best Parameters:**

```python
{
    "n_neighbors": 40,
    "weights": "distance",
    "algorithm": "auto",
    "p": 2
}
```

- **Performance Improvement:**
  - F1 Score: 0.2023 → 0.3045 (+0.1022)
- **Training Script:** [`scripts/knn_tuning/tune_knn_v0.py`](scripts/knn_tuning/tune_knn_v0.py)

#### Ensemble Exploration

- Combinations tested: [Random Forest / Decision Tree / Navie Bayes]
- Voting mechanisms: [hard or soft / weighted]
- Final ensemble composition: [Random Forest / Decision Tree]
- Justification: Random forests and single decision tree models perform best, and voting can better leverage their strengths.

## Acknowledgment

Yes. AI and machine translation tools have been used to generate material in this document.

### Details of use

### Details of AI Tool Usage

| Tool               | Use                                          | Prompt(s)                                  | Section                  | Date       |
| ------------------ | -------------------------------------------- | ------------------------------------------ | ------------------------ | ---------- |
| Claude 4.5 sonnect | Generating a draft conclusion that I adapted | How can I do cross-validation use skleanrn | model design, preprocess | 05.10.2025 |
