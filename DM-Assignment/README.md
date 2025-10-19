# INFS4203/7203 Data Mining Project

## Project Information

- Student ID: s4860387
- Project Type: Data-oriented Project
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

   - Numerical features: [median imputation]
   - Categorical features: [mode imputation]

2. **Feature Normalization**
   #TODO:

   - Method: StandardScaler
   - Applied to: use in outlier detection on LOF method
   - Justification: [reason for selection]

3. **Outlier Detection and Handling**
   #TODO:

   - Detection method: [e.g., IQR / Z-score]
   - Handling strategy: [removal / capping / retention]
   - Justification: [rationale]

4. **Categorical Feature Encoding**
   - Method: OrdinalEncoder
   - Applied to: list of categorical columns
   - Justification:

### Classification Model(s)

- **Primary Model**: Random Forest / Decision Tree
- **Model Rationale**: [why this model was chosen]
- **Ensemble Strategy**: [if applicable, describe voting mechanism]
  - Component models: Random Forest + Decision Tree
  - Combination method: soft weighted voting

### Hyperparameters

#### **Model 1: Random Forest**

- "n_estimators": 80,
- "max_depth": 10,
- "min_samples_split": 2,
- "min_samples_leaf": 2,
- "max_features": None,
- "class_weight": "balanced",
- "random_state": 42,

#### **Model 2: Decision Tree**

- "criterion": "gini",
- "splitter": "best",
- "max_depth": 10,
- "min_samples_split": 40,
- "min_samples_leaf": 4,
- "max_features": None,
- "random_state": RANDOM_SEED,

#### **Model 3: Voting Classifier**

- "voting": "soft",
- "weights": [2, 1],

### Cross-Validation Results on Training Data

- Accuracy: X.XXX ± Y.YYY
- F1 Score: 0.6505 ± 0.0164
- CV Strategy: 5-fold cross-validation
- Random Seed: 42

## Reproduction Instructions

### File Structure

```
DM-Assignment/
├── main.py                 # Main execution file
├── preprocessing.py        # Preprocessing pipeline
├── model_selection.py      # Model selection and tuning
├── train_evaluate.py       # Training and evaluation
├── utils.py               # Utility functions
├── train.csv              # Training data
├── test_data.csv          # Test data
├── s4860387.infs4203      # Result report
├── requirements.txt       # Package dependencies
└── README.md              # This file
```

### Step-by-Step Execution

#### Step 1: Environment Preparation

```bash
pip install -r requirements.txt
```

#### Step 2: Run Complete Pipeline

```bash
python main.py
```

This will:

1. Load and preprocess training data
2. Perform cross-validation on training data
3. Train final model on full training data
4. Generate predictions on test data
5. Create result report file: s4860387.infs4203

#### Step 3: Verify Results

The generated file `s4860387.infs4203` should contain:

- Rows 1-2713: Test predictions (0 or 1)
- Row 2714: Cross-validation accuracy, F1 score

## Model Selection and Hyperparameter Tuning

### Baseline Model Comparison

Four baseline models were evaluated:

1. Decision Tree
2. Random Forest
3. k-Nearest Neighbors
4. Naive Bayes

**Evaluation Results:**

| Model         | Accuracy        | F1 Score        |
| ------------- | --------------- | --------------- |
| Random Forest | 0.8331 ± 0.0069 | 0.5742 ± 0.0226 |
| Decision Tree | 0.7650 ± 0.0067 | 0.5344 ± 0.0185 |
| Naïve Bayes   | 0.7915 ± 0.0038 | 0.3997 ± 0.0233 |
| k-NN          | 0.7159 ± 0.0068 | 0.2003 ± 0.0075 |

### Preprocessing Technique Selection

Evaluated combinations:
#TODO:

- Missing value strategies: [list methods tested]
- Normalization methods: [list methods tested]
- Outlier handling: [list strategies tested]
- Encoding schemes: [list methods tested]

**Selection Process:**

- Compared each technique using 5-fold CV on training data
- Selected configuration that maximized F1 score

### Hyperparameter Tuning Process

#TODO:
**Search Strategy:** [Grid Search / Random Search / Manual Tuning]

**Parameter Ranges Explored:**

#TODO:

- Model 1:
  - Parameter 1: [min, max] or [list of values]
  - Parameter 2: [min, max] or [list of values]
  - Rationale: [why these ranges]

**Tuning Results:**

- Best parameters: [final parameter set]
- Performance improvement: [before vs after tuning]

### Ensemble Exploration [if applicable]

- Combinations tested: [list ensemble combinations]
- Voting mechanisms: [majority / weighted]
- Final ensemble composition: [selected combination]
- Justification: [why this ensemble]
