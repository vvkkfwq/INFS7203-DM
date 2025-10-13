import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import load_data, DataPreprocessor
from src.utils.config import TRAIN_FILE, TEST_FILE, RANDOM_SEED, CV_FOLDS, CV_SHUFFLE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


# Build the best model
def create_best_model():

    print("\n   Model Configuration:")

    # Random Forest

    # Random Forest
    rf_params = {
        "n_estimators": 80,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": None,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    rf = RandomForestClassifier(**rf_params)

    print("\n     Random Forest Parameter:")
    for key, value in rf_params.items():
        if key != "n_jobs":
            print(f"      - {key}: {value}")

    # Decision Tree
    dt_params = {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": 10,
        "min_samples_split": 40,
        "min_samples_leaf": 4,
        "max_features": None,
        "random_state": RANDOM_SEED,
    }

    dt = DecisionTreeClassifier(**dt_params)

    print("\n     Decision Tree Parameter:")
    for key, value in dt_params.items():
        print(f"      - {key}: {value}")

    # Voting Classifier
    voting_params = {"voting": "soft", "weights": [2, 1]}

    voting_clf = VotingClassifier(estimators=[("rf", rf), ("dt", dt)], **voting_params)

    print("\n      Voting Classifier:")
    for key, value in voting_params.items():
        print(f"      - {key}: {value}")

    return voting_clf


# Train model
def train_and_evaluate(model, X, y):
    """Train the model and calculate cross-validation scores."""

    # Cross-validation
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    print("   Running 5-fold cross-validation...")

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=["f1", "accuracy"], n_jobs=-1
    )

    print("   Cross-validation completed!")

    # Calculate statistics
    results = {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
    }

    # Train the final model on the complete training set.
    print("   Training final model on complete dataset...")
    model.fit(X, y)
    print("   Final model training completed!")

    return model, results


def save_submission(predictions, cv_accuracy, cv_f1, output_file="s4860387.infs4203"):
    """Save and submit file"""

    with open(output_file, "w") as f:

        for pred in predictions:
            f.write(f"{pred},\n")

        # Write last line: accuracy,f1,
        f.write(f"{cv_accuracy:.3f}, {cv_f1:.3f},\n")

    print(f" Submission file saved: {output_file}")


def main():
    print("=" * 70)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/6] Loading training data...")
    X_train, y_train, X_test = load_data(TRAIN_FILE, TEST_FILE)

    print(f"  ✓ Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"  ✓ Class distribution: {dict(y_train.value_counts())}")

    # Step 2: Preprocess data
    print("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"  ✓ Preprocessing complete")
    print(
        f"  ✓ Missing values after preprocessing(train set): {X_train_processed.isnull().sum().sum()}"
    )
    print(
        f"  ✓ Missing values after preprocessing(test set): {X_test_processed.isnull().sum().sum()}"
    )

    # Step 3: Create models
    print("\n[3/6] Creating models: RF + DT with VotingClassifier...")
    model = create_best_model()

    # Step 4: Traning
    print("\n[4/6] Training and evaluating...")
    model, cv_results = train_and_evaluate(model, X_train_processed, y_train)

    # Print results
    print(f"\n Cross-validation results:")
    print(
        f"  ✓ Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}"
    )
    print(f"  ✓ F1 Score: {cv_results['f1_mean']:.3f} ± {cv_results['f1_std']:.3f}")

    # Predict
    print("\n[5/6] Predict test set...")
    predictions = model.predict(X_test_processed)

    print("\n[6/6] Saving submission file...")
    save_submission(
        predictions,
        round(cv_results["accuracy_mean"], 3),
        round(cv_results["f1_mean"], 3),
    )

    print("=" * 70)
    print("\n Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
