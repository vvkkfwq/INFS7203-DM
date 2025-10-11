"""
Preprocessing Emperiment
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.metrics import f1_score, accuracy_score

from src.utils.config import CV_FOLDS, CV_SHUFFLE, RANDOM_SEED, TRAIN_FILE, TARGET_COL
from src.preprocessing import (
    GlobalMedianModeImputer,
    GlobalMeanModeImputer,
    ConstantImputer,
    ClassSpecificImputer,
    IsolationForestDetector,
    LOFDetector,
)

np.random.seed(RANDOM_SEED)


# Comparison pipeline
def compare_preprocessing_pipelines():
    """
    Compare different preprocessing pipeline combinations (imputation + outlier detection).

    Pipeline: Imputation → Outlier Detection → Encoding → Training
    """

    print("=" * 70)
    print("Starting Preprocessing Pipeline Comparison")
    print("=" * 70)

    print("\n1. Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df.drop(TARGET_COL, axis=1)
    y = train_df[TARGET_COL]

    print(f"\n2. Dataset Info:")
    print(f"   Training samples: {len(X)}")
    print(f"   Data shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    num_cols = [col for col in X.columns if col.startswith("Num_")]
    cat_cols = [col for col in X.columns if col.startswith("Nom_")]

    print(f"\n3. Feature Types:")
    print(f"   Numerical features: {len(num_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")

    # Define default strategies
    imputation_strategies = {
        "GlobalMedianModeImputer": lambda: GlobalMedianModeImputer(num_cols, cat_cols),
        "GlobalMeanModeImputer": lambda: GlobalMeanModeImputer(num_cols, cat_cols),
        "ConstantImputer": lambda: ConstantImputer(num_cols, cat_cols),
        "ClassSpecificImputer": lambda: ClassSpecificImputer(num_cols, cat_cols),
    }

    outlier_strategies = {
        "NoOutlier": lambda: None,
        "IsoForest_0.05": lambda: IsolationForestDetector(num_cols, contamination=0.05),
        "IsoForest_0.10": lambda: IsolationForestDetector(num_cols, contamination=0.10),
        "LOF_k20": lambda: LOFDetector(num_cols, n_neighbors=20, contamination=0.10),
    }

    feature_encoder_strategies = {
        "OrdinalEncoder": lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
        "TargetEncoder": lambda: TargetEncoder(random_state=RANDOM_SEED),
    }

    print(f"\n4. Pipeline Configurations:")
    print(f"   Imputation strategies: {len(imputation_strategies)}")
    print(f"   Outlier strategies: {len(outlier_strategies)}")
    print(f"   Feature encoder strategies: {len(feature_encoder_strategies)}")
    print(
        f"   Total combinations: {len(imputation_strategies) * len(outlier_strategies) * len(feature_encoder_strategies)}"
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    results = []
    total_combinations = (
        len(imputation_strategies)
        * len(outlier_strategies)
        * len(feature_encoder_strategies)
    )
    current_combination = 0

    # Iterate over all combinations of imputation + outlier detection
    for impute_name, imputer_factory in imputation_strategies.items():
        for outlier_name, outlier_detector_factory in outlier_strategies.items():
            for encoder_name, encoder_factory in feature_encoder_strategies.items():

                current_combination += 1
                pipeline_name = f"{impute_name}+{outlier_name}+{encoder_name}"

                print(f"\n{'='*70}")
                print(
                    f"[{current_combination}/{total_combinations}] Pipeline: {pipeline_name}"
                )
                print(f"{'='*70}")

                f1_scores = []
                acc_scores = []
                outlier_counts = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

                    X_train_fold = X.iloc[train_idx].copy().reset_index(drop=True)
                    X_val_fold = X.iloc[val_idx].copy().reset_index(drop=True)
                    y_train_fold = y.iloc[train_idx].copy().reset_index(drop=True)
                    y_val_fold = y.iloc[val_idx].copy().reset_index(drop=True)

                    # Step 1: Imputation
                    imputer = imputer_factory()

                    if isinstance(imputer, ClassSpecificImputer):
                        imputer.fit_with_label(X_train_fold, y_train_fold)
                        X_train_filled = imputer.transform_with_label(
                            X_train_fold, y_train_fold
                        )

                        # for validation set
                        imputer.fit(X_train_fold)
                        X_val_filled = imputer.transform(X_val_fold)
                    else:
                        imputer.fit(X_train_fold)
                        X_train_filled = imputer.transform(X_train_fold)
                        X_val_filled = imputer.transform(X_val_fold)

                    # Step 2: Outlier Detection
                    n_before = len(X_train_filled)

                    outlier_detector = outlier_detector_factory()

                    if outlier_detector is not None:
                        outlier_detector.fit(X_train_filled)
                        inlier_mask = outlier_detector.get_inlier_mask()
                        X_train_filled = X_train_filled[inlier_mask].reset_index(
                            drop=True
                        )
                        y_train_fold = y_train_fold[inlier_mask].reset_index(drop=True)

                        n_removed = n_before - len(X_train_filled)
                        outlier_counts.append(n_removed)
                    else:
                        outlier_counts.append(0)

                    # Step 3: Feature Encoding
                    encoder = encoder_factory()

                    if isinstance(encoder, TargetEncoder):
                        encoder.fit(X_train_filled[cat_cols], y_train_fold)
                    else:
                        # OrdinalEncoder
                        encoder.fit(X_train_filled[cat_cols])

                    X_train_filled[cat_cols] = encoder.transform(
                        X_train_filled[cat_cols]
                    )
                    X_val_filled[cat_cols] = encoder.transform(X_val_filled[cat_cols])

                    # Step 4: Train model
                    model = DecisionTreeClassifier(
                        random_state=RANDOM_SEED, max_depth=10
                    )
                    model.fit(X_train_filled, y_train_fold)

                    # Step 5: Predict and evaluate
                    y_pred = model.predict(X_val_filled)
                    f1 = f1_score(y_val_fold, y_pred)
                    acc = accuracy_score(y_val_fold, y_pred)

                    f1_scores.append(f1)
                    acc_scores.append(acc)

                    print(
                        f"  Fold {fold}: F1={f1:.4f}, Accuracy={acc:.4f}, "
                        f"Removed={outlier_counts[-1]} ({outlier_counts[-1]/n_before*100:.1f}%)"
                    )

                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                mean_acc = np.mean(acc_scores)
                std_acc = np.std(acc_scores)
                mean_outliers = np.mean(outlier_counts)

                results.append(
                    {
                        "Pipeline": pipeline_name,
                        "Imputation": impute_name,
                        "Outlier_Detection": outlier_name,
                        "Feature_Encoding": encoder_name,
                        "F1_Mean": mean_f1,
                        "F1_Std": std_f1,
                        "Accuracy_Mean": mean_acc,
                        "Accuracy_Std": std_acc,
                        "Avg_Outliers_Removed": mean_outliers,
                    }
                )

                print(
                    f"\n  Summary: F1={mean_f1:.4f} ± {std_f1:.4f}, "
                    f"Accuracy={mean_acc:.4f} ± {std_acc:.4f}, "
                    f"Avg Removed={mean_outliers:.0f}"
                )

    print("\n" + "=" * 70)
    print("Results Summary (Sorted by F1 Score)")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1_Mean", ascending=False)

    print(results_df.to_string(index=False))

    # Highlight best pipeline
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"🏆 Best Pipeline: {best['Pipeline']}")
    print(f"   Imputation: {best['Imputation']}")
    print(f"   Outlier Detection: {best['Outlier_Detection']}")
    print(f"   Feature Encoding: {best['Feature_Encoding']}")
    print(f"   F1 Score: {best['F1_Mean']:.4f} ± {best['F1_Std']:.4f}")
    print(f"   Accuracy: {best['Accuracy_Mean']:.4f} ± {best['Accuracy_Std']:.4f}")
    print(f"   Avg Outliers Removed: {best['Avg_Outliers_Removed']:.0f}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    try:
        results = compare_preprocessing_pipelines()
        print("\n✅ Comparison complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
