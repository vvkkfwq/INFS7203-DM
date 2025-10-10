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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.utils.config import CV_FOLDS, CV_SHUFFLE, RANDOM_SEED, TRAIN_FILE, TARGET_COL
from src.preprocessing import (
    GlobalMedianModeImputer,
    GlobalMeanModeImputer,
    ConstantImputer,
    ClassSpecificImputer,
)

np.random.seed(RANDOM_SEED)


def compare_imputation_strategies():

    print("=" * 70)
    print("Starting experiments with preprocessing parameters.")
    print("=" * 70)

    print("\n1. Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df.drop(TARGET_COL, axis=1)
    y = train_df[TARGET_COL]

    print(f"\n2. Dataset Info:")
    print(f"   Training samples: {len(X)}")
    print(f"   Data shape: {X.shape}")
    print(f"   Training set missing rate: {X.isnull().sum().sum() / X.size * 100:.2f}%")
    print(
        f"   Rows with at least one missing: {X.isnull().any(axis=1).sum()} ({X.isnull().any(axis=1).sum()/len(X)*100:.2f}%)"
    )

    num_cols = [col for col in X.columns if col.startswith("Num_")]
    cat_cols = [col for col in X.columns if col.startswith("Nom_")]

    print(f"\n3. Feature Type:")
    print(f"   Numerical features: {len(num_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")

    print(f"\n4. Target Distribution:")
    print(f"   {y.value_counts()}")
    print(
        f"   Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1"
    )

    strategies = {
        "GlobalMedianModeImputer": lambda: GlobalMedianModeImputer(num_cols, cat_cols),
        "GlobalMeanModeImputer": lambda: GlobalMeanModeImputer(num_cols, cat_cols),
        "ConstantImputer": lambda: ConstantImputer(num_cols, cat_cols),
        "ClassSpecificImputer": lambda: ClassSpecificImputer(num_cols, cat_cols),
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_SEED
    )

    results = []

    for strategy_name, strategy_factory in strategies.items():
        print(f"\n{'='*70}")
        print(f"Strategy name: {strategy_name}")
        print(f"{'='*70}")

        f1_scores = []
        acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

            X_train_fold = X.iloc[train_idx].reset_index(drop=True)
            X_val_fold = X.iloc[val_idx].reset_index(drop=True)
            y_train_fold = y.iloc[train_idx].reset_index(drop=True)
            y_val_fold = y.iloc[val_idx].reset_index(drop=True)

            imputer = strategy_factory()

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

            # Feature Encoding
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(X_train_filled[cat_cols])

            X_train_filled[cat_cols] = encoder.transform(X_train_filled[cat_cols])
            X_val_filled[cat_cols] = encoder.transform(X_val_filled[cat_cols])

            # Train model
            model = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10)
            model.fit(X_train_filled, y_train_fold)

            # Predict and evaluate
            y_pred = model.predict(X_val_filled)
            f1 = f1_score(y_val_fold, y_pred)
            acc = accuracy_score(y_val_fold, y_pred)

            f1_scores.append(f1)
            acc_scores.append(acc)

            print(f"  Fold {fold}: F1={f1:.4f}, Accuracy={acc:.4f}")

        # 6.7 汇总结果
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_acc = np.mean(acc_scores)
        std_acc = np.std(acc_scores)

        results.append(
            {
                "Strategy": strategy_name,
                "F1_Mean": mean_f1,
                "F1_Std": std_f1,
                "Accuracy_Mean": mean_acc,
                "Accuracy_Std": std_acc,
            }
        )

        print(
            f"\n  汇总: F1={mean_f1:.4f} ± {std_f1:.4f}, "
            f"Accuracy={mean_acc:.4f} ± {std_acc:.4f}"
        )

    # 7. 显示结果
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1_Mean", ascending=False)

    print(results_df.to_string(index=False))

    # 8. 高亮最佳策略
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"🏆 最佳策略: {best['Strategy']}")
    print(f"   F1 Score: {best['F1_Mean']:.4f} ± {best['F1_Std']:.4f}")
    print(f"   Accuracy: {best['Accuracy_Mean']:.4f} ± {best['Accuracy_Std']:.4f}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    try:
        results = compare_imputation_strategies()
        print("\n✅ 比较完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
