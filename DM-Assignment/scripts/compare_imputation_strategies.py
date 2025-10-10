"""
比较不同插补策略的性能（正确的交叉验证方式）

确保没有数据泄露：
- 每个 fold 独立创建插补器
- 只在训练集上 fit
- 验证集只 transform
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score

from src.utils.config import RANDOM_SEED, TRAIN_FILE
from src.preprocessing import (
    GlobalMedianModeImputer,
    GlobalMeanModeImputer,
    ConstantImputer,
)

# 设置随机种子
np.random.seed(RANDOM_SEED)


def compare_imputation_strategies():
    """
    比较不同插补策略的性能

    方法：5-fold 交叉验证
    模型：DecisionTreeClassifier (max_depth=10)
    指标：F1 Score (primary), Accuracy (secondary)
    """

    print("=" * 70)
    print("比较插补策略性能（无数据泄露的交叉验证）")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df.drop("Target (Col44)", axis=1)
    y = train_df["Target (Col44)"]

    print(f"   数据形状: {X.shape}")
    print(f"   缺失值比例: {X.isnull().sum().sum() / X.size * 100:.2f}%")

    # 2. 识别特征类型
    num_cols = [col for col in X.columns if col.startswith("Num_")]
    cat_cols = [col for col in X.columns if col.startswith("Nom_")]

    print(f"\n2. 特征类型:")
    print(f"   数值特征: {len(num_cols)} 列")
    print(f"   类别特征: {len(cat_cols)} 列")

    # 3. 定义要比较的策略
    strategies = {
        "GlobalMedianModeImputer": lambda: GlobalMedianModeImputer(num_cols, cat_cols),
        "GlobalMeanModeImputer": lambda: GlobalMeanModeImputer(num_cols, cat_cols),
        "ConstantImputer": lambda: ConstantImputer(num_cols, cat_cols),
    }

    # 4. 交叉验证设置
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 5. 存储结果
    results = []

    # 6. 对每个策略进行交叉验证
    for strategy_name, strategy_factory in strategies.items():
        print(f"\n{'='*70}")
        print(f"策略: {strategy_name}")
        print(f"{'='*70}")

        f1_scores = []
        acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            # 6.1 分割数据
            X_train_fold = X.iloc[train_idx].reset_index(drop=True)
            X_val_fold = X.iloc[val_idx].reset_index(drop=True)
            y_train_fold = y.iloc[train_idx].reset_index(drop=True)
            y_val_fold = y.iloc[val_idx].reset_index(drop=True)

            # 6.2 创建新的插补器（每个 fold 独立）
            imputer = strategy_factory()

            # 6.3 插补（只在训练集上 fit）
            imputer.fit(X_train_fold)
            X_train_filled = imputer.transform(X_train_fold)
            X_val_filled = imputer.transform(X_val_fold)

            # 验证无缺失值
            assert X_train_filled.isnull().sum().sum() == 0, "训练集仍有缺失值！"
            assert X_val_filled.isnull().sum().sum() == 0, "验证集仍有缺失值！"

            # 6.4 编码（只在训练集上 fit）
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(X_train_filled[cat_cols])

            X_train_filled[cat_cols] = encoder.transform(X_train_filled[cat_cols])
            X_val_filled[cat_cols] = encoder.transform(X_val_filled[cat_cols])

            # 6.5 训练模型
            model = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10)
            model.fit(X_train_filled, y_train_fold)

            # 6.6 评估
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
