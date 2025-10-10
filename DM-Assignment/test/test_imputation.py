"""
测试插补模块功能

验证 GlobalMedianModeImputer 的正确性
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.preprocessing import GlobalMedianModeImputer
from src.utils.config import RANDOM_SEED, TRAIN_FILE

# 设置随机种子
np.random.seed(RANDOM_SEED)


def test_global_median_mode_imputer():
    """测试 GlobalMedianModeImputer 基本功能"""

    print("=" * 70)
    print("测试 GlobalMedianModeImputer")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. 加载训练数据...")
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df.drop("Target (Col44)", axis=1)
    y = train_df["Target (Col44)"]

    print(f"   数据形状: {X.shape}")
    print(f"   缺失值总数: {X.isnull().sum().sum()}")
    print(f"   缺失值比例: {X.isnull().sum().sum() / X.size * 100:.2f}%")

    # 2. 识别特征类型
    print("\n2. 识别特征类型...")
    num_cols = [col for col in X.columns if col.startswith("Num_")]
    cat_cols = [col for col in X.columns if col.startswith("Nom_")]

    print(f"   数值特征: {len(num_cols)} 列")
    print(f"   类别特征: {len(cat_cols)} 列")

    # 3. 创建插补器
    print("\n3. 创建 GlobalMedianModeImputer...")
    imputer = GlobalMedianModeImputer(num_cols=num_cols, cat_cols=cat_cols)
    print(f"   ✓ 插补器创建成功")

    # 4. 训练插补器
    print("\n4. 在训练集上 fit...")
    imputer.fit(X)
    print(f"   ✓ fit 完成")

    # 验证学到的参数
    print(f"   学到的数值特征中位数样例 (前3个):")
    for i, col in enumerate(num_cols[:3]):
        median_value = imputer.num_imputer_.statistics_[i]
        print(f"     {col}: {median_value:.4f}")

    print(f"   学到的类别特征众数样例 (前3个):")
    for i, col in enumerate(cat_cols[:3]):
        mode_value = imputer.cat_imputer_.statistics_[i]
        print(f"     {col}: {mode_value}")

    # 5. 应用插补
    print("\n5. 应用插补到训练集...")
    X_filled = imputer.transform(X)
    print(f"   ✓ transform 完成")
    print(f"   插补后形状: {X_filled.shape}")
    print(f"   插补后缺失值总数: {X_filled.isnull().sum().sum()}")

    # 6. 验证结果
    print("\n6. 验证插补结果...")

    # 检查1: 无缺失值
    assert X_filled.isnull().sum().sum() == 0, "❌ 仍有缺失值！"
    print(f"   ✓ 检查1: 插补后无缺失值")

    # 检查2: 形状不变
    assert X_filled.shape == X.shape, "❌ 形状改变！"
    print(f"   ✓ 检查2: 数据形状不变")

    # 检查3: 列名不变
    assert list(X_filled.columns) == list(X.columns), "❌ 列名改变！"
    print(f"   ✓ 检查3: 列名顺序不变")

    # 检查4: 非缺失值不变
    # 选择第一行（无缺失值的行）
    non_missing_rows = X.dropna()
    if len(non_missing_rows) > 0:
        sample_row_idx = non_missing_rows.index[0]
        original_row = X.loc[sample_row_idx]
        filled_row = X_filled.loc[sample_row_idx]

        # 对于数值特征
        for col in num_cols:
            assert original_row[col] == filled_row[col], f"❌ {col} 值改变！"

        # 对于类别特征
        for col in cat_cols:
            assert original_row[col] == filled_row[col], f"❌ {col} 值改变！"

        print(f"   ✓ 检查4: 非缺失值保持不变")

    # 7. 测试fit_transform
    print("\n7. 测试 fit_transform...")
    imputer2 = GlobalMedianModeImputer(num_cols=num_cols, cat_cols=cat_cols)
    X_filled2 = imputer2.fit_transform(X)

    assert X_filled2.isnull().sum().sum() == 0, "❌ fit_transform 失败！"
    print(f"   ✓ fit_transform 正常工作")

    # 8. 测试训练/测试一致性
    print("\n8. 测试训练/测试集一致性...")
    # 模拟分割数据
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=RANDOM_SEED)

    # 创建新的插补器
    imputer3 = GlobalMedianModeImputer(num_cols=num_cols, cat_cols=cat_cols)
    imputer3.fit(X_train)

    X_train_filled = imputer3.transform(X_train)
    X_test_filled = imputer3.transform(X_test)

    assert X_train_filled.isnull().sum().sum() == 0, "❌ 训练集有缺失值！"
    assert X_test_filled.isnull().sum().sum() == 0, "❌ 测试集有缺失值！"
    print(f"   ✓ 训练集和测试集均无缺失值")
    print(f"   训练集形状: {X_train_filled.shape}")
    print(f"   测试集形状: {X_test_filled.shape}")

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！GlobalMedianModeImputer 工作正常")
    print("=" * 70)


def test_factory_function():
    """测试 get_imputer 工厂函数"""

    print("\n" + "=" * 70)
    print("测试 get_imputer 工厂函数")
    print("=" * 70)

    from src.preprocessing import get_imputer

    # 加载数据
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df.drop("Target (Col44)", axis=1)

    num_cols = [col for col in X.columns if col.startswith("Num_")]
    cat_cols = [col for col in X.columns if col.startswith("Nom_")]

    # 测试工厂函数
    print("\n1. 使用工厂函数创建插补器...")
    imputer = get_imputer("median_mode", num_cols=num_cols, cat_cols=cat_cols)
    print(f"   ✓ 创建成功: {type(imputer).__name__}")

    # 测试功能
    print("\n2. 测试插补功能...")
    imputer.fit(X)
    X_filled = imputer.transform(X)

    assert X_filled.isnull().sum().sum() == 0, "❌ 插补失败！"
    print(f"   ✓ 插补成功，无缺失值")

    # 测试错误处理
    print("\n3. 测试错误处理...")
    try:
        get_imputer("invalid_strategy", num_cols=num_cols, cat_cols=cat_cols)
        print("   ❌ 应该抛出 ValueError!")
    except ValueError as e:
        print(f"   ✓ 正确抛出异常: {str(e)[:50]}...")

    try:
        get_imputer("median_mode", num_cols=None, cat_cols=cat_cols)
        print("   ❌ 应该抛出 ValueError!")
    except ValueError as e:
        print(f"   ✓ 正确抛出异常: {str(e)[:50]}...")

    print("\n" + "=" * 70)
    print("✅ 工厂函数测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        # 运行测试
        test_global_median_mode_imputer()
        test_factory_function()

        print("\n🎉 全部测试完成！插补模块可以正常使用了。")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
