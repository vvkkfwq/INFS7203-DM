# 新模块化训练框架使用指南

## 📚 概述

新的模块化框架极大简化了模型训练和调优代码,消除了重复代码,提供了统一优雅的接口。

## 🏗️ 框架架构

```
src/
├── models/                      # 核心训练模块
│   ├── base_trainer.py          # 基础训练器抽象类
│   ├── baseline_trainer.py      # Baseline模型训练器
│   ├── hyperparameter_tuner.py  # 超参数调优器
│   └── configs/                 # 模型配置
│       ├── model_configs.py     # 模型工厂和元数据
│       └── param_grids.py       # 参数网格定义
└── utils/                       # 工具函数
    ├── metrics.py               # 指标计算和展示
    └── io.py                    # 结果保存/加载
```

## 🚀 快速开始

### 1. 训练Baseline模型 (2行代码!)

```python
from src.models import BaselineTrainer

trainer = BaselineTrainer(model_name="random_forest")
results = trainer.train()
```

### 2. 超参数调优 (2行代码!)

```python
from src.models import HyperparameterTuner

tuner = HyperparameterTuner(model_name="random_forest", param_grid_version="v3")
results = tuner.train()
```

### 3. 更简单 - 使用便捷函数 (1行代码!)

```python
from src.models.baseline_trainer import train_baseline_model
from src.models.hyperparameter_tuner import tune_model

# 训练baseline
results = train_baseline_model("decision_tree")

# 调优模型
results = tune_model("random_forest", param_grid_version="v3", baseline_score=0.5742)
```

## 📖 详细使用示例

### 示例1: 训练所有Baseline模型

```python
from src.models import BaselineTrainer

models = ["decision_tree", "random_forest", "knn", "naive_bayes"]

for model_name in models:
    trainer = BaselineTrainer(model_name=model_name)
    results = trainer.train()
    print(f"{model_name}: F1 = {results['cv_scores']['f1_mean']:.4f}")
```

### 示例2: 调优Random Forest (多个版本)

```python
from src.models import HyperparameterTuner

# 尝试不同的参数网格版本
for version in ["v1", "v2", "v3", "v4"]:
    tuner = HyperparameterTuner(
        model_name="random_forest",
        param_grid_version=version,
        baseline_score=0.5742
    )
    results = tuner.train()
```

### 示例3: 使用自定义参数网格

```python
from src.models import HyperparameterTuner

custom_grid = {
    "n_estimators": [80, 100, 120],
    "max_depth": [8, 10, 12],
    "min_samples_split": [2, 3],
}

tuner = HyperparameterTuner(
    model_name="random_forest",
    custom_param_grid=custom_grid,
    baseline_score=0.5742
)
results = tuner.train()
```

### 示例4: 保存和加载模型

```python
from src.models import BaselineTrainer
from src.utils.io import save_model, load_model

# 训练并保存
trainer = BaselineTrainer(model_name="random_forest")
results = trainer.train()
trainer.save_model("models/my_random_forest.pkl")

# 加载模型
model_data = load_model("models/my_random_forest.pkl")
model = model_data["model"]
preprocessor = model_data["preprocessor"]
```

## 🎯 支持的模型

| 模型名称 | model_name | 需要特征缩放 | 支持并行 |
|---------|-----------|------------|---------|
| Decision Tree | `decision_tree` | ❌ | ❌ |
| Random Forest | `random_forest` | ❌ | ✅ |
| k-NN | `knn` | ✅ | ✅ |
| Naïve Bayes | `naive_bayes` | ❌ | ❌ |

## 📊 参数网格版本

### Decision Tree
- `v1`: 初始广泛搜索
- `v2`: 基于v1结果的精化搜索
- `v3`: 进一步优化的搜索空间

### Random Forest
- `v1`: 初始广泛搜索
- `v2`: 精化搜索 + class_weight调整
- `v3`: 专注搜索 + balanced class weights (推荐)
- `v4`: 围绕最佳结果的微调

### k-NN
- `v1`: 初始搜索
- `v2`: 精化搜索

### Naïve Bayes
- `v1`: var_smoothing参数调整

## 💡 核心优势对比

### 旧代码 (原有方式)

```python
# 需要 80+ 行代码
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# 加载数据
X_train, y_train = load_data(TRAIN_FILE)

# 预处理
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)

# 初始化模型
model = RandomForestClassifier(random_state=RANDOM_SEED)

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
scores = cross_validate(model, X_train_processed, y_train, cv=cv, ...)

# ... 更多代码处理结果 ...
```

### 新代码 (模块化框架)

```python
# 只需要 2 行代码!
from src.models import BaselineTrainer

trainer = BaselineTrainer(model_name="random_forest")
results = trainer.train()
```

**代码减少**: ~97% ✨

## 🔧 高级功能

### 自定义数据文件

```python
trainer = BaselineTrainer(
    model_name="random_forest",
    data_file="path/to/custom_data.csv"
)
```

### 使用已有的预处理器

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
# ... 预先拟合preprocessor ...

trainer = BaselineTrainer(
    model_name="random_forest",
    preprocessor=preprocessor
)
```

### 自定义模型参数

```python
trainer = BaselineTrainer(
    model_name="random_forest",
    model_params={"n_estimators": 200, "max_depth": 15}
)
```

### 静默模式

```python
results = train_baseline_model("random_forest", verbose=False)
```

## 📦 运行示例脚本

```bash
# 快速开始示例
python examples/quick_start.py

# Baseline训练示例
python examples/train_baseline_example.py

# 超参数调优示例
python examples/tune_model_example.py
```

## 🎓 最佳实践

1. **先训练Baseline** - 使用默认参数建立性能基线
2. **选择最佳模型** - 比较所有baseline,选择F1 Score最高的
3. **逐步调优** - 从v1开始,逐步尝试v2, v3等版本
4. **保存结果** - 使用`save_tuning_results()`保存每次实验
5. **记录对比** - 始终提供baseline_score来对比改进

## 🔍 常见问题

**Q: 如何添加新的参数网格版本?**

A: 在 `src/models/configs/param_grids.py` 中添加:

```python
RANDOM_FOREST_GRIDS = {
    # ... 现有版本 ...
    "v5": {
        "n_estimators": [100, 150],
        "max_depth": [10, 12],
        # ...
    }
}
```

**Q: 如何支持新的模型?**

A: 在 `src/models/configs/model_configs.py` 中注册模型:

```python
MODEL_REGISTRY = {
    # ... 现有模型 ...
    "new_model": NewModelClass,
}
```

**Q: 结果保存在哪里?**

A: 默认保存在 `results/` 目录,可以自定义路径。

## 📝 总结

新的模块化框架提供了:

- ✅ **极简接口** - 2行代码完成训练
- ✅ **零重复** - DRY原则,统一的训练流程
- ✅ **高可维护** - 修改一处,所有模型受益
- ✅ **易扩展** - 添加新模型/参数网格只需配置
- ✅ **优雅设计** - 清晰的抽象和职责分离

享受更高效的机器学习开发体验! 🚀
