# 项目技术速查卡 | Project Technical Quick Reference
# INFS7203 Data Mining Project

> 快速查找技术细节和关键数据  
> Quick reference for technical details and key data

---

## 📊 项目核心数据 | Core Project Data

### 数据集基本信息 | Dataset Basic Info
```
训练集 Training Set:
├── 样本数 Samples: 10,853
├── 特征数 Features: 43
│   ├── 数值特征 Numerical: 25 (Num_Col1-25)
│   └── 类别特征 Categorical: 18 (Nom_Col26-43)
├── 标签 Target: Binary (0/1)
└── 类别分布 Class Distribution: 75% (0) / 25% (1)

测试集 Test Set:
├── 样本数 Samples: 2,713
└── 特征数 Features: 43 (same as training)
```

### 数据质量 | Data Quality
```
缺失值 Missing Values:
├── 存在缺失值 Has missing: Yes (NaN)
├── 缺失率范围 Rate range: 0-15%
└── 处理方法 Handling: Mean/Mode Imputation

异常值 Outliers:
├── 检测方法 Detection: Local Outlier Factor (LOF)
├── 移除数量 Removed: 109 samples
└── 移除比例 Percentage: 1.0% of training data

类别不平衡 Class Imbalance:
├── 比例 Ratio: 3:1 (75:25)
└── 处理方法 Solution: class_weight='balanced'
```

---

## 🔧 技术栈详情 | Technology Stack Details

### Python环境 | Python Environment
```python
Python Version: 3.10.18
Operating System: macOS 26
Processor: 2.6 GHz 6-Core Intel Core i7

核心依赖 Core Dependencies:
├── numpy==2.2.6
├── pandas==2.3.2
└── scikit-learn==1.7.2
```

### 开发工具 | Development Tools
```
IDE/Editors:
├── Jupyter Notebook (数据探索 Data Exploration)
└── VS Code/PyCharm (代码开发 Code Development)

版本控制 Version Control:
└── Git + GitHub

文档工具 Documentation:
└── Markdown + LaTeX
```

---

## 🤖 机器学习模型详情 | ML Models Details

### 1. Random Forest (随机森林)

#### 最优超参数 | Optimal Hyperparameters
```python
RandomForestClassifier(
    n_estimators=100,        # 树的数量
    max_depth=10,            # 最大深度
    min_samples_split=8,     # 最小分裂样本数
    min_samples_leaf=2,      # 最小叶子节点样本数
    max_features=None,       # 分裂时考虑的特征数
    class_weight='balanced', # 类别权重平衡
    random_state=42          # 随机种子
)
```

#### 性能指标 | Performance Metrics
```
基线 Baseline:
├── Accuracy: 0.8313 ± 0.0071
└── F1 Score: 0.5664 ± 0.0289

调优后 After Tuning:
├── Accuracy: 0.8350 ± 0.0065
└── F1 Score: 0.6470 ± 0.0115

提升 Improvement:
└── F1: +0.0806 (+14.2%)
```

#### 调优历程 | Tuning History
```
v0: 基线 Baseline → F1: 0.5664
v1: 初步调参 Initial tuning → F1: 0.5995
v2: 类别权重 Class weight → F1: 0.6419
v3: 细化参数 Fine-tune → F1: 0.6470
```

---

### 2. Decision Tree (决策树)

#### 最优超参数 | Optimal Hyperparameters
```python
DecisionTreeClassifier(
    criterion='gini',         # 分裂标准
    splitter='random',        # 分裂策略
    max_depth=10,             # 最大深度
    min_samples_split=30,     # 最小分裂样本数
    min_samples_leaf=2,       # 最小叶子节点样本数
    max_features=None,        # 特征数
    class_weight='balanced',  # 类别权重
    random_state=42           # 随机种子
)
```

#### 性能指标 | Performance Metrics
```
基线 Baseline:
├── Accuracy: 0.7720 ± 0.0062
└── F1 Score: 0.5425 ± 0.0144

调优后 After Tuning:
├── Accuracy: 0.7880 ± 0.0075
└── F1 Score: 0.6124 ± 0.0143

提升 Improvement:
└── F1: +0.0699 (+12.9%)
```

---

### 3. k-Nearest Neighbors (k近邻)

#### 最优超参数 | Optimal Hyperparameters
```python
Pipeline([
    ('scaler', StandardScaler()),  # 特征标准化 (关键!)
    ('knn', KNeighborsClassifier(
        n_neighbors=40,     # 邻居数量
        weights='distance', # 距离权重
        algorithm='auto',   # 算法选择
        p=2                 # 欧氏距离
    ))
])
```

#### 性能指标 | Performance Metrics
```
基线 Baseline (无标准化 No scaling):
└── F1 Score: 0.2023 ± 0.0094

调优后 After Tuning (有标准化 With scaling):
└── F1 Score: 0.3045 ± 0.0112

提升 Improvement:
└── F1: +0.1022 (+50.5%)

结论 Conclusion:
StandardScaler对k-NN至关重要，但性能仍不如树模型
```

---

### 4. Naïve Bayes (朴素贝叶斯)

#### 最优超参数 | Optimal Hyperparameters
```python
GaussianNB(
    var_smoothing=1e-09  # 方差平滑参数
)
```

#### 性能指标 | Performance Metrics
```
基线和调优后 Baseline & Tuned:
├── Accuracy: 0.7916 ± 0.0055
└── F1 Score: 0.4014 ± 0.0253

结论 Conclusion:
朴素贝叶斯参数调整空间有限，性能不如树模型
```

---

### 5. Voting Classifier (投票集成)

#### 配置详情 | Configuration Details
```python
VotingClassifier(
    estimators=[
        ('rf', random_forest),    # Random Forest
        ('dt', decision_tree)     # Decision Tree
    ],
    voting='hard',        # 硬投票 (多数表决)
    weights=[1, 1]        # 等权重
)
```

#### 性能指标 | Performance Metrics
```
最终性能 Final Performance:
├── Accuracy: 0.795 ± 0.009
├── F1 Score: 0.650 ± 0.012  ✅ 达到满分
└── Test F1: 0.650

相比单模型 vs Single Models:
├── vs RF (0.647): +0.003
└── vs DT (0.612): +0.038
```

---

## 🔄 数据预处理流程 | Data Preprocessing Pipeline

### 完整流程图 | Complete Pipeline
```
原始数据 Raw Data
    ↓
[1] 缺失值填补 Missing Value Imputation
    ├── 数值特征 Numerical → Mean
    └── 类别特征 Categorical → Mode
    ↓
[2] 特征编码 Feature Encoding
    └── 类别特征 Categorical → OrdinalEncoder
    ↓
[3] 异常值检测 Outlier Detection
    ├── 标准化 StandardScaler (仅用于检测)
    ├── LOF检测 LOF Detection
    └── 移除异常 Remove outliers
    ↓
[4] 最终训练集 Final Training Set
    └── 10,744 samples (移除109个异常值)
```

### 预处理实验对比 | Preprocessing Experiments
```
测试的方案 Tested Schemes:
├── 方案1 Scheme 1: GlobalMeanMode → F1: 0.XXX
├── 方案2 Scheme 2: GlobalMedianMode → F1: 0.XXX
├── 方案3 Scheme 3: ConstantFilling → F1: 0.XXX
└── 方案4 Scheme 4: ClassSpecific → F1: 0.XXX

最优方案 Best Scheme: GlobalMeanMode (方案1)
```

---

## 📈 模型评估方法 | Model Evaluation Methods

### 交叉验证策略 | Cross-Validation Strategy
```python
StratifiedKFold(
    n_splits=5,         # 5折
    shuffle=True,       # 打乱数据
    random_state=42     # 随机种子
)

评估指标 Metrics:
├── Accuracy (准确率)
├── F1 Score (F1分数) - 主要指标 Primary
├── Precision (精确率)
└── Recall (召回率)
```

### 为什么选择F1 Score | Why F1 Score
```
原因 Reasons:
1. 数据类别不平衡 (75:25)
2. 同时考虑精确率和召回率
3. 课程要求的主要评估指标
4. 适合二分类问题
```

---

## 🔍 超参数调优细节 | Hyperparameter Tuning Details

### Grid Search 配置 | Grid Search Configuration
```python
GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',           # 优化目标
    n_jobs=-1,              # 并行计算
    verbose=2,              # 详细输出
    return_train_score=True # 返回训练分数
)
```

### Random Forest 参数搜索空间 | RF Parameter Grid
```python
param_grid_rf = {
    'n_estimators': [50, 80, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None],
    'class_weight': ['balanced']
}
# 总组合数 Total combinations: 3 × 3 × 4 × 3 × 1 × 1 = 108
```

### Decision Tree 参数搜索空间 | DT Parameter Grid
```python
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [10, 20, 30, 40, 50],
    'min_samples_leaf': [2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}
# 总组合数 Total combinations: 2 × 2 × 4 × 5 × 5 × 3 × 2 = 2,400
```

---

## 🎯 关键性能数字 | Key Performance Numbers

### 记住这些数字 | Remember These Numbers

```
基线到最终 Baseline → Final:
├── 起点 Start: F1 = 0.5664
├── 终点 End: F1 = 0.650
└── 提升 Improvement: +0.0836 (+14.8%)

关键里程碑 Key Milestones:
├── 基线对比 Baseline: 0.5664 → 选择RF和DT
├── 参数调优 Tuning: 0.6470 → RF性能最优
├── 类别权重 Class weight: 0.6419 → 处理不平衡
└── 模型集成 Ensemble: 0.650 → 达到满分 ✅

训练成本 Training Cost:
├── 单次训练 Single train: ~30秒
├── 5折CV: ~2.5分钟
└── Grid Search: ~30-60分钟
```

---

## 🏆 项目亮点数字 | Project Highlights in Numbers

```
📊 数据处理 Data Processing:
├── 处理样本数 Samples: 10,853
├── 特征维度 Features: 43
├── 缺失值填补 Imputation: 100%
└── 异常值移除 Outliers: 109 (1.0%)

🔬 实验规模 Experiment Scale:
├── 测试算法数 Algorithms: 4
├── 预处理方案 Preprocessing: 4+
├── 参数组合 Param combos: 2,500+
└── 训练模型数 Models trained: 50+

💻 代码质量 Code Quality:
├── 代码行数 Lines of code: 2,000+
├── 文档行数 Documentation: 1,500+
├── 模块数量 Modules: 10+
└── 测试脚本 Scripts: 20+

⏱️ 时间投入 Time Investment:
├── 数据探索 EDA: 10 hours
├── 预处理 Preprocessing: 8 hours
├── 模型调优 Tuning: 25 hours
├── 代码整理 Refactoring: 10 hours
└── 文档编写 Documentation: 8 hours
总计 Total: ~60 hours
```

---

## 🗂️ 项目文件结构速览 | Project File Structure Quick View

```
DM-Assignment/
├── 📄 README.md (1,400行)
├── 📄 requirements.txt
├── 📁 src/ (源代码 Source)
│   ├── main.py (主入口 Main entry)
│   ├── models/ (模型 Models)
│   ├── preprocessing/ (预处理 Preprocessing)
│   └── utils/ (工具 Utilities)
├── 📁 scripts/ (实验 Experiments)
│   ├── baseline/
│   ├── decision_tree_tuning/
│   ├── random_forest_tuning/
│   ├── ensemble_tuning/
│   └── preprocessing/
├── 📁 notebooks/ (笔记本 Notebooks)
│   ├── 01_data_exploration.ipynb
│   ├── 02_missing_value_detail.ipynb
│   └── 03_missing_value_experiments.ipynb
├── 📁 data/ (数据 Data)
│   ├── train.csv
│   └── test_data.csv
├── 📁 results/ (结果 Results)
│   ├── baseline_results.csv
│   └── tuning_results/
├── 📁 reports/ (报告 Reports)
│   ├── data_exploration_report.md
│   └── cardinality.md
├── 📁 docs/ (文档 Documentation)
│   ├── Project_Requirements.md
│   └── Task_Checklist.md
└── 📄 s4860387.infs4203 (最终结果 Final result)
```

---

## 💡 快速回答模板 | Quick Answer Templates

### 30秒版本 | 30-Second Version
```
"我在数据挖掘课程中完成了一个二分类项目，
使用Random Forest和Decision Tree集成，
处理了10,853条训练数据，43个特征，
最终F1 Score达到0.650，获得满分。"
```

### 1分钟版本 | 1-Minute Version
```
"项目目标是构建二分类器预测用户行为。
数据有10,853个样本，43个特征，存在类别不平衡(75:25)和缺失值。

我的方法：
1. 数据探索和预处理（填补缺失值、检测异常值）
2. 对比4种算法（RF、DT、KNN、NB）
3. 超参数调优（Grid Search + 5折CV）
4. 集成学习（Hard Voting）

最终F1达到0.650，相比基线提升14.8%，获得满分。"
```

### 2分钟版本 | 2-Minute Version
```
"这是一个完整的机器学习项目，从数据分析到模型部署。

数据特点：
- 10,853条训练数据，43个特征（25数值+18类别）
- 类别不平衡75:25
- 存在缺失值和异常值

我的解决方案分4步：
1. EDA：发现不平衡和缺失值问题
2. 预处理：均值/众数填补、LOF异常检测、序数编码
3. 模型选择：对比4种算法，树模型表现最好
4. 优化提升：超参数调优+类别权重+集成学习

关键突破点：
- 使用class_weight='balanced'处理不平衡，F1+7%
- Hard Voting集成RF和DT，达到0.650目标

技术栈：Python + scikit-learn + Pandas
代码采用模块化OOP设计，完全可复现。"
```

---

## 🔗 相关链接 | Related Links

```
GitHub仓库 Repository:
https://github.com/vvkkfwq/INFS7203-DM

主要文档 Main Documents:
├── ASSIGNMENT_SUMMARY.md (项目总结 Project summary)
├── INTERVIEW_GUIDE.md (面试指南 Interview guide)
└── TECHNICAL_REFERENCE.md (本文档 This document)

课程信息 Course Info:
├── 课程 Course: INFS7203 Data Mining
├── 学校 Institution: University of Queensland
└── 学期 Semester: 2025 S2
```

---

## ✅ 面试前检查清单 | Pre-Interview Checklist

```
□ 能在1分钟内流利介绍项目
□ 记住关键性能数字（0.650, 14.8%, 10,853等）
□ 能解释每个技术选择的原因
□ 准备好GitHub链接
□ 能画出模型架构图
□ 能解释preprocessing pipeline
□ 理解交叉验证的作用
□ 能讨论遇到的挑战和解决方案
□ 准备好回答"为什么"类问题
□ 自信且诚实的态度
```

---

**最后更新 Last Updated**: 2026-01-30  
**版本 Version**: 1.0

---

祝面试成功！Good luck with your interviews! 🎉
