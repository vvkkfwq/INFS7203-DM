# INFS7203 数据挖掘课程作业总结
# INFS7203 Data Mining Course Assignment Summary

> 本文档旨在帮助回忆课程作业内容，并为求职面试做准备
> This document helps recall assignment content and prepare for job interviews

---

## 📚 课程基本信息 | Course Information

**课程名称 | Course**: INFS7203 Data Mining  
**学校 | Institution**: University of Queensland (昆士兰大学)  
**学期 | Semester**: 2025 年 9-10 月 | September-October 2025  
**学生学号 | Student ID**: s4860387

---

## 🎯 主要作业概览 | Main Assignment Overview

### 作业类型 | Assignment Type
**Track 1: Data-oriented Project** - 面向数据的二分类机器学习项目

### 核心任务 | Core Task
开发一个二分类器来预测测试数据标签（0 或 1），目标是最大化 F1 Score

Develop a binary classifier to predict test data labels (0 or 1), aiming to maximize F1 Score

### 数据集规模 | Dataset Size
- **训练集 | Training Set**: 10,853 样本，43 个特征（25 个数值 + 18 个类别）
- **测试集 | Test Set**: 2,713 样本，43 个特征
- **挑战 | Challenge**: 类别不平衡（75:25），存在缺失值

---

## 🛠️ 技术栈与实现 | Technology Stack & Implementation

### 1. 编程语言和工具 | Programming Languages & Tools

```
核心技术 | Core Technologies:
├── Python 3.10.18
├── Pandas 2.3.2 (数据处理 | Data Processing)
├── NumPy 2.2.6 (数值计算 | Numerical Computing)
├── Scikit-learn 1.7.2 (机器学习 | Machine Learning)
└── Jupyter Notebook (数据探索 | Data Exploration)
```

### 2. 机器学习算法 | Machine Learning Algorithms

允许使用的算法（Week 2-8 课程内容）：
Allowed algorithms (Week 2-8 course content):

1. **Decision Tree (决策树)** ✅ 主要模型之一
2. **Random Forest (随机森林)** ✅ 主要模型之一
3. **k-Nearest Neighbors (k近邻)** - 辅助实验
4. **Naïve Bayes (朴素贝叶斯)** - 辅助实验

**禁止使用 | Not Allowed**: XGBoost, Neural Networks, Deep Learning

### 3. 数据预处理流程 | Data Preprocessing Pipeline

```python
预处理步骤 | Preprocessing Steps:

1. 缺失值处理 | Missing Value Imputation
   ├── 数值特征 | Numerical: Mean Imputation
   └── 类别特征 | Categorical: Mode Imputation

2. 异常值检测 | Outlier Detection
   ├── 方法 | Method: Local Outlier Factor (LOF)
   ├── 处理策略 | Strategy: Removal
   └── 移除数量 | Removed: 109 samples (1.0% of training data)

3. 类别编码 | Categorical Encoding
   └── 方法 | Method: OrdinalEncoder

4. 特征标准化 | Feature Normalization
   └── 方法 | Method: StandardScaler (仅用于LOF检测)
```

### 4. 模型选择与调优 | Model Selection & Tuning

#### 基线模型对比 | Baseline Model Comparison

| 模型 Model | 准确率 Accuracy | F1 分数 F1 Score |
|------------|----------------|-----------------|
| Random Forest | 0.8313 ± 0.0071 | 0.5664 ± 0.0289 |
| Decision Tree | 0.7720 ± 0.0062 | 0.5425 ± 0.0144 |
| Naïve Bayes | 0.7916 ± 0.0055 | 0.4014 ± 0.0253 |
| k-NN | 0.7188 ± 0.0051 | 0.2023 ± 0.0094 |

#### 超参数调优 | Hyperparameter Tuning

**方法 | Method**: Grid Search with 5-Fold Cross-Validation

**最终模型 | Final Model**: Ensemble (Random Forest + Decision Tree)

**Random Forest 最优参数 | Optimal Parameters**:
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 8,
    'min_samples_leaf': 2,
    'max_features': None,
    'class_weight': 'balanced',
    'random_state': 42
}
```

**Decision Tree 最优参数 | Optimal Parameters**:
```python
{
    'criterion': 'gini',
    'splitter': 'random',
    'max_depth': 10,
    'min_samples_split': 30,
    'min_samples_leaf': 2,
    'max_features': None,
    'class_weight': 'balanced',
    'random_state': 42
}
```

**集成策略 | Ensemble Strategy**: Hard Voting Classifier

---

## 📊 项目成果 | Project Achievements

### 最终性能指标 | Final Performance Metrics

```
交叉验证结果 | Cross-Validation Results (on Training Data):
├── Accuracy: 0.795 ± 0.009
├── F1 Score: 0.650 ± 0.012
└── CV Strategy: 5-fold StratifiedKFold

测试集结果 | Test Set Results:
└── F1 Score: 0.650 (达到满分标准 | Full marks achieved)
```

### 性能提升历程 | Performance Improvement Journey

```
1. 基线模型 | Baseline:
   Random Forest F1 = 0.5664

2. 超参数调优 | Hyperparameter Tuning:
   Random Forest F1 = 0.6470 (+0.0806, +14.2%)

3. 类别权重平衡 | Class Weight Balancing:
   Decision Tree F1 = 0.6124 (+0.0699 from baseline)

4. 集成学习 | Ensemble Learning:
   Voting Classifier F1 = 0.650 (+0.0836 from baseline)
   ✅ 达到满分目标 (≥ 0.65)
```

---

## 🏗️ 项目代码结构 | Project Code Structure

```
DM-Assignment/
├── src/                          # 源代码模块 | Source Code Modules
│   ├── main.py                  # 主程序入口 | Main Entry
│   ├── models/                  # 模型训练框架 | Model Training
│   │   ├── base_trainer.py     # 抽象基类 | Base Class
│   │   ├── baseline_trainer.py # 基线模型 | Baseline
│   │   ├── hyperparameter_tuner.py # 超参数调优 | Tuning
│   │   └── voting_trainer.py   # 集成模型 | Ensemble
│   ├── preprocessing/           # 数据预处理 | Preprocessing
│   │   ├── best_preprocessor.py
│   │   ├── imputation.py
│   │   └── outlier_detection.py
│   └── utils/                   # 工具函数 | Utilities
│       ├── config.py
│       ├── metrics.py
│       └── io.py
├── scripts/                     # 实验脚本 | Experimental Scripts
│   ├── baseline/               # 基线训练 | Baseline Training
│   ├── decision_tree_tuning/   # DT 调优 | DT Tuning
│   ├── random_forest_tuning/   # RF 调优 | RF Tuning
│   ├── knn_tuning/             # KNN 调优 | KNN Tuning
│   ├── navie_bayes_tuning/     # NB 调优 | NB Tuning
│   ├── ensemble_tuning/        # 集成调优 | Ensemble Tuning
│   └── preprocessing/          # 预处理实验 | Preprocessing Exp
├── notebooks/                   # Jupyter 笔记本 | Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_missing_value_detail.ipynb
│   └── 03_missing_value_experiments.ipynb
├── data/                        # 数据集 | Datasets
├── results/                     # 结果输出 | Results
├── reports/                     # 实验报告 | Reports
└── docs/                        # 文档 | Documentation
```

---

## 💡 核心技能展示 | Core Skills Demonstrated

### 1. 数据科学全流程 | End-to-End Data Science

- ✅ **数据探索性分析 (EDA)**: 分析数据分布、缺失值、异常值
- ✅ **特征工程**: 缺失值处理、类别编码、异常值检测
- ✅ **模型选择**: 系统性对比多种算法
- ✅ **超参数调优**: Grid Search 方法
- ✅ **模型集成**: Voting Classifier
- ✅ **性能评估**: 交叉验证、F1 Score

### 2. Python 编程能力 | Python Programming

- ✅ **模块化设计**: 面向对象编程，代码结构清晰
- ✅ **工程化实践**: 配置管理、日志记录、结果复现
- ✅ **代码质量**: 文档齐全、可维护性强

### 3. 机器学习理论 | Machine Learning Theory

- ✅ **算法理解**: 决策树、随机森林、集成学习原理
- ✅ **问题分析**: 识别类别不平衡问题并解决
- ✅ **实验设计**: 系统化的模型调优策略

### 4. 项目管理能力 | Project Management

- ✅ **任务规划**: 详细的任务清单（Task_Checklist.md）
- ✅ **版本控制**: Git 管理代码和实验
- ✅ **文档编写**: 完整的 README 和技术报告

---

## 🎤 面试准备要点 | Interview Talking Points

### 项目介绍模板 | Project Introduction Template

**中文版本**:
```
"在INFS7203数据挖掘课程中，我完成了一个二分类机器学习项目。
项目使用了10,853个样本的训练数据，包含43个特征，面临类别不平衡和缺失值的挑战。

我采用了系统化的方法：
1. 进行探索性数据分析，识别数据特征
2. 设计并测试多种预处理方案
3. 对比Decision Tree、Random Forest等4种算法
4. 使用Grid Search进行超参数调优
5. 通过集成学习提升性能

最终，我的模型在测试集上达到了0.650的F1 Score，获得满分。
项目代码采用模块化设计，具有良好的可维护性和可复现性。"
```

**English Version**:
```
"In the INFS7203 Data Mining course, I completed a binary classification 
machine learning project. The project used 10,853 training samples with 
43 features, facing challenges of class imbalance and missing values.

I adopted a systematic approach:
1. Conducted exploratory data analysis to identify data characteristics
2. Designed and tested multiple preprocessing pipelines
3. Compared 4 algorithms including Decision Tree and Random Forest
4. Performed hyperparameter tuning using Grid Search
5. Improved performance through ensemble learning

Finally, my model achieved an F1 Score of 0.650 on the test set, 
receiving full marks. The project code is modularly designed with 
good maintainability and reproducibility."
```

### 技术深度问题准备 | Technical Deep-Dive Preparation

**Q1: 如何处理类别不平衡问题？**  
How did you handle class imbalance?

**A**: 我使用了 `class_weight='balanced'` 参数来自动调整类权重，使模型对少数类给予更多关注。这使 Decision Tree 的 F1 Score 从 0.5425 提升到 0.6124，提升了 12.9%。

I used the `class_weight='balanced'` parameter to automatically adjust class weights, making the model pay more attention to the minority class. This improved the Decision Tree's F1 Score from 0.5425 to 0.6124, an increase of 12.9%.

---

**Q2: 为什么选择集成学习？**  
Why did you choose ensemble learning?

**A**: 单个模型调优后接近但未达到目标（0.647），通过 Hard Voting 结合 Random Forest 和 Decision Tree 的优势，最终达到 0.650。这展示了集成学习可以提升模型鲁棒性。

After tuning individual models, they approached but didn't reach the target (0.647). By combining Random Forest and Decision Tree through Hard Voting, I achieved 0.650, demonstrating that ensemble learning can improve model robustness.

---

**Q3: 如何确保实验可复现？**  
How did you ensure reproducibility?

**A**: 
1. 固定所有随机种子为 42
2. 记录完整的软件环境（Python 3.10, scikit-learn 1.7.2）
3. 使用 requirements.txt 管理依赖
4. 详细的 README 文档说明运行步骤

1. Fixed all random seeds to 42
2. Recorded complete software environment
3. Managed dependencies with requirements.txt
4. Detailed README documentation for reproduction steps

---

**Q4: 遇到的最大挑战是什么？**  
What was the biggest challenge?

**A**: 最大挑战是在有限的算法选择下（仅允许4种基础算法）达到高性能要求。我通过：
1. 系统化的预处理实验找到最佳配置
2. 网格搜索细致调优超参数
3. 使用集成学习整合多模型优势
最终突破了性能瓶颈。

The biggest challenge was achieving high performance with limited algorithm choices (only 4 basic algorithms allowed). I overcame this by:
1. Systematic preprocessing experiments to find optimal configuration
2. Fine-tuning hyperparameters through grid search
3. Integrating multiple models through ensemble learning
This approach ultimately broke through the performance bottleneck.

---

## 📈 学习成果与收获 | Learning Outcomes

### 技术能力 | Technical Skills
- ✅ 掌握完整的机器学习项目流程
- ✅ 熟练使用 scikit-learn 进行模型开发
- ✅ 理解并解决实际数据挑战（不平衡、缺失值）
- ✅ 能够进行系统化的模型调优和评估

### 软技能 | Soft Skills
- ✅ 项目规划和时间管理
- ✅ 技术文档编写能力
- ✅ 问题分析和解决能力
- ✅ 实验设计和迭代优化思维

---

## 📂 补充实验作业 | Additional Practical Work

### Week 2 Practical (Prac-W2)
- **内容 | Content**: 数据挖掘基础概念和分析技术入门
- **格式 | Format**: LaTeX 文档 (prac2.tex)

### Week 3 Practical (Prac-W3)
- **内容 | Content**: 决策树算法实现和分析
- **文件 | Files**: 
  - `dt.ipynb` - Decision Tree Jupyter Notebook
  - `prac3.tex` - LaTeX 报告

---

## 🎓 课程学习路径 | Course Learning Path

```
Week 1-2: 数据挖掘基础 | Data Mining Fundamentals
  └── 数据探索、预处理基础

Week 3-4: 分类算法 | Classification Algorithms
  └── Decision Tree, Naïve Bayes

Week 5-6: 集成学习 | Ensemble Learning
  └── Random Forest, Bagging, Boosting

Week 7-8: 模型评估与优化 | Model Evaluation & Optimization
  └── Cross-Validation, Hyperparameter Tuning

Week 9-11: 项目实施 | Project Implementation
  └── 数据分析 → 模型开发 → 性能优化 → 提交
```

---

## 🔗 相关资源 | Related Resources

### 项目仓库 | Project Repository
- GitHub: [vvkkfwq/INFS7203-DM](https://github.com/vvkkfwq/INFS7203-DM)

### 关键文档 | Key Documents
- `README.md` - 项目总览和运行说明
- `docs/Project_Requirements.md` - 项目需求详细说明（中文）
- `docs/Task_Checklist.md` - 任务清单和实验记录
- `DM-Assignment/README.md` - 最终模型配置和结果

---

## ✨ 总结 | Summary

这个项目展示了我在数据科学和机器学习领域的综合能力，从数据分析到模型部署的完整流程。通过系统化的方法和持续的实验优化，我成功达到了项目目标，并积累了宝贵的实践经验。

This project demonstrates my comprehensive capabilities in data science and machine learning, covering the complete workflow from data analysis to model deployment. Through systematic methods and continuous experimental optimization, I successfully achieved the project goals and gained valuable practical experience.

---

_最后更新 | Last Updated: 2026-01-30_
