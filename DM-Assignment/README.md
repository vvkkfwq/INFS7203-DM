# INFS7203 Track1 Project - Binary Classification

> Student: s4860387
> Course: INFS7203 Data Mining  
> Semester: 2, 2025

---

## 项目概述

本项目针对二分类问题，使用课程 Week 2-8 所学的数据挖掘技术（Decision Tree, Random Forest, k-NN, Naïve Bayes），训练分类器对测试数据进行预测。

**数据集**:

- 训练集: 10,853 样本 × 43 特征 (25 数值 + 18 类别)
- 测试集: 2,713 样本 × 43 特征

**评估指标**: F1 Score (标签"1"为正类)

**目标**: 最大化测试集 F1 Score (≥0.65 满分)

---

## 环境要求

### 操作系统

```
[待填写]
例如: Ubuntu 22.04 LTS / Windows 11 / macOS Ventura
```

### Python 版本

```
[待填写]
例如: Python 3.10.12
```

### 依赖包

```
[待填写]
例如:
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

---

## 安装步骤

### 1. 创建虚拟环境 (推荐)

```bash
conda create -n dm python=3.10
conda activate dm
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 项目结构

```
[待完善 - 代码开发后填写]

sXXXXXXX/
├── README.md
├── main.py
├── train.csv
├── test_data.csv
└── ...
```

---

## 运行步骤

### 完整流程

```bash
python main.py
```

**预计运行时间**: [待测试后填写]

### 输出文件

- `sXXXXXXX.infs4203` - 提交用的结果文件

---

## 最终配置

### 数据预处理

**缺失值处理**:

```
[待完善 - 实验后确定]
```

**特征缩放**:

```
[待完善 - 实验后确定]
```

**类别编码**:

```
[待完善 - 实验后确定]
```

**其他处理**:

```
[待完善 - 如有需要]
```

---

### 分类模型

**主模型**:

```
[待完善 - 调优后确定]
例如: Random Forest / Decision Tree / k-NN / Ensemble
```

**模型参数**:

```
[待完善 - 调优后填写]
```

**集成策略** (如使用):

```
[待完善 - 如使用集成方法]
```

**交叉验证配置**:

```
折数: 5
策略: StratifiedKFold
随机种子: 42
```

---

## 方法选择理由

### 预处理方案选择

**缺失值处理**:

```
[待填写 - 说明为什么选择这个方法，对比了哪些方案]
```

**特征缩放**:

```
[待填写 - 说明选择理由]
```

**类别编码**:

```
[待填写 - 说明选择理由]
```

---

### 模型选择理由

```
[待填写 - 详细说明为什么选择这个模型]

包括:
1. 在基线测试中的表现
2. 模型的优势特点
3. 适合数据集的原因
4. 与其他模型的对比
```

---

### 超参数调优过程

```
[待填写 - 说明调优的过程和发现]

包括:
1. 搜索的参数空间
2. 使用的搜索方法 (GridSearch/RandomSearch)
3. 关键参数的影响
4. 最终参数的选择依据
```

---

### 集成策略理由 (如使用)

```
[待填写 - 如使用集成方法，说明理由]

包括:
1. 为什么需要集成
2. 选择哪些模型组合
3. 集成方法的选择 (Voting/Stacking等)
4. 性能提升情况
```

---

## 实验结果

### 基线模型对比

| Model         | Accuracy        | F1 Score        | Notes                                |
| ------------- | --------------- | --------------- | ------------------------------------ |
| Random Forest | 0.8331 ± 0.0069 | 0.5742 ± 0.0226 | Best baseline, high tuning potential |
| Decision Tree | 0.7650 ± 0.0067 | 0.5344 ± 0.0185 | Good baseline, ready for tuning      |
| Naïve Bayes   | 0.7915 ± 0.0038 | 0.3997 ± 0.0233 | Moderate performance                 |
| k-NN          | 0.7159 ± 0.0068 | 0.2003 ± 0.0075 | Poor, needs StandardScaler           |

### 最终模型性能 (交叉验证)

```
模型: [待填写]
Accuracy: [待填写] ± [标准差]
F1 Score: [待填写] ± [标准差]
Precision: [待填写] ± [标准差]
Recall: [待填写] ± [标准差]
```

---

## 可复现性说明

### 随机种子

所有随机操作使用固定种子: `RANDOM_SEED = 42`

### 数据来源

- 训练集: train.csv (课程提供)
- 测试集: test_data.csv (课程提供)

### 环境信息

详见"环境要求"部分

---

## AI 工具使用声明

```
[如使用AI工具辅助，在此说明]

例如:
- 使用Claude/ChatGPT辅助代码调试
- 使用AI生成可视化代码
- 所有核心算法和决策基于独立实验
```

---

## 注意事项

### 技术合规

✅ 仅使用 Week 2-8 课程技术  
✅ 固定随机种子确保可复现  
✅ 代码为.py 格式 (非.ipynb)

### 文件检查

✅ train.csv 和 test_data.csv 已包含  
✅ 生成的 sXXXXXXX.infs4203 格式正确  
✅ requirements.txt 完整

---

## 联系方式

**课程邮箱**: infs4203@eecs.uq.edu.au

---

**最后更新**: [填写日期]
