# INFS7203 Data Mining Assignment

## Project Overview

This project implements binary classification on a real-world mixed dataset containing both numerical and categorical features.

## Project Structure

```
DM-Assignment/
├── data/
│   ├── train.csv                    # 训练数据 (10,853样本)
│   └── test_data.csv               # 测试数据 (2,713样本)
├── src/
│   ├── main.py                     # 主程序入口
│   ├── preprocessing.py            # 数据预处理模块
│   ├── models.py                   # 模型定义和训练
│   ├── evaluation.py               # 交叉验证和评估
│   └── utils.py                    # 工具函数
├── experiments/
│   ├── hyperparameter_tuning.py   # 超参数调优
│   └── model_selection.py         # 模型选择实验
├── results/
│   └── sXXXXXXX.infs7203          # 最终结果报告
├── README.md                       # 项目文档
└── requirements.txt               # 环境依赖
```

## Dataset Description

- **Training data**: `data/train.csv` (~10,854 samples)
- **Test data**: `data/test_data.csv` (~2,714 samples)
- **Features**: 43 total features
  - 25 numerical features (Num_Col1 - Num_Col25)
  - 18 categorical features (Nom_Col26 - Nom_Col43)
- **Target**: Binary classification (0/1)
- **Missing values**: Present throughout the dataset

## Objective

Develop and evaluate machine learning models for binary classification on this mixed-type dataset, handling missing values and categorical encoding appropriately.

## Data Characteristics

The dataset represents real-world challenges including:

- Mixed data types (numerical and categorical)
- Missing values requiring preprocessing
- Imbalanced or varied feature distributions
- Need for proper train/test evaluation
