# 作业回忆速查表 | Assignment Quick Recall Sheet

> 快速回忆项目内容，5分钟准备面试
> Quick recall for 5-minute interview prep

---

## 🎓 基本信息

**课程**：INFS7203 数据挖掘  
**学校**：昆士兰大学 (University of Queensland)  
**时间**：2025年9-10月  
**成绩**：满分 (F1 Score 0.650)

---

## 💡 一句话总结

"我独立完成了一个包含10,853条数据、43个特征的二分类机器学习项目，通过系统化的数据预处理、模型对比、超参数调优和集成学习，最终达到F1 Score 0.650的满分标准。"

---

## 🎯 项目三大亮点

### 1. 系统化方法论
- ✅ 不是凭直觉，而是通过对比实验做决策
- ✅ 测试了4种缺失值处理方案，选择最优
- ✅ 对比了4种机器学习算法
- ✅ Grid Search 超参数调优

### 2. 问题解决能力
- ✅ 识别并解决类别不平衡问题 (75:25)
- ✅ 有效处理缺失值 (NaN)
- ✅ 通过集成学习突破性能瓶颈
- ✅ 从0.566提升到0.650 (+14.8%)

### 3. 工程化能力
- ✅ 模块化代码设计 (OOP)
- ✅ 完整的文档和注释
- ✅ Git版本控制
- ✅ 完全可复现

---

## 📊 关键数字（必须记住！）

```
数据规模：
- 训练集：10,853 条
- 测试集：2,713 条  
- 特征数：43 个 (25数值 + 18类别)
- 类别比：75:25 (不平衡)

性能指标：
- 起点：F1 = 0.5664 (Random Forest基线)
- 终点：F1 = 0.650 (集成模型)
- 提升：+14.8%
- 准确率：79.5%

技术栈：
- Python 3.10
- scikit-learn 1.7
- pandas 2.3
- numpy 2.2
```

---

## 🛠️ 用到的技术

### 机器学习算法
1. **Random Forest** ⭐ 主要模型
2. **Decision Tree** ⭐ 主要模型  
3. k-Nearest Neighbors (辅助实验)
4. Naïve Bayes (辅助实验)

### 数据预处理
- 缺失值填补：Mean/Mode
- 异常值检测：LOF (移除109个样本)
- 类别编码：OrdinalEncoder
- 特征标准化：StandardScaler (仅用于LOF)

### 模型优化
- 超参数调优：Grid Search
- 交叉验证：5-fold StratifiedKFold
- 类别权重：class_weight='balanced'
- 集成学习：Hard Voting

---

## 🚀 项目流程（5步）

```
第1步：数据探索 (EDA)
↓ 发现：类别不平衡、缺失值、异常值

第2步：数据预处理
↓ 处理：填补缺失值、检测异常、编码类别

第3步：基线模型对比
↓ 结果：RF和DT表现最好

第4步：超参数调优
↓ 提升：RF F1从0.566→0.647

第5步：模型集成
↓ 达标：F1=0.650 ✅
```

---

## 💬 面试时这样说

### 中文版（1分钟）

"在数据挖掘课程中，我完成了一个二分类项目。数据有1万多条记录，43个特征，存在类别不平衡和缺失值。

我的方法分四步：
第一，数据探索，发现问题；
第二，设计预处理流程，用均值众数填补缺失值，用LOF检测异常值；
第三，对比测试4种算法，发现Random Forest和Decision Tree最好；
第四，通过Grid Search调优参数，再用Hard Voting集成两个模型。

最终F1 Score达到0.650，获得满分。整个项目代码模块化设计，完全可复现。"

### English Version (1 minute)

"In my Data Mining course, I completed a binary classification project with over 10,000 records and 43 features, facing class imbalance and missing values.

My approach had four steps:
First, exploratory analysis to identify issues;
Second, designed preprocessing pipeline with mean/mode imputation and LOF outlier detection;
Third, compared 4 algorithms and found Random Forest and Decision Tree performed best;
Fourth, tuned hyperparameters via Grid Search and ensembled models with Hard Voting.

Final F1 Score reached 0.650, earning full marks. The project code is modularly designed and fully reproducible."

---

## 🤔 面试常见问题速答

### Q: 遇到的最大挑战？
**A**: 类别不平衡问题。我通过添加class_weight='balanced'参数解决，使模型对少数类更敏感，F1提升了约7%。

### Q: 为什么用集成学习？
**A**: 单模型调优后接近但未达标(0.647)。我发现RF和DT在某些样本上预测不同，具有互补性，通过Hard Voting组合后达到0.650。

### Q: 如何防止过拟合？
**A**: 三方面：1) 用5折交叉验证；2) 限制树深度(max_depth=10)和分裂样本数；3) 在独立测试集验证。训练F1约0.79，测试0.65，差距合理。

### Q: 如何保证可复现？
**A**: 1) 固定所有随机种子为42；2) 记录完整环境(requirements.txt)；3) 详细README文档；4) 模块化代码设计。

---

## 📁 项目结构速记

```
DM-Assignment/
├── src/           # 源代码(main.py + 模型 + 预处理 + 工具)
├── scripts/       # 实验脚本(基线/调优/集成)
├── notebooks/     # 数据探索笔记本
├── data/          # 训练和测试数据
├── results/       # 实验结果
├── docs/          # 项目文档
└── README.md      # 完整说明
```

---

## ⭐ 展示你的优势

### 技术能力
- ✅ 完整的ML项目经验
- ✅ 熟练使用scikit-learn
- ✅ 掌握数据预处理技巧
- ✅ 理解模型调优方法

### 软技能  
- ✅ 系统化思维
- ✅ 问题分析能力
- ✅ 代码工程能力
- ✅ 文档编写能力

### 学习态度
- ✅ 注重实验对比
- ✅ 追求代码质量
- ✅ 完善的文档习惯
- ✅ 持续优化精神

---

## 🔗 查看详细文档

如需更详细的信息，请查看：

- **[ASSIGNMENT_SUMMARY.md](./ASSIGNMENT_SUMMARY.md)** - 完整项目总结
- **[INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md)** - 详细面试指南
- **[TECHNICAL_REFERENCE.md](./TECHNICAL_REFERENCE.md)** - 技术细节速查

---

## ✅ 面试前5分钟检查

```
□ 记住关键数字 (0.650, 10,853, 43, 14.8%)
□ 能流利说出1分钟介绍
□ 准备好GitHub链接
□ 能解释技术选择原因
□ 能讨论遇到的挑战
□ 自信且真诚
```

---

**重要提示**：
- 不要背诵，用自己的话说
- 强调思维过程，不只是结果
- 准备好回答"为什么"
- 诚实面对不懂的问题

---

**祝你面试成功！加油！💪**

_创建时间：2026-01-30_
