# INFS4203 Track1 项目任务清单

> 按顺序完成，每完成一项就打勾 ✅

---

## 📅 时间线

**总时间**: 4-5 周
**建议提交**: 2025-10-18
**最终截止**: 2025-10-20 13:00

---

## Week 1: 数据探索和预处理 (10-15 小时)

### Phase 1: 探索性数据分析 (EDA)

- [x] 下载并加载 train.csv 和 test_data.csv
- [x] 检查数据形状和基本信息
- [x] 分析标签分布 (是否平衡)
- [x] 统计缺失值比例
- [x] 可视化数值特征分布
- [x] 分析类别特征的唯一值数量
- [x] 检查异常值
- [x] 分析特征与标签的关系
- [x] 撰写 EDA 报告

**输出**: EDA 报告 + 可视化图表

### Phase 2: 预处理实验

- [x] 设计 4 种缺失值处理方案
  - 方案 1: 全局数值 mean/类别 mode
  - 方案 2: 全局数值 median/类别 mode
  - 方案 3: 全局 Constant Filling distinct special value
  - 方案 4: Class-Specific Imputation with mean/mode
- [x] 设计类别编码方案
  - OrdinalEncoder ✅
  - One-Hot Encoding
  - Label Encoding
- [ ] 设计 2-3 种特征缩放方案（树模型暂不需要）
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
- [x] 使用交叉验证对比各方案
- [x] 记录每个方案的 CV F1 分数
- [x] 选择最佳预处理组合

**输出**: 最佳预处理方案 + 实验对比表

---

## Week 2: 基线模型和初步调优 (15-20 小时)

### Phase 3: 基线模型训练

- [x] 创建项目代码结构
- [x] 实现预处理管道
- [x] 训练 Decision Tree (默认参数)
- [x] 训练 Random Forest (默认参数)
- [x] 训练 k-NN (默认参数)
- [x] 训练 Naïve Bayes (默认参数)
- [ ] 5 折交叉验证评估
- [ ] 记录每个模型的 Accuracy 和 F1
- [ ] 创建性能对比表

**输出**: 4 个基线模型 + 性能对比表

### Phase 4: 超参数调优 (开始)

- [ ] 选出最有潜力的 2-3 个模型
- [ ] 为 Random Forest 定义参数搜索空间
  - n_estimators: [100, 200, 300, 400]
  - max_depth: [10, 15, 20, 25, None]
  - min_samples_split: [2, 5, 10, 15]
- [ ] 运行 GridSearchCV
- [ ] 记录最佳参数和 F1 提升

**输出**: 部分模型的最佳参数

---

## Week 3: 深度调优和集成 (15-20 小时)

### Phase 4: 超参数调优 (继续)

- [ ] 为 k-NN 定义参数搜索空间
  - n_neighbors: [3, 5, 7, 9, 11, 13, 15]
  - weights: ['uniform', 'distance']
  - metric: ['euclidean', 'manhattan']
- [ ] 运行 GridSearchCV
- [ ] 为 Decision Tree 调优 (如需要)
- [ ] 比较所有调优后的模型
- [ ] 选出 F1 最高的模型

**输出**: 所有模型的最佳参数 + 性能排名

### Phase 5: 集成方法

- [ ] 尝试 Hard Voting Classifier
- [ ] 尝试 Soft Voting Classifier
- [ ] 优化集成权重
- [ ] 对比单模型 vs 集成性能
- [ ] 决定最终模型 (单模型或集成)

**输出**: 最终模型选择 + 性能对比

---

## Week 4: 最终验证和文档 (10-15 小时)

### Phase 6: 最终验证

- [ ] 固定所有随机种子 (RANDOM_SEED=42)
- [ ] 在完整训练集上训练最终模型
- [ ] 5 折交叉验证获得 CV 分数
- [ ] 在测试集上预测
- [ ] 生成结果文件 sXXXXXXX.infs4203
- [ ] 验证结果文件格式
  - [ ] 2714 行
  - [ ] 前 2713 行是 0 或 1
  - [ ] 每行末尾有逗号
  - [ ] 最后一行格式正确

**输出**: sXXXXXXX.infs4203 文件

### 代码整理

- [ ] 重构代码为模块化结构
- [ ] 添加清晰的注释和 docstring
- [ ] 确保 main.py 可以一键运行
- [ ] 测试代码在新环境的可复现性
- [ ] Python 用户: 转换所有.ipynb 为.py
- [ ] 创建 requirements.txt

**输出**: 整洁的代码包

### 文档编写

- [ ] 编写 README.md
  - [ ] 环境说明 (OS, Python 版本, 包版本)
  - [ ] 安装步骤
  - [ ] 运行步骤 (详细的一步步指南)
  - [ ] 最终配置 (预处理+模型+参数)
  - [ ] 方法选择理由
  - [ ] AI 工具使用声明 (如有)
- [ ] 创建实验日志文档 (可选但推荐)
- [ ] 准备展示材料 (为 presentation 准备)

**输出**: 完整的 README.md

### 打包和验证

- [ ] 将所有文件打包为 sXXXXXXX.zip
- [ ] 检查 ZIP 内容完整性
  - [ ] README.md
  - [ ] main.py
  - [ ] 所有代码文件
  - [ ] train.csv
  - [ ] test_data.csv
  - [ ] sXXXXXXX.infs4203
- [ ] 验证 ZIP 文件大小 <100MB
- [ ] 在新目录解压测试
- [ ] 运行代码确认可复现

**输出**: sXXXXXXX.zip 文件

---

## Week 5: 提交和缓冲 (5 小时)

### 提交前最终检查

- [ ] 运行提交检查清单 (见下方)
- [ ] 所有文件命名正确
- [ ] 文件格式完全符合要求
- [ ] 代码完全可复现

### 提交到 Blackboard

- [ ] 登录 Blackboard → Assessment → Project
- [ ] 提交 sXXXXXXX.infs4203
  - 链接: Report submission
  - Turnitin 标题: sXXXXXXX.infs4203
- [ ] 提交 sXXXXXXX.zip
  - 链接: Code submission
  - Turnitin 标题: sXXXXXXX.zip
- [ ] 确认两个文件都成功上传
- [ ] 截图保存提交确认

### 缓冲时间

- [ ] 处理任何意外问题
- [ ] 如需要则修正并重新提交

---

## 📋 提交前终极检查清单

### 文件命名

- [ ] 结果文件: sXXXXXXX.infs4203 (学号正确)
- [ ] 代码包: sXXXXXXX.zip (学号正确)
- [ ] 没有大写字母、额外字符或错误

### 结果文件格式

- [ ] 总共 2714 行
- [ ] Row 1-2713: 每行一个整数(0 或 1)+逗号
- [ ] Row 2714: accuracy,f1, (3 位小数+逗号)
- [ ] 运行验证脚本检查

### 代码质量

- [ ] 所有文件都是.py (不是.ipynb)
- [ ] 有 main.py 入口文件
- [ ] 所有随机种子固定为 42
- [ ] 代码有清晰注释
- [ ] 代码可以一键运行

### README 完整性

- [ ] 环境说明完整
- [ ] 安装步骤清晰
- [ ] 运行步骤详细
- [ ] 最终配置说明
- [ ] 方法选择理由

### ZIP 内容

- [ ] README.md ✓
- [ ] main.py ✓
- [ ] 所有代码文件 ✓
- [ ] train.csv ✓
- [ ] test_data.csv ✓
- [ ] sXXXXXXX.infs4203 ✓
- [ ] requirements.txt ✓
- [ ] 文件大小 <100MB ✓

### Turnitin 提交

- [ ] Report submission 标题与文件名一致
- [ ] Code submission 标题与文件名一致
- [ ] 两个文件都成功上传
- [ ] 提交时间在 deadline 前

### 技术合规

- [ ] 只使用了 Decision Tree, Random Forest, k-NN, Naïve Bayes
- [ ] 没有使用 XGBoost, Neural Networks 等禁止技术
- [ ] 所有技术都在 Week 2-8 范围内

### 可复现性

- [ ] 在新环境测试过代码
- [ ] 结果与报告一致
- [ ] 随机种子已固定
- [ ] 依赖包版本已记录

---

## 🎯 性能目标

**最低目标**: F1 ≥ 0.60 (10 分 - 及格)
**良好目标**: F1 ≥ 0.63 (16 分 - 优秀)
**理想目标**: F1 ≥ 0.65 (20 分 - 满分)

---

## 💡 关键提示

### 记录实验

每次实验都记录:

- 配置 (预处理+模型+参数)
- 结果 (Accuracy, F1 均值 ± 标准差)
- 观察和下一步计划

### 固定随机种子

所有地方都用同一个种子:

```python
RANDOM_SEED = 42
random.seed(42)
np.random.seed(42)
model = RandomForest(random_state=42)
cv = StratifiedKFold(..., random_state=42)
```

### 系统化而非随机

按逻辑顺序实验，每步基于前一步结果，不盲目尝试。

### 提前提交

建议 10 月 18 日完成，留 2 天缓冲应对突发情况。

---

## ✅ 完成标志

当所有任务都打勾后:

- 你有一个可复现的完整项目
- 你有符合格式的提交文件
- 你已提前提交到 Blackboard
- 你可以自信地等待成绩

**祝顺利完成项目！🎉**
