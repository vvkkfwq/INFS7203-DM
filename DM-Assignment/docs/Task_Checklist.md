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
- [x] 设计异常检测处理方案
  - LOF
  - IsolationFores
- [x] 设计类别编码方案
  - OrdinalEncoder ✅
  - TargetEncoding
- [x] 实现完整的 pipeline 作预处理对比实验
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
  - [ ] 设计 2-3 种特征缩放方案（只需在 knn 中实现）
  - StandardScaler ✅
  - MinMaxScaler
  - RobustScaler
- [x] 训练 Naïve Bayes (默认参数)
- [x] 5 折交叉验证评估
- [x] 记录每个模型的 Accuracy 和 F1
- [x] 创建性能对比表

**输出**: 4 个基线模型 + 性能对比表

### Phase 4: 超参数调优

- [x] 选出最有潜力的 2-3 个模型
- [x] 为 Random Forest 定义参数搜索空间
  - n_estimators: [100, 200, 300]
  - max_depth: [10, 20, None]
  - min_samples_split: [2, 5, 10]
  - max_features: ["sqrt", "log2", None]
  - min_samples_leaf: [1, 2, 4]
- [x] 运行 GridSearchCV
- [x] 记录最佳参数和 F1 提升
  - **最佳参数**: max_depth=10, n_estimators=100, min_samples_leaf=2, min_samples_split=2, max_features=None
  - **性能提升**: F1 Score: 0.5742 → 0.5995 (+0.0253, +4.4%)
  - **当前与目标差距**: 0.65 - 0.5995 = 0.0505 (还需提升 5 个百分点)
- [x] 为 Decision Tree 定义参数搜索空间
  - criterion: ['gini', 'entropy']
  - splitter: ['best', 'random']
  - max_depth: [10, 15, 20, 25, None]
  - min_samples_split: [2, 5, 10, 20]
  - min_samples_leaf: [1, 2, 4, 8]
  - max_features: ['sqrt', 'log2', None]
- [x] 运行 GridSearchCV
- [x] 记录最佳参数和 F1 提升
  - **最佳参数**: criterion='gini', splitter='best', max_depth=10, min_samples_split=20, min_samples_leaf=4, max_features=None
  - **性能提升**: F1 Score: 0.5344 → 0.6039 (+0.0695, +13.0%) 🏆
  - **当前与目标差距**: 0.65 - 0.6039 = 0.0461 (还需提升 4.6 个百分点)
- [x] 为 k-NN 定义参数搜索空间
  - n_neighbors: [3, 5, 7, 9, 11, 13, 15]
  - weights: ['uniform', 'distance']
  - metric: ['euclidean', 'manhattan']
  - 需要添加 StandardScaler 预处理
- [x] 运行 GridSearchCV
- [x] 记录最佳参数和 F1 提升
  - **最佳参数**: metric='manhattan', n_neighbors=3, weights='uniform'
  - **性能提升**: F1 Score: 0.2003 → 0.4320 (+0.2317, +115.7%) 🎉 StandardScaler 效果显著
  - **当前与目标差距**: 0.65 - 0.4320 = 0.2180 (远不如树模型,不适合作为最终模型)

---

## Week 3: 深度调优和集成 (15-20 小时)

### Phase 4: 超参数调优 (继续)

- [x] 尝试调整 Decision Tree 网络搜索的 range
- [x] 尝试调整 RF 网络搜索的 range

### Phase 5: 进阶优化策略

#### 子阶段 5.1: 类别不平衡处理 (优先级最高 🔥)

- [x] Decision Tree 添加 `class_weight='balanced'` 参数
  - F1: 0.6092 ± 0.0143
- [x] Random Forest 测试 `class_weight='balanced'`
  - F1: 0.6419 ± 0.0119
- [ ] 对比 class_weight 前后的 Precision/Recall/F1 变化
- [x] 记录最佳 class_weight 配置

**输出**: class_weight 实验结果 + 性能提升报告

#### 子阶段 5.2: 集成方法 (如果 5.1 未达标)

- [x] 尝试 Hard Voting Classifier
  - 组合: Decision Tree (tuned) + Random Forest (tuned)
- [x] 尝试 Soft Voting Classifier
  - 需要支持 predict_proba 的模型
- [x] 实验不同集成权重
- [x] 对比单模型 vs 集成性能
- [x] 决定最终模型 (单模型或集成)

**输出**: 最终模型选择 + 性能对比

#### 子阶段 5.3: 预处理-分类器协同优化 (备选)

- [ ] 实验 Decision Tree + StandardScaler 组合
- [ ] 实验 Random Forest + MinMaxScaler 组合
- [ ] 测试 OneHotEncoder vs OrdinalEncoder 对树模型的影响
- [ ] 为 Naïve Bayes 专门设计预处理策略
  - 尝试特征离散化 (KBinsDiscretizer)
  - 测试不同编码方式

**输出**: 预处理-模型最佳组合表

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

**最低目标**: F1 ≥ 0.60 (10 分 - 及格) ✅ 已达成 (0.6039)
**良好目标**: F1 ≥ 0.63 (16 分 - 优秀)
**理想目标**: F1 ≥ 0.65 (20 分 - 满分) ⏳ 差距 0.0461 (4.6%)

---

## 🚀 后续优化路线图

基于当前进度 (Decision Tree F1=0.6039) 和课程建议，按优先级排序：

### 路线 A: 快速达标路线 (推荐 ⭐⭐⭐)

1. **类别不平衡处理** (子阶段 5.1)

   - 实现难度: ⭐☆☆☆☆ (仅需修改一个参数)
   - 预期收益: ⭐⭐⭐⭐☆ (F1 +0.02~0.05)
   - 时间成本: 30 分钟
   - **关键理由**: F1 Score 对类别不平衡非常敏感,且数据集有明显的 75:25 不平衡

2. **如果未达标**: Voting Classifier (子阶段 5.2)
   - 实现难度: ⭐⭐☆☆☆
   - 预期收益: ⭐⭐⭐☆☆ (F1 +0.01~0.03)
   - 时间成本: 1-2 小时

### 路线 B: 深度探索路线 (学习导向)

1. **预处理-分类器协同实验** (子阶段 5.3)
   - 探索不同预处理对各模型的影响
   - 可能发现 Naïve Bayes 的最佳配置
   - 时间成本: 3-4 小时
   - 适合: 时间充裕且想深入理解算法的同学

### 路线 C: 保守稳妥路线

如果当前 F1=0.6039 已满意:

- 直接进入 Phase 6 (最终验证和提交)
- 风险: 可能无法获得满分 (20 分)
- 优势: 节省时间,确保稳定提交

### 💡 推荐决策流程

```
开始
  ↓
尝试 class_weight='balanced' (30分钟)
  ↓
F1 ≥ 0.65?
  ├─ 是 → 进入 Phase 6 提交 🎉
  └─ 否 → F1 ≥ 0.63?
           ├─ 是 → 考虑是否尝试 Voting (追求满分)
           └─ 否 → 必须尝试 Voting + 预处理优化
```

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
