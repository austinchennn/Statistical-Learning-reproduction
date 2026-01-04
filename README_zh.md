# 统计学习模型复现项目

从零开始实现经典统计学习算法，配合真实数据集和交互式可视化。

**[English Version](./README.md)**

## 项目简介

本项目复现统计学习理论中的核心算法，包括决策树和梯度提升模型。每个实现都包含：

- **完整算法实现**：从零开始实现，不依赖 scikit-learn
- **真实数据验证**：使用银行贷款数据集进行测试
- **交互式可视化**：基于 Matplotlib 的树结构可视化
- **详细文档注释**：包含数学原理和代码解释

## 项目结构

```
.
├── DecisionTree.ipynb        # 决策树（ID3算法）
├── XGBoost.ipynb             # XGBoost 实现
├── data/
│   └── bankloan.csv          # 银行贷款数据集（5000样本，14特征）
└── README_zh.md              # 本文件
```

## 实现的模型

### 1. 决策树 (ID3算法)
**位置**: `DecisionTree.ipynb`

#### 核心概念
ID3（迭代二分器第3代）算法通过递归选择**信息增益**最大的特征来分割数据集，构建决策树。

#### 关键组件

**信息熵**（Shannon熵）：
$$H(D) = -\sum_{k} p_k \log_2 p_k$$

其中 $p_k$ 是数据集 $D$ 中第 $k$ 类的比例。

**信息增益**：
$$IG(D, A) = H(D) - \sum_{v} \frac{|D_v|}{|D|} H(D_v)$$

其中 $A$ 是特征，$D_v$ 是特征 $A$ 等于值 $v$ 的数据子集。

**算法步骤**：
1. 计算当前数据集的熵
2. 对每个特征计算信息增益
3. 选择信息增益最大的特征作为分割点
4. 递归划分数据集，构建子树
5. 停止条件：(a) 节点中所有样本属于同一类别，或 (b) 所有特征都已使用

#### 实现细节

**树的数据结构**：字典递归表示法
```python
# 示例树结构
{
    'Age': {
        0: {'Work': {0: 'No', 1: 'Yes'}},  # 青年 -> 是否工作的分支
        1: 'Yes',                           # 中年 -> 是
        2: 'Yes'                            # 老年 -> 是
    }
}
```

**核心函数**：
- `calcShannonEnt(dataSet)`: 计算数据集熵
- `chooseBestFeatureToSplit(dataSet)`: 使用信息增益找最优特征
- `creatTree(dataset, labels)`: 递归构建决策树
- `createPlot(myTree)`: Matplotlib 可视化

**数据集**：银行贷款数据集（移除ID后4387个样本）
- 特征：年龄、工作、房产、贷款额度、教育程度等（13个特征）
- 目标：贷款批准与否（是/否）

#### 可视化效果
基于 Matplotlib 的树形可视化：
- 蓝色节点：决策节点（特征分割点）
- 黄色节点：叶子节点（预测结果）
- 边上标签：特征值和决策路径

---

### 2. XGBoost 实现
**位置**: `XGBoost.ipynb`

*(详细介绍待补充 - 先进的梯度提升方法)*

---

## 数据格式

### CSV 转数据集
`csv_to_dataset()` 函数将 CSV 文件转换为所需格式：

```python
# 返回 (dataSet, labels)
# dataSet: [[特征1, 特征2, ..., 目标], ...]
# labels: ['特征1', '特征2', ..., '特征名']

dataSet, labels = csv_to_dataset(
    'data/bankloan.csv',
    exclude_cols=['ID']  # 移除ID列
)
```

---

## 数学原理

### 决策树理论

**停止条件**：
1. **纯节点**：所有样本属于同一类别
2. **特征用尽**：所有特征都已用于分割
3. **多数表决**：特征用完但样本仍混合时，选择多数类

**复杂度分析**：
- 时间：$O(n \cdot m \cdot \log m)$，其中 $n$ = 样本数，$m$ = 特征数
- 空间：$O(m \cdot d)$，其中 $d$ = 树深度

### 信息论基础

**熵**衡量集合中的混乱程度：
- $H = 0$：所有样本同一类（纯）
- $H = 1$：均匀分布（最大混乱）

**信息增益**量化特征减少不确定性的程度。

---

## 使用方法

### 运行决策树

```python
# 加载数据
dataSet_clean, labels_clean = csv_to_dataset(
    '/path/to/bankloan.csv',
    exclude_cols=['ID']
)

# 构建树
tree_clean = creatTree(dataSet_clean, labels_clean[:])

# 可视化
createPlot(tree_clean)

# 获取统计信息
print(f"叶子节点数: {getNumLeafs(tree_clean)}")
print(f"树的深度: {getTreeDepth(tree_clean)}")
```

### 在银行贷款数据集上的结果

```
🎯 决策树可视化（Matplotlib）
================================================================================
📊 Matplotlib方案统计: 叶子节点=4387, 深度=3
```

在清理后的数据集上，树的深度为 3，共有 4,387 个叶子节点。

---

## 依赖环境

```
pandas>=1.0.0
matplotlib>=3.0.0
numpy>=1.18.0
```

### Jupyter 环境要求
- Jupyter Notebook 或 JupyterLab
- Python 3.7+

---

## 安装与运行

### 1. 克隆仓库
```bash
git clone https://github.com/austinchennn/Statistical-Learning-reproduction.git
cd Statistical-Learning-reproduction
```

### 2. 安装依赖
```bash
pip install pandas matplotlib numpy jupyter
```

### 3. 运行 Notebook
```bash
jupyter notebook DecisionTree.ipynb
```

---

## 算法对比与扩展

### 当前实现：ID3算法
- **优点**：简单易懂、结果可解释、处理分类特征
- **缺点**：容易过拟合、贪心方法、不处理缺失值

### 潜在改进方向
1. **剪枝**：通过后剪枝或减错剪枝减少过拟合
2. **C4.5算法**：使用增益率处理连续特征
3. **CART算法**：二叉树形式，支持回归问题
4. **集成方法**：与 Bagging（随机森林）或 Boosting（XGBoost）结合

---

## 数据集说明

### 银行贷款数据集 (`data/bankloan.csv`)
- **样本数**：5,000 条（移除ID后 4,387 条）
- **特征数**：13 个特征（1个ID + 12个预测特征）
- **目标变量**：贷款批准状态（是/否）
- **特征类型**：包含分类和数值特征
- **类别分布**：均衡的二分类

---

## 关键代码解析

### 1. 信息熵计算

```python
def calcShannonEnt(dataSet) -> float:
    """
    计算数据集的信息熵
    dataSet: 数据集
    return: 信息熵
    """
    numexamples = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numexamples
        shannonEnt -= prob * log(prob, 2)  # 公式: -∑ p_k * log2(p_k)
    return shannonEnt
```

### 2. 选择最优特征

```python
def chooseBestFeatureToSplit(dataSet) -> int:
    """
    使用信息增益选择最优特征
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    
    bestInfoGain = 0.0
    bestFeature_index = -1
    
    for i in range(numFeatures):
        featList_value = [example[i] for example in dataSet]
        uniqueVals = set(featList_value)
        newEntropy = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy  # 信息增益
        
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature_index = i
    
    return bestFeature_index
```

### 3. 递归构建决策树

```python
def creatTree(dataset, labels, featureLabels=[]):
    """
    递归构建决策树
    """
    classList = [example[-1] for example in dataset]
    
    # 停止条件1：所有样本同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 停止条件2：所有特征已用
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    
    # 选择最优特征
    bestFeat_index = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat_index]
    
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat_index]
    
    # 递归构建子树
    featValues = [example[bestFeat_index] for example in dataset]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(
            splitDataSet(dataset, bestFeat_index, value),
            sublabels,
            featureLabels
        )
    
    return myTree
```

---

## 参考文献

1. Quinlan, J. R. (1986). "归纳决策树". Machine Learning, 1(1), 81-106.
2. Shannon, C. E. (1948). "通信的数学理论". Bell System Technical Journal.
3. Mitchell, T. M. (1997). "机器学习". McGraw-Hill.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "统计学习的要素".

---

## 项目目标

✅ **教育目的**：从第一原理理解决策树算法  
✅ **代码实现**：从零开始编写算法  
✅ **可视化**：创建可解释的视觉表现  
✅ **实践验证**：在真实数据集上验证  
🔄 **扩展发展**：添加高级变体（随机森林、梯度提升）

---

## 作者

Austin Chen

## 许可证

MIT 许可证 - 详见 LICENSE 文件

---

## 贡献指南

欢迎任何形式的贡献！您可以：
- 报告问题（Issue）
- 提出改进建议
- 提交拉取请求（Pull Request）
- 分享更多数据集

---

**最后更新时间**：2026年1月4日
