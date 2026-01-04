# Statistical Learning Reproduction

A comprehensive implementation of classic statistical learning models with real-world datasets and interactive visualizations.

**[中文版本](./README_zh.md)**

## Project Overview

This project reproduces core algorithms from statistical learning theory, including decision trees and gradient boosting models. Each implementation includes:

- **Complete Algorithm Implementation**: From scratch implementations without relying on scikit-learn
- **Real Dataset Testing**: Validated on actual datasets (Bank Loan Dataset)
- **Interactive Visualizations**: Matplotlib-based tree visualizations for interpretability
- **Detailed Documentation**: Mathematical explanations and code comments

## Project Structure

```
.
├── DecisionTree.ipynb        # Decision Tree (ID3 Algorithm)
├── XGBoost.ipynb             # XGBoost Implementation
├── data/
│   └── bankloan.csv          # Bank Loan Dataset (5000 samples, 14 features)
└── README.md                 # This file
```

## Models Implemented

### 1. Decision Tree (ID3 Algorithm)
**Location**: `DecisionTree.ipynb`

#### Core Concept
The ID3 (Iterative Dichotomiser 3) algorithm builds a decision tree by recursively selecting the feature with the highest **information gain** to split the dataset.

#### Key Components

**Information Entropy** (Shannon Entropy):
$$H(D) = -\sum_{k} p_k \log_2 p_k$$

Where $p_k$ is the proportion of class $k$ in dataset $D$.

**Information Gain**:
$$IG(D, A) = H(D) - \sum_{v} \frac{|D_v|}{|D|} H(D_v)$$

Where $A$ is a feature, $D_v$ is the subset of $D$ where feature $A$ equals value $v$.

**Algorithm Steps**:
1. Calculate entropy of the current dataset
2. For each feature, calculate information gain
3. Select the feature with maximum information gain as split point
4. Recursively partition data and build subtrees
5. Stop when: (a) all samples in a node belong to one class, or (b) all features are used

#### Implementation Details

**Tree Structure**: Dictionary-based recursive representation
```python
# Example tree structure
{
    'Age': {
        0: {'Work': {0: 'No', 1: 'Yes'}},  # Youth -> Work decision
        1: 'Yes',                           # Middle-age -> Yes
        2: 'Yes'                            # Senior -> Yes
    }
}
```

**Key Functions**:
- `calcShannonEnt(dataSet)`: Calculate dataset entropy
- `chooseBestFeatureToSplit(dataSet)`: Find optimal feature using information gain
- `creatTree(dataset, labels)`: Recursively build decision tree
- `createPlot(myTree)`: Matplotlib visualization

**Dataset**: Bank Loan Dataset (4,387 samples after ID removal)
- Features: Age, Work, Home, Loan amount, Education, etc. (13 features)
- Target: Loan approval (Yes/No)

#### Visualization
Matplotlib-based tree visualization with:
- Blue nodes: Decision nodes (feature splits)
- Yellow nodes: Leaf nodes (predictions)
- Edge labels: Feature values and decision paths

---

### 2. XGBoost Implementation
**Location**: `XGBoost.ipynb`

*(Details to be added - advanced gradient boosting method)*

---

## Data Format

### CSV to Dataset Conversion
The `csv_to_dataset()` function converts CSV files to the required format:

```python
# Returns (dataSet, labels)
# dataSet: [[feature1, feature2, ..., target], ...]
# labels: ['Feature1', 'Feature2', ..., 'FeatureName']

dataSet, labels = csv_to_dataset(
    'data/bankloan.csv',
    exclude_cols=['ID']  # Remove ID column
)
```

---

## Mathematical Background

### Decision Tree Theory

**Stopping Criteria**:
1. **Pure node**: All samples belong to same class
2. **No more features**: All features have been used for splitting
3. **Majority voting**: When features exhausted but mixed classes remain

**Complexity**:
- Time: $O(n \cdot m \cdot \log m)$ where $n$ = samples, $m$ = features
- Space: $O(m \cdot d)$ where $d$ = tree depth

### Information Theory

**Entropy** measures disorder in a set:
- $H = 0$: All samples same class (pure)
- $H = 1$: Equal distribution (maximum disorder)

**Information Gain** quantifies how much a feature reduces uncertainty.

---

## Usage

### Running Decision Tree

```python
# Load data
dataSet_clean, labels_clean = csv_to_dataset(
    '/path/to/bankloan.csv',
    exclude_cols=['ID']
)

# Build tree
tree_clean = creatTree(dataSet_clean, labels_clean[:])

# Visualize
createPlot(tree_clean)

# Get statistics
print(f"Leaves: {getNumLeafs(tree_clean)}")
print(f"Depth: {getTreeDepth(tree_clean)}")
```

### Results on Bank Loan Dataset

```
🎯 Decision Tree Visualization (Matplotlib)
================================================================================
📊 Matplotlib Statistics: Leaves=4387, Depth=3
```

The tree achieves a depth of 3 with 4,387 leaf nodes on the cleaned dataset.

---

## Requirements

```
pandas>=1.0.0
matplotlib>=3.0.0
numpy>=1.18.0
```

### For Jupyter Notebook
- Jupyter Notebook or JupyterLab
- Python 3.7+

---

## Installation & Execution

### 1. Clone Repository
```bash
git clone https://github.com/austinchennn/Statistical-Learning-reproduction.git
cd Statistical-Learning-reproduction
```

### 2. Install Dependencies
```bash
pip install pandas matplotlib numpy jupyter
```

### 3. Run Notebooks
```bash
jupyter notebook DecisionTree.ipynb
```

---

## Algorithm Comparison & Extensions

### Current Implementation: ID3
- **Pros**: Simple, interpretable, handles categorical data
- **Cons**: Tends to overfit, greedy approach, doesn't handle missing values

### Potential Improvements
1. **Pruning**: Reduce overfitting through post-pruning or reduced error pruning
2. **C4.5 Algorithm**: Handle continuous features with gain ratio
3. **CART Algorithm**: Binary trees, support for regression
4. **Ensemble Methods**: Combine with bagging (Random Forest) or boosting (XGBoost)

---

## Dataset Information

### Bank Loan Dataset (`data/bankloan.csv`)
- **Samples**: 5,000 (4,387 after ID removal)
- **Features**: 13 features (1 ID + 12 predictive features)
- **Target**: Loan approval status (Yes/No)
- **Feature Types**: Categorical and numerical
- **Class Distribution**: Balanced binary classification

---

## References

1. Quinlan, J. R. (1986). "Induction of Decision Trees". Machine Learning, 1(1), 81-106.
2. Shannon, C. E. (1948). "A Mathematical Theory of Communication". Bell System Technical Journal.
3. Mitchell, T. M. (1997). "Machine Learning". McGraw-Hill.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning".

---

## Project Goals

✅ **Educational**: Understand decision tree algorithms from first principles  
✅ **Implementation**: Code algorithms from scratch  
✅ **Visualization**: Create interpretable, visual representations  
✅ **Real-world Testing**: Validate on actual datasets  
🔄 **Extension**: Add advanced variants (Random Forest, Gradient Boosting)

---

## Author

Austin Chen

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share additional datasets

---

**Last Updated**: January 4, 2026
