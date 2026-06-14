# Classical ML Algorithms — Comprehensive Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland  
**Focus:** Data Scientist / ML Engineer — Risk Scoring, Time-Series Forecasting, Anomaly Detection, Production ML  
**Key Models Built:** LSTM + Ensemble Risk Scoring, Income Estimation, Isolation Forest Anomaly Detection, XGBoost/Random Forest in Production  
**Document Scope:** End-to-end coverage of classical ML algorithms, theory, math, code, and interview strategy

---

## Table of Contents

1. [Supervised Learning — Classification](#1-supervised-learning--classification)
2. [Supervised Learning — Regression](#2-supervised-learning--regression)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Anomaly Detection](#4-anomaly-detection)
5. [Ensemble Methods](#5-ensemble-methods)
6. [Model Selection and Validation](#6-model-selection-and-validation)
7. [Feature Engineering](#7-feature-engineering)
8. [Common Interview Questions (10+)](#8-common-interview-questions)
9. [Key Takeaways](#9-key-takeaways)

---

# 1. Supervised Learning — Classification

---

## 1.1 Logistic Regression

### Core Idea

Despite the name, **Logistic Regression** is a **classification** algorithm. It models the probability that an input belongs to the positive class by applying a sigmoid function to a linear combination of features.

### Mathematical Formulation

**Linear part (logit):**

$$
z = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

**Sigmoid (logistic) function:**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Predicted probability:**

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

**Decision rule:**

$$
\hat{y} = \begin{cases} 1 & \text{if } P(y=1 \mid \mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

### Sigmoid Properties

| Property | Value |
|----------|-------|
| Range | (0, 1) |
| $\sigma(0)$ | 0.5 |
| Monotonic | Yes — always increasing |
| Derivative | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| Saturation | Gradients vanish as $\|z\| \to \infty$ |

### Decision Boundary

The decision boundary is where $P(y=1) = 0.5$, i.e., $\mathbf{w}^T \mathbf{x} + b = 0$. For 2D features, this is a straight line. For higher dimensions, it is a hyperplane. Logistic regression can only learn **linear** decision boundaries.

### Loss Function — Binary Cross-Entropy

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

- When $y=1$: penalizes low $\hat{p}$
- When $y=0$: penalizes high $\hat{p}$
- Convex function — guarantees a global minimum via gradient descent

### Regularization

| Type | Penalty Term | Effect |
|------|-------------|--------|
| **L1 (Lasso)** | $\lambda \sum \|w_j\|$ | Sparse weights — drives coefficients to exactly zero (feature selection) |
| **L2 (Ridge)** | $\lambda \sum w_j^2$ | Shrinks weights — prevents any single feature from dominating |
| **Elastic Net** | $\alpha \lambda \sum \|w_j\| + \frac{(1-\alpha)}{2} \lambda \sum w_j^2$ | Combines L1 and L2 — both sparsity and regularization |

In scikit-learn, `LogisticRegression(C=1.0, penalty='l2')` — **C is the inverse regularization strength**: smaller C = stronger regularization.

### Multiclass Extension

- **One-vs-Rest (OvR):** Train K binary classifiers, each separating one class from all others.
- **Softmax (Multinomial):** Generalize sigmoid to K classes:

$$
P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}
$$

### When to Use Logistic Regression

- **Baseline model** — always start here for classification
- When you need **interpretable** coefficients (odds ratios)
- **Linearly separable** or near-linearly separable data
- **Low-dimensional** data or when you want fast training
- Regulatory environments requiring model explainability (finance, healthcare)

### Code Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000))
])

pipe.fit(X_train, y_train)
probs = pipe.predict_proba(X_test)[:, 1]
preds = pipe.predict(X_test)

# Interpret coefficients
import numpy as np
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': pipe.named_steps['clf'].coef_[0],
    'odds_ratio': np.exp(pipe.named_steps['clf'].coef_[0])
}).sort_values('coefficient', ascending=False)
```

---

## 1.2 Decision Trees

### Core Idea

A **Decision Tree** recursively partitions the feature space into regions using axis-aligned splits. Each internal node tests a feature threshold; each leaf holds a prediction. It is a **non-parametric**, **interpretable** model.

### Splitting Criteria

#### Gini Impurity

$$
\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

- Measures the probability of misclassifying a randomly chosen element
- Range: 0 (pure node) to $1 - 1/K$ (uniform distribution)
- Preferred by CART (Classification and Regression Trees) — default in scikit-learn

#### Entropy (Information Gain)

$$
\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

- Measures the information content / uncertainty in the node
- Range: 0 (pure node) to $\log_2(K)$ (uniform)

#### Information Gain

$$
\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
$$

- Used by ID3 and C4.5 algorithms
- Selects the feature-threshold combination that maximizes IG (or equivalently, maximizes impurity reduction)

#### Gini vs Entropy — Practical Comparison

| Aspect | Gini | Entropy |
|--------|------|---------|
| Computation | Faster (no log) | Slightly slower |
| Tendency | Isolates largest class in its own branch | Produces more balanced splits |
| Performance | Nearly identical in practice | Nearly identical in practice |
| Default in sklearn | Yes (`criterion='gini'`) | No (`criterion='entropy'`) |

### Splitting Process (CART Algorithm)

```
For each node:
    For each feature f:
        For each unique threshold t:
            Split data into left (f <= t) and right (f > t)
            Calculate weighted impurity of children
    Choose (f*, t*) that minimizes weighted impurity
    Create left child and right child
    Recurse until stopping criteria met
```

### Pruning

Overfitting is the main weakness of decision trees. Two strategies:

| Strategy | How | Parameters |
|----------|-----|------------|
| **Pre-pruning (early stopping)** | Stop growing before full depth | `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` |
| **Post-pruning (cost-complexity)** | Grow full tree, then remove branches | `ccp_alpha` (higher = more pruning) |

**Cost-complexity pruning** minimizes:

$$
R_\alpha(T) = R(T) + \alpha \cdot |T|
$$

where $R(T)$ is the total misclassification rate, $|T|$ is the number of leaf nodes, and $\alpha$ is the complexity parameter.

### Pros and Cons

| Pros | Cons |
|------|------|
| Highly interpretable — visualize tree | High variance — small data changes = different tree |
| No feature scaling needed | Prone to overfitting (deep trees memorize noise) |
| Handles mixed data types | Cannot learn diagonal / non-axis-aligned boundaries |
| Captures non-linear relationships | Greedy splitting — no global optimum guarantee |
| Fast training and inference | Unstable — high sensitivity to training data |

### Code Example

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    criterion='gini',
    random_state=42
)
dt.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=['Low', 'High'],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree — Risk Classification")
plt.show()
```

---

## 1.3 Random Forest

### Core Idea

**Random Forest** is a **bagging ensemble** of decision trees. It reduces variance by averaging predictions from many decorrelated trees, each trained on a bootstrapped subset of data with a random subset of features at each split.

### Algorithm

```
1. Draw B bootstrap samples from training data (sample with replacement)
2. For each bootstrap sample b = 1, ..., B:
   a. Grow a decision tree T_b:
      - At each node, randomly select m features from all p features
      - Find the best split among those m features
      - Grow tree to maximum depth (no pruning by default)
3. Aggregate predictions:
   - Classification: majority vote across all B trees
   - Regression: average prediction across all B trees
```

### Key Hyperparameters

| Parameter | Role | Typical Values |
|-----------|------|----------------|
| `n_estimators` | Number of trees | 100–1000+ |
| `max_features` | Features per split | `sqrt(p)` for classification, `p/3` for regression |
| `max_depth` | Tree depth limit | None (grow fully) or tune |
| `min_samples_split` | Min samples to split | 2–20 |
| `min_samples_leaf` | Min samples in leaf | 1–10 |
| `bootstrap` | Whether to bootstrap | True (default) |

### Why Random Feature Sampling?

Without feature randomness, all trees would split on the same dominant features — making them **correlated**. The variance reduction from averaging correlated predictors is:

$$
\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

where $\rho$ is the average correlation between trees. Lower $\rho$ = more variance reduction. Feature sampling decreases $\rho$.

### Out-of-Bag (OOB) Error

Each bootstrap sample leaves out ~36.8% of data (the **out-of-bag** samples). These can be used as a **free validation set**:

$$
P(\text{not selected in one draw}) = \left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368
$$

```python
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Accuracy: {rf.oob_score_:.4f}")  # No separate validation needed
```

### Feature Importance

**Mean Decrease in Impurity (MDI):**
- Sum of impurity decreases weighted by the number of samples reaching each node, for each feature
- Fast, but **biased** toward high-cardinality features

**Permutation Importance:**
- Shuffle each feature column and measure accuracy drop
- Model-agnostic, unbiased, but slower

```python
from sklearn.inspection import permutation_importance

# MDI importance (built-in)
mdi_importance = rf.feature_importances_

# Permutation importance (more reliable)
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importance = perm_result.importances_mean
```

### Rahul's Experience — Risk Scoring with Random Forest

> In my risk scoring system for insurance, I used **Random Forest** as part of an **ensemble** alongside LSTM for temporal patterns. The RF component handled **tabular features** — claim history counts, policy age, coverage ratios, loss ratios. I leveraged **permutation importance** to identify top risk drivers, which was critical for stakeholder explainability.
>
> Key decisions:
> - **500 trees** with `max_depth=15` — balanced performance and training time
> - **OOB score** as quick validation during development
> - Feature importance analysis revealed that **claim frequency** and **loss ratio** were the strongest predictors — aligning with domain knowledge
> - Used RF probability outputs as a feature for the final stacking ensemble

---

## 1.4 Gradient Boosting (XGBoost, LightGBM, CatBoost)

### Core Idea — Sequential Learning

**Gradient Boosting** builds trees **sequentially**, where each new tree corrects the errors of the previous ensemble. It performs **gradient descent in function space**.

### Algorithm — Gradient Boosted Decision Trees (GBDT)

```
Initialize: F_0(x) = argmin_c Σ L(y_i, c)  (e.g., mean for regression, log-odds for classification)

For m = 1 to M:
    1. Compute pseudo-residuals (negative gradient of loss):
       r_im = -∂L(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)
    
    2. Fit a shallow decision tree h_m(x) to the pseudo-residuals
    
    3. Update the model:
       F_m(x) = F_{m-1}(x) + η · h_m(x)
       
       where η is the learning rate (shrinkage)
```

### Why "Gradient" Boosting?

For **regression** with MSE loss:

$$
L(y, F(x)) = \frac{1}{2}(y - F(x))^2 \implies r_i = -\frac{\partial L}{\partial F(x_i)} = y_i - F_{m-1}(x_i) = \text{residual}
$$

Each tree literally fits the **residuals**. For other losses (log-loss, Huber), the pseudo-residuals are the negative gradients of the respective loss function.

### XGBoost Objective — Regularized Loss

$$
\text{Obj} = \sum_{i=1}^{N} L(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)
$$

$$
\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

where $T$ is the number of leaves, $w_j$ are leaf weights, $\gamma$ penalizes complexity (pruning), and $\lambda$ is L2 regularization on leaf weights.

### XGBoost Split Gain

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

where $G = \sum g_i$ (sum of first-order gradients) and $H = \sum h_i$ (sum of second-order gradients). Split only happens if Gain > 0.

### XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Tree growth** | Level-wise (breadth-first) | Leaf-wise (best-first) | Symmetric (balanced) |
| **Speed** | Fast | Faster (2–5x) | Moderate |
| **Categorical features** | Needs encoding | Native support (optimal split) | Native support (ordered encoding) |
| **GPU support** | Yes | Yes | Yes (excellent) |
| **Missing values** | Default direction learned | Native handling | Native handling |
| **Regularization** | L1 + L2 + `gamma` | L1 + L2 | L2 + random permutations |
| **Overfitting control** | `gamma`, `min_child_weight` | `min_data_in_leaf`, `lambda_l1` | Built-in ordered boosting |
| **Best for** | Structured/tabular data | Large datasets, speed | Categorical-heavy data |

### Key Hyperparameters

| Parameter | XGBoost Name | LightGBM Name | Role | Typical Range |
|-----------|-------------|---------------|------|---------------|
| Number of trees | `n_estimators` | `n_estimators` | Model complexity | 100–5000 |
| Learning rate | `learning_rate` / `eta` | `learning_rate` | Shrinkage per step | 0.01–0.3 |
| Tree depth | `max_depth` | `max_depth` | Per-tree complexity | 3–10 |
| Subsample ratio | `subsample` | `bagging_fraction` | Row sampling per tree | 0.6–1.0 |
| Feature sampling | `colsample_bytree` | `feature_fraction` | Column sampling per tree | 0.6–1.0 |
| Min child weight | `min_child_weight` | `min_child_samples` | Minimum samples in leaf | 1–100 |
| L1 regularization | `alpha` / `reg_alpha` | `lambda_l1` | Sparsity | 0–10 |
| L2 regularization | `lambda` / `reg_lambda` | `lambda_l2` | Weight shrinkage | 0–10 |
| Complexity penalty | `gamma` | `min_split_gain` | Minimum gain to split | 0–5 |

### Learning Rate and n_estimators Trade-off

Lower learning rate + more trees = better generalization but slower training:

```python
# Aggressive — fast, but may overfit
xgb_fast = XGBClassifier(learning_rate=0.3, n_estimators=100)

# Conservative — slower, but generalizes better
xgb_slow = XGBClassifier(learning_rate=0.01, n_estimators=3000, early_stopping_rounds=50)
```

### Early Stopping

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
print(f"Best iteration: {model.best_iteration}")
```

### Rahul's Experience — Tesla Price Forecasting with XGBoost

> For **Tesla stock price forecasting**, I used **XGBoost regression** to predict next-day Open, High, Low, and Close prices. The model consumed **engineered features**: RSI, MACD, OBV, Bollinger Bands, GARCH volatility estimates, and lagged price ratios.
>
> Key decisions:
> - **Optuna** for hyperparameter tuning — found optimal `max_depth=7`, `learning_rate=0.015`, `n_estimators=2800` with early stopping
> - **Time-series cross-validation** (expanding window) — never leaked future data
> - **Feature importance** from XGBoost highlighted MACD crossover signals and 5-day lagged returns as top predictors
> - Final XGBoost model achieved **RMSE of ~$3.2** on daily close price predictions
> - Saved 4 separate models (one per target: Open, High, Low, Close) using `joblib`

---

## 1.5 Support Vector Machines (SVM)

### Core Idea — Margin Maximization

SVM finds the **hyperplane** that maximizes the **margin** — the distance between the decision boundary and the nearest data points from each class (called **support vectors**).

### Hard-Margin SVM (Linearly Separable)

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

- The margin width is $\frac{2}{\|\mathbf{w}\|}$
- Minimizing $\|\mathbf{w}\|^2$ maximizes the margin
- Only works for linearly separable data

### Soft-Margin SVM (C Parameter)

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i
$$

$$
\text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

where $\xi_i$ are **slack variables** allowing misclassification.

| C Value | Behavior |
|---------|----------|
| Large C | Small margin, fewer violations — risk overfitting |
| Small C | Large margin, more violations allowed — risk underfitting |

### The Kernel Trick

For non-linearly separable data, SVM maps inputs to a **higher-dimensional** space where a linear separator exists:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$

The **kernel trick** computes the dot product in the high-dimensional space **without explicitly computing** $\phi(\mathbf{x})$.

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$ | Linearly separable data, high-dimensional text |
| **Polynomial** | $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | Feature interactions up to degree d |
| **RBF (Gaussian)** | $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ | Most common; maps to infinite dimensions |
| **Sigmoid** | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | Similar to neural networks; rarely used |

### RBF Kernel — gamma Parameter

- **Large gamma:** Each point's influence is local → complex boundary → overfitting
- **Small gamma:** Each point's influence is global → smooth boundary → underfitting

### When to Use SVM

- **Small to medium** datasets (training is $O(N^2)$ to $O(N^3)$)
- **High-dimensional** data (text classification with TF-IDF)
- When you need a **maximum-margin** classifier
- Binary classification with clear separation

### When NOT to Use SVM

- Large datasets ($N > 100K$) — too slow
- When probability estimates are needed (SVM doesn't natively output probabilities; Platt scaling adds overhead)
- When interpretability is critical

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Always scale for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)
```

---

## 1.6 K-Nearest Neighbors (KNN)

### Core Idea

KNN is a **lazy learner** — it stores the entire training set and classifies new points by a **majority vote** of their K nearest neighbors.

### Algorithm

```
For a new query point x_q:
    1. Compute distance from x_q to ALL training points
    2. Select the K closest training points
    3. Classification: majority vote among K neighbors
       Regression: average (or weighted average) of K neighbors' values
```

### Distance Metrics

| Metric | Formula | Properties |
|--------|---------|------------|
| **Euclidean (L2)** | $\sqrt{\sum (x_i - y_i)^2}$ | Sensitive to scale; most common |
| **Manhattan (L1)** | $\sum \|x_i - y_i\|$ | More robust to outliers |
| **Minkowski** | $\left(\sum \|x_i - y_i\|^p\right)^{1/p}$ | Generalizes L1 (p=1) and L2 (p=2) |
| **Cosine** | $1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$ | For text/embeddings (direction, not magnitude) |

### Choosing K

| K Value | Behavior |
|---------|----------|
| Small K (1–3) | Complex boundary, sensitive to noise — overfitting |
| Large K | Smooth boundary, underfitting |
| Even K | Risk of ties — use odd K or weighted voting |

**Selection strategy:** Use cross-validation to find the K with lowest validation error. Typical range: 3–20.

### Curse of Dimensionality

In high dimensions, **all points become equidistant**. The concept of "nearest" loses meaning:

$$
\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \quad \text{as} \quad \text{dimensions} \to \infty
$$

**Mitigation:** PCA / feature selection before KNN; use appropriate distance metrics.

### Pros and Cons

| Pros | Cons |
|------|------|
| No training phase (lazy) | Slow prediction: $O(Nd)$ per query |
| Nonparametric — no assumptions | Memory-intensive — stores all data |
| Simple and intuitive | Sensitive to irrelevant features |
| Works well with small data | Curse of dimensionality |

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
knn.fit(X_train_s, y_train)
```

---

## 1.7 Naive Bayes

### Core Idea — Bayes' Theorem

Naive Bayes classifies by applying **Bayes' theorem** with a **strong independence assumption** — all features are conditionally independent given the class.

### Bayes' Theorem

$$
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}
$$

### Naive Independence Assumption

$$
P(\mathbf{x} \mid y) = P(x_1 \mid y) \cdot P(x_2 \mid y) \cdots P(x_n \mid y) = \prod_{j=1}^{n} P(x_j \mid y)
$$

### Classification Rule

$$
\hat{y} = \arg\max_{y} \, P(y) \prod_{j=1}^{n} P(x_j \mid y)
$$

In practice, use **log-probabilities** to avoid numerical underflow:

$$
\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{j=1}^{n} \log P(x_j \mid y) \right]
$$

### Variants

| Variant | Feature Type | $P(x_j \mid y)$ Model | Use Case |
|---------|-------------|------------------------|----------|
| **Gaussian NB** | Continuous | Normal distribution | General numerical features |
| **Multinomial NB** | Counts/frequencies | Multinomial distribution | **Text classification** (word counts, TF-IDF) |
| **Bernoulli NB** | Binary | Bernoulli distribution | Binary features (word presence/absence) |
| **Complement NB** | Counts | Complement of class | **Imbalanced** text classification |

### Laplace Smoothing

To handle zero probabilities (a feature value never seen with a class):

$$
P(x_j \mid y) = \frac{\text{count}(x_j, y) + \alpha}{\text{count}(y) + \alpha \cdot |V|}
$$

where $\alpha = 1$ is Laplace smoothing and $|V|$ is the vocabulary size.

### Text Classification Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0))
])

text_clf.fit(X_train_text, y_train)
predictions = text_clf.predict(X_test_text)
```

### When to Use

- **Text classification** — spam detection, sentiment analysis, document categorization
- **Very fast** training and prediction
- Small datasets where you need quick baseline
- When the independence assumption approximately holds

---

# 2. Supervised Learning — Regression

---

## 2.1 Linear Regression

### Core Idea

Linear regression models the relationship between a dependent variable $y$ and independent variables $\mathbf{x}$ as a linear function:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p
$$

### Ordinary Least Squares (OLS)

Minimize the sum of squared residuals:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = \arg\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2
$$

**Closed-form solution (Normal Equation):**

$$
\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### Assumptions of Linear Regression

| Assumption | Description | Violation Test |
|------------|-------------|----------------|
| **Linearity** | Relationship is linear | Residual plots |
| **Independence** | Observations are independent | Durbin-Watson test |
| **Homoscedasticity** | Constant variance of errors | Breusch-Pagan test |
| **Normality** | Errors are normally distributed | Q-Q plot, Shapiro-Wilk |
| **No multicollinearity** | Features not highly correlated | VIF (Variance Inflation Factor) |

### R-Squared ($R^2$)

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- $R^2 = 1$: perfect fit
- $R^2 = 0$: model predicts the mean (no better than baseline)
- $R^2 < 0$: model worse than predicting the mean

**Adjusted $R^2$:** Penalizes adding unnecessary features:

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(N - 1)}{N - p - 1}
$$

---

## 2.2 Ridge, Lasso, and Elastic Net

### Ridge Regression (L2)

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \left[ \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2 + \lambda \| \mathbf{w} \|_2^2 \right]
$$

- **Closed-form:** $\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$
- Shrinks coefficients toward zero but **never exactly zero**
- Handles **multicollinearity** by stabilizing the matrix inversion
- Use when: all features are potentially relevant

### Lasso Regression (L1)

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \left[ \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2 + \lambda \| \mathbf{w} \|_1 \right]
$$

- No closed-form — solved by coordinate descent
- Drives coefficients to **exactly zero** — performs **automatic feature selection**
- Use when: you suspect many features are irrelevant

### Elastic Net (L1 + L2)

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \left[ \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2 + \lambda_1 \| \mathbf{w} \|_1 + \lambda_2 \| \mathbf{w} \|_2^2 \right]
$$

- Combines L1 sparsity with L2 stability
- Handles groups of correlated features better than Lasso alone
- `l1_ratio` controls the L1/L2 mix (0 = Ridge, 1 = Lasso)

### Comparison Table

| Method | Penalty | Feature Selection | Multicollinearity | Coefficients |
|--------|---------|-------------------|--------------------|--------------|
| OLS | None | No | Fails (singular matrix) | Unbiased, high variance |
| Ridge | L2 | No | Handles well | Shrunk, never zero |
| Lasso | L1 | Yes | Selects one from group | Sparse (many zeros) |
| Elastic Net | L1 + L2 | Yes | Handles well | Sparse with grouping |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge
ridge = Ridge(alpha=1.0)

# Lasso
lasso = Lasso(alpha=0.01)

# Elastic Net
enet = ElasticNet(alpha=0.01, l1_ratio=0.5)

# Tune alpha with CV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
```

---

## 2.3 Polynomial Regression

### Core Idea

Polynomial regression models **non-linear** relationships by adding polynomial feature terms while keeping the model **linear in its parameters**:

$$
\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_d x^d
$$

This is still linear regression applied to **transformed** features $[x, x^2, x^3, \dots, x^d]$.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('reg', LinearRegression())
])

poly_model.fit(X_train, y_train)
```

### Caution

- Degree > 3 often overfits
- Feature count explodes: $\binom{p + d}{d}$ features for $p$ original features at degree $d$
- Combine with Ridge/Lasso regularization to control overfitting

---

## 2.4 Tree-Based Regression (Decision Tree / RF / XGBoost)

All tree-based classifiers have regression counterparts:

| Model | scikit-learn Class | Prediction Method |
|-------|--------------------|-------------------|
| Decision Tree | `DecisionTreeRegressor` | Mean of leaf samples |
| Random Forest | `RandomForestRegressor` | Average across tree means |
| XGBoost | `XGBRegressor` | Sum of leaf values |
| LightGBM | `LGBMRegressor` | Sum of leaf values |

**Key difference from classification:** Splitting criterion uses **MSE** (or MAE, Friedman MSE) instead of Gini/entropy.

$$
\text{MSE split criterion} = \frac{N_L}{N} \text{MSE}_L + \frac{N_R}{N} \text{MSE}_R
$$

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)
rf_reg.fit(X_train, y_train)

# XGBoost Regression
xgb_reg = XGBRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, eval_metric='rmse'
)
xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

---

# 3. Unsupervised Learning

---

## 3.1 K-Means Clustering

### Core Idea

Partition $N$ data points into $K$ clusters by minimizing the **within-cluster sum of squares (WCSS)**:

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \| \mathbf{x}_i - \boldsymbol{\mu}_k \|_2^2
$$

### Algorithm (Lloyd's)

```
1. Initialize K centroids (randomly or via K-Means++)
2. Repeat until convergence:
   a. ASSIGN: each point to the nearest centroid (using L2 distance)
      c_i = argmin_k ||x_i - μ_k||²
   b. UPDATE: recompute each centroid as the mean of its assigned points
      μ_k = (1/|C_k|) Σ_{x_i ∈ C_k} x_i
3. Stop when assignments don't change (or max iterations reached)
```

### K-Means++ Initialization

Standard random initialization can produce poor clusters. **K-Means++** selects initial centroids that are spread apart:

1. Choose first centroid uniformly at random
2. For each subsequent centroid, select point with probability proportional to $D(x)^2$ (squared distance to nearest existing centroid)

This yields $O(\log K)$-competitive solution compared to optimal — drastically reduces poor convergence.

### Choosing K

**Elbow Method:**
- Plot WCSS vs. K
- Look for the "elbow" where adding more clusters yields diminishing returns
- Subjective — not always a clear elbow

**Silhouette Score:**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

where $a(i)$ = average distance to same-cluster points, $b(i)$ = average distance to nearest-cluster points.

- Range: $[-1, 1]$
- $s \approx 1$: well-clustered
- $s \approx 0$: on boundary
- $s < 0$: likely in wrong cluster

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss = []
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow
plt.plot(range(2, 11), wcss, 'bx-')
plt.xlabel('K'); plt.ylabel('WCSS'); plt.title('Elbow Method')
```

### Limitations

- Assumes **spherical**, equally-sized clusters
- Sensitive to **initialization** (mitigated by K-Means++)
- Must specify K in advance
- Sensitive to **outliers** (centroids get pulled)
- Uses **L2 norm** — affected by scale (always standardize first)
- Cannot handle **non-convex** cluster shapes

---

## 3.2 K-Medoids (PAM)

### Core Idea

Like K-Means but uses **actual data points** (medoids) as cluster centers instead of means. More robust to outliers.

### K-Medoids vs K-Means

| Aspect | K-Means | K-Medoids |
|--------|---------|-----------|
| Center | Mean (virtual point) | Actual data point |
| Distance | L2 (Euclidean) | Any metric (L1, L2, custom) |
| Outlier robustness | Low (mean pulled by outliers) | High (medoid stays on data) |
| Complexity | $O(NKI)$ | $O(N^2 K I)$ — slower |
| Interpretability | Centroid may not be a real point | Center is always a real observation |

### Algorithm (Partitioning Around Medoids — PAM)

```
1. Initialize K medoids (from actual data points)
2. Repeat until convergence:
   a. ASSIGN: each point to nearest medoid
   b. SWAP: for each medoid m and each non-medoid o:
      - Compute total cost if o replaces m
      - If cost decreases, perform the swap
3. Stop when no swap reduces total cost
```

### When to Use

- When you need **interpretable** cluster centers (actual data points)
- Non-Euclidean distances (e.g., Manhattan, edit distance for strings)
- Presence of outliers
- Small to medium datasets (PAM is quadratic)

```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=3, metric='manhattan', random_state=42)
kmedoids.fit(X_scaled)
print(f"Medoid indices: {kmedoids.medoid_indices_}")
```

---

## 3.3 DBSCAN (Density-Based Spatial Clustering)

### Core Idea

Clusters are **dense regions** separated by **sparse regions**. Points in low-density areas are classified as noise. No need to specify the number of clusters.

### Key Concepts

| Term | Definition |
|------|-----------|
| **eps (ε)** | Maximum distance between two points to be considered neighbors |
| **min_samples** | Minimum number of points within ε to form a dense region |
| **Core point** | Has ≥ min_samples points within ε |
| **Border point** | Within ε of a core point but doesn't have min_samples neighbors |
| **Noise point** | Neither core nor border — outlier |

### Algorithm

```
1. For each point p:
   a. Find all points within ε distance (ε-neighborhood)
   b. If |neighborhood| ≥ min_samples → p is a CORE point
2. Form clusters by connecting core points that are within ε of each other
3. Assign border points to the cluster of the nearest core point
4. Points not assigned to any cluster → NOISE (label = -1)
```

### Choosing eps and min_samples

- **min_samples:** Rule of thumb = $\max(2 \times \text{dimensions}, 5)$
- **eps:** Use the **k-distance graph** (k = min_samples):
  - Compute k-th nearest neighbor distance for each point
  - Sort in ascending order and plot
  - The "elbow" suggests optimal eps

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# k-distance plot to choose eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel('Points'); plt.ylabel('5th NN Distance'); plt.title('K-Distance Plot')

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
n_clusters = len(set(labels) - {-1})
n_noise = (labels == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

### DBSCAN vs K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| Shape | Spherical only | **Arbitrary shapes** |
| K required | Yes | No (auto-detected) |
| Outliers | No handling (assigns all) | Labels as noise (-1) |
| Cluster sizes | Roughly equal | Can be different |
| Density | Assumes uniform | Requires uniform density within clusters |
| Scalability | $O(NK)$ | $O(N \log N)$ with spatial indexing |

---

## 3.4 Hierarchical Clustering

### Core Idea

Builds a **hierarchy of clusters** — either by merging small clusters (agglomerative, bottom-up) or splitting large ones (divisive, top-down). The result is a **dendrogram**.

### Agglomerative Algorithm

```
1. Start: each point is its own cluster (N clusters)
2. Repeat until one cluster remains:
   a. Compute distance between all cluster pairs
   b. Merge the two closest clusters
   c. Update distance matrix
3. Cut the dendrogram at desired height to get K clusters
```

### Linkage Methods

| Linkage | Distance Between Clusters | Behavior |
|---------|--------------------------|----------|
| **Single** | $\min_{a \in A, b \in B} d(a, b)$ | Chain effect — elongated clusters |
| **Complete** | $\max_{a \in A, b \in B} d(a, b)$ | Compact, spherical clusters |
| **Average** | $\frac{1}{|A||B|} \sum_{a,b} d(a, b)$ | Balanced — most commonly used |
| **Ward** | Minimizes increase in total WCSS | Produces equal-size clusters — most popular |

### Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X_scaled, method='ward', metric='euclidean')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Cut dendrogram at k=3 clusters
labels = fcluster(Z, t=3, criterion='maxclust')
```

### When to Use

- You want to **visualize** the cluster hierarchy (dendrogram)
- Don't know the number of clusters in advance
- Small to medium datasets ($O(N^2)$ memory, $O(N^3)$ time for naive implementation)
- Need **nested** cluster structure

---

## 3.5 PCA (Principal Component Analysis)

### Core Idea

PCA finds a new coordinate system (principal components) that captures **maximum variance** in the data. Each component is a linear combination of original features, and components are **orthogonal** to each other.

### Mathematical Formulation

1. **Center** the data: $\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}$
2. **Covariance matrix:** $\mathbf{C} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c$
3. **Eigendecomposition:** $\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$
4. **Sort** eigenvectors by eigenvalue (largest first)
5. **Project:** $\mathbf{Z} = \mathbf{X}_c \mathbf{V}_d$ where $\mathbf{V}_d$ contains the top $d$ eigenvectors

### Explained Variance

$$
\text{Explained variance ratio}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}
$$

Select $d$ components such that cumulative explained variance ≥ 95% (common threshold).

### When to Use

- **Dimensionality reduction** before ML models (reduce multicollinearity, noise)
- **Visualization** (project to 2D or 3D)
- **Speed up** training on high-dimensional data
- Preprocessing for algorithms sensitive to dimensionality (KNN, SVM)

### Limitations

- Captures only **linear** relationships
- Components lose interpretability — they're abstract linear combinations
- Sensitive to feature scales — **always standardize** first
- Assumes variance = importance (may not always hold)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

# Scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('PCA Scree Plot')
```

---

## 3.6 t-SNE and UMAP

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose:** Non-linear dimensionality reduction for **visualization** (typically 2D/3D).

**How it works:**
1. Compute pairwise similarities in high-D using Gaussian kernel
2. Compute pairwise similarities in low-D using **Student's t-distribution** (heavy tails prevent crowding)
3. Minimize KL divergence between the two distributions via gradient descent

| Parameter | Role | Typical |
|-----------|------|---------|
| `perplexity` | Balance between local/global structure | 5–50 (try 30) |
| `n_iter` | Gradient descent iterations | 1000+ |
| `learning_rate` | Step size | 200 (auto is fine) |

**Key caveats:**
- **Non-deterministic** — different runs give different layouts
- **Cannot** project new data (no transform method; must refit)
- Distances between **distant** clusters are meaningless
- Cluster sizes don't reflect actual data proportions
- **Use only for visualization** — not for downstream ML

### UMAP (Uniform Manifold Approximation and Projection)

**Advantages over t-SNE:**

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Speed | Slow ($O(N^2)$) | Fast ($O(N \log N)$) |
| Scalability | ~10K points | 1M+ points |
| Global structure | Poorly preserved | Better preserved |
| Determinism | Non-deterministic | More stable (with random_state) |
| Transform new data | No | Yes (`transform()` method) |
| Use in pipelines | Visualization only | Visualization AND as preprocessing |

```python
from sklearn.manifold import TSNE
import umap

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
# New data can be projected
X_new_umap = reducer.transform(X_new)
```

---

# 4. Anomaly Detection

---

## 4.1 Isolation Forest

### Core Idea

Anomalies are **few** and **different**. In a random partitioning tree, anomalies are isolated (separated from the rest) in **fewer splits** than normal points.

### Algorithm

```
Training (build forest of isolation trees):
1. For each tree t = 1 to T:
   a. Sample a subset of data (typically 256 points)
   b. Build an isolation tree:
      - Pick a random feature
      - Pick a random split value between min and max of that feature
      - Recursively partition until each point is isolated or max depth reached

Scoring:
1. For each data point x:
   a. Pass through each tree and record path length h(x)
   b. Anomaly score:
      s(x, N) = 2^(-E[h(x)] / c(N))
   where c(N) is the average path length of unsuccessful search in BST
   
   - s ≈ 1: anomaly (short path)
   - s ≈ 0.5: normal (average path)
   - s ≈ 0: very normal (long path)
```

### Why Short Paths = Anomalies

Normal points are densely packed — it takes many random splits to isolate them. Anomalies sit in sparse regions — a single split can separate them.

```
        [Random Split on Feature 3]
       /                            \
    ANOMALY                   [Split Feature 1]
    (isolated!)              /                 \
                        [Split F5]          [Split F2]
                        /      \            /      \
                      ...     ...         ...     ...
                    (normal points need many splits)
```

### Hyperparameters

| Parameter | Role | Typical |
|-----------|------|---------|
| `n_estimators` | Number of isolation trees | 100–300 |
| `max_samples` | Subsample size per tree | 256 or 'auto' |
| `contamination` | Expected proportion of anomalies | 0.01–0.1 |
| `max_features` | Features per tree | 1.0 (all) |

### Rahul's Experience — Disaster/Anomaly Detection

> In the **risk scoring system**, I implemented **Isolation Forest** for anomaly detection in insurance claims data. The goal was to flag **unusual claim patterns** — sudden spikes, outlier amounts, suspicious frequency patterns.
>
> Key implementation decisions:
> - `contamination=0.05` — estimated 5% anomalous claims based on domain knowledge
> - Features: claim frequency, average claim amount, time between claims, loss ratio
> - **Path length analysis** helped explain to stakeholders WHY a claim was flagged as anomalous
> - Combined IF scores with the main ensemble model — anomaly score became an additional feature
> - Achieved **92% precision** on known fraudulent/unusual claims in historical validation

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination=0.05,
    random_state=42
)

# Fit and predict (-1 = anomaly, 1 = normal)
iso_forest.fit(X_train)
predictions = iso_forest.predict(X_test)
scores = iso_forest.decision_function(X_test)  # Lower = more anomalous

# Flag anomalies
anomalies = X_test[predictions == -1]
print(f"Detected {len(anomalies)} anomalies out of {len(X_test)} samples")
```

---

## 4.2 Local Outlier Factor (LOF)

### Core Idea

LOF measures the **local density** of a point relative to its neighbors. A point with substantially lower density than its neighbors is considered an outlier.

### Key Concepts

1. **k-distance(p):** Distance from point p to its k-th nearest neighbor
2. **Reachability distance:** $\text{reach-dist}_k(p, o) = \max(k\text{-distance}(o), d(p, o))$ — smooths out density estimation
3. **Local Reachability Density (LRD):**

$$
\text{LRD}_k(p) = \frac{1}{\frac{1}{k} \sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}
$$

4. **Local Outlier Factor:**

$$
\text{LOF}_k(p) = \frac{1}{k} \sum_{o \in N_k(p)} \frac{\text{LRD}_k(o)}{\text{LRD}_k(p)}
$$

### Interpretation

| LOF Value | Meaning |
|-----------|---------|
| LOF ≈ 1 | Similar density to neighbors — normal |
| LOF >> 1 | Much lower density than neighbors — outlier |
| LOF < 1 | Denser than neighbors — inlier |

### Advantage Over Global Methods

LOF detects **local** outliers — points that are outliers relative to their neighborhood, even if they appear normal globally. This is crucial for datasets with clusters of varying density.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
predictions = lof.fit_predict(X)  # -1 = outlier, 1 = inlier
scores = lof.negative_outlier_factor_  # More negative = more outlier
```

---

## 4.3 One-Class SVM

### Core Idea

Learn a boundary that encloses the **normal** data in feature space. Points outside this boundary are anomalies. Uses the **kernel trick** to map to high-dimensional space.

### Formulation

$$
\min_{\mathbf{w}, \rho, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu N} \sum_{i} \xi_i - \rho
$$

subject to: $\mathbf{w}^T \phi(\mathbf{x}_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$

where $\nu$ is the upper bound on the fraction of outliers.

### When to Use

- Only **normal** data available for training (true one-class setting)
- Data is **not too high-dimensional**
- Small to medium datasets
- When you need a clear decision boundary in kernel space

```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_train_normal)  # Train on normal data only
predictions = oc_svm.predict(X_test)  # -1 = anomaly, 1 = normal
```

---

## 4.4 Autoencoders for Anomaly Detection

### Core Idea

Train an autoencoder on **normal data** to learn a compressed representation. Normal data reconstructs well; anomalies have **high reconstruction error** because the model hasn't learned their patterns.

### Architecture

```
Input → Encoder → Bottleneck (latent) → Decoder → Reconstruction
  x          z (compressed)                   x̂ = Decoder(z)

Anomaly score = ||x - x̂||² (reconstruction error)
```

### Why It Works

- Normal patterns are captured in the latent space
- Anomalies lie outside the learned manifold → decoder can't reconstruct them well
- Threshold on reconstruction error separates normal from anomalous

```python
import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# Train on normal data, score by reconstruction error
model = AnomalyAutoencoder(input_dim=X_train.shape[1])
criterion = nn.MSELoss(reduction='none')

# After training...
with torch.no_grad():
    reconstructions = model(X_test_tensor)
    errors = criterion(reconstructions, X_test_tensor).mean(dim=1)
    threshold = errors.quantile(0.95)  # Top 5% are anomalies
    anomalies = errors > threshold
```

### Comparison of Anomaly Detection Methods

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **Isolation Forest** | Fast, scalable, no density estimation | Struggles with local anomalies | General tabular data |
| **LOF** | Detects local outliers | Slow for large data | Varying-density clusters |
| **One-Class SVM** | Kernel trick for complex boundaries | Slow, needs tuning | Small datasets, one-class |
| **Autoencoder** | Learns complex patterns, high-dim data | Needs training, threshold selection | High-dim / sequential data |

---

# 5. Ensemble Methods

---

## 5.1 Bagging (Bootstrap Aggregating)

### Core Idea

Train multiple models on **random bootstrap samples** of the data and **aggregate** their predictions. Reduces **variance** without increasing bias.

### How It Works

```
1. Create B bootstrap samples (sample with replacement, same size as original)
2. Train one base model per sample (typically decision trees)
3. Aggregate:
   - Classification: majority vote
   - Regression: average
```

### Why It Works — Variance Reduction

For i.i.d. estimators with variance $\sigma^2$:

$$
\text{Var}(\text{average}) = \frac{\sigma^2}{B}
$$

For correlated estimators (correlation $\rho$):

$$
\text{Var}(\text{average}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
$$

Bagging reduces the second term. Random Forest further reduces $\rho$ via feature randomization.

---

## 5.2 Boosting

### Core Idea

Train models **sequentially**, where each new model focuses on the **mistakes** of the previous ensemble. Reduces **bias** (and sometimes variance).

### AdaBoost

```
1. Initialize equal weights: w_i = 1/N for all samples
2. For m = 1 to M:
   a. Train weak learner h_m on weighted data
   b. Compute weighted error: ε_m = Σ w_i · I(h_m(x_i) ≠ y_i)
   c. Compute learner weight: α_m = 0.5 · ln((1 - ε_m) / ε_m)
   d. Update sample weights:
      - Misclassified: w_i ← w_i · exp(α_m)
      - Correctly classified: w_i ← w_i · exp(-α_m)
   e. Normalize weights
3. Final prediction: H(x) = sign(Σ α_m · h_m(x))
```

### Gradient Boosting (see Section 1.4 for full details)

Each tree fits the **negative gradient** of the loss function — generalizes AdaBoost to arbitrary differentiable losses.

---

## 5.3 Stacking (Stacked Generalization)

### Core Idea

Train a **meta-learner** on the predictions of multiple base models. The meta-learner learns which base models to trust for different regions of the input space.

### Architecture

```
Layer 1 (Base Models):
    Model A (e.g., Random Forest)     → predictions_A
    Model B (e.g., XGBoost)           → predictions_B
    Model C (e.g., Logistic Regression) → predictions_C

Layer 2 (Meta-Learner):
    Input: [predictions_A, predictions_B, predictions_C]
    Meta-model (e.g., Logistic Regression or XGBoost)
    → Final Prediction
```

### Critical: Prevent Leakage

Base model predictions for the meta-learner's training must come from **out-of-fold** predictions (cross-validation), not the training set — otherwise, the meta-learner learns to simply trust the most overfit base model.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    final_estimator=LogisticRegression(),
    cv=5,  # Out-of-fold predictions for meta-learner training
    stack_method='predict_proba'
)

stacking.fit(X_train, y_train)
```

---

## 5.4 Voting

### Hard Voting

$$
\hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_M(\mathbf{x}))
$$

Majority vote of predicted class labels.

### Soft Voting

$$
\hat{y} = \arg\max_k \sum_{m=1}^{M} w_m \cdot P_m(y = k \mid \mathbf{x})
$$

Weighted average of predicted **probabilities** — generally better because it leverages confidence levels.

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300)),
        ('xgb', XGBClassifier(n_estimators=300)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    voting='soft',           # Use probability averaging
    weights=[2, 3, 1]       # Trust XGBoost most
)

voting.fit(X_train, y_train)
```

---

## 5.5 Bagging vs Boosting — Comprehensive Comparison

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel (independent models) | Sequential (each depends on previous) |
| **Goal** | Reduce **variance** | Reduce **bias** (and variance) |
| **Base learners** | Strong (deep trees) | Weak (shallow trees, stumps) |
| **Sample weighting** | Uniform (bootstrap) | Adaptive (focus on errors) |
| **Overfitting risk** | Low (averaging smooths) | Higher (can overfit noisy data) |
| **Sensitivity to noise** | Robust | Sensitive (amplifies noise) |
| **Parallelizable** | Yes | No |
| **Example** | Random Forest | XGBoost, LightGBM, AdaBoost |
| **Typical tree depth** | Deep (unlimited) | Shallow (3–8) |

### When to Choose What

| Scenario | Recommendation |
|----------|---------------|
| Noisy data, prone to overfitting | Bagging (Random Forest) |
| Underfitting, need more accuracy | Boosting (XGBoost) |
| Need fast training, parallelism | Bagging |
| Kaggle competitions, maximum accuracy | Boosting (+ stacking) |
| Explainability required | Single tree or RF with SHAP |

---

# 6. Model Selection and Validation

---

## 6.1 Train / Validation / Test Split

```
┌──────────────────────────────────────────────────────────┐
│                    Full Dataset                          │
├─────────────────────┬──────────────┬─────────────────────┤
│  Training (60-70%)  │  Val (15-20%)│  Test (15-20%)      │
│  Learn parameters   │  Tune hyper- │  Final evaluation   │
│                     │  parameters  │  (touch ONCE)       │
└─────────────────────┴──────────────┴─────────────────────┘
```

**Rules:**
- **Never** tune hyperparameters on the test set
- **Never** use test set for any decision-making during development
- Report final metrics on the test set **once**

---

## 6.2 Cross-Validation

### K-Fold Cross-Validation

```
Fold 1: [Val] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Val] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Val] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Val] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Val]

Final score = average of 5 fold scores
```

### Stratified K-Fold

Preserves the **class distribution** in each fold. Critical for **imbalanced** datasets.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    scores.append(model.score(X[val_idx], y[val_idx]))
print(f"Mean CV score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### Time-Series Cross-Validation (Expanding Window)

**Never** shuffle time-series data. Use **expanding** or **sliding** window:

```
Fold 1: [Train] [Val]
Fold 2: [Train Train] [Val]
Fold 3: [Train Train Train] [Val]
Fold 4: [Train Train Train Train] [Val]
Fold 5: [Train Train Train Train Train] [Val]
```

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
```

> **Rahul's note:** In my Tesla price prediction project, I used `TimeSeriesSplit` with expanding windows to ensure no future data leaked into training. This is **non-negotiable** for any time-series ML system.

---

## 6.3 Bias-Variance Trade-off

### Decomposition of Expected Error

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

| Term | Definition | Example |
|------|-----------|---------|
| **Bias** | Error from overly simplistic model assumptions | Linear model on nonlinear data |
| **Variance** | Error from sensitivity to training data fluctuations | Deep tree memorizing noise |
| **Irreducible noise** | Inherent randomness in the data | Measurement error |

### Visual Intuition

```
                    Model Complexity →
    
    High │ ↘ Bias²                      
         │    ↘            ↗ Variance   
   Error │      ↘        ↗             
         │        ↘    ↗               
         │    Total  ↘↗  Error          
    Low  │           * ← Sweet spot     
         └──────────────────────────────
           Simple              Complex
           (Underfit)          (Overfit)
```

### Practical Strategies

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| **High bias** (underfitting) | Low train score, low val score | Add features, increase model complexity, reduce regularization |
| **High variance** (overfitting) | High train score, low val score | Add data, increase regularization, simpler model, dropout, early stopping |
| **Good fit** | Both train and val scores are high and similar | Ship it |

---

## 6.4 Hyperparameter Tuning

### Grid Search

Exhaustive search over a predefined grid. Simple but scales exponentially.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}
# 3 × 4 × 3 = 36 combinations × 5 folds = 180 fits

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### Random Search

Samples random combinations. Often finds good hyperparameters faster than grid search, especially in high-dimensional search spaces.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'n_estimators': randint(100, 2000),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.005, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    XGBClassifier(), param_distributions,
    n_iter=100, cv=5, scoring='f1', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

### Bayesian Optimization (Optuna)

Uses a **probabilistic model** (Tree-structured Parzen Estimator in Optuna) to guide the search toward promising regions. Far more efficient than grid or random search.

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100)
    }

    model = XGBClassifier(**params, eval_metric='logloss', random_state=42)

    # Time-series CV or Stratified CV
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Comparison

| Method | Pros | Cons | Trials Needed |
|--------|------|------|---------------|
| **Grid Search** | Exhaustive, reproducible | Exponential cost, misses between grid points | All combinations |
| **Random Search** | Faster, explores more dimensions | No learning from past trials | 50–200 |
| **Bayesian (Optuna)** | Smart exploration, converges fast | More complex setup | 50–500 |

---

## 6.5 Feature Selection

### Filter Methods

Evaluate features **independently** of the model using statistical tests.

| Method | Criterion | For |
|--------|-----------|-----|
| Correlation | Pearson/Spearman | Regression |
| Chi-squared | $\chi^2$ independence test | Classification (categorical) |
| Mutual Information | MI(X; Y) | Both (captures non-linear) |
| ANOVA F-test | Variance ratio between classes | Classification (continuous) |

```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest

selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = np.array(feature_names)[selector.get_support()]
```

### Wrapper Methods

Use the **model's performance** to evaluate feature subsets.

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Forward Selection** | Start empty, add best feature each step | Finds good subsets | Slow $O(p^2)$ |
| **Backward Elimination** | Start with all, remove worst each step | Less greedy | Even slower |
| **Recursive Feature Elimination (RFE)** | Recursively remove least important | Built into sklearn | Depends on model |

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    step=1, cv=5, scoring='f1', min_features_to_select=5
)
rfecv.fit(X_train, y_train)
print(f"Optimal features: {rfecv.n_features_}")
print(f"Selected: {np.array(feature_names)[rfecv.support_]}")
```

### Embedded Methods

Feature selection is **built into** the model training process.

| Method | Model | How |
|--------|-------|-----|
| **L1 regularization** | Lasso, LogReg(L1) | Coefficients shrunk to zero |
| **Tree importance** | RF, XGBoost | MDI or gain-based importance |
| **Permutation importance** | Any model | Shuffle feature, measure accuracy drop |
| **SHAP values** | Any model | Game-theoretic feature attribution |

```python
import shap

# SHAP for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

# 7. Feature Engineering

---

## 7.1 Feature Scaling

### Why Scale?

Algorithms using distance (KNN, SVM, K-Means) or gradient-based optimization (logistic regression, neural networks) are sensitive to feature scales. Tree-based models (RF, XGBoost) are **not** affected by scaling.

| Scaler | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| **StandardScaler** | $\frac{x - \mu}{\sigma}$ | Mean=0, Std=1 | Default choice; assumes ~Gaussian |
| **MinMaxScaler** | $\frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | [0, 1] | Bounded features; neural networks |
| **RobustScaler** | $\frac{x - \text{median}}{\text{IQR}}$ | Centered on median | Data with **outliers** |
| **MaxAbsScaler** | $\frac{x}{\max(|x|)}$ | [-1, 1] | Sparse data |

### Critical Rules

1. **Fit on training data ONLY** — transform train and test with the same parameters
2. Scaling inside a `Pipeline` prevents data leakage automatically

```python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', RobustScaler()),  # Robust to outliers in insurance data
    ('model', XGBClassifier())
])
pipe.fit(X_train, y_train)  # Scaler learns from train only
```

---

## 7.2 Categorical Encoding

| Method | How | When |
|--------|-----|------|
| **One-Hot Encoding** | Binary column per category | Low cardinality (< 20), no ordinal relationship |
| **Label Encoding** | Integer per category | Ordinal features (Low=0, Medium=1, High=2) |
| **Target Encoding** | Replace category with mean of target | High cardinality; use with CV to prevent leakage |
| **Ordinal Encoding** | Integer per rank | Ordinal features with clear ordering |
| **Frequency Encoding** | Replace with count/frequency | When frequency carries signal |
| **Binary Encoding** | Binary representation of integer | High cardinality, memory-efficient |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

# One-Hot (low cardinality)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Target Encoding (high cardinality) — use with CV
te = TargetEncoder(smoothing=10)
X_train['city_encoded'] = te.fit_transform(X_train['city'], y_train)
X_test['city_encoded'] = te.transform(X_test['city'])

# CatBoost handles categoricals natively
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(cat_features=['city', 'state', 'product_type'])
```

---

## 7.3 Missing Value Handling

| Strategy | Method | When |
|----------|--------|------|
| **Deletion** | Drop rows/columns with missing values | < 5% missing, MCAR (Missing Completely at Random) |
| **Mean/Median** | Replace with column mean/median | Numerical, low missing rate |
| **Mode** | Replace with most frequent value | Categorical |
| **KNN Imputation** | Impute based on similar samples | Moderate missing rate, correlated features |
| **Iterative (MICE)** | Model each feature as function of others | Complex missing patterns |
| **Indicator column** | Add binary "is_missing" flag | Missingness itself is informative |

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Simple imputation
mean_imp = SimpleImputer(strategy='median')

# KNN imputation
knn_imp = KNNImputer(n_neighbors=5)

# MICE (iterative)
mice_imp = IterativeImputer(max_iter=10, random_state=42)

# Add missing indicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('model', RandomForestClassifier())
])
```

**Note:** XGBoost and LightGBM handle missing values **natively** — they learn the optimal split direction for missing values during training.

---

## 7.4 Feature Creation — Domain-Specific

### Rahul's Technical Indicators for Tesla Forecasting

| Feature | Formula / Description | Signal |
|---------|----------------------|--------|
| **RSI** (Relative Strength Index) | $\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$ where RS = avg gain / avg loss over 14 periods | Overbought (>70) / Oversold (<30) |
| **MACD** (Moving Average Convergence Divergence) | EMA(12) - EMA(26), Signal line = EMA(9) of MACD | Trend direction and momentum |
| **OBV** (On-Balance Volume) | Cumulative sum: +volume on up days, -volume on down days | Volume confirms price trends |
| **Bollinger Bands** | Middle: SMA(20), Upper/Lower: ±2σ from SMA | Volatility and breakout signals |
| **GARCH Volatility** | GARCH(1,1) estimated conditional variance | Volatility clustering |

```python
import ta  # Technical analysis library

# RSI
df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

# MACD
macd = ta.trend.MACD(df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_hist'] = macd.macd_diff()

# Bollinger Bands
bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['bb_pband'] = bb.bollinger_pband()  # %B

# OBV
df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

# Lagged features
for lag in [1, 2, 3, 5, 10]:
    df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)
    df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)

# Rolling statistics
for window in [5, 10, 20]:
    df[f'rolling_mean_{window}'] = df['Close'].rolling(window).mean()
    df[f'rolling_std_{window}'] = df['Close'].rolling(window).std()
    df[f'rolling_volume_mean_{window}'] = df['Volume'].rolling(window).mean()
```

### General Feature Engineering Patterns

| Pattern | Example | Domain |
|---------|---------|--------|
| **Ratios** | claim_amount / coverage_amount | Insurance |
| **Differences** | current_price - moving_avg | Finance |
| **Aggregations** | mean_claim_per_policy | Insurance |
| **Time-based** | days_since_last_claim | Insurance |
| **Interaction** | feature_A × feature_B | General |
| **Polynomial** | feature^2, feature^3 | General |
| **Binning** | age_group = cut(age, bins) | General |
| **Log transform** | log(1 + skewed_feature) | Skewed distributions |

---

# 8. Common Interview Questions

---

## Q1: "Random Forest vs XGBoost — When would you use each?"

**Answer:**

| Factor | Random Forest | XGBoost |
|--------|---------------|---------|
| Training | Parallel — faster | Sequential — slower |
| Goal | Reduce variance | Reduce bias + variance |
| Overfitting | Resistant (averaging) | Needs careful tuning |
| Noisy data | Better (robust to noise) | Can amplify noise |
| Max accuracy | Good | Usually better |
| Interpretability | Feature importance | SHAP, feature importance |
| Default | Good out-of-box | Needs tuning for best results |

> "In my risk scoring project, I used **Random Forest** as a stable baseline and **XGBoost** as the primary model. RF's OOB error gave quick validation estimates during development. XGBoost with Optuna tuning consistently outperformed RF by 2-3% F1, but RF was more robust when data distribution shifted slightly. In the final stacking ensemble, both were base learners."

---

## Q2: "How does gradient boosting work? Explain step by step."

**Answer:**

"Gradient boosting builds an additive model in a forward stage-wise fashion. It starts with a simple prediction — like the mean for regression. Then it computes the **negative gradient** of the loss function with respect to the current predictions — for MSE, these are simply the residuals. A shallow decision tree is fitted to these pseudo-residuals. The tree's predictions are scaled by a learning rate and added to the ensemble. This process repeats — each new tree corrects the remaining errors.

The key insight is that we're doing **gradient descent in function space** — instead of adjusting parameters, we're adding whole functions (trees) to minimize the loss. XGBoost extends this with a second-order Taylor approximation of the loss and adds regularization on tree complexity — leaf count and leaf weight magnitude — which gives it the famous regularized objective."

---

## Q3: "Explain the bias-variance trade-off with a concrete example."

**Answer:**

"Consider predicting house prices. A **linear regression** with just square footage has **high bias** — it assumes a simple linear relationship, missing neighborhood effects, quality, etc. It consistently underpredicts expensive houses and overpredicts cheap ones. The error is consistent across different training samples.

Now take a **deep decision tree** that memorizes every training example. It has **high variance** — on one training sample it predicts $500K for a specific house, on a different sample it predicts $350K. The predictions swing wildly based on which data it sees.

The sweet spot is somewhere in between — like a **regularized gradient-boosted model** with appropriate depth and learning rate. In my Tesla forecasting work, I found that `max_depth=7` with `learning_rate=0.015` gave the best validation RMSE — deeper trees overfit (high variance), shallower trees missed complex patterns (high bias)."

---

## Q4: "K-Means vs DBSCAN — When to use each?"

**Answer:**

"**K-Means** when:
- You know (or can estimate) the number of clusters
- Clusters are roughly spherical and similar-sized
- Data is large (K-Means scales well)
- You need centroids for interpretation

**DBSCAN** when:
- You don't know the number of clusters
- Clusters have **arbitrary shapes** (crescents, rings)
- There are **outliers** you want identified as noise
- Clusters have varying density (with limitations)

The fundamental difference: K-Means assigns every point to a cluster. DBSCAN can label points as **noise** — which is crucial for anomaly-rich data. In clustering insurance claims, I'd use DBSCAN because claim patterns form irregular shapes and outlier detection is valuable."

---

## Q5: "How does Isolation Forest detect anomalies?"

**Answer:**

"Isolation Forest exploits the key property of anomalies — they are **few** and **different**. It builds random trees by randomly selecting features and random split values. Anomalies, being isolated in sparse regions, require **fewer random splits** to separate from the rest.

The anomaly score is based on the **average path length** across all trees. Short average path = anomaly. The algorithm normalizes this by the expected path length of a balanced binary search tree for comparison.

In my disaster detection work, I used Isolation Forest on sensor data — sudden pressure drops, temperature spikes, or unusual vibration patterns had dramatically shorter path lengths than normal readings. The beauty is that it's **unsupervised** — no labeled anomalies needed for training, and it scales linearly with data size."

---

## Q6: "How do you handle class imbalance?"

**Answer (structured):**

**Data-level techniques:**
- **Oversampling** minority class: SMOTE (synthetic samples), random oversampling
- **Undersampling** majority class: random, Tomek links, NearMiss
- **Combination:** SMOTE + Tomek links

**Algorithm-level techniques:**
- **Class weights:** `class_weight='balanced'` in sklearn; `scale_pos_weight` in XGBoost
- **Cost-sensitive learning:** Higher penalty for misclassifying minority
- **Threshold tuning:** Adjust decision threshold using precision-recall curve

**Evaluation-level techniques:**
- **Don't use accuracy** — use F1, AUC-ROC, AUC-PR, Matthews Correlation Coefficient
- **Stratified cross-validation** to preserve class ratios
- **Precision-Recall curve** (more informative than ROC for severe imbalance)

```python
# XGBoost class weight
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
xgb = XGBClassifier(scale_pos_weight=n_neg/n_pos)

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

> "In my risk scoring system, the high-risk class was only ~8% of data. I used `scale_pos_weight` in XGBoost, stratified CV, and evaluated primarily on F1 and AUC-PR. SMOTE helped the RF base model but didn't improve XGBoost (which handles imbalance better natively)."

---

## Q7: "Explain feature importance methods and their trade-offs."

**Answer:**

| Method | How | Pros | Cons |
|--------|-----|------|------|
| **MDI (Gini Importance)** | Sum of impurity decreases per feature across all trees | Fast, built-in | Biased toward high-cardinality features |
| **Permutation Importance** | Shuffle feature, measure score drop | Model-agnostic, unbiased | Slower, affected by correlated features |
| **SHAP Values** | Game-theoretic: Shapley values for each feature per prediction | Local + global, consistent, theoretically grounded | Computationally expensive |
| **Drop-column** | Remove feature, retrain, measure score drop | Most accurate | Very expensive ($O(p)$ retraining) |
| **Coefficient magnitude** | Absolute value of linear model coefficients | Simple, interpretable | Only for linear models, needs scaling |

> "I always start with **built-in importance** for quick iteration, then use **permutation importance** for unbiased rankings, and **SHAP** for stakeholder presentations. In my risk scoring model, MDI showed 'claim_count' as top feature, which SHAP confirmed — and SHAP additionally showed that its effect was non-linear (marginal impact decreased above 10 claims)."

---

## Q8: "What is the kernel trick and why does it matter?"

**Answer:**

"The kernel trick lets SVMs learn **non-linear** decision boundaries without explicitly mapping data to higher dimensions. Instead of computing $\phi(\mathbf{x})$ (which could be infinite-dimensional for RBF), we compute the **dot product** in that space directly via a kernel function: $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$.

For example, the RBF kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ implicitly maps to an **infinite-dimensional** space — computing this directly would be impossible, but the kernel function gives us the answer in $O(d)$ time.

This matters beyond SVMs — kernel methods appear in kernel PCA, Gaussian processes, and kernel ridge regression."

---

## Q9: "Walk me through how you'd build a model from scratch for a new dataset."

**Answer (my actual workflow):**

```
1. UNDERSTAND: EDA — distributions, correlations, target balance, missing rates
2. CLEAN: Handle missing values, remove duplicates, fix data types
3. ENGINEER: Create domain features (like my technical indicators for finance)
4. BASELINE: LogisticRegression or simple RandomForest — establish a floor
5. VALIDATE: Set up proper CV (stratified for classification, time-series for temporal)
6. ITERATE: Try XGBoost/LightGBM, tune with Optuna
7. ENSEMBLE: Stack best models if marginal gains justify complexity
8. INTERPRET: SHAP analysis, feature importance, sanity-check with domain knowledge
9. DEPLOY: Serialize model, build API (FastAPI), monitor in production
```

> "This is exactly the pipeline I followed for the income estimation model — EDA revealed heavy skew in income, so I log-transformed the target. Feature engineering (employment duration ratios, zip-code median income) improved XGBoost RMSE by 15%. Optuna tuning squeezed out another 3%. Final model deployed with FastAPI endpoint."

---

## Q10: "How does regularization prevent overfitting?"

**Answer:**

"Regularization adds a **penalty** on model complexity to the loss function:

$$
\text{Total Loss} = \text{Data Loss} + \lambda \cdot \text{Complexity Penalty}
$$

**L2 (Ridge)** penalizes large weights — forces the model to distribute importance across features rather than relying on a few with huge coefficients. Geometrically, it constrains weights to lie within a sphere.

**L1 (Lasso)** penalizes the absolute sum of weights — pushes some coefficients to exactly zero, performing automatic feature selection. Geometrically, it constrains weights to a diamond shape (whose corners touch the axes — hence the sparsity).

For trees, regularization takes the form of `max_depth`, `min_samples_leaf`, and XGBoost's `gamma` (minimum gain to split) and `lambda` (L2 on leaf weights). In my XGBoost models, setting `gamma=1` prevented spurious splits on noise, and `reg_lambda=5` reduced leaf weights — together dropping validation RMSE by ~8%."

---

## Q11: "Explain the differences between L1 and L2 regularization."

**Answer:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| Penalty | $\lambda \sum \|w_j\|$ | $\lambda \sum w_j^2$ |
| Geometry | Diamond (L1 ball) | Sphere (L2 ball) |
| Sparsity | Yes — drives weights to zero | No — only shrinks toward zero |
| Feature selection | Built-in | No |
| Solution | No closed form (coordinate descent) | Closed form: $(X^TX + \lambda I)^{-1}X^Ty$ |
| Correlated features | Picks one, ignores rest | Distributes weight among correlated |
| Best when | Many irrelevant features | All features somewhat relevant |

"The geometric intuition: the L1 ball has **corners** on the axes. The loss contour is most likely to first touch the constraint region at a corner — meaning some weights are exactly zero. The L2 ball is smooth, so the intersection point has all weights non-zero but shrunk."

---

## Q12: "How would you detect and handle multicollinearity?"

**Answer:**

**Detection:**
1. **Correlation matrix:** Pearson correlation > 0.8–0.9 between features
2. **VIF (Variance Inflation Factor):** VIF > 5–10 indicates problematic multicollinearity

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the R-squared from regressing feature j on all other features.

**Handling:**
- Remove one from each highly correlated pair
- PCA to create orthogonal components
- Ridge regression (designed for multicollinearity)
- Regularization in general
- Tree-based models (naturally handle collinearity by selecting one feature per split)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['Feature'] = feature_names
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))
```

---

# 9. Key Takeaways

---

## Algorithm Selection Cheat Sheet

```
Is your problem classification or regression?
│
├─ Classification
│   ├─ Need interpretability? → Logistic Regression / Decision Tree
│   ├─ Small data, high dimensions? → SVM (linear kernel) / Naive Bayes
│   ├─ Text data? → Naive Bayes / SVM with TF-IDF
│   ├─ Tabular data, max accuracy? → XGBoost / LightGBM
│   ├─ Need robustness to noise? → Random Forest
│   └─ Lazy baseline? → KNN (small data) / Logistic Regression (any data)
│
├─ Regression
│   ├─ Linear relationship? → Linear / Ridge / Lasso
│   ├─ Many irrelevant features? → Lasso / Elastic Net
│   ├─ Tabular, max accuracy? → XGBoost / LightGBM
│   └─ Non-linear, interpretable? → Decision Tree / Random Forest
│
├─ Clustering
│   ├─ Know K, spherical clusters? → K-Means
│   ├─ Arbitrary shapes, outliers? → DBSCAN
│   ├─ Need hierarchy / dendrogram? → Hierarchical (Ward)
│   └─ Robust to outliers, small data? → K-Medoids
│
└─ Anomaly Detection
    ├─ General tabular? → Isolation Forest
    ├─ Varying density? → LOF
    ├─ Only normal data available? → One-Class SVM / Autoencoder
    └─ High-dimensional / sequential? → Autoencoder
```

## Rahul's Top Production Lessons

| Lesson | Context |
|--------|---------|
| **Always start with a baseline** | LogReg for classification, mean for regression — know what "good" looks like |
| **Feature engineering > model complexity** | My technical indicators (RSI, MACD, GARCH) improved XGBoost RMSE more than any hyperparameter change |
| **Time-series needs special CV** | Using regular k-fold on temporal data leaked future → inflated metrics by ~15% |
| **Ensemble wisely** | Stacking RF + XGBoost + LSTM gave 3% lift over best single model in risk scoring |
| **SHAP for stakeholders** | Domain experts trusted the model after seeing SHAP force plots confirming known risk factors |
| **Monitor in production** | Feature distributions drift — retrain triggers based on PSI (Population Stability Index) |
| **Isolation Forest for free anomaly flags** | Adding IF anomaly score as a feature improved downstream model by 1.5% |
| **Optuna > Grid Search** | Found better XGBoost hyperparameters in 200 trials than grid search over 1000+ combinations |

## Quick Reference — Complexity and Scalability

| Algorithm | Training | Prediction | Memory |
|-----------|----------|------------|--------|
| Logistic Regression | $O(Npd)$ | $O(p)$ | $O(p)$ |
| Decision Tree | $O(Np \log N)$ | $O(\log N)$ | $O(\text{nodes})$ |
| Random Forest | $O(B \cdot N' p' \log N')$ | $O(B \log N)$ | $O(B \cdot \text{nodes})$ |
| XGBoost | $O(M \cdot Np \log N)$ | $O(M \cdot \text{depth})$ | $O(M \cdot \text{nodes})$ |
| SVM (RBF) | $O(N^2 p)$ to $O(N^3)$ | $O(N_{sv} \cdot p)$ | $O(N_{sv} \cdot p)$ |
| KNN | $O(1)$ (lazy) | $O(Np)$ | $O(Np)$ |
| K-Means | $O(NKdI)$ | $O(Kd)$ | $O(Kd)$ |
| DBSCAN | $O(N \log N)$ | N/A | $O(N)$ |
| Isolation Forest | $O(T \cdot N' \log N')$ | $O(T \cdot \log N')$ | $O(T \cdot \text{nodes})$ |

---

*Document prepared for Rahul Sharma — Data Scientist interview preparation. Covers classical ML algorithms with mathematical foundations, production code patterns, and personalized project experience from risk scoring, Tesla forecasting, anomaly detection, and ensemble modeling.*
