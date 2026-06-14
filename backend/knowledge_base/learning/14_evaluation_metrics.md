# Evaluation Metrics for ML & LLMs — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD  
**Focus:** Data Scientist / ML Engineer — Classification, Regression, NLP, LLMs, RAG, Computer Vision  
**Document Scope:** Comprehensive coverage of evaluation metrics across ML, deep learning, NLP, retrieval, and LLM systems

---

## Table of Contents

1. [Classification Metrics](#1-classification-metrics)
2. [Regression Metrics](#2-regression-metrics)
3. [Text Generation Metrics](#3-text-generation-metrics)
4. [Retrieval Metrics](#4-retrieval-metrics)
5. [Object Detection Metrics](#5-object-detection-metrics)
6. [LLM-Specific Metrics](#6-llm-specific-metrics)
7. [RAG-Specific Metrics](#7-rag-specific-metrics)
8. [A/B Testing Metrics](#8-ab-testing-metrics)
9. [Summary Tables by Domain](#9-summary-tables-by-domain)
10. [Common Interview Questions & Answers](#10-common-interview-questions--answers)
11. [Key Takeaways](#11-key-takeaways)

---

# 1. Classification Metrics

---

## 1.1 Confusion Matrix

The **confusion matrix** is the foundation of all classification metrics. For binary classification:

|  | **Predicted Positive** | **Predicted Negative** |
|--|------------------------|------------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **TP:** Correctly predicted positives (cancer detected when cancer exists)
- **FP:** Incorrectly predicted positives — **Type I error** (healthy patient flagged as cancerous)
- **FN:** Incorrectly predicted negatives — **Type II error** (cancer missed)
- **TN:** Correctly predicted negatives (healthy patient confirmed healthy)

**Multi-class extension:** An $N \times N$ matrix where rows are actual classes and columns are predicted classes. Diagonal entries are correct predictions; off-diagonal entries reveal which classes confuse the model.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
# array([[3, 1],
#        [1, 4]])

disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

---

## 1.2 Accuracy

**Formula:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**What it measures:** The fraction of total predictions that are correct.

**When to use:**
- Balanced class distribution (roughly 50/50)
- All misclassification types are equally costly

**When NOT to use:**
- **Imbalanced datasets** — a model that always predicts "Not Fraud" on a dataset with 99.5% non-fraud achieves 99.5% accuracy but catches zero fraud. This is the **accuracy paradox**.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
# Manual: (TP + TN) / total = (4 + 3) / 10 = 0.70
```

---

## 1.3 Precision

**Formula:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interpretation:** Of all instances the model predicted as positive, what fraction are actually positive?

**Optimize precision when:**
- **False positives are expensive** — spam filter (legitimate email in spam folder), hiring decisions, content recommendation
- You need high **confidence** in positive predictions

**Example:** An email spam filter with 95% precision means 5% of emails flagged as spam were actually legitimate.

---

## 1.4 Recall (Sensitivity / True Positive Rate)

**Formula:**

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretation:** Of all actual positive instances, what fraction did the model correctly identify?

**Optimize recall when:**
- **False negatives are expensive** — cancer detection (missing a tumor), fraud detection (missing fraud), security threat detection
- Missing a positive case has severe consequences

**Example:** A cancer screening model with 98% recall catches 98 out of 100 cancerous cases — only 2 slip through.

---

## 1.5 Precision–Recall Trade-off

Precision and recall are inversely related. As you **lower the classification threshold**:
- More instances are predicted positive → **recall increases** (fewer FN)
- But more false positives sneak in → **precision decreases** (more FP)

```
Threshold ↓  →  Predict more positives  →  Recall ↑, Precision ↓
Threshold ↑  →  Predict fewer positives →  Recall ↓, Precision ↑
```

**How to choose the threshold:**
| Scenario | Priority | Threshold Strategy |
|----------|----------|--------------------|
| Cancer detection | Recall | Lower threshold (catch every case) |
| Spam filtering | Precision | Higher threshold (don't block real email) |
| Balanced need | F1 | Threshold that maximizes F1 |
| Business-driven | Custom | Cost-sensitive: minimize $C_{FP} \cdot FP + C_{FN} \cdot FN$ |

---

## 1.6 F1-Score

**Formula (Harmonic Mean of Precision and Recall):**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**Why harmonic mean, not arithmetic?**
- The harmonic mean **penalizes extreme imbalances**. If precision = 1.0 and recall = 0.01, the arithmetic mean gives 0.505 (misleadingly high), but the harmonic mean gives 0.0198 (correctly punishes the near-zero recall).

**Generalized F-beta score:**

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

| $\beta$ | Weight | Use Case |
|---------|--------|----------|
| $\beta = 0.5$ | Precision weighted 2× more | Spam detection, legal doc classification |
| $\beta = 1$ | Equal weight | Balanced classification (default) |
| $\beta = 2$ | Recall weighted 2× more | Cancer screening, fraud detection |

---

## 1.7 Multi-Class Averaging: Micro, Macro, Weighted

For $C$ classes, each class $c$ has its own $TP_c$, $FP_c$, $FN_c$.

### Macro Average

Compute the metric **independently for each class**, then take the **unweighted mean**:

$$\text{Macro-Precision} = \frac{1}{C} \sum_{c=1}^{C} \text{Precision}_c$$

- Treats **all classes equally** regardless of support (sample count)
- Gives **more weight to rare classes** → good for detecting if the model ignores minority classes
- Use when: class balance matters and you want to see per-class performance

### Micro Average

Aggregate all $TP$, $FP$, $FN$ globally, then compute the metric:

$$\text{Micro-Precision} = \frac{\sum_{c} TP_c}{\sum_{c} TP_c + \sum_{c} FP_c}$$

- Equivalent to **overall accuracy** in multi-class single-label classification
- **Dominated by majority class** performance
- Use when: overall correctness matters more than per-class fairness

### Weighted Average

Compute per-class metric, then take the **support-weighted mean**:

$$\text{Weighted-Precision} = \sum_{c=1}^{C} \frac{n_c}{N} \cdot \text{Precision}_c$$

where $n_c$ is the number of true samples for class $c$ and $N$ is the total.

- **Balances between macro and micro** — accounts for class imbalance without ignoring minority classes entirely
- Use when: you want a single number that reflects both class-level and instance-level performance

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Multi-class predictions
y_true_mc = [0, 1, 2, 2, 0, 1, 2, 0, 1, 1]
y_pred_mc = [0, 2, 2, 2, 0, 1, 0, 0, 1, 2]

print(classification_report(y_true_mc, y_pred_mc, target_names=["Cat", "Dog", "Bird"]))
#               precision    recall  f1-score   support
#          Cat       0.75      1.00      0.86         3
#          Dog       0.50      0.50      0.50         4
#         Bird       0.67      0.67      0.67         3
#     accuracy                           0.70        10
#    macro avg       0.64      0.72      0.67        10
# weighted avg       0.63      0.70      0.66        10

# Individual calls
f1_macro    = f1_score(y_true_mc, y_pred_mc, average='macro')
f1_micro    = f1_score(y_true_mc, y_pred_mc, average='micro')
f1_weighted = f1_score(y_true_mc, y_pred_mc, average='weighted')
```

**Quick decision table:**

| Averaging | Prioritizes | Best For |
|-----------|-------------|----------|
| Macro | Rare classes equally | Imbalanced data, fairness |
| Micro | Overall correctness | Balanced data, aggregate accuracy |
| Weighted | Proportional balance | Imbalanced data, single summary metric |

---

## 1.8 ROC Curve & AUC-ROC

### ROC Curve

The **Receiver Operating Characteristic** curve plots:
- **x-axis:** False Positive Rate $= \frac{FP}{FP + TN}$ (also called 1 − Specificity)
- **y-axis:** True Positive Rate $= \frac{TP}{TP + FN}$ (Recall)

across **all classification thresholds** from 0 to 1.

```
TPR (Recall)
 1.0 ┤            ╭──────────
     │          ╭─╯
     │        ╭─╯          ← Good model (bows toward top-left)
     │      ╭─╯
 0.5 ┤    ╭─╯
     │  ╭─╯     ╱ ← Random classifier (diagonal)
     │╭─╯      ╱
     ╰───────╱──────────
 0.0        0.5       1.0  FPR
```

### AUC-ROC

**Area Under the ROC Curve** — a single scalar summarizing model performance across all thresholds.

$$\text{AUC} = \int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))$$

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.5 | Random classifier (no discrimination) |
| < 0.5 | Worse than random (labels may be flipped) |
| 0.7–0.8 | Acceptable |
| 0.8–0.9 | Good |
| 0.9+ | Excellent |

**Probabilistic interpretation:** AUC = probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

**Strengths:**
- **Threshold-invariant** — evaluates model across all operating points
- **Scale-invariant** — doesn't depend on the predicted probability calibration

**Weaknesses:**
- **Misleading under class imbalance** — with 99.9% negatives, even small FPR corresponds to many false positives. AUC can look great while the model floods users with false alarms.
- Treats all thresholds equally, but in practice only a narrow threshold range matters.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_prob = [0.1, 0.4, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6, 0.85, 0.35]

auc = roc_auc_score(y_true, y_prob)
print(f"AUC-ROC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.show()
```

---

## 1.9 PR Curve & AUC-PR (Average Precision)

### PR Curve

Plots **Precision (y)** vs. **Recall (x)** across all thresholds.

**Why PR curve over ROC?**

Under **severe class imbalance** (e.g., 1% positive), ROC-AUC can remain high even when the model generates many false positives, because FPR stays low (denominator is large). The PR curve is **more informative** because precision directly reflects the false positive burden.

$$\text{AP} = \sum_{k} (R_k - R_{k-1}) \cdot P_k$$

| Scenario | Use ROC-AUC | Use PR-AUC |
|----------|-------------|------------|
| Balanced classes | Yes | Either |
| Imbalanced (rare positive) | Misleading | **Yes** |
| Care about both classes | Yes | Less suitable |
| Fraud / disease detection | No | **Yes** |

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

ap = average_precision_score(y_true, y_prob)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend(); plt.show()
```

---

## 1.10 Log Loss (Binary Cross-Entropy)

**Formula:**

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Multi-class generalization:**

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})$$

**What it measures:** How well the **predicted probabilities** match the true distribution. Unlike accuracy (which only cares about the argmax), log loss rewards **calibrated, confident correct predictions** and heavily punishes confident wrong predictions.

| Prediction | True Label | Log Loss Contribution |
|------------|------------|----------------------|
| $\hat{p} = 0.99$ | 1 | $-\log(0.99) = 0.01$ (very low — good) |
| $\hat{p} = 0.51$ | 1 | $-\log(0.51) = 0.67$ (moderate) |
| $\hat{p} = 0.01$ | 1 | $-\log(0.01) = 4.61$ (catastrophic) |

**When to use:**
- When probability calibration matters (insurance risk scoring, medical diagnosis)
- As a training loss for logistic regression and neural networks
- When comparing models on Kaggle classification competitions (common metric)

```python
from sklearn.metrics import log_loss

ll = log_loss(y_true, y_prob)
print(f"Log Loss: {ll:.4f}")
```

---

## 1.11 When to Use Which Classification Metric

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| Balanced binary | Accuracy, F1 | All errors equally bad |
| Imbalanced binary (rare positive) | **PR-AUC**, Recall, F1 | ROC-AUC can be misleading |
| Cost-sensitive (FP ≠ FN cost) | F-beta, custom cost | Align metric with business cost |
| Probability ranking needed | AUC-ROC | Threshold-independent ranking |
| Calibrated probabilities needed | **Log Loss** | Penalizes overconfidence |
| Multi-class balanced | Macro-F1, Accuracy | Per-class fairness |
| Multi-class imbalanced | **Macro-F1**, Weighted-F1 | Don't let majority class dominate |
| Multi-label | Micro-F1, Hamming Loss | Each sample can have multiple labels |
| Information retrieval | Precision@k, MAP | Ranked list of results |

---

# 2. Regression Metrics

---

## 2.1 Mean Squared Error (MSE)

**Formula:**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Penalizes **large errors disproportionately** (quadratic penalty)
- Units: squared units of target (e.g., dollars²)
- Corresponds to MLE under Gaussian noise assumption
- **Use when:** large errors are especially bad, no major outliers

---

## 2.2 Root Mean Squared Error (RMSE)

**Formula:**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- Same as MSE but **in original units** of the target → more interpretable
- Still penalizes large errors heavily
- **Most commonly reported** regression metric in practice and competitions

---

## 2.3 Mean Absolute Error (MAE)

**Formula:**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Linear penalty → **robust to outliers**
- Corresponds to predicting the **median**
- Units: same as target
- **Use when:** dataset has outliers, all errors equally important

---

## 2.4 Mean Absolute Percentage Error (MAPE)

**Formula:**

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- **Scale-independent** — expressed as a percentage → easy to interpret and compare across different scales
- **Problem:** undefined when $y_i = 0$; biased toward underestimation (denominator is actual value)
- **Use when:** stakeholders want error as a percentage (e.g., "model is off by 8% on average")

---

## 2.5 R-Squared ($R^2$)

**Formula:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

where $SS_{res}$ = residual sum of squares, $SS_{tot}$ = total sum of squares, $\bar{y}$ = mean of $y$.

**Interpretation:**
- The **proportion of variance** in the target explained by the model
- $R^2 = 1$: perfect prediction
- $R^2 = 0$: model is no better than predicting the mean
- $R^2 < 0$: model is **worse** than predicting the mean (possible with very bad models)

**Limitation:** $R^2$ **never decreases** when you add more features — even useless ones. This can misleadingly inflate the score as model complexity grows.

---

## 2.6 Adjusted R-Squared

**Formula:**

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where $n$ = number of samples, $p$ = number of predictors.

- **Penalizes adding irrelevant features** — only increases if the new feature improves the model more than expected by chance
- **Use when:** comparing models with different numbers of features
- Standard in linear regression model selection

---

## 2.7 Regression Metrics Comparison

| Metric | Formula | Units | Outlier Sensitive? | Interpretability | Best For |
|--------|---------|-------|-------------------|-----------------|----------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | Squared | **Very** | Low (squared units) | Optimization objective |
| RMSE | $\sqrt{MSE}$ | Original | **Very** | **High** | General reporting |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Original | **Robust** | **High** | Robust estimation |
| MAPE | $\frac{100}{n}\sum|\frac{y-\hat{y}}{y}|$ | Percent | Moderate | **Very high** | Business reporting |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | Unitless | Moderate | **High** | Explained variance |
| Adj. $R^2$ | Penalized $R^2$ | Unitless | Moderate | **High** | Feature selection |

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}, R²: {r2:.4f}")
```

---

# 3. Text Generation Metrics

---

## 3.1 BLEU (Bilingual Evaluation Understudy)

**Purpose:** Measure **precision** of generated n-grams against reference text. Originally designed for **machine translation**.

### Formula

**Modified n-gram precision** (clipped to avoid counting repeated words):

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min\left(\text{Count}(\text{n-gram}, \hat{y}),\; \text{Count}(\text{n-gram}, y)\right)}{\sum_{\text{n-gram} \in \hat{y}} \text{Count}(\text{n-gram}, \hat{y})}$$

**Brevity Penalty (BP):** Penalizes short translations that achieve high precision by omitting content:

$$BP = \begin{cases} 1 & \text{if } |\hat{y}| > |y| \\ e^{1 - |y|/|\hat{y}|} & \text{if } |\hat{y}| \leq |y| \end{cases}$$

**Final BLEU score (BLEU-4 typically):**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where $w_n = \frac{1}{N}$ (uniform weights), $N = 4$ by default.

### Interpretation

| BLEU Score | Quality |
|------------|---------|
| > 0.40 | High quality, near human |
| 0.30–0.40 | Understandable, good |
| 0.20–0.30 | Gist preserved, imperfect |
| < 0.20 | Poor quality |

### Key Properties

- **Precision-based** — measures how much of the generated text appears in the reference
- **Does NOT capture recall** — a short, highly precise output scores well
- **Brevity penalty** compensates for the lack of recall
- **n-gram matching only** — no understanding of synonyms or paraphrasing
- **Corpus-level metric** — designed to average over many sentences, less reliable on single examples

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]

bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))    # Unigram
bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
print(f"BLEU-1: {bleu_1:.4f}, BLEU-4: {bleu_4:.4f}")
```

---

## 3.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Purpose:** Measure **recall** of reference content in the generated text. Designed for **summarization** evaluation.

### Variants

| Variant | What It Measures | Formula |
|---------|-----------------|---------|
| **ROUGE-1** | Unigram overlap | $\frac{\text{Matching unigrams}}{\text{Total unigrams in reference}}$ |
| **ROUGE-2** | Bigram overlap | $\frac{\text{Matching bigrams}}{\text{Total bigrams in reference}}$ |
| **ROUGE-L** | Longest Common Subsequence (LCS) | $F_{\text{LCS}} = \frac{(1 + \beta^2) \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 \cdot P_{\text{LCS}}}$ |
| ROUGE-S | Skip-bigram overlap | Allows gaps between words |

### ROUGE-L Detail

$$R_{\text{LCS}} = \frac{\text{LCS}(y, \hat{y})}{|y|} \quad\quad P_{\text{LCS}} = \frac{\text{LCS}(y, \hat{y})}{|\hat{y}|}$$

- LCS captures **sentence-level word ordering** without requiring consecutive matches
- More flexible than n-gram metrics — accounts for word reordering
- Does not require contiguous matching

### BLEU vs. ROUGE

| Aspect | BLEU | ROUGE |
|--------|------|-------|
| Orientation | **Precision** (what % of generated text is correct) | **Recall** (what % of reference content is captured) |
| Primary domain | Translation | Summarization |
| Unit | Precision of n-grams in candidate | Recall of n-grams from reference |
| Penalty | Brevity penalty for short outputs | None (recall naturally rewards longer outputs) |
| Key limitation | Ignores synonyms | Ignores precision unless F-score variant used |

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat near the window."
candidate = "A cat was sitting on the mat by the window."

scores = scorer.score(reference, candidate)
for key, value in scores.items():
    print(f"{key}: Precision={value.precision:.3f}, Recall={value.recall:.3f}, F1={value.fmeasure:.3f}")
```

---

## 3.3 METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Purpose:** Address BLEU's limitations by incorporating **synonyms, stemming, and word order**.

### How It Works

1. **Alignment:** Create word-to-word alignment between candidate and reference using:
   - Exact string match
   - Stemmed match (e.g., "running" → "run")
   - Synonym match (via WordNet)
   - Paraphrase match (from paraphrase tables)

2. **Unigram precision and recall:**

$$P = \frac{\text{matched unigrams}}{|\hat{y}|} \quad\quad R = \frac{\text{matched unigrams}}{|y|}$$

3. **Parameterized F-score** (recall-weighted):

$$F = \frac{10 \cdot P \cdot R}{R + 9P}$$

4. **Fragmentation penalty** (penalizes disordered matches):

$$\text{Penalty} = 0.5 \cdot \left(\frac{\text{chunks}}{\text{matched unigrams}}\right)^3$$

where "chunks" = the minimum number of contiguous aligned groups.

5. **Final score:**

$$\text{METEOR} = F \cdot (1 - \text{Penalty})$$

**Advantages over BLEU:**
- Captures synonyms ("big" ↔ "large") and morphological variants
- Word-order penalty captures fluency
- Better correlation with human judgment at sentence level

---

## 3.4 BERTScore

**Purpose:** Use **contextual embeddings** to measure semantic similarity between generated and reference text. The **modern** evaluation metric that captures meaning beyond surface-level word overlap.

### How It Works

1. **Encode** both candidate and reference sentences using a pre-trained model (e.g., `roberta-large`)
2. For each token in the candidate, find the **maximum cosine similarity** with any token in the reference (and vice versa)
3. Compute:

$$P_{\text{BERT}} = \frac{1}{|\hat{y}|} \sum_{\hat{x} \in \hat{y}} \max_{x \in y} \cos(\mathbf{h}_{\hat{x}}, \mathbf{h}_{x})$$

$$R_{\text{BERT}} = \frac{1}{|y|} \sum_{x \in y} \max_{\hat{x} \in \hat{y}} \cos(\mathbf{h}_{x}, \mathbf{h}_{\hat{x}})$$

$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

### Key Properties

- Captures **paraphrases and semantic equivalence** ("vehicle" ≈ "car")
- Uses **contextual** embeddings (not just word2vec) — "bank" in "river bank" vs. "bank account" gets different representations
- **Higher correlation** with human judgment than BLEU/ROUGE
- **Limitation:** Computationally expensive; results depend on the underlying model

```python
from bert_score import score as bert_score

cands = ["A cat was sitting on the mat."]
refs  = ["The cat sat on the mat."]

P, R, F1 = bert_score(cands, refs, lang="en", model_type="roberta-large")
print(f"BERTScore — P: {P.mean():.4f}, R: {R.mean():.4f}, F1: {F1.mean():.4f}")
```

---

## 3.5 Perplexity (PPL)

**Purpose:** Measure how **fluent** or **surprised** a language model is by a text sequence. Lower perplexity = the model finds the text more predictable = better language modeling.

### Formula

Given a sequence of tokens $w_1, w_2, \ldots, w_N$:

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i})\right) = e^{\text{Cross-Entropy}}$$

Equivalently, perplexity is the **exponentiation of the average cross-entropy loss**:

$$\text{PPL} = e^{H} = e^{CE}$$

### Interpretation

| Perplexity | Meaning |
|-----------|---------|
| 1 | Perfect — model is 100% certain of every next token |
| 10 | On average, the model is "choosing" between ~10 equally likely tokens |
| 100 | High uncertainty — model struggles with the text |
| ↓ Lower | Better language model (more fluent, less surprised) |

### Key Properties

- **Intrinsic metric** — measures the model's likelihood assignment, not task quality
- Used to compare language models (GPT-2 vs. GPT-3, different training configs)
- **Limitation:** A model can have low perplexity but generate repetitive or irrelevant text. Perplexity doesn't measure factuality, relevance, or usefulness.
- **Not comparable** across different tokenizers (different vocabulary → different perplexity baseline)

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    perplexity = torch.exp(outputs.loss)

print(f"Perplexity: {perplexity.item():.2f}")
```

---

## 3.6 Text Generation Metrics Comparison

| Metric | Type | Captures Meaning? | Speed | Primary Domain | Human Correlation |
|--------|------|-------------------|-------|----------------|-------------------|
| BLEU | N-gram precision | No | Fast | Translation | Moderate |
| ROUGE | N-gram recall | No | Fast | Summarization | Moderate |
| METEOR | Semantic alignment | Partial (synonyms) | Medium | Translation | Good |
| BERTScore | Embedding similarity | **Yes** | Slow | General NLG | **High** |
| Perplexity | Model likelihood | No (fluency only) | Fast | Language modeling | Low (indirect) |

---

# 4. Retrieval Metrics

---

## 4.1 Precision@k

**Formula:**

$$\text{Precision@}k = \frac{\text{Number of relevant documents in top-}k}{k}$$

**Example:** Out of top-5 retrieved documents, 3 are relevant → Precision@5 = 3/5 = 0.60.

**Use when:** You care about the **quality** of retrieved results (e.g., search engine result page).

---

## 4.2 Recall@k

**Formula:**

$$\text{Recall@}k = \frac{\text{Number of relevant documents in top-}k}{\text{Total number of relevant documents}}$$

**Example:** There are 10 relevant documents total, and 3 appear in the top-5 → Recall@5 = 3/10 = 0.30.

**Use when:** You need to ensure **coverage** — critical for RAG systems where missing a relevant document means missing evidence for the answer.

---

## 4.3 Mean Reciprocal Rank (MRR)

**Formula:**

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the **first** relevant document for query $i$.

**Example:**
- Query 1: first relevant doc at rank 3 → $\frac{1}{3}$
- Query 2: first relevant doc at rank 1 → $\frac{1}{1}$
- Query 3: first relevant doc at rank 2 → $\frac{1}{2}$
- MRR = $\frac{1}{3}(\frac{1}{3} + 1 + \frac{1}{2}) = 0.611$

**Use when:** Only the **first correct result matters** — question answering, "I'm Feeling Lucky" scenarios, conversational search.

---

## 4.4 Normalized Discounted Cumulative Gain (nDCG)

nDCG is the gold standard for evaluating **ranked retrieval** with **graded relevance** (relevance = 0, 1, 2, 3, ...).

### Step 1: Discounted Cumulative Gain (DCG)

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

- Documents at **higher ranks contribute more** (logarithmic discount)
- Higher relevance scores contribute exponentially more ($2^{rel} - 1$)

### Step 2: Ideal DCG (IDCG)

Sort all documents by relevance in descending order and compute DCG on this ideal ranking:

$$\text{IDCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)}$$

### Step 3: Normalize

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

**Range:** $[0, 1]$ — 1.0 means the ranking is perfectly ordered by relevance.

### Worked Example

| Rank | Document | Relevance | $\frac{2^{rel}-1}{\log_2(rank+1)}$ |
|------|----------|-----------|--------------------------------------|
| 1 | Doc A | 3 | $\frac{7}{1.0} = 7.00$ |
| 2 | Doc B | 2 | $\frac{3}{1.585} = 1.89$ |
| 3 | Doc C | 0 | $\frac{0}{2.0} = 0.00$ |
| 4 | Doc D | 1 | $\frac{1}{2.322} = 0.43$ |
| 5 | Doc E | 3 | $\frac{7}{2.585} = 2.71$ |

DCG@5 = 7.00 + 1.89 + 0.00 + 0.43 + 2.71 = **12.03**

Ideal order: [3, 3, 2, 1, 0] → IDCG@5 = 7.00 + 4.42 + 1.50 + 0.43 + 0.00 = **13.35**

nDCG@5 = 12.03 / 13.35 = **0.901**

**Use when:** You have **graded relevance labels** and care about ranking quality — web search, recommendation systems.

---

## 4.5 Hit Rate@k

**Formula:**

$$\text{Hit Rate@}k = \frac{\text{Number of queries with at least one relevant doc in top-}k}{\text{Total number of queries}}$$

- Binary: did the system return *anything* relevant in the top-k?
- Simplest retrieval metric — often the first thing measured for RAG systems
- **Use when:** You just need to know if the retriever is surfacing relevant documents at all

---

## 4.6 Mean Average Precision (MAP)

### Average Precision (AP) for a Single Query

$$\text{AP} = \frac{1}{\text{|relevant docs|}} \sum_{k=1}^{n} \text{Precision@}k \cdot \mathbb{1}[\text{doc}_k \text{ is relevant}]$$

Compute Precision@k only at positions where a relevant document appears, then average.

### MAP

$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$

### Worked Example

Query: 5 documents retrieved, relevance = [1, 0, 1, 0, 1], total relevant = 3

| k | Relevant? | Precision@k | Include? |
|---|-----------|-------------|----------|
| 1 | Yes | 1/1 = 1.00 | Yes |
| 2 | No | 1/2 = 0.50 | No |
| 3 | Yes | 2/3 = 0.67 | Yes |
| 4 | No | 2/4 = 0.50 | No |
| 5 | Yes | 3/5 = 0.60 | Yes |

AP = (1.00 + 0.67 + 0.60) / 3 = **0.756**

**Use when:** You want a **single number** summarizing ranking quality across multiple queries — standard in information retrieval benchmarks (TREC).

---

## 4.7 Retrieval Metrics Comparison

| Metric | Graded Relevance? | Position Sensitive? | Aggregation | Best For |
|--------|-------------------|--------------------|-----------  |----------|
| Precision@k | Binary | No (within top-k) | Per-query | Result page quality |
| Recall@k | Binary | No | Per-query | Coverage / RAG |
| MRR | Binary | **Yes** (first hit) | Across queries | QA, single-answer |
| nDCG@k | **Yes** (graded) | **Yes** (log decay) | Per-query | Web search, recommendations |
| Hit Rate@k | Binary | No | Across queries | RAG sanity check |
| MAP | Binary | **Yes** | Across queries | IR benchmarks |

---

# 5. Object Detection Metrics

---

## 5.1 Intersection over Union (IoU)

**Formula:**

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$$

```
┌──────────────┐
│  Ground Truth │
│    ┌─────┼────────┐
│    │/////│        │
└────┼─────┘        │
     │   Predicted  │
     └──────────────┘

     Overlap = //// region
     Union   = total shaded area of both boxes
```

**Range:** $[0, 1]$ — 0 means no overlap, 1 means perfect overlap.

**IoU threshold for a "correct" detection:**

| Threshold | Meaning |
|-----------|---------|
| IoU ≥ 0.5 | Standard (Pascal VOC) |
| IoU ≥ 0.75 | Strict |
| IoU ∈ [0.5, 0.95] step 0.05 | COCO evaluation (averaged) |

---

## 5.2 mAP (Mean Average Precision) for Detection

### Pascal VOC: mAP@0.5

1. For each class, rank detections by confidence score
2. A detection is TP if IoU ≥ 0.5 with a ground-truth box (each GT matched only once)
3. Compute Precision-Recall curve per class
4. Calculate AP = area under PR curve (using 11-point interpolation or all-point interpolation)
5. **mAP@0.5** = mean of AP across all classes

### COCO: mAP@[.5:.95]

1. Compute AP at 10 IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95
2. Average across all thresholds **and** all classes

$$\text{mAP}_{COCO} = \frac{1}{10} \sum_{t \in \{0.5, 0.55, \ldots, 0.95\}} \text{mAP}@t$$

**Why COCO is stricter:** At IoU = 0.5, a loosely aligned box counts as correct. COCO demands precision at higher IoU thresholds, rewarding tight, well-localized predictions.

### Comparison

| Protocol | IoU Threshold(s) | Stringency | Used In |
|----------|-------------------|------------|---------|
| Pascal VOC | 0.5 | Lenient | VOC challenges, quick benchmarks |
| COCO | 0.50 to 0.95 (step 0.05) | **Strict** | COCO benchmark, production models |
| COCO AP-small | Same, objects < 32² px | Very strict | Small object detection |
| COCO AP-large | Same, objects > 96² px | Moderate | Large object detection |

```python
# Using torchmetrics
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
preds = [{"boxes": torch.tensor([[100, 100, 200, 200]]),
          "scores": torch.tensor([0.9]),
          "labels": torch.tensor([0])}]
targets = [{"boxes": torch.tensor([[105, 105, 195, 195]]),
            "labels": torch.tensor([0])}]

metric.update(preds, targets)
result = metric.compute()
print(f"mAP@0.5: {result['map_50']:.4f}, mAP@0.75: {result['map_75']:.4f}")
```

---

# 6. LLM-Specific Metrics

---

## 6.1 Faithfulness / Hallucination Detection

**What it measures:** Does the generated answer faithfully reflect the source material (retrieved context), or does it contain **hallucinated** claims?

### Types of Hallucination

| Type | Definition | Example |
|------|-----------|---------|
| **Intrinsic** | Contradicts the source | Source: "Founded in 2015" → Output: "Founded in 2012" |
| **Extrinsic** | Adds information not in the source | Source mentions only revenue → Output adds employee count |
| **Factual** | States something objectively false | "Paris is the capital of Germany" |

### Measuring Faithfulness

**1. NLI-Based (Natural Language Inference):**
- Decompose the answer into atomic claims
- For each claim, check if it is **entailed** by the context using an NLI model
- Faithfulness = fraction of entailed claims

$$\text{Faithfulness} = \frac{\text{Number of entailed claims}}{\text{Total claims in answer}}$$

**2. LLM-as-Judge:**
- Prompt a strong LLM (e.g., GPT-4) to evaluate whether the answer is supported by the context
- Typically uses a structured rubric (1–5 scale or binary yes/no per claim)

**3. Token-Level Overlap:**
- Measure how much of the answer can be traced back to the context via token overlap or BERTScore

### Frameworks

| Framework | Method | Open Source? |
|-----------|--------|-------------|
| RAGAS | NLI + LLM judge | Yes |
| TruLens | Multiple evaluators | Yes |
| DeepEval | LLM-based | Yes |
| Vectara HHEM | Hallucination evaluation model | Yes |

---

## 6.2 Citation Precision & Recall

For systems that generate answers **with citations** (e.g., "[1], [2]"):

**Citation Precision:**

$$\text{Citation Precision} = \frac{\text{Citations that actually support the claim}}{\text{Total citations generated}}$$

High precision = the model doesn't cite irrelevant sources.

**Citation Recall:**

$$\text{Citation Recall} = \frac{\text{Claims supported by at least one correct citation}}{\text{Total claims in the answer}}$$

High recall = every claim in the answer is properly attributed.

**Why it matters:** In production RAG systems, users trust the system based on citations. Incorrect citations erode trust faster than missing ones.

---

## 6.3 Human Evaluation (HHH Framework)

For LLMs, automated metrics often fail to capture what humans care about. The **HHH** framework:

| Dimension | What It Measures | Example Rubric |
|-----------|-----------------|----------------|
| **Helpfulness** | Does the response actually answer the question? Is it complete and useful? | 1 = irrelevant, 5 = thorough and actionable |
| **Harmlessness** | Is the response safe? Free from toxic, biased, or dangerous content? | Binary (safe / unsafe) or graded |
| **Honesty** | Does the model acknowledge uncertainty? Avoid fabrication? | 1 = confidently wrong, 5 = calibrated and truthful |

### Evaluation Methods

| Method | Description | Cost |
|--------|-------------|------|
| Likert scale | Rate each dimension 1–5 | Medium |
| Pairwise comparison | "Which response is better: A or B?" | Lower bias |
| Best-of-N ranking | Rank N responses from best to worst | Most informative |
| Win rate | % of times model A preferred over B | Simple to communicate |
| Elo rating | Chess-style rating from many pairwise comparisons | Chatbot Arena approach |

---

## 6.4 Answer Confidence via Log Probabilities

**What it measures:** How confident the model is in each generated token, using the raw **log-probabilities** from the LLM.

**Sequence-level confidence:**

$$\text{Confidence}(\hat{y}) = \exp\left(\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i})\right)$$

This is the geometric mean of token-level probabilities — equivalent to perplexity inverted.

**Use cases:**
- **Selective generation:** Only show answers above a confidence threshold
- **Abstention:** "I'm not sure" when confidence is below threshold
- **Routing:** Low-confidence queries → human review
- **Calibration analysis:** Are 90%-confident answers correct 90% of the time?

**Limitation:** LLMs are often **poorly calibrated** — they can be highly confident and wrong (especially after RLHF which squashes logprobs toward confident outputs).

---

# 7. RAG-Specific Metrics

---

## 7.1 Document Ingestion Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Coverage %** | $\frac{\text{Documents successfully ingested}}{\text{Total documents in corpus}} \times 100$ | Are all documents processed? Missing docs = missing knowledge |
| **Dedup Ratio** | $\frac{\text{Unique chunks}}{\text{Total chunks before dedup}}$ | Near-duplicate chunk elimination. Ratio close to 1.0 = few duplicates |
| **Chunk Overlap Recall** | $\frac{\text{Key passages appearing in at least one chunk}}{\text{Total key passages}}$ | Does the chunking strategy preserve complete information? Especially critical at chunk boundaries |

**Why these matter:** If ingestion drops 15% of documents silently, even a perfect retriever can't find answers from missing content. Ingestion quality is the foundation.

---

## 7.2 Retrieval-Stage Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Hit-Rate@k** | % of queries where at least one relevant doc is in top-k | > 0.90 for production |
| **nDCG@k** | Ranking quality with graded relevance | > 0.75 |
| **MRR@k** | How high the first relevant doc appears | > 0.70 |
| **Latency P95** | 95th-percentile retrieval time | < 200ms for real-time |

**Retrieval is the bottleneck:** If the retriever fails, the generator cannot compensate. A "garbage in, garbage out" principle applies.

---

## 7.3 Generation-Stage Metrics

| Metric | Formula / Description | What It Measures |
|--------|----------------------|-----------------|
| **Exact Match (EM)** | $\mathbb{1}[\hat{y} = y]$ after normalization | Strict correctness (extractive QA) |
| **Token-F1** | F1 over token overlap between prediction and reference | Partial credit for token matches |
| **ROUGE-L** | LCS-based F1 | Content overlap for longer answers |
| **Faithfulness** | NLI entailment ratio | Hallucination rate |
| **Citation Accuracy** | Are citations correct and complete? | Source attribution quality |

### Token-F1 Detail

$$P_{tok} = \frac{|\text{pred tokens} \cap \text{ref tokens}|}{|\text{pred tokens}|} \quad\quad R_{tok} = \frac{|\text{pred tokens} \cap \text{ref tokens}|}{|\text{ref tokens}|}$$

$$F1_{tok} = \frac{2 \cdot P_{tok} \cdot R_{tok}}{P_{tok} + R_{tok}}$$

**Use when:** Extractive QA (SQuAD-style) where the answer is a short span.

---

## 7.4 OCR Quality Metrics

When RAG systems ingest scanned documents or images, OCR quality directly affects downstream performance.

### Character Error Rate (CER)

$$\text{CER} = \frac{S + D + I}{N}$$

where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total characters in ground truth.

### Word Error Rate (WER)

$$\text{WER} = \frac{S_w + D_w + I_w}{N_w}$$

Same formula at the **word** level.

| Metric | Level | Good Value | Impact on RAG |
|--------|-------|------------|---------------|
| CER | Character | < 2% | Misspellings break exact retrieval |
| WER | Word | < 5% | Misrecognized words = missing information |

**Why it matters for RAG:** OCR errors propagate through the pipeline — a misspelled entity won't match during retrieval, and the LLM may generate incorrect information from garbled text.

---

## 7.5 End-to-End RAG Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG EVALUATION FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  INGESTION    │    │   RETRIEVAL       │    │   GENERATION              │  │
│  │               │    │                   │    │                           │  │
│  │ • Coverage %  │───▶│ • Hit-Rate@k     │───▶│ • Exact Match             │  │
│  │ • Dedup Ratio │    │ • nDCG@k         │    │ • Token-F1                │  │
│  │ • Chunk Recall│    │ • MRR@k          │    │ • ROUGE-L                 │  │
│  │ • OCR CER/WER│    │ • Latency P95    │    │ • Faithfulness            │  │
│  │               │    │ • Precision@k    │    │ • Citation Accuracy       │  │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘  │
│         │                     │                         │                   │
│         ▼                     ▼                         ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    END-TO-END METRICS                                │   │
│  │  • Answer Correctness (EM + F1)                                     │   │
│  │  • User Satisfaction (thumbs up/down, CSAT)                         │   │
│  │  • Task Completion Rate                                              │   │
│  │  • Total Latency (retrieval + generation)                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 8. A/B Testing Metrics

---

## 8.1 Statistical Significance

**Definition:** A result is statistically significant if it is **unlikely to have occurred by random chance** alone.

In A/B testing, you're comparing a **control group (A)** and a **treatment group (B)** to determine if the observed difference in a metric (conversion rate, CTR, revenue, etc.) is real or due to noise.

---

## 8.2 Hypothesis Testing Framework

| Component | Description |
|-----------|-------------|
| **Null Hypothesis ($H_0$)** | No difference between A and B ($\mu_A = \mu_B$) |
| **Alternative Hypothesis ($H_1$)** | There is a difference ($\mu_A \neq \mu_B$) |
| **Test Statistic** | Standardized measure of the observed difference |
| **p-value** | Probability of observing this result (or more extreme) if $H_0$ is true |
| **Significance Level ($\alpha$)** | Threshold for rejecting $H_0$ (typically 0.05) |

---

## 8.3 p-value

**Formula (for z-test on proportions):**

$$z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}$$

where $\hat{p} = \frac{x_A + x_B}{n_A + n_B}$ is the pooled proportion.

**Interpretation:**

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| p < 0.01 | Strong rejection of $H_0$ | Very strong evidence of a real difference |
| p < 0.05 | Reject $H_0$ | Standard threshold — statistically significant |
| 0.05 ≤ p < 0.10 | Marginal | Suggestive but not conclusive |
| p ≥ 0.10 | Fail to reject $H_0$ | No significant evidence of difference |

**Common misconception:** p-value is NOT the probability that $H_0$ is true. It's the probability of seeing data this extreme *assuming* $H_0$ is true.

---

## 8.4 Effect Size

**Why it matters:** A result can be statistically significant but **practically insignificant**. With a large enough sample, even a 0.01% difference in conversion rate will be "significant." Effect size tells you **how big** the difference actually is.

### Cohen's d (for continuous outcomes):

$$d = \frac{\bar{X}_B - \bar{X}_A}{s_p}$$

where $s_p = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$ is the pooled standard deviation.

| Cohen's d | Effect Size |
|-----------|-------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Relative Lift (for proportions):

$$\text{Lift} = \frac{\hat{p}_B - \hat{p}_A}{\hat{p}_A} \times 100\%$$

---

## 8.5 Sample Size Calculation

Before running a test, calculate the **minimum sample size** needed to detect a meaningful effect.

**Formula (for proportions):**

$$n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \cdot 2\hat{p}(1 - \hat{p})}{(\delta)^2}$$

where:
- $Z_{\alpha/2} \approx 1.96$ for 95% confidence
- $Z_\beta \approx 0.84$ for 80% power
- $\delta$ = minimum detectable effect (MDE)
- $\hat{p}$ = baseline conversion rate

**Key trade-offs:**

| Want to detect... | Requires... |
|-------------------|-------------|
| Smaller effect ($\delta \downarrow$) | Larger sample size |
| Higher confidence ($\alpha \downarrow$) | Larger sample size |
| Higher power ($1 - \beta \uparrow$) | Larger sample size |

---

## 8.6 Type I and Type II Errors

| Error Type | Definition | A/B Testing Context | Controlled By |
|------------|-----------|---------------------|---------------|
| **Type I ($\alpha$)** | Reject $H_0$ when it's true | "Ship a change that has no real effect" | Significance level (typically 5%) |
| **Type II ($\beta$)** | Fail to reject $H_0$ when $H_1$ is true | "Miss a real improvement" | Power = $1 - \beta$ (typically 80%) |

**Statistical Power = $1 - \beta$:** The probability of correctly detecting a true effect. Industry standard: **80%** minimum.

```python
from scipy import stats
import numpy as np

# A/B test: comparing conversion rates
n_A, n_B = 10000, 10000
conv_A, conv_B = 520, 580  # conversions
p_A, p_B = conv_A / n_A, conv_B / n_B

# Pooled proportion
p_pool = (conv_A + conv_B) / (n_A + n_B)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z_stat = (p_B - p_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

lift = (p_B - p_A) / p_A * 100

print(f"Control: {p_A:.2%}, Treatment: {p_B:.2%}")
print(f"Lift: {lift:.1f}%, z-stat: {z_stat:.3f}, p-value: {p_value:.4f}")
print(f"Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
```

---

## 8.7 A/B Testing Best Practices for ML Systems

| Practice | Why |
|----------|-----|
| **Pre-register** your hypothesis and metrics | Avoid p-hacking (testing until something is significant) |
| **Run for full duration** | Don't peek and stop early (inflates Type I error) |
| Use **sequential testing** (if peeking needed) | Adjusts significance threshold for multiple looks |
| Test **one change at a time** | Isolate the causal effect |
| Check for **novelty effects** | Users may engage with anything new temporarily |
| Guard rails: **monitor harm metrics** | Stop if safety metrics degrade significantly |
| Segment analysis | Check if effect holds across user segments |

---

# 9. Summary Tables by Domain

---

## 9.1 Metrics by ML Task

| Task | Primary Metrics | Secondary Metrics | Watch Out For |
|------|----------------|-------------------|---------------|
| **Binary Classification** | PR-AUC, F1, Recall | ROC-AUC, Log Loss | Class imbalance → don't use accuracy |
| **Multi-class Classification** | Macro-F1, Weighted-F1 | Accuracy, Confusion Matrix | Majority class dominance |
| **Regression** | RMSE, MAE | R², MAPE | Outliers inflate MSE/RMSE |
| **Ranking / Recommendation** | nDCG@k, MAP | MRR, Hit Rate@k | Position bias in training data |
| **Machine Translation** | BLEU, METEOR | BERTScore, TER | BLEU unreliable for single sentences |
| **Summarization** | ROUGE-L, ROUGE-2 | BERTScore, Faithfulness | ROUGE misses semantic equivalence |
| **Question Answering** | EM, Token-F1 | BERTScore | Exact Match is very strict |
| **Object Detection** | mAP@[.5:.95] | mAP@0.5, AP per class | Small vs. large object performance |
| **Language Modeling** | Perplexity | Bits per character | Not comparable across tokenizers |
| **RAG System** | Faithfulness, Hit-Rate@k | ROUGE-L, nDCG, Latency | Evaluate retrieval and generation separately |
| **LLM Chatbot** | Human eval (HHH), Win rate | Elo rating | Automated metrics poorly correlate with usefulness |

---

## 9.2 Metrics by Business Objective

| Business Goal | Metric to Optimize | Example |
|---------------|-------------------|---------|
| Minimize false alarms | **Precision** | Spam filter, fraud alerts |
| Never miss a case | **Recall** | Cancer screening, security threats |
| Balance precision & recall | **F1-Score** | General classification |
| Rank best results first | **nDCG@k**, **MAP** | Search engine, recommendations |
| Probability calibration | **Log Loss**, Brier Score | Insurance pricing, medical triage |
| User satisfaction | **Human eval**, A/B test | Chatbot, content generation |
| Latency-sensitive | **P95 latency** + quality metric | Real-time retrieval |
| Content fidelity | **Faithfulness**, **CER/WER** | RAG, document processing |

---

## 9.3 Quick-Reference: Metric Formulas

| Metric | Formula | Range |
|--------|---------|-------|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | [0, 1] |
| Precision | $\frac{TP}{TP+FP}$ | [0, 1] |
| Recall | $\frac{TP}{TP+FN}$ | [0, 1] |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | [0, 1] |
| ROC-AUC | Area under TPR vs FPR curve | [0, 1] |
| Log Loss | $-\frac{1}{N}\sum[y\log\hat{p}+(1-y)\log(1-\hat{p})]$ | [0, ∞) |
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | [0, ∞) |
| RMSE | $\sqrt{MSE}$ | [0, ∞) |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | [0, ∞) |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | (-∞, 1] |
| BLEU | BP · exp(Σ wₙ log pₙ) | [0, 1] |
| ROUGE-L | LCS-based F1 | [0, 1] |
| Perplexity | $e^{CE}$ | [1, ∞) |
| IoU | $\frac{Intersection}{Union}$ | [0, 1] |
| nDCG@k | $\frac{DCG@k}{IDCG@k}$ | [0, 1] |
| MRR | $\frac{1}{|Q|}\sum\frac{1}{rank_i}$ | (0, 1] |

---

# 10. Common Interview Questions & Answers

---

### Q1: Your fraud detection model has 99.5% accuracy. Your manager is thrilled. Should you be?

**Answer:** No — this is the **accuracy paradox**. If only 0.5% of transactions are fraudulent, a model that predicts "not fraud" for everything achieves 99.5% accuracy while catching zero fraud.

**What to do instead:**
- Report **Precision, Recall, and F1** for the fraud class specifically
- Use **PR-AUC** (not ROC-AUC, which is inflated under imbalance)
- Look at the confusion matrix: how many actual fraud cases did we catch (TP) vs. miss (FN)?
- Consider business cost: missing one fraud case ($10K) may cost more than 100 false alerts ($50 each)

---

### Q2: When would you choose RMSE over MAE, and vice versa?

**Answer:**
- **RMSE** penalizes large errors quadratically → choose when **large deviations are especially costly** (e.g., predicting delivery time — being off by 2 hours once is worse than being off by 30 minutes four times)
- **MAE** penalizes all errors linearly → choose when **outliers are expected and you don't want them to dominate** (e.g., housing prices with a few mansions)
- **RMSE ≥ MAE** always. The gap between them indicates how many large errors exist. If RMSE >> MAE, you have a few big outlier errors.

---

### Q3: You're evaluating a summarization model. ROUGE-L is 0.45 but the summaries read poorly. What's going on?

**Answer:** ROUGE measures **surface-level overlap** (n-grams, LCS) with a reference, not quality:
- ROUGE can be high if the model copies phrases from the source, even if the summary is **incoherent, redundant, or misses the main point**
- ROUGE doesn't capture **fluency, coherence, or factual accuracy**

**Better approach:**
- Add **BERTScore** for semantic similarity
- Add **faithfulness** evaluation (NLI-based) to check for hallucinations
- Use **human evaluation** for coherence and usefulness
- Consider **LLM-as-judge** for scalable qualitative assessment

---

### Q4: Explain the difference between ROC-AUC and PR-AUC. When is ROC-AUC misleading?

**Answer:**
- **ROC-AUC** plots TPR vs. FPR. When the negative class is enormous (99.9% negative), even a small FPR (say 0.1%) means thousands of false positives. But the ROC curve still looks great because FPR = FP/(FP+TN), and TN is huge.
- **PR-AUC** plots Precision vs. Recall. Precision = TP/(TP+FP) is **not inflated** by a large TN, so it directly reflects the false positive burden.

**Rule of thumb:** For imbalanced datasets (fraud, rare disease, anomaly detection), **always use PR-AUC**.

---

### Q5: How does nDCG differ from MAP, and when would you use each?

**Answer:**
- **MAP** uses **binary relevance** (relevant or not). It's the average of Precision@k at each relevant position.
- **nDCG** supports **graded relevance** (0, 1, 2, 3). It uses logarithmic position discounting and exponential relevance gain.

**When to use which:**
- If you only have binary labels → MAP
- If you have relevance grades (e.g., "perfect", "good", "fair", "bad") → **nDCG**
- nDCG is standard in modern search/recommendation systems
- MAP is standard in traditional IR benchmarks (TREC)

---

### Q6: How would you evaluate a RAG system end-to-end?

**Answer:** Evaluate **each stage independently** plus the end-to-end pipeline:

1. **Ingestion:** Coverage %, dedup ratio, chunk overlap recall, OCR CER/WER
2. **Retrieval:** Hit-Rate@k (≥ 0.90), nDCG@k, MRR, Latency P95 (< 200ms)
3. **Generation:** EM, Token-F1, ROUGE-L, Faithfulness (NLI-based, ≥ 0.85), Citation accuracy
4. **End-to-end:** Answer correctness, user satisfaction (thumbs up/down), task completion rate

**Key insight:** If the retriever fails (Hit-Rate < 0.70), no amount of generator quality can save the system. Always debug retrieval first.

---

### Q7: What is perplexity and why can't you compare it across models with different tokenizers?

**Answer:**
- **Perplexity** = $e^{CE}$ = exponential of average cross-entropy loss. It represents how "surprised" the model is by the text. Lower = better.
- A model with perplexity 15 is, on average, choosing between ~15 equally likely tokens.

**Tokenizer dependence:** Different tokenizers split text into different numbers of tokens. A BPE tokenizer might produce 100 tokens for a sentence while a character-level tokenizer produces 500. Since CE is averaged per token, the denominator changes — making the scores incomparable. It's like comparing "miles per gallon" when one car measures in liters.

---

### Q8: Your A/B test shows a 2% lift with p = 0.03. Should you ship it?

**Answer:** Statistically significant (p < 0.05), but consider:

1. **Effect size:** Is 2% lift meaningful for the business? A 2% increase in CTR might be huge for ads but trivial for a settings page.
2. **Practical significance:** Does the improvement justify the engineering cost to maintain the change?
3. **Multiple testing:** Did you run multiple tests? If you tested 20 metrics, one will be significant by chance at α = 0.05 (Bonferroni correction needed).
4. **Segment analysis:** Does the lift hold across all user segments, or is it driven by one subgroup?
5. **Guardrail metrics:** Did any harm metrics (latency, error rate, churn) degrade?
6. **Duration:** Was the test run long enough to account for day-of-week effects and novelty?

**Ship only if:** the lift is practically meaningful, guardrails are clean, and the result is consistent across segments.

---

### Q9: How do you evaluate hallucination in LLMs? What's the state of the art?

**Answer:**
Hallucination evaluation has three main approaches:

1. **NLI-based decomposition** (most principled): Break the answer into atomic claims, use an NLI model (e.g., DeBERTa fine-tuned on MNLI) to check if each claim is entailed by the source context. Faithfulness = % entailed.

2. **LLM-as-Judge** (most scalable): Use GPT-4 / Claude to evaluate faithfulness with a rubric. Surprisingly high correlation with human judgment when well-prompted, but introduces model bias and cost.

3. **Specialized models** (emerging): Vectara's HHEM (Hughes Hallucination Evaluation Model), fine-tuned specifically for hallucination detection.

**State of the art:** Frameworks like **RAGAS** combine multiple signals (faithfulness, relevance, answer correctness) and are becoming the standard for RAG evaluation. However, **no single automated metric reliably catches all hallucination types** — human evaluation remains the gold standard for high-stakes applications.

---

### Q10: When would you use BERTScore over BLEU or ROUGE?

**Answer:**
- **BERTScore** captures **semantic similarity** via contextual embeddings — "automobile" and "car" score high even though they share no n-grams
- **BLEU/ROUGE** rely on exact token overlap — they miss valid paraphrases entirely

**Use BERTScore when:**
- Evaluating **open-ended generation** where many valid phrasings exist
- Comparing **paraphrased** or **abstractive** outputs
- You need better **correlation with human judgment**

**Use BLEU/ROUGE when:**
- You need a **fast, well-understood baseline** metric
- Working in established benchmarks that report BLEU/ROUGE (for comparability)
- The task has **constrained outputs** (e.g., translation) where surface overlap is meaningful

---

# 11. Key Takeaways

---

1. **No single metric tells the full story.** Always report multiple complementary metrics. Accuracy alone is dangerous for imbalanced data; ROUGE alone is insufficient for generation quality.

2. **Match the metric to the business objective.** If missing fraud costs $10K but a false alarm costs $5, optimize for recall with an acceptable precision floor — not F1.

3. **Class imbalance changes everything.** Under imbalance: use PR-AUC (not ROC-AUC), Macro-F1 (not accuracy), and always inspect the confusion matrix.

4. **Surface-level text metrics (BLEU, ROUGE) miss semantics.** BERTScore is the modern alternative for evaluating meaning preservation. For LLMs, faithfulness and human evaluation are essential.

5. **Evaluate RAG systems stage by stage.** A brilliant generator can't compensate for a broken retriever. Diagnose failures at the ingestion, retrieval, and generation levels independently.

6. **nDCG is the gold standard for ranked retrieval.** It handles graded relevance and position-sensitive discounting. Use MAP when you only have binary labels.

7. **Statistical significance ≠ practical significance.** A p-value of 0.01 means the effect is real, but it might be tiny. Always report effect size alongside p-values.

8. **LLM evaluation is an open problem.** Automated metrics (perplexity, BERTScore, RAGAS faithfulness) are useful proxies, but human evaluation remains the gold standard for helpfulness, harmlessness, and honesty.

9. **Perplexity measures fluency, not quality.** A model can have low perplexity while producing repetitive, irrelevant, or harmful content. Never use perplexity as the sole evaluation metric for LLM outputs.

10. **For object detection, COCO mAP@[.5:.95] is the industry standard.** It demands precise localization across multiple IoU thresholds, unlike the lenient Pascal VOC mAP@0.5.

---

*Document Version: 1.0 | Last Updated: February 2026*  
*Covers: Classification, Regression, NLG, Retrieval, Detection, LLM, RAG, A/B Testing*
