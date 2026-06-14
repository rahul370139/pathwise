# Explainability & Interpretability — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Model Transparency, Trustworthy ML, Regulatory Compliance

> **Context:** Your `19_classical_ml.md` briefly mentions SHAP for feature selection. This guide provides the full explainability toolkit — critical for regulated industries (finance, healthcare) and increasingly asked at every ML interview.

---

# Table of Contents

1. [Why Explainability Matters](#1-why-explainability-matters)
2. [Interpretable vs Explainable Models](#2-interpretable-vs-explainable-models)
3. [Global Explanations](#3-global-explanations)
4. [Local Explanations](#4-local-explanations)
5. [SHAP Deep Dive](#5-shap-deep-dive)
6. [LIME Deep Dive](#6-lime-deep-dive)
7. [Deep Learning Explainability](#7-deep-learning-explainability)
8. [Model Cards and Documentation](#8-model-cards-and-documentation)
9. [Calibration](#9-calibration)
10. [Interview Questions with Strong Answers](#10-interview-questions-with-strong-answers)

---

# **1. Why Explainability Matters**

| Stakeholder | Why They Need Explanations |
|-------------|---------------------------|
| **Data Scientist** | Debug model, detect data leakage, validate it's learning the right signals |
| **Business Leader** | Trust model decisions, understand what drives outcomes |
| **Regulator** | GDPR "right to explanation," EU AI Act high-risk requirements, US fair lending laws |
| **End User** | Understand why a recommendation/decision was made, build trust |
| **Auditor** | Verify model isn't discriminating, check for bias |

**Real-world stakes:**
- A credit model must explain why an applicant was denied (US Equal Credit Opportunity Act)
- A healthcare model must justify its diagnosis to a doctor
- A hiring model must demonstrate it's not discriminating (NYC Local Law 144)

---

# **2. Interpretable vs Explainable Models**

```
┌──────────────────────────────────────────────────────────────────┐
│         INTERPRETABILITY SPECTRUM                                  │
│                                                                   │
│  INHERENTLY INTERPRETABLE ──────────── POST-HOC EXPLAINABLE     │
│                                                                   │
│  Linear Regression    Decision Tree    Random     XGBoost  DNN   │
│  Logistic Regression  Rule Lists       Forest     LSTM     LLM   │
│  GAM                  Scorecard        SVM                       │
│                                                                   │
│  ◄── You can READ the model ──────── You need TOOLS to ──►     │
│      (coefficients, rules)           explain it (SHAP, LIME)    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

| Approach | Definition | Example |
|----------|-----------|---------|
| **Interpretable** | Model structure is simple enough that humans can directly understand the decision process | Linear regression: "income ×0.3 + age ×0.1 = score" |
| **Explainable** | Model is complex (black-box), but we apply external methods to explain individual predictions | SHAP values for an XGBoost prediction |

**Key insight for interviews:** "Interpretability" and "explainability" are often used interchangeably, but they're different. Interpretability is a property of the model. Explainability is a property of the analysis we do on the model.

---

# **3. Global Explanations**

Global methods explain the model's **overall behavior** — which features matter in general.

## **3.1 Feature Importance (Tree-Based)**

| Method | How | Pros | Cons |
|--------|-----|------|------|
| **Gini/Entropy importance** | How much each feature reduces impurity across all splits | Fast, built-in | Biased toward high-cardinality features |
| **Permutation importance** | Shuffle feature, measure accuracy drop | Model-agnostic, unbiased | Slow, affected by correlated features |
| **SHAP feature importance** | Mean absolute SHAP values across all predictions | Theoretically grounded, handles interactions | Computationally expensive for large datasets |

## **3.2 Partial Dependence Plots (PDP)**

Shows the **marginal effect** of a feature on the prediction, averaging over all other features.

```
Prediction
    │          ╱───────── Feature effect plateaus
    │         ╱
    │        ╱
    │       ╱
    │  ────╱                ← As income increases,
    │                          approval probability increases
    │                          then plateaus around $80K
    └──────────────────────────► Income
```

**Limitation:** Assumes features are independent. If income and education are correlated, PDP can show effects for **impossible combinations** (high income + no education).

## **3.3 Individual Conditional Expectation (ICE)**

Like PDP, but shows one line **per individual** instead of the average. Reveals heterogeneity — does the feature effect differ across subgroups?

## **3.4 Accumulated Local Effects (ALE)**

Fixes PDP's correlation problem by computing effects based on **local changes** rather than marginal averages. More reliable when features are correlated.

---

# **4. Local Explanations**

Local methods explain **individual predictions** — why did the model make THIS decision for THIS input?

## **4.1 Overview**

| Method | Approach | Model-Agnostic? | Pros | Cons |
|--------|----------|-----------------|------|------|
| **SHAP** | Shapley values from game theory | Yes (but tree-optimized variants exist) | Theoretically grounded, consistent | Can be slow |
| **LIME** | Local linear approximation | Yes | Intuitive, works on any model | Unstable, sensitive to perturbation |
| **Counterfactual** | "What minimal change would flip the decision?" | Yes | Actionable ("if income were $5K higher...") | Multiple valid counterfactuals |
| **Anchor** | "What conditions ALWAYS lead to this decision?" | Yes | Rule-based, easy to understand | Can be overly specific |

---

# **5. SHAP Deep Dive**

## **5.1 Shapley Values — The Theory**

From cooperative game theory: how much does each player (feature) contribute to the total payout (prediction)?

**Properties that make Shapley values unique:**
| Property | Meaning |
|----------|---------|
| **Efficiency** | All SHAP values sum to the difference between prediction and baseline |
| **Symmetry** | Two features with identical contributions get equal SHAP values |
| **Dummy** | A feature that doesn't contribute gets SHAP value = 0 |
| **Additivity** | SHAP values of combined models = sum of individual SHAP values |

**Intuition:** For each feature, consider ALL possible subsets of other features. Measure the marginal contribution of adding this feature to each subset. The Shapley value is the **weighted average** of all these marginal contributions.

## **5.2 SHAP Variants**

| Variant | For | Speed |
|---------|-----|-------|
| **KernelSHAP** | Any model (model-agnostic) | Slow (sampling-based) |
| **TreeSHAP** | Tree-based models (XGBoost, RF, LightGBM) | Fast (exact, polynomial time) |
| **DeepSHAP** | Neural networks | Medium (uses DeepLIFT as baseline) |
| **LinearSHAP** | Linear models | Very fast (exact, closed form) |
| **GradientSHAP** | Neural networks (gradient-based) | Medium |

## **5.3 SHAP Visualization Types**

| Plot | What It Shows | When to Use |
|------|-------------|-------------|
| **Waterfall** | Feature contributions for ONE prediction | Explaining a single decision |
| **Force plot** | Same as waterfall but horizontal | Quick single-prediction visualization |
| **Beeswarm (summary)** | SHAP values for ALL features across ALL predictions | Understanding global feature importance + direction |
| **Dependence** | SHAP value vs feature value for one feature | Understanding feature effect (like PDP but better) |
| **Bar** | Mean absolute SHAP per feature | Simple global importance ranking |

## **5.4 Reading a SHAP Waterfall**

```
Base value (average prediction) = 0.35

Feature contributions:
  income = $85K        → +0.15 (pushes prediction UP)
  credit_history = 7yr → +0.10 (pushes prediction UP)
  debt_ratio = 0.42    → -0.08 (pushes prediction DOWN)
  age = 28             → -0.03 (pushes prediction DOWN)
  employment = 3yr     → +0.01 (slight UP push)

Final prediction: 0.35 + 0.15 + 0.10 - 0.08 - 0.03 + 0.01 = 0.50

"This applicant has a 50% approval probability. The main positive
drivers are high income (+0.15) and long credit history (+0.10).
The main negative driver is a moderately high debt ratio (-0.08)."
```

---

# **6. LIME Deep Dive**

## **6.1 How LIME Works**

```
┌──────────────────────────────────────────────────────────────────┐
│                    LIME ALGORITHM                                  │
│                                                                   │
│  1. Pick the instance to explain                                 │
│                                                                   │
│  2. PERTURB: Generate N neighbors by randomly perturbing         │
│     features of the original instance                            │
│                                                                   │
│  3. PREDICT: Get black-box model predictions for all neighbors  │
│                                                                   │
│  4. WEIGHT: Assign higher weights to neighbors closer to        │
│     the original instance (kernel function)                      │
│                                                                   │
│  5. FIT: Train a simple interpretable model (linear regression,  │
│     decision tree) on the weighted neighbors                     │
│                                                                   │
│  6. EXPLAIN: The simple model's coefficients = local feature     │
│     importances for this specific prediction                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **6.2 LIME for Different Data Types**

| Data Type | Perturbation Method |
|-----------|-------------------|
| **Tabular** | Randomly change feature values (sample from training distribution) |
| **Text** | Randomly remove words/tokens |
| **Images** | Randomly mask superpixels (segments of the image) |

## **6.3 SHAP vs LIME**

| Aspect | SHAP | LIME |
|--------|------|------|
| **Theory** | Game theory (Shapley values), solid theoretical foundation | Local linear approximation, heuristic |
| **Consistency** | Guaranteed: same explanation for same prediction | Can vary between runs (random perturbations) |
| **Speed** | TreeSHAP is fast; KernelSHAP is slow | Generally fast |
| **Additivity** | SHAP values sum to prediction - baseline | No such guarantee |
| **Global → Local** | Can aggregate local SHAP for global explanations | Designed for local only |
| **Adoption** | Industry standard (more widely used) | Still used, especially for quick prototyping |

**Interview recommendation:** Lead with SHAP. Mention LIME as an alternative you'd use for quick iteration or when SHAP is too slow (very large models, real-time explanations).

---

# **7. Deep Learning Explainability**

## **7.1 Methods**

| Method | How | For |
|--------|-----|-----|
| **Attention weights** | Visualize which inputs the attention mechanism focuses on | Transformers (NLP, Vision) |
| **Integrated Gradients** | Accumulate gradients along path from baseline to input | Any differentiable model |
| **GradCAM** | Gradient-weighted class activation maps | CNNs (highlights image regions) |
| **SHAP (DeepSHAP)** | Shapley values using DeepLIFT reference | Neural networks |
| **Saliency maps** | Gradient of output w.r.t. input pixels | Image classification |
| **Concept Activation Vectors (TCAV)** | Test if model uses human-understandable concepts | Any model (concept-level explanations) |

## **7.2 Attention ≠ Explanation**

**Important caveat:** Attention weights show what the model **attends to**, not necessarily what it **uses for the decision**. Research (Jain & Wallace, 2019) showed attention doesn't always correlate with feature importance. Use attention for intuition, not as a rigorous explanation.

## **7.3 For LLMs**

| Method | How | Use Case |
|--------|-----|----------|
| **Chain-of-Thought** | Model explains its reasoning step by step | Self-explanation (prompt engineering) |
| **Logprobs** | Token-level probabilities reveal confidence | Uncertainty estimation |
| **Probing** | Train classifiers on internal representations | Research: what does the model know? |
| **Mechanistic interpretability** | Reverse-engineer circuits in model weights | Research: how does the model work? |

---

# **8. Model Cards and Documentation**

## **8.1 What Is a Model Card?**

A standardized document that describes a model's intended use, performance, limitations, and ethical considerations. Proposed by Mitchell et al. (2019, Google).

## **8.2 Model Card Template**

| Section | Contents |
|---------|---------|
| **Model Details** | Name, version, type, framework, training date |
| **Intended Use** | Primary use cases, users, out-of-scope uses |
| **Training Data** | Source, size, preprocessing, known biases |
| **Evaluation** | Metrics, test sets, per-subgroup performance |
| **Ethical Considerations** | Potential harms, fairness analysis, sensitive features |
| **Caveats & Limitations** | Known failure modes, data gaps, where NOT to use |
| **Recommendations** | Best practices for deployment, monitoring needs |

**Why it matters:** EU AI Act requires documentation for high-risk AI systems. Model cards are becoming the industry standard for this.

---

# **9. Calibration**

## **9.1 What Is Calibration?**

A model is **calibrated** if when it says "70% probability," the event actually occurs 70% of the time. Many models output scores that aren't true probabilities.

```
WELL CALIBRATED:        OVERCONFIDENT:          UNDERCONFIDENT:
Predicted  Actual       Predicted  Actual       Predicted  Actual
0.1        ~10%         0.1        ~5%          0.1        ~20%
0.5        ~50%         0.5        ~30%         0.5        ~70%
0.9        ~90%         0.9        ~70%         0.9        ~98%
```

## **9.2 Why Calibration Matters**

- **Credit scoring:** P(default) = 0.05 should mean exactly 5% of applicants at that score default
- **Medical diagnosis:** P(disease) = 0.8 should mean 80% of patients at that score have the disease
- **Any decision with a threshold:** If the model is miscalibrated, the threshold is wrong

## **9.3 Measuring Calibration**

| Metric | How |
|--------|-----|
| **Reliability diagram** | Plot predicted probability vs observed frequency in bins |
| **Expected Calibration Error (ECE)** | Weighted average of |predicted - actual| per bin |
| **Brier Score** | Mean squared error of probabilistic predictions |

## **9.4 Calibration Methods**

| Method | How | When |
|--------|-----|------|
| **Platt Scaling** | Fit logistic regression on model scores → calibrated probabilities | Binary classification, simple |
| **Isotonic Regression** | Fit non-parametric isotonic regression on scores | More flexible, needs more data |
| **Temperature Scaling** | Divide logits by temperature T before softmax | Neural networks, multi-class |

**Important:** Calibrate on a **held-out calibration set** (not the training or test set).

---

# **10. Interview Questions with Strong Answers**

---

## **Q1: "Explain SHAP values and when you'd use them."**

> SHAP values are based on Shapley values from cooperative game theory. For each prediction, every feature gets a SHAP value representing its **marginal contribution** to the prediction relative to a baseline (average prediction). Crucially, SHAP values sum to the difference between the individual prediction and the baseline — they provide a complete additive attribution.
>
> I use SHAP when I need to: (1) explain individual predictions to stakeholders (waterfall plots), (2) understand global feature importance (beeswarm plots), (3) debug models (if SHAP shows a leaky feature is #1, I have a problem), (4) satisfy regulatory requirements for explainability.
>
> For tree-based models, I use **TreeSHAP** which is exact and fast (polynomial time). For neural networks, **DeepSHAP**. For any other model, **KernelSHAP** (slower but universal).
>
> At Fibe, SHAP would have been essential for the credit scorecard — regulators require explanations for denial decisions, and SHAP gives feature-level attribution for each applicant.

---

## **Q2: "What's the difference between SHAP and LIME? When would you choose one over the other?"**

> Both explain individual predictions, but they work fundamentally differently:
>
> **SHAP** computes theoretically grounded Shapley values — the average marginal contribution of each feature across all possible feature coalitions. It has mathematical guarantees: efficiency (values sum to prediction - baseline), consistency, and symmetry.
>
> **LIME** fits a local linear model around the prediction by perturbing the input and training a simple model on the weighted neighborhood. It's an approximation without the theoretical guarantees of SHAP.
>
> **I prefer SHAP** for most applications because: (1) explanations are deterministic and reproducible, (2) SHAP values are additive (they sum to the prediction), (3) TreeSHAP is fast for tree models, (4) it's the industry standard.
>
> **I'd use LIME** when: (1) I need very fast explanations at serving time (LIME can be faster than KernelSHAP for non-tree models), (2) I'm explaining text or image models (LIME's perturbation approach is very intuitive for these), (3) quick prototyping before committing to SHAP.

---

## **Q3: "How do you handle explainability for a deep learning model?"**

> Multiple layers depending on the use case:
>
> **Architecture-level:** If explainability is critical (healthcare, finance), I might choose an inherently more interpretable architecture — like TFT (Temporal Fusion Transformer) which has built-in variable importance and attention interpretability, rather than a black-box LSTM.
>
> **Post-hoc methods:** For CNNs, I use GradCAM to visualize which image regions influence the prediction. For transformers, I examine attention patterns (with the caveat that attention isn't always faithful to the decision). For any model, DeepSHAP provides feature-level attribution.
>
> **Self-explanation:** For LLMs, Chain-of-Thought prompting makes the model explain its reasoning. This isn't rigorous (the model might rationalize rather than truly explain), but it's useful for building user trust and debugging.
>
> In my radiology report generation project (LLaVA-NeXT), explainability was crucial — doctors need to know which image regions influenced the generated report. GradCAM on the vision encoder + attention visualization on the cross-attention layers provides this.

---

## **Q4: "What is model calibration and why does it matter?"**

> A calibrated model's predicted probabilities match observed frequencies. If the model says "70% chance of default," then among all applicants with that score, approximately 70% should actually default.
>
> **Why it matters:** In credit scoring, the predicted probability IS the business metric — it directly translates to expected loss. If the model is overconfident (predicts 10% default but actual is 20%), the lender takes on more risk than intended. If underconfident, they reject profitable applicants.
>
> I check calibration with a **reliability diagram** (predicted vs actual in bins) and ECE (Expected Calibration Error). Most tree-based models are NOT well-calibrated out of the box — I apply **Platt scaling** (logistic regression on model scores) or **isotonic regression** on a held-out calibration set.
>
> Key rule: calibrate AFTER all model selection and hyperparameter tuning, on a dataset the model hasn't seen during training or validation.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│      EXPLAINABILITY TAKEAWAYS                                     │
│                                                                   │
│  1. SHAP is the industry standard for feature attribution        │
│     (TreeSHAP for trees, DeepSHAP for NNs, KernelSHAP for any) │
│                                                                   │
│  2. GLOBAL (PDP, feature importance) for overall understanding   │
│     LOCAL (SHAP waterfall, LIME) for individual decisions        │
│                                                                   │
│  3. INTERPRETABLE models when possible (regulated domains)       │
│     POST-HOC explanations when black-box is necessary            │
│                                                                   │
│  4. CALIBRATION: predicted probabilities must match reality      │
│     (Platt scaling, isotonic regression on held-out set)         │
│                                                                   │
│  5. MODEL CARDS document intended use, limitations, and bias     │
│     (becoming required under EU AI Act)                          │
│                                                                   │
│  6. ATTENTION ≠ EXPLANATION — use with caution                  │
│                                                                   │
│  7. Explainability is a REGULATORY REQUIREMENT in finance,       │
│     healthcare, and hiring — not optional                        │
└──────────────────────────────────────────────────────────────────┘
```
