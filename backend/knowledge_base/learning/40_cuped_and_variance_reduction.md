# CUPED & Variance Reduction in Experiments — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Online Experimentation, Variance Reduction, CUPED, Stratification, Regression Adjustment

> **Context:** Your file `24_statistics_and_ab_testing.md` covers t-tests, power, and SRM. This guide goes deeper into the **single most important practical technique** in modern experimentation platforms: **variance reduction**. CUPED is the headline name, but the family includes stratification, post-stratification, regression adjustment, doubly-robust estimators, and machine-learned controls. Microsoft, Meta, Netflix, LinkedIn, Booking, Airbnb, DoorDash, and OpenAI all run on this.

---

## Table of Contents

1. [Why Variance Reduction Matters](#1-why-variance-reduction-matters)
2. [The Mean and its Variance — Where the Noise Comes From](#2-the-mean-and-its-variance--where-the-noise-comes-from)
3. [CUPED — The Core Idea](#3-cuped--the-core-idea)
4. [CUPED — Derivation and Math](#4-cuped--derivation-and-math)
5. [Choosing the Pre-Experiment Covariate](#5-choosing-the-pre-experiment-covariate)
6. [Multi-Covariate CUPED (CUPED++ / Regression Adjustment)](#6-multi-covariate-cuped)
7. [Stratification vs Post-Stratification vs CUPED](#7-stratification-vs-post-stratification-vs-cuped)
8. [CUPED for Ratio Metrics (The Delta Method)](#8-cuped-for-ratio-metrics)
9. [ML-Driven Variance Reduction (MLRATE)](#9-ml-driven-variance-reduction-mlrate)
10. [Doubly Robust Estimators](#10-doubly-robust-estimators)
11. [Practical Pitfalls and Failure Modes](#11-practical-pitfalls-and-failure-modes)
12. [Implementation Patterns at Platform Scale](#12-implementation-patterns-at-platform-scale)
13. [Interview Questions with Strong Answers](#13-interview-questions-with-strong-answers)
14. [Key Takeaways](#14-key-takeaways)

---

# 1. Why Variance Reduction Matters

## 1.1 The Business Problem

You want to detect a 0.5% lift in revenue per user. Revenue is heavy-tailed. The required sample size for a vanilla t-test is **enormous** — sometimes 10x your traffic for the week. Three options:

1. **Wait longer** — slow iteration, business hates it.
2. **Run more traffic** — capacity-limited, you can only run so many experiments in parallel.
3. **Reduce variance** — same sensitivity with less data.

Variance reduction is the only one that *scales experimentation capacity* without scaling traffic.

## 1.2 The Sensitivity Equation

Required sample size for a two-sided test with significance $\alpha$ and power $1-\beta$ is approximately:

$$
n \;\approx\; \frac{2 \cdot (z_{\alpha/2} + z_\beta)^2 \cdot \sigma^2}{\Delta^2}
$$

| Symbol | Meaning |
|--------|---------|
| $\sigma^2$ | Variance of the metric |
| $\Delta$ | Minimum detectable effect (MDE) |
| $z$ | Normal quantile |

**Cut $\sigma^2$ by 50% → cut required sample size by 50%.** Variance reduction is a multiplier on every experiment your platform ever runs.

## 1.3 A Mental Picture

```
┌────────────────────────────────────────────────────────────────┐
│             WHY CUPED FEELS LIKE MAGIC                         │
│                                                                │
│   Treatment effect = noise + signal                            │
│                                                                │
│   Without CUPED:                                               │
│   ┌─────────────────────────────────────────┐                  │
│   │ noise ████████████████████ signal ▓▓ │  ← signal lost     │
│   └─────────────────────────────────────────┘                  │
│                                                                │
│   With CUPED (removes predictable pre-period noise):           │
│   ┌─────────────────────────────────────────┐                  │
│   │ noise ████ signal ▓▓                  │  ← signal visible  │
│   └─────────────────────────────────────────┘                  │
│                                                                │
│   Same data. Same effect. ~50% less noise.                     │
└────────────────────────────────────────────────────────────────┘
```

---

# 2. The Mean and its Variance — Where the Noise Comes From

For a metric $Y$ with sample mean $\bar{Y}$:

$$
\text{Var}(\bar{Y}_T - \bar{Y}_C) = \frac{\sigma_T^2}{n_T} + \frac{\sigma_C^2}{n_C}
$$

The variance of the **difference** is what determines confidence intervals and p-values. There are exactly **three levers** to shrink it:

| Lever | How |
|-------|-----|
| **Increase n** | Run more traffic / longer (slow, expensive) |
| **Shrink $\sigma^2$** | Variance reduction (CUPED, stratification, regression adjustment) |
| **Use a more sensitive estimator** | E.g., trimmed mean, transformed outcome |

CUPED works on lever #2 by **subtracting predictable variation** from the outcome.

---

# 3. CUPED — The Core Idea

**CUPED** = **Controlled Experiment Using Pre-Experiment Data** (Deng, Xu, Kohavi, Walker — Microsoft, 2013).

## 3.1 One-Sentence Definition

> **Adjust the outcome by subtracting the part that is predictable from a pre-experiment covariate, then run the usual difference-in-means on the adjusted outcome.**

## 3.2 The Recipe

```
1. Pick a pre-experiment covariate X strongly correlated with the outcome Y.
   (X must be measured BEFORE assignment — otherwise it is post-treatment.)

2. Estimate θ = Cov(Y, X) / Var(X) on pooled data.

3. Compute the CUPED-adjusted outcome:
   Y* = Y - θ · (X - mean(X))

4. Run the usual diff-in-means / t-test / CI on Y* instead of Y.
```

## 3.3 The Intuition

Imagine you want to measure whether a UI change increases sessions/user this week. A user's **last-week sessions** strongly predict **this-week sessions** (correlation ~0.6). CUPED literally **subtracts each user's predicted sessions** based on last week, leaving only the **residual** — which is where the treatment effect lives.

```
This-week sessions = (baseline self) + (noise) + (treatment effect)

CUPED removes "baseline self" → leaves noise + treatment effect.
```

**Why is this unbiased?** Because $X$ is pre-experiment, treatment assignment is independent of $X$, so subtracting $\theta(X - \bar{X})$ does not bias the estimate — it just removes noise common to both arms.

---

# 4. CUPED — Derivation and Math

## 4.1 Variance of an Adjusted Estimator

For the adjusted outcome $Y^* = Y - \theta X$ (centering omitted — it's a constant shift):

$$
\text{Var}(Y^*) = \text{Var}(Y) - 2\theta\,\text{Cov}(Y, X) + \theta^2 \text{Var}(X)
$$

Differentiate w.r.t. $\theta$ and set to zero:

$$
\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}
$$

This is just the **OLS slope** of $Y$ regressed on $X$. Substituting back:

$$
\text{Var}(Y^*) = \text{Var}(Y) \cdot (1 - \rho^2)
$$

where $\rho = \text{Corr}(Y, X)$.

## 4.2 The Magic Number: $\rho^2$

| $\rho$ (correlation) | Variance reduction | Effective sample multiplier |
|---|---|---|
| 0.0 | 0% | 1.0× |
| 0.3 | 9% | 1.1× |
| 0.5 | 25% | 1.33× |
| 0.7 | 49% | ~2× |
| 0.8 | 64% | ~2.8× |
| 0.9 | 81% | ~5.3× |

**Rule of thumb:** A pre-period covariate with **$\rho \geq 0.5$** typically halves your runtime. This is why platforms invest heavily in finding strong pre-period covariates.

## 4.3 Why CUPED is Unbiased

Treatment assignment $T$ is random and independent of $X$ (since $X$ is pre-experiment), so:

$$
\mathbb{E}[\bar{Y}^*_T - \bar{Y}^*_C] = \mathbb{E}[\bar{Y}_T - \bar{Y}_C] - \theta \cdot \underbrace{\mathbb{E}[\bar{X}_T - \bar{X}_C]}_{=0 \text{ by randomization}}
$$

> **Interview soundbite:** *"CUPED is unbiased because randomization guarantees X is balanced in expectation across arms. We only subtract predictable variation, not signal."*

## 4.4 Python Implementation (Minimal)

```python
import numpy as np
import pandas as pd

def cuped_estimate(df, outcome_col, covariate_col, treatment_col):
    """Returns CUPED-adjusted ATE estimate, SE, and variance reduction."""
    Y = df[outcome_col].values
    X = df[covariate_col].values
    T = df[treatment_col].values

    # Estimate theta on POOLED data (treatment+control combined)
    theta = np.cov(Y, X, ddof=1)[0, 1] / np.var(X, ddof=1)

    # Adjusted outcome
    Y_adj = Y - theta * (X - X.mean())

    # Diff-in-means on adjusted outcome
    mu_t = Y_adj[T == 1].mean()
    mu_c = Y_adj[T == 0].mean()
    ate = mu_t - mu_c

    var_t = Y_adj[T == 1].var(ddof=1) / (T == 1).sum()
    var_c = Y_adj[T == 0].var(ddof=1) / (T == 0).sum()
    se = np.sqrt(var_t + var_c)

    var_reduction = 1 - Y_adj.var(ddof=1) / Y.var(ddof=1)

    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - 1.96 * se,
        "ci_high": ate + 1.96 * se,
        "variance_reduction": var_reduction,
        "theta": theta,
    }
```

> **Production note:** Many teams estimate $\theta$ on **the control arm only** to avoid any chance that treatment effects leak into the slope. This is a small bias-variance trade-off; pooled is fine if randomization is clean.

---

# 5. Choosing the Pre-Experiment Covariate

## 5.1 The Two Hard Rules

1. **Measured before assignment** — anything post-assignment can leak treatment effects and bias the estimator.
2. **Same units as outcome** — for users, pre-period user metrics; for sessions, pre-period session metrics.

## 5.2 Good vs Bad Covariates

| Outcome | Best covariate | Bad covariate (why) |
|---------|---------------|---------------------|
| This-week revenue/user | Pre-period (e.g., last 4 weeks) revenue/user | This-week revenue elsewhere on site (post-treatment) |
| Sessions/day | Pre-period sessions/day | Active during experiment (post-treatment) |
| Search CTR | Pre-period CTR for same user | Search query type (high cardinality, weak signal) |
| Order rate | Pre-period order rate | Account age (low correlation) |

## 5.3 Standard Engineering Pattern

```
For every metric you care about, precompute the same metric for the
pre-experiment window (typically 2-8 weeks before assignment).

Join on user_id BEFORE assignment date.

Cold-start users (no pre-period data) → impute with mean, OR drop, OR
include a binary indicator. Document the choice; don't switch silently.
```

## 5.4 Cold Start

For new users with no pre-period:
- **Option A:** Set $X = \bar{X}$ (the mean — contributes 0 to adjustment)
- **Option B:** Include a "new user" indicator alongside $X$ (multi-covariate CUPED)
- **Option C:** Drop them from analysis (only if you also document the population)

---

# 6. Multi-Covariate CUPED

Single-covariate CUPED uses one $X$. **CUPED++ / regression adjustment** uses many:

$$
Y^* = Y - \mathbf{X}^\top \hat{\boldsymbol{\theta}}
$$

where $\mathbf{X}$ is a vector of pre-period covariates (pre-revenue, pre-sessions, country dummies, device dummies, tenure...) and $\hat{\boldsymbol{\theta}}$ is the OLS coefficient vector.

## 6.1 Equivalence to OLS

This is equivalent to estimating ATE as the treatment coefficient in:

$$
Y_i = \alpha + \tau T_i + \mathbf{X}_i^\top \boldsymbol{\beta} + \varepsilon_i
$$

> **Important nuance (Lin, 2013):** Plain OLS adjustment can be biased in small samples. Use **fully-interacted regression** (interact each covariate with treatment):
>
> $$
> Y_i = \alpha + \tau T_i + \mathbf{X}_i^\top \boldsymbol{\beta} + (T_i \cdot \mathbf{X}_i)^\top \boldsymbol{\gamma} + \varepsilon_i
> $$
>
> This is **Lin's estimator** and is the gold standard in modern platforms.

## 6.2 Practical Limits

- Adding too many covariates → overfitting → reduced variance reduction in practice
- Use **L2 regularization** when number of covariates is large
- Common pattern: 3–10 hand-picked strong covariates is enough

---

# 7. Stratification vs Post-Stratification vs CUPED

These are the **three sibling techniques**. Interviewers love to ask the difference.

| Technique | Where it happens | Mechanism | Use when |
|-----------|------------------|-----------|----------|
| **Stratification** | At **assignment** time | Randomize *within* strata (country, device) | You can intervene on the assignment system |
| **Post-stratification** | At **analysis** time | Reweight strata to match overall distribution | Assignment already done; categorical balance is needed |
| **CUPED / regression adjustment** | At **analysis** time | Subtract predicted noise from continuous covariates | Continuous pre-period signal is available |

## 7.1 They Can Be Combined

Real platforms do all three:

1. **Stratify** on big drivers (country, device, platform) at assignment.
2. **CUPED** on pre-period metric continuously.
3. **Post-stratify** on segments observed only after assignment (e.g., "logged in this week").

## 7.2 Why Stratification Alone Isn't Enough

Stratification balances on **categorical** features. Pre-period revenue or session count is **continuous**, and binning into strata loses information. CUPED exploits the full continuous covariate.

---

# 8. CUPED for Ratio Metrics

## 8.1 The Ratio Metric Problem

Click-through rate **at the session or query level** is a ratio:

$$
\text{CTR} = \frac{\sum \text{clicks}}{\sum \text{impressions}}
$$

The **user** is the experimental unit, but the metric averages **events**. Naïve user-mean of session-CTR is wrong, and `total_clicks / total_impressions` per arm needs the **delta method** for variance.

## 8.2 The Delta Method (Sketch)

For a ratio $R = \frac{\bar{N}}{\bar{D}}$:

$$
\text{Var}(R) \approx \frac{1}{\bar{D}^2}\,\text{Var}(\bar{N}) + \frac{\bar{N}^2}{\bar{D}^4}\,\text{Var}(\bar{D}) - \frac{2 \bar{N}}{\bar{D}^3}\,\text{Cov}(\bar{N}, \bar{D})
$$

This is computed **per arm**, then differences are tested.

## 8.3 CUPED for Ratios

Apply CUPED **separately to the numerator and denominator** (or to user-level transformed metrics), then plug into the delta method:

```
N* = N - θ_N · (X_N - mean(X_N))
D* = D - θ_D · (X_D - mean(X_D))
R* = mean(N*) / mean(D*)
```

This is the Microsoft/LinkedIn pattern for things like search CTR, ad CTR, conversion-per-impression.

---

# 9. ML-Driven Variance Reduction (MLRATE)

**MLRATE** = **Machine Learning Regression-Adjusted Treatment Effects** (Guo et al., 2021).

## 9.1 The Idea

Instead of a hand-picked linear covariate, train a **flexible ML model** to predict $Y$ from pre-period features:

$$
\hat{Y}_i = g(\mathbf{X}_i)
$$

Then use $\hat{Y}_i$ as the CUPED covariate:

$$
Y_i^* = Y_i - \theta (\hat{Y}_i - \bar{\hat{Y}})
$$

## 9.2 Why It Helps

- Captures **non-linear** relationships
- Captures **interactions** between covariates
- Works well when you have many noisy features

## 9.3 The Catch — Sample Splitting

To avoid bias, train the predictor on a **separate fold** (cross-fitting):

```
Split data into K folds.
For each fold k:
  Train predictor on data NOT in fold k.
  Predict on fold k.
Combine predictions across folds.
Run CUPED with predicted values as covariate.
```

This is the **double machine learning** pattern (Chernozhukov et al.) applied to RCTs.

---

# 10. Doubly Robust Estimators

## 10.1 Setup

In **observational** studies (or RCTs with non-compliance / missing outcomes), you have:

- A **propensity model** $\hat{e}(X) = P(T=1 \mid X)$
- An **outcome model** $\hat{m}_t(X) = \mathbb{E}[Y \mid X, T=t]$

The **doubly robust estimator** combines both:

$$
\hat{\tau}_{DR} = \frac{1}{n}\sum_i \left[ \hat{m}_1(X_i) - \hat{m}_0(X_i) + \frac{T_i(Y_i - \hat{m}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{m}_0(X_i))}{1-\hat{e}(X_i)} \right]
$$

## 10.2 Why "Doubly Robust"

The estimator is **consistent** if *either* the propensity model *or* the outcome model is correctly specified — not both. This is enormously useful in practice.

> **In RCTs:** propensity is known (50/50, for instance), so DR collapses to **augmented inverse propensity weighting (AIPW)** with an outcome model — essentially CUPED++ with explicit weights.

---

# 11. Practical Pitfalls and Failure Modes

## 11.1 Pitfall: Post-Treatment Covariate

```
BAD:  Use this-week pre-conversion as CUPED covariate.
WHY:  It is measured DURING the experiment, so it absorbs the
      treatment effect. Your ATE becomes biased toward zero.
```

**Rule:** every byte of $X$ must be timestamped **strictly before** assignment.

## 11.2 Pitfall: Weak Covariate Inflating Confidence

When $\rho$ is very small, CUPED reduces variance by ~0%. If you blindly report "CUPED-adjusted CI" people think it's tighter — it isn't. Always log **the estimated variance reduction** alongside the result.

## 11.3 Pitfall: Heavy-Tailed Outcomes

Revenue is often power-law. CUPED reduces variance proportionally but the residual is still heavy-tailed:

- Combine CUPED with **winsorization** (cap top 0.5%–1% of values).
- Combine with **log transform** when interpretable (note: changes the estimand to log-ATE).
- Use **trimmed means** for robustness.

## 11.4 Pitfall: Different Theta Per Segment

If $\theta$ differs strongly by segment, a pooled $\theta$ is suboptimal. Either:
- Estimate $\theta$ per segment and re-aggregate (stratified CUPED).
- Use multi-covariate CUPED with segment interactions.

## 11.5 Pitfall: Mixed Units (User vs Session)

If the experiment randomizes by **user** but the metric is **session-level**, CUPED must operate at the user level (aggregate sessions per user pre-period and post-period). Don't apply user-level $\theta$ to session-level data — variance estimates will be wrong.

## 11.6 Pitfall: Estimating Theta on Treatment Arm Only

Estimating $\theta$ on data that includes a strong treatment effect can bias the slope. **Always estimate $\theta$ on either pooled data or the control arm.** Document the choice in the platform.

---

# 12. Implementation Patterns at Platform Scale

## 12.1 Where CUPED Lives in a Modern Platform

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXPERIMENTATION PIPELINE                       │
│                                                                 │
│  ┌─────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │  Assignment │──▶│  Event logs   │──▶│  Metric tables   │   │
│  └─────────────┘   └───────────────┘   └────────┬─────────┘   │
│                                                  │             │
│  ┌─────────────────────────────────────┐         ▼             │
│  │   Pre-period feature store          │  ┌──────────────┐    │
│  │   (per-unit covariates frozen at    │─▶│  CUPED job   │    │
│  │    assignment time)                 │  │  computes θ, │    │
│  └─────────────────────────────────────┘  │  adjusts Y   │    │
│                                            └──────┬───────┘    │
│                                                    ▼            │
│                                            ┌──────────────┐    │
│                                            │   Analysis   │    │
│                                            │ (ATE, CI, p) │    │
│                                            └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 12.2 Engineering Concerns

| Concern | Solution |
|---------|----------|
| **Feature freshness** | Freeze pre-period covariates at assignment moment; never use "as-of-now" tables |
| **Storage cost** | Maintain pre-period feature tables for ~90 days; archive after |
| **Compute cost** | Estimate $\theta$ once per analysis run; cache across metrics |
| **Multiple metrics** | Compute CUPED per metric; share covariates across metrics where possible |
| **Reproducibility** | Pin covariate snapshots with hashes; never recompute mid-experiment |
| **Backward compatibility** | Always log both CUPED-adjusted and unadjusted estimates |

## 12.3 PySpark Sketch (for "Spark on Spark" JD)

```python
from pyspark.sql import functions as F, Window

# 1. Pre-period covariates (per user, frozen at assignment date)
pre = (events
       .filter(F.col("ts") < F.col("assignment_ts"))
       .filter(F.col("ts") >= F.date_sub(F.col("assignment_ts"), 28))
       .groupBy("user_id")
       .agg(F.sum("revenue").alias("pre_revenue"),
            F.count("session_id").alias("pre_sessions")))

# 2. Post-period outcome (per user, during experiment window)
post = (events
        .filter(F.col("ts") >= F.col("assignment_ts"))
        .filter(F.col("ts") <  F.date_add(F.col("assignment_ts"), 14))
        .groupBy("user_id")
        .agg(F.sum("revenue").alias("Y")))

# 3. Join with assignment
panel = (assignments
         .join(pre,  "user_id", "left")
         .join(post, "user_id", "left")
         .fillna(0, subset=["pre_revenue", "pre_sessions", "Y"]))

# 4. Estimate theta in driver after collecting summary stats
stats = panel.agg(
    F.covar_samp("Y", "pre_revenue").alias("cov_y_x"),
    F.var_samp("pre_revenue").alias("var_x"),
    F.mean("pre_revenue").alias("mean_x")
).collect()[0]

theta = stats["cov_y_x"] / stats["var_x"]
mean_x = stats["mean_x"]

# 5. Adjusted outcome (back to Spark)
panel = panel.withColumn("Y_adj", F.col("Y") - F.lit(theta) * (F.col("pre_revenue") - F.lit(mean_x)))

# 6. ATE on adjusted outcome
result = panel.groupBy("variant").agg(
    F.mean("Y_adj").alias("mean_Y_adj"),
    F.var_samp("Y_adj").alias("var_Y_adj"),
    F.count("*").alias("n")
)
```

## 12.4 Platform Defaults (Sane Choices)

| Setting | Recommendation |
|---------|---------------|
| Pre-period window | 28 days (or 14 if metric is high-frequency) |
| Default covariate | Same metric, pre-period, per unit |
| Theta estimation | Pooled across arms |
| Cold start | Impute mean + new-user indicator |
| Reporting | Always show "with CUPED" and "without CUPED" |

---

# 13. Interview Questions with Strong Answers

## Q1: "What is CUPED and why does it work?"

> CUPED stands for **Controlled Experiments Using Pre-Experiment Data**. It reduces the variance of the treatment effect estimator by subtracting predictable pre-period variation from the outcome before computing the difference in means. Specifically, you adjust the outcome as $Y^* = Y - \theta(X - \bar{X})$ where $X$ is a pre-experiment covariate and $\theta = \text{Cov}(Y,X)/\text{Var}(X)$. It works because randomization guarantees $X$ is balanced across arms in expectation, so subtracting a function of $X$ doesn't bias the estimate — it just removes noise common to both arms. The variance of the adjusted outcome is $(1-\rho^2)$ times the original, so a covariate with correlation 0.7 cuts your variance in half, doubling the effective sample size.

## Q2: "What makes a good CUPED covariate?"

> Three things: it must be **pre-experiment** (no post-treatment information), **highly correlated** with the outcome (typically $\rho \geq 0.3$ to be worth the engineering), and **available for most units** (otherwise cold-start dominates). The standard choice is the *same metric* measured in the *pre-experiment window* at the *same granularity* as the experimental unit. For example, if you randomize by user and measure weekly revenue, your covariate is per-user revenue in the prior 4 weeks. If correlation is weak, look for richer covariates: pre-period engagement composite, ML-predicted outcome (MLRATE), or multi-covariate regression adjustment.

## Q3: "What if the pre-period covariate is missing for new users?"

> A few options, each documented in the platform: (1) **impute with the mean** — contributes zero to the adjustment for that user, which is unbiased but reduces variance reduction for that subpopulation; (2) **add a binary indicator** for "no pre-period data" so the regression treats them as a separate group; (3) **stratified analysis** — run CUPED only on the established-user stratum, and report new-user ATE separately. The choice matters because new users behave differently and pooling can attenuate effects. I'd pick (2) for most cases — it's principled and keeps the full sample.

## Q4: "Why estimate theta on pooled data instead of just control?"

> Pooled estimation is more efficient (uses more data) and is unbiased *in expectation* under randomization because the treatment effect averages out in the covariance. However, if there's a strong treatment effect, $\theta_{\text{pooled}}$ can be slightly biased in finite samples. Microsoft's original CUPED paper uses pooled; some platforms use **control-only** for an extra layer of conservatism — at the cost of higher variance in $\theta$. The sensible default for high-traffic experiments is **pooled**; for small samples or risky launches, control-only.

## Q5: "Can CUPED bias the treatment effect estimate?"

> Properly applied, no. CUPED is provably unbiased because $X$ is pre-experiment and independent of $T$. Bias creeps in when: (a) the "pre-period" covariate accidentally includes post-assignment data — a common data engineering bug; (b) the analyst peeks at the outcome before choosing the covariate — a form of garden-of-forking-paths; (c) $\theta$ is estimated from one arm only and that arm has a strong effect that contaminates the slope. Defenses: freeze covariate tables at assignment time, pre-register the covariate, and prefer pooled theta.

## Q6: "How does CUPED compare to stratified randomization?"

> They attack the same problem — reduce variance by accounting for known structure — but at different points in the pipeline. **Stratification** acts at *assignment time* on categorical features (e.g., assign within country × device strata). **CUPED** acts at *analysis time* on continuous pre-period features. Stratification can't exploit continuous signal without loss from binning; CUPED can. In practice, top platforms do **both**: stratify on a few high-impact categorical dimensions at assignment, and CUPED-adjust on continuous pre-period metrics at analysis. They compose multiplicatively in variance reduction.

## Q7: "Walk me through implementing CUPED in a Spark pipeline at scale."

> Five stages: (1) **Pre-period feature build** — for each user, compute the pre-experiment covariate (e.g., 28-day revenue) frozen at assignment timestamp; this becomes a table joined on user_id and experiment_id. (2) **Post-period outcome build** — same shape, but for the experiment window. (3) **Theta estimation** — compute $\text{Cov}(Y,X)/\text{Var}(X)$ via `covar_samp` and `var_samp` aggregates in Spark; this is one shuffle. (4) **Outcome adjustment** — apply `Y - theta * (X - mean(X))` as a column expression. (5) **Standard analysis** on the adjusted column. The engineering challenges are freezing covariates correctly (no leakage), caching theta per experiment-metric pair, and handling cold-start cleanly. I'd also log the realized variance reduction so we can detect when CUPED is silently doing nothing.

## Q8: "When does CUPED fail or underperform?"

> Several failure modes. First, **weak correlation** ($\rho < 0.2$) — variance reduction is minimal and the engineering cost may not be worth it. Second, **heavy-tailed outcomes** — CUPED reduces variance proportionally, but a single whale user still dominates; combine with winsorization. Third, **non-linear relationships** — single-covariate CUPED only captures linear structure; use MLRATE or polynomial features. Fourth, **distribution shifts** between pre and post periods — if the world changed (seasonality, holiday), $\theta$ estimated on old data doesn't generalize. Fifth, **leakage** — the covariate accidentally includes post-treatment info, biasing the estimate. The platform should monitor realized $\rho$ per experiment and alert when CUPED contributes less than expected.

## Q9: "How would you extend CUPED to ratio metrics like CTR?"

> Ratio metrics need the **delta method** for variance because the user is the unit but the metric averages events. The pattern: (1) define the metric as $R = \mathbb{E}[N]/\mathbb{E}[D]$ for user-level numerator $N$ (clicks) and denominator $D$ (impressions); (2) apply CUPED to $N$ and $D$ separately using their own pre-period covariates — typically $N_{\text{pre}}$ and $D_{\text{pre}}$; (3) compute $R^* = \bar{N}^*/\bar{D}^*$ per arm; (4) compute the variance using the delta method on the adjusted moments. This is the standard pattern in Microsoft's ExP platform and LinkedIn's XLNT, and it can give substantial reduction (50–70%) for search and ads ratio metrics where both clicks and impressions are predictable from history.

## Q10: "What's the difference between CUPED and regression adjustment?"

> Single-covariate CUPED **is** a regression adjustment — specifically, OLS of $Y$ on $X$ with the slope used to residualize $Y$. The term "CUPED" emphasizes the pre-experiment timing constraint and the operational pattern in experimentation platforms. **Multi-covariate regression adjustment** (sometimes called CUPED++) generalizes to many covariates. The state-of-the-art is **Lin's estimator** — fit a regression with treatment, covariates, and treatment×covariate interactions, then take the treatment coefficient. Lin's estimator is **never worse** asymptotically than the unadjusted ATE and is **strictly better** when covariates are predictive. Most modern platforms default to Lin-style adjustment under the hood.

---

# 14. Key Takeaways

```
┌────────────────────────────────────────────────────────────────┐
│                CUPED — INTERVIEW TAKEAWAYS                     │
│                                                                │
│  1. Variance reduction = sample-size multiplier. Every         │
│     experiment on the platform benefits forever.               │
│                                                                │
│  2. CUPED formula: Y* = Y - θ·(X - mean(X)),                  │
│     θ = Cov(Y,X)/Var(X). Variance shrinks by factor (1-ρ²).   │
│                                                                │
│  3. Covariate MUST be pre-experiment. Post-treatment           │
│     covariates bias the estimate.                              │
│                                                                │
│  4. Sane default: same metric, pre-period (28d), per unit.    │
│                                                                │
│  5. Stratify + CUPED + post-stratify compose multiplicatively.│
│                                                                │
│  6. Lin's estimator (fully-interacted OLS) is the gold        │
│     standard for multi-covariate adjustment.                  │
│                                                                │
│  7. For ratio metrics, apply CUPED to numerator and           │
│     denominator separately + delta method for variance.       │
│                                                                │
│  8. MLRATE uses cross-fit ML predictions as the covariate     │
│     — captures non-linearity.                                 │
│                                                                │
│  9. Always log realized variance reduction. Silent              │
│     underperformance is a real failure mode.                   │
│                                                                │
│  10. CUPED is not "just statistical sugar" — it's a platform  │
│      capability that determines how fast a company learns.     │
└────────────────────────────────────────────────────────────────┘
```

**One-liner you can deliver in an interview:**

> *"CUPED is the workhorse of modern experimentation platforms — it cuts experiment runtime in half by subtracting predictable pre-period variation, costs almost nothing once the feature store exists, and composes cleanly with stratification, regression adjustment, and ML-based extensions like MLRATE."*

---

**Further reading**

- Deng, Xu, Kohavi, Walker (2013) — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data*
- Lin (2013) — *Agnostic notes on regression adjustments to experimental data*
- Guo et al. (2021) — *Machine Learning for Variance Reduction in Online Experiments (MLRATE)*
- Kohavi, Tang, Xu (2020) — *Trustworthy Online Controlled Experiments* (book)
