# Statistics & A/B Testing — Comprehensive Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland (4.0 GPA — Probability & Statistics)  
**Focus:** Data Scientist — A/B Testing, Statistical Modeling, Experiment Design, Risk Analytics  
**Core Skills:** Hypothesis Testing, Bayesian Inference, Experiment Design, Python (scipy, statsmodels, numpy)  
**Document Scope:** End-to-end coverage of probability, statistics, A/B testing, Bayesian methods, and ML-adjacent statistical concepts — theory, math, code, and interview strategy

---

## Table of Contents

1. [Probability Fundamentals](#1-probability-fundamentals)
2. [Probability Distributions](#2-probability-distributions)
3. [Descriptive Statistics](#3-descriptive-statistics)
4. [Inferential Statistics](#4-inferential-statistics)
5. [Hypothesis Testing](#5-hypothesis-testing)
6. [A/B Testing](#6-ab-testing)
7. [Regression Analysis](#7-regression-analysis)
8. [Bayesian Statistics](#8-bayesian-statistics)
9. [Common Statistical Concepts in ML](#9-common-statistical-concepts-in-ml)
10. [Common Interview Questions (10+)](#10-common-interview-questions)
11. [Key Takeaways](#11-key-takeaways)

---

# 1. Probability Fundamentals

---

## 1.1 Probability Rules

Probability is the mathematical framework for quantifying uncertainty. Every statistical test, every ML model's loss function, and every A/B test decision ultimately rests on probability theory.

### Sample Space and Events

| Term | Definition |
|------|-----------|
| Sample space ($\Omega$) | The set of all possible outcomes |
| Event ($A$) | A subset of the sample space |
| $P(A)$ | Probability of event $A$, where $0 \leq P(A) \leq 1$ |
| $P(\Omega) = 1$ | Something must happen |
| $P(\emptyset) = 0$ | The impossible event has probability zero |

### Kolmogorov Axioms

1. **Non-negativity:** $P(A) \geq 0$ for every event $A$
2. **Normalization:** $P(\Omega) = 1$
3. **Countable additivity:** For mutually exclusive events $A_1, A_2, \dots$:

$$
P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)
$$

### Addition Rule

**Mutually exclusive events** ($A \cap B = \emptyset$):

$$
P(A \cup B) = P(A) + P(B)
$$

**General addition rule** (events may overlap):

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

> **Interview tip:** The subtraction of $P(A \cap B)$ corrects for double-counting outcomes in the intersection.

### Multiplication Rule

**General:**

$$
P(A \cap B) = P(A) \cdot P(B \mid A) = P(B) \cdot P(A \mid B)
$$

**For independent events:**

$$
P(A \cap B) = P(A) \cdot P(B)
$$

### Complement Rule

$$
P(A^c) = 1 - P(A)
$$

Often useful when computing $P(A)$ directly is hard — compute the complement instead.

### Law of Total Probability

If $B_1, B_2, \dots, B_n$ partition the sample space:

$$
P(A) = \sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)
$$

> **Example:** You want $P(\text{click})$. Users come from mobile (60%) and desktop (40%). Click rates differ by device. Total click probability = weighted sum across segments.

---

## 1.2 Conditional Probability

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

**Interpretation:** The probability of $A$ occurring *given that* $B$ has already occurred. Conditioning restricts the sample space to outcomes where $B$ is true.

### Common Mistake

$P(A \mid B) \neq P(B \mid A)$ in general. This confusion is called the **prosecutor's fallacy** — confusing the probability of the evidence given innocence with the probability of innocence given the evidence.

---

## 1.3 Bayes' Theorem

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

Expanding the denominator via the law of total probability:

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B \mid A) \, P(A) + P(B \mid A^c) \, P(A^c)}
$$

| Component | Name | Meaning |
|-----------|------|---------|
| $P(A)$ | Prior | What we believed before seeing data |
| $P(B \mid A)$ | Likelihood | How likely the data is under hypothesis $A$ |
| $P(A \mid B)$ | Posterior | Updated belief after observing data |
| $P(B)$ | Evidence (marginal likelihood) | Normalizing constant |

### Classic Example — Medical Testing

- Disease prevalence: $P(D) = 0.001$ (1 in 1,000)
- Test sensitivity: $P(+ \mid D) = 0.99$
- Test specificity: $P(- \mid D^c) = 0.99$, so $P(+ \mid D^c) = 0.01$

$$
P(D \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.00099 + 0.00999} \approx 0.09
$$

Even with a 99% accurate test, a positive result gives only a ~9% chance of disease when the base rate is low. This is the **base rate fallacy**.

```python
# Bayes' theorem — medical test
p_disease = 0.001
sensitivity = 0.99  # P(+|D)
false_positive_rate = 0.01  # P(+|~D)

p_pos = sensitivity * p_disease + false_positive_rate * (1 - p_disease)
p_disease_given_pos = (sensitivity * p_disease) / p_pos

print(f"P(Disease | Positive Test) = {p_disease_given_pos:.4f}")
# Output: P(Disease | Positive Test) = 0.0902
```

---

## 1.4 Independence

Two events $A$ and $B$ are **independent** if and only if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

Equivalently: $P(A \mid B) = P(A)$ — knowing $B$ occurred tells you nothing about $A$.

### Independence vs. Mutual Exclusivity

| Property | Independent | Mutually Exclusive |
|----------|-------------|-------------------|
| $P(A \cap B)$ | $= P(A) P(B)$ | $= 0$ |
| Can both occur? | Yes | No |
| Knowing one tells about other? | No | Yes (if one happens, the other didn't) |

> **Interview gotcha:** If $P(A) > 0$ and $P(B) > 0$, mutually exclusive events are **never** independent (because $P(A \cap B) = 0 \neq P(A)P(B)$).

### Conditional Independence

$A$ and $B$ are conditionally independent given $C$ if:

$$
P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)
$$

This is the core assumption behind **Naive Bayes** classifiers — features are independent given the class label.

---

## 1.5 Random Variables

A **random variable** $X$ is a function that maps outcomes in the sample space to real numbers.

### Discrete Random Variables

Takes countable values. Described by a **probability mass function (PMF)**:

$$
P(X = x) = p(x), \quad \sum_x p(x) = 1
$$

**CDF:** $F(x) = P(X \leq x) = \sum_{t \leq x} p(t)$

### Continuous Random Variables

Takes values in an interval. Described by a **probability density function (PDF)**:

$$
f(x) \geq 0, \quad \int_{-\infty}^{\infty} f(x) \, dx = 1
$$

Note: $P(X = x) = 0$ for continuous variables. Instead: $P(a \leq X \leq b) = \int_a^b f(x) \, dx$

**CDF:** $F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt$

---

## 1.6 Expectation, Variance, and Standard Deviation

### Expectation (Mean)

The expected value is the "center of mass" of a distribution.

**Discrete:**

$$
E[X] = \mu = \sum_x x \cdot P(X = x)
$$

**Continuous:**

$$
E[X] = \mu = \int_{-\infty}^{\infty} x \, f(x) \, dx
$$

**Key properties:**
- $E[aX + b] = aE[X] + b$ (linearity)
- $E[X + Y] = E[X] + E[Y]$ (always — even for dependent variables)
- $E[XY] = E[X]E[Y]$ only if $X$ and $Y$ are independent

### Variance

Variance measures the spread of a distribution around its mean.

$$
\text{Var}(X) = \sigma^2 = E[(X - \mu)^2] = E[X^2] - (E[X])^2
$$

**Key properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$ (constants shift, scalars square)
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$
- If independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### Standard Deviation

$$
\text{SD}(X) = \sigma = \sqrt{\text{Var}(X)}
$$

Same units as $X$ (unlike variance which is in squared units).

---

## 1.7 Covariance and Correlation

### Covariance

Measures linear association between two random variables.

$$
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]
$$

| Value | Interpretation |
|-------|---------------|
| $\text{Cov}(X,Y) > 0$ | $X$ and $Y$ tend to increase together |
| $\text{Cov}(X,Y) < 0$ | One increases as the other decreases |
| $\text{Cov}(X,Y) = 0$ | No linear relationship (but could have non-linear dependence!) |

### Pearson Correlation Coefficient

Normalized covariance — dimensionless, bounded between -1 and 1.

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}, \quad -1 \leq \rho \leq 1
$$

| Value | Interpretation |
|-------|---------------|
| $\rho = 1$ | Perfect positive linear relationship |
| $\rho = -1$ | Perfect negative linear relationship |
| $\rho = 0$ | No linear relationship |

> **Critical interview point:** Correlation = 0 does NOT imply independence. $X$ and $X^2$ can be uncorrelated yet completely dependent. Independence implies zero correlation, but not vice versa.

```python
import numpy as np

np.random.seed(42)
x = np.random.uniform(-5, 5, 10000)
y = x ** 2  # deterministic relationship

print(f"Correlation: {np.corrcoef(x, y)[0, 1]:.4f}")
# Output: ~0.00 (near zero correlation despite perfect dependence)
```

---

# 2. Probability Distributions

---

## 2.1 Discrete Distributions

### Bernoulli Distribution

Single trial with two outcomes (success/failure).

$$
X \sim \text{Bernoulli}(p)
$$

| Parameter | Meaning |
|-----------|---------|
| $p$ | Probability of success |
| PMF | $P(X=1) = p, \quad P(X=0) = 1-p$ |
| $E[X]$ | $p$ |
| $\text{Var}(X)$ | $p(1-p)$ |

**Use case:** Single coin flip, single user converts or not, single ad click.

---

### Binomial Distribution

Number of successes in $n$ independent Bernoulli trials.

$$
X \sim \text{Binomial}(n, p)
$$

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \dots, n
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $np$ |
| $\text{Var}(X)$ | $np(1-p)$ |

**Use case:** Number of heads in 100 coin flips, number of users who convert out of 1,000 visitors.

```python
from scipy import stats

# P(at least 55 heads in 100 fair coin flips)
n, p = 100, 0.5
rv = stats.binom(n, p)
p_at_least_55 = 1 - rv.cdf(54)
print(f"P(X >= 55) = {p_at_least_55:.4f}")  # ~0.1841
```

---

### Poisson Distribution

Number of events in a fixed interval when events occur independently at a constant average rate.

$$
X \sim \text{Poisson}(\lambda)
$$

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $\lambda$ |
| $\text{Var}(X)$ | $\lambda$ (mean = variance — defining characteristic) |

**Use case:** Number of server errors per hour, customer arrivals per minute, emails received per day.

> **Key insight:** Poisson is the limit of Binomial($n, p$) as $n \to \infty$, $p \to 0$, and $np = \lambda$ stays constant. Use Poisson when $n$ is large and $p$ is small.

---

### Geometric Distribution

Number of trials until the **first** success.

$$
X \sim \text{Geometric}(p)
$$

$$
P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \dots
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $1/p$ |
| $\text{Var}(X)$ | $(1-p)/p^2$ |

**Memoryless property:** $P(X > s+t \mid X > s) = P(X > t)$. Past failures don't affect future success probability.

**Use case:** Number of job applications until an offer, A/B test trials until first conversion.

---

## 2.2 Continuous Distributions

### Normal (Gaussian) Distribution

The most important distribution in statistics — the foundation of most hypothesis tests and confidence intervals.

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $\mu$ |
| $\text{Var}(X)$ | $\sigma^2$ |
| Symmetry | Symmetric around $\mu$ |
| 68-95-99.7 rule | 68% within $\pm 1\sigma$, 95% within $\pm 2\sigma$, 99.7% within $\pm 3\sigma$ |

**Standard normal:** $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

**Why so important:**
1. CLT guarantees sample means converge to normal
2. Many natural phenomena are approximately normal
3. Maximum entropy distribution for a given mean and variance
4. Sum of independent normals is normal

```python
from scipy import stats
import numpy as np

# Standard normal — P(Z > 1.96)
print(f"P(Z > 1.96) = {1 - stats.norm.cdf(1.96):.4f}")  # 0.0250

# Normal with mean=100, std=15 — P(X > 130)
rv = stats.norm(loc=100, scale=15)
print(f"P(X > 130) = {1 - rv.cdf(130):.4f}")  # 0.0228
```

---

### Uniform Distribution

All values in $[a, b]$ are equally likely.

$$
X \sim \text{Uniform}(a, b)
$$

$$
f(x) = \frac{1}{b - a}, \quad a \leq x \leq b
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $\frac{a+b}{2}$ |
| $\text{Var}(X)$ | $\frac{(b-a)^2}{12}$ |

**Use case:** Random number generation, initial weight sampling, p-values under $H_0$.

---

### Exponential Distribution

Time between events in a Poisson process. The continuous analog of the Geometric distribution.

$$
X \sim \text{Exponential}(\lambda)
$$

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $1/\lambda$ |
| $\text{Var}(X)$ | $1/\lambda^2$ |
| Memoryless | $P(X > s+t \mid X > s) = P(X > t)$ |

**Use case:** Time between customer arrivals, time until next server failure, radioactive decay.

---

### Beta Distribution

Distribution on $[0, 1]$ — ideal for modeling probabilities and proportions.

$$
X \sim \text{Beta}(\alpha, \beta)
$$

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $\frac{\alpha}{\alpha + \beta}$ |
| $\text{Var}(X)$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

**Why critical in data science:**
- Conjugate prior for the Bernoulli/Binomial likelihood in Bayesian inference
- Beta(1,1) = Uniform(0,1) — uninformative prior
- Used extensively in Bayesian A/B testing
- Models conversion rates, click-through rates, success probabilities

```python
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

x = np.linspace(0, 1, 200)
params = [(1, 1), (2, 5), (5, 2), (10, 10), (0.5, 0.5)]

fig, ax = plt.subplots(figsize=(10, 6))
for a, b in params:
    ax.plot(x, stats.beta.pdf(x, a, b), label=f"Beta({a},{b})")
ax.legend()
ax.set_title("Beta Distribution — Various Parameters")
ax.set_xlabel("x")
ax.set_ylabel("Density")
plt.show()
```

---

### Gamma Distribution

Generalizes the exponential distribution. Models waiting times for multiple events.

$$
X \sim \text{Gamma}(\alpha, \beta)
$$

$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

| Property | Value |
|----------|-------|
| $E[X]$ | $\alpha / \beta$ |
| $\text{Var}(X)$ | $\alpha / \beta^2$ |
| Special case | Gamma(1, $\lambda$) = Exponential($\lambda$) |
| Special case | Gamma($n/2$, 1/2) = Chi-squared($n$) |

**Use case:** Conjugate prior for the Poisson rate parameter, modeling insurance claim amounts, Bayesian inference.

---

## 2.3 Distribution Selection Guide

| Scenario | Distribution |
|----------|-------------|
| Yes/No outcome (single trial) | Bernoulli |
| Count of successes in $n$ trials | Binomial |
| Count of rare events in fixed interval | Poisson |
| Trials until first success | Geometric |
| Continuous measurement, symmetric | Normal |
| Equal probability over an interval | Uniform |
| Time between independent events | Exponential |
| Modeling a probability/proportion | Beta |
| Waiting time for $\alpha$ events | Gamma |
| Sum of squared standard normals | Chi-squared |
| Ratio of two chi-squared variables | F-distribution |
| Small-sample mean with unknown variance | t-distribution |

---

## 2.4 Central Limit Theorem (CLT)

The most important theorem in statistics.

> **Statement:** Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables with mean $\mu$ and finite variance $\sigma^2$. As $n \to \infty$:
>
> $$
> \bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
> $$
>
> Equivalently:
>
> $$
> \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
> $$

**Why it matters:**
1. Justifies using normal-based confidence intervals and hypothesis tests even when the underlying population is not normal
2. Sample means of *any* distribution (with finite variance) become approximately normal for large $n$
3. Rule of thumb: $n \geq 30$ is usually sufficient (but depends on the skewness of the population)
4. Foundation of A/B testing statistical analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# CLT demonstration: sampling from an exponential (skewed) distribution
np.random.seed(42)
population = np.random.exponential(scale=2, size=100_000)
sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(5000)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(population, bins=50, density=True, alpha=0.7)
axes[0].set_title("Population (Exponential — Skewed)")
axes[1].hist(sample_means, bins=50, density=True, alpha=0.7)
axes[1].set_title("Distribution of Sample Means (n=50) — Approximately Normal")
plt.tight_layout()
plt.show()
```

---

## 2.5 Law of Large Numbers (LLN)

> **Weak LLN:** As $n \to \infty$, the sample mean $\bar{X}_n$ converges in probability to the population mean $\mu$:
>
> $$
> \bar{X}_n \xrightarrow{P} \mu
> $$

**Practical meaning:** The more data you collect, the closer your sample mean gets to the true mean. This justifies using sample averages as estimators.

**CLT vs. LLN:**

| Concept | What it says |
|---------|-------------|
| LLN | Sample mean converges to population mean |
| CLT | Tells you the *distribution* of the sample mean (normal) and how fast it converges |

---

# 3. Descriptive Statistics

---

## 3.1 Measures of Central Tendency

### Mean (Arithmetic Average)

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

- Sensitive to outliers (a single extreme value can shift it dramatically)
- Best for symmetric distributions

### Median

The middle value when data is sorted. For $n$ observations:
- Odd $n$: middle value
- Even $n$: average of two middle values

- **Robust to outliers** — not affected by extreme values
- Better measure of center for skewed distributions
- Salaries, home prices → always report median

### Mode

The most frequently occurring value. Can be:
- **Unimodal:** one peak
- **Bimodal:** two peaks (may indicate two sub-populations)
- **Multimodal:** multiple peaks

### When to Use Which

| Measure | Use when... | Example |
|---------|-------------|---------|
| Mean | Distribution is symmetric, no extreme outliers | Test scores, heights |
| Median | Data is skewed or has outliers | Income, home prices |
| Mode | Categorical data or identifying peaks | Most common product, survey responses |

### Relationship Under Skewness

| Skew | Order |
|------|-------|
| Right-skewed (positive) | Mode < Median < Mean |
| Symmetric | Mode ≈ Median ≈ Mean |
| Left-skewed (negative) | Mean < Median < Mode |

---

## 3.2 Measures of Spread

### Variance and Standard Deviation

**Population variance:**

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
$$

**Sample variance (Bessel's correction):**

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

> **Why $n-1$?** Dividing by $n$ underestimates the population variance (biased). Using $n-1$ gives an unbiased estimator. This is **Bessel's correction**. Intuition: the sample mean "uses up" one degree of freedom.

### Interquartile Range (IQR)

$$
\text{IQR} = Q_3 - Q_1
$$

- $Q_1$ (25th percentile): 25% of data below this value
- $Q_2$ (50th percentile): median
- $Q_3$ (75th percentile): 75% of data below this value

**Outlier detection (Tukey's fences):**
- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$
- Points outside fences are flagged as outliers (used in box plots)

### Percentiles

The $k$-th percentile is the value below which $k\%$ of the data falls.

```python
import numpy as np

data = np.random.exponential(scale=5, size=1000)
print(f"Mean:   {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std:    {np.std(data, ddof=1):.2f}")  # ddof=1 for sample std
print(f"IQR:    {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
print(f"90th percentile: {np.percentile(data, 90):.2f}")
```

---

## 3.3 Skewness and Kurtosis

### Skewness

Measures the asymmetry of the distribution.

$$
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}
$$

| Value | Shape |
|-------|-------|
| Skewness = 0 | Symmetric |
| Skewness > 0 | Right-skewed (long right tail) |
| Skewness < 0 | Left-skewed (long left tail) |

### Kurtosis

Measures the "tailedness" — how much of the distribution is in the tails vs. the center.

$$
\text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}
$$

**Excess kurtosis** = Kurtosis - 3 (so the normal distribution has excess kurtosis = 0).

| Type | Excess Kurtosis | Tails |
|------|----------------|-------|
| Mesokurtic | 0 | Normal-like tails |
| Leptokurtic | > 0 | Heavy tails, more outliers (e.g., t-distribution, financial returns) |
| Platykurtic | < 0 | Light tails, fewer outliers (e.g., uniform) |

```python
from scipy import stats

data = np.random.exponential(scale=5, size=10000)
print(f"Skewness: {stats.skew(data):.3f}")          # ~2.0 (right-skewed)
print(f"Excess Kurtosis: {stats.kurtosis(data):.3f}")  # ~6.0 (leptokurtic)
```

---

# 4. Inferential Statistics

---

## 4.1 Sampling Methods

Inferential statistics draws conclusions about a **population** from a **sample**. The sampling method directly affects the validity of your inferences.

### Simple Random Sampling (SRS)

Every individual in the population has an equal probability of being selected.

- **Pros:** Unbiased, easy to implement
- **Cons:** May miss rare sub-groups, requires a complete sampling frame

### Stratified Sampling

Divide the population into homogeneous **strata** (e.g., age groups, regions), then sample from each stratum.

- **Pros:** Ensures representation of all sub-groups, reduces variance
- **Cons:** Requires knowledge of population structure
- **Example:** Sampling 100 users — ensure proportional representation of mobile vs. desktop

### Cluster Sampling

Divide the population into **clusters** (e.g., schools, cities), randomly select clusters, then sample all or some members within selected clusters.

- **Pros:** Cost-effective when population is geographically dispersed
- **Cons:** Higher variance than SRS or stratified; clusters may be internally homogeneous

### Comparison Table

| Method | Variance | Cost | Representation |
|--------|----------|------|---------------|
| Simple Random | Moderate | Moderate | May miss minorities |
| Stratified | Low | Higher | Guaranteed sub-group coverage |
| Cluster | Higher | Low | Depends on cluster diversity |

> **Interview relevance:** In A/B testing, randomization is a form of simple random sampling — each user is randomly assigned to control or treatment.

---

## 4.2 Point Estimation

A **point estimate** is a single value used to estimate a population parameter.

| Parameter | Estimator | Symbol |
|-----------|----------|--------|
| Population mean ($\mu$) | Sample mean | $\bar{x}$ |
| Population variance ($\sigma^2$) | Sample variance | $s^2$ |
| Population proportion ($p$) | Sample proportion | $\hat{p}$ |

**Desirable properties of estimators:**

| Property | Meaning |
|----------|---------|
| Unbiased | $E[\hat{\theta}] = \theta$ — on average, hits the true value |
| Consistent | As $n \to \infty$, $\hat{\theta} \to \theta$ |
| Efficient | Smallest variance among all unbiased estimators |

---

## 4.3 Standard Error

The standard error (SE) measures how much a sample statistic varies from sample to sample.

**Standard error of the mean:**

$$
\text{SE}(\bar{x}) = \frac{s}{\sqrt{n}}
$$

**Standard error of a proportion:**

$$
\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

> **SE vs. SD:** Standard deviation measures variability *within* a single sample. Standard error measures variability *of a statistic across* many samples.

---

## 4.4 Confidence Intervals

A **confidence interval** provides a range of plausible values for a population parameter.

### Formula (for the mean, large sample or known $\sigma$):

$$
\text{CI} = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

For unknown $\sigma$ and small $n$, use the t-distribution:

$$
\text{CI} = \bar{x} \pm t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}
$$

### Common Confidence Levels

| Confidence Level | $z_{\alpha/2}$ |
|-----------------|-----------------|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

### Correct Interpretation

> "If we repeated this experiment many times and computed a 95% CI each time, approximately 95% of those intervals would contain the true population parameter."

**Common WRONG interpretation:** "There is a 95% probability that the true parameter is in this interval." (The parameter is fixed — it's either in the interval or not. The probability statement is about the *procedure*, not the specific interval.)

### CI for a Proportion

$$
\text{CI} = \hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

```python
import numpy as np
from scipy import stats

# 95% CI for the mean
data = np.random.normal(loc=50, scale=10, size=100)
n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # standard error of mean
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
print(f"Sample Mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# 95% CI for a proportion
p_hat = 0.12  # observed conversion rate
n_prop = 5000
se_prop = np.sqrt(p_hat * (1 - p_hat) / n_prop)
ci_prop = (p_hat - 1.96 * se_prop, p_hat + 1.96 * se_prop)
print(f"\nProportion: {p_hat}")
print(f"95% CI: ({ci_prop[0]:.4f}, {ci_prop[1]:.4f})")
```

### Factors Affecting CI Width

| Factor | Effect on CI width |
|--------|-------------------|
| Increase confidence level | Wider |
| Increase sample size $n$ | Narrower |
| Higher variability ($\sigma$) | Wider |

---

# 5. Hypothesis Testing

---

## 5.1 Framework

Hypothesis testing is the formal process of using sample data to decide between two competing claims about a population parameter.

### Steps

1. **State hypotheses:**
   - $H_0$ (null hypothesis): No effect, no difference, status quo
   - $H_1$ (alternative hypothesis): There IS an effect, a difference, a change

2. **Choose significance level $\alpha$** (typically 0.05)

3. **Compute test statistic** from the sample data

4. **Determine the p-value** (or compare to critical value)

5. **Make a decision:** Reject $H_0$ if p-value < $\alpha$

### Null and Alternative Hypotheses

| Test Type | $H_0$ | $H_1$ |
|-----------|---------|---------|
| Two-tailed | $\mu = \mu_0$ | $\mu \neq \mu_0$ |
| Right-tailed | $\mu \leq \mu_0$ | $\mu > \mu_0$ |
| Left-tailed | $\mu \geq \mu_0$ | $\mu < \mu_0$ |

---

## 5.2 Errors and Power

| | $H_0$ is TRUE | $H_0$ is FALSE |
|---|---|---|
| **Reject $H_0$** | Type I Error ($\alpha$) — False Positive | Correct (Power = $1 - \beta$) |
| **Fail to reject $H_0$** | Correct (True Negative) | Type II Error ($\beta$) — False Negative |

### Type I Error ($\alpha$)

- Rejecting $H_0$ when it is actually true
- "Finding a difference when there is none"
- Controlled by significance level (typically $\alpha = 0.05$)
- **A/B testing analog:** Declaring a winner when there's no real difference

### Type II Error ($\beta$)

- Failing to reject $H_0$ when it is actually false
- "Missing a real difference"
- Depends on sample size, effect size, and variance

### Statistical Power

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_0 \text{ is false})
$$

Factors that increase power:
1. Larger sample size
2. Larger effect size
3. Higher $\alpha$ (but increases Type I error)
4. Lower variance

> **Industry standard:** Power ≥ 0.80 (80% chance of detecting a real effect).

---

## 5.3 The p-value

### Definition

> The p-value is the probability of observing a test statistic as extreme as (or more extreme than) the one computed, **assuming $H_0$ is true**.

$$
\text{p-value} = P(\text{data as extreme or more} \mid H_0 \text{ true})
$$

### Interpretation

- Small p-value (< $\alpha$): Data is unlikely under $H_0$ → Reject $H_0$
- Large p-value (≥ $\alpha$): Data is consistent with $H_0$ → Fail to reject $H_0$

### Common Misconceptions (Interview Gold)

| Misconception | Correct Statement |
|--------------|-------------------|
| "p-value = probability that $H_0$ is true" | p-value = probability of the data given $H_0$ is true |
| "p = 0.03 means 3% chance the null is true" | It means: if $H_0$ were true, we'd see data this extreme 3% of the time |
| "p > 0.05 means no effect" | It means we lack evidence to reject $H_0$ — absence of evidence ≠ evidence of absence |
| "Small p-value → large effect" | p-value says nothing about effect SIZE; with huge $n$, tiny effects can yield tiny p-values |
| "p-value measures practical importance" | It measures statistical significance only — practical significance is a separate question |

---

## 5.4 One-Tailed vs. Two-Tailed Tests

| Aspect | One-Tailed | Two-Tailed |
|--------|-----------|-----------|
| $H_1$ | Direction specified ($>$ or $<$) | Direction not specified ($\neq$) |
| Rejection region | One tail only | Both tails |
| Power | Higher for detecting effect in specified direction | Split across both directions |
| When to use | Strong prior belief about direction | Default choice; more conservative |

> **Interview answer:** "I default to two-tailed tests unless there is a strong, pre-specified reason to test in one direction only. Two-tailed tests are more conservative and protect against unexpected effects."

---

## 5.5 Common Hypothesis Tests

### z-test

**When:** Large sample ($n \geq 30$) or known population variance.

**Test statistic:**

$$
z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

For proportions:

$$
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
$$

where $\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$ is the pooled proportion.

---

### t-test

**When:** Small sample or unknown population variance.

**One-sample t-test:**

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad \text{df} = n - 1
$$

**Two-sample (independent) t-test:**

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

**Paired t-test:** For before/after or matched pairs — test if the mean *difference* = 0.

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad \text{where } d_i = x_{i,\text{after}} - x_{i,\text{before}}
$$

**Assumptions:**
1. Data is continuous
2. Samples are independent (except paired)
3. Approximately normal (or $n$ is large enough for CLT)
4. Equal variances (for pooled t-test; Welch's t-test doesn't require this)

```python
from scipy import stats
import numpy as np

# Two-sample t-test: comparing conversion amounts
control = np.random.normal(loc=50, scale=10, size=200)
treatment = np.random.normal(loc=53, scale=10, size=200)

t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)  # Welch's
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")
```

---

### Chi-Squared Test

**Test for Independence:** Are two categorical variables related?

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

where $O_i$ = observed frequency, $E_i$ = expected frequency under independence.

```python
from scipy import stats
import numpy as np

# Chi-squared test for independence
# Contingency table: rows = device type, cols = converted/not
observed = np.array([
    [120, 880],   # mobile: 120 converted, 880 didn't
    [200, 800],   # desktop: 200 converted, 800 didn't
])

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi2: {chi2:.4f}, p-value: {p_value:.4f}, dof: {dof}")
print(f"Expected frequencies:\n{expected}")
```

**Goodness of Fit:** Does observed data match an expected distribution?

```python
# Are dice fair? 600 rolls
observed = np.array([110, 95, 100, 90, 105, 100])
expected = np.array([100, 100, 100, 100, 100, 100])

chi2, p_value = stats.chisquare(observed, f_exp=expected)
print(f"Chi2: {chi2:.2f}, p-value: {p_value:.4f}")
```

---

### ANOVA (Analysis of Variance)

Compares means across **three or more groups**. Tests whether at least one group mean differs from the others.

**One-Way ANOVA:**

$$
F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{MS_{\text{between}}}{MS_{\text{within}}}
$$

| Component | Formula |
|-----------|---------|
| $SS_{\text{between}}$ | $\sum n_j (\bar{x}_j - \bar{x})^2$ |
| $SS_{\text{within}}$ | $\sum \sum (x_{ij} - \bar{x}_j)^2$ |
| $MS = SS/\text{df}$ | Mean square |

**Assumptions:** Independence, normality, homogeneity of variances (Levene's test).

```python
from scipy import stats
import numpy as np

# One-way ANOVA: 3 pricing strategies
group_a = np.random.normal(52, 8, 100)
group_b = np.random.normal(55, 8, 100)
group_c = np.random.normal(50, 8, 100)

f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
```

**Two-Way ANOVA:** Tests effects of TWO independent variables and their interaction.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

# Two-way ANOVA example
np.random.seed(42)
data = pd.DataFrame({
    'revenue': np.random.normal(50, 10, 120),
    'pricing': np.repeat(['low', 'medium', 'high'], 40),
    'device': np.tile(np.repeat(['mobile', 'desktop'], 20), 3)
})

model = ols('revenue ~ C(pricing) * C(device)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

> **Post-hoc tests:** If ANOVA is significant, use Tukey's HSD, Bonferroni, or Scheffe to identify *which* groups differ.

---

### Non-Parametric Tests

Use when data violates normality assumptions or is ordinal.

| Parametric Test | Non-Parametric Equivalent | When to Use |
|----------------|--------------------------|-------------|
| Independent t-test | **Mann-Whitney U** | Comparing two independent groups, non-normal data |
| Paired t-test | **Wilcoxon Signed-Rank** | Comparing paired samples, non-normal differences |
| One-way ANOVA | **Kruskal-Wallis** | Comparing 3+ independent groups, non-normal data |

```python
from scipy import stats
import numpy as np

# Mann-Whitney U test
control = np.random.exponential(5, 100)
treatment = np.random.exponential(6, 100)

u_stat, p_value = stats.mannwhitneyu(control, treatment, alternative='two-sided')
print(f"U-statistic: {u_stat:.2f}, p-value: {p_value:.4f}")

# Wilcoxon Signed-Rank test (paired)
before = np.random.exponential(5, 50)
after = before + np.random.normal(1, 2, 50)

w_stat, p_value = stats.wilcoxon(after - before)
print(f"W-statistic: {w_stat:.2f}, p-value: {p_value:.4f}")
```

---

### Hypothesis Test Selection Flowchart

```
What are you comparing?
│
├── One group vs. known value
│   ├── Normal/large n → One-sample t-test or z-test
│   └── Non-normal, small n → Wilcoxon Signed-Rank
│
├── Two groups
│   ├── Independent
│   │   ├── Normal → Independent t-test (Welch's)
│   │   └── Non-normal → Mann-Whitney U
│   └── Paired (before/after)
│       ├── Normal differences → Paired t-test
│       └── Non-normal → Wilcoxon Signed-Rank
│
├── Three or more groups
│   ├── Normal → ANOVA → post-hoc (Tukey's HSD)
│   └── Non-normal → Kruskal-Wallis
│
└── Categorical variables
    ├── Independence → Chi-squared test of independence
    └── Distribution fit → Chi-squared goodness of fit
```

---

# 6. A/B Testing

---

## 6.1 What is A/B Testing?

An **A/B test** (or split test, randomized controlled experiment) randomly assigns users into two groups:

- **Control (A):** Existing experience (status quo)
- **Treatment (B):** New variant (change you want to test)

You then measure a **metric of interest** and use statistical tests to determine if the treatment caused a significant change.

### Why A/B Testing Matters

- **Gold standard for causal inference** — randomization eliminates confounders
- **Data-driven decisions** — moves beyond opinions and intuition
- **Risk mitigation** — test changes on a subset before rolling out to everyone
- **Measurable impact** — quantifies the exact effect of a change

> **Key principle:** Correlation ≠ causation. A/B tests, when properly designed, establish **causation**.

---

## 6.2 Designing an A/B Test

### Step-by-Step Framework

| Step | Description |
|------|-------------|
| 1. Formulate hypothesis | "Changing the CTA button color from blue to green will increase conversion rate" |
| 2. Choose primary metric | Conversion rate, revenue per user, engagement time |
| 3. Define guardrail metrics | Metrics that should NOT degrade (page load time, error rate) |
| 4. Calculate sample size | Power analysis based on MDE, baseline rate, $\alpha$, $\beta$ |
| 5. Randomize users | Random assignment to control/treatment |
| 6. Run the experiment | For the pre-determined duration |
| 7. Analyze results | Statistical test + practical significance |
| 8. Make a decision | Ship, iterate, or discard |

### Choosing Metrics

**Primary metric (OEC — Overall Evaluation Criterion):** The single metric the test is designed to move.

| Metric Type | Examples |
|-------------|---------|
| Conversion rate | % of users who sign up, purchase, click |
| Revenue per user | Average revenue generated per user |
| Engagement | Time on site, pages per session, DAU/MAU |
| Retention | Day-7 or Day-30 return rate |

**Guardrail metrics:** Metrics that should stay neutral or not degrade.
- Page load time
- Error rates
- Customer support tickets
- Revenue (if testing engagement feature)

---

## 6.3 Sample Size Calculation and Power Analysis

### The Three Levers

| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Significance level | $\alpha$ | 0.05 |
| Power | $1 - \beta$ | 0.80 |
| Minimum Detectable Effect (MDE) | $\delta$ | Business-dependent |

### Formula for Two-Proportion z-test

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_1 - p_2)^2}
$$

Simplified (assuming equal group sizes and $p_1 \approx p_2 \approx p$):

$$
n_{\text{per group}} = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2p(1-p)}{\delta^2}
$$

where $\delta = p_2 - p_1$ is the MDE.

### Effect Size

**Cohen's h** (for proportions):

$$
h = 2 \arcsin(\sqrt{p_2}) - 2 \arcsin(\sqrt{p_1})
$$

**Cohen's d** (for means):

$$
d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}}
$$

| Cohen's d | Effect Size |
|-----------|-------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Python: Sample Size Calculation

```python
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np

# ========================================
# Method 1: Proportion-based (conversion rate)
# ========================================
baseline_rate = 0.10   # current conversion rate: 10%
mde = 0.02             # want to detect a 2 percentage point lift → 12%

effect_size = proportion_effectsize(baseline_rate, baseline_rate + mde)
analysis = NormalIndPower()
sample_size = analysis.solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    ratio=1.0,  # equal group sizes
    alternative='two-sided'
)
print(f"Effect size (Cohen's h): {effect_size:.4f}")
print(f"Sample size per group: {int(np.ceil(sample_size))}")
print(f"Total sample size: {int(np.ceil(sample_size)) * 2}")

# ========================================
# Method 2: Continuous metric (e.g., revenue per user)
# ========================================
baseline_mean = 50.0
expected_lift = 3.0     # $3 increase
pooled_std = 20.0

cohen_d = expected_lift / pooled_std
analysis_t = TTestIndPower()
sample_size_t = analysis_t.solve_power(
    effect_size=cohen_d,
    alpha=0.05,
    power=0.80,
    ratio=1.0,
    alternative='two-sided'
)
print(f"\nCohen's d: {cohen_d:.4f}")
print(f"Sample size per group: {int(np.ceil(sample_size_t))}")
```

### Sample Size Sensitivity

| Change | Effect on Required Sample Size |
|--------|-------------------------------|
| Smaller MDE | **Much larger** (quadratic: halving MDE → 4x sample) |
| Higher power | Larger |
| Lower $\alpha$ | Larger |
| Higher baseline variance | Larger |

---

## 6.4 Running the Test

### Randomization

- **Unit of randomization:** Usually the user (not the page view or session)
- **Hash-based assignment:** Hash(user_id + experiment_id) → deterministic, reproducible
- **Stratified randomization:** Ensure balanced groups across important covariates (device, country)

### How Long to Run

$$
\text{Duration (days)} = \frac{\text{Total required sample size}}{\text{Daily eligible traffic}}
$$

**Minimum duration:** At least 1 full week to capture day-of-week effects. Ideally 2+ weeks.

### SRM Check (Sample Ratio Mismatch)

Before analyzing, verify that the split is approximately 50/50 (or whatever ratio you designed).

```python
from scipy import stats

# SRM check
n_control = 4950
n_treatment = 5050
expected_ratio = 0.5

chi2_stat = (n_control - (n_control + n_treatment) * expected_ratio) ** 2 / \
            ((n_control + n_treatment) * expected_ratio) + \
            (n_treatment - (n_control + n_treatment) * (1 - expected_ratio)) ** 2 / \
            ((n_control + n_treatment) * (1 - expected_ratio))

p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
print(f"SRM Chi2: {chi2_stat:.4f}, p-value: {p_value:.4f}")
print(f"SRM detected: {p_value < 0.01}")
```

> If SRM is detected (p < 0.01), the experiment is compromised — investigate before analyzing.

---

## 6.5 Analyzing A/B Test Results

### Two-Proportion z-test (Conversion Rate)

```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Results
n_control = 5000
conversions_control = 500  # 10%
n_treatment = 5000
conversions_treatment = 575  # 11.5%

# Two-proportion z-test
count = np.array([conversions_control, conversions_treatment])
nobs = np.array([n_control, n_treatment])

z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

p_control = conversions_control / n_control
p_treatment = conversions_treatment / n_treatment
lift = (p_treatment - p_control) / p_control * 100

# Confidence interval for the difference
se_diff = np.sqrt(p_control * (1 - p_control) / n_control +
                  p_treatment * (1 - p_treatment) / n_treatment)
ci_lower = (p_treatment - p_control) - 1.96 * se_diff
ci_upper = (p_treatment - p_control) + 1.96 * se_diff

print(f"Control rate: {p_control:.4f}")
print(f"Treatment rate: {p_treatment:.4f}")
print(f"Relative lift: {lift:.2f}%")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"95% CI for difference: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"Statistically significant: {p_value < 0.05}")
```

### Welch's t-test (Continuous Metrics)

```python
from scipy import stats
import numpy as np

# Revenue per user
np.random.seed(42)
revenue_control = np.random.lognormal(mean=3.5, sigma=1.0, size=5000)
revenue_treatment = np.random.lognormal(mean=3.55, sigma=1.0, size=5000)

t_stat, p_value = stats.ttest_ind(revenue_control, revenue_treatment, equal_var=False)

mean_c = np.mean(revenue_control)
mean_t = np.mean(revenue_treatment)
lift = (mean_t - mean_c) / mean_c * 100

print(f"Control mean revenue: ${mean_c:.2f}")
print(f"Treatment mean revenue: ${mean_t:.2f}")
print(f"Relative lift: {lift:.2f}%")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

## 6.6 Statistical Significance vs. Practical Significance

| Concept | Question |
|---------|----------|
| Statistical significance | "Is the observed difference unlikely due to chance?" |
| Practical significance | "Is the observed difference large enough to matter for the business?" |

**Example:** A 0.01% lift in conversion rate may be statistically significant with n = 10 million, but shipping the change may not be worth the engineering cost.

**Framework for decisions:**

| Stat. Significant? | Practically Significant? | Action |
|---------------------|--------------------------|--------|
| Yes | Yes | **Ship it** |
| Yes | No | Probably not worth shipping |
| No | — | Need more data or accept null |
| No | N/A (underpowered) | Run longer or increase traffic |

> **Interview answer:** "I always look at both the p-value and the effect size with its confidence interval. A tiny p-value with a negligible effect size in a massive sample doesn't warrant action."

---

## 6.7 Common Pitfalls

### Peeking Problem

**What:** Checking results repeatedly during the test and stopping early when p < 0.05.

**Why it's bad:** Inflates Type I error. If you check daily for 30 days, the effective $\alpha$ can exceed 30%.

**Solution:** Pre-commit to a fixed sample size/duration, or use **sequential testing** methods.

### Multiple Testing Problem

**What:** Testing many variants or metrics simultaneously without correction.

**Why it's bad:** With 20 independent tests at $\alpha = 0.05$, expected false positives = 1.

**Corrections:**

| Method | Approach | Strictness |
|--------|----------|-----------|
| **Bonferroni** | $\alpha_{\text{adj}} = \alpha / m$ | Most conservative |
| **Holm-Bonferroni** | Step-down: order p-values, compare to $\alpha/(m-k+1)$ | Less conservative than Bonferroni |
| **Benjamini-Hochberg (FDR)** | Controls false discovery rate instead of FWER | Least conservative |

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Simulated p-values from testing 10 metrics
p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.12, 0.25, 0.45, 0.67, 0.89])

# Bonferroni correction
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print("Bonferroni:")
for i, (p, adj_p, rej) in enumerate(zip(p_values, pvals_bonf, reject_bonf)):
    print(f"  Metric {i+1}: p={p:.3f}, adj_p={adj_p:.3f}, reject={rej}")

# Benjamini-Hochberg (FDR)
reject_bh, pvals_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print("\nBenjamini-Hochberg (FDR):")
for i, (p, adj_p, rej) in enumerate(zip(p_values, pvals_bh, reject_bh)):
    print(f"  Metric {i+1}: p={p:.3f}, adj_p={adj_p:.3f}, reject={rej}")
```

### Simpson's Paradox

**What:** A trend that appears in several groups reverses when the groups are combined.

| | Desktop | Mobile | Combined |
|---|---|---|---|
| **Control** | 20/100 (20%) | 50/400 (12.5%) | 70/500 (14%) |
| **Treatment** | 80/400 (20%) | 15/100 (15%) | 95/500 (19%) |

Treatment looks better overall (19% vs 14%), but within each device segment, the rates are similar or control is better for desktop. The difference is driven by the mix of traffic.

**Solution:** Always segment your analysis. Use stratified analysis or include covariates in your model.

### Novelty and Primacy Effects

- **Novelty effect:** Users interact more with a new feature simply because it's new (effect fades)
- **Primacy effect:** Users resist change initially (effect disappears as they adapt)

**Solution:** Run tests long enough (2-4 weeks), analyze by cohort.

### Network Effects and Interference (SUTVA Violations)

**SUTVA:** Stable Unit Treatment Value Assumption — one user's treatment doesn't affect another user's outcome.

Violated in social networks, marketplaces, ride-sharing (if treatment users interact with control users).

**Solution:** Cluster randomization (randomize at the group level, e.g., by city or network cluster).

---

## 6.8 Sequential Testing

**Problem:** Classic fixed-sample tests don't allow early stopping without inflating $\alpha$.

**Solution:** Sequential testing methods control Type I error while allowing continuous monitoring.

### Group Sequential Methods

- Pre-define interim analysis points (e.g., at 25%, 50%, 75%, 100% of target sample)
- Use adjusted significance thresholds at each look (O'Brien-Fleming, Pocock boundaries)

| Method | Early Looks | Final Look | Style |
|--------|------------|------------|-------|
| O'Brien-Fleming | Very strict (hard to stop early) | Close to 0.05 | Conservative early |
| Pocock | Equal threshold at each look | Below 0.05 | Equal spending |

### Always Valid p-values

More modern approach: compute p-values that are valid at any stopping time. Used at tech companies like Netflix and Spotify.

---

## 6.9 Bayesian A/B Testing

Instead of frequentist hypothesis testing, the Bayesian approach directly computes the **probability that B is better than A**.

### Framework

1. **Prior:** Start with a prior distribution for each variant's conversion rate (e.g., Beta(1, 1) — uniform)
2. **Update:** After observing data, compute the posterior using Bayes' theorem
3. **Decision:** Compute $P(\text{treatment} > \text{control})$

### Beta-Binomial Model

For conversion rate testing:

- Prior: $p \sim \text{Beta}(\alpha_0, \beta_0)$
- Data: $k$ conversions in $n$ trials
- Posterior: $p \mid \text{data} \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

```python
from scipy import stats
import numpy as np

# Bayesian A/B test for conversion rates
# Prior: Beta(1, 1) — uninformative
alpha_prior, beta_prior = 1, 1

# Observed data
n_control, conv_control = 5000, 500       # 10% conversion
n_treatment, conv_treatment = 5000, 575   # 11.5% conversion

# Posterior distributions
posterior_control = stats.beta(
    alpha_prior + conv_control,
    beta_prior + n_control - conv_control
)
posterior_treatment = stats.beta(
    alpha_prior + conv_treatment,
    beta_prior + n_treatment - conv_treatment
)

# Monte Carlo simulation: P(treatment > control)
n_simulations = 100_000
samples_control = posterior_control.rvs(n_simulations)
samples_treatment = posterior_treatment.rvs(n_simulations)

prob_treatment_better = np.mean(samples_treatment > samples_control)
expected_lift = np.mean((samples_treatment - samples_control) / samples_control) * 100

print(f"P(Treatment > Control) = {prob_treatment_better:.4f}")
print(f"Expected relative lift: {expected_lift:.2f}%")
print(f"95% credible interval for control rate: "
      f"({posterior_control.ppf(0.025):.4f}, {posterior_control.ppf(0.975):.4f})")
print(f"95% credible interval for treatment rate: "
      f"({posterior_treatment.ppf(0.025):.4f}, {posterior_treatment.ppf(0.975):.4f})")
```

### Bayesian vs. Frequentist A/B Testing

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Output | p-value, CI | Posterior distribution, P(B > A) |
| Interpretation | "Reject or fail to reject at $\alpha$" | "90% probability treatment is better" |
| Sample size | Fixed, pre-determined | Flexible — can stop anytime |
| Peeking | Invalid without correction | Valid — posterior is always valid |
| Prior needed | No | Yes (can use uninformative) |
| Complexity | Simpler | More computationally expensive |
| Common at | Google, LinkedIn | Netflix, VWO, Dynamic Yield |

---

## 6.10 Multi-Armed Bandit

An **adaptive** alternative to A/B testing that balances exploration (learning which variant is best) and exploitation (allocating traffic to the winner).

### Epsilon-Greedy

- With probability $\epsilon$: explore (random variant)
- With probability $1-\epsilon$: exploit (best-performing variant)

### Thompson Sampling

1. For each variant, maintain a Beta posterior for conversion rate
2. Sample from each posterior
3. Serve the variant with the highest sampled value
4. Update posterior with observed outcome

```python
import numpy as np
from scipy import stats

class ThompsonSamplingAB:
    def __init__(self, n_variants=2):
        self.alphas = np.ones(n_variants)  # successes + 1
        self.betas = np.ones(n_variants)   # failures + 1
    
    def select_variant(self):
        """Sample from each posterior, return the variant with highest sample."""
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)
    
    def update(self, variant, reward):
        """Update posterior based on observed reward (0 or 1)."""
        if reward == 1:
            self.alphas[variant] += 1
        else:
            self.betas[variant] += 1
    
    def get_stats(self):
        for i, (a, b) in enumerate(zip(self.alphas, self.betas)):
            mean = a / (a + b)
            print(f"Variant {i}: α={a:.0f}, β={b:.0f}, "
                  f"estimated rate={mean:.4f}")

# Simulation
np.random.seed(42)
bandit = ThompsonSamplingAB(n_variants=2)
true_rates = [0.10, 0.12]  # true conversion rates

for t in range(10000):
    variant = bandit.select_variant()
    reward = np.random.binomial(1, true_rates[variant])
    bandit.update(variant, reward)

bandit.get_stats()
```

### A/B Test vs. Multi-Armed Bandit

| Aspect | A/B Test | Multi-Armed Bandit |
|--------|----------|-------------------|
| Traffic allocation | Fixed 50/50 | Adaptive — shifts to winner |
| Regret | Higher (equal traffic to losing variant) | Lower (minimizes opportunity cost) |
| Statistical rigor | Full hypothesis testing framework | Weaker inferential guarantees |
| When to use | Need definitive causal answer | Want to minimize lost conversions during test |
| Common use | Feature launches, major changes | Content optimization, ad selection |

---

## 6.11 When to Stop a Test

| Criterion | Stop when... |
|-----------|-------------|
| Pre-determined sample size reached | $n$ per group hits the power-calculated target |
| Pre-determined duration reached | Minimum 1-2 weeks to capture weekly patterns |
| Sequential testing boundary crossed | For group sequential or always-valid methods |
| Bayesian: high confidence | P(B > A) > 0.95 or expected loss < threshold |
| Guardrail metric violated | Critical metric (e.g., error rate) degrades significantly |
| SRM detected | Experiment integrity compromised — investigate first |

> **Never stop because:** "The p-value just hit 0.049" mid-experiment without pre-planned sequential testing.

---

# 7. Regression Analysis

---

## 7.1 Linear Regression

### Model

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

### Assumptions (LINE)

| Letter | Assumption | Test/Check |
|--------|-----------|-----------|
| **L** — Linearity | Relationship between $X$ and $Y$ is linear | Residual plots, component plots |
| **I** — Independence | Observations are independent | Study design, Durbin-Watson test (autocorrelation) |
| **N** — Normality | Residuals are normally distributed | Q-Q plot, Shapiro-Wilk test |
| **E** — Equal variance (Homoscedasticity) | Residual variance is constant | Residual vs fitted plot, Breusch-Pagan test |

### OLS Estimation

Minimize the sum of squared residuals:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

**R-squared:**

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
$$

**Adjusted R-squared** (penalizes additional predictors):

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

### Interpretation of Coefficients

$\hat{\beta}_j$: Holding all other variables constant, a one-unit increase in $x_j$ is associated with a $\hat{\beta}_j$ change in $y$.

```python
import statsmodels.api as sm
import numpy as np

np.random.seed(42)
n = 200
X = np.random.randn(n, 3)
y = 5 + 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 2

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

---

## 7.2 Multicollinearity

When predictors are highly correlated, coefficient estimates become unstable.

### Variance Inflation Factor (VIF)

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the R-squared from regressing $x_j$ on all other predictors.

| VIF | Interpretation |
|-----|---------------|
| 1 | No multicollinearity |
| 1-5 | Moderate — usually acceptable |
| 5-10 | High — investigate |
| > 10 | Severe — action needed (drop variable, PCA, regularization) |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

X = pd.DataFrame(np.random.randn(200, 4), columns=['x1', 'x2', 'x3', 'x4'])
X['x5'] = X['x1'] * 0.9 + np.random.randn(200) * 0.1  # highly correlated with x1

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

---

## 7.3 Logistic Regression

For binary outcomes. Models the **log-odds** as a linear function:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

$$
P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}
$$

### Coefficient Interpretation

$e^{\beta_j}$ = **odds ratio**: a one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$, holding all else constant.

- $e^{\beta_j} > 1$: increases odds of $y=1$
- $e^{\beta_j} < 1$: decreases odds of $y=1$
- $e^{\beta_j} = 1$: no effect

### Loss Function

Binary cross-entropy (log loss):

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
$$

---

## 7.4 Regularization in Regression

| Method | Penalty | Effect |
|--------|---------|--------|
| **Ridge (L2)** | $\lambda \sum \beta_j^2$ | Shrinks coefficients toward zero; keeps all features |
| **Lasso (L1)** | $\lambda \sum \|\beta_j\|$ | Can shrink coefficients to exactly zero → feature selection |
| **Elastic Net** | $\alpha \lambda \sum \|\beta_j\| + \frac{(1-\alpha)}{2} \lambda \sum \beta_j^2$ | Combines L1 and L2 |

**When to use:**
- Many features → Lasso (automatic feature selection)
- Correlated features → Ridge (more stable) or Elastic Net
- Default starting point → Elastic Net

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=50, n_informative=10, noise=10)

ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)

print(f"Ridge non-zero coefficients: {sum(abs(ridge.coef_) > 0.01)}")
print(f"Lasso non-zero coefficients: {sum(abs(lasso.coef_) > 0.01)}")
print(f"Elastic Net non-zero coefficients: {sum(abs(enet.coef_) > 0.01)}")
```

---

# 8. Bayesian Statistics

---

## 8.1 Bayesian Inference Framework

### Core Equation

$$
P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}
$$

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
$$

| Component | Symbol | Role |
|-----------|--------|------|
| Prior | $P(\theta)$ | Belief about parameter before seeing data |
| Likelihood | $P(\text{data} \mid \theta)$ | Probability of observed data given parameter |
| Posterior | $P(\theta \mid \text{data})$ | Updated belief after seeing data |
| Evidence | $P(\text{data})$ | Normalizing constant (often intractable) |

### Interpretation

- **Prior:** Encodes domain knowledge, previous experiments, or uncertainty
- **Likelihood:** What the current data tells us
- **Posterior:** The synthesis — prior belief updated by evidence
- With more data, the prior becomes less influential (posterior is dominated by likelihood)

---

## 8.2 Bayesian vs. Frequentist

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameter | Fixed but unknown | Random variable with a distribution |
| Probability of | Long-run frequency | Degree of belief |
| $P(\theta \mid \text{data})$ | Not defined | Directly computed |
| Confidence/Credible interval | "95% of such intervals contain $\theta$" | "95% probability $\theta$ is in this interval" |
| Prior information | Not used | Incorporated formally |
| Small samples | Can be unreliable | Handles well (prior regularizes) |
| Computational cost | Usually low | Often high (MCMC) |
| Hypothesis testing | p-values | Bayes factors, posterior probabilities |

> **Interview answer:** "I use frequentist methods when I need quick, well-understood tests with clear false positive control — like standard A/B testing. I use Bayesian methods when I want to incorporate prior knowledge, need interpretable probability statements, or want to monitor experiments continuously."

---

## 8.3 Conjugate Priors

When the posterior has the same distributional form as the prior, the prior is called **conjugate** to the likelihood.

| Likelihood | Conjugate Prior | Posterior |
|-----------|----------------|-----------|
| Bernoulli/Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known variance) | Normal | Normal |
| Normal (known mean) | Inverse-Gamma | Inverse-Gamma |
| Multinomial | Dirichlet | Dirichlet |
| Exponential | Gamma | Gamma |

### Example: Beta-Binomial

- Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$
- Data: $k$ successes in $n$ trials
- Posterior: $\theta \mid \text{data} \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Prior: mild belief that conversion is around 10%
alpha_prior, beta_prior = 10, 90  # Beta(10, 90) → mean = 0.1

# Observed: 60 conversions in 500 users
k, n = 60, 500

# Posterior
alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)

x = np.linspace(0, 0.25, 200)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, stats.beta.pdf(x, alpha_prior, beta_prior), label='Prior', linewidth=2)
ax.plot(x, stats.beta.pdf(x, k + 1, n - k + 1), label='Likelihood (rescaled)', 
        linewidth=2, linestyle='--')
ax.plot(x, stats.beta.pdf(x, alpha_post, beta_post), label='Posterior', linewidth=2)
ax.axvline(k/n, color='gray', linestyle=':', label=f'MLE = {k/n:.3f}')
ax.legend()
ax.set_xlabel('Conversion Rate (θ)')
ax.set_ylabel('Density')
ax.set_title('Bayesian Updating: Beta-Binomial')
plt.show()

print(f"Prior mean: {alpha_prior / (alpha_prior + beta_prior):.4f}")
print(f"MLE: {k/n:.4f}")
print(f"Posterior mean: {alpha_post / (alpha_post + beta_post):.4f}")
print(f"95% Credible Interval: ({stats.beta.ppf(0.025, alpha_post, beta_post):.4f}, "
      f"{stats.beta.ppf(0.975, alpha_post, beta_post):.4f})")
```

---

## 8.4 MAP Estimation

**Maximum A Posteriori (MAP):** The mode of the posterior distribution.

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \text{data}) = \arg\max_\theta \left[\log P(\text{data} \mid \theta) + \log P(\theta)\right]
$$

### MAP vs. MLE

| Method | Objective | Regularization |
|--------|----------|---------------|
| MLE | $\arg\max_\theta P(\text{data} \mid \theta)$ | None |
| MAP | $\arg\max_\theta P(\text{data} \mid \theta) P(\theta)$ | Prior acts as regularizer |

**Connection to regularization:**
- Gaussian prior on weights → **L2 regularization** (Ridge)
- Laplacian prior on weights → **L1 regularization** (Lasso)

> **Interview insight:** "Ridge regression is equivalent to MAP estimation with a Gaussian prior on the coefficients. Lasso is equivalent to MAP estimation with a Laplace prior."

---

# 9. Common Statistical Concepts in ML

---

## 9.1 Maximum Likelihood Estimation (MLE)

### Principle

Choose parameter values that maximize the probability of the observed data:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^{n} P(x_i \mid \theta)
$$

In practice, maximize the **log-likelihood** (converts products to sums):

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i \mid \theta)
$$

### Example: MLE for Normal Distribution

Given i.i.d. samples $x_1, \dots, x_n$ from $\mathcal{N}(\mu, \sigma^2)$:

$$
\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum x_i
$$

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum (x_i - \bar{x})^2
$$

Note: The MLE for variance divides by $n$ (biased), not $n-1$.

### MLE in ML

- Logistic regression: minimizing binary cross-entropy = maximizing log-likelihood of Bernoulli model
- Linear regression: minimizing MSE = maximizing log-likelihood of Gaussian model
- Neural network classification with softmax: cross-entropy loss = negative log-likelihood of categorical distribution

---

## 9.2 Cross-Entropy

### Information Theory Origin

**Information content** of an event with probability $p$:

$$
I(p) = -\log_2(p) \quad \text{(in bits)}
$$

Rare events carry more information.

### Entropy

Average information content of a distribution:

$$
H(p) = -\sum_x p(x) \log p(x)
$$

| Distribution | Entropy |
|-------------|---------|
| Deterministic (one outcome has prob 1) | 0 (no uncertainty) |
| Uniform over $k$ outcomes | $\log k$ (maximum uncertainty) |
| Biased coin $p = 0.9$ | 0.47 bits |
| Fair coin $p = 0.5$ | 1.0 bit |

### Cross-Entropy

Measures how well distribution $q$ approximates the true distribution $p$:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

- Always $H(p, q) \geq H(p)$
- Minimizing cross-entropy → making $q$ as close to $p$ as possible

**In ML classification:** $p$ = true labels (one-hot), $q$ = model's predicted probabilities.

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})
$$

For binary classification:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]
$$

---

## 9.3 KL Divergence

Measures how one distribution $q$ diverges from a reference distribution $p$:

$$
D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

**Properties:**
- $D_{\text{KL}}(p \| q) \geq 0$ (Gibbs' inequality)
- $D_{\text{KL}}(p \| q) = 0 \iff p = q$
- **NOT symmetric:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general
- Not a true distance metric (no triangle inequality, no symmetry)

**In ML:**
- **VAEs:** KL divergence between the encoder's posterior and the prior $\mathcal{N}(0, I)$
- **Knowledge distillation:** KL divergence between teacher and student softmax outputs
- **Policy gradient (RL):** KL constraint in TRPO/PPO

**Relationship:**

$$
\text{Cross-Entropy} = \text{Entropy} + \text{KL Divergence}
$$

Since entropy of true labels is constant, minimizing cross-entropy = minimizing KL divergence.

---

## 9.4 Mutual Information

Quantifies the amount of information obtained about one variable by observing another:

$$
I(X; Y) = D_{\text{KL}}(P(X,Y) \| P(X)P(Y)) = H(X) + H(Y) - H(X, Y)
$$

- $I(X; Y) = 0 \iff X$ and $Y$ are independent
- $I(X; Y) = H(X) \iff Y$ completely determines $X$

**In ML:** Feature selection (mutual information between features and target), representation learning (InfoNCE loss in contrastive learning).

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, random_state=42)
mi_scores = mutual_info_classif(X, y, random_state=42)

for i, score in enumerate(mi_scores):
    print(f"Feature {i}: MI = {score:.4f}")
```

---

## 9.5 Bias-Variance Decomposition

For any estimator, the expected prediction error decomposes as:

$$
E\left[(y - \hat{f}(x))^2\right] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2_\epsilon
$$

| Component | Definition | Source |
|-----------|-----------|--------|
| Bias$^2$ | $(E[\hat{f}(x)] - f(x))^2$ | Systematic error from wrong assumptions |
| Variance | $E[(\hat{f}(x) - E[\hat{f}(x)])^2]$ | Sensitivity to training data fluctuations |
| Irreducible error | $\sigma^2_\epsilon$ | Noise in the data itself |

### Bias-Variance Tradeoff

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Low (e.g., linear model) | High | Low | Underfitting |
| Optimal | Balanced | Balanced | Minimum |
| High (e.g., deep tree) | Low | High | Overfitting |

**Regularization reduces variance** (at the cost of slightly increased bias) → often reduces total error.

---

# 10. Common Interview Questions

---

## Q1: "Explain p-value in simple terms."

> **Answer:** "The p-value is the probability of seeing results as extreme as what we observed, if the null hypothesis — that there is no effect — were actually true. If this probability is very small (below our threshold, usually 0.05), we conclude the data is inconsistent with 'no effect' and reject the null. It is NOT the probability that the null is true."

---

## Q2: "How would you design an A/B test for a new checkout flow?"

> **Answer:**  
> 1. **Hypothesis:** "The new checkout flow will increase purchase completion rate."
> 2. **Primary metric:** Checkout conversion rate (users who start checkout → complete purchase).
> 3. **Guardrail metrics:** Revenue per user, page load time, error rates.
> 4. **Sample size:** Calculate using power analysis — baseline 30% conversion, want to detect 2pp lift, $\alpha = 0.05$, power = 0.80 → compute required $n$ per group.
> 5. **Randomization:** Hash-based on user ID. Ensure new and returning users are balanced (stratified).
> 6. **Duration:** At least 2 full weeks to capture weekday/weekend effects.
> 7. **Pre-checks:** SRM validation to ensure 50/50 split.
> 8. **Analysis:** Two-proportion z-test for conversion rate. Welch's t-test for revenue per user. Report effect size, CI, and p-value.
> 9. **Decision:** Require both statistical significance (p < 0.05) AND practical significance (business-meaningful lift).

---

## Q3: "What is the difference between Type I and Type II errors?"

> **Answer:**  
> - **Type I (False Positive):** Rejecting the null when it's true. Controlled by $\alpha$ (significance level). Example: Declaring a new feature improves conversion when it actually doesn't.
> - **Type II (False Negative):** Failing to reject the null when it's false. Related to power ($1 - \beta$). Example: Missing a real improvement because the test was underpowered.
> - The tradeoff: lowering $\alpha$ reduces Type I errors but increases Type II errors. The only way to reduce both is to increase sample size.

---

## Q4: "What is the Central Limit Theorem and why does it matter?"

> **Answer:** "The CLT states that the distribution of the sample mean approaches a normal distribution as sample size increases, regardless of the underlying population distribution (as long as it has finite variance). This is why we can use z-tests and t-tests for A/B testing even when the underlying data is not normal — we're testing the sample mean, which IS approximately normal for large $n$. It's the theoretical foundation of most frequentist hypothesis tests."

---

## Q5: "How do you determine sample size for an A/B test?"

> **Answer:** "I use a power analysis with four inputs: (1) the baseline metric value, (2) the minimum detectable effect (MDE) — the smallest change worth detecting, (3) significance level $\alpha$ (usually 0.05), and (4) desired power (usually 80%). The key tradeoff is that detecting smaller effects requires much larger samples — halving the MDE requires roughly 4x the sample size. I use Python's `statsmodels` power analysis functions to compute this."

---

## Q6: "Bayesian vs. frequentist — when would you use each?"

> **Answer:**  
> - **Frequentist:** When I need well-established, industry-standard methods with clear false positive guarantees — standard A/B testing, regulatory contexts, simple hypothesis tests.
> - **Bayesian:** When I want to incorporate prior knowledge (e.g., results from previous experiments), need direct probability statements ("90% chance treatment is better"), want to monitor experiments continuously without peeking problems, or am working with small samples where priors provide meaningful regularization.
> - In practice, I often use frequentist methods for initial test design and sample size calculation, and Bayesian analysis for richer inference and continuous monitoring.

---

## Q7: "What is Simpson's paradox? Give a real example."

> **Answer:** "Simpson's paradox occurs when a trend that appears in several sub-groups reverses when the groups are combined. Classic example: a treatment may look better overall, but if you segment by gender or device type, control is actually better in every sub-group. The reversal happens because of an imbalance in how subjects are distributed across groups. In A/B testing, this means I always segment my analysis — I don't just look at the aggregate metric."

---

## Q8: "How do you handle multiple comparisons?"

> **Answer:** "When testing multiple hypotheses simultaneously (e.g., 10 metrics in an A/B test), the probability of at least one false positive increases dramatically. With 20 independent tests at $\alpha = 0.05$, the family-wise error rate is about 64%. Solutions:
> - **Bonferroni correction:** Divide $\alpha$ by the number of tests. Simple but very conservative.
> - **Holm-Bonferroni:** Step-down procedure — less conservative than Bonferroni while still controlling FWER.
> - **Benjamini-Hochberg:** Controls the false discovery rate (FDR) instead of FWER — more powerful, appropriate for exploratory analyses.
> - **Pre-specify one primary metric:** The cleanest approach — one OEC doesn't need correction."

---

## Q9: "Explain the difference between correlation and causation."

> **Answer:** "Correlation measures the strength of a statistical association between two variables. Causation means one variable actually influences the other. Correlation does not imply causation because of confounders — ice cream sales and drowning deaths are correlated because both increase in summer (confound: temperature), not because one causes the other. Establishing causation requires either (1) a randomized controlled experiment (A/B test) that eliminates confounders through random assignment, or (2) careful observational study designs with causal inference methods (instrumental variables, difference-in-differences, regression discontinuity)."

---

## Q10: "A test shows p = 0.06. What do you do?"

> **Answer:** "I would NOT call this 'almost significant' and ship the change. A p-value just above 0.05 is not evidence for the alternative hypothesis. I would:
> 1. Check the confidence interval — does it include practically meaningful effects?
> 2. Look at the effect size — how large is the observed difference?
> 3. Consider whether the test was adequately powered — if underpowered, I might extend the test.
> 4. Look at segment-level data for patterns (while being careful about multiple testing).
> 5. If the effect size is potentially meaningful and the test was underpowered, I'd recommend running a follow-up experiment with a larger sample.
> 6. I would never lower $\alpha$ post-hoc to make results significant."

---

## Q11: "What is the Law of Large Numbers and how is it different from CLT?"

> **Answer:** "The Law of Large Numbers (LLN) says the sample mean converges to the true population mean as $n$ increases — it guarantees accuracy. The CLT goes further by specifying that the distribution of the sample mean is approximately normal and quantifies how fast the convergence happens ($\sigma/\sqrt{n}$). LLN tells you WHERE the sample mean goes, CLT tells you what SHAPE the distribution of sample means takes."

---

## Q12: "When would you use a non-parametric test?"

> **Answer:** "I use non-parametric tests (Mann-Whitney U, Wilcoxon, Kruskal-Wallis) when: (1) the data is clearly non-normal and the sample is small (CLT doesn't help), (2) the data is ordinal (rankings, Likert scales), (3) there are extreme outliers that distort means, or (4) the distribution has heavy tails. The tradeoff: non-parametric tests are generally less powerful than their parametric counterparts when the parametric assumptions hold — so if data is approximately normal, I prefer t-tests/ANOVA."

---

# 11. Key Takeaways

---

### Probability

- Bayes' theorem is the foundation for updating beliefs with evidence — master the medical test example.
- Independence and conditional independence are distinct and critical for Naive Bayes and graphical models.
- Correlation ≠ independence. Zero correlation implies no LINEAR relationship; the variables can still be dependent.

### Distributions

- Know when to use each distribution. Bernoulli → single trial, Binomial → count of successes, Poisson → rare events, Normal → CLT-driven inference, Beta → modeling probabilities, Exponential → time between events.
- CLT is the most important theorem: it justifies almost all frequentist hypothesis tests.

### Hypothesis Testing

- p-value is NOT the probability of $H_0$ being true — drill this.
- Always report effect sizes and confidence intervals alongside p-values.
- Know the test selection flowchart: type of data × number of groups × parametric or not.

### A/B Testing

- Design BEFORE you run: hypothesis, primary metric, sample size, duration.
- Never peek without sequential testing methods.
- Correct for multiple comparisons (Bonferroni, BH-FDR).
- Statistical significance ≠ practical significance — always check both.
- Bayesian A/B testing gives directly interpretable probabilities and avoids peeking problems.

### Bayesian

- Prior × Likelihood ∝ Posterior — master this formula.
- Conjugate priors give closed-form posteriors.
- MAP with Gaussian prior = Ridge; MAP with Laplace prior = Lasso.

### ML Connections

- MLE is the foundation: cross-entropy loss = negative log-likelihood.
- KL divergence connects to cross-entropy: minimizing one minimizes the other.
- Bias-variance tradeoff explains why regularization works.

---

**Rahul's Competitive Edge:**
- 4.0 GPA in Probability & Statistics at UMD — deep mathematical foundations
- A/B Testing experience — can design, execute, and analyze experiments end-to-end
- Python proficiency with scipy, statsmodels, numpy — can code any test or simulation
- Bayesian AND frequentist competency — can choose the right framework for each situation
- Statistical rigor in ML work — understands why loss functions work, how regularization connects to priors

---

*Prepared for Rahul Sharma | Data Scientist Interview Preparation | Statistics & A/B Testing*
