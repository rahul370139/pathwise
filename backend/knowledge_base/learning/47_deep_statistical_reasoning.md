# Deep Statistical Reasoning — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Probabilistic Reasoning, Estimator Properties, Convergence, Likelihood, Asymptotics, Inference Foundations

> **Context:** Files `24_statistics_and_ab_testing.md` and `33_causal_inference_and_experimentation.md` cover the *methods*. This guide covers the *foundations* — the deep statistical reasoning that lets you derive new estimators, justify them, and defend them in front of skeptical interviewers. This is the difference between "I know t-tests" and "I can think from first principles when the question doesn't fit a textbook recipe." It's also exactly what senior ICs at OpenAI, Meta Research, Microsoft Research, Google Research, and Netflix Causal Inference are tested on.

---

## Table of Contents

1. [Why Deep Statistical Reasoning Matters](#1-why-deep-statistical-reasoning-matters)
2. [Estimators — Bias, Variance, Consistency, Efficiency](#2-estimators)
3. [Likelihood and MLE — From Scratch](#3-likelihood-and-mle)
4. [The Information Inequality and Cramér-Rao](#4-the-information-inequality-and-cramer-rao)
5. [Asymptotic Theory — LLN, CLT, Delta Method](#5-asymptotic-theory)
6. [Bayesian vs Frequentist — The Real Debate](#6-bayesian-vs-frequentist)
7. [Bootstrap — Why It Works](#7-bootstrap)
8. [Multiple Testing — FWER vs FDR](#8-multiple-testing)
9. [Type-I, Type-II, and Power — Deeply](#9-type-i-type-ii-and-power)
10. [P-Values — What They Are and Are Not](#10-p-values)
11. [The Likelihood Ratio Test and Wald, Score, LR Trio](#11-the-lr-wald-score-trio)
12. [Identification vs Estimation](#12-identification-vs-estimation)
13. [Concentration Inequalities](#13-concentration-inequalities)
14. [Interview Questions with Strong Answers](#14-interview-questions-with-strong-answers)
15. [Key Takeaways](#15-key-takeaways)

---

# 1. Why Deep Statistical Reasoning Matters

## 1.1 The Senior IC Bar

A senior IC at OpenAI / Meta / Microsoft Research isn't asked "what's a t-test?". They're asked:

- "Your A/B test metric is heavy-tailed. The t-test assumes normality. Is the CI you computed valid? Justify."
- "We're moving from N=10M to N=1B. Does anything change about how we should run statistical tests? Why?"
- "Your CUPED-adjusted CI is narrower. Did you really gain information, or are you fooling yourself?"
- "If we use Bayesian sequential testing instead of frequentist, what changes about our error guarantees?"

Each of these requires *thinking from foundations*. This guide gives you that scaffolding.

## 1.2 The Mental Model

Statistics is built on three pillars:

```
┌─────────────────────────────────────────────────────────────────┐
│             THE THREE PILLARS OF STATISTICAL REASONING           │
│                                                                  │
│  1. PROBABILITY                                                  │
│     A model of uncertainty. Random variables, distributions.    │
│                                                                  │
│  2. ESTIMATION                                                   │
│     Given data, what is the parameter? What's its uncertainty?  │
│                                                                  │
│  3. INFERENCE                                                    │
│     Given an estimate, what claims can we make about the world? │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Most interview "trick" questions are testing whether you can move between pillars cleanly.

---

# 2. Estimators

## 2.1 Definition

An **estimator** $\hat{\theta}_n = T(X_1, \dots, X_n)$ is a function of data that produces a guess at parameter $\theta$.

## 2.2 The Four Properties

| Property | Math | Meaning |
|----------|------|---------|
| **Unbiased** | $\mathbb{E}[\hat{\theta}] = \theta$ | On average, correct |
| **Consistent** | $\hat{\theta}_n \xrightarrow{p} \theta$ | Converges to truth as n grows |
| **Efficient** | Achieves Cramér-Rao lower bound | Minimum variance among unbiased estimators |
| **Sufficient** | Captures all data info about $\theta$ | Can throw away rest of data |

## 2.3 Unbiased ≠ Consistent

A famous gotcha:

- An estimator can be unbiased but not consistent: $\hat{\theta} = X_1$ is unbiased but doesn't shrink with n.
- An estimator can be biased but consistent: $\hat{\theta} = (1/n) \sum X_i + 1/n$ is biased by $1/n$ but converges to truth as $n \to \infty$.

## 2.4 Mean Squared Error

$$
\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2
$$

This is **the** fundamental trade-off. James-Stein, Ridge, Lasso, Bayesian shrinkage all trade bias for variance to reduce MSE.

## 2.5 Example: Sample Variance

| Estimator | Formula | Bias |
|-----------|---------|------|
| $s^2_{\text{biased}}$ | $\frac{1}{n}\sum(X_i - \bar{X})^2$ | Underestimates true variance by factor $(n-1)/n$ |
| $s^2_{\text{unbiased}}$ | $\frac{1}{n-1}\sum(X_i - \bar{X})^2$ | Unbiased (the famous "Bessel correction") |

The reason: using $\bar{X}$ instead of $\mu$ "uses up" a degree of freedom.

---

# 3. Likelihood and MLE

## 3.1 The Likelihood Function

Given data $\mathbf{X} = (X_1, \dots, X_n)$ and a parametric model $f(x; \theta)$:

$$
L(\theta; \mathbf{X}) = \prod_{i=1}^n f(X_i; \theta)
$$

This is **the probability of the observed data**, viewed as a function of $\theta$.

## 3.2 Log-Likelihood

$$
\ell(\theta; \mathbf{X}) = \sum_{i=1}^n \log f(X_i; \theta)
$$

Always work with log-likelihood — it's numerically stable and turns products into sums.

## 3.3 Maximum Likelihood Estimator

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta; \mathbf{X})
$$

Solve by setting the **score function** (derivative of log-likelihood) to zero:

$$
\frac{\partial \ell}{\partial \theta} = 0
$$

## 3.4 Properties of MLE

| Property | Holds when |
|----------|-----------|
| **Consistent** | Almost always (under mild regularity) |
| **Asymptotically normal** | Most regular cases |
| **Asymptotically efficient** | Achieves Cramér-Rao bound asymptotically |
| **Invariant** | If $\hat{\theta}_{\text{MLE}}$ is MLE, then $g(\hat{\theta}_{\text{MLE}})$ is MLE of $g(\theta)$ |

MLE may be **biased** in finite samples. The famous example: MLE of variance in a normal is $\frac{1}{n}\sum(X_i-\bar{X})^2$, biased by factor $\frac{n-1}{n}$.

## 3.5 MLE for Normal

$X_i \sim \mathcal{N}(\mu, \sigma^2)$. Log-likelihood:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(X_i-\mu)^2
$$

Setting derivatives to zero:

$$
\hat{\mu}_{\text{MLE}} = \bar{X}, \quad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i-\bar{X})^2
$$

## 3.6 MLE for Logistic Regression

Log-likelihood for binary Y:

$$
\ell(\beta) = \sum_i \left[ Y_i \log p_i + (1-Y_i) \log(1-p_i) \right]
$$

with $p_i = \sigma(X_i^\top \beta)$. Solved via iteratively-reweighted least squares or gradient descent. No closed form.

---

# 4. The Information Inequality and Cramér-Rao

## 4.1 Fisher Information

$$
I(\theta) = -\mathbb{E}\left[ \frac{\partial^2 \ell}{\partial \theta^2} \right] = \mathbb{E}\left[ \left( \frac{\partial \ell}{\partial \theta} \right)^2 \right]
$$

The expected curvature of the log-likelihood. Large $I(\theta)$ means the data sharply pinpoints $\theta$; small means there's a lot of uncertainty.

## 4.2 Cramér-Rao Lower Bound

For any unbiased estimator $\hat{\theta}$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n \cdot I(\theta)}
$$

No estimator can do better than this — it's a fundamental information-theoretic limit.

## 4.3 Why CRLB Matters

When you propose a new estimator, the question is **how does its variance compare to CRLB?** If it's at the bound, you've done the best possible. If it's above, you're leaving information on the table.

## 4.4 Example: Bernoulli

For $X \sim \text{Bernoulli}(p)$:

$$
I(p) = \frac{1}{p(1-p)}
$$

CRLB for $\hat{p}$: $\text{Var}(\hat{p}) \geq \frac{p(1-p)}{n}$. The sample mean achieves this exactly.

## 4.5 The Score Function

$$
S(\theta) = \frac{\partial \ell}{\partial \theta}
$$

Has the property $\mathbb{E}[S(\theta_0)] = 0$ at the true value. Forms the basis of the **score test**.

---

# 5. Asymptotic Theory

## 5.1 Convergence in Probability

$$
X_n \xrightarrow{p} X \iff \forall \epsilon > 0, \quad P(|X_n - X| > \epsilon) \to 0
$$

## 5.2 Convergence in Distribution

$$
X_n \xrightarrow{d} X \iff F_n(x) \to F(x) \text{ at all continuity points of } F
$$

## 5.3 Law of Large Numbers (LLN)

For iid samples with finite mean $\mu$:

$$
\bar{X}_n \xrightarrow{p} \mu
$$

Sample averages converge to true means.

## 5.4 Central Limit Theorem (CLT)

For iid with mean $\mu$ and finite variance $\sigma^2$:

$$
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)
$$

The sample mean is approximately normal for large n, **regardless of the underlying distribution** (with finite variance).

> **CLT is why so much of statistics works.** Most estimators are sample averages or asymptotically equivalent to them.

## 5.5 Delta Method

For a smooth function $g$:

$$
\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2)
$$

Used to compute variance of transformed estimators (ratios, log-odds, etc.).

## 5.6 Delta Method for Ratios (Used in A/B Testing)

For $R = \bar{X} / \bar{Y}$:

$$
\text{Var}(R) \approx \frac{1}{\bar{Y}^2} \text{Var}(\bar{X}) + \frac{\bar{X}^2}{\bar{Y}^4} \text{Var}(\bar{Y}) - \frac{2\bar{X}}{\bar{Y}^3} \text{Cov}(\bar{X}, \bar{Y})
$$

This is **the** formula for variance of ratio metrics (CTR, conversion rate per click) in experimentation.

## 5.7 Slutsky's Theorem

If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{p} c$, then:
- $X_n + Y_n \xrightarrow{d} X + c$
- $X_n Y_n \xrightarrow{d} cX$
- $X_n / Y_n \xrightarrow{d} X/c$ (if $c \neq 0$)

This is the tool that lets you replace unknown quantities (like $\sigma^2$) with their consistent estimates (like $s^2$) when computing test statistics.

---

# 6. Bayesian vs Frequentist

## 6.1 The Core Difference

| Frequentist | Bayesian |
|-------------|---------|
| $\theta$ is a fixed unknown | $\theta$ is a random variable |
| Probability = long-run frequency | Probability = degree of belief |
| Data is random | Data is fixed (conditional on observing it) |
| Inference: confidence intervals, p-values | Inference: posteriors, credible intervals |

## 6.2 Bayes' Theorem

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} \propto L(\theta; D) \cdot \pi(\theta)
$$

- **Prior** $\pi(\theta)$: your belief before seeing data
- **Likelihood** $L(\theta; D)$: how well $\theta$ explains the data
- **Posterior** $P(\theta | D)$: your belief after seeing data

## 6.3 Conjugate Priors

Special prior-likelihood pairs where the posterior has the same form as the prior:

| Likelihood | Conjugate prior | Posterior |
|-----------|-----------------|----------|
| Bernoulli | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known σ) | Normal | Normal |
| Normal (unknown σ) | Normal-inverse-gamma | Same |

## 6.4 Credible Interval vs Confidence Interval

**Confidence interval (frequentist):** "If we repeated the experiment many times, 95% of these intervals would contain the true parameter." A statement about the procedure, not the specific interval.

**Credible interval (Bayesian):** "Given the data and prior, there is a 95% probability the parameter is in this interval." A direct statement about the parameter.

These are **not the same thing**, and conflating them is a common interview pitfall.

## 6.5 When to Be Bayesian

- Small samples where priors carry information
- Sequential / streaming decisions
- When you want direct probability statements
- Multi-armed bandits (Thompson sampling)
- Bayesian neural networks for uncertainty

## 6.6 When to Be Frequentist

- Large samples (prior doesn't matter)
- Regulatory contexts requiring fixed Type-I error
- When defaults are well-understood by the audience

---

# 7. Bootstrap

## 7.1 The Idea

Resample with replacement from your data. Compute the statistic on each resample. The distribution of bootstrap statistics approximates the sampling distribution of the original statistic.

## 7.2 Why It Works (Glivenko-Cantelli)

The empirical distribution function converges to the true CDF as n grows. So sampling from the empirical CDF (which is what bootstrap does) is asymptotically equivalent to sampling from the true distribution.

## 7.3 Algorithm

```python
import numpy as np

def bootstrap_ci(data, statistic, n_boot=10000, alpha=0.05):
    boots = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return lo, hi
```

## 7.4 Types of Bootstrap CIs

| Type | Method | When to use |
|------|--------|------------|
| **Percentile** | Just take quantiles of boot distribution | Simple, common |
| **BCa** | Bias-corrected and accelerated | Better for skewed statistics |
| **Studentized** | Pivot using SE estimate | Best coverage |

## 7.5 When Bootstrap Fails

- **Extreme order statistics** (e.g., max, min): bootstrap doesn't converge to the right distribution.
- **Heavy-tailed distributions** without finite variance.
- **Non-iid data**: must use block bootstrap (time series) or cluster bootstrap (panel).
- **Discontinuous statistics**: e.g., the median when n is small.

## 7.6 Why It's Useful in Practice

Bootstrap doesn't require knowing the analytic SE formula. For complex estimators (CUPED, weighted means, custom metrics), bootstrap is often easier and more correct than deriving SE by hand.

---

# 8. Multiple Testing

## 8.1 The Problem

Run 20 independent tests at $\alpha=0.05$. The probability of at least one false positive is:

$$
P(\text{any false +}) = 1 - 0.95^{20} \approx 0.64
$$

So 64% chance of a false discovery just by running many tests.

## 8.2 Family-Wise Error Rate (FWER)

Probability of **any** false positive across all tests.

### Bonferroni
$\alpha^* = \alpha / m$ for m tests. Conservative, simple.

### Holm-Bonferroni
Sort p-values ascending. Reject smallest if $p_{(1)} \leq \alpha/m$, next if $p_{(2)} \leq \alpha/(m-1)$, etc. Less conservative, valid under any dependence.

## 8.3 False Discovery Rate (FDR)

$$
\text{FDR} = \mathbb{E}\left[ \frac{\text{false rejections}}{\text{total rejections}} \right]
$$

The **expected proportion of false discoveries among rejections**.

### Benjamini-Hochberg (BH) Procedure

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$
2. Find largest k such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all hypotheses with $p_i \leq p_{(k)}$

Controls FDR at $\alpha$ under independence or positive regression dependence.

## 8.4 When to Use FWER vs FDR

| Setting | Use |
|---------|-----|
| Clinical trial, regulatory | FWER (any false positive is catastrophic) |
| Experiment metric panel (10-50 metrics) | FWER (Holm) |
| High-throughput screening (1000s of genes) | FDR (BH) |
| Many segment cuts in A/B test | FDR or skip multiplicity if pre-registered |

## 8.5 Pre-Registration vs Multiple Testing

If you pre-register **the** primary metric and decision rule, you don't need multiplicity correction for that metric. Multiplicity is needed when you scan many metrics or segments without pre-specifying.

---

# 9. Type-I, Type-II, and Power — Deeply

## 9.1 The Errors

| Truth \ Decision | Reject $H_0$ | Don't reject $H_0$ |
|------------------|----------------|---------------------|
| $H_0$ true | **Type I error** (α) | Correct |
| $H_0$ false | Correct | **Type II error** (β) |

- **Significance level** $\alpha$ = P(Type I) = false positive rate.
- **Power** $1 - \beta$ = P(reject $H_0$ | $H_0$ false) = true positive rate.

## 9.2 The Asymmetry

You explicitly control $\alpha$ (pick 0.05). $\beta$ emerges from sample size, effect size, and variance:

$$
n = \frac{2 (z_{\alpha/2} + z_\beta)^2 \sigma^2}{\Delta^2}
$$

## 9.3 Power Curve

Power increases with:
- **Larger sample size** (linear in $n$)
- **Larger true effect** (quadratic in $\Delta$)
- **Lower variance** (linear in $1/\sigma^2$)
- **Higher significance threshold** (more permissive $\alpha$ → more power but more false positives)

## 9.4 Why Underpowered Studies Are Worse Than No Study

Underpowered studies (β > 0.5) yield mostly null results. The few "significant" findings are heavily **biased toward exaggerated effects** (Type-S sign errors, Type-M magnitude errors — Gelman & Carlin).

> **Lesson:** if you can't run a powered study, sometimes don't run one at all — or change the design (variance reduction, longer duration).

---

# 10. P-Values

## 10.1 Definition

$p$-value = P(test statistic at least as extreme as observed | $H_0$ true).

## 10.2 What P-Values Are Not

| Misconception | Reality |
|--------------|---------|
| P($H_0$ true | data) | No — that's a Bayesian quantity |
| Effect size | No — large samples make tiny effects significant |
| Replication probability | No — p < 0.05 doesn't mean 95% chance of replicating |
| Importance | No — p-value depends on n, not just signal strength |

## 10.3 The Garden of Forking Paths

Even without explicit p-hacking, **researcher degrees of freedom** (which segment to analyze, which transformation, which model) inflate Type I error. Pre-registration is the defense.

## 10.4 Practical Recommendation (ASA 2016 Statement)

- Don't use p-values mechanically (p < 0.05 = ship).
- Always report effect sizes and CIs.
- Distinguish statistical significance from practical importance.
- Be transparent about all analyses run.

---

# 11. The LR, Wald, Score Trio

For testing $H_0: \theta = \theta_0$, there are three asymptotically equivalent tests:

## 11.1 Wald Test

$$
W = \frac{(\hat{\theta}_{\text{MLE}} - \theta_0)^2}{\text{Var}(\hat{\theta}_{\text{MLE}})} \sim \chi^2_1
$$

Fits the alternative, computes how far the MLE is from the null.

## 11.2 Likelihood Ratio Test

$$
\Lambda = -2 \log \frac{L(\theta_0)}{L(\hat{\theta}_{\text{MLE}})} \sim \chi^2_1
$$

Compares the likelihood at the null vs the MLE. **Generally most powerful**.

## 11.3 Score Test (Lagrange Multiplier)

$$
S = \frac{[\ell'(\theta_0)]^2}{I(\theta_0)} \sim \chi^2_1
$$

Only requires fitting under the null — useful when alternative model is expensive.

## 11.4 When Each Is Preferred

| Test | Use when |
|------|----------|
| **Wald** | MLE is easy, you want simple CI |
| **LR** | You want maximum power |
| **Score** | Fitting under alternative is expensive |

All three are equivalent under $H_0$ as $n \to \infty$. For small samples, LR is usually most reliable.

---

# 12. Identification vs Estimation

## 12.1 The Distinction

- **Identification**: is the parameter recoverable from the data + assumptions?
- **Estimation**: how do we compute the estimator?

A parameter can be **identified but hard to estimate** (e.g., quantile estimation in heavy-tailed data) or **unidentified yet easy to compute** something (the computation just doesn't have causal meaning).

## 12.2 In Causal Inference

- DiD identifies ATT under parallel trends.
- IV identifies LATE under exclusion and monotonicity.
- RDD identifies LATE at the threshold under continuity.

The identifying assumption is the *story*. The estimator implements the *computation*. Both matter.

---

# 13. Concentration Inequalities

## 13.1 Markov's Inequality

For any non-negative random variable X:

$$
P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}
$$

The crudest bound, but works without distributional assumptions.

## 13.2 Chebyshev's Inequality

$$
P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
$$

For any distribution with finite variance.

## 13.3 Hoeffding's Inequality

For independent bounded random variables $X_i \in [a, b]$:

$$
P\left(\left|\bar{X}_n - \mu\right| \geq t\right) \leq 2 \exp\left(-\frac{2nt^2}{(b-a)^2}\right)
$$

This is **why empirical means concentrate quickly** even without normality. Foundation for finite-sample guarantees in ML.

## 13.4 Bernstein's Inequality

Sharper than Hoeffding when variance is much smaller than the range. Common in PAC learning bounds.

## 13.5 Why These Matter

- **PAC bounds** in machine learning are concentration inequalities applied to risk.
- **Empirical-Bernstein** bounds appear in bandit algorithms.
- **Conformal prediction** uses these for distribution-free uncertainty.

---

# 14. Interview Questions with Strong Answers

## Q1: "What's the difference between unbiased and consistent?"

> Unbiased means the estimator's expected value equals the parameter for any finite n. Consistent means the estimator converges in probability to the parameter as n grows. They're independent properties. The sample mean is both. $\hat{\theta} = X_1$ is unbiased (E[X_1] = μ) but not consistent (doesn't shrink). $\hat{\theta} = \bar{X} + 1/n$ is biased by 1/n but consistent because the bias vanishes. In practice, consistency is what you usually want for large-sample work; unbiasedness is more useful for small samples or when MSE matters and bias would compound.

## Q2: "Derive the maximum likelihood estimator for the mean of a normal distribution."

> For $X_i \sim N(\mu, \sigma^2)$, the log-likelihood is $\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(X_i - \mu)^2$. Taking the derivative with respect to μ and setting to zero: $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum(X_i - \mu) = 0$ gives $\hat{\mu}_{\text{MLE}} = \bar{X}$. The MLE of μ is the sample mean. Second derivative check: $\frac{\partial^2 \ell}{\partial \mu^2} = -\frac{n}{\sigma^2} < 0$, confirming a maximum. The MLE achieves the Cramér-Rao bound — its variance is $\sigma^2/n$, and no unbiased estimator can do better.

## Q3: "What does the central limit theorem say and why is it so important in practice?"

> The CLT says that for iid random variables with finite mean μ and finite variance σ², the standardized sample mean $\sqrt{n}(\bar{X}-\mu)/\sigma$ converges in distribution to a standard normal as n grows. Regardless of the original distribution. This is enormous because almost every estimator we use — sample means, regression coefficients, ATEs in A/B tests — is asymptotically a sample mean or linear function of sample means. So at large n, we can apply normal-theory inference (CIs, t-tests) even when the data itself is non-normal. The caveats: CLT requires finite variance (fails for heavy-tailed Pareto), and "large n" depends on skewness — n=30 is often enough for symmetric data but inadequate for skewed data.

## Q4: "Explain the delta method and where you'd use it."

> The delta method gives the asymptotic distribution of a smooth function of an asymptotically normal estimator. If $\sqrt{n}(\bar{X} - \mu) \xrightarrow{d} N(0, \sigma^2)$ and g is differentiable at μ, then $\sqrt{n}(g(\bar{X}) - g(\mu)) \xrightarrow{d} N(0, [g'(\mu)]^2 \sigma^2)$. In A/B testing this is the workhorse for ratio metrics. CTR is clicks/impressions, both estimated. We need the variance of the ratio. Direct application: $\text{Var}(R) \approx \frac{1}{\bar{Y}^2}\text{Var}(\bar{X}) + \frac{\bar{X}^2}{\bar{Y}^4}\text{Var}(\bar{Y}) - \frac{2\bar{X}}{\bar{Y}^3}\text{Cov}(\bar{X}, \bar{Y})$. It's also how you get the variance of log-odds in logistic regression, the variance of survival function estimates, and SE for any transformed metric.

## Q5: "Why does the bootstrap work?"

> The empirical distribution function (EDF) — the staircase function that puts mass 1/n on each data point — converges to the true CDF as n grows (Glivenko-Cantelli theorem). Bootstrap resamples from this EDF, so asymptotically it's sampling from the true distribution. The distribution of statistics computed on bootstrap samples then approximates the sampling distribution of the statistic on real data. The practical advantage is no analytic formula needed — for complex estimators like CUPED-adjusted ratios or weighted means, bootstrap gives you a CI without algebra. The failure modes are heavy-tailed data, extreme order statistics, and non-iid data — for time series you need block bootstrap, for clusters you need cluster bootstrap.

## Q6: "What's wrong with running 20 independent t-tests at α=0.05?"

> The probability of at least one false positive grows to $1 - 0.95^{20} \approx 64\%$. If your 20 tests are truly under the null, you'll likely declare some "significant" purely by chance. The cure depends on the cost of false positives. If a single false positive is catastrophic (regulatory, medical), use **FWER control** — Bonferroni ($\alpha/m$) or Holm — which guarantees the chance of any false positive stays at α. If you expect many real effects and just want to control the proportion of false positives among rejections, use **FDR control** — Benjamini-Hochberg — which is less conservative. In A/B testing platforms, the standard pattern is Holm across the metric panel and pre-registration of primary metrics so no multiplicity correction is needed for the primary decision.

## Q7: "What's a p-value, exactly?"

> A p-value is the probability of observing a test statistic at least as extreme as the one we got, **assuming the null hypothesis is true**. It's a property of the data given a model, not a property of the truth. Critically, it is not P(H₀ | data) — that's a Bayesian posterior requiring a prior. It is not the probability the result will replicate. It is not effect size. With large enough n, you can get p < 0.001 for an effect of 0.001%, which is statistically clear but practically meaningless. So I always report p-values alongside effect sizes and confidence intervals, and I avoid hard p < 0.05 thresholds in favor of pre-registered decision rules. The ASA's 2016 statement makes this explicit.

## Q8: "Confidence interval vs credible interval — explain the difference precisely."

> A frequentist 95% confidence interval is a procedure: if we ran this experiment many times, 95% of the intervals we constructed would contain the true parameter. The interval is random; the parameter is fixed. We cannot say "there's a 95% probability the parameter is in this specific interval" — once computed, it either contains it or doesn't. A Bayesian 95% credible interval directly says "given my prior and the data, there's a 95% posterior probability the parameter is in this interval." The parameter is random (in the epistemic sense); the interval is fixed conditional on data. They're mathematically different objects. They often give similar numbers when the prior is weak and the data is rich, but the interpretation is fundamentally different. Interviewers love testing whether you mix them up.

## Q9: "Walk me through Hoeffding's inequality and why it matters in ML."

> Hoeffding's inequality says that for independent bounded random variables $X_i \in [a,b]$, the sample mean concentrates around the true mean exponentially fast: $P(|\bar{X}_n - \mu| \geq t) \leq 2\exp(-2nt^2/(b-a)^2)$. The key point is that this holds **for any distribution** with bounded support, no normality assumed. In machine learning, you apply it to risk: the empirical risk on a held-out set is close to the true risk with high probability. This is the foundation of **PAC learning bounds**. Used in: bandit algorithms (UCB), conformal prediction (distribution-free CIs), generalization bounds, and finite-sample analysis of online algorithms. Stronger variants like Bernstein use variance information to give sharper bounds.

## Q10: "Your A/B test has heavy-tailed revenue data. Is the t-test CI valid?"

> The t-test relies on CLT for the sample mean to be approximately normal. For heavy-tailed but finite-variance data, CLT eventually kicks in but **convergence is slow** — sample sizes of thousands may not be enough. The CI's coverage can be substantially off. Diagnostics: bootstrap the mean and compare its distribution to normal; compute skewness and kurtosis; plot a QQ plot of the mean across resamples. Fixes in order of pragmatism: **(1)** winsorize the top 0.1-1% of values to reduce tail influence; **(2)** log-transform (changes the estimand to log-mean, but often more interpretable for revenue); **(3)** use bootstrap (BCa) for CIs that don't rely on normality; **(4)** use trimmed means; **(5)** use rank-based tests like Mann-Whitney if effect interpretation is less critical. CUPED variance reduction also helps because reducing variance speeds up the CLT effectively. I'd report results with both standard t-test and bootstrap-BCa intervals and document any divergence.

---

# 15. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│       DEEP STATISTICAL REASONING — INTERVIEW TAKEAWAYS              │
│                                                                     │
│  1. MSE = bias² + variance. Always the fundamental trade-off.     │
│                                                                     │
│  2. Unbiased ≠ consistent. Know both definitions cold.             │
│                                                                     │
│  3. MLE is consistent, asymptotically normal, asymptotically       │
│     efficient under regularity. May be biased in finite samples.   │
│                                                                     │
│  4. Cramér-Rao gives the lower bound on variance. MLE achieves it │
│     asymptotically.                                                │
│                                                                     │
│  5. CLT is why so much of statistics works. But requires finite    │
│     variance.                                                       │
│                                                                     │
│  6. Delta method for variance of transformed estimators           │
│     (ratios, log-odds).                                            │
│                                                                     │
│  7. Bootstrap works because empirical CDF → true CDF (Glivenko-   │
│     Cantelli). Fails for extremes and heavy tails.                │
│                                                                     │
│  8. FWER controls "any" false positive; FDR controls "proportion" │
│     of false positives among rejections.                           │
│                                                                     │
│  9. P-values are NOT P(H₀|data), NOT effect size, NOT replication │
│     probability.                                                   │
│                                                                     │
│  10. Concentration inequalities (Hoeffding, Bernstein) give       │
│      finite-sample guarantees in ML.                               │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"Deep statistical reasoning is the ability to derive estimators, justify their properties, and defend their use when textbook recipes don't apply. The foundations — likelihood, asymptotics, information, concentration — let you reason from first principles when the data violates assumptions."*

---

**Further reading**

- Casella & Berger — *Statistical Inference* (the standard graduate text)
- Wasserman — *All of Statistics* (compact, modern, broad)
- Lehmann & Casella — *Theory of Point Estimation*
- van der Vaart — *Asymptotic Statistics*
- Boucheron, Lugosi, Massart — *Concentration Inequalities*
- Robert — *The Bayesian Choice*
- Efron & Hastie — *Computer Age Statistical Inference*
