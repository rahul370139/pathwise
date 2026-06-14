# Econometrics Deep Dive — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / Economist — Panel Data, Discrete Choice, Identification, Simultaneous Equations, Time-Series Econometrics

> **Context:** `33_causal_inference_and_experimentation.md` covers DiD, IV, RDD, and synthetic control at the level a Data Scientist uses them. This guide goes one layer deeper for roles that lean **economist** (Amazon Economist, Uber Marketplace Scientist, Airbnb Economist, ad-tech, fintech). Topics: panel dynamics, discrete choice models, simultaneous equations, identification strategy, weak instruments, robust SE, structural vs reduced-form estimation, time-series econometrics. This is what makes the difference between "uses DiD" and "designs identification strategies."

---

## Table of Contents

1. [What Econometrics Adds Over Statistics](#1-what-econometrics-adds-over-statistics)
2. [Identification Strategy — The Core Skill](#2-identification-strategy)
3. [OLS Refresher — Assumptions and Violations](#3-ols-refresher--assumptions-and-violations)
4. [Panel Data — Pooled, FE, RE](#4-panel-data--pooled-fe-re)
5. [Panel Dynamics & Lagged Dependents](#5-panel-dynamics--lagged-dependents)
6. [Instrumental Variables Deep Dive](#6-instrumental-variables-deep-dive)
7. [Simultaneous Equations Models](#7-simultaneous-equations-models)
8. [Discrete Choice Models — Logit, Probit, Multinomial](#8-discrete-choice-models)
9. [Selection Models — Heckman](#9-selection-models--heckman)
10. [Time-Series Econometrics Essentials](#10-time-series-econometrics-essentials)
11. [Structural vs Reduced-Form Estimation](#11-structural-vs-reduced-form-estimation)
12. [Robust Standard Errors and Clustering](#12-robust-standard-errors-and-clustering)
13. [Interview Questions with Strong Answers](#13-interview-questions-with-strong-answers)
14. [Key Takeaways](#14-key-takeaways)

---

# 1. What Econometrics Adds Over Statistics

| Statistics | Econometrics |
|-----------|--------------|
| Fits models, makes predictions | Estimates **causal parameters** under explicit assumptions |
| Treats data as draws from a distribution | Treats data as outcomes of an **economic process** |
| Focus: estimation, hypothesis testing | Focus: **identification** — can the parameter be recovered from the data? |
| Linear regression as a fitting tool | Linear regression as a **causal estimator** under exclusion + exogeneity |
| Cross-validation | Specification testing, robust SE, instruments |

> **Interview soundbite:** *"A statistician fits a model. An econometrician asks: what economic mechanism makes the parameter I estimate equal to the parameter I want?"*

---

# 2. Identification Strategy

## 2.1 Definition

**Identification** = whether the parameter of interest is recoverable, even with infinite data.

```
DATA + ASSUMPTIONS = PARAMETER

If data alone (no assumptions) → many parameters consistent → unidentified.
If data + assumptions → one parameter consistent → identified.
```

## 2.2 The Causal Identification Triplet

Every causal claim needs three things:

1. **Variation** — something moves the treatment (random, natural, or quasi-random).
2. **Exclusion** — that source of variation affects outcome **only** through treatment.
3. **Monotonicity / no defiers** — the assumption that allows you to interpret the average effect.

## 2.3 Identification Strategy Examples

| Strategy | Source of variation | Exclusion claim |
|----------|--------------------|-----------------| 
| **RCT** | Random assignment | Randomization affects outcome only via treatment |
| **DiD** | Policy timing | Pre-period parallel trends would have continued |
| **IV** | Instrument shifts treatment | Instrument doesn't affect outcome directly |
| **RDD** | Threshold-based assignment | Units just above/below threshold are otherwise comparable |
| **Natural experiment** | External shock (lottery, weather) | The shock doesn't directly affect Y |

## 2.4 Tests vs Stories

An identification strategy is fundamentally a **story** (assumption) you can sometimes partially test:

- DiD → test pre-period parallel trends
- IV → test first-stage F-stat, overidentification test (if multiple instruments)
- RDD → test density at threshold (McCrary), continuity of covariates
- RCT → SRM and balance checks

Tests provide **evidence consistent with**, not **proof of**, identification.

---

# 3. OLS Refresher — Assumptions and Violations

## 3.1 The Linear Model

$$
Y = X\beta + \varepsilon, \quad \mathbb{E}[\varepsilon \mid X] = 0
$$

## 3.2 Gauss-Markov Assumptions

| Assumption | Meaning | Violation symptom |
|-----------|---------|-------------------|
| **Linearity in parameters** | Y is linear function of $\beta$ | Wrong functional form |
| **Random sample** | iid (or strict exogeneity in time series) | Selection bias |
| **No perfect multicollinearity** | X has full column rank | Singular design matrix |
| **Zero conditional mean** | $\mathbb{E}[\varepsilon|X]=0$ | Omitted variable bias / endogeneity |
| **Homoskedasticity** | $\text{Var}(\varepsilon|X)=\sigma^2$ | Heteroskedasticity (use robust SE) |
| **No autocorrelation** | $\text{Cov}(\varepsilon_i, \varepsilon_j)=0$ | Time-series serial correlation |

> Under all six, OLS is **BLUE** (Best Linear Unbiased Estimator).

## 3.3 The Big Three Violations to Worry About

1. **Omitted variable bias (OVB)** — confounder correlated with both X and Y. Cure: include the variable, find an instrument, or use a design (DiD, RDD).
2. **Reverse causality** — Y also affects X. Cure: IV or natural experiment.
3. **Measurement error** — X measured with noise. Cure: IV or careful data work.

## 3.4 The OVB Formula

Suppose true model: $Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$, you omit $X_2$. Then:

$$
\hat{\beta}_1^{\text{OLS}} = \beta_1 + \beta_2 \cdot \frac{\text{Cov}(X_1, X_2)}{\text{Var}(X_1)}
$$

The bias direction is the **product of**:
- the omitted variable's effect on Y ($\beta_2$)
- its correlation with the included variable ($\text{Cov}(X_1, X_2)/\text{Var}(X_1)$)

> **Interview move:** when the interviewer asks "what's the bias?", apply this formula in your head.

---

# 4. Panel Data — Pooled, FE, RE

## 4.1 Setup

Panel = N units observed over T periods: $Y_{it}$, $X_{it}$ for unit i, period t.

## 4.2 Three Estimators

### Pooled OLS
Treat the panel as a giant cross-section, ignore the panel structure.

$$
Y_{it} = \alpha + X_{it}\beta + \varepsilon_{it}
$$

**Problem:** ignores unit-specific unobserved heterogeneity. Biased if $\alpha_i$ (unit fixed effect) is correlated with $X_{it}$.

### Fixed Effects (FE / Within)
Each unit has its own intercept absorbing time-invariant unobservables:

$$
Y_{it} = \alpha_i + X_{it}\beta + \varepsilon_{it}
$$

Estimation: subtract unit means (within transformation):

$$
(Y_{it} - \bar{Y}_i) = (X_{it} - \bar{X}_i)\beta + (\varepsilon_{it} - \bar{\varepsilon}_i)
$$

**Pros:** controls for ALL time-invariant confounders (observed and unobserved).
**Cons:** loses identifying variation; can't estimate effects of time-invariant variables.

### Random Effects (RE)
Treats $\alpha_i$ as a random variable independent of $X_{it}$.

$$
Y_{it} = \alpha + X_{it}\beta + u_i + \varepsilon_{it}, \quad u_i \perp X_{it}
$$

**Pros:** more efficient than FE if assumption holds.
**Cons:** biased if $u_i$ is correlated with $X_{it}$ (almost always the case in observational data).

## 4.3 Hausman Test

Tests RE vs FE: under the null, both are consistent and RE is efficient; under the alternative, only FE is consistent.

$$
H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' [\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})
$$

If $p < 0.05$, prefer FE.

## 4.4 Two-Way Fixed Effects (TWFE)

Add **time fixed effects** to absorb common shocks:

$$
Y_{it} = \alpha_i + \gamma_t + X_{it}\beta + \varepsilon_{it}
$$

This is the workhorse of empirical economics. Be aware of recent literature on **TWFE pitfalls with staggered adoption** (Goodman-Bacon, Callaway-Sant'Anna).

## 4.5 Code

```python
import statsmodels.formula.api as smf

# Fixed effects via dummies (or use linearmodels.PanelOLS)
fe_model = smf.ols(
    "Y ~ X1 + X2 + C(unit_id) + C(year)",
    data=panel_df
).fit(cov_type="cluster", cov_kwds={"groups": panel_df["unit_id"]})
```

---

# 5. Panel Dynamics & Lagged Dependents

## 5.1 The Dynamic Panel Model

$$
Y_{it} = \rho Y_{i,t-1} + X_{it}\beta + \alpha_i + \varepsilon_{it}
$$

$\rho$ captures persistence: e.g., if Y is consumer spending, $\rho \approx 0.7$ means 70% of last period's spending carries over.

## 5.2 The Nickell Bias

OLS with FE on dynamic panels is **biased** because $Y_{i,t-1}$ is mechanically correlated with the within-transformed error term. The bias scales as $O(1/T)$ — disappears with long panels, severe with short ones.

**Mitigations:**
- **Arellano-Bond (GMM)** — instruments $Y_{i,t-1}$ with deeper lags ($Y_{i,t-2}, Y_{i,t-3}, ...$).
- **System GMM (Blundell-Bond)** — uses both differences and levels as moment conditions.
- **Bias-corrected estimators** (Kiviet) for short panels.

## 5.3 Why It Matters

Many product / marketplace metrics are **persistent** — yesterday's GMV is the best predictor of today's GMV. Ignoring this autoregressive structure overstates the precision of any covariate's effect.

---

# 6. Instrumental Variables Deep Dive

## 6.1 Recap

Endogeneity: $\text{Cov}(X, \varepsilon) \neq 0$. Direct OLS is biased.

IV: find Z such that:
1. **Relevance**: $\text{Cov}(Z, X) \neq 0$.
2. **Exclusion**: $Z$ affects $Y$ only through $X$ — $\text{Cov}(Z, \varepsilon) = 0$.

## 6.2 Two-Stage Least Squares (2SLS)

$$
\hat{X} = \pi_0 + \pi_1 Z + \nu \quad \text{(first stage)}
$$

$$
Y = \beta_0 + \beta_1 \hat{X} + \eta \quad \text{(second stage)}
$$

$\hat{\beta}_1$ is the IV estimate.

## 6.3 Weak Instruments

If Z is weakly correlated with X, the first-stage F-stat is small and 2SLS is **biased toward OLS** and has fat-tailed inference.

**Rule of thumb:** First-stage F-stat > 10. Stock-Yogo critical values for stricter cutoffs.

```python
from linearmodels.iv import IV2SLS
model = IV2SLS.from_formula(
    "Y ~ 1 + W + [X ~ Z]", data=df
).fit()
print(model.first_stage)  # F-stat
```

## 6.4 LATE Interpretation

With heterogeneous effects, IV identifies the **Local Average Treatment Effect** on **compliers** — units whose treatment status was changed by the instrument:

$$
\text{LATE} = \mathbb{E}[Y(1) - Y(0) \mid \text{complier}]
$$

Not the ATE for the full population. Important caveat in interpretation.

## 6.5 Overidentification (Sargan/Hansen J)

If you have more instruments than endogenous variables, you can test exclusion via the J-test:

$$
H_0: \text{all instruments are exogenous}
$$

Reject → at least one instrument is invalid (you don't know which).

## 6.6 Common Instruments in the Wild

| Domain | Endogenous variable | Instrument |
|--------|--------------------|------------|
| Education economics | Years of schooling | Quarter of birth (compulsory schooling laws) |
| Health | Hospitalization | Distance to hospital |
| Online platforms | Treatment compliance | Random encouragement to use new feature |
| Marketplaces | Price | Cost shocks (gas, freight) |
| Ads | Ad exposure | Random ad inventory variation |

---

# 7. Simultaneous Equations Models

## 7.1 When Two Variables Move Together

Supply and demand:

$$
Q_d = \alpha_d - \beta_d P + \varepsilon_d \quad \text{(demand)}
$$
$$
Q_s = \alpha_s + \beta_s P + \varepsilon_s \quad \text{(supply)}
$$

At equilibrium, $Q_d = Q_s$ — both P and Q are endogenous. OLS of Q on P doesn't recover demand or supply elasticity; it recovers a mishmash.

## 7.2 Identification via Exclusion Restrictions

If something shifts only supply (e.g., a tax on producers), you can identify demand. If something shifts only demand (e.g., consumer income shocks), you can identify supply.

This is the foundation of **structural** demand and supply estimation.

## 7.3 Order and Rank Conditions

An equation is **identified** if:
- **Order condition**: number of excluded exogenous variables ≥ number of included endogenous variables minus one.
- **Rank condition**: the matrix of excluded exogenous variables has full rank.

(In practice you check by inspection of the structural model.)

## 7.4 Three-Stage Least Squares (3SLS)

For a system of simultaneous equations, 3SLS uses cross-equation information to improve efficiency over equation-by-equation 2SLS.

---

# 8. Discrete Choice Models

## 8.1 Why You Can't Just Use Linear Probability

For binary Y ∈ {0, 1}:

$$
Y = X\beta + \varepsilon
$$

The **linear probability model (LPM)** gives unbounded predictions and heteroskedastic errors. It's still used for ATE interpretation when effects are small, but classical discrete choice uses **logit** or **probit**.

## 8.2 Logit / Logistic Regression

$$
P(Y=1 \mid X) = \frac{\exp(X\beta)}{1 + \exp(X\beta)}
$$

Estimation: MLE. The **log-odds** are linear in X:

$$
\log \frac{P(Y=1)}{P(Y=0)} = X\beta
$$

Coefficient $\beta_k$: a one-unit increase in $X_k$ increases the log-odds by $\beta_k$.

## 8.3 Probit

Same idea but with normal CDF:

$$
P(Y=1 \mid X) = \Phi(X\beta)
$$

In practice, logit and probit give very similar marginal effects.

## 8.4 Marginal Effects

For interpretability, report **average marginal effects (AME)** — the average derivative of $P(Y=1)$ w.r.t. $X_k$ across the sample.

```python
from statsmodels.discrete.discrete_model import Logit
model = Logit(y, X).fit()
ame = model.get_margeff(at="overall")
print(ame.summary())
```

## 8.5 Multinomial Logit

For categorical outcomes with K > 2 alternatives (e.g., choice of mode of transport):

$$
P(Y=j \mid X) = \frac{\exp(X\beta_j)}{\sum_{k=1}^K \exp(X\beta_k)}
$$

The **IIA assumption** (Independence of Irrelevant Alternatives) is restrictive — relative odds of choosing j vs k should not depend on availability of other alternatives.

**Generalizations:** nested logit (allows correlation within groups), mixed logit (random coefficients).

## 8.6 Discrete Choice in Industry

| Domain | Use case |
|--------|---------|
| Marketplaces | Modeling product/listing choice given features |
| Ad-tech | Click probability prediction (binary logit) |
| Pricing | Willingness-to-pay estimation |
| Transportation | Mode choice (car/bus/bike/walk) |
| Demand estimation | BLP-style models for differentiated products |

---

# 9. Selection Models — Heckman

## 9.1 The Problem

You want to estimate $\mathbb{E}[Y \mid X]$, but Y is only observed for a non-random subset (e.g., wages only observed for people who work).

## 9.2 The Heckman Two-Step

**Step 1: Selection equation (probit).** Model the probability of observing Y:

$$
P(\text{observed} = 1 \mid Z) = \Phi(Z\gamma)
$$

Compute the **inverse Mills ratio** $\lambda = \phi(Z\gamma) / \Phi(Z\gamma)$.

**Step 2: Outcome equation.** Estimate:

$$
Y = X\beta + \rho \sigma \lambda + \eta
$$

The coefficient on $\lambda$ tests for selection bias.

## 9.3 Identification

Heckman requires an **exclusion restriction**: at least one variable in Z that does NOT appear in X. Without it, identification is by functional form (fragile).

## 9.4 When to Use

- Wage equations (employment selection)
- Marketing response models (only observe Y for treated users)
- Survey non-response correction

---

# 10. Time-Series Econometrics Essentials

## 10.1 Stationarity

A series is **weakly stationary** if mean, variance, and autocovariance don't depend on time.

**Test:** Augmented Dickey-Fuller (ADF). Null hypothesis: unit root (non-stationary).

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series)
print(f"ADF stat: {result[0]:.3f}, p: {result[1]:.3f}")
```

## 10.2 Differencing

If non-stationary, take first differences: $\Delta Y_t = Y_t - Y_{t-1}$. Most macro series need first differences; some need second.

## 10.3 Spurious Regression

Regressing one non-stationary series on another gives **fake R²** and significant t-stats — even with no relationship. The cure: ensure both series are stationary OR they are cointegrated.

## 10.4 Cointegration

Two non-stationary series $X_t, Y_t$ are **cointegrated** if a linear combination is stationary:

$$
Y_t - \beta X_t = u_t \text{ where } u_t \text{ is stationary}
$$

**Engle-Granger** and **Johansen** tests check for cointegration. If cointegrated, run an **Error Correction Model (ECM)**.

## 10.5 Granger Causality

X **Granger-causes** Y if past values of X help predict Y, beyond past values of Y alone.

$$
Y_t = \alpha + \sum_{i=1}^p \beta_i Y_{t-i} + \sum_{j=1}^q \gamma_j X_{t-j} + \varepsilon_t
$$

Test $H_0: \gamma_1 = \gamma_2 = \dots = 0$ via F-test.

> **Caveat:** Granger causality is **predictive precedence**, NOT causation in the structural sense.

## 10.6 ARIMA, VAR

| Model | Use |
|-------|-----|
| **AR(p)** | Y depends on its own lags |
| **MA(q)** | Y depends on past errors |
| **ARMA(p,q)** | Both |
| **ARIMA(p,d,q)** | After d differences |
| **VAR(p)** | Vector autoregression — multiple series, each depends on lags of all |
| **VECM** | VAR with cointegration |

> See `34_time_series_and_forecasting.md` for forecasting-focused treatment.

---

# 11. Structural vs Reduced-Form Estimation

## 11.1 The Distinction

| Reduced form | Structural |
|-------------|-----------|
| Estimate a parameter (e.g., ATE) without modeling the mechanism | Model the economic process generating the data |
| Direct inference from data | Counterfactual policy simulation |
| Identifies "what is the effect of X on Y" | Identifies "what would happen if we changed the parameter $\theta$" |
| Cheaper, more transparent | More demanding assumptions |
| Examples: DiD, IV, RDD | Examples: BLP, dynamic discrete choice, structural search models |

## 11.2 When to Choose Which

| Question | Approach |
|----------|---------|
| "What's the effect of this ad campaign on sales?" | Reduced form |
| "If we raised our commission rate by 1%, how would sellers respond and what's the new equilibrium price?" | Structural |
| "Does X cause Y?" | Reduced form |
| "What's the welfare gain from a new feature?" | Structural |

## 11.3 The Hybrid Approach

Many modern papers do **both**:
1. Reduced-form descriptive evidence (DiD, event study).
2. Structural model calibrated to the reduced-form moments.
3. Counterfactual simulations from the structural model.

This gives "the answer is X, and here's the mechanism."

---

# 12. Robust Standard Errors and Clustering

## 12.1 The Sandwich Estimator

White (heteroskedasticity-robust) SE:

$$
\hat{V}_{\text{robust}} = (X'X)^{-1} \left( \sum_i e_i^2 x_i x_i' \right) (X'X)^{-1}
$$

Cluster-robust (Liang-Zeger):

$$
\hat{V}_{\text{cluster}} = (X'X)^{-1} \left( \sum_{g} X_g' e_g e_g' X_g \right) (X'X)^{-1}
$$

## 12.2 When to Cluster

| Setting | Cluster on |
|---------|-----------|
| Panel | Unit (individual, firm, geo) |
| Repeated cross-section in geos | Geo |
| Households surveyed | Household |
| Treatment assigned at school level | School |

## 12.3 Two-Way Clustering

If errors are correlated along **two** dimensions (e.g., firm and year), use two-way clustering:

$$
\hat{V}_{\text{2-way}} = \hat{V}_{\text{firm}} + \hat{V}_{\text{year}} - \hat{V}_{\text{firm} \cap \text{year}}
$$

## 12.4 Bootstrap Alternatives

When G is small or model is complex:
- **Block bootstrap** for time series
- **Wild cluster bootstrap** for few clusters
- **Pairs bootstrap** for general

---

# 13. Interview Questions with Strong Answers

## Q1: "What is identification and why do econometricians obsess over it?"

> Identification is whether the parameter you want is recoverable from the data, even with infinite samples. A model is identified if data + assumptions imply a unique value of the parameter. Econometricians obsess over it because economic data is almost always observational — there's selection, simultaneity, omitted variables — and the parameter you care about (a causal effect, an elasticity) is rarely the parameter OLS recovers. The identification strategy is the explicit chain of assumptions that maps your data to your causal claim. Without it, you have correlations, not causal estimates. Every empirical paper opens by stating the identification strategy because that's the heart of the contribution.

## Q2: "Why use fixed effects instead of random effects?"

> Fixed effects allow the unit-specific intercept to be **arbitrarily correlated** with the regressors. This is what controls for time-invariant unobserved confounders — the whole point of panel data. Random effects assumes the unit intercept is **uncorrelated** with regressors, which is a strong assumption that's almost always violated in observational data. The Hausman test formalizes this: under the null, both are consistent and RE is efficient; under the alternative, only FE is consistent. In practice, for almost all microeconometric applications, FE is the safer default. RE is preferred only when you have strong reasons to believe units are exchangeable — e.g., random subjects in a clinical trial.

## Q3: "Explain instrumental variables and the LATE interpretation."

> Instrumental variables solve endogeneity by finding a source of variation Z that affects the outcome only through the endogenous variable X. The estimator (2SLS) projects X onto Z, then regresses Y on the projection. Z must satisfy relevance (Cov(Z, X) ≠ 0) and exclusion (Z affects Y only through X). With **heterogeneous treatment effects**, IV doesn't identify the average treatment effect on the full population — it identifies the **Local Average Treatment Effect (LATE)** on **compliers**, the units whose X was changed by Z. This is a critical caveat: if compliers differ from the population in their potential outcomes, LATE is not what policymakers care about. The classic example is quarter-of-birth as an instrument for schooling — LATE estimates the return to schooling for kids whose schooling was affected by compulsory laws, not the average return for everyone.

## Q4: "What is a weak instrument and how do you detect it?"

> A weak instrument is one that's only weakly correlated with the endogenous variable. The 2SLS estimator is biased toward OLS and has fat-tailed sampling distributions, so confidence intervals are wrong. The standard diagnostic is the **first-stage F-statistic** — rule of thumb F > 10 for a single instrument. Stock-Yogo provides stricter critical values for multiple instruments. If the F-stat is below threshold, you can use weak-instrument-robust inference like Anderson-Rubin confidence sets, but the better fix is to find a stronger instrument or accept that the parameter isn't well-identified.

## Q5: "Explain Simpson's paradox in a regression context."

> Simpson's paradox is when an effect that's positive in aggregate is negative within every subgroup, or vice versa. In regression terms, it's the classic omitted variable bias case where the omitted variable is correlated with the included regressor and the outcome in opposite directions. The fix is to include the omitted variable as a covariate. If the variable is unobserved, you need a design — fixed effects to absorb time-invariant heterogeneity, or an instrument. Whenever I see an effect that contradicts intuition, my first check is to slice by an obvious confounder; if the sign flips, the aggregate result is being driven by composition shifts, not the causal effect.

## Q6: "Walk me through a difference-in-differences analysis."

> DiD compares the **change** in outcome between treated and control groups before and after treatment, identifying ATT under parallel trends. The estimator:

$$
\hat{\tau}_{\text{DiD}} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}})
$$

> The key assumption is **parallel trends**: in the absence of treatment, treated and control would have moved together. You can't test this in the post period (impossible), but you can build evidence with pre-period trends — plot both groups for several pre-treatment periods and visually verify they move in parallel. The regression form is $Y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \varepsilon_{it}$ — unit and time fixed effects, treatment indicator. Cluster SE at the unit level. Recent literature (Goodman-Bacon, Callaway-Sant'Anna) shows that TWFE with **staggered adoption** can mis-estimate when treatment timing varies; in those cases, use event-study or stacked DiD designs.

## Q7: "Logit vs probit vs LPM — when do you use each?"

> All three model binary outcomes. **LPM (linear probability model)** is OLS on a 0/1 outcome. Pros: easy interpretation (coefficient = change in P(Y=1)), fixed effects work naturally, no convergence issues. Cons: predicts probabilities outside [0,1], errors are heteroskedastic. **Logit** uses the logistic CDF: bounded predictions, log-odds linear in X, MLE estimation. **Probit** uses the normal CDF: very similar to logit in practice, slightly tighter tails. I use **LPM** when the focus is on ATE and effects are small (probabilities stay in interior) — gives clean interpretation. I use **logit** for prediction tasks and when probabilities span [0,1] heavily. I report **average marginal effects** for logit/probit to make coefficients interpretable. Probit is mostly historical convention in some literatures.

## Q8: "What's the difference between Granger causality and actual causality?"

> Granger causality is purely predictive — X Granger-causes Y if past X helps predict Y beyond past Y alone. It's a statement about information flow in a time series, not about a structural mechanism. **Real** causality requires intervention semantics — what would Y be if we set X to a specific value? Granger causality can be present with no structural cause (e.g., both X and Y respond to a common shock, but X reacts faster). And structural causality can exist without Granger causality (e.g., X causes Y instantaneously and contemporaneously, but past X doesn't help predict). So Granger causality is a useful diagnostic for time-series modeling but doesn't license causal claims about policy interventions.

## Q9: "How do you handle endogeneity in a panel where lagged Y is a regressor?"

> The Nickell bias problem. Within-transformed (FE) OLS on $Y_{it} = \rho Y_{i,t-1} + X_{it}\beta + \alpha_i + \varepsilon_{it}$ is biased because subtracting unit means makes $Y_{i,t-1}$ mechanically correlated with the transformed error. The bias is order 1/T, so it's serious for short panels. The standard fix is **Arellano-Bond GMM**: difference the equation to eliminate fixed effects, then use deeper lags of Y ($Y_{i,t-2}, Y_{i,t-3}, ...$) as instruments for $\Delta Y_{i,t-1}$. **System GMM (Blundell-Bond)** adds level moment conditions for efficiency. The diagnostics to check: AR(2) test in residuals (should fail to reject), Hansen J-test for instrument validity, and instrument count (too many → overfitted). In practice for long panels (T > 30) the bias is small and FE-OLS is fine.

## Q10: "What is the difference between reduced-form and structural estimation?"

> **Reduced form** estimates a causal parameter (an ATE, an elasticity, a coefficient) directly from data using a design (DiD, IV, RDD, experiment). It's transparent and minimally model-dependent. Limitation: it answers only the question you ran the design for, and can't simulate counterfactual policies that change the underlying parameters of the system. **Structural** estimation specifies the full economic model — utility functions, profit functions, equilibrium conditions — and estimates its parameters. This lets you simulate any policy counterfactual: "what if the commission rate changed by 2%?" "what if the search ranker were perfectly accurate?" The cost is more assumptions (functional form, equilibrium, distributional). At top tech firms with marketplace teams, you'll see both — reduced-form evidence for the headline result, structural simulation for the policy decision. I'd choose reduced form when the question is "did this work?" and structural when it's "what should we do?".

---

# 14. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│           ECONOMETRICS — INTERVIEW TAKEAWAYS                        │
│                                                                     │
│  1. Identification is the heart of econometrics. Every causal     │
│     claim is a story about variation + exclusion.                  │
│                                                                     │
│  2. OVB formula: \(\hat\beta_1 = \beta_1 + \beta_2 \cdot          │
│     Cov(X_1,X_2)/Var(X_1)\). Use it to predict bias direction.    │
│                                                                     │
│  3. Panel FE absorbs all time-invariant confounding. Default     │
│     over RE unless exchangeability is plausible.                  │
│                                                                     │
│  4. Dynamic panels: Nickell bias. Use Arellano-Bond / system GMM. │
│                                                                     │
│  5. IV: relevance + exclusion. F-stat > 10. LATE on compliers.   │
│                                                                     │
│  6. Discrete choice: logit/probit MLE; report average marginal   │
│     effects. LPM acceptable when effects small.                    │
│                                                                     │
│  7. Selection: Heckman two-step requires exclusion restriction.   │
│                                                                     │
│  8. Time series: check stationarity, beware spurious regression,  │
│     test for cointegration.                                        │
│                                                                     │
│  9. Granger causality ≠ structural causality.                     │
│                                                                     │
│  10. Reduced form for "did it work"; structural for "what should  │
│      we do".                                                       │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"Econometrics is the science of recovering causal parameters from observational data. The toolkit — IV, panel FE, DiD, discrete choice, structural models — is built around one question: under what assumption does the data identify the parameter I want?"*

---

**Further reading**

- Wooldridge — *Econometric Analysis of Cross Section and Panel Data*
- Angrist & Pischke — *Mostly Harmless Econometrics*
- Cameron & Trivedi — *Microeconometrics: Methods and Applications*
- Stock & Watson — *Introduction to Econometrics*
- Hayashi — *Econometrics*
- Hansen — *Econometrics* (online, Bruce Hansen at UW-Madison)
