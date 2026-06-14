# Causal Inference & Advanced Experimentation — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Causal Reasoning, Observational Studies, Treatment Effect Estimation

> **Context:** Your file `24_statistics_and_ab_testing.md` covers randomized experiments (A/B tests), hypothesis testing, and Bayesian methods well. This guide covers what happens when **you CAN'T randomize** — the observational causal inference toolkit that Meta, Netflix, Uber, DoorDash, and Airbnb ask about heavily.

---

# Table of Contents

1. [Why Causal Inference Matters](#1-why-causal-inference-matters)
2. [The Causal Framework](#2-the-causal-framework)
3. [Difference-in-Differences (DiD)](#3-difference-in-differences-did)
4. [Propensity Score Methods](#4-propensity-score-methods)
5. [Instrumental Variables (IV)](#5-instrumental-variables-iv)
6. [Regression Discontinuity Design (RDD)](#6-regression-discontinuity-design-rdd)
7. [Synthetic Control Method](#7-synthetic-control-method)
8. [Uplift Modeling & Heterogeneous Treatment Effects](#8-uplift-modeling--heterogeneous-treatment-effects)
9. [Causal Discovery and DAGs](#9-causal-discovery-and-dags)
10. [When to Use Which Method](#10-when-to-use-which-method)
11. [Interview Questions with Strong Answers](#11-interview-questions-with-strong-answers)

---

# **1. Why Causal Inference Matters**

## **1.1 The Core Problem**

A/B tests (RCTs) are the gold standard for causal claims, but in practice **you often can't randomize**:

| Scenario | Why You Can't A/B Test | Need Causal Inference |
|----------|----------------------|----------------------|
| Effect of a new product feature launched to everyone | Already rolled out, no control group | DiD, Synthetic Control |
| Effect of a marketing campaign on revenue | Can't ethically withhold marketing from some | Propensity Score Matching |
| Effect of education on earnings | Can't randomly assign education levels | Instrumental Variables |
| Effect of a policy change (e.g., price increase) | Applied to everyone above a threshold | Regression Discontinuity |
| Personalized treatment assignment | Want to know WHO benefits most | Uplift Modeling |

## **1.2 Correlation ≠ Causation — The Three Threats**

```
┌──────────────────────────────────────────────────────────────────┐
│          WHY CORRELATION ≠ CAUSATION                              │
│                                                                   │
│  1. CONFOUNDING                                                  │
│     X ← C → Y  (C causes both X and Y)                          │
│     Example: Ice cream sales (X) and drowning (Y) both           │
│     increase in summer (C = temperature)                         │
│                                                                   │
│  2. REVERSE CAUSATION                                            │
│     X ← Y  (you think X→Y, but Y→X)                             │
│     Example: Hospitals have high death rates, but hospitals      │
│     don't cause death — sick people go to hospitals              │
│                                                                   │
│  3. SELECTION BIAS                                               │
│     Only observe a biased subset of the population               │
│     Example: Studying effect of exercise on health, but only    │
│     healthy people exercise → survivorship bias                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

# **2. The Causal Framework**

## **2.1 Potential Outcomes Framework (Rubin Causal Model)**

For each unit i, there are two **potential outcomes**:
- Y_i(1) = outcome if treated
- Y_i(0) = outcome if not treated

**Individual Treatment Effect (ITE):** τ_i = Y_i(1) - Y_i(0)

**The Fundamental Problem of Causal Inference:** We can only observe ONE of these for each unit. You either got the treatment or you didn't. The unobserved outcome is the **counterfactual**.

**Average Treatment Effect (ATE):** τ = E[Y(1) - Y(0)]

**Average Treatment Effect on the Treated (ATT):** τ_ATT = E[Y(1) - Y(0) | T=1]

## **2.2 Assumptions for Causal Identification**

| Assumption | Meaning | Violated When |
|-----------|---------|---------------|
| **Unconfoundedness (Ignorability)** | Treatment assignment is independent of potential outcomes, given covariates | There are unobserved confounders |
| **SUTVA** | Unit's outcome depends only on its own treatment, not others' | Network effects, spillovers |
| **Overlap (Positivity)** | Every unit has nonzero probability of being treated | Some covariate combinations always/never get treated |
| **Consistency** | Treatment is well-defined | Multiple versions of treatment exist |

## **2.3 Pearl's Structural Causal Model (SCM)**

Uses **Directed Acyclic Graphs (DAGs)** to represent causal relationships:

```
   Education
      │
      ▼
   Income ──────► Health
      ▲              ▲
      │              │
   Family ───────────┘
   Background

Reading: Education affects Income. Income affects Health.
Family Background affects both Income and Health (confounder).
To estimate Education → Income, must control for Family Background.
```

**Key DAG concepts:**
- **Confounder:** Common cause of treatment and outcome — must control for it
- **Mediator:** On the causal path from treatment to outcome — do NOT control for it (if you want total effect)
- **Collider:** Common effect of two variables — do NOT control for it (opens a biased path)

---

# **3. Difference-in-Differences (DiD)**

## **3.1 Core Idea**

Compare the **change over time** in the treatment group vs the **change over time** in the control group. The "difference of differences" removes time-invariant confounders.

```
┌──────────────────────────────────────────────────────────────────┐
│                   DIFFERENCE-IN-DIFFERENCES                       │
│                                                                   │
│  Outcome                                                         │
│   │                                                              │
│   │                    ╱ Treatment (actual)                      │
│   │                   ╱                                          │
│   │            ●─────╱         ← Treatment Effect (DiD estimate)│
│   │           ╱     ╱ · · · · ·                                  │
│   │          ╱     ╱         Counterfactual (if no treatment)   │
│   │         ╱     ╱                                              │
│   │   ●────╱─────╱── Control group                              │
│   │       ╱                                                      │
│   │      ╱                                                       │
│   └──────┼────────────┼───────────────────► Time                │
│        Before      After                                         │
│                  Treatment                                        │
│                                                                   │
│  DiD = (Y_treat_after - Y_treat_before)                         │
│      - (Y_control_after - Y_control_before)                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **3.2 The Parallel Trends Assumption**

DiD requires that **in the absence of treatment**, the treatment and control groups would have followed parallel trends. This is THE key assumption — it's untestable (counterfactual) but you can check pre-treatment trends for evidence.

**Checking parallel trends:**
- Plot pre-treatment trends for both groups — should be roughly parallel
- Run a placebo test: apply DiD to pre-treatment periods only (should find no effect)
- Test for statistical difference in pre-treatment slopes

## **3.3 DiD Formula**

```
Y_it = β₀ + β₁·Treatment_i + β₂·Post_t + β₃·(Treatment_i × Post_t) + ε_it

β₃ = the DiD estimate = the causal effect of the treatment
```

```python
import statsmodels.formula.api as smf

model = smf.ols("revenue ~ treatment + post + treatment:post", data=df).fit()
# coefficient on treatment:post is the DiD estimate
```

## **3.4 When to Use DiD**

- **A policy/feature was launched at a specific time** to a specific group
- You have **before and after data** for both treated and untreated groups
- **Parallel trends** assumption is plausible
- Examples: Feature rollout to one country first, minimum wage increase in one state, new recommendation algorithm for premium users

## **3.5 Variants**

| Variant | What It Handles |
|---------|----------------|
| **Staggered DiD** | Treatment applied at different times to different groups |
| **Event study** | Show effect dynamics before and after treatment (leads and lags) |
| **Triple difference (DDD)** | Add a third differencing dimension for robustness |
| **Two-way fixed effects** | Panel data with unit and time fixed effects |

---

# **4. Propensity Score Methods**

## **4.1 Core Idea**

When treatment is not random, treated and control groups differ systematically. Propensity score methods **balance** the groups on observed characteristics so you can make apples-to-apples comparisons.

**Propensity Score:** e(X) = P(Treatment = 1 | X) — the probability of receiving treatment given observed characteristics.

## **4.2 How It Works**

```
┌──────────────────────────────────────────────────────────────────┐
│             PROPENSITY SCORE MATCHING                              │
│                                                                   │
│  Step 1: Estimate propensity scores                              │
│          Logistic regression: P(T=1 | X₁, X₂, ..., Xₚ)         │
│                                                                   │
│  Step 2: Match treated to control units with similar scores      │
│          Nearest neighbor, caliper, kernel                       │
│                                                                   │
│  Before Matching:          After Matching:                       │
│  Treated: mostly young,    Treated: diverse mix                  │
│           high income       Control: similar mix ← balanced     │
│  Control: mostly old,                                            │
│           low income                                             │
│                                                                   │
│  Step 3: Estimate treatment effect on matched sample             │
│          ATT = mean(Y_treated) - mean(Y_matched_control)         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **4.3 Matching Methods**

| Method | How | Trade-off |
|--------|-----|-----------|
| **Nearest Neighbor** | Match each treated unit to the closest control on propensity score | Simple, but discards many controls |
| **Caliper Matching** | Only match within a distance threshold | Better balance, but may lose treated units |
| **Kernel Matching** | Weighted average of all controls, weighted by distance | Uses all data, no discarding |
| **Stratification** | Divide into strata by propensity score, compare within strata | Easy to implement, check balance per stratum |

## **4.4 Inverse Propensity Weighting (IPW)**

Instead of matching, **reweight** observations so the treated and control groups look similar:

```
Weight for treated: w = 1/e(X)
Weight for control: w = 1/(1 - e(X))

ATE = E[Y·T/e(X)] - E[Y·(1-T)/(1-e(X))]
```

**Advantage over matching:** Uses all data (no discarding), can be combined with regression (doubly robust estimation).

## **4.5 Doubly Robust Estimation**

Combines regression adjustment AND propensity score weighting. The estimate is consistent if **either** the regression model OR the propensity model is correctly specified — you only need one to be right.

## **4.6 Key Limitation**

Propensity score methods only control for **observed** confounders. If there are unobserved confounders affecting both treatment and outcome, the estimate is biased. This is the fundamental difference from randomized experiments.

---

# **5. Instrumental Variables (IV)**

## **5.1 Core Idea**

When there are **unobserved confounders**, find an **instrument** — a variable that affects the treatment but NOT the outcome (except through the treatment).

```
┌──────────────────────────────────────────────────────────────────┐
│                INSTRUMENTAL VARIABLES                              │
│                                                                   │
│  Instrument (Z) ──────► Treatment (T) ──────► Outcome (Y)       │
│                              ▲                     ▲             │
│                              │                     │             │
│                              └── Unobserved ───────┘             │
│                                  Confounder (U)                  │
│                                                                   │
│  Z affects Y ONLY through T (exclusion restriction)              │
│  Z is correlated with T (relevance)                              │
│  Z is independent of U (exogeneity)                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **5.2 Classic Examples**

| Question | Instrument | Rationale |
|----------|-----------|-----------|
| Effect of education on earnings | Draft lottery (Vietnam era) | Lottery affects military service (→education), random so unrelated to ability |
| Effect of class size on test scores | Birth date (Angrist & Lavy) | Maimonides' rule: class splits at threshold based on enrollment count |
| Effect of smoking on health | Cigarette taxes | Taxes affect price → smoking, but don't directly affect health |

## **5.3 Two-Stage Least Squares (2SLS)**

**Stage 1:** Regress Treatment on Instrument (and covariates)
```
T̂ᵢ = α₀ + α₁·Zᵢ + covariates
```

**Stage 2:** Regress Outcome on Predicted Treatment
```
Yᵢ = β₀ + β₁·T̂ᵢ + covariates
```

β₁ = the causal effect (Local Average Treatment Effect for compliers)

## **5.4 Instrument Validity Checks**

| Condition | Test | What to Look For |
|-----------|------|-----------------|
| **Relevance** | First-stage F-statistic | F > 10 (rule of thumb); weak instruments bias toward OLS |
| **Exclusion restriction** | Untestable (argue theoretically) | Z affects Y only through T |
| **Exogeneity** | Untestable (argue theoretically) | Z is as-good-as-random |
| **Over-identification** | Sargan/Hansen test (with 2+ instruments) | All instruments are valid |

---

# **6. Regression Discontinuity Design (RDD)**

## **6.1 Core Idea**

When treatment is assigned based on a **threshold** of a running variable, units just above and just below the threshold are nearly identical — creating a "local randomized experiment" at the cutoff.

```
┌──────────────────────────────────────────────────────────────────┐
│            REGRESSION DISCONTINUITY                                │
│                                                                   │
│  Outcome                                                         │
│   │           ●                                                  │
│   │          ● ●     ← Treatment Effect (jump at cutoff)        │
│   │         ●   ───────────● ●                                  │
│   │        ●    │           ●  ●                                │
│   │   ●   ●     │              ●                                │
│   │  ● ●●       │                                               │
│   │ ●            │                                               │
│   │              │                                               │
│   └──────────────┼────────────────────────► Running Variable    │
│               Cutoff                                             │
│               (threshold)                                        │
│                                                                   │
│  Units just below cutoff ≈ units just above (local randomness)  │
│  Causal effect = size of the jump                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **6.2 Classic Examples**

| Scenario | Running Variable | Cutoff | Treatment |
|----------|-----------------|--------|-----------|
| Effect of scholarship on GPA | Test score | Minimum score for scholarship | Getting scholarship |
| Effect of drinking age on mortality | Age | 21 (US) | Legal drinking |
| Effect of winning election on policy | Vote share | 50% | Being in office |
| Effect of ad exposure on purchase | Bid price | Auction threshold | Ad shown |

## **6.3 Sharp vs Fuzzy RDD**

| Type | What Happens at Cutoff | Estimation |
|------|----------------------|------------|
| **Sharp** | ALL units above cutoff get treatment, NONE below do | Simple regression at discontinuity |
| **Fuzzy** | Probability of treatment jumps but isn't 100% at cutoff | IV approach (cutoff as instrument for treatment) |

## **6.4 Key Considerations**

- **Bandwidth selection:** How close to the cutoff do you look? Too wide → bias. Too narrow → high variance.
- **Manipulation check:** Can units manipulate the running variable to be just above cutoff? (McCrary density test)
- **Local validity:** RDD only estimates the treatment effect **at the cutoff** — doesn't generalize to the full population.

---

# **7. Synthetic Control Method**

## **7.1 Core Idea**

When a treatment applies to a **single unit** (a country, city, or store), construct a **synthetic counterfactual** by finding a weighted combination of untreated units that closely matches the treated unit's pre-treatment trajectory.

```
┌──────────────────────────────────────────────────────────────────┐
│              SYNTHETIC CONTROL                                     │
│                                                                   │
│  Outcome                                                         │
│   │                                                              │
│   │               Treated unit (actual)                          │
│   │              ╱                                               │
│   │        ●────╱──────── Gap = Treatment Effect                │
│   │       ╱    ╱                                                 │
│   │      ╱    ╱  ← Synthetic control (weighted donors)          │
│   │     ╱    ╱    = 0.3×Unit_A + 0.5×Unit_B + 0.2×Unit_C       │
│   │    ╱   ╱                                                     │
│   │   ╱  ╱                                                       │
│   │  ╱ ╱   ← Pre-treatment: synthetic matches treated closely  │
│   │ ╱╱                                                           │
│   └──┼──────────────────────────────────────► Time              │
│    Treatment                                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **7.2 When to Use**

- Treatment applied to **one or very few units** (can't do DiD with one treated unit)
- Long pre-treatment period available
- Good "donor pool" of untreated units
- Classic example: Abadie et al. — effect of terrorism on Basque Country GDP

## **7.3 Steps**

1. Select donor pool (untreated units similar to treated)
2. Find weights that minimize pre-treatment difference between treated and synthetic control
3. Post-treatment gap between treated and synthetic = estimated effect
4. Placebo tests: apply the method to each donor unit — if many show similar gaps, your result isn't robust

---

# **8. Uplift Modeling & Heterogeneous Treatment Effects**

## **8.1 Core Idea**

Standard causal inference estimates the **Average Treatment Effect**. Uplift modeling asks: **who benefits most from treatment?** This is critical for personalization.

```
┌──────────────────────────────────────────────────────────────────┐
│                UPLIFT MODELING                                      │
│                                                                   │
│  Customer Segments by Treatment Response:                        │
│                                                                   │
│  ┌──────────────┬───────────────────────────────────────┐       │
│  │              │     Response WITHOUT treatment         │       │
│  │              │     No Purchase    │   Purchase        │       │
│  ├──────────────┼───────────────────┼───────────────────┤       │
│  │ Response     │   PERSUADABLES   │   SURE THINGS     │       │
│  │ WITH         │   (Target these!)│   (Would buy       │       │
│  │ treatment:   │                  │    anyway)         │       │
│  │ Purchase     │                  │                    │       │
│  ├──────────────┼───────────────────┼───────────────────┤       │
│  │ Response     │   LOST CAUSES   │   DO NOT DISTURB   │       │
│  │ WITH         │   (Won't buy     │   (Treatment       │       │
│  │ treatment:   │    regardless)   │    backfires!)     │       │
│  │ No Purchase  │                  │                    │       │
│  └──────────────┴───────────────────┴───────────────────┘       │
│                                                                   │
│  Uplift = P(Purchase|Treatment) - P(Purchase|No Treatment)       │
│  Focus marketing spend on PERSUADABLES only                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **8.2 Meta-Learners for Heterogeneous Treatment Effects**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **S-Learner** | Single model on (X, T) → Y; vary T to get effect | Simple | Can ignore treatment if effect is small |
| **T-Learner** | Two separate models: Y(T=1) and Y(T=0); difference = uplift | Easy to implement | No information sharing between models |
| **X-Learner** | Two-stage: train T- and C-models, impute counterfactuals, combine | Better with imbalanced treatment groups | More complex |
| **DR-Learner** | Doubly robust: combines outcome regression and propensity scores | Robust to misspecification | Most complex |
| **Causal Forest** | Random forest adapted for treatment effect estimation | Nonparametric, handles interactions | Computationally heavy |

## **8.3 Industry Applications**

| Company | Uplift Use Case |
|---------|----------------|
| **Netflix** | Who should get a promotional offer to reduce churn? |
| **Uber** | Which riders respond to surge pricing discounts? |
| **Marketing** | Target only persuadables to maximize ROI |
| **Healthcare** | Which patients benefit from an intervention? |
| **Lending** | Who benefits from a credit line increase vs who defaults? |

---

# **9. Causal Discovery and DAGs**

## **9.1 Building Causal DAGs**

| Step | What To Do |
|------|-----------|
| 1. **List variables** | Treatment, outcome, all potential confounders |
| 2. **Draw hypothesized arrows** | Based on domain knowledge (not data) |
| 3. **Identify confounders** | Common causes of treatment and outcome |
| 4. **Identify mediators** | On the causal pathway (don't control if you want total effect) |
| 5. **Identify colliders** | Common effects — don't condition on them |
| 6. **Apply backdoor criterion** | Find sufficient adjustment set to block confounding paths |

## **9.2 Adjustment Criteria**

| Criterion | When to Use | What It Does |
|-----------|-------------|-------------|
| **Backdoor** | Most common | Find variable set that blocks all backdoor paths from T to Y |
| **Front-door** | When backdoor fails (unobserved confounder) | Use mediator + two conditional independence assumptions |

## **9.3 Common DAG Mistakes**

| Mistake | Why It's Wrong | Consequence |
|---------|---------------|-------------|
| Controlling for a **collider** | Opens a non-causal path | Introduces bias (Berkson's paradox) |
| Controlling for a **mediator** | Blocks part of the causal effect | Underestimates treatment effect |
| Ignoring a **confounder** | Leaves a backdoor path open | Biased treatment effect estimate |
| Assuming no unobserved confounders | Almost always unrealistic | Overconfidence in results |

---

# **10. When to Use Which Method**

```
START: What's your data situation?
│
├── You can randomize?
│   └── A/B test (RCT) — always preferred if possible
│
├── Treatment assigned at a specific TIME to a GROUP?
│   ├── Multiple treated/control groups over time → DiD
│   └── Only one treated unit → Synthetic Control
│
├── Treatment assigned based on a THRESHOLD?
│   └── Regression Discontinuity (sharp or fuzzy)
│
├── Treatment is self-selected (observational)?
│   ├── You can identify a valid instrument → IV (2SLS)
│   └── No instrument, but have good covariates → Propensity Score Methods
│
└── Want to know WHO benefits (personalization)?
    └── Uplift Modeling (meta-learners, causal forests)
```

| Method | Key Assumption | Handles Unobserved Confounders? | Effect Estimated |
|--------|---------------|-------------------------------|-----------------|
| **A/B Test** | Random assignment | Yes (by design) | ATE |
| **DiD** | Parallel trends | Removes time-invariant unobserved | ATT |
| **PSM/IPW** | Unconfoundedness (no unobserved confounders) | No | ATE or ATT |
| **IV** | Valid instrument (relevance + exclusion) | Yes | LATE (compliers) |
| **RDD** | No manipulation at cutoff | Yes (locally) | Local ATE at cutoff |
| **Synthetic Control** | Good pre-treatment match | Time-invariant ones | ATT for treated unit |
| **Uplift** | Depends on base method | Depends | CATE (conditional ATE) |

---

# **11. Interview Questions with Strong Answers**

---

## **Q1: "You can't run an A/B test. How would you estimate the causal effect of a new feature?"**

> It depends on how the feature was rolled out:
>
> If it was launched at a **specific time to a specific group** (e.g., premium users), I'd use **Difference-in-Differences** — compare the change in the target metric for premium users (before/after) vs free users (before/after). The key assumption is parallel trends, which I'd validate by checking pre-launch metric trajectories.
>
> If it was rolled out to **everyone at once**, I'd look for a **synthetic control** — find a weighted combination of comparable markets or segments that didn't receive the feature, and use that as the counterfactual.
>
> If the feature was assigned based on a **threshold** (e.g., users with >1000 sessions), I'd use **Regression Discontinuity** — comparing users just above and below the threshold.
>
> For all these approaches, I'd run **sensitivity analyses and placebo tests** to validate the assumptions.

---

## **Q2: "Explain propensity score matching. What are its limitations?"**

> Propensity score matching balances treated and control groups on observed characteristics. I estimate the probability of treatment given covariates (using logistic regression or GBM), then match each treated unit to a control unit with a similar propensity score. This creates pseudo-randomization on observables.
>
> **Limitations:** The critical one is that PSM only controls for **observed** confounders. If there's an unobserved variable affecting both treatment and outcome, PSM gives biased estimates. It also requires the **overlap assumption** — there must be comparable control units for each treated unit. If some treated units have propensity scores near 1 (almost certain to be treated), there are no good matches.
>
> When I use PSM, I always: (1) check covariate balance after matching, (2) try multiple matching methods (nearest neighbor, kernel), (3) perform sensitivity analysis to estimate how strong an unobserved confounder would need to be to overturn the result.

---

## **Q3: "What is uplift modeling and when would you use it?"**

> Standard predictive modeling predicts P(Y|X). Uplift modeling predicts the **incremental** effect of treatment: P(Y|X, T=1) - P(Y|X, T=0). It tells you **who benefits** from an intervention.
>
> This is critical for targeting: in a marketing campaign, not everyone responds to a coupon the same way. "Sure things" would buy anyway (don't waste the coupon), "lost causes" won't buy regardless, and "do not disturb" customers actually churn more when contacted. We want to find **persuadables** — people who buy BECAUSE of the coupon.
>
> Implementation: I'd use a T-Learner (separate models for treated and control) or X-Learner (two-stage with counterfactual imputation), trained on data from a randomized holdout. Then rank users by predicted uplift and only target the top segment. This typically improves campaign ROI by 20-40% compared to untargeted approaches.
>
> At Fibe, this could have been applied to credit line increase decisions — target customers who increase spending (persuadables) without increasing those who default (do not disturb).

---

## **Q4: "You notice that users who use Feature X have 30% higher retention. Does Feature X cause higher retention?"**

> Not necessarily. This is a classic correlation vs causation question. There are at least three alternative explanations:
>
> **Confounding:** Power users are both more likely to discover Feature X AND more likely to retain. The confounder is "user engagement level."
>
> **Reverse causation:** Users who are already engaged (and would retain anyway) explore more features, including X.
>
> **Selection bias:** Feature X requires a multi-step onboarding. Users who complete it are more committed to begin with.
>
> To estimate the **causal effect**, I'd ideally run an A/B test (randomly promote Feature X to some users). If that's not possible, I'd use propensity score matching (match Feature X users to similar non-users on observed engagement metrics) or instrumental variables (find something that affects Feature X usage but not retention directly, like random variation in UI placement).
>
> I'd also check for **dose-response**: does more usage of Feature X correlate with more retention? That strengthens (but doesn't prove) the causal case.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│      CAUSAL INFERENCE TAKEAWAYS                                   │
│                                                                   │
│  1. A/B tests are gold standard, but often infeasible           │
│                                                                   │
│  2. DiD: For before/after + treatment/control comparisons       │
│     (requires parallel trends)                                   │
│                                                                   │
│  3. PSM/IPW: For observational data with good covariates        │
│     (can't handle unobserved confounders)                        │
│                                                                   │
│  4. IV: For unobserved confounders, if you have an instrument   │
│     (strongest but hardest to find valid instruments)            │
│                                                                   │
│  5. RDD: For threshold-based treatment assignment                │
│     (local estimate only, at the cutoff)                         │
│                                                                   │
│  6. Synthetic Control: For single treated unit with donor pool  │
│                                                                   │
│  7. Uplift Modeling: For personalized treatment effects          │
│     (who to target for maximum incremental impact)               │
│                                                                   │
│  8. DAGs help you THINK about causality — draw them before      │
│     running any analysis                                         │
│                                                                   │
│  9. ALWAYS do sensitivity analysis and placebo tests            │
│                                                                   │
│  10. Know the papers: Angrist, Imbens, Rubin (Nobel 2021)       │
└──────────────────────────────────────────────────────────────────┘
```
