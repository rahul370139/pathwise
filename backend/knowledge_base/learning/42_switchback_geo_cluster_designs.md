# Switchback, Geo, and Cluster Randomized Designs — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Experiment Design under Interference, Marketplace & Ops Experimentation

> **Context:** Files `24_statistics_and_ab_testing.md` and `33_causal_inference_and_experimentation.md` assume the **SUTVA** condition (Stable Unit Treatment Value Assumption) — that one user's outcome doesn't depend on another's treatment. In marketplaces, ad auctions, social networks, and ride-sharing, SUTVA is *broken*. This guide is the practical playbook for designs that handle **interference** and **spillovers**. Uber, Lyft, DoorDash, Instacart, Airbnb, and ad platforms all rely on these.

---

## Table of Contents

1. [Why User-Level A/B Breaks](#1-why-user-level-ab-breaks)
2. [The SUTVA Assumption and Its Failures](#2-the-sutva-assumption-and-its-failures)
3. [Cluster Randomization](#3-cluster-randomization)
4. [Geo Experiments](#4-geo-experiments)
5. [Switchback Experiments](#5-switchback-experiments)
6. [Two-Sided Randomization](#6-two-sided-randomization)
7. [Synthetic Control for Quasi-Experiments](#7-synthetic-control-for-quasi-experiments)
8. [Variance Estimation Under Clustering](#8-variance-estimation-under-clustering)
9. [Power & Sample Size for Cluster Designs](#9-power--sample-size-for-cluster-designs)
10. [Bias-Variance Trade-offs Across Designs](#10-bias-variance-trade-offs-across-designs)
11. [Decision Tree: Which Design to Use](#11-decision-tree-which-design-to-use)
12. [Interview Questions with Strong Answers](#12-interview-questions-with-strong-answers)
13. [Key Takeaways](#13-key-takeaways)

---

# 1. Why User-Level A/B Breaks

## 1.1 The Classic Setup

Standard A/B test:
- Randomize **users** to control/treatment.
- Assume **user A's outcome is independent of user B's treatment**.
- Run t-test. Done.

## 1.2 Where It Breaks — Three Canonical Examples

### Example A: Ride-sharing dispatch

Treatment: a better dispatch algorithm that's faster to match riders to drivers. If **riders** are randomized, treatment riders steal drivers from control riders → control wait times *increase* purely because treatment grabbed the supply. The measured lift is **inflated**.

### Example B: Ad auctions

Treatment: a new bidding model. If **advertisers** are randomized, treatment advertisers win more auctions, raising prices and shifting click distributions for **everyone**. SUTVA broken on the same auction.

### Example C: Social feed ranking

Treatment: a new feed ranker. If **users** are randomized, treatment users may engage more and produce more content, which **non-randomly** appears in control users' feeds. The control arm contains spillover from the treatment arm.

> **Common thread:** randomizing the wrong unit lets treatment "leak" into control through a shared resource — supply, prices, content, network connections.

## 1.3 What Happens to the Estimate

| Spillover direction | Effect on user-level A/B estimate |
|---------------------|-----------------------------------|
| Treatment **steals** from control (negative spillover on control) | Estimate **overstates** the true effect |
| Treatment **helps** control (positive spillover) | Estimate **understates** the true effect |
| Treatment changes a global price/quality | Estimate could go either way; biased |

The fix is **not** statistical correction. It's **changing the unit of randomization** to a level where interference is contained.

---

# 2. The SUTVA Assumption and Its Failures

## 2.1 SUTVA Formally

**SUTVA** has two parts:

1. **No interference** — unit i's outcome depends only on unit i's treatment.
2. **No hidden variations** — each treatment has a single, well-defined version.

When SUTVA holds, the potential outcomes $Y_i(0)$ and $Y_i(1)$ are well-defined and ATE estimation is clean.

## 2.2 When SUTVA Fails

| Failure mode | Example | Mitigation |
|-------------|---------|-----------|
| **Resource competition** | Drivers, ad inventory | Cluster on shared resource |
| **Network effects** | Social feeds, messaging | Cluster on community, ego-network design |
| **Global equilibrium** | Marketplace prices | Switchback or geo experiments |
| **Information leakage** | Word-of-mouth, virality | Geographic clustering |
| **Cross-device same user** | Streaming, search | Use stable user ID, not device |

## 2.3 The Estimand You Want

Under SUTVA: **ATE** is well-defined.
Under interference: you want the **TTE** (Total Treatment Effect) — the outcome difference between a world where *everyone* gets treatment and a world where *nobody* does. Cluster designs estimate this **policy-level** effect.

---

# 3. Cluster Randomization

## 3.1 Core Idea

Randomize **groups of users** (clusters) rather than individuals. All users in a cluster get the same variant. Interference *within* a cluster is OK; interference *across* clusters is assumed negligible.

```
USER-LEVEL A/B               CLUSTER A/B
┌──────────────────┐        ┌──────────────────┐
│ user1 → control  │        │ cluster1 (NY):   │
│ user2 → treatment│        │   all → control  │
│ user3 → treatment│        │                  │
│ user4 → control  │        │ cluster2 (SF):   │
│ ...              │        │   all → treatment│
└──────────────────┘        └──────────────────┘
```

## 3.2 What Counts as a Cluster

| Domain | Natural cluster |
|--------|----------------|
| Ride-sharing | City × hour, or geohash |
| Ads | Auction (impression-level) or advertiser |
| Social | Friend groups (community detection) |
| E-commerce | Geographic region, store |
| Streaming | Household |

## 3.3 The Cost — Effective Sample Size

Within a cluster, users are **correlated**. The effective sample size is:

$$
n_{\text{eff}} = \frac{n}{1 + (\bar{m} - 1) \cdot \text{ICC}}
$$

| Symbol | Meaning |
|--------|---------|
| $n$ | Total users |
| $\bar{m}$ | Average cluster size |
| **ICC** | Intra-cluster correlation (0 = independent, 1 = perfectly correlated) |

> **Example:** 100k users in 100 clusters (1k each) with ICC=0.05 → $n_{\text{eff}} = 100{,}000 / (1 + 999 \cdot 0.05) \approx 1{,}960$. Effective sample is ~50× smaller!

**Implication:** cluster designs need more total users than user-level A/B for the same power.

## 3.4 Estimator and Inference

The standard analysis is **cluster-robust standard errors**:

$$
\hat{\tau} = \bar{Y}_T - \bar{Y}_C, \quad \text{SE}_{\text{cluster}} = \sqrt{\frac{\sigma_T^2/k_T + \sigma_C^2/k_C}{k}}
$$

where $k_T, k_C$ are the **number of clusters** in each arm (not users!), and $\sigma^2$ is the variance of cluster means.

```python
import numpy as np

def cluster_ate(cluster_means_T, cluster_means_C):
    """Compute ATE and CI from cluster means."""
    ate = cluster_means_T.mean() - cluster_means_C.mean()
    se = np.sqrt(cluster_means_T.var(ddof=1) / len(cluster_means_T)
               + cluster_means_C.var(ddof=1) / len(cluster_means_C))
    return ate, ate - 1.96*se, ate + 1.96*se
```

---

# 4. Geo Experiments

## 4.1 What They Are

A **geo experiment** is a cluster design where the cluster is a **geographic region**: city, DMA (designated market area), zip code, or geohash.

## 4.2 Why Geos Are Special

- **Markets are local** — ride-sharing supply, restaurant inventory, ad spend all play out within a region.
- **Natural barrier** — users in NYC don't share Uber rides with users in LA.
- **Logging is cheap** — every event is already geo-tagged.

## 4.3 Typical Setup

```
┌─────────────────────────────────────────────────────────────┐
│                 GEO EXPERIMENT DESIGN                       │
│                                                             │
│  Step 1: Define cluster units                              │
│          (e.g., 100 DMAs in the US)                         │
│                                                             │
│  Step 2: Stratify-match clusters                           │
│          Pair similar DMAs (size, behavior pre-period)     │
│                                                             │
│  Step 3: Randomly assign within pairs                      │
│          One DMA per pair → treatment, other → control     │
│                                                             │
│  Step 4: Run experiment for ≥ 2-4 weeks                    │
│                                                             │
│  Step 5: Analyze with paired-difference or DiD             │
└─────────────────────────────────────────────────────────────┘
```

## 4.4 Matched-Pair Geo Design

To squeeze power out of few clusters:

1. **Pre-period embedding** — characterize each DMA by historical metrics (sessions/user, revenue, demographics).
2. **Pair** geographically distant but behaviorally similar DMAs (e.g., Austin ↔ Portland).
3. **Randomly assign** within each pair.
4. **Analyze pair-wise differences** for tighter CIs.

This is what Google's `GeoX` package does.

## 4.5 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Too few clusters | Wide CIs | Match-pair + synthetic control |
| Heterogeneous clusters | Variance dominated by outlier (e.g., NYC) | Weight by population OR log-transform |
| Spillover across borders | Treatment effect bleeds to control DMA | Larger geos (DMA > zip) or border buffers |
| Time-period contamination | Holiday hits one arm harder | Run ≥ 4 weeks, multi-period analysis |

## 4.6 Geo Experiment Math

For matched pairs (k pairs, one treated each):

$$
\hat{\tau} = \frac{1}{k}\sum_{i=1}^{k}(Y_{i,T} - Y_{i,C}), \quad \text{SE} = \frac{s_d}{\sqrt{k}}
$$

where $s_d$ is the standard deviation of pair-wise differences.

---

# 5. Switchback Experiments

## 5.1 What It Is

Randomize over **time slots** rather than users. The entire market is on treatment in some time windows and on control in others.

```
                Switchback for City: Austin
   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
   │ T    │ C    │ C    │ T    │ C    │ T    │ T    │
   ├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
   │ 9am  │ 10am │ 11am │ 12pm │ 1pm  │ 2pm  │ 3pm  │
   └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
   Every hour: random assignment to T or C
```

## 5.2 Why Switchback Solves Interference

At any moment in time, **everyone in the city** is on the same variant. No within-time spillover possible.

## 5.3 Where It Works Best

| Domain | Reason |
|--------|--------|
| Ride-sharing / dispatch | Supply/demand equilibrium changes within minutes |
| Food delivery (DoorDash, Uber Eats) | Same as above |
| Auction-clearing markets | Equilibrium per auction batch |
| Pricing changes | Need to measure consumer + supplier response together |

## 5.4 The Core Problem — Carryover

A treatment in slot t can affect outcomes in slot t+1 (e.g., a driver matched in treatment slot may still be busy in next control slot). This **biases** the estimate.

**Mitigations:**

1. **Burn-in / washout windows** between slots — discard data from the first N minutes after a switch.
2. **Longer slots** — reduce relative size of carryover.
3. **Carryover modeling** — estimate effect as ATE adjusted for previous slot's variant.

## 5.5 Typical Slot Length

| Domain | Typical slot length |
|--------|--------------------|
| Ride dispatch | 30 min – 1 hour |
| Pricing | 1 hour – 6 hours |
| Search ranking (recovery dynamics fast) | 15 min |
| Long carryover effects (e.g., booking) | 1 day |

## 5.6 Analysis Patterns

### Diff-in-means by slot

$$
\hat{\tau} = \bar{Y}_{\text{slots with T}} - \bar{Y}_{\text{slots with C}}
$$

Variance: across slots, not across individual events. This is critical — **the slot is the unit**, not the rider/order.

### Fixed-effects regression

$$
Y_{rt} = \alpha_r + \gamma_t + \tau D_t + \varepsilon_{rt}
$$

| Symbol | Meaning |
|--------|---------|
| $r$ | Region (if multi-city) |
| $t$ | Slot |
| $\alpha_r$ | Region fixed effect (controls for permanent regional differences) |
| $\gamma_t$ | Time fixed effect (controls for hour-of-day, day-of-week) |
| $D_t$ | Treatment indicator for slot t |
| $\tau$ | Treatment effect |

## 5.7 Switchback Pitfalls

| Pitfall | Fix |
|---------|-----|
| Carryover between slots | Washout windows |
| Treatment imbalance over time (e.g., all peak hours got treatment by chance) | Stratified random by hour-of-day |
| Confounded with daily seasonality | Match pairs of (treatment, control) within day |
| Switch costs (e.g., system warmup) | First few minutes excluded |
| Too short slots → high variance | Tune by power calc |

---

# 6. Two-Sided Randomization

## 6.1 When You Need It

In two-sided marketplaces (Airbnb hosts ↔ guests, Uber drivers ↔ riders), a feature can affect **both** sides. Randomizing only one side misses the half of the effect that flows through the other.

## 6.2 Designs

### One-side randomization (simple)
Randomize riders only. Limitation: driver side reacts to aggregated rider behavior; effects on driver supply are missed.

### Two-side independent randomization
Randomize riders AND drivers independently. Estimate side-specific effects + interaction. Cost: 4 cells (T-T, T-C, C-T, C-C) → power loss.

### Cluster two-side (recommended)
Randomize one side at the **cluster** level (e.g., cities for drivers) and individuals on the other side. The "driver" cluster sees a coherent supply-side policy; rider-side randomization is clean within.

## 6.3 Bipartite Experiments

A formalization in the literature: bipartite experiments randomize at the **edge** level (e.g., per ride) but estimate effects on supply (drivers) and demand (riders) separately, with cluster-robust standard errors per side.

---

# 7. Synthetic Control for Quasi-Experiments

## 7.1 When Cluster Designs Aren't Possible

Sometimes you launch a feature in **one city** (no control geo). Or a policy change is regional and you can't randomize. Use **synthetic control**.

## 7.2 The Method (Brief)

1. **One treated unit** (e.g., city A gets a new pricing policy on date $t_0$).
2. **Many candidate control units** (other cities, same period).
3. **Build a weighted average** of control units that closely matches the treated unit's pre-period trajectory.
4. **Post-period gap** between actual treated and synthetic control = treatment effect.

$$
Y^{\text{synth}}_t = \sum_i w_i \cdot Y_i^{(C)}_t, \quad w_i \geq 0, \quad \sum w_i = 1
$$

with weights chosen to minimize pre-period prediction error.

## 7.3 When to Use

- Only one or few treated geos
- Sufficient pre-period history (≥ 6 months ideal)
- Stable economic conditions
- No anticipation effects

## 7.4 Inference

Standard errors are tricky. Use **placebo tests** — run synthetic control on each control geo as if it were treated, build a distribution of pseudo-effects, compare your actual treated geo's effect against that distribution.

> Full synthetic control coverage is in `33_causal_inference_and_experimentation.md` §7.

---

# 8. Variance Estimation Under Clustering

## 8.1 The Wrong Way (Naïve)

Treating individual observations as independent → **standard errors too small** → false positives.

## 8.2 Cluster-Robust SE (CR0/CR1/CR2)

The "Liang-Zeger" sandwich estimator:

$$
\hat{V}_{\text{cluster}} = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' \hat{u}_g \hat{u}_g' X_g \right) (X'X)^{-1}
$$

| Variant | When to use |
|---------|------------|
| **CR0** | Many clusters (G > 50) |
| **CR1** | Default; small-sample correction |
| **CR2** | Few clusters (G < 30) — bias-corrected |

Most stats libraries (`statsmodels`, R `sandwich`, Stata `cluster`) implement these.

## 8.3 Wild Cluster Bootstrap

When clusters are very few (G < 20), the asymptotic CR-SE breaks. Use **wild cluster bootstrap**:

1. Fit the model, get residuals.
2. For each bootstrap iteration b:
   - For each cluster g, draw $w_g \in \{-1, +1\}$ (Rademacher).
   - Multiply cluster's residuals by $w_g$.
   - Refit and store coefficient.
3. Use bootstrap distribution for inference.

This is the **gold standard** for small-G geo experiments.

## 8.4 Code Sketch (Python)

```python
import statsmodels.formula.api as smf

# Cluster-robust regression
model = smf.ols("Y ~ T + region + time", data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cluster_id"]}
)
print(model.summary())
```

---

# 9. Power & Sample Size for Cluster Designs

## 9.1 Design Effect

The "design effect" tells you how much sample size to inflate:

$$
\text{DE} = 1 + (\bar{m} - 1) \cdot \text{ICC}
$$

Required sample size for cluster design:

$$
n_{\text{cluster}} = n_{\text{individual}} \cdot \text{DE}
$$

| ICC | Cluster size | Design effect | Sample inflation |
|-----|-------------|---------------|------------------|
| 0.01 | 100 | 2.0 | 2× |
| 0.05 | 100 | 5.95 | 6× |
| 0.10 | 100 | 10.9 | 11× |
| 0.05 | 1000 | 50.95 | 51× |

> **Lesson:** big clusters dramatically reduce power even at small ICC.

## 9.2 Estimating ICC

From pre-experiment data:

$$
\text{ICC} = \frac{\sigma^2_{\text{between cluster}}}{\sigma^2_{\text{between}} + \sigma^2_{\text{within}}}
$$

Fit a random-effects model with cluster as the grouping variable; ICC is part of the output.

## 9.3 The Lever You Have

If sample is limited:
- **Fewer, bigger clusters** → less efficient (high DE)
- **Many smaller clusters** → more efficient

So when designing a geo experiment, prefer **DMAs over states**, and **states over country**.

---

# 10. Bias-Variance Trade-offs Across Designs

| Design | Bias if SUTVA broken | Variance | Best for |
|--------|---------------------|----------|----------|
| **User-level A/B** | High (when interference present) | Low | Independent users (most product changes) |
| **Cluster (user community)** | Low | Medium | Network effects |
| **Geo** | Low | Medium-High | Marketplaces, local equilibrium |
| **Switchback** | Low | High (few "units" = slots) | Ride dispatch, pricing, auctions |
| **Two-sided** | Low | High (split power) | Marketplaces with both sides reacting |
| **Synthetic control** | Low if pre-period match good | Variable | One treated unit |

> **Bias vs variance** is the constant fight. User-level A/B is the **most powerful** but the **most biased** under interference. Switchback is the **least biased** but eats power.

---

# 11. Decision Tree: Which Design to Use

```
┌─────────────────────────────────────────────────────────────────┐
│                  DESIGN DECISION TREE                           │
│                                                                 │
│  Q1: Does the treatment affect other users via a shared        │
│      resource, network, or price?                              │
│                                                                 │
│      NO  →  USER-LEVEL A/B                                     │
│      YES →  go to Q2                                           │
│                                                                 │
│  Q2: Is the spillover bounded geographically?                  │
│                                                                 │
│      YES →  GEO EXPERIMENT (DMAs, cities)                      │
│      NO  →  go to Q3                                           │
│                                                                 │
│  Q3: Does the effect equilibrate within minutes/hours?         │
│                                                                 │
│      YES →  SWITCHBACK                                         │
│      NO  →  go to Q4                                           │
│                                                                 │
│  Q4: Is there a natural network/community structure?           │
│                                                                 │
│      YES →  CLUSTER (community detection)                      │
│      NO  →  go to Q5                                           │
│                                                                 │
│  Q5: Is only one geo/unit treated?                             │
│                                                                 │
│      YES →  SYNTHETIC CONTROL                                  │
│      NO  →  reconsider (maybe SUTVA holds well enough?)       │
└─────────────────────────────────────────────────────────────────┘
```

---

# 12. Interview Questions with Strong Answers

## Q1: "Why can't you just run a user-level A/B test on Uber's new dispatch?"

> Because the treatment changes the **allocation of a shared resource** — drivers. If we put riders into treatment vs control on the same map at the same time, the treatment group's algorithm pulls drivers away from the control group. Control wait times go *up* not because of the algorithm itself but because they lost supply. The measured effect is overstated. The fix is to randomize at a level where the shared resource is contained — usually a **geo × time switchback** so the entire city is on one variant in any given time slot. Inside that slot, there's no competition between arms.

## Q2: "Walk me through designing a switchback experiment for a new dispatch algorithm."

> Five steps. **Step 1**: choose slot length — long enough that carryover from the prior slot has dissipated, short enough that we get many slots for statistical power. For dispatch, 30 minutes to 1 hour is typical. **Step 2**: stratify by hour-of-day and day-of-week — randomize within each (hour, weekday) bucket so the treatment isn't accidentally biased toward peak periods. **Step 3**: define washout windows — the first 5–10 minutes of each slot are excluded because rides matched in the prior variant are still in progress. **Step 4**: run for ≥ 2 weeks to cover normal weekly cycles. **Step 5**: analyze with a fixed-effects regression — slot is the unit, region and time fixed effects, treatment indicator. Standard errors clustered at the slot level. The estimand is the average effect of the new dispatch on per-slot metrics like wait time, conversion, or rides completed.

## Q3: "What is the design effect and why does it matter?"

> The design effect quantifies the loss of statistical power when you randomize clusters instead of individuals. Formula: $\text{DE} = 1 + (\bar{m}-1) \cdot \text{ICC}$ where $\bar{m}$ is the average cluster size and ICC is the intra-cluster correlation. If you randomize 100 users in clusters of size 10 with ICC=0.05, your effective sample size isn't 100, it's $100 / [1 + 9 \cdot 0.05] = 69$. At DMA-level geo experiments with thousands of users per cluster, the design effect can be 50× or more, meaning you need 50× more total users for the same power. This is why geo designs trade bias for variance — they're unbiased under interference but very expensive in sample. You can mitigate by using **more, smaller clusters** rather than fewer big ones.

## Q4: "How do you handle carryover in a switchback experiment?"

> Three layers. **First, washout windows** — exclude the first N minutes of each slot from analysis. For ride-sharing, rides in progress at the moment of switch persist for ~15 minutes, so the first 15 minutes after a switch are tossed. **Second, slot length** — make slots long enough (~1 hour) that the carryover region is a small fraction of the slot. **Third, explicit modeling** — include a lagged variant indicator in the regression: $Y_t = \alpha + \tau D_t + \beta D_{t-1} + \gamma_t + \varepsilon$. The coefficient $\tau$ is the contemporaneous effect; $\beta$ measures spillover. If $\beta$ is large, the design is breaking and you need longer slots or longer washouts. We continuously monitor $\beta$ as a diagnostic on the platform.

## Q5: "Geo experiments have few clusters — how do you get reliable inference with G=20 DMAs?"

> Asymptotic cluster-robust standard errors (Liang-Zeger CR1) assume G is large, and break down below ~30 clusters. Below that, two pragmatic tools. **Wild cluster bootstrap** — refit the model many times with randomly flipped cluster residuals to build an empirical distribution of the treatment coefficient under the null. **Matched-pair design** — pair similar DMAs pre-experiment and randomize within each pair, then analyze pair-wise differences. The pair structure dramatically reduces variance because within-pair covariates are nearly identical. With both techniques, even G=20 can give defensible p-values. The platform should default to wild cluster bootstrap for any analysis with G < 30 and warn the analyst.

## Q6: "When would you use synthetic control over a cluster A/B?"

> Synthetic control shines when you can't randomize at all — typically because the feature is rolled out to a specific geo (one city) for business reasons, or because of a regulatory event. With one treated unit, there's no within-arm variability to estimate SE the usual way. Synthetic control builds a counterfactual from a weighted combination of untreated units, chosen to match the treated unit's pre-period trajectory closely. You then compare actual post-period to the synthetic counterfactual. Inference is via **placebo tests** — run the same procedure on each control unit and compare. The main assumption is that pre-period match quality + parallel trends carry forward; if the world changes in the post-period for reasons unrelated to treatment, the estimate is biased. For a true experiment with multiple treated and control units, cluster A/B is cleaner.

## Q7: "What's the difference between a cluster experiment and a stratified experiment?"

> They sound similar but solve different problems. **Stratification** is about *balance* — you randomize within strata to ensure both arms have similar demographics. Individuals are still the unit. **Clustering** is about *interference* — entire groups are randomized together because individuals within a group can affect each other. You'd combine both: cluster at the geo level (because of marketplace interference) and stratify pairs of geos by population and pre-period metrics (to ensure the two arms are comparable). Clustering hurts power because of design effect; stratification helps power by reducing variance. They're complementary tools.

## Q8: "Suppose you ran a geo experiment and got SRM (8 DMAs in treatment, 12 in control instead of 10/10). What do you do?"

> First, do not analyze the experiment as if SRM is OK. SRM in a geo experiment can mean: (1) randomization bug — likely if assignments came from a hash function not properly salted; (2) DMA-level filtering after assignment (e.g., one DMA was excluded for a data quality issue); (3) sample-aware enrollment (some DMAs opted out). Walk back through the assignment audit log and check that every DMA in the population was eligible and randomized correctly. If the imbalance is structural (one DMA had a recording outage), you may proceed with the analysis but must report the population — the estimand is now "ATE among DMAs with valid data," not "all DMAs." Document, escalate, and re-run if possible. With only 20 DMAs total, every cluster matters.

## Q9: "How would you A/B test a search ranking change at a marketplace like Etsy?"

> Two competing considerations: ranking changes affect what sellers experience (their items get more or fewer views) and what buyers see. A naive user-level A/B on buyers measures the buyer-side effect cleanly but ignores the seller side. If treatment buyers click on rare niche items, those sellers get a temporary boost — but in the steady state, all sellers see the new ranker, so the effect could be different. For an Etsy-scale platform, I'd do: (1) **user-level A/B** for short-term buyer-side metrics (CTR, conversion); (2) **geo experiment** if the marketplace has strong regional dynamics; (3) **post-launch holdout** to measure long-run cumulative effects on seller mix and platform health. The platform should report all three views and let leadership see the full picture.

## Q10: "What's the biggest pitfall you've seen in cluster designs?"

> Underestimating ICC. Teams often see "100k users across 100 clusters" and assume their power is set; in reality the effective sample is much smaller. The first deliverable from an analyst doing a geo design should be **ICC estimation from historical data** — fit a random-effects model on the candidate metric pre-experiment and report $\sigma^2_{\text{between}} / \sigma^2_{\text{total}}$. For typical marketplace metrics (orders, GMV, sessions), ICC at city level is often 0.05–0.20, which translates to massive design effects when clusters are large. The platform should require an ICC report as a gate to launching any cluster experiment, the same way it requires a power calc for user-level tests.

---

# 13. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│       SWITCHBACK / GEO / CLUSTER — INTERVIEW TAKEAWAYS              │
│                                                                     │
│  1. SUTVA breaks when users share a resource (drivers, ad         │
│     inventory), price, or network. User-level A/B then biases.     │
│                                                                     │
│  2. CLUSTER design = randomize groups. Estimand is policy-level   │
│     (TTE), not individual-level (ATE).                             │
│                                                                     │
│  3. GEO = cluster on geography. Use matched-pair design for       │
│     power.                                                         │
│                                                                     │
│  4. SWITCHBACK = randomize over TIME. Solves spatial interference  │
│     but introduces carryover.                                      │
│                                                                     │
│  5. Design effect: DE = 1 + (m-1)·ICC. Big clusters destroy power.│
│                                                                     │
│  6. Cluster-robust SE (CR1) needed for inference. Wild cluster    │
│     bootstrap when G < 30.                                         │
│                                                                     │
│  7. Two-sided marketplaces may need two-sided randomization OR    │
│     cluster on one side + user-level on the other.                 │
│                                                                     │
│  8. Synthetic control: one treated unit, many control units, fit  │
│     a counterfactual on pre-period.                                │
│                                                                     │
│  9. Carryover in switchback: handle with washout windows, longer   │
│     slots, and explicit lagged-treatment regression.               │
│                                                                     │
│  10. The trade-off is ALWAYS bias vs variance. User A/B = least    │
│      bias when SUTVA holds, most bias when it doesn't. Cluster =  │
│      reverse.                                                      │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"When users interact through a shared resource — drivers, inventory, prices, social graph — user-level A/B silently biases the result. Cluster, geo, and switchback designs trade statistical power for unbiased estimation of the policy-level effect. The art is knowing when to make that trade."*

---

**Further reading**

- Athey, Imbens (2017) — *The Econometrics of Randomized Experiments*
- Bojinov, Simchi-Levi, Zhao (2020) — *Design and Analysis of Switchback Experiments*
- Glynn, Kulkarni, Hubbard (2020) — *Adaptive Switchback Designs*
- Abadie, Diamond, Hainmueller (2010) — *Synthetic Control Methods*
- Eckles, Karrer, Ugander (2017) — *Design and Analysis of Experiments in Networks*
- Bakshy, Eckles, Yan, Rosenn (2012) — *Social Influence in Social Advertising*
