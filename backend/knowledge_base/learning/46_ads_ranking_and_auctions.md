# Ads Ranking & Auctions — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — Ads Ranking, pCTR, Calibration, Auctions, Multi-Objective Ranking

> **Context:** `35_recommendation_systems.md` covers generic LTR (Learning to Rank). This guide is the **ads-specific** deep dive that interviewers at Google Ads, Meta Ads, Microsoft Ads, Amazon Ads, TikTok, Reddit, Pinterest, X / Twitter, and OpenAI (if they monetize via sponsored content) test for. It covers auctions (VCG, GSP), pCTR / pCVR prediction, calibration, expected value scoring, budget pacing, multi-objective ranking, and the experimentation patterns unique to ads.

---

## Table of Contents

1. [Ads vs Organic Ranking — The Differences](#1-ads-vs-organic-ranking--the-differences)
2. [The Ads Auction](#2-the-ads-auction)
3. [pCTR Prediction — The Workhorse Model](#3-pctr-prediction--the-workhorse-model)
4. [Calibration — The Hidden Killer](#4-calibration--the-hidden-killer)
5. [Expected Value Scoring](#5-expected-value-scoring)
6. [Multi-Objective Ads Ranking](#6-multi-objective-ads-ranking)
7. [Budget Pacing](#7-budget-pacing)
8. [Position Bias and Click Modeling](#8-position-bias-and-click-modeling)
9. [Exploration vs Exploitation](#9-exploration-vs-exploitation)
10. [Ads Experimentation Patterns](#10-ads-experimentation-patterns)
11. [Ad Quality and User Welfare](#11-ad-quality-and-user-welfare)
12. [Interview Questions with Strong Answers](#12-interview-questions-with-strong-answers)
13. [Key Takeaways](#13-key-takeaways)

---

# 1. Ads vs Organic Ranking — The Differences

## 1.1 Goals

| Organic ranking | Ads ranking |
|----------------|-------------|
| Maximize user engagement / satisfaction | Maximize **a weighted combination of**: ad revenue, user satisfaction, advertiser ROI |
| Single objective (usually) | Multi-objective |
| Free | Paid (advertiser pays per click / impression / conversion) |
| No explicit incentive compatibility | Strict auction rules — must be incentive-compatible |
| User is the only "customer" | Three "customers": user, advertiser, platform |

## 1.2 The Three-Sided Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                THE ADS THREE-SIDED PROBLEM                       │
│                                                                  │
│      USER (sees ads)            ADVERTISER (pays for ads)       │
│         │                           │                           │
│         │ wants:                    │ wants:                    │
│         │  relevant                 │  ROI                      │
│         │  non-annoying             │  high-intent users         │
│         │  good experience          │  cost-effective bidding   │
│         │                           │                           │
│         ▼                           ▼                           │
│         ─────────  PLATFORM  ───────────                        │
│                       │                                         │
│                       │ wants:                                 │
│                       │  short-term: ad revenue                │
│                       │  long-term: user retention             │
│                       │             advertiser retention       │
│                       │             ecosystem health           │
└─────────────────────────────────────────────────────────────────┘
```

Optimizing only short-term ad revenue burns the user surface and the long-term user base. Modern ads ranking is **multi-objective**, balancing immediate revenue against user welfare proxies.

---

# 2. The Ads Auction

## 2.1 Why Auctions

When demand for ad slots exceeds supply, the platform needs a mechanism to allocate slots and price them. Auctions are the mechanism that **discovers true willingness to pay** while being mostly fair across advertisers.

## 2.2 The Generalized Second-Price (GSP) Auction

Used by Google, Facebook (historically), most major platforms.

```
For a single ad slot:
1. Each advertiser submits a bid (max willingness to pay per click).
2. Compute ad rank: ad_rank = bid × pCTR (× quality)
3. Highest ad_rank wins.
4. Winner pays the price needed to beat the next ad rank.
```

The "second price" twist: you pay just enough to beat the next bidder, not your full bid. This **encourages truthful bidding** approximately.

### Example

| Advertiser | Bid | pCTR | ad_rank |
|------------|-----|------|---------|
| A | $5.00 | 0.04 | 0.20 |
| B | $3.00 | 0.10 | 0.30 |
| C | $2.00 | 0.05 | 0.10 |

B wins. Price B pays per click = ad_rank_of_A / pCTR_B + ε = 0.20/0.10 + 0.01 = $2.01.

## 2.3 The Vickrey-Clarke-Groves (VCG) Auction

Strictly truthful auction. Each bidder pays the externality they impose on others — the lost value others would have gotten without this bidder.

VCG is **dominant-strategy truthful** (you can never gain by misreporting your bid). GSP is only approximately truthful but more transparent.

Facebook switched from GSP to VCG-like in 2017.

## 2.4 Reserve Prices

Minimum bid required to participate. Set per-slot, per-keyword. Functions:
- Prevent low-quality ads from winning at trivial prices
- Capture revenue when there's only one bidder

Setting reserves is a **revenue optimization** problem informed by demand curves.

## 2.5 Auction Setup in Practice

```python
def run_auction(ads, slot, reserve_price):
    """Simple GSP for one slot."""
    scored = [
        (a, a.bid * a.pCTR * a.quality_score)
        for a in ads
        if a.bid * a.pCTR >= reserve_price
    ]
    scored.sort(key=lambda x: -x[1])

    if not scored:
        return None, 0.0

    winner, winner_rank = scored[0]
    second_rank = scored[1][1] if len(scored) > 1 else reserve_price
    price = second_rank / (winner.pCTR * winner.quality_score) + 0.01

    return winner, min(price, winner.bid)
```

---

# 3. pCTR Prediction — The Workhorse Model

## 3.1 What pCTR Is

**Predicted Click-Through Rate** = the model's estimate of P(user clicks | ad shown).

$$
\hat{p}_{\text{CTR}}(u, a, c) = P(\text{click} \mid \text{user } u, \text{ad } a, \text{context } c)
$$

This is the most-deployed ML model in the world. Every ad impression, every platform.

## 3.2 Why pCTR Matters

In the auction: ad_rank = bid × pCTR. Poor pCTR estimation = wrong winners = lost revenue + bad user experience.

If pCTR is **biased high** for some ads: they win too often, users see irrelevant ads, advertisers pay more than they should.
If pCTR is **biased low**: those ads lose unfairly, advertisers leave.

## 3.3 Features

| Category | Examples |
|----------|----------|
| **User** | Demographics, history, recent searches, embedding |
| **Ad** | Creative text, image features, advertiser, category |
| **Context** | Time, device, page, query, surface |
| **User × Ad** | Past CTR on similar ads, embedding similarity |
| **Cross features** | Interaction terms (user.country × ad.category) |

## 3.4 Model Architectures

| Era | Architecture |
|-----|-------------|
| 2000s | Logistic regression with hand-crafted cross features |
| 2010s | Gradient boosting (LightGBM, XGBoost) on tabular features |
| Late 2010s | Wide & Deep, DeepFM, DCN — combine memorization and generalization |
| 2020s | Transformers, attention-based sequence models, sparse embeddings, large-scale embeddings |

Production systems often use **two-tower** (user tower + ad tower) for retrieval, plus a **cross-feature heavy** model for final ranking.

## 3.5 Training Setup

- **Label:** click (1) / no click (0)
- **Loss:** binary cross-entropy
- **Training data:** all impressions, often subsampled by user or session
- **Scale:** billions of impressions per day at top platforms
- **Refresh cadence:** continuous (online learning) for top tier; daily/hourly for most

## 3.6 Sample Selection Bias

We only observe labels for ads that were **shown**. Ads that lost the auction never got an impression and thus no click label. This is **selection bias**:
- Training on what was shown → model learns to predict CTR conditional on selection
- Deployed against new candidates → many are "out of distribution"

**Mitigation:**
- **Exploration** (small % random impressions)
- **Importance weighting** by inverse propensity
- **Counterfactual log labels** when available

---

# 4. Calibration — The Hidden Killer

## 4.1 What Is Calibration

A model is **calibrated** if its predicted probabilities match observed frequencies:

$$
\text{Among predictions of } \hat{p} = 0.1, \text{ the empirical CTR is } 10\%.
$$

A model can be **good at ranking** (AUC = 0.9) but **poorly calibrated** (predicts 0.5 when truth is 0.1).

## 4.2 Why Calibration Matters in Ads

In the auction: ad_rank = bid × pCTR. If pCTR is **2x too high**, the ad wins twice as often as it should, and the winner pays 2x too much.

For organic ranking, calibration doesn't matter (only relative order does). **For ads, calibration is critical** because it determines absolute price.

## 4.3 Measuring Calibration

### Calibration plot (reliability diagram)

Bin predicted probabilities, plot mean predicted vs observed frequency:

```
Empirical
 CTR  ▲       ★ ←—— perfectly calibrated
      │      /
      │     ★
      │    /
      │   ★
      │  /
      │ ★
      └──────────► Predicted CTR
```

### Expected Calibration Error (ECE)

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \bar{p}_b - \bar{y}_b \right|
$$

where bins are over predicted probability ranges.

### Population-level calibration

$$
\text{Avg pCTR over all impressions} \approx \text{Empirical CTR overall}
$$

This is the **first** check in production — easy to monitor.

## 4.4 Why Models Become Mis-Calibrated

| Reason | Mechanism |
|--------|-----------|
| **Class imbalance** | CTR ~1-5%; training on imbalanced data → conservative predictions |
| **Downsampling negatives** | Common for tractability; biases pCTR upward unless corrected |
| **Distribution shift** | Train and serve distributions differ; calibration drifts |
| **Position bias** | Train data has position-dependent CTRs; serving may put ad in different slot |
| **Model architecture** | Some losses (e.g., focal loss) are not calibrated by default |

## 4.5 Fixing Calibration

### Platt scaling

Fit a logistic regression on the predicted logit:

$$
p_{\text{cal}} = \sigma(a \cdot \text{logit}(\hat{p}) + b)
$$

Two parameters; very effective.

### Isotonic regression

Non-parametric monotonic mapping from raw to calibrated. More flexible than Platt.

### Negative-sampling correction

If training subsampled negatives by factor $r$, the true probability is:

$$
p_{\text{true}} = \frac{p_{\text{model}}}{p_{\text{model}} + (1 - p_{\text{model}})/r}
$$

### Post-hoc rescaling

Periodically rescale predictions by the ratio `empirical_CTR / avg_predicted_CTR`. Simple and effective.

---

# 5. Expected Value Scoring

## 5.1 The General Form

Ranking by raw pCTR ignores **what an advertiser pays**. The auction needs:

$$
\text{Score} = \text{expected revenue} = \text{bid} \cdot \hat{p}_{\text{CTR}}
$$

For conversion-based bidding (advertiser pays per conversion, e.g., purchase):

$$
\text{Score} = \hat{p}_{\text{conv}} \cdot \text{value}_{\text{conv}}
$$

where conversion is further decomposed: $\hat{p}_{\text{conv}} = \hat{p}_{\text{CTR}} \cdot \hat{p}_{\text{conv|click}}$.

## 5.2 Multi-Outcome Scoring

For long sessions and indirect conversions:

$$
\text{Score} = \sum_k w_k \cdot \hat{p}_k \cdot v_k
$$

where $k$ indexes outcomes (click, page view, add to cart, purchase, lifetime value).

## 5.3 Bid Types

| Type | Mechanism | Use |
|------|-----------|-----|
| **CPC** (cost per click) | Pay if user clicks | Brand awareness, traffic |
| **CPM** (cost per mille = 1000 impressions) | Pay per impression | Reach |
| **CPA** (cost per action / acquisition) | Pay if user converts | Direct response |
| **tCPA** (target CPA) | Platform optimizes to hit a target | Most automated bidding |
| **Smart Bidding (Google)** | Platform sets bid to maximize advertiser goal | Almost all modern campaigns |

## 5.4 The Bid Translation Layer

Internally, platforms convert all bid types to a common metric — **expected value per impression (eVPI)**:

$$
\text{eVPI} = \begin{cases}
\hat{p}_{\text{CTR}} \cdot \text{bid}_{\text{CPC}} & \text{if CPC} \\
\text{bid}_{\text{CPM}} / 1000 & \text{if CPM} \\
\hat{p}_{\text{CTR}} \cdot \hat{p}_{\text{conv|click}} \cdot \text{bid}_{\text{CPA}} & \text{if CPA} \\
\end{cases}
$$

This is the auction's currency. All bid types compete fairly when reduced to eVPI.

---

# 6. Multi-Objective Ads Ranking

## 6.1 The Objectives

Modern ads ranking optimizes:

1. **Revenue** — short-term platform earnings
2. **Advertiser value** — long-term advertiser ROI
3. **User satisfaction** — engagement, retention, complaint rate
4. **Diversity** — variety of advertisers, categories
5. **Fairness** — no advertiser unfairly suppressed

## 6.2 Linear Combination

$$
\text{Final score} = \alpha \cdot eVPI + \beta \cdot \text{quality} + \gamma \cdot \text{relevance} - \delta \cdot \text{annoyance}
$$

Weights tuned via A/B tests on a long-term outcome (often a holdout-based monthly DAU lift).

## 6.3 Constrained Optimization

Maximize revenue **subject to** user satisfaction constraints:

$$
\max \sum_t \text{revenue}_t \quad \text{s.t.} \quad \text{user satisfaction}_t \geq \theta
$$

Solved via **Lagrangian** approaches: introduce a shadow price for user satisfaction, optimize jointly.

## 6.4 Pareto Frontier

Plot revenue vs user satisfaction over a grid of weights. The **Pareto frontier** is the set of non-dominated configurations. Pick the operating point based on business priorities.

```
Revenue
   ▲
   │      ★ point A (max revenue, low satisfaction)
   │   ★
   │ ★
   │   ★
   │     ★
   │       ★ point Z (max satisfaction, low revenue)
   └──────────────► User satisfaction
```

## 6.5 Long-Term Holdouts for Multi-Objective

Short-term metric: revenue up, complaints up. Ship or not?

Run a **long-term holdout**: 1% of users never see the new ranker. Measure DAU, retention, ARPU over months. If long-term ARPU stays positive in the treated population vs holdout, the ranker is net positive even if short-term complaints rose.

---

# 7. Budget Pacing

## 7.1 The Problem

Advertisers set daily / monthly budgets. The platform must spread spend across the day to avoid:
- Running out of budget early (missing valuable later impressions)
- Underspending (advertiser leaves money on the table)

## 7.2 Naive Pacing

Spend at a constant rate over the day. Misses fact that some hours have higher value.

## 7.3 Optimization Pacing

Define a **pacing multiplier** $\rho \in [0, 1]$ on bids:

$$
\text{effective bid} = \rho \cdot \text{bid}
$$

Adjust $\rho$ throughout the day so cumulative spend follows the optimal path.

If ahead of budget, decrease $\rho$. If behind, increase. Control-theory-style PID loops are common.

## 7.4 Throttling vs Pacing

| Throttling | Pacing |
|------------|--------|
| Skip auctions to save budget | Lower bid to win fewer |
| Misses entire impressions | Wins cheaper impressions |
| Aggressive | Smoother |

Pacing is preferred unless throttling is required by SLA.

---

# 8. Position Bias and Click Modeling

## 8.1 What Is Position Bias

Users click higher-ranked items more, **regardless of relevance**:

```
Position 1: 25% CTR
Position 2: 15% CTR
Position 3: 10% CTR
...
Position 10: 2% CTR
```

If you train pCTR naively, the model learns: "high position → high CTR" instead of "high relevance → high CTR." Then it overestimates pCTR for ads that have historically been in top positions.

## 8.2 The Cascade Model

Models the click decision as:

```
User scans from top to bottom.
At each position, with some probability they CLICK and stop.
With remaining probability, they SKIP and continue.
```

This gives a position-conditional click probability:

$$
P(\text{click} \mid \text{relevance}, \text{position}) = P(\text{examined} \mid \text{position}) \cdot P(\text{click} \mid \text{examined, relevance})
$$

Training on **examination probability × relevance** separates the two effects.

## 8.3 Position Bias Estimation

Three approaches:

1. **Result randomization**: occasionally shuffle results to learn unbiased position effects.
2. **Regression-based**: train a click model with position as a feature, estimate the marginal.
3. **EM-style**: alternate between estimating examination probability and relevance.

## 8.4 IPS (Inverse Propensity Scoring)

For each click, weight by 1 / P(impression | logging policy). Common in counterfactual evaluation of new rankers.

---

# 9. Exploration vs Exploitation

## 9.1 Why Explore

If the model is uncertain about a new ad's pCTR, always trusting the prediction can be wrong. **Exploration** — showing the ad despite uncertainty — gathers data to improve future predictions.

## 9.2 Common Strategies

| Strategy | Mechanism |
|----------|----------|
| **ε-greedy** | With prob ε, show a random ad; otherwise greedy |
| **Thompson sampling** | Sample pCTR from the posterior, rank by sample |
| **UCB** | Score = pCTR + uncertainty bonus |
| **Random small fraction** | 1-5% of impressions are pure-random for unbiased training |

## 9.3 Industry Practice

Most platforms reserve **1-10% of impressions for exploration**. The cost is short-term revenue loss; the benefit is unbiased training data for the next model iteration.

---

# 10. Ads Experimentation Patterns

## 10.1 The Auction Interference Problem

If you A/B test a new pCTR model on users, treatment-user impressions go through the new model — but **the auction is shared with control-user impressions** in some surfaces. Treatment ads compete against control ads. The auction equilibrium shifts.

**Result:** measured effect overstates / understates true effect.

## 10.2 Solutions

### User-level A/B (when surface is well-isolated)
For surface-specific changes where one user's auction doesn't affect another's (e.g., personalized feeds with isolated auctions per user), user-level A/B works.

### Auction-level A/B (impression-level)
Randomize **each auction** independently. Half the auctions run the new model, half the old. The same user can be in both arms.

- **Pro:** clean comparison of model performance per auction
- **Con:** can't measure user-level downstream effects (next-day return)

### Cohort hold-back
Reserve a cohort of advertisers / campaigns to old ranker. Compare cohort-level metrics. Useful for long-term ROI experiments.

### Geo experiment
Standard cluster design — see `42_switchback_geo_cluster_designs.md`.

## 10.3 Counterfactual Evaluation (Offline)

Train a new model. Replay historical impressions through it. Compute would-be revenue using IPS.

$$
\hat{R}^{\text{new}} = \frac{1}{n} \sum_i w_i \cdot r_i \quad \text{where } w_i = \frac{P(\text{action by new}|i)}{P(\text{action by old}|i)}
$$

**Variance** grows when policies disagree strongly → use IPS clipping or self-normalized IPS.

## 10.4 Counterfactual Logging Hygiene

For IPS to work, you must log:
- Predicted pCTR by the **logging model** for each candidate
- Action probability under logging policy
- Outcome (clicked / converted)

Without these, counterfactual replay is impossible.

---

# 11. Ad Quality and User Welfare

## 11.1 Why Care About Quality

Short-term revenue maximization burns the user surface. Long-term DAU drops, advertisers lose access to engaged users, ecosystem dies.

## 11.2 Quality Signals

| Signal | What it captures |
|--------|------------------|
| **CTR-relative-to-position** | Whether users find the ad relevant |
| **Bounce rate** | Did they bounce immediately after click? |
| **Time on landing page** | Did they engage? |
| **Conversion rate** | Did they actually convert? |
| **Negative feedback** (hide, complain) | Did they actively dislike it? |
| **Repeat clicks** | Are users learning to avoid this ad? |

## 11.3 Quality in the Auction

Most platforms incorporate a **quality score**:

$$
\text{ad\_rank} = \text{bid} \cdot \hat{p}_{\text{CTR}} \cdot Q
$$

where $Q$ penalizes low-quality ads. Google's "Quality Score" is the famous example.

## 11.4 User-Side Experiments

For long-term welfare:
- Long-running 1% holdout that never sees ads (or sees minimal ads)
- Measure DAU, retention, complaint rate over months
- Comparing this to ads-receiving users tells you the **net welfare effect** of advertising at current intensity

This is one of the most strategically important measurements at any ad-supported business.

---

# 12. Interview Questions with Strong Answers

## Q1: "Explain how the ads auction works on Google or Meta."

> It's a generalized second-price (GSP) auction with quality adjustment. For each ad slot, every eligible advertiser has a bid (max willingness to pay per click). The platform computes an **ad rank** = bid × predicted CTR × quality score. The highest ad rank wins. The winner pays the minimum needed to beat the second-place ad rank — translated back to per-click terms. This gives advertisers an incentive to bid truthfully because they don't usually pay their full bid. The pCTR is the platform's prediction of how often users will click — so a less relevant ad with a high bid can lose to a more relevant ad with a lower bid if the pCTR difference is large enough. The quality score is a multiplier that penalizes low-quality ads beyond just CTR — covers landing page quality, ad text relevance, etc.

## Q2: "What's the difference between ranking quality and calibration in ads?"

> Ranking quality measures whether the model orders items correctly — AUC is the standard metric. Calibration measures whether the predicted probabilities match observed frequencies — among predictions of 0.1, do 10% of impressions get clicks? In organic ranking, only ranking matters; the absolute probability doesn't enter the decision. In ads, **calibration is critical** because the auction multiplies pCTR by bid to get ad rank, and uses that to set price. A model that's 2x too high in calibrated probability makes ads win twice as often as they should and overcharges advertisers. The diagnostic: divide impressions into pCTR bins, compute empirical CTR per bin, plot the reliability diagram. Fix with Platt scaling, isotonic regression, or post-hoc rescaling to the global average.

## Q3: "How would you debug a pCTR model that has good AUC but poor revenue performance?"

> Likely calibration. AUC measures rank order; revenue depends on absolute probability. Diagnostics: **(1)** plot reliability diagram — predictions vs observed CTR per bin. **(2)** check average pCTR vs empirical overall CTR — should match within 5%. **(3)** check calibration by segment — country, device, time of day. **(4)** check class imbalance and negative subsampling — many production training pipelines subsample negatives without correcting the resulting bias. **(5)** check feature distributions for serving vs training drift — if a feature shifted, the model's prior calibration breaks. **(6)** check position bias — if training data has position-dependent CTRs and serving puts ads in different positions, calibration is broken. Fix: post-hoc rescaling for simple cases, retrain with isotonic regression head, or fix the underlying drift.

## Q4: "What is position bias and how do you handle it in pCTR training?"

> Users click on higher-positioned items more, regardless of relevance — position 1 might get 25% CTR, position 5 might get 8%, just from position effect alone. If you train pCTR on raw click data, the model attributes that gap to relevance. Then it overpredicts CTR for ads that have historically been in top slots. Three handling approaches: **(1)** model click as `P(examined | position) × P(click | examined, relevance)` — the cascade or position-bias model. Train both factors jointly. **(2)** result randomization — periodically shuffle the result list to learn the position effect under controlled conditions. **(3)** inverse-propensity scoring — weight each impression by 1 / P(this ad got this position | logging policy), giving unbiased relevance estimates. Most modern platforms use a hybrid: position as an input feature during training, ignored at serving (or set to a canonical position).

## Q5: "Walk me through a multi-objective ads ranker."

> Linear combination of objectives with weights tuned via long-term experiments. The score is `α · eVPI + β · quality + γ · relevance − δ · annoyance`. eVPI is expected value per impression (revenue proxy). Quality and relevance are platform-defined; annoyance might be hide rate or complaint rate. Weights are not arbitrary — they're chosen so that pushing the weight slider trades off short-term revenue against long-term user satisfaction in a known way. The right way to tune them is **long-term holdout experiments**: 1% of users see no ads, 1% see aggressive ads, 1% see balanced; measure 90-day retention. The Pareto frontier of revenue-vs-satisfaction shows the operating points; leadership picks based on strategic priorities (growth phase → more user-favoring; profitability phase → more revenue-favoring). I would never recommend optimizing only revenue, because the long-term user costs are real and lag the short-term gains.

## Q6: "How would you experiment with a new ads ranking model?"

> Multiple layers. **(1)** Offline counterfactual evaluation with IPS on logged data — measures expected revenue if we replaced the old model with the new one, weighted by inverse propensity. Cheap, fast, but noisy. **(2)** Auction-level A/B — randomize at the impression level, half through new model, half through old. Clean comparison of per-auction revenue but ignores user-level downstream effects. **(3)** User-level A/B — randomize users to new or old ranker. Captures user-level effects (return rate, complaints) but suffers from auction interference if treatment and control compete in the same auctions. **(4)** Geo or surface-level holdout for long-term welfare measurement. The right move is **(2) then (3) then (4)** in sequence: cheap offline, then small auction-level, then user-level, then geo/long-term. Each stage screens out risky ideas before the next more expensive stage.

## Q7: "How does budget pacing work?"

> The advertiser sets a daily budget. The platform needs to spread spend across the day to capture the highest-value impressions, not exhaust budget by 10 AM. Mechanism: a **pacing multiplier** `ρ ∈ [0, 1]` is applied to the advertiser's bid in real time. If the campaign is ahead of pace (spending too fast), `ρ` decreases — the advertiser bids less aggressively, wins fewer auctions, and slows down. If behind, `ρ` increases. The target spend curve over the day is determined by historical hourly distributions of valuable traffic; the controller looks like a PID loop in practice. Edge cases: campaigns with very high vs very low budgets behave differently; new campaigns with no history need cold-start defaults. The KPI for pacing quality is the **delivery uniformity**: how close does the achieved spend curve match the target spend curve. Poor pacing harms advertiser ROI even if total spend matches.

## Q8: "What is IPS and when do you use it?"

> Inverse Propensity Scoring is a counterfactual evaluation technique. Suppose the production system shows ad A with probability `p(A | impression)`. You want to know what would have happened if you'd used a new system that would have shown ad A' instead. IPS estimates the new system's expected reward by reweighting historical impressions: $\hat{R}^{\text{new}} = \sum_i \frac{P_{\text{new}}(a_i | x_i)}{P_{\text{old}}(a_i | x_i)} r_i$. The ratio of new vs old action probabilities is the "propensity weight." It's unbiased if both systems have non-zero probability on every action. Limitations: high variance when the two systems disagree strongly; can be stabilized with clipping (cap weights at some max) or self-normalization (divide by sum of weights). IPS is the standard tool for offline policy evaluation in ads, used to screen new rankers before live tests.

## Q9: "How do you handle exploration in ads ranking?"

> Production rankers are greedy on best-known pCTR, which means new ads with no history get under-ranked and stay under-ranked — a death spiral. To break this, allocate a small fraction of traffic (typically 1-5%) to exploration. Strategies: **(1)** ε-greedy — random ad with probability ε. Simple but wasteful. **(2)** Thompson sampling — for each candidate, draw pCTR from the posterior and rank by sample. Naturally balances exploration and exploitation; new ads have wider posteriors so get more exploration. **(3)** UCB — score = pCTR + uncertainty bonus. Similar effect, deterministic. **(4)** Reserved exploration slot — every result page has one slot dedicated to a less-shown ad. The trade-off is short-term revenue loss vs long-term unbiased training data and a healthier ad ecosystem (new advertisers get a chance). Most platforms allocate 1-3% to pure exploration plus Thompson sampling in the main ranker.

## Q10: "What's the most common bug you've seen in ads experiments?"

> Auction interference. A team A/B tests a new ranker by randomizing users. Treatment users go through the new ranker, control users through the old. Both populate the same shared advertiser supply. In the same auction window, treatment-user impressions compete with control-user impressions for the same advertisers' budgets. If the new ranker is "smarter" and steers users to high-value ads, treatment exhausts ad budgets faster, leaving control users to see cheaper / lower-quality ads. The measured user-level effect is then partly the model's effect and partly the budget-displacement effect. The fix: **auction-level randomization** if the auction can be isolated, or **advertiser-level holdouts** if budget is the shared resource. I'd document this explicitly in the experiment proposal and the platform should refuse to publish results without addressing it.

---

# 13. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│        ADS RANKING & AUCTIONS — INTERVIEW TAKEAWAYS                 │
│                                                                     │
│  1. Ads ≠ organic. Three customers (user, advertiser, platform).  │
│                                                                     │
│  2. GSP auction: ad_rank = bid × pCTR × quality. Winner pays 2nd  │
│     place's rank divided by their own pCTR.                       │
│                                                                     │
│  3. pCTR is THE workhorse model. Billions of impressions, lots of │
│     features, online refresh.                                      │
│                                                                     │
│  4. CALIBRATION matters more than ranking for ads. Misc-alibrated │
│     pCTR breaks auction prices.                                   │
│                                                                     │
│  5. Expected value per impression (eVPI) is the auction's currency.│
│     All bid types convert to eVPI.                                 │
│                                                                     │
│  6. Multi-objective: revenue + user satisfaction + advertiser     │
│     value, tuned via long-term holdouts.                          │
│                                                                     │
│  7. Position bias contaminates training data. Cascade models /    │
│     result randomization / IPS to handle.                          │
│                                                                     │
│  8. Exploration is required to discover new good ads. 1-5% of     │
│     impressions reserved for exploration.                          │
│                                                                     │
│  9. Auction interference is the BIG experimentation pitfall.      │
│     Auction-level randomization or geo when needed.                │
│                                                                     │
│  10. Long-term holdouts measure ad welfare effect. Critical for   │
│      ecosystem health, not just short-term revenue.                │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"Ads ranking is the only system where the model's calibration directly determines price. Get pCTR right, get the auction right, and balance short-term revenue with long-term user welfare via long-running holdouts. Everything else is engineering on top of those four invariants."*

---

**Further reading**

- Edelman, Ostrovsky, Schwarz (2007) — *Internet Advertising and the Generalized Second-Price Auction*
- McMahan et al. (2013) — *Ad Click Prediction: a View from the Trenches* (Google)
- Cheng et al. (2016) — *Wide & Deep Learning for Recommender Systems*
- Chapelle & Manavoglu (2014) — *Simple and Scalable Response Prediction for Display Advertising*
- Joachims et al. (2017) — *Unbiased Learning-to-Rank with Biased Feedback*
- Tagami et al. (2013) — *CTR Prediction for Contextual Advertising*
- Athey & Imbens — *The Economics of Auctions* (working paper survey)
