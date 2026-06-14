# Marketplace Science — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Two-Sided Marketplaces, Liquidity, Matching, Take Rate, Network Effects

> **Context:** `37_product_sense_and_business_metrics.md` lists marketplace metrics at one line each. This is the **deep dive** that the JD asks for at companies like Airbnb, Uber, DoorDash, Etsy, Amazon, eBay, Lyft, Instacart, Doordash, Booking, Vinted, OpenAI (marketplace of API customers + model providers). It covers two-sided dynamics, liquidity, search congestion, thickness, cold start on both sides, matching, commission/take-rate trade-offs, and the experimentation patterns unique to marketplaces.

---

## Table of Contents

1. [What Makes a Marketplace Different](#1-what-makes-a-marketplace-different)
2. [Two-Sided Network Effects](#2-two-sided-network-effects)
3. [Liquidity, Thickness, and Density](#3-liquidity-thickness-and-density)
4. [Search and Matching](#4-search-and-matching)
5. [Cold Start on Both Sides](#5-cold-start-on-both-sides)
6. [Take Rate and Commission Economics](#6-take-rate-and-commission-economics)
7. [Pricing in Marketplaces](#7-pricing-in-marketplaces)
8. [Marketplace Health Metrics](#8-marketplace-health-metrics)
9. [Experimentation in Marketplaces](#9-experimentation-in-marketplaces)
10. [Reputation, Trust, and Selection](#10-reputation-trust-and-selection)
11. [Common Marketplace Failure Modes](#11-common-marketplace-failure-modes)
12. [Interview Questions with Strong Answers](#12-interview-questions-with-strong-answers)
13. [Key Takeaways](#13-key-takeaways)

---

# 1. What Makes a Marketplace Different

## 1.1 Definition

A **marketplace** is a platform that matches two (or more) groups of participants — usually a **demand side** (buyers, riders, guests) and a **supply side** (sellers, drivers, hosts). The platform doesn't typically own the inventory; it facilitates transactions.

| Marketplace | Demand | Supply | What's traded |
|------------|--------|--------|--------------|
| Uber | Riders | Drivers | Rides |
| Airbnb | Guests | Hosts | Nights |
| DoorDash | Eaters | Restaurants + Dashers | Meals |
| Etsy | Buyers | Sellers | Handmade goods |
| eBay | Buyers | Sellers | Long-tail goods |
| Amazon (3P) | Buyers | Third-party sellers | Goods |
| OpenAI API | API consumers | Model providers (1P + 3P) | Tokens / compute |

## 1.2 Why Marketplaces Are Hard

A marketplace has **two products** — one for each side — and **two growth flywheels** that depend on each other. You can't optimize one side in isolation. The discipline that handles this is called **marketplace science**.

```
┌─────────────────────────────────────────────────────────────────┐
│              THE MARKETPLACE FLYWHEEL                            │
│                                                                  │
│     More Buyers ──────▶ More Demand ──────▶ Better $/Seller    │
│         ▲                                          │            │
│         │                                          ▼            │
│   Better Selection                          More Sellers        │
│   & Lower Prices ◀───────  Inventory  ◀──── Join Platform      │
│                                                                  │
│   (Cross-side network effects: each side's value grows with    │
│   the other side's size)                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. Two-Sided Network Effects

## 2.1 Cross-Side vs Same-Side

| Type | Direction | Example |
|------|-----------|---------|
| **Cross-side, positive** | More of side A → side B happier | More riders → more rides for drivers |
| **Cross-side, negative** | More of side A → side B unhappier | More sellers → harder for sellers to stand out |
| **Same-side, positive** | More of side A → side A happier | More users → more network density |
| **Same-side, negative** | More of side A → side A unhappier | More drivers → less revenue per driver |

> **Insight:** Most healthy marketplaces have **positive cross-side** and **negative same-side** network effects. Drivers love riders but compete with each other.

## 2.2 The Chicken-and-Egg Problem

A marketplace with zero sellers has no buyers and vice versa. Strategies:

| Strategy | Example |
|----------|---------|
| **Seed one side aggressively** | Subsidize drivers in launch cities (Uber early days) |
| **Single-side product first** | OpenTable started as a restaurant booking tool, then added consumers |
| **Niche then expand** | Etsy started with handmade goods only |
| **Tap existing supply** | Airbnb leveraged existing rental properties |
| **Demand aggregation** | DoorDash leveraged existing restaurant supply |

---

# 3. Liquidity, Thickness, and Density

## 3.1 Liquidity

The probability that a buyer who arrives can complete a transaction quickly.

$$
\text{Liquidity} = P(\text{transaction} \mid \text{intent})
$$

| Marketplace | Liquidity proxy |
|------------|----------------|
| Uber | % requests matched within 5 min |
| Airbnb | % searches that lead to a booking |
| Etsy | % buyer sessions with a purchase |
| DoorDash | % orders accepted by a restaurant within 1 min |

Low liquidity = buyers leave. High liquidity = transactions happen quickly, both sides happy.

## 3.2 Thickness

The **size** of the market — how many active participants on each side.

- **Thick market**: many buyers and sellers; matching is easy.
- **Thin market**: few participants; long search times, frequent mismatches.

For dense urban markets (NYC, SF), Uber is thick. For sparse rural markets, Uber is thin → long waits, sometimes no ride.

## 3.3 Density

Combining thickness with **proximity** — participants must not just exist but be **near each other** in time, space, or category.

| Marketplace | Density dimension |
|------------|-------------------|
| Ride-sharing | Space × time |
| Airbnb | Travel dates × location |
| Job marketplace | Skill × location |
| Online auctions | Active listings × buyer attention |

> **Lesson:** "1 million sellers nationally" is meaningless if no city has more than 5. Marketplaces succeed locally before they succeed globally.

## 3.4 The Quantitative Measures

### Match rate
$$
\text{Match rate} = \frac{\text{successful matches}}{\text{intents (searches, requests)}}
$$

### Fill rate (for ad / dispatch markets)
$$
\text{Fill rate} = \frac{\text{requests served}}{\text{requests received}}
$$

### Time to fill
Median time between an intent and a transaction.

### Search abandonment
% of searches that end without a click/contact/purchase. Inverse of liquidity.

---

# 4. Search and Matching

## 4.1 The Core Problem

A buyer wants to find the best seller (or any acceptable seller, fast). The marketplace must:
1. **Index** the supply (which sellers exist, with what attributes).
2. **Retrieve** candidates given a query.
3. **Rank** them by predicted value.
4. **Match** buyer to seller (or let buyer choose).

This is **search and matching** — the central technical problem of a marketplace.

## 4.2 Search Congestion

When many buyers chase the same supply (top-rated listings, top-rated drivers), the supply gets **congested**. Some buyers get matched, others wait.

| Symptom | Diagnostic |
|---------|-----------|
| Top sellers booked solid weeks ahead | Supply concentration index |
| Buyers complaining about availability | Search-to-match rate dropping for popular categories |
| New sellers can't get bookings | Long-tail visibility metrics |

## 4.3 Diversification in Ranking

Pure greedy ranking (always show top seller) leads to:
- **Winner-take-all** dynamics
- **Rich get richer** feedback loop
- **No supply growth incentives**

**Mitigations:**
- **Position bias correction** in ranking
- **Exploration boost** for new listings (10% exploration traffic to long tail)
- **Diversity constraint** in the result page (no more than k items from same seller/category)
- **Geographic balancing** (don't only show downtown options)

## 4.4 Matching Mechanisms

| Mechanism | Example | Notes |
|-----------|---------|-------|
| **Search & select** | Airbnb, Etsy | Buyer browses and picks |
| **Algorithmic match** | Uber dispatch | Platform decides match |
| **Bid / auction** | eBay, ad auctions | Price discovery via competition |
| **Reverse auction** | Upwork, Catch (Australia) | Sellers bid for buyer's job |
| **Hybrid** | DoorDash (algorithm + dasher choice) | Platform proposes; supply accepts/declines |

## 4.5 Two-Sided Ranking

Modern marketplaces rank with **both sides' utility in mind**:

$$
\text{Score} = \alpha \cdot P(\text{buyer clicks}) + \beta \cdot P(\text{seller accepts}) + \gamma \cdot P(\text{successful transaction})
$$

Optimizing only buyer-side CTR can lead to ranking listings that look attractive but never convert (low seller acceptance, cancellations, refunds).

---

# 5. Cold Start on Both Sides

## 5.1 The Two Cold Starts

| Side | Cold start problem | Mitigation |
|------|--------------------|------------|
| **Demand cold start** | New buyer, no history → unclear what they want | Use signup signals, popular items, broader exploration |
| **Supply cold start** | New seller, no history → unclear quality | Trust signals (verification, photos), boost in ranking, low-volume signals |

## 5.2 Why Marketplace Cold Start Is Worse Than RecSys

In RecSys (Netflix), cold start is a one-sided problem — recommend to new users. In a marketplace:
- A new seller earns no money if no buyers can find them.
- They quit before generating enough data to be ranked.
- This is a **death spiral**: new supply churns before becoming established.

## 5.3 Supply-Side Cold Start Strategies

| Strategy | Mechanism |
|----------|----------|
| **Trust scaffolding** | ID verification, badge, photo verification |
| **Listing assistance** | AI-generated descriptions, suggested prices |
| **Initial boost** | Exploration traffic, "new" badge for prominence |
| **Bootstrap signals** | Use seller's external reviews (eBay → linked PayPal feedback) |
| **Bayesian smoothing** | Treat new sellers as having a category-average rating until N reviews accumulate |

## 5.4 The Bayesian Average for Ratings

For a seller with $n$ ratings averaging $\bar{r}$, the shrunken score is:

$$
r_{\text{shrunken}} = \frac{n \cdot \bar{r} + k \cdot \mu}{n + k}
$$

where $\mu$ is the prior mean (e.g., overall marketplace average) and $k$ is the strength of the prior (e.g., 10 ratings).

This prevents a 5-star seller with 1 review from outranking a 4.8-star seller with 100 reviews.

---

# 6. Take Rate and Commission Economics

## 6.1 What Is Take Rate

The fraction of GMV (Gross Merchandise Value) the platform captures.

$$
\text{Take rate} = \frac{\text{Platform revenue}}{\text{GMV}}
$$

| Marketplace | Approx take rate |
|------------|------------------|
| eBay | ~12% |
| Airbnb | ~14% |
| DoorDash | ~20-30% (incl. delivery fees) |
| Uber | ~25% |
| Etsy | ~6.5% + fees |
| App Store | 15-30% |
| Amazon 3P | ~15% + FBA fees |

## 6.2 The Take Rate Trade-Off

Higher take rate → more revenue per transaction.
But also → less seller margin → sellers leave OR raise prices → buyers leave OR fewer transactions.

The **profit-maximizing take rate** depends on the **price elasticity of both sides**.

## 6.3 Elasticity-Based Pricing

If buyer side is more elastic (sensitive to price), pass costs to seller. If seller side is more elastic, pass to buyer.

In practice, platforms charge **both sides** (Uber service fee + driver commission) and tune the split.

## 6.4 Why Take Rate Matters for Experimentation

Take rate changes are **massive experiments**: they affect:
- Seller supply (some leave at higher commission)
- Listing prices (sellers raise prices, partially passing through)
- Buyer conversion
- GMV
- Revenue

You can rarely A/B test a take-rate change at user level (interference) — you need **geo experiments** or **synthetic control** when policy applies platform-wide.

---

# 7. Pricing in Marketplaces

## 7.1 Pricing Roles

| Mechanism | Who sets price | Example |
|-----------|---------------|--------|
| **Platform-set** | The platform | Uber surge, Lyft pricing |
| **Seller-set** | Sellers | Airbnb, Etsy, eBay |
| **Auction** | Bidders | eBay auctions, Google Ads |
| **Algorithmic + seller acceptance** | Platform suggests, seller agrees | Amazon's automated repricing |

## 7.2 Surge / Dynamic Pricing

When demand spikes (rush hour, snowstorm), prices rise. Goals:
- **Ration demand** (reduce excess searches)
- **Recruit supply** (drivers come out for higher fares)
- **Maintain liquidity** (reduce wait times to acceptable)

The math: demand elasticity × supply elasticity → equilibrium price function.

## 7.3 Smart Pricing (Airbnb)

Algorithmically suggest prices to hosts based on:
- Local market demand
- Listing features
- Calendar gaps
- Competition
- Lead time to booking

**Lift:** hosts who accept smart pricing see ~5-15% revenue lift on average. But accepting is a choice → endogenous → real lift estimation requires causal inference.

## 7.4 Price Experimentation

Pricing is high-stakes and hard to test:
- Buyer-side price elasticity is sharp.
- Same-user inconsistency erodes trust (showing different prices to same user).
- Spillovers — control buyers can see treatment-side prices in shared listings.

Typical approach: **geo experiment** for systemic pricing changes; **personalization within strict bounds** for individual offers.

---

# 8. Marketplace Health Metrics

## 8.1 The Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│              MARKETPLACE METRIC HIERARCHY                        │
│                                                                  │
│  NORTH STAR                                                     │
│  └── GMV / completed transactions / nights booked              │
│                                                                  │
│  PRIMARY METRICS                                                 │
│  ├── Liquidity (match rate, time-to-fill)                      │
│  ├── Take rate × GMV = revenue                                  │
│  └── DAU / weekly active per side                              │
│                                                                  │
│  HEALTH METRICS                                                  │
│  ├── New seller retention (D0→D30→D90)                         │
│  ├── New buyer first-transaction rate                          │
│  ├── Seller concentration (Herfindahl on GMV)                  │
│  ├── Geographic spread                                         │
│  └── Repeat purchase rate                                      │
│                                                                  │
│  GUARDRAILS                                                      │
│  ├── Buyer NPS / satisfaction                                   │
│  ├── Seller NPS / earnings stability                           │
│  ├── Refund / cancellation rate                                │
│  └── Trust & safety incidents per 1000 transactions            │
└─────────────────────────────────────────────────────────────────┘
```

## 8.2 The Single Most Important Health Metric

For most marketplaces: **repeat transaction rate**. It captures whether the experience was good enough that participants come back. It's the leading indicator of every long-term metric.

## 8.3 Concentration Metrics

The **Herfindahl-Hirschman Index (HHI)** of GMV by seller:

$$
\text{HHI} = \sum_i s_i^2
$$

where $s_i$ is seller i's share of GMV. HHI near 0 = competitive; HHI near 1 = monopolistic.

Healthy marketplaces have HHI < 0.1 in any category. Concentration is a leading indicator of supply churn ("only top 5 sellers earn anything").

## 8.4 Cohort-Based Metrics

Marketplace effects play out over **months**, not days. Standard cohort views:

| Cohort metric | Definition |
|--------------|-----------|
| **D30 buyer retention** | % of buyers who bought again 30 days after first purchase |
| **D90 seller retention** | % of new sellers still active 90 days post-listing |
| **LTV by cohort** | Lifetime value of buyers acquired in month M |
| **Time-to-second-transaction** | How fast new buyers come back |

---

# 9. Experimentation in Marketplaces

## 9.1 The Big Problem

Marketplaces violate SUTVA. User-level A/B tests **systematically over-estimate** treatment effects because treatment competes with control for the same supply.

> See `42_switchback_geo_cluster_designs.md` for full coverage.

## 9.2 Patterns That Work

| Pattern | Use |
|---------|-----|
| **Geo experiment** | Pricing, dispatch, search ranking, fee changes |
| **Switchback** | Real-time dispatch, surge, auction parameters |
| **Two-sided cluster** | When supply also reacts (driver-side feature) |
| **Synthetic control** | Single-geo pilot or policy rollout |
| **User-level A/B** | UI changes, search filter UX (when supply effect is small) |

## 9.3 Long-Term Holdouts

A small slice (e.g., 1%) of one side never gets the new feature, for months. Used to measure:
- **Cumulative impact** beyond novelty
- **Cohort effects** (do new users behave differently long-term?)
- **Decay** of treatment effect

## 9.4 Surrogate Metrics

You can't wait 90 days for D90 retention to ship a feature. Use **short-term surrogates** that historically predict long-term outcomes:

| Surrogate | Predicts |
|-----------|----------|
| Week-1 engagement | D30 retention |
| First-week orders | LTV |
| First-listing photos & description quality | Seller D90 retention |
| First-search-to-message latency | Buyer conversion |

Validate surrogate-outcome relationships periodically; don't let them drift.

---

# 10. Reputation, Trust, and Selection

## 10.1 The Lemon Problem

If buyers can't observe quality before transacting, **adverse selection** drives quality down: good sellers can't differentiate and exit, leaving only bad sellers (Akerlof's lemons).

## 10.2 Solutions

| Solution | Mechanism |
|----------|----------|
| **Ratings & reviews** | Crowd-sourced quality signal |
| **Verification badges** | ID checks, business verification |
| **Insurance / guarantees** | Platform underwrites the buyer's risk |
| **Money-back policies** | Buyer protection |
| **Curation** | Platform pre-screens sellers (e.g., 1stDibs) |

## 10.3 Reputation Metrics

| Metric | Issue | Fix |
|--------|------|-----|
| Average rating | Sparsity (new sellers have noisy averages) | Bayesian smoothing |
| Number of reviews | Doesn't reflect quality, just volume | Combine with rating |
| Response rate | Game-able (sellers respond fast but unhelpfully) | Weight by helpfulness signal |
| Cancellation rate | Sellers cancel hard bookings | Penalize cancellations more than no-shows |

## 10.4 Review Bias

- **Selection bias:** only motivated users leave reviews → tail bias (very happy or very angry).
- **Reciprocity bias:** sellers rate buyers → both inflate to avoid retaliation.
- **Anchoring:** users see existing reviews before writing → herd toward consensus.

Solutions: blind reviews (Airbnb hides ratings until both sides post), specific scenario-based questions instead of star ratings, weighted aggregation by reviewer experience.

---

# 11. Common Marketplace Failure Modes

## 11.1 Disintermediation

Buyers and sellers transact off-platform to avoid fees. Symptoms: many messages, few transactions. Defenses:
- **Hold escrow** until completion
- **Hide contact info** before booking
- **Build platform-only value** (insurance, ratings)

## 11.2 Fake / Spam Supply

Spammers list fake inventory to capture leads. Defenses: verification, anomaly detection, listing quality scores.

## 11.3 Concentration Death Spiral

Top 1% of sellers capture 80% of GMV → new sellers churn → only top 1% remains → buyers find limited selection → buyers leave.

Defenses: exploration in ranking, ranking diversity, supply-side experimentation.

## 11.4 Negative Cross-Side Externalities

Sometimes more supply hurts demand (e.g., too many cheap, low-quality sellers crowd out quality and erode trust). Trade-off: supply growth vs supply quality.

## 11.5 The Geographic Trap

Marketplaces succeed locally first. Trying to expand "horizontally" before any city is dense enough is a known failure pattern (lots of failed delivery startups).

---

# 12. Interview Questions with Strong Answers

## Q1: "What's the difference between liquidity and thickness?"

> **Thickness** is the raw scale — how many participants on each side. **Liquidity** is the probability that an arriving participant completes a transaction. You can have a thick market with poor liquidity (lots of listings but mismatched to demand) or a thin market with high liquidity (small but tight matches). For a marketplace operator, liquidity is the more actionable metric because it captures whether the system is *working*, not just how many people use it. Thickness is necessary for liquidity — a market of 5 listings can't be liquid in any meaningful sense — but it's not sufficient. The job of marketplace science is to grow thickness while simultaneously improving the matching that converts thickness into liquidity.

## Q2: "How would you design an experiment to test a new commission structure?"

> Commission changes are platform-wide policy changes that violate SUTVA — every seller's pricing decision affects every other seller's. User-level A/B is biased. The right design is a **geo experiment**: pick a set of matched geo pairs, randomize one of each pair to the new commission. Run for at least 4 weeks to capture supply response (sellers raise prices, some exit). Analyze GMV, seller retention, buyer conversion, and net revenue at the geo level with cluster-robust SE. Pre-period parallel trends matter — pair the geos on behavioral history, not just size. If the geos are few (typical for matched-pair design), use wild cluster bootstrap for inference. I'd also pair this with a **structural simulation** of seller pricing response to verify the experiment is consistent with a sensible equilibrium model. Decision: ship only if both geo evidence and structural model agree the long-run effect on net revenue is positive without harming seller retention.

## Q3: "Walk me through the cold start problem on the supply side."

> A new seller has no history, no ratings, no signals to rank on. If the ranker uses past performance, the new seller is buried, gets no traffic, never accumulates the data needed to rank well, and churns before becoming established. This is a death spiral that quietly kills supply growth. Mitigations: **trust scaffolding** (verification badges so buyers don't need ratings), **listing assistance** (AI-generated descriptions, suggested prices to reduce listing friction), **initial exploration boost** (allocate 5-10% of impressions to new sellers regardless of predicted CTR), **Bayesian smoothing** of ratings (shrink toward the prior, so a new 5-star seller doesn't outrank an established 4.7), and **bootstrap signals** from external reputation (eBay used PayPal feedback). The platform should track new-seller D90 retention as a primary health metric; if it drops, supply growth will break before you notice in GMV.

## Q4: "What is search congestion and how do you fix it?"

> Search congestion happens when many buyers chase the same supply — typically the top-rated, most popular listings — and that supply runs out (booked solid, drivers all busy). The symptom is that buyers see "unavailable" or wait long times; sellers in the long tail get no traffic. Diagnostic metrics: supply concentration (HHI of GMV by seller), top-1% share of impressions, search-to-match conversion in popular categories. Fixes: **position-bias-corrected ranking** so top sellers don't get every impression by default; **diversity constraints** (no more than 3 items per seller on the first page); **exploration boost** for new sellers; **redirect-to-alternative** if top choices are unavailable; and **demand shaping** through pricing — surge in supply-constrained slots pushes some buyers to alternative times or sellers.

## Q5: "How would you measure the value of a new feature for sellers?"

> Multiple lenses. **Direct usage** — what % of eligible sellers adopt the feature? If <10%, the feature is irrelevant. **Within-seller lift** — for sellers who adopt, what's their GMV / earnings change compared to a holdout of non-adopters with similar pre-period? Use propensity-score matching or instrumental variables (if adoption is endogenous, e.g., motivated sellers adopt first). **Retention impact** — does the feature reduce churn for adopters vs matched non-adopters at D30 and D90? **Marketplace-level GMV** — geo-experiment the feature rollout to see if total platform GMV moves, not just within-seller. **Buyer satisfaction** — does the feature change the listings buyers see and their satisfaction with them? The seller side is usually noisy; expect long experiment durations and combine multiple methods rather than relying on one.

## Q6: "Sellers report earnings dropped after a search ranking change. How do you investigate?"

> Walk down the dependency. **First, verify the data** — is the ranking change actually live, and for what fraction of sessions? Pull config and exposure logs. **Second, slice by impression volume** — did high-volume sellers gain at the expense of low-volume? That would indicate ranking shifted weight toward top sellers (concentration). **Third, segment by category** — is the effect uniform or concentrated? **Fourth, check supply elasticity** — did sellers stop listing or change prices? A ranking change can cause supply-side response that shows up in next-period earnings, not the current period. **Fifth, look at the buyer-side metrics** — if conversion rate increased but earnings dropped, the new ranker is showing cheaper listings; net GMV may be flat or up while per-seller earnings shifted. Document the diagnostic, propose a fix (rebalance ranking to value smaller sellers), and run a new experiment.

## Q7: "What's the right unit of randomization for a ride-sharing dispatch experiment?"

> Not the rider. Riders interact through shared supply — treatment riders steal drivers from control riders, biasing the estimate. Not the driver either, because the new dispatch is for everyone in the area at once. The right unit is **time × geo**: switchback experiments where the entire city operates on one variant for ~1 hour, then switches. Within a slot, everyone sees the same dispatch, so no within-slot interference. Across slots, you have variation between treatment and control. Slot length tuned to longer than typical carryover (5-15 min washout). Analyze with slot as the unit, fixed effects for geo and time-of-day, cluster-robust SE at the slot level. The estimand is the policy-level treatment effect, not the individual-rider effect.

## Q8: "When does take rate go up vs down?"

> Take rate moves with elasticity and competitive pressure. **Goes up when:** the platform has dominant market share (less competitive constraint); the platform offers high-value services beyond matching (insurance, payments, logistics); seller alternatives are weak. **Goes down when:** competitors offer lower take; supply elasticity is high (sellers leave at higher rates); regulatory pressure (EU app store action); strategic platform expansion phase (subsidize to grow). At Uber, take rate has fluctuated 22-30% depending on driver supply pressure. At Etsy, fee structure changes regularly trigger seller protests because crafts sellers are elastic. The platform DS team's job is to estimate marketplace-wide elasticity from geo experiments and inform pricing strategy quantitatively.

## Q9: "How do you handle network effects when running experiments?"

> Three patterns depending on the network. **Bounded network (geographic):** use geo experiments — the network is contained within a city, so cross-city interference is negligible. **Time-bounded network (real-time matching):** use switchback experiments where the entire system is on one variant per slot. **Social network (user-graph):** use **ego-network clustering** — find tight subgraphs where treatment effects are bounded by community boundaries. The general principle: identify the **resource or interaction channel** that creates interference, and randomize at a granularity that contains it. If the channel can't be contained (truly global network effects), you may need to accept biased experiments and triangulate with observational evidence or structural models. The platform should be explicit about which experiments are SUTVA-safe and which aren't, and route hard cases to special analysis.

## Q10: "What's the most important health metric for a marketplace and why?"

> Repeat transaction rate, on both sides. Specifically: **D30 buyer retention** (% of new buyers who transact again within 30 days of first transaction) and **D90 seller retention** (% of new sellers still active 90 days post-listing). These are the leading indicators of *every* downstream metric — GMV, revenue, LTV — and they capture whether the marketplace is genuinely working for participants, not just acquiring them. GMV can grow with bad retention (high acquisition spend) for a while, but it's not sustainable. If retention is healthy, growth compounds; if it's not, the marketplace is leaking and any feature lifts you measure today won't survive long-term. So I'd watch retention as the north-star health metric and use GMV / liquidity / revenue as the operating dashboards.

---

# 13. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│         MARKETPLACE SCIENCE — INTERVIEW TAKEAWAYS                  │
│                                                                     │
│  1. Marketplaces are TWO products, ONE platform. Optimize both    │
│     sides; never one in isolation.                                 │
│                                                                     │
│  2. Cross-side network effects = the flywheel.                    │
│                                                                     │
│  3. Liquidity (P(transaction|intent)) > thickness (raw scale).    │
│     Density is local.                                              │
│                                                                     │
│  4. Cold start is two problems. Supply cold start kills supply    │
│     growth and is invisible until churn shows up months later.    │
│                                                                     │
│  5. Take rate = elasticity battle. Higher take, less seller        │
│     margin, fewer sellers, fewer buyers.                           │
│                                                                     │
│  6. SUTVA breaks in marketplaces. Geo and switchback are the      │
│     experiment designs that work.                                  │
│                                                                     │
│  7. Search congestion → concentration → death spiral. Diversify  │
│     ranking, explore long tail.                                    │
│                                                                     │
│  8. Surrogate metrics + long-term holdouts together capture both  │
│     speed and durability.                                          │
│                                                                     │
│  9. Reputation is the trust mechanism. Bayesian smoothing for     │
│     small samples; blind reviews to reduce bias.                   │
│                                                                     │
│  10. Repeat transaction rate is the single most important         │
│      marketplace health metric.                                    │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"Marketplaces succeed by growing liquidity faster than congestion, by solving cold-start on both sides simultaneously, and by experimenting with the right unit of randomization. Most marketplace failures look like growth problems on the surface — they're really matching problems underneath."*

---

**Further reading**

- Hagiu & Wright — *Multi-Sided Platforms* (Strategic Management Journal)
- Rochet & Tirole — *Platform Competition in Two-Sided Markets*
- Andrew Chen — *The Cold Start Problem* (book)
- Sangeet Paul Choudary — *Platform Revolution*
- Hagiu — *Strategic Decisions for Multisided Platforms*
- Tang, Hooi, Liang, Bakshy — papers on geo experiments
- Airbnb Engineering blog — multiple posts on marketplace ranking, host onboarding
