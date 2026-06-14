# Experimentation Platforms — Systems & Architecture Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / Platform Engineer — Experimentation Systems, Assignment, Metric Pipelines, Lifecycle, Governance

> **Context:** Your file `24_statistics_and_ab_testing.md` is methodology-heavy. This guide is the **systems** counterpart — the architecture, APIs, data models, and lifecycle that make A/B testing actually trustworthy at scale. This is what Meta, Microsoft (ExP), LinkedIn (XLNT), Netflix (XP), Airbnb (ERF), Uber (Morpheus), and OpenAI's experimentation team build. It is also exactly what the OpenAI experimentation JD is asking for.

---

## Table of Contents

1. [What Is an Experimentation Platform?](#1-what-is-an-experimentation-platform)
2. [The Five Subsystems](#2-the-five-subsystems)
3. [Assignment Service — Deterministic Bucketing](#3-assignment-service)
4. [Configuration & Namespaces](#4-configuration--namespaces)
5. [Exposure Logging](#5-exposure-logging)
6. [Metric Definitions & the Metric Store](#6-metric-definitions--the-metric-store)
7. [Analysis Pipeline](#7-analysis-pipeline)
8. [Experiment Lifecycle](#8-experiment-lifecycle)
9. [Triggered Analysis](#9-triggered-analysis)
10. [Scaling to Thousands of Concurrent Experiments](#10-scaling-to-thousands-of-concurrent-experiments)
11. [Governance, Trust, and Best Practices](#11-governance-trust-and-best-practices)
12. [Reference Architectures (Industry)](#12-reference-architectures-industry)
13. [Interview Questions with Strong Answers](#13-interview-questions-with-strong-answers)
14. [Key Takeaways](#14-key-takeaways)

---

# 1. What Is an Experimentation Platform?

## 1.1 Definition

An **experimentation platform** is the engineering infrastructure that lets a company:

1. **Define** an experiment (what's changing, who is eligible, which metrics matter)
2. **Assign** users (or sessions, devices, etc.) to variants deterministically
3. **Expose** them to the right variant via the product
4. **Log** the exposure and downstream events
5. **Compute** metrics from logs
6. **Analyze** with rigorous statistics
7. **Decide** whether to ship, kill, or iterate

> An experimentation platform is **not** a t-test. The t-test is 0.1% of the work. The other 99.9% is data engineering, governance, and trust.

## 1.2 Why a Platform (Instead of Ad-Hoc Tests)

| Problem with ad-hoc | What a platform solves |
|--------------------|------------------------|
| Each team writes its own assignment code | One deterministic assignment service |
| SQL queries differ per analyst | Centralized, versioned metric definitions |
| Results not reproducible | Frozen analysis specs + audit log |
| No way to detect logging bugs | Standardized SRM + health checks |
| Hundreds of experiments collide | Namespaces and mutually-exclusive layers |
| Stats correctness varies by person | Defaults wired in: CUPED, SRM, sequential, etc. |
| Decisions are political | Pre-registered decision rules |

## 1.3 Scale at Top Companies

| Company | Experiments/yr | Concurrent experiments |
|---------|---------------|----------------------|
| **Microsoft (Bing, Office)** | ~25,000+ | thousands |
| **Meta** | ~30,000+ | thousands |
| **Booking.com** | ~25,000+ | thousands |
| **Netflix** | ~thousands | hundreds |
| **LinkedIn** | ~thousands | hundreds |

At this scale, **automation and trust** are the entire job.

---

# 2. The Five Subsystems

```
┌─────────────────────────────────────────────────────────────────────┐
│              EXPERIMENTATION PLATFORM — SUBSYSTEMS                  │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │ 1. CONFIG   │───▶│ 2. ASSIGNMENT│──▶│ 3. EXPOSURE │            │
│  │   SERVICE   │    │   SERVICE    │   │   LOGGER    │            │
│  └─────────────┘    └─────────────┘    └──────┬──────┘            │
│                                                │                    │
│                                                ▼                    │
│                          ┌──────────────────────────────┐          │
│                          │ 4. METRIC PIPELINE           │          │
│                          │   (event logs → metric tables)│          │
│                          └──────────────┬───────────────┘          │
│                                          │                          │
│                                          ▼                          │
│                          ┌──────────────────────────────┐          │
│                          │ 5. ANALYSIS SERVICE          │          │
│                          │   (ATE, CI, SRM, CUPED, p)   │          │
│                          └──────────────┬───────────────┘          │
│                                          │                          │
│                                          ▼                          │
│                          ┌──────────────────────────────┐          │
│                          │ DASHBOARD / API / DECISION UI │         │
│                          └──────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

Each subsystem has its own data model, SLA, and failure modes. Interviewers test whether you can talk about them **separately and clearly**.

---

# 3. Assignment Service

## 3.1 What It Does

Given a `user_id` (or any unit) and an `experiment_id`, return a **variant** — deterministically, consistently, and with the right traffic split.

## 3.2 The Core Algorithm — Hash-Based Bucketing

```python
def assign(unit_id: str, experiment: ExperimentConfig) -> str:
    """Deterministic, salted, hash-based assignment."""
    h = hashlib.sha256(
        f"{experiment.id}|{experiment.salt}|{unit_id}".encode()
    ).hexdigest()
    bucket = int(h[:8], 16) % 10000  # 0..9999

    for variant in experiment.variants:
        if bucket < variant.upper_bound_bp:  # basis points
            return variant.name
```

| Property | Why it matters |
|----------|---------------|
| **Deterministic** | Same user gets the same variant across page loads, devices, retries |
| **Salted per experiment** | Different experiments are independent (no correlation in assignments) |
| **Modulo over hash** | Uniform distribution across buckets |
| **No DB round-trip** | Sub-millisecond at the edge |

## 3.3 Anti-Patterns

| Anti-pattern | Why it's bad |
|-------------|-------------|
| `random.random()` per request | Same user gets different variants → no causal claim |
| Reuse same salt across experiments | Strong correlation between assignments (variance underestimated) |
| Hash only `user_id` (no experiment_id) | One bucketing → can't isolate effects |
| Store assignment in DB lookup | Latency, single point of failure |
| Reassign on each session | Breaks within-user comparisons |

## 3.4 Unit of Randomization

The choice has huge implications:

| Unit | When to use | Watch out for |
|------|------------|---------------|
| **User** | Most product changes | Cross-device users (need stable ID) |
| **Session** | UI changes with short engagement | Same user sees both arms → leakage |
| **Device** | Pre-login product, mobile apps | One user with multiple devices |
| **Account** | B2B / SaaS | Account = multiple humans |
| **Cluster (geo/community)** | Network effects, marketplaces | Coarser buckets → smaller effective n |
| **Time window (switchback)** | Marketplaces, dispatch | Carryover effects |

> See `42_switchback_geo_cluster_designs.md` for cluster / switchback details.

## 3.5 Traffic Allocation

Common patterns:

- **50/50 split** — equal arms, max power
- **Asymmetric (e.g., 99/1)** — risk-averse launch, slow ramp
- **Multi-arm (e.g., 25/25/25/25)** — test multiple variants; requires multiplicity adjustment
- **Holdout** — small slice (e.g., 5%) never sees the new feature, to measure long-term cumulative impact

## 3.6 Ramp-Up Pattern

```
Day 0:   Treatment at 1% (smoke test, watch crashes/errors)
Day 1:   5% if guardrails clean
Day 3:   25%
Day 7:   50%
Day 14:  Decision point
```

Ramp-up exists to **detect breakage early** without nuking the metric — not to peek at significance.

---

# 4. Configuration & Namespaces

## 4.1 Experiment Config (Example)

```yaml
experiment_id: search_ranking_v3
owner: search-team
hypothesis: "New BM25-weighted ranker improves CTR by ≥1%"
unit: user_id
allocation:
  control:   { traffic: 0.50, params: { ranker: "current" } }
  treatment: { traffic: 0.50, params: { ranker: "bm25_v3" } }
eligibility:
  - country in [US, CA, UK]
  - device in [desktop, mobile_web]
metrics:
  primary:    [search_ctr]
  secondary:  [searches_per_session, dwell_time]
  guardrails: [latency_p95, error_rate, complaint_rate]
analysis:
  cuped: true
  cuped_covariate: pre_period_ctr
  multiple_testing: holm_bonferroni
  sequential: lan_demets_obrien_fleming
  pre_registered_lookpoints: [day_7, day_14]
duration:
  min_days: 14
  max_days: 28
salt: 0x9a2c
```

> **Why YAML/JSON config?** Every aspect of the experiment is **frozen in code**, version-controlled, reviewable, and reproducible.

## 4.2 Namespaces — Avoiding Collisions

When 500 experiments run at once, two things can collide:

1. **Same user in many experiments** → multiple treatments stack. Fine for *independent* experiments. Disastrous if both touch the same feature.
2. **Same feature changed by two experiments** → behavior is undefined.

The fix: **layers / namespaces** (Google "Overlapping Experiment Infrastructure" paper, 2010).

```
┌─────────────────────────────────────────────────────────────────┐
│                       LAYER STACK                               │
│                                                                 │
│  Layer A (Ranking):   100% traffic, partitioned into            │
│                        experiments that ONLY touch ranking.     │
│                        A user is in exactly one experiment.     │
│                                                                 │
│  Layer B (Pricing):   100% traffic, partitioned into            │
│                        experiments that ONLY touch pricing.     │
│                                                                 │
│  Layer C (UI):        100% traffic, partitioned for UI tests.   │
│                                                                 │
│  → A user is in (one ranking exp) × (one pricing exp) × (one   │
│    UI exp). These are independent because the layers don't     │
│    overlap functionally.                                        │
└─────────────────────────────────────────────────────────────────┘
```

| Term | Meaning |
|------|---------|
| **Domain** | A pool of traffic carved out for a class of experiments |
| **Layer** | A slice of code/feature space; experiments within a layer are mutually exclusive |
| **Launch layer** | Used for permanent rollouts (no longer experimental) |
| **Holdout layer** | Permanent control group (never sees a feature) |

## 4.3 Mutually Exclusive vs Independent

| Setup | Use case |
|-------|----------|
| **Mutually exclusive** (same layer) | Two ranking algorithms — can't run on same user |
| **Independent** (different layers) | Ranking change and UI color change — orthogonal, fine to overlap |

## 4.4 Salt Strategy

Each layer has a unique salt. Each experiment within a layer has a sub-salt. Salts are rotated on a schedule (e.g., per quarter) to break long-term correlations and bucket habituation.

---

# 5. Exposure Logging

## 5.1 The Exposure Event

When a user actually **sees** the variant, the platform logs an event:

```json
{
  "event": "exposure",
  "ts": "2026-05-15T01:23:45Z",
  "unit_id": "user_4711",
  "experiment_id": "search_ranking_v3",
  "variant": "treatment",
  "config_version": "v3.2",
  "request_id": "abc123"
}
```

## 5.2 Why Exposure ≠ Assignment

**Assignment** says "if this user shows up, they'd be in treatment." **Exposure** says "this user actually saw treatment." A user who never visited the surface should be **excluded** from analysis — they couldn't have been affected.

> **The exposure-vs-assignment distinction is one of the most common interview questions for experimentation platform roles.**

## 5.3 Triggered Analysis Uses Exposure

A user is **triggered** when they first hit the surface that could be impacted by the experiment. Triggered analysis restricts to:

$$
\text{Triggered population} = \{ u : u \text{ exposed within experiment window} \}
$$

This sharply increases sensitivity if only a fraction of users are exposed (e.g., new search filter only affects 10% of searches).

## 5.4 Exposure Logging Failure Modes

| Failure | Symptom | Fix |
|---------|--------|-----|
| Missing logs | Treatment count < expected | Server-side logging, not client-side |
| Duplicate logs | Inflated exposure counts | Idempotent dedupe by `(unit, experiment, ts_bucket)` |
| Lagged logs | Late-arriving data after analysis runs | Window-buffered analysis (wait 6h before computing) |
| Variant mismatch | Logged variant ≠ delivered variant | End-to-end test on every config change |

---

# 6. Metric Definitions & the Metric Store

## 6.1 The Problem

Every analyst at a company writes a slightly different SQL for "DAU" or "conversion rate." Five years later, nobody knows which definition the C-suite is looking at.

**Solution:** centralize metric definitions in code.

## 6.2 A Metric Definition

```yaml
metric_id: search_ctr
display_name: "Search click-through rate"
unit: user_id
type: ratio
numerator:
  event: search_result_click
  filter: position <= 10
denominator:
  event: search_impression
window_days: 14
cuped_covariate: pre_period_search_ctr
guardrail_thresholds:
  red:  -0.5%   # if treatment - control drops > 0.5%, block ship
  yellow: -0.1%
owner: search-team
created: 2024-08-01
```

## 6.3 Metric Types

| Type | Example | Notes |
|------|--------|-------|
| **Count** | sessions/user | Often Poisson-ish; check zero inflation |
| **Continuous** | revenue/user | Often heavy-tailed; winsorize |
| **Proportion** | conversion rate | Use proportion z-test or Wilson CI |
| **Ratio** | CTR (clicks/impressions) | Delta method for variance |
| **Time-to-event** | time to first purchase | Survival analysis or simplified buckets |

## 6.4 Metric Hierarchy

| Level | Purpose | Example |
|-------|---------|---------|
| **Primary (OEC)** | The single metric you optimize | Search CTR |
| **Secondary** | Other indicators of success | Searches/session |
| **Guardrail** | Must not degrade | Latency p95, crash rate |
| **Diagnostic** | Investigation only | Per-segment lifts |

> OEC = Overall Evaluation Criterion (Kohavi). The North Star metric for the experiment.

## 6.5 Storage Layout

```
┌────────────────────────────────────────────────────────────────┐
│                METRIC STORE — DATA LAYERS                       │
│                                                                 │
│  RAW EVENTS              (Kafka, S3, Spark/Glue)               │
│  ↓                                                              │
│  CLEANED EVENTS          (deduped, schema-validated)            │
│  ↓                                                              │
│  USER-DAY AGGREGATES     (sessions, clicks, $/user, etc.)       │
│  ↓                                                              │
│  EXPERIMENT-USER METRICS (joined with assignment + exposure)    │
│  ↓                                                              │
│  EXPERIMENT-METRIC SCORES (ATE, CI, p-value, SRM, CUPED)        │
└────────────────────────────────────────────────────────────────┘
```

## 6.6 Metric SQL Pattern

```sql
-- USER-DAY: agg of events
CREATE TABLE user_day_metrics AS
SELECT
    user_id,
    DATE(ts) AS dt,
    COUNT(*) FILTER (WHERE event='search') AS searches,
    COUNT(*) FILTER (WHERE event='click')  AS clicks,
    SUM(price) FILTER (WHERE event='purchase') AS revenue
FROM events
GROUP BY 1, 2;

-- EXPERIMENT-USER: join with assignment + exposure
CREATE TABLE exp_user_metrics AS
SELECT
    a.experiment_id, a.variant, a.user_id,
    SUM(m.clicks)        AS clicks_post,
    SUM(m.searches)      AS searches_post,
    SUM(m.revenue)       AS revenue_post
FROM assignments a
JOIN exposures e USING (user_id, experiment_id)
JOIN user_day_metrics m
  ON m.user_id = a.user_id
 AND m.dt >= a.assignment_dt
 AND m.dt <  a.assignment_dt + INTERVAL '14 days'
WHERE a.experiment_id = 'search_ranking_v3'
GROUP BY 1,2,3;
```

---

# 7. Analysis Pipeline

## 7.1 What the Analysis Service Does

For each `(experiment, metric)` pair:

1. Load the **experiment-user metric** table.
2. Apply CUPED (if configured).
3. Compute ATE, SE, CI, p-value.
4. Apply multiple-testing correction across metrics.
5. Apply sequential boundaries (if multiple looks).
6. Compute SRM and other health checks.
7. Generate segment lifts (HTE).
8. Persist results with a snapshot hash.

## 7.2 Pseudocode

```python
def analyze(experiment, metric, data):
    # 1. Filter to triggered population if requested
    if metric.triggered:
        data = data[data.exposed]

    # 2. CUPED adjustment if configured
    if metric.cuped_covariate:
        data["Y"] = cuped_adjust(data["Y"], data[metric.cuped_covariate])

    # 3. ATE and CI
    ate, se = diff_in_means(data, "variant", "Y")
    ci = (ate - 1.96*se, ate + 1.96*se)
    p_value = two_sided_p(ate, se)

    # 4. SRM (sample ratio mismatch)
    srm_p = srm_check(data["variant"], experiment.allocation)

    # 5. Apply multiple-testing
    p_adj = holm_bonferroni([p_value, ...other metrics...])

    # 6. Sequential boundary
    if experiment.sequential:
        p_seq = sequential_adjust(p_value, look_index)

    return AnalysisResult(ate, ci, p_value, p_adj, srm_p, ...)
```

## 7.3 Latency Tiers

| Tier | Use case | Latency |
|------|----------|---------|
| **Near-real-time (NRT)** | Smoke test, ramp gates | Minutes |
| **Daily** | Standard analysis | < 1 day |
| **Backfill** | Re-run after schema or metric change | Hours/days |

NRT is *not* for decision-making — it's for detecting **bugs**.

---

# 8. Experiment Lifecycle

```
┌───────────────────────────────────────────────────────────────────┐
│                  EXPERIMENT LIFECYCLE STATES                       │
│                                                                    │
│   DRAFT ──▶ REVIEW ──▶ APPROVED ──▶ RUNNING ──▶ STOPPED            │
│                                          │                          │
│                                          ▼                          │
│                                   ANALYZED ──▶ LAUNCHED / KILLED   │
│                                                                    │
│   ANY STATE: can transition to ARCHIVED                            │
└───────────────────────────────────────────────────────────────────┘
```

| State | What happens | Gate to next state |
|-------|--------------|--------------------|
| **Draft** | Owner writes config, picks metrics | Self-review |
| **Review** | Peer + DS review (power, metrics, guardrails) | Approver checks pre-registration |
| **Approved** | Locked config, ready to schedule | Manual or automatic start |
| **Running** | Traffic flowing; daily health checks | Min duration met, no critical alerts |
| **Stopped** | Traffic to 0 or 100 | Final analysis runs |
| **Analyzed** | Decision document generated | Decision meeting |
| **Launched** | Treatment becomes new control (moved to launch layer) | — |
| **Killed** | Treatment removed; reasons logged | — |

> The lifecycle is also a **policy enforcement mechanism**: you can't run an experiment without a primary metric, a power calculation, or an owner.

---

# 9. Triggered Analysis

## 9.1 What It Is

Most experiments only affect a fraction of users:

- A new checkout flow → only affects users who reach checkout
- A new search ranker → only affects users who search
- A new mobile-app onboarding → only affects new installs

If we analyze the entire population, the treatment effect is **diluted** by users who never encountered the change.

**Triggered analysis** restricts to users who were **actually exposed**.

## 9.2 Triggering Conditions

```yaml
trigger:
  type: event
  event: search_request
  window_after_assignment_days: 14
```

A user is triggered as soon as they first hit the surface within the window.

## 9.3 Math: Why Triggering Helps

If only fraction $p$ of users trigger, and the true effect on triggered users is $\delta$, then:

| Quantity | Value |
|----------|------|
| Population ATE | $p \cdot \delta$ |
| Triggered ATE | $\delta$ |
| Population SE | scaled by full $n$ |
| Triggered SE | scaled by $p \cdot n$, but with smaller $\sigma^2_{\text{triggered}}$ |

The trade-off: smaller sample but cleaner signal. In practice, **triggered analysis almost always wins** when $p < 0.5$.

## 9.4 Common Pitfall — Conditioning on Treatment

You can't trigger based on a behavior that happens **only** in the treatment group (or differently across arms). That biases the comparison. Trigger must be **symmetric**: e.g., "user issued a search" is fine; "user saw the new ranking" is not (only treatment users see it).

> **Counterfactual trigger:** the trigger must fire in *both* arms for the *same* upstream event.

---

# 10. Scaling to Thousands of Concurrent Experiments

## 10.1 The Compute Problem

If you have:
- 5,000 active experiments
- 50 metrics each
- 10 segments per metric

That's **2.5M analysis computations daily**. Approaches:

| Strategy | Trick |
|----------|-------|
| **Pre-aggregate** | Materialize user-day-metric tables once; reuse across experiments |
| **Pre-CUPED-covariate** | Compute pre-period covariates centrally, not per-experiment |
| **Pre-segment cuts** | Pre-compute slice tables (country × device × ...) |
| **Shared compute** | One Spark job per metric × hundreds of experiments |
| **Approximate where safe** | Sketches (HLL for cardinality), reservoir sampling for long tails |

## 10.2 The Throughput Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  THROUGHPUT-ORIENTED ANALYSIS                       │
│                                                                     │
│  Daily Spark Job (one)                                              │
│   ├─ scans last 24h of events                                       │
│   ├─ joins with ALL active experiments' assignments                 │
│   ├─ writes one big (experiment, metric, segment, day) cube         │
│   └─ partitioned by experiment_id for fast slicing                  │
│                                                                     │
│  Analysis Service (online)                                          │
│   ├─ reads cube                                                     │
│   ├─ applies CUPED, SRM, sequential adjustments                    │
│   └─ writes results to OLAP/Postgres for dashboards                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 10.3 The Coverage Problem

When you can run only N experiments per layer, **demand exceeds capacity**. Solutions:

- **Smaller traffic per experiment** (more concurrent, slower individually)
- **Variance reduction** (CUPED) → smaller traffic for same sensitivity
- **Surrogate metrics** that move faster
- **Priority queues** — high-impact experiments get more traffic

---

# 11. Governance, Trust, and Best Practices

## 11.1 Pre-Registration

Before running, the experiment owner registers:
- Hypothesis
- Primary metric
- Decision rule (e.g., "ship if primary lift ≥ 0.5% with p ≤ 0.05 AND no guardrail red")
- Look-points (for sequential testing)

Pre-registration prevents **HARKing** (Hypothesizing After Results are Known) and **p-hacking** by metric-shopping.

## 11.2 Sample Ratio Mismatch (SRM)

Daily chi-square test on actual vs expected traffic split. SRM is **always** a blocker — if it fails, **do not analyze**, investigate first.

## 11.3 Reproducibility

Every analysis result is keyed by:
```
(experiment_id, config_version, metric_id, metric_version,
 analysis_code_version, data_snapshot_hash)
```

This is the basis for trust. "We can reproduce any result in the platform" is a culture decision, not a feature.

## 11.4 Decision Document

Auto-generated after experiment ends:

```
Experiment:       search_ranking_v3
Primary metric:   search_ctr
  ATE:           +0.74% [+0.31%, +1.17%], p=0.001
  CUPED applied: yes, var reduction 38%
Guardrails:
  latency_p95:   neutral [-0.2ms, +0.5ms]
  error_rate:    neutral
SRM:             clean (p=0.41)
Recommendation:  SHIP
```

## 11.5 Anti-Patterns the Platform Should Block

| Anti-pattern | Platform defense |
|-------------|------------------|
| Peeking | Sequential testing or pre-registered look-points |
| Metric shopping | Lock primary metric at start |
| Segment shopping | Multiple-testing correction across segments |
| HARKing | Pre-registered hypotheses, versioned |
| Stopping early on noise | Min-duration enforcement |
| Ignoring guardrails | Hard block in decision UI |

## 11.6 Roles

| Role | Responsibility |
|------|---------------|
| **Experiment owner** | Hypothesis, primary metric, decisions |
| **DS reviewer** | Power, design, statistical soundness |
| **Platform DS / DRI** | Methodology, defaults, incident response |
| **Data engineer** | Pipelines, joins, SLAs |
| **Eng lead** | Code change, rollout safety |

---

# 12. Reference Architectures (Industry)

## 12.1 Microsoft ExP

- Layered design, decades of refinement
- Heavy emphasis on **CUPED** and **SRM**
- **OEC** (Overall Evaluation Criterion) framework
- Publishes extensively (Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments*)

## 12.2 Meta

- **PlanOut** language for experiment configuration
- **Deltoid** analytics pipeline
- Massive layer system with cross-experiment trust

## 12.3 LinkedIn XLNT

- **T-REX** (testing rigorous experimentation)
- Strong **ramp framework** with automated gates
- Real-time NRT analytics

## 12.4 Netflix XP

- A/B for content + product
- Heavy emphasis on **quasi-experiments** when randomization is hard (causal team)

## 12.5 Airbnb ERF

- **Experiment Reporting Framework**
- Marketplace-specific tooling (geo experiments)

## 12.6 Booking

- 1,000+ experiments running simultaneously
- Strong **culture** of experimentation as core decision tool

---

# 13. Interview Questions with Strong Answers

## Q1: "Walk me through the architecture of an experimentation platform."

> Five subsystems. **Config service** stores experiment definitions — YAML/JSON with hypothesis, allocations, metrics, guardrails — versioned and reviewed. **Assignment service** is a deterministic hash function: `bucket = hash(experiment_id, salt, user_id) mod 10000`, then bucket maps to a variant. Sub-millisecond, no DB. **Exposure logger** captures the moment a user actually sees the variant — critical for triggered analysis. **Metric pipeline** runs daily: events → user-day aggregates → experiment-user joins → metric tables. Built on Spark or similar for scale. **Analysis service** consumes metric tables and produces ATE, CI, SRM, CUPED adjustments, multiple-testing-corrected p-values, segment lifts. Results are written to an OLAP store for dashboards and decision documents. The whole pipeline is reproducible by `(config_version, data_snapshot, analysis_code_version)`.

## Q2: "What is the difference between assignment and exposure? Why does it matter?"

> Assignment is *which arm a user would be in if they showed up*. Exposure is *which arm a user actually saw*. The distinction matters because most experiments don't affect every user — a new checkout flow only affects users who reach checkout. If you analyze the full assigned population, you dilute the signal with users who never encountered the change. Triggered analysis filters to exposed users to recover sensitivity. The catch: triggering must be **symmetric** — the trigger event must fire equally in both arms for the same upstream behavior, otherwise you bias the comparison. So we trigger on the upstream surface (`search_request`), not on the downstream new-feature interaction (`saw_new_ranker`, which only treatment users can fire).

## Q3: "How do you ensure two experiments don't interfere with each other?"

> Two mechanisms. First, **namespaces / layers** — Google's overlapping experiment infrastructure. Experiments that touch the same feature are placed in the same layer and are mutually exclusive (a user is in one of them). Experiments in different layers can overlap freely. Second, **salting** — each layer and each experiment has its own salt added to the hash, so assignments are decorrelated. If both mechanisms are in place, a user can be in hundreds of orthogonal experiments without confounding. The platform also runs **A/A tests** continuously to catch unintended correlations.

## Q4: "How does the platform detect sample ratio mismatch and what do you do when it fires?"

> Chi-square goodness-of-fit on actual vs expected variant counts, run daily and per-segment. If p < 0.001 we declare SRM. SRM is never a metric problem — it's a system problem. Causes: bucketing bug, downstream filtering that's correlated with variant (e.g., logged-out users dropping in one arm), bot traffic differing across arms, caching that prefers one variant, sticky cookies misbehaving. The response is: **stop analyzing, start investigating**. Check the assignment audit log first, then the exposure pipeline, then the metric pipeline. Once root cause is found, decide: re-run from scratch or scope the analysis to the unaffected segment. The platform should never auto-publish results when SRM is red.

## Q5: "How would you design the metric store?"

> Centralized, versioned, code-defined. A metric is a YAML or Python class with: unit of analysis (`user_id`), type (count, ratio, continuous), numerator and denominator events, window, CUPED covariate, guardrail thresholds, and owner. Stored in a registry alongside the platform code. The pipeline reads from this registry to build the data layer hierarchy: raw events → cleaned events → user-day aggregates → experiment-user metrics → experiment-metric scores. The benefit is a single source of truth: "DAU" means the same thing for every dashboard and every experiment. Changes to a metric definition bump a version, and historical results stay tied to their original version so they remain reproducible.

## Q6: "How does the platform handle thousands of concurrent experiments efficiently?"

> Two pillars: pre-aggregation and shared compute. Pre-aggregation means user-day metric tables are built once per day and reused across all experiments — so adding the 5,001st experiment doesn't add a Spark pass over events. Shared compute means one job scans the events, joins assignments for all active experiments, and writes a partitioned `(experiment, metric, segment, day)` cube. The analysis service reads from the cube. We also pre-compute CUPED covariates (pre-period metrics) centrally, since the same covariate is used across many experiments. For low-traffic experiments, the platform may require longer minimum durations or recommend variance reduction techniques. Capacity planning is itself a science — we monitor "experiments queued vs active" and allocate traffic via priority.

## Q7: "Walk me through what happens when an experimenter clicks 'analyze'."

> The analysis service loads the experiment config, pulls the experiment-user metric tables for primary, secondary, and guardrail metrics, applies CUPED with the configured pre-period covariate, runs the triggered-population filter, computes diff-in-means and SE per metric, runs SRM, applies multiple-testing correction (Holm-Bonferroni or BH-FDR) across metrics, applies sequential boundary if this is a pre-registered look-point, computes pre-defined segment cuts for HTE, and writes a result blob keyed by `(experiment, config_version, analysis_code_version, data_hash)`. The dashboard renders the result with banners for SRM, guardrail violations, or under-powered metrics. The decision document is auto-generated.

## Q8: "How do you balance rigor with operational simplicity?"

> Sane defaults baked in, configurable when needed. Default: CUPED on for any metric with a pre-period covariate. Default: Holm-Bonferroni across the metric panel. Default: sequential testing with pre-registered looks at day 7 and 14. Default: SRM gates analysis. The owner *can* opt out, but that's a documented choice reviewed by a platform DS. This makes the rigorous path the default path and forces the friction-cost onto the path of least rigor — which is the right incentive structure. Most teams happily live with defaults; the platform DS team only spends time on the 5% of experiments with unusual designs.

## Q9: "How do you handle long-running experiments (e.g., 90 days for retention metrics)?"

> A few patterns. First, **surrogate metrics**: identify near-term metrics (week-1 engagement) that historically predict long-term outcomes (30-day retention), so you can decide quickly with provisional confidence. Second, **continuous holdout**: a small slice of users never sees the new feature, so you can measure cumulative impact over months even after launch. Third, **calibration runs**: periodically re-run shorter experiments to verify the surrogate-to-outcome relationship hasn't shifted. Fourth, **always-on monitoring**: even after launch, primary metrics are tracked vs the holdout to catch decay (novelty effects, primacy effects). For 90-day metrics specifically, we accept that you can't ship every change in a week; the platform's job is to make the trade-off explicit, not pretend you can.

## Q10: "What's the hardest part of running an experimentation platform at scale?"

> Trust. Not statistics, not compute — trust. People stop believing results if a single high-profile experiment is wrong. The platform must make the right thing the easy thing: SRM enforcement, reproducibility, frozen configs, pre-registered hypotheses, automated guardrails, default CUPED, audit logs, and clear post-mortems when something goes wrong. The platform DS team's job is half methodology and half evangelism — every result needs to be defensible, and every failure mode (logging bug, bucket leak, novelty effect, attribution window mismatch) needs a written-up incident review. When trust is high, decisions are fast and capital is deployed efficiently; when trust is low, every result needs a meeting and the company moves slower. The hardest engineering problems are downstream of this trust problem.

---

# 14. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│         EXPERIMENTATION PLATFORM TAKEAWAYS                          │
│                                                                     │
│  1. Five subsystems: config, assignment, exposure, metrics,        │
│     analysis. Each has its own data model and failure modes.       │
│                                                                     │
│  2. Assignment = deterministic salted hash. Exposure ≠ assignment. │
│     Triggered analysis uses exposure, not assignment.              │
│                                                                     │
│  3. Layers/namespaces let thousands of experiments run             │
│     concurrently without interference.                              │
│                                                                     │
│  4. Metrics are CODE, versioned, owned. No more ad-hoc SQL.        │
│                                                                     │
│  5. Lifecycle (draft → review → approved → running → analyzed     │
│     → launched/killed) is a policy enforcement mechanism.         │
│                                                                     │
│  6. SRM is a HARD gate on analysis. Never ignore.                  │
│                                                                     │
│  7. CUPED, sequential testing, multiple-testing correction are     │
│     platform defaults, not analyst opt-ins.                        │
│                                                                     │
│  8. Pre-aggregation + shared compute = scale to thousands of       │
│     experiments without linear cost growth.                        │
│                                                                     │
│  9. Reproducibility = (config, data, code) hashes. Trust depends   │
│     on this completely.                                            │
│                                                                     │
│  10. Trust is the product. Statistics is the means.                 │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"A modern experimentation platform is five subsystems — config, assignment, exposure, metric pipeline, analysis — plus governance that turns rigorous statistics into the default operating mode of the company. Statistics is 1% of the job; data engineering and trust are the other 99%."*

---

**Further reading**

- Tang, Agarwal, Liu, Walker (2010) — *Overlapping Experiment Infrastructure*
- Kohavi, Tang, Xu (2020) — *Trustworthy Online Controlled Experiments*
- Bakshy, Eckles, Bernstein (2014) — *Designing and Deploying Online Field Experiments (PlanOut)*
- Xu et al. (2015) — *From Infrastructure to Culture: A/B Testing Challenges in Large Scale Social Networks*
- Crook, Frasca, Kohavi, Longbotham (2009) — *Seven Pitfalls to Avoid when Running Controlled Experiments on the Web*
