# Debugging Measurement Systems — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Experiment Reliability, Anomaly Investigation, Logging Issues, Bias Detection

> **Context:** Most experimentation interviews assume the data is clean and ask about p-values. The hardest, most senior question on platforms like OpenAI's is the opposite: **the data is dirty, and the senior IC's job is to figure out how.** This guide covers the playbook for SRM investigations, exposure-join bugs, attribution mistakes, novelty/primacy, ghost lifts, and the classic post-mortems every IC has to write at some point. It is the difference between "I know t-tests" and "I can run a trustworthy platform."

---

## Table of Contents

1. [Why Measurement Debugging Is the Senior IC Job](#1-why-measurement-debugging-is-the-senior-ic-job)
2. [The Trust Hierarchy](#2-the-trust-hierarchy)
3. [Sample Ratio Mismatch — Full Playbook](#3-sample-ratio-mismatch--full-playbook)
4. [Exposure Join Bugs](#4-exposure-join-bugs)
5. [Attribution Window Errors](#5-attribution-window-errors)
6. [Duplicate Events](#6-duplicate-events)
7. [Logging Lag & Late Arriving Data](#7-logging-lag--late-arriving-data)
8. [Novelty and Primacy Effects](#8-novelty-and-primacy-effects)
9. [Ghost Lifts — Effects That Don't Exist](#9-ghost-lifts--effects-that-dont-exist)
10. [Funnel Bugs and Counterfactual Triggers](#10-funnel-bugs-and-counterfactual-triggers)
11. [Simpson's Paradox in Experiments](#11-simpsons-paradox-in-experiments)
12. [The Investigation Workflow](#12-the-investigation-workflow)
13. [Anomaly Detection Tooling](#13-anomaly-detection-tooling)
14. [Interview Questions with Strong Answers](#14-interview-questions-with-strong-answers)
15. [Key Takeaways](#15-key-takeaways)

---

# 1. Why Measurement Debugging Is the Senior IC Job

## 1.1 The Real World

In a perfect world, every experiment is a clean RCT and the only question is "is p < 0.05?". In the real world:

- Logging is implemented by humans who make mistakes.
- Pipelines have late-arriving data, retries, deduplication bugs.
- Bot traffic and crawlers contaminate metrics.
- Browsers cache pages, breaking exposure logs.
- Mobile app updates ship at different times.
- Caches across regions vary.
- Edge cases (logged-out users, signups mid-experiment) confuse joins.

**Roughly 1 in 5 experiments has a measurement problem detectable on the first pass.** The senior IC's job is to know what to look for and not ship a wrong decision.

## 1.2 The Mindset Shift

| Junior IC | Senior IC |
|-----------|----------|
| "p = 0.04, ship it" | "p = 0.04, but the SRM check is borderline — let me investigate" |
| Trusts the result | Asks "what would make this result look like this if it were wrong?" |
| Treats anomalies as edge cases | Treats anomalies as the **expected** mode of failure |
| Reads metric dashboards | Reads exposure logs, raw events, deduplication keys |

---

# 2. The Trust Hierarchy

When an experiment shows a surprising result, work **down** this list:

```
┌─────────────────────────────────────────────────────────────────┐
│              MEASUREMENT TRUST HIERARCHY                         │
│                                                                  │
│  1. Is randomization sound?         (SRM, bucketing)            │
│  2. Is exposure logging correct?    (counts match, no dup/miss) │
│  3. Is the metric definition right? (units, attribution)        │
│  4. Are external shocks affecting one arm differently? (bots,   │
│     mobile rollouts, regional outages)                          │
│  5. Is the analysis math right?     (CUPED, sequential, MTC)    │
│  6. Is the effect real?              (novelty, primacy, segments)│
└─────────────────────────────────────────────────────────────────┘
```

Never skip levels. Most ICs jump from #1 directly to #6 ("must be a novelty effect") and miss obvious data bugs.

---

# 3. Sample Ratio Mismatch — Full Playbook

## 3.1 What SRM Tests

You configured 50/50 traffic split. The observed counts are 51,234 control and 49,876 treatment. Is the imbalance just noise, or is something broken?

```python
from scipy.stats import chisquare

obs = [51234, 49876]
expected_split = [0.5, 0.5]
expected_counts = [sum(obs) * p for p in expected_split]

chi2, p = chisquare(obs, expected_counts)
print(f"chi2={chi2:.2f}, p={p:.4f}")
# Rule of thumb: p < 0.001 → SRM
```

## 3.2 SRM Thresholds

| p-value | Status | Action |
|---------|--------|--------|
| p > 0.05 | Healthy | Proceed |
| 0.01 < p < 0.05 | Yellow | Inspect, but not blocking |
| 0.001 < p < 0.01 | Orange | Investigate before publishing |
| p < 0.001 | Red (SRM) | **Block analysis. Investigate first.** |

## 3.3 SRM by Segment

Overall SRM can be clean even when SRM exists within a segment (e.g., one country, one device). Always run **segmented SRM**.

```python
for segment in ["US", "UK", "DE", "JP"]:
    seg_df = df[df.country == segment]
    obs = seg_df.groupby("variant").size().values
    chi2, p = chisquare(obs, [obs.sum()/2]*2)
    print(f"{segment}: chi2={chi2:.2f}, p={p:.4f}")
```

## 3.4 Root Causes — Walk Down This List

| Cause | Symptom | Diagnostic |
|-------|--------|-----------|
| **Bucketing bug** | Hash function changed or wrong salt | Re-run assignment for sample of users, compare |
| **Filter applied post-randomization** | Some variant filtered out (e.g., country, device) | Check filters in the pipeline that ran *after* assignment |
| **Bot traffic** | Imbalance largest for non-human user agents | Stratify SRM by `is_bot` flag |
| **Caching / CDN** | One variant cached at edge, served extra | Check edge logs vs origin logs |
| **Browser plugin** | Treatment variant breaks a script, users bail | Check session counts by browser type |
| **Mobile app version** | New variant only enabled in newer app versions | Stratify by app version |
| **Page load bug** | Treatment crashes; users reload as control | Check error rates per variant |
| **Sticky cookie misbehavior** | Same user toggling variants on revisit | Check assignment audit log for duplicate user-variant pairs |
| **Server-side sampling** | Logging system samples one variant differently | Check raw event counts vs logged counts |
| **Re-randomization** | Experiment was paused/restarted mid-run | Check timeline of config changes |

## 3.5 Real-World SRM Story (Template)

> **Incident:** Search experiment shows treatment lift of +1.8% on CTR. SRM chi-square p < 0.0001 (52/48 split).
>
> **Investigation:** Stratified SRM by country revealed imbalance largest in Brazil. Further stratification by app version showed Brazil users on app v4.2 only saw treatment (the new ranker required a feature flag that defaulted on in v4.2 in Brazil specifically). Control users on v4.2 in Brazil were being assigned to treatment by the app's local fallback logic.
>
> **Root cause:** Feature flag default in mobile app v4.2 for Brazil bypassed server-side bucketing.
>
> **Resolution:** Fix the mobile app, re-run the experiment. The original result is **invalid** even though p < 0.05 — because the treatment population is fundamentally different.

## 3.6 What SRM Tells You (and Doesn't)

SRM detects **imbalance in counts**. It does NOT detect:
- Imbalance in *composition* with matched counts (e.g., 50/50 but treatment over-represents power users).
- Effects of within-arm filtering.

For composition issues, check **pre-period covariates** balanced by arm using **standardized mean differences** (rule of thumb: |SMD| > 0.1 is concerning).

---

# 4. Exposure Join Bugs

## 4.1 What Goes Wrong

The analysis joins `assignments` × `exposures` × `metric_events`. Bugs:

| Bug | Effect |
|-----|--------|
| Join on `user_id` but treatment uses `device_id` | Drops cross-device users |
| Filter on `assignment_dt <= event_dt` missing | Pre-assignment events get attributed to treatment |
| Outer vs inner join confusion | Users with no events drop silently (selection bias) |
| Time zones mismatched | Off-by-N-hours bucketing |
| Event schema changed mid-experiment | Some events have null variant attribution |

## 4.2 Diagnostic Queries

```sql
-- Diagnostic 1: Assignment dates straddle the experiment window
SELECT MIN(assignment_dt), MAX(assignment_dt), COUNT(*)
FROM assignments
WHERE experiment_id = 'search_v3';

-- Diagnostic 2: Events before assignment (should be ZERO)
SELECT COUNT(*)
FROM exp_user_metrics a
JOIN events e USING (user_id, experiment_id)
WHERE e.ts < a.assignment_dt;

-- Diagnostic 3: Users with no events at all (% per arm)
SELECT
    variant,
    SUM(CASE WHEN total_events = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS zero_event_rate
FROM exp_user_metrics
GROUP BY variant;
```

## 4.3 The "Funnel Inversion" Bug

A common bug: the analyst inadvertently filters by **outcome** rather than **upstream behavior**.

```sql
-- WRONG: filters by outcome
SELECT * FROM exp_user_metrics WHERE clicks > 0;
-- This drops users who didn't click. Treatment may have moved zero-click users into one-click users → biases base rate.

-- RIGHT: filter by upstream activity
SELECT * FROM exp_user_metrics WHERE impressions > 0;
```

> **Rule:** filter on **eligibility** or **upstream activity**, never on the **outcome itself**.

---

# 5. Attribution Window Errors

## 5.1 The Question

If a user is assigned on day 0, sees treatment on day 2, clicks an ad on day 8, and buys on day 12 — **which experiment owns the buy?**

## 5.2 Common Windows

| Window | Use |
|--------|-----|
| **Same session** | UI-level changes (button color) |
| **24h post-exposure** | Search ranking |
| **7-day post-exposure** | Recommendation systems |
| **30-day post-exposure** | Marketing campaigns, retention |
| **Lifetime** | Long-term holdouts |

## 5.3 Pitfalls

| Pitfall | Example | Fix |
|---------|---------|-----|
| **Window too short** | Retention experiment with 24h window — misses returning users | Match window to metric horizon |
| **Window too long** | UI experiment with 30d window — many other things happened | Match window to behavioral horizon |
| **Last-touch vs first-touch** | User in multiple experiments — which one gets credit? | Define attribution policy upfront |
| **Cross-device events** | User assigned on mobile, buys on desktop | Use stable user identity |

## 5.4 First-Exposure Anchoring

A common pattern: anchor the user's experiment window to their **first exposure**, not assignment.

```
Assignment day:    Day 0 (user_id in bucket)
First exposure:    Day 3 (user actually visited surface)
Analysis window:   [Day 3, Day 17] = 14 days from exposure
```

This is the basis for **triggered analysis** and is much more sensitive than assignment-anchored windows.

---

# 6. Duplicate Events

## 6.1 Sources

- **Client retries** on flaky networks.
- **At-least-once delivery** in Kafka.
- **Multiple analytics SDKs** (Google Analytics + internal logger both firing).
- **Webhook duplicates** from third parties.

## 6.2 Detection

```sql
-- Find events with same (user, event_type, ts) duplicate
SELECT user_id, event_type, ts, COUNT(*) AS n
FROM events
GROUP BY 1, 2, 3
HAVING COUNT(*) > 1
ORDER BY n DESC
LIMIT 100;
```

## 6.3 Dedup Patterns

| Pattern | Best for |
|---------|---------|
| **Idempotency key** (event_id from client) | Best — explicit dedup |
| **Hash of (user, type, ts_bucket)** | When event_id missing |
| **Window dedup** | Take only first event per (user, type) per 5-min window |
| **Server-side dedup** | At the ingestion layer, not analysis layer |

## 6.4 Why It Bites Experiments

If treatment arm has more retries (e.g., the new feature loads slowly and users retry more), duplicate-event rates differ between arms. CTR, sessions, and engagement metrics all inflate for the arm with more dups.

**Diagnostic:** dedup rate per arm. If dedup rate differs by > 0.1%, flag.

---

# 7. Logging Lag & Late Arriving Data

## 7.1 The Problem

Real-time analysis on day-0 data sees only 70% of events; the rest arrive late (mobile clients with poor connectivity, batched uploads, etc.). If the lag rate differs by arm, the analysis is biased.

## 7.2 Patterns

| Pattern | Use |
|---------|-----|
| **Wait window** (e.g., 6h after end-of-day) | Standard daily analysis |
| **Late-arrival backfill** | Re-run analysis daily for last 7 days |
| **Late-data SLO** | Discount events older than N days as "late" and exclude |

## 7.3 Diagnostic

```sql
-- Distribution of event_ts vs ingest_ts
SELECT
    DATEDIFF(HOUR, event_ts, ingest_ts) AS lag_hours,
    variant,
    COUNT(*) AS n
FROM events
GROUP BY 1, 2
ORDER BY 1;
```

If treatment has heavier right tail of lag, the early daily analyses overstate or understate effects.

---

# 8. Novelty and Primacy Effects

## 8.1 Definitions

| Effect | Description | Direction of bias |
|--------|-------------|-------------------|
| **Novelty** | Users click on new things just because they're new | Initial lift, fades over time → **overstates** true effect |
| **Primacy** | Users dislike change, slow to adopt | Initial dip, recovers over time → **understates** true effect |

## 8.2 Detection

Plot daily metric difference over the experiment duration:

```
Day 1  |████████  +5%
Day 2  |█████     +3%
Day 3  |███       +2%
Day 4  |██        +1%
Day 5  |██        +1%
Day 6  |██        +1%
Day 7  |█         +0.5%
       └──────────────
       → Novelty: large initial effect, decays
```

```
Day 1  |   -3%
Day 2  |   -2%
Day 3  | +0.5%
Day 4  | +1%
Day 5  | +2%
Day 6  | +2%
Day 7  | +2.5%
       └──────────────
       → Primacy: dip then recovery
```

## 8.3 Mitigation

- **Run longer.** Day 14+ is usually past novelty for most products.
- **Restrict to power users.** If novelty only affects casual users, the power-user segment shows steady state.
- **Compare to long-running holdout** if available — gives a true steady-state baseline.

## 8.4 Interview Pitch

> "When I see a strong day-1 effect that decays over the next week, my prior is novelty. We should hold off the launch decision until at least day 14 and ideally have a parallel small long-running holdout to monitor the steady state."

---

# 9. Ghost Lifts — Effects That Don't Exist

## 9.1 What Is a Ghost Lift

A statistically significant treatment effect that **isn't real** — it's an artifact of the measurement system.

## 9.2 The Five Most Common Ghosts

| Ghost | Cause | Detection |
|-------|-------|----------|
| **Exposure-counting ghost** | Treatment users counted more exposures (e.g., new feature triggers extra page views) | Compare per-user exposure counts |
| **Survivorship ghost** | Bad users drop out of treatment more (e.g., crashes) | Compare retention/churn rates per arm |
| **Selection ghost** | Filter applied after randomization correlated with variant | Check applied filters' selection rates |
| **Timing ghost** | Treatment rolled out at different time than control measurement | Check assignment timestamps per arm |
| **Logging ghost** | New code logs additional events | Diff event schema across deployment |

## 9.3 Example Ghost — Survivorship

> Treatment shows +6% revenue/user. But treatment had a crash bug that caused 8% of users to uninstall. The remaining treatment users are **the most engaged 92%**. They naturally spend more than the full control population. **The lift is not from the feature — it's from filtering out the bottom 8%.**

**Fix:** include uninstalled users with revenue = 0 (ITT analysis).

## 9.4 Detect Survivorship Ghosts

```sql
SELECT variant, COUNT(*) AS n_assigned, SUM(active) AS n_active
FROM assignments a
LEFT JOIN active_users u USING (user_id)
GROUP BY variant;
```

If `n_active / n_assigned` differs significantly across arms, you're at risk.

---

# 10. Funnel Bugs and Counterfactual Triggers

## 10.1 Funnel Metrics

`signup → first_action → repeat_action → purchase`. Each step is a conversion rate. Treatment can shift each step independently, and bugs at any step affect downstream rates.

## 10.2 Counterfactual Trigger Bug

A trigger condition that **only the treatment arm can satisfy** silently filters out control users.

**Wrong:**
```
trigger: "saw the new banner"
```
Only treatment has the new banner. Control can never trigger. Triggered analysis would be 100% treatment.

**Right:**
```
trigger: "visited the page where the new banner would have appeared"
```
Both arms can trigger.

## 10.3 Diagnostic

Run `trigger_rate` by arm. They should be approximately equal (within 1% relative).

---

# 11. Simpson's Paradox in Experiments

## 11.1 The Phenomenon

An effect that's positive **overall** can be negative in **every segment** — and vice versa — when segment sizes differ between arms.

## 11.2 Canonical Example

> Treatment increases conversion overall by +1%. But:
> - **Mobile users:** treatment -2%
> - **Desktop users:** treatment -1%
>
> How? The treatment shifted users toward desktop (which has higher baseline conversion), and the shift in composition dominates the segment-level effects.

## 11.3 Detection

Check per-segment lift alongside the overall:

```python
overall = (df[df.variant=="T"].converted.mean()
         - df[df.variant=="C"].converted.mean())

segment_lifts = df.groupby("segment").apply(
    lambda g: g[g.variant=="T"].converted.mean()
            - g[g.variant=="C"].converted.mean()
)

if (overall > 0) and (segment_lifts < 0).all():
    print("⚠ Simpson's paradox suspected")
```

## 11.4 Resolution

- Run **post-stratification** to weight segments by their natural distribution.
- Run **segment-fixed regression** (controls for segment composition).
- Often this surfaces the real story: "treatment doesn't help conversion, it just moves users into a higher-converting segment."

---

# 12. The Investigation Workflow

## 12.1 The 7-Step Playbook

```
┌─────────────────────────────────────────────────────────────────┐
│        WHEN AN EXPERIMENT RESULT LOOKS WRONG: PLAYBOOK          │
│                                                                 │
│  1. CHECK SRM.                                                  │
│     Overall, by country, device, app-version, login state.     │
│                                                                 │
│  2. CHECK EXPOSURE COUNTS.                                      │
│     Are per-user exposures the same shape per arm?             │
│                                                                 │
│  3. CHECK PRE-PERIOD BALANCE.                                   │
│     |SMD| for top covariates by arm.                           │
│                                                                 │
│  4. CHECK SURVIVAL.                                             │
│     Active rate, churn rate, error rate per arm.               │
│                                                                 │
│  5. CHECK DAILY TIME-SERIES OF THE EFFECT.                     │
│     Does it look like novelty/primacy? Sudden jump? Drift?     │
│                                                                 │
│  6. CHECK SEGMENT-LEVEL LIFTS.                                  │
│     Simpson? Is the effect concentrated in one weird segment?  │
│                                                                 │
│  7. RECOMPUTE WITH UNADJUSTED ESTIMATOR.                       │
│     Toggle CUPED off. Does the effect persist?                 │
│                                                                 │
│  Only after all 7 are clean: trust the result.                 │
└─────────────────────────────────────────────────────────────────┘
```

## 12.2 What to Write in the Post-Mortem

| Section | Content |
|---------|---------|
| **Symptom** | What did the dashboard say? |
| **Initial hypothesis** | What did the team think was going on? |
| **Diagnostics run** | List of queries/checks (with code links) |
| **Root cause** | The actual bug, with evidence |
| **Resolution** | What changed |
| **Prevention** | What the platform now does automatically to catch this in the future |

This document is gold for interview storytelling.

---

# 13. Anomaly Detection Tooling

## 13.1 Automated Checks the Platform Should Run

Every experiment, every day, automatically:

| Check | Threshold | Severity |
|-------|-----------|---------|
| SRM overall | p < 0.001 | Red — block analysis |
| SRM by segment | p < 0.001 in any segment | Yellow — investigate |
| Pre-period balance SMD | |SMD| > 0.10 on key covariates | Yellow |
| Trigger rate balance | Relative diff > 1% | Yellow |
| Survival rate balance | Relative diff > 0.5% | Yellow |
| Duplicate event rate | Relative diff > 0.1% | Yellow |
| Late-data lag | > 10% events arriving after window | Yellow |
| Daily effect stability | Std/Mean > 2 across days | Yellow |
| Variant-specific error rate | > 0.5% relative | Red |
| Segment-level effect direction inconsistency | Simpson check fails | Yellow |

## 13.2 The Dashboard Should Show

- A **traffic light** per check (green/yellow/red).
- **Drill-downs** to the underlying diagnostic query.
- **Historical context** — "is this anomaly typical or unusual?"

## 13.3 Alerts vs Blockers

| Issue | Alert | Block analysis |
|-------|------|---------------|
| SRM p < 0.001 | Yes | Yes |
| Late data > 10% | Yes | No (warn, allow opt-in) |
| Novelty (effect day-1 vs day-7 diff > 30%) | Yes | No |
| Daily effect std/mean > 2 | Yes | No |

---

# 14. Interview Questions with Strong Answers

## Q1: "Your A/B test shows treatment lift of +2.3%, p=0.001. What checks do you run before recommending ship?"

> Seven checks. **First, SRM** — chi-square on observed vs expected variant counts overall, then stratified by country, device, app version, and logged-in state. Any p < 0.001 is a block. **Second, exposure counts** — do treatment and control users have the same shape of exposure-per-user distributions? **Third, pre-period balance** — compute SMD for top covariates (sessions/user, revenue/user, tenure) and flag |SMD| > 0.1. **Fourth, survival** — is the active-user rate the same in both arms? A bug that crashes treatment users could artificially inflate effects via survivorship. **Fifth, daily time-series** — plot the effect day-by-day; novelty looks like a decay, primacy looks like a U-shape. **Sixth, segments** — compute lift by major segments to catch Simpson's paradox. **Seventh, toggle CUPED off** and recompute — if the effect only exists with CUPED, the covariate may be contaminated. Only after all seven are clean do I trust the result.

## Q2: "You see SRM with p=0.0008. Walk me through your investigation."

> The walk-down order: bucketing, filtering, traffic. **First, bucketing audit** — pull 1000 random user IDs from each arm and re-run the hash function in code; do the assignments match what the platform recorded? If not, the bucketing service has a bug. **Second, filter audit** — list every filter applied between assignment and analysis. Each one is a potential leak. Common culprits: country filter applied after assignment, app-version filter, eligibility re-check. **Third, downstream traffic audit** — look at edge logs, CDN logs, error rates per arm. A treatment that crashes the page sends users to a fallback that's logged differently. **Fourth, stratify SRM by every available dimension** — country, device, browser, app version, time-of-day, time-since-signup. The dimension that explains the imbalance often points directly at the bug. I document each step in a post-mortem so the platform team can add an automated check for this failure mode going forward.

## Q3: "What is a 'ghost lift' and have you seen one?"

> A ghost lift is a statistically significant effect that isn't real — it's an artifact of measurement. The five common types: survivorship (treatment crashes some users, the remaining users are higher-quality by selection), exposure counting (treatment double-logs an event), selection filters applied post-randomization, timing drift (treatment ramped at a different time than control measurement), and logging schema drift (new code emits events the metric pipeline silently accepts). I diagnose by toggling the simplest possible estimator (no CUPED, no triggering) and rerunning. If the lift survives the simplification, it's more likely real. If it disappears, one of the layers introduced bias. I write up every ghost lift as a post-mortem so the platform learns automatically.

## Q4: "Treatment shows a strong day-1 lift of +8% that decays to +0.5% by day 7. Ship or not?"

> Almost certainly a novelty effect. The pattern of large initial effect with rapid decay is classic — users click new buttons, see new layouts, just because they're new. I would not ship on day-7 data alone. My recommendation: extend the experiment to at least day 14, ideally day 21. Compare the day-14-to-21 effect to a long-running holdout if available to validate steady state. Also slice by user tenure — if the day-7 effect is real for new users but novelty for established users, you may have two different stories and the ship decision differs by population. Don't conflate "p < 0.05" with "effect is durable" — those are different questions.

## Q5: "What is Simpson's paradox and how would you detect it in an A/B test?"

> Simpson's paradox is when a treatment effect that's positive overall is negative in every segment, or vice versa. It happens when the treatment shifts users' composition into higher-baseline-rate segments. Example: treatment increases conversion +1% overall but -2% on mobile and -1% on desktop. The mechanism is that treatment moved users toward desktop (higher baseline conversion), and the composition shift overwhelms the per-segment effects. To detect: compute the lift overall and the lift per segment. If the signs differ between overall and segments, suspect Simpson. To resolve: report both views — overall (the population-level effect) and per-segment (the conditional effect). Often this surfaces a meaningful story that flat overall numbers hide. Post-stratification can also re-weight to a fixed segment distribution.

## Q6: "How would you debug an experiment where the treatment shows a lift but only on mobile?"

> Several hypotheses. **Hypothesis 1 — Real heterogeneity:** the feature genuinely helps mobile users more. To confirm, check whether the mobile-specific effect is consistent across countries and time, and whether it has a plausible mechanism. **Hypothesis 2 — Logging asymmetry:** mobile and desktop have different SDKs and log differently; maybe the treatment is correctly captured on mobile but not on desktop. Compare event volumes pre-experiment for control on both platforms — should be stable. **Hypothesis 3 — Mobile bot traffic:** crawlers and click-fraud often spike on mobile. Stratify by user-agent quality score. **Hypothesis 4 — App version interaction:** the new feature is only enabled in newer app versions, which skew younger / more engaged. Stratify by app version. **Hypothesis 5 — Selection on retention:** the experiment kept different users on each platform. Check pre-period balance per platform. I'd report the full diagnostic and let the team see whether the mobile-only effect is interesting or suspicious.

## Q7: "Late-arriving data: walk me through how you handle it."

> Two practices. **First, wait windows** — daily analysis runs 6 hours after end-of-day so most late events have landed. For mobile-heavy products, 24-48 hours is more typical because mobile clients batch uploads on Wi-Fi. **Second, late-data backfill** — re-run analysis daily for the last 7 days so any newly-arrived events for day-N-7 are incorporated. The platform should monitor the **lag distribution** per arm; if treatment and control have different lag distributions (e.g., treatment crashes on weak networks so retries are slower), the analysis is biased. Diagnostic: histogram of `event_ts → ingest_ts` lag by variant; should be near-identical. I'd also expose a "data completeness percentage" on the dashboard so analysts know not to make decisions on partial data.

## Q8: "What's the most common bug you see in experiment exposure logging?"

> Variant mismatch — the variant logged at exposure time doesn't match what was actually served. Causes: client-side caching of an older variant, server-side feature flag race conditions, multiple eligibility checks running independently and disagreeing. The diagnostic is end-to-end: instrument the actual delivery path (the response that reached the user) with the variant served, and reconcile that against the assignment audit log. Any divergence is a logging bug. The platform should monitor "served variant matches assigned variant" as a daily SLO; any drop below 99.5% is an alert. A close second is duplicate exposure events from retries or multiple SDKs firing — solved with idempotent dedup keys at ingest.

## Q9: "Tell me about a time you caught a bug in a measurement system."

> A search ranking experiment was showing +3% CTR lift consistently for two weeks. Everyone wanted to ship. I noticed the SRM was borderline (p=0.008, not red but not green). Stratifying by country revealed the imbalance was almost entirely in India, where treatment had 8% more users. Stratifying further by app version: the treatment was rolled out to v5.0+ users, but the eligibility check in India was running an older config that allowed v4.8 users into treatment. Effectively, the treatment population in India was a mix of properly-bucketed users plus extra v4.8 users. The v4.8 users had higher baseline CTR (older app, heavier users). The "+3% lift" was largely the composition shift. After fixing the eligibility check and rerunning, the true treatment effect was +0.4%. We didn't ship. The post-mortem led to a platform check: "treatment population composition matches assignment composition by app version" became an automated daily test.

## Q10: "How do you build trust in experimentation results across a company?"

> Three things. **First, defaults that prevent the common mistakes** — SRM gates analysis, CUPED is default-on, multiple-testing is default-on, primary metric is locked at start. Make the right thing the easy thing. **Second, reproducibility** — every result keyed by `(config, data snapshot, code version)`, and anyone can re-run any analysis at any time. **Third, post-mortems for every failure** — when an experiment ships and the metric doesn't move post-launch, or when a result is later proven wrong, write up what happened and the platform learns. Over time, the platform accumulates dozens of these post-mortems, each one resulting in a new automated check. After a year of this, junior analysts can't ship a wrong result because the platform won't let them. That's how trust scales.

---

# 15. Key Takeaways

```
┌────────────────────────────────────────────────────────────────────┐
│        DEBUGGING MEASUREMENT — INTERVIEW TAKEAWAYS                  │
│                                                                     │
│  1. 1 in 5 experiments has a measurement problem. The senior IC's │
│     job is to find it before shipping wrong decisions.             │
│                                                                     │
│  2. ALWAYS check SRM first — overall and stratified by segment.   │
│                                                                     │
│  3. Exposure ≠ assignment. Counterfactual triggers must fire in   │
│     both arms.                                                     │
│                                                                     │
│  4. Late data, duplicates, and survivorship are the silent killers│
│     of false lifts.                                                │
│                                                                     │
│  5. Novelty (decay) and primacy (recovery) require longer        │
│     experiments or holdouts.                                       │
│                                                                     │
│  6. Simpson's paradox: overall effect can flip sign vs segments.  │
│     Always slice.                                                  │
│                                                                     │
│  7. Ghost lifts: real-looking p < 0.05 effects that aren't real.  │
│     Toggle CUPED off, simplify the estimator, see if it survives. │
│                                                                     │
│  8. The trust hierarchy: randomization → exposure → metric → data │
│     → analysis → effect. Walk it in order.                         │
│                                                                     │
│  9. Post-mortems are the platform's institutional memory. Every   │
│     bug becomes a future automated check.                          │
│                                                                     │
│  10. Trust is the product. Statistics is the means.                │
└────────────────────────────────────────────────────────────────────┘
```

**One-liner for interviews:**

> *"Senior experimentation work is half statistics and half debugging. Most surprising results are bugs, not breakthroughs. The job is to know exactly which bug to look for first."*

---

**Further reading**

- Crook, Frasca, Kohavi, Longbotham (2009) — *Seven Pitfalls to Avoid when Running Controlled Experiments on the Web*
- Fabijan et al. (2019) — *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments*
- Kohavi, Tang, Xu (2020) — *Trustworthy Online Controlled Experiments* (ch. on diagnosing surprising results)
- Dmitriev et al. (2017) — *A Dirty Dozen: Twelve Common Metric Interpretation Pitfalls in Online Controlled Experiments*
