# P14 — Marketplace Experimentation Lab (Synthetic A/B + Causal Platform)

**Candidate:** Rahul Sharma | **Role:** Data Scientist / Research Engineer (Experimentation)
**Project Type:** Personal / Portfolio | **Codebase:** `platform_exp/`
**Resume Line:** *"Built a self-contained marketplace experimentation platform (FastAPI + pandas) that turns experiment config, assignments, and events into reproducible readouts — ITT/triggered/CACE estimands, SRM and balance diagnostics, sequential monitoring, DiD/IPW causal triangulation, and a decision card — validated against 13 injected failure-mode scenarios."*
**Document Scope:** Complete understanding of the project — architecture, statistics, code map, design decisions, debugging war-stories, and interview answers.

> Companion learning notes (same repo, `interview_prep/learning/`):
> `33_causal_inference_and_experimentation`, `24_statistics_and_ab_testing`,
> `40_cuped_and_variance_reduction`, `41_experimentation_platform_systems`,
> `42_switchback_geo_cluster_designs`, `43_debugging_measurement_systems`,
> `44_econometrics_deep_dive`, `45_marketplace_science`, `47_deep_statistical_reasoning`.
> This doc is the project-specific glue that maps those concepts onto real code.

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview-star)
2. [Architecture & Code Map](#2-architecture--code-map)
3. [Deep Technical Walkthrough](#3-deep-technical-walkthrough)
   - 3.1 [Synthetic Generator & Scenarios](#31-synthetic-generator--scenarios)
   - 3.2 [Assignment Engine](#32-assignment-engine)
   - 3.3 [Metrics & Estimators](#33-metrics--estimators)
   - 3.4 [Inference: CIs, Power, Sequential Looks](#34-inference-cis-power-sequential-looks)
   - 3.5 [Reliability Diagnostics](#35-reliability-diagnostics)
   - 3.6 [Triggered Estimands & CACE](#36-triggered-estimands--cace)
   - 3.7 [Causal Triangulation (DiD / IPW / Uplift)](#37-causal-triangulation-did--ipw--uplift)
   - 3.8 [Decision Rules Engine](#38-decision-rules-engine)
   - 3.9 [Reproducibility & Manifest](#39-reproducibility--manifest)
   - 3.10 [Olist End-to-End Replay](#310-olist-end-to-end-replay)
   - 3.11 [Frontend / Dashboard](#311-frontend--dashboard)
4. [Scenario Coverage Table](#4-scenario-coverage-table)
5. [Topics You Must Know](#5-topics-you-must-know)
6. [Interview Questions & Model Answers](#6-interview-questions--model-answers)
7. [Debugging War-Stories](#7-debugging-war-stories)
8. [Red Flags & How to Handle](#8-red-flags--how-to-handle)
9. [Key Takeaways](#9-key-takeaways)

---

# 1. Project Overview (STAR)

## 1.1 STAR Summary

| STAR | Detail |
|------|--------|
| **Situation** | Experimentation platforms are usually judged on whether they ship the right verdict when something is *wrong* — SRM, logging regressions, interference, Simpson's paradox. You can't learn that from a single clean A/B test, and you can't safely break production to practice. |
| **Task** | Build a self-contained lab that (a) **generates** marketplace experiments with *known* ground truth and *injectable* defects, (b) runs the full readout pipeline (assignment → events → metrics → inference → diagnostics → causal add-ons → decision), and (c) proves the platform **detects** each injected defect. Make every run reproducible and reviewable through an API + dashboard. |
| **Action** | (1) Wrote a deterministic NumPy generator producing users, assignments, and event logs with 13 scenario plugins. (2) Implemented estimators from first principles (Welch, delta-method ratios, cluster-level SE, CUPED). (3) Built a diagnostics suite (SRM overall/segmented/**cluster-aware**, standardized-difference balance, logging health incl. temporal regression, trigger overlap, interference). (4) Added observational triangulation (panelized DiD + event study, Hájek IPW with influence-function SE, E-value, T-/X-learner). (5) Wrapped everything in a manifest with a SHA-256 verify path. (6) Shipped a vanilla-JS decision-narrative dashboard and a signature test that asserts each scenario's fingerprint. |
| **Result** | 13/13 scenarios produce their expected signature (asserted in `tests/test_scenario_signatures.py`); a golden test pins the healthy-lift readout; the demo + `verify` CLI reproduce byte-identical manifests. The dashboard renders a verdict-first narrative (Ship / Investigate / Hold) with RCT–DiD–IPW triangulation, and `make ci` (install + test + demo + verify) is green. |

## 1.2 Why It's Interesting (the "research engineer" angle)

The hard part is **not** computing a t-test. It's the *judgement layer*: knowing **which estimand answers the question**, knowing **when a diagnostic invalidates the estimate**, and encoding that into rules that don't fire false alarms on healthy experiments yet reliably catch real defects. Two concrete examples that this project gets right (and that most "dump stats to HTML" demos get wrong):

1. **Triggered analysis is post-treatment conditioning.** A treatment that improves search *changes who triggers*, so the triggered cohort is no longer randomized. The platform reports ITT and triggered side-by-side, flags overlap gaps, and the decision card only escalates when the gap exceeds the same threshold the triggering health check uses.
2. **Cluster-randomized ≠ user-randomized.** When markets are the randomization unit, a *user-level* SRM χ² and a *user-level* covariate-balance check will both "fail" by construction (each market is wholly one arm). The platform detects cluster designs and switches SRM **and** balance to the cluster level, and suppresses user-level segmented SRM. Cluster-robust standard errors — not a launch block — are the remedy for residual between-cluster imbalance.

## 1.3 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Simulation | NumPy, pandas | Deterministic users/assignments/events + injectable defects |
| Estimators | NumPy, SciPy, statsmodels | Welch t, delta-method ratio, OLS interactions, χ² SRM |
| Causal | scikit-learn | Logistic propensity, IPW, NN matching, RandomForest T-/X-learners |
| Config | YAML + dataclasses | Versioned experiment + scenario configs |
| API | FastAPI + Uvicorn | `/api/run`, `/api/olist/run`, `/api/manifest/verify`, SPA routes |
| Frontend | Vanilla ES modules + SVG | Decision-narrative dashboard, no build step, no JS deps |
| Reproducibility | hashlib (SHA-256), git SHA | Analysis manifest + `verify` CLI |
| Tests | pytest | Golden readout + 12-scenario signature suite |
| Tooling | Makefile, requirements.lock, GitHub Actions | `make demo/test/serve/verify/ci` |

---

# 2. Architecture & Code Map

```
                 MARKETPLACE EXPERIMENTATION LAB — PIPELINE
 ============================================================================

  CONFIG (YAML)                GENERATE                    ANALYZE
  +----------------+      +--------------------+     +----------------------+
  | marketplace_*  |      | generator.py       |     | metrics.py           |
  | scenarios/*.yaml|---->| - users            |---->| - mean_difference    |
  | randomization: |      | - assignment.py    |     | - ratio_difference   |
  |  individual /  |      | - events (logs)    |     | - cuped, cluster SE  |
  |  stratified /  |      | - scenario_plugins |     | - power / MDE        |
  |  cluster       |      |   inject defects   |     +----------+-----------+
  +----------------+      +---------+----------+                |
                                    |                           v
                                    |              +----------------------+
                                    |              | diagnostics.py       |
                                    |              | SRM (user/cluster),  |
                                    |              | balance, logging,    |
                                    |              | triggering,          |
                                    |              | interference         |
                                    |              +----------+-----------+
                                    v                         |
                          +--------------------+              v
                          | causal.py          |   +----------------------+
                          | DiD + event study  |   | stats.py             |
                          | IPW (Hajek) + CI   |   | sequential_looks     |
                          | E-value, overlap   |   | hte, interactions,   |
                          | T-/X-learner       |   | cace_wald            |
                          +---------+----------+   +----------+-----------+
                                    |                         |
                                    +-----------+-------------+
                                                v
                                    +----------------------+
                                    | pipeline.py          |
                                    | run_scenario()       |
                                    | + decision.py        |
                                    | + manifest (SHA-256) |
                                    +----------+-----------+
                                               |
                       +-----------------------+----------------------+
                       v                       v                      v
                  cli.py (CLI)          api.py (FastAPI)        frontend/ (SPA)
                  run/demo/verify       /api/run, /api/...      decision narrative
```

**Module responsibilities (`platform_exp/exp_platform/`):**

| Module | Responsibility |
|--------|----------------|
| `config.py` | Dataclasses for experiment + randomization config; YAML loader. |
| `generator.py` | Deterministic synthetic users/events; scenario corruption; assignment audit. |
| `scenario_plugins.py` | Registry of the 13 scenarios and their parameters. |
| `assignment.py` | Hash-salted individual, stratified, and cluster assignment + audit. |
| `metrics.py` | Event→user join, metric specs, per-metric estimate orchestration. |
| `stats.py` | Estimators (Welch, delta-method ratio, cluster, CUPED), SRM χ², SMD, power/MDE, HTE, interactions, CACE, sequential looks. |
| `diagnostics.py` | Health checks + cluster-aware SRM/balance + summary rollup. |
| `causal.py` | DiD/event study, propensity IPW + matching, E-value, overlap, uplift. |
| `secondary.py` | Olist canonical adapter + observational DiD readout. |
| `decision.py` | Ship/Investigate/Hold rules with severities + doc links. |
| `pipeline.py` | Orchestrates a full run + builds the reproducibility manifest. |
| `api.py` / `cli.py` | HTTP and command-line entry points. |
| `help_docs.py` | Server-side Markdown→HTML for in-app playbook links. |

---

# 3. Deep Technical Walkthrough

## 3.1 Synthetic Generator & Scenarios

The generator (`generator.py`) is the **source of truth** because it knows the true effect. It draws users with pre-period covariates (`pre_spend`, `pre_purchase_rate`, `tenure_days`, `price_sensitivity`, `market`, `device`, `acquisition_channel`, `is_loyal`), assigns arms, then simulates an event log (impressions → clicks → carts → purchases → sessions) over a 14-day post window. Each **scenario plugin** corrupts exactly one part of that chain so the defect's fingerprint is known in advance:

- `null` injects no effect (false-positive control).
- `lift` injects a genuine positive effect with valid randomization.
- `srm` corrupts the assignment service so the realized treatment share drifts.
- `logging_bug` under-logs treatment clicks and duplicates control purchases.
- `cluster_lift` randomizes whole **markets** (cluster design) with a real effect.

Key design choice: **determinism**. Everything is seeded; the same `(scenario, n_users, seed)` reproduces the same data, which is what makes the golden test and manifest verification meaningful.

## 3.2 Assignment Engine

`assignment.py` supports three randomization kinds, all via a salted hash of the unit id (stable, stateless, reproducible):

- **Individual:** hash(user + salt) → bucket by arm weights.
- **Stratified:** hash within strata keys so each stratum matches target shares (variance reduction + guaranteed balance on strata).
- **Cluster:** assign at the cluster unit (e.g., `market`); every user inherits its cluster's arm. The audit records `kind`, `n_clusters`, and `cluster_col` — these survive scenario corruption (a bug I had to fix; see §7).

## 3.3 Metrics & Estimators

All estimators live in `stats.py` and are implemented from first principles (no black-box wrappers), which is the point for an interview.

**Welch's two-sample t (means)** — unequal variances, Satterthwaite dof:

$$
SE = \sqrt{\tfrac{s_c^2}{n_c} + \tfrac{s_t^2}{n_t}}, \quad
   \nu = \frac{(s_c^2/n_c + s_t^2/n_t)^2}{\frac{(s_c^2/n_c)^2}{n_c-1} + \frac{(s_t^2/n_t)^2}{n_t-1}}
$$

**Delta-method ratio metric** (e.g., CTR = clicks/impressions where the unit is the *user*): the variance of a ratio of means uses the linearized influence function
$\;g_i = (\text{num}_i - r\cdot \text{den}_i)/\overline{\text{den}}$, then $\mathrm{Var}(\hat r) = \mathrm{Var}(g)/n$. This is the correct treatment for **ratio metrics with a random denominator** — a classic interview trap (you can't just t-test the per-user ratio).

**Cluster-level SE:** aggregate to cluster means (or cluster sums for ratios) first, then run the same Welch/delta-method across clusters. Effective n becomes the number of clusters, which is why cluster designs are low-powered.

**CUPED:** $ y^{cuped}_i = y_i - \theta (x_i - \bar x) $ with $ \theta = \mathrm{Cov}(y,x)/\mathrm{Var}(x) $. Reported with realized `variance_reduction` (≈ $\rho^2$).

## 3.4 Inference: CIs, Power, Sequential Looks

- **CIs:** t-based for means, normal for delta-method ratios.
- **Power / MDE:** $ \text{MDE} = (z_{\alpha/2} + z_{power})\sqrt{2\sigma^2/n} $ — the planning sanity check.
- **Sequential looks (`sequential_looks`):** computes z at each information fraction $t$ and compares to **two** reference boundaries: an O'Brien–Fleming approximation $z_{\alpha/2}/\sqrt{t}$ (spends little α early) and a **Bonferroni** boundary $z_{1-\alpha/(2K)}$ over $K$ planned looks. This is the "don't peek naively" guardrail — naive repeated testing inflates Type-I error well above α.

## 3.5 Reliability Diagnostics

`diagnostics.py` runs six checks and rolls them into a summary status that the decision card consumes:

1. **Overall SRM** — χ² of observed vs expected arm counts. **Cluster-aware:** for cluster designs it tests *cluster* counts, not user counts.
2. **Segmented SRM** — SRM within `market`/`device`/`channel` to localize an assignment bug. **Suppressed for cluster designs** (each cluster is wholly one arm → guaranteed false "fail").
3. **Balance** — max |standardized mean difference| on pre-period covariates. **Cluster-aware:** computed on cluster-mean covariates, capped at "warn" (residual imbalance is handled by cluster-robust SE).
4. **Logging health** — duplicate event/purchase rates, cross-arm click/purchase-rate gaps, **and a mid-experiment temporal click regression** (early vs late click-rate ratio between arms) to catch instrumentation drift.
5. **Triggering** — search/cart trigger-rate gap between arms (post-treatment denominator warning).
6. **Interference** — SD of market-level conversion lift (large spread = spillover/HTE).

## 3.6 Triggered Estimands & CACE

- **ITT** answers "effect of *assigning* treatment" — always randomization-valid.
- **Triggered** conditions on a post-assignment behavior (e.g., searched). Reported with overlap diagnostics; the warning fires only when the cross-arm overlap gap exceeds the same threshold the triggering health check uses (kept consistent at 0.03 — see §7).
- **CACE / Wald IV** (`cace_wald`): $ \text{CACE} = \frac{\text{ITT}_{outcome}}{\text{ITT}_{exposure}} $ with a delta-method SE, labeled exploratory and gated on first-stage movement + monotonicity assumptions.

## 3.7 Causal Triangulation (DiD / IPW / Uplift)

For the **observational** lens (and Olist), `causal.py` implements:

- **DiD + event study:** 2×2 difference-in-differences plus per-relative-day gaps with a baseline at $t=-1$ and a pre-trend slope check (parallel-trends diagnostic).
- **IPW (Hájek / stabilized):** weights $1/\hat e$ for treated and $1/(1-\hat e)$ for control, **self-normalized** (sums of weights in the denominator), with an **influence-function SE**. Propensities clipped to [0.02, 0.98].
- **Overlap diagnostics:** common-support range + per-arm propensity histogram + mass-in-support.
- **E-value (VanderWeele):** $ E = RR + \sqrt{RR(RR-1)} $ — the minimum confounder strength (on the risk-ratio scale) needed to explain away the association. Sensitivity, not proof.
- **T-/X-learner uplift:** RandomForest baselines for targeting exploration, explicitly *not* causal evidence.

The **triangulation card** lines up RCT-ITT, DiD-KPI, and IPW for the same KPI and checks **direction agreement** — the headline "3 of 3 estimators agree" line on the dashboard. Agreement across designs with different assumptions is the strongest practical evidence.

## 3.8 Decision Rules Engine

`decision.py` converts numbers into an advisory verdict with explicit **severities**:

- **Hold** if any reliability check *fails* (you cannot trust the estimate).
- **Investigate** if there are warnings, the primary KPI CI doesn't clear zero, or triggered overlap moves.
- **Ship** only when the primary KPI interval is positive *and* diagnostics pass.

Each fired rule carries a `reason` and a `/help/...` doc link (rendered server-side). Crucially the verdict is computed from **severity fields**, not string-matching — a refactor that fixed an ordering bug where "Investigate" could mask a "Hold".

## 3.9 Reproducibility & Manifest

`pipeline.py` attaches a manifest to every readout: SHA-256 of the resolved config, the seed, the git SHA, the Python/library versions, and an `analysis_hash`. `cli.py verify` (and `GET /api/manifest/verify?path=`) recompute the hash and assert it matches — so a saved readout can be proven to come from a specific config + code state. The dashboard's Reproducibility tab shows a live "✓ Manifest verified" badge.

## 3.10 Olist End-to-End Replay

`secondary.py` adapts the bundled Olist e-commerce CSVs into the platform's canonical event schema (vectorized with `pd.concat`, not a Python loop), then runs an **observational** DiD using a policy pivot date and a configurable set of treated states (`treated_states=SP,RJ,...`). It is explicitly labeled *observational replay with synthetic assignment* — the synthetic generator remains the causal source of truth; Olist is the realism layer.

## 3.11 Frontend / Dashboard

A **build-step-free** SPA in `frontend/` (vanilla ES modules + hand-rolled SVG charts):

- `scripts/` — `state.js` (pub/sub store), `router.js` (hash/`?tab=` routing + keyboard shortcuts), `api.js`, `format.js`, `ui.js`, `main.js` (bootstrap), plus `components/` (decision card, reliability strip, forest plot, event-study, sequential, overlap histogram, truth-vs-detection, manifest, scenario library) and `tabs/` (one renderer per tab).
- `styles/` — design tokens, base, components, charts, and a print stylesheet; dark/light/system themes.
- Verdict-first information architecture, shareable URLs, **Compare mode** (two scenarios side-by-side), and a truth-vs-detection panel that scores each scenario's injected defect against what the platform caught.

---

# 4. Scenario Coverage Table

| Scenario | Injected truth | Expected platform signature |
|----------|----------------|------------------------------|
| `null` | No effect | SRM passes, balance passes, primary CI covers 0 |
| `lift` | Real positive effect | SRM passes, CUPED reduces variance, revenue+conversion up → **Ship** |
| `srm` | Assignment defect | Overall SRM fails, device segmented SRM sharp → **Hold** |
| `logging_bug` | Under-logged clicks + dup purchases | Logging warns/fails, duplicate rate > 0 |
| `simpson` | Non-random assignment | Balance fails, propensity disagrees with naive readout |
| `cluster_interference` | Market spillovers | High market-level lift SD; cluster design recommended |
| `ranking_exposure_bug` | Offline-only gain | NDCG up but exposure logging gap negative; online invalid |
| `novelty_effect` | Early lift decays | Sequential early > final movement → investigate |
| `primacy_effect` | Clicks up, value flat | CTR positive but revenue/conversion misses threshold |
| `seasonality_confound` | Assignment ~ seasonality | Balance fails, propensity adjustment disagrees |
| `instrumentation_regression` | Logging drops after day 8 | Temporal click regression flagged in logging health |
| `segment_interference` | Loyal-user spillovers | HTE/interaction panel flags heterogeneity |
| `cluster_lift` | Market-cluster effect | Cluster-level SRM passes, cluster SE used → wide CI → **Investigate** |

`tests/test_scenario_signatures.py` encodes a predicate per scenario; CI fails if a fingerprint disappears.

---

# 5. Topics You Must Know

1. **ITT vs Triggered vs CACE** — when each is valid; post-treatment conditioning bias.
2. **Delta method for ratio metrics** — why per-user ratios need the influence-function variance.
3. **SRM** — what it is, why even a 0.1% imbalance with large n is a red flag, and the χ² test.
4. **CUPED** — variance reduction via a pre-period covariate, $\theta$, and the $\rho^2$ intuition.
5. **Sequential testing** — α-spending, O'Brien–Fleming vs Pocock vs Bonferroni, the peeking problem.
6. **DiD** — parallel trends, event-study pre-trends, two-way comparison.
7. **IPW** — propensity overlap/positivity, stabilized (Hájek) weights, clipping.
8. **E-value** — sensitivity to unmeasured confounding (and its limits).
9. **Cluster randomization** — design effect, why few clusters = low power, cluster-robust SE, and why user-level SRM/balance are invalid.
10. **Interference / SUTVA** — marketplace spillovers and how cluster designs mitigate them.

---

# 6. Interview Questions & Model Answers

**Q1. Why not just compute the per-user CTR and run a t-test?**
Because CTR is a ratio of two random quantities (clicks and impressions) and impressions vary per user. The variance of the ratio-of-means is not the variance of per-user ratios; you need the delta method (linearize around the mean: $g_i = (\text{num}_i - r\,\text{den}_i)/\overline{\text{den}}$). The platform's `ratio_difference` does exactly this.

**Q2. The treatment increased the triggered-cohort conversion. Can we report that as the effect?**
Carefully. Triggering (e.g., searching) is a *post-assignment* behavior; if treatment changes who triggers, the triggered cohort is no longer a randomized comparison. We report ITT as the valid headline and triggered as a precision-oriented secondary, with an overlap diagnostic. If the cross-arm overlap gap is large, the triggered estimand is confounded.

**Q3. A cluster-randomized test shows a 1.1 SMD imbalance and user-level SRM p≈1e-139. Is it broken?**
No — that's the design, not a defect. With market-level randomization, every user inherits its market's arm, so user-level SRM and user-level balance compare whole markets and "fail" by construction. The correct diagnostics are cluster-level SRM (cluster counts vs expected) and cluster-mean balance, and residual imbalance across few clusters is absorbed by cluster-robust standard errors. (This is implemented in `diagnostics.py` and was a specific bug I fixed.)

**Q4. How do you stop people from peeking?**
Compute group-sequential boundaries. The platform reports an O'Brien–Fleming boundary $z_{\alpha/2}/\sqrt{t}$ (conservative early, near-nominal at the end) and a Bonferroni boundary over the planned looks, and flags whether the naive z crossed either. Naive repeated 0.05 testing can push Type-I error past 20%.

**Q5. RCT, DiD, and IPW disagree. What now?**
Triangulation works *because* the designs rely on different assumptions (randomization vs parallel trends vs ignorability+overlap). Disagreement is information: check overlap for IPW, pre-trends for DiD, and SRM/balance for the RCT. The decision card already keys "ship" on direction agreement + passing diagnostics.

**Q6. What makes a result reproducible here?**
A manifest: SHA-256 of the resolved config, the seed, git SHA, and library versions, with an `analysis_hash`. `verify` recomputes it. Same inputs + same code ⇒ identical hash, so a saved readout is auditable.

---

# 7. Debugging War-Stories

These are real fixes made while hardening the platform — good "tell me about a bug you found" material.

1. **Flagship `lift` never shipped.** The triggered-overlap warn threshold (0.02) was tighter than the triggering health check (0.03). A natural 2.2pp search-rate gap from a *genuine* lift tripped the stricter threshold and fired an "Investigate" rule, so the healthy demo showed *Investigate* instead of *Ship*. Fix: align both thresholds at 0.03 so the decision rule keys off the same diagnostic.

2. **`cluster_lift` always Held.** User-level SRM and user-level balance were applied to a market-randomized design and "failed" catastrophically (nyc: 0 control / 630 treatment, p≈1e-139; SMD 1.1). Fix: detect cluster designs, run **cluster-level** SRM and balance, suppress user-level segmented SRM, and cap cluster balance at "warn" (cluster-robust SE is the remedy). Result: a statistically honest *Investigate* (wide CI from only 8 clusters) instead of a spurious Hold.

3. **Dashboard tabs were blank.** The router's tab-click handler updated visibility but never invoked the render callback, so every tab except the initial one rendered empty. Fix: have `navigate()` call the registered `onTabChange` callback.

4. **A persistent "Compare mode" bar.** A `.compare-bar { display:flex }` rule overrode the HTML `hidden` attribute, so the bar was always visible. Fix: add `.compare-bar[hidden] { display:none }`.

5. **Audit fields dropped after corruption.** The SRM-corruption path rebuilt the assignment audit and lost `kind` / `n_clusters` / `cluster_col`, which later broke cluster-aware logic. Fix: preserve those fields through corruption.

---

# 8. Red Flags & How to Handle

| Red flag a reviewer might raise | Honest response |
|---------------------------------|------------------|
| "It's synthetic — so what?" | Synthetic ground truth is the *feature*: it's the only way to verify that a platform **catches** SRM, logging bugs, and interference. The Olist path adds real-data realism. |
| "Your uplift models aren't causal." | Correct, and they're labeled *exploratory*. They're for targeting exploration; randomized ITT remains the evidence. |
| "CACE SE ignores covariance." | It's a delta-method approximation, gated on first-stage movement and labeled exploratory — not a launch-grade estimand. |
| "Few-cluster designs are weak." | Exactly why `cluster_lift` lands on *Investigate* with a wide cluster-robust CI — the platform is being honest about low power, not hiding it. |
| "Decision rules are simplistic." | They're advisory and explicitly require product/science review; the value is consistency and traceable reasons, not auto-launch. |

---

# 9. Key Takeaways

- The differentiator is the **judgement layer**: choosing the right estimand, knowing when a diagnostic invalidates an estimate, and encoding that without false alarms.
- **Estimators from first principles** (Welch, delta-method, cluster SE, CUPED, Hájek IPW with influence-function SE) — not library black boxes — so you can defend every number.
- **Design-aware diagnostics**: SRM and balance must respect the randomization unit; cluster designs need cluster-level checks and cluster-robust SE.
- **Triangulation > any single estimator**: agreement across RCT, DiD, and IPW (each with different assumptions) is the strongest practical evidence.
- **Reproducibility is non-negotiable**: deterministic seeds + a verifiable manifest make every readout auditable.
- **Failure-mode tests**: 13 scenarios with asserted signatures turn "trust me" into a green CI.
