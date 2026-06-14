# P01 — Behaviour Scorecard & Automated Risk Scoring | Fibe (EarlySalary)

> **Rahul Sharma** | Data Scientist | Fibe (formerly EarlySalary), Pune, India
> **Duration:** September 2021 – June 2022

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
3. [Monitoring & Governance](#3-monitoring--governance)
4. [Key Metrics & Results](#4-key-metrics--results)
5. [Topics You Must Know (Study Guide)](#5-topics-you-must-know-study-guide)
6. [Interview Questions & Answers (25+)](#6-interview-questions--answers)
7. [Potential Red Flags & How to Handle](#7-potential-red-flags--how-to-handle)
8. [Key Takeaways & Talking Points](#8-key-takeaways--talking-points)

---

## 1. Project Overview

### 1.1 STAR Summary (Interview-Ready)

**Situation**
At Fibe (formerly EarlySalary), a fast-growing digital lending fintech in India, customer risk assessment relied on a slow, semi-manual process. Analysts pulled data from multiple sources, manually computed risk indicators, and assembled scores—taking up to **3 days per batch cycle**. Results were sometimes inconsistent across analysts, and the turnaround made it difficult for the business to respond quickly to lending opportunities or emerging risk events.

**Task**
I was tasked with revamping the entire risk scoring workflow end-to-end: unifying the data pipeline, building a robust and interpretable credit risk model using 1,000+ bureau variables, fully automating the scoring pipeline, and instituting monitoring and governance guardrails—all while ensuring regulatory and business alignment.

**Approach & Action**

| Phase | What I Did |
|-------|-----------|
| **Data Integration** | Partnered with data engineering and business stakeholders to consolidate data from transaction histories, credit bureau records (CIBIL/Experian), and customer behavioral signals into a single unified schema. Cleaned, deduplicated, and reconciled disparate formats. |
| **Feature Engineering** | From 1,000+ raw bureau variables, engineered predictive features—recent spending patterns, payment consistency indices, account age buckets, utilization ratios—and applied WOE/IV-based selection plus monotonic-relationship constraints to ensure regulatory interpretability. |
| **Model Building** | Evaluated logistic regression, random forest, XGBoost, and LightGBM. Selected **logistic regression** for the final scorecard due to its interpretability, monotonicity control, and regulatory acceptance in credit decisions. Achieved **ROC-AUC of 0.94**. |
| **Scorecard Translation** | Converted logistic regression coefficients into a points-based scorecard that business teams and credit policy managers could directly use for decision thresholds. |
| **Automation (Knime)** | Built an end-to-end automated pipeline in Knime—from data ingestion, feature computation, scoring, to output delivery—with built-in quality gates. |
| **Monitoring & Dashboards** | Implemented PSI (Population Stability Index) and CSI (Characteristic Stability Index) monitors. Built interactive dashboards for real-time score distribution tracking and executive reporting. |

**Result**
- **ROC-AUC: 0.94** — strong discriminatory power between defaulters (30+ DPD) and non-defaulters
- **15× turnaround improvement** — from 3 days to a few hours
- Directly drove credit-policy changes adopted by the lending operations team
- Dashboards enabled proactive risk monitoring and early drift detection

---

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FIBE RISK SCORING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐                  │
│  │  Credit Bureau│  │  Transaction │  │  Behavioral      │                  │
│  │  Records      │  │  Histories   │  │  Signals         │                  │
│  │  (CIBIL/      │  │  (Internal   │  │  (App usage,     │                  │
│  │   Experian)   │  │   banking)   │  │   repayment      │                  │
│  └──────┬───────┘  └──────┬───────┘  │   patterns)      │                  │
│         │                  │          └────────┬─────────┘                  │
│         ▼                  ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │              DATA INTEGRATION LAYER                   │                  │
│  │  • Schema unification   • Deduplication               │                  │
│  │  • Missing value treatment  • Type reconciliation     │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │           FEATURE ENGINEERING ENGINE                   │                  │
│  │  • 1000+ raw bureau vars → refined feature set        │                  │
│  │  • WOE binning & IV computation                       │                  │
│  │  • Monotonic relationship validation                  │                  │
│  │  • VIF / correlation / LASSO filtering                │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │          MODEL TRAINING & VALIDATION                  │                  │
│  │  • Logistic Regression (final)                        │                  │
│  │  • Cross-validation (stratified 5-fold)               │                  │
│  │  • Out-of-time validation                             │                  │
│  │  • ROC-AUC: 0.94  |  KS: ~0.75  |  Gini: ~0.88      │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │        SCORECARD TRANSLATION                          │                  │
│  │  • Log-odds → points mapping                          │                  │
│  │  • PDO (Points to Double the Odds) calibration        │                  │
│  │  • Business-interpretable risk bands                  │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │         KNIME AUTOMATION PIPELINE                     │                  │
│  │  • Scheduled data ingestion                           │                  │
│  │  • Automated scoring + quality gates                  │                  │
│  │  • PSI / CSI drift monitors                           │                  │
│  │  • Output delivery to downstream systems              │                  │
│  └──────────────────────┬───────────────────────────────┘                  │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │          DASHBOARDS & GOVERNANCE                      │                  │
│  │  • Score distribution monitoring                      │                  │
│  │  • PSI trend dashboard  • Feature drift alerts        │                  │
│  │  • Executive risk summaries                           │                  │
│  │  • Credit policy decision support                     │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Tech Stack

| Category | Tools / Technologies |
|----------|---------------------|
| **Language** | Python (pandas, NumPy, scikit-learn, statsmodels) |
| **Data Sources** | CIBIL / Experian bureau data, internal transaction DB, behavioral logs |
| **Feature Engineering** | WOE/IV binning (custom + `scorecardpy`), VIF, LASSO |
| **Modeling** | Logistic Regression (statsmodels + sklearn), XGBoost/LightGBM (benchmarked) |
| **Automation** | Knime Analytics Platform (workflow orchestration) |
| **Monitoring** | PSI / CSI computations (custom Python), dashboards |
| **Dashboards** | Tableau / Excel-based executive dashboards |
| **Database** | SQL (PostgreSQL / MySQL for internal data warehouse) |
| **Version Control** | Git |

---

### 1.4 Timeline & Team Structure

| Phase | Timeline | My Role |
|-------|----------|---------|
| Discovery & data audit | Sept – Oct 2021 (6 weeks) | Led data assessment, stakeholder interviews |
| Data integration & cleaning | Oct – Nov 2021 (6 weeks) | Hands-on ETL, schema unification |
| Feature engineering & selection | Nov 2021 – Jan 2022 (8 weeks) | Primary owner of feature pipeline |
| Model development & validation | Jan – Mar 2022 (8 weeks) | Model building, validation, scorecard dev |
| Automation (Knime) & dashboards | Mar – May 2022 (8 weeks) | Pipeline automation, dashboard design |
| Deployment, monitoring & handover | May – Jun 2022 (4 weeks) | Production rollout, documentation |

**Team:** 1 senior data scientist (mentor/reviewer), 2 data engineers (pipeline support), 1 business analyst (domain context), credit policy team (stakeholders). **I was the primary modeler and pipeline builder.**

---

## 2. Deep Technical Walkthrough

### 2.1 Data Sources & Integration Challenges

#### Data Sources

| Source | Key Variables | Volume |
|--------|--------------|--------|
| **Credit Bureau (CIBIL/Experian)** | Credit score, number of accounts, DPD history, enquiry count, utilization, account types, payment history | 1,000+ variables per customer |
| **Internal Transactions** | Loan disbursement history, EMI payment records, bounce rates, prepayment behavior | Millions of transaction rows |
| **Behavioral Signals** | App login frequency, time-of-day patterns, device metadata, repayment reminder interactions | Semi-structured logs |

#### Integration Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Schema mismatch** — Bureau data came in fixed-width/XML formats; internal data was in SQL tables; behavioral data was semi-structured JSON logs | Built a unified ETL pipeline to parse each source into a standardized customer-level feature table with consistent column naming conventions |
| **Temporal alignment** — Bureau snapshots were monthly; transaction data was real-time; behavioral data was event-based | Defined a common observation window (e.g., "features as of date X, target measured at date X + 6 months") to prevent data leakage |
| **Missing data** — Bureau records had ~15-30% missing values for newer/thin-file customers | Applied domain-informed imputation: missing credit score → "No bureau history" indicator variable; missing utilization → median imputation within risk segment; created binary flags for missingness |
| **Duplicate records** — Multiple bureau records per customer due to name/address variations | Fuzzy matching on PAN + phone number + DOB; kept the most recent bureau pull |
| **Volume** — 1,000+ raw bureau variables per customer | Systematic reduction: removed zero-variance → correlated pairs → IV < 0.02 → VIF > 5 → final ~40-60 features |

---

### 2.2 Feature Engineering Details

#### From 1,000+ Bureau Variables to a Final Feature Set

```
1,000+ raw variables
    │
    ▼  Step 1: Remove zero/near-zero variance (< 1% unique values)
   ~700 variables
    │
    ▼  Step 2: Remove high-missing (> 70% missing)
   ~500 variables
    │
    ▼  Step 3: WOE binning + IV computation
   ~200 variables with IV > 0.02
    │
    ▼  Step 4: Correlation filtering (|r| > 0.7 → keep higher IV)
   ~100 variables
    │
    ▼  Step 5: VIF < 5 check (multicollinearity removal)
   ~60 variables
    │
    ▼  Step 6: Monotonic relationship validation
   ~50 variables
    │
    ▼  Step 7: Business review & final LASSO confirmation
   ~40-50 final features in scorecard
```

#### Key Engineered Features

| Feature Category | Examples | Rationale |
|-----------------|----------|-----------|
| **Spending Patterns** | Avg monthly spend (last 3/6/12 months), spend volatility, category-wise spend ratios | Recent financial stress signals |
| **Payment Consistency** | % on-time payments (last 6/12/24 months), longest streak of on-time, DPD frequency | Direct predictor of future default |
| **Account Age** | Age of oldest account, avg account age, % accounts > 3 years | Stability and credit maturity |
| **Utilization** | Overall utilization ratio, max utilization across cards, utilization trend (increasing/decreasing) | High utilization = higher risk |
| **Enquiry Behavior** | # hard enquiries (last 3/6 months), enquiry-to-account ratio | Credit-hungry behavior |
| **Bureau Score Derivatives** | Score bucket, score change (delta over 6 months), score × utilization interaction | Combining bureau intelligence |
| **Behavioral** | App engagement score, avg days between logins, repayment reminder response rate | Soft signals of intent |

#### WOE (Weight of Evidence) Binning — Worked Example

WOE transforms categorical/continuous variables into a metric that captures the predictive relationship with the binary target:

```
WOE_i = ln( (% of Non-Events in bin_i) / (% of Events in bin_i) )
```

**Example: Credit Utilization Ratio**

| Bin | Range | # Good | # Bad | % Good | % Bad | WOE | IV Contribution |
|-----|-------|--------|-------|--------|-------|-----|-----------------|
| 1 | 0–20% | 5,000 | 200 | 0.33 | 0.10 | ln(0.33/0.10) = 1.19 | (0.33–0.10)×1.19 = 0.274 |
| 2 | 20–50% | 4,500 | 400 | 0.30 | 0.20 | ln(0.30/0.20) = 0.41 | (0.30–0.20)×0.41 = 0.041 |
| 3 | 50–80% | 3,500 | 600 | 0.23 | 0.30 | ln(0.23/0.30) = −0.27 | (0.23–0.30)×(−0.27) = 0.019 |
| 4 | 80–100% | 2,000 | 800 | 0.13 | 0.40 | ln(0.13/0.40) = −1.12 | (0.13–0.40)×(−1.12) = 0.302 |
| **Total** | | **15,000** | **2,000** | **1.00** | **1.00** | | **IV = 0.636** |

**Interpretation:** IV = 0.636 → **Strong predictor** (IV > 0.3 is strong). WOE is monotonically decreasing—higher utilization = more negative WOE = higher risk. This is exactly the monotonic pattern regulators expect.

#### IV (Information Value) Thresholds

| IV Range | Predictive Power | Action |
|----------|-----------------|--------|
| < 0.02 | Useless | Drop |
| 0.02 – 0.1 | Weak | Consider dropping |
| 0.1 – 0.3 | Medium | Include |
| 0.3 – 0.5 | Strong | Include |
| > 0.5 | Suspicious (possible overfit) | Investigate |

#### Monotonic Relationship Requirement

In credit risk, regulators and auditors require that each feature's relationship with default probability is **monotonic** — i.e., as the feature value increases (or decreases), the predicted risk should move consistently in one direction.

**Why?** If your model says "people who earn $50K are higher risk than those earning $30K but lower risk than those earning $40K," that's not explainable to a regulator or a customer who was denied credit.

**How I enforced it:**
1. **WOE binning** naturally produces monotonic transformations when bins are merged correctly
2. **Visual inspection** of WOE plots for each variable — ensured smooth monotonic curves
3. **Bin merging** — adjacent bins with non-monotonic WOE were merged until monotonicity was restored
4. **Business validation** — confirmed that the direction of each variable aligned with credit domain knowledge

```python
# Example: Checking monotonicity of WOE values
import numpy as np

def is_monotonic(woe_values):
    """Check if WOE values are monotonically increasing or decreasing."""
    diffs = np.diff(woe_values)
    return np.all(diffs >= 0) or np.all(diffs <= 0)

# Example usage
woe_utilization = [1.19, 0.41, -0.27, -1.12]
print(f"Monotonic: {is_monotonic(woe_utilization)}")  # True — monotonically decreasing
```

---

### 2.3 Why Logistic Regression Over Complex Models

This is a **critical interview question** — expect it every time.

| Criterion | Logistic Regression | XGBoost / Random Forest |
|-----------|-------------------|------------------------|
| **Interpretability** | Full coefficient transparency — each feature's contribution is a weight × WOE | Black-box; SHAP/LIME needed post-hoc |
| **Regulatory acceptance** | Gold standard in credit risk (Basel/RBI norms) | Growing acceptance, but still questioned by auditors |
| **Monotonicity** | Naturally preserved via WOE-transformed inputs | Requires explicit monotonic constraints |
| **Scorecard conversion** | Direct log-odds → points mapping | No natural scorecard form |
| **Audit trail** | Clear: "Feature X contributed Y points to the score" | Complex explanations needed |
| **Model risk** | Low — well-understood failure modes | Higher — prone to overfitting, harder to validate |
| **Performance gap** | AUC 0.94 | AUC ~0.95–0.96 (marginal gain) |
| **Business adoption** | Credit officers understand it | Requires significant training |

**Key talking point:** *"The marginal 1-2% AUC gain from ensemble models was not worth the regulatory and interpretability cost. A logistic regression with 0.94 AUC that can be fully explained to regulators and converted into a simple scorecard is far more valuable in production credit decisioning than a black-box model with 0.96 AUC."*

**Benchmark results from my experiments:**

| Model | ROC-AUC | KS Statistic | Gini |
|-------|---------|-------------|------|
| Logistic Regression (WOE) | 0.94 | ~0.75 | ~0.88 |
| Random Forest | 0.95 | ~0.78 | ~0.90 |
| XGBoost | 0.96 | ~0.80 | ~0.92 |
| LightGBM | 0.955 | ~0.79 | ~0.91 |

---

### 2.4 Model Training Details

#### Target Variable Definition

```
Target = 1 if customer is 30+ Days Past Due (DPD) within 6-month performance window
Target = 0 otherwise

Observation point: Date of feature snapshot
Performance window: 6 months after observation point
```

**Default rate:** ~10-12% (moderate class imbalance)

#### Train/Test/Validation Split Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                    FULL DATASET (TIME-ORDERED)                │
├──────────────────┬──────────────────┬────────────────────────┤
│    TRAIN (60%)   │  VALIDATION (20%)│  OUT-OF-TIME TEST (20%)│
│  Jan 2020 –      │  Jul 2021 –      │  Jan 2022 –            │
│  Jun 2021        │  Dec 2021        │  Mar 2022              │
├──────────────────┴──────────────────┴────────────────────────┤
│  Within TRAIN: 5-fold stratified cross-validation            │
└──────────────────────────────────────────────────────────────┘
```

**Why out-of-time (OOT) validation?**
- Credit risk data is non-stationary—economic conditions change over time
- In-sample and even random hold-out validation can be overly optimistic
- OOT mimics real deployment: model trained on past data, scored on future data
- Regulators specifically ask for OOT performance metrics

#### Hyperparameter Tuning (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

# Logistic Regression with L1/L2 regularization tuning
model = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1, 10, 100],       # Inverse regularization strength
    penalty='l2',                               # L2 preferred for scorecard stability
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    class_weight='balanced',                    # Handle class imbalance
    max_iter=1000,
    solver='lbfgs'
)

model.fit(X_train_woe, y_train)
print(f"Best C: {model.C_[0]}")
print(f"Train AUC: {roc_auc_score(y_train, model.predict_proba(X_train_woe)[:, 1]):.4f}")
print(f"Val AUC:   {roc_auc_score(y_val, model.predict_proba(X_val_woe)[:, 1]):.4f}")
print(f"OOT AUC:   {roc_auc_score(y_oot, model.predict_proba(X_oot_woe)[:, 1]):.4f}")
```

#### Class Imbalance Handling

With ~10-12% default rate, moderate imbalance was addressed by:

1. **`class_weight='balanced'`** in logistic regression — upweights the minority class inversely proportional to frequency
2. **Stratified splitting** — all splits maintained the same default ratio
3. **Did NOT use SMOTE** — in credit risk, synthetic oversampling can create unrealistic borrower profiles; prefer reweighting
4. **Threshold optimization** — optimized the decision threshold on the validation set to maximize the business-relevant metric (e.g., maximize KS, or minimize a cost function of false positives vs. false negatives)

---

### 2.5 Scorecard Development

A **credit scorecard** converts logistic regression output into a points-based system that credit officers can use directly.

#### Scorecard Formula

```
Score = Offset + Σ (WOE_i × β_i × Factor) + Σ (Base_points_i)
```

Where:
- **Factor** = PDO / ln(2)
- **Offset** = Target_Score − (Factor × ln(Target_Odds))
- **PDO** = Points to Double the Odds (industry standard: 20 or 50)
- **β_i** = logistic regression coefficient for feature i

#### Worked Example

```
Assumptions:
  - Target Score = 600  (at odds 50:1, i.e., 2% default rate)
  - PDO = 20            (every 20-point increase = odds double = risk halves)

Factor = 20 / ln(2) = 28.85
Offset = 600 − 28.85 × ln(50) = 600 − 112.88 = 487.12

For a customer with:
  - Credit Utilization WOE = −0.27 (bin 3: 50-80%), coeff β = −1.5
  - Payment History WOE = 0.41 (bin 2: mostly on-time), coeff β = −2.0
  - Enquiry Count WOE = 1.19 (bin 1: low enquiries), coeff β = −0.8

Points from utilization  = (−0.27) × (−1.5) × 28.85 = 11.68
Points from payment hist = (0.41) × (−2.0) × 28.85 = −23.66
Points from enquiries    = (1.19) × (−0.8) × 28.85 = −27.47

Total Score = 487.12 + 11.68 + (−23.66) + (−27.47) + ... (other features)
```

#### Risk Band Mapping

| Score Range | Risk Band | Approval Decision | Approx Default Rate |
|-------------|-----------|-------------------|-------------------|
| 700+ | Very Low Risk | Auto-approve | < 2% |
| 600–699 | Low Risk | Approve with standard terms | 2–5% |
| 500–599 | Medium Risk | Manual review / higher interest | 5–15% |
| 400–499 | High Risk | Decline or collateral required | 15–30% |
| < 400 | Very High Risk | Decline | > 30% |

---

### 2.6 Knime Automation Pipeline

#### Why Knime?

| Reason | Detail |
|--------|--------|
| **Existing infrastructure** | Fibe's analytics team already used Knime; no new tooling approval needed |
| **Visual workflow** | Non-technical stakeholders could understand and audit the pipeline |
| **Scheduling** | Built-in scheduler for batch runs without cron/Airflow setup |
| **Python integration** | Knime nodes can call Python scripts, so all scikit-learn code ran inside Knime |
| **Enterprise features** | Logging, error handling, email notifications on failure |

#### Workflow Design

```
┌─────────────────────────────────────────────────────────┐
│                 KNIME WORKFLOW                           │
│                                                         │
│  [Scheduler Trigger]                                    │
│        │                                                │
│        ▼                                                │
│  [DB Connector] ──→ [SQL Query: Extract raw data]       │
│        │                                                │
│        ▼                                                │
│  [Python Script: Data cleaning & feature engineering]   │
│        │                                                │
│        ▼                                                │
│  [Quality Gate: Check row counts, nulls, distributions] │
│        │                                                │
│        ├── FAIL → [Email Alert] → [STOP]                │
│        │                                                │
│        ▼ PASS                                           │
│  [Python Script: Apply WOE mapping & score]             │
│        │                                                │
│        ▼                                                │
│  [PSI Check: Compare score dist to reference]           │
│        │                                                │
│        ├── PSI > 0.25 → [Alert: Major drift detected]   │
│        ├── PSI > 0.10 → [Warning: Moderate drift]       │
│        │                                                │
│        ▼                                                │
│  [CSI Check: Per-feature stability]                     │
│        │                                                │
│        ▼                                                │
│  [DB Writer: Write scores to production table]          │
│        │                                                │
│        ▼                                                │
│  [Dashboard Refresh: Update Tableau/Excel reports]      │
│        │                                                │
│        ▼                                                │
│  [Email: Send summary report to stakeholders]           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 2.7 Dashboard Design

#### Executive Dashboard Components

| Dashboard Panel | What It Shows | Audience |
|----------------|---------------|----------|
| **Score Distribution** | Histogram of scores, overlaid with previous month | Risk team, management |
| **Approval Rate Trend** | % approved per risk band over time | Credit policy, business |
| **PSI Trend** | Monthly PSI values with threshold lines (0.1 / 0.25) | Model monitoring team |
| **Feature Drift** | Top 5 features with highest CSI values | Data science team |
| **Default Rate by Band** | Actual vs predicted default rate per score band | Model validation |
| **Portfolio Composition** | % of new loans in each risk band | Business, investors |

---

## 3. Monitoring & Governance

### 3.1 PSI (Population Stability Index)

**What it is:** PSI measures how much the distribution of model scores (or any feature) has shifted between two time periods — typically between the development sample and a recent scoring sample.

**Formula:**

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Where:
- **Expected%** = proportion in each bin from the development/reference sample
- **Actual%** = proportion in each bin from the current/monitoring sample
- Sum is over all bins (typically 10 decile bins)

**Worked Example:**

| Score Bin | Expected % (Dev) | Actual % (Current) | (A − E) | ln(A/E) | (A−E) × ln(A/E) |
|-----------|-------------------|--------------------|-----------|---------|--------------------|
| 0–10th pctl | 10.0% | 12.0% | 0.020 | 0.182 | 0.0036 |
| 10–20th | 10.0% | 11.0% | 0.010 | 0.095 | 0.0010 |
| 20–30th | 10.0% | 10.5% | 0.005 | 0.049 | 0.0002 |
| 30–40th | 10.0% | 9.5% | −0.005 | −0.051 | 0.0003 |
| 40–50th | 10.0% | 9.0% | −0.010 | −0.105 | 0.0011 |
| 50–60th | 10.0% | 9.0% | −0.010 | −0.105 | 0.0011 |
| 60–70th | 10.0% | 10.0% | 0.000 | 0.000 | 0.0000 |
| 70–80th | 10.0% | 10.0% | 0.000 | 0.000 | 0.0000 |
| 80–90th | 10.0% | 9.5% | −0.005 | −0.051 | 0.0003 |
| 90–100th | 10.0% | 9.5% | −0.005 | −0.051 | 0.0003 |
| **Total** | | | | | **PSI = 0.0079** |

**PSI Interpretation Thresholds:**

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| **< 0.10** | No significant shift | Continue monitoring |
| **0.10 – 0.25** | Moderate shift | Investigate; consider recalibration |
| **> 0.25** | Major shift | Likely need to rebuild the model |

**Python Implementation:**

```python
import numpy as np

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index.
    
    Parameters:
        expected: array-like, scores from development sample
        actual: array-like, scores from current sample
        bins: number of bins (default 10)
    
    Returns:
        psi_value: float
    """
    # Create bins from the expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate proportions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

---

### 3.2 CSI (Characteristic Stability Index)

**What it is:** CSI applies the same PSI formula but to **individual features** rather than the overall score. It helps identify *which specific variables* are drifting.

```
CSI_j = Σ (Actual%_j - Expected%_j) × ln(Actual%_j / Expected%_j)
```

Where j = feature j, and bins are the WOE bins defined during development.

**Why CSI matters alongside PSI:**
- PSI tells you the score distribution shifted
- CSI tells you **which features** caused the shift
- Example: PSI = 0.18 (moderate drift). CSI reveals that "avg_utilization" has CSI = 0.30 while all other features have CSI < 0.05. Root cause: a new credit card product launched, changing utilization patterns.

**CSI Monitoring Table (example):**

| Feature | CSI Value | Status | Action |
|---------|-----------|--------|--------|
| credit_utilization_woe | 0.03 | ✅ Stable | None |
| payment_history_woe | 0.08 | ✅ Stable | None |
| enquiry_count_woe | 0.22 | ⚠️ Moderate drift | Investigate |
| account_age_woe | 0.02 | ✅ Stable | None |
| spend_volatility_woe | 0.31 | 🔴 Major drift | Rebin / rebuild feature |

---

### 3.3 Model Governance in Fintech

#### Regulatory Framework

| Regulation | Relevance |
|-----------|-----------|
| **RBI Guidelines (India)** | Fair lending practices; model explainability for adverse action notices; data privacy |
| **Basel II/III** | IRB (Internal Ratings-Based) approach requires validated PD models with documented methodology |
| **IFRS 9** | Expected Credit Loss (ECL) modeling requires PD, LGD, EAD models with validated assumptions |
| **IT Act, 2000 (India)** | Data privacy and security requirements for financial data |

#### Governance Practices I Implemented

| Practice | Description |
|----------|-------------|
| **Model documentation** | Full model development document (MDD) covering data, methodology, results, limitations |
| **Challenger model framework** | Maintained a champion (logistic regression) vs. challenger (XGBoost) comparison |
| **Periodic validation** | Monthly PSI/CSI checks; quarterly full model validation |
| **Adverse action codes** | Mapped top negative scorecard factors to reason codes for rejected applicants |
| **Data integrity checks** | Automated checks for schema changes, null spikes, distribution anomalies before scoring |
| **Access controls** | Score data access restricted to authorized personnel; audit logging |
| **Model inventory** | Documented in central model registry with version, performance metrics, approval status |

#### Data Integrity Checks (Automated in Pipeline)

```python
def data_integrity_checks(df, reference_stats):
    """Pre-scoring data quality gates."""
    checks = {}
    
    # 1. Row count check (±20% of expected)
    checks['row_count'] = abs(len(df) - reference_stats['expected_rows']) / \
                          reference_stats['expected_rows'] < 0.20
    
    # 2. Null rate check (no feature > 50% null)
    checks['null_rates'] = (df.isnull().mean() < 0.50).all()
    
    # 3. Schema check (all expected columns present)
    checks['schema'] = set(reference_stats['expected_columns']).issubset(df.columns)
    
    # 4. Value range check (no impossible values)
    checks['credit_score_range'] = df['bureau_score'].between(300, 900).all()
    
    # 5. Duplicate check
    checks['no_duplicates'] = not df['customer_id'].duplicated().any()
    
    all_passed = all(checks.values())
    return all_passed, checks
```

---

## 4. Key Metrics & Results

### 4.1 ROC-AUC: 0.94

**What ROC-AUC means in credit context:**

ROC-AUC measures the model's ability to **rank-order** borrowers by risk. An AUC of 0.94 means:
- If you randomly pick one defaulter and one non-defaulter, there is a **94% probability** that the model assigns a higher risk score to the defaulter
- In credit risk, AUC > 0.80 is considered good; > 0.90 is excellent
- This is measured on the **out-of-time** validation set, not just in-sample

**Performance across splits:**

| Dataset | ROC-AUC | KS Statistic | Gini |
|---------|---------|-------------|------|
| Train | 0.95 | 0.77 | 0.90 |
| Validation (in-time) | 0.94 | 0.76 | 0.88 |
| Out-of-Time Test | 0.94 | 0.75 | 0.88 |

**Stability across splits** is a key positive signal — the model generalizes well and is not overfit.

### 4.2 15× Processing Time Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| End-to-end turnaround | ~3 days (72 hours) | ~4-5 hours | **~15×** |
| Manual steps | 12+ manual handoffs | 0 (fully automated) | 100% reduction |
| Analyst time per cycle | ~20 person-hours | ~1 hour (review only) | 95% reduction |
| Scoring frequency | Weekly/ad-hoc | Daily (automated) | 7× increase |
| Error rate | ~5% manual errors | < 0.1% (automated checks) | 50× reduction |

### 4.3 Business Impact

| Impact | Description |
|--------|-------------|
| **Credit policy changes** | Score-based approval thresholds replaced subjective analyst judgment for initial screening |
| **Faster loan decisions** | Reduced time-to-decision for applicants; improved customer experience |
| **Risk segmentation** | Portfolio segmented into clear risk bands; differentiated pricing by risk tier |
| **Proactive monitoring** | Drift detection enabled early intervention before portfolio quality degraded |
| **Scalability** | Automated pipeline could score 10× more applications without additional headcount |
| **Regulatory compliance** | Fully documented, explainable model met RBI/audit requirements |

---

## 5. Topics You Must Know (Study Guide)

### 5.1 Logistic Regression — Deep Dive

**Why it matters:** The core of your scorecard. You must be able to explain every aspect.

#### Core Concepts

**The Model:**

```
P(default = 1 | X) = σ(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)

Where σ(z) = 1 / (1 + e^(-z))    ← Sigmoid function
```

**Log-Odds (Logit):**

```
ln(P / (1-P)) = β₀ + β₁X₁ + ... + βₙXₙ

This is a LINEAR function — that's why logistic regression is a linear classifier.
The log-odds are linear in the features.
```

**Odds Ratio:**

```
Odds = P(default) / P(no default) = e^(β₀ + β₁X₁ + ...)

For a 1-unit increase in X₁:
  New odds / Old odds = e^(β₁)

If β₁ = 0.5:  e^0.5 = 1.65 → odds increase by 65% for each unit increase in X₁
If β₁ = -0.3: e^(-0.3) = 0.74 → odds decrease by 26% for each unit increase in X₁
```

**Regularization:**

| Type | Formula | Effect | Use Case |
|------|---------|--------|----------|
| **L1 (Lasso)** | λ × Σ\|βᵢ\| | Sparse coefficients; feature selection | When you want automatic variable elimination |
| **L2 (Ridge)** | λ × Σβᵢ² | Shrinks coefficients; handles multicollinearity | Preferred for scorecards (stability) |
| **Elastic Net** | α×L1 + (1−α)×L2 | Compromise | When you want some sparsity with stability |

**Coefficients Interpretation (with WOE inputs):**

When inputs are WOE-transformed:
- All coefficients should be **negative** (higher WOE = lower risk → lower probability of default)
- A positive coefficient on a WOE variable is a red flag — investigate for data issues or overfitting
- Magnitude indicates feature importance (larger |β| = more predictive)

---

### 5.2 ROC-AUC vs PR-AUC

| Metric | Best For | Sensitive To | Your Project |
|--------|----------|-------------|-------------|
| **ROC-AUC** | Overall discriminatory power; balanced assessment | Not sensitive to class imbalance | Primary metric (AUC = 0.94) |
| **PR-AUC** | Evaluating performance on the **minority class** | Very sensitive to class imbalance | Useful as secondary metric when default rate is very low (< 5%) |

**When to use which:**
- **ROC-AUC** — standard for credit risk scorecards; measures rank-ordering across all thresholds; reported to regulators
- **PR-AUC** — use when the default rate is extremely low (e.g., 1%) and you care specifically about precision-recall tradeoffs at the risky end

**Key insight for interviews:** *"In our case with ~10% default rate, ROC-AUC was the appropriate primary metric. If the default rate had been < 2%, I would have also closely monitored PR-AUC because ROC-AUC can look optimistically high with extreme class imbalance."*

---

### 5.3 Feature Selection Methods

| Method | What It Does | Threshold | Used In This Project? |
|--------|-------------|-----------|----------------------|
| **IV (Information Value)** | Measures predictive power of a feature against binary target | IV > 0.02 to keep; > 0.5 suspicious | ✅ Primary method |
| **WOE (Weight of Evidence)** | Transforms features to encode target relationship; enables monotonicity check | N/A (transformation) | ✅ Core technique |
| **Correlation Matrix** | Identifies redundant features (\|r\| > 0.7) | Keep the one with higher IV | ✅ Yes |
| **VIF (Variance Inflation Factor)** | Detects multicollinearity | VIF < 5 (or < 10) | ✅ Yes |
| **LASSO (L1)** | Regularization-based selection; drives coefficients to zero | λ tuned via CV | ✅ Final confirmation |
| **Stepwise (forward/backward)** | Statistical significance-based | p-value < 0.05 | Considered, but IV/WOE preferred |

**VIF Formula:**

```
VIF_j = 1 / (1 − R²_j)

Where R²_j = R-squared from regressing feature j on all other features

VIF = 1     → No multicollinearity
VIF = 5     → Moderate (borderline)
VIF > 10    → Severe multicollinearity → remove
```

---

### 5.4 PSI and CSI — Formulas & Interpretation

(Covered in detail in Section 3.1 and 3.2 above)

**Quick reference:**

```
PSI / CSI = Σ (Actual%ᵢ − Expected%ᵢ) × ln(Actual%ᵢ / Expected%ᵢ)

Thresholds:
  < 0.10   → Stable
  0.10-0.25 → Investigate
  > 0.25   → Rebuild/recalibrate
```

---

### 5.5 Class Imbalance Handling in Credit Risk

| Technique | How It Works | Pros | Cons | Credit Risk Recommendation |
|-----------|-------------|------|------|---------------------------|
| **Class weights** | Upweight minority class in loss function | Simple; no synthetic data | May increase false positives | ✅ Preferred |
| **SMOTE** | Generate synthetic minority samples | Increases minority representation | Creates unrealistic borrower profiles | ⚠️ Use cautiously; not ideal for credit |
| **Undersampling** | Remove majority class samples | Faster training | Loses information | ❌ Avoid for scorecard development |
| **Threshold tuning** | Adjust classification threshold post-training | Preserves model; business-aligned | Requires careful calibration | ✅ Recommended |
| **Cost-sensitive learning** | Assign different misclassification costs | Business-aligned (cost of default vs. lost revenue) | Requires cost estimation | ✅ Advanced option |

---

### 5.6 Scorecard Development Methodology

**Steps (industry standard):**

1. **Define target** — DPD threshold (30/60/90 days), observation + performance window
2. **Sample design** — Ensure representative, time-based train/test
3. **Feature engineering** — Bureau, behavioral, transactional variables
4. **WOE binning** — Transform continuous features into WOE; ensure monotonicity
5. **Feature selection** — IV, correlation, VIF, business review
6. **Model fitting** — Logistic regression on WOE-transformed features
7. **Scorecard scaling** — Convert log-odds to points (PDO, base score, base odds)
8. **Validation** — In-sample, out-of-sample, out-of-time; KS, Gini, AUC
9. **Calibration** — Ensure predicted probabilities match observed default rates
10. **Implementation** — Automate scoring; set monitoring (PSI/CSI)

---

### 5.7 Credit Risk Regulations (Basics)

| Framework | Key Points for Interview |
|-----------|------------------------|
| **Basel II/III** | Pillar 1 requires banks to hold capital proportional to credit risk. IRB approach allows use of internal models for PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default). Models must be validated annually. |
| **IFRS 9** | Requires Expected Credit Loss (ECL) estimation. Three-stage model: Stage 1 (performing), Stage 2 (significant increase in risk), Stage 3 (credit-impaired). PD models feed directly into ECL calculations. |
| **RBI Guidelines (India)** | Fair lending practices; prohibition of discriminatory variables (religion, caste); requirement for adverse action notices explaining loan denial reasons; digital lending guidelines for fintechs. |
| **Fairness** | Must not use protected attributes (directly or as proxies). Disparate impact testing required. Model must produce consistent, explainable outcomes. |

---

### 5.8 KS Statistic (Kolmogorov-Smirnov)

**What it is:** KS measures the maximum separation between the cumulative distribution functions (CDFs) of defaulters and non-defaulters.

```
KS = max |CDF_good(score) − CDF_bad(score)|
```

**Interpretation:**

| KS Value | Quality |
|----------|---------|
| < 20 | Poor |
| 20 – 40 | Acceptable |
| 40 – 60 | Good |
| 60 – 75 | Very Good |
| > 75 | Excellent (or suspicious — check for leakage) |

**My project: KS ≈ 0.75** — very good separation.

**Relationship to AUC:** KS ≈ 2 × (AUC − 0.5) approximately. For AUC = 0.94, KS ≈ 0.88 theoretically, but the actual relationship depends on the shape of the ROC curve. Observed KS ~0.75 is consistent.

---

### 5.9 Gini Coefficient

```
Gini = 2 × AUC − 1

For AUC = 0.94:  Gini = 2 × 0.94 − 1 = 0.88
```

**Interpretation:** Gini = 0 means random; Gini = 1 means perfect separation. Industry standard for credit risk reporting.

**CAP Curve (Cumulative Accuracy Profile):** Gini is the ratio of the area between the model's CAP curve and the random line, to the area between the perfect model's CAP curve and the random line.

---

### 5.10 Monotonic Constraints in Credit Modeling

**Why required:**
- Regulators expect that "more risk = higher risk score" consistently
- Non-monotonic models produce counterintuitive explanations
- Customers denied credit deserve consistent, logical reasons

**How to enforce:**
- WOE binning (natural monotonic transformation when bins are properly merged)
- Monotonic constraints in tree models (XGBoost: `monotone_constraints` parameter)
- Business rule validation post-modeling

---

### 5.11 Knime Workflow Automation

**Key interview points:**
- Knime = visual, node-based workflow tool (think: Alteryx competitor)
- Integrates with Python, R, SQL, Spark
- Drag-and-drop pipeline: data read → transform → model → score → write
- Built-in scheduling (Knime Server) or OS-level cron
- Good for enterprise environments where full Python pipelines may not be approved

---

### 5.12 Data Drift vs Concept Drift

| Type | Definition | Example | Detection |
|------|-----------|---------|-----------|
| **Data Drift** (covariate shift) | Input feature distributions change | Average bureau scores increase due to economic recovery | PSI, CSI on features |
| **Concept Drift** | Relationship between features and target changes | Same bureau score now predicts different default rates (e.g., post-COVID) | Monitor actual vs predicted default rates; back-testing |
| **Label Drift** | Distribution of the target variable changes | Default rate drops from 10% to 5% | Monitor target rate trends |

**Key insight:** *"PSI and CSI detect data drift effectively. To detect concept drift, you need to monitor actual default rates against predicted rates over time — which takes months because you need the performance window to close."*

---

### 5.13 Model Validation: In-Sample, Out-of-Sample, Out-of-Time

| Validation Type | What It Means | Purpose |
|----------------|---------------|---------|
| **In-sample** | Performance on training data | Baseline; should be highest |
| **Out-of-sample (OOS)** | Random hold-out from same time period | Tests generalization (but same time period) |
| **Out-of-time (OOT)** | Data from a **future** time period not seen during training | Most realistic; simulates production deployment |
| **Cross-validation** | K-fold stratified CV within training set | Robust estimate of expected performance |

**In credit risk, OOT is the gold standard.** Regulators and model validation teams specifically ask for OOT performance.

---

## 6. Interview Questions & Answers

### Behavioral / STAR Questions

---

#### Q1: "Walk me through this project."

**Answer (2-minute version):**

> "At Fibe, a digital lending fintech, customer risk assessment was a manual process taking up to 3 days per cycle with inconsistent results. I was tasked with automating and improving the entire risk scoring workflow.
>
> First, I worked with data engineering to integrate data from three sources — credit bureau records with 1,000+ variables, internal transaction histories, and behavioral signals — into a unified customer-level table.
>
> For feature engineering, I used WOE binning and Information Value to systematically reduce 1,000+ raw variables to about 40-50 highly predictive features. I enforced monotonic relationships to ensure regulatory compliance.
>
> I evaluated multiple models — logistic regression, random forest, XGBoost — and chose logistic regression because in credit risk, interpretability is critical. You need to explain to regulators and customers why a loan was denied. The logistic regression achieved 0.94 ROC-AUC, which was only 1-2 points below ensemble models.
>
> I converted the model into a points-based scorecard and automated the entire pipeline using Knime — from data ingestion through scoring to output delivery. I also built PSI and CSI monitors for drift detection and created executive dashboards.
>
> The result was a 15× improvement in turnaround time and the scorecard directly influenced credit policy changes across the lending portfolio."

---

#### Q2: "Why logistic regression over XGBoost / Random Forest?"

**Answer:**

> "I actually benchmarked all three — XGBoost gave about 0.96 AUC versus 0.94 for logistic regression. But in credit risk, that 2% marginal gain comes with significant costs:
>
> **Regulatory**: Indian fintech regulations and Basel frameworks require that we can explain every credit decision. With logistic regression, I can say 'this customer was declined because their utilization was too high (−50 points) and their payment history showed 3 missed payments (−80 points).' With XGBoost, I'd need SHAP values which are approximate and harder to defend in an audit.
>
> **Scorecard conversion**: Logistic regression coefficients translate directly into a points-based scorecard that credit officers use daily. There's no natural equivalent for tree ensembles.
>
> **Monotonicity**: With WOE inputs, logistic regression naturally respects monotonic relationships. In XGBoost, you can add monotone constraints, but it's more complex and less transparent.
>
> **Stability**: Logistic regression models tend to be more stable over time — fewer hyperparameters, less prone to overfitting, easier to validate.
>
> So the decision was: 0.94 AUC with full interpretability, regulatory compliance, and ease of production maintenance versus 0.96 AUC with significant overhead. The business chose interpretability."

---

#### Q3: "How did you handle 1,000+ features?"

**Answer:**

> "I used a systematic funnel approach. Starting with 1,000+ raw bureau variables:
>
> 1. **First pass** — removed zero/near-zero variance features (those that were >99% a single value). This cut roughly 30%.
> 2. **Second** — removed features with >70% missing values, unless the missingness itself was predictive (in which case I created a binary indicator).
> 3. **WOE/IV computation** — I applied WOE binning to every remaining feature and computed Information Value. Anything with IV below 0.02 was dropped as not predictive. This was the biggest filter — got us down to about 200 features.
> 4. **Correlation filtering** — For pairs with |correlation| > 0.7, I kept the one with higher IV.
> 5. **VIF check** — Removed features until all VIF values were below 5 to handle multicollinearity.
> 6. **Monotonicity validation** — Confirmed each remaining feature showed a monotonic WOE pattern. Non-monotonic features were re-binned or dropped.
> 7. **Final business review** — Sat with the credit policy team to sanity-check that each feature made business sense.
>
> The result was about 40-50 features in the final scorecard — each one interpretable, predictive, and business-validated."

---

#### Q4: "What is PSI and how did you use it?"

**Answer:**

> "PSI — Population Stability Index — measures how much the distribution of scores has shifted between two periods. The formula sums the product of the percentage-point difference and the log-ratio across bins.
>
> I used it in two ways:
> 1. **Score-level PSI** — After each batch scoring run, I compared the new score distribution to the original development sample. This told me if the overall model output was drifting.
> 2. **Feature-level CSI** — I applied the same calculation to individual features to pinpoint *which* inputs were causing any score drift.
>
> The thresholds are: below 0.10 is stable, 0.10 to 0.25 means investigate, above 0.25 means the model likely needs rebuilding. These were automated as quality gates in our Knime pipeline — a PSI above 0.25 would trigger an alert and pause auto-scoring."

---

#### Q5: "How do you ensure model fairness in credit decisions?"

**Answer:**

> "Several layers. First, we excluded protected attributes — religion, caste, gender, marital status — not just directly but also checked for proxy variables. For example, a geographic feature might correlate heavily with a protected class, so I checked for that.
>
> Second, I performed disparate impact analysis — checking whether approval rates differed significantly across demographic groups. The 80% rule is a common threshold: if the approval rate for any group is less than 80% of the rate for the most-approved group, it warrants investigation.
>
> Third, the choice of logistic regression itself supports fairness — since every decision can be decomposed into specific factor contributions, any denial can be explained with concrete reasons (adverse action codes), which is both a regulatory requirement and a fairness mechanism.
>
> Fourth, the WOE binning approach treats missing data explicitly, which prevents biasing against thin-file (young or underserved) customers who might have fewer bureau records."

---

#### Q6: "What is WOE/IV?"

**Answer:**

> "WOE — Weight of Evidence — is a transformation that replaces each feature's values with a metric capturing how much each bin of that feature separates defaulters from non-defaulters. The formula is: WOE = ln(percentage of non-defaulters in the bin / percentage of defaulters in the bin).
>
> Positive WOE means the bin has proportionally more good customers; negative WOE means more bad customers. The beauty of WOE is that it naturally creates a monotonic encoding when bins are properly defined, and it converts all features — whether continuous or categorical — into a common, comparable scale.
>
> IV — Information Value — is the sum of (difference in percentages × WOE) across all bins. It tells you the total predictive power of a feature. Below 0.02 is not useful; 0.02 to 0.1 is weak; 0.1 to 0.3 is medium; above 0.3 is strong. Anything above 0.5 is suspiciously strong and might indicate data leakage or overfit."

---

#### Q7: "How did you validate the model over time?"

**Answer:**

> "Three layers of validation:
>
> **Pre-deployment:** I used out-of-time validation — trained on data through June 2021, validated on July–December 2021, and tested on January–March 2022. The AUC remained at 0.94 across all splits, confirming temporal stability.
>
> **Ongoing monitoring:** After deployment, the Knime pipeline ran PSI and CSI checks on every batch. Monthly, I reviewed the PSI trend and the actual-vs-predicted default rate for each score band.
>
> **Periodic revalidation:** We planned quarterly deep dives — re-running the full validation suite (AUC, KS, Gini, lift analysis) on the latest data, and comparing champion (current model) vs. challenger (retrained or alternative model) performance. This was documented as part of the model governance framework."

---

#### Q8: "What would you do differently?"

**Answer:**

> "Three things:
>
> 1. **Champion-challenger in production**: I would set up a live A/B test framework where a small percentage of decisions use a challenger model (e.g., XGBoost with SHAP explanations) to continuously benchmark against the logistic regression champion.
>
> 2. **Real-time scoring**: Our pipeline was batch-based (daily). For a more mature system, I'd push toward real-time API-based scoring using FastAPI or similar, so decisions happen at application time.
>
> 3. **More behavioral features**: We had basic app engagement signals, but modern fintechs use much richer alternative data — app usage patterns, device graph data, transaction categorization. If I had more time, I would have explored these more deeply. Of course, any new data source needs careful assessment for regulatory compliance and bias."

---

#### Q9: "How did you present results to non-technical stakeholders?"

**Answer:**

> "I structured presentations around business outcomes, not technical metrics. Instead of leading with 'AUC is 0.94,' I said: 'For every 100 customers we approve, we'll correctly identify 94 out of 100 who would have defaulted before they're approved.' I translated everything into money — 'this model would have prevented X lakhs in losses over the last year based on backtesting.'
>
> I used visual dashboards showing score distributions, approval rates by risk band, and projected portfolio performance. I created a one-page scorecard summary showing which factors contribute most to a customer's score — credit officers could immediately understand it.
>
> For credit policy changes, I presented simulation tables: 'If we move the cutoff from 500 to 550, approval rate drops by 8% but default rate drops by 40%.' This gave business leaders the data to make informed trade-off decisions."

---

#### Q10: "What if the model started drifting?"

**Answer:**

> "My response depends on the severity:
>
> **Mild drift (PSI 0.10–0.15):** Investigate root cause using CSI. If it's one or two features, check whether the data source changed, there was a new product launch, or macroeconomic shift. May just need recalibration — adjust the base score or thresholds without rebuilding.
>
> **Moderate drift (PSI 0.15–0.25):** More serious investigation. Compare actual default rates against predicted. If the model is still rank-ordering well (AUC holds) but probabilities are miscalibrated, recalibrate using Platt scaling or isotonic regression. If rank-ordering has degraded, begin rebuilding.
>
> **Severe drift (PSI > 0.25):** Trigger model rebuild. Use the latest data to retrain, following the same WOE/IV pipeline. Run full validation. Present to model governance committee for approval before deployment.
>
> In all cases, the first step is 'is this drift or a data quality issue?' — a broken data feed can look like drift. That's why the data integrity checks run *before* the PSI computation."

---

#### Q11: "How did you handle missing data from bureau records?"

**Answer:**

> "Missing bureau data is very common — new-to-credit customers, thin files, or data lags. My approach:
>
> 1. **Treated missingness as a feature**: For key variables like bureau score, I created a binary flag 'has_bureau_score' (0/1). The absence of data is itself informative — no bureau history often correlates with higher risk (or conversely, very young/underserved population).
>
> 2. **Separate WOE bin for missing**: Instead of imputing and pretending the data exists, I gave missing values their own WOE bin. This lets the model learn the empirical default rate for customers with missing data.
>
> 3. **Domain-informed imputation for numerical features**: Where a feature was missing but the customer had other bureau data (e.g., utilization missing but accounts present), I used median imputation within the risk segment. But I always kept the binary missing flag alongside.
>
> 4. **No imputation when data is fundamentally absent**: For customers with zero bureau history, I didn't impute a fake score — I used the indicator variables and relied on alternative data (behavioral/transaction features) for these customers.
>
> This approach is important because regulators don't like models that hallucinate data for underserved populations."

---

#### Q12: "Explain the monotonic relationship requirement."

**Answer:**

> "In credit risk, monotonicity means that as a feature value increases risk, the model's predicted probability of default should consistently increase — no zig-zags.
>
> For example, if credit utilization goes from 20% to 40% to 60% to 80%, the risk should either consistently increase or consistently decrease. If our model said '60% utilization is riskier than 80% utilization,' that's non-monotonic and has two problems:
>
> 1. **Explainability**: A customer denied at 60% utilization would rightfully ask why someone at 80% was approved. There's no defensible explanation.
>
> 2. **Regulatory**: Auditors check for monotonic patterns. Non-monotonic models are considered unreliable and may not get approved.
>
> WOE binning naturally encourages monotonicity — I merged adjacent bins until the WOE values formed a smooth monotonic curve. For any remaining non-monotonic features, I either re-engineered the binning or dropped the feature."

---

#### Q13: "What regulatory considerations did you face?"

**Answer:**

> "Several. India's credit landscape has specific regulatory requirements:
>
> 1. **Explainability**: RBI's fair lending guidelines require that customers denied credit receive clear reasons. Our points-based scorecard made this straightforward — we mapped the top 3-4 negative scoring factors to standardized adverse action codes.
>
> 2. **Data privacy**: Credit bureau data usage is governed by the Credit Information Companies Act. We only used bureau data for its intended purpose and maintained proper consent and access controls.
>
> 3. **Non-discrimination**: We excluded variables that could serve as proxies for protected classes and tested for disparate impact.
>
> 4. **Model documentation**: We maintained a full model development document covering methodology, data sources, performance metrics, known limitations, and validation results — which is essential for any future audit.
>
> 5. **Digital lending guidelines**: As a fintech, Fibe fell under RBI's digital lending framework, which has specific requirements around transparency in credit decisions.
>
> These considerations directly influenced our choice of logistic regression over black-box models."

---

### Technical Questions

---

#### Q14: "Explain the sigmoid function and how it relates to logistic regression."

**Answer:**

> "The sigmoid function σ(z) = 1/(1+e^(-z)) maps any real number to a probability between 0 and 1. In logistic regression, z is the linear combination of features (β₀ + β₁X₁ + ...), and the sigmoid transforms this into a default probability.
>
> Key properties: σ(0) = 0.5 (when log-odds are zero, it's a coin flip); σ approaches 1 as z → ∞ and 0 as z → −∞; its derivative σ'(z) = σ(z)(1−σ(z)), which is important for gradient-based optimization.
>
> In the scorecard context, we don't use the sigmoid output directly. We work in log-odds space: ln(P/(1-P)) = β₀ + β₁X₁ + ... This linear relationship is what gets converted into scorecard points."

---

#### Q15: "What's the difference between Gini and AUC? How are they related?"

**Answer:**

> "Gini = 2 × AUC − 1. So AUC of 0.94 corresponds to Gini of 0.88. AUC ranges from 0.5 (random) to 1.0 (perfect), while Gini ranges from 0 (random) to 1 (perfect). They convey the same information — Gini is just a rescaled version. In some regions (Europe, India), credit risk teams prefer reporting Gini; in others (US), AUC is standard. Know both."

---

#### Q16: "How does SMOTE work and why didn't you use it?"

**Answer:**

> "SMOTE creates synthetic minority class samples by picking a minority instance, finding its k nearest neighbors (in feature space), and generating new points along the line segments connecting them.
>
> I didn't use it for two reasons. First, our imbalance was moderate (~10% default rate), not extreme — `class_weight='balanced'` was sufficient. Second, and more importantly, SMOTE in credit risk creates synthetic borrowers with feature combinations that may not exist in reality. A synthetic customer with high income but very high utilization and multiple delinquencies might not represent any real borrower profile. This can teach the model patterns that don't exist in the real population, and regulators are skeptical of models trained on synthetic data."

---

#### Q17: "Explain the KS statistic and how you computed it."

**Answer:**

> "KS measures the maximum vertical distance between the cumulative distribution of defaulters and non-defaulters when sorted by the model score. You sort all customers by score, compute the running cumulative proportion of good and bad customers at each score threshold, and find where the gap is largest.
>
> A KS of 0.75 means that at the optimal cutoff point, the model captures 75% more cumulative defaulters than cumulative non-defaulters compared to random. It's complementary to AUC — while AUC averages performance across all thresholds, KS focuses on the single point of maximum separation."

```python
from scipy.stats import ks_2samp

def compute_ks(y_true, y_pred_proba):
    """Compute KS statistic for credit risk model."""
    good_scores = y_pred_proba[y_true == 0]
    bad_scores = y_pred_proba[y_true == 1]
    ks_stat, p_value = ks_2samp(good_scores, bad_scores)
    return ks_stat

# Example
# ks = compute_ks(y_test, model.predict_proba(X_test)[:, 1])
# print(f"KS Statistic: {ks:.4f}")
```

---

#### Q18: "What is data leakage and how did you prevent it?"

**Answer:**

> "Data leakage is when information from the target or the future 'leaks' into the training features, giving unrealistically good performance that doesn't hold in production.
>
> In credit risk, common leakage sources include:
> - Using account status variables that are updated *after* the observation point (e.g., current DPD when predicting future DPD)
> - Including variables that directly encode the target (e.g., 'loan_written_off' flag)
> - Performing WOE binning on the full dataset instead of just the training set
>
> I prevented it by:
> 1. Strict temporal cutoff — features computed as of date T, target measured at T+6 months. No feature could reference data after T.
> 2. WOE binning fitted only on training data, then applied (transformed) to validation and test.
> 3. Feature audit — reviewed every variable with the business team to confirm it would be available at prediction time.
> 4. Suspicious IV check — any feature with IV > 0.5 was investigated for leakage."

---

#### Q19: "How would you deploy this model as a real-time API?"

**Answer:**

> "If I were to evolve this from batch to real-time:
>
> 1. **Serialize the model**: Export the WOE binning maps (dictionary) and logistic regression coefficients. The scoring function is just: apply WOE mapping, multiply by coefficients, add intercept, convert to score.
>
> 2. **API framework**: FastAPI for the scoring endpoint. Input = customer features, Output = risk score + risk band + top contributing factors.
>
> 3. **Latency**: The scoring itself is just dictionary lookups + a dot product — sub-millisecond. The bottleneck would be the bureau API call.
>
> 4. **Monitoring**: Log every request/response. Batch the logs nightly for PSI computation. Alert if the score distribution deviates.
>
> 5. **Fallback**: If the bureau API is down, have a degraded model that uses only internal features (lower accuracy but still functional)."

---

#### Q20: "What's the difference between PDO and scaling in a scorecard?"

**Answer:**

> "PDO — Points to Double the Odds — defines the scorecard's scale. If PDO = 20, then a 20-point increase means the odds of being good double (i.e., risk halves). The base score and base odds anchor the scale.
>
> For example, if we set: Score 600 = odds 50:1 (2% default rate) with PDO 20, then:
> - Score 620 = odds 100:1 (1% default rate)
> - Score 580 = odds 25:1 (4% default rate)
>
> The formulas are: Factor = PDO / ln(2), Offset = Base_Score − Factor × ln(Base_Odds). Each feature's points = −(β × WOE × Factor). The industry standards for PDO are typically 20 or 50 points."

---

#### Q21: "How did you select the optimal cutoff threshold?"

**Answer:**

> "The default 0.5 threshold is rarely optimal for credit risk. I used a few approaches:
>
> 1. **KS-based threshold**: The score point where KS is maximized — this maximizes the separation between good and bad customers.
>
> 2. **Cost-based optimization**: Defined the cost of a false negative (approving a defaulter — cost of default) vs. false positive (rejecting a good customer — lost revenue). Optimized the threshold to minimize total expected cost.
>
> 3. **Business constraint**: The credit policy team specified a target approval rate (e.g., 'we want to approve ~70% of applicants'). I found the score threshold that achieved this rate while reporting the expected default rate.
>
> Ultimately, the business used multiple thresholds — auto-approve above 600, manual review for 450–600, auto-decline below 450."

---

#### Q22: "What is the lift chart / gains table and how did you use it?"

**Answer:**

> "A gains table (or lift chart) divides the scored population into deciles by predicted risk and shows the concentration of defaults in each decile.
>
> In a good model, the riskiest decile (top 10%) should capture a disproportionate share of actual defaults. For my model, the top decile captured ~45% of all defaults (lift = 4.5×), and the top 3 deciles captured ~80%.
>
> This is operationally useful: 'If we only have capacity to review 30% of applicants manually, focusing on the top 3 risk deciles captures 80% of potential defaults.' It translates model performance into resource allocation decisions."

---

#### Q23: "Explain L1 vs L2 regularization in the context of your scorecard."

**Answer:**

> "L1 (Lasso) adds the absolute value of coefficients as a penalty, driving some to exactly zero — it performs feature selection. L2 (Ridge) adds squared coefficients, shrinking them toward zero but never exactly reaching zero — it handles multicollinearity.
>
> For the scorecard, I used L2 as the primary regularization because:
> 1. I had already done feature selection via IV/WOE, so I didn't need L1's selection effect
> 2. L2 produces more stable coefficient estimates, which means more stable scorecard points
> 3. L2 handles any residual multicollinearity that survived VIF filtering
>
> I did use L1 as a *confirmation step* — I ran LASSO separately and verified that the features it retained overlapped with my IV/WOE-selected features. This gave me confidence that the feature set was robust."

---

#### Q24: "Tell me about a time you had to push back on a stakeholder's request."

**Answer (STAR):**

> **Situation:** The credit policy head wanted to include 'employer name' as a scorecard variable because historically, employees of certain large companies had lower default rates.
>
> **Task:** I needed to evaluate whether this was a sound decision from both a modeling and ethical perspective.
>
> **Action:** I analyzed the feature and found that: (a) it was a proxy for income level, which we already captured directly, (b) it was highly granular (thousands of employers) making it unstable, (c) it could introduce discriminatory bias — employees of large tech companies vs. small businesses, which correlates with socioeconomic background.
>
> I presented my analysis showing that the feature had high IV but added minimal incremental AUC when income-related features were already in the model. I also highlighted the regulatory risk.
>
> **Result:** The stakeholder agreed to drop the feature. We instead added 'employer tenure' (continuous, months) which captured the stability signal without the discrimination risk.

---

#### Q25: "How would you handle model performance degradation post-COVID?"

**Answer:**

> "Post-COVID is a classic concept drift scenario — the relationship between features and default fundamentally changed.
>
> 1. **Short-term**: Apply segment-level recalibration. If the overall default rate shifted, adjust the base odds in the scorecard. If specific segments shifted (e.g., hospitality workers), adjust thresholds for those segments.
>
> 2. **Medium-term**: Exclude the COVID period from training as an anomalous period, OR include it with appropriate sample weights that reflect the new normal.
>
> 3. **Rebuild**: Retrain on post-COVID data once enough performance window data has accumulated (6-12 months post-pandemic). Include COVID-recovery features (e.g., 'months since moratorium ended', 'payment resume speed').
>
> 4. **Hybrid approach**: Maintain separate pre-COVID and post-COVID models, blended based on recency.
>
> The key challenge is that performance windows mean you don't see the true default rates until 6-12 months later — so early signals from PSI/CSI become even more critical for proactive intervention."

---

## 7. Potential Red Flags & How to Handle

### Red Flag 1: "Why logistic regression in 2022? Wasn't that outdated?"

**How to handle:**

> "Far from outdated — logistic regression remains the industry standard for credit risk scorecards globally. JPMorgan, HSBC, all major banks still use logistic regression-based scorecards. The reason is regulatory: Basel IRB frameworks, RBI guidelines, and model audit processes are all built around interpretable models. The 'state of the art' in credit risk isn't about using the most complex model — it's about building the most reliable, explainable, and maintainable model.
>
> That said, I did benchmark against XGBoost and other ensembles. The performance gap was marginal (1-2% AUC), and the interpretability gap was massive. In a domain where you must explain every credit denial, logistic regression is the pragmatic choice."

---

### Red Flag 2: "Was 0.94 AUC realistic? That seems very high."

**How to handle:**

> "Fair question — and I asked myself the same thing. Here's why it's credible:
>
> 1. **Rich bureau data**: With 1,000+ bureau variables including DPD history, payment patterns, utilization, and enquiry behavior, the signal is very strong. Bureau data is the most predictive data source in credit risk.
>
> 2. **30 DPD target**: We used 30 days past due as the default definition. This is a relatively 'easy' target to predict compared to, say, write-off (180+ DPD). At 30 DPD, there are clear leading indicators.
>
> 3. **Leakage checks**: I verified there was no data leakage — no feature had IV > 0.5, the OOT performance matched in-sample, and all features were confirmed available at prediction time.
>
> 4. **Industry context**: For behaviour scorecards (predicting existing customers' risk using their full history), AUC of 0.85-0.95 is normal. For application scorecards (new customers with less data), 0.70-0.80 is more typical. My model was a behaviour scorecard with rich data — 0.94 is within the expected range.
>
> 5. **Consistent across splits**: Train AUC 0.95, validation 0.94, OOT 0.94 — no significant drop, ruling out overfitting."

---

### Red Flag 3: "Knime? Why not Airflow or a proper Python pipeline?"

**How to handle:**

> "Pragmatic decision based on organizational context:
>
> 1. **Existing tooling**: Fibe's analytics team already had Knime licenses and expertise. Introducing Airflow would have required DevOps support, infrastructure setup, and training.
>
> 2. **Audit-friendly**: Knime's visual workflows meant that the credit policy team and auditors could *see* the pipeline. A Python script is a black box to non-coders.
>
> 3. **Python inside Knime**: The actual model logic ran in Python nodes within Knime. We got the best of both worlds — scikit-learn for modeling, Knime for orchestration and visualization.
>
> 4. **Speed to production**: We got the automated pipeline running in weeks, not months. In a fast-moving fintech, time-to-production matters.
>
> If I were at a company with mature ML infrastructure, I'd absolutely use Airflow/Prefect + MLflow + FastAPI. But given Fibe's context at the time, Knime was the right choice."

---

### Red Flag 4: "Only 10 months on the project? Did you see long-term results?"

**How to handle:**

> "Good question. Within my tenure, I deployed the model, ran it through two full scoring cycles, and confirmed stable PSI metrics. The out-of-time validation gave confidence in forward-looking performance. I also set up the monitoring framework so the team could track long-term drift after my departure.
>
> For credit risk, true long-term validation requires 12-18 months of performance data (to see full default cycles). I designed the validation framework to capture this and handed it off with clear documentation. From what I know, the model continued to perform well, but I always frame results based on what I directly observed and validated."

---

## 8. Key Takeaways & Talking Points

### The "5 Things" Framework

When asked "what did you learn from this project?" or similar open-ended questions, here are your go-to points:

| # | Talking Point | Why It's Compelling |
|---|--------------|-------------------|
| 1 | **Interpretability > complexity in regulated domains** | Shows maturity — you understand that the best model isn't always the most complex one |
| 2 | **Feature engineering matters more than model selection** | 1,000+ variables → 40 final features drove more value than model choice. Most interviewers love hearing this. |
| 3 | **Monitoring is not an afterthought** | PSI/CSI monitoring, governance framework, dashboards — shows you think about the full ML lifecycle, not just training |
| 4 | **Translating tech to business impact** | Dashboards, scorecard points, policy recommendations — you didn't just build a model, you drove decisions |
| 5 | **Pragmatic tool choice** | Chose Knime despite knowing Python deeply — right tool for the right context. Shows you're not dogmatic. |

### Resume Bullet Point (reference)

> "Led feature-engineering & scorecard modeling on 1000+ bureau vars; lifted ROC-AUC to 0.94 and translated model outputs into executive dashboards and decision frameworks, driving credit-policy changes. Automated Python scripts with Knime, added AI-governance checks (data integrity, security), shrinking turnaround 15x."

### Quantified Impact Summary

| Metric | Value |
|--------|-------|
| Features processed | 1,000+ → ~50 final |
| ROC-AUC | 0.94 (OOT validated) |
| KS Statistic | ~0.75 |
| Gini | ~0.88 |
| Turnaround improvement | 15× (3 days → few hours) |
| Manual steps eliminated | 100% automated |
| Analyst time saved | ~95% per cycle |
| Business outcome | Credit policy changes adopted enterprise-wide |

### Storytelling Arc (for interviews)

```
1. HOOK:      "Our risk assessment took 3 days and was inconsistent..."
2. CHALLENGE: "1,000+ raw variables, multiple data sources, regulatory constraints..."
3. APPROACH:  "Systematic feature engineering, interpretable modeling, full automation..."
4. RESULT:    "0.94 AUC, 15× faster, directly shaped credit policy..."
5. LEARNING:  "In regulated domains, interpretability wins over complexity..."
```

---

## Quick Reference Card

| If They Ask About... | Key Points to Hit |
|----------------------|------------------|
| **The model** | Logistic regression, WOE/IV, 0.94 AUC, scorecard conversion |
| **Features** | 1,000+ → ~50 via IV/correlation/VIF/monotonicity funnel |
| **Why not XGBoost** | Regulatory, interpretability, monotonicity, scorecard; marginal AUC gain |
| **Monitoring** | PSI (score drift), CSI (feature drift), thresholds 0.10/0.25 |
| **Business impact** | 15× turnaround, credit policy changes, automated pipeline |
| **Governance** | Model documentation, adverse action codes, fairness testing, validation framework |
| **Technical depth** | Log-odds, sigmoid, regularization, KS, Gini, class weights, OOT validation |
| **Stakeholder mgmt** | Dashboards, business simulations, scorecard explanations, pushed back when needed |
| **What you'd change** | Champion-challenger, real-time scoring, richer behavioral features |

---

*Last updated: February 2026 | Prepared for Rahul Sharma's interview preparation*
