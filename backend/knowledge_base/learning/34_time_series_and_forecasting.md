# Time Series Analysis & Forecasting — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist — Forecasting, Temporal Modeling, Anomaly Detection, Neural Forecasting

> **Context:** You have time-series experience from Tesla stock forecasting (LSTM, XGBoost) and Jet2 (walk-forward validation, LSTM/GRU, drift monitoring). This guide covers the **theory and methods** interviewers expect you to articulate clearly.

---

# Table of Contents

1. [Time Series Fundamentals](#1-time-series-fundamentals)
2. [Classical Statistical Methods](#2-classical-statistical-methods)
3. [Machine Learning for Time Series](#3-machine-learning-for-time-series)
4. [Deep Learning for Time Series](#4-deep-learning-for-time-series)
5. [Modern Neural Forecasting](#5-modern-neural-forecasting)
6. [Evaluation and Validation](#6-evaluation-and-validation)
7. [Anomaly Detection in Time Series](#7-anomaly-detection-in-time-series)
8. [Practical Considerations](#8-practical-considerations)
9. [Interview Questions with Strong Answers](#9-interview-questions-with-strong-answers)

---

# **1. Time Series Fundamentals**

## **1.1 Key Components**

Every time series can be decomposed into:

```
y(t) = Trend(t) + Seasonality(t) + Residual(t)    (additive)
y(t) = Trend(t) × Seasonality(t) × Residual(t)    (multiplicative)
```

| Component | What It Is | Example |
|-----------|-----------|---------|
| **Trend** | Long-term direction (up, down, flat) | E-commerce revenue growing 10% YoY |
| **Seasonality** | Regular periodic patterns | Higher sales in Q4 (holiday), daily peaks at noon |
| **Cyclicity** | Irregular long-term oscillations | Economic cycles (not fixed period) |
| **Residual/Noise** | Random variation after removing trend + seasonality | Unpredictable day-to-day fluctuations |

## **1.2 Stationarity**

A series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time.

**Why it matters:** Most classical models (ARIMA, VAR) assume stationarity. Non-stationary series must be transformed first.

| Test | What It Tests | How to Interpret |
|------|-------------|-----------------|
| **ADF (Augmented Dickey-Fuller)** | H₀: unit root (non-stationary) | p < 0.05 → stationary |
| **KPSS** | H₀: stationary | p < 0.05 → non-stationary |
| **Visual inspection** | Plot + rolling mean/std | If mean/variance drift → non-stationary |

**Making a series stationary:**
- **Differencing:** y'(t) = y(t) - y(t-1). Apply d times until stationary.
- **Log transform:** Stabilizes variance for exponentially growing series
- **Seasonal differencing:** y'(t) = y(t) - y(t-m) where m is seasonal period

## **1.3 Autocorrelation**

| Function | What It Shows | Used For |
|----------|-------------|----------|
| **ACF** (Autocorrelation Function) | Correlation between y(t) and y(t-k) for all lags k | Identify MA order (q) — ACF cuts off after q |
| **PACF** (Partial ACF) | Direct correlation at lag k, removing intermediate lags | Identify AR order (p) — PACF cuts off after p |

```
ACF cuts off at lag q, PACF decays → MA(q)
PACF cuts off at lag p, ACF decays → AR(p)
Both decay gradually → ARMA(p,q) needed
```

---

# **2. Classical Statistical Methods**

## **2.1 ARIMA (AutoRegressive Integrated Moving Average)**

ARIMA(p, d, q):
- **AR(p):** y(t) depends on its own past p values (autoregressive)
- **I(d):** Series is differenced d times to achieve stationarity (integrated)
- **MA(q):** y(t) depends on past q forecast errors (moving average)

```
y'(t) = c + φ₁y'(t-1) + ... + φₚy'(t-p) + θ₁ε(t-1) + ... + θqε(t-q) + ε(t)
         ─────────── AR part ───────────   ────────── MA part ──────────
```

**SARIMA** adds seasonal terms: ARIMA(p,d,q)(P,D,Q)m where m is the seasonal period (12 for monthly, 7 for daily with weekly pattern).

### Model Selection

| Method | How |
|--------|-----|
| **Box-Jenkins** | Visual ACF/PACF → guess (p,d,q) → fit → check residuals → iterate |
| **Auto ARIMA** | Grid search over (p,d,q) minimizing AIC/BIC (`pmdarima.auto_arima`) |
| **AIC/BIC** | AIC = penalized log-likelihood. Lower is better. BIC penalizes complexity more. |

### Residual Diagnostics

A good ARIMA fit should have residuals that are:
- White noise (no autocorrelation) — check with Ljung-Box test
- Normally distributed — check with QQ plot
- Homoscedastic (constant variance) — check with visual inspection

## **2.2 Exponential Smoothing (ETS)**

Instead of explicit AR/MA terms, weights past observations with exponentially decaying weights.

| Method | What It Captures | Parameters |
|--------|-----------------|------------|
| **Simple ES** | Level only | α (smoothing) |
| **Holt's** | Level + trend | α, β |
| **Holt-Winters** | Level + trend + seasonality | α, β, γ |

**ETS(Error, Trend, Seasonality):** A systematic framework with 30 model variants combining:
- Error: Additive (A) or Multiplicative (M)
- Trend: None (N), Additive (A), Additive Damped (Ad), Multiplicative (M)
- Seasonality: None (N), Additive (A), Multiplicative (M)

## **2.3 Prophet (Meta)**

Decomposable model: y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)

| Component | How It's Modeled |
|-----------|-----------------|
| **Trend** | Piecewise linear or logistic growth with automatic changepoints |
| **Seasonality** | Fourier series (weekly, yearly, custom) |
| **Holidays** | User-specified binary regressors |
| **Regressors** | External variables (temperature, promotions, etc.) |

**When to use Prophet:**
- Strong seasonal effects (weekly, yearly)
- Business time series with holiday effects
- Missing data and outliers (handles gracefully)
- Need fast, reasonable baseline (not always the most accurate)

## **2.4 VAR (Vector Autoregression)**

For **multivariate** time series — when multiple series influence each other.

Each variable is regressed on its own past values AND past values of all other variables:

```
y₁(t) = c₁ + A₁₁y₁(t-1) + A₁₂y₂(t-1) + ε₁(t)
y₂(t) = c₂ + A₂₁y₁(t-1) + A₂₂y₂(t-1) + ε₂(t)
```

**Granger Causality:** Variable X "Granger-causes" Y if past values of X help predict Y beyond Y's own past values. (Not true causality — see file 33.)

---

# **3. Machine Learning for Time Series**

## **3.1 Feature Engineering for Time Series ML**

ML models (XGBoost, LightGBM) don't inherently understand temporal order. You must **engineer temporal features**:

| Feature Type | Examples |
|-------------|---------|
| **Lag features** | y(t-1), y(t-7), y(t-30) |
| **Rolling statistics** | Mean, std, min, max over windows (7d, 30d) |
| **Date features** | Day of week, month, quarter, is_holiday, is_weekend |
| **Trend features** | Linear trend counter, days since event |
| **Difference features** | y(t) - y(t-1), y(t) - y(t-7) |
| **Fourier features** | sin(2πkt/T), cos(2πkt/T) for seasonality |
| **External features** | Weather, promotions, competitor prices |

## **3.2 XGBoost / LightGBM for Forecasting**

| Advantage | Limitation |
|-----------|-----------|
| Handles nonlinear relationships naturally | Requires manual feature engineering |
| Can incorporate exogenous variables easily | Can't extrapolate beyond training range |
| Fast training and inference | Doesn't capture sequential dependencies inherently |
| Often wins tabular forecasting competitions | Needs careful temporal validation |

## **3.3 Multi-Step Forecasting Strategies**

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Recursive** | Train one-step model; feed predictions back as inputs | Simple, one model | Error accumulates over horizon |
| **Direct** | Train separate model for each horizon (h=1, h=2, ...) | No error accumulation | Many models, can't share info |
| **DirRec** | Hybrid: predictions from shorter horizons as features | Balances both | Complex |
| **Seq2Seq** | Model directly outputs full horizon | End-to-end | Needs deep learning architecture |

---

# **4. Deep Learning for Time Series**

## **4.1 RNN / LSTM / GRU**

| Architecture | Key Mechanism | Strength | Limitation |
|-------------|--------------|----------|-----------|
| **RNN** | Hidden state h(t) = f(x(t), h(t-1)) | Sequential processing | Vanishing gradients, short memory |
| **LSTM** | Forget/input/output gates + cell state | Long-range dependencies, gating controls info flow | Slow training, sequential (no parallelism) |
| **GRU** | Reset/update gates (simpler than LSTM) | Faster than LSTM, comparable performance | Less expressive than LSTM on some tasks |

**Key LSTM insight for interviews:** The **cell state** is the key innovation — it flows through time with minimal transformation (multiplicative gates instead of matrix multiplications), allowing gradients to propagate over long sequences without vanishing.

## **4.2 Temporal Convolutional Networks (TCN)**

Use **dilated causal convolutions** instead of recurrence:

```
Dilation = 1:  ■─■─■─■─■  (every timestep)
Dilation = 2:  ■───■───■  (every 2nd timestep)
Dilation = 4:  ■───────■  (every 4th timestep)

Stack layers with increasing dilation → exponential receptive field growth
```

**Advantages over LSTM:** Parallelizable (no sequential dependency), stable gradients, adjustable receptive field. Often faster and competitive in accuracy.

## **4.3 Transformers for Time Series**

| Model | Key Innovation |
|-------|---------------|
| **Informer** (2021) | ProbSparse self-attention (O(L log L) vs O(L²)), distilling encoder |
| **Autoformer** (2021) | Auto-correlation mechanism + series decomposition |
| **PatchTST** (2023) | Treats time series as patches (like ViT for images), channel-independent |
| **TimesFM** (2024) | Google's foundation model for time series (pre-trained, zero-shot) |
| **Chronos** (2024) | Amazon's pre-trained time series model (tokenizes values) |

**Important debate:** Recent work (Zeng et al., 2023 — "Are Transformers Effective for Time Series Forecasting?") showed that a **simple linear model** can outperform many transformer-based models. The takeaway: always compare against simple baselines.

---

# **5. Modern Neural Forecasting**

## **5.1 N-BEATS (Neural Basis Expansion Analysis)**

Pure deep learning architecture designed specifically for time series:
- Stack of fully connected blocks
- Each block predicts a backcast (past reconstruction) and forecast
- Blocks are stacked with residual learning
- Interpretable variant decomposes into trend + seasonality

## **5.2 Temporal Fusion Transformer (TFT)**

| Component | Purpose |
|-----------|---------|
| **Variable Selection Network** | Learns which inputs matter (interpretability) |
| **LSTM Encoder** | Captures local temporal patterns |
| **Multi-head Attention** | Captures long-range dependencies |
| **Gating Mechanisms** | Skip irrelevant features |
| **Quantile Outputs** | Produces prediction intervals, not just point forecasts |

**Why TFT is popular:** It handles **static covariates** (store location), **known future inputs** (holidays, planned promotions), and **observed past inputs** (past sales, weather) — all with built-in interpretability.

## **5.3 Foundation Models for Time Series**

| Model | By | Approach |
|-------|-----|---------|
| **TimesFM** | Google | Pre-trained on 100B+ time points, zero-shot forecasting |
| **Chronos** | Amazon | Tokenizes real values, uses language model architecture |
| **Lag-Llama** | Salesforce | Univariate probabilistic forecasting with Llama backbone |
| **Moirai** | Salesforce | Any-variate universal forecaster |

**Key concept:** Like LLMs for text, these models are pre-trained on massive time series corpora and can forecast **zero-shot** (without task-specific training). This is the "AI emergence" changing time series forecasting.

---

# **6. Evaluation and Validation**

## **6.1 Time Series Cross-Validation**

**NEVER use random train/test splits for time series.** Always preserve temporal order.

```
WRONG (random split):
Train: [Jan, Mar, May, Jul, Sep, Nov]
Test:  [Feb, Apr, Jun, Aug, Oct, Dec]  ← data leakage!

RIGHT (temporal split):
Train: [Jan → Aug]
Test:  [Sep → Dec]

BEST (expanding window / walk-forward):
Fold 1: Train [Jan-Jun]  → Test [Jul]
Fold 2: Train [Jan-Jul]  → Test [Aug]
Fold 3: Train [Jan-Aug]  → Test [Sep]
Fold 4: Train [Jan-Sep]  → Test [Oct]
```

## **6.2 Metrics**

| Metric | Formula Intuition | Good For | Watch Out |
|--------|------------------|----------|-----------|
| **MAE** | Mean absolute error | Interpretable in original units | Treats all errors equally |
| **RMSE** | Root mean squared error | Penalizes large errors more | Sensitive to outliers |
| **MAPE** | Mean absolute percentage error | Scale-independent, intuitive % | Undefined when actuals = 0, asymmetric |
| **sMAPE** | Symmetric MAPE | Fixes MAPE asymmetry | Still problematic near zero |
| **MASE** | MAE scaled by naive forecast MAE | Scale-independent, handles zeros | Less intuitive |
| **WAPE** | Weighted absolute percentage error | Good for intermittent series | Dominated by high-volume items |
| **Quantile Loss** | Pinball loss for prediction intervals | Probabilistic forecasting | Need to choose quantiles |

## **6.3 Baseline Models (Always Compare Against)**

| Baseline | What It Does | When It's Hard to Beat |
|----------|-------------|----------------------|
| **Naive** | Forecast = last observed value | Random walk / efficient market series |
| **Seasonal Naive** | Forecast = value from same season last year | Strong seasonal patterns |
| **Moving Average** | Forecast = average of last k values | Stable, low-noise series |
| **Linear Trend** | Forecast = linear extrapolation | Series with clear linear trend |

---

# **7. Anomaly Detection in Time Series**

## **7.1 Methods**

| Method | Approach | Best For |
|--------|----------|----------|
| **Statistical** | Z-score, Grubbs test, IQR on residuals | Simple, univariate |
| **Decomposition** | STL decomposition → flag large residuals | Seasonal series |
| **Prophet** | Fit Prophet → detect points outside prediction interval | Business metrics with seasonality |
| **Isolation Forest** | Random partitioning on feature-engineered time series | Multivariate, non-parametric |
| **Autoencoders** | Train to reconstruct normal patterns → high reconstruction error = anomaly | Complex patterns, multivariate |
| **Matrix Profile** | Finds discord (unusual subsequences) | Subsequence anomalies, motif discovery |

## **7.2 Changepoint Detection**

Detecting structural changes (trend shifts, variance changes):

| Method | What It Detects |
|--------|----------------|
| **CUSUM** | Cumulative sum detects mean shifts |
| **PELT** (Pruned Exact Linear Time) | Optimal multiple changepoints |
| **Bayesian Online Changepoint Detection** | Real-time, streaming changepoints |
| **Prophet changepoints** | Automatic trend changepoints in Prophet model |

---

# **8. Practical Considerations**

## **8.1 Common Pitfalls**

| Pitfall | Why It Happens | Fix |
|---------|---------------|-----|
| **Data leakage** | Using future info in features | Strict temporal ordering, walk-forward validation |
| **Not differencing** | Applying ARIMA to non-stationary data | ADF test → difference if needed |
| **Ignoring seasonality** | Modeling yearly data with daily noise | Decompose first, or use seasonal models |
| **Over-differencing** | Differencing a stationary series | Check ADF before and after differencing |
| **Confusing correlation with causation** | Granger causality ≠ true causality | Use causal methods (see file 33) |
| **Point forecasts only** | No uncertainty quantification | Use prediction intervals (quantile regression, conformal) |

## **8.2 When to Use What**

| Situation | Recommended Approach |
|-----------|---------------------|
| Quick baseline, few features | ARIMA / ETS / Prophet |
| Many exogenous features | XGBoost / LightGBM with temporal features |
| Long sequences, complex patterns | LSTM / TCN / Transformer |
| Multiple related series | VAR (classical) or global models (LightGBM on all series) |
| Zero-shot (no historical data for this series) | Foundation models (Chronos, TimesFM) |
| Need interpretability | Prophet (decomposition) or TFT (variable importance) |
| Need prediction intervals | Quantile regression, conformal prediction, TFT |

---

# **9. Interview Questions with Strong Answers**

---

## **Q1: "Explain the difference between ARIMA and Prophet. When would you use each?"**

> ARIMA is a parametric statistical model — ARIMA(p,d,q) where you explicitly specify autoregressive, differencing, and moving average orders. It requires stationarity, careful order selection via ACF/PACF or AIC, and doesn't handle missing data or outliers well. It's mathematically rigorous and works well for short to medium-term univariate forecasting when you understand the data's structure.
>
> Prophet is a decomposable model (trend + seasonality + holidays) built by Meta for business time series. It uses piecewise linear or logistic growth for trend, Fourier series for seasonality, and user-specified holiday effects. It handles missing data, outliers, and changepoints automatically. It's easier to use and more robust, but less statistically rigorous.
>
> **My rule:** ARIMA when I want statistical rigor and the series is well-behaved. Prophet when I have business time series with strong seasonality, holidays, and need a quick reliable baseline. At Jet2, we used LSTM/GRU for call volume forecasting because the patterns had complex nonlinear dependencies that neither ARIMA nor Prophet captured well.

---

## **Q2: "How do you handle non-stationarity in time series?"**

> First, I diagnose: visual inspection (plot the series, rolling mean/std), ADF test (H₀: non-stationary), and KPSS test (H₀: stationary). I use both because they test complementary hypotheses.
>
> If non-stationary, I apply **differencing** (first or seasonal). For a series with both trend and seasonality, I might need d=1 (trend) plus D=1 (seasonal) — this gives me SARIMA's I and D components.
>
> For variance non-stationarity (increasing spread over time), I apply **log or Box-Cox transform** before differencing.
>
> For ML approaches, non-stationarity is less critical because tree-based models learn arbitrary mappings. But I still engineer **differenced features** (y(t) - y(t-1)) and **percentage change features** to help the model capture changes rather than levels.

---

## **Q3: "Walk me through how you'd build a demand forecasting system."**

> 1. **Understand the business:** What's being forecasted (daily sales per SKU?), forecast horizon (7 days? 90 days?), decision it supports (inventory? staffing?), accuracy requirements.
>
> 2. **EDA:** Plot the series. Identify trend, seasonality (weekly? yearly?), changepoints (COVID? new product launch?). Check for missing data, outliers, intermittent zeros.
>
> 3. **Baselines first:** Seasonal naive (last year's same week) and moving average. If these are hard to beat, the signal might be weak.
>
> 4. **Feature engineering:** Calendar features (day of week, holidays, paydays), external signals (weather, promotions, competitor events), lag features (sales yesterday, last week, last year same day), rolling statistics.
>
> 5. **Model selection:** For hundreds of SKUs, I'd use a **global model** (one LightGBM trained on all SKUs with SKU-level features) rather than individual models. For key SKUs, compare against SARIMA and Prophet baselines.
>
> 6. **Validation:** Walk-forward cross-validation. Metrics: MASE (scale-independent, handles zeros) and WAPE (weighted by volume so high-volume SKUs matter more).
>
> 7. **Production:** Deploy with Airflow scheduling, drift monitoring on forecast accuracy, automatic fallback to seasonal naive if model accuracy degrades.

---

## **Q4: "How are foundation models changing time series forecasting?"**

> Models like Chronos (Amazon) and TimesFM (Google) are pre-trained on billions of time points across diverse domains. They can forecast **zero-shot** — you give them a time series they've never seen, and they produce reasonable forecasts without any fine-tuning.
>
> This is a paradigm shift. Previously, forecasting required per-series or per-domain model training. Now, for many practical use cases, a foundation model provides competitive accuracy with zero setup time.
>
> **Limitations:** They're still weaker than well-tuned domain-specific models on specialized data. They struggle with very long horizons and domain-specific patterns (e.g., financial markets with regime changes). And they're compute-heavy for inference.
>
> **My view:** Foundation models are the new baseline. Start with Chronos/TimesFM zero-shot, then fine-tune or build custom models only where the foundation model underperforms. This dramatically speeds up the forecasting development cycle.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│          TIME SERIES TAKEAWAYS                                    │
│                                                                   │
│  1. STATIONARITY is prerequisite for classical methods           │
│     (ADF test → difference → verify)                             │
│                                                                   │
│  2. ACF/PACF identify ARIMA orders — learn to read them         │
│                                                                   │
│  3. ALWAYS compare against BASELINES (naive, seasonal naive)    │
│                                                                   │
│  4. WALK-FORWARD validation — never random splits for TS        │
│                                                                   │
│  5. ML (XGBoost) needs FEATURE ENGINEERING to encode time       │
│                                                                   │
│  6. LSTM/GRU capture sequential dependencies but are being      │
│     overtaken by Transformers and foundation models              │
│                                                                   │
│  7. FOUNDATION MODELS (Chronos, TimesFM) = new zero-shot       │
│     baseline for forecasting                                     │
│                                                                   │
│  8. PREDICTION INTERVALS matter more than point forecasts       │
│     in production (quantile regression, conformal prediction)   │
│                                                                   │
│  9. ANOMALY DETECTION: decompose → flag residual outliers       │
│     (or use autoencoders for complex patterns)                   │
│                                                                   │
│  10. MASE > MAPE for evaluation (handles zeros, scale-free)    │
└──────────────────────────────────────────────────────────────────┘
```
