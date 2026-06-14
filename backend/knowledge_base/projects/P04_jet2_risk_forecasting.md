# P04 — Integrated Data Forecasting for Airline Operations (Jet2)

**Candidate:** Rahul Sharma | **Role:** Data Scientist | **Duration:** June 2022 – August 2024  
**Company:** Jet2 | **Industry:** Aviation / Travel  
**Resume Line:** *"Deployed LSTM/GRU models with walk-forward validation in Dataiku, scheduling automated jobs and alerts; ensured data integrity with Pydantic and sustained <2% model drift through continuous monitoring, integrating Tableau for insights."*  
**Document Scope:** Complete interview preparation — architecture, math, code, behavioral answers, and red-flag handling

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview-star)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
   - 2.1 [Data Integration & Pipeline](#21-data-integration--pipeline)
   - 2.2 [Feature Engineering](#22-feature-engineering)
   - 2.3 [Outlier Detection](#23-outlier-detection)
   - 2.4 [Model Comparison & Selection](#24-model-comparison--selection)
   - 2.5 [Walk-Forward Validation](#25-walk-forward-validation)
   - 2.6 [Hyperparameter Tuning with Optuna](#26-hyperparameter-tuning-with-optuna)
   - 2.7 [Deployment in Dataiku](#27-deployment-in-dataiku)
   - 2.8 [Monitoring & Drift Detection](#28-monitoring--drift-detection)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know](#4-topics-you-must-know)
5. [Interview Questions & Model Answers (25+)](#5-interview-questions--model-answers)
6. [Red Flags & How to Handle](#6-red-flags--how-to-handle)
7. [Key Takeaways](#7-key-takeaways)

---

# 1. Project Overview (STAR)

---

## 1.1 STAR Summary

| STAR | Detail |
|------|--------|
| **Situation** | Jet2's operations and finance teams relied on several disparate data streams — flight routes, passenger bookings, revenue, competitor activity, regional targets — but datasets arrived in complex, nested JSON formats with no unified schema. Decisions around pricing, crew allocation, and capacity planning were reactive rather than data-driven. |
| **Task** | Integrate these disparate datasets into a structured data warehouse. Build predictive models to forecast key operational risk scores (demand, revenue shortfall, route underperformance). Enable the business to make **proactive** adjustments. |
| **Action** | (1) Designed a data pipeline flattening nested JSON into Snowflake star schema. (2) Performed EDA with outlier detection (IQR, z-score). (3) Engineered domain features: rate-of-sale, holiday flags, competitor activity. (4) Trained and compared Random Forest, XGBoost, LSTM/GRU. (5) Used Optuna for hyperparameter tuning and walk-forward validation for time-series integrity. (6) Deployed in Dataiku with automated scheduling, Pydantic validation, and Tableau dashboards. |
| **Result** | Significant MAPE reduction (from ~18% to ~7%), enabling proactive pricing adjustments, optimized crew allocation, and faster market response. Sustained <2% model drift through continuous monitoring over 18+ months. |

---

## 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END ARCHITECTURE                               │
│                                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐     │
│  │  DATA        │    │  DATA WAREHOUSE  │    │  ML PIPELINE             │     │
│  │  SOURCES     │    │                  │    │                          │     │
│  │              │    │                  │    │  ┌───────────────────┐   │     │
│  │ Flight APIs ─┼───►│  Snowflake       │    │  │ Feature Engine    │   │     │
│  │ Booking Sys ─┼───►│  ┌────────────┐  │───►│  │ - Rate of Sale   │   │     │
│  │ Revenue DB  ─┼───►│  │ Star Schema│  │    │  │ - Holiday Flags  │   │     │
│  │ Competitor  ─┼───►│  │ Fact +     │  │    │  │ - Lag Features   │   │     │
│  │ Regional    ─┼───►│  │ Dimension  │  │    │  │ - Rolling Stats  │   │     │
│  │ Targets      │    │  └────────────┘  │    │  └────────┬──────────┘   │     │
│  └─────────────┘    └──────────────────┘    │           │              │     │
│        │                     │               │           ▼              │     │
│        │  JSON Flattening    │               │  ┌───────────────────┐   │     │
│        │  + Pydantic Valid.  │               │  │ Model Training    │   │     │
│        ▼                     │               │  │ RF / XGB / LSTM   │   │     │
│  ┌─────────────┐             │               │  │ + Optuna Tuning   │   │     │
│  │ Python ETL  │             │               │  │ + Walk-Forward CV │   │     │
│  │ Scripts     │             │               │  └────────┬──────────┘   │     │
│  │ - flatten() │             │               │           │              │     │
│  │ - validate()│             │               │           ▼              │     │
│  │ - load()    │             │               │  ┌───────────────────┐   │     │
│  └─────────────┘             │               │  │ Champion Model    │   │     │
│                              │               │  │ (LSTM/GRU)        │   │     │
│                              │               │  └────────┬──────────┘   │     │
│                              │               └───────────┼──────────────┘     │
│                              │                           │                    │
│                              │                           ▼                    │
│                    ┌─────────┴────────────────────────────────────┐           │
│                    │          DATAIKU  (MLOps)                     │           │
│                    │  ┌────────────┐  ┌──────────┐  ┌──────────┐  │           │
│                    │  │ Scheduled  │  │ Drift    │  │ Alerts   │  │           │
│                    │  │ Jobs       │  │ Monitor  │  │ System   │  │           │
│                    │  │ (Daily)    │  │ (<2%)    │  │          │  │           │
│                    │  └────────────┘  └──────────┘  └──────────┘  │           │
│                    └─────────────────────────┬────────────────────┘           │
│                                              │                                │
│                                              ▼                                │
│                              ┌─────────────────────────┐                      │
│                              │      TABLEAU             │                      │
│                              │  ┌───────┐ ┌──────────┐ │                      │
│                              │  │ Risk  │ │ Route    │ │                      │
│                              │  │ Score │ │ Perf.    │ │                      │
│                              │  │ Dash  │ │ Forecast │ │                      │
│                              │  └───────┘ └──────────┘ │                      │
│                              └─────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Sources | REST APIs, internal DBs, CSV feeds | Flight routes, bookings, revenue, competitor data |
| ETL / Validation | Python, Pydantic, custom scripts | JSON flattening, schema validation, type enforcement |
| Data Warehouse | Snowflake | Centralized star schema, historical storage |
| EDA / Analysis | pandas, numpy, scipy, matplotlib, seaborn | Outlier detection, statistical analysis, visualization |
| Feature Engineering | pandas, numpy, custom modules | Rate-of-sale, holiday flags, lag/rolling features |
| Modeling | scikit-learn, XGBoost, PyTorch (LSTM/GRU) | Random Forest, Gradient Boosting, Deep Learning |
| Hyperparameter Tuning | Optuna | Bayesian optimization with TPE sampler + pruning |
| Validation | Walk-forward (custom implementation) | Time-series cross-validation without data leakage |
| MLOps / Deployment | Dataiku | Pipeline orchestration, scheduling, monitoring |
| Visualization | Tableau | Executive dashboards, risk score views |
| Monitoring | Custom + Dataiku | MAPE tracking, PSI for drift, alert triggers |

---

# 2. Deep Technical Walkthrough

---

## 2.1 Data Integration & Pipeline

### The Problem with Nested JSON

Airline operational data arrives from multiple APIs in deeply nested JSON structures. A single booking record might contain nested passenger arrays, nested route segments, nested fare breakdowns, etc.

**Example raw JSON (simplified):**

```json
{
  "flight_id": "JET2-1042",
  "route": {
    "origin": {"code": "LBA", "city": "Leeds"},
    "destination": {"code": "PMI", "city": "Palma"}
  },
  "bookings": [
    {
      "booking_ref": "BK-9981",
      "passengers": [
        {"name": "John Doe", "fare_class": "economy", "price": 189.50},
        {"name": "Jane Doe", "fare_class": "economy", "price": 189.50}
      ],
      "add_ons": {"luggage": true, "seat_selection": true, "meals": 2}
    }
  ],
  "competitor_data": {
    "ryanair": {"same_route_price": 159.00, "frequency": 3},
    "easyjet": {"same_route_price": 175.00, "frequency": 2}
  },
  "targets": {
    "region": "Balearics",
    "load_factor_target": 0.88,
    "revenue_target_gbp": 42000
  }
}
```

### JSON Flattening Strategy

```python
import json
import pandas as pd
from typing import Any

def flatten_json(nested: dict, prefix: str = '', sep: str = '_') -> dict:
    """Recursively flatten nested JSON into a flat dictionary."""
    flat = {}
    for key, value in nested.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(flatten_json(item, f"{new_key}{sep}{i}", sep))
                else:
                    flat[f"{new_key}{sep}{i}"] = item
        else:
            flat[new_key] = value
    return flat

# Usage
with open('flight_data.json') as f:
    raw = json.load(f)

flat_records = [flatten_json(record) for record in raw]
df = pd.DataFrame(flat_records)
```

### Snowflake Star Schema Design

```
┌─────────────────────────────────────────────────────────┐
│                  SNOWFLAKE STAR SCHEMA                    │
│                                                           │
│                  ┌──────────────────┐                     │
│                  │   FACT_BOOKINGS  │                     │
│                  │  ──────────────  │                     │
│                  │  booking_id (PK) │                     │
│                  │  flight_id (FK)  │                     │
│                  │  route_id  (FK)  │                     │
│                  │  date_id   (FK)  │                     │
│                  │  pax_count       │                     │
│                  │  revenue_gbp     │                     │
│                  │  load_factor     │                     │
│                  │  risk_score      │                     │
│                  └──────┬───────────┘                     │
│           ┌─────────────┼─────────────┐                   │
│           │             │             │                   │
│    ┌──────▼──────┐ ┌────▼─────┐ ┌────▼──────┐            │
│    │ DIM_ROUTE   │ │ DIM_DATE │ │ DIM_FLIGHT│            │
│    │ ──────────  │ │ ──────── │ │ ────────  │            │
│    │ route_id    │ │ date_id  │ │ flight_id │            │
│    │ origin_code │ │ date     │ │ aircraft  │            │
│    │ dest_code   │ │ day_of_wk│ │ capacity  │            │
│    │ region      │ │ month    │ │ departure │            │
│    │ distance_km │ │ quarter  │ │ status    │            │
│    │ comp_count  │ │ is_holid │ │ crew_id   │            │
│    └─────────────┘ │ is_event │ └───────────┘            │
│                    │ season   │                           │
│                    └──────────┘                           │
│                                                           │
│    ┌──────────────┐  ┌───────────────┐                    │
│    │ DIM_COMPET.  │  │ FACT_TARGETS  │                    │
│    │ ──────────── │  │ ───────────── │                    │
│    │ competitor_id│  │ target_id     │                    │
│    │ airline_name │  │ route_id (FK) │                    │
│    │ route_id(FK) │  │ date_id  (FK) │                    │
│    │ price_gbp    │  │ lf_target     │                    │
│    │ frequency    │  │ rev_target    │                    │
│    └──────────────┘  └───────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

**Why star schema:**
- Optimized for analytical (OLAP) queries — Snowflake excels at this
- Denormalized dimensions reduce JOINs, faster query performance
- Clean separation of facts (measurable events) and dimensions (descriptive context)
- Easy for Tableau to consume via live/extract connections

### Pydantic Validation

Pydantic enforces data contracts at the pipeline boundary, catching type mismatches and missing fields before data enters Snowflake.

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date

class RouteRecord(BaseModel):
    origin_code: str = Field(..., min_length=3, max_length=3)
    destination_code: str = Field(..., min_length=3, max_length=3)
    region: str
    distance_km: float = Field(gt=0)

class BookingRecord(BaseModel):
    booking_id: str
    flight_id: str
    route: RouteRecord
    booking_date: date
    departure_date: date
    pax_count: int = Field(ge=1, le=300)
    revenue_gbp: float = Field(ge=0)
    load_factor: float = Field(ge=0.0, le=1.0)

    @validator('departure_date')
    def departure_after_booking(cls, v, values):
        if 'booking_date' in values and v < values['booking_date']:
            raise ValueError('departure_date must be >= booking_date')
        return v

    @validator('revenue_gbp')
    def revenue_consistency(cls, v, values):
        if 'pax_count' in values and values['pax_count'] > 0:
            per_pax = v / values['pax_count']
            if per_pax > 5000:  # sanity check
                raise ValueError(f'Revenue per pax {per_pax:.2f} is unrealistic')
        return v

class CompetitorRecord(BaseModel):
    competitor_name: str
    route_id: str
    price_gbp: Optional[float] = Field(default=None, ge=0)
    frequency_per_week: Optional[int] = Field(default=None, ge=0)

# Validation in pipeline
def validate_and_load(raw_records: List[dict]) -> pd.DataFrame:
    valid, invalid = [], []
    for rec in raw_records:
        try:
            validated = BookingRecord(**rec)
            valid.append(validated.dict())
        except Exception as e:
            invalid.append({'record': rec, 'error': str(e)})

    if invalid:
        log_validation_errors(invalid)  # alert on bad records
        print(f"⚠ {len(invalid)}/{len(raw_records)} records failed validation")

    return pd.DataFrame(valid)
```

**Why Pydantic over manual checks:**
- Declarative schema — self-documenting
- Automatic type coercion (e.g., string "42" → int 42)
- Custom validators for business rules
- Clear error messages for debugging
- Can generate JSON Schema for documentation

---

## 2.2 Feature Engineering

### 2.2.1 Rate-of-Sale (ROS)

**What it is:** Rate-of-sale measures how quickly seats are being booked on a given flight/route over a specific time window. It is the most critical pricing signal in airline revenue management.

**Formula:**

$$
\text{ROS}(t) = \frac{\text{Bookings in window } [t - w,\ t]}{\text{Window size } w}
$$

For example, with a 7-day window:

$$
\text{ROS}_{\text{7d}}(t) = \frac{\text{Bookings in last 7 days}}{7}
$$

**Derived features:**

$$
\text{ROS Acceleration} = \text{ROS}_{\text{7d}}(t) - \text{ROS}_{\text{7d}}(t - 7)
$$

$$
\text{ROS Ratio} = \frac{\text{ROS}_{\text{current}}}{\text{ROS}_{\text{same period last year}}}
$$

**Why it matters for airline pricing:**
- **High ROS (selling fast):** Opportunity to raise prices — demand is strong
- **Low ROS (selling slow):** Risk of empty seats — may need promotional fares
- **ROS acceleration:** If ROS is increasing, demand is picking up — hold/raise prices
- **ROS vs. historical:** Compares to same period last year — seasonal context

```python
def compute_rate_of_sale(df: pd.DataFrame, windows: list = [3, 7, 14, 28]) -> pd.DataFrame:
    """Compute rate-of-sale features for multiple windows."""
    df = df.sort_values(['route_id', 'departure_date', 'booking_date'])

    for w in windows:
        # Rolling bookings count over window
        df[f'ros_{w}d'] = (
            df.groupby(['route_id', 'departure_date'])['pax_count']
            .transform(lambda x: x.rolling(window=w, min_periods=1).sum() / w)
        )

    # ROS acceleration (7-day ROS change week-over-week)
    df['ros_7d_acceleration'] = (
        df.groupby(['route_id', 'departure_date'])['ros_7d']
        .transform(lambda x: x.diff(7))
    )

    # ROS ratio vs. same period last year
    df['ros_yoy_ratio'] = df['ros_7d'] / df['ros_7d_ly'].replace(0, np.nan)

    return df
```

### 2.2.2 Holiday Flags & Special Events

Airline demand is heavily influenced by school holidays, bank holidays, and special events.

```python
import numpy as np

# UK school holiday periods (approximate)
UK_SCHOOL_HOLIDAYS = {
    'february_half_term': {'month': 2, 'day_range': (10, 18)},
    'easter':             {'month': 4, 'day_range': (1, 17)},
    'may_half_term':      {'month': 5, 'day_range': (27, 31)},
    'summer':             {'month_range': (7, 8)},
    'october_half_term':  {'month': 10, 'day_range': (21, 29)},
    'christmas':          {'month': 12, 'day_range': (18, 31)},
}

UK_BANK_HOLIDAYS = [
    # Manually maintained or fetched via UK gov API
    '2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01',
    '2023-05-29', '2023-08-28', '2023-12-25', '2023-12-26',
]

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday and special event binary flags."""
    dt = pd.to_datetime(df['departure_date'])

    # School holiday flag
    df['is_school_holiday'] = 0
    for name, period in UK_SCHOOL_HOLIDAYS.items():
        if 'month_range' in period:
            mask = dt.dt.month.between(*period['month_range'])
        else:
            mask = (dt.dt.month == period['month']) & \
                   (dt.dt.day.between(*period['day_range']))
        df.loc[mask, 'is_school_holiday'] = 1

    # Bank holiday flag
    df['is_bank_holiday'] = dt.dt.strftime('%Y-%m-%d').isin(UK_BANK_HOLIDAYS).astype(int)

    # Days until departure (demand curve position)
    df['days_to_departure'] = (dt - pd.to_datetime(df['booking_date'])).dt.days

    # Weekend departure flag
    df['is_weekend_departure'] = (dt.dt.dayofweek >= 5).astype(int)

    # Season encoding (cyclical)
    df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)

    # Day of week (cyclical)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    return df
```

**Why cyclical encoding:** Month 12 (December) and month 1 (January) are sequential, but one-hot or ordinal encoding treats them as distant. Sine/cosine encoding preserves this circular relationship.

### 2.2.3 Competitor Activity Features

```python
def add_competitor_features(df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Merge competitor pricing and frequency data."""
    # Average competitor price for same route
    comp_agg = comp_df.groupby(['route_id', 'date']).agg(
        avg_comp_price=('price_gbp', 'mean'),
        min_comp_price=('price_gbp', 'min'),
        max_comp_price=('price_gbp', 'max'),
        num_competitors=('competitor_name', 'nunique'),
        total_comp_frequency=('frequency_per_week', 'sum'),
    ).reset_index()

    df = df.merge(comp_agg, on=['route_id', 'date'], how='left')

    # Price differential: our price vs. competitors
    df['price_diff_vs_avg_comp'] = df['our_price_gbp'] - df['avg_comp_price']
    df['price_ratio_vs_min_comp'] = df['our_price_gbp'] / df['min_comp_price'].replace(0, np.nan)

    return df
```

### 2.2.4 Lagged Features & Rolling Statistics

```python
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling features for time-series modeling."""
    df = df.sort_values(['route_id', 'date'])
    group = df.groupby('route_id')

    # Lagged features
    for lag in [1, 3, 7, 14, 28]:
        df[f'bookings_lag_{lag}d'] = group['daily_bookings'].shift(lag)
        df[f'revenue_lag_{lag}d'] = group['daily_revenue'].shift(lag)
        df[f'load_factor_lag_{lag}d'] = group['load_factor'].shift(lag)

    # Rolling statistics
    for window in [7, 14, 28]:
        df[f'bookings_roll_mean_{window}d'] = (
            group['daily_bookings'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        )
        df[f'bookings_roll_std_{window}d'] = (
            group['daily_bookings'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        )
        df[f'revenue_roll_mean_{window}d'] = (
            group['daily_revenue'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        )

    # Exponentially weighted moving average (more weight to recent)
    df['bookings_ewm_7d'] = (
        group['daily_bookings'].transform(lambda x: x.ewm(span=7).mean())
    )

    return df
```

### Complete Feature Summary Table

| Category | Feature | Type | Rationale |
|----------|---------|------|-----------|
| **Rate-of-Sale** | `ros_3d`, `ros_7d`, `ros_14d`, `ros_28d` | Float | Demand velocity at multiple granularities |
| | `ros_7d_acceleration` | Float | Is demand speeding up or slowing down? |
| | `ros_yoy_ratio` | Float | Demand vs. same period last year |
| **Calendar** | `is_school_holiday` | Binary | UK school holidays drive leisure travel |
| | `is_bank_holiday` | Binary | Bank holidays create long-weekend demand |
| | `is_weekend_departure` | Binary | Weekend vs. weekday demand patterns |
| | `days_to_departure` | Int | Position on demand curve (early vs. last-minute) |
| | `month_sin`, `month_cos` | Float | Cyclical month encoding for seasonality |
| | `dow_sin`, `dow_cos` | Float | Cyclical day-of-week encoding |
| **Competitor** | `avg_comp_price` | Float | Competitive pricing landscape |
| | `price_diff_vs_avg_comp` | Float | Our price premium/discount vs. market |
| | `num_competitors` | Int | Market saturation on the route |
| **Temporal (Lags)** | `bookings_lag_{1,3,7,14,28}d` | Float | Autoregressive signals |
| | `revenue_lag_{1,3,7,14,28}d` | Float | Revenue momentum |
| **Temporal (Rolling)** | `bookings_roll_mean_{7,14,28}d` | Float | Smoothed trend |
| | `bookings_roll_std_{7,14,28}d` | Float | Volatility / uncertainty |
| | `bookings_ewm_7d` | Float | Recent-weighted demand trend |
| **Route** | `distance_km`, `region` | Float/Cat | Route characteristics |
| | `capacity` | Int | Aircraft seat count |

---

## 2.3 Outlier Detection

### 2.3.1 IQR Method

The Interquartile Range (IQR) method is a robust, non-parametric approach to outlier detection.

**Formula:**

$$
\text{IQR} = Q_3 - Q_1
$$

$$
\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}
$$

$$
\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}
$$

Any value below the lower bound or above the upper bound is flagged as an outlier. The 1.5 multiplier covers approximately 99.3% of normally distributed data.

**Why 1.5?** John Tukey's empirical choice — it balances sensitivity (catching real outliers) and specificity (not flagging normal variation). Using 3.0×IQR flags only extreme outliers.

```python
def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """Detect outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    is_outlier = (df[column] < lower) | (df[column] > upper)
    print(f"IQR Outliers in '{column}': {is_outlier.sum()} / {len(df)} "
          f"({100*is_outlier.mean():.1f}%) | Bounds: [{lower:.2f}, {upper:.2f}]")
    return is_outlier
```

```
┌─────────────────────────────────────────────────────────────────┐
│                   IQR BOX PLOT ANATOMY                          │
│                                                                 │
│  Outliers     Lower       Q1    Median   Q3       Upper  Outlier│
│     ●        Fence                                 Fence    ●   │
│     ●          │                                     │      ●   │
│  ───●──────────┼─────┬──────┬──────┬─────┼──────────┼──────●──  │
│                │     │      │      │     │          │           │
│                │     │      │      │     │          │           │
│            Q1-1.5×IQR│◄────IQR────►│   Q3+1.5×IQR              │
│                      Q1    Med    Q3                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3.2 Z-Score Method

The z-score measures how many standard deviations a data point is from the mean. Assumes approximately normal distribution.

**Formula:**

$$
z = \frac{x - \mu}{\sigma}
$$

Typical threshold: $|z| > 3$ (only 0.27% of normal data exceeds this).

```python
from scipy import stats

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using z-score method."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    is_outlier = pd.Series(False, index=df.index)
    is_outlier.loc[df[column].notna()] = z_scores > threshold

    print(f"Z-score Outliers in '{column}': {is_outlier.sum()} / {len(df)} "
          f"({100*is_outlier.mean():.1f}%) | Threshold: |z| > {threshold}")
    return is_outlier
```

### When to Use Which

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **IQR** | Skewed data, unknown distribution | Robust to extreme values, non-parametric | May flag too many in heavy-tailed distributions |
| **Z-score** | Near-normal distributions | Simple, interpretable, well-understood threshold | Sensitive to outliers themselves (mean/std shift) |

**In this project:** Used IQR for revenue and booking data (often right-skewed) and z-score for rate-of-sale metrics (more normally distributed after log transform).

---

## 2.4 Model Comparison & Selection

### 2.4.1 Random Forest

**How it works (Bagging):**

```
┌──────────────────────────────────────────────────────────┐
│                 RANDOM FOREST (BAGGING)                   │
│                                                           │
│  Training Data ──► Bootstrap Sample 1 ──► Decision Tree 1 │
│       │          ► Bootstrap Sample 2 ──► Decision Tree 2 │
│       │          ► Bootstrap Sample 3 ──► Decision Tree 3 │
│       │          ► ...                                    │
│       │          ► Bootstrap Sample N ──► Decision Tree N │
│       │                                        │          │
│       │                                        ▼          │
│       │                              ┌─────────────────┐  │
│       │                              │   AGGREGATE     │  │
│       │                              │  Mean (regress) │  │
│       │                              │  Vote (classif) │  │
│       │                              └─────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Key concepts:**
- **Bootstrap aggregating (bagging):** Each tree trained on a random subset of rows (with replacement)
- **Feature randomness:** At each split, only a random subset of features considered ($\sqrt{p}$ for classification, $p/3$ for regression)
- **Reduces variance** without increasing bias
- **Out-of-Bag (OOB) error:** Each tree hasn't seen ~37% of data — use this as a free validation set

**Feature importance:**
- **Mean Decrease in Impurity (MDI):** Average reduction in MSE across all splits using that feature
- **Permutation importance:** Shuffle a feature's values and measure prediction degradation

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)

print(f"OOB R²: {rf_model.oob_score_:.4f}")

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = importances.nlargest(15)
```

### 2.4.2 XGBoost

**How it works (Gradient Boosting):**

```
┌──────────────────────────────────────────────────────────────┐
│                  XGBOOST (GRADIENT BOOSTING)                  │
│                                                               │
│  Iteration 1: Base prediction (e.g., mean of target)          │
│       │                                                       │
│       ▼                                                       │
│  Compute residuals = y_true - prediction                      │
│       │                                                       │
│       ▼                                                       │
│  Iteration 2: Fit tree to RESIDUALS                           │
│       │  prediction += learning_rate × tree_2(X)              │
│       ▼                                                       │
│  Compute NEW residuals                                        │
│       │                                                       │
│       ▼                                                       │
│  Iteration 3: Fit tree to NEW residuals                       │
│       │  prediction += learning_rate × tree_3(X)              │
│       ▼                                                       │
│  ...repeat for N rounds...                                    │
│                                                               │
│  Final: ŷ = base + η·tree₁(X) + η·tree₂(X) + ... + η·treeₙ │
│                                                               │
│  Key: each tree corrects the mistakes of the ensemble so far  │
└──────────────────────────────────────────────────────────────┘
```

**XGBoost innovations over plain gradient boosting:**
1. **Regularization:** L1 ($\alpha$) and L2 ($\lambda$) penalties on leaf weights
2. **Second-order gradients:** Uses both gradient and Hessian for better splits
3. **Column subsampling:** Random feature subset per tree (like RF)
4. **Weighted quantile sketch:** Efficient approximate split finding
5. **Sparsity-aware:** Handles missing values natively
6. **Pruning:** max_depth + post-pruning instead of stopping early

**SHAP values for interpretability:**

```python
import xgboost as xgb
import shap

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    eval_metric='mae',
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# SHAP analysis
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
```

### 2.4.3 LSTM / GRU (Deep Learning for Sequences)

**LSTM Architecture:**

```
┌───────────────────────────────────────────────────────────────┐
│                     LSTM CELL                                  │
│                                                                │
│  Previous cell state C(t-1) ──────────────────────────────►    │
│            │                    │           │           │       │
│            │              ┌────▼───┐  ┌────▼───┐  ┌────▼───┐  │
│            │              │ Forget  │  │ Input  │  │ Output │  │
│            │              │  Gate   │  │  Gate  │  │  Gate  │  │
│            │              │ σ(Wf·   │  │ σ(Wi·  │  │ σ(Wo·  │  │
│            │              │ [h,x]+bf│  │ [h,x]  │  │ [h,x]  │  │
│            │              │   )     │  │ +bi)   │  │ +bo)   │  │
│            │              └────┬───┘  └───┬────┘  └────┬───┘  │
│            │                   │          │             │       │
│            │                   │    ┌─────▼─────┐      │       │
│            │                   │    │ Candidate  │      │       │
│            │                   │    │ C̃ = tanh(  │      │       │
│            │                   │    │ Wc·[h,x]  │      │       │
│            │                   │    │  + bc)     │      │       │
│            │                   │    └─────┬─────┘      │       │
│            │                   │          │             │       │
│            ▼                   ▼          ▼             │       │
│    C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)               │       │
│            │                                            │       │
│            ▼                                            │       │
│    h(t) = o(t) ⊙ tanh(C(t))  ◄─────────────────────────┘       │
│            │                                                    │
│            ▼                                                    │
│    Output h(t) ──► next timestep / prediction layer             │
└───────────────────────────────────────────────────────────────┘
```

**Gate Mechanisms:**

| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | What to discard from cell state (e.g., an old seasonal pattern that is no longer relevant) |
| **Input** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | What new information to store (e.g., a sudden demand spike) |
| **Candidate** | $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ | New candidate values to add to cell state |
| **Cell update** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | Combine old (filtered) + new information |
| **Output** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | What part of cell state to expose as output |
| **Hidden state** | $h_t = o_t \odot \tanh(C_t)$ | Output / passed to next timestep |

**GRU (Gated Recurrent Unit):**

Simplified version of LSTM with **2 gates** instead of 3:

| Gate | Formula | Maps to LSTM |
|------|---------|-------------|
| **Update** | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ | Combined forget + input gate |
| **Reset** | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ | Controls how much past info to forget |
| **Hidden** | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ | Simplified cell update |

**GRU advantages:** Fewer parameters (faster training, less overfitting on smaller datasets), comparable performance to LSTM for many tasks.

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # single output: risk score / demand forecast
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        return self.fc(last_hidden)

# Sequence creation for time-series
def create_sequences(data: np.ndarray, targets: np.ndarray, seq_length: int = 30):
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

# Training loop
model = LSTMForecaster(input_dim=len(feature_columns), hidden_dim=128, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

### 2.4.4 Model Comparison Results

| Model | MAPE | MAE | RMSE | Training Time | Interpretability |
|-------|------|-----|------|---------------|-----------------|
| Random Forest | 11.2% | 4.8 | 6.3 | ~5 min | High (feature importance) |
| XGBoost | 9.1% | 3.9 | 5.1 | ~8 min | High (SHAP) |
| **LSTM** | **7.2%** | **3.1** | **4.2** | ~45 min | Low (black box) |
| GRU | 7.8% | 3.4 | 4.5 | ~35 min | Low (black box) |

### Why LSTM Won for This Use Case

1. **Temporal dependencies:** Airline demand has complex, multi-horizon patterns (weekly, monthly, seasonal, annual). LSTM's cell state can maintain information across these long horizons.
2. **Sequential nature:** Booking patterns are inherently sequential — the order matters. RF and XGBoost see each row independently.
3. **Non-linear temporal interactions:** The gates learn when to remember and forget, capturing patterns like "demand always dips 3 weeks before summer departure then surges."
4. **Multi-variate sequences:** LSTM naturally handles multiple features evolving together over time.

**Why not ARIMA/Prophet?**
- ARIMA: Univariate, requires stationarity, struggles with many exogenous features
- Prophet: Good for single time series with strong seasonality, but we had ~200 routes each needing forecasts + many exogenous features
- LSTM: Handles multi-variate, multi-route forecasting with arbitrary feature sets

---

## 2.5 Walk-Forward Validation

### What It Is

Walk-forward validation (also called **time-series cross-validation**) is the gold standard for validating time-series models. Unlike k-fold CV, it respects the temporal order of data.

**Key principle:** The model must never see future data during training. Only past data trains, only future data validates.

### Expanding Window vs. Sliding Window

```
┌──────────────────────────────────────────────────────────────────────┐
│                   EXPANDING WINDOW WALK-FORWARD                      │
│                                                                      │
│  Time ──────────────────────────────────────────────────►            │
│                                                                      │
│  Fold 1: [===TRAIN===]  [VAL]                                       │
│  Fold 2: [=====TRAIN=====]  [VAL]                                   │
│  Fold 3: [========TRAIN========]  [VAL]                             │
│  Fold 4: [==========TRAIN==========]  [VAL]                         │
│  Fold 5: [=============TRAIN=============]  [VAL]                   │
│                                                                      │
│  ✓ Training set grows each fold                                      │
│  ✓ More training data → potentially better models                    │
│  ✓ Tests on progressively more recent data                           │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                   SLIDING WINDOW WALK-FORWARD                        │
│                                                                      │
│  Fold 1: [===TRAIN===]  [VAL]                                       │
│  Fold 2:    [===TRAIN===]  [VAL]                                    │
│  Fold 3:       [===TRAIN===]  [VAL]                                 │
│  Fold 4:          [===TRAIN===]  [VAL]                              │
│  Fold 5:             [===TRAIN===]  [VAL]                           │
│                                                                      │
│  ✓ Fixed-size training window                                        │
│  ✓ Better when old data is less relevant (concept drift)             │
│  ✓ Consistent computational cost per fold                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Why Not K-Fold for Time Series?

```
┌──────────────────────────────────────────────────────────────────────┐
│                PROBLEM WITH K-FOLD ON TIME SERIES                    │
│                                                                      │
│  Standard 5-fold CV might produce:                                   │
│                                                                      │
│  Fold 1: [VAL] [TRAIN] [TRAIN] [TRAIN] [TRAIN]                     │
│            │       │                                                 │
│            └───────┘  ← FUTURE data used to train,                   │
│                         then predict PAST = DATA LEAKAGE!            │
│                                                                      │
│  Fold 3: [TRAIN] [TRAIN] [VAL] [TRAIN] [TRAIN]                     │
│                              │     │                                 │
│                              └─────┘  ← Model sees future data       │
│                                        in training set               │
│                                                                      │
│  Result: OVERLY OPTIMISTIC metrics that don't generalize!            │
└──────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from typing import List, Tuple
import numpy as np

def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: int = 365,  # at least 1 year of training
    test_size: int = 30,         # 30-day forecast horizon
    expanding: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward train/test splits."""
    splits = []
    total_test = n_splits * test_size

    if min_train_size + total_test > n_samples:
        raise ValueError("Not enough data for requested splits")

    for i in range(n_splits):
        test_end = n_samples - (n_splits - i - 1) * test_size
        test_start = test_end - test_size

        if expanding:
            train_start = 0
        else:
            train_start = max(0, test_start - min_train_size)

        train_idx = np.arange(train_start, test_start)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits

# Usage with model evaluation
def evaluate_walk_forward(model_class, X, y, n_splits=5):
    """Evaluate model using walk-forward validation."""
    splits = walk_forward_split(len(X), n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        fold_metrics.append({'fold': fold+1, 'mape': mape, 'mae': mae, 'rmse': rmse})
        print(f"Fold {fold+1}: MAPE={mape:.2f}%, MAE={mae:.2f}, RMSE={rmse:.2f}")

    avg = pd.DataFrame(fold_metrics).mean()
    print(f"\nAverage: MAPE={avg['mape']:.2f}%, MAE={avg['mae']:.2f}, RMSE={avg['rmse']:.2f}")
    return fold_metrics
```

### Prevents Data Leakage — Why This Matters

| Leakage Type | Standard K-Fold | Walk-Forward |
|-------------|----------------|--------------|
| Future → Past leakage | YES (future data trains model to predict past) | NO (strict temporal order) |
| Seasonal pattern leakage | YES (model sees next year's seasonality in training) | NO (model must predict unseen future seasons) |
| Overly optimistic metrics | YES (metrics inflated by 3-8% typically) | NO (realistic performance estimate) |
| Distribution shift detection | NO (mixed time periods mask drift) | YES (each fold is a real forward-in-time test) |

---

## 2.6 Hyperparameter Tuning with Optuna

### What is Optuna?

Optuna is a next-generation hyperparameter optimization framework that uses **Bayesian optimization** (specifically **TPE — Tree-structured Parzen Estimator**) instead of grid/random search.

### TPE (Tree-structured Parzen Estimator) — How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                   TPE BAYESIAN OPTIMIZATION                          │
│                                                                      │
│  Instead of modeling P(score | hyperparams) directly:                │
│                                                                      │
│  TPE models two distributions:                                       │
│                                                                      │
│    l(x) = P(hyperparams | score < threshold)  ← "good" trials       │
│    g(x) = P(hyperparams | score ≥ threshold)  ← "bad" trials        │
│                                                                      │
│  Next trial picks hyperparams that MAXIMIZE:                         │
│                                                                      │
│                        l(x)                                          │
│    EI(x) ∝  ─────────────────                                       │
│                        g(x)                                          │
│                                                                      │
│  → Regions where good configs are likely and bad configs unlikely    │
│                                                                      │
│  Iteration flow:                                                     │
│  ┌──────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐        │
│  │Trial │──►│ Evaluate │──►│ Update    │──►│ Sample next  │──►...   │
│  │  1   │   │ Score    │   │ l(x),g(x) │   │ from l(x)/g(x)│       │
│  └──────┘   └──────────┘   └───────────┘   └──────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

### Why TPE > Grid Search / Random Search

| Method | Trials Needed | Finds Optimum | Handles Interactions | Adaptive |
|--------|--------------|---------------|---------------------|----------|
| Grid Search | $O(k^n)$ exponential | Exhaustive but slow | No | No |
| Random Search | ~60 for 95% coverage | Good for low-dimensional | Partially | No |
| **TPE (Optuna)** | ~30-100 typically | Focuses on promising regions | **Yes** | **Yes** |

### Pruning

Optuna's **MedianPruner** stops unpromising trials early:

```
Trial 1: epoch 1=0.15, epoch 2=0.12, epoch 3=0.10, ... epoch 50=0.07  ← COMPLETE
Trial 2: epoch 1=0.18, epoch 2=0.17, epoch 3=0.16  ← PRUNED (worse than median)
Trial 3: epoch 1=0.14, epoch 2=0.11, epoch 3=0.09, ... epoch 50=0.06  ← COMPLETE
Trial 4: epoch 1=0.20, epoch 2=0.19  ← PRUNED (worse than median at step 2)
```

This saves enormous compute time — bad hyperparameter combinations are killed early.

### Full Optuna Implementation

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.Trial) -> float:
    """Optuna objective for LSTM hyperparameter tuning."""

    # Define search space
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    seq_length = trial.suggest_int('seq_length', 14, 60, step=7)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # Build model with suggested hyperparams
    model = LSTMForecaster(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Walk-forward validation (outer loop)
    fold_mapes = []
    for fold_idx, (train_idx, val_idx) in enumerate(wf_splits):
        X_train_seq, y_train_seq = create_sequences(X_scaled[train_idx], y[train_idx], seq_length)
        X_val_seq, y_val_seq = create_sequences(X_scaled[val_idx], y[val_idx], seq_length)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                          torch.tensor(y_train_seq, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False  # no shuffle for time series!
        )

        # Training
        model.train()
        for epoch in range(50):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X).squeeze()
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Pruning: report intermediate value
            trial.report(epoch_loss / len(train_loader), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val_seq, dtype=torch.float32)).squeeze().numpy()
        mape = np.mean(np.abs((y_val_seq - val_pred) / y_val_seq)) * 100
        fold_mapes.append(mape)

    return np.mean(fold_mapes)

# Run optimization
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best MAPE: {study.best_value:.2f}%")
print(f"Best params: {study.best_params}")
```

**Best hyperparameters found (example):**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 192 |
| `num_layers` | 2 |
| `dropout` | 0.25 |
| `learning_rate` | 3.2e-4 |
| `seq_length` | 28 (4 weeks) |
| `batch_size` | 64 |
| `weight_decay` | 1.5e-5 |

---

## 2.7 Deployment in Dataiku

### What is Dataiku?

Dataiku DSS (Data Science Studio) is an enterprise MLOps platform that provides:
- Visual pipeline building + code (Python, R, SQL)
- Automated scheduling and orchestration
- Model versioning and A/B testing
- Monitoring dashboards and alerting
- Integration with Snowflake, Tableau, and other tools

### Why Dataiku (Not Custom Deployment)?

| Factor | Dataiku | Custom (Flask/FastAPI + Airflow) |
|--------|---------|-------------------------------|
| Setup time | Days | Weeks |
| Team adoption | Non-technical stakeholders can view/trigger | Engineers only |
| Scheduling | Built-in with UI | Requires Airflow/Cron setup |
| Monitoring | Built-in drift/performance dashboards | Custom implementation |
| Compliance/Audit | Enterprise-grade logging | Manual setup |
| Cost to maintain | Minimal (managed) | Ongoing DevOps effort |

**For Jet2's use case:** The operations team needed to interact with model outputs, and the data science team was small. Dataiku provided the best balance of power and accessibility.

### Deployment Architecture in Dataiku

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATAIKU  DSS  PIPELINE                         │
│                                                                   │
│  ┌──────────────┐                                                │
│  │ SCENARIO     │  (Runs daily at 6:00 AM UTC)                    │
│  │ "Daily       │                                                │
│  │  Forecast"   │                                                │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  STEP 1      │    │  STEP 2      │    │  STEP 3      │        │
│  │  Data Pull   │───►│  Feature     │───►│  Inference   │        │
│  │  (Snowflake  │    │  Engineering │    │  (LSTM       │        │
│  │   query)     │    │  (Python     │    │   model)     │        │
│  │              │    │   recipe)    │    │              │        │
│  └──────────────┘    └──────────────┘    └──────┬───────┘        │
│                                                  │               │
│         ┌────────────────────────────────────────┘               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │  STEP 4      │    │  STEP 5      │    │  STEP 6      │        │
│  │  Post-       │───►│  Drift       │───►│  Write to    │        │
│  │  Processing  │    │  Check       │    │  Snowflake   │        │
│  │  + Pydantic  │    │  (<2%        │    │  + Trigger   │        │
│  │  Validation  │    │   threshold) │    │  Tableau     │        │
│  └──────────────┘    └──────────────┘    └──────┬───────┘        │
│                                                  │               │
│                              ┌────────────────────┘               │
│                              ▼                                   │
│                    ┌──────────────────┐                           │
│                    │  ALERTS          │                           │
│                    │  - Drift > 2%    │                           │
│                    │  - MAPE spike    │                           │
│                    │  - Data quality  │                           │
│                    │  → Email / Slack │                           │
│                    └──────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2.8 Monitoring & Drift Detection

### What is Model Drift?

Model drift occurs when the statistical properties of the data or the relationship between features and target change over time, causing model performance to degrade.

**Types of drift:**

| Type | What Changes | Example in Airline Context |
|------|-------------|---------------------------|
| **Data drift** (covariate shift) | Input feature distributions | Competitor exits a route → `num_competitors` distribution shifts |
| **Concept drift** | Relationship between features and target | Post-COVID: price sensitivity increased → same features produce different demand |
| **Label drift** | Target variable distribution | Revenue targets changed after route restructuring |

### Population Stability Index (PSI)

PSI quantifies how much a feature's distribution has shifted between a reference period and the current period.

**Formula:**

$$
\text{PSI} = \sum_{i=1}^{n} (p_i - q_i) \times \ln\left(\frac{p_i}{q_i}\right)
$$

Where:
- $p_i$ = proportion of observations in bin $i$ for the **current** period
- $q_i$ = proportion of observations in bin $i$ for the **reference** period

**Interpretation:**

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant shift | Continue monitoring |
| 0.1 – 0.25 | Moderate shift | Investigate, may need retraining |
| > 0.25 | Significant shift | **Retrain model immediately** |

```python
import numpy as np

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    # Create bins from reference distribution
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    # Convert to proportions (avoid zeros)
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + n_bins)
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

def monitor_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                   features: list, threshold: float = 0.02) -> dict:
    """Monitor drift across all features. Alert if any exceeds threshold."""
    drift_report = {}
    for feature in features:
        psi = compute_psi(reference_data[feature].values, current_data[feature].values)
        drift_report[feature] = {
            'psi': psi,
            'status': 'ALERT' if psi > threshold else 'OK'
        }

    alerted = [f for f, v in drift_report.items() if v['status'] == 'ALERT']
    if alerted:
        send_alert(f"Drift detected in {len(alerted)} features: {alerted}")

    return drift_report
```

### MAPE Tracking Over Time

```python
def track_mape_over_time(actuals: pd.Series, predictions: pd.Series,
                          dates: pd.Series, window: int = 7) -> pd.DataFrame:
    """Track rolling MAPE to detect performance degradation."""
    df = pd.DataFrame({
        'actual': actuals, 'predicted': predictions, 'date': dates
    }).sort_values('date')

    df['abs_pct_error'] = np.abs((df['actual'] - df['predicted']) / df['actual']) * 100
    df['rolling_mape'] = df['abs_pct_error'].rolling(window).mean()

    # Alert if MAPE exceeds threshold
    latest_mape = df['rolling_mape'].iloc[-1]
    if latest_mape > 10:  # alert threshold
        send_alert(f"MAPE spike detected: {latest_mape:.1f}% (threshold: 10%)")

    return df
```

### How <2% Drift Was Sustained

1. **Daily PSI monitoring:** Computed PSI for top 15 features daily; any PSI > 0.1 triggered investigation
2. **Weekly MAPE check:** Rolling 7-day MAPE tracked; alerts if > 10%
3. **Monthly retraining:** Scheduled retraining with latest 18 months of data (expanding window)
4. **Quarterly model review:** Full model comparison (RF vs. XGB vs. LSTM) to ensure champion model still optimal
5. **Event-driven retraining:** Major events (new route launches, competitor changes) triggered ad-hoc retraining

The "< 2% drift" refers to keeping feature PSI values below 0.02 on average across monitored features, achieved through proactive retraining before drift accumulated.

---

# 3. Key Metrics & Results

---

## 3.1 MAPE — Mean Absolute Percentage Error

**Formula:**

$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

**Why MAPE:**
- **Scale-independent:** Can compare across routes with different booking volumes
- **Business-intuitive:** "Our forecast is off by X%" is understandable to operations team
- **Industry standard** for demand forecasting in aviation

**Limitation:** Undefined when $y_i = 0$; asymmetric (penalizes overestimates less than underestimates for small values). We mitigated this by filtering out near-zero actuals and also tracking symmetric MAPE (sMAPE).

## 3.2 Results Summary

| Metric | Before (Baseline) | After (LSTM) | Improvement |
|--------|-------------------|-------------|-------------|
| MAPE | ~18% | ~7% | ~61% reduction |
| MAE (bookings/day) | ~8.2 | ~3.1 | 62% reduction |
| RMSE | ~10.5 | ~4.2 | 60% reduction |
| Model drift (avg PSI) | N/A (no monitoring) | <0.02 (sustained) | — |
| Forecast horizon | 7 days | 30 days | 4× longer |
| Time to decision | 2-3 days (manual) | Same-day (automated) | 2-3× faster |

## 3.3 Business Impact

| Area | Impact |
|------|--------|
| **Pricing** | Proactive fare adjustments based on demand forecasts → improved yield management |
| **Crew allocation** | Better demand forecasts → optimized crew rostering, reduced overtime costs |
| **Capacity planning** | Route-level demand forecasts → informed aircraft allocation decisions |
| **Revenue management** | Rate-of-sale alerts → faster response to demand changes |
| **Market response** | Competitor activity features → quicker reaction to competitor pricing |

---

# 4. Topics You Must Know

---

## 4.1 Time Series Fundamentals

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time.

**Types:**
- **Strict stationarity:** Joint distribution of any collection of time indices is invariant to time shift
- **Weak (wide-sense) stationarity:** Mean is constant, autocovariance depends only on lag (not time)

**Why it matters:** Many statistical models (ARIMA) assume stationarity. Non-stationary data can produce spurious correlations.

**Tests:**
- **ADF (Augmented Dickey-Fuller):** Tests null hypothesis of unit root (non-stationarity). p < 0.05 → stationary.
- **KPSS:** Tests null hypothesis of stationarity. p < 0.05 → non-stationary.

**Making data stationary:** Differencing ($y'_t = y_t - y_{t-1}$), log transform, seasonal differencing.

### Seasonality, Trend, Decomposition

```
┌──────────────────────────────────────────────────────────────┐
│              TIME SERIES DECOMPOSITION                        │
│                                                               │
│  Observed(t) = Trend(t) + Seasonal(t) + Residual(t)          │
│                                  (additive)                   │
│  Observed(t) = Trend(t) × Seasonal(t) × Residual(t)          │
│                                  (multiplicative)             │
│                                                               │
│  ┌──────────────────────────────────────┐                     │
│  │ Observed:  ╱╲  ╱╲  ╱╲  ╱╲  trending │                     │
│  │           ╱  ╲╱  ╲╱  ╲╱  ╲  upward  │                     │
│  ├──────────────────────────────────────┤                     │
│  │ Trend:          ╱                    │                     │
│  │              ╱                       │  (long-term         │
│  │           ╱                          │   direction)        │
│  ├──────────────────────────────────────┤                     │
│  │ Seasonal: ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿       │  (repeating         │
│  │                                      │   pattern)          │
│  ├──────────────────────────────────────┤                     │
│  │ Residual:  · ·   · ·  · ·   · ·     │  (noise /           │
│  │             · ·  ·  · · ·  ·  ·      │   unexplained)      │
│  └──────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

**In airline context:**
- **Trend:** Long-term growth in air travel demand
- **Seasonality:** Summer peak, Christmas peak, winter lull — repeats annually
- **Weekly seasonality:** Friday/Sunday flights busier (weekend travel)
- **Residual:** Unpredictable events (weather disruptions, competitor actions)

### Autocorrelation

- **ACF (Autocorrelation Function):** Correlation between $y_t$ and $y_{t-k}$ at lag $k$. Includes indirect effects.
- **PACF (Partial ACF):** Direct correlation between $y_t$ and $y_{t-k}$, removing effects of intermediate lags.
- Used to determine ARIMA parameters: PACF cutoff → AR order (p); ACF cutoff → MA order (q).

---

## 4.2 LSTM/GRU Architecture Deep Dive

*(Covered in detail in Section 2.4.3 above)*

**Key points to articulate in interview:**
1. LSTM solves the **vanishing gradient problem** of vanilla RNNs via the cell state (highway of information)
2. The **forget gate** is the key innovation — it learns WHAT to forget (not just WHAT to remember)
3. GRU is a simplification: merges forget+input into **update gate**, removes separate cell state
4. For our use case, LSTM slightly outperformed GRU (complex, long-horizon dependencies), but GRU trained 25% faster

---

## 4.3 Evaluation Metrics — Formulas & When to Use

| Metric | Formula | When to Use | Pros | Cons |
|--------|---------|-------------|------|------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust metric, all values similar scale | Intuitive, robust to outliers | Not differentiable at 0, scale-dependent |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | When large errors are especially costly | Penalizes large errors more | Sensitive to outliers, same units as target |
| **MAPE** | $\frac{100}{n}\sum\left\|\frac{y_i-\hat{y}_i}{y_i}\right\|$ | Comparing across different scales | Scale-independent, business-intuitive | Undefined at $y=0$, asymmetric |
| **sMAPE** | $\frac{100}{n}\sum\frac{2\|y_i-\hat{y}_i\|}{\|y_i\|+\|\hat{y}_i\|}$ | When MAPE's asymmetry is a problem | Bounded [0,200], symmetric | Less intuitive |
| **R²** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Explained variance | Normalized, easy to compare | Can be negative, misleading for non-linear |

---

## 4.4 XGBoost Deep Dive

**Objective function:**

$$
\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

Where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ is the regularization term ($T$ = number of leaves, $w$ = leaf weights).

**Key hyperparameters:**

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `n_estimators` | 100–5000 | Number of boosting rounds |
| `max_depth` | 3–12 | Tree depth (controls model complexity) |
| `learning_rate` (η) | 0.01–0.3 | Shrinkage (lower = more trees needed but better generalization) |
| `subsample` | 0.6–1.0 | Row subsampling per tree |
| `colsample_bytree` | 0.6–1.0 | Feature subsampling per tree |
| `reg_alpha` (α) | 0–10 | L1 regularization on leaf weights |
| `reg_lambda` (λ) | 0–10 | L2 regularization on leaf weights |
| `min_child_weight` | 1–20 | Minimum sum of instance weight in a child |

---

## 4.5 Random Forest Deep Dive

**Bootstrap Aggregating (Bagging):**
1. Draw N bootstrap samples (with replacement) from training data
2. Train a decision tree on each sample
3. At each split, randomly select $m$ features from $p$ total ($m \approx \sqrt{p}$ for classification, $m \approx p/3$ for regression)
4. Average predictions (regression) or majority vote (classification)

**Out-of-Bag (OOB) Error:**
- Each tree doesn't see ~37% of training data (the ones not sampled)
- Use these unseen samples as a validation set — no separate hold-out needed
- OOB error ≈ test error (proven to be an unbiased estimate)

**Feature Importance Methods:**
- **MDI (Mean Decrease in Impurity):** Biased toward high-cardinality features
- **Permutation importance:** Unbiased — shuffle feature, measure performance drop

---

## 4.6 Optuna Deep Dive

*(Covered in detail in Section 2.6 above)*

**Key talking points:**
1. TPE is more sample-efficient than random search (finds better configs in fewer trials)
2. Pruning saves 40-60% of compute by killing bad trials early
3. The `log=True` parameter for learning rate samples uniformly in log-space (essential for parameters spanning orders of magnitude)
4. Optuna's `study.optimize()` is thread-safe and supports parallel/distributed tuning

---

## 4.7 Snowflake Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 SNOWFLAKE ARCHITECTURE                        │
│                                                               │
│  ┌─────────────────────────────────────────────┐              │
│  │           CLOUD SERVICES LAYER               │              │
│  │  (Authentication, Query Optimization,        │              │
│  │   Metadata, Access Control)                  │              │
│  └─────────────────────────────────────────────┘              │
│                          │                                    │
│  ┌─────────────────────────────────────────────┐              │
│  │         COMPUTE LAYER (Virtual Warehouses)   │              │
│  │                                              │              │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │              │
│  │  │ WH: ETL  │  │ WH: ML   │  │WH:Tableau│   │              │
│  │  │ (XS)     │  │ (Medium) │  │ (Small)  │   │              │
│  │  └──────────┘  └──────────┘  └──────────┘   │              │
│  │  Independent scaling — no resource contention│              │
│  └─────────────────────────────────────────────┘              │
│                          │                                    │
│  ┌─────────────────────────────────────────────┐              │
│  │             STORAGE LAYER                    │              │
│  │  (Centralized, compressed columnar storage)  │              │
│  │  All warehouses read from same storage       │              │
│  └─────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

**Key concepts:**
- **Separation of compute and storage:** Scale independently; pay only for what you use
- **Virtual warehouses:** Isolated compute clusters — ETL doesn't slow down Tableau queries
- **Zero-copy cloning:** Create instant copies of tables for testing without duplicating storage
- **Time travel:** Query data as it existed at any point in the last 90 days
- **Stages:** Named locations (internal or external S3/GCS/Azure) for loading data

---

## 4.8 Dataiku as MLOps Platform

**Key capabilities used:**
- **Visual flow:** Drag-and-drop pipeline design with code nodes (Python recipes)
- **Scenarios:** Automated triggers (time-based, data arrival, API call)
- **Model evaluation store:** Track performance over time, compare model versions
- **API node:** Serve model predictions via REST API if needed
- **Webapp:** Build dashboards for stakeholders directly in Dataiku

---

## 4.9 Data Drift Monitoring

*(Covered in detail in Section 2.8 above)*

**Additional statistical tests for drift:**
- **Kolmogorov-Smirnov (KS) test:** Non-parametric test comparing two distributions
- **Chi-squared test:** For categorical features
- **Jensen-Shannon divergence:** Symmetric version of KL divergence
- **Page-Hinkley test:** Sequential drift detection for streaming data

---

## 4.10 Pydantic for Data Validation

*(Covered in detail in Section 2.1 above)*

**Key interview points:**
- Pydantic v2 (built on Rust) is 5-50× faster than v1
- `BaseModel` for strict validation; `@validator` for custom business rules
- Automatic JSON Schema generation for API documentation
- Use `Field(...)` for constraints: `ge`, `le`, `min_length`, `regex`

---

## 4.11 Airline Revenue Management Basics

**Key concepts:**
- **Yield management:** Dynamically adjusting prices based on demand, time-to-departure, competitor prices
- **Load factor:** Percentage of available seats filled: $\text{LF} = \frac{\text{Passengers}}{\text{Capacity}}$
- **Revenue per Available Seat Kilometer (RASK):** $\frac{\text{Revenue}}{\text{Seats} \times \text{Distance}}$ — key airline KPI
- **Fare classes:** Economy, premium, business — different price points for same seat depending on booking conditions (flexibility, advance purchase)
- **Demand curve:** Typically: slow build → acceleration → last-minute surge (or drop for leisure routes)
- **Overbooking:** Airlines intentionally sell more tickets than seats, using no-show prediction models

---

## 4.12 Rate-of-Sale in Travel Industry

*(Covered in detail in Section 2.2.1 above)*

**Quick elevator pitch:** "Rate-of-sale tells us how quickly we're selling seats. It's the velocity of demand. If ROS is above historical average for this time before departure, demand is strong — we can raise prices. If it's below, we need to act — either promotional pricing or reallocate capacity."

---

# 5. Interview Questions & Model Answers

---

### Q1: "Walk me through this project end-to-end."

**Answer (2-3 minutes):**

> "At Jet2, the operations and finance teams were using multiple disconnected data streams — flight schedules, booking systems, revenue databases, competitor pricing feeds, and regional performance targets. The data arrived in complex nested JSON formats with no unified schema, making analysis slow and inconsistent.
>
> My task was two-fold: first, unify these datasets into a structured data warehouse; second, build predictive models to forecast operational risk scores — specifically demand, revenue shortfall, and route underperformance.
>
> For data integration, I designed Python ETL scripts that flattened nested JSON structures and loaded them into a Snowflake star schema. I used Pydantic for schema validation to catch data quality issues before they entered the warehouse.
>
> For feature engineering, I created domain-specific features like rate-of-sale metrics — which measure how quickly seats are selling relative to historical benchmarks — along with holiday flags, competitor pricing differentials, and temporal features like lagged bookings and rolling averages.
>
> I trained and compared three model families: Random Forest as a robust baseline, XGBoost for its gradient boosting strength, and LSTM neural networks for capturing temporal patterns. I used walk-forward validation instead of k-fold to prevent data leakage, and Optuna for Bayesian hyperparameter tuning.
>
> The LSTM model won — it reduced MAPE from about 18% to about 7% by capturing complex temporal dependencies that the tree-based models missed. I deployed it in Dataiku with daily automated runs, integrated Tableau for stakeholder dashboards, and set up continuous monitoring. We sustained less than 2% model drift over 18 months through proactive retraining triggers."

---

### Q2: "Why walk-forward validation over k-fold for time series?"

**Answer:**

> "Standard k-fold cross-validation randomly shuffles data, which means future data points can end up in the training set while past data points are in the validation set. For time series, this creates data leakage — the model essentially gets to peek at the future.
>
> Walk-forward validation respects temporal order. In each fold, the training set contains only past data, and the validation set is always a future time window. This gives us a realistic estimate of how the model will perform in production, where it always predicts the future from the past.
>
> We used the expanding window variant, where each subsequent fold has a larger training set. This was appropriate because airline demand patterns are relatively stable, so older data was still informative. For a rapidly changing environment, a sliding window approach might be better."

---

### Q3: "LSTM vs. XGBoost — why did LSTM win for this forecasting task?"

**Answer:**

> "Both are strong models, but they approach the data fundamentally differently. XGBoost sees each sample independently — it doesn't inherently understand that row 100 comes after row 99. We can engineer lagged features to give it temporal context, but it's still making independent predictions.
>
> LSTM processes the entire sequence at once. Its gates — forget, input, output — learn which patterns to remember and which to discard across the sequence. For airline demand, this is crucial because there are complex, overlapping temporal patterns: weekly cycles, seasonal trends, and multi-week booking curves.
>
> Specifically, LSTM captured the booking velocity pattern — how the rate of bookings accelerates and decelerates in the weeks before departure — more naturally than XGBoost with hand-crafted lag features. The LSTM reduced MAPE from XGBoost's 9.1% to 7.2%.
>
> That said, XGBoost remained our fallback. It's faster to train, more interpretable via SHAP values, and performs well for shorter horizons. For production, we used LSTM for the primary 30-day forecast and XGBoost for quick 7-day recalibrations."

---

### Q4: "How did you handle seasonality?"

**Answer:**

> "Airline demand is deeply seasonal, so we addressed it at multiple levels.
>
> First, in feature engineering: I created cyclical encodings using sine and cosine transforms for month and day-of-week, so the model understands that December and January are adjacent months. We also had explicit binary flags for UK school holidays, bank holidays, and known special events.
>
> Second, year-over-year features: rate-of-sale ratios comparing current demand to the same period last year provided seasonal context.
>
> Third, the LSTM architecture itself: with a 28-day sequence length, the model had enough context to learn weekly patterns. For annual seasonality, the historical features and calendar encodings bridged the gap.
>
> Fourth, in validation: walk-forward validation ensured each fold tested the model's ability to predict a future period it hadn't seen, including seasonal transitions."

---

### Q5: "What is rate-of-sale and why does it matter?"

**Answer:**

> "Rate-of-sale measures how quickly seats are being booked over a given time window. For example, a 7-day ROS of 15 means we're selling 15 seats per day on average over the last week.
>
> It matters because it's the primary signal for pricing decisions. If ROS is above the historical benchmark for this many days before departure, demand is strong and we can hold or raise prices. If it's below, we might need promotional fares to fill the aircraft.
>
> We computed ROS at multiple windows — 3-day, 7-day, 14-day, 28-day — to capture demand at different time scales. We also derived ROS acceleration (is demand speeding up or slowing down?) and year-over-year ROS ratios for seasonal context. These rate-of-sale features were consistently among the top 5 most important features across all our models."

---

### Q6: "How did you deploy the model?"

**Answer:**

> "We deployed in Dataiku, which is an enterprise MLOps platform. The pipeline ran as a daily scheduled scenario triggered at 6 AM UTC:
>
> Step 1: Pull fresh data from Snowflake. Step 2: Run feature engineering via a Python recipe. Step 3: Run inference through the trained LSTM model. Step 4: Post-process and validate outputs with Pydantic. Step 5: Run a drift check — compute PSI on key features and compare prediction distributions. Step 6: Write forecasts back to Snowflake and trigger Tableau dashboard refresh.
>
> If drift exceeded the 2% threshold, the system sent alerts via email and Slack. Monthly retraining was automated, and the champion model was automatically compared against the previous version on a holdout window before promotion.
>
> We chose Dataiku over a custom deployment because the operations team — non-engineers — needed to view, understand, and sometimes trigger the pipeline. Dataiku's visual interface made this accessible without building a custom internal tool."

---

### Q7: "How did you monitor model drift?"

**Answer:**

> "We implemented a three-tier monitoring system.
>
> First, **feature drift** via Population Stability Index (PSI). We computed PSI daily for the top 15 features, comparing the last 30 days against the training reference distribution. PSI below 0.1 was green, 0.1-0.25 was yellow (investigate), above 0.25 was red (retrain). We targeted keeping average PSI below 0.02.
>
> Second, **performance monitoring** via rolling MAPE. We tracked 7-day and 30-day rolling MAPE on actuals vs. predictions. Any spike above 10% triggered an alert.
>
> Third, **prediction distribution monitoring**. We checked whether the distribution of model outputs was shifting — for example, if the model started consistently predicting higher demand than actuals, that's a systematic bias developing.
>
> We sustained less than 2% drift over 18 months through monthly retraining with an expanding window and event-driven retraining for major market changes."

---

### Q8: "Explain Optuna hyperparameter tuning — how does TPE work?"

**Answer:**

> "Optuna uses a Bayesian optimization algorithm called TPE — Tree-structured Parzen Estimator. Unlike grid search, which exhaustively evaluates all combinations, or random search, which samples blindly, TPE builds a probabilistic model of which hyperparameter regions produce good results.
>
> Specifically, TPE maintains two density functions: l(x), the distribution of hyperparameters from trials that performed well, and g(x), from trials that performed poorly. It then samples the next set of hyperparameters to maximize the ratio l(x)/g(x) — meaning it focuses on regions where good results are likely and bad results are unlikely.
>
> This makes it much more sample-efficient. We found near-optimal LSTM hyperparameters in about 60 trials, where grid search over the same space would have required thousands.
>
> We also used Optuna's MedianPruner, which stops unpromising trials early. If a trial's intermediate loss after 10 epochs is worse than the median of all completed trials at that epoch, it's killed. This saved us roughly 50% of compute time."

---

### Q9: "How did you handle missing data?"

**Answer:**

> "Missing data was common, especially in competitor feeds and early-stage booking data. We used a multi-strategy approach:
>
> For temporal features (lagged bookings, rolling averages), missing values at series boundaries were handled with `min_periods=1` in rolling calculations, so the model used whatever data was available rather than dropping rows.
>
> For competitor data, missing prices were imputed using the last known value (forward-fill) since competitor prices don't change hourly. If a competitor had no data for a route at all, we imputed with the route's average competitor price.
>
> For the LSTM specifically, we masked missing values and used the model's ability to handle variable-length sequences. For XGBoost, we leveraged its native missing value handling — it learns the optimal split direction for missing values during training.
>
> Pydantic validation caught truly invalid data — nulls in required fields, impossible values — at the pipeline boundary, so the model never saw garbage data."

---

### Q10: "What would you do differently if you started this project today?"

**Answer:**

> "Three things. First, I'd explore Temporal Fusion Transformers (TFT) — a newer architecture specifically designed for multi-horizon time series forecasting. It combines attention mechanisms with LSTM-like recurrence and has built-in interpretability through variable importance and temporal attention weights.
>
> Second, I'd implement a more sophisticated feature store, possibly using a tool like Feast. Our feature engineering code was tightly coupled to the pipeline, making it harder to reuse features across different models and teams.
>
> Third, I'd set up A/B testing for model deployment rather than a simple champion/challenger swap. This would let us measure the actual business impact of model improvements — like whether a 1% MAPE reduction actually translates to better pricing decisions — rather than just evaluating offline metrics."

---

### Q11: "Explain the JSON flattening you did. Why was it necessary?"

**Answer:**

> "The raw data arrived as deeply nested JSON — a booking record contained nested passenger arrays, nested fare breakdowns, nested route objects. You can't directly load this into a relational table because each record has variable structure and depth.
>
> I wrote recursive flattening functions that traversed the JSON tree and produced flat key-value pairs with composite keys. For example, `bookings.0.passengers.0.fare_class` became the column `bookings_0_passengers_0_fare_class`. For array elements, we also extracted aggregate statistics — total passengers, sum of fares — rather than keeping every nested element.
>
> The flattened data was then loaded into Snowflake's star schema with proper dimension tables. This normalized structure made it queryable by SQL and consumable by Tableau — the operations team went from 'we can't use this data' to running their own ad-hoc analyses."

---

### Q12: "How does Random Forest differ from XGBoost fundamentally?"

**Answer:**

> "The core difference is bagging versus boosting.
>
> Random Forest uses bagging — it trains many independent decision trees on bootstrap samples of the data, each seeing a random subset of features at each split, then averages their predictions. This primarily reduces **variance** while keeping bias similar to a single tree.
>
> XGBoost uses gradient boosting — it trains trees **sequentially**, where each new tree corrects the errors (residuals) of the ensemble so far. This primarily reduces **bias** by iteratively improving the model's fit.
>
> In practice, XGBoost typically achieves lower error on structured/tabular data because of this iterative error correction. But it's more prone to overfitting and more sensitive to hyperparameters. Random Forest is more robust out-of-the-box.
>
> For our project, XGBoost outperformed Random Forest (9.1% vs. 11.2% MAPE), but both were surpassed by the LSTM due to the temporal nature of the data."

---

### Q13: "What is the Snowflake schema design and why did you choose star schema?"

**Answer:**

> "We designed a classic star schema with a central fact table — FACT_BOOKINGS — surrounded by dimension tables: DIM_ROUTE, DIM_DATE, DIM_FLIGHT, DIM_COMPETITOR. The fact table contained measurable events — booking counts, revenue, load factors, risk scores — while dimension tables held descriptive attributes.
>
> Star schema was chosen for three reasons: First, it's optimized for analytical queries — Snowflake's columnar storage performs best with denormalized dimensions that minimize JOINs. Second, it maps naturally to Tableau's data model, making dashboard creation straightforward. Third, it's easy for non-technical users to understand — they can query 'bookings by route by month' without navigating complex normalized relationships."

---

### Q14: "What is Population Stability Index (PSI) and how did you use it?"

**Answer:**

> "PSI quantifies how much a feature's distribution has shifted between two time periods. It divides both distributions into bins, computes the proportion difference in each bin, and weights it by the log-ratio. Mathematically: PSI = sum of (p_i - q_i) × ln(p_i/q_i).
>
> Values below 0.1 indicate no significant shift. Between 0.1 and 0.25 suggests moderate shift — worth investigating. Above 0.25 means significant shift — model should be retrained.
>
> We computed PSI daily for our top 15 features, comparing the last 30 days against the training period distribution. This caught drift early — for instance, when a competitor exited a route, the `num_competitors` feature shifted significantly, and the PSI alert triggered before the model's MAPE degraded."

---

### Q15: "Why Pydantic and not just pandas for validation?"

**Answer:**

> "Pandas can do type checking and basic validation, but it operates at the DataFrame level after data is already loaded. Pydantic validates at the record level before data enters the pipeline, acting as a strict contract.
>
> With Pydantic, I defined exactly what each record should look like: field types, ranges, constraints, and custom business rules. For example, departure date must be after booking date, revenue per passenger must be under £5,000 (sanity check), airport codes must be exactly 3 characters.
>
> When a record violated any constraint, it was logged with a detailed error message and excluded from processing. This prevented garbage data from corrupting our warehouse and gave us a clear audit trail of data quality issues. We could then work with data providers to fix systematic problems at the source."

---

### Q16: "How do you choose between MAPE, MAE, and RMSE?"

**Answer:**

> "It depends on what matters to the business.
>
> MAPE is percentage-based, so it's scale-independent — useful when you're comparing forecast accuracy across routes with very different booking volumes. It's also business-intuitive: 'our forecast is off by 7%' is immediately understandable. We used MAPE as our primary metric for this reason.
>
> MAE treats all errors equally and is in the same units as the target, making it concrete: 'we're off by 3.1 bookings per day.' It's robust to outliers because it doesn't square errors.
>
> RMSE penalizes large errors more due to squaring. It's the right choice when large errors are especially costly. For airline operations, a 50-booking forecasting error is much more than 5× worse than a 10-booking error — it can mean crew shortages — so RMSE is also relevant."

---

### Q17: "What's the difference between data drift and concept drift?"

**Answer:**

> "Data drift — also called covariate shift — is when the input feature distributions change. For example, if competitor prices across all routes suddenly drop 20%, the distribution of our `avg_comp_price` feature shifts. The model might still make reasonable predictions if the learned relationship is robust.
>
> Concept drift is when the relationship between features and the target changes. For example, during COVID, the same price and same historical demand pattern produced very different actual bookings — customer behavior fundamentally changed. The model's learned patterns no longer applied.
>
> Concept drift is more dangerous because you can't detect it from features alone — you need to monitor actual prediction performance against ground truth. We monitored both: PSI for data drift (features) and rolling MAPE for concept drift (performance degradation)."

---

### Q18: "Explain gradient clipping — why did you use it for the LSTM?"

**Answer:**

> "Gradient clipping caps the gradient norm during backpropagation to prevent exploding gradients — a common problem in RNNs, including LSTMs. When the gradient norm exceeds a threshold, it's scaled down proportionally.
>
> We set `max_norm=1.0`, meaning if the total gradient norm exceeded 1.0, all gradients were scaled by `1.0 / actual_norm`. This prevents any single batch from making a catastrophically large parameter update that destroys what the model has learned.
>
> While LSTMs were specifically designed to mitigate vanishing gradients via the cell state highway, exploding gradients can still occur, especially with long sequences and deep architectures. Clipping is a simple, effective safeguard."

---

### Q19: "Tell me about a challenge you faced and how you overcame it."

**Answer (Behavioral — STAR):**

> "One significant challenge was **data quality inconsistency** from the competitor pricing feed.
>
> **Situation:** About 3 months into production, our model's MAPE started creeping up on certain routes. The PSI alerts flagged significant drift in competitor features.
>
> **Task:** Diagnose the root cause and fix it without disrupting the daily forecasting pipeline.
>
> **Action:** I traced the issue to a third-party competitor pricing API that had silently changed its response format — prices that were previously in GBP were now being returned in EUR for certain routes. Our Pydantic validation wasn't catching this because the values were still valid floats, just in the wrong currency.
>
> I added a currency validation step comparing competitor prices against historical ranges per route, flagged anomalous conversions, and added a currency normalization layer to the pipeline. I also implemented a more robust data contract with the API provider.
>
> **Result:** MAPE returned to baseline within 48 hours of the fix, and the new validation prevented similar issues from occurring again. This experience reinforced the importance of not just validating data types, but validating data semantics."

---

### Q20: "How did you communicate results to non-technical stakeholders?"

**Answer:**

> "Through Tableau dashboards designed for three audiences.
>
> For the **operations team**: a route-level dashboard showing today's demand forecast, risk score (color-coded red/amber/green), and rate-of-sale trend. They could see at a glance which routes needed attention.
>
> For **finance/revenue management**: a portfolio-level view showing forecast revenue by region, variance from targets, and competitor positioning. This fed directly into their weekly pricing review.
>
> For **senior leadership**: a monthly executive summary — aggregate accuracy metrics, business impact highlights (revenue lift from proactive pricing decisions), and a model health indicator.
>
> I also ran a quarterly 'model explainability' session where I used SHAP waterfall plots to show why specific forecasts were high or low. This built trust — stakeholders understood the model's reasoning, not just its output."

---

### Q21: "If the LSTM starts performing poorly, what's your debugging process?"

**Answer:**

> "I follow a systematic checklist:
>
> **1. Data quality:** Has the input data changed? Check PSI for feature drift, look for missing data spikes, schema changes, or upstream ETL failures.
>
> **2. Concept drift:** Compare prediction distributions against actuals over time. If the model is systematically over- or under-predicting, the underlying relationship may have shifted.
>
> **3. Feature importance shift:** Run SHAP on recent predictions and compare to training-time feature importance. If a previously unimportant feature is now dominating, something changed.
>
> **4. Temporal patterns:** Check if degradation correlates with specific time periods — maybe a new seasonal pattern emerged (like a new school holiday schedule).
>
> **5. Retrain and compare:** Retrain on the latest data and compare against the current model on a hold-out window. If retraining fixes it, it was data/concept drift. If not, the architecture may need revision.
>
> **6. Architecture check:** As a last resort, re-run the model comparison (RF vs. XGB vs. LSTM). In some cases, a simpler model might outperform if the data regime has fundamentally changed."

---

### Q22: "What is walk-forward validation's relationship to production performance?"

**Answer:**

> "Walk-forward validation is the closest simulation of production conditions we can do offline. Each fold replicates the real-world scenario: the model is trained on all available past data and tested on a future window it hasn't seen.
>
> The average MAPE across walk-forward folds was within 0.5% of our actual production MAPE over the first 6 months, validating that our offline evaluation accurately predicted production performance. This is much better than k-fold, which typically overestimates performance by 3-8% for time series due to data leakage.
>
> However, walk-forward still can't capture truly novel events — like a new competitor entering a market or a pandemic. That's why continuous monitoring is essential alongside strong offline validation."

---

### Q23: "Explain how you'd implement the LSTM inference pipeline in production."

**Answer:**

> "The inference pipeline runs daily in Dataiku:
>
> 1. **Data pull:** Query Snowflake for the latest 30 days of data per route (our sequence length is 28, plus buffer).
>
> 2. **Feature engineering:** Apply the same transformation pipeline used during training — rate-of-sale computation, holiday flags, competitor features, lag calculations. Critically, this is the same code module, not a reimplementation, to avoid training-serving skew.
>
> 3. **Scaling:** Apply the saved StandardScaler from training. Never refit the scaler on inference data.
>
> 4. **Sequence creation:** Shape the data into (routes, 28 timesteps, n_features) tensors.
>
> 5. **Model inference:** Load the saved PyTorch model, set to `model.eval()`, disable dropout, run forward pass with `torch.no_grad()` for efficiency.
>
> 6. **Inverse scaling:** Transform predictions back to original scale.
>
> 7. **Validation:** Pydantic check on outputs — predictions within realistic bounds.
>
> 8. **Write results:** Push to Snowflake, update Tableau."

---

### Q24: "How did you validate that the model's business impact was real?"

**Answer:**

> "We validated business impact through a controlled comparison period.
>
> During the first two months of deployment, we ran the model alongside the existing manual process. The operations team received model forecasts but continued making their own assessments. We then compared decisions and outcomes:
>
> - Routes where the model's forecast would have triggered a price adjustment earlier than the manual process: we measured what the revenue impact would have been.
>
> - Routes where the model flagged risk (low demand) 2 weeks before the operations team noticed: these represented crew allocation savings.
>
> This wasn't a formal A/B test (we couldn't randomly assign routes), but the before/after comparison with the same routes across seasons gave us confidence that the MAPE improvement translated to real operational gains."

---

### Q25: "What is the difference between expanding window and sliding window walk-forward?"

**Answer:**

> "Expanding window starts the training set at a fixed point and grows it with each fold — you're always using ALL available historical data. Sliding window keeps a fixed-size training window that moves forward.
>
> We chose expanding window because airline data accumulates slowly — each route generates one data point per day — and historical patterns (seasonality, booking curves) remained relevant over our 2-year dataset. Discarding old data would have meant less seasonal coverage.
>
> Sliding window is better when there's concept drift — when older data is actively misleading. For example, if we were forecasting post-COVID demand, pre-COVID data might hurt rather than help, so a 12-month sliding window would be more appropriate."

---

# 6. Red Flags & How to Handle

---

## 6.1 "What was the exact MAPE?"

**Red flag:** Interviewers may push for exact numbers to test if you actually ran the project.

**How to handle:** Give a confident range, not a suspiciously precise number.

> "Before the project, the teams were working with rule-based estimates that were roughly 15-20% off on average — so around 18% MAPE baseline. Our final LSTM model brought that down to the 6-8% range, depending on the route and season. Summer routes with stronger historical patterns performed better — around 5-6%. Winter routes with more volatility were closer to 8-10%. The overall portfolio average settled at about 7%."

**Why this works:** Shows you understand that metrics vary by segment, season, and route — not a single magical number.

---

## 6.2 "Why LSTM and not ARIMA or Prophet?"

**Red flag:** Using deep learning for time series when simpler methods might suffice.

**How to handle:**

> "We did evaluate ARIMA and Prophet early on. ARIMA required stationarity transformations and is fundamentally univariate — we'd need separate models per route and couldn't easily incorporate the 30+ exogenous features we engineered. Prophet handles seasonality well for a single time series, but we had ~200 routes, each needing forecasts with shared features like competitor pricing that affected multiple routes simultaneously.
>
> LSTM let us build a single multi-variate model that learned cross-feature temporal interactions. The engineering cost was higher, but the performance gain from 11% to 7% MAPE justified it, especially given the business impact of each percentage point.
>
> If I were working with a single time series and limited features, Prophet would absolutely be my first choice."

---

## 6.3 "Why Dataiku and not custom deployment (Docker/K8s/Airflow)?"

**Red flag:** Interviewer may perceive Dataiku as a no-code shortcut.

**How to handle:**

> "Dataiku was chosen for the organizational context, not the technical requirements. Jet2's data science team was small, and the primary users — operations and finance — needed to interact with the pipeline. Dataiku's visual interface meant they could view pipeline status, inspect intermediate data, and understand the workflow without engineering support.
>
> For the ML components themselves, I wrote custom Python code within Dataiku's Python recipe nodes — the modeling, feature engineering, and monitoring logic was all code-based. Dataiku handled the orchestration, scheduling, and UI layer.
>
> For a larger engineering team with dedicated MLOps support, I'd likely use Airflow for orchestration and a custom serving stack. The right tool depends on the team and context."

---

## 6.4 "How do you know the model wasn't just memorizing patterns?"

**How to handle:**

> "Three safeguards: First, walk-forward validation — the model never saw future data during training, so good performance on the validation folds means genuine generalization, not memorization. Second, dropout regularization in the LSTM (0.25) and weight decay in the optimizer — these actively prevent overfitting. Third, we monitored the gap between training and validation MAPE — it stayed within 1-2%, indicating no significant overfitting.
>
> Additionally, the model maintained its performance in production over 18 months across multiple seasonal cycles it hadn't been trained on. That's the strongest evidence of generalization."

---

## 6.5 "2% drift seems too good — how is that possible?"

**How to handle:**

> "To clarify: the < 2% refers to the average Population Stability Index across our monitored features, not that drift was only 2% of something. A PSI of 0.02 indicates very mild distributional shift.
>
> We achieved this through proactive retraining — monthly scheduled retraining with an expanding window, plus event-driven retraining when the market changed. So drift never had time to accumulate significantly. Think of it like maintaining a car: regular oil changes prevent engine failure, and regular retraining prevents drift accumulation.
>
> There were individual features that occasionally spiked above 0.1 PSI — competitor pricing after a fare war, for example — but those triggered immediate investigation and retraining, keeping the average low."

---

# 7. Key Takeaways

---

## For Quick Review Before Interview

1. **Architecture story:** JSON feeds → Pydantic validation → Snowflake star schema → Python feature engineering → LSTM model (Optuna-tuned) → Dataiku deployment → Tableau dashboards

2. **Why LSTM won:** Captures temporal dependencies and multi-variate interactions that tree-based models with hand-crafted lag features couldn't match

3. **Walk-forward validation:** Gold standard for time series — respects temporal order, prevents data leakage, gives realistic production estimates

4. **Rate-of-sale:** The velocity of seat sales — single most important feature for airline pricing decisions

5. **Optuna TPE:** Bayesian optimization that's more sample-efficient than grid/random search; pruning saves ~50% compute

6. **<2% drift:** Sustained via proactive monthly retraining + event-driven retraining; monitored via PSI and rolling MAPE

7. **Business impact:** MAPE reduced from ~18% to ~7%, enabling proactive pricing, optimized crew allocation, faster market response

8. **Key numbers to remember:**
   - MAPE: ~18% → ~7% (portfolio average)
   - Model drift: <2% (0.02 PSI average)
   - Forecast horizon: 30 days
   - Sequence length: 28 days
   - Features: ~35 engineered features
   - Models compared: 4 (RF, XGB, LSTM, GRU)
   - Duration: 18+ months in production

## One-Sentence Project Pitch

> "I built an end-to-end demand forecasting system at Jet2 that unified five disconnected data streams into a Snowflake warehouse, trained an LSTM model with walk-forward validation and Optuna tuning that cut MAPE from 18% to 7%, and deployed it in Dataiku with automated monitoring that sustained less than 2% model drift over 18 months — enabling proactive pricing and crew allocation decisions."

---

*Last updated: February 2026 | Prepared for DS/MLE interview preparation*
