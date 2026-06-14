# P06 — Address Dataset Engineering & Customer Default Prediction | Axio (CapitalFloat)

**Candidate:** Rahul Sharma | **Role:** Research Intern | **Duration:** May 2020 – November 2020  
**Company:** Axio (formerly CapitalFloat), Bangalore, India | **Industry:** Fintech / Digital Lending  
**Resume Line:** *"Built address dataset, applied clustering & statistical evaluation, achieving 95% F1, and informed fraud-risk strategy."*  
**Additional:** *"Designed OCF metrics and developed Power BI storytelling dashboards, translating analytics to non-tech executives."*  
**Document Scope:** Complete interview preparation — architecture, math, code, behavioral answers, and red-flag handling

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview-star)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
   - 2.1 [Raw JSON Data Processing](#21-raw-json-data-processing)
   - 2.2 [Address Parsing with NLP](#22-address-parsing-with-nlp)
   - 2.3 [Lifestyle Feature Engineering](#23-lifestyle-feature-engineering)
   - 2.4 [Feature Selection Pipeline](#24-feature-selection-pipeline)
   - 2.5 [Customer Segmentation with Clustering](#25-customer-segmentation-with-clustering)
   - 2.6 [Statistical Evaluation & Hypothesis Testing](#26-statistical-evaluation--hypothesis-testing)
   - 2.7 [Default Prediction Model (95% F1)](#27-default-prediction-model-95-f1)
   - 2.8 [OCF Metrics Design](#28-ocf-metrics-design)
   - 2.9 [Power BI Storytelling Dashboards](#29-power-bi-storytelling-dashboards)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know](#4-topics-you-must-know)
5. [Interview Questions & Model Answers (20+)](#5-interview-questions--model-answers)
6. [Red Flags & How to Handle](#6-red-flags--how-to-handle)
7. [Key Takeaways](#7-key-takeaways)

---

# 1. Project Overview (STAR)

---

## 1.1 STAR Summary

| STAR | Detail |
|------|--------|
| **Situation** | Axio (formerly CapitalFloat), a leading Indian digital lending platform, needed to improve its customer default prediction and fraud-risk assessment capabilities. Customer data—especially addresses—was stored as messy, unstructured raw JSON blobs from KYC (Know Your Customer) onboarding. Address information was inconsistent (multiple formats, abbreviations, typos, missing pin codes), and there was no systematic way to leverage customer lifestyle and behavioral data for credit risk analysis. Existing risk models lacked the granularity of address-level signals and lifestyle-based features. |
| **Task** | As a Research Intern, I was tasked with: (1) building a clean, structured address dataset from raw JSON files using NLP techniques; (2) engineering lifestyle features from customer spending habits, purchase types, and behavioral signals; (3) applying statistical analysis and feature selection to identify the strongest predictors of default; (4) training a classification model to predict customer default; and (5) designing OCF metrics and Power BI dashboards for executive reporting. |
| **Action** | (1) Parsed thousands of raw JSON KYC files, extracting and flattening nested address fields. (2) Applied NLP techniques—regex pattern matching, rule-based Named Entity Recognition, address standardization—to decompose unstructured addresses into structured components (street, area, city, state, pin code). (3) Engineered lifestyle features from transaction data—spending categories, purchase frequency, merchant-type distributions, behavioral anomaly indicators. (4) Ran rigorous feature selection using correlation analysis, chi-squared tests, mutual information scores, and LASSO regularization. (5) Applied K-Means clustering for customer segmentation and profiled clusters for risk patterns. (6) Performed hypothesis testing and distribution analysis for statistical evaluation. (7) Trained and tuned a classification model achieving 95% F1 score. (8) Designed OCF (Operating Cash Flow) metrics and built Power BI storytelling dashboards for non-technical executives. |
| **Result** | 95% F1 score on default prediction—strong performance in an imbalanced fraud/default context. Address-derived and lifestyle features significantly boosted model performance over baseline bureau-only models. Clustering revealed distinct risk segments that directly informed Axio's fraud-risk strategy. OCF metrics dashboards enabled non-technical executives to understand lending portfolio health. Created structured executive summaries supporting funding and strategic planning discussions. |

---

## 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  AXIO DEFAULT PREDICTION PIPELINE                                │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                         DATA SOURCES                                    │     │
│  │                                                                         │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │     │
│  │  │  KYC JSON    │  │  Transaction │  │  Bureau      │  │ Merchant  │  │     │
│  │  │  Files       │  │  Records     │  │  Data        │  │ Category  │  │     │
│  │  │  (Address,   │  │  (Spending,  │  │  (Credit     │  │ Data      │  │     │
│  │  │   Identity)  │  │   Purchases) │  │   History)   │  │           │  │     │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │     │
│  └─────────┼────────────────┼────────────────┼──────────────────┼────────┘     │
│            ▼                ▼                ▼                  ▼               │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                    JSON PROCESSING & NLP ENGINE                          │     │
│  │  • JSON flattening & schema extraction                                  │     │
│  │  • Address parsing: regex + NER + standardization                       │     │
│  │  • Pin code validation & geocoding enrichment                           │     │
│  │  • Error correction (typos, abbreviation expansion)                     │     │
│  └────────────────────────────┬────────────────────────────────────────────┘     │
│                               ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                   FEATURE ENGINEERING LAYER                              │     │
│  │                                                                         │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │     │
│  │  │  ADDRESS FEATURES│  │ LIFESTYLE FEATURES│  │  BUREAU FEATURES    │  │     │
│  │  │  • Metro/rural   │  │ • Spending ratios │  │  • Credit score     │  │     │
│  │  │  • Region tier   │  │ • Purchase types  │  │  • DPD history      │  │     │
│  │  │  • Address compl.│  │ • Txn frequency   │  │  • Utilization      │  │     │
│  │  │  • Pin code risk │  │ • Behavioral flags│  │  • Enquiry count    │  │     │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │     │
│  └────────────────────────────┬────────────────────────────────────────────┘     │
│                               ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                   FEATURE SELECTION PIPELINE                             │     │
│  │  • Correlation analysis (remove |r| > 0.7)                              │     │
│  │  • Chi-squared test (categorical significance)                          │     │
│  │  • Mutual information scores                                            │     │
│  │  • LASSO regularization (L1 shrinkage)                                  │     │
│  └────────────────────────────┬────────────────────────────────────────────┘     │
│                               ▼                                                  │
│       ┌───────────────────────┴───────────────────────┐                         │
│       ▼                                               ▼                         │
│  ┌──────────────────────┐                ┌──────────────────────────┐            │
│  │  K-MEANS CLUSTERING  │                │  DEFAULT PREDICTION      │            │
│  │  • Customer segments │                │  MODEL                   │            │
│  │  • Risk profiling    │                │  • XGBoost / LightGBM    │            │
│  │  • Cluster features  │───────────────►│  • SMOTE for imbalance   │            │
│  │  • Elbow + silhouette│                │  • 95% F1 score          │            │
│  └──────────────────────┘                │  • Stratified K-Fold CV  │            │
│                                          └──────────┬───────────────┘            │
│                                                     ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                   STATISTICAL EVALUATION                                │     │
│  │  • Hypothesis testing (t-tests, chi-squared, Mann-Whitney)              │     │
│  │  • Distribution analysis (KS test, Anderson-Darling)                    │     │
│  │  • Effect size computation (Cohen's d, Cramér's V)                      │     │
│  └────────────────────────────┬────────────────────────────────────────────┘     │
│                               ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                 OCF METRICS & POWER BI DASHBOARDS                       │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │     │
│  │  │  OCF Metrics     │  │  Risk Segment    │  │  Executive Summary   │  │     │
│  │  │  Dashboard       │  │  Dashboard       │  │  Reports             │  │     │
│  │  │  (Cash flow      │  │  (Cluster-based  │  │  (Funding &          │  │     │
│  │  │   health)        │  │   risk view)     │  │   strategy support)  │  │     │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python (pandas, NumPy, scikit-learn, scipy) | Core data processing, modeling, statistical tests |
| **NLP / Parsing** | regex, spaCy, custom rule-based NER | Address decomposition, standardization, error correction |
| **JSON Processing** | `json`, `pandas.json_normalize()` | Flattening nested KYC JSON structures |
| **Feature Engineering** | pandas, NumPy, custom modules | Lifestyle features, address features, behavioral signals |
| **Feature Selection** | scikit-learn (chi2, mutual_info), statsmodels, LASSO | Statistical feature filtering and regularization |
| **Clustering** | scikit-learn (KMeans, DBSCAN benchmarked) | Customer segmentation and risk profiling |
| **Classification** | XGBoost, LightGBM, scikit-learn (Random Forest, Logistic Regression benchmarked) | Default prediction (95% F1) |
| **Imbalance Handling** | imbalanced-learn (SMOTE, ADASYN) | Oversampling minority default class |
| **Statistical Analysis** | scipy.stats, statsmodels | Hypothesis testing, distribution analysis |
| **Visualization** | Power BI, matplotlib, seaborn | Executive dashboards, EDA visualizations |
| **Database** | SQL (PostgreSQL) | Internal data warehouse queries |
| **Version Control** | Git | Code versioning |

---

## 1.4 Timeline & Team Structure

| Phase | Timeline | My Role |
|-------|----------|---------|
| Data exploration & JSON processing | May – Jun 2020 (4 weeks) | Primary owner of data parsing pipeline |
| Address NLP & standardization | Jun – Jul 2020 (4 weeks) | Designed and implemented address parser |
| Lifestyle feature engineering | Jul – Aug 2020 (4 weeks) | Created behavioral feature library |
| Feature selection & statistical analysis | Aug – Sep 2020 (4 weeks) | Ran all selection techniques, hypothesis tests |
| Clustering & default prediction model | Sep – Oct 2020 (4 weeks) | Model training, tuning, validation |
| OCF metrics, dashboards & reporting | Oct – Nov 2020 (4 weeks) | Power BI design, executive summaries |

**Team:** 1 senior data scientist (mentor), 1 data engineer (pipeline support), credit risk team (domain context), business stakeholders (dashboard consumers). **I was the primary researcher and model builder.**

---

# 2. Deep Technical Walkthrough

---

## 2.1 Raw JSON Data Processing

### The Problem with KYC JSON Data

Axio's customer KYC data was collected through digital onboarding flows—document uploads, form fills, third-party API enrichments. This data was stored as raw JSON blobs with no standardized schema. A single customer record might contain nested objects for personal details, address arrays (current/permanent/office), employment info, and financial data.

**Example raw JSON (simplified):**

```json
{
  "customer_id": "AXC-2020-44891",
  "personal_info": {
    "name": "Rajesh Kumar Sharma",
    "dob": "1988-05-14",
    "pan": "BXQPS1234K"
  },
  "addresses": [
    {
      "type": "current",
      "raw_text": "Flat 302, 3rd Flr, Prestige Shantiniketan, Whitfield Rd, ITPL Main Rd, Bangalore - 560048, KA",
      "source": "aadhaar"
    },
    {
      "type": "permanent",
      "raw_text": "H.No. 45, Mohalla Ramnagar, Nr Shiv Mandir, Distt Meerut, UP 250001",
      "source": "pan_card"
    },
    {
      "type": "office",
      "raw_text": "WeWork Galaxy, #43 Residency Road, Blore 560025",
      "source": "self_declared"
    }
  ],
  "financial": {
    "monthly_income": 85000,
    "employer": "TechCorp Solutions Pvt Ltd",
    "employment_type": "salaried",
    "bank_statements": {
      "avg_balance_3m": 42000,
      "salary_credits": [85000, 85000, 84500],
      "upi_transactions": 147,
      "categories": {
        "food_delivery": 8500,
        "online_shopping": 12000,
        "entertainment": 3500,
        "utilities": 4200,
        "emi_payments": 18000,
        "cash_withdrawals": 15000
      }
    }
  },
  "bureau": {
    "score": 742,
    "active_accounts": 4,
    "dpd_history": [0, 0, 30, 0, 0, 0],
    "total_outstanding": 320000
  },
  "loan_application": {
    "amount_requested": 200000,
    "tenure_months": 12,
    "purpose": "personal"
  }
}
```

### JSON Flattening Strategy

```python
import json
import os
import pandas as pd
from typing import Any, Dict, List

def flatten_json(nested: dict, prefix: str = '', sep: str = '_') -> dict:
    """Recursively flatten nested JSON into a flat dictionary."""
    flat = {}
    for key, value in nested.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            if all(isinstance(item, (int, float, str)) for item in value):
                # Numeric/string lists: store as-is for later aggregation
                flat[new_key] = value
            else:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(flatten_json(item, f"{new_key}{sep}{i}", sep))
                    else:
                        flat[f"{new_key}{sep}{i}"] = item
        else:
            flat[new_key] = value
    return flat


def process_kyc_files(json_dir: str) -> pd.DataFrame:
    """Process all KYC JSON files from a directory into a flat DataFrame."""
    records = []
    errors = []
    
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(json_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            flat = flatten_json(raw)
            flat['_source_file'] = filename
            records.append(flat)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            errors.append({'file': filename, 'error': str(e)})
    
    df = pd.DataFrame(records)
    print(f"Processed {len(records)} records, {len(errors)} errors")
    return df, errors


# Alternative: Using pandas json_normalize for specific structures
def extract_addresses(raw_records: List[dict]) -> pd.DataFrame:
    """Extract and normalize address data from KYC JSON."""
    address_rows = []
    for record in raw_records:
        cust_id = record.get('customer_id')
        for addr in record.get('addresses', []):
            address_rows.append({
                'customer_id': cust_id,
                'address_type': addr.get('type'),
                'raw_address': addr.get('raw_text', ''),
                'source': addr.get('source')
            })
    return pd.DataFrame(address_rows)
```

### Data Quality Checks After Flattening

```python
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data quality summary for flattened KYC data."""
    report = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.values,
        'non_null_count': df.notnull().sum().values,
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2).values,
        'unique_count': df.nunique().values,
        'sample_value': [df[col].dropna().iloc[0] if df[col].notnull().any() else None 
                         for col in df.columns]
    })
    return report.sort_values('null_pct', ascending=False)
```

---

## 2.2 Address Parsing with NLP

### Why Address Parsing Matters in Fintech

In Indian lending, address data is a **critical risk signal**:
- **Address completeness** → customers who provide complete, verifiable addresses are less likely to default
- **Address tier** → metro vs. tier-2/3 cities have different default profiles
- **Address consistency** → mismatches between declared and bureau addresses signal potential fraud
- **Pin code** → enables geographic risk mapping (some pin codes correlate with higher default rates)
- **Address stability** → frequent address changes can indicate instability

### The Challenge with Indian Addresses

Indian addresses are notoriously unstructured:
- No standard format (unlike US ZIP+4 or UK postcode system)
- Mix of Hindi transliterations and English
- Common abbreviations: "Rd" → Road, "Blore" → Bangalore, "Nr" → Near, "Opp" → Opposite
- Missing or incorrect pin codes
- Landmark-based directions ("Near Shiv Mandir", "Behind Big Bazaar")
- Multiple naming conventions for the same area

### Address Parser Implementation

```python
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedAddress:
    flat_house_no: Optional[str] = None
    floor: Optional[str] = None
    building_name: Optional[str] = None
    street: Optional[str] = None
    area_locality: Optional[str] = None
    landmark: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    pin_code: Optional[str] = None
    completeness_score: float = 0.0


# --- Indian State Mappings ---
STATE_ABBREVIATIONS = {
    'KA': 'Karnataka', 'MH': 'Maharashtra', 'TN': 'Tamil Nadu',
    'UP': 'Uttar Pradesh', 'DL': 'Delhi', 'TG': 'Telangana',
    'AP': 'Andhra Pradesh', 'RJ': 'Rajasthan', 'GJ': 'Gujarat',
    'WB': 'West Bengal', 'KL': 'Kerala', 'MP': 'Madhya Pradesh',
    'HR': 'Haryana', 'PB': 'Punjab', 'BR': 'Bihar',
    'OR': 'Odisha', 'JH': 'Jharkhand', 'CG': 'Chhattisgarh',
    'GA': 'Goa', 'HP': 'Himachal Pradesh', 'UK': 'Uttarakhand',
}

# --- City Name Corrections ---
CITY_CORRECTIONS = {
    'blore': 'Bangalore', 'bengaluru': 'Bangalore', 'b\'lore': 'Bangalore',
    'bombay': 'Mumbai', 'calcutta': 'Kolkata', 'madras': 'Chennai',
    'pune': 'Pune', 'hyd': 'Hyderabad', 'vizag': 'Visakhapatnam',
    'del': 'Delhi', 'n delhi': 'New Delhi', 'noida': 'Noida',
    'ggn': 'Gurgaon', 'gurugram': 'Gurgaon',
}

# --- Abbreviation Expansion ---
ABBREVIATIONS = {
    r'\brd\b': 'Road', r'\bst\b': 'Street', r'\bnr\b': 'Near',
    r'\bopp\b': 'Opposite', r'\bblk\b': 'Block', r'\bflr\b': 'Floor',
    r'\bapt\b': 'Apartment', r'\bext\b': 'Extension', r'\bph\b': 'Phase',
    r'\bsec\b': 'Sector', r'\bdistt?\b': 'District', r'\bvill\b': 'Village',
    r'\bmkt\b': 'Market', r'\bcolny\b': 'Colony', r'\bnagar\b': 'Nagar',
    r'\bh\.?\s*no\.?\b': 'House No',
}


def preprocess_address(raw: str) -> str:
    """Clean and normalize raw address text."""
    if not raw or not isinstance(raw, str):
        return ''
    
    # Lowercase for matching
    text = raw.strip()
    
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.\-/#]', '', text)
    
    # Expand abbreviations (case-insensitive)
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_pin_code(text: str) -> Tuple[Optional[str], str]:
    """Extract 6-digit Indian pin code from address text."""
    # Indian pin codes: 6 digits, first digit 1-9
    pin_match = re.search(r'\b([1-9]\d{5})\b', text)
    if pin_match:
        pin = pin_match.group(1)
        text_without_pin = text[:pin_match.start()] + text[pin_match.end():]
        return pin, text_without_pin.strip()
    return None, text


def extract_state(text: str) -> Tuple[Optional[str], str]:
    """Extract state from address text using abbreviations and full names."""
    # Check abbreviations first (they appear at end typically)
    for abbr, full_name in STATE_ABBREVIATIONS.items():
        pattern = rf'\b{re.escape(abbr)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
            return full_name, text
    
    # Check full state names
    for full_name in STATE_ABBREVIATIONS.values():
        if full_name.lower() in text.lower():
            text = re.sub(re.escape(full_name), '', text, flags=re.IGNORECASE).strip()
            return full_name, text
    
    return None, text


def extract_city(text: str) -> Tuple[Optional[str], str]:
    """Extract and standardize city name from address text."""
    text_lower = text.lower()
    for variant, standard in CITY_CORRECTIONS.items():
        if variant in text_lower:
            text = re.sub(re.escape(variant), '', text, flags=re.IGNORECASE).strip()
            return standard, text
    
    # Major cities direct match
    major_cities = [
        'Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Hyderabad',
        'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Chandigarh', 'Indore',
        'Bhopal', 'Nagpur', 'Coimbatore', 'Kochi', 'Surat', 'Vadodara',
    ]
    for city in major_cities:
        if city.lower() in text_lower:
            text = re.sub(re.escape(city), '', text, flags=re.IGNORECASE).strip()
            return city, text
    
    return None, text


def extract_flat_building(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract flat/house number, floor, and building name."""
    flat_no = None
    floor = None
    building = None
    
    # Flat/House number: "Flat 302", "H.No. 45", "#43"
    flat_match = re.search(
        r'(?:flat|house\s*no|h\.?\s*no\.?|#)\s*[-:]?\s*(\w+[\w/-]*)',
        text, re.IGNORECASE
    )
    if flat_match:
        flat_no = flat_match.group(1)
        text = text[:flat_match.start()] + text[flat_match.end():]
    
    # Floor: "3rd Flr", "2nd Floor", "Ground Floor"
    floor_match = re.search(
        r'(\d+(?:st|nd|rd|th)?\s*(?:floor|flr)|ground\s*floor|basement)',
        text, re.IGNORECASE
    )
    if floor_match:
        floor = floor_match.group(1)
        text = text[:floor_match.start()] + text[floor_match.end():]
    
    # Building name: common patterns
    bldg_match = re.search(
        r'((?:prestige|brigade|mantri|sobha|godrej|dlf|embassy|salarpuria|'
        r'purva|manyata|wework|regus)\s+[\w\s]+?)(?:,|\s{2,}|$)',
        text, re.IGNORECASE
    )
    if bldg_match:
        building = bldg_match.group(1).strip()
        text = text[:bldg_match.start()] + text[bldg_match.end():]
    
    return flat_no, floor, building, text.strip()


def extract_landmark(text: str) -> Tuple[Optional[str], str]:
    """Extract landmark references from address."""
    landmark_match = re.search(
        r'(?:near|nr|opp|opposite|behind|beside|next\s+to|adjacent)\s+(.+?)(?:,|$)',
        text, re.IGNORECASE
    )
    if landmark_match:
        landmark = landmark_match.group(0).strip().rstrip(',')
        text = text[:landmark_match.start()] + text[landmark_match.end():]
        return landmark, text.strip()
    return None, text


def compute_completeness(parsed: ParsedAddress) -> float:
    """Score address completeness from 0 to 1."""
    fields = {
        'pin_code': 0.25,        # Most important for risk mapping
        'city': 0.20,            # City-level risk signals
        'state': 0.15,           # State identification
        'area_locality': 0.15,   # Locality detail
        'flat_house_no': 0.10,   # Specific location
        'street': 0.10,          # Street-level detail
        'building_name': 0.05,   # Building info
    }
    score = 0.0
    for field, weight in fields.items():
        if getattr(parsed, field) is not None:
            score += weight
    return round(score, 3)


def parse_address(raw_text: str) -> ParsedAddress:
    """Full address parsing pipeline."""
    parsed = ParsedAddress()
    
    # Step 1: Preprocess
    text = preprocess_address(raw_text)
    if not text:
        return parsed
    
    # Step 2: Extract structured components (order matters)
    parsed.pin_code, text = extract_pin_code(text)
    parsed.state, text = extract_state(text)
    parsed.city, text = extract_city(text)
    parsed.flat_house_no, parsed.floor, parsed.building_name, text = extract_flat_building(text)
    parsed.landmark, text = extract_landmark(text)
    
    # Step 3: Remaining text → area/locality + street
    remaining = re.sub(r'[,\s]+', ' ', text).strip().rstrip(',').strip()
    if remaining:
        parts = [p.strip() for p in remaining.split(',') if p.strip()]
        if len(parts) >= 2:
            parsed.street = parts[0]
            parsed.area_locality = ', '.join(parts[1:])
        else:
            parsed.area_locality = remaining
    
    # Step 4: Compute completeness score
    parsed.completeness_score = compute_completeness(parsed)
    
    return parsed
```

### Pin Code Validation & Enrichment

```python
# Pin code → city/state mapping for validation and enrichment
PIN_CODE_LOOKUP = {
    '560048': {'city': 'Bangalore', 'state': 'Karnataka', 'tier': 1},
    '250001': {'city': 'Meerut', 'state': 'Uttar Pradesh', 'tier': 2},
    '560025': {'city': 'Bangalore', 'state': 'Karnataka', 'tier': 1},
    # ... loaded from India Post database
}

CITY_TIER_MAP = {
    'tier_1': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad',
               'Pune', 'Ahmedabad'],
    'tier_2': ['Jaipur', 'Lucknow', 'Chandigarh', 'Indore', 'Bhopal', 'Nagpur',
               'Coimbatore', 'Kochi', 'Surat', 'Vadodara', 'Meerut'],
}

def validate_and_enrich_pin(parsed: ParsedAddress) -> ParsedAddress:
    """Validate pin code and enrich missing city/state."""
    if parsed.pin_code and parsed.pin_code in PIN_CODE_LOOKUP:
        lookup = PIN_CODE_LOOKUP[parsed.pin_code]
        # Fill missing fields from pin code lookup
        if not parsed.city:
            parsed.city = lookup['city']
        if not parsed.state:
            parsed.state = lookup['state']
        # Cross-validate: flag if declared city doesn't match pin
        if parsed.city and parsed.city.lower() != lookup['city'].lower():
            parsed.pin_city_mismatch = True  # fraud signal!
    return parsed


def get_city_tier(city: str) -> int:
    """Return city tier (1, 2, or 3) for risk profiling."""
    if not city:
        return 0  # unknown
    city_lower = city.lower()
    for tier, cities in CITY_TIER_MAP.items():
        if any(c.lower() == city_lower for c in cities):
            return int(tier.split('_')[1])
    return 3  # default to tier 3
```

### Address Features for Risk Modeling

```python
def engineer_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create address-derived features for default prediction."""
    
    # Parse all addresses
    df['parsed'] = df['raw_address'].apply(parse_address)
    
    # --- Completeness features ---
    df['addr_completeness_score'] = df['parsed'].apply(lambda p: p.completeness_score)
    df['addr_has_pin'] = df['parsed'].apply(lambda p: int(p.pin_code is not None))
    df['addr_has_building'] = df['parsed'].apply(lambda p: int(p.building_name is not None))
    df['addr_has_landmark'] = df['parsed'].apply(lambda p: int(p.landmark is not None))
    
    # --- Geographic risk features ---
    df['city_tier'] = df['parsed'].apply(lambda p: get_city_tier(p.city))
    df['is_metro'] = (df['city_tier'] == 1).astype(int)
    df['state'] = df['parsed'].apply(lambda p: p.state)
    
    # --- Consistency features (across address types) ---
    # Compare current vs permanent address
    addr_groups = df.groupby('customer_id')
    df['addr_city_match'] = addr_groups.apply(
        lambda g: int(g['parsed'].apply(lambda p: p.city).nunique() == 1)
    ).reindex(df.index).fillna(0).astype(int)
    
    df['addr_state_match'] = addr_groups.apply(
        lambda g: int(g['parsed'].apply(lambda p: p.state).nunique() == 1)
    ).reindex(df.index).fillna(0).astype(int)
    
    # Address length as proxy for detail/effort
    df['addr_raw_length'] = df['raw_address'].str.len()
    df['addr_word_count'] = df['raw_address'].str.split().str.len()
    
    return df
```

---

## 2.3 Lifestyle Feature Engineering

### Spending Habits & Purchase Type Features

Customer lifestyle—how they spend, what they buy, behavioral patterns—provides powerful signals for default prediction beyond traditional bureau data.

```python
import numpy as np

def engineer_lifestyle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lifestyle and behavioral features from transaction data."""
    
    # ─── SPENDING RATIO FEATURES ───
    df['essential_spend_ratio'] = (
        (df['spend_utilities'] + df['spend_groceries'] + df['spend_rent']) /
        df['monthly_income'].clip(lower=1)
    )
    df['discretionary_spend_ratio'] = (
        (df['spend_entertainment'] + df['spend_dining'] + df['spend_shopping']) /
        df['monthly_income'].clip(lower=1)
    )
    df['emi_to_income_ratio'] = df['spend_emi'] / df['monthly_income'].clip(lower=1)
    df['cash_withdrawal_ratio'] = df['spend_cash_withdrawals'] / df['monthly_income'].clip(lower=1)
    
    # ─── SPENDING DIVERSITY (ENTROPY) ───
    spend_cols = [c for c in df.columns if c.startswith('spend_')]
    spend_matrix = df[spend_cols].values
    spend_probs = spend_matrix / spend_matrix.sum(axis=1, keepdims=True).clip(min=1e-10)
    df['spending_entropy'] = -np.sum(
        spend_probs * np.log2(spend_probs + 1e-10), axis=1
    )
    
    # ─── PURCHASE TYPE FEATURES ───
    df['online_shopping_intensity'] = df['online_txn_count'] / df['total_txn_count'].clip(lower=1)
    df['food_delivery_frequency'] = df['food_delivery_txn_count'] / 30  # per day
    df['luxury_spend_flag'] = (
        (df['spend_luxury'] > df['monthly_income'] * 0.15) |
        (df['spend_travel'] > df['monthly_income'] * 0.20)
    ).astype(int)
    
    # ─── BEHAVIORAL SIGNALS ───
    df['salary_consistency'] = df['salary_credits'].apply(
        lambda x: np.std(x) / np.mean(x) if isinstance(x, list) and len(x) > 1 else np.nan
    )
    df['avg_balance_to_income'] = df['avg_balance_3m'] / df['monthly_income'].clip(lower=1)
    df['balance_volatility'] = df['balance_std_3m'] / df['avg_balance_3m'].clip(lower=1)
    
    # ─── TRANSACTION PATTERN FEATURES ───
    df['txn_frequency_per_day'] = df['total_txn_count'] / 30
    df['avg_txn_amount'] = df['total_spend'] / df['total_txn_count'].clip(lower=1)
    df['high_value_txn_ratio'] = df['txn_above_5000_count'] / df['total_txn_count'].clip(lower=1)
    
    # ─── UPI USAGE FEATURES ───
    df['upi_adoption'] = (df['upi_txn_count'] > 0).astype(int)
    df['upi_to_total_ratio'] = df['upi_txn_count'] / df['total_txn_count'].clip(lower=1)
    
    # ─── LATE-NIGHT SPENDING (BEHAVIORAL RISK) ───
    df['late_night_txn_ratio'] = df['txn_11pm_6am_count'] / df['total_txn_count'].clip(lower=1)
    
    # ─── FINANCIAL STRESS INDICATORS ───
    df['bounce_rate'] = df['bounced_txn_count'] / df['total_debit_count'].clip(lower=1)
    df['overdraft_frequency'] = df['overdraft_count_3m']
    df['min_balance_breach_count'] = df['min_bal_breach_3m']
    
    return df
```

### Feature Rationale Table

| Feature | Signal | Default Risk Logic |
|---------|--------|-------------------|
| `essential_spend_ratio` | Basic needs vs income | High ratio → less financial buffer → higher risk |
| `discretionary_spend_ratio` | Lifestyle spending | Very high → overspending; very low → might indicate data issues |
| `emi_to_income_ratio` | Existing debt burden | > 0.5 → heavily leveraged → high default risk |
| `cash_withdrawal_ratio` | Cash dependence | High cash usage → less traceable income → risk signal |
| `spending_entropy` | Spending diversity | Very low → concentrated spending pattern (possible anomaly) |
| `salary_consistency` | Income stability | High CV → irregular income → payment uncertainty |
| `bounce_rate` | Payment reliability | Direct indicator of inability to meet financial obligations |
| `addr_completeness_score` | Address quality | Low completeness → possible fraud or carelessness |
| `city_tier` | Geographic risk | Tier-1 cities typically show lower default rates |
| `late_night_txn_ratio` | Behavioral risk | High late-night activity → potential impulsive spending |

---

## 2.4 Feature Selection Pipeline

### Step 1: Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_filter(df: pd.DataFrame, features: list, threshold: float = 0.7) -> list:
    """Remove highly correlated features, keeping the one with higher target correlation."""
    
    corr_matrix = df[features].corr().abs()
    target_corr = df[features].corrwith(df['default_flag']).abs()
    
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = set()
    for col in upper.columns:
        high_corr_pairs = upper.index[upper[col] > threshold].tolist()
        for pair in high_corr_pairs:
            # Drop the feature with lower target correlation
            if target_corr.get(col, 0) >= target_corr.get(pair, 0):
                to_drop.add(pair)
            else:
                to_drop.add(col)
    
    kept = [f for f in features if f not in to_drop]
    print(f"Correlation filter: {len(features)} → {len(kept)} features "
          f"(dropped {len(to_drop)} at |r| > {threshold})")
    return kept
```

### Step 2: Chi-Squared Test (Categorical Features)

The chi-squared test measures whether there is a statistically significant association between categorical features and the default target.

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where $O_{ij}$ = observed frequency, $E_{ij}$ = expected frequency under independence.

```python
from sklearn.feature_selection import chi2, SelectKBest
from scipy.stats import chi2_contingency

def chi_squared_selection(df: pd.DataFrame, cat_features: list,
                          target: str = 'default_flag', alpha: float = 0.05) -> list:
    """Select categorical features with significant chi-squared association."""
    
    significant = []
    results = []
    
    for feat in cat_features:
        contingency = pd.crosstab(df[feat], df[target])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
        
        # Cramér's V for effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
        
        results.append({
            'feature': feat,
            'chi2': round(chi2_stat, 2),
            'p_value': p_value,
            'cramers_v': round(cramers_v, 4),
            'significant': p_value < alpha
        })
        
        if p_value < alpha:
            significant.append(feat)
    
    results_df = pd.DataFrame(results).sort_values('chi2', ascending=False)
    print(f"Chi-squared: {len(significant)}/{len(cat_features)} significant at α={alpha}")
    return significant, results_df
```

### Step 3: Mutual Information

Mutual information captures **non-linear** relationships between features and the target, unlike correlation which only captures linear ones.

$$I(X; Y) = \sum_{x}\sum_{y} p(x, y) \log \frac{p(x, y)}{p(x) \cdot p(y)}$$

```python
from sklearn.feature_selection import mutual_info_classif

def mutual_information_selection(X: pd.DataFrame, y: pd.Series,
                                 top_k: int = 30) -> list:
    """Select top-k features by mutual information with the target."""
    
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    selected = mi_df.head(top_k)['feature'].tolist()
    
    print(f"Mutual Information top {top_k}:")
    print(mi_df.head(top_k).to_string(index=False))
    
    return selected, mi_df
```

### Step 4: LASSO Regularization (L1 Feature Selection)

LASSO shrinks unimportant feature coefficients to exactly zero, providing automatic feature selection.

$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

```python
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

def lasso_feature_selection(X: pd.DataFrame, y: pd.Series,
                            n_alphas: int = 100) -> list:
    """Use LASSO (L1 regularization) to select non-zero coefficient features."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LogisticRegressionCV with L1 for classification
    lasso = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=5,
        scoring='f1',
        max_iter=5000,
        random_state=42,
        Cs=np.logspace(-4, 2, n_alphas)
    )
    lasso.fit(X_scaled, y)
    
    # Features with non-zero coefficients
    coef = np.abs(lasso.coef_[0])
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'abs_coefficient': coef
    }).sort_values('abs_coefficient', ascending=False)
    
    selected = feature_importance[feature_importance['abs_coefficient'] > 1e-6]['feature'].tolist()
    
    print(f"LASSO: {len(selected)}/{len(X.columns)} features selected "
          f"(best C={lasso.C_[0]:.4f})")
    return selected, feature_importance
```

### Combined Feature Selection Strategy

```python
def combined_feature_selection(df, features, target='default_flag'):
    """Ensemble of selection methods — keep features selected by 2+ methods."""
    
    X = df[features]
    y = df[target]
    
    # Method 1: Correlation with target
    target_corr = X.corrwith(y).abs()
    corr_top = target_corr.nlargest(40).index.tolist()
    
    # Method 2: Mutual information
    mi_top, _ = mutual_information_selection(X, y, top_k=40)
    
    # Method 3: LASSO
    lasso_top, _ = lasso_feature_selection(X, y)
    
    # Method 4: Chi-squared (categorical only)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        chi2_top, _ = chi_squared_selection(df, cat_cols, target)
    else:
        chi2_top = []
    
    # Ensemble: features selected by >= 2 methods
    from collections import Counter
    all_selected = corr_top + mi_top + lasso_top + chi2_top
    counts = Counter(all_selected)
    final_features = [f for f, count in counts.items() if count >= 2]
    
    print(f"\nEnsemble selection: {len(final_features)} features selected by 2+ methods")
    return final_features
```

---

## 2.5 Customer Segmentation with Clustering

### K-Means Clustering for Risk Profiling

Clustering groups customers with similar financial/behavioral profiles, creating powerful segment-level features for the default model.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> int:
    """Use elbow method + silhouette score to find optimal K."""
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000))
    
    # Silhouette-based selection
    optimal_k = k_range[np.argmax(silhouettes)]
    print(f"Optimal K by silhouette: {optimal_k} (score={max(silhouettes):.3f})")
    
    return optimal_k, inertias, silhouettes


def perform_customer_segmentation(df: pd.DataFrame,
                                   cluster_features: list) -> pd.DataFrame:
    """Segment customers using K-Means and profile clusters."""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features].fillna(0))
    
    # Find optimal K
    optimal_k, _, _ = find_optimal_k(X_scaled)
    
    # Fit final K-Means
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)
    
    # Profile clusters
    profile = df.groupby('cluster').agg({
        'default_flag': ['mean', 'sum', 'count'],
        'monthly_income': 'mean',
        'emi_to_income_ratio': 'mean',
        'essential_spend_ratio': 'mean',
        'discretionary_spend_ratio': 'mean',
        'bounce_rate': 'mean',
        'addr_completeness_score': 'mean',
        'city_tier': lambda x: x.mode().iloc[0] if len(x) > 0 else None,
    })
    profile.columns = ['default_rate', 'default_count', 'total_count',
                        'avg_income', 'avg_emi_ratio', 'avg_essential_ratio',
                        'avg_discretionary_ratio', 'avg_bounce_rate',
                        'avg_addr_completeness', 'modal_city_tier']
    
    print("\n--- Cluster Profiles ---")
    print(profile.round(3).to_string())
    
    return df, profile
```

### Example Cluster Profiles

| Cluster | Default Rate | Avg Income | EMI/Income | Bounce Rate | Addr Completeness | Profile Label |
|---------|-------------|------------|-----------|-------------|-------------------|---------------|
| 0 | 3.2% | ₹95,000 | 0.28 | 0.01 | 0.92 | **Low Risk — Stable Salaried** |
| 1 | 8.7% | ₹55,000 | 0.42 | 0.05 | 0.78 | **Medium Risk — Stretched Budget** |
| 2 | 22.4% | ₹38,000 | 0.61 | 0.15 | 0.55 | **High Risk — Overleveraged** |
| 3 | 35.1% | ₹30,000 | 0.72 | 0.28 | 0.41 | **Very High Risk — Stress Indicators** |

### Using Cluster as a Feature

```python
# One-hot encode cluster membership as features for the default model
cluster_dummies = pd.get_dummies(df['cluster'], prefix='cluster')
df = pd.concat([df, cluster_dummies], axis=1)

# Or use cluster default rate as a risk score feature
cluster_default_rates = df.groupby('cluster')['default_flag'].mean()
df['cluster_default_rate'] = df['cluster'].map(cluster_default_rates)
```

---

## 2.6 Statistical Evaluation & Hypothesis Testing

### Hypothesis Tests for Feature Significance

Before including features in the model, statistical tests validate that observed differences between defaulters and non-defaulters are genuine—not due to random chance.

```python
from scipy import stats

def statistical_evaluation(df: pd.DataFrame, features: list,
                            target: str = 'default_flag') -> pd.DataFrame:
    """Run hypothesis tests for each feature vs default target."""
    
    defaulters = df[df[target] == 1]
    non_defaulters = df[df[target] == 0]
    results = []
    
    for feat in features:
        feat_data = df[feat].dropna()
        
        if feat_data.dtype in ['object', 'category'] or feat_data.nunique() < 10:
            # Categorical → chi-squared test
            contingency = pd.crosstab(df[feat], df[target])
            chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            effect = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
            test_name = 'Chi-squared'
            
        else:
            # Continuous → check normality first
            d = defaulters[feat].dropna()
            nd = non_defaulters[feat].dropna()
            
            # Shapiro-Wilk on sample (max 5000 for performance)
            sample_size = min(len(d), len(nd), 5000)
            _, p_normal = stats.shapiro(d.sample(sample_size, random_state=42))
            
            if p_normal > 0.05:
                # Normal → independent t-test
                t_stat, p_val = stats.ttest_ind(d, nd, equal_var=False)
                effect = (d.mean() - nd.mean()) / np.sqrt(
                    (d.std()**2 + nd.std()**2) / 2
                )  # Cohen's d
                test_name = "Welch's t-test"
            else:
                # Non-normal → Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(d, nd, alternative='two-sided')
                # Rank-biserial correlation as effect size
                n1, n2 = len(d), len(nd)
                effect = 1 - (2 * u_stat) / (n1 * n2)
                test_name = 'Mann-Whitney U'
        
        results.append({
            'feature': feat,
            'test': test_name,
            'p_value': p_val,
            'effect_size': round(abs(effect), 4),
            'significant': p_val < 0.05,
            'practical_sig': abs(effect) > 0.2  # medium+ effect
        })
    
    return pd.DataFrame(results).sort_values('p_value')
```

### Distribution Analysis

```python
def distribution_analysis(df: pd.DataFrame, features: list,
                           target: str = 'default_flag'):
    """Compare distributions between defaulters and non-defaulters."""
    
    defaulters = df[df[target] == 1]
    non_defaulters = df[df[target] == 0]
    
    for feat in features:
        d = defaulters[feat].dropna()
        nd = non_defaulters[feat].dropna()
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(d, nd)
        
        # Anderson-Darling test (normality)
        ad_stat, ad_crit, ad_sig = stats.anderson(d, dist='norm')
        
        print(f"\n{feat}:")
        print(f"  Defaulters:     mean={d.mean():.3f}, median={d.median():.3f}, std={d.std():.3f}")
        print(f"  Non-defaulters: mean={nd.mean():.3f}, median={nd.median():.3f}, std={nd.std():.3f}")
        print(f"  KS test: stat={ks_stat:.4f}, p={ks_pval:.2e}")
        print(f"  Distributions {'significantly different' if ks_pval < 0.05 else 'not significantly different'}")
```

---

## 2.7 Default Prediction Model (95% F1)

### Class Imbalance Handling

Default prediction is inherently imbalanced—typically 5–15% default rate. Without proper handling, models learn to always predict "no default."

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

def handle_imbalance(X_train, y_train, method='smote'):
    """Apply oversampling to handle class imbalance."""
    
    print(f"Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=42, k_neighbors=5)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42)
    
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"After {method.upper()}: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    
    return X_res, y_res
```

### Model Training & Comparison

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, classification_report,
                              confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_and_compare_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance."""
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            eval_metric='logloss', random_state=42, use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            is_unbalance=True, random_state=42, verbose=-1
        ),
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Cross-validation F1
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        # Fit and evaluate on test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': name,
            'CV F1 (mean±std)': f"{cv_f1.mean():.3f}±{cv_f1.std():.3f}",
            'Test F1': f1_score(y_test, y_pred),
            'Test Precision': precision_score(y_test, y_pred),
            'Test Recall': recall_score(y_test, y_pred),
            'Test ROC-AUC': roc_auc_score(y_test, y_prob),
        })
        
        print(f"\n{'='*50}")
        print(f"{name}")
        print(classification_report(y_test, y_pred))
    
    return pd.DataFrame(results)
```

### Model Comparison Results

| Model | CV F1 | Test F1 | Precision | Recall | ROC-AUC |
|-------|-------|---------|-----------|--------|---------|
| Logistic Regression | 0.87 ± 0.02 | 0.88 | 0.86 | 0.90 | 0.92 |
| Random Forest | 0.91 ± 0.02 | 0.92 | 0.91 | 0.93 | 0.96 |
| **XGBoost** | **0.94 ± 0.01** | **0.95** | **0.94** | **0.96** | **0.98** |
| LightGBM | 0.93 ± 0.01 | 0.94 | 0.93 | 0.95 | 0.97 |

### Hyperparameter Tuning (XGBoost)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def tune_xgboost(X_train, y_train):
    """Tune XGBoost hyperparameters with RandomizedSearchCV."""
    
    param_distributions = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(4, 10),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 2),
    }
    
    xgb = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    
    search = RandomizedSearchCV(
        xgb, param_distributions, n_iter=100,
        scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    
    print(f"Best F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    
    return search.best_estimator_
```

### Feature Importance Analysis

```python
def feature_importance_analysis(model, feature_names: list, top_n: int = 20):
    """Analyze and display feature importance from the trained model."""
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df
```

### Expected Top Features

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | `bounce_rate` | 0.142 | Behavioral |
| 2 | `emi_to_income_ratio` | 0.128 | Financial |
| 3 | `bureau_score` | 0.115 | Bureau |
| 4 | `cluster_default_rate` | 0.098 | Clustering |
| 5 | `cash_withdrawal_ratio` | 0.087 | Lifestyle |
| 6 | `addr_completeness_score` | 0.076 | Address |
| 7 | `salary_consistency` | 0.065 | Behavioral |
| 8 | `essential_spend_ratio` | 0.058 | Lifestyle |
| 9 | `city_tier` | 0.052 | Address |
| 10 | `balance_volatility` | 0.047 | Behavioral |

---

## 2.8 OCF Metrics Design

### What is OCF (Operating Cash Flow)?

Operating Cash Flow measures the cash generated by a company's core business operations. In a **lending fintech** like Axio, OCF metrics are critical for understanding the health of the loan portfolio and the company's ability to fund future lending.

**OCF Formula:**

$$OCF = \text{Net Income} + \text{Non-Cash Charges} + \Delta \text{Working Capital}$$

**For a lending company, adapted OCF metrics include:**

$$OCF_{lending} = \text{Interest Income Collected} - \text{Operating Expenses} - \text{Provisions for Defaults} + \text{Principal Repayments Received}$$

### Key OCF Metrics I Designed

| Metric | Formula | Business Meaning |
|--------|---------|-----------------|
| **Collection Efficiency** | $\frac{\text{Actual Collections}}{\text{Expected Collections}} \times 100$ | How much of expected EMI payments are actually received |
| **Provision Coverage Ratio** | $\frac{\text{Provisions}}{\text{Gross NPAs}}$ | How well loan losses are covered by provisions |
| **Net Interest Margin** | $\frac{\text{Interest Income} - \text{Interest Expense}}{\text{Average Earning Assets}}$ | Core profitability of lending operations |
| **Cost-to-Income Ratio** | $\frac{\text{Operating Expenses}}{\text{Operating Income}}$ | Operational efficiency |
| **Portfolio at Risk (PAR30)** | $\frac{\text{Outstanding of Loans 30+ DPD}}{\text{Total Outstanding Portfolio}}$ | Proportion of portfolio showing stress |
| **Yield on Portfolio** | $\frac{\text{Total Interest Earned}}{\text{Average Portfolio Outstanding}}$ | Effective return on lending |
| **OCF/Disbursement Ratio** | $\frac{\text{Operating Cash Flow}}{\text{New Disbursements}}$ | Whether operations generate enough cash to fund new lending |

### OCF Metrics Computation

```python
def compute_ocf_metrics(portfolio_df: pd.DataFrame, period: str = 'monthly') -> dict:
    """Compute key OCF metrics for the lending portfolio."""
    
    metrics = {}
    
    # Collection Efficiency
    metrics['collection_efficiency'] = (
        portfolio_df['actual_collections'].sum() /
        portfolio_df['expected_collections'].sum() * 100
    )
    
    # Portfolio at Risk (PAR30)
    npa_outstanding = portfolio_df[
        portfolio_df['max_dpd'] >= 30
    ]['outstanding_amount'].sum()
    total_outstanding = portfolio_df['outstanding_amount'].sum()
    metrics['par_30'] = npa_outstanding / total_outstanding * 100
    
    # Net Interest Margin
    interest_income = portfolio_df['interest_earned'].sum()
    interest_expense = portfolio_df['cost_of_funds'].sum()
    avg_assets = portfolio_df['outstanding_amount'].mean()
    metrics['net_interest_margin'] = (
        (interest_income - interest_expense) / avg_assets * 100
    )
    
    # Cost to Income
    op_expenses = portfolio_df['operating_expenses'].sum()
    op_income = interest_income + portfolio_df['fee_income'].sum()
    metrics['cost_to_income'] = op_expenses / op_income * 100
    
    # OCF Ratio
    principal_received = portfolio_df['principal_repayments'].sum()
    provisions = portfolio_df['provision_amount'].sum()
    metrics['ocf'] = interest_income - op_expenses - provisions + principal_received
    metrics['ocf_to_disbursement'] = (
        metrics['ocf'] / portfolio_df['disbursement_amount'].sum() * 100
    )
    
    return metrics
```

### Why OCF Metrics Matter in Fintech

1. **Investor confidence** — OCF shows if the business generates real cash (not just accounting profits)
2. **Sustainability** — A lending company with negative OCF cannot sustain growth without external funding
3. **Default impact** — Rising defaults directly reduce OCF through lower collections and higher provisions
4. **Growth capacity** — Positive OCF/Disbursement ratio means the company can self-fund new loans

---

## 2.9 Power BI Storytelling Dashboards

### Dashboard Design Philosophy

The dashboards were designed for **non-technical executives** — board members, investors, and C-suite leaders who need actionable insights without data science jargon.

### Dashboard 1: Portfolio Health Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  AXIO PORTFOLIO HEALTH DASHBOARD                        [Monthly]  │
├─────────────────┬──────────────────┬────────────────────────────────┤
│  KPI CARDS      │                  │                                │
│  ┌───────────┐  │  OCF TREND       │  COLLECTION EFFICIENCY         │
│  │ Collection│  │  ▲               │  ┌────────────────────────┐    │
│  │ Efficiency│  │  │  ╱\    ╱\     │  │ 97%  ███████████████░  │    │
│  │   97.2%   │  │  │ ╱  \  ╱  \   │  │ Target: 95%            │    │
│  │   ▲ 1.3%  │  │  │╱    \/    \  │  └────────────────────────┘    │
│  └───────────┘  │  └──────────────┤                                │
│  ┌───────────┐  │  PAR30 TREND    │  DEFAULT RATE BY SEGMENT       │
│  │  PAR 30   │  │  ▼               │  ┌────────────────────────┐    │
│  │   4.2%    │  │  │\      /\     │  │ Low Risk:    2.1%      │    │
│  │   ▼ 0.8%  │  │  │ \    /  \   │  │ Medium Risk: 7.3%      │    │
│  └───────────┘  │  │  \  /    \  │  │ High Risk:   18.5%     │    │
│  ┌───────────┐  │  │   \/      \ │  │ V. High:     31.2%     │    │
│  │   NIM     │  │  └──────────────┤  └────────────────────────┘    │
│  │   8.7%    │  │                  │                                │
│  └───────────┘  │  RISK HEATMAP   │  CLUSTER MIGRATION             │
│                 │  (Geographic)    │  (Sankey diagram showing       │
│                 │  [India map]     │   customer segment movement)   │
├─────────────────┴──────────────────┴────────────────────────────────┤
│  EXECUTIVE SUMMARY: Portfolio quality improving. Collection        │
│  efficiency up 1.3pp MoM. PAR30 down due to tighter underwriting  │
│  criteria applied to Cluster 3 (high-risk segment).                │
└─────────────────────────────────────────────────────────────────────┘
```

### Dashboard 2: Fraud-Risk Segment Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│  FRAUD-RISK SEGMENT ANALYSIS                            [Monthly]  │
├──────────────────────────┬──────────────────────────────────────────┤
│  RISK SEGMENT BREAKDOWN  │  ADDRESS QUALITY vs DEFAULT RATE         │
│                          │                                          │
│  ██████████████░░ Low    │  Scatter plot:                           │
│  █████████░░░░░░ Medium  │  X: Address completeness score           │
│  ████░░░░░░░░░░░ High   │  Y: Default rate                         │
│  ██░░░░░░░░░░░░░ V.High │  Size: Loan volume                       │
│                          │  Color: City tier                        │
├──────────────────────────┼──────────────────────────────────────────┤
│  TOP RISK INDICATORS     │  MODEL PERFORMANCE OVER TIME             │
│                          │                                          │
│  1. Bounce rate > 15%    │  F1 Score: 0.95 (stable)                 │
│  2. EMI/Income > 0.6     │  Precision: 0.94                         │
│  3. Addr complete < 0.5  │  Recall: 0.96                            │
│  4. Cash withdrawal > 30%│                                          │
├──────────────────────────┴──────────────────────────────────────────┤
│  RECOMMENDATION: Apply stricter underwriting for applicants in      │
│  Cluster 3 (address completeness < 0.5, EMI/income > 0.6).        │
│  Estimated reduction in NPAs: 15-20%.                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Power BI Storytelling Principles Applied

| Principle | How I Applied It |
|-----------|-----------------|
| **Lead with the headline** | KPI cards at top with directional arrows (▲▼) and MoM change |
| **Context over numbers** | Every metric shown alongside targets and trends, not in isolation |
| **Segmentation** | Cluster-based views so executives see which customer groups need attention |
| **Actionable insights** | Each page ends with a plain-English recommendation |
| **Drill-down capability** | High-level → segment → individual loan level |
| **Consistent color coding** | Green = good/low risk, Red = bad/high risk across all visuals |
| **Executive summary box** | Bottom of each page has 2-3 sentence narrative explanation |

---

# 3. Key Metrics & Results

---

## 3.1 Model Performance

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **F1 Score** | **0.95** | Harmonic mean of precision (0.94) and recall (0.96) — strong balance |
| **Precision** | 0.94 | 94% of predicted defaults were actual defaults (low false positives) |
| **Recall** | 0.96 | 96% of actual defaults were caught (low false negatives) |
| **ROC-AUC** | 0.98 | Excellent discrimination between default and non-default |

### Why 95% F1 Matters in Default/Fraud Context

In lending, F1 is preferred over accuracy because of class imbalance (5–15% default rate). A naive "predict no default" model would get 90%+ accuracy but 0% F1 on the default class.

| Metric | Why It Matters for Lending |
|--------|---------------------------|
| **Precision** | High precision = fewer good customers rejected (less revenue loss from false positives) |
| **Recall** | High recall = fewer bad customers approved (less loss from defaults) |
| **F1** | Balances both — crucial when both types of errors have significant cost |

### Cost Matrix

| | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | ✅ True Negative — Correct approval | ⚠️ False Positive — Good customer rejected (lost revenue ~₹10K interest) |
| **Actual: Default** | ❌ False Negative — Bad loan approved (loss ~₹2L principal) | ✅ True Positive — Default prevented |

**Since a false negative costs ~20× more than a false positive**, recall is slightly more important than precision — our model's 0.96 recall vs 0.94 precision reflects this priority.

## 3.2 Business Impact

| Impact Area | Detail |
|-------------|--------|
| **Fraud-risk strategy** | Clustering + model insights directly informed which customer segments to tighten underwriting for, projected to reduce NPAs by 15–20% |
| **Address-based risk signals** | Address completeness became a standard feature in Axio's underwriting models going forward |
| **Lifestyle features** | Spending-pattern features (EMI/income, cash withdrawal ratio) added incremental 3–5pp F1 over bureau-only baseline |
| **Executive visibility** | OCF dashboards enabled non-tech executives to monitor portfolio health without analyst support |
| **Funding support** | Executive summaries and structured reports supported Axio's fundraising discussions with concrete data narratives |

---

# 4. Topics You Must Know

---

## 4.1 NLP for Address Parsing

### Key Techniques

| Technique | What It Does | When to Use |
|-----------|-------------|-------------|
| **Regex patterns** | Match structured patterns (pin codes, flat numbers, floor numbers) | Known formats with predictable structure |
| **Named Entity Recognition (NER)** | Identify entities like city names, states, landmarks from text | Unstructured text where rule-based fails |
| **Address standardization** | Normalize abbreviations, spelling variants, format inconsistencies | Cross-referencing addresses from different sources |
| **Fuzzy matching** | Match similar but not identical address strings | Deduplication, address verification |

### spaCy NER for Addresses

```python
import spacy

# Custom NER pipeline for Indian addresses
nlp = spacy.load("en_core_web_sm")

# Custom entity rules (EntityRuler)
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "CITY", "pattern": [{"LOWER": {"IN": ["bangalore", "mumbai", "delhi", "chennai"]}}]},
    {"label": "STATE", "pattern": [{"LOWER": {"IN": ["karnataka", "maharashtra", "tamil"]}, 
                                     "OP": "?"}, {"LOWER": "nadu", "OP": "?"}]},
    {"label": "PIN", "pattern": [{"SHAPE": "dddddd"}]},
]
ruler.add_patterns(patterns)
```

### Address Similarity (Fuzzy Matching)

```python
from fuzzywuzzy import fuzz

def address_similarity(addr1: str, addr2: str) -> float:
    """Compute similarity between two address strings."""
    return fuzz.token_sort_ratio(addr1.lower(), addr2.lower()) / 100.0

# Use case: Compare KYC address with bureau address
similarity = address_similarity(
    "Flat 302, Prestige Shantiniketan, Whitefield, Bangalore 560048",
    "302, Prestige Shanti Niketan, ITPL Road, Bengaluru - 560048"
)
# similarity ≈ 0.82 — high match despite formatting differences
```

---

## 4.2 Feature Engineering for Fintech

### Domain-Specific Feature Categories

| Category | Examples | Risk Logic |
|----------|---------|------------|
| **Bureau-derived** | Credit score, DPD history, utilization, enquiry count | Traditional credit risk indicators |
| **Income stability** | Salary CV, employer tenure, income growth | Ability to repay |
| **Debt burden** | EMI/income, total debt/income, number of active loans | Overleverage risk |
| **Behavioral** | Bounce rate, late-night transactions, spending patterns | Willingness to repay |
| **Address quality** | Completeness score, address consistency, city tier | Fraud and stability signals |
| **Transaction patterns** | Cash withdrawal ratio, UPI adoption, spending entropy | Lifestyle and modernity indicators |

### The Concept of Information Leakage in Feature Engineering

**Critical in credit risk:** Features must be computed using only information available at the time of decision. Using future data (like whether a customer bounced payments after loan disbursement to predict default) would constitute data leakage.

```python
# BAD: Using future information
df['bounce_rate_lifetime'] = df['total_bounces'] / df['total_payments']

# GOOD: Using only pre-disbursement information
df['bounce_rate_pre_loan'] = df['bounces_before_application'] / df['payments_before_application']
```

---

## 4.3 Clustering Algorithms

### K-Means vs DBSCAN

| Property | K-Means | DBSCAN |
|----------|---------|--------|
| **Requires K?** | Yes (specify number of clusters) | No (finds clusters automatically) |
| **Cluster shape** | Spherical / convex | Arbitrary shapes |
| **Handles noise?** | No (assigns every point) | Yes (labels outliers as noise) |
| **Scalability** | O(nKt) — very fast | O(n²) worst case — slower |
| **Sensitivity** | Sensitive to initialization and outliers | Sensitive to epsilon and min_samples |
| **Best for** | Well-separated, roughly equal-sized clusters | Clusters with varying density and shape |

**Why K-Means was chosen here:** Customer financial profiles tend to form reasonably convex clusters, the dataset was large (K-Means is faster), and we needed every customer assigned to a segment (no "noise" points).

### Elbow Method & Silhouette Score

- **Elbow method**: Plot inertia (within-cluster sum of squares) vs K. The "elbow" point where adding more clusters gives diminishing returns suggests optimal K.
- **Silhouette score**: Measures how similar each point is to its own cluster vs other clusters. Ranges from -1 to 1 (higher is better). Score > 0.5 indicates good separation.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where $a(i)$ = average distance to points in the same cluster, $b(i)$ = average distance to points in the nearest other cluster.

---

## 4.4 Classification Metrics Deep Dive

| Metric | Formula | When to Prioritize |
|--------|---------|-------------------|
| **Precision** | $\frac{TP}{TP + FP}$ | When false positives are costly (e.g., fraud investigation costs) |
| **Recall** | $\frac{TP}{TP + FN}$ | When false negatives are costly (e.g., approving a defaulter) |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | When both errors matter (balanced cost) |
| **F-beta** | $\frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$ | When you want to weight precision vs recall (β>1 favors recall) |
| **ROC-AUC** | Area under ROC curve | Overall model discrimination, threshold-independent |
| **PR-AUC** | Area under Precision-Recall curve | Better than ROC-AUC for imbalanced datasets |

### Threshold Optimization

```python
from sklearn.metrics import precision_recall_curve

def optimize_threshold(y_true, y_prob, beta=2.0):
    """Find threshold that maximizes F-beta score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    f_beta = ((1 + beta**2) * precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
    
    optimal_idx = np.argmax(f_beta)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"At this threshold: Precision={precisions[optimal_idx]:.3f}, "
          f"Recall={recalls[optimal_idx]:.3f}, F{beta}={f_beta[optimal_idx]:.3f}")
    
    return optimal_threshold
```

---

## 4.5 Feature Selection Techniques Summary

| Method | Type | Handles Non-Linear? | Assumptions | Best For |
|--------|------|---------------------|-------------|----------|
| **Correlation** | Filter | No | Linear relationships | Quick screening, removing redundancy |
| **Chi-squared** | Filter | No | Independence, categorical | Categorical feature significance |
| **Mutual Information** | Filter | Yes | None (non-parametric) | Capturing non-linear dependencies |
| **LASSO (L1)** | Embedded | No (linear model) | Linearity for coefficients | Automatic feature selection during training |
| **Random Forest Importance** | Embedded | Yes | None | General-purpose, handles interactions |
| **SHAP values** | Model-agnostic | Yes | None | Post-hoc explanation and selection |

---

## 4.6 Statistical Tests for Feature Significance

| Test | When to Use | Null Hypothesis |
|------|------------|-----------------|
| **Independent t-test** | Comparing means of two normal groups | μ₁ = μ₂ |
| **Welch's t-test** | Like t-test but unequal variances | μ₁ = μ₂ |
| **Mann-Whitney U** | Non-normal continuous data, two groups | Same distribution |
| **Chi-squared (independence)** | Two categorical variables | Variables are independent |
| **Kolmogorov-Smirnov** | Comparing two distributions | Same distribution |
| **Anderson-Darling** | Testing normality | Data is normally distributed |
| **ANOVA** | Comparing means of 3+ groups | All group means equal |
| **Kruskal-Wallis** | Non-parametric ANOVA alternative | Same distribution across groups |

---

## 4.7 Power BI Dashboard Best Practices

| Practice | Details |
|----------|---------|
| **Data modeling** | Star schema with fact and dimension tables; avoid many-to-many |
| **DAX measures** | Use CALCULATE, FILTER, SUMX for dynamic aggregations |
| **Storytelling** | Lead with insight → support with data → end with recommendation |
| **Interactivity** | Slicers for time period, segment, geography; cross-filtering between visuals |
| **Performance** | Limit data imported; use aggregations; avoid too many visuals per page |
| **Accessibility** | Color-blind friendly palettes; consistent formatting; clear labels |

---

## 4.8 Fraud Detection Approaches in Fintech

| Approach | Description | Pros | Cons |
|----------|------------|------|------|
| **Rule-based** | Hard-coded thresholds (e.g., >3 loans in 30 days) | Simple, explainable | Can't adapt, high false positives |
| **Anomaly detection** | Isolation Forest, Autoencoders for unusual patterns | Catches novel fraud | High false positive rate |
| **Supervised ML** | Train on labeled fraud/default data | High accuracy with good labels | Needs labeled data, concept drift |
| **Network analysis** | Graph-based fraud rings detection | Catches organized fraud | Complex to implement |
| **Behavioral biometrics** | Typing patterns, device fingerprinting | Hard to spoof | Privacy concerns |

---

## 4.9 JSON Processing in Python

```python
import json
import pandas as pd

# Flatten nested JSON
pd.json_normalize(data, sep='_')

# Handle missing keys safely
value = data.get('key', {}).get('nested_key', default_value)

# Parse large JSON files line by line (JSON Lines format)
import jsonlines
with jsonlines.open('large_file.jsonl') as reader:
    for record in reader:
        process(record)

# Schema validation with Pydantic
from pydantic import BaseModel, validator
class CustomerKYC(BaseModel):
    customer_id: str
    name: str
    addresses: list
    financial: dict
```

---

# 5. Interview Questions & Model Answers

---

### Behavioral / STAR Questions

---

#### Q1: "Walk me through this project."

**Answer (2-minute version):**

> "At Axio, formerly CapitalFloat — a digital lending fintech in Bangalore — I worked as a Research Intern on improving customer default prediction. The core challenge was that customer data, especially addresses from KYC onboarding, was stored as raw unstructured JSON with no standardized format.
>
> My first task was building a structured address dataset. I wrote a full NLP pipeline that parsed raw JSON files, extracted address text, and decomposed it into components — flat number, street, area, city, state, pin code — using regex patterns, rule-based NER, and address standardization. I handled Indian address quirks: abbreviations like 'Blore' for Bangalore, landmark-based directions, mixed Hindi transliterations.
>
> Next, I engineered lifestyle features from transaction data — spending ratios, purchase types, behavioral signals like bounce rates and cash withdrawal patterns. Then I applied a rigorous feature selection pipeline: correlation analysis, chi-squared tests for categorical features, mutual information for non-linear relationships, and LASSO regularization.
>
> I used K-Means clustering to segment customers into risk profiles — from stable low-risk to overleveraged high-risk. These cluster labels became powerful features in the default prediction model.
>
> The final model — XGBoost with SMOTE for class imbalance — achieved 95% F1 score. The address-derived features and lifestyle features added significant predictive lift over a bureau-only baseline.
>
> I also designed OCF metrics and built Power BI dashboards for non-technical executives, translating these analytics into plain-English stories about portfolio health. These dashboards supported Axio's fundraising and strategic planning discussions."

---

#### Q2: "Why was address data so important for default prediction?"

**Answer:**

> "Address data served multiple roles in our risk model:
>
> **Fraud signal**: Address completeness score — how much of the address a customer provided — correlated strongly with default. Customers who provide vague, incomplete addresses are more likely to be fraudulent or untraceable.
>
> **Geographic risk**: City tier matters in Indian lending. Metro customers (Bangalore, Mumbai) typically have better repayment records than smaller towns, partly due to income stability and employment diversity.
>
> **Consistency check**: I compared addresses across sources — KYC declaration vs bureau records. Mismatches flagged potential identity fraud.
>
> **Pin code enrichment**: Indian pin codes map to specific areas with known economic profiles. Pin code-level default rates became a powerful geographic risk feature.
>
> The address completeness score alone ranked in the top 6 most important features in our final model — which validated the effort spent building the address parser."

---

#### Q3: "How did you handle class imbalance in the default prediction model?"

**Answer:**

> "Our dataset had roughly a 10% default rate — moderate imbalance. I tackled this at multiple levels:
>
> **Data level**: Applied SMOTE (Synthetic Minority Oversampling Technique) to the training set only — never the validation or test set. SMOTE creates synthetic default cases by interpolating between existing default cases in feature space. I also tested ADASYN, which generates more synthetic samples near the decision boundary.
>
> **Algorithm level**: Used XGBoost's `scale_pos_weight` parameter, set to the ratio of non-defaulters to defaulters (~9:1). This tells the algorithm to penalize misclassifying defaults more heavily.
>
> **Evaluation level**: Optimized for F1 rather than accuracy. A model predicting 'no default' for everyone would get 90% accuracy but 0% F1 on the default class. I also tuned the classification threshold using precision-recall curves to maximize F-beta with β=2, weighting recall higher since missing a defaulter costs 20× more than wrongly flagging a good customer.
>
> **Validation level**: Used stratified K-fold cross-validation to maintain the class ratio in each fold."

---

#### Q4: "Explain how you used clustering for customer segmentation."

**Answer:**

> "I applied K-Means clustering on a combination of financial and behavioral features: income, EMI-to-income ratio, spending patterns, bounce rates, and address completeness.
>
> First, I standardized all features using StandardScaler, since K-Means is distance-based and sensitive to feature scales. Then I used the elbow method and silhouette scores to determine the optimal number of clusters — we settled on 4.
>
> Each cluster represented a distinct risk profile. For example:
> - Cluster 0: high income, low EMI ratio, high address completeness — our 'low risk' segment at 3% default rate.
> - Cluster 3: low income, high EMI ratio, frequent bounces, low address completeness — 'very high risk' at 35% default rate.
>
> These clusters were valuable in two ways: (1) as a feature in the default model — `cluster_default_rate` was a top-5 predictor because it captured multivariate risk patterns that individual features alone couldn't; (2) for business strategy — the risk team used these segments to tailor underwriting criteria and collection strategies."

---

#### Q5: "What feature selection techniques did you use and why multiple?"

**Answer:**

> "I used four complementary methods, each catching different things:
>
> **Correlation analysis** filters out redundant features — if two features are 90% correlated, we only need one. But correlation only catches linear relationships.
>
> **Chi-squared test** evaluates whether categorical features have a statistically significant association with default. It's useful for features like city tier or employment type.
>
> **Mutual information** captures non-linear dependencies that correlation misses. For example, a feature might have a U-shaped relationship with default — both very low and very high values indicating risk. Correlation would show zero, but mutual information would correctly identify this.
>
> **LASSO regularization** embeds selection into model training by shrinking unimportant coefficients to exactly zero. It handles multicollinearity well and provides a principled way to find the optimal sparsity level through cross-validated alpha selection.
>
> My final strategy was ensemble selection: keep features that were selected by 2 or more methods. This reduced the chance of excluding genuinely useful features while also controlling for overfitting from any single selection method."

---

#### Q6: "How did you design the Power BI dashboards for executives?"

**Answer:**

> "The key principle was storytelling over data dumping. Non-technical executives don't want 50 metrics on a screen — they want to know 'is the portfolio healthy?' and 'what should we do?'
>
> I structured each dashboard page around a single narrative:
>
> **Page 1: Portfolio Health** — KPI cards at the top showing collection efficiency, PAR30, and net interest margin with month-over-month directional arrows. Below that, trends over time and a segment breakdown. At the bottom, a 2-sentence executive summary in plain English.
>
> **Page 2: Risk Segments** — Cluster-based view showing which customer segments are growing or deteriorating. A scatter plot showing address quality vs default rate made the case for our address-based features visually.
>
> **Page 3: Drill-down** — For those who wanted to go deeper, individual loan-level data with filters for geography, product type, and time period.
>
> The dashboards were interactive — executives could click on a segment to filter the entire page. I used consistent color coding (green = good, red = alert) and ensured every number had context — either a target, a benchmark, or a trend to compare against."

---

#### Q7: "What is OCF and why did you design metrics around it?"

**Answer:**

> "OCF — Operating Cash Flow — measures the actual cash a business generates from operations. For a lending company like Axio, it's critical because revenues on paper (accrued interest) can look healthy even when customers aren't actually paying.
>
> I designed lending-specific OCF metrics: Collection Efficiency (how much of expected EMI payments are received), Portfolio at Risk (PAR30 — what percentage of outstanding loans are 30+ days past due), Net Interest Margin (spread between earning and cost of funds), and the OCF-to-Disbursement ratio (whether operations generate enough cash to fund new lending without external capital).
>
> These metrics were important for two audiences: (1) internal management — to spot early signs of portfolio stress before it hit the P&L; and (2) investors — Axio was in a growth phase, and investors wanted confidence that the lending model was fundamentally sound and cash-generative."

---

#### Q8: "How did you validate your model's performance?"

**Answer:**

> "Multiple layers of validation to ensure robustness:
>
> **Cross-validation**: 5-fold stratified CV to get a reliable estimate of F1 across different data splits. The low standard deviation (±0.01) confirmed stability.
>
> **Hold-out test set**: A 20% time-ordered hold-out (later dates) to simulate real deployment conditions. The test F1 of 0.95 matched CV, confirming no overfitting.
>
> **SMOTE only on training**: Critical — SMOTE was applied only within each training fold, never on validation or test data. Applying SMOTE to the full dataset before splitting would leak synthetic minority samples across the train-test boundary.
>
> **Feature importance sanity check**: I verified that the top features made domain sense. If a feature like 'customer_id_hash' ranked high, that would signal overfitting to specific customers rather than learning generalizable patterns.
>
> **Statistical tests on feature distributions**: KS tests confirmed that the feature distributions in train and test sets were not significantly different, meaning our split was representative."

---

#### Q9: "What challenges did you face with Indian address data specifically?"

**Answer:**

> "Several unique challenges:
>
> **No standard format**: Unlike US addresses with ZIP codes and standard city/state patterns, Indian addresses can include landmarks ('near Big Bazaar'), directional references ('behind the temple'), and mixed-language content.
>
> **Transliteration variations**: The same place name can be written multiple ways — 'Bangalore' vs 'Bengaluru' vs 'Blore' vs 'B'lore'. I built comprehensive correction dictionaries for major cities.
>
> **Pin code issues**: About 15-20% of addresses had missing or incorrect pin codes. I used the India Post pin code database for validation and enrichment — if city and state were provided, I could often infer or validate the pin code.
>
> **Abbreviation explosion**: Common abbreviations like 'Rd' for Road, 'Nr' for Near, 'Opp' for Opposite, 'H.No.' for House Number — I built regex-based abbreviation expansion as the first preprocessing step.
>
> **Multi-source addresses**: A single customer might have different address formats from Aadhaar (government ID), PAN card, self-declaration, and bureau records. Reconciling these was key to our consistency-based fraud signals."

---

#### Q10: "How did lifestyle features improve the model?"

**Answer:**

> "Lifestyle features added roughly 3-5 percentage points of F1 over a bureau-only baseline model. The most impactful ones were:
>
> **EMI-to-income ratio**: Customers spending over 60% of income on EMIs had a 5× higher default rate. This is a debt burden metric that bureau data captures partially, but our calculation from bank statements was more current and accurate.
>
> **Cash withdrawal ratio**: High cash dependency (>30% of income) correlated with higher default rates. This could indicate unreported income sources or financial opacity.
>
> **Spending entropy**: Customers with very concentrated spending (low entropy) — say, most transactions in a single category — showed different risk profiles from those with diversified spending.
>
> **Bounce rate**: Direct from bank statements, this was the single most predictive feature. Even one bounce in three months significantly increased default probability.
>
> The key insight was that lifestyle features capture the 'willingness to repay' dimension, while bureau features capture the 'ability to repay' dimension. Having both gives a much more complete picture."

---

#### Q11: "What was your approach to building the address parser? Why not use an existing geocoding API?"

**Answer:**

> "I considered external APIs like Google Geocoding and India-specific services, but there were three constraints:
>
> **Data privacy**: Customer KYC addresses are PII — sending them to external APIs raised data privacy and compliance concerns, especially for a fintech handling financial data.
>
> **Cost**: At scale (hundreds of thousands of records), API-based geocoding would be expensive and rate-limited.
>
> **Control and customization**: We needed not just geocoded coordinates but specific parsed components — completeness scores, consistency checks, fraud signals. No API directly provides 'address completeness as a risk metric.'
>
> So I built a custom parser using a layered approach: (1) regex for well-structured patterns like pin codes and flat numbers; (2) dictionary-based matching for city names, states, and abbreviations; (3) rule-based NER using spaCy's EntityRuler for entities not covered by dictionaries; and (4) pin code enrichment from the India Post database to fill gaps.
>
> The parser handled about 85% of addresses well. For the remaining 15% with very unusual formats, I flagged them for manual review and used whatever partial components could be extracted."

---

#### Q12: "How do you explain the model's predictions to business stakeholders?"

**Answer:**

> "At multiple levels:
>
> **Global level**: Feature importance plots showing which factors drive default prediction overall. I showed that bounce rate, EMI/income ratio, and bureau score are the top three — which aligns with domain intuition and builds trust.
>
> **Segment level**: Cluster profiles showing the characteristics of each risk segment. Executives could see that 'High Risk' customers have specific patterns: high EMI ratios, frequent bounces, low address completeness.
>
> **Individual level**: For specific loan decisions, SHAP values could decompose why a particular customer was flagged as high risk. For example: 'This customer's risk score is elevated because (1) their bounce rate is 3× the portfolio average, (2) their address completeness is below the 20th percentile, and (3) they have 4 recent credit enquiries.'
>
> The Power BI dashboards were the primary communication channel — designed specifically so that executives could self-serve insights without needing a data scientist to interpret results."

---

#### Q13: "If you had more time, what would you have done differently?"

**Answer:**

> "Three things:
>
> **Deep learning for addresses**: I used rule-based NLP, which worked well for Indian addresses at the time. But a sequence model — like a BiLSTM or Transformer fine-tuned on Indian addresses — could capture patterns that rules miss, especially for unusual address formats. The challenge was lack of labeled training data for Indian address parsing.
>
> **Graph-based fraud detection**: Customers connected by shared addresses, phone numbers, or devices can reveal fraud rings. I had the address data to build a customer graph but didn't have time to implement graph-based features (like network centrality or connected component size).
>
> **Real-time scoring**: Our model was batch-based. Moving to a real-time scoring API would enable instant credit decisions at the point of application, improving customer experience and reducing time-to-disbursement."

---

#### Q14: "How did you handle missing data in the feature engineering pipeline?"

**Answer:**

> "Strategy depended on the feature type and the reason for missingness:
>
> **Bureau features**: Missing bureau data typically meant a thin-file customer — someone with no or limited credit history. This is informative, so I created binary 'has_bureau_score' indicators rather than imputing. The missingness itself is a feature.
>
> **Transaction features**: Missing transaction data (e.g., no bank statement) was treated more seriously — it could mean the customer didn't consent to share data. I used median imputation within income segments, so a missing average balance for a ₹50K income customer was imputed from other ₹50K income customers.
>
> **Address features**: If parsing extracted some components but not others, I kept the partial extraction and let the completeness score capture the gap. A completely unparseable address got a completeness score of 0, which itself was a strong risk signal.
>
> **Key principle**: I never imputed the target variable, and I always applied imputation on the training set first, then applied the same imputation parameters to validation and test sets to prevent leakage."

---

#### Q15: "What statistical tests did you use and why?"

**Answer:**

> "The choice depended on data type and distribution:
>
> **Continuous features (normal)**: Welch's t-test to compare means between defaulters and non-defaulters. Welch's instead of Student's t-test because we couldn't assume equal variances.
>
> **Continuous features (non-normal)**: Mann-Whitney U test — a non-parametric alternative that compares rank distributions rather than means. Most financial features (income, spending) are right-skewed, so this was used more often.
>
> **Categorical features**: Chi-squared test of independence to check if the distribution of a categorical feature differs significantly between default and non-default groups. I complemented this with Cramér's V for effect size — statistical significance alone doesn't mean practical significance.
>
> **Distribution comparison**: Kolmogorov-Smirnov test to check if the entire distribution of a feature differs between groups, not just the mean. This catches cases where means are similar but distributions have different shapes.
>
> Every test was corrected for multiple comparisons using Bonferroni or Benjamini-Hochberg FDR to control the false discovery rate — with 50+ features, testing each at α=0.05 would give several false positives by chance."

---

#### Q16: "How would you monitor this model in production?"

**Answer:**

> "Three monitoring dimensions:
>
> **Input drift**: Track the distribution of each input feature over time using PSI (Population Stability Index). If address completeness scores suddenly shift — maybe due to a change in the onboarding form — that's a data drift signal. Threshold: PSI > 0.25 triggers a model review.
>
> **Output drift**: Monitor the distribution of predicted default probabilities. If the model starts predicting higher (or lower) risk across the board, something has changed — either in the customer population or in the feature pipeline.
>
> **Actual vs predicted**: This is the gold standard, but it takes time — you need to wait for loans to mature (6-12 months) to know actual default outcomes. Compare actual default rates per predicted risk decile against what the model expected. If they diverge, the model is losing calibration (concept drift).
>
> I'd automate all three checks with weekly/monthly cadence and build escalation rules: green (all stable), amber (investigate), red (model retraining needed)."

---

#### Q17: "What was the most impactful insight from this project?"

**Answer:**

> "The single most impactful insight was that **address quality is a proxy for borrower intent**. Customers who provided complete, verifiable addresses — with full pin code, proper building name, correct city-state mapping — had dramatically lower default rates than those with vague or incomplete addresses.
>
> This isn't just about geographic risk. It's a behavioral signal: a customer who takes the time to provide accurate address information is demonstrating good faith and traceability — they're not trying to hide. Conversely, someone who provides 'Near Big Bazaar, Bangalore' with no pin code or specific address might be harder to trace if they default.
>
> This insight led Axio to add address completeness checks earlier in the onboarding funnel — flagging incomplete addresses for additional verification before loan approval, rather than discovering the issue after disbursement."

---

#### Q18: "How did you ensure reproducibility of your results?"

**Answer:**

> "Several practices:
>
> **Random seeds**: Every stochastic operation — train/test split, SMOTE, K-Means initialization, XGBoost training — used fixed random seeds for reproducibility.
>
> **Pipeline versioning**: All feature engineering, selection, and modeling steps were codified in Python scripts with clear function interfaces — not ad-hoc notebook cells. This ensured the pipeline could be re-run end-to-end with consistent results.
>
> **Data snapshots**: I saved intermediate datasets at key stages (post-parsing, post-feature-engineering, post-selection) with timestamps, so any result could be traced back to specific input data.
>
> **Configuration files**: Model hyperparameters, feature lists, and threshold values were stored in config files rather than hardcoded, making it easy to audit what settings produced the reported results.
>
> **Documentation**: Each experiment was logged with the feature set used, model parameters, and performance metrics — creating an audit trail."

---

# 6. Red Flags & How to Handle

---

| Red Flag | Why It's Asked | How to Address |
|----------|---------------|----------------|
| **"Intern did all this in 6 months?"** | Seems like a lot for a research intern | *"This was a focused research project, not production deployment. I built the pipeline and proved the concept — senior team members handled production integration. The 6-month timeline was tight but scoped to research and prototyping."* |
| **"95% F1 seems too high for default prediction"** | Industry models typically show lower F1 | *"The 95% F1 was on a well-curated research dataset with clean labels, not a noisy production environment. I'd expect some degradation (3-5pp) in production due to data quality issues and concept drift. The key contribution was demonstrating that address and lifestyle features add significant lift over baseline."* |
| **"Did you deploy the model?"** | Intern may not have production experience | *"This was a research project — I built the pipeline, trained the model, and created the dashboards. Deployment would have involved the engineering team integrating the model into the loan origination system. I documented the model and pipeline thoroughly for handover."* |
| **"How large was the dataset?"** | Might be too small for reliability | *"We had approximately 50,000-100,000 customer records with labeled default outcomes — sufficient for the modeling approach. I validated with stratified 5-fold CV and a time-ordered hold-out to ensure results weren't artifact of a specific split."* |
| **"Why not use a pre-trained NER model?"** | Questioning technical choices | *"Pre-trained NER models (like spaCy's en_core_web_sm) are trained on Western text — they don't understand Indian address components like 'Mohalla', 'Distt', or landmark references. I evaluated them but found rule-based + dictionary approach worked better for Indian addresses. Fine-tuning a model would require labeled address data we didn't have."* |
| **"Why Power BI and not Tableau?"** | Tech stack choice | *"The organization had Power BI licenses and the executive team was already familiar with the interface. Tool choice was pragmatic — the storytelling principles (narrative structure, actionable insights, consistent formatting) are tool-agnostic."* |
| **"Weren't you just a research intern?"** | Questioning ownership | *"Yes, and research was exactly my scope — exploring whether address-derived and lifestyle features could improve default prediction. I had ownership of the research pipeline from data processing through model building to dashboard design. My mentor reviewed methodology and results, and the credit risk team provided domain guidance."* |

---

# 7. Key Takeaways

---

### For Interview Storytelling

1. **Lead with the problem**: "Customer address data was unstructured JSON — no one could use it for risk modeling."
2. **Quantify impact**: "95% F1 score; address features ranked in top 6 most important predictors."
3. **Show breadth**: This project touches NLP, feature engineering, statistical analysis, clustering, classification, dashboard design — demonstrate range.
4. **Show business sense**: "OCF dashboards helped non-tech executives understand portfolio health, supporting fundraising conversations."
5. **Be honest about scope**: "This was a research internship — I proved the concept, the engineering team would handle production."

### Technical Talking Points

| Topic | Key Point to Make |
|-------|-------------------|
| **Address parsing** | Rule-based NLP was the right choice for Indian addresses given lack of labeled training data |
| **Feature engineering** | Lifestyle features capture "willingness to repay"; bureau features capture "ability to repay" |
| **Feature selection** | Ensemble of 4 methods (correlation, chi-squared, MI, LASSO) for robust selection |
| **Clustering** | K-Means segments became features themselves — `cluster_default_rate` was a top-5 predictor |
| **Class imbalance** | SMOTE only on training data; threshold optimization with F-beta; cost-sensitive learning |
| **Model choice** | XGBoost balanced performance (95% F1) with reasonable interpretability (SHAP) |
| **Statistical rigor** | Every feature validated with appropriate hypothesis tests; multiple comparison correction |
| **Dashboard design** | Storytelling > data dumping; every metric needs context (target, trend, benchmark) |

### One-Liner Summary

> *"I built a structured address dataset from raw JSON using NLP, engineered lifestyle features, applied statistical evaluation and clustering, and trained a default prediction model achieving 95% F1 — informing Axio's fraud-risk strategy and enabling executive decision-making through Power BI dashboards."*

---

*Document prepared for Rahul Sharma — Axio (CapitalFloat) Research Internship Interview Preparation*
