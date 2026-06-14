# GenAI for Data Scientists — The New DS Toolkit (2025-2026)

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist in the AI Era — How LLMs Transform Traditional DS Workflows

> **Why this matters:** The DS role is evolving rapidly. Companies now expect data scientists to leverage LLMs for productivity, understand foundation models beyond NLP, and know when AI-native solutions replace traditional approaches. This file covers what's **new and different** — how AI changes the DS job itself.

---

# Table of Contents

1. [How the DS Role Is Changing](#1-how-the-ds-role-is-changing)
2. [LLM-Augmented Data Science Workflows](#2-llm-augmented-data-science-workflows)
3. [Foundation Models for Tabular Data](#3-foundation-models-for-tabular-data)
4. [Synthetic Data Generation](#4-synthetic-data-generation)
5. [Text-to-SQL and Natural Language Analytics](#5-text-to-sql-and-natural-language-analytics)
6. [AI-Pair Programming for DS](#6-ai-pair-programming-for-ds)
7. [AutoML in the LLM Era](#7-automl-in-the-llm-era)
8. [LLM-as-Judge for Evaluation](#8-llm-as-judge-for-evaluation)
9. [Multimodal Data Science](#9-multimodal-data-science)
10. [The Modern DS Tech Stack (2026)](#10-the-modern-ds-tech-stack-2026)
11. [Interview Questions with Strong Answers](#11-interview-questions-with-strong-answers)

---

# **1. How the DS Role Is Changing**

## **1.1 The Shift**

```
┌──────────────────────────────────────────────────────────────────┐
│          DATA SCIENTIST ROLE EVOLUTION                             │
│                                                                   │
│  PRE-LLM DS (2018-2022):              AI-ERA DS (2024-2026):    │
│                                                                   │
│  • Write SQL from scratch              • LLM drafts SQL, you    │
│  • Manual EDA with matplotlib            validate and refine     │
│  • Feature engineering by hand         • Conversational EDA      │
│  • Train models from scratch           • LLM suggests features   │
│  • Write reports manually              • Foundation models as    │
│  • Mostly tabular/structured data        strong baselines        │
│                                        • Auto-generated reports  │
│                                        • Multimodal data (text + │
│                                          images + tabular)       │
│                                                                   │
│  WHAT STAYS THE SAME:                                            │
│  • Statistical thinking                                          │
│  • Causal reasoning                                              │
│  • Business understanding                                        │
│  • Experiment design                                             │
│  • Model evaluation and validation                               │
│  • Communication of insights                                     │
│  • Knowing WHEN and WHY, not just HOW                           │
│                                                                   │
│  WHAT'S NEW:                                                     │
│  • LLMs as productivity multipliers                              │
│  • Foundation models as baselines                                │
│  • Prompt engineering as a core skill                            │
│  • Multimodal reasoning                                          │
│  • Synthetic data generation                                     │
│  • AI-augmented experimentation                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **1.2 The 80/20 Rule Has Shifted**

Old reality: 80% data prep, 20% modeling.
New reality: **LLMs accelerate the 80%** (data cleaning, EDA, feature engineering) so you can spend more time on what matters — **problem framing, experiment design, model validation, and business impact.**

## **1.3 What LLMs Can't Replace (Your Durable Skills)**

| Skill | Why LLMs Can't Do It |
|-------|---------------------|
| **Causal thinking** | LLMs confuse correlation with causation, can't design valid causal studies |
| **Problem framing** | Knowing WHICH question to ask requires domain + business judgment |
| **Experiment design** | A/B test design requires understanding of SUTVA, power, practical significance |
| **Stakeholder communication** | Understanding what the VP cares about, translating to action |
| **Ethical judgment** | Fairness, privacy, "should we even build this?" |
| **Validation** | Knowing when a model is wrong, understanding failure modes |

---

# **2. LLM-Augmented Data Science Workflows**

## **2.1 EDA with LLMs**

Instead of writing 50 lines of pandas + matplotlib, describe what you want:

```
Traditional:
df.describe() → df.corr() → df.hist() → manual interpretation → 45 min

LLM-Augmented:
"Analyze this dataset. Show distribution of each feature, correlation matrix,
identify potential target leakage, suggest features that may need transformation,
and flag any data quality issues."
→ Comprehensive EDA in 5 min → You validate and dive deeper where needed
```

**Tools:** ChatGPT Code Interpreter, Claude Artifacts, Cursor + Copilot, Jupyter AI, pandas-ai

## **2.2 Feature Engineering with LLMs**

```
Prompt: "I'm building a credit default prediction model. My features are:
income, employment_length, loan_amount, interest_rate, home_ownership,
annual_income, dti_ratio, open_accounts, total_accounts.

Suggest 10 engineered features with rationale for each."

LLM suggests:
1. loan_to_income_ratio = loan_amount / annual_income (leverage indicator)
2. payment_to_income = (loan_amount * interest_rate/12) / (annual_income/12)
3. credit_utilization_proxy = open_accounts / total_accounts
4. income_stability_proxy = employment_length * annual_income
... (domain-informed suggestions you validate)
```

**You still need to:** Validate features statistically, check for leakage, evaluate importance after training.

## **2.3 Code Generation for DS**

LLMs excel at:
- Writing boilerplate (data loading, preprocessing pipelines)
- Converting analysis descriptions to pandas/SQL code
- Generating visualization code from specifications
- Writing unit tests for data pipelines
- Translating between frameworks (pandas ↔ PySpark ↔ SQL)

LLMs struggle with:
- Complex multi-step statistical reasoning
- Novel algorithm implementation (they repeat patterns from training data)
- Knowing when code is WRONG (you must validate)

---

# **3. Foundation Models for Tabular Data**

## **3.1 The Big Question: Can LLMs Beat XGBoost on Tables?**

For years, XGBoost/LightGBM dominated tabular ML. Now foundation models are entering this space:

| Model | Approach | Key Claim |
|-------|---------|-----------|
| **TabPFN** (2022) | Prior-fitted network; trained on synthetic tabular datasets, does inference in a single forward pass | Competitive with tuned XGBoost on small datasets (< 10K rows), requires NO training |
| **CARTE** (2024) | Pre-trained on many tabular datasets, fine-tuned to new task | Better generalization to new domains |
| **TabLLM** | Serialize tabular data as text, use LLM for classification | Works but generally slower and less accurate than tree methods |
| **UniPredict** | Universal tabular prediction across datasets | Zero-shot tabular prediction |

## **3.2 When Do They Win?**

| Scenario | Foundation Model | XGBoost/LightGBM |
|----------|-----------------|-------------------|
| **< 1000 rows** | Often wins (trained on many datasets, generalizes from few examples) | Overfits with limited data |
| **1K-100K rows** | Competitive | Usually wins with tuning |
| **> 100K rows** | Slower, often worse | Clear winner (efficient, well-understood) |
| **Zero-shot (no training data for this specific task)** | Only option | Not possible |
| **Cold start** | Handles gracefully | Needs training data |

## **3.3 Practical Takeaway**

> "Foundation models for tabular data are the new baseline for small datasets and cold-start scenarios. For large datasets with enough training data, XGBoost/LightGBM still dominate. The smart approach: try TabPFN as a zero-shot baseline, then train XGBoost if you have enough data and need the best accuracy."

---

# **4. Synthetic Data Generation**

## **4.1 Why Synthetic Data?**

| Use Case | Problem | Synthetic Data Solution |
|----------|---------|----------------------|
| **Privacy** | Can't share real patient data | Generate synthetic patient records with same statistical properties |
| **Class imbalance** | 0.1% fraud rate, not enough fraud examples | Generate realistic synthetic fraud cases |
| **Data augmentation** | Small training set | Augment with synthetic examples |
| **Testing** | Need test data for development | Generate realistic test fixtures |
| **Bias mitigation** | Training data underrepresents a group | Generate balanced synthetic data |

## **4.2 Methods**

| Method | How | Best For |
|--------|-----|----------|
| **SMOTE / ADASYN** | Interpolate between existing minority samples | Simple tabular oversampling |
| **Gaussian Copula** | Model feature distributions and correlations | Preserving statistical relationships |
| **CTGAN** (Conditional Tabular GAN) | GAN trained on tabular data | Complex distributions, mixed types |
| **VAE-based** | Variational autoencoder generates samples | Smooth latent space, good diversity |
| **LLM-based** | Describe the data distribution, LLM generates records | When you need domain-realistic records with context |
| **Diffusion models** | TabDDPM — denoising diffusion for tabular | State-of-the-art quality, handles mixed types |

## **4.3 Validation of Synthetic Data**

Always validate that synthetic data preserves the properties you care about:

| Check | How |
|-------|-----|
| **Statistical fidelity** | Compare marginal distributions, correlations, summary stats |
| **ML utility** | Train model on synthetic, test on real — compare to train-on-real performance |
| **Privacy** | Distance to nearest real record (should not be too close) |
| **Diversity** | Synthetic shouldn't just copy training data |

## **4.4 LLMs for Synthetic Data**

```
Prompt: "Generate 10 synthetic customer support tickets for an airline company.
Each should have: ticket_id, category (billing/technical/complaint/inquiry),
priority (high/medium/low), description (1-2 sentences), and resolution_time_hours.

Make them realistic with varied language, realistic resolution times,
and realistic category distribution."
```

LLMs generate contextually rich synthetic text data that traditional methods can't — product reviews, support tickets, medical notes, user feedback.

---

# **5. Text-to-SQL and Natural Language Analytics**

## **5.1 What Is Text-to-SQL?**

Convert natural language questions into SQL queries:

```
User: "What were the top 5 products by revenue last quarter?"

LLM generates:
SELECT product_name, SUM(revenue) as total_revenue
FROM sales
WHERE sale_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
  AND sale_date < DATE_TRUNC('quarter', CURRENT_DATE)
GROUP BY product_name
ORDER BY total_revenue DESC
LIMIT 5;
```

## **5.2 Challenges**

| Challenge | Why It's Hard |
|-----------|-------------|
| **Schema understanding** | LLM needs to know table names, columns, relationships, types |
| **Ambiguity** | "Last quarter" — calendar quarter? fiscal quarter? Last 3 months? |
| **Complex queries** | Multi-table joins, window functions, CTEs are error-prone |
| **Security** | SQL injection risk, access control (user shouldn't query all tables) |
| **Validation** | How do you know the generated SQL is correct? |

## **5.3 Production Architecture**

```
┌──────────────────────────────────────────────────────────────────┐
│            TEXT-TO-SQL PIPELINE                                     │
│                                                                   │
│  User Question                                                    │
│       │                                                          │
│       ▼                                                          │
│  SCHEMA RETRIEVAL                                                │
│  └── Retrieve relevant table schemas (not all tables)            │
│      Use embeddings to match question to relevant tables         │
│       │                                                          │
│       ▼                                                          │
│  SQL GENERATION                                                  │
│  └── LLM generates SQL given question + relevant schemas        │
│       │                                                          │
│       ▼                                                          │
│  VALIDATION                                                      │
│  └── Parse SQL, check syntax, verify table/column names exist   │
│      Check for dangerous operations (DROP, DELETE, UPDATE)       │
│       │                                                          │
│       ▼                                                          │
│  EXECUTION                                                       │
│  └── Run against read-only replica with timeout and row limit   │
│       │                                                          │
│       ▼                                                          │
│  ANSWER GENERATION                                               │
│  └── LLM interprets results and generates natural language      │
│      answer with data visualization suggestions                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **5.4 Tools**

| Tool | What It Does |
|------|-------------|
| **Vanna.ai** | Open-source text-to-SQL with RAG over your schema |
| **DuckDB + LLM** | Local analytics with LLM query generation |
| **LangChain SQL Agent** | LLM agent that iteratively queries and analyzes databases |
| **Databricks Genie** | Enterprise text-to-SQL for Databricks lakehouses |

---

# **6. AI-Pair Programming for DS**

## **6.1 How to Use LLMs Effectively in DS Work**

| Task | LLM Role | Your Role |
|------|----------|-----------|
| **Data loading** | Write boilerplate code | Specify format, schema, edge cases |
| **EDA** | Generate exploration code, suggest analyses | Validate findings, decide what matters |
| **Feature engineering** | Suggest domain features, implement transforms | Evaluate features, check for leakage |
| **Model training** | Generate training pipeline code | Choose model, set up validation, tune |
| **Visualization** | Generate plot code from description | Interpret visualizations, tell the story |
| **Report writing** | Draft findings, format tables | Validate claims, add nuance, communicate |
| **Documentation** | Generate docstrings, README content | Review for accuracy |

## **6.2 Anti-Patterns (What NOT to Do)**

| Anti-Pattern | Risk |
|-------------|------|
| **Trusting LLM output without validation** | Wrong analysis → wrong business decision |
| **Copy-pasting code without understanding** | Can't debug, can't explain to stakeholders |
| **Using LLM for novel statistical analysis** | LLMs repeat common patterns; may not apply correctly to your specific data |
| **Skipping the "think" step** | LLMs give you speed but not judgment |
| **Over-relying on LLM for causal claims** | LLMs confuse correlation with causation routinely |

## **6.3 The "LLM + Human" Workflow**

```
1. THINK: What question am I trying to answer? (Human)
2. GENERATE: Ask LLM to write the analysis code (LLM)
3. VALIDATE: Review code for correctness, run it, check results (Human)
4. INTERPRET: What does this mean for the business? (Human)
5. ITERATE: Ask LLM to refine based on findings (LLM)
6. COMMUNICATE: Draft with LLM help, finalize with judgment (Human)
```

---

# **7. AutoML in the LLM Era**

## **7.1 AutoML Tools**

| Tool | What It Automates | When to Use |
|------|------------------|-------------|
| **AutoGluon** (Amazon) | Model selection, stacking, hyperparameter tuning | Fast competitive baseline, tabular/multimodal |
| **H2O AutoML** | End-to-end pipeline (prep → train → tune → explain) | Enterprise environments |
| **PyCaret** | Low-code ML pipeline | Quick prototyping, comparison |
| **FLAML** (Microsoft) | Cost-effective hyperparameter tuning | When compute budget is limited |
| **Google Vertex AI AutoML** | Managed AutoML for tabular, vision, NLP | GCP-native, production-ready |

## **7.2 AutoML + LLMs**

The new pattern: **LLM selects what to try, AutoML runs it, LLM interprets results.**

```
User: "Build the best model for this credit default dataset."

LLM Agent:
1. Analyzes dataset (types, distributions, target balance)
2. Recommends: "Given class imbalance (3% positive), I'll try
   SMOTE + LightGBM, XGBoost with scale_pos_weight, and
   balanced random forest. Metric: PR-AUC (not ROC-AUC, because
   of imbalance)."
3. Generates AutoGluon/PyCaret code
4. Runs experiments
5. Interprets results: "LightGBM + SMOTE achieved 0.42 PR-AUC.
   Top features: debt_ratio, payment_history, credit_utilization."
6. Suggests next steps: "Try feature interactions, or collect
   more data on the underrepresented segment."
```

## **7.3 When to Use AutoML vs Manual**

| Scenario | AutoML | Manual |
|----------|--------|--------|
| Quick baseline | ✓ | Overkill |
| Hackathon / competition | ✓ (fast iteration) | After AutoML plateaus |
| Production model with SLAs | Start with AutoML, refine manually | ✓ (need control over every choice) |
| Regulated domain (explainability) | Risky (may pick black-box ensemble) | ✓ (choose interpretable model) |
| Novel problem | AutoML as exploration | ✓ (domain expertise guides choices) |

---

# **8. LLM-as-Judge for Evaluation**

## **8.1 What Is LLM-as-Judge?**

Use an LLM to evaluate the quality of outputs (from another LLM, or from any system):

| Application | How |
|-------------|-----|
| **Text quality** | "Rate this summary on a scale of 1-5 for accuracy, relevance, and clarity" |
| **Classification audit** | "Does this label match the text? Is the sentiment correct?" |
| **Data quality** | "Is this record realistic? Does this address look valid?" |
| **Model comparison** | "Which of these two responses is better for the given question?" |

## **8.2 When It Works / Doesn't**

| Works Well | Doesn't Work Well |
|-----------|------------------|
| Subjective quality assessment (writing, summaries) | Factual accuracy (LLM might agree with wrong facts) |
| Comparing two options (pairwise preference) | Precise numerical evaluation |
| Screening large datasets for quality | High-stakes regulatory evaluation |
| Rapid iteration on prompt quality | Replacing rigorous statistical validation |

## **8.3 Best Practices**

- Use **pairwise comparison** (which is better: A or B?) rather than absolute scoring (rate 1-5) — more reliable
- Use **rubrics** with specific criteria — don't just ask "is this good?"
- **Calibrate** against human judgments on a sample
- Be aware of **position bias** (LLMs tend to prefer the first option) — randomize order

---

# **9. Multimodal Data Science**

## **9.1 The Multimodal Shift**

Traditional DS: tabular data → features → model → prediction.
Modern DS: tabular + text + images + time series → unified model → richer predictions.

| Data Type Combination | Example | Approach |
|----------------------|---------|----------|
| **Tabular + Text** | Customer features + support ticket text → churn prediction | Embed text with LLM, concatenate with tabular features |
| **Tabular + Images** | Product metadata + product photos → price prediction | CLIP image embeddings + tabular features |
| **Text + Images** | Medical notes + X-rays → diagnosis | Vision-language models (LLaVA, GPT-4V) |
| **Tabular + Time Series** | Customer demographics + purchase history → LTV | Temporal features + static features in unified model |

## **9.2 Your Multimodal Experience**

- **Fashion RecSys:** CLIP (vision + text) + BLIP captions + segmentation masks → multimodal retrieval
- **Radiology:** LLaVA-NeXT → image + text → structured report generation
- **Jet2:** Text (call transcripts) + tabular (customer data) + time series (call volumes)

This is a differentiator — most candidates only have tabular experience.

---

# **10. The Modern DS Tech Stack (2026)**

```
┌──────────────────────────────────────────────────────────────────┐
│          MODERN DATA SCIENTIST TECH STACK                         │
│                                                                   │
│  ANALYSIS & EDA                                                  │
│  ├── Python (pandas, polars)                                     │
│  ├── SQL (advanced)                                              │
│  ├── Jupyter + LLM assistance (Cursor, Copilot)                 │
│  └── DuckDB (local analytics)                                   │
│                                                                   │
│  ML & MODELING                                                   │
│  ├── XGBoost / LightGBM (still king for tabular)               │
│  ├── PyTorch / Transformers (deep learning)                     │
│  ├── AutoGluon / PyCaret (quick baselines)                      │
│  ├── Foundation models (TabPFN, Chronos for zero-shot)          │
│  └── scikit-learn (classical ML)                                │
│                                                                   │
│  LLM & GENAI                                                    │
│  ├── OpenAI / Anthropic / Gemini APIs                           │
│  ├── LangChain / LangGraph (agent orchestration)                │
│  ├── Hugging Face Transformers (local models)                   │
│  └── vLLM / Ollama (local inference)                            │
│                                                                   │
│  DATA INFRASTRUCTURE                                             │
│  ├── Snowflake / BigQuery / Databricks (warehouse)              │
│  ├── Airflow / Dagster (orchestration)                          │
│  ├── dbt (transform)                                            │
│  ├── Great Expectations (data quality)                          │
│  └── Feature stores (Feast, Tecton)                             │
│                                                                   │
│  MLOPS & DEPLOYMENT                                              │
│  ├── MLflow (experiment tracking, model registry)               │
│  ├── Docker + Kubernetes (serving)                              │
│  ├── FastAPI (model APIs)                                       │
│  ├── FAISS / pgvector (vector search)                           │
│  └── Weights & Biases (experiment tracking)                     │
│                                                                   │
│  PRODUCTIVITY                                                    │
│  ├── Cursor / GitHub Copilot (AI pair programming)              │
│  ├── ChatGPT / Claude (analysis, writing, debugging)            │
│  └── Notion AI / Gamma (documentation, presentations)           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

# **11. Interview Questions with Strong Answers**

---

## **Q1: "How has GenAI changed the data scientist's role?"**

> GenAI has been a **productivity multiplier** for the mechanical parts of DS — EDA, code writing, feature engineering, report drafting — while making the **strategic parts even more important**.
>
> Specifically: I can now generate an EDA notebook with LLM assistance in 15 minutes instead of 2 hours. Feature engineering benefits from LLM suggestions grounded in domain knowledge. Boilerplate code (data loading, preprocessing, model training pipelines) is largely generated.
>
> But what GenAI CAN'T do is: frame the right business question, design a valid experiment (causal reasoning), validate that a model is appropriate for a specific context, or communicate insights that drive action. These "judgment" skills are more valuable than ever because the technical barrier is lower — the differentiation is in thinking, not coding.
>
> The role has shifted from "person who writes Python to analyze data" to "person who uses AI-augmented tools to rapidly generate and validate data-driven insights that inform business decisions."

---

## **Q2: "Can LLMs replace XGBoost for tabular prediction?"**

> Not yet, for most practical scenarios. XGBoost/LightGBM remain the best choice for tabular data with > 1000 training examples. They're fast to train, easy to interpret, and well-understood.
>
> However, **foundation models for tabular data** (TabPFN, CARTE) are becoming competitive for **small datasets** (< 1000 rows) and **zero-shot scenarios** (predicting on a new task without task-specific training). This is genuinely new — previously, small datasets meant poor models.
>
> My practical approach: start with TabPFN as a zero-shot baseline (takes seconds, no training), then train XGBoost if I have enough data. If XGBoost significantly outperforms, use it. If TabPFN is close, the development time savings might justify keeping it.
>
> The future is likely **hybrid**: foundation models for initial exploration and small-data scenarios, tree-based models for production with sufficient data, and LLMs for feature extraction from unstructured columns (text, descriptions).

---

## **Q3: "How would you use synthetic data in a real project?"**

> At Fibe, we had severe class imbalance in the credit default model (< 3% default rate). I'd use synthetic data to:
>
> 1. **Augment minority class:** Generate realistic synthetic default cases using CTGAN (trained on real defaults), increasing the effective training size for the minority class. Validate that synthetic defaults have similar feature distributions to real ones.
>
> 2. **Privacy-compliant sharing:** If regulators or external auditors need to validate the model but can't see real customer data, I'd generate a synthetic dataset with matching statistical properties (using Gaussian Copula) and demonstrate model performance on it.
>
> 3. **Testing and development:** Generate synthetic test data for CI/CD pipeline validation so developers can test model serving without real customer data.
>
> **Critical validation:** I'd always measure ML utility — train on synthetic, test on real. If the model trained on synthetic data performs within 5% of the model trained on real data, the synthetic data is useful. I'd also check privacy (nearest-neighbor distance to real records must be above a threshold).

---

## **Q4: "Walk me through building a text-to-SQL system for a business analytics team."**

> **Architecture:** The business analyst types a natural language question → the system generates SQL → executes it → returns results with a natural language explanation.
>
> **Schema handling:** I'd use RAG over the database schema. Embed table/column descriptions and store in a vector DB. When a question comes in, retrieve the most relevant tables (not all — a large database might have hundreds of tables). Include column descriptions, sample values, and known join relationships in the prompt.
>
> **SQL generation:** Pass the question + relevant schema to an LLM. Use few-shot examples of (question, SQL) pairs specific to this company's data model. Include common gotchas ("fiscal year starts in April," "revenue table excludes refunds").
>
> **Validation layer:** Parse the SQL to check syntax, verify all referenced tables and columns exist, block dangerous operations (DELETE, UPDATE, DROP). Execute against a read-only replica with a timeout (30s) and row limit (10K).
>
> **Answer generation:** Feed the SQL results back to the LLM to generate a natural language answer with the key insight, not just raw numbers.
>
> **Iteration:** Log all queries and flag low-confidence ones for human SQL review. Use feedback to improve few-shot examples and schema descriptions over time.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│      GENAI FOR DATA SCIENTISTS TAKEAWAYS                          │
│                                                                   │
│  1. LLMs ACCELERATE the mechanical DS work (EDA, code,          │
│     features, reports) but DON'T REPLACE judgment                │
│                                                                   │
│  2. DURABLE SKILLS: causal thinking, experiment design,          │
│     problem framing, stakeholder communication                   │
│                                                                   │
│  3. XGBoost STILL WINS on tabular data with sufficient rows     │
│     Foundation models win on SMALL data and ZERO-SHOT            │
│                                                                   │
│  4. SYNTHETIC DATA: use for privacy, class imbalance, testing   │
│     Always validate: statistical fidelity + ML utility + privacy │
│                                                                   │
│  5. TEXT-TO-SQL is real but needs validation layer               │
│     (schema RAG + SQL parsing + read-only execution)             │
│                                                                   │
│  6. AI PAIR PROGRAMMING: LLM generates, YOU validate            │
│     Never trust LLM output without checking                     │
│                                                                   │
│  7. LLM-AS-JUDGE: useful for subjective evaluation at scale,   │
│     but calibrate against human judgments                        │
│                                                                   │
│  8. MULTIMODAL is the new frontier: tabular + text + images     │
│     Your CLIP and LLaVA experience is a differentiator          │
│                                                                   │
│  9. The modern DS uses LLMs as TOOLS, not as REPLACEMENTS      │
│     for statistical and business reasoning                       │
└──────────────────────────────────────────────────────────────────┘
```
