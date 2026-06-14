# System Design for ML — Comprehensive Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland (4.0 GPA)
**Focus:** Data Scientist / ML Engineer — Production ML Systems, LLMs, RAG, Recommendation Systems, Time Series
**Deployed Projects:** PathWise (FastAPI + Supabase + LLM agents), Fashion Reco (multimodal CLIP), Medical RAG (CheXagent + retrieval), Tesla Forecasting (XGBoost + LSTM + GARCH)
**Document Scope:** End-to-end ML system design — frameworks, architecture patterns, common design questions, scalability, data pipelines, trade-offs, and interview strategy

---

## Table of Contents

1. [ML System Design Framework](#1-ml-system-design-framework)
2. [Common ML System Design Questions](#2-common-ml-system-design-questions)
   - [2a. Design a Recommendation System](#2a-design-a-recommendation-system)
   - [2b. Design a Search/Retrieval System](#2b-design-a-searchretrieval-system)
   - [2c. Design a RAG-Based QA System](#2c-design-a-rag-based-qa-system)
   - [2d. Design a Real-Time Fraud Detection System](#2d-design-a-real-time-fraud-detection-system)
   - [2e. Design a Text Classification System](#2e-design-a-text-classification-system)
   - [2f. Design an LLM-Powered Chatbot](#2f-design-an-llm-powered-chatbot)
   - [2g. Design a Time Series Forecasting System](#2g-design-a-time-series-forecasting-system)
3. [Scalability Patterns](#3-scalability-patterns)
4. [Data Pipeline Design](#4-data-pipeline-design)
5. [Trade-Offs to Discuss](#5-trade-offs-to-discuss)
6. [Monitoring, Observability, and Iteration](#6-monitoring-observability-and-iteration)
7. [Interview Tips and Communication Strategy](#7-interview-tips-and-communication-strategy)
8. [Common Interview Questions with Frameworks for Answering](#8-common-interview-questions-with-frameworks-for-answering)
9. [Key Takeaways](#9-key-takeaways)

---

# 1. ML System Design Framework

---

Every ML system design interview follows a structured progression. Interviewers evaluate your ability to think end-to-end — from ambiguous business requirements to production-ready architecture. The framework below is your skeleton for *every* ML design question.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     ML SYSTEM DESIGN — 6-STEP FRAMEWORK                         │
│                                                                                  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                        │
│   │  STEP 1      │──▶│  STEP 2      │──▶│  STEP 3      │                        │
│   │  Clarify     │   │  Define      │   │  Data        │                        │
│   │  Requirements│   │  Metrics     │   │  Pipeline    │                        │
│   └──────────────┘   └──────────────┘   └──────┬───────┘                        │
│                                                 │                                │
│   ┌──────────────┐   ┌──────────────┐   ┌──────▼───────┐                        │
│   │  STEP 6      │◀──│  STEP 5      │◀──│  STEP 4      │                        │
│   │  Monitor &   │   │  Serving &   │   │  Model       │                        │
│   │  Iterate     │   │  Inference   │   │  Selection   │                        │
│   └──────────────┘   └──────────────┘   └──────────────┘                        │
│                                                                                  │
│   Time allocation (45 min interview):                                            │
│   Step 1: 5 min │ Step 2: 5 min │ Step 3: 10 min │                              │
│   Step 4: 10 min │ Step 5: 10 min │ Step 6: 5 min                               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Clarify Requirements (5 minutes)

**Goal:** Convert an ambiguous prompt into a scoped design problem. This is where most candidates fail — they jump straight to models.

### Functional Requirements

These describe *what* the system does:

| Question to Ask | Why It Matters |
|----------------|---------------|
| Who are the users? | Consumer-facing vs internal-facing changes latency requirements |
| What is the input/output? | Text → label? Image → embedding? Query → ranked list? |
| What actions does the system support? | Upload, search, recommend, classify, generate |
| Does the system need to handle cold start? | New users/items with no history |
| Is it real-time or batch? | Drives architecture fundamentally |
| What is the scale? | 1K users vs 100M users changes everything |

### Non-Functional Requirements

| Requirement | Typical Targets | Impact on Design |
|------------|----------------|-----------------|
| Latency | p50 < 100ms, p99 < 500ms | Determines model complexity, caching strategy |
| Throughput | 1K–100K QPS | Horizontal scaling, async processing |
| Availability | 99.9% (8.7h downtime/year) | Redundancy, failover, health checks |
| Freshness | Real-time to daily updates | Streaming vs batch pipeline |
| Cost | Budget for GPU, storage, API calls | Model size, self-hosted vs API |
| Privacy | PII handling, GDPR, HIPAA | Data anonymization, on-prem vs cloud |

### Constraints and Assumptions

Always state your assumptions explicitly:

```
"I'll assume we have ~50M users and ~10M items in the catalog."
"I'll assume we can tolerate up to 200ms latency for the ranking service."
"I'll assume we have access to 6 months of historical interaction data."
"I'll assume we're deploying on AWS with a moderate GPU budget."
```

> **Rahul's Tip:** In my PathWise project, I started by clarifying: users are students/career-changers, input is PDF documents + career interests, output is personalized learning paths. This scoping drove every downstream decision — from chunking strategy to LLM selection (Groq for speed, Cohere for embeddings).

---

## Step 2: Define Metrics (5 minutes)

**Goal:** Establish how you will measure success — both offline (during development) and online (in production). This signals to the interviewer that you think like a practitioner, not an academic.

### Offline Metrics (Development & Validation)

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Classification | F1-score, AUC-ROC | Precision, Recall, Log-loss |
| Ranking / Recommendation | NDCG@k, MAP@k | MRR, Hit Rate@k, Recall@k |
| Regression / Forecasting | RMSE, MAE | MAPE, R², Directional Accuracy |
| Information Retrieval | Recall@k, MRR | Precision@k, NDCG |
| Generation (LLM) | BLEU, ROUGE-L | BERTScore, Human eval, Faithfulness |
| Anomaly Detection | Precision-Recall AUC | F1 at chosen threshold |

### Online Metrics (Production)

| Metric | What It Measures | Example |
|--------|-----------------|---------|
| CTR (Click-Through Rate) | User engagement | Clicks / Impressions |
| Conversion Rate | Business impact | Purchases / Clicks |
| Revenue per Session | Direct business value | Revenue / Sessions |
| User Retention (D1, D7, D30) | Long-term engagement | Users returning on day N |
| Session Duration | Content quality | Avg time in app |
| Query Success Rate | Search effectiveness | Queries with clicks / Total queries |

### Guardrail Metrics

These are metrics that must *not* degrade when you ship a new model:

| Guardrail | Why It Matters |
|-----------|---------------|
| Latency p99 | New model shouldn't slow down the system |
| Error rate | Model shouldn't crash more often |
| Diversity | Recommendations shouldn't become a filter bubble |
| Fairness | Model shouldn't discriminate by protected attributes |
| Content safety | LLM outputs shouldn't be toxic or harmful |

### Counter Metrics

Metrics that move in the *opposite* direction to your primary metric:

```
Primary: Increase CTR on recommendations
Counter: Watch that avg. purchase value doesn't drop (clickbait recs)

Primary: Reduce false positives in fraud detection
Counter: Watch that false negatives don't spike (missed fraud)
```

> **Rahul's Tip:** In my Tesla forecasting project, I used RMSE and MAE as offline metrics but added **directional accuracy** (did the model predict the correct direction of price movement?) as the most important business metric — because a trader doesn't care about RMSE if the predicted direction is wrong.

---

## Step 3: Data Pipeline (10 minutes)

**Goal:** Design how data flows from raw sources to features ready for model consumption.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE ARCHITECTURE                            │
│                                                                               │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│   │  Data     │──▶│  Data    │──▶│  Feature │──▶│  Feature │──▶│  Feature │  │
│   │  Sources  │   │  Ingest  │   │  Eng.    │   │  Store   │   │  Serving │  │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                                               │
│   • Databases    • Kafka/Kinesis • Transforms   • Feast/Tecton • Online:     │
│   • APIs         • Airflow/      • Aggregates   • Redis         low-latency  │
│   • Logs           Prefect       • Embeddings   • Offline:      • Offline:   │
│   • User events  • S3/GCS       • Joins         S3/BigQuery    batch scoring │
│   • Third-party  • Spark                                                      │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Data Collection

| Source | Examples | Considerations |
|--------|----------|---------------|
| User interactions | Clicks, purchases, dwell time | Event tracking, schema consistency |
| Content metadata | Item descriptions, images, categories | Structured vs unstructured |
| User profiles | Demographics, preferences, history | Privacy, PII handling |
| External data | Weather, market data, APIs | Rate limits, data freshness |
| Logs | Server logs, application logs | Volume, parsing, retention |

### Data Storage

| Storage Type | Use Case | Technology |
|-------------|----------|-----------|
| Data warehouse | Historical analytics, training data | BigQuery, Snowflake, Redshift |
| Data lake | Raw unstructured data | S3/GCS + Delta Lake / Iceberg |
| Operational DB | User profiles, metadata | PostgreSQL, DynamoDB |
| Cache | Feature serving, hot data | Redis, Memcached |
| Vector store | Embeddings for retrieval | Pinecone, Weaviate, pgvector, FAISS |
| Message queue | Event streaming | Kafka, SQS, Pub/Sub |

### Feature Engineering

| Feature Category | Examples | Computation |
|-----------------|----------|------------|
| Raw features | User age, item price | Direct from source |
| Aggregate features | Avg purchases/week, 7-day click count | Window functions |
| Cross features | User × Item interaction count | Join + aggregate |
| Embedding features | User embedding, item embedding | Pre-computed via model |
| Time-decay features | Exponentially weighted click history | Online computation |
| Real-time features | Items in cart right now, current session length | Stream processing |

> **Rahul's Tip:** In PathWise, data flows as: PDF upload → `pdf_to_text` (PyMuPDF) → `chunk_text` (400-word chunks with 50-word overlap) → `embed_chunks` (Cohere embed-v4) → store in Supabase (pgvector). This is a classic ingest → transform → embed → store pipeline that I've implemented end-to-end.

---

## Step 4: Model Selection and Training (10 minutes)

**Goal:** Choose the right model architecture for the task, justify your choice, and describe the training strategy.

### Model Selection Decision Tree

```
                            ┌─────────────────┐
                            │  What is the     │
                            │  task?           │
                            └────────┬────────┘
                  ┌──────────────────┼──────────────────┐
                  ▼                  ▼                   ▼
          ┌──────────────┐  ┌──────────────┐    ┌──────────────┐
          │ Classification│  │   Ranking/   │    │  Generation  │
          │ / Regression  │  │   Retrieval  │    │  / Chat      │
          └──────┬───────┘  └──────┬───────┘    └──────┬───────┘
                 │                  │                    │
    ┌────────────┤         ┌───────┤            ┌───────┤
    ▼            ▼         ▼       ▼            ▼       ▼
 ┌──────┐  ┌──────┐  ┌──────┐ ┌──────┐   ┌──────┐ ┌──────┐
 │Tabular│  │ Text/│  │Two-  │ │Learn-│   │ Fine-│ │ RAG+ │
 │ Data  │  │Image │  │Tower │ │to-   │   │ Tune │ │ LLM  │
 │       │  │      │  │      │ │Rank  │   │      │ │      │
 │XGBoost│  │ BERT │  │DSSM/ │ │Lambda│   │ LoRA │ │GPT-4 │
 │LightGB│  │ViT   │  │ANN   │ │MART  │   │ QLoRA│ │Llama │
 │ M     │  │CLIP  │  │      │ │      │   │      │ │      │
 └──────┘  └──────┘  └──────┘ └──────┘   └──────┘ └──────┘
```

### Model Selection Criteria

| Criterion | Simple Model (Logistic Reg, XGBoost) | Complex Model (Deep Learning, LLM) |
|-----------|-------------------------------------|-------------------------------------|
| Data volume | < 100K samples | > 1M samples |
| Feature types | Tabular, structured | Text, images, multimodal |
| Interpretability | High (required by regulation) | Lower (acceptable) |
| Latency | < 10ms | 50ms–2s acceptable |
| Cold start | Can handle with rules | Needs embedding-based fallback |
| Iteration speed | Fast (hours to retrain) | Slow (days, expensive GPUs) |

### Training Strategy

| Component | Description |
|-----------|------------|
| Train/Val/Test Split | Time-based split for temporal data; stratified for imbalanced classes |
| Cross-validation | k-fold for small datasets; single hold-out for large-scale |
| Hyperparameter tuning | Optuna / Ray Tune / Bayesian optimization |
| Distributed training | Data parallelism (DDP), model parallelism (FSDP) for large models |
| Regularization | Dropout, weight decay, early stopping, data augmentation |
| Handling imbalance | SMOTE, class weights, focal loss, downsampling majority |

> **Rahul's Tip:** For my Fashion Recommendation system, I used CLIP (contrastive learning) to create a shared image-text embedding space. Model selection was driven by the multimodal nature of the problem — users search with text queries but products are represented as images. CLIP handles this cross-modal retrieval natively.

---

## Step 5: Serving and Inference (10 minutes)

**Goal:** Design how the trained model serves predictions in production.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       MODEL SERVING ARCHITECTURE                              │
│                                                                               │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│   │  Client   │──▶│  API Gateway │──▶│  Model       │──▶│  Post-       │     │
│   │  Request  │   │  (Rate limit,│   │  Service     │   │  Processing  │     │
│   │           │   │   Auth, LB)  │   │  (FastAPI /  │   │  (Filter,    │     │
│   └──────────┘   └──────────────┘   │   TorchServe)│   │   Rerank)    │     │
│                                      └──────┬───────┘   └──────┬───────┘     │
│                                             │                   │             │
│                                      ┌──────▼───────┐   ┌──────▼───────┐     │
│                                      │  Feature     │   │  Response    │     │
│                                      │  Store       │   │  Cache       │     │
│                                      │  (Redis)     │   │  (Redis/CDN) │     │
│                                      └──────────────┘   └──────────────┘     │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Serving Patterns

| Pattern | Latency | Throughput | Use Case |
|---------|---------|-----------|----------|
| **Online (synchronous)** | Low (ms) | Medium | Real-time predictions (search, reco) |
| **Batch (offline)** | High (hours) | Very high | Pre-compute scores, email campaigns |
| **Near-real-time (async)** | Medium (seconds) | High | Event-driven updates, notifications |
| **Streaming** | Low-medium | High | Continuous scoring (fraud, monitoring) |

### Inference Optimization

| Technique | Speedup | Quality Loss | When to Use |
|-----------|---------|-------------|-------------|
| Model distillation | 2–10× | 1–3% | Deploy smaller model in production |
| Quantization (INT8/INT4) | 2–4× | 0.5–2% | Edge/mobile, reduce GPU memory |
| ONNX Runtime | 1.5–3× | 0% | Framework-agnostic serving |
| TensorRT | 2–5× | < 1% | NVIDIA GPU inference |
| Batching | 2–8× | 0% | High throughput, batch requests together |
| Caching | 10–100× | 0% | Repeated queries, popular items |
| Approximate NN (HNSW, IVF) | 5–50× | 1–5% recall loss | Large-scale vector search |

### Deployment Strategies

| Strategy | Description | Risk Level |
|----------|------------|-----------|
| **Shadow mode** | New model runs in parallel, results logged but not served | Zero risk |
| **A/B test** | Split traffic between old and new model | Low risk |
| **Canary deployment** | Route 1–5% traffic to new model, monitor, ramp up | Low risk |
| **Blue-green** | Two identical environments, switch DNS | Medium risk |
| **Feature flags** | Toggle model per user segment | Flexible |

> **Rahul's Tip:** PathWise uses FastAPI with async endpoints, CORS middleware, and Supabase for persistence. I deployed to Railway with a Procfile (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`). For inference, I use Groq (fast LLM inference) and Cohere (embeddings) as external API services — this is the "ML-as-a-Service" pattern where you offload heavy GPU work to specialized providers.

---

## Step 6: Monitoring and Iteration (5 minutes)

**Goal:** Show you think beyond launch day. Production ML systems decay without active monitoring.

### What to Monitor

| Category | Metrics | Tools |
|----------|---------|-------|
| **Model performance** | Accuracy, NDCG, CTR drift | Custom dashboards, Evidently AI |
| **Data quality** | Missing values, schema violations, distribution shift | Great Expectations, Monte Carlo |
| **System health** | Latency p50/p99, error rate, throughput | Prometheus + Grafana, Datadog |
| **Business metrics** | Revenue, conversion, retention | Amplitude, Mixpanel, internal dashboards |
| **Fairness** | Disparate impact, equalized odds | Aequitas, Fairlearn |

### Data and Concept Drift

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      DRIFT DETECTION PIPELINE                           │
│                                                                          │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│   │Production │──▶│ Feature      │──▶│ Statistical  │──▶│  Alert +   │  │
│   │  Data     │   │ Distribution │   │  Test        │   │  Retrain   │  │
│   │  Stream   │   │  Monitor     │   │ (KS, PSI,   │   │  Trigger   │  │
│   │           │   │              │   │  Chi², JS)   │   │            │  │
│   └──────────┘   └──────────────┘   └──────────────┘   └────────────┘  │
│                                                                          │
│   Types of Drift:                                                        │
│   • Data drift:    P(X) changes (feature distributions shift)            │
│   • Concept drift: P(Y|X) changes (relationship between X and Y shifts) │
│   • Label drift:   P(Y) changes (class balance shifts)                   │
│   • Upstream drift: Schema changes from data producer                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Retraining Strategies

| Strategy | When to Use | Complexity |
|----------|------------|-----------|
| **Scheduled** (daily/weekly) | Stable domains, regular data inflow | Low |
| **Triggered** (drift detected) | High-stakes systems, volatile domains | Medium |
| **Continuous** (online learning) | Fraud detection, ads, fast-changing signals | High |
| **Champion-challenger** | Always keep best model, compare against new candidates | Medium |

### Feedback Loops

```
User Action → Log Event → Join with Ground Truth → Compute Metric → 
→ Dashboard / Alert → Retrain Decision → Deploy New Model → Repeat
```

> **Rahul's Tip:** Always mention monitoring in your design. Even one sentence — "I'd set up Evidently to track feature drift and Prometheus for latency monitoring" — signals production maturity. In PathWise, I use `loguru` for structured logging across all agent actions (see the `BaseAgent.log_action` pattern) and track user engagement through Supabase analytics.

---

# 2. Common ML System Design Questions

---

## 2a. Design a Recommendation System

---

### Problem Statement

*"Design a recommendation system for an e-commerce platform with 50M users and 10M products."*

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│             RECOMMENDATION SYSTEM — TWO-STAGE ARCHITECTURE                   │
│                                                                              │
│   ┌──────────┐                                                               │
│   │  User    │                                                               │
│   │  Request │                                                               │
│   └────┬─────┘                                                               │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────┐                       │
│   │            STAGE 1: CANDIDATE GENERATION          │                       │
│   │         (Reduce 10M items → 1000 candidates)      │                       │
│   │                                                    │                       │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐ │                       │
│   │   │ Collaborative│  │  Content   │  │ Popularity │ │                       │
│   │   │  Filtering  │  │  Based     │  │  Based     │ │                       │
│   │   │ (ALS, ANN)  │  │ (Embedding │  │ (Trending, │ │                       │
│   │   │             │  │  similarity│  │  New items) │ │                       │
│   │   └──────┬─────┘  └──────┬─────┘  └──────┬─────┘ │                       │
│   │          └───────────┬───────────────────┘         │                       │
│   │                      ▼                             │                       │
│   │              Merge & Deduplicate                   │                       │
│   └──────────────────────┬─────────────────────────────┘                       │
│                          │ ~1000 candidates                                   │
│                          ▼                                                    │
│   ┌──────────────────────────────────────────────────┐                       │
│   │              STAGE 2: RANKING                     │                       │
│   │         (Score 1000 → Return top 20)              │                       │
│   │                                                    │                       │
│   │   Features:                                        │                       │
│   │   • User features (age, history, segment)          │                       │
│   │   • Item features (category, price, popularity)    │                       │
│   │   • Cross features (user×item affinity)            │                       │
│   │   • Context (time of day, device, location)        │                       │
│   │                                                    │                       │
│   │   Model: XGBoost / Deep Neural Network / DCN-v2    │                       │
│   └──────────────────────┬─────────────────────────────┘                       │
│                          │ scored + ranked                                    │
│                          ▼                                                    │
│   ┌──────────────────────────────────────────────────┐                       │
│   │              STAGE 3: RE-RANKING                  │                       │
│   │       (Business rules, diversity, freshness)      │                       │
│   │                                                    │                       │
│   │   • Remove out-of-stock items                      │                       │
│   │   • Ensure category diversity                      │                       │
│   │   • Boost promoted items                           │                       │
│   │   • Apply fairness constraints                     │                       │
│   └──────────────────────┬─────────────────────────────┘                       │
│                          ▼                                                    │
│                    Final 20 items → User                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Approach Comparison

| Approach | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **Content-based** | Recommend items similar to what user liked (item features) | No cold-start for items; transparent | Limited discovery; feature engineering heavy |
| **Collaborative filtering** | Recommend items liked by similar users | Discovers unexpected items; no feature engineering | Cold-start for new users/items; sparse data |
| **Matrix factorization** | Decompose user-item matrix into latent factors (SVD, ALS) | Handles sparsity well; scalable | Static; doesn't capture sequential patterns |
| **Two-tower (DSSM)** | Separate user and item encoders; score = dot product | Fast ANN retrieval; handles cold-start with features | Dot product limits interaction modeling |
| **Deep ranking (DCN, DeepFM)** | Neural network with cross features for pointwise scoring | Rich feature interactions | Expensive to serve; harder to interpret |
| **Sequential (SASRec, BERT4Rec)** | Transformer on user's action sequence | Captures temporal patterns, session context | Data hungry; complex serving |
| **Hybrid** | Combine multiple candidate generators + neural ranker | Best of all worlds | System complexity |

### Embedding-Based Retrieval

```python
# Two-tower architecture (conceptual)
class UserTower(nn.Module):
    def forward(self, user_features):
        # user_id embedding + age + gender + historical embeddings
        return self.mlp(torch.cat([user_emb, user_features], dim=-1))  # [B, 128]

class ItemTower(nn.Module):
    def forward(self, item_features):
        # item_id embedding + category + price + image embedding
        return self.mlp(torch.cat([item_emb, item_features], dim=-1))  # [B, 128]

# Training: contrastive loss (in-batch negatives)
# Serving: pre-compute all item embeddings → HNSW index → ANN lookup at query time
```

### Cold-Start Strategies

| Scenario | Strategy |
|----------|---------|
| New user, no history | Popularity-based, demographic-based, or ask preferences (onboarding quiz) |
| New item, no interactions | Content-based features (title, category, image embedding) |
| Session-based (anonymous) | Session-based reco (GRU4Rec, SASRec on current session) |

### Your Experience: Fashion Recommendation System

> In my **Fashion Recommendation System**, I used **CLIP embeddings** to build a multimodal retrieval system where users can search by text ("red summer dress") and the system retrieves visually similar products from an image catalog. Architecture:
>
> 1. **Offline:** Encode all product images with CLIP's image encoder → store embeddings in FAISS index
> 2. **Online:** Encode user query with CLIP's text encoder → ANN search in FAISS → return top-k
> 3. **Ranking:** Re-rank by availability, price range, user preference history
>
> This is a **content-based** system with cross-modal retrieval — a strong example to discuss in interviews because it demonstrates embedding-based candidate generation.

---

## 2b. Design a Search/Retrieval System

---

### Problem Statement

*"Design a search system for a job platform with 100M job listings."*

### Architecture: 4-Stage Funnel

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SEARCH / RETRIEVAL — 4-STAGE FUNNEL                       │
│                                                                              │
│   Query: "senior data scientist remote machine learning"                     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────┐           │
│   │  STAGE 1: QUERY UNDERSTANDING                                │           │
│   │  • Tokenization, spell correction, expansion                 │           │
│   │  • Intent classification (navigational vs informational)     │           │
│   │  • Entity extraction: role=data scientist, level=senior,     │           │
│   │    location=remote, skill=machine learning                   │           │
│   └──────────────────────────┬───────────────────────────────────┘           │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────┐           │
│   │  STAGE 2: CANDIDATE RETRIEVAL (100M → 10K)                   │           │
│   │                                                               │           │
│   │  ┌──────────────────┐  ┌──────────────────┐                  │           │
│   │  │   Sparse          │  │    Dense          │                  │           │
│   │  │   (BM25/Lucene)   │  │    (Embedding ANN)│                  │           │
│   │  │   Exact keyword    │  │    Semantic match  │                  │           │
│   │  │   match            │  │    (captures       │                  │           │
│   │  │                    │  │     synonyms)      │                  │           │
│   │  └────────┬───────────┘  └────────┬───────────┘                  │           │
│   │           └──────────┬───────────┘                              │           │
│   │                      ▼                                          │           │
│   │           Reciprocal Rank Fusion (RRF)                          │           │
│   │           Score = Σ 1/(k + rank_i)                              │           │
│   └──────────────────────┬───────────────────────────────────────┘           │
│                          ▼                                                   │
│   ┌──────────────────────────────────────────────────────────────┐           │
│   │  STAGE 3: RANKING (10K → 100)                                │           │
│   │  • Learning-to-Rank model (LambdaMART, neural L2R)           │           │
│   │  • Features: BM25 score, embedding similarity, recency,      │           │
│   │    click history, job quality signals                         │           │
│   │  • Optimize for NDCG@10                                      │           │
│   └──────────────────────┬───────────────────────────────────────┘           │
│                          ▼                                                   │
│   ┌──────────────────────────────────────────────────────────────┐           │
│   │  STAGE 4: RE-RANKING (100 → 20 displayed)                   │           │
│   │  • Personalization boost (user preference matching)          │           │
│   │  • Diversity (don't show 20 jobs from same company)          │           │
│   │  • Freshness (boost recent postings)                         │           │
│   │  • Business rules (promoted listings, geographic bias)       │           │
│   └──────────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Sparse vs Dense Retrieval

| Aspect | Sparse (BM25) | Dense (Embedding) |
|--------|--------------|-------------------|
| Matching | Exact keyword overlap | Semantic similarity |
| Handles synonyms | No ("ML" ≠ "machine learning") | Yes (learned representations) |
| Zero-shot | Works immediately | Needs trained embeddings |
| Efficiency | Inverted index, very fast | ANN index (HNSW, IVF), fast |
| Interpretability | High (term weights) | Low (vector distances) |
| Best for | Known-item search, exact match | Exploratory search, fuzzy queries |

### Learning to Rank (L2R)

| Approach | Loss Function | Description |
|----------|-------------|-------------|
| **Pointwise** | MSE / Cross-entropy | Predict relevance score for each (query, doc) independently |
| **Pairwise** | RankNet / LambdaRank | Learn relative ordering between pairs of documents |
| **Listwise** | LambdaMART / ListNet | Optimize ranking metric (NDCG) directly over the full list |

> **Rahul's Tip:** Hybrid search (BM25 + dense) with RRF fusion is the gold standard. In my PathWise system, lesson search uses Cohere embeddings + Supabase pgvector for semantic search. In a real production search system, I'd add BM25 via Elasticsearch for exact keyword match and combine with RRF.

---

## 2c. Design a RAG-Based QA System

---

### Problem Statement

*"Design a question-answering system for a company's internal documentation (10K+ documents)."*

### End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      RAG-BASED QA SYSTEM ARCHITECTURE                        │
│                                                                              │
│   ═══════════ OFFLINE PIPELINE (Indexing) ═══════════                        │
│                                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│   │ Documents │──▶│  Parse   │──▶│  Chunk   │──▶│  Embed   │               │
│   │ (PDF,HTML,│   │ (PyMuPDF,│   │ (Semantic│   │ (Cohere  │               │
│   │  Docs)    │   │  Unstr.) │   │  or fixed│   │  embed-  │               │
│   └──────────┘   └──────────┘   │  window) │   │  v4)     │               │
│                                  └──────────┘   └────┬─────┘               │
│                                                      │                      │
│                                                      ▼                      │
│                                              ┌──────────────┐               │
│                                              │ Vector Store  │               │
│                                              │ (pgvector /   │               │
│                                              │  Pinecone /   │               │
│                                              │  Weaviate)    │               │
│                                              └──────────────┘               │
│                                                                              │
│   ═══════════ ONLINE PIPELINE (Query) ═══════════                            │
│                                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│   │  User    │──▶│  Query       │──▶│  Hybrid       │──▶│  Re-rank    │    │
│   │  Query   │   │  Rewriting   │   │  Retrieval    │   │  (Cross-    │    │
│   │          │   │  (HyDE,      │   │  (BM25 +      │   │   encoder / │    │
│   │          │   │   step-back) │   │   Dense + RRF) │   │   Cohere)   │    │
│   └──────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘    │
│                                                                │            │
│                                                                ▼            │
│                                                     ┌──────────────────┐    │
│                                                     │   LLM Generation │    │
│                                                     │   (Prompt with   │    │
│                                                     │    retrieved     │    │
│                                                     │    context +     │    │
│                                                     │    citations)    │    │
│                                                     └──────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Chunking Strategies

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|---------|
| Fixed-window | 512–1024 tokens | 10–20% | General purpose, simple |
| Semantic chunking | Variable (by topic) | None (boundary-aware) | Documents with clear sections |
| Recursive (LangChain) | 512–1024 tokens | 50–200 tokens | Hierarchical documents |
| Sentence-level | 3–5 sentences | 1 sentence | FAQ, short-answer retrieval |
| Parent-child | Small chunks for retrieval, return parent | N/A | Best of both worlds |

### Vector Database Selection

| Vector DB | Type | Strengths | Limitations |
|-----------|------|-----------|-------------|
| **FAISS** | Library (in-process) | Blazing fast, free, great for prototyping | No persistence, no metadata filtering |
| **pgvector** | PostgreSQL extension | SQL + vectors in one DB, ACID compliance | Slower at >10M vectors |
| **Pinecone** | Managed SaaS | Zero-ops, auto-scaling, metadata filtering | Vendor lock-in, cost at scale |
| **Weaviate** | Self-hosted / Cloud | Hybrid search built-in, GraphQL API | Operational complexity |
| **Qdrant** | Self-hosted / Cloud | Fast, rich filtering, Rust-based | Smaller community |
| **ChromaDB** | Lightweight | Easy prototyping, in-memory | Not production-grade at scale |

### Advanced RAG Patterns

| Pattern | Description | When to Use |
|---------|------------|-------------|
| **HyDE** | Generate hypothetical answer, embed it, retrieve similar docs | Improve retrieval for abstract queries |
| **Step-back prompting** | Ask a more general question first, then specific | Multi-hop reasoning questions |
| **Query decomposition** | Break complex query into sub-queries, retrieve for each | Complex questions requiring multiple docs |
| **Contextual compression** | Compress retrieved docs to only relevant sentences | Reduce context length, improve generation |
| **Self-RAG** | Model decides when to retrieve and self-evaluates | Adaptive retrieval, reduce noise |
| **Agentic RAG** | LLM orchestrates retrieval as a tool call | Complex multi-step research tasks |

### Your Experience: PathWise and Medical RAG

> **PathWise RAG Pipeline:** I built a complete RAG system for educational content:
> - **Ingestion:** `pdf_to_text` using PyMuPDF → `chunk_text` (400 words, 50 overlap) → `embed_chunks` via Cohere embed-v4
> - **Storage:** Supabase with pgvector extension
> - **Retrieval:** Cosine similarity search with top-k retrieval
> - **Generation:** Groq LLM (Llama 3) with retrieved context for Q&A, flashcard generation, and concept map creation
> - **Agentic layer:** `SummarizerAgent` and `DiagnosticAgent` that chain retrieval → generation → validation → repair
>
> **Medical RAG (CheXagent):** Built a medical image + text RAG system for radiology report QA:
> - Multimodal embeddings (image + text) for retrieval
> - Strict faithfulness constraints (hallucination is dangerous in medical contexts)
> - Evaluated with clinical metrics alongside standard NLP metrics

---

## 2d. Design a Real-Time Fraud Detection System

---

### Problem Statement

*"Design a system that detects fraudulent transactions in real-time for a payment platform processing 10K transactions/second."*

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME FRAUD DETECTION ARCHITECTURE                      │
│                                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│   │Transaction│──▶│  Feature     │──▶│  ML Scoring  │──▶│  Decision    │    │
│   │  Event    │   │  Engineering │   │  Service     │   │  Engine      │    │
│   │  (Kafka)  │   │  (Real-time  │   │  (Ensemble)  │   │  (Rules +   │    │
│   │           │   │   + Precomp) │   │              │   │   ML score)  │    │
│   └──────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘    │
│                                                                │            │
│                         ┌──────────────────────────────────────┤            │
│                         │                                      │            │
│                         ▼                                      ▼            │
│                  ┌──────────────┐                      ┌──────────────┐     │
│                  │  Human       │                      │  Allow /     │     │
│                  │  Review      │                      │  Block /     │     │
│                  │  Queue       │                      │  Challenge   │     │
│                  │  (Medium     │                      │  (Real-time  │     │
│                  │   risk)      │                      │   response)  │     │
│                  └──────────────┘                      └──────────────┘     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  FEEDBACK LOOP                                                       │   │
│   │  Human labels → Retrain → Champion-challenger evaluation → Deploy    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering for Fraud

| Feature Category | Examples | Computation |
|-----------------|----------|------------|
| **Transaction features** | Amount, currency, merchant category, payment method | Raw event fields |
| **Velocity features** | Transactions in last 1h/24h/7d, amount in last 24h | Sliding window aggregations (Redis/Flink) |
| **Behavioral features** | Avg transaction amount, usual time-of-day, common merchants | Pre-computed user profiles |
| **Device/IP features** | New device?, IP geolocation, IP reputation, proxy detection | External enrichment |
| **Graph features** | Shared device with flagged accounts, merchant risk score | Graph database (Neo4j) |
| **Anomaly features** | Z-score of amount vs user history, deviation from pattern | Real-time computation |

### Model Architecture

| Component | Model | Reason |
|-----------|-------|--------|
| **Primary scorer** | XGBoost / LightGBM | Fast inference (<5ms), handles tabular features well, interpretable |
| **Anomaly detector** | Isolation Forest / Autoencoder | Catches novel fraud patterns not in training data |
| **Sequence model** | LSTM / Transformer on transaction sequence | Captures temporal patterns in user behavior |
| **Ensemble** | Weighted combination of above | Robustness: different models catch different fraud types |

### Key Challenges

| Challenge | Solution |
|-----------|---------|
| Extreme class imbalance (0.1% fraud) | Focal loss, SMOTE, cost-sensitive learning, anomaly detection |
| Real-time latency (<50ms) | Pre-computed features in Redis, lightweight model, async enrichment |
| Adversarial behavior (fraudsters adapt) | Regular retraining, online learning, anomaly detection for novel patterns |
| False positives hurt user experience | Multi-tier: allow / challenge (2FA) / block; adjust thresholds by user trust |
| Labeling delay (chargebacks take weeks) | Semi-supervised learning, use rules + analyst labels for fast feedback |

---

## 2e. Design a Text Classification System

---

### Problem Statement

*"Design a system to classify customer support tickets into 50 categories with 1M+ tickets per day."*

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   TEXT CLASSIFICATION PIPELINE                                │
│                                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│   │  Raw     │──▶│  Preprocessing│──▶│  Feature     │──▶│  Model       │    │
│   │  Text    │   │  • Clean HTML │   │  Extraction  │   │  Prediction  │    │
│   │  Input   │   │  • Normalize  │   │  • TF-IDF    │   │              │    │
│   │          │   │  • Language   │   │  • BERT emb. │   │  ┌────────┐  │    │
│   │          │   │    detect     │   │  • Few-shot  │   │  │LoRA-FT│  │    │
│   └──────────┘   └──────────────┘   │    prompt    │   │  │BERT   │  │    │
│                                      └──────────────┘   │  │LLM    │  │    │
│                                                          │  └────────┘  │    │
│                                                          └──────┬───────┘    │
│                                                                 │            │
│   ┌──────────────────────────────────────────────────────────────┘            │
│   │                                                                          │
│   ▼                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐           │
│   │  Post-Processing                                              │           │
│   │  • Confidence thresholding (low confidence → human review)    │           │
│   │  • Multi-label handling                                       │           │
│   │  • Route to appropriate team                                  │           │
│   └──────────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Approach Selection

| Approach | Labeled Data Needed | Latency | Accuracy | Best For |
|----------|-------------------|---------|----------|---------|
| **TF-IDF + Logistic Regression** | 1K+ per class | < 5ms | Good | Baseline, high-throughput |
| **Sentence-BERT + KNN** | 50–100 per class | 10–50ms | Good | Low-label regime |
| **Fine-tuned BERT/DeBERTa** | 500+ per class | 20–100ms | Excellent | When accuracy matters most |
| **Few-shot (GPT-4, Claude)** | 0–10 examples | 500ms–2s | Good–Excellent | Rapid prototyping, new categories |
| **LoRA fine-tuned LLM** | 100–500 per class | 100–500ms | Excellent | Best quality, moderate data |

### Handling 50 Categories

| Challenge | Solution |
|-----------|---------|
| Class imbalance | Class weights, focal loss, oversampling rare classes |
| Hierarchical taxonomy | Hierarchical classification (top category → sub-category) |
| Ambiguous categories | Allow multi-label, use top-2 predictions with confidence |
| Evolving categories | Regular retraining, few-shot for new categories, active learning |

### Your Experience: Legal Docs and Call Transcripts

> In my projects, I've applied text classification to:
> - **Legal document classification:** Scraped legal data, classified into document types using NLP pipelines. Key challenge was long documents — solved with section-level classification + aggregation.
> - **Call transcript analysis:** Classified customer service interactions by topic and sentiment. Used embedding-based features + XGBoost for high-throughput classification.
>
> **Key learning:** Start with TF-IDF + Logistic Regression as a baseline (5 minutes to build, surprisingly competitive). Move to BERT fine-tuning only if the baseline doesn't meet accuracy requirements.

---

## 2f. Design an LLM-Powered Chatbot

---

### Problem Statement

*"Design a customer-facing chatbot that can answer questions about products, handle orders, and escalate to humans."*

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LLM-POWERED CHATBOT ARCHITECTURE                          │
│                                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────────────────────────┐    │
│   │  User    │──▶│  API Gateway │──▶│  ORCHESTRATOR (LangGraph /       │    │
│   │  Message │   │  • Auth      │   │  Custom State Machine)           │    │
│   │          │   │  • Rate limit│   │                                   │    │
│   │          │   │  • Session   │   │  ┌───────────────────────────┐   │    │
│   └──────────┘   └──────────────┘   │  │  Intent Router            │   │    │
│                                      │  │  • FAQ → RAG pipeline     │   │    │
│                                      │  │  • Order → API tools      │   │    │
│                                      │  │  • Complaint → Escalation │   │    │
│                                      │  │  • Chitchat → LLM direct  │   │    │
│                                      │  └───────────────────────────┘   │    │
│                                      │                                   │    │
│                                      │  ┌───────┐  ┌───────┐  ┌──────┐ │    │
│                                      │  │  RAG  │  │ Tools │  │ LLM  │ │    │
│                                      │  │ (KB   │  │ (Order│  │(GPT-4│ │    │
│                                      │  │ search│  │  API, │  │ etc.)│ │    │
│                                      │  │  )    │  │  CRM) │  │      │ │    │
│                                      │  └───────┘  └───────┘  └──────┘ │    │
│                                      └──────────────────────────────────┘    │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  GUARDRAILS & SAFETY                                                  │   │
│   │  • Input: toxicity filter, PII detection, prompt injection defense    │   │
│   │  • Output: hallucination check, compliance filter, tone enforcement   │   │
│   │  • Fallback: "I don't know" rather than fabricate                     │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  CACHING & OPTIMIZATION                                               │   │
│   │  • Semantic cache: hash(embedding(query)) → cached response           │   │
│   │  • Prompt cache: reuse system prompt across sessions                   │   │
│   │  • Response streaming: SSE for real-time feel                          │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Options | Trade-off |
|----------|---------|-----------|
| **LLM selection** | GPT-4 (best quality) vs Claude (long context) vs Llama (self-hosted) vs Groq (speed) | Quality ↔ Cost ↔ Latency ↔ Privacy |
| **Orchestration** | LangGraph (stateful) vs LangChain (simple) vs Custom (control) | Flexibility ↔ Development speed |
| **Memory** | Conversation buffer vs summary vs vector store | Context quality ↔ Token cost |
| **Tool use** | Function calling vs ReAct prompting | Reliability ↔ Flexibility |
| **Caching** | Exact match vs semantic cache | Hit rate ↔ Freshness |

### Guardrails

| Layer | Technique | Purpose |
|-------|-----------|---------|
| Input | Regex + classifier for PII | Remove SSN, credit card numbers |
| Input | Prompt injection detection | Block "ignore previous instructions" attacks |
| Input | Toxicity classifier | Block abusive input |
| Output | Grounding check against retrieved docs | Prevent hallucination |
| Output | Topic classifier | Ensure response stays on-topic |
| Output | NeMo Guardrails / Guardrails AI | Programmable safety rails |

### Your Experience: PathWise Chatbot

> In **PathWise**, I built an LLM-powered educational chatbot with:
> - **Orchestrator:** Custom agent router (`AgentRouter` class) that routes to `SummarizerAgent` or `DiagnosticAgent` based on user intent
> - **RAG integration:** Chat is grounded in the user's uploaded PDF content (chunked, embedded, retrieved)
> - **Session management:** In-memory conversation store with per-user history tracking
> - **Adaptive responses:** Explanation level (beginner/intermediate/advanced) and framework preference per user
> - **Tools:** Flashcard generation, quiz creation, concept map generation — all callable by the agent
>
> Architecture: FastAPI → Agent Router → [RAG retrieval | Tool calls | Direct LLM] → Structured response → User

---

## 2g. Design a Time Series Forecasting System

---

### Problem Statement

*"Design a system that forecasts stock prices for the next 1–5 days to inform a trading strategy."*

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                TIME SERIES FORECASTING SYSTEM ARCHITECTURE                    │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  DATA INGESTION                                                       │   │
│   │  • Market data API (Yahoo Finance, Alpha Vantage) — OHLCV data       │   │
│   │  • Alternative data: sentiment (Twitter/Reddit), macroeconomic       │   │
│   │  • Scheduled daily pulls + real-time websocket for intraday          │   │
│   └──────────────────────────────┬───────────────────────────────────────┘   │
│                                  ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  FEATURE ENGINEERING                                                  │   │
│   │                                                                       │   │
│   │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐ │   │
│   │  │ Technical Indicators│  │ Statistical Features │  │  Lag         │ │   │
│   │  │ • RSI (14-day)      │  │ • Rolling mean/std   │  │  Features    │ │   │
│   │  │ • MACD (12,26,9)    │  │ • GARCH volatility   │  │  • Close t-1 │ │   │
│   │  │ • Bollinger Bands   │  │ • Autocorrelation    │  │  • Close t-5 │ │   │
│   │  │ • OBV (On-Balance   │  │ • Stationarity (ADF) │  │  • Return    │ │   │
│   │  │   Volume)           │  │ • Hurst exponent     │  │    t-1..t-20│ │   │
│   │  │ • ATR               │  │                      │  │             │ │   │
│   │  └─────────────────────┘  └─────────────────────┘  └──────────────┘ │   │
│   └──────────────────────────┬───────────────────────────────────────────┘   │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  MODEL ENSEMBLE                                                       │   │
│   │                                                                       │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │   │
│   │  │  ARIMA/  │  │ XGBoost  │  │   LSTM   │  │  Ensemble /      │    │   │
│   │  │  SARIMA  │  │ LightGBM │  │ Bi-LSTM  │  │  Stacking        │    │   │
│   │  │ (linear  │  │ (tabular │  │ (sequence│  │  (weighted avg   │    │   │
│   │  │  trends) │  │  features│  │  patterns│  │   or meta-model) │    │   │
│   │  │          │  │  + feat. │  │  over    │  │                   │    │   │
│   │  │          │  │  import.)│  │  time)   │  │                   │    │   │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │   │
│   └──────────────────────────┬───────────────────────────────────────────┘   │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  VALIDATION & SERVING                                                 │   │
│   │  • Walk-forward validation (NO random split for time series!)         │   │
│   │  • Expanding or sliding window                                        │   │
│   │  • Prediction intervals (quantile regression or conformal)            │   │
│   │  • Daily batch predictions → Dashboard / API / Trading engine         │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering Deep Dive

| Feature | Formula / Description | Signal |
|---------|----------------------|--------|
| **RSI** | 100 − 100/(1 + avg_gain/avg_loss) over 14 days | Overbought (>70) / Oversold (<30) |
| **MACD** | EMA(12) − EMA(26); Signal = EMA(MACD, 9) | Trend direction and momentum |
| **Bollinger Bands** | Middle = SMA(20); Upper/Lower = Middle ± 2σ | Volatility, mean reversion |
| **OBV** | Cumulative sum of volume × sign(price change) | Volume confirms price trend |
| **GARCH(1,1)** | σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} | Volatility clustering |
| **ATR** | Average of max(High−Low, |High−Close_prev|, |Low−Close_prev|) | Volatility measure |
| **Hurst Exponent** | H > 0.5 trending, H < 0.5 mean-reverting, H ≈ 0.5 random walk | Regime detection |

### Walk-Forward Validation

```
Standard CV (WRONG for time series — data leakage!):
┌───────────────────────────────────────────┐
│ Fold 1: [test][ train ][ train ][ train ] │  ← Future data in training!
│ Fold 2: [ train ][test][ train ][ train ] │
└───────────────────────────────────────────┘

Walk-Forward Validation (CORRECT):
┌───────────────────────────────────────────┐
│ Fold 1: [====TRAIN====][TEST]             │
│ Fold 2: [======TRAIN======][TEST]         │
│ Fold 3: [========TRAIN========][TEST]     │
│ Fold 4: [==========TRAIN==========][TEST] │
└───────────────────────────────────────────┘
Always train on past, test on future. Never look ahead.
```

### Model Comparison

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|---------|
| **ARIMA/SARIMA** | Interpretable, captures linear trends/seasonality | Can't capture non-linear patterns | Baseline, interpretable forecasts |
| **XGBoost** | Feature importance, handles non-linear, fast | No native sequence modeling | Tabular features, feature engineering |
| **LSTM** | Learns long-range dependencies | Slow to train, overfits on small data | Sequence patterns, multivariate |
| **Prophet** | Handles holidays, changepoints, easy to use | Limited for complex patterns | Business metrics with seasonality |
| **Temporal Fusion Transformer** | Attention over time, interpretable attention weights | Complex, data hungry | Multiple related time series |
| **N-BEATS / N-HiTS** | Pure DL, no feature engineering needed | Needs lots of data | Large-scale forecasting |

### Your Experience: Tesla Price Forecasting

> In my **Tesla Forecasting Project**, I built an end-to-end system:
> - **Features:** RSI, MACD, Bollinger Bands, OBV, GARCH volatility, 20+ lag features, rolling statistics
> - **Models:** Compared ARIMA, XGBoost, LSTM, and Bidirectional LSTM
> - **XGBoost** performed best for 1-day ahead predictions due to the rich feature engineering (feature importance showed RSI and GARCH volatility as top predictors)
> - **LSTM** captured multi-day sequential patterns better for 5-day forecasts
> - **Validation:** Walk-forward with expanding window; reported RMSE, MAE, MAPE, and directional accuracy
> - **Key insight:** Feature engineering (domain-specific indicators) mattered more than model complexity. XGBoost with good features beat LSTM with raw OHLCV data.
> - **Serialization:** Saved models as `.joblib` files with separate feature and target scalers for reproducible inference

---

# 3. Scalability Patterns

---

Scalability is what separates a data science prototype from a production ML system. Interviewers want to hear that you understand how your ML system behaves when traffic 10×.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SCALABILITY PATTERNS FOR ML SYSTEMS                     │
│                                                                              │
│   ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│   │  COMPUTE           │    │  DATA              │    │  MODEL             │   │
│   │  SCALABILITY       │    │  SCALABILITY       │    │  SCALABILITY       │   │
│   │                    │    │                    │    │                    │   │
│   │  • Horizontal      │    │  • Sharding        │    │  • Model           │   │
│   │    scaling (more   │    │  • Read replicas   │    │    distillation   │   │
│   │    instances)      │    │  • Partitioning    │    │  • Quantization   │   │
│   │  • Vertical        │    │  • Caching layers  │    │  • Batch inference│   │
│   │    scaling (bigger │    │  • CDN for static  │    │  • Model sharding │   │
│   │    machines)       │    │    assets          │    │    (tensor        │   │
│   │  • Auto-scaling    │    │  • Data lakes for  │    │    parallelism)   │   │
│   │    (CPU/GPU usage  │    │    cold storage    │    │  • Cascade models │   │
│   │    triggers)       │    │                    │    │    (cheap → exp.) │   │
│   └───────────────────┘    └───────────────────┘    └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Horizontal vs Vertical Scaling

| Aspect | Horizontal (Scale Out) | Vertical (Scale Up) |
|--------|----------------------|---------------------|
| Approach | Add more machines | Get a bigger machine |
| Cost curve | Linear | Exponential (diminishing returns) |
| Ceiling | Practically unlimited | Hardware limits |
| Complexity | Need load balancing, distributed state | Simple, single machine |
| Best for | Stateless services, inference APIs | Training, single-model inference |
| Example | 10 × g4dn.xlarge for inference | 1 × p4d.24xlarge for training |

### Load Balancing

| Strategy | How It Works | Best For |
|----------|-------------|---------|
| Round robin | Requests distributed evenly across servers | Homogeneous servers, similar request cost |
| Least connections | Route to server with fewest active requests | Variable request processing time |
| Weighted | Route more traffic to more powerful servers | Heterogeneous fleet |
| Consistent hashing | Same user → same server (session affinity) | Caching, stateful services |
| ML-aware | Route based on model version, GPU availability | Multi-model serving |

### Caching Strategy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CACHING LAYERS                                   │
│                                                                          │
│  Request → CDN Cache → API Gateway Cache → Application Cache → DB Cache  │
│            (static     (response for       (Redis: features,  (query     │
│             assets)     popular queries)    embeddings,        cache,     │
│                                             predictions)       pg cache)  │
│                                                                          │
│  Cache strategies:                                                        │
│  • TTL-based:      Expire after N seconds (good for changing data)       │
│  • LRU:            Evict least recently used (bounded memory)            │
│  • Write-through:  Update cache on write (consistency)                   │
│  • Write-behind:   Async cache update (performance)                      │
│  • Cache-aside:    App checks cache, loads from DB on miss               │
└──────────────────────────────────────────────────────────────────────────┘
```

### Async Processing

| Pattern | Technology | Use Case |
|---------|-----------|----------|
| Message queue | Kafka, SQS, RabbitMQ | Decouple producers from consumers |
| Task queue | Celery, Bull | Background processing (model retraining, batch scoring) |
| Event streaming | Kafka Streams, Flink | Real-time feature computation, event processing |
| Pub/Sub | Google Pub/Sub, SNS | Fan-out notifications, multi-consumer events |

### Database Scaling

| Technique | Description | Trade-off |
|-----------|------------|-----------|
| **Read replicas** | Copies of primary DB that serve reads | Eventual consistency; great for read-heavy ML feature lookups |
| **Sharding** | Partition data across multiple DBs (by user_id, region) | Complexity; cross-shard queries are expensive |
| **Denormalization** | Store pre-joined data for fast reads | Storage cost; update complexity |
| **Connection pooling** | Reuse DB connections (PgBouncer) | Reduces connection overhead |
| **Materialized views** | Pre-computed query results | Stale data; great for feature aggregations |

---

# 4. Data Pipeline Design

---

## Batch vs Streaming

| Aspect | Batch Processing | Stream Processing |
|--------|-----------------|-------------------|
| Latency | Hours | Milliseconds to seconds |
| Throughput | Very high | High (with scaling) |
| Complexity | Lower | Higher (exactly-once, ordering) |
| Tools | Spark, Airflow, dbt | Kafka Streams, Flink, Spark Streaming |
| Use case | Training data, daily features, reports | Real-time features, fraud detection, alerts |
| Cost | Lower (runs periodically) | Higher (always running) |

### Lambda vs Kappa Architecture

```
LAMBDA ARCHITECTURE (Batch + Speed layers):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   Raw Data ──┬──▶ Batch Layer (Spark) ──▶ Batch Views     │
│              │                                ↓            │
│              │                          ┌──────────┐       │
│              └──▶ Speed Layer (Flink) ──▶│ Serving  │       │
│                   (real-time approx.)   │  Layer   │       │
│                                         └──────────┘       │
│   Pros: Accurate batch + low-latency streaming             │
│   Cons: Two codebases to maintain                          │
└────────────────────────────────────────────────────────────┘

KAPPA ARCHITECTURE (Streaming only):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   Raw Data ──▶ Stream Layer (Kafka + Flink) ──▶ Serving   │
│                (replay log for reprocessing)               │
│                                                            │
│   Pros: Single codebase, simpler                           │
│   Cons: Hard to debug, replay can be slow                  │
└────────────────────────────────────────────────────────────┘
```

## Feature Stores

A feature store is a centralized platform for storing, serving, and managing ML features.

| Component | Description | Technology |
|-----------|------------|-----------|
| **Offline store** | Historical features for training | S3/GCS + Parquet, BigQuery, Delta Lake |
| **Online store** | Low-latency features for inference | Redis, DynamoDB, Bigtable |
| **Feature registry** | Metadata, lineage, documentation | Feast, Tecton, Hopsworks |
| **Transformation engine** | Compute features from raw data | Spark, Flink, SQL |

### Why Feature Stores Matter

```
WITHOUT Feature Store:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Training     │     │ Batch       │     │ Real-time   │
│ Pipeline     │     │ Inference   │     │ Inference   │
│              │     │             │     │             │
│ SELECT ...   │     │ SELECT ...  │     │ SELECT ...  │
│ (SQL v1)     │     │ (SQL v2)    │     │ (Python v3) │
│              │     │  DIFFERENT! │     │  DIFFERENT! │
└─────────────┘     └─────────────┘     └─────────────┘
↑ Training-serving skew! Features computed differently. ↑

WITH Feature Store:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Training     │     │ Batch       │     │ Real-time   │
│ Pipeline     │     │ Inference   │     │ Inference   │
│              │     │             │     │             │
│  feast.get() │     │  feast.get()│     │  feast.get()│
│  (SAME!)     │     │  (SAME!)    │     │  (SAME!)    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       └──────────────────┬──────────────────┘
                          ▼
                 ┌─────────────────┐
                 │  Feature Store  │
                 │  (Single source │
                 │   of truth)     │
                 └─────────────────┘
```

## Data Versioning

| Tool | Approach | Best For |
|------|---------|---------|
| **DVC** | Git-like versioning for data (pointers in git, data in S3) | Small-medium teams, simple setup |
| **Delta Lake** | ACID transactions on data lake, time travel | Large-scale, Spark ecosystem |
| **lakeFS** | Git-like branching for data lakes | Advanced data management |
| **MLflow** | Model + data artifact tracking | Experiment tracking focused |

## Schema Evolution

| Strategy | Description |
|----------|------------|
| **Additive** | Only add new columns (backward compatible) |
| **Schema registry** | Enforce schemas at write time (Confluent Schema Registry) |
| **Schema-on-read** | Flexible schema, validate at consumption |
| **Migration scripts** | Versioned migrations (Alembic for SQL, custom for NoSQL) |

---

# 5. Trade-Offs to Discuss

---

Discussing trade-offs is the single most important signal of seniority in ML system design interviews. Junior candidates propose one solution. Senior candidates present options with trade-offs.

### Core Trade-Off Matrix

| Trade-Off | Option A | Option B | How to Decide |
|-----------|----------|----------|--------------|
| **Latency vs Accuracy** | Simple model (XGBoost, 5ms) | Complex model (BERT, 100ms) | What's the latency SLA? Is the accuracy gap significant? |
| **Cost vs Performance** | CPU inference ($0.10/1K req) | GPU inference ($1.00/1K req) | Revenue impact of better model vs infrastructure cost |
| **Simple vs Complex** | Logistic regression + heuristics | Deep learning ensemble | Data volume, team expertise, maintenance burden |
| **Real-time vs Batch** | Stream processing (Flink) | Batch processing (Spark) | How stale can predictions be? Cost of infrastructure |
| **Build vs Buy** | Custom model pipeline | SageMaker / Vertex AI | Time-to-market, team size, customization needs |
| **Precision vs Recall** | High precision (fewer false positives) | High recall (fewer false negatives) | Cost of false positive vs false negative |
| **Exploration vs Exploitation** | Show diverse recs (explore) | Show proven winners (exploit) | Cold-start phase vs optimization phase |
| **Freshness vs Stability** | Retrain daily | Retrain monthly | Data volatility, concept drift rate |
| **Privacy vs Utility** | Differential privacy, federated learning | Centralized training on raw data | Regulatory requirements, user trust |
| **Generality vs Specialization** | One model for all users | Per-segment or per-user models | Data volume per segment, maintenance cost |

### Framework for Discussing Trade-Offs

```
1. State the trade-off clearly:
   "There's a trade-off between latency and model accuracy here."

2. Present both options:
   "Option A: XGBoost with 5ms latency and 0.82 AUC
    Option B: Fine-tuned BERT with 80ms latency and 0.89 AUC"

3. Relate to requirements:
   "Given our p99 latency requirement of 100ms, BERT is feasible,
    but adds GPU cost. If budget is tight, XGBoost with better
    features might close the accuracy gap."

4. Make a recommendation:
   "I'd start with XGBoost as the baseline, measure the business
    impact, and upgrade to BERT only if the accuracy delta
    translates to meaningful revenue improvement."
```

### The "Cascade" Pattern (Have Your Cake and Eat It Too)

When latency vs accuracy seems like a hard trade-off, use a cascade:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MODEL CASCADE PATTERN                            │
│                                                                      │
│  Request ──▶ [Simple Model] ──── Confident? ──── Yes ──▶ Return     │
│                                      │                               │
│                                      No (uncertain)                  │
│                                      │                               │
│                                      ▼                               │
│                              [Complex Model] ──▶ Return              │
│                                                                      │
│  Example:                                                            │
│  • 80% of requests handled by XGBoost (5ms) — high confidence       │
│  • 20% escalated to BERT (100ms) — low confidence cases             │
│  • Average latency: 0.8×5 + 0.2×100 = 24ms (much better!)          │
│  • Accuracy: near-BERT level (complex model handles hard cases)     │
└──────────────────────────────────────────────────────────────────────┘
```

---

# 6. Monitoring, Observability, and Iteration

---

## The Four Pillars of ML Observability

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ML OBSERVABILITY — FOUR PILLARS                            │
│                                                                              │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────┐ │
│   │  DATA QUALITY   │  │  MODEL PERF    │  │  SYSTEM HEALTH │  │ BUSINESS │ │
│   │                 │  │                │  │                │  │ METRICS  │ │
│   │  • Schema       │  │  • Accuracy    │  │  • Latency     │  │          │ │
│   │    validation   │  │    drift       │  │    p50/p99     │  │  • CTR   │ │
│   │  • Distribution │  │  • Prediction  │  │  • Error rate  │  │  • Rev   │ │
│   │    shift        │  │    distribution│  │  • GPU util.   │  │  • NPS   │ │
│   │  • Missing      │  │  • Feature     │  │  • Memory      │  │  • Conv  │ │
│   │    values       │  │    importance  │  │  • Throughput   │  │    rate  │ │
│   │  • Freshness    │  │    shift       │  │  • Queue depth │  │          │ │
│   └────────────────┘  └────────────────┘  └────────────────┘  └──────────┘ │
│                                                                              │
│   Tools: Great Expectations │ Evidently, Fiddler │ Prometheus,   │ Amplitude│
│          Monte Carlo        │ WhyLabs, Arize    │ Grafana, DD   │ Mixpanel │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Alerting Strategy

| Severity | Trigger | Action |
|----------|---------|--------|
| **P0 — Critical** | Model serving is down, error rate > 10% | PagerDuty, immediate response, auto-rollback |
| **P1 — High** | Accuracy dropped > 5%, latency spike > 3× | Slack alert, investigate within 1 hour |
| **P2 — Medium** | Data drift detected, feature distribution shift | Ticket created, investigate within 1 day |
| **P3 — Low** | Slow training job, minor metric degradation | Weekly review, log for analysis |

## Continuous Improvement Cycle

```
Deploy Model v1 → Monitor Metrics → Detect Degradation →
→ Root Cause Analysis (data drift? concept drift? bug?) →
→ Fix (retrain? new features? new model?) → A/B Test Fix →
→ Deploy Model v2 → Monitor → Repeat
```

---

# 7. Interview Tips and Communication Strategy

---

## The STAR-D Framework for ML System Design

Adapt the STAR method for system design:

| Step | Action | Time | Signals to Interviewer |
|------|--------|------|----------------------|
| **S — Scope** | Clarify requirements, state assumptions | 5 min | Doesn't jump to solutions |
| **T — Trade-offs** | Identify key trade-offs, discuss options | 5 min | Thinks critically |
| **A — Architecture** | Draw end-to-end system, explain data flow | 15 min | Can design at scale |
| **R — Reasoning** | Justify model choice, feature engineering | 10 min | Deep ML knowledge |
| **D — Deployment** | Serving, monitoring, iteration plan | 10 min | Production experience |

## Communication Tips

### Do:

1. **Start with requirements** — "Before I start designing, let me clarify a few things..."
2. **Draw diagrams** — Even in a phone screen, describe the boxes and arrows
3. **Discuss trade-offs explicitly** — "There's a trade-off between X and Y here. Given our constraints, I'd choose X because..."
4. **Mention monitoring** — "After deployment, I'd monitor X, Y, Z to detect drift"
5. **Reference your experience** — "In my PathWise project, I faced a similar challenge and solved it by..."
6. **Quantify when possible** — "This would handle ~10K QPS with p99 < 200ms"
7. **Think aloud** — Let the interviewer follow your reasoning
8. **Ask clarifying questions** — It shows maturity, not weakness

### Don't:

1. **Don't jump to models** — System design is 80% infrastructure, 20% model
2. **Don't over-engineer** — Start simple, add complexity only when justified
3. **Don't forget the human** — Who labels data? Who reviews edge cases?
4. **Don't ignore cost** — A system that works but costs $1M/month is not a good design
5. **Don't skip evaluation** — Always explain how you'd measure success
6. **Don't present one option** — Always present at least two approaches with trade-offs

## How to Draw Diagrams

Even in a virtual interview, you can describe architecture clearly:

```
"The system has three main components:
 1. An API layer (FastAPI) that receives requests and handles auth
 2. A feature store (Redis for online, S3 for offline) that serves features
 3. A model service that runs inference

Data flows like this:
 User request → API gateway → Feature enrichment (parallel lookup in Redis) →
 → Model inference → Post-processing (business rules) → Response

For the offline pipeline:
 Raw events → Kafka → Spark transformation → Feature store + Training data lake
 → Scheduled training → Model registry → A/B test deployment"
```

## Connecting to Your Experience

Always have 2–3 "go-to" project stories ready:

| Scenario | Your Project | Key Talking Points |
|----------|-------------|-------------------|
| RAG / Retrieval system | PathWise, Medical RAG | Chunking strategy, embedding choice, pgvector, agentic RAG |
| Recommendation system | Fashion Reco (CLIP) | Cross-modal embeddings, FAISS index, content-based retrieval |
| Time series / Forecasting | Tesla Price Prediction | Feature engineering (RSI, GARCH), walk-forward validation, XGBoost vs LSTM |
| LLM application | PathWise Chatbot | Agent architecture, tool use, guardrails, session management |
| Deployment / MLOps | PathWise on Railway | FastAPI, Procfile, environment management, async endpoints |
| Data pipeline | PathWise PDF ingestion | Parse → chunk → embed → store → retrieve pipeline |

---

# 8. Common Interview Questions with Frameworks for Answering

---

## Question 1: "Design a content recommendation feed (like TikTok/YouTube)"

### Framework

```
Requirements → Two-stage architecture → Features → Metrics → Iteration

1. SCOPE:
   - 100M+ users, 10M+ videos, real-time feed
   - Optimize for: engagement (watch time), retention (daily active users)
   - Constraints: <200ms latency, personalized per user

2. ARCHITECTURE:
   Candidate Generation (embedding ANN, 10M → 500)
   → Ranking (deep neural network with cross features)
   → Re-ranking (diversity, freshness, ad insertion)

3. KEY FEATURES:
   - User: watch history embedding, demographic, session context
   - Video: content embedding (from video frames), engagement stats, creator features
   - Cross: user-video affinity, similar users' engagement

4. MODEL:
   - Candidate gen: Two-tower model, updated daily
   - Ranking: Wide & Deep or DCN-v2, updated hourly
   - Handle cold start: content-based features for new videos, explore-exploit for new users

5. METRICS:
   - Offline: NDCG@20, Watch time prediction accuracy
   - Online: Avg watch time, D7 retention, diversity score
   - Guardrails: Content safety, creator fairness, latency p99

6. MONITORING:
   - A/B test all model changes (at least 7 days)
   - Track engagement distribution (avoid filter bubbles)
   - Monitor for popularity bias and recency bias
```

---

## Question 2: "Design a real-time bidding (RTB) system for ad targeting"

### Framework

```
1. SCOPE:
   - Bid on ad impressions in <100ms, millions of bid requests/second
   - Predict: P(click), P(conversion), expected revenue

2. ARCHITECTURE:
   Bid request → Feature lookup (Redis, <5ms) → Model scoring (<10ms)
   → Bid price calculation → Response

3. MODELS:
   - CTR model: Logistic Regression / DeepFM (fast inference)
   - Calibration: Platt scaling (predicted probabilities must be accurate for bidding)
   - Bid = P(click) × Value(click) - apply second-price auction logic

4. KEY TRADE-OFF:
   - Latency vs accuracy: Must respond in <100ms total (including network)
   - Use lightweight models with pre-computed features

5. MONITORING:
   - Calibration curve (are predicted CTRs accurate?)
   - Win rate, cost per acquisition, ROI
```

---

## Question 3: "How would you handle a model that performs well offline but poorly in production?"

### Framework

```
Systematic debugging checklist:

1. DATA ISSUES:
   - Training-serving skew (features computed differently)
   - Data leakage in training (future data leaked into features)
   - Distribution shift (production data differs from training data)

2. FEATURE ISSUES:
   - Missing features at serving time
   - Feature computation latency (stale features)
   - Feature type mismatch (int vs float)

3. INFRASTRUCTURE ISSUES:
   - Model version mismatch (wrong model deployed)
   - Preprocessing inconsistency (tokenizer, scaler not matching)
   - Batching artifacts (padding, truncation)

4. EVALUATION ISSUES:
   - Offline evaluation metric doesn't align with business metric
   - Offline test set not representative of production traffic
   - Selection bias in offline evaluation

5. DEBUGGING STEPS:
   a. Log model inputs and outputs in production (sample 1%)
   b. Re-run offline evaluation on production-sampled data
   c. Compare feature distributions: training vs production
   d. Check for missing/null features at serving time
   e. Shadow mode: run new model alongside old, compare outputs
```

> **Rahul's Tip:** In PathWise, I encountered this when LLM responses worked well in testing but degraded with real user inputs. The issue was input diversity — real users ask questions in unexpected formats. I fixed it by adding query rewriting in the chat pipeline and improving prompt engineering with few-shot examples.

---

## Question 4: "Design a system to detect hate speech in user-generated content"

### Framework

```
1. SCOPE:
   - Platform with 500M posts/day, <5s detection latency
   - Balance: safety (catch harmful content) vs freedom (avoid over-censorship)

2. ARCHITECTURE:
   Content posted → Async scoring queue → ML classifier →
   → Decision: allow / flag for review / auto-remove
   → Human review for flagged content → Feedback to model

3. MODEL APPROACH:
   - Tier 1: Keyword/regex blocklist (instant, high precision)
   - Tier 2: Fine-tuned BERT classifier (100ms, nuanced detection)
   - Tier 3: LLM judge for ambiguous cases (1-2s, highest quality)

4. KEY CHALLENGES:
   - Multilingual content → Multilingual BERT or per-language models
   - Adversarial evasion (leetspeak, Unicode tricks) → Data augmentation
   - Context-dependent (sarcasm, reclaimed language) → Context-aware models
   - Evolving hate speech patterns → Regular retraining with new examples

5. METRICS:
   - Precision (don't censor legitimate speech)
   - Recall (don't miss harmful content)
   - Latency (how fast content is moderated)
   - Human review rate (operational cost)
```

---

## Question 5: "How would you reduce inference latency from 500ms to 50ms?"

### Framework

```
Systematic optimization checklist (ordered by ease of implementation):

1. CACHING (biggest bang for buck):
   - Exact-match cache: Redis cache for repeated queries
   - Semantic cache: Embed query → find similar cached query → return cached result
   - Expected improvement: 10-100× for cache hits

2. MODEL OPTIMIZATION:
   - Quantization: FP32 → INT8 (2-4× speedup, <1% accuracy loss)
   - Distillation: Large model → small student model (2-10× speedup)
   - Pruning: Remove less important weights (1.5-3× speedup)
   - ONNX export + TensorRT optimization

3. INFRASTRUCTURE:
   - GPU inference (if currently on CPU)
   - Batch incoming requests (dynamic batching with TorchServe)
   - Async preprocessing (overlap feature lookup with other work)
   - Horizontal scaling (more replicas behind load balancer)

4. ARCHITECTURE:
   - Model cascade: cheap model first, expensive model only for uncertain cases
   - Pre-computation: pre-compute and cache embeddings for all items
   - Approximate nearest neighbor instead of exact similarity

5. NETWORK:
   - gRPC instead of REST (2-3× faster serialization)
   - Connection pooling to downstream services
   - Co-locate model service with feature store (same region/AZ)
```

---

## Question 6: "Design an anomaly detection system for server metrics"

### Framework

```
1. SCOPE:
   - Monitor 10K servers, 100+ metrics each, detect anomalies in <1 min
   - Types: point anomalies, contextual, collective

2. DATA:
   - Time series: CPU, memory, disk, network, request rate, error rate
   - Metadata: server role, deployment events, maintenance windows

3. MODELS:
   - Statistical: Z-score, EWMA, Seasonal decomposition (simple, interpretable)
   - ML: Isolation Forest, One-class SVM (handles multivariate)
   - Deep Learning: LSTM autoencoder (learns normal patterns, flags reconstruction errors)
   - Ensemble: Combine multiple detectors, alert when 2+ agree

4. KEY CHALLENGES:
   - Seasonality (weekday/weekend, business hours)
   - Deployment artifacts (legitimate changes look like anomalies)
   - Alert fatigue (too many false positives → operators ignore alerts)

5. DESIGN CHOICES:
   - Per-metric vs multivariate detection
   - Static threshold vs adaptive threshold
   - Point anomaly vs pattern anomaly
```

---

## Question 7: "How do you handle class imbalance in a classification problem?"

### Framework

```
Layered approach (ordered from simplest to most complex):

1. DATA LEVEL:
   - Oversampling minority (SMOTE, ADASYN)
   - Undersampling majority (random, Tomek links, NearMiss)
   - Data augmentation (text: back-translation, paraphrase; image: flip, rotate)

2. ALGORITHM LEVEL:
   - Class weights (sklearn: class_weight='balanced')
   - Focal loss: γ parameter down-weights easy examples
   - Cost-sensitive learning: different misclassification costs

3. EVALUATION:
   - DON'T use accuracy (misleading with imbalanced data)
   - USE: Precision-Recall AUC, F1, Recall at fixed Precision
   - Use stratified cross-validation

4. THRESHOLD TUNING:
   - Default 0.5 threshold is almost always wrong for imbalanced data
   - Use PR curve to find optimal threshold for your business need
   - Example: fraud detection → optimize for high recall (catch fraud)
     even at cost of some false positives

5. ARCHITECTURE:
   - Anomaly detection frame (one-class SVM, autoencoders)
   - Cascade: rule-based filter → ML on filtered set
   - Active learning: prioritize labeling uncertain examples
```

---

## Question 8: "Walk me through how you'd approach a brand-new ML project at a company"

### Framework

```
Week 1-2: UNDERSTAND
├── Meet stakeholders, understand the business problem
├── Define success metrics (what does "good" look like?)
├── Audit existing data: what's available? quality? biases?
├── Identify constraints: latency, cost, regulatory, team skills
└── Write 1-page design doc with scope and approach

Week 3-4: BASELINE
├── Build simplest possible baseline (heuristic or simple model)
├── Set up evaluation framework (offline metrics, test set)
├── Build data pipeline (even if manual, get data flowing)
├── Establish the baseline metric (this is your bar to beat)
└── Demo to stakeholders, get early feedback

Week 5-8: ITERATE
├── Feature engineering (biggest ROI in most ML projects)
├── Try 2-3 model architectures
├── Error analysis: where does the model fail? why?
├── Tune hyperparameters (Optuna / Bayesian optimization)
└── Iterate based on error analysis

Week 9-10: PRODUCTIONIZE
├── Build serving pipeline (FastAPI, model registry)
├── Set up A/B test framework
├── Shadow mode deployment
├── Monitoring and alerting setup
└── Documentation and handoff

Week 11-12: LAUNCH & LEARN
├── Gradual rollout (canary → 10% → 50% → 100%)
├── Monitor online metrics
├── Collect user feedback
├── Plan next iteration
└── Retrain schedule established
```

> **Rahul's Tip:** This mirrors my approach on every project. For PathWise, week 1-2 was understanding the student learning problem and defining what "personalized learning" means. The baseline was a simple keyword search over PDF content. Then I iterated to embeddings + semantic search + agentic RAG — each step validated by testing with real users.

---

## Question 9: "Compare microservices vs monolith for an ML system"

### Framework

| Aspect | Monolith | Microservices |
|--------|---------|--------------|
| Development speed | Faster initially | Faster at scale (parallel teams) |
| Deployment | All-or-nothing | Independent service deployment |
| Scaling | Scale entire app | Scale individual services |
| Debugging | Single codebase | Distributed tracing needed |
| ML-specific | Easy feature sharing | Clear model service boundaries |
| Best for | Small team, MVP/prototype | Large team, multiple models in production |

```
ML-SPECIFIC MICROSERVICES PATTERN:

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Feature   │  │  Model A │  │  Model B │  │  Model C │
│ Service   │  │ (Reco)   │  │ (Search) │  │ (Fraud)  │
│ (shared)  │  │          │  │          │  │          │
└─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │             │             │
      └─────────────┴─────────────┴─────────────┘
                          │
                    ┌─────▼────┐
                    │  API     │
                    │  Gateway │
                    └──────────┘

Each model service:
• Has its own codebase, Docker image, deployment cycle
• Can be scaled independently (GPU for model B, CPU for model A)
• Can use different frameworks (PyTorch for A, TensorFlow for B)
• Shares feature service to avoid training-serving skew
```

---

## Question 10: "What's your experience deploying ML systems to production?"

### Your Answer Template

```
"I've deployed multiple ML systems to production. Let me walk you through
my most complex deployment — PathWise:

ARCHITECTURE:
• FastAPI backend with 50+ endpoints, deployed on Railway
• Supabase (PostgreSQL + pgvector) for data persistence and vector search
• External ML services: Groq for fast LLM inference, Cohere for embeddings
• Frontend on Vercel (React/Next.js)

ML COMPONENTS:
• RAG pipeline: PDF ingestion → chunking → embedding → retrieval → generation
• Career matching: RIASEC quiz → cosine similarity → AI-enhanced roadmaps
• Multi-agent system: SummarizerAgent, DiagnosticAgent with validation and repair loops
• Knowledge distillation: LLM generates flashcards, quizzes, concept maps

PRODUCTION CONSIDERATIONS:
• Async endpoints for long-running operations (PDF processing, LLM calls)
• In-memory LRU cache with TTL for lesson content (reduces API calls)
• Structured logging with Loguru for debugging
• CORS middleware for cross-origin frontend requests
• Environment-based configuration (dev/staging/prod)

LESSONS LEARNED:
• Start with managed services (Supabase, Railway) to move fast
• Async is essential when calling external LLM APIs
• LRU caching with TTL dramatically reduces latency for repeated content
• Structured logging saves hours of debugging in production"
```

---

# 9. Key Takeaways

---

## The ML System Design Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ML SYSTEM DESIGN — INTERVIEW CHEAT SHEET                  │
│                                                                              │
│  1. ALWAYS start with requirements (5 min)                                   │
│     • Functional: What does the system do?                                   │
│     • Non-functional: Latency, throughput, availability, cost                │
│     • Constraints: Data volume, team size, regulatory                        │
│                                                                              │
│  2. ALWAYS define metrics before designing (5 min)                           │
│     • Offline: How do you evaluate during development?                       │
│     • Online: How do you measure success in production?                      │
│     • Guardrails: What must NOT degrade?                                     │
│                                                                              │
│  3. Data pipeline is 50% of the system (10 min)                              │
│     • Collection → Storage → Feature Engineering → Feature Store             │
│     • Address: freshness, consistency, training-serving skew                 │
│                                                                              │
│  4. Model is just one component (10 min)                                     │
│     • Start simple, justify complexity                                       │
│     • Discuss training strategy, validation, and offline evaluation          │
│     • Two-stage: candidate generation → ranking (for retrieval/reco)         │
│                                                                              │
│  5. Serving is where ML meets engineering (10 min)                           │
│     • Online vs batch vs streaming                                           │
│     • Optimization: caching, quantization, distillation, batching            │
│     • Deployment: shadow → canary → A/B test → full rollout                  │
│                                                                              │
│  6. ALWAYS mention monitoring (5 min)                                        │
│     • Data quality, model performance, system health, business metrics       │
│     • Drift detection → retraining trigger                                   │
│     • Feedback loops for continuous improvement                              │
│                                                                              │
│  7. Trade-offs win interviews                                                │
│     • Present at least 2 options for every major decision                    │
│     • "Given our constraints, I'd choose X because..."                       │
│     • Latency vs accuracy, cost vs performance, simple vs complex            │
│                                                                              │
│  8. Connect to YOUR experience                                               │
│     • PathWise → RAG, chatbot, deployment, agentic AI                         │
│     • Fashion Reco → embeddings, retrieval, cross-modal search               │
│     • Tesla Forecasting → feature engineering, walk-forward, XGBoost vs LSTM │
│     • Medical RAG → faithfulness, multimodal retrieval                       │
│                                                                              │
│  "The best ML system design is one where the model is the least              │
│   interesting part — because the data pipeline, feature store, serving       │
│   layer, and monitoring are all designed so well that any reasonable         │
│   model will work."                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick Reference: System Design Vocabulary

| Term | Definition |
|------|-----------|
| **Training-serving skew** | Features computed differently at training vs inference time |
| **Feature store** | Centralized platform for computing, storing, and serving features |
| **Two-tower model** | Separate encoders for query and candidate, score = dot product |
| **NDCG** | Normalized Discounted Cumulative Gain — ranking quality metric |
| **Walk-forward validation** | Time series CV: always train on past, test on future |
| **Shadow deployment** | Run new model in parallel with production, compare outputs |
| **Canary deployment** | Route small % of traffic to new model, monitor, ramp up |
| **Champion-challenger** | Always have a baseline model to compare new models against |
| **Concept drift** | The relationship between inputs and outputs changes over time |
| **Data drift** | Input feature distributions change over time |
| **Model cascade** | Chain of models: fast-cheap first, slow-expensive only if uncertain |
| **RRF (Reciprocal Rank Fusion)** | Combine ranked lists: score = Σ 1/(k + rank_i) |
| **HyDE** | Hypothetical Document Embedding — generate fake answer, embed, retrieve |
| **Lambda architecture** | Batch + speed layers for data processing |
| **Online learning** | Update model incrementally with each new data point |
| **A/B test** | Split traffic to compare model variants with statistical significance |

---

*Last updated: February 2026 | Next review: Before next interview*
*Part of Rahul Sharma's Interview Preparation Series (Document 26 of N)*
