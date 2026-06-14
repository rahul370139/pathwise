# Data Pipeline Orchestration — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Engineer / ML Engineer — Pipeline Design, Workflow Orchestration, ETL/ELT

> **Why this matters:** Every data science and ML engineering role at scale involves pipeline orchestration. Airflow dominates at big companies (Google, Airbnb, Netflix, Uber). Dagster and Prefect are winning at startups. Your experience with PySpark ETL (ATCS) and Dataiku pipelines (Jet2) gives you a foundation — this guide fills in the orchestration layer.

---

# Table of Contents

1. [What Is Pipeline Orchestration](#1-what-is-pipeline-orchestration)
2. [Apache Airflow](#2-apache-airflow)
3. [Dagster](#3-dagster)
4. [Prefect](#4-prefect)
5. [Orchestrator Comparison](#5-orchestrator-comparison)
6. [ML Pipeline Orchestration](#6-ml-pipeline-orchestration)
7. [Pipeline Design Patterns](#7-pipeline-design-patterns)
8. [Event-Driven Architectures](#8-event-driven-architectures)
9. [Interview Questions with Strong Answers](#9-interview-questions-with-strong-answers)

---

# **1. What Is Pipeline Orchestration**

## **1.1 Definition**

Orchestration is about **defining, scheduling, and monitoring** the execution of dependent tasks (DAGs) — ensuring that task B runs only after task A succeeds, handling retries on failure, and providing observability into what's running.

```
┌──────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION vs EXECUTION                       │
│                                                                   │
│  ORCHESTRATION (the conductor):                                  │
│  "Run task A, then when A succeeds, run B and C in parallel,    │
│   when both B and C succeed, run D. If B fails, retry 3 times.  │
│   If still fails, alert the on-call engineer."                   │
│                                                                   │
│  EXECUTION (the musicians):                                      │
│  The actual compute — PySpark job, Python script, dbt model,     │
│  API call, model training, etc.                                  │
│                                                                   │
│  Orchestrators don't DO the work — they coordinate WHO does      │
│  WHAT and WHEN, and handle failure gracefully.                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **1.2 DAGs (Directed Acyclic Graphs)**

Every orchestrator uses DAGs to represent task dependencies:

```
        ┌──────────┐
        │  Extract  │
        │   Data    │
        └─────┬─────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐     ┌──────────┐
│  Clean   │     │  Validate│
│  Data    │     │  Schema  │
└────┬─────┘     └─────┬────┘
     │                 │
     └────────┬────────┘
              ▼
        ┌──────────┐
        │ Transform │
        │   (dbt)   │
        └─────┬─────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐     ┌──────────┐
│  Train   │     │  Update  │
│  Model   │     │Dashboard │
└──────────┘     └──────────┘
```

**Why acyclic?** Cycles would create infinite loops. If task A depends on B and B depends on A, the DAG would never complete.

---

# **2. Apache Airflow**

## **2.1 Overview**

Created at Airbnb (2014), now an Apache top-level project. The **most widely adopted** orchestrator in the industry. Used at Google, Airbnb, Uber, Netflix, Lyft, and most data-heavy companies.

## **2.2 Core Concepts**

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **DAG** | A pipeline definition (Python file) | A recipe |
| **Task** | A single unit of work in a DAG | A step in the recipe |
| **Operator** | Template for a type of task | PythonOperator, BashOperator, SparkSubmitOperator |
| **Sensor** | Task that waits for a condition | Wait for file to appear in S3 |
| **XCom** | Cross-communication between tasks | Pass small data between tasks |
| **Executor** | How tasks are run (local, Celery, Kubernetes) | The kitchen (local stove vs commercial kitchen) |
| **Scheduler** | Determines when DAGs and tasks run | The timer |
| **Connection** | Credentials for external systems | Stored securely, referenced by ID |
| **Variable** | Global configuration values | Airflow Variables (key-value store) |
| **Pool** | Concurrency limiter | Only allow N simultaneous DB connections |

## **2.3 Architecture**

```
┌──────────────────────────────────────────────────────────────────┐
│                    AIRFLOW ARCHITECTURE                            │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │   Web    │    │Scheduler │    │ Metadata │                   │
│  │  Server  │◄──►│          │◄──►│   DB     │                   │
│  │  (UI)    │    │(triggers)│    │(Postgres)│                   │
│  └──────────┘    └─────┬────┘    └──────────┘                   │
│                        │                                         │
│                        │ sends tasks to                          │
│                        ▼                                         │
│  ┌──────────────────────────────────────────┐                   │
│  │              EXECUTOR                     │                   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐     │                   │
│  │  │Worker 1│  │Worker 2│  │Worker 3│     │                   │
│  │  └────────┘  └────────┘  └────────┘     │                   │
│  │                                          │                   │
│  │  Options:                                │                   │
│  │  • LocalExecutor (single machine)        │                   │
│  │  • CeleryExecutor (distributed workers)  │                   │
│  │  • KubernetesExecutor (pod per task)     │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **2.4 Airflow DAG Example**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="ml_training_pipeline",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training"],
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }
) as dag:

    extract = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features_from_db,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=run_great_expectations_checks,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_xgboost_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_log_metrics,
    )

    deploy = PythonOperator(
        task_id="deploy_if_better",
        python_callable=conditional_deploy,
    )

    extract >> validate >> train >> evaluate >> deploy
```

## **2.5 Airflow Strengths and Weaknesses**

| Strengths | Weaknesses |
|-----------|------------|
| Massive ecosystem (1000+ operators) | DAGs are defined at parse time (not runtime dynamic) |
| Battle-tested at massive scale | Heavy infrastructure (needs DB, scheduler, workers) |
| Strong scheduling (cron, data-aware) | Testing DAGs locally is awkward |
| Rich UI for monitoring | XComs are limited (small data only) |
| Managed options (MWAA, Cloud Composer, Astronomer) | Steep learning curve for advanced patterns |

---

# **3. Dagster**

## **3.1 Overview**

Dagster is a **data-asset-centric** orchestrator. Instead of thinking "what tasks should I run?" (Airflow), Dagster thinks **"what data assets should I produce?"** This is a paradigm shift.

## **3.2 Core Concepts**

| Concept | What It Is | Airflow Equivalent |
|---------|-----------|-------------------|
| **Asset** | A data artifact the pipeline produces | No direct equivalent (Airflow is task-centric) |
| **Op** | A computation (function) | Task/Operator |
| **Job** | A collection of ops/assets to execute | DAG |
| **Resource** | External system connection | Connection |
| **IO Manager** | How assets are stored/loaded | Custom XCom + storage logic |
| **Sensor** | Triggers based on external events | Same concept |
| **Schedule** | Time-based triggers | Same concept |
| **Partition** | Data divided by time/category | Like Airflow execution_date but more flexible |

## **3.3 Asset-Centric vs Task-Centric**

```
AIRFLOW (Task-centric):              DAGSTER (Asset-centric):
"Run these tasks in order"           "Produce these data assets"

┌───────────┐                        ┌───────────────┐
│ extract   │                        │ raw_events    │  (S3 parquet)
│ (task)    │                        │ (asset)       │
└─────┬─────┘                        └───────┬───────┘
      │                                      │
      ▼                                      ▼
┌───────────┐                        ┌───────────────┐
│ transform │                        │ clean_events  │  (Snowflake table)
│ (task)    │                        │ (asset)       │
└─────┬─────┘                        └───────┬───────┘
      │                                      │
      ▼                                      ▼
┌───────────┐                        ┌───────────────┐
│ load      │                        │ daily_metrics │  (Snowflake table)
│ (task)    │                        │ (asset)       │
└───────────┘                        └───────────────┘

Airflow asks: "Did the task run?"     Dagster asks: "Is the data fresh?"
```

## **3.4 Dagster Example**

```python
from dagster import asset, Definitions, MaterializeResult

@asset(description="Raw events from production database")
def raw_events(context):
    df = extract_from_postgres("events")
    context.log.info(f"Extracted {len(df)} events")
    return df

@asset(description="Cleaned and validated events")
def clean_events(raw_events):  # dependency declared via argument name
    df = raw_events.dropna(subset=["user_id", "event_type"])
    df = df[df["event_type"].isin(VALID_TYPES)]
    return df

@asset(description="Daily aggregated metrics for dashboard")
def daily_metrics(clean_events):
    return clean_events.groupby("date").agg(
        total_events=("event_id", "count"),
        unique_users=("user_id", "nunique"),
    )

defs = Definitions(assets=[raw_events, clean_events, daily_metrics])
```

**Notice:** Dependencies are declared through function arguments — `clean_events` takes `raw_events` as input, so Dagster automatically knows the dependency.

## **3.5 Dagster Strengths and Weaknesses**

| Strengths | Weaknesses |
|-----------|------------|
| Asset-centric (matches how data teams think) | Smaller ecosystem than Airflow |
| Excellent local development and testing | Newer — less battle-tested at extreme scale |
| Type system and data validation built in | Steeper paradigm shift for Airflow users |
| Software-defined assets with lineage | Fewer managed hosting options |
| Partition support (time, category) | Smaller community (but growing fast) |

---

# **4. Prefect**

## **4.1 Overview**

Prefect positions itself as **"Airflow, but simpler."** It uses a Pythonic, decorator-based API with minimal boilerplate. Strong focus on developer experience and ease of deployment.

## **4.2 Core Concepts**

| Concept | What It Is | Airflow Equivalent |
|---------|-----------|-------------------|
| **Flow** | A pipeline (decorated Python function) | DAG |
| **Task** | A unit of work (decorated function) | Task/Operator |
| **Deployment** | How/where/when a flow runs | DAG schedule + infra config |
| **Work Pool** | Where tasks execute | Executor |
| **Block** | Reusable configuration (credentials, etc.) | Connection + Variable |
| **Artifact** | Tracked result from a task | XCom (but richer) |

## **4.3 Prefect Example**

```python
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

@task(retries=3, retry_delay_seconds=60)
def extract_data():
    return query_database("SELECT * FROM events WHERE date = CURRENT_DATE")

@task
def validate(df):
    assert len(df) > 0, "No data extracted!"
    assert df["user_id"].notna().all(), "Null user_ids found"
    return df

@task
def train_model(df):
    model = XGBClassifier().fit(df[features], df[target])
    return model

@flow(name="daily-training", log_prints=True)
def training_pipeline():
    data = extract_data()
    clean_data = validate(data)
    model = train_model(clean_data)
    print(f"Model trained with {len(clean_data)} samples")

if __name__ == "__main__":
    training_pipeline()  # runs locally like any Python script
```

**Key difference from Airflow:** The flow is just a Python function. You can run it locally with `python pipeline.py`, debug with a debugger, and test with pytest. No special infrastructure needed for development.

## **4.4 Prefect Strengths and Weaknesses**

| Strengths | Weaknesses |
|-----------|------------|
| Extremely Pythonic and simple | Smaller ecosystem than Airflow |
| Run locally like normal Python | Less established at very large scale |
| Dynamic flows (runtime decisions) | Prefect Cloud is paid for team features |
| Excellent error handling and retries | Less granular scheduling than Airflow |
| Quick to set up and deploy | Fewer enterprise compliance features |

---

# **5. Orchestrator Comparison**

## **5.1 Head-to-Head**

| Feature | Airflow | Dagster | Prefect |
|---------|---------|---------|---------|
| **Paradigm** | Task-centric (DAGs) | Asset-centric | Flow-centric (Pythonic) |
| **Language** | Python | Python | Python |
| **Scheduling** | Cron, data-aware, timetables | Schedules, sensors, lazy auto-materialize | Schedules, event-driven |
| **Local dev** | Awkward (need full setup) | Excellent (`dagster dev`) | Excellent (just run Python) |
| **Testing** | Difficult | Built-in test utilities | Standard pytest |
| **UI** | Good (task status, logs, Gantt) | Great (asset lineage, catalog) | Good (flow runs, logs) |
| **Scale** | Proven at massive scale | Growing, used at mid-large companies | Growing, mid-scale proven |
| **Managed** | MWAA, Cloud Composer, Astronomer | Dagster Cloud | Prefect Cloud |
| **Community** | Largest | Growing fast | Medium |
| **Learning curve** | Steep | Medium (new paradigm) | Gentle |
| **Best for** | Large orgs, complex schedules | Data teams, asset lineage | Startups, quick iteration |

## **5.2 Decision Guide**

```
Which orchestrator should I use?

├── Large enterprise with existing Airflow?
│   └── Stay with Airflow. Migration cost > benefit.
│
├── New data platform, care about data lineage?
│   └── Dagster. Asset-centric model is future-forward.
│
├── Small team, want to ship fast?
│   └── Prefect. Lowest friction, most Pythonic.
│
├── ML-heavy pipeline with experiments?
│   └── Consider Dagster (partitions, IO managers) or
│       specialized tools (Kubeflow, MLflow + Airflow).
│
└── Just need cron + retries, nothing fancy?
    └── Prefect (or even a simple cron + Python script).
```

---

# **6. ML Pipeline Orchestration**

## **6.1 ML-Specific Orchestration Challenges**

| Challenge | Why It's Hard | Solution |
|-----------|-------------|----------|
| **Data + code versioning** | Model depends on both data version and code version | DVC + Git, or MLflow tracking |
| **GPU scheduling** | Training tasks need GPU, most tasks don't | KubernetesExecutor with GPU node pools |
| **Long-running tasks** | Training can take hours/days | Async sensors, spot instance handling |
| **Experiment tracking** | Multiple experiments running with different params | MLflow/W&B integration within orchestrator |
| **Conditional deployment** | Deploy only if new model is better | Branching logic based on eval metrics |
| **Data drift monitoring** | Need to retrigger training when data drifts | Sensors monitoring drift metrics |

## **6.2 ML Pipeline Pattern (Airflow)**

```
┌──────────────────────────────────────────────────────────────────┐
│               ML PIPELINE — COMMON DAG STRUCTURE                  │
│                                                                   │
│  ┌────────────┐                                                  │
│  │ Data       │ ← Sensor waits for new data in S3               │
│  │ Ingestion  │                                                  │
│  └─────┬──────┘                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌────────────┐   ┌────────────┐                                │
│  │ Feature    │   │ Data       │ ← Parallel                     │
│  │ Engineering│   │ Validation │   (Great Expectations)          │
│  └─────┬──────┘   └─────┬──────┘                                │
│        │                 │                                        │
│        └────────┬────────┘                                       │
│                 ▼                                                 │
│  ┌────────────────────┐                                          │
│  │   Model Training   │ ← GPU task (KubernetesExecutor)         │
│  │   (with MLflow)    │                                          │
│  └──────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│  ┌────────────────────┐                                          │
│  │   Model Evaluation │ ← Compare vs production model           │
│  └──────────┬─────────┘                                          │
│             │                                                    │
│    ┌────────┴────────┐                                           │
│    ▼                 ▼                                            │
│  Better?           Not better?                                   │
│    │                 │                                            │
│    ▼                 ▼                                            │
│  ┌──────────┐    ┌──────────┐                                   │
│  │  Deploy  │    │  Log &   │                                   │
│  │  Model   │    │  Alert   │                                   │
│  └──────────┘    └──────────┘                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **6.3 ML-Specific Orchestration Tools**

| Tool | Focus | Relationship to General Orchestrators |
|------|-------|--------------------------------------|
| **Kubeflow Pipelines** | ML pipelines on Kubernetes | Full ML pipeline framework (training, serving, experiments) |
| **MLflow** | Experiment tracking, model registry | Complements orchestrators (Airflow + MLflow is common) |
| **ZenML** | MLOps pipeline framework | Orchestrator-agnostic (can use Airflow, Kubeflow as backend) |
| **Metaflow** (Netflix) | Data science pipelines | Python-centric, similar philosophy to Prefect |

---

# **7. Pipeline Design Patterns**

## **7.1 ETL vs ELT**

```
ETL (Extract → Transform → Load):
Source → Transform in Python/Spark → Load to DW
Best for: Complex transformations, data cleaning, legacy systems

ELT (Extract → Load → Transform):
Source → Load raw to DW → Transform with dbt/SQL in DW
Best for: Modern cloud DWs (Snowflake, BigQuery) that are cheap and fast
```

| Aspect | ETL | ELT |
|--------|-----|-----|
| **Transform where** | In pipeline (Python/Spark) | In warehouse (SQL/dbt) |
| **Best for** | Complex logic, unstructured data | Structured data, SQL-friendly transforms |
| **Scalability** | Depends on compute cluster | Leverages DW compute (often cheaper) |
| **Modern trend** | Decreasing | Increasing (dbt revolution) |

## **7.2 Idempotent Pipelines**

Running the same pipeline twice should produce the same result (not duplicate data):

| Strategy | How |
|----------|-----|
| **DELETE + INSERT** | Delete today's partition, then insert fresh data |
| **MERGE / UPSERT** | Update existing rows, insert new ones |
| **Partition overwrite** | Write entire partition, replacing previous |
| **Deduplication key** | Use unique key to prevent duplicates |

## **7.3 Backfill Pattern**

Re-process historical data when logic changes:

```
# Airflow: backfill the last 30 days
airflow dags backfill ml_training_pipeline \
    --start-date 2026-03-01 \
    --end-date 2026-03-31
```

**Key concept:** Your pipeline must be parameterized by date/partition so that backfills work. Don't hardcode `today()` — use the orchestrator's execution date.

## **7.4 Data Quality Checks (Inline)**

| Check | Tool | When |
|-------|------|------|
| **Schema validation** | Great Expectations, Pandera | After extraction, before transform |
| **Null/completeness** | Custom SQL/Python checks | After each stage |
| **Freshness** | dbt tests, custom sensors | Before pipeline starts |
| **Distribution drift** | Evidently, custom stats | Before training |
| **Row count sanity** | Simple assertions | After each stage |

---

# **8. Event-Driven Architectures**

## **8.1 Schedule-Driven vs Event-Driven**

| Approach | How | Best For |
|----------|-----|----------|
| **Schedule-driven** | Run at fixed intervals (every hour, daily) | Batch processing, regular reports |
| **Event-driven** | Trigger when event occurs (file lands, API called) | Real-time processing, streaming |
| **Hybrid** | Schedule with event-based short-circuits | Most production ML pipelines |

## **8.2 Event Sources for ML Pipelines**

| Event | Trigger | Action |
|-------|---------|--------|
| New data file in S3 | S3 event notification → SQS → Airflow sensor | Start ingestion pipeline |
| Model drift detected | Monitoring alert → webhook | Trigger retraining pipeline |
| New model registered | MLflow webhook | Trigger evaluation + deployment pipeline |
| API error rate spike | CloudWatch alarm → EventBridge | Trigger incident + rollback pipeline |
| Feature store updated | Feature store event | Trigger dependent model retraining |

## **8.3 Message Queues and Streaming**

| Tool | Type | Use Case |
|------|------|----------|
| **Apache Kafka** | Distributed event streaming | Real-time event pipelines, high throughput |
| **AWS SQS** | Managed message queue | Simple async task queues |
| **AWS Kinesis** | Managed streaming | Real-time data streaming (lighter than Kafka) |
| **Redis Streams** | Lightweight streaming | Low-latency event processing |
| **Pub/Sub (GCP)** | Managed messaging | Event-driven architectures on GCP |

---

# **9. Interview Questions with Strong Answers**

---

## **Q1: "Compare Airflow, Dagster, and Prefect. Which would you choose for a new ML platform?"**

> They represent three paradigms:
>
> **Airflow** is task-centric — you define what to run and when. It has the largest ecosystem (1000+ operators), proven scale (Google, Netflix, Uber), and the most managed options. But it's heavy to set up, hard to test locally, and its DAG model makes it awkward for data-asset-focused workflows.
>
> **Dagster** is asset-centric — you define what data to produce, and the orchestrator figures out what tasks to run. This matches how data teams actually think ("is my daily_metrics table fresh?"). It has great lineage, testing, and partition support. Ideal for data-heavy teams.
>
> **Prefect** is flow-centric — it's the most Pythonic. Flows are just decorated functions you can run locally, debug with a debugger, and test with pytest. Lowest friction, fastest to get started. Best for small teams and startups.
>
> For a **new ML platform,** I'd choose:
> - **Dagster** if data lineage and asset management are priorities (data-intensive ML)
> - **Airflow** if integrating with an existing enterprise ecosystem with many external systems
> - **Prefect** if the team is small and needs to ship fast

---

## **Q2: "What is idempotency in pipelines and why does it matter?"**

> An idempotent pipeline produces the same result whether you run it once or ten times. If I accidentally trigger a pipeline twice, or need to rerun after a partial failure, I shouldn't get duplicate data.
>
> Common strategies: partition-level overwrite (delete the day's data, then rewrite it), upsert/merge (update existing rows by key, insert new ones), or deduplication at load time.
>
> Why it matters: in production, pipelines fail midway and get retried. Without idempotency, a retry after partial failure creates data quality issues — duplicate rows, inflated metrics, incorrect model training data.
>
> At Jet2, our ETL pipeline was idempotent by design — each daily run would overwrite the day's partition in the data warehouse, so retries were safe.

---

## **Q3: "How would you handle a pipeline that takes 6 hours to train a model but the data changes hourly?"**

> Separate the data pipeline from the training pipeline:
>
> **Hourly pipeline:** Extracts and transforms new data, appends to feature store, runs data quality checks. Lightweight, runs every hour.
>
> **Daily training pipeline:** Triggered once daily (or when drift is detected). Reads the accumulated features from the feature store, trains the model, evaluates against the production model, deploys if better.
>
> The key insight: not every data update needs a model retrain. Define **retraining triggers**: scheduled (daily), metric-based (accuracy drops below threshold), or data-based (distribution shift > threshold). Use a sensor or event to trigger retraining only when needed.
>
> For the 6-hour training job specifically: use spot instances to reduce cost, implement checkpointing so training can resume after interruption, and use async monitoring (Airflow sensor polling for job completion).

---

## **Q4: "Explain the difference between ETL and ELT. When would you use each?"**

> **ETL** (Extract-Transform-Load): Data is transformed *before* loading into the warehouse. The transformation happens in the pipeline compute (Python, Spark). Best for complex transformations on unstructured data, or when the warehouse doesn't support the computation.
>
> **ELT** (Extract-Load-Transform): Data is loaded raw into the warehouse *first*, then transformed using SQL/dbt within the warehouse. Best for modern cloud data warehouses (Snowflake, BigQuery) that are massively parallel and cheap to compute.
>
> The industry trend is toward ELT because modern warehouses decouple compute from storage, making it cheap to load everything and transform in place. dbt has become the standard for the "T" in ELT.
>
> **My practice:** At ATCS, we used **ETL** with PySpark because the transformation involved complex NLP processing (document classification) that required Python libraries — not doable in SQL. For simpler analytics pipelines, I'd use ELT with dbt.

---

## **Q5: "You notice that your daily pipeline has been silently producing wrong data for the past week. How do you fix it?"**

> 1. **Triage:** How wrong? Is the data completely missing, partially correct, or subtly incorrect? Check downstream consumers — are models/dashboards affected?
>
> 2. **Root cause:** Check the pipeline logs for the past week. What changed? New data source schema? Code deployment? Infrastructure change? Data quality issue upstream?
>
> 3. **Fix the bug:** Deploy the corrected pipeline code.
>
> 4. **Backfill:** Run the pipeline for each affected day. Because pipelines should be idempotent and date-parameterized, this is just `airflow dags backfill --start-date 2026-03-28 --end-date 2026-04-05`.
>
> 5. **Prevent recurrence:**
>    - Add data quality checks (Great Expectations) that would have caught this
>    - Add alerting on key metrics (row counts, null rates, distribution stats)
>    - Add integration tests that validate the pipeline output against expected schema
>
> The meta-lesson: silent failures are worse than loud failures. Invest in data quality checks and monitoring so you catch issues in hours, not weeks.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│          DATA PIPELINE ORCHESTRATION TAKEAWAYS                    │
│                                                                   │
│  1. Orchestrators coordinate WHEN and HOW tasks run, not the    │
│     tasks themselves — understand the separation                 │
│                                                                   │
│  2. Airflow = industry standard (task-centric, massive          │
│     ecosystem). Dagster = modern alternative (asset-centric).   │
│     Prefect = simplest (most Pythonic).                          │
│                                                                   │
│  3. IDEMPOTENCY is non-negotiable — pipelines must be safe     │
│     to re-run (partition overwrite, upsert, dedup)              │
│                                                                   │
│  4. ETL for complex transforms (NLP, CV). ELT for analytics    │
│     (dbt + modern warehouse).                                    │
│                                                                   │
│  5. ML pipelines have unique challenges: GPU scheduling,        │
│     long training jobs, conditional deployment, drift triggers  │
│                                                                   │
│  6. Data quality checks at EVERY stage (not just at the end)   │
│                                                                   │
│  7. Event-driven > pure schedule-driven for reactive pipelines  │
│     (drift detection → retrain, new data → ingest)             │
│                                                                   │
│  8. Backfill capability = pipelines must be parameterized      │
│     by date/partition, never hardcode `today()`                 │
└──────────────────────────────────────────────────────────────────┘
```
