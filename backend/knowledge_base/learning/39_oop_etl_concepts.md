# FusionSpan Interview Prep — OOP & ETL/Data Engineering Concepts

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, University of Maryland (4.0 GPA)  
**Target Role:** Data Engineer / Data Integration Specialist — FusionSpan  
**Focus Areas:** Python OOP, ETL/ELT Pipelines, SQL, Cloud Data Platforms, REST APIs, Agile  
**Why This Document:** Covers the core conceptual topics FusionSpan will assess — OOP fundamentals, ETL architecture, cloud warehousing, and API design — in a way you can speak about fluently.

---

## Table of Contents

1. [Python OOP Concepts](#1-python-oop-concepts)
2. [Why Python? — How to Answer](#2-why-python--how-to-answer)
3. [ETL / ELT Concepts](#3-etl--elt-concepts)
4. [Data Integration Patterns](#4-data-integration-patterns)
5. [Cloud Data Platforms (Snowflake, AWS, GCP)](#5-cloud-data-platforms)
6. [REST APIs — Build & Consume](#6-rest-apis--build--consume)
7. [SQL Essentials Cheat Sheet](#7-sql-essentials-cheat-sheet)
8. [ML/AI Quick Reference](#8-mlai-quick-reference)
9. [Agile/Scrum Essentials](#9-agilescrum-essentials)
10. [FusionSpan — Company & Role Fit](#10-fusionspan--company--role-fit)
11. [Interview Talking Points — Your Projects](#11-interview-talking-points--your-projects)

---

# 1. Python OOP Concepts

OOP is about modeling your code as **objects** that bundle data (attributes) and behavior (methods). Even if the role is data engineering, interviewers ask OOP to test your software design maturity.

---

## 1.1 The Four Pillars of OOP

```
┌─────────────────────────────────────────────────────────────────┐
│                    FOUR PILLARS OF OOP                            │
├────────────────┬────────────────┬───────────────┬───────────────┤
│  Encapsulation │  Abstraction   │  Inheritance  │ Polymorphism  │
│                │                │               │               │
│  Bundle data + │  Hide complex  │  Reuse code   │ Same interface│
│  methods;      │  internals;    │  via parent   │ different     │
│  control       │  expose only   │  classes;     │ behavior;     │
│  access        │  what matters  │  IS-A         │ duck typing   │
└────────────────┴────────────────┴───────────────┴───────────────┘
```

---

## 1.2 Classes, Objects, and `__init__`

A **class** is a blueprint. An **object** is an instance of that blueprint.

```python
class ETLPipeline:
    """Encapsulates a complete ETL workflow."""

    pipeline_count = 0  # class variable — shared across all instances

    def __init__(self, name: str, source: str, destination: str):
        # Instance variables — unique per object
        self.name = name
        self.source = source
        self.destination = destination
        self._records_processed = 0       # protected by convention
        self.__connection_string = None   # name-mangled private
        ETLPipeline.pipeline_count += 1

    def extract(self) -> list[dict]:
        """Pull data from source."""
        raw_data = self._read_source()
        self._records_processed = len(raw_data)
        return raw_data

    def transform(self, data: list[dict]) -> list[dict]:
        """Clean, validate, enrich."""
        return [self._clean_record(r) for r in data if self._is_valid(r)]

    def load(self, data: list[dict]) -> int:
        """Write to destination."""
        rows_loaded = self._write_destination(data)
        return rows_loaded

    def run(self) -> dict:
        """Execute full ETL flow."""
        raw = self.extract()
        clean = self.transform(raw)
        loaded = self.load(clean)
        return {"extracted": len(raw), "loaded": loaded}

    def __repr__(self) -> str:
        return f"ETLPipeline('{self.name}', {self.source} → {self.destination})"

    def _read_source(self):
        """Protected method — internal implementation."""
        ...

    def _clean_record(self, record):
        ...

    def _is_valid(self, record):
        ...

    def _write_destination(self, data):
        ...
```

### Key Concepts to Explain

| Concept | What It Is | Python Mechanism |
|---------|-----------|-----------------|
| `self` | Reference to the current instance | Passed automatically as first arg |
| `__init__` | Constructor — initializes instance state | Called when `ETLPipeline(...)` is invoked |
| Class variable | Shared across ALL instances | `pipeline_count` — lives on the class |
| Instance variable | Unique per object | `self.name` — lives on the instance |
| `_name` | Protected (convention) | Signals "internal use" — not enforced |
| `__name` | Private (name-mangled) | Python rewrites to `_ClassName__name` |

---

## 1.3 Inheritance — Code Reuse via IS-A

Inheritance lets a child class **reuse and extend** a parent class.

```python
class FileExtractor:
    """Base extractor for any file-based source."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def validate_path(self) -> bool:
        import os
        return os.path.exists(self.filepath)

    def extract(self) -> list[dict]:
        raise NotImplementedError("Subclasses must implement extract()")


class CSVExtractor(FileExtractor):
    """Extracts data from CSV files."""

    def __init__(self, filepath: str, delimiter: str = ","):
        super().__init__(filepath)  # call parent __init__
        self.delimiter = delimiter

    def extract(self) -> list[dict]:
        import csv
        with open(self.filepath) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            return list(reader)


class JSONExtractor(FileExtractor):
    """Extracts data from JSON files."""

    def extract(self) -> list[dict]:
        import json
        with open(self.filepath) as f:
            return json.load(f)


class ParquetExtractor(FileExtractor):
    """Extracts data from Parquet files."""

    def extract(self) -> list[dict]:
        import pandas as pd
        df = pd.read_parquet(self.filepath)
        return df.to_dict(orient="records")
```

**Interview answer for "Why inheritance?":**
> "Inheritance lets me define common behavior once in a base class and specialize it in subclasses. In my ETL work, all file extractors share path validation logic, but CSV, JSON, and Parquet each have different parsing. The base class enforces the contract (`extract()` must be implemented), and each child handles its own format. This is the Open/Closed Principle — open for extension, closed for modification."

---

## 1.4 Polymorphism — Same Interface, Different Behavior

Polymorphism means you can treat different object types through the same interface.

```python
def run_extraction(extractor: FileExtractor) -> list[dict]:
    """Works with ANY extractor subclass — polymorphism."""
    if not extractor.validate_path():
        raise FileNotFoundError(f"Path not found: {extractor.filepath}")
    return extractor.extract()

# All three work through the same function
csv_data = run_extraction(CSVExtractor("data.csv"))
json_data = run_extraction(JSONExtractor("data.json"))
parquet_data = run_extraction(ParquetExtractor("data.parquet"))
```

**Python's duck typing:** "If it walks like a duck and quacks like a duck, it's a duck." Python doesn't require formal inheritance for polymorphism — any object with an `extract()` method works.

---

## 1.5 Encapsulation — Controlled Access

```python
class DatabaseConnection:
    """Encapsulates connection details and lifecycle."""

    def __init__(self, host: str, port: int, database: str):
        self._host = host
        self._port = port
        self._database = database
        self.__password = None  # truly private
        self._connection = None

    @property
    def is_connected(self) -> bool:
        """Read-only computed property."""
        return self._connection is not None

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self._host}:{self._port}/{self._database}"

    def connect(self):
        if self._connection is None:
            self._connection = self._create_connection()
        return self._connection

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def _create_connection(self):
        """Internal — users shouldn't call this directly."""
        ...
```

**Why encapsulation matters:** The caller doesn't need to know HOW the connection is created — they just call `connect()` and `disconnect()`. Internal state (`_connection`, `__password`) is hidden. The `@property` decorator exposes `is_connected` as a read-only attribute.

---

## 1.6 Abstraction — Abstract Base Classes

```python
from abc import ABC, abstractmethod

class DataTransformer(ABC):
    """Abstract base — defines the contract all transformers must follow."""

    @abstractmethod
    def transform(self, data: list[dict]) -> list[dict]:
        """Every transformer MUST implement this."""
        ...

    def validate(self, data: list[dict]) -> bool:
        """Concrete method — shared default behavior."""
        return len(data) > 0


class NullCleaner(DataTransformer):
    def transform(self, data: list[dict]) -> list[dict]:
        return [{k: v for k, v in row.items() if v is not None} for row in data]


class TypeCaster(DataTransformer):
    def __init__(self, schema: dict[str, type]):
        self.schema = schema

    def transform(self, data: list[dict]) -> list[dict]:
        result = []
        for row in data:
            casted = {}
            for k, v in row.items():
                if k in self.schema:
                    casted[k] = self.schema[k](v)
                else:
                    casted[k] = v
            result.append(casted)
        return result

# Cannot instantiate ABC directly
# t = DataTransformer()  # TypeError!

# But concrete subclasses work
cleaner = NullCleaner()
caster = TypeCaster({"age": int, "salary": float})
```

---

## 1.7 Composition vs Inheritance — When to Use Each

```python
class ETLJob:
    """Uses COMPOSITION — has-a extractor, transformer, loader."""

    def __init__(self, extractor, transformers: list, loader):
        self.extractor = extractor          # HAS-A relationship
        self.transformers = transformers    # List of transformer objects
        self.loader = loader

    def execute(self):
        data = self.extractor.extract()
        for transformer in self.transformers:
            data = transformer.transform(data)
        return self.loader.load(data)

# Flexible — mix and match any extractor + transformers + loader
job = ETLJob(
    extractor=CSVExtractor("sales.csv"),
    transformers=[NullCleaner(), TypeCaster({"amount": float})],
    loader=SnowflakeLoader(table="sales_clean")
)
job.execute()
```

| Use | Inheritance (IS-A) | Composition (HAS-A) |
|-----|-------------------|---------------------|
| When | Clear "is a type of" relationship | "Uses" or "contains" relationship |
| Example | CSVExtractor IS-A FileExtractor | ETLJob HAS-A extractor |
| Flexibility | Rigid hierarchy | Swap components at runtime |
| Coupling | Tight (child depends on parent) | Loose (components are independent) |
| Rule of thumb | **Prefer composition** over inheritance |

---

## 1.8 Design Patterns in Data Engineering

### Factory Pattern — Creating extractors dynamically

```python
class ExtractorFactory:
    """Creates the right extractor based on file extension."""

    _registry = {
        ".csv": CSVExtractor,
        ".json": JSONExtractor,
        ".parquet": ParquetExtractor,
    }

    @classmethod
    def create(cls, filepath: str) -> FileExtractor:
        ext = "." + filepath.rsplit(".", 1)[-1].lower()
        extractor_cls = cls._registry.get(ext)
        if not extractor_cls:
            raise ValueError(f"Unsupported format: {ext}")
        return extractor_cls(filepath)

# Usage
extractor = ExtractorFactory.create("data.parquet")  # returns ParquetExtractor
```

### Singleton Pattern — One database connection pool

```python
class ConnectionPool:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_connections: int = 5):
        if not hasattr(self, "_initialized"):
            self._pool = []
            self._max = max_connections
            self._initialized = True
```

---

## 1.9 OOP Interview Quick Answers

**"What are the four pillars of OOP?"**
> Encapsulation (bundle data + methods, control access), Abstraction (hide complexity, expose interface), Inheritance (code reuse via parent-child), Polymorphism (same interface, different behavior).

**"Explain the difference between a class and an object."**
> A class is a blueprint (like `ETLPipeline`); an object is a concrete instance (`my_pipeline = ETLPipeline("daily_sales", ...)`). You can create many objects from one class.

**"When would you use inheritance vs composition?"**
> Inheritance when there's a clear IS-A relationship (CSVExtractor IS-A FileExtractor). Composition when an object USES another (ETLJob HAS-A extractor). I default to composition because it's more flexible and loosely coupled.

**"What is `super()` in Python?"**
> `super()` gives access to the parent class's methods. In `__init__`, `super().__init__(...)` calls the parent's constructor so the child doesn't have to reimplement shared initialization. It follows the MRO (Method Resolution Order) for correct behavior in multiple inheritance.

---

# 2. Why Python? — How to Answer

> **"Why did you choose Python for your projects?"**

**Framework answer:**

"I chose Python for three reasons specific to data engineering and ML:

1. **Ecosystem depth** — PySpark for distributed ETL, Pandas for local data manipulation, SQLAlchemy for database interaction, FastAPI for building APIs, and scikit-learn/PyTorch for ML. No other language has this breadth for data work.

2. **Readability and velocity** — Data pipelines evolve constantly. Python's clean syntax means I can iterate on transformation logic quickly and hand off pipelines to teammates who understand them immediately. In my ATCS work, PySpark ETL pipelines I wrote were maintained by the team long after I left — because they were readable.

3. **Interoperability** — Python connects everything. I've used it to extract from REST APIs (`requests`/`httpx`), read Parquet/CSV/JSON, transform with PySpark or Pandas, and load into Snowflake, MSSQL, and PostgreSQL. It's the glue language for modern data stacks.

For performance-critical paths, Python delegates to optimized C/Rust backends — NumPy, Spark's JVM engine, Pydantic v2's Rust core. So you get developer velocity without sacrificing runtime performance where it matters."

---

# 3. ETL / ELT Concepts

---

## 3.1 What is ETL?

**ETL = Extract, Transform, Load** — the process of moving data from source systems to a target system (data warehouse, database, data lake).

```
┌──────────────────────────────────────────────────────────────────┐
│                         ETL Pipeline                              │
│                                                                  │
│  ┌──────────┐    ┌──────────────────┐    ┌─────────────────┐    │
│  │ EXTRACT  │───►│    TRANSFORM     │───►│      LOAD       │    │
│  │          │    │                  │    │                 │    │
│  │ Sources: │    │ • Clean nulls    │    │ Targets:        │    │
│  │ • CSV    │    │ • Deduplicate    │    │ • Snowflake     │    │
│  │ • JSON   │    │ • Type cast      │    │ • PostgreSQL    │    │
│  │ • APIs   │    │ • Validate       │    │ • Data Lake     │    │
│  │ • DBs    │    │ • Enrich/join    │    │ • MSSQL         │    │
│  │ • Kafka  │    │ • Normalize      │    │ • S3/GCS        │    │
│  │ • S3     │    │ • Aggregate      │    │                 │    │
│  └──────────┘    └──────────────────┘    └─────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3.2 ETL vs ELT — Know the Difference

```
ETL: Source → [Transform on ETL server] → Load into Warehouse
ELT: Source → Load raw into Warehouse → [Transform INSIDE warehouse]
```

| Aspect | ETL | ELT |
|--------|-----|-----|
| **Transform location** | External server (Python, Spark) | Inside the warehouse (SQL, dbt) |
| **When to use** | Complex transformations, unstructured data, data cleansing before loading | Warehouse has strong compute (Snowflake, BigQuery) |
| **Data preservation** | Only transformed data lands in warehouse | Raw data preserved — transform later as needs evolve |
| **Tools** | PySpark, Python scripts, Informatica, Talend | dbt, Snowflake Tasks, BigQuery scheduled queries |
| **Schema** | Schema-on-write (define upfront) | Schema-on-read (flexible) |
| **Modern trend** | Still used for heavy preprocessing | **Dominant** in cloud-native architectures |
| **Your experience** | PySpark ETL at ATCS, Python scripts | Can discuss both approaches |

**Interview answer:**
> "Modern stacks are converging on ELT because cloud warehouses like Snowflake have massive compute. But ETL is still necessary when data needs significant cleansing before it's useful — parsing unstructured logs, masking PII, or standardizing formats from heterogeneous sources. In my ATCS work, I used ETL because we needed PySpark for heavy data quality validation and type coercion before loading to MSSQL."

---

## 3.3 The Three Stages in Detail

### EXTRACT — Getting Data Out

| Source Type | Method | Python Tools |
|-------------|--------|-------------|
| **Flat files** (CSV, JSON, Parquet) | File read | `pandas`, `pyspark.read`, `csv`, `json` |
| **Databases** | SQL query via JDBC/ODBC | `SQLAlchemy`, `psycopg2`, `pyodbc`, PySpark JDBC |
| **REST APIs** | HTTP GET/POST | `requests`, `httpx`, `aiohttp` |
| **Message queues** | Subscribe/consume | `kafka-python`, `confluent-kafka` |
| **Cloud storage** | SDK download | `boto3` (S3), `google-cloud-storage` (GCS) |
| **Web scraping** | HTML parsing | `beautifulsoup4`, `scrapy` |

```python
# Extract from CSV
import pandas as pd
df = pd.read_csv("sales_2024.csv", dtype={"customer_id": str})

# Extract from API
import requests
response = requests.get("https://api.example.com/orders", params={"date": "2024-01-01"})
orders = response.json()

# Extract from database
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("SELECT * FROM orders WHERE date > '2024-01-01'", engine)

# Extract from S3 with PySpark
df = spark.read.parquet("s3://data-lake/orders/year=2024/")
```

### TRANSFORM — Cleaning and Enriching

Common transformations you should be able to discuss:

| Transformation | What It Does | Example |
|---------------|-------------|---------|
| **Null handling** | Fill, drop, or flag missing values | `df.fillna({"amount": 0})` |
| **Deduplication** | Remove duplicate records | `df.drop_duplicates(subset=["order_id"])` |
| **Type casting** | Convert strings to proper types | `df["date"] = pd.to_datetime(df["date"])` |
| **Schema validation** | Ensure required columns exist with correct types | Custom validation framework |
| **Data quality checks** | Completeness, uniqueness, freshness | Assert row counts, null rates |
| **Normalization** | Standardize formats (dates, phone numbers, addresses) | Parse and reformat |
| **Enrichment** | Join with reference data | Join customer table to add region |
| **Aggregation** | Summarize at different granularities | Daily → monthly rollup |
| **Filtering** | Remove invalid/irrelevant records | Drop rows where amount < 0 |
| **Derived columns** | Create new features from existing | `profit = revenue - cost` |

```python
# PySpark transformation example (from your ATCS experience)
from pyspark.sql import functions as F

cleaned = (
    raw_df
    .filter(F.col("amount").isNotNull())
    .filter(F.col("amount") > 0)
    .withColumn("order_date", F.to_date("order_date_str", "yyyy-MM-dd"))
    .withColumn("amount", F.col("amount").cast("double"))
    .dropDuplicates(["order_id"])
    .withColumn("order_month", F.date_trunc("month", "order_date"))
)

# Data quality validation
total_rows = cleaned.count()
null_rate = cleaned.filter(F.col("customer_id").isNull()).count() / total_rows
assert null_rate < 0.01, f"Null rate {null_rate:.2%} exceeds 1% threshold"
```

### LOAD — Writing to Target

| Strategy | What It Does | When |
|----------|-------------|------|
| **Full load (overwrite)** | Replace entire table | Small tables, first load, recovery |
| **Incremental (append)** | Add only new/changed records | Large tables, frequent updates |
| **Upsert (MERGE)** | Insert new, update existing | SCD Type 1 (overwrite old values) |
| **SCD Type 2** | Keep history with valid_from/valid_to dates | Audit trail needed |

```python
# PySpark load to MSSQL via JDBC (your ATCS approach)
cleaned.write \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://host:1433;databaseName=analytics") \
    .option("dbtable", "dbo.sales_clean") \
    .option("user", "etl_user") \
    .option("password", password) \
    .mode("append") \
    .save()

# Load to Snowflake
cleaned.write \
    .format("snowflake") \
    .options(**snowflake_options) \
    .option("dbtable", "SALES_CLEAN") \
    .mode("overwrite") \
    .save()

# Pandas to PostgreSQL
df.to_sql("sales_clean", engine, if_exists="append", index=False, method="multi")
```

---

## 3.4 ETL Orchestration — Airflow

FusionSpan may ask about orchestrating pipelines. Know Airflow basics:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "rahul",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_sales_etl",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    extract = PythonOperator(
        task_id="extract_from_source",
        python_callable=extract_sales_data,
    )

    transform = PythonOperator(
        task_id="transform_and_validate",
        python_callable=transform_data,
    )

    load = PythonOperator(
        task_id="load_to_warehouse",
        python_callable=load_to_snowflake,
    )

    quality_check = PythonOperator(
        task_id="data_quality_check",
        python_callable=run_quality_checks,
    )

    extract >> transform >> load >> quality_check
```

**Key Airflow concepts to know:**
- **DAG** — Directed Acyclic Graph of tasks
- **Operators** — PythonOperator, BashOperator, SnowflakeOperator
- **Sensors** — Wait for a condition (file exists, API ready)
- **XCom** — Pass data between tasks
- **Backfill** — Reprocess historical dates
- **`{{ ds }}`** — Jinja template for execution date

---

## 3.5 Data Quality Framework

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Quality Dimensions                      │
├──────────────┬──────────────────────────────────────────────┤
│ Completeness │ Required fields populated? Null rates < 1%?  │
│ Uniqueness   │ Primary keys unique? No duplicate records?   │
│ Freshness    │ Data arriving on time? MAX(updated_at) OK?   │
│ Validity     │ Values in expected range? Formats correct?   │
│ Consistency  │ Totals match across sources? FK integrity?   │
│ Volume       │ Row count within expected bounds?            │
└──────────────┴──────────────────────────────────────────────┘
```

**Your ATCS experience talking point:**
> "At ATCS, I built a reusable data quality validation layer in PySpark. Before loading to MSSQL, every batch went through completeness checks (null rates per column), uniqueness validation (no duplicate primary keys), type coercion validation, and statistical profiling. If any check failed, the pipeline halted and alerted the team. This resulted in zero data quality escapes into production analytics."

---

## 3.6 Incremental Loading & CDC

| Method | How | Pros | Cons |
|--------|-----|------|------|
| **Timestamp-based** | `WHERE updated_at > last_sync` | Simple, no special access | Misses deletes |
| **Log-based CDC** | Read DB transaction log (WAL, binlog) | Captures all changes, no source impact | Requires DB admin access |
| **Snapshot diff** | Compare full snapshots | Works with any source | Expensive for large tables |
| **Snowflake Streams** | Built-in CDC on tables | Native, easy | Snowflake-only |

---

# 4. Data Integration Patterns

## 4.1 Batch vs Streaming vs Micro-Batch

| Pattern | Latency | Tools | When |
|---------|---------|-------|------|
| **Batch** | Hours | PySpark, Airflow, dbt | Nightly reports, historical loads |
| **Micro-batch** | Minutes | Spark Structured Streaming | Near-real-time dashboards |
| **Streaming** | Seconds | Kafka + Flink, Kinesis | Real-time alerts, fraud detection |

## 4.2 Data Lake vs Data Warehouse vs Lakehouse

| Concept | Storage | Schema | Tools | Best For |
|---------|---------|--------|-------|----------|
| **Data Lake** | Raw files (S3, GCS) | Schema-on-read | Spark, Presto | Unstructured/semi-structured data |
| **Data Warehouse** | Structured tables | Schema-on-write | Snowflake, BigQuery, Redshift | Analytics, BI |
| **Lakehouse** | Structured files + ACID | Both | Delta Lake, Iceberg + Spark | Best of both worlds |

---

# 5. Cloud Data Platforms

FusionSpan uses Snowflake. Be familiar with cloud warehouse concepts even if you haven't used Snowflake daily.

## 5.1 Snowflake — Key Concepts

| Feature | What It Does | Why It Matters |
|---------|-------------|---------------|
| **Separate compute + storage** | Scale each independently | Pay only for what you use |
| **Virtual Warehouses** | Independent compute clusters | Different sizes for ETL vs analytics |
| **Time Travel** | Query historical data (up to 90 days) | Undo mistakes, audit changes |
| **Zero-Copy Cloning** | Instant clone of tables/databases | Free dev/test environments |
| **Snowpipe** | Continuous auto-ingestion from S3/GCS | Near-real-time data loading |
| **Streams + Tasks** | Built-in CDC + scheduled SQL | No external orchestrator needed |
| **VARIANT type** | Native JSON/semi-structured support | Parse JSON in SQL |

## 5.2 AWS Data Services — Quick Reference

| Service | Purpose | Equivalent |
|---------|---------|-----------|
| **S3** | Object storage (data lake) | GCS, Azure Blob |
| **Redshift** | Data warehouse | Snowflake, BigQuery |
| **Glue** | Serverless ETL (PySpark) | Dataflow, Data Factory |
| **Athena** | Query S3 with SQL | BigQuery, Presto |
| **Lambda** | Serverless functions | Cloud Functions |
| **Kinesis** | Real-time streaming | Kafka, Pub/Sub |

## 5.3 GCP Data Services

| Service | Purpose |
|---------|---------|
| **BigQuery** | Serverless warehouse, pay-per-query |
| **Dataflow** | Managed Apache Beam (batch + stream) |
| **Cloud Storage** | Object storage (data lake) |
| **Pub/Sub** | Message queue for streaming |
| **Dataproc** | Managed Spark/Hadoop clusters |

---

# 6. REST APIs — Build & Consume

FusionSpan values API knowledge. Know how to both build and consume RESTful APIs.

## 6.1 HTTP Methods

| Method | Operation | Idempotent? | Body? | Example |
|--------|-----------|------------|-------|---------|
| `GET` | Read | Yes | No | `GET /api/customers/123` |
| `POST` | Create | No | Yes | `POST /api/customers` with JSON body |
| `PUT` | Replace | Yes | Yes | `PUT /api/customers/123` |
| `PATCH` | Partial update | No | Yes | `PATCH /api/customers/123` |
| `DELETE` | Delete | Yes | Rarely | `DELETE /api/customers/123` |

## 6.2 Consuming APIs in Python

```python
import requests

# GET — fetch data
response = requests.get(
    "https://api.example.com/customers",
    headers={"Authorization": "Bearer <token>"},
    params={"status": "active", "limit": 100}
)
response.raise_for_status()
customers = response.json()

# POST — create data
new_customer = {"name": "Acme Corp", "email": "info@acme.com"}
response = requests.post(
    "https://api.example.com/customers",
    json=new_customer,
    headers={"Authorization": "Bearer <token>"}
)
created = response.json()

# Pagination pattern
all_results = []
page = 1
while True:
    resp = requests.get(f"https://api.example.com/data?page={page}&limit=100")
    data = resp.json()
    if not data["results"]:
        break
    all_results.extend(data["results"])
    page += 1
```

## 6.3 Building APIs with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Data Pipeline API")

class PipelineRequest(BaseModel):
    source: str
    destination: str
    schedule: str = "daily"

class PipelineResponse(BaseModel):
    pipeline_id: int
    status: str

@app.post("/api/pipelines", response_model=PipelineResponse, status_code=201)
async def create_pipeline(request: PipelineRequest):
    pipeline_id = create_pipeline_in_db(request)
    return PipelineResponse(pipeline_id=pipeline_id, status="created")

@app.get("/api/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: int):
    pipeline = get_pipeline_from_db(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline
```

**Your differentiator:** "In my PathWise project, I built a full production FastAPI backend — 10,687 lines of code, 50+ Pydantic models, async LLM integration, and deployment on Railway. I understand the full lifecycle from schema design to deployment."

---

# 7. SQL Essentials Cheat Sheet

Quick reference for the most commonly asked SQL patterns. (Full details in your `22_data_engineering_and_sql.md`.)

| Pattern | SQL | When Asked |
|---------|-----|-----------|
| **2nd highest salary** | `SELECT MAX(salary) FROM emp WHERE salary < (SELECT MAX(salary) FROM emp)` | Classic interview question |
| **Running total** | `SUM(amount) OVER (ORDER BY date)` | Window functions |
| **Top N per group** | `ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)` then filter `WHERE rn <= N` | Analytical queries |
| **Deduplication** | `ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn` then delete `WHERE rn > 1` | Data cleaning |
| **Self join** | `JOIN employees m ON e.manager_id = m.emp_id` | Hierarchical queries |
| **MERGE/UPSERT** | `MERGE INTO target USING source ON ... WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT` | ETL loading |

---

# 8. ML/AI Quick Reference

FusionSpan may ask about ML broadly. Keep answers concise.

| Topic | Your 30-Second Answer |
|-------|----------------------|
| **Types of ML** | Supervised (labeled data: classification, regression), Unsupervised (clustering, dimensionality reduction), Reinforcement Learning |
| **Model you know best** | XGBoost for tabular data — used it in Tesla stock forecasting with Optuna tuning |
| **Evaluation metrics** | Classification: F1, AUC-ROC. Regression: RMSE, MAE. Always compare against a baseline |
| **Overfitting** | Model memorizes training noise. Fix with regularization, more data, simpler model, cross-validation |
| **LLMs** | Used Groq (Llama-3.3) in production for text summarization and career recommendations in PathWise |
| **Pre-trained models** | Used BERT for legal doc classification at ATCS, Sentence Transformers for embeddings |

---

# 9. Agile/Scrum Essentials

FusionSpan mentions "participate in Agile/Scrum."

| Concept | What It Is |
|---------|-----------|
| **Sprint** | Fixed time-box (usually 2 weeks) to deliver increment |
| **Daily Standup** | 15-min meeting: what I did, what I'll do, blockers |
| **Sprint Planning** | Team selects work for the upcoming sprint from backlog |
| **Sprint Review** | Demo completed work to stakeholders |
| **Sprint Retrospective** | Team reflects on what went well / what to improve |
| **Product Backlog** | Prioritized list of features/tasks |
| **User Story** | "As a [user], I want [goal] so that [benefit]" |
| **Story Points** | Relative effort estimation (Fibonacci: 1, 2, 3, 5, 8, 13) |

**Your answer:** "At both Jet2 and ATCS, I worked in Agile sprints. At Jet2, we had 2-week sprints with daily standups. I'd present ML model updates in sprint reviews, and we used Jira to track stories. I'm comfortable with iterative development — building an MVP, getting feedback, and refining."

---

# 10. FusionSpan — Company & Role Fit

## What FusionSpan Does

- **Focus:** Technology solutions for **associations and nonprofits**
- **Core platform:** Salesforce CRM customization and integration
- **Data team:** Builds data pipelines, integrations, and analytics for clients
- **Tech stack:** Snowflake, Python, SQL, cloud platforms, Salesforce APIs
- **Culture:** Remote-first, Agile teams, collaborative

## "Why FusionSpan?"

> "FusionSpan's focus on enterprise data solutions for associations is compelling because it combines the technical depth I love — building ETL pipelines, working with cloud warehouses like Snowflake, and integrating diverse data sources — with meaningful impact for organizations that drive community and professional development. I'm excited about the data engineering challenges of integrating Salesforce data with cloud analytics platforms, and my experience building production data pipelines with PySpark, Python, and SQL maps directly to what your data team does."

## "What do you know about our work?"

> "FusionSpan specializes in technology solutions for associations and membership organizations, with deep expertise in Salesforce/CRM implementations. Your data team works on building data integration pipelines that connect Salesforce with analytics platforms like Snowflake, enabling organizations to make data-driven decisions about member engagement, retention, and operations. I find the challenge of integrating diverse data sources — CRM data, event systems, financial platforms — into a unified analytics layer particularly interesting."

---

# 11. Interview Talking Points — Your Projects

## Projects Most Relevant to FusionSpan

| Project | Relevance | Key Points |
|---------|-----------|------------|
| **ATCS PySpark ETL** | **HIGHEST** | Production ETL pipelines, data quality, MSSQL loading |
| **PathWise (FastAPI)** | HIGH | REST API design, database integration, production deployment |
| **Jet2 Risk Forecasting** | HIGH | Feature engineering, data pipelines, production ML |
| **FIBE Scorecard** | MEDIUM | Data transformation, feature engineering at scale |
| **Tesla Forecasting** | MEDIUM | Data extraction (APIs), feature engineering, XGBoost |

## 30-Second ATCS ETL Story

> "At ATCS, I engineered PySpark ETL pipelines that processed large-scale production datasets for downstream analytics and ML. The pipeline ingested data from flat files, CSVs, and database extracts, applied schema validation, null handling, type coercion, and deduplication, then loaded clean data into MSSQL via JDBC. I also built a reusable data quality validation framework with configurable rules — completeness, referential integrity, and statistical profiling. The result was zero data quality escapes into production, and the pipeline architecture was adopted across multiple client projects."

---

*Prepared for Rahul Sharma — FusionSpan Data Engineering Interview. Covers OOP fundamentals, ETL/ELT architecture, cloud platforms, REST APIs, Agile practices, and company fit.*
