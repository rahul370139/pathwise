# P05 — PySpark ETL, Mask R-CNN Detection & Legal Document Classification | ATCS

> **Rahul Sharma** | Associate Data Scientist | Advanced Technology Consulting Service (ATCS), Jaipur, India
> **Duration:** November 2020 – September 2021

---

## Table of Contents

1. [Project Overview (All Three Projects)](#1-project-overview)
2. [Project A: PySpark ETL Pipeline with Data Quality Validation](#2-project-a-pyspark-etl-pipeline-with-data-quality-validation)
3. [Project B: Mask R-CNN Document Detection (85% mAP)](#3-project-b-mask-r-cnn-document-detection-85-map)
4. [Project C: Legal Document Classification (NLP)](#4-project-c-legal-document-classification-nlp)
5. [Topics You Must Know (Comprehensive Study Guide)](#5-topics-you-must-know-comprehensive-study-guide)
6. [Interview Questions & Answers (25+)](#6-interview-questions--answers)
7. [Red Flags & How to Handle](#7-red-flags--how-to-handle)
8. [Key Takeaways & Talking Points](#8-key-takeaways--talking-points)

---

## 1. Project Overview

### 1.1 STAR Summary (Interview-Ready — Unified Narrative)

**Situation**
At ATCS (Advanced Technology Consulting Service), a technology consulting firm in Jaipur, India, clients across industries needed to digitize, classify, and process large volumes of heterogeneous documents — from scanned contracts and legal filings to structured transactional data. The existing workflows relied on manual sorting, ad-hoc scripting, and fragile batch jobs that couldn't scale. Data quality issues routinely propagated into downstream analytics, document classification was labor-intensive, and there was no automated detection pipeline for extracting structured regions from scanned documents.

**Task**
As an Associate Data Scientist, I was responsible for three interconnected workstreams:
1. **Building a production-grade PySpark ETL pipeline** that ingested raw data from multiple sources, applied rigorous data quality validation, and loaded clean data into MSSQL for downstream analytics.
2. **Training a Mask R-CNN model** using transfer learning in TensorFlow for detecting and segmenting regions of interest (stamps, signatures, tables, headers) in scanned documents, achieving 85% mAP and deploying the model to Azure Blob Storage for inference.
3. **Developing an NLP-based legal document classification system** using transfer learning and modular preprocessing pipelines to automate legal document tagging and labeling accuracy validation.

**Approach & Action**

| Phase | What I Did |
|-------|-----------|
| **ETL Pipeline Design** | Designed and implemented a PySpark-based ETL pipeline reading from flat files, CSVs, and database extracts; applied schema validation, null handling, type coercion, and deduplication before loading to MSSQL via JDBC. |
| **Data Quality Framework** | Built a reusable data quality validation layer with configurable rules — completeness checks, referential integrity, statistical profiling, and anomaly flagging — that ran as a pre-analytics gate. |
| **Mask R-CNN Training** | Fine-tuned a Mask R-CNN model (ResNet-101-FPN backbone) pretrained on COCO using TensorFlow/Keras on a custom-annotated document dataset. Implemented data augmentation, anchor tuning, and learning rate scheduling. |
| **Model Deployment** | Packaged the trained Mask R-CNN model and deployed to Azure Blob Storage, building an inference pipeline that pulled models from blob, ran detection on incoming documents, and returned bounding boxes + masks. |
| **Legal NLP Classification** | Built a classification model for legal document tagging using transfer learning (BERT-based) with NLP preprocessing (tokenization, stopword removal, legal entity extraction). Created modular pipelines for automated document ingestion. |
| **Labeling Validation** | Developed labeling accuracy validation logic — comparing model predictions against human annotations using confusion matrices, per-class F1, and inter-annotator agreement (Cohen's Kappa). |

**Result**
- PySpark ETL pipeline processing data reliably into MSSQL with **zero data quality escapes** into production analytics
- Mask R-CNN achieving **85% mAP** on document region detection, deployed to Azure Blob for scalable inference
- Legal document classification model with high accuracy on multi-label tagging, reducing manual labeling effort by ~70%
- Modular, reusable pipelines adopted across multiple client engagements at ATCS

---

### 1.2 Combined Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATCS DOCUMENT INTELLIGENCE PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║                   PROJECT A: PySpark ETL Pipeline                        ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────┐  ┌──────────┐  ┌──────────┐                              ║   │
│  ║  │ CSV/Flat │  │ Database │  │ External │                              ║   │
│  ║  │ Files    │  │ Extracts │  │ APIs     │                              ║   │
│  ║  └────┬─────┘  └────┬─────┘  └────┬─────┘                              ║   │
│  ║       │              │              │                                    ║   │
│  ║       ▼              ▼              ▼                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │          PySpark Ingestion Layer             │                        ║   │
│  ║  │  • spark.read.csv / .parquet / .jdbc         │                        ║   │
│  ║  │  • Schema inference + explicit casting       │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Data Quality Validation Layer          │                        ║   │
│  ║  │  • Schema checks (column names, types)       │                        ║   │
│  ║  │  • Null/missing value analysis               │                        ║   │
│  ║  │  • Type validation & coercion                │                        ║   │
│  ║  │  • Deduplication (exact + fuzzy)             │                        ║   │
│  ║  │  • Statistical profiling & anomaly flags     │                        ║   │
│  ║  │  • Referential integrity checks              │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Transformation & Enrichment            │                        ║   │
│  ║  │  • Column renaming, type casting             │                        ║   │
│  ║  │  • Derived features, aggregations            │                        ║   │
│  ║  │  • Partitioning for load optimization        │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │        MSSQL Load (JDBC)                     │                        ║   │
│  ║  │  • Batch writes via JDBC connector           │                        ║   │
│  ║  │  • Upsert logic (merge/overwrite)            │                        ║   │
│  ║  │  • Transaction management                    │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║              PROJECT B: Mask R-CNN Document Detection                    ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────────┐   ┌──────────────────┐                                ║   │
│  ║  │  Scanned     │   │  Annotation Tool │                                ║   │
│  ║  │  Documents   │   │  (VIA/LabelMe)   │                                ║   │
│  ║  │  (PDF/TIFF)  │   │  COCO-format     │                                ║   │
│  ║  └──────┬───────┘   └────────┬─────────┘                                ║   │
│  ║         │                     │                                          ║   │
│  ║         ▼                     ▼                                          ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │        Data Preprocessing                    │                        ║   │
│  ║  │  • Image resizing (1024×1024)                │                        ║   │
│  ║  │  • Augmentation (flip, rotate, brightness)   │                        ║   │
│  ║  │  • Train/Val/Test split (70/15/15)           │                        ║   │
│  ║  │  • Mask generation from polygon annotations  │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │    Mask R-CNN (Transfer Learning)            │                        ║   │
│  ║  │    Backbone: ResNet-101 + FPN                │                        ║   │
│  ║  │    Pretrained: COCO weights                  │                        ║   │
│  ║  │                                              │                        ║   │
│  ║  │  ┌─────────┐ ┌─────┐ ┌─────────┐ ┌──────┐  │                        ║   │
│  ║  │  │ResNet-  │→│ FPN │→│  RPN    │→│ ROI  │  │                        ║   │
│  ║  │  │101      │ │     │ │(Region  │ │Align │  │                        ║   │
│  ║  │  │Backbone │ │     │ │Proposal)│ │      │  │                        ║   │
│  ║  │  └─────────┘ └─────┘ └─────────┘ └──┬───┘  │                        ║   │
│  ║  │                                      │      │                        ║   │
│  ║  │                    ┌─────────────────┼──────────────┐                 ║   │
│  ║  │                    ▼                 ▼              ▼                 ║   │
│  ║  │              ┌──────────┐   ┌──────────┐   ┌──────────┐             ║   │
│  ║  │              │  Class   │   │  BBox    │   │  Mask    │             ║   │
│  ║  │              │  Head    │   │  Head    │   │  Head    │             ║   │
│  ║  │              │(Softmax) │   │(Regress) │   │(FCN)    │             ║   │
│  ║  │              └──────────┘   └──────────┘   └──────────┘             ║   │
│  ║  │                                                                      │   │
│  ║  │    Result: 85% mAP                                                   │   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │      Azure Blob Storage Deployment           │                        ║   │
│  ║  │  • Model weights (.pb / SavedModel)          │                        ║   │
│  ║  │  • Inference API pulls model on startup      │                        ║   │
│  ║  │  • Batch + real-time inference support        │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗   │
│  ║           PROJECT C: Legal Document Classification (NLP)                ║   │
│  ║                                                                         ║   │
│  ║  ┌──────────────┐   ┌──────────────────┐                                ║   │
│  ║  │  Legal Docs  │   │  Human Labels    │                                ║   │
│  ║  │  (Contracts, │   │  (Category tags, │                                ║   │
│  ║  │   Filings,   │   │   multi-label)   │                                ║   │
│  ║  │   Briefs)    │   │                  │                                ║   │
│  ║  └──────┬───────┘   └────────┬─────────┘                                ║   │
│  ║         │                     │                                          ║   │
│  ║         ▼                     ▼                                          ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │         NLP Preprocessing Pipeline           │                        ║   │
│  ║  │  • Text extraction (OCR if scanned)          │                        ║   │
│  ║  │  • Legal tokenization (domain-aware)         │                        ║   │
│  ║  │  • Stopword removal + legal stopword list    │                        ║   │
│  ║  │  • Named Entity Recognition (legal entities) │                        ║   │
│  ║  │  • Section segmentation                      │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │      Transfer Learning Classification        │                        ║   │
│  ║  │  • BERT / Legal-BERT fine-tuning             │                        ║   │
│  ║  │  • Multi-label classification head            │                        ║   │
│  ║  │  • Class-weighted loss for imbalance          │                        ║   │
│  ║  └──────────────────┬──────────────────────────┘                        ║   │
│  ║                     ▼                                                    ║   │
│  ║  ┌─────────────────────────────────────────────┐                        ║   │
│  ║  │       Labeling Accuracy Validation           │                        ║   │
│  ║  │  • Confusion matrix per class                │                        ║   │
│  ║  │  • Per-class precision / recall / F1         │                        ║   │
│  ║  │  • Cohen's Kappa (inter-annotator)           │                        ║   │
│  ║  │  • Active learning feedback loop             │                        ║   │
│  ║  └─────────────────────────────────────────────┘                        ║   │
│  ╚═══════════════════════════════════════════════════════════════════════════╝   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Tech Stack

| Category | Tools / Technologies |
|----------|---------------------|
| **Languages** | Python, SQL, PySpark |
| **ETL / Data** | Apache Spark (PySpark), MSSQL, JDBC, Pandas |
| **Computer Vision** | TensorFlow, Keras, Mask R-CNN, OpenCV |
| **NLP** | Hugging Face Transformers, BERT, spaCy, NLTK |
| **Cloud** | Azure Blob Storage, Azure VMs |
| **Annotation** | VGG Image Annotator (VIA), LabelMe, COCO format |
| **Monitoring** | Custom data quality validators, logging frameworks |
| **Version Control** | Git, Azure DevOps |

---

## 2. Project A: PySpark ETL Pipeline with Data Quality Validation

### 2.1 Problem Context

The client required a robust, scalable pipeline to:
- Ingest data from heterogeneous sources (flat files, CSVs, database extracts)
- Apply rigorous data quality validation before any analytics or reporting
- Load validated data into MSSQL for downstream BI tools and analytics teams
- Handle daily batch loads with growing data volumes (GBs per day)

**Why PySpark over plain Python/Pandas?**
- Data volumes exceeded single-node memory capacity
- PySpark's distributed computation enabled horizontal scaling
- Native support for reading multiple file formats
- Catalyst optimizer for efficient query execution plans
- Seamless integration with JDBC for MSSQL writes

---

### 2.2 ETL vs. ELT — Why ETL?

| Aspect | ETL (Extract-Transform-Load) | ELT (Extract-Load-Transform) |
|--------|------------------------------|------------------------------|
| **Transform location** | In the processing engine (Spark) | In the target database |
| **When to use** | When data quality must be validated before loading | When target DB has strong compute (Snowflake, BigQuery) |
| **Data quality** | Validated before it enters the target DB | Loaded raw, then cleaned in-place |
| **Our choice** | **ETL** — we needed to guarantee no dirty data entered MSSQL | — |

**Key decision rationale:** MSSQL was used primarily for reporting and analytics. Loading unvalidated data would have corrupted downstream dashboards and reports. ETL ensured that only clean, validated data reached MSSQL.

---

### 2.3 PySpark Pipeline — Deep Dive

#### 2.3.1 Extraction Phase

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

spark = SparkSession.builder \
    .appName("ATCS_ETL_Pipeline") \
    .config("spark.jars", "/path/to/mssql-jdbc-9.4.0.jre11.jar") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Define explicit schema (never rely solely on inference in production)
document_schema = StructType([
    StructField("doc_id", StringType(), nullable=False),
    StructField("doc_type", StringType(), nullable=True),
    StructField("client_name", StringType(), nullable=True),
    StructField("received_date", DateType(), nullable=True),
    StructField("page_count", IntegerType(), nullable=True),
    StructField("file_path", StringType(), nullable=False),
    StructField("status", StringType(), nullable=True),
    StructField("confidence_score", DoubleType(), nullable=True),
])

# Read with explicit schema
raw_df = spark.read \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .schema(document_schema) \
    .csv("/data/incoming/documents/*.csv")

print(f"Records read: {raw_df.count()}")
print(f"Corrupt records: {raw_df.filter(raw_df['_corrupt_record'].isNotNull()).count()}")
```

#### 2.3.2 Data Quality Validation Layer

This was the critical differentiator — a reusable, configurable validation framework.

```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from functools import reduce

class DataQualityValidator:
    """Reusable data quality validation framework for PySpark DataFrames."""
    
    def __init__(self, df, table_name):
        self.df = df
        self.table_name = table_name
        self.issues = []
        self.total_rows = df.count()
    
    def check_schema(self, expected_schema: StructType):
        """Validate that DataFrame schema matches expected schema."""
        actual_cols = set(self.df.columns)
        expected_cols = set([f.name for f in expected_schema.fields])
        
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        
        if missing:
            self.issues.append(f"SCHEMA_ERROR: Missing columns: {missing}")
        if extra:
            self.issues.append(f"SCHEMA_WARNING: Unexpected columns: {extra}")
        
        # Type validation
        for field in expected_schema.fields:
            if field.name in actual_cols:
                actual_type = dict(self.df.dtypes).get(field.name)
                expected_type = field.dataType.simpleString()
                if actual_type != expected_type:
                    self.issues.append(
                        f"TYPE_ERROR: {field.name} expected {expected_type}, got {actual_type}"
                    )
        return self
    
    def check_nulls(self, critical_columns, threshold=0.05):
        """Flag columns where null ratio exceeds threshold."""
        for col_name in critical_columns:
            null_count = self.df.filter(F.col(col_name).isNull()).count()
            null_ratio = null_count / self.total_rows if self.total_rows > 0 else 0
            
            if null_ratio > threshold:
                self.issues.append(
                    f"NULL_ERROR: {col_name} has {null_ratio:.2%} nulls "
                    f"(threshold: {threshold:.2%})"
                )
        return self
    
    def check_duplicates(self, key_columns):
        """Detect duplicate rows based on key columns."""
        dup_count = self.total_rows - self.df.dropDuplicates(key_columns).count()
        if dup_count > 0:
            self.issues.append(
                f"DUPLICATE_ERROR: {dup_count} duplicate rows on keys {key_columns}"
            )
        return self
    
    def check_value_ranges(self, column, min_val=None, max_val=None):
        """Validate numeric columns fall within expected ranges."""
        if min_val is not None:
            below = self.df.filter(F.col(column) < min_val).count()
            if below > 0:
                self.issues.append(
                    f"RANGE_ERROR: {column} has {below} values below {min_val}"
                )
        if max_val is not None:
            above = self.df.filter(F.col(column) > max_val).count()
            if above > 0:
                self.issues.append(
                    f"RANGE_ERROR: {column} has {above} values above {max_val}"
                )
        return self
    
    def check_referential_integrity(self, column, reference_df, ref_column):
        """Ensure foreign key values exist in reference table."""
        orphan_count = self.df.join(
            reference_df, self.df[column] == reference_df[ref_column], "left_anti"
        ).count()
        if orphan_count > 0:
            self.issues.append(
                f"REF_INTEGRITY_ERROR: {orphan_count} orphan records in {column}"
            )
        return self
    
    def validate(self, fail_on_error=True):
        """Execute all checks and return results."""
        errors = [i for i in self.issues if "ERROR" in i]
        warnings = [i for i in self.issues if "WARNING" in i]
        
        report = {
            "table": self.table_name,
            "total_rows": self.total_rows,
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": self.issues,
            "passed": len(errors) == 0
        }
        
        if fail_on_error and not report["passed"]:
            raise DataQualityError(
                f"Validation failed for {self.table_name}: {errors}"
            )
        
        return report

class DataQualityError(Exception):
    pass
```

**Usage in the pipeline:**

```python
# Run validation before loading
validator = DataQualityValidator(raw_df, "document_metadata")
report = validator \
    .check_schema(document_schema) \
    .check_nulls(critical_columns=["doc_id", "file_path"], threshold=0.0) \
    .check_nulls(critical_columns=["doc_type", "client_name"], threshold=0.05) \
    .check_duplicates(key_columns=["doc_id"]) \
    .check_value_ranges("page_count", min_val=1, max_val=10000) \
    .check_value_ranges("confidence_score", min_val=0.0, max_val=1.0) \
    .validate(fail_on_error=True)

print(f"Validation passed: {report['passed']}")
```

#### 2.3.3 Transformation Phase

```python
# Clean and transform validated data
transformed_df = raw_df \
    .withColumn("doc_type", F.upper(F.trim(F.col("doc_type")))) \
    .withColumn("client_name", F.initcap(F.trim(F.col("client_name")))) \
    .withColumn("received_date", F.to_date(F.col("received_date"), "yyyy-MM-dd")) \
    .withColumn("ingestion_timestamp", F.current_timestamp()) \
    .withColumn("status", F.coalesce(F.col("status"), F.lit("PENDING"))) \
    .withColumn("confidence_score",
                F.when(F.col("confidence_score").isNull(), F.lit(0.0))
                 .otherwise(F.col("confidence_score"))) \
    .dropDuplicates(["doc_id"])

# Add derived columns
transformed_df = transformed_df \
    .withColumn("is_high_confidence", F.col("confidence_score") >= 0.85) \
    .withColumn("processing_priority",
                F.when(F.col("doc_type") == "LEGAL", 1)
                 .when(F.col("doc_type") == "FINANCIAL", 2)
                 .otherwise(3))
```

#### 2.3.4 Loading to MSSQL via JDBC

```python
# MSSQL JDBC connection properties
jdbc_url = "jdbc:sqlserver://atcs-prod-db.database.windows.net:1433;databaseName=DocumentDB"
connection_properties = {
    "user": "etl_service_account",
    "password": "${DB_PASSWORD}",  # injected from environment / secrets manager
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    "batchsize": "10000",
    "isolationLevel": "READ_COMMITTED"
}

# Write to MSSQL — overwrite for full refresh, append for incremental
transformed_df.write \
    .mode("append") \
    .jdbc(url=jdbc_url, table="dbo.document_metadata", properties=connection_properties)

# Verify write
verify_count = spark.read \
    .jdbc(url=jdbc_url, table="dbo.document_metadata", properties=connection_properties) \
    .count()

print(f"Records in MSSQL after load: {verify_count}")
```

---

### 2.4 PySpark Core Concepts (Must-Know)

#### RDDs vs. DataFrames vs. Datasets

| Feature | RDD | DataFrame | Dataset (Scala/Java) |
|---------|-----|-----------|----------------------|
| **Abstraction** | Low-level, distributed collection | Distributed table with named columns | Typed DataFrame |
| **Optimization** | No Catalyst optimization | Catalyst + Tungsten optimized | Catalyst + Tungsten optimized |
| **Schema** | No schema | Has schema (StructType) | Strongly typed schema |
| **API** | Functional (map, filter, reduce) | SQL-like (select, filter, groupBy) | Combines both |
| **Use case** | Unstructured data, custom logic | Structured/semi-structured data | Type-safe operations |
| **Performance** | Slowest (no optimization) | Fastest (optimizer + code gen) | Fast (with type safety) |

**Interview talking point:** "We used DataFrames exclusively because our data was structured (CSVs, database tables), and DataFrames benefit from Catalyst optimization which gave us 2-5x better performance over equivalent RDD operations."

#### Transformations vs. Actions

```
TRANSFORMATIONS (Lazy — build execution plan)          ACTIONS (Eager — trigger computation)
─────────────────────────────────────────────          ────────────────────────────────────
select(), filter(), groupBy(), join()                  count(), collect(), show(), write()
withColumn(), drop(), distinct()                       first(), take(), foreach()
union(), repartition(), coalesce()                     toPandas(), saveAsTable()

Key insight: Spark builds a DAG of transformations. Nothing executes until an action triggers it.
This enables Catalyst to optimize the entire plan.
```

#### Narrow vs. Wide Transformations

```
NARROW (no shuffle — fast)         WIDE (shuffle required — expensive)
────────────────────────────       ──────────────────────────────────
map(), filter(), select()          groupBy(), join(), distinct()
withColumn(), union()              repartition(), orderBy()

Why it matters: Wide transformations cause data shuffle across the network,
which is the #1 performance bottleneck in Spark.
```

#### Spark Execution Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SPARK EXECUTION FLOW                             │
│                                                                      │
│  User Code (PySpark)                                                 │
│       │                                                              │
│       ▼                                                              │
│  Logical Plan                                                        │
│       │                                                              │
│       ▼                                                              │
│  Catalyst Optimizer ──→ Optimized Logical Plan                       │
│       │                  • Predicate pushdown                        │
│       │                  • Column pruning                            │
│       │                  • Constant folding                          │
│       │                  • Join reordering                           │
│       ▼                                                              │
│  Physical Plan (chosen by cost-based optimizer)                      │
│       │                                                              │
│       ▼                                                              │
│  Tungsten Engine (Code Generation)                                   │
│       │    • Whole-stage code generation                             │
│       │    • Off-heap memory management                              │
│       ▼                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                              │
│  │  Stage 1 │  │  Stage 2 │  │  Stage 3 │  (separated by shuffles)   │
│  │ Tasks=N  │→ │ Tasks=M  │→ │ Tasks=K  │                            │
│  └─────────┘  └─────────┘  └─────────┘                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 MSSQL Integration — Key Patterns

**JDBC Batch Size Tuning:**
```python
# Default batchsize is 1000 — too small for large loads
# We tuned to 10,000 based on benchmarking
#   1,000 → 45 min for 2M rows
#  10,000 → 12 min for 2M rows
#  50,000 → 10 min (diminishing returns, memory pressure)
```

**Handling Upsert (Merge) Logic:**
```python
# PySpark doesn't natively support UPSERT
# Strategy: Write to staging table, then MERGE in MSSQL

# Step 1: Write to staging
transformed_df.write \
    .mode("overwrite") \
    .jdbc(url=jdbc_url, table="dbo.staging_document_metadata", properties=connection_properties)

# Step 2: Execute MERGE via pyodbc
import pyodbc
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
cursor.execute("""
    MERGE dbo.document_metadata AS target
    USING dbo.staging_document_metadata AS source
    ON target.doc_id = source.doc_id
    WHEN MATCHED THEN
        UPDATE SET target.status = source.status,
                   target.confidence_score = source.confidence_score,
                   target.ingestion_timestamp = source.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (doc_id, doc_type, client_name, received_date, 
                page_count, file_path, status, confidence_score, ingestion_timestamp)
        VALUES (source.doc_id, source.doc_type, source.client_name, source.received_date,
                source.page_count, source.file_path, source.status, 
                source.confidence_score, source.ingestion_timestamp);
""")
conn.commit()
```

---

## 3. Project B: Mask R-CNN Document Detection (85% mAP)

### 3.1 Problem Context

Scanned documents contained regions of interest — stamps, signatures, tables, headers, handwritten notes — that needed to be detected and segmented for downstream processing (extraction, classification, archival). Manual identification was slow and error-prone.

**Why Mask R-CNN (not YOLO, Faster R-CNN, etc.)?**

| Model | Detection | Segmentation | Speed | Our Need |
|-------|-----------|-------------|-------|----------|
| **YOLO** | Bounding boxes | No masks | Very fast | Not enough — we needed pixel-level masks |
| **Faster R-CNN** | Bounding boxes | No masks | Moderate | No segmentation capability |
| **Mask R-CNN** | Bounding boxes + classes | **Pixel-level masks** | Moderate | **Chosen** — we needed both detection and segmentation |
| **U-Net** | No detection | Semantic segmentation | Fast | No instance-level distinction |

**Key requirement:** We needed to distinguish between overlapping elements (e.g., a signature overlapping a stamp). Only instance segmentation (Mask R-CNN) provides per-instance masks.

---

### 3.2 Mask R-CNN Architecture — Complete Breakdown

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       MASK R-CNN ARCHITECTURE                                 │
│                                                                               │
│  Input Image (1024 × 1024 × 3)                                               │
│       │                                                                       │
│       ▼                                                                       │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 1: BACKBONE — ResNet-101                                         ║ │
│  ║                                                                         ║ │
│  ║  Conv1 ──→ Pool ──→ Res2 (C2) ──→ Res3 (C3) ──→ Res4 (C4) ──→ Res5 (C5) ║
│  ║   │                  256ch          512ch          1024ch         2048ch ║ │
│  ║   │                    │              │               │              │   ║ │
│  ║   │                    │              │               │              │   ║ │
│  ╚═══│════════════════════│══════════════│═══════════════│══════════════│═══╝ │
│       │                    │              │               │              │     │
│       │                    ▼              ▼               ▼              ▼     │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 2: FEATURE PYRAMID NETWORK (FPN)                                 ║ │
│  ║                                                                         ║ │
│  ║  Top-down pathway + lateral connections:                                ║ │
│  ║                                                                         ║ │
│  ║  C5 ──→ P5 (1/32 scale) ──→ upsample ──→ + ──→ P4 (1/16 scale)       ║ │
│  ║                                            │                            ║ │
│  ║  C4 ────────────────────────→ 1×1 conv ────┘    upsample               ║ │
│  ║                                                   │                     ║ │
│  ║  C3 ────────────────────────→ 1×1 conv ──→ + ────→ P3 (1/8 scale)     ║ │
│  ║                                                                         ║ │
│  ║  C2 ────────────────────────→ 1×1 conv ──→ + ────→ P2 (1/4 scale)     ║ │
│  ║                                                                         ║ │
│  ║  All Pn have 256 channels                                               ║ │
│  ║  Purpose: Detect objects at multiple scales                             ║ │
│  ╚═══════════════════════════════════════════════════════════════════════════╝ │
│       │                                                                       │
│       ▼                                                                       │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 3: REGION PROPOSAL NETWORK (RPN)                                 ║ │
│  ║                                                                         ║ │
│  ║  For each location on each feature map (P2–P5):                         ║ │
│  ║    • Generate k anchor boxes (k=15: 5 scales × 3 aspect ratios)        ║ │
│  ║    • Two outputs per anchor:                                            ║ │
│  ║      1. Objectness score (object vs. background)   [2k scores]         ║ │
│  ║      2. Bounding box refinement (dx, dy, dw, dh)   [4k offsets]       ║ │
│  ║                                                                         ║ │
│  ║  Pipeline:                                                              ║ │
│  ║    Anchors → IoU with GT → Binary classification + BBox regression      ║ │
│  ║           → NMS (Non-Maximum Suppression) → Top-N proposals (~2000)    ║ │
│  ║                                                                         ║ │
│  ║  Anchor scales: [32, 64, 128, 256, 512]                                ║ │
│  ║  Anchor ratios: [0.5, 1, 2]                                            ║ │
│  ╚══════════════════════════════════════════════╤════════════════════════════╝ │
│                                                  │                             │
│                                                  ▼                             │
│  ╔═══════════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 4: ROI ALIGN                                                     ║ │
│  ║                                                                         ║ │
│  ║  Problem: ROI Pooling (Faster R-CNN) uses quantization → misalignment  ║ │
│  ║  Solution: ROI Align uses bilinear interpolation → no quantization     ║ │
│  ║                                                                         ║ │
│  ║  For each proposal:                                                     ║ │
│  ║    1. Map ROI to feature map coordinates (no rounding!)                 ║ │
│  ║    2. Divide into grid (e.g., 7×7 for class/bbox, 14×14 for mask)     ║ │
│  ║    3. Sample 4 points per bin using bilinear interpolation              ║ │
│  ║    4. Max-pool or avg-pool within each bin                              ║ │
│  ║                                                                         ║ │
│  ║  Why this matters: Mask prediction requires pixel-level alignment.      ║ │
│  ║  Even 1-pixel misalignment degrades mask quality significantly.         ║ │
│  ╚══════════════════════════════════════════════╤════════════════════════════╝ │
│                                                  │                             │
│                        ┌─────────────────────────┼──────────────────┐          │
│                        ▼                         ▼                  ▼          │
│  ╔══════════════╗  ╔══════════════╗  ╔═══════════════════════════╗             │
│  ║  CLASS HEAD  ║  ║  BBOX HEAD   ║  ║      MASK HEAD            ║             │
│  ║              ║  ║              ║  ║                           ║             │
│  ║  FC layers   ║  ║  FC layers   ║  ║  4 conv layers (3×3)     ║             │
│  ║  → Softmax   ║  ║  → Regression║  ║  → deconv (2×2, s=2)    ║             │
│  ║              ║  ║  (dx,dy,dw,dh)  ║  → 1×1 conv per class   ║             │
│  ║  Output:     ║  ║  Output:     ║  ║  → Sigmoid              ║             │
│  ║  N+1 classes ║  ║  4 coords    ║  ║                           ║             │
│  ║  per ROI     ║  ║  per ROI     ║  ║  Output: 28×28 binary    ║             │
│  ║              ║  ║              ║  ║  mask per class per ROI  ║             │
│  ╚══════════════╝  ╚══════════════╝  ╚═══════════════════════════╝             │
│                                                                               │
│  LOSS FUNCTION:                                                               │
│  L = L_cls + L_bbox + L_mask                                                  │
│    = CrossEntropy + SmoothL1 + BinaryCrossEntropy (per-class, only on GT class)│
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 Key Architectural Components Explained

#### 3.3.1 ResNet-101 Backbone

ResNet-101 consists of 101 layers organized into 4 residual stages (conv2_x through conv5_x). Each stage uses **residual (skip) connections** that solve the vanishing gradient problem:

```
Residual Block:
x ──→ [Conv → BN → ReLU → Conv → BN] ──→ (+) ──→ ReLU ──→ output
 │                                          ↑
 └──────────── identity shortcut ───────────┘

output = F(x) + x    (where F(x) is the learned residual)
```

**Why ResNet-101 (not ResNet-50)?** Deeper backbone extracts richer features for complex document layouts. ResNet-101 gave ~2% mAP improvement over ResNet-50 in our experiments.

#### 3.3.2 Feature Pyramid Network (FPN)

FPN addresses the **multi-scale detection problem**: small objects (text) need high-resolution features, while large objects (tables) need high-level semantic features.

```
Without FPN:                           With FPN:
Only detect at one scale               Detect at all scales simultaneously

C5 (low-res, high-semantic) → detect   P5 → large objects (tables, full pages)
                                        P4 → medium objects (paragraphs, figures)
                                        P3 → small-medium objects (signatures, stamps)
                                        P2 → small objects (text lines, icons)
```

**FPN assignment rule:** An ROI of width w and height h is assigned to pyramid level:

```
k = floor(k₀ + log₂(√(wh) / 224))

where k₀ = 4 (P4 is the canonical level for 224×224 ROIs)
```

#### 3.3.3 ROI Align vs. ROI Pooling

```
ROI Pooling (Faster R-CNN):                ROI Align (Mask R-CNN):
┌─────────────────────┐                    ┌─────────────────────┐
│ ROI on feature map   │                    │ ROI on feature map   │
│ x=3.75 → quantize→4 │                    │ x=3.75 → keep 3.75  │
│ y=2.3  → quantize→2 │                    │ y=2.3  → keep 2.3   │
│                      │                    │                      │
│ Division: quantize   │                    │ Division: exact      │
│ each bin boundary    │                    │ Bilinear interpolate │
│                      │                    │ at sample points     │
│ → Misaligned by up   │                    │ → Pixel-perfect      │
│   to 0.5 pixels      │                    │   alignment          │
└─────────────────────┘                    └─────────────────────┘

Impact: ROI Align improved mask AP by ~3% (from the original Mask R-CNN paper)
```

#### 3.3.4 Mask Head — Decoupled Prediction

**Critical design choice:** The mask head predicts a binary mask **per class**, independently of classification. This decouples mask prediction from class prediction.

```
Traditional approach: Predict one multi-class mask → classes compete for pixels
Mask R-CNN approach: Predict K binary masks (one per class) → use class head to select

This avoids inter-class competition and improves mask quality.
Loss is computed only on the mask corresponding to the ground-truth class.
```

---

### 3.4 Transfer Learning Strategy

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING PIPELINE                                │
│                                                                              │
│  Phase 1: Start with COCO-pretrained weights                                 │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  COCO Dataset: 330K images, 80 object classes            │                │
│  │  Pretrained Mask R-CNN: Strong general feature extraction │                │
│  │  Backbone (ResNet-101) + FPN + RPN already trained       │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
│  Phase 2: Freeze backbone, train heads (10-20 epochs)                        │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  Frozen: ResNet-101 backbone + FPN                       │                │
│  │  Trainable: RPN + Classification head + BBox head + Mask │                │
│  │  Learning rate: 0.001                                    │                │
│  │  Purpose: Adapt detection heads to document domain       │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
│  Phase 3: Unfreeze all, fine-tune end-to-end (30-50 epochs)                  │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │  All layers trainable                                    │                │
│  │  Learning rate: 0.0001 (10× lower to avoid catastrophic  │                │
│  │                         forgetting)                       │                │
│  │  LR schedule: Step decay at epochs 30, 40                │                │
│  │  Data augmentation: horizontal flip, rotation (±15°),    │                │
│  │                     brightness/contrast, random crop      │                │
│  │  Result: 85% mAP                                         │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Training Code (simplified):**

```python
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib

class DocumentConfig(Config):
    NAME = "document_detection"
    NUM_CLASSES = 1 + 5  # background + (stamp, signature, table, header, handwriting)
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    # Backbone
    BACKBONE = "resnet101"
    
    # RPN anchors - tuned for document elements
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Training
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Image sizing
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    
    # Detection
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    
    # Training proposals
    TRAIN_ROIS_PER_IMAGE = 200
    
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20

config = DocumentConfig()

# Load pretrained COCO weights
model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs")
model.load_weights("mask_rcnn_coco.h5", by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

# Phase 2: Train heads only
model.train(train_dataset, val_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers="heads")

# Phase 3: Fine-tune all layers
model.train(train_dataset, val_dataset,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=50,
            layers="all")
```

---

### 3.5 Mean Average Precision (mAP) — 85% Explained

#### What is mAP?

mAP is the primary metric for object detection. It measures both **localization accuracy** (are bounding boxes correct?) and **classification accuracy** (are labels correct?).

**Step-by-step mAP computation:**

```
Step 1: For each detection, compute IoU with ground truth

         Area of Overlap
  IoU = ─────────────────────
         Area of Union

         |A ∩ B|
       = ─────────
         |A ∪ B|

Step 2: At a given IoU threshold (e.g., 0.5), classify detections:
  - True Positive (TP): IoU ≥ threshold AND correct class
  - False Positive (FP): IoU < threshold OR wrong class
  - False Negative (FN): Ground truth with no matching detection

Step 3: Compute Precision and Recall at each detection (sorted by confidence):
  
           TP                    TP
  P = ──────────       R = ──────────
       TP + FP              TP + FN

Step 4: Plot Precision-Recall curve, compute AP (area under PR curve):
  
  AP = ∫₀¹ p(r) dr    (using 11-point or all-point interpolation)

Step 5: Average AP across all classes:

         1   C
  mAP = ─── Σ  APᵢ
         C  i=1

  where C = number of classes
```

**COCO mAP (our metric) averages over multiple IoU thresholds:**

```
mAP@[0.5:0.95] = average of mAP at IoU thresholds 0.50, 0.55, 0.60, ..., 0.95

This is stricter than Pascal VOC's mAP@0.5 (which only evaluates at IoU=0.5)
```

**Our 85% mAP breakdown (illustrative):**

| Class | AP@0.5 | AP@[0.5:0.95] |
|-------|--------|----------------|
| Stamp | 0.92 | 0.88 |
| Signature | 0.87 | 0.82 |
| Table | 0.91 | 0.86 |
| Header | 0.89 | 0.85 |
| Handwriting | 0.78 | 0.72 |
| **Mean** | **0.87** | **0.85** |

**Interview talking point:** "85% mAP at the COCO standard (IoU 0.5:0.95) is a strong result for a domain-specific detection task. For context, state-of-the-art on COCO general objects is around 50% mAP, but our narrower domain (5 document element classes) with consistent visual patterns allowed higher accuracy."

---

### 3.6 Azure Blob Storage Deployment

```
┌────────────────────────────────────────────────────────────────────┐
│                  INFERENCE DEPLOYMENT ARCHITECTURE                  │
│                                                                    │
│  ┌──────────────────────────────────┐                              │
│  │    Azure Blob Storage             │                              │
│  │    Container: model-artifacts     │                              │
│  │                                   │                              │
│  │    /models/                       │                              │
│  │      mask_rcnn_document_v1.h5     │  ← Model weights            │
│  │      config.json                  │  ← Hyperparameters          │
│  │      class_names.json             │  ← Label mapping            │
│  │    /inference-results/            │  ← Detection outputs        │
│  └──────────────┬───────────────────┘                              │
│                  │                                                  │
│                  ▼                                                  │
│  ┌──────────────────────────────────┐                              │
│  │    Inference Service              │                              │
│  │    (Azure VM / Container)         │                              │
│  │                                   │                              │
│  │    1. Pull model from Blob        │                              │
│  │    2. Load into TF Serving        │                              │
│  │    3. Accept document images      │                              │
│  │    4. Run Mask R-CNN inference     │                              │
│  │    5. Return:                     │                              │
│  │       - Bounding boxes            │                              │
│  │       - Class labels + confidence │                              │
│  │       - Pixel-level masks         │                              │
│  │    6. Write results back to Blob  │                              │
│  └──────────────────────────────────┘                              │
│                                                                    │
│  Azure Blob SDK (Python):                                          │
│  from azure.storage.blob import BlobServiceClient                  │
│  blob_client = BlobServiceClient.from_connection_string(conn_str)  │
│  container = blob_client.get_container_client("model-artifacts")   │
│  blob = container.get_blob_client("models/mask_rcnn_document_v1.h5")│
│  with open("local_model.h5", "wb") as f:                           │
│      f.write(blob.download_blob().readall())                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Why Azure Blob Storage (not Azure ML Registry, not S3)?**
- Client was on Azure ecosystem — Blob was already provisioned
- Simple versioning via blob naming (v1, v2, etc.) or blob snapshots
- Cost-effective for storing large model files (~250 MB per model)
- Direct integration with Azure VMs running inference
- No additional service overhead (vs. spinning up Azure ML)

---

## 4. Project C: Legal Document Classification (NLP)

### 4.1 Problem Context

Legal teams across ATCS clients needed to classify incoming documents (contracts, court filings, briefs, memos, compliance documents) into predefined categories for routing, archival, and compliance tracking. Manual classification was:
- **Slow:** Paralegal teams spending hours per batch
- **Inconsistent:** Different annotators applied labels differently
- **Costly:** Senior legal staff reviewing edge cases

**Goal:** Build an automated classification pipeline with transfer learning and validate labeling accuracy.

---

### 4.2 NLP Preprocessing Pipeline

Legal text has unique characteristics that require domain-specific preprocessing:

```python
import spacy
import re
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_lg")

class LegalTextPreprocessor:
    """Domain-aware preprocessing for legal documents."""
    
    # Legal-specific stopwords (common but non-discriminative in legal text)
    LEGAL_STOPWORDS = {
        "herein", "hereinafter", "thereof", "thereby", "whereas",
        "pursuant", "notwithstanding", "aforementioned", "undersigned",
        "witnesseth", "hereunder", "heretofore"
    }
    
    # Legal entity patterns
    LEGAL_ENTITY_PATTERNS = [
        r"(?:Section|§)\s*\d+[\.\d]*",           # Section references
        r"(?:Article|Art\.)\s+[IVXLC]+",           # Article references  
        r"\d+\s+U\.?S\.?C\.?\s+§?\s*\d+",         # USC citations
        r"\d+\s+F\.\s*(?:2d|3d|4th)\s+\d+",       # Federal Reporter
        r"[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+",       # Case names
    ]
    
    def __init__(self, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.max_length = max_length
    
    def clean_text(self, text):
        """Remove formatting artifacts from legal documents."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_legal_entities(self, text):
        """Extract legal citations and references."""
        entities = []
        for pattern in self.LEGAL_ENTITY_PATTERNS:
            entities.extend(re.findall(pattern, text))
        
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "LAW", "DATE"):
                entities.append(f"{ent.label_}:{ent.text}")
        
        return entities
    
    def segment_sections(self, text):
        """Split legal document into logical sections."""
        section_pattern = r'(?:^|\n)(?:ARTICLE|SECTION|CLAUSE|WHEREAS|NOW\s+THEREFORE)\s+'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        return [s.strip() for s in sections if s.strip()]
    
    def preprocess(self, text):
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        entities = self.extract_legal_entities(text)
        sections = self.segment_sections(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "entities": entities,
            "num_sections": len(sections)
        }
```

---

### 4.3 Transfer Learning for Legal Document Classification

#### Why Transfer Learning for NLP?

```
From-scratch training:                 Transfer learning:
─────────────────────                  ───────────────────
• Need millions of labeled samples     • Need hundreds to thousands
• Weeks of training on GPU clusters    • Hours of fine-tuning on single GPU
• No language understanding            • Pre-learned syntax, semantics, reasoning
• Risk: poor generalization            • Strong baseline from pretraining
```

#### Model Selection

| Model | Parameters | Pretraining | Legal Domain? | Our Choice |
|-------|-----------|-------------|---------------|------------|
| BERT-base | 110M | General English (BooksCorpus, Wikipedia) | No | Baseline |
| **Legal-BERT** | 110M | Legal corpora (case law, legislation, contracts) | **Yes** | **Selected** |
| RoBERTa | 125M | General English (larger corpus) | No | Considered |
| Longformer | 149M | Long documents (up to 4096 tokens) | No | For long docs |

**Legal-BERT was selected** because it was pretrained on 12 GB of legal text (EU legislation, UK/US case law, contracts), giving it superior understanding of legal terminology, citation patterns, and clause structures.

#### Fine-Tuning Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LegalDocumentClassifier(nn.Module):
    """Multi-label legal document classifier using Legal-BERT."""
    
    def __init__(self, num_labels, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits

# Document categories
LEGAL_CATEGORIES = [
    "contract", "court_filing", "legal_brief", "memo",
    "compliance", "patent", "regulatory", "correspondence"
]

model = LegalDocumentClassifier(num_labels=len(LEGAL_CATEGORIES))

# Class-weighted loss for imbalanced data
class_weights = torch.tensor([1.0, 2.5, 1.8, 1.2, 3.0, 2.0, 2.2, 1.5])
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=500  # warmup
)
```

#### Training Loop

```python
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).int()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    report = classification_report(
        all_labels, all_preds,
        target_names=LEGAL_CATEGORIES,
        zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return macro_f1, report
```

---

### 4.4 Document Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   DOCUMENT INGESTION PIPELINE                           │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ Scanned PDFs│  │ Digital PDFs│  │ Word Docs   │                     │
│  │ (images)    │  │ (text-based)│  │ (.docx)     │                     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
│         │                │                │                             │
│         ▼                ▼                ▼                             │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Text Extraction Module                   │                  │
│  │  • Scanned → OCR (Tesseract / Azure Form Recog.) │                  │
│  │  • Digital PDF → PyMuPDF / pdfplumber             │                  │
│  │  • DOCX → python-docx                             │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          NLP Preprocessing                        │                  │
│  │  • Legal tokenization                             │                  │
│  │  • Entity extraction                              │                  │
│  │  • Section segmentation                           │                  │
│  │  • Feature extraction                             │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Classification Model                     │                  │
│  │  • Legal-BERT inference                           │                  │
│  │  • Multi-label prediction                         │                  │
│  │  • Confidence scoring per label                   │                  │
│  └──────────────────────┬───────────────────────────┘                  │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────┐                  │
│  │          Validation & Routing                     │                  │
│  │  • Confidence thresholding                        │                  │
│  │  • Low-confidence → human review queue            │                  │
│  │  • High-confidence → auto-tagged and routed       │                  │
│  │  • Results logged for labeling accuracy tracking  │                  │
│  └──────────────────────────────────────────────────┘                  │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 4.5 Labeling Accuracy Validation

A core deliverable was validating that model labels matched human annotations reliably.

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    cohen_kappa_score, f1_score
)

class LabelingAccuracyValidator:
    """Validate model predictions against human annotations."""
    
    def __init__(self, class_names):
        self.class_names = class_names
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """Per-class confusion matrix for multi-label."""
        results = {}
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            tn, fp, fn, tp = cm.ravel()
            results[class_name] = {
                "TP": int(tp), "FP": int(fp), 
                "FN": int(fn), "TN": int(tn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            }
            p, r = results[class_name]["precision"], results[class_name]["recall"]
            results[class_name]["f1"] = 2*p*r / (p+r) if (p+r) > 0 else 0
        return results
    
    def compute_cohens_kappa(self, y_true, y_pred):
        """
        Cohen's Kappa measures inter-annotator agreement 
        (model vs. human), correcting for chance agreement.
        
        κ = (p_o - p_e) / (1 - p_e)
        
        where:
          p_o = observed agreement
          p_e = expected agreement by chance
        
        Interpretation:
          κ < 0.20  → Poor agreement
          0.20-0.40 → Fair
          0.41-0.60 → Moderate  
          0.61-0.80 → Substantial
          0.81-1.00 → Almost perfect
        """
        kappas = {}
        for i, class_name in enumerate(self.class_names):
            kappas[class_name] = cohen_kappa_score(y_true[:, i], y_pred[:, i])
        kappas["macro_avg"] = np.mean(list(kappas.values()))
        return kappas
    
    def full_validation_report(self, y_true, y_pred):
        """Generate comprehensive validation report."""
        cm_results = self.compute_confusion_matrix(y_true, y_pred)
        kappas = self.compute_cohens_kappa(y_true, y_pred)
        
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        
        report = {
            "per_class_metrics": cm_results,
            "cohens_kappa": kappas,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.class_names, zero_division=0
            )
        }
        return report
```

**Cohen's Kappa Formula:**

```
            p_o - p_e
    κ = ──────────────
            1 - p_e

where:
    p_o = (TP + TN) / N                          (observed agreement)
    p_e = (TP+FP)(TP+FN)/N² + (FN+TN)(FP+TN)/N²  (expected by chance)

Example:
    100 documents, model agrees with human on 85
    By chance, agreement would be 52%
    
    κ = (0.85 - 0.52) / (1 - 0.52) = 0.33 / 0.48 = 0.69 (Substantial agreement)
```

---

## 5. Topics You Must Know (Comprehensive Study Guide)

### 5.1 PySpark & Data Engineering

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **RDD vs DataFrame** | RDD is low-level untyped; DataFrame has schema + Catalyst optimization | Explain performance difference, when to use each |
| **Transformations vs Actions** | Lazy vs eager; DAG construction; `explain()` to see plan | Shows understanding of Spark execution model |
| **Catalyst Optimizer** | Logical plan → optimized logical plan → physical plan → code gen | Predicate pushdown, column pruning, join reordering |
| **Tungsten Engine** | Off-heap memory, whole-stage code generation, cache-aware computation | Explains why Spark is fast |
| **Partitioning** | `repartition()` vs `coalesce()`; hash vs range partitioning | Shuffle performance, write optimization |
| **Broadcast Joins** | Small table broadcast to all executors; avoids shuffle | Critical for ETL joins with dimension tables |
| **Data Skew** | Uneven key distribution → some tasks take much longer | Salting, repartitioning, adaptive query execution |
| **JDBC Integration** | Connection pooling, batch size tuning, predicate pushdown | Production loading patterns |
| **Checkpointing** | Truncate lineage for long DAGs; `persist()` vs `checkpoint()` | Fault tolerance in long pipelines |
| **ETL Design Patterns** | Slowly Changing Dimensions, CDC, idempotency, deduplication | Production data engineering fundamentals |

### 5.2 Object Detection & Computer Vision

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **R-CNN Family Evolution** | R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN | Shows understanding of progression and why each improvement was needed |
| **Anchor Boxes** | Predefined boxes at each feature map location; scales + ratios | Core to RPN; tuning anchors affects detection quality |
| **Feature Pyramid Network** | Top-down + lateral connections; multi-scale detection | Detecting small and large objects simultaneously |
| **ROI Align vs ROI Pooling** | Bilinear interpolation vs quantization; alignment matters for masks | Key innovation in Mask R-CNN |
| **Non-Maximum Suppression** | Remove redundant detections; IoU threshold selection | Post-processing step; soft-NMS as alternative |
| **IoU (Intersection over Union)** | Overlap metric; used for matching detections to ground truth | Foundation of all detection metrics |
| **mAP Computation** | PR curve → AP → average across classes → average across IoU thresholds | Know PASCAL VOC vs COCO evaluation protocols |
| **Transfer Learning** | Pretrained backbone → freeze → fine-tune; learning rate scheduling | Standard practice; explain why it works |
| **Data Augmentation** | Geometric (flip, rotate, crop), photometric (brightness, contrast) | Critical for small datasets |
| **Instance vs Semantic Segmentation** | Instance: per-object masks; Semantic: per-pixel class labels | Explains why Mask R-CNN, not U-Net |

### 5.3 NLP & Text Classification

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **BERT Architecture** | Bidirectional encoder; [CLS] token; MLM + NSP pretraining | Foundation of transfer learning for NLP |
| **Tokenization** | WordPiece (BERT), BPE (GPT), SentencePiece (T5) | Subword tokenization handles OOV words |
| **Fine-Tuning Strategy** | Freeze/unfreeze layers; learning rate warmup; discriminative LR | Practical transfer learning knowledge |
| **Multi-Label Classification** | BCE loss per label; sigmoid (not softmax); threshold tuning | Different from multi-class; explain the distinction |
| **Class Imbalance** | Weighted loss, oversampling (SMOTE), focal loss, stratified splits | Common in real-world classification tasks |
| **Legal NLP Challenges** | Long documents, domain terminology, citation parsing, section structure | Domain-specific preprocessing knowledge |
| **Evaluation Metrics** | Precision, Recall, F1 (micro/macro/weighted), Cohen's Kappa | Know formulas and when to use each |
| **Active Learning** | Model identifies uncertain samples → human labels → retrain | Efficient use of annotation budget |
| **Named Entity Recognition** | spaCy, Hugging Face token classification; legal entity types | Preprocessing for document understanding |
| **Document Embeddings** | [CLS] token, mean pooling, max pooling; Sentence-BERT | Representing documents as vectors |

### 5.4 Azure & Deployment

| Topic | Key Concepts | Why It Matters |
|-------|-------------|----------------|
| **Azure Blob Storage** | Containers, blobs, access tiers (hot/cool/archive), SAS tokens | Model artifact storage and versioning |
| **Azure Blob SDK** | BlobServiceClient, upload/download, streaming, metadata | Practical code knowledge |
| **Model Versioning** | Naming conventions, blob snapshots, metadata tags | Production model management |
| **TF Serving** | SavedModel format, REST/gRPC endpoints, batching | Serving TensorFlow models at scale |
| **Inference Pipeline** | Model loading → preprocessing → inference → postprocessing → output | End-to-end deployment understanding |
| **Cost Optimization** | Right-sizing VMs, spot instances, access tier selection | Real-world deployment considerations |

### 5.5 Core Formulas You Must Know

```
PRECISION:      P = TP / (TP + FP)     "Of all positive predictions, how many were correct?"

RECALL:         R = TP / (TP + FN)     "Of all actual positives, how many did we find?"

F1 SCORE:       F1 = 2PR / (P + R)     "Harmonic mean of precision and recall"

IoU:            IoU = |A ∩ B| / |A ∪ B|   "Overlap between predicted and ground truth boxes"

mAP:            mAP = (1/C) Σ APᵢ         "Average of per-class Average Precision"

AP:             AP = ∫₀¹ p(r) dr           "Area under the Precision-Recall curve"

COHEN'S KAPPA:  κ = (p_o - p_e) / (1 - p_e)   "Agreement correcting for chance"

BCE LOSS:       L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]   "Binary cross-entropy per label"

SMOOTH L1:      L = { 0.5x²      if |x| < 1        "BBox regression loss"
                    { |x| - 0.5   otherwise

CROSS-ENTROPY:  L = -Σ yᵢ·log(ŷᵢ)                  "Classification loss"
```

---

## 6. Interview Questions & Answers

### Project A: PySpark ETL Pipeline

---

**Q1: Walk me through your PySpark ETL pipeline at ATCS.**

**A:** "At ATCS, I built a PySpark-based ETL pipeline that ingested data from multiple heterogeneous sources — CSVs, flat files, and database extracts — validated data quality rigorously, and loaded clean data into MSSQL for downstream analytics.

The pipeline had four stages:
1. **Extraction**: PySpark read from various formats using explicit schemas (never relying solely on inference in production) with corrupt record tracking.
2. **Validation**: A reusable DataQualityValidator class ran schema checks, null analysis, duplicate detection, range validation, and referential integrity checks. If critical checks failed, the pipeline halted before any data reached MSSQL.
3. **Transformation**: Standardization (trimming, type casting), derived column creation, and deduplication.
4. **Loading**: Batch writes to MSSQL via JDBC with tuned batch sizes (10,000 rows per batch, which we benchmarked as optimal). For upsert scenarios, we used a staging table approach with MSSQL's MERGE statement."

---

**Q2: Why PySpark instead of plain Python/Pandas?**

**A:** "Three reasons. First, data volumes exceeded single-machine memory — we were processing gigabytes per day, and Pandas would have required chunking and manual parallelism. Second, PySpark's Catalyst optimizer automatically optimized our query plans — things like predicate pushdown when reading from JDBC, column pruning, and join reordering. We got 2-5x better performance over equivalent Pandas operations without manual tuning. Third, PySpark gave us a clear path to scale horizontally. When data volumes grew, we could add executors without rewriting code."

---

**Q3: What data quality checks did you implement, and why?**

**A:** "I built a configurable DataQualityValidator framework with five categories of checks:

1. **Schema validation**: Verified column names, data types, and nullable constraints matched expected schemas. This caught upstream changes early.
2. **Null analysis**: Flagged columns exceeding configurable null thresholds — 0% for critical fields like primary keys, 5% for non-critical fields.
3. **Duplicate detection**: Identified duplicate rows based on composite keys using `dropDuplicates()`.
4. **Range validation**: Ensured numeric fields fell within business-valid ranges (e.g., confidence scores between 0 and 1, page counts between 1 and 10,000).
5. **Referential integrity**: Verified foreign keys existed in reference tables using left-anti joins.

The key design decision was making this a pre-load gate. If critical checks failed, the pipeline raised a `DataQualityError` and halted, ensuring no dirty data ever reached MSSQL. Warnings (like unexpected extra columns) were logged but didn't block the load."

---

**Q4: How did you handle the PySpark-to-MSSQL JDBC connection? Any performance issues?**

**A:** "JDBC writing was the main bottleneck initially. Three optimizations made a major difference:

1. **Batch size tuning**: Default JDBC batch size is 1,000 rows. We benchmarked different sizes and found 10,000 to be optimal — it reduced our 2M-row load from 45 minutes to 12 minutes. Going higher (50,000) gave diminishing returns and increased memory pressure.

2. **Partition-aware writing**: We repartitioned the DataFrame before writing to match the MSSQL table's natural partitioning, which improved parallelism.

3. **Staging + MERGE pattern**: For incremental loads, we wrote to a staging table first (overwrite mode), then used MSSQL's MERGE statement to handle upsert logic. This was more efficient than trying to do row-level upsert through JDBC."

---

**Q5: Explain the difference between ETL and ELT. Why did you choose ETL?**

**A:** "ETL transforms data in the processing engine before loading to the target. ELT loads raw data first and transforms inside the target database.

We chose ETL because MSSQL was used primarily for reporting and analytics — not as a data lake with powerful compute. Loading unvalidated data would have corrupted dashboards. With ETL, PySpark acted as a quality gate: data was cleaned, validated, and transformed before it ever entered MSSQL. If we'd been loading into Snowflake or BigQuery, ELT might have made sense because those systems have strong compute engines that can handle transformation efficiently."

---

**Q6: What are narrow vs. wide transformations in Spark? Why does it matter?**

**A:** "Narrow transformations — like `map`, `filter`, `select`, `withColumn` — can be computed on each partition independently without data movement. Wide transformations — like `groupBy`, `join`, `distinct`, `orderBy` — require shuffling data across the network to co-locate matching keys.

It matters because shuffles are the number one performance bottleneck in Spark. Each shuffle means serializing data, writing to disk, transferring over the network, and deserializing. In our pipeline, I minimized shuffles by filtering early (predicate pushdown), using broadcast joins for small dimension tables, and coalescing partitions instead of repartitioning when reducing partition count."

---

### Project B: Mask R-CNN

---

**Q7: Explain the Mask R-CNN architecture and why you chose it.**

**A:** "Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. The architecture has four key stages:

1. **Backbone (ResNet-101 + FPN)**: Extracts multi-scale feature maps. ResNet-101 provides deep feature extraction with residual connections that solve vanishing gradients. FPN creates a feature pyramid with top-down connections, enabling detection of objects at different scales.

2. **Region Proposal Network (RPN)**: Slides over feature maps and proposes ~2000 candidate regions using anchor boxes. Each anchor gets an objectness score and bounding box refinement.

3. **ROI Align**: This is a key innovation over Faster R-CNN's ROI Pooling. Instead of quantizing ROI coordinates (which causes misalignment), ROI Align uses bilinear interpolation to sample features at exact floating-point positions. This is critical for mask quality — even 1-pixel misalignment degrades masks significantly.

4. **Three parallel heads**: Classification head (what class), bounding box head (refined coordinates), and mask head (28x28 binary mask per class).

I chose Mask R-CNN because we needed instance segmentation — not just bounding boxes. When a signature overlaps a stamp, we needed separate masks for each. YOLO and Faster R-CNN only give bounding boxes; U-Net gives semantic segmentation without distinguishing instances."

---

**Q8: Explain your transfer learning approach for Mask R-CNN.**

**A:** "I used a two-phase transfer learning strategy:

**Phase 1 — Head training (20 epochs)**: I loaded COCO-pretrained weights for the entire network but froze the ResNet-101 backbone and FPN. Only the RPN, classification head, bbox head, and mask head were trainable. I excluded the final classification and bbox layers from weight loading since our class count (5 document classes) differed from COCO (80 classes). Learning rate was 0.001.

**Phase 2 — Full fine-tuning (30 more epochs)**: I unfroze all layers and trained end-to-end with a 10x lower learning rate (0.0001) to avoid catastrophic forgetting. I used step decay at epochs 30 and 40.

The rationale is that early layers learn general features (edges, textures, shapes) that transfer well across domains. By fine-tuning with a lower learning rate, we preserve these general features while adapting higher layers to document-specific patterns."

---

**Q9: What does 85% mAP mean, and how is it calculated?**

**A:** "mAP — Mean Average Precision — is the standard metric for object detection. Here's how it works:

For each class, you rank all detections by confidence, compute precision and recall at each threshold, and plot the Precision-Recall curve. AP is the area under this curve. mAP averages AP across all classes.

Under the COCO evaluation protocol (which we used), mAP is computed at 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05, then averaged. This is stricter than PASCAL VOC which only evaluates at IoU 0.5.

Our 85% mAP means that, on average across our 5 classes and across all IoU thresholds, the model correctly detected and localized 85% of document elements. For context, state-of-the-art on COCO general objects is around 50% mAP, but our narrower domain with more consistent visual patterns enabled higher performance."

---

**Q10: Explain ROI Align and why it's better than ROI Pooling.**

**A:** "ROI Pooling in Faster R-CNN quantizes floating-point ROI coordinates to integer pixel positions. For example, if the ROI starts at x=3.75, it rounds to 4. This quantization happens twice — once when mapping the ROI to the feature map, and once when dividing into bins. The cumulative error can be several pixels, which doesn't matter much for classification but significantly degrades mask quality.

ROI Align avoids all quantization. It keeps ROI coordinates as floating-point, divides the ROI into bins at exact positions, and uses bilinear interpolation to sample 4 points per bin. This gives pixel-perfect alignment between the ROI and the feature map.

The original Mask R-CNN paper showed that ROI Align improved mask AP by about 3 percentage points over ROI Pooling — a substantial gain for a single architectural change."

---

**Q11: How did you deploy the model to Azure Blob Storage?**

**A:** "The deployment had three parts:

1. **Model packaging**: After training, I exported the Mask R-CNN model as a TensorFlow SavedModel format, along with a config file (hyperparameters, class names, preprocessing settings).

2. **Upload to Azure Blob**: Used the Azure Blob Storage SDK to upload model artifacts to a `model-artifacts` container with versioned naming. Each model version was a separate blob with metadata tags (training date, mAP, dataset version).

3. **Inference service**: An inference service running on an Azure VM pulled the latest model from Blob on startup, loaded it into memory, and exposed an API for document processing. For each incoming document image, it ran preprocessing, inference, NMS, and returned bounding boxes, class labels, confidence scores, and pixel-level masks.

Azure Blob was chosen over Azure ML Registry because the client's infrastructure was already Blob-centric, and the simplicity of blob storage fit our versioning needs without the overhead of a full MLOps platform."

---

### Project C: Legal Document Classification

---

**Q12: How did you approach legal document classification using NLP?**

**A:** "I built a multi-label classification pipeline with three main components:

1. **Domain-aware preprocessing**: Legal text is unique — full of citations, cross-references, archaic terms, and long, complex sentences. I built a preprocessing pipeline that handled legal-specific tokenization, extracted legal entities (case citations, statute references, party names), segmented documents into logical sections, and removed formatting artifacts.

2. **Transfer learning with Legal-BERT**: Rather than general BERT, I used Legal-BERT, which was pretrained on 12GB of legal text. This gave the model a strong understanding of legal terminology, citation patterns, and clause structures. I added a classification head (two FC layers with dropout) on top of the [CLS] token output and fine-tuned the entire model.

3. **Labeling accuracy validation**: I built a validation framework that compared model predictions against human annotations using per-class confusion matrices, F1 scores, and Cohen's Kappa to measure inter-rater agreement. Documents with low confidence were routed to a human review queue."

---

**Q13: Why Legal-BERT over standard BERT?**

**A:** "Standard BERT was pretrained on BooksCorpus and Wikipedia — general English text. Legal text has a fundamentally different distribution: different vocabulary (tort, estoppel, indemnification), different sentence structures (extremely long, nested clauses), different conventions (citation formats, section numbering).

Legal-BERT was pretrained on 12GB of legal text including EU legislation, UK case law, US contracts, and regulatory filings. In our experiments, Legal-BERT outperformed standard BERT by ~4-5% macro F1 on our classification task — a significant improvement that came 'for free' just by choosing the right pretrained model. The domain-specific vocabulary coverage meant fewer [UNK] tokens and better subword representations of legal terms."

---

**Q14: How does multi-label differ from multi-class classification?**

**A:** "In multi-class, each document belongs to exactly one class — you use softmax activation (probabilities sum to 1) and categorical cross-entropy loss. In multi-label, a document can belong to multiple classes simultaneously — a contract might also be tagged as 'compliance' and 'regulatory.'

For multi-label, I used sigmoid activation (independent probability per class, each between 0 and 1) and binary cross-entropy loss computed independently for each label. At inference, I applied a threshold (tuned via validation) to each sigmoid output to determine which labels to assign. This meant optimizing the threshold was important — too low gives false positives, too high gives false negatives."

---

**Q15: Explain Cohen's Kappa and why you used it.**

**A:** "Cohen's Kappa measures agreement between two raters (in our case, model vs. human annotator), correcting for the probability of agreement by chance.

The formula is κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is expected agreement by chance. If the model agrees with humans 85% of the time, but we'd expect 52% agreement by chance (due to class distribution), then κ = (0.85 - 0.52) / (1 - 0.52) = 0.69, which indicates substantial agreement.

I used it because accuracy alone is misleading with imbalanced labels. If 90% of documents are 'contracts', a model that predicts 'contract' for everything gets 90% accuracy but κ ≈ 0. Kappa exposes this. In our system, documents with low per-class Kappa were flagged for annotation review and model retraining."

---

### Cross-Project Questions

---

**Q16: How did the three projects at ATCS relate to each other?**

**A:** "They formed a cohesive document intelligence platform. The PySpark ETL pipeline handled structured data — metadata, client records, processing logs — bringing clean data into MSSQL for analytics. The Mask R-CNN model handled the visual layer — detecting and segmenting elements within scanned documents (stamps, signatures, tables). The NLP classification handled the textual layer — reading document content and classifying by type.

In practice, a scanned document would first go through Mask R-CNN to identify regions, then text extraction (OCR on detected text regions), then NLP classification to tag the document type, and the metadata would flow through the ETL pipeline into MSSQL for reporting and tracking."

---

**Q17: What was the most challenging part of working at ATCS?**

**A:** "The heterogeneity of client data. Each client had different document formats, different quality levels, different labeling conventions. Building reusable, configurable components was essential — the DataQualityValidator had configurable rules per client, the Mask R-CNN was trained on a combined dataset but with client-specific class mappings, and the NLP pipeline had modular preprocessing steps that could be toggled per document type.

The other challenge was annotation quality for the Mask R-CNN task. We had limited annotated data (a few hundred documents), which made transfer learning from COCO essential and data augmentation critical for preventing overfitting."

---

**Q18: How did you handle limited training data for Mask R-CNN?**

**A:** "Four strategies:

1. **Transfer learning**: Starting from COCO-pretrained weights meant the model already understood edges, textures, and spatial relationships. We only needed to teach it document-specific patterns.

2. **Aggressive data augmentation**: Horizontal flip, rotation (±15°), random brightness/contrast adjustment, and random cropping. This effectively multiplied our dataset.

3. **Two-phase training**: First training only the heads (with backbone frozen) meant we needed fewer samples to adapt the detection layers. Only then did we fine-tune the full network with a lower learning rate.

4. **Careful validation**: With limited data, I used stratified train/val/test splits and monitored for overfitting using validation mAP. Early stopping was applied when validation mAP plateaued."

---

**Q19: Walk me through a Spark optimization you performed.**

**A:** "The biggest optimization was addressing a data skew issue in one of our join operations. We were joining a large transaction table (~50M rows) with a client table (~5K rows) on client_id. The join was slow because certain clients had millions of transactions while others had tens.

The fix was a broadcast join. Since the client table was only ~5KB, I used `F.broadcast(client_df)` to send the entire client table to every executor. This eliminated the shuffle entirely — each executor could join its partition of the transaction table with the local copy of the client table. The join went from 15 minutes (with shuffle) to 45 seconds (with broadcast)."

---

**Q20: How would you improve the Mask R-CNN model beyond 85% mAP?**

**A:** "Several paths:

1. **More training data**: The biggest lever. More annotated documents would improve generalization, especially for underrepresented classes like handwriting.

2. **Better backbone**: Try ResNeXt-101 or a Swin Transformer backbone, which have shown improvements over ResNet in detection tasks.

3. **Anchor optimization**: Run K-means on ground truth bounding boxes to find optimal anchor scales and ratios for our specific document layout patterns.

4. **Multi-scale training**: Train with multiple input resolutions (800, 1024, 1333) to make the model more robust to document scanning variations.

5. **Test-time augmentation (TTA)**: Run inference on the image plus flipped/multi-scale versions and merge predictions. This typically adds 1-2% mAP.

6. **Soft-NMS**: Replace hard NMS with Soft-NMS, which decays confidence of overlapping detections instead of removing them entirely. Helps when document elements overlap."

---

**Q21: What's the difference between instance segmentation and semantic segmentation?**

**A:** "Semantic segmentation assigns a class label to every pixel, but doesn't distinguish between instances. If there are two signatures in a document, semantic segmentation labels all signature pixels identically — you can't tell them apart.

Instance segmentation assigns a class label AND a unique identity to each object. With Mask R-CNN, each signature gets its own bounding box, class label, and pixel-level mask. We needed this because downstream processing required extracting each document element individually — knowing that 'there are two signatures' and 'here is each one specifically' matters for verification workflows."

---

**Q22: Explain the multi-task loss function in Mask R-CNN.**

**A:** "Mask R-CNN uses a multi-task loss combining three components:

**L = L_cls + L_bbox + L_mask**

1. **L_cls (Classification loss)**: Cross-entropy loss over N+1 classes (N objects + background). Computed for each ROI.

2. **L_bbox (Bounding box regression loss)**: Smooth L1 loss on the 4 bounding box coordinates (dx, dy, dw, dh). Smooth L1 is less sensitive to outliers than L2:
   - If |x| < 1: loss = 0.5x²
   - Otherwise: loss = |x| - 0.5

3. **L_mask (Mask loss)**: Per-pixel binary cross-entropy, but only for the ground-truth class. This is the key insight — by decoupling mask prediction from classification, we avoid inter-class competition for pixels and improve mask quality.

The mask loss is computed on the predicted mask corresponding to the ground-truth class only, not on all K class masks. This is more efficient and avoids the need for the mask branch to implicitly re-learn classification."

---

**Q23: How did you ensure your ETL pipeline was idempotent?**

**A:** "Idempotency — running the pipeline twice produces the same result as running it once — was achieved through three mechanisms:

1. **Deduplication in the quality layer**: Before loading, we removed duplicates based on composite keys, so re-ingesting the same source data didn't create duplicate records.

2. **Staging + MERGE pattern**: For incremental loads, data went to a staging table first (overwrite mode clears it each run), then MERGE handled the upsert logic — updating existing records and inserting new ones. Running this twice on the same data just updates records to the same values.

3. **Watermarking**: For time-based incremental loads, we tracked a high-water mark (the latest timestamp successfully processed) and only ingested records newer than that mark. This prevented reprocessing and ensured monotonic progress."

---

**Q24: How did you evaluate model performance during the Mask R-CNN fine-tuning process?**

**A:** "I used a multi-metric evaluation approach during training:

1. **Primary metric — mAP@[0.5:0.95]**: The COCO-standard mAP computed on the validation set after each epoch. This was the metric I used for model selection and early stopping.

2. **Per-class AP breakdown**: I tracked AP for each class individually. This helped identify which classes were underperforming — for example, handwriting detection lagged behind stamps and signatures due to high visual variability.

3. **Loss curves**: I monitored training and validation losses for all three heads (classification, bbox, mask) separately. Divergence between training and validation loss indicated overfitting.

4. **Qualitative inspection**: For every validation epoch, I visualized predictions on a fixed set of 20 representative images. This caught issues that metrics alone might miss — like the model correctly detecting elements but with sloppy mask boundaries.

5. **Confidence calibration**: I checked that confidence scores correlated with actual precision. If the model assigned 0.9 confidence, roughly 90% of those predictions should be correct."

---

**Q25: How would you handle a scenario where the ETL pipeline fails mid-load?**

**A:** "Several safeguards:

1. **Transaction semantics**: For the MSSQL MERGE step, the entire operation runs within a transaction. If it fails mid-way, it rolls back — the target table is unchanged.

2. **Staging table isolation**: Data goes to a staging table first. If the MERGE fails, the staging table has the data and can be re-attempted without re-running the full Spark pipeline.

3. **Idempotent design**: Because of deduplication and the MERGE pattern, re-running the pipeline on the same data is safe and produces the correct result.

4. **Alerting and logging**: The pipeline logged each stage's completion with record counts. Any failure triggered an alert with the stage, error message, and a link to the full logs. This enabled quick diagnosis and re-run.

5. **Checkpointing**: For very long-running pipelines, I used Spark checkpointing to persist intermediate DataFrames to disk. If the pipeline crashed after the transform stage but before loading, we could resume from the checkpoint instead of re-reading and re-transforming all data."

---

## 7. Red Flags & How to Handle

### Red Flag 1: "PySpark for small data?"

**What they're probing:** Did you use PySpark where Pandas would have sufficed?

**How to handle:** "We started with Pandas for prototyping, but daily data volumes quickly exceeded single-node memory. Even with chunking, Pandas required manual parallelism and couldn't benefit from Catalyst optimization. PySpark was a forward-looking choice — we knew data would grow, and having a distributed pipeline from the start avoided a painful migration later."

---

### Red Flag 2: "85% mAP sounds high for limited data."

**What they're probing:** Is this number inflated by overfitting, test-set leakage, or a too-easy task?

**How to handle:** "85% mAP on COCO evaluation (IoU 0.5-0.95) is strong but reasonable for domain-specific detection. Our domain — scanned documents — has more consistent visual patterns than general object detection (COCO natural images). We achieved this through transfer learning from COCO (strong initialization), aggressive augmentation, and two-phase training. I validated with a held-out test set that was strictly separated from training data, and I monitored for overfitting via train/val loss divergence."

---

### Red Flag 3: "Why not just use a pre-built document AI service?"

**What they're probing:** Cost-benefit analysis, build vs. buy.

**How to handle:** "We evaluated Azure Form Recognizer and AWS Textract. Two issues: first, these services offered general-purpose document extraction but didn't have our specific class taxonomy (stamps, signatures, handwriting regions in the specific document types our clients handled). Second, the per-page pricing at our volume was significantly more expensive than running our own model on an Azure VM. Building custom gave us full control over the detection classes, threshold tuning, and model iteration cycle."

---

### Red Flag 4: "Did you write the Mask R-CNN implementation yourself?"

**What they're probing:** Do you understand the architecture or just call an API?

**How to handle:** "I used the Matterport Mask R-CNN implementation as a base, which implements the architecture in TensorFlow/Keras. My work focused on adapting it to our domain — creating the custom dataset class, tuning anchor configurations, implementing the two-phase training strategy, adding document-specific augmentations, and building the deployment pipeline. I have a deep understanding of the architecture — I can explain ROI Align, FPN, the multi-task loss, and why each design choice matters."

---

### Red Flag 5: "How did you validate the NLP model on legal documents when labels might be subjective?"

**What they're probing:** Annotation quality, evaluation rigor.

**How to handle:** "Subjectivity in legal document labels was a real challenge. We addressed it three ways: First, we created detailed labeling guidelines with examples for each category, especially for borderline cases. Second, we had multiple annotators label a subset of documents and computed Cohen's Kappa to measure inter-annotator agreement — only categories with substantial agreement (κ > 0.61) were kept. Third, I built a validation pipeline that flagged low-confidence predictions for human review, creating an active learning feedback loop that improved both the model and the annotation guidelines over time."

---

### Red Flag 6: "What about MSSQL vs. a modern cloud data warehouse?"

**What they're probing:** Technology choice awareness.

**How to handle:** "MSSQL was the client's existing infrastructure, and switching to Snowflake or BigQuery wasn't in scope. Within that constraint, we maximized MSSQL's capabilities — using columnstore indexes for analytical queries, table partitioning for efficient date-range queries, and the MERGE statement for efficient upserts. If I were designing this today with no constraints, I'd likely choose a modern cloud warehouse for the analytics layer, but the PySpark processing and data quality patterns would be the same."

---

### Red Flag 7: "Isn't FPN overkill for document detection?"

**What they're probing:** Do you understand why FPN is needed vs. just using a deeper backbone?

**How to handle:** "FPN was essential because document elements span a wide range of scales. A table might occupy 60% of the page, while a small stamp might be 5%. Without FPN, detection at a single scale would either miss small elements (if using high-level features) or fail to understand context for large elements (if using low-level features). FPN gives us strong features at every scale, which was critical for our ~10x range of object sizes."

---

## 8. Key Takeaways & Talking Points

### For Behavioral Questions

| Theme | Talking Point |
|-------|--------------|
| **End-to-end ownership** | "At ATCS, I owned three interconnected workstreams — from ETL pipeline design to computer vision model training to NLP classification deployment. This taught me to think in systems, not just models." |
| **Production mindset** | "The data quality validation framework was as important as any ML model. In production, bad data causes more damage than a suboptimal model." |
| **Transfer learning expertise** | "I applied transfer learning in two very different domains — CV (COCO → documents with Mask R-CNN) and NLP (general text → legal domain with BERT). The principles are the same: pretrain on large general data, fine-tune on domain-specific data, manage the learning rate carefully." |
| **Scale thinking** | "I chose PySpark from the start because I anticipated data growth. Forward-looking infrastructure decisions saved us from a painful migration later." |

### Technical Differentiators

| What Makes This Stand Out | Why |
|---------------------------|-----|
| Mask R-CNN is rarely discussed in interviews | Shows deep CV knowledge beyond simple classification |
| PySpark + data quality is a production data engineering skill | Not just "model.fit()" — real pipeline thinking |
| Transfer learning across both CV and NLP | Versatility and understanding of the common principle |
| Azure Blob deployment | Cloud deployment experience, not just notebook experiments |
| Multi-project narrative | Shows breadth and ability to work across the stack |

### 30-Second Elevator Pitch

"At ATCS, I was an Associate Data Scientist building a document intelligence platform. I designed a PySpark ETL pipeline with a reusable data quality validation framework that ensured zero dirty data reached our MSSQL analytics layer. I trained a Mask R-CNN model using transfer learning from COCO, achieving 85% mAP on detecting document elements like stamps, signatures, and tables — and deployed it to Azure Blob Storage for inference. I also built an NLP classification system using Legal-BERT for automated legal document tagging with labeling accuracy validation. Together, these three systems formed an end-to-end document processing pipeline — from ingestion and quality validation, through visual detection and text classification, to clean analytics-ready data."

### Questions to Ask the Interviewer (Show Depth)

1. "What does your data quality framework look like? Are validation rules defined declaratively or programmatically?"
2. "Are you using instance segmentation or just bounding box detection for your document processing pipeline?"
3. "How do you handle model versioning and rollback in production? Azure ML Registry, MLflow, or custom?"
4. "What's your approach to handling class imbalance in production NLP classifiers — weighted loss, oversampling, or something else?"

---

*Document prepared for Rahul Sharma — ATCS (Advanced Technology Consulting Service) interview preparation. Covers PySpark ETL, Mask R-CNN, and Legal Document Classification.*
