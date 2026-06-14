# Cloud, MLOps, and Deployment — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — AWS, GCP, Azure, Docker, Kubernetes, CI/CD, MLOps, LLM Deployment
**Deployed Projects:** PathWise (Railway), AgentOps RCA (Railway), Frontend Apps (Vercel)

---

# Table of Contents

1. [Cloud Platforms](#1-cloud-platforms)
2. [Docker](#2-docker)
3. [Kubernetes](#3-kubernetes)
4. [CI/CD](#4-cicd)
5. [MLOps](#5-mlops)
6. [ML Model Deployment](#6-ml-model-deployment)
7. [LLM Deployment](#7-llm-deployment)
8. [Monitoring and Observability](#8-monitoring-and-observability)
9. [Infrastructure as Code](#9-infrastructure-as-code)
10. [Security Best Practices for ML Systems](#10-security-best-practices-for-ml-systems)
11. [Cost Optimization Strategies](#11-cost-optimization-strategies)
12. [Your Deployment Experience](#12-your-deployment-experience)
13. [Common Interview Questions with Strong Answers](#13-common-interview-questions-with-strong-answers)
14. [Quick Reference Cards](#14-quick-reference-cards)
15. [Key Takeaways](#15-key-takeaways)

---

# **1. Cloud Platforms**

---

## **1.1 AWS (Amazon Web Services)**

AWS is the most mature cloud provider (~32% market share). For ML engineers, the key services are:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS FOR ML ENGINEERS                         │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐     │
│   │   EC2    │  │    S3    │  │  Lambda  │  │  SageMaker    │     │
│   │ Compute  │  │ Storage  │  │Serverless│  │ ML Platform   │     │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬────────┘     │
│        │              │              │               │              │
│   ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌──────┴────────┐     │
│   │  ECS/EKS │  │   RDS    │  │ Bedrock  │  │ CloudFormation│     │
│   │Container │  │ Database │  │ LLM APIs │  │     IaC       │     │
│   └──────────┘  └──────────┘  └──────────┘  └───────────────┘     │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐     │
│   │ DynamoDB │  │   ECR    │  │   IAM    │  │  CloudWatch   │     │
│   │ NoSQL DB │  │Container │  │ Security │  │  Monitoring   │     │
│   │          │  │ Registry │  │          │  │               │     │
│   └──────────┘  └──────────┘  └──────────┘  └───────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### EC2 (Elastic Compute Cloud)

Virtual machines in the cloud with full OS control.

| Instance Family | Use Case | Example |
|----------------|----------|---------|
| `t3` / `t4g` | Dev/test, small workloads | `t3.medium` (2 vCPU, 4 GB) |
| `c5` / `c6i` | CPU-intensive (feature engineering) | `c5.4xlarge` (16 vCPU) |
| `r5` / `r6i` | Memory-intensive (large datasets in RAM) | `r5.2xlarge` (64 GB RAM) |
| `p3` / `p4d` | GPU training (V100, A100) | `p3.2xlarge` (1x V100) |
| `g4dn` / `g5` | GPU inference (T4, A10G) | `g4dn.xlarge` (1x T4) |
| `g6` | Latest GPU inference (L4) | `g6.xlarge` (1x L4, 24 GB) |
| `inf1` / `inf2` | AWS Inferentia chips (cost-effective inference) | `inf2.xlarge` |
| `trn1` | AWS Trainium chips (cost-effective training) | `trn1.2xlarge` |

**Key concepts:**
- **AMIs (Amazon Machine Images):** Pre-configured OS images (e.g., Deep Learning AMI with PyTorch pre-installed)
- **Spot Instances:** Up to 90% cheaper, but can be interrupted — ideal for training jobs
- **Reserved Instances:** 1-3 year commitment, 30–72% savings for always-on inference
- **Auto Scaling Groups:** Automatically add/remove instances based on load
- **Placement Groups:** Control instance placement for low-latency GPU communication during distributed training

```python
import boto3

ec2 = boto3.client('ec2')

# Launch a GPU instance for training with spot pricing
response = ec2.run_instances(
    ImageId='ami-0abcdef1234567890',  # Deep Learning AMI
    InstanceType='g4dn.xlarge',        # 1x T4 GPU
    MinCount=1, MaxCount=1,
    InstanceMarketOptions={
        'MarketType': 'spot',
        'SpotOptions': {
            'SpotInstanceType': 'one-time',
            'MaxPrice': '0.30'  # Max hourly price
        }
    },
    TagSpecifications=[{
        'ResourceType': 'instance',
        'Tags': [{'Key': 'Project', 'Value': 'fraud-model-training'}]
    }]
)
instance_id = response['Instances'][0]['InstanceId']
print(f"Launched spot instance: {instance_id}")
```

### S3 (Simple Storage Service)

Object storage with virtually unlimited capacity. The backbone of AWS data pipelines.

```python
import boto3

s3 = boto3.client('s3')

# Upload a trained model
s3.upload_file('model.pt', 'ml-models-bucket', 'v1/model.pt')

# Download training data
s3.download_file('ml-data-bucket', 'train.parquet', 'local_train.parquet')

# List model versions
response = s3.list_objects_v2(Bucket='ml-models-bucket', Prefix='v')
for obj in response['Contents']:
    print(f"  {obj['Key']} — {obj['Size'] / 1e6:.1f} MB — {obj['LastModified']}")

# Generate a presigned URL for temporary access (model downloads)
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'ml-models-bucket', 'Key': 'v2/model.pt'},
    ExpiresIn=3600  # 1 hour
)
print(f"Temporary download link: {url}")
```

**Storage classes:**
- **S3 Standard:** Frequent access (active training data, recent model artifacts)
- **S3 Intelligent-Tiering:** Auto-moves between tiers based on access patterns
- **S3 Standard-IA:** Infrequent access (older training datasets)
- **S3 Glacier Instant Retrieval:** Archive with millisecond retrieval (compliance copies)
- **S3 Glacier Deep Archive:** Cheapest storage, 12-hour retrieval (regulatory backups)

**S3 for ML pipelines — best practices:**
- Use **versioning** for model artifacts (rollback to previous versions)
- Use **lifecycle policies** to auto-transition old data to cheaper tiers
- Enable **server-side encryption** (SSE-S3 or SSE-KMS) for sensitive training data
- Use **S3 Transfer Acceleration** for uploading large datasets from remote locations
- Structure keys as `s3://bucket/project/version/artifact` for organization

### Lambda (Serverless Compute)

Event-driven, auto-scaling functions. Pay only for execution time.

```python
# lambda_function.py — lightweight ML inference
import json
import boto3
import numpy as np

# Model loaded at cold start (stays warm for ~15 min)
model = None

def load_model():
    global model
    s3 = boto3.client('s3')
    s3.download_file('ml-models', 'classifier.pkl', '/tmp/model.pkl')
    import joblib
    model = joblib.load('/tmp/model.pkl')

def lambda_handler(event, context):
    if model is None:
        load_model()
    
    body = json.loads(event['body'])
    features = np.array(body['features']).reshape(1, -1)
    prediction = model.predict(features).tolist()
    probability = model.predict_proba(features).max().item()
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'prediction': prediction,
            'confidence': round(probability, 4),
            'remaining_time_ms': context.get_remaining_time_in_millis()
        })
    }
```

**Lambda limits to know:**
- Max 15 min execution, 10 GB memory, 10 GB ephemeral storage
- Cold start latency: 100ms–2s (mitigate with **Provisioned Concurrency**)
- **Container image support:** Deploy up to 10 GB Docker images (great for ML!)
- **Lambda SnapStart:** Pre-initialized execution environments (Java, .NET)
- 1000 concurrent executions by default (request increase for production)

**When Lambda fits for ML:**
- Lightweight models (scikit-learn, XGBoost) under 10 GB
- Low, bursty traffic (< 100 RPS average)
- Event-driven inference (S3 upload → predict → store results)
- NOT suitable for: GPU models, high-throughput, or latency-critical (<10ms)

### SageMaker

End-to-end ML platform — the AWS answer to Vertex AI.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAGEMAKER WORKFLOW                          │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Studio  │──→│ Training │──→│  Model   │──→│Endpoints │    │
│  │Notebooks │   │   Jobs   │   │ Registry │   │(Hosting) │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │               │           │
│  Data Wrangler  Experiments    Versioning     Auto-scaling     │
│  Feature Store  Hyperparameter Approval       A/B Testing      │
│  Ground Truth   Tuning         Gates          Shadow Testing   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │             SageMaker Pipelines (MLOps)               │      │
│  │  Orchestrate: Data Prep → Train → Eval → Deploy      │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**Key components:**
- **SageMaker Training Jobs:** Managed training on any instance type, data pulled from S3
- **SageMaker Endpoints:** Real-time inference with auto-scaling and A/B traffic routing
- **SageMaker Pipelines:** CI/CD for ML (DAG-based orchestration)
- **SageMaker Feature Store:** Centralized feature management (online + offline)
- **SageMaker Model Monitor:** Automated drift detection on deployed endpoints
- **SageMaker Clarify:** Bias detection and explainability
- **SageMaker JumpStart:** Pre-trained model hub (deploy LLMs with 1 click)

```python
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Launch a distributed training job
estimator = PyTorch(
    entry_point='train.py',
    source_dir='./src',
    role=role,
    instance_type='ml.p3.2xlarge',       # 1x V100 GPU
    instance_count=2,                     # Distributed across 2 instances
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'epochs': 50,
        'batch-size': 64,
        'learning-rate': 0.001
    },
    use_spot_instances=True,              # Up to 90% cost savings
    max_wait=7200,                        # Max wait for spot
    max_run=3600,                         # Max training time
    checkpoint_s3_uri='s3://my-bucket/checkpoints/'
)

estimator.fit({
    'train': 's3://my-bucket/data/train/',
    'val': 's3://my-bucket/data/val/'
})
```

### Bedrock (Managed LLM Service)

Access foundation models (Claude, Llama, Titan, Mistral) without managing infrastructure.

```python
import boto3, json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Invoke Claude on Bedrock
response = bedrock.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Explain MLOps in 3 sentences."}],
        "max_tokens": 256,
        "temperature": 0.7
    })
)
result = json.loads(response['body'].read())
print(result['content'][0]['text'])

# Streaming response (for chatbots)
response = bedrock.invoke_model_with_response_stream(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Write a deployment checklist."}],
        "max_tokens": 1024
    })
)

for event in response['body']:
    chunk = json.loads(event['chunk']['bytes'])
    if chunk['type'] == 'content_block_delta':
        print(chunk['delta'].get('text', ''), end='', flush=True)
```

**Bedrock additional features:**
- **Knowledge Bases:** RAG with managed vector store (auto-chunks, auto-embeds documents)
- **Agents:** Autonomous agents that can call APIs and query knowledge bases
- **Guardrails:** Content filtering, PII redaction, topic blocking
- **Fine-tuning:** Custom fine-tuning of select foundation models
- **Model evaluation:** Compare model outputs with human or automated evaluation

### RDS vs DynamoDB

| Feature | RDS | DynamoDB |
|---------|-----|----------|
| Type | Relational (SQL) | NoSQL (key-value / document) |
| Use Case | Structured data, complex queries, JOINs | High-throughput, low-latency, key-based access |
| Scaling | Vertical (bigger instance) + read replicas | Horizontal (auto-partitioning) |
| ML Use | Experiment metadata, user data, structured logs | Feature store, real-time features, session data |
| Pricing | Instance-based (always on) | Pay-per-request or provisioned capacity |
| Latency | ~1-5ms | Single-digit ms at any scale |
| Consistency | ACID transactions | Eventually consistent (or strong per-read) |

**Interview tip:** "For ML systems, I'd use RDS for experiment metadata and model registry data (complex queries, JOINs between experiments and metrics). I'd use DynamoDB for real-time feature serving (single-digit ms reads by user_id) and session state."

### ECS vs EKS

| Feature | ECS (Elastic Container Service) | EKS (Elastic Kubernetes Service) |
|---------|--------------------------------|----------------------------------|
| Orchestrator | AWS proprietary | Kubernetes (portable) |
| Complexity | Simpler, AWS-native | More complex, portable |
| Use When | AWS-only, simpler apps | Multi-cloud, complex microservices |
| Fargate | Serverless containers (no instance management) | Serverless containers (Fargate on EKS) |
| GPU Support | Yes (EC2 launch type) | Yes (GPU node groups) |
| Cost | Slightly cheaper | ~$0.10/hr for control plane + worker nodes |
| Ecosystem | AWS-only tools | Helm, ArgoCD, Istio, entire K8s ecosystem |

---

## **1.2 GCP (Google Cloud Platform)**

GCP shines in data analytics and AI/ML — BigQuery and Vertex AI are best-in-class.

### Key Services

| Service | Equivalent to (AWS) | Purpose |
|---------|---------------------|---------|
| Compute Engine | EC2 | Virtual machines |
| Cloud Storage | S3 | Object storage |
| Cloud Functions | Lambda | Serverless functions |
| Vertex AI | SageMaker | ML platform (AutoML + custom training) |
| BigQuery | Redshift + Athena | Serverless data warehouse |
| GKE | EKS | Managed Kubernetes (best-in-class) |
| Cloud Run | Fargate | Serverless containers |
| Artifact Registry | ECR | Container/package registry |
| Cloud Build | CodeBuild | CI/CD |
| Pub/Sub | SNS + SQS | Messaging / event streaming |
| TPUs | No equivalent | Custom AI accelerators |

### Vertex AI

Google's unified ML platform. Strongest for AutoML and custom training.

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Deploy a model to an endpoint
model = aiplatform.Model.upload(
    display_name="fraud-detector-v2",
    artifact_uri="gs://my-bucket/model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    traffic_percentage=100,
    accelerator_type="NVIDIA_TESLA_T4",  # Optional GPU
    accelerator_count=1
)

# Get predictions
prediction = endpoint.predict(instances=[{"feature1": 0.5, "feature2": 1.2}])
print(f"Predictions: {prediction.predictions}")
```

**Vertex AI Pipelines (KubeFlow-based):**

```python
from google.cloud import aiplatform
from kfp.v2 import dsl, compiler

@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "scikit-learn"])
def train_model(dataset_path: str, model_output: dsl.Output[dsl.Model]):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib, os
    
    df = pd.read_csv(dataset_path)
    X, y = df.drop("target", axis=1), df["target"]
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    
    os.makedirs(model_output.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_output.path, "model.pkl"))

@dsl.pipeline(name="fraud-detection-pipeline")
def ml_pipeline(dataset_path: str = "gs://bucket/data.csv"):
    train_task = train_model(dataset_path=dataset_path)

compiler.Compiler().compile(ml_pipeline, "pipeline.json")
```

### BigQuery

Serverless, petabyte-scale analytics. Extremely powerful for ML feature engineering.

```sql
-- BigQuery ML: Train a model directly in SQL
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  max_iterations=50,
  learn_rate=0.1,
  data_split_method='AUTO_SPLIT'
) AS
SELECT
  user_tenure_days,
  total_purchases,
  avg_session_minutes,
  support_tickets_count,
  churned
FROM `project.dataset.user_features`;

-- Predict on new data
SELECT * FROM ML.PREDICT(MODEL `project.dataset.churn_model`,
  (SELECT * FROM `project.dataset.new_users`)
);

-- Evaluate model performance
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.churn_model`);

-- Export model for deployment outside BigQuery
SELECT * FROM ML.EXPORT_MODEL(MODEL `project.dataset.churn_model`,
  'gs://my-bucket/exported-model/'
);
```

### Cloud Run (Serverless Containers)

Deploy any containerized app with automatic scaling to zero.

```bash
# Build and deploy to Cloud Run (one command)
gcloud run deploy ml-api \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --port 8000 \
    --set-env-vars "MODEL_VERSION=v2,LOG_LEVEL=INFO"
```

```yaml
# service.yaml for Cloud Run (advanced)
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ml-inference-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"  # Always-on CPU
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/my-project/ml-api:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              memory: "4Gi"
              cpu: "2"
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 12
```

**Cloud Run strengths for ML:**
- Scales to zero (pay nothing when idle) — great for dev/staging
- GPU support (L4) in preview — enables LLM inference
- Scales up fast (new instances in seconds)
- Automatic HTTPS, custom domains, IAM-based auth

### GCP TPUs

Unique advantage — custom-designed AI accelerators:
- **TPU v4:** 275 TFLOPS BF16, excellent for transformer training
- **TPU v5e:** Cost-optimized for inference and fine-tuning
- **TPU Pods:** Up to 4096 chips interconnected for massive distributed training
- Used by Google to train PaLM, Gemini, and other large models

---

## **1.3 Azure (Microsoft Azure)**

Azure integrates deeply with the Microsoft ecosystem. Dominant in enterprise, strongest for OpenAI workloads.

### Key Services

| Service | Equivalent to (AWS) | Purpose |
|---------|---------------------|---------|
| Virtual Machines | EC2 | Compute (NCasT4_v3 for GPU) |
| Blob Storage | S3 | Object storage |
| Azure Functions | Lambda | Serverless |
| Azure Machine Learning | SageMaker | ML platform |
| Cosmos DB | DynamoDB + DocumentDB | Multi-model NoSQL (globally distributed) |
| AKS | EKS | Managed Kubernetes |
| Azure OpenAI Service | Bedrock | Managed LLM access (GPT-4, o1, DALL-E) |
| Azure AI Search | OpenSearch | Vector search + hybrid search |
| Azure DevOps | GitHub Actions + CodeBuild | CI/CD platform |
| Azure Container Apps | Cloud Run | Serverless containers |

### Azure ML

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.ai.ml.constants import AssetTypes

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# Register a model
model = ml_client.models.create_or_update(
    Model(
        name="fraud-detector",
        path="./model/",
        type=AssetTypes.CUSTOM_MODEL,
        description="Fraud detection model v2"
    )
)

# Create an online endpoint
endpoint = ManagedOnlineEndpoint(name="fraud-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy model (blue-green deployment)
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="fraud-endpoint",
    model=f"azureml:fraud-detector:1",
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route 100% traffic to "blue"
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### Azure OpenAI Service

The unique selling point — direct access to OpenAI models with enterprise features.

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-02-01",
    azure_endpoint="https://my-resource.openai.azure.com"
)

response = client.chat.completions.create(
    model="gpt-4-turbo",  # deployment name
    messages=[
        {"role": "system", "content": "You are a helpful ML assistant."},
        {"role": "user", "content": "Explain data drift in production ML systems."}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response.choices[0].message.content)
```

**Azure OpenAI advantages over direct OpenAI API:**
- Data stays within your Azure subscription (compliance)
- Virtual network integration (private endpoints)
- Content filtering customization
- Regional deployment (data residency)
- Enterprise SLA (99.9% uptime)

### Cosmos DB

Globally distributed, multi-model database with single-digit millisecond latency.

**Consistency levels (unique to Cosmos DB — interview favorite):**

| Level | Description | Use Case |
|-------|-------------|----------|
| **Strong** | Linearizable reads (always latest write) | Financial transactions |
| **Bounded Staleness** | Reads lag by at most K versions or T seconds | Leaderboards |
| **Session** | Consistent within a client session | Most popular — user profiles |
| **Consistent Prefix** | Reads never see out-of-order writes | Social feeds |
| **Eventual** | Lowest latency, no ordering guarantee | Likes count, view count |

**Interview insight:** "Cosmos DB's five consistency levels are a trade-off between latency and consistency. Session consistency is the sweet spot for most ML applications — it guarantees read-your-own-writes within a user session while keeping latency low."

---

## **1.4 Cloud Platform Comparison — When to Use Which**

| Criterion | AWS | GCP | Azure |
|-----------|-----|-----|-------|
| **Best for** | Broadest service catalog, enterprise | Data/ML, analytics, Kubernetes | Microsoft shops, hybrid cloud |
| **ML Platform** | SageMaker (most features) | Vertex AI (best AutoML) | Azure ML (best OpenAI integration) |
| **LLM Access** | Bedrock (Claude, Llama, Mistral) | Vertex AI (Gemini, Claude) | Azure OpenAI (GPT-4, o1, GPT-4o) |
| **Data Warehouse** | Redshift | BigQuery (best-in-class) | Synapse Analytics |
| **Kubernetes** | EKS | GKE (best-in-class, Autopilot) | AKS |
| **Serverless Containers** | Fargate (ECS/EKS) | Cloud Run (best DX) | Container Apps |
| **Vector Search** | OpenSearch | Vertex AI Vector Search | Azure AI Search |
| **Pricing** | Complex, many levers | Simpler, sustained discounts | EA discounts, hybrid benefits |
| **GPU Availability** | Broadest selection | TPUs (unique advantage) | Good for OpenAI workloads |
| **Custom Chips** | Inferentia/Trainium | TPUs | Maia (coming) |
| **Free Tier** | 12 months | $300 credit + always-free | $200 credit + 12 months |
| **Market Share** | ~32% | ~12% | ~23% |

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│               WHICH CLOUD SHOULD I USE?                         │
│                                                                  │
│  Need GPT-4 / OpenAI? ──────────────────→ Azure                │
│  Need BigQuery / best K8s? ─────────────→ GCP                  │
│  Need broadest services / ecosystem? ───→ AWS                  │
│  Need TPUs for large model training? ───→ GCP                  │
│  Enterprise with Microsoft stack? ──────→ Azure                │
│  Startup / general purpose? ────────────→ AWS (most resources) │
│  Cost-sensitive inference? ─────────────→ GCP (Cloud Run)      │
│  Multi-cloud strategy? ────────────────→ Kubernetes on any     │
│  RAG / vector search? ─────────────────→ Azure AI Search       │
│  Managed LLM with guardrails? ─────────→ AWS Bedrock           │
│  Cheapest training at scale? ──────────→ GCP TPUs or AWS Spot  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Cloud Reality

**In practice, most companies use 2+ clouds.** Common patterns:
- **AWS for compute + storage**, GCP for BigQuery analytics
- **Azure for OpenAI models**, AWS for everything else
- **GCP for ML training** (TPUs), AWS for serving
- **Kubernetes as the abstraction layer** — same YAML deploys to EKS, GKE, or AKS

---

# **2. Docker**

---

## **2.1 Containers vs Virtual Machines**

```
┌────────────────────────────────────────────────────────────────────┐
│           VIRTUAL MACHINES vs CONTAINERS                           │
│                                                                    │
│   Virtual Machine                    Container                     │
│   ┌──────────────┐                  ┌──────────────┐              │
│   │     App      │                  │     App      │              │
│   ├──────────────┤                  ├──────────────┤              │
│   │   Binaries   │                  │   Binaries   │              │
│   ├──────────────┤                  └──────┬───────┘              │
│   │  Guest OS    │                         │                      │
│   │ (full Linux) │               ┌─────────┴─────────┐           │
│   ├──────────────┤               │  Container Engine  │           │
│   │  Hypervisor  │               │    (Docker)        │           │
│   ├──────────────┤               ├───────────────────┤           │
│   │   Host OS    │               │     Host OS        │           │
│   ├──────────────┤               ├───────────────────┤           │
│   │  Hardware    │               │    Hardware        │           │
│   └──────────────┘               └───────────────────┘           │
│                                                                    │
│   Size: GBs           Size: MBs                                   │
│   Boot: minutes       Boot: seconds                               │
│   Isolation: strong   Isolation: process-level                    │
│   Overhead: high      Overhead: minimal                           │
└────────────────────────────────────────────────────────────────────┘
```

| Feature | Virtual Machine | Container |
|---------|----------------|-----------|
| **Size** | Gigabytes | Megabytes |
| **Startup** | Minutes | Seconds (or sub-second) |
| **Isolation** | Full OS isolation (hypervisor) | Process-level (namespaces, cgroups) |
| **Overhead** | High (full guest OS) | Minimal (shared kernel) |
| **Portability** | Limited (hypervisor-dependent) | Run anywhere Docker runs |
| **Density** | 10s per host | 100s per host |
| **Use Case** | Different OS, strong security requirements | Microservices, ML serving, CI/CD |

**How containers work (Linux kernel features):**
- **Namespaces:** Isolate PID, network, mount, user, IPC — each container sees its own world
- **Cgroups (Control Groups):** Limit CPU, memory, I/O per container
- **Union File System (OverlayFS):** Layer-based filesystem — share base layers between containers

**Interview insight:** "Containers share the host OS kernel, which makes them lightweight and fast. VMs include a full guest OS, giving stronger isolation but higher overhead. For ML model serving, containers are almost always the right choice — you get reproducibility, fast scaling, and consistent environments across dev, staging, and production."

---

## **2.2 Dockerfile — Building ML Images**

### Essential Dockerfile Instructions

| Instruction | Purpose | Example |
|------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11-slim` |
| `RUN` | Execute commands during build | `RUN pip install -r requirements.txt` |
| `COPY` | Copy files from host to image | `COPY ./app /app` |
| `ADD` | Copy + extract archives + download URLs | `ADD model.tar.gz /models/` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `EXPOSE` | Document which port the container listens on | `EXPOSE 8000` |
| `ENV` | Set environment variables | `ENV MODEL_PATH=/models/v1` |
| `ARG` | Build-time variables (not in final image) | `ARG PYTHON_VERSION=3.11` |
| `CMD` | Default command when container starts | `CMD ["uvicorn", "main:app"]` |
| `ENTRYPOINT` | Fixed command (CMD becomes arguments) | `ENTRYPOINT ["python"]` |
| `HEALTHCHECK` | Container health verification | `HEALTHCHECK CMD curl -f http://localhost:8000/health` |
| `USER` | Set the user to run as | `USER appuser` |
| `LABEL` | Add metadata | `LABEL version="2.1"` |

**CMD vs ENTRYPOINT:**
```dockerfile
# CMD: Easy to override at runtime
CMD ["uvicorn", "main:app", "--port", "8000"]
# docker run myimage uvicorn main:app --port 9000  ← replaces entire CMD

# ENTRYPOINT: Fixed command, CMD becomes default arguments
ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--port", "8000"]
# docker run myimage main:app --port 9000  ← only replaces CMD (arguments)
```

### Production ML Dockerfile

```dockerfile
# ─── Stage 1: Build dependencies ───
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ─── Stage 2: Production image ───
FROM python:3.11-slim AS production

# Metadata labels
LABEL maintainer="rahul@example.com"
LABEL version="2.1.3"
LABEL description="ML Inference API"

WORKDIR /app

# Copy only installed packages from builder (no compiler tools)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### GPU-Enabled Dockerfile for ML

```dockerfile
# GPU inference with NVIDIA CUDA
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA support
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Verify GPU access at build time
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run with GPU access
docker build -t ml-api:gpu .
docker run --gpus all -p 8000:8000 ml-api:gpu

# Or specify specific GPU
docker run --gpus '"device=0"' -p 8000:8000 ml-api:gpu
```

---

## **2.3 Docker Compose — Multi-Container Applications**

```yaml
# docker-compose.yml — ML API + Model Store + Monitoring
version: "3.9"

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        PYTHON_VERSION: "3.11"
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/latest
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/mldb
      - LOG_LEVEL=INFO
    volumes:
      - model-store:/models
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: mldb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  model-store:
  pgdata:
  redis-data:
  prometheus-data:
  grafana-data:
```

**Docker Compose commands:**

```bash
docker compose up -d                 # Start all services (detached)
docker compose ps                    # Check status
docker compose logs -f ml-api        # Follow logs for one service
docker compose down                  # Stop all services
docker compose down -v               # Stop and remove volumes
docker compose build --no-cache      # Rebuild without cache
docker compose exec ml-api bash      # Shell into running container
docker compose up --scale ml-api=3   # Run 3 instances of ml-api
```

---

## **2.4 Docker Best Practices**

### Multi-Stage Builds (Reduce Image Size by 60–80%)

```dockerfile
# BAD: Single stage — 1.2 GB image
FROM python:3.11
RUN apt-get update && apt-get install -y build-essential
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]

# GOOD: Multi-stage — 350 MB image
FROM python:3.11 AS builder
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .
CMD ["python", "main.py"]
```

### Layer Caching — Order Matters

```dockerfile
# BAD: Any code change invalidates pip cache
COPY . .
RUN pip install -r requirements.txt

# GOOD: Dependencies cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

**Rule:** Put things that change least frequently first (base image, system deps, pip install) and things that change most frequently last (application code).

### .dockerignore

```
# .dockerignore — exclude from build context
__pycache__/
*.pyc
.git/
.gitignore
.env
.env.*
*.md
tests/
data/raw/
notebooks/
.vscode/
*.egg-info/
wandb/
mlruns/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.log
```

### Docker Security Best Practices

```dockerfile
# 1. Use specific image tags (not :latest)
FROM python:3.11.7-slim-bookworm

# 2. Run as non-root
RUN useradd --create-home appuser
USER appuser

# 3. Don't store secrets in images — use runtime env vars or mounted secrets
# BAD: ENV API_KEY=secret123
# GOOD: Pass at runtime: docker run -e API_KEY=$API_KEY ...

# 4. Minimize installed packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# 5. Scan images for vulnerabilities
# docker scout cve myimage:latest
# trivy image myimage:latest
```

### Key Docker Principles Summary

1. **Use slim/alpine base images** — `python:3.11-slim` not `python:3.11`
2. **Multi-stage builds** to separate build-time (compilers) and runtime
3. **Copy `requirements.txt` before code** for layer caching
4. **Run as non-root user** in production
5. **One process per container** — don't run supervisor in containers
6. **Use `.dockerignore`** to minimize build context
7. **Pin specific image tags** (not `latest`) in production
8. **Add health checks** to every production container
9. **Don't store secrets** in images — use environment variables
10. **Scan images** for vulnerabilities before deployment

---

## **2.5 Docker Networking**

```
┌───────────────────────────────────────────────────────────────┐
│                  DOCKER NETWORK TYPES                          │
│                                                                │
│  bridge (default)    host           overlay         none      │
│  ┌────┐ ┌────┐      Container      Multi-host      No        │
│  │ C1 │ │ C2 │      shares host    networking      network   │
│  └─┬──┘ └─┬──┘      network        (Swarm/K8s)              │
│    └──┬───┘         stack                                     │
│   docker0                                                      │
│   bridge                                                       │
└───────────────────────────────────────────────────────────────┘
```

```bash
# Create a custom bridge network (recommended over default)
docker network create ml-network

# Run containers on same network (can communicate by name)
docker run -d --name ml-api --network ml-network ml-api:latest
docker run -d --name redis --network ml-network redis:7

# ml-api can now reach redis at: redis://redis:6379
# (DNS resolution by container name within the network)
```

---

# **3. Kubernetes**

---

## **3.1 Core Concepts**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KUBERNETES ARCHITECTURE                           │
│                                                                     │
│  ┌──────────────── Control Plane ────────────────┐                 │
│  │  API Server  │ Scheduler │ Controller │ etcd  │                 │
│  └──────────────────────────────────────────────┘                 │
│            │                                                        │
│  ┌─────── Worker Node 1 ───────┐  ┌─────── Worker Node 2 ───────┐ │
│  │ ┌─────┐  ┌─────┐  ┌─────┐  │  │ ┌─────┐  ┌─────┐           │ │
│  │ │ Pod │  │ Pod │  │ Pod │  │  │ │ Pod │  │ Pod │           │ │
│  │ │(API)│  │(API)│  │(Wkr)│  │  │ │(API)│  │(Wkr)│           │ │
│  │ └─────┘  └─────┘  └─────┘  │  │ └─────┘  └─────┘           │ │
│  │     kubelet │ kube-proxy    │  │     kubelet │ kube-proxy    │ │
│  └─────────────────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Pod** | Smallest deployable unit. One or more containers sharing network/storage | A single apartment |
| **Deployment** | Manages replicas of Pods, handles rolling updates and rollbacks | Building management company |
| **Service** | Stable network endpoint to access Pods (load balances across Pods) | Building's street address |
| **Ingress** | HTTP routing rules (paths, TLS termination, host-based routing) | Lobby directory |
| **Namespace** | Virtual cluster for isolation (dev, staging, prod) | Different floors |
| **ConfigMap** | Non-sensitive configuration (key-value or files) | Settings file |
| **Secret** | Sensitive data (API keys, passwords) — base64-encoded | Safe deposit box |
| **PersistentVolume** | Durable storage that survives pod restarts | External hard drive |
| **DaemonSet** | Runs one pod per node (monitoring agents, log collectors) | Building superintendent |
| **Job / CronJob** | Run-to-completion tasks (one-time or scheduled) | Maintenance worker |
| **StatefulSet** | Pods with stable identity and persistent storage | Reserved parking spots |

**Control Plane components:**
- **API Server:** Front door to K8s — all `kubectl` commands go here
- **etcd:** Distributed key-value store — stores all cluster state
- **Scheduler:** Assigns pods to nodes based on resource requirements
- **Controller Manager:** Runs control loops (ensures desired state = actual state)

---

## **3.2 Deploying an ML API on Kubernetes**

### Deployment Manifest

```yaml
# ml-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: production
  labels:
    app: ml-api
    version: v2
    team: ml-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # At most 1 extra pod during update
      maxUnavailable: 0     # Zero downtime
  template:
    metadata:
      labels:
        app: ml-api
        version: v2
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: ml-api
          image: myregistry/ml-api:v2.1.3
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          resources:
            requests:
              cpu: "500m"           # 0.5 CPU cores
              memory: "1Gi"
              nvidia.com/gpu: "1"   # Request 1 GPU
            limits:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"
          env:
            - name: MODEL_VERSION
              valueFrom:
                configMapKeyRef:
                  name: ml-config
                  key: model_version
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: ml-secrets
                  key: api_key
            - name: LOG_LEVEL
              value: "INFO"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            failureThreshold: 30    # 5 minutes for model loading
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 15"]  # Drain connections
      nodeSelector:
        gpu-type: nvidia-t4
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
```

**Three types of probes (know the difference):**
- **Startup Probe:** Checks during container startup — important for ML models that take time to load. Disables liveness/readiness until it passes.
- **Liveness Probe:** Is the container healthy? If it fails, K8s **restarts** the container.
- **Readiness Probe:** Is the container ready to receive traffic? If it fails, K8s **removes it from load balancing** (but doesn't restart).

### Service and Ingress

```yaml
# ml-api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
  namespace: production
spec:
  selector:
    app: ml-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP        # Internal only — Ingress handles external traffic
---
# ml-api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.mlservice.com
      secretName: ml-api-tls
  rules:
    - host: api.mlservice.com
      http:
        paths:
          - path: /v2/predict
            pathType: Prefix
            backend:
              service:
                name: ml-api-service
                port:
                  number: 80
```

**Service types:**
- **ClusterIP:** Internal only (default) — accessible only within the cluster
- **NodePort:** Exposes on each node's IP at a static port (30000–32767)
- **LoadBalancer:** Provisions a cloud load balancer (e.g., AWS ALB/NLB)
- **ExternalName:** Maps to an external DNS name (CNAME)

---

## **3.3 Horizontal Pod Autoscaler (HPA)**

Automatically scales pods based on CPU, memory, or custom metrics.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: inference_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100        # Can double pods
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300    # Wait 5 min before scaling down
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### KEDA (Kubernetes Event-Driven Autoscaling)

KEDA extends HPA with event-driven scaling — scale based on external metrics.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: ml-api-keda
spec:
  scaleTargetRef:
    name: ml-api
  minReplicaCount: 0       # Scale to zero when idle!
  maxReplicaCount: 20
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: inference_queue_depth
        query: sum(ml_api_queue_depth)
        threshold: "10"
    - type: rabbitmq
      metadata:
        queueName: inference-jobs
        queueLength: "5"
```

---

## **3.4 ConfigMaps and Secrets**

```yaml
# ConfigMap for non-sensitive configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: production
data:
  model_version: "v2.1.3"
  batch_size: "32"
  log_level: "INFO"
  feature_flags: |
    {
      "enable_new_model": true,
      "shadow_mode": false,
      "ab_test_percentage": 10
    }
---
# Secret for sensitive data (values are base64-encoded)
apiVersion: v1
kind: Secret
metadata:
  name: ml-secrets
  namespace: production
type: Opaque
data:
  api_key: c2VjcmV0LWtleS0xMjM=          # echo -n 'secret-key-123' | base64
  db_password: cGFzc3dvcmQxMjM=
```

**Better secret management in production:**
- **AWS Secrets Manager** + External Secrets Operator
- **HashiCorp Vault** + Vault Secrets Operator
- **Sealed Secrets** (encrypted in git, decrypted in cluster)

```yaml
# External Secrets Operator — sync from AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-api-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: ml-secrets
  data:
    - secretKey: api_key
      remoteRef:
        key: production/ml-api
        property: api_key
```

---

## **3.5 Helm Charts**

Helm is the package manager for Kubernetes — templated, reusable configurations.

```
my-ml-app/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default configuration values
├── values-staging.yaml # Environment-specific overrides
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   └── _helpers.tpl    # Template helpers
└── charts/             # Sub-chart dependencies
```

```yaml
# values.yaml
replicaCount: 3
image:
  repository: myregistry/ml-api
  tag: v2.1.3
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: "2"
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  host: api.mlservice.com
  tls: true

env:
  MODEL_VERSION: v2.1.3
  LOG_LEVEL: INFO
```

```yaml
# templates/deployment.yaml (templated)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-app.fullname" . }}
  labels:
    {{- include "ml-app.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "ml-app.selectorLabels" . | nindent 6 }}
  template:
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

```bash
# Install / upgrade
helm install ml-api ./my-ml-app --namespace production --values values-prod.yaml
helm upgrade ml-api ./my-ml-app --set image.tag=v2.2.0

# Rollback to previous revision
helm rollback ml-api 1

# Template locally (debug without deploying)
helm template ml-api ./my-ml-app --values values-prod.yaml

# List releases
helm list --namespace production
```

---

## **3.6 When to Use K8s vs Simpler Solutions**

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Single ML API, low traffic | **Railway / Cloud Run** | K8s is overkill — 10x simpler |
| Multiple microservices, team > 5 | **Kubernetes** | Worth the complexity for orchestration |
| Need GPU scheduling at scale | **Kubernetes** | GPU node pools, tolerations, scheduling |
| Startup / MVP phase | **Railway, Render, or Cloud Run** | Ship faster, migrate later |
| Enterprise, strict compliance | **Kubernetes** | Fine-grained RBAC, network policies, audit |
| Batch training jobs | **K8s Jobs** or cloud-native (SageMaker) | K8s if multi-cloud; SageMaker if AWS-only |
| Scale to zero needed | **Cloud Run** or **KEDA on K8s** | Cloud Run is simpler for this |
| Multi-cloud portability | **Kubernetes** | Same YAML deploys everywhere |

---

# **4. CI/CD**

---

## **4.1 GitHub Actions — Core Concepts**

```
┌─────────────────────────────────────────────────────────────────┐
│                 GITHUB ACTIONS ANATOMY                           │
│                                                                  │
│  Workflow (.github/workflows/ci.yml)                            │
│  ├── Trigger: on push to main, PR to main                       │
│  ├── Job 1: lint-and-test                                       │
│  │   ├── Step 1: Checkout code                                  │
│  │   ├── Step 2: Setup Python 3.11                              │
│  │   ├── Step 3: Cache pip dependencies                         │
│  │   ├── Step 4: Install dependencies                           │
│  │   ├── Step 5: Run linter (ruff)                              │
│  │   ├── Step 6: Run type checker (mypy)                        │
│  │   └── Step 7: Run tests (pytest + coverage)                  │
│  └── Job 2: build-and-deploy (needs: lint-and-test)             │
│      ├── Step 1: Build Docker image                             │
│      ├── Step 2: Scan image for vulnerabilities                 │
│      ├── Step 3: Push to registry                               │
│      ├── Step 4: Deploy to production                           │
│      └── Step 5: Post-deploy health check                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Terminology

| Term | Meaning |
|------|---------|
| **Workflow** | YAML file defining automated processes (`.github/workflows/`) |
| **Trigger (`on`)** | Event that starts workflow (push, PR, schedule, `workflow_dispatch`) |
| **Job** | Group of steps running on one runner (parallel by default) |
| **Step** | Individual task within a job |
| **Runner** | Machine that executes jobs (`ubuntu-latest`, `self-hosted`) |
| **Action** | Reusable unit of work (`actions/checkout@v4`) |
| **Secret** | Encrypted env variable (`${{ secrets.API_KEY }}`) — never in logs |
| **Artifact** | File produced by a workflow (test reports, built images) |
| **Matrix** | Run the same job with different configs (Python 3.10, 3.11, 3.12) |
| **Environment** | Deployment target with protection rules (approval, wait timer) |
| **Concurrency** | Control parallel runs (cancel in-progress on new push) |

---

## **4.2 Complete CI/CD Pipeline for ML**

```yaml
# .github/workflows/ml-pipeline.yml
name: ML API CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:          # Manual trigger

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true    # Cancel previous run on new push

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-api
  PYTHON_VERSION: "3.11"

jobs:
  # ─── Job 1: Code Quality ───
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install linters
        run: pip install ruff mypy
      
      - name: Run ruff (linting + formatting)
        run: |
          ruff check .
          ruff format --check .
      
      - name: Run type checking
        run: mypy --ignore-missing-imports src/

  # ─── Job 2: Tests ───
  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio httpx
      
      - name: Run tests with coverage
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379
        run: |
          pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=80 \
            --junitxml=test-results.xml \
            -v
      
      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: htmlcov/

  # ─── Job 3: ML Model Validation ───
  model-validation:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Validate model loading
        run: |
          python -c "
          from src.model import load_model
          model = load_model('models/latest/')
          print(f'Model loaded successfully: {type(model).__name__}')
          "
      
      - name: Run model quality checks
        run: |
          python scripts/validate_model.py \
            --model-path models/latest/ \
            --test-data data/holdout.parquet \
            --min-accuracy 0.90 \
            --max-latency-ms 100

  # ─── Job 4: Build and Push Docker Image ───
  build:
    runs-on: ubuntu-latest
    needs: [test, model-validation]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest
      
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64
      
      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          format: 'table'
          severity: 'HIGH,CRITICAL'
          exit-code: '1'

  # ─── Job 5: Deploy ───
  deploy:
    runs-on: ubuntu-latest
    needs: build
    environment: production    # Requires manual approval
    steps:
      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          curl -X POST \
            "https://backboard.railway.app/graphql/v2" \
            -H "Authorization: Bearer $RAILWAY_TOKEN" \
            -H "Content-Type: application/json" \
            -d '{"query": "mutation { serviceRedeploy(id: \"${{ secrets.RAILWAY_SERVICE_ID }}\") }"}'
      
      - name: Verify deployment health
        run: |
          echo "Waiting for deployment to start..."
          sleep 30
          for i in {1..30}; do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.myservice.com/health)
            if [ "$STATUS" = "200" ]; then
              echo "✓ Deployment healthy!"
              exit 0
            fi
            echo "Waiting for deployment... (attempt $i/30, status: $STATUS)"
            sleep 10
          done
          echo "✗ Deployment health check failed!"
          exit 1
      
      - name: Notify on success
        if: success()
        run: |
          echo "Deployment succeeded! Image: ${{ needs.build.outputs.image-tag }}"
      
      - name: Notify on failure
        if: failure()
        run: |
          echo "Deployment FAILED! Investigate immediately."
```

---

## **4.3 Deployment Strategies**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT STRATEGIES                              │
│                                                                     │
│  ROLLING UPDATE            BLUE-GREEN             CANARY            │
│  ┌────┐ ┌────┐            ┌────────┐             ┌────────┐        │
│  │v1  │→│v2  │            │  Blue  │ (current)   │ 95% v1 │        │
│  │v1  │→│v2  │            │  (v1)  │             │  5% v2 │        │
│  │v1  │ │v1  │            ├────────┤             ├────────┤        │
│  │v1  │ │v1  │            │ Green  │ (new)       │ 90% v1 │        │
│  └────┘ └────┘            │  (v2)  │             │ 10% v2 │        │
│                            └───┬────┘             ├────────┤        │
│  Gradual replacement       Switch traffic         │  0% v1 │        │
│  Zero-downtime             at once (instant)      │100% v2 │        │
│  Default K8s strategy      Easy rollback          └────────┘        │
│                            2x resources           Gradual shift     │
│                            needed                 Risk-based        │
└─────────────────────────────────────────────────────────────────────┘
```

| Strategy | How It Works | Pros | Cons | Best For |
|----------|-------------|------|------|----------|
| **Rolling Update** | Replace pods one-by-one | Zero downtime, resource efficient | Slow rollback, mixed versions during update | Default for most services |
| **Blue-Green** | Run two identical environments, switch traffic | Instant rollback, clean cutover | 2x infrastructure cost | Non-ML services, databases |
| **Canary** | Route small % of traffic to new version | Low risk, real-world validation | Complex routing, need good metrics | ML model deployments |
| **Shadow / Dark Launch** | Mirror traffic to new version (no user impact) | Safest, compare outputs side-by-side | Highest resource cost, complex setup | High-risk model changes |
| **A/B Testing** | Route specific user segments to different versions | Statistical comparison, business metrics | Requires feature flags, complex analysis | ML model comparison |

### Canary Deployment with Nginx Ingress

```yaml
# canary-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"    # 10% traffic to canary
spec:
  rules:
    - host: api.mlservice.com
      http:
        paths:
          - path: /predict
            pathType: Prefix
            backend:
              service:
                name: ml-api-canary-service    # Points to v2 pods
                port:
                  number: 80
```

### GitOps with ArgoCD

ArgoCD watches a git repo and auto-deploys when manifests change.

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-api
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/rahul/ml-api
    targetRevision: main
    path: k8s/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

**GitOps flow:** Developer pushes to git → ArgoCD detects change → syncs to cluster → no direct `kubectl apply` in production.

---

# **5. MLOps**

---

## **5.1 The ML Lifecycle**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML LIFECYCLE (MLOps)                          │
│                                                                     │
│   ┌──────┐   ┌──────────┐   ┌────────┐   ┌────────┐              │
│   │ Data │──→│ Feature  │──→│ Train  │──→│Evaluate│              │
│   │Ingest│   │Engineer  │   │ Model  │   │& Valid │              │
│   └──────┘   └──────────┘   └────────┘   └───┬────┘              │
│                                               │                    │
│                                          Pass gates?               │
│                                          │         │               │
│                                         Yes        No              │
│                                          │         │               │
│                                          ▼         ▼               │
│   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌────────┐         │
│   │ Retrain  │←──│ Monitor  │←──│ Serve  │   │ Debug  │         │
│   │(trigger) │   │& Alert   │   │& Scale │   │& Fix   │         │
│   └──────────┘   └──────────┘   └────────┘   └────────┘         │
│        │                                                          │
│        └──────── Continuous Loop ────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

**MLOps maturity levels:**

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Level 0** | Manual | Jupyter notebooks, manual deployment, no monitoring, no versioning |
| **Level 1** | ML Pipeline Automation | Automated training pipeline (Airflow/Prefect), manual deployment, basic monitoring |
| **Level 2** | CI/CD for ML | Automated testing + building + deployment, model registry, experiment tracking |
| **Level 3** | Full Automation | Auto-retraining on drift detection, A/B testing, shadow deployment, feature store |

**Key difference from traditional DevOps:**
- DevOps: Code changes → test → deploy
- MLOps: Code changes + **data changes** + **model changes** → test → deploy → **monitor for drift** → retrain

---

## **5.2 Experiment Tracking**

### MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="rf-v2-tuned"):
    # Log parameters
    params = {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5}
    mlflow.log_params(params)
    
    # Log dataset info
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", X_train.shape[1])
    
    # Train
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    
    # Log model artifact with signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_test, y_pred)
    
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="fraud-detector",
        signature=signature,
        input_example=X_test[:1]
    )
    
    # Log feature importance plot
    fig = plot_feature_importance(model, feature_names)
    mlflow.log_figure(fig, "feature_importance.png")
    
    # Log confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm_fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Metrics: {metrics}")
```

### Weights & Biases (W&B)

```python
import wandb
import torch

wandb.init(
    project="fraud-detection",
    name="transformer-v3",
    config={
        "architecture": "TabTransformer",
        "hidden_dim": 256,
        "num_heads": 8,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "epochs": 50,
        "optimizer": "AdamW",
        "scheduler": "cosine"
    },
    tags=["production-candidate", "transformer"]
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_metrics = evaluate(model, val_loader)
    
    # Log metrics (auto-creates beautiful dashboards)
    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_metrics["accuracy"],
        "val/f1": val_metrics["f1"],
        "val/auc": val_metrics["auc"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch": epoch
    })
    
    # Log sample predictions as a table
    if epoch % 10 == 0:
        table = wandb.Table(columns=["input", "true", "pred", "confidence"])
        for i in range(min(20, len(val_preds))):
            table.add_data(str(X_val[i][:5]), y_val[i], val_preds[i], val_confs[i])
        wandb.log({"predictions": table})

# Log final model
artifact = wandb.Artifact("fraud-model-v3", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)

wandb.finish()
```

**MLflow vs W&B Comparison:**

| Feature | MLflow | W&B |
|---------|--------|-----|
| **Hosting** | Self-hosted or Databricks managed | Cloud SaaS (free tier available) |
| **UI** | Functional, table-based | Beautiful, interactive dashboards |
| **Model Registry** | Built-in (stages: None→Staging→Production→Archived) | Artifacts + Model Registry |
| **Experiment Comparison** | Table + parallel coordinates plot | Sweeps, interactive charts |
| **Hyperparameter Tuning** | None built-in (use Optuna separately) | W&B Sweeps (Bayesian, grid, random) |
| **LLM Tracking** | MLflow Tracing (newer) | W&B Traces (mature) |
| **Best For** | Open-source, on-prem, Databricks shops | Teams, collaboration, LLM projects |
| **Cost** | Free (self-hosted) | Free tier, then $50+/user/month |

---

## **5.3 Model Registry**

A model registry is a central repository for managing ML model versions, stages, and metadata.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL REGISTRY                              │
│                                                                  │
│  Model: fraud-detector                                          │
│  ├── v1.0 [Archived]    — accuracy: 0.89 — 2024-01-15          │
│  ├── v2.0 [Production]  — accuracy: 0.93 — 2024-03-20          │
│  ├── v2.1 [Staging]     — accuracy: 0.94 — 2024-05-10          │
│  └── v3.0 [Development] — accuracy: 0.95 — 2024-06-01          │
│                                                                  │
│  Stage Transitions:                                              │
│  Development → Staging → Production → Archived                   │
│                 ↑                                                 │
│            (approval gate / automated tests)                     │
│                                                                  │
│  Metadata per version:                                           │
│  - Training dataset hash                                         │
│  - Hyperparameters                                               │
│  - Evaluation metrics (accuracy, F1, AUC)                       │
│  - Training duration                                             │
│  - Git commit hash                                               │
│  - Dependencies (requirements.txt hash)                         │
└─────────────────────────────────────────────────────────────────┘
```

```python
from mlflow import MlflowClient

client = MlflowClient()

# Register a new model version
model_uri = f"runs:/{run_id}/model"
model_version = client.create_model_version(
    name="fraud-detector",
    source=model_uri,
    run_id=run_id,
    description="RandomForest v2 with tuned hyperparameters"
)

# Transition model to staging (for testing)
client.transition_model_version_stage(
    name="fraud-detector",
    version=model_version.version,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="fraud-detector",
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True    # Auto-archive current production
)

# Load production model for serving
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")
predictions = model.predict(new_data)
```

---

## **5.4 Feature Stores**

A feature store is a centralized system for managing, serving, and sharing ML features.

```
┌─────────────────────────────────────────────────────────────────┐
│                       FEATURE STORE                              │
│                                                                  │
│  ┌──────────┐     ┌─────────────────┐     ┌──────────────┐     │
│  │ Raw Data │────→│ Feature Pipeline │────→│ Offline Store │     │
│  │ (S3, DB) │     │ (Spark, Flink)  │     │ (Parquet, DB) │     │
│  └──────────┘     └───────┬─────────┘     └──────┬───────┘     │
│                           │                       │              │
│                           │              Training data           │
│                           │                       ↓              │
│                           │              ┌──────────────┐        │
│                           └─────────────→│ Online Store  │       │
│                                          │ (Redis, DDB)  │       │
│                                          └──────┬───────┘       │
│                                                  │               │
│                                          Real-time serving       │
│                                                  ↓               │
│                                          ┌──────────────┐        │
│                                          │  ML Model    │        │
│                                          │  (Inference) │        │
│                                          └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**Why feature stores matter:**
- **Training-serving skew:** Ensure features computed for training are identical to those served in production
- **Feature reuse:** Multiple models share the same features (compute once, serve everywhere)
- **Point-in-time correctness:** Get features as they were at a specific timestamp (avoid data leakage)
- **Low-latency serving:** Online store (Redis, DynamoDB) serves features in milliseconds

**Popular feature stores:**
- **Feast** (open-source): Offline/online store, provider-agnostic, Kubernetes-native
- **SageMaker Feature Store:** AWS-native, tight SageMaker integration
- **Vertex AI Feature Store:** GCP-native, BigQuery integration
- **Tecton:** Enterprise-grade, real-time feature engineering, streaming features

```python
# Feast example
from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")

# Get historical features for training (point-in-time correct)
training_df = store.get_historical_features(
    entity_df=entity_df,    # DataFrame with entity IDs and timestamps
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration",
        "user_features:days_since_last_login",
        "transaction_features:avg_transaction_amount_7d",
        "transaction_features:num_transactions_24h"
    ]
).to_df()

# Get online features for real-time inference
online_features = store.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration",
        "transaction_features:avg_transaction_amount_7d"
    ],
    entity_rows=[{"user_id": "user_123"}]
).to_dict()

print(f"Features: {online_features}")
```

---

## **5.5 Model Serving — Batch vs Real-Time**

| Aspect | Batch Inference | Real-Time Inference | Near Real-Time |
|--------|----------------|-------------------|----------------|
| **Latency** | Minutes to hours | Milliseconds to seconds | Seconds to minutes |
| **Volume** | Large datasets at once | One request at a time | Mini-batches |
| **Infrastructure** | Spark, cron jobs, Airflow | API server, load balancer | Kafka + consumer |
| **Cost** | Cheaper per prediction | More expensive (always-on) | Moderate |
| **Use Case** | Recommendations, reports | Fraud detection, chatbots | Anomaly detection, alerts |
| **Freshness** | Stale (computed periodically) | Live (computed on demand) | Nearly live |
| **Scaling** | Scale compute to data | Scale compute to traffic | Scale consumers |

---

## **5.6 A/B Testing Models**

```
┌─────────────────────────────────────────────────────────────────┐
│                    A/B TESTING ML MODELS                         │
│                                                                  │
│                    ┌──────────────┐                              │
│  User Request ───→│ Load Balancer │                             │
│                    └──────┬───────┘                              │
│                     ┌─────┴─────┐                               │
│                     │  Router   │                               │
│                     │(50%/50%)  │                               │
│                     └──┬────┬───┘                               │
│                   ┌────┘    └────┐                              │
│                   ▼              ▼                               │
│            ┌──────────┐   ┌──────────┐                         │
│            │ Model A  │   │ Model B  │                         │
│            │ (control)│   │(variant) │                         │
│            └────┬─────┘   └────┬─────┘                         │
│                 │              │                                │
│                 └──────┬───────┘                                │
│                        ▼                                        │
│               ┌──────────────┐                                 │
│               │   Log Both   │  ← Record: model_id, input,    │
│               │   Results    │    prediction, confidence,      │
│               └──────┬───────┘    timestamp, user_id           │
│                      ▼                                          │
│            ┌───────────────────┐                               │
│            │ Statistical Test  │                               │
│            │ (t-test, χ² test) │                               │
│            │ p < 0.05? → Ship │                               │
│            └───────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

**Key metrics to track in ML A/B tests:**
- **Business metrics:** Click-through rate, conversion, revenue per user
- **Model metrics:** Accuracy, latency P50/P95, error rate, confidence distribution
- **Statistical significance:** Require p < 0.05 and sufficient sample size before deciding
- **Minimum detectable effect (MDE):** Determine smallest meaningful improvement
- **Power analysis:** Calculate required sample size before starting the test

---

## **5.7 Model Monitoring — Drift Detection**

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML MONITORING LANDSCAPE                      │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │   DATA DRIFT     │  │  CONCEPT DRIFT   │                    │
│  │                   │  │                   │                    │
│  │ Input distribution│  │ P(Y|X) changes   │                    │
│  │ P(X) changes     │  │ (relationship     │                    │
│  │                   │  │  between features │                    │
│  │ Example:         │  │  and target shifts)│                    │
│  │ Age distribution │  │                   │                    │
│  │ shifts in prod   │  │ Example:          │                    │
│  │                   │  │ Fraud patterns    │                    │
│  │ Detection:       │  │ evolve over time  │                    │
│  │ KS test, PSI,    │  │                   │                    │
│  │ Jensen-Shannon   │  │ Detection:        │                    │
│  │                   │  │ Performance decay,│                    │
│  └──────────────────┘  │ prediction drift  │                    │
│                         └──────────────────┘                    │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ PREDICTION DRIFT │  │  WHAT TO DO      │                    │
│  │                   │  │                   │                    │
│  │ P(Y_hat) changes │  │ 1. Alert team    │                    │
│  │ without input    │  │ 2. Root cause    │                    │
│  │ drift            │  │    analysis      │                    │
│  │                   │  │ 3. Retrain with  │                    │
│  │ Example: Model   │  │    recent data   │                    │
│  │ outputs more     │  │ 4. Deploy new    │                    │
│  │ "fraud" labels   │  │    model version │                    │
│  │ than usual       │  │ 5. Update        │                    │
│  │                   │  │    monitors      │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### Drift Detection with Evidently AI

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric

# Compare training data distribution vs production data
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset(),
    DatasetDriftMetric()
])

column_mapping = ColumnMapping(
    target="label",
    prediction="prediction",
    numerical_features=["feature_1", "feature_2", "feature_3"],
    categorical_features=["category_a", "category_b"]
)

report.run(
    reference_data=training_df,      # What the model was trained on
    current_data=production_df,      # Recent production data
    column_mapping=column_mapping
)

# Get results
result = report.as_dict()
drift_detected = result["metrics"][2]["result"]["dataset_drift"]
drift_share = result["metrics"][2]["result"]["drift_share"]

if drift_detected:
    print(f"⚠ DATA DRIFT DETECTED! {drift_share:.1%} of features drifted.")
    # Trigger retraining pipeline
    trigger_retraining_pipeline()
else:
    print(f"✓ No significant drift. {drift_share:.1%} feature drift (below threshold).")
```

### Population Stability Index (PSI)

PSI quantifies how much a variable's distribution has shifted:

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant shift | Continue monitoring |
| 0.1 – 0.2 | Moderate shift | Investigate, increase monitoring frequency |
| > 0.2 | Significant shift | Retrain model, investigate root cause |

$$
\text{PSI} = \sum_{i=1}^{n} (p_i^{\text{actual}} - p_i^{\text{expected}}) \times \ln\left(\frac{p_i^{\text{actual}}}{p_i^{\text{expected}}}\right)
$$

### Data Versioning with DVC

```bash
# Initialize DVC in your ML project
dvc init

# Track a large dataset (stored in S3, not git)
dvc add data/training_data.parquet

# Push data to remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push

# Reproduce the entire pipeline
dvc repro

# Compare metrics between branches
dvc metrics diff main..feature-branch
```

```yaml
# dvc.yaml — ML pipeline definition
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/
      - src/preprocess.py
    outs:
      - data/processed/
  
  train:
    cmd: python src/train.py
    deps:
      - data/processed/
      - src/train.py
    params:
      - train.n_estimators
      - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.csv
```

---

## **5.8 ML Pipeline Orchestration**

| Tool | Best For | Key Feature |
|------|----------|-------------|
| **Apache Airflow** | Complex DAGs, many integrations | Mature, large ecosystem, Python DAGs |
| **Prefect** | Modern Python-native orchestration | Easy setup, decorator-based, good DX |
| **Dagster** | Data-aware orchestration | Software-defined assets, strong typing |
| **SageMaker Pipelines** | AWS-native ML pipelines | Tight SageMaker integration |
| **Vertex AI Pipelines** | GCP-native (KubeFlow-based) | Best for GCP shops |
| **Kubeflow Pipelines** | Kubernetes-native | Good for K8s-heavy orgs |

```python
# Prefect example — ML training pipeline
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(path: str):
    import pandas as pd
    return pd.read_parquet(path)

@task
def preprocess(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("target", axis=1))
    y = df["target"].values
    return X, y, scaler

@task
def train_model(X, y, params: dict):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

@task
def evaluate(model, X_test, y_test) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}

@flow(name="fraud-detection-training")
def training_pipeline(data_path: str, params: dict):
    df = load_data(data_path)
    X, y, scaler = preprocess(df)
    model = train_model(X, y, params)
    metrics = evaluate(model, X, y)
    print(f"Training complete! Metrics: {metrics}")
    return model, metrics

# Run
training_pipeline(
    data_path="s3://bucket/data.parquet",
    params={"n_estimators": 200, "max_depth": 15}
)
```

---

# **6. ML Model Deployment**

---

## **6.1 REST API with FastAPI + Uvicorn**

The most common approach for real-time ML inference.

```python
# main.py — Production ML inference API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# Global model reference
model = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on {device}...")
    
    model = torch.jit.load("model.pt", map_location=device)
    model.eval()
    
    # Warm up the model (first inference is always slower)
    dummy = torch.randn(1, 10).to(device)
    with torch.no_grad():
        _ = model(dummy)
    
    logger.info("Model loaded and warmed up successfully")
    
    yield  # App runs here
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU memory cleared")

app = FastAPI(title="ML Inference API", version="2.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, max_length=100)

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.perf_counter()
    
    try:
        input_tensor = torch.tensor([request.features], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        latency = (time.perf_counter() - start) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            model_version="v2.1.3",
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    if gpu_available:
        gpu_memory = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "gpu_available": gpu_available,
        "gpu_memory_used": gpu_memory
    }

@app.get("/ready")
async def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}
```

```bash
# Run with Uvicorn (production settings)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Or with Gunicorn (process manager) + Uvicorn (ASGI workers)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 \
    --timeout 120 --graceful-timeout 30 --keep-alive 5
```

---

## **6.2 Serverless Deployment (Lambda / Cloud Functions)**

Best for: low traffic, spiky workloads, cost optimization.

```python
# AWS Lambda handler with container image
import json
import torch
import os

# Model loaded at cold start
model = None

def load_model():
    global model
    model_path = os.environ.get("MODEL_PATH", "/opt/model/model.pt")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

def handler(event, context):
    if model is None:
        load_model()
    
    body = json.loads(event.get("body", "{}"))
    features = torch.tensor([body["features"]], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "cold_start": context.get_remaining_time_in_millis() > 14000,
            "remaining_time_ms": context.get_remaining_time_in_millis()
        })
    }
```

**GCP Cloud Function equivalent:**

```python
import functions_framework
import torch
import json

model = None

@functions_framework.http
def predict(request):
    global model
    if model is None:
        model = torch.jit.load("model.pt", map_location="cpu")
        model.eval()
    
    data = request.get_json()
    features = torch.tensor([data["features"]], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, dim=1).item()
    
    return json.dumps({"prediction": prediction})
```

---

## **6.3 Batch Inference**

```python
# batch_inference.py — Process large dataset with a trained model
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def batch_predict(model_path: str, data_path: str, output_path: str, batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    df = pd.read_parquet(data_path)
    features = torch.tensor(df.drop("id", axis=1).values, dtype=torch.float32)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Predicting"):
            batch = batch.to(device)
            output = model(batch)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
            all_predictions.extend(preds)
            all_confidences.extend(confs)
    
    df["prediction"] = all_predictions
    df["confidence"] = all_confidences
    df[["id", "prediction", "confidence"]].to_parquet(output_path)
    print(f"Wrote {len(df)} predictions to {output_path}")

if __name__ == "__main__":
    batch_predict(
        model_path="models/fraud_detector_v2.pt",
        data_path="s3://data/daily_transactions.parquet",
        output_path="s3://predictions/daily_scores.parquet"
    )
```

---

## **6.4 Streaming Inference**

For real-time event processing (fraud detection, anomaly detection).

```python
# Kafka consumer for streaming inference
from confluent_kafka import Consumer, Producer
import json, torch

consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'ml-inference-group',
    'auto.offset.reset': 'latest'
})
consumer.subscribe(['transactions'])

producer = Producer({'bootstrap.servers': 'kafka:9092'})

model = torch.jit.load("fraud_model.pt")
model.eval()

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Consumer error: {msg.error()}")
        continue
    
    transaction = json.loads(msg.value())
    features = torch.tensor([transaction['features']], dtype=torch.float32)
    
    with torch.no_grad():
        score = torch.sigmoid(model(features)).item()
    
    result = {
        'transaction_id': transaction['id'],
        'fraud_score': round(score, 4),
        'is_fraud': score > 0.8,
        'model_version': 'v2.1.3'
    }
    
    producer.produce('predictions', json.dumps(result).encode())
    producer.flush()
    
    if result['is_fraud']:
        producer.produce('fraud-alerts', json.dumps(result).encode())
        producer.flush()
```

---

## **6.5 Edge Deployment**

Deploying models on edge devices (mobile, IoT, embedded systems).

| Framework | Target | Model Format | Use Case |
|-----------|--------|-------------|----------|
| **ONNX Runtime** | CPU, GPU, mobile, web | `.onnx` | Cross-platform inference |
| **TensorFlow Lite** | Mobile, microcontrollers | `.tflite` | Android/iOS apps, IoT |
| **Core ML** | Apple devices | `.mlmodel` | iOS/macOS/watchOS apps |
| **TorchScript** | Any PyTorch runtime | `.pt` | Server + mobile |
| **TensorRT** | NVIDIA GPUs | `.engine` | Maximum GPU throughput |
| **OpenVINO** | Intel hardware | IR format | Intel CPUs, iGPUs, VPUs |

### ONNX Export and Inference

```python
import torch
import onnxruntime as ort
import numpy as np

# Export PyTorch model to ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["features"],
    output_names=["prediction"],
    dynamic_axes={
        "features": {0: "batch_size"},
        "prediction": {0: "batch_size"}
    },
    opset_version=17
)

# Run inference with ONNX Runtime (2-5x faster than PyTorch on CPU)
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_data = np.random.randn(1, 10).astype(np.float32)
result = session.run(None, {"features": input_data})
print(f"Prediction: {result[0]}")
```

---

## **6.6 Multi-Model Serving with NVIDIA Triton**

```
# model_repository structure
model_repository/
├── fraud_detector/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── recommendation/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── text_classifier/
    ├── config.pbtxt
    └── 1/
        └── model.savedmodel/
```

```bash
# Run Triton Inference Server
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $PWD/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models
```

**Triton advantages:** Serves ONNX, PyTorch, TensorFlow, TensorRT models simultaneously from a single server. Dynamic batching, model ensembles, GPU scheduling.

---

## **6.7 Model Format Comparison**

| Format | Framework | Pros | Cons |
|--------|-----------|------|------|
| **ONNX** | Any → Any | Cross-platform, optimized runtimes | May lose custom ops |
| **TorchScript** | PyTorch | Native PyTorch, JIT compilation | PyTorch-only ecosystem |
| **SavedModel** | TensorFlow | Full TF ecosystem, TF Serving | Large file size |
| **TensorRT** | NVIDIA GPUs | Maximum throughput (2-6x speedup) | GPU-only, NVIDIA-only, compile per GPU |
| **Core ML** | Apple | Best on Apple hardware | Apple ecosystem only |
| **GGUF** | llama.cpp | CPU-optimized, quantized | Primarily for LLMs |

---

# **7. LLM Deployment**

---

## **7.1 vLLM — High-Performance LLM Serving**

vLLM uses **PagedAttention** — inspired by OS virtual memory — to manage KV cache efficiently.

```
┌─────────────────────────────────────────────────────────────────┐
│                   vLLM ARCHITECTURE                              │
│                                                                  │
│  ┌────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  Incoming  │───→│   Scheduler      │───→│  GPU Workers   │  │
│  │  Requests  │    │                  │    │                │  │
│  └────────────┘    │ • Continuous     │    │ • PagedAttn    │  │
│                     │   batching      │    │ • KV Cache     │  │
│  ┌────────────┐    │ • Priority queue │    │   (paged)      │  │
│  │  Streaming │←───│ • Preemption     │←───│ • Tensor       │  │
│  │  Responses │    │ • Prefix caching │    │   Parallel     │  │
│  └────────────┘    └──────────────────┘    └────────────────┘  │
│                                                                  │
│  Key Innovation: PagedAttention                                 │
│  • KV cache stored in non-contiguous memory blocks (pages)     │
│  • Eliminates memory fragmentation (wastes ~60-80% in naive)   │
│  • Enables 2-4x higher throughput vs HuggingFace generate()    │
│  • Supports continuous batching (add requests mid-generation)  │
│  • Prefix caching: share KV cache for common system prompts    │
└─────────────────────────────────────────────────────────────────┘
```

```python
# Deploy with vLLM (Python API)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,          # Split across 2 GPUs
    gpu_memory_utilization=0.90,     # Use 90% of GPU memory
    max_model_len=8192,
    dtype="auto",                    # Auto-select precision (BF16 if available)
    quantization="awq",             # Use AWQ quantization
    enable_prefix_caching=True       # Cache system prompt KV
)

# Batch inference (very efficient)
prompts = ["Explain MLOps:", "What is Docker?", "Describe CI/CD:"]
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(f"{output.prompt[:30]}... → {output.outputs[0].text[:100]}...")
```

```bash
# Run as OpenAI-compatible server (production)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --max-num-seqs 256
```

```python
# Client code (works with any OpenAI SDK)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention."}
    ],
    temperature=0.7,
    max_tokens=256,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## **7.2 TGI (Text Generation Inference)**

HuggingFace's production-grade inference server.

```bash
# Run with Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --quantize awq \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 4096 \
    --max-concurrent-requests 128
```

**TGI vs vLLM:**

| Feature | vLLM | TGI |
|---------|------|-----|
| **Throughput** | Higher (PagedAttention) | Good (Flash Attention 2) |
| **OpenAI API compatible** | Yes | Yes |
| **Quantization** | AWQ, GPTQ, SqueezeLLM, BitsAndBytes | AWQ, GPTQ, EETQ |
| **Tensor parallelism** | Yes | Yes |
| **Structured output** | Yes (guided decoding, JSON mode) | Yes (grammar-based) |
| **Speculative decoding** | Yes | Yes |
| **Multi-LoRA** | Yes (LoRAX) | Limited |
| **Best for** | Maximum throughput, multi-LoRA | HuggingFace ecosystem, quick setup |

---

## **7.3 Ollama — Local LLM Inference**

Run LLMs locally for development and privacy-sensitive use cases.

```bash
# Install and run
ollama pull llama3.1:8b
ollama run llama3.1:8b "Explain Docker in 3 sentences"

# List available models
ollama list

# Serve as API (starts on localhost:11434)
ollama serve
```

```python
# Use Ollama from Python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.1:8b",
    "prompt": "Explain the difference between Docker and Kubernetes",
    "stream": False,
    "options": {"temperature": 0.7, "num_predict": 256}
})
print(response.json()["response"])

# With the ollama Python library
import ollama
response = ollama.chat(model='llama3.1:8b', messages=[
    {"role": "user", "content": "What is MLOps?"}
])
print(response['message']['content'])
```

**Ollama strengths:** Dead simple setup, automatic quantization, Modelfile for custom configs, great for dev/testing.

---

## **7.4 Quantization for Deployment**

Reduce model size and increase throughput by lowering precision.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION METHODS                          │
│                                                                  │
│  FP32 (32-bit) ─→ FP16 (16-bit) ─→ INT8 (8-bit) ─→ INT4       │
│  100% size        50% size          25% size        12.5% size  │
│  Baseline         ~0% quality loss  ~1% loss        ~2-5% loss  │
│                                                                  │
│  Method        Bits  Approach                 Quality  Speed     │
│  ────────────  ────  ──────────────────────  ───────  ─────     │
│  GPTQ          4     Post-training, layer    Good     Fast      │
│                       by layer calibration                       │
│  AWQ           4     Activation-aware,       Better   Fast      │
│                       protects salient wts                       │
│  GGUF          2-8   CPU-optimized           Good     CPU-fast  │
│                       (llama.cpp format)                         │
│  BitsAndBytes  4/8   Dynamic quantization    Good     Easy      │
│  SmoothQuant   8     Activation smoothing    Best@8b  Moderate  │
│  FP8           8     Native FP8 (H100)      Great    Fastest   │
│                                                                  │
│  Rule of thumb: AWQ > GPTQ for serving quality                  │
│                 GGUF for CPU deployment                           │
│                 BitsAndBytes for quick experiments               │
└─────────────────────────────────────────────────────────────────┘
```

```python
# Load 4-bit quantized model with BitsAndBytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (better than FP4)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True        # Nested quantization (saves more memory)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Original: ~16 GB → Quantized: ~5 GB (fits on single T4 GPU)
print(f"Memory: {model.get_memory_footprint() / 1e9:.1f} GB")
```

---

## **7.5 GPU Selection Guide**

| GPU | VRAM | FP16 TFLOPS | Best For | Approx. Cost (cloud/hr) |
|-----|------|-------------|----------|------------------------|
| **T4** | 16 GB | 65 | Small model inference (≤ 7B quantized) | $0.50–$0.75 |
| **L4** | 24 GB | 121 | Medium inference (7B–13B) | $0.80–$1.00 |
| **A10G** | 24 GB | 125 | Medium inference + fine-tuning | $1.00–$1.50 |
| **A100 40GB** | 40 GB | 312 | Large model training/inference | $3.00–$4.00 |
| **A100 80GB** | 80 GB | 312 | Large model training (34B–70B) | $5.00–$6.00 |
| **H100 SXM** | 80 GB | 989 | Maximum performance, FP8 native | $8.00–$12.00 |

**Quick VRAM sizing rule:**

```
Model VRAM (FP16) ≈ Parameters × 2 bytes
Model VRAM (INT4) ≈ Parameters × 0.5 bytes + overhead

Examples:
  7B  FP16 ≈ 14 GB  → T4 (tight), L4/A10G (comfortable)
  7B  INT4 ≈ 5 GB   → T4 easily
  13B FP16 ≈ 26 GB  → A10G (tight), A100-40GB
  13B INT4 ≈ 8 GB   → T4 or L4
  34B FP16 ≈ 68 GB  → A100-80GB or 2x A100-40GB
  34B INT4 ≈ 20 GB  → L4 or A10G
  70B FP16 ≈ 140 GB → 2x A100-80GB (tensor parallel)
  70B INT4 ≈ 35 GB  → Single A100-40GB
```

---

# **8. Monitoring and Observability**

---

## **8.1 The Three Pillars**

```
┌─────────────────────────────────────────────────────────────────┐
│              THREE PILLARS OF OBSERVABILITY                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   METRICS    │  │    LOGS      │  │   TRACES     │         │
│  │              │  │              │  │              │         │
│  │ "What is     │  │ "What       │  │ "How does a  │         │
│  │  happening?" │  │  happened?" │  │  request flow│         │
│  │              │  │              │  │  through the │         │
│  │ Prometheus   │  │ ELK Stack   │  │  system?"    │         │
│  │ + Grafana    │  │ Loki        │  │              │         │
│  │ Datadog      │  │ CloudWatch  │  │ Jaeger       │         │
│  │              │  │ Datadog     │  │ OpenTelemetry│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  + Fourth pillar for ML: PREDICTION MONITORING                  │
│  ┌──────────────┐                                               │
│  │  ML Metrics  │  Data drift, concept drift, prediction       │
│  │  Evidently   │  distribution, feature distributions          │
│  │  WhyLabs     │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## **8.2 Prometheus + Grafana**

### Instrumenting a FastAPI ML Service

```python
# metrics.py — Custom Prometheus metrics for ML API
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request, Response
import time

# Define metrics
REQUEST_COUNT = Counter(
    "ml_api_requests_total",
    "Total prediction requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "ml_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_api_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

MODEL_LOADED = Gauge(
    "ml_api_model_loaded",
    "Whether the model is loaded (1) or not (0)"
)

GPU_MEMORY_USED = Gauge(
    "ml_api_gpu_memory_used_bytes",
    "GPU memory currently in use"
)

PREDICTION_CLASS_COUNT = Counter(
    "ml_api_prediction_class_total",
    "Count of predictions per class",
    ["predicted_class"]
)

# Middleware for automatic metrics collection
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    
    return response

# Expose metrics endpoint for Prometheus to scrape
async def metrics_endpoint():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "ml-api"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["ml-api:8000"]
    scrape_interval: 5s

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  - job_name: "nvidia-gpu"
    static_configs:
      - targets: ["dcgm-exporter:9400"]
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
  - name: ml-api-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(ml_api_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 1s for 5 minutes"
          description: "The ML API P95 latency is {{ $value }}s"

      - alert: HighErrorRate
        expr: >
          rate(ml_api_requests_total{status_code=~"5.."}[5m]) 
          / rate(ml_api_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5% for 2 minutes"

      - alert: LowPredictionConfidence
        expr: histogram_quantile(0.50, ml_api_prediction_confidence) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Median prediction confidence below 0.5 — possible data drift"

      - alert: GPUMemoryHigh
        expr: ml_api_gpu_memory_used_bytes / 1e9 > 14
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory > 14 GB (near T4 limit)"

      - alert: ModelNotLoaded
        expr: ml_api_model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML model is not loaded!"
```

---

## **8.3 OpenTelemetry**

Vendor-agnostic observability framework — collect traces, metrics, and logs.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanExporter(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Custom spans for ML-specific tracing
@app.post("/predict")
async def predict(request: PredictionRequest):
    with tracer.start_as_current_span("prediction_pipeline") as span:
        # Preprocessing
        with tracer.start_as_current_span("preprocess") as prep_span:
            features = preprocess(request.data)
            prep_span.set_attribute("feature_count", len(features))
        
        # Model inference
        with tracer.start_as_current_span("model_inference") as inf_span:
            result = model.predict(features)
            inf_span.set_attribute("model_version", "v2.1.3")
            inf_span.set_attribute("confidence", float(result.confidence))
            inf_span.set_attribute("predicted_class", result.label)
        
        # Postprocessing
        with tracer.start_as_current_span("postprocess"):
            response = format_response(result)
        
        span.set_attribute("total_features", len(features))
        return response
```

---

## **8.4 Key Metrics for ML Systems**

| Metric | What It Measures | Target | Alert Threshold |
|--------|-----------------|--------|-----------------|
| **Latency P50** | Median response time | < 100ms | > 200ms |
| **Latency P95** | 95th percentile | < 500ms | > 1s |
| **Latency P99** | 99th percentile | < 1s | > 2s |
| **Throughput** | Requests per second | SLA-dependent | < 80% expected |
| **Error Rate** | % of 5xx responses | < 0.1% | > 1% |
| **GPU Utilization** | % of GPU compute used | 60–85% | < 20% or > 95% |
| **GPU Memory** | VRAM usage | < 90% of available | > 95% |
| **Model Accuracy** | Production accuracy vs baseline | Within 2% of training | > 5% drop |
| **Data Drift (PSI)** | Feature distribution shift | PSI < 0.1 | PSI > 0.2 |
| **Prediction Distribution** | Output distribution change | Stable histogram | KL divergence > 0.1 |
| **Cold Start Rate** | % of requests hitting cold start | < 5% | > 20% |

### SLOs and SLAs

```
SLI (Service Level Indicator):
  "The proportion of requests served within 500ms"

SLO (Service Level Objective):
  "99.9% of requests served within 500ms over a 30-day rolling window"

SLA (Service Level Agreement):
  "If we fail to meet 99.5% availability, customers get a credit"

Error Budget:
  0.1% of requests can be slow or fail per month
  = 43.2 minutes of downtime per month
```

---

# **9. Infrastructure as Code**

---

## **9.1 Terraform**

Declarative IaC tool. Provider-agnostic (AWS, GCP, Azure, and 3000+ providers).

```hcl
# main.tf — Deploy ML inference on AWS ECS
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "ml-terraform-state"
    key    = "prod/ml-api.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "ml_api" {
  name                 = "ml-api"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "ml_cluster" {
  name = "ml-inference-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "ml_api" {
  family                   = "ml-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name  = "ml-api"
    image = "${aws_ecr_repository.ml_api.repository_url}:latest"
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "MODEL_VERSION", value = var.model_version },
      { name = "LOG_LEVEL",     value = "INFO" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/ml-api"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

# ECS Service with auto-scaling
resource "aws_ecs_service" "ml_api" {
  name            = "ml-api-service"
  cluster         = aws_ecs_cluster.ml_cluster.id
  task_definition = aws_ecs_task_definition.ml_api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.ml_api.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ml_api.arn
    container_name   = "ml-api"
    container_port   = 8000
  }
}

# Auto-scaling
resource "aws_appautoscaling_target" "ml_api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.ml_cluster.name}/${aws_ecs_service.ml_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "ml-api-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ml_api.resource_id
  scalable_dimension = aws_appautoscaling_target.ml_api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ml_api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# Variables
variable "model_version" {
  description = "ML model version to deploy"
  type        = string
  default     = "v2.1.3"
}

variable "private_subnets" {
  description = "Private subnet IDs for ECS tasks"
  type        = list(string)
}
```

```bash
# Terraform workflow
terraform init        # Initialize providers and backend
terraform plan        # Preview changes (dry run)
terraform apply       # Apply changes (prompts for confirmation)
terraform destroy     # Tear down infrastructure

# Advanced
terraform plan -out=plan.tfplan    # Save plan to file
terraform apply plan.tfplan        # Apply saved plan (no prompt)
terraform state list               # List managed resources
terraform import aws_s3_bucket.existing my-bucket  # Import existing resource
```

---

## **9.2 CloudFormation (AWS-Native)**

```yaml
# cloudformation-ml-api.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: ML API Infrastructure

Parameters:
  ModelVersion:
    Type: String
    Default: v2.1.3
  Environment:
    Type: String
    AllowedValues: [dev, staging, production]

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub ml-inference-cluster-${Environment}

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: ml-api
      Cpu: '2048'
      Memory: '4096'
      NetworkMode: awsvpc
      RequiresCompatibilities: [FARGATE]
      ContainerDefinitions:
        - Name: ml-api
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/ml-api:latest'
          PortMappings:
            - ContainerPort: 8000
          Environment:
            - Name: MODEL_VERSION
              Value: !Ref ModelVersion
```

---

## **9.3 Pulumi (Infrastructure as Real Code)**

```python
# __main__.py — Pulumi with Python
import pulumi
import pulumi_aws as aws

# ECS Cluster
cluster = aws.ecs.Cluster("ml-cluster",
    settings=[{"name": "containerInsights", "value": "enabled"}]
)

# ECR Repository
repo = aws.ecr.Repository("ml-api",
    image_tag_mutability="IMMUTABLE",
    image_scanning_configuration={"scanOnPush": True}
)

# Fargate Task Definition
task_def = aws.ecs.TaskDefinition("ml-api-task",
    family="ml-api",
    cpu="2048",
    memory="4096",
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    container_definitions=pulumi.Output.json_dumps([{
        "name": "ml-api",
        "image": repo.repository_url.apply(lambda url: f"{url}:latest"),
        "portMappings": [{"containerPort": 8000}],
        "environment": [
            {"name": "MODEL_VERSION", "value": "v2.1.3"}
        ]
    }])
)

pulumi.export("cluster_name", cluster.name)
pulumi.export("repo_url", repo.repository_url)
```

---

## **9.4 IaC Comparison**

| Feature | Terraform | CloudFormation | Pulumi |
|---------|-----------|---------------|--------|
| **Language** | HCL (declarative) | YAML/JSON | Python, TypeScript, Go, C# |
| **Multi-cloud** | Yes (3000+ providers) | AWS only | Yes |
| **State** | Remote (S3, Terraform Cloud) | Managed by AWS | Pulumi Cloud or self-managed |
| **Learning Curve** | Moderate (HCL is unique) | Moderate | Low (uses familiar languages) |
| **Best For** | Multi-cloud, team collaboration | AWS-only shops | Developers who prefer code over config |
| **Drift Detection** | `terraform plan` | Drift detection feature | `pulumi preview` |
| **Ecosystem** | Largest (modules, providers) | AWS-native, deep integration | Growing, strong community |

---

# **10. Security Best Practices for ML Systems**

---

## **10.1 Secret Management**

```
┌────────────────────────────────────────────────────────────────┐
│              SECRET MANAGEMENT HIERARCHY                        │
│                                                                 │
│  WORST ─────────────────────────────────────────── BEST        │
│                                                                 │
│  Hardcoded    .env files    Env vars in    Cloud secret    Vault│
│  in code      in git        CI/CD          manager              │
│                             (encrypted)    (AWS SM, GCP SM)     │
│                                                                 │
│  ✗ Never!     ✗ Never!     ✓ OK for       ✓ Good          ✓✓  │
│               (use .env     non-prod                       Best│
│                .gitignore)                                      │
└────────────────────────────────────────────────────────────────┘
```

**Rules:**
1. **Never** commit secrets to git (use `.gitignore`, `git-secrets` pre-commit hook)
2. Use **cloud secret managers** (AWS Secrets Manager, GCP Secret Manager) for production
3. Use **environment variables** for non-production (Railway, Vercel dashboards)
4. Rotate secrets regularly (automated rotation in AWS SM)
5. Use **least-privilege IAM roles** — no wildcard `*` permissions

## **10.2 Container Security**

```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim-bookworm

# Don't run as root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Minimize attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --chown=appuser:appgroup . .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

# Read-only filesystem where possible
# VOLUME ["/tmp"]

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Scan image for vulnerabilities
trivy image myimage:latest

# Use Docker Scout
docker scout cve myimage:latest

# Sign images (cosign)
cosign sign myregistry/ml-api:v2.1.3
```

## **10.3 Network Security**

- **Private subnets:** ML APIs in private subnets, only load balancer is public
- **Security groups:** Only allow port 8000 from the load balancer
- **TLS everywhere:** Encrypt in transit (TLS 1.2+)
- **API authentication:** API keys, OAuth2, or IAM for service-to-service

---

# **11. Cost Optimization Strategies**

---

## **11.1 Compute Cost Optimization**

| Strategy | Savings | Use Case |
|----------|---------|----------|
| **Spot/Preemptible Instances** | 60–90% | Training jobs (checkpoint frequently) |
| **Reserved Instances (1yr)** | 30–40% | Always-on inference endpoints |
| **Reserved Instances (3yr)** | 50–72% | Long-term stable workloads |
| **Savings Plans** | 20–40% | Flexible compute commitment |
| **Right-sizing** | 20–50% | Analyze actual usage, downsize |
| **Scale to zero** | 80–100% idle savings | Dev/staging environments |
| **Quantization** | 50–75% GPU savings | Serve INT4 instead of FP16 |

## **11.2 ML-Specific Cost Tips**

```
Training:
  • Use spot instances for training (save 60-90%)
  • Checkpoint every N steps (recover from spot interruptions)
  • Use mixed precision (FP16/BF16) — 2x throughput, same GPU
  • Profile GPU utilization — if < 50%, use a smaller instance

Inference:
  • Quantize models (INT4 AWQ) — fit larger models on smaller GPUs
  • Batch requests — higher GPU utilization, lower cost per inference
  • Scale to zero for dev/staging (Cloud Run, KEDA)
  • Use CPU for small models (scikit-learn, XGBoost) — no GPU needed
  • Cache frequent predictions in Redis

Storage:
  • Use S3 lifecycle policies (move old models to Glacier)
  • Compress training data (Parquet instead of CSV = 70% smaller)
  • Delete old experiment artifacts (MLflow cleanup)
```

---

# **12. Your Deployment Experience**

---

## **12.1 PathWise — Railway Deployment**

PathWise is a microlearning API with AI-powered career guidance, deployed as a production service.

```
┌─────────────────────────────────────────────────────────────────┐
│                   PATHWISE ARCHITECTURE                           │
│                                                                  │
│  ┌──────────┐     ┌──────────────────────┐    ┌─────────────┐  │
│  │ Frontend │────→│  Railway (Backend)    │───→│  Supabase   │  │
│  │ (Vercel) │     │                      │    │  (Postgres  │  │
│  │ Next.js  │     │  FastAPI + Uvicorn   │    │   + Auth)   │  │
│  └──────────┘     │  ┌────────────────┐  │    └─────────────┘  │
│                    │  │ Career Matcher │  │                     │
│                    │  │ Study Agent    │  │    ┌─────────────┐  │
│                    │  │ Distiller      │  │───→│  Groq / LLM │  │
│                    │  │ Learn Tools    │  │    │    APIs      │  │
│                    │  │ Dashboard      │  │    └─────────────┘  │
│                    │  │ Validators     │  │                     │
│                    │  │ Mastery        │  │    ┌─────────────┐  │
│                    │  └────────────────┘  │───→│   Cohere    │  │
│                    │                      │    │ (Embeddings)│  │
│                    └──────────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**System scale:**
- **13 Python modules**, 10,687 lines of code
- **50+ Pydantic schemas** for request/response validation
- **74 async functions** with proper error handling
- **2,292-line main.py** with 20+ API endpoints

**Key deployment details:**

| Aspect | Implementation |
|--------|---------------|
| **Platform** | Railway (PaaS with auto-scaling, health checks, zero-downtime deploys) |
| **Start Command** | `python -m uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Health Check** | Railway polls `/health` endpoint to verify container readiness |
| **Environment Variables** | API keys (Groq, Supabase, Cohere) stored securely in Railway dashboard |
| **CORS** | Configured to allow frontend origin (`app.add_middleware(CORSMiddleware, ...)`) |
| **Scaling** | Railway auto-provisions resources based on traffic; restarts on failure |
| **Database** | Supabase (managed PostgreSQL) with connection pooling |
| **LLM Integration** | Groq for fast inference (Llama 3.1/Mixtral), async HTTP calls |
| **Embedding** | Cohere for semantic search over learning content |
| **Error Handling** | Graceful degradation — if LLM fails, return cached/default responses |

**Interview talking point:** "I deployed PathWise, a microlearning API with 20+ endpoints, on Railway. The backend is FastAPI with Uvicorn running 74 async functions. I configured health checks, CORS, restart policies, and secure environment variable management. The frontend is on Vercel with automatic preview deployments for PRs. The system integrates Supabase for persistence, Groq for LLM inference, and Cohere for embeddings."

---

## **12.2 AgentOps RCA — Railway Deployment**

Root Cause Analysis system for incident management, also deployed on Railway.

**Railway configuration (`railway.json`):**

```json
{
  "build": { "builder": "NIXPACKS" },
  "deploy": {
    "startCommand": "uvicorn backend_fastapi:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/",
    "healthcheckTimeout": 60,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Key decisions:**
- Used **Nixpacks builder** — auto-detects Python, installs deps from `requirements.txt`
- Health check on root path with **60s timeout** (LLM calls can be slow on cold start)
- **Restart on failure** with 10 retries — handles transient LLM API errors gracefully
- Environment-based configuration — same codebase runs in dev and production

---

## **12.3 Frontend — Vercel Deployment**

| Aspect | Detail |
|--------|--------|
| **Framework** | Next.js (React) |
| **Deployment** | Git-push deploy (push to `main` → auto-deploy to production) |
| **Preview Environments** | Every PR gets a unique URL for testing |
| **Edge Functions** | Serverless functions at CDN edge for low latency |
| **Environment Variables** | API URLs and keys stored in Vercel project settings |
| **Custom Domain** | Vercel handles DNS, TLS certificates automatically |
| **Build Optimization** | Incremental Static Regeneration (ISR) for fast page loads |

---

## **12.4 Docker Containerization**

Experience with Docker across projects:

```dockerfile
# Example from PathWise context
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## **12.5 Architecture Decisions I Can Discuss**

| Decision | Choice | Why |
|----------|--------|-----|
| Railway over AWS ECS | Railway | Right tool for the scale — simpler, faster iteration, team of 1 |
| Supabase over self-hosted PostgreSQL | Supabase | Managed database, built-in auth, real-time subscriptions |
| Groq over OpenAI | Groq | 10x faster inference (custom hardware), lower latency for users |
| FastAPI over Flask | FastAPI | Async support, auto-docs, Pydantic validation, type safety |
| Vercel over Netlify | Vercel | Best Next.js support (same company), preview deployments |
| Monolith over microservices | Monolith | One team, one codebase — microservices would add complexity without benefit |

---

# **13. Common Interview Questions with Strong Answers**

---

## **Q1: "How would you deploy an ML model to production?"**

> "It depends on the use case. For **real-time inference**, I'd containerize the model in a FastAPI app with Docker, write a health check endpoint, and deploy to a managed platform — Railway or Cloud Run for small-to-medium scale, or Kubernetes (EKS/GKE) for enterprise. The model gets loaded once at startup via a lifespan handler, and I'd use Uvicorn with multiple workers for concurrency.
>
> For **batch inference**, I'd use a scheduled job (Airflow, cron, or SageMaker Processing) that reads from S3/GCS, runs predictions in batches with a DataLoader, and writes results back.
>
> For **LLM serving**, I'd use vLLM or TGI behind an OpenAI-compatible API, with AWQ quantization to fit on a single GPU. vLLM's PagedAttention gives 2–4x throughput over naive generation.
>
> In all cases, I'd add: (1) CI/CD with GitHub Actions for automated testing and deployment, (2) Prometheus metrics for latency/throughput/error monitoring, (3) Evidently or similar for data drift detection, and (4) a model registry (MLflow) for version management with staging gates."

---

## **Q2: "Docker vs Kubernetes — when do you use each?"**

> "Docker is for **packaging** — it ensures your application runs identically everywhere by bundling code, dependencies, and runtime into a container. Kubernetes is for **orchestrating** — it manages many Docker containers at scale with auto-scaling, self-healing, and traffic routing.
>
> For a single ML API with moderate traffic, Docker alone (deployed on Railway, Cloud Run, or ECS Fargate) is sufficient and much simpler. You don't need Kubernetes.
>
> I'd reach for Kubernetes when: (a) I have multiple microservices that need to communicate, (b) I need fine-grained auto-scaling based on custom metrics like GPU utilization or inference queue depth, (c) I need GPU scheduling across a pool of nodes with tolerations and node selectors, or (d) the organization requires strong RBAC, network policies, and multi-tenancy.
>
> In my experience, I deployed FastAPI apps on Railway with Docker — simple, effective, and the right tool for the scale. For a large organization managing dozens of ML models, Kubernetes with Helm charts, HPA, and GitOps via ArgoCD would be justified."

---

## **Q3: "What is MLOps and why does it matter?"**

> "MLOps applies DevOps principles to machine learning. The core insight is that ML systems have two types of technical debt that regular software doesn't: **data dependencies** and **model decay**.
>
> A model that's 95% accurate at training time can degrade to 80% in production because input data distributions shift (data drift) or the relationship between features and target changes (concept drift). This doesn't happen with regular software.
>
> MLOps addresses this with: (1) **Automated pipelines** — data → train → evaluate → deploy as a repeatable pipeline, not Jupyter notebooks. (2) **Experiment tracking** — MLflow or W&B to track every hyperparameter, metric, and artifact. (3) **Model registry** — versioned models with staging gates before production. (4) **Feature stores** — consistent features between training and serving. (5) **Monitoring** — track prediction distributions, feature drift (PSI), and model performance over time. (6) **Automated retraining** — trigger retraining when drift exceeds thresholds.
>
> The maturity levels range from Level 0 (manual everything) to Level 3 (fully automated retraining and deployment). Most teams should aim for Level 2 — automated CI/CD for ML with monitoring."

---

## **Q4: "How do you monitor ML models in production?"**

> "I monitor at three levels:
>
> **Infrastructure level:** Prometheus + Grafana for latency (P50/P95/P99), throughput (RPS), error rates, GPU utilization, and memory usage. I set alerts for P95 latency > 1s and error rate > 5%.
>
> **Model level:** I track prediction distributions over time. If the distribution of model outputs shifts significantly (more high-confidence predictions, or predictions clustering around a single class), that's an early warning sign. I use Evidently AI to compute PSI (Population Stability Index) — if PSI > 0.2, I trigger investigation.
>
> **Data level:** I compare production feature distributions against training distributions using statistical tests (KS test, Jensen-Shannon divergence). This catches data drift before it impacts model quality.
>
> When drift is detected: (1) Alert the team, (2) analyze which features drifted and by how much, (3) retrain with recent data if appropriate, (4) deploy new model through the registry with canary traffic splitting, (5) monitor the new model's performance before full rollout."

---

## **Q5: "Blue-green vs canary deployment — when do you use each?"**

> "**Blue-green** maintains two identical environments. You deploy the new version to the idle environment, test it, then switch all traffic at once. It's great when you need instant rollback — just switch back to the original environment. The downside is 2x infrastructure cost during deployment.
>
> **Canary** gradually routes a small percentage of traffic (5–10%) to the new version while monitoring metrics. If everything looks good, you increase to 25%, 50%, 100%. It's better for ML models because you can compare real-world performance between model versions on actual traffic.
>
> I'd use **blue-green** for non-ML services where correctness is binary (it works or it doesn't). I'd use **canary** for ML model deployments where I want to compare accuracy, latency, and business metrics between the old and new model on real traffic before fully committing.
>
> There's also **shadow deployment** (dark launch) — mirror all traffic to the new model without returning its predictions to users. This is the safest approach for high-stakes models like fraud detection, where you compare outputs offline before switching."

---

## **Q6: "How would you serve an LLM in production?"**

> "For a self-hosted LLM, I'd use **vLLM** as the inference server. vLLM's PagedAttention manages KV cache like virtual memory pages — this eliminates memory fragmentation and gives 2–4x higher throughput than naive HuggingFace `model.generate()`. I'd enable continuous batching so new requests can join an in-progress batch, and prefix caching to share KV cache for common system prompts.
>
> For deployment: (1) **Quantize** the model — AWQ for 4-bit gives the best quality-to-size ratio. A 7B model drops from 14GB to ~5GB, fitting comfortably on a T4. (2) **Containerize** with the NVIDIA CUDA base image. (3) **Deploy** on a GPU instance (g4dn.xlarge on AWS for T4, or a GPU node pool in K8s). (4) **Serve** behind an OpenAI-compatible API endpoint for easy client integration. (5) **Monitor** token throughput, time-to-first-token, and GPU memory.
>
> For cost optimization at scale: use Spot instances for non-critical workloads, or AWS Inferentia chips for inference-only workloads. For multi-LoRA serving (personalized models), vLLM supports loading multiple LoRA adapters on a single base model.
>
> If self-hosting isn't required, I'd use a managed service: AWS Bedrock, Azure OpenAI, or Vertex AI — they handle scaling, quantization, and infrastructure automatically."

---

## **Q7: "How would you set up CI/CD for an ML project?"**

> "I use GitHub Actions with four stages:
>
> (1) **Lint** — ruff for Python linting and formatting checks, mypy for type checking. Fast feedback — fails in 30 seconds.
>
> (2) **Test** — pytest with coverage. I test API endpoints (using httpx/TestClient), data preprocessing logic, and model loading. For ML-specific tests, I include a small fixture dataset and verify predictions are within expected ranges. I enforce a coverage threshold (80%+).
>
> (3) **Build** — Docker image built with multi-stage builds to minimize size, pushed to a container registry (GHCR or ECR). I use GitHub Actions cache for Docker layers. I also run Trivy vulnerability scanning on the image.
>
> (4) **Deploy** — triggered only on main branch merges, with a manual approval gate via GitHub Environments. For Railway, I hit the redeploy API. For Kubernetes, I update the Helm values with the new image tag. I include a post-deploy health check that polls the `/health` endpoint for up to 5 minutes.
>
> For ML-specific additions: I'd add a **model validation step** that loads the new model, runs it against a holdout test set, and fails the pipeline if accuracy drops below a threshold. This prevents deploying a broken model."

---

## **Q8: "What happens when your ML model's performance degrades in production?"**

> "This is a systematic investigation process:
>
> **Step 1 — Detect:** Monitoring alerts fire — either from metrics (latency spike, error rate increase) or from drift detection (PSI > 0.2, accuracy below baseline). Or a business stakeholder reports unexpected results.
>
> **Step 2 — Diagnose:** I check: (a) Was there a recent code deployment? (check CI/CD logs), (b) Did input data change? (feature distribution analysis with Evidently), (c) Did upstream data pipelines break? (check data freshness and quality), (d) Did the world change? (concept drift — e.g., seasonal patterns, market shifts, new fraud techniques).
>
> **Step 3 — Mitigate (short-term):** If a deployment caused it, rollback immediately. If data quality is the issue, add input validation and reject out-of-distribution inputs. If it's drift — switch to the most recent validated model from the registry.
>
> **Step 4 — Fix (long-term):** Retrain with recent data, update feature engineering if needed, adjust monitoring thresholds. Deploy the new model through the registry with canary traffic splitting.
>
> **Step 5 — Prevent:** Add automated retraining triggers for drift thresholds. Improve data validation at ingestion time. Add integration tests for upstream data dependencies. Document the incident for the team."

---

## **Q9: "Explain the difference between data drift, concept drift, and prediction drift."**

> "These are three distinct types of drift in production ML systems:
>
> **Data drift** (also called covariate shift): The input feature distributions change. P(X) shifts. Example: your fraud model was trained on US transactions, but production starts getting more EU transactions with different spending patterns. Features like `avg_amount` and `time_of_day` now have different distributions.
>
> **Concept drift**: The relationship between features and target changes. P(Y|X) shifts. Example: what constitutes fraud evolves over time — new attack vectors emerge that the model wasn't trained on. The same feature values now map to different labels.
>
> **Prediction drift**: The model's output distribution changes. P(Y_hat) shifts. This can happen due to data drift (different inputs → different predictions) or due to model degradation.
>
> **Detection methods:**
> - Data drift: PSI, KS test, Jensen-Shannon divergence on feature distributions
> - Concept drift: Monitor accuracy/F1 against ground truth (with a delay for labeling)
> - Prediction drift: Monitor prediction histograms over time
>
> **Data drift is the earliest signal** — you can detect it before accuracy drops. Concept drift is the hardest to detect because you need ground truth labels, which are often delayed."

---

## **Q10: "How do you handle secrets and API keys in deployment?"**

> "I follow a strict hierarchy: never in code, never in git, always encrypted at rest.
>
> For development: I use `.env` files that are in `.gitignore`. The code loads them with `python-dotenv` or `pydantic-settings`.
>
> For CI/CD: GitHub Actions secrets — encrypted, never logged, available as `${{ secrets.KEY }}`.
>
> For production: cloud secret managers — AWS Secrets Manager or GCP Secret Manager. These support automatic rotation, audit logging, and fine-grained IAM access control. In Kubernetes, I use the External Secrets Operator to sync secrets from the cloud manager into K8s Secrets.
>
> For platforms like Railway and Vercel: environment variables configured in the dashboard — encrypted at rest, injected at runtime, never visible in logs.
>
> I also use pre-commit hooks like `git-secrets` to prevent accidentally committing API keys."

---

## **Q11: "Walk me through your deployment architecture for a recent project."**

> "For PathWise, I designed a microlearning platform with AI-powered career guidance. The architecture has three main components:
>
> **Backend (Railway):** A FastAPI application with 13 Python modules and 20+ endpoints. It handles PDF processing, flashcard generation, career matching, and interview simulation — all powered by Groq's LLM API for fast inference and Cohere for embeddings. The start command is `uvicorn main:app --host 0.0.0.0 --port $PORT` via a Procfile. Railway provides automatic health checks, restart policies, and zero-downtime deploys.
>
> **Database (Supabase):** Managed PostgreSQL for storing user data, lessons, flashcards, quiz results, and progress tracking. Supabase also handles authentication and provides a REST API and real-time subscriptions.
>
> **Frontend (Vercel):** Next.js application with automatic deployment on git push. Every PR gets a preview URL for testing. Environment variables store the backend API URL.
>
> Key operational decisions: I chose Railway over AWS ECS because it was the right tool for a solo developer — simpler, faster iteration cycle. I chose Groq over OpenAI for 10x faster LLM inference. I implemented graceful degradation — if the LLM API is down, the system falls back to cached responses."

---

## **Q12: "How would you reduce the cost of ML inference in production?"**

> "Cost optimization for ML inference happens at multiple levels:
>
> **Model level:** Quantize from FP16 to INT4 (AWQ) — this cuts GPU memory by 75% and lets you use cheaper GPUs. A 7B model goes from needing an A10G ($1.50/hr) to fitting on a T4 ($0.50/hr). Also consider distilling a large model into a smaller one for your specific task.
>
> **Infrastructure level:** Use auto-scaling (HPA or KEDA) to scale with demand. Scale to zero during off-peak hours with Cloud Run or KEDA. Use spot instances for non-critical workloads (60–90% savings). Right-size instances — if GPU utilization is consistently under 50%, you're overpaying.
>
> **Request level:** Enable dynamic batching in vLLM/Triton — batch multiple requests into a single GPU forward pass. Cache frequent predictions in Redis. Use request deduplication for identical inputs.
>
> **Architecture level:** For low-latency, high-volume predictions on small models (XGBoost, small NNs), skip the GPU entirely and serve on CPU — it's 10x cheaper. Only use GPUs for models that actually need them (large neural nets, LLMs).
>
> **Practical example:** Moving a 7B LLM from FP16 on A100 ($4/hr) to AWQ INT4 on T4 ($0.50/hr) is an 8x cost reduction with minimal quality loss."

---

# **14. Quick Reference Cards**

---

## **14.1 Docker Commands Cheat Sheet**

```bash
# Build & Run
docker build -t myimage:v1 .
docker run -d -p 8000:8000 --name myapp myimage:v1
docker run --gpus all -p 8000:8000 myimage:gpu

# Manage
docker ps                        # Running containers
docker ps -a                     # All containers (including stopped)
docker logs -f myapp             # Follow logs
docker exec -it myapp bash       # Shell into container
docker stop myapp && docker rm myapp

# Images
docker images                    # List images
docker image prune -a            # Remove unused images
docker system df                 # Disk usage

# Registry
docker tag myimage:v1 registry.com/myimage:v1
docker push registry.com/myimage:v1

# Compose
docker compose up -d
docker compose down -v
docker compose logs -f service_name
```

## **14.2 Kubernetes Commands Cheat Sheet**

```bash
# Cluster Info
kubectl cluster-info
kubectl get nodes

# Deployments
kubectl apply -f deployment.yaml
kubectl get deployments -n production
kubectl rollout status deployment/ml-api
kubectl rollout undo deployment/ml-api    # Rollback

# Pods
kubectl get pods -n production
kubectl describe pod <pod-name>
kubectl logs <pod-name> -f
kubectl exec -it <pod-name> -- bash

# Services & Ingress
kubectl get svc,ingress -n production

# Scaling
kubectl scale deployment ml-api --replicas=5
kubectl autoscale deployment ml-api --min=2 --max=10 --cpu-percent=70

# Debugging
kubectl top pods                 # CPU/memory per pod
kubectl get events --sort-by=.metadata.creationTimestamp
```

## **14.3 Cloud CLI Cheat Sheet**

```bash
# AWS
aws s3 cp model.pt s3://bucket/model.pt
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
aws ecs update-service --cluster my-cluster --service my-service --force-new-deployment

# GCP
gcloud run deploy my-service --source . --region us-central1
gcloud builds submit --tag gcr.io/PROJECT/IMAGE
gcloud container clusters get-credentials CLUSTER --region REGION

# Azure
az acr login --name myregistry
az ml online-deployment create --file deployment.yaml
az aks get-credentials --resource-group mygroup --name mycluster
```

---

# **15. Key Takeaways**

---

```
┌─────────────────────────────────────────────────────────────────────┐
│                       KEY TAKEAWAYS                                  │
│                                                                     │
│  CLOUD                                                              │
│  ✓ AWS = broadest services; GCP = best data/ML/K8s; Azure = MSFT  │
│  ✓ Know EC2 instance families: g4dn (inference), p3/p4d (training) │
│  ✓ S3 for artifacts, Lambda for lightweight inference               │
│  ✓ Choose cloud based on team, ecosystem, and workload, not hype   │
│  ✓ Most companies use 2+ clouds — K8s is the abstraction layer     │
│                                                                     │
│  DOCKER                                                             │
│  ✓ Multi-stage builds cut image size 60-80%                        │
│  ✓ Layer caching: COPY requirements.txt before code                │
│  ✓ Always run as non-root user in production                       │
│  ✓ One process per container, health checks on every container     │
│  ✓ Scan images for vulnerabilities (Trivy, Docker Scout)           │
│                                                                     │
│  KUBERNETES                                                         │
│  ✓ Only use K8s when complexity is justified (multi-service, GPU   │
│    scheduling, enterprise compliance)                              │
│  ✓ Know: Pods, Deployments, Services, HPA, ConfigMaps, Secrets    │
│  ✓ Railway/Cloud Run for most ML APIs; K8s for complex systems     │
│  ✓ Three probe types: startup (model loading), liveness, readiness │
│  ✓ GitOps with ArgoCD for production K8s deployments               │
│                                                                     │
│  CI/CD                                                              │
│  ✓ Pipeline: lint → test → model-validate → build → scan → deploy │
│  ✓ Canary deployment for ML models (compare real-world metrics)    │
│  ✓ Always include health checks and post-deploy verification       │
│  ✓ GitHub Actions concurrency groups to cancel stale runs          │
│                                                                     │
│  MLOps                                                              │
│  ✓ ML lifecycle is a continuous loop, not a one-time deployment    │
│  ✓ Experiment tracking (MLflow/W&B) is non-negotiable              │
│  ✓ Monitor three levels: infrastructure, model, data               │
│  ✓ Data drift (PSI > 0.2) → retrain; concept drift → investigate  │
│  ✓ Feature stores prevent training-serving skew                    │
│  ✓ DVC for data versioning, Prefect/Airflow for orchestration      │
│                                                                     │
│  LLM DEPLOYMENT                                                     │
│  ✓ vLLM for self-hosted (PagedAttention = 2-4x throughput)        │
│  ✓ AWQ quantization: best quality-to-size ratio for serving       │
│  ✓ GPU sizing: 7B INT4→T4; 13B INT4→L4; 70B INT4→A100-40GB      │
│  ✓ Managed services (Bedrock, Azure OpenAI) when infra isn't core │
│  ✓ Prefix caching for common system prompts                       │
│                                                                     │
│  MONITORING                                                         │
│  ✓ Prometheus + Grafana for infra metrics                          │
│  ✓ Evidently AI for data/model drift detection                     │
│  ✓ OpenTelemetry for distributed tracing                           │
│  ✓ Key metrics: P50/P95/P99 latency, error rate, GPU utilization  │
│  ✓ Set SLOs (e.g., 99.9% of requests < 500ms)                    │
│                                                                     │
│  SECURITY & COST                                                    │
│  ✓ Never commit secrets; use cloud secret managers                 │
│  ✓ Quantize models for 4-8x GPU cost reduction                    │
│  ✓ Spot instances for training (60-90% savings)                    │
│  ✓ Scale to zero for dev/staging environments                      │
│                                                                     │
│  YOUR EXPERIENCE                                                    │
│  ✓ PathWise: FastAPI + Uvicorn on Railway with health checks,       │
│    13 modules, 20+ endpoints, Groq LLM, Supabase, Cohere          │
│  ✓ Vercel for frontend with preview deployments per PR             │
│  ✓ Docker containerization with proper layer caching               │
│  ✓ Chose Railway over K8s — right tool for the scale               │
│  ✓ Know when to use simple PaaS vs complex orchestration           │
└─────────────────────────────────────────────────────────────────────┘
```

---

**Final advice:** In interviews, always anchor your answer in the *right tool for the right scale*. Don't say "I'd use Kubernetes" for a single-model API, and don't say "I'd use Railway" for a 50-model enterprise deployment. Show that you understand the spectrum from simple PaaS to full Kubernetes orchestration, and that you can choose wisely based on team size, traffic patterns, compliance needs, and cost constraints.

The three most important principles:
1. **Start simple, scale when needed** — Railway/Cloud Run → ECS/Cloud Run → Kubernetes
2. **Automate everything** — CI/CD, testing, deployment, monitoring, retraining
3. **Monitor relentlessly** — Infrastructure (Prometheus), Model (Evidently), Data (PSI) — because ML models degrade silently
