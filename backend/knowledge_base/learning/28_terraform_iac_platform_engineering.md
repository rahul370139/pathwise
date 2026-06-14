# Terraform, IaC & Platform Engineering — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** ML/AI Engineer — Cloud Infrastructure, DevOps for ML, Platform Engineering

> **Context:** Your file `23_cloud_mlops_deployment.md` covers basic Terraform HCL and an IaC comparison table. This guide goes deeper into Terraform concepts, modules, state management, and the broader IaC/PaC/SaC paradigm that interviewers ask about.

---

# Table of Contents

1. [Infrastructure as Code — Philosophy](#1-infrastructure-as-code--philosophy)
2. [Terraform Deep Dive](#2-terraform-deep-dive)
3. [Terraform State Management](#3-terraform-state-management)
4. [Terraform Modules and Reusability](#4-terraform-modules-and-reusability)
5. [Terraform Workspaces and Environments](#5-terraform-workspaces-and-environments)
6. [Platform as Code (PaC)](#6-platform-as-code-pac)
7. [Service as Code (SaC)](#7-service-as-code-sac)
8. [Configuration as Code (CaC)](#8-configuration-as-code-cac)
9. [GitOps and IaC Workflows](#9-gitops-and-iac-workflows)
10. [IaC for ML Infrastructure](#10-iac-for-ml-infrastructure)
11. [Interview Questions with Strong Answers](#11-interview-questions-with-strong-answers)

---

# **1. Infrastructure as Code — Philosophy**

## **1.1 What Is IaC?**

IaC means **managing and provisioning infrastructure through machine-readable definition files** rather than manual processes (clicking in consoles, running ad-hoc commands).

```
TRADITIONAL (Manual)                    IaC (Automated)
────────────────────                    ───────────────
1. Open AWS Console                     1. Write main.tf
2. Click "Create EC2"                   2. terraform plan
3. Fill in form fields                  3. terraform apply
4. Click "Launch"                       4. Git commit
5. Repeat for each resource             5. Repeatable forever
6. Hope you remember settings           6. Version-controlled
7. Document in wiki (outdated)          7. Self-documenting
```

## **1.2 Core Principles**

| Principle | Meaning | Why It Matters |
|-----------|---------|----------------|
| **Declarative** | Describe *what* you want, not *how* to get there | Engine figures out the steps; idempotent |
| **Idempotent** | Running the same code twice produces the same result | Safe to re-apply; no drift |
| **Version-controlled** | Infrastructure code lives in Git | Audit trail, rollbacks, code review |
| **Immutable** | Replace resources instead of modifying in-place | No configuration drift, consistent environments |
| **Self-documenting** | The code IS the documentation | No stale wikis; source of truth |

## **1.3 The "As Code" Spectrum**

```
┌────────────────────────────────────────────────────────────────┐
│              THE "AS CODE" SPECTRUM                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Infrastructure as Code (IaC)                                  │
│  └── VMs, networks, storage, databases, load balancers         │
│      Tools: Terraform, CloudFormation, Pulumi, CDK             │
│                                                                 │
│  Configuration as Code (CaC)                                   │
│  └── OS configs, packages, app settings, feature flags         │
│      Tools: Ansible, Chef, Puppet, Salt                        │
│                                                                 │
│  Platform as Code (PaC)                                        │
│  └── Developer self-service platforms, golden paths            │
│      Tools: Backstage + Terraform, Crossplane, Humanitec      │
│                                                                 │
│  Service as Code (SaC)                                         │
│  └── Full service definitions (infra + config + monitoring     │
│      + alerts + runbooks) as a single deployable unit          │
│      Tools: Kubernetes + Helm, Score, Waypoint                 │
│                                                                 │
│  Policy as Code                                                │
│  └── Security rules, compliance checks, governance             │
│      Tools: OPA/Rego, Sentinel, Checkov, tfsec                │
│                                                                 │
│  Pipeline as Code                                              │
│  └── CI/CD pipeline definitions                                │
│      Tools: GitHub Actions YAML, Jenkinsfile, .gitlab-ci.yml  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

# **2. Terraform Deep Dive**

## **2.1 Core Concepts**

| Concept | What It Is | Example |
|---------|-----------|---------|
| **Provider** | Plugin that talks to a cloud/service API | `hashicorp/aws`, `hashicorp/google`, `hashicorp/azurerm` |
| **Resource** | A single infrastructure object | `aws_instance`, `aws_s3_bucket`, `google_compute_instance` |
| **Data Source** | Read-only reference to existing infrastructure | `data "aws_ami" "latest"` — look up latest AMI |
| **Variable** | Input parameter | `variable "region" { default = "us-east-1" }` |
| **Output** | Exported value (like a return value) | `output "api_url" { value = aws_lb.main.dns_name }` |
| **Module** | Reusable package of resources | `module "vpc" { source = "./modules/vpc" }` |
| **State** | JSON file tracking what Terraform manages | `terraform.tfstate` — maps config to real resources |
| **Backend** | Where state is stored | S3, Terraform Cloud, GCS, Azure Blob |

## **2.2 Terraform Lifecycle**

```
┌──────────────────────────────────────────────────────────────┐
│                 TERRAFORM LIFECYCLE                            │
│                                                               │
│  terraform init                                               │
│  └── Downloads providers, initializes backend, installs      │
│      modules. Run once per directory (or after changes).     │
│             │                                                 │
│             ▼                                                 │
│  terraform plan                                               │
│  └── Compares desired state (code) with actual state         │
│      (state file). Shows what will change.                   │
│      Creates: +   Modifies: ~   Destroys: -                 │
│             │                                                 │
│             ▼                                                 │
│  terraform apply                                              │
│  └── Executes the plan. Creates/modifies/destroys            │
│      resources. Updates state file.                          │
│             │                                                 │
│             ▼                                                 │
│  terraform destroy                                            │
│  └── Tears down all managed resources.                       │
│      Useful for dev/test environments.                       │
│                                                               │
│  OTHER COMMANDS:                                              │
│  terraform validate  — Check syntax without calling APIs     │
│  terraform fmt       — Auto-format HCL files                 │
│  terraform output    — Show output values                    │
│  terraform import    — Bring existing resource under mgmt    │
│  terraform state     — Advanced state manipulation           │
│  terraform taint     — Mark resource for recreation          │
│  terraform graph     — Generate dependency graph (DOT)       │
└──────────────────────────────────────────────────────────────┘
```

## **2.3 HCL (HashiCorp Configuration Language) Essentials**

### **Resources**

```hcl
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "rahul-ml-artifacts-${var.environment}"
  tags = {
    Project     = "ml-pipeline"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

### **Variables and Locals**

```hcl
variable "environment" {
  description = "Deployment environment"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  type    = string
  default = "t3.medium"
}

locals {
  common_tags = {
    Project   = "ml-pipeline"
    Env       = var.environment
    ManagedBy = "terraform"
  }
  name_prefix = "ml-${var.environment}"
}
```

### **Data Sources**

```hcl
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning AMI (Ubuntu 22.04)*"]
  }
}

resource "aws_instance" "training" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "g4dn.xlarge"
}
```

### **Outputs**

```hcl
output "api_endpoint" {
  description = "URL of the ML API"
  value       = "https://${aws_lb.main.dns_name}/predict"
  sensitive   = false
}
```

## **2.4 Key HCL Features for Interviews**

| Feature | Syntax | Use Case |
|---------|--------|----------|
| **count** | `count = var.create ? 1 : 0` | Conditionally create resources |
| **for_each** | `for_each = toset(var.environments)` | Create multiple instances from a collection |
| **dynamic blocks** | `dynamic "ingress" { ... }` | Generate repeated nested blocks |
| **depends_on** | `depends_on = [aws_iam_role.this]` | Explicit dependency (when implicit isn't enough) |
| **lifecycle** | `lifecycle { prevent_destroy = true }` | Prevent accidental deletion |
| **provisioners** | `provisioner "local-exec" { ... }` | Run scripts after resource creation (use sparingly) |

---

# **3. Terraform State Management**

## **3.1 What Is State?**

State is a **JSON file** (`terraform.tfstate`) that maps your Terraform configuration to real-world resources. It's the source of truth for what Terraform manages.

```
┌────────────────────────────────────────────────────┐
│              TERRAFORM STATE                         │
│                                                      │
│  Your Code (main.tf)     State File        Cloud    │
│  ─────────────────       ──────────       ─────     │
│  resource "aws_s3"  ←→  "aws_s3":       S3 Bucket  │
│    bucket = "foo"        id = "foo"      (real)     │
│                          arn = "..."                 │
│                          region = "..."              │
│                                                      │
│  terraform plan compares code vs state              │
│  terraform apply updates cloud to match code        │
│  terraform refresh updates state to match cloud     │
└────────────────────────────────────────────────────┘
```

## **3.2 Remote State (Production Must-Have)**

**Never** store state locally in production. Use a remote backend:

```hcl
terraform {
  backend "s3" {
    bucket         = "company-terraform-state"
    key            = "ml-platform/prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"  # state locking
    encrypt        = true
  }
}
```

| Backend | Best For | State Locking | Encryption |
|---------|----------|---------------|------------|
| **S3 + DynamoDB** | AWS-centric teams | DynamoDB table | SSE-S3/KMS |
| **GCS** | GCP-centric teams | Built-in | Built-in |
| **Azure Blob** | Azure-centric teams | Built-in lease | Built-in |
| **Terraform Cloud** | Teams wanting managed state | Built-in | Built-in |

## **3.3 State Locking**

Prevents two people from running `terraform apply` simultaneously (which would corrupt state):

```
Developer A: terraform apply  → Acquires lock → Modifies state → Releases lock
Developer B: terraform apply  → Blocked (lock held) → Waits → Proceeds after A
```

**Without locking:** Two concurrent applies could create duplicate resources or corrupt state.

## **3.4 State Operations**

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `terraform state list` | Show all managed resources | Audit what Terraform controls |
| `terraform state show <resource>` | Show details of one resource | Debug a specific resource |
| `terraform state mv` | Rename/move resource in state | Refactoring without destroying |
| `terraform state rm` | Remove from state (don't destroy) | "Un-manage" a resource |
| `terraform import` | Bring existing resource into state | Adopt pre-existing infrastructure |

---

# **4. Terraform Modules and Reusability**

## **4.1 What Are Modules?**

Modules are **reusable packages of Terraform resources** — like functions in programming.

```
modules/
├── vpc/
│   ├── main.tf        # VPC, subnets, route tables
│   ├── variables.tf   # region, cidr_block, etc.
│   └── outputs.tf     # vpc_id, subnet_ids
├── ecs-service/
│   ├── main.tf        # ECS task, service, ALB
│   ├── variables.tf   # image, cpu, memory, etc.
│   └── outputs.tf     # service_url
└── ml-endpoint/
    ├── main.tf        # SageMaker endpoint
    ├── variables.tf   # model_name, instance_type
    └── outputs.tf     # endpoint_url
```

### Using a Module

```hcl
module "ml_api" {
  source = "./modules/ecs-service"

  service_name   = "fraud-detection-api"
  image          = "123456.dkr.ecr.us-east-1.amazonaws.com/fraud-api:v2"
  cpu            = 2048
  memory         = 4096
  desired_count  = 2
  environment    = var.environment
}

output "fraud_api_url" {
  value = module.ml_api.service_url
}
```

## **4.2 Module Best Practices**

| Practice | Rationale |
|----------|-----------|
| One module per logical component | VPC, ECS service, database — each is a module |
| Use variables for everything configurable | Don't hardcode values |
| Always define outputs | Consumers need to reference module resources |
| Pin module versions | `source = "git::https://...?ref=v1.2.0"` |
| Use Terraform Registry modules when available | Battle-tested, community-maintained |

---

# **5. Terraform Workspaces and Environments**

## **5.1 Workspaces**

Workspaces let you manage **multiple environments** (dev, staging, prod) with the same code but separate state files.

```bash
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

terraform workspace select prod
terraform apply  # applies to prod state
```

```hcl
resource "aws_instance" "api" {
  instance_type = terraform.workspace == "prod" ? "c5.2xlarge" : "t3.medium"
  tags = {
    Environment = terraform.workspace
  }
}
```

## **5.2 Workspaces vs Directory Structure**

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **Workspaces** | Same code, different state per workspace | Simple, DRY | Limited env differences, same backend |
| **Directory per env** | `envs/dev/`, `envs/prod/` with `.tfvars` | Full flexibility per env | Code duplication risk |
| **Terragrunt** | Wrapper tool for DRY multi-env | Best of both, DRY + flexible | Additional tool to learn |

**Recommendation for interviews:** "For small projects, workspaces work fine. For larger teams, I prefer a directory-per-environment structure with shared modules, or Terragrunt for DRY management."

---

# **6. Platform as Code (PaC)**

## **6.1 What Is PaC?**

PaC extends IaC to provide **developer self-service platforms** defined in code. Instead of developers needing to understand Terraform/Kubernetes, they interact with a higher-level abstraction — a "platform" that provisions standardized environments.

```
┌────────────────────────────────────────────────────────────────┐
│             IaC vs PaC                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IaC (Infrastructure as Code):                                 │
│  Developer writes: "I need an EC2 instance with 4 vCPUs,      │
│  8GB RAM, in subnet X, with security group Y, IAM role Z..."  │
│  → Deep cloud knowledge required                               │
│                                                                 │
│  PaC (Platform as Code):                                       │
│  Developer writes: "I need a compute environment for ML        │
│  training, size: medium, GPU: yes"                             │
│  → Platform translates to correct cloud resources              │
│  → Guardrails, security, compliance baked in                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## **6.2 Key PaC Tools**

| Tool | What It Does | Key Concept |
|------|-------------|-------------|
| **Backstage** (Spotify) | Internal developer portal; service catalog + templates | Software templates define "golden paths" |
| **Crossplane** | Kubernetes-based infrastructure management | Compositions let you define platform-level abstractions |
| **Humanitec** | Platform orchestrator; dynamic resource management | Score workload spec → Humanitec provisions everything |
| **Port** | Internal developer portal with self-service actions | Developers click buttons, platform does the rest |

## **6.3 Crossplane Example (PaC)**

```yaml
# Platform team defines a "CompositeMLEnvironment"
apiVersion: platform.example.com/v1
kind: MLEnvironment
metadata:
  name: fraud-model-training
spec:
  size: large          # Abstraction — dev doesn't need to know instance types
  gpu: true
  framework: pytorch
  storage: 100Gi
  team: risk-analytics
```

The platform team's Crossplane Composition translates this into:
- An AWS `g4dn.2xlarge` instance
- An EBS volume with 100GB
- A VPC with the team's subnet
- IAM roles scoped to the team
- CloudWatch monitoring and alerts

**Developers don't write Terraform. They write a simple spec; the platform handles the rest.**

## **6.4 Why PaC Matters**

- **Reduces cognitive load:** Data scientists shouldn't need to know about VPC CIDRs
- **Enforces standards:** Security, tagging, cost controls baked into templates
- **Speeds delivery:** New environments in minutes, not weeks
- **Governance at scale:** Platform team controls what's possible

---

# **7. Service as Code (SaC)**

## **7.1 What Is SaC?**

SaC defines an **entire service** — not just its infrastructure, but its config, monitoring, alerting, scaling rules, and runbooks — as a single codified unit.

```
┌────────────────────────────────────────────────────────────────┐
│              SERVICE AS CODE — FULL STACK                        │
│                                                                 │
│  service.yaml (or equivalent):                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │  name: fraud-detection-api               │                  │
│  │  team: risk-analytics                    │                  │
│  │                                          │                  │
│  │  compute:                                │  ← Infrastructure│
│  │    type: container                       │                  │
│  │    image: fraud-api:v2.1                 │                  │
│  │    replicas: 3                           │                  │
│  │    cpu: 2, memory: 4Gi                   │                  │
│  │                                          │                  │
│  │  networking:                             │  ← Networking    │
│  │    port: 8000                            │                  │
│  │    ingress: /api/v1/predict              │                  │
│  │    tls: true                             │                  │
│  │                                          │                  │
│  │  scaling:                                │  ← Auto-scaling  │
│  │    min: 2, max: 10                       │                  │
│  │    metric: cpu, target: 70%              │                  │
│  │                                          │                  │
│  │  monitoring:                             │  ← Observability │
│  │    healthcheck: /health                  │                  │
│  │    metrics: [latency_p99, error_rate]    │                  │
│  │    dashboard: grafana/fraud-api          │                  │
│  │                                          │                  │
│  │  alerting:                               │  ← Alerts        │
│  │    - error_rate > 1%: page-oncall        │                  │
│  │    - latency_p99 > 500ms: slack-warning  │                  │
│  │                                          │                  │
│  │  dependencies:                           │  ← Dependencies  │
│  │    - postgres: fraud-db                  │                  │
│  │    - redis: feature-cache                │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
│  ONE file defines EVERYTHING about the service.                │
│  Git push → CI/CD provisions all of the above.                 │
└────────────────────────────────────────────────────────────────┘
```

## **7.2 SaC Tools**

| Tool | Approach |
|------|----------|
| **Score** (Humanitec) | Workload specification that's platform-agnostic |
| **Waypoint** (HashiCorp) | URL, deploy, release workflow for any platform |
| **Helm Charts** | Kubernetes service packaging with templates |
| **Docker Compose** | Local multi-service definitions (simple SaC) |

## **7.3 IaC vs PaC vs SaC — Key Differences**

| Aspect | IaC | PaC | SaC |
|--------|-----|-----|-----|
| **Who writes it** | DevOps / Platform engineer | Platform team (templates) + Developers (specs) | Service owner / team |
| **Abstraction level** | Low (cloud primitives) | Medium (platform abstractions) | High (full service) |
| **Scope** | Individual resources | Curated resource bundles | Entire service lifecycle |
| **Knowledge needed** | Cloud-specific (AWS, GCP) | Platform API | Service requirements only |
| **Example** | Terraform `.tf` files | Crossplane Composition | Score `score.yaml` |
| **Primary user** | Infrastructure teams | Both platform & dev teams | Development teams |

---

# **8. Configuration as Code (CaC)**

## **8.1 What Is CaC?**

CaC manages **application and OS-level configuration** — not the infrastructure itself, but what runs on it.

| IaC Creates... | CaC Configures... |
|----------------|-------------------|
| The EC2 instance | Python version, packages, users |
| The Kubernetes cluster | Helm values, ConfigMaps, Secrets |
| The database server | Database settings, users, schemas |

## **8.2 CaC Tools**

| Tool | Approach | Language | Best For |
|------|----------|----------|----------|
| **Ansible** | Agentless, push-based | YAML playbooks | Config management, provisioning, app deployment |
| **Chef** | Agent-based, pull-based | Ruby (recipes) | Complex config management at scale |
| **Puppet** | Agent-based, pull-based | Puppet DSL | Enterprise config enforcement |
| **Salt** | Agent or agentless | YAML + Jinja2 | Remote execution + config management |

## **8.3 Ansible Example (Most Common for ML)**

```yaml
# setup_ml_server.yml
- hosts: gpu_servers
  become: yes
  tasks:
    - name: Install NVIDIA drivers
      apt:
        name: nvidia-driver-535
        state: present

    - name: Install Python and pip
      apt:
        name: [python3.11, python3-pip]
        state: present

    - name: Install ML dependencies
      pip:
        name: [torch, transformers, fastapi, uvicorn]
        state: present

    - name: Copy model artifacts
      copy:
        src: ./models/fraud_model_v2.pt
        dest: /opt/ml/models/
        mode: '0644'

    - name: Start inference service
      systemd:
        name: ml-api
        state: started
        enabled: yes
```

---

# **9. GitOps and IaC Workflows**

## **9.1 GitOps Principles**

```
┌────────────────────────────────────────────────────────────────┐
│                    GITOPS WORKFLOW                               │
│                                                                 │
│  Developer ──► Git Push ──► PR Review ──► Merge to main        │
│                                              │                  │
│                                              ▼                  │
│                                     CI/CD Pipeline              │
│                                     ┌──────────────────┐       │
│                                     │ terraform plan    │       │
│                                     │ (post as PR       │       │
│                                     │  comment)         │       │
│                                     └────────┬─────────┘       │
│                                              │ on merge         │
│                                              ▼                  │
│                                     ┌──────────────────┐       │
│                                     │ terraform apply   │       │
│                                     │ (auto on main)    │       │
│                                     └──────────────────┘       │
│                                                                 │
│  Git is the SINGLE SOURCE OF TRUTH.                            │
│  No manual changes. No console clicking.                       │
│  Everything goes through PR → Review → Merge → Apply.          │
└────────────────────────────────────────────────────────────────┘
```

## **9.2 Terraform CI/CD Best Practices**

| Practice | Implementation |
|----------|---------------|
| **PR-based workflow** | `terraform plan` on PR, `terraform apply` on merge |
| **Plan as PR comment** | Use `atlantis` or GitHub Actions to post plan output |
| **State locking** | DynamoDB (AWS), built-in (Terraform Cloud) |
| **Policy checks** | `tfsec`, `checkov`, `OPA/Sentinel` in CI pipeline |
| **Separate state per env** | `envs/dev/`, `envs/prod/` with different backends |
| **Module versioning** | Pin module versions in production |

---

# **10. IaC for ML Infrastructure**

## **10.1 Common ML Resources to Codify**

| Resource | Terraform Resource Type | Why Codify It |
|----------|----------------------|---------------|
| GPU instances for training | `aws_instance` (g4dn/p4d) | Expensive; need auto-shutdown |
| Model artifact storage | `aws_s3_bucket` | Versioning, lifecycle rules |
| Model registry | `aws_sagemaker_model` | Track model versions |
| Feature store | `aws_glue_catalog_database` | Consistent feature definitions |
| Inference endpoints | `aws_ecs_service` or `aws_sagemaker_endpoint` | Auto-scaling, versioned deployment |
| Experiment tracking | `aws_instance` (MLflow server) | Centralized experiment management |
| Vector databases | `aws_opensearch_domain` | RAG infrastructure |
| Monitoring dashboards | `grafana_dashboard` | Pre-built observability |

## **10.2 ML Platform Architecture (IaC)**

```
┌───────────────────────────────────────────────────────────┐
│         ML PLATFORM — ALL DEFINED IN TERRAFORM             │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Training  │  │ Feature  │  │ Model    │                │
│  │ Cluster   │  │ Store    │  │ Registry │                │
│  │ (EKS/GPU) │  │ (Feast)  │  │ (MLflow) │                │
│  └─────┬─────┘  └─────┬─────┘ └─────┬────┘               │
│        │               │              │                    │
│        └───────────────┼──────────────┘                    │
│                        │                                   │
│                   ┌────┴────┐                              │
│                   │  CI/CD  │                              │
│                   │Pipeline │                              │
│                   └────┬────┘                              │
│                        │                                   │
│  ┌──────────┐  ┌──────┴─────┐  ┌──────────┐              │
│  │ Inference │  │ Monitoring │  │ Vector   │              │
│  │ Services  │  │ (Grafana)  │  │ DB (RAG) │              │
│  │ (ECS)     │  │            │  │(OpenSearch)│             │
│  └──────────┘  └────────────┘  └──────────┘              │
└───────────────────────────────────────────────────────────┘
```

---

# **11. Interview Questions with Strong Answers**

---

## **Q1: "What is Terraform state and why is it important?"**

> Terraform state is a JSON file that maps your configuration to real-world resources. When you write `resource "aws_s3_bucket" "data"`, Terraform creates it and records the bucket's ID, ARN, region, and all attributes in state.
>
> State is critical because it's how Terraform knows what it manages. Without state, `terraform plan` can't compute the diff between desired and actual infrastructure. It's also the mechanism for dependency tracking — Terraform uses state to determine the order of operations.
>
> **In production:** State must be stored remotely (S3, Terraform Cloud) with locking (DynamoDB) to prevent concurrent modifications. State contains sensitive data (database passwords, etc.), so it must be encrypted. We use backend configuration for this, and we never commit `terraform.tfstate` to git.

---

## **Q2: "Explain the difference between IaC, PaC, and SaC."**

> They sit on a spectrum of abstraction:
>
> **IaC** (Infrastructure as Code) defines individual cloud resources — VMs, networks, buckets. You write Terraform or CloudFormation and you need deep cloud knowledge. The audience is DevOps/platform engineers.
>
> **PaC** (Platform as Code) provides **higher-level abstractions** on top of IaC. A platform team defines "golden paths" — e.g., a Crossplane Composition that says "an ML environment = GPU instance + storage + monitoring." Developers request an "ML environment, size medium" without knowing the underlying cloud details.
>
> **SaC** (Service as Code) goes furthest — it defines the **entire service lifecycle** in one spec: compute, networking, scaling, monitoring, alerting, dependencies. A service owner writes one `score.yaml` file, and the platform provisions everything end-to-end.
>
> **The trend:** Companies are moving UP this stack. IaC solved "infrastructure should be code." PaC and SaC solve "developers shouldn't need to be infrastructure experts."

---

## **Q3: "How do you manage multiple environments (dev/staging/prod) with Terraform?"**

> Three approaches, each with trade-offs:
>
> 1. **Workspaces:** Same code, different state per workspace. Simple but limited — environments often need different configurations.
>
> 2. **Directory per environment:** `envs/dev/main.tf`, `envs/prod/main.tf` each calling shared modules with different `.tfvars`. More flexible but risks code duplication.
>
> 3. **Terragrunt:** A wrapper that lets you keep code DRY while managing per-environment configuration. Best for large teams.
>
> In my practice, I use **shared modules + tfvars per environment** for medium projects, and Terragrunt for anything with 3+ environments and multiple teams.

---

## **Q4: "What is drift detection and how do you handle it?"**

> Drift is when the actual state of infrastructure diverges from the Terraform code — someone manually changed a setting in the console, or an external process modified a resource.
>
> Detection: `terraform plan` shows drift as changes. If someone manually changed an instance type from `t3.medium` to `t3.large`, the plan will show it wants to change it back to `t3.medium`.
>
> Handling strategies:
> - **Preventive:** Lock down console access with IAM; enforce all changes go through Terraform.
> - **Detective:** Run `terraform plan` on a schedule (e.g., nightly CI job) and alert on drift.
> - **Corrective:** `terraform apply` to enforce the desired state (after review). Or `terraform refresh` to update state to match reality if the manual change was intentional.

---

## **Q5: "You need to deploy an ML inference service. Walk me through the Terraform setup."**

> I'd structure this as Terraform modules:
>
> 1. **Networking module:** VPC, private subnets, security groups, ALB
> 2. **Container module:** ECR repository, ECS task definition with the model container, ECS service behind the ALB
> 3. **Scaling module:** Auto-scaling policy (target tracking on CPU/memory or custom metric like request latency)
> 4. **Monitoring module:** CloudWatch alarms for error rate, latency P99, model drift metrics
>
> Each module has its own `variables.tf` and `outputs.tf`. The root `main.tf` wires them together. State lives in S3 with DynamoDB locking. CI/CD runs `terraform plan` on PR, `terraform apply` on merge to main.
>
> For model updates specifically, I'd version the Docker image tag as a Terraform variable, so deploying a new model version is just changing `model_version = "v2.2"` in the tfvars and merging the PR.

---

## **Q6: "What is Policy as Code and why does it matter?"**

> Policy as Code means expressing security, compliance, and governance rules as machine-readable code that runs automatically in CI/CD.
>
> Tools: **OPA/Rego** (general policy engine), **HashiCorp Sentinel** (Terraform-native), **Checkov** / **tfsec** (static analysis for Terraform).
>
> Example: "No S3 bucket can be public" → a Rego rule that checks every `aws_s3_bucket` resource for `acl != "public-read"`. This runs in CI before `terraform apply` — if violated, the PR is blocked.
>
> It matters because at scale, you can't rely on code review to catch security issues. Policy as Code automates compliance and makes governance scalable.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│          TERRAFORM & PLATFORM ENGINEERING TAKEAWAYS               │
│                                                                   │
│  1. Terraform state is SACRED — remote, locked, encrypted        │
│                                                                   │
│  2. Modules = reusable functions for infrastructure               │
│                                                                   │
│  3. IaC → PaC → SaC is the evolution:                            │
│     Resources → Platform abstractions → Full service specs       │
│                                                                   │
│  4. GitOps: Git is the single source of truth for infra          │
│     PR → Plan → Review → Merge → Apply                           │
│                                                                   │
│  5. Policy as Code: Automated compliance (OPA, Sentinel)         │
│                                                                   │
│  6. For ML: Codify training clusters, model registries,          │
│     inference endpoints, monitoring — everything in Terraform    │
│                                                                   │
│  7. Know when to use workspaces vs directories vs Terragrunt    │
│                                                                   │
│  8. Drift detection: Scheduled `terraform plan` + locked-down   │
│     console access                                                │
└──────────────────────────────────────────────────────────────────┘
```
