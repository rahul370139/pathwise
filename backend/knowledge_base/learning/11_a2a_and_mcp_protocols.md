# A2A & MCP Protocols — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, CLIP LoRA Fine-Tuning, Multimodal Systems, MCP

---

# Table of Contents

1. [A2A Protocol — "HTTP for AI Agents"](#1-a2a-protocol--http-for-ai-agents)
2. [MCP — Model Context Protocol](#2-mcp--model-context-protocol)
3. [A2A vs MCP — How They Relate](#3-a2a-vs-mcp--how-they-relate)
4. [Inter-Model Communication](#4-inter-model-communication)
5. [Practical Architecture](#5-practical-architecture)
6. [Common Interview Questions](#6-common-interview-questions-with-strong-answers)
7. [Key Takeaways](#7-key-takeaways)

---

# **1. A2A Protocol — "HTTP for AI Agents"**

---

## **1.1 What Is A2A?**

A2A (**Agent-to-Agent**) is an **open, vendor-neutral protocol** that standardizes how autonomous AI agents discover, authenticate, and communicate with each other — regardless of the framework, model, or vendor behind them.

Think of it this way:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        THE ANALOGY                                   │
│                                                                      │
│   HTTP  = universal standard for web pages to talk to browsers       │
│   A2A   = universal standard for AI agents to talk to each other     │
│                                                                      │
│   Before HTTP  → every vendor had a proprietary browsing protocol    │
│   Before A2A   → every vendor has a proprietary agent protocol       │
│                                                                      │
│   HTTP gave us  → the interoperable World Wide Web                   │
│   A2A gives us  → the interoperable Agent Web                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Without A2A:** If you build an agent on LangChain and your partner's agent runs on CrewAI, they can't natively talk to each other. You'd need custom glue code, proprietary APIs, or middleware — fragile and vendor-locked.

**With A2A:** Both agents expose a standard Agent Card, speak JSON-RPC 2.0 over HTTPS, and can exchange tasks, artifacts, and status updates without knowing each other's internals.

> **Interview tip:** Lead with the HTTP analogy. Interviewers may not know A2A deeply — grounding it in something universally understood (HTTP) immediately conveys the value.

---

## **1.2 Design Principles**

A2A was developed by Google in collaboration with 50+ partners (Salesforce, SAP, Atlassian, MongoDB, etc.) and is built on five core principles:

| # | Principle | What It Means | Why It Matters |
|---|---|---|---|
| 1 | **Embrace Agentic Capabilities** | Agents are first-class citizens, not just API endpoints. They can negotiate, delegate, and collaborate. | Enables complex multi-agent workflows (planning, delegation, escalation). |
| 2 | **Build on Existing Web Standards** | Uses HTTPS, JSON-RPC 2.0, Server-Sent Events (SSE), OAuth 2.0. | Zero new infrastructure — runs on the web stack engineers already know. |
| 3 | **Secure by Default** | Enterprise-grade auth (OAuth 2.0 / API keys), encrypted transport (TLS), capability-based access control. | Production-ready from day one; satisfies compliance requirements. |
| 4 | **Support Long-Running Tasks** | Tasks can span minutes, hours, or days. Status polling and streaming keep clients informed. | Real-world tasks (code review, data pipelines, research) aren't instant. |
| 5 | **Modality Agnostic** | Supports text, images, audio, video, structured data, and embeddings as message payloads. | Agents can collaborate on multimodal tasks without protocol changes. |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    A2A DESIGN PRINCIPLES                             │
│                                                                     │
│    ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│    │  Agentic   │  │    Web     │  │  Secure    │                   │
│    │ First-Class│  │ Standards  │  │ by Default │                   │
│    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                   │
│          │               │               │                          │
│          └───────────────┼───────────────┘                          │
│                          │                                          │
│                    ┌─────┴──────┐                                    │
│                    │    A2A     │                                    │
│                    │  Protocol  │                                    │
│                    └─────┬──────┘                                    │
│                          │                                          │
│          ┌───────────────┼───────────────┐                          │
│          │               │               │                          │
│    ┌─────┴──────┐  ┌─────┴──────┐  ┌─────┴──────┐                   │
│    │   Long-    │  │  Modality  │  │  Vendor    │                   │
│    │  Running   │  │  Agnostic  │  │  Neutral   │                   │
│    └────────────┘  └────────────┘  └────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **1.3 How A2A Works**

### **Step 1: Agent Card — Discovery**

Every A2A-compliant agent publishes an **Agent Card** at a well-known URL:

```
GET https://agent.example.com/.well-known/agent.json
```

The Agent Card is a JSON document that describes the agent's identity, capabilities, authentication requirements, and supported modalities:

```json
{
  "name": "ResearchAssistant",
  "description": "Searches academic papers, summarizes findings, generates citations",
  "version": "2.1.0",
  "url": "https://research-agent.example.com",
  "skills": [
    {
      "id": "paper_search",
      "name": "Academic Paper Search",
      "description": "Searches arXiv, PubMed, Semantic Scholar for relevant papers",
      "inputModes": ["text"],
      "outputModes": ["text", "application/json"]
    },
    {
      "id": "summarize",
      "name": "Paper Summarization",
      "description": "Generates structured summaries of academic papers",
      "inputModes": ["text", "application/pdf"],
      "outputModes": ["text", "text/markdown"]
    }
  ],
  "authentication": {
    "schemes": ["OAuth2", "Bearer"],
    "credentials": "https://research-agent.example.com/auth"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "provider": {
    "organization": "Rahul's ML Lab",
    "url": "https://ml-lab.example.com"
  }
}
```

> **Interview tip:** The Agent Card is the "business card" of an AI agent. It answers three questions: *Who are you? What can you do? How do I authenticate?*

---

### **Step 2: Task Submission — `POST /tasks/send`**

Once a client agent discovers a remote agent's capabilities, it submits a **Task**:

```json
POST https://research-agent.example.com/tasks/send
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tasks/send",
  "params": {
    "id": "task-abc-123",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "Find the top 5 papers on LoRA fine-tuning published after 2023 and summarize each in 3 bullet points."
        }
      ]
    },
    "metadata": {
      "priority": "high",
      "timeout_seconds": 300,
      "callback_url": "https://orchestrator.example.com/callbacks/task-abc-123"
    }
  }
}
```

**Task lifecycle:**

```
┌───────────────────────────────────────────────────────────────────┐
│                      TASK STATE MACHINE                            │
│                                                                    │
│   ┌───────────┐    ┌────────────┐    ┌─────────────┐              │
│   │ submitted │───▶│  working   │───▶│  completed  │              │
│   └───────────┘    └─────┬──────┘    └─────────────┘              │
│                          │                                        │
│                          ├──────────▶ ┌─────────────┐             │
│                          │            │   failed    │             │
│                          │            └─────────────┘             │
│                          │                                        │
│                          ├──────────▶ ┌─────────────┐             │
│                          │            │  canceled   │             │
│                          │            └─────────────┘             │
│                          │                                        │
│                          └──────────▶ ┌──────────────────┐        │
│                                       │ input-required   │        │
│                                       │ (needs human/    │        │
│                                       │  agent feedback) │        │
│                                       └──────────────────┘        │
└───────────────────────────────────────────────────────────────────┘
```

| State | Meaning |
|---|---|
| `submitted` | Task received, queued for processing |
| `working` | Agent is actively processing the task |
| `input-required` | Agent needs clarification or additional input |
| `completed` | Task finished successfully |
| `failed` | Task encountered an unrecoverable error |
| `canceled` | Task was cancelled by the client or system |

---

### **Step 3: Streaming — `tasks/sendSubscribe` with SSE**

For long-running tasks, A2A supports **Server-Sent Events (SSE)** so the client receives real-time progress updates:

```json
POST https://research-agent.example.com/tasks/sendSubscribe
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": "req-002",
  "method": "tasks/sendSubscribe",
  "params": {
    "id": "task-abc-123",
    "message": {
      "role": "user",
      "parts": [
        { "type": "text", "text": "Analyze this 500-page PDF and extract all methodology sections." }
      ]
    }
  }
}
```

**SSE response stream:**

```
event: status
data: {"taskId": "task-abc-123", "state": "working", "message": "Parsing PDF... page 12/500"}

event: status
data: {"taskId": "task-abc-123", "state": "working", "message": "Extracting methodology... 3 sections found so far"}

event: artifact
data: {"taskId": "task-abc-123", "artifact": {"type": "text/markdown", "name": "methodology_sections.md", "parts": [{"type": "text", "text": "## Section 1: ..."}]}}

event: status
data: {"taskId": "task-abc-123", "state": "completed"}
```

---

### **Step 4: Artifact Delivery**

When a task completes, the remote agent returns **Artifacts** — the deliverables:

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "id": "task-abc-123",
    "state": "completed",
    "artifacts": [
      {
        "name": "top_5_lora_papers.md",
        "parts": [
          {
            "type": "text",
            "text": "## Top 5 LoRA Papers (2023-2025)\n\n### 1. QLoRA: Efficient Finetuning...\n- Enables 65B parameter model fine-tuning on single 48GB GPU\n- Uses 4-bit NormalFloat quantization\n- Matches full 16-bit fine-tuning performance..."
          }
        ],
        "metadata": {
          "mimeType": "text/markdown",
          "generatedAt": "2025-12-15T10:30:00Z"
        }
      },
      {
        "name": "citation_data.json",
        "parts": [
          {
            "type": "data",
            "data": {
              "papers": [
                {"title": "QLoRA", "arxiv_id": "2305.14314", "citations": 2847},
                {"title": "LoRA+", "arxiv_id": "2402.12354", "citations": 412}
              ]
            }
          }
        ]
      }
    ],
    "history": [
      {"role": "user", "parts": [{"type": "text", "text": "Find the top 5 papers..."}]},
      {"role": "agent", "parts": [{"type": "text", "text": "I found 5 highly relevant papers..."}]}
    ]
  }
}
```

> **Interview tip:** Artifacts are multimodal — they can contain text, structured JSON data, images, audio, or binary files. This is what makes A2A modality-agnostic.

---

## **1.4 Roles in the A2A Ecosystem**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         A2A ROLE ARCHITECTURE                            │
│                                                                          │
│   ┌──────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│   │              │  POST   │                  │  POST   │             │ │
│   │  Client      │────────▶│  Broker/Router   │────────▶│   Remote    │ │
│   │  Agent       │◀────────│  (optional)      │◀────────│   Agent     │ │
│   │              │  SSE    │                  │  SSE    │             │ │
│   └──────────────┘         └──────────────────┘         └─────────────┘ │
│                                                                          │
│   Initiates tasks          Routes, load-balances,       Receives tasks,  │
│   Consumes artifacts       authenticates, logs          returns artifacts │
│                                                                          │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                          │
│   DIRECT MODE (no broker):                                               │
│                                                                          │
│   ┌──────────────┐         ┌──────────────┐                              │
│   │  Client      │────────▶│   Remote     │                              │
│   │  Agent       │◀────────│   Agent      │                              │
│   └──────────────┘         └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

| Role | Responsibility | Example |
|---|---|---|
| **Client Agent** | Discovers remote agents, submits tasks, consumes results | An orchestrator that delegates "search the web" to a search agent |
| **Remote Agent** | Advertises capabilities via Agent Card, processes tasks, returns artifacts | A specialized code-review agent that analyzes PRs |
| **Broker / Router** | (Optional) Sits between client and remote agents; handles service discovery, load balancing, auth verification, logging | An API gateway that routes tasks to the best-fit agent from a registry |

---

## **1.5 Message Types**

A2A message parts can carry different content modalities. The three canonical inter-agent message types are:

| Message Type | Payload | Latency | Use Case |
|---|---|---|---|
| **`latent_exchange`** | Dense float32/float16 embeddings | ~2 ms | Agent A sends a vector representation to Agent B for similarity search, bypassing NL overhead |
| **`cognitive_call`** | Natural-language text | ~200-800 ms | Agent A asks Agent B a question in plain English: "What are the key findings of paper X?" |
| **`tool_invocation`** | JSON-Schema-validated function call | ~50-150 ms | Agent A requests Agent B to execute a specific tool: `{"tool": "sql_query", "params": {"query": "SELECT..."}}` |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MESSAGE TYPE COMPARISON                            │
│                                                                      │
│   LATENT EXCHANGE        COGNITIVE CALL         TOOL INVOCATION      │
│   ─────────────────     ──────────────────     ──────────────────    │
│   [0.23, -0.87, ...]    "Summarize paper      {"tool": "search",    │
│                           on LoRA tuning"        "params": {         │
│   Agent ──▶ Agent        Agent ──▶ Agent          "q": "LoRA"}}     │
│                                                                      │
│   Fastest, lossy         Richest, slowest      Structured, typed     │
│   No NL overhead         Full NL reasoning     Schema-validated      │
│   Embedding space        Token-level comm.     Deterministic exec.   │
└─────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Latent exchange is the key differentiator for **inter-model** communication — it bypasses the text bottleneck entirely, allowing agents to share compressed representations at near-zero latency.

---

## **1.6 A2A Architecture — End-to-End**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        A2A END-TO-END ARCHITECTURE                           │
│                                                                              │
│  ┌─────────┐    ┌───────────────────────────────────────────────┐            │
│  │  User   │    │              ORCHESTRATOR AGENT                │            │
│  │ (human) │    │                                               │            │
│  └────┬────┘    │  ┌────────────┐  ┌────────────┐  ┌─────────┐ │            │
│       │         │  │ Task Queue │  │ State Mgr  │  │ Router  │ │            │
│       │ NL req  │  └────────────┘  └────────────┘  └────┬────┘ │            │
│       ▼         └───────────────────────────────────────┬┘─────-┘            │
│  ┌─────────┐                                            │                    │
│  │  Front  │◀──────────── SSE / WebSocket ──────────────┤                    │
│  │  End    │                                            │                    │
│  └─────────┘                                            │                    │
│                                    ┌────────────────────┼──────────────┐     │
│                                    │                    │              │     │
│                                    ▼                    ▼              ▼     │
│                          ┌──────────────┐    ┌──────────────┐  ┌──────────┐ │
│                          │  Research    │    │  Code        │  │  Data    │ │
│                          │  Agent      │    │  Agent       │  │  Agent   │ │
│                          │             │    │              │  │          │ │
│                          │ /.well-known│    │ /.well-known │  │/.well-   │ │
│                          │ /agent.json │    │ /agent.json  │  │known/    │ │
│                          │             │    │              │  │agent.json│ │
│                          └──────┬──────┘    └──────┬───────┘  └────┬─────┘ │
│                                 │                  │               │       │
│                                 ▼                  ▼               ▼       │
│                          ┌──────────┐       ┌──────────┐    ┌──────────┐   │
│                          │ arXiv    │       │ GitHub   │    │ pgvector │   │
│                          │ PubMed   │       │ Linters  │    │ BigQuery │   │
│                          │ S.Scholar│       │ Sandbox  │    │ Snowflake│   │
│                          └──────────┘       └──────────┘    └──────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## **1.7 Memory Footprint**

A2A is designed to be **lightweight**:

| Component | Memory | Notes |
|---|---|---|
| Agent Card (cached) | ~2-5 KB | JSON; refreshed on TTL expiry |
| Task object (active) | ~5-15 KB | Includes message history, state, metadata |
| SSE connection | ~8-12 KB | Per open streaming connection |
| Message part (text) | ~1-5 KB | Depends on content length |
| Message part (embedding) | ~4 KB | 1024-dim float32 vector |
| **Total per active chat** | **< 50 KB** | Excludes artifact payloads |

> **Interview tip:** < 50 KB per active chat means you can hold ~20,000 concurrent agent conversations per GB of RAM — A2A is designed for high-density agent deployments.

---

## **1.8 Deployment Pipeline**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     A2A DEPLOYMENT PIPELINE                               │
│                                                                          │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────────┐   │
│  │   DEV     │──▶│  CI/CD    │──▶│   K8s     │──▶│ OBSERVABILITY    │   │
│  └───────────┘   └───────────┘   └───────────┘   └──────────────────┘   │
│                                                                          │
│  • Write agent   • Validate     • Helm chart    • Prometheus metrics     │
│    code            Agent Card     per agent      • Grafana dashboards    │
│  • Define        • Run A2A      • HPA for       • OpenTelemetry traces  │
│    Agent Card      conformance    auto-scaling   • Alerting on task      │
│  • Local test      tests        • Service mesh     failure rates         │
│    with mock     • Integration    (Istio/Linkerd)• Log aggregation       │
│    agents          tests        • TLS            • SLA monitoring        │
│  • Schema        • Security       termination                            │
│    validation      scanning     • Secrets mgmt                           │
│               • Push image                                               │
│                 to registry                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

| Stage | Tools | Key Actions |
|---|---|---|
| **Dev** | Python/TS SDK, local Docker | Implement agent logic, define Agent Card, unit tests |
| **CI/CD** | GitHub Actions, GitLab CI | Agent Card schema validation, conformance tests, security scan, container build |
| **K8s** | Helm, Kustomize, ArgoCD | Deploy as microservices, HPA autoscaling, service mesh for mTLS |
| **Observability** | Prometheus, Grafana, OTel | Task latency P50/P95/P99, failure rate, concurrent connections, token throughput |

---

## **1.9 Why A2A Matters**

| Benefit | Explanation |
|---|---|
| **Interoperability** | LangChain agent ↔ CrewAI agent ↔ AutoGen agent — all speak the same protocol |
| **Future-Proof** | As new LLMs and frameworks appear, they just need to implement A2A — no rewiring |
| **Enterprise-Ready** | OAuth 2.0, audit logs, capability-based access control, encrypted transport |
| **Vendor Neutrality** | No lock-in to Google, OpenAI, or Anthropic — the protocol is open-spec |
| **Composability** | Build complex workflows by wiring together specialized agents like LEGO blocks |

---

# **2. MCP — Model Context Protocol**

---

## **2.1 What Is MCP?**

MCP (**Model Context Protocol**) is a **standardized JSON envelope** that carries **facts, memories, tool definitions, and metadata** alongside prompts when communicating with LLMs. It was introduced by Anthropic and has become the emerging standard for structured context delivery.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        THE ANALOGY                                    │
│                                                                       │
│   Without MCP:                                                        │
│   ┌──────────────────────────────────────┐                            │
│   │  "Here's some context... also this   │                            │
│   │   tool exists... and remember that   │ ──▶ LLM ──▶ 🤷 inconsistent│
│   │   thing from before... oh and here's │                            │
│   │   my system prompt..."               │                            │
│   └──────────────────────────────────────┘                            │
│                                                                       │
│   With MCP:                                                           │
│   ┌───────────────────────┐                                           │
│   │  MCP Packet           │                                           │
│   │  ┌─ header (routing) ─┤                                           │
│   │  ├─ context (facts)  ─┤ ──▶ LLM ──▶ ✅ structured, consistent    │
│   │  ├─ tools (schemas)  ─┤                                           │
│   │  └─ state (memory)   ─┤                                           │
│   └───────────────────────┘                                           │
│                                                                       │
│   MCP = USB-C for LLM context                                        │
│   One standard plug, works with every model                           │
└──────────────────────────────────────────────────────────────────────┘
```

**The core idea:** Instead of duct-taping context into prompts differently for every LLM, MCP defines a **universal envelope** that any model can consume — making context delivery **plug-and-play**.

---

## **2.2 Why MCP?**

| Problem (Without MCP) | Solution (With MCP) |
|---|---|
| Every LLM integration needs custom context formatting | One standard envelope works for all LLMs |
| Tool definitions scattered across codebases | Tools declared once in MCP schema, auto-discovered |
| Memory/state management is ad-hoc | Structured state section carries conversation history, embeddings, preferences |
| No audit trail for what context was sent | MCP packets are serializable — every turn is logged and reproducible |
| Switching LLM providers requires rewriting glue code | Swap the model, keep the MCP envelope — same interface |

```
┌─────────────────────────────────────────────────────────────────────┐
│               WITHOUT MCP vs WITH MCP                                │
│                                                                      │
│   WITHOUT MCP (Duct-Tape)          WITH MCP (Plug-and-Play)         │
│   ─────────────────────            ──────────────────────────       │
│                                                                      │
│   App ──[custom format]──▶ GPT-4   App ──[MCP packet]──▶ GPT-4     │
│   App ──[different fmt]──▶ Claude   App ──[MCP packet]──▶ Claude    │
│   App ──[yet another]───▶ Gemini   App ──[MCP packet]──▶ Gemini    │
│   App ──[more glue]─────▶ Llama    App ──[MCP packet]──▶ Llama     │
│                                                                      │
│   4 integrations = 4 formats       4 integrations = 1 format        │
│   N models = N formats             N models = 1 format              │
└─────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** "MCP solves the N×M integration problem. Without it, N applications × M models = N×M custom integrations. With MCP, it's N + M — each side just implements the standard once."

---

## **2.3 How MCP Works — The Pipeline**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP DATA FLOW PIPELINE                               │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │  PRODUCER   │───▶│  PACKAGER   │───▶│  CONSUMER   │───▶│  OBSERVER   │  │
│   │             │    │             │    │   (LLM)     │    │             │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│   Sources of          Assembles the      Receives the      Monitors,         │
│   context:            MCP packet:        packet and        logs, audits:     │
│                                          generates a                         │
│   • Vector DB         • Header           response          • Token counts    │
│   • User profile      • Context slots                      • Latency         │
│   • Tool registry     • Tool schemas     • GPT-4           • Compliance      │
│   • Conversation      • State/memory     • Claude          • Cost tracking   │
│     history           • Constraints      • Gemini          • Drift detection │
│   • External APIs                        • Llama           • Replay/debug    │
│   • Document store                       • Mistral                           │
│                                                                              │
│   ═══════════════════════════════════════════════════════════════════════    │
│   Example flow:                                                              │
│                                                                              │
│   pgvector ──▶ ┐                                                             │
│   user prefs ──┤  Packager ──▶ MCP Packet ──▶ Claude 3.5 ──▶ Response       │
│   tool defs ───┤                                    │                        │
│   memory ──────┘                                    └──▶ Observer (log)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Role | Example |
|---|---|---|
| **Producer** | Generates raw context from data sources | Vector DB retrieves top-5 chunks; user profile loads preferences |
| **Packager** | Assembles context into a structured MCP packet | Combines RAG results + tool schemas + memory into a single envelope |
| **Consumer** | The LLM that receives and processes the packet | Claude generates a response grounded in the provided context |
| **Observer** | Monitors, logs, and audits the entire flow | Logs token count, latency, cost; enables compliance auditing and replay |

---

## **2.4 MCP Packet Structure**

An MCP packet has four sections: **header**, **context**, **tools**, and **state**.

```json
{
  "header": {
    "version": "1.2",
    "request_id": "mcp-req-789",
    "timestamp": "2025-12-15T10:30:00Z",
    "model_hint": "claude-3.5-sonnet",
    "max_tokens": 4096,
    "constraints": {
      "temperature": 0.3,
      "response_format": "markdown",
      "safety_level": "strict"
    }
  },

  "context": [
    {
      "role": "system",
      "content": "You are a senior ML engineer assistant. Answer with code examples when relevant.",
      "priority": 1
    },
    {
      "role": "rag",
      "source": "pgvector://ml-knowledge-base",
      "chunks": [
        {
          "text": "LoRA adds low-rank matrices A and B to frozen weight W, so W' = W + BA where B ∈ R^{d×r} and A ∈ R^{r×k}...",
          "score": 0.94,
          "metadata": {"doc_id": "lora-paper-2106", "page": 3}
        },
        {
          "text": "QLoRA extends LoRA with 4-bit NormalFloat quantization, enabling 65B model fine-tuning on a single 48GB GPU...",
          "score": 0.91,
          "metadata": {"doc_id": "qlora-paper-2305", "page": 1}
        }
      ],
      "priority": 2
    },
    {
      "role": "user",
      "content": "Explain the difference between LoRA and QLoRA with a code example.",
      "priority": 3
    }
  ],

  "tools": [
    {
      "name": "execute_python",
      "description": "Execute Python code in a sandboxed environment and return stdout/stderr",
      "input_schema": {
        "type": "object",
        "properties": {
          "code": {"type": "string", "description": "Python code to execute"},
          "timeout_seconds": {"type": "integer", "default": 30}
        },
        "required": ["code"]
      }
    },
    {
      "name": "search_papers",
      "description": "Search academic papers on arXiv and Semantic Scholar",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "max_results": {"type": "integer", "default": 5},
          "year_from": {"type": "integer"}
        },
        "required": ["query"]
      }
    }
  ],

  "state": {
    "conversation_id": "conv-456",
    "turn_number": 3,
    "memory": {
      "short_term": [
        {"turn": 1, "summary": "User asked about fine-tuning methods for LLMs"},
        {"turn": 2, "summary": "Discussed full fine-tuning vs parameter-efficient methods"}
      ],
      "long_term": {
        "user_expertise": "advanced",
        "preferred_framework": "PyTorch",
        "topics_covered": ["transformers", "attention", "fine-tuning"]
      },
      "embeddings": {
        "conversation_vector": [0.12, -0.45, 0.78, "... (768-dim)"],
        "user_intent_vector": [0.34, 0.67, -0.23, "... (768-dim)"]
      }
    },
    "token_budget": {
      "total_limit": 128000,
      "used_so_far": 3420,
      "remaining": 124580
    }
  }
}
```

### **Section Breakdown:**

| Section | Purpose | Contents |
|---|---|---|
| **`header`** | Routing & configuration | Version, request ID, model hint, token limits, temperature, safety constraints |
| **`context`** | The knowledge payload | System prompts, RAG chunks (with scores & metadata), user messages — each with priority |
| **`tools`** | Available tool definitions | JSON-Schema-validated function signatures the model can call |
| **`state`** | Memory & conversation tracking | Short-term memory, long-term preferences, embedding vectors, token budget tracking |

---

## **2.5 JSON-RPC 2.0 — MCP's Transport Layer**

MCP uses **JSON-RPC 2.0** as its transport protocol, with three architectural roles:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MCP JSON-RPC 2.0 ARCHITECTURE                          │
│                                                                           │
│   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐       │
│   │     HOST     │        │    CLIENT    │        │    SERVER    │       │
│   │              │        │              │        │              │       │
│   │  IDE/App/    │ spawn  │  Protocol    │ JSON-  │  Context     │       │
│   │  Platform    │───────▶│  Bridge      │ RPC    │  Provider    │       │
│   │              │        │              │◀──────▶│              │       │
│   │  Manages     │        │  Translates  │  2.0   │  Serves:     │       │
│   │  lifecycle,  │        │  app needs   │        │  • Resources │       │
│   │  security,   │        │  to MCP      │        │  • Tools     │       │
│   │  permissions │        │  requests    │        │  • Prompts   │       │
│   └──────────────┘        └──────────────┘        └──────────────┘       │
│                                                                           │
│   Examples:                Examples:              Examples:               │
│   • Cursor IDE             • Built into host      • File system server   │
│   • Claude Desktop         • SDK libraries        • Database connector   │
│   • Custom apps            • LangChain adapter    • API bridge           │
│                                                   • Vector DB server     │
│                                                                           │
│   ═══════════════════════════════════════════════════════════════════     │
│                                                                           │
│   JSON-RPC 2.0 Message Examples:                                         │
│                                                                           │
│   Request:  {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}   │
│   Response: {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}            │
│   Notify:   {"jsonrpc":"2.0","method":"notifications/progress",          │
│              "params":{"token":"abc","progress":0.75}}                    │
└──────────────────────────────────────────────────────────────────────────┘
```

| Role | Responsibility | Multiplicity |
|---|---|---|
| **Host** | The application (IDE, chatbot, platform) that manages security, permissions, and MCP client lifecycles | 1 Host : N Clients |
| **Client** | Protocol bridge that maintains a 1:1 connection with an MCP server, translating application needs into MCP requests | 1 Client : 1 Server |
| **Server** | Provides context (resources, tools, prompts) to clients via the MCP protocol | 1 Server : N Resources |

---

## **2.6 MCP Architecture — "Model-in-the-Loop"**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MCP "MODEL-IN-THE-LOOP" ARCHITECTURE                       │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐     │
│   │                        HOST APPLICATION                              │     │
│   │                     (Cursor IDE / Claude Desktop)                    │     │
│   │                                                                      │     │
│   │   ┌───────────┐    ┌───────────┐    ┌───────────┐                    │     │
│   │   │  MCP      │    │  MCP      │    │  MCP      │                    │     │
│   │   │ Client A  │    │ Client B  │    │ Client C  │                    │     │
│   │   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘                    │     │
│   │         │                │                │                          │     │
│   └─────────┼────────────────┼────────────────┼──────────────────────────┘     │
│             │                │                │                                │
│      ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐                        │
│      │ MCP Server  │  │ MCP Server │  │ MCP Server  │                        │
│      │ (Files)     │  │ (Database) │  │ (Git)       │                        │
│      │             │  │            │  │             │                        │
│      │ Resources:  │  │ Resources: │  │ Resources:  │                        │
│      │ • read_file │  │ • query    │  │ • git_log   │                        │
│      │ • list_dir  │  │ • schema   │  │ • git_diff  │                        │
│      │ • search    │  │ • insert   │  │ • git_blame │                        │
│      └─────────────┘  └────────────┘  └─────────────┘                        │
│                                                                               │
│   ═══════════════════════════════════════════════════════════════════════     │
│                                                                               │
│   The "Model-in-the-Loop" Pattern:                                           │
│                                                                               │
│   1. User sends prompt in Host                                               │
│   2. Host's LLM decides which MCP tools/resources it needs                   │
│   3. MCP Clients fetch context from MCP Servers                              │
│   4. Context packaged into MCP packet → sent to LLM                          │
│   5. LLM reasons with full context → may call more tools                     │
│   6. Loop continues until task is complete                                    │
│                                                                               │
│   User ──▶ Host ──▶ LLM ──▶ MCP Client ──▶ MCP Server ──▶ Data              │
│                      ▲                                       │                │
│                      └───────────── context ◀────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** The "model-in-the-loop" pattern is key — the LLM is an *active participant* that decides what context it needs, not a passive consumer of pre-assembled prompts. This is what makes MCP agentic.

---

## **2.7 Memory Footprint**

MCP packets are designed to be **compact**:

| Component | Size | Notes |
|---|---|---|
| Header | ~200-500 bytes | Routing metadata, constraints |
| Context (system prompt) | ~500 bytes - 2 KB | Depends on prompt length |
| Context (RAG chunks) | ~1-5 KB | Typically 3-5 chunks at ~200-1000 chars each |
| Tools section | ~500 bytes - 2 KB | Depends on number of tool definitions |
| State (short-term memory) | ~200 bytes - 1 KB | Summarized conversation turns |
| State (embeddings) | ~3-6 KB | If carrying conversation/intent vectors |
| **Total per turn** | **~1-10 KB** | Excludes the LLM's own token processing |

```
┌─────────────────────────────────────────────────────────────────────┐
│              MCP vs RAW PROMPT — SIZE COMPARISON                     │
│                                                                      │
│   Raw prompt (unstructured):                                         │
│   ├── System prompt (1 KB)                                           │
│   ├── Conversation history (5 KB)    ← redundant, uncompressed      │
│   ├── RAG context (3 KB)             ← no metadata, no scores       │
│   └── Tool instructions (2 KB)       ← free-text, error-prone       │
│   Total: ~11 KB (with waste and no structure)                        │
│                                                                      │
│   MCP packet (structured):                                           │
│   ├── Header (0.3 KB)               ← routing, dedup                │
│   ├── Context (3 KB)                ← prioritized, scored           │
│   ├── Tools (1 KB)                  ← schema-validated              │
│   └── State (1.5 KB)               ← compressed memory             │
│   Total: ~5.8 KB (structured, auditable, zero waste)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **2.8 Production Pipeline**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      MCP PRODUCTION PIPELINE                                  │
│                                                                               │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐ │
│  │  DEV   │─▶│ CI/CD  │─▶│ DEPLOY │─▶│ SCALE  │─▶│ OBSERVE │─▶│ ITERATE  │ │
│  └────────┘  └────────┘  └────────┘  └────────┘  └─────────┘  └──────────┘ │
│                                                                               │
│  • Define      • Schema     • MCP       • Token     • Log every  • A/B test  │
│    MCP           validate     servers     budget      MCP packet   context    │
│    schemas       packets      as K8s      quotas      (audit       strategies │
│  • Build       • Test tool    sidecars    per         trail)     • Optimize   │
│    MCP           schemas    • Feature     tenant    • Monitor      chunk       │
│    servers     • Fuzz test    flags     • Cache       token        selection  │
│  • Local         context      for new     frequent    usage      • Update     │
│    testing       payloads     context     RAG       • Alert on     tool       │
│  • Mock LLM   • Cost          sources     chunks      cost         schemas   │
│    responses     estimation                           spikes     • Retrain   │
│                                                     • Drift        embeddings│
│                                                       detection              │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Key Actions |
|---|---|
| **Dev** | Define MCP schemas, build MCP servers for each data source, test with mock LLMs |
| **CI/CD** | Validate packet schemas, fuzz-test context payloads, estimate token costs per packet |
| **Deploy** | Run MCP servers as K8s sidecars or standalone services, feature-flag new context sources |
| **Scale** | Set per-tenant token budgets, cache frequently-used RAG chunks, shard vector DBs |
| **Observe** | Log every MCP packet for audit/replay, monitor token usage and cost, detect context drift |
| **Iterate** | A/B test context strategies, optimize chunk selection algorithms, update tool schemas |

---

## **2.9 Benefits of MCP**

| Benefit | Explanation |
|---|---|
| **Portability** | Same MCP packet works with GPT-4, Claude, Gemini, Llama — swap models, not code |
| **Compliance & Audit** | Every MCP packet is serializable JSON — store it, replay it, prove what the model saw |
| **Cost-Cutting** | Structured context with priorities and token budgets prevents wasted tokens |
| **Future-Proof** | As new models appear, they just need to read MCP packets — no rewriting integrations |
| **Debuggability** | When a model hallucinates, inspect the MCP packet to see exactly what context it had |
| **Composability** | MCP servers are modular — add a new data source by deploying a new MCP server |

---

# **3. A2A vs MCP — How They Relate**

---

## **3.1 The Key Distinction**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    A2A vs MCP — THE RELATIONSHIP                              │
│                                                                               │
│                                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐    │
│   │                                                                      │    │
│   │   MCP = WHAT gets shared                                            │    │
│   │         (data, memory, tool definitions, context)                    │    │
│   │                                                                      │    │
│   │   A2A = HOW it's transported                                        │    │
│   │         (JSON-RPC handshake, discovery, task lifecycle)              │    │
│   │                                                                      │    │
│   └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   Together, they form a complete agent communication stack:                   │
│                                                                               │
│   ┌─────────────────────────────────────────────┐                            │
│   │           APPLICATION LAYER                  │                            │
│   │     (Agent logic, orchestration)             │                            │
│   ├─────────────────────────────────────────────┤                            │
│   │           MCP LAYER (Content)                │  ◀─ WHAT is sent          │
│   │     (Context, tools, memory, state)          │                            │
│   ├─────────────────────────────────────────────┤                            │
│   │           A2A LAYER (Transport)              │  ◀─ HOW it's sent         │
│   │     (Discovery, auth, tasks, streaming)      │                            │
│   ├─────────────────────────────────────────────┤                            │
│   │           NETWORK LAYER                      │                            │
│   │     (HTTPS, TLS, SSE, WebSocket)             │                            │
│   └─────────────────────────────────────────────┘                            │
│                                                                               │
│   Think of it like email:                                                    │
│   • SMTP  = A2A  (the protocol for sending/receiving)                        │
│   • MIME  = MCP  (the format of the content inside)                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** "MCP is the envelope; A2A is the postal service." This one-liner captures the relationship perfectly.

---

## **3.2 Using A2A to Shuttle MCP Packets Among Agents**

In a multi-agent system, **A2A carries MCP packets** between agents:

```
┌────────────────────────────────────────────────────────────────────────────┐
│             A2A TRANSPORTING MCP PACKETS                                    │
│                                                                             │
│   ┌────────────────┐                          ┌────────────────┐           │
│   │  Agent A       │                          │  Agent B       │           │
│   │  (Orchestrator)│                          │  (Researcher)  │           │
│   │                │     A2A Task Request      │                │           │
│   │  Builds MCP    │  ┌──────────────────┐    │  Receives MCP  │           │
│   │  packet with   │──│  A2A Envelope    │───▶│  packet,       │           │
│   │  context for   │  │  ┌────────────┐  │    │  processes it, │           │
│   │  Agent B       │  │  │ MCP Packet │  │    │  builds new    │           │
│   │                │  │  │ (context,  │  │    │  MCP packet    │           │
│   │                │  │  │  tools,    │  │    │  with results  │           │
│   │  Receives MCP  │  │  │  state)    │  │    │                │           │
│   │  response      │◀─│  └────────────┘  │────│                │           │
│   │  packet        │  │  A2A Task Result  │    │                │           │
│   │                │  └──────────────────┘    │                │           │
│   └────────────────┘                          └────────────────┘           │
│                                                                             │
│   Flow:                                                                     │
│   1. Agent A constructs an MCP packet (context + tools + state)            │
│   2. MCP packet embedded in A2A task message                               │
│   3. A2A handles discovery, auth, transport, streaming                     │
│   4. Agent B extracts MCP packet, feeds it to its LLM                      │
│   5. Agent B constructs response MCP packet                                │
│   6. Response MCP packet returned via A2A artifact                         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## **3.3 Comparison Table**

| Dimension | A2A | MCP |
|---|---|---|
| **Full Name** | Agent-to-Agent Protocol | Model Context Protocol |
| **Created By** | Google + 50+ partners | Anthropic |
| **Purpose** | Inter-agent communication & coordination | Structured context delivery to LLMs |
| **Layer** | Transport / coordination | Content / data |
| **Analogy** | SMTP (email delivery) | MIME (email content format) |
| **Scope** | Agent ↔ Agent | Application ↔ LLM |
| **Key Primitive** | Task (submit, stream, complete) | Packet (header, context, tools, state) |
| **Discovery** | Agent Card at `/.well-known/agent.json` | Server capabilities via `initialize` handshake |
| **Auth** | OAuth 2.0, API keys, mTLS | Delegated to Host application |
| **Transport** | JSON-RPC 2.0 over HTTPS + SSE | JSON-RPC 2.0 over stdio / HTTP+SSE |
| **Streaming** | `tasks/sendSubscribe` with SSE | Notifications via JSON-RPC |
| **State** | Task state machine (submitted → completed) | Conversation state in `state` section |
| **Memory per unit** | < 50 KB per active chat | 1-10 KB per turn |
| **Multimodal** | Yes (text, images, audio, embeddings) | Yes (text, embeddings, structured data) |
| **Together** | A2A shuttles MCP packets between agents | MCP defines what's inside those packets |

---

## **3.4 When to Use Which**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DECISION MATRIX                                    │
│                                                                      │
│   Scenario                          Use A2A?    Use MCP?             │
│   ────────────────────────────────  ─────────   ─────────           │
│   Single app calling one LLM          No         YES                │
│   Two agents collaborating            YES        YES                │
│   Multi-agent orchestration           YES        YES                │
│   RAG pipeline (app → LLM)            No         YES                │
│   Agent discovery & routing           YES        No                 │
│   Tool definition standardization     Maybe      YES                │
│   Cross-vendor agent interop          YES        No                 │
│   Compliance audit trail              Optional   YES                │
│   Long-running agent tasks            YES        No                 │
│                                                                      │
│   Rule of thumb:                                                     │
│   • Agent ↔ Agent communication = A2A                                │
│   • App/Agent ↔ LLM context = MCP                                   │
│   • Multi-agent with context passing = BOTH                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

# **4. Inter-Model Communication**

---

## **4.1 Multimodal vs Inter-Model — The Distinction**

These terms are often confused. They are orthogonal concepts:

```
┌─────────────────────────────────────────────────────────────────────┐
│              MULTIMODAL vs INTER-MODEL                                │
│                                                                      │
│   MULTIMODAL                        INTER-MODEL                      │
│   ────────────────                  ────────────────                 │
│   One model, many data types        Many models, coordinating        │
│                                                                      │
│   ┌──────────┐                      ┌──────────┐   ┌──────────┐     │
│   │          │ ◀── text             │  Model A │──▶│  Model B │     │
│   │  Single  │ ◀── image            │ (GPT-4)  │   │ (Claude) │     │
│   │  Model   │ ◀── audio            └──────────┘   └──────────┘     │
│   │ (GPT-4V) │ ◀── video                   │               │        │
│   │          │                              ▼               ▼        │
│   └──────────┘                      ┌──────────┐   ┌──────────┐     │
│                                     │  Model C │──▶│  Model D │     │
│   Handles different input           │ (Llama)  │   │ (Gemini) │     │
│   modalities within one model       └──────────┘   └──────────┘     │
│                                                                      │
│                                     Multiple models exchanging       │
│                                     information via messages,        │
│                                     embeddings, or tool calls        │
│                                                                      │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                                                      │
│   You CAN combine both:                                              │
│   Inter-model communication between multimodal models                │
│   e.g., GPT-4V sends image analysis to Claude for report writing     │
└─────────────────────────────────────────────────────────────────────┘
```

| Aspect | Multimodal | Inter-Model |
|---|---|---|
| **Definition** | One model processes multiple data types (text, image, audio) | Multiple models exchange information and coordinate |
| **Example** | GPT-4V analyzing an image and generating text about it | GPT-4 delegates code review to Claude, gets results back |
| **Protocol** | Model-internal (architecture-level) | A2A, MCP, custom APIs |
| **Scaling** | Vertical (bigger model) | Horizontal (more specialized models) |
| **Key Advantage** | Rich understanding of diverse inputs | Specialization, redundancy, best-of-breed per task |

---

## **4.2 Coordination Patterns**

### **Pattern 1: Router-Evaluator**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ROUTER-EVALUATOR PATTERN                             │
│                                                                      │
│                    ┌────────────┐                                     │
│                    │   Router   │                                     │
│   User request ──▶│  (GPT-4)   │                                     │
│                    └──────┬─────┘                                     │
│                           │                                          │
│              ┌────────────┼────────────┐                              │
│              ▼            ▼            ▼                              │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│        │ Worker A │ │ Worker B │ │ Worker C │                        │
│        │ (Claude) │ │ (Gemini) │ │ (Llama)  │                        │
│        └────┬─────┘ └────┬─────┘ └────┬─────┘                        │
│             │            │            │                               │
│             └────────────┼────────────┘                               │
│                          ▼                                            │
│                   ┌────────────┐                                      │
│                   │ Evaluator  │                                      │
│                   │  (GPT-4)   │──▶ Best response                     │
│                   └────────────┘                                      │
│                                                                      │
│   Router classifies the request and dispatches to the best           │
│   worker. Evaluator scores outputs and selects the winner.           │
└─────────────────────────────────────────────────────────────────────┘
```

**Use case:** When different models excel at different tasks (Claude for writing, Gemini for multimodal, Llama for speed).

---

### **Pattern 2: Plan-Act-Reflect**

```
┌─────────────────────────────────────────────────────────────────────┐
│                 PLAN-ACT-REFLECT PATTERN                              │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────┐       │
│   │                                                          │       │
│   │  ┌────────┐     ┌────────┐     ┌──────────┐             │       │
│   │  │  PLAN  │────▶│  ACT   │────▶│ REFLECT  │─────┐       │       │
│   │  │        │     │        │     │          │     │       │       │
│   │  │ Break  │     │Execute │     │ Evaluate │     │       │       │
│   │  │ task   │     │ steps  │     │ results  │     │       │       │
│   │  │ into   │     │ using  │     │ check if │     │       │       │
│   │  │ steps  │     │ tools  │     │ goal met │     │       │       │
│   │  └────────┘     └────────┘     └──────────┘     │       │       │
│   │       ▲                                         │       │       │
│   │       └─────────── revise plan ─────────────────┘       │       │
│   │                                                          │       │
│   │              LOOP UNTIL GOAL ACHIEVED                    │       │
│   └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│   Can use different models per phase:                                │
│   • PLAN:    GPT-4 (strong reasoning)                                │
│   • ACT:     Llama-70B (fast execution via tool calls)               │
│   • REFLECT: Claude (careful evaluation, safety checks)              │
└─────────────────────────────────────────────────────────────────────┘
```

**Use case:** Complex multi-step tasks (research reports, data analysis pipelines, code refactoring).

---

### **Pattern 3: Concurrent Specialists**

```
┌─────────────────────────────────────────────────────────────────────┐
│               CONCURRENT SPECIALISTS PATTERN                         │
│                                                                      │
│                    ┌──────────────┐                                   │
│   User request ──▶│ Orchestrator │                                   │
│                    └──────┬───────┘                                   │
│                           │                                          │
│              ┌────────────┼────────────┐                              │
│              │            │            │                              │
│              ▼            ▼            ▼                              │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│        │ Security │ │ Code     │ │ Docs     │    PARALLEL             │
│        │ Reviewer │ │ Analyst  │ │ Writer   │    EXECUTION            │
│        │ (Claude) │ │ (GPT-4)  │ │ (Gemini) │                        │
│        └────┬─────┘ └────┬─────┘ └────┬─────┘                        │
│             │            │            │                               │
│             └────────────┼────────────┘                               │
│                          ▼                                            │
│                   ┌──────────────┐                                    │
│                   │  Aggregator  │──▶ Combined report                 │
│                   └──────────────┘                                    │
│                                                                      │
│   Multiple specialist agents run in parallel on different            │
│   aspects of the same input. Results are merged.                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Use case:** Code review (security + style + performance checks in parallel), document processing.

---

## **4.3 Message Types in Inter-Model Communication**

| Type | What's Exchanged | Pros | Cons | When to Use |
|---|---|---|---|---|
| **Latent Exchange** | Dense embedding vectors (float32/16) | Ultra-fast (~2ms), compact, bypasses tokenization | Lossy, requires shared embedding space, not human-readable | Similarity search delegation, nearest-neighbor lookups, model chaining in shared latent space |
| **Cognitive Call** | Natural language text | Rich, flexible, human-debuggable | Slow (~200-800ms), token-expensive, ambiguity-prone | Complex reasoning delegation, creative tasks, multi-turn dialogue |
| **Tool Invocation** | JSON-Schema validated function calls | Deterministic, typed, verifiable | Limited to predefined schemas, no free-form reasoning | API calls, database queries, code execution, structured actions |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     CHOOSING A MESSAGE TYPE                                │
│                                                                           │
│   Need speed + efficiency?  ──────────────▶  Latent Exchange              │
│   Need reasoning + nuance?  ──────────────▶  Cognitive Call               │
│   Need reliability + types? ──────────────▶  Tool Invocation              │
│                                                                           │
│   In practice: systems combine all three.                                 │
│   Example pipeline:                                                       │
│                                                                           │
│   1. Latent Exchange: pass embedding to retrieval agent    (2ms)          │
│   2. Cognitive Call: ask summarizer to explain results     (500ms)        │
│   3. Tool Invocation: store summary in database           (50ms)         │
│                                                                           │
│   Total: ~552ms for a retrieve-summarize-store pipeline                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# **5. Practical Architecture**

---

## **5.1 Full-Stack Multi-Agent System**

Here is a realistic production architecture that combines A2A, MCP, and inter-model communication:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION MULTI-AGENT ARCHITECTURE                         │
│                                                                                    │
│  ┌─────────────┐                                                                   │
│  │  Front-End  │  React / Next.js                                                  │
│  │  (User UI)  │  WebSocket for streaming                                          │
│  └──────┬──────┘                                                                   │
│         │ WebSocket / REST                                                         │
│         ▼                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                     ORCHESTRATOR (LangGraph)                              │      │
│  │                                                                          │      │
│  │  • Receives user requests                                                │      │
│  │  • Plans execution via Plan-Act-Reflect                                  │      │
│  │  • Builds MCP packets for each sub-task                                  │      │
│  │  • Routes via A2A to specialized agents                                  │      │
│  │  • Aggregates results                                                    │      │
│  │  • Manages conversation state                                            │      │
│  │                                                                          │      │
│  │  Tech: Python, LangGraph, FastAPI, Redis (task queue)                    │      │
│  └──────────────────────────────┬───────────────────────────────────────────┘      │
│                                 │ A2A Protocol (JSON-RPC 2.0 over HTTPS)           │
│                                 │                                                  │
│              ┌──────────────────┼──────────────────┐                               │
│              │                  │                   │                               │
│              ▼                  ▼                   ▼                               │
│  ┌───────────────────┐ ┌────────────────┐ ┌────────────────────┐                   │
│  │  Research Agent   │ │  Code Agent    │ │  Data Agent        │                   │
│  │                   │ │                │ │                    │                   │
│  │  MCP Server:      │ │  MCP Server:   │ │  MCP Server:       │                   │
│  │  • arXiv search   │ │  • code exec   │ │  • SQL query       │                   │
│  │  • paper parse    │ │  • lint/test   │ │  • chart gen       │                   │
│  │  • citation gen   │ │  • git ops     │ │  • data transform  │                   │
│  │                   │ │                │ │                    │                   │
│  │  LLM: Claude 3.5  │ │  LLM: GPT-4   │ │  LLM: Llama-70B   │                   │
│  │  (best at writing)│ │  (best at code)│ │  (fast + private)  │                   │
│  └────────┬──────────┘ └───────┬────────┘ └────────┬───────────┘                   │
│           │                    │                   │                               │
│           ▼                    ▼                   ▼                               │
│  ┌────────────────────────────────────────────────────────────────┐                │
│  │                    SHARED DATA LAYER                            │                │
│  │                                                                │                │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │                │
│  │  │   pgvector   │  │   Redis      │  │   S3 / MinIO         │ │                │
│  │  │   (vectors)  │  │   (cache,    │  │   (artifacts,        │ │                │
│  │  │              │  │    state)    │  │    documents)        │ │                │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │                │
│  └────────────────────────────────────────────────────────────────┘                │
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐                │
│  │                    EXTERNAL TOOLS                               │                │
│  │                                                                │                │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐ │                │
│  │  │ Tavily    │  │ GitHub    │  │ Snowflake │  │ Sandbox    │ │                │
│  │  │ (search)  │  │ (repos)   │  │ (analytics)│ │ (code exec)│ │                │
│  │  └───────────┘  └───────────┘  └───────────┘  └────────────┘ │                │
│  └────────────────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## **5.2 Memory Budget Table**

For a production deployment serving 1,000 concurrent users:

| Component | Per-User Memory | × 1,000 Users | Notes |
|---|---|---|---|
| A2A task state | ~15 KB | 15 MB | Active task + message history |
| MCP packet (current turn) | ~8 KB | 8 MB | Context + tools + state |
| Conversation memory (short-term) | ~5 KB | 5 MB | Last 5 turns, summarized |
| Conversation memory (long-term) | ~2 KB | 2 MB | User preferences, topic history |
| SSE connection overhead | ~10 KB | 10 MB | Per open streaming connection |
| WebSocket (frontend) | ~8 KB | 8 MB | Browser ↔ orchestrator |
| **Total per user** | **~48 KB** | **48 MB** | |

| Shared Component | Memory | Notes |
|---|---|---|
| Agent Card cache (10 agents) | ~50 KB | Refreshed on TTL |
| Tool schema registry | ~20 KB | All tool definitions |
| pgvector index (1M vectors) | ~4 GB | 1024-dim float32, HNSW index |
| Redis (state cache) | ~500 MB | Task states, session data |
| Model weights (if self-hosted) | ~40-140 GB | Depends on model (Llama-70B ≈ 140 GB FP16) |

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MEMORY BUDGET SUMMARY                              │
│                                                                      │
│   Protocol overhead (1K users):           ~48 MB                     │
│   Shared infrastructure:                  ~5 GB (without models)     │
│   Self-hosted model (optional):           ~40-140 GB                 │
│                                                                      │
│   Key insight: A2A + MCP protocol overhead is negligible             │
│   compared to model weights and vector indexes.                      │
│   The protocols are designed for high-density deployments.           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **5.3 Example: End-to-End Request Flow**

Here's a concrete example tracing a user request through the full stack:

```
┌─────────────────────────────────────────────────────────────────────────┐
│   USER: "Compare LoRA vs QLoRA performance on Llama-2-7B and          │
│          show me a chart"                                              │
│                                                                        │
│   STEP 1: Front-end → Orchestrator (WebSocket)                        │
│   ────────────────────────────────────────────                        │
│   Front-end sends user message to orchestrator.                       │
│                                                                        │
│   STEP 2: Orchestrator PLANS (Plan-Act-Reflect)                       │
│   ─────────────────────────────────────────────                       │
│   Plan: [1] Search papers  [2] Extract data  [3] Generate chart       │
│                                                                        │
│   STEP 3: Orchestrator → Research Agent (A2A)                         │
│   ────────────────────────────────────────────                        │
│   Builds MCP packet:                                                  │
│   • context: user query + system prompt for research                  │
│   • tools: [search_papers, parse_pdf]                                 │
│   • state: conversation history                                       │
│   Wraps in A2A task → POST to Research Agent                          │
│                                                                        │
│   STEP 4: Research Agent processes (Claude 3.5)                       │
│   ─────────────────────────────────────────────                       │
│   • Searches arXiv via MCP tool                                       │
│   • Parses relevant papers                                            │
│   • Returns artifact: structured comparison data (JSON)               │
│                                                                        │
│   STEP 5: Orchestrator → Data Agent (A2A)                             │
│   ────────────────────────────────────────                            │
│   Builds new MCP packet:                                              │
│   • context: comparison data from Step 4                              │
│   • tools: [generate_chart, sql_query]                                │
│   • state: updated with research findings                             │
│   Wraps in A2A task → POST to Data Agent                              │
│                                                                        │
│   STEP 6: Data Agent processes (Llama-70B)                            │
│   ─────────────────────────────────────────                           │
│   • Generates matplotlib chart                                        │
│   • Returns artifact: chart image (PNG) + summary (text)              │
│                                                                        │
│   STEP 7: Orchestrator → Front-end (WebSocket)                        │
│   ─────────────────────────────────────────────                       │
│   Aggregates results, streams final response:                         │
│   • Text: "Here's the comparison between LoRA and QLoRA..."           │
│   • Image: performance chart                                          │
│   • Citations: paper links                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# **6. Common Interview Questions with Strong Answers**

---

## **Q1: "What is A2A and how does it differ from MCP?"**

**Strong Answer:**

"A2A — Agent-to-Agent Protocol — is an open standard for how AI agents discover, authenticate, and communicate with each other. Think of it as HTTP for AI agents. It handles the transport layer: agent discovery via Agent Cards, task submission via JSON-RPC 2.0, streaming via Server-Sent Events, and lifecycle management.

MCP — Model Context Protocol — is a standardized JSON envelope that carries structured context (facts, tools, memory, state) alongside prompts to LLMs. It handles the content layer: what information the model sees.

The key relationship: **MCP is what gets shared; A2A is how it's transported.** In a multi-agent system, A2A carries MCP packets between agents. Like email: SMTP is the delivery protocol (A2A), MIME is the content format (MCP). They're complementary, not competing."

---

## **Q2: "How would you design an inter-agent communication system for a data science team's workflow?"**

**Strong Answer:**

"I'd design a multi-agent system using both A2A and MCP:

1. **Architecture:** An orchestrator agent (LangGraph) receives user requests and routes them via A2A to specialized agents — a Research Agent (paper search, RAG), a Code Agent (execution, testing), and a Data Agent (SQL, visualization).

2. **Communication:** Each agent exposes an Agent Card describing its skills. The orchestrator discovers agents dynamically, submits tasks via A2A's `tasks/send`, and streams progress via SSE.

3. **Context passing:** For each sub-task, I'd build an MCP packet containing the relevant context (RAG chunks, prior results), available tools (search, execute, query), and conversation state. This ensures each agent has exactly the context it needs — no more, no less.

4. **Coordination pattern:** Plan-Act-Reflect — the orchestrator plans the workflow, dispatches actions to workers, and reflects on results before deciding next steps.

5. **Data layer:** pgvector for embeddings, Redis for state caching, S3 for artifacts.

This gives us specialization (best model per task), scalability (agents scale independently), and auditability (every MCP packet is logged)."

---

## **Q3: "What security does A2A provide, and why does it matter?"**

**Strong Answer:**

"A2A is secure by default — a critical design principle. Here's what it provides:

1. **Authentication:** OAuth 2.0 and API key support, declared in the Agent Card's `authentication` section. Agents must authenticate before exchanging tasks.

2. **Transport encryption:** All communication runs over HTTPS/TLS, with optional mTLS for service-mesh deployments.

3. **Capability-based access control:** Agent Cards declare specific skills. A client agent can only invoke capabilities that are advertised — there's no arbitrary code execution.

4. **Task isolation:** Each task has its own ID, state machine, and lifecycle. One agent can't interfere with another agent's tasks.

5. **Audit trail:** Every task submission, status change, and artifact delivery is a JSON-RPC message that can be logged and audited.

This matters because enterprise deployments need to trust that agents won't leak data, escalate privileges, or execute unauthorized actions. A2A makes these guarantees at the protocol level, not as afterthoughts."

---

## **Q4: "How do MCP packets help with compliance and auditability?"**

**Strong Answer:**

"MCP packets are fully serializable JSON envelopes, which means every single turn of LLM interaction has a complete, structured record of:

1. **What context was provided:** Every RAG chunk (with source document IDs and relevance scores), every system prompt, every piece of user input — all in the `context` section.

2. **What tools were available:** The exact tool schemas the model could call, in the `tools` section.

3. **What state was carried:** Conversation history, user preferences, token budgets — in the `state` section.

4. **Routing metadata:** Timestamp, request ID, model used, constraints — in the `header`.

For compliance, this is gold. If a regulator asks 'Why did the model produce this output?', you can replay the exact MCP packet, feed it to the same model version, and reproduce the result. You can prove what the model saw, what tools it had, and what constraints it operated under.

In practice, I'd store MCP packets in an append-only log (like Kafka or a compliance database), enable retention policies, and build dashboards for anomaly detection — flagging packets where the model was given unusually high permissions or sensitive context."

---

## **Q5: "Explain the three inter-model message types and when you'd use each."**

**Strong Answer:**

"The three types are:

1. **Latent Exchange** — passing raw embedding vectors between models. This is the fastest (~2ms) because it bypasses text entirely. I'd use it when one agent needs to hand off a vector representation to another for similarity search or when chaining models that share an embedding space. The tradeoff is it's lossy and not human-readable.

2. **Cognitive Call** — natural language messages between agents, like one agent asking another a question in English. This is the richest but slowest (~200-800ms) due to tokenization overhead. I'd use it for complex reasoning delegation where the nuance of natural language matters — like asking a writing agent to refine a summary.

3. **Tool Invocation** — JSON-Schema validated function calls. Deterministic, typed, and verifiable. I'd use it when one agent needs another to execute a specific action — run a SQL query, call an API, execute code. The schema validation ensures reliability.

In practice, a well-designed system uses all three. For example: latent exchange for fast retrieval (2ms), cognitive call for reasoning (500ms), tool invocation for action execution (50ms). The orchestrator chooses the message type based on the sub-task requirements."

---

## **Q6: "How would you monitor and observe a multi-agent A2A system in production?"**

**Strong Answer:**

"I'd build observability around four pillars:

1. **Metrics (Prometheus + Grafana):**
   - Task latency: P50, P95, P99 per agent and task type
   - Task success/failure rates per agent
   - Concurrent active tasks and SSE connections
   - Token throughput (tokens/second per agent)
   - MCP packet sizes (detect bloated context)

2. **Traces (OpenTelemetry):**
   - Distributed tracing across the full A2A task chain
   - Each A2A task gets a trace ID that propagates through all agents
   - Visualize the complete request flow: orchestrator → agent A → agent B → response
   - Identify bottlenecks (which agent is slowest?)

3. **Logs (structured JSON):**
   - Every MCP packet logged with request ID for compliance replay
   - Task state transitions logged for debugging
   - Tool invocation results logged for error analysis

4. **Alerts:**
   - Task failure rate > 5% → page on-call
   - P95 latency > 10s → investigate slow agent
   - Token cost per request > threshold → check for context explosion
   - Agent Card discovery failures → check network/DNS

The key insight is that A2A's JSON-RPC format makes all of this easy — every message is structured, timestamped, and carries a request ID. You don't need custom instrumentation; the protocol is inherently observable."

---

## **Q7: "What are the tradeoffs between using a single large model vs a multi-agent A2A system?"**

**Strong Answer:**

| Dimension | Single Large Model | Multi-Agent (A2A) |
|---|---|---|
| **Simplicity** | Simple — one API call | Complex — orchestration, multiple services |
| **Latency** | Lower (one hop) | Higher (multiple hops between agents) |
| **Cost** | Can be expensive (GPT-4 for everything) | Can be cheaper (route simple tasks to smaller/cheaper models) |
| **Specialization** | Jack-of-all-trades | Best model for each task |
| **Reliability** | Single point of failure | Redundancy, failover between agents |
| **Scalability** | Vertical (bigger model) | Horizontal (more agents) |
| **Auditability** | Single log | Distributed traces (richer but more complex) |
| **Context window** | Limited by model's window | Each agent gets focused context (via MCP) |

"My recommendation: start simple with a single model. Add multi-agent A2A architecture when you hit specific pain points — context window limits, cost optimization needs, or tasks that clearly benefit from specialization. Don't over-engineer upfront."

---

# **7. Key Takeaways**

---

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          KEY TAKEAWAYS                                        │
│                                                                               │
│  1. A2A is "HTTP for AI agents" — an open, vendor-neutral protocol for       │
│     agent discovery, authentication, task management, and streaming.          │
│     Built on web standards (HTTPS, JSON-RPC 2.0, SSE, OAuth 2.0).           │
│                                                                               │
│  2. MCP is "USB-C for LLM context" — a standardized JSON envelope that       │
│     carries structured context (facts, tools, memory, state) to any LLM.     │
│     Solves the N×M integration problem.                                      │
│                                                                               │
│  3. MCP is WHAT gets shared; A2A is HOW it's transported.                    │
│     They're complementary layers in a complete agent communication stack.     │
│                                                                               │
│  4. A2A Agent Cards (/.well-known/agent.json) enable dynamic discovery —     │
│     agents can find and assess each other's capabilities at runtime.          │
│                                                                               │
│  5. MCP packets are serializable, auditable, and reproducible —              │
│     critical for enterprise compliance and debugging.                         │
│                                                                               │
│  6. Three inter-model message types: latent exchange (fast, lossy),          │
│     cognitive call (rich, slow), tool invocation (structured, typed).         │
│     Production systems combine all three.                                    │
│                                                                               │
│  7. Coordination patterns (Router-Evaluator, Plan-Act-Reflect,               │
│     Concurrent Specialists) determine how agents collaborate.                │
│                                                                               │
│  8. Both protocols are lightweight — <50 KB per active A2A chat,             │
│     1-10 KB per MCP turn. Protocol overhead is negligible vs model weights.  │
│                                                                               │
│  9. Start simple (single model), evolve to multi-agent (A2A + MCP)           │
│     when you hit context limits, cost ceilings, or specialization needs.     │
│                                                                               │
│ 10. The future: A2A + MCP together enable an "Agent Web" where               │
│     specialized AI agents from any vendor can discover, authenticate,        │
│     and collaborate — just like web services do today with REST + JSON.       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

**Quick-Reference Cheat Sheet:**

| Concept | One-Liner |
|---|---|
| **A2A** | Open protocol for agent-to-agent discovery, auth, and task exchange |
| **MCP** | Standardized JSON envelope carrying context, tools, and state to LLMs |
| **Agent Card** | JSON at `/.well-known/agent.json` — an agent's "business card" |
| **MCP Packet** | `{header, context, tools, state}` — everything the model needs in one envelope |
| **Latent Exchange** | Passing embeddings between models (~2ms, lossy) |
| **Cognitive Call** | NL messages between agents (~200-800ms, rich) |
| **Tool Invocation** | JSON-Schema function calls (~50-150ms, typed) |
| **Router-Evaluator** | Route request to best worker, evaluate outputs |
| **Plan-Act-Reflect** | Break down → execute → evaluate → loop |
| **Concurrent Specialists** | Parallel specialist agents, merge results |

---

*Prepared for Rahul Sharma — Data Scientist / ML Engineer Interview Preparation*
