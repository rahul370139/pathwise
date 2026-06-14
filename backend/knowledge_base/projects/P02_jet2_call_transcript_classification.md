# PROJECT: LLM-Based Automated Call Transcript Classification — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Role:** Data Scientist @ Jet2 Travel Technology (June 2022 – Aug 2024)
**Resume Line:** *"Processed 1M+ call transcripts; adapted pretrained LLMs via supervised fine-tuning and applied few-shot prompting at inference, achieving 88% Jaccard recall and enabling real-time decision-making in contact centre operations."*

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview-star)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know](#4-topics-you-must-know)
5. [Interview Questions (20+) with Model Answers](#5-interview-questions-20-with-model-answers)
6. [Potential Red Flags & How to Handle](#6-potential-red-flags--how-to-handle)
7. [Key Takeaways](#7-key-takeaways)

---

# 1. Project Overview (STAR)

---

## 1.1 STAR Summary

| STAR Element | Description |
|---|---|
| **Situation** | Jet2's customer service team handled millions of calls annually. Transcripts piled up with no automated way to extract insights or classify call intent reliably. Manual review was slow, expensive, and inconsistent — agents labelled the same call differently. Leadership needed structured data from unstructured conversations to drive operational decisions. |
| **Task** | Build an end-to-end system to automatically **summarize** and **classify** 1M+ call transcripts into a hierarchical intent taxonomy (primary, secondary, tertiary intent), replacing manual review entirely. |
| **Approach** | (1) Collaborated with product/engineering to define and consolidate hundreds of raw intents into actionable groups. (2) Used Snowflake Cortex built-in functions for transcript summarization and normalization. (3) Implemented few-shot LLM classification with iterative prompt refinement based on SME feedback. (4) Applied teacher-student distillation — 70B teacher → 7B student — for cost-effective production inference. (5) Automated the data pipeline with drift monitoring and operational dashboards. |
| **Result** | **88% Jaccard recall** on multi-intent classification. Dramatically cut manual review time. Integrated into daily operational dashboards enabling real-time decision-making. Earned internal recognition as a key AI innovation at Jet2. |

---

## 1.2 End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  CALL TRANSCRIPT CLASSIFICATION — SYSTEM ARCHITECTURE             │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐     │
│  │  CALL CENTER  │    │  TELEPHONY / IVR  │    │  ASR (Speech-to-Text)      │     │
│  │  Agent + Cust │───►│  System           │───►│  Transcription Engine      │     │
│  │  Live Calls   │    │  (Recording)      │    │  Raw Text Output           │     │
│  └──────────────┘    └──────────────────┘    └────────────┬────────────────┘     │
│                                                           │                      │
│                                                           ▼                      │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                        SNOWFLAKE DATA PLATFORM                         │      │
│  │                                                                        │      │
│  │  ┌──────────────────┐   ┌──────────────────────────────────────┐      │      │
│  │  │  RAW TRANSCRIPTS  │   │  SNOWFLAKE CORTEX (Built-in LLM)    │      │      │
│  │  │  Staging Table    │──►│                                      │      │      │
│  │  │  1M+ records      │   │  CORTEX.SUMMARIZE() → Summaries     │      │      │
│  │  │  Partitioned by   │   │  CORTEX.COMPLETE()  → Classification│      │      │
│  │  │  date             │   │  CORTEX.SENTIMENT() → Sentiment     │      │      │
│  │  └──────────────────┘   └──────────────┬───────────────────────┘      │      │
│  │                                         │                              │      │
│  └─────────────────────────────────────────┼──────────────────────────────┘      │
│                                             │                                     │
│                                             ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                     CLASSIFICATION PIPELINE                            │      │
│  │                                                                        │      │
│  │  Phase 1: TEACHER MODEL (70B LLM)                                     │      │
│  │  ┌────────────────────────────────────────────────────┐               │      │
│  │  │  • Few-shot prompting with curated examples        │               │      │
│  │  │  • Multi-intent classification (primary/sec/tert)  │               │      │
│  │  │  • Generates soft labels + confidence scores       │               │      │
│  │  │  • Iterative prompt refinement with SME feedback   │               │      │
│  │  └──────────────────────┬─────────────────────────────┘               │      │
│  │                          │ Soft labels / distillation data             │      │
│  │                          ▼                                             │      │
│  │  Phase 2: STUDENT MODEL (7B LLM)                                      │      │
│  │  ┌────────────────────────────────────────────────────┐               │      │
│  │  │  • Fine-tuned on teacher's outputs                 │               │      │
│  │  │  • Supervised fine-tuning (SFT) on labeled data    │               │      │
│  │  │  • 10x lower latency, 10x lower cost              │               │      │
│  │  │  • Validated against teacher: 88% Jaccard recall   │               │      │
│  │  └──────────────────────┬─────────────────────────────┘               │      │
│  │                          │                                             │      │
│  └──────────────────────────┼─────────────────────────────────────────────┘      │
│                              │                                                    │
│                              ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐      │
│  │                   PRODUCTION & MONITORING                              │      │
│  │                                                                        │      │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐      │      │
│  │  │  Classified   │  │  Drift Monitor   │  │  Operational       │      │      │
│  │  │  Transcripts  │  │  (Label dist.,   │  │  Dashboards        │      │      │
│  │  │  Table        │  │  embedding drift, │  │  (Tableau/BI)     │      │      │
│  │  │              │  │  confidence drop) │  │                    │      │      │
│  │  └──────────────┘  └──────────────────┘  └────────────────────┘      │      │
│  │                                                                        │      │
│  └────────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Platform** | Snowflake | Data warehouse, storage, compute |
| **Summarization** | Snowflake Cortex (`SUMMARIZE`, `COMPLETE`) | LLM-powered transcript summarization |
| **Teacher LLM** | 70B parameter model (Llama 2 70B / Mixtral class) | High-quality few-shot classification |
| **Student LLM** | 7B parameter model (Llama 2 7B / Mistral 7B class) | Production-grade fast inference |
| **Fine-tuning** | SFT (Supervised Fine-Tuning), LoRA/QLoRA | Efficient model adaptation |
| **Orchestration** | Snowflake Tasks, Stored Procedures | Automated daily pipeline |
| **Monitoring** | Custom drift detection, label distribution tracking | Model quality assurance |
| **Dashboards** | Tableau / Snowflake Dashboards | Operational reporting |
| **Languages** | Python, SQL | Implementation |

---

## 1.4 Scale & Scope

| Dimension | Value |
|---|---|
| Total transcripts processed | **1,000,000+** |
| Daily new transcripts | ~2,000–5,000 |
| Number of raw intent categories (before consolidation) | 200+ |
| Number of consolidated intent groups | ~30–50 (hierarchical: primary → secondary → tertiary) |
| Teacher model size | **70B parameters** |
| Student model size | **7B parameters** (~10x smaller) |
| Classification type | **Multi-label** (primary + secondary + tertiary intent) |
| Evaluation metric | **Jaccard recall = 88%** |
| Latency improvement (student vs teacher) | ~**8–10x faster** |
| Cost reduction | ~**10–15x cheaper** per inference |

---

# 2. Deep Technical Walkthrough

---

## 2.1 Data Pipeline: Raw Transcripts → Classification

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE FLOW                           │
│                                                                      │
│  Step 1: INGESTION                                                   │
│  ┌────────────────────────────────────────────────────┐             │
│  │  Raw call recordings → ASR transcription            │             │
│  │  Output: { call_id, timestamp, agent_id,            │             │
│  │           raw_transcript, duration, metadata }      │             │
│  │  Loaded into Snowflake staging table (daily batch)  │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 2: CLEANING & NORMALIZATION                                    │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  • Remove PII (names, card numbers, booking refs)   │             │
│  │  • Normalize agent/customer speaker labels           │             │
│  │  • Handle ASR errors, filler words, truncation       │             │
│  │  • Filter very short calls (< 30 seconds)            │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 3: SUMMARIZATION (Snowflake Cortex)                            │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  SELECT SNOWFLAKE.CORTEX.SUMMARIZE(                 │             │
│  │    clean_transcript                                  │             │
│  │  ) AS summary                                        │             │
│  │  FROM transcripts_clean;                             │             │
│  │                                                      │             │
│  │  5-page transcript → 2-3 sentence summary            │             │
│  │  Preserves key intent signals, removes noise         │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 4: CLASSIFICATION (LLM — Teacher or Student)                   │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  Few-shot prompt + summary → structured output       │             │
│  │  { primary_intent, secondary_intent,                 │             │
│  │    tertiary_intent, confidence }                     │             │
│  └────────────────────┬───────────────────────────────┘             │
│                        │                                             │
│  Step 5: STORAGE & DOWNSTREAM                                        │
│  ┌────────────────────┴───────────────────────────────┐             │
│  │  Classified results → production table               │             │
│  │  Feeds dashboards, reporting, operational tools      │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why summarize before classifying?**
- Raw transcripts are long (often 2,000–10,000 tokens), exceeding efficient context window limits
- Summarization strips noise (pleasantries, repetitions, ASR errors) while preserving intent signals
- Dramatically reduces inference cost — classifying a 3-sentence summary is ~50x cheaper than a full transcript
- Cortex `SUMMARIZE()` runs natively inside Snowflake, avoiding data egress

---

## 2.2 Intent Taxonomy Design

### The Problem

Raw call logs from agents had **200+ free-text intent labels** — many were duplicates, misspellings, or overlapping categories. Examples:

```
Raw labels (sample):
  "flight change", "change flight", "flight amendment", "amend booking"
  "baggage complaint", "lost luggage", "missing bag", "bag not arrived"
  "refund request", "want money back", "refund", "cancellation refund"
  "hotel query", "accommodation question", "hotel booking issue"
```

### The Solution: Hierarchical Intent Mapping

Collaborated with product managers and contact centre leads to create a **three-level taxonomy**:

```
INTENT TAXONOMY (Hierarchical)
═══════════════════════════════════════════════════════════════
Level 1 (Primary)       Level 2 (Secondary)         Level 3 (Tertiary)
─────────────────       ───────────────────         ──────────────────
BOOKING_CHANGE    ───►  Flight Amendment      ───►  Date Change
                        Hotel Amendment               Passenger Name
                        Package Modification          Upgrade Request

CANCELLATION      ───►  Full Cancellation     ───►  Voluntary
                        Partial Cancellation          Involuntary (airline)
                        Refund Status                 Insurance Claim

COMPLAINT         ───►  Service Quality       ───►  Agent Behavior
                        Baggage Issues                Lost Luggage
                        Delay/Disruption              Flight Delay > 3hrs

INFORMATION       ───►  Booking Enquiry       ───►  Payment Status
                        Travel Requirements           Visa/Docs
                        Loyalty/Rewards               Points Balance

SALES             ───►  New Booking           ───►  Flight Only
                        Add-ons                       Hotel + Flight
                        Upgrades                      Extra Baggage
═══════════════════════════════════════════════════════════════
```

### Design Principles

1. **Mutually Exclusive at Each Level:** A call's primary intent falls into exactly one Level 1 category
2. **Multi-Intent Support:** A single call can have primary = `BOOKING_CHANGE`, secondary = `COMPLAINT` (e.g., customer wants to change a flight AND complains about service)
3. **Actionability:** Each leaf category maps to a concrete operational action or team
4. **SME Validation:** Domain experts reviewed every mapping; ambiguous cases resolved by majority vote
5. **Iterative Refinement:** Taxonomy was versioned; new intents discovered via low-confidence predictions were fed back into taxonomy updates

---

## 2.3 Snowflake Cortex: What It Is and How We Used It

### What is Snowflake Cortex?

Snowflake Cortex is Snowflake's **built-in AI/ML layer** that provides LLM-powered functions directly accessible via SQL. No need to move data outside Snowflake.

```
┌────────────────────────────────────────────────────────────┐
│                  SNOWFLAKE CORTEX OVERVIEW                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LLM Functions (SQL-callable)                         │   │
│  │                                                       │   │
│  │  CORTEX.COMPLETE(model, prompt)                       │   │
│  │    → General text generation, classification           │   │
│  │    → Models: llama3-70b, mistral-large, mixtral-8x7b  │   │
│  │                                                       │   │
│  │  CORTEX.SUMMARIZE(text)                               │   │
│  │    → Abstractive summarization of text                 │   │
│  │                                                       │   │
│  │  CORTEX.SENTIMENT(text)                               │   │
│  │    → Sentiment score [-1, 1]                          │   │
│  │                                                       │   │
│  │  CORTEX.TRANSLATE(text, from, to)                     │   │
│  │    → Machine translation                              │   │
│  │                                                       │   │
│  │  CORTEX.EXTRACT_ANSWER(text, question)                │   │
│  │    → Extractive QA                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Key Benefits:                                               │
│  • Data never leaves Snowflake → compliance & security       │
│  • SQL interface → accessible to analysts, not just ML eng   │
│  • Serverless → no GPU provisioning                          │
│  • Scalable → processes millions of rows via warehouse       │
│  • Pay-per-query → no idle compute costs                     │
└────────────────────────────────────────────────────────────┘
```

### How We Used Cortex

**1. Summarization:**

```sql
-- Summarize each raw transcript using Cortex
SELECT
    call_id,
    SNOWFLAKE.CORTEX.SUMMARIZE(raw_transcript) AS call_summary
FROM call_transcripts_clean
WHERE call_date = CURRENT_DATE() - 1;
```

**2. Classification (Teacher - via COMPLETE):**

```sql
-- Classify summarized transcripts using few-shot prompting
SELECT
    call_id,
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3-70b',
        CONCAT(
            'You are a call intent classifier for a travel company. ',
            'Classify the following call summary into intents.\n\n',
            -- Few-shot examples
            'Example 1:\nSummary: "Customer called to change flight date...',
            'from March 15 to March 22 for 2 passengers."\n',
            'Output: {"primary": "BOOKING_CHANGE", "secondary": ',
            '"Flight Amendment", "tertiary": "Date Change"}\n\n',
            'Example 2:\nSummary: "Customer complained about lost luggage...',
            'on arrival at Malaga and requested compensation."\n',
            'Output: {"primary": "COMPLAINT", "secondary": ',
            '"Baggage Issues", "tertiary": "Lost Luggage"}\n\n',
            -- Actual input
            'Now classify:\nSummary: "', call_summary, '"\nOutput:'
        )
    ) AS classification_json
FROM call_summaries;
```

**Why Snowflake Cortex (not external API)?**
- **Data Governance:** Call transcripts contain PII — data never leaves Snowflake's security perimeter
- **No Infrastructure:** No GPU clusters to manage, no model hosting
- **SQL-Native:** Analysts and engineers both can use it; low barrier to adoption
- **Cost-Effective at Scale:** Serverless billing, scales with warehouse size
- **Compliance:** Jet2 operates under UK/EU data regulations (GDPR); keeping data in-platform simplifies compliance

---

## 2.4 Few-Shot Learning for Classification

### What is Few-Shot Learning?

Few-shot learning provides the LLM with **a small number of labeled examples** (typically 3–10) within the prompt to guide its classification behavior — no weight updates needed.

```
┌────────────────────────────────────────────────────────────────┐
│              LEARNING PARADIGMS COMPARISON                       │
│                                                                  │
│  ZERO-SHOT          FEW-SHOT              FINE-TUNING           │
│  ┌────────────┐    ┌────────────────┐    ┌─────────────────┐   │
│  │ No examples │    │ 3-10 examples  │    │ Thousands of    │   │
│  │ in prompt   │    │ in prompt      │    │ labeled samples │   │
│  │             │    │                │    │ + weight update  │   │
│  │ "Classify   │    │ "Example 1:    │    │                 │   │
│  │  this call" │    │  Summary: X    │    │ Training loop   │   │
│  │             │    │  Intent: Y     │    │ Loss + backprop │   │
│  │ Relies on   │    │ Example 2: ... │    │ Epochs of data  │   │
│  │ pretraining │    │ Now classify:" │    │                 │   │
│  │ knowledge   │    │                │    │ Model weights   │   │
│  │ alone       │    │ In-context     │    │ permanently     │   │
│  │             │    │ learning       │    │ changed         │   │
│  └────────────┘    └────────────────┘    └─────────────────┘   │
│                                                                  │
│  Accuracy:  Low ◄────────────────────────────────────► High     │
│  Cost:      Low ◄────────────────────────────────────► High     │
│  Speed:     Fast ◄───────────────────────────────────► Slow     │
│  Flex:      High ◄───────────────────────────────────► Low      │
└────────────────────────────────────────────────────────────────┘
```

### Our Few-Shot Strategy

**Example Selection Criteria:**
1. **Representative:** Chose 5–8 examples covering each primary intent category
2. **Edge Cases Included:** At least 1–2 ambiguous examples (e.g., a call that is both a complaint and a booking change)
3. **Diversity:** Examples varied in length, tone, complexity
4. **Hard Negatives:** Included examples that look similar but have different intents (e.g., "refund enquiry" vs "refund request")

**Prompt Structure:**

```
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM: You are a call intent classifier for Jet2 Travel.   │
│  Classify each call summary into hierarchical intents.        │
│  Always return valid JSON with primary, secondary, tertiary.  │
│                                                                │
│  INTENT DEFINITIONS:                                           │
│  [List of all valid intent categories with brief descriptions] │
│                                                                │
│  EXAMPLES:                                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Summary: "Customer wants to move flight from Leeds       │ │
│  │ Bradford to Antalya from June 5 to June 12."            │ │
│  │ → {"primary":"BOOKING_CHANGE",                          │ │
│  │    "secondary":"Flight Amendment",                      │ │
│  │    "tertiary":"Date Change",                            │ │
│  │    "confidence":0.95}                                   │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Summary: "Caller asked about luggage allowance for      │ │
│  │ Tenerife trip, also mentioned wanting to add extra bag."│ │
│  │ → {"primary":"INFORMATION",                             │ │
│  │    "secondary":"Travel Requirements",                   │ │
│  │    "tertiary":"Extra Baggage",                          │ │
│  │    "confidence":0.88}                                   │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ [3-6 more examples covering edge cases]                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  INPUT: "<actual call summary>"                                │
│  OUTPUT (JSON):                                                │
└──────────────────────────────────────────────────────────────┘
```

### Iterative Prompt Refinement Process

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│ Draft   │     │ Run on 500   │     │ SME Review   │     │ Measure  │
│ Prompt  │────►│ sample calls │────►│ of errors    │────►│ Jaccard  │
│ v1      │     │              │     │              │     │ recall   │
└─────────┘     └──────────────┘     └──────────────┘     └────┬─────┘
                                                                │
     ┌──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│  Error Analysis:                              │
│  • Which intents are confused most often?     │
│  • Are definitions clear enough?              │
│  • Do examples cover the failure cases?       │
│                                               │
│  Refinement Actions:                          │
│  • Add clarifying notes to intent definitions │
│  • Swap in better few-shot examples           │
│  • Add chain-of-thought reasoning step        │
│  • Adjust output format constraints           │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
               Prompt v2, v3, ... vN
               (repeat until metric plateau)
```

We went through **~8 prompt iterations** before finalizing, improving Jaccard recall from ~72% (v1) to ~90% (teacher, final version).

---

## 2.5 Teacher-Student Distillation: Complete Walkthrough

### Why Distillation?

| Dimension | Teacher (70B) | Student (7B) |
|---|---|---|
| **Parameters** | ~70 billion | ~7 billion |
| **Inference latency** | ~8–15 sec/call | ~0.8–1.5 sec/call |
| **Cost per 1M calls** | $$$$$ (very expensive) | $ (10–15x cheaper) |
| **Quality** | Best-in-class accuracy | ~97% of teacher quality |
| **Production viability** | Too slow/expensive for daily batch | Suitable for daily batch on 5K calls |
| **GPU requirement** | 4x A100 (80GB) or large Cortex warehouse | 1x A100 or medium Cortex warehouse |

**Bottom line:** The 70B teacher gives gold-standard labels but is too expensive to run on every transcript daily. The 7B student replicates the teacher's behavior at a fraction of the cost.

### How Distillation Worked

```
┌──────────────────────────────────────────────────────────────────────┐
│                  TEACHER-STUDENT DISTILLATION PIPELINE                │
│                                                                      │
│  STEP 1: Teacher Generates Training Data                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  70B Teacher + few-shot prompt                                 │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Run on ~50,000 diverse transcripts                            │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Output: (summary, intent_labels, confidence)                  │  │
│  │  Filter: Keep only high-confidence predictions (conf > 0.85)   │  │
│  │  Result: ~40,000 high-quality labeled examples                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 2: Student Training (Supervised Fine-Tuning)                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  7B Base Model (e.g., Mistral-7B / Llama-2-7B)                │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  Fine-tune on teacher's (input, output) pairs                  │  │
│  │  Using SFT (Supervised Fine-Tuning)                            │  │
│  │  Format: input = call summary, output = JSON intent labels     │  │
│  │                                                                │  │
│  │  Training Config:                                              │  │
│  │  • LoRA rank: 16–64, alpha: 32–128                            │  │
│  │  • Learning rate: 1e-4 to 2e-5                                │  │
│  │  • Epochs: 3–5                                                │  │
│  │  • Batch size: 8–16 (gradient accumulation)                   │  │
│  │  • Validation split: 10%                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 3: Validation                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Hold-out set: 5,000 transcripts                               │  │
│  │  Compare: Student predictions vs Teacher predictions            │  │
│  │  Also:    Student predictions vs Human gold labels              │  │
│  │                                                                │  │
│  │  Metrics:                                                      │  │
│  │  • Jaccard recall (student vs teacher): 88%                    │  │
│  │  • Jaccard recall (student vs human):   ~85%                   │  │
│  │  • Per-intent precision & recall                               │  │
│  │  • Confusion matrix across primary intents                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  STEP 4: Deployment                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Student model deployed for daily inference                    │  │
│  │  Teacher model used periodically for:                          │  │
│  │    • Quality audits (sample 500 calls/week)                   │  │
│  │    • Generating labels for new intent categories               │  │
│  │    • Refreshing student training data quarterly                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Distillation Details

**Why SFT on teacher outputs (not classic KD with KL divergence)?**

For generative LLMs, classic KD (matching logit distributions via KL divergence) requires access to the teacher's full output probability distribution over the entire vocabulary at every token position. This is:
- Computationally prohibitive for 70B models (storing/transferring logits for 32K+ vocab tokens per position)
- Often impractical when using Cortex/API-based inference (no logit access)

Instead, we used **sequence-level distillation** — the student learns to replicate the teacher's *text outputs* directly via supervised fine-tuning. This is the approach used by most production LLM distillation systems (Alpaca, Vicuna, etc.).

```
Classic KD (Hinton):        Student matches teacher's PROBABILITY DISTRIBUTION
                            Loss = KL(teacher_logits, student_logits)
                            Requires: full logit access

Sequence-Level Distillation: Student matches teacher's TEXT OUTPUT
                            Loss = Cross-entropy on (input, teacher_output) pairs
                            Requires: only teacher's generated text
                            Used by: Alpaca, Vicuna, Orca, our system
```

### Confidence-Based Filtering

Not all teacher predictions are equally reliable. We filtered training data by confidence:

```python
# Pseudocode for teacher data filtering
teacher_predictions = run_teacher_on_batch(transcripts_50k)

high_quality = [
    pred for pred in teacher_predictions
    if pred['confidence'] > 0.85
    and pred['primary_intent'] in VALID_INTENTS
    and is_valid_json(pred['output'])
]

# ~40,000 of 50,000 passed filtering
# The remaining ~10,000 had low confidence or malformed output
```

---

## 2.6 Multi-Label vs Multi-Class Classification

This project is **multi-label** — a single call can have multiple intents simultaneously.

| Aspect | Multi-Class | Multi-Label |
|---|---|---|
| **Definition** | Each sample belongs to exactly ONE class | Each sample can belong to MULTIPLE classes |
| **Example** | "This email is spam" or "not spam" | "This call is BOOKING_CHANGE + COMPLAINT" |
| **Output** | Single class label / argmax of softmax | Set of labels / independent sigmoid per label |
| **Loss function** | Cross-entropy (softmax) | Binary cross-entropy (sigmoid per class) |
| **Evaluation** | Accuracy, macro/micro F1 | Jaccard index, subset accuracy, hamming loss |
| **Our approach** | — | LLM outputs JSON with all applicable intents |

**Our framing:** The LLM generates structured JSON containing all applicable intents. The prompt explicitly instructs: *"A call may have multiple intents. List ALL that apply."*

```json
// Example: Multi-intent call
{
  "primary_intent": "BOOKING_CHANGE",
  "secondary_intent": "COMPLAINT",
  "tertiary_intent": "Date Change",
  "all_intents": ["BOOKING_CHANGE", "COMPLAINT", "Flight Amendment", "Date Change", "Service Quality"],
  "confidence": 0.91
}
```

---

## 2.7 Jaccard Similarity / Recall: Why This Metric

### The Formula

For multi-label classification where each sample has a **set** of predicted labels and a **set** of true labels:

**Jaccard Similarity (per sample):**

```
                    |Predicted ∩ True|
J(Predicted, True) = ─────────────────
                    |Predicted ∪ True|
```

**Jaccard Recall (our metric):**

```
                         |Predicted ∩ True|
Jaccard Recall = ─────────────────
                         |True|
```

This measures: **of the true intents, what fraction did the model correctly predict?**

### Concrete Example

```
True intents:      {BOOKING_CHANGE, COMPLAINT, Date Change}
Predicted intents: {BOOKING_CHANGE, Date Change, Service Quality}

Jaccard Similarity = |{BOOKING_CHANGE, Date Change}| / |{BOOKING_CHANGE, COMPLAINT, Date Change, Service Quality}|
                   = 2 / 4 = 0.50

Jaccard Recall     = |{BOOKING_CHANGE, Date Change}| / |{BOOKING_CHANGE, COMPLAINT, Date Change}|
                   = 2 / 3 = 0.67

(We missed COMPLAINT; we over-predicted Service Quality)
```

### Why Jaccard Recall (Not Accuracy, Not F1)?

| Metric | Problem for Our Use Case |
|---|---|
| **Exact Match Accuracy** | Too strict — requires ALL intents to match exactly. A call with 4 intents where we get 3 right scores 0%. |
| **Hamming Loss** | Treats all labels equally. Missing "primary intent" penalized same as missing a rare tertiary. |
| **Standard F1** | Designed for single-label. Doesn't naturally handle set-based evaluation. |
| **Jaccard Recall** | Measures coverage — "did we capture the customer's true intents?" Partial credit for partial matches. Operationally, recall matters more: missing an intent means missing an action item. |

**88% Jaccard Recall means:** On average, the student model correctly identifies 88% of the true intents assigned to each call.

---

# 3. Key Metrics & Results

---

## 3.1 Performance Summary

| Metric | Value | Context |
|---|---|---|
| **Jaccard Recall (student vs gold)** | **88%** | Student model on held-out test set against human-verified labels |
| **Jaccard Recall (teacher vs gold)** | ~91% | Teacher model baseline — student retains ~97% of teacher quality |
| **Primary Intent Accuracy** | ~93% | Exact match on just the top-level intent |
| **Processing Time (teacher)** | ~8–15 sec/transcript | Including summarization + classification |
| **Processing Time (student)** | ~0.8–1.5 sec/transcript | ~10x faster |
| **Daily Throughput** | 2,000–5,000 transcripts/day | Fully automated, no manual intervention |
| **Total Processed** | 1,000,000+ | Over the project lifetime |
| **Cost Reduction** | ~10–15x | Student vs teacher inference cost |
| **Manual Review Reduction** | ~85–90% | Humans only review low-confidence predictions |

## 3.2 Business Impact

| Impact Area | Description |
|---|---|
| **Operational Efficiency** | Contact centre managers get real-time intent distribution — can staff teams to match demand (e.g., spike in cancellation calls after disruption). |
| **Agent Training** | Identified common complaint patterns → targeted training for agents on high-frequency issues. |
| **Product Insights** | Discovered previously unknown intent clusters (e.g., post-COVID travel anxiety calls became a detectable category). |
| **Cost Savings** | Eliminated ~4 FTEs worth of manual transcript review labor. |
| **Recognition** | Earned internal recognition as a **key AI innovation** at Jet2. Presented to senior leadership. |
| **Scalability** | System scales linearly — handles seasonal spikes (summer holidays, disruptions) without manual intervention. |

## 3.3 Timeline & Iterations

```
Month 1-2:  Intent taxonomy design + stakeholder alignment
Month 3:    Snowflake Cortex setup, summarization pipeline
Month 4-5:  Few-shot prompt engineering (teacher model), ~8 iterations
Month 6:    Teacher model evaluation, SME validation
Month 7-8:  Student distillation, SFT training, validation
Month 9:    Production deployment, dashboard integration
Month 10+:  Monitoring, drift detection, continuous improvement
```

---

# 4. Topics You Must Know

---

## 4.1 LLMs: Core Architecture

### Transformer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  TRANSFORMER BLOCK (Decoder-Only)            │
│                                                              │
│  Input Tokens → Token Embeddings + Positional Embeddings     │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Multi-Head Self-Attention (Causal/Masked)           │    │
│  │  • Q, K, V projections from input                    │    │
│  │  • Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V        │    │
│  │  • Causal mask: can only attend to past tokens       │    │
│  │  • Multiple heads capture different relationships     │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                             │ + Residual + LayerNorm         │
│                             ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Feed-Forward Network (FFN)                          │    │
│  │  • Linear → GELU/SiLU → Linear                      │    │
│  │  • Expands dimension (e.g., 4096 → 16384 → 4096)    │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                             │ + Residual + LayerNorm         │
│                             ▼                                │
│  Repeat N times (e.g., 32 layers for 7B, 80 for 70B)       │
│       │                                                      │
│       ▼                                                      │
│  Output → Linear → Softmax → Next Token Probability          │
└─────────────────────────────────────────────────────────────┘
```

**Key concepts to articulate in interviews:**
- **Tokenization:** Subword tokenization (BPE/SentencePiece). A word like "unbelievable" might be split into ["un", "believ", "able"]. Affects context window utilization.
- **Context Window:** Maximum number of tokens the model can process at once. Llama 2: 4K tokens; Llama 3: 8K–128K. Our summarization step ensures transcripts fit within context limits.
- **Temperature:** Controls randomness of output. T=0: deterministic (greedy), T=1: standard sampling, T>1: more creative. For classification, we used **T=0 or T=0.1** for consistency.
- **Top-p / Top-k:** Nucleus sampling parameters. For classification, deterministic decoding is preferred.

---

## 4.2 Few-Shot vs Zero-Shot vs Fine-Tuning

| Dimension | Zero-Shot | Few-Shot | Fine-Tuning |
|---|---|---|---|
| **Training data needed** | 0 | 3–10 examples (in prompt) | 1,000–100,000+ labeled samples |
| **Model weights change?** | No | No | Yes |
| **Cost per inference** | Low | Medium (longer prompt) | Low (after training) |
| **Upfront cost** | Zero | Minimal (craft prompt) | High (GPU hours, data labeling) |
| **Adaptability** | Limited to pretraining knowledge | Good — can define custom intents via examples | Best — model specializes to your domain |
| **When to use** | Simple tasks, well-known categories | Custom categories, domain-specific tasks | High-volume production, specialized domains |
| **Our project** | Initial exploration | Teacher model (production prompt) | Student model (SFT on teacher data) |

**Interview-ready framing:**
> "We used few-shot for the teacher because it gave us rapid iteration — we could change intent definitions in minutes by editing the prompt, without retraining. For the student, we fine-tuned because it needed to be fast and cheap at inference; the few-shot examples would add latency and token cost at scale."

---

## 4.3 Knowledge Distillation: Deep Dive

### Classic KD (Hinton et al., 2015)

```
Loss = α · KL(softmax(teacher_logits / T), softmax(student_logits / T)) · T²
     + (1-α) · CE(student_logits, hard_labels)

Where:
  T     = Temperature (typically 2-20, softens distributions)
  α     = Weight balancing soft vs hard loss (typically 0.5-0.9)
  KL    = KL Divergence
  CE    = Cross-Entropy
```

**Temperature's role:**
- High T → softer probability distributions → more information about inter-class relationships
- At T=1: Teacher is confident (e.g., [0.95, 0.03, 0.02]) → student learns little beyond the correct answer
- At T=5: Same teacher outputs become [0.45, 0.30, 0.25] → student learns relative similarities

### Sequence-Level Distillation (Our Approach for LLMs)

```
┌──────────────────────────────────────────────────────────────┐
│  SEQUENCE-LEVEL DISTILLATION (for generative LLMs)            │
│                                                                │
│  1. Teacher generates text output Y for input X                │
│     Y = teacher.generate(X)                                    │
│                                                                │
│  2. Collect dataset: D = {(X₁, Y₁), (X₂, Y₂), ..., (Xₙ, Yₙ)}│
│                                                                │
│  3. Fine-tune student on D using standard causal LM loss:      │
│     Loss = -Σ log P_student(yₜ | y<t, X)                      │
│                                                                │
│  This is exactly SFT — but the "labels" come from the teacher  │
│  rather than humans.                                           │
│                                                                │
│  Advantages over classic KD:                                   │
│  • No need for teacher logits (works with API models)          │
│  • Simpler implementation                                      │
│  • Works for generative tasks (not just classification)        │
│  • Standard SFT tooling (LoRA, DeepSpeed, etc.)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 4.4 Prompt Engineering for Classification

### Key Techniques

| Technique | Description | How We Used It |
|---|---|---|
| **System prompt** | Define the model's role and constraints | "You are a Jet2 call intent classifier..." |
| **Intent definitions** | Explicit descriptions of each category | Listed all valid intents with 1-line definitions |
| **Few-shot examples** | Labeled input-output pairs | 5–8 representative examples per prompt |
| **Chain-of-thought** | Ask model to reason before classifying | "First explain why, then output JSON" (tried but added latency without improving accuracy significantly) |
| **Structured output** | Force JSON format | "Output only valid JSON: {primary, secondary, tertiary}" |
| **Negative instructions** | Tell model what NOT to do | "Do not create new intent categories. Use only the provided list." |
| **Confidence scores** | Ask model to self-assess | "Include confidence 0.0–1.0 for each prediction" |

### Chain-of-Thought for Classification (Tried, Partially Adopted)

```
Prompt with CoT:
"Step 1: Identify the main topic of the call.
 Step 2: Determine if there are secondary issues.
 Step 3: Map to the closest intent category.
 Step 4: Output JSON."

Result: Improved accuracy by ~2% on ambiguous calls,
        but increased latency by ~40% and token usage by ~3x.
        
Decision: Used CoT only for the teacher during training data generation,
          not for the student in production (cost/latency trade-off).
```

---

## 4.5 Snowflake Architecture & Cortex

### Snowflake Architecture Basics

```
┌──────────────────────────────────────────────────────┐
│              SNOWFLAKE ARCHITECTURE                    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  CLOUD SERVICES LAYER                             │ │
│  │  • Authentication, metadata, query optimization   │ │
│  │  • Query parsing, access control                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  COMPUTE LAYER (Virtual Warehouses)               │ │
│  │  • Independent, scalable compute clusters         │ │
│  │  • XS → 4XL, auto-suspend/resume                 │ │
│  │  • Multiple warehouses = no contention            │ │
│  │  • Cortex ML functions run on compute layer       │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  STORAGE LAYER                                    │ │
│  │  • Columnar, compressed, immutable micro-partitions│ │
│  │  • Automatic clustering                           │ │
│  │  • Time travel (up to 90 days)                    │ │
│  │  • S3/Azure Blob/GCS under the hood              │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Key Snowflake concepts for interviews:**
- **Stages:** Landing zones for data ingestion (internal or external like S3)
- **Warehouses:** Independent compute resources; can scale up (bigger) or out (multi-cluster)
- **Separation of Storage and Compute:** Scale independently; pay for each separately
- **Micro-partitions:** Immutable columnar storage units (~50–500MB compressed)
- **Zero-copy Cloning:** Instantly clone tables/databases without copying data
- **Cortex Functions:** SQL-callable LLM/ML functions that run on Snowflake compute

---

## 4.6 Multi-Intent Classification Approaches

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **Binary Relevance** | Train N independent binary classifiers (one per intent) | Simple, parallelizable | Ignores label correlations |
| **Classifier Chains** | Chain of binary classifiers; each uses previous predictions as features | Captures dependencies | Order-dependent |
| **Label Powerset** | Treat each unique label set as a single class | Captures full correlations | Exponential class space |
| **Seq2Seq / LLM** | Generate the set of labels as text | Flexible, handles any combination | Harder to evaluate, may hallucinate |
| **Our approach** | LLM generates JSON with all intents | Natural multi-label via generation | Requires structured output enforcement |

---

## 4.7 Text Summarization: Extractive vs Abstractive

| Aspect | Extractive | Abstractive |
|---|---|---|
| **Method** | Selects important sentences verbatim from text | Generates new sentences that capture the meaning |
| **Output** | Subset of original sentences | Novel phrasing, may paraphrase |
| **Models** | TextRank, BERT-extractive, LexRank | T5, BART, GPT, LLM-based |
| **Faithfulness** | High (exact quotes) | Risk of hallucination |
| **Fluency** | Can be choppy (sentence fragments) | Natural, coherent |
| **Our choice** | — | **Abstractive** (Cortex SUMMARIZE) — produces concise, natural summaries better suited for downstream LLM classification |

---

## 4.8 Data Drift Monitoring for NLP Models

```
┌──────────────────────────────────────────────────────────────┐
│              DRIFT MONITORING STRATEGY                         │
│                                                                │
│  1. LABEL DISTRIBUTION DRIFT                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: distribution of predicted intents over time       │ │
│  │  Metric: Jensen-Shannon divergence vs. baseline month     │ │
│  │  Alert: if JS divergence > threshold (e.g., 0.1)         │ │
│  │                                                           │ │
│  │  Example: If "CANCELLATION" jumps from 15% to 40%,       │ │
│  │  it could be real (disruption) or drift (model broken).   │ │
│  │  Cross-reference with actual cancellation rates.          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  2. CONFIDENCE SCORE MONITORING                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: average confidence score per day/week             │ │
│  │  Alert: if mean confidence drops below threshold          │ │
│  │  Interpretation: model encountering unfamiliar inputs     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  3. INPUT DRIFT (EMBEDDING-BASED)                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Track: embedding distribution of incoming summaries      │ │
│  │  Method: Compute centroid of training data embeddings     │ │
│  │          Measure drift of new data from centroid          │ │
│  │  Alert: if cosine distance to centroid increases          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  4. HUMAN AUDIT LOOP                                           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Weekly: SMEs review random sample of 200 predictions     │ │
│  │  Focus on: low-confidence predictions, new edge cases     │ │
│  │  Feed back: corrections → retrain student quarterly       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  5. TEACHER AUDIT                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Monthly: Run teacher model on 500 random transcripts     │ │
│  │  Compare: student vs teacher agreement rate               │ │
│  │  Alert: if agreement drops below 85%                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 4.9 Model Distillation in Production: Latency vs Accuracy Trade-offs

| Factor | Optimize for Latency | Optimize for Accuracy | Our Choice |
|---|---|---|---|
| **Model size** | Smaller (7B) | Larger (70B) | 7B student for daily, 70B teacher for audits |
| **Quantization** | INT4/INT8 | FP16/FP32 | INT8 for student in production |
| **Batch size** | Large (throughput) | Small (latency per request) | Large batch — daily batch job, latency not critical per-request |
| **Context length** | Shorter summaries | Full transcripts | Short summaries (via Cortex SUMMARIZE) |
| **Few-shot examples** | 0–2 (shorter prompt) | 5–8 (more context) | 0 for student (fine-tuned), 5–8 for teacher |
| **Decoding** | Greedy (T=0) | Beam search / sampling | Greedy for both (classification task) |

---

# 5. Interview Questions (20+) with Model Answers

---

## Q1: "Walk me through this project end to end."

> **Answer:**
> "At Jet2, our contact centre handled millions of calls annually, generating massive volumes of transcripts that needed to be categorized by intent for operational decision-making. Manual review was slow and inconsistent.
>
> I built an end-to-end automated classification system. First, I worked with product managers and domain experts to consolidate over 200 raw intent labels into a clean, hierarchical taxonomy — primary, secondary, and tertiary intents. Think of it as: Level 1 is 'Booking Change,' Level 2 is 'Flight Amendment,' Level 3 is 'Date Change.'
>
> The data pipeline ran inside Snowflake. Raw transcripts were cleaned, then summarized using Snowflake Cortex's built-in SUMMARIZE function — this reduced each transcript from thousands of tokens to a concise 2–3 sentence summary, which dramatically reduced classification cost and improved signal quality.
>
> For classification, I used a two-stage approach. First, a 70B parameter teacher model with few-shot prompting — I provided 5–8 labeled examples in the prompt and iterated through about 8 prompt versions based on SME feedback, going from ~72% to ~91% Jaccard recall.
>
> But the 70B model was too expensive for daily production use on thousands of transcripts. So I applied teacher-student distillation: I ran the teacher on ~50,000 transcripts, filtered for high-confidence predictions, and used those as training data to fine-tune a 7B student model via LoRA. The student achieved 88% Jaccard recall — retaining ~97% of the teacher's quality at ~10x lower cost and latency.
>
> The student model ran daily in production, with monitoring for label distribution drift, confidence degradation, and periodic teacher audits. Results fed into operational dashboards that contact centre managers used to allocate staff, identify training needs, and spot emerging issues in real time."

---

## Q2: "Why did you use few-shot learning instead of fine-tuning for the teacher?"

> **Answer:**
> "There were three main reasons. First, iteration speed — with few-shot, I could change intent definitions, add examples, or restructure the prompt in minutes. Fine-tuning would require hours of GPU training for every change. We went through about 8 major prompt iterations; that would have been 8 training runs.
>
> Second, we were still finalizing the intent taxonomy. Few-shot let us experiment with different category definitions without committing to expensive retraining.
>
> Third, data availability. At the start, we didn't have a large labeled dataset — that's exactly what the teacher was generating. You can't fine-tune without labels, and we needed the few-shot teacher to create those labels.
>
> Once we had stable teacher outputs, we fine-tuned the student for production because: (a) the student needed to be fast and cheap, and (b) the few-shot examples added ~500 tokens to every prompt, which multiplied by thousands of daily calls is significant extra cost."

---

## Q3: "Explain teacher-student distillation in the context of your project."

> **Answer:**
> "Knowledge distillation is about transferring the capabilities of a large, expensive model to a smaller, production-friendly one.
>
> In our case, the teacher was a 70B parameter LLM that we used with few-shot prompting. It was highly accurate — ~91% Jaccard recall — but too expensive and slow for daily production on thousands of transcripts.
>
> For distillation, we ran the teacher on about 50,000 diverse transcripts and collected its outputs — the predicted intents in JSON format along with confidence scores. We filtered for high-confidence predictions above 0.85, giving us ~40,000 high-quality training examples.
>
> Then we took a 7B base model and fine-tuned it using supervised fine-tuning (LoRA) on these teacher-generated input-output pairs. The student learned to map call summaries to intent labels by imitating the teacher's behavior.
>
> This is technically 'sequence-level distillation' — the student matches the teacher's text output, not its internal probability distributions. We didn't need access to the teacher's logits, which was important since we used Cortex for inference.
>
> The result: the 7B student achieved 88% Jaccard recall — close to the teacher's 91% — at roughly 10x lower cost and 10x lower latency."

---

## Q4: "How did you validate the student model against the teacher?"

> **Answer:**
> "We used a multi-layered validation approach.
>
> First, we held out 5,000 transcripts that the teacher had classified but that weren't included in the student's training data. We compared the student's predictions against the teacher's on this set — measuring Jaccard recall, per-intent precision and recall, and generating confusion matrices.
>
> Second, we had a separate set of ~1,000 transcripts with human gold labels from our domain experts. We evaluated both the teacher and student against these gold labels to ensure we weren't just measuring teacher-student agreement but actual correctness.
>
> Third, we did qualitative error analysis. I sampled 200 cases where the student and teacher disagreed, categorized the disagreements, and found that most were on genuinely ambiguous calls where even human annotators had low agreement. The student wasn't making systematic errors — it was struggling in the same areas the teacher struggled.
>
> Finally, we tracked per-intent metrics. The student was strong on high-frequency intents like BOOKING_CHANGE and CANCELLATION (>90% recall) and weaker on rare intents like loyalty/rewards queries (~75%), which makes sense given fewer training examples."

---

## Q5: "Why Jaccard recall as your primary metric?"

> **Answer:**
> "Two reasons: the nature of the task and the business need.
>
> First, this is a multi-label problem — each call can have multiple intents. Traditional accuracy requires an exact match of the entire label set, which is too strict. If a call has 4 intents and we get 3 right, exact match gives 0% credit. Jaccard gives partial credit proportional to overlap.
>
> Second, recall matters more than precision operationally. If we miss an intent, the contact centre might not route a concern to the right team. If we over-predict an intent, the worst case is an unnecessary review — much less costly than missing a complaint or a cancellation request.
>
> Jaccard recall specifically measures: of the true intents, what fraction did we capture? At 88%, we're capturing nearly 9 out of 10 true intents on average. We also tracked Jaccard similarity (which penalizes over-prediction) and per-intent F1 for a complete picture."

---

## Q6: "How did you handle ambiguous intents?"

> **Answer:**
> "Ambiguity was one of the biggest challenges. A customer might start asking about a refund, then pivot to rescheduling — is the primary intent cancellation or booking change?
>
> We handled this at multiple levels. In taxonomy design, we allowed multi-label assignment — a call can legitimately be CANCELLATION + BOOKING_CHANGE. In prompt engineering, we included explicit examples of ambiguous calls and showed the model how to handle them — 'when a call discusses both X and Y, classify as both.'
>
> For cases where even the model was uncertain, we used the confidence score. Predictions below 0.7 confidence were flagged for human review. In practice, about 10–15% of calls fell into this low-confidence bucket, which was manageable for the team.
>
> We also discovered that many 'ambiguous' calls weren't truly ambiguous — they just had multiple intents that we hadn't anticipated. This feedback loop led us to add new intent categories in taxonomy v2 and v3."

---

## Q7: "What if a call has no clear intent?"

> **Answer:**
> "We handled this explicitly in the taxonomy with an 'OTHER/UNCLASSIFIABLE' category, and in the prompt with instructions like: 'If the call does not clearly match any intent, classify as OTHER and explain why in the reasoning field.'
>
> In practice, about 3–5% of calls fell into this bucket. These were typically: very short calls where the customer hung up quickly, wrong-number calls, or highly unusual requests outside our taxonomy.
>
> We monitored the rate of OTHER classifications over time. A spike might indicate that a new type of call was emerging that our taxonomy didn't cover — which happened once when a new booking policy launched and generated a flood of unfamiliar queries. We quickly added the relevant intent category."

---

## Q8: "How did you scale to 1M+ transcripts?"

> **Answer:**
> "Scalability came from three design decisions.
>
> First, everything ran inside Snowflake. Cortex functions process data in parallel across warehouse nodes. We used a Medium to Large warehouse for classification jobs, which could handle thousands of transcripts per hour.
>
> Second, the summarization step was critical for scale. Classifying a 3-sentence summary is ~50x cheaper than classifying a 10,000-token raw transcript. This made the 70B teacher feasible for generating training data and the 7B student very efficient in production.
>
> Third, the pipeline was fully automated via Snowflake Tasks — a scheduled job that ran nightly, picking up new transcripts, summarizing, classifying, and loading results into the production table. No manual intervention needed.
>
> The student model handled the daily volume (2,000–5,000 transcripts) in under 2 hours of batch processing. The historical backfill of 1M+ transcripts was done over a few weeks using the teacher model on a larger warehouse."

---

## Q9: "What would you improve if you could redo this project?"

> **Answer:**
> "Three things. First, I'd invest more in active learning. Instead of randomly sampling transcripts for the teacher, I'd use the student's confidence scores to prioritize uncertain or disagreement cases — this would make the teacher's labeling budget go further.
>
> Second, I'd explore structured fine-tuning approaches like RLHF or DPO with human preferences on the student model. Our SMEs could rank model outputs rather than just provide labels, which could improve quality on ambiguous cases.
>
> Third, I'd build a more sophisticated drift detection system. We tracked label distributions and confidence scores, but I'd add embedding-based drift detection and automated retraining triggers. Currently, retraining was manual and quarterly — I'd make it continuous."

---

## Q10: "How did you handle prompt engineering iterations?"

> **Answer:**
> "Very systematically. Each prompt version was documented with: the exact prompt text, the intent definitions, the few-shot examples, and the evaluation results.
>
> The process was: (1) Draft a prompt version. (2) Run it on a fixed evaluation set of 500 transcripts. (3) Measure Jaccard recall and per-intent metrics. (4) Have 2–3 SMEs review the errors — they'd annotate whether each error was a model mistake or an ambiguous case. (5) Identify patterns — which intents are confused, which examples are misleading. (6) Refine the prompt — update definitions, swap examples, add constraints. (7) Repeat.
>
> Some key discoveries: Adding explicit intent definitions improved recall by ~5%. Including one 'hard negative' example per intent pair that was commonly confused reduced those specific confusions by ~40%. And forcing JSON output format eliminated ~3% of responses that were valid classifications but unparseable."

---

## Q11: "Walk me through the cost analysis — teacher vs student in production."

> **Answer:**
> "Let me break this down concretely. With 5,000 transcripts per day:
>
> **Teacher (70B):** Each classification uses roughly ~800 input tokens (prompt + summary + few-shot examples) and ~100 output tokens. At typical LLM API rates, that's roughly $0.01–0.02 per call. At 5,000 calls/day × 365 days, that's ~$18,000–$36,000/year just for classification, plus summarization costs.
>
> **Student (7B, fine-tuned):** No few-shot examples needed (knowledge baked in), so input is ~200 tokens (just the summary + short instruction) and ~100 output tokens. Cost is roughly $0.001–0.002 per call. Annual: ~$1,800–$3,600.
>
> That's a **10x cost reduction**, and that's before accounting for latency — the student processes calls ~10x faster, which means we can use a smaller Snowflake warehouse and save on compute costs too.
>
> We still used the teacher for monthly audits (~500 calls/month) and quarterly retraining data generation (~5,000 calls/quarter), which added ~$500–$1,000/year. Total system cost was a fraction of what teacher-only would have been."

---

## Q12: "How did you monitor for drift in NLP?"

> **Answer:**
> "We tracked drift at multiple levels.
>
> Label distribution drift: We compared the weekly distribution of predicted intents against a baseline period using Jensen-Shannon divergence. A significant shift could mean either real change (e.g., holiday disruption causing cancellation spike) or model degradation. We cross-referenced with business metrics to distinguish.
>
> Confidence drift: We monitored average prediction confidence over time. A gradual decline suggests the model is encountering inputs it wasn't trained on — perhaps new products, policies, or customer language patterns.
>
> Teacher-student agreement: Monthly, we ran the teacher on 500 random transcripts and checked agreement with the student. If agreement dropped below 85%, it triggered a review.
>
> Human audit: Weekly random sample of 200 predictions reviewed by domain experts. Error rate was tracked over time.
>
> We set up alerts in our monitoring dashboard for each of these. In practice, we detected one significant drift event — after a new booking policy launched, confidence scores dropped and OTHER classifications spiked. We quickly added new intent categories and refreshed the student model."

---

## Q13: "What's the difference between Jaccard similarity and Jaccard recall?"

> **Answer:**
> "Jaccard similarity measures symmetric overlap: |intersection| / |union|. It penalizes both missing labels (false negatives) and extra labels (false positives) equally.
>
> Jaccard recall measures: |intersection| / |true set size|. It only cares about coverage — did we capture the true intents? It doesn't penalize over-prediction.
>
> We chose Jaccard recall because operationally, missing an intent is worse than flagging an extra one. If we miss a complaint, it goes unaddressed. If we over-predict a complaint, the worst case is unnecessary review.
>
> We also tracked Jaccard similarity for a complete picture — our similarity score was ~82%, meaning we had some over-prediction. That was acceptable because the over-predictions were often on related intents."

---

## Q14: "How does Snowflake Cortex compare to using an external LLM API?"

> **Answer:**
> "Three main advantages of Cortex for our use case.
>
> First, data governance. Call transcripts contain PII — names, booking references, sometimes card details. With Cortex, data never leaves Snowflake's security perimeter. Using an external API would require data egress, PII masking, and additional compliance review under GDPR.
>
> Second, operational simplicity. Everything runs as SQL queries. Our data engineers could maintain and extend the pipeline without ML infrastructure expertise. No separate model hosting, no API keys, no rate limiting.
>
> Third, cost model. Cortex uses Snowflake's serverless compute — you pay for what you use, no idle GPU costs. For our batch processing pattern (process 5K transcripts at night, idle during the day), this was significantly cheaper than maintaining dedicated GPU instances.
>
> The trade-off: Cortex offers a fixed set of models and you have less control over fine-tuning. For the student model, we needed more control, so we used separate infrastructure for training but could potentially serve the student through Cortex as well."

---

## Q15: "Tell me about a time the model failed and how you fixed it."

> **Answer:**
> "One notable failure was with 'transfer to another department' calls. About 5% of calls involved a customer being transferred — the LLM was classifying based on the transferring agent's brief notes rather than the customer's actual intent.
>
> The symptom: high confusion between INFORMATION and BOOKING_CHANGE for transferred calls. The summary captured 'Agent explained they'd transfer to booking amendments team' but missed the customer's original request.
>
> The fix was two-fold. First, in the summarization prompt, I added an instruction to focus on the customer's stated needs, not the agent's actions. Second, in the classification prompt, I added an example showing a transferred call and how to classify based on customer intent, not routing action.
>
> This improved the problem category from ~60% accuracy to ~85% accuracy, and overall Jaccard recall improved by about 2 points."

---

## Q16: "How did you handle class imbalance in the intent taxonomy?"

> **Answer:**
> "Our intent distribution was heavily skewed — BOOKING_CHANGE and INFORMATION accounted for ~50% of calls, while categories like LOYALTY or SPECIAL_ASSISTANCE were under 3%.
>
> For the teacher (few-shot), imbalance is less of an issue since the model sees examples of every category in the prompt. But we ensured rare categories had at least one clear example.
>
> For the student (fine-tuning), we used several strategies: (1) Stratified sampling to ensure rare categories were proportionally represented in training data. (2) Over-sampling rare categories by ~2–3x. (3) Evaluating per-intent metrics separately — we didn't let high performance on common intents mask poor performance on rare ones.
>
> Despite this, rare intents still had lower recall (~75% vs ~93% for common intents). We accepted this trade-off but flagged rare-intent predictions for human review when confidence was below 0.8."

---

## Q17: "Why not just fine-tune a single model instead of teacher-student?"

> **Answer:**
> "We could have, but there's a bootstrapping problem. Fine-tuning requires labeled data, and we didn't have it at scale. Manual labeling of 50,000 transcripts with a three-level intent taxonomy would take months of expert time.
>
> The teacher-student approach solved this: the teacher (few-shot, no training data needed) generated labels at scale, and the student used those labels for fine-tuning. It's a form of automated data labeling.
>
> Additionally, the teacher gave us a quality ceiling to aim for and a validation reference. We could measure how close the student got to the teacher, and we could audit by comparing their outputs on the same inputs.
>
> If we had started with a large manually labeled dataset, direct fine-tuning of a 7B model would have been simpler. But given our starting point of zero labels, teacher-student was the right architecture."

---

## Q18: "How do you ensure the model outputs valid JSON?"

> **Answer:**
> "Multiple strategies. In the prompt, I included explicit formatting instructions and examples of valid JSON output. I also added negative instructions: 'Do not include explanatory text outside the JSON block.'
>
> For the student (fine-tuned), the model learned the output format from training data, so JSON validity was high (~98% of responses were parseable).
>
> For the remaining ~2%, we had a fallback parser: regex extraction of key fields, and if that failed, the transcript was flagged for reprocessing with a slightly different prompt (adding 'Output ONLY valid JSON, nothing else' as a prefix).
>
> In more recent systems, I'd use constrained decoding or tool-calling / function-calling APIs that guarantee structured output. But at the time, prompt engineering + post-processing was effective enough."

---

## Q19: "What's your understanding of how tokenization affects this pipeline?"

> **Answer:**
> "Tokenization matters in two places. First, context window utilization — a raw transcript might be 5,000–10,000 tokens depending on the tokenizer. If the context window is 4K (Llama 2), that transcript doesn't even fit. Our summarization step compressed transcripts to ~50–150 tokens, making context windows a non-issue.
>
> Second, cost — LLM inference is priced per token. The summarization step reduced classification input tokens by ~50x, which directly translates to cost savings.
>
> Third, the few-shot examples in the teacher prompt consumed ~500 tokens. For the student (fine-tuned), no examples were needed, saving those tokens per inference.
>
> A subtlety: travel-specific terminology and booking codes (like 'PNR', specific airport codes) might be poorly tokenized by general-purpose tokenizers, potentially splitting them into subword fragments. This was another reason summarization helped — the summary used natural language rather than codes."

---

## Q20: "How would you extend this system to real-time classification?"

> **Answer:**
> "Currently, the system runs as a nightly batch job. For real-time:
>
> First, the student model would need to be served as a low-latency API — deployed on a GPU instance with optimized inference (vLLM, TensorRT-LLM, or similar). At ~1 second per call with the 7B model, real-time is feasible.
>
> Second, the summarization step would need to be moved inline — either a lighter extractive summarization or a very efficient small model, since Cortex SUMMARIZE has some latency.
>
> Third, I'd add a streaming pipeline — instead of batch SQL queries, use Snowflake's Snowpipe or a Kafka/Kinesis stream to process transcripts as they arrive.
>
> Fourth, for true real-time (during the call), we'd need streaming ASR and incremental classification — classifying intent as the conversation progresses. This is a harder problem but valuable for routing calls mid-conversation."

---

## Q21: "How did the dashboard integration work?"

> **Answer:**
> "The classified transcripts landed in a Snowflake production table with columns for call_id, timestamp, agent_id, primary/secondary/tertiary intent, confidence score, and the original summary.
>
> Contact centre managers had Tableau dashboards connected to these tables via Snowflake's native Tableau connector. Dashboards showed: real-time intent distribution, trending intents over time, agent performance by intent category, and anomaly alerts when unusual patterns emerged.
>
> One powerful use case: during flight disruptions, the CANCELLATION intent would spike. Managers could see this in near-real-time and reallocate agents from SALES to CANCELLATION teams before customer wait times increased. Before our system, this reallocation happened hours later based on manual observation."

---

## Q22: "How did you handle PII in the transcripts?"

> **Answer:**
> "PII handling was multi-layered. First, raw transcripts were processed through a PII detection and masking pipeline before summarization — names replaced with [CUSTOMER], card numbers with [CARD_REDACTED], booking references with [BOOKING_REF].
>
> Second, Snowflake's data governance features — row-level security, masking policies, and role-based access control — ensured that only authorized roles could access raw transcripts. The classification pipeline operated on summarized, PII-masked versions.
>
> Third, by keeping everything in Snowflake via Cortex, we avoided data egress — the transcripts never left the platform, which simplified GDPR compliance significantly. This was actually one of the strongest arguments for Cortex over external APIs."

---

# 6. Potential Red Flags & How to Handle

---

## 6.1 "Did you actually train the student model on teacher outputs?"

**The concern:** Interviewers may wonder if you really did distillation or just used the teacher directly.

**How to address:**
> "Yes — the student was fine-tuned via LoRA on approximately 40,000 teacher-generated examples. The training data was (input: call summary, output: teacher's JSON classification). After fine-tuning, the student model ran independently — no teacher in the loop at inference time. We validated the student against both the teacher and human gold labels on held-out data. The student achieved 88% Jaccard recall compared to the teacher's 91%, and we tracked this metric continuously."

**If pressed on details:**
- Mention LoRA rank (16–64), learning rate schedule, number of epochs (3–5)
- Explain that you used sequence-level distillation, not classic KD with logits
- Note that the training loop was standard SFT using libraries like Hugging Face TRL or Axolotl

---

## 6.2 "88% Jaccard recall — is that good enough?"

**The concern:** 88% sounds like 12% of intents are missed.

**How to frame it:**

> "88% Jaccard recall means we correctly capture ~9 out of every 10 true intents per call. For a multi-label task with a 30+ category taxonomy at three levels, this is strong performance. For context:
>
> - Our primary intent accuracy (just the top-level) was ~93%, which is excellent.
> - The remaining errors are concentrated on (a) rare intents with few training examples and (b) genuinely ambiguous calls where even human annotators disagreed.
> - We handle the gap operationally: low-confidence predictions (~10–15% of calls) are flagged for human review, so the effective system accuracy including human-in-the-loop is higher.
> - The alternative — manual classification — had estimated inter-annotator agreement of ~80%, so our system is actually more consistent than the manual process it replaced."

---

## 6.3 "Why few-shot for the teacher instead of fine-tuning a large model?"

**The concern:** Few-shot might seem like a shortcut.

**How to frame it:**

> "Few-shot was a deliberate choice, not a shortcut. We faced a cold-start problem — no labeled data existed at scale. Fine-tuning requires labels; the teacher's job was to create those labels.
>
> Few-shot also gave us rapid iteration — 8 prompt versions in weeks, not months of training. Once we had stable teacher outputs, we invested in fine-tuning the student for production.
>
> The teacher's few-shot performance at ~91% Jaccard recall validated that this approach works. If we'd had a large labeled dataset upfront, direct fine-tuning would have been reasonable — but that dataset didn't exist."

---

## 6.4 "How much of this was your work vs the team?"

**How to frame it:**

> "I was the primary data scientist on this project. Specifically, I owned: the intent taxonomy design (collaborating with product for domain knowledge), the Cortex pipeline setup, prompt engineering for the teacher, the distillation process for the student, evaluation methodology, and drift monitoring.
>
> I worked with a data engineer who helped with Snowflake infrastructure and pipeline orchestration, and product managers who provided domain expertise for intent definitions and SME validation. The dashboard was built by our BI team based on the data I produced."

---

## 6.5 "What if the ASR (speech-to-text) is noisy?"

**How to address:**
> "ASR noise was a real issue — roughly 5–8% word error rate on our transcripts. We handled this in two ways. First, the summarization step acted as a denoiser — the LLM could understand the gist even with some garbled words, and the summary was cleaner than the raw transcript. Second, we explicitly included noisy examples in our few-shot prompts so the classifier learned to handle imperfect input. We did consider adding an explicit error correction step, but the summarization approach was effective enough."

---

# 7. Key Takeaways

---

## 7.1 Technical Takeaways

| # | Takeaway |
|---|---|
| 1 | **Summarize before classifying.** Reducing input length cuts cost, improves signal, and sidesteps context window limits. |
| 2 | **Few-shot is powerful for cold-start.** When you have no labels, few-shot prompting can bootstrap a labeling pipeline. |
| 3 | **Teacher-student distillation works.** A 7B model retained ~97% of a 70B model's quality at 10x lower cost. |
| 4 | **Iterate prompts systematically.** Version prompts, measure with fixed eval sets, use SME feedback — treat prompt engineering like software development. |
| 5 | **Jaccard metrics for multi-label.** Standard accuracy metrics don't work for set-valued predictions. |
| 6 | **Data governance drives architecture.** PII concerns made Snowflake Cortex the right choice over external APIs. |

## 7.2 Behavioral / Process Takeaways

| # | Takeaway |
|---|---|
| 1 | **Cross-functional collaboration is essential.** Taxonomy design required product knowledge; engineering built the pipeline; SMEs validated quality. |
| 2 | **Start simple, iterate.** Zero-shot → few-shot → distillation. Each step built on the previous. |
| 3 | **Monitoring is not optional.** NLP models drift — especially when language, products, and policies change. |
| 4 | **Think production from day one.** The teacher was never meant for production — the student was the plan all along. |

## 7.3 One-Liner Summary for Each Interview Style

| Interview Style | Your One-Liner |
|---|---|
| **Behavioral** | "I led the design and implementation of an LLM-based classification system that automated intent extraction from 1M+ call transcripts, earning internal recognition as a key AI innovation." |
| **Technical** | "I used teacher-student distillation — few-shot prompting on a 70B model to generate training data, then supervised fine-tuning on a 7B model achieving 88% Jaccard recall at 10x lower cost." |
| **System Design** | "End-to-end NLP pipeline inside Snowflake: ASR transcripts → Cortex summarization → LLM classification → operational dashboards, with automated drift monitoring." |
| **Product/Impact** | "Replaced manual transcript review for a contact centre handling millions of calls, enabling real-time staffing decisions and surfacing previously hidden customer intent patterns." |

---

## 7.4 Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE — JET2 PROJECT                 │
│                                                                    │
│  Role:     Data Scientist, Jet2 Travel Technology                  │
│  Period:   June 2022 – Aug 2024                                    │
│  Scale:    1M+ transcripts, 2K-5K/day                              │
│                                                                    │
│  Pipeline: Raw calls → ASR → Snowflake → Cortex SUMMARIZE         │
│            → LLM Classification → Dashboards                       │
│                                                                    │
│  Models:   Teacher = 70B (few-shot) → Student = 7B (SFT/LoRA)     │
│  Metric:   88% Jaccard Recall (multi-label, 3-level taxonomy)      │
│                                                                    │
│  Key Tech: Snowflake Cortex, LLM distillation, few-shot learning,  │
│            multi-label classification, LoRA fine-tuning             │
│                                                                    │
│  Impact:   ~85-90% reduction in manual review                      │
│            10-15x cost reduction (student vs teacher)               │
│            Real-time operational dashboards                         │
│            Internal recognition as key AI innovation                │
│                                                                    │
│  Differentiators:                                                   │
│  • Cold-start → teacher-student pipeline (no labels needed upfront)│
│  • Data governance via Cortex (PII never leaves Snowflake)         │
│  • Systematic prompt engineering (8 iterations, measured)           │
│  • Production-grade with drift monitoring                           │
└──────────────────────────────────────────────────────────────────┘
```

---

*Document prepared for Rahul Sharma — Jet2 Travel Technology project interview preparation.*
*Last updated: February 2026*
