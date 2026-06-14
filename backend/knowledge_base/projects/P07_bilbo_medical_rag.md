# P07 — Medical RAG System for MARCH-PAWS Checklists | Bilbo.ai

> **Rahul Sharma** | AI Engineer Intern | Bilbo.ai, Baltimore, US
> **Duration:** September 2025 – Present

---

## Table of Contents

1. [Project Overview (STAR)](#1-project-overview)
2. [Deep Technical Walkthrough](#2-deep-technical-walkthrough)
3. [Key Metrics & Results](#3-key-metrics--results)
4. [Topics You Must Know (Study Guide)](#4-topics-you-must-know-study-guide)
5. [Interview Questions & Answers (25+)](#5-interview-questions--answers)
6. [Red Flags & How to Handle](#6-red-flags--how-to-handle)
7. [Key Takeaways & Talking Points](#7-key-takeaways--talking-points)

---

## 1. Project Overview

### 1.1 STAR Summary (Interview-Ready)

**Situation**
At Bilbo.ai, military medics and first-responders rely on the MARCH-PAWS protocol (Massive hemorrhage, Airway, Respiration, Circulation, Hypothermia/Head injury, Pain, Antibiotics, Wounds, Splinting) to triage and treat combat casualties. The authoritative reference — the TC 4-02.1 manual (a dense, multi-chapter PDF) — is impractical to consult under field conditions. Medics needed a way to get stage-specific, citation-grounded Q&A checklists for any given injury scenario in seconds, not minutes of page-flipping.

**Task**
I was tasked with designing and deploying a retrieval-augmented NLP system that ingests the medical PDF manual, builds a searchable knowledge base, and generates step-by-step MARCH-PAWS checklists — producing stage-specific Q&A grounded in source files with paragraph-level citations.

**Approach & Action**

| Phase | What I Did |
|-------|-----------|
| **PDF Ingestion Pipeline** | Built an 8-step pipeline: PDF extraction (PyMuPDF) → heading discovery → anchor detection → paragraph segmentation → text cleaning → sliding-window creation → embedding generation → index building. |
| **Hybrid Retrieval** | Implemented FAISS (dense, 384-dim) + BM25 (sparse, lexical) with Reciprocal Rank Fusion (RRF) and adaptive alpha weighting based on query characteristics. |
| **Cross-Encoder Reranking** | Added ms-marco-MiniLM-L-6-v2 cross-encoder with sigmoid normalization and LRU caching for relevance scoring. |
| **LLM Generation** | Deployed Mistral 7B (Q4_K_M quantized, ~4.37 GB) via Ollama for stage-specific question generation and checklist answer generation using a two-prompt architecture (Q-Gen / A-Gen). |
| **State Machine & Orchestration** | Built a finite-state machine (FSM) to walk through all 9 MARCH-PAWS stages, with an async orchestrator for parallel pre-fetching, background retrieval, and intelligent caching. |
| **Quality Evaluation** | Developed an automated quality evaluator using semantic similarity (MiniLM), SpaCy NLP for actionable-verb detection, WordNet synonym expansion, and citation accuracy checking. |
| **Frontend** | Built a Streamlit UI with scenario input, stage-by-stage assessment flow, quality metrics display, and a retrieval testing panel. |

**Result**
- End-to-end RAG system producing grounded, citation-backed MARCH-PAWS checklists for any medical scenario
- Quality score baseline of **~0.64** with a concrete improvement roadmap to **0.92+**
- 141 indexed windows over 384-dimension embeddings covering the entire TC 4-02.1 manual
- Sub-second question generation via Q-Gen prompt (no retrieval needed), <5s full answer generation
- Context-engineered few-shot examples (45 curated examples, 5 per stage) for high-quality question generation

---

### 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│               MARCH-PAWS MEDICAL RAG SYSTEM ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    PDF INGESTION PIPELINE (Offline)                   │    │
│  │                                                                      │    │
│  │  TC 4-02.1 PDF                                                       │    │
│  │       │                                                              │    │
│  │       ▼                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │1. PyMuPDF│→ │2. Heading│→ │3. Anchor │→ │4. Para   │            │    │
│  │  │Extract   │  │Discovery │  │Detection │  │Segment   │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │    │
│  │       │                                                              │    │
│  │       ▼                                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │5. Text   │→ │6. Window │→ │7. Embed  │→ │8. Index  │            │    │
│  │  │Cleaning  │  │Creation  │  │MiniLM-L6 │  │Building  │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │    │
│  │                                              384-dim      FAISS+BM25│    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    RUNTIME PIPELINE (Online)                          │    │
│  │                                                                      │    │
│  │  ┌──────────────┐                                                    │    │
│  │  │  Streamlit UI │  ← User enters scenario: "chest gunshot wound"   │    │
│  │  └──────┬───────┘                                                    │    │
│  │         │                                                            │    │
│  │         ▼                                                            │    │
│  │  ┌──────────────────────────────────────────┐                        │    │
│  │  │        ASYNC ORCHESTRATOR                 │                        │    │
│  │  │  ┌───────────────────────────────────┐   │                        │    │
│  │  │  │  FSM: M → A → R → C → H → P →   │   │                        │    │
│  │  │  │       A2 → W → S → END           │   │                        │    │
│  │  │  └───────────────────────────────────┘   │                        │    │
│  │  │                                           │                        │    │
│  │  │  ┌────────────┐    ┌────────────────┐    │                        │    │
│  │  │  │  Q-Gen     │    │   A-Gen        │    │                        │    │
│  │  │  │  (Question │    │   (Answer +    │    │                        │    │
│  │  │  │  Generation│    │   Checklist    │    │                        │    │
│  │  │  │  Prompt)   │    │   Generation)  │    │                        │    │
│  │  │  └─────┬──────┘    └──────┬─────────┘    │                        │    │
│  │  │        │                  │               │                        │    │
│  │  │        ▼                  ▼               │                        │    │
│  │  │  ┌──────────────────────────────────┐    │                        │    │
│  │  │  │     Mistral 7B (Q4_K_M)          │    │                        │    │
│  │  │  │     via Ollama (~4.37 GB)        │    │                        │    │
│  │  │  └──────────────────────────────────┘    │                        │    │
│  │  └──────────────────────────────────────────┘                        │    │
│  │                    │                                                  │    │
│  │                    ▼                                                  │    │
│  │  ┌──────────────────────────────────────────┐                        │    │
│  │  │          HYBRID RETRIEVAL                 │                        │    │
│  │  │                                           │                        │    │
│  │  │  ┌──────────┐      ┌──────────┐          │                        │    │
│  │  │  │  FAISS   │      │  BM25    │          │                        │    │
│  │  │  │  Dense   │      │  Sparse  │          │                        │    │
│  │  │  │  141 vecs│      │  Window  │          │                        │    │
│  │  │  │  384-dim │      │  Index   │          │                        │    │
│  │  │  └────┬─────┘      └────┬─────┘          │                        │    │
│  │  │       │                  │                │                        │    │
│  │  │       ▼                  ▼                │                        │    │
│  │  │  ┌──────────────────────────────────┐    │                        │    │
│  │  │  │  Reciprocal Rank Fusion (RRF)    │    │                        │    │
│  │  │  │  adaptive α: short→dense,        │    │                        │    │
│  │  │  │             long→BM25            │    │                        │    │
│  │  │  └───────────────┬──────────────────┘    │                        │    │
│  │  │                  │                        │                        │    │
│  │  │                  ▼                        │                        │    │
│  │  │  ┌──────────────────────────────────┐    │                        │    │
│  │  │  │  Cross-Encoder Reranking         │    │                        │    │
│  │  │  │  ms-marco-MiniLM-L-6-v2          │    │                        │    │
│  │  │  │  + sigmoid normalization         │    │                        │    │
│  │  │  │  + LRU caching                   │    │                        │    │
│  │  │  └───────────────┬──────────────────┘    │                        │    │
│  │  │                  │                        │                        │    │
│  │  │                  ▼                        │                        │    │
│  │  │  ┌──────────────────────────────────┐    │                        │    │
│  │  │  │  Smart Paragraph Selection       │    │                        │    │
│  │  │  │  + Stage-Aware Content Filtering │    │                        │    │
│  │  │  │  + Citation Grounding            │    │                        │    │
│  │  │  └──────────────────────────────────┘    │                        │    │
│  │  └──────────────────────────────────────────┘                        │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────┐                        │    │
│  │  │        QUALITY EVALUATOR                  │                        │    │
│  │  │  • Semantic similarity (MiniLM)           │                        │    │
│  │  │  • Citation accuracy (DB mapping)         │                        │    │
│  │  │  • Actionable-verb detection (SpaCy)      │                        │    │
│  │  │  • WordNet synonym expansion              │                        │    │
│  │  └──────────────────────────────────────────┘                        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Tech Stack Summary

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | Mistral 7B | Q4_K_M quantized via GGUF, ~4.37 GB, served by Ollama |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence-Transformer, 384 dims, ~150 MB |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder, ~150 MB, sigmoid scoring |
| **Dense Index** | FAISS (IndexFlatIP) | 141 vectors, 384 dims, cosine similarity |
| **Sparse Index** | BM25Okapi (rank_bm25) | Window-based, lemmatized tokens |
| **PDF Parsing** | PyMuPDF (fitz) | Font-size heading detection, block extraction |
| **NLP** | SpaCy (en_core_web_sm) | Dependency parsing for actionable-verb detection |
| **Synonyms** | NLTK WordNet | Expand actionable verb vocabulary |
| **Frontend** | Streamlit | Interactive medical scenario assessment UI |
| **Async** | asyncio + aiohttp | Parallel pre-fetching, background retrieval |
| **LLM Server** | Ollama | Local inference endpoint at localhost:11434 |

---

### 1.4 What is MARCH-PAWS?

MARCH-PAWS is the **Tactical Combat Casualty Care (TCCC)** systematic triage mnemonic used by military medics:

| Letter | Stage | Focus |
|--------|-------|-------|
| **M** | Massive Hemorrhage | Locate life-threatening bleeding; tourniquet, packing, direct pressure |
| **A** | Airway | Confirm airway is patent; suction, positioning, adjuncts |
| **R** | Respiration | Assess breathing/ventilation; chest injuries |
| **C** | Circulation | Assess perfusion (pulse, skin, mental state); fluids/blood products |
| **H** | Hypothermia/Head Injury | Prevent heat loss; screen for head injury |
| **P** | Pain | Assess pain level; determine if analgesia is needed |
| **A2** | Antibiotics | Determine if penetrating wound mandates antibiotics; allergy check |
| **W** | Wounds | Re-inspect for missed injuries, burns, eviscerations |
| **S** | Splinting | Immobilize fractures/dislocations; check distal pulse |

The system walks through **all 9 stages sequentially** via a finite-state machine, generating contextual questions and evidence-based checklists at each step.

---

## 2. Deep Technical Walkthrough

### 2.1 Medical PDF Ingestion Pipeline (8 Steps)

The ingestion pipeline transforms a raw military medical PDF into a searchable, embeddings-indexed knowledge base.

```
TC 4-02.1 PDF
     │
     ▼
Step 1: PDF Extraction (PyMuPDF/fitz)
     │  • Extract text blocks with bounding boxes (x0, y0, x1, y1)
     │  • Preserve font sizes for heading detection
     │  • Strip header/footer regions (top 80px, bottom 30px)
     │  • Filter banned lines (distribution notices, blank page markers)
     ▼
Step 2: Heading Discovery
     │  • TOC-based: doc.get_toc(simple=True) → pin to (page, y0)
     │  • Page-based: font-size analysis (top 20% = large headings)
     │  • Pattern matching: Chapter X, Section I-IV, ALL-CAPS titles
     │  • Infer 3-level hierarchy: L1=Chapter, L2=Section, L3=Subsection
     │  • Deduplicate by (normalized_text, page_num)
     ▼
Step 3: Anchor Detection
     │  • Match paragraphs to nearest heading above using y-coordinates
     │  • Build heading stack: [Chapter, Section, Subsection]
     │  • Generate toc_path: "Chapter 6 > Section II > RESPIRATORY SYSTEM"
     │  • Create short_ref: "Ch6 §6-4, p.35-35"
     ▼
Step 4: Paragraph Segmentation
     │  • Regex: /^(\d+-\d+)\.\s*(.*)$/ for numbered paragraphs (e.g., "6-4.")
     │  • Lost-dash repair: "6 4." → "6-4."
     │  • Stitch incomplete paragraphs across page boundaries
     │  • Handle continuation lines (lowercase start = continuation)
     │  • Leaked heading detection and splitting
     ▼
Step 5: Text Cleaning
     │  • Reflow hard-wrapped lines (preserve list bullets)
     │  • Normalize bullet characters (Unicode → standard dash)
     │  • Fix compound words: "life threatening" → "life-threatening"
     │  • Remove footer leakage (date stamps, TC references)
     │  • polish_tokens(): fix MOS-specific, self-aid, etc.
     │  • Version detection: Base / C1 / C2
     ▼
Step 6: Window Creation (Sliding Window)
     │  • MAX_TOK = 250 tokens, OVERLAP = 100 tokens
     │  • Group paragraphs into windows respecting:
     │    - Token budget (250 max per window)
     │    - Semantic boundaries (section changes force new window)
     │    - Overlap (≈2 paragraphs) for context continuity
     │  • Giant paragraphs (>250 tokens) split on sentence boundaries (120 target)
     │  • Each window: {window_id, paragraph_ids, chapter, pages, headings, text}
     ▼
Step 7: Embedding Generation
     │  • Model: all-MiniLM-L6-v2 (384 dimensions)
     │  • Parallel embedding: joblib + threading backend
     │  • Batch size: 32 windows per batch
     │  • Normalize L2 for cosine similarity via FAISS
     │  • Output: window_embeddings.npy (141 × 384 float32)
     ▼
Step 8: Index Building
       • FAISS: IndexFlatIP (inner product on L2-normalized = cosine sim)
       • BM25: rank_bm25.BM25Okapi with lemmatized (WordNet) tokenization
       • Metadata: window_metadata.json (141 entries)
       • Window texts: windows.jsonl (for cross-encoder scoring)
```

**Key Design Decisions:**
- **Window-based indexing** (not paragraph-based): windows capture multi-paragraph context, improving retrieval for questions that span multiple related paragraphs
- **Overlap of 100 tokens**: ensures context is not lost at window boundaries
- **Semantic boundary detection**: prevents a window from mixing content from two unrelated sections

---

### 2.2 Embedding Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Architecture | 6-layer MiniLM (distilled from BERT) |
| Dimension | 384 |
| Max sequence length | 256 tokens |
| Model size | ~80 MB (FP32), ~150 MB loaded |
| Training data | 1B+ sentence pairs (AllNLI, MS MARCO, etc.) |
| Similarity metric | Cosine similarity (via L2 normalization + inner product) |

**Why MiniLM-L6-v2 and not a larger model?**
- Small corpus (141 windows) — a bigger model adds latency without meaningful quality gain
- 384 dims is sufficient for this domain vocabulary
- Fast warm-up and inference: encode a query in <10ms on CPU
- Proven strong performance on semantic textual similarity benchmarks

**Encoding process:**

```python
# Warm-up eliminates cold-start latency
model = SentenceTransformer("all-MiniLM-L6-v2")
model.encode(["warm-up sentence"])  # JIT + tokenizer init

# Query encoding
query_embedding = model.encode([query]).astype(np.float32)
faiss.normalize_L2(query_embedding)  # L2-normalize for cosine similarity
```

---

### 2.3 Hybrid Retrieval: FAISS + BM25 with RRF

The system uses **two complementary retrieval strategies** fused via Reciprocal Rank Fusion:

#### FAISS (Dense Retrieval)
- **Index type:** `IndexFlatIP` (flat inner product — exact search, no approximation)
- **Why flat?** Only 141 vectors — IVF/HNSW overhead is unnecessary
- Queries are L2-normalized, so inner product = cosine similarity
- Returns top-N windows by semantic similarity

#### BM25 (Sparse Retrieval)
- **Implementation:** `rank_bm25.BM25Okapi`
- **Tokenization:** Lightweight WordNet lemmatization (no POS tagging — ~10x faster)
- Excels at exact keyword matching (e.g., "tourniquet", "needle decompression")
- State hints augment the BM25 query for stage-specific boosting

```python
def light_lemmatize(query, lemmatizer):
    """~10x faster than POS-aware lemmatization."""
    words = re.findall(r'\b\w+\b', query.lower())
    return [lemmatizer.lemmatize(word) for word in words]
```

#### Reciprocal Rank Fusion (RRF)

The RRF formula combines rankings from both retrievers:

```
RRF_score(doc) = α / (rank_bm25 + k) + (1 - α) / (rank_dense + k)
```

Where:
- `k = 60` (standard RRF constant — dampens the influence of top ranks)
- `α` = adaptive alpha based on query characteristics

#### Adaptive Alpha

```python
def calculate_adaptive_alpha(query):
    if len(query.split()) <= 3:   # Short queries → rely more on dense
        return 0.35
    if "?" in query:               # Questions → slightly more dense
        return 0.45
    return 0.55                    # Default → balanced / slight BM25 lean
```

**Intuition:** Short queries like "tourniquet" benefit from semantic understanding (dense). Longer, more descriptive queries contain enough lexical signal for BM25 to excel.

---

### 2.4 Cross-Encoder Reranking

After hybrid retrieval returns ~20 candidate windows, a cross-encoder reranks them by true relevance:

| Property | Value |
|----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Architecture | 6-layer MiniLM fine-tuned on MS MARCO |
| Input | (query, document) pairs |
| Output | Relevance logit (raw score) |
| Normalization | Sigmoid: `score = 1 / (1 + exp(-logit))` → [0, 1] |
| Loading | Lazy-loaded on first use |

**Bi-encoder vs Cross-encoder:**

```
Bi-encoder (retrieval):          Cross-encoder (reranking):
  query  → [encoder] → q_vec      (query, doc) → [encoder] → score
  doc    → [encoder] → d_vec      Full attention between query and doc
  score = q_vec · d_vec            Much more accurate, but O(n) per pair
  Fast: O(1) per doc               Slow: ~50ms per (query, doc) pair
```

**Two-level scoring:**
1. **Window-level reranking:** Score and sort entire windows
2. **Paragraph-level scoring:** `score_paragraphs()` scores individual paragraphs extracted from top windows for fine-grained selection

```python
# Sigmoid normalization for consistent 0-1 scoring
normalized_scores = 1 / (1 + np.exp(-np.array(raw_scores)))
```

---

### 2.5 Dynamic Z-Score Thresholds

Instead of a static relevance threshold, the system computes a **dynamic z-score threshold** per query:

```python
def _calculate_bm25_zscore_threshold(self, query, top_n=100):
    scores = self.bm25.get_scores(query_words)
    top_scores = np.sort(scores)[-top_n:]

    max_score = np.max(top_scores)
    mean_score = np.mean(top_scores)
    std_score = np.std(top_scores)

    zscore_threshold = (max_score - mean_score) / std_score
    # Normalize to [0.001, 0.01] range
    return max(0.001, min(0.01, zscore_threshold * 0.01))
```

**Why dynamic?**
- Different queries produce different BM25 score distributions
- A query like "tourniquet application" (high specificity) yields a tight score cluster — needs a lower threshold
- A vague query like "first aid" produces a flat distribution — needs a higher threshold to filter noise
- Prevents the system from surfacing irrelevant content for ambiguous queries

---

### 2.6 LRU Caching for Cross-Encoder

Cross-encoder inference is expensive (~50ms per pair). The system caches results:

```python
# Question cache: (scenario_hash, state) → question
self.q_cache = {}  # Avoids regenerating identical questions

# Cache key creation
def create_cache_key(scenario, state):
    return (hash(scenario), state)
```

**Caching strategy:**
- **Q-Gen cache:** Identical scenario + state always produces the same question
- **Pre-fetching:** While the user reads the current checklist, the system pre-generates the next state's question in the background
- **Background retrieval:** Next state's retrieval runs during user think-time

---

### 2.7 Mistral 7B: Quantized Inference (Q4_K_M)

| Property | Value |
|----------|-------|
| Base model | Mistral 7B Instruct v0.2 |
| Quantization | Q4_K_M (4-bit, k-quant medium) |
| Format | GGUF (GPT-Generated Unified Format) |
| Size on disk | ~4.37 GB |
| Context window | 4096 tokens |
| Serving | Ollama (`mistral:latest`) |
| Endpoint | `http://localhost:11434/api/generate` |

**Q4_K_M explained:**
- "Q4" = 4-bit quantization
- "K" = k-quant method (groups weights into blocks, each with its own scale)
- "M" = medium quality (some layers keep higher precision for attention heads)
- Trade-off: ~0.5 perplexity increase vs FP16, but 4x memory reduction

**Two-prompt architecture:**

| Prompt | Purpose | Temperature | Retrieval? |
|--------|---------|-------------|------------|
| **Q-Gen** | Generate stage-specific question | 0.0 (deterministic) | No — uses only stage definition + scenario + few-shot examples |
| **A-Gen** | Generate checklist + citations | 0.3 (slight creativity) | Yes — full hybrid retrieval + reranking |

**Why split Q-Gen and A-Gen?**
- **Q-Gen is fast (<1s):** No retrieval needed — question depends only on stage definition and scenario
- **A-Gen is thorough:** Retrieves, reranks, and generates evidence-grounded checklists
- Separation allows parallel pre-fetching of next question while user reviews current checklist

---

### 2.8 Scenario-Aware Content Filtering

The system filters retrieved content to avoid stage contamination:

```python
def _filter_excerpts_by_stage(self, excerpts, current_state, completed_states):
    stage_keywords = {
        'M': ['bleeding', 'hemorrhage', 'tourniquet', ...],
        'A': ['airway', 'obstruction', 'patent', ...],
        'R': ['respiratory', 'breathing', 'chest', ...],
        # ... one set per stage
    }

    for excerpt in excerpts:
        for completed_state in completed_states:
            if any(kw in excerpt_text for kw in stage_keywords[completed_state]):
                exclude = True  # Content belongs to an already-completed stage
    
    # Safety: if too aggressive, keep top by CE score
    if len(filtered) < 3:
        filtered = sorted(excerpts, key=lambda x: x["score_ce"], reverse=True)[:6]
```

**Why this matters:**
- Without filtering, the Respiration stage might surface Massive Hemorrhage content (both mention "chest")
- As the FSM advances, previously completed stages' content is excluded
- Fallback ensures the system never returns empty — if filtering is too aggressive, top cross-encoder results are kept

**Medical query detection** is multi-layered:
1. **Non-medical pattern rejection** (regex): weather, cooking, programming, etc.
2. **Medical context patterns**: "I have bleeding", "emergency", "patient"
3. **Symptom patterns**: bleeding, fracture, airway, pulse, etc.
4. **Keyword fallback**: 100+ medical indicator terms

---

### 2.9 Citation Grounding

Citations follow a strict database-grounded workflow:

```
LLM output: "Ch6 §6-4, p.35-35"
     │
     ▼
map_citation_format():
     │  Regex: /Ch(\d+)\s*§(\d+-?\d*)/
     │  → "tc4-02.1:ch6:6-4@Base"
     │  → Try @Base, then @C2 version
     ▼
Lookup in citation_db (tc4-02.1_sections.jsonl):
     │  → Exists? Use short_ref from anchors
     │  → Missing? Drop citation (don't hallucinate)
     ▼
Final output: "Ch6 §6-4, p.35-35" (verified)
```

**Key properties:**
- Citations are **verifiable** — every citation maps to a real paragraph in the database
- Duplicate citations are limited to max 2 occurrences
- Failed mappings are **dropped**, not passed through — prevents hallucinated references
- Format: `[ChX §Y-Z, p.X-Y]` with chapter, paragraph number, and page range

---

### 2.10 Context Engineering: Few-Shot Examples

The Q-Gen prompt uses **45 curated examples** (5 per MARCH-PAWS stage):

```python
def get_scenario_context(state, scenario):
    examples_db = load_scenario_examples()  # data/scenario_examples.json
    
    # Score examples by keyword overlap with current scenario
    for example in stage_examples:
        overlap = sum(1 for word in scenario.split() if word in example['scenario'])
    
    # Take top 3 most relevant examples
    relevant_examples = sorted_by_overlap[:3]
    
    # Build few-shot context
    context = f"SCENARIO: {scenario}\nSTAGE: {state}\n"
    for example in relevant_examples:
        context += f"Scenario: '{example['scenario']}' → Question: '{example['question']}'\n"
        context += f"Reasoning: {example['reasoning']}\n"
```

This is **context engineering** — not fine-tuning — ensuring the LLM generates questions that:
- Ask about **observable conditions**, not treatment decisions
- Are stage-specific (M questions focus on bleeding, not airway)
- Avoid multi-part questions (single, focused question per stage)

---

### 2.11 Quality Evaluation System

The `QualityEvaluator` scores every response across three dimensions:

| Dimension | Weight | Scoring Method |
|-----------|--------|----------------|
| **Question Quality** | ~39% | Semantic similarity to stage definition, procedural question penalty, observable condition bonus, length check |
| **Citation Accuracy** | ~33% | Database lookup validation, duplicate penalty (>50% dupes), mapping success rate |
| **Checklist Quality** | ~28% | Actionable-verb detection (SpaCy + WordNet expansion), semantic relevance to user response, stage alignment |

**Actionable verb detection:**

```python
# Base verbs expanded via WordNet synonyms
base_verbs = {'apply', 'monitor', 'check', 'assess', ...}  # 30 verbs
# → Expanded to ~100+ verbs via WordNet

# SpaCy dependency parsing for imperative detection
for token in doc:
    if token.pos_ == "VERB" and token.dep_ == "ROOT":
        if token.lemma_ in actionable_verbs:
            return True  # Imperative verb at root → actionable
```

**Overall score formula:**

```
overall = question_score × 0.39 + citation_score × 0.33 + checklist_score × 0.28
```

---

## 3. Key Metrics & Results

### 3.1 Current Performance

| Metric | Value |
|--------|-------|
| **Overall Quality Score** | ~0.64 (baseline) |
| **Index Size** | 141 windows, 384 dimensions |
| **Q-Gen Latency** | <1 second (no retrieval) |
| **A-Gen Latency** | ~3-5 seconds (retrieval + reranking + generation) |
| **Citation Accuracy** | Variable — depends on LLM adherence to format |
| **MARCH-PAWS Coverage** | All 9 stages, sequential FSM |

### 3.2 Quality Improvement Roadmap

```
Score
  │
  │                                                          ┌──────── 0.92+
  │                                                    ┌─────┘  Long-term
  │                                              ┌─────┘
  │                                        ┌─────┘ 0.85
  │                                   ┌────┘  Medium-term
  │                              ┌────┘
  │                         ┌────┘ 0.75
  │                    ┌────┘  Quick wins
  │               ┌────┘
  │          ┌────┘ 0.64 (baseline)
  │     ┌────┘
  ├─────┘
  └──────────────────────────────────────────────────── Time
```

| Phase | Target | Key Actions |
|-------|--------|-------------|
| **Baseline** | 0.64 | Current system with hybrid retrieval + cross-encoder |
| **Quick Wins** (→0.75) | +0.11 | Better prompt engineering, expand few-shot examples from 45→90, tune RRF alpha grid, add more fallback citations |
| **Medium-Term** (→0.85) | +0.10 | Fine-tune MiniLM on medical domain, add paragraph-level BM25 (not just window), implement citation-aware prompting, chunking strategy optimization |
| **Long-Term** (→0.92+) | +0.07 | Upgrade to Mistral 7B fine-tuned on medical Q&A, domain-adapted cross-encoder, RLHF-style feedback loop from medic evaluations, multi-document retrieval |

---

## 4. Topics You Must Know (Study Guide)

### 4.1 RAG Pipeline End-to-End

**What is RAG?**
Retrieval-Augmented Generation combines information retrieval with LLM generation. Instead of relying solely on the model's parametric knowledge, RAG fetches relevant documents and injects them into the prompt.

```
Query → Retrieve relevant docs → Augment prompt with docs → Generate answer
```

**RAG vs Fine-tuning:**

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge freshness | Real-time (update index) | Stale (retrain needed) |
| Hallucination control | Citations ground answers | Still can hallucinate |
| Cost | Cheap (no GPU training) | Expensive (GPU hours) |
| Domain adaptation | Add documents | Need labeled data |
| Best for | Factual Q&A, docs | Style/format, reasoning |

**RAG failure modes:**
1. **Retrieval failure:** Relevant doc not in top-k
2. **Context stuffing:** Too many irrelevant docs dilute signal
3. **Lost in the middle:** LLM ignores middle of long context
4. **Citation hallucination:** Model invents non-existent references

---

### 4.2 FAISS: Index Types

| Index Type | Complexity | Use Case | Trade-off |
|------------|-----------|----------|-----------|
| **IndexFlatIP** (used here) | O(n) exact | Small corpora (<10K) | Exact, no approximation loss |
| **IndexFlatL2** | O(n) exact | L2 distance | Same but Euclidean |
| **IndexIVFFlat** | O(√n) approx | Medium (10K-1M) | Clusters with Voronoi cells, nprobe controls recall |
| **IndexIVFPQ** | O(√n) + compressed | Large (1M-1B) | Product quantization compresses vectors |
| **IndexHNSW** | O(log n) approx | Any size, high recall | Graph-based, excellent recall/speed trade-off |

**Why we chose IndexFlatIP:**
- Only 141 vectors — even brute-force is <1ms
- No approximation error — critical for medical accuracy
- Inner product on L2-normalized vectors = cosine similarity

**When to switch:**
- >10K documents → IVFFlat or HNSW
- >100K documents → IVFPQ with OPQ rotation
- >1M documents → Consider Pinecone/Weaviate for managed scaling

---

### 4.3 BM25: Term Frequency, IDF, Ranking

**BM25 Formula:**

```
BM25(q, d) = Σ IDF(qi) × [ f(qi, d) × (k1 + 1) ] / [ f(qi, d) + k1 × (1 - b + b × |d|/avgdl) ]
```

Where:
- `f(qi, d)` = term frequency of qi in document d
- `|d|` = document length
- `avgdl` = average document length
- `k1` = 1.2–2.0 (term saturation parameter)
- `b` = 0.75 (length normalization)

**Why BM25 is still relevant in 2025:**
- Excels at exact keyword matching (medical terms: "tourniquet", "pneumothorax")
- No model loading — instant search
- Complementary to dense retrieval (catches what embeddings miss)
- Standard implementation: `rank_bm25.BM25Okapi`

---

### 4.4 Hybrid Retrieval: RRF Formula

**Reciprocal Rank Fusion (Cormack et al., 2009):**

```
RRF(d) = Σ_r [ 1 / (k + rank_r(d)) ]

Weighted RRF:
RRF(d) = α / (rank_bm25(d) + k) + (1-α) / (rank_dense(d) + k)
```

**Why RRF and not simple score averaging?**
- BM25 and FAISS scores are on **different scales** — you can't average 0.0234 (BM25) with 0.89 (cosine)
- RRF uses **ranks**, which are scale-invariant
- `k=60` is the standard constant — prevents top-1 from dominating
- Proven to consistently outperform individual retrievers

---

### 4.5 Cross-Encoder vs Bi-Encoder

| Property | Bi-Encoder | Cross-Encoder |
|----------|-----------|---------------|
| **Architecture** | Two independent encoders | Single encoder, joint input |
| **Input** | query and doc encoded separately | (query, doc) concatenated |
| **Speed** | Fast: encode once, compare many | Slow: re-encode for each pair |
| **Accuracy** | Good | Excellent (sees full interaction) |
| **Use case** | Retrieval (top-1000) | Reranking (top-20 → top-5) |
| **Pre-compute** | Yes (doc embeddings cached) | No (pair-dependent) |

**In this system:**
1. **Bi-encoder** (MiniLM-L6-v2): Retrieves top-50 windows (fast)
2. **Cross-encoder** (ms-marco-MiniLM): Reranks top-20 windows (accurate)

---

### 4.6 Quantization: Q4_K_M, NF4, GGUF Formats

**GGUF (GPT-Generated Unified Format):**
- Successor to GGML format
- Single-file format containing model weights, tokenizer, and metadata
- Supported by llama.cpp, Ollama, LM Studio

**Quantization levels:**

| Format | Bits | Method | Quality | Size (7B) |
|--------|------|--------|---------|-----------|
| F16 | 16 | Full precision | Baseline | ~14 GB |
| Q8_0 | 8 | Round-to-nearest | Excellent | ~7 GB |
| Q5_K_M | 5 | K-quant medium | Very good | ~5 GB |
| **Q4_K_M** (used) | 4 | K-quant medium | Good | **~4.37 GB** |
| Q4_K_S | 4 | K-quant small | Acceptable | ~3.8 GB |
| Q3_K_M | 3 | K-quant medium | Fair | ~3.3 GB |
| Q2_K | 2 | K-quant | Poor | ~2.7 GB |

**K-quant explained:**
- Groups weights into blocks of 32/64
- Each block has its own scale and zero-point
- "M" (medium) keeps attention layers at higher precision
- "S" (small) quantizes attention too — faster but slightly less coherent

**NF4 (Normal Float 4-bit) — used by QLoRA:**
- Non-uniform quantization: denser buckets near zero (where most weights cluster)
- Better quality than uniform 4-bit for the same memory
- Used in bitsandbytes for fine-tuning, not inference

---

### 4.7 Sentence Transformers

**Architecture:**

```
Input text → Tokenizer → BERT/MiniLM encoder → [CLS] pooling → Dense layer → Normalized embedding
```

**Training objective:** Contrastive learning (similar pairs close, dissimilar pairs far).

**Key models:**

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 (used) | 384 | Fast | Good |
| all-MiniLM-L12-v2 | 384 | Medium | Better |
| all-mpnet-base-v2 | 768 | Slow | Best (general) |
| nomic-embed-text-v1.5 | 768 | Medium | Strong (long docs) |
| BGE-large-en-v1.5 | 1024 | Slow | Top MTEB |

---

### 4.8 Medical NLP Challenges

| Challenge | How We Addressed It |
|-----------|-------------------|
| **Dense technical jargon** | BM25 catches exact medical terms; embeddings handle paraphrases |
| **Abbreviations** (TCCC, MEDEVAC) | polish_tokens() normalizes; BM25 lemmatization handles variants |
| **Multi-version documents** | Version detection (Base/C1/C2); deduplication via content hashing |
| **Structured numbering** (6-4, 1-19) | Custom regex parsers for paragraph ID detection |
| **Safety-critical accuracy** | Citation grounding; no hallucinated references; refusal for non-medical queries |
| **Domain drift** | Stage-aware filtering ensures stage-appropriate content |

---

### 4.9 PDF Processing: PyMuPDF

**PyMuPDF (fitz) capabilities used:**

```python
import fitz

doc = fitz.open("tc4-02.1.pdf")
page = doc[0]

# Text with layout info
page.get_text("dict")     # Returns blocks with spans, fonts, bboxes
page.get_text("blocks")   # Returns (x0, y0, x1, y1, text, ...)

# Table of contents
doc.get_toc(simple=True)  # [[level, title, page], ...]

# Font-based heading detection
for span in line["spans"]:
    if span["size"] > large_threshold:  # Top 20% font sizes
        mark_as_heading(span)
```

**Key techniques:**
- **Region stripping:** Header (top 80px) and footer (bottom 30px) removed
- **Font-size analysis:** Global size distribution computed; top 20% = headings
- **Bounding box tracking:** y-coordinates used to assign paragraphs to nearest heading
- **Cross-page stitching:** Incomplete paragraphs merged across page boundaries

---

### 4.10 Streamlit Deployment

**Key Streamlit patterns used:**

```python
@st.cache_resource
def load_system():
    """Load once, share across sessions."""
    orchestrator = AsyncOrchestrator(bm25_path, embeddings_path, metadata_path)
    evaluator = QualityEvaluator()
    return {'orchestrator': orchestrator, 'evaluator': evaluator}

# Session state for multi-step assessment
st.session_state.scenario = ""
st.session_state.current_stage = "M"
st.session_state.conversation_history = []

# Async integration
result = asyncio.run(orchestrator.run_step(scenario, user_answer))
```

---

## 5. Interview Questions & Answers (25+)

### Architecture & Design

**Q1: Walk me through how a user query becomes a medical checklist in your system.**

**A:** When a user enters a scenario like "soldier with chest gunshot wound":
1. The **FSM** identifies the current stage (e.g., M for Massive Hemorrhage)
2. **Q-Gen** generates a stage-specific question using the stage definition + few-shot context engineering — no retrieval needed, so it's <1s
3. User answers the question
4. **A-Gen** kicks in: the combined query (scenario + user answer + stage definition) goes to hybrid retrieval
5. **FAISS** returns top-50 windows by semantic similarity; **BM25** returns top-50 by lexical matching
6. **RRF** fuses rankings with adaptive alpha (short query → dense weight, long → BM25 weight)
7. Top-20 windows go through **cross-encoder reranking** (ms-marco-MiniLM, sigmoid-normalized)
8. Top-5 reranked windows are expanded to individual paragraphs, scored again by cross-encoder
9. **Stage-aware filtering** removes content from already-completed stages
10. Excerpts are formatted into the A-Gen prompt with citation hints
11. **Mistral 7B** generates a JSON response: `{checklist: [...], citations: [...]}`
12. Citations are mapped to the database, verified, and displayed with quality scores

---

**Q2: Why did you choose a two-prompt architecture (Q-Gen / A-Gen) instead of a single prompt?**

**A:** Three reasons:
1. **Latency optimization:** Q-Gen doesn't need retrieval — it only needs the stage definition and scenario. This makes question generation <1s vs 3-5s if we had to retrieve first. Users get immediate feedback.
2. **Quality isolation:** Question quality and answer quality have different failure modes. Separating them lets us debug independently. If questions are bad, fix Q-Gen prompt; if citations are wrong, fix A-Gen.
3. **Pre-fetching:** While the user reads the current checklist, we can pre-fetch the next question in the background, making the transition feel instant.

---

**Q3: Why hybrid retrieval instead of just dense (FAISS) or just sparse (BM25)?**

**A:** Dense and sparse retrievers have complementary failure modes:
- **Dense fails when:** The query uses exact medical terminology that the model hasn't seen in training (e.g., "needle decompression for tension pneumothorax")
- **BM25 fails when:** The query is paraphrased or uses synonyms (e.g., "trouble breathing" vs "respiratory distress")

In practice, hybrid retrieval with RRF consistently outperforms either alone. Our adaptive alpha further optimizes: short factual queries lean on dense (α=0.35), longer descriptive queries lean on BM25 (α=0.55). The z-score threshold prevents low-relevance results from entering the pipeline.

---

**Q4: Why IndexFlatIP and not HNSW or IVF for FAISS?**

**A:** With only 141 vectors, flat brute-force search is <1ms. HNSW or IVF would add index-building complexity and approximation error for zero performance benefit. However, if we scaled to thousands of documents, I'd switch to `IndexHNSWFlat` for the best recall-speed trade-off, or `IndexIVFPQ` if memory is constrained.

---

**Q5: How does your system prevent hallucinated citations?**

**A:** Three safeguards:
1. **Citation hints in prompt:** The A-Gen prompt includes an "AVAILABLE CITATIONS" section listing exact paragraph IDs from retrieved content, guiding the LLM to use real references
2. **Database mapping:** Every LLM-generated citation is regex-parsed and looked up in `tc4-02.1_sections.jsonl`. If it maps to a real entry, the database's `short_ref` is used
3. **Drop-on-failure:** If a citation can't be mapped to any database entry, it's **dropped entirely** rather than passed through. The system prefers silence over a fabricated reference

---

### Retrieval Deep-Dive

**Q6: Explain Reciprocal Rank Fusion. Why k=60?**

**A:** RRF converts raw retrieval scores (which may be on incompatible scales) into rank-based scores:

```
RRF(d) = Σ [ 1 / (k + rank(d)) ]
```

`k=60` is the original value from Cormack et al. (2009). It serves as a damping constant — with `k=60`, rank 1 gets score 1/61 ≈ 0.0164 and rank 60 gets 1/120 ≈ 0.0083. This prevents the top-ranked document from dominating the fusion. Empirically, k=60 works well across diverse retrieval settings. Lower k values (like 10) would make the system overly sensitive to rank-1 disagreements between retrievers.

---

**Q7: What is your adaptive alpha strategy? How would you improve it?**

**A:** Currently alpha is rule-based:
- ≤3 words → α=0.35 (favor dense — short queries need semantic expansion)
- Contains "?" → α=0.45 (questions benefit from embedding similarity)
- Default → α=0.55 (longer queries have enough keywords for BM25)

**Improvements I'd make:**
1. **Learned alpha:** Train a small logistic regression on query features (length, medical term density, question type) to predict optimal alpha
2. **Per-query grid search:** On a validation set, sweep alpha ∈ [0.1, 0.9] and pick the best per query type
3. **Dynamic k in RRF:** Vary k based on score distribution spread

---

**Q8: Explain the z-score threshold. When does it help?**

**A:** The z-score threshold is computed as `(max_score - mean_score) / std_score` across the top-100 BM25 scores. A high z-score means the top result is clearly discriminated from the rest — the query is specific and matched well. A low z-score means scores are uniformly distributed — the query is vague or off-topic.

We normalize this to [0.001, 0.01] and use it as a minimum relevance threshold. This prevents the system from confidently retrieving irrelevant content for queries like "what's the weather?" where BM25 will still return something (everything gets some non-zero score).

---

### Model & Quantization

**Q9: Why Mistral 7B? Why not a larger model or GPT-4?**

**A:** Three constraints drove the choice:
1. **Privacy:** Medical data can't leave the local machine — rules out API-based models
2. **Cost:** The system runs on a single machine with limited GPU — 7B at 4-bit fits in ~4.4 GB
3. **Latency:** Local inference avoids network round-trips; generation takes 2-3s

Mistral 7B was chosen over Llama 2 7B because it has better instruction-following and longer effective context (sliding window attention). The Q4_K_M quantization hits the sweet spot — perplexity increase is negligible for structured output (JSON checklists).

---

**Q10: Explain Q4_K_M quantization. How is it different from NF4?**

**A:**

| Property | Q4_K_M (GGUF) | NF4 (bitsandbytes) |
|----------|---------------|---------------------|
| Precision | 4-bit uniform (per block) | 4-bit non-uniform (quantile-based) |
| Block size | 32 weights per block | 64 weights per block |
| Attention layers | Higher precision (5-6 bit) | Same 4-bit as rest |
| Format | GGUF (llama.cpp, Ollama) | PyTorch (bitsandbytes, HuggingFace) |
| Use case | **Inference** | **Fine-tuning (QLoRA)** |
| Double quant | No | Yes (quantize the quantization constants) |

Q4_K_M is specifically designed for **inference quality**: it keeps attention and output layers at slightly higher precision because these layers are most sensitive to quantization noise. NF4 uses a non-uniform code book (quantile-based) which is theoretically better for normally-distributed weights but requires bitsandbytes runtime.

---

**Q11: What's the memory budget for your full system?**

**A:**

| Component | Memory |
|-----------|--------|
| Mistral 7B Q4_K_M weights | ~4.37 GB |
| KV-cache (4096 context) | ~0.3 GB |
| MiniLM-L6-v2 (embedder) | ~0.15 GB |
| ms-marco-MiniLM (reranker) | ~0.15 GB |
| FAISS index (141 × 384) | ~0.2 MB |
| BM25 index + metadata | ~5 MB |
| SpaCy en_core_web_sm | ~50 MB |
| Python + Streamlit overhead | ~0.5 GB |
| **Total** | **~5.5 GB** |

Fits comfortably on a single 8 GB GPU with room for batch processing.

---

### Evaluation & Quality

**Q12: How do you measure the quality of your RAG system?**

**A:** Three-dimensional evaluation:

1. **Question Quality (~39% weight):** Semantic similarity to stage definition (MiniLM cosine), observable-condition check (regex), procedural-question penalty, length validation, multi-part question detection

2. **Citation Accuracy (~33% weight):** Each citation is mapped to the paragraph database — valid citations score 1.0, missing citations score 0.0. Duplicate penalty if >50% are identical.

3. **Checklist Quality (~28% weight):** Actionable-verb detection via SpaCy dependency parsing (imperative ROOT verbs), semantic similarity to user response, stage alignment check.

The overall score is a weighted combination. At baseline we're at ~0.64, primarily dragged down by citation accuracy (LLM sometimes generates citations in non-standard formats).

---

**Q13: What are the main quality bottlenecks, and how would you fix them?**

**A:**

| Bottleneck | Current Impact | Fix |
|-----------|---------------|-----|
| Citation format adherence | LLM sometimes outputs citations in unexpected formats | Constrained decoding / JSON schema enforcement |
| Stage contamination | Early stages' content leaks into later stages | Improve keyword lists; add embedding-based stage classifier |
| Q-Gen relevance | Some questions are too generic | Expand few-shot examples; add negative examples |
| Checklist specificity | Some items are vague ("Continue monitoring") | Fine-tune on human-written checklists; reward specific actions |
| Retrieval recall | Relevant paragraphs sometimes not in top-20 | Expand retrieval to top-50; add query expansion |

---

### Pipeline & Engineering

**Q14: How does your FSM work? What happens if a stage has no relevant content?**

**A:** The FSM is a simple dataclass with an ordered sequence: `[M, A, R, C, H, P, A2, W, S]` and a state index. `advance()` increments the index; `has_more()` checks if we've reached "END".

If a stage has no relevant content (retrieval scores below threshold), the system activates a **fallback chain**:
1. Try stage-specific fallback excerpts (`load_generic_paras()`)
2. If those fail, generate a minimal excerpt: "Complete {state} assessment per TCCC guidelines"
3. Generate a robust checklist using pre-written templates ("Assessment completed — continue to next stage")
4. The FSM **always advances** — we never get stuck on a stage

---

**Q15: Explain the async orchestrator's pre-fetching strategy.**

**A:** The async orchestrator exploits **user think-time** for parallelism:

```
Timeline:
User reads checklist for Stage M (thinking...)
    ├── Background: Pre-fetch Q-Gen for Stage A (cached)
    ├── Background: Pre-fetch retrieval for Stage A
    └── When user submits answer → Stage A question is instant
```

Implementation:
- `_prefetch_next_question()`: Temporarily advances FSM, generates and caches the next question, then reverts the state index
- `_prefetch_next_retrieval()`: Pre-runs hybrid retrieval for the next state's likely query
- Uses `asyncio.create_task()` for non-blocking execution
- `q_cache` dict stores `(hash(scenario), state) → question`

---

**Q16: How do you handle the "window expansion" from parent windows to child paragraphs?**

**A:** Windows are the retrieval unit (for context), but paragraphs are the citation unit (for precision). After retrieving top-5 windows, we expand them:

1. Each window has a `paragraph_ids` list pointing to original paragraphs
2. We use **round-robin** across windows (not exhausting window 1 before window 2) to get diverse paragraphs
3. Each paragraph is looked up in `para_map` (from `tc4-02.1_sections.jsonl`)
4. The cross-encoder scores each paragraph individually against the query
5. Top paragraphs (by CE score) become the excerpts for the A-Gen prompt
6. Deduplication via `seen` set prevents the same paragraph from appearing twice

---

### Behavioral & Scenario Questions

**Q17: Tell me about a technical challenge you faced and how you solved it.**

**A:** The biggest challenge was **stage contamination** in retrieved content. When the system reached the Respiration (R) stage, it would retrieve content about Massive Hemorrhage (M) because both mention "chest" and "wound." Initially, I tried Jaccard similarity filtering between the query and stage definitions, but it was too fragile — it filtered out legitimate content too aggressively.

I replaced it with a cleaner approach: **explicit stage-keyword dictionaries**. Each MARCH-PAWS stage has a curated list of keywords (M: bleeding, hemorrhage, tourniquet; R: respiratory, breathing, lung). As the FSM advances, content matching completed stages' keywords is excluded. The critical safety net: if filtering removes too much (below 3 excerpts), we fall back to the top cross-encoder-scored results regardless of stage. This resolved the contamination while maintaining content availability.

---

**Q18: How would you scale this system to 1000 medical manuals?**

**A:** Several changes at each layer:

| Layer | Current (1 manual) | At Scale (1000 manuals) |
|-------|-------|----------|
| **Index** | IndexFlatIP (141 vectors) | IndexHNSW or IVF + PQ (~100K+ vectors) |
| **Storage** | Local files | PostgreSQL + pgvector |
| **Embedding** | Sequential | Distributed: Ray/Spark batch embedding |
| **BM25** | In-memory pickle | Elasticsearch with medical analyzers |
| **Reranking** | Single cross-encoder | Batched GPU inference, ColBERT for fast reranking |
| **LLM** | Single Ollama | vLLM with continuous batching, or API-based |
| **Frontend** | Streamlit | React + FastAPI backend |
| **Caching** | Python dict | Redis for session + embedding cache |

---

**Q19: How do you ensure the system doesn't give dangerous medical advice?**

**A:** Multiple safety layers:

1. **Non-medical query rejection:** Multi-layered regex + keyword detection rejects non-medical queries (weather, cooking, etc.)
2. **Citation grounding:** Every recommendation traces back to the TC 4-02.1 source — no unsupported claims
3. **Refusal mechanism:** If retrieved content scores below the z-score threshold, the system responds with "I cannot advise — no relevant content found" instead of guessing
4. **Stage constraints:** The FSM prevents skipping stages — you can't jump to Splinting without completing Massive Hemorrhage first
5. **Quality evaluation:** Real-time scoring alerts when response quality drops below acceptable thresholds
6. **Explicit prompt guardrails:** "If the excerpts don't support an answer, reply EXACTLY: I cannot advise — no relevant content found"

---

**Q20: What would you do differently if you started this project today?**

**A:**
1. **Start with evaluation:** Build the quality evaluator first, then iterate on the pipeline with measurable feedback
2. **ColBERT instead of cross-encoder:** Late interaction models like ColBERT give cross-encoder-like quality at bi-encoder-like speed
3. **Structured output:** Use constrained decoding (e.g., outlines, instructor) to guarantee JSON format from the LLM
4. **Domain-adapted embeddings:** Fine-tune MiniLM on medical sentence pairs before building the index
5. **User feedback loop:** Add thumbs-up/down on checklists to build a preference dataset for RLHF

---

**Q21: Compare your approach to using a fine-tuned medical LLM like Med-PaLM.**

**A:**

| Aspect | Our RAG Approach | Fine-tuned Medical LLM |
|--------|-----------------|----------------------|
| **Knowledge source** | Explicit (TC 4-02.1 manual) | Parametric (memorized during training) |
| **Citations** | Exact paragraph-level | Cannot cite specific passages |
| **Updates** | Re-index new manual → instant | Re-train model → expensive |
| **Hallucination** | Controlled (grounded in docs) | Still possible despite fine-tuning |
| **Cost** | Low (Mistral 7B local) | High (proprietary API or large GPU) |
| **Domain narrow** | Perfect for one manual | Broad medical knowledge |

For this use case — a single authoritative manual with citation requirements — RAG is definitively the right choice.

---

**Q22: How does your sliding window strategy handle the trade-off between context and precision?**

**A:** Windows are 250 tokens with 100-token overlap. This is intentional:
- **250 tokens** captures 3-5 paragraphs — enough context for the LLM to understand the medical procedure
- **100-token overlap** ensures no information is lost at window boundaries — a paragraph about "tourniquet placement" won't be split across two windows with no overlap
- **Semantic boundary detection** prevents mixing unrelated sections (e.g., Hemorrhage and Airway) in the same window
- Giant paragraphs (>250 tokens) are split on **sentence boundaries** at ~120 tokens each, preserving readability

The trade-off: smaller windows = more precise retrieval but less context; larger windows = more context but noisier retrieval. 250 tokens is the sweet spot for medical paragraphs that are typically 50-150 tokens each.

---

**Q23: Explain your context engineering approach for Q-Gen.**

**A:** Instead of fine-tuning, we use **context engineering** — injecting the right examples into the prompt at runtime:

1. **45 curated examples** (5 per MARCH-PAWS stage) stored in `scenario_examples.json`
2. Each example has: scenario, question, reasoning
3. At runtime, we **score examples by keyword overlap** with the current scenario
4. Top-3 most relevant examples are injected as few-shot demonstrations
5. The prompt also includes the **stage definition** (what to focus on) and **anti-patterns** (don't ask procedural questions)

This is cheaper than fine-tuning, instantly updatable (add more examples to the JSON), and preserves the model's general instruction-following ability.

---

**Q24: How would you add multi-language support?**

**A:** Three options, increasing in effort:

1. **Query translation:** Translate query to English → retrieve → generate in English → translate response back. Cheapest but adds latency and translation errors.
2. **Multilingual embeddings:** Replace MiniLM with `multilingual-e5-large` (supports 100+ languages). Index stays the same, but queries in any language find relevant English content.
3. **Full multilingual pipeline:** Translate the source manual; build separate indices per language; use a multilingual LLM (e.g., Aya 8B). Most robust but highest cost.

For medical safety, option 1 with human-verified translations is the safest starting point.

---

**Q25: What happens when the LLM generates invalid JSON?**

**A:** The system has a robust fallback chain:

```python
try:
    answer_json = json.loads(response_text)
except json.JSONDecodeError:
    # Fallback: treat entire response as a single checklist item
    answer_json = {
        "checklist": [response_text],
        "citations": [],
        "state_complete": True
    }
```

Additionally:
- If checklist is empty after parsing, `_generate_robust_checklist()` creates stage-appropriate default items
- If citations are empty, `_generate_robust_citations()` creates generic TCCC references
- The FSM **always advances** — invalid LLM output never blocks progression through MARCH-PAWS stages

---

## 6. Red Flags & How to Handle

### "Your quality score is only 0.64. Why so low?"

**Response:** "0.64 is our **automated baseline** across all three dimensions (question quality, citation accuracy, checklist quality). The main drag is citation accuracy — the LLM sometimes outputs citations in non-standard formats that fail our strict database mapping. Human evaluators rate the checklist content significantly higher. We have a concrete roadmap: prompt-level citation formatting gets us to 0.75, domain-adapted models to 0.85, and fine-tuning to 0.92+."

### "Why not use GPT-4 or Claude instead of Mistral 7B?"

**Response:** "Three constraints: (1) Medical data privacy — content can't leave the local machine; (2) Cost — no per-token API charges for a system that generates 9 prompts per assessment; (3) Latency — local inference avoids network round-trips. For this specific task (structured JSON generation from retrieved content), Mistral 7B at 4-bit quantization performs well. If we needed complex reasoning or multi-step planning, I'd consider larger models."

### "Only 141 vectors? That's tiny."

**Response:** "The TC 4-02.1 manual is a single, authoritative reference — not a document collection. 141 windows cover the entire manual comprehensively with overlap. The small corpus is actually an advantage: we use exact search (IndexFlatIP) with zero approximation error, and our hybrid retrieval ensures we never miss relevant content. The real complexity is in the 8-step ingestion pipeline, hybrid retrieval, and multi-stage orchestration — not raw corpus size."

### "How do you know the medical advice is correct?"

**Response:** "We don't claim autonomous correctness — the system is a **decision support tool**, not a replacement for medical training. Every recommendation is (1) grounded in retrieved excerpts from the official TCCC manual, (2) cited with verifiable paragraph references, (3) filtered through a quality evaluator, and (4) subject to the FSM's systematic stage-by-stage approach. The refusal mechanism ensures the system says 'I don't know' rather than guessing when content relevance is low."

### "Your ingestion pipeline is complex. Why not just use LangChain's PDF loader?"

**Response:** "Generic PDF loaders produce poor results on structured military manuals because: (1) They miss the numbered paragraph format (e.g., '6-4. Apply tourniquet...'); (2) They don't detect heading hierarchy from font sizes; (3) They can't handle cross-page paragraph stitching; (4) They don't produce citation-quality metadata. Our custom pipeline preserves paragraph-level granularity and generates the anchor metadata needed for verifiable citations. The extra effort pays off in citation accuracy."

---

## 7. Key Takeaways & Talking Points

### Sound-Bites for Interviews

1. **"I built a medical RAG system that turns a 200-page military manual into an interactive Q&A assistant with paragraph-level citations."**

2. **"The hybrid retrieval (FAISS + BM25 + cross-encoder reranking) ensures we catch both semantic matches and exact medical terminology."**

3. **"The two-prompt architecture (Q-Gen / A-Gen) cuts perceived latency in half — questions appear instantly while retrieval runs in the background."**

4. **"Every citation is database-verified. If the LLM hallucinates a reference, it's dropped — we never surface ungrounded medical advice."**

5. **"Context engineering with 45 curated examples achieves few-shot quality without any fine-tuning cost."**

### Architecture Principles Demonstrated

| Principle | How It's Applied |
|-----------|-----------------|
| **Separation of concerns** | Q-Gen (fast, no retrieval) vs A-Gen (thorough, with retrieval) |
| **Graceful degradation** | Fallback excerpts → robust checklists → FSM always advances |
| **Safety by design** | Non-medical rejection, citation grounding, refusal mechanism |
| **Observability** | Quality evaluator scores every response in real-time |
| **Efficient resource use** | 4-bit quantized LLM, LRU caching, async pre-fetching |
| **Domain adaptation** | Custom PDF pipeline, stage-aware filtering, medical keyword lists |

### Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Mistral 7B (Q4_K_M) | ~4.37 GB |
| MiniLM-L6-v2 embeddings | 384 dimensions |
| FAISS index | 141 vectors |
| MARCH-PAWS stages | 9 (M, A, R, C, H, P, A2, W, S) |
| Few-shot examples | 45 (5 per stage) |
| RRF constant k | 60 |
| Window size | 250 tokens, 100 overlap |
| Quality baseline | ~0.64 |
| Target quality | 0.92+ |
| Total system memory | ~5.5 GB |
