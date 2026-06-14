# RAG (Retrieval-Augmented Generation) Systems — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, RAG Systems, NLP Pipelines

---

# Table of Contents

1. [What Is RAG](#1-what-is-rag)
2. [RAG Pipeline Step by Step](#2-rag-pipeline-step-by-step)
3. [Hybrid Retrieval](#3-hybrid-retrieval)
4. [Reciprocal Rank Fusion (RRF)](#4-reciprocal-rank-fusion-rrf)
5. [Cross-Encoder Re-Ranking vs MMR](#5-cross-encoder-re-ranking-vs-mmr)
6. [Tuning Knobs](#6-tuning-knobs)
7. [Recall–Latency–RAM Triangle](#7-recalllatencyram-triangle)
8. [Advanced RAG Patterns](#8-advanced-rag-patterns)
9. [Production Considerations](#9-production-considerations)
10. [Rahul's Project Deep-Dives](#10-rahuls-project-deep-dives)
11. [Common Interview Questions](#11-common-interview-questions)
12. [Key Takeaways](#12-key-takeaways)

---

# **1. What Is RAG**

---

## **1.1 Motivation: Why LLMs Need Retrieval**

Large Language Models have a fundamental limitation: **they hallucinate**. Their knowledge is frozen at the pre-training cutoff, they can't access private or domain-specific corpora, and they confidently fabricate plausible-sounding but incorrect facts.

| Problem | LLM Alone | LLM + RAG |
|---------|-----------|-----------|
| Knowledge cutoff | Stale answers after training date | Live retrieval from updated corpus |
| Domain expertise | Generic, shallow | Grounded in your specific documents |
| Hallucination | Frequent confabulation | Answers anchored to source text |
| Auditability | Black-box generation | Citations trace back to documents |
| Cost of updates | Full fine-tune ($$$) | Re-index documents (cheap) |

**Key insight:** Fine-tuning teaches a model *how* to respond (style, format). RAG teaches a model *what* to respond with (facts, evidence). Most production systems need both.

---

## **1.2 Core Concept**

RAG decouples **knowledge storage** (retrieval corpus) from **reasoning** (LLM). At query time, relevant documents are retrieved and injected into the LLM's context window as grounding evidence.

```
RAG = Retrieve(query, corpus) → Context + Query → LLM → Grounded Answer
```

---

## **1.3 Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG SYSTEM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │  Documents   │──── Ingest & Clean ──── Chunk ──── Embed ────┐        │
│  │ PDFs, HTML,  │      (parse, dedupe)  (512-1024   (Cohere,   │        │
│  │  Docs, APIs  │                        tokens)    OpenAI)    │        │
│  └─────────────┘                                               │        │
│                                                                ▼        │
│                                                    ┌──────────────────┐ │
│                                                    │   Vector Store   │ │
│                                                    │  (FAISS / Pine-  │ │
│                                                    │  cone / Weaviate │ │
│                                                    │  / pgvector)     │ │
│                                                    └────────┬─────────┘ │
│                                                             │           │
│  ┌──────────┐     ┌─────────────┐     ┌──────────┐        │           │
│  │  User     │────▶│  Query      │────▶│ Dense    │◀───────┘           │
│  │  Query    │     │  Embedding  │     │ Retrieval│                    │
│  └──────────┘     └─────────────┘     └────┬─────┘                    │
│       │                                     │                          │
│       │           ┌─────────────┐          │   ┌──────────────┐       │
│       └──────────▶│  BM25 /     │──────────┼──▶│  Fusion /    │       │
│                   │  Sparse     │          │   │  RRF / Linear│       │
│                   └─────────────┘          │   └──────┬───────┘       │
│                                            │          │               │
│                                            │          ▼               │
│                                            │   ┌──────────────┐      │
│                                            │   │  Re-Ranker    │      │
│                                            │   │ (Cross-Enc.)  │      │
│                                            │   └──────┬───────┘      │
│                                            │          │               │
│                                            │          ▼               │
│                                            │   ┌──────────────┐      │
│                                            │   │  Context Eng. │      │
│                                            │   │  MMR, Budget, │      │
│                                            │   │  Compression  │      │
│                                            │   └──────┬───────┘      │
│                                            │          │               │
│                                            │          ▼               │
│                                            │   ┌──────────────┐      │
│                                            └──▶│     LLM       │      │
│                                                │  Generation   │      │
│                                                │  (cite, refuse│      │
│                                                │  if unsure)   │      │
│                                                └──────┬───────┘      │
│                                                       │               │
│                                                       ▼               │
│                                                ┌──────────────┐      │
│                                                │  Answer +     │      │
│                                                │  Citations    │      │
│                                                └──────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# **2. RAG Pipeline Step by Step**

---

## **2a. Ingest & Clean**

The first step is getting raw documents into a processable format.

### Parsing

| Source | Tool / Approach |
|--------|----------------|
| PDF (selectable text) | `PyMuPDF` (fitz), `pdfplumber`, `PyPDF2` |
| PDF (scanned) | OCR via `Tesseract`, `AWS Textract`, `Azure Document Intelligence` |
| HTML | `BeautifulSoup`, `trafilatura` (extracts main content) |
| Word/DOCX | `python-docx` |
| Structured (CSV/JSON) | `pandas`, custom parsers |

### Cleaning Pipeline

```python
def clean_document(raw_text: str) -> str:
    # 1. Normalize unicode
    text = unicodedata.normalize("NFKC", raw_text)
    # 2. Strip boilerplate (headers, footers, page numbers)
    text = remove_boilerplate(text)
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 4. Detect language (optional)
    lang = detect(text)
    return text
```

### Deduplication

- **Exact:** Hash-based (MD5/SHA256 of normalized text)
- **Near-duplicate:** MinHash + LSH (Locality-Sensitive Hashing) for fuzzy matching
- **Semantic:** Embedding cosine similarity > 0.95 → deduplicate

### Metadata Extraction

Always store alongside chunks: `title`, `section`, `page_number`, `source_url`, `timestamp`, `tags`, `author`. This metadata powers **filtered retrieval** later.

---

## **2b. Chunking**

Chunking is the **most impactful design decision** in a RAG pipeline. Bad chunking = bad retrieval, no matter how good your embedding model.

### Chunking Strategies

| Strategy | Description | Typical Config | Pros | Cons |
|----------|-------------|---------------|------|------|
| **Fixed-Window** | Split every N tokens | 512–1024 tokens | Simple, predictable | Cuts mid-sentence |
| **Overlap** | Sliding window with overlap | 10–20% overlap (e.g., 50–128 tokens) | Preserves context at boundaries | Slightly more storage |
| **Sentence-Based** | Split on sentence boundaries | Use `nltk.sent_tokenize` | Natural boundaries | Uneven chunk sizes |
| **Semantic** | Group sentences by embedding similarity | Threshold on cosine distance | Topically coherent | Expensive to compute |
| **Heading-Based** | Split on document headings (H1, H2, H3) | Respect document structure | Preserves document hierarchy | Requires structured docs |
| **Recursive / Hierarchical** | Try large split → fall back to smaller | LangChain `RecursiveCharacterTextSplitter` | Adaptive | More complex logic |

### The Overlap Principle

```
Chunk N:   [..........tokens 1-512..........]
                                    ├── overlap ──┤
Chunk N+1:                   [..........tokens 463-974..........]
```

**Why overlap?** A key fact might straddle a chunk boundary. With 10–20% overlap, the fact appears in at least one chunk in full.

### Window-Based Chunking (Used in Medical RAG)

A more sophisticated approach: create **parent windows** that span multiple paragraphs, indexed with stride overlap. At retrieval time, retrieve the parent window, then drill down to the most relevant child paragraphs using a cross-encoder.

```
Document:  [Para1] [Para2] [Para3] [Para4] [Para5] [Para6] [Para7]

Window 1:  [Para1  Para2  Para3  Para4]
Window 2:         [Para2  Para3  Para4  Para5]     ← stride overlap
Window 3:                [Para3  Para4  Para5  Para6]
```

### PathWise Chunking (Production Example)

```python
CHUNK_WORDS = 400    # ~512 tokens
OVERLAP = 50         # ~12.5% overlap

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP:]  # sliding window overlap
    if cur:
        chunks.append(" ".join(cur))
    return chunks
```

---

## **2c. Embedding**

Embedding converts text chunks into dense vectors in a shared semantic space, enabling similarity search.

### Embedding Models Comparison

| Model | Dimensions | Provider | MTEB Score | Notes |
|-------|-----------|----------|-----------|-------|
| `text-embedding-3-large` | 3072 | OpenAI | ~64.6 | Best commercial; supports dim reduction |
| `text-embedding-3-small` | 1536 | OpenAI | ~62.3 | Cost-efficient |
| `embed-english-light-v3.0` | 384 | Cohere | ~60+ | Lightweight, fast; **used in PathWise** |
| `embed-english-v3.0` | 1024 | Cohere | ~64.5 | Full Cohere model |
| `e5-large-v2` | 1024 | Microsoft | ~63.5 | Open-source, strong |
| `bge-large-en-v1.5` | 1024 | BAAI | ~63.6 | Open-source, SOTA |
| `all-MiniLM-L6-v2` | 384 | SBERT | ~56.3 | Tiny, fast; **used in Medical RAG** |
| `GTE-Qwen2-7B` | 3584 | Alibaba | ~65+ | LLM-based, SOTA |

### L2 Normalization for Cosine Similarity

When using cosine similarity, **always L2-normalize** your vectors before indexing. This converts dot product into cosine similarity:

```
                    a · b                    a · b
cosine(a, b) = ──────────── = a_norm · b_norm  (if ||a|| = ||b|| = 1)
               ||a|| · ||b||

After L2-normalization: cosine(a, b) = dot(a, b)
```

**Why this matters:** FAISS's `IndexFlatIP` (inner product) becomes cosine similarity after normalization, avoiding the overhead of explicit cosine computation.

```python
# FAISS normalization (in-place, memory efficient)
embeddings = np.asarray(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings)         # now ||v|| = 1 for all v
index = faiss.IndexFlatIP(dimension)   # IP ≡ cosine after normalization
index.add(embeddings)
```

### Embedding Best Practices

1. **Match query and document types:** Cohere distinguishes `input_type="search_document"` vs `"search_query"` — always use the correct type
2. **Batch embed:** Process chunks in batches (e.g., 96 at a time) to minimize API calls
3. **Cache embeddings:** Never re-embed unchanged documents
4. **Dimensionality vs quality:** 384-dim is sufficient for most use cases; 1024+ for high-precision domains

---

## **2d. Indexing**

### Vector Database Options

| Vector DB | Index Types | Managed? | Filtering | Scale | Notes |
|-----------|------------|----------|-----------|-------|-------|
| **FAISS** | Flat, IVF, HNSW, PQ | No (library) | Post-filter | Millions+ | Gold standard for local; **used in Medical RAG** |
| **Pinecone** | Proprietary | Yes (SaaS) | Pre-filter (metadata) | Billions | Serverless option, easy to start |
| **Weaviate** | HNSW | Self-host or SaaS | Hybrid (pre+post) | Millions | GraphQL API, module ecosystem |
| **pgvector** | IVFFlat, HNSW | Postgres extension | SQL WHERE | Millions | Familiar if you already use Postgres |
| **Qdrant** | HNSW | Self-host or SaaS | Pre-filter | Millions | Rust-based, fast |
| **Chroma** | HNSW | No (library) | Metadata filter | Thousands | Lightweight, prototyping |
| **Milvus** | IVF, HNSW, DiskANN | Self-host or Zilliz | Attribute filter | Billions | Enterprise scale |

### Index Types Deep-Dive

#### Flat (Brute-Force)
```
Query → compare with ALL vectors → exact top-k
```
- **Recall:** 100% (exact)
- **Latency:** O(n) — linear scan
- **RAM:** Full vectors in memory
- **Use when:** < 100k vectors, need perfect recall

#### IVF (Inverted File Index)
```
Offline: Cluster vectors into nlist centroids using k-means
Query:  1. Find nprobe nearest centroids
        2. Search only vectors in those clusters
```
- **Recall:** ~95–99% (with proper nprobe)
- **Latency:** O(n/nlist × nprobe)
- **RAM:** Full vectors + centroid table
- **Key params:** `nlist` (number of clusters), `nprobe` (clusters to search)

#### HNSW (Hierarchical Navigable Small World)
```
Layer 3:  [A] ────────────────── [D]           (sparse, long jumps)
Layer 2:  [A] ──── [C] ──────── [D]
Layer 1:  [A] ─ [B] ─ [C] ─ [D] ─ [E]        (dense, short jumps)
Layer 0:  [A]-[B]-[C]-[D]-[E]-[F]-[G]-[H]     (all vectors)

Search: Start at top layer → greedy walk → drop layer → repeat
```
- **Recall:** 95–99.9% (tunable)
- **Latency:** O(log n) — navigable graph
- **RAM:** Vectors + adjacency lists (~1.5× raw vectors)
- **Key params:** `M` (edges per node), `efConstruction` (build quality), `efSearch` (search quality)
- **Trade-off:** Higher M/ef = better recall but more RAM and slower build

#### Product Quantization (PQ)
- Compresses vectors by splitting into sub-vectors and quantizing
- **RAM:** 4–32× reduction
- **Recall:** Lossy (~90–95%)
- **Use when:** Billions of vectors, RAM constrained

### Choosing an Index

```
                       Recall
                         ▲
                         │
                    100% │  Flat ●
                         │            ● HNSW (high ef)
                         │       ● HNSW (default)
                     95% │  ● IVF (high nprobe)
                         │
                     90% │         ● IVF+PQ
                         │
                         └────────────────────────▶
                              Latency (ms/query)
                         1ms    10ms    50ms   100ms
```

---

## **2e. Retrieval**

### Dense Retrieval (Semantic)

The standard approach: embed the query with the same model, find nearest neighbors in vector space.

```python
# Dense retrieval with FAISS
query_embedding = model.encode([query]).astype(np.float32)
faiss.normalize_L2(query_embedding)
scores, indices = index.search(query_embedding, top_k)  # top-k nearest
```

**Strengths:** Captures semantic meaning ("heart attack" ↔ "myocardial infarction")
**Weakness:** Misses exact terms, IDs, formulas, rare tokens

### Sparse Retrieval (Lexical)

BM25 is the standard sparse retriever. It scores based on term frequency, inverse document frequency, and document length.

```
                           (k₁ + 1) · tf(t,d)                          N - df(t) + 0.5
BM25(q, d) = Σ  ──────────────────────────────────────  ×  log ────────────────────
             t∈q  tf(t,d) + k₁ · (1 - b + b · |d|/avgdl)              df(t) + 0.5
```

Where: `k₁ = 1.2`, `b = 0.75` (standard), `tf` = term frequency, `df` = document frequency, `N` = total docs

```python
from rank_bm25 import BM25Okapi

# Build BM25 index with lemmatization
tokenized_corpus = [lemmatize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Search
query_tokens = lemmatize(query)
scores = bm25.get_scores(query_tokens)
top_k_indices = np.argsort(-scores)[:k]
```

**Strengths:** Exact keyword matching, drug names, IDs, formulas, zero-shot on new domains
**Weakness:** No semantic understanding ("car" ≠ "automobile")

### SPLADE (Learned Sparse)

A neural model that produces sparse representations — combines the interpretability of sparse retrieval with learned semantic expansion. Tokens not in the vocabulary get zero weight; related terms get non-zero weight.

### Filtered Retrieval

Combine vector search with metadata filters for precision:

```python
# Pinecone-style filtered query
results = index.query(
    vector=query_embedding,
    filter={"source": "medical_textbook", "year": {"$gte": 2020}},
    top_k=10
)
```

**Pre-filtering** (filter then search) vs **post-filtering** (search then filter): Pre-filtering is faster but may miss relevant results if filters are too aggressive.

---

## **2f. Re-Ranking**

Initial retrieval casts a wide net (top-50 to top-100). Re-ranking applies a more expensive model to the shortlist to improve precision.

### Bi-Encoder vs Cross-Encoder

```
Bi-Encoder (retrieval):           Cross-Encoder (re-ranking):

  Query ──▶ [Encoder] ──▶ q_vec     Query ──┐
                                              ├──▶ [Encoder] ──▶ relevance score
  Doc   ──▶ [Encoder] ──▶ d_vec     Doc   ──┘
                                    
  score = dot(q_vec, d_vec)         score = model(concat(query, doc))
  
  ✓ Fast (precompute doc vectors)   ✓ Much more accurate
  ✗ Independent encoding            ✗ O(n) forward passes (can't precompute)
```

**Bi-encoders** encode query and document independently → fast but less accurate.
**Cross-encoders** process query-document pairs jointly → slow but highly accurate.

### Cross-Encoder Models

| Model | Base | Training Data | Speed | Quality |
|-------|------|--------------|-------|---------|
| `ms-marco-MiniLM-L-6-v2` | MiniLM | MS MARCO | Fast | Good; **used in Medical RAG** |
| `ms-marco-MiniLM-L-12-v2` | MiniLM | MS MARCO | Medium | Better |
| `monoT5-base` | T5 | MS MARCO | Slow | Excellent |
| `monoT5-3b` | T5-3B | MS MARCO | Very slow | SOTA |

### ColBERT (Late Interaction)

A middle ground: encode query and document independently but compute a **token-level** interaction at search time.

```
Query tokens:  [q₁, q₂, q₃, q₄]
Doc tokens:    [d₁, d₂, d₃, d₄, d₅]

Score = Σᵢ maxⱼ (qᵢ · dⱼ)    (MaxSim: each query token matches its best doc token)
```

- **Faster than cross-encoder** (precompute doc token embeddings)
- **More accurate than bi-encoder** (token-level matching)
- **Trade-off:** Higher storage (store per-token embeddings)

### Re-Ranking Pipeline (Medical RAG Example)

```python
class CrossEncoderReranker:
    def __init__(self):
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, windows: List[Dict]) -> List[Dict]:
        self._load_model()
        pairs = [(query, w["text"]) for w in windows]
        raw_scores = self._model.predict(pairs)
        
        # Normalize scores to [0, 1] with sigmoid
        normalized = 1 / (1 + np.exp(-raw_scores))
        
        for i, w in enumerate(windows):
            w["score_ce"] = float(normalized[i])
        
        return sorted(windows, key=lambda x: x["score_ce"], reverse=True)
```

---

## **2g. Context Engineering**

After retrieval and re-ranking, you must **engineer the context** that goes into the LLM prompt.

### MMR (Maximal Marginal Relevance)

Balances **relevance** to the query with **diversity** among selected documents:

```
MMR = argmax  [ λ · sim(dᵢ, q)  −  (1−λ) · max  sim(dᵢ, dⱼ) ]
      dᵢ∈R\S                          dⱼ∈S

Where:
  R = candidate set (retrieved docs)
  S = already selected docs
  λ = diversity parameter (0.5–0.7 typical)
  λ=1.0 → pure relevance, λ=0.0 → pure diversity
```

**Why MMR?** Without it, top-k chunks are often near-duplicates (paraphrases of the same fact). MMR ensures the LLM sees diverse evidence.

### Token Budgeting

The LLM has a finite context window. You must **budget** tokens carefully:

```
Context Window (e.g., 8192 tokens)
├── System prompt:          ~200 tokens
├── Conversation history:   ~500 tokens
├── Retrieved chunks:       ~4000 tokens  ← your budget
├── User query:             ~100 tokens
└── Generation headroom:    ~3392 tokens
```

**Rules of thumb:**
- Allocate 40–60% of context to retrieved chunks
- Truncate or compress chunks to fit budget
- Prefer fewer high-quality chunks over many low-quality ones

### Compression Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Map-reduce summarization** | Summarize each chunk, then summarize summaries | Long documents, broad queries |
| **Key-point extraction** | Extract bullet points from each chunk | Factual Q&A |
| **LLMLingua** | Learned token pruning | Token-constrained scenarios |
| **Chunk truncation** | Keep first N tokens per chunk | Simple, predictable |

### Context Assembly (PathWise Example)

```python
# PathWise: Retrieve top-6 chunks, assemble context for Groq LLM
q_embed = await generate_content_embedding(message)
sims = find_similar_content(q_embed, embeds, top_k=6)
top_indices = [i for i, _ in sims]
top_texts = [chunks[i] for i in top_indices if i < len(chunks)]
retrieval = "\n\n".join(top_texts)

llm_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": f"Relevant document context:\n{retrieval}"},
    *conversation_history[-10:],
    {"role": "user", "content": message}
]
```

---

## **2h. Generation**

### Cite Sources

Always instruct the LLM to cite which chunks it used:

```
System Prompt:
"Answer the question using ONLY the provided context. 
Cite your sources as [Doc X, Page Y]. 
If the context is insufficient, say 'I don't have enough information to answer this.'"
```

### Refuse When Insufficient Evidence

A RAG system that never refuses is a hallucination machine. Implement **refusal logic:**

1. **Retrieval confidence:** If max retrieval score < threshold → refuse
2. **LLM self-assessment:** Ask the model to rate its confidence
3. **Multi-layer detection:** Pattern matching for out-of-domain queries (used in Medical RAG)

### Generation Models

| Model | Context | Speed | Quality | Cost |
|-------|---------|-------|---------|------|
| GPT-4o | 128K | Fast | Excellent | $$$$ |
| GPT-4o-mini | 128K | Very fast | Good | $$ |
| Claude 3.5 Sonnet | 200K | Fast | Excellent | $$$ |
| Mistral 7B | 32K | Fast (local) | Good; **used in Medical RAG** | Free (self-hosted) |
| Llama 3.1 70B | 128K | Medium | Very good | Free (self-hosted) |
| Groq (Llama/Mixtral) | 32K | Ultra-fast | Good; **used in PathWise** | $ |

---

## **2i. Evaluation**

RAG evaluation has **two dimensions**: retrieval quality and generation quality.

### Retrieval Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Recall@k** | (relevant docs in top-k) / (total relevant docs) | Coverage — are all relevant docs found? |
| **Precision@k** | (relevant docs in top-k) / k | Signal-to-noise — are retrieved docs useful? |
| **MRR** (Mean Reciprocal Rank) | 1/\|Q\| × Σ 1/rank_i | How high is the first relevant result? |
| **nDCG@k** | DCG@k / IDCG@k | Ranked quality — are better docs ranked higher? |
| **Hit Rate@k** | queries with ≥1 relevant doc in top-k / total queries | Binary: did we find *something* useful? |

```
         Σᵢ (2^relᵢ - 1) / log₂(i + 1)
nDCG@k = ────────────────────────────────
         IDCG@k (ideal ordering)
```

### Generation Metrics

| Metric | What It Measures | Automated? |
|--------|-----------------|-----------|
| **Faithfulness** | Does the answer only contain info from retrieved context? | Yes (LLM-as-judge) |
| **Answer Relevancy** | Does the answer address the question? | Yes (LLM-as-judge) |
| **ROUGE-L** | Longest common subsequence overlap with reference | Yes |
| **F1** | Token-level precision/recall vs reference answer | Yes |
| **BERTScore** | Semantic similarity between answer and reference | Yes |
| **Human eval** | Expert judgment | No (gold standard) |

### Evaluation Frameworks

- **RAGAS** — Automated RAG evaluation (faithfulness, relevancy, context recall)
- **DeepEval** — Unit-test-style RAG evaluation
- **LangSmith** — Tracing + evaluation in LangChain ecosystem
- **Custom:** Log `(query, retrieved_chunks, answer, feedback)` → offline analysis

---

# **3. Hybrid Retrieval**

---

## **3.1 Why Hybrid?**

Dense retrieval excels at **semantic matching** but fails on:
- Exact terms: drug names ("metformin"), IDs ("CVE-2024-1234"), formulas
- Rare tokens: medical jargon, proper nouns not well-represented in embedding training data
- Short factoid queries: "What is the half-life of amoxicillin?"

Sparse retrieval (BM25) handles these perfectly but misses semantic paraphrases.

**Hybrid = best of both worlds.**

```
┌──────────────┐     ┌──────────────┐
│  Dense       │     │  Sparse      │
│  (FAISS)     │     │  (BM25)      │
│              │     │              │
│ "heart attack│     │ "myocardial  │
│  treatment"  │     │  infarction" │
│  ✓ finds MI  │     │  ✓ exact     │
│    content   │     │    match     │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
    ┌──────────────────────────┐
    │     FUSION / RRF         │
    │  Combine both rankings   │
    └────────────┬─────────────┘
                 │
                 ▼
          ┌────────────┐
          │  Top-k     │
          │  Results   │
          └────────────┘
```

## **3.2 Linear Combination**

The simplest fusion: weighted sum of normalized scores.

```
Score(d) = λ × Dense_score(d) + (1 − λ) × Sparse_score(d)

Where:
  λ ∈ [0, 1] — weighting parameter
  λ = 0.5 → equal weight (balanced)
  λ = 0.7 → favor dense (semantic queries)
  λ = 0.3 → favor sparse (keyword queries)
```

**Problem:** Dense and sparse scores are on different scales. You must **normalize** before combining:
- Min-max normalization: `score_norm = (score - min) / (max - min)`
- Z-score normalization: `score_norm = (score - mean) / std`

## **3.3 Adaptive Weighting**

Different queries benefit from different λ values:

| Query Type | Optimal λ (BM25 weight) | Reasoning |
|-----------|------------------------|-----------|
| Short keyword (≤3 words) | 0.35 | More semantic (dense) | 
| Question with `?` | 0.45 | Slightly more semantic |
| Long descriptive query | 0.55 | Balanced lexical/semantic |
| Exact ID / formula lookup | 0.70+ | Sparse dominates |

```python
# Adaptive alpha from Medical RAG project
def calculate_adaptive_alpha(query: str) -> float:
    query = query.strip()
    if len(query.split()) <= 3:
        return 0.35   # short → more dense (semantic)
    if "?" in query:
        return 0.45   # question → slightly more dense
    return 0.55       # default → balanced
```

---

# **4. Reciprocal Rank Fusion (RRF)**

---

## **4.1 Why RRF Over Linear Combination?**

Linear combination requires score normalization (different scales). RRF uses **ranks instead of scores**, making it scale-invariant.

## **4.2 Standard RRF Formula**

```
                    1
RRF(d) = Σ  ─────────────
         r   rank_r(d) + k

Where:
  r = each ranking system (BM25, dense, etc.)
  rank_r(d) = rank of document d in system r (1-indexed)
  k = smoothing constant (typically 60)
```

**Intuition:** A document ranked #1 in any system gets a high score. A document ranked #1 in *both* systems gets a very high score. The constant `k` prevents the score from being dominated by the top-1 result.

## **4.3 Weighted RRF (Used in Medical RAG)**

Extend RRF with adaptive weighting:

```
                     α                    (1 − α)
RRF(d) = ─────────────────── + ───────────────────────
         rank_BM25(d) + 60     rank_Dense(d) + 60

Where α = adaptive_alpha(query)
```

### Implementation (from Medical RAG)

```python
def search(self, query, k=6, bm25_n=50, faiss_n=50, alpha=None):
    # Adaptive alpha based on query type
    alpha = alpha or calculate_adaptive_alpha(query)
    
    # Get ranked results from both systems
    bm25_results = self._bm25_search_with_ranks(query, bm25_n)
    dense_results = self._dense_search_with_ranks(query, faiss_n)
    
    # Reciprocal Rank Fusion
    fused = defaultdict(float)
    
    for idx, score, rank in bm25_results:
        fused[idx] += alpha / (rank + 60)
    
    for idx, score, rank in dense_results:
        fused[idx] += (1 - alpha) / (rank + 60)
    
    # Return top-k by fused score
    sorted_windows = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return sorted_windows
```

## **4.4 RRF vs Linear Combination**

| Property | Linear Combination | RRF |
|----------|-------------------|-----|
| Input | Raw scores | Ranks |
| Normalization needed | Yes | No |
| Scale-invariant | No | Yes |
| Tuning complexity | Normalize + weight | Only weight (α) and k |
| Works with heterogeneous retrievers | Difficult | Easy |
| Standard in production | Less common | Very common (Elasticsearch, Pinecone) |

---

# **5. Cross-Encoder Re-Ranking vs MMR**

---

## **5.1 Comparison Table**

| Property | Cross-Encoder Re-Ranking | MMR (Maximal Marginal Relevance) |
|----------|-------------------------|----------------------------------|
| **Goal** | Maximize relevance | Balance relevance + diversity |
| **Input** | (query, doc) pairs | query + candidate set |
| **Output** | Relevance score per doc | Subset of diverse, relevant docs |
| **Computation** | O(n) forward passes through transformer | O(n²) pairwise similarity (precomputable) |
| **Latency** | ~10–50ms for 20 docs (MiniLM) | < 5ms (matrix operations) |
| **When to use** | Always (after initial retrieval) | When top-k chunks are redundant |
| **Position in pipeline** | After retrieval, before context assembly | After re-ranking, during context assembly |
| **Models** | ms-marco-MiniLM, monoT5 | No model needed (cosine + greedy selection) |

## **5.2 When to Use Each**

```
Retrieval (top-50)
    │
    ▼
Cross-Encoder Re-Ranking (top-50 → top-10)    ← Use ALWAYS for precision
    │
    ▼
MMR (top-10 → top-5 diverse)                  ← Use when chunks are redundant
    │
    ▼
Context Assembly → LLM
```

**Use cross-encoder when:** You need the most relevant documents, period. It is the single biggest quality improvement you can make to a RAG pipeline.

**Use MMR when:** Your corpus has many near-duplicate or paraphrased passages. Without MMR, 4 of your 5 context chunks might contain essentially the same information, wasting your token budget.

**Best practice:** Use both, in sequence. Re-rank for quality, then MMR for diversity.

---

# **6. Tuning Knobs**

---

Every RAG system has a set of hyperparameters that trade off quality, latency, and cost. Here is the complete reference:

## **6.1 Chunking Parameters**

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|---------------|
| Chunk size | 256–2048 tokens | Larger → more context per chunk, fewer chunks needed | 512–1024 for general; 256–512 for precise Q&A |
| Overlap | 0–25% | More → better boundary coverage, more storage | 10–20% (50–128 tokens) |
| Splitting strategy | Fixed / Semantic / Heading | Semantic > heading > fixed (usually) | Start with fixed + overlap, upgrade if quality lags |

## **6.2 Retrieval Parameters**

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|---------------|
| top-k (initial retrieval) | 10–100 | More → better recall, more noise | 30–50 for hybrid, re-rank to top-5–10 |
| top-k (after re-ranking) | 3–10 | More → more context for LLM | 5–6 (budget dependent) |
| Hybrid λ / α | 0.0–1.0 | 0 = pure dense, 1 = pure sparse | 0.3–0.5 for most; adaptive is best |
| BM25 k₁ | 1.0–2.0 | Term frequency saturation | 1.2 (standard) |
| BM25 b | 0.0–1.0 | Document length normalization | 0.75 (standard) |

## **6.3 HNSW Index Parameters**

| Parameter | Range | Impact |
|-----------|-------|--------|
| **M** (edges per node) | 8–64 | Higher → better recall, more RAM. Default: 16 |
| **efConstruction** | 64–512 | Higher → better graph quality, slower build. Default: 200 |
| **efSearch** | 16–512 | Higher → better recall at search time, slower queries. Default: 50–100 |

```
Recall@10 vs efSearch (typical):
  efSearch=16  → ~92% recall
  efSearch=64  → ~97% recall
  efSearch=200 → ~99% recall
  efSearch=512 → ~99.5% recall  (diminishing returns)
```

## **6.4 IVF Index Parameters**

| Parameter | Range | Impact |
|-----------|-------|--------|
| **nlist** (number of clusters) | √n to 4√n | More clusters → finer partitions. Typical: 100–10000 |
| **nprobe** (clusters to search) | 1–nlist | More → better recall, slower. Typical: 5–20% of nlist |

```
Example for 1M vectors:
  nlist = 1000  (√1M ≈ 1000)
  nprobe = 50   (5% of nlist) → ~96% recall
  nprobe = 100  (10% of nlist) → ~98% recall
```

## **6.5 Re-Ranking Parameters**

| Parameter | Range | Impact |
|-----------|-------|--------|
| Re-rank top-N | 10–100 | More candidates → better quality, higher latency |
| Score threshold | 0.0–1.0 | Filter low-confidence results |
| MMR λ | 0.0–1.0 | 1.0 = pure relevance, 0.0 = pure diversity. Typical: 0.5–0.7 |

## **6.6 Tuning Decision Tree**

```
Problem: Low answer quality
├── Relevant docs not in top-k?
│   ├── YES → Increase k, try hybrid search, check chunk size
│   └── NO → Docs are retrieved but LLM ignores them
│           ├── Too many chunks → Reduce k, use MMR
│           └── Chunks too long → Reduce chunk size, compress
│
Problem: High latency
├── Retrieval slow?
│   ├── YES → Switch from Flat to HNSW/IVF, reduce efSearch/nprobe
│   └── NO → Re-ranking slow?
│           ├── YES → Reduce re-rank candidates, use smaller cross-encoder
│           └── NO → LLM slow → Use smaller model or streaming
│
Problem: High RAM usage
├── Index too large?
│   ├── YES → Use PQ compression, reduce M in HNSW, use IVF
│   └── NO → Embedding storage → Use lower-dim model (384 vs 1024)
```

---

# **7. Recall–Latency–RAM Triangle**

---

Every indexing and retrieval decision is a trade-off between three resources:

```
                    RECALL
                   (quality)
                     /\
                    /  \
                   /    \
                  /      \
                 / Sweet   \
                /  Spot     \
               /    ★        \
              /               \
             /                 \
            /                   \
           ────────────────────── 
        LATENCY              RAM
        (speed)            (memory)
```

## **7.1 Trade-Off Matrix**

| Configuration | Recall | Latency | RAM | When to Use |
|---------------|--------|---------|-----|-------------|
| Flat + exact search | 100% | High (O(n)) | Full vectors | < 100K docs, batch offline |
| HNSW (M=32, ef=200) | ~99% | Low (O(log n)) | 1.5× vectors | Most production systems |
| HNSW (M=8, ef=32) | ~93% | Very low | 1.1× vectors | Extreme latency constraints |
| IVF (nprobe=10%) | ~97% | Medium | Full vectors + centroids | 1M+ docs, moderate latency OK |
| IVF + PQ | ~90% | Medium | 4–32× smaller | Billions of docs, RAM constrained |
| Flat + GPU | 100% | Very low | GPU VRAM | < 10M vectors, GPU available |

## **7.2 Practical Guidance**

| Scale | Recommended Index | Expected Perf |
|-------|------------------|---------------|
| < 10K docs | Flat (brute-force) | < 1ms, 100% recall |
| 10K–1M docs | HNSW (M=16, ef=128) | 1–5ms, ~98% recall |
| 1M–100M docs | IVF-HNSW or IVF-PQ | 5–20ms, ~95% recall |
| 100M+ docs | DiskANN, Milvus, Pinecone | 10–50ms, ~93% recall |

---

# **8. Advanced RAG Patterns**

---

## **8.1 Query Decomposition**

Complex questions are broken into sub-questions, each answered independently, then synthesized.

```
User: "Compare the treatment protocols for tension pneumothorax 
       in adults vs pediatric patients."

Decomposed:
  Q1: "What is the treatment protocol for tension pneumothorax in adults?"
  Q2: "What is the treatment protocol for tension pneumothorax in pediatric patients?"
  Q3: "What are the key differences between adult and pediatric treatment?"

Each sub-query → Retrieve → Answer → Synthesize final answer
```

**When to use:** Multi-part questions, comparison queries, questions requiring information from multiple sections.

## **8.2 Multi-Hop RAG**

Some questions require chaining multiple retrieval steps:

```
Q: "What company did the CEO of the firm that acquired Twitter work for before?"

Hop 1: "What firm acquired Twitter?" → X Corp / Elon Musk
Hop 2: "What company did Elon Musk work for before?" → Zip2 / PayPal

Each hop retrieves context for the next hop.
```

### Implementation Pattern

```python
async def multi_hop_rag(query: str, max_hops: int = 3):
    context = ""
    for hop in range(max_hops):
        # Retrieve based on current query + accumulated context
        retrieved = retrieve(query, context)
        context += format_chunks(retrieved)
        
        # Ask LLM if it can answer or needs more info
        result = llm_generate(query, context)
        if result.is_sufficient:
            return result.answer
        
        # Generate follow-up query for next hop
        query = result.follow_up_query
    
    return "Could not find sufficient information."
```

## **8.3 Self-RAG (Self-Reflective RAG)**

The model decides **when** to retrieve and **critiques** its own outputs:

```
┌──────────────┐
│   User Query  │
└──────┬───────┘
       ▼
┌──────────────────────┐
│ Should I retrieve?    │ ← LLM self-assessment
│  (Retrieve / No-Ret)  │
└──────┬───────────────┘
       │ If Retrieve:
       ▼
┌──────────────────────┐
│ Retrieve → Generate   │
│ with context          │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ Is this supported?    │ ← Critique token
│  (Supported / Not)    │
└──────┬───────────────┘
       │ If Supported:
       ▼
┌──────────────────────┐
│ Is this useful?       │ ← Utility token
│  (Score 1-5)          │
└──────────────────────┘
```

**Key insight:** Self-RAG trains special tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) into the model during fine-tuning. The model learns to be its own quality gatekeeper.

## **8.4 CRAG (Corrective RAG)**

CRAG adds a **confidence evaluator** after retrieval:

```
Query → Retrieve → Evaluate confidence
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          Correct    Ambiguous  Incorrect
          (conf>0.7) (0.3-0.7) (conf<0.3)
              │         │         │
              ▼         ▼         ▼
          Use as-is   Refine    Web search
                     (extract   or refuse
                      key info)
```

**When to use:** When retrieval quality is inconsistent and you need a safety net against poor retrieval.

## **8.5 Agentic RAG**

Combine RAG with tool-using agents:

```
Agent receives query → Decides:
  1. "I need to search the knowledge base" → RAG retrieval
  2. "I need to call an API" → Tool call
  3. "I need to run a calculation" → Code execution
  4. "I can answer directly" → Generate from memory
  5. "I need multiple sources" → Parallel RAG + API
```

The agent orchestrates multiple retrieval strategies, tools, and reasoning steps dynamically.

---

# **9. Production Considerations**

---

## **9.1 Caching**

| Cache Layer | What to Cache | TTL | Impact |
|-------------|--------------|-----|--------|
| **Query embedding cache** | `hash(query) → embedding` | 1 hour | Avoid re-embedding identical queries |
| **Result cache** | `hash(query) → retrieved_chunks` | 5–15 min | Skip retrieval entirely for repeated queries |
| **LLM response cache** | `hash(query + context) → answer` | 5–15 min | Skip generation (biggest cost saver) |
| **Cross-encoder cache** | `hash(query, doc) → score` | 1 hour | LRU cache for repeated re-rankings |

```python
from functools import lru_cache
import hashlib

# LRU cache for cross-encoder (used in Medical RAG)
@lru_cache(maxsize=1024)
def cached_rerank(query_hash: str, doc_hash: str) -> float:
    return cross_encoder.predict([(query, doc)])[0]
```

## **9.2 Async Architecture**

For production throughput, make retrieval and re-ranking async:

```python
async def rag_pipeline(query: str):
    # 1. Embed query (async API call)
    query_embedding = await embed_async(query)
    
    # 2. Parallel retrieval (BM25 + dense simultaneously)
    bm25_results, dense_results = await asyncio.gather(
        bm25_search_async(query),
        dense_search_async(query_embedding)
    )
    
    # 3. Fuse results
    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    
    # 4. Re-rank (run blocking model in thread pool)
    reranked = await asyncio.to_thread(cross_encoder.rerank, query, fused[:20])
    
    # 5. Generate (async LLM call)
    answer = await llm_generate_async(query, reranked[:5])
    return answer
```

## **9.3 Error Handling**

| Failure Mode | Mitigation |
|-------------|------------|
| Embedding API down | Fallback embeddings (TF-IDF hash or cached) |
| Vector DB timeout | Circuit breaker + cached results |
| LLM rate limit | Retry with exponential backoff + secondary key |
| Empty retrieval | Graceful refusal ("I don't have relevant information") |
| Cross-encoder OOM | Reduce batch size, use smaller model |
| Corrupt index | Health check on startup, rebuild from embeddings |

### PathWise Error Handling (Production Pattern)

```python
async def _call_groq_with_key(self, messages, api_key, key_type):
    max_retries = 3
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            res = await client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                # Extract retry-after from error message
                wait_s = parse_retry_after(e) or backoff
                await asyncio.sleep(wait_s)
                backoff *= 2
                continue
            return ""
        except httpx.TimeoutException:
            await asyncio.sleep(backoff)
            backoff *= 2
            continue
    return ""

# Dual API key fallback
result = await self._call_groq_with_key(messages, self.groq_api_key, "primary")
if not result and self.groq_api_key_2:
    result = await self._call_groq_with_key(messages, self.groq_api_key_2, "fallback")
```

## **9.4 Monitoring & Observability**

### Key Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Retrieval latency (p50/p95) | < 50ms / < 200ms | > 500ms |
| Re-ranking latency (p50/p95) | < 30ms / < 100ms | > 200ms |
| End-to-end latency (p50/p95) | < 2s / < 5s | > 10s |
| Retrieval recall (sampled) | > 85% | < 70% |
| Faithfulness score (sampled) | > 0.8 | < 0.6 |
| Refusal rate | 5–15% | > 30% or < 1% |
| Cache hit rate | > 40% | < 20% |
| Embedding API errors | < 0.1% | > 1% |

### Logging Pipeline

```
Query → [Log: query, timestamp, user_id]
  → Retrieval → [Log: chunk_ids, scores, latency]
    → Re-ranking → [Log: reranked_ids, ce_scores, latency]
      → Generation → [Log: answer, token_count, model, latency]
        → User feedback → [Log: thumbs_up/down, correction]
```

## **9.5 Index Updates & Freshness**

| Strategy | Description | Use When |
|----------|-------------|----------|
| **Full rebuild** | Re-index everything | Small corpus (< 100K docs) |
| **Incremental** | Add/update only changed docs | Medium corpus, frequent updates |
| **Streaming** | Real-time upsert as docs arrive | Large corpus, real-time requirements |
| **Versioned** | Blue-green index swap | Zero-downtime updates |

---

# **10. Rahul's Project Deep-Dives**

---

## **10.1 PathWise — AI Learning Platform RAG**

### System Overview

PathWise is a production AI-powered learning platform where users upload PDFs and chat with their documents. The RAG system enables contextual Q&A, lesson generation, quiz creation, and flashcard generation — all grounded in the uploaded material.

### Architecture

```
┌────────────┐     ┌──────────────┐     ┌──────────────────┐
│ User Upload │────▶│  PyMuPDF     │────▶│  chunk_text()    │
│ (PDF)       │     │  (fitz)      │     │  400 words/chunk │
└────────────┘     │  pdf_to_text │     │  50 word overlap │
                   └──────────────┘     └────────┬─────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  Cohere API      │
                                        │  embed-english-  │
                                        │  light-v3.0      │
                                        │  384 dimensions  │
                                        └────────┬─────────┘
                                                  │
                   ┌──────────────────────────────┘
                   │
                   ▼                        ┌──────────────────┐
          ┌──────────────────┐              │  User Query      │
          │  In-Memory Store │              │                  │
          │  chunks[]        │◀─────────────│  embed query via │
          │  chunk_embeds[]  │   top-6 by   │  Cohere (input_  │
          └──────────────────┘   cosine sim │  type="search_   │
                   │                        │  query")         │
                   │                        └──────────────────┘
                   ▼
          ┌──────────────────┐
          │  Groq LLM API    │
          │  (Llama / Mixtral│
          │  via Groq)       │
          │                  │
          │  System prompt + │
          │  retrieved chunks│
          │  + conversation  │
          │  history (last   │
          │  10 messages)    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Response:       │
          │  • Chat answer   │
          │  • Lesson (map-  │
          │    reduce summ.) │
          │  • Quiz / Cards  │
          └──────────────────┘
```

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | Cohere `embed-english-light-v3.0` (384d) | Lightweight, fast API, good quality-cost ratio |
| Chunk size | 400 words (~512 tokens) | Balanced context vs granularity |
| Overlap | 50 words (~12.5%) | Preserves context at boundaries |
| Similarity | Cosine similarity (manual dot product) | Standard for normalized embeddings |
| Top-k | 6 chunks | Fits in Groq context window with headroom |
| Summarization | Map-reduce (chunk → summary → merge) | Handles long PDFs without token overflow |
| LLM | Groq (Llama/Mixtral) | Ultra-fast inference, free/cheap tier |
| Error handling | Dual API key fallback + exponential backoff | Production reliability |
| Conversation | Last 10 messages in context | Balance between memory and token budget |

### What I'd Improve (Interview Talking Points)

1. **Move to vector DB:** Current in-memory store doesn't persist across restarts → FAISS or pgvector
2. **Add hybrid search:** BM25 for exact terms (code snippets, formulas in PDFs)
3. **Add re-ranking:** Cross-encoder would significantly improve precision
4. **Evaluation pipeline:** Automated faithfulness scoring with RAGAS
5. **Chunk by headings:** PDF structure-aware chunking instead of fixed window

---

## **10.2 Medical RAG — MARCH-PAWS Clinical Decision Support**

### System Overview

A sophisticated medical RAG system for the MARCH-PAWS trauma care protocol. Processes a medical textbook into a searchable knowledge base, generates stage-specific clinical questions, and produces evidence-based answers with citations.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MEDICAL RAG PIPELINE                              │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │ Medical       │──── Section Parsing ──── Window Creation ────┐    │
│  │ Textbook      │     (JSONL paragraphs)   (multi-para windows │    │
│  │ (TC4-02)      │                           with stride)       │    │
│  └──────────────┘                                               │    │
│                                                                  │    │
│         ┌───────────────────────────────────────────────────────┘    │
│         │                                                            │
│         ▼                              ▼                             │
│  ┌──────────────┐              ┌──────────────┐                     │
│  │ all-MiniLM-  │              │ rank-bm25    │                     │
│  │ L6-v2        │              │ BM25Okapi    │                     │
│  │ 384-dim      │              │ + NLTK       │                     │
│  │ embeddings   │              │ lemmatize    │                     │
│  └──────┬───────┘              └──────┬───────┘                     │
│         │                              │                             │
│         ▼                              ▼                             │
│  ┌──────────────┐              ┌──────────────┐                     │
│  │ FAISS        │              │ BM25 Index   │                     │
│  │ IndexFlatIP  │              │ (pickle)     │                     │
│  │ (L2-norm →   │              │ k₁=1.2      │                     │
│  │  cosine)     │              │ b=0.75       │                     │
│  └──────┬───────┘              └──────┬───────┘                     │
│         │                              │                             │
│         └──────────┬───────────────────┘                             │
│                    ▼                                                  │
│         ┌────────────────────┐                                       │
│         │  Reciprocal Rank   │                                       │
│         │  Fusion (RRF)      │                                       │
│         │  α / (rank + 60)   │                                       │
│         │  Adaptive α:       │                                       │
│         │   short→0.35       │                                       │
│         │   question→0.45    │                                       │
│         │   default→0.55     │                                       │
│         └────────┬───────────┘                                       │
│                  ▼                                                    │
│         ┌────────────────────┐                                       │
│         │  Cross-Encoder     │                                       │
│         │  Re-Ranking        │                                       │
│         │  ms-marco-MiniLM   │                                       │
│         │  -L-6-v2           │                                       │
│         │  Window → Para     │                                       │
│         │  drill-down        │                                       │
│         └────────┬───────────┘                                       │
│                  ▼                                                    │
│         ┌────────────────────┐                                       │
│         │  Scenario-Aware    │                                       │
│         │  Filtering         │                                       │
│         │  (MARCH-PAWS FSM) │                                       │
│         │  Stage-specific    │                                       │
│         │  query rewriting   │                                       │
│         └────────┬───────────┘                                       │
│                  ▼                                                    │
│         ┌────────────────────┐                                       │
│         │  Mistral 7B        │                                       │
│         │  (via Ollama)      │                                       │
│         │  T=0.0 (questions) │                                       │
│         │  T=0.3 (answers)   │                                       │
│         │  32K context       │                                       │
│         └────────┬───────────┘                                       │
│                  ▼                                                    │
│         ┌────────────────────┐                                       │
│         │  Clinical Q&A      │                                       │
│         │  + Citations       │                                       │
│         │  + Refusal Logic   │                                       │
│         │  (4-layer OOD      │                                       │
│         │   detection)       │                                       │
│         └────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `all-MiniLM-L6-v2` (384d) | Lightweight, local inference, good with medical terms |
| Dense index | FAISS `IndexFlatIP` + L2 normalization | Exact cosine search at laptop scale |
| Sparse index | BM25Okapi + NLTK lemmatization | Exact medical term matching |
| Fusion | Weighted RRF (adaptive α, k=60) | Scale-invariant, query-adaptive |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, accurate, LRU-cached |
| Re-rank pipeline | Window → rerank → drill to paragraphs → rerank again | Two-stage re-ranking for precision |
| LLM | Mistral 7B via Ollama | Local, private, 32K context, free |
| Temperatures | 0.0 for Q-gen, 0.3 for A-gen | Deterministic questions, slightly creative answers |
| Refusal | 4-layer OOD detection (regex → embedding → BM25 z-score → LLM) | 100% accuracy on non-medical queries |
| Architecture | Async orchestrator with background pre-fetch | Low-latency UX, parallel retrieval |

### Retrieval Pipeline Detail

```
Query: "What is the treatment for tension pneumothorax?"
  │
  ├─ BM25 search (50 candidates):
  │   Lemmatize: ["treatment", "tension", "pneumothorax"]
  │   → Exact keyword matches on medical terms
  │
  ├─ FAISS search (50 candidates):
  │   Embed query → L2-normalize → IndexFlatIP search
  │   → Semantic matches (synonyms, related concepts)
  │
  ├─ RRF Fusion (adaptive α = 0.45 for question):
  │   fused[idx] += 0.45 / (bm25_rank + 60) + 0.55 / (dense_rank + 60)
  │   → Top-6 windows by fused score
  │
  ├─ Z-score threshold check:
  │   If max BM25 score is statistically significant → proceed
  │   Otherwise → low-confidence refusal
  │
  ├─ Cross-encoder re-ranking (top-20 → top-15):
  │   Score each (query, window_text) pair
  │   Sigmoid normalize scores
  │
  ├─ Smart paragraph extraction:
  │   From top-15 windows → extract all child paragraphs
  │   Re-score paragraphs with cross-encoder
  │   Return top-10 paragraphs
  │
  └─ Generate answer with Mistral 7B:
      Context = top-10 paragraphs + stage definition
      → Clinical answer with citations + checklist
```

---

# **11. Common Interview Questions**

---

## **Q1: "How would you build a RAG system for 1,000 internal documents?"**

**Framework answer:**

> "I'd approach this in five phases:
> 
> **1. Ingest:** Parse documents (PDFs via PyMuPDF, HTML via trafilatura) into text + metadata (title, section, page, source). Clean and deduplicate.
> 
> **2. Chunk & Embed:** Split into 512-token chunks with 10% overlap. For 1K docs, that's roughly 10K–50K chunks. Embed with a model like Cohere embed-v3 or E5-large. At this scale, a flat FAISS index is fine — exact search on 50K vectors takes < 5ms.
> 
> **3. Hybrid Retrieval:** BM25 for exact keyword matching plus dense search for semantic. Fuse with RRF. This handles both 'show me policy #XYZ' and 'what's our vacation policy' queries.
> 
> **4. Re-rank & Generate:** Cross-encoder re-ranking on top-20 candidates, keep top-5. Feed into GPT-4o-mini or Claude with citations. Instruct the model to refuse when evidence is insufficient.
> 
> **5. Evaluate & Iterate:** Sample 50–100 query-answer pairs, measure faithfulness and recall@5. Set up logging to catch bad retrievals. Iterate on chunk size and hybrid weights.
> 
> In my PathWise project, I built exactly this pipeline — Cohere embeddings, cosine similarity retrieval, top-6 chunks, Groq LLM — and it handled real user-uploaded PDFs in production."

---

## **Q2: "How do you handle hallucination in RAG?"**

> "Hallucination in RAG comes from three sources, each with a mitigation:
> 
> **1. Retrieval failure** (wrong docs retrieved) → Improve retrieval: hybrid search, re-ranking, better chunking. In my Medical RAG, I used FAISS + BM25 hybrid with cross-encoder re-ranking to maximize retrieval precision.
> 
> **2. Context ignored** (LLM generates from parametric memory) → Prompt engineering: explicitly instruct 'Answer ONLY from the provided context. If not found, say so.' Use structured prompts with clear delimiter between context and instructions.
> 
> **3. Over-generation** (LLM extrapolates beyond context) → Post-generation verification: check if answer tokens appear in retrieved context (faithfulness score). Use Self-RAG or CRAG for automatic critique. Implement refusal logic — in my Medical RAG, I built a 4-layer out-of-domain detection system with 100% accuracy on non-medical queries.
> 
> The nuclear option is **constrained decoding** where you only allow the LLM to generate tokens that appear in the context, but this is usually too restrictive."

---

## **Q3: "Explain dense vs sparse retrieval trade-offs."**

> "**Dense retrieval** (embedding-based) captures semantic similarity. 'Heart attack' matches 'myocardial infarction.' But it struggles with exact terms — drug names, IDs, formulas — because these get lost in the embedding space.
> 
> **Sparse retrieval** (BM25) does exact keyword matching. It perfectly finds 'metformin 500mg' but can't match 'diabetes medication' to the same document.
> 
> The solution is **hybrid retrieval**. In my Medical RAG, I combined FAISS (dense) with BM25 (sparse) using Reciprocal Rank Fusion. I used adaptive alpha weighting — short queries favor dense (α=0.35), longer queries balance both (α=0.55). This gave the best of both worlds: semantic understanding plus exact term matching."

---

## **Q4: "What is hybrid search and why do you need it?"**

> "Hybrid search combines dense (semantic) and sparse (lexical) retrieval. You need it because neither alone is sufficient:
> 
> Dense misses: exact IDs, code snippets, formulas, proper nouns, rare medical terms.
> Sparse misses: paraphrases, synonyms, conceptual matches.
> 
> The combination is fused via RRF (Reciprocal Rank Fusion) or linear weighted scoring. RRF is preferred because it uses ranks instead of scores, so you don't need to normalize across different score distributions.
> 
> I implemented this in my Medical RAG with `RRF(d) = α/(rank_BM25 + 60) + (1-α)/(rank_FAISS + 60)`, where α adapts based on query type. This outperformed either retriever alone on our evaluation set."

---

## **Q5: "How do you evaluate a RAG system?"**

> "I evaluate on two axes: **retrieval quality** and **generation quality**.
> 
> **Retrieval:** Recall@k (are the relevant chunks in top-k?), MRR (how high is the first relevant result?), nDCG (is the ranking quality good?). I typically test on a held-out set of 50–100 queries with known relevant documents.
> 
> **Generation:** Faithfulness (does the answer only use info from context?), Answer Relevancy (does it address the question?), ROUGE-L/F1 against reference answers. For automated evaluation, I use RAGAS or LLM-as-judge with GPT-4.
> 
> **End-to-end:** User satisfaction (thumbs up/down), refusal rate (should be 5–15%), latency percentiles (p50 < 2s, p95 < 5s).
> 
> The most important metric is **faithfulness** — a RAG system that hallucinates is worse than no system at all."

---

## **Q6: "Walk me through how you'd optimize a RAG system that has poor answer quality."**

> "I'd debug systematically:
> 
> **Step 1: Is it a retrieval problem or a generation problem?** Look at the retrieved chunks for failing queries. If the right information isn't in the chunks, it's retrieval. If it is but the answer is wrong, it's generation.
> 
> **Step 2: If retrieval:** Try hybrid search if not already using it. Tune chunk size (smaller for precise Q&A, larger for summary). Add a cross-encoder re-ranker — this is usually the single biggest quality improvement. Check if metadata filtering could help.
> 
> **Step 3: If generation:** Improve the prompt — be more explicit about citing sources and refusing when unsure. Try a stronger model. Use MMR to reduce redundancy in context. Check if the context is too long (LLMs lose focus in the middle — the 'lost in the middle' problem).
> 
> **Step 4: Systematic evaluation.** Build a test set, measure recall@5 and faithfulness, iterate."

---

## **Q7: "What is the 'lost in the middle' problem in RAG?"**

> "Research shows that LLMs attend most to information at the **beginning** and **end** of the context window, with reduced attention to the middle. This means if your most relevant chunk is placed in position 3 of 6, the model might underweight it.
> 
> **Mitigations:**
> 1. Put the most relevant chunks first (already done by re-ranking).
> 2. Use fewer, higher-quality chunks instead of many mediocre ones.
> 3. Some systems reverse the order (least relevant first, most relevant last) so the most relevant is near the end.
> 4. Use map-reduce: summarize each chunk independently, then synthesize — avoids the positional bias entirely.
> 
> In my PathWise project, I used map-reduce summarization for exactly this reason when generating lessons from long PDFs."

---

## **Q8: "How would you scale a RAG system from 1K to 10M documents?"**

> "Several things change:
> 
> **Indexing:** Move from Flat (exact, O(n)) to HNSW (approximate, O(log n)). For 10M 384-dim vectors, HNSW with M=16 uses ~30GB RAM and serves queries in 1–5ms with ~98% recall.
> 
> **Storage:** Move from in-memory to a managed vector DB (Pinecone, Weaviate, Qdrant). Enables horizontal scaling, filtered search, and persistence.
> 
> **Ingestion:** Batch processing pipeline (Airflow/Prefect) for incremental updates. Don't re-embed unchanged documents.
> 
> **Retrieval:** Metadata pre-filtering becomes critical — searching 10M vectors for a query about 'Q3 2024 financials' is wasteful if you can filter to the 10K finance documents first.
> 
> **Caching:** Embedding cache, result cache, and LLM response cache become essential at scale. A 40% cache hit rate can halve your infrastructure costs.
> 
> **Evaluation:** Automated monitoring — track retrieval recall and faithfulness on a rolling sample. Alert when quality degrades (e.g., after a big document ingestion)."

---

## **Q9: "Compare FAISS, Pinecone, and pgvector. When would you use each?"**

> "**FAISS:** Library, not a service. Best for local/self-hosted deployments where you need full control. Supports IVF, HNSW, PQ, GPU. I used it in my Medical RAG because we needed local inference (medical data privacy) and the scale was manageable (~10K windows).
> 
> **Pinecone:** Fully managed SaaS. Best for teams that want zero infrastructure overhead. Serverless option is cost-effective for sporadic traffic. Pre-filtering on metadata is native. Use when you want to ship fast and don't need self-hosting.
> 
> **pgvector:** PostgreSQL extension. Best when you already use Postgres and want vector search alongside relational queries in a single database. Great for hybrid transactional + analytical workloads. HNSW support added recently. Use when your data is already in Postgres and scale is < 5M vectors."

---

# **12. Key Takeaways**

---

1. **RAG decouples knowledge from reasoning.** The retrieval corpus is your knowledge base; the LLM is your reasoning engine. Update knowledge by re-indexing, not re-training.

2. **Chunking is the most impactful design decision.** Bad chunks → bad retrieval → bad answers. Start with 512 tokens + 10% overlap, iterate from there.

3. **Hybrid search (dense + sparse) beats either alone.** Dense captures semantics, sparse captures exact terms. Use RRF for scale-invariant fusion.

4. **Cross-encoder re-ranking is the single biggest quality improvement.** It's cheap (20 forward passes on top-20 candidates) and dramatically improves precision.

5. **Evaluate both retrieval and generation.** Recall@k for retrieval, faithfulness for generation. A RAG system that hallucinates is worse than no system at all.

6. **The recall–latency–RAM triangle governs all index decisions.** Flat for < 100K, HNSW for < 10M, IVF+PQ for billions.

7. **Advanced patterns (Self-RAG, CRAG, multi-hop, query decomposition) solve specific failure modes.** Don't add complexity until you've measured that you need it.

8. **Production RAG needs caching, async, error handling, and monitoring.** The retrieval pipeline is only half the work.

9. **Always build a refusal mechanism.** A system that says "I don't know" when appropriate is more trustworthy than one that always answers.

10. **Start simple, measure, iterate.** The best RAG system is the one that serves your users well, not the one with the most components.

---

*Rahul's edge: Built two production RAG systems with distinct architectures — a lightweight Cohere-based system for education (PathWise) and a sophisticated hybrid search + cross-encoder system for medical domain (MARCH-PAWS). Can speak to trade-offs from real deployment experience, not just theory.*

---
