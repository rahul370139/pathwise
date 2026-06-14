# Context Engineering — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, RAG Systems, Context Window Optimization, Production ML

---

# Table of Contents

1. [What Is Context Engineering?](#1-what-is-context-engineering)
2. [The 5-Step Recipe](#2-the-5-step-recipe)
3. [Context Window Management](#3-context-window-management)
4. [Compression Techniques](#4-compression-techniques)
5. [KV-Cache and Memory](#5-kv-cache-and-memory)
6. [Advanced Techniques](#6-advanced-techniques)
7. [MVP Architecture — Single T4 16GB Node](#7-mvp-architecture--single-t4-16gb-node)
8. [Production Pipeline](#8-production-pipeline)
9. [Prompt Engineering vs Context Engineering](#9-prompt-engineering-vs-context-engineering)
10. [Common Interview Questions](#10-common-interview-questions-with-strong-answers)
11. [Key Takeaways](#11-key-takeaways)

---

# **1. What Is Context Engineering?**

---

## **1.1 The Core Idea**

Context Engineering is the discipline of **selecting, compressing, ordering, and injecting the right tokens** into a language model's context window so the model produces the best possible output — given its finite attention budget.

Think of it this way:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        THE TOY-BOX ANALOGY                              │
│                                                                         │
│   A child (the LLM) can only hold a few toys at once (context window)  │
│                                                                         │
│   Context Engineering = choosing which toys to put in the child's hands │
│                                                                         │
│   Too many toys          → overwhelmed, drops important ones            │
│   Wrong toys             → plays the wrong game entirely                │
│   Right toys, right order → builds exactly what you asked for           │
│                                                                         │
│   Your job: curate the toy box so the child succeeds every time.       │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Open with the toy-box analogy. It immediately signals you understand that LLM quality is **bottlenecked by what goes into the context**, not just the model weights.

---

## **1.2 Formal Definition**

| Term | Definition |
|------|-----------|
| **Context Engineering** | The art and science of curating, compressing, and structuring the **retrieved information** (chunks, memories, tool outputs) that occupy the context window — maximizing relevance per token. |
| **Context Window** | The fixed-size input buffer (measured in tokens) a model can attend to during a single forward pass. |
| **Token Budget** | The allocation plan that divides the context window among system prompt, retrieved chunks, user query, and generation buffer. |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      CONTEXT WINDOW LAYOUT                               │
│                                                                          │
│  ┌──────────┐  ┌─────────────────────────┐  ┌───────┐  ┌─────────────┐  │
│  │  System   │  │   Retrieved Chunks      │  │ User  │  │ Generation  │  │
│  │  Prompt   │  │   (Context-Engineered)  │  │ Query │  │   Buffer    │  │
│  │  ~500 tok │  │   ~variable             │  │ ~200  │  │  ~1000 tok  │  │
│  └──────────┘  └─────────────────────────┘  └───────┘  └─────────────┘  │
│  ◄──────────────── Total Context Window (e.g., 8192 tokens) ──────────► │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **1.3 Context Engineering ≠ Prompt Engineering**

This distinction trips up many candidates:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   PROMPT ENGINEERING          CONTEXT ENGINEERING                    │
│   ─────────────────           ───────────────────                    │
│                                                                      │
│   "How you ASK"              "What you FEED"                         │
│                                                                      │
│   Designing the query,        Curating, compressing, and ordering   │
│   instructions, few-shot      the retrieved chunks, memories,       │
│   examples, CoT triggers      tool outputs that fill the window     │
│                                                                      │
│   ┌──────────────┐            ┌──────────────┐                       │
│   │  "Summarize  │            │  chunk_1.txt  │                      │
│   │   the key    │            │  chunk_5.txt  │                      │
│   │   findings"  │            │  memory_ctx   │                      │
│   └──────────────┘            └──────────────┘                       │
│         ↓                           ↓                                │
│   Goes into the               Goes into the                         │
│   QUERY portion               CONTEXT portion                       │
│                                                                      │
│   BOTH work together inside the same context window.                │
└──────────────────────────────────────────────────────────────────────┘
```

| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|-------------------|
| **What** | Query design, instructions, CoT | Chunk selection, compression, ordering |
| **Analogy** | Writing the exam question | Choosing which textbook pages to hand out |
| **Scope** | ~200-500 tokens | ~2000-100k+ tokens |
| **Tools** | Few-shot, CoT, role prompts | Retrievers, re-rankers, compressors, token budgets |
| **Failure mode** | Bad question → bad answer | Wrong context → hallucination |

> **Key insight:** You can write a perfect prompt, but if you feed garbage context, the model will hallucinate. Context engineering is the **higher-leverage** problem in RAG.

---

# **2. The 5-Step Recipe**

---

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    THE 5-STEP CONTEXT ENGINEERING RECIPE                  │
│                                                                          │
│   ① CHUNK  →  ② EMBED & INDEX  →  ③ RETRIEVE  →  ④ RE-RANK  →  ⑤ PACK │
│                                     & COMPRESS       & INJECT            │
│                                                                          │
│   Raw Docs     Vector DB         Top-K Candidates  Scored &     Final   │
│   → Snippets   (FAISS/PG/Pine)  (Semantic+BM25)   Compressed   Prompt  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **Step 1: Chunk — Split Documents into Overlapping Snippets**

### Why Chunk?

LLMs can't consume entire 500-page PDFs. Chunking breaks documents into digestible pieces that can be individually embedded, indexed, and retrieved.

### Chunking Strategies

| Strategy | How It Works | Best For | Chunk Size |
|----------|-------------|----------|------------|
| **Fixed-size** | Split every N tokens with M overlap | Simple docs, logs | 256-512 tokens |
| **Recursive Character** | Split on `\n\n` → `\n` → `. ` → ` ` hierarchy | General-purpose | 300-600 tokens |
| **Semantic Splitting** | Embed sentences, split where cosine similarity drops | Technical docs, mixed-topic | Variable |
| **Document-aware** | Respect headings, tables, code blocks | Markdown, HTML, code | Variable |
| **Agentic / Proposition** | LLM extracts atomic propositions per chunk | Knowledge-dense text | 1-3 sentences |

### The 15% Overlap Rule

```
┌─────────────────────────────────────────────────────────────────┐
│                     OVERLAP VISUALIZATION                        │
│                                                                  │
│   Document: [A A A A A B B B B B C C C C C D D D D D]          │
│                                                                  │
│   Chunk 1:  [A A A A A B B]                                     │
│   Chunk 2:          [A B B B B B C C]     ← 15% overlap         │
│   Chunk 3:                  [B C C C C C D D]                   │
│   Chunk 4:                          [C D D D D D]               │
│                                                                  │
│   Overlap ensures no information is lost at chunk boundaries.   │
│   15% is the empirical sweet spot: enough continuity,           │
│   not so much that you duplicate excessively.                   │
└─────────────────────────────────────────────────────────────────┘
```

### Code Example — Semantic Chunking with LangChain

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Semantic chunker: splits where meaning shifts
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # split at 95th-percentile dissimilarity
    breakpoint_threshold_amount=95,
)

chunks = chunker.create_documents([raw_document_text])
print(f"Created {len(chunks)} semantic chunks")
# Typical: 500-page PDF → 800-1500 chunks
```

```python
# Alternative: Recursive with 15% overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # tokens
    chunk_overlap=77,         # ~15% of 512
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,      # swap with tiktoken for exact token count
)
chunks = splitter.split_documents(documents)
```

> **Interview tip:** Mention semantic chunking to stand out, but know that recursive character splitting is the production workhorse (faster, predictable sizes, good enough for 90% of use cases).

---

## **Step 2: Embed & Index — Store Vectors for Fast Retrieval**

### Embedding Models

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| `text-embedding-3-small` (OpenAI) | 1536 | Fast | Good | $0.02/1M tokens |
| `text-embedding-3-large` (OpenAI) | 3072 | Medium | Excellent | $0.13/1M tokens |
| `all-MiniLM-L6-v2` (open-source) | 384 | Very Fast | Good | Free |
| `bge-large-en-v1.5` (open-source) | 1024 | Medium | Excellent | Free |
| `nomic-embed-text` (open-source) | 768 | Fast | Very Good | Free |

### Vector Database Options

| Database | Type | Best For | Index Types | Scale |
|----------|------|---------|-------------|-------|
| **FAISS** | Library (in-memory) | Prototyping, single-node | IVF-PQ, HNSW | ~10M vectors |
| **pgvector** | Postgres extension | Already using Postgres | HNSW, IVFFlat | ~5M vectors |
| **Pinecone** | Managed cloud | Production, zero-ops | Proprietary | Billions |
| **Weaviate** | Self-hosted / cloud | Hybrid search | HNSW | ~100M vectors |
| **Qdrant** | Self-hosted / cloud | Filtering + search | HNSW | ~100M vectors |
| **ChromaDB** | Embedded | Quick prototyping | HNSW | ~1M vectors |

### Index Algorithms — HNSW vs IVF-PQ

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INDEX COMPARISON                                  │
│                                                                          │
│   HNSW (Hierarchical Navigable Small World)                             │
│   ─────────────────────────────────────────                             │
│   • Graph-based: builds navigable layers of proximity links             │
│   • O(log N) query time                                                 │
│   • HIGH recall (>98%), HIGH memory (~6 KB per vector for 1536-dim)     │
│   • Best when: you need accuracy and have sufficient RAM                │
│                                                                          │
│   IVF-PQ (Inverted File + Product Quantization)                         │
│   ─────────────────────────────────────────────                         │
│   • Cluster-based: partitions space into Voronoi cells + compresses     │
│   • O(√N) query time                                                    │
│   • MODERATE recall (~92-95%), LOW memory (~64 bytes per vector)        │
│   • Best when: you have billions of vectors and limited RAM             │
│                                                                          │
│   ┌──────────┬────────────┬────────────┬───────────────┐                │
│   │          │   HNSW     │   IVF-PQ   │   Flat (Brute)│                │
│   ├──────────┼────────────┼────────────┼───────────────┤                │
│   │ Recall   │   >98%     │   92-95%   │   100%        │                │
│   │ Speed    │   ~1ms     │   ~0.5ms   │   ~100ms      │                │
│   │ Memory/v │   ~6 KB    │   ~64 B    │   ~6 KB       │                │
│   │ Build    │   Slow     │   Medium   │   None        │                │
│   └──────────┴────────────┴────────────┴───────────────┘                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Memory Math — 6 KB Per Vector

```
Memory per vector (HNSW, float32):
  = dimensions × 4 bytes  +  graph overhead
  = 1536 × 4              +  ~512 bytes (neighbor links)
  = 6,144 + 512
  ≈ 6 KB per vector

1 million vectors × 6 KB = 6 GB RAM
10 million vectors × 6 KB = 60 GB RAM  ← needs dedicated node
```

### Code Example — FAISS HNSW Index

```python
import faiss
import numpy as np

d = 1536                          # embedding dimension
M = 32                            # neighbors per node (higher = better recall, more RAM)
ef_construction = 200             # build-time search depth

# Create HNSW index
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 64          # query-time depth (tune for speed/recall)

# Add vectors
embeddings = np.random.rand(100_000, d).astype("float32")
index.add(embeddings)

# Search
query = np.random.rand(1, d).astype("float32")
distances, indices = index.search(query, k=20)   # top-20 neighbors
```

---

## **Step 3: Retrieve — Semantic Search + Hybrid**

### Retrieval Strategies

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL SPECTRUM                                 │
│                                                                          │
│   Pure Keyword (BM25)          Hybrid              Pure Semantic         │
│   ────────────────────    ──────────────────    ────────────────────     │
│   • Exact term matching   • Best of both        • Meaning-based         │
│   • Fast, interpretable   • RRF or weighted     • Handles synonyms      │
│   • Misses synonyms       • α=0.7 sem + 0.3 kw • Misses exact terms    │
│                                                                          │
│   "machine learning"      "machine learning"    "machine learning"      │
│   finds: "machine         finds: "machine       finds: "ML", "deep     │
│   learning" only          learning" + "ML" +    learning", "neural      │
│                           "deep learning"       networks"               │
│                                                                          │
│   ★ RECOMMENDED: Hybrid retrieval with α ≈ 0.7 (semantic-heavy)        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why Top-K ≈ 20?

| Stage | Count | Rationale |
|-------|-------|-----------|
| **Initial retrieval** | k = 20-50 | Cast a wide net; recall > precision at this stage |
| **After re-ranking** | Top 5-8 | Cross-encoder narrows to high-precision set |
| **After compression** | 3-5 chunks | Fit within token budget with max information density |

### Code Example — Hybrid Search

```python
from rank_bm25 import BM25Okapi
import numpy as np

# --- BM25 Keyword Search ---
tokenized_corpus = [doc.split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query.split())

# --- Semantic Search (from FAISS/pgvector) ---
query_embedding = embed_model.encode([query])
distances, sem_indices = faiss_index.search(query_embedding, k=20)
sem_scores = 1 / (1 + distances[0])  # convert distance to similarity

# --- Hybrid Fusion (Reciprocal Rank Fusion) ---
def reciprocal_rank_fusion(rankings, k=60):
    """Fuse multiple ranked lists using RRF."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

bm25_ranking = np.argsort(-bm25_scores)[:20]
sem_ranking = sem_indices[0]

final_ranking = reciprocal_rank_fusion([bm25_ranking, sem_ranking])
top_k_docs = [corpus_texts[i] for i in final_ranking[:20]]
```

### Metadata Filtering

```python
# pgvector with metadata filters
results = supabase.rpc("match_documents", {
    "query_embedding": query_vec,
    "match_threshold": 0.78,
    "match_count": 20,
    "filter": {
        "source": "annual_report_2024",
        "department": "engineering"
    }
}).execute()
```

---

## **Step 4: Re-Rank & Compress — Maximize Relevance Per Token**

### Re-Ranking with Cross-Encoders

```
┌──────────────────────────────────────────────────────────────────────────┐
│              BI-ENCODER vs CROSS-ENCODER                                 │
│                                                                          │
│   Bi-Encoder (Retrieval)           Cross-Encoder (Re-Ranking)           │
│   ──────────────────────           ──────────────────────────           │
│   Query → [Encoder] → vec_q       (Query, Doc) → [Encoder] → score     │
│   Doc   → [Encoder] → vec_d                                             │
│   score = cosine(vec_q, vec_d)     Jointly attends to both — much       │
│                                    more accurate but O(N×Q) so only     │
│   Fast: encode once, compare all   used on the top-K from retrieval     │
│   Used for: initial top-K fetch    Used for: precision re-ordering      │
│                                                                          │
│   Typical flow:                                                         │
│   20 candidates (bi-encoder) → 5 winners (cross-encoder)               │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Example — Cross-Encoder Re-Ranking

```python
from sentence_transformers import CrossEncoder

# Load re-ranking model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)

# Score each (query, document) pair
pairs = [(query, doc) for doc in top_k_docs]
scores = reranker.predict(pairs)

# Sort by relevance score
ranked = sorted(zip(scores, top_k_docs), reverse=True)
top_5 = [doc for score, doc in ranked[:5]]
```

### Deduplication

```python
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_chunks(chunks, embeddings, threshold=0.92):
    """Remove near-duplicate chunks based on cosine similarity."""
    keep = [0]
    for i in range(1, len(chunks)):
        sims = cosine_similarity([embeddings[i]], [embeddings[j] for j in keep])[0]
        if max(sims) < threshold:
            keep.append(i)
    return [chunks[i] for i in keep]
```

### Semantic Compression — 6-8x Token Reduction

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   COMPRESSION PIPELINE                                   │
│                                                                          │
│   Original Chunk (512 tokens)                                           │
│   ┌──────────────────────────────────────────┐                          │
│   │ "The quarterly report shows revenue of   │                          │
│   │ $4.2B, an increase of 12% year over year.│                          │
│   │ The company attributed this growth to ... │                          │
│   │ [detailed breakdown of 15 line items]     │                          │
│   │ ... resulting in improved margins."       │                          │
│   └──────────────────────────────────────────┘                          │
│                       │                                                  │
│                       ▼  Semantic Compression                           │
│                                                                          │
│   Compressed (72 tokens) — 7.1x reduction                               │
│   ┌──────────────────────────────────────────┐                          │
│   │ "Q4 revenue: $4.2B (+12% YoY). Growth    │                          │
│   │ driven by cloud services (+34%) and       │                          │
│   │ enterprise deals. Operating margin: 28%." │                          │
│   └──────────────────────────────────────────┘                          │
│                                                                          │
│   Accuracy loss: < 1 percentage point on downstream QA tasks            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code Example — LLMLingua Compression

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)

compressed = compressor.compress_prompt(
    context=[chunk.page_content for chunk in top_5],
    instruction="Answer the user's question based on the context.",
    question=user_query,
    target_token=800,          # compress to ~800 tokens
    condition_compare=True,
    rank_method="longllmlingua",
)

print(f"Original: {compressed['origin_tokens']} tokens")
print(f"Compressed: {compressed['compressed_tokens']} tokens")
print(f"Ratio: {compressed['ratio']:.1f}x reduction")
# Typical output: Original: 4800 → Compressed: 720 → 6.7x reduction
```

---

## **Step 5: Package & Inject — Build the Final Prompt**

### Token Budget Planning

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TOKEN BUDGET TEMPLATE (8K window)                      │
│                                                                          │
│   Component              Tokens    % of Window                          │
│   ─────────────────────  ──────    ───────────                          │
│   System Prompt           500       6.1%                                │
│   Retrieved Context      5,500     67.1%    ← Context Engineering zone │
│   User Query              200       2.4%                                │
│   Few-shot Examples       800       9.8%                                │
│   Generation Buffer      1,192     14.6%    ← Model's output space     │
│   ─────────────────────  ──────    ───────────                          │
│   TOTAL                  8,192     100%                                 │
│                                                                          │
│   Rule of thumb:                                                        │
│   generation_buffer = max(expected_output_len × 1.5, 512)              │
│   retrieved_context = window - system - query - buffer - few_shot       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Prompt Assembly — Preserving Order

```python
import tiktoken

def build_prompt(system_prompt, chunks, user_query, model="gpt-4o", max_tokens=8192):
    """Assemble final prompt within token budget, preserving chunk order."""
    enc = tiktoken.encoding_for_model(model)

    system_tokens = len(enc.encode(system_prompt))
    query_tokens = len(enc.encode(user_query))
    generation_buffer = 1024
    few_shot_tokens = 0  # adjust if using examples

    available = max_tokens - system_tokens - query_tokens - generation_buffer - few_shot_tokens

    # Pack chunks in ORDER (first = most relevant) until budget fills
    selected_chunks = []
    used_tokens = 0
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk))
        if used_tokens + chunk_tokens > available:
            break
        selected_chunks.append(chunk)
        used_tokens += chunk_tokens

    # Assemble structured context
    context_block = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(selected_chunks)
    )

    prompt = f"""{system_prompt}

## Retrieved Context
{context_block}

## User Question
{user_query}

## Instructions
Answer based ONLY on the provided context. If the context doesn't contain the answer, say "I don't have enough information to answer this."
"""
    return prompt, used_tokens

prompt, tokens_used = build_prompt(system_prompt, compressed_chunks, user_query)
print(f"Context tokens used: {tokens_used}/{8192}")
```

### MCP Context Array Format

For systems using Model Context Protocol (MCP), context is structured as a typed array:

```python
# MCP-style context injection
context_array = [
    {
        "type": "system",
        "content": system_prompt,
        "metadata": {"role": "instructions"}
    },
    {
        "type": "retrieval",
        "content": compressed_chunk_1,
        "metadata": {"source": "annual_report.pdf", "page": 12, "score": 0.94}
    },
    {
        "type": "retrieval",
        "content": compressed_chunk_2,
        "metadata": {"source": "tech_spec.md", "section": "architecture", "score": 0.91}
    },
    {
        "type": "memory",
        "content": "User previously asked about Q3 results.",
        "metadata": {"turn": 3, "type": "conversation_history"}
    },
    {
        "type": "user",
        "content": user_query,
        "metadata": {"timestamp": "2025-01-15T10:30:00Z"}
    }
]
```

---

# **3. Context Window Management**

---

## **3.1 Token Budget Tiers**

Different models give you radically different budgets. Your context engineering strategy must adapt:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW TIERS                                   │
│                                                                          │
│   4K tokens (GPT-3.5-turbo-0613)                                        │
│   ┌──────────────────┐                                                   │
│   │ ██░░░░░░░░░░░░░░ │  Tight — 2-3 compressed chunks max              │
│   └──────────────────┘  Strategy: aggressive compression, short system  │
│                                                                          │
│   8K tokens (GPT-4, Llama 3-8B)                                        │
│   ┌──────────────────────────────────┐                                   │
│   │ ████████░░░░░░░░░░░░░░░░░░░░░░░ │  Standard — 5-8 chunks           │
│   └──────────────────────────────────┘  Strategy: rerank top-5, light   │
│                                          compression                     │
│                                                                          │
│   32K tokens (GPT-4-32k, Claude 2)                                      │
│   ┌──────────────────────────────────────────────────┐                   │
│   │ ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░ │  Generous         │
│   └──────────────────────────────────────────────────┘  Strategy: more  │
│                         chunks, keep diversity, summarize long docs      │
│                                                                          │
│   128K tokens (GPT-4-turbo, Claude 3, Gemini 1.5)                      │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ ████████████████████████████████████████████████████████████████ │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│   Massive — entire documents fit. BUT "lost in the middle" still bites. │
│   Strategy: still prioritize ordering; front-load critical info.        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Budget Calculation Formula

```
available_context = window_size - system_tokens - query_tokens - generation_buffer
chunks_to_include = available_context / avg_chunk_tokens

Example (8K window):
  available = 8192 - 500 - 200 - 1024 = 6,468 tokens
  chunks    = 6,468 / 512 ≈ 12 raw chunks or 4-5 after compression
```

---

## **3.2 The "Lost in the Middle" Problem**

This is a **critical** concept for interviews. Research from Stanford/Berkeley (Liu et al., 2023) demonstrated:

```
┌──────────────────────────────────────────────────────────────────────────┐
│               "LOST IN THE MIDDLE" PHENOMENON                            │
│                                                                          │
│   Accuracy                                                              │
│   100% ┤                                                                │
│    95% ┤ ★★★                                             ★★★            │
│    90% ┤      ★★                                     ★★                 │
│    85% ┤         ★★                               ★★                   │
│    80% ┤            ★★                         ★★                      │
│    75% ┤               ★★                   ★★                         │
│    70% ┤                  ★★★★★★★★★★★★★★★★                             │
│    65% ┤                                                                │
│        └──────────────────────────────────────────────                  │
│         Position 1  ...  Position 10  ...  Position 20                  │
│         (Beginning)      (Middle)          (End)                        │
│                                                                          │
│   KEY FINDING:                                                          │
│   • Models attend strongly to the BEGINNING and END of context          │
│   • Information buried in the MIDDLE is often ignored or degraded       │
│   • Effect worsens as context length increases                          │
│   • Even 128K-window models suffer from this                            │
│                                                                          │
│   ★ MITIGATION STRATEGIES:                                              │
│   1. Place most relevant chunks at BEGINNING and END                    │
│   2. Use "sandwich" ordering: important-filler-important                │
│   3. Keep context concise — don't fill just because you can             │
│   4. Repeat critical instructions at the end (recency bias)            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Code — Sandwich Ordering

```python
def sandwich_order(ranked_chunks):
    """
    Re-order chunks for 'lost in the middle' mitigation.
    Place most relevant at beginning and end; least relevant in middle.
    """
    if len(ranked_chunks) <= 2:
        return ranked_chunks

    n = len(ranked_chunks)
    beginning = ranked_chunks[:n // 3]         # top-third at start
    middle = ranked_chunks[n // 3: 2 * n // 3] # mid-relevance in center
    end = ranked_chunks[2 * n // 3:]           # next-best at end

    return beginning + middle + list(reversed(end))
```

> **Interview tip:** If asked "How would you handle a 128K context window?", don't just say "put everything in." Show you understand that more context ≠ better output due to the lost-in-the-middle effect.

---

# **4. Compression Techniques**

---

## **4.1 Technique Comparison**

| Technique | Token Reduction | Accuracy Impact | Latency | Best For |
|-----------|----------------|----------------|---------|----------|
| **Semantic Compression** (LLMLingua) | 6-8x | <1 pt loss | ~200ms | General RAG |
| **Map-Reduce Summarization** | 10-20x | 2-3 pt loss | ~2-5s (LLM calls) | Long documents |
| **Key-Point Extraction** | 8-15x | 1-2 pt loss | ~1-3s | Reports, papers |
| **Sliding Window Attention** | N/A (arch-level) | Minimal | Built-in | Long-context models |
| **Extractive Compression** | 3-5x | <0.5 pt loss | ~100ms | Preserving exact quotes |

---

## **4.2 Semantic Compression — Deep Dive**

Semantic compression uses a small language model to identify and remove tokens that contribute minimally to meaning, achieving 6-8x reduction with negligible accuracy loss.

```
┌──────────────────────────────────────────────────────────────────────────┐
│               SEMANTIC COMPRESSION MECHANISM                             │
│                                                                          │
│   Input (48 tokens):                                                    │
│   "The comprehensive quarterly financial report that was released       │
│    yesterday clearly demonstrates and shows that the overall total      │
│    revenue for this particular quarter increased significantly by       │
│    approximately 12 percent compared to the same period last year."     │
│                                                                          │
│           ▼ Token-level importance scoring (perplexity-based)           │
│                                                                          │
│   Score:  [0.1, 0.2, 0.8, 0.6, 0.1, ...]                             │
│           ↓ remove low-importance tokens                                │
│                                                                          │
│   Output (8 tokens):                                                    │
│   "quarterly revenue increased 12 percent year-over-year"              │
│                                                                          │
│   Reduction: 48 → 8 tokens = 6x compression                            │
│   Semantic loss: negligible for downstream QA                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **4.3 Map-Reduce Summarization**

For extremely long documents that exceed even 128K windows:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MAP-REDUCE SUMMARIZATION                               │
│                                                                          │
│   Document (500 pages)                                                  │
│   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  ... ┌────┐                      │
│   │ C1 │ │ C2 │ │ C3 │ │ C4 │ │ C5 │      │ Cn │   n chunks          │
│   └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘      └──┬─┘                      │
│      │      │      │      │      │            │                         │
│      ▼      ▼      ▼      ▼      ▼            ▼    MAP phase           │
│   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐      ┌────┐  (parallel LLM       │
│   │ S1 │ │ S2 │ │ S3 │ │ S4 │ │ S5 │      │ Sn │   summarize calls)  │
│   └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘      └──┬─┘                      │
│      │      │      │      │      │            │                         │
│      └──────┴──────┴──┬───┴──────┴────────────┘                         │
│                       │                                                  │
│                       ▼   REDUCE phase (combine summaries)              │
│                   ┌────────┐                                             │
│                   │ Final  │                                             │
│                   │Summary │   Fits in context window                    │
│                   └────────┘                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Map-reduce: summarize each chunk, then combine
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.invoke({"input_documents": chunks})
```

---

## **4.4 Sliding Window Attention**

An architecture-level approach (used in Mistral, Longformer) that doesn't require explicit compression:

```
┌──────────────────────────────────────────────────────────────────────────┐
│               SLIDING WINDOW ATTENTION                                   │
│                                                                          │
│   Full Self-Attention (standard transformer):                           │
│   Every token attends to every other token → O(n²) memory              │
│                                                                          │
│   Token:  1  2  3  4  5  6  7  8  9  10                                │
│        1  ★  ★  ★  ★  ★  ★  ★  ★  ★  ★                                │
│        2  ★  ★  ★  ★  ★  ★  ★  ★  ★  ★                                │
│        ...                                                              │
│                                                                          │
│   Sliding Window Attention (window_size=4):                             │
│   Each token attends only to W nearby tokens → O(n×W) memory           │
│                                                                          │
│   Token:  1  2  3  4  5  6  7  8  9  10                                │
│        1  ★  ★  ★  ★  ·  ·  ·  ·  ·  ·                                │
│        2  ★  ★  ★  ★  ★  ·  ·  ·  ·  ·                                │
│        3  ·  ★  ★  ★  ★  ★  ·  ·  ·  ·                                │
│        4  ·  ·  ★  ★  ★  ★  ★  ·  ·  ·                                │
│        ...                                                              │
│                                                                          │
│   + Global attention tokens (CLS, query) attend to everything           │
│   Result: handles much longer contexts with bounded memory              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# **5. KV-Cache and Memory**

---

## **5.1 What Is the KV-Cache?**

During autoregressive generation, the transformer recomputes Keys and Values for all previous tokens at every step. The **KV-cache** stores these to avoid redundant computation.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     KV-CACHE MECHANISM                                    │
│                                                                          │
│   Without KV-Cache:                                                     │
│   Step 1: compute K,V for [tok1]                                        │
│   Step 2: compute K,V for [tok1, tok2]           ← recomputes tok1!    │
│   Step 3: compute K,V for [tok1, tok2, tok3]     ← recomputes 1 & 2!  │
│   ... O(n²) total compute                                               │
│                                                                          │
│   With KV-Cache:                                                        │
│   Step 1: compute K,V for [tok1], STORE in cache                        │
│   Step 2: compute K,V for [tok2] only, append to cache                  │
│   Step 3: compute K,V for [tok3] only, append to cache                  │
│   ... O(n) total compute — massive speedup                              │
│                                                                          │
│   Trade-off: speed ↑↑↑  but  memory ↑↑↑                                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **5.2 KV-Cache Memory Formula**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    KV-CACHE MEMORY FORMULA                                │
│                                                                          │
│   KV_cache_bytes = 2 × n_layers × seq_len × d_model × dtype_bytes      │
│                                                                          │
│   Where:                                                                │
│     2          = one K tensor + one V tensor per layer                   │
│     n_layers   = number of transformer layers                           │
│     seq_len    = current sequence length (grows during generation)       │
│     d_model    = hidden dimension of the model                          │
│     dtype_bytes = bytes per element (fp16 = 2, fp32 = 4, int8 = 1)     │
│                                                                          │
│   Simplified (single-layer):                                            │
│   KV_bytes_per_layer = 2 × seq_len × d_model × dtype_bytes             │
└──────────────────────────────────────────────────────────────────────────┘
```

### Concrete Examples

| Model | Layers | d_model | 128K seq (FP16) | 128K seq (INT8) |
|-------|--------|---------|-----------------|-----------------|
| **GPT-2** (small) | 12 | 768 | 4.7 GB | 2.4 GB |
| **Llama-3-8B** | 32 | 4096 | 67 GB | 33.5 GB |
| **Llama-3-70B** | 80 | 8192 | 336 GB | 168 GB |
| **GPT-4 (est.)** | ~120 | ~12288 | ~756 GB | ~378 GB |

### Quick Estimation

```python
def kv_cache_size_gb(n_layers, d_model, seq_len, dtype_bytes=2):
    """Calculate KV-cache memory in GB.
    
    Args:
        n_layers: Number of transformer layers
        d_model: Hidden dimension
        seq_len: Sequence length (context + generated tokens)
        dtype_bytes: 2 for fp16, 1 for int8, 4 for fp32
    
    Returns:
        Memory in GB
    """
    bytes_total = 2 * n_layers * seq_len * d_model * dtype_bytes
    return bytes_total / (1024 ** 3)

# Llama-3-8B at 128K context, FP16
print(f"{kv_cache_size_gb(32, 4096, 131072, 2):.1f} GB")
# Output: 67.1 GB

# Same model, INT8 quantized KV-cache
print(f"{kv_cache_size_gb(32, 4096, 131072, 1):.1f} GB")
# Output: 33.5 GB

# Practical: 8K context on 8B model, FP16
print(f"{kv_cache_size_gb(32, 4096, 8192, 2):.2f} GB")
# Output: 4.19 GB
```

---

## **5.3 KV-Cache Quantization**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                KV-CACHE QUANTIZATION COMPARISON                          │
│                                                                          │
│   Precision    Bytes/Element    Cache Size (8B, 128K)    Quality        │
│   ─────────    ─────────────    ────────────────────     ─────────      │
│   FP32         4 bytes          134.2 GB                 Baseline       │
│   FP16         2 bytes           67.1 GB (-50%)          ~Identical     │
│   INT8         1 byte            33.5 GB (-75%)          <0.5% loss     │
│   INT4         0.5 bytes         16.8 GB (-87.5%)        ~1-2% loss     │
│                                                                          │
│   ★ INT8 KV-cache quantization is the sweet spot:                       │
│     - 75% memory reduction                                              │
│     - Negligible quality degradation                                    │
│     - Supported by vLLM, TensorRT-LLM, llama.cpp                       │
│                                                                          │
│   ★ At INT4, up to 87% cache reduction is achievable                    │
│     with techniques like KIVI (per-channel quantization)                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **5.4 128K Context ≈ 0.5 GB Rule of Thumb**

For practical estimation in interviews:

```
128K tokens on a ~7B parameter model (FP16):
  KV-cache ≈ 0.5 GB per forward pass (with grouped-query attention)
  
Why lower than the formula suggests?
  • Grouped-Query Attention (GQA): shares KV heads → 4-8x reduction
  • Llama-3-8B uses GQA with 8 KV heads (vs 32 attention heads)
  • Effective: 2 × 32 layers × 128K × (4096/4) × 2 bytes ≈ 0.5 GB

This is the number to quote in interviews for "How much RAM does 128K context need?"
```

---

# **6. Advanced Techniques**

---

## **6.1 MMR — Maximal Marginal Relevance**

MMR balances **relevance** to the query with **diversity** among selected chunks, preventing redundant context:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MMR FORMULA                                            │
│                                                                          │
│   MMR = argmax  [ λ × Sim(dᵢ, Q)  −  (1 − λ) × max Sim(dᵢ, dⱼ) ]    │
│         dᵢ∈R\S                            dⱼ∈S                          │
│                                                                          │
│   Where:                                                                │
│     Q  = query                                                          │
│     R  = retrieved candidate set                                        │
│     S  = already-selected set                                           │
│     dᵢ = candidate document                                            │
│     λ  = diversity parameter (0.5 = balanced, 0.7 = relevance-heavy)   │
│                                                                          │
│   λ = 1.0  →  pure relevance (may get redundant chunks)                │
│   λ = 0.0  →  pure diversity (may get irrelevant chunks)               │
│   λ = 0.5  →  balanced (typical starting point)                        │
│   λ = 0.7  →  relevance-biased (good for focused QA)                   │
│                                                                          │
│   Example: Query = "What are the side effects of aspirin?"              │
│                                                                          │
│   Without MMR (top-3 by relevance):                                     │
│     1. "Aspirin side effects include stomach bleeding..."               │
│     2. "Common side effects of aspirin: GI bleeding..."   ← redundant  │
│     3. "Aspirin-related adverse effects: gastric issues..." ← redundant│
│                                                                          │
│   With MMR (λ=0.6):                                                     │
│     1. "Aspirin side effects include stomach bleeding..."               │
│     2. "Aspirin can interact with blood thinners..."      ← diverse    │
│     3. "Dosage recommendations to minimize side effects..." ← diverse  │
└──────────────────────────────────────────────────────────────────────────┘
```

```python
def mmr_select(query_embedding, candidate_embeddings, candidates, k=5, lambda_param=0.7):
    """Select top-k chunks using Maximal Marginal Relevance."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    query_sims = cosine_similarity([query_embedding], candidate_embeddings)[0]
    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(k):
        if not remaining:
            break

        mmr_scores = []
        for idx in remaining:
            relevance = query_sims[idx]
            if selected_indices:
                diversity_penalty = max(
                    cosine_similarity(
                        [candidate_embeddings[idx]],
                        [candidate_embeddings[j] for j in selected_indices]
                    )[0]
                )
            else:
                diversity_penalty = 0

            mmr = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            mmr_scores.append((mmr, idx))

        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]
```

---

## **6.2 Self-RAG — Model Decides When to Retrieve**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SELF-RAG FLOW                                     │
│                                                                          │
│   Standard RAG: ALWAYS retrieve → inject → generate                     │
│   Self-RAG:     Model DECIDES whether retrieval is needed               │
│                                                                          │
│   ┌─────────┐     ┌─────────────────┐                                   │
│   │  User   │────▶│  LLM evaluates  │                                   │
│   │  Query  │     │  "Do I need     │                                   │
│   └─────────┘     │   external      │                                   │
│                   │   knowledge?"   │                                   │
│                   └────────┬────────┘                                    │
│                            │                                             │
│                  ┌─────────┴──────────┐                                  │
│                  ▼                    ▼                                   │
│            [Retrieve=YES]      [Retrieve=NO]                            │
│                  │                    │                                   │
│            ┌─────▼─────┐       ┌─────▼─────┐                            │
│            │ Retrieve   │       │ Generate  │                            │
│            │ chunks     │       │ directly  │                            │
│            └─────┬─────┘       └───────────┘                            │
│                  │                                                       │
│            ┌─────▼──────────┐                                            │
│            │ Self-critique: │                                            │
│            │ Is chunk       │                                            │
│            │ relevant?      │                                            │
│            │ Is response    │                                            │
│            │ supported?     │                                            │
│            └─────┬──────────┘                                            │
│                  │                                                       │
│         ┌────────┴────────┐                                              │
│         ▼                 ▼                                              │
│   [Supported]       [Not Supported]                                     │
│   Output answer     Retrieve more / Regenerate                          │
│                                                                          │
│   Special tokens: [Retrieve], [IsRel], [IsSup], [IsUse]                │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key benefits of Self-RAG:**
- Reduces unnecessary retrieval calls (saves latency + cost)
- Model self-critiques its own outputs against evidence
- Outperforms standard RAG on knowledge-intensive tasks by 5-10%

---

## **6.3 Query Decomposition**

Complex queries often need multiple retrieval passes:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QUERY DECOMPOSITION                                    │
│                                                                          │
│   Original Query:                                                       │
│   "Compare the revenue growth of Apple and Microsoft in Q3 2024        │
│    and explain which company had better cloud performance."              │
│                                                                          │
│            ▼  LLM decomposes into sub-queries                           │
│                                                                          │
│   Sub-Query 1: "Apple Q3 2024 revenue growth"                          │
│   Sub-Query 2: "Microsoft Q3 2024 revenue growth"                      │
│   Sub-Query 3: "Apple cloud services performance Q3 2024"              │
│   Sub-Query 4: "Microsoft Azure cloud performance Q3 2024"             │
│                                                                          │
│            ▼  Retrieve for each sub-query independently                 │
│                                                                          │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                          │
│   │Chunks 1│ │Chunks 2│ │Chunks 3│ │Chunks 4│                          │
│   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘                          │
│       └──────────┴──────┬───┴──────────┘                                │
│                         ▼                                                │
│              Merge, deduplicate, re-rank                                 │
│                         ▼                                                │
│              Context-engineered prompt → LLM                            │
└──────────────────────────────────────────────────────────────────────────┘
```

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Automatically generates multiple sub-queries
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    llm=llm,
)

# Single call retrieves for all decomposed queries
docs = retriever.invoke("Compare Apple and Microsoft cloud performance in Q3 2024")
```

---

## **6.4 Multi-Hop Reasoning**

For questions requiring chained reasoning across multiple documents:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HOP REASONING                                    │
│                                                                          │
│   Question: "Who is the CEO of the company that acquired Instagram?"    │
│                                                                          │
│   Hop 1: "Which company acquired Instagram?"                            │
│   Retrieved: "Facebook acquired Instagram in 2012 for $1B"             │
│   Answer 1: Facebook (now Meta)                                         │
│                                                                          │
│   Hop 2: "Who is the CEO of Meta?"                                     │
│   Retrieved: "Mark Zuckerberg is the CEO of Meta Platforms"            │
│   Answer 2: Mark Zuckerberg                                            │
│                                                                          │
│   Final: "Mark Zuckerberg is the CEO of Meta, which acquired Instagram"│
│                                                                          │
│   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐       │
│   │ Query   │────▶│ Retrieve │────▶│ Extract │────▶│ Retrieve │──▶ ...│
│   │ (Hop 1) │     │ Hop 1    │     │ Entity  │     │ Hop 2    │       │
│   └─────────┘     └──────────┘     └─────────┘     └──────────┘       │
│                                                                          │
│   Frameworks: IRCoT, ReAct, ITER-RETGEN                                │
│   Key challenge: error propagation — wrong hop 1 → wrong everything    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# **7. MVP Architecture — Single T4 16GB Node**

---

## **7.1 End-to-End Architecture**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 MVP RAG ARCHITECTURE (Single T4 16GB)                    │
│                                                                          │
│   ┌──────────┐                                                          │
│   │   User   │                                                          │
│   │  Query   │                                                          │
│   └────┬─────┘                                                          │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────────────────┐               │
│   │              ORCHESTRATOR (FastAPI)                   │               │
│   │              ~200 MB RAM                              │               │
│   │                                                       │               │
│   │  1. Receive query                                     │               │
│   │  2. Embed query (local bi-encoder)                    │               │
│   │  3. Search vector DB                                  │               │
│   │  4. Re-rank with cross-encoder                        │               │
│   │  5. Compress chunks                                   │               │
│   │  6. Build prompt (token budget)                       │               │
│   │  7. Call LLM                                          │               │
│   │  8. Return response                                   │               │
│   └────┬──────────┬──────────────┬────────────────┬──────┘               │
│        │          │              │                │                       │
│        ▼          ▼              ▼                ▼                       │
│   ┌────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐            │
│   │Chunker │ │ Vector DB  │ │ Re-Ranker  │ │     LLM      │            │
│   │        │ │ (FAISS)    │ │ (Cross-Enc)│ │ (Llama-3-8B) │            │
│   │Runs at │ │            │ │            │ │              │             │
│   │ingest  │ │ In-memory  │ │ MiniLM-12  │ │ 4-bit GPTQ  │            │
│   │time    │ │ HNSW index │ │ ~120MB     │ │ ~4.5 GB      │            │
│   └────────┘ └────────────┘ └────────────┘ └──────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **7.2 Memory Footprint Breakdown**

```
┌──────────────────────────────────────────────────────────────────────────┐
│              MEMORY BUDGET: T4 16GB GPU + 32GB System RAM                │
│                                                                          │
│   Component                  GPU VRAM        System RAM                 │
│   ──────────────────────     ────────        ──────────                 │
│   LLM Weights (8B, 4-bit)   4.5 GB          —                          │
│   KV-Cache (8K context)      1.2 GB          —                          │
│   Cross-Encoder (reranker)   0.5 GB          —                          │
│   Bi-Encoder (embeddings)    0.3 GB          —                          │
│   CUDA Overhead              0.5 GB          —                          │
│   ──────────────────────     ────────        ──────────                 │
│   GPU Subtotal               7.0 GB          —                          │
│   GPU Headroom               9.0 GB          —                          │
│                                                                          │
│   Orchestrator (FastAPI)     —               0.2 GB                     │
│   FAISS Index (500K vecs)    —               3.0 GB                     │
│   Document Store             —               0.5 GB                     │
│   Python Runtime + Libs      —               1.5 GB                     │
│   ──────────────────────     ────────        ──────────                 │
│   RAM Subtotal               —               5.2 GB                     │
│   RAM Headroom               —               26.8 GB                    │
│                                                                          │
│   ★ Comfortable fit on a single T4 16GB node!                           │
│   ★ Can serve ~5-10 concurrent users at ~2-3 tokens/sec                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### Scaling Notes

| Metric | Single T4 16GB | Production (A100 80GB) |
|--------|---------------|----------------------|
| Model size | 8B 4-bit | 70B 4-bit or 8B FP16 |
| Context window | 8K practical | 32-128K |
| Concurrent users | 5-10 | 50-100 (with vLLM batching) |
| Throughput | 2-3 tok/s | 30-50 tok/s per user |
| Vector DB | FAISS in-memory | Pinecone / pgvector (distributed) |

---

# **8. Production Pipeline**

---

## **8.1 Full Pipeline — From Data to Deployment**

```
┌──────────────────────────────────────────────────────────────────────────┐
│              PRODUCTION CONTEXT ENGINEERING PIPELINE                      │
│                                                                          │
│   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐            │
│   │  DATA   │──▶│  INDEX  │──▶│ RETRIEVAL│──▶│COMPRESSION│            │
│   │  PREP   │   │  BUILD  │   │   TUNE   │   │           │            │
│   └─────────┘   └─────────┘   └──────────┘   └─────┬─────┘            │
│                                                      │                   │
│   • Parse PDFs,   • Embed all    • Tune k, α      • Configure          │
│     HTML, docs      chunks       • Eval recall       LLMLingua         │
│   • Clean text    • Build HNSW   • Add BM25        • Set target        │
│   • Chunk with    • Deploy to    • Test filters      tokens            │
│     overlap         pgvector/    • Benchmark        • Validate          │
│   • Add metadata    Pinecone       hybrid mix        quality           │
│                                                      │                   │
│                                                      ▼                   │
│   ┌───────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐           │
│   │  ITERATE  │◀──│OBSERVA- │◀──│  CI/CD   │◀──│ PROMPT   │           │
│   │           │   │ BILITY  │   │          │   │ ASSEMBLY │           │
│   └───────────┘   └─────────┘   └──────────┘   └──────────┘           │
│                                                                          │
│   • A/B test     • Log every   • Automated     • Token budget          │
│     chunk sizes    retrieval     regression      allocation            │
│   • Tune re-       + score       tests         • Sandwich              │
│     ranker       • Track       • Eval suite      ordering             │
│   • Update         hallucin.     (RAGAS)       • System prompt         │
│     embeddings   • Latency    • Staging env      versioning           │
│                    dashboards                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **8.2 Stage-by-Stage Detail**

### Stage 1: Data Prep

```python
# Production-grade document processing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Parse with layout awareness (tables, images, headers)
elements = partition_pdf(
    "annual_report.pdf",
    strategy="hi_res",           # OCR + layout detection
    extract_images_in_pdf=True,
)

# Chunk respecting document structure
chunks = chunk_by_title(
    elements,
    max_characters=1500,
    overlap=225,                 # ~15%
    combine_text_under_n_chars=200,
)

# Add metadata for filtering
for chunk in chunks:
    chunk.metadata.update({
        "source": "annual_report_2024.pdf",
        "department": "finance",
        "date_ingested": "2025-01-15",
    })
```

### Stage 2: Index Build

```python
# Production: pgvector with SQLAlchemy
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://user:pass@db:5432/vectors")

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(1536),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))
    # Create HNSW index for fast similarity search
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS docs_embedding_idx
        ON documents USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """))
    conn.commit()
```

### Stage 3: Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Evaluate RAG pipeline quality
result = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,          # Is the answer grounded in context?
        answer_relevancy,      # Does the answer address the question?
        context_precision,     # Are retrieved chunks relevant?
        context_recall,        # Did we retrieve all needed information?
    ],
)

print(result)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.85, 'context_recall': 0.79}
```

### Stage 4: Observability

```python
# Log every retrieval for debugging and improvement
import json
import logging

logger = logging.getLogger("context_engineering")

def log_retrieval(query, chunks, scores, final_prompt_tokens):
    logger.info(json.dumps({
        "query": query,
        "num_chunks_retrieved": len(chunks),
        "top_score": float(max(scores)),
        "min_score": float(min(scores)),
        "mean_score": float(sum(scores) / len(scores)),
        "prompt_tokens": final_prompt_tokens,
        "token_utilization": final_prompt_tokens / 8192,  # % of window used
        "timestamp": datetime.utcnow().isoformat(),
    }))
```

---

# **9. Prompt Engineering vs Context Engineering**

---

## **9.1 Side-by-Side Comparison**

```
┌──────────────────────────────────────────────────────────────────────────┐
│           PROMPT ENGINEERING vs CONTEXT ENGINEERING                       │
│                                                                          │
│   ┌─────────────────────────┐   ┌─────────────────────────┐             │
│   │   PROMPT ENGINEERING    │   │  CONTEXT ENGINEERING    │             │
│   │   "Designing the exam   │   │  "Choosing which        │             │
│   │    question"            │   │   textbook pages to     │             │
│   │                         │   │   hand out"             │             │
│   ├─────────────────────────┤   ├─────────────────────────┤             │
│   │                         │   │                         │             │
│   │ • Query formulation     │   │ • Chunk selection       │             │
│   │ • System prompt design  │   │ • Re-ranking            │             │
│   │ • Few-shot examples     │   │ • Compression           │             │
│   │ • Chain-of-Thought      │   │ • Token budgeting       │             │
│   │ • Output formatting     │   │ • Ordering (sandwich)   │             │
│   │ • Role specification    │   │ • Deduplication         │             │
│   │                         │   │ • Metadata filtering    │             │
│   ├─────────────────────────┤   ├─────────────────────────┤             │
│   │ Tokens used: ~200-500   │   │ Tokens used: ~2K-100K+ │             │
│   │ Controls: HOW model     │   │ Controls: WHAT model    │             │
│   │           thinks        │   │           knows         │             │
│   └─────────────────────────┘   └─────────────────────────┘             │
│                                                                          │
│   ┌─────────────────────────────────────────────────────┐               │
│   │              THEY WORK TOGETHER                      │               │
│   │                                                      │               │
│   │  Context Eng. fills the window with the right info  │               │
│   │  Prompt Eng. tells the model what to DO with it     │               │
│   │                                                      │               │
│   │  Bad context + great prompt = hallucination          │               │
│   │  Great context + bad prompt = unfocused answer       │               │
│   │  Great context + great prompt = production-quality   │               │
│   └─────────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **9.2 When to Invest in Which**

| Scenario | Primary Investment | Why |
|----------|-------------------|-----|
| RAG chatbot over company docs | **Context Engineering** | Answer quality depends on retrieving the right chunks |
| Creative writing assistant | **Prompt Engineering** | No external retrieval; it's all about instructions |
| Code generation from docs | **Both equally** | Need right context (API docs) + right instructions (format) |
| Multi-turn agent with tools | **Context Engineering** | Managing memory, tool outputs, and conversation history |
| Simple Q&A (no retrieval) | **Prompt Engineering** | Model uses parametric knowledge only |

---

# **10. Common Interview Questions with Strong Answers**

---

## **Q1: "What is context engineering and why does it matter?"**

> **Strong answer:**
> "Context engineering is the practice of selecting, compressing, ordering, and injecting the right tokens into a model's context window to maximize output quality. Think of it as choosing which toys to put in the model's hands — the model can only work with what you give it.
>
> It matters because in RAG systems, the retrieval quality is the bottleneck, not the model. You can have GPT-4, but if you feed it irrelevant or redundant chunks, it will hallucinate or give poor answers. Context engineering is about maximizing relevance per token within your budget — which directly impacts faithfulness, accuracy, and cost."

---

## **Q2: "How do you manage a limited context window?"**

> **Strong answer:**
> "I use a token budgeting approach. First, I calculate the available space: window size minus system prompt, user query, and generation buffer. Then I apply a multi-stage pipeline:
>
> 1. **Retrieve broadly** (top-20 via hybrid search)
> 2. **Re-rank precisely** (cross-encoder narrows to top-5)
> 3. **Compress aggressively** (LLMLingua for 6-8x token reduction with <1pt accuracy loss)
> 4. **Order strategically** (sandwich ordering to mitigate 'lost in the middle')
> 5. **Pack within budget** (greedy token-counting packer)
>
> For a concrete example: with an 8K window, after reserving ~1700 tokens for system prompt, query, and generation, I have ~6500 tokens for context. After compression, that's equivalent to 40-50K tokens of raw retrieval — quite powerful."

---

## **Q3: "What is the 'lost in the middle' problem?"**

> **Strong answer:**
> "Liu et al. (2023) showed that language models attend disproportionately to information at the beginning and end of the context window. Information placed in the middle is effectively 'lost' — the model has lower recall and accuracy on it.
>
> This holds even for models with 128K windows. The practical implication: don't just dump everything in. I use sandwich ordering — most relevant chunks at the beginning and end, lower-relevance in the middle. I also keep contexts concise rather than filling the window just because I can. Finally, I repeat critical instructions at the end of the prompt to exploit recency bias."

---

## **Q4: "How would you reduce token usage in a RAG system?"**

> **Strong answer:**
> "There are multiple levers at different stages:
>
> 1. **Chunking**: Smaller, semantic chunks instead of large fixed-size ones — only retrieve what's needed
> 2. **Retrieval precision**: Re-ranking with cross-encoders reduces the number of chunks needed from 20 to 5
> 3. **Deduplication**: Near-duplicate chunks waste tokens — I remove chunks with >92% cosine similarity
> 4. **Semantic compression**: LLMLingua achieves 6-8x reduction with negligible quality loss
> 5. **Caching**: Store compressed versions of frequently-retrieved chunks
> 6. **KV-cache quantization**: INT8 quantization saves 75% of cache memory, letting you serve more users
>
> In production, I've seen these techniques combined reduce token usage by 10-15x compared to naive RAG."

---

## **Q5: "Explain the tradeoffs between HNSW and IVF-PQ for vector indexing."**

> **Strong answer:**
> "Both are approximate nearest neighbor algorithms, optimized for different constraints:
>
> **HNSW** builds a navigable graph with hierarchical layers. It gives excellent recall (>98%) with O(log N) query time, but costs ~6KB per vector — so 10M vectors need 60GB RAM. It's ideal when you prioritize accuracy and have sufficient memory.
>
> **IVF-PQ** partitions the vector space into clusters (IVF) and compresses vectors using product quantization (PQ). It uses only ~64 bytes per vector — 100x less than HNSW. Recall is lower (~92-95%), but it scales to billions of vectors on modest hardware.
>
> For most RAG applications with <5M documents, I default to HNSW (via pgvector or FAISS). For web-scale applications with billions of vectors, IVF-PQ or a managed service like Pinecone makes more sense."

---

## **Q6: "Walk me through designing a RAG system from scratch."**

> **Strong answer:**
> "I follow a structured pipeline:
>
> 1. **Data Prep**: Parse documents with layout-aware tools (Unstructured, LlamaParse). Clean, chunk with 15% overlap using recursive character splitting. Add metadata (source, date, section).
>
> 2. **Index Build**: Embed chunks with a model like `text-embedding-3-small`. Store in pgvector with HNSW index (m=16, ef_construction=200). This handles up to 5M vectors on a single Postgres instance.
>
> 3. **Retrieval**: Hybrid search — 70% semantic weight + 30% BM25. Retrieve top-20 candidates. Apply metadata filters when applicable.
>
> 4. **Re-rank & Compress**: Cross-encoder re-ranks to top-5. Apply MMR (λ=0.7) for diversity. Optionally compress with LLMLingua for token-constrained windows.
>
> 5. **Prompt Assembly**: Token budget allocation. Sandwich ordering for lost-in-the-middle mitigation. System prompt with clear instructions and output format.
>
> 6. **Evaluation**: RAGAS metrics — faithfulness, context precision, context recall, answer relevancy. Set up regression tests and observability logging.
>
> 7. **Iterate**: A/B test chunk sizes, embedding models, re-ranker thresholds. Monitor retrieval scores and hallucination rates in production."

---

## **Q7: "What is Self-RAG and when would you use it?"**

> **Strong answer:**
> "Self-RAG is a framework where the model itself decides whether to retrieve external knowledge, and then self-critiques whether the retrieved context supports its answer. It uses special tokens like [Retrieve], [IsRelevant], [IsSupported] to gate retrieval and validate outputs.
>
> I'd use Self-RAG when: (a) not every query needs retrieval — some are simple enough to answer from parametric knowledge; (b) retrieval latency is a concern — skipping unnecessary fetches speeds up response; (c) I need verifiable, grounded outputs with built-in citation checking.
>
> The tradeoff is complexity: Self-RAG requires a fine-tuned model that produces these special tokens, versus standard RAG which works with any off-the-shelf LLM."

---

## **Q8: "How do you calculate KV-cache memory requirements?"**

> **Strong answer:**
> "The formula is: `KV_bytes = 2 × n_layers × seq_len × d_model × dtype_bytes`.
>
> The factor of 2 accounts for both the Key and Value tensors stored per layer. For Llama-3-8B (32 layers, d_model=4096) at 8K context in FP16: that's `2 × 32 × 8192 × 4096 × 2 = 4.3 GB`.
>
> In practice, modern models use Grouped-Query Attention (GQA) which shares KV heads — Llama-3 uses 8 KV heads vs 32 attention heads, giving a 4x reduction. So the effective cache is closer to ~1 GB for 8K context.
>
> For the quick interview estimate: 128K context on a 7-8B model with GQA ≈ 0.5 GB. KV-cache quantization (INT8) can further halve that."

---

# **11. Key Takeaways**

---

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    KEY TAKEAWAYS — CONTEXT ENGINEERING                    │
│                                                                          │
│   1. CONTEXT > PROMPT                                                   │
│      In RAG, what you feed the model matters more than how you ask.     │
│      Bad context → hallucination, regardless of prompt quality.          │
│                                                                          │
│   2. THE 5-STEP RECIPE                                                  │
│      Chunk → Embed & Index → Retrieve → Re-Rank/Compress → Package     │
│      Each step has tunable knobs that compound in quality gains.        │
│                                                                          │
│   3. TOKEN BUDGETING IS NON-NEGOTIABLE                                  │
│      Always plan: system + context + query + buffer = window.           │
│      Never "fill the window and hope for the best."                     │
│                                                                          │
│   4. LOST IN THE MIDDLE IS REAL                                         │
│      Place critical info at the BEGINNING and END of context.           │
│      Use sandwich ordering. Keep context concise.                       │
│                                                                          │
│   5. COMPRESSION IS A SUPERPOWER                                        │
│      6-8x token reduction with <1pt accuracy loss.                      │
│      Equivalent to expanding your context window for free.              │
│                                                                          │
│   6. KV-CACHE MATH MATTERS                                              │
│      2 × layers × seq_len × d_model × dtype.                           │
│      128K context ≈ 0.5 GB (with GQA). Quantize to INT8 for 75% save. │
│                                                                          │
│   7. HYBRID RETRIEVAL WINS                                              │
│      Semantic + BM25 with RRF fusion outperforms either alone.          │
│      Re-rank with cross-encoders for precision.                         │
│                                                                          │
│   8. DIVERSITY MATTERS (MMR)                                             │
│      Top-5 by relevance often gives 5 copies of the same info.         │
│      MMR (λ=0.7) balances relevance with diversity.                     │
│                                                                          │
│   9. PRODUCTION = PIPELINE + OBSERVABILITY                              │
│      Log every retrieval score. Track hallucination rates.              │
│      Evaluate with RAGAS. A/B test everything.                          │
│                                                                          │
│  10. KNOW THE FULL STACK                                                │
│      Chunking → Embedding → Indexing → Retrieval → Re-Ranking →        │
│      Compression → Token Budgeting → Prompt Assembly → Evaluation       │
│      Being able to discuss each stage signals senior-level depth.       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

**Quick-Reference Cheat Sheet:**

| Concept | Key Number | When to Cite |
|---------|-----------|--------------|
| Chunk overlap | 15% | "How do you chunk documents?" |
| HNSW memory | ~6 KB/vector | "How much RAM for vector search?" |
| Compression ratio | 6-8x | "How do you reduce token usage?" |
| MMR lambda | 0.7 (relevance-heavy) | "How do you ensure diverse retrieval?" |
| KV-cache formula | 2 × L × S × D × dtype | "Memory estimation for LLM inference?" |
| 128K cache (GQA) | ~0.5 GB | "How much RAM for long context?" |
| KV quant savings | up to 87% (INT4) | "How do you optimize inference memory?" |
| Lost-in-middle | Begin + End > Middle | "How do you order context chunks?" |
| Hybrid alpha | 0.7 semantic / 0.3 BM25 | "Best retrieval strategy?" |
| Top-K → Rerank | 20 → 5 | "How many chunks to retrieve?" |

---

*Prepared for Data Scientist / ML Engineer interviews — LLMs, RAG, and Production ML Systems.*
