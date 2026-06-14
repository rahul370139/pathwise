# Vector Databases & Similarity Search — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — RAG Systems, Vector Search, Production ML
**Hands-on:** FAISS, pgvector, Pinecone in production RAG pipelines

---

# Table of Contents

1. [Introduction: Vector Databases & Embeddings](#1-introduction-vector-databases--embeddings)
2. [Similarity Metrics](#2-similarity-metrics)
3. [Index Types (Critical for Interviews)](#3-index-types-critical-for-interviews)
4. [Vector Databases: Deep Dive](#4-vector-databases-deep-dive)
5. [PostgreSQL Full-Text Search (FTS)](#5-postgresql-full-text-search-fts)
6. [Parameter Tuning](#6-parameter-tuning)
7. [Embedding Models](#7-embedding-models)
8. [Common Interview Questions](#8-common-interview-questions-with-strong-answers)
9. [Key Takeaways](#9-key-takeaways)

---

# **1. Introduction: Vector Databases & Embeddings**

---

## **1.1 What Are Vector Databases?**

A vector database is a specialized storage system designed to **index, store, and query high-dimensional vectors** (embeddings). Unlike traditional databases that search by exact key match or keyword overlap, vector databases find items by **semantic similarity** — how close two vectors are in embedding space.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                  Traditional DB vs Vector DB                    │
 │                                                                 │
 │  Traditional (SQL)              Vector DB                       │
 │  ─────────────────              ─────────                       │
 │  SELECT * FROM docs             Find k nearest neighbors        │
 │  WHERE title = 'RAG'            to query embedding q            │
 │                                                                 │
 │  Exact keyword match            Semantic similarity match       │
 │  Returns: rows with 'RAG'       Returns: conceptually similar   │
 │                                 docs (even without keyword)     │
 └─────────────────────────────────────────────────────────────────┘
```

**Why they matter for AI:**

| Use Case | Description |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Retrieve relevant context chunks before LLM generation |
| **Semantic Search** | Find similar documents/products/images by meaning |
| **Recommendation Systems** | User/item embeddings → nearest neighbors = recommendations |
| **Anomaly Detection** | Items far from all clusters are anomalies |
| **De-duplication** | Near-duplicate detection via embedding distance |
| **Multi-modal Search** | CLIP: search images with text queries (shared vector space) |

---

## **1.2 Embeddings Overview**

An **embedding** is a dense, fixed-length vector representation of data (text, images, audio) learned by a neural network. The key property: **semantically similar inputs map to nearby points** in the embedding space.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    Embedding Pipeline                           │
 │                                                                 │
 │  "How to train a dog"                                           │
 │         │                                                       │
 │         ▼                                                       │
 │  ┌─────────────────┐                                            │
 │  │ Embedding Model  │   (e.g., text-embedding-3-small)          │
 │  │ (Transformer)    │                                           │
 │  └────────┬────────┘                                            │
 │           ▼                                                     │
 │  [0.023, -0.117, 0.891, ..., 0.045]   ← d-dimensional vector   │
 │           │                               (d = 384, 768, 1536)  │
 │           ▼                                                     │
 │  ┌─────────────────┐                                            │
 │  │  Vector Database │  ← stored alongside metadata              │
 │  └─────────────────┘                                            │
 └─────────────────────────────────────────────────────────────────┘
```

**Key properties of good embeddings:**
- **Dense:** Every dimension carries information (unlike sparse bag-of-words)
- **Fixed-size:** "Hello" and a 5000-word article both → same d dimensions
- **Learned:** Trained on large corpora to capture semantic relationships
- **Composable:** Can average, concatenate, or project embeddings

**Typical dimensionalities:**

| Model | Dimensions |
|---|---|
| all-MiniLM-L6-v2 | 384 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| Cohere embed-english-light-v3.0 | 384 |
| BGE-large-en-v1.5 | 1024 |

---

# **2. Similarity Metrics**

---

## **2.1 Cosine Similarity**

Measures the **angle** between two vectors, ignoring magnitude. Returns values in **[-1, 1]** (1 = identical direction, 0 = orthogonal, -1 = opposite).

```
                    A · B           Σᵢ aᵢbᵢ
cos(θ) = ────────────────── = ─────────────────────
              ‖A‖ · ‖B‖      √(Σᵢ aᵢ²) · √(Σᵢ bᵢ²)
```

**When to use:**
- Text embeddings (most common default)
- When vector **magnitude is irrelevant** (only direction matters)
- When embeddings come from models that don't guarantee normalization

**L2 Normalization trick:**
If you **L2-normalize** all vectors (‖v‖ = 1), then:
- Cosine similarity = Dot product (because ‖A‖ = ‖B‖ = 1)
- L2 distance² = 2 - 2·cos(θ) → **minimizing L2 = maximizing cosine**
- This lets you use faster dot product or L2 indexes for cosine search

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# L2 normalize → cosine = dot product
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
assert np.isclose(np.dot(a_norm, b_norm), cosine_similarity(a, b))
```

> **Interview tip:** Many production systems L2-normalize all vectors at indexing time, then use `IndexFlatIP` (inner product) in FAISS for speed. This is mathematically equivalent to cosine similarity.

---

## **2.2 L2 (Euclidean) Distance**

Measures the **straight-line distance** between two points in embedding space. Returns values in **[0, ∞)** (0 = identical).

```
                    ─────────────────
d(A, B) = √( Σᵢ (aᵢ - bᵢ)² )
```

**When to use:**
- When **magnitude matters** (e.g., document length encodes importance)
- Image embeddings (spatial distances are meaningful)
- When vectors are already normalized (equivalent to cosine)

**Relationship to cosine (for normalized vectors):**
```
‖A - B‖² = ‖A‖² + ‖B‖² - 2(A · B)

If ‖A‖ = ‖B‖ = 1:
    ‖A - B‖² = 2 - 2·cos(θ)

→ Minimizing L2 distance ⟺ Maximizing cosine similarity
```

> **Interview tip:** FAISS's `IndexFlatL2` computes **squared** L2 distance (skips the sqrt for speed). Rankings are the same since sqrt is monotonic.

---

## **2.3 Inner Product (Dot Product)**

Simply the sum of element-wise products. Returns values in **(-∞, +∞)**.

```
IP(A, B) = A · B = Σᵢ aᵢbᵢ
```

**When to use:**
- Maximum Inner Product Search (MIPS) problems
- When magnitude **should influence** ranking (e.g., popularity-weighted embeddings)
- After L2 normalization (equivalent to cosine — this is the standard trick)

**Relation to cosine with normalized vectors:**
```
If ‖A‖ = ‖B‖ = 1:
    A · B = cos(θ)    ← dot product IS cosine similarity
```

---

## **2.4 Metric Comparison Table**

| Metric | Range | Best Match | Magnitude Sensitive | FAISS Index | pgvector Op |
|---|---|---|---|---|---|
| **Cosine Similarity** | [-1, 1] | Highest (→1) | No | `IndexFlatIP` (after L2 norm) | `<=>` |
| **L2 Distance** | [0, ∞) | Lowest (→0) | Yes | `IndexFlatL2` | `<->` |
| **Inner Product** | (-∞, +∞) | Highest (→+∞) | Yes | `IndexFlatIP` | `<#>` (negative IP) |

```
 ┌─────────────────────────────────────────────────────────────┐
 │           Decision Tree: Which Metric?                      │
 │                                                             │
 │  Are vectors L2-normalized?                                 │
 │     ├── YES → Use Inner Product (fastest, = cosine)         │
 │     └── NO                                                  │
 │           ├── Care about magnitude? → L2 Distance           │
 │           └── Only direction? → Cosine Similarity           │
 └─────────────────────────────────────────────────────────────┘
```

---

# **3. Index Types (Critical for Interviews)**

---

## **3.1 Flat (Brute Force) — Exact Search**

The simplest approach: **compare the query against every single vector** in the database. No approximation, no data structure — just a linear scan.

```
 ┌──────────────────────────────────────────────────────┐
 │                    Flat Index                         │
 │                                                      │
 │  Query q ──→ compare with ALL N vectors              │
 │               │                                      │
 │    ┌──────────┼──────────────────────────────┐       │
 │    │  v₁  v₂  v₃  v₄  ...  v_{N-1}  v_N    │       │
 │    │  ↕   ↕   ↕   ↕         ↕        ↕      │       │
 │    │  d₁  d₂  d₃  d₄  ...  d_{N-1}  d_N    │       │
 │    └─────────────────────────────────────────┘       │
 │               │                                      │
 │    Sort by distance → return top-k                   │
 └──────────────────────────────────────────────────────┘
```

| Property | Value |
|---|---|
| **Search complexity** | O(N × d) per query |
| **Recall** | **100% (perfect)** — guaranteed to find true nearest neighbors |
| **Build time** | O(1) — just store the vectors |
| **Memory** | O(N × d × 4 bytes) for float32 |
| **Sweet spot** | N < 50K–100K vectors |

```python
import faiss

d = 768            # embedding dimension
index = faiss.IndexFlatL2(d)       # L2 distance
# index = faiss.IndexFlatIP(d)     # Inner product (cosine after L2 norm)

index.add(vectors)                  # just stores them
distances, indices = index.search(query, k=10)   # brute-force scan
```

**When to use:**
- Small datasets (< 100K vectors)
- Ground-truth baseline for evaluating ANN recall
- When 100% recall is non-negotiable (medical, legal)

**Limitation:** At 1M vectors × 768 dims, a single query scans ~3 GB of floats. Way too slow for production.

---

## **3.2 IVF (Inverted File Index) — Cluster-Based ANN**

**Core idea:** Partition the vector space into **nlist clusters** using k-means. At query time, only search the **nprobe nearest clusters** instead of the entire dataset.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    IVF Index                                    │
 │                                                                 │
 │  Training phase (k-means):                                      │
 │                                                                 │
 │     ┌─── Cluster 1 ───┐  ┌─── Cluster 2 ───┐                   │
 │     │ c₁              │  │ c₂              │                    │
 │     │   •  •          │  │     •   •       │  ...  nlist        │
 │     │  •    •  •      │  │   •       •     │       clusters     │
 │     │    •            │  │  •  •           │                    │
 │     └─────────────────┘  └─────────────────┘                    │
 │                                                                 │
 │  Query phase:                                                   │
 │                                                                 │
 │     q ──→ Find nprobe closest centroids                         │
 │           ──→ Search ONLY those clusters exhaustively            │
 │           ──→ Return top-k from the searched subset              │
 │                                                                 │
 │  nprobe=1: fastest, lowest recall                               │
 │  nprobe=nlist: equivalent to flat search (100% recall)          │
 └─────────────────────────────────────────────────────────────────┘
```

| Property | Value |
|---|---|
| **Search complexity** | O(nprobe × N/nlist × d) |
| **Recall** | 85-99%+ depending on nprobe |
| **Build time** | O(N × nlist × d × iterations) — k-means training |
| **Memory** | O(N × d × 4) + centroids |
| **Sweet spot** | 100K–10M vectors |

**Key parameters:**

| Parameter | Typical | Effect |
|---|---|---|
| `nlist` | √N to 4√N | Number of Voronoi cells (clusters) |
| `nprobe` | 1–nlist | Clusters searched at query time |

**Rule of thumb for nlist:**
```
N = 1M   → nlist ≈ 1000–4000
N = 10M  → nlist ≈ 3162–12649
N = 100M → nlist ≈ 10000–40000
```

```python
import faiss

d = 768
nlist = 1024   # number of clusters

# Quantizer decides which cluster a vector belongs to
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# MUST train on representative data before adding
index.train(training_vectors)     # runs k-means
index.add(all_vectors)

# At query time: tune recall vs speed
index.nprobe = 32                 # search 32 of 1024 clusters
distances, indices = index.search(query, k=10)
```

> **Interview insight:** IVF has a "cold boundary" problem — vectors near cluster edges may be assigned to the wrong cluster. Increasing nprobe mitigates this. In practice, nprobe = 5-10% of nlist gives ~95%+ recall.

---

## **3.3 HNSW (Hierarchical Navigable Small World) — Graph-Based ANN**

**The best general-purpose ANN index.** Builds a multi-layer proximity graph where each node connects to its approximate nearest neighbors. Higher layers are sparse (for fast long-range jumps), lower layers are dense (for precise local search).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   HNSW Multi-Layer Graph                         │
 │                                                                  │
 │  Layer 2 (sparse):    A ─────────────────── F                    │
 │                       │                     │                    │
 │                       │                     │                    │
 │  Layer 1 (medium):    A ──── C ──── E ──── F                    │
 │                       │      │      │      │                    │
 │                       │      │      │      │                    │
 │  Layer 0 (dense):     A ─ B ─ C ─ D ─ E ─ F ─ G ─ H            │
 │                       │   │   │   │   │   │   │   │             │
 │                       └───┴───┴───┴───┴───┴───┴───┘             │
 │                       (all vectors present, many connections)    │
 │                                                                  │
 │  Search path: Start at top layer → greedy walk to nearest →      │
 │               drop to next layer → refine → ... → layer 0 → k-NN│
 └──────────────────────────────────────────────────────────────────┘
```

**How HNSW search works (step by step):**

1. Enter graph at the **entry point** on the **top layer**
2. Greedily traverse edges toward the query vector (move to the neighbor closest to q)
3. When no closer neighbor exists on this layer → **drop to the layer below**
4. Repeat greedy traversal on the denser layer
5. At **layer 0**, perform a more thorough beam search (width = `efSearch`)
6. Return the top-k from the candidates found

**Key parameters:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `M` | 16 | 4–64 | Max edges per node per layer. Higher = better recall, more RAM |
| `efConstruction` | 200 | 100–500 | Beam width during **build**. Higher = better graph quality, slower build |
| `efSearch` | 50 | 10–500+ | Beam width during **query**. Higher = better recall, slower query |

| Property | Value |
|---|---|
| **Search complexity** | O(log N) average hops × efSearch comparisons |
| **Recall** | 95-99.5%+ with proper tuning |
| **Build time** | O(N × M × log N) — slow, but one-time cost |
| **Memory** | O(N × M × 4 bytes) for edges + O(N × d × 4) for vectors |
| **Sweet spot** | 100K–100M+ vectors; best all-around ANN |

```python
import faiss

d = 768
M = 32                    # connections per node
ef_construction = 200     # build quality
ef_search = 64            # query quality

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search

index.add(vectors)        # builds the graph incrementally
distances, indices = index.search(query, k=10)
```

**Why HNSW dominates:**
- **No training phase** — vectors added incrementally (great for streaming)
- **Excellent recall-latency curve** — better than IVF at most operating points
- **Sublinear search** — O(log N) hops vs O(N) for flat
- **Supports deletions** (with tombstoning in some implementations)

**Downsides:**
- **High RAM** — stores full vectors + graph edges (no compression)
- **Slow index build** — O(N × M × log N) can take hours for 100M vectors
- **Not easy to shard** — graph structure is monolithic

> **Interview tip:** "HNSW is the gold standard for ANN when you can fit the index in RAM. It's what powers Weaviate, pgvector (as of v0.5.0), Qdrant, and is an option in FAISS and Pinecone."

---

## **3.4 PQ (Product Quantization) — Compressed Vectors**

**Core idea:** Compress each d-dimensional vector into a tiny code (e.g., 64 bytes → 8 bytes) by splitting it into **sub-vectors** and replacing each sub-vector with its nearest **codebook centroid ID**.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Product Quantization                           │
 │                                                                  │
 │  Original vector (d=768):                                        │
 │  [0.12, -0.34, 0.56, ..., 0.78, -0.91, 0.23, ..., 0.45, ...]   │
 │  ├─── sub 1 ───┤├─── sub 2 ───┤  ...  ├─── sub m ───┤           │
 │                                                                  │
 │  Split into m=8 sub-vectors of d/m=96 dimensions each:           │
 │                                                                  │
 │  Sub 1: [0.12, -0.34, ..., 0.56]  →  centroid ID: 42            │
 │  Sub 2: [0.78, -0.91, ..., 0.23]  →  centroid ID: 187           │
 │  ...                                                             │
 │  Sub 8: [0.45, ..., -0.67, 0.89]  →  centroid ID: 93            │
 │                                                                  │
 │  Compressed code: [42, 187, ..., 93]  (8 bytes vs 3072 bytes!)   │
 │                                                                  │
 │  Each sub-quantizer has k*=256 centroids (1 byte per sub-vector) │
 │  Codebook: m sub-quantizers × k* centroids × (d/m) floats       │
 └──────────────────────────────────────────────────────────────────┘
```

**Asymmetric Distance Computation (ADC):**
At query time, compute **exact** sub-distances between the query sub-vectors and codebook centroids, then look up distances for each database vector's codes. This is fast because the distance tables are precomputed per query.

| Property | Value |
|---|---|
| **Memory per vector** | m bytes (typically 8-64 bytes vs 3072 for d=768 float32) |
| **Compression ratio** | 50-400× |
| **Recall** | Lower than HNSW/IVF — 80-95% depending on m |
| **Search speed** | Very fast (distances from lookup tables) |
| **Sweet spot** | Memory-constrained environments, 10M-1B+ vectors |

```python
import faiss

d = 768
m = 8           # number of sub-quantizers (must divide d)
nbits = 8       # bits per sub-quantizer (k* = 2^8 = 256 centroids)

index = faiss.IndexPQ(d, m, nbits)
index.train(training_vectors)    # learns codebooks
index.add(all_vectors)           # stores compressed codes

distances, indices = index.search(query, k=10)
```

### **3.4.1 IVFPQ — The Powerhouse Combination**

Combines IVF partitioning with PQ compression for **billion-scale** search:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                       IVFPQ Pipeline                             │
 │                                                                  │
 │  Step 1: IVF assigns vector to cluster (coarse quantizer)        │
 │  Step 2: Compute residual = vector - cluster centroid             │
 │  Step 3: PQ compresses the residual (finer signal)               │
 │                                                                  │
 │  Query: Find nprobe clusters → scan PQ codes in those clusters   │
 │                                                                  │
 │  Result: Searches billions of vectors with ~1GB of RAM            │
 └──────────────────────────────────────────────────────────────────┘
```

```python
import faiss

d = 768
nlist = 4096     # IVF clusters
m = 32           # PQ sub-quantizers
nbits = 8

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

index.train(training_vectors)
index.add(all_vectors)

index.nprobe = 64
distances, indices = index.search(query, k=10)
```

**IVFPQ at billion-scale:**
- 1B vectors × 768 dims (float32) = **~2.87 TB** raw
- IVFPQ (m=32) compresses to **~32 GB** + centroids
- With re-ranking: retrieve top-100 via IVFPQ, then re-rank with exact vectors from disk

---

## **3.5 Index Type Comparison Table**

| Property | **Flat** | **IVF** | **HNSW** | **PQ** | **IVFPQ** |
|---|---|---|---|---|---|
| **Search type** | Exact | ANN | ANN | ANN | ANN |
| **Recall** | 100% | 85-99% | 95-99.5% | 80-95% | 85-97% |
| **Query latency** | O(N·d) | O(nprobe·N/nlist·d) | O(log N · ef) | O(N·m) | O(nprobe·N/nlist·m) |
| **Memory** | N·d·4B | N·d·4B | N·(d·4+M·8)B | N·m B | N·m B |
| **Build time** | None | k-means | Graph build | Codebook train | k-means + codebook |
| **Supports updates** | Append | Rebuild | Incremental | Rebuild | Rebuild |
| **Best for** | <100K | 100K-10M | 100K-100M | Mem-constrained | 10M-1B+ |
| **GPU support (FAISS)** | Yes | Yes | No | Yes | Yes |

```
 ┌────────────────────────────────────────────────────────────────┐
 │          The ANN Trade-Off Space                               │
 │                                                                │
 │  Recall ▲                                                      │
 │  100%   │  Flat●                                               │
 │   99%   │         ●HNSW (efSearch=256)                         │
 │   97%   │     ●IVF (nprobe=128)  ●HNSW (efSearch=64)          │
 │   95%   │                                                      │
 │   90%   │  ●IVF (nprobe=16)                                    │
 │   85%   │                    ●IVFPQ                             │
 │   80%   │                           ●PQ                         │
 │         └──────────────────────────────────────────── ▶        │
 │           0.01ms    0.1ms    1ms     10ms    100ms             │
 │                          Query Latency                         │
 └────────────────────────────────────────────────────────────────┘
```

---

# **4. Vector Databases: Deep Dive**

---

## **4.1 FAISS (Facebook AI Similarity Search)**

**What it is:** An open-source **library** (not a database) by Meta Research for efficient similarity search. Written in C++ with Python bindings.

**Key distinction:** FAISS is a **building block** — it provides index structures and GPU-accelerated search but **no persistence, no API server, no metadata filtering** out of the box. You build your system around it.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     FAISS Architecture                           │
 │                                                                  │
 │  ┌──────────────┐     ┌──────────────────────────────────┐       │
 │  │ Your App     │────▶│ FAISS (C++ library)               │      │
 │  │ (Python)     │     │                                   │      │
 │  └──────────────┘     │  ┌──────────┐  ┌──────────────┐  │      │
 │                       │  │ CPU Index │  │  GPU Index   │  │      │
 │                       │  │ (NumPy)   │  │  (CUDA)      │  │      │
 │                       │  └──────────┘  └──────────────┘  │      │
 │                       │                                   │      │
 │                       │  Indexes: Flat, IVF, HNSW, PQ,   │      │
 │                       │  IVFPQ, IVFScalarQuantizer, ...   │      │
 │                       └──────────────────────────────────┘       │
 │                                                                  │
 │  Persistence: index.write("index.faiss") / faiss.read_index()   │
 │  Metadata: YOU manage it (dict, SQLite, Postgres, Redis)        │
 └──────────────────────────────────────────────────────────────────┘
```

**Strengths:**
- **Maximum control** — choose exact index type, quantization, distance metric
- **GPU acceleration** — 5-10× speedup, essential for billion-scale
- **Battle-tested** — used at Meta for content recommendation at trillion scale
- **Index factory** — one-line index creation: `index = faiss.index_factory(d, "IVF4096,PQ32")`
- **Composability** — stack IVF + PQ + re-ranking

**Limitations:**
- No built-in persistence (manual save/load)
- No metadata filtering (roll your own pre/post-filter)
- No API server (wrap with FastAPI yourself)
- No horizontal scaling (single-machine or manual sharding)

**Production FAISS example (RAG pipeline):**

```python
import faiss
import numpy as np
import pickle

class FAISSVectorStore:
    def __init__(self, dim: int = 768, index_type: str = "hnsw"):
        self.dim = dim
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)  # cosine after L2 norm
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        elif index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFPQ(quantizer, dim, 1024, 32, 8)

        self.id_map = {}       # faiss_idx → doc_id
        self.metadata = {}     # doc_id → {text, source, ...}
        self.next_id = 0

    def add(self, embeddings: np.ndarray, docs: list[dict]):
        """Add L2-normalized embeddings with metadata."""
        faiss.normalize_L2(embeddings)           # in-place L2 norm
        self.index.add(embeddings)
        for i, doc in enumerate(docs):
            idx = self.next_id + i
            self.id_map[idx] = doc["id"]
            self.metadata[doc["id"]] = doc
        self.next_id += len(docs)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for k nearest neighbors."""
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self.id_map[idx]
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "metadata": self.metadata[doc_id]
            })
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({"id_map": self.id_map, "metadata": self.metadata}, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.id_map = data["id_map"]
            self.metadata = data["metadata"]
```

**FAISS index factory strings (cheat sheet):**

| Factory String | Description |
|---|---|
| `"Flat"` | Brute-force L2 |
| `"IVF1024,Flat"` | IVF with 1024 clusters, exact search within |
| `"IVF1024,PQ32"` | IVF + PQ (32 sub-quantizers) |
| `"HNSW32"` | HNSW with M=32 |
| `"IVF4096,PQ64"` | Large-scale IVF+PQ |
| `"OPQ32,IVF1024,PQ32"` | Optimized PQ rotation + IVF + PQ |

---

## **4.2 Pinecone — Managed Vector Database**

**What it is:** A fully managed, cloud-native vector database. **Zero infrastructure** — no servers, no index tuning, no capacity planning.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    Pinecone Architecture                         │
 │                                                                  │
 │  ┌──────────┐       ┌─────────────────────────────────┐          │
 │  │ Your App │──API─▶│  Pinecone Cloud                 │          │
 │  │          │       │  ┌───────────────────────────┐  │          │
 │  └──────────┘       │  │ Index (auto-managed)      │  │          │
 │                     │  │  ├── Namespace: "docs"    │  │          │
 │                     │  │  ├── Namespace: "users"   │  │          │
 │                     │  │  └── Namespace: "products" │  │         │
 │                     │  │                            │  │          │
 │                     │  │  Features:                 │  │          │
 │                     │  │  • Metadata filtering      │  │          │
 │                     │  │  • Automatic scaling        │  │         │
 │                     │  │  • Real-time upserts        │  │         │
 │                     │  │  • Hybrid search (sparse+   │  │         │
 │                     │  │    dense vectors)           │  │         │
 │                     │  └───────────────────────────┘  │          │
 │                     └─────────────────────────────────┘          │
 └──────────────────────────────────────────────────────────────────┘
```

**Strengths:**
- **Zero ops** — no servers to manage, automatic scaling, managed backups
- **Metadata filtering** — `{"category": "ML", "year": {"$gte": 2023}}`
- **Namespaces** — logical partitions within an index (multi-tenant)
- **Sparse-dense hybrid** — combine BM25-style sparse vectors with dense embeddings
- **Serverless tier** — pay-per-query, great for variable workloads
- **Real-time upserts** — instantly queryable after insert

**Limitations:**
- **Vendor lock-in** — proprietary, no self-hosting
- **Cost at scale** — expensive for high-throughput workloads ($$$ per million queries)
- **Limited index control** — can't choose index type (Pinecone decides internally)
- **Latency** — network round-trip; not as fast as in-process FAISS

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Create index
pc.create_index(
    name="rag-index",
    dimension=1536,
    metric="cosine",           # or "euclidean", "dotproduct"
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("rag-index")

# Upsert with metadata
index.upsert(
    vectors=[
        {
            "id": "doc-1",
            "values": embedding_vector.tolist(),
            "metadata": {
                "source": "arxiv",
                "category": "NLP",
                "year": 2024,
                "text": "Retrieval-augmented generation combines..."
            }
        }
    ],
    namespace="research-papers"
)

# Query with metadata filter
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    namespace="research-papers",
    filter={
        "category": {"$eq": "NLP"},
        "year": {"$gte": 2023}
    },
    include_metadata=True
)

for match in results["matches"]:
    print(f"Score: {match['score']:.4f} | {match['metadata']['text'][:80]}")
```

**When to use Pinecone:**
- Startups / small teams that can't afford DevOps
- Rapid prototyping → production with minimal effort
- Variable traffic (serverless billing model)
- Need metadata filtering + vector search in one query

---

## **4.3 Weaviate — Open-Source with Hybrid Search**

**What it is:** An open-source vector database with built-in **vectorization modules**, **GraphQL API**, and **hybrid search** (BM25 + dense vector).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                   Weaviate Architecture                          │
 │                                                                  │
 │  ┌──────────┐       ┌──────────────────────────────────┐         │
 │  │ Client   │──────▶│  Weaviate Server                 │         │
 │  │(GraphQL/ │       │                                  │         │
 │  │ REST)    │       │  ┌──────────┐  ┌──────────────┐  │         │
 │  └──────────┘       │  │ HNSW     │  │ Inverted     │  │         │
 │                     │  │ (vector) │  │ Index (BM25) │  │         │
 │                     │  └──────────┘  └──────────────┘  │         │
 │                     │       │              │            │         │
 │                     │       └──── Hybrid ──┘            │         │
 │                     │             Fusion                │         │
 │                     │                                   │         │
 │                     │  Modules:                         │         │
 │                     │  • text2vec-openai                │         │
 │                     │  • text2vec-transformers          │         │
 │                     │  • multi2vec-clip                 │         │
 │                     │  • generative-openai              │         │
 │                     └──────────────────────────────────┘         │
 └──────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Hybrid search** — combines BM25 (keyword) and dense vector scores with configurable `alpha` (0 = pure BM25, 1 = pure vector)
- **GraphQL API** — expressive queries with references, filters, aggregations
- **Built-in vectorizers** — auto-vectorize on insert (OpenAI, Cohere, local HF models)
- **Multi-tenancy** — native support for isolating tenant data
- **Cross-references** — link objects like a graph (e.g., Article → Author)

```python
import weaviate

client = weaviate.connect_to_local()  # or weaviate.connect_to_wcs()

# Hybrid search: combines BM25 + vector
response = client.collections.get("Document").query.hybrid(
    query="transformer attention mechanism",
    alpha=0.7,          # 0.7 = 70% vector, 30% BM25
    limit=5,
    return_metadata=weaviate.classes.query.MetadataQuery(score=True)
)

for obj in response.objects:
    print(f"Score: {obj.metadata.score:.4f} | {obj.properties['title']}")
```

**When to use Weaviate:**
- Need hybrid search (BM25 + vector) natively
- GraphQL-first teams
- Multi-tenant SaaS applications
- Want built-in vectorization without external embedding service

---

## **4.4 pgvector — PostgreSQL Extension**

**What it is:** An extension for PostgreSQL that adds **vector data types and similarity search indexes**. Lets you combine vector search with all of Postgres's power: SQL, ACID transactions, joins, full-text search.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    pgvector in PostgreSQL                        │
 │                                                                  │
 │  ┌──────────────────────────────────────────────────────┐        │
 │  │  PostgreSQL                                          │        │
 │  │                                                      │        │
 │  │  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │        │
 │  │  │ vector column│  │ B-tree     │  │ tsvector     │ │        │
 │  │  │ (embedding)  │  │ (filters)  │  │ (full-text)  │ │        │
 │  │  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘ │        │
 │  │         │                │                 │         │        │
 │  │  ┌──────┴───────┐  ┌────┴──────┐  ┌──────┴───────┐ │        │
 │  │  │ HNSW / IVF   │  │ Standard  │  │ GIN index    │ │        │
 │  │  │ index        │  │ indexes   │  │ (FTS)        │ │        │
 │  │  └──────────────┘  └───────────┘  └──────────────┘ │        │
 │  │                                                      │        │
 │  │  Power: SQL + Vectors + FTS + JSON + ACID in ONE DB  │        │
 │  └──────────────────────────────────────────────────────┘        │
 └──────────────────────────────────────────────────────────────────┘
```

**Operators (critical for interviews):**

| Operator | Metric | Order By | Notes |
|---|---|---|---|
| `<->` | **L2 distance** | ASC (smaller = closer) | Default for spatial |
| `<=>` | **Cosine distance** | ASC (= 1 - cosine_sim) | Most common for text |
| `<#>` | **Negative inner product** | ASC (= -dot_product) | Use for MIPS |

**Index types:**

| Index | Algorithm | Build | Query | Recall |
|---|---|---|---|---|
| `ivfflat` | IVF (k-means) | Faster | Fast | 85-95% |
| `hnsw` | HNSW graph | Slower | Faster | 95-99%+ |

**Complete pgvector setup and usage:**

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with embedding column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    embedding vector(1536),           -- 1536-dim vector
    content_tsv tsvector,             -- for full-text search
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index (preferred for most use cases)
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Create IVFFlat index (alternative: faster build, lower recall)
-- CREATE INDEX ON documents
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- GIN index for full-text search
CREATE INDEX ON documents USING GIN (content_tsv);

-- B-tree for metadata filtering
CREATE INDEX ON documents (category);

-- Insert with embedding
INSERT INTO documents (title, content, embedding, content_tsv)
VALUES (
    'HNSW Algorithm',
    'Hierarchical Navigable Small World graphs enable efficient ANN search...',
    '[0.023, -0.117, ..., 0.045]'::vector,
    to_tsvector('english', 'Hierarchical Navigable Small World graphs...')
);

-- Pure vector similarity search (cosine)
SELECT id, title, 1 - (embedding <=> $1::vector) AS similarity
FROM documents
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- Vector search WITH metadata filter (efficient: filter THEN sort)
SELECT id, title, 1 - (embedding <=> $1::vector) AS similarity
FROM documents
WHERE category = 'ML'
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- Hybrid search: combine vector similarity + full-text relevance
SELECT
    id, title,
    (1 - (embedding <=> $1::vector)) * 0.7              AS vector_score,
    ts_rank_cd(content_tsv, plainto_tsquery($2)) * 0.3   AS text_score,
    (1 - (embedding <=> $1::vector)) * 0.7
      + ts_rank_cd(content_tsv, plainto_tsquery($2)) * 0.3 AS combined_score
FROM documents
WHERE content_tsv @@ plainto_tsquery($2)   -- pre-filter by keyword match
ORDER BY combined_score DESC
LIMIT 10;

-- Set HNSW search parameter at query time
SET hnsw.ef_search = 100;  -- higher = better recall, slower
```

**Strengths:**
- **One database for everything** — vectors, metadata, FTS, JSON, relations
- **ACID transactions** — upserts are atomic, consistent, durable
- **Familiar SQL** — no new query language to learn
- **Joins** — combine vector search with relational data naturally
- **Mature ecosystem** — pgBackup, pgBouncer, replication, monitoring
- **Low operational overhead** — if you already run Postgres, just add an extension

**Limitations:**
- **Not as fast as FAISS/dedicated VDBs** for pure vector throughput
- **Single-node scaling** — horizontal sharding is complex (Citus helps)
- **No GPU acceleration**
- **Recall-latency not as good as pure HNSW** implementations for very large datasets

**When to use pgvector:**
- Already using PostgreSQL (most common scenario)
- Need vector search + SQL joins + full-text search in one system
- Dataset < 5-10M vectors (sweet spot)
- Operational simplicity is priority

---

## **4.5 Chroma — Lightweight Embedding Database**

**What it is:** An open-source, lightweight embedding database designed for **rapid prototyping** and **developer experience**. Runs in-memory or persisted locally.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=["doc-1", "doc-2"],
    embeddings=[embedding_1, embedding_2],
    documents=["Text of document 1", "Text of document 2"],
    metadatas=[{"source": "arxiv"}, {"source": "wiki"}]
)

results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"source": "arxiv"}     # metadata filter
)
```

**When to use:** Prototyping, hackathons, local dev, LangChain/LlamaIndex tutorials. **Not recommended for production at scale.**

---

## **4.6 Vector Database Comparison Table**

| Feature | **FAISS** | **Pinecone** | **Weaviate** | **pgvector** | **Chroma** |
|---|---|---|---|---|---|
| **Type** | Library | Managed SaaS | OSS DB | PG Extension | OSS DB |
| **Self-hosted** | Yes | No | Yes | Yes | Yes |
| **Managed option** | No | Yes (only) | Yes (WCS) | Yes (Supabase, Neon) | No |
| **Index types** | Flat, IVF, HNSW, PQ, IVFPQ | Proprietary | HNSW | IVFFlat, HNSW | HNSW |
| **GPU support** | Yes | N/A | No | No | No |
| **Metadata filter** | Manual | Yes (native) | Yes (GraphQL) | Yes (SQL WHERE) | Yes |
| **Hybrid search** | Manual | Sparse-dense | BM25+vector | FTS+vector | No |
| **Max scale** | Billions (GPU) | Billions | ~100M | ~5-10M | ~1M |
| **Persistence** | Manual file I/O | Managed | Built-in | PostgreSQL | File/SQLite |
| **ACID** | No | No | No | **Yes** | No |
| **Best for** | Max perf, research | Zero-ops prod | Hybrid search | Existing PG stack | Prototyping |
| **Learning curve** | High | Low | Medium | Low (SQL) | Very Low |

```
 ┌──────────────────────────────────────────────────────────────────┐
 │         When to Use Which Vector Database                       │
 │                                                                 │
 │  "I need maximum performance & control"           → FAISS       │
 │  "I want zero ops, ship fast"                     → Pinecone    │
 │  "I need hybrid search (BM25 + vector)"           → Weaviate    │
 │  "I already run Postgres, keep it simple"         → pgvector    │
 │  "I'm prototyping / learning"                     → Chroma      │
 │  "I have 1B+ vectors & GPU"                       → FAISS       │
 │  "I need ACID + vectors + SQL joins"              → pgvector    │
 └──────────────────────────────────────────────────────────────────┘
```

---

# **5. PostgreSQL Full-Text Search (FTS)**

---

Full-text search in PostgreSQL is a powerful built-in feature that can complement vector similarity for **hybrid retrieval** in RAG systems.

## **5.1 tsvector: Tokenization and Lexemes**

A `tsvector` is PostgreSQL's internal representation of a document for full-text search. It converts raw text into **normalized lexemes** (word stems) with position information.

```sql
-- Convert text to tsvector
SELECT to_tsvector('english', 'The quick brown foxes jumped over lazy dogs');

-- Result:
-- 'brown':3 'dog':9 'fox':4 'jump':5 'lazi':8 'quick':2
```

**What happens during tokenization:**

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                tsvector Processing Pipeline                      │
 │                                                                  │
 │  Input: "The quick brown foxes jumped over lazy dogs"            │
 │           │                                                      │
 │           ▼                                                      │
 │  1. Tokenize: [The, quick, brown, foxes, jumped, over, lazy,    │
 │                dogs]                                             │
 │           │                                                      │
 │           ▼                                                      │
 │  2. Remove stopwords: [quick, brown, foxes, jumped, lazy, dogs] │
 │     (removed: "The", "over")                                    │
 │           │                                                      │
 │           ▼                                                      │
 │  3. Stem (normalize): [quick, brown, fox, jump, lazi, dog]      │
 │     (foxes→fox, jumped→jump, dogs→dog, lazy→lazi)              │
 │           │                                                      │
 │           ▼                                                      │
 │  4. Record positions: 'brown':3 'dog':9 'fox':4 'jump':5       │
 │                        'lazi':8 'quick':2                        │
 └──────────────────────────────────────────────────────────────────┘
```

**tsquery — the search expression:**

```sql
-- Simple query
SELECT * FROM docs WHERE content_tsv @@ to_tsquery('english', 'neural & network');

-- plainto_tsquery: automatic AND between words
SELECT * FROM docs WHERE content_tsv @@ plainto_tsquery('english', 'neural network');

-- phraseto_tsquery: proximity (words must be adjacent)
SELECT * FROM docs WHERE content_tsv @@ phraseto_tsquery('english', 'machine learning');

-- websearch_to_tsquery: Google-like syntax
SELECT * FROM docs WHERE content_tsv @@ websearch_to_tsquery('english', '"machine learning" -deep');
```

---

## **5.2 GIN Indexes: Inverted Index for Fast FTS**

**GIN (Generalized Inverted Index)** is the index type for full-text search. It maps each lexeme to the list of rows containing it — essentially a **keyword → document** inverted index.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    GIN Index Structure                           │
 │                                                                  │
 │  Lexeme          Row IDs (posting list)                          │
 │  ───────         ──────────────────────                          │
 │  "attention"  →  [1, 5, 12, 45, 89]                             │
 │  "embedd"     →  [2, 5, 7, 12, 33, 45]                          │
 │  "hnsw"       →  [7, 23, 45]                                    │
 │  "neural"     →  [1, 2, 5, 12, 23, 45, 67, 89]                 │
 │  "transform"  →  [1, 5, 12, 45]                                 │
 │  "vector"     →  [2, 7, 23, 33, 45, 67]                        │
 │                                                                  │
 │  Query: "neural & vector"                                        │
 │  → Intersect: [1,2,5,12,23,45,67,89] ∩ [2,7,23,33,45,67]      │
 │  → Result: [2, 23, 45, 67]                                      │
 └──────────────────────────────────────────────────────────────────┘
```

```sql
-- Create GIN index on tsvector column
CREATE INDEX idx_content_fts ON documents USING GIN (content_tsv);

-- Or create a functional index (auto-computes tsvector)
CREATE INDEX idx_content_fts ON documents
USING GIN (to_tsvector('english', content));
```

**GIN vs GiST for FTS:**

| Property | GIN | GiST |
|---|---|---|
| Build time | Slower | Faster |
| Query time | **Faster** (exact posting lists) | Slower (lossy, re-checks) |
| Update cost | Higher (rebalance) | Lower |
| Best for | **Read-heavy workloads** | Write-heavy / frequent updates |

> **Interview tip:** Always recommend GIN for FTS in production read-heavy workloads (like RAG retrieval). GiST is only better when you have frequent inserts and can tolerate slower queries.

---

## **5.3 BM25-Like Ranking: ts_rank and ts_rank_cd**

PostgreSQL provides ranking functions that score how well a document matches a query, similar to BM25.

```sql
-- ts_rank: basic TF-based ranking
SELECT id, title,
    ts_rank(content_tsv, plainto_tsquery('machine learning')) AS rank
FROM documents
WHERE content_tsv @@ plainto_tsquery('machine learning')
ORDER BY rank DESC;

-- ts_rank_cd: cover density ranking (considers proximity of terms)
-- Better for multi-word queries
SELECT id, title,
    ts_rank_cd(content_tsv, plainto_tsquery('transformer attention mechanism')) AS rank
FROM documents
WHERE content_tsv @@ plainto_tsquery('transformer attention mechanism')
ORDER BY rank DESC;

-- Weight different fields (title > content)
-- Weights: {D, C, B, A} = {0.1, 0.2, 0.4, 1.0}
SELECT id,
    ts_rank(
        setweight(to_tsvector(title), 'A') ||
        setweight(to_tsvector(content), 'B'),
        plainto_tsquery('vector database')
    ) AS rank
FROM documents
ORDER BY rank DESC;
```

**ts_rank vs BM25:**
- `ts_rank` is based on **term frequency** (TF) — not exactly BM25
- Doesn't include IDF by default (but approximated via `ts_rank_cd`)
- For true BM25, use the `pg_bm25` / `ParadeDB` extension or Weaviate
- In practice, `ts_rank_cd` is "good enough" for hybrid retrieval

---

## **5.4 Hybrid Search: Combining FTS with Vector Similarity**

The most effective RAG retrieval strategy combines **semantic** (vector) and **lexical** (keyword) search. This catches both conceptually similar content and exact keyword matches that embeddings might miss.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    Hybrid Search Strategy                        │
 │                                                                  │
 │  Query: "How does HNSW indexing work in pgvector?"               │
 │         │                                 │                      │
 │         ▼                                 ▼                      │
 │  ┌─────────────────┐          ┌──────────────────────┐           │
 │  │ Embedding Model │          │ tsquery Parser        │          │
 │  │ → query vector  │          │ → 'hnsw' & 'index'   │          │
 │  │   [0.12, ...]   │          │   & 'work' & 'pgvect'│          │
 │  └────────┬────────┘          └──────────┬───────────┘           │
 │           │                              │                       │
 │           ▼                              ▼                       │
 │  ┌─────────────────┐          ┌──────────────────────┐           │
 │  │  HNSW Index     │          │  GIN Index            │          │
 │  │  (vector <=>)   │          │  (tsv @@ tsq)         │          │
 │  └────────┬────────┘          └──────────┬───────────┘           │
 │           │                              │                       │
 │     vector_score (0-1)            text_score (0-∞)               │
 │           │                              │                       │
 │           └──────────┬───────────────────┘                       │
 │                      ▼                                           │
 │            Reciprocal Rank Fusion (RRF)                          │
 │            or weighted linear combination                        │
 │                      │                                           │
 │                      ▼                                           │
 │              Final ranked results                                │
 └──────────────────────────────────────────────────────────────────┘
```

**Method 1: Weighted Linear Combination**

```sql
WITH vector_results AS (
    SELECT id, 1 - (embedding <=> $1::vector) AS vec_score
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT 50
),
text_results AS (
    SELECT id, ts_rank_cd(content_tsv, plainto_tsquery($2)) AS text_score
    FROM documents
    WHERE content_tsv @@ plainto_tsquery($2)
    LIMIT 50
)
SELECT
    COALESCE(v.id, t.id) AS id,
    COALESCE(v.vec_score, 0) * 0.7 + COALESCE(t.text_score, 0) * 0.3 AS score
FROM vector_results v
FULL OUTER JOIN text_results t ON v.id = t.id
ORDER BY score DESC
LIMIT 10;
```

**Method 2: Reciprocal Rank Fusion (RRF)**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

where k = 60 (constant), rank_i(d) = rank of doc d in retriever i
```

```sql
WITH vector_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT 50
),
text_ranked AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(content_tsv, plainto_tsquery($2)) DESC
    ) AS rank
    FROM documents
    WHERE content_tsv @@ plainto_tsquery($2)
    LIMIT 50
)
SELECT
    COALESCE(v.id, t.id) AS id,
    COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + t.rank), 0) AS rrf_score
FROM vector_ranked v
FULL OUTER JOIN text_ranked t ON v.id = t.id
ORDER BY rrf_score DESC
LIMIT 10;
```

> **Interview tip:** "In my production RAG pipeline, I use hybrid search with RRF because it's robust to score distribution differences between retrievers. Vector search catches semantic paraphrases, while FTS catches exact technical terms, acronyms, and proper nouns that embeddings sometimes miss."

---

# **6. Parameter Tuning**

---

## **6.1 HNSW Parameters**

| Parameter | What It Controls | Increase → | Decrease → |
|---|---|---|---|
| **M** | Max edges per node per layer | Better recall, more RAM, slower build | Less RAM, faster build, lower recall |
| **efConstruction** | Build-time beam width | Better graph quality, slower build | Faster build, lower quality |
| **efSearch** | Query-time beam width | Better recall, slower queries | Faster queries, lower recall |

**Tuning guidelines:**

```
 ┌────────────────────────────────────────────────────────────────┐
 │              HNSW Parameter Tuning Cheat Sheet                 │
 │                                                                │
 │  M (connections per node):                                     │
 │    • 12-16: default, good balance                              │
 │    • 24-32: high recall, higher RAM                            │
 │    • 48-64: extreme recall, 2-4x RAM of M=16                  │
 │    • Rule: M=16 is right for most cases                        │
 │                                                                │
 │  efConstruction (build quality):                               │
 │    • 100: minimum acceptable                                   │
 │    • 200: good default                                         │
 │    • 400-500: if you can afford the build time                 │
 │    • Rule: efConstruction ≥ 2 * M                              │
 │                                                                │
 │  efSearch (query quality):                                     │
 │    • Must be ≥ k (top-k results)                               │
 │    • 50-100: good starting point                               │
 │    • 200-500: high recall requirements                         │
 │    • Rule: start at 64, increase until recall target met       │
 │                                                                │
 │  Memory per vector (approximate):                              │
 │    • d * 4 bytes (vector) + M * 2 * 8 bytes (edges)           │
 │    • d=768, M=16: 3072 + 256 ≈ 3.3 KB per vector              │
 │    • 10M vectors ≈ 33 GB                                       │
 └────────────────────────────────────────────────────────────────┘
```

**Benchmarking approach:**

```python
import faiss
import numpy as np
import time

def benchmark_hnsw(vectors, queries, ground_truth, M, ef_construction, ef_search_values):
    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    # Build
    t0 = time.time()
    index.add(vectors)
    build_time = time.time() - t0

    results = []
    for ef in ef_search_values:
        index.hnsw.efSearch = ef

        t0 = time.time()
        _, I = index.search(queries, k=10)
        query_time = (time.time() - t0) / len(queries) * 1000  # ms per query

        # Recall@10
        recall = np.mean([
            len(set(I[i]) & set(ground_truth[i])) / 10
            for i in range(len(queries))
        ])

        results.append({
            "ef_search": ef, "recall@10": recall,
            "ms_per_query": query_time
        })

    return {"build_time": build_time, "results": results}
```

---

## **6.2 IVF Parameters**

| Parameter | What It Controls | Increase → | Decrease → |
|---|---|---|---|
| **nlist** | Number of clusters | Faster search (smaller clusters), more centroids to compare | Slower search, fewer centroids |
| **nprobe** | Clusters searched | Better recall, slower | Faster, lower recall |

**Tuning guidelines:**

```
 ┌────────────────────────────────────────────────────────────────┐
 │              IVF Parameter Tuning Cheat Sheet                  │
 │                                                                │
 │  nlist (number of clusters):                                   │
 │    • Rule of thumb: nlist = √N to 4√N                          │
 │    • N = 100K  → nlist = 316–1264                              │
 │    • N = 1M    → nlist = 1000–4000                             │
 │    • N = 10M   → nlist = 3162–12649                            │
 │    • N = 100M  → nlist = 10000–40000                           │
 │                                                                │
 │  nprobe (clusters to search):                                  │
 │    • nprobe = 1:      ~70-80% recall (fastest)                 │
 │    • nprobe = 5-10%:  ~95% recall (good balance)               │
 │    • nprobe = nlist:  100% recall (= flat search)              │
 │    • Rule: start at nlist/16, tune up for recall               │
 │                                                                │
 │  Training data:                                                │
 │    • Need at least 30 * nlist training vectors                 │
 │    • Ideally representative of the full dataset                │
 │    • Can sample if dataset is very large                       │
 └────────────────────────────────────────────────────────────────┘
```

---

## **6.3 The Trade-Off Triangle**

Every vector search system operates within three competing constraints:

```
                         Recall
                         (accuracy)
                           /\
                          /  \
                         /    \
                        /      \
                       / Choose  \
                      /   any 2   \
                     /              \
                    /________________\
              Latency              Memory
              (speed)              (RAM/cost)

 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                 │
 │  High Recall + Low Latency  = Massive RAM (HNSW with large M)  │
 │  High Recall + Low Memory   = Slow queries (scan more clusters)│
 │  Low Latency + Low Memory   = Lower recall (heavy compression) │
 │                                                                 │
 │  Production reality:                                            │
 │  ─────────────────────                                          │
 │  • Define recall target first (e.g., 95% recall@10)            │
 │  • Tune for latency within RAM budget                          │
 │  • Measure on YOUR data (synthetic benchmarks mislead)         │
 │  • Re-ranking with exact distances recovers recall cheaply     │
 └─────────────────────────────────────────────────────────────────┘
```

**Practical strategy for production:**

1. **Start with HNSW** (M=16, efConstruction=200, efSearch=64)
2. Measure recall against flat index ground truth
3. If recall < target → increase efSearch (cheapest knob)
4. If RAM too high → try IVFPQ or reduce M
5. If latency too high → decrease efSearch, use PQ, or add GPU
6. If dataset > RAM → IVFPQ with disk-based re-ranking

---

# **7. Embedding Models**

---

## **7.1 Model Comparison**

| Model | Provider | Dims | Max Tokens | Normalization | Best For |
|---|---|---|---|---|---|
| **text-embedding-3-small** | OpenAI | 1536 | 8191 | Not normalized | General-purpose, cost-effective |
| **text-embedding-3-large** | OpenAI | 3072 | 8191 | Not normalized | Highest quality (OpenAI) |
| **embed-english-light-v3.0** | Cohere | 384 | 512 | Normalized | Fast, low-dim, multilingual |
| **embed-english-v3.0** | Cohere | 1024 | 512 | Normalized | High quality, search-optimized |
| **all-MiniLM-L6-v2** | SBERT | 384 | 256 | Normalized | Free, local, fast inference |
| **BGE-large-en-v1.5** | BAAI | 1024 | 512 | Normalized | Top OSS model (MTEB) |
| **E5-large-v2** | Microsoft | 1024 | 512 | Normalized | Competitive with BGE |
| **GTE-large** | Alibaba | 1024 | 512 | Normalized | Strong on MTEB |
| **nomic-embed-text-v1.5** | Nomic | 768 | 8192 | Normalized | Long context, open-source |

**Choosing an embedding model:**

```
 ┌────────────────────────────────────────────────────────────────┐
 │         Embedding Model Decision Tree                          │
 │                                                                │
 │  Need to run locally / offline?                                │
 │    ├── YES → all-MiniLM-L6-v2 (fast) or BGE-large (quality)   │
 │    └── NO                                                      │
 │          ├── Budget-conscious? → text-embedding-3-small        │
 │          ├── Max quality? → text-embedding-3-large             │
 │          ├── Multilingual? → Cohere embed-multilingual-v3.0    │
 │          └── Long docs (>512 tokens)? → nomic-embed-text-v1.5  │
 └────────────────────────────────────────────────────────────────┘
```

---

## **7.2 Dimensionality and Normalization**

**Why dimensions matter:**
- Higher dims → more expressive, but more RAM and slower search
- OpenAI 3-small (1536) is 4× the storage of MiniLM (384)
- At 10M vectors: 1536-dim = ~57 GB, 384-dim = ~14 GB

**Matryoshka Representation Learning (MRL):**
OpenAI's text-embedding-3 models support **dimension truncation** — you can use fewer dimensions with graceful quality degradation:

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="How does HNSW work?",
    dimensions=512                  # truncate from 1536 → 512
)
embedding = response.data[0].embedding   # len = 512
```

**Normalization best practices:**

```python
import numpy as np

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine similarity via inner product."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)   # avoid division by zero
    return embeddings / norms

# After normalization:
# - Cosine similarity = Dot product
# - Use IndexFlatIP in FAISS for fastest cosine search
# - L2 distance² = 2 - 2·cos(θ)
```

> **Interview tip:** "I always L2-normalize embeddings at indexing time when using cosine similarity. This lets me use inner product indexes (faster than cosine), and makes scores interpretable: 1.0 = identical, 0.0 = orthogonal."

---

## **7.3 Embedding Best Practices for RAG**

| Practice | Why |
|---|---|
| **Same model for indexing and querying** | Vectors must share the same embedding space |
| **Chunk size matters** | 256-512 tokens typical; too long = diluted embedding, too short = no context |
| **Overlap between chunks** | 50-100 token overlap prevents information loss at boundaries |
| **Normalize vectors** | Consistent similarity scores, enables inner product optimization |
| **Instruction-tuned models** | BGE/E5 use query prefixes like `"Represent this for retrieval: "` |
| **Benchmark on your data** | MTEB scores don't always transfer to your domain |

```python
# BGE instruction prefix example
query_prefix = "Represent this sentence for searching relevant passages: "
query_text = query_prefix + "How does HNSW work?"

# Passage has no prefix
passage_text = "HNSW builds a multi-layer proximity graph..."

query_emb = model.encode(query_text)
passage_emb = model.encode(passage_text)
```

---

# **8. Common Interview Questions (With Strong Answers)**

---

## **Q1: "How does HNSW work? Walk me through the algorithm."**

**Strong answer:**

> "HNSW is a graph-based approximate nearest neighbor algorithm that builds a multi-layer navigable small-world graph.
>
> **Construction:** Each new vector is inserted into layer 0 (always) and probabilistically into higher layers (with exponentially decreasing probability — like a skip list). For each layer, the algorithm finds the M closest existing nodes and creates bidirectional edges to them. The efConstruction parameter controls how many candidates we evaluate during this process.
>
> **Search:** We start at the entry point on the top layer and greedily walk toward the query — always moving to the neighbor closest to q. When we can't improve on the current layer, we drop to the layer below and continue with more edges available. At layer 0, we do a beam search with width efSearch, maintaining a priority queue of candidates.
>
> **Why it works:** The top layers are sparse and enable long-range jumps (like an express highway), while layer 0 is dense for precise local navigation. This gives O(log N) average query time.
>
> **Key parameters:** M controls graph connectivity (higher = better recall, more RAM), efConstruction controls build-time search quality, efSearch controls query-time recall. The standard defaults are M=16, efConstruction=200, efSearch=64.
>
> In production, HNSW gives 95-99%+ recall with sub-millisecond queries on million-scale datasets. It's the default index in Weaviate, pgvector (v0.5+), and Qdrant."

---

## **Q2: "FAISS vs Pinecone vs pgvector — when would you use each?"**

**Strong answer:**

> "These serve different points on the control-vs-convenience spectrum:
>
> **FAISS** is my choice when I need maximum performance, GPU acceleration, or am working at billion-scale. It's a library, not a database, so I get full control over index types (IVF, HNSW, PQ, IVFPQ) but have to handle persistence, metadata filtering, and serving myself. I'd use it for a high-throughput recommendation system or when I need sub-millisecond latency.
>
> **Pinecone** is ideal when I want zero operational overhead. It's fully managed — I just push vectors and query. Great for startups or teams without dedicated infrastructure engineers. The trade-off is cost at scale and vendor lock-in. I'd use it for a RAG product where time-to-market matters more than per-query cost.
>
> **pgvector** is my default choice when the team already runs PostgreSQL. It lets me combine vector similarity with SQL joins, metadata filters, full-text search, and ACID transactions in one database. The sweet spot is under 5-10M vectors. I'd use it for an enterprise RAG system where data consistency matters and the vector corpus fits in a single Postgres instance.
>
> In my last project, I started with pgvector for the prototype (fast development, hybrid search with FTS), then benchmarked to see if we needed to migrate. The dataset was ~2M vectors, so pgvector handled it fine — no need to add infrastructure complexity."

---

## **Q3: "How would you design a vector search system for 10M documents?"**

**Strong answer:**

> "Here's my approach at 10M-document scale:
>
> **1. Embedding pipeline:** Batch-process documents through an embedding model (I'd use text-embedding-3-small for cost efficiency or BGE-large for OSS). Chunk documents to 512 tokens with 100-token overlap. L2-normalize all embeddings.
>
> **2. Index selection:** At 10M docs with 1536 dims, raw storage is ~57 GB. Two paths:
> - If RAM budget allows: **HNSW** (M=16, efConstruction=200) gives best recall. ~70 GB with graph overhead.
> - If memory-constrained: **IVFPQ** (nlist=4096, m=32) compresses to ~320 MB, with re-ranking from disk for top candidates.
>
> **3. Infrastructure:**
> - Option A: pgvector with HNSW index if team already runs Postgres and can provision a high-memory instance.
> - Option B: FAISS served behind FastAPI with metadata in a separate Postgres/Redis store if we need GPU acceleration or finer control.
> - Option C: Pinecone if minimal ops is the priority.
>
> **4. Hybrid retrieval:** Combine vector search (top 50) with BM25/FTS (top 50) using Reciprocal Rank Fusion. This catches exact keyword matches that embeddings miss.
>
> **5. Re-ranking:** Use a cross-encoder (e.g., ms-marco-MiniLM) to re-rank the top 20 candidates for higher precision.
>
> **6. Monitoring:** Track recall (vs periodic flat index ground truth), p99 latency, embedding drift, and query-result relevance."

---

## **Q4: "What is ANN vs exact nearest neighbor search? When would you choose exact?"**

**Strong answer:**

> "Exact nearest neighbor (k-NN) computes the distance between the query and every vector in the database — guaranteed to return the true k closest vectors. It's O(N × d) per query. In FAISS, this is `IndexFlatL2` or `IndexFlatIP`.
>
> ANN (Approximate Nearest Neighbor) uses data structures like HNSW graphs or IVF clusters to prune the search space, checking only a subset of vectors. It trades a small amount of recall (typically 1-5%) for massive speedup — O(log N) for HNSW vs O(N) for flat.
>
> **I'd choose exact search when:**
> - Dataset is small (< 50-100K vectors) — brute force is fast enough
> - 100% recall is legally or ethically required (medical retrieval, legal discovery)
> - Building ground truth for benchmarking ANN index recall
> - The query runs infrequently (batch processing, not real-time)
>
> **I'd choose ANN when:**
> - Dataset exceeds 100K vectors (ANN becomes necessary)
> - Real-time latency requirements (< 50ms)
> - Can tolerate 95-99% recall (most RAG, search, recommendation use cases)
>
> In practice, I always build a flat index first as a baseline, compute ground-truth k-NN for a query sample, then measure my ANN index's recall@k against it."

---

## **Q5: "How do you tune HNSW parameters for a production system?"**

**Strong answer:**

> "I follow a systematic approach:
>
> **Step 1: Establish baseline.** Build a flat index and compute ground-truth k-NN for ~1000 representative queries. This is my recall benchmark.
>
> **Step 2: Fix M (index build parameter).** Start with M=16 (industry default). Only increase to 32 if recall at maximum efSearch is still below target. M is the most expensive knob — doubling it nearly doubles RAM for the graph edges.
>
> **Step 3: Set efConstruction high.** Use efConstruction = 200 (minimum 2×M). This is a one-time build cost, so I err on the higher side. Going from 200 to 400 rarely helps unless M > 32.
>
> **Step 4: Tune efSearch (the main runtime knob).** efSearch must be ≥ k. I sweep efSearch = [32, 64, 128, 256, 512], measuring recall@10 and p99 latency for each. I plot the recall-latency curve and pick the operating point that meets my SLA.
>
> **Typical results (1M vectors, d=768):**
>
> | efSearch | Recall@10 | p99 Latency |
> |---|---|---|
> | 32 | 92% | 0.5ms |
> | 64 | 96% | 0.8ms |
> | 128 | 98.5% | 1.5ms |
> | 256 | 99.3% | 3ms |
>
> For RAG, I usually target 95%+ recall@10, so efSearch=64-128 is my sweet spot.
>
> **Step 5: Monitor in production.** Periodically sample queries, compute exact k-NN, and track recall drift — especially after large data ingestion that may degrade graph quality."

---

## **Q6: "How does Product Quantization compress vectors? What's the trade-off?"**

**Strong answer:**

> "PQ divides each d-dimensional vector into m equal sub-vectors and independently quantizes each sub-vector to its nearest centroid in a learned codebook. Each sub-vector is replaced by an 8-bit centroid ID, compressing a d=768 float32 vector from 3,072 bytes to just m bytes (e.g., 32 bytes for m=32).
>
> **How it works:**
> 1. Split vector into m sub-vectors of d/m dimensions
> 2. For each sub-space, run k-means to learn 256 centroids (k* = 2^8)
> 3. Replace each sub-vector with its nearest centroid ID (1 byte)
>
> **At query time (Asymmetric Distance Computation):**
> - Compute exact distances between query sub-vectors and all centroids (precompute distance table)
> - For each database vector, look up sub-distances from the table and sum → approximate total distance
>
> **Trade-off:** PQ gives 50-400× compression but sacrifices recall (typically 80-95%). The key insight is combining PQ with IVF (IVFPQ): IVF prunes the search space first, then PQ enables scanning millions of compressed codes in memory. For recovering precision, I use a two-stage approach: retrieve top-100 with IVFPQ, then re-rank with exact distances from full vectors stored on disk."

---

## **Q7: "How do you evaluate retrieval quality in a RAG system?"**

**Strong answer:**

> "I evaluate at two levels — retrieval quality and end-to-end generation quality:
>
> **Retrieval metrics:**
> - **Recall@k:** Fraction of ground-truth relevant docs in top-k results (most important)
> - **MRR (Mean Reciprocal Rank):** 1/rank of first relevant result
> - **nDCG@k:** Accounts for graded relevance and ranking position
> - **Precision@k:** Fraction of top-k that are actually relevant
>
> **Practical measurement:**
> 1. Create a test set of ~100-500 queries with labeled relevant passages
> 2. Compare vector search results against labels
> 3. Benchmark ANN recall against flat index (how much does the index approximate?)
>
> **End-to-end RAG metrics:**
> - **Answer correctness:** Does the LLM produce the right answer given retrieved context?
> - **Faithfulness:** Does the answer stay grounded in the retrieved passages?
> - **Context relevance:** Are the retrieved passages actually useful for answering?
>
> **Tools I use:** RAGAS framework for automated evaluation, custom recall@k scripts against flat index ground truth, and periodic human evaluation samples.
>
> The most common failure mode: the embedding model doesn't capture domain-specific semantics well. I've found that hybrid search (vector + BM25) consistently outperforms pure vector by 5-15% recall on technical corpora."

---

## **Q8: "Explain the difference between pre-filtering and post-filtering in vector search."**

**Strong answer:**

> "This comes up when you need to combine metadata filters with vector similarity — for example, 'find similar documents but only from 2024 in the ML category.'
>
> **Post-filtering:** Run vector search first (get top-k from the full index), then filter results by metadata. Simple but problematic — if most of your top-100 don't match the filter, you get few or zero results.
>
> **Pre-filtering:** Filter the candidate set by metadata first, then run vector search only on matching vectors. Guarantees you get k results matching the filter, but can be slow if the filter is very selective (small candidate set can't use the ANN index efficiently).
>
> **Integrated filtering (best):** The index natively understands both vectors and metadata. Pinecone, Weaviate, and pgvector support this — the query planner considers both the metadata predicate and vector proximity together.
>
> ```sql
> -- pgvector: integrated filtering (Postgres query planner optimizes)
> SELECT id, title, 1 - (embedding <=> $1::vector) AS similarity
> FROM documents
> WHERE category = 'ML' AND year >= 2024   -- metadata filter
> ORDER BY embedding <=> $1::vector         -- vector similarity
> LIMIT 10;
> ```
>
> In FAISS, I implement this manually with an `IDSelector` or by maintaining separate sub-indexes per category. In pgvector, the Postgres query planner handles it automatically using partial indexes."

---

## **Q9: "What happens when your embedding model changes? How do you handle re-indexing?"**

**Strong answer:**

> "Model changes require **complete re-embedding** because vectors from different models live in incompatible spaces — you can't mix them. This is one of the most operationally expensive events in a vector search system.
>
> **My strategy:**
>
> 1. **Blue-green indexing:** Build the new index alongside the old one. Store raw text alongside vectors so you can re-embed without going back to source systems.
>
> 2. **Shadow testing:** Route a percentage of queries to both old and new indexes, compare result quality (A/B test on retrieval metrics).
>
> 3. **Atomic swap:** Once validated, swap the index pointer (e.g., update an alias in Pinecone, swap the FAISS index file, or use a Postgres view).
>
> 4. **Version metadata:** Tag each vector with its embedding model version so you can identify and re-process stale embeddings incrementally.
>
> **Prevention:** I choose embedding models carefully and prefer stable, versioned APIs (OpenAI's text-embedding-3-small with fixed dimensions) over rapidly-changing open-source models for production. When I must use OSS models, I pin the exact checkpoint."

---

# **9. Key Takeaways**

---

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                 VECTOR DATABASE CHEAT SHEET                      │
 │                                                                  │
 │  1. L2-normalize embeddings → use inner product = cosine        │
 │                                                                  │
 │  2. Index selection:                                             │
 │     • <100K vectors → Flat (brute force, 100% recall)           │
 │     • 100K-10M     → HNSW (best recall-latency)                │
 │     • 10M-1B+      → IVFPQ (compressed, scalable)              │
 │                                                                  │
 │  3. Database selection:                                          │
 │     • Max control/GPU → FAISS                                   │
 │     • Zero ops        → Pinecone                                │
 │     • Hybrid search   → Weaviate or pgvector                   │
 │     • Already on PG   → pgvector (one DB for everything)       │
 │     • Prototyping     → Chroma                                  │
 │                                                                  │
 │  4. Always use hybrid search in production RAG                  │
 │     (vector + BM25/FTS, fuse with RRF)                          │
 │                                                                  │
 │  5. HNSW tuning: M=16, efConstruction=200, efSearch=64-128      │
 │     efSearch is the primary runtime knob                        │
 │                                                                  │
 │  6. Evaluate retrieval: recall@k against flat index ground truth│
 │                                                                  │
 │  7. Two-stage retrieval: ANN top-100 → re-rank with exact       │
 │     distances or cross-encoder → return top-10                  │
 │                                                                  │
 │  8. pgvector operators: <-> (L2), <=> (cosine), <#> (neg IP)   │
 │                                                                  │
 │  9. Embedding models: normalize, same model for index & query,  │
 │     chunk 256-512 tokens, benchmark on YOUR data                │
 │                                                                  │
 │  10. The trade-off triangle: Recall ↔ Latency ↔ Memory          │
 │      You can optimize two at the expense of the third           │
 └──────────────────────────────────────────────────────────────────┘
```

**One-liner summaries for rapid recall:**

| Topic | One-Liner |
|---|---|
| **Vector DB** | Stores embeddings and finds nearest neighbors by semantic similarity |
| **Cosine similarity** | Measures angle between vectors; range [-1, 1]; direction-only |
| **L2 distance** | Euclidean straight-line distance; magnitude-sensitive |
| **Dot product** | Equals cosine when vectors are L2-normalized |
| **Flat index** | Brute-force O(N); 100% recall; baseline and small datasets |
| **IVF** | Cluster-based ANN; nlist clusters, nprobe searched; fast to build |
| **HNSW** | Multi-layer graph ANN; best general-purpose; O(log N) search |
| **PQ** | Sub-vector quantization; 50-400× compression; lower recall |
| **IVFPQ** | IVF + PQ; billion-scale with limited RAM |
| **FAISS** | Meta's library; max performance/control; manual everything else |
| **Pinecone** | Managed SaaS; zero ops; metadata filters; costs at scale |
| **Weaviate** | OSS; HNSW + BM25 hybrid; GraphQL; multi-tenant |
| **pgvector** | PG extension; SQL + vectors + FTS; ACID; <10M sweet spot |
| **Hybrid search** | Vector + BM25/FTS; RRF fusion; always better than vector-only |
| **Recall@k** | Primary retrieval metric; benchmark ANN against flat ground truth |

---

*Last updated: February 2026*
