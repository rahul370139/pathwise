# Recommendation Systems — Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — Personalization, Retrieval, Ranking, Multimodal Recommendations

> **Context:** You built a Multimodal Fashion Recommendation System (CLIP + FAISS + BLIP, 88% MOS). This guide covers the **full RecSys stack** that interviewers expect — from classical methods to modern two-tower and LLM-powered approaches.

---

# Table of Contents

1. [RecSys Landscape](#1-recsys-landscape)
2. [Collaborative Filtering](#2-collaborative-filtering)
3. [Content-Based Filtering](#3-content-based-filtering)
4. [Matrix Factorization](#4-matrix-factorization)
5. [Deep Learning for Recommendations](#5-deep-learning-for-recommendations)
6. [Two-Tower Architecture (Retrieval)](#6-two-tower-architecture-retrieval)
7. [Learning-to-Rank (Ranking)](#7-learning-to-rank-ranking)
8. [Retrieval + Ranking Pipeline](#8-retrieval--ranking-pipeline)
9. [Cold Start Problem](#9-cold-start-problem)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [LLMs in Recommendations (2025+)](#11-llms-in-recommendations-2025)
12. [Interview Questions with Strong Answers](#12-interview-questions-with-strong-answers)

---

# **1. RecSys Landscape**

```
┌──────────────────────────────────────────────────────────────────┐
│              RECOMMENDATION SYSTEM TAXONOMY                       │
│                                                                   │
│  COLLABORATIVE FILTERING                                         │
│  └── "Users who liked X also liked Y"                           │
│      ├── User-based CF                                          │
│      ├── Item-based CF                                          │
│      └── Matrix Factorization (SVD, ALS, NMF)                  │
│                                                                   │
│  CONTENT-BASED FILTERING                                         │
│  └── "This item has features similar to items you liked"        │
│      ├── TF-IDF + cosine similarity                             │
│      ├── Embedding similarity (CLIP, BERT)                      │
│      └── Feature-based classifiers                              │
│                                                                   │
│  HYBRID                                                          │
│  └── Combine collaborative + content signals                    │
│      ├── Weighted combination                                   │
│      ├── Feature stacking                                       │
│      └── Cascading (content for cold start → CF when enough data│
│                                                                   │
│  DEEP LEARNING                                                   │
│  └── Neural models for complex interactions                     │
│      ├── Neural CF (NCF)                                        │
│      ├── Two-Tower (retrieval)                                  │
│      ├── Wide & Deep                                            │
│      ├── DIN/DIEN (attention over user history)                 │
│      └── Transformers (SASRec, BERT4Rec)                        │
│                                                                   │
│  KNOWLEDGE-BASED                                                 │
│  └── "Given your constraints, here are options"                 │
│      ├── Rule-based                                             │
│      └── Constraint satisfaction                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

# **2. Collaborative Filtering**

## **2.1 User-Based CF**

Find users **similar to you**, recommend what they liked.

```
User-Item Matrix:
              Item1  Item2  Item3  Item4  Item5
User A (you):   5      3      ?      1      ?
User B:         5      ?      4      1      2
User C:         ?      3      5      ?      4
User D:         5      4      4      1      ?

User A is similar to User B (both rated Item1=5, Item4=1)
User B rated Item3=4 → Recommend Item3 to User A
```

**Similarity measures:**
| Measure | Formula Intuition | Best For |
|---------|------------------|----------|
| **Cosine similarity** | Angle between rating vectors | When rating scales differ across users |
| **Pearson correlation** | Correlation of centered ratings | When users have different rating baselines |
| **Jaccard** | Overlap of rated items | Implicit feedback (binary) |

## **2.2 Item-Based CF**

Find items **similar to items you liked**, recommend them.

More stable than user-based CF because item relationships change less often than user preferences.

**Amazon's approach:** "Customers who bought X also bought Y" is item-based CF at scale.

## **2.3 CF Limitations**

| Problem | Description | Solution |
|---------|-------------|----------|
| **Cold Start (user)** | New user has no ratings | Content-based fallback, onboarding survey |
| **Cold Start (item)** | New item has no ratings | Content features, editorial promotion |
| **Sparsity** | 99%+ of user-item pairs are unobserved | Matrix factorization, dimensionality reduction |
| **Scalability** | O(users × items) for similarity computation | Approximate methods, ANN |
| **Popularity bias** | Popular items dominate | Diversity-aware ranking, explore/exploit |

---

# **3. Content-Based Filtering**

Recommend items with **features similar** to items the user has liked.

| Feature Type | Example | Encoding |
|-------------|---------|----------|
| Text (title, description) | "Blue cotton dress, summer collection" | TF-IDF, BERT embeddings |
| Image | Product photo | CNN features, CLIP visual embeddings |
| Categorical | Brand, category, color | One-hot, entity embeddings |
| Numerical | Price, rating, recency | Normalized, bucketized |

**Your Fashion RecSys project:** Used CLIP embeddings (both visual and text) + BLIP captions + segmentation masks — this is content-based at the representation level, combined with FAISS retrieval for scalability.

**Advantage:** Works for new items (no cold start on items).
**Limitation:** Doesn't leverage collective user behavior. Only recommends similar items (no serendipity).

---

# **4. Matrix Factorization**

## **4.1 Core Idea**

Decompose the sparse user-item matrix R into two low-rank matrices: R ≈ U × V^T

- U = user matrix (users × k latent factors)
- V = item matrix (items × k latent factors)
- Each user and item is represented as a k-dimensional vector
- Predicted rating: r̂(u,i) = u · vᵢ (dot product)

```
Users × Items (sparse)     Users × k      k × Items
┌─────────────────┐      ┌─────────┐    ┌──────────┐
│ 5  ?  ?  1  ?   │      │ u₁ u₂   │    │ v₁ᵀ     │
│ ?  4  ?  ?  2   │  ≈   │ u₁ u₂   │  × │ v₂ᵀ     │
│ ?  ?  3  ?  5   │      │ u₁ u₂   │    │ v₃ᵀ     │
│ 4  ?  ?  2  ?   │      │ u₁ u₂   │    │ v₄ᵀ     │
└─────────────────┘      └─────────┘    └──────────┘
```

## **4.2 Algorithms**

| Algorithm | Optimization | Notes |
|-----------|-------------|-------|
| **SVD** | Singular Value Decomposition | Classical, requires imputing missing values |
| **Funk SVD** | SGD on observed ratings only | Netflix Prize approach, scalable |
| **ALS** | Alternating Least Squares | Parallelizable, used in Spark MLlib |
| **NMF** | Non-Negative Matrix Factorization | Interpretable factors (all non-negative) |
| **BPR** | Bayesian Personalized Ranking | Optimizes ranking (pairwise loss), not rating prediction |

## **4.3 Implicit vs Explicit Feedback**

| Type | Data | Signal | Challenges |
|------|------|--------|------------|
| **Explicit** | Ratings (1-5 stars), reviews | Clear preference | Sparse (most users don't rate) |
| **Implicit** | Clicks, purchases, watch time, scrolls | Abundant, noisy | No negative signal (not clicking ≠ dislike), confidence varies |

**Implicit feedback handling:**
- **Weighted Matrix Factorization (Hu et al.):** Confidence increases with more interactions. c(u,i) = 1 + α × interactions.
- **BPR (Bayesian Personalized Ranking):** Pairwise loss — user prefers interacted item over random non-interacted item.

---

# **5. Deep Learning for Recommendations**

## **5.1 Neural Collaborative Filtering (NCF)**

Replace dot product with a neural network: instead of u · v, pass [u; v] through MLP to capture nonlinear user-item interactions.

## **5.2 Wide & Deep (Google)**

Combine a **wide** component (memorization via cross-product features) with a **deep** component (generalization via embeddings + MLP).

```
Wide: "Users who bought diapers AND beer → higher likelihood"  (memorized combo)
Deep: "User embedding × item embedding → MLP → generalized preference"
```

## **5.3 Sequential Recommendations**

Model the user's interaction **sequence** to capture evolving preferences:

| Model | Architecture | Key Idea |
|-------|-------------|----------|
| **GRU4Rec** | GRU on session history | Session-based, captures short-term intent |
| **SASRec** | Transformer (self-attention) | Attention over interaction history |
| **BERT4Rec** | Bidirectional (masked item prediction) | Pre-train with masked items, like BERT for NLP |
| **DIN** (Alibaba) | Attention over user history w.r.t. candidate item | Soft-search: items in history most relevant to candidate get higher weight |

## **5.4 Graph Neural Networks for RecSys**

Model user-item interactions as a bipartite graph:

| Model | Approach |
|-------|---------|
| **PinSage** (Pinterest) | GraphSAGE on pin-board graph for visual recommendations |
| **LightGCN** | Simplified GCN: only neighborhood aggregation, no feature transformation |
| **NGCF** | Neural graph collaborative filtering |

---

# **6. Two-Tower Architecture (Retrieval)**

## **6.1 Core Idea**

Separate **user encoder** and **item encoder** produce embeddings that are compared via dot product or cosine similarity. The key advantage: **item embeddings can be pre-computed** and indexed for fast retrieval.

```
┌──────────────────────────────────────────────────────────────────┐
│                TWO-TOWER ARCHITECTURE                              │
│                                                                   │
│  ┌──────────────┐              ┌──────────────┐                  │
│  │  USER TOWER   │              │  ITEM TOWER   │                 │
│  │              │              │              │                  │
│  │  User ID     │              │  Item ID     │                  │
│  │  Demographics│              │  Category    │                  │
│  │  Past history│              │  Title embed │                  │
│  │  Context     │              │  Image embed │                  │
│  │      │       │              │      │       │                  │
│  │      ▼       │              │      ▼       │                  │
│  │   [MLP]      │              │   [MLP]      │                  │
│  │      │       │              │      │       │                  │
│  │      ▼       │              │      ▼       │                  │
│  │  User Embed  │              │  Item Embed  │                  │
│  │  (128-dim)   │              │  (128-dim)   │                  │
│  └──────┬───────┘              └──────┬───────┘                  │
│         │                             │                          │
│         └──────── dot product ────────┘                          │
│                       │                                          │
│                   score(u, i)                                    │
│                                                                   │
│  Training: contrastive loss (positive pairs from interactions,   │
│           in-batch negatives or hard negatives)                  │
│                                                                   │
│  Serving: pre-compute all item embeddings → FAISS/ScaNN index   │
│           At query time: encode user → ANN search → top-K items │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **6.2 Training: Loss Functions**

| Loss | How It Works |
|------|-------------|
| **Binary Cross-Entropy** | Positive (interacted) vs negative (not interacted) pairs |
| **Contrastive / InfoNCE** | Push positive pairs close, push negatives apart |
| **In-batch Negatives** | All other items in the batch serve as negatives (efficient) |
| **Hard Negative Mining** | Sample negatives that are close to the user (harder to distinguish) |

## **6.3 Negative Sampling Strategies**

| Strategy | How | Trade-off |
|----------|-----|-----------|
| **Random** | Uniformly sample from all items | Easy, but most negatives are too easy |
| **Popularity-weighted** | Sample popular items more often | Harder negatives |
| **In-batch** | Other positives in batch are negatives | Free, GPU-efficient, can have false negatives |
| **Hard negatives** | Items close in embedding space but not interacted | Best learning signal, expensive to mine |

---

# **7. Learning-to-Rank (Ranking)**

After retrieval gives ~100-1000 candidates, a **ranker** orders them by predicted relevance.

## **7.1 Ranking Approaches**

| Approach | Loss | How |
|----------|------|-----|
| **Pointwise** | Regression/classification on individual items | Predict P(click\|user, item). Items sorted by predicted score |
| **Pairwise** | BPR, RankNet — compare pairs | Loss on whether model correctly orders item A > item B |
| **Listwise** | ListNet, LambdaMART — optimize list-level metric | Directly optimize nDCG or MAP on the full ranked list |

## **7.2 LambdaMART**

The industry workhorse for ranking. Gradient-boosted trees optimized for ranking metrics (nDCG). Used at Microsoft (Bing), Yahoo, Airbnb.

| Feature Category | Examples |
|-----------------|---------|
| **User features** | Tenure, past purchase rate, avg basket size |
| **Item features** | Price, category, popularity, recency |
| **User-Item features** | Past clicks on this category, price affinity |
| **Context features** | Time of day, device, location |
| **Retrieval score** | Score from the two-tower retrieval stage |

---

# **8. Retrieval + Ranking Pipeline**

## **8.1 The Full Production Stack**

```
┌──────────────────────────────────────────────────────────────────┐
│          PRODUCTION RECOMMENDATION PIPELINE                       │
│                                                                   │
│  Candidate Pool: ~10M items                                      │
│       │                                                          │
│       ▼                                                          │
│  RETRIEVAL (multiple sources, fast, ~10ms)                       │
│  ├── Two-tower ANN: ~500 candidates                             │
│  ├── Collaborative filter: ~200 candidates                      │
│  ├── Content-based: ~200 candidates                             │
│  ├── Popularity: ~100 candidates                                │
│  └── Business rules: ~50 candidates (promoted, new items)       │
│       │                                                          │
│       ▼ ~1000 candidates (deduplicated, merged)                 │
│                                                                   │
│  PRE-RANKING (lightweight model, ~5ms)                           │
│  └── Fast scoring to reduce ~1000 → ~200                        │
│       │                                                          │
│       ▼ ~200 candidates                                          │
│                                                                   │
│  RANKING (heavy model, ~20ms)                                    │
│  └── Full feature model (user×item×context features)            │
│      LambdaMART or deep ranking model                           │
│       │                                                          │
│       ▼ ~50 ranked items                                         │
│                                                                   │
│  RE-RANKING (business logic, ~2ms)                               │
│  ├── Diversity injection (MMR, DPP)                             │
│  ├── Freshness boost                                            │
│  ├── Remove already seen / purchased                            │
│  ├── Ad slot insertion                                          │
│  └── Fairness constraints                                       │
│       │                                                          │
│       ▼ ~20 items displayed                                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

# **9. Cold Start Problem**

| Scenario | Challenge | Solutions |
|----------|----------|----------|
| **New user** | No interaction history | Content-based (use demographics), popular items, onboarding quiz, bandits (explore/exploit) |
| **New item** | No user interactions | Content features (image, text embeddings), item-item similarity from content, editorial boost, metadata-based CF |
| **New platform** | No data at all | Rule-based, editorial curation, transfer learning from similar domain |

**Key interview insight:** The cold start problem is why **hybrid systems** dominate production. Use content-based for cold items/users, collaborative filtering for warm entities, and blend them with a learned weighting.

---

# **10. Evaluation Metrics**

## **10.1 Offline Metrics**

| Metric | What It Measures | Formula Intuition |
|--------|-----------------|------------------|
| **Precision@K** | Fraction of top-K that are relevant | Relevant in top-K / K |
| **Recall@K** | Fraction of relevant items captured in top-K | Relevant in top-K / Total relevant |
| **nDCG@K** | Ranked quality (higher-ranked relevant items score more) | Discounted by position (log₂) |
| **MRR** | Position of first relevant item | 1 / rank of first relevant |
| **Hit Rate@K** | Was there ANY relevant item in top-K? | Binary per user, averaged |
| **MAP@K** | Average precision across all relevant items | Average of P@k for each relevant k |
| **AUC** | Probability that a positive is ranked above a random negative | Standard AUC computation |

## **10.2 Online Metrics (A/B Test)**

| Metric | What It Measures | Why |
|--------|-----------------|-----|
| **CTR (Click-Through Rate)** | Clicks / Impressions | Engagement signal |
| **Conversion Rate** | Purchases / Clicks | Revenue proxy |
| **Revenue per Session** | Total revenue / Sessions | Ultimate business metric |
| **Diversity** | Unique categories/brands in recs | Serendipity, avoids filter bubble |
| **Coverage** | Fraction of catalog recommended | Avoiding popularity bias |
| **Dwell Time** | Time spent on recommended content | Quality of engagement |

## **10.3 Offline vs Online Gap**

A model that improves offline metrics (nDCG) may **not** improve online metrics (revenue). This happens because:
- Offline evals don't capture exploration value
- Position bias in training data (items shown in position 1 get more clicks regardless)
- Offline metrics don't capture long-term user satisfaction (binging vs diverse discovery)

---

# **11. LLMs in Recommendations (2025+)**

## **11.1 How LLMs Are Changing RecSys**

| Application | How |
|-------------|-----|
| **Feature extraction** | Generate item descriptions, extract attributes from reviews |
| **User understanding** | Summarize user preference from interaction history in natural language |
| **Conversational recs** | "Show me summer dresses under $50 similar to what I bought last month" |
| **Explanation generation** | "We recommend this because you enjoyed similar sci-fi novels with strong female leads" |
| **Cross-domain transfer** | LLM encodes items from any domain in shared semantic space |
| **Cold start** | Rich item representation from description alone (no interactions needed) |

## **11.2 LLM-Based Recommendation Architectures**

| Approach | How It Works |
|----------|-------------|
| **LLM as ranker** | Format candidates as text, ask LLM to rank them |
| **LLM as feature extractor** | Use LLM embeddings as item/user features in traditional pipeline |
| **LLM as conversational agent** | User describes preferences in natural language, LLM retrieves + explains |
| **LLM for synthetic data** | Generate training interactions for cold items |

---

# **12. Interview Questions with Strong Answers**

---

## **Q1: "Design a recommendation system for an e-commerce platform."**

> I'd build a **multi-stage pipeline: retrieval → ranking → re-ranking**.
>
> **Retrieval:** Multiple candidate generators running in parallel — (1) two-tower model trained on user-item interactions for personalized candidates, (2) item-based CF for "similar to recently viewed," (3) popularity baseline for new users, (4) content-based using product embeddings for cold items. Merge and deduplicate to ~1000 candidates.
>
> **Ranking:** A LambdaMART or deep ranking model scoring each candidate with rich features — user purchase history features, item attributes, user-item interaction features (has the user browsed this category?), and contextual features (time, device). Optimize for nDCG against purchase labels.
>
> **Re-ranking:** Inject diversity (no more than 3 items from the same brand), apply business rules (boosted items, sponsored placements), remove recently purchased items.
>
> **Evaluation:** A/B test with revenue per session as the primary metric, CTR and diversity as guardrail metrics.

---

## **Q2: "How do you handle the cold start problem?"**

> Three strategies depending on what's cold:
>
> **New user:** Start with popular and trending items (non-personalized). Use onboarding signals — first search query, first category browsed — to rapidly personalize using a bandit approach (explore popular items across categories, exploit categories where user shows interest). As interactions accumulate, transition to collaborative filtering.
>
> **New item:** Use content features (product images → CLIP embedding, title → BERT embedding) to place the item in the existing embedding space. Find its nearest neighbors among established items and recommend it to users who liked those neighbors. Also give it an exploration bonus in the re-ranking stage so it gets impressions.
>
> **In my Fashion RecSys project:** I used CLIP embeddings to represent items multimodally (image + text), which naturally handles cold items — a new product with an image and description immediately has a meaningful embedding without any user interaction data.

---

## **Q3: "Explain the two-tower architecture. Why is it popular for retrieval?"**

> Two-tower separates user and item encoding into independent towers. Each produces an embedding, and the score is their dot product or cosine similarity.
>
> **Why it's popular:** Item embeddings are **computed offline and indexed** in an ANN system (FAISS, ScaNN). At serving time, you only need to encode the user (one forward pass) and do an ANN search — this is O(log N) instead of scoring all N items. For a catalog of 10M items, this is the difference between 10ms and 10 minutes.
>
> **Trade-off:** The dot-product interaction is **less expressive** than a full cross-network (can't capture complex user-item feature interactions). That's why it's used for retrieval (fast, approximate), and a more powerful model handles ranking (slow, precise) on the smaller candidate set.

---

## **Q4: "What's the difference between implicit and explicit feedback? How do you handle implicit?"**

> **Explicit:** User directly expresses preference (5-star rating, thumbs up). Clear signal but sparse — most users don't rate.
>
> **Implicit:** User behavior signals preference (clicks, purchases, time spent, scrolls). Abundant but noisy — a click doesn't mean love, and NOT clicking doesn't mean dislike (maybe the user never saw the item).
>
> For implicit feedback, I use **BPR (Bayesian Personalized Ranking):** assume the user prefers items they interacted with over items they didn't, and optimize pairwise ranking loss. Or **weighted matrix factorization** where confidence in a preference scales with interaction strength (clicked once = low confidence, purchased = high confidence, clicked 10 times = very high confidence).
>
> The key challenge with implicit data: **no negative examples.** You can't distinguish "dislike" from "never saw." Hard negative mining — sampling items similar to what the user liked but didn't interact with — is crucial for training effective models.

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│          RECOMMENDATION SYSTEMS TAKEAWAYS                         │
│                                                                   │
│  1. Production RecSys = RETRIEVAL (fast, 1000s) +               │
│     RANKING (precise, 100s) + RE-RANKING (business logic)       │
│                                                                   │
│  2. Two-tower is the standard RETRIEVAL architecture —          │
│     pre-compute item embeddings, ANN search at query time       │
│                                                                   │
│  3. COLLABORATIVE FILTERING for warm users/items,               │
│     CONTENT-BASED for cold start → always HYBRID                │
│                                                                   │
│  4. IMPLICIT FEEDBACK dominates real world — handle with        │
│     BPR, weighted MF, or contrastive learning                   │
│                                                                   │
│  5. COLD START is the #1 practical challenge — solve with       │
│     content features, bandits, onboarding signals               │
│                                                                   │
│  6. nDCG for offline eval, REVENUE for online A/B tests        │
│     (offline gains don't always translate to online)            │
│                                                                   │
│  7. LLMs are entering RecSys: conversational recs, cold-start   │
│     embeddings, explanation generation                           │
│                                                                   │
│  8. Always inject DIVERSITY in re-ranking — filter bubbles     │
│     hurt long-term user satisfaction                             │
└──────────────────────────────────────────────────────────────────┘
```
