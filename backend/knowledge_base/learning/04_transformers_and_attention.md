# Transformers & Attention Mechanisms — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, Transformers, NLP Systems

---

# Table of Contents

1. [The Attention Mechanism](#1-the-attention-mechanism)
2. [Multi-Head Attention](#2-multi-head-attention)
3. [Self-Attention vs Cross-Attention](#3-self-attention-vs-cross-attention)
4. [Positional Encoding](#4-positional-encoding)
5. [Transformer Architecture](#5-transformer-architecture)
6. [Feed-Forward Network (FFN)](#6-feed-forward-network-ffn)
7. [Key Architectural Details](#7-key-architectural-details)
8. [Computational Complexity](#8-computational-complexity)
9. [Scaling Laws](#9-scaling-laws)
10. [Key Model Variants](#10-key-model-variants)
11. [Common Interview Questions](#11-common-interview-questions-with-strong-answers)
12. [Key Takeaways for LLM Interviews](#12-key-takeaways-for-llm-interviews)

---

# **1. The Attention Mechanism**

---

## **1.1 Intuition: Why Attention Was Invented**

Before transformers, **seq2seq models** (Sutskever et al., 2014) used an encoder RNN to compress an entire input sequence into a single fixed-length context vector, then a decoder RNN to generate output from that vector.

```
 ┌─────────────────────────────────────────────────────┐
 │              Seq2Seq WITHOUT Attention               │
 │                                                      │
 │  x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → x₄ → [h₄] │
 │                                              │       │
 │                                         context (c)  │
 │                                              │       │
 │                              [s₁] → [s₂] → [s₃]    │
 │                               y₁     y₂     y₃     │
 └─────────────────────────────────────────────────────┘
```

**The bottleneck problem:**
- The entire source sentence (possibly 50+ tokens) is crammed into one vector `c`.
- Long-range dependencies are lost — the model "forgets" earlier tokens.
- Performance degrades sharply as input length increases.

**Bahdanau Attention (2015)** solved this by letting the decoder **look back at all encoder hidden states** at each step, computing a weighted combination:

```
context_t = Σ α_{t,i} · h_i       (weighted sum over all encoder states)
α_{t,i}   = softmax(score(s_t, h_i))   (alignment scores)
```

This was the birth of the attention mechanism — the decoder **attends** to the most relevant parts of the input at each generation step.

**Key insight:** Instead of compressing everything into one vector, let the model dynamically decide what to focus on.

---

## **1.2 Scaled Dot-Product Attention**

The Transformer (Vaswani et al., 2017) generalized attention into a purely algebraic operation on three matrices:

```
                    Q · Kᵀ
Attention(Q,K,V) = softmax(─────────) · V
                    √(d_k)
```

Where:
- **Q** (Query): `[n × d_k]` — what we're looking for
- **K** (Key): `[m × d_k]` — what's available to match against
- **V** (Value): `[m × d_v]` — the actual content to retrieve

### Step-by-Step Computation

```
 Step 1: Compute raw attention scores
         S = Q · Kᵀ                    → [n × m] matrix

 Step 2: Scale by √d_k
         S_scaled = S / √d_k

 Step 3: (Optional) Apply mask
         S_masked = S_scaled + mask    (−∞ for positions to ignore)

 Step 4: Apply softmax row-wise
         A = softmax(S_masked)         → [n × m] (rows sum to 1)

 Step 5: Weighted sum of values
         Output = A · V                → [n × d_v]
```

```
 ┌───────────────────────────────────────────┐
 │         Scaled Dot-Product Attention       │
 │                                            │
 │    Q ──┐                                   │
 │        ├──→ MatMul ──→ Scale ──→ (Mask)    │
 │    K ──┘               ÷√d_k      │       │
 │                                 Softmax    │
 │                                    │       │
 │    V ──────────────────────→ MatMul        │
 │                                    │       │
 │                                 Output     │
 └───────────────────────────────────────────┘
```

---

## **1.3 Why Scale by √d_k?**

This is a **very common interview question**. Here's the precise reasoning:

When `d_k` is large, the dot products `q · k` grow in magnitude. If `q` and `k` are random vectors with entries ~ N(0, 1), then:

```
E[q · k] = 0
Var[q · k] = d_k
```

So the **standard deviation** of the dot product is `√d_k`. As `d_k` grows:

| d_k | Std Dev of q·k | Typical dot product range |
|-----|----------------|--------------------------|
| 64  | 8.0            | [-24, +24]               |
| 128 | 11.3           | [-34, +34]               |
| 512 | 22.6           | [-68, +68]               |

**Problem:** When dot products are large, softmax **saturates** — it pushes almost all probability mass onto the largest value. In this regime:

- Gradients become **extremely small** (vanishing gradient problem).
- The model can't learn nuanced attention distributions.
- It behaves like hard argmax instead of soft weighting.

**Solution:** Dividing by `√d_k` normalizes the variance back to 1, keeping dot products in a range where softmax produces **meaningful, non-degenerate gradients**.

```
 Without scaling (d_k=512):  softmax([23, 25, 2]) ≈ [0.12, 0.88, 0.00]  ← near one-hot
 With scaling (÷√512≈22.6):  softmax([1.0, 1.1, 0.09]) ≈ [0.35, 0.39, 0.26] ← informative
```

> **Interview tip:** If asked "why not divide by d_k?" — because that would make the variance 1/d_k (too small), causing softmax to output near-uniform distributions. √d_k is the Goldilocks scaling.

---

## **1.4 Q, K, V Analogy**

The best analogy is a **search engine / database lookup**:

| Component | Analogy | Role |
|-----------|---------|------|
| **Query (Q)** | Your Google search query | "What am I looking for?" |
| **Key (K)** | The title/index of each webpage | "How relevant is each item?" |
| **Value (V)** | The actual content of each webpage | "What information do I retrieve?" |

**How it works:**
1. You type a **query** ("best pizza NYC").
2. The search engine compares your query against **keys** (page titles, metadata).
3. Pages with high query-key match get high attention scores.
4. You read the **values** (actual page content) of the top results.
5. The final output is a **weighted blend** of values, weighted by relevance.

**In a transformer:**
- Each token generates its own Q, K, V vectors via learned linear projections.
- Q says: "I'm looking for tokens related to [this concept]."
- K says: "I contain information about [this concept]."
- V says: "Here's my actual content if you need me."

```
 Token:  "The cat sat on the mat"
 
 "sat" (Query) asks: "Who did the sitting? Where was it?"
   → "cat" (Key) responds:  high relevance score (subject)
   → "mat" (Key) responds:  high relevance score (location)
   → "The" (Key) responds:  low relevance score (article)
```

**Technical detail:** Q, K, V are produced by **separate learned projections**:
```
Q = X · W_Q    (W_Q: [d_model × d_k])
K = X · W_K    (W_K: [d_model × d_k])
V = X · W_V    (W_V: [d_model × d_v])
```

This means the model learns **different representations** for "what to search for" vs "what to be found by" vs "what to contribute" — a crucial design choice.

---

# **2. Multi-Head Attention**

---

## **2.1 Why Multiple Heads?**

A single attention head can only capture **one type of relationship** at a time. Language has many simultaneous relationships:

- **Syntactic:** subject-verb agreement ("The dogs **are** running")
- **Semantic:** meaning similarity ("king" attends to "queen")
- **Positional:** nearby words ("New" attends to "York")
- **Coreference:** pronoun resolution ("She" attends to "Alice")

Multi-head attention runs **h parallel attention computations**, each with its own learned projections, allowing the model to capture multiple relationship types simultaneously.

---

## **2.2 Formula**

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) · W_O

where headᵢ = Attention(Q · W_Qⁱ, K · W_Kⁱ, V · W_Vⁱ)
```

```
 ┌──────────────────────────────────────────────────────────┐
 │                    Multi-Head Attention                   │
 │                                                          │
 │  Q ──→ [W_Q¹] ──→ head₁ ──┐                             │
 │  K ──→ [W_K¹] ──→         │                             │
 │  V ──→ [W_V¹] ──→         │                             │
 │                            │                             │
 │  Q ──→ [W_Q²] ──→ head₂ ──┼──→ Concat ──→ [W_O] ──→ Out│
 │  K ──→ [W_K²] ──→         │                             │
 │  V ──→ [W_V²] ──→         │                             │
 │                            │                             │
 │  Q ──→ [W_Qʰ] ──→ headₕ ──┘                             │
 │  K ──→ [W_Kʰ] ──→                                       │
 │  V ──→ [W_Vʰ] ──→                                       │
 └──────────────────────────────────────────────────────────┘
```

**Dimensions:**

```
Input:   Q, K, V each [batch, seq_len, d_model]
Per head: W_Qⁱ [d_model, d_k], W_Kⁱ [d_model, d_k], W_Vⁱ [d_model, d_v]
          where d_k = d_v = d_model / h
Head output:  [batch, seq_len, d_k]
Concat:       [batch, seq_len, h × d_k] = [batch, seq_len, d_model]
W_O:          [d_model, d_model]
Final output: [batch, seq_len, d_model]
```

**Key insight:** Total computation cost is roughly the same as single-head attention with full dimensionality, because each head operates on a reduced dimension `d_k = d_model / h`.

---

## **2.3 How Heads Specialize**

Research (Clark et al., 2019; Voita et al., 2019) shows that attention heads naturally specialize:

| Head Type | What It Learns | Example |
|-----------|---------------|---------|
| **Positional** | Attends to adjacent tokens | Previous/next token patterns |
| **Syntactic** | Subject-verb, modifier-noun | "dogs" ↔ "are" across distance |
| **Semantic** | Meaning-based similarity | "happy" ↔ "joyful" |
| **Rare token** | Attends to [SEP], [CLS], punctuation | Delimiter-focused heads |
| **Coreference** | Pronoun → antecedent | "she" → "Marie Curie" |
| **Induction** | Copy/pattern completion | [A][B]...[A] → predicts [B] |

> **Interview insight:** Not all heads are equally important. Voita et al. (2019) showed that many heads can be pruned without significant performance loss — only a subset are "essential."

---

## **2.4 Typical Configurations**

| Model | d_model | Heads (h) | d_k = d_model/h | Layers |
|-------|---------|-----------|------------------|--------|
| BERT-base | 768 | 12 | 64 | 12 |
| BERT-large | 1024 | 16 | 64 | 24 |
| GPT-2 | 768 | 12 | 64 | 12 |
| GPT-3 (175B) | 12288 | 96 | 128 | 96 |
| LLaMA-7B | 4096 | 32 | 128 | 32 |
| LLaMA-70B | 8192 | 64 | 128 | 80 |
| Mistral-7B | 4096 | 32 | 128 | 32 |

**Pattern:** `d_k` is typically 64 or 128 across all models. Models scale by increasing both the number of heads and layers.

---

# **3. Self-Attention vs Cross-Attention**

---

## **3.1 Self-Attention**

In self-attention, **Q, K, and V all come from the same sequence**. Each token attends to every other token (and itself) in the same sequence.

```
 Input: "The cat sat"
 
 Q, K, V all derived from: ["The", "cat", "sat"]

 Attention matrix (each row attends to all columns):
          The   cat   sat
 The   [ 0.1   0.3   0.6 ]
 cat   [ 0.2   0.1   0.7 ]
 sat   [ 0.1   0.8   0.1 ]
```

**Where it's used:**
- **Encoder self-attention** (BERT): Fully bidirectional — every token sees every other token.
- **Decoder self-attention** (GPT): Causal/masked — each token only sees previous tokens.

---

## **3.2 Cross-Attention**

In cross-attention, **Q comes from one sequence, while K and V come from a different sequence**. This is how the decoder "reads" the encoder's output.

```
 Encoder output (source):  ["Le", "chat", "est", "assis"]
 Decoder input  (target):  ["The", "cat"]

 Q from decoder: ["The", "cat"]
 K, V from encoder: ["Le", "chat", "est", "assis"]

 Cross-attention matrix:
              Le    chat   est   assis
 The       [ 0.5   0.1    0.3   0.1  ]
 cat       [ 0.1   0.7    0.1   0.1  ]
```

**Where it's used:**
- Encoder-decoder models: T5, BART, mBART, Whisper
- Translation, summarization, any seq2seq task
- Vision-language models: Q from text, K/V from image features

---

## **3.3 Causal (Masked) Self-Attention**

For autoregressive generation (GPT-style), each token can only attend to **itself and previous tokens**. This is enforced with a causal mask:

```
 Mask matrix (−∞ = masked, 0 = allowed):

          t₁    t₂    t₃    t₄
 t₁    [  0    −∞    −∞    −∞  ]
 t₂    [  0     0    −∞    −∞  ]
 t₃    [  0     0     0    −∞  ]
 t₄    [  0     0     0     0  ]

 After adding to scores and applying softmax:
 t₁ attends to: [t₁]
 t₂ attends to: [t₁, t₂]
 t₃ attends to: [t₁, t₂, t₃]
 t₄ attends to: [t₁, t₂, t₃, t₄]
```

**Why causal masking matters:**
- Prevents "cheating" — the model can't look at future tokens during training.
- Matches the inference regime where tokens are generated one at a time.
- Enables teacher forcing during training (parallel computation of all positions).

---

## **3.4 Comparison Table**

| Property | Self-Attention (Encoder) | Self-Attention (Decoder) | Cross-Attention |
|----------|--------------------------|--------------------------|-----------------|
| Q source | Same sequence | Same sequence | Decoder |
| K,V source | Same sequence | Same sequence | Encoder |
| Masking | None (bidirectional) | Causal (lower triangular) | None |
| Example model | BERT | GPT | T5, BART |
| Use case | Understanding, classification | Generation | Seq2seq |

---

# **4. Positional Encoding**

---

## **4.1 Why Positional Encoding Is Needed**

Transformers process all tokens **in parallel** — there's no recurrence, no convolution, no inherent notion of order. Without positional information:

```
 "The dog bit the man" and "The man bit the dog" 
 would produce IDENTICAL representations!
```

Self-attention is a **set operation** — it's permutation-equivariant. We must explicitly inject position information.

---

## **4.2 Sinusoidal Encoding (Vaswani et al., 2017)**

The original Transformer uses fixed sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in the sequence (0, 1, 2, ...)
- `i` = dimension index (0, 1, ..., d_model/2 - 1)
- `d_model` = embedding dimension

```
 Visualization (d_model=8, first 6 positions):

 pos  dim0  dim1  dim2  dim3  dim4  dim5  dim6  dim7
  0   0.00  1.00  0.00  1.00  0.00  1.00  0.00  1.00
  1   0.84  0.54  0.10  0.99  0.01  1.00  0.00  1.00
  2   0.91 -0.42  0.20  0.98  0.02  1.00  0.00  1.00
  3   0.14 -0.99  0.30  0.95  0.03  1.00  0.00  1.00
  4  -0.76 -0.65  0.39  0.92  0.04  1.00  0.00  1.00
  5  -0.96  0.28  0.48  0.88  0.05  1.00  0.01  1.00
       ↑ high freq              ↑ low frequency
```

**Key Properties:**
- **Unique encoding:** Each position gets a distinct vector.
- **Relative positions:** `PE(pos+k)` can be represented as a linear function of `PE(pos)` — the model can learn relative offsets.
- **Generalization:** Can extrapolate to longer sequences than seen in training (in theory).
- **No learned parameters:** Fixed, deterministic — added to token embeddings.

```
Final input = Token_Embedding(x) + Positional_Encoding(pos)
```

---

## **4.3 Learned Positional Embeddings (BERT, GPT-2)**

Instead of fixed sinusoids, learn a **position embedding matrix**:

```
PE = nn.Embedding(max_seq_length, d_model)

input = token_embed(x) + PE(positions)
```

| Property | Sinusoidal | Learned |
|----------|-----------|---------|
| Parameters | 0 (fixed) | max_len × d_model |
| Extrapolation | Theoretically yes | No — max_len is hard limit |
| Performance | Slightly worse | Slightly better (within training range) |
| Used by | Original Transformer | BERT, GPT-2, GPT-3 |

**Limitation:** Learned embeddings have a fixed maximum length (512 for BERT, 1024 for GPT-2). Cannot generalize beyond training length.

---

## **4.4 RoPE — Rotary Position Embedding**

**Used by:** LLaMA, LLaMA 2/3, Mistral, Qwen, CodeLlama, Phi-3

RoPE (Su et al., 2021) encodes position by **rotating** the query and key vectors in 2D subspaces:

```
 Core idea: Apply a rotation matrix R(θ) to Q and K based on position

 q_rotated(pos) = R(pos·θ) · q
 k_rotated(pos) = R(pos·θ) · k

 where R(θ) = [cos θ  -sin θ]   (2D rotation matrix)
               [sin θ   cos θ]
```

**Why RoPE is elegant:**

The dot product `q_rotated(m) · k_rotated(n)` depends only on the **relative position** `(m - n)`, not absolute positions:

```
q(m)ᵀ · k(n) = f(q, k, m-n)
```

This gives the model **relative position awareness** without explicit relative position biases.

**RoPE properties:**
- Decays attention naturally with distance (built-in distance decay).
- Better length generalization than learned embeddings.
- Can be extended with techniques like **NTK-aware scaling**, **YaRN**, **Code Llama's rope scaling** for longer contexts.
- Applied only to Q and K (not V) — values are position-independent.

```
 ┌──────────────────────────────────────────┐
 │        RoPE: Rotate in 2D subspaces      │
 │                                           │
 │  d_model split into d/2 pairs:            │
 │  (dim0, dim1), (dim2, dim3), ...          │
 │                                           │
 │  Each pair rotated by pos * θᵢ:           │
 │  θᵢ = 10000^(-2i/d)                       │
 │                                           │
 │  [q₀]     [cos(mθ₀)  -sin(mθ₀)] [q₀]    │
 │  [q₁]  =  [sin(mθ₀)   cos(mθ₀)] [q₁]    │
 └──────────────────────────────────────────┘
```

---

## **4.5 ALiBi — Attention with Linear Biases**

**Used by:** BLOOM, MPT

ALiBi (Press et al., 2022) takes a completely different approach — **no positional encoding at all**. Instead, it adds a linear bias to the attention scores based on distance:

```
Attention_score(i, j) = q_i · k_j - m · |i - j|

where m is a head-specific slope (fixed, not learned)
```

```
 Bias matrix (for one head with slope m):

          t₁    t₂    t₃    t₄
 t₁    [  0    -m    -2m   -3m  ]
 t₂    [ -m     0    -m    -2m  ]
 t₃    [ -2m   -m     0    -m   ]
 t₄    [ -3m   -2m   -m     0   ]
```

**Properties:**
- No learned positional parameters.
- Excellent length extrapolation — trained on 1024 tokens, works on 2048+.
- Different heads use different slopes `m` → different "attention spans."
- Very simple to implement.

---

## **4.6 Positional Encoding Comparison**

| Method | Type | Relative Pos | Extrapolation | Used By |
|--------|------|-------------|---------------|---------|
| Sinusoidal | Fixed additive | Implicit | Moderate | Original Transformer |
| Learned | Learned additive | No | Poor | BERT, GPT-2/3 |
| RoPE | Rotation on Q,K | Yes (explicit) | Good (with scaling) | LLaMA, Mistral, Phi |
| ALiBi | Attention bias | Yes (explicit) | Excellent | BLOOM, MPT |
| Relative PE | Learned bias | Yes | Moderate | T5, DeBERTa |

---

# **5. Transformer Architecture**

---

## **5.1 Full Block Diagram**

```
 ┌─────────────────────────────────────────────────────────┐
 │                  TRANSFORMER BLOCK                       │
 │                                                          │
 │  Input Tokens                                            │
 │       │                                                  │
 │       ▼                                                  │
 │  ┌──────────────┐                                        │
 │  │ Token Embed   │ + Positional Encoding                 │
 │  └──────┬───────┘                                        │
 │         │                                                │
 │    ╔════╧════════════════════════════╗                    │
 │    ║        Transformer Layer ×N     ║                    │
 │    ║                                 ║                    │
 │    ║  ┌──────────┐                   ║                    │
 │    ║  │ LayerNorm │                  ║                    │
 │    ║  └────┬─────┘                   ║                    │
 │    ║       ▼                         ║                    │
 │    ║  ┌──────────────────┐           ║                    │
 │    ║  │ Multi-Head Attn   │          ║                    │
 │    ║  └────┬─────────────┘           ║                    │
 │    ║       │                         ║                    │
 │    ║       + ←── Residual Connection ║                    │
 │    ║       │                         ║                    │
 │    ║  ┌──────────┐                   ║                    │
 │    ║  │ LayerNorm │                  ║                    │
 │    ║  └────┬─────┘                   ║                    │
 │    ║       ▼                         ║                    │
 │    ║  ┌──────────────────┐           ║                    │
 │    ║  │ Feed-Forward (FFN)│          ║                    │
 │    ║  └────┬─────────────┘           ║                    │
 │    ║       │                         ║                    │
 │    ║       + ←── Residual Connection ║                    │
 │    ║       │                         ║                    │
 │    ╚═══════╧═════════════════════════╝                    │
 │         │                                                │
 │         ▼                                                │
 │  ┌──────────────┐                                        │
 │  │  Final LN     │                                       │
 │  └──────┬───────┘                                        │
 │         ▼                                                │
 │  ┌──────────────┐                                        │
 │  │ Output Head   │  (LM head / classifier)               │
 │  └──────────────┘                                        │
 └─────────────────────────────────────────────────────────┘
```

---

## **5.2 Encoder-Only Architecture (BERT)**

```
 ┌────────────────────────────────────┐
 │          ENCODER-ONLY (BERT)       │
 │                                    │
 │  [CLS] The cat sat [SEP] [PAD]    │
 │    │    │   │   │    │     │       │
 │    ▼    ▼   ▼   ▼    ▼     ▼       │
 │  ┌──────────────────────────┐      │
 │  │ Token + Position + Segment│     │
 │  │       Embeddings          │     │
 │  └───────────┬──────────────┘      │
 │              │                     │
 │  ┌───────────▼──────────────┐      │
 │  │ Bidirectional Self-Attn  │ ×12  │
 │  │ (full attention, no mask)│      │
 │  └───────────┬──────────────┘      │
 │              │                     │
 │         [CLS] repr → classifier    │
 │     or  all reprs → token tasks    │
 └────────────────────────────────────┘
```

**Key characteristics:**
- **Bidirectional** — every token attends to every other token.
- Pre-trained with **Masked Language Modeling (MLM)** — predict [MASK] tokens.
- Also uses **Next Sentence Prediction (NSP)**.
- Cannot do autoregressive generation natively.

**Best for:** Classification, NER, extractive QA, sentence embeddings, semantic similarity.

**Models:** BERT, RoBERTa, DeBERTa, ALBERT, DistilBERT, ELECTRA.

---

## **5.3 Decoder-Only Architecture (GPT)**

```
 ┌────────────────────────────────────────┐
 │           DECODER-ONLY (GPT)           │
 │                                        │
 │  The   cat  sat   on   the   →  mat    │
 │   │     │    │    │    │         ↑      │
 │   ▼     ▼    ▼    ▼    ▼      predict   │
 │  ┌──────────────────────────┐          │
 │  │ Token + Position Embed   │          │
 │  └───────────┬──────────────┘          │
 │              │                         │
 │  ┌───────────▼──────────────┐          │
 │  │  Causal Self-Attention   │ ×N       │
 │  │  (lower-triangular mask) │          │
 │  └───────────┬──────────────┘          │
 │              │                         │
 │  ┌───────────▼──────────────┐          │
 │  │     LM Head (softmax     │          │
 │  │   over vocabulary)       │          │
 │  └──────────────────────────┘          │
 └────────────────────────────────────────┘
```

**Key characteristics:**
- **Autoregressive** — each token only sees previous tokens (causal mask).
- Pre-trained with **next-token prediction** (language modeling).
- Naturally suited for text generation.
- The dominant architecture for modern LLMs.

**Best for:** Text generation, chatbots, code generation, reasoning, general-purpose LLMs.

**Models:** GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Phi, Falcon, Claude, PaLM.

---

## **5.4 Encoder-Decoder Architecture (T5, BART)**

```
 ┌──────────────────────────────────────────────────────────┐
 │              ENCODER-DECODER (T5 / BART)                 │
 │                                                          │
 │  ENCODER                        DECODER                  │
 │  ┌─────────────────┐           ┌──────────────────┐      │
 │  │ Source tokens    │           │ Target tokens     │     │
 │  │ "Translate: the  │          │ "<s> Le chat"     │     │
 │  │  cat sat"        │          │                    │     │
 │  └────────┬────────┘           └────────┬──────────┘     │
 │           │                             │                │
 │  ┌────────▼────────┐           ┌────────▼──────────┐     │
 │  │ Bidirectional   │           │ Causal Self-Attn  │     │
 │  │ Self-Attention  │           └────────┬──────────┘     │
 │  └────────┬────────┘                    │                │
 │           │                    ┌────────▼──────────┐     │
 │           │                    │ Cross-Attention    │     │
 │           └──────────────────→ │ Q=decoder, KV=enc │     │
 │                                └────────┬──────────┘     │
 │                                         │                │
 │                                ┌────────▼──────────┐     │
 │                                │ Feed-Forward       │     │
 │                                └────────┬──────────┘     │
 │                                         ▼                │
 │                                    Output tokens         │
 └──────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- Encoder processes input bidirectionally.
- Decoder generates output autoregressively.
- **Cross-attention** bridges encoder and decoder.
- Three attention types per decoder layer: causal self-attn, cross-attn, FFN.

**Best for:** Translation, summarization, question answering, any input→output task.

**Models:** T5, BART, mBART, mT5, FLAN-T5, Whisper (speech-to-text).

---

## **5.5 Architecture Comparison**

| Feature | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---------|-------------|-------------|-----------------|
| Attention | Bidirectional | Causal | Bidirect + Causal + Cross |
| Pre-training | MLM | Next-token | Span corruption / denoising |
| Generation | No (natively) | Yes | Yes |
| Understanding | Excellent | Good | Good |
| Param efficiency | High | High | Lower (2 stacks) |
| Dominant use | Classification, NER | LLMs, chat, code | Translation, seq2seq |
| Key model | BERT, RoBERTa | GPT-4, LLaMA | T5, BART |

---

# **6. Feed-Forward Network (FFN)**

---

## **6.1 Standard FFN (Original Transformer)**

Each transformer layer contains a position-wise FFN applied independently to each token:

```
FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
```

```
                d_model        d_ff          d_model
 x ──→ [Linear] ──→ [GELU] ──→ [Linear] ──→ output
        W₁ (up)                  W₂ (down)
       d_model→d_ff             d_ff→d_model
```

- **Expansion ratio:** Typically `d_ff = 4 × d_model` (e.g., d_model=768 → d_ff=3072).
- **Activation:** Original used ReLU; modern models use GELU or SwiGLU.
- **Position-wise:** Same FFN applied to each token independently (parameters shared across positions, not across layers).

---

## **6.2 SwiGLU FFN (LLaMA-Style)**

Modern LLMs (LLaMA, Mistral, Phi) use the **SwiGLU** variant (Shazeer, 2020):

```
FFN_SwiGLU(x) = (Swish(x · W_gate) ⊙ (x · W_up)) · W_down
```

Where `⊙` is element-wise multiplication and `Swish(z) = z · σ(z)`.

```
          ┌──→ [W_gate] ──→ Swish(·) ──┐
 x ──────┤                              ⊙ ──→ [W_down] ──→ output
          └──→ [W_up]   ──────────────→─┘
```

**Three projection matrices:**

| Matrix | Shape | Role |
|--------|-------|------|
| `W_gate` (gate_proj) | d_model → d_ff | Gating signal |
| `W_up` (up_proj) | d_model → d_ff | Content signal |
| `W_down` (down_proj) | d_ff → d_model | Projection back |

**Why SwiGLU?**
- Empirically outperforms standard ReLU FFN.
- The gating mechanism allows the network to selectively filter information.
- Slightly more parameters (3 matrices vs 2) but better quality per parameter.

**Note:** With SwiGLU, d_ff is typically `(8/3) × d_model` (rounded to nearest multiple of 256) to keep total parameter count similar to the 4x standard FFN.

---

## **6.3 What the FFN Actually Does**

The attention layer handles **token interaction** (which tokens relate to which). The FFN handles **per-token feature transformation** — it's where:

- Factual knowledge is stored (Geva et al., 2021 — "Transformer FFNs are key-value memories")
- Non-linear feature combinations are computed
- The model performs "thinking" on individual token representations

```
 Attention: "Who should I pay attention to?" (mixing)
 FFN:       "Given what I've gathered, what do I conclude?" (processing)
```

---

# **7. Key Architectural Details**

---

## **7.1 Pre-LN vs Post-LN**

**Post-LN (Original Transformer):**
```
output = LayerNorm(x + Sublayer(x))
```

**Pre-LN (GPT-2, LLaMA, most modern LLMs):**
```
output = x + Sublayer(LayerNorm(x))
```

```
 Post-LN:                      Pre-LN:
 x ──→ Sublayer ──→ Add ──→ LN    x ──→ LN ──→ Sublayer ──→ Add
       ↑              ↑                  ↑                    ↑
       └── residual ──┘                  └──── residual ──────┘
```

| Aspect | Post-LN | Pre-LN |
|--------|---------|--------|
| Training stability | Less stable, needs warmup | More stable |
| Gradient flow | Can have gradient issues | Smoother gradients |
| Performance | Slightly better (with careful tuning) | Slightly worse but easier |
| Used by | Original Transformer, BERT | GPT-2, LLaMA, Mistral |

**Why Pre-LN dominates now:** It makes training much more stable, especially for very deep models (96+ layers). The residual connection goes directly from input to output without passing through LayerNorm, ensuring good gradient flow.

---

## **7.2 Residual Connections**

Every sub-layer (attention, FFN) has a **skip/residual connection**:

```
output = x + Sublayer(x)
```

**Why residual connections are critical:**
- Enable training of very deep networks (32–96+ layers).
- Gradients flow directly through the skip path — avoids vanishing gradients.
- Each layer learns a **residual function** (what to add) rather than the full transformation.
- The "residual stream" view: all layers read from and write to a shared residual stream.

```
 ┌──────────────────────────────────────────┐
 │         The Residual Stream View          │
 │                                           │
 │  embed → [+attn₁] → [+ffn₁] → [+attn₂]  │
 │          → [+ffn₂] → ... → [+attnₙ]      │
 │          → [+ffnₙ] → output              │
 │                                           │
 │  Each layer adds its contribution to a    │
 │  running sum (the residual stream)        │
 └──────────────────────────────────────────┘
```

---

## **7.3 KV-Cache for Inference**

During autoregressive generation, the model generates one token at a time. Without caching, at step `t`, we'd recompute attention over all `t` tokens — wasting computation on tokens we already processed.

**KV-Cache stores the Key and Value tensors** from previous steps:

```
 Without KV-cache (step t):
   Q = [q₁, q₂, ..., qₜ]     ← compute all
   K = [k₁, k₂, ..., kₜ]     ← compute all
   V = [v₁, v₂, ..., vₜ]     ← compute all
   Attention(Q, K, V)          ← O(t² · d) computation

 With KV-cache (step t):
   Q = [qₜ]                   ← only new token
   K = cached_K + [kₜ]        ← append new key
   V = cached_V + [vₜ]        ← append new value
   Attention(Q, K, V)          ← O(t · d) computation
```

```
 Step 1: "The"     → compute k₁,v₁, cache them
 Step 2: "cat"     → compute k₂,v₂, append to cache. Q=[q₂], K=[k₁,k₂]
 Step 3: "sat"     → compute k₃,v₃, append to cache. Q=[q₃], K=[k₁,k₂,k₃]
 ...
 Step t: new_token → compute kₜ,vₜ, append. Q=[qₜ], K=[k₁,...,kₜ]
```

**Memory cost:** KV-cache grows linearly with sequence length:

```
KV-cache memory = 2 × n_layers × n_heads × seq_len × d_head × bytes_per_param
```

For LLaMA-70B with 8K context (FP16):
```
= 2 × 80 × 64 × 8192 × 128 × 2 bytes ≈ 20 GB
```

This is why **long-context models are memory-hungry** during inference.

---

## **7.4 Flash Attention**

**Flash Attention** (Dao et al., 2022) is a hardware-aware attention algorithm that is:
- **Exact** — produces the same result as standard attention (not an approximation).
- **Memory-efficient** — O(N) memory instead of O(N²).
- **Faster** — 2-4x speedup on typical workloads.

**The key insight: attention is IO-bound, not compute-bound.**

Standard attention materializes the full N×N attention matrix in GPU HBM (slow memory):

```
 Standard Attention (IO pattern):
 
 1. Compute S = Q·Kᵀ      → Write N×N matrix to HBM
 2. Compute A = softmax(S) → Read N×N, write N×N to HBM
 3. Compute O = A·V        → Read N×N, read V, write output

 Total HBM reads/writes: O(N² · d + N²) — dominated by N² attention matrix
```

Flash Attention uses **tiling and online softmax** to avoid materializing the full matrix:

```
 Flash Attention (IO pattern):

 1. Load blocks of Q, K, V from HBM to SRAM (fast on-chip memory)
 2. Compute attention for that block in SRAM
 3. Use online softmax to maintain running statistics
 4. Write only the final output block back to HBM

 Total HBM reads/writes: O(N · d) — linear!
```

```
 ┌──────────────────────────────────────────┐
 │            GPU Memory Hierarchy           │
 │                                           │
 │  SRAM (on-chip): ~20 MB, ~19 TB/s        │
 │  HBM  (off-chip): ~40 GB, ~1.5 TB/s      │
 │                                           │
 │  Flash Attention keeps computation in     │
 │  SRAM as much as possible, minimizing     │
 │  expensive HBM reads/writes.              │
 └──────────────────────────────────────────┘
```

**Flash Attention 2** (Dao, 2023): Further optimized with better parallelism and work partitioning — additional 2x speedup.

**Flash Attention 3** (2024): Targets H100 GPUs, uses FP8, asynchronous operations.

---

## **7.5 Grouped Query Attention (GQA)**

**Used by:** LLaMA 2 (70B), Mistral, Gemma

Standard Multi-Head Attention (MHA) has separate K, V projections per head. This means the **KV-cache** scales linearly with the number of heads — expensive for large models.

**GQA** groups multiple query heads to share a single K,V head:

```
 Multi-Head Attention (MHA):    h_q = h_kv = 32
 ┌──────────────────────────────────────────┐
 │ Q heads:  q₁  q₂  q₃  q₄  ... q₃₂      │
 │ KV heads: kv₁ kv₂ kv₃ kv₄ ... kv₃₂     │
 │           ↕   ↕   ↕   ↕       ↕         │
 │           1:1 mapping                     │
 └──────────────────────────────────────────┘

 Grouped Query Attention (GQA):  h_q = 32, h_kv = 8
 ┌──────────────────────────────────────────┐
 │ Q heads:  q₁ q₂ q₃ q₄ | q₅ q₆ q₇ q₈ | ...│
 │ KV heads:    kv₁       |    kv₂        | ...│
 │           4:1 mapping                     │
 └──────────────────────────────────────────┘

 Multi-Query Attention (MQA):  h_q = 32, h_kv = 1
 ┌──────────────────────────────────────────┐
 │ Q heads:  q₁ q₂ q₃ q₄ ... q₃₂          │
 │ KV heads:         kv₁                   │
 │           32:1 mapping (all share one)   │
 └──────────────────────────────────────────┘
```

| Method | KV Heads | KV-Cache Size | Quality | Speed |
|--------|----------|---------------|---------|-------|
| MHA | h | 1× (baseline) | Best | Baseline |
| GQA | h/g (e.g., 8) | g× reduction | Near MHA | Faster |
| MQA | 1 | h× reduction | Slightly worse | Fastest |

**GQA is the sweet spot** — nearly as good as MHA quality, with significantly reduced KV-cache memory and faster inference.

---

# **8. Computational Complexity**

---

## **8.1 Self-Attention is O(n² · d)**

For a sequence of length `n` with dimension `d`:

```
 Q · Kᵀ:     [n × d] · [d × n] = [n × n]    → O(n² · d) compute, O(n²) memory
 softmax:    [n × n]                           → O(n²) 
 A · V:      [n × n] · [n × d] = [n × d]      → O(n² · d) compute
```

**Total:** O(n² · d) computation, O(n²) memory for the attention matrix.

---

## **8.2 Why This Is a Problem**

```
 Sequence Length    Attention Matrix Size    Relative Cost
      128                 16,384               1×
      512                262,144              16×
     1024              1,048,576              64×
     2048              4,194,304             256×
     8192             67,108,864           4,096×
    32768          1,073,741,824          65,536×
   128000         16,384,000,000       1,000,000×
```

For GPT-4 with 128K context: the raw attention matrix for a single head and single layer would be 128K × 128K = 16.4 billion entries. With 96 heads and 120 layers, this becomes intractable without optimizations.

---

## **8.3 Solutions for Long Contexts**

| Approach | Complexity | Key Idea | Models |
|----------|-----------|----------|--------|
| **Full Attention** | O(n²) | Standard, exact | GPT, BERT |
| **Flash Attention** | O(n²) compute, O(n) memory | IO-aware tiling | LLaMA, Mistral |
| **Sparse Attention** | O(n√n) or O(n·k) | Attend to subset | GPT-3, BigBird |
| **Longformer** | O(n) | Local + global attention | Longformer, LED |
| **Linear Attention** | O(n·d) | Kernel trick, no softmax | Linear Transformer |
| **Ring Attention** | O(n²/p) | Distribute across devices | Very long context |
| **Sliding Window** | O(n·w) | Fixed window size | Mistral (w=4096) |
| **State Space Models** | O(n·d) | Recurrent alternative | Mamba, S4 |

**Sliding Window Attention (Mistral):**
```
 Full attention:           Sliding window (w=3):
 ████████                  ███░░░░░
 ████████                  ░███░░░░
 ████████                  ░░███░░░
 ████████                  ░░░███░░
 ████████                  ░░░░███░
 ████████                  ░░░░░███

 Every token sees all      Each token sees only
                           w=3 nearest tokens
```

Mistral-7B uses sliding window in attention layers but information still propagates across the full context through **stacked layers** — information can travel `w × n_layers` positions total.

---

# **9. Scaling Laws**

---

## **9.1 Kaplan et al. (2020) — Neural Scaling Laws**

OpenAI discovered that LLM performance (measured by loss) follows **power laws** with respect to three factors:

```
L(N) ∝ N^(-αₙ)     Loss vs Parameters
L(D) ∝ D^(-α_d)    Loss vs Dataset size
L(C) ∝ C^(-α_c)    Loss vs Compute
```

**Key findings:**
- Performance improves smoothly and predictably with scale.
- **Parameters matter more than data** (Kaplan's original finding, later revised).
- Shape of the model (depth vs width) matters less than total parameters.
- These laws held across 7 orders of magnitude.

```
 Loss
  │╲
  │  ╲
  │    ╲
  │      ╲──────
  │              ╲──────
  │                      ╲─────── (diminishing returns)
  └───────────────────────────── log(Parameters or Compute)
```

---

## **9.2 Chinchilla Scaling (Hoffmann et al., 2022)**

DeepMind's Chinchilla paper **revised** Kaplan's findings:

> **For compute-optimal training, model size and dataset size should scale proportionally.**

**Chinchilla optimal ratio:** ~20 tokens per parameter.

| Model | Parameters | Training Tokens | Tokens/Param | Status |
|-------|-----------|----------------|-------------|--------|
| GPT-3 | 175B | 300B | 1.7 | **Undertrained** |
| Gopher | 280B | 300B | 1.1 | **Very undertrained** |
| Chinchilla | 70B | 1.4T | 20 | **Optimal** |
| LLaMA | 65B | 1.4T | 21.5 | **~Optimal** |
| LLaMA 2 | 70B | 2T | 28.6 | **Over-trained** (intentionally) |
| Mistral-7B | 7B | ~8T* | ~1140* | **Very over-trained** |

**Key insight:** Many earlier LLMs were **too large for their data budget**. It's better to train a smaller model on more data than a larger model on less data (for the same compute budget).

**Modern trend:** The industry has moved toward **over-training** (more tokens/param than Chinchilla optimal) because inference cost scales with model size but not training data. A smaller, over-trained model is cheaper to deploy.

---

## **9.3 Practical Implications**

```
 Compute-optimal frontier (Chinchilla):
 
 Compute Budget     Optimal Params    Optimal Tokens
 10¹⁸ FLOPs        ~400M             ~8B
 10²⁰ FLOPs        ~4B               ~80B
 10²² FLOPs        ~40B              ~800B
 10²³ FLOPs        ~70B              ~1.4T
 10²⁴ FLOPs        ~200B             ~4T
```

> **Interview tip:** If asked "how would you decide model size?", mention Chinchilla scaling, then note the modern preference for over-training smaller models for inference efficiency.

---

# **10. Key Model Variants**

---

## **10.1 Comprehensive Model Reference**

| Model | Org | Year | Arch | Params | Special Innovation |
|-------|-----|------|------|--------|-------------------|
| **BERT** | Google | 2018 | Encoder | 110M/340M | MLM pre-training, bidirectional |
| **GPT-2** | OpenAI | 2019 | Decoder | 1.5B | Showed emergent zero-shot abilities |
| **T5** | Google | 2020 | Enc-Dec | 11B | "Text-to-text" unification |
| **GPT-3** | OpenAI | 2020 | Decoder | 175B | In-context learning, few-shot |
| **PaLM** | Google | 2022 | Decoder | 540B | Pathways system, chain-of-thought |
| **Chinchilla** | DeepMind | 2022 | Decoder | 70B | Compute-optimal scaling |
| **LLaMA** | Meta | 2023 | Decoder | 7–65B | Open-weight, efficient, RoPE, SwiGLU |
| **LLaMA 2** | Meta | 2023 | Decoder | 7–70B | GQA, 4K→128K context, RLHF |
| **Mistral-7B** | Mistral | 2023 | Decoder | 7B | Sliding window attn, GQA, punches above weight |
| **Phi-2/3** | Microsoft | 2023–24 | Decoder | 2.7B/3.8B | "Textbook quality" data, small but mighty |
| **GPT-4** | OpenAI | 2023 | Decoder* | ~1.8T (MoE)* | SOTA reasoning, multimodal |
| **LLaMA 3** | Meta | 2024 | Decoder | 8–405B | 15T tokens, 128K context, GQA |

*Rumored / speculated architecture

---

## **10.2 What Makes Each Special**

### BERT
- **Innovation:** Proved bidirectional pre-training is powerful for understanding.
- **MLM objective:** Randomly mask 15% of tokens, predict them from context.
- **Impact:** Revolutionized NLP benchmarks, spawned RoBERTa, DeBERTa, ALBERT.

### GPT-2 / GPT-3
- **Innovation:** Scaled the decoder-only + language modeling paradigm.
- **GPT-3:** Discovered in-context learning — few-shot prompting without fine-tuning.
- **Impact:** Launched the "scaling is all you need" era.

### T5
- **Innovation:** Cast every NLP task as text-to-text ("translate English to German: ...", "summarize: ...").
- **Impact:** Unified framework for all NLP tasks. FLAN-T5 added instruction tuning.

### LLaMA
- **Innovation:** Showed open models can match proprietary ones. Clean architecture with:
  - Pre-LN (RMSNorm)
  - RoPE positional encoding
  - SwiGLU FFN
  - No bias terms
- **Impact:** Foundation for most open-source LLM ecosystem (Alpaca, Vicuna, etc.).

### Mistral-7B
- **Innovation:** Sliding window attention + GQA at 7B scale.
- Outperforms LLaMA 2 13B on many benchmarks despite being half the size.
- **Impact:** Proved efficiency innovations can compensate for fewer parameters.

### Phi Series (Microsoft)
- **Innovation:** "Data quality over data quantity" — trained on curated "textbook-quality" data.
- Phi-2 (2.7B) matches or exceeds models 10x its size on reasoning benchmarks.
- **Impact:** Challenged the "bigger is better" assumption.

---

## **10.3 Modern LLM Architecture Template**

Almost all modern open LLMs follow this blueprint (established by LLaMA):

```
 ┌────────────────────────────────────────┐
 │     Modern LLM Architecture (2024+)    │
 │                                        │
 │  Normalization:   RMSNorm (Pre-LN)     │
 │  Attention:       GQA or MHA           │
 │  Position:        RoPE                 │
 │  FFN:             SwiGLU               │
 │  Bias:            None (no bias terms) │
 │  Activation:      SiLU/Swish           │
 │  Vocab size:      32K–128K             │
 │  Context:         4K–128K+             │
 │  Training:        BF16 or FP8          │
 └────────────────────────────────────────┘
```

---

# **11. Common Interview Questions (With Strong Answers)**

---

## **Q1: "Explain the transformer architecture."**

**Strong answer framework (2 minutes):**

> "The transformer, introduced by Vaswani et al. in 2017, replaces recurrence with self-attention. The core idea is that every token can attend to every other token in parallel, computing relevance scores via scaled dot-product attention.
>
> A transformer block has two sub-layers: multi-head self-attention and a position-wise feed-forward network, each wrapped with residual connections and layer normalization.
>
> There are three variants: **encoder-only** (BERT) for understanding tasks using bidirectional attention; **decoder-only** (GPT) for generation using causal masking; and **encoder-decoder** (T5) for sequence-to-sequence tasks with cross-attention.
>
> Modern LLMs are almost exclusively decoder-only, using innovations like RoPE for positional encoding, SwiGLU for FFN, GQA for efficient KV-caching, and Flash Attention for memory-efficient training."

---

## **Q2: "Why is attention O(n²)?"**

> "Self-attention computes a score between every pair of tokens. For a sequence of length n, we compute Q·Kᵀ which is an [n×d] times [d×n] matrix multiplication, producing an n×n attention matrix. Each entry requires O(d) multiplications, giving O(n²·d) total. The n² memory for the attention matrix is the real bottleneck — for a 128K context, that's 16 billion entries per head per layer.
>
> Solutions include Flash Attention (same O(n²) compute but O(n) memory via tiling), sliding window attention (O(n·w) where w is the window size), and architectural alternatives like state space models (Mamba) that are O(n·d)."

---

## **Q3: "What is KV-cache?"**

> "During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, we'd recompute the K and V projections for all previous tokens at every step — O(n²) total work across n steps.
>
> KV-cache stores the key and value tensors from all previous steps. At step t, we only compute the new kₜ, vₜ, append them to the cache, and compute attention between the single new query qₜ and all cached keys/values. This reduces per-step compute from O(t·d) to O(t·d) but avoids redundant recomputation.
>
> The trade-off is memory: KV-cache for LLaMA-70B with 8K context requires ~20GB. This is why GQA (Grouped Query Attention) is important — it reduces KV-cache size by sharing K,V heads across multiple query heads."

---

## **Q4: "Encoder-only vs decoder-only vs encoder-decoder?"**

> "**Encoder-only** (BERT): bidirectional attention, trained with masked language modeling. Each token sees the full context in both directions. Best for classification, NER, and semantic similarity. Can't generate text natively.
>
> **Decoder-only** (GPT): causal attention with a triangular mask, trained with next-token prediction. Each token only sees previous tokens. Best for generation, and has become the dominant LLM architecture because of its simplicity and strong emergent abilities through scaling.
>
> **Encoder-decoder** (T5): encoder processes input bidirectionally, decoder generates output autoregressively with cross-attention to the encoder. Best for tasks with clear input-output structure like translation and summarization.
>
> The trend has been toward decoder-only: it's simpler, scales well, and with enough data and parameters, can handle understanding tasks too (via instruction tuning and in-context learning)."

---

## **Q5: "What is Flash Attention?"**

> "Flash Attention is a hardware-aware, exact attention algorithm that reduces memory from O(n²) to O(n) and is 2-4x faster than standard attention.
>
> The key insight is that standard attention is IO-bound — the bottleneck is reading/writing the n×n attention matrix to GPU HBM (high-bandwidth memory), not the actual computation. Flash Attention uses tiling to process attention in blocks that fit in the GPU's fast on-chip SRAM (~20MB, ~19 TB/s) instead of repeatedly accessing the slower HBM (~40GB, ~1.5 TB/s).
>
> It uses an online softmax algorithm to compute exact softmax across tiles without ever materializing the full attention matrix. The result is mathematically identical to standard attention — it's purely an implementation optimization, not an approximation.
>
> Flash Attention 2 and 3 further optimize parallelism and support FP8 on newer hardware."

---

## **Q6: "Explain positional encoding."**

> "Transformers process tokens in parallel with no inherent sense of order — self-attention is permutation-equivariant. Without positional encoding, 'dog bites man' and 'man bites dog' would produce the same representation.
>
> The original Transformer used fixed sinusoidal functions of different frequencies, which encode absolute position and allow the model to learn relative positions. BERT and GPT-2 used learned position embeddings — a simple embedding lookup table indexed by position.
>
> Modern LLMs primarily use RoPE (Rotary Position Embedding), which encodes position by rotating query and key vectors in 2D subspaces. The elegant property is that the dot product between rotated Q and K depends only on their relative position, not absolute position. RoPE also generalizes better to longer sequences with techniques like NTK-aware scaling.
>
> ALiBi is another approach that adds a distance-dependent linear bias directly to attention scores, with excellent length extrapolation properties."

---

## **Q7: "What is RoPE?"**

> "RoPE — Rotary Position Embedding — encodes position information by applying a rotation to query and key vectors before computing attention. Each pair of dimensions (d₀, d₁), (d₂, d₃), etc. is rotated by an angle proportional to the position, with different frequencies for different dimension pairs (similar to sinusoidal encoding).
>
> The key mathematical property: when you compute the dot product of a rotated query at position m with a rotated key at position n, the result depends only on (m-n), the relative distance. This gives the model explicit relative position awareness.
>
> RoPE has a natural distance decay — attention scores decrease for distant tokens. It's applied only to Q and K, not V. It's used by LLaMA, Mistral, Phi, and most modern open LLMs. For extending to longer contexts, techniques like NTK-aware scaling and YaRN modify the RoPE frequencies to generalize beyond the training context length."

---

# **12. Key Takeaways for LLM Interviews**

---

## **Must-Know Concepts (Guaranteed to Come Up)**

| # | Topic | One-Line Summary |
|---|-------|-----------------|
| 1 | Scaled dot-product attention | softmax(QKᵀ/√d_k)V — scale prevents softmax saturation |
| 2 | Multi-head attention | Multiple parallel attention heads capture different relationships |
| 3 | KV-cache | Store past K,V to avoid recomputation during generation |
| 4 | Encoder vs Decoder | Bidirectional understanding vs causal generation |
| 5 | Positional encoding | Inject position info; RoPE is modern standard |
| 6 | Residual connections | Enable deep networks, gradient flow |
| 7 | O(n²) complexity | Quadratic in sequence length, addressed by Flash Attention |
| 8 | Pre-LN | Normalize before sublayer, not after (modern default) |

---

## **Differentiator Concepts (Set You Apart)**

| # | Topic | Why It Impresses |
|---|-------|-----------------|
| 1 | Flash Attention | Shows you understand hardware-software co-design |
| 2 | GQA/MQA | Shows awareness of inference optimization |
| 3 | SwiGLU FFN | Shows you know modern architecture details |
| 4 | Chinchilla scaling | Shows you think about training efficiency |
| 5 | Residual stream view | Shows mechanistic interpretability awareness |
| 6 | RoPE internals | Shows depth beyond surface-level knowledge |
| 7 | Sliding window attention | Shows practical knowledge (Mistral) |
| 8 | Over-training trend | Shows you follow cutting-edge developments |

---

## **Common Pitfalls to Avoid**

- **Don't say "attention replaces RNNs because it's faster."** Say it enables parallelization and captures long-range dependencies without sequential bottleneck.
- **Don't confuse self-attention with cross-attention.** Always specify where Q, K, V come from.
- **Don't say "transformers have no positional information."** They do — through positional encoding. They have no **inherent** positional information.
- **Don't say "Flash Attention is approximate."** It's exact — purely an implementation optimization.
- **Don't forget the FFN.** Attention gets all the hype, but the FFN has more parameters and stores factual knowledge.

---

## **Quick Reference: The Modern LLM Stack**

```
 ┌─────────────────────────────────────────────────────┐
 │              MODERN LLM (LLaMA-style)               │
 │                                                      │
 │  Tokenizer:       BPE (SentencePiece)               │
 │  Embedding:       Token embedding (no segment)       │
 │  Position:        RoPE                               │
 │  Norm:            RMSNorm (Pre-LN)                   │
 │  Attention:       GQA + Flash Attention              │
 │  FFN:             SwiGLU (gate, up, down projections)│
 │  Bias:            None                               │
 │  Precision:       BF16 training, INT4/INT8 inference │
 │  KV-Cache:        Yes (with GQA for efficiency)      │
 │  Context:         8K–128K+ (with RoPE scaling)       │
 │  Training:        Chinchilla-inspired, often over-    │
 │                   trained for inference efficiency    │
 │  Alignment:       SFT → RLHF/DPO                    │
 └─────────────────────────────────────────────────────┘
```

---

## **Interview Day Checklist**

- [ ] Can you draw the transformer block diagram from memory?
- [ ] Can you write the attention formula and explain each component?
- [ ] Can you explain why we scale by √d_k?
- [ ] Can you compare encoder-only, decoder-only, encoder-decoder?
- [ ] Can you explain KV-cache and why GQA helps?
- [ ] Can you explain Flash Attention at a high level?
- [ ] Can you describe RoPE and why it replaced learned embeddings?
- [ ] Can you discuss Chinchilla scaling and its implications?
- [ ] Can you name 3+ architectural differences between BERT and LLaMA?
- [ ] Can you explain why attention is O(n²) and name 2+ solutions?

---

*Last updated: February 2026*
*Prepared for: Rahul Sharma — Data Scientist / ML Engineer interviews*
