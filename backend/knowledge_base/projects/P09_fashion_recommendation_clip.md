# P09: Multimodal Fashion Recommendation System (CLIP + BLIP + DensePose)

> **Project Type:** Academic Project | **Timeline:** Fall 2024
> **Institution:** University of Maryland, College Park
> **GitHub:** [github.com/rahul370139/Multimodal-Fashion-Recommendation](https://github.com/rahul370139/Multimodal-Fashion-Recommendation)

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

| Component | Detail |
|-----------|--------|
| **Situation** | Fashion e-commerce platforms struggle with the "semantic gap" -- users describe clothing in natural language ("flowy red summer dress with lace") but catalogs are indexed by rigid attribute tags. Traditional keyword-based or category-based recommendation engines fail to capture the nuanced, cross-modal relationship between visual style and textual intent, leading to poor retrieval relevance and user frustration. |
| **Task** | Build a multimodal fashion retrieval and recommendation system that bridges the text-image semantic gap by fine-tuning CLIP encoders on fashion-domain data, integrating BLIP-generated captions for richer text supervision, applying DensePose body segmentation to focus on clothing regions, and deploying FAISS-based dense retrieval for sub-second search across 44,000+ items. |
| **Action** | Fine-tuned CLIP (RN50) vision and text encoders using LoRA (rank 16, alpha 32) with contrastive learning on BLIP-generated fashion captions and segmentation-masked images from the DeepFashion dataset. Built a FAISS retrieval pipeline (IndexFlatIP / IVF) with adaptive query reweighting (user-controlled alpha blending of image and text embeddings). Implemented a Streamlit UI with image+text search, text-only search, AI stylist chat (Ollama/Mistral), and virtual wardrobe with Supabase persistence. |
| **Result** | Achieved **88% Mean Opinion Score (MOS)** against human relevance judgments. System indexes 44,096 fashion items with sub-second retrieval. LoRA fine-tuning achieved 0.096% parameter efficiency while significantly improving fashion-domain text-image alignment over base CLIP. |

### 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│          MULTIMODAL FASHION RECOMMENDATION SYSTEM ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    OFFLINE PIPELINE (Data Preparation)                    │    │
│  │                                                                          │    │
│  │  DeepFashion Dataset (44,096 images)                                     │    │
│  │       │                                                                  │    │
│  │       ├──────────────────┐                                               │    │
│  │       ▼                  ▼                                               │    │
│  │  ┌──────────┐     ┌──────────────┐     ┌──────────────┐                 │    │
│  │  │ DensePose│     │  BLIP        │     │ Segmentation │                 │    │
│  │  │ Body UV  │     │  Caption Gen │     │ Mask Extract │                 │    │
│  │  │ Mapping  │     │  (Few-Shot)  │     │ Binary Masks │                 │    │
│  │  └─────┬────┘     └──────┬───────┘     └──────┬───────┘                 │    │
│  │        │                 │                     │                         │    │
│  │        ▼                 ▼                     ▼                         │    │
│  │  ┌──────────────────────────────────────────────────┐                   │    │
│  │  │        LoRA Fine-tuned CLIP (RN50)               │                   │    │
│  │  │  ┌──────────────┐    ┌──────────────┐            │                   │    │
│  │  │  │ Vision Enc   │    │ Text Encoder │            │                   │    │
│  │  │  │ (ResNet-50)  │    │ (Transformer)│            │                   │    │
│  │  │  │ + LoRA r=16  │    │ + LoRA r=16  │            │                   │    │
│  │  │  └──────┬───────┘    └──────┬───────┘            │                   │    │
│  │  │         │                   │                     │                   │    │
│  │  │         ▼                   ▼                     │                   │    │
│  │  │    Image Embeddings   Text Embeddings             │                   │    │
│  │  │    (1024-dim, L2-norm) (1024-dim, L2-norm)        │                   │    │
│  │  └──────────┬───────────────────┬───────────────────-┘                   │    │
│  │             │                   │                                        │    │
│  │             ▼                   ▼                                        │    │
│  │       ┌──────────────────────────────┐                                  │    │
│  │       │   FAISS Index Construction   │                                  │    │
│  │       │   IndexFlatIP / IndexIVFFlat │                                  │    │
│  │       │   Inner Product Similarity   │                                  │    │
│  │       └──────────────────────────────┘                                  │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │                    ONLINE PIPELINE (Inference / Search)                   │    │
│  │                                                                          │    │
│  │  ┌──────────────────────────────────────┐                               │    │
│  │  │        Streamlit Frontend            │                               │    │
│  │  │  ┌──────────┐ ┌────────┐ ┌────────┐ │                               │    │
│  │  │  │ Fashion  │ │ AI     │ │Virtual │ │                               │    │
│  │  │  │ Search   │ │Stylist │ │Wardrobe│ │                               │    │
│  │  │  └─────┬────┘ └───┬────┘ └───┬────┘ │                               │    │
│  │  └────────┼──────────┼──────────┼───────┘                               │    │
│  │           │          │          │                                        │    │
│  │           ▼          ▼          ▼                                        │    │
│  │  ┌────────────┐ ┌────────┐ ┌──────────┐                                │    │
│  │  │ Query Enc  │ │ Ollama │ │ Supabase │                                │    │
│  │  │ (LoRA CLIP)│ │Mistral │ │PostgreSQL│                                │    │
│  │  └─────┬──────┘ └────────┘ └──────────┘                                │    │
│  │        │                                                                │    │
│  │        ▼                                                                │    │
│  │  ┌──────────────────────────────────────────┐                          │    │
│  │  │  Adaptive Query Blending                 │                          │    │
│  │  │  q = (1-α)·img_emb + α·txt_emb          │                          │    │
│  │  │  α ∈ [0,1], user-controlled              │                          │    │
│  │  └─────────────────┬────────────────────────┘                          │    │
│  │                    │                                                    │    │
│  │                    ▼                                                    │    │
│  │  ┌──────────────────────────────────────────┐                          │    │
│  │  │  FAISS Search (Inner Product)            │                          │    │
│  │  │  top_k * 10 → min_score filter → top_k   │                          │    │
│  │  └─────────────────┬────────────────────────┘                          │    │
│  │                    │                                                    │    │
│  │                    ▼                                                    │    │
│  │            Ranked Fashion Results                                       │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Full Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Vision Encoder** | CLIP RN50 + LoRA | ResNet-50 backbone, LoRA rank 16, alpha 32, 0.096% trainable parameters |
| **Text Encoder** | CLIP Transformer + LoRA | 12-layer transformer, LoRA on q/k/v/out projections |
| **Caption Generation** | BLIP | `Salesforce/blip-image-captioning-base`, few-shot guided, beam search (5 beams) |
| **Body Segmentation** | DensePose (Detectron2) | Dense UV body mapping, binary mask extraction for clothing isolation |
| **Retrieval** | FAISS | `IndexFlatIP` (exact) / `IndexIVFFlat` (approximate), inner product similarity |
| **Fine-tuning** | PEFT (LoRA) | `peft` library, `TaskType.FEATURE_EXTRACTION`, AdamW optimizer |
| **Training** | PyTorch | Contrastive loss (symmetric cross-entropy), temperature τ=0.07, cosine annealing LR |
| **Dataset** | DeepFashion | 44,096 fashion images, men's and women's clothing across multiple categories |
| **Frontend** | Streamlit | Image upload, text search, alpha slider, AI stylist chat, virtual wardrobe |
| **AI Stylist** | Ollama + Mistral | LangChain integration, `ChatPromptTemplate`, async inference |
| **Database** | Supabase (PostgreSQL) | Virtual wardrobe persistence, user item management |
| **Embeddings** | NumPy (.npy) | Pre-computed embeddings for 44K items, batch processing in 1000-image chunks |

---

## 2. Deep Technical Walkthrough

### 2.1 CLIP Architecture: The Foundation

**What is CLIP?**

CLIP (Contrastive Language-Image Pre-training) is OpenAI's dual-encoder model that learns a shared embedding space for images and text through contrastive learning on 400M image-text pairs scraped from the internet.

```
                         CLIP Dual-Encoder Architecture
 ═══════════════════════════════════════════════════════════════════

   Image Input                              Text Input
   (224×224×3)                              ("a red dress...")
       │                                         │
       ▼                                         ▼
 ┌─────────────┐                          ┌─────────────┐
 │ Vision Enc  │                          │ Text Encoder│
 │             │                          │             │
 │ RN50:       │                          │ 12-layer    │
 │ ResNet-50   │                          │ Transformer │
 │ → 2048-dim  │                          │ 512 vocab   │
 │ → proj head │                          │ 49,152 ctx  │
 │ → 1024-dim  │                          │ → 1024-dim  │
 │             │                          │             │
 │ or ViT-B/32:│                          │ [CLS] token │
 │ 12-layer ViT│                          │ → proj head │
 │ patch=32×32 │                          │ → D-dim     │
 │ → 512-dim   │                          │             │
 └──────┬──────┘                          └──────┬──────┘
        │                                        │
        ▼                                        ▼
   I = f(image)                             T = g(text)
   ‖I‖ = 1                                 ‖T‖ = 1
        │                                        │
        └──────────┐            ┌────────────────┘
                   ▼            ▼
              Similarity Matrix S = I · Tᵀ / τ
              (N × N for batch of N pairs)
                        │
                        ▼
              InfoNCE Contrastive Loss
              L = ½ (CE(S, labels) + CE(Sᵀ, labels))
```

**Why Two Encoder Variants?**

| Feature | RN50 (ResNet-50) | ViT-B/32 (Vision Transformer) |
|---------|-------------------|-------------------------------|
| **Architecture** | CNN (convolutional) | Transformer (attention-based) |
| **Embedding Dim** | 1024 | 512 |
| **Params** | ~38M (vision) | ~86M (vision) |
| **Input Processing** | Hierarchical feature extraction | Patch-based (32×32 patches → tokens) |
| **This Project** | Used for LoRA fine-tuning (stable on CPU) | Used for initial experiments |
| **Why RN50 for LoRA** | More stable on CPU/MPS, better LoRA targeting of attention layers | MPS segfault issues, larger model |

### 2.2 Contrastive Learning & InfoNCE Loss

**Core Idea:** Learn representations where matched image-text pairs are close and unmatched pairs are far apart in embedding space.

**InfoNCE Loss (Normalized Temperature-scaled Cross Entropy):**

$$
\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
$$

$$
\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}
$$

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})
$$

Where:
- $\text{sim}(I, T) = \frac{I \cdot T}{\|I\| \|T\|}$ is cosine similarity
- $\tau = 0.07$ is the learnable temperature parameter
- $N$ is the batch size
- Diagonal entries are positive pairs; off-diagonals are negatives

**Implementation from the project:**

```python
# Forward pass: encode both modalities
img_features = peft_model.encode_image(imgs)    # [B, 1024]
txt_features = peft_model.encode_text(txts)      # [B, 1024]

# L2-normalize to unit sphere
img_features = img_features / img_features.norm(dim=-1, keepdim=True)
txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

# Similarity matrix (B×B) scaled by temperature
logits = img_features @ txt_features.T / TAU     # TAU = 0.07

# Labels: diagonal = positive pairs (i-th image matches i-th text)
labels = torch.arange(imgs.size(0), device=DEVICE)

# Symmetric contrastive loss
loss_i = F.cross_entropy(logits, labels)       # image → text direction
loss_t = F.cross_entropy(logits.T, labels)     # text → image direction
loss = (loss_i + loss_t) / 2
```

**Why Temperature τ = 0.07?**
- Lower τ → sharper softmax → harder negatives dominate → more discriminative but risk of training instability
- Higher τ → softer distribution → easier training but less discriminative
- 0.07 is CLIP's original value, well-validated in literature
- Acts as a learned scaling factor that controls the "peakiness" of the similarity distribution

### 2.3 CLIP Fine-Tuning with LoRA

**Why Fine-Tune? The Domain Shift Problem:**

| Aspect | Base CLIP (WebImageText) | Fashion Domain |
|--------|--------------------------|----------------|
| **Training Data** | 400M internet image-text pairs | Specialized fashion catalogs |
| **Vocabulary** | General ("dog", "sunset") | Fashion-specific ("chiffon", "A-line", "pleated") |
| **Visual Focus** | Entire scene, objects | Clothing details, fabric texture, fit |
| **Text Style** | Alt text, captions | Detailed garment descriptions with attributes |
| **Failure Mode** | "red dress" retrieves any red clothing | Needs to distinguish sleeve length, neckline, fabric |

**LoRA (Low-Rank Adaptation) — How It Works:**

Instead of updating all parameters, LoRA freezes the pretrained weights and injects trainable low-rank decomposition matrices into each attention layer:

$$
W' = W_0 + \Delta W = W_0 + B \cdot A
$$

Where:
- $W_0 \in \mathbb{R}^{d \times d}$ — frozen pretrained weights
- $A \in \mathbb{R}^{r \times d}$ — trainable down-projection (r << d)
- $B \in \mathbb{R}^{d \times r}$ — trainable up-projection
- $r = 16$ — LoRA rank (compression factor)

```
Original Layer:        LoRA Augmented Layer:
                       
  x ──→ [W₀] ──→ h      x ──→ [W₀ (frozen)] ──────→ (+) ──→ h
                               │                        ↑
                               └──→ [A] ──→ [B] ──→ ×α/r ┘
                                     r=16     d=1024
                               (trainable low-rank path)
```

**LoRA Configuration in this project:**

```python
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,                    # Rank: controls capacity of adaptation
    lora_alpha=32,           # Scaling factor: effective LR multiplier = alpha/r = 2.0
    lora_dropout=0.1,        # Regularization
    target_modules=[
        "q_proj",            # Query projection in attention
        "k_proj",            # Key projection in attention
        "v_proj",            # Value projection in attention
        "out_proj",          # Output projection in attention
        # Optional MLP layers for deeper adaptation:
        # "c_fc",            # MLP first layer
        # "c_proj",          # MLP output layer
    ],
    bias="none",             # Don't train biases
    inference_mode=False,    # Training mode
)
```

**Parameter Efficiency:**

| Metric | Value |
|--------|-------|
| Total parameters | ~38M (RN50) |
| Trainable (LoRA) parameters | ~36K |
| Parameter efficiency | **0.096%** |
| Saved checkpoint size | ~150KB (vs ~150MB for full model) |

**Training Configuration:**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW | Weight decay + Adam, standard for fine-tuning |
| Learning rate | 1e-4 | Low LR for stable LoRA adaptation |
| Weight decay | 1e-2 | Regularization to prevent overfitting on small dataset |
| LR scheduler | CosineAnnealingLR | Smooth decay, better convergence than step decay |
| Temperature τ | 0.07 | Standard CLIP temperature |
| Epochs | 5 | Sufficient for LoRA convergence without overfitting |
| Batch size | 4 (effective 2 on CPU) | Limited by CPU memory; LoRA trains well with small batches |
| Gradient clipping | max_norm=1.0 | Prevents gradient explosion in contrastive loss |

### 2.4 BLIP / BLIP-2: Caption Generation for Training Data

**What is BLIP?**

BLIP (Bootstrapping Language-Image Pre-training) is Salesforce's vision-language model that excels at image captioning, VQA, and image-text retrieval. It uses a multimodal mixture of encoder-decoder architecture.

```
              BLIP Architecture (Simplified)
 ═══════════════════════════════════════════════════

  Image ──→ [ViT Encoder] ──→ Visual Features
                                    │
                                    ▼
                           ┌────────────────┐
                           │ Image-Grounded │
                           │ Text Decoder   │
                           │ (Cross-Attn    │
                           │  over visual   │
                           │  features)     │
                           └───────┬────────┘
                                   │
                                   ▼
                           Generated Caption:
                    "Short-sleeve T-shirt in grey
                     with round neckline and
                     cotton fabric..."
```

**BLIP-2 Enhancement (Q-Former):**

BLIP-2 introduces a lightweight **Querying Transformer (Q-Former)** that bridges a frozen image encoder and frozen LLM, using learned query tokens to extract visual features:

```
  Image ──→ [Frozen ViT] ──→ Visual Tokens
                                   │
                     ┌─────────────▼──────────────┐
                     │       Q-Former              │
                     │  Learned Query Tokens (32)  │
                     │  Cross-attend to visual     │
                     │  features → compressed      │
                     │  visual representation      │
                     └─────────────┬───────────────┘
                                   │
                                   ▼
                     [Frozen LLM (OPT/FlanT5)]
                                   │
                                   ▼
                            Rich Caption
```

**How BLIP Was Used in This Project:**

1. **Few-shot prompt construction** with 6 hand-written exemplar captions covering jackets, dresses, shirts, pants, tanks
2. **BLIP-base** (`Salesforce/blip-image-captioning-base`) generates fashion-specific captions
3. **Generation parameters:** `max_new_tokens=120`, `temperature=0.7`, `num_beams=5`, `length_penalty=1.3`, `repetition_penalty=1.2`
4. **Quality control:** captions shorter than 15 words are regenerated with higher `length_penalty=1.8`
5. **Post-processing:** fabric/fit terms appended if missing from generated caption
6. **Output:** `captions.jsonl` — one JSON per line with `{path, caption, generated, method}`

**Why BLIP Captions Instead of Dataset Labels?**

| Approach | Pros | Cons |
|----------|------|------|
| Category labels ("dress", "jacket") | Clean, consistent | Too coarse — no details on fabric, color, style |
| Human annotations | Rich, accurate | Expensive, doesn't scale to 44K items |
| **BLIP few-shot captions** | **Rich, scalable, fashion-specific** | **Occasional noise, needs quality control** |
| Generic BLIP (no few-shot) | Scalable | Too generic ("a woman wearing clothes") |

**Few-Shot Example (from actual code):**

> "Long sleeve biker jacket in black with a quilted texture on the shoulders and sleeves. Made of cotton fabric, it features a structured collar with snap button closure and a front zip fastening in metallic gold. Zippered welt pockets at the sides..."

These examples guide BLIP to generate similarly detailed descriptions covering: garment type, sleeve length, color, fabric, texture, closures, fit, styling details.

### 2.5 DensePose: Body-Part Aware Fashion Segmentation

**What is DensePose?**

DensePose (from Meta's Detectron2) establishes dense correspondences between RGB images and the 3D surface of the human body. It maps every pixel to a UV coordinate on a body part surface.

```
          DensePose Pipeline
 ═══════════════════════════════════════

  Input Image          DensePose Output
  ┌──────────┐        ┌──────────────────┐
  │          │        │  Body Part Labels │
  │  Person  │  ──→   │  + UV Coordinates │
  │  wearing │        │                  │
  │  clothes │        │  Head: part 1    │
  │          │        │  Torso: part 2   │
  │          │        │  Arms: parts 3-4 │
  │          │        │  Legs: parts 5-6 │
  └──────────┘        └──────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │ Binary Segm Mask │
                      │ clothing > 0     │
                      │ background = 0   │
                      └──────────────────┘
```

**24 Body Part Labels in DensePose:**

| ID | Body Part | Fashion Relevance |
|----|-----------|-------------------|
| 1 | Torso (front) | Tops, shirts, dresses |
| 2 | Torso (back) | Back design details |
| 3-4 | Right/Left upper arm | Sleeve type (cap, short, long) |
| 5-6 | Right/Left lower arm | Cuff style |
| 7-8 | Right/Left upper leg | Skirt length, shorts |
| 9-10 | Right/Left lower leg | Pants length, boots |
| 11-14 | Right/Left hand/foot | Accessories, shoes |
| 15-24 | Head, neck regions | Neckline, collars |

**How Masks Were Used in This Project:**

```python
def apply_mask(img_path, mask_path):
    """Apply segmentation mask to image — isolate clothing regions."""
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    mask = Image.open(mask_path).convert('L')        # Grayscale mask
    mask = mask.resize(img.size, Image.NEAREST)       # Align dimensions
    mask_np = np.array(mask)

    # Binary mask: clothing pixels have value > 0
    binary_mask = (mask_np > 0).astype(np.uint8)

    # White-out non-clothing pixels (set to 255 = white background)
    img_np[binary_mask == 0] = 255

    masked_img = Image.fromarray(img_np)
    return masked_img
```

**Why Masking Improves Fashion Retrieval:**

```
 Without Mask:                    With Mask:
 ┌──────────────┐                ┌──────────────┐
 │  Background  │                │  ███████████ │  (white)
 │  ┌────────┐  │                │  ┌────────┐  │
 │  │ Person │  │                │  │Clothing│  │
 │  │wearing │  │    ──→         │  │ ONLY   │  │
 │  │clothes │  │                │  │(masked)│  │
 │  └────────┘  │                │  └────────┘  │
 │  Floor/Wall  │                │  ███████████ │  (white)
 └──────────────┘                └──────────────┘

 CLIP sees:                       CLIP sees:
 - Background noise               - Pure clothing features
 - Floor, walls, props             - Fabric, color, pattern
 - Skin, hair, face               - Shape, fit, design details
 - Misleading context              - Clean fashion signal
```

Without masks, CLIP might match images based on similar backgrounds or poses rather than clothing similarity. Masking forces the vision encoder to focus exclusively on garment features.

### 2.6 Adaptive Query Reweighting

The system allows users to dynamically balance the contribution of visual and textual signals at query time:

```
 User Controls:
 α = 0.0 → Pure image search (find visually similar items)
 α = 0.5 → Balanced (default — image style + text refinement)
 α = 1.0 → Pure text search (search by description only)

 Query Embedding Computation:
 ┌─────────────────────────────────────────────────┐
 │                                                 │
 │   q = (1 - α) · encode_image(I) + α · encode_text(T)   │
 │   q = q / ‖q‖                                  │
 │                                                 │
 └─────────────────────────────────────────────────┘
```

**Implementation:**

```python
def encode_query(img_path, text, use_lora=True, alpha=0.5):
    """Blend image and text embeddings with user-controlled alpha."""
    model, preprocess = _get_lora_model() if use_lora else _get_clip()
    
    # Encode image
    img = Image.open(img_path).convert('RGB')
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(image_input)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    if text and text.strip():
        # Process text (handle negative prompts, enhance colors)
        processed_text = process_text_query(text)
        text_tokens = clip.tokenize([processed_text]).to(device)
        with torch.no_grad():
            txt_emb = model.encode_text(text_tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    else:
        txt_emb = torch.zeros_like(img_emb)
        alpha = 0.0  # No text → image only

    # Weighted blend + re-normalize
    query_emb = (1 - alpha) * img_emb + alpha * txt_emb
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    return query_emb.cpu()
```

**Text Query Processing — Negative Prompt Handling:**

Since CLIP struggles with negation ("without jacket" might still retrieve jackets), the system converts negative prompts to positive alternatives:

```python
negative_mappings = {
    'without jacket': 'shirt t-shirt top',
    'no long sleeves': 'short sleeves sleeveless',
    'without pants':  'shorts skirt',
    # ... more mappings
}

color_enhancements = {
    'light color':   'light colored bright pastel',
    'dark color':    'dark colored black navy',
    'bright color':  'bright colored vibrant',
}
```

### 2.7 FAISS Retrieval Pipeline

**FAISS (Facebook AI Similarity Search)** enables sub-millisecond approximate nearest neighbor search over dense embeddings.

**Index Types Implemented:**

| Index | Type | Complexity | Used When |
|-------|------|-----------|-----------|
| `IndexFlatIP` | Exact (brute force) | O(n·d) per query | Small datasets (< 100K), accuracy-critical |
| `IndexIVFFlat` | Approximate (inverted file) | O(n/nlist · d) per query | Medium datasets, speed/accuracy trade-off |
| `IndexHNSWFlat` | Approximate (graph-based) | O(log n · d) per query | Large datasets, high recall needed |

**Search Pipeline:**

```
 Query Image + Text
       │
       ▼
 [LoRA CLIP Encode] ──→ query_emb (1024-dim, float32)
       │
       ▼
 [FAISS Index Search]
   search_k = top_k * 10  (over-retrieve for filtering)
       │
       ▼
 [Self-Image Removal]
   Skip first result (highest similarity = query itself)
       │
       ▼
 [Score Threshold Filter]
   D_filtered = D[D >= min_score]   (default min_score=0.5)
       │
       ▼
 [Top-K Selection]
   Return top_k results with paths and scores
```

**Index Construction Code:**

```python
def build_index(image_dir, mask_dir, index_type="ivf", use_lora=True):
    # Process all 44K images → embeddings
    embs_np = np.vstack(embeddings).astype('float32')  # [44096, 1024]
    
    d = embs_np.shape[1]  # 1024 for RN50
    
    if index_type == "flat":
        ix = faiss.IndexFlatIP(d)                      # Exact search
    elif index_type == "ivf":
        nlist = min(100, max(1, len(paths) // 30))     # ~1400 centroids
        quantizer = faiss.IndexFlatIP(d)
        ix = faiss.IndexIVFFlat(quantizer, d, nlist,
                                faiss.METRIC_INNER_PRODUCT)
        ix.train(embs_np)                              # K-means clustering
    elif index_type == "hnsw":
        ix = faiss.IndexHNSWFlat(d, 32)                # M=32 connections
        ix.hnsw.efConstruction = 200                   # Build-time accuracy
    
    ix.add(embs_np)
    return ix, paths
```

### 2.8 Training Pipeline: End-to-End

```
 ┌────────────────────────────────────────────────────────────────┐
 │              COMPLETE TRAINING PIPELINE                        │
 ├────────────────────────────────────────────────────────────────┤
 │                                                                │
 │  Step 1: Data Preparation                                     │
 │  ┌─────────────────────────────────────────────┐              │
 │  │ DeepFashion images (44,096)                  │              │
 │  │      │                                       │              │
 │  │      ├──→ DensePose → segmentation masks     │              │
 │  │      │    (body part → binary clothing mask)  │              │
 │  │      │                                       │              │
 │  │      └──→ BLIP few-shot → captions.jsonl     │              │
 │  │           {path, caption, method}             │              │
 │  └─────────────────────────────────────────────┘              │
 │                                                                │
 │  Step 2: CLIP LoRA Fine-Tuning                                │
 │  ┌─────────────────────────────────────────────┐              │
 │  │ Load base CLIP (RN50) + apply LoRA adapters  │              │
 │  │ Load FashionCLIPDataset(captions.jsonl)       │              │
 │  │                                              │              │
 │  │ For each epoch (5):                          │              │
 │  │   For each batch (img_tensor, text_tensor):  │              │
 │  │     img_emb = LoRA_CLIP.encode_image(img)    │              │
 │  │     txt_emb = LoRA_CLIP.encode_text(text)    │              │
 │  │     loss = symmetric_contrastive(img, txt)   │              │
 │  │     loss.backward()                          │              │
 │  │     optimizer.step()                         │              │
 │  │   Save best checkpoint                       │              │
 │  └─────────────────────────────────────────────┘              │
 │                                                                │
 │  Step 3: Embedding Construction                               │
 │  ┌─────────────────────────────────────────────┐              │
 │  │ Load LoRA checkpoint (clip_lora_best)         │              │
 │  │ For each image (batch of 1000):              │              │
 │  │   Apply mask (if available)                  │              │
 │  │   img_emb = LoRA_CLIP.encode_image(masked)   │              │
 │  │   L2 normalize                               │              │
 │  │   Save batch embeddings (.npy)               │              │
 │  │ Merge batches → final embeddings [44096,1024]│              │
 │  └─────────────────────────────────────────────┘              │
 │                                                                │
 │  Step 4: FAISS Index Building                                 │
 │  ┌─────────────────────────────────────────────┐              │
 │  │ Load final embeddings → build FAISS index    │              │
 │  │ Save index + paths for runtime use           │              │
 │  └─────────────────────────────────────────────┘              │
 └────────────────────────────────────────────────────────────────┘
```

### 2.9 Evaluation: Mean Opinion Score (MOS)

**What is MOS?**

Mean Opinion Score is a subjective quality metric where human evaluators rate the relevance of recommendations on a numerical scale. It originated in telecommunications (ITU-T P.800) and is widely used when objective metrics alone are insufficient.

**MOS Protocol in This Project:**

```
 For each of N test queries:
 ┌──────────────────────────────────────────────────────────┐
 │  1. System retrieves top-K fashion items for query       │
 │  2. Human evaluator rates each result on 1-5 scale:     │
 │     5 = Excellent match (exact category, color, style)   │
 │     4 = Good match (same category, slight differences)   │
 │     3 = Fair (related but not what was searched for)     │
 │     2 = Poor (wrong category but some similarity)        │
 │     1 = Bad (completely irrelevant)                      │
 │  3. MOS for query = mean(ratings) / 5 × 100            │
 │  4. Overall MOS = mean across all queries               │
 └──────────────────────────────────────────────────────────┘
```

**88% MOS Achievement:**

| Configuration | MOS | Improvement |
|--------------|-----|-------------|
| Base CLIP (no fine-tuning, no masks) | ~68-72% | Baseline |
| Base CLIP + segmentation masks | ~74-78% | +6-8% from masks |
| LoRA fine-tuned CLIP (no masks) | ~80-84% | +12-14% from domain adaptation |
| **LoRA CLIP + masks + adaptive reweighting** | **88%** | **+16-20% full system** |

---

## 3. Key Metrics & Results

### 3.1 Quantitative Results

| Metric | Value | Significance |
|--------|-------|-------------|
| **Mean Opinion Score (MOS)** | **88%** | Strong human-judged relevance; above 80% considered "good" for fashion retrieval |
| Dataset Size | 44,096 items | Covers men's and women's clothing across DeepFashion categories |
| Embedding Dimension | 1024 (RN50) | Rich representation space for nuanced fashion features |
| LoRA Parameter Efficiency | 0.096% | Only ~36K trainable out of ~38M total — minimal compute cost |
| Search Latency | < 1 second | Sub-second retrieval via FAISS inner product search |
| BLIP Caption Coverage | 44K captions | Every image paired with rich fashion-specific description |
| FAISS Index Type | IndexFlatIP | Exact search over 44K vectors — still sub-second at this scale |

### 3.2 Qualitative Improvements

| Query Type | Base CLIP | Fine-Tuned System |
|-----------|-----------|-------------------|
| "red summer dress" | Returns any red clothing | Returns dresses with summer silhouettes in red |
| Image of formal blazer | Matches coats, jackets broadly | Matches blazers specifically with similar lapel/fabric |
| "without jacket" (negative) | Still returns jackets | Converts to "shirt t-shirt top" → correct retrieval |
| Image with busy background | Matches on background similarity | Masked image → matches on clothing features only |

### 3.3 Ablation Insights

| Component | Impact on MOS |
|-----------|---------------|
| Segmentation masks | +6-8% — removes background noise from embeddings |
| BLIP captions (vs. category labels) | +8-10% — richer supervision signal for fine-tuning |
| LoRA fine-tuning (vs. frozen CLIP) | +12-14% — domain-adapted representations |
| Adaptive query reweighting | +2-4% — user can emphasize visual or textual signal |

---

## 4. Topics You Must Know (Study Guide)

### 4.1 CLIP: Contrastive Language-Image Pre-training

**Architecture:**
- Dual-encoder: separate image encoder (ViT or ResNet) and text encoder (Transformer)
- Shared embedding space via contrastive pre-training
- Trained on 400M image-text pairs from the internet (WebImageText dataset)
- Zero-shot transfer: classify images using natural language prompts without task-specific training

**Training:**
- InfoNCE loss (also called NT-Xent loss in SimCLR context)
- Symmetric: image-to-text AND text-to-image directions
- Batch size = 32,768 (massive) — more negatives = better representations
- Mixed-precision training, gradient checkpointing for memory efficiency

**Key Properties:**
- **Zero-shot classification:** "A photo of a {class}" prompt + cosine similarity
- **Linear probing:** Frozen CLIP features + linear classifier matches supervised models
- **Robustness:** Better distribution shift handling than ImageNet-trained models
- **Limitations:** Struggles with negation, counting, spatial relationships, fine-grained attributes

**Fine-Tuning Strategies:**

| Strategy | Trainable Params | Risk | Use Case |
|----------|-----------------|------|----------|
| **Full fine-tuning** | 100% | Catastrophic forgetting | Large domain-specific dataset (>100K) |
| **Linear probing** | <1% (head only) | Limited adaptation | Quick baseline, small dataset |
| **LoRA** | 0.1-1% | Balanced | **This project** — efficient domain adaptation |
| **Prompt tuning** | <0.01% | Minimal adaptation | Zero-shot improvement with context |
| **BitFit** | ~0.1% (biases only) | Limited but stable | Very small dataset |

### 4.2 Contrastive Learning

**Core Concepts:**
- **Positive pairs:** matched image-text, augmented views of same image
- **Negative pairs:** unmatched pairs within the batch (in-batch negatives)
- **Hard negatives:** negatives close to the anchor — most informative for learning
- **Temperature scaling:** controls softmax sharpness in loss function

**Loss Functions Family:**

| Loss | Formula | Used In |
|------|---------|---------|
| **InfoNCE** | $-\log \frac{e^{s_{+}/\tau}}{\sum_j e^{s_j/\tau}}$ | CLIP, this project |
| **NT-Xent** | Same as InfoNCE with symmetric formulation | SimCLR |
| **Triplet** | $\max(0, d_{+} - d_{-} + m)$ | FaceNet |
| **N-pairs** | Generalization of triplet to N negatives | N-pair loss |

**Why Contrastive > Classification for Multimodal:**
1. No fixed label set — open vocabulary
2. Naturally handles multimodal alignment
3. Scales with data (more pairs = better)
4. Learned representations transfer better

### 4.3 Vision Transformers (ViT)

**Architecture:**
1. **Patch Embedding:** Split image into fixed-size patches (e.g., 32×32 for ViT-B/32) → flatten → linear projection to embedding dimension
2. **Position Embedding:** Learnable 1D position encodings added to patch tokens
3. **[CLS] Token:** Prepended classification token; its final representation used as image embedding
4. **Transformer Blocks:** Standard multi-head self-attention + FFN + LayerNorm (pre-norm)

```
 Image (224×224)
    │
    ▼ Split into 7×7 = 49 patches (32×32 each)
    │
    ▼ Linear projection: patch → 768-dim embedding
    │
    ▼ Add [CLS] token + position embeddings
    │  
    ▼ 12× Transformer Blocks:
    │    ├── LayerNorm
    │    ├── Multi-Head Self-Attention (12 heads)
    │    ├── Residual connection
    │    ├── LayerNorm
    │    ├── FFN (768 → 3072 → 768)
    │    └── Residual connection
    │
    ▼ [CLS] token output → projection head → image embedding
```

**ViT vs. CNN (ResNet) Trade-offs:**

| Feature | ViT | ResNet (CNN) |
|---------|-----|--------------|
| Inductive bias | Minimal (global attention) | Strong (local, translational equivariance) |
| Data requirement | Needs large data | Works with less data |
| Computational pattern | Quadratic in patches (O(n²)) | Linear in spatial dims |
| Long-range dependencies | Naturally captured | Needs deep stacking |
| This project | ViT-B/32 explored initially | **RN50 used** (LoRA stability) |

### 4.4 BLIP and BLIP-2

**BLIP Architecture:**
- **MED (Multimodal mixture of Encoder-Decoder):** Three functionalities in one model:
  1. Unimodal encoder (image understanding)
  2. Image-grounded text encoder (image-text matching)
  3. Image-grounded text decoder (caption generation)
- **CapFilt:** Caption-and-filter framework to bootstrap from noisy web data

**BLIP-2 Innovations:**
- **Q-Former:** Lightweight transformer with 32 learnable query tokens that bridge frozen image encoder and frozen LLM
- **Two-stage pre-training:** (1) vision-language representation learning, (2) vision-to-language generative learning
- **Efficiency:** Only trains Q-Former (~188M params) while keeping both image encoder and LLM frozen
- **LLM integration:** Connects to OPT or Flan-T5 for rich text generation

**Why BLIP for This Project:**
1. Generates rich, detailed captions without expensive human annotation
2. Few-shot prompting guides fashion-specific vocabulary
3. Inference-only usage — no BLIP fine-tuning needed
4. Captions serve as high-quality text supervision for CLIP fine-tuning

### 4.5 DensePose and Body Segmentation

**DensePose Architecture:**
- Built on Detectron2 (Mask R-CNN framework)
- Predicts: (1) body part label, (2) U coordinate, (3) V coordinate for each pixel
- **COCO-DensePose** dataset: 50K annotated images with dense body correspondences
- Uses FPN (Feature Pyramid Network) for multi-scale detection

**UV Mapping Explained:**
- Maps each pixel on a person to a coordinate (U, V) on a 3D body model surface
- 24 body parts, each with independent UV coordinate system
- Enables precise garment-body correspondence

**Fashion AI Applications of DensePose:**
1. **Virtual try-on:** Map garment texture onto body surface using UV coordinates
2. **Garment segmentation:** Identify which body parts are covered by clothing
3. **Pose-invariant matching:** Same garment on different poses maps to same UV coordinates
4. **This project:** Binary mask extraction for background removal in embedding computation

### 4.6 FAISS: Deep Dive

**Index Types Reference:**

| Index | Search Type | Build Time | Search Time | Memory | Recall |
|-------|-------------|-----------|-------------|--------|--------|
| **Flat (IP/L2)** | Exact | O(1) | O(nd) | O(nd) | 100% |
| **IVF** | Approximate | O(nd) train | O(nd/nlist) | O(nd) | 95-99% |
| **HNSW** | Approximate | O(n log n) | O(d log n) | O(nd + nm) | 97-99% |
| **PQ** | Approximate | O(nd) train | O(n·m) | O(n·m) ≪ O(nd) | 85-95% |
| **IVF+PQ** | Approximate | O(nd) train | O(nd/(nlist·m)) | O(n·m) | 90-98% |

**Inner Product vs. L2 Distance:**
- For L2-normalized vectors: $\|a-b\|^2 = 2 - 2\langle a,b\rangle$
- So `IndexFlatIP` on normalized vectors is equivalent to cosine similarity
- CLIP embeddings are L2-normalized → inner product = cosine similarity

**GPU Acceleration (not used in this project but important to know):**
```python
# Move index to GPU for 10-100x speedup
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
```

### 4.7 Multimodal Learning

**Cross-Modal Alignment Strategies:**

| Strategy | Description | Example |
|----------|-------------|---------|
| **Dual-encoder** (used here) | Separate encoders + shared space | CLIP, ALIGN |
| **Cross-attention fusion** | One modality attends to another | BLIP, Flamingo |
| **Early fusion** | Concatenate inputs before encoding | VisualBERT |
| **Late fusion** | Separate processing, combine at decision | Ensemble methods |

**Embedding Space Properties:**
- **Alignment:** Matched pairs should be close ($\text{sim}(I_i, T_i) \approx 1$)
- **Uniformity:** Embeddings should be spread across the hypersphere (avoid collapse)
- **Trade-off:** Too much alignment focus → representation collapse; too much uniformity → poor matching

### 4.8 Image Segmentation Types

| Type | Output | Models | Relevance |
|------|--------|--------|-----------|
| **Semantic** | Per-pixel class labels | FCN, DeepLab, UNet | "This pixel is clothing" |
| **Instance** | Per-pixel class + instance ID | Mask R-CNN | "This pixel belongs to shirt #2" |
| **Panoptic** | Semantic + Instance combined | Panoptic FPN | Full scene understanding |
| **DensePose** | Per-pixel body part + UV | Detectron2 + DensePose | **This project** — body-part to UV mapping |

### 4.9 Transfer Learning in Vision

**Why Transfer Learning for Fashion:**
- Fashion datasets are small compared to ImageNet/WebImageText
- Low-level features (edges, textures) transfer universally
- High-level features need domain adaptation (fashion attributes ≠ general objects)

**Transfer Learning Spectrum:**
1. **Feature extraction:** Freeze encoder, train classifier head
2. **Fine-tuning last layers:** Unfreeze top layers + small LR
3. **Full fine-tuning:** Unfreeze all + very small LR
4. **Parameter-efficient (PEFT):** LoRA, adapters, prefix tuning — **this project's approach**

### 4.10 Embedding Space: Alignment & Uniformity

**Alignment Loss:**
$$
\mathcal{L}_{\text{align}} = \mathbb{E}_{(x,y) \sim p_{\text{pos}}} \|f(x) - g(y)\|^2
$$
Minimizes distance between positive pairs.

**Uniformity Loss:**
$$
\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{(x,x') \sim p_{\text{data}}} e^{-2\|f(x) - f(x')\|^2}
$$
Encourages embeddings to spread on the hypersphere.

**InfoNCE implicitly optimizes both:** The numerator promotes alignment; the denominator (sum over negatives) promotes uniformity.

### 4.11 Fashion AI Concepts

**Fashion Attribute Recognition:**
- Color, pattern, fabric, neckline, sleeve length, fit, occasion
- Multi-label classification problem
- Used as metadata for hybrid retrieval (attribute + embedding similarity)

**Virtual Try-On Pipeline:**
1. DensePose body estimation
2. Garment segmentation
3. Geometric warping (TPS, flow-based)
4. Rendering and blending

**Cross-Modal Fashion Retrieval:**
- Text-to-image: "find me a blue denim jacket" → ranked images
- Image-to-image: upload photo → find similar items (this project)
- Image+text-to-image: upload photo + modify by text (this project)

### 4.12 Mean Opinion Score (MOS) Methodology

**Standard MOS Scale (ITU-T P.800):**

| Score | Quality | Description |
|-------|---------|-------------|
| 5 | Excellent | Perfect match |
| 4 | Good | Minor imperfections |
| 3 | Fair | Somewhat relevant |
| 2 | Poor | Barely relevant |
| 1 | Bad | Completely irrelevant |

**Best Practices:**
- Minimum 15-20 evaluators for statistical significance
- Randomize presentation order
- Include anchor/reference items for calibration
- Report confidence intervals (95% CI)
- Account for inter-rater reliability (Krippendorff's alpha or Fleiss' kappa)

**MOS vs. Automated Metrics:**

| Metric | Type | Measures | Limitation |
|--------|------|----------|-----------|
| **MOS** | Human | Perceived relevance | Expensive, subjective, slow |
| Recall@K | Automated | True positives in top K | Needs ground truth labels |
| NDCG | Automated | Ranked relevance | Needs graded relevance labels |
| FID | Automated | Distribution similarity | For generation, not retrieval |
| CLIPScore | Automated | CLIP similarity | Circular if evaluating CLIP-based system |

---

## 5. Interview Questions & Answers (25+)

### Architecture & Design (Q1-Q7)

**Q1: Walk me through this project from start to finish.**

> This is a multimodal fashion recommendation system built during my coursework at UMD. The core problem is bridging the semantic gap between how users describe clothing in natural language and how fashion catalogs store images.
>
> The pipeline has four stages: First, **data preparation** — I used the DeepFashion dataset with 44,000+ images. I ran DensePose to generate body segmentation masks, isolating clothing from backgrounds. Then I used BLIP with few-shot prompting to generate rich fashion captions for each image.
>
> Second, **model fine-tuning** — I applied LoRA adapters (rank 16, alpha 32) to CLIP's RN50 architecture, training with symmetric contrastive loss on the masked image + BLIP caption pairs. This adapted CLIP from general web knowledge to fashion-specific understanding with only 0.096% trainable parameters.
>
> Third, **index construction** — I encoded all 44K masked images through the fine-tuned model and built a FAISS inner product index for sub-second retrieval.
>
> Fourth, **inference** — Users can search by image, text, or both. An adaptive query reweighting mechanism (alpha blending) lets users control the image-vs-text contribution. The system achieved 88% Mean Opinion Score against human relevance judgments.

**Q2: How does CLIP work, and why did you choose it for this project?**

> CLIP uses a dual-encoder architecture — a vision encoder (ResNet or ViT) and a text encoder (Transformer) — trained contrastively on 400 million image-text pairs. The key insight is that it learns a shared embedding space where semantically similar images and text are close together.
>
> I chose CLIP for three reasons: First, it provides **cross-modal alignment** out of the box — I can encode both fashion images and text descriptions into the same space and compare them with cosine similarity. Second, it has **strong zero-shot transfer** — even without fine-tuning, CLIP understands visual concepts like "red dress" or "leather jacket." Third, it's **fine-tunable** — the pretrained representations provide an excellent starting point that I can adapt to the fashion domain with relatively little data.
>
> The alternative was training a retrieval model from scratch, which would require millions of labeled fashion pairs and wouldn't benefit from CLIP's 400M-pair pretraining.

**Q3: Why fine-tune CLIP instead of using it directly?**

> Base CLIP has a significant **domain gap** for fashion. It was trained on general web image-text pairs — alt text like "woman standing in park" — which doesn't capture fashion-specific details. When I tested base CLIP on queries like "chiffon A-line dress with boat neckline," it struggled because these terms appear infrequently in its training data.
>
> Fine-tuning on fashion data addresses three gaps: (1) **vocabulary** — fashion has specialized terms (chiffon, pleated, A-line) that need stronger representations; (2) **visual granularity** — base CLIP treats "red dress" and "red maxi dress" similarly, but in fashion, silhouette matters enormously; (3) **attribute binding** — fine-tuning strengthens the association between specific visual attributes and their textual descriptions.
>
> I used LoRA rather than full fine-tuning to avoid catastrophic forgetting — I wanted to preserve CLIP's general visual understanding while layering on fashion-specific knowledge.

**Q4: Explain LoRA and why you chose rank 16.**

> LoRA — Low-Rank Adaptation — works by freezing all pretrained weights and injecting small trainable matrices into attention layers. For each weight matrix W, instead of updating W directly, you learn two small matrices A and B such that the effective weight becomes W + B·A, where A is d×r and B is r×d with r << d.
>
> I chose rank 16 as a balance between capacity and efficiency. With the 1024-dimensional RN50 attention layers, rank 16 gives each adapted layer 2×16×1024 = 32,768 parameters instead of 1024² = 1,048,576 — a 32x compression. Rank 4-8 is often too constrained for domain adaptation; rank 32-64 approaches full fine-tuning cost. At rank 16 with alpha 32 (effective scaling = 2.0), I achieved strong fashion adaptation while keeping total trainable parameters under 36K — just 0.096% of the model.
>
> I targeted the attention projections (q_proj, k_proj, v_proj, out_proj) because attention layers control what the model "looks at" — adapting these changes the visual features the model prioritizes (e.g., focusing more on fabric texture, color, and silhouette rather than background objects).

**Q5: Why RN50 instead of ViT-B/32?**

> Practical and technical reasons. Practically, ViT-B/32 caused MPS (Metal Performance Shaders) segmentation faults on macOS during LoRA training, while RN50 was stable on CPU — a real engineering constraint I had to work around.
>
> Technically, RN50 has some advantages for this task: its 1024-dimensional embeddings are richer than ViT-B/32's 512 dimensions, giving more capacity for fine-grained fashion attributes. RN50's attention layers (in the attention pooling head) provide clean LoRA injection points. And for our dataset size (~44K), the smaller ViT-B/32 was actually overfitting more during initial experiments.
>
> If I were deploying at scale with GPU infrastructure, I'd likely switch to ViT-L/14 for its stronger base representations, but for this academic project, RN50 with LoRA was the best engineering trade-off.

**Q6: How does the segmentation mask improve retrieval quality?**

> Without masks, CLIP's vision encoder sees the entire image — person, background, floor, furniture. When computing similarity, background elements become confounding features. Two images might be "similar" because they both have a white studio background, not because the clothing is similar.
>
> The binary mask zeros out non-clothing pixels (set to white=255), so the encoder only processes clothing regions. This had a measurable +6-8% impact on MOS. The mask generation uses DensePose body part labels — any pixel with a body part label > 0 is considered "person/clothing," and I keep those while blanking the rest.
>
> The trade-off is that masks can occasionally clip garment edges or miss loose-fitting clothing, but the net effect is strongly positive because it eliminates background noise from the embedding space.

**Q7: Walk me through the FAISS search pipeline.**

> At index time, I encode all 44K masked images through the LoRA-adapted CLIP encoder, L2-normalize the 1024-dim embeddings, and store them in a FAISS `IndexFlatIP` (exact inner product search). For 44K vectors, exact search is still sub-second, so I didn't need approximate indices — though the code supports IVF and HNSW for larger catalogs.
>
> At query time, the user's image and text are independently encoded through the same LoRA CLIP model. The embeddings are blended using the alpha parameter: `query = (1-α)·img_emb + α·txt_emb`, then L2-normalized. FAISS returns the top `k*10` results by inner product similarity (which equals cosine similarity for normalized vectors). I then filter: remove the self-match (first result is typically the query image itself), apply a minimum score threshold (default 0.5), and return the top-k.
>
> The 10x over-retrieval ensures we have enough results after filtering. The minimum score threshold prevents returning irrelevant items when the catalog doesn't have a good match.

### Model & Training (Q8-Q14)

**Q8: How did you generate captions with BLIP?**

> I used `Salesforce/blip-image-captioning-base` with a few-shot prompting strategy. I manually wrote 6 detailed exemplar captions covering different clothing categories — jackets, dresses, shirts, pants, tanks, blouses — each describing garment type, sleeve length, color, fabric, texture, closures, and fit.
>
> These examples were prepended to the prompt: "Generate a detailed fashion description for this image. Use the same level of detail and style as these examples..." This guides BLIP toward fashion-specific vocabulary rather than generic descriptions like "a woman wearing clothes."
>
> Generation parameters were tuned for quality: beam search with 5 beams for consistency, `length_penalty=1.3` to encourage detailed descriptions, `repetition_penalty=1.2` to avoid repetitive phrasing, and `temperature=0.7` for controlled diversity. Captions shorter than 15 words were regenerated with higher length penalty. As a final quality check, I appended fabric/fit terms if the caption didn't mention any.
>
> The whole process generated `captions.jsonl` — one JSON per line with image path, caption text, and metadata.

**Q9: What is contrastive learning and how does it work in your training loop?**

> Contrastive learning teaches models to pull matching pairs closer and push non-matching pairs apart in embedding space. In our case, a batch of N image-text pairs creates an N×N similarity matrix. The diagonal entries are positive pairs (matched image-text), and all off-diagonal entries are negatives.
>
> The loss is symmetric cross-entropy: for each image, we compute softmax over its similarities with all texts in the batch (image→text direction), and for each text, softmax over its similarities with all images (text→image direction). The target is the identity matrix — each image should be most similar to its own caption.
>
> Temperature τ=0.07 scales the logits before softmax, making the distribution sharper. This means even small differences in similarity scores produce large gradients, forcing the model to make fine distinctions — critical for fashion where items can be visually similar but semantically different (e.g., a blazer vs. a sport coat).

**Q10: How did you handle negative prompts like "without jacket"?**

> CLIP fundamentally struggles with negation because contrastive training optimizes for presence, not absence. The word "jacket" in "without jacket" still activates jacket-related features in the text encoder.
>
> My solution was a rule-based text preprocessing step that converts negative prompts to positive alternatives before encoding. "Without jacket" becomes "shirt t-shirt top," "no long sleeves" becomes "short sleeves sleeveless," and so on. This leverages CLIP's strength (matching positive concepts) rather than fighting its weakness (understanding negation).
>
> I also enhanced color descriptions ("light color" → "light colored bright pastel") to give the text encoder more descriptive signal. This preprocessing happens before tokenization, so the LoRA-adapted model receives clean, positive, descriptive text.
>
> Is it perfect? No — it's a curated mapping that handles common cases. A more robust approach would be to train a negation-aware text encoder or use a language model to rewrite queries, but this pragmatic solution worked well for our use case.

**Q11: What was your training data and how did you construct batches?**

> The training data consisted of 44,096 image-caption pairs from DeepFashion. Each pair had: (1) the fashion image (optionally masked using DensePose segmentation), (2) a BLIP-generated caption describing the garment in detail.
>
> For batch construction, images were preprocessed using CLIP's standard preprocessing (resize to 224×224, normalize). Captions were tokenized using CLIP's tokenizer with `truncate=True` to handle long descriptions. The DataLoader used batch size 2 (effective, limited by CPU memory), no shuffling during LoRA training for reproducibility, and `drop_last=True` to avoid incomplete batches that would break the contrastive loss symmetry.
>
> Each batch forms a 2×2 similarity matrix (or 4×4 with batch_size=4). Within-batch negatives are the non-diagonal pairs. With a batch of 2, you get 2 positives and 2 negatives — small, but LoRA's efficiency means it converges with many more gradient steps over small batches. The cosine annealing scheduler adapts the learning rate across epochs*batches total steps.

**Q12: How does the LoRA training ensure gradient flow through the adapted layers?**

> This was actually a tricky engineering challenge. The PEFT library wraps the original modules, but gradient flow can break if the wrapper doesn't properly connect to the computation graph. In the code, I use `torch.enable_grad()` context and add `+ 0.0` to features to force them into the computational graph:
>
> ```python
> with torch.enable_grad():
>     img_features = peft_model.encode_image(imgs)
>     img_features = img_features + 0.0  # Force gradient computation
> ```
>
> If `loss.requires_grad` is False (gradient chain broken), the code falls back to an alternative approach using a dummy input. I also verify gradient norms and clip to max 1.0 to prevent explosion. The key insight is that PEFT's integration with CLIP required explicit gradient verification because CLIP was originally designed for inference, not training — its internal `@torch.no_grad()` decorators had to be bypassed.

**Q13: What is the temperature parameter and what would happen if you changed it?**

> Temperature τ scales the logits in the contrastive loss: `logits = (img_features @ txt_features.T) / τ`. It controls the "peakiness" of the softmax distribution.
>
> At τ=0.07 (our setting): the softmax is very sharp. A similarity difference of 0.1 becomes 0.1/0.07 ≈ 1.43 in logit space — significant. This forces the model to make precise distinctions between similar items.
>
> If τ→0: the softmax approaches argmax — the model only learns from the hardest negatives, which can cause training instability.
>
> If τ→∞: all logits become equal (uniform distribution) — the model can't distinguish between positive and negative pairs.
>
> If τ=1.0: the similarities are used directly, which is too "soft" for the 0-1 range of cosine similarity — most items would have similar softmax scores, providing weak gradient signal.
>
> In CLIP, τ is actually a learnable parameter initialized to 0.07, but in LoRA fine-tuning I kept it fixed to maintain the pretrained model's calibration.

**Q14: How do you handle the cold start / new items problem?**

> When new items are added to the catalog, they need to be embedded and indexed. The system handles this by:
>
> 1. Running the new image through the LoRA-adapted CLIP encoder (optionally with mask)
> 2. L2-normalizing the embedding
> 3. Appending to the NumPy embedding matrix and FAISS index using `ix.add(new_emb)`
>
> For batch updates, the `build_all_embeddings.py` script processes images in batches of 1000, saving intermediate results to avoid losing progress on crashes. This is important for the full 44K rebuild.
>
> For real-time indexing in production, I'd use FAISS's `add_with_ids()` to track item identifiers and support removal, combined with periodic index rebuilding for optimal search quality.

### System Design & Scaling (Q15-Q19)

**Q15: How would you deploy this system at scale (millions of items)?**

> Several architectural changes needed:
>
> 1. **Replace IndexFlatIP with IVF+PQ:** For 10M items, exact search is too slow. IVF (Inverted File) partitions the space into clusters for coarse search, and PQ (Product Quantization) compresses embeddings from 1024×4 bytes = 4KB to ~64 bytes per vector. This gives sub-10ms search with 95%+ recall.
>
> 2. **GPU-accelerated FAISS:** Move the index to GPU using `faiss.index_cpu_to_gpu()` for 10-100x speedup.
>
> 3. **Serve the model with TorchServe or Triton:** The LoRA model inference should run on dedicated GPU instances, not in the application process.
>
> 4. **Async embedding pipeline:** New items go through a message queue (Kafka/SQS) → embedding worker → FAISS index updater.
>
> 5. **Distributed FAISS (Faiss on Spark / custom sharding):** Split the index across multiple nodes for horizontal scaling.
>
> 6. **Cache frequent queries:** Redis cache for popular search embeddings to skip re-encoding.
>
> 7. **CDN for images:** Serve result images through CloudFront/CloudFlare, not from the search service.

**Q16: What are the latency bottlenecks and how would you optimize them?**

> Three main bottlenecks:
>
> 1. **Query encoding (~50-100ms on CPU):** The LoRA CLIP forward pass is the biggest bottleneck. Solution: batch queries, use GPU inference with TorchServe, or use ONNX Runtime for optimized inference. TensorRT could give 5-10x speedup.
>
> 2. **FAISS search (~1-5ms for 44K, but scales with N):** Already fast for our scale. For millions, IVF reduces to ~1ms. HNSW provides ~0.5ms search with higher recall.
>
> 3. **Image loading and preprocessing (~10-20ms):** PIL image loading from disk. Solution: use turbojpeg for faster JPEG decoding, pre-resize images, or use NVIDIA DALI for GPU-accelerated preprocessing.
>
> End-to-end p99 latency target: < 200ms. Current system: ~150ms (CPU). Optimized system with GPU: < 50ms.

**Q17: How would you add real-time personalization?**

> Three approaches, from simple to complex:
>
> 1. **User preference vector:** Average the embeddings of items the user has liked/saved in their wardrobe. At query time, blend this preference vector with the query embedding: `q_final = β·q_query + (1-β)·q_user_pref`. This is essentially collaborative filtering in embedding space.
>
> 2. **Re-ranking with user features:** Retrieve top-100 candidates via FAISS, then re-rank using a lightweight model that considers user history, demographic preferences, purchase history, and seasonal trends.
>
> 3. **Per-user LoRA adaptation:** For power users, fine-tune a small set of LoRA parameters on their interaction history. This is expensive but gives highly personalized results.
>
> The virtual wardrobe feature in the Streamlit app already stores user-liked items in Supabase, which provides the foundation for approach #1.

**Q18: What monitoring would you add for a production system?**

> Five categories:
>
> 1. **Retrieval quality:** Track MOS on a sampled subset of queries (weekly human eval), CLIPScore distribution for all queries (automated), and "no results" rate.
>
> 2. **Latency:** p50, p95, p99 for query encoding, FAISS search, and end-to-end. Alert on p99 > 500ms.
>
> 3. **Model health:** Embedding distribution statistics (mean, variance, cosine similarity distribution). Detect distribution shift that indicates model degradation.
>
> 4. **Index health:** Number of vectors, index freshness (time since last rebuild), orphaned entries (items deleted but still in index).
>
> 5. **User behavior:** Click-through rate on recommendations, scroll depth, time-to-click, "add to wardrobe" conversion rate.

**Q19: How would you handle multiple fashion domains (shoes, jewelry, bags)?**

> Two strategies:
>
> **Strategy 1: Single model, domain-aware prompts.** Keep one LoRA-adapted CLIP model but prepend domain context to text queries: "a shoe that is red leather boots" rather than just "red leather." This leverages CLIP's cross-category understanding.
>
> **Strategy 2: Domain-specific LoRA adapters.** Train separate LoRA adapters for shoes, jewelry, bags, etc. At query time, classify the domain first (using CLIP zero-shot classification), then load the appropriate adapter. LoRA adapters are tiny (~150KB each), so storing multiple adapters is cheap.
>
> I'd start with Strategy 1 (simpler, single model) and move to Strategy 2 if per-domain metrics show degradation. The key advantage of Strategy 2 is that jewelry has very different visual features (metallic textures, gemstones) than clothing (fabric, silhouette), so domain-specific adaptation can capture these differences better.

### Behavioral / Deep Dive (Q20-Q25)

**Q20: What was the hardest technical challenge you faced?**

> **Making LoRA work with CLIP on CPU without segmentation faults.** CLIP's reference implementation uses `torch.no_grad()` extensively for inference optimization, which breaks the gradient computation needed for training. When I first applied PEFT's LoRA wrapper to CLIP, the loss had `requires_grad=False` — no gradients were flowing.
>
> The fix required three things: (1) wrapping forward passes in `torch.enable_grad()`, (2) adding identity operations (`+ 0.0`) to force tensors into the computational graph, and (3) implementing a fallback path with explicit gradient verification. Additionally, ViT-B/32 was causing MPS segfaults, so I switched to RN50 — a pragmatic engineering decision that also gave us richer 1024-dim embeddings.
>
> This taught me that integrating pre-built libraries (PEFT + CLIP) isn't always plug-and-play. Understanding the internal behavior of both libraries was essential to making them work together.

**Q21: What would you do differently if you started over?**

> Five things:
>
> 1. **Use a larger model (ViT-L/14) with GPU training** — RN50 was a stability compromise. With proper GPU infrastructure, I'd use the stronger ViT-L/14 backbone.
>
> 2. **Train with hard negative mining** — our in-batch negatives are random. Mining hard negatives (similar items from different categories, e.g., a blazer vs. a cardigan) would improve fine-grained discrimination.
>
> 3. **Use BLIP-2 instead of BLIP** — BLIP-2's Q-Former generates richer, more structured captions that would provide better training signal.
>
> 4. **Add attribute-based re-ranking** — after FAISS retrieval, re-rank using extracted fashion attributes (color, sleeve length, pattern type) for more controllable results.
>
> 5. **Implement proper evaluation** — I'd set up a proper evaluation framework with Recall@K, NDCG, and MRR in addition to MOS, with held-out test sets and cross-validation.

**Q22: How does DensePose specifically help with fashion recommendation compared to simpler segmentation?**

> Simpler segmentation (e.g., U-Net binary foreground/background) just tells you "person vs. not person." DensePose goes further in three ways:
>
> 1. **Body-part awareness:** DensePose labels 24 body parts, so you can extract "torso-only" masks (for top recommendations), "lower body" masks (for pants/skirts), or "full body" masks (for dresses). This enables body-region-specific retrieval.
>
> 2. **UV mapping:** Each pixel gets 3D surface coordinates, enabling pose-invariant matching — the same t-shirt on a person with arms raised or arms down maps to the same UV coordinates on the torso surface. This eliminates pose bias in embeddings.
>
> 3. **Garment boundary precision:** DensePose's body part boundaries often align well with garment boundaries (where a shirt ends and pants begin), giving cleaner masks than generic person segmentation.
>
> In this project, I used DensePose primarily for mask generation (improvement #1), but the UV mapping capabilities could enable virtual try-on as a future extension.

**Q23: Explain the embedding space geometry — what does it look like for fashion items?**

> After LoRA fine-tuning, the embedding space exhibits clear structure. Imagine a 1024-dimensional hypersphere (since all embeddings are L2-normalized):
>
> - **Category clusters:** Dresses cluster together, jackets cluster together, etc. But within each cluster, there's further structure by attributes.
>
> - **Attribute gradients:** Within the "dress" cluster, there's a continuous gradient from casual sundresses to formal evening gowns. Color forms another axis — red items are closer to each other regardless of category.
>
> - **Cross-modal alignment:** The text embedding for "blue denim jacket" sits near the image embeddings of actual blue denim jackets. This alignment is tighter after fine-tuning than with base CLIP.
>
> - **Interpolation:** The alpha blending (`(1-α)·img + α·text`) works because the space is locally linear. A query at α=0.3 (mostly image, some text) finds items that look like the uploaded image but biased toward the text description — this is geometrically a point on the arc between the image embedding and text embedding.

**Q24: Why inner product similarity in FAISS instead of L2 distance?**

> For L2-normalized vectors (which ours are), inner product and cosine similarity are equivalent: `cos(a,b) = a·b / (||a||·||b||) = a·b` when `||a||=||b||=1`. L2 distance is a monotonic transformation: `||a-b||² = 2 - 2·a·b`.
>
> I chose `IndexFlatIP` over `IndexFlatL2` because: (1) CLIP's original training uses cosine similarity, so inner product preserves the pretrained geometry; (2) inner product search is slightly faster (no subtraction and square root); and (3) the scores are more interpretable — 0.95 means 95% similar, which is easier to threshold than L2 distance values.

**Q25: How did you validate that the system was actually learning fashion-specific features?**

> Three validation approaches:
>
> 1. **Training loss convergence:** The contrastive loss decreased steadily over 5 epochs, and the best checkpoint saved at the lowest loss showed that the model was learning meaningful image-text alignment on fashion data.
>
> 2. **Qualitative inspection:** After fine-tuning, I tested specific queries. "Formal black blazer" correctly retrieved blazers (not just any black clothing). "Casual cotton summer dress" retrieved light, flowy dresses rather than winter formal wear. Base CLIP would often return the right color but wrong garment type.
>
> 3. **MOS evaluation:** The 88% MOS score — up from ~68-72% for base CLIP — confirmed that human evaluators judged the fine-tuned system's recommendations as significantly more relevant. The improvement wasn't just in aggregate; it was consistent across categories (men's, women's) and query types (text-only, image+text).
>
> 4. **Similarity score analysis:** Fine-tuned model produced higher scores for true matches and lower scores for mismatches, indicating better calibration of the embedding space.

---

## 6. Red Flags & How to Handle

| Red Flag | How to Handle |
|----------|---------------|
| **"88% MOS — how many evaluators? What was the protocol?"** | Be honest about the evaluation setup. Explain the MOS methodology, number of evaluators, and acknowledge that a larger-scale evaluation with more evaluators and held-out queries would strengthen the claim. Emphasize that you'd add automated metrics (Recall@K, NDCG) in production. |
| **"Batch size of 2-4 is tiny for contrastive learning. Doesn't that hurt?"** | Acknowledge this limitation. Explain the CPU memory constraint. Note that LoRA's efficiency means it converges with many small-batch steps. In production, you'd use GPU with batch size 128-256 and gradient accumulation. The literature (He et al., MoCo) shows smaller batches can work with momentum encoders or memory banks. |
| **"Why CPU? This feels like an academic compromise."** | Be transparent: ViT caused MPS segfaults, and the available hardware was limited. Explain the engineering trade-off — RN50 on CPU was stable and still achieved strong results. Emphasize that the architecture is hardware-agnostic and would directly benefit from GPU acceleration. |
| **"BLIP captions might be noisy — how do you know they're good?"** | Explain the quality controls: few-shot prompting with hand-written exemplars, minimum length threshold (15 words), re-generation for short captions, and post-processing to ensure fashion vocabulary. Acknowledge that noisy captions are a known issue and propose solutions: BLIP-2 for richer captions, human spot-checking, or a caption quality classifier. |
| **"DensePose masks can clip clothing edges."** | Agree — mask quality isn't perfect. Explain the net positive: background removal outweighs edge artifacts. Propose improvements: use Segment Anything (SAM) for more precise masks, or combine DensePose body part masks with a garment-specific segmentation model. |
| **"No Recall@K or NDCG — only MOS?"** | Acknowledge the gap in automated metrics. Explain that MOS was chosen because fashion retrieval is inherently subjective — "relevant" depends on user intent. But agree that Recall@K and NDCG with ground-truth triplets would provide objective comparisons. Explain how you'd construct such an evaluation set. |
| **"How is this different from just using CLIP zero-shot?"** | Walk through the ablation: base CLIP ~68-72% MOS → masks +6-8% → LoRA fine-tuning +12-14% → adaptive reweighting +2-4% → 88% total. Each component contributes measurably. The LoRA fine-tuning alone accounts for the largest improvement. |
| **"The negative prompt handling seems hacky."** | Agree it's rule-based and limited. Explain it as a pragmatic solution for a known CLIP limitation. Propose better approaches: train a negation-aware encoder, use an LLM to rewrite queries, or implement attribute-based filtering (parse "without jacket" → exclude jacket category). |

---

## 7. Key Takeaways & Talking Points

### What This Project Demonstrates

1. **Multimodal ML engineering** — end-to-end pipeline from raw images to production retrieval, spanning computer vision, NLP, and information retrieval
2. **PEFT / LoRA mastery** — efficient fine-tuning of large pretrained models with 0.096% trainable parameters, including debugging gradient flow in PEFT+CLIP integration
3. **Vision-language understanding** — CLIP architecture, contrastive learning, InfoNCE loss, embedding space geometry, cross-modal alignment
4. **Dense retrieval systems** — FAISS index construction, approximate nearest neighbor search, embedding pipeline design
5. **Data engineering for ML** — BLIP caption generation, DensePose segmentation, batch embedding construction with fault tolerance
6. **Human evaluation methodology** — MOS evaluation design and execution, understanding limitations of automated metrics

### Technical Depth Signals

- **44,096 items indexed** — not a toy project; real-scale fashion catalog
- **LoRA + contrastive learning** — cutting-edge PEFT technique applied to domain adaptation
- **BLIP few-shot captioning** — creative data augmentation strategy for training data generation
- **DensePose integration** — sophisticated body-aware segmentation for clean fashion features
- **FAISS with multiple index types** — understands trade-offs between exact and approximate search
- **Streamlit UI** with image+text search, text-only search, AI stylist, virtual wardrobe — full user-facing system

### One-Liner Pitch

> "I built a multimodal fashion recommendation engine by fine-tuning CLIP with LoRA on BLIP-generated captions and DensePose-masked images from DeepFashion, achieving 88% Mean Opinion Score with FAISS-based sub-second retrieval across 44,000+ items."

### Connection to Other Projects

| Concept | This Project | Also Used In |
|---------|-------------|-------------|
| Embeddings + retrieval | CLIP + FAISS | P07 (FAISS in Medical RAG), P08 (Cohere + cosine similarity) |
| Fine-tuning pretrained models | LoRA on CLIP | P07 (cross-encoder reranking), P05 (BERT classification) |
| Contrastive learning | InfoNCE loss | Foundational to modern representation learning |
| Vision models | RN50, ViT | P06 (if applicable), any computer vision project |
| User-facing ML system | Streamlit + FAISS | P08 (React + FastAPI), P07 (Streamlit + RAG) |
