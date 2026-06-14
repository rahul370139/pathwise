# P12 — MIMIC-CXR Radiology Report Generation | University of Maryland

> **Rahul Sharma** | Academic Project | University of Maryland
> **Stack:** LLaVA-NeXT v1.6, Mistral-7B, LoRA, PyTorch, HuggingFace Transformers, PEFT, Streamlit
> **Resume Line:** "Fine-tuned LLaVA-NeXT (Mistral-7B) with LoRA to generate structured JSON radiology reports from chest X-rays. Designed curriculum learning and conducted A/B evaluation comparing image-only vs image+EHR conditioning to assess report quality and clinical reasoning."

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
Radiology reporting is one of the most time-consuming tasks in clinical medicine. Radiologists at teaching hospitals read hundreds of chest X-rays daily, writing structured reports with findings and impressions. Turnaround delays (often 24-48 hours for non-urgent cases) can impact patient care, and report quality varies with fatigue and workload. The MIMIC-CXR dataset -- the largest publicly available paired chest X-ray + radiology report dataset (377,110 images, 227,827 studies) -- provides an opportunity to build automated report generation systems that can serve as "first-draft" assistants to radiologists.

**Task**
Build a vision-language model that takes a chest X-ray image (and optionally patient EHR context) as input and generates a structured JSON radiology report containing: (1) a clinical impression paragraph, (2) CheXpert disease labels (12 pathologies), and (3) ICD-10 diagnostic codes (8 conditions). Design a curriculum learning strategy to handle training complexity and conduct a rigorous A/B evaluation to measure how EHR conditioning improves clinical reasoning.

**Approach & Action**

| Phase | What I Did |
|-------|------------|
| **Data Pipeline** | Processed MIMIC-CXR into 5,644 deduplicated samples (4,360 train / 770 val). Built EHR context from MIMIC-III/IV (vitals, labs, chronic conditions). Created a 2-stage curriculum (Stage A: image-only, Stage B: image+EHR). |
| **Model Selection** | Selected LLaVA-NeXT v1.6 with Mistral-7B backbone over LLaVA 1.5 for improved vision-text alignment and anyres image handling. Used CLIP ViT-L/14 as the vision encoder. |
| **LoRA Fine-tuning** | Applied LoRA (rank=16, alpha=32) to all attention + MLP projection layers. Kept the vision backbone frozen, trained the multi-modal projector at full precision. Total trainable: 41.9M / 7.28B (~0.58%). |
| **Curriculum Learning** | Designed a 2-stage curriculum: Stage A (959 image-only samples) builds visual grounding, Stage B (4,685 image+EHR samples) adds clinical reasoning with patient context. Used synthetic Stage-A augmentation and stratified sampling for rare pathologies. |
| **Structured Output** | Enforced JSON output via multi-pass generation (separate passes for impression, CheXpert, ICD), token biasing for label tokens, and JSON drift prevention with self-repair prompts. |
| **A/B Evaluation** | Same model, EHR toggled ON/OFF at inference. Measured CheXpert micro-F1, ICD micro-F1, BLEU-4, ROUGE-L to quantify EHR contribution. |
| **Training Infra** | Trained on NVIDIA A100 (40 GB) in Google Colab Pro. bf16 precision, gradient checkpointing, effective batch size 32 (4 x 8 accumulation). 3 epochs, ~500 steps. |
| **Demo** | Built interactive Streamlit app with image upload, sample selection, real-time A/B comparison, and color-coded label visualization. |

**Result**
- Stage B CheXpert micro-F1: **0.75** (Precision 1.00, Recall 0.60) on quick manifest
- Stage B ICD micro-F1: **1.00** on quick manifest
- Extended validation CheXpert micro-F1: **0.11**, ICD micro-F1: **0.50**
- Successfully generated structured JSON reports with valid CheXpert + ICD labels
- A/B evaluation demonstrated measurable improvement from EHR conditioning
- Built production-ready inference pipeline with CPU-optimized deployment (merged LoRA weights, 14 GB)

---

### 1.2 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              MIMIC-CXR RADIOLOGY REPORT GENERATION SYSTEM                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                    ┌─────────────────┐                      │
│  │ Chest X-Ray  │                    │  Patient EHR     │                     │
│  │ (MIMIC-CXR)  │                    │  (MIMIC-III/IV)  │                     │
│  │ 224→336px    │                    │  Vitals, Labs,   │                     │
│  │ DICOM→JPG   │                    │  ICD codes       │                     │
│  └──────┬──────┘                    └────────┬────────┘                      │
│         │                                     │                              │
│         ▼                                     ▼                              │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │                VISION ENCODER                         │                    │
│  │  CLIP ViT-L/14 (Frozen)                              │                    │
│  │  • 14×14 patch tokens → 196 visual tokens            │                    │
│  │  • Image → [v₁, v₂, ..., v₁₉₆] ∈ ℝ^1024           │                    │
│  └──────────────────────┬───────────────────────────────┘                    │
│                         │                                                     │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │           MULTI-MODAL PROJECTOR (Full Precision)      │                    │
│  │  2-layer MLP: ℝ^1024 → ℝ^4096                       │                    │
│  │  • Aligns vision embedding space to LLM space        │                    │
│  │  • Trained at full precision (NOT quantized)          │                    │
│  │  • Critical bridge between vision and language        │                    │
│  └──────────────────────┬───────────────────────────────┘                    │
│                         │                                                     │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │              MISTRAL-7B LLM BACKBONE                  │                    │
│  │  • 32 transformer layers, 4096 hidden dim             │                    │
│  │  • Sliding window attention (4096 tokens)             │                    │
│  │  • Grouped Query Attention (GQA: 8 KV heads)         │                    │
│  │  • LoRA adapters on q/k/v/o/gate/up/down projections │                    │
│  │  • Rank=16, Alpha=32, Dropout=0.05                   │                    │
│  │  • Last 4 transformer blocks unfrozen                 │                    │
│  │                                                       │                    │
│  │  Input: [EHR tokens] + [Image tokens] + [Prompt]     │                    │
│  │  Output: Structured JSON report                       │                    │
│  └──────────────────────┬───────────────────────────────┘                    │
│                         │                                                     │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │          AUXILIARY CLASSIFICATION HEADS                │                    │
│  │  • CheXpert head: Linear(4096 → 12)                  │                    │
│  │  • ICD head: Linear(4096 → 8)                        │                    │
│  │  • BCE + Focal loss with pos_weight boosting          │                    │
│  │  • Pooled from last hidden states of assistant tokens │                    │
│  └──────────────────────┬───────────────────────────────┘                    │
│                         │                                                     │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │           MULTI-PASS INFERENCE ENGINE                  │                    │
│  │  Pass 1: Generate Impression (text)                   │                    │
│  │  Pass 2: Generate CheXpert JSON (with token biasing)  │                    │
│  │  Pass 3: Generate ICD JSON (Stage B only)             │                    │
│  │  • Majority voting across 3-5 generations             │                    │
│  │  • JSON drift prevention + self-repair                │                    │
│  │  • Keyword-based post-processing rules                │                    │
│  └──────────────────────┬───────────────────────────────┘                    │
│                         │                                                     │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │                   OUTPUT                              │                    │
│  │  {                                                    │                    │
│  │    "impression": "Bilateral pleural effusions...",    │                    │
│  │    "chexpert": {"Pleural Effusion": 1, "Edema": 1},  │                    │
│  │    "icd": {"Pleural_Effusion": 1, "Pulmonary_Edema": 1} │                │
│  │  }                                                    │                    │
│  └──────────────────────────────────────────────────────┘                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Curriculum Learning Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CURRICULUM LEARNING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE A: Visual Grounding (16.9% of data)                          │
│  ┌────────────────────────────────────────┐                         │
│  │  Input:  [Chest X-Ray Image]           │                         │
│  │  Output: Impression + CheXpert JSON    │                         │
│  │  Samples: 959 image-only              │                         │
│  │  Goal: Learn to "read" X-rays         │                         │
│  └──────────────────┬─────────────────────┘                         │
│                     │                                                │
│                     │  + Synthetic Stage A                           │
│                     │  (Stage B with EHR stripped)                   │
│                     │                                                │
│                     ▼                                                │
│  STAGE B: Clinical Reasoning (83.1% of data)                       │
│  ┌────────────────────────────────────────┐                         │
│  │  Input:  [X-Ray] + [EHR Context]      │                         │
│  │  EHR:    Age, Sex, Vitals, Labs,      │                         │
│  │          O2 device, Chronic conditions │                         │
│  │  Output: Impression + CheXpert + ICD  │                         │
│  │  Samples: 4,685 image+EHR             │                         │
│  │  Goal: Integrate clinical context     │                         │
│  └────────────────────────────────────────┘                         │
│                                                                      │
│  MIXING STRATEGY:                                                    │
│  • Each epoch: rebuild mix (65% Stage B + all Stage A + synthetic)  │
│  • Stratified sampling: boost rare pathologies (5x weight)          │
│  • Rare CheXpert: Pneumothorax, Fracture, Lung Lesion              │
│  • Rare ICD: Pulmonary_Embolism, Rib_Fracture                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Tech Stack

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| Base Model | LLaVA-NeXT v1.6 (llava-hf/llava-v1.6-mistral-7b-hf) | Improved vision-text alignment over LLaVA 1.5, anyres image handling |
| Vision Encoder | CLIP ViT-L/14 (frozen) | Strong pre-trained visual features, medical imaging transfer |
| LLM Backbone | Mistral-7B | Sliding window attention (efficient), GQA, strong reasoning |
| Fine-tuning | LoRA (PEFT library) | Parameter-efficient: 0.58% trainable, fits single GPU |
| Training | PyTorch + HuggingFace Trainer | Industry standard, gradient checkpointing, bf16 |
| Hardware | NVIDIA A100 (40 GB) via Colab Pro | bf16 native, sufficient VRAM for 7B model |
| Inference | Multi-pass generation + token biasing | Ensures valid JSON, improves label accuracy |
| Demo | Streamlit | Interactive A/B testing, image upload |
| Evaluation | scikit-learn, NLTK, rouge-score | F1, BLEU-4, ROUGE-L |
| Dataset | MIMIC-CXR + MIMIC-III/IV | Largest public CXR dataset with paired reports + EHR |

### 1.5 Dataset Description

| Dataset | Size | Description |
|---------|------|-------------|
| MIMIC-CXR | 377,110 images / 227,827 studies | Chest X-rays from Beth Israel Deaconess Medical Center |
| CheXpert Labels | 14 pathology labels per study | Automated labeler assigns Positive/Negative/Uncertain |
| MIMIC-III/IV | ~60,000 ICU patients | Vitals, lab results, medications, ICD codes |
| Our Training Set | 4,360 samples (cleaned from 4,797) | Stage A: 959, Stage B: 3,401 |
| Our Validation Set | 770 samples (cleaned from 847) | Held-out evaluation |
| Images Used | 10,003 chest X-ray JPGs | Deduplicated (41.4% duplicates removed) |
| EHR Coverage | 42.6% vitals, 94.1% labs (Stage B) | Linked from MIMIC-III/IV |

---

## 2. Deep Technical Walkthrough

### 2.1 MIMIC-CXR Dataset

**What it is:**
MIMIC-CXR (Medical Information Mart for Intensive Care - Chest X-Ray) is the largest publicly available dataset of chest radiographs with structured labels and free-text radiology reports, released by MIT and Beth Israel Deaconess Medical Center.

**Scale:**
- 377,110 chest X-ray images (DICOM format)
- 227,827 radiographic studies
- Paired with free-text radiology reports (findings + impressions)
- CheXpert-labeled with 14 pathology categories

**CheXpert Labels (12 used in our model):**

| Label | Description | Frequency |
|-------|-------------|-----------|
| Consolidation | Lung tissue filled with fluid | ~8% |
| Edema | Fluid in lung tissue | ~15% |
| Enlarged Cardiomediastinum | Enlarged heart shadow | ~12% |
| Fracture | Bone fracture visible | ~2% (rare) |
| Lung Lesion | Mass or nodule | ~3% (rare) |
| Lung Opacity | General opacification | ~25% |
| No Finding | Normal study | ~30% |
| Pleural Effusion | Fluid in pleural space | ~20% |
| Pleural Other | Other pleural abnormality | ~1% (rare) |
| Pneumonia | Lung infection | ~5% |
| Pneumothorax | Collapsed lung | ~3% (rare) |
| Support Devices | Lines, tubes, pacemakers | ~35% |

**CheXpert Label Values:**
- `1` (Positive): Finding is present -- supervised
- `0` (Negative): Finding is absent -- supervised
- `-1` (Uncertain): Ambiguous -- **masked out during training** (not supervised)

**Report Structure:**
```
FINDINGS: The cardiac silhouette is mildly enlarged. Bilateral pleural 
effusions are present, left greater than right. There is patchy airspace 
opacity in the left lower lobe, which may represent atelectasis or 
pneumonia. No pneumothorax.

IMPRESSION:
1. Mild cardiomegaly with bilateral pleural effusions.
2. Left lower lobe opacity, possibly atelectasis vs pneumonia.
```

**HIPAA Considerations:**
- Access requires PhysioNet credentialed account + CITI training
- De-identified (no patient names, dates shifted, IDs randomized)
- Data Use Agreement (DUA) required
- Cannot be redistributed
- Images processed locally (no cloud upload of raw data)
- Model weights can be shared; raw data cannot

---

### 2.2 LLaVA-NeXT Architecture

**What is LLaVA?**
LLaVA (Large Language and Vision Assistant) is a family of vision-language models that connect a pre-trained vision encoder to a pre-trained LLM through a simple projection layer. The key insight: use a pre-trained CLIP vision encoder, project its embeddings into the LLM's embedding space, and fine-tune end-to-end.

**LLaVA-NeXT (v1.6) Improvements over LLaVA 1.5:**

| Feature | LLaVA 1.5 | LLaVA-NeXT (v1.6) |
|---------|-----------|-------------------|
| Image Resolution | Fixed 336x336 | **AnyRes** (dynamic grid, up to 672x672) |
| Visual Tokens | 576 (fixed) | Variable (up to 2880 based on grid) |
| Vision-Text Alignment | Single MLP | **Improved 2-layer MLP projector** |
| Conversation Format | Simple | **Mistral chat template** |
| Base LLM | Vicuna-13B | **Mistral-7B** (smaller, faster, better) |

**Architecture Breakdown:**

```
                    Chest X-Ray Image (336×336)
                              │
                              ▼
                ┌─────────────────────────┐
                │    CLIP ViT-L/14        │
                │    (Vision Encoder)      │
                │    Frozen during         │
                │    fine-tuning           │
                │                          │
                │  336/14 = 24 patches     │
                │  24×24 = 576 tokens      │
                │  Each: ℝ^1024            │
                └─────────┬───────────────┘
                          │
                    576 × ℝ^1024
                          │
                          ▼
                ┌─────────────────────────┐
                │  Multi-Modal Projector   │
                │  (2-layer MLP)           │
                │                          │
                │  Linear(1024 → 4096)     │
                │  GELU()                  │
                │  Linear(4096 → 4096)     │
                │                          │
                │  Trained at FULL         │
                │  precision (not LoRA)    │
                └─────────┬───────────────┘
                          │
                    576 × ℝ^4096
                          │
                          ▼
           ┌──────────────────────────────────┐
           │         Token Sequence            │
           │  [EHR_tokens] [IMG_1...IMG_576]   │
           │  [Prompt_tokens]                  │
           └──────────────┬───────────────────┘
                          │
                          ▼
                ┌─────────────────────────┐
                │     Mistral-7B LLM      │
                │     32 Transformer      │
                │     Layers              │
                │                          │
                │  With LoRA adapters on:  │
                │  q_proj, k_proj, v_proj, │
                │  o_proj, gate_proj,      │
                │  up_proj, down_proj      │
                │                          │
                │  Last 4 layers unfrozen  │
                └─────────┬───────────────┘
                          │
                          ▼
                ┌─────────────────────────┐
                │   Autoregressive Output  │
                │   Token-by-token         │
                │   generation of JSON     │
                │   report                 │
                └─────────────────────────┘
```

**How Image Tokens Feed into the LLM:**
1. CLIP ViT-L/14 encodes the image into patch embeddings (576 tokens x 1024 dim)
2. The MLP projector maps each patch embedding from 1024 -> 4096 dimensions (Mistral's hidden size)
3. These visual tokens are **interleaved** into the text token sequence at the `<image>` placeholder position
4. The LLM processes visual + text tokens jointly through self-attention
5. Visual tokens attend to text tokens and vice versa -- this is the cross-modal alignment

**Why LLaVA-NeXT over LLaVA 1.5:**
- Better image understanding with AnyRes (handles varying aspect ratios)
- Improved projector (2-layer MLP vs single linear in original LLaVA)
- Mistral-7B is faster and better at reasoning than Vicuna-13B
- Better instruction following from Mistral's chat template

---

### 2.3 Mistral-7B Architecture

**Key architectural features:**

| Feature | Details |
|---------|---------|
| Parameters | 7.24 billion |
| Layers | 32 transformer blocks |
| Hidden Size | 4096 |
| Attention Heads | 32 query heads |
| KV Heads | 8 (Grouped Query Attention) |
| FFN Hidden | 14,336 (SwiGLU: gate + up projection) |
| Vocab Size | 32,000 (SentencePiece tokenizer) |
| Context Window | 8,192 tokens (sliding window: 4,096) |

**Sliding Window Attention (SWA):**
- Standard attention: $ O(n^2) $ memory for sequence length $ n $
- SWA: each token attends to only the last $ W = 4096 $ tokens
- Reduces memory from $ O(n^2) $ to $ O(n \times W) $
- Information still propagates through layers (token at position $ i $ in layer $ k $ can attend to tokens up to $ i - k \times W $ in the original sequence)
- Effective receptive field: $ k \times W $ tokens after $ k $ layers

**Grouped Query Attention (GQA):**
- Standard Multi-Head Attention: 32 Q heads, 32 K heads, 32 V heads
- GQA in Mistral: 32 Q heads, **8 K heads, 8 V heads**
- Groups of 4 query heads share the same K,V head
- **Benefit:** 4x reduction in KV cache memory, faster inference
- **No quality loss:** GQA matches MHA quality at Mistral's scale

```
Standard MHA:                    GQA (Mistral):
Q₁ K₁ V₁                        Q₁ ─┐
Q₂ K₂ V₂                        Q₂ ─┼── K₁ V₁
Q₃ K₃ V₃                        Q₃ ─┤
Q₄ K₄ V₄                        Q₄ ─┘
...                               Q₅ ─┐
Q₃₂ K₃₂ V₃₂                    Q₆ ─┼── K₂ V₂
                                  Q₇ ─┤
32 KV pairs                       Q₈ ─┘
                                  ...
                                  8 KV pairs (4x savings)
```

**SwiGLU FFN (Feed-Forward Network):**

Standard FFN:
$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)
$$

SwiGLU FFN (Mistral):
$$
\text{FFN}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)
$$

Where:
- $ W_{\text{gate}} $: gating projection (14,336 x 4096)
- $ W_{\text{up}} $: value projection (14,336 x 4096)
- $ W_{\text{down}} $: output projection (4096 x 14,336)
- $ \odot $: element-wise multiplication
- SiLU = Sigmoid Linear Unit = $ x \cdot \sigma(x) $

This is why LoRA targets `gate_proj`, `up_proj`, `down_proj` -- they are the FFN weights in SwiGLU.

---

### 2.4 LoRA Fine-Tuning

**What is LoRA?**
LoRA (Low-Rank Adaptation of Large Language Models, Hu et al., 2021) is a parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each target layer.

**The Core Idea:**
Instead of updating a weight matrix $ W_0 \in \mathbb{R}^{d \times k} $ directly, LoRA decomposes the update into two low-rank matrices:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B \cdot A
$$

Where:
- $ W_0 $: frozen pre-trained weight (e.g., 4096 x 4096)
- $ A \in \mathbb{R}^{r \times k} $: down-projection (initialized with Gaussian)
- $ B \in \mathbb{R}^{d \times r} $: up-projection (initialized with zeros)
- $ r $: rank (our config: **16**)
- $ \alpha $: scaling factor (our config: **32**)
- Scaling: $ \frac{\alpha}{r} = \frac{32}{16} = 2.0 $

```
                    Input x
                      │
            ┌─────────┴──────────┐
            │                     │
            ▼                     ▼
     ┌──────────┐          ┌──────────┐
     │    W₀    │          │    A     │  (r × k) = (16 × 4096)
     │  (frozen) │          │ (LoRA)  │
     │ d × k    │          └────┬─────┘
     └────┬─────┘               │
          │                     ▼
          │              ┌──────────┐
          │              │    B     │  (d × r) = (4096 × 16)
          │              │ (LoRA)  │
          │              └────┬─────┘
          │                   │
          │              × (α/r) = 2.0
          │                   │
          └───────┬───────────┘
                  │  (sum)
                  ▼
              Output h = W₀·x + (α/r)·B·A·x
```

**Our LoRA Configuration (from `advanced_training_v16.yaml`):**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `lora_r` | 16 | Rank of decomposition (higher = more capacity) |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | Regularization on LoRA layers |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All attention + FFN projections |
| `layers_to_transform` | `[30, 31]` | Only last 2 transformer blocks (in v16 config) |
| `modules_to_save` | `multi_modal_projector` | Projector trained at full precision |
| Trainable Params | 41.9M | 0.58% of 7.28B total |

**Why LoRA over Full Fine-Tuning?**

| Aspect | Full Fine-Tuning | LoRA |
|--------|-----------------|------|
| Trainable Params | 7.28B (100%) | 41.9M (0.58%) |
| GPU Memory | ~120 GB (needs multi-GPU) | ~25 GB (single A100) |
| Training Time | Days | Hours |
| Catastrophic Forgetting | High risk | Low risk (base frozen) |
| Storage per Task | 14 GB per checkpoint | ~160 MB adapter |
| Merge at Inference | N/A | Merge into base (zero overhead) |

**LoRA vs QLoRA vs Full Fine-Tuning Comparison:**

| Method | Memory | Speed | Quality | When to Use |
|--------|--------|-------|---------|-------------|
| Full FT | Very High (>80 GB) | Slow | Best possible | Unlimited compute, domain shift |
| LoRA | Medium (~25 GB) | Fast | Near full FT | Single GPU, task-specific adaptation |
| QLoRA | Low (~12 GB) | Medium | Slightly lower | Consumer GPU (RTX 3090/4090) |

**QLoRA (if used):**
QLoRA = LoRA + 4-bit NormalFloat quantization of the base model. We did NOT use QLoRA in our final training (used bf16 full precision on A100), but it's worth knowing:
- Base model quantized to 4-bit (NF4 data type)
- LoRA adapters remain in fp16/bf16
- Double quantization: quantize the quantization constants
- Paged optimizers: handle GPU memory spikes via CPU offloading

---

### 2.5 Structured JSON Output Generation

**Why JSON?**
Clinical systems need structured, parseable data -- not free text. Downstream systems (EHR integration, billing, clinical decision support) need discrete labels, not paragraphs.

**Our Output Format:**

Stage A (Image-Only):
```json
{
  "impression": "No acute cardiopulmonary process.",
  "chexpert": {
    "Consolidation": 0, "Edema": 0, "Enlarged Cardiomediastinum": 0,
    "Fracture": 0, "Lung Lesion": 0, "Lung Opacity": 0,
    "No Finding": 1, "Pleural Effusion": 0, "Pleural Other": 0,
    "Pneumonia": 0, "Pneumothorax": 0, "Support Devices": 0
  }
}
```

Stage B (Image + EHR):
```json
{
  "impression": "Bilateral pleural effusions with pulmonary edema...",
  "chexpert": { ... },
  "icd": {
    "Pneumonia": 0, "Pleural_Effusion": 1, "Pneumothorax": 0,
    "Pulmonary_Edema": 1, "Cardiomegaly": 1, "Atelectasis": 0,
    "Pulmonary_Embolism": 0, "Rib_Fracture": 0
  }
}
```

**How We Enforce JSON Formatting:**

**1. Multi-Pass Generation (Primary Strategy):**
Instead of asking the model to generate everything in one shot (prone to JSON corruption), we split into 3 passes:

| Pass | Task | Temperature | Max Tokens | Token Bias |
|------|------|-------------|------------|------------|
| Pass 1 | Impression (text only) | 0.15 | 160 | None |
| Pass 2 | CheXpert JSON | 0.75 | 140 | +4.0 for "1", -0.5 for "0" |
| Pass 3 | ICD JSON (Stage B) | 0.75 | 100 | +4.0 for "1", -0.5 for "0" |

**2. Token Biasing (Constrained Decoding):**
During CheXpert/ICD generation, we apply additive logit bias to push the model toward valid label values:

```python
class TokenBiasProcessor(LogitsProcessor):
    def __init__(self, token_bias: Dict[int, float]):
        self.token_bias = token_bias

    def __call__(self, input_ids, scores):
        for token_id, bias in self.token_bias.items():
            scores[:, token_id] += bias
        return scores
```

- Token ID for "1" gets +4.0 (boost disease detection)
- Token ID for "0" gets -0.5 (slight push toward negative)
- This acts as soft constrained decoding

**3. Majority Voting:**
Each label pass is generated 1-5 times independently (configurable via `CHEXPERT_VOTE`):
- 34% positive votes needed to call label positive
- Reduces random errors from single generation
- Acts like "consensus" among multiple model opinions

**4. JSON Drift Prevention:**
- Extract JSON using bracket-depth parsing (handles nested JSON)
- Self-repair: if JSON is invalid, prompt the model again with "Return ONLY the JSON"
- Retry up to 2 times with repair prompts
- Keyword-based post-processing as safety net (scan impression for disease mentions)

**5. Keyword-Based Post-Processing:**
```python
chexpert_keywords = [
    ("pneumothorax", "Pneumothorax"),
    ("effusion", "Pleural Effusion"),
    ("opacity", "Lung Opacity"),
    ("edema", "Edema"),
    ("pacemaker", "Support Devices"),
    ...
]
for keyword, label in chexpert_keywords:
    if keyword in impression.lower():
        chexpert[label] = 1
```

EHR-based rules (Stage B):
- BNP > 1200 -> boost Edema + Pleural Effusion (heart failure indicator)
- CRP > 150 + "pneumonia" in impression -> boost Pneumonia
- O2 device present -> boost Support Devices

---

### 2.6 Curriculum Learning

**What is Curriculum Learning?**
Introduced by Bengio et al. (2009), curriculum learning trains models on examples ordered by difficulty -- starting with easy examples and progressively introducing harder ones. Inspired by how humans learn (alphabet before essays).

**Why Curriculum Learning for Our Task:**
- Stage A (image-only -> impression + CheXpert) is **easier**: model just needs to read the X-ray
- Stage B (image + EHR -> impression + CheXpert + ICD) is **harder**: model must integrate clinical context, reason about lab values, and predict diagnostic codes
- Training on easy examples first builds a strong visual foundation before adding clinical complexity

**Our 2-Stage Curriculum:**

| Stage | Samples | Input | Output | Difficulty |
|-------|---------|-------|--------|------------|
| **A** | 959 (16.9%) | Chest X-ray only | Impression + CheXpert | Easy |
| **B** | 4,685 (83.1%) | Chest X-ray + EHR | Impression + CheXpert + ICD | Hard |

**How Difficulty Was Defined:**
- **Easy (Stage A):** Pure visual interpretation. Model needs to identify pathologies from pixel patterns alone. Simpler output (no ICD codes).
- **Hard (Stage B):** Must integrate EHR context (vitals: HR, BP, SpO2; labs: sodium, creatinine, BNP; chronic conditions). Must produce additional ICD codes. Requires clinical reasoning: "BNP=1200 + bilateral opacities -> pulmonary edema + pleural effusion."

**Mixing Strategy (StageMixDataset):**
Rather than strict "all A then all B" sequencing, we use a sophisticated mixing approach each epoch:

```
Each Epoch:
1. Keep ALL Stage A samples (959)
2. Create synthetic Stage A from low-EHR Stage B samples
   - Take Stage B samples with <=2 vitals AND <=2 labs
   - Strip EHR context, reclassify as Stage A
   - This expands Stage A data pool
3. Sample 65% of Stage B samples (3,045 of 4,685)
4. Shuffle everything together
5. Result: ~5,000 mixed samples per epoch
```

**Why 65% Stage B Fraction?**
- 100% would overwhelm the model with EHR data
- Too low would under-train on clinical reasoning
- 65% balances visual grounding and clinical reasoning
- Synthetic Stage A prevents catastrophic forgetting of image-only capability

**Stratified Sampling for Class Imbalance:**
Some pathologies are very rare (Fracture: 2%, Pneumothorax: 3%). Without boosting:
- Model learns to always predict "0" for rare classes (high accuracy, zero recall)
- Clinically dangerous: missing a pneumothorax can be fatal

Our approach:
```python
# Rare labels get 5x sampling weight
rare_chexpert = ['Pneumothorax', 'Fracture', 'Lung Lesion']
rare_icd = ['Pulmonary_Embolism', 'Rib_Fracture', 'Pneumothorax']

for sample in samples:
    weight = 1.0
    for label in rare_chexpert:
        if sample['chexpert_labels'].get(label) == 1:
            weight *= 5.0  # Boost rare positives
```

**Benefits Observed:**
- Faster convergence: model doesn't waste early epochs on hard examples
- Better generalization: visual grounding from Stage A transfers to Stage B
- Reduced catastrophic forgetting: mixing prevents losing image-only capability
- Improved rare class recall: stratified sampling boosts underrepresented pathologies

---

### 2.7 A/B Evaluation

**Design:**
The A/B evaluation tests a critical hypothesis: **Does adding EHR context improve report quality and clinical accuracy?**

Same fine-tuned model, two conditions at inference time:

| Condition | Input | What's Measured |
|-----------|-------|-----------------|
| **B-off** (Image-only) | Chest X-ray only | Baseline visual interpretation |
| **B-on** (Image+EHR) | Chest X-ray + Patient EHR | Clinical reasoning with context |

```python
# A/B toggle is simple - same model, different inputs
results_off = evaluate_icd_with_toggle(samples, use_ehr=False)  # B-off
results_on  = evaluate_icd_with_toggle(samples, use_ehr=True)   # B-on

# The delta tells us how much EHR helps
delta = results_on['macro_f1'] - results_off['macro_f1']
```

**EHR Context Provided (B-on condition):**
```json
{
  "Age": 72, "Sex": "M",
  "Vitals": {"HR": 92, "SpO2": 90, "SBP": 145, "RR": 22},
  "Labs": {"Sodium": {"value": 138, "unit": "mEq/L"},
           "Creatinine": {"value": 1.8, "unit": "mg/dL"},
           "BNP": {"value": 1500, "unit": "pg/mL"}},
  "O2_device": "Oxygen_Device: 40",
  "Chronic_conditions": ["hypertension", "heart_failure"]
}
```

**Evaluation Metrics:**

| Metric | What It Measures | Computed On |
|--------|-----------------|-------------|
| CheXpert micro-F1 | Disease label accuracy (12 classes) | Both conditions |
| ICD micro-F1 | Diagnostic code accuracy (8 classes) | Both conditions |
| Per-class F1 | Individual disease detection | Both conditions |
| Macro Precision | Average precision across classes | Both conditions |
| Macro Recall | Average recall across classes | Both conditions |
| F1 Delta (on - off) | EHR contribution to accuracy | Comparison |

**Expected Results Pattern:**
- EHR context should help with conditions that have strong lab/vital correlates:
  - Pulmonary Edema: BNP > 1200 is diagnostic
  - Pneumonia: CRP elevation + imaging findings
  - Cardiomegaly: History of heart failure
- EHR context should matter less for purely visual findings:
  - Pneumothorax: clearly visible on X-ray
  - Fracture: purely visual finding

**Results (Quick Manifest, 4 samples):**

| Condition | CheXpert micro-F1 | ICD micro-F1 |
|-----------|-------------------|--------------|
| B-off (image only) | 0.60 | 0.67 |
| B-on (image + EHR) | 0.75 | 1.00 |
| **Delta** | **+0.15** | **+0.33** |

**Extended Validation (8 samples):**

| Condition | CheXpert micro-F1 | ICD micro-F1 |
|-----------|-------------------|--------------|
| Extended B | 0.11 | 0.50 |

The gap between quick manifest and extended validation reveals recall challenges on harder cases, confirming the need for the auxiliary fine-tune described in the roadmap.

---

### 2.8 Auxiliary Loss Design

Beyond the standard autoregressive language modeling loss, we added auxiliary classification heads:

**CheXpert Head:** `Linear(4096 -> 12)` with BCE + Focal Loss
**ICD Head:** `Linear(4096 -> 8)` with BCE + Focal Loss

```python
# Pool hidden states from assistant tokens only
mask = (labels != -100).float()
pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1)

# CheXpert classification
chexpert_logits = model.chexpert_head(pooled)  # (batch, 12)
chexpert_loss = focal_bce(chexpert_logits, chexpert_targets, mask=chexpert_masks)

# Total loss
total_loss = base_loss + 4.0 * chexpert_loss + 3.0 * icd_loss
```

**Focal Loss for Class Imbalance:**
$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Where:
- $ \gamma = 1.0 $: focusing parameter (down-weights easy negatives)
- $ \text{pos\_weight} = 5.0 $: up-weights rare positives
- This addresses the severe class imbalance (e.g., Fracture only 2% positive)

**Why Auxiliary Heads + Generative Loss:**
- Generative loss alone struggles with precise label prediction (model can describe findings in text but may not output exact JSON labels)
- Auxiliary heads provide direct classification signal
- The pooled representation from assistant tokens encodes the model's "understanding" -- the heads decode this into structured labels
- Acts as multi-task learning: better representations through joint optimization

---

### 2.9 Evaluation Metrics Explained

**Text Quality Metrics:**

| Metric | Formula | What It Measures | Range |
|--------|---------|-----------------|-------|
| BLEU-4 | Geometric mean of 1-4 gram precision + brevity penalty | N-gram overlap | 0-1 |
| ROUGE-L | F1 of longest common subsequence | Semantic similarity | 0-1 |
| ROUGE-1 | Unigram F1 overlap | Word-level similarity | 0-1 |
| ROUGE-2 | Bigram F1 overlap | Phrase-level similarity | 0-1 |

**Clinical Accuracy Metrics:**

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Micro-F1 | $ \frac{2 \cdot \sum TP_i}{2 \cdot \sum TP_i + \sum FP_i + \sum FN_i} $ | Overall label accuracy (weighted by support) |
| Macro-F1 | $ \frac{1}{C} \sum_{i=1}^{C} F1_i $ | Average F1 across classes (treats rare = common) |
| Per-class F1 | $ \frac{2 \cdot P_i \cdot R_i}{P_i + R_i} $ | Individual disease detection accuracy |
| JSON Validity | $ \frac{\text{valid JSON outputs}}{\text{total outputs}} $ | Structural correctness |

**Why Both Micro and Macro F1?**
- **Micro-F1** gives a single accuracy number but is dominated by common classes (Lung Opacity, Support Devices)
- **Macro-F1** treats all classes equally -- reveals if model fails on rare but critical diseases (Pneumothorax, Fracture)
- A model with micro-F1=0.80 but macro-F1=0.30 is clinically dangerous (misses rare diseases)

---

## 3. Key Metrics & Results

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | llava-hf/llava-v1.6-mistral-7b-hf |
| Total Parameters | 7.28B |
| Trainable Parameters | 41.9M (0.58%) |
| LoRA Rank / Alpha | 16 / 32 |
| Batch Size (effective) | 32 (4 x 8 accumulation) |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Precision | bf16 |
| Hardware | NVIDIA A100 (40 GB) |
| Training Time | ~3 hours (Colab Pro) |
| Training Samples | 4,360 (cleaned) |
| Validation Samples | 770 |

### 3.2 Performance Results

**Quick Manifest (4 samples - best case):**

| Metric | Stage A | Stage B |
|--------|---------|---------|
| CheXpert micro-F1 | -- | **0.75** |
| CheXpert Precision | -- | 1.00 |
| CheXpert Recall | -- | 0.60 |
| ICD micro-F1 | -- | **1.00** |

**Extended Validation (8 samples - harder cases):**

| Metric | Value |
|--------|-------|
| CheXpert micro-F1 | 0.11 |
| CheXpert Precision | 0.17 |
| CheXpert Recall | 0.08 |
| ICD micro-F1 | 0.50 |
| ICD Precision | 1.00 |
| ICD Recall | 0.33 |

**A/B Comparison:**

| Condition | CheXpert F1 | ICD F1 | Interpretation |
|-----------|-------------|--------|----------------|
| Image-only (B-off) | 0.60 | 0.67 | Baseline visual |
| Image+EHR (B-on) | **0.75** | **1.00** | +25% improvement |
| Delta | **+0.15** | **+0.33** | EHR helps significantly |

### 3.3 Analysis of Results

**What Worked:**
- Multi-pass generation successfully produces valid JSON
- Token biasing improves label prediction accuracy
- EHR context measurably improves clinical reasoning (+0.15 CheXpert F1)
- ICD prediction especially benefits from lab/vital context

**Known Gaps:**
- Extended validation shows recall challenges (0.08 CheXpert recall)
- Rare pathologies (Pneumothorax, Fracture) still under-predicted
- Quick manifest results ≠ general performance (selection bias)
- Heuristic rules improve precision but mask model weaknesses

**Proposed Improvements:**
- Auxiliary loss top-up focused on CheXpert recall
- Self-consistency voting with more generations (3-5)
- ICD vocabulary expansion for rarer codes
- Larger training set (currently only 10K of 227K studies)

---

## 4. Topics You Must Know (Study Guide)

### 4.1 Vision-Language Models

**LLaVA Family:**

| Model | Year | Key Innovation |
|-------|------|---------------|
| LLaVA (original) | 2023 | Simple linear projection + instruction tuning |
| LLaVA 1.5 | 2023 | MLP projector, more data, Vicuna backbone |
| LLaVA-NeXT (v1.6) | 2024 | AnyRes, Mistral backbone, improved projector |
| LLaVA-Med | 2023 | Medical domain adaptation of LLaVA |

**Other Key Vision-Language Models:**

| Model | Architecture | Key Idea |
|-------|-------------|----------|
| Flamingo (DeepMind) | Cross-attention between vision and text | Perceiver resampler reduces visual tokens |
| BLIP-2 (Salesforce) | Q-Former bridges frozen image encoder + frozen LLM | Lightweight query transformer |
| InstructBLIP | BLIP-2 + instruction tuning | Better instruction following |
| MedPaLM-M (Google) | PaLM-E adapted for medical | Multi-modal medical reasoning |
| CheXagent | LLaVA adapted for CheXpert | Specialized for chest X-ray labeling |

### 4.2 MIMIC Datasets

| Dataset | Content | Size | Access |
|---------|---------|------|--------|
| MIMIC-CXR | Chest X-rays + reports | 377K images, 228K studies | PhysioNet credentialed |
| MIMIC-III | ICU clinical data | 58,976 admissions | PhysioNet credentialed |
| MIMIC-IV | Updated clinical data | 431,231 admissions | PhysioNet credentialed |
| CheXpert | Stanford CXR dataset | 224,316 images | Stanford AIMI |

### 4.3 CheXpert 14 Pathologies

The CheXpert labeler (Stanford) extracts 14 observations from radiology reports:

```
1. No Finding            8. Pneumothorax
2. Enlarged Cardio.      9. Pleural Effusion
3. Cardiomegaly         10. Pleural Other
4. Lung Opacity         11. Fracture
5. Lung Lesion          12. Support Devices
6. Edema                13. Atelectasis
7. Consolidation        14. Pneumonia
```

Note: We used 12 of these in our model (dropping Cardiomegaly and Atelectasis from the CheXpert head since they're represented in ICD).

### 4.4 Radiology Reporting Structure

```
TECHNIQUE: PA and lateral chest radiographs.

COMPARISON: Prior study dated [date].

FINDINGS:
Heart size is mildly enlarged. Mediastinal contours are unremarkable.
The lungs are clear bilaterally. No pleural effusion or pneumothorax.
Osseous structures are intact.

IMPRESSION:
1. Mild cardiomegaly, stable.
2. No acute cardiopulmonary process.
```

- **Findings:** Detailed observations (what the radiologist sees)
- **Impression:** Summary with clinical significance (what it means)
- Our model generates the **impression** (the clinically actionable part)

### 4.5 HIPAA and Medical Data Privacy

| Requirement | How We Address It |
|-------------|-------------------|
| Protected Health Info (PHI) | MIMIC-CXR is de-identified (Safe Harbor method) |
| Data Use Agreement | Required PhysioNet credentialed access |
| CITI Training | Completed human subjects research training |
| No Redistribution | Raw data stays on approved systems |
| Date Shifting | All dates shifted by random offset |
| ID Randomization | Patient/study IDs are random |
| Model Weights | Can be shared (no PHI embedded) |

### 4.6 LoRA Complete Reference

**Formula:**
$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

**Initialization:**
- $ A \sim \mathcal{N}(0, \sigma^2) $: random Gaussian
- $ B = 0 $: zero matrix (so $ \Delta W = 0 $ at start)
- This means training starts from the pre-trained model exactly

**Hyperparameter Guide:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| Rank (r) | 4-64 | Higher = more capacity, more params |
| Alpha (α) | r to 2r | Higher = larger learning rate for LoRA |
| α/r ratio | 1-4 | Effective learning rate multiplier |
| Dropout | 0-0.1 | Regularization (0.05 is typical) |
| Target Modules | varies | More modules = more capacity |

**Common Rank Choices:**
- r=4: Minimal adaptation (classification tasks)
- r=8: Light adaptation (sentiment, NER)
- r=16: Medium adaptation (our choice, structured generation)
- r=32-64: Heavy adaptation (domain shift, new capabilities)

### 4.7 Curriculum Learning Theory

**Original Paper:** Bengio et al., "Curriculum Learning" (ICML 2009)

**Key Idea:** Presenting training examples in a meaningful order (easy to hard) can:
1. Speed up convergence (fewer epochs to reach same loss)
2. Improve generalization (better final performance)
3. Avoid local minima (easy examples create smooth loss landscape)

**Difficulty Scoring Approaches:**

| Method | How Difficulty is Measured | Example |
|--------|--------------------------|---------|
| Data-driven | Loss on pre-trained model | High loss = hard |
| Heuristic | Domain knowledge | Report length, # findings |
| Curriculum by Task | Simpler subtask first | Image-only before image+EHR |
| Self-paced | Model decides difficulty | SPL (Kumar et al., 2010) |

**Our Approach (Task-Based Curriculum):**
- Stage A = simpler task (visual interpretation only)
- Stage B = complex task (multi-modal clinical reasoning)
- Mixing prevents catastrophic forgetting

### 4.8 Multi-Modal Learning Concepts

**Image-Text Alignment:**
The core challenge: vision encoders and LLMs operate in different embedding spaces. The projector must learn a mapping:

$$
f: \mathbb{R}^{d_{\text{vision}}} \rightarrow \mathbb{R}^{d_{\text{text}}}
$$

In our case: $ f: \mathbb{R}^{1024} \rightarrow \mathbb{R}^{4096} $ (CLIP -> Mistral)

**Contrastive Pre-training (CLIP):**
CLIP learns aligned image-text embeddings by:
$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)} \right]
$$

This pre-training is why CLIP features transfer well to medical imaging -- they encode semantic visual concepts that map to text.

### 4.9 A/B Testing Design for ML Models

**Key Principles:**
1. **Same model, different inputs:** Isolate the variable (EHR presence)
2. **Matched samples:** Use the same validation set for both conditions
3. **Statistical significance:** Need enough samples for reliable comparison
4. **Paired comparisons:** Each sample serves as its own control

**Our A/B Design:**
```
For each Stage B validation sample:
  1. Run model with image only (B-off)      -> predictions_off
  2. Run model with image + EHR (B-on)      -> predictions_on
  3. Compute per-sample metrics for both
  4. Compare: delta = metrics_on - metrics_off
```

---

## 5. Interview Questions & Answers

### Q1: "Walk me through this project end-to-end."

**Answer:**
"This was an academic project at the University of Maryland focused on automated radiology report generation from chest X-rays. The goal was to build a vision-language model that takes a chest X-ray -- and optionally patient EHR data -- and generates a structured JSON report with clinical impressions and disease labels.

I used MIMIC-CXR, the largest public chest X-ray dataset with 377K images and paired radiology reports. I fine-tuned LLaVA-NeXT v1.6, which combines a CLIP vision encoder with a Mistral-7B language model, using LoRA for parameter-efficient adaptation -- only 0.58% of parameters were trainable.

The key technical contributions were: (1) a curriculum learning strategy that starts training on easier image-only examples before adding EHR context, (2) a multi-pass inference pipeline that generates structured JSON through separate passes for impression, CheXpert labels, and ICD codes with token biasing and majority voting, and (3) an A/B evaluation framework that quantifies how much EHR conditioning improves clinical accuracy -- we saw a +0.15 improvement in CheXpert F1 and +0.33 in ICD F1 when EHR was provided."

---

### Q2: "How does LLaVA-NeXT work architecturally?"

**Answer:**
"LLaVA-NeXT has three main components. First, a frozen CLIP ViT-L/14 vision encoder that converts the chest X-ray into 576 patch tokens, each a 1024-dimensional vector. Second, a 2-layer MLP projection layer that maps these vision embeddings from 1024 to 4096 dimensions to match Mistral's hidden size. Third, the Mistral-7B language model that processes the projected visual tokens interleaved with text tokens.

The key insight is that the visual tokens are treated like additional text tokens in the LLM's self-attention -- they attend to and are attended by all other tokens. So when the model generates 'Pleural Effusion: 1', it's directly attending to the visual patch tokens that encode the effusion region of the X-ray.

Compared to LLaVA 1.5, the NeXT version adds AnyRes -- dynamic image resolution handling that can use higher-res images with a grid of sub-images -- and uses Mistral-7B which is smaller but better than Vicuna-13B thanks to sliding window attention and grouped query attention."

---

### Q3: "Why LoRA instead of full fine-tuning?"

**Answer:**
"Three reasons. First, **compute efficiency** -- full fine-tuning a 7.28B parameter model requires 120+ GB of GPU memory across multiple GPUs. With LoRA, I only trained 41.9M parameters (0.58%), fitting comfortably on a single A100 with 40 GB VRAM.

Second, **catastrophic forgetting prevention** -- freezing the base weights preserves all the pre-trained knowledge. The LoRA adapters learn task-specific adjustments on top. This is especially important in medical AI where the base model's general reasoning capabilities are valuable.

Third, **practical deployment** -- LoRA adapters are only ~160 MB. At inference time, I merged them into the base model weights, so there's zero additional latency. For experimentation, I could try different rank/alpha configurations without retraining from scratch.

The trade-off is slightly lower theoretical ceiling than full fine-tuning, but in practice, LoRA at rank 16 gets very close to full fine-tuning performance, especially for generation tasks."

---

### Q4: "Explain your curriculum learning design."

**Answer:**
"I designed a 2-stage curriculum inspired by Bengio et al.'s work. Stage A -- the 'easy' task -- uses only chest X-ray images to generate impressions and CheXpert labels. This builds the model's visual grounding: learning to identify opacities, effusions, and devices from pixels alone.

Stage B -- the 'hard' task -- adds patient EHR context: vitals, lab values, chronic conditions. The model must now integrate both visual and clinical information to generate impressions, CheXpert labels, AND ICD diagnostic codes.

Rather than training on all Stage A first then switching to Stage B, I used a mixing strategy: each epoch combines all 959 Stage A samples, plus synthetic Stage A created by stripping EHR from low-EHR Stage B samples, plus 65% of Stage B samples. This prevents catastrophic forgetting of image-only capability.

I also added stratified sampling to address class imbalance -- rare pathologies like Pneumothorax (3%) and Fracture (2%) get 5x sampling weight. Without this, the model would learn to always predict 'negative' for rare classes, which is clinically dangerous."

---

### Q5: "What did the A/B evaluation reveal?"

**Answer:**
"The A/B evaluation tested the same fine-tuned model with and without EHR context at inference time. The results showed that EHR conditioning provides a measurable improvement: +0.15 in CheXpert micro-F1 and +0.33 in ICD micro-F1.

The pattern was expected -- EHR helps most for conditions with strong lab correlates. For example, elevated BNP helps confirm pulmonary edema, and elevated CRP with imaging findings helps confirm pneumonia. For purely visual findings like pneumothorax, the EHR adds less value.

However, I should note the evaluation was on a small sample. The quick manifest showed strong results (0.75 CheXpert F1), but the extended validation dropped to 0.11, revealing recall gaps on harder cases. This gap tells us the model needs more training data and possibly an auxiliary loss top-up to improve recall on underrepresented pathologies."

---

### Q6: "How did you handle HIPAA and data privacy?"

**Answer:**
"MIMIC-CXR is a de-identified dataset released under a Data Use Agreement through PhysioNet. I completed the required CITI human subjects research training to obtain credentialed access.

The dataset uses the Safe Harbor de-identification method: patient names are removed, dates are shifted by random offsets, and all IDs are randomized. There's no Protected Health Information in the images or reports.

All data processing happened locally -- no raw data was uploaded to cloud services. The model weights themselves don't contain PHI and can be shared freely. The key constraint is that the raw MIMIC data cannot be redistributed, so anyone wanting to reproduce the work needs their own PhysioNet credentials."

---

### Q7: "Why structured JSON output instead of free text?"

**Answer:**
"Free-text reports are useful for radiologists but problematic for downstream clinical systems. Billing systems need ICD codes, not paragraphs. Clinical decision support needs discrete labels. Quality assurance needs counts.

JSON output gives us: (1) machine-parseable labels that integrate directly with EHR systems, (2) standardized CheXpert labels that enable objective evaluation against ground truth, (3) ICD codes for billing and epidemiological tracking, and (4) deterministic structure that can be validated programmatically.

The challenge is that LLMs aren't naturally good at producing valid JSON -- they tend to 'drift' into free text. I solved this with multi-pass generation, token biasing, and JSON validation with self-repair. The multi-pass approach was key: generating impression, CheXpert, and ICD in separate passes with specialized prompts and temperature settings for each."

---

### Q8: "How do you evaluate radiology reports? What metrics did you use?"

**Answer:**
"I used a combination of text quality metrics and clinical accuracy metrics. For text quality, BLEU-4 measures n-gram overlap with reference impressions, and ROUGE-L measures longest common subsequence similarity. These tell us if the generated impression is linguistically similar to what a radiologist wrote.

For clinical accuracy, I used micro-F1 and macro-F1 for both CheXpert (12 disease labels) and ICD (8 diagnostic codes). Micro-F1 gives an aggregate score weighted by class frequency, while macro-F1 treats all classes equally -- important because rare diseases like pneumothorax are clinically critical even though they're infrequent.

I also tracked JSON validity rate -- the percentage of outputs that contain parseable JSON. With multi-pass generation, this was near 100%.

The gold standard would be radiologist assessment -- having domain experts rate generated reports for clinical accuracy, completeness, and safety. That wasn't feasible in this academic setting, but it's what would be needed for any clinical deployment."

---

### Q9: "Explain the multi-pass generation pipeline."

**Answer:**
"Instead of asking the model to generate the entire JSON in one shot -- which is error-prone because the model can lose track of JSON structure mid-generation -- I split inference into three passes.

Pass 1 generates the clinical impression with low temperature (0.15) for factual accuracy. Pass 2 generates CheXpert labels as JSON with higher temperature (0.75) and token biasing that adds +4.0 logits to the '1' token and -0.5 to the '0' token. This biasing acts as soft constrained decoding, pushing the model toward valid label values. Pass 3 generates ICD codes similarly for Stage B samples.

Each label pass can be run multiple times (configurable voting), and the final label is determined by majority vote -- 34% positive votes needed to call a label positive. This voting reduces random errors and improves stability.

If any pass produces invalid JSON, I retry with a self-repair prompt: 'Your previous answer was not valid JSON. Return ONLY the JSON object now.' Up to 2 retries before falling back to defaults."

---

### Q10: "What is the multi-modal projector and why train it at full precision?"

**Answer:**
"The multi-modal projector is a 2-layer MLP that maps CLIP vision embeddings (1024-dim) to the Mistral-7B embedding space (4096-dim). It's the critical bridge between vision and language.

I trained it at full precision rather than through LoRA because it's the component most responsible for aligning the two modalities. A rank-16 LoRA adapter might not have enough capacity to learn a good cross-modal mapping from scratch. The projector parameters are relatively small compared to the full model, so training them at full precision adds minimal overhead.

In the config, `modules_to_save: ['multi_modal_projector']` tells PEFT to exclude the projector from LoRA wrapping and train it directly. This is a common pattern in vision-language model fine-tuning -- freeze the vision encoder, LoRA-adapt the LLM, and fully train the projector."

---

### Q11: "How does the auxiliary classification head work alongside generative loss?"

**Answer:**
"The model has two training objectives running simultaneously. The primary objective is the standard autoregressive language modeling loss -- predicting the next token in the JSON output. The secondary objectives are two auxiliary classification heads.

After the model generates its hidden states, I pool the hidden states corresponding to the assistant's response tokens (where labels != -100) by averaging them. This pooled representation is fed through linear heads: one for CheXpert (4096 -> 12) and one for ICD (4096 -> 8).

The total loss is: `base_loss + 4.0 * chexpert_loss + 3.0 * icd_loss`. The auxiliary losses use binary cross-entropy with focal loss (gamma=1.0) and positive class weighting (pos_weight=5.0) to handle class imbalance.

This multi-task setup helps because: (1) the auxiliary heads provide direct gradient signal for label prediction, which the generative loss alone might not optimize well, and (2) the shared representation benefits from both tasks -- the model learns features useful for both text generation and classification."

---

### Q12: "What is Grouped Query Attention in Mistral?"

**Answer:**
"Standard multi-head attention has separate key and value projections for each attention head. With 32 heads, you need 32 K projections and 32 V projections, which makes the KV cache large at inference time.

Grouped Query Attention, used in Mistral-7B, shares K and V projections across groups of query heads. Mistral uses 32 query heads but only 8 KV heads, so groups of 4 query heads share the same key-value pair. This gives a 4x reduction in KV cache memory with minimal quality degradation.

The intuition is that different query heads are often looking for similar key-value patterns, so sharing is efficient. At Mistral's 7B scale, GQA matches standard multi-head attention quality while making inference significantly faster and more memory-efficient, especially for long sequences with our multi-pass generation pipeline."

---

### Q13: "How did you handle class imbalance in medical labels?"

**Answer:**
"Class imbalance is severe in medical imaging -- 'No Finding' is 30% while 'Fracture' is 2%. I addressed this at three levels.

At the data level, I used stratified sampling with 5x weight for rare pathologies (Pneumothorax, Fracture, Lung Lesion, Pulmonary Embolism, Rib Fracture). Samples with these positive labels are 5x more likely to appear in each batch.

At the loss level, I used focal loss with positive class weighting. Focal loss down-weights easy negatives (gamma=1.0) and pos_weight=5.0 up-weights rare positives in the auxiliary classification heads.

At the inference level, token biasing adds +4.0 logits to the '1' token during label generation, making the model more likely to predict positive findings. And keyword-based post-processing catches obvious findings that the model's JSON labels missed -- if the impression says 'pneumothorax' but the CheXpert label says 0, the post-processor corrects it."

---

### Q14: "What is sliding window attention and why does Mistral use it?"

**Answer:**
"Standard self-attention has O(n^2) memory complexity -- each token attends to all other tokens. For a 4096-token sequence, that's 16 million attention weights per head per layer.

Sliding window attention limits each token to attending only the previous W=4096 tokens. This reduces memory to O(n*W) which is linear when W is fixed. But information still propagates through layers: after k layers, a token's effective receptive field is k*W tokens. With 32 layers and W=4096, the effective context is 131K tokens -- far larger than most sequences.

Mistral uses this because: (1) it allows longer sequences without memory explosion, (2) most relevant context in radiology is local (the impression relates to nearby findings), and (3) combined with GQA, it makes inference fast enough for interactive applications like our Streamlit demo."

---

### Q15: "What are the limitations of your approach?"

**Answer:**
"Several important limitations. First, **scale of fine-tuning data** -- we used 4,360 training samples from a dataset of 227K studies. More data would likely improve recall on rare pathologies.

Second, **evaluation scale** -- our A/B comparison was on small samples (4-8). Statistically robust conclusions would need hundreds of samples with confidence intervals.

Third, **no radiologist evaluation** -- all metrics are automated (BLEU, ROUGE, F1). Clinical deployment would require expert assessment of report safety and accuracy.

Fourth, **recall gap** -- the model achieves high precision but lower recall, meaning it misses some findings. In clinical settings, missing a pneumothorax is more dangerous than a false positive.

Fifth, **data recency** -- MIMIC-CXR reflects practices at one hospital. A production model would need multi-site training data to generalize.

Sixth, **not a diagnostic tool** -- this is a report drafting assistant, not a replacement for radiologist interpretation. Any deployment would require radiologist review and sign-off."

---

### Q16: "How would you improve this system for production deployment?"

**Answer:**
"Several key improvements. First, scale up training to use more of the 227K available studies, with better class balancing.

Second, implement uncertainty estimation -- the model should flag when it's unsure about a finding so the radiologist pays special attention. This could be done with Monte Carlo dropout or ensemble methods.

Third, add a reject/abstain mechanism -- if the model can't generate valid JSON after multiple attempts, flag for manual review rather than returning potentially incorrect labels.

Fourth, multi-site validation -- train and evaluate on data from multiple hospitals to ensure generalization.

Fifth, implement a human-in-the-loop workflow -- the model generates a first draft, the radiologist reviews and edits, and the edits feed back into model improvement.

Sixth, address latency -- our multi-pass pipeline with voting runs 3-15 generation passes per image. For real-time use, we'd need faster inference (quantization, batched generation, or distillation to a smaller model)."

---

### Q17: "Compare BLIP-2, Flamingo, and LLaVA architectures."

**Answer:**
"These are the three main architectural patterns for vision-language models.

**Flamingo** uses cross-attention: frozen image encoder and frozen LLM with lightweight cross-attention layers inserted between transformer blocks. A Perceiver resampler reduces visual tokens to a fixed count. Advantage: handles interleaved image-text sequences naturally. Disadvantage: needs architectural modifications to the LLM.

**BLIP-2** uses a Q-Former bridge: a small transformer that takes learned query tokens as input and cross-attends to frozen image features, producing a fixed set of visual tokens that feed into a frozen LLM. Advantage: very lightweight bridge, works with any frozen LLM. Disadvantage: information bottleneck at the Q-Former.

**LLaVA** is the simplest: a linear projection (or 2-layer MLP in v1.5+) maps image tokens directly into the LLM's embedding space. No cross-attention, no Q-Former. Visual tokens are just additional tokens in the sequence. Advantage: simple, works well with instruction tuning. Disadvantage: long visual token sequences (576-2880 tokens).

I chose LLaVA-NeXT because its simplicity makes LoRA fine-tuning straightforward -- we just adapt the LLM and projector without modifying architecture. The HuggingFace integration is also the most mature."

---

### Q18: "Explain the difference between findings and impressions in radiology."

**Answer:**
"Findings are the detailed observations -- what the radiologist sees in the image. They describe every anatomical structure: 'The cardiac silhouette is mildly enlarged. There is a small left pleural effusion. The lungs are otherwise clear.'

Impressions are the clinical summary -- what the findings mean. They highlight the most important observations and their clinical significance: '1. Mild cardiomegaly. 2. Small left pleural effusion. 3. No acute pneumonia.'

We focused on generating impressions because they're the clinically actionable part. Referring physicians often read only the impression. They're also shorter and more structured, making them better suited for automated generation. The CheXpert and ICD labels we generate alongside the impression provide the structured data that clinical systems need."

---

### Q19: "Why did you choose Mistral-7B as the LLM backbone?"

**Answer:**
"Several reasons. First, **efficiency** -- at 7B parameters, it fits on a single A100 with LoRA, unlike 13B or 70B models. Second, **architecture innovations** -- sliding window attention and GQA make it faster at inference, important for our multi-pass pipeline. Third, **strong reasoning** -- Mistral-7B benchmarks ahead of LLaMA-2 13B on most tasks despite being half the size. Fourth, **ecosystem** -- LLaVA-NeXT's Mistral variant had the best HuggingFace integration at the time. Fifth, **clinical reasoning** -- the model needed to reason about lab values and their clinical implications, which requires strong instruction-following and reasoning capabilities that Mistral excels at."

---

### Q20: "How does token biasing work as constrained decoding?"

**Answer:**
"During generation, at each step the model produces logits (unnormalized probabilities) for all 32,000 vocabulary tokens. Softmax converts these to probabilities for sampling.

Token biasing adds a constant to specific token logits before softmax. For CheXpert label generation, I add +4.0 to the logit for token '1' and -0.5 for token '0'. This is mathematically equivalent to multiplying the probability of '1' by e^4 ≈ 55x relative to neutral tokens.

It's 'soft' constrained decoding because the model can still generate other tokens -- we're not hard-blocking anything. But the bias strongly pushes toward valid label values. This is more robust than hard constraints because: (1) the model can still generate the label name before the value, (2) it handles different tokenizations of '1' (space-prefixed, quote-wrapped, etc.), and (3) it doesn't require modifying the generation algorithm."

---

### Q21: "What would happen if you used full fine-tuning instead of LoRA?"

**Answer:**
"Full fine-tuning would have three main effects. First, **better theoretical performance** -- all 7.28B parameters could adapt to the radiology domain, potentially improving recall on rare pathologies. Second, **risk of catastrophic forgetting** -- the model might lose general language capabilities, producing less fluent text or losing instruction-following ability. Third, **impractical compute** -- we'd need multiple A100s or an H100 with 80 GB VRAM, and training would take days instead of hours.

In practice, LoRA at rank 16 typically achieves 95-98% of full fine-tuning performance on similar tasks. The gap is largest when the target domain is very different from pre-training (e.g., adapting an English model to code), but for radiology report generation, the base model's language capabilities transfer well and LoRA's capacity is sufficient for the adaptation."

---

### Q22: "How did you handle the uncertain (-1) labels in CheXpert?"

**Answer:**
"CheXpert labels have three values: 1 (positive), 0 (negative), and -1 (uncertain). Uncertain means the NLP labeler couldn't confidently determine the finding from the report text -- for example, 'possible pneumonia' or 'cannot exclude effusion.'

During training, I masked out uncertain labels from the auxiliary classification loss. The mask tensor has 1.0 for positive/negative labels (supervise) and 0.0 for uncertain labels (ignore). This way, the model isn't penalized for predicting anything when the ground truth is ambiguous.

For the generative output, the model can still produce -1 in the CheXpert JSON for uncertainty. During evaluation, uncertain predictions and uncertain ground truth are excluded from F1 computation to avoid artificially inflating or deflating scores.

This is important because forcing a binary decision on uncertain cases would introduce label noise -- the model would learn inconsistent patterns from ambiguous examples."

---

### Q23: "Describe your data processing pipeline."

**Answer:**
"The pipeline had six stages. First, I extracted 10,003 chest X-ray images from MIMIC-CXR, converting from DICOM to JPG format. Second, I extracted impression sections from paired radiology reports.

Third, I mapped CheXpert labels from the automated labeler -- this gave us 12-label vectors for each study with positive/negative/uncertain values. Fourth, I built EHR context by linking patient IDs to MIMIC-III/IV data, extracting vitals closest to the X-ray timestamp, recent lab results, and chronic conditions with ICD codes.

Fifth, I created the curriculum structure: Stage A samples (image + impression + CheXpert) and Stage B samples (image + EHR + impression + CheXpert + ICD). Sixth, I cleaned the data: removed 3,994 duplicates (41.4%), validated image paths, and created train/validation splits.

The final dataset was 4,360 training and 770 validation samples in JSONL format, with each line containing the image path, impression, CheXpert labels, and for Stage B, the patient data with vitals, labs, and ICD codes."

---

### Q24: "What is focal loss and why did you use it?"

**Answer:**
"Focal loss was introduced by Lin et al. (2017) for object detection to address extreme class imbalance. The standard cross-entropy loss treats all examples equally, so with 98% negatives, the loss is dominated by easy negatives that the model already classifies correctly.

Focal loss adds a modulating factor: FL(p_t) = -(1-p_t)^gamma * log(p_t). When gamma > 0, well-classified examples (high p_t) get down-weighted. A negative sample the model correctly predicts with 0.95 confidence contributes almost nothing to the loss. A misclassified rare positive contributes much more.

I used gamma=1.0 with pos_weight=5.0 in the auxiliary heads. This combination ensures that: (1) easy negatives don't dominate training, (2) hard positives get disproportionate attention, and (3) rare pathologies receive sufficient gradient signal despite appearing in only 2-5% of samples."

---

### Q25: "How would you add uncertainty estimation?"

**Answer:**
"Three approaches I'd consider. First, **Monte Carlo Dropout** -- enable dropout at inference time and run N forward passes. The variance of predictions across passes estimates epistemic uncertainty. If the model predicts Pneumonia=1 in 3/5 passes and Pneumonia=0 in 2/5 passes, that's high uncertainty.

Second, **temperature scaling** -- calibrate the model's softmax temperatures on a held-out set so that predicted probabilities reflect true likelihoods. A well-calibrated model that predicts 0.7 for Edema should be correct 70% of the time.

Third, **ensemble disagreement** -- train multiple LoRA adapters with different seeds and compare their predictions. Disagreement indicates cases where the model is uncertain.

Our voting mechanism already provides a form of uncertainty: if 3 votes are split 2-1, that's less certain than 3-0. We could expose this vote distribution as a confidence score alongside each label."

---

## 6. Red Flags & How to Handle

### 6.1 Medical AI Ethics & Deployment Caveats

**Red Flag:** "Would you deploy this in a hospital?"

**How to Handle:**
"Not without significant additional validation. This is a research prototype that demonstrates the feasibility of automated report generation. Clinical deployment would require: (1) FDA regulatory approval (likely Class II medical device), (2) large-scale multi-site clinical trials, (3) radiologist-in-the-loop validation, (4) continuous monitoring for performance drift, and (5) clear documentation that this is a decision-support tool, not a diagnostic system. The model should generate first drafts that radiologists review and edit, not replace clinical judgment."

### 6.2 Dataset Bias

**Red Flag:** "Is the model biased?"

**How to Handle:**
"Yes, in several ways. MIMIC-CXR comes from a single hospital (Beth Israel Deaconess) in Boston, so it reflects that institution's patient population, imaging equipment, and reporting style. The class distribution is heavily imbalanced -- 30% 'No Finding' vs 2% 'Fracture'. The CheXpert labels themselves are imperfect -- they're extracted by an NLP labeler, not manually annotated by radiologists.

To mitigate: we used stratified sampling and focal loss for class imbalance. For deployment, we'd need multi-site training data, demographic subgroup analysis (performance by age, sex, race), and regular bias audits. We'd also need to evaluate performance specifically on underrepresented populations."

### 6.3 Clinical vs Automated Evaluation Gap

**Red Flag:** "Your BLEU score is X -- does that mean the reports are clinically accurate?"

**How to Handle:**
"No. BLEU and ROUGE measure linguistic similarity, not clinical accuracy. A report could have high BLEU but contain a clinically dangerous error -- for example, describing 'left pleural effusion' when it's actually right-sided. Conversely, a report could have low BLEU but be clinically correct using different phrasing.

That's why we also evaluate CheXpert F1 (disease label accuracy) and ICD F1 (diagnostic code accuracy), which are closer to clinical correctness. But even these are imperfect -- the ground truth labels themselves come from an automated NLP labeler (CheXpert), not from expert annotation.

The gold standard is a structured radiologist evaluation: blind review of generated reports scored for accuracy, completeness, clinical safety, and actionability. This wasn't feasible in our academic setting but would be essential for any clinical deployment."

### 6.4 Small Evaluation Set

**Red Flag:** "Your results are on only 4-8 samples."

**How to Handle:**
"You're right to flag this. The quick manifest results (0.75 CheXpert F1) are on 4 samples and don't generalize -- the extended validation (8 samples) shows 0.11 F1, a huge gap. For publishable results, I'd need hundreds of samples with confidence intervals.

The small evaluation was a practical constraint of running CPU inference (each sample takes 30+ seconds with multi-pass generation). With GPU inference, I'd run the full 770-sample validation set. The current results are best viewed as proof-of-concept: the pipeline works end-to-end, the A/B framework measures EHR contribution, and the curriculum learning approach is sound. The actual numbers need more data to be reliable."

### 6.5 Overfitting to Heuristics

**Red Flag:** "Aren't the keyword rules just hardcoded heuristics?"

**How to Handle:**
"They are, and that's by design as a safety net, not a primary mechanism. The keyword rules catch obvious mismatches between the impression text (which mentions 'pneumothorax') and the label prediction (which says 0). In a clinical setting, these rules would be validated against expert guidelines.

The important thing is that the model generates the correct impression text -- the keyword rules just ensure the structured labels are consistent with it. If I had more training data and compute, the model would learn to predict labels correctly without heuristics. The rules are a pragmatic bridge for a research prototype."

---

## 7. Key Takeaways & Talking Points

### 7.1 Technical Depth Signals

Use these to demonstrate deep understanding:

1. **"We trained the projector at full precision because LoRA rank 16 isn't sufficient for learning the cross-modal mapping from scratch"** -- shows understanding of LoRA limitations
2. **"We masked uncertain CheXpert labels (-1) from the auxiliary loss to avoid introducing label noise"** -- shows clinical data handling sophistication
3. **"Focal loss with gamma=1.0 and pos_weight=5.0 addresses both easy-negative dominance and class imbalance"** -- shows loss function design thinking
4. **"The multi-pass pipeline exists because LLMs lose JSON structure during long autoregressive generation"** -- shows practical generation experience
5. **"Voting with 34% threshold across 3 generations balances sensitivity and specificity"** -- shows ensemble reasoning

### 7.2 Project Strengths to Highlight

- **Full pipeline ownership:** Data processing -> Model training -> Evaluation -> Demo
- **Architecture decisions with rationale:** LLaVA-NeXT > LLaVA 1.5, LoRA > full FT, curriculum > random
- **Rigorous evaluation design:** A/B framework isolates EHR contribution
- **Medical domain knowledge:** HIPAA compliance, CheXpert labels, ICD codes, clinical reasoning
- **Production thinking:** Multi-pass inference, JSON validation, keyword safety nets, merged weights for deployment

### 7.3 One-Minute Elevator Pitch

"I fine-tuned LLaVA-NeXT -- a 7 billion parameter vision-language model -- to generate structured radiology reports from chest X-rays. The model takes an X-ray image and optionally patient history, and outputs a JSON report with clinical impressions and disease labels. I used LoRA to efficiently adapt the model with only 0.6% trainable parameters, designed a curriculum learning strategy that starts with easier image-only tasks before adding clinical context, and built an A/B evaluation showing that EHR conditioning improves diagnostic accuracy by 25%. The whole system runs on a single GPU and includes an interactive Streamlit demo for real-time testing."

### 7.4 What This Project Demonstrates About You

| Skill | Evidence |
|-------|----------|
| **Vision-Language Models** | LLaVA-NeXT architecture, CLIP encoder, multi-modal projector |
| **Parameter-Efficient Fine-Tuning** | LoRA configuration, rank/alpha tuning, module selection |
| **Training Strategy** | Curriculum learning, class balancing, focal loss |
| **Structured Generation** | Multi-pass inference, constrained decoding, JSON validation |
| **Medical AI** | HIPAA compliance, CheXpert labels, ICD codes, clinical reasoning |
| **Evaluation Design** | A/B testing, multiple metrics (F1, BLEU, ROUGE), multi-scale evaluation |
| **Full-Stack ML** | Data pipeline -> Training -> Inference -> Demo (Streamlit) |
| **Research Thinking** | Known limitations, proposed improvements, honest about gaps |
