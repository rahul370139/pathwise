# Fine-Tuning & Parameter-Efficient Fine-Tuning (PEFT) — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, CLIP LoRA Fine-Tuning, Multimodal Systems

---

# Table of Contents

1. [Overview: Why Fine-Tuning Matters](#1-overview-why-fine-tuning-matters)
2. [Data Preparation for Fine-Tuning](#2-data-preparation-for-fine-tuning)
3. [Full Fine-Tuning](#3-full-fine-tuning)
4. [LoRA (Low-Rank Adaptation)](#4-lora-low-rank-adaptation)
5. [QLoRA (Quantized LoRA)](#5-qlora-quantized-lora)
6. [AdaLoRA (Adaptive LoRA)](#6-adalora-adaptive-lora)
7. [Adapters, Prefix Tuning & Prompt Tuning](#7-adapters-prefix-tuning--prompt-tuning)
8. [Instruction Tuning (SFT)](#8-instruction-tuning-sft)
9. [Preference Optimization: DPO & PPO RLHF](#9-preference-optimization-dpo--ppo-rlhf)
10. [Evaluation & Packaging](#10-evaluation--packaging)
11. [Decision Tree: Choosing the Right Method](#11-decision-tree-choosing-the-right-method)
12. [Common Interview Questions](#12-common-interview-questions-with-strong-answers)
13. [Key Takeaways](#13-key-takeaways)

---

# **1. Overview: Why Fine-Tuning Matters**

---

## **1.1 The Pre-train → Adapt Pipeline**

Modern LLMs follow a two-stage lifecycle:

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM LIFECYCLE                             │
│                                                              │
│  Stage 1: PRE-TRAINING                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │  Massive corpus (TB of text)                      │        │
│  │  Next-token prediction / MLM                      │        │
│  │  Result: General-purpose foundation model         │        │
│  │  Cost: $1M - $100M+ (weeks on 1000s of GPUs)     │        │
│  └──────────────────────────────────────────────────┘        │
│                          │                                    │
│                          ▼                                    │
│  Stage 2: ADAPTATION (you are here)                          │
│  ┌──────────────────────────────────────────────────┐        │
│  │  Fine-tune on YOUR domain / task data             │        │
│  │  Instruction tuning, preference alignment         │        │
│  │  Result: Specialized, aligned model               │        │
│  │  Cost: $10 - $10K (hours on 1-8 GPUs)            │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

**Why can't we just prompt?** Pre-trained models know *language*, but they don't know *your* task format, domain jargon, or preferred style. Fine-tuning encodes that knowledge directly into the weights.

---

## **1.2 When to Fine-Tune vs Prompt Engineering vs RAG**

| Criterion | Prompt Engineering | RAG | Fine-Tuning |
|---|---|---|---|
| **Setup effort** | Minutes | Hours–Days | Days–Weeks |
| **Data needed** | 0 (few-shot examples) | A document corpus | 100–100K+ labeled examples |
| **What it changes** | Input context | Retrieved context | Model weights |
| **Best for** | Quick prototyping, simple tasks | Knowledge-heavy QA, up-to-date info | Style/format control, domain specialization, latency-critical |
| **Hallucination control** | Limited | Good (grounded in docs) | Moderate (learns patterns, can still hallucinate) |
| **Latency** | Baseline | Higher (retrieval + generation) | Lowest (knowledge baked in) |
| **Cost per query** | Token cost for long prompts | Retrieval infra + token cost | Low (smaller prompts, possibly smaller model) |

**Decision heuristic:**

```
Is the task knowledge-heavy with frequently changing data?
  └─ YES → RAG (optionally + fine-tuning for format)
  └─ NO  → Do you need specific style/format/behavior?
             └─ YES → Fine-tuning
             └─ NO  → Prompt engineering (start here first)
```

> **Interview tip:** These aren't mutually exclusive. The strongest production systems often combine RAG + fine-tuned models. Fine-tune for *behavior* (style, format, tool use), RAG for *knowledge* (facts, documents).

---

## **1.3 The Fine-Tuning Spectrum**

```
Parameter Efficiency vs. Adaptation Power

  ◄──── Fewer trainable params ────────────── More trainable params ────►

  Prompt      Prefix       Adapters      LoRA /       Full
  Tuning      Tuning       (Houlsby)     QLoRA        Fine-Tuning
  ─────────────────────────────────────────────────────────────────────
  ~0.01%      ~0.1%        ~0.5-2%       ~0.5-5%      100%
  params      params       params        params       params

  Minimal     Moderate     Good          Excellent    Maximum
  adaptation  adaptation   adaptation    adaptation   adaptation

  1 GPU       1 GPU        1 GPU         1 GPU        4-8+ GPUs
  (any)       (any)        (16GB+)       (16-24GB)    (80GB each)
```

---

# **2. Data Preparation for Fine-Tuning**

---

## **2.1 Tokenization: Always Use the Base Model's Tokenizer**

The tokenizer and model vocabulary are **inseparable**. Using a mismatched tokenizer will produce garbage.

```python
from transformers import AutoTokenizer

# Always load tokenizer matched to model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Verify special tokens
print(tokenizer.bos_token)    # <|begin_of_text|>
print(tokenizer.eos_token)    # <|end_of_text|>
print(tokenizer.pad_token)    # Often None for causal LMs

# Fix missing pad token (common for causal LMs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**Key rules:**
- **Never** train a new tokenizer for fine-tuning (unless pre-training from scratch)
- **Always** set `padding_side="left"` for causal/decoder-only LMs during batched inference
- **Always** add EOS tokens at the end of each training example
- If adding special tokens, resize embeddings: `model.resize_token_embeddings(len(tokenizer))`

---

## **2.2 Data Formats**

### **a) SFT / Instruction Format**

The standard format for teaching a model to follow instructions:

```json
{
  "instruction": "Summarize the following article in 3 bullet points.",
  "input": "The Federal Reserve announced today that...",
  "output": "• The Fed raised interest rates by 25bps\n• Inflation remains above target\n• Markets reacted with modest gains"
}
```

Variants: Alpaca format, ShareGPT (multi-turn), OpenAI chat format.

### **b) Classification Format**

```json
{
  "text": "This product exceeded my expectations!",
  "label": "positive"
}
```

For sequence classification, add a classification head or format as instruction:

```json
{
  "instruction": "Classify the sentiment of the following review.",
  "input": "This product exceeded my expectations!",
  "output": "positive"
}
```

### **c) RLHF / DPO Preference Format**

For preference optimization (DPO, KTO, ORPO):

```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old.",
  "chosen": "Imagine you have two magic coins. When you flip one and it lands heads, the other one — no matter how far away — always lands tails...",
  "rejected": "Quantum entanglement is a phenomenon in quantum mechanics whereby the quantum state of a particle becomes correlated with the quantum state..."
}
```

### **d) Multi-Turn Conversation**

```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "What are symptoms of type 2 diabetes?"},
    {"role": "assistant", "content": "Common symptoms include increased thirst, frequent urination..."},
    {"role": "user", "content": "How is it diagnosed?"},
    {"role": "assistant", "content": "Diagnosis typically involves fasting blood glucose tests..."}
  ]
}
```

---

## **2.3 Cleaning & Preprocessing**

```
┌──────────────────────────────────────────────────────┐
│              DATA CLEANING PIPELINE                   │
│                                                       │
│  Raw Data                                             │
│    │                                                  │
│    ▼                                                  │
│  1. Deduplication (exact + fuzzy with MinHash/LSH)    │
│    │                                                  │
│    ▼                                                  │
│  2. Length filtering                                   │
│     - Remove too short (< 10 tokens)                  │
│     - Cap at max_seq_len (2048 / 4096)                │
│    │                                                  │
│    ▼                                                  │
│  3. Quality filtering                                 │
│     - Remove garbled text, encoding errors             │
│     - Filter low-quality (perplexity score)            │
│    │                                                  │
│    ▼                                                  │
│  4. PII removal (emails, phone numbers, SSNs)         │
│    │                                                  │
│    ▼                                                  │
│  5. Format validation                                 │
│     - Ensure all required fields present               │
│     - Validate instruction/output pairs                │
│    │                                                  │
│    ▼                                                  │
│  Clean Dataset                                        │
└──────────────────────────────────────────────────────┘
```

```python
from datasets import load_dataset, Dataset

dataset = load_dataset("json", data_files="data.jsonl", split="train")

# Deduplication
def dedup(dataset):
    seen = set()
    unique = []
    for ex in dataset:
        key = ex["instruction"] + ex["output"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return Dataset.from_list(unique)

# Length capping
def cap_length(example):
    tokens = tokenizer(example["output"], truncation=True, max_length=2048)
    return {"output": tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)}

dataset = dataset.map(cap_length)
```

---

## **2.4 Chat Templates**

Modern models (Llama 3, Mistral, ChatML) expect specific prompt formatting. **Never hard-code templates** — use the tokenizer's built-in `apply_chat_template`:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is LoRA?"},
]

# Automatically applies the correct chat format for the model
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # Return string (not token IDs)
    add_generation_prompt=True # Add assistant turn prefix for inference
)
print(formatted)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
# What is LoRA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**Why this matters:** Mismatched chat templates are the #1 silent cause of poor fine-tuning results. The model learns to associate specific token patterns with roles — break the pattern and it outputs garbage.

---

# **3. Full Fine-Tuning**

---

## **3.1 What It Is**

Full fine-tuning updates **every parameter** in the model. For a 7B model, that's ~7 billion float32 parameters (28 GB just for weights, ~4× that for optimizer states + gradients).

```
┌────────────────────────────────────────────────────────┐
│                 FULL FINE-TUNING                        │
│                                                         │
│  Pre-trained Model (all params frozen = NO)             │
│  ┌─────────────────────────────────────────────┐        │
│  │  Embedding Layer          ← UPDATED         │        │
│  │  Transformer Block 1      ← UPDATED         │        │
│  │  Transformer Block 2      ← UPDATED         │        │
│  │  ...                      ← UPDATED         │        │
│  │  Transformer Block N      ← UPDATED         │        │
│  │  LM Head                  ← UPDATED         │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
│  Trainable Parameters: 100%                             │
│  Memory: ~4× model size (weights + grads + optimizer)   │
└────────────────────────────────────────────────────────┘
```

---

## **3.2 When to Use Full Fine-Tuning**

| Scenario | Reason |
|---|---|
| **Huge proprietary dataset** (100K+ examples) | Enough data to justify updating all parameters |
| **Specialized architecture changes** | Adding new heads, modifying attention patterns |
| **Small models (<1B params)** | Memory is manageable; full FT is often better than PEFT |
| **Maximum performance needed** | Full FT is the ceiling — PEFT trades perf for efficiency |
| **Domain is very far from pre-training** | Medical, legal, code — need deep weight changes |

---

## **3.3 Memory Budget**

For a model with `P` parameters in fp32:

```
Memory ≈ P × (4 bytes weights
            + 4 bytes gradients
            + 8 bytes optimizer states [Adam: m + v])
       = P × 16 bytes

7B model:  7B × 16 = 112 GB  →  Need 2×A100 (80GB) minimum
13B model: 13B × 16 = 208 GB →  Need 4×A100
70B model: 70B × 16 = 1.12 TB → Need 16×A100

With bf16 mixed precision:
  7B model ≈ 70 GB → 1×A100 (80GB) barely fits
  13B model ≈ 130 GB → 2×A100
```

---

## **3.4 Distributed Training: DeepSpeed & FSDP**

### **DeepSpeed ZeRO Stages**

```
┌──────────────────────────────────────────────────────────────┐
│ ZeRO Stage │  What's Sharded          │  Memory Reduction    │
│────────────│──────────────────────────│──────────────────────│
│ Stage 0    │  Nothing (DDP)           │  1×                  │
│ Stage 1    │  Optimizer states         │  ~4×                 │
│ Stage 2    │  + Gradients              │  ~8×                 │
│ Stage 3    │  + Parameters             │  ~N× (N = #GPUs)    │
│ + Offload  │  + CPU/NVMe offloading    │  Near-infinite       │
└──────────────────────────────────────────────────────────────┘
```

- **ZeRO-1:** Shard optimizer states across GPUs. Most common starting point.
- **ZeRO-2:** Also shard gradients. Good balance of speed and memory.
- **ZeRO-3:** Shard everything including parameters. Essential for 70B+ models. Higher communication overhead.

### **FSDP (Fully Sharded Data Parallel)**

PyTorch-native alternative to DeepSpeed ZeRO-3. Shards parameters, gradients, and optimizer states across GPUs.

```python
# In training config
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    fsdp_plugin=fsdp_plugin  # or use DeepSpeed config
)
```

---

## **3.5 Full Fine-Tuning Code Example**

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # FlashAttention for speed
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

dataset = load_dataset("json", data_files="train.jsonl", split="train")
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./full_ft_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,        # Effective batch = 2 × 8 = 16
    learning_rate=2e-5,                   # Lower LR for full FT (vs PEFT)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                            # bfloat16 mixed precision
    gradient_checkpointing=True,          # Trade compute for memory
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    deepspeed="ds_config_zero3.json",     # DeepSpeed ZeRO-3 config
    report_to="wandb",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./full_ft_final")
```

**DeepSpeed ZeRO-3 config (`ds_config_zero3.json`):**
```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

---

# **4. LoRA (Low-Rank Adaptation)**

---

## **4.1 Core Intuition**

**Key insight (Aghajanyan et al., 2020):** Pre-trained language models have a **low intrinsic dimensionality** — you don't need to update all parameters. The weight updates during fine-tuning lie in a low-rank subspace.

**LoRA (Hu et al., 2021)** exploits this: instead of updating the full weight matrix `W`, we learn a low-rank decomposition of the *change* to `W`.

---

## **4.2 How LoRA Works — The Math**

For a pre-trained weight matrix `W₀ ∈ ℝ^(d×k)`:

```
Standard fine-tuning:  W = W₀ + ΔW       (ΔW ∈ ℝ^(d×k), full-rank)

LoRA:                  W' = W₀ + (α/r) · A · B

Where:
  W₀ ∈ ℝ^(d×k)   — frozen pre-trained weights
  A  ∈ ℝ^(d×r)    — trainable "down-projection"
  B  ∈ ℝ^(r×k)    — trainable "up-projection"
  r  << min(d, k)  — rank (typically 8–64)
  α  ∈ ℝ           — scaling factor (lora_alpha)
```

**Visual:**

```
                    Frozen W₀
Input x ─────────► [d × k] ──────────► + ──────► Output
    │                                   ▲
    │              Trainable LoRA       │
    └──────► [d × r] ──► [r × k] ──────┘
                A    ×    B
             (down)     (up)
          scaled by (α / r)
```

**Parameter count comparison:**

```
Full weight matrix W:  d × k parameters
LoRA adapters A + B:   d × r + r × k = r × (d + k) parameters

Example (Llama attention, d=k=4096, r=16):
  Full:  4096 × 4096 = 16,777,216 parameters
  LoRA:  16 × (4096 + 4096) = 131,072 parameters
  Reduction: 99.2%
```

---

## **4.3 Initialization**

- **A** is initialized with random Gaussian values (Kaiming uniform)
- **B** is initialized to **zero**
- This ensures `A · B = 0` at the start → the model begins identical to the pre-trained model
- Training smoothly moves away from the pre-trained weights

---

## **4.4 Scaling Factor**

The effective update is scaled by `α / r`:

```
ΔW = (α / r) · A · B
```

- **`r` (rank):** Controls the capacity of the low-rank update. Higher r = more expressive but more parameters.
- **`α` (lora_alpha):** Scaling hyperparameter. Common practice: set `α = 2r` (so effective scale = 2).
- **Effective learning rate** for LoRA params is approximately `(α / r) × base_lr`.

| Rank (r) | lora_alpha | Effective Scale (α/r) | Trainable Params (per layer) | Use Case |
|---|---|---|---|---|
| 8 | 16 | 2.0 | ~65K | Quick experiments, simple tasks |
| 16 | 32 | 2.0 | ~131K | General purpose (recommended start) |
| 32 | 64 | 2.0 | ~262K | Complex domain adaptation |
| 64 | 128 | 2.0 | ~524K | Approaching full FT expressiveness |

---

## **4.5 Which Modules to Target**

In a standard Transformer block:

```
┌────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK                      │
│                                                     │
│  Multi-Head Attention:                              │
│    q_proj  (query projection)    ← LoRA target ✓   │
│    k_proj  (key projection)      ← LoRA target ✓   │
│    v_proj  (value projection)    ← LoRA target ✓   │
│    o_proj  (output projection)   ← LoRA target ✓   │
│                                                     │
│  MLP / Feed-Forward:                                │
│    gate_proj / up_proj           ← LoRA target ✓   │
│    down_proj                     ← LoRA target ✓   │
│                                                     │
│  LayerNorm (few params)          ← Usually skip ✗   │
│  Embeddings                      ← Usually skip ✗   │
└────────────────────────────────────────────────────┘
```

**Recommendations:**
- **Minimum:** `q_proj`, `v_proj` (original LoRA paper)
- **Better:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (all attention projections)
- **Best (common):** All attention + MLP projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Targeting MLP layers improves factual knowledge adaptation

---

## **4.6 LoRA Performance**

| Metric | Full FT | LoRA (r=16) | Improvement |
|---|---|---|---|
| Trainable params (7B model) | 7B (100%) | ~20M (0.3%) | **99.7% reduction** |
| Training memory | 70+ GB | 2–4 GB (+ frozen model) | **~90% reduction** |
| Training speed | Baseline | 3–5× faster | Fewer params to update |
| Task performance | Best | 95–100% of full FT | Negligible gap for most tasks |
| Checkpoint size | ~14 GB | ~20–80 MB | **99%+ reduction** |

---

## **4.7 LoRA Code Example**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor (alpha/r = 2)
    lora_dropout=0.05,             # Dropout on LoRA layers
    target_modules=[               # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",                   # Don't train bias terms
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# trainable params: 20,971,520 || all params: 8,051,232,768 || trainable%: 0.2604

# Training (using Hugging Face Trainer or TRL SFTTrainer)
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,            # Higher LR than full FT — LoRA adapters need it
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()

# Save only the LoRA adapter weights (~20-80MB)
model.save_pretrained("./lora_adapter")
```

---

## **4.8 LoRA for CLIP (My Experience)**

> *In my project, I applied LoRA to CLIP's vision encoder for medical image classification (radiology reports), keeping the text encoder frozen.*

```python
from peft import LoraConfig, get_peft_model

# LoRA on CLIP's vision encoder only
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # CLIP attention
    lora_dropout=0.1,
    modules_to_save=["visual_projection"],  # Also train projection head
)

clip_model = get_peft_model(clip_model, lora_config)
# Trainable: ~0.5% of total parameters
# Result: Competitive with full fine-tuning on CheXpert classification
```

**Key insight for CLIP LoRA:** Target attention projections in the vision encoder while keeping the text encoder frozen. This preserves CLIP's text understanding while adapting visual features to your domain.

---

# **5. QLoRA (Quantized LoRA)**

---

## **5.1 What It Is**

**QLoRA (Dettmers et al., 2023)** = LoRA + 4-bit quantization of the base model. The frozen base weights are stored in 4-bit precision (NF4), while LoRA adapters train in bf16/fp16.

```
┌──────────────────────────────────────────────────────────┐
│                       QLoRA                               │
│                                                           │
│  Base Model Weights:  4-bit quantized (NF4) ← FROZEN     │
│  LoRA Adapters (A,B): bfloat16              ← TRAINABLE  │
│  Computation:         bf16 (dequantize on the fly)        │
│                                                           │
│  Key innovations:                                         │
│  1. NF4 (4-bit NormalFloat): optimal for normally-        │
│     distributed weights                                   │
│  2. Double quantization: quantize the quantization        │
│     constants too (saves ~0.37 bits/param)                │
│  3. Paged optimizers: offload optimizer states to CPU     │
│     when GPU runs out of memory                           │
└──────────────────────────────────────────────────────────┘
```

---

## **5.2 Memory Comparison**

| Method | 7B Model Memory | 13B Model Memory | 70B Model Memory |
|---|---|---|---|
| Full FT (fp32) | ~112 GB | ~208 GB | ~1.12 TB |
| Full FT (bf16) | ~70 GB | ~130 GB | ~560 GB |
| LoRA (bf16 base) | ~16 GB | ~28 GB | ~140 GB |
| **QLoRA (4-bit base)** | **~6 GB** | **~10 GB** | **~40 GB** |

> **This is why QLoRA is revolutionary:** You can fine-tune a 7B model on a single 8GB consumer GPU, or a 70B model on a single A100 (80GB).

---

## **5.3 QLoRA Code Example**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load weights in 4-bit
    bnb_4bit_quant_type="nf4",            # NormalFloat4 (optimal distribution)
    bnb_4bit_compute_dtype=torch.bfloat16,# Compute in bf16 during forward pass
    bnb_4bit_use_double_quant=True,       # Double quantization (saves memory)
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# Prepare model for k-bit training
# (freezes base, casts LoRA to fp32 for stable training)
model = prepare_model_for_kbit_training(model)

# LoRA config (same as regular LoRA)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 20,971,520 || all params: 4,529,823,744 || trainable%: 0.46

# Training proceeds identically to LoRA
# But now the whole thing fits in ~6GB VRAM for a 7B model!
```

---

## **5.4 QLoRA: Quality vs Efficiency Trade-off**

| Aspect | LoRA (bf16) | QLoRA (4-bit) |
|---|---|---|
| Base model precision | bf16 (16-bit) | NF4 (4-bit) |
| Training quality | Baseline | ~98-99% of LoRA |
| Memory usage | 2–4× less than full FT | 6–8× less than full FT |
| Training speed | Fast | Slightly slower (dequantize overhead) |
| Best for | When you have enough VRAM | When GPU-constrained |

> **Interview tip:** QLoRA's quality loss is minimal because (a) NF4 is information-theoretically optimal for normally-distributed weights, and (b) the LoRA adapters still train in full precision. The quantization noise in the frozen base is compensated by the adapters.

---

# **6. AdaLoRA (Adaptive LoRA)**

---

## **6.1 Motivation**

Standard LoRA uses the **same rank `r`** for every layer. But not all layers are equally important for a task — some need more capacity, others need less.

**AdaLoRA (Zhang et al., 2023)** dynamically adjusts the rank of each layer during training via importance-based pruning:

```
┌───────────────────────────────────────────────────────┐
│               AdaLoRA: Dynamic Rank                    │
│                                                        │
│  Start: All layers at init_r (e.g., 64)               │
│                                                        │
│  Layer 1:  r=64 → (prune unimportant singular values)  │
│            → r=32  (less important for this task)       │
│                                                        │
│  Layer 2:  r=64 → r=48  (moderately important)         │
│                                                        │
│  Layer 12: r=64 → r=64  (critical — keep full rank)    │
│                                                        │
│  End: Average rank = target_r (e.g., 16)               │
│       Total params ≈ same as LoRA(r=16)                │
│       But allocated WHERE THEY MATTER                  │
└───────────────────────────────────────────────────────┘
```

---

## **6.2 How It Works**

AdaLoRA parameterizes weight changes using SVD (Singular Value Decomposition):

```
ΔW = P · Λ · Q

Where:
  P ∈ ℝ^(d×r)  — left singular vectors
  Λ ∈ ℝ^(r×r)  — diagonal matrix of singular values
  Q ∈ ℝ^(r×k)  — right singular vectors
```

During training, singular values with low importance scores are pruned, effectively reducing the rank for that layer.

---

## **6.3 Key Parameters**

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=64,           # Starting rank for all layers
    target_r=16,         # Target average rank after pruning
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    tinit=200,           # Steps before pruning starts
    tfinal=1000,         # Steps by which pruning should finish
    deltaT=10,           # Pruning frequency (every N steps)
    beta1=0.85,          # Importance score smoothing
    beta2=0.85,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
```

| Parameter | Description | Typical Value |
|---|---|---|
| `init_r` | Initial rank for all layers | 48–64 |
| `target_r` | Target average rank after pruning | 8–16 |
| `tinit` | Training steps before pruning begins | 200 |
| `tfinal` | Steps by which pruning should be complete | ~80% of total steps |
| `deltaT` | How often to prune (every N steps) | 10 |

---

## **6.4 When to Use AdaLoRA**

- **Use AdaLoRA when:** You suspect different layers need different capacities (common in deeper models), or when you want to maximize quality at a fixed parameter budget.
- **Stick with LoRA when:** Simplicity matters, or you've already tuned rank manually.
- **Trade-off:** AdaLoRA adds ~10% training overhead for the pruning schedule.

---

# **7. Adapters, Prefix Tuning & Prompt Tuning**

---

## **7.1 Adapter Layers (Houlsby et al., 2019)**

Adapters are small bottleneck modules **inserted between** existing Transformer layers. The original model is frozen; only adapters are trained.

```
┌─────────────────────────────────────────────────┐
│          ADAPTER ARCHITECTURE                    │
│                                                  │
│  Input                                           │
│    │                                             │
│    ▼                                             │
│  [Multi-Head Attention]  ← FROZEN                │
│    │                                             │
│    ▼                                             │
│  ┌──────────────────┐                            │
│  │  ADAPTER MODULE   │  ← TRAINABLE              │
│  │  ┌──────────────┐ │                           │
│  │  │ Down-project  │ │  d → m  (m << d)         │
│  │  │ NonLinearity  │ │  ReLU / GELU             │
│  │  │ Up-project    │ │  m → d                   │
│  │  └──────────────┘ │                           │
│  │  + Residual       │                           │
│  └──────────────────┘                            │
│    │                                             │
│    ▼                                             │
│  [Feed-Forward]  ← FROZEN                        │
│    │                                             │
│    ▼                                             │
│  ┌──────────────────┐                            │
│  │  ADAPTER MODULE   │  ← TRAINABLE              │
│  └──────────────────┘                            │
│    │                                             │
│    ▼                                             │
│  Output                                          │
└─────────────────────────────────────────────────┘
```

**Key properties:**
- Bottleneck dimension `m` controls capacity (typically 64–256)
- ~0.5–2% of total parameters
- Adds latency (extra sequential computation per layer)
- Can stack multiple adapters for multi-task

---

## **7.2 Prefix Tuning (Li & Liang, 2021)**

Prefix tuning prepends **trainable virtual tokens** to the Key and Value matrices in every attention layer. The model's original parameters are completely frozen.

```
┌──────────────────────────────────────────────────────────┐
│              PREFIX TUNING                                 │
│                                                           │
│  Standard Attention:                                      │
│    Q = [q₁, q₂, ..., qₙ]                                │
│    K = [k₁, k₂, ..., kₙ]                                │
│    V = [v₁, v₂, ..., vₙ]                                │
│                                                           │
│  With Prefix (prefix_length = p):                         │
│    K = [P_k₁, ..., P_kₚ, k₁, k₂, ..., kₙ]             │
│    V = [P_v₁, ..., P_vₚ, v₁, v₂, ..., vₙ]             │
│         ↑ trainable       ↑ from frozen model            │
│                                                           │
│  Trainable params: 2 × num_layers × prefix_len × d_model │
│  ~0.1% of total parameters                                │
└──────────────────────────────────────────────────────────┘
```

```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,    # Prefix length
    prefix_projection=True,   # Use MLP to reparameterize (more stable)
    encoder_hidden_size=1024, # Hidden size of reparameterization MLP
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~300K (0.004% for a 7B model)
```

---

## **7.3 Prompt Tuning (Lester et al., 2021)**

The simplest PEFT method. Prepend `k` trainable "soft" tokens to the **input embeddings** only (not every layer like prefix tuning).

```
┌──────────────────────────────────────────────────────┐
│              PROMPT TUNING                             │
│                                                       │
│  Input: [soft₁, soft₂, ..., softₖ, tok₁, tok₂, ...]│
│          ↑ trainable              ↑ actual input      │
│                                                       │
│  Only modifies input embedding layer                  │
│  Trainable params: k × d_model                        │
│  ~0.001-0.01% of total parameters                     │
│                                                       │
│  k=20, d_model=4096: 20 × 4096 = 81,920 params       │
└──────────────────────────────────────────────────────┘
```

```python
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,           # Initialize from text
    prompt_tuning_init_text="Classify the sentiment:",  # Seed text
    tokenizer_name_or_path="meta-llama/Llama-3.1-8B",
)
model = get_peft_model(model, config)
```

---

## **7.4 Comparison**

| Method | Where | Trainable % | Latency Impact | Best For |
|---|---|---|---|---|
| **Adapter** | Between layers | 0.5–2% | Moderate (sequential) | Multi-task, modular composition |
| **Prefix Tuning** | K, V in all layers | ~0.1% | Minimal | Style/domain steering |
| **Prompt Tuning** | Input embeddings only | ~0.01% | None | Quick experiments, classification |
| **LoRA** | Weight matrices | 0.3–5% | **None** (merged at inference) | Best general-purpose PEFT |

> **Why LoRA dominates:** At inference time, LoRA adapters can be **merged into the base weights** (`W' = W + A·B`), adding zero latency. Adapters and prefix tuning cannot be merged — they always add inference overhead.

---

# **8. Instruction Tuning (SFT)**

---

## **8.1 What Is SFT?**

**Supervised Fine-Tuning (SFT)** teaches a pre-trained model to follow instructions by training on (instruction, response) pairs. This is the critical step that turns a base model (e.g., Llama-3.1-8B) into a chat model (Llama-3.1-8B-Instruct).

```
Base Model: "The capital of France"  →  "is Paris. The capital of Germany is Berlin..."
                                         (continues generating)

SFT Model:  "What is the capital of France?" → "The capital of France is Paris."
                                                  (answers, then stops)
```

---

## **8.2 SFTTrainer from TRL**

The `trl` library provides `SFTTrainer`, a specialized trainer for instruction tuning:

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig

# Load model (optionally with LoRA / QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# LoRA config (optional — can also do full SFT)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Dataset in conversation format
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")
# Each example: {"messages": [{"role": "user", "content": "..."}, 
#                              {"role": "assistant", "content": "..."}]}

# SFT Config
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,             # 1e-4 to 3e-4 is the sweet spot for SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_seq_length=4096,            # 2048–4096 is typical
    bf16=True,
    packing=True,                   # Pack multiple short examples per sequence
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

# Train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,        # Pass LoRA config here
)
trainer.train()
```

---

## **8.3 Packing for Throughput**

Without packing, short examples waste GPU memory on padding:

```
Without packing (wasteful):
  Seq 1: [instruction1 | response1 | PAD PAD PAD PAD PAD PAD PAD]
  Seq 2: [instruction2 | response2 | PAD PAD PAD PAD]
  Seq 3: [instruction3 | response3 | PAD PAD PAD PAD PAD]

With packing (efficient):
  Seq 1: [instruction1 | response1 | EOS | instruction2 | response2 | EOS | inst...]
  Seq 2: [instruction3 | response3 | EOS | instruction4 | response4 | EOS | ...]
```

**Benefits:** 2–5× throughput improvement for datasets with variable-length examples.

> **Caution:** When packing, ensure attention masks prevent cross-contamination between packed examples (SFTTrainer handles this automatically with `DataCollatorForCompletionOnlyLM` or similar).

---

## **8.4 SFT Hyperparameters**

| Parameter | Recommended Range | Notes |
|---|---|---|
| Learning rate | 1e-4 to 3e-4 (LoRA), 1e-5 to 5e-5 (full FT) | Higher for PEFT methods |
| Epochs | 1–3 | More = overfitting risk; 1 epoch often enough for large datasets |
| Batch size (effective) | 16–128 | Via gradient accumulation |
| Max seq length | 2048–4096 | Match model's context window |
| Warmup | 3–10% of total steps | Prevents early instability |
| LR scheduler | Cosine or linear decay | Cosine is most common |
| Weight decay | 0.01–0.1 | Light regularization |

---

# **9. Preference Optimization: DPO & PPO RLHF**

---

## **9.1 Why Preference Optimization?**

SFT teaches the model *what* to say but doesn't teach it *what not* to say. Preference optimization teaches the model to prefer good responses over bad ones.

```
SFT:                                  Preference:
"Here's a good response" ─ learn it   "Response A is BETTER than B" ─ learn to prefer A

SFT problem: The model might generate plausible but unhelpful/harmful responses
Preference solution: Directly optimize for human preferences
```

---

## **9.2 DPO (Direct Preference Optimization)**

**DPO (Rafailov et al., 2023)** is a simpler alternative to PPO/RLHF that doesn't need a separate reward model. It directly optimizes the policy from preference pairs.

### **How DPO Works**

Given preference pairs (prompt, chosen response, rejected response):

```
DPO Loss:

  L_DPO = -E[ log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                      - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

Where:
  π_θ   = the model being trained (policy)
  π_ref = the frozen reference model (usually SFT model)
  y_w   = chosen (winning) response
  y_l   = rejected (losing) response
  β     = temperature (controls deviation from reference)
  σ     = sigmoid function
```

**Intuition:** DPO increases the probability of chosen responses and decreases the probability of rejected responses, relative to the reference model. The `β` parameter prevents the model from deviating too far from the reference.

### **DPO Code Example**

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

# Load SFT model as starting point
model = AutoModelForCausalLM.from_pretrained(
    "path/to/sft_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

# Reference model (frozen copy of SFT model)
ref_model = AutoModelForCausalLM.from_pretrained(
    "path/to/sft_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA config for DPO training
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# DPO config
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # KL penalty (0.05–0.5; lower = more deviation)
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,          # Lower LR for DPO (vs SFT)
    max_length=2048,
    max_prompt_length=512,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

# Dataset format: {"prompt": ..., "chosen": ..., "rejected": ...}
dataset = load_dataset("json", data_files="dpo_data.jsonl", split="train")

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,         # Frozen reference
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
```

---

## **9.3 PPO RLHF (Proximal Policy Optimization)**

The original RLHF approach used by OpenAI for ChatGPT.

### **PPO RLHF Pipeline**

```
┌────────────────────────────────────────────────────────────┐
│                   PPO RLHF PIPELINE                        │
│                                                            │
│  Step 1: Train SFT model (supervised fine-tuning)          │
│  ┌──────────────────────┐                                  │
│  │ Instruction → Response │ → SFT Model (π_sft)            │
│  └──────────────────────┘                                  │
│                │                                           │
│                ▼                                           │
│  Step 2: Train Reward Model                                │
│  ┌──────────────────────────────┐                          │
│  │ Human rankings: A > B > C     │ → Reward Model (R)      │
│  │ (prompt, response) → scalar   │                         │
│  └──────────────────────────────┘                          │
│                │                                           │
│                ▼                                           │
│  Step 3: PPO Optimization Loop                             │
│  ┌──────────────────────────────────────────────┐          │
│  │ For each batch of prompts:                    │          │
│  │   1. Generate responses using π_θ             │          │
│  │   2. Score with Reward Model R                │          │
│  │   3. Compute PPO objective:                   │          │
│  │      reward = R(prompt, response)             │          │
│  │              - β · KL(π_θ || π_ref)           │          │
│  │   4. Update π_θ with PPO (clipped objective)  │          │
│  └──────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

### **PPO Code Example (TRL)**

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load policy model (SFT checkpoint + value head)
model = AutoModelForCausalLMWithValueHead.from_pretrained("path/to/sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("path/to/sft_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("path/to/reward_model")

# PPO Config
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=4,
    ppo_epochs=4,
    kl_penalty="kl",              # KL penalty type
    init_kl_coef=0.2,             # Initial KL coefficient
    adap_kl_ctrl=True,            # Adaptive KL control
    cliprange=0.2,                # PPO clipping
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    # 1. Generate responses
    query_tensors = [tokenizer.encode(q, return_tensors="pt") for q in batch["prompt"]]
    response_tensors = [ppo_trainer.generate(q, max_new_tokens=256) for q in query_tensors]

    # 2. Compute rewards
    rewards = [compute_reward(reward_model, q, r) for q, r in zip(query_tensors, response_tensors)]

    # 3. PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

---

## **9.4 DPO vs PPO Comparison**

| Aspect | DPO | PPO RLHF |
|---|---|---|
| **Reward model** | Not needed (implicit) | Required (separate model) |
| **Complexity** | Simple (one training loop) | Complex (3 models: policy, ref, reward) |
| **Memory** | 2 models (policy + ref) | 3–4 models |
| **Stability** | Very stable | Sensitive to hyperparams |
| **Data** | Offline preference pairs | Can use online generation |
| **Performance** | Comparable for most tasks | Potentially stronger for complex alignment |
| **Training time** | Fast (single pass) | Slow (generate + score + update loop) |
| **Adoption** | Rapidly growing (2024–2025) | Original approach (2022–2023) |

> **When to use DPO:** Default choice. Simpler, more stable, and works well for most alignment tasks.
>
> **When to use PPO:** When you need online learning (model generates + gets feedback), or for very nuanced alignment where iterative refinement helps.

---

# **10. Evaluation & Packaging**

---

## **10.1 Offline Evaluation Metrics**

| Metric | Use Case | Formula / Description |
|---|---|---|
| **ROUGE-L** | Summarization | Longest common subsequence between generated and reference |
| **BLEU** | Translation | n-gram precision with brevity penalty |
| **BERTScore** | Semantic similarity | Cosine similarity of BERT embeddings |
| **Recall@k** | Retrieval/classification | % of relevant items in top-k |
| **Exact Match** | QA, code generation | Binary: is the output exactly correct? |
| **Perplexity** | Language modeling | `exp(-1/N · Σ log P(xᵢ))` — lower is better |
| **Pass@k** | Code generation | % of problems solved in k attempts |
| **Human rubric** | General quality | Structured 1–5 scoring on helpfulness, accuracy, style |
| **LLM-as-Judge** | Scalable evaluation | GPT-4/Claude rates outputs on criteria |

```python
# Example: Computing ROUGE-L
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = scorer.score(
    "The capital of France is Paris.",
    "Paris is the capital city of France."
)
print(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")  # ~0.72
```

---

## **10.2 Online Evaluation**

| Method | What It Measures | Implementation |
|---|---|---|
| **A/B testing** | User preference at scale | Route 50% traffic to each model |
| **Latency (P50/P99)** | Speed | Time-to-first-token, tokens/second |
| **Thumbs up/down rate** | User satisfaction | Track in production |
| **Task success rate** | End-to-end quality | Did the user achieve their goal? |
| **Hallucination rate** | Factual grounding | Manual audit or NLI-based detection |

---

## **10.3 Merging LoRA Adapters**

After training, merge LoRA weights into the base model for zero-overhead inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")

# Merge LoRA into base weights: W' = W + (alpha/r) * A * B
merged_model = model.merge_and_unload()

# Save the merged model (full-size, no adapter dependency)
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# The merged model is now a standard HF model — no PEFT dependency needed
```

> **Key insight:** `merge_and_unload()` mathematically combines `W₀ + (α/r) · A · B` into a single weight matrix. The resulting model is identical in behavior but no longer needs the PEFT library at inference time.

---

## **10.4 Serving in Production**

| Tool | Description | Key Feature |
|---|---|---|
| **vLLM** | High-throughput LLM serving | PagedAttention, continuous batching |
| **TGI (Text Generation Inference)** | HuggingFace serving framework | Flash attention, quantization, watermarking |
| **Triton Inference Server** | NVIDIA's production server | Multi-model, GPU scheduling |
| **TensorRT-LLM** | NVIDIA optimized inference | Max throughput on NVIDIA GPUs |
| **Ollama** | Local inference | Easy setup, GGUF format |

```bash
# Serve with vLLM (most popular for production)
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --dtype bfloat16
```

---

## **10.5 Post-Training Quantization for Inference**

After fine-tuning, quantize the merged model for faster/cheaper inference:

| Method | Bits | Quality Loss | Speed | Use Case |
|---|---|---|---|---|
| **AWQ** | 4-bit | Minimal (~1%) | Very fast | Production deployment |
| **GPTQ** | 4-bit | Minimal (~1%) | Fast | Production deployment |
| **GGUF** | 2–8 bit | Varies | Moderate | Local/edge deployment (llama.cpp, Ollama) |
| **FP8** | 8-bit | Negligible | Fast | H100 GPUs |
| **bitsandbytes** | 4/8-bit | Minimal | Moderate | Training (QLoRA) + inference |

> **AWQ vs GPTQ:** Both produce excellent 4-bit models. AWQ (Activation-aware Weight Quantization) is generally faster and has slightly better quality for chat models. GPTQ (Generative Pre-Training Quantization) is more established and widely supported.

---

# **11. Decision Tree: Choosing the Right Method**

---

```
START: You want to adapt an LLM to your task
│
├─ Do you have labeled training data?
│  └─ NO → Prompt Engineering / Few-shot / RAG
│  └─ YES ↓
│
├─ How much data do you have?
│  ├─ < 100 examples → Prompt Tuning or Few-shot
│  ├─ 100 – 1K examples → LoRA or QLoRA
│  ├─ 1K – 100K examples → LoRA / QLoRA / Full FT (if resources allow)
│  └─ > 100K examples → Full Fine-Tuning (best ROI)
│
├─ What's your GPU budget?
│  ├─ Consumer GPU (8-16GB) → QLoRA (only option for 7B+ models)
│  ├─ Single A100 (40-80GB) → LoRA (bf16) or QLoRA
│  ├─ Multi-GPU (2-8× A100) → Full FT with DeepSpeed/FSDP
│  └─ No GPU → Use cloud APIs or prompt engineering
│
├─ What's your model size?
│  ├─ Small (< 1B) → Full FT (affordable, best quality)
│  ├─ Medium (1B – 13B) → LoRA or QLoRA
│  └─ Large (30B – 70B+) → QLoRA (or LoRA with multi-GPU)
│
├─ Do you need to align with human preferences?
│  └─ YES → SFT first, then DPO (simple) or PPO (powerful)
│  └─ NO ↓
│
├─ Do you need zero inference overhead?
│  └─ YES → LoRA (merge_and_unload) or Full FT
│  └─ NO → Adapters, Prefix Tuning, or Prompt Tuning are fine
│
└─ Is your task simple classification or style transfer?
   └─ YES → Prompt Tuning or Prefix Tuning (minimal params)
   └─ NO → LoRA (best general-purpose PEFT)
```

---

### **Quick Reference Table**

| Method | Trainable % | GPU Memory | Quality | Inference Cost | Setup Complexity |
|---|---|---|---|---|---|
| Prompt Engineering | 0% | 0 | Baseline | High (long prompts) | Trivial |
| Prompt Tuning | ~0.01% | Minimal | Low–Medium | Low overhead | Easy |
| Prefix Tuning | ~0.1% | Low | Medium | Low overhead | Easy |
| Adapters | ~1% | Low–Medium | Good | Moderate overhead | Easy |
| **LoRA** | ~0.3–5% | Low–Medium | **Excellent** | **Zero** (merged) | **Easy** |
| **QLoRA** | ~0.3–5% | **Very Low** | **Very Good** | **Zero** (merged) | Easy |
| AdaLoRA | ~0.3–5% | Low–Medium | Excellent | Zero (merged) | Moderate |
| Full Fine-Tuning | 100% | Very High | **Best** | Zero | Complex (infra) |

---

# **12. Common Interview Questions with Strong Answers**

---

### **Q1: "How would you fine-tune a 70B model on limited GPU (e.g., single A100 80GB)?"**

**Answer:**

"I'd use **QLoRA** — that's the go-to for large models on limited hardware.

**Setup:** Load the 70B model in 4-bit NF4 quantization using `BitsAndBytesConfig`, which reduces the model from ~140GB (bf16) to ~35GB. Then attach LoRA adapters (r=16–32) targeting all attention and MLP projections. The LoRA adapters train in bf16, adding only ~100MB of trainable parameters.

**Memory breakdown:** ~35GB for the 4-bit model + ~2-3GB for LoRA params + optimizer states + activations ≈ 50-60GB. This fits on a single A100 80GB with gradient checkpointing enabled.

**If even QLoRA doesn't fit:** I'd use DeepSpeed ZeRO-3 with CPU offloading to spread the model across GPU + CPU RAM, or use multi-GPU parallelism.

**Key flags:** `gradient_checkpointing=True`, `bf16=True`, `per_device_train_batch_size=1` with `gradient_accumulation_steps=16`."

---

### **Q2: "Explain LoRA — how does it work?"**

**Answer:**

"LoRA is based on the insight that weight updates during fine-tuning have **low intrinsic rank**. Instead of learning a full-rank update ΔW, LoRA decomposes it into two small matrices:

**`W' = W₀ + (α/r) · A · B`**

Where A ∈ ℝ^(d×r) and B ∈ ℝ^(r×k), with r << min(d,k). For a 4096×4096 attention layer with r=16, that's 131K parameters instead of 16.7M — a 99.2% reduction.

**Key details:**
- B is initialized to zero, so the model starts identical to the pre-trained model
- α/r controls the effective learning rate for LoRA params
- At inference, LoRA is merged: W' = W₀ + A·B — zero overhead
- I target q, k, v, output projections and MLP layers

In my experience with CLIP LoRA fine-tuning for medical imaging, I got 95%+ of full fine-tuning performance while training 100× fewer parameters."

---

### **Q3: "LoRA vs full fine-tuning — trade-offs?"**

**Answer:**

"There are three main trade-offs:

1. **Quality ceiling:** Full FT can reach marginally higher performance (~1-5% better on benchmarks), because it has more capacity to learn. But in practice, LoRA matches full FT for most tasks.

2. **Efficiency:** LoRA is dramatically more efficient:
   - ~99% fewer trainable parameters
   - 3-5× less memory → train on 1 GPU instead of 4-8
   - 3-5× faster training
   - Tiny checkpoint sizes (~50MB vs ~14GB for a 7B model)

3. **Catastrophic forgetting:** LoRA is naturally more resistant because the base model is frozen. Full FT can overwrite general capabilities if you're not careful with learning rate and data mixing.

**My rule of thumb:** Start with LoRA. Only move to full FT if (a) you have >100K high-quality examples, (b) the domain is very different from pre-training, and (c) you have the GPU budget."

---

### **Q4: "What is QLoRA and when would you use it?"**

**Answer:**

"QLoRA combines LoRA with 4-bit quantization. The base model weights are loaded in NF4 (4-bit NormalFloat) — an information-theoretically optimal format for normally-distributed weights. LoRA adapters still train in bf16.

**Three key innovations:** (1) NF4 quantization type, (2) double quantization — quantizing the quantization constants saves ~0.37 bits/param, (3) paged optimizers that offload to CPU on OOM.

**When to use it:** When you're GPU-constrained. QLoRA lets you fine-tune a 7B model on 6GB VRAM, or a 70B model on a single A100. The quality trade-off is minimal — typically 98-99% of full LoRA performance.

**When NOT to use it:** If you have plenty of VRAM, use regular LoRA in bf16 — slightly better quality and faster training (no dequantization overhead)."

---

### **Q5: "When would you use RAG vs fine-tuning?"**

**Answer:**

"They solve different problems and are often complementary:

**Use RAG when:**
- The knowledge base changes frequently (fine-tuned knowledge is static)
- You need citations/grounding (RAG retrieves specific documents)
- You need factual accuracy on a large corpus
- You don't have labeled training data

**Use fine-tuning when:**
- You need specific output format/style/behavior
- You need to reduce latency (no retrieval step)
- You need to reduce per-query cost (shorter prompts)
- The domain requires reasoning patterns not in the base model

**In practice, combine both:** Fine-tune for behavior (tool use, format, style) and use RAG for knowledge (facts, documents). For example, fine-tune a model to generate structured medical reports, then use RAG to ground it in patient records."

---

### **Q6: "DPO vs PPO — when would you choose each?"**

**Answer:**

"**DPO is my default** for preference alignment. It's simpler (single training loop, no reward model), more stable (less hyperparameter sensitivity), and matches PPO on most benchmarks.

**I'd choose PPO when:**
1. I need **online learning** — where the model generates responses and gets real-time feedback. DPO is offline-only.
2. The reward signal is **complex or multi-dimensional** — PPO can handle a sophisticated reward model that captures nuances DPO's implicit reward can't.
3. I need **iterative self-improvement** — PPO's generate-score-update loop enables the model to learn from its own mistakes.

**Practically:** DPO for 90% of use cases. PPO when you have the engineering resources and need maximum alignment quality (e.g., ChatGPT-level systems)."

---

### **Q7: "Walk me through your fine-tuning pipeline end-to-end."**

**Answer:**

"Here's my standard pipeline:

1. **Data preparation:** Collect and clean instruction/response pairs. Deduplicate, filter for quality, cap length at max_seq_len. Apply the model's chat template using `apply_chat_template`. Split 90/10 train/eval.

2. **Method selection:** For most tasks, I start with QLoRA (fast iteration, low cost). If quality isn't sufficient, I upgrade to LoRA with bf16, then full FT as a last resort.

3. **SFT phase:** Train with SFTTrainer from TRL. LoRA r=16, lr=2e-4, cosine schedule, 1-3 epochs. Use packing for throughput. Monitor training loss + eval loss on wandb.

4. **Preference alignment (if needed):** Collect or generate preference pairs. Run DPO with β=0.1, lr=5e-5, 1 epoch. Evaluate with LLM-as-judge.

5. **Evaluation:** Compute ROUGE/BLEU/BERTScore on held-out set. Run human eval on 100-200 examples. Check for regressions on general benchmarks.

6. **Packaging:** Merge LoRA with `merge_and_unload()`. Quantize to AWQ 4-bit for serving. Deploy with vLLM behind an OpenAI-compatible API.

7. **Monitoring:** A/B test against the previous model. Track latency, thumbs-up rate, and hallucination rate."

---

### **Q8: "How do you prevent catastrophic forgetting during fine-tuning?"**

**Answer:**

"Catastrophic forgetting is when the model loses its general capabilities while learning the new task. Here's how I mitigate it:

1. **Use PEFT (LoRA/QLoRA):** The base model is frozen — it's architecturally impossible to overwrite pre-trained knowledge. This is the single most effective defense.

2. **Lower learning rate:** For full FT, use 1e-5 to 5e-5 (10-100× lower than PEFT). High LR = aggressive overwriting.

3. **Data mixing:** Blend task-specific data with general instruction-following data. Common ratio: 80% task + 20% general.

4. **Fewer epochs:** 1-3 epochs max. More epochs = more forgetting.

5. **Regularization:** Weight decay (0.01-0.1), dropout, early stopping based on eval set.

6. **Evaluation on general benchmarks:** Track performance on MMLU, HellaSwag, etc. alongside your task metrics. If general scores drop, you're overfitting.

7. **Elastic Weight Consolidation (advanced):** Penalize changes to parameters that are important for prior tasks. Rarely needed with LoRA."

---

### **Q9: "You mentioned LoRA fine-tuning CLIP. Walk me through that."**

**Answer:**

"In my radiology project, I needed to adapt CLIP for medical image classification — CheXpert chest X-ray findings.

**Challenge:** CLIP is trained on natural images + web text. Medical images are distribution-shifted — different visual features, specialized terminology.

**Approach:**
1. Applied LoRA (r=8, α=16) to the **vision encoder only** — specifically q_proj, k_proj, v_proj, out_proj in the ViT attention layers.
2. Kept the **text encoder frozen** — CLIP's text understanding of medical terms was already decent.
3. Added `modules_to_save=['visual_projection']` to also train the projection head that maps vision features to the shared embedding space.
4. Trained on (X-ray image, finding label) pairs using contrastive loss.

**Result:** ~0.5% of parameters trained. Matched full vision encoder fine-tuning performance on CheXpert AUC metrics. Training took ~2 hours on a single A100 instead of ~8 hours for full FT.

**Key insight:** For multimodal models, you often only need to adapt one modality. Freeze the side that already understands your data well."

---

# **13. Key Takeaways**

---

```
┌──────────────────────────────────────────────────────────────────────┐
│                     KEY TAKEAWAYS FOR INTERVIEWS                     │
│                                                                      │
│  1. LoRA is the default PEFT method — learn it deeply                │
│     W' = W₀ + (α/r) · A · B                                        │
│     Zero inference overhead via merge_and_unload()                   │
│                                                                      │
│  2. QLoRA = LoRA + 4-bit NF4 quantization                           │
│     Fine-tune 7B on 6GB VRAM, 70B on one A100                       │
│                                                                      │
│  3. The pipeline: Data → SFT (LoRA/QLoRA) → DPO → Eval → Merge     │
│                                                                      │
│  4. RAG for knowledge, fine-tuning for behavior                      │
│     Best systems combine both                                        │
│                                                                      │
│  5. DPO > PPO for simplicity and stability                           │
│     Only use PPO when you need online learning                       │
│                                                                      │
│  6. Always use the base model's tokenizer + chat template            │
│     Mismatched templates = silent failure                             │
│                                                                      │
│  7. Start simple, scale up:                                          │
│     Prompt Eng → LoRA → QLoRA → Full FT                             │
│     Don't reach for full FT unless you've proven you need it         │
│                                                                      │
│  8. Evaluation matters more than training:                           │
│     Offline metrics + human eval + A/B tests + monitoring            │
│                                                                      │
│  9. Serve efficiently:                                               │
│     merge_and_unload() → AWQ/GPTQ quantize → vLLM/TGI              │
│                                                                      │
│  10. Catastrophic forgetting is solved by PEFT                       │
│      For full FT: low LR + data mixing + few epochs                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

### **Papers to Reference in Interviews**

| Paper | Year | Key Contribution |
|---|---|---|
| **LoRA** (Hu et al.) | 2021 | Low-rank adaptation of large language models |
| **QLoRA** (Dettmers et al.) | 2023 | 4-bit quantized LoRA — fine-tune 65B on single GPU |
| **AdaLoRA** (Zhang et al.) | 2023 | Adaptive rank allocation across layers |
| **DPO** (Rafailov et al.) | 2023 | Direct preference optimization without reward model |
| **InstructGPT** (Ouyang et al.) | 2022 | RLHF pipeline for aligning LLMs |
| **Scaling Laws** (Kaplan et al.) | 2020 | How model/data/compute scale affects performance |
| **Intrinsic Dimensionality** (Aghajanyan et al.) | 2020 | Theoretical basis for why LoRA works |
| **Prefix Tuning** (Li & Liang) | 2021 | Trainable prefixes for K/V in attention |
| **Prompt Tuning** (Lester et al.) | 2021 | Soft prompts in input embedding space |
| **Adapters** (Houlsby et al.) | 2019 | Bottleneck adapter modules between layers |

---

*Last updated: February 2026*
