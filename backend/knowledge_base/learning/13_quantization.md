# Model Quantization — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LLMs, Model Compression, Quantization, Edge Deployment

---

# Table of Contents

1. [What Is Quantization?](#1-what-is-quantization)
2. [Why Quantization Matters](#2-why-quantization-matters)
3. [The Quantization Formula](#3-the-quantization-formula)
4. [Bit-Width Options](#4-bit-width-options)
5. [Quantization Methods — Deep Dive](#5-quantization-methods--deep-dive)
6. [Architecture Integration — Where Quantization Lives](#6-architecture-integration--where-quantization-lives)
7. [Practical Tools & Code](#7-practical-tools--code)
8. [Memory Examples — Real Numbers](#8-memory-examples--real-numbers)
9. [Production Pipeline](#9-production-pipeline)
10. [When NOT to Quantize](#10-when-not-to-quantize)
11. [Common Interview Questions with Strong Answers](#11-common-interview-questions-with-strong-answers)
12. [Key Takeaways](#12-key-takeaways)

---

# **1. What Is Quantization?**

---

## **1.1 Core Concept**

Quantization is the process of **reducing the numerical precision** of a neural network's weights (and optionally activations) — for example, converting 32-bit floating-point numbers to 8-bit or 4-bit integers — to shrink model size, speed up inference, and lower resource consumption.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION — THE BIG PICTURE                       │
│                                                                         │
│   ORIGINAL MODEL (FP16)              QUANTIZED MODEL (INT4)            │
│   ┌──────────────────────┐           ┌──────────────────────┐          │
│   │  Weight: 0.07421875  │           │  Weight: 7 (INT4)    │          │
│   │  Stored as: 16 bits  │ ──────►   │  Stored as: 4 bits   │          │
│   │  Per-param: 2 bytes  │           │  Per-param: 0.5 byte │          │
│   │  7B model ≈ 13 GB    │           │  7B model ≈ 3.5 GB   │          │
│   └──────────────────────┘           └──────────────────────┘          │
│                                                                         │
│   Analogy:  JPEG compresses images by discarding imperceptible detail.  │
│             Quantization compresses models by discarding imperceptible   │
│             weight precision.                                           │
│                                                                         │
│             It's a "lossy ZIP for neural networks."                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Neural networks are remarkably resilient to precision reduction. Most weight values cluster around zero, and the model's function depends on the *relative relationships* between weights, not their exact floating-point values. This redundancy is what quantization exploits.

---

## **1.2 What Gets Quantized?**

| Component | Quantizable? | Notes |
|---|---|---|
| **Weights** (W) | Yes — always | Primary target. Static after training, easy to calibrate. |
| **Activations** (A) | Yes — often | Dynamic per-input. Requires calibration data or dynamic range tracking. |
| **KV-Cache** | Yes — increasingly | Large memory consumer during generation. FP8/INT8 KV-cache is common. |
| **Gradients** | Rarely | Only in mixed-precision training. Not traditional "quantization." |
| **Optimizer states** | Rarely | 8-bit Adam (bitsandbytes) quantizes momentum/variance. |

```
┌─────────────────────────────────────────────────────────────────────┐
│              WHAT GETS QUANTIZED IN A TRANSFORMER                    │
│                                                                     │
│   Input tokens                                                      │
│       │                                                             │
│       ▼                                                             │
│   ┌────────────────┐                                                │
│   │  Embedding      │ ◄── Weights: can be quantized (INT8)          │
│   └───────┬────────┘                                                │
│           ▼                                                         │
│   ┌────────────────┐                                                │
│   │  QKV Linear    │ ◄── Weights: primary target (INT4/INT8)       │
│   │  Projections   │     Activations: can be quantized              │
│   └───────┬────────┘                                                │
│           ▼                                                         │
│   ┌────────────────┐                                                │
│   │  Attention     │ ◄── KV-Cache: quantizable (FP8/INT8)          │
│   │  (Softmax)     │     Scores kept in higher precision            │
│   └───────┬────────┘                                                │
│           ▼                                                         │
│   ┌────────────────┐                                                │
│   │  MLP / FFN     │ ◄── Weights: primary target (INT4/INT8)       │
│   │  (Gate, Up,    │     Largest weight matrices in transformer     │
│   │   Down proj)   │                                                │
│   └───────┬────────┘                                                │
│           ▼                                                         │
│   ┌────────────────┐                                                │
│   │  LM Head       │ ◄── Weights: sometimes quantized               │
│   └───────┬────────┘                                                │
│           ▼                                                         │
│     Output logits                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** When asked "what is quantization?", lead with the lossy-ZIP analogy, then immediately pivot to *what* gets quantized (weights, activations, KV-cache) and *why* the model tolerates it (weight redundancy, clustering near zero). This shows you understand both the concept and the mechanism.

---

## **1.3 Symmetric vs Asymmetric Quantization**

| Property | Symmetric | Asymmetric |
|---|---|---|
| **Zero point** | Always 0 | Computed (non-zero) |
| **Range** | \[-max, +max\] | \[min, max\] |
| **Best for** | Weights (centered near 0) | Activations (often positive, e.g., after ReLU) |
| **Speed** | Faster (no zero-point subtraction) | Slightly slower |
| **Formula** | x̂ = scale × int_x | x̂ = scale × (int_x − zero_point) |

```
Symmetric quantization:                 Asymmetric quantization:

    FP range: [-3.0 ... +3.0]              FP range: [0.0 ... 6.0]
         │                                       │
         ▼                                       ▼
    INT8 range: [-128 ... +127]             INT8 range: [0 ... 255]
    zero_point = 0                          zero_point = 0  (maps to FP 0.0)
    scale = 3.0 / 127 ≈ 0.0236             scale = 6.0 / 255 ≈ 0.0235
```

---

# **2. Why Quantization Matters**

---

## **2.1 The Four Pillars**

```
┌─────────────────────────────────────────────────────────────────────────┐
│               WHY QUANTIZATION MATTERS — FOUR PILLARS                   │
│                                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│   │    COST       │  │    SPEED     │  │   ENERGY     │  │  EDGE    │  │
│   │              │  │              │  │              │  │          │  │
│   │  Cheaper     │  │  Fewer bytes │  │  Lower power │  │  Fits on │  │
│   │  GPUs        │  │  per matmul  │  │  draw per    │  │  phones, │  │
│   │              │  │              │  │  token       │  │  laptops │  │
│   │  A100 80GB → │  │  Smaller     │  │              │  │  RPi     │  │
│   │  fits on     │  │  cache lines │  │  Green AI    │  │          │  │
│   │  T4 16GB     │  │              │  │              │  │  No cloud│  │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │
│                                                                         │
│         $$$                2-4×               🌍               📱        │
│      savings            speedup            savings          portable    │
└─────────────────────────────────────────────────────────────────────────┘
```

## **2.2 Detailed Impact Breakdown**

| Pillar | FP16 Baseline | INT8 Quantized | INT4 Quantized |
|---|---|---|---|
| **Memory** (7B model) | ~13 GB | ~6.5 GB | ~3.5 GB |
| **GPU requirement** | A100 40GB | T4 16GB | T4 16GB (with room) |
| **GPU cost/hr** (cloud) | ~$3.50/hr | ~$0.70/hr | ~$0.70/hr |
| **Throughput** | 1× baseline | 1.5–2× | 2–4× |
| **Latency (TTFT)** | 1× baseline | 0.6–0.8× | 0.4–0.6× |
| **Energy/token** | 1× baseline | ~0.5× | ~0.3× |
| **Accuracy** | Baseline | ~99% of FP16 | 95–98% of FP16 |

## **2.3 Why Speed Improves (Not Just Memory)**

```
┌─────────────────────────────────────────────────────────────────────────┐
│            WHY SMALLER BITS = FASTER INFERENCE                          │
│                                                                         │
│   1. MEMORY BANDWIDTH IS THE BOTTLENECK                                │
│      ─────────────────────────────────                                  │
│      LLM inference is memory-bound (not compute-bound).                │
│      The GPU spends most time waiting for weights to arrive             │
│      from HBM (High Bandwidth Memory) → SRAM (on-chip cache).         │
│                                                                         │
│      FP16: Load 2 bytes per weight  →  INT4: Load 0.5 bytes            │
│      = 4× more weights per memory transaction                          │
│      = 4× better utilization of memory bandwidth                       │
│                                                                         │
│   2. CACHE EFFICIENCY                                                  │
│      ────────────────                                                   │
│      Smaller weights → more fit in L2/SRAM cache                       │
│      → fewer cache misses → fewer stalls                               │
│                                                                         │
│   3. TENSOR CORE SUPPORT                                               │
│      ───────────────────                                                │
│      NVIDIA GPUs have INT8/INT4 Tensor Cores that execute              │
│      more operations per clock cycle than FP16 cores.                  │
│      A100: INT8 = 2× TOPS vs FP16                                     │
│      H100: FP8 = 2× TOPS vs FP16                                      │
│                                                                         │
│   4. BATCH SIZE SCALING                                                │
│      ──────────────────                                                 │
│      Smaller model → fits larger batch sizes in GPU memory             │
│      → better GPU utilization → higher throughput                      │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Many candidates say "quantization saves memory." Go further: explain *why* it also speeds up inference (memory bandwidth bottleneck, cache line efficiency, dedicated integer tensor cores). This separates you from the crowd.

---

# **3. The Quantization Formula**

---

## **3.1 Core Math**

The fundamental quantization/dequantization equations:

**Quantize (float → int):**

$$
x_{int} = \text{round}\!\left(\frac{x_{float}}{\text{scale}}\right) + \text{zero\_point}
$$

**Dequantize (int → float):**

$$
\hat{x} = \text{scale} \times (x_{int} - \text{zero\_point})
$$

**Where:**

$$
\text{scale} = \frac{x_{max} - x_{min}}{2^b - 1}
$$

$$
\text{zero\_point} = \text{round}\!\left(-\frac{x_{min}}{\text{scale}}\right)
$$

- `b` = target bit-width (e.g., 8 for INT8, 4 for INT4)
- `x_max`, `x_min` = the observed (or calibrated) range of the floating-point values
- `scale` = the step size between adjacent quantized levels
- `zero_point` = the integer value that maps to floating-point zero

---

## **3.2 Worked Example**

```
Example: Quantize the value 0.0723 to INT8 (unsigned, 0-255)

Given: calibrated range x_min = -1.0, x_max = 1.0, bit-width b = 8

Step 1: Compute scale
    scale = (x_max - x_min) / (2^b - 1)
          = (1.0 - (-1.0)) / (256 - 1)
          = 2.0 / 255
          = 0.00784

Step 2: Compute zero_point
    zero_point = round(-x_min / scale)
               = round(1.0 / 0.00784)
               = round(127.55)
               = 128

Step 3: Quantize
    x_int = round(0.0723 / 0.00784) + 128
          = round(9.22) + 128
          = 9 + 128
          = 137

Step 4: Dequantize (verify)
    x̂ = 0.00784 × (137 - 128)
      = 0.00784 × 9
      = 0.07056

    Quantization error: |0.0723 - 0.07056| = 0.00174  (< 0.25% of range)
```

---

## **3.3 Per-Tensor vs Per-Channel vs Group Quantization**

| Granularity | Scale/ZP | Accuracy | Overhead | Used By |
|---|---|---|---|---|
| **Per-tensor** | 1 scale for entire weight matrix | Lowest | Minimal | Basic PTQ, TFLite |
| **Per-channel** | 1 scale per output channel | Medium | Low | PyTorch default, INT8 PTQ |
| **Per-group** (g=128) | 1 scale per group of 128 weights | Highest | Moderate | GPTQ, AWQ, NF4 |
| **Per-element** | 1 scale per weight | Perfect | Too high | Not practical |

```
Per-tensor:     1 scale for [4096 × 4096] matrix = 1 extra float
Per-channel:    1 scale per row  = 4096 extra floats
Per-group:      1 scale per 128 weights = 4096 × 4096 / 128 = 131,072 extra floats (FP16)
                Overhead: 131,072 × 2 bytes = 256 KB per matrix (acceptable)
```

> **Interview tip:** The "group size" parameter (commonly g=128) is a critical design choice. Smaller groups = better accuracy but more overhead. GPTQ and AWQ both default to group_size=128 as a sweet spot. If asked "how does GPTQ maintain accuracy at 4-bit?", group quantization is a big part of the answer.

---

# **4. Bit-Width Options**

---

## **4.1 The Precision Ladder**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  THE PRECISION LADDER                                    │
│                                                                         │
│   FP32  ████████████████████████████████  32 bits   (4 bytes/param)    │
│         Full training precision                                         │
│                                                                         │
│   FP16  ████████████████                  16 bits   (2 bytes/param)    │
│         Standard inference baseline                                     │
│                                                                         │
│   BF16  ████████████████                  16 bits   (2 bytes/param)    │
│         Same range as FP32, less precision                              │
│                                                                         │
│   FP8   ████████                           8 bits   (1 byte/param)     │
│         H100 native, training + inference                               │
│                                                                         │
│   INT8  ████████                           8 bits   (1 byte/param)     │
│         256 levels, workhorse of PTQ                                    │
│                                                                         │
│   INT4  ████                               4 bits   (0.5 byte/param)   │
│         16 levels, aggressive but works                                 │
│                                                                         │
│   NF4   ████                               4 bits   (0.5 byte/param)   │
│         Quantile-based, optimal for normals                             │
│                                                                         │
│   INT2  ██                                 2 bits   (0.25 byte/param)  │
│         4 levels, research-only                                         │
│                                                                         │
│   1-bit █                                  1 bit    (0.125 byte/param) │
│         Binary: {-1, +1}, BitNet                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## **4.2 Comparison Table**

| Bit-Width | Bytes/Param | Size vs FP16 | Levels | Typical Speedup | Quality Loss | When to Pick |
|---|---|---|---|---|---|---|
| **FP32** | 4.0 | 2.0× larger | ~4 billion | 0.5× (slower) | None | Training only |
| **FP16** | 2.0 | 1.0× (baseline) | 65,536 | 1.0× (baseline) | Negligible | Default inference |
| **BF16** | 2.0 | 1.0× | 65,536 | 1.0× | Negligible | Training + inference (Ampere+) |
| **FP8 (E4M3)** | 1.0 | 0.5× | 256 | 1.5–2.0× | <1% | H100/H200 inference |
| **INT8** | 1.0 | 0.5× | 256 | 1.5–2.0× | <1% perplexity | Production serving, broad GPU support |
| **INT4** | 0.5 | 0.25× | 16 | 2.0–4.0× | 1–3% perplexity | Memory-constrained, consumer GPUs |
| **NF4** | 0.5 | 0.25× | 16 (non-uniform) | 2.0–3.5× | <1% (for fine-tuning) | QLoRA fine-tuning |
| **INT2** | 0.25 | 0.125× | 4 | 3–5× (theoretical) | 5–15% | Research only |
| **1-bit** | 0.125 | 0.0625× | 2 | 5–10× (theoretical) | Significant | BitNet research |

## **4.3 FP16 vs BF16 — What's the Difference?**

```
FP16 (Half Precision):           BF16 (Brain Floating Point):
┌─────────────────────────┐      ┌─────────────────────────┐
│ 1 sign │ 5 exp │ 10 man │      │ 1 sign │ 8 exp │ 7 man  │
└─────────────────────────┘      └─────────────────────────┘
  Range: ±65,504                   Range: ±3.4 × 10^38
  Precision: ~3.3 decimal digits   Precision: ~2.4 decimal digits

  ✓ More precision (10-bit mantissa)   ✓ Same range as FP32 (8-bit exponent)
  ✗ Limited range → overflow risk      ✗ Less precision
  ✗ Needs loss scaling in training     ✓ Drop-in replacement for FP32 training
```

> **Interview tip:** BF16 was designed by Google Brain specifically for deep learning. It sacrifices precision for range — and in neural networks, range matters more than precision because activations can vary by orders of magnitude. If asked "FP16 or BF16?", say BF16 for training (no loss scaling needed), FP16 for inference (more precision, range is known).

---

# **5. Quantization Methods — Deep Dive**

---

## **5.1 Uniform Post-Training Quantization (PTQ)**

The simplest and most widely-used quantization approach. The model is trained normally in full precision, then weights are quantized **after training** using a small calibration dataset to determine the value ranges.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  POST-TRAINING QUANTIZATION PIPELINE                    │
│                                                                         │
│   ┌───────────┐    ┌────────────┐    ┌───────────┐    ┌─────────────┐  │
│   │  Trained   │    │ Calibration│    │ Compute   │    │  Quantized  │  │
│   │  FP16     │───►│  Dataset   │───►│ Scale &   │───►│  INT8/INT4  │  │
│   │  Model    │    │ (100-1000  │    │ Zero-Point│    │  Model      │  │
│   │           │    │  samples)  │    │ per layer │    │             │  │
│   └───────────┘    └────────────┘    └───────────┘    └─────────────┘  │
│                                                                         │
│   Time: minutes (no retraining)                                        │
│   Data: 100-1000 calibration samples (unlabeled)                       │
│   Accuracy: ~99% of FP16 at INT8, ~95-98% at INT4                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**How It Works:**

1. **Freeze** the trained FP16/FP32 model
2. **Run** calibration data through the model, recording min/max (or percentile) of weights and activations per layer
3. **Compute** scale and zero_point for each tensor using the recorded ranges
4. **Map** every floating-point value to its nearest integer bucket
5. **Replace** FP linear layers with INT linear layers + stored scale factors

**Calibration Strategies:**

| Strategy | Method | Pro | Con |
|---|---|---|---|
| **Min-Max** | Use observed min/max | Simple | Outlier-sensitive |
| **Percentile** | Use 99.9th percentile | Robust to outliers | Clips extreme values |
| **Entropy (KL)** | Minimize KL divergence between FP and INT distributions | Best accuracy | Slower calibration |
| **MSE** | Minimize reconstruction error | Good balance | Moderate cost |

```python
# PyTorch PTQ — Static Quantization Example
import torch
from torch.quantization import quantize_dynamic, prepare, convert

# OPTION 1: Dynamic quantization (weights only, simplest)
model_fp32 = load_my_model()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},      # Quantize all Linear layers
    dtype=torch.qint8        # Target INT8
)

# OPTION 2: Static quantization (weights + activations)
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
model_prepared = prepare(model_fp32)

# Run calibration data through the model
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

model_int8 = convert(model_prepared)
print(f"Size reduction: {model_size(model_fp32)/model_size(model_int8):.1f}×")
```

---

## **5.2 Quantization-Aware Training (QAT)**

QAT inserts **fake quantization nodes** during training so the model *learns* to be robust to low-precision arithmetic. The forward pass simulates quantization (round to integer, dequantize back), but gradients flow through using the **Straight-Through Estimator (STE)** since rounding is non-differentiable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              QUANTIZATION-AWARE TRAINING (QAT)                          │
│                                                                         │
│   Forward Pass:                                                        │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    │
│   │  FP32    │     │  Fake    │     │  FP32    │     │  Loss    │    │
│   │  Weights │────►│ Quantize │────►│  Matmul  │────►│          │    │
│   │          │     │  + Deq.  │     │          │     │          │    │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘    │
│                         │                                              │
│                         │  Simulated INT8 precision                    │
│                         │  (round to grid, add noise)                  │
│                                                                         │
│   Backward Pass:                                                       │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                      │
│   │  FP32    │     │  STE     │     │  FP32    │                      │
│   │  Weight  │◄────│ (pass    │◄────│  Grad    │                      │
│   │  Update  │     │ gradient │     │          │                      │
│   │          │     │ through) │     │          │                      │
│   └──────────┘     └──────────┘     └──────────┘                      │
│                                                                         │
│   Key: Model learns to place weights on the quantization grid          │
│        → lower quantization error when actually deployed               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Straight-Through Estimator (STE):**

$$
\frac{\partial \text{round}(x)}{\partial x} \approx 1 \quad (\text{in backprop, treat round as identity})
$$

```python
# PyTorch QAT Example
import torch
from torch.quantization import prepare_qat, convert

model = load_pretrained_model()
model.train()

# Set QAT config
model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
model_qat = prepare_qat(model)

# Train (fine-tune) with fake-quant nodes active
optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-5)
for epoch in range(3):
    for batch in train_loader:
        loss = model_qat(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Convert fake-quant model to true INT8
model_int8 = convert(model_qat.eval())
```

**PTQ vs QAT:**

| Aspect | PTQ (Post-Training) | QAT (Quantization-Aware) |
|---|---|---|
| **Training required** | No (minutes to run calibration) | Yes (fine-tune for epochs) |
| **Calibration data** | 100–1000 unlabeled samples | Full training dataset |
| **Accuracy at INT8** | 98–99% of FP16 | 99–100% of FP16 |
| **Accuracy at INT4** | 93–97% of FP16 | 97–99% of FP16 |
| **Cost** | Low (just calibration) | High (GPU hours for training) |
| **When to use** | Good enough accuracy, fast turnaround | Maximum accuracy at low bit-widths |
| **Common in LLMs?** | Yes (dominant for LLMs) | Less common (too expensive to retrain) |

---

## **5.3 GPTQ — GPU-Optimized Post-Training Quantization**

GPTQ (Frantar et al., 2022) is a **one-shot, layer-wise** PTQ method based on Optimal Brain Quantization (OBQ). It quantizes weights **one layer at a time**, using a second-order (Hessian-based) optimization to minimize the output reconstruction error.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GPTQ ALGORITHM                                   │
│                                                                         │
│   For each layer (sequentially):                                       │
│                                                                         │
│   1. Collect Hessian information using calibration data                 │
│      H = 2 × X^T × X   (where X = layer inputs from calibration)      │
│                                                                         │
│   2. For each column group (block of 128 weights):                     │
│      a. Find the quantized value that minimizes:                       │
│         ‖ W_full × X  −  W_quant × X ‖²                               │
│      b. Quantize weights in this block                                 │
│      c. Compensate remaining weights to absorb the error               │
│         (error correction / weight update using Hessian inverse)       │
│                                                                         │
│   3. Move to next layer (feed quantized outputs forward)               │
│                                                                         │
│   ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│   │ Layer 1     │──►│ Layer 2  │──►│ Layer 3  │──►│  ...     │       │
│   │ Quantize +  │   │ Quantize │   │ Quantize │   │          │       │
│   │ Compensate  │   │          │   │          │   │          │       │
│   └─────────────┘   └──────────┘   └──────────┘   └──────────┘       │
│                                                                         │
│   Key insight: By adjusting remaining FP weights to compensate for     │
│   each quantized column's error, GPTQ achieves much better accuracy   │
│   than naive round-to-nearest at INT4.                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Group quantization** (group_size=128): each group of 128 weights shares a scale/zero-point
- **Hessian-based error compensation**: adjusts remaining weights to absorb quantization error
- **One-shot**: runs in ~4 hours for a 175B model on a single GPU
- **INT4 with near-INT8 accuracy**: the error compensation is why GPTQ works so well at 4-bit

```python
# GPTQ with auto-gptq library
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 1. Configure
quantize_config = BaseQuantizeConfig(
    bits=4,               # INT4 quantization
    group_size=128,       # Per-group quantization
    desc_act=True,        # Activation-order quantization (slower but better)
    damp_percent=0.01,    # Dampening for Hessian inverse (numerical stability)
)

# 2. Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)

# 3. Quantize (requires calibration data)
calibration_data = [tokenizer(text, return_tensors="pt") for text in calibration_texts]
model.quantize(calibration_data)

# 4. Save quantized model
model.save_quantized("./llama2-7b-gptq-4bit")

# 5. Load for inference
model = AutoGPTQForCausalLM.from_quantized(
    "./llama2-7b-gptq-4bit",
    device="cuda:0",
    use_triton=True,      # Use Triton kernels for fast inference
)
```

---

## **5.4 AWQ — Activation-Aware Weight Quantization**

AWQ (Lin et al., 2023) observes that **not all weights are equally important** — a small fraction (0.1–1%) of weights corresponding to channels with large activation magnitudes are **salient** and dominate model quality. AWQ protects these weights.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   AWQ — KEY INSIGHT                                      │
│                                                                         │
│   Observation: Weight importance ∝ Activation magnitude                │
│                                                                         │
│   Activation magnitudes per channel:                                   │
│   Channel:    1    2    3    4    5    6    7    8    9   10            │
│   Magnitude:  0.1  0.2  0.1  8.5  0.3  0.1  0.2  0.1  7.2  0.2      │
│                            ↑ salient                   ↑ salient       │
│                                                                         │
│   Naive INT4:  quantize all channels equally                           │
│                → salient channels get same noisy quantization          │
│                → BIG accuracy drop                                     │
│                                                                         │
│   AWQ approach: scale UP salient weight channels before quantization   │
│                 → their quantization error shrinks                     │
│                 → scale DOWN during inference (fused with dequant)     │
│                 → negligible overhead, significant accuracy gain       │
│                                                                         │
│   ┌───────────────────────────┐                                        │
│   │   W' = W × diag(s)       │  ← scale salient channels up          │
│   │   Q(W') has less error   │  ← quantization error reduced          │
│   │   Y = Q(W') × (X / s)   │  ← compensate in activation side       │
│   └───────────────────────────┘                                        │
│                                                                         │
│   Finding optimal s:                                                   │
│     s* = argmin_s ‖ Q(W·diag(s)) · (diag(s)⁻¹·X) − W·X ‖²          │
│     Solved with simple grid search per channel                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **No retraining** needed (PTQ method)
- **Faster** than GPTQ (no Hessian computation)
- **Better generalization**: GPTQ is optimized for the calibration set; AWQ is more robust across tasks
- **Hardware-friendly**: the per-channel scaling fuses into existing dequantization kernels

```python
# AWQ with autoawq library
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. Load model
model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Configure and quantize
quant_config = {
    "zero_point": True,       # Asymmetric quantization
    "q_group_size": 128,      # Group size
    "w_bit": 4,               # 4-bit weights
    "version": "GEMM",        # Kernel version (GEMM or GEMV)
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",     # Built-in calibration dataset
)

# 3. Save
model.save_quantized("./llama2-7b-awq-4bit")
```

---

## **5.5 NF4 — NormalFloat4 (QLoRA's Method)**

NF4 (Dettmers et al., 2023) is the quantization format behind **QLoRA**. It is based on the observation that pretrained neural network weights follow an approximately **normal distribution**. Instead of uniform quantization bins, NF4 uses **quantile-based bins** that are optimal for normally-distributed data.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   NF4 — NORMALFLOAT4                                    │
│                                                                         │
│   Uniform INT4 (16 evenly-spaced bins):                                │
│   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤                   │
│   Most weights are near zero → many bins wasted on sparse tails        │
│                                                                         │
│   NF4 (16 quantile-based bins):                                        │
│   ├────┼───┼──┼─┼┼┼┼┼┼─┼──┼───┼────┤                                  │
│                  ▲▲▲▲▲▲                                                │
│                  Dense bins near zero where most weights live           │
│                                                                         │
│   How NF4 works:                                                       │
│   1. Assume weights ~ N(0, σ²)                                         │
│   2. Normalize: w_norm = w / absmax(w)   → range [-1, 1]              │
│   3. Compute 16 quantile breakpoints of N(0,1) in [-1, 1]             │
│   4. Map each weight to nearest quantile → store 4-bit index          │
│   5. Lookup table: index → quantile value (for dequantization)        │
│                                                                         │
│   The 16 NF4 quantile values (precomputed):                           │
│   {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,               │
│    -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,                    │
│    0.4407, 0.5626, 0.7230, 1.0}                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Double Quantization — NF4's Secret Weapon:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DOUBLE QUANTIZATION                                   │
│                                                                         │
│   Problem: Each group of 64 weights needs an FP32 scale factor         │
│            → 32 bits / 64 weights = 0.5 extra bits per weight          │
│            → significant overhead at 4-bit!                            │
│                                                                         │
│   Solution: Quantize the scale factors themselves to FP8               │
│                                                                         │
│   Step 1: Quantize weights to NF4 (4-bit)                              │
│           with group_size=64, producing FP32 scales                    │
│                                                                         │
│   Step 2: Collect all FP32 scales (one per group)                      │
│           Quantize these scales to FP8 (8-bit)                         │
│           with a second-level block_size=256                           │
│                                                                         │
│   Memory per weight:                                                   │
│     NF4 weight:           4 bits                                       │
│     First-level scale:    8/64 = 0.125 bits   (FP8 scale per 64 wts) │
│     Second-level scale:   32/64/256 ≈ 0.002 bits  (FP32 per 256)    │
│     Total:               ~4.127 bits per weight                       │
│                                                                         │
│   Without double quant:  4 + 32/64 = 4.5 bits per weight             │
│   Savings:               ~0.37 bits/param → ~325 MB for a 7B model   │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# NF4 Quantization with bitsandbytes (QLoRA)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure NF4 with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",            # Use NF4 (not uniform INT4)
    bnb_4bit_use_double_quant=True,       # Enable double quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute in BF16 (dequant for matmul)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
# Model loaded in ~3.5 GB (vs 13 GB FP16)
# Ready for QLoRA fine-tuning with peft
```

---

## **5.6 SmoothQuant — INT8 Everything**

SmoothQuant (Xiao et al., 2022) tackles the biggest challenge in activation quantization: **outlier channels**. In large language models, a few activation channels have magnitudes 100× larger than the rest, making INT8 activation quantization fail (the scale is dominated by outliers, wasting precision on the majority of values).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 SMOOTHQUANT — THE PROBLEM                               │
│                                                                         │
│   Y = X × W^T                                                          │
│                                                                         │
│   Activations X (per channel):                                         │
│   [0.2, 0.1, 0.3, 85.0, 0.2, 0.1, 0.4, 92.0, 0.1, 0.3]             │
│                    ↑ outlier                 ↑ outlier                  │
│                                                                         │
│   If we quantize X to INT8 with max=92:                                │
│     scale = 92 / 127 = 0.724                                          │
│     0.2 → round(0.2/0.724) = 0   ← all small values collapse to 0!   │
│     0.3 → round(0.3/0.724) = 0                                        │
│     Result: catastrophic accuracy loss                                 │
│                                                                         │
│                 SMOOTHQUANT — THE SOLUTION                              │
│                                                                         │
│   Insight: Y = X × W^T = (X × diag(s)⁻¹) × (diag(s) × W^T)         │
│            = X_smooth × W_scaled                                       │
│                                                                         │
│   Choose s per channel to "smooth" activations:                        │
│     s_j = max(|X_j|)^α / max(|W_j|)^(1-α)     where α ∈ [0, 1]     │
│                                                                         │
│   Before smoothing:    X = [0.2, 0.1, 85.0, ...]   W = [0.5, 0.3, ...│
│   After smoothing:     X' = [0.2, 0.1, 2.1, ...]   W' = [0.5, 0.3, .│
│                              ↑ much smaller!          ↑ slightly larger │
│                                                                         │
│   Now both X' and W' are easy to quantize to INT8!                     │
│                                                                         │
│   Result: Full INT8 (W8A8) inference — both weights AND activations   │
│           → maximum hardware utilization of INT8 tensor cores          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key advantage:** SmoothQuant enables **W8A8** (weight-8bit, activation-8bit), which is the sweet spot for hardware efficiency — INT8 GEMM kernels on NVIDIA GPUs are 2× faster than FP16.

---

## **5.7 Methods Comparison Table**

| Method | Type | Bit-Width | What's Quantized | Calibration | Speed to Quantize | Accuracy | Best For |
|---|---|---|---|---|---|---|---|
| **Uniform PTQ** | Post-training | INT8 | W (or W+A) | 100–1000 samples | Minutes | Good | Quick deployment, INT8 |
| **QAT** | Training-time | INT8/INT4 | W + A | Full train set | Hours–days | Best | Max accuracy at low bits |
| **GPTQ** | Post-training | INT4 (mostly) | W only | 128–256 samples | Hours (1 GPU) | Very good | 4-bit weight-only quantization |
| **AWQ** | Post-training | INT4 | W only | 128–256 samples | Minutes–hours | Very good | Fastest to quantize, robust |
| **NF4** | Post-training | 4-bit | W only | None (abs-max) | Minutes | Good | QLoRA fine-tuning |
| **SmoothQuant** | Post-training | INT8 (W8A8) | W + A | Calibration set | Minutes | Very good | Full INT8 with activations |

```
┌─────────────────────────────────────────────────────────────────────────┐
│          DECISION FLOWCHART: WHICH METHOD TO USE?                       │
│                                                                         │
│   Need INT8 serving with activations quantized?                        │
│     YES → SmoothQuant (W8A8)                                           │
│     NO ↓                                                               │
│                                                                         │
│   Need 4-bit for memory savings?                                       │
│     NO → Use PTQ INT8 (simplest, good accuracy)                        │
│     YES ↓                                                              │
│                                                                         │
│   Will you fine-tune after quantization?                               │
│     YES → NF4 + QLoRA                                                  │
│     NO ↓                                                               │
│                                                                         │
│   Need fastest quantization time?                                      │
│     YES → AWQ (faster, more robust than GPTQ)                         │
│     NO → GPTQ (slightly better accuracy on some benchmarks)            │
│                                                                         │
│   Accuracy still not good enough?                                      │
│     → QAT (expensive but best accuracy at any bit-width)               │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** If asked "GPTQ vs AWQ?", say: GPTQ uses Hessian-based compensation (more math, slower), AWQ uses activation-aware scaling (simpler, faster). GPTQ can edge out on perplexity on the calibration distribution, but AWQ generalizes better across tasks and is faster to run. For production, AWQ is increasingly preferred.

---

# **6. Architecture Integration — Where Quantization Lives**

---

## **6.1 Quantization Placement in a Transformer**

```
┌─────────────────────────────────────────────────────────────────────────┐
│           QUANTIZATION PLACEMENT IN A TRANSFORMER BLOCK                 │
│                                                                         │
│   Input (BF16/FP16)                                                    │
│       │                                                                 │
│       ▼                                                                 │
│   ┌────────────────────────────────────────────────────────────┐       │
│   │  LAYER NORM  (always FP16/FP32 — too sensitive to quantize) │       │
│   └───────────────────────┬────────────────────────────────────┘       │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────┐                      │
│   │  Q, K, V LINEAR PROJECTIONS                  │                      │
│   │  ┌──────┐  ┌──────┐  ┌──────┐               │                      │
│   │  │W_Q   │  │W_K   │  │W_V   │  ◄── INT4/INT8 weights             │
│   │  │INT4  │  │INT4  │  │INT4  │      Dequant fused with GEMM        │
│   │  └──┬───┘  └──┬───┘  └──┬───┘               │                      │
│   │     │         │         │                    │                      │
│   │  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐               │                      │
│   │  │Deq + │  │Deq + │  │Deq + │  ← Fused: dequant happens          │
│   │  │GEMM  │  │GEMM  │  │GEMM  │    inside the matrix multiply       │
│   │  │(FP16)│  │(FP16)│  │(FP16)│    kernel, not as separate step     │
│   │  └──┬───┘  └──┬───┘  └──┬───┘               │                      │
│   └─────┼─────────┼─────────┼────────────────────┘                      │
│         ▼         ▼         ▼                                           │
│   ┌─────────────────────────────────────────────┐                      │
│   │  ATTENTION: Q × K^T / √d_k → Softmax → × V │                      │
│   │  Computed in FP16/BF16 (precision matters)   │                      │
│   │                                              │                      │
│   │  KV-Cache: can be FP8/INT8 quantized         │                      │
│   │  (reduces memory during long generation)     │                      │
│   └───────────────────────┬─────────────────────┘                      │
│                           ▼                                             │
│   ┌────────────────────────────────────────────────────────────┐       │
│   │  OUTPUT PROJECTION (W_O)  ◄── INT4/INT8                     │       │
│   └───────────────────────┬────────────────────────────────────┘       │
│                           ▼                                             │
│   ┌────────────────────────────────────────────────────────────┐       │
│   │  LAYER NORM  (FP16/FP32)                                    │       │
│   └───────────────────────┬────────────────────────────────────┘       │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────┐                      │
│   │  MLP / FFN                                   │                      │
│   │  ┌────────┐   ┌────────┐   ┌────────┐       │                      │
│   │  │Gate Proj│   │Up Proj │   │Down Prj│       │                      │
│   │  │INT4    │   │INT4    │   │INT4    │  ◄── Largest weight         │
│   │  │        │   │        │   │        │      matrices (4096×11008   │
│   │  └───┬────┘   └───┬────┘   └───┬────┘      in Llama-2-7B)        │
│   │      │            │            │             │                      │
│   │   Dequant+GEMM  Dequant+GEMM  Dequant+GEMM │                      │
│   │      │            │            │             │                      │
│   │      └─── SiLU(gate) × up ────┘             │                      │
│   │                  │                           │                      │
│   │                  ▼                           │                      │
│   │             Down projection                  │                      │
│   └───────────────────────┬─────────────────────┘                      │
│                           ▼                                             │
│                     Residual Add (FP16)                                 │
│                           │                                             │
│                           ▼                                             │
│                     Next Layer...                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## **6.2 Dequantization Fused with GEMM**

The critical performance trick: **never actually store dequantized weights**. The dequantization happens *inside* the matrix multiplication kernel.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              FUSED DEQUANT + GEMM (How It Actually Runs)                │
│                                                                         │
│   NAIVE (slow — don't do this):                                        │
│   1. Load INT4 weights from HBM → GPU registers                       │
│   2. Dequantize INT4 → FP16 in a separate kernel (write back to HBM)  │
│   3. Load FP16 weights from HBM → do GEMM                             │
│   → 2× memory traffic, extra kernel launch overhead                   │
│                                                                         │
│   FUSED (fast — how tools actually work):                              │
│   1. Load INT4 weights from HBM → GPU registers                       │
│   2. Dequantize INT4 → FP16 in registers (no memory write)            │
│   3. Immediately feed into GEMM in the same kernel                     │
│   → 1× memory traffic (load INT4 only), no overhead                   │
│                                                                         │
│   GPU Memory (HBM)         GPU Registers/SRAM                          │
│   ┌──────────────┐         ┌──────────────────────────┐               │
│   │ INT4 Weights │────────►│ Dequant → FP16 → GEMM   │               │
│   │ + Scales     │  load   │ (all fused in one kernel)│               │
│   └──────────────┘         └──────────────────────────┘               │
│                                                                         │
│   Libraries that implement fused kernels:                              │
│   • bitsandbytes (bnb_4bit_compute_dtype)                              │
│   • exllama / exllamav2 (GPTQ kernels)                                 │
│   • Marlin (GPTQ/AWQ optimized kernels, 4× faster than naive)         │
│   • TensorRT-LLM (NVIDIA's optimized runtime)                         │
│   • vLLM (uses Marlin and cutlass internally)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## **6.3 Handling Outlier Channels**

```
┌─────────────────────────────────────────────────────────────────────────┐
│              MIXED-PRECISION FOR OUTLIER CHANNELS                       │
│                                                                         │
│   Problem: ~0.1% of hidden-dim channels have activation magnitudes     │
│   100× larger than the rest. Quantizing them to INT8 destroys info.   │
│                                                                         │
│   Solution: Keep outlier channels in FP16, quantize the rest to INT8   │
│                                                                         │
│   Weight matrix W [4096 × 4096]:                                       │
│   ┌────────────────────────────────────────────────────────────┐       │
│   │ INT8 │ INT8 │ FP16 │ INT8 │ INT8 │ INT8 │ FP16 │ INT8 │ ← cols  │
│   │      │      │outlr │      │      │      │outlr │      │         │
│   │      │      │      │      │      │      │      │      │         │
│   │      │      │      │      │      │      │      │      │         │
│   └────────────────────────────────────────────────────────────┘       │
│                                                                         │
│   Implementation (LLM.int8() by bitsandbytes):                         │
│   1. During calibration, identify outlier channels (|x| > threshold)   │
│   2. Decompose matmul: Y = X_normal × W_normal^T + X_outlier × W_out^T│
│   3. First term: INT8 GEMM (fast)                                      │
│   4. Second term: FP16 GEMM (accurate, small)                          │
│   5. Sum results                                                       │
│                                                                         │
│   Typically only ~6-10 outlier columns out of 4096-5120               │
│   → 99.8% of compute is still INT8                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** When discussing architecture integration, emphasize three things: (1) LayerNorm and Softmax stay in high precision, (2) dequant is fused with GEMM for zero-overhead, (3) outlier channels get special treatment. This shows you understand the *engineering* of quantization, not just the theory.

---

# **7. Practical Tools & Code**

---

## **7.1 bitsandbytes**

The most popular library for on-the-fly quantization with Hugging Face Transformers. Developed by Tim Dettmers.

```python
# === bitsandbytes — 8-bit loading (LLM.int8()) ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 8-bit config (mixed INT8 + FP16 for outliers)
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,    # Outlier threshold (channels > 6.0 stay FP16)
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_8bit,
    device_map="auto",
)
# Llama-2-7B: 13 GB FP16 → ~7 GB INT8

# === bitsandbytes — 4-bit NF4 loading (for QLoRA) ===
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 (vs "fp4" for uniform)
    bnb_4bit_use_double_quant=True,       # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # Dequant to BF16 for compute
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_4bit,
    device_map="auto",
)
# Llama-2-7B: 13 GB FP16 → ~3.5 GB NF4

# === QLoRA fine-tuning with 4-bit base ===
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model_qlora = get_peft_model(model_4bit, lora_config)
model_qlora.print_trainable_parameters()
# trainable params: 13,107,200 || all params: 3,540,389,888 || trainable%: 0.37%
```

---

## **7.2 AutoGPTQ**

```python
# === GPTQ Quantization with auto-gptq ===
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare calibration data
calibration_texts = [
    "The transformer architecture was introduced in...",
    "Machine learning models can be compressed by...",
    # ... 128 diverse text samples
]
calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_texts
]

# Configure GPTQ
quantize_config = BaseQuantizeConfig(
    bits=4,               # 4-bit quantization
    group_size=128,       # Group quantization
    desc_act=True,        # Activation order (better accuracy, slower)
    sym=True,             # Symmetric quantization
    damp_percent=0.01,    # Hessian dampening factor
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
model.quantize(calibration_data)
model.save_quantized("./llama2-7b-gptq-4bit")

# Load quantized model for inference
model_quantized = AutoGPTQForCausalLM.from_quantized(
    "./llama2-7b-gptq-4bit",
    device="cuda:0",
    use_triton=False,         # Set True for Triton kernels
    inject_fused_attention=True,
    inject_fused_mlp=True,
)
```

---

## **7.3 AutoAWQ**

```python
# === AWQ Quantization with autoawq ===
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama2-7b-awq-4bit"

# Load
model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",       # "GEMM" for batched, "GEMV" for single-request
}
model.quantize(tokenizer, quant_config=quant_config)

# Save (compatible with vLLM, TGI, transformers)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

---

## **7.4 TensorRT-LLM (NVIDIA)**

```python
# === TensorRT-LLM — High-performance inference engine ===
# TensorRT-LLM uses a build step to compile optimized engines

# Step 1: Convert HuggingFace model to TensorRT-LLM checkpoint
# (command line)
"""
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./trt_ckpt/llama2-7b-int4 \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4_awq \
    --per_group \
    --group_size 128
"""

# Step 2: Build TensorRT engine
"""
trtllm-build \
    --checkpoint_dir ./trt_ckpt/llama2-7b-int4 \
    --output_dir ./trt_engines/llama2-7b-int4 \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --paged_kv_cache enable
"""

# Step 3: Run inference
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="./trt_engines/llama2-7b-int4")
output = llm.generate(
    "What is quantization?",
    SamplingParams(temperature=0.7, max_tokens=256)
)
print(output.text)
```

---

## **7.5 vLLM**

```python
# === vLLM — Serving quantized models ===
from vllm import LLM, SamplingParams

# Load GPTQ-quantized model
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",    # Pre-quantized GPTQ model
    quantization="gptq",                    # Specify quantization format
    dtype="half",                            # Compute dtype
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

# Load AWQ-quantized model
llm_awq = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half",
)

# Load with on-the-fly bitsandbytes quantization
llm_bnb = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)

# Generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Explain model quantization in simple terms."], sampling_params)
for output in outputs:
    print(output.outputs[0].text)

# === vLLM serving (OpenAI-compatible API) ===
"""
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq \
    --dtype half \
    --max-model-len 4096 \
    --port 8000
"""
```

---

## **7.6 ONNX Runtime & TFLite**

```python
# === ONNX Runtime — Cross-platform quantized inference ===
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# Dynamic INT8 quantization (simplest)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
)

# Static quantization (with calibration)
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class MyCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = iter(calibration_data)

    def get_next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

quantize_static(
    model_input="model.onnx",
    model_output="model_int8_static.onnx",
    calibration_data_reader=MyCalibrationReader(calib_data),
    quant_format=ort.quantization.QuantFormat.QDQ,       # Quantize-Dequantize format
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)

# Inference
session = ort.InferenceSession("model_int8.onnx")
result = session.run(None, {"input": input_data})
```

```python
# === TFLite — Mobile/Edge quantization ===
import tensorflow as tf

# Post-training dynamic range quantization
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Post-training full integer quantization (INT8 weights + activations)
def representative_dataset():
    for data in calibration_data:
        yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model_int8 = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model_int8)
```

---

## **7.7 Tools Comparison Table**

| Tool | Supported Formats | Target Hardware | Integration | Strengths |
|---|---|---|---|---|
| **bitsandbytes** | INT8 (LLM.int8), NF4 | NVIDIA GPU | HuggingFace native | Easiest setup, QLoRA |
| **AutoGPTQ** | GPTQ (INT4/INT3/INT2) | NVIDIA GPU | HuggingFace, vLLM | Best GPTQ implementation |
| **AutoAWQ** | AWQ (INT4) | NVIDIA GPU | HuggingFace, vLLM, TGI | Fast quantization, robust |
| **TensorRT-LLM** | INT8, INT4, FP8 | NVIDIA GPU | Triton Server | Maximum NVIDIA performance |
| **vLLM** | GPTQ, AWQ, bnb, FP8 | NVIDIA GPU | OpenAI-compatible API | Best serving framework |
| **ONNX Runtime** | INT8 (dynamic/static) | CPU, GPU, NPU | Cross-platform | Edge/cloud versatility |
| **TFLite** | INT8, FP16, dynamic | Mobile, Edge TPU | Android, iOS, RPi | Mobile deployment |
| **llama.cpp** | GGUF (Q4_0 to Q8_0) | CPU, Metal, CUDA | CLI, server mode | CPU inference, Apple Silicon |

---

# **8. Memory Examples — Real Numbers**

---

## **8.1 Llama-2-7B Memory Breakdown**

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLAMA-2-7B MEMORY BREAKDOWN                                │
│                                                                         │
│   Parameter count: 6.74 billion                                        │
│                                                                         │
│   Format          Bytes/Param   Model Size   GPU Required              │
│   ─────────────   ───────────   ──────────   ────────────              │
│   FP32            4.0           ~26.0 GB     A100 80GB                 │
│   FP16 / BF16     2.0           ~13.0 GB     A100 40GB / A10G 24GB    │
│   INT8 (bnb)      1.0           ~ 6.5 GB     T4 16GB / RTX 3090      │
│   GPTQ-INT4       0.56*         ~ 3.8 GB     T4 16GB (room to spare) │
│   NF4 (QLoRA)     0.52*         ~ 3.5 GB     T4 16GB / RTX 3060 12GB│
│   GGUF Q4_K_M     0.56*         ~ 3.8 GB     CPU with 8GB+ RAM       │
│                                                                         │
│   * Includes scale/zero-point overhead (~0.06 bytes/param)             │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  But weights aren't everything! During inference, also:      │      │
│   │                                                              │      │
│   │  KV-Cache (for generation):                                  │      │
│   │  • Per token: 2 × n_layers × n_heads × d_head × 2 bytes    │      │
│   │  • Llama-2-7B: 2 × 32 × 32 × 128 × 2 = 0.5 MB per token  │      │
│   │  • At seq_len=4096: 0.5 × 4096 = ~2 GB per batch item      │      │
│   │                                                              │      │
│   │  Total GPU memory needed:                                    │      │
│   │  FP16 model + KV-cache(4096) + overhead ≈ 13 + 2 + 1 = 16GB│      │
│   │  NF4 model  + KV-cache(4096) + overhead ≈ 3.5 + 2 + 1 = 6.5GB    │
│   │                                                              │      │
│   │  Quantized KV-Cache (FP8):                                   │      │
│   │  NF4 model + FP8-KV(4096) + overhead ≈ 3.5 + 1 + 1 = 5.5GB│      │
│   └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

## **8.2 Scaling to Larger Models**

| Model | Params | FP16 | INT8 | INT4/NF4 | Minimum GPU (INT4) |
|---|---|---|---|---|---|
| **Llama-2-7B** | 6.7B | 13 GB | 6.5 GB | 3.5 GB | RTX 3060 12GB |
| **Llama-2-13B** | 13B | 25 GB | 12.5 GB | 7 GB | RTX 3090 24GB |
| **Llama-2-70B** | 70B | 130 GB | 65 GB | 35 GB | 2× A100 40GB |
| **Llama-3-8B** | 8B | 16 GB | 8 GB | 4.5 GB | RTX 4060 Ti 16GB |
| **Llama-3-70B** | 70B | 140 GB | 70 GB | 38 GB | 2× A100 40GB |
| **Mixtral-8x7B** | 47B (12B active) | 90 GB | 45 GB | 24 GB | A100 40GB |
| **GPT-3** | 175B | 350 GB | 175 GB | 90 GB | 8× A100 40GB → 2× A100 |
| **Llama-3-405B** | 405B | 810 GB | 405 GB | 210 GB | 4× A100 80GB |

```
┌─────────────────────────────────────────────────────────────────────────┐
│         COST IMPACT — CLOUD DEPLOYMENT EXAMPLE                          │
│                                                                         │
│   Serving Llama-2-70B:                                                 │
│                                                                         │
│   FP16:  4× A100 80GB   → $14.00/hr (AWS p4d)                         │
│   INT8:  2× A100 40GB   → $ 6.50/hr (AWS p4d)                         │
│   INT4:  1× A100 80GB   → $ 3.50/hr (AWS p4d)                         │
│   INT4:  2× A10G 24GB   → $ 2.40/hr (AWS g5)   ← cheapest option     │
│                                                                         │
│   Annual savings (24/7 serving):                                       │
│   FP16 → INT4: ($14.00 - $2.40) × 8,760 hrs = $101,544 / year        │
│                                                                         │
│   That's a >80% cost reduction with ~2% perplexity increase.          │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** When asked about quantization, always bring it back to concrete numbers. "Llama-2-7B goes from 13GB to 3.5GB with NF4, fitting on a consumer RTX 3060" is far more impactful than abstract discussion about bit-widths.

---

# **9. Production Pipeline**

---

## **9.1 End-to-End Quantization Deployment**

```
┌─────────────────────────────────────────────────────────────────────────┐
│         PRODUCTION QUANTIZATION PIPELINE                                │
│                                                                         │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│   │ 1.SCOPE │──►│ 2.PREP   │──►│3.QUANTIZE│──►│4.VALIDATE│           │
│   │         │   │          │   │          │   │          │           │
│   │ Define  │   │ Collect  │   │ Run GPTQ │   │ Run eval │           │
│   │ target  │   │ calib    │   │ / AWQ /  │   │ suite    │           │
│   │ bit-wid │   │ data     │   │ NF4      │   │          │           │
│   └─────────┘   └──────────┘   └──────────┘   └────┬─────┘           │
│                                                      │                 │
│                                                      ▼                 │
│                                               ┌──────────┐            │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐  │ Quality  │            │
│   │7.ITERATE│◄──│ 6.DEPLOY │◄──│5.PACKAGE │◄─│ Gate     │            │
│   │         │   │          │   │          │  │ Pass?    │            │
│   │ Canary  │   │ Helm     │   │ Docker   │  │          │            │
│   │ rollout │   │ chart    │   │ image    │  │ YES → ──►│            │
│   │ monitor │   │ k8s      │   │ artifact │  │ NO → fix │            │
│   └─────────┘   └──────────┘   └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

## **9.2 Each Step in Detail**

### **Step 1: Scope**
```
Decisions:
├── Target hardware: GPU type (T4? A10G? A100?), memory budget
├── Bit-width: INT8 (safe) vs INT4 (aggressive) vs NF4 (if fine-tuning)
├── Method: GPTQ vs AWQ vs SmoothQuant vs bitsandbytes
├── Acceptable quality loss: <1% perplexity for INT8, <3% for INT4
└── Latency / throughput targets: p50, p99, tokens/sec
```

### **Step 2: Prepare Calibration Data**
```python
# Calibration dataset: 128-512 diverse samples from your domain
# Should represent the actual input distribution

calibration_data = []
for source in ["user_queries", "documents", "instructions"]:
    samples = load_samples(source, n=128)
    calibration_data.extend(samples)

# Tokenize with proper truncation
calibration_tokens = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_data
]
```

### **Step 3: Quantize**
```bash
# AWQ example (preferred for production serving)
python -m awq.entry \
    --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 \
    --q_group_size 128 \
    --output_path ./quantized_model
```

### **Step 4: Validate — Regression Tests**
```python
# Evaluation suite for quantized model
eval_suite = {
    "perplexity":      evaluate_perplexity(model_q, test_data),
    "mmlu_accuracy":   evaluate_mmlu(model_q),
    "humaneval_pass1": evaluate_code_gen(model_q),
    "latency_p50":     benchmark_latency(model_q, p=50),
    "latency_p99":     benchmark_latency(model_q, p=99),
    "throughput":      benchmark_throughput(model_q),
}

# Quality gate
assert eval_suite["perplexity"] < baseline_ppl * 1.03, "Perplexity regression > 3%"
assert eval_suite["mmlu_accuracy"] > baseline_mmlu * 0.97, "MMLU regression > 3%"
assert eval_suite["latency_p99"] < 200, "P99 latency > 200ms"
```

### **Step 5: Package**
```dockerfile
# Dockerfile for quantized model serving
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN pip install vllm==0.4.0

COPY ./quantized_model /app/model

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--quantization", "awq", \
     "--dtype", "half", \
     "--max-model-len", "4096", \
     "--port", "8000"]
```

### **Step 6: Deploy (Helm + Kubernetes)**
```yaml
# helm values.yaml
replicaCount: 2
image:
  repository: myregistry/llm-quantized
  tag: "v1.0-awq-int4"
resources:
  limits:
    nvidia.com/gpu: 1        # 1 GPU per pod
    memory: "24Gi"
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
nodeSelector:
  gpu-type: "nvidia-t4"      # Target cheap T4 GPUs
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### **Step 7: Iterate — Canary Rollout**
```
Canary strategy:
├── Deploy quantized model to 5% of traffic
├── Monitor for 24-48 hours:
│   ├── Latency (p50, p99)
│   ├── Error rate
│   ├── User satisfaction (thumbs up/down)
│   └── Task-specific metrics (BLEU, accuracy, etc.)
├── If metrics are stable: ramp to 25% → 50% → 100%
└── If regression detected: instant rollback to FP16 deployment
```

---

# **10. When NOT to Quantize**

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│               WHEN NOT TO QUANTIZE                                      │
│                                                                         │
│   1. ACCURACY-CRITICAL TASKS                                           │
│      ──────────────────────                                             │
│      • Medical diagnosis, legal analysis, financial modeling            │
│      • When even 1% accuracy drop is unacceptable                      │
│      • When the model is already borderline on benchmarks              │
│      → Use FP16/BF16 and invest in better hardware instead            │
│                                                                         │
│   2. SMALL MODELS (< 1B parameters)                                   │
│      ───────────────────────────                                        │
│      • Already fits in memory easily                                   │
│      • Less redundancy → more sensitive to quantization                │
│      • INT4 on a 125M model can lose 10-15% accuracy                  │
│      → Diminishing returns; the memory savings aren't worth it        │
│                                                                         │
│   3. DURING TRAINING                                                   │
│      ──────────────                                                     │
│      • Gradients need full precision to converge properly              │
│      • Weight updates are tiny (1e-5 scale) — low precision kills them│
│      • Exception: mixed-precision training (FP16 forward, FP32 master │
│        weights) and 8-bit optimizers are fine                          │
│      → Quantize for inference, not for training (except QAT)          │
│                                                                         │
│   4. WHEN LATENCY ISN'T THE BOTTLENECK                                │
│      ────────────────────────────────                                   │
│      • If your bottleneck is network I/O, database, or preprocessing  │
│      • Quantizing the model won't help if it's waiting for data       │
│      → Profile first, quantize only if model inference is the limit   │
│                                                                         │
│   5. WHEN YOU NEED TO FINE-TUNE FREQUENTLY                            │
│      ────────────────────────────────                                   │
│      • GPTQ/AWQ models can't be fine-tuned (weights are frozen ints)  │
│      • Exception: NF4 + QLoRA is specifically designed for this       │
│      → Use NF4/QLoRA, or keep FP16 and quantize at serving time      │
│                                                                         │
│   6. RESEARCH / ABLATION STUDIES                                       │
│      ─────────────────────────                                          │
│      • When you need exact reproducibility                             │
│      • Quantization adds a confounding variable                        │
│      → Keep FP32/FP16 during research, quantize for deployment        │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Knowing when *not* to use a technique is just as impressive as knowing how to use it. If asked "would you quantize model X?", always ask about the use case, accuracy requirements, and model size before recommending quantization.

---

# **11. Common Interview Questions with Strong Answers**

---

## **Q1: "What is model quantization and why do we need it?"**

**Strong answer:**

> "Quantization reduces the numerical precision of model weights — and optionally activations — from high-precision formats like FP16 (16 bits) down to INT8 (8 bits) or even INT4 (4 bits). Think of it as a lossy ZIP for neural networks.
>
> We need it for four reasons: **cost** (a 70B model drops from needing four A100 GPUs to one), **speed** (LLM inference is memory-bandwidth-bound, so halving the bytes per weight nearly doubles throughput), **energy** (fewer bytes transferred = less power), and **portability** (a 7B model at NF4 fits on a consumer GPU or even a laptop with 8GB RAM).
>
> The key insight is that neural network weights are highly redundant — they cluster near zero, and the model's function depends on relative relationships between weights, not exact values. This redundancy is what makes quantization work with minimal accuracy loss."

---

## **Q2: "Explain the difference between GPTQ, AWQ, and NF4."**

**Strong answer:**

> "All three are post-training quantization methods targeting 4-bit weights, but they differ in approach:
>
> **GPTQ** uses Hessian-based optimization — it quantizes weights column by column within each layer, using second-order information to adjust remaining weights and compensate for each column's quantization error. It's mathematically rigorous and produces excellent perplexity on the calibration distribution, but it's slower to run and can overfit to the calibration set.
>
> **AWQ** takes a simpler approach: it observes that a small fraction of weight channels (those with large activation magnitudes) are 'salient' and disproportionately affect accuracy. AWQ scales these salient channels up before quantization so they get less relative error, then compensates at inference time. It's faster to quantize and generalizes better across tasks.
>
> **NF4** is the format behind QLoRA. It uses quantile-based bins optimized for normally-distributed weights — denser bins near zero where most weights live. Combined with double quantization (quantizing the scale factors themselves to FP8), it achieves ~4.1 bits per parameter. Its unique advantage is that you can fine-tune on top of NF4-quantized weights using LoRA adapters.
>
> In practice: use AWQ or GPTQ for pure inference serving, NF4 when you plan to fine-tune after quantization."

---

## **Q3: "How does quantization affect model accuracy? What's the typical degradation?"**

**Strong answer:**

> "The accuracy impact depends on three factors: the bit-width, the quantization method, and the model size.
>
> At **INT8**, most methods (PTQ, SmoothQuant) achieve less than 1% perplexity increase on language models — practically negligible. At **INT4**, it's more nuanced: naive round-to-nearest can lose 5-10%, but GPTQ and AWQ with group quantization (group_size=128) typically keep it under 2-3% perplexity increase.
>
> Larger models are more robust to quantization — a 70B model quantized to INT4 often performs better than a 7B model at FP16. This is because larger models have more redundancy.
>
> The accuracy loss is also task-dependent. Perplexity might barely change while specific capabilities (math reasoning, code generation) could degrade more noticeably. That's why production validation should include task-specific benchmarks, not just perplexity.
>
> For critical applications, I'd recommend: quantize to INT4, run your full eval suite, and compare against FP16 baseline. If any benchmark degrades beyond your threshold, try INT8 or SmoothQuant W8A8 instead."

---

## **Q4: "What's the difference between PTQ and QAT? When would you use each?"**

**Strong answer:**

> "**PTQ (Post-Training Quantization)** takes a trained model, runs a small calibration dataset through it to determine value ranges, and maps weights to lower precision. It takes minutes, requires no training, and needs only 100-1000 unlabeled samples.
>
> **QAT (Quantization-Aware Training)** inserts fake quantization nodes during training, so the model learns to be robust to low-precision rounding. It uses the Straight-Through Estimator (STE) for backpropagation through non-differentiable rounding operations. It requires the full training dataset and GPU hours for fine-tuning.
>
> The tradeoff is clear: **PTQ is cheap and fast, QAT is expensive but more accurate**, especially at low bit-widths. At INT8, PTQ is usually sufficient — the accuracy gap is negligible. At INT4, QAT can recover 2-3% accuracy over PTQ.
>
> For LLMs specifically, PTQ dominates because retraining a 7B-70B model is extremely expensive. Methods like GPTQ and AWQ are essentially 'enhanced PTQ' that close the accuracy gap with QAT by using smarter quantization strategies (Hessian compensation, activation-aware scaling). QAT makes more sense for smaller, task-specific models where training cost is manageable."

---

## **Q5: "How would you deploy a quantized 7B model in production?"**

**Strong answer:**

> "I'd follow a seven-step pipeline:
>
> **1. Scope:** For a 7B model, INT4 (AWQ) is the sweet spot — it shrinks from 13GB to ~3.8GB, fitting on a single T4 GPU ($0.70/hr vs $3.50/hr for A100).
>
> **2. Calibration data:** Collect 128-512 representative samples from our actual user traffic — diverse enough to cover the input distribution.
>
> **3. Quantize:** Use AutoAWQ with group_size=128, 4-bit, GEMM kernels. AWQ over GPTQ because it's faster and generalizes better.
>
> **4. Validate:** Run the full eval suite — perplexity on held-out data, task-specific benchmarks (MMLU, HumanEval if it's a code model), and latency benchmarks. Set a quality gate: less than 3% degradation on any metric.
>
> **5. Package:** Docker image with vLLM as the serving engine (PagedAttention for efficient KV-cache, continuous batching for throughput). The AWQ model loads with one flag: `--quantization awq`.
>
> **6. Deploy:** Kubernetes with Helm. Node selector for T4 GPU nodes. Horizontal pod autoscaler based on request queue depth.
>
> **7. Canary:** Route 5% of traffic to the quantized model for 48 hours. Monitor latency, error rates, and user feedback. If stable, ramp to 100% and decommission the FP16 deployment.
>
> The result: same model quality (within 2-3%), 80% lower serving cost, 2-3× higher throughput."

---

## **Q6: "What is the SmoothQuant technique and when would you use it?"**

**Strong answer:**

> "SmoothQuant solves the activation quantization problem. In large language models, certain activation channels have magnitudes 100× larger than the rest — these outliers make INT8 activation quantization fail because the scale is dominated by outliers, collapsing small values to zero.
>
> SmoothQuant's insight is mathematical: since Y = X × W, we can insert a diagonal scaling matrix: Y = (X × S⁻¹) × (S × W). This 'smooths' the spiky activations by dividing by per-channel scales, while multiplying the corresponding weight channels by those same scales. The outliers shift from activations into weights, where they're easier to quantize.
>
> The migration strength α controls how much difficulty shifts from activations to weights. At α=0.5, it's an even split. The optimal α is found per-layer using a small calibration set.
>
> I'd use SmoothQuant when I need **W8A8** (both weights and activations in INT8), which unlocks INT8 GEMM kernels that are 2× faster than FP16 on NVIDIA GPUs. This is different from GPTQ/AWQ, which are weight-only quantization (W4A16) — the activations still run in FP16. SmoothQuant is ideal when throughput is the primary concern and you have hardware with fast INT8 tensor cores."

---

## **Q7: "What is double quantization in NF4, and why does it matter?"**

**Strong answer:**

> "Double quantization addresses the overhead problem of group quantization. In NF4, each group of 64 weights shares an FP32 scale factor. That's 32 bits / 64 weights = 0.5 extra bits per parameter — which is significant when your weights are only 4 bits!
>
> Double quantization adds a second level: collect all the FP32 scale factors, treat them as a new 'tensor,' and quantize those to FP8 with a block size of 256. Now the overhead is 8/64 = 0.125 bits per parameter for the first-level scales, plus a negligible ~0.002 bits for the second-level FP32 scales.
>
> Total: 4 + 0.125 + 0.002 ≈ 4.127 bits per parameter, versus 4.5 bits without double quantization. For a 7B model, that's about 325 MB saved — meaningful when you're trying to fit on a 12GB consumer GPU."

---

# **12. Key Takeaways**

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KEY TAKEAWAYS                                       │
│                                                                         │
│  1. Quantization = fewer bits per weight/activation, a "lossy ZIP      │
│     for neural networks." Models tolerate it because weights are        │
│     redundant and cluster near zero.                                    │
│                                                                         │
│  2. It's not just about memory — it speeds up inference because        │
│     LLMs are memory-bandwidth-bound. Fewer bytes = faster loads        │
│     from HBM = higher tokens/sec.                                      │
│                                                                         │
│  3. INT8 is the safe default (99% quality, 2× compression).           │
│     INT4 is the aggressive choice (95-98% quality, 4× compression).   │
│                                                                         │
│  4. The formula: x̂ = scale × (int_x − zero_point), where             │
│     scale = (x_max − x_min) / (2^b − 1). Group quantization          │
│     (g=128) is the accuracy secret at 4-bit.                           │
│                                                                         │
│  5. GPTQ: Hessian-based column-wise optimization. Mathematically      │
│     rigorous, slightly slower.                                         │
│     AWQ: Activation-aware scaling. Simpler, faster, better             │
│     generalization.                                                    │
│     NF4: Quantile bins for normal distributions + double quant.       │
│     Purpose-built for QLoRA fine-tuning.                               │
│     SmoothQuant: Migrate outliers from activations to weights.         │
│     Enables W8A8 for maximum hardware utilization.                     │
│                                                                         │
│  6. LayerNorm and Softmax stay in FP16/FP32. Dequantization is        │
│     fused with GEMM for zero overhead. Outlier channels get            │
│     mixed-precision treatment.                                         │
│                                                                         │
│  7. Tools: bitsandbytes (easiest, QLoRA), AutoGPTQ (GPTQ),           │
│     AutoAWQ (AWQ), vLLM (serving), TensorRT-LLM (max NVIDIA perf),   │
│     ONNX-RT/TFLite (edge).                                            │
│                                                                         │
│  8. Concrete: Llama-2-7B goes from 13GB FP16 → 3.5GB NF4,           │
│     fitting on a $0.70/hr T4 instead of a $3.50/hr A100.             │
│     Llama-2-70B: from 4× A100 ($14/hr) → 1× A100 ($3.50/hr).       │
│                                                                         │
│  9. Production: Scope → Calibrate → Quantize → Validate (quality     │
│     gate) → Package (Docker) → Deploy (K8s) → Canary rollout.        │
│                                                                         │
│  10. Don't quantize: accuracy-critical tasks, small models (<1B),     │
│      during training, when inference isn't the bottleneck.             │
│                                                                         │
│  REMEMBER: "Quantization is the single highest-ROI optimization       │
│  for LLM deployment — it's the first thing you should try before      │
│  distillation, pruning, or custom hardware."                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Last updated: February 2026*
