# Deep Learning Architectures — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — Deep Learning, PyTorch, TensorFlow, Computer Vision, Multimodal Systems

---

# Table of Contents

1. [Neural Network Fundamentals](#1-neural-network-fundamentals)
2. [CNNs — Convolutional Neural Networks](#2-cnns--convolutional-neural-networks)
3. [RNNs — Recurrent Neural Networks](#3-rnns--recurrent-neural-networks)
4. [Sequence-to-Sequence & Attention](#4-sequence-to-sequence--attention)
5. [Autoencoders](#5-autoencoders)
6. [GANs — Generative Adversarial Networks](#6-gans--generative-adversarial-networks)
7. [Diffusion Models](#7-diffusion-models)
8. [Vision Transformers (ViT) & CLIP](#8-vision-transformers-vit--clip)
9. [Mixture of Experts (MoE)](#9-mixture-of-experts-moe)
10. [State Space Models & Mamba](#10-state-space-models--mamba)
11. [Architecture Selection Guide](#11-architecture-selection-guide)
12. [Common Interview Questions](#12-common-interview-questions-with-strong-answers)
13. [Key Takeaways](#13-key-takeaways)

---

# **1. Neural Network Fundamentals**

---

## **1.1 The Perceptron — Where It All Started**

The perceptron (Rosenblatt, 1958) is the simplest neural unit — a single neuron that computes a weighted sum of inputs, adds a bias, and passes the result through an activation function:

$$
y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

```
┌──────────────────────────────────────────────────────┐
│                   SINGLE PERCEPTRON                   │
│                                                       │
│   x₁ ──w₁──┐                                        │
│              │                                        │
│   x₂ ──w₂──┼──→ [Σ + b] ──→ σ(z) ──→ y             │
│              │                                        │
│   x₃ ──w₃──┘                                        │
│                                                       │
│   z = w₁x₁ + w₂x₂ + w₃x₃ + b                      │
│   y = σ(z)     where σ is an activation function      │
└──────────────────────────────────────────────────────┘
```

**Limitation:** A single perceptron can only learn linearly separable functions. It cannot solve XOR — this was the famous Minsky & Papert (1969) critique that stalled neural network research for over a decade.

---

## **1.2 Multi-Layer Perceptron (MLP) — Feedforward Networks**

An MLP stacks multiple layers of neurons — input, one or more hidden layers, and an output layer — with non-linear activations between them. Every neuron in layer $l$ connects to every neuron in layer $l+1$ (fully connected / dense).

```
┌────────────────────────────────────────────────────────────┐
│                    MULTI-LAYER PERCEPTRON                    │
│                                                              │
│   Input Layer      Hidden Layer 1     Hidden Layer 2    Output│
│                                                              │
│   x₁ ─────────┬──→ [h₁¹] ─────┬──→ [h₁²] ────┬──→ ŷ₁     │
│                │               │               │             │
│   x₂ ─────────┼──→ [h₂¹] ─────┼──→ [h₂²] ────┼──→ ŷ₂     │
│                │               │               │             │
│   x₃ ─────────┼──→ [h₃¹] ─────┼──→ [h₃²] ────┘             │
│                │               │                             │
│   x₄ ─────────┘──→ [h₄¹] ─────┘                             │
│                                                              │
│   h⁽¹⁾ = σ(W⁽¹⁾x + b⁽¹⁾)                                   │
│   h⁽²⁾ = σ(W⁽²⁾h⁽¹⁾ + b⁽²⁾)                                │
│   ŷ    = softmax(W⁽³⁾h⁽²⁾ + b⁽³⁾)                          │
└────────────────────────────────────────────────────────────┘
```

**PyTorch implementation:**

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
```

---

## **1.3 Forward Propagation**

Forward propagation is the process of passing input data through the network layer by layer to produce a prediction. For an $L$-layer network:

$$
\mathbf{a}^{(0)} = \mathbf{x} \quad \text{(input)}
$$
$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \quad \text{(pre-activation)}
$$
$$
\mathbf{a}^{(l)} = \sigma^{(l)}(\mathbf{z}^{(l)}) \quad \text{(post-activation)}
$$

**Key point:** During forward pass, we cache each $\mathbf{z}^{(l)}$ and $\mathbf{a}^{(l)}$ because they are needed during backpropagation.

```
┌──────────────────────────────────────────────────────┐
│                   FORWARD PROPAGATION                 │
│                                                       │
│   x ──→ [z¹ = W¹x + b¹] ──→ [a¹ = σ(z¹)] ──→      │
│         [z² = W²a¹ + b²] ──→ [a² = σ(z²)] ──→      │
│         [z³ = W³a² + b³] ──→ [ŷ = softmax(z³)] ──→  │
│                                                       │
│         ──→ Loss = L(ŷ, y)                            │
│                                                       │
│   Cache: z¹, a¹, z², a², z³  (needed for backward)   │
└──────────────────────────────────────────────────────┘
```

---

## **1.4 Backpropagation & The Chain Rule**

Backpropagation computes gradients of the loss with respect to every parameter by systematically applying the **chain rule** from calculus. Starting from the loss, gradients flow backwards through each layer.

**The chain rule in action:**

For a parameter $W^{(l)}$ in layer $l$:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{z}^{(L)}} \cdot \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{a}^{(L-1)}} \cdots \frac{\partial \mathbf{z}^{(l)}}{\partial W^{(l)}}
$$

**Layer-by-layer backward computation:**

$$
\delta^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot \sigma'(\mathbf{z}^{(L)}) \quad \text{(output layer error)}
$$
$$
\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)}) \quad \text{(hidden layer error)}
$$
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot \left(\mathbf{a}^{(l-1)}\right)^T \quad \text{(weight gradient)}
$$
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \delta^{(l)} \quad \text{(bias gradient)}
$$

```
┌─────────────────────────────────────────────────────────────┐
│                     BACKPROPAGATION                           │
│                                                               │
│   Loss ← ∂L/∂ŷ ← ∂ŷ/∂z³ ← ∂z³/∂a² ← ∂a²/∂z² ← ...      │
│                                                               │
│   For each layer l (right to left):                           │
│     1. Compute δ⁽ˡ⁾ = error signal at layer l                │
│     2. ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ                         │
│     3. ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾                                       │
│     4. Pass δ⁽ˡ⁾ back: δ⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀ δ⁽ˡ⁾ ⊙ σ'(z⁽ˡ⁻¹⁾) │
│                                                               │
│   Time complexity: Same as forward pass (one pass backward)   │
│   Space complexity: Must store all activations (memory!)      │
└─────────────────────────────────────────────────────────────┘
```

> **Interview tip:** "Backpropagation is just the chain rule applied recursively. It's efficient because it reuses intermediate computations — each layer's gradient is computed from the layer above, not recomputed from scratch. This makes it O(n) per parameter, not O(n²)."

---

## **1.5 Gradient Descent Variants**

| Variant | Update Rule | Batch Size | Pros | Cons |
|---|---|---|---|---|
| **Batch GD** | $W = W - \eta \nabla_W \mathcal{L}_{\text{full}}$ | Entire dataset | Stable convergence | Slow, high memory |
| **Stochastic GD** | $W = W - \eta \nabla_W \mathcal{L}_i$ | 1 sample | Fast updates, escapes local minima | Noisy, oscillates |
| **Mini-batch SGD** | $W = W - \eta \nabla_W \mathcal{L}_B$ | 32–512 samples | Best of both, GPU efficient | Requires tuning batch size |

**Modern optimizers build on SGD:**

| Optimizer | Key Idea | When to Use |
|---|---|---|
| **SGD + Momentum** | Exponential moving average of gradients | CNNs, when you want fine-grained control |
| **Adam** | Adaptive LR per parameter + momentum | Default choice, most deep learning tasks |
| **AdamW** | Adam with decoupled weight decay | Transformers (standard for LLMs) |

---

## **1.6 Vanishing & Exploding Gradients**

The deepest problem in training deep networks. During backpropagation, gradients are multiplied by the derivative of the activation function and the weight matrix at each layer:

$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \prod_{l=2}^{L} \left( W^{(l)} \cdot \sigma'(z^{(l-1)}) \right) \cdot \frac{\partial \mathcal{L}}{\partial z^{(L)}}
$$

**Vanishing gradients:** If $\|W^{(l)} \cdot \sigma'(z^{(l)})\| < 1$ consistently, the product shrinks exponentially with depth. Early layers receive near-zero gradients and stop learning.

**Exploding gradients:** If $\|W^{(l)} \cdot \sigma'(z^{(l)})\| > 1$ consistently, the product grows exponentially. Weights become NaN.

```
┌───────────────────────────────────────────────────────────┐
│           GRADIENT FLOW THROUGH DEEP NETWORKS              │
│                                                            │
│   Vanishing (sigmoid/tanh):                                │
│   Layer 1    Layer 2    Layer 5    Layer 10    Layer 20     │
│   |∇| ≈ 1   |∇| ≈ 0.25  |∇| ≈ 10⁻³  |∇| ≈ 10⁻⁶  |∇| ≈ 10⁻¹² │
│                                                            │
│   Sigmoid derivative max = 0.25 → 0.25²⁰ ≈ 10⁻¹²         │
│                                                            │
│   Exploding:                                               │
│   Layer 1    Layer 2    Layer 5    Layer 10                 │
│   |∇| ≈ 1   |∇| ≈ 2    |∇| ≈ 32   |∇| ≈ 1024 → NaN!    │
└───────────────────────────────────────────────────────────┘
```

**Solutions:**

| Problem | Solution | Mechanism |
|---|---|---|
| Vanishing | **ReLU activation** | Gradient = 1 for positive inputs (no saturation) |
| Vanishing | **Residual connections** | $y = F(x) + x$ — gradient flows through identity shortcut |
| Vanishing | **Batch normalization** | Normalizes activations, keeps gradients in a healthy range |
| Vanishing | **LSTM / GRU** | Gating mechanism controls gradient flow in sequences |
| Exploding | **Gradient clipping** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| Exploding | **Proper initialization** | Xavier/Glorot, He/Kaiming — variance-preserving init |
| Both | **Layer normalization** | Stabilizes training, especially in transformers |

---

## **1.7 Universal Approximation Theorem**

**Statement:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, given a non-linear activation function.

**What it means:**
- MLPs are theoretically powerful enough to model *any* function
- The catch: it says nothing about *how many* neurons are needed (could be exponentially many) or *how to find* the right weights

**What it does NOT mean:**
- It does NOT mean shallow networks are practical — you might need billions of neurons in one layer
- It does NOT guarantee that gradient descent will find the optimal weights
- Deep networks are empirically far more parameter-efficient than wide shallow ones

> **Interview tip:** "The universal approximation theorem says an MLP *can* approximate anything, but depth gives you exponential efficiency. A deep ResNet with 50 layers and 25M parameters outperforms a single-hidden-layer network that would need billions of neurons for the same task. Depth is about compositionality — each layer builds increasingly abstract features."

---

# **2. CNNs — Convolutional Neural Networks**

---

## **2.1 The Convolution Operation**

A convolution slides a learnable filter (kernel) over the input, computing element-wise multiplications and summing them into a single output value. This operation is **translation-equivariant** — a pattern is detected regardless of where it appears in the image.

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)
$$

```
┌──────────────────────────────────────────────────────────────┐
│                    CONVOLUTION OPERATION                       │
│                                                               │
│   Input (5×5)          Filter (3×3)        Output (3×3)       │
│   ┌─────────────┐     ┌─────────┐        ┌─────────┐         │
│   │ 1  0  1  0  1│     │ 1  0  1 │        │ 4  3  4 │         │
│   │ 0  1  0  1  0│     │ 0  1  0 │        │ 2  4  3 │         │
│   │ 1  0  1  0  1│  *  │ 1  0  1 │   =    │ 4  3  4 │         │
│   │ 0  1  0  1  0│     └─────────┘        └─────────┘         │
│   │ 1  0  1  0  1│                                            │
│   └─────────────┘                                             │
│                                                               │
│   Output size = (W - K + 2P) / S + 1                          │
│   W = input size, K = kernel size, P = padding, S = stride    │
└──────────────────────────────────────────────────────────────┘
```

### **Key Hyperparameters**

| Parameter | Description | Effect |
|---|---|---|
| **Filters (channels)** | Number of different kernels | Each learns a different feature (edges, textures, etc.) |
| **Kernel size** | Spatial extent of the filter (3×3, 5×5, 7×7) | Larger = wider receptive field per layer |
| **Stride** | Step size of the sliding window | Stride > 1 downsamples (reduces spatial dims) |
| **Padding** | Zero-padding around input border | `same` padding preserves spatial dimensions |
| **Dilation** | Spacing between kernel elements | Increases receptive field without adding parameters |

### **Why CNNs over MLPs for Images?**

| Property | MLP | CNN |
|---|---|---|
| **Parameter sharing** | Each weight is unique | Same filter reused across entire image |
| **Translation equivariance** | Must learn pattern at every location | Pattern detected regardless of position |
| **Parameters for 224×224 RGB** | 224×224×3 × 1024 = ~150M (first layer!) | 3×3×3×64 = ~1.7K (first conv layer) |
| **Spatial structure** | Flattened — loses 2D relationships | Preserves spatial hierarchy |

---

## **2.2 Pooling Operations**

```
┌────────────────────────────────────────────────────────────┐
│                     POOLING OPERATIONS                      │
│                                                             │
│   Max Pooling (2×2, stride 2):          Average Pooling:    │
│   ┌─────────┐     ┌─────┐              ┌─────────┐  ┌───┐ │
│   │ 1  3 │ 2  1│   │ 3  6│              │ 1  3│2  1│  │2  4││
│   │ 4  6 │ 8  2│ → │ 8  8│              │ 4  6│8  2│→ │4  5││
│   │─────────│     └─────┘              └─────────┘  └───┘ │
│   │ 5  2 │ 3  1│   │ 5  3│                                 │
│   │ 7  4 │ 8  1│ → │ 7  8│   Global Average Pooling:       │
│   └─────────┘     └─────┘   [H×W×C] → [1×1×C]             │
│                              Mean over entire spatial dim    │
│                              Replaces FC layers (NiN, 2013) │
└────────────────────────────────────────────────────────────┘
```

| Pooling Type | Operation | Use Case |
|---|---|---|
| **Max Pooling** | Takes maximum value in window | Most common; preserves strongest activation |
| **Average Pooling** | Takes mean value in window | Smoother; used in some architectures |
| **Global Average Pooling (GAP)** | Average over entire spatial dimension | Replaces final FC layer, reduces overfitting |

---

## **2.3 The CNN Building Block: Conv → BN → ReLU → Pool**

Modern CNNs follow a consistent pattern within each block:

```
┌───────────────────────────────────────────────────────┐
│           STANDARD CNN BLOCK (Modern Pattern)          │
│                                                        │
│   Input                                                │
│     │                                                  │
│     ▼                                                  │
│   [Conv2d]  ← learnable filters extract features       │
│     │                                                  │
│     ▼                                                  │
│   [BatchNorm2d]  ← normalize activations per channel   │
│     │                                                  │
│     ▼                                                  │
│   [ReLU]  ← non-linearity, kills negative values       │
│     │                                                  │
│     ▼                                                  │
│   [MaxPool2d]  ← downsample spatial dimensions 2×       │
│     │                                                  │
│     ▼                                                  │
│   Output (half spatial dims, deeper channels)          │
│                                                        │
│   Note: Some architectures (ResNet) use stride-2 conv  │
│   instead of pooling for downsampling.                 │
└───────────────────────────────────────────────────────┘
```

```python
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
```

---

## **2.4 Key CNN Architectures — Evolution**

```
┌────────────────────────────────────────────────────────────────────┐
│              CNN ARCHITECTURE EVOLUTION (1998–2020)                 │
│                                                                     │
│  Year  Architecture   Depth   Params   Key Innovation               │
│  ─────────────────────────────────────────────────────────────────  │
│  1998  LeNet-5         5      60K     First practical CNN (digits)  │
│  2012  AlexNet         8      60M     ReLU, Dropout, GPU training   │
│  2014  VGGNet         16/19   138M    3×3 filters everywhere        │
│  2014  GoogLeNet      22      6.8M    Inception modules (parallel)  │
│  2015  ResNet        50/152   25M     Skip/residual connections     │
│  2017  DenseNet       121     8M      Dense connections (all→all)   │
│  2019  EfficientNet   B0–B7   5–66M   Compound scaling (NAS)       │
│  2020  ViT            —       86M     Patches → Transformer         │
└────────────────────────────────────────────────────────────────────┘
```

### **LeNet-5 (LeCun, 1998)**
- First successful CNN — digit recognition on MNIST
- Architecture: Conv→Pool→Conv→Pool→FC→FC→Output
- Introduced the foundational pattern of alternating convolution and pooling

### **AlexNet (Krizhevsky, 2012) — The Deep Learning Revolution**
- Won ImageNet 2012 by a massive margin (top-5 error: 16.4% vs 26.2%)
- Innovations: ReLU activation, dropout regularization, GPU training, data augmentation
- Proved that deep CNNs + big data + GPUs = breakthrough performance

### **VGGNet (Simonyan & Zisserman, 2014)**
- Key insight: Replace large filters with stacks of **3×3 filters**
- Two 3×3 convs have the same receptive field as one 5×5, but fewer parameters and more non-linearity
- VGG-16 (16 layers): simple, uniform architecture — easy to understand and modify

### **ResNet (He et al., 2015) — Skip Connections**

The most influential CNN architecture. Introduced **residual/skip connections** that enabled training networks with 100+ layers.

**The core idea:** Instead of learning $H(x)$, learn the residual $F(x) = H(x) - x$, so the layer computes $H(x) = F(x) + x$.

$$
\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

```
┌──────────────────────────────────────────────────────────────┐
│                   RESIDUAL BLOCK (ResNet)                      │
│                                                               │
│                   ┌─────────────────────────┐                 │
│                   │                         │                 │
│   x ──┬──→ [Conv 3×3] → [BN] → [ReLU]     │ (identity       │
│       │          │                          │  shortcut)      │
│       │    [Conv 3×3] → [BN]               │                 │
│       │          │                          │                 │
│       └──────────+──────────────────────────┘                 │
│                  │                                            │
│                [ReLU] → output = F(x) + x                     │
│                                                               │
│   WHY IT WORKS:                                               │
│   • Gradient flows directly through shortcut: ∂y/∂x = 1      │
│   • Even if F(x) gradients vanish, identity gradient = 1      │
│   • Worst case: layer learns F(x) = 0 → identity mapping     │
│   • Adding layers can't hurt — just learn to do nothing       │
└──────────────────────────────────────────────────────────────┘
```

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # skip connection
        return nn.functional.relu(out)
```

### **Inception / GoogLeNet (Szegedy, 2014)**
- Key idea: Apply **multiple filter sizes in parallel** (1×1, 3×3, 5×5) and concatenate
- 1×1 convolutions as bottleneck layers to reduce computation
- 22 layers deep but only 6.8M params (vs VGG's 138M)

```
┌──────────────────────────────────────────┐
│           INCEPTION MODULE                │
│                                           │
│              Input                         │
│         ┌──────┼──────┬──────┐            │
│         │      │      │      │            │
│       [1×1]  [1×1]  [1×1] [MaxPool]      │
│         │    [3×3]  [5×5]  [1×1]         │
│         │      │      │      │            │
│         └──────┴──────┴──────┘            │
│              Concatenate                   │
│                                           │
│   Multi-scale feature extraction           │
│   1×1 convs reduce channel dimension      │
└──────────────────────────────────────────┘
```

### **EfficientNet (Tan & Le, 2019)**
- Used **Neural Architecture Search (NAS)** to find optimal base architecture (B0)
- **Compound scaling**: uniformly scales depth, width, and resolution with fixed ratios
- EfficientNet-B7 achieves better accuracy than ResNet with 8.4× fewer parameters

$$
\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi
$$

Subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, where $\phi$ is the compound coefficient.

---

## **2.5 Transfer Learning with CNNs**

Transfer learning is using a model pre-trained on a large dataset (ImageNet: 1.2M images, 1000 classes) and adapting it to a new task with limited data.

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING STRATEGIES                    │
│                                                                   │
│   Strategy 1: Feature Extraction (Freeze backbone)                │
│   ┌─────────────────────────┐  ┌──────────────┐                  │
│   │ Pre-trained Conv Layers  │→│ New FC Head   │                  │
│   │ (FROZEN — no gradient)   │  │ (trained)     │                  │
│   └─────────────────────────┘  └──────────────┘                  │
│   Use when: Very small dataset (< 1K images)                      │
│                                                                   │
│   Strategy 2: Fine-Tuning (Unfreeze some/all layers)              │
│   ┌───────────┬──────────────┐  ┌──────────────┐                 │
│   │ Early Conv │ Later Conv   │→│ New FC Head   │                 │
│   │ (FROZEN)   │ (fine-tuned) │  │ (trained)     │                 │
│   └───────────┴──────────────┘  └──────────────┘                 │
│   Use when: Medium dataset (1K–100K), similar domain              │
│                                                                   │
│   Strategy 3: Full Fine-Tuning                                    │
│   ┌─────────────────────────┐  ┌──────────────┐                  │
│   │ All Layers Fine-Tuned    │→│ New FC Head   │                  │
│   │ (small learning rate)    │  │ (trained)     │                  │
│   └─────────────────────────┘  └──────────────┘                  │
│   Use when: Large dataset (100K+), different domain               │
└──────────────────────────────────────────────────────────────────┘
```

```python
import torchvision.models as models

# Load pre-trained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Strategy 1: Feature extraction — freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification head
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Strategy 2: Fine-tuning — unfreeze later layers
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
```

---

## **2.6 CNN Applications**

| Task | Architecture | Output |
|---|---|---|
| **Image Classification** | ResNet, EfficientNet | Single label per image |
| **Object Detection** | Faster R-CNN, YOLO, SSD | Bounding boxes + class labels |
| **Semantic Segmentation** | U-Net, DeepLab, FCN | Per-pixel class labels |
| **Instance Segmentation** | Mask R-CNN | Per-pixel labels + instance IDs |
| **Pose Estimation** | HRNet, OpenPose | Keypoint coordinates |

---

# **3. RNNs — Recurrent Neural Networks**

---

## **3.1 Vanilla RNN**

RNNs process sequential data by maintaining a **hidden state** that acts as memory, updated at each time step:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$
$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

```
┌───────────────────────────────────────────────────────────────┐
│                     VANILLA RNN (UNROLLED)                      │
│                                                                 │
│   x₁         x₂         x₃         x₄                         │
│   │           │           │           │                         │
│   ▼           ▼           ▼           ▼                         │
│  [RNN] ──→  [RNN] ──→  [RNN] ──→  [RNN]                       │
│  h₀→ h₁      h₁→ h₂     h₂→ h₃     h₃→ h₄                   │
│   │           │           │           │                         │
│   ▼           ▼           ▼           ▼                         │
│   y₁          y₂          y₃          y₄                       │
│                                                                 │
│   Same weights (W_hh, W_xh, W_hy) shared across all steps      │
│   Hidden state h_t carries "memory" of past inputs              │
└───────────────────────────────────────────────────────────────┘
```

**The Vanishing Gradient Problem in RNNs:**

During backpropagation through time (BPTT), gradients must flow through repeated multiplication by $W_{hh}$:

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_{hh} \cdot \text{diag}(\tanh'(z_t))
$$

Since $\tanh'(x) \leq 1$, this product shrinks exponentially for long sequences. A vanilla RNN effectively "forgets" inputs from more than ~10-20 steps back.

---

## **3.2 LSTM — Long Short-Term Memory**

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem by introducing a **cell state** $C_t$ (a "highway" for information) controlled by three **gates**:

```
┌──────────────────────────────────────────────────────────────────┐
│                        LSTM CELL                                  │
│                                                                   │
│                    Cell State C_{t-1} ─────────────→ C_t          │
│                         │         ×           +        │          │
│                         │     (forget)    (new info)   │          │
│                         │         │           │        │          │
│   ┌─────────────────────┼─────────┼───────────┼────────┤          │
│   │                     │         │           │        │          │
│   │   [Forget Gate] ────┘    [Input Gate]  [Gate Gate] │          │
│   │   f_t = σ(...)           i_t = σ(...)  g_t = tanh  │          │
│   │                                                    │          │
│   │   [Output Gate]                                    │          │
│   │   o_t = σ(...)                                     │          │
│   │        │                                           │          │
│   │        ▼                                           │          │
│   │   h_t = o_t ⊙ tanh(C_t) ──────────────────────────┘          │
│   │                                                               │
│   │   Inputs: x_t (current input), h_{t-1} (prev hidden state)   │
│   └───────────────────────────────────────────────────────────────┘
```

### **Gate Formulas (memorize these for interviews)**

**Forget Gate** — decides what to discard from cell state:
$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**Input Gate** — decides which new values to store:
$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

**Candidate Cell State** — creates new candidate values:
$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

**Cell State Update** — forget old info, add new info:
$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
$$

**Output Gate** — decides what to output:
$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

**Hidden State** — filtered cell state:
$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
$$

> **Key insight for interviews:** The cell state $C_t$ is the secret. It flows through time with only element-wise operations (multiply by forget gate, add input gate). No matrix multiplication means gradients don't vanish along this path. The gradient through the cell state is:
> $$\frac{\partial C_T}{\partial C_1} = \prod_{t=2}^{T} f_t$$
> As long as forget gates stay close to 1, gradients flow unimpeded.

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))         # [B, T, E]
        output, (hidden, cell) = self.lstm(embedded)       # output: [B, T, 2H]
        # Concatenate final forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # [B, 2H]
        return self.fc(self.dropout(hidden))               # [B, output_dim]
```

---

## **3.3 GRU — Gated Recurrent Unit**

GRU (Cho et al., 2014) simplifies LSTM by merging the cell state and hidden state into one, using only two gates:

**Reset Gate** — controls how much past information to forget:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$

**Update Gate** — controls the balance between old and new state (acts like combined forget + input gate):
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
$$

**Candidate Hidden State:**
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)
$$

**Hidden State Update:**
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

```
┌──────────────────────────────────────────────────────────┐
│                       GRU CELL                            │
│                                                           │
│   h_{t-1} ───────────────────────────────────→ (1-z_t)   │
│       │                                          ×       │
│       ├──→ [Reset Gate: r_t = σ(...)]            │       │
│       │         │                                +──→ h_t│
│       │    r_t ⊙ h_{t-1}                        │       │
│       │         │                            z_t ×       │
│       │    [Candidate: h̃_t = tanh(...)]          │       │
│       │                                          │       │
│       └──→ [Update Gate: z_t = σ(...)] ──────────┘       │
│                                                           │
│   LSTM vs GRU:                                            │
│   • GRU: 2 gates, no separate cell state → fewer params  │
│   • GRU: ~25% fewer parameters than LSTM                  │
│   • Performance: roughly comparable on most tasks          │
│   • GRU preferred when: limited data, faster training     │
│   • LSTM preferred when: longer sequences, more capacity   │
└──────────────────────────────────────────────────────────┘
```

---

## **3.4 Bidirectional RNNs**

Process sequences in both forward and backward directions, capturing both past and future context:

```
┌──────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL RNN                           │
│                                                               │
│   Forward:   x₁ → [h₁→] → x₂ → [h₂→] → x₃ → [h₃→]       │
│                                                               │
│   Backward:  x₁ ← [h₁←] ← x₂ ← [h₂←] ← x₃ ← [h₃←]     │
│                                                               │
│   Output at t:  h_t = [h_t→ ; h_t←]  (concatenated)          │
│                                                               │
│   Use cases:                                                  │
│   • NER: "I went to [New York] last week"                     │
│     → both surrounding words help identify entity              │
│   • Sentiment: future words clarify past ones                  │
│   • NOT for autoregressive generation (can't see future)      │
└──────────────────────────────────────────────────────────────┘
```

---

## **3.5 RNN Applications & Personal Experience**

| Application | Why RNN/LSTM | Example |
|---|---|---|
| **Time series forecasting** | Natural sequential structure | Stock prices, sensor data |
| **Sequence classification** | Variable-length input | Sentiment analysis, spam detection |
| **Sequence-to-sequence** | Input→output mapping | Translation (pre-transformer) |
| **Language modeling** | Next-token prediction | Text generation (pre-GPT era) |

### **Personal Experience: LSTM for Risk Scoring at Jet2**

> At Jet2, I built an LSTM-based model for temporal risk scoring on customer behavior sequences. The key was capturing how risk indicators evolved over time — a customer's booking patterns, cancellation history, and claim sequences formed natural temporal dependencies that a simple tabular model would miss. The LSTM's cell state preserved long-range signals (like a claim from 6 months ago) while the forget gate learned to downweight irrelevant historical noise. This approach improved early detection of high-risk patterns by capturing sequential dependencies that traditional feature engineering (lag features, rolling windows) couldn't fully represent.

---

# **4. Sequence-to-Sequence & Attention**

---

## **4.1 Encoder-Decoder Architecture**

The seq2seq model (Sutskever et al., 2014) uses two RNNs: an **encoder** that reads the input sequence into a fixed-length context vector, and a **decoder** that generates the output sequence from that context.

```
┌──────────────────────────────────────────────────────────────────┐
│                  ENCODER-DECODER (Seq2Seq)                        │
│                                                                   │
│   Encoder (processes input):                                      │
│   "How" → [h₁] → "are" → [h₂] → "you" → [h₃]                  │
│                                              │                    │
│                                         context = h₃              │
│                                              │                    │
│   Decoder (generates output):                │                    │
│                                        <SOS> → [s₁] → "Comment"  │
│                                                  │                │
│                                           "Comment" → [s₂] → "allez"│
│                                                         │         │
│                                                  "allez" → [s₃] → "vous"│
│                                                            │      │
│                                                     "vous" → [s₄] → <EOS>│
│                                                                   │
│   BOTTLENECK: Entire input compressed into single vector h₃      │
│   Long sequences → information loss → poor translations           │
└──────────────────────────────────────────────────────────────────┘
```

---

## **4.2 Teacher Forcing**

During training, the decoder receives the **ground-truth previous token** as input instead of its own prediction:

```
┌──────────────────────────────────────────────────────────────┐
│                    TEACHER FORCING                             │
│                                                               │
│   Without teacher forcing (autoregressive):                    │
│   <SOS> → ŷ₁ → ŷ₂ → ŷ₃    (errors compound!)               │
│           ↓                                                    │
│   If ŷ₁ is wrong, ŷ₂ and ŷ₃ are based on wrong input        │
│                                                               │
│   With teacher forcing:                                        │
│   <SOS> → ŷ₁    y₁ → ŷ₂    y₂ → ŷ₃   (ground truth input)  │
│                                                               │
│   Pros: Faster convergence, stable training                    │
│   Cons: Exposure bias — at inference, model sees its own       │
│         (potentially wrong) predictions, not ground truth      │
│                                                               │
│   Solution: Scheduled sampling — gradually reduce teacher      │
│   forcing probability during training                          │
└──────────────────────────────────────────────────────────────┘
```

---

## **4.3 Attention Mechanism**

Attention solves the bottleneck problem by letting the decoder look at **all** encoder hidden states, computing a weighted combination at each decoding step.

### **Bahdanau Attention (Additive, 2015)**

$$
e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{s}_{t-1} + \mathbf{W}_2 \mathbf{h}_i) \quad \text{(alignment score)}
$$
$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})} \quad \text{(attention weight via softmax)}
$$
$$
\mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{h}_i \quad \text{(context vector)}
$$

### **Luong Attention (Multiplicative, 2015)**

Three scoring functions:
$$
\text{dot:} \quad e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i
$$
$$
\text{general:} \quad e_{t,i} = \mathbf{s}_t^T \mathbf{W} \mathbf{h}_i
$$
$$
\text{concat:} \quad e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W}[\mathbf{s}_t; \mathbf{h}_i])
$$

```
┌──────────────────────────────────────────────────────────────────┐
│              ATTENTION IN SEQ2SEQ                                  │
│                                                                   │
│   Encoder states:  h₁    h₂    h₃    h₄                         │
│                    │      │      │      │                         │
│                    α₁=0.1 α₂=0.6 α₃=0.2 α₄=0.1  ← attention    │
│                    │      │      │      │            weights      │
│                    └──────┴──────┴──────┘                         │
│                           │                                       │
│                    context = Σ αᵢhᵢ   ← weighted sum             │
│                           │                                       │
│                    [s_t, context] → ŷ_t                           │
│                                                                   │
│   Each decoding step computes fresh attention over ALL encoder    │
│   states — decoder dynamically focuses on relevant input parts    │
│                                                                   │
│   This directly inspired self-attention in Transformers (2017)    │
└──────────────────────────────────────────────────────────────────┘
```

### **From Attention to Transformers**

The progression:
1. **Seq2seq** (2014): Fixed context vector → bottleneck
2. **Bahdanau attention** (2015): Dynamic context per step → still sequential (RNN)
3. **Self-attention** (2017): Every token attends to every other → fully parallel
4. **Transformer** (2017): Self-attention + positional encoding → replaced RNNs entirely

> **Interview tip:** "Attention was originally a patch for the seq2seq bottleneck. The Transformer's insight was: if attention is doing the heavy lifting anyway, why keep the RNN? Remove it entirely, use self-attention between all positions in parallel, and add positional encodings to preserve order. This gave us O(1) sequential operations instead of O(n), enabling massive parallelism on GPUs."

---

# **5. Autoencoders**

---

## **5.1 Standard Autoencoder**

An autoencoder learns a compressed representation by training a network to reconstruct its own input. The **encoder** maps input to a lower-dimensional **bottleneck** (latent space), and the **decoder** reconstructs from it.

$$
\text{Encoder:} \quad \mathbf{z} = f_\theta(\mathbf{x}) \quad \text{(compress)}
$$
$$
\text{Decoder:} \quad \hat{\mathbf{x}} = g_\phi(\mathbf{z}) \quad \text{(reconstruct)}
$$
$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \quad \text{(reconstruction loss)}
$$

```
┌────────────────────────────────────────────────────────────────┐
│                    STANDARD AUTOENCODER                          │
│                                                                  │
│   Input (784)                              Output (784)          │
│   ████████████                             ████████████          │
│       │                                         ▲                │
│       ▼                                         │                │
│   [FC: 512]  ──→  Encoder                  [FC: 512]  ← Decoder │
│       │                                         ▲                │
│       ▼                                         │                │
│   [FC: 256]                                [FC: 256]             │
│       │                                         ▲                │
│       ▼                                         │                │
│   [FC: 64]   ←── Bottleneck (z) ──→       [FC: 64]              │
│              ←── Latent Space ──→                                │
│                                                                  │
│   Goal: z captures the essential structure of x                  │
│   If reconstruction is good, z is a meaningful representation    │
└────────────────────────────────────────────────────────────────┘
```

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z
```

---

## **5.2 Variational Autoencoder (VAE)**

A VAE differs from a standard autoencoder in a fundamental way: instead of encoding to a single point in latent space, it encodes to a **probability distribution** — specifically, a Gaussian parameterized by mean $\mu$ and variance $\sigma^2$.

```
┌──────────────────────────────────────────────────────────────────┐
│                   VARIATIONAL AUTOENCODER                          │
│                                                                    │
│   Standard AE:    x → [Encoder] → z (point) → [Decoder] → x̂     │
│                                                                    │
│   VAE:            x → [Encoder] → μ, σ² (distribution)            │
│                                     │                              │
│                               z ~ N(μ, σ²)  ← sample              │
│                                     │                              │
│                              [Decoder] → x̂                        │
│                                                                    │
│   Key difference: z is SAMPLED from a learned distribution         │
│   → latent space is smooth and continuous                          │
│   → can generate NEW data by sampling z ~ N(0, I)                 │
└──────────────────────────────────────────────────────────────────┘
```

### **The Reparameterization Trick**

Sampling is not differentiable — we can't backpropagate through a random sampling operation. The trick: express the sample as a deterministic function of the parameters plus external noise:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

```
┌───────────────────────────────────────────────────────────────┐
│              REPARAMETERIZATION TRICK                           │
│                                                                 │
│   Problem:  z ~ N(μ, σ²) ← can't backprop through sampling     │
│                                                                 │
│   Solution: z = μ + σ ⊙ ε,   ε ~ N(0, I)                      │
│                                                                 │
│   Now gradients flow through μ and σ:                           │
│   ∂z/∂μ = 1,   ∂z/∂σ = ε                                      │
│                                                                 │
│   The randomness (ε) is external — not part of the graph        │
└───────────────────────────────────────────────────────────────┘
```

### **VAE Loss: ELBO (Evidence Lower Bound)**

The VAE loss has two terms:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL Regularization}}
$$

| Term | Role | Intuition |
|---|---|---|
| **Reconstruction** | How well can decoder reconstruct x from z? | Forces z to capture information |
| **KL Divergence** | How close is $q(z|x)$ to the prior $N(0, I)$? | Forces latent space to be smooth, regular |

The KL term has a closed-form solution for Gaussians:

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

---

## **5.3 Denoising Autoencoder**

Adds noise to the input and trains the network to reconstruct the clean original:

$$
\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
$$
$$
\mathcal{L} = \|\mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}}))\|^2
$$

**Why this helps:** Prevents the autoencoder from learning the identity function. The model must learn the underlying structure/manifold to reconstruct clean data from noisy input. This is the core idea behind **diffusion models** — denoising autoencoders applied iteratively.

---

## **5.4 Autoencoder Applications**

| Application | How | Example |
|---|---|---|
| **Dimensionality Reduction** | Use encoder output z as compressed features | Alternative to PCA (non-linear) |
| **Anomaly Detection** | Train on normal data; high reconstruction error = anomaly | Fraud detection, defect detection |
| **Generative (VAE)** | Sample z ~ N(0, I), decode to generate new data | Image generation, molecular design |
| **Denoising** | Remove noise from images, audio | Image restoration |
| **Pre-training** | Use encoder as feature extractor for downstream tasks | Semi-supervised learning |

---

# **6. GANs — Generative Adversarial Networks**

---

## **6.1 The Adversarial Framework**

GANs (Goodfellow et al., 2014) consist of two neural networks in a **minimax game**:

- **Generator (G):** Takes random noise $z \sim p(z)$ and generates fake data $G(z)$
- **Discriminator (D):** Classifies data as real (from training set) or fake (from generator)

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

```
┌──────────────────────────────────────────────────────────────────────┐
│                     GAN ARCHITECTURE                                  │
│                                                                       │
│   Random Noise                                                        │
│   z ~ N(0, I) ──→ [Generator G] ──→ Fake Image ──┐                  │
│                    (learns to                      │                  │
│                     fool D)                        ▼                  │
│                                              [Discriminator D]        │
│                                              Real or Fake?            │
│   Training Data ──→ Real Image ──────────────→     ▲                 │
│                                                    │                  │
│                                              [Loss → Update]          │
│                                                                       │
│   D's goal: Correctly classify real vs fake (maximize accuracy)       │
│   G's goal: Generate data that D classifies as real (fool D)          │
│                                                                       │
│   At equilibrium: D(G(z)) = 0.5 (can't distinguish)                  │
│                   G has learned the data distribution                  │
└──────────────────────────────────────────────────────────────────────┘
```

### **Training Procedure (Alternating)**

```
For each training iteration:
    ┌────────────────────────────────────────────────────────┐
    │  Step 1: Train Discriminator (k steps, typically k=1)  │
    │                                                        │
    │  • Sample real batch x from dataset                    │
    │  • Sample noise z, generate fake = G(z)                │
    │  • L_D = -[log D(x) + log(1 - D(G(z)))]              │
    │  • Update D to maximize classification accuracy        │
    ├────────────────────────────────────────────────────────┤
    │  Step 2: Train Generator (1 step)                      │
    │                                                        │
    │  • Sample noise z, generate fake = G(z)                │
    │  • L_G = -log D(G(z))   ← or log(1-D(G(z)))          │
    │  • Update G to fool D (make D output close to 1)       │
    └────────────────────────────────────────────────────────┘
```

---

## **6.2 Training Challenges**

### **Mode Collapse**
Generator learns to produce only a few types of outputs that fool the discriminator, ignoring the full diversity of the data distribution.

```
┌────────────────────────────────────────────────────────┐
│                   MODE COLLAPSE                         │
│                                                         │
│   Real Distribution:   Generated (collapsed):           │
│                                                         │
│      ●  ●  ●              ● ● ● ● ●                   │
│    ●        ●                                           │
│   ●    ●●    ●         All generated samples            │
│    ●        ●          cluster in one mode               │
│      ●  ●  ●                                            │
│   (3 modes)            (only 1 mode captured)            │
└────────────────────────────────────────────────────────┘
```

### **Training Instability**
- G and D must be balanced — if D is too strong, G gets no useful gradient; if D is too weak, G doesn't improve
- Oscillation: G and D keep "chasing" each other without converging
- Non-convergence: minimax games have no guarantee of reaching equilibrium with SGD

### **Solutions**

| Issue | Solution | Mechanism |
|---|---|---|
| Mode collapse | **WGAN** (Wasserstein loss) | Earth mover's distance — smoother gradients everywhere |
| Instability | **Spectral normalization** | Constrains Lipschitz constant of D |
| Instability | **Progressive growing** | Start with low-res, progressively add layers |
| Vanishing G gradients | **Non-saturating loss** | Use $-\log D(G(z))$ instead of $\log(1 - D(G(z)))$ |
| Evaluation | **FID score** | Frechet Inception Distance — measures quality + diversity |

---

## **6.3 GAN Variants**

| Variant | Key Innovation | Application |
|---|---|---|
| **DCGAN** (2015) | Conv layers in G/D, BN, no FC layers | Stable image generation |
| **WGAN** (2017) | Wasserstein distance, gradient penalty | Stable training, no mode collapse |
| **Conditional GAN** | G and D receive class label as input | Class-specific generation |
| **StyleGAN** (2019) | Mapping network + style injection at each layer | Photorealistic face generation, style control |
| **CycleGAN** (2017) | Unpaired image-to-image translation, cycle consistency | Horse↔Zebra, photo↔painting |
| **Pix2Pix** (2017) | Paired image-to-image translation, U-Net generator | Sketch→Photo, segmentation→image |

### **CycleGAN — Unpaired Translation**

```
┌──────────────────────────────────────────────────────────────┐
│                    CYCLEGAN                                    │
│                                                               │
│   Domain A (horses)              Domain B (zebras)            │
│       │                               │                       │
│       ▼                               ▼                       │
│   G_AB: A → B                    G_BA: B → A                 │
│       │                               │                       │
│       ▼                               ▼                       │
│   Fake B ──→ [D_B: real?]       Fake A ──→ [D_A: real?]     │
│       │                               │                       │
│       ▼                               ▼                       │
│   G_BA(G_AB(a)) ≈ a             G_AB(G_BA(b)) ≈ b           │
│   ↑ Cycle Consistency Loss       ↑ Cycle Consistency Loss    │
│                                                               │
│   No paired data needed! Just two collections of images.      │
└──────────────────────────────────────────────────────────────┘
```

```python
class Generator(nn.Module):
    """Simple DCGAN-style generator"""
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512 × 4 × 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            # 512 × 4 × 4 → 256 × 8 × 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            # 256 × 8 × 8 → 128 × 16 × 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            # 128 × 16 × 16 → 64 × 32 × 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            # 64 × 32 × 32 → 3 × 64 × 64
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, z.size(1), 1, 1))
```

---

# **7. Diffusion Models**

---

## **7.1 Core Concept: Denoise to Generate**

Diffusion models learn to generate data by learning to **reverse a gradual noising process**. They add Gaussian noise to data step-by-step (forward process), then train a neural network to reverse each step (reverse process).

```
┌──────────────────────────────────────────────────────────────────────┐
│                      DIFFUSION MODEL                                  │
│                                                                       │
│   Forward Process (fixed, no learning):                               │
│   x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ x_T ≈ N(0, I)                     │
│   [clean]  [add noise]  [add noise]     [pure noise]                 │
│                                                                       │
│   q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)              │
│                                                                       │
│   Reverse Process (learned):                                          │
│   x_T ──→ x_{T-1} ──→ ... ──→ x₁ ──→ x₀                           │
│   [noise]  [denoise]          [denoise]  [clean image!]               │
│                                                                       │
│   p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))     │
│                                                                       │
│   Training: predict the noise that was added at each step             │
│   L = E[‖ε - ε_θ(x_t, t)‖²]     (simple MSE on noise prediction)   │
└──────────────────────────────────────────────────────────────────────┘
```

### **Forward Process (Adding Noise)**

At each step $t$, add a small amount of Gaussian noise:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

A key property: we can jump directly to any step $t$ from $\mathbf{x}_0$:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

### **Reverse Process (Denoising)**

A neural network $\epsilon_\theta$ learns to predict the noise $\epsilon$ that was added:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$

---

## **7.2 U-Net Backbone**

The denoising network is typically a **U-Net** — an encoder-decoder with skip connections at each resolution level:

```
┌──────────────────────────────────────────────────────────────────┐
│                    U-NET FOR DIFFUSION                             │
│                                                                   │
│   Noisy Image x_t                   Predicted Noise ε_θ          │
│   + Timestep t                                                    │
│       │                                      ▲                    │
│       ▼                                      │                    │
│   [Conv Block 64]  ───── skip connection ──→ [Conv Block 64]     │
│       │                                      ▲                    │
│       ▼ (downsample)                         │ (upsample)         │
│   [Conv Block 128] ───── skip connection ──→ [Conv Block 128]    │
│       │                                      ▲                    │
│       ▼ (downsample)                         │ (upsample)         │
│   [Conv Block 256] ───── skip connection ──→ [Conv Block 256]    │
│       │                                      ▲                    │
│       ▼ (downsample)                         │ (upsample)         │
│   [Conv Block 512] ───── skip connection ──→ [Conv Block 512]    │
│       │                                      ▲                    │
│       ▼                                      │                    │
│       └─────────→ [Bottleneck 1024] ─────────┘                   │
│                                                                   │
│   + Self-attention layers at lower resolutions                    │
│   + Timestep embedding injected via FiLM conditioning             │
│   + Text conditioning via cross-attention (for text-guided gen)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## **7.3 Latent Diffusion & Stable Diffusion**

Running diffusion in pixel space is extremely expensive (e.g., 512×512×3 = 786K dimensions). **Latent diffusion** (Rombach et al., 2022) runs the diffusion process in a compressed latent space:

```
┌────────────────────────────────────────────────────────────────────┐
│                  LATENT DIFFUSION (Stable Diffusion)                │
│                                                                     │
│   Step 1: Pre-train a VAE                                           │
│   Image (512×512×3) → [VAE Encoder] → Latent (64×64×4)            │
│                                          ↑                          │
│   Step 2: Run diffusion in latent space  │                          │
│   Noise → [U-Net Denoiser] → Clean Latent                          │
│              ↑                                                      │
│         [Text Encoder (CLIP)] ← "a cat sitting on a throne"        │
│                                                                     │
│   Step 3: Decode to pixels                                          │
│   Clean Latent (64×64×4) → [VAE Decoder] → Image (512×512×3)      │
│                                                                     │
│   Why latent space?                                                 │
│   • 48× compression (512² × 3 → 64² × 4)                         │
│   • Same quality, massively reduced compute                         │
│   • Training: days on 8 GPUs instead of weeks on hundreds          │
└────────────────────────────────────────────────────────────────────┘
```

### **Key Models**

| Model | Organization | Key Feature |
|---|---|---|
| **DDPM** (2020) | Berkeley | Foundational paper, pixel-space diffusion |
| **Stable Diffusion** (2022) | Stability AI | Latent diffusion, open-source, CLIP text conditioning |
| **DALL-E 2** (2022) | OpenAI | CLIP prior + diffusion decoder |
| **DALL-E 3** (2023) | OpenAI | Native text rendering, tighter T5 integration |
| **Imagen** (2022) | Google | T5-XXL text encoder, pixel-space cascaded diffusion |
| **Flux** (2024) | Black Forest Labs | Transformer-based architecture replacing U-Net |

---

# **8. Vision Transformers (ViT) & CLIP**

---

## **8.1 Vision Transformer (ViT)**

ViT (Dosovitskiy et al., 2020) applies the transformer architecture directly to images by splitting an image into fixed-size patches and treating each patch as a "token."

```
┌──────────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER (ViT)                           │
│                                                                       │
│   Input Image (224×224)                                               │
│   ┌──┬──┬──┬──┬──┬──┬──┐                                            │
│   │P₁│P₂│P₃│P₄│P₅│P₆│P₇│  Split into 16×16 patches                │
│   ├──┼──┼──┼──┼──┼──┼──┤  → (224/16)² = 196 patches                 │
│   │P₈│P₉│..│..│..│..│Pₙ│                                            │
│   └──┴──┴──┴──┴──┴──┴──┘                                            │
│       │                                                               │
│       ▼                                                               │
│   [Linear Projection: 16×16×3 = 768 → D]   Patch Embeddings         │
│       │                                                               │
│       ▼                                                               │
│   [CLS] + [P₁ᵉ, P₂ᵉ, ..., Pₙᵉ] + Position Embeddings             │
│       │                                                               │
│       ▼                                                               │
│   ┌─────────────────────────────────────┐                            │
│   │     Transformer Encoder × L          │                            │
│   │  ┌─────────────────────────────┐    │                            │
│   │  │  Multi-Head Self-Attention   │    │                            │
│   │  │  + Residual + LayerNorm      │    │  × 12 layers (ViT-Base)   │
│   │  ├─────────────────────────────┤    │                            │
│   │  │  Feed-Forward Network        │    │                            │
│   │  │  + Residual + LayerNorm      │    │                            │
│   │  └─────────────────────────────┘    │                            │
│   └─────────────────────────────────────┘                            │
│       │                                                               │
│       ▼                                                               │
│   [CLS token output] → [MLP Head] → Classification                   │
│                                                                       │
│   ViT-Base:  12 layers, 768 dim, 12 heads, 86M params               │
│   ViT-Large: 24 layers, 1024 dim, 16 heads, 307M params             │
│   ViT-Huge:  32 layers, 1280 dim, 16 heads, 632M params             │
└──────────────────────────────────────────────────────────────────────┘
```

### **Why ViT Works for Vision**

| Aspect | CNNs | ViT |
|---|---|---|
| **Inductive bias** | Strong (locality, translation equivariance) | Weak (learns everything from data) |
| **Receptive field** | Local → builds up with depth | Global from layer 1 (every patch attends to all) |
| **Data requirement** | Works well with limited data | Needs large datasets (ImageNet-21K or JFT-300M) |
| **Scaling** | Diminishing returns beyond ~100 layers | Scales consistently with data + compute |
| **Performance** | Best with limited data | Best with large data + compute |

> **Key insight:** ViT has less inductive bias than CNNs — it doesn't assume locality or translation equivariance. This makes it worse with small data but gives it the freedom to learn more optimal representations with enough data. With sufficient pre-training data, ViT surpasses even the best CNNs.

---

## **8.2 CLIP — Contrastive Language-Image Pre-training**

CLIP (Radford et al., 2021) learns a **joint embedding space** for images and text using contrastive learning on 400M image-text pairs from the internet.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIP ARCHITECTURE                              │
│                                                                       │
│   Images                          Text Descriptions                   │
│   ┌──────┐                        ┌────────────────┐                 │
│   │ 🖼️₁ │──→ [Image Encoder] ──→ I₁  ·  T₁ ←── [Text Encoder] ←── "a dog" │
│   │ 🖼️₂ │──→ [Image Encoder] ──→ I₂  ·  T₂ ←── [Text Encoder] ←── "a cat" │
│   │ 🖼️₃ │──→ [Image Encoder] ──→ I₃  ·  T₃ ←── [Text Encoder] ←── "sunset"│
│   └──────┘                                                           │
│                                                                       │
│   Contrastive Loss Matrix (N × N):                                    │
│                    T₁     T₂     T₃                                  │
│           I₁  [  HIGH   low    low  ]   ← match                      │
│           I₂  [  low    HIGH   low  ]   ← match                      │
│           I₃  [  low    low    HIGH ]   ← match                      │
│                                                                       │
│   Maximize cosine similarity for matched (diagonal) pairs             │
│   Minimize cosine similarity for unmatched (off-diagonal) pairs       │
│                                                                       │
│   Image Encoder: ViT-L/14 (best) or ResNet-50/101                    │
│   Text Encoder:  Transformer (63M params, 12 layers)                 │
│   Projection: Both mapped to shared 512-d or 768-d space             │
└──────────────────────────────────────────────────────────────────────┘
```

### **CLIP's Superpower: Zero-Shot Classification**

No fine-tuning needed — just describe the classes in text:

```python
import torch
from PIL import Image

# Pseudo-code for zero-shot classification with CLIP
image_features = clip_model.encode_image(image)    # [1, 512]
text_features = clip_model.encode_text([            # [N, 512]
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
])

# Cosine similarity → pick highest
similarity = (image_features @ text_features.T)     # [1, N]
prediction = similarity.argmax(dim=-1)              # class index
```

### **Personal Experience: CLIP for Fashion Recommendation**

> In my fashion recommendation project, I used CLIP (ViT-B/32 and ViT-L/14) to build a multimodal recommendation system. The key insight was that CLIP's joint embedding space naturally bridges the gap between what users *describe* ("red casual dress for summer") and what products *look like*. I embedded both product images and textual descriptions into CLIP's shared space, then used cosine similarity for retrieval. For improved performance, I fine-tuned CLIP on domain-specific fashion data using contrastive learning with curated image-text pairs. This approach outperformed traditional visual similarity methods because it captured semantic attributes (style, occasion, material) that pixel-level features miss.

---

## **8.3 DINOv2 & SAM**

### **DINOv2 (Meta, 2023)**
- Self-supervised ViT training using self-distillation (no labels needed)
- Student network learns from teacher network (exponential moving average)
- Produces rich visual features that transfer well to any downstream task
- State-of-the-art for feature extraction without fine-tuning

### **SAM — Segment Anything Model (Meta, 2023)**

```
┌──────────────────────────────────────────────────────────────────┐
│                SEGMENT ANYTHING MODEL (SAM)                       │
│                                                                   │
│   Image → [ViT Image Encoder] → Image Embedding                  │
│                                       │                           │
│   Prompt ──────────────────────→ [Prompt Encoder]                │
│   (point, box, text, mask)            │                           │
│                                       ▼                           │
│                              [Lightweight Mask Decoder]           │
│                                       │                           │
│                                  Segmentation Masks               │
│                                                                   │
│   Pre-trained on 11M images with 1B masks                         │
│   Zero-shot: segment anything in any image                        │
│   Foundation model for segmentation tasks                         │
└──────────────────────────────────────────────────────────────────┘
```

---

# **9. Mixture of Experts (MoE)**

---

## **9.1 Core Concept: Sparse Activation**

MoE replaces a single large feed-forward layer with multiple **expert** sub-networks, only activating a few per input. A **router (gating network)** decides which experts to use.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MIXTURE OF EXPERTS (MoE)                          │
│                                                                       │
│                         Input Token x                                 │
│                              │                                        │
│                              ▼                                        │
│                    [Router / Gating Network]                          │
│                    g(x) = softmax(W_g · x)                           │
│                              │                                        │
│              ┌───────────────┼───────────────┐                       │
│              │               │               │                        │
│         g₁ = 0.7        g₂ = 0.0        g₃ = 0.3                    │
│              │                               │                        │
│              ▼                               ▼                        │
│         [Expert 1]     [Expert 2]       [Expert 3]                   │
│         (ACTIVE)       (INACTIVE)       (ACTIVE)                     │
│              │                               │                        │
│              └───────────┬───────────────────┘                       │
│                          │                                            │
│              Output = Σ gᵢ · Expertᵢ(x)                             │
│                    = 0.7 · E₁(x) + 0.3 · E₃(x)                     │
│                                                                       │
│   Total params: 8 experts × 7B each = 56B total                     │
│   Active params: top-2 experts × 7B = 14B per token                 │
│   → Large capacity, efficient inference                               │
└──────────────────────────────────────────────────────────────────────┘
```

### **Why MoE Matters**

$$
\text{Output} = \sum_{i=1}^{N} g_i(\mathbf{x}) \cdot E_i(\mathbf{x}), \quad \text{where only top-}k \text{ gates are non-zero}
$$

| Aspect | Dense Model | MoE Model |
|---|---|---|
| **Active params per token** | All parameters | Only top-k expert params |
| **Total params** | = active params | >> active params (e.g., 8×) |
| **Training FLOPs** | Proportional to total params | Proportional to active params |
| **Inference speed** | Slower for same quality | Faster per token for same quality |
| **Memory** | All params in memory | All params must be in memory (downside) |

### **Key Challenges**

| Challenge | Problem | Solution |
|---|---|---|
| **Load balancing** | Some experts get all the tokens, others are underused | Auxiliary load-balancing loss |
| **Expert collapse** | Router converges to always pick same experts | Noise injection in router, capacity factors |
| **Communication** | Experts may be on different GPUs | Expert parallelism, all-to-all communication |

---

## **9.2 Key MoE Models**

| Model | Experts | Active | Total Params | Key Innovation |
|---|---|---|---|---|
| **Switch Transformer** (2021) | 128 | Top-1 | 1.6T | Simplified routing, top-1 expert |
| **Mixtral 8x7B** (2024) | 8 | Top-2 | 47B (13B active) | Open-source, competitive with GPT-3.5 |
| **Mixtral 8x22B** (2024) | 8 | Top-2 | 176B (39B active) | Larger scale Mixtral |
| **DeepSeek-V2** (2024) | 160 | Top-6 | 236B (21B active) | Fine-grained experts, multi-head latent attention |
| **GPT-4** (rumored) | ~16 | Top-2 | ~1.8T | Not confirmed, widely speculated |

> **Interview tip:** "MoE gives you the performance of a very large model with the inference cost of a much smaller one. The key insight is that not every input needs every parameter — a math question activates different 'brain regions' than a poetry question. But you trade off memory (all experts must be loaded) and training complexity (load balancing is tricky)."

---

# **10. State Space Models & Mamba**

---

## **10.1 The Problem with Transformers for Long Sequences**

Self-attention is $O(n^2)$ in sequence length — doubling the context doubles both compute and memory quadratically. For very long sequences (100K+ tokens, genomics, audio), this becomes prohibitive.

```
┌──────────────────────────────────────────────────────────────────┐
│             ATTENTION COMPLEXITY vs SEQUENCE LENGTH                │
│                                                                   │
│   Sequence Length    Attention FLOPs     Memory (Attention Map)   │
│   ─────────────     ───────────────     ────────────────────     │
│   1K                1M                  4 MB                      │
│   4K                16M                 64 MB                     │
│   16K               256M                1 GB                      │
│   64K               4B                  16 GB                     │
│   256K              64B                 256 GB  ← impractical     │
│                                                                   │
│   State Space Models: O(n) compute AND memory                     │
│   → Can handle millions of tokens                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## **10.2 State Space Models (S4 → Mamba)**

State space models (SSMs) are inspired by continuous-time dynamical systems. They maintain a **hidden state** that is updated linearly at each time step:

**Continuous form:**
$$
\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t)
$$
$$
\mathbf{y}(t) = \mathbf{C}\mathbf{h}(t) + \mathbf{D}\mathbf{x}(t)
$$

**Discretized form (what we actually compute):**
$$
\mathbf{h}_t = \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t
$$
$$
\mathbf{y}_t = \mathbf{C} \mathbf{h}_t
$$

```
┌──────────────────────────────────────────────────────────────────┐
│                    STATE SPACE MODEL                               │
│                                                                   │
│   x₁        x₂        x₃        x₄                              │
│   │          │          │          │                               │
│   ▼          ▼          ▼          ▼                               │
│   B·x₁      B·x₂      B·x₃      B·x₄                           │
│   +          +          +          +                               │
│   A·h₀→ h₁  A·h₁→ h₂  A·h₂→ h₃  A·h₃→ h₄                     │
│   │          │          │          │                               │
│   C·h₁      C·h₂      C·h₃      C·h₄                           │
│   │          │          │          │                               │
│   ▼          ▼          ▼          ▼                               │
│   y₁         y₂         y₃         y₄                             │
│                                                                   │
│   Key property: Can be computed as convolution (parallel, fast)   │
│   OR as recurrence (sequential, efficient for generation)         │
│                                                                   │
│   Dual mode:                                                      │
│   • Training: Convolve → O(n log n) via FFT                      │
│   • Inference: Recurrence → O(1) per step (constant memory!)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## **10.3 Mamba — Selective State Space Model**

Mamba (Gu & Dao, 2023) makes SSMs **input-dependent** — the matrices A, B, C change based on the input, allowing the model to selectively remember or forget information.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MAMBA                                         │
│                                                                       │
│   S4 (previous SSM):        Mamba (selective SSM):                   │
│   A, B, C are FIXED         A, B, C are INPUT-DEPENDENT              │
│   (same transform for       (different transform based               │
│    every input)              on what the input is)                    │
│                                                                       │
│   Mamba Block:                                                        │
│   ┌────────────────────────────────────────┐                         │
│   │  Input x                               │                         │
│   │    │                                    │                         │
│   │    ├──→ [Linear] → [Conv1d] → [SiLU]  │                         │
│   │    │         │                          │                         │
│   │    │    [Selective SSM]                 │                         │
│   │    │    B(x), C(x), Δ(x) ← input-dep. │                         │
│   │    │         │                          │                         │
│   │    │    [× gate from parallel branch]   │                         │
│   │    │         │                          │                         │
│   │    └──→ [Linear] → output              │                         │
│   └────────────────────────────────────────┘                         │
│                                                                       │
│   Complexity: O(n) compute, O(1) memory per step at inference         │
│   vs Transformer: O(n²) compute, O(n) memory per step                │
│                                                                       │
│   Results: Matches Transformer quality at same FLOPs on most tasks    │
│   Especially strong on: long sequences, audio, genomics, time series  │
└──────────────────────────────────────────────────────────────────────┘
```

### **Mamba vs Transformer**

| Aspect | Transformer | Mamba |
|---|---|---|
| **Complexity** | O(n²) attention | O(n) linear |
| **Generation** | O(n) per token (KV cache) | O(1) per token (fixed state) |
| **Long sequences** | Degrades quadratically | Handles gracefully |
| **In-context learning** | Strong (full attention) | Competitive but still catching up |
| **Hardware efficiency** | Flash Attention helps | Custom CUDA kernels (hardware-aware) |
| **Maturity** | 7+ years, battle-tested | New (2023), rapidly evolving |

> **Interview tip:** "Mamba is the most promising alternative to transformers for long sequences. Its key innovation is making the state space model input-dependent — so it can selectively remember relevant information and forget noise, similar to how LSTM gates work but with O(n) complexity. For language modeling, it matches transformers at the same compute budget. The real advantage is generation: constant memory per step vs growing KV cache."

---

# **11. Architecture Selection Guide**

---

```
┌──────────────────────────────────────────────────────────────────────┐
│              WHEN TO USE WHICH ARCHITECTURE                           │
│                                                                       │
│   Data Type              Best Architecture          Why               │
│   ────────────────       ────────────────────       ──────────────    │
│   Images (small data)    CNN (ResNet/EfficientNet)  Strong inductive  │
│                                                      bias for vision  │
│                                                                       │
│   Images (large data)    ViT / DINOv2               Scales better,    │
│                                                      global attention │
│                                                                       │
│   Text / NLP             Transformer (decoder-only)  Self-attention   │
│                                                      + scaling laws   │
│                                                                       │
│   Short sequences        LSTM / GRU                  Simple, fewer    │
│   (< 500 tokens)                                     params needed    │
│                                                                       │
│   Long sequences         Mamba / SSM                 O(n) complexity  │
│   (> 100K tokens)                                                     │
│                                                                       │
│   Image generation       Diffusion (Stable Diff.)    Best quality     │
│                          or GAN (StyleGAN)           Fast generation  │
│                                                                       │
│   Image + Text           CLIP / multimodal           Joint embedding  │
│                                                                       │
│   Anomaly detection      Autoencoder / VAE           Reconstruction   │
│                                                      error as signal  │
│                                                                       │
│   Image segmentation     U-Net / SAM                 Skip connections │
│                                                      preserve detail  │
│                                                                       │
│   Large-scale LLM        MoE Transformer             Sparse compute,  │
│                          (Mixtral, DeepSeek)          efficient scaling│
└──────────────────────────────────────────────────────────────────────┘
```

---

# **12. Common Interview Questions (With Strong Answers)**

---

## **Q1: "CNN vs RNN — when would you use each?"**

> "CNNs and RNNs address fundamentally different data structures. **CNNs** exploit spatial locality — they're designed for grid-structured data like images where nearby pixels are correlated and the same pattern (edge, texture) can appear anywhere. The convolution operation provides translation equivariance and massive parameter sharing.
>
> **RNNs** are for sequential data with temporal dependencies — time series, text, audio — where order matters and information from past steps influences the current prediction. LSTMs and GRUs extend vanilla RNNs by solving the vanishing gradient problem through gating mechanisms.
>
> In practice, however, transformers have largely replaced RNNs for NLP tasks because self-attention captures long-range dependencies without the sequential bottleneck. And for some sequence tasks, 1D CNNs (like TCN) can be faster than RNNs. So the real modern decision is often: CNN for vision (especially with limited data), transformer for text and large-scale vision, and LSTM/GRU for simpler sequence tasks where transformer overhead isn't justified."

---

## **Q2: "Explain LSTM gates — what does each one do?"**

> "An LSTM has three gates and a cell state. The **forget gate** is a sigmoid that decides what information to discard from the cell state — if the forget gate outputs 0 for a dimension, that information is erased; if 1, it's fully retained. This allows the LSTM to forget irrelevant past information.
>
> The **input gate** (also sigmoid) decides which new values to add to the cell state, modulated by a candidate cell state computed with tanh. Together, they control what new information is written to memory.
>
> The **output gate** (sigmoid) decides what parts of the cell state to expose as the hidden state output. The cell state is passed through tanh (squashing to [-1, 1]) and multiplied by the output gate.
>
> The cell state is the key innovation — it flows through time with only element-wise operations (multiply by forget gate, add from input gate). No matrix multiplications in the cell state path means gradients can flow across many time steps without vanishing. This is why the gradient through the cell state is just a product of forget gate values, which the network learns to keep close to 1 when it needs to remember."

---

## **Q3: "What are skip/residual connections and why do they work?"**

> "Residual connections, introduced in ResNet, add the input of a block directly to its output: $y = F(x) + x$. Instead of learning the full mapping $H(x)$, the layers only learn the residual $F(x) = H(x) - x$.
>
> They solve two problems. First, **vanishing gradients**: during backpropagation, the gradient through a residual connection is at least 1 (from the identity path), regardless of how small the learned function's gradient is. This enables training networks with hundreds of layers. Second, **degradation**: paradoxically, deeper plain networks can have higher training error than shallower ones. With residual connections, adding a layer can't hurt — in the worst case, the layer learns $F(x) = 0$ and the block reduces to identity.
>
> Mathematically, the gradient flowing through a residual block is:
> $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$
>
> The identity matrix $I$ ensures the gradient never drops below 1. This is the same principle that makes LSTM cell states work — a highway for gradient flow."

---

## **Q4: "How do GANs work? What are the main challenges?"**

> "GANs consist of two networks — a generator and a discriminator — trained adversarially. The generator takes random noise and produces fake data; the discriminator classifies data as real or fake. The generator's objective is to fool the discriminator; the discriminator's objective is to correctly classify.
>
> Formally, this is a minimax game where D maximizes and G minimizes: $V = E[\log D(x)] + E[\log(1-D(G(z)))]$. At Nash equilibrium, the generator has learned the true data distribution and the discriminator outputs 0.5 everywhere.
>
> The main challenges are: **mode collapse** — the generator only produces a few types of outputs instead of the full distribution; **training instability** — balancing G and D is delicate; if D is too strong, G gets no gradient; if too weak, G has no incentive to improve; and **evaluation** — there's no single loss to monitor. We use metrics like FID (Frechet Inception Distance) and IS (Inception Score).
>
> WGAN addressed stability by replacing the JS divergence with Wasserstein distance, which provides gradients everywhere even when distributions don't overlap. Today, diffusion models have largely overtaken GANs for image generation quality, but GANs remain relevant for real-time generation due to their single forward-pass inference."

---

## **Q5: "What is a Vision Transformer and how does it differ from a CNN?"**

> "ViT splits an image into fixed-size patches (typically 16×16), linearly projects each patch into an embedding, adds positional encodings, prepends a learnable [CLS] token, and processes everything through a standard transformer encoder. The [CLS] token's final representation is used for classification.
>
> The key differences from CNNs: ViT has **no built-in spatial inductive bias** — it doesn't assume locality or translation equivariance. Every patch can attend to every other patch from the first layer, giving it a global receptive field immediately. CNNs build up receptive fields gradually through stacked layers.
>
> This means ViT needs more data to learn what CNNs know by design. With small datasets, CNNs win. But with large-scale pre-training (ImageNet-21K, JFT-300M), ViT surpasses CNNs because it's not constrained by CNN's locality assumption and can learn more flexible representations. DeiT showed that with proper training techniques (strong augmentation, distillation), ViTs can be competitive even with ImageNet-1K alone."

---

## **Q6: "Explain transfer learning — when and how would you use it?"**

> "Transfer learning uses knowledge from a model pre-trained on a large dataset (like ImageNet or CLIP's 400M image-text pairs) to improve performance on a new, usually smaller dataset. The core insight is that early layers learn universal features — edges, textures, shapes — that transfer across tasks, while later layers learn task-specific features.
>
> I'd use it in three scenarios: **Feature extraction** when I have very little data (freeze the backbone, train only a new classification head); **partial fine-tuning** when I have moderate data (freeze early layers, fine-tune later layers + head with a small learning rate); and **full fine-tuning** when I have sufficient data but want the pre-trained initialization for faster convergence.
>
> In my CLIP-based fashion recommendation project, I used transfer learning by taking CLIP's pre-trained image and text encoders (which learned general visual-semantic alignment) and fine-tuning them on fashion-specific image-text pairs. The pre-trained representations gave us a strong starting point, and domain-specific fine-tuning improved retrieval accuracy for fashion attributes like style, material, and occasion."

---

## **Q7: "What is the reparameterization trick in VAEs and why is it needed?"**

> "In a VAE, the encoder outputs parameters of a distribution — mean μ and log-variance log(σ²) — and we sample z from this distribution. The problem is that sampling is a stochastic operation and we can't compute gradients through it. If z = sample(N(μ, σ²)), there's no ∂z/∂μ because sampling is not a deterministic function of the parameters.
>
> The reparameterization trick expresses the same random variable as: z = μ + σ ⊙ ε, where ε ~ N(0, I). Now z is a deterministic, differentiable function of μ and σ (the learnable parameters), plus external randomness ε that doesn't depend on the parameters. We get ∂z/∂μ = 1 and ∂z/∂σ = ε, so gradients flow normally through the network.
>
> This was one of the key insights that made VAEs trainable with standard backpropagation. Without it, we'd need high-variance REINFORCE-style gradient estimators."

---

## **Q8: "Compare diffusion models vs GANs for image generation."**

> "Diffusion models and GANs represent two paradigms for generative modeling. **GANs** generate images in a single forward pass through a generator — fast inference but tricky to train due to the adversarial dynamics, mode collapse, and the need to carefully balance generator and discriminator.
>
> **Diffusion models** generate images by iteratively denoising from pure noise over many steps (typically 20-1000 steps). They're trained with a simple MSE loss on noise prediction, making training much more stable. The downside is slow inference — each step requires a full forward pass through the U-Net.
>
> In terms of quality, diffusion models now produce higher-quality, more diverse images (lower FID scores). They handle complex scenes and text-guided generation better. But for real-time applications, GANs are still preferred because of single-step generation.
>
> The field is converging though — consistency models and distilled diffusion models can generate in 1-4 steps, and Stable Diffusion + LCM (Latent Consistency Models) can now produce images in ~1 second. Meanwhile, latent diffusion reduced the computational cost by running diffusion in a compressed latent space rather than pixel space."

---

## **Q9: "What is Mixture of Experts and why is it used in large language models?"**

> "MoE replaces the dense feed-forward layer in a transformer with multiple expert sub-networks, where a routing network selects the top-k experts per token. The key benefit is decoupling total model capacity from per-token compute cost.
>
> For example, Mixtral 8×7B has 47 billion total parameters but only activates ~13 billion per token (top-2 of 8 experts). This means it has the knowledge capacity of a ~47B model but the inference speed closer to a 13B model. It achieves quality competitive with much larger dense models like LLaMA-2 70B while being significantly faster.
>
> The main challenges are load balancing (ensuring all experts get used, not just a few popular ones — solved with auxiliary balancing losses) and the memory footprint (all expert weights must be in memory even though most are inactive per token). Expert parallelism across GPUs helps with the latter but adds communication overhead.
>
> MoE is becoming the default for frontier models because the scaling curve of adding more experts is more compute-efficient than making existing layers wider. DeepSeek-V2 pushed this further with 160 fine-grained experts and multi-head latent attention."

---

## **Q10: "What is the difference between a standard autoencoder and a VAE?"**

> "A standard autoencoder maps inputs to a fixed point in latent space and reconstructs from that point. It's a purely deterministic encoding with no constraints on the latent space structure. The problem is that the latent space can be irregular — gaps between encoded points mean that sampling random latent vectors often produces garbage when decoded.
>
> A VAE maps inputs to a *distribution* (parameterized by mean and variance), samples from that distribution, and reconstructs. The KL divergence term in the loss regularizes the latent space to be close to a standard normal distribution N(0, I). This makes the latent space smooth and continuous — you can sample any point from N(0, I), decode it, and get a plausible output. Interpolating between two points produces smooth transitions.
>
> The trade-off: VAEs produce slightly blurrier outputs because the KL term prevents the model from encoding exact information. There's a reconstruction-regularization tension. Standard autoencoders give sharper reconstructions but can't generate new samples meaningfully."

---

# **13. Key Takeaways**

---

```
┌──────────────────────────────────────────────────────────────────────┐
│           KEY TAKEAWAYS FOR DEEP LEARNING ARCHITECTURE INTERVIEWS     │
│                                                                       │
│  1. FUNDAMENTALS MATTER                                               │
│     • Backprop = chain rule applied recursively                       │
│     • Vanishing gradients → ReLU, residual connections, BN            │
│     • Universal approx. theorem: MLPs CAN, depth makes it PRACTICAL  │
│                                                                       │
│  2. CNNs: THE VISION WORKHORSE                                       │
│     • Conv → BN → ReLU → Pool (the building block)                   │
│     • ResNet skip connections: most impactful innovation              │
│     • Transfer learning: almost never train from scratch              │
│                                                                       │
│  3. RNNs → TRANSFORMERS: THE SEQUENCE EVOLUTION                      │
│     • Vanilla RNN → LSTM (gates solve vanishing gradients)            │
│     • Seq2seq bottleneck → attention → self-attention → transformers  │
│     • Transformers won because parallelism + scaling                  │
│                                                                       │
│  4. GENERATIVE MODELS: THE NEW FRONTIER                              │
│     • VAE: smooth latent space, blurry outputs, principled math       │
│     • GAN: sharp outputs, unstable training, mode collapse            │
│     • Diffusion: best quality, slow inference (getting faster)        │
│                                                                       │
│  5. MODERN TRENDS                                                     │
│     • ViT / CLIP: transformers conquering vision too                  │
│     • MoE: sparse activation for efficient scaling                    │
│     • Mamba/SSM: O(n) alternative for long sequences                  │
│     • Multimodal: CLIP, Flamingo, GPT-4V — text + image together     │
│                                                                       │
│  6. YOUR DIFFERENTIATORS                                              │
│     • LSTM for risk scoring at Jet2 (temporal sequence modeling)      │
│     • CLIP for fashion recommendation (multimodal retrieval)          │
│     • PyTorch + TensorFlow hands-on with both frameworks              │
│     • Transfer learning in production CV systems                      │
│     • End-to-end: architecture selection → training → deployment      │
│                                                                       │
│  GOLDEN RULE: Know WHY each architecture exists (what problem it      │
│  solves), not just HOW it works. Interviewers want to see that you    │
│  can choose the right tool for the right problem.                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

*Last updated: February 2026*
