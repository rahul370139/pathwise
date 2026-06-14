# Loss Functions — Interview Preparation Guide

**Candidate:** Rahul Sharma | MS Data Science, UMD | 4+ years experience  
**Core Skills:** LLMs, Deep Learning, PyTorch, TensorFlow, RAG Systems  
**Document Scope:** Comprehensive coverage of loss functions for Data Scientist / ML Engineer interviews

---

## 1. What Are Loss Functions?

A **loss function** (also called cost function or objective function) quantifies the discrepancy between a model's predictions and the ground-truth targets. It is the single scalar signal that drives gradient-based optimization during training.

| Concept | Role |
|---|---|
| **Forward pass** | Model produces predictions $\hat{y}$ |
| **Loss computation** | $\mathcal{L}(y, \hat{y})$ measures error |
| **Backward pass** | Gradients $\nabla_\theta \mathcal{L}$ flow back through the network |
| **Parameter update** | Optimizer (SGD, Adam, etc.) adjusts weights $\theta$ to minimize $\mathcal{L}$ |

**Why the choice of loss function matters:**
- Defines *what* the model learns to optimize
- Affects convergence speed, stability, and final performance
- Must align with the evaluation metric and business objective
- Different tasks (regression, classification, generation, alignment) require fundamentally different losses

---

## 2. Regression Losses

### 2.1 Mean Squared Error (MSE / L2 Loss)

**Formula:**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**When to use:**
- Standard regression tasks (price prediction, sensor forecasting)
- When large errors are disproportionately costly
- Value heads in reinforcement learning (predicting expected reward)

**Why it works:**
- Differentiable everywhere → clean gradients
- Squaring penalizes large errors heavily → model focuses on outliers
- Corresponds to maximum likelihood estimation (MLE) under Gaussian noise assumption

**Limitations:**
- Sensitive to outliers (a single extreme point dominates the gradient)
- Scale-dependent (loss magnitude depends on target units)

**PyTorch implementation:**

```python
import torch
import torch.nn as nn

# Built-in
criterion = nn.MSELoss(reduction='mean')  # 'mean', 'sum', or 'none'
loss = criterion(predictions, targets)

# Manual
loss_manual = torch.mean((predictions - targets) ** 2)
```

---

### 2.2 Mean Absolute Error (MAE / L1 Loss)

**Formula:**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**When to use:**
- Targets contain outliers you don't want to over-penalize
- Robust regression (housing prices with extreme values, skewed distributions)
- When you want the model to predict the **median** rather than the mean

**Why it works:**
- Linear penalty → every error contributes equally regardless of magnitude
- More robust to outliers than MSE
- Corresponds to MLE under Laplace noise assumption

**Limitations:**
- Gradient is constant (±1) at all non-zero errors → can be unstable near zero
- Not differentiable at exactly zero (subgradient is used)
- Convergence can be slower than MSE for well-behaved data

**PyTorch:**

```python
criterion = nn.L1Loss(reduction='mean')
loss = criterion(predictions, targets)
```

---

### 2.3 Huber Loss (Smooth L1 Loss)

**Formula:**

$$\mathcal{L}_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**When to use:**
- When you want the **best of MSE + MAE**: quadratic for small errors, linear for large errors
- Object detection bounding-box regression (Faster R-CNN, SSD)
- **RLHF value heads** — predicting expected reward where outlier returns exist
- Any regression task with occasional noisy labels

**Why it works:**
- Smooth and differentiable everywhere (unlike MAE at zero)
- Outlier-robust (unlike MSE)
- The hyperparameter $\delta$ controls the transition point; often $\delta = 1.0$

**PyTorch:**

```python
criterion = nn.SmoothL1Loss(beta=1.0)  # beta = delta
loss = criterion(predictions, targets)

# Or explicitly
criterion = nn.HuberLoss(delta=1.0)
```

---

### 2.4 Root Mean Squared Error (RMSE)

**Formula:**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

**Relation to MSE:**
- Same unit as the target variable (interpretable)
- Commonly used as an **evaluation metric** rather than a training loss
- Gradient of RMSE is proportional to gradient of MSE (minimizing MSE minimizes RMSE)
- Often reported in Kaggle competitions and business contexts because of interpretability

---

### Regression Loss Comparison

| Loss | Outlier Robustness | Differentiability | Predicts | Use Case |
|---|---|---|---|---|
| MSE | Low | Everywhere | Mean | Standard regression |
| MAE | High | Not at 0 | Median | Robust regression |
| Huber | Medium–High | Everywhere | Mean (approx) | Noisy regression, RL value heads |
| RMSE | Low | Everywhere | Mean | Reporting metric |

---

## 3. Classification Losses

### 3.1 Cross-Entropy Loss (CE)

> **This is the CORE loss function for training LLMs.** Every GPT, LLaMA, Mistral, and Gemma model is pre-trained by minimizing cross-entropy over the vocabulary at each token position.

**Formula (multi-class):**

$$\text{CE} = -\sum_{c=1}^{C} y_c \log(\hat{p}_c)$$

For one-hot labels (standard case, single correct class $k$):

$$\text{CE} = -\log(\hat{p}_k)$$

where $\hat{p} = \text{softmax}(z)$ and $z$ are the logits.

**For LLMs (next-token prediction):**

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

This is the **autoregressive cross-entropy loss** — at every position $t$, the model predicts a probability distribution over the vocabulary $V$ (e.g., 32K–128K tokens), and we penalize low probability on the correct next token.

**Why CE is perfect for language modeling:**
- Vocabulary is a categorical distribution → CE is the natural loss
- Minimizing CE = maximizing log-likelihood of the training corpus
- Gradient is simple: $\nabla_{z_k} \text{CE} = \hat{p}_k - y_k$ (predicted minus true)
- Works with teacher forcing (ground-truth tokens as input)

**PyTorch implementation:**

```python
import torch
import torch.nn as nn

# CrossEntropyLoss combines LogSoftmax + NLLLoss internally
# Input: raw logits (NOT softmax outputs), shape [batch, num_classes]
# Target: class indices, shape [batch]
criterion = nn.CrossEntropyLoss(
    weight=None,            # optional class weights for imbalance
    ignore_index=-100,      # ignore padding tokens (critical for LLMs)
    label_smoothing=0.0     # optional smoothing (0.1 common in transformers)
)

# LLM example: logits shape [batch, seq_len, vocab_size]
logits = model(input_ids)                    # [B, T, V]
logits_flat = logits.view(-1, vocab_size)    # [B*T, V]
targets_flat = target_ids.view(-1)           # [B*T]
loss = criterion(logits_flat, targets_flat)

# With label smoothing (used in many transformer papers)
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Label Smoothing:** Instead of hard one-hot targets, distribute a small probability $\epsilon$ across all classes:

$$y_c^{\text{smooth}} = (1 - \epsilon) \cdot y_c + \frac{\epsilon}{C}$$

This prevents the model from becoming overconfident and improves generalization.

---

### 3.2 Binary Cross-Entropy (BCE)

**Formula:**

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**When to use:**
- **Binary classification** (spam/not-spam, positive/negative)
- **Multi-label classification** (an image can be "cat" AND "outdoor" simultaneously)
- Each output neuron is independent with sigmoid activation

**Sigmoid + BCE pattern:**

```python
# Option 1: Apply sigmoid yourself + BCELoss
criterion = nn.BCELoss()
probs = torch.sigmoid(logits)
loss = criterion(probs, targets)  # targets are floats in [0, 1]

# Option 2 (PREFERRED): BCEWithLogitsLoss (numerically stable)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)  # pass raw logits, sigmoid is internal
```

**Why prefer `BCEWithLogitsLoss`:** It fuses sigmoid and BCE in a numerically stable way using the log-sum-exp trick, avoiding the $\log(0)$ problem.

---

### 3.3 Negative Log-Likelihood (NLL)

**Formula:**

$$\text{NLL} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{p}_{y_i})$$

**Relation to Cross-Entropy:**
- NLL and CE are **equivalent** when targets are one-hot encoded
- In PyTorch, `nn.NLLLoss` expects **log-probabilities** as input (apply `log_softmax` first)
- `nn.CrossEntropyLoss` = `nn.LogSoftmax` + `nn.NLLLoss` combined

```python
# These two are equivalent:
# Approach 1
log_probs = torch.log_softmax(logits, dim=-1)
loss = nn.NLLLoss()(log_probs, targets)

# Approach 2
loss = nn.CrossEntropyLoss()(logits, targets)
```

**Interview note:** When someone says "NLL loss for training a language model," they mean cross-entropy. The terms are used interchangeably in the LLM literature.

---

### 3.4 Focal Loss

**Formula:**

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the predicted probability for the true class, $\alpha_t$ is a class-balancing weight, and $\gamma \geq 0$ is the focusing parameter (typically $\gamma = 2$).

**When to use:**
- **Severe class imbalance** (e.g., 1:1000 ratio of positives to negatives)
- **Object detection** — most anchor boxes are background (easy negatives dominate standard CE)
- Introduced in RetinaNet (Lin et al., 2017)

**How it handles imbalance:**
- When $\gamma = 0$, focal loss reduces to standard CE
- For well-classified examples ($p_t \to 1$), the $(1-p_t)^\gamma$ factor → 0, **down-weighting easy examples**
- For misclassified examples ($p_t \to 0$), the factor → 1, full loss is preserved
- The model focuses its learning capacity on **hard, misclassified examples**

**PyTorch implementation:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss
```

---

### 3.5 Hinge Loss (SVM Loss)

**Formula:**

$$\mathcal{L}_{\text{hinge}} = \max(0, 1 - y \cdot f(x))$$

where $y \in \{-1, +1\}$ and $f(x)$ is the raw score (not a probability).

**Multi-class (Weston-Watkins):**

$$\mathcal{L} = \sum_{j \neq y_i} \max(0, f_j(x) - f_{y_i}(x) + \Delta)$$

**When to use:**
- Support Vector Machines (SVMs)
- Maximum-margin classification
- Tasks where you want a clear decision boundary with a margin $\Delta$

**Key properties:**
- Non-differentiable at the hinge point (subgradient used)
- Only penalizes samples within the margin or misclassified → sparse gradients
- Not commonly used in deep learning (CE preferred), but foundational interview concept

---

## 4. Embedding / Metric Learning Losses

> These losses are critical for **RAG systems, dense retrieval, semantic search, and representation learning** — directly relevant to building production LLM applications.

### 4.1 Contrastive Loss (InfoNCE)

**InfoNCE Formula:**

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\sum_{j=0}^{K} \exp(\text{sim}(q, k_j) / \tau)}$$

where $q$ is the query, $k^+$ is the positive key, $k_j$ includes the positive and all negatives, $\text{sim}$ is cosine similarity, and $\tau$ is the temperature.

**Where it is used:**
- **CLIP** (OpenAI): aligns image and text embeddings — images and their captions are positives; other pairs in the batch are negatives
- **SimCLR**: self-supervised visual representation learning — two augmented views of the same image are positives
- **Dense Retrievers for RAG** (DPR, Contriever, E5, BGE): query-document pairs from labeled data are positives; other documents in the batch are hard negatives

**Why it's powerful for RAG:**
- Trains a bi-encoder to place semantically similar items close in embedding space
- At inference, use approximate nearest neighbor (ANN) search for fast retrieval
- The temperature $\tau$ controls the sharpness of the distribution (lower $\tau$ = harder contrasts)

**PyTorch sketch:**

```python
def info_nce_loss(query_emb, key_emb, temperature=0.07):
    # query_emb, key_emb: [batch_size, embed_dim], L2-normalized
    # Positive pairs: query_emb[i] <-> key_emb[i]
    logits = query_emb @ key_emb.T / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss
```

---

### 4.2 Triplet Loss

**Formula:**

$$\mathcal{L}_{\text{triplet}} = \max(0,\ d(a, p) - d(a, n) + \alpha)$$

where $a$ = anchor, $p$ = positive (same class), $n$ = negative (different class), $d$ is a distance metric (Euclidean or cosine), and $\alpha$ is the margin.

**When to use:**
- **Face identification** (FaceNet) — anchor face, same person = positive, different person = negative
- **Passage retrieval** — anchor query, relevant passage = positive, irrelevant passage = negative
- Fine-grained similarity learning where you have explicit triplets

**Hard negative mining:**
- **Semi-hard negatives:** $d(a,p) < d(a,n) < d(a,p) + \alpha$ (within margin but correctly ordered)
- **Hard negatives:** $d(a,n) < d(a,p)$ (incorrect ordering — most informative)
- Without proper mining, many triplets are trivial and contribute zero gradient

**PyTorch:**

```python
criterion = nn.TripletMarginLoss(margin=1.0, p=2)  # p=2 for Euclidean
loss = criterion(anchor, positive, negative)

# With cosine distance
criterion = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - nn.functional.cosine_similarity(x, y),
    margin=0.3
)
```

**Triplet vs. Contrastive loss for interviews:**

| Aspect | Triplet Loss | Contrastive (InfoNCE) |
|---|---|---|
| Input | (anchor, positive, negative) | (query, positive, in-batch negatives) |
| Negatives per sample | 1 | batch_size − 1 |
| Mining | Requires explicit hard negative mining | In-batch negatives are "free" |
| Modern preference | Less common now | Dominant in CLIP, DPR, SimCLR |

---

### 4.3 Cosine Similarity Loss

**Formula:**

$$\mathcal{L} = 1 - \cos(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$$

(for positive pairs; for negative pairs, the formulation varies)

**When to use:**
- **Embedding alignment** — forcing two representations into the same space
- Sentence-BERT (SBERT) training
- When magnitude shouldn't matter, only direction

**PyTorch:**

```python
criterion = nn.CosineEmbeddingLoss(margin=0.0)
# labels: +1 for similar pairs, -1 for dissimilar pairs
loss = criterion(embedding_a, embedding_b, labels)
```

---

### 4.4 Margin Ranking Loss

**Formula:**

$$\mathcal{L} = \max(0, -y \cdot (x_1 - x_2) + \text{margin})$$

where $y = +1$ means $x_1$ should rank higher than $x_2$, and $y = -1$ the reverse.

**When to use:**
- **Re-rankers in RAG pipelines**: a cross-encoder scores (query, document) pairs; margin ranking ensures the relevant document scores higher than irrelevant ones
- Learning-to-rank systems
- Preference learning (prefer response A over response B)

**PyTorch:**

```python
criterion = nn.MarginRankingLoss(margin=1.0)
loss = criterion(score_relevant, score_irrelevant, target=torch.ones_like(score_relevant))
```

---

### 4.5 NT-Xent (Normalized Temperature-Scaled Cross-Entropy)

**Formula:**

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**Context:**
- The loss used in **SimCLR** (Chen et al., 2020)
- A specific form of InfoNCE applied to augmented image pairs
- For a batch of $N$ images, create $2N$ augmented views; each image's two views form a positive pair; the remaining $2(N-1)$ views are negatives
- Temperature $\tau = 0.5$ is typical; lower values sharpen the contrastive signal

---

## 5. Generative Model Losses

### 5.1 GAN Loss (Minimax / Adversarial)

**Original formulation (Goodfellow et al., 2014):**

$$\min_G \max_D\ \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

| Component | Objective |
|---|---|
| **Discriminator** $D$ | Maximize: correctly classify real ($D(x) \to 1$) and fake ($D(G(z)) \to 0$) |
| **Generator** $G$ | Minimize: fool the discriminator ($D(G(z)) \to 1$) |

**Variants:**
- **Non-saturating GAN loss:** Generator maximizes $\mathbb{E}[\log D(G(z))]$ instead (better gradients early in training)
- **Wasserstein GAN (WGAN):** Uses Earth Mover's distance, removes log — more stable training
- **Least Squares GAN (LSGAN):** MSE instead of CE — smoother gradients

**Training instability issues:**
- Mode collapse (generator produces limited variety)
- Vanishing gradients when discriminator becomes too strong
- Careful hyperparameter tuning and architectural choices required

---

### 5.2 VAE Loss (Reconstruction + KL Divergence)

**Formula (ELBO — Evidence Lower Bound):**

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL Regularization}}$$

| Term | Purpose |
|---|---|
| **Reconstruction loss** | Ensure the decoder can reconstruct the input from the latent code (MSE for continuous, CE for discrete) |
| **KL divergence** | Regularize the encoder's posterior $q(z|x)$ to stay close to the prior $p(z) = \mathcal{N}(0, I)$ |

**For Gaussian encoder** $q(z|x) = \mathcal{N}(\mu, \sigma^2)$:

$$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

**KL annealing / $\beta$-VAE:** Weight the KL term with $\beta$ to control the trade-off:
- $\beta < 1$: better reconstruction, less structured latent space
- $\beta > 1$: more disentangled representations, worse reconstruction

---

### 5.3 Diffusion Loss

**Simplified denoising objective (DDPM, Ho et al., 2020):**

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

where:
- $x_t$ = noised input at timestep $t$
- $\epsilon$ = the actual noise added
- $\epsilon_\theta$ = neural network predicting the noise

**Key insight:** The model learns to predict the noise $\epsilon$ added at each timestep. This is mathematically equivalent to learning to denoise, and at inference, the model iteratively removes noise to generate samples.

**Used in:** Stable Diffusion, DALL-E 2, Imagen, Sora

---

## 6. Alignment Losses (RLHF / RLAIF)

> These are the most interview-relevant losses for LLM roles today. They define how we align pre-trained models with human preferences.

### 6.1 PPO Loss (Proximal Policy Optimization for RLHF)

**The RLHF pipeline:**

```
Step 1: Pre-train LLM with CE loss (next-token prediction)
    ↓
Step 2: Supervised Fine-Tuning (SFT) with CE on instruction-response pairs
    ↓
Step 3: Train a Reward Model (RM) on human preference data (ranking loss)
    ↓
Step 4: Optimize the LLM with PPO using the RM as the reward signal
```

**PPO Clipped Objective:**

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio between the new and old policy
- $\hat{A}_t$ is the advantage estimate (how much better was this action than average?)
- $\epsilon$ is the clip range (typically 0.2) — prevents too-large policy updates

**In the LLM context:**
- "State" = prompt + tokens generated so far
- "Action" = the next token
- "Reward" = reward model's score for the complete response
- A KL penalty $\beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ is added to prevent the policy from diverging too far from the SFT model

**Full PPO-RLHF objective:**

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + c_1 \mathcal{L}_{\text{value}} - c_2 H(\pi_\theta) + \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $\mathcal{L}_{\text{value}}$ is the value function loss (MSE or Huber), $H$ is entropy bonus (encourages exploration), and the KL term prevents reward hacking.

---

### 6.2 DPO Loss (Direct Preference Optimization)

**Formula:**

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right) \right]$$

where:
- $y_w$ = chosen (winning) response
- $y_l$ = rejected (losing) response
- $\pi_\theta$ = the policy being trained
- $\pi_{\text{ref}}$ = the reference (SFT) model
- $\beta$ = temperature parameter
- $\sigma$ = sigmoid function

**Why DPO is a simpler alternative to PPO:**

| Aspect | PPO (RLHF) | DPO |
|---|---|---|
| Requires reward model? | Yes (separate model) | No (implicit) |
| Requires RL training loop? | Yes (complex, unstable) | No (supervised loss) |
| Training pipeline | Pretrain → SFT → RM → PPO | Pretrain → SFT → DPO |
| Memory overhead | High (4 models in memory) | Lower (2 models: policy + reference) |
| Stability | Tricky to tune | More stable |
| Used by | InstructGPT, early ChatGPT | LLaMA 2 Chat, Zephyr, many open models |

**Key insight for interviews:** DPO shows that you can derive the optimal policy directly from preference data without training a separate reward model. The reward is implicitly defined by the policy and reference model.

---

### 6.3 Reward-Weighted Cross-Entropy

**Formula:**

$$\mathcal{L} = -\sum_{t} r(y) \cdot \log P_\theta(y_t | y_{<t}, x)$$

**When to use:**
- A simpler alternative to PPO where you weight the standard CE loss by the reward
- High-reward responses get higher weight in the loss
- Used in REINFORCE-style alignment and some rejection sampling approaches (e.g., Best-of-N sampling for training)

---

### 6.4 KL Divergence

**Formula:**

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

For continuous distributions:

$$D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx$$

**Properties:**
- $D_{KL} \geq 0$ (Gibbs' inequality); equals 0 iff $P = Q$
- **Asymmetric:** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
  - Forward KL ($D_{KL}(P \| Q)$): mean-seeking (Q covers all modes of P)
  - Reverse KL ($D_{KL}(Q \| P)$): mode-seeking (Q concentrates on one mode of P)

**Where KL divergence is used in ML:**

| Application | Role of KL |
|---|---|
| **Knowledge Distillation** | Match student's distribution to teacher's soft targets |
| **VAE Regularization** | Push encoder posterior toward standard normal prior |
| **RLHF/PPO** | Penalty to keep policy close to reference model |
| **Mutual Information** | $I(X;Y) = D_{KL}(P_{XY} \| P_X P_Y)$ |
| **Variational Inference** | Minimize $D_{KL}(q(z) \| p(z|x))$ |

**PyTorch:**

```python
import torch.nn.functional as F

# KL between two distributions (log_probs vs probs)
kl = F.kl_div(
    input=log_q,         # log probabilities of Q (approximation)
    target=p,            # probabilities of P (true distribution)
    reduction='batchmean',
    log_target=False     # set True if target is also log-probs
)
```

---

## 7. Self-Supervised Losses

### 7.1 Masked Language Modeling (MLM) — BERT

**Procedure:**
1. Randomly mask ~15% of tokens in the input
2. Replace with `[MASK]` (80%), random token (10%), or keep original (10%)
3. Predict the original token at each masked position

**Loss:**

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\backslash \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions and $x_{\backslash \mathcal{M}}$ is the input with masks.

**Key insight:** This is cross-entropy, but only computed at masked positions. The model uses **bidirectional context** (both left and right), making BERT an encoder model excellent for understanding tasks (classification, NER, QA).

---

### 7.2 Next-Token Prediction (Causal LM) — GPT

**Loss:**

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_1, \ldots, x_{t-1})$$

**Key insight:** This is the standard autoregressive cross-entropy loss. The causal mask ensures each position can only attend to previous positions. This is the pre-training objective for GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and virtually all modern LLMs.

---

### 7.3 Denoising Objectives — T5, BART

**T5 (Span Corruption):**
- Randomly replace contiguous spans of tokens with sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...)
- The model generates the original spans
- Loss: CE on the decoder output predicting the masked spans

**BART (Multiple Noising Strategies):**
- Token masking, token deletion, text infilling, sentence permutation, document rotation
- The model reconstructs the original text
- Loss: CE between decoder output and original text

**Key insight:** These encoder-decoder models use more aggressive corruption than BERT's 15% masking, forcing the model to learn richer representations.

---

## 8. Knowledge Distillation Loss

**Combined Loss:**

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \underbrace{D_{KL}\left(\sigma(z_s / \tau) \| \sigma(z_t / \tau)\right) \cdot \tau^2}_{\text{Soft Target Loss}} + (1 - \alpha) \cdot \underbrace{\text{CE}(y, \sigma(z_s))}_{\text{Hard Target Loss}}$$

where:
- $z_t$ = teacher logits, $z_s$ = student logits
- $\tau$ = temperature (typically 2–20; higher = softer distributions)
- $\sigma$ = softmax function
- $\alpha$ = blending weight (typically 0.5–0.9)
- The $\tau^2$ factor compensates for the reduced gradient magnitude at high temperatures

**Why soft targets help:**
- A teacher that outputs [cat: 0.7, dog: 0.2, car: 0.1] provides more information than a hard label [cat: 1, dog: 0, car: 0]
- The dark knowledge in soft targets reveals inter-class relationships
- Enables training smaller, faster models with minimal performance loss

**Used in:** DistilBERT, TinyBERT, model compression for edge deployment

**PyTorch sketch:**

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)

    return alpha * kl_loss + (1 - alpha) * ce_loss
```

---

## 9. Perplexity as a Metric

**Formula:**

$$\text{PPL} = e^{\text{CE}} = e^{-\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})}$$

equivalently:

$$\text{PPL} = \exp\left(-\frac{1}{T} \log P_\theta(x_1, x_2, \ldots, x_T)\right) = P_\theta(x_1, \ldots, x_T)^{-1/T}$$

**Interpretation:**
- Perplexity = the effective number of equally-likely tokens the model is choosing from at each step
- **Lower is better** — a PPL of 1 means perfect prediction; PPL of 50,000 means random guessing over a 50K vocabulary
- PPL = 20 means the model is "as confused as if choosing from 20 equally likely tokens"

**Typical values:**

| Model | Perplexity (approx.) | Dataset |
|---|---|---|
| GPT-2 (1.5B) | ~18 | WikiText-103 |
| GPT-3 (175B) | ~9 | LAMBADA |
| LLaMA 2 (70B) | ~3.5 | Internal benchmarks |

**Interview note:** Perplexity is the standard metric for evaluating language models during pre-training. It is the exponentiation of cross-entropy loss — directly comparable across models using the same tokenizer and test set.

---

## 10. Summary Comparison Table

| Loss | Domain | When Used | LLM Context |
|---|---|---|---|
| **MSE** | Regression | Continuous targets | Value heads in RLHF |
| **MAE** | Regression | Robust regression | — |
| **Huber** | Regression | Noisy regression | Value function in PPO |
| **Cross-Entropy** | Classification | Multi-class prediction | **Core LLM pre-training & SFT loss** |
| **BCE** | Classification | Binary / multi-label | Toxicity classifiers, guardrails |
| **Focal** | Classification | Class imbalance | Token-level imbalance tasks |
| **Hinge** | Classification | SVMs, margin-based | — |
| **InfoNCE** | Metric Learning | Contrastive learning | **Dense retrievers for RAG (DPR, E5)** |
| **Triplet** | Metric Learning | Explicit triplet mining | Passage retrieval, face ID |
| **Cosine Similarity** | Metric Learning | Embedding alignment | Sentence-BERT |
| **Margin Ranking** | Metric Learning | Pairwise ranking | **Re-rankers in RAG** |
| **NT-Xent** | Self-Supervised | Contrastive (SimCLR) | Vision-language pre-training |
| **GAN (minimax)** | Generative | Adversarial training | Text GANs (rare now) |
| **VAE (ELBO)** | Generative | Variational inference | Latent text models |
| **Diffusion** | Generative | Denoising generation | Image/video gen (Sora, SD) |
| **PPO** | Alignment | RLHF | **InstructGPT, ChatGPT** |
| **DPO** | Alignment | Direct preference | **LLaMA 2 Chat, Zephyr, modern alignment** |
| **KL Divergence** | Regularization | Distribution matching | **Distillation, VAE, RLHF penalty** |
| **MLM** | Self-Supervised | Masked prediction | **BERT pre-training** |
| **CLM (Next-Token)** | Self-Supervised | Autoregressive | **GPT pre-training** |
| **Distillation** | Compression | KL + CE combined | **DistilBERT, TinyLLaMA** |

---

## 11. Common Interview Questions & Answers

### Q1: "What loss function do LLMs use?"

> LLMs are trained using **cross-entropy loss** (next-token prediction). At each position in the sequence, the model predicts a probability distribution over the entire vocabulary, and the loss penalizes low probability on the correct next token. The total loss is averaged over all token positions.
>
> For alignment (making the model helpful and safe), additional losses are used: **PPO** (RLHF with a reward model) or **DPO** (direct preference optimization from human preference pairs).
>
> Perplexity ($e^{\text{CE}}$) is the standard evaluation metric derived from this loss.

---

### Q2: "Explain cross-entropy loss."

> Cross-entropy measures the difference between the predicted probability distribution and the true distribution. For classification with one-hot labels, it simplifies to $-\log(\hat{p}_k)$ — the negative log-probability assigned to the correct class.
>
> It has a clean gradient ($\hat{p} - y$), is the MLE objective for categorical distributions, and works seamlessly with softmax. This is why it's the universal loss for both classification and language modeling.
>
> In practice, PyTorch's `nn.CrossEntropyLoss` takes raw logits (not softmax outputs) for numerical stability, and supports features like `ignore_index` (for padding in LLMs) and `label_smoothing`.

---

### Q3: "When would you use triplet loss vs. contrastive loss?"

> **Triplet loss** operates on explicit (anchor, positive, negative) triplets and requires careful hard-negative mining to avoid trivial triplets that don't contribute to learning. It was dominant in early metric learning (FaceNet, 2015).
>
> **Contrastive loss (InfoNCE)** is now preferred because it uses all other samples in the batch as negatives — giving $B-1$ negatives "for free" with a batch size of $B$. This is much more sample-efficient. CLIP, SimCLR, and dense retrievers like DPR all use InfoNCE variants.
>
> I'd use **triplet loss** only when I have curated, meaningful triplets (e.g., specific hard negatives from a knowledge graph). For most embedding and retrieval tasks, especially in RAG, **InfoNCE/contrastive loss** is the standard choice.

---

### Q4: "What is KL divergence and where is it used?"

> KL divergence $D_{KL}(P \| Q)$ measures how much information is lost when approximating distribution $P$ with distribution $Q$. It is non-negative, asymmetric, and equals zero only when the two distributions are identical.
>
> In ML, it appears in three major places:
> 1. **Knowledge distillation:** The student model's softmax output is pushed to match the teacher's soft targets using KL divergence.
> 2. **VAE regularization:** The KL term in the ELBO forces the encoder's posterior $q(z|x)$ toward the standard normal prior $p(z)$.
> 3. **RLHF:** A KL penalty $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ prevents the aligned model from drifting too far from the supervised fine-tuned model, avoiding reward hacking.

---

### Q5: "Difference between DPO and PPO?"

> Both align LLMs with human preferences, but their approaches differ fundamentally:
>
> **PPO** follows the classic RLHF pipeline: train a separate reward model on preference data, then optimize the LLM as an RL agent to maximize that reward. It requires 4 models in memory (policy, reference, reward model, value head) and an RL training loop with clipped surrogate objectives.
>
> **DPO** reformulates the problem as supervised learning. It directly optimizes the policy on preference pairs (chosen vs. rejected responses) using a closed-form loss derived from the reward model's optimality condition. No separate reward model or RL loop is needed — just the policy and a frozen reference model.
>
> **In practice:** DPO is simpler, more stable, and cheaper to train. PPO can be more powerful when the reward model is high-quality and the RL training is well-tuned. Many production systems (LLaMA 2, Zephyr) use DPO; OpenAI's InstructGPT used PPO.

---

### Q6: "How does focal loss handle class imbalance?"

> Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to standard cross-entropy. When the model is confident about an easy example ($p_t$ close to 1), this factor approaches 0, dramatically reducing the loss contribution. For hard, misclassified examples ($p_t$ close to 0), the factor stays near 1, preserving the full loss.
>
> With $\gamma = 2$ (the typical value), an easy example with $p_t = 0.9$ has its loss reduced by 100x compared to standard CE. This forces the model to focus its learning capacity on the minority class — the hard examples where it struggles.
>
> It was introduced in RetinaNet for object detection, where 99.9% of anchor boxes are background (negatives). By down-weighting the easy negatives, focal loss allowed a one-stage detector to match two-stage detector accuracy for the first time.

---

## 12. Key Interview Takeaways

1. **Cross-entropy is the foundation of LLMs.** Pre-training, SFT, MLM, CLM — they all minimize CE over token distributions. Be able to write it from scratch and explain why it works.

2. **Know the alignment loss landscape.** PPO vs. DPO is the hottest topic in LLM interviews. Understand the full pipeline: Pretrain (CE) → SFT (CE) → Alignment (PPO or DPO).

3. **Contrastive losses power RAG.** InfoNCE trains the dense retrievers (DPR, E5, BGE) that make retrieval-augmented generation possible. Understand temperature, in-batch negatives, and hard negative mining.

4. **KL divergence is everywhere.** Distillation, VAEs, RLHF regularization, variational inference. Know the formula, the asymmetry, and the forward-vs-reverse distinction.

5. **Perplexity = $e^{\text{CE}}$.** This is how we evaluate language models. Lower is better. Be able to interpret a PPL score.

6. **Match the loss to the task:**
   - Regression → MSE / Huber
   - Classification → CE / Focal
   - Embedding → InfoNCE / Triplet
   - Generation → Adversarial / ELBO / Diffusion
   - Alignment → PPO / DPO
   - Compression → Distillation (KL + CE)

7. **Know the PyTorch API.** `nn.CrossEntropyLoss` takes logits (not softmax), `ignore_index=-100` is critical for padding, `BCEWithLogitsLoss` is preferred over `BCELoss`, and `label_smoothing` is a one-line upgrade.

8. **Connect losses to real systems you've built.** Tie your answers to specific projects — RAG pipelines, fine-tuned LLMs, classification models — to show practical depth beyond textbook knowledge.

---

*Last updated: February 2026 | Prepared for ML Engineer / Data Scientist interviews*
