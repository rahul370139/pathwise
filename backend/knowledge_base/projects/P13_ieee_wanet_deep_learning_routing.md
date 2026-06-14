# Publication: Deep Learning for Reliable Data Transmission in WANETs

**Paper:** "Reliable Data Transmission in Wireless Ad Hoc Networks using Advanced Deep Learning Techniques"
**Venue:** IEEE ICSCDS 2025 (3rd International Conference on Sustainable Computing and Data Communication Systems)
**Role:** Co-author — contributed to the **deep learning architecture design and implementation** (CNN-LSTM hybrid model)
**DOI:** 10.1109/ICSCDS65426.2025.11167851

---

# Table of Contents

1. [Paper Overview (The 2-Minute Pitch)](#1-paper-overview)
2. [Problem Statement](#2-problem-statement)
3. [Deep Learning Architecture — CNN-LSTM Hybrid](#3-deep-learning-architecture--cnn-lstm-hybrid)
4. [Why CNN + LSTM (Design Rationale)](#4-why-cnn--lstm-design-rationale)
5. [Training Details](#5-training-details)
6. [Results and Key Numbers](#6-results-and-key-numbers)
7. [Your Contribution — How to Frame It](#7-your-contribution--how-to-frame-it)
8. [Deep Learning Concepts You Must Know](#8-deep-learning-concepts-you-must-know)
9. [Interview Questions You'll Get Asked](#9-interview-questions-youll-get-asked)

---

# **1. Paper Overview**

## **The 2-Minute Pitch**

> "I co-authored an IEEE paper on applying deep learning to improve data transmission in wireless ad hoc networks — these are decentralized networks with no fixed infrastructure, where nodes are mobile (think disaster recovery, military comms, vehicular networks).
>
> The core challenge is that traditional routing protocols like AODV and DSR use static rules and can't adapt quickly to constantly changing network topology — nodes move, links break, interference fluctuates.
>
> My contribution was designing a **hybrid CNN-LSTM architecture** where the CNN extracts spatial patterns from the network state (which nodes are where, which links are strong) and the LSTM captures temporal patterns (how the network evolves over time). This lets the model predict optimal routing paths in real time.
>
> We tested on NS-3 simulations and showed the DL model outperformed AODV and DSR on all four metrics: **+42% throughput, +9% packet delivery, -15% latency, -23% energy consumption**. The paper was published at IEEE ICSCDS 2025."

---

# **2. Problem Statement**

## **What Are WANETs?**

Wireless Ad Hoc Networks (WANETs) are **decentralized, infrastructure-less** wireless networks where mobile nodes communicate directly with each other (peer-to-peer), forming a mesh.

```
┌──────────────────────────────────────────────────────────────────┐
│          WANET — DYNAMIC NETWORK                                  │
│                                                                   │
│  Time T=0:                      Time T=1 (nodes moved):         │
│                                                                   │
│    A ──── B                       A ──── B ──── E               │
│    │      │                              │                       │
│    │      │                              │                       │
│    C ──── D ──── E                C      D                       │
│                                                                   │
│  Nodes move → topology changes → links break/form               │
│  Traditional routing (AODV/DSR): uses fixed rules, slow to adapt│
│  Our DL model: PREDICTS future state, adapts routing preemptively│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **Why Traditional Protocols Fail**

| Protocol | How It Works | Limitation |
|----------|-------------|-----------|
| **AODV** (Ad hoc On-Demand Distance Vector) | Discovers routes on-demand, maintains routing table | Reactive — discovers routes AFTER they're needed; can't predict link failure |
| **DSR** (Dynamic Source Routing) | Source specifies full route in packet header | Route becomes stale as nodes move; overhead from carrying full route |

**Both protocols are rule-based and reactive.** They wait for a link to break, then scramble to find a new route — causing packet loss, delay spikes, and wasted energy on retransmissions.

## **Our Solution**

Replace the reactive decision-making with a **predictive deep learning model** that:
1. **Sees** the current network spatial layout (CNN)
2. **Remembers** how the network has evolved (LSTM)
3. **Predicts** the best route before links fail (proactive routing)

---

# **3. Deep Learning Architecture — CNN-LSTM Hybrid**

## **3.1 High-Level Architecture**

```
┌──────────────────────────────────────────────────────────────────┐
│              CNN-LSTM HYBRID ARCHITECTURE                          │
│                                                                   │
│  INPUT: Network state data                                       │
│  ├── Node positions (x, y coordinates)                           │
│  ├── Link quality metrics (signal strength, BER)                 │
│  ├── Traffic patterns (packets/sec per link)                     │
│  └── Historical route data (past routing decisions)              │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │              CNN BLOCK (Spatial Feature Extraction)│            │
│  │                                                   │            │
│  │  Conv1D layers extract spatial patterns:          │            │
│  │  • Node density clusters                          │            │
│  │  • Link quality neighborhoods                     │            │
│  │  • Congestion hotspots                            │            │
│  │  • Proximity and connectivity patterns            │            │
│  │                                                   │            │
│  │  Input: Network snapshot (spatial features)       │            │
│  │  Output: Compressed spatial feature vector        │            │
│  └──────────────────────┬────────────────────────────┘            │
│                         │                                         │
│                         ▼                                         │
│  ┌─────────────────────────────────────────────────┐             │
│  │           LSTM BLOCK (Temporal Prediction)        │            │
│  │                                                   │            │
│  │  LSTM layers process sequence of CNN outputs:     │            │
│  │  • Learn how network evolves over time            │            │
│  │  • Capture route change patterns                  │            │
│  │  • Predict link quality degradation               │            │
│  │  • Forecast congestion before it happens          │            │
│  │                                                   │            │
│  │  Input: Sequence of spatial features [t-k, ..., t]│            │
│  │  Output: Predicted next network state             │            │
│  └──────────────────────┬────────────────────────────┘            │
│                         │                                         │
│                         ▼                                         │
│  ┌─────────────────────────────────────────────────┐             │
│  │            OUTPUT LAYER (Routing Decision)        │            │
│  │                                                   │            │
│  │  Dense layers produce:                            │            │
│  │  • Optimal next-hop for each node pair            │            │
│  │  • Predicted packet delivery probability          │            │
│  │  • Estimated delay per route                      │            │
│  │  • Energy cost estimate                           │            │
│  └─────────────────────────────────────────────────┘             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **3.2 CNN Block — What It Does**

The CNN processes each **network snapshot** as a structured data input:

| Input Feature | What It Captures | Why CNN Helps |
|--------------|-----------------|---------------|
| Node positions | Where nodes are in the network area | CNN detects **spatial clusters** (dense vs sparse regions) |
| Link quality | Signal strength between connected nodes | CNN identifies **quality neighborhoods** (areas of good/bad connectivity) |
| Traffic load | Data volume on each link | CNN finds **congestion patterns** (hotspots) |
| Connection topology | Which nodes can communicate | CNN learns **structural features** (bottlenecks, bridges) |

**Why CNN (not just MLP)?** The network state has **spatial structure** — nearby nodes affect each other. Convolutions capture local spatial patterns (like neighborhoods of congestion) that a fully-connected layer would miss or need far more parameters to learn.

## **3.3 LSTM Block — What It Does**

The LSTM processes a **sequence of CNN feature vectors** over time:

```
Time:    t-4     t-3     t-2     t-1      t      → t+1 (predicted)
          │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼
        [CNN]   [CNN]   [CNN]   [CNN]   [CNN]
          │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼
        ┌────────────────────────────────────┐
        │            LSTM Sequence           │ → Predicted network state
        └────────────────────────────────────┘
```

| Temporal Pattern | What LSTM Learns | Routing Impact |
|-----------------|-----------------|----------------|
| Node mobility trends | "Node C is moving away from Node D" | Pre-route traffic before link breaks |
| Link degradation | "Link A-B quality has been declining for 3 timesteps" | Switch to alternate route proactively |
| Traffic bursts | "Traffic on this path spikes every 5 minutes" | Load-balance before congestion hits |
| Route stability | "This route has been stable for 10 timesteps" | Prefer stable routes, avoid volatile ones |

**Why LSTM (not vanilla RNN)?** WANETs have both **short-term** dynamics (sudden link failure) and **long-term** patterns (mobility trends, periodic traffic). LSTM's gating mechanism (forget gate, input gate, output gate) handles this much better than vanilla RNNs, which suffer from vanishing gradients on longer sequences.

## **3.4 Combined CNN-LSTM Flow**

1. At each timestep, **CNN** processes the current network snapshot → spatial feature vector
2. **LSTM** takes the sequence of recent spatial features (sliding window) → predicts future network state
3. **Output layer** uses the prediction to select optimal routing: which path will have the best delivery ratio, lowest delay, and least energy cost
4. Routing decision is applied to the network; next timestep → repeat

---

# **4. Why CNN + LSTM (Design Rationale)**

## **4.1 The Two Dimensions of Network Data**

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  SPATIAL (within one snapshot)    TEMPORAL (across snapshots) │
│  ─────────────────────────────    ──────────────────────────  │
│  "Where are the nodes?"          "How is the network         │
│  "Which links are strong?"        changing over time?"        │
│  "Where is congestion?"          "Will this link break soon?" │
│                                                               │
│  CNN excels here ─────────────── LSTM excels here            │
│                                                               │
│  Together: "Given where everything is NOW and how it's been  │
│  CHANGING, what's the BEST route for the NEXT moment?"       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## **4.2 Why Not Alternatives?**

| Alternative | Why We Didn't Use It |
|------------|---------------------|
| **CNN only** | No temporal awareness — can't predict future link failures |
| **LSTM only** | Treats all features equally — doesn't capture spatial neighborhood structure |
| **MLP only** | No spatial or temporal inductive bias — needs far more data to learn patterns CNN/LSTM get for free |
| **Transformer** | Computationally expensive for resource-constrained ad hoc nodes; LSTM is more efficient for moderate sequence lengths |
| **GRU instead of LSTM** | Would also work (fewer parameters); we chose LSTM for stronger long-range dependency handling |
| **GNN (Graph Neural Network)** | Elegant for network topology but adds complexity; CNN on structured features was simpler and sufficient |

---

# **5. Training Details**

## **5.1 Data Generation**

| Aspect | Detail |
|--------|--------|
| **Simulator** | NS-3 (Network Simulator 3) — industry-standard for network research |
| **Mobility models** | Random Waypoint (random movement) + Manhattan Grid (street-pattern movement) |
| **Network scenarios** | Low/medium/high node density + sparse topology |
| **Features collected** | Node positions, link quality (SNR/BER), traffic load, routing decisions, packet delivery outcomes |
| **Split** | Training set (historical simulations) / Test set (real-time simulation data) |

## **5.2 Loss Functions**

| Task | Loss Function | Why |
|------|-------------|-----|
| **Regression** (predicting throughput, delay, packet loss) | **MSE (Mean Squared Error)** | Standard for continuous predictions; penalizes large errors heavily |
| **Classification** (predicting best route/next hop) | **Categorical Cross-Entropy** | Standard for multi-class classification (choosing among route options) |

**MSE formula from the paper:**

$$ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$

Where y_i is actual network metric value and ŷ_i is the CNN-LSTM predicted value.

## **5.3 Framework**

| Component | Choice |
|-----------|--------|
| **Framework** | TensorFlow + Keras |
| **Acceleration** | GPU training |
| **Evaluation** | Comparison against AODV and DSR baselines on NS-3 |

---

# **6. Results and Key Numbers**

**Memorize these — interviewers will ask for specific numbers.**

## **6.1 Throughput (Mbps)**

| Scenario | CNN-LSTM | AODV | DSR | Improvement vs AODV |
|----------|---------|------|-----|-------------------|
| Low Density | **5.8** | 4.2 | 4.5 | +38% |
| Medium Density | **7.1** | 5.0 | 5.3 | **+42%** |
| High Density | **6.5** | 4.7 | 5.1 | +38% |
| Sparse | **5.9** | 3.8 | 4.2 | +55% |

## **6.2 Packet Delivery Ratio (%)**

| Scenario | CNN-LSTM | AODV | DSR |
|----------|---------|------|-----|
| Low Density | **95.2** | 87.1 | 89.4 |
| Medium Density | **93.8** | 85.3 | 88.1 |
| High Density | **91.7** | 84.5 | 86.3 |
| Sparse | **94.1** | 80.4 | 82.7 |

**Key number:** 95.2% PDR vs 87.1% (AODV) = **~9% absolute improvement** at low density.

## **6.3 End-to-End Delay (ms)**

| Scenario | CNN-LSTM | AODV | DSR |
|----------|---------|------|-----|
| Low Density | **125** | 145 | 138 |
| Medium Density | **110** | 130 | 125 |
| High Density | **115** | 135 | 130 |
| Sparse | **120** | 150 | 145 |

**Key number:** 110ms vs 130ms (AODV) = **15% lower latency** at medium density.

## **6.4 Energy Consumption (Joules/node)**

| Scenario | CNN-LSTM | AODV | DSR |
|----------|---------|------|-----|
| Low Density | **2.3** | 3.0 | 2.8 |
| Medium Density | **2.5** | 3.3 | 3.1 |
| High Density | **2.8** | 3.5 | 3.3 |
| Sparse | **2.2** | 3.1 | 2.9 |

**Key number:** 2.3J vs 3.0J (AODV) = **23% less energy** at low density. Critical for battery-powered devices.

## **6.5 Summary: Why DL Won**

| Why DL Model Outperformed | Mechanism |
|--------------------------|-----------|
| **Higher throughput** | CNN identifies congestion-free paths; LSTM predicts future congestion → routes around it |
| **Better packet delivery** | Predictive routing avoids paths about to fail → fewer dropped packets |
| **Lower delay** | Fewer retransmissions (because fewer failures) + pre-computed optimal paths |
| **Less energy** | Fewer failed transmissions = fewer retries = less power wasted |

---

# **7. Your Contribution — How to Frame It**

When an interviewer asks "What was your contribution?":

> "I contributed to the **deep learning architecture design and implementation**. Specifically:
>
> 1. **Designed the hybrid CNN-LSTM pipeline** — the CNN processes each network snapshot to extract spatial features (node density patterns, link quality neighborhoods, congestion hotspots), and the LSTM processes the sequence of these spatial features to predict future network states.
>
> 2. **Implemented the model** in TensorFlow/Keras with GPU-accelerated training. Handled the data preprocessing pipeline from NS-3 simulator output to model-ready tensors.
>
> 3. **Selected and tuned the loss functions** — MSE for regression tasks (predicting throughput and delay) and categorical cross-entropy for classification (selecting optimal routes).
>
> 4. **Ran the comparative evaluation** against AODV and DSR baselines, producing the performance comparisons across throughput, PDR, delay, and energy metrics.
>
> The networking domain (WANET protocols, NS-3 simulation setup) was handled by the other co-authors who have electrical/networking backgrounds."

---

# **8. Deep Learning Concepts You Must Know**

Since you contributed the DL part, interviewers will probe your understanding of every DL choice:

## **8.1 CNN — Why and How**

**Why CNN for non-image data?**
CNNs aren't just for images. They excel at detecting **local patterns** in structured data. In our case, the network state is organized spatially — nodes have positions, links have local quality metrics. 1D convolutions slide over these features to detect patterns like "cluster of high-quality links" or "congested neighborhood."

**Convolution operation:** Filter slides across input, computes dot product at each position → produces feature map that highlights where the pattern occurs.

**Key hyperparameters:** Kernel size (receptive field width), number of filters (how many patterns to detect), stride, padding.

## **8.2 LSTM — Why and How**

**The vanishing gradient problem:** Vanilla RNNs multiply gradients through many timesteps during backpropagation → gradients shrink exponentially → model can't learn long-range dependencies.

**LSTM solution:** Three gates + cell state:

| Gate | What It Does | In Our Context |
|------|-------------|----------------|
| **Forget gate** | Decides what to discard from cell state | "This link quality pattern from 10 steps ago is no longer relevant (node moved away)" |
| **Input gate** | Decides what new info to store | "A new congestion pattern just emerged — remember this" |
| **Output gate** | Decides what to output from cell state | "Based on remembered patterns, output the predicted next state" |
| **Cell state** | Long-term memory highway | Carries information across many timesteps without degradation |

**Why not GRU?** GRU combines forget and input gates (simpler, faster). LSTM has separate gates, giving it more control over what to remember/forget. For network state prediction where both short-term (sudden link failure) and long-term (mobility trends) patterns matter, LSTM's extra expressiveness helps.

## **8.3 The Reliability Equation from the Paper**

The paper defines transmission reliability R as:

```
R = PDR × (1 - BER)
```

Where PDR = Packet Delivery Ratio and BER = Bit Error Rate. Higher R = more reliable. The DL model optimizes routing to maximize R.

---

# **9. Interview Questions You'll Get Asked**

---

## **Q1: "Tell me about your IEEE publication."**

> (Use the 2-minute pitch from Section 1 — practice this until it's smooth.)

---

## **Q2: "Why did you choose CNN-LSTM instead of [other architecture]?"**

> The network data has two distinct dimensions that need different inductive biases:
>
> **Spatial:** At any given moment, the network has a spatial structure — node positions, link quality neighborhoods, congestion clusters. CNNs are designed to detect local spatial patterns via convolutions. A fully-connected layer would treat all features independently, missing these neighborhood effects.
>
> **Temporal:** The network changes over time — nodes move, links degrade, traffic patterns shift. LSTM handles sequential dependencies with its gating mechanism, learning both short-term dynamics (sudden link failure) and long-term trends (node mobility patterns).
>
> We considered alternatives: a **Transformer** would work for temporal modeling but is computationally expensive — ad hoc networks have resource-constrained nodes. **GNNs** are elegant for graph-structured data but add complexity; our CNN approach on structured features was simpler and achieved strong results. **GRU** instead of LSTM would also work with fewer parameters; we chose LSTM for its stronger long-range dependency handling, which matters for predicting slow mobility trends.

---

## **Q3: "Why MSE as the loss function? What about other options?"**

> MSE was chosen for the regression tasks (predicting continuous metrics like throughput and delay) because it heavily penalizes large prediction errors — in networking, a route prediction that's way off is much worse than one that's slightly off (a bad route means dropped packets, wasted energy). MSE's squared penalty aligns with this asymmetric cost.
>
> For the route selection (classification) component, we used categorical cross-entropy, which is the standard loss for multi-class classification — choosing the optimal next-hop from a set of candidate nodes.
>
> Alternatives we could have considered: MAE (less sensitive to outliers, but we actually want outlier sensitivity here), Huber loss (compromise between MSE and MAE), or a custom loss incorporating both PDR and delay as a weighted objective.

---

## **Q4: "How did you prevent overfitting, given that network simulations can generate limited diversity?"**

> Multiple strategies:
> 1. **Diverse simulation scenarios** — We tested across four network configurations (low/medium/high density + sparse topology) with two mobility models (Random Waypoint and Manhattan Grid), giving the model exposure to varied network conditions.
> 2. **Train/test split** — Historical simulation data for training, separate real-time simulation data for testing — no data leakage.
> 3. **Regularization** — Standard techniques (dropout between CNN and LSTM blocks, early stopping based on validation loss).
> 4. **Model complexity** — The hybrid architecture is relatively lightweight (CNN for spatial compression, LSTM for temporal prediction) rather than an unnecessarily deep network.

---

## **Q5: "What are the limitations of this approach?"**

> I'd highlight three:
>
> 1. **Computational cost on edge devices:** The CNN-LSTM model needs meaningful compute for inference. In real WANETs, nodes are often battery-powered embedded devices. Deploying this model would require either model compression (quantization, pruning) or offloading inference to a more capable node.
>
> 2. **Simulation-to-real gap:** We validated on NS-3 simulations, which approximate real network behavior but don't capture every real-world effect (physical obstacles, electromagnetic interference, hardware variance). Deploying in real WANETs would require domain adaptation or fine-tuning on real data.
>
> 3. **Scalability:** As the number of nodes grows very large (100s-1000s), the spatial feature space grows, and the model may need architectural changes to scale — perhaps a GNN approach that naturally handles variable-size graphs, or hierarchical routing where clusters have their own models.

---

## **Q6: "How does your model handle a node that suddenly fails (hard failure)?"**

> The model handles this through its **temporal prediction capability**. The LSTM learns patterns that precede node failures — degrading link quality, increasing error rates, reduced responsiveness. When these precursors appear, the model proactively routes traffic away from that node before it fails.
>
> For truly sudden failures (no precursor), the model detects the failure at the next timestep (dropped packets, missing node in the snapshot), and the CNN immediately identifies the changed topology. The LSTM then incorporates this new information and adjusts routing in the next prediction cycle.
>
> The key advantage over AODV/DSR: traditional protocols only discover the failure after a timeout (seconds), then initiate a new route discovery process (more seconds). Our model reacts within one prediction cycle (milliseconds), significantly reducing the failure impact.

---

## **Q7: "If you were to extend this work, what would you do?"**

> Three directions:
>
> 1. **Graph Neural Networks:** Replace the CNN with a GNN that directly operates on the network graph structure. This would naturally handle variable-size networks and capture topology structure more faithfully than converting to structured features.
>
> 2. **Attention mechanisms:** Replace or augment the LSTM with a Transformer-style attention over the temporal sequence. This could help the model weigh recent critical events (like a sudden topology change) more than distant routine observations, without the sequential processing bottleneck of LSTM.
>
> 3. **Reinforcement learning:** Frame routing as an RL problem where the agent (each node) learns a routing policy through interaction with the network environment. The CNN-LSTM could serve as the policy network. This would enable truly adaptive behavior without needing labeled "optimal route" training data.
>
> 4. **Model compression:** Apply quantization and knowledge distillation to make the model deployable on resource-constrained edge devices, which is the real deployment target for WANETs.

---

## **Q8: "What's the practical impact — where would this actually be deployed?"**

> The paper targets three application domains:
>
> 1. **Disaster recovery:** When cell towers are destroyed (earthquake, hurricane), first responders set up ad hoc networks. Reliable communication with low latency saves lives. Our model's 95% PDR vs 87% (AODV) means fewer lost messages.
>
> 2. **Military operations:** Tactical networks in the field where nodes (soldiers, vehicles) are constantly moving. Energy efficiency (23% less) extends mission duration for battery-powered devices.
>
> 3. **Vehicular networks (VANETs):** Cars communicating for collision avoidance, traffic optimization. Low latency (110ms vs 130ms) is critical for real-time safety applications.

---

# **Quick Reference Card**

```
┌──────────────────────────────────────────────────────────────────┐
│  PAPER QUICK REFERENCE                                           │
│                                                                   │
│  What: CNN-LSTM for reliable routing in ad hoc networks          │
│  Where: IEEE ICSCDS 2025                                         │
│  Your role: DL architecture design and implementation            │
│                                                                   │
│  Architecture: CNN (spatial) → LSTM (temporal) → Dense (routing) │
│  Loss: MSE (regression) + Cross-Entropy (classification)         │
│  Framework: TensorFlow/Keras, NS-3 simulator                    │
│  Baselines: AODV, DSR                                           │
│                                                                   │
│  KEY NUMBERS (memorize):                                         │
│  • Throughput: 7.1 Mbps vs 5.0 (AODV) — +42%                  │
│  • PDR: 95.2% vs 87.1% (AODV) — +9% absolute                  │
│  • Delay: 110ms vs 130ms (AODV) — 15% lower                    │
│  • Energy: 2.3J vs 3.0J (AODV) — 23% less                     │
│                                                                   │
│  WHY IT WORKS: Predictive routing (prevent failures before       │
│  they happen) vs reactive routing (fix after failure)            │
└──────────────────────────────────────────────────────────────────┘
```
