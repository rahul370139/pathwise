# Agentic Reasoning Patterns — Deep Dive Interview Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** AI/ML Engineer — Agentic Systems, LLM Reasoning, Multi-Agent Orchestration

> **Context:** Your file `10_agentic_ai_and_multi_agent.md` covers ReAct, memory, tool use, and agent frameworks well. This guide goes **deeper into reasoning patterns** that interviewers at both big tech and startups increasingly ask about.

---

# Table of Contents

1. [Pattern Landscape Overview](#1-pattern-landscape-overview)
2. [Chain-of-Thought (CoT)](#2-chain-of-thought-cot)
3. [Tree-of-Thoughts (ToT)](#3-tree-of-thoughts-tot)
4. [ReAct — Extended Analysis](#4-react--extended-analysis)
5. [ReWOO (Reasoning Without Observation)](#5-rewoo-reasoning-without-observation)
6. [Plan-and-Execute](#6-plan-and-execute)
7. [Reflexion and Self-Reflection](#7-reflexion-and-self-reflection)
8. [LATS (Language Agent Tree Search)](#8-lats-language-agent-tree-search)
9. [Multi-Agent Collaboration Patterns](#9-multi-agent-collaboration-patterns)
10. [Pattern Selection Guide](#10-pattern-selection-guide)
11. [Interview Questions with Strong Answers](#11-interview-questions-with-strong-answers)

---

# **1. Pattern Landscape Overview**

These patterns sit on a spectrum from simple prompting to full autonomous agents:

```
PROMPTING PATTERNS              AGENTIC PATTERNS              MULTI-AGENT PATTERNS
─────────────────────           ──────────────────            ────────────────────
Chain-of-Thought (CoT)          ReAct                         Debate / Adversarial
Self-Consistency                ReWOO                         Voting / Ensemble
Tree-of-Thoughts (ToT)         Plan-and-Execute              Supervisor-Worker
                                Reflexion                     Handoff / Relay
                                LATS                          Mixture-of-Agents
```

**Key distinction:** Prompting patterns enhance a *single LLM call* (or a few). Agentic patterns involve *loops with tool use*. Multi-agent patterns coordinate *multiple agents*.

```
┌───────────────────────────────────────────────────────────────────┐
│              REASONING PATTERN EVOLUTION                          │
│                                                                   │
│  Standard       "Answer this question"                           │
│  Prompting      → Single pass, no structure                      │
│       │                                                          │
│       ▼                                                          │
│  Chain-of-       "Think step by step"                            │
│  Thought         → Linear reasoning chain                        │
│       │                                                          │
│       ├──► Self-Consistency: Sample N chains, majority vote      │
│       │                                                          │
│       ▼                                                          │
│  Tree-of-        Explore multiple reasoning branches             │
│  Thoughts        → BFS/DFS over thought tree, evaluate & prune  │
│       │                                                          │
│       ▼                                                          │
│  ReAct           Interleave Thought → Action → Observation       │
│       │          → Grounded reasoning with tool use              │
│       │                                                          │
│       ├──► ReWOO: Plan ALL steps first, then execute batch       │
│       │                                                          │
│       ├──► Plan-and-Execute: Planner + Executor separation       │
│       │                                                          │
│       ▼                                                          │
│  Reflexion       Execute → Evaluate → Reflect → Retry            │
│       │          → Learn from failures within a session          │
│       │                                                          │
│       ▼                                                          │
│  LATS            Tree search over action sequences               │
│       │          → Monte Carlo exploration of agent trajectories │
│       │                                                          │
│       ▼                                                          │
│  Multi-Agent     Multiple specialized agents collaborate         │
│  Systems         → Debate, vote, hand off, supervise             │
└───────────────────────────────────────────────────────────────────┘
```

---

# **2. Chain-of-Thought (CoT)**

**Paper:** *Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*

## **2.1 Core Idea**

Instead of asking an LLM to jump directly to an answer, you prompt it to produce **intermediate reasoning steps**. This dramatically improves performance on math, logic, and multi-hop reasoning tasks.

```
WITHOUT CoT                          WITH CoT
──────────                           ────────
Q: Roger has 5 tennis balls.         Q: Roger has 5 tennis balls.
He buys 2 more cans of 3.           He buys 2 more cans of 3.
How many does he have?               How many does he have?

A: 11                                A: Roger starts with 5 balls.
   (wrong)                              2 cans × 3 balls = 6 new balls.
                                        5 + 6 = 11 balls.
                                        The answer is 11. ✓
```

## **2.2 Three Variants**

### **Zero-Shot CoT**

Simply append **"Let's think step by step"** to the prompt. No examples needed.

```
Prompt: "[Question]. Let's think step by step."
```

**Why it works:** This phrase activates the model's latent reasoning capabilities. The model has seen many step-by-step solutions in training data, and this phrase triggers that distribution.

**When to use:** Quick prototyping, when you don't have good few-shot examples, or when the task domain is broad.

### **Few-Shot CoT**

Provide 2-5 examples of (question, reasoning chain, answer) before the actual question.

```
Example 1:
Q: A store had 22 apples. They sold 15 and got 8 more. How many?
A: Start: 22. Sold 15 → 22-15=7. Got 8 → 7+8=15. Answer: 15.

Example 2:
Q: [another example with reasoning chain]

Now answer:
Q: [actual question]
A:
```

**When to use:** When you need consistently formatted reasoning for a specific domain (finance, medical, legal). Few-shot examples steer *how* the model reasons, not just *that* it reasons.

### **Self-Consistency (CoT-SC)**

**Paper:** *Wang et al. (2023), "Self-Consistency Improves Chain of Thought Reasoning"*

Sample **N different reasoning chains** (via temperature > 0) and take the **majority vote** on the final answer.

```
┌──────────────────────────────────────────────────────┐
│               SELF-CONSISTENCY                        │
│                                                       │
│  Question ──┬──► Chain 1 → Answer: 42                │
│             ├──► Chain 2 → Answer: 42                │
│             ├──► Chain 3 → Answer: 38                │
│             ├──► Chain 4 → Answer: 42                │
│             └──► Chain 5 → Answer: 40                │
│                                                       │
│  Majority Vote: 42 (3/5)  ──► Final Answer: 42      │
└──────────────────────────────────────────────────────┘
```

**Key insight:** Different reasoning paths may lead to different answers. The correct answer tends to appear more frequently because there are generally more valid reasoning paths leading to it.

**Trade-off:** N× cost and latency for better accuracy. Typically N=5-10.

## **2.3 When CoT Fails**

- Tasks requiring **factual recall** (CoT can't help if the model doesn't know the fact)
- Very simple tasks (CoT adds unnecessary verbosity)
- Tasks where the model **rationalizes wrong answers** (confident but incorrect chains)
- When **grounding in real data** is needed → use ReAct instead

## **2.4 Interview Framing**

> "CoT is the foundation of all agentic reasoning. ReAct extends CoT by interleaving actions. ToT extends CoT by exploring multiple branches. Understanding CoT is understanding why agents can reason at all."

---

# **3. Tree-of-Thoughts (ToT)**

**Paper:** *Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"*

## **3.1 Core Idea**

While CoT follows a **single linear chain**, ToT explores **multiple reasoning branches** simultaneously, evaluates them, and prunes bad paths — like a human considering multiple approaches before committing.

```
                          CoT vs ToT

CoT (Linear):          ToT (Branching):
                                
Input                  Input
  │                      │
  ▼                      ▼
Step 1               ┌──Step 1a──┐
  │                  │           │
  ▼                  ▼           ▼
Step 2            Step 2a     Step 2b  ← evaluate, prune 2b
  │                  │
  ▼                  ▼
Step 3            Step 3a
  │                  │
  ▼                  ▼
Answer             Answer
```

## **3.2 How It Works**

```
┌──────────────────────────────────────────────────────────────────┐
│                    TREE-OF-THOUGHTS ALGORITHM                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. DECOMPOSE: Break problem into sequential "thought steps"     │
│                                                                   │
│  2. GENERATE: At each step, propose K candidate thoughts         │
│     (using the LLM with sampling or separate propose-prompt)     │
│                                                                   │
│  3. EVALUATE: Score each candidate thought                       │
│     • Self-evaluation: LLM rates "sure/maybe/impossible"         │
│     • Or use a separate evaluator / value function               │
│                                                                   │
│  4. SEARCH: Use BFS or DFS to explore the tree                   │
│     • BFS: Explore all branches at depth d before going deeper   │
│     • DFS: Go deep on promising branch, backtrack if stuck       │
│                                                                   │
│  5. PRUNE: Drop branches rated "impossible" or below threshold   │
│                                                                   │
│  6. SELECT: Return the path from root to best leaf node          │
└──────────────────────────────────────────────────────────────────┘
```

## **3.3 Concrete Example: "Game of 24"**

Task: Use numbers [4, 5, 6, 10] and +, -, ×, ÷ to make 24.

```
Root: [4, 5, 6, 10]
├── Thought 1a: 10 - 4 = 6     → [5, 6, 6]     eval: "maybe"
│   ├── Thought 2a: 5 + 6 = 11 → [6, 11]        eval: "impossible" (can't make 24) ✗
│   └── Thought 2b: 6 - 5 = 1  → [6, 1]         eval: "impossible" ✗
├── Thought 1b: 5 × 4 = 20     → [6, 10, 20]    eval: "maybe"
│   ├── Thought 2c: 10 - 6 = 4 → [4, 20]        eval: "sure" (4 + 20 = 24) ✓
│   └── ...
└── Thought 1c: 10 + 6 = 16    → [4, 5, 16]     eval: "maybe"
    └── ...
```

## **3.4 BFS vs DFS in ToT**

| Strategy | How It Works | Best For | Risk |
|----------|-------------|----------|------|
| **BFS** | Explore all K candidates at depth d, keep top-B, go to d+1 | Problems with clear good/bad signals early | High memory (keeps B×K states) |
| **DFS** | Go deep on best candidate, backtrack if dead end | Problems requiring deep exploration | May get stuck in bad subtree |

## **3.5 When to Use ToT vs CoT**

| Scenario | Use CoT | Use ToT |
|----------|---------|---------|
| Straightforward reasoning | ✓ | Overkill |
| Multiple valid approaches | Limited | ✓ |
| Need to explore and backtrack | Can't | ✓ |
| Creative problem solving | Limited | ✓ |
| Cost-sensitive | ✓ (1 call) | ✗ (many calls) |
| Puzzles, planning, code generation | Okay | Better |

## **3.6 Simplified ToT Implementation Sketch**

```python
def tree_of_thoughts(problem, max_depth=3, branching=3, beam_width=2):
    current_states = [{"thoughts": [], "state": problem}]

    for depth in range(max_depth):
        candidates = []
        for state in current_states:
            # Generate K candidate next-thoughts
            new_thoughts = llm.generate(
                f"Given: {state['state']}\nPropose {branching} different next steps.",
                n=branching
            )
            for thought in new_thoughts:
                # Evaluate each candidate
                score = llm.evaluate(
                    f"Rate this reasoning step (sure/maybe/impossible): {thought}"
                )
                if score != "impossible":
                    candidates.append({
                        "thoughts": state["thoughts"] + [thought],
                        "state": apply_thought(state["state"], thought),
                        "score": score
                    })
        # BFS beam search: keep top-B candidates
        current_states = sorted(candidates, key=rank)[:beam_width]

    return current_states[0]  # Best path
```

> **Interview tip:** ToT is asked about at research-oriented companies (Google DeepMind, OpenAI, Anthropic) and AI startups. The key differentiator from CoT: **exploration and evaluation**. CoT is greedy (one path); ToT is search (many paths with pruning).

---

# **4. ReAct — Extended Analysis**

> Your file 10 covers ReAct well. Here we add **comparison context** and **failure modes**.

## **4.1 ReAct vs Act-Only vs Reason-Only**

**Paper:** *Yao et al. (2023), "ReAct: Synergizing Reasoning and Acting in Language Models"*

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  REASON-ONLY (CoT):    Think → Think → Think → Answer         │
│  Problem: Hallucination, no grounding in real data             │
│                                                                │
│  ACT-ONLY:             Action → Action → Action → Answer      │
│  Problem: Wrong tool selection, no strategic planning          │
│                                                                │
│  ReAct:                Think → Act → Observe → Think → ...    │
│  Best of both: Reasoning guides action, observations ground   │
│  reasoning                                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## **4.2 ReAct Failure Modes**

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Repetitive actions** | Agent calls same tool with same args in a loop | Add loop detection, max iteration limit |
| **Reasoning derailment** | Long traces lose coherence | Summarize scratchpad periodically |
| **Hallucinated tool calls** | Agent invents tools that don't exist | Strict tool schemas, validation layer |
| **Premature termination** | Agent says "Final Answer" too early | Require minimum evidence before finishing |
| **Token budget overflow** | Long traces exceed context window | Sliding window over observations, summarization |

## **4.3 ReAct Efficiency Problem**

ReAct interleaves reasoning and action in **every step**. This means:
- Each step requires an LLM call *and* a tool call (sequential)
- N steps = N LLM calls + N tool calls
- Total latency is high for multi-step tasks

This motivates **ReWOO** (next section) — plan all reasoning first, then execute all actions.

---

# **5. ReWOO (Reasoning Without Observation)**

**Paper:** *Xu et al. (2023), "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models"*

## **5.1 Core Idea**

ReWOO **separates planning from execution**. Instead of the ReAct loop (think → act → observe → think → act...), ReWOO does:

1. **Plan phase:** Generate the ENTIRE plan with placeholders for tool results
2. **Execute phase:** Run all tool calls (potentially in parallel)
3. **Solve phase:** Combine all observations with the plan to produce the final answer

```
┌────────────────────────────────────────────────────────────────┐
│                     ReWOO vs ReAct                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ReAct (Interleaved):                                         │
│  Think₁ → Act₁ → Obs₁ → Think₂ → Act₂ → Obs₂ → Answer      │
│  [LLM]   [Tool]  [wait]  [LLM]   [Tool]  [wait]  [LLM]      │
│  Total: 3 LLM calls + 2 tool calls (sequential)              │
│                                                                │
│  ReWOO (Separated):                                           │
│  ┌─────── PLAN (1 LLM call) ───────┐                         │
│  │ Step1: Search(X) → #E1           │                         │
│  │ Step2: Search(Y) → #E2           │                         │
│  │ Step3: Compare #E1 and #E2       │                         │
│  └──────────────────────────────────┘                         │
│            │                                                   │
│            ▼                                                   │
│  ┌─────── EXECUTE (parallel) ───────┐                         │
│  │ #E1 = Search(X) → "result 1"     │                         │
│  │ #E2 = Search(Y) → "result 2"     │  ← can run in parallel │
│  └──────────────────────────────────┘                         │
│            │                                                   │
│            ▼                                                   │
│  ┌─────── SOLVE (1 LLM call) ───────┐                         │
│  │ Given plan + observations,        │                         │
│  │ produce final answer              │                         │
│  └──────────────────────────────────┘                         │
│  Total: 2 LLM calls + 2 tool calls (parallel possible)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## **5.2 Plan Format**

```
Plan: To compare GDP of France and Germany, I need to:
#P1: Search for France's GDP in 2025
#E1 = Search("France GDP 2025")
#P2: Search for Germany's GDP in 2025
#E2 = Search("Germany GDP 2025")
#P3: Compare #E1 and #E2 to determine which is larger and by how much
```

`#E1`, `#E2` are **evidence placeholders** that get filled during execution.

## **5.3 Advantages and Limitations**

| Aspect | ReAct | ReWOO |
|--------|-------|-------|
| **LLM calls** | N (one per step) | 2 (plan + solve) |
| **Token usage** | High (repeated context) | ~5× less (no observation in planning prompt) |
| **Parallelism** | None (sequential by design) | Tool calls can run in parallel |
| **Adaptability** | High (can change plan mid-execution) | Low (plan is fixed) |
| **Error recovery** | Can adapt after bad observation | Must re-plan from scratch |
| **Best for** | Exploratory tasks, unknown # of steps | Well-structured tasks, predictable tools |
| **Latency** | High (sequential LLM + tool) | Lower (fewer LLM calls, parallel tools) |

## **5.4 When to Choose ReWOO**

- Tasks where the **plan is predictable** (e.g., "compare X and Y" always needs 2 searches)
- **Cost-sensitive** applications (fewer LLM calls = less cost)
- **Latency-sensitive** applications (parallel tool execution)
- Tasks where tool results are **independent** of each other

## **5.5 When NOT to Choose ReWOO**

- Tasks where step N+1 **depends on the result of step N** (e.g., "search for X, then based on what you find, search for Y")
- Exploratory tasks where the number of steps is unknown
- Tasks requiring **dynamic replanning**

> **Interview tip:** ReWOO is a less commonly known pattern. Mentioning it shows depth. The key insight: "ReAct couples reasoning with observations, which is powerful but expensive. ReWOO decouples them — plan once, execute in batch, solve once — reducing token usage by 5× with comparable accuracy on structured tasks."

---

# **6. Plan-and-Execute**

**Originated from:** *Wang et al. (2023), "Plan-and-Solve Prompting"* + LangGraph implementation

## **6.1 Core Idea**

Separate the system into two distinct agents:
1. **Planner** — Creates a high-level plan (list of steps)
2. **Executor** — Executes each step using tools, returns results
3. **Replanner** (optional) — Updates the plan based on execution results

```
┌──────────────────────────────────────────────────────────────────┐
│                   PLAN-AND-EXECUTE PATTERN                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Query: "Research competitor pricing and draft a report"    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────┐                                 │
│  │         PLANNER             │                                 │
│  │  (Strategic LLM, e.g. GPT-4)│                                │
│  │                             │                                 │
│  │  Plan:                      │                                 │
│  │  1. Search for competitor A │                                 │
│  │  2. Search for competitor B │                                 │
│  │  3. Extract pricing data    │                                 │
│  │  4. Create comparison table │                                 │
│  │  5. Draft executive summary │                                 │
│  └──────────┬──────────────────┘                                 │
│             │                                                    │
│             ▼                                                    │
│  ┌─────────────────────────────┐                                 │
│  │        EXECUTOR             │                                 │
│  │  (Capable LLM + tools)     │  ◄── executes step by step     │
│  │                             │      with tool access           │
│  │  Step 1 → result₁          │                                 │
│  │  Step 2 → result₂          │                                 │
│  │  ...                        │                                 │
│  └──────────┬──────────────────┘                                 │
│             │                                                    │
│             ▼                                                    │
│  ┌─────────────────────────────┐                                 │
│  │       REPLANNER             │                                 │
│  │  (After each step or batch) │                                 │
│  │                             │                                 │
│  │  "Step 2 failed. Adjust:   │                                 │
│  │   Replace step 2 with      │                                 │
│  │   alternative search..."    │                                 │
│  └─────────────────────────────┘                                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **6.2 Why Separate Planner and Executor?**

| Benefit | Explanation |
|---------|-------------|
| **Model efficiency** | Use a powerful (expensive) model for planning, cheaper model for execution |
| **Clearer architecture** | Each component has a single responsibility |
| **Easier debugging** | Can inspect the plan independently of execution |
| **Replanning** | Can update the plan mid-execution without restarting |
| **Parallelism** | Independent steps can be executed concurrently |

## **6.3 Plan-and-Execute vs ReAct vs ReWOO**

| Aspect | ReAct | ReWOO | Plan-and-Execute |
|--------|-------|-------|------------------|
| **Planning** | Implicit (per-step) | Upfront (one shot) | Upfront + replanning |
| **Execution** | LLM + tools interleaved | Tools batch then LLM | Separate executor agent |
| **Adaptability** | High (step-by-step) | Low (fixed plan) | Medium (replan after steps) |
| **Model usage** | Same model throughout | Same model (plan + solve) | Can use different models |
| **Best for** | Exploratory tasks | Cost-sensitive batch tasks | Complex multi-step projects |

## **6.4 Implementation Sketch (LangGraph)**

```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    current_step: int
    results: dict
    final_answer: str

def planner(state):
    plan = llm_strong.invoke(f"Create a step-by-step plan for: {state['input']}")
    return {"plan": parse_steps(plan), "current_step": 0}

def executor(state):
    step = state["plan"][state["current_step"]]
    result = agent_with_tools.invoke(step)
    results = {**state["results"], state["current_step"]: result}
    return {"results": results, "current_step": state["current_step"] + 1}

def replanner(state):
    updated_plan = llm_strong.invoke(
        f"Original plan: {state['plan']}\nResults so far: {state['results']}\n"
        f"Update remaining steps if needed."
    )
    return {"plan": parse_steps(updated_plan)}

def should_continue(state):
    if state["current_step"] >= len(state["plan"]):
        return "finish"
    return "execute"

graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("replanner", replanner)
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue,
    {"execute": "replanner", "finish": END})
graph.add_edge("replanner", "executor")
```

> **Interview tip:** Plan-and-Execute is the pattern behind many production systems (Devin, GPT-Engineer, AutoGPT v2). When asked about building complex agents, describe this: "I'd use a Planner (strong model) to decompose the task, an Executor (efficient model with tools) to carry out each step, and a Replanner to adapt when things go wrong."

---

# **7. Reflexion and Self-Reflection**

**Paper:** *Shinn et al. (2023), "Reflexion: Language Agents with Verbal Reinforcement Learning"*

## **7.1 Core Idea**

Reflexion lets an agent **learn from its own failures** within a single session — without updating model weights. After a failed attempt, the agent generates a **verbal reflection** ("what went wrong and how to fix it") and stores it in memory for the next attempt.

```
┌──────────────────────────────────────────────────────────────────┐
│                    REFLEXION PATTERN                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Attempt 1:                                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Task → Agent executes → Result → Evaluator → FAIL ✗     │    │
│  └──────────────────────────────┬───────────────────────────┘    │
│                                 │                                 │
│                                 ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ REFLECT: "I failed because I searched with too broad a    │    │
│  │ query. Next time, I should use specific keywords and      │    │
│  │ filter by date range."                                    │    │
│  │                                                           │    │
│  │ → Store reflection in MEMORY                              │    │
│  └──────────────────────────────┬───────────────────────────┘    │
│                                 │                                 │
│                                 ▼                                 │
│  Attempt 2:                                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Task + Previous Reflections → Agent executes (improved)  │    │
│  │ → Result → Evaluator → PASS ✓                            │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## **7.2 Three Components**

| Component | Role | Implementation |
|-----------|------|----------------|
| **Actor** | Executes the task (agent with tools) | ReAct agent, code generator, etc. |
| **Evaluator** | Determines success/failure + quality score | Unit tests, LLM judge, human feedback |
| **Self-Reflection** | Analyzes failure and generates improvement advice | LLM prompted: "Why did this fail? How to improve?" |

## **7.3 Why "Verbal Reinforcement Learning"?**

Traditional RL updates model weights after rewards. Reflexion instead:
- Stores **natural language feedback** in a memory buffer
- Uses this feedback as **additional context** in subsequent attempts
- Achieves RL-like improvement **without any gradient updates**

This is powerful because:
- No fine-tuning needed (works with any API model)
- Reflections are **interpretable** (you can read why it improved)
- Works within a single episode / session

## **7.4 Reflexion vs Simple Retry**

| Approach | What Happens on Failure | Memory | Improvement Rate |
|----------|------------------------|--------|-----------------|
| **Simple Retry** | Same prompt, hope for different output | None | Low (random variance) |
| **Retry + Error Message** | Include error in next attempt | Error only | Medium |
| **Reflexion** | Reflect on *why* it failed, store insight, retry with insight | Verbal reflections | High (targeted improvement) |

## **7.5 Self-Reflection Prompt Template**

```
You attempted the following task and failed.

Task: {task_description}
Your attempt: {agent_trajectory}
Result: {result}
Expected: {expected_outcome}
Error: {error_details}

Reflect on what went wrong. Be specific:
1. What was the root cause of the failure?
2. What should you do differently next time?
3. What assumptions were wrong?

Provide a concise reflection (2-3 sentences) that will help you succeed on the next attempt.
```

## **7.6 Applications**

- **Code generation:** Agent writes code → tests fail → reflects on test errors → writes better code (HumanEval benchmark: 91% pass with Reflexion vs 80% without)
- **Question answering:** Agent answers wrong → reflects on reasoning error → corrects approach
- **Decision making:** Agent makes suboptimal choice → evaluator flags it → agent adjusts strategy

> **Interview tip:** Reflexion is increasingly asked at companies building autonomous coding agents (Cursor, Devin, Copilot). The key insight: "It's RL for LLMs without weight updates — the model learns from failures through natural language self-critique stored in working memory."

---

# **8. LATS (Language Agent Tree Search)**

**Paper:** *Zhou et al. (2023), "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models"*

## **8.1 Core Idea**

LATS combines the best of ToT and ReAct: it applies **Monte Carlo Tree Search (MCTS)** over **agent action trajectories**. Instead of committing to one path of actions, LATS explores multiple paths, evaluates them, and backtracks when needed.

```
┌──────────────────────────────────────────────────────────────────┐
│                   LATS: Tree Search over Actions                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Root: Initial task state                                        │
│  ├── Action A₁ (search X)                                       │
│  │   ├── Action A₂ (analyze result) → Score: 0.7               │
│  │   └── Action A₃ (search Y)                                   │
│  │       └── Action A₄ (compare) → Score: 0.9  ← BEST PATH    │
│  ├── Action B₁ (calculate first)                                │
│  │   └── Action B₂ (search) → Score: 0.3  ← PRUNED            │
│  └── Action C₁ (ask clarification)                              │
│      └── Score: 0.1  ← PRUNED                                   │
│                                                                   │
│  MCTS Phases:                                                    │
│  1. SELECT: Choose promising node (UCB1 exploration/exploitation)│
│  2. EXPAND: Generate new child actions from selected node        │
│  3. EVALUATE: LLM scores the new state (0-1)                    │
│  4. BACKPROPAGATE: Update scores up the tree                    │
│                                                                   │
│  Unlike ToT: Actions are REAL (tool calls), not just thoughts   │
│  Unlike ReAct: Multiple action sequences explored, not just one │
└──────────────────────────────────────────────────────────────────┘
```

## **8.2 LATS vs Other Patterns**

| Pattern | Explores Multiple Paths? | Uses Tools? | Self-Evaluates? | Backtracking? |
|---------|-------------------------|-------------|-----------------|---------------|
| CoT | No | No | No | No |
| ToT | Yes (thoughts) | No | Yes | Yes |
| ReAct | No | Yes | No | No |
| Reflexion | No (retries full task) | Yes | Yes | Full restart |
| **LATS** | **Yes (actions)** | **Yes** | **Yes** | **Yes (partial)** |

## **8.3 When to Use LATS**

- Tasks where **the first approach often fails** and you want to explore alternatives
- **High-stakes decisions** where you want to evaluate multiple strategies before committing
- Tasks with **combinatorial action spaces** (many possible tool sequences)
- When you need **both exploration (ToT) and grounding (ReAct)**

**Trade-off:** LATS is the most expensive pattern — it makes many LLM calls for search, evaluation, and backpropagation. Only use when accuracy matters more than cost.

> **Interview tip:** LATS shows you understand the cutting edge. Frame it as: "LATS unifies the exploration of Tree-of-Thoughts with the tool-grounding of ReAct. It's Monte Carlo Tree Search applied to agent trajectories — the same algorithm that powers AlphaGo, but applied to LLM agents."

---

# **9. Multi-Agent Collaboration Patterns**

## **9.1 Debate / Adversarial Pattern**

Two or more agents argue different positions, then a judge synthesizes.

```
┌──────────────────────────────────────────────────────────────────┐
│                    DEBATE PATTERN                                  │
│                                                                   │
│  Question: "Should we use microservices or monolith?"            │
│                                                                   │
│  ┌──────────┐        ┌──────────┐                                │
│  │ Agent A   │        │ Agent B   │                               │
│  │ (Pro-Micro)│       │(Pro-Mono) │                               │
│  └─────┬─────┘        └─────┬─────┘                              │
│        │ "Microservices      │ "Monolith is simpler,             │
│        │  enable independent │  faster to develop,               │
│        │  scaling..."        │  easier to debug..."              │
│        └──────────┬──────────┘                                   │
│                   ▼                                               │
│         ┌──────────────────┐                                     │
│         │   JUDGE AGENT    │                                     │
│         │  Synthesizes both│                                     │
│         │  perspectives    │                                     │
│         └──────────────────┘                                     │
│                   │                                               │
│                   ▼                                               │
│  "For your team size (5 devs) and stage (MVP), start with       │
│   a modular monolith. Migrate to microservices when you hit     │
│   scaling bottlenecks on specific services."                     │
└──────────────────────────────────────────────────────────────────┘
```

**Why it works:** Forces consideration of multiple perspectives. Reduces bias toward a single approach.

## **9.2 Voting / Ensemble Pattern**

Multiple agents independently solve the same task, results are aggregated.

```
Task ──┬──► Agent 1 → Answer A
       ├──► Agent 2 → Answer A      → Majority Vote → A
       ├──► Agent 3 → Answer B
       └──► Agent 4 → Answer A
```

**Comparison with Self-Consistency:** Self-Consistency uses the *same model* with sampling. Voting uses *different agents* (potentially different models, prompts, or tools), which provides more diverse reasoning.

## **9.3 Supervisor-Worker Pattern**

A supervisor agent delegates tasks to specialized workers and aggregates results.

```
┌──────────────────────────────────────────────────────────┐
│                  SUPERVISOR-WORKER                         │
│                                                           │
│              ┌─────────────┐                              │
│              │  SUPERVISOR  │                              │
│              │  (Orchestrator)│                            │
│              └──┬───┬───┬──┘                              │
│                 │   │   │                                  │
│        ┌────────┘   │   └────────┐                        │
│        ▼            ▼            ▼                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  │ Research  │ │  Code    │ │ Review   │                  │
│  │ Worker   │ │ Worker   │ │ Worker   │                  │
│  └──────────┘ └──────────┘ └──────────┘                  │
│                                                           │
│  Supervisor decides: who to call, in what order,          │
│  whether to retry, when to finish                        │
└──────────────────────────────────────────────────────────┘
```

**Implementation:** This is LangGraph's most common production pattern. The supervisor is a node that uses an LLM to route to worker nodes based on the current state.

## **9.4 Handoff / Relay Pattern**

Agents pass control to each other sequentially, each contributing their specialty.

```
User Query → Agent A (Research) → Agent B (Analysis) → Agent C (Writing) → Output
                  │                     │                     │
             Web search           Data processing        Report drafting
```

**Use case:** OpenAI's Swarm framework uses this pattern. Each agent has a `handoff` function that transfers control to another agent.

## **9.5 Mixture-of-Agents (MoA)**

**Paper:** *Wang et al. (2024)*

Multiple "proposer" agents generate responses, then an "aggregator" model synthesizes them into a superior response. Can be layered (multi-round).

```
Layer 1:  Model A → Response A₁
          Model B → Response B₁    → Aggregator → Refined Response
          Model C → Response C₁

Layer 2:  Model A (with Layer 1 output) → Response A₂
          Model B (with Layer 1 output) → Response B₂  → Aggregator → Final
          Model C (with Layer 1 output) → Response C₂
```

**Key insight:** Weaker models as proposers + strong model as aggregator can outperform the strong model alone. This is because diverse weak responses give the aggregator more material to work with.

## **9.6 Multi-Agent Pattern Comparison**

| Pattern | Agents | Communication | Best For | Latency | Cost |
|---------|--------|--------------|----------|---------|------|
| **Debate** | 2-3 + judge | Sequential arguments | Balanced analysis, reducing bias | High | High |
| **Voting** | N independent | No communication | Accuracy on factual tasks | Low (parallel) | N× |
| **Supervisor** | 1 supervisor + N workers | Hub-and-spoke | Complex projects with subtasks | Medium | Variable |
| **Handoff** | N sequential | Linear relay | Pipeline workflows (research→analysis→report) | High | Medium |
| **MoA** | N proposers + aggregator | Layered | Maximum quality, ensemble reasoning | High | High |

---

# **10. Pattern Selection Guide**

## **10.1 Decision Flowchart**

```
START: What kind of task?
│
├── Simple reasoning (math, logic)?
│   └── Chain-of-Thought (or Self-Consistency if accuracy matters)
│
├── Complex problem with multiple approaches?
│   └── Tree-of-Thoughts
│
├── Needs real-world data (search, APIs)?
│   ├── Steps are predictable and independent?
│   │   └── ReWOO (batch plan + parallel execute)
│   ├── Steps are exploratory, each depends on last?
│   │   └── ReAct (interleaved reasoning + acting)
│   └── Complex project with many steps?
│       └── Plan-and-Execute (planner + executor + replanner)
│
├── Task might fail, want to improve from failures?
│   └── Reflexion (execute → evaluate → reflect → retry)
│
├── High-stakes, need to explore multiple strategies?
│   └── LATS (tree search over action sequences)
│
└── Needs multiple perspectives or specialties?
    ├── Want balanced analysis? → Debate
    ├── Want max accuracy? → Voting / MoA
    ├── Want task decomposition? → Supervisor-Worker
    └── Want pipeline workflow? → Handoff
```

## **10.2 Cost-Accuracy Trade-off**

```
                 High Accuracy
                      │
              LATS ●  │  ● MoA
                      │
     Reflexion ●      │      ● Debate
                      │
Plan-and-Execute ●    │    ● Supervisor
                      │
          ReAct ●     │     ● Voting
                      │
         ReWOO ●      │
                      │
   Self-Consistency ● │
                      │
            CoT ●     │
                      │
         ──────────────┼──────────────
          Low Cost     │     High Cost
```

## **10.3 Production Recommendations**

| Company Stage | Recommended Patterns | Rationale |
|--------------|---------------------|-----------|
| **MVP / Startup** | ReAct, CoT | Simple, fast to implement, low cost |
| **Growth** | Plan-and-Execute, ReWOO | Better reliability, cost-efficiency at scale |
| **Enterprise** | Supervisor-Worker, Reflexion, Debate | Complex workflows, quality assurance, audit trails |
| **Research / High-stakes** | LATS, MoA, ToT | Maximum accuracy, cost is secondary |

---

# **11. Interview Questions with Strong Answers**

---

## **Q1: "Explain Chain-of-Thought prompting and when you'd use it vs Tree-of-Thoughts."**

> CoT asks the model to show its reasoning steps before answering — either via "think step by step" (zero-shot) or with worked examples (few-shot). It's a **linear reasoning chain** that dramatically improves math, logic, and multi-hop tasks with minimal overhead (just one LLM call).
>
> Tree-of-Thoughts extends CoT by generating **multiple candidate reasoning paths** at each step, evaluating them ("sure/maybe/impossible"), and searching the tree using BFS or DFS. It's like CoT with backtracking.
>
> **When to choose:**
> - **CoT:** Straightforward reasoning where one good chain suffices. Cost: 1 LLM call.
> - **CoT + Self-Consistency:** When accuracy matters. Sample N chains, majority vote. Cost: N calls.
> - **ToT:** When the problem has **multiple valid approaches** and the first approach often fails — puzzles, creative writing, planning. Cost: K × depth calls.

---

## **Q2: "What is ReWOO and how does it differ from ReAct?"**

> ReWOO stands for **Reasoning Without Observation**. In ReAct, the agent interleaves thinking and acting — each step waits for the tool result before planning the next step. This is powerful but creates a sequential bottleneck.
>
> ReWOO **decouples planning from execution**: the LLM generates the entire plan upfront with placeholders (#E1, #E2) for tool results, then all tool calls execute (potentially in parallel), and finally one LLM call combines everything into the answer.
>
> The trade-off: ReWOO uses **~5× fewer tokens** and enables parallel tool execution, making it faster and cheaper. But it can't adapt mid-execution — if the plan was wrong, you need to start over. ReAct can pivot after each observation.
>
> **My rule:** Use ReWOO for **predictable, structured tasks** (compare two things, look up N items). Use ReAct for **exploratory tasks** where each step depends on what you find.

---

## **Q3: "Describe the Reflexion pattern. How is it different from just retrying?"**

> Reflexion is **verbal reinforcement learning** — the agent learns from failures without updating weights. After a failed attempt, instead of just retrying (which is random), the agent generates a **natural language reflection**: what went wrong, what to do differently, what assumptions were incorrect.
>
> This reflection is stored in memory and included as context in the next attempt. So the agent doesn't just retry — it retries **with accumulated wisdom from past failures**.
>
> Concretely: attempt → evaluate (did it pass tests? was the answer correct?) → if fail, reflect → store reflection → retry with reflections in context.
>
> On coding benchmarks like HumanEval, Reflexion achieves 91% pass rate vs 80% for standard approaches. The key is that reflections are **targeted and specific** ("I forgot to handle the edge case of empty lists") rather than generic retry noise.

---

## **Q4: "How would you design an agent system for a complex task like autonomous code generation?"**

> I'd combine **Plan-and-Execute** with **Reflexion**:
>
> 1. **Planner** (GPT-4 class): Breaks the coding task into steps — understand requirements, design interfaces, implement core logic, write tests, integrate.
> 2. **Executor** (efficient model + code tools): Executes each step — generates code, runs it in a sandbox, checks for errors.
> 3. **Evaluator**: Runs test suite, checks lint, validates output.
> 4. **Reflexion loop**: If tests fail, the agent reflects on *why* (wrong algorithm? edge case? type error?), stores the reflection, and retries that step with the insight.
> 5. **Replanner**: If a step fails multiple times, the planner revises the overall approach.
>
> For critical decisions (architecture choices), I might add a **Debate** pattern: one agent argues for approach A, another for approach B, and the planner picks based on the arguments.

---

## **Q5: "Compare the agentic patterns in terms of cost, latency, and reliability."**

> | Pattern | LLM Calls | Latency | Cost | Reliability | Use Case |
> |---------|-----------|---------|------|-------------|----------|
> | CoT | 1 | Low | Low | Medium | Quick reasoning tasks |
> | Self-Consistency | N | Medium | N× | High | Critical accuracy needs |
> | ToT | K×D | High | High | High | Complex multi-path problems |
> | ReAct | N steps | High | Medium | Medium | Tool-using tasks |
> | ReWOO | 2 | Low | Low | Medium | Structured tool tasks |
> | Plan-Execute | 2 + N steps | Medium | Medium | High | Complex projects |
> | Reflexion | Attempts × steps | High | High | Very High | Tasks where failure is common |
> | LATS | Search × depth | Very High | Very High | Highest | High-stakes decisions |

---

## **Q6: "What's the Mixture-of-Agents approach and when would you use it?"**

> MoA uses multiple LLMs as "proposers" that each generate independent responses, then an "aggregator" model synthesizes them into a final, superior response. It can be multi-layered — each layer's output feeds into the next round of proposals.
>
> The surprising finding: **weaker models as proposers + strong model as aggregator often outperforms the strong model alone**. This is because diverse proposers generate a wider range of ideas, examples, and reasoning paths that the aggregator can cherry-pick from.
>
> I'd use it when **quality is paramount and latency/cost are secondary** — generating important customer-facing content, making high-stakes recommendations, or in any scenario where I want the equivalent of "multiple expert opinions synthesized by a senior expert."

---

## **Q7: "You're building a customer support agent. Which pattern(s) do you choose and why?"**

> For a customer support agent, I'd use:
>
> 1. **ReAct** as the base pattern — the agent needs to look up customer info (tool: CRM query), check order status (tool: order API), and reason about what to do next. Each step depends on the previous observation.
>
> 2. **Router pattern** to classify intent first — billing, technical, returns, general inquiry — and route to a specialized sub-agent for each.
>
> 3. **Reflexion light** — if the customer says "that's not what I asked," the agent reflects on its misunderstanding and adjusts. Store common misunderstanding patterns as reflections.
>
> I would NOT use ToT (overkill for support), LATS (too expensive for high-volume support), or ReWOO (support steps are highly dependent on each other — you can't plan everything upfront without knowing the customer's problem).

---

# **Key Takeaways**

```
┌──────────────────────────────────────────────────────────────────┐
│                  AGENTIC PATTERNS — KEY TAKEAWAYS                │
│                                                                   │
│  1. CoT is the FOUNDATION — all other patterns build on it      │
│                                                                   │
│  2. ReAct = CoT + Tools (most commonly asked in interviews)     │
│                                                                   │
│  3. ReWOO = ReAct with decoupled planning (5× cheaper)          │
│                                                                   │
│  4. Plan-and-Execute = separate planning and execution agents    │
│     (production pattern behind Devin, GPT-Engineer)              │
│                                                                   │
│  5. ToT = exploring multiple reasoning branches (search)        │
│                                                                   │
│  6. Reflexion = learning from failures via verbal reflection    │
│     (RL without weight updates)                                  │
│                                                                   │
│  7. LATS = tree search over agent actions (highest accuracy,    │
│     highest cost)                                                │
│                                                                   │
│  8. Multi-agent: Debate for balance, Voting for accuracy,       │
│     Supervisor for complexity, Handoff for pipelines, MoA       │
│     for maximum quality                                          │
│                                                                   │
│  9. START SIMPLE (CoT/ReAct), add complexity only when needed   │
│                                                                   │
│  10. Know the papers: Wei'22 (CoT), Yao'23 (ToT, ReAct),      │
│      Xu'23 (ReWOO), Shinn'23 (Reflexion), Zhou'23 (LATS)      │
└──────────────────────────────────────────────────────────────────┘
```
