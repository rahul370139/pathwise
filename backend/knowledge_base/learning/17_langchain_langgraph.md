# LangChain & LangGraph — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LangChain, LangGraph, MCP, RAG Systems, Agentic AI

---

# Table of Contents

1. [What Is LangChain?](#1-what-is-langchain)
2. [Core Components](#2-core-components)
3. [Chains — Composing LLM Pipelines](#3-chains--composing-llm-pipelines)
4. [LCEL Deep Dive — LangChain Expression Language](#4-lcel-deep-dive--langchain-expression-language)
5. [Memory Types](#5-memory-types)
6. [Tools and Tool Use](#6-tools-and-tool-use)
7. [Document Loaders and Text Splitters](#7-document-loaders-and-text-splitters)
8. [Building a RAG Chain — End-to-End](#8-building-a-rag-chain--end-to-end)
9. [Building an Agent](#9-building-an-agent)
10. [What Is LangGraph?](#10-what-is-langgraph)
11. [LangGraph Core Concepts](#11-langgraph-core-concepts)
12. [LangGraph State Management](#12-langgraph-state-management)
13. [Building a LangGraph Agent — Full Example](#13-building-a-langgraph-agent--full-example)
14. [LangChain vs LangGraph](#14-langchain-vs-langgraph)
15. [Production Patterns](#15-production-patterns)
16. [Common Interview Questions](#16-common-interview-questions-with-strong-answers)
17. [Key Takeaways](#17-key-takeaways)

---

# **1. What Is LangChain?**

---

## **1.1 Definition**

LangChain is an **open-source framework** for building applications powered by large language models (LLMs). It provides modular, composable abstractions that let you connect LLMs to external data sources, tools, and memory — turning a stateless text-completion model into a **context-aware, action-capable system**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     LANGCHAIN — THE BIG PICTURE                          │
│                                                                          │
│   Raw LLM API                        LangChain Application               │
│   ──────────                          ────────────────────               │
│   "Complete this text"                "Answer user questions about        │
│                                        their uploaded PDFs using          │
│   → Stateless                          retrieval-augmented generation,    │
│   → No memory                          remember conversation history,    │
│   → No tools                           search the web when needed,       │
│   → No data access                     and cite sources."                │
│                                                                          │
│   LangChain bridges the gap between raw LLM capabilities and            │
│   production-grade applications.                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** Don't just say "LangChain is a framework for LLMs." Explain what it *enables* — connecting LLMs to memory, tools, and data — and why that matters for production systems.

---

## **1.2 LangChain Ecosystem**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       LANGCHAIN ECOSYSTEM (2024+)                        │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ langchain-    │  │  langchain   │  │  langgraph   │  │  langsmith  │  │
│  │ core          │  │              │  │              │  │             │  │
│  │               │  │  Higher-     │  │  Graph-based │  │  Observa-   │  │
│  │  Base ab-     │  │  level       │  │  orchest-    │  │  bility,    │  │
│  │  stractions,  │  │  chains,     │  │  ration,     │  │  tracing,   │  │
│  │  LCEL,        │  │  agents,     │  │  stateful    │  │  eval,      │  │
│  │  runnables    │  │  retrieval   │  │  multi-actor │  │  testing    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
│         ▲                  ▲                  ▲                ▲          │
│         │                  │                  │                │          │
│         └──────────── All work together ──────┘                │          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  langchain-community / langchain-openai / langchain-anthropic   │    │
│  │  Third-party integrations (700+ integrations)                   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

| Package | Purpose |
|---------|---------|
| **langchain-core** | Base abstractions: Runnables, LCEL, prompts, output parsers |
| **langchain** | Chains, agents, retrieval strategies — higher-level composition |
| **langgraph** | Graph-based orchestration with cycles, state, and human-in-the-loop |
| **langsmith** | Observability platform — tracing, evaluation, monitoring |
| **langchain-community** | 700+ third-party integrations (vector stores, LLMs, tools) |
| **langchain-openai** | OpenAI-specific wrappers (ChatOpenAI, OpenAIEmbeddings) |

---

## **1.3 When to Use LangChain**

| Use Case | Why LangChain |
|----------|---------------|
| RAG applications | Built-in document loaders, splitters, retrievers, and chain templates |
| Chatbots with memory | Conversation memory abstractions out of the box |
| Tool-using agents | Standardized tool interface, agent executors, function calling |
| Multi-step LLM pipelines | LCEL provides clean, composable pipeline syntax |
| Prototyping to production | Same abstractions work in notebooks and deployed services |

---

# **2. Core Components**

---

## **2.1 Component Overview**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LANGCHAIN CORE COMPONENTS                              │
│                                                                          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐  │
│  │  LLMs / │   │ Prompt  │   │ Output  │   │  Memory │   │  Tools   │  │
│  │  Chat   │   │Templates│   │ Parsers │   │         │   │          │  │
│  │  Models │   │         │   │         │   │         │   │          │  │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬─────┘  │
│       │              │              │              │              │       │
│       └──────────────┴──────┬───────┴──────────────┴──────────────┘       │
│                             │                                            │
│                             ▼                                            │
│                    ┌────────────────┐                                     │
│                    │   Chains /     │                                     │
│                    │   LCEL Pipes   │                                     │
│                    └────────┬───────┘                                     │
│                             │                                            │
│                             ▼                                            │
│                    ┌────────────────┐                                     │
│                    │   Agents /     │                                     │
│                    │   LangGraph    │                                     │
│                    └────────────────┘                                     │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│  │  Document    │   │    Text      │   │  Retrievers  │                  │
│  │  Loaders     │   │  Splitters   │   │  (VectorDB)  │                  │
│  └──────────────┘   └──────────────┘   └──────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **2.2 LLMs and Chat Models**

LangChain wraps LLM providers behind a **uniform interface** so you can swap models without changing application logic.

```python
# LLM (completion-style) — older pattern
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
response = llm.invoke("Explain gradient descent in one sentence.")

# Chat Model (message-based) — modern standard
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o", temperature=0)
from langchain_core.messages import HumanMessage, SystemMessage

response = chat.invoke([
    SystemMessage(content="You are a concise ML tutor."),
    HumanMessage(content="What is backpropagation?")
])
print(response.content)
```

| Feature | LLM (Completion) | Chat Model (Messages) |
|---------|-------------------|----------------------|
| Input | Single string | List of messages (System, Human, AI) |
| Output | String | AIMessage object |
| Use case | Simple completions | Conversations, function calling, agents |
| Modern usage | Legacy | **Recommended** |

---

## **2.3 Prompt Templates**

Prompt templates separate **prompt logic** from **data** — enabling reuse, testing, and versioning.

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who explains concepts for {audience}."),
    ("human", "{question}")
])

# Format with variables
messages = prompt.format_messages(
    role="data science tutor",
    audience="interview candidates",
    question="What is a transformer?"
)

# Few-shot prompt
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "What is overfitting?", "output": "When a model memorizes training data..."},
    {"input": "What is regularization?", "output": "Techniques to prevent overfitting..."},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ML interview coach. Answer concisely."),
    few_shot,
    ("human", "{question}")
])
```

---

## **2.4 Output Parsers**

Output parsers **structure LLM output** into Python objects — critical for downstream processing.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Simple string parser
str_parser = StrOutputParser()

# Structured JSON output with Pydantic
class InterviewAnswer(BaseModel):
    concept: str = Field(description="The ML concept")
    explanation: str = Field(description="Clear explanation")
    example: str = Field(description="Practical example")

json_parser = JsonOutputParser(pydantic_object=InterviewAnswer)

# Use in a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in JSON format.\n{format_instructions}"),
    ("human", "Explain {concept}")
])

chain = prompt | chat | json_parser
result = chain.invoke({
    "concept": "attention mechanism",
    "format_instructions": json_parser.get_format_instructions()
})
# result = InterviewAnswer(concept="attention mechanism", explanation="...", example="...")
```

---

# **3. Chains — Composing LLM Pipelines**

---

## **3.1 What Are Chains?**

A **chain** is a sequence of operations where the output of one step feeds into the next. Chains are the fundamental building blocks for multi-step LLM applications.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CHAIN TYPES                                      │
│                                                                          │
│  SEQUENTIAL CHAIN          ROUTER CHAIN            MAP-REDUCE CHAIN      │
│  ────────────────          ────────────             ──────────────────    │
│                                                                          │
│  Step 1 → Step 2          Input                    Chunk 1 ─┐            │
│    │                        │                       Chunk 2 ─┤→ Combine  │
│    ▼                     ┌──┴──┐                    Chunk 3 ─┤   (Reduce)│
│  Step 3 → Output         │Route│                    Chunk N ─┘            │
│                          └┬─┬─┬┘                      ▲                  │
│  Linear pipeline.     Chain Chain Chain           Map step:              │
│  Each step gets          A    B    C             process each            │
│  previous output.     (specialized)              chunk in parallel.      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **3.2 Sequential Chain**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 1: Generate a technical explanation
step1 = ChatPromptTemplate.from_messages([
    ("system", "You are an ML expert."),
    ("human", "Explain {topic} in detail.")
]) | llm | StrOutputParser()

# Step 2: Simplify for interviews
step2 = ChatPromptTemplate.from_messages([
    ("system", "Simplify this ML explanation for a job interview. "
               "Keep it under 3 sentences."),
    ("human", "{explanation}")
]) | llm | StrOutputParser()

# Compose: output of step1 feeds into step2
from langchain_core.runnables import RunnablePassthrough

sequential_chain = (
    {"explanation": step1}
    | step2
)

result = sequential_chain.invoke({"topic": "gradient descent"})
```

---

## **3.3 Router Chain**

Routes input to specialized sub-chains based on classification.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Specialized chains
ml_chain = ChatPromptTemplate.from_messages([
    ("system", "You are an ML expert. Answer ML questions."),
    ("human", "{question}")
]) | llm | StrOutputParser()

code_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert. Answer coding questions."),
    ("human", "{question}")
]) | llm | StrOutputParser()

general_chain = ChatPromptTemplate.from_messages([
    ("system", "Answer the question helpfully."),
    ("human", "{question}")
]) | llm | StrOutputParser()

# Router function
def route(info):
    question = info["question"].lower()
    if any(kw in question for kw in ["model", "training", "neural", "ml", "ai"]):
        return ml_chain
    elif any(kw in question for kw in ["code", "python", "function", "debug"]):
        return code_chain
    else:
        return general_chain

# Router chain using RunnableLambda
from langchain_core.runnables import RunnableLambda

router_chain = RunnableLambda(route)
result = router_chain.invoke({"question": "How does batch normalization work?"})
```

---

## **3.4 Map-Reduce Chain**

Processes multiple chunks independently (map) then combines results (reduce). Essential for handling documents that exceed context windows.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Map step: summarize each chunk
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text chunk in 2-3 sentences."),
    ("human", "{chunk}")
])
map_chain = map_prompt | llm | StrOutputParser()

# Reduce step: combine summaries
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "Combine these summaries into a single coherent summary."),
    ("human", "{summaries}")
])
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Full map-reduce
def map_reduce(chunks: list[str]) -> str:
    # Map: process each chunk
    summaries = map_chain.batch([{"chunk": c} for c in chunks])
    # Reduce: combine
    combined = "\n---\n".join(summaries)
    return reduce_chain.invoke({"summaries": combined})
```

---

# **4. LCEL Deep Dive — LangChain Expression Language**

---

## **4.1 What Is LCEL?**

LCEL (LangChain Expression Language) is a **declarative, composable syntax** for building LLM pipelines using the **pipe operator** (`|`). It is the modern, recommended way to build chains in LangChain.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       LCEL — CORE IDEA                                   │
│                                                                          │
│   Traditional (imperative):         LCEL (declarative):                  │
│   ─────────────────────────         ───────────────────                  │
│   result = prompt.format(...)       chain = prompt | llm | parser        │
│   response = llm.call(result)       result = chain.invoke({"topic": x})  │
│   parsed = parser.parse(response)                                        │
│                                                                          │
│   LCEL composes Runnables with |                                         │
│   Every component is a Runnable with:                                    │
│     .invoke()   — single input                                           │
│     .batch()    — multiple inputs                                        │
│     .stream()   — streaming output                                       │
│     .ainvoke()  — async single input                                     │
│     .abatch()   — async multiple inputs                                  │
│     .astream()  — async streaming                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** LCEL is not just syntactic sugar. Every LCEL chain automatically gets **streaming, batching, async, retries, fallbacks, and LangSmith tracing** — for free. That's the real value.

---

## **4.2 The Pipe Operator**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# The | operator connects Runnables
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful ML tutor."),
        ("human", "Explain {concept} simply.")
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0)
    | StrOutputParser()
)

# invoke: single call
result = chain.invoke({"concept": "attention mechanism"})

# stream: token-by-token
for chunk in chain.stream({"concept": "attention mechanism"}):
    print(chunk, end="", flush=True)

# batch: multiple inputs at once
results = chain.batch([
    {"concept": "transformers"},
    {"concept": "LSTM"},
    {"concept": "CNN"},
])

# async
import asyncio
result = asyncio.run(chain.ainvoke({"concept": "GAN"}))
```

---

## **4.3 RunnableParallel**

Runs multiple chains **simultaneously** and merges their outputs into a dict.

```python
from langchain_core.runnables import RunnableParallel

# Define parallel branches
analysis = RunnableParallel(
    summary=ChatPromptTemplate.from_messages([
        ("human", "Summarize this text: {text}")
    ]) | llm | StrOutputParser(),

    sentiment=ChatPromptTemplate.from_messages([
        ("human", "What is the sentiment of: {text}")
    ]) | llm | StrOutputParser(),

    keywords=ChatPromptTemplate.from_messages([
        ("human", "Extract 5 keywords from: {text}")
    ]) | llm | StrOutputParser(),
)

# All three run in parallel
result = analysis.invoke({"text": "LangChain is an amazing framework..."})
# result = {"summary": "...", "sentiment": "positive", "keywords": "..."}
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    RunnableParallel EXECUTION                             │
│                                                                          │
│                          Input                                           │
│                       {"text": "..."}                                    │
│                            │                                             │
│              ┌─────────────┼─────────────┐                               │
│              ▼             ▼             ▼                                │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐                            │
│        │ summary  │ │sentiment │ │ keywords │  ← runs in parallel        │
│        │  chain   │ │  chain   │ │  chain   │                            │
│        └────┬─────┘ └────┬─────┘ └────┬─────┘                            │
│             │            │            │                                   │
│             └────────────┼────────────┘                                   │
│                          ▼                                               │
│              {"summary": "...",                                           │
│               "sentiment": "...",                                         │
│               "keywords": "..."}                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **4.4 RunnableLambda**

Wraps any Python function into a Runnable, making it composable in LCEL chains.

```python
from langchain_core.runnables import RunnableLambda

# Wrap a custom function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_step = RunnableLambda(format_docs)

# Use in a chain
chain = retriever | format_step | prompt | llm | StrOutputParser()

# Lambda with error handling
def safe_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": text}

safe_parser = RunnableLambda(safe_parse)
```

---

## **4.5 RunnablePassthrough**

Passes input through unchanged — used to **forward input alongside** transformed data.

```python
from langchain_core.runnables import RunnablePassthrough

# Common RAG pattern: pass the question alongside retrieved docs
chain = (
    {
        "context": retriever | format_docs,   # retrieves and formats docs
        "question": RunnablePassthrough()      # passes the question through unchanged
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("What is attention in transformers?")
# prompt receives: {"context": "<retrieved docs>", "question": "What is attention...?"}
```

---

## **4.6 RunnableBranch (Conditional Logic)**

```python
from langchain_core.runnables import RunnableBranch

# Conditional routing based on input
branch = RunnableBranch(
    (lambda x: "code" in x["topic"].lower(), code_chain),
    (lambda x: "math" in x["topic"].lower(), math_chain),
    general_chain  # default fallback
)

result = branch.invoke({"topic": "code optimization"})
```

---

## **4.7 Fallbacks and Retries**

```python
# Fallback: if GPT-4 fails, try GPT-3.5
primary = ChatOpenAI(model="gpt-4o", temperature=0)
fallback = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

reliable_llm = primary.with_fallbacks([fallback])

# Retry: automatically retry on failure
from langchain_core.runnables import RunnableConfig

chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1
)
```

---

## **4.8 LCEL Runnable Summary Table**

| Runnable | Purpose | Example |
|----------|---------|---------|
| **Pipe `\|`** | Connect components sequentially | `prompt \| llm \| parser` |
| **RunnableParallel** | Run branches simultaneously | `RunnableParallel(a=chain_a, b=chain_b)` |
| **RunnableLambda** | Wrap any Python function | `RunnableLambda(my_func)` |
| **RunnablePassthrough** | Forward input unchanged | `RunnablePassthrough()` in dict |
| **RunnableBranch** | Conditional routing | `RunnableBranch((cond, chain), default)` |
| **with_fallbacks()** | Graceful degradation | `llm.with_fallbacks([backup_llm])` |
| **with_retry()** | Automatic retry on failure | `chain.with_retry(stop_after_attempt=3)` |

---

# **5. Memory Types**

---

## **5.1 Why Memory Matters**

LLMs are **stateless** — each API call is independent. Memory modules inject conversation history into prompts so the model can maintain context across turns.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     MEMORY — THE PROBLEM                                 │
│                                                                          │
│   Turn 1: "My name is Rahul"         → LLM: "Nice to meet you, Rahul"  │
│   Turn 2: "What's my name?"          → LLM: "I don't know your name"   │
│                                              ← WITHOUT memory!           │
│                                                                          │
│   With memory:                                                           │
│   Turn 2: "What's my name?"          → LLM: "Your name is Rahul"       │
│                                              ← Memory injected Turn 1    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **5.2 Memory Type Comparison**

| Memory Type | How It Works | Token Cost | Best For |
|-------------|-------------|------------|----------|
| **ConversationBufferMemory** | Stores **all** messages verbatim | Grows linearly | Short conversations (<20 turns) |
| **ConversationBufferWindowMemory** | Keeps last **k** turns | Fixed (k turns) | Medium conversations, bounded cost |
| **ConversationSummaryMemory** | LLM **summarizes** conversation periodically | Fixed (summary size) | Long conversations |
| **ConversationSummaryBufferMemory** | Hybrid: recent turns verbatim + older turns summarized | Bounded | Production chatbots |
| **VectorStoreMemory** | Embeds messages, retrieves **relevant** ones via similarity | Fixed (top-k results) | Large-scale, topic-switching chats |
| **EntityMemory** | Extracts and tracks **entities** mentioned | Moderate | Entity-heavy conversations (CRM, support) |

---

## **5.3 Code Examples**

```python
# 1. ConversationBufferMemory — stores everything
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "My name is Rahul"},
    {"output": "Nice to meet you, Rahul!"}
)
memory.save_context(
    {"input": "I work on LangChain projects"},
    {"output": "That's great! LangChain is a powerful framework."}
)
print(memory.load_memory_variables({}))
# {'history': [HumanMessage("My name is Rahul"), AIMessage("Nice to meet you..."), ...]}

# 2. ConversationBufferWindowMemory — sliding window
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3, return_messages=True)
# Only keeps the last 3 exchanges

# 3. ConversationSummaryMemory — summarizes old context
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=ChatOpenAI(temperature=0))
# Uses an LLM to progressively summarize conversation history
# After many turns: "The user is Rahul, a data scientist. He's working on..."

# 4. VectorStoreMemory — semantic retrieval of past messages
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["placeholder"], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

memory = VectorStoreRetrieverMemory(retriever=retriever)
# Retrieves the most relevant past messages, not just recent ones
```

---

## **5.4 Choosing the Right Memory**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   MEMORY DECISION TREE                                    │
│                                                                          │
│   How long are your conversations?                                       │
│         │                                                                │
│    ┌────┴────┐                                                           │
│    │         │                                                           │
│   Short    Long (50+ turns)                                              │
│   (<20      │                                                            │
│   turns)    ├── Need topic switching? → VectorStoreMemory                │
│    │        ├── Need entity tracking? → EntityMemory                     │
│    │        └── General chatbot?      → ConversationSummaryBufferMemory  │
│    │                                                                     │
│    └── ConversationBufferMemory (simple, complete history)               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# **6. Tools and Tool Use**

---

## **6.1 What Are Tools?**

Tools are **functions the LLM can invoke** to interact with the outside world — search the web, query databases, execute code, or call APIs. The LLM decides *when* and *how* to use tools based on the user's request.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      TOOL USE FLOW                                       │
│                                                                          │
│   User Query: "What's the weather in NYC?"                               │
│                                                                          │
│   ┌───────┐    Decides to     ┌──────────────┐    Returns     ┌───────┐ │
│   │  LLM  │ ──────────────► │  weather_api  │ ──────────── │  LLM  │  │
│   │       │  use a tool       │  tool         │  result       │       │  │
│   │       │                   └──────────────┘               │       │  │
│   │       │                                                   │       │  │
│   │       │◄──────────────────────────────────────────────── │       │  │
│   └───────┘   Formats final answer using tool result          └───────┘  │
│                                                                          │
│   Answer: "It's 72°F and sunny in NYC right now."                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **6.2 Defining Custom Tools**

```python
from langchain_core.tools import tool

# Method 1: @tool decorator (recommended)
@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use when the user asks
    about recent events, weather, or real-time data."""
    # In production: call a real search API (Tavily, SerpAPI, etc.)
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any math calculations."""
    try:
        result = eval(expression)  # In production: use a safe math parser
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def query_database(sql: str) -> str:
    """Execute a SQL query against the company database.
    Use when the user asks about sales, users, or business metrics."""
    # In production: connect to real database
    return f"Query results for: {sql}"

# Method 2: StructuredTool for complex inputs
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")

search_tool = StructuredTool.from_function(
    func=lambda query, num_results: f"Top {num_results} results for: {query}",
    name="advanced_search",
    description="Advanced web search with configurable result count.",
    args_schema=SearchInput,
)
```

---

## **6.3 Built-in Tool Integrations**

```python
# Web Search (Tavily — recommended by LangChain)
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)

# Python Code Execution
from langchain_experimental.tools import PythonREPLTool
python_tool = PythonREPLTool()

# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Requests (API calls)
from langchain_community.tools import RequestsGetTool
from langchain_community.utilities import TextRequestsWrapper
requests_tool = RequestsGetTool(requests_wrapper=TextRequestsWrapper())
```

---

## **6.4 Binding Tools to a Model (Function Calling)**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind tools to the LLM
tools = [search_web, calculate, query_database]
llm_with_tools = llm.bind_tools(tools)

# The LLM now returns tool_calls when appropriate
response = llm_with_tools.invoke("What is 15% of 2847?")
print(response.tool_calls)
# [{'name': 'calculate', 'args': {'expression': '0.15 * 2847'}, 'id': '...'}]
```

---

# **7. Document Loaders and Text Splitters**

---

## **7.1 Document Loaders**

Document loaders **ingest data** from various sources into LangChain's `Document` format.

```python
# PDF Loader
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("research_paper.pdf")
docs = loader.load()  # List[Document], each page = one document

# Web Page Loader
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/blog-post")
docs = loader.load()

# CSV Loader
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()  # Each row = one document

# Directory Loader (batch load)
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Database Loader (SQL)
from langchain_community.document_loaders import SQLDatabaseLoader
loader = SQLDatabaseLoader(
    query="SELECT title, content FROM articles WHERE published = true",
    db=database
)
docs = loader.load()
```

| Loader | Source | Format |
|--------|--------|--------|
| PyPDFLoader | PDF files | One doc per page |
| WebBaseLoader | Web URLs | One doc per URL |
| CSVLoader | CSV files | One doc per row |
| TextLoader | Plain text | One doc per file |
| UnstructuredLoader | Any format (HTML, Word, etc.) | Auto-detects format |
| DirectoryLoader | Entire directories | Batch loading with glob patterns |
| JSONLoader | JSON files | Configurable jq-style extraction |

---

## **7.2 Text Splitters**

Text splitters **break large documents** into smaller chunks for embedding and retrieval. The goal: chunks that are **semantically coherent** and fit within embedding model token limits.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      WHY SPLIT TEXT?                                      │
│                                                                          │
│   Raw Document (50,000 tokens)                                           │
│   ┌────────────────────────────────────────────┐                         │
│   │  Chapter 1: Introduction...................│                         │
│   │  Chapter 2: Methods.......................│  → Too large to embed    │
│   │  Chapter 3: Results.......................│    or fit in context     │
│   │  Chapter 4: Discussion....................│                         │
│   └────────────────────────────────────────────┘                         │
│                                                                          │
│   After Splitting (500 tokens each, with overlap)                        │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
│   │Chunk │ │Chunk │ │Chunk │ │Chunk │ │Chunk │  → Embeddable,           │
│   │  1   │ │  2   │ │  3   │ │  4   │ │  5   │    retrievable           │
│   └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                          │
│       ◄─overlap─►                                                        │
│   Overlap preserves context at chunk boundaries.                         │
└──────────────────────────────────────────────────────────────────────────┘
```

```python
# Recursive Character Text Splitter (most versatile — recommended)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # max characters per chunk
    chunk_overlap=200,      # overlap between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # split hierarchy
)
chunks = splitter.split_documents(docs)

# Token-Based Splitter (when you need precise token counts)
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,       # max tokens per chunk
    chunk_overlap=50,     # token overlap
    encoding_name="cl100k_base"  # tiktoken encoding (GPT-4)
)
chunks = splitter.split_documents(docs)

# Markdown Header Splitter (preserves document structure)
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = splitter.split_text(markdown_text)
```

| Splitter | Strategy | Best For |
|----------|----------|----------|
| **RecursiveCharacterTextSplitter** | Splits by hierarchy: paragraphs → sentences → words | General purpose (default choice) |
| **TokenTextSplitter** | Splits by token count using tiktoken | Precise token budget control |
| **MarkdownHeaderTextSplitter** | Splits on markdown headers, preserves hierarchy | Structured docs, wikis |
| **HTMLHeaderTextSplitter** | Splits on HTML tags | Web content |
| **SemanticChunker** | Splits by semantic similarity (embedding-based) | High-quality retrieval needs |

---

# **8. Building a RAG Chain — End-to-End**

---

## **8.1 RAG Pipeline Architecture**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE — FULL FLOW                               │
│                                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐      │
│  │  1. LOAD   │──▶│  2. SPLIT  │──▶│ 3. EMBED   │──▶│  4. STORE  │      │
│  │  Documents │   │  Into      │   │  With      │   │  In Vector │      │
│  │  (PDF,web) │   │  Chunks    │   │  Model     │   │  Database  │      │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘      │
│                                                                          │
│                      INGESTION (offline, batch)                          │
│  ════════════════════════════════════════════════════════════════════     │
│                      RETRIEVAL (online, per query)                       │
│                                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐      │
│  │  5. QUERY  │──▶│ 6.RETRIEVE │──▶│ 7.AUGMENT  │──▶│8. GENERATE │      │
│  │  User asks │   │  Top-k     │   │  Inject    │   │  LLM makes │      │
│  │  question  │   │  similar   │   │  into      │   │  answer    │      │
│  │            │   │  chunks    │   │  prompt    │   │  + sources │      │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **8.2 Complete RAG Implementation**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Step 1: Load Documents ─────────────────────────────────────────────
loader = PyPDFLoader("ml_textbook.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ─── Step 2: Split into Chunks ──────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ─── Step 3: Create Embeddings and Store ─────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# ─── Step 4: Create Retriever ───────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # retrieve top 4 chunks
)

# ─── Step 5: Build RAG Chain with LCEL ──────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable ML assistant. Answer the question
    based ONLY on the provided context. If the context doesn't contain
    enough information, say so. Always cite your sources.

    Context:
    {context}"""),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ─── Step 6: Query ──────────────────────────────────────────────────────
answer = rag_chain.invoke("What is the vanishing gradient problem?")
print(answer)

# With streaming
for chunk in rag_chain.stream("Explain batch normalization"):
    print(chunk, end="", flush=True)
```

---

## **8.3 Advanced RAG Techniques**

```python
# 1. Multi-Query Retriever — generates multiple query perspectives
from langchain.retrievers import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(temperature=0)
)
# Generates 3 query variations → retrieves from each → deduplicates

# 2. Contextual Compression — re-ranks and filters retrieved chunks
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# 3. Ensemble Retriever — combines keyword + semantic search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 4

ensemble = EnsembleRetriever(
    retrievers=[bm25, vectorstore.as_retriever()],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)
```

---

# **9. Building an Agent**

---

## **9.1 What Is an Agent?**

An agent is an LLM that **decides** which actions to take — selecting tools, calling them, observing results, and deciding next steps — rather than following a fixed chain.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CHAIN vs AGENT                                         │
│                                                                          │
│   Chain (deterministic):                Agent (autonomous):              │
│   ─────────────────────                  ──────────────────              │
│   Step 1 → Step 2 → Step 3              ┌───────────┐                   │
│   (always the same path)                 │   Think   │                   │
│                                          └─────┬─────┘                   │
│                                                │                         │
│                                          ┌─────┴─────┐                   │
│                                          │   Act     │ → call tool       │
│                                          └─────┬─────┘                   │
│                                                │                         │
│                                          ┌─────┴─────┐                   │
│                                          │  Observe  │ ← read result     │
│                                          └─────┬─────┘                   │
│                                                │                         │
│                                          Done? ──No──▶ loop back         │
│                                           │                              │
│                                          Yes → final answer              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **9.2 The ReAct Pattern**

ReAct (Reasoning + Acting) interleaves **thinking** and **tool use**:

```
User: "What was Apple's revenue last quarter compared to Google's?"

Agent Thought: I need to search for Apple's latest quarterly revenue.
Agent Action: search_web("Apple Q3 2024 quarterly revenue")
Observation: Apple reported $85.8 billion in revenue for Q3 2024.

Agent Thought: Now I need Google's revenue for the same quarter.
Agent Action: search_web("Google Alphabet Q3 2024 quarterly revenue")
Observation: Alphabet reported $88.3 billion in revenue for Q3 2024.

Agent Thought: I have both numbers. Let me compare them.
Agent Action: calculate("88.3 - 85.8")
Observation: 2.5

Agent Thought: I can now give a comprehensive answer.
Final Answer: In Q3 2024, Google (Alphabet) reported $88.3B in revenue,
while Apple reported $85.8B — Google's revenue was $2.5B higher.
```

---

## **9.3 Building an Agent with LangChain**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # In production: use Tavily, SerpAPI, etc.
    return f"Search results for '{query}': [simulated results]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Input should be a valid Python expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create agent (using LangGraph's prebuilt ReAct agent)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_web, calculate, get_current_date]

agent = create_react_agent(llm, tools)

# Run agent
response = agent.invoke({
    "messages": [HumanMessage(content="What is 15% of today's S&P 500 value?")]
})

# Stream agent actions
for event in agent.stream({
    "messages": [HumanMessage(content="Compare Python and Rust performance")]
}):
    for value in event.values():
        if "messages" in value:
            for msg in value["messages"]:
                print(f"[{msg.type}]: {msg.content[:200]}")
```

---

## **9.4 Function Calling (Tool Calls)**

Modern agents use **function calling** (structured tool invocation) rather than parsing text output:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind tools for function calling
llm_with_tools = llm.bind_tools([search_web, calculate])

# The model returns structured tool_calls
response = llm_with_tools.invoke([
    HumanMessage(content="What is the square root of 144?")
])

print(response.tool_calls)
# [{'name': 'calculate', 'args': {'expression': 'import math; math.sqrt(144)'}, 'id': 'call_abc123'}]

# Execute the tool call
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_map = {"calculate": calculate, "search_web": search_web}
    result = tool_map[tool_call["name"]].invoke(tool_call["args"])
    print(f"Tool result: {result}")
```

---

# **10. What Is LangGraph?**

---

## **10.1 Definition**

LangGraph is a **graph-based orchestration framework** built on top of LangChain that enables building **stateful, multi-step, cyclic** agent workflows. While LangChain chains are linear pipelines, LangGraph lets you define complex control flows with **branching, looping, parallel execution, and human-in-the-loop** patterns.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH — THE CORE INSIGHT                            │
│                                                                          │
│  LangChain Chain:                LangGraph:                              │
│  ───────────────                  ─────────                              │
│  A → B → C → D                   A ──→ B ──→ C                          │
│  (linear, no loops)                │    ↑    │                           │
│                                    │    │    ▼                           │
│                                    ▼    │    D ──→ END                   │
│                                    E ───┘                                │
│                                   (cycles, branches,                     │
│                                    conditional routing)                  │
│                                                                          │
│  Think of it as:                                                         │
│    • LangChain = railroad track (fixed path)                             │
│    • LangGraph = road network (choose direction at each intersection)    │
└──────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** LangGraph exists because real-world agents need **cycles** (retry a step), **conditional branching** (route based on output), and **persistent state** (track progress). Linear chains can't express these patterns.

---

## **10.2 Why LangGraph?**

| Problem with Chains | LangGraph Solution |
|---------------------|--------------------|
| No loops — can't retry or iterate | Cycles: nodes can revisit previous nodes |
| Linear execution only | Conditional edges: route based on state |
| No shared state between steps | TypedDict state accessible to all nodes |
| No persistence across runs | Built-in checkpointing and memory |
| No human-in-the-loop | Interrupt nodes for human approval |
| Hard to debug complex flows | Graph visualization and LangSmith integration |

---

# **11. LangGraph Core Concepts**

---

## **11.1 The Building Blocks**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH BUILDING BLOCKS                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ StateGraph                                                       │    │
│  │  The container that holds the entire workflow definition.        │    │
│  │                                                                  │    │
│  │  ┌──────┐        ┌──────┐        ┌──────┐                       │    │
│  │  │ Node │──edge──│ Node │──edge──│ Node │                       │    │
│  │  │  A   │        │  B   │   │    │  C   │                       │    │
│  │  └──────┘        └──────┘   │    └──────┘                       │    │
│  │                             │                                    │    │
│  │                    conditional                                   │    │
│  │                      edge ──────▶ ┌──────┐                      │    │
│  │                                   │ Node │                      │    │
│  │                                   │  D   │                      │    │
│  │                                   └──────┘                      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  State:    TypedDict — shared data all nodes can read/write              │
│  Nodes:    Python functions — do the work (call LLM, tools, logic)       │
│  Edges:    Connections — define execution order                          │
│  Cond. Edges: Functions that return next node name based on state        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **11.2 Nodes**

Nodes are **Python functions** that receive the current state and return state updates.

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only list
    next_step: str

# Define nodes (functions that modify state)
def call_llm(state: AgentState) -> dict:
    """Node that calls the LLM."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def process_result(state: AgentState) -> dict:
    """Node that processes the LLM response."""
    last_message = state["messages"][-1]
    if "FINAL" in last_message.content:
        return {"next_step": "end"}
    return {"next_step": "continue"}
```

---

## **11.3 Edges and Conditional Edges**

```python
from langgraph.graph import StateGraph, START, END

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("llm", call_llm)
graph.add_node("process", process_result)

# Add edges
graph.add_edge(START, "llm")          # START → llm (always)
graph.add_edge("llm", "process")      # llm → process (always)

# Conditional edge: decides next node based on state
def should_continue(state: AgentState) -> str:
    if state.get("next_step") == "end":
        return "end"
    return "llm"  # loop back

graph.add_conditional_edges(
    "process",                          # from this node
    should_continue,                    # routing function
    {
        "end": END,                     # if returns "end" → END
        "llm": "llm"                   # if returns "llm" → loop back
    }
)

# Compile and run
app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Solve this step by step: 2x + 5 = 15")],
    "next_step": ""
})
```

---

## **11.4 Graph Visualization**

```python
# Generate a visual representation of the graph
from IPython.display import Image

Image(app.get_graph().draw_mermaid_png())

# Or get the Mermaid diagram string
print(app.get_graph().draw_mermaid())
```

---

# **12. LangGraph State Management**

---

## **12.1 TypedDict State**

State is defined as a **TypedDict** — a typed dictionary that all nodes share.

```python
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator

# Basic state
class SimpleState(TypedDict):
    query: str
    result: str

# Rich agent state with reducers
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # append reducer
    documents: list[str]
    current_step: str
    error_count: int
    final_answer: Optional[str]
```

---

## **12.2 Reducers**

Reducers define **how state updates are merged** — critical for understanding LangGraph state management.

```python
import operator
from typing import Annotated

class State(TypedDict):
    # APPEND reducer: new values are ADDED to the list
    messages: Annotated[list, operator.add]

    # OVERWRITE (default): new value REPLACES the old
    current_step: str

    # Custom reducer: define your own merge logic
    error_count: Annotated[int, lambda old, new: old + new]
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       REDUCERS EXPLAINED                                 │
│                                                                          │
│  OVERWRITE (default):                                                    │
│    State: {"step": "A"}                                                  │
│    Update: {"step": "B"}                                                 │
│    Result: {"step": "B"}          ← new replaces old                     │
│                                                                          │
│  APPEND (operator.add):                                                  │
│    State: {"messages": [msg1, msg2]}                                     │
│    Update: {"messages": [msg3]}                                          │
│    Result: {"messages": [msg1, msg2, msg3]}   ← appended                 │
│                                                                          │
│  CUSTOM (lambda):                                                        │
│    State: {"errors": 2}                                                  │
│    Update: {"errors": 1}                                                 │
│    Result: {"errors": 3}          ← custom logic (sum)                   │
└──────────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** The `messages: Annotated[list, operator.add]` pattern is one of the most common things you'll see in LangGraph. It ensures every node **appends** new messages to the conversation history rather than overwriting it.

---

## **12.3 Checkpointing (Persistent Memory)**

LangGraph can **save and restore state** across runs — enabling long-running conversations, resumable workflows, and fault tolerance.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# In-memory checkpointer (for development)
checkpointer = MemorySaver()

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools)
graph.add_edge("tools", "agent")

# Compile WITH checkpointer
app = graph.compile(checkpointer=checkpointer)

# Run with a thread_id — state persists across calls
config = {"configurable": {"thread_id": "user-123-session-1"}}

# First interaction
result1 = app.invoke(
    {"messages": [HumanMessage(content="My name is Rahul")]},
    config=config
)

# Second interaction — REMEMBERS the first!
result2 = app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)
# The agent remembers "Rahul" because state was checkpointed

# Production: use persistent checkpointers
# from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.sqlite import SqliteSaver
# checkpointer = PostgresSaver(conn_string="postgresql://...")
```

---

## **12.4 Human-in-the-Loop**

LangGraph can **pause execution** and wait for human approval before proceeding.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

# Compile with interrupt BEFORE the execute node
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"]  # pause here for human review
)

config = {"configurable": {"thread_id": "review-session"}}

# Run — will pause before "execute"
result = app.invoke(
    {"messages": [HumanMessage(content="Delete all test records")]},
    config=config
)
# Execution pauses here — human reviews the plan

# After human approval, resume execution
result = app.invoke(None, config=config)  # continues from checkpoint
```

---

# **13. Building a LangGraph Agent — Full Example**

---

## **13.1 Complete ReAct Agent with LangGraph**

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import operator

# ─── 1. Define State ────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# ─── 2. Define Tools ────────────────────────────────────────────────────
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': LangGraph is a framework by LangChain for building stateful agents."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

tools = [search, calculate]

# ─── 3. Define Nodes ────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """The agent reasons and decides whether to use tools."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Use LangGraph's prebuilt ToolNode for tool execution
tool_node = ToolNode(tools)

# ─── 4. Define Routing Logic ────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Decide whether to use tools or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ─── 5. Build the Graph ─────────────────────────────────────────────────
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")  # after tools, go back to agent

# ─── 6. Compile ─────────────────────────────────────────────────────────
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ─── 7. Visualize ───────────────────────────────────────────────────────
# app.get_graph().draw_mermaid()
# Produces:
#   START → agent → {tools → agent (loop)} or → END

# ─── 8. Run ─────────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": "demo-session"}}

# Turn 1
result = app.invoke(
    {"messages": [HumanMessage(content="Search for what LangGraph is, then calculate 42 * 17")]},
    config=config
)

for msg in result["messages"]:
    print(f"[{msg.type}]: {msg.content[:150] if msg.content else '[tool_call]'}")
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│              REACT AGENT GRAPH — EXECUTION FLOW                          │
│                                                                          │
│                    ┌─────────┐                                           │
│                    │  START  │                                            │
│                    └────┬────┘                                            │
│                         │                                                │
│                         ▼                                                │
│                    ┌─────────┐                                            │
│             ┌─────│  agent   │─────┐                                     │
│             │     │ (reason) │     │                                      │
│             │     └─────────┘     │                                      │
│             │                     │                                      │
│        has tool_calls        no tool_calls                               │
│             │                     │                                      │
│             ▼                     ▼                                      │
│        ┌─────────┐          ┌─────────┐                                  │
│        │  tools  │          │   END   │                                  │
│        │(execute)│          └─────────┘                                  │
│        └────┬────┘                                                       │
│             │                                                            │
│             └───────────▶ back to agent (loop)                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **13.2 Advanced: Parallel Execution and Branching**

```python
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict):
    query: str
    web_results: str
    db_results: str
    final_report: str

def search_web(state):
    return {"web_results": f"Web results for: {state['query']}"}

def search_database(state):
    return {"db_results": f"DB results for: {state['query']}"}

def synthesize(state):
    report = f"Report combining:\n- Web: {state['web_results']}\n- DB: {state['db_results']}"
    return {"final_report": report}

# Build graph with parallel branches
graph = StateGraph(ResearchState)
graph.add_node("web_search", search_web)
graph.add_node("db_search", search_database)
graph.add_node("synthesize", synthesize)

# Parallel: START fans out to both search nodes
graph.add_edge(START, "web_search")
graph.add_edge(START, "db_search")

# Both converge into synthesize
graph.add_edge("web_search", "synthesize")
graph.add_edge("db_search", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()
result = app.invoke({"query": "LangGraph best practices"})
```

---

## **13.3 Error Handling and Retry**

```python
class RobustState(TypedDict):
    messages: Annotated[list, operator.add]
    retry_count: int
    error: Optional[str]

MAX_RETRIES = 3

def agent_with_retry(state: RobustState) -> dict:
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "error": None}
    except Exception as e:
        return {
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }

def should_retry(state: RobustState) -> str:
    if state.get("error") and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    elif state.get("error"):
        return "fail"
    elif hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
        return "tools"
    return "end"

graph = StateGraph(RobustState)
graph.add_node("agent", agent_with_retry)
graph.add_node("tools", tool_node)
graph.add_node("error_handler", lambda s: {"messages": [AIMessage(content=f"Error: {s['error']}")]})

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_retry, {
    "tools": "tools",
    "retry": "agent",        # loop back to retry
    "fail": "error_handler",
    "end": END
})
graph.add_edge("tools", "agent")
graph.add_edge("error_handler", END)
```

---

# **14. LangChain vs LangGraph**

---

## **14.1 Comparison Table**

| Dimension | LangChain (Chains/LCEL) | LangGraph |
|-----------|-------------------------|-----------|
| **Execution model** | Linear pipeline (DAG) | Graph with cycles |
| **Control flow** | Sequential, parallel (RunnableParallel) | Branching, looping, conditional |
| **State management** | Passed through pipe; optional memory | First-class TypedDict state with reducers |
| **Persistence** | Manual (external DB) | Built-in checkpointing |
| **Human-in-the-loop** | Not natively supported | `interrupt_before`/`interrupt_after` |
| **Error handling** | `with_retry()`, `with_fallbacks()` | Conditional edges for retry logic |
| **Visualization** | Limited | Graph visualization (Mermaid) |
| **Complexity** | Simpler to learn | More powerful, steeper learning curve |
| **Use case** | Simple chains, RAG, sequential pipelines | Agents, multi-step workflows, complex orchestration |
| **Streaming** | Token-level streaming | Node-level + token-level streaming |
| **Debugging** | LangSmith traces | LangSmith traces + graph visualization |

---

## **14.2 When to Use Which**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  DECISION FRAMEWORK                                      │
│                                                                          │
│  Is your workflow a simple pipeline?                                     │
│    YES → Use LangChain / LCEL                                            │
│      Examples:                                                           │
│        • prompt → LLM → parser                                           │
│        • RAG: retrieve → format → generate                               │
│        • Summarization chain                                             │
│        • Classification pipeline                                         │
│                                                                          │
│  Does your workflow need any of these?                                   │
│    • Loops / retries                                                     │
│    • Conditional branching (dynamic routing)                             │
│    • Persistent state across interactions                                │
│    • Human-in-the-loop approval                                          │
│    • Multi-agent collaboration                                           │
│    • Complex error recovery                                              │
│    YES → Use LangGraph                                                   │
│      Examples:                                                           │
│        • ReAct agent with tool calling                                   │
│        • Multi-step research with iterative refinement                   │
│        • Workflow with human approval gates                              │
│        • Multi-agent system (planner + executor + reviewer)              │
│                                                                          │
│  Rule of thumb:                                                          │
│    Start with LCEL. Upgrade to LangGraph when you need cycles or state.  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **14.3 Can They Work Together?**

**Yes — and they should.** LangGraph nodes often contain LCEL chains internally.

```python
from langgraph.graph import StateGraph, START, END

# LCEL chain used INSIDE a LangGraph node
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

def rag_node(state):
    """LangGraph node that runs an LCEL chain internally."""
    answer = rag_chain.invoke(state["question"])
    return {"answer": answer}

def review_node(state):
    """Quality check on the RAG answer."""
    if len(state["answer"]) < 50:
        return {"needs_retry": True}
    return {"needs_retry": False}

# LangGraph adds the orchestration layer
graph = StateGraph(state_schema)
graph.add_node("rag", rag_node)        # LCEL chain inside
graph.add_node("review", review_node)   # quality check
graph.add_edge(START, "rag")
graph.add_edge("rag", "review")
graph.add_conditional_edges("review", retry_logic)
```

---

# **15. Production Patterns**

---

## **15.1 Streaming Responses**

```python
# LCEL streaming (token-by-token)
chain = prompt | llm | StrOutputParser()

async def stream_response(question: str):
    async for chunk in chain.astream({"question": question}):
        yield chunk  # send to client (e.g., via SSE or WebSocket)

# LangGraph streaming (node-by-node + tokens)
async for event in app.astream_events(
    {"messages": [HumanMessage(content="Explain transformers")]},
    config=config,
    version="v2"
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        # Token-level streaming from LLM
        print(event["data"]["chunk"].content, end="", flush=True)
    elif kind == "on_chain_end":
        # Node completed
        print(f"\n[Node completed: {event['name']}]")
```

---

## **15.2 LangSmith — Observability and Tracing**

```python
import os

# Enable LangSmith tracing (set environment variables)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

# Every chain/agent call is now automatically traced
# View traces at: https://smith.langchain.com

# Manual tracing with run names
result = chain.invoke(
    {"question": "What is attention?"},
    config={"run_name": "rag-query-attention"}
)

# Add metadata and tags for filtering
result = chain.invoke(
    {"question": "What is attention?"},
    config={
        "metadata": {"user_id": "rahul", "session": "interview-prep"},
        "tags": ["rag", "production"]
    }
)
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LANGSMITH TRACE VIEW                                   │
│                                                                          │
│  rag_chain                                    Total: 2.3s, $0.004       │
│  ├── RunnableParallel                         0.8s                       │
│  │   ├── retriever (context)                  0.7s                       │
│  │   │   └── FAISS.similarity_search          0.7s → 4 docs             │
│  │   └── RunnablePassthrough (question)       0.0s                       │
│  ├── ChatPromptTemplate                       0.0s                       │
│  ├── ChatOpenAI (gpt-4o)                      1.4s, 847 tokens          │
│  └── StrOutputParser                          0.0s                       │
│                                                                          │
│  LangSmith shows latency, cost, tokens, inputs/outputs for every step.  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **15.3 Cost Management**

```python
from langchain_community.callbacks import get_openai_callback

# Track token usage and cost
with get_openai_callback() as cb:
    result = chain.invoke({"question": "Explain attention"})
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")

# Strategies for cost reduction
# 1. Use cheaper models for simple tasks
cheap_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 10-20x cheaper
expensive_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Cache responses
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())  # identical queries return cached results

# 3. Limit retrieval tokens
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # fewer chunks = less cost

# 4. Use max_tokens to bound output
llm = ChatOpenAI(model="gpt-4o", max_tokens=500)
```

---

## **15.4 Rate Limiting**

```python
# Rate limiting with max_concurrency
results = chain.batch(
    [{"q": f"Question {i}"} for i in range(100)],
    config={"max_concurrency": 5}  # max 5 parallel API calls
)

# Custom rate limiter
import asyncio
from langchain_core.runnables import RunnableLambda

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.semaphore = asyncio.Semaphore(calls_per_minute)
        self.delay = 60.0 / calls_per_minute

    async def __call__(self, input_data):
        async with self.semaphore:
            await asyncio.sleep(self.delay)
            return input_data

rate_limited_chain = RunnableLambda(RateLimiter(30)) | chain
```

---

## **15.5 Error Handling Patterns**

```python
# 1. Fallbacks — graceful model degradation
chain = (
    prompt
    | ChatOpenAI(model="gpt-4o").with_fallbacks([
        ChatOpenAI(model="gpt-4o-mini"),
        ChatOpenAI(model="gpt-3.5-turbo")
    ])
    | StrOutputParser()
)

# 2. Retry with exponential backoff
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1,
    retry_if_exception_type=(Exception,)
)

# 3. Structured error handling in chains
from langchain_core.runnables import RunnableLambda

def safe_invoke(input_data):
    try:
        return chain.invoke(input_data)
    except Exception as e:
        return {"error": str(e), "fallback": "Unable to process this request."}

safe_chain = RunnableLambda(safe_invoke)
```

---

# **16. Common Interview Questions with Strong Answers**

---

## **Q1: "What is LangChain and when would you use it?"**

> **Strong answer:**
>
> LangChain is an open-source framework for building applications powered by LLMs. It provides modular, composable abstractions — prompt templates, output parsers, memory, tools, retrievers — that you compose into pipelines using LCEL, the LangChain Expression Language.
>
> I'd use it when I need to go beyond simple LLM API calls: building RAG systems that connect LLMs to private data, creating chatbots with conversation memory, or building tool-using agents. The real value is that LCEL chains automatically get streaming, batching, async, retries, and observability via LangSmith — so you write it once and get production features for free.
>
> In my projects, I've used LangChain for building career recommendation systems with RAG retrieval and for creating multi-tool agents that search the web and query databases.

---

## **Q2: "Explain LangChain vs LangGraph. When would you choose each?"**

> **Strong answer:**
>
> LangChain's LCEL is designed for **linear pipelines** — prompt goes in, flows through components, answer comes out. Think of it like a railroad track: efficient, predictable, but no loops or detours.
>
> LangGraph is for **graph-based workflows** with cycles, conditional branching, and persistent state. Think of it like a road network where you choose direction at each intersection.
>
> **I start with LCEL** for straightforward tasks: RAG chains, summarization, classification. **I upgrade to LangGraph** when I need: (1) **loops** — the agent retries or iterates, (2) **conditional routing** — different paths based on LLM output, (3) **persistent state** — checkpointing across sessions, or (4) **human-in-the-loop** — pausing for approval.
>
> They also work together beautifully — LangGraph nodes often contain LCEL chains internally. LCEL handles the data flow *within* a step; LangGraph orchestrates *between* steps.

---

## **Q3: "How would you build a RAG system with LangChain?"**

> **Strong answer:**
>
> I'd follow the standard RAG pipeline — load, split, embed, store, retrieve, generate — using LangChain's LCEL for the chain composition.
>
> 1. **Load** documents using `PyPDFLoader`, `WebBaseLoader`, or `DirectoryLoader` depending on the source.
> 2. **Split** into chunks with `RecursiveCharacterTextSplitter` — I typically use 1000-character chunks with 200-character overlap to preserve context at boundaries.
> 3. **Embed** using `OpenAIEmbeddings` (text-embedding-3-small for cost efficiency) and store in **FAISS** for development or **Pinecone/Weaviate** for production.
> 4. **Retrieve** using the vectorstore's `as_retriever()` with top-k=4. For better quality, I'd add a `MultiQueryRetriever` to generate query variations or an `EnsembleRetriever` combining BM25 keyword search with semantic search.
> 5. **Generate** with an LCEL chain: `{context: retriever | format_docs, question: RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
>
> For production, I'd add LangSmith tracing, response caching, fallback models, and source citation in the prompt template.

---

## **Q4: "What is LCEL and why does it matter?"**

> **Strong answer:**
>
> LCEL — LangChain Expression Language — is the declarative syntax for composing LLM pipelines using the pipe operator (`|`). Instead of writing imperative code with manual function calls, you declare: `chain = prompt | llm | parser`.
>
> The reason it matters goes beyond syntax. Every LCEL chain is a **Runnable** that automatically supports `.invoke()`, `.batch()`, `.stream()`, `.ainvoke()`, `.astream()` — plus retries, fallbacks, and LangSmith tracing. You write the pipeline once and get six execution modes for free.
>
> Key Runnables: `RunnableParallel` for concurrent branches, `RunnableLambda` for wrapping custom functions, `RunnablePassthrough` for forwarding input alongside transformations, and `RunnableBranch` for conditional routing.
>
> The classic example is the RAG pattern: `{context: retriever | format_docs, question: RunnablePassthrough()} | prompt | llm | parser` — this retrieves documents in parallel with passing the question through, then formats everything into a prompt.

---

## **Q5: "How do you handle state in LangGraph?"**

> **Strong answer:**
>
> LangGraph uses a **TypedDict** as the shared state that all nodes can read and update. The key innovation is **reducers** — annotations that define how state updates are merged.
>
> For example, `messages: Annotated[list, operator.add]` means new messages are *appended* to the list, not overwritten. Without a reducer, fields are overwritten by default. You can also write custom reducers for things like incrementing counters.
>
> For **persistence**, LangGraph has built-in **checkpointing**. You compile the graph with a `MemorySaver` (development) or `PostgresSaver` (production), and every state transition is automatically saved. Using a `thread_id` in the config, you can resume conversations across sessions — the agent remembers everything.
>
> For **human-in-the-loop**, you use `interrupt_before=["node_name"]` when compiling. The graph pauses before that node, saves state to the checkpoint, and waits. After human review, you resume with `app.invoke(None, config)` and it continues from exactly where it stopped.

---

## **Q6: "How would you handle errors and retries in a LangGraph agent?"**

> **Strong answer:**
>
> In LangGraph, error handling is structural — you encode retry logic directly into the graph topology using conditional edges.
>
> My pattern: I track `retry_count` and `error` in the state. The agent node wraps its LLM call in a try/except, incrementing retry_count on failure. A conditional edge checks: if there's an error and retries < MAX_RETRIES, route back to the agent node (loop). If retries are exhausted, route to an error handler node that returns a graceful fallback message. If no error and no tool calls, route to END.
>
> For tool-level errors, the ToolNode can be wrapped to catch exceptions and return error messages as ToolMessages, which the agent sees and can adapt to.
>
> At the LLM level, I also use `.with_fallbacks()` to cascade from GPT-4o to GPT-4o-mini if the primary model is unavailable, and `.with_retry()` for transient API errors with exponential backoff. These compose with LangGraph's graph-level retry.

---

## **Q7: "What is the ReAct pattern and how do you implement it?"**

> **Strong answer:**
>
> ReAct stands for **Reasoning + Acting** — the agent interleaves thinking steps with tool use. At each turn, it: (1) reasons about what to do next, (2) selects and calls a tool, (3) observes the result, and (4) decides whether to continue or provide a final answer.
>
> In LangGraph, I implement this as a two-node cycle: an **agent node** that calls an LLM with bound tools, and a **tools node** that executes tool calls. The conditional edge checks if the agent's response contains `tool_calls` — if yes, route to the tools node; if no, route to END. After tools execute, the edge loops back to the agent.
>
> LangGraph provides `create_react_agent(llm, tools)` as a prebuilt implementation, but understanding the underlying graph is important for customization — adding guardrails, human approval before dangerous tool calls, or parallel tool execution.

---

## **Q8: "How do you choose between different memory types in LangChain?"**

> **Strong answer:**
>
> The choice depends on conversation length and retrieval pattern:
>
> For **short conversations** (<20 turns), `ConversationBufferMemory` works — just store everything. Simple and complete, but token cost grows linearly.
>
> For **long conversations**, `ConversationSummaryBufferMemory` is my go-to — it keeps recent messages verbatim and progressively summarizes older ones. You get the detail of recent context with bounded token cost.
>
> For **topic-switching conversations** where the user might revisit something from 50 messages ago, `VectorStoreMemory` is ideal — it embeds all messages and retrieves semantically relevant ones, not just recent ones.
>
> In LangGraph, I prefer using the **built-in checkpointing** over LangChain memory modules. The checkpointer persists the entire state (including message history) automatically, and I can control what goes into the LLM's context by manipulating the messages list in the state — trimming old messages, summarizing, or filtering by relevance.

---

# **17. Key Takeaways**

---

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    KEY TAKEAWAYS — INTERVIEW READY                       │
│                                                                          │
│  1. LangChain is a FRAMEWORK that connects LLMs to memory, tools,       │
│     and data. LCEL is its declarative composition syntax.                │
│                                                                          │
│  2. LCEL chains are RUNNABLES — every pipe gives you invoke, stream,     │
│     batch, async, retries, and tracing for free.                         │
│                                                                          │
│  3. RAG = Load → Split → Embed → Store → Retrieve → Generate.           │
│     Use RecursiveCharacterTextSplitter + FAISS + LCEL chain.             │
│                                                                          │
│  4. LangGraph extends LangChain with GRAPH-BASED orchestration —         │
│     cycles, conditional edges, persistent state, human-in-the-loop.     │
│                                                                          │
│  5. Use LCEL for LINEAR pipelines (RAG, summarization, classification).  │
│     Use LangGraph when you need LOOPS, BRANCHES, or STATE.               │
│                                                                          │
│  6. LangGraph state = TypedDict + reducers. The Annotated[list,          │
│     operator.add] pattern for messages is the most common pattern.       │
│                                                                          │
│  7. Checkpointing enables PERSISTENCE (resume conversations) and         │
│     HUMAN-IN-THE-LOOP (interrupt_before / interrupt_after).              │
│                                                                          │
│  8. ReAct agents = agent node + tools node + conditional edge loop.      │
│     LangGraph's create_react_agent() gives you this prebuilt.            │
│                                                                          │
│  9. Production requires: LangSmith tracing, fallback models, caching,    │
│     rate limiting, cost tracking, and structured error handling.          │
│                                                                          │
│  10. They work TOGETHER: LangGraph nodes contain LCEL chains.            │
│      LCEL handles flow within a step; LangGraph orchestrates between.    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

**Prepared for:** Data Scientist / ML Engineer interviews
**Framework versions:** LangChain 0.3+, LangGraph 0.2+
**Last reviewed:** February 2026
