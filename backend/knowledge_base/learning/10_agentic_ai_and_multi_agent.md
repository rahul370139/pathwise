# Agentic AI & Multi-Agent Systems — Interview Preparation Guide

**Candidate:** Rahul Sharma | **Experience:** 4+ years | **Education:** MS Data Science, UMD
**Focus:** Data Scientist / ML Engineer — LangChain, LangGraph, MCP, FastAPI, Multi-Agent Orchestration

---

# Table of Contents

1. [What Is Agentic AI](#1-what-is-agentic-ai)
2. [Single Agent Architecture](#2-single-agent-architecture)
3. [Agent Frameworks — Deep Comparison](#3-agent-frameworks--deep-comparison)
4. [Multi-Agent System Design](#4-multi-agent-system-design)
5. [Agentic Patterns](#5-agentic-patterns)
6. [Memory in Agents](#6-memory-in-agents)
7. [Tool Use and Function Calling](#7-tool-use-and-function-calling)
8. [LLM Selection for Agents](#8-llm-selection-for-agents)
9. [Deployment Stack](#9-deployment-stack)
10. [Production Considerations](#10-production-considerations)
11. [Common Interview Questions](#11-common-interview-questions-with-strong-answers)
12. [Key Takeaways](#12-key-takeaways)

---

# **1. What Is Agentic AI**

---

## **1.1 Definition**

An **AI agent** is a system that uses an LLM as its core reasoning engine to **autonomously plan, decide, and execute** multi-step tasks — perceiving its environment, selecting tools, taking actions, and reflecting on outcomes without explicit step-by-step human instruction.

```
┌───────────────────────────────────────────────────────────────────────┐
│                        AGENTIC AI — CORE LOOP                         │
│                                                                       │
│         ┌──────────┐                                                  │
│         │ PERCEIVE │ ◄─── Observe environment, read inputs,           │
│         └────┬─────┘      parse user query, gather context            │
│              │                                                        │
│              ▼                                                        │
│         ┌──────────┐                                                  │
│         │   PLAN   │ ◄─── Decompose task into sub-goals,              │
│         └────┬─────┘      decide strategy, select tools               │
│              │                                                        │
│              ▼                                                        │
│         ┌──────────┐                                                  │
│         │   ACT    │ ◄─── Execute tool calls, API requests,           │
│         └────┬─────┘      code execution, web searches                │
│              │                                                        │
│              ▼                                                        │
│         ┌──────────┐                                                  │
│         │ REFLECT  │ ◄─── Evaluate results, check quality,            │
│         └────┬─────┘      decide: done? retry? adjust plan?           │
│              │                                                        │
│              ▼                                                        │
│         ┌──────────┐                                                  │
│         │  OUTPUT  │ ◄─── Return final answer, summary, or artifact   │
│         └──────────┘                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## **1.2 What Makes AI "Agentic"**

Not every LLM call is agentic. The spectrum:

```
Simple Prompt → Chain-of-Thought → Tool Use → Single Agent → Multi-Agent
───────────────────────────────────────────────────────────────────────►
                        Increasing Autonomy
```

| Capability | Non-Agentic LLM | Agentic AI |
|---|---|---|
| **Decision-Making** | User decides every step | Agent decides next action |
| **Tool Use** | None | Calls APIs, DBs, code executors |
| **Memory** | Stateless (per call) | Maintains state across steps |
| **Error Recovery** | Fails silently or hallucinates | Detects errors, retries, adapts |
| **Multi-Step** | Single prompt → single response | Plans and executes chains of actions |
| **Autonomy** | Human-in-the-loop always | Can operate with minimal supervision |

---

## **1.3 Autonomous Decision-Making**

Agents make decisions at every step of execution:

```
User: "Analyze Q4 sales data and create a report with charts"

Agent Reasoning:
├── Step 1: PLAN    → I need to (a) load data, (b) analyze trends,
│                     (c) generate charts, (d) compile report
├── Step 2: TOOL    → Use SQL tool to query database
├── Step 3: ACT     → Execute pandas analysis on returned data
├── Step 4: TOOL    → Use matplotlib tool to generate charts
├── Step 5: REFLECT → Check: are charts meaningful? Data complete?
├── Step 6: ACT     → Compile findings into report format
└── Step 7: OUTPUT  → Return formatted report with charts
```

**Key insight:** The agent *chose* the tools, *decided* the execution order, and *validated* intermediate results — the user only specified the goal.

---

## **1.4 Tool Use — The Superpower of Agents**

Tools transform an LLM from a text-completion engine into an **action-capable system**:

| Tool Category | Examples | What It Enables |
|---|---|---|
| **Search** | Google, Bing, Tavily, SerpAPI | Real-time information access |
| **Code Execution** | Python REPL, Jupyter, sandboxes | Computation, data analysis |
| **Databases** | SQL executor, vector store queries | Structured data retrieval |
| **APIs** | REST calls, GraphQL, MCP servers | External service integration |
| **File I/O** | Read/write files, parse PDFs | Document processing |
| **Communication** | Email, Slack, webhooks | Notify and interact |
| **Browser** | Playwright, Selenium | Web scraping, form filling |

> **Interview tip:** When asked "What makes an agent different from a chatbot?", emphasize three things: **autonomy** (decides its own steps), **tool use** (takes real-world actions), and **memory** (maintains context across steps). A chatbot is reactive; an agent is proactive.

---

# **2. Single Agent Architecture**

---

## **2.1 The Perception → Planning → Action → Reflection Loop**

Every single agent follows this cognitive loop, inspired by how humans approach complex tasks:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    SINGLE AGENT ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        PERCEPTION LAYER                         │   │
│  │  • Parse user query         • Read conversation history         │   │
│  │  • Extract intent & entities • Fetch relevant memories          │   │
│  │  • Understand current context • Identify available tools        │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        PLANNING LAYER                           │   │
│  │  • Decompose task into sub-tasks                                │   │
│  │  • Determine execution order (sequential or parallel)           │   │
│  │  • Select tools for each sub-task                               │   │
│  │  • Estimate resource needs                                      │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         ACTION LAYER                            │   │
│  │  • Execute tool calls          • Parse tool responses           │   │
│  │  • Handle errors & retries     • Chain tool outputs             │   │
│  │  • Manage parallel execution   • Enforce timeouts               │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       REFLECTION LAYER                          │   │
│  │  • Evaluate action outcomes vs expected results                 │   │
│  │  • Assess quality / completeness                                │   │
│  │  • Decide: ✓ Done  |  ↺ Retry  |  ✎ Revise Plan               │   │
│  │  • Store learnings in memory for future tasks                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│                    ┌─────────┴──────────┐                              │
│                    │   Task Complete?    │                              │
│                    ├─── YES → Output ───┤                              │
│                    └─── NO → Loop Back ─┘                              │
│                              │                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         MEMORY STORE                            │   │
│  │  Short-Term │ Long-Term │ Episodic │ Working                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## **2.2 Detailed Breakdown of Each Layer**

### **Perception**

```python
# Perception: Extract structured intent from raw user input
from langchain_core.prompts import ChatPromptTemplate

perception_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's request and extract:
    1. Primary intent (e.g., analyze, generate, search, compare)
    2. Entities mentioned (e.g., datasets, topics, dates)
    3. Constraints (e.g., format, length, deadline)
    4. Required tools (e.g., SQL, web search, code execution)
    Return structured JSON."""),
    ("human", "{user_input}")
])
```

### **Planning**

```python
# Planning: Decompose into sub-tasks with tool assignments
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a task planner. Given the user's intent:
    1. Break into atomic sub-tasks
    2. Assign tools to each sub-task
    3. Determine execution order (sequential vs parallel)
    4. Identify dependencies between sub-tasks
    
    Return a JSON plan:
    {{"steps": [
        {{"id": 1, "task": "...", "tool": "...", "depends_on": []}},
        ...
    ]}}"""),
    ("human", "Intent: {intent}\nEntities: {entities}\nTools available: {tools}")
])
```

### **Action**

```python
# Action: Execute tools with error handling
import asyncio
from langchain.tools import Tool

async def execute_step(step: dict, tools: dict[str, Tool]):
    tool = tools[step["tool"]]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await tool.ainvoke(step["input"])
            return {"step_id": step["id"], "status": "success", "output": result}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"step_id": step["id"], "status": "failed", "error": str(e)}
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### **Reflection**

```python
# Reflection: Evaluate whether the action met expectations
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the action result:
    - Does it answer the sub-task?
    - Is the data complete and accurate?
    - Are there errors or missing information?
    - Should we retry, adjust the plan, or proceed?
    
    Respond with:
    {{"verdict": "pass|retry|revise", "reason": "...", "next_action": "..."}}"""),
    ("human", "Task: {task}\nExpected: {expected}\nActual Result: {result}")
])
```

---

## **2.3 Example: End-to-End Single Agent Execution**

```
User: "What were the top 3 trending AI papers this week? Summarize each."

PERCEIVE:
  Intent: search + summarize
  Entities: AI papers, this week, top 3
  Tools needed: web_search, text_summarizer

PLAN:
  Step 1: Search "trending AI papers this week" → web_search
  Step 2: Extract top 3 paper URLs → parse_results
  Step 3: For each paper, fetch abstract → web_fetch
  Step 4: Summarize each abstract → LLM summarization
  Step 5: Format as ranked list → output

ACT:
  Step 1: web_search("top trending AI papers February 2026") → 10 results
  Step 2: Filter to top 3 by citation/relevance → 3 URLs
  Step 3: Fetch abstracts for each → 3 abstracts
  Step 4: LLM summarizes each → 3 summaries

REFLECT:
  ✓ Got 3 papers? Yes
  ✓ Summaries coherent? Yes
  ✓ From this week? Verify dates... Yes
  → Verdict: PASS → Output final answer
```

---

# **3. Agent Frameworks — Deep Comparison**

---

## **3.1 LangChain**

**What it is:** The most widely adopted framework for building LLM-powered applications with tool integration, prompt orchestration, and memory management.

**Core Concepts:**

```
┌────────────────────────────────────────────────────────────────────┐
│                    LANGCHAIN ARCHITECTURE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌───────────┐  │
│  │  Models   │   │   Prompts    │   │  Chains  │   │  Agents   │  │
│  │ ChatGPT   │   │  Templates   │   │ LCEL     │   │ ReAct     │  │
│  │ Claude    │   │  Few-Shot    │   │ Sequential│  │ OpenAI    │  │
│  │ Mistral   │   │  Structured  │   │ Parallel │   │ Tool-Call │  │
│  └──────────┘   └──────────────┘   └──────────┘   └───────────┘  │
│                                                                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│  │  Memory   │   │    Tools     │   │     Retrievers           │   │
│  │ Buffer    │   │  Functions   │   │  Vector Stores           │   │
│  │ Summary   │   │  APIs        │   │  BM25, Hybrid            │   │
│  │ Vector    │   │  Custom      │   │  Multi-Query             │   │
│  └──────────┘   └──────────────┘   └──────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **LCEL (LangChain Expression Language):** Declarative chaining with `|` pipe operator
- **Tool Integration:** 100+ built-in tool wrappers, easy custom tool creation
- **Prompt Orchestration:** Templates, few-shot examples, structured output parsing
- **Memory:** Conversation buffer, summary, vector-backed, entity memory

**Code Example — LangChain Agent with Tools:**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Define custom tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Set up tools
search = TavilySearchResults(max_results=3)
tools = [search, calculate]

# Create agent with prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use tools when needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({"input": "What is the population of Tokyo times 2?"})
```

**When to use LangChain:**
- Rapid prototyping of LLM applications
- Simple sequential chains and single-agent tool use
- When you need extensive community tool integrations
- As the "model layer" before more complex orchestration

---

## **3.2 LangGraph**

**What it is:** A graph-based orchestration framework (built on LangChain) for building **stateful, multi-step, branching, and cyclical** agent workflows as directed graphs.

**Why LangGraph over LangChain Agents:**

| Feature | LangChain AgentExecutor | LangGraph |
|---|---|---|
| Execution model | Linear chain | Directed graph (cycles allowed) |
| Branching | Limited | Full conditional branching |
| Parallel execution | Not native | Native `Send()` API |
| State management | Implicit | Explicit `TypedDict` state |
| Human-in-the-loop | Manual | Built-in `interrupt()` |
| Persistence | External | Built-in checkpointing |
| Streaming | Basic | Token-level streaming |
| Subgraphs | No | Yes — composable graphs |

**Core Concepts:**

```
┌────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Nodes = Functions that transform state                            │
│  Edges = Transitions between nodes (conditional or fixed)          │
│  State = TypedDict passed through the graph                        │
│                                                                    │
│  ┌─────────┐   should_search?   ┌─────────────┐                   │
│  │  START   │──── YES ──────────▶│   search    │──┐               │
│  │ (router) │                    └─────────────┘  │               │
│  │         │                                      ▼               │
│  │         │    ┌─────────────┐   ┌─────────────────────┐         │
│  │         │────▶│  generate   │◀──│  process_results   │         │
│  │         │ NO  └──────┬──────┘   └─────────────────────┘         │
│  └─────────┘            │                                          │
│                         ▼                                          │
│                 ┌──────────────┐    good?    ┌──────────┐          │
│                 │   reflect    │──── YES ───▶│   END    │          │
│                 │   (grade)    │             └──────────┘          │
│                 └──────┬───────┘                                   │
│                        │ NO                                        │
│                        └──────── Loop back to search ──────────►   │
└────────────────────────────────────────────────────────────────────┘
```

**Code Example — LangGraph Stateful Agent:**

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: str
    final_answer: str
    iteration: int

# 2. Define Nodes
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher(state: AgentState) -> AgentState:
    """Search for information using tools."""
    query = state["messages"][-1].content
    # Simulate search
    results = f"Search results for: {query}"
    return {"search_results": results, "iteration": state.get("iteration", 0) + 1}

def writer(state: AgentState) -> AgentState:
    """Generate answer based on research."""
    response = llm.invoke([
        SystemMessage(content="Write a concise answer based on the search results."),
        HumanMessage(content=f"Results: {state['search_results']}")
    ])
    return {"final_answer": response.content}

def reflector(state: AgentState) -> AgentState:
    """Evaluate the quality of the answer."""
    evaluation = llm.invoke([
        SystemMessage(content="Grade this answer: is it complete and accurate? Reply PASS or FAIL."),
        HumanMessage(content=state["final_answer"])
    ])
    return {"messages": [HumanMessage(content=evaluation.content)]}

# 3. Define Conditional Edges
def should_continue(state: AgentState) -> Literal["researcher", "end"]:
    last_msg = state["messages"][-1].content
    if "PASS" in last_msg or state.get("iteration", 0) >= 3:
        return "end"
    return "researcher"

# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reflector", reflector)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reflector")
graph.add_conditional_edges("reflector", should_continue, {
    "researcher": "researcher",
    "end": END
})

# 5. Compile and Run
app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Explain quantum computing advances in 2026")],
    "search_results": "",
    "final_answer": "",
    "iteration": 0
})
```

**LangGraph Parallel Execution with `Send()`:**

```python
from langgraph.constants import Send

def route_to_specialists(state: AgentState):
    """Fan out to multiple specialist agents in parallel."""
    tasks = state["sub_tasks"]
    return [Send(task["specialist"], {"task": task}) for task in tasks]

graph.add_conditional_edges("planner", route_to_specialists)
```

**When to use LangGraph:**
- Complex workflows with branching/looping logic
- Multi-agent orchestration requiring shared state
- Human-in-the-loop approval workflows
- Production systems needing persistence and checkpointing
- Parallel tool execution and fan-out/fan-in patterns

---

## **3.3 CrewAI**

**What it is:** A framework for **role-based multi-agent collaboration** where you define a "crew" of agents, each with a distinct role, goal, and backstory, working together on tasks.

**Core Concepts:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CrewAI ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CREW (orchestrator)                                                │
│  ├── Agent: "Senior Researcher"                                     │
│  │   ├── Role: Research and gather data                             │
│  │   ├── Goal: Find comprehensive, accurate information             │
│  │   ├── Backstory: "You are a seasoned researcher with 10yr..."    │
│  │   └── Tools: [search, web_scraper]                               │
│  │                                                                  │
│  ├── Agent: "Data Analyst"                                          │
│  │   ├── Role: Analyze data and extract insights                    │
│  │   ├── Goal: Provide actionable analysis                          │
│  │   ├── Backstory: "Expert data scientist skilled in..."           │
│  │   └── Tools: [python_repl, sql_executor]                         │
│  │                                                                  │
│  └── Agent: "Report Writer"                                         │
│      ├── Role: Compile findings into reports                        │
│      ├── Goal: Create clear, professional reports                   │
│      ├── Backstory: "Technical writer who transforms data..."       │
│      └── Tools: [file_writer, chart_generator]                      │
│                                                                     │
│  PROCESS: Sequential | Hierarchical                                 │
│  Sequential: Researcher → Analyst → Writer                          │
│  Hierarchical: Manager assigns tasks dynamically                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Code Example:**

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover the latest AI trends and breakthroughs",
    backstory="You are an expert AI researcher at a top-tier research lab.",
    tools=[search_tool, scrape_tool],
    llm="gpt-4o",
    verbose=True
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze research findings and identify key patterns",
    backstory="You specialize in distilling complex research into clear insights.",
    tools=[python_repl],
    llm="gpt-4o"
)

writer = Agent(
    role="Technical Writer",
    goal="Write a compelling, well-structured report",
    backstory="You are a skilled communicator who makes technical topics accessible.",
    llm="gpt-4o"
)

# Define tasks
research_task = Task(
    description="Research the top 5 AI trends for 2026",
    expected_output="A detailed list of trends with supporting evidence",
    agent=researcher
)

analysis_task = Task(
    description="Analyze the research and rank trends by impact",
    expected_output="Ranked trends with impact scores and justification",
    agent=analyst
)

report_task = Task(
    description="Write a professional report on AI trends",
    expected_output="A polished 2-page report with sections and citations",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

result = crew.kickoff()
```

**When to use CrewAI:**
- Multi-agent collaboration with clear role separation
- Simulating team dynamics (researcher + analyst + writer)
- When you want high-level abstraction over multi-agent orchestration
- Rapid prototyping of multi-agent workflows

---

## **3.4 AutoGen (Microsoft)**

**What it is:** A framework for building **multi-agent conversational systems** where agents interact through natural language messages, optionally with human participation.

```
┌──────────────────────────────────────────────────────────────────┐
│                    AutoGen CONVERSATION FLOW                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────┐    message    ┌─────────────┐                   │
│   │  User      │─────────────▶│  Assistant   │                   │
│   │  Proxy     │◀─────────────│  Agent       │                   │
│   └────────────┘    reply     └──────┬───────┘                   │
│         ▲                            │                            │
│         │                    code_execution                       │
│         │                            │                            │
│         │                    ┌───────▼───────┐                    │
│         └────────────────────│  Code Agent   │                    │
│              result          │  (sandboxed)  │                    │
│                              └───────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Conversational agents** that chat with each other to solve problems
- **AssistantAgent:** LLM-powered agent that generates responses and code
- **UserProxyAgent:** Executes code, acts as human proxy
- **GroupChat:** Multiple agents in a shared conversation
- **Human-in-the-loop:** Configurable human approval at any step

**Code Example:**

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

assistant = AssistantAgent(
    name="AI_Assistant",
    llm_config={"model": "gpt-4o", "temperature": 0},
    system_message="You are a helpful AI assistant. Write Python code to solve tasks."
)

coder = AssistantAgent(
    name="Coder",
    llm_config={"model": "gpt-4o"},
    system_message="You write clean, efficient Python code. Always include error handling."
)

reviewer = AssistantAgent(
    name="Reviewer",
    llm_config={"model": "gpt-4o"},
    system_message="You review code for bugs, style, and performance issues."
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # or "ALWAYS" for human-in-the-loop
    code_execution_config={"work_dir": "workspace", "use_docker": True}
)

# Group chat
group_chat = GroupChat(
    agents=[user_proxy, assistant, coder, reviewer],
    messages=[],
    max_round=10
)
manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4o"})

user_proxy.initiate_chat(manager, message="Build a FastAPI endpoint for user registration")
```

**When to use AutoGen:**
- Code generation and execution workflows
- Simulating multi-person conversations (agent-to-agent debate)
- When code execution in sandboxed environments is critical
- Research prototyping with conversational agent teams

---

## **3.5 Haystack (deepset)**

**What it is:** A production-ready framework for building NLP/LLM pipelines, now with **agent capabilities using the ReAct paradigm** — combining pipeline-based retrieval with tool-using agents.

**Key Features:**
- Pipeline-based architecture (DAG of components)
- Built-in support for RAG, document processing, embeddings
- Agent component using ReAct loop
- Strong production focus (logging, evaluation, deployment)

**Code Example:**

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

# Define tools for the agent
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Results for: {query}"

search_tool = Tool(
    name="web_search",
    description="Search the web for current information",
    function=web_search,
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"]
    }
)

# Create agent with ReAct-style reasoning
chat_generator = OpenAIChatGenerator(
    model="gpt-4o",
    tools=[search_tool]
)
```

**When to use Haystack:**
- Production RAG pipelines that need agent capabilities
- Document-heavy workflows (legal, medical, enterprise)
- When you need strong evaluation and monitoring built in
- Teams already invested in the Haystack ecosystem

---

## **3.6 smolagents (HuggingFace)**

**What it is:** A lightweight, Pythonic agent framework from HuggingFace where agents solve tasks by **generating and executing Python code** rather than JSON tool calls.

```
Traditional Agent:                      smolagents:
  Thought → JSON tool call →              Thought → Python code →
  Parse response → Next thought            Execute code → Next thought

Example tool call (JSON):              Example tool call (Code):
  {"tool": "search",                     results = search("AI trends")
   "args": {"query": "AI trends"}}       summary = summarize(results[:3])
```

**Key Features:**
- **Code-based agents:** Generate Python code instead of JSON tool calls
- **Simpler mental model:** The LLM writes Python, Python runs, results come back
- **Lightweight:** Minimal abstractions, easy to understand internals
- **HuggingFace Hub integration:** Share and load tools from the Hub
- **Multi-agent support:** `ManagedAgent` for hierarchical orchestration

**Code Example:**

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city to get weather for.
    
    Returns:
        Weather information string.
    """
    return f"Weather in {city}: 72°F, sunny"

agent = CodeAgent(
    tools=[get_weather],
    model=HfApiModel("Qwen/Qwen2.5-72B-Instruct"),
    max_steps=5
)

result = agent.run("What's the weather in San Francisco and should I bring an umbrella?")
```

**When to use smolagents:**
- Quick prototyping with minimal boilerplate
- When code generation is preferable to JSON tool calls
- Projects heavily integrated with HuggingFace ecosystem
- Lightweight agents that don't need complex orchestration

---

## **3.7 Framework Comparison Table**

| Framework | Approach | Orchestration | Best For | Key Strengths | Limitations |
|---|---|---|---|---|---|
| **LangChain** | Chain-based, LCEL pipes | Sequential chains | Rapid prototyping, tool integration | Huge ecosystem, 100+ tools, great docs | Can get complex; chains aren't graphs |
| **LangGraph** | Graph-based (DAG + cycles) | Stateful directed graph | Complex multi-step workflows | Branching, parallelism, persistence, human-in-loop | Steeper learning curve |
| **CrewAI** | Role-based crews | Sequential or hierarchical | Multi-agent role collaboration | Intuitive role abstraction, fast setup | Less fine-grained control |
| **AutoGen** | Conversational agents | Message passing | Code generation, agent debates | Built-in code execution, group chat | Verbose conversations, cost can spike |
| **Haystack** | Pipeline components + ReAct | DAG pipeline | Production RAG + agents | Strong evaluation, production-ready | Smaller agent ecosystem |
| **smolagents** | Python code generation | Code execution | Lightweight, HuggingFace ecosystem | Minimal abstraction, code-first | Less mature, fewer integrations |

**Decision Flowchart:**

```
Need agents?
├── Simple single agent + tools → LangChain
├── Complex workflow with loops/branches → LangGraph
├── Role-based team collaboration → CrewAI
├── Code generation + execution → AutoGen
├── Production RAG + agents → Haystack
└── Quick prototype, HuggingFace → smolagents
```

> **Interview tip:** When asked "Which framework would you choose?", don't just name one. Show your reasoning: "It depends on the use case. For a production workflow with branching logic and human approval steps, I'd choose **LangGraph** because of its stateful graph execution and checkpointing. For rapid prototyping of a research agent, **LangChain** or **CrewAI** would get us to an MVP faster."

---

# **4. Multi-Agent System Design**

---

## **4.1 Design Principles**

```
┌────────────────────────────────────────────────────────────────────┐
│              MULTI-AGENT DESIGN PRINCIPLES                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐       │
│  │   MODULAR    │   │   SCALABLE   │   │  INTEROPERABLE    │       │
│  │             │   │              │   │                   │       │
│  │ Each agent  │   │ Add/remove   │   │ Agents communicate│       │
│  │ has ONE job │   │ agents w/o   │   │ via standard      │       │
│  │ Single      │   │ breaking     │   │ protocols (MCP,   │       │
│  │ responsibility│ │ the system   │   │ HTTP, messages)   │       │
│  └─────────────┘   └──────────────┘   └───────────────────┘       │
│                                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐       │
│  │  RESILIENT   │   │  OBSERVABLE  │   │  COST-AWARE       │       │
│  │             │   │              │   │                   │       │
│  │ Graceful    │   │ Log every    │   │ Budget tokens per │       │
│  │ degradation │   │ agent step,  │   │ agent, use cheaper│       │
│  │ Fallback    │   │ trace tool   │   │ models for simple │       │
│  │ strategies  │   │ calls        │   │ sub-tasks         │       │
│  └─────────────┘   └──────────────┘   └───────────────────┘       │
└────────────────────────────────────────────────────────────────────┘
```

| Principle | What It Means | How to Implement |
|---|---|---|
| **Modularity** | Each agent encapsulates one capability | Separate agent classes, clear interfaces |
| **Scalability** | System handles load by adding agents | Stateless agents behind load balancers |
| **Interoperability** | Agents from different frameworks can cooperate | Standard communication (MCP, HTTP, gRPC) |
| **Resilience** | System works even if an agent fails | Circuit breakers, fallback agents, retries |
| **Observability** | Every action is traceable | Structured logging, distributed tracing |
| **Cost Awareness** | Token budgets per agent/task | Smaller models for routing, large for reasoning |

---

## **4.2 Agent Roles in Multi-Agent Systems**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT ROLE HIERARCHY                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                     ┌──────────────────┐                               │
│                     │   SUPERVISOR      │                               │
│                     │   (Orchestrator)  │                               │
│                     │                  │                               │
│                     │ • Routes tasks   │                               │
│                     │ • Monitors agents│                               │
│                     │ • Aggregates     │                               │
│                     └────────┬─────────┘                               │
│                              │                                         │
│          ┌───────────────────┼───────────────────┐                     │
│          │                   │                   │                     │
│  ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐             │
│  │   PLANNER     │  │  RESEARCHER   │  │   EXECUTOR    │             │
│  │               │  │               │  │               │             │
│  │ Decomposes    │  │ Searches,     │  │ Runs code,    │             │
│  │ complex tasks │  │ retrieves,    │  │ calls APIs,   │             │
│  │ into sub-tasks│  │ reads docs    │  │ writes files  │             │
│  └───────────────┘  └───────────────┘  └───────────────┘             │
│                                                                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│  │   ANALYST     │  │   CRITIC      │  │  HUMAN PROXY  │             │
│  │               │  │  (Reflector)  │  │               │             │
│  │ Processes     │  │ Reviews       │  │ Escalates to  │             │
│  │ data, finds   │  │ outputs,      │  │ human for     │             │
│  │ patterns      │  │ flags issues  │  │ approval      │             │
│  └───────────────┘  └───────────────┘  └───────────────┘             │
└────────────────────────────────────────────────────────────────────────┘
```

| Role | Responsibility | Typical Model | Tools |
|---|---|---|---|
| **Supervisor** | Route tasks, aggregate results, handle failures | GPT-4o / Claude (strong reasoning) | None (orchestration only) |
| **Planner** | Decompose goals into sub-tasks with dependencies | GPT-4o | Task decomposition prompts |
| **Researcher** | Gather information from external sources | GPT-4o-mini (cost-efficient) | Web search, RAG, document loaders |
| **Analyst** | Process data, compute metrics, extract insights | GPT-4o / Code Interpreter | Python REPL, SQL, pandas |
| **Executor** | Take real-world actions (API calls, writes) | GPT-4o-mini | APIs, file I/O, webhooks |
| **Critic / Reflector** | Review outputs for quality, accuracy, completeness | Claude (strong at evaluation) | None (LLM reasoning only) |

---

## **4.3 Communication Protocols**

Agents must communicate to collaborate. The choice of protocol depends on latency requirements, scale, and coupling:

```
┌────────────────────────────────────────────────────────────────────┐
│              AGENT COMMUNICATION PROTOCOLS                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. DIRECT FUNCTION CALLS (Tight Coupling)                         │
│     Agent A ──── calls ────▶ Agent B.run(task)                     │
│     • Simplest, fastest                                            │
│     • No network overhead                                          │
│     • Tightly coupled — hard to scale independently                │
│                                                                    │
│  2. HTTP / REST / gRPC (Loose Coupling)                            │
│     Agent A ──── POST ────▶ /api/agent-b/run                      │
│     • Agents as microservices                                      │
│     • Independently deployable                                     │
│     • gRPC for performance-critical paths                          │
│                                                                    │
│  3. MESSAGE BROKERS (Event-Driven)                                 │
│     Agent A ──── publish ────▶ [Kafka/RabbitMQ] ────▶ Agent B     │
│     • Fully decoupled, async                                       │
│     • Built-in retry, dead-letter queues                           │
│     • Best for high-throughput, event-driven systems               │
│                                                                    │
│  4. WEBSOCKETS (Real-Time Streaming)                               │
│     Agent A ◀────── bidirectional ──────▶ Agent B                  │
│     • Real-time updates                                            │
│     • Streaming token-by-token                                     │
│     • Good for interactive/collaborative agents                    │
│                                                                    │
│  5. MCP (Model Context Protocol)                                   │
│     Agent ◀────── standardized ──────▶ MCP Server (tools/data)    │
│     • Anthropic's open standard                                    │
│     • Plug-and-play tool servers                                   │
│     • Agents discover tools dynamically                            │
└────────────────────────────────────────────────────────────────────┘
```

| Protocol | Latency | Coupling | Scale | Best For |
|---|---|---|---|---|
| Direct calls | ~μs | Tight | Single process | Prototyping, simple systems |
| HTTP/REST | ~ms | Loose | Horizontal | Microservice agents |
| gRPC | ~ms (faster) | Loose | Horizontal | High-performance inter-agent |
| Kafka/RabbitMQ | ~ms-s | None | Massive | Event-driven, async pipelines |
| WebSockets | ~ms | Medium | Moderate | Streaming, real-time collaboration |
| MCP | ~ms | Standardized | Extensible | Tool discovery, interop |

---

## **4.4 Orchestration Patterns**

### **Hierarchical Orchestration**

```
                    ┌──────────────┐
                    │  Supervisor   │
                    │  (top-level)  │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼───────┐
     │  Research   │ │  Analysis  │ │  Writing   │
     │  Sub-Sup    │ │  Agent     │ │  Agent     │
     └──────┬──────┘ └────────────┘ └────────────┘
            │
     ┌──────┼──────┐
     │             │
  ┌──▼───┐   ┌───▼──┐
  │ Web  │   │ Doc  │
  │Search│   │Parse │
  └──────┘   └──────┘
```

- **Pros:** Clear chain of command, easy to debug, natural task delegation
- **Cons:** Supervisor is a bottleneck, single point of failure

### **Decentralized / Peer-to-Peer Orchestration**

```
     ┌──────────┐       ┌──────────┐
     │ Agent A   │◀─────▶│ Agent B   │
     └─────┬────┘       └─────┬────┘
           │                   │
           └──────┐   ┌───────┘
                  ▼   ▼
            ┌──────────┐
            │ Agent C   │
            └──────────┘

  Shared Message Bus / State Store
  ═══════════════════════════════
```

- **Pros:** No single bottleneck, resilient, agents self-organize
- **Cons:** Harder to debug, potential for circular loops, coordination overhead

### **Hybrid (Recommended for Production)**

Use hierarchical for high-level orchestration + decentralized for specialist agent communication:

```
Supervisor (hierarchical)
├── Research Team (decentralized: web + doc + data agents collaborate)
├── Analysis Agent (single)
└── Review Team (decentralized: critic + fact-checker collaborate)
```

---

## **4.5 Task Decomposition and Routing**

```python
# LangGraph-based task decomposition and routing
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class OrchestratorState(TypedDict):
    query: str
    sub_tasks: list[dict]
    results: dict
    final_output: str

def decompose_task(state: OrchestratorState) -> OrchestratorState:
    """LLM decomposes user query into sub-tasks with specialist assignments."""
    prompt = f"""Decompose this task into sub-tasks.
    Assign each to a specialist: researcher, analyst, writer.
    
    Task: {state['query']}
    
    Return JSON: [{{"task": "...", "specialist": "...", "priority": 1}}]"""
    
    response = llm.invoke(prompt)
    sub_tasks = parse_json(response.content)
    return {"sub_tasks": sub_tasks}

def route_to_specialist(state: OrchestratorState) -> str:
    """Route to the next unfinished sub-task's specialist."""
    for task in state["sub_tasks"]:
        if task["task"] not in state.get("results", {}):
            return task["specialist"]
    return "aggregator"

graph = StateGraph(OrchestratorState)
graph.add_node("decomposer", decompose_task)
graph.add_node("researcher", researcher_agent)
graph.add_node("analyst", analyst_agent)
graph.add_node("writer", writer_agent)
graph.add_node("aggregator", aggregate_results)

graph.add_edge(START, "decomposer")
graph.add_conditional_edges("decomposer", route_to_specialist)
graph.add_conditional_edges("researcher", route_to_specialist)
graph.add_conditional_edges("analyst", route_to_specialist)
graph.add_conditional_edges("writer", route_to_specialist)
graph.add_edge("aggregator", END)
```

---

# **5. Agentic Patterns**

---

## **5.1 Router-Evaluator Pattern**

Route the user request to a specialist agent, then evaluate the output quality.

```
┌────────────────────────────────────────────────────────────────────┐
│                    ROUTER-EVALUATOR PATTERN                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  User Query                                                        │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────┐                                                      │
│  │  ROUTER  │──── "This is a coding question" ────┐               │
│  │ (classify)│                                     │               │
│  └──────────┘                                      ▼               │
│      │          ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│      ├─ Math ──▶│ Math Agent │  │ Code Agent │  │ Research   │    │
│      ├─ Code ──▶│            │  │     ▲      │  │ Agent      │    │
│      └─Research▶│            │  │     │      │  │            │    │
│                 └────────────┘  └─────┼──────┘  └────────────┘    │
│                                       │                            │
│                                       ▼                            │
│                               ┌──────────────┐                     │
│                               │  EVALUATOR   │                     │
│                               │              │                     │
│                               │ Quality ≥ 8? │                     │
│                               │ YES → Output │                     │
│                               │ NO → Retry   │                     │
│                               └──────────────┘                     │
└────────────────────────────────────────────────────────────────────┘
```

```python
# Router-Evaluator in LangGraph
def router(state):
    classification = llm.invoke(
        f"Classify this query into one of: math, code, research.\nQuery: {state['query']}"
    )
    return {"route": classification.content.strip().lower()}

def evaluator(state):
    score = llm.invoke(
        f"Rate this answer 1-10 for quality.\nQuestion: {state['query']}\nAnswer: {state['answer']}"
    )
    return {"quality_score": int(score.content.strip())}

def should_retry(state) -> Literal["retry", "output"]:
    return "output" if state["quality_score"] >= 8 else "retry"
```

---

## **5.2 Plan-Act-Reflect Pattern**

The planner drafts a strategy, executors carry it out, and a reflector evaluates.

```
┌────────────────────────────────────────────────────────────────────┐
│                   PLAN-ACT-REFLECT PATTERN                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐     ┌───────────────────────────────┐               │
│  │ PLANNER  │────▶│        EXECUTION AGENTS       │               │
│  │          │     │  ┌─────┐ ┌─────┐ ┌─────────┐ │               │
│  │ Creates  │     │  │ A1  │ │ A2  │ │   A3    │ │               │
│  │ step-by- │     │  │Web  │ │SQL  │ │Compute  │ │               │
│  │ step plan│     │  └──┬──┘ └──┬──┘ └────┬────┘ │               │
│  └──────────┘     │     │      │        │       │               │
│       ▲           │     └──────┴────────┘       │               │
│       │           └──────────────┬───────────────┘               │
│       │                          │                                │
│       │                          ▼                                │
│       │                  ┌──────────────┐                         │
│       │                  │  REFLECTOR   │                         │
│       └── revise plan ◄──│              │                         │
│                          │ • Complete?  │                         │
│                          │ • Accurate?  │                         │
│                          │ • Next step? │                         │
│                          └──────────────┘                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## **5.3 Concurrent Specialists (Fan-Out / Fan-In)**

Multiple specialist agents work in **parallel** on independent sub-tasks, and a merger combines results.

```
┌────────────────────────────────────────────────────────────────────┐
│               CONCURRENT SPECIALISTS PATTERN                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│                    ┌──────────┐                                     │
│                    │ SPLITTER │                                     │
│                    │ (decompose)│                                   │
│                    └─────┬────┘                                     │
│           ┌──────────────┼──────────────┐                          │
│           │              │              │                          │
│           ▼              ▼              ▼                          │
│    ┌────────────┐ ┌────────────┐ ┌────────────┐                   │
│    │ Specialist │ │ Specialist │ │ Specialist │  ← parallel       │
│    │ A (Legal)  │ │ B (Finance)│ │ C (Tech)   │                   │
│    └─────┬──────┘ └─────┬──────┘ └─────┬──────┘                   │
│          │              │              │                           │
│          └──────────────┼──────────────┘                           │
│                         ▼                                          │
│                  ┌────────────┐                                     │
│                  │   MERGER   │                                     │
│                  │ (aggregate)│                                     │
│                  └────────────┘                                     │
└────────────────────────────────────────────────────────────────────┘
```

```python
# Fan-out / Fan-in with LangGraph
import asyncio

async def fan_out_to_specialists(state):
    tasks = state["sub_tasks"]
    # Execute all specialists in parallel
    results = await asyncio.gather(*[
        specialist_agents[task["type"]].ainvoke(task) 
        for task in tasks
    ])
    return {"specialist_results": results}

def merge_results(state):
    combined = "\n\n".join(
        f"## {r['specialist']}\n{r['output']}" 
        for r in state["specialist_results"]
    )
    final = llm.invoke(f"Synthesize these findings:\n{combined}")
    return {"final_output": final.content}
```

---

## **5.4 Tool-Augmented Agent Pattern**

The agent dynamically selects and invokes tools based on the task, using function calling.

```
┌────────────────────────────────────────────────────────────────────┐
│                TOOL-AUGMENTED AGENT PATTERN                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  User: "What's the stock price of AAPL and plot a 30-day chart?"  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────┐        │
│  │                  AGENT (LLM Core)                      │        │
│  │                                                        │        │
│  │  Think: I need stock data → use stock_price tool       │        │
│  │  Think: I need a chart   → use chart_generator tool    │        │
│  │                                                        │        │
│  │  Available Tools:                                      │        │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────────────┐   │        │
│  │  │stock_price │ │ chart_gen│ │ web_search         │   │        │
│  │  │            │ │          │ │                    │   │        │
│  │  │ get_price()│ │ plot()   │ │ search()           │   │        │
│  │  │ history()  │ │ bar()    │ │ fetch_page()       │   │        │
│  │  └────────────┘ └──────────┘ └────────────────────┘   │        │
│  │                                                        │        │
│  │  Execution:                                            │        │
│  │  1. stock_price.history("AAPL", days=30) → data       │        │
│  │  2. chart_gen.plot(data, title="AAPL 30-Day") → img   │        │
│  │  3. Compose response with data + chart                 │        │
│  └────────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────────┘
```

---

## **5.5 ReAct (Reasoning + Acting)**

The most fundamental agentic pattern. The agent **interleaves reasoning traces with actions** in a loop.

```
┌────────────────────────────────────────────────────────────────────┐
│                     ReAct PATTERN                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  LOOP:                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ THOUGHT: "I need to find the GDP of France in 2025.        │   │
│  │           Let me search for this."                          │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ ACTION:  search("GDP of France 2025")                       │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ OBSERVATION: "France GDP in 2025 was $3.1 trillion..."     │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ THOUGHT: "I found the GDP. Now I need to compare it with   │   │
│  │           Germany's GDP to answer the user's question."     │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ ACTION:  search("GDP of Germany 2025")                      │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ OBSERVATION: "Germany GDP in 2025 was $4.5 trillion..."    │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ THOUGHT: "I have both values. France: $3.1T, Germany:      │   │
│  │           $4.5T. Germany's GDP is 45% higher. I can now    │   │
│  │           provide the final answer."                        │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ FINAL ANSWER: "Germany's GDP ($4.5T) exceeds France's      │   │
│  │                ($3.1T) by approximately 45%."               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

**Why ReAct Works:**

| Component | Purpose | Benefit |
|---|---|---|
| **Thought** | Explicit reasoning before action | Reduces hallucination, makes decisions traceable |
| **Action** | Tool call or environment interaction | Grounds reasoning in real data |
| **Observation** | Tool/environment response | Provides factual evidence |
| **Loop** | Iterate until task is complete | Handles multi-step reasoning |

**ReAct Implementation:**

```python
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template("""Answer the following question using the tools provided.

You have access to the following tools:
{tools}

Use the following format:

Thought: reason about what to do
Action: the tool to use, one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}""")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
```

> **Interview tip:** ReAct is the most commonly asked-about pattern. Know the paper: *Yao et al. (2023), "ReAct: Synergizing Reasoning and Acting in Language Models."* The key insight is that reasoning traces help the model plan and update plans, while actions ground the reasoning in real observations.

---

# **6. Memory in Agents**

---

## **6.1 Memory Architecture**

```
┌────────────────────────────────────────────────────────────────────┐
│                   AGENT MEMORY ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SHORT-TERM MEMORY                         │   │
│  │  • Conversation buffer (last N messages)                    │   │
│  │  • Current session context                                  │   │
│  │  • Token window: fits in LLM context window                 │   │
│  │  • Lifespan: single conversation / session                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     LONG-TERM MEMORY                        │   │
│  │  • Vector store (ChromaDB, Pinecone, pgvector)              │   │
│  │  • Stores embeddings of past interactions, facts, docs      │   │
│  │  • Retrieved via semantic similarity at query time          │   │
│  │  • Lifespan: persistent across sessions                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    EPISODIC MEMORY                           │   │
│  │  • Summaries of past task executions                        │   │
│  │  • "Last time I searched for stock data, I used Yahoo API   │   │
│  │     and it worked well" → stored as episode                 │   │
│  │  • Helps agent learn from past successes/failures           │   │
│  │  • Lifespan: persistent, indexed by task type               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    WORKING MEMORY                            │   │
│  │  • Current task context: sub-task list, partial results     │   │
│  │  • Active plan state (which steps done, which pending)      │   │
│  │  • Scratchpad for intermediate computations                 │   │
│  │  • Lifespan: current task execution only                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## **6.2 Detailed Memory Comparison**

| Memory Type | Storage | Capacity | Retrieval | Lifespan | Implementation |
|---|---|---|---|---|---|
| **Short-Term** | In-context (prompt) | Limited by context window (128K tokens) | Direct inclusion in prompt | Single session | `ConversationBufferMemory` |
| **Long-Term** | Vector DB (Pinecone, Chroma) | Unlimited | Semantic search (cosine similarity) | Persistent | Embedding + vector store |
| **Episodic** | Structured DB or vector store | Moderate | Task-type index + similarity | Persistent | Custom store with metadata |
| **Working** | State dict / graph state | Task-scoped | Direct access by key | Current task | `TypedDict` in LangGraph |

---

## **6.3 Memory Implementation Examples**

### **Short-Term: Conversation Buffer**

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep last 10 exchanges in context
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# Automatically injects chat history into prompt
memory.save_context(
    {"input": "What's the weather?"},
    {"output": "It's 72°F and sunny in SF."}
)
```

### **Long-Term: Vector Store Memory**

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Persist memories as embeddings
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./memory_store"
)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Save a memory
memory.save_context(
    {"input": "User prefers Python over JavaScript"},
    {"output": "Noted. Will default to Python for code examples."}
)

# Later — retrieve relevant memories
relevant = memory.load_memory_variables({"input": "Write a web scraper"})
# Returns: memories about user's Python preference
```

### **Episodic: Task Execution Summaries**

```python
# Store episode after task completion
episode = {
    "task_type": "data_analysis",
    "query": "Analyze Q4 sales data",
    "tools_used": ["sql_executor", "pandas_repl", "chart_generator"],
    "outcome": "success",
    "duration_seconds": 45,
    "lessons": "User prefers bar charts over pie charts. SQL query on 'sales' table.",
    "timestamp": "2026-02-16T10:30:00Z"
}
episodic_store.add(episode)

# Retrieve relevant episodes for new task
past_episodes = episodic_store.search("analyze sales data", k=3)
# Agent can now reuse successful strategies from past tasks
```

### **Working Memory: LangGraph State**

```python
class AgentWorkingMemory(TypedDict):
    # Current task decomposition
    plan: list[dict]             # [{"step": 1, "task": "...", "status": "done"}]
    current_step: int            # Which step we're on
    intermediate_results: dict   # {step_id: result}
    scratchpad: str             # Free-form notes for the agent
    error_log: list[str]        # Track failures for retry logic
```

> **Interview tip:** When asked about agent memory, draw this 4-type taxonomy. Emphasize that **short-term memory** is just the context window, **long-term memory** requires a vector store for semantic retrieval, **episodic memory** is how agents learn from past executions, and **working memory** tracks the current task state. Most production agents need all four.

---

# **7. Tool Use and Function Calling**

---

## **7.1 How Tool Use Works**

```
┌────────────────────────────────────────────────────────────────────┐
│                    TOOL USE FLOW                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. DEFINE: Register tools with JSON schemas                       │
│                    │                                               │
│                    ▼                                               │
│  2. INJECT: Send tool definitions to LLM alongside user message   │
│                    │                                               │
│                    ▼                                               │
│  3. DECIDE: LLM outputs a structured tool call (not free text)    │
│             {"name": "search", "arguments": {"query": "..."}}     │
│                    │                                               │
│                    ▼                                               │
│  4. EXECUTE: Application parses the tool call, executes function  │
│                    │                                               │
│                    ▼                                               │
│  5. RETURN: Send tool result back to LLM as a new message         │
│                    │                                               │
│                    ▼                                               │
│  6. RESPOND: LLM generates final answer incorporating tool result │
└────────────────────────────────────────────────────────────────────┘
```

---

## **7.2 JSON Schema Tool Definitions**

Tools are defined using **JSON Schema** so the LLM knows what's available and how to call each tool:

```json
{
  "type": "function",
  "function": {
    "name": "get_stock_price",
    "description": "Get the current stock price for a given ticker symbol. Use when the user asks about stock prices or market data.",
    "parameters": {
      "type": "object",
      "properties": {
        "ticker": {
          "type": "string",
          "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')"
        },
        "include_history": {
          "type": "boolean",
          "description": "Whether to include 30-day price history",
          "default": false
        }
      },
      "required": ["ticker"]
    }
  }
}
```

**Best Practices for Tool Definitions:**

| Aspect | Good Practice | Why It Matters |
|---|---|---|
| **Name** | `get_stock_price` (verb_noun) | Clear action intent |
| **Description** | Detailed, includes "when to use" | LLM uses this to decide tool selection |
| **Parameters** | Strong types, clear descriptions | Reduces malformed calls |
| **Required** | Mark truly required params | Prevents missing argument errors |
| **Enums** | Use when values are constrained | Prevents invalid inputs |

---

## **7.3 OpenAI Function Calling Format**

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the product database by query. Returns top matching products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food", "books"],
                        "description": "Product category filter"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# LLM call with tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Find me the best headphones under $200"}],
    tools=tools,
    tool_choice="auto"  # "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
)

# Check if LLM wants to call a tool
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
    # Output: Tool: search_database
    # Output: Args: {"query": "headphones", "category": "electronics", "max_results": 5}
    
    # Execute the tool and send result back
    tool_result = execute_tool(tool_call.function.name, tool_call.function.arguments)
    
    # Send tool result back to LLM
    follow_up = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Find me the best headphones under $200"},
            message,  # assistant message with tool_call
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            }
        ]
    )
```

---

## **7.4 How Agents Decide Which Tool to Use**

The LLM selects tools based on:

```
┌────────────────────────────────────────────────────────────────────┐
│              TOOL SELECTION DECISION PROCESS                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. SEMANTIC MATCHING                                              │
│     User query → match against tool descriptions                   │
│     "What's the weather?" → matches "get_weather" tool            │
│                                                                    │
│  2. PARAMETER EXTRACTION                                           │
│     Extract entities from query → map to tool parameters           │
│     "Weather in Tokyo" → {"city": "Tokyo"}                        │
│                                                                    │
│  3. MULTI-TOOL PLANNING                                            │
│     Complex queries → select multiple tools in sequence            │
│     "Compare AAPL vs MSFT" → [get_stock("AAPL"), get_stock("MSFT")]│
│                                                                    │
│  4. CONFIDENCE THRESHOLDING                                        │
│     If no tool matches well → respond with LLM knowledge alone    │
│     "What is machine learning?" → no tool needed                  │
│                                                                    │
│  5. PARALLEL vs SEQUENTIAL                                         │
│     Independent calls → parallel (OpenAI supports this natively)  │
│     Dependent calls → sequential (result of A feeds into B)       │
└────────────────────────────────────────────────────────────────────┘
```

> **Interview tip:** When asked "How does an LLM know which tool to use?", explain that tool descriptions are injected into the system prompt. The LLM uses **semantic understanding** to match the user's intent to the most relevant tool. Good tool descriptions are critical — they're essentially "prompt engineering for tools."

---

# **8. LLM Selection for Agents**

---

## **8.1 Model Comparison for Agentic Workloads**

| Model | Provider | Context Window | Tool Calling | Reasoning | Cost (approx.) | Best For |
|---|---|---|---|---|---|---|
| **GPT-4o** | OpenAI | 128K | Excellent (native) | Very strong | $2.50/M in, $10/M out | Complex orchestration, production agents |
| **GPT-4o-mini** | OpenAI | 128K | Good | Good | $0.15/M in, $0.60/M out | Routing, simple sub-tasks, cost-sensitive |
| **Claude 3.5 Sonnet** | Anthropic | 200K | Excellent | Very strong, careful | ~$3/M in, $15/M out | Evaluation, safety-critical, long context |
| **Claude 3.5 Haiku** | Anthropic | 200K | Good | Good | ~$0.25/M in, $1.25/M out | Fast routing, classification, cheap tasks |
| **Mistral Large** | Mistral | 128K | Good | Strong | ~$2/M in, $6/M out | EU compliance, multilingual agents |
| **LLaMA 3.1 70B** | Meta (open) | 128K | Community wrappers | Good | Self-hosted (GPU cost) | On-premise, data-sensitive workloads |
| **LLaMA 3.1 8B** | Meta (open) | 128K | Community wrappers | Moderate | Self-hosted (cheap) | Edge agents, simple routing |
| **Phi-3.5 Mini** | Microsoft (open) | 128K | Limited | Moderate | Self-hosted (very cheap) | On-device, lightweight agents |
| **Gemini 1.5 Pro** | Google | 1M+ | Good | Strong | ~$1.25/M in, $5/M out | Very long context, multimodal agents |

---

## **8.2 Model Selection Strategy**

```
┌────────────────────────────────────────────────────────────────────┐
│            TIERED MODEL STRATEGY FOR MULTI-AGENT SYSTEMS           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  TIER 1 — ROUTING & CLASSIFICATION (cheap, fast)                   │
│  ├── Model: GPT-4o-mini / Claude 3.5 Haiku / Phi-3.5              │
│  ├── Tasks: Intent classification, task routing, simple parsing    │
│  └── Cost: ~$0.15-0.25/M input tokens                             │
│                                                                    │
│  TIER 2 — EXECUTION & TOOL USE (balanced)                          │
│  ├── Model: GPT-4o-mini / Mistral Large / LLaMA 70B               │
│  ├── Tasks: Tool calling, data extraction, code generation         │
│  └── Cost: ~$0.15-2/M input tokens                                │
│                                                                    │
│  TIER 3 — REASONING & PLANNING (expensive, powerful)               │
│  ├── Model: GPT-4o / Claude 3.5 Sonnet                            │
│  ├── Tasks: Complex planning, reflection, quality evaluation       │
│  └── Cost: ~$2.50-3/M input tokens                                │
│                                                                    │
│  TIER 4 — SPECIALIZED (use case specific)                          │
│  ├── Model: Gemini 1.5 Pro (long context), LLaMA (on-premise)     │
│  ├── Tasks: Very long documents, data-sensitive workloads          │
│  └── Cost: Variable                                                │
└────────────────────────────────────────────────────────────────────┘
```

**Cost Optimization Rule:** Use the **cheapest model that can reliably perform the task**. A supervisor agent can use GPT-4o for planning while its sub-agents use GPT-4o-mini for execution — this can reduce costs by 90%+ with minimal quality loss.

---

# **9. Deployment Stack**

---

## **9.1 Production Architecture**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT STACK                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                         CLIENT LAYER                             │    │
│  │  React/Next.js Frontend  ◄────▶  WebSocket/REST                 │    │
│  └──────────────────────────────┬───────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────▼───────────────────────────────────┐    │
│  │                      API GATEWAY / LOAD BALANCER                 │    │
│  │  Nginx / AWS ALB / Kong                                          │    │
│  │  Rate limiting, auth, SSL termination                            │    │
│  └──────────────────────────────┬───────────────────────────────────┘    │
│                                 │                                        │
│  ┌──────────────────────────────▼───────────────────────────────────┐    │
│  │                      FastAPI BACKEND                             │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │ /api/agent   │  │ /api/tasks   │  │ /api/ws/stream      │   │    │
│  │  │ (invoke)     │  │ (status)     │  │ (WebSocket)         │   │    │
│  │  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │    │
│  │         │                                                        │    │
│  │  ┌──────▼────────────────────────────────────────────────────┐   │    │
│  │  │              AGENT ORCHESTRATION LAYER                    │   │    │
│  │  │  LangGraph StateGraph → nodes → conditional edges        │   │    │
│  │  │  Checkpointing → Redis / PostgreSQL                      │   │    │
│  │  └───────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      DATA & STORAGE LAYER                        │    │
│  │  ┌────────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────┐  │    │
│  │  │ PostgreSQL │  │  Redis      │  │ Pinecone  │  │ S3/GCS   │  │    │
│  │  │ (metadata) │  │  (cache,    │  │ (vectors) │  │ (files)  │  │    │
│  │  │            │  │   state)    │  │           │  │          │  │    │
│  │  └────────────┘  └─────────────┘  └───────────┘  └──────────┘  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      OBSERVABILITY LAYER                         │    │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐   │    │
│  │  │ Prometheus +    │  │ OpenTelemetry│  │ LangSmith /       │   │    │
│  │  │ Grafana         │  │ (traces)     │  │ AgentOps          │   │    │
│  │  │ (metrics)       │  │              │  │ (LLM-specific)    │   │    │
│  │  └─────────────────┘  └──────────────┘  └───────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    INFRASTRUCTURE LAYER                           │    │
│  │  Docker Containers → Kubernetes (EKS/GKE)                        │    │
│  │  CI/CD: GitHub Actions → Build → Test → Deploy                   │    │
│  │  Secrets: AWS Secrets Manager / Vault                            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## **9.2 FastAPI Backend for Agents**

```python
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.graph import StateGraph
import uuid
import asyncio

app = FastAPI(title="Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AgentRequest(BaseModel):
    query: str
    session_id: str | None = None
    max_steps: int = 10

class AgentResponse(BaseModel):
    task_id: str
    status: str
    result: str | None = None

# In-memory task store (use Redis in production)
tasks: dict[str, dict] = {}

@app.post("/api/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """Invoke the agent asynchronously."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running", "result": None}
    
    background_tasks.add_task(run_agent, task_id, request)
    return AgentResponse(task_id=task_id, status="running")

@app.get("/api/agent/status/{task_id}", response_model=AgentResponse)
async def get_status(task_id: str):
    """Check agent task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    return AgentResponse(task_id=task_id, status=task["status"], result=task["result"])

@app.websocket("/api/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """Stream agent responses token-by-token via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            # Stream agent execution
            async for event in agent_graph.astream_events(
                {"messages": [{"role": "user", "content": query}]},
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        await websocket.send_json({"type": "token", "content": token})
                elif event["event"] == "on_tool_start":
                    await websocket.send_json({
                        "type": "tool_start", 
                        "tool": event["name"]
                    })
            
            await websocket.send_json({"type": "done"})
    except Exception as e:
        await websocket.close(code=1001)

async def run_agent(task_id: str, request: AgentRequest):
    """Execute agent in background."""
    try:
        result = await agent_graph.ainvoke({
            "messages": [{"role": "user", "content": request.query}],
            "max_steps": request.max_steps
        })
        tasks[task_id] = {"status": "completed", "result": result["final_answer"]}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "result": str(e)}

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
```

---

## **9.3 Docker Configuration**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/agents
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## **9.4 Monitoring & Observability**

```python
# OpenTelemetry instrumentation for agent tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(BatchSpanExporter(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent-service")

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Custom spans for agent steps
async def traced_agent_step(step_name: str, func, *args):
    with tracer.start_as_current_span(f"agent.{step_name}") as span:
        span.set_attribute("agent.step", step_name)
        span.set_attribute("agent.model", "gpt-4o")
        try:
            result = await func(*args)
            span.set_attribute("agent.status", "success")
            return result
        except Exception as e:
            span.set_attribute("agent.status", "error")
            span.record_exception(e)
            raise
```

```yaml
# Prometheus metrics endpoint (add to FastAPI)
# pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Auto-instrument all endpoints
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

AGENT_INVOCATIONS = Counter("agent_invocations_total", "Total agent invocations", ["status"])
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent execution latency", buckets=[1, 5, 10, 30, 60, 120])
TOKEN_USAGE = Counter("llm_tokens_total", "Total LLM tokens used", ["model", "type"])
```

---

## **9.5 CI/CD with GitHub Actions**

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent Service

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=. --cov-report=xml
      - run: ruff check .

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t agent-service:${{ github.sha }} .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker tag agent-service:${{ github.sha }} $ECR_REGISTRY/agent-service:${{ github.sha }}
          docker push $ECR_REGISTRY/agent-service:${{ github.sha }}
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/agent-service \
            agent-service=$ECR_REGISTRY/agent-service:${{ github.sha }}
          kubectl rollout status deployment/agent-service --timeout=300s
```

---

# **10. Production Considerations**

---

## **10.1 Error Handling & Fallbacks**

```
┌────────────────────────────────────────────────────────────────────┐
│                 ERROR HANDLING STRATEGY                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Level 1: RETRY WITH BACKOFF                                       │
│  ├── Transient errors (rate limits, timeouts, 5xx)                 │
│  ├── Exponential backoff: 1s → 2s → 4s → 8s                      │
│  └── Max 3 retries per tool call                                  │
│                                                                    │
│  Level 2: FALLBACK MODEL                                           │
│  ├── Primary: GPT-4o → Fallback: Claude 3.5 Sonnet               │
│  ├── If primary provider is down, switch automatically             │
│  └── Maintain a ranked list of model alternatives                  │
│                                                                    │
│  Level 3: FALLBACK TOOL                                            │
│  ├── Primary: Tavily search → Fallback: DuckDuckGo               │
│  ├── If API is unavailable, use alternative                        │
│  └── Cache recent results for degraded mode                        │
│                                                                    │
│  Level 4: GRACEFUL DEGRADATION                                     │
│  ├── If all tools fail, respond from LLM knowledge alone          │
│  ├── Clearly communicate limitations to user                       │
│  └── "I couldn't access live data, but based on my knowledge..."  │
│                                                                    │
│  Level 5: CIRCUIT BREAKER                                          │
│  ├── If a tool fails N times in M minutes, stop calling it        │
│  ├── Periodic health checks to re-enable                           │
│  └── Prevents cascading failures                                   │
└────────────────────────────────────────────────────────────────────┘
```

```python
# Production error handling pattern
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

class AgentError(Exception):
    pass

class ToolExecutor:
    def __init__(self):
        self.circuit_breakers = {}  # tool_name → failure_count
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError))
    )
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        # Check circuit breaker
        if self.circuit_breakers.get(tool_name, 0) >= 5:
            raise AgentError(f"Circuit breaker open for {tool_name}")
        
        try:
            result = await self.tools[tool_name].ainvoke(args)
            self.circuit_breakers[tool_name] = 0  # Reset on success
            return result
        except Exception as e:
            self.circuit_breakers[tool_name] = self.circuit_breakers.get(tool_name, 0) + 1
            raise
    
    async def execute_with_fallback(self, primary: str, fallback: str, args: dict) -> str:
        try:
            return await self.execute_tool(primary, args)
        except AgentError:
            return await self.execute_tool(fallback, args)
```

---

## **10.2 Cost Management**

| Strategy | Implementation | Savings |
|---|---|---|
| **Tiered models** | Use GPT-4o-mini for routing, GPT-4o for reasoning | 80-90% on routing costs |
| **Prompt caching** | Cache system prompts (OpenAI supports this natively) | 50% on repeated prefixes |
| **Response caching** | Cache identical queries in Redis (TTL-based) | 100% on cache hits |
| **Token budgets** | Set max_tokens per agent, per task | Prevents runaway costs |
| **Early termination** | Stop if confidence is high after fewer steps | 30-50% on multi-step tasks |
| **Batch processing** | Batch non-urgent requests, use batch API pricing | 50% on batch-eligible calls |

```python
# Token budget enforcement
class TokenBudget:
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.used_tokens = 0
    
    def check_budget(self, estimated_tokens: int) -> bool:
        return (self.used_tokens + estimated_tokens) <= self.max_tokens
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        self.used_tokens += input_tokens + output_tokens
    
    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens
```

---

## **10.3 Latency Optimization**

```
┌────────────────────────────────────────────────────────────────────┐
│                LATENCY OPTIMIZATION TECHNIQUES                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. STREAMING                                                      │
│     Stream tokens to UI as they're generated                       │
│     Perceived latency drops from 5s to <500ms (first token)       │
│                                                                    │
│  2. PARALLEL TOOL EXECUTION                                        │
│     Execute independent tool calls concurrently                    │
│     3 sequential API calls (3s) → 3 parallel calls (1s)          │
│                                                                    │
│  3. SPECULATIVE EXECUTION                                          │
│     Start likely next steps before current step completes          │
│     Pre-fetch data that's probably needed                          │
│                                                                    │
│  4. PROMPT OPTIMIZATION                                            │
│     Shorter prompts = faster inference                              │
│     Remove redundant instructions, use concise schemas             │
│                                                                    │
│  5. MODEL ROUTING                                                  │
│     Simple queries → fast small model (50ms)                      │
│     Complex queries → powerful model (2-5s)                       │
│                                                                    │
│  6. CONNECTION POOLING                                              │
│     Reuse HTTP connections to LLM APIs                             │
│     Saves TLS handshake time (~100ms per request)                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## **10.4 Observability Checklist**

| What to Monitor | Tool | Why |
|---|---|---|
| **Latency per agent step** | OpenTelemetry traces | Identify bottleneck nodes |
| **Token usage per model** | Custom Prometheus counters | Cost tracking and budgeting |
| **Tool success/failure rate** | Prometheus + Grafana | Circuit breaker triggers |
| **Error rates by type** | Structured logging (JSON) | Debug and prioritize fixes |
| **Agent trace (full execution)** | LangSmith / AgentOps | End-to-end replay of agent reasoning |
| **User satisfaction** | Feedback endpoints | Track if agents actually help |
| **Queue depth** | Message broker metrics | Capacity planning |

---

# **11. Common Interview Questions with Strong Answers**

---

## **Q1: "How would you design a multi-agent system for a complex task?"**

**Strong Answer:**

> I'd follow a structured approach:
> 
> **1. Task Analysis:** First, I'd decompose the complex task into atomic sub-tasks and identify dependencies. For example, a "generate market research report" task breaks into: data collection, competitor analysis, trend identification, and report writing.
> 
> **2. Agent Roles:** I'd assign specialist agents: a **Researcher** (web search + document retrieval), an **Analyst** (data processing + pattern recognition), a **Writer** (report generation), and a **Supervisor** (orchestration + quality control).
> 
> **3. Architecture:** I'd use **LangGraph** for orchestration because it supports stateful execution, conditional branching (e.g., if research is insufficient, loop back), and parallel execution (fan-out to multiple researchers).
> 
> **4. Communication:** Agents communicate via shared state in the LangGraph `TypedDict`. For production scale, I'd use message queues (Redis or Kafka) for async communication.
> 
> **5. Model Selection:** Supervisor uses GPT-4o (strong reasoning), Researcher uses GPT-4o-mini (cost-efficient for search tasks), Analyst uses GPT-4o (complex analysis), Writer uses Claude Sonnet (strong writing).
> 
> **6. Production Hardening:** Error handling with retries and fallback models, token budgets per agent, OpenTelemetry tracing, and circuit breakers for external APIs.

---

## **Q2: "Compare LangChain vs LangGraph vs CrewAI — when would you use each?"**

**Strong Answer:**

> These three solve different problems at different abstraction levels:
> 
> **LangChain** is a foundational toolkit — think of it as the "standard library" for LLM apps. It gives you model wrappers, prompt templates, memory, and tool integrations. I'd use it for simple single-agent use cases or as the building block inside a larger system. It's great for rapid prototyping.
> 
> **LangGraph** is built *on top of* LangChain and adds graph-based orchestration. It treats your workflow as a directed graph with nodes (functions) and edges (transitions). I'd choose LangGraph when I need **loops** (agent retries until quality passes), **branching** (route to different specialists), **parallel execution** (fan-out to multiple agents), or **persistence** (checkpointing for long-running tasks). It's my go-to for production multi-agent systems.
> 
> **CrewAI** operates at the highest abstraction level — you define agents by their "role," "goal," and "backstory," then crew them together. I'd choose CrewAI for **role-based collaboration** where the team metaphor is natural (researcher + analyst + writer). It's faster to set up but gives you less fine-grained control over execution flow compared to LangGraph.
> 
> **In practice,** I often use LangChain for the model/tool layer, LangGraph for orchestration, and I'd consider CrewAI for quick prototypes that I'd later migrate to LangGraph for production.

---

## **Q3: "What is the ReAct pattern and why is it effective?"**

**Strong Answer:**

> ReAct stands for **Reasoning + Acting**, from the 2023 paper by Yao et al. It's a prompting/execution pattern where the agent interleaves **reasoning traces** (thinking out loud) with **actions** (tool calls) in a loop:
> 
> **Thought** → **Action** → **Observation** → **Thought** → ... → **Final Answer**
> 
> Why it works:
> 1. **Reasoning traces reduce hallucination** — the model explicitly plans before acting, catching potential errors early.
> 2. **Actions ground reasoning in facts** — instead of guessing, the agent retrieves real data.
> 3. **The loop enables multi-step problem solving** — the agent can chain multiple tool calls, each informed by previous observations.
> 4. **Observability** — you can inspect the thought traces to understand *why* the agent made each decision.
> 
> Compared to alternatives: **Act-only** (just call tools without reasoning) leads to poor tool selection. **Reason-only** (Chain-of-Thought without tools) leads to hallucination. ReAct combines the best of both.

---

## **Q4: "How do agents handle errors and recover gracefully?"**

**Strong Answer:**

> I implement a multi-layered error handling strategy:
> 
> **Layer 1 — Retry with exponential backoff:** For transient errors (rate limits, timeouts), retry 2-3 times with increasing delays. Most transient issues resolve within seconds.
> 
> **Layer 2 — Fallback models:** If the primary LLM (e.g., GPT-4o) is down, automatically switch to a fallback (Claude Sonnet). I maintain a ranked fallback chain.
> 
> **Layer 3 — Fallback tools:** If a search API fails, switch to an alternative (Tavily → DuckDuckGo). Cache recent results as a degraded-mode backup.
> 
> **Layer 4 — Self-correction via reflection:** After a tool returns an error or unexpected result, the agent's reflection step detects the issue and revises its plan. For example, if a SQL query fails, the reflector might suggest fixing the query syntax and retrying.
> 
> **Layer 5 — Circuit breakers:** If a tool fails repeatedly (e.g., 5 times in 10 minutes), I "open the circuit" — stop calling that tool and switch to alternatives. Periodic health checks re-enable it.
> 
> **Layer 6 — Graceful degradation:** If all tools fail, the agent falls back to its parametric knowledge with a clear disclaimer: "I couldn't access live data, but based on my training..."
> 
> **Layer 7 — Human escalation:** For high-stakes decisions, the agent can interrupt execution and ask a human for guidance using LangGraph's `interrupt()`.

---

## **Q5: "Where is agentic AI heading? What's the future?"**

**Strong Answer:**

> Several major trends are converging:
> 
> **1. Standardized tool protocols (MCP):** Anthropic's Model Context Protocol is creating a universal standard for agent-tool communication — like USB for AI tools. This will make agents truly plug-and-play with any tool server.
> 
> **2. Longer context windows + better memory:** As models handle 1M+ tokens (Gemini) and memory systems mature, agents will maintain richer context across longer interactions and complex multi-session tasks.
> 
> **3. Smaller, specialized agent models:** We're seeing a trend toward fine-tuned small models (8B parameters) that are specifically trained for tool calling and agentic reasoning, making agents cheaper and faster.
> 
> **4. Multi-agent-as-a-service:** Platforms are emerging where you can compose agents from a marketplace — pick a research agent, a coding agent, a data analysis agent, and wire them together.
> 
> **5. Agent evaluation and safety:** As agents take more autonomous actions, we'll need robust evaluation frameworks (testing agent reliability, safety guardrails, cost constraints) and standards for agent behavior.
> 
> **6. Real-world integration:** Agents will increasingly interact with real-world systems — managing cloud infrastructure, executing financial transactions, controlling robotics — moving beyond text-in/text-out.

---

## **Q6: "How would you evaluate if an agent is performing well?"**

**Strong Answer:**

> Agent evaluation is multi-dimensional. I'd track:
> 
> **1. Task Success Rate:** Did the agent complete the task correctly? Measure against ground truth or human evaluation. For open-ended tasks, use LLM-as-judge (a separate model grades the output).
> 
> **2. Tool Efficiency:** How many tool calls did the agent make vs. the minimum needed? Unnecessary tool calls waste money and time. Measure tool precision (% of useful calls).
> 
> **3. Latency:** End-to-end time from user query to final answer. Break down by step: planning time, tool execution time, reflection time.
> 
> **4. Cost:** Total tokens consumed per task. Track by model and by agent role. Set up alerts for cost anomalies.
> 
> **5. Error Recovery Rate:** When things go wrong, how often does the agent recover vs. fail completely? A good agent should recover from transient errors 95%+ of the time.
> 
> **6. Safety:** Does the agent ever produce harmful outputs, leak sensitive data, or take dangerous actions? Implement red-team testing.
> 
> I'd build an evaluation suite using datasets of representative tasks and run the agent against them on every deployment — similar to how we use test suites in traditional software.

---

## **Q7: "Explain the difference between hierarchical and decentralized multi-agent orchestration."**

**Strong Answer:**

> **Hierarchical orchestration** has a clear chain of command. A supervisor agent receives the task, decomposes it, delegates sub-tasks to specialist agents, monitors their progress, and aggregates results. Think of it like a project manager directing a team.
> 
> - *Pros:* Easy to debug (clear execution trace), natural task delegation, single point of coordination.
> - *Cons:* Supervisor is a bottleneck and single point of failure; all communication routes through it.
> 
> **Decentralized orchestration** has agents communicating peer-to-peer via a shared message bus or state store. Each agent can trigger other agents directly without a central coordinator.
> 
> - *Pros:* More resilient (no single point of failure), agents can self-organize, better for highly parallel workloads.
> - *Cons:* Harder to debug, risk of circular dependencies, coordination overhead.
> 
> **In practice, I prefer a hybrid approach:** hierarchical for high-level task orchestration (a supervisor routes major task categories) with decentralized communication within specialist teams (e.g., researcher agents collaborate peer-to-peer on sub-queries). This gives you the debuggability of hierarchy with the resilience of decentralization.

---

## **Q8: "How do you manage memory in a long-running agent system?"**

**Strong Answer:**

> I use a four-tier memory architecture:
> 
> **Short-term memory** is the conversation buffer — the last N messages that fit in the LLM's context window. This gives the agent immediate conversational context.
> 
> **Working memory** tracks the current task state — which sub-tasks are done, intermediate results, the active plan. In LangGraph, this is the `TypedDict` state that flows through the graph.
> 
> **Long-term memory** is a vector store (Pinecone, ChromaDB, pgvector) where I persist important facts, user preferences, and domain knowledge as embeddings. At each turn, I retrieve the top-K most relevant memories via semantic search and inject them into the prompt.
> 
> **Episodic memory** stores summaries of past task executions — which tools worked, which approaches failed, what the user preferred. This lets the agent learn from experience: "Last time I analyzed sales data, the user preferred bar charts over pie charts."
> 
> The key challenge is **memory retrieval quality**. Bad retrieval means the agent gets irrelevant context, which actually hurts performance. I address this with hybrid retrieval (semantic + keyword), re-ranking, and a relevance threshold — only inject memories above a similarity score of 0.75.

---

## **Q9: "What is MCP (Model Context Protocol) and how does it relate to agents?"**

**Strong Answer:**

> MCP is Anthropic's open standard for connecting AI agents to external tools and data sources — think of it as **"USB-C for AI."** Before MCP, every agent framework had its own way of defining and calling tools, creating fragmentation.
> 
> MCP defines a standardized protocol where:
> - **MCP Servers** expose tools (functions an agent can call) and resources (data an agent can read) via a standard JSON-RPC interface.
> - **MCP Clients** (agents) discover available tools dynamically by querying the server, then call them using a standard format.
> 
> This means an agent built with LangGraph can connect to the same MCP server as an agent built with CrewAI — **tool servers become reusable across frameworks.**
> 
> In my projects, I use MCP to expose FastAPI endpoints as tool servers that any agent can discover and use, making the tool layer completely decoupled from the agent layer. This is especially powerful in multi-agent systems where different agents might be built with different frameworks but need access to the same tools.

---

## **Q10: "Walk me through how you'd build a production agent from scratch."**

**Strong Answer:**

> Here's my step-by-step approach:
> 
> **1. Define the scope:** What task does the agent solve? What tools does it need? What's the expected input/output? Start narrow — a well-scoped agent beats a do-everything agent.
> 
> **2. Prototype with LangChain:** Build a simple ReAct agent with the core tools. Test it manually on 20-30 representative queries. Identify failure modes.
> 
> **3. Upgrade to LangGraph:** Once I understand the workflow, model it as a graph — add branching for different query types, loops for quality checking, parallel execution where possible.
> 
> **4. Add memory:** Short-term (conversation buffer) + long-term (vector store for persistent knowledge) + working memory (LangGraph state for task progress).
> 
> **5. Wrap in FastAPI:** Create REST + WebSocket endpoints. Add authentication, rate limiting, request validation.
> 
> **6. Add observability:** OpenTelemetry for tracing, Prometheus for metrics, LangSmith for LLM-specific traces.
> 
> **7. Build evaluation suite:** 100+ test cases covering happy paths, edge cases, and adversarial inputs. Automate this in CI/CD.
> 
> **8. Containerize and deploy:** Docker → Kubernetes. Set up auto-scaling based on queue depth. Implement health checks.
> 
> **9. Monitor and iterate:** Watch real usage patterns, identify common failures, add new tools, tune prompts. Agent development is iterative — the first version is never the last.

---

# **12. Key Takeaways**

---

```
┌────────────────────────────────────────────────────────────────────┐
│                       KEY TAKEAWAYS                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. AGENTIC AI = LLM + Autonomy + Tools + Memory                  │
│     Agents PERCEIVE, PLAN, ACT, and REFLECT in a loop.            │
│                                                                    │
│  2. FRAMEWORK SELECTION MATTERS                                    │
│     LangChain (tools/chains) → LangGraph (complex orchestration)  │
│     → CrewAI (role-based teams) → AutoGen (code execution)        │
│     Choose based on complexity, not hype.                          │
│                                                                    │
│  3. MULTI-AGENT = DIVIDE AND CONQUER                               │
│     Modular agents with clear roles, communicating via             │
│     standard protocols. Hierarchical + decentralized hybrid.       │
│                                                                    │
│  4. PATTERNS ARE YOUR PLAYBOOK                                     │
│     ReAct for reasoning + acting.                                  │
│     Router-Evaluator for quality control.                          │
│     Plan-Act-Reflect for complex tasks.                            │
│     Fan-out/Fan-in for parallelism.                                │
│                                                                    │
│  5. MEMORY IS CRITICAL                                             │
│     Short-term (buffer) + Long-term (vectors) +                   │
│     Episodic (past experiences) + Working (current task).          │
│     Bad memory retrieval hurts more than no memory.                │
│                                                                    │
│  6. PRODUCTION ≠ PROTOTYPE                                         │
│     Error handling (retries, fallbacks, circuit breakers).          │
│     Cost management (tiered models, caching, budgets).             │
│     Observability (traces, metrics, LLM-specific monitoring).      │
│     Evaluation (automated test suites, LLM-as-judge).             │
│                                                                    │
│  7. COST OPTIMIZATION = USE THE RIGHT MODEL PER TASK               │
│     GPT-4o-mini for routing ($0.15/M) vs GPT-4o for               │
│     reasoning ($2.50/M). 90%+ savings possible.                    │
│                                                                    │
│  8. MCP IS THE FUTURE OF TOOL INTEROPERABILITY                     │
│     Standard protocol for agent ↔ tool communication.              │
│     Decouple tools from frameworks. Build once, use everywhere.   │
│                                                                    │
│  9. THE DEPLOYMENT STACK IS STANDARD ENGINEERING                   │
│     FastAPI + Docker + K8s + Prometheus + GitHub Actions.           │
│     Agent-specific additions: LangSmith, token tracking,           │
│     checkpointing.                                                 │
│                                                                    │
│  10. AGENTS ARE ITERATIVE, NOT ONE-SHOT                            │
│      Start narrow, prototype fast, evaluate rigorously,            │
│      harden for production, monitor and iterate.                   │
│      The first version is never the final version.                 │
└────────────────────────────────────────────────────────────────────┘
```

---

*Last updated: February 2026 | Part of Rahul Sharma's ML/AI Interview Preparation Series*
