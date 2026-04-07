# Building a ReAct AI Agent from Scratch with LangGraph and a Free LLM

> A beginner-friendly, step-by-step tutorial on how to build an AI agent that thinks, uses tools, and loops until it finds the answer — running entirely for free.

![Agent Flow](flow.png)

---

## Who Is This For?

This tutorial is written for absolute beginners. If you have never built an AI agent before, this is your starting point. By the end you will understand:

- What a ReAct agent is and why it matters
- How an LLM decides to call a tool
- How LangGraph turns that loop into a graph
- How all the pieces fit together — with real, runnable code

The full source code is 3 small files. Let us walk through every line.

---

## What Are We Building?

We are building an AI agent that can answer this question:

> *"What is the temperature in Tokyo right now? List it and then triple it."*

A plain ChatGPT-style LLM cannot answer this — its knowledge is frozen at training time and it cannot do reliable math. Our agent will:

1. **Search the web** for Tokyo's live temperature using Tavily
2. **Triple the number** using a custom Python function
3. **Combine both results** into a natural language answer

Here is what the final output looks like:

```
Hello ReAct LangGraph with Function Calling
Model: HuggingFace (Qwen/Qwen2.5-7B-Instruct)
The current temperature in Tokyo is 28°C. Tripled, that is 84°C.
```

---

## Part 1 — The Big Idea: ReAct Pattern

**ReAct = Reason + Act**, introduced in the 2022 paper [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629).

The core insight is simple: instead of answering in one shot, let the AI **think → act → think → act** in a loop until it has everything it needs.

```
  User Question
       │
       ▼
  ┌─────────────────────┐
  │       THINK         │  ← LLM reads the question and decides:
  │  (run_agent_reason) │    "Do I need a tool, or do I know the answer?"
  └──────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
  tool needed?     no tool needed
     │                │
     ▼                ▼
  ┌──────────┐    ┌──────────┐
  │   ACT    │    │   END    │
  │(tool_node│    │  (done)  │
  └────┬─────┘    └──────────┘
       │
       │  result goes back into state
       ▼
  ┌─────────────────────┐
  │       THINK         │  ← LLM reads the tool result and thinks again
  └─────────────────────┘
       (loop repeats)
```

**Why is this powerful?**

| Without ReAct | With ReAct |
|---|---|
| LLM answers from memory only | LLM can look things up in real time |
| Cannot run code or math reliably | Can call any Python function |
| One shot — no retry | Loops until confident |

---

## Part 2 — The Building Blocks

Before writing code, understand these 5 concepts.

### 2.1 — State (The Shared Notepad)

State is the **shared memory** of the agent. Every step reads from it and writes to it. Think of it like a notepad that is passed between steps.

In our project the state is `MessagesState` — a list of all messages exchanged so far:

```
MessagesState
└── "messages": list
      │
      ├── [0] HumanMessage       ← the user's question
      │
      ├── [1] AIMessage          ← LLM says: "I need to search"
      │         .tool_calls = [{"name": "TavilySearch", "args": {...}, "id": "call_abc"}]
      │
      ├── [2] ToolMessage        ← search result
      │         .content       = "Tokyo is 28°C"
      │         .tool_call_id  = "call_abc"   ← links back to [1]
      │
      ├── [3] AIMessage          ← LLM says: "Now triple it"
      │         .tool_calls = [{"name": "triple", "args": {"num": 28.0}, "id": "call_def"}]
      │
      ├── [4] ToolMessage        ← math result
      │         .content       = "84.0"
      │         .tool_call_id  = "call_def"
      │
      └── [5] AIMessage          ← LLM writes final answer
                .content       = "Tokyo is 28°C. Tripled: 84°C."
                .tool_calls    = []    ← EMPTY = agent is done
```

Key takeaways:
- `tool_calls` is a **field inside `AIMessage`** — not a separate object
- When `tool_calls = []` (empty list), the agent stops
- `tool_call_id` links each `ToolMessage` result back to the exact tool call that requested it

### 2.2 — Nodes (The Steps)

A **node** is just a Python function. It receives state, does one job, returns updated state.

We have two nodes:

| Node | Job | File |
|------|-----|------|
| `run_agent_reasoning` | Sends messages to the LLM, gets a response | [nodes.py](nodes.py) |
| `tool_node` | Reads `tool_calls`, runs the functions, returns results | [nodes.py](nodes.py) |

### 2.3 — Edges (The Connections)

Edges connect nodes. There are two types:

- **Fixed edge** — always goes to the same next node
  ```
  ACT → always → THINK
  ```
- **Conditional edge** — a function decides where to go next
  ```
  THINK → (tool_calls present?) → ACT
  THINK → (tool_calls empty?)   → END
  ```

### 2.4 — Tool Calling (Smart Decision, Dumb Execution)

When the LLM wants to use a tool, it outputs a structured message called a `tool_call`:

```json
{
  "name": "triple",
  "args": { "num": 28.0 },
  "id":   "call_abc123"
}
```

Then `ToolNode` executes it. Here is the important distinction:

> **The LLM is smart. ToolNode is dumb.**

The LLM decides *which* tool to call and *what arguments* to pass. `ToolNode` is just a **dictionary lookup** — no intelligence whatsoever:

```
ToolNode's internal phone book (built when you pass tools):
  {
    "TavilySearch": <TavilySearch object>,
    "triple":       <triple function>
  }

When tool_call arrives with name="triple":
  → look up "triple" in the dict
  → call triple(num=28.0)
  → return 84.0
```

Like a **doctor and pharmacist**: the LLM (doctor) writes the prescription deciding what is needed. `ToolNode` (pharmacist) reads the label and hands over exactly that — no judgment.

> This is why the function name matters. If you rename `triple` to `multiply_by_three`, the `tool_call` name must match exactly.

### 2.5 — The Graph

A **graph** wires all the nodes and edges into a runnable workflow. Once compiled it works like a function: you pass in a question, it runs the loop, it returns the final state.

---

## Part 3 — Project Structure

```
.
├── react.py       ← LLM + Tools setup
├── nodes.py       ← THINK and ACT node definitions
├── main.py        ← Graph assembly and entry point
├── flow.png       ← Auto-generated visual diagram of the graph
├── pyproject.toml ← Python dependencies (managed by Poetry)
└── .env           ← Your secret API keys (never committed to git)
```

---

## Part 4 — Code Walkthrough

### 4.1 — [react.py](react.py) — LLM + Tools

This file answers two questions: *What tools does the agent have?* and *Which AI model is it using?*

#### The Tools

**Tool 1 — Web Search** (pre-built, from Tavily):

```python
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(max_results=1)
# Input:  "current temperature Tokyo"  (str)
# Output: {"results": [{"content": "Tokyo is 28°C", "score": 0.94}]}  (dict)
```

**Tool 2 — Triple a number** (custom, written by us):

```python
from langchain_core.tools import tool

@tool
def triple(num: float) -> float:
    """Multiplies the given number by 3 and returns the result."""
    return float(num) * 3

# triple(28.0)  → 84.0
# triple(5.0)   → 15.0
```

The `@tool` decorator registers the function name `"triple"` so `ToolNode` can find it by name later.

**The docstring is critical** — the LLM reads it to know when and how to call this tool. Write it clearly.

#### The Tools Registry

```python
tools = [tavily_tool, triple]
```

This list is used in two places:
1. `llm.bind_tools(tools)` — tells the LLM what tools are available
2. `ToolNode(tools)` — lets `ToolNode` actually execute them

#### The LLM Backend Switch

We support two free backends. Switch with one line in [react.py](react.py):

```python
USE_HUGGINGFACE = True   # free cloud model — needs internet
USE_HUGGINGFACE = False  # local Ollama model — private, offline
```

| | HuggingFace | Ollama |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` | `qwen3:8b` |
| Cost | Free (~1,000 req/day) | Free (unlimited) |
| Setup | HF token only | Ollama server required |
| Privacy | Data sent to HF servers | Fully private |

---

### 4.2 — [nodes.py](nodes.py) — THINK and ACT

This file defines the two nodes.

#### Node 1 — THINK (`run_agent_reasoning`)

```python
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

What happens here step by step:

1. Take the full conversation history from `state["messages"]`
2. Prepend a system message (the AI's instructions)
3. Send everything to the LLM
4. The LLM responds with either a `tool_calls` list or plain `content`
5. Return the new `AIMessage` — LangGraph appends it to state automatically

The system message sets the AI's role:

```python
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""
```

#### Node 2 — ACT (`tool_node`)

```python
tool_node = ToolNode(tools)
```

Just one line. `ToolNode` is pre-built by LangGraph. It:

1. Reads `tool_calls` from the last `AIMessage`
2. Looks up each tool name in its internal dictionary
3. Calls the matching function with the given arguments
4. Returns `ToolMessage` objects with the results

**No intelligence here** — pure name lookup and function execution.

---

### 4.3 — [main.py](main.py) — Graph Assembly

This file builds and runs the graph.

#### The Routing Function

```python
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return ACT   # non-empty list → truthy → go to ACT node
    return END       # empty list    → falsy  → stop graph
```

This is the brain of the loop. It inspects the last message:
- `tool_calls` is a non-empty list → LLM wants to run tools → continue
- `tool_calls` is `[]` empty → LLM has its final answer → stop

#### Building the Graph

```python
flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)   # step 1: THINK
flow.add_node(ACT, tool_node)                       # step 2: ACT

flow.set_entry_point(AGENT_REASON)                  # always start by thinking

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)                    # after acting, think again

app = flow.compile()                                # lock and make runnable
```

The graph looks like this (also saved to [flow.png](flow.png)):

```
[START]
   │
   ▼
[agent_reason] ──── tool_calls? YES ────► [act]
       ▲                                    │
       └────────────────────────────────────┘
                   (loop back)

[agent_reason] ──── tool_calls? NO  ────► [END]
```

#### Running the Agent

```python
res = app.invoke({
    "messages": [HumanMessage(content="What is the temperature in Tokyo? Triple it.")]
})
print(res["messages"][-1].content)
```

`app.invoke()` runs the full loop and returns the final state. `res["messages"][-1]` is always the last `AIMessage` — the final answer.

---

## Part 5 — Setup Guide

### Step 1 — Install Poetry

Poetry manages Python dependencies for this project.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add to `~/.zshrc`:

```bash
export PATH="/Users/$USER/.local/bin:$PATH"
```

Reload:

```bash
source ~/.zshrc
```

> **macOS tip:** If you get a venv symlink error with system Python, use Conda:
> ```bash
> curl -sSL https://install.python-poetry.org | /opt/miniconda3/bin/python3 -
> ```

> **Path tip:** If `poetry` is still not found, call it directly:
> ```bash
> /Users/$USER/.local/bin/poetry --version
> ```

### Step 2 — Install Dependencies

```bash
poetry install
```

All packages are declared in [pyproject.toml](pyproject.toml) and installed automatically — including `langchain`, `langgraph`, `langchain-ollama`, `langchain-huggingface`, `langchain-tavily`, and `python-dotenv`.

### Step 3 — Create Your Free Accounts

You need three free accounts. Here is exactly how to set each one up.

---

#### Tavily — Real-Time Web Search

Tavily is a search engine built for AI agents. It returns clean text results instead of raw HTML.

1. Sign up free at **tavily.com**
2. Go to **Dashboard → API Keys**
3. Copy your key — starts with `tvly-`
4. Free tier: **1,000 searches/month**

---

#### LangSmith — Agent Tracing

LangSmith records every step of your agent's run in a visual timeline. You can see exactly which tool was called, what it returned, and how long it took. Invaluable for debugging.

1. Sign up free at **smith.langchain.com**
2. Go to **Settings → API Keys → Create API Key**
3. Copy your key — starts with `lsv2_`
4. Free tier: **unlimited traces for personal use**

---

#### HuggingFace — Free LLM Inference

HuggingFace hosts thousands of open-source AI models and lets you run them via API for free.

1. Sign up free at **huggingface.co** (no credit card needed)
2. Go to **Settings → Access Tokens**
3. Click **New token** → name it (e.g. `react-agent`)
4. Enable **"Make calls to Inference Providers"** ← required, not enabled by default
5. Copy your token — starts with `hf_`
6. Free tier: **~1,000 requests/day** (~200 full agent runs)

> **Common error:** If you see `403 Forbidden`, your token is missing the
> "Make calls to Inference Providers" permission. Edit the token on HuggingFace and enable it.

---

#### Configure `.env`

Create `.env` in the project root:

```env
TAVILY_API_KEY=tvly-your-key-here

LANGSMITH_API_KEY=lsv2_your-key-here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="personal"

# Only needed if USE_HUGGINGFACE = True in react.py
HUGGINGFACEHUB_API_TOKEN=hf_your-token-here
```

> Your `.env` is in `.gitignore` — these keys are never committed to git.

### Step 4 — Choose Your LLM Backend

Open [react.py](react.py) and set the flag:

```python
USE_HUGGINGFACE = True   # free cloud (recommended for beginners)
USE_HUGGINGFACE = False  # local Ollama (recommended for privacy)
```

If using Ollama, verify the server is reachable:

```bash
curl http://192.168.10.114:11434/api/tags
# qwen3:8b must appear in the output
```

---

## Part 6 — Run It

```bash
poetry run python main.py
```

Expected output:

```
Hello ReAct LangGraph with Function Calling
Model: HuggingFace (Qwen/Qwen2.5-7B-Instruct)
The current temperature in Tokyo is 28°C. Tripled, that is 84°C.
```

### Try Your Own Question

Edit the `HumanMessage` in [main.py](main.py):

```python
HumanMessage(content="Your question here")
```

Examples to try:
```python
"What is the current Bitcoin price? Triple it."
"What is the population of Seoul? Triple it."
"What did NASA announce this week?"
```

---

## Part 7 — How It All Fits Together

Here is the complete picture of one agent run, end to end:

```
You type a question
        │
        ▼
  HumanMessage added to state["messages"]
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  THINK (run_agent_reasoning)            │
  │  LLM reads: system + all messages       │
  │  LLM output: AIMessage                  │
  │    └── .tool_calls = [TavilySearch]     │ ← not empty → continue
  └──────────────────┬──────────────────────┘
                     │
        AIMessage appended to state
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  ACT (tool_node)                        │
  │  reads tool_call name → "TavilySearch"  │
  │  looks up in dict → TavilySearch object │
  │  calls TavilySearch("Tokyo temp")       │
  │  returns ToolMessage("Tokyo is 28°C")   │
  └──────────────────┬──────────────────────┘
                     │
        ToolMessage appended to state
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  THINK again                            │
  │  LLM reads: all messages including      │
  │             the search result           │
  │  LLM output: AIMessage                  │
  │    └── .tool_calls = [triple(28.0)]     │ ← not empty → continue
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  ACT again                              │
  │  calls triple(num=28.0) → 84.0          │
  │  returns ToolMessage("84.0")            │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  THINK again                            │
  │  LLM reads: all messages                │
  │  LLM output: AIMessage                  │
  │    └── .content    = "28°C. Tripled: 84°C."
  │    └── .tool_calls = []                 │ ← EMPTY → END
  └──────────────────┬──────────────────────┘
                     │
                     ▼
              print final answer
```

---

## Further Reading

| Topic | Link |
|-------|------|
| ReAct paper (2022) | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| LangGraph docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| LangChain tool calling | [python.langchain.com/docs/concepts/tool_calling](https://python.langchain.com/docs/concepts/tool_calling/) |
| Ollama model library | [ollama.com/library](https://ollama.com/library) |
| HuggingFace models | [huggingface.co/models](https://huggingface.co/models) |
| Tavily docs | [docs.tavily.com](https://docs.tavily.com) |
