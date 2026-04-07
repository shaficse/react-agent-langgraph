# ============================================================
# main.py — Graph Assembly & Entry Point
# ============================================================
#
# WHAT IS THIS FILE?
# This is the "main" file — the one you run to start the agent.
# It has two jobs:
#   1. BUILD the graph — connect all the nodes (think + act)
#      into a complete, runnable workflow.
#   2. RUN the agent — send a question and print the answer.
#
# WHAT IS A GRAPH?
# A graph is a way to describe a workflow as:
#   - NODES  = steps / actions (e.g. "think", "act")
#   - EDGES  = connections between steps (arrows)
#
# Think of it like a flowchart:
#
#   [START]
#      │
#      ▼
#   [THINK]  ←──────────────────────┐
#      │                            │
#      ├── tool needed? YES ──► [ACT]
#      │
#      └── tool needed? NO ──► [END]  (print final answer)
#
# The agent keeps looping THINK → ACT → THINK until
# the LLM decides it has enough information to answer.
#
# WHAT IS STATE?
# State is the shared memory of the graph.
# In our case: a list of all messages exchanged so far.
# Every node reads this list and adds new messages to it.
#
# EXAMPLE — full state after one complete run:
#   messages = [
#     HumanMessage("What is Tokyo's temperature? Triple it."),  ← user's question
#     AIMessage(tool_calls=[TavilySearch("Tokyo temp")]),       ← LLM: "I need to search"
#     ToolMessage("Tokyo is 28°C"),                             ← search result
#     AIMessage(tool_calls=[triple(28.0)]),                     ← LLM: "Now triple it"
#     ToolMessage("84.0"),                                      ← math result
#     AIMessage("The temperature is 28°C. Tripled: 84°C.")      ← final answer ← LAST
#   ]
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from dotenv import load_dotenv
# Loads API keys from the .env file.

from langchain_core.messages import HumanMessage
# HumanMessage represents a message from the user (you).
# TYPE: HumanMessage
#   .content (str) — the text of your question
# Example:
#   HumanMessage(content="What is the weather in Tokyo?")
#   → role: "user", content: "What is the weather in Tokyo?"

from langgraph.graph import MessagesState, StateGraph, END
# StateGraph — the main graph builder object.
#              You add nodes and edges to it, then compile it.
# MessagesState — the state schema: {"messages": list[BaseMessage]}
# END — a special string ("__end__") that tells LangGraph to stop.
#       When should_continue() returns END, the graph finishes.

from nodes import run_agent_reasoning, tool_node
# Import our two nodes from nodes.py:
#   run_agent_reasoning → the THINK step (calls the LLM)
#   tool_node           → the ACT step (runs the tools)

load_dotenv()


# ------------------------------------------------------------
# NODE NAME CONSTANTS
# ------------------------------------------------------------
# TYPE: str  (text strings used as identifiers)
#
# We give each node a name (a string) when we add it to the graph.
# By storing these names as constants, we avoid typos.
# If we typed "agent_reson" by mistake, Python would not warn us.
# A constant lets us reuse the same name safely everywhere.
#
# Examples of what these strings look like:
#   AGENT_REASON = "agent_reason"  → name of the thinking node
#   ACT          = "act"           → name of the acting node
#   LAST         = -1              → in Python, -1 means the last
#                                    item in a list
#                                    e.g. [a, b, c][-1] → c
# ------------------------------------------------------------
AGENT_REASON = "agent_reason"
ACT          = "act"
LAST         = -1


# ------------------------------------------------------------
# ROUTING FUNCTION — should_continue()
# ------------------------------------------------------------
#
# WHAT IS A ROUTING FUNCTION?
# After the THINK step runs, we need to decide:
#   - Did the LLM call a tool? → go to ACT
#   - Did the LLM give a final answer? → go to END
#
# This function makes that decision by looking at the last
# message in state and checking if it has tool_calls.
#
# WHAT IS A CONDITIONAL EDGE?
# A normal edge always goes to the same next node.
# A conditional edge uses a function (this one!) to decide
# which node to go to next. That is why we call it "conditional"
# — the destination depends on a condition (if/else logic).
#
# TYPE: MessagesState → str
#   Input:  the current state (conversation history)
#   Output: a string — the name of the next node to visit
#
# EXAMPLE 1 — LLM called a tool (tool_calls is not empty):
#   state["messages"][-1] = AIMessage(
#     tool_calls=[{"name": "TavilySearch", "args": {...}}]
#   )
#   → last_message.tool_calls is truthy (has items) → return "act"
#
# EXAMPLE 2 — LLM gave a final answer (tool_calls is empty):
#   state["messages"][-1] = AIMessage(
#     content="The temperature is 28°C. Tripled: 84°C.",
#     tool_calls=[]    ← empty list = falsy in Python
#   )
#   → last_message.tool_calls is falsy (empty) → return END
#
# What is "truthy" and "falsy"?
#   In Python, some values are treated as True/False in an if:
#   Truthy: any non-empty list [1,2,3], non-zero number, non-empty string
#   Falsy:  empty list [], zero 0, empty string "", None
#   So  if []:  → does NOT run  (empty list = falsy)
#       if [1]: → DOES run      (non-empty list = truthy)
# ------------------------------------------------------------
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][LAST]  # get the most recent message

    if last_message.tool_calls:
        # tool_calls is a non-empty list → LLM wants to run a tool
        return ACT      # → go to the "act" node

    # tool_calls is empty or None → LLM gave its final answer
    return END          # → stop the graph


# ------------------------------------------------------------
# BUILD THE GRAPH
# ------------------------------------------------------------
#
# WHAT IS StateGraph?
# StateGraph is LangGraph's graph builder.
# We give it our state schema (MessagesState) so it knows what
# the state looks like. Then we add nodes and edges to it.
#
# STEPS TO BUILD A GRAPH:
#   1. Create the graph object
#   2. Add nodes (the steps)
#   3. Set the entry point (where to start)
#   4. Add edges (the connections between steps)
#   5. Compile (lock the graph and make it runnable)
# ------------------------------------------------------------

# Step 1 — Create the graph
# MessagesState tells LangGraph our state is a dict with "messages"
flow = StateGraph(MessagesState)

# Step 2 — Add nodes
# add_node(name, function):
#   name     = a string identifier for this node
#   function = the Python function to run at this step
flow.add_node(AGENT_REASON, run_agent_reasoning)  # "agent_reason" → THINK step
flow.add_node(ACT, tool_node)                      # "act"          → ACT  step

# Step 3 — Set entry point
# The graph always starts at this node.
# Our agent should always start by THINKING first.
flow.set_entry_point(AGENT_REASON)

# Step 4a — Add a CONDITIONAL edge from AGENT_REASON
# After thinking, the graph calls should_continue() to decide
# where to go next.
#
# The mapping dict explains what each return value means:
#   if should_continue() returns END  → go to END  (stop)
#   if should_continue() returns "act" → go to "act" node
flow.add_conditional_edges(
    AGENT_REASON,       # FROM this node
    should_continue,    # call this function to decide where to go
    {END: END, ACT: ACT}  # map return values to node names
)

# Step 4b — Add an UNCONDITIONAL edge from ACT back to AGENT_REASON
# After running a tool, ALWAYS go back to thinking.
# This closes the ReAct loop:  THINK → ACT → THINK → ACT → ...
flow.add_edge(ACT, AGENT_REASON)


# ------------------------------------------------------------
# COMPILE THE GRAPH
# ------------------------------------------------------------
#
# compile() does two things:
#   1. Validates the graph — checks for missing nodes, bad edges, etc.
#   2. Returns a runnable app object (like a compiled program)
#
# After compile(), the graph is immutable (cannot be changed).
# You interact with it using:
#   app.invoke(input)  → run the graph and return the final state
#   app.stream(input)  → run step-by-step and yield each state
#
# draw_mermaid_png() saves a picture of the graph to flow.png
# so you can visually see the nodes and edges. Open the file
# in any image viewer.
# ------------------------------------------------------------
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# ------------------------------------------------------------
# RUN THE AGENT
# ------------------------------------------------------------
#
# if __name__ == "__main__":
#   This Python idiom means: "only run the code below if this
#   file is run directly (not imported by another file)."
#
#   When you run:  python main.py  → __name__ is "__main__" → runs
#   When imported: from main import app → __name__ is "main" → skips
# ------------------------------------------------------------
if __name__ == "__main__":
    from react import USE_HUGGINGFACE

    # Print which model is active so we know what is running
    backend = "HuggingFace (Qwen/Qwen2.5-7B-Instruct)" if USE_HUGGINGFACE else "Ollama (qwen3:8b @ 192.168.10.114)"
    print(f"Hello ReAct LangGraph with Function Calling")
    print(f"Model: {backend}")

    # app.invoke() starts the graph and runs until END.
    #
    # INPUT TYPE: dict matching MessagesState schema
    #   {"messages": [HumanMessage(...)]}
    #
    # We wrap the user's question in a HumanMessage.
    # HumanMessage just adds the "role: user" tag to the text.
    #
    # Example question breakdown:
    #   "What is the current temperature in Tokyo in Celsius?" → triggers TavilySearch
    #   "Also convert it to Fahrenheit"                        → triggers celsius_to_fahrenheit()
    #
    # The graph will loop:
    #   1. THINK → LLM calls TavilySearch("Tokyo temperature")
    #   2. ACT   → search runs, returns "Tokyo is 28°C"
    #   3. THINK → LLM calls celsius_to_fahrenheit(28.0)
    #   4. ACT   → conversion runs, returns 82.4
    #   5. THINK → LLM writes final answer, no tool_calls → END
    res = app.invoke({
        "messages": [
            HumanMessage(content="What is the current temperature in Tokyo in Celsius? Also convert it to Fahrenheit.")
        ]
    })

    # res is the final state: {"messages": [...all messages...]}
    #
    # res["messages"] is a list of all messages in order:
    #   [0] HumanMessage  — the user's question
    #   [1] AIMessage     — LLM calls TavilySearch
    #   [2] ToolMessage   — search result: "Tokyo is 28°C"
    #   [3] AIMessage     — LLM calls celsius_to_fahrenheit(28.0)
    #   [4] ToolMessage   — conversion result: "82.4"
    #   [5] AIMessage     — final answer  ← this is LAST (-1)
    #
    # .content is the text of the message (a str)
    # Example: "Tokyo is currently 28°C, which is 82.4°F."
    print(res["messages"][LAST].content)
