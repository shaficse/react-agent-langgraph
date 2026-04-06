# ============================================================
# nodes.py — Graph Node Definitions
# ============================================================
# THEORY: What is a Node in LangGraph?
# A LangGraph graph is made of NODES and EDGES.
# Each node is a Python function that:
#   1. Receives the current graph STATE
#   2. Does some work (call LLM, run a tool, transform data…)
#   3. Returns a dict of state UPDATES
#
# This file defines the two core nodes of our ReAct agent:
#   - run_agent_reasoning : the "Think" step  (calls the LLM)
#   - tool_node           : the "Act"  step  (executes tools)
# ============================================================

from dotenv import load_dotenv
from langgraph.graph import MessagesState   # built-in state schema: {"messages": [...]}
from langgraph.prebuilt import ToolNode     # pre-built node that runs tool_calls automatically

from react import llm, tools               # our configured LLM and tools list

load_dotenv()


# ------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------
# The system message sets the agent's persona and behaviour.
# It is prepended to every LLM call so the model always has
# context about its role, regardless of which turn it is on.
# ------------------------------------------------------------
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


# ------------------------------------------------------------
# NODE 1 — Agent Reasoning ("Think" step)
# ------------------------------------------------------------
# THEORY: In the ReAct pattern, "Reason" means the model looks
# at the conversation history and decides what to do next:
#   a) Call a tool  → output contains tool_calls
#   b) Give a final answer → output is plain text
#
# MessagesState is a TypedDict with one key: "messages".
# LangGraph automatically appends returned messages to the list,
# so we just return {"messages": [new_message]}.
# ------------------------------------------------------------
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    # Prepend the system message to the full conversation history
    # so the LLM always has its instructions at the top of context.
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]

    # Call the LLM. The response is either:
    #   - An AIMessage with .tool_calls  → agent wants to use a tool
    #   - An AIMessage with .content     → agent has a final answer
    response = llm.invoke(messages)

    return {"messages": [response]}


# ------------------------------------------------------------
# NODE 2 — Tool Execution ("Act" step)
# ------------------------------------------------------------
# THEORY: Once the LLM decides to call a tool, the ToolNode
# reads the tool_calls from the last AIMessage, executes the
# matching Python function(s), and returns ToolMessage(s)
# containing the results. These results are appended to the
# state and fed back to the LLM in the next reasoning step.
#
# ToolNode handles multiple parallel tool calls automatically.
# ------------------------------------------------------------
tool_node = ToolNode(tools)
