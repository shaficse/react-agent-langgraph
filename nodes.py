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
# TYPE: str
# The system message sets the agent's persona and behaviour.
# It is prepended to every LLM call so the model always has
# context about its role, regardless of which turn it is on.
#
# Example of what the full message list looks like each call:
#   [
#     {"role": "system",    "content": "You are a helpful assistant..."},
#     {"role": "user",      "content": "What is the temperature in Tokyo?"},
#     {"role": "assistant", "content": None, "tool_calls": [{"name": "TavilySearch", ...}]},
#     {"role": "tool",      "content": "Tokyo is currently 28°C"},
#   ]
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
# TYPE: MessagesState → MessagesState
#
# Input state example:
#   {
#     "messages": [
#       HumanMessage(content="What is the temperature in Tokyo?")
#     ]
#   }
#
# Output when LLM decides to call a tool:
#   {
#     "messages": [
#       AIMessage(
#         content="",
#         tool_calls=[{"name": "TavilySearch",
#                      "args": {"query": "current temperature Tokyo"},
#                      "id": "call_abc123"}]
#       )
#     ]
#   }
#
# Output when LLM gives a final answer (no tool needed):
#   {
#     "messages": [
#       AIMessage(content="The temperature in Tokyo is 28°C. Tripled, that is 84°C.")
#     ]
#   }
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
# TYPE: MessagesState → MessagesState
#
# Input — last message must be an AIMessage with tool_calls:
#   {
#     "messages": [
#       ...,
#       AIMessage(tool_calls=[
#         {"name": "TavilySearch", "args": {"query": "Tokyo temperature"}, "id": "call_abc123"},
#         {"name": "triple",       "args": {"num": 28.0},                  "id": "call_def456"}
#       ])
#     ]
#   }
#
# Output — one ToolMessage per tool call:
#   {
#     "messages": [
#       ToolMessage(content="Tokyo is 28°C",  tool_call_id="call_abc123"),
#       ToolMessage(content="84.0",           tool_call_id="call_def456")
#     ]
#   }
#
# ToolNode handles multiple parallel tool calls automatically.
# ------------------------------------------------------------
tool_node = ToolNode(tools)
