# ============================================================
# nodes.py — Graph Node Definitions
# ============================================================
#
# WHAT IS THIS FILE?
# This file defines the two "steps" of our AI agent's loop.
# Each step is called a NODE in LangGraph.
#
# WHAT IS A NODE?
# A node is just a Python function that:
#   1. Receives the current STATE (the conversation so far)
#   2. Does something (calls the LLM, runs a tool, etc.)
#   3. Returns updated STATE (adds new messages to the conversation)
#
# WHAT IS STATE?
# State is the shared memory of the graph — it is passed from
# node to node as the agent runs. In our case, the state is
# simply a list of messages (the conversation history).
#
# Think of state like a notepad that every node can read and
# write to. Each node adds its output to the notepad before
# passing it to the next node.
#
# OUR TWO NODES:
#   1. run_agent_reasoning  → the "THINK" step
#      Sends the conversation to the LLM and gets a response.
#      The LLM either calls a tool OR gives a final answer.
#
#   2. tool_node            → the "ACT" step
#      Reads the tool call from the LLM's response, runs the
#      actual Python function, and returns the result.
#
# THE LOOP:
#   THINK → (tool needed?) → ACT → THINK → ... → final answer
# ============================================================


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from dotenv import load_dotenv
# Loads secret keys from the .env file into memory.

from langgraph.graph import MessagesState
# MessagesState is a pre-built "state" data type from LangGraph.
# It is a dictionary with one key: "messages"
# The value is a list of messages (the conversation history).
#
# What does MessagesState look like?
# {
#   "messages": [
#     HumanMessage(content="What is Tokyo's temperature?"),
#     AIMessage(content="Let me search for that.", tool_calls=[...]),
#     ToolMessage(content="Tokyo is 28°C", tool_call_id="abc"),
#     AIMessage(content="The temperature is 28°C.")
#   ]
# }
#
# LangGraph automatically APPENDS new messages to the list each
# time a node returns {"messages": [new_message]}.
# You do not need to manage the list manually.

from langgraph.prebuilt import ToolNode
# ToolNode is a ready-made node provided by LangGraph.
# It automatically:
#   1. Reads the tool_calls from the last AIMessage
#   2. Finds the matching Python function in our tools list
#   3. Calls the function with the given arguments
#   4. Wraps the result in a ToolMessage and returns it
# We do not need to write this logic ourselves.
#
# HOW DOES IT FIND THE MATCHING FUNCTION? — Pure name lookup, NO intelligence.
#
# When you write @tool above a function, LangGraph registers the
# function's name as the tool's identifier:
#
#   @tool
#   def triple(num: float) -> float:   ← "triple" becomes the tool name
#       ...
#
# When ToolNode(tools) is created, it builds an internal dictionary
# (like a phone book / lookup table):
#
#   {
#     "TavilySearch": <TavilySearch object>,
#     "triple":       <triple function>
#   }
#
# When an AIMessage arrives with a tool_call:
#   AIMessage(tool_calls=[
#     {"name": "triple", "args": {"num": 28.0}, "id": "call_abc"}
#   ])
#
# ToolNode does this — simplified:
#   tool_name = tool_call["name"]           # "triple"
#   tool_fn   = tool_map[tool_name]         # looks up "triple" in the dict
#   result    = tool_fn.invoke({"num": 28}) # calls it → 84.0
#
# That is it. Zero thinking. Just: read name → find function → call it.
#
# ALL the intelligence lives in the LLM (the THINK step).
# The LLM decides WHICH tool to call and WHAT arguments to pass.
# ToolNode is just a dumb dispatcher — like a pharmacist who reads
# the name on the prescription label and hands over the exact medicine.
# No judgment, no thinking.
#
#   LLM (smart)              ToolNode (dumb)
#   ──────────────────────   ──────────────────────────────────
#   "I need to triple 28" →  name="triple", args={"num": 28.0}
#                            → dict lookup: "triple" → triple()
#                            → triple(num=28.0) → 84.0
#                            → ToolMessage(content="84.0")
#
# IMPORTANT: the name in tool_call MUST exactly match the function name.
# You can verify any tool's name like this:
#   print(triple.name)        → "triple"
#   print(tavily_tool.name)   → "tavily_search"

from react import llm, tools
# Import the LLM and tools we configured in react.py.
# llm   = the AI model (HuggingFace or Ollama)
# tools = the list of tools [tavily_tool, triple]

load_dotenv()


# ------------------------------------------------------------
# SYSTEM MESSAGE
# ------------------------------------------------------------
# TYPE: str  (a string — text surrounded by quotes)
#
# WHAT IS A SYSTEM MESSAGE?
# In a chat with an AI, there are three types of messages:
#
#   "system"    → Instructions for the AI. The user never
#                 sees this. It sets the AI's personality,
#                 role, and rules.
#                 Example: "You are a helpful assistant."
#
#   "user"      → The human's message.
#                 Example: "What is the weather in Tokyo?"
#
#   "assistant" → The AI's reply.
#                 Example: "The weather in Tokyo is 28°C."
#
# We prepend the system message to EVERY call to the LLM so
# the model always remembers its role, no matter how long the
# conversation gets.
#
# Example of what the full message list looks like each call:
#   [
#     {"role": "system",    "content": "You are a helpful assistant..."},
#     {"role": "user",      "content": "What is Tokyo's temperature?"},
#     {"role": "assistant", "content": "", "tool_calls": [...]},
#     {"role": "tool",      "content": "Tokyo is 28°C"},
#   ]
# ------------------------------------------------------------
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


# ------------------------------------------------------------
# NODE 1 — run_agent_reasoning (the "THINK" step)
# ------------------------------------------------------------
#
# WHAT DOES THIS NODE DO?
# It sends the full conversation history to the LLM and gets
# back the LLM's response.
#
# The LLM can respond in ONE of two ways:
#
#   WAY 1 — Tool Call (the LLM wants more information):
#     The LLM returns an AIMessage with a tool_calls field.
#     This tells the graph: "Run this tool for me and come back."
#     Example AIMessage with tool_call:
#       AIMessage(
#         content="",
#         tool_calls=[
#           {
#             "name": "TavilySearch",
#             "args": {"query": "current temperature Tokyo"},
#             "id": "call_abc123"
#           }
#         ]
#       )
#
#   WAY 2 — Final Answer (the LLM has enough information):
#     The LLM returns an AIMessage with plain text content.
#     This tells the graph: "I have my answer, we're done."
#     Example AIMessage with final answer:
#       AIMessage(
#         content="The temperature in Tokyo is 28°C. Tripled, that is 84°C.",
#         tool_calls=[]     ← empty list = no tools needed
#       )
#
# PARAMETERS:
#   state (MessagesState) — the current conversation history.
#     Example:
#       state = {
#         "messages": [
#           HumanMessage(content="What is Tokyo's temperature? Triple it.")
#         ]
#       }
#
# RETURNS:
#   dict with key "messages" containing a list with one new message.
#   LangGraph appends this to the existing messages automatically.
#   Example return:
#     {
#       "messages": [
#         AIMessage(tool_calls=[{"name": "TavilySearch", ...}])
#       ]
#     }
# ------------------------------------------------------------
def run_agent_reasoning(state: MessagesState) -> MessagesState:

    # Build the full message list for the LLM:
    # [system_message] + all previous messages in state
    #
    # The * (star/splat) operator unpacks a list into individual items.
    # Example:
    #   existing = [msg1, msg2]
    #   combined = [system, *existing]
    #   → combined = [system, msg1, msg2]
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]

    # Send the messages to the LLM and get a response.
    # response is an AIMessage object.
    # It will have either .tool_calls (if LLM wants to use a tool)
    # or .content (if LLM has a final answer).
    response = llm.invoke(messages)

    # Return the new message wrapped in a list.
    # LangGraph automatically appends it to state["messages"].
    return {"messages": [response]}


# ------------------------------------------------------------
# NODE 2 — tool_node (the "ACT" step)
# ------------------------------------------------------------
#
# WHAT DOES THIS NODE DO?
# It looks at the last AIMessage in state, finds the tool_calls
# inside it, runs each tool function, and returns the results
# as ToolMessage objects.
#
# EXAMPLE FLOW:
#
#   Input state (last message has tool_calls):
#     state["messages"][-1] = AIMessage(
#       tool_calls=[
#         {"name": "TavilySearch", "args": {"query": "Tokyo temp"}, "id": "call_aaa"},
#         {"name": "triple",       "args": {"num": 28.0},           "id": "call_bbb"}
#       ]
#     )
#
#   ToolNode runs:
#     → TavilySearch("Tokyo temp")  → "Tokyo is currently 28°C"
#     → triple(28.0)                → 84.0
#
#   Output (two ToolMessages added to state):
#     ToolMessage(content="Tokyo is currently 28°C", tool_call_id="call_aaa")
#     ToolMessage(content="84.0",                    tool_call_id="call_bbb")
#
# WHY tool_call_id?
# Each tool_call has a unique ID (e.g. "call_aaa").
# The ToolMessage uses the same ID so the LLM can match each
# result back to the correct tool call it made.
#
# ToolNode handles ALL of this automatically — we just pass
# our tools list and it takes care of the rest.
# ------------------------------------------------------------
tool_node = ToolNode(tools)
