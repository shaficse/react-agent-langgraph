# ============================================================
# main.py — Graph Assembly & Entry Point
# ============================================================
# THEORY: What is LangGraph?
# LangGraph is a library for building stateful, multi-step AI
# workflows as directed graphs. Think of it like a flowchart
# where each box (node) is a function and each arrow (edge)
# is a transition rule.
#
# Key concepts:
#   StateGraph    — the graph object; holds nodes + edges
#   MessagesState — shared state passed between nodes;
#                   essentially a list of chat messages
#   Conditional Edge — an edge whose target depends on the
#                      output of a routing function
#   END           — special sentinel that terminates the graph
#
# ReAct Loop (Reason + Act):
#   ┌─────────────────────┐
#   │   agent_reason      │  ← LLM thinks: tool call or done?
#   └────────┬────────────┘
#            │ tool_calls present?
#      YES ──┘         └── NO → END (return final answer)
#            ↓
#   ┌─────────────────────┐
#   │       act           │  ← execute the tool(s)
#   └────────┬────────────┘
#            │ always loop back
#            └──────────────→ agent_reason
# ============================================================

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()


# ------------------------------------------------------------
# Node name constants
# ------------------------------------------------------------
# TYPE: str
# Used as node identifiers when registering and connecting nodes.
# Defining them as constants avoids typos from raw strings.
#
# Example:
#   AGENT_REASON = "agent_reason"  →  flow.add_node("agent_reason", ...)
#   ACT          = "act"           →  flow.add_node("act", ...)
# ------------------------------------------------------------
AGENT_REASON = "agent_reason"   # reasoning node identifier
ACT = "act"                      # tool-execution node identifier
LAST = -1                        # index shorthand for the latest message in the list


# ------------------------------------------------------------
# Routing Function (Conditional Edge)
# ------------------------------------------------------------
# THEORY: A conditional edge is a function that inspects the
# current state and returns the NAME of the next node to visit.
# This is how branching logic works in LangGraph.
#
# TYPE: MessagesState → str
#
# Input — current graph state (messages list):
#   state["messages"] = [
#     HumanMessage(content="What is the temperature in Tokyo?"),
#     AIMessage(tool_calls=[{"name": "TavilySearch", ...}])   ← has tool_calls → return "act"
#   ]
#
#   state["messages"] = [
#     HumanMessage(content="What is the temperature in Tokyo?"),
#     AIMessage(content="The temperature is 28°C.")            ← no tool_calls → return END
#   ]
#
# Return values:
#   "act" (str) → route to the tool execution node
#   END   (str) → "__end__" — stop the graph, return final state
# ------------------------------------------------------------
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][LAST]

    if last_message.tool_calls:
        # LLM requested one or more tool calls → execute them
        return ACT

    # No tool calls → LLM gave a final answer → stop
    return END


# ------------------------------------------------------------
# Build the Graph
# ------------------------------------------------------------
# TYPE: StateGraph
# StateGraph takes a state schema (MessagesState here) and lets
# you register nodes and edges to define the execution flow.
#
# Graph structure after all add_node / add_edge calls:
#
#   [START] → agent_reason ──(tool_calls?)──► act
#                  ▲                           │
#                  └───────────────────────────┘
#             (loop back)
#
#   agent_reason ──(no tool_calls)──► [END]
# ------------------------------------------------------------
flow = StateGraph(MessagesState)

# Register nodes — each node is a callable that transforms state
flow.add_node(AGENT_REASON, run_agent_reasoning)  # "Think" node
flow.add_node(ACT, tool_node)                      # "Act"   node

# Set the entry point — where execution begins
flow.set_entry_point(AGENT_REASON)

# Conditional edge from reasoning node:
#   should_continue() decides whether we go to ACT or END
#
# Mapping dict — keys are possible return values of should_continue():
#   { END: END, ACT: ACT }
#   e.g. if should_continue() returns "act"      → go to node "act"
#        if should_continue() returns "__end__"  → stop graph
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {END: END, ACT: ACT},
)

# Unconditional edge: after acting, always reason again
# This closes the ReAct loop
flow.add_edge(ACT, AGENT_REASON)


# ------------------------------------------------------------
# Compile the Graph
# ------------------------------------------------------------
# TYPE: CompiledStateGraph (runnable, like a function)
# compile() validates the graph structure and returns a runnable
# object. After this point the graph is immutable.
# draw_mermaid_png() generates a visual diagram saved to flow.png
# ------------------------------------------------------------
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# ------------------------------------------------------------
# Run the Agent
# ------------------------------------------------------------
if __name__ == "__main__":
    from react import USE_HUGGINGFACE
    backend = "HuggingFace (Qwen/Qwen2.5-7B-Instruct)" if USE_HUGGINGFACE else "Ollama (qwen3:8b @ 192.168.10.114)"
    print(f"Hello ReAct LangGraph with Function Calling")
    print(f"Model: {backend}")

    # TYPE: dict  →  { "messages": list[BaseMessage] }
    # Input to app.invoke() must match the state schema (MessagesState).
    # We pass one HumanMessage to kick off the conversation.
    #
    # HumanMessage example:
    #   HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")
    #   → role: "user", content: str
    #
    # The graph runs the ReAct loop until should_continue() returns END.
    res = app.invoke({
        "messages": [
            HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")
        ]
    })

    # res["messages"] → list[BaseMessage]  (full conversation history)
    # TYPE: list[-1]  → AIMessage
    #   .content  str  → the final text answer from the LLM
    #
    # Example:
    #   res["messages"] = [
    #     HumanMessage(content="What is the temperature..."),
    #     AIMessage(tool_calls=[{"name": "TavilySearch", ...}]),
    #     ToolMessage(content="Tokyo is 28°C"),
    #     AIMessage(tool_calls=[{"name": "triple", "args": {"num": 28.0}}]),
    #     ToolMessage(content="84.0"),
    #     AIMessage(content="The temperature in Tokyo is 28°C. Tripled, that is 84°C.")  ← LAST
    #   ]
    print(res["messages"][LAST].content)
