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
#   StateGraph  — the graph object; holds nodes + edges
#   MessagesState — shared state passed between nodes;
#                   essentially a list of chat messages
#   Conditional Edge — an edge whose target depends on the
#                      output of a routing function
#   END         — special sentinel that terminates the graph
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
# Node name constants — avoids magic strings throughout code
# ------------------------------------------------------------
AGENT_REASON = "agent_reason"   # reasoning node identifier
ACT = "act"                      # tool-execution node identifier
LAST = -1                        # index shorthand for the latest message


# ------------------------------------------------------------
# Routing Function (Conditional Edge)
# ------------------------------------------------------------
# THEORY: A conditional edge is a function that inspects the
# current state and returns the NAME of the next node to visit.
# This is how branching logic works in LangGraph.
#
# Here we check the last message:
#   - If it contains tool_calls → the LLM wants to act → go to "act"
#   - If not → the LLM produced a final answer → END the graph
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
flow = StateGraph(MessagesState)  # create a graph whose state is a messages list

# Register nodes — each node is a callable that transforms state
flow.add_node(AGENT_REASON, run_agent_reasoning)  # "Think" node
flow.add_node(ACT, tool_node)                      # "Act"   node

# Set the entry point — where execution begins
flow.set_entry_point(AGENT_REASON)

# Conditional edge from reasoning node:
#   should_continue() decides whether we go to ACT or END
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
# compile() validates the graph structure and returns a runnable
# object. After this point the graph is immutable.
# draw_mermaid_png() generates a visual diagram of the flow.
# ------------------------------------------------------------
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# ------------------------------------------------------------
# Run the Agent
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")

    # Wrap the user's question in a HumanMessage and invoke the graph.
    # The graph runs the ReAct loop until should_continue() returns END.
    res = app.invoke({
        "messages": [
            HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")
        ]
    })

    # The final answer is always the last message in the messages list
    print(res["messages"][LAST].content)
