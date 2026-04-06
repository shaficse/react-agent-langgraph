# ============================================================
# react.py — LLM + Tools Setup
# ============================================================
# THEORY: What is a Tool-Calling LLM?
# A standard LLM only generates text. When we "bind tools" to it,
# we tell the model: "You may choose to call one of these functions
# instead of replying directly." The model then outputs a structured
# tool_call (function name + arguments) instead of plain text.
# The graph (main.py) intercepts that call, runs the real function,
# and feeds the result back to the model for its final answer.
# ============================================================

from dotenv import load_dotenv
from langchain_core.tools import tool      # decorator that turns a Python fn into a LangChain tool
from langchain_ollama import ChatOllama    # Ollama LLM adapter for LangChain
from langchain_tavily import TavilySearch  # web-search tool backed by Tavily API

load_dotenv()  # read TAVILY_API_KEY and LANGSMITH_* from .env


# ------------------------------------------------------------
# TOOL 1 — Web Search (Tavily)
# ------------------------------------------------------------
# TavilySearch is a pre-built tool. It accepts a query string
# and returns real-time web results. max_results=1 keeps the
# context concise — the LLM only sees the top result.
# The model calls this when it needs live/factual information
# it wasn't trained on (e.g. today's weather, stock prices).
# ------------------------------------------------------------
tavily_tool = TavilySearch(max_results=1)


# ------------------------------------------------------------
# TOOL 2 — Custom Function Tool (triple)
# ------------------------------------------------------------
# The @tool decorator converts any Python function into a
# LangChain-compatible tool. The docstring is critical: the LLM
# reads it to decide WHEN and HOW to call this function.
# Always write clear, descriptive docstrings for your tools.
# ------------------------------------------------------------
@tool
def triple(num: float) -> float:
    """
    Multiplies the given number by 3 and returns the result.

    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


# ------------------------------------------------------------
# Tools Registry
# ------------------------------------------------------------
# All tools available to the agent are listed here.
# This list is passed to both the LLM (so it knows what exists)
# and the ToolNode (so it can actually execute them).
# ------------------------------------------------------------
tools = [tavily_tool, triple]


# ------------------------------------------------------------
# LLM Setup — Ollama (local, no API key needed)
# ------------------------------------------------------------
# ChatOllama connects to a locally running Ollama server.
# - model:       which model to use (qwen3:8b is fast & capable)
# - base_url:    address of the Ollama server on your network
# - temperature: 0 = deterministic output (best for agents that
#                must reason reliably, not creatively)
#
# .bind_tools(tools) attaches the tools list to the LLM so that
# on every call the model is aware of available functions and
# can emit tool_call objects in its response.
# ------------------------------------------------------------
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://192.168.10.114:11434",
    temperature=0,
).bind_tools(tools)
