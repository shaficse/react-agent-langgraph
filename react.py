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
#
# LLM BACKEND OPTIONS (set USE_HUGGINGFACE below):
#   False → Local Ollama (no API key, needs Ollama server running)
#   True  → HuggingFace Serverless Inference API (free, needs HF token)
# ============================================================

from dotenv import load_dotenv
from langchain_core.tools import tool      # decorator that turns a Python fn into a LangChain tool
from langchain_tavily import TavilySearch  # web-search tool backed by Tavily API

load_dotenv()  # read TAVILY_API_KEY, HUGGINGFACEHUB_API_TOKEN, LANGSMITH_* from .env


# ------------------------------------------------------------
# Backend Switch
# ------------------------------------------------------------
# Set to True  → use HuggingFace free Serverless Inference API
# Set to False → use local Ollama server
# ------------------------------------------------------------
USE_HUGGINGFACE = True


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
# LLM Setup
# ------------------------------------------------------------
if USE_HUGGINGFACE:
    # ----------------------------------------------------------
    # Option A — HuggingFace Serverless Inference API (FREE)
    # ----------------------------------------------------------
    # THEORY: HuggingFace hosts thousands of open-source models
    # and provides a free Serverless Inference API (rate-limited).
    # You only need a free HF account and an API token.
    #
    # Requirements:
    #   - Free account at huggingface.co
    #   - Token at huggingface.co/settings/tokens (read access)
    #   - HUGGINGFACEHUB_API_TOKEN set in .env
    #
    # Model choice — must support tool/function calling:
    #   "Qwen/Qwen2.5-7B-Instruct"         ← recommended (same family as local qwen3)
    #   "mistralai/Mistral-7B-Instruct-v0.3" ← alternative
    #
    # Note: Free tier has rate limits (~requests per minute).
    # For heavier use, upgrade to HF Pro or run locally via Ollama.
    # ----------------------------------------------------------
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.01,   # HF endpoint doesn't accept exactly 0; use near-zero
    )
    llm = ChatHuggingFace(llm=endpoint).bind_tools(tools)

else:
    # ----------------------------------------------------------
    # Option B — Local Ollama (no API key, fully private)
    # ----------------------------------------------------------
    # THEORY: Ollama runs open-source models locally on your machine
    # or LAN. No data leaves your network. Ideal for privacy or
    # offline development. Requires Ollama to be running and the
    # model to be pulled beforehand.
    #
    # Requirements:
    #   - Ollama server running at base_url
    #   - Model pulled: `ollama pull qwen3:8b`
    # ----------------------------------------------------------
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://192.168.10.114:11434",
        temperature=0,
    ).bind_tools(tools)
