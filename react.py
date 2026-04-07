# ============================================================
# react.py — LLM + Tools Setup
# ============================================================
#
# WHAT IS THIS FILE?
# This file is the "brain and hands" of our AI agent.
#   - Brain = the LLM (the AI model that thinks and decides)
#   - Hands = the Tools (functions the AI can run to get data)
#
# WHAT IS AN LLM?
# LLM stands for Large Language Model. It is an AI model trained
# on huge amounts of text. It can read your question and write
# a reply — but by itself it cannot browse the web or do math
# beyond its training. That is why we give it TOOLS.
#
# WHAT IS A TOOL?
# A Tool is just a normal Python function that the AI is allowed
# to call. For example, we give it:
#   - A web search tool  → to look up live data (weather, news)
#   - A math tool        → to triple a number
#
# HOW DOES TOOL CALLING WORK?
# Step 1: User asks "What is Tokyo's temperature? Then triple it."
# Step 2: LLM reads the question and thinks: "I need to search."
# Step 3: LLM outputs a special message called a "tool_call":
#           { "name": "TavilySearch", "args": {"query": "Tokyo temp"} }
# Step 4: Our graph (main.py) sees the tool_call, runs the function,
#         and sends the result back to the LLM.
# Step 5: LLM now knows the temperature. It thinks: "Now triple it."
# Step 6: LLM calls the triple tool → gets the result.
# Step 7: LLM writes the final answer in plain text.
#
# LLM BACKEND OPTIONS:
#   USE_HUGGINGFACE = True  → Free cloud model (HuggingFace API)
#   USE_HUGGINGFACE = False → Local model on your own machine (Ollama)
# ============================================================


# ------------------------------------------------------------
# IMPORTS — bringing in code from other libraries
# ------------------------------------------------------------
# "import" means: go find this library and load its code so
# we can use it in this file.

from dotenv import load_dotenv
# dotenv reads your .env file and loads the secret keys (API keys)
# as environment variables so we can access them with os.getenv().
# Think of .env as a private notepad where you store passwords.

from langchain_core.tools import tool
# This is a "decorator" — a special marker (@tool) that we put
# above a Python function to tell LangChain:
# "Hey, this function is a Tool — the AI is allowed to call it."

from langchain_tavily import TavilySearch
# Tavily is a search engine built for AI agents.
# TavilySearch is a ready-made tool that searches the web and
# returns the top results as clean text (not raw HTML).

load_dotenv()
# This line actually reads the .env file and loads the keys.
# Must be called before we use any API keys.
# Example .env file:
#   TAVILY_API_KEY=tvly-abc123
#   HUGGINGFACEHUB_API_TOKEN=hf_xyz789


# ------------------------------------------------------------
# BACKEND SWITCH
# ------------------------------------------------------------
# TYPE: bool (True or False — the simplest data type in Python)
#
# A bool can only be one of two values:
#   True  → yes / on
#   False → no  / off
#
# We use it here as a simple on/off switch to choose our LLM.
#
# Examples:
#   USE_HUGGINGFACE = True   → use the free HuggingFace cloud model
#   USE_HUGGINGFACE = False  → use your local Ollama model
# ------------------------------------------------------------
USE_HUGGINGFACE = True


# ------------------------------------------------------------
# TOOL 1 — Web Search (TavilySearch)
# ------------------------------------------------------------
# WHAT IS IT?
# TavilySearch is a pre-built tool that searches the internet.
# We do not need to write this one ourselves — it is provided
# by the langchain-tavily library.
#
# HOW DOES IT WORK?
# The LLM passes a search query (a string) to this tool.
# The tool searches the web and returns a result (a dict).
#
# PARAMETER:
#   max_results (int) — how many web results to return.
#   We use 1 to keep the response short and focused.
#   More results = more context for the LLM but also more tokens.
#
# INPUT  TYPE: str
#   Example: "current weather in Tokyo"
#
# OUTPUT TYPE: dict (a key-value data structure)
#   Example:
#   {
#     "query": "current weather in Tokyo",
#     "results": [
#       {
#         "url": "https://weather.com/tokyo",
#         "content": "Tokyo: 28°C, sunny.",
#         "score": 0.94        ← relevance score between 0 and 1
#       }
#     ],
#     "response_time": 0.72   ← how many seconds the search took
#   }
# ------------------------------------------------------------
tavily_tool = TavilySearch(max_results=1)


# ------------------------------------------------------------
# TOOL 2 — Custom Conversion Tool (celsius_to_fahrenheit)
# ------------------------------------------------------------
# WHAT IS IT?
# This is a tool we wrote ourselves. It converts a temperature
# from Celsius (°C) to Fahrenheit (°F) — a real-world unit
# conversion that pairs naturally with a live weather search.
#
# WHY IS THIS MORE MEANINGFUL THAN A SIMPLE MULTIPLY?
# When the agent searches for "current temperature in Tokyo"
# it gets a value in Celsius (most of the world uses Celsius).
# A user in the US may need Fahrenheit. This tool bridges that
# gap — making the agent genuinely useful in a real scenario.
#
# WHY @tool?
# The @tool decorator wraps this plain Python function so that
# LangChain understands it is a tool the LLM can call.
# Without @tool, LangChain would not know this function exists.
#
# THE DOCSTRING IS VERY IMPORTANT:
# The text inside the triple-quotes (""" ... """) is called a
# docstring. The LLM reads this docstring to understand:
#   - What this tool does
#   - When to call it
#   - What argument to pass
# If your docstring is unclear, the LLM may call the tool
# incorrectly or not call it at all.
#
# TYPE HINTS  (celsius: float) -> float
# These tell Python (and the LLM) what data types are expected:
#   celsius: float  → the input must be a decimal number (°C)
#   -> float        → the output will also be a decimal number (°F)
#
# THE FORMULA:  °F = (°C × 9/5) + 32
#
# Examples of calling celsius_to_fahrenheit():
#   celsius_to_fahrenheit(0.0)   → 32.0    (freezing point of water)
#   celsius_to_fahrenheit(28.0)  → 82.4    (warm Tokyo summer day)
#   celsius_to_fahrenheit(100.0) → 212.0   (boiling point of water)
#   celsius_to_fahrenheit(-40.0) → -40.0   (where °C and °F meet)
#   celsius_to_fahrenheit(37.0)  → 98.6    (human body temperature)
# ------------------------------------------------------------
@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Converts a temperature from Celsius to Fahrenheit.

    param celsius: temperature in Celsius (°C)
    returns: temperature in Fahrenheit (°F)
    """
    return (float(celsius) * 9 / 5) + 32
    # Formula: °F = (°C × 9/5) + 32
    # float(celsius) ensures consistent float output even if an int is passed.
    # e.g. celsius_to_fahrenheit(28) → 82.4


# ------------------------------------------------------------
# TOOLS REGISTRY
# ------------------------------------------------------------
# TYPE: list  (an ordered collection of items in Python)
#
# What is a list?
#   A list holds multiple items in order, inside square brackets [].
#   Examples:
#     numbers = [1, 2, 3]
#     names   = ["Alice", "Bob"]
#     mixed   = [1, "hello", True]
#
# Here we collect all our tools into one list.
# This list is used in two places:
#   1. Given to the LLM via .bind_tools(tools)
#      → so the model knows what tools are available
#   2. Given to ToolNode in nodes.py
#      → so the graph can actually run the tools
#
# Current tools:
#   tools[0] = tavily_tool  → web search
#   tools[1] = triple       → multiply by 3
#
# To add more tools later, just add them to this list:
#   tools = [tavily_tool, triple, my_new_tool]
# ------------------------------------------------------------
tools = [tavily_tool, celsius_to_fahrenheit]


# ------------------------------------------------------------
# LLM SETUP
# ------------------------------------------------------------
# Here we create the actual AI model object.
# We use an if/else to pick which backend to use based on
# the USE_HUGGINGFACE flag we set above.
# ------------------------------------------------------------

if USE_HUGGINGFACE:
    # ----------------------------------------------------------
    # OPTION A — HuggingFace Serverless Inference API (FREE)
    # ----------------------------------------------------------
    # WHAT IS HUGGINGFACE?
    # HuggingFace is a platform that hosts thousands of open-source
    # AI models for free. Instead of running the model on your own
    # computer, you send your message to HuggingFace's servers and
    # they run the model and send back the reply.
    #
    # Think of it like: instead of cooking yourself, you order food
    # from a restaurant (HuggingFace) — the kitchen (GPU servers)
    # is theirs, not yours.
    #
    # REQUIREMENTS:
    #   1. Free account at huggingface.co
    #   2. API token with "Make calls to Inference Providers" permission
    #   3. HUGGINGFACEHUB_API_TOKEN=hf_... in your .env file
    #
    # WHAT IS HuggingFaceEndpoint?
    # This object represents the connection to a specific model
    # hosted on HuggingFace. It holds the settings for one model.
    #
    # PARAMETERS EXPLAINED:
    #
    #   repo_id (str) — the model's unique name on HuggingFace.
    #     Format: "organization/model-name"
    #     Example: "Qwen/Qwen2.5-7B-Instruct"
    #       - "Qwen"            → the organization that made it
    #       - "Qwen2.5-7B"      → version 2.5, 7 Billion parameters
    #       - "Instruct"        → fine-tuned to follow instructions
    #     Other examples:
    #       "mistralai/Mistral-7B-Instruct-v0.3"
    #       "meta-llama/Meta-Llama-3-8B-Instruct"
    #
    #   task (str) — what kind of AI task this model performs.
    #     "text-generation" means: given text in, produce text out.
    #     Other tasks exist: "image-classification", "translation"
    #     but for chat agents we always use "text-generation".
    #
    #   max_new_tokens (int) — the maximum number of new "tokens"
    #     the model is allowed to generate in one response.
    #     What is a token? A token is roughly a word or word-piece.
    #       "Hello world" ≈ 2 tokens
    #       "What is the temperature?" ≈ 5 tokens
    #     Examples:
    #       max_new_tokens=256   → short replies only
    #       max_new_tokens=1024  → medium replies (our choice)
    #       max_new_tokens=4096  → long detailed replies
    #
    #   temperature (float) — controls how creative vs predictable
    #     the model's output is. Range: 0.0 to 2.0
    #     Examples:
    #       temperature=0.0  → always picks the most likely word
    #                          (best for facts, math, tool calling)
    #       temperature=0.7  → some randomness, more natural replies
    #       temperature=1.5  → very random, creative but unreliable
    #     Note: HuggingFace API rejects exactly 0.0, so we use 0.01
    #     (effectively the same — extremely low randomness).
    # ----------------------------------------------------------
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.01,       # as close to 0 as HF allows
    )

    # ChatHuggingFace wraps the endpoint to support chat-style
    # conversations (with roles: system / user / assistant).
    # .bind_tools(tools) attaches our tools list to the model so
    # it knows what functions are available on every call.
    llm = ChatHuggingFace(llm=endpoint).bind_tools(tools)

else:
    # ----------------------------------------------------------
    # OPTION B — Local Ollama (no API key, fully private)
    # ----------------------------------------------------------
    # WHAT IS OLLAMA?
    # Ollama is an application that runs AI models directly on
    # your own computer or local network. No data is sent to any
    # cloud server — everything happens on your machine.
    #
    # Think of it like: instead of ordering food (HuggingFace),
    # you cook at home (your own GPU/CPU). Slower if your hardware
    # is limited, but private, free, and always available offline.
    #
    # REQUIREMENTS:
    #   1. Ollama installed and running
    #   2. Model pulled: run `ollama pull qwen3:8b` in terminal
    #   3. Server accessible at base_url
    #
    # PARAMETERS EXPLAINED:
    #
    #   model (str) — which model to load in Ollama.
    #     Must match exactly what `ollama list` shows.
    #     Examples:
    #       "qwen3:8b"       → Qwen3, 8 billion parameters (our choice)
    #       "llama3:latest"  → Meta's Llama 3
    #       "deepseek-r1:8b" → DeepSeek R1 reasoning model
    #
    #   base_url (str) — the web address of the Ollama server.
    #     Examples:
    #       "http://localhost:11434"          → Ollama on this machine
    #       "http://192.168.10.114:11434"     → Ollama on another machine
    #                                           on the same WiFi network
    #     Port 11434 is Ollama's default port.
    #
    #   temperature (float) — same as above. 0 = deterministic.
    #     For agents that must reason reliably, always use 0.
    # ----------------------------------------------------------
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://192.168.10.114:11434",
        temperature=0,
    ).bind_tools(tools)
    # .bind_tools(tools) — same as HuggingFace above.
    # Tells the model: "You have access to these tools."
    # The model will output tool_call objects when it wants to use them.
