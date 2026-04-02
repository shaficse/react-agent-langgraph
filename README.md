# ReAct Agent with Function Calling (LangGraph + Ollama)

A ReAct (Reasoning + Acting) agent built with LangGraph that uses tool calling to answer questions. Runs locally via Ollama.

![Agent Flow](flow.png)

## Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Ollama](https://ollama.com) running on `192.168.10.114:11434` with `qwen3:8b` pulled
- A [Tavily](https://tavily.com) API key (for web search)

## Setup

1. **Install dependencies**

   ```bash
   poetry install
   pip install langchain-ollama
   ```

2. **Configure environment**

   Create a `.env` file in the project root:

   ```env
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Ensure Ollama is running and the model is available**

   ```bash
   curl http://192.168.10.114:11434/api/tags
   # qwen3:8b should appear in the response
   ```

## Run

```bash
poetry run python main.py
```

The agent will answer the query defined in `main.py` and print the result. It also generates `flow.png` showing the agent graph.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — builds the LangGraph flow and runs the agent |
| `nodes.py` | Agent reasoning node and tool node definitions |
| `react.py` | LLM and tools setup (Ollama `qwen3:8b` + Tavily + `triple`) |

## Tools

| Tool | Description |
|------|-------------|
| `TavilySearch` | Web search (max 1 result) |
| `triple` | Multiplies a number by 3 |

## Changing the Query

Edit the `HumanMessage` content in [main.py](main.py#L37):

```python
res = app.invoke({"messages": [HumanMessage(content="Your question here")]})
```
