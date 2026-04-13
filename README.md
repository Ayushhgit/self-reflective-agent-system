# 🤖 AREA – Autonomous Research & Execution Agent

> A production-ready, multi-agent AI system built with **LangGraph** that autonomously plans, retrieves, executes tools, synthesises answers, evaluates quality, and reflects to improve itself.

---

## 📌 Project Overview

AREA is a **capstone-level agentic AI system** that demonstrates the full lifecycle of autonomous decision-making:

1. **Planner** – Decomposes complex queries into ordered steps.
2. **Router** – Intelligently routes execution to RAG, Tool, or Direct path.
3. **Retriever** – Performs embedding-based similarity search (FAISS + RAG).
4. **Tool Executor** – Runs Python code and mathematical calculations safely.
5. **Synthesizer** – Generates comprehensive, structured answers.
6. **Evaluator** – Scores answers on correctness, completeness & hallucination.
7. **Reflector** – Produces improvement plans when quality falls short.
8. **Memory** – Persists conversation history across turns.

---

## 🏗️ Architecture

```
    User Query
       │
       ▼
┌─────────────┐
│   Planner   │  ← Generates step-by-step execution plan
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Router    │  ← Decides: rag | tool | direct
└──────┬──────┘
       │
   ┌───┴───────────┐
   ▼               ▼
┌──────┐       ┌──────┐
│ RAG  │       │ Tool │    (or skip both → direct)
│Retr. │       │ Exec.│
└──┬───┘       └──┬───┘
   └──────┬────────┘
          ▼
   ┌─────────────┐
   │ Synthesizer │  ← Draft answer
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  Evaluator  │  ← Score (0–1) + Critique
   └──────┬──────┘
          │
     ┌────┴──────────────────┐
     ▼ score < 0.85          ▼ score ≥ 0.85
┌──────────┐           ┌──────────┐
│Reflector │──────────▶│  Memory  │──▶ Final Answer
└──────────┘(loop≤3)   └──────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| Multi-step Planning | Query decomposed into 3–6 actionable steps |
| Smart Routing | LLM + heuristic routing to rag / tool / direct |
| RAG System | FAISS vector store + sentence-transformer embeddings |
| Python Execution | Safe subprocess-isolated code runner with 15s timeout |
| Calculator Tool | AST-safe mathematical expression evaluator |
| Self-Evaluation | Scores answers on correctness, completeness, hallucination |
| Reflection Loop | Iterative improvement (max 3 cycles by default) |
| Conversation Memory | Persists Q&A history across turns |
| FastAPI Backend | REST API with `/query`, `/history`, `/documents` endpoints |
| Streamlit Frontend | Interactive UI with reasoning steps display |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangGraph** `>=0.2` – Multi-agent graph orchestration
- **LangChain** `>=0.3` – LLM abstractions, chains, prompts
- **FAISS** – Vector similarity search
- **sentence-transformers** – Local embeddings (no API key needed)
- **Groq API** – Ultra-fast LLM inference (Llama 3.3 70B)
- **FastAPI + Uvicorn** – Production REST API
- **Streamlit** – Interactive web UI
- **python-dotenv** – Configuration management

---

## 📦 Project Structure

```
capstone_project/
├── main.py                  # CLI entry point / interactive REPL
├── config.py                # Environment-based configuration
├── state.py                 # AgentState TypedDict
├── llm_factory.py           # LLM provider abstraction
├── requirements.txt
├── .env.example
│
├── agents/                  # LangGraph node implementations
│   ├── planner.py           # Multi-step planning
│   ├── router.py            # Smart routing
│   ├── retriever.py         # RAG retrieval
│   ├── tool_executor.py     # Tool dispatch
│   ├── synthesizer.py       # Answer generation
│   ├── evaluator.py         # Quality scoring
│   ├── reflector.py         # Self-reflection
│   └── memory_agent.py      # Conversation memory
│
├── tools/                   # Executable tools
│   ├── python_executor.py   # Safe Python runner
│   ├── calculator.py        # AST-safe calculator
│   └── web_search.py        # DuckDuckGo search
│
├── rag/                     # Retrieval-Augmented Generation
│   ├── embeddings.py        # Embedding model factory
│   ├── vector_store.py      # FAISS vector store
│   └── knowledge_base.py    # Default knowledge documents
│
├── memory/
│   └── conversation_memory.py  # In-session history
│
├── graph/
│   └── workflow.py          # LangGraph graph definition
│
├── api/
│   └── app.py               # FastAPI REST backend
│
├── ui/
│   └── app.py               # Streamlit frontend
│
└── docs/
    └── documentation.md     # Academic documentation
```

---

## 🚀 Setup & Installation

### 1. Prerequisites

- Python 3.10 or higher
- A Groq API key (free at [console.groq.com](https://console.groq.com)) **or** an OpenAI API key

### 2. Install with uv

```bash
# Navigate to the project directory
cd capstone_project

# Install uv if you don't have it
pip install uv

# Create venv + install all dependencies in one command
uv sync
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (or OPENAI_API_KEY)
```

Minimum required in `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

### 4. Run

**Interactive CLI (REPL):**
```bash
python main.py
```

**Single query:**
```bash
python main.py --query "Explain transformer attention mechanism"
```

**Demo mode (3 built-in queries):**
```bash
python main.py --demo
```

**FastAPI server:**
```bash
python main.py --api
# or
uvicorn api.app:app --reload --port 8000
```

**Streamlit UI:**
```bash
python main.py --ui
# or
streamlit run ui/app.py
```

---

## 🌐 API Reference

### POST /query

```json
{
  "query": "Explain transformers and give a PyTorch example",
  "max_iters": 3
}
```

Response:
```json
{
  "query": "...",
  "final_answer": "...",
  "score": 0.92,
  "iterations": 1,
  "route": "rag",
  "plan": ["step 1", "step 2", "..."],
  "critique": "...",
  "tool_output": ""
}
```

### GET /health
### GET /history
### POST /memory/clear
### POST /documents  `{ "texts": ["doc1", "doc2"] }`

---

## 🧪 Example Queries

| Query | Expected Route |
|---|---|
| "Explain transformers and give PyTorch code" | `rag` → `tool` |
| "Write and optimise a Python quicksort function" | `tool` |
| "Generate a research report on RAG vs fine-tuning" | `rag` |
| "What is LangGraph?" | `rag` |
| "Calculate 2^32 + sqrt(144)" | `tool` |

---

## 🔑 Key Design Decisions

1. **TypedDict state** flows through all nodes — easy to inspect and debug.
2. **Heuristic + LLM routing** — fast keyword matching first, LLM as fallback.
3. **Subprocess isolation** for Python execution — prevents code injection.
4. **AST-safe calculator** — no `eval()` security risks.
5. **lru_cache** on vector store and graph — single initialization, reused.
6. **Configurable thresholds** via `.env` — tunable without code changes.

---

## 🔮 Future Improvements

- Integrate web search as a retrieval source alongside FAISS.
- Add persistent vector memory (ChromaDB) for cross-session knowledge.
- Implement agent-level tracing with LangSmith.
- Deploy to cloud (AWS Lambda + API Gateway / Streamlit Cloud).
- Add multi-modal support (image + text queries).

---

## 👨‍💻 Author

Built as a capstone project demonstrating advanced **Agentic AI** system design with LangGraph.

---

