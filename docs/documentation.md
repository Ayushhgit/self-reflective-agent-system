# AREA – Autonomous Research & Execution Agent
## Capstone Project Documentation

**Course:** B.Tech / MCA – Artificial Intelligence & Machine Learning  
**Academic Year:** 2025–2026  
**Project Type:** Major Capstone Project  

---

## 1. Title

**AREA: Autonomous Research & Execution Agent**  
*A Multi-Agent AI System Using LangGraph for Intelligent Query Resolution*

---

## 2. Problem Statement

### 2.1 Background

Large Language Models (LLMs) such as GPT-4 and Llama 3 have demonstrated impressive capabilities in text generation and question answering. However, they suffer from several critical limitations:

- **Hallucinations**: LLMs can confidently generate factually incorrect information.
- **Static Knowledge**: Pre-trained models lack access to real-time or domain-specific information.
- **Single-Pass Reasoning**: Standard LLM calls do not verify or improve their own output.
- **No Specialisation**: A single LLM prompt is ill-suited to handle queries that require code execution, mathematical reasoning, or document retrieval simultaneously.

### 2.2 The Gap

Existing chatbot systems process a query with a single LLM call and return the first generated response, regardless of its quality. There is no:

- Step-by-step planning before execution.
- Dynamic selection of the right information-retrieval strategy.
- Automated evaluation of answer quality.
- Iterative self-improvement when quality is insufficient.

### 2.3 Objective

Design and implement an **autonomous multi-agent system** that:

1. Intelligently decomposes complex queries into actionable steps.
2. Selects the appropriate execution path (RAG / Tool / Direct).
3. Retrieves grounded information from a vector knowledge base.
4. Executes code or mathematical tools when required.
5. Produces a comprehensive draft answer.
6. Evaluates the answer for correctness, completeness, and hallucination.
7. Iteratively refines the answer until it meets a quality threshold.
8. Maintains conversation context across multiple interactions.

---

## 3. Solution Overview

### 3.1 Proposed Approach

AREA (Autonomous Research & Execution Agent) addresses the above limitations through a **graph-based multi-agent pipeline** implemented using LangGraph. Each agent in the pipeline is a specialised node with a clearly defined responsibility.

### 3.2 High-Level Flow

```
User Query → Planner → Router → [Retriever | Tool | Direct]
           → Synthesizer → Evaluator → [Memory (accept) | Reflector (improve)]
           → Final Answer
```

### 3.3 Self-Improvement Mechanism

The key innovation is the **evaluate–reflect–synthesise** loop:

- The Evaluator scores the draft answer (0–1) on three dimensions.
- If the score is below 0.85, the Reflector analyses the critique and generates targeted improvement instructions.
- The Synthesizer re-drafts the answer using these instructions.
- This loop repeats up to 3 times (configurable), ensuring quality without infinite loops.

---

## 4. System Architecture

### 4.1 Agent Graph

AREA is implemented as a **directed stateful graph** using LangGraph's `StateGraph`. Each node is a Python function that receives the shared `AgentState` and returns an updated copy.

```
┌──────────────────────────────────────────────────────────┐
│                     AgentState (TypedDict)                │
│  query · plan · route · retrieved_docs · tool_output     │
│  draft_answer · final_answer · score · critique          │
│  iterations · max_iters · memory                         │
└──────────────────────────────────────────────────────────┘
         │ flows through every node │
```

### 4.2 Node Descriptions

| Node | File | Responsibility |
|---|---|---|
| `planner_node` | `agents/planner.py` | Generates 3–6 step execution plan via LLM |
| `router_node` | `agents/router.py` | Routes to rag / tool / direct using heuristics + LLM |
| `retriever_node` | `agents/retriever.py` | FAISS similarity search, retrieves top-4 chunks |
| `tool_node` | `agents/tool_executor.py` | Dispatches to Python executor or calculator |
| `synthesis_node` | `agents/synthesizer.py` | Produces comprehensive draft answer |
| `evaluator_node` | `agents/evaluator.py` | Scores answer (0–1), generates critique |
| `reflection_node` | `agents/reflector.py` | Creates improvement plan from critique |
| `memory_node` | `agents/memory_agent.py` | Promotes draft to final, saves to history |

### 4.3 Conditional Edges

Two conditional edges implement the dynamic branching:

**Edge 1 – Routing (after router_node):**
```python
"rag"    → retriever_node
"tool"   → tool_node
"direct" → synthesis_node
```

**Edge 2 – Evaluation (after evaluator_node):**
```python
score ≥ 0.85 OR iterations ≥ max_iters → memory_node (accept)
score < 0.85 AND iterations < max_iters → reflection_node (improve)
```

### 4.4 RAG Subsystem

- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, no API key required)
- **Vector Store**: FAISS with L2-normalised cosine similarity
- **Knowledge Base**: 11 pre-loaded documents covering Transformers, RAG, LangGraph, LangChain, Neural Networks, and Agentic AI
- **Retrieval**: Top-4 most similar chunks per query

### 4.5 Tool Subsystem

| Tool | Implementation | Use Case |
|---|---|---|
| Python Executor | `tools/python_executor.py` | Runs code in an isolated subprocess (15s timeout) |
| Calculator | `tools/calculator.py` | AST-safe math expression evaluator |
| Web Search | `tools/web_search.py` | DuckDuckGo search (optional) |

---

## 5. Features

### 5.1 Multi-Step Planning
The Planner uses a system prompt that forces the LLM to output a JSON array of 3–6 concrete sub-steps. This prevents shallow, one-shot responses and ensures thorough coverage of multi-part queries.

### 5.2 Intelligent Routing
Routing uses a two-layer decision process:
1. **Keyword heuristics** (O(1), no API call) for obvious cases (code-related → tool, research-related → rag).
2. **LLM-based routing** for ambiguous queries, using a zero-temperature model call.

### 5.3 Retrieval-Augmented Generation (RAG)
FAISS provides sub-millisecond nearest-neighbour search. Sentence-transformer embeddings capture semantic meaning, enabling retrieval of conceptually related chunks even without keyword overlap.

### 5.4 Safe Code Execution
Python code from the LLM is written to a temporary file and executed in a separate subprocess. This prevents:
- Access to the host Python globals.
- Infinite loops (15-second hard timeout).
- The calculator uses Python's `ast` module – no `eval()` – preventing code injection.

### 5.5 Self-Evaluation
The Evaluator scores answers on three quantitative dimensions (each 0–10):
- **Correctness**: Factual accuracy.
- **Completeness**: Coverage of all query aspects.
- **Hallucination**: Presence of unsupported claims (10 = none).

These are combined into an `overall_score` (0–1). A threshold of 0.85 is used to decide acceptance.

### 5.6 Reflection Loop
If quality is insufficient, the Reflector generates an actionable bullet-point improvement plan. This plan is injected into the Synthesizer's context in the next iteration, guiding it to address specific weaknesses rather than regenerating blindly.

### 5.7 Conversation Memory
Each Q&A pair is stored with metadata (score, route, timestamp). This enables:
- Context-aware responses in multi-turn conversations.
- Session analytics.
- Knowledge accumulation across queries.

---

## 6. Tech Stack

| Component | Technology | Version |
|---|---|---|
| Agent Orchestration | LangGraph | ≥ 0.2 |
| LLM Abstraction | LangChain | ≥ 0.3 |
| LLM Provider | Groq (Llama 3.3 70B) | Latest |
| Embeddings | sentence-transformers | ≥ 3.0 |
| Vector Store | FAISS-CPU | ≥ 1.8 |
| API Backend | FastAPI + Uvicorn | ≥ 0.115 |
| Frontend | Streamlit | ≥ 1.38 |
| Language | Python | 3.10+ |
| Configuration | python-dotenv | ≥ 1.0 |

### Why These Choices?

- **LangGraph over LangChain agents**: LangGraph provides explicit state management and supports cycles (needed for the reflection loop), which standard LangChain agents do not support natively.
- **FAISS over ChromaDB**: FAISS is lighter, faster, and requires no separate service. Suitable for academic demos.
- **Groq over OpenAI**: Groq's inference speed is 10–20× faster than OpenAI, making demo interactions feel instantaneous. Free tier is available.
- **Sentence-transformers**: Enables local embedding generation with no API cost or rate limits.

---

## 7. Screenshots & Demo

> *(Replace placeholder text below with actual screenshots during submission)*

**Figure 1 – Streamlit UI Main Screen**  
`[Screenshot: Query input area, Run button, example queries in sidebar]`

**Figure 2 – Pipeline Progress**  
`[Screenshot: Live status panel showing Planner → Router → Retriever → Synthesizer → Evaluator progress]`

**Figure 3 – Query Result Display**  
`[Screenshot: Quality score badge (0.92), route badge (RAG), iteration count, final answer in markdown]`

**Figure 4 – Execution Plan Expander**  
`[Screenshot: Numbered plan steps generated by the Planner for a complex query]`

**Figure 5 – FastAPI Swagger UI**  
`[Screenshot: /docs page showing POST /query endpoint with request/response schema]`

**Figure 6 – Architecture Diagram**  
`[See Section 4.1 – text-based architecture diagram]`

---

## 8. Unique Points & Innovations

### 8.1 Self-Reflection + Quality Gate
Unlike standard RAG pipelines that return the first generated answer, AREA implements a **quality gate** backed by LLM-based scoring. Only answers that score ≥ 0.85 on correctness, completeness, and hallucination are accepted. This dramatically reduces low-quality outputs.

### 8.2 Structured Planning Before Execution
AREA separates **intent understanding** (Planner) from **information gathering** (Retriever/Tool) and **answer generation** (Synthesizer). This prevents the common failure mode where an LLM jumps directly to answering without fully understanding what information is needed.

### 8.3 Dual Routing Strategy
The two-layer routing (keyword heuristics + LLM fallback) balances speed and accuracy. Simple queries are routed in microseconds; complex queries get the full LLM decision-making power.

### 8.4 Sandboxed Code Execution
The Python executor isolates user-generated code in a subprocess with a strict timeout. This is a practical safety measure that production agentic systems must implement but is often overlooked in academic projects.

### 8.5 Modular, Testable Architecture
Every agent is a pure function `(AgentState) → AgentState`. This makes unit testing trivial: mock the LLM, pass a state dict, verify the output state. The graph itself is also independently testable.

---

## 9. Future Improvements

### 9.1 Short-term
- **Web Search Integration**: Add real-time web search as a retrieval source, combining it with FAISS results.
- **Streaming Output**: Stream tokens to the Streamlit UI for faster perceived response times.
- **Unit Tests**: Add pytest test suite with mocked LLM responses.

### 9.2 Medium-term
- **Persistent Vector Memory**: Use ChromaDB for cross-session knowledge accumulation.
- **LangSmith Tracing**: Add observability with LangSmith for debugging and performance monitoring.
- **Multi-modal Support**: Process image + text queries using vision-capable models.

### 9.3 Long-term
- **Cloud Deployment**: Deploy FastAPI on AWS/GCP with Redis for distributed memory.
- **Multi-Agent Collaboration**: Add specialised sub-agents (code writer, fact checker, summariser) that collaborate via message passing.
- **Fine-tuned Evaluator**: Train a dedicated evaluation model instead of using an LLM judge.
- **Human-in-the-Loop**: Add approval steps for high-stakes decisions.

---

## 10. Conclusion

AREA demonstrates that a well-structured multi-agent system, built on modern frameworks like LangGraph, can significantly outperform single-pass LLM inference in terms of answer quality, reliability, and adaptability. The self-evaluation and reflection loop are the key differentiators, enabling the system to identify and correct its own weaknesses autonomously.

This project provides a strong foundation for building production-grade agentic AI systems and is directly applicable to domains such as research assistance, automated code review, and intelligent tutoring systems.

---

*Document prepared for academic submission – Capstone Project Evaluation*
