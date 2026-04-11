"""
Interactive UI for the Autonomous Research & Execution Agent.

Run with:
    streamlit run ui/app.py

Features
--------
- Query input with example queries.
- Live progress indicator.
- Expandable reasoning steps (plan, route, iterations).
- Formatted final answer with markdown rendering.
- Quality score badge.
- Conversation history sidebar.
"""

import sys
import os
import time

# Resolve imports when run as `streamlit run ui/app.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from graph.workflow import run_query
from memory.conversation_memory import get_history, clear as clear_memory

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AREA – Autonomous Research & Execution Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 14px;
}
.score-high   { background: #d4edda; color: #155724; }
.score-medium { background: #fff3cd; color: #856404; }
.score-low    { background: #f8d7da; color: #721c24; }
.route-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    background: #e2e3e5;
    color: #383d41;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    max_iters = st.slider("Max Reflection Iterations", 1, 5, 3)
    st.divider()

    st.subheader("📚 Example Queries")
    examples = [
        "Explain transformer attention mechanism and give a PyTorch implementation",
        "Write and optimise a Python quicksort function with time complexity analysis",
        "Generate a research report comparing RAG vs fine-tuning for LLMs",
        "What is LangGraph and how does it enable multi-agent AI systems?",
        "Calculate the 50th Fibonacci number using dynamic programming in Python",
    ]
    for ex in examples:
        if st.button(ex[:55] + "...", key=ex):
            st.session_state["prefill"] = ex

    st.divider()
    st.subheader("🧠 Conversation History")
    history = get_history()
    if history:
        for i, r in enumerate(reversed(history[-5:]), 1):
            with st.expander(f"Q{i}: {r['query'][:40]}..."):
                st.write(f"**Score:** {r['score']}")
                st.write(f"**Route:** {r['route']}")
                st.write(r["answer"][:300] + "...")
        if st.button("Clear History"):
            clear_memory()
            if "session_memory" in st.session_state:
                st.session_state["session_memory"] = []
            st.rerun()
    else:
        st.info("No conversation history yet.")


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.title("🤖 AREA – Autonomous Research & Execution Agent")
st.caption(
    "A multi-agent AI system powered by LangGraph · Plans → Routes → Retrieves "
    "→ Synthesises → Evaluates → Reflects"
)

# Prefill from sidebar example buttons
prefill = st.session_state.pop("prefill", "")

query = st.text_area(
    "Enter your query or task:",
    value=prefill,
    height=100,
    placeholder="e.g. Explain transformers and give a PyTorch code example...",
)

col1, col2 = st.columns([1, 5])
run_btn = col1.button("🚀 Run", type="primary", use_container_width=True)
col2.markdown("")

# ─── Run the agent ───────────────────────────────────────────────────────────
if run_btn and query.strip():
    if "session_memory" not in st.session_state:
        st.session_state["session_memory"] = []

    with st.status("🔄 Running AREA pipeline...", expanded=True) as status:
        st.write("📋 **Planner** – decomposing query into steps...")
        time.sleep(0.3)
        st.write("🔀 **Router** – selecting execution strategy...")
        time.sleep(0.3)

        start = time.time()
        try:
            result = run_query(
                query=query,
                memory=st.session_state["session_memory"],
                max_iters=max_iters,
            )
            elapsed = time.time() - start
            st.session_state["session_memory"] = result.get(
                "memory", st.session_state["session_memory"]
            )

            route = result.get("route", "direct")
            if route == "rag":
                st.write("📚 **Retriever** – searching knowledge base...")
            elif route == "tool":
                st.write("🛠️ **Tool Executor** – running code/tool...")
            time.sleep(0.2)
            st.write("✍️ **Synthesizer** – generating draft answer...")
            time.sleep(0.2)
            st.write("🔍 **Evaluator** – scoring answer quality...")
            iters = result.get("iterations", 0)
            if iters > 0:
                st.write(f"🔄 **Reflector** – ran {iters} reflection iteration(s)...")
            st.write("💾 **Memory** – saving to conversation history...")
            status.update(label=f"✅ Done in {elapsed:.1f}s", state="complete")
        except Exception as exc:
            status.update(label="❌ Error", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # ── Results ──────────────────────────────────────────────────────────────
    st.divider()

    # Metrics row
    score = result.get("score", 0.0)
    iterations = result.get("iterations", 0)
    route = result.get("route", "?")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quality Score", f"{score:.2f} / 1.00")
    m2.metric("Reflection Iterations", iterations)
    m3.metric("Execution Route", route.upper())
    m4.metric("Response Time", f"{elapsed:.1f}s")

    # Score badge
    cls = "score-high" if score >= 0.85 else ("score-medium" if score >= 0.6 else "score-low")
    label = "Excellent" if score >= 0.85 else ("Good" if score >= 0.6 else "Needs Work")
    st.markdown(
        f'<span class="score-badge {cls}">● {label} ({score:.2f})</span>',
        unsafe_allow_html=True,
    )

    # Plan
    plan = result.get("plan", [])
    if plan:
        with st.expander("📋 Execution Plan", expanded=False):
            for i, step in enumerate(plan, 1):
                st.markdown(f"**{i}.** {step}")

    # Tool output
    tool_out = result.get("tool_output", "")
    if tool_out:
        with st.expander("🛠️ Tool Output", expanded=False):
            st.code(tool_out)

    # Critique
    critique = result.get("critique", "")
    if critique and iterations > 0:
        with st.expander("🔍 Evaluator Critique", expanded=False):
            st.markdown(critique)

    # Final answer
    st.subheader("💡 Final Answer")
    st.markdown(result.get("final_answer", "No answer generated."))

elif run_btn and not query.strip():
    st.warning("Please enter a query before clicking Run.")
