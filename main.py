"""
CLI entry point.  Run queries directly from the terminal.

Usage
-----
    python main.py
    python main.py --query "Explain transformer attention"
    python main.py --query "Write a Python merge sort" --max-iters 2
    python main.py --demo          # Run all three built-in demo queries
    python main.py --api           # Start the FastAPI server
    python main.py --ui            # Start the Streamlit UI
"""

import argparse
import logging
import sys
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AREA")

# ─── Demo queries ─────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    "Explain the transformer attention mechanism and provide a PyTorch implementation.",
    "Write a Python function to compute Fibonacci numbers using dynamic programming, then optimise it.",
    "Generate a research report comparing RAG vs fine-tuning for large language models.",
]


def print_result(result: dict):
    """Pretty-print an AREA result to stdout."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"QUERY   : {result.get('query', '')}")
    print(f"ROUTE   : {result.get('route', '').upper()}")
    print(f"SCORE   : {result.get('score', 0):.2f} / 1.00")
    print(f"ITERS   : {result.get('iterations', 0)}")
    plan = result.get("plan", [])
    if plan:
        print(f"\nPLAN ({len(plan)} steps):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
    tool_out = result.get("tool_output", "")
    if tool_out:
        print(f"\nTOOL OUTPUT:\n{textwrap.indent(tool_out[:500], '  ')}")
    critique = result.get("critique", "")
    if critique and result.get("iterations", 0) > 0:
        print(f"\nCRITIQUE:\n{textwrap.indent(critique[:400], '  ')}")
    print(f"\nFINAL ANSWER:\n{sep}")
    print(result.get("final_answer", "No answer generated."))
    print(sep)


def run_cli(query: str, max_iters: int):
    """Run the AREA pipeline for a single query via CLI."""
    from graph.workflow import run_query

    print(f"\n🤖 AREA – Processing: {query[:80]}{'...' if len(query) > 80 else ''}\n")
    result = run_query(query=query, max_iters=max_iters)
    print_result(result)


def run_demo(max_iters: int):
    """Run the three built-in demonstration queries."""
    from graph.workflow import run_query

    print("\n" + "=" * 70)
    print("   AREA – DEMO MODE  (3 representative queries)")
    print("=" * 70)
    memory = []
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"\n[Demo {i}/3]")
        result = run_query(query=q, memory=memory, max_iters=max_iters)
        memory = result.get("memory", memory)
        print_result(result)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AREA – Autonomous Research & Execution Agent"
    )
    parser.add_argument("--query", "-q", type=str, help="Query to process")
    parser.add_argument(
        "--max-iters", "-m", type=int, default=3, help="Max reflection iterations (default: 3)"
    )
    parser.add_argument("--demo", action="store_true", help="Run built-in demo queries")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--ui", action="store_true", help="Start Streamlit UI")

    args = parser.parse_args()

    if args.api:
        import uvicorn
        from config import API_HOST, API_PORT
        print(f"\n🚀 Starting AREA API on http://{API_HOST}:{API_PORT}")
        uvicorn.run("api.app:app", host=API_HOST, port=API_PORT, reload=True)

    elif args.ui:
        import subprocess
        ui_path = "ui/app.py"
        print(f"\n🎨 Starting AREA Streamlit UI…")
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])

    elif args.demo:
        run_demo(max_iters=args.max_iters)

    elif args.query:
        run_cli(args.query, max_iters=args.max_iters)

    else:
        # Interactive REPL
        print("\n🤖 AREA – Autonomous Research & Execution Agent")
        print("Type your query and press Enter. Type 'exit' to quit.\n")
        memory = []
        from graph.workflow import run_query

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            max_iters = args.max_iters
            result = run_query(query=query, memory=memory, max_iters=max_iters)
            memory = result.get("memory", memory)
            print_result(result)


if __name__ == "__main__":
    main()
