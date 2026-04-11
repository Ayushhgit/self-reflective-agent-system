import sys
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs during tests

TEST_CASES = [
    {
        "id": 1,
        "name": "Transformer Explanation + Code",
        "query": (
            "Explain the transformer attention mechanism in detail "
            "and provide a working PyTorch implementation."
        ),
        "expected_route": "rag",
        "expected_score_min": 0.7,
    },
    {
        "id": 2,
        "name": "Python Function Optimisation",
        "query": (
            "Write a Python function to find the nth Fibonacci number using "
            "dynamic programming. Include time and space complexity analysis."
        ),
        "expected_route": "tool",
        "expected_score_min": 0.7,
    },
    {
        "id": 3,
        "name": "Research Report: RAG vs Fine-tuning",
        "query": (
            "Generate a research report comparing Retrieval-Augmented Generation (RAG) "
            "with fine-tuning for large language models. Include use cases, "
            "advantages, and limitations of each approach."
        ),
        "expected_route": "rag",
        "expected_score_min": 0.7,
    },
]


def run_test(tc: dict, memory: list):
    """Run a single test case and return (passed, result)."""
    from graph.workflow import run_query

    print(f"\n{'='*65}")
    print(f"TEST {tc['id']}: {tc['name']}")
    print(f"{'='*65}")
    print(f"Query: {tc['query'][:80]}...")

    result = run_query(query=tc["query"], memory=memory, max_iters=2)

    score = result.get("score", 0.0)
    route = result.get("route", "unknown")
    iters = result.get("iterations", 0)
    answer = result.get("final_answer", "")

    print(f"\nRoute     : {route.upper()}")
    print(f"Score     : {score:.2f} / 1.00")
    print(f"Iterations: {iters}")
    print(f"Plan steps: {len(result.get('plan', []))}")
    print(f"\nAnswer (first 400 chars):\n{answer[:400]}...")

    passed = score >= tc["expected_score_min"]
    status = "✅ PASS" if passed else "⚠️ BELOW THRESHOLD"
    print(f"\nResult: {status} (min expected: {tc['expected_score_min']:.2f})")

    return passed, result


def main():
    print("\n🤖 AREA – Test Suite")
    print("Running 3 canonical demo queries...\n")

    memory = []
    results = []
    for tc in TEST_CASES:
        try:
            passed, result = run_test(tc, memory)
            memory = result.get("memory", memory)
            results.append(passed)
        except Exception as exc:
            print(f"\n❌ ERROR in Test {tc['id']}: {exc}")
            results.append(False)

    passed_count = sum(results)
    print(f"\n{'='*65}")
    print(f"SUMMARY: {passed_count}/{len(TEST_CASES)} tests passed")
    print(f"{'='*65}\n")
    return 0 if passed_count == len(TEST_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
