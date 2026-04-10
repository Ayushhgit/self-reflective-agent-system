"""
Checks prerequisites, installs dependencies, and builds the FAISS vector
store from the default knowledge base.

Run with:
    python setup.py
"""

import subprocess
import sys
import os


def run(cmd, desc):
    print(f"  → {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ✗ Failed: {desc}")
        return False
    print(f"  ✓ {desc}")
    return True


def main():
    print("\n🤖 AREA – Setup")
    print("=" * 50)

    # 1. Python version check
    major, minor = sys.version_info[:2]
    print(f"\n[1/4] Python {major}.{minor}", end=" ")
    if major < 3 or (major == 3 and minor < 10):
        print("✗  (requires Python 3.10+)")
        sys.exit(1)
    print("✓")

    # 2. Install requirements
    print("\n[2/4] Installing dependencies via uv...")
    ok = run("uv sync", "uv sync")
    if not ok:
        print("  Run manually: pip install -r requirements.txt")

    # 3. Check .env
    print("\n[3/4] Checking .env configuration...")
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("  ✓ Created .env from .env.example")
            print("  ⚠  IMPORTANT: Edit .env and add your GROQ_API_KEY or OPENAI_API_KEY")
        else:
            print("  ✗ .env.example not found")
    else:
        print("  ✓ .env exists")
        # Check if API key is set
        with open(".env") as f:
            content = f.read()
        if "your_groq_api_key_here" in content or "your_openai_api_key_here" in content:
            print("  ⚠  API key placeholder detected – please edit .env")

    # 4. Pre-build vector store
    print("\n[4/4] Pre-building FAISS vector store...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rag.vector_store import get_vector_store
        vs = get_vector_store()
        print(f"  ✓ Vector store ready")
    except Exception as exc:
        print(f"  ⚠  Vector store build deferred (will build on first run): {exc}")

    print("\n" + "=" * 50)
    print("✅ Setup complete!\n")
    print("Next steps:")
    print("  1. Edit .env and add your API key")
    print("  2. python main.py                    # Interactive REPL")
    print("  3. python main.py --demo             # Run demo queries")
    print("  4. python main.py --api              # Start API server")
    print("  5. streamlit run ui/app.py           # Start Streamlit UI")
    print()


if __name__ == "__main__":
    main()
