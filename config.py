"""
Configuration module - loads environment variables and sets defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Provider ────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")   # "openai" | "groq"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

# ─── Vector Store ─────────────────────────────────────────────────────────────
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")

# ─── Agent Behaviour ──────────────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.88"))

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
