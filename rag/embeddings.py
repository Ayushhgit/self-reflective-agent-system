"""
Returns a LangChain-compatible embeddings object.

Priority:
  1. OpenAI embeddings (if USE_OPENAI_EMBEDDINGS=true and key available)
  2. HuggingFace sentence-transformers (local, no API key needed) – default
"""

import logging
from functools import lru_cache

from config import EMBEDDING_MODEL, USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached embeddings instance.
    Uses HuggingFace by default (no API key required).
    """
    if USE_OPENAI_EMBEDDINGS and OPENAI_API_KEY:
        logger.info("[Embeddings] Using OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    logger.info("[Embeddings] Using HuggingFace '%s'.", EMBEDDING_MODEL)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc
