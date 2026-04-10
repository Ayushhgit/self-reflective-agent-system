"""
Manages a FAISS-backed vector store for document retrieval.

Features
--------
- Auto-creates the index if it does not exist on disk.
- Seeds with a default knowledge base on first run.
- Exposes add_documents() and similarity_search() via the LangChain interface.
"""

import logging
import os
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from config import VECTOR_DB_PATH
from rag.embeddings import get_embeddings
from rag.knowledge_base import DEFAULT_DOCUMENTS

logger = logging.getLogger(__name__)

INDEX_PATH = os.path.join(VECTOR_DB_PATH, "faiss_index")


@lru_cache(maxsize=1)
def get_vector_store():
    """
    Return a cached FAISS vector store.
    Loads from disk if it exists; otherwise builds from the default knowledge base.
    """
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        logger.info("[VectorStore] Loading existing FAISS index from %s", INDEX_PATH)
        try:
            return FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.warning("[VectorStore] Load failed (%s) – rebuilding.", exc)

    logger.info("[VectorStore] Building new FAISS index from default knowledge base.")
    docs = [
        Document(page_content=text, metadata={"source": f"kb_{i}"})
        for i, text in enumerate(DEFAULT_DOCUMENTS)
    ]
    vs = FAISS.from_documents(docs, embeddings)
    _save(vs)
    return vs


def add_documents(texts: List[str], metadatas: List[dict] = None):
    """
    Add new documents to the vector store and persist the updated index.

    Parameters
    ----------
    texts     : List of plain-text document strings.
    metadatas : Optional list of metadata dicts (one per text).
    """
    from langchain_community.vectorstores import FAISS

    vs = get_vector_store()
    metadatas = metadatas or [{"source": f"added_{i}"} for i in range(len(texts))]
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    vs.add_documents(docs)
    _save(vs)
    # Invalidate cache so next call reloads
    get_vector_store.cache_clear()
    logger.info("[VectorStore] Added %d documents.", len(texts))


def _save(vs):
    os.makedirs(INDEX_PATH, exist_ok=True)
    vs.save_local(INDEX_PATH)
    logger.info("[VectorStore] Saved FAISS index to %s", INDEX_PATH)
