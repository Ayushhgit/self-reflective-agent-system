"""
Uses the FAISS-backed vector store to retrieve the most relevant document
chunks for the current query, and stores them in state.retrieved_docs.
"""

import logging

from state import AgentState
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

TOP_K = 4   # Number of chunks to retrieve


def retriever_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Retriever

    Responsibilities
    ----------------
    - Embed the query.
    - Perform similarity search against the FAISS vector store.
    - Return the top-K most relevant chunks in state.retrieved_docs.
    """
    query = state["query"]
    logger.info("[Retriever] Retrieving documents for: %s", query[:80])

    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(query, k=TOP_K)
        retrieved = [doc.page_content for doc in docs]
        logger.info("[Retriever] Retrieved %d chunks.", len(retrieved))
    except Exception as exc:
        logger.warning("[Retriever] Vector store error: %s – falling back to empty.", exc)
        retrieved = []

    return {**state, "retrieved_docs": retrieved}
