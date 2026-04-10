"""
LLM factory – returns the configured chat model at runtime.
"""

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)


def get_llm(temperature: float = 0.1):
    """
    Return an instantiated LangChain chat model based on LLM_PROVIDER.

    Supports:
        - "groq"   → ChatGroq  (llama-3.3-70b-versatile by default)
        - "openai" → ChatOpenAI (gpt-4o-mini by default)
    """
    if LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
            # openai/gpt-oss-20b is a reasoning model – pass extra_body for
            # reasoning_effort; temperature is ignored by the model but kept
            # for interface compatibility.
            extra = {}
            if "gpt-oss" in GROQ_MODEL:
                extra = {"extra_body": {"reasoning_effort": "medium"}}
            return ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=temperature,
                **extra,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc

    # Default → OpenAI
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=temperature,
        )
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc
