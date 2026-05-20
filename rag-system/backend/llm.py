"""
LLM factory — supports Groq and NVIDIA NIM.

All calls are wrapped with tenacity retries: up to 3 attempts with exponential
back-off, retrying on rate-limit (429) and transient server (5xx) errors.
"""
from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import SETTINGS
from .logging_config import setup_logging  # noqa: F401

log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "429", "503", "502", "timeout", "connection"))


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _invoke_with_retry(llm, messages):
    return llm.invoke(messages)


def get_chat_llm(model: str | None = None, temperature: float | None = None,
                 num_predict: int | None = None):
    model = model or SETTINGS.llm_model
    temperature = SETTINGS.llm_temperature if temperature is None else temperature
    provider = SETTINGS.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not SETTINGS.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
            )
        return ChatGroq(
            api_key=SETTINGS.groq_api_key,
            model=model,
            temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI
        if not SETTINGS.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to your .env file or environment variables."
            )
        return ChatOpenAI(
            api_key=SETTINGS.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model,
            temperature=temperature,
            max_tokens=num_predict or 2048,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER: '{provider}'. Set LLM_PROVIDER=groq or LLM_PROVIDER=nvidia."
    )


def get_rewrite_llm():
    return get_chat_llm(
        model=SETTINGS.rewrite_model,
        temperature=0.0,
        num_predict=200,
    )
