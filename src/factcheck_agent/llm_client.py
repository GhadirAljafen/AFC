"""LLM client abstractions and concrete implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .config import get_config

# If you use OpenAI:
# pip install openai>=1.0.0
try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError:
    AsyncOpenAI = None  # handled gracefully below


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Perform a single-turn completion for a given prompt."""
        ...

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Perform a chat-style completion with a list of messages."""
        ...


class OpenAILLMClient(LLMClient):
    """
    Concrete LLM client using OpenAI's async SDK.

    This assumes:
    - OPENAI_API_KEY is set in the environment (or .env)
    - You have installed: `pip install openai`
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",  # change to any model you prefer
        temperature: float = 0.2,
    ) -> None:
        config = get_config()
        api_key = getattr(config, "OPENAI_API_KEY", None)

        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is not installed. "
                "Install it with: pip install openai"
            )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your environment or .env file."
            )

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Single-turn completion using chat API under the hood.

        We wrap the prompt into a simple system+user chat for consistency.
        """
        messages = [
            {"role": "system", "content": "You are a helpful language model."},
            {"role": "user", "content": prompt},
        ]

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return resp.choices[0].message.content or ""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Chat-style completion.

        Expects messages of the form:
        [{"role": "system" | "user" | "assistant", "content": "..."}]
        """
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return resp.choices[0].message.content or ""


class DummyLLMClient(LLMClient):
    """
    Very simple LLM client for tests and offline runs.

    It does NOT call any external API.
    """

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return f"[DUMMY COMPLETION] {prompt[:100]}"

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[DUMMY CHAT REPLY] {last[:100]}"


def get_default_llm_client(use_dummy_if_missing_key: bool = True) -> LLMClient:
    """
    Factory to get a default LLM client for the pipeline.

    - If OPENAI_API_KEY is set -> return OpenAILLMClient.
    - If the OpenAI SDK is missing and `use_dummy_if_missing_key=True`, fall back to DummyLLMClient.
    - If no key, and use_dummy_if_missing_key=True -> return DummyLLMClient.
    - Otherwise, raise a RuntimeError.
    """
    config = get_config()
    openai_key: Optional[str] = getattr(config, "OPENAI_API_KEY", None)

    if openai_key:
        if AsyncOpenAI is None:
            if use_dummy_if_missing_key:
                # Graceful fallback if the SDK is not installed
                return DummyLLMClient()
            raise ImportError(
                "OPENAI_API_KEY is set but the openai package is not installed. "
                "Install it with: pip install openai"
            )
        return OpenAILLMClient()

    if use_dummy_if_missing_key:
        return DummyLLMClient()

    raise RuntimeError(
        "No LLM API key configured and dummy client disabled. "
        "Set OPENAI_API_KEY or call get_default_llm_client(use_dummy_if_missing_key=True)."
    )
