"""LLM provider abstractions for zenrube."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional

LOGGER = logging.getLogger("zenrube.providers")


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def query(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class RubeProvider(LLMProvider):
    name = "rube"

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    def set_client(self, client: Any) -> None:
        self.client = client

    def query(
        self, prompt: str, *, model: Optional[str] = None, **kwargs: Any
    ) -> Any:  # pragma: no cover - integration only
        if self.client is None:
            raise RuntimeError("Rube invoke_llm client not configured")
        return self.client(prompt=prompt, model=model, **kwargs)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self, client: Optional[Any] = None, model_fallback: str = "gpt-4o-mini"
    ) -> None:
        self.client = client
        self.model_fallback = model_fallback

    def query(
        self, prompt: str, *, model: Optional[str] = None, **kwargs: Any
    ) -> Any:  # pragma: no cover - integration only
        if self.client is None:
            try:
                import openai  # type: ignore
            except ImportError as exc:
                raise RuntimeError("openai package not installed") from exc
            self.client = openai.ChatCompletion

        model_name = model or self.model_fallback
        response = self.client.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response["choices"][0]["message"]["content"], response


class ProviderRegistry:
    _providers: ClassVar[Dict[str, LLMProvider]] = {}
    _default: ClassVar[str] = "rube"

    @classmethod
    def register(cls, provider: LLMProvider) -> None:
        cls._providers[provider.name] = provider
        LOGGER.debug("Registered provider %s", provider.name)

    @classmethod
    def configure(cls, provider_name: str, provider: LLMProvider) -> None:
        cls._providers[provider_name] = provider
        LOGGER.debug("Configured provider %s", provider_name)

    @classmethod
    def get(cls, provider_name: Optional[str] = None) -> LLMProvider:
        name = provider_name or cls._default
        if name not in cls._providers:
            raise KeyError(f"Provider '{name}' is not registered")
        return cls._providers[name]

    @classmethod
    def set_default(cls, provider_name: str) -> None:
        if provider_name not in cls._providers:
            raise KeyError(f"Provider '{provider_name}' is not registered")
        cls._default = provider_name
        LOGGER.debug("Set default provider to %s", provider_name)

    @classmethod
    def ensure_default(cls) -> None:
        if cls._default not in cls._providers:
            cls.register(RubeProvider())

    @classmethod
    def available(cls) -> Dict[str, LLMProvider]:
        return dict(cls._providers)


# Ensure default provider is ready
ProviderRegistry.ensure_default()
ProviderRegistry.register(OpenAIProvider())
