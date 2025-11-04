from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from zenrube import cache, config, experts, providers
import zenrube


class EchoProvider(providers.LLMProvider):
    name = "echo"

    def query(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **_: Any,
    ) -> Tuple[str, Dict[str, Optional[str]]]:
        return prompt, {"model": model}


def test_cache_manager_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    cache.CacheManager.configure(cache.InMemoryCache(), ttl=1)
    key = cache.build_cache_key("demo", "memory")
    cache.CacheManager.set(key, {"value": 1})
    assert cache.CacheManager.get(key) == {"value": 1}

    original_time = cache.time.time

    def fake_time() -> float:
        return original_time() + 10

    monkeypatch.setattr(cache.time, "time", fake_time)
    assert cache.CacheManager.get(key) is None
    monkeypatch.setattr(cache.time, "time", original_time)


def test_cache_manager_file(tmp_path: Path) -> None:
    cache.CacheManager.configure(cache.FileCache(tmp_path), ttl=1)
    key = cache.build_cache_key("demo", "file")
    cache.CacheManager.set(key, {"value": 42})
    assert cache.CacheManager.get(key) == {"value": 42}


def test_load_config_registers_custom_expert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_content = """
custom_experts:
  test_custom:
    name: Test Custom
    system_prompt: You respond with test data
logging:
  level: debug
"""
    config_path = tmp_path / ".zenrube.yml"
    config_path.write_text(yaml_content, encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    data = config.load_config(config_path)
    assert data["logging"]["level"].upper() == "DEBUG"
    expert = experts.get_expert("test_custom")
    assert "test data" in expert.system_prompt


def test_build_synthesis_config_overrides() -> None:
    overrides: Dict[str, Any] = {
        "synthesis_style": "critical",
        "parallel_execution": False,
        "provider": "echo",
        "experts": ["pragmatic_engineer"],
        "logging_level": "warning",
    }
    synthesis_config = config.build_synthesis_config(overrides)
    assert synthesis_config.synthesis_style == "critical"
    assert synthesis_config.parallel_execution is False
    assert synthesis_config.provider == "echo"
    assert synthesis_config.logging_level == "WARNING"


def test_provider_registry_and_configure_rube_client() -> None:
    providers.ProviderRegistry.register(EchoProvider())
    providers.ProviderRegistry.set_default("echo")
    provider = providers.ProviderRegistry.get()
    assert isinstance(provider, EchoProvider)

    result, metadata = provider.query("ping", model="test-model")
    assert result == "ping"
    assert metadata["model"] == "test-model"

    def dummy_client(
        prompt: str,
        *,
        model: Optional[str] = None,
        **_: Any,
    ) -> Tuple[str, Dict[str, Optional[str]]]:
        return f"dummy:{prompt}", {"model": model}

    zenrube.configure_rube_client(dummy_client)
    response, meta = zenrube.invoke_llm("hello", provider="rube")
    assert response == "dummy:hello"
    assert meta["model"] is None
