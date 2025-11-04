"""Configuration loading for zenrube-mcp."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml  # type: ignore[import-untyped]

from .experts import ExpertDefinition, register_custom_expert
from .models import SynthesisConfig

LOGGER = logging.getLogger("zenrube.config")

DEFAULT_CONFIG: Dict[str, Any] = {
    "experts": ["pragmatic_engineer", "systems_architect", "security_analyst"],
    "synthesis_style": "balanced",
    "parallel_execution": True,
    "provider": "rube",
    "logging": {"level": "INFO"},
    "cache": {"backend": "memory", "ttl": 300},
}

CONFIG_LOCATIONS = (
    Path.cwd() / ".zenrube.yml",
    Path.home() / ".zenrube.yml",
)


def _deep_update(
    target: MutableMapping[str, Any], source: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(value, Mapping) and isinstance(existing, MutableMapping):
            _deep_update(existing, value)  # type: ignore[index]
        else:
            target[key] = value
    return target


def load_config(path: Optional[os.PathLike[str]] = None) -> Dict[str, Any]:
    """Load configuration from YAML files."""

    config: Dict[str, Any] = DEFAULT_CONFIG.copy()

    locations: Iterable[Path]
    if path:
        locations = (Path(path),)
    else:
        locations = CONFIG_LOCATIONS

    for location in locations:
        if location.exists():
            try:
                with location.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                    if not isinstance(data, Mapping):
                        raise TypeError("Config root must be a mapping")
                    _deep_update(config, data)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - logging side effect
                LOGGER.warning(
                    "Failed to load config from %s: %s",
                    location,
                    exc,
                )

    _register_configured_experts(config)
    return config


def _register_configured_experts(config: Mapping[str, Any]) -> None:
    custom_experts = config.get("custom_experts", {})
    if not isinstance(custom_experts, Mapping):
        raise TypeError("custom_experts must be a mapping")

    for slug, definition in custom_experts.items():
        if not isinstance(definition, Mapping):
            raise TypeError(f"Expert definition for {slug} must be a mapping")
        register_custom_expert(
            slug,
            ExpertDefinition(
                name=definition.get("name", slug.replace("_", " ").title()),
                system_prompt=definition.get("system_prompt", ""),
                description=definition.get("description"),
                prompt_template=definition.get("prompt_template"),
            ),
        )


def build_synthesis_config(
    overrides: Optional[Mapping[str, Any]] = None,
) -> SynthesisConfig:
    base_config = load_config()
    merged: Dict[str, Any] = {
        "synthesis_style": base_config.get("synthesis_style", "balanced"),
        "parallel_execution": base_config.get("parallel_execution", True),
        "provider": base_config.get("provider", "rube"),
        "experts": list(base_config.get("experts", [])),
        "cache_ttl_seconds": base_config.get("cache", {}).get("ttl"),
        "logging_level": base_config.get("logging", {}).get("level", "INFO"),
        "debug": base_config.get("logging", {}).get("debug", False),
        "max_workers": (
            base_config.get("parallel", {}).get("max_workers")
            if isinstance(base_config.get("parallel"), Mapping)
            else None
        ),
    }
    if overrides:
        merged.update(overrides)
    return SynthesisConfig(**merged)
