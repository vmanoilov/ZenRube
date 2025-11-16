"""
Embeddings Configuration Loader for ZenRube

This module provides utilities for loading embeddings provider configuration.
It supports environment variable resolution for API keys.

Author: ZenRube Core Engineer
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_DIR = Path(__file__).parent.parent.parent / "data"
CONFIG_FILE = CONFIG_DIR / "embeddings_config.json"

# Default configuration template
DEFAULT_EMBEDDINGS_CONFIG = {
    "provider": "openai",
    "api_key": "",
    "model": "text-embedding-3-small",
    "base_url": "https://api.openai.com/v1"
}


def load_config() -> Optional[Dict[str, Any]]:
    """
    Load embeddings configuration from JSON file.

    Returns:
        Optional[Dict[str, Any]]: Embeddings configuration dictionary, or None if file doesn't exist.
    """
    try:
        if not CONFIG_FILE.exists():
            logger.warning(f"Embeddings config file not found at {CONFIG_FILE}")
            return None

        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Resolve environment variables
        config = resolve_env_var(config)

        logger.info(f"Successfully loaded embeddings config from {CONFIG_FILE}")
        return config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in embeddings config file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load embeddings config: {e}")
        return None


def resolve_env_var(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in config values that start with 'ENV:'.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with resolved environment variables
    """
    resolved = config.copy()

    for key, value in resolved.items():
        if isinstance(value, str) and value.startswith("ENV:"):
            env_var = value[4:]  # Remove "ENV:" prefix
            resolved_value = os.environ.get(env_var)
            if resolved_value is None:
                logger.warning(f"Environment variable '{env_var}' not found for config key '{key}'")
                resolved[key] = ""
            else:
                resolved[key] = resolved_value
                logger.debug(f"Resolved environment variable '{env_var}' for config key '{key}'")

    return resolved


def save_embeddings_config(config: Dict[str, Any]) -> bool:
    """
    Save embeddings configuration to JSON file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure config directory exists
        CONFIG_DIR.mkdir(exist_ok=True)

        # Save configuration
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved embeddings config to {CONFIG_FILE}")
        return True

    except Exception as e:
        logger.error(f"Failed to save embeddings config: {e}")
        return False


def create_default_config() -> bool:
    """
    Create a default embeddings configuration file.

    Returns:
        bool: True if successful, False otherwise.
    """
    return save_embeddings_config(DEFAULT_EMBEDDINGS_CONFIG)


def get_config_file_path() -> Path:
    """
    Get the path to the embeddings configuration file.

    Returns:
        Path: Path object pointing to the config file.
    """
    return CONFIG_FILE