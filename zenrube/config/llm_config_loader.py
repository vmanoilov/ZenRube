"""
LLM Configuration Loader for Zenrube MCP

This module provides utilities for loading and saving LLM provider configuration
for the LLM Connector Expert. It follows the same architectural patterns as
other Zenrube configuration utilities.

Author: vladinc@gmail.com
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "llm_config.json"

# Default configuration template
DEFAULT_LLM_CONFIG = {
    "provider": "",
    "api_key": "",
    "model": "",
    "endpoint": ""
}


def get_llm_config() -> Optional[Dict[str, Any]]:
    """
    Load LLM configuration from JSON file.
    
    Returns:
        Optional[Dict[str, Any]]: LLM configuration dictionary, or None if file doesn't exist.
    """
    try:
        if not CONFIG_FILE.exists():
            logger.info(f"LLM config file not found at {CONFIG_FILE}")
            return None
            
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        logger.info(f"Successfully loaded LLM config from {CONFIG_FILE}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in LLM config file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load LLM config: {e}")
        return None


def save_llm_config(config: Dict[str, Any]) -> bool:
    """
    Save LLM configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure config directory exists
        CONFIG_DIR.mkdir(exist_ok=True)
        
        # Validate configuration structure
        if not _validate_config_structure(config):
            logger.error("Invalid LLM config structure")
            return False
        
        # Save configuration
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully saved LLM config to {CONFIG_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save LLM config: {e}")
        return False


def _validate_config_structure(config: Dict[str, Any]) -> bool:
    """
    Validate that the configuration has the required structure.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    required_fields = ["provider", "api_key", "model", "endpoint"]
    
    # Check if all required fields exist
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    return True


def create_default_config() -> bool:
    """
    Create a default LLM configuration file.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    return save_llm_config(DEFAULT_LLM_CONFIG)


def reset_config() -> bool:
    """
    Reset LLM configuration to default (empty) state.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    return save_llm_config(DEFAULT_LLM_CONFIG.copy())


def get_config_file_path() -> Path:
    """
    Get the path to the LLM configuration file.
    
    Returns:
        Path: Path object pointing to the config file.
    """
    return CONFIG_FILE