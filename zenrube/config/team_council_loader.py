"""
Team Council Configuration Loader

This module provides functionality to load and manage team council configuration
for the Zenrube multi-brain orchestration system.

Author: Kilo Code
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class TeamCouncilConfigLoader:
    """
    Loader for team council configuration settings.
    
    Provides methods to load the team council configuration JSON file and
    access specific configuration values like enabled brains, synthesis settings, etc.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path (Optional[Path]): Path to the config file. If None, uses default location.
        """
        if config_path is None:
            # Default path relative to this module
            current_dir = Path(__file__).parent
            self.config_path = current_dir / "team_council_config.json"
        else:
            self.config_path = config_path
        
        self._config: Optional[Dict[str, Any]] = None
        logger.info(f"TeamCouncilConfigLoader initialized with config path: {self.config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load the team council configuration from JSON file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file contains invalid JSON.
        """
        try:
            logger.info(f"Loading team council config from: {self.config_path}")
            
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self._config = config
            logger.info(f"Successfully loaded team council config with {len(config)} keys")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load team council config: {e}")
            raise
    
    def get_enabled_brains(self) -> List[str]:
        """
        Get the list of enabled brain experts for the council.
        
        Returns:
            List[str]: List of enabled brain expert names.
        """
        config = self._config or self.load_config()
        brains = config.get("enabled_brains", [])
        
        if not brains:
            logger.warning("No enabled brains found in config, returning empty list")
            return []
        
        logger.info(f"Found {len(brains)} enabled brains: {brains}")
        return brains
    
    def get_max_brain_outputs(self) -> int:
        """
        Get the maximum number of brain outputs to process.
        
        Returns:
            int: Maximum number of brain outputs.
        """
        config = self._config or self.load_config()
        max_outputs = config.get("max_brain_outputs", 8)
        
        logger.debug(f"Max brain outputs: {max_outputs}")
        return max_outputs
    
    def get_synthesis_settings(self) -> Dict[str, Any]:
        """
        Get synthesis-related configuration settings.
        
        Returns:
            Dict[str, Any]: Dictionary containing synthesis settings.
        """
        config = self._config or self.load_config()
        settings = {
            "use_remote_llm_for_synthesis": config.get("use_remote_llm_for_synthesis", True),
            "synthesis_provider": config.get("synthesis_provider", "llm_connector"),
            "critique_style": config.get("critique_style", "blunt_constructive"),
            "roasting_enabled": config.get("roasting_enabled", True)
        }
        
        logger.info(f"Synthesis settings: {settings}")
        return settings
    
    def is_critique_enabled(self) -> bool:
        """
        Check if critique/roasting functionality is enabled.
        
        Returns:
            bool: True if critique is enabled.
        """
        settings = self.get_synthesis_settings()
        return settings.get("roasting_enabled", True)
    
    def get_critique_style(self) -> str:
        """
        Get the style for critique/roasting.
        
        Returns:
            str: Critique style string.
        """
        settings = self.get_synthesis_settings()
        return settings.get("critique_style", "blunt_constructive")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Dict[str, Any]: Complete configuration.
        """
        return self._config or self.load_config()


# Singleton instance for easy access
_config_loader: Optional[TeamCouncilConfigLoader] = None


def get_team_council_config() -> TeamCouncilConfigLoader:
    """
    Get a singleton instance of the team council config loader.
    
    Returns:
        TeamCouncilConfigLoader: Config loader instance.
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = TeamCouncilConfigLoader()
    return _config_loader


def get_enabled_brains() -> List[str]:
    """
    Convenience function to get enabled brains.
    
    Returns:
        List[str]: List of enabled brain expert names.
    """
    return get_team_council_config().get_enabled_brains()


def get_synthesis_settings() -> Dict[str, Any]:
    """
    Convenience function to get synthesis settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing synthesis settings.
    """
    return get_team_council_config().get_synthesis_settings()