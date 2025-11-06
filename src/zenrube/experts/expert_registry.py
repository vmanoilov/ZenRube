"""
Expert Registry Module for Zenrube MCP

This module provides the ExpertRegistry class which acts as Zenrube's central brain registry
for discovering, validating, and loading all available expert modules.

Author: vladinc@gmail.com
"""

import os
import importlib
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level metadata for the expert registry itself
EXPERT_METADATA = {
    "name": "expert_registry",
    "version": "1.0",
    "description": "Discovers, validates, and loads available Zenrube expert modules.",
    "author": "vladinc@gmail.com"
}


class ExpertRegistry:
    """
    Central registry for discovering, validating, and loading Zenrube expert modules.
    
    This class provides functionality to scan the experts directory, validate module
    metadata, and dynamically import expert classes for use in the Zenrube system.
    """
    
    def __init__(self, experts_dir: Optional[str] = None):
        """
        Initialize the ExpertRegistry.
        
        Args:
            experts_dir (Optional[str]): Path to the experts directory. 
                                       Defaults to src/zenrube/experts/ relative to current file.
        """
        if experts_dir is None:
            # Get the directory containing this file and navigate to experts directory
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            self.experts_dir = current_file_dir
        else:
            self.experts_dir = experts_dir
        
        logger.info(f"ExpertRegistry initialized with experts directory: {self.experts_dir}")
    
    def discover_experts(self) -> Dict[str, str]:
        """
        Discovers all expert modules in the experts directory.
        
        Scans the src/zenrube/experts/ directory for .py files (excluding __init__.py 
        and expert_registry.py), imports each module dynamically, validates that each 
        module has valid EXPERT_METADATA, and returns a dict mapping expert name to 
        module path.
        
        Returns:
            Dict[str, str]: Dictionary mapping expert name to module path.
                           Example: {"data_cleaner": "zenrube.experts.data_cleaner"}
        
        Raises:
            Exception: If directory scanning fails or critical import issues occur.
        """
        logger.info(f"Scanning experts directory: {self.experts_dir}")
        
        if not os.path.exists(self.experts_dir):
            logger.error(f"Experts directory does not exist: {self.experts_dir}")
            return {}
        
        if not os.path.isdir(self.experts_dir):
            logger.error(f"Path is not a directory: {self.experts_dir}")
            return {}
        
        discovered_experts = {}
        
        try:
            # Get all .py files in the directory
            python_files = [f for f in os.listdir(self.experts_dir) 
                          if f.endswith('.py') and f not in ['__init__.py', 'expert_registry.py']]
            
            logger.info(f"Found {len(python_files)} Python files to scan: {python_files}")
            
            for filename in python_files:
                try:
                    # Remove .py extension to get module name
                    module_name = filename[:-3]  # Remove '.py'
                    
                    # Construct the full module path for dynamic import
                    module_path = f"zenrube.experts.{module_name}"
                    
                    # Dynamically import the module
                    module = importlib.import_module(module_path)
                    
                    # Validate the module has required metadata
                    if self.validate_metadata(module):
                        # Get expert name from metadata
                        expert_name = getattr(module.EXPERT_METADATA, 'name', module_name)
                        discovered_experts[expert_name] = module_path
                        logger.info(f"Successfully discovered expert: {expert_name} ({module_path})")
                    else:
                        logger.warning(f"Skipping invalid module: {module_name}")
                        
                except ModuleNotFoundError as e:
                    logger.error(f"Failed to import module {module_name}: {e}")
                except AttributeError as e:
                    logger.error(f"Module {module_name} missing required attributes: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing module {module_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to scan experts directory: {e}")
            raise
        
        logger.info(f"Discovered {len(discovered_experts)} valid experts: {list(discovered_experts.keys())}")
        return discovered_experts
    
    def load_expert(self, name: str) -> Any:
        """
        Dynamically loads and returns an instance of the specified expert.
        
        Args:
            name (str): The name of the expert to load.
        
        Returns:
            Any: An instance of the expert class.
        
        Raises:
            ModuleNotFoundError: If the expert module is not found.
            AttributeError: If the expert class is not found in the module.
            Exception: If expert instantiation fails.
        """
        logger.info(f"Loading expert: {name}")
        
        # First, discover experts to get the module path
        discovered_experts = self.discover_experts()
        
        if name not in discovered_experts:
            available_experts = list(discovered_experts.keys())
            raise ModuleNotFoundError(
                f"Expert '{name}' not found. Available experts: {available_experts}"
            )
        
        module_path = discovered_experts[name]
        
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)
            
            # Determine the expert class name (capitalize first letter of expert name)
            class_name = ''.join(word.capitalize() for word in name.split('_')) + 'Expert'
            
            # Get the expert class from the module
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Expert class '{class_name}' not found in module {module_path}"
                )
            
            expert_class = getattr(module, class_name)
            
            # Instantiate the expert class
            expert_instance = expert_class()
            
            logger.info(f"Successfully loaded expert: {name} (class: {class_name})")
            return expert_instance
            
        except ModuleNotFoundError as e:
            logger.error(f"Failed to import expert module {name}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Expert class not found for {name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate expert {name}: {e}")
            raise
    
    def validate_metadata(self, module: Any) -> bool:
        """
        Validates that a module has the required EXPERT_METADATA and all necessary keys.
        
        Checks for EXPERT_METADATA presence and validates that all required keys
        (name, version, description, author) are present and non-empty. Logs warnings
        if metadata is invalid and returns False.
        
        Args:
            module (Any): The imported module to validate.
        
        Returns:
            bool: True if metadata is valid, False otherwise.
        """
        logger.debug(f"Validating metadata for module: {module.__name__}")
        
        # Check if EXPERT_METADATA exists
        if not hasattr(module, 'EXPERT_METADATA'):
            logger.warning(f"Module {module.__name__} missing EXPERT_METADATA")
            return False
        
        metadata = module.EXPERT_METADATA
        
        # Check if EXPERT_METADATA is a dictionary
        if not isinstance(metadata, dict):
            logger.warning(f"Module {module.__name__} EXPERT_METADATA is not a dictionary")
            return False
        
        # Required keys in metadata
        required_keys = ['name', 'version', 'description', 'author']
        
        # Validate each required key
        missing_keys = []
        empty_keys = []
        
        for key in required_keys:
            if key not in metadata:
                missing_keys.append(key)
            elif not metadata[key] or not str(metadata[key]).strip():
                empty_keys.append(key)
        
        # Report validation issues
        if missing_keys:
            logger.warning(f"Module {module.__name__} EXPERT_METADATA missing keys: {missing_keys}")
        
        if empty_keys:
            logger.warning(f"Module {module.__name__} EXPERT_METADATA has empty keys: {empty_keys}")
        
        # Return False if there are any issues
        if missing_keys or empty_keys:
            return False
        
        # Validate specific data types and formats
        if not isinstance(metadata['version'], str):
            logger.warning(f"Module {module.__name__} version should be a string")
            return False
        
        # If all validations pass
        logger.debug(f"Module {module.__name__} metadata validation passed")
        return True
    
    def list_available_experts(self) -> List[str]:
        """
        Get a list of all available expert names.
        
        Returns:
            List[str]: List of available expert names.
        """
        discovered_experts = self.discover_experts()
        return list(discovered_experts.keys())
    
    def get_expert_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific expert.
        
        Args:
            name (str): The name of the expert to get info about.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing expert metadata and info,
                                    or None if expert not found.
        """
        try:
            discovered_experts = self.discover_experts()
            
            if name not in discovered_experts:
                return None
            
            module_path = discovered_experts[name]
            module = importlib.import_module(module_path)
            
            if not self.validate_metadata(module):
                return None
            
            return {
                'name': module.EXPERT_METADATA['name'],
                'version': module.EXPERT_METADATA['version'],
                'description': module.EXPERT_METADATA['description'],
                'author': module.EXPERT_METADATA['author'],
                'module_path': module_path,
                'class_name': ''.join(word.capitalize() for word in name.split('_')) + 'Expert'
            }
            
        except Exception as e:
            logger.error(f"Failed to get expert info for {name}: {e}")
            return None