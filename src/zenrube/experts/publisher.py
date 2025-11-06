"""
Publisher Expert Module for Zenrube MCP

This module provides a PublisherExpert class that connects Zenrube's internal ExpertRegistry
with the Rube.app marketplace system. It collects all available expert metadata, structures
it into a manifest, and prepares it for publishing or export.

Author: vladinc@gmail.com
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from .expert_registry import ExpertRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level metadata for the publisher expert
EXPERT_METADATA = {
    "name": "publisher",
    "version": "1.0",
    "description": "Generates and validates Zenrube expert manifests for Rube.app marketplace publishing.",
    "author": "vladinc@gmail.com"
}


class PublisherExpert:
    """
    Expert class for generating and managing Zenrube expert manifests.
    
    This class connects the ExpertRegistry with marketplace publishing by collecting
    all available expert metadata and structuring it into a standardized manifest format
    suitable for the Rube.app marketplace.
    """
    
    def __init__(self):
        """
        Initialize the PublisherExpert with an ExpertRegistry instance.
        """
        self.registry = ExpertRegistry()
        logger.info("PublisherExpert initialized with ExpertRegistry")
    
    def generate_manifest(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive manifest of all available experts.
        
        This method discovers all experts using the ExpertRegistry, collects their
        EXPERT_METADATA, and structures it into a standardized manifest format.
        
        Args:
            output_path (Optional[str]): If provided, writes the manifest as JSON to this path.
        
        Returns:
            Dict[str, Any]: The generated manifest dictionary containing expert information.
                          Structure: {
                              "manifest_version": "1.0",
                              "publisher": "vladinc@gmail.com", 
                              "experts": [list of expert metadata]
                          }
        """
        logger.info("Generating expert manifest")
        
        try:
            # Discover all available experts
            discovered_experts = self.registry.discover_experts()
            
            if not discovered_experts:
                logger.warning("No experts discovered - manifest will be empty")
                manifest = {
                    "manifest_version": "1.0",
                    "publisher": "vladinc@gmail.com",
                    "experts": []
                }
            else:
                # Collect metadata for each discovered expert
                expert_metadata_list = []
                
                for expert_name in discovered_experts.keys():
                    try:
                        expert_info = self.registry.get_expert_info(expert_name)
                        if expert_info:
                            # Extract only the required fields for the manifest
                            expert_entry = {
                                "name": expert_info['name'],
                                "version": expert_info['version'], 
                                "description": expert_info['description'],
                                "author": expert_info['author']
                            }
                            expert_metadata_list.append(expert_entry)
                            logger.debug(f"Added expert to manifest: {expert_name}")
                        else:
                            logger.warning(f"Failed to get info for expert: {expert_name}")
                            
                    except Exception as e:
                        logger.error(f"Error collecting metadata for expert {expert_name}: {e}")
                        continue
                
                manifest = {
                    "manifest_version": "1.0",
                    "publisher": "vladinc@gmail.com",
                    "experts": expert_metadata_list
                }
            
            logger.info(f"Generated manifest with {len(manifest['experts'])} experts")
            
            # Write to file if output path is provided
            if output_path:
                self._write_manifest_to_file(manifest, output_path)
            
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to generate manifest: {e}")
            # Return a basic manifest structure even on error
            return {
                "manifest_version": "1.0",
                "publisher": "vladinc@gmail.com",
                "experts": []
            }
    
    def validate_manifest(self, manifest: Dict[str, Any]) -> bool:
        """
        Validate the structure and content of a manifest dictionary.
        
        This method checks that the manifest has all required keys and that each
        expert entry contains the necessary fields. It logs warnings for any
        missing or invalid data.
        
        Args:
            manifest (Dict[str, Any]): The manifest dictionary to validate.
        
        Returns:
            bool: True if the manifest is valid, False otherwise.
        """
        logger.info("Validating manifest structure and content")
        
        if not isinstance(manifest, dict):
            logger.error("Manifest is not a dictionary")
            return False
        
        # Check for required top-level keys
        required_keys = ['manifest_version', 'publisher', 'experts']
        missing_keys = [key for key in required_keys if key not in manifest]
        
        if missing_keys:
            logger.warning(f"Manifest missing required keys: {missing_keys}")
            return False
        
        # Validate manifest_version
        if not isinstance(manifest['manifest_version'], str) or not manifest['manifest_version']:
            logger.warning("Manifest missing or invalid 'manifest_version'")
            return False
        
        # Validate publisher
        if not isinstance(manifest['publisher'], str) or not manifest['publisher']:
            logger.warning("Manifest missing or invalid 'publisher'")
            return False
        
        # Validate experts list
        if not isinstance(manifest['experts'], list):
            logger.warning("Manifest 'experts' is not a list")
            return False
        
        # Validate each expert entry
        valid_experts = 0
        required_expert_keys = ['name', 'version', 'description', 'author']
        
        for i, expert in enumerate(manifest['experts']):
            if not isinstance(expert, dict):
                logger.warning(f"Expert entry {i} is not a dictionary")
                continue
            
            # Check for required expert keys
            missing_expert_keys = [key for key in required_expert_keys if key not in expert]
            empty_expert_keys = []
            
            for key in required_expert_keys:
                if key in expert:
                    if not expert[key] or not str(expert[key]).strip():
                        empty_expert_keys.append(key)
            
            # Report issues for this expert
            if missing_expert_keys:
                logger.warning(f"Expert {i} missing required keys: {missing_expert_keys}")
            if empty_expert_keys:
                logger.warning(f"Expert {i} has empty keys: {empty_expert_keys}")
            
            # Validate specific data types
            if 'version' in expert and not isinstance(expert['version'], str):
                logger.warning(f"Expert {i} has invalid version type (should be string)")
                continue
            
            # If no issues found, count as valid
            if not missing_expert_keys and not empty_expert_keys:
                valid_experts += 1
            else:
                logger.warning(f"Expert {i} validation failed")
        
        logger.info(f"Manifest validation: {valid_experts}/{len(manifest['experts'])} experts valid")
        return valid_experts == len(manifest['experts'])
    
    def export_manifest(self, output_path: str) -> None:
        """
        Write the generated manifest to a file.
        
        This method creates a complete manifest by calling generate_manifest() and
        then exports it to the specified file path as JSON.
        
        Args:
            output_path (str): Path where the manifest should be written.
                            Default: ./zenrube_manifest.json
        """
        if not output_path:
            output_path = "./zenrube_manifest.json"
            logger.info(f"No output path specified, using default: {output_path}")
        
        try:
            # Generate the manifest (this will also write to file if we pass output_path)
            manifest = self.generate_manifest(output_path)
            
            # Verify the file was written successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Successfully exported manifest to {output_path} ({file_size} bytes)")
            else:
                logger.error(f"Failed to create manifest file at {output_path}")
                
        except Exception as e:
            logger.error(f"Failed to export manifest: {e}")
            raise
    
    def _write_manifest_to_file(self, manifest: Dict[str, Any], output_path: str) -> None:
        """
        Internal method to write a manifest dictionary to a JSON file.
        
        Args:
            manifest (Dict[str, Any]): The manifest dictionary to write.
            output_path (str): Path where the file should be written.
        
        Raises:
            Exception: If file writing fails.
        """
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.debug(f"Created output directory: {output_dir}")
            
            # Write the manifest as JSON with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Manifest written to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write manifest to {output_path}: {e}")
            raise