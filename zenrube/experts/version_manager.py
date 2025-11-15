"""
Version Manager Expert for Zenrube MCP.

This module provides versioning, changelog management, and auto-increment logic
for Zenrube experts using semantic versioning principles.
"""

import json
import os
import re
import logging
from typing import Any, Dict
from datetime import datetime, timezone

# Configure logging for the version manager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expert metadata - defines this expert's information
EXPERT_METADATA = {
    "name": "version_manager",
    "version": "1.0.0",
    "description": "Handles semantic versioning and changelog tracking for Zenrube experts.",
    "author": "vladinc@gmail.com"
}

# In-memory storage for expert versions (simulation of persistent storage)
# In a real implementation, this would be stored in a database or file
_EXPERT_VERSIONS = {}


class VersionManagerExpert:
    """
    Expert class for managing semantic versioning and changelog tracking.
    
    This expert handles version management for all Zenrube experts, including
    version retrieval, version bumping according to semantic versioning rules,
    manifest synchronization, and changelog recording.
    """
    
    def __init__(self):
        """Initialize the VersionManagerExpert."""
        self.changelog_file = "./zenrube_changelog.json"
        logger.info("VersionManagerExpert initialized")
    
    def get_current_version(self, expert_name: str) -> str:
        """
        Retrieve the current version of a specific expert.
        
        Args:
            expert_name (str): The name of the expert whose version to retrieve.
            
        Returns:
            str: The current version string in format "MAJOR.MINOR.PATCH".
                 Returns "1.0.0" if expert not found (default version).
        """
        logger.info(f"Getting current version for expert: {expert_name}")
        
        # Check if we have a stored version for this expert
        if expert_name in _EXPERT_VERSIONS:
            current_version = _EXPERT_VERSIONS[expert_name]
            logger.info(f"Found stored version for {expert_name}: {current_version}")
            return current_version
        
        # If not stored, return default version and store it
        default_version = "1.0.0"
        _EXPERT_VERSIONS[expert_name] = default_version
        logger.info(f"No stored version found for {expert_name}, returning default: {default_version}")
        return default_version
    
    def bump_version(self, expert_name: str, level: str = "patch") -> str:
        """
        Increment the version of a specific expert according to semantic versioning.
        
        Semantic Versioning: MAJOR.MINOR.PATCH
        - MAJOR: Breaking changes that are not backward compatible
        - MINOR: New features that are backward compatible
        - PATCH: Bug fixes that are backward compatible
        
        Args:
            expert_name (str): The name of the expert whose version to bump.
            level (str): The level of version bump - "major", "minor", or "patch".
                        Defaults to "patch".
        
        Returns:
            str: The new version string after incrementing.
            
        Raises:
            ValueError: If level is not one of "major", "minor", or "patch".
        """
        logger.info(f"Bumping version for {expert_name} with level: {level}")
        
        # Validate the level parameter
        valid_levels = ["major", "minor", "patch"]
        if level not in valid_levels:
            raise ValueError(f"Invalid version bump level '{level}'. Must be one of: {valid_levels}")
        
        # Get current version
        current_version = self.get_current_version(expert_name)
        
        # Parse the current version (expects format "MAJOR.MINOR.PATCH")
        try:
            version_parts = current_version.split('.')
            if len(version_parts) != 3:
                raise ValueError(f"Invalid version format: {current_version}")
            
            major, minor, patch = map(int, version_parts)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid current version '{current_version}', resetting to 1.0.0")
            major, minor, patch = 1, 0, 0
        
        # Increment the appropriate part based on the level
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        # Create new version string
        new_version = f"{major}.{minor}.{patch}"
        
        # Store the new version
        _EXPERT_VERSIONS[expert_name] = new_version
        logger.info(f"Version bumped from {current_version} to {new_version} for {expert_name}")
        
        return new_version
    
    def update_manifest_versions(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update all expert versions in a manifest to match current stored versions.
        
        This method iterates through the experts section of a manifest and updates
        each expert's version to match the current version stored in the version manager.
        
        Args:
            manifest (Dict[str, Any]): The manifest dictionary containing expert information.
                                     Expected structure: {"experts": {"expert_name": {"version": "x.x.x"}}}
        
        Returns:
            Dict[str, Any]: The updated manifest with synchronized versions.
        """
        logger.info("Updating manifest versions")
        
        # Make a copy of the manifest to avoid modifying the original
        updated_manifest = manifest.copy()
        
        # Check if manifest has experts section
        if "experts" not in updated_manifest:
            logger.warning("No 'experts' section found in manifest")
            return updated_manifest
        
        # Iterate through each expert in the manifest
        for expert_name, expert_data in updated_manifest["experts"].items():
            if isinstance(expert_data, dict) and "version" in expert_data:
                # Get the current version for this expert
                current_version = self.get_current_version(expert_name)
                
                # Update the version in the manifest
                old_version = expert_data["version"]
                expert_data["version"] = current_version
                
                logger.info(f"Updated {expert_name} version from {old_version} to {current_version}")
            else:
                logger.warning(f"Expert {expert_name} has invalid data structure or missing version")
        
        logger.info("Manifest version update completed")
        return updated_manifest
    
    def record_changelog(self, expert_name: str, change_summary: str, old_version: str = None, new_version: str = None) -> None:
        """
        Record a changelog entry for an expert version change.
        
        Creates a changelog entry that includes timestamp, expert name, old version,
        new version, and change summary. The entry is appended to a changelog file
        stored at ./zenrube_changelog.json.
        
        Args:
            expert_name (str): The name of the expert for which to record the change.
            change_summary (str): A summary of the changes made (e.g., "Fixed critical bug",
                                "Added new feature", "Breaking API changes").
            old_version (str, optional): The old version. If not provided, will try to infer
                                       from version history or use current version.
            new_version (str, optional): The new version. If not provided, will use current version.
        """
        logger.info(f"Recording changelog for {expert_name}: {change_summary}")
        
        # Get versions - try to get from history first
        if old_version is None:
            # Try to get from version history by checking all experts
            # For simplicity, we'll use the current version as old_version if not provided
            old_version = self.get_current_version(expert_name)
        
        if new_version is None:
            # Use the current stored version as new version
            new_version = _EXPERT_VERSIONS.get(expert_name, old_version)
        
        # Create changelog entry
        changelog_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expert_name": expert_name,
            "old_version": old_version,
            "new_version": new_version,
            "change_summary": change_summary,
            "author": EXPERT_METADATA["author"]
        }
        
        # Load existing changelog or create new one
        changelog_data = []
        if os.path.exists(self.changelog_file):
            try:
                with open(self.changelog_file, 'r', encoding='utf-8') as f:
                    changelog_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading changelog file: {e}. Starting with empty changelog.")
                changelog_data = []
        
        # Append new entry
        changelog_data.append(changelog_entry)
        
        # Write updated changelog back to file
        try:
            with open(self.changelog_file, 'w', encoding='utf-8') as f:
                json.dump(changelog_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Changelog entry recorded successfully for {expert_name}")
        except IOError as e:
            logger.error(f"Error writing to changelog file: {e}")
            raise
    
    def get_expert_versions(self) -> Dict[str, str]:
        """
        Get all currently stored expert versions.
        
        Returns:
            Dict[str, str]: Dictionary mapping expert names to their current versions.
        """
        logger.info("Retrieving all expert versions")
        return _EXPERT_VERSIONS.copy()
    
    def reset_expert_version(self, expert_name: str, version: str = "1.0.0") -> None:
        """
        Reset an expert's version to a specific value.
        
        Args:
            expert_name (str): The name of the expert to reset.
            version (str): The version to set. Defaults to "1.0.0".
        """
        logger.info(f"Resetting {expert_name} version to {version}")
        _EXPERT_VERSIONS[expert_name] = version
    
    def bump_version_and_record(self, expert_name: str, level: str = "patch", change_summary: str = None) -> str:
        """
        Bump an expert's version and automatically record it in the changelog.
        
        This is a convenience method that combines bump_version and record_changelog
        to provide a more convenient API for common use cases.
        
        Args:
            expert_name (str): The name of the expert whose version to bump.
            level (str): The level of version bump - "major", "minor", or "patch".
                        Defaults to "patch".
            change_summary (str, optional): A summary of the changes. If not provided,
                                          a default summary will be generated based on the level.
        
        Returns:
            str: The new version string after incrementing.
        """
        if change_summary is None:
            change_summary = f"Version bumped at {level} level"
        
        # Get old version before bumping
        old_version = self.get_current_version(expert_name)
        
        # Bump the version
        new_version = self.bump_version(expert_name, level)
        
        # Record the changelog with explicit old and new versions
        self.record_changelog(expert_name, change_summary, old_version, new_version)
        
        return new_version