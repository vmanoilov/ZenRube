"""
Auto Publisher Expert Module for Zenrube MCP

This module provides an AutoPublisherExpert class that automates the entire publishing workflow
for Zenrube experts. It detects version updates, regenerates expert manifests, and publishes
them automatically to the Rube.app marketplace.

Author: vladinc@gmail.com
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import required expert modules
from zenrube.experts.version_manager import VersionManagerExpert
from zenrube.experts.publisher import PublisherExpert
from zenrube.experts.rube_adapter import RubeAdapterExpert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expert metadata - defines this expert's information
EXPERT_METADATA = {
    "name": "autopublisher",
    "version": "1.0",
    "description": "Automates expert version detection, manifest regeneration, and publication to Rube.app.",
    "author": "vladinc@gmail.com"
}


class AutoPublisherExpert:
    """
    Expert class for automating Zenrube's publishing workflow.
    
    This expert orchestrates the entire publishing process by:
    1. Detecting version updates using VersionManagerExpert
    2. Regenerating expert manifests using PublisherExpert
    3. Publishing to Rube.app using RubeAdapterExpert
    4. Recording changelog entries for all updates
    """
    
    def __init__(self):
        """
        Initialize the AutoPublisherExpert with required expert dependencies.
        """
        self.version_manager = VersionManagerExpert()
        self.publisher = PublisherExpert()
        self.rube_adapter = RubeAdapterExpert()
        self.activity_log_path = "./zenrube_activity.log"
        logger.info("AutoPublisherExpert initialized with all dependencies")
    
    def check_for_updates(self) -> Dict[str, str]:
        """
        Detect version updates by comparing current versions to last recorded changelog.
        
        This method compares the current versions of all experts stored in the version
        manager against the versions recorded in the last changelog entry for each expert.
        It identifies which experts have changed since their last recorded version.
        
        Returns:
            Dict[str, str]: Dictionary mapping expert names to their new versions.
                           Empty dict if no updates found.
        """
        self._log_activity("Starting version update detection")
        
        try:
            # Get all current expert versions from version manager
            current_versions = self.version_manager.get_expert_versions()
            logger.info(f"Current versions: {len(current_versions)} experts found")
            
            if not current_versions:
                logger.info("No current versions found - no updates to check")
                return {}
            
            # Load changelog to find last recorded versions
            changelog_versions = self._get_last_recorded_versions()
            logger.info(f"Changelog versions: {len(changelog_versions)} experts found")
            
            # Compare current versions with last recorded versions
            updated_experts = {}
            
            for expert_name, current_version in current_versions.items():
                last_recorded_version = changelog_versions.get(expert_name)
                
                if last_recorded_version is None:
                    # Expert not in changelog - consider as new (not an update)
                    logger.debug(f"Expert {expert_name} not in changelog - skipping")
                    continue
                
                if current_version != last_recorded_version:
                    # Version has changed
                    updated_experts[expert_name] = current_version
                    logger.info(f"Update detected: {expert_name} {last_recorded_version} -> {current_version}")
                else:
                    logger.debug(f"No change: {expert_name} version unchanged")
            
            if updated_experts:
                self._log_activity(f"Detected updates for {len(updated_experts)} experts: {list(updated_experts.keys())}")
            else:
                self._log_activity("No version updates detected")
                
            logger.info(f"Update check complete: {len(updated_experts)} experts have updates")
            return updated_experts
            
        except Exception as e:
            logger.error(f"Error during update detection: {e}")
            self._log_activity(f"Update detection failed: {e}")
            raise
    
    def run_autopublish(self) -> Dict[str, Any]:
        """
        Run the complete automated publishing workflow.
        
        This method orchestrates the entire publishing process:
        1. Checks for version updates
        2. Regenerates the expert manifest
        3. Publishes it to Rube.app
        4. Verifies the publication
        5. Records changelog entries for all updates
        
        Returns:
            Dict[str, Any]: Publication response from Rube.app API containing
                           status, manifest_id, and other publication details.
                           
        Raises:
            Exception: If any step in the publishing workflow fails.
        """
        self._log_activity("Starting automated publishing workflow")
        
        try:
            # Step 1: Check for version updates
            self._log_activity("Step 1: Checking for version updates")
            updated_experts = self.check_for_updates()
            
            if not updated_experts:
                self._log_activity("No updates found - skipping publication")
                return {
                    "status": "skipped",
                    "message": "No version updates detected - publication skipped",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            self._log_activity(f"Found updates for {len(updated_experts)} experts: {list(updated_experts.keys())}")
            
            # Step 2: Generate manifest
            self._log_activity("Step 2: Generating expert manifest")
            manifest = self.publisher.generate_manifest()
            
            if not manifest.get('experts'):
                raise Exception("Generated manifest is empty - no experts to publish")
            
            logger.info(f"Generated manifest with {len(manifest['experts'])} experts")
            
            # Step 3: Publish manifest
            self._log_activity("Step 3: Publishing manifest to Rube.app")
            publish_response = self.rube_adapter.publish_manifest(manifest)
            
            # Step 4: Verify publication
            self._log_activity("Step 4: Verifying publication")
            is_published = self.rube_adapter.verify_publication(publish_response)
            
            if not is_published:
                raise Exception(f"Publication verification failed: {publish_response.get('message', 'Unknown error')}")
            
            # Step 5: Record changelog entries for all updates
            self._log_activity("Step 5: Recording changelog entries")
            self._record_update_changelogs(updated_experts)
            
            # Final success logging
            manifest_id = publish_response.get('manifest_id', 'unknown')
            experts_count = publish_response.get('experts_published', 0)
            
            self._log_activity(f"Publication completed successfully - ID: {manifest_id}, Experts: {experts_count}")
            logger.info(f"Auto-publish workflow completed successfully")
            
            return publish_response
            
        except Exception as e:
            error_msg = f"Auto-publish workflow failed: {e}"
            logger.error(error_msg)
            self._log_activity(f"ERROR: {error_msg}")
            raise
    
    def _log_activity(self, message: str) -> None:
        """
        Log a human-readable activity message with timestamp to the activity log file.
        
        This method appends activity messages to a log file for audit tracking and
        debugging purposes. Each entry includes a timestamp and is formatted for
        easy human reading.
        
        Args:
            message (str): The activity message to log.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            log_entry = f"[{timestamp}] {message}\n"
            
            # Append to activity log file
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            # Also log to standard logger for real-time monitoring
            logger.info(message)
            
        except Exception as e:
            # If file logging fails, at least log to standard logger
            logger.error(f"Failed to write to activity log: {e}")
            logger.info(f"ACTIVITY: {message}")  # Fallback to standard logging
    
    def _get_last_recorded_versions(self) -> Dict[str, str]:
        """
        Get the last recorded version for each expert from the changelog.
        
        This helper method parses the changelog file to extract the most recent
        version entry for each expert. This is used to compare against current
        versions to detect updates.
        
        Returns:
            Dict[str, str]: Dictionary mapping expert names to their last recorded versions.
        """
        last_versions = {}
        
        try:
            # Load changelog if it exists
            if not os.path.exists(self.version_manager.changelog_file):
                return last_versions
            
            with open(self.version_manager.changelog_file, 'r', encoding='utf-8') as f:
                changelog_data = json.load(f)
            
            # Process changelog entries to get latest version for each expert
            for entry in changelog_data:
                if isinstance(entry, dict):
                    expert_name = entry.get('expert_name')
                    expert_version = entry.get('new_version')
                    
                    if expert_name and expert_version:
                        # Store the latest version for each expert
                        last_versions[expert_name] = expert_version
            
            logger.debug(f"Extracted {len(last_versions)} expert versions from changelog")
            return last_versions
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading changelog file: {e}")
            return last_versions
    
    def _record_update_changelogs(self, updated_experts: Dict[str, str]) -> None:
        """
        Record changelog entries for all updated experts.
        
        This helper method records changelog entries for each expert that was updated
        during the publishing process. It uses the VersionManagerExpert's record_changelog
        method with appropriate change summaries.
        
        Args:
            updated_experts (Dict[str, str]): Dictionary of expert names to their new versions.
        """
        try:
            for expert_name, new_version in updated_experts.items():
                # Create appropriate change summary based on version changes
                change_summary = f"Auto-publish: Updated during automated publication workflow"
                
                # Record the changelog entry
                self.version_manager.record_changelog(
                    expert_name=expert_name,
                    change_summary=change_summary,
                    new_version=new_version
                )
                
                self._log_activity(f"Recorded changelog for {expert_name} v{new_version}")
            
            logger.info(f"Recorded changelog entries for {len(updated_experts)} experts")
            
        except Exception as e:
            logger.error(f"Error recording changelog entries: {e}")
            raise