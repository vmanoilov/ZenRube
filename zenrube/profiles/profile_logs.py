"""
Profile Logs for Zenrube Team Council
Simple logging system for profile validation and generation events

Author: vladinc@gmail.com
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ProfileLogs:
    """
    Simple logging system for profile-related events.
    Provides basic logging functionality without heavy dependencies.
    """
    
    def __init__(self, log_file: str = "zenrube_activity.log"):
        """Initialize the profile logs."""
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        logger.info("ProfileLogs initialized")
    
    def add_log(self, log_entry: Dict[str, Any]) -> None:
        """
        Add a log entry to the profile logs.
        
        Args:
            log_entry (Dict[str, Any]): Log entry to add
        """
        try:
            # Add timestamp and format
            timestamp = datetime.now().isoformat()
            formatted_entry = {
                "timestamp": timestamp,
                "type": "profile_log",
                **log_entry
            }
            
            self.logs.append(formatted_entry)
            
            # Keep only last 100 logs to prevent memory issues
            if len(self.logs) > 100:
                self.logs = self.logs[-100:]
            
            # Also log to regular logger
            logger.info(f"Profile Log: {log_entry.get('action', 'unknown')} - {log_entry.get('profile_id', 'no_id')}")
            
        except Exception as e:
            logger.error(f"Failed to add log entry: {e}")
    
    def get_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get profile logs.
        
        Args:
            limit (Optional[int]): Maximum number of logs to return
            
        Returns:
            List[Dict[str, Any]]: List of log entries
        """
        if limit:
            return self.logs[-limit:] if len(self.logs) > limit else self.logs
        return self.logs.copy()
    
    def get_logs_by_action(self, action: str) -> List[Dict[str, Any]]:
        """
        Get logs filtered by action type.
        
        Args:
            action (str): Action type to filter by
            
        Returns:
            List[Dict[str, Any]]: Filtered log entries
        """
        return [log for log in self.logs if log.get("action") == action]
    
    def get_logs_by_profile_id(self, profile_id: str) -> List[Dict[str, Any]]:
        """
        Get logs filtered by profile ID.
        
        Args:
            profile_id (str): Profile ID to filter by
            
        Returns:
            List[Dict[str, Any]]: Filtered log entries
        """
        return [log for log in self.logs if log.get("profile_id") == profile_id]
    
    def export_logs(self, format: str = "json") -> str:
        """
        Export logs in specified format.
        
        Args:
            format (str): Export format ("json" or "text")
            
        Returns:
            str: Exported logs as string
        """
        if format.lower() == "json":
            return json.dumps(self.logs, indent=2, ensure_ascii=False)
        elif format.lower() == "text":
            lines = []
            for log in self.logs:
                timestamp = log.get("timestamp", "unknown")
                action = log.get("action", "unknown")
                profile_id = log.get("profile_id", "no_id")
                lines.append(f"[{timestamp}] {action}: {profile_id}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_logs(self) -> None:
        """Clear all logs."""
        self.logs.clear()
        logger.info("Profile logs cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the logs.
        
        Returns:
            Dict[str, Any]: Statistics about logged events
        """
        if not self.logs:
            return {"total_logs": 0}
        
        action_counts = {}
        status_counts = {}
        
        for log in self.logs:
            action = log.get("action", "unknown")
            status = log.get("status", "unknown")
            
            action_counts[action] = action_counts.get(action, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_logs": len(self.logs),
            "actions": action_counts,
            "statuses": status_counts,
            "date_range": {
                "earliest": self.logs[0].get("timestamp") if self.logs else None,
                "latest": self.logs[-1].get("timestamp") if self.logs else None
            }
        }
    
    def log_profile_generation(self, profile_id: str, brains_count: int, 
                             primary_domain: str, method: str) -> None:
        """Log profile generation event."""
        self.add_log({
            "action": "profile_generation",
            "profile_id": profile_id,
            "brains_count": brains_count,
            "primary_domain": primary_domain,
            "method": method,
            "status": "generated"
        })
    
    def log_profile_validation(self, profile_id: str, validation_passed: bool, 
                             issues: List[str] = None) -> None:
        """Log profile validation event."""
        self.add_log({
            "action": "profile_validation",
            "profile_id": profile_id,
            "validation_passed": validation_passed,
            "issues": issues or [],
            "status": "passed" if validation_passed else "failed"
        })
    
    def log_profile_rejection(self, profile_id: str, reason: str) -> None:
        """Log profile rejection event."""
        self.add_log({
            "action": "profile_rejection",
            "profile_id": profile_id,
            "reason": reason,
            "status": "rejected"
        })
    
    def log_auto_repair(self, profile_id: str, original_brains: int, 
                       repaired_brains: int, changes: List[str]) -> None:
        """Log auto-repair event."""
        self.add_log({
            "action": "profile_auto_repair",
            "profile_id": profile_id,
            "original_brains": original_brains,
            "repaired_brains": repaired_brains,
            "changes": changes,
            "status": "repaired"
        })
    
    def search_logs(self, query: str) -> List[Dict[str, Any]]:
        """
        Search logs for entries containing the query.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, Any]]: Matching log entries
        """
        query_lower = query.lower()
        matches = []
        
        for log in self.logs:
            # Search in all string values
            log_str = json.dumps(log).lower()
            if query_lower in log_str:
                matches.append(log)
        
        return matches