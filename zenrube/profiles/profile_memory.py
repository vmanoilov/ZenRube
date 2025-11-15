"""
Profile Memory for Zenrube Team Council
Stores last 5 rejected profiles to avoid repeating bad combinations

Author: vladinc@gmail.com
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProfileMemory:
    """
    Memory system for rejected profiles.
    Stores signatures of bad profiles to avoid repeating them.
    
    Features:
    - Stores last 5 rejected profiles
    - Creates fingerprints for profile matching
    - Prevents exact profile repetitions
    - Automatic cleanup of old entries
    """
    
    def __init__(self, max_rejected: int = 5, expiry_hours: int = 24):
        """
        Initialize profile memory.
        
        Args:
            max_rejected (int): Maximum number of rejected profiles to store
            expiry_hours (int): Hours after which rejected profiles expire
        """
        self.max_rejected = max_rejected
        self.expiry_hours = expiry_hours
        self.rejected_profiles: List[Dict[str, Any]] = []
        logger.info(f"ProfileMemory initialized (max_rejected={max_rejected}, expiry_hours={expiry_hours})")
    
    def add_rejected_profile(self, profile: Dict[str, Any], reason: str) -> None:
        """
        Add a rejected profile to memory.
        
        Args:
            profile (Dict[str, Any]): The rejected profile
            reason (str): Reason for rejection
        """
        try:
            # Create profile signature
            signature = self._create_profile_signature(profile)
            
            # Check if this profile is already rejected
            if self._is_duplicate_rejection(signature):
                logger.debug(f"Profile already in rejection memory: {signature}")
                return
            
            # Create rejection record
            rejection_record = {
                "signature": signature,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "profile_fingerprint": self._create_profile_fingerprint(profile),
                "brains": profile.get("brains", []),
                "primary_domain": profile.get("primary_domain", "general"),
                "full_profile": profile  # Store for debugging
            }
            
            # Add to memory
            self.rejected_profiles.append(rejection_record)
            
            # Cleanup old entries
            self._cleanup_expired_profiles()
            
            # Enforce max limit
            while len(self.rejected_profiles) > self.max_rejected:
                self.rejected_profiles.pop(0)  # Remove oldest
            
            logger.info(f"Added rejected profile to memory: {signature} (reason: {reason})")
            
        except Exception as e:
            logger.error(f"Failed to add rejected profile: {e}")
    
    def is_rejected_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Check if a profile matches a previously rejected profile.
        
        Args:
            profile (Dict[str, Any]): Profile to check
            
        Returns:
            bool: True if profile was previously rejected
        """
        try:
            # Cleanup expired profiles first
            self._cleanup_expired_profiles()
            
            profile_signature = self._create_profile_signature(profile)
            profile_fingerprint = self._create_profile_fingerprint(profile)
            
            for rejected in self.rejected_profiles:
                # Check for exact signature match
                if rejected["signature"] == profile_signature:
                    logger.debug(f"Profile matches rejected signature: {profile_signature}")
                    return True
                
                # Check for high similarity fingerprint match (90%+ similarity)
                if self._calculate_fingerprint_similarity(
                    profile_fingerprint, rejected["profile_fingerprint"]
                ) > 0.9:
                    logger.debug(f"Profile matches rejected fingerprint similarity")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rejected profile: {e}")
            return False
    
    def get_rejection_reason(self, profile: Dict[str, Any]) -> Optional[str]:
        """
        Get the reason why a profile was rejected.
        
        Args:
            profile (Dict[str, Any]): Profile to check
            
        Returns:
            Optional[str]: Rejection reason if found
        """
        try:
            profile_signature = self._create_profile_signature(profile)
            
            for rejected in self.rejected_profiles:
                if rejected["signature"] == profile_signature:
                    return rejected["reason"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get rejection reason: {e}")
            return None
    
    def get_rejected_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all rejected profiles.
        
        Returns:
            List[Dict[str, Any]]: List of rejected profile records
        """
        return self.rejected_profiles.copy()
    
    def clear_rejected_profiles(self) -> None:
        """Clear all rejected profiles from memory."""
        self.rejected_profiles.clear()
        logger.info("All rejected profiles cleared from memory")
    
    def cleanup_old_profiles(self) -> int:
        """
        Cleanup old and expired profiles.
        
        Returns:
            int: Number of profiles removed
        """
        initial_count = len(self.rejected_profiles)
        self._cleanup_expired_profiles()
        removed_count = initial_count - len(self.rejected_profiles)
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired rejected profiles")
        
        return removed_count
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the profile memory.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        if not self.rejected_profiles:
            return {
                "total_rejected": 0,
                "memory_usage": "empty"
            }
        
        # Count by domain
        domain_counts = {}
        reason_counts = {}
        
        for rejected in self.rejected_profiles:
            domain = rejected.get("primary_domain", "unknown")
            reason = rejected.get("reason", "unknown")
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Calculate age statistics
        now = datetime.now()
        ages = []
        for rejected in self.rejected_profiles:
            try:
                timestamp = datetime.fromisoformat(rejected["timestamp"])
                age_hours = (now - timestamp).total_seconds() / 3600
                ages.append(age_hours)
            except:
                pass
        
        return {
            "total_rejected": len(self.rejected_profiles),
            "max_capacity": self.max_rejected,
            "memory_usage": f"{len(self.rejected_profiles)}/{self.max_rejected}",
            "domain_distribution": domain_counts,
            "reason_distribution": reason_counts,
            "age_statistics": {
                "average_age_hours": sum(ages) / len(ages) if ages else 0,
                "oldest_age_hours": max(ages) if ages else 0,
                "newest_age_hours": min(ages) if ages else 0
            },
            "expiry_hours": self.expiry_hours
        }
    
    def _create_profile_signature(self, profile: Dict[str, Any]) -> str:
        """Create a unique signature for a profile."""
        brains = sorted(profile.get("brains", []))
        primary_domain = profile.get("primary_domain", "general")
        
        # Create signature string
        signature_str = f"{primary_domain}:{'-'.join(brains)}"
        
        # Hash it for consistency
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    def _create_profile_fingerprint(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a more detailed fingerprint for similarity matching."""
        return {
            "brains": sorted(profile.get("brains", [])),
            "primary_domain": profile.get("primary_domain", "general"),
            "brain_count": len(profile.get("brains", [])),
            "has_synthesis": any(brain in ["llm_connector", "summarizer"] for brain in profile.get("brains", [])),
            "has_security": "security_analyst" in profile.get("brains", []),
            "has_data": "data_cleaner" in profile.get("brains", []),
            "has_creative": "semantic_router" in profile.get("brains", [])
        }
    
    def _calculate_fingerprint_similarity(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """Calculate similarity between two fingerprints."""
        if fp1["brain_count"] == 0 or fp2["brain_count"] == 0:
            return 0.0
        
        # Domain match bonus
        domain_bonus = 1.0 if fp1["primary_domain"] == fp2["primary_domain"] else 0.0
        
        # Brain overlap ratio
        brains1 = set(fp1["brains"])
        brains2 = set(fp2["brains"])
        overlap = len(brains1.intersection(brains2))
        union = len(brains1.union(brains2))
        overlap_ratio = overlap / union if union > 0 else 0.0
        
        # Capability matches
        capability_matches = 0
        capability_checks = ["has_synthesis", "has_security", "has_data", "has_creative"]
        
        for check in capability_checks:
            if fp1.get(check) == fp2.get(check):
                capability_matches += 1
        
        capability_score = capability_matches / len(capability_checks)
        
        # Weighted combination
        similarity = (
            domain_bonus * 0.3 +
            overlap_ratio * 0.5 +
            capability_score * 0.2
        )
        
        return min(1.0, similarity)
    
    def _is_duplicate_rejection(self, signature: str) -> bool:
        """Check if a rejection with this signature already exists."""
        return any(rejected["signature"] == signature for rejected in self.rejected_profiles)
    
    def _cleanup_expired_profiles(self) -> None:
        """Remove expired profiles from memory."""
        if not self.rejected_profiles:
            return
        
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.expiry_hours)
        
        # Filter out expired profiles
        valid_profiles = []
        
        for rejected in self.rejected_profiles:
            try:
                timestamp = datetime.fromisoformat(rejected["timestamp"])
                if timestamp > cutoff_time:
                    valid_profiles.append(rejected)
            except Exception:
                # If we can't parse the timestamp, keep the profile
                valid_profiles.append(rejected)
        
        self.rejected_profiles = valid_profiles
    
    def export_rejected_profiles(self, format: str = "json") -> str:
        """
        Export rejected profiles for analysis.
        
        Args:
            format (str): Export format ("json" or "text")
            
        Returns:
            str: Exported rejected profiles
        """
        if format.lower() == "json":
            return json.dumps(self.rejected_profiles, indent=2, ensure_ascii=False)
        elif format.lower() == "text":
            lines = []
            for rejected in self.rejected_profiles:
                signature = rejected["signature"]
                reason = rejected["reason"]
                timestamp = rejected["timestamp"]
                domain = rejected["primary_domain"]
                brains = ", ".join(rejected["brains"])
                lines.append(f"[{timestamp}] {domain} - {signature}: {reason}")
                lines.append(f"  Brains: {brains}")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")