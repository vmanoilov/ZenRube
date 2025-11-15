"""
Compatibility Matrix for Zenrube Team Council
Manages brain compatibility and incompatibility relationships

Author: vladinc@gmail.com
"""

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class CompatibilityMatrix:
    """
    Manages compatibility relationships between different brain types.
    Provides relevance scoring and incompatibility detection.
    """
    
    def __init__(self):
        """Initialize the compatibility matrix."""
        self._setup_compatibility_matrix()
        self._setup_incompatibility_pairs()
        logger.info("CompatibilityMatrix initialized")
    
    def _setup_compatibility_matrix(self):
        """Setup compatibility scores between different brain types."""
        # Relevance matrix: scores from 0.0 (not relevant) to 1.0 (highly relevant)
        self.compatibility_matrix = {
            "cybersec": {
                "summarizer": 0.7, "systems_architect": 0.9, "security_analyst": 1.0,
                "data_cleaner": 0.6, "semantic_router": 0.8, "llm_connector": 0.8,
                "publisher": 0.5, "autopublisher": 0.3, "version_manager": 0.4,
                "rube_adapter": 0.6, "rube_orchestrator": 0.7
            },
            "coding": {
                "summarizer": 0.8, "systems_architect": 0.9, "security_analyst": 0.8,
                "data_cleaner": 0.7, "semantic_router": 0.6, "llm_connector": 0.9,
                "publisher": 0.4, "autopublisher": 0.5, "version_manager": 0.8,
                "rube_adapter": 0.7, "rube_orchestrator": 0.8
            },
            "creative": {
                "summarizer": 0.8, "systems_architect": 0.6, "security_analyst": 0.3,
                "data_cleaner": 0.4, "semantic_router": 0.9, "llm_connector": 1.0,
                "publisher": 0.9, "autopublisher": 0.8, "version_manager": 0.2,
                "rube_adapter": 0.5, "rube_orchestrator": 0.7
            },
            "feelprint": {
                "summarizer": 0.9, "systems_architect": 0.4, "security_analyst": 0.3,
                "data_cleaner": 0.5, "semantic_router": 0.8, "llm_connector": 0.9,
                "publisher": 0.7, "autopublisher": 0.6, "version_manager": 0.2,
                "rube_adapter": 0.6, "rube_orchestrator": 0.8
            },
            "hardware": {
                "summarizer": 0.7, "systems_architect": 0.9, "security_analyst": 0.7,
                "data_cleaner": 0.8, "semantic_router": 0.5, "llm_connector": 0.7,
                "publisher": 0.4, "autopublisher": 0.3, "version_manager": 0.6,
                "rube_adapter": 0.7, "rube_orchestrator": 0.8
            },
            "data": {
                "summarizer": 0.9, "systems_architect": 0.7, "security_analyst": 0.6,
                "data_cleaner": 1.0, "semantic_router": 0.9, "llm_connector": 0.9,
                "publisher": 0.4, "autopublisher": 0.4, "version_manager": 0.5,
                "rube_adapter": 0.6, "rube_orchestrator": 0.7
            },
            "business": {
                "summarizer": 0.9, "systems_architect": 0.5, "security_analyst": 0.5,
                "data_cleaner": 0.6, "semantic_router": 0.7, "llm_connector": 0.8,
                "publisher": 0.9, "autopublisher": 0.8, "version_manager": 0.3,
                "rube_adapter": 0.5, "rube_orchestrator": 0.7
            },
            "general": {
                "summarizer": 0.8, "systems_architect": 0.6, "security_analyst": 0.5,
                "data_cleaner": 0.6, "semantic_router": 0.7, "llm_connector": 0.8,
                "publisher": 0.6, "autopublisher": 0.6, "version_manager": 0.4,
                "rube_adapter": 0.6, "rube_orchestrator": 0.7
            }
        }
        
        # Default brain domain assignments
        self.brain_domains = {
            "summarizer": "general",
            "systems_architect": "coding",
            "security_analyst": "cybersec", 
            "data_cleaner": "data",
            "semantic_router": "general",
            "llm_connector": "general",
            "publisher": "business",
            "autopublisher": "business",
            "version_manager": "coding",
            "rube_adapter": "general",
            "rube_orchestrator": "general"
        }
    
    def _setup_incompatibility_pairs(self):
        """Setup incompatible brain pairs that should not work together."""
        # Brain pairs that are incompatible (will conflict or provide redundant value)
        self.incompatible_pairs = [
            ("autopublisher", "publisher"),  # Redundant functionality
            ("version_manager", "rube_adapter"),  # Different abstraction levels
            ("data_cleaner", "semantic_router"),  # Different data focus areas
            ("summarizer", "llm_connector"),  # Overlap in summarization
            ("security_analyst", "publisher"),  # Different priorities (security vs marketing)
        ]
        
        # Convert to set for faster lookup
        self.incompatible_pairs_set = set()
        for pair in self.incompatible_pairs:
            # Add both orderings
            self.incompatible_pairs_set.add((pair[0], pair[1]))
            self.incompatible_pairs_set.add((pair[1], pair[0]))
    
    def get_compatibility_score(self, domain: str, brain: str) -> float:
        """
        Get compatibility score between domain and brain.
        
        Args:
            domain (str): Task domain (e.g., "cybersec", "coding")
            brain (str): Brain expert name
            
        Returns:
            float: Compatibility score 0.0-1.0
        """
        if domain in self.compatibility_matrix and brain in self.compatibility_matrix[domain]:
            return self.compatibility_matrix[domain][brain]
        
        # Try fallback to general domain
        if "general" in self.compatibility_matrix and brain in self.compatibility_matrix["general"]:
            return self.compatibility_matrix["general"][brain] * 0.8  # Reduced score for fallback
        
        # Default compatibility if not found
        return 0.3
    
    def get_brain_domain(self, brain: str) -> str:
        """Get the domain a brain typically works in."""
        return self.brain_domains.get(brain, "general")
    
    def are_incompatible(self, brain1: str, brain2: str) -> bool:
        """
        Check if two brains are incompatible.
        
        Args:
            brain1 (str): First brain name
            brain2 (str): Second brain name
            
        Returns:
            bool: True if incompatible
        """
        return (brain1, brain2) in self.incompatible_pairs_set
    
    def filter_incompatible_brains(self, brains: List[str]) -> List[str]:
        """
        Remove incompatible brain pairs from a list.
        
        Args:
            brains (List[str]): List of brain names
            
        Returns:
            List[str]: Filtered list without incompatible pairs
        """
        compatible_brains = []
        used_brains = set()
        
        for brain in brains:
            if brain in used_brains:
                continue
            
            # Check if this brain is incompatible with any already selected
            is_compatible = True
            for used_brain in used_brains:
                if self.are_incompatible(brain, used_brain):
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible_brains.append(brain)
                used_brains.add(brain)
        
        return compatible_brains
    
    def get_top_compatible_brains(self, domain: str, available_brains: List[str], 
                                max_brains: int = 5) -> List[Tuple[str, float]]:
        """
        Get top compatible brains for a domain.
        
        Args:
            domain (str): Task domain
            available_brains (List[str]): List of available brain names
            max_brains (int): Maximum number of brains to return
            
        Returns:
            List[Tuple[str, float]]: List of (brain_name, compatibility_score) sorted by score
        """
        brain_scores = []
        
        for brain in available_brains:
            score = self.get_compatibility_score(domain, brain)
            brain_scores.append((brain, score))
        
        # Sort by score descending
        brain_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out incompatible pairs while maintaining top scores
        compatible_scores = []
        used_brains = set()
        
        for brain, score in brain_scores:
            if len(compatible_scores) >= max_brains:
                break
            
            if brain not in used_brains:
                # Check compatibility with already selected brains
                is_compatible = True
                for used_brain in used_brains:
                    if self.are_incompatible(brain, used_brain):
                        is_compatible = False
                        break
                
                if is_compatible:
                    compatible_scores.append((brain, score))
                    used_brains.add(brain)
        
        return compatible_scores
    
    def get_brain_pair_incompatibility_reason(self, brain1: str, brain2: str) -> str:
        """
        Get reason why two brains are incompatible.
        
        Args:
            brain1 (str): First brain name
            brain2 (str): Second brain name
            
        Returns:
            str: Reason for incompatibility
        """
        reasons = {
            ("autopublisher", "publisher"): "Redundant publishing functionality",
            ("version_manager", "rube_adapter"): "Different abstraction levels cause conflicts",
            ("data_cleaner", "semantic_router"): "Different data focus areas",
            ("summarizer", "llm_connector"): "Functionality overlap in summarization",
            ("security_analyst", "publisher"): "Conflicting priorities (security vs marketing)",
        }
        
        return reasons.get((brain1, brain2), "Unknown incompatibility reason")
    
    def get_domain_brain_stats(self, domain: str) -> Dict[str, any]:
        """
        Get statistics about brain compatibility for a domain.
        
        Args:
            domain (str): Domain to analyze
            
        Returns:
            Dict[str, any]: Statistics about the domain
        """
        if domain not in self.compatibility_matrix:
            return {"error": f"Unknown domain: {domain}"}
        
        scores = list(self.compatibility_matrix[domain].values())
        
        return {
            "domain": domain,
            "total_brains": len(scores),
            "average_compatibility": sum(scores) / len(scores),
            "max_compatibility": max(scores),
            "min_compatibility": min(scores),
            "high_compatibility_brains": [
                brain for brain, score in self.compatibility_matrix[domain].items() 
                if score >= 0.8
            ],
            "low_compatibility_brains": [
                brain for brain, score in self.compatibility_matrix[domain].items() 
                if score <= 0.4
            ]
        }