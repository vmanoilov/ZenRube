"""
Dynamic Profile Engine for Zenrube Team Council
Generates intelligent brain profiles based on task classification and compatibility

Author: vladinc@gmail.com
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .classification_engine import ClassificationEngine
from .compatibility_matrix import CompatibilityMatrix

logger = logging.getLogger(__name__)


class DynamicProfileEngine:
    """
    Generates dynamic brain profiles based on task classification and compatibility analysis.
    
    Steps:
    1. Accept task + classification
    2. Pick 2-5 brains (top scoring)
    3. Add at most 1 from secondary domain
    4. Avoid incompatible pairs
    5. Always include llm_connector if enabled
    """
    
    def __init__(self, llm_connector_enabled: bool = True):
        """Initialize the dynamic profile engine."""
        self.classification_engine = ClassificationEngine()
        self.compatibility_matrix = CompatibilityMatrix()
        self.llm_connector_enabled = llm_connector_enabled
        self.min_brains = 2
        self.max_brains = 5
        
        logger.info("DynamicProfileEngine initialized")
    
    def generate_profile(self, task: str, available_brains: List[str]) -> Dict[str, Any]:
        """
        Generate a dynamic brain profile for a task.
        
        Args:
            task (str): Task description to generate profile for
            available_brains (List[str]): List of available brain expert names
            
        Returns:
            Dict[str, Any]: Generated profile with brains, scoring, and reasoning
        """
        logger.info(f"Generating profile for task: {task[:100]}...")
        
        try:
            # Step 1: Classify the task
            classification = self.classification_engine.classify_task(task)
            logger.info(f"Task classified as: {classification['primary']} (confidence: {classification['confidence']:.2f})")
            
            # Step 2: Generate draft profile
            draft_profile = self._generate_draft_profile(classification, available_brains)
            
            # Step 3: Optimize profile
            optimized_profile = self._optimize_profile(draft_profile, classification)
            
            logger.info(f"Profile generated with {len(optimized_profile['brains'])} brains")
            return optimized_profile
            
        except Exception as e:
            logger.error(f"Profile generation failed: {e}")
            return self._create_fallback_profile(task, available_brains)
    
    def _generate_draft_profile(self, classification: Dict[str, Any], available_brains: List[str]) -> Dict[str, Any]:
        """Generate a draft profile based on classification."""
        primary_domain = classification["primary"]
        secondary_domain = classification.get("secondary")
        confidence = classification["confidence"]
        
        # Step 2: Pick top compatible brains for primary domain
        primary_brains = self.compatibility_matrix.get_top_compatible_brains(
            primary_domain, available_brains, self.max_brains
        )
        
        # Filter out incompatible pairs
        primary_brains = self._filter_incompatible_brains([brain for brain, _ in primary_brains])
        
        # Step 3: Add secondary domain brain if available and compatible
        secondary_brain = None
        if secondary_domain and len(primary_brains) < self.max_brains:
            secondary_brain = self._get_secondary_domain_brain(secondary_domain, available_brains, primary_brains)
        
        # Step 4: Always include llm_connector if enabled and not already included
        brains = primary_brains.copy()
        if self.llm_connector_enabled and "llm_connector" not in brains:
            brains.append("llm_connector")
        
        # Ensure we have at least the minimum number of brains
        if len(brains) < self.min_brains:
            brains = self._ensure_minimum_brains(brains, available_brains, primary_domain)
        
        # Remove incompatible pairs from final list
        brains = self._filter_incompatible_brains(brains)
        
        return {
            "brains": brains,
            "primary_domain": primary_domain,
            "secondary_domain": secondary_domain,
            "confidence": confidence,
            "classification_method": classification.get("method", "unknown"),
            "signals": classification.get("signals", {}),
            "draft_score": self._calculate_draft_score(brains, primary_domain, confidence)
        }
    
    def _get_secondary_domain_brain(self, secondary_domain: str, available_brains: List[str], 
                                  primary_brains: List[str]) -> Optional[str]:
        """Get the best brain from secondary domain that doesn't conflict."""
        secondary_brains = self.compatibility_matrix.get_top_compatible_brains(
            secondary_domain, available_brains, 3
        )
        
        for brain, score in secondary_brains:
            if brain not in primary_brains:
                # Check if compatible with all primary domain brains
                compatible = True
                for primary_brain in primary_brains:
                    if self.compatibility_matrix.are_incompatible(brain, primary_brain):
                        compatible = False
                        break
                
                if compatible:
                    return brain
        
        return None
    
    def _ensure_minimum_brains(self, brains: List[str], available_brains: List[str], 
                             primary_domain: str) -> List[str]:
        """Ensure we have at least the minimum number of brains."""
        if len(brains) >= self.min_brains:
            return brains
        
        # Add brains from primary domain that aren't already included
        for brain in available_brains:
            if len(brains) >= self.min_brains:
                break
            if brain not in brains:
                # Check compatibility score with primary domain
                score = self.compatibility_matrix.get_compatibility_score(primary_domain, brain)
                if score >= 0.5:  # Minimum threshold for inclusion
                    brains.append(brain)
        
        return brains
    
    def _filter_incompatible_brains(self, brains: List[str]) -> List[str]:
        """Remove incompatible brain pairs from a list."""
        return self.compatibility_matrix.filter_incompatible_brains(brains)
    
    def _calculate_draft_score(self, brains: List[str], primary_domain: str, confidence: float) -> float:
        """Calculate draft profile score."""
        if not brains:
            return 0.0
        
        # Calculate average compatibility score
        compat_scores = []
        for brain in brains:
            score = self.compatibility_matrix.get_compatibility_score(primary_domain, brain)
            compat_scores.append(score)
        
        avg_compat = sum(compat_scores) / len(compat_scores) if compat_scores else 0.0
        
        # Score components
        brain_count_score = min(1.0, len(brains) / 3.0)  # Prefer 3+ brains
        diversity_score = len(set(self.compatibility_matrix.get_brain_domain(brain) for brain in brains)) / 4.0
        confidence_score = confidence
        
        # Weighted average
        draft_score = (avg_compat * 0.4 + brain_count_score * 0.3 + 
                      diversity_score * 0.2 + confidence_score * 0.1)
        
        return min(1.0, draft_score)
    
    def _optimize_profile(self, draft_profile: Dict[str, Any], classification: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the draft profile for better performance."""
        brains = draft_profile["brains"]
        primary_domain = draft_profile["primary_domain"]
        confidence = draft_profile["confidence"]
        
        # Optimize brain selection for diversity and compatibility
        optimized_brains = self._optimize_brain_selection(brains, primary_domain, classification)
        
        # Recalculate scores
        optimized_score = self._calculate_draft_score(optimized_brains, primary_domain, confidence)
        
        return {
            **draft_profile,
            "brains": optimized_brains,
            "optimized_score": optimized_score,
            "optimization_applied": optimized_brains != brains,
            "profile_id": self._generate_profile_id(optimized_brains, primary_domain)
        }
    
    def _optimize_brain_selection(self, brains: List[str], primary_domain: str, 
                                classification: Dict[str, Any]) -> List[str]:
        """Optimize brain selection for better diversity and compatibility."""
        if len(brains) <= self.min_brains:
            return brains
        
        # Score each brain for inclusion
        brain_scores = []
        for brain in brains:
            compat_score = self.compatibility_matrix.get_compatibility_score(primary_domain, brain)
            domain = self.compatibility_matrix.get_brain_domain(brain)
            
            # Bonus for domain diversity (but not too much)
            domain_bonus = 0.1 if domain != primary_domain else 0.0
            
            # Penalty for potential noise
            noise_penalty = 0.1 if self._is_potential_noise_brain(brain, primary_domain) else 0.0
            
            total_score = compat_score + domain_bonus - noise_penalty
            brain_scores.append((brain, total_score))
        
        # Sort by score and select top brains
        brain_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select brains while maintaining compatibility
        selected_brains = []
        used_brains = set()
        
        for brain, score in brain_scores:
            if len(selected_brains) >= self.max_brains:
                break
            
            # Check compatibility with already selected
            compatible = True
            for selected_brain in selected_brains:
                if self.compatibility_matrix.are_incompatible(brain, selected_brain):
                    compatible = False
                    break
            
            if compatible:
                selected_brains.append(brain)
                used_brains.add(brain)
        
        return selected_brains if len(selected_brains) >= self.min_brains else brains
    
    def _is_potential_noise_brain(self, brain: str, primary_domain: str) -> bool:
        """Check if a brain might add noise rather than value."""
        brain_domain = self.compatibility_matrix.get_brain_domain(brain)
        compatibility = self.compatibility_matrix.get_compatibility_score(primary_domain, brain)
        
        # Low compatibility brains might add noise
        return compatibility < 0.4
    
    def _generate_profile_id(self, brains: List[str], primary_domain: str) -> str:
        """Generate a unique ID for the profile."""
        brain_hash = "-".join(sorted(brains))
        return f"{primary_domain}_{hash(brain_hash) % 10000:04d}"
    
    def _create_fallback_profile(self, task: str, available_brains: List[str]) -> Dict[str, Any]:
        """Create a fallback profile when generation fails."""
        logger.warning("Using fallback profile generation")
        
        # Simple fallback: use summarizer, llm_connector, and one other brain
        fallback_brains = ["summarizer"]
        
        if self.llm_connector_enabled:
            fallback_brains.append("llm_connector")
        
        # Add one more brain if available
        for brain in available_brains:
            if brain not in fallback_brains and brain != "llm_connector":
                fallback_brains.append(brain)
                break
        
        return {
            "brains": fallback_brains,
            "primary_domain": "general",
            "secondary_domain": None,
            "confidence": 0.5,
            "classification_method": "fallback",
            "signals": {"fallback_reason": "profile_generation_failed"},
            "draft_score": 0.6,
            "optimized_score": 0.6,
            "optimization_applied": False,
            "profile_id": f"fallback_{hash(task) % 10000:04d}",
            "fallback": True
        }
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """Get statistics about the profile engine."""
        return {
            "min_brains": self.min_brains,
            "max_brains": self.max_brains,
            "llm_connector_enabled": self.llm_connector_enabled,
            "classification_engine_type": type(self.classification_engine).__name__,
            "compatibility_matrix_type": type(self.compatibility_matrix).__name__
        }