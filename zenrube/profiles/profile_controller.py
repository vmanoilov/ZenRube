"""
Profile Controller for Zenrube Team Council
Lean but powerful profile validation, scoring, and auto-repair system

Author: vladinc@gmail.com
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .dynamic_profile_engine import DynamicProfileEngine
from .profile_logs import ProfileLogs
from .profile_memory import ProfileMemory

logger = logging.getLogger(__name__)


class ProfileController:
    """
    Profile validation, scoring, and auto-repair controller.
    
    Features:
    1. Structure validation (2-5 brains, synthesis brain required)
    2. Dry-run relevance test
    3. Auto-repair (one retry)
    4. Lean scoring (0-10)
    5. Roast tone governor
    6. Mode context merger
    7. Bad profile memory
    8. Human profile summary
    9. Coherence fuse
    """
    
    def __init__(self, llm_connector_enabled: bool = True):
        """Initialize the profile controller."""
        self.profile_engine = DynamicProfileEngine(llm_connector_enabled)
        self.profile_logs = ProfileLogs()
        self.profile_memory = ProfileMemory()
        self.roast_governor = RoastToneGovernor()
        
        # Scoring weights
        self.scoring_weights = {
            "domain_relevance": 0.40,
            "compatibility": 0.30, 
            "dry_run_relevance": 0.20,
            "noise_potential": 0.10
        }
        
        # Roast level mapping
        self.domain_roast_defaults = {
            "cybersec": 1, "coding": 1, "creative": 2, 
            "feelprint": 0, "hardware": 1
        }
        
        logger.info("ProfileController initialized")
    
    def validate_and_refine_profile(self, profile: Dict[str, Any], task: str) -> Dict[str, Any]:
        """
        Validate and refine a profile through multiple validation steps.
        
        Args:
            profile (Dict[str, Any]): Draft profile to validate and refine
            task (str): Original task description
            
        Returns:
            Dict[str, Any]: Validated and refined profile with summary
        """
        logger.info(f"Validating profile with {len(profile.get('brains', []))} brains")
        
        try:
            # Step 1: Structure validation
            validation_result = self._validate_structure(profile)
            if not validation_result["valid"]:
                return self._handle_validation_failure(validation_result, task)
            
            # Step 2: Dry-run relevance test
            relevance_result = self._perform_dry_run_relevance_test(profile, task)
            
            # Step 3: Auto-repair if needed
            if relevance_result["removed_brains"]:
                profile = self._auto_repair_profile(profile, task, relevance_result)
                # Retry relevance test
                relevance_result = self._perform_dry_run_relevance_test(profile, task)
            
            # Step 4: Calculate final scores
            scoring_result = self._calculate_profile_scores(profile, task, relevance_result)
            
            # Step 5: Apply roast tone governor
            roast_level = self._determine_roast_level(profile, task)
            
            # Step 6: Check bad profile memory
            if self._is_bad_profile(profile):
                logger.warning("Profile matches previously rejected profile")
                return self._handle_bad_profile(profile, task)
            
            # Step 7: Human profile summary
            summary = self._generate_human_summary(profile, scoring_result, relevance_result)
            
            # Step 8: Coherence fuse
            coherence_result = self._perform_coherence_fuse(profile, task, summary)
            
            # Step 9: Compile final result
            final_profile = {
                **profile,
                "validation_passed": True,
                "scoring": scoring_result,
                "relevance_test": relevance_result,
                "roast_level": roast_level,
                "summary": summary,
                "coherence": coherence_result,
                "status": "validated",
                "validation_timestamp": self._get_timestamp()
            }
            
            # Log the validation
            self.profile_logs.add_log({
                "action": "profile_validation",
                "profile_id": profile.get("profile_id"),
                "brains_count": len(profile.get("brains", [])),
                "final_score": scoring_result.get("overall_score", 0.0),
                "status": "success"
            })
            
            return final_profile
            
        except Exception as e:
            logger.error(f"Profile validation failed: {e}")
            return self._handle_validation_error(profile, str(e), task)
    
    def _validate_structure(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Validate profile structure."""
        brains = profile.get("brains", [])
        issues = []
        
        # Check minimum/maximum brains
        if len(brains) < 2:
            issues.append(f"Too few brains: {len(brains)} (minimum 2)")
        elif len(brains) > 5:
            issues.append(f"Too many brains: {len(brains)} (maximum 5)")
        
        # Check for synthesis brain (llm_connector or summarizer)
        has_synthesis = any(brain in ["llm_connector", "summarizer"] for brain in brains)
        if not has_synthesis:
            issues.append("Missing synthesis brain (llm_connector or summarizer required)")
        
        # Check for incompatible pairs
        incompatible_pairs = self._find_incompatible_pairs(brains)
        if incompatible_pairs:
            issues.append(f"Incompatible brain pairs: {incompatible_pairs}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "removed_brains": [] if len(issues) == 0 else self._suggest_structure_fixes(brains, issues)
        }
    
    def _find_incompatible_pairs(self, brains: List[str]) -> List[Tuple[str, str]]:
        """Find incompatible brain pairs in a list."""
        pairs = []
        for i, brain1 in enumerate(brains):
            for brain2 in brains[i+1:]:
                if self.profile_engine.compatibility_matrix.are_incompatible(brain1, brain2):
                    pairs.append((brain1, brain2))
        return pairs
    
    def _suggest_structure_fixes(self, brains: List[str], issues: List[str]) -> List[str]:
        """Suggest brains to remove to fix structure issues."""
        to_remove = []
        
        # For incompatible pairs, suggest removing the lower-scoring brain
        incompatible_pairs = self._find_incompatible_pairs(brains)
        for brain1, brain2 in incompatible_pairs:
            # Simple heuristic: prefer to keep llm_connector, summarizer
            if brain1 in ["llm_connector", "summarizer"]:
                to_remove.append(brain2)
            elif brain2 in ["llm_connector", "summarizer"]:
                to_remove.append(brain1)
            else:
                to_remove.append(brain2)  # Arbitrary removal
        
        return list(set(to_remove))  # Remove duplicates
    
    def _perform_dry_run_relevance_test(self, profile: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Step 2: Test brain relevance through dry-run."""
        brains = profile.get("brains", [])
        relevance_scores = {}
        removed_brains = []
        
        for brain in brains:
            try:
                relevance_score = self._test_brain_relevance(brain, task)
                relevance_scores[brain] = relevance_score
                
                # Remove brains with low relevance
                if relevance_score < 0.4:
                    removed_brains.append(brain)
                    
            except Exception as e:
                logger.warning(f"Dry-run test failed for brain {brain}: {e}")
                relevance_scores[brain] = 0.0
                removed_brains.append(brain)
        
        # Create a mock brain response for scoring purposes
        mock_responses = {
            brain: f"Mock response for {brain} relevance test" 
            for brain in brains if brain not in removed_brains
        }
        
        return {
            "relevance_scores": relevance_scores,
            "removed_brains": removed_brains,
            "mock_responses": mock_responses,
            "test_completed": True
        }
    
    def _test_brain_relevance(self, brain: str, task: str) -> float:
        """Test if a brain is relevant to the task."""
        # Simple relevance scoring based on keyword matching
        brain_keywords = {
            "security_analyst": ["security", "vulnerable", "risk", "threat"],
            "data_cleaner": ["data", "clean", "process", "format"],
            "systems_architect": ["system", "architecture", "design", "structure"],
            "summarizer": ["summary", "overview", "concise", "brief"],
            "semantic_router": ["semantic", "router", "classify", "categorize"],
            "llm_connector": ["creative", "analysis", "comprehensive", "reasoning"],
            "publisher": ["publish", "content", "format", "output"],
            "autopublisher": ["auto", "publish", "format", "output"],
            "version_manager": ["version", "manage", "control", "track"],
            "rube_adapter": ["adapt", "convert", "transform", "bridge"]
        }
        
        task_lower = task.lower()
        brain_kw = brain_keywords.get(brain, [])
        
        matches = sum(1 for kw in brain_kw if kw in task_lower)
        return min(1.0, matches / max(1, len(brain_kw) / 2))
    
    def _auto_repair_profile(self, profile: Dict[str, Any], task: str, 
                           relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Auto-repair profile after relevance test."""
        logger.info("Auto-repairing profile based on relevance test")
        
        brains = profile.get("brains", [])
        removed_brains = relevance_result.get("removed_brains", [])
        available_brains = [b for b in brains if b not in removed_brains]
        
        # Remove irrelevant brains
        repaired_brains = available_brains
        
        # Add top compatible brains if we have too few
        if len(repaired_brains) < 2:
            # Get replacement brains from primary domain
            primary_domain = profile.get("primary_domain", "general")
            all_available = self._get_all_available_brains()
            replacement_brains = []
            
            for brain in all_available:
                if brain not in repaired_brains and brain not in removed_brains:
                    compat_score = self.profile_engine.compatibility_matrix.get_compatibility_score(
                        primary_domain, brain
                    )
                    if compat_score >= 0.5:
                        replacement_brains.append((brain, compat_score))
            
            # Sort by compatibility and add top ones
            replacement_brains.sort(key=lambda x: x[1], reverse=True)
            for brain, score in replacement_brains[:2]:
                repaired_brains.append(brain)
        
        # Ensure synthesis brain
        if not any(brain in ["llm_connector", "summarizer"] for brain in repaired_brains):
            if "llm_connector" not in removed_brains:
                repaired_brains.append("llm_connector")
            elif "summarizer" not in removed_brains:
                repaired_brains.append("summarizer")
        
        # Remove incompatible pairs
        repaired_brains = self.profile_engine.compatibility_matrix.filter_incompatible_brains(repaired_brains)
        
        return {
            **profile,
            "brains": repaired_brains,
            "auto_repaired": True,
            "repair_details": {
                "original_count": len(brains),
                "removed_count": len(removed_brains),
                "final_count": len(repaired_brains)
            }
        }
    
    def _get_all_available_brains(self) -> List[str]:
        """Get list of all available brain experts."""
        return [
            "summarizer", "systems_architect", "security_analyst", "data_cleaner",
            "semantic_router", "llm_connector", "publisher", "autopublisher", 
            "version_manager", "rube_adapter"
        ]
    
    def _calculate_profile_scores(self, profile: Dict[str, Any], task: str, 
                                relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Calculate comprehensive profile scores."""
        brains = profile.get("brains", [])
        primary_domain = profile.get("primary_domain", "general")
        
        scores = {}
        
        # 1. Domain relevance (40%)
        domain_scores = []
        for brain in brains:
            score = self.profile_engine.compatibility_matrix.get_compatibility_score(primary_domain, brain)
            domain_scores.append(score)
        domain_relevance = sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
        
        # 2. Compatibility (30%)
        compatibility_score = self._calculate_compatibility_score(brains)
        
        # 3. Dry-run relevance (20%)
        relevance_scores = relevance_result.get("relevance_scores", {})
        avg_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0.0
        
        # 4. Noise potential (10%) - lower is better
        noise_score = self._calculate_noise_potential(brains, primary_domain)
        
        # Weighted overall score
        overall_score = (
            domain_relevance * self.scoring_weights["domain_relevance"] +
            compatibility_score * self.scoring_weights["compatibility"] +
            avg_relevance * self.scoring_weights["dry_run_relevance"] +
            (1.0 - noise_score) * self.scoring_weights["noise_potential"]
        )
        
        return {
            "domain_relevance": domain_relevance,
            "compatibility": compatibility_score,
            "dry_run_relevance": avg_relevance,
            "noise_potential": noise_score,
            "overall_score": overall_score,
            "score_breakdown": {
                "domain_weight": self.scoring_weights["domain_relevance"],
                "compatibility_weight": self.scoring_weights["compatibility"], 
                "relevance_weight": self.scoring_weights["dry_run_relevance"],
                "noise_weight": self.scoring_weights["noise_potential"]
            }
        }
    
    def _calculate_compatibility_score(self, brains: List[str]) -> float:
        """Calculate compatibility score for a group of brains."""
        if len(brains) <= 1:
            return 1.0
        
        compatible_pairs = 0
        total_pairs = 0
        
        for i, brain1 in enumerate(brains):
            for brain2 in brains[i+1:]:
                total_pairs += 1
                if not self.profile_engine.compatibility_matrix.are_incompatible(brain1, brain2):
                    compatible_pairs += 1
        
        return compatible_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _calculate_noise_potential(self, brains: List[str], primary_domain: str) -> float:
        """Calculate potential for noise from brain selection."""
        if not brains:
            return 1.0
        
        noise_count = 0
        for brain in brains:
            compatibility = self.profile_engine.compatibility_matrix.get_compatibility_score(primary_domain, brain)
            if compatibility < 0.4:
                noise_count += 1
        
        return noise_count / len(brains)
    
    def _determine_roast_level(self, profile: Dict[str, Any], task: str) -> int:
        """Step 5: Determine appropriate roast level."""
        primary_domain = profile.get("primary_domain", "general")
        
        # Default roast levels by domain
        default_level = self.domain_roast_defaults.get(primary_domain, 1)
        
        # Adjust based on task characteristics
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["urgent", "critical", "emergency"]):
            default_level = min(2, default_level + 1)
        elif any(word in task_lower for word in ["sensitive", "delicate", "careful"]):
            default_level = max(0, default_level - 1)
        
        return max(0, min(2, default_level))
    
    def _is_bad_profile(self, profile: Dict[str, Any]) -> bool:
        """Step 6: Check if profile matches previously rejected profiles."""
        profile_signature = self._create_profile_signature(profile)
        return self.profile_memory.is_rejected_profile(profile_signature)
    
    def _create_profile_signature(self, profile: Dict[str, Any]) -> str:
        """Create a signature for profile comparison."""
        brains = profile.get("brains", [])
        primary_domain = profile.get("primary_domain", "general")
        return f"{primary_domain}:{'-'.join(sorted(brains))}"
    
    def _handle_bad_profile(self, profile: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Handle profile that matches previously rejected profile."""
        logger.warning("Profile matches rejected profile, generating alternative")
        
        # Generate alternative profile
        alternative_profile = self.profile_engine.generate_profile(task, self._get_all_available_brains())
        
        # Mark as alternative
        alternative_profile["alternative_generated"] = True
        alternative_profile["alternative_reason"] = "original_profile_rejected"
        
        return alternative_profile
    
    def _generate_human_summary(self, profile: Dict[str, Any], scoring_result: Dict[str, Any], 
                              relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Generate human-readable profile summary."""
        brains = profile.get("brains", [])
        primary_domain = profile.get("primary_domain", "general")
        
        # Why these brains
        why_these = []
        for brain in brains:
            domain = self.profile_engine.compatibility_matrix.get_brain_domain(brain)
            if domain == primary_domain:
                why_these.append(f"{brain} (primary domain)")
            else:
                why_these.append(f"{brain} (domain: {domain})")
        
        # Dropped brains
        dropped_brains = relevance_result.get("removed_brains", [])
        
        return {
            "why_these_brains": f"Selected {', '.join(why_these)} for {primary_domain} domain task",
            "dropped_brains": f"Removed {', '.join(dropped_brains)} due to low relevance" if dropped_brains else "No brains removed",
            "final_domain": primary_domain,
            "confidence_level": "high" if scoring_result.get("overall_score", 0) > 0.7 else "medium" if scoring_result.get("overall_score", 0) > 0.4 else "low"
        }
    
    def _perform_coherence_fuse(self, profile: Dict[str, Any], task: str, 
                              summary: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Perform coherence fuse to check if profile solves the task."""
        try:
            # Mock coherence check (in real implementation, this would call llm_connector)
            profile_str = f"Brains: {', '.join(profile.get('brains', []))}, Domain: {profile.get('primary_domain')}"
            
            # Simple coherence heuristics
            has_synthesis = any(brain in ["llm_connector", "summarizer"] for brain in profile.get("brains", []))
            has_domain_expert = any(self.profile_engine.compatibility_matrix.get_compatibility_score(
                profile.get("primary_domain", "general"), brain) > 0.7 for brain in profile.get("brains", [])
            )
            
            coherent = has_synthesis and has_domain_expert and len(profile.get("brains", [])) >= 2
            
            if coherent:
                return {
                    "coherent": True,
                    "explanation": "Profile has appropriate synthesis capability and domain expertise",
                    "confidence": "high"
                }
            else:
                return {
                    "coherent": False,
                    "explanation": "Profile may lack key capabilities for this task",
                    "confidence": "low",
                    "coherence_warning": "Consider adding more domain-specific brains or synthesis capability"
                }
                
        except Exception as e:
            return {
                "coherent": True,  # Default to true on error
                "explanation": "Coherence check failed, proceeding with profile",
                "confidence": "uncertain",
                "error": str(e)
            }
    
    def _handle_validation_failure(self, validation_result: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Handle validation failure by attempting auto-repair."""
        logger.warning("Profile validation failed, attempting auto-repair")
        
        # Try to create a basic valid profile
        basic_profile = self.profile_engine.generate_profile(task, self._get_all_available_brains())
        
        return {
            **basic_profile,
            "validation_passed": False,
            "validation_issues": validation_result["issues"],
            "status": "validation_failed_auto_repaired",
            "error": "Profile structure validation failed, auto-repaired"
        }
    
    def _handle_validation_error(self, profile: Dict[str, Any], error: str, task: str) -> Dict[str, Any]:
        """Handle validation error."""
        return {
            **profile,
            "validation_passed": False,
            "status": "validation_error",
            "error": error,
            "fallback_profile": True
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class RoastToneGovernor:
    """Manages roast tone levels for different domains and tasks."""
    
    def __init__(self):
        self.levels = {
            0: "none",    # No roasting, constructive only
            1: "mild",    # Gentle constructive criticism
            2: "spicy"    # Sharp, direct feedback
        }
    
    def get_roast_level_info(self, level: int) -> Dict[str, Any]:
        """Get information about a roast level."""
        return {
            "level": level,
            "name": self.levels.get(level, "unknown"),
            "description": self._get_level_description(level)
        }
    
    def _get_level_description(self, level: int) -> str:
        """Get description for roast level."""
        descriptions = {
            0: "No roasting - constructive feedback only",
            1: "Mild roasting - gentle constructive criticism", 
            2: "Spicy roasting - sharp, direct feedback"
        }
        return descriptions.get(level, "Unknown roast level")