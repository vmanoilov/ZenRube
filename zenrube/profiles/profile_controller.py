"""
Updated Profile Controller for Zenrube Dynamic Personality System

Now exposes primary_domain, secondary_domain, and roast_level for personality engine.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

from .dynamic_profile_engine import DynamicProfileEngine
from .classification_engine import ClassificationEngine
from .compatibility_matrix import CompatibilityMatrix
from .profile_memory import ProfileMemory
from .profile_logs import ProfileLogs
from .personality_presets import RoastLevel
from .personality_engine import PersonalityEngine, SelectionCriteria


class ProfileController:
    """Central controller for profile management with personality integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.profile_engine = DynamicProfileEngine()
        self.classification_engine = ClassificationEngine()
        self.compatibility_matrix = CompatibilityMatrix()
        self.profile_memory = ProfileMemory()
        self.profile_logger = ProfileLogs()
        
        # Personality system integration
        from .personality_engine import personality_engine
        self.personality_engine = personality_engine
        
        # Safety integration
        from .personality_safety import safety_governor
        self.safety_governor = safety_governor
        
        # Cache for domain classification
        self.domain_cache = {}
        
        # Configuration
        self.cache_ttl_minutes = self.config.get("cache_ttl_minutes", 30)
    
    def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        roast_level: Optional[RoastLevel] = None,
        team_composition: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process request with personality-aware profile management"""
        
        # Classify domains for personality engine
        primary_domain, secondary_domain = self._classify_domains(request, context)
        
        # Prepare criteria for personality selection
        criteria = SelectionCriteria(
            task_type=self._determine_task_type(request, context),
            primary_domain=primary_domain,
            secondary_domain=secondary_domain,
            roast_level=roast_level,
            team_mood=context.get("team_mood") if context else None,
            risk_tolerance=context.get("risk_tolerance") if context else None,
            previous_modes=context.get("previous_modes") if context else None
        )
        
        # Get base profile
        base_profile = self.profile_engine.get_profile(
            request, user_profile, context
        )
        
        # Enhance with personality-aware expert selection
        personality_enhanced_profile = self._enhance_with_personality(
            base_profile, criteria, team_composition
        )
        
        # Log the personality-enhanced decision
        self.profile_logger.log_decision(
            request, personality_enhanced_profile, {
                "primary_domain": primary_domain,
                "secondary_domain": secondary_domain,
                "roast_level": roast_level.value if roast_level else 0,
                "personality_assignments": {
                    expert: mode.mode_id 
                    for expert, mode in personality_enhanced_profile.get("personality_assignments", {}).items()
                }
            }
        )
        
        return personality_enhanced_profile
    
    def _classify_domains(self, request: str, context: Optional[Dict[str, Any]]) -> tuple:
        """Classify request into primary and secondary domains"""
        
        # Check cache first
        cache_key = f"{request}:{str(context)}"
        if cache_key in self.domain_cache:
            cached_time, domains = self.domain_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return domains
        
        # Use classification engine
        classification_result = self.classification_engine.classify_task(request)
        
        primary_domain = classification_result.get("primary_domain", "general")
        secondary_domain = classification_result.get("secondary_domain", None)
        
        # Cache the result
        self.domain_cache[cache_key] = (datetime.now(), (primary_domain, secondary_domain))
        
        return primary_domain, secondary_domain
    
    def _determine_task_type(self, request: str, context: Optional[Dict[str, Any]]) -> str:
        """Determine the type of task for personality selection"""
        
        # Simple task type detection based on keywords
        request_lower = request.lower()
        
        task_indicators = {
            "analysis": ["analyze", "examine", "review", "assess", "investigate"],
            "creative": ["create", "design", "brainstorm", "generate", "invent"],
            "routine": ["process", "routine", "regular", "standard", "normal"],
            "debug": ["debug", "fix", "error", "problem", "issue", "troubleshoot"],
            "brainstorm": ["brainstorm", "ideas", "think", "suggest", "propose"],
            "review": ["review", "check", "verify", "validate", "test"]
        }
        
        for task_type, indicators in task_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                return task_type
        
        return "general"
    
    def _enhance_with_personality(
        self,
        base_profile: Dict[str, Any],
        criteria: SelectionCriteria,
        team_composition: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhance profile with personality-aware expert assignments"""
        
        expert_assignments = base_profile.get("expert_assignments", {})
        
        # Build expert assignments dict for personality engine
        expert_instances = {}
        for expert_name, assignment in expert_assignments.items():
            expert_instances[assignment["type"]] = assignment
        
        # Get personality assignments
        personality_assignments = self.personality_engine.assign_personalities(
            expert_instances, criteria
        )
        
        # Apply safety governor to all personality modes
        safe_assignments = {}
        safety_events = []
        
        for expert_type, mode in personality_assignments.items():
            safe_mode, was_modified, events = self.safety_governor.apply_safety_governor(
                mode, criteria, expert_type
            )
            safe_assignments[expert_type] = safe_mode
            safety_events.extend(events)
        
        # Build personality prefixes for LLM prompts
        personality_prefixes = {}
        for expert_type, mode in safe_assignments.items():
            prefix = self.personality_engine.build_personality_prefix(mode, context=criteria.to_dict())
            personality_prefixes[expert_type] = prefix
        
        # Build the final profile structure with required fields
        final_profile = {
            "experts": list(expert_assignments.keys()),
            "primary_domain": criteria.primary_domain,
            "secondary_domain": criteria.secondary_domain,
            "roast_level": criteria.roast_level.value if criteria.roast_level else 0,
            "task_type": criteria.task_type,
            "context": base_profile.get("context", {}),
            "personality_assignments": safe_assignments,
            "personality_prefixes": personality_prefixes,
            "safety_events": [event.to_dict() for event in safety_events],
            "team_diversity": self._calculate_team_diversity(safe_assignments)
        }
        
        return final_profile
    
    def _calculate_team_diversity(self, personality_assignments: Dict[str, Any]) -> float:
        """Calculate diversity score for the personality team"""
        if not personality_assignments:
            return 0.0
        
        # Analyze roast level distribution
        roast_levels = [mode.roast_level for mode in personality_assignments.values()]
        unique_roasts = len(set(roast_levels))
        total_experts = len(personality_assignments)
        
        diversity_score = unique_roasts / total_experts
        return min(1.0, diversity_score)
    
    def get_personality_status(self) -> Dict[str, Any]:
        """Get current personality system status"""
        return {
            "engine_analytics": self.personality_engine.get_selection_analytics(),
            "safety_status": self.safety_governor.get_safety_status(),
            "domain_cache_size": len(self.domain_cache),
            "recent_decisions": self.profile_logger.get_recent_decisions(limit=5)
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.domain_cache.clear()
        self.profile_engine.clear_cache()
        self.compatibility_matrix.clear_cache()
    
    def reset_personality_state(self, expert_type: Optional[str] = None):
        """Reset personality system state"""
        if expert_type:
            self.safety_governor.clear_expert_status(expert_type)
        else:
            # Reset all state would require accessing private attributes
            pass

    def _validate_structure(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate profile structure and constraints"""
        issues = []

        # Check required fields
        if "brains" not in profile:
            issues.append("Missing 'brains' field")
        if "primary_domain" not in profile:
            issues.append("Missing 'primary_domain' field")

        # Check brain count constraints
        if "brains" in profile:
            brain_count = len(profile["brains"])
            if brain_count < 2:
                issues.append("Too few brains (minimum 2)")
            elif brain_count > 5:
                issues.append("Too many brains (maximum 5)")

        # Check for duplicate brains
        if "brains" in profile and len(profile["brains"]) != len(set(profile["brains"])):
            issues.append("Duplicate brains in profile")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _perform_dry_run_relevance_test(self, profile: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Perform dry-run relevance testing"""
        # Simple mock implementation - in real system would test actual relevance
        brains = profile.get("brains", [])
        relevance_scores = {}

        # Mock scoring based on brain names and task keywords
        for brain in brains:
            score = 0.5  # Base score
            if "security" in task.lower() and "security" in brain:
                score += 0.3
            if "data" in task.lower() and "data" in brain:
                score += 0.2
            relevance_scores[brain] = min(1.0, score)

        # Remove low-relevance brains
        removed_brains = [brain for brain, score in relevance_scores.items() if score < 0.4]

        return {
            "relevance_scores": relevance_scores,
            "removed_brains": removed_brains,
            "test_completed": True
        }

    def _auto_repair_profile(self, profile: Dict[str, Any], task: str, relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-repair profile by adding missing brains or adjusting composition"""
        repaired_profile = profile.copy()
        brains = repaired_profile.get("brains", [])
        domain = repaired_profile.get("primary_domain", "general")

        # Remove low-relevance brains first
        removed_brains = relevance_result.get("removed_brains", [])
        brains = [brain for brain in brains if brain not in removed_brains]

        # Add default brains based on domain to ensure good coverage
        default_brains = {
            "cybersec": ["security_analyst", "systems_architect"],
            "creative": ["pattern_brain", "llm_connector"],
            "general": ["summarizer", "data_cleaner"]
        }
        additional_brains = default_brains.get(domain, ["summarizer", "data_cleaner"])
        for brain in additional_brains:
            if brain not in brains and len(brains) < 5:  # Max 5
                brains.append(brain)

        # Ensure minimum brains
        if len(brains) < 2:
            if "llm_connector" not in brains:
                brains.append("llm_connector")

        repaired_profile["brains"] = brains
        repaired_profile["auto_repaired"] = True

        return repaired_profile

    def _calculate_profile_scores(self, profile: Dict[str, Any], task: str, relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive profile scores"""
        brains = profile.get("brains", [])
        domain = profile.get("primary_domain", "general")
        relevance_scores = relevance_result.get("relevance_scores", {})

        # Domain relevance score
        domain_relevance = 0.8  # Mock - would check domain alignment

        # Compatibility score
        compatibility = 0.7  # Mock - would check brain compatibility

        # Dry-run relevance score
        dry_run_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0.5

        # Noise potential (inverse of relevance)
        noise_potential = 1.0 - dry_run_relevance

        # Overall score
        overall_score = (domain_relevance * 0.3 + compatibility * 0.3 + dry_run_relevance * 0.4)

        return {
            "domain_relevance": domain_relevance,
            "compatibility": compatibility,
            "dry_run_relevance": dry_run_relevance,
            "noise_potential": noise_potential,
            "overall_score": overall_score
        }

    def _generate_human_summary(self, profile: Dict[str, Any], scoring_result: Dict[str, Any], relevance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable profile summary"""
        brains = profile.get("brains", [])
        domain = profile.get("primary_domain", "general")
        overall_score = scoring_result.get("overall_score", 0.5)
        removed_brains = relevance_result.get("removed_brains", [])

        # Determine confidence level
        if overall_score >= 0.8:
            confidence = "high"
        elif overall_score >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate explanations
        why_these_brains = f"Selected {len(brains)} brains optimized for {domain} domain tasks"
        dropped_brains = f"Removed {len(removed_brains)} low-relevance brains" if removed_brains else "No brains removed"

        return {
            "why_these_brains": why_these_brains,
            "dropped_brains": dropped_brains,
            "final_domain": domain,
            "confidence_level": confidence
        }

    def _perform_coherence_fuse(self, profile: Dict[str, Any], task: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Perform coherence checking"""
        brains = profile.get("brains", [])
        domain = profile.get("primary_domain", "general")

        # Simple coherence check
        coherent = len(brains) >= 2 and len(brains) <= 5

        explanation = "Profile meets basic coherence requirements" if coherent else "Profile fails coherence check"
        confidence = "high" if coherent else "low"

        return {
            "coherent": coherent,
            "explanation": explanation,
            "confidence": confidence
        }

    def _determine_roast_level(self, profile: Dict[str, Any], task: str) -> int:
        """Determine appropriate roast level for profile"""
        domain = profile.get("primary_domain", "general")

        # Domain-based defaults
        domain_defaults = {
            "cybersec": 1,
            "creative": 2,
            "coding": 1,
            "hardware": 1,
            "feelprint": 0,
            "general": 1
        }

        return domain_defaults.get(domain, 1)

    def validate_and_refine_profile(self, profile: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Complete profile validation and refinement pipeline"""
        # Structure validation
        structure_valid = self._validate_structure(profile)

        if not structure_valid["valid"]:
            return {
                "status": "failed",
                "validation_passed": False,
                "issues": structure_valid["issues"]
            }

        # Dry-run relevance test
        relevance_result = self._perform_dry_run_relevance_test(profile, task)

        # Auto-repair if needed
        if len(profile.get("brains", [])) < 2 or relevance_result.get("removed_brains"):
            profile = self._auto_repair_profile(profile, task, relevance_result)

        # Calculate scores
        scoring_result = self._calculate_profile_scores(profile, task, relevance_result)

        # Generate summary
        summary = self._generate_human_summary(profile, scoring_result, relevance_result)

        # Determine roast level
        roast_level = self._determine_roast_level(profile, task)

        # Coherence check
        coherence = self._perform_coherence_fuse(profile, task, summary)

        return {
            "brains": profile.get("brains", []),
            "primary_domain": profile.get("primary_domain", "general"),
            "validation_passed": True,
            "scoring": scoring_result,
            "relevance_test": relevance_result,
            "roast_level": roast_level,
            "summary": summary,
            "coherence": coherence,
            "auto_repaired": profile.get("auto_repaired", False)
        }


class RoastToneGovernor:
    """Governor for roast tone levels based on domain and context"""

    def __init__(self):
        self.domain_defaults = {
            "cybersec": 1,
            "creative": 2,
            "coding": 1,
            "hardware": 1,
            "feelprint": 0,
            "general": 1
        }

    def get_roast_level(self, domain: str, context: Optional[Dict[str, Any]] = None) -> int:
        """Get appropriate roast level for domain"""
        base_level = self.domain_defaults.get(domain, 1)

        # Adjust based on context
        if context:
            if context.get("sensitive", False):
                return 0
            if context.get("creative_task", False):
                return min(base_level + 1, 3)
            if context.get("critical_security", False):
                return min(base_level + 1, 3)

        return base_level


# Global profile controller instance
profile_controller = ProfileController()