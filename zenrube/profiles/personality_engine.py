# personality_engine.py
"""
Personality Engine Module (Phase 2)
Implements personality selection and assignment logic for Zenrube expert brains.
Uses ONLY the experts and modes defined in personality_presets.py
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .personality_presets import get_personality, get_neutral_personality


@dataclass
class SelectionCriteria:
    task_type: str
    primary_domain: Optional[str] = None
    secondary_domain: Optional[str] = None
    roast_level: Optional[int] = None
    team_mood: Optional[str] = None
    risk_tolerance: Optional[int] = None
    previous_modes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "primary_domain": self.primary_domain,
            "secondary_domain": self.secondary_domain,
            "roast_level": self.roast_level,
            "team_mood": self.team_mood,
            "risk_tolerance": self.risk_tolerance,
            "previous_modes": self.previous_modes
        }


def select_personality_mode(brain_name: str, domain: str, roast_level: int, task_type: str) -> Dict[str, Any]:
    # Get available modes for this brain
    neutral_cfg = get_neutral_personality(brain_name)
    if not neutral_cfg:
        return {}
    
    # Get the specific alternate mode for each brain type
    alternate_modes = {
        "security_analyst": "turing_strategist",
        "pragmatic_engineer": "builder_minimalist", 
        "pattern_brain": "da_vinci_ideator",
        "data_cleaner": "info_kondo",
        "semantic_router": "sherlock_navigator",
        "feelprint_brain": "jung_listener",
        "llm_connector": "neutral_researcher"
    }
    
    alternate_mode_name = alternate_modes.get(brain_name)
    alternate_cfg = get_personality(brain_name, alternate_mode_name) if alternate_mode_name else {}
    
    if not alternate_cfg:
        return neutral_cfg
    
    # Roast logic: If roast_level == 0, force neutral_mode only
    if roast_level == 0:
        return neutral_cfg
    
    # Roast logic: If roast_level >= personality.allowed_roast, choose alternate mode
    alternate_roast = alternate_cfg.get("allowed_roast", 0)
    if roast_level >= alternate_roast:
        return alternate_cfg
    
    # Domain logic
    if domain == "emotional":
        # Always choose neutral_mode for emotional tasks (safety)
        return neutral_cfg
    elif domain == "technical":
        # Prefer mode with lower allowed_roast and higher detail_level
        neutral_detail = neutral_cfg.get("detail_level", 1)
        alternate_detail = alternate_cfg.get("detail_level", 1)
        if alternate_detail > neutral_detail and alternate_roast < roast_level:
            return alternate_cfg
        else:
            return neutral_cfg
    elif domain == "creative":
        # Prefer mode with slightly higher risk_tolerance
        neutral_risk = neutral_cfg.get("risk_tolerance", 0)
        alternate_risk = alternate_cfg.get("risk_tolerance", 0)
        if alternate_risk > neutral_risk and alternate_roast <= roast_level:
            return alternate_cfg
        else:
            return neutral_cfg
    
    # Default fallback
    return neutral_cfg


def assign_personalities(profile: Dict[str, Any], domain: str, roast_level: int) -> Dict[str, Dict[str, Any]]:
    experts = profile.get("experts", [])
    task_type = profile.get("task_type", "neutral")
    
    personalities = {}
    for brain_name in experts:
        personality_cfg = select_personality_mode(brain_name, domain, roast_level, task_type)
        personalities[brain_name] = personality_cfg
    
    return personalities


def build_personality_prefix(brain_name: str, personality_cfg: Dict[str, Any], task: str) -> str:
    tone = personality_cfg.get("tone", "neutral")
    communication_style = personality_cfg.get("communication_style", "neutral")
    thinking_style = personality_cfg.get("thinking_style", "logical")
    detail_level = personality_cfg.get("detail_level", 1)
    critique_intensity = personality_cfg.get("critique_intensity", 1)

    return f"[tone={tone} | style={communication_style} | thinking={thinking_style} | detail={detail_level} | critique={critique_intensity}]"


class PersonalityEngine:
    """Engine for managing personality assignments and analytics"""

    def __init__(self):
        self.selection_history = []

    def assign_personalities(self, expert_instances: Dict[str, Any], criteria: SelectionCriteria) -> Dict[str, Dict[str, Any]]:
        """Assign personalities to experts based on criteria"""
        # Convert expert_instances to the format expected by assign_personalities
        profile = {
            "experts": list(expert_instances.keys()),
            "task_type": criteria.task_type
        }

        domain = criteria.primary_domain or "general"
        roast_level = criteria.roast_level or 0

        personalities = assign_personalities(profile, domain, roast_level)

        # Record the assignment
        self.selection_history.append({
            "criteria": criteria.to_dict(),
            "assignments": personalities
        })

        return personalities

    def build_personality_prefix(self, mode: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Build personality prefix for LLM prompts"""
        task = context.get("task_type", "general") if context else "general"
        return build_personality_prefix("", mode, task)

    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get analytics about personality selections"""
        if not self.selection_history:
            return {"total_selections": 0, "most_common_domain": None}

        domains = [entry["criteria"]["primary_domain"] for entry in self.selection_history if entry["criteria"]["primary_domain"]]
        most_common_domain = max(set(domains), key=domains.count) if domains else None

        return {
            "total_selections": len(self.selection_history),
            "most_common_domain": most_common_domain,
            "recent_selections": self.selection_history[-5:]  # Last 5
        }


# Global personality engine instance
personality_engine = PersonalityEngine()