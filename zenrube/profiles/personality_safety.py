# personality_safety.py
"""
Personality Safety Governor Module (Phase 3A)
Implements safety constraints and overrides for Zenrube personality system.
Ensures safe and appropriate personality mode selection.
"""

from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from .personality_presets import get_neutral_personality


@dataclass
class SafetyEvent:
    event_type: str
    description: str
    brain_name: str = ""
    original_mode: str = ""
    new_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "description": self.description,
            "brain_name": self.brain_name,
            "original_mode": self.original_mode,
            "new_mode": self.new_mode
        }


def apply_safety_governor(domain: str, roast_level: int, task_type: str, personalities: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    overrides_applied = 0
    neutral_fallback_used = False
    reasons = []
    
    adjusted_personalities = personalities.copy()
    
    # Rule 1: Roast Clamping
    for brain_name, personality_cfg in personalities.items():
        if roast_level == 0:
            adjusted_personalities[brain_name] = {
                "tone": "neutral",
                "communication_style": "clear, structured",
                "thinking_style": "logical",
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 2,
                "allowed_roast": 1
            }
            overrides_applied += 1
            reasons.append(f"Forced neutral_mode for {brain_name} (roast_level=0)")
        elif roast_level < personality_cfg.get("allowed_roast", 0):
            adjusted_personalities[brain_name] = {
                "tone": "neutral",
                "communication_style": "clear, structured",
                "thinking_style": "logical",
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 2,
                "allowed_roast": 1
            }
            overrides_applied += 1
            reasons.append(f"Clamped {brain_name} to neutral_mode (insufficient roast_level)")
    
    # Rule 2: Chaos Cooling (simple version)
    if domain == "emotional":
        for brain_name in adjusted_personalities:
            adjusted_personalities[brain_name] = {
                "tone": "neutral",
                "communication_style": "clear, structured",
                "thinking_style": "logical",
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 2,
                "allowed_roast": 1
            }
            overrides_applied += 1
            reasons.append("Forced neutral_mode for all experts (emotional domain)")

    if task_type == "sensitive":
        for brain_name in adjusted_personalities:
            adjusted_personalities[brain_name] = {
                "tone": "neutral",
                "communication_style": "clear, structured",
                "thinking_style": "logical",
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 2,
                "allowed_roast": 1
            }
            overrides_applied += 1
            reasons.append("Forced neutral_mode for all experts (sensitive task_type)")
    
    # Rule 3: Neutral Fallback
    if overrides_applied >= 2:
        for brain_name in adjusted_personalities:
            adjusted_personalities[brain_name] = get_neutral_personality(brain_name)
        neutral_fallback_used = True
        reasons.append("Neutral fallback triggered (2+ overrides)")
    
    # Rule 4: Safety Summary
    safety_summary = {
        "overrides_applied": overrides_applied,
        "neutral_fallback_used": neutral_fallback_used,
        "reason": "; ".join(reasons) if reasons else "No safety overrides applied"
    }
    
    return adjusted_personalities, safety_summary


class SafetyGovernor:
    """Safety governor for personality mode selection"""

    def __init__(self):
        self.expert_status = {}

    def apply_safety_governor(self, mode: Dict[str, Any], criteria: Any, expert_type: str) -> Tuple[Dict[str, Any], bool, List[SafetyEvent]]:
        """Apply safety governor to a personality mode"""
        events = []
        was_modified = False

        # Simple safety check: if roast_level is 0, force neutral
        roast_level = getattr(criteria, 'roast_level', None)
        if roast_level == 0:
            # Get neutral mode - for simplicity, return a neutral config
            safe_mode = {
                "tone": "neutral",
                "communication_style": "clear, structured",
                "thinking_style": "logical",
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 2,
                "allowed_roast": 1
            }
            was_modified = True
            events.append(SafetyEvent(
                event_type="roast_level_zero",
                description="Forced neutral mode due to roast_level=0",
                brain_name=expert_type,
                original_mode=mode.get("tone", "unknown"),
                new_mode="neutral"
            ))
        else:
            safe_mode = mode

        return safe_mode, was_modified, events

    def clear_expert_status(self, expert_type: Optional[str] = None):
        """Clear expert status"""
        if expert_type:
            self.expert_status.pop(expert_type, None)
        else:
            self.expert_status.clear()

    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety status"""
        return {
            "expert_status_count": len(self.expert_status),
            "safety_events_logged": 0  # Could track this
        }


# Global safety governor instance
safety_governor = SafetyGovernor()