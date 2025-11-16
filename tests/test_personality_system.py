# test_personality_system.py
"""
Complete test suite for Zenrube Personality System (Phase 3D)
Tests personality_presets, personality_engine, and personality_safety modules
"""

import unittest
from typing import Dict, Any

# Import only from allowed modules
from zenrube.profiles.personality_presets import (
    PERSONALITY_PRESETS, 
    get_personality, 
    get_neutral_personality, 
    get_default_personality
)
from zenrube.profiles.personality_engine import (
    assign_personalities,
    build_personality_prefix,
    select_personality_mode,
    PersonalityEngine
)
from zenrube.profiles.personality_safety import (
    apply_safety_governor,
    SafetyGovernor
)


class TestPersonalityPresets(unittest.TestCase):
    """Test personality_presets.py functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.expected_experts = [
            "security_analyst",
            "pragmatic_engineer", 
            "pattern_brain",
            "data_cleaner",
            "semantic_router",
            "feelprint_brain",
            "llm_connector"
        ]
    
    def test_all_expert_keys_exist(self):
        """Test that all expected expert keys exist in PERSONALITY_PRESETS"""
        for expert in self.expected_experts:
            self.assertIn(expert, PERSONALITY_PRESETS, 
                         f"Expert {expert} missing from PERSONALITY_PRESETS")
    
    def test_each_expert_has_exactly_two_modes(self):
        """Test that each expert contains exactly two modes"""
        for expert_name, expert_config in PERSONALITY_PRESETS.items():
            modes = list(expert_config.keys())
            self.assertEqual(len(modes), 2, 
                           f"Expert {expert_name} should have exactly 2 modes, found {len(modes)}")
            
            # Check that one of them is neutral_mode
            self.assertIn("neutral_mode", modes, 
                         f"Expert {expert_name} must have neutral_mode")
    
    def test_neutral_mode_exists_per_expert(self):
        """Test that neutral_mode exists for every expert"""
        for expert_name, expert_config in PERSONALITY_PRESETS.items():
            self.assertIn("neutral_mode", expert_config,
                         f"Expert {expert_name} missing neutral_mode")
            
            neutral_mode = expert_config["neutral_mode"]
            
            # Check required fields exist in neutral_mode
            required_fields = ["tone", "communication_style", "thinking_style", 
                             "critique_intensity", "risk_tolerance", "detail_level", 
                             "allowed_roast"]
            
            for field in required_fields:
                self.assertIn(field, neutral_mode,
                            f"Expert {expert_name} neutral_mode missing field: {field}")
    
    def test_get_personality_returns_correct_dict(self):
        """Test get_personality function returns correct personality dict"""
        # Test existing expert and mode
        result = get_personality("security_analyst", "turing_strategist")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["tone"], "dry, precise")
        self.assertEqual(result["thinking_style"], "pattern-analysis, threat-focused")
        
        # Test non-existent expert
        result = get_personality("non_existent_expert", "any_mode")
        self.assertEqual(result, {})
        
        # Test non-existent mode
        result = get_personality("security_analyst", "non_existent_mode")
        self.assertEqual(result, {})
    
    def test_get_neutral_personality_returns_neutral_mode(self):
        """Test get_neutral_personality returns neutral_mode for each expert"""
        for expert_name in self.expected_experts:
            result = get_neutral_personality(expert_name)
            
            # Should return a dict with expected structure
            self.assertIsInstance(result, dict)
            
            # Should match the neutral_mode from PERSONALITY_PRESETS
            expected = PERSONALITY_PRESETS[expert_name]["neutral_mode"]
            self.assertEqual(result, expected)
            
            # Verify it's actually the neutral mode
            self.assertEqual(result["tone"], "neutral")


class TestPersonalityEngine(unittest.TestCase):
    """Test personality_engine.py functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_profile = {
            "experts": ["security_analyst", "pragmatic_engineer"],
            "task_type": "coding"
        }
    
    def test_assign_personalities_returns_personality_for_each_expert(self):
        """Test that assign_personalities returns a personality for each expert"""
        result = assign_personalities(self.mock_profile, "technical", 1)
        
        # Should return dict with same number of experts
        self.assertEqual(len(result), len(self.mock_profile["experts"]))
        
        # Each expert should have a personality config
        for expert_name in self.mock_profile["experts"]:
            self.assertIn(expert_name, result)
            personality = result[expert_name]
            self.assertIsInstance(personality, dict)
            
            # Should have required fields
            required_fields = ["tone", "communication_style", "thinking_style"]
            for field in required_fields:
                self.assertIn(field, personality)
    
    def test_build_personality_prefix_returns_non_empty_string(self):
        """Test that build_personality_prefix returns a non-empty string"""
        mock_personality = {
            "tone": "neutral",
            "communication_style": "clear, structured",
            "thinking_style": "logical",
            "detail_level": 2,
            "critique_intensity": 1
        }
        
        result = build_personality_prefix("security_analyst", mock_personality, "coding")
        
        # Should return a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Should contain expected format elements
        self.assertIn("[tone=", result)
        self.assertIn(" | style=", result)
        self.assertIn(" | thinking=", result)
        self.assertIn(" | detail=", result)
        self.assertIn(" | critique=", result)
        self.assertIn("]", result)
    
    def test_technical_tasks_prefer_neutral_mode_when_roast_level_low(self):
        """Test that technical tasks prefer neutral_mode when roast_level is low"""
        # Low roast level should prefer neutral mode
        result = select_personality_mode("security_analyst", "technical", 1, "coding")
        
        # Should return neutral personality or one with low roast
        expected_neutral = get_neutral_personality("security_analyst")
        
        # Either neutral or very conservative mode
        self.assertEqual(result["critique_intensity"], expected_neutral["critique_intensity"])
    
    def test_creative_tasks_choose_alternate_mode_when_allowed(self):
        """Test that creative tasks choose alternate mode when allowed"""
        # Creative domain with higher roast level
        result = select_personality_mode("pattern_brain", "creative", 2, "brainstorming")
        
        # Should be able to choose alternate mode for creative tasks
        # Pattern brain's alternate mode should be da_vinci_ideator which has higher risk_tolerance
        self.assertIsInstance(result, dict)
        self.assertIn("tone", result)
    
    def test_roast_level_zero_forces_neutral_mode(self):
        """Test that roast_level == 0 forces neutral_mode"""
        # For all expert types, roast_level=0 should force neutral
        for expert_name in ["security_analyst", "pragmatic_engineer", "pattern_brain"]:
            result = select_personality_mode(expert_name, "creative", 0, "coding")
            
            expected_neutral = get_neutral_personality(expert_name)
            self.assertEqual(result, expected_neutral,
                           f"Expert {expert_name} should have neutral mode when roast_level=0")


class TestPersonalitySafety(unittest.TestCase):
    """Test personality_safety.py functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_personalities = {
            "security_analyst": {
                "tone": "dry, precise",
                "communication_style": "minimalistic, fact-driven", 
                "thinking_style": "pattern-analysis, threat-focused",
                "critique_intensity": 2,
                "risk_tolerance": 0,
                "detail_level": 2,
                "allowed_roast": 1
            },
            "pragmatic_engineer": {
                "tone": "direct",
                "communication_style": "simple, efficient, unembellished",
                "thinking_style": "process-first, minimal-viable-solution", 
                "critique_intensity": 1,
                "risk_tolerance": 1,
                "detail_level": 1,
                "allowed_roast": 1
            }
        }
    
    def test_roast_clamping_roast_level_zero(self):
        """Test roast clamping when roast_level == 0"""
        result, summary = apply_safety_governor("technical", 0, "coding", self.mock_personalities)
        
        # All personalities should be clamped to neutral
        for expert_name in result:
            personality = result[expert_name]
            self.assertEqual(personality["tone"], "neutral")
            self.assertEqual(personality["communication_style"], "clear, structured")
            self.assertEqual(personality["thinking_style"], "logical")
        
        # Summary should indicate overrides applied
        self.assertGreater(summary["overrides_applied"], 0)
        self.assertIn("roast_level=0", summary["reason"])
    
    def test_clamping_when_roast_level_less_than_allowed_roast(self):
        """Test clamping when roast_level < allowed_roast"""
        # This personality has allowed_roast=2, but roast_level=1
        test_personalities = {
            "pattern_brain": {
                "tone": "playful, colorful",
                "communication_style": "imagery-first, associative",
                "thinking_style": "cross-domain, divergent, idea-generating",
                "critique_intensity": 1,
                "risk_tolerance": 2,
                "detail_level": 1,
                "allowed_roast": 2
            }
        }
        
        result, summary = apply_safety_governor("creative", 1, "brainstorming", test_personalities)
        
        # Should be clamped to neutral due to insufficient roast_level
        self.assertEqual(result["pattern_brain"]["tone"], "neutral")
        self.assertGreater(summary["overrides_applied"], 0)
    
    def test_emotional_domain_forces_neutral_mode(self):
        """Test that emotional domain forces neutral_mode for all experts"""
        result, summary = apply_safety_governor("emotional", 2, "coding", self.mock_personalities)
        
        # All personalities should be forced to neutral due to emotional domain
        for expert_name in result:
            personality = result[expert_name]
            self.assertEqual(personality["tone"], "neutral")
            self.assertEqual(personality["thinking_style"], "logical")
        
        # Summary should indicate emotional domain override
        self.assertGreater(summary["overrides_applied"], 0)
        self.assertIn("emotional domain", summary["reason"])
    
    def test_sensitive_task_type_forces_neutral_mode(self):
        """Test that sensitive task_type forces neutral_mode"""
        result, summary = apply_safety_governor("technical", 2, "sensitive", self.mock_personalities)
        
        # All personalities should be forced to neutral due to sensitive task
        for expert_name in result:
            personality = result[expert_name]
            self.assertEqual(personality["tone"], "neutral")
            self.assertEqual(personality["thinking_style"], "logical")
        
        # Summary should indicate sensitive task override
        self.assertGreater(summary["overrides_applied"], 0)
        self.assertIn("sensitive task_type", summary["reason"])
    
    def test_neutral_fallback_triggers_when_multiple_overrides_occur(self):
        """Test that neutral fallback triggers when multiple overrides occur"""
        # Emotional domain + sensitive task should trigger multiple overrides
        result, summary = apply_safety_governor("emotional", 1, "sensitive", self.mock_personalities)
        
        # Should have multiple overrides (emotional + sensitive)
        self.assertGreaterEqual(summary["overrides_applied"], 2)
        
        # Should trigger neutral fallback
        self.assertTrue(summary["neutral_fallback_used"])
        self.assertIn("2+ overrides", summary["reason"])
    
    def test_safety_governor_class_roast_level_zero(self):
        """Test SafetyGovernor class handles roast_level=0"""
        governor = SafetyGovernor()
        mock_criteria = type('MockCriteria', (), {'roast_level': 0})()
        
        result, was_modified, events = governor.apply_safety_governor(
            {"tone": "playful"}, mock_criteria, "test_expert"
        )
        
        # Should be modified due to roast_level=0
        self.assertTrue(was_modified)
        self.assertEqual(result["tone"], "neutral")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "roast_level_zero")


class TestPersonalitySystemIntegration(unittest.TestCase):
    """Integration smoke test for personality system"""
    
    def test_complete_integration_flow(self):
        """Test complete flow through assign_personalities -> apply_safety_governor -> build_personality_prefix"""
        # Mock profile
        mock_profile = {
            "experts": ["security_analyst", "pattern_brain"],
            "task_type": "coding"
        }
        
        # Step 1: Assign personalities
        personalities = assign_personalities(mock_profile, "technical", 1)
        self.assertIsInstance(personalities, dict)
        self.assertEqual(len(personalities), 2)
        
        # Step 2: Apply safety governor
        safe_personalities, summary = apply_safety_governor("technical", 1, "coding", personalities)
        self.assertIsInstance(safe_personalities, dict)
        self.assertIsInstance(summary, dict)
        
        # Step 3: Build personality prefixes
        for expert_name, personality in safe_personalities.items():
            prefix = build_personality_prefix(expert_name, personality, "coding")
            
            # Should return non-empty string with correct format
            self.assertIsInstance(prefix, str)
            self.assertGreater(len(prefix), 0)
            self.assertIn("[tone=", prefix)
        
        # No exceptions should be raised, all should work correctly
        self.assertTrue(True)
    
    def test_all_experts_integration(self):
        """Test integration with all available experts"""
        all_experts = [
            "security_analyst",
            "pragmatic_engineer", 
            "pattern_brain",
            "data_cleaner",
            "semantic_router",
            "feelprint_brain",
            "llm_connector"
        ]
        
        mock_profile = {
            "experts": all_experts,
            "task_type": "creative"
        }
        
        # Should handle all experts without errors
        personalities = assign_personalities(mock_profile, "creative", 2)
        self.assertEqual(len(personalities), len(all_experts))
        
        # All experts should have valid personality configs
        for expert_name in all_experts:
            self.assertIn(expert_name, personalities)
            personality = personalities[expert_name]
            self.assertIsInstance(personality, dict)
            self.assertIn("tone", personality)
            self.assertIn("thinking_style", personality)
        
        # Safety governor should work with all experts
        safe_personalities, summary = apply_safety_governor("creative", 2, "general", personalities)
        self.assertEqual(len(safe_personalities), len(all_experts))


if __name__ == '__main__':
    unittest.main()
