"""
Test suite for Zenrube Personality System
Tests personality_presets, personality_engine, and personality_safety modules.
"""

import unittest
from zenrube.profiles.personality_presets import PERSONALITY_PRESETS, get_personality, get_neutral_personality, get_default_personality
from zenrube.profiles.personality_engine import select_personality_mode, assign_personalities, build_personality_prefix
from zenrube.profiles.personality_safety import apply_safety_governor


class TestPersonalityPresets(unittest.TestCase):
    """Test personality_presets.py functionality"""
    
    def setUp(self):
        self.expected_experts = [
            "security_analyst", "pragmatic_engineer", "pattern_brain",
            "data_cleaner", "semantic_router", "feelprint_brain", "llm_connector"
        ]
        
        self.expected_alternate_modes = {
            "security_analyst": "turing_strategist",
            "pragmatic_engineer": "builder_minimalist",
            "pattern_brain": "da_vinci_ideator",
            "data_cleaner": "info_kondo",
            "semantic_router": "sherlock_navigator",
            "feelprint_brain": "jung_listener",
            "llm_connector": "neutral_researcher"
        }
    
    def test_all_expert_keys_exist(self):
        """Test that all expected expert keys exist in PERSONALITY_PRESETS"""
        for expert in self.expected_experts:
            self.assertIn(expert, PERSONALITY_PRESETS, f"Missing expert: {expert}")
    
    def test_each_expert_has_exactly_two_modes(self):
        """Test that each expert contains exactly two modes"""
        for expert, modes in PERSONALITY_PRESETS.items():
            self.assertEqual(len(modes), 2, f"Expert {expert} should have exactly 2 modes")
            self.assertIn("neutral_mode", modes, f"Expert {expert} missing neutral_mode")
            # Check that the alternate mode exists
            alternate_mode = self.expected_alternate_modes.get(expert)
            if alternate_mode:
                self.assertIn(alternate_mode, modes, f"Expert {expert} missing alternate mode: {alternate_mode}")
    
    def test_neutral_mode_existence_per_expert(self):
        """Test neutral_mode existence per expert"""
        for expert in self.expected_experts:
            neutral_cfg = get_neutral_personality(expert)
            self.assertIsNotNone(neutral_cfg, f"neutral_mode not found for {expert}")
            self.assertIsInstance(neutral_cfg, dict, f"neutral_mode should be dict for {expert}")
            # Check required fields
            required_fields = ["tone", "communication_style", "thinking_style", 
                             "critique_intensity", "risk_tolerance", "detail_level", "allowed_roast"]
            for field in required_fields:
                self.assertIn(field, neutral_cfg, f"Missing field {field} in neutral_mode for {expert}")
    
    def test_get_personality_returns_correct_dict(self):
        """Test get_personality returns correct dict"""
        # Test existing personality
        personality = get_personality("security_analyst", "turing_strategist")
        self.assertIsInstance(personality, dict, "get_personality should return dict")
        self.assertEqual(personality["tone"], "dry, precise")
        
        # Test non-existing personality
        personality = get_personality("nonexistent", "nonexistent")
        self.assertEqual(personality, {}, "get_personality should return empty dict for invalid inputs")
    
    def test_get_neutral_personality_returns_neutral_mode(self):
        """Test get_neutral_personality returns neutral_mode"""
        neutral_cfg = get_neutral_personality("security_analyst")
        expected_neutral = PERSONALITY_PRESETS["security_analyst"]["neutral_mode"]
        self.assertEqual(neutral_cfg, expected_neutral)
        
        # Test non-existing expert
        neutral_cfg = get_neutral_personality("nonexistent")
        self.assertEqual(neutral_cfg, {}, "get_neutral_personality should return empty dict for invalid expert")


class TestPersonalityEngine(unittest.TestCase):
    """Test personality_engine.py functionality"""
    
    def setUp(self):
        self.sample_profile = {
            "experts": ["security_analyst", "pattern_brain", "data_cleaner"],
            "task_type": "technical",
            "context": {}
        }
    
    def test_assign_personalities_returns_personality_for_each_expert(self):
        """Test assign_personalities returns a personality for each expert"""
        personalities = assign_personalities(self.sample_profile, "technical", 1)
        
        self.assertEqual(len(personalities), len(self.sample_profile["experts"]))
        for expert in self.sample_profile["experts"]:
            self.assertIn(expert, personalities, f"Missing personality for {expert}")
            self.assertIsInstance(personalities[expert], dict, f"Personality should be dict for {expert}")
    
    def test_build_personality_prefix_returns_non_empty_string(self):
        """Test build_personality_prefix returns a non-empty string"""
        personality_cfg = {
            "tone": "neutral",
            "communication_style": "clear, structured",
            "thinking_style": "logical, evidence-first",
            "detail_level": 2,
            "critique_intensity": 1
        }
        
        prefix = build_personality_prefix("security_analyst", personality_cfg, "technical")
        
        self.assertIsInstance(prefix, str, "build_personality_prefix should return string")
        self.assertGreater(len(prefix), 0, "prefix should not be empty")
        self.assertIn("tone=neutral", prefix, "prefix should contain tone")
        self.assertIn("style=clear, structured", prefix, "prefix should contain communication style")
    
    def test_technical_tasks_prefer_neutral_mode_when_roast_level_low(self):
        """Test technical tasks prefer neutral_mode when roast_level is low"""
        # Low roast level should prefer neutral_mode
        personality = select_personality_mode("security_analyst", "technical", 0, "technical")
        self.assertEqual(personality.get("tone"), "neutral")
        
        # Very low roast level should force neutral_mode
        personality = select_personality_mode("security_analyst", "technical", 1, "technical")
        # With low roast level, should stick to neutral_mode due to roast constraints
    
    def test_creative_tasks_choose_alternate_mode_when_allowed(self):
        """Test creative tasks choose alternate mode when allowed"""
        # Higher roast level should allow alternate mode for creative tasks
        personality = select_personality_mode("pattern_brain", "creative", 2, "creative")
        # Should potentially choose da_vinci_ideator for creative tasks with sufficient roast
    
    def test_roast_level_zero_forces_neutral_mode(self):
        """Test roast_level == 0 forces neutral_mode"""
        personality = select_personality_mode("security_analyst", "creative", 0, "creative")
        self.assertEqual(personality.get("tone"), "neutral")
        
        personality = select_personality_mode("pattern_brain", "creative", 0, "creative")
        self.assertEqual(personality.get("tone"), "soft")  # pattern_brain neutral_mode tone


class TestPersonalitySafety(unittest.TestCase):
    """Test personality_safety.py functionality"""
    
    def setUp(self):
        self.sample_personalities = {
            "security_analyst": {
                "tone": "dry, precise",
                "allowed_roast": 2
            },
            "pattern_brain": {
                "tone": "playful, colorful",
                "allowed_roast": 2
            }
        }
    
    def test_roast_clamping_roast_level_zero(self):
        """Test roast clamping when roast_level == 0"""
        adjusted_personalities, safety_summary = apply_safety_governor(
            "technical", 0, "normal", self.sample_personalities
        )
        
        # All personalities should be clamped to neutral
        for personality in adjusted_personalities.values():
            self.assertEqual(personality.get("tone"), "neutral")
        
        self.assertGreater(safety_summary["overrides_applied"], 0, "Should have overrides for roast_level=0")
    
    def test_clamping_when_roast_level_less_than_allowed_roast(self):
        """Test clamping when roast_level < allowed_roast"""
        # roast_level=1, but security_analyst has allowed_roast=2
        adjusted_personalities, safety_summary = apply_safety_governor(
            "technical", 1, "normal", self.sample_personalities
        )
        
        # security_analyst should be clamped
        self.assertEqual(adjusted_personalities["security_analyst"].get("tone"), "neutral")
    
    def test_emotional_domain_forces_neutral_mode(self):
        """Test emotional domain forces neutral_mode"""
        adjusted_personalities, safety_summary = apply_safety_governor(
            "emotional", 2, "normal", self.sample_personalities
        )
        
        # All personalities should be neutral for emotional domain
        for personality in adjusted_personalities.values():
            self.assertEqual(personality.get("tone"), "neutral")
    
    def test_sensitive_task_type_forces_neutral_mode(self):
        """Test sensitive task_type forces neutral_mode"""
        adjusted_personalities, safety_summary = apply_safety_governor(
            "technical", 2, "sensitive", self.sample_personalities
        )
        
        # All personalities should be neutral for sensitive tasks
        for personality in adjusted_personalities.values():
            self.assertEqual(personality.get("tone"), "neutral")
    
    def test_neutral_fallback_triggers_multiple_overrides(self):
        """Test neutral fallback triggers when multiple overrides occur"""
        # Combine emotional domain with roast clamping
        adjusted_personalities, safety_summary = apply_safety_governor(
            "emotional", 0, "sensitive", self.sample_personalities
        )
        
        # Should trigger neutral fallback
        self.assertTrue(safety_summary["neutral_fallback_used"], "Should trigger neutral fallback")
        
        # All personalities should be neutral
        for personality in adjusted_personalities.values():
            self.assertEqual(personality.get("tone"), "neutral")


class TestIntegrationSmokeTest(unittest.TestCase):
    """Integration smoke test for the personality system"""
    
    def test_end_to_end_personality_flow(self):
        """Test the complete personality flow with mock data"""
        # Create mock profile
        mock_profile = {
            "experts": ["security_analyst", "data_cleaner"],
            "task_type": "analysis",
            "context": {}
        }
        
        # Step 1: assign personalities
        try:
            personalities = assign_personalities(mock_profile, "technical", 1)
            self.assertIsInstance(personalities, dict)
        except Exception as e:
            self.fail(f"assign_personalities failed: {e}")
        
        # Step 2: apply safety governor
        try:
            adjusted_personalities, safety_summary = apply_safety_governor(
                "technical", 1, "analysis", personalities
            )
            self.assertIsInstance(adjusted_personalities, dict)
            self.assertIsInstance(safety_summary, dict)
            self.assertIn("overrides_applied", safety_summary)
            self.assertIn("neutral_fallback_used", safety_summary)
        except Exception as e:
            self.fail(f"apply_safety_governor failed: {e}")
        
        # Step 3: build personality prefixes
        try:
            for brain, cfg in adjusted_personalities.items():
                prefix = build_personality_prefix(brain, cfg, "analysis")
                self.assertIsInstance(prefix, str)
                self.assertGreater(len(prefix), 0)
        except Exception as e:
            self.fail(f"build_personality_prefix failed: {e}")
    
    def test_output_has_correct_fields(self):
        """Test that integration produces output with correct fields"""
        mock_profile = {
            "experts": ["pattern_brain"],
            "task_type": "creative",
            "context": {}
        }
        
        personalities = assign_personalities(mock_profile, "creative", 2)
        adjusted_personalities, safety_summary = apply_safety_governor(
            "creative", 2, "creative", personalities
        )
        
        # Check personality configurations have required fields
        for brain, cfg in adjusted_personalities.items():
            required_fields = ["tone", "communication_style", "thinking_style",
                             "critique_intensity", "risk_tolerance", "detail_level", "allowed_roast"]
            for field in required_fields:
                self.assertIn(field, cfg, f"Missing field {field} in {brain}")
        
        # Check safety summary has required fields
        required_summary_fields = ["overrides_applied", "neutral_fallback_used", "reason"]
        for field in required_summary_fields:
            self.assertIn(field, safety_summary, f"Missing field {field} in safety_summary")


if __name__ == '__main__':
    unittest.main()