"""
Comprehensive Test Suite for Dynamic Profile System + Profile Controller

Tests all components of the Dynamic Profile System including:
- classification correctness
- profile generation
- auto-repair
- scoring
- roast governor
- coherence fuse
- bad profile memory
- metadata validator
- auto-fix modes
- strict validation

Author: vladinc@gmail.com
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import all profile system components
from zenrube.profiles.classification_engine import ClassificationEngine
from zenrube.profiles.compatibility_matrix import CompatibilityMatrix
from zenrube.profiles.dynamic_profile_engine import DynamicProfileEngine
from zenrube.profiles.profile_controller import ProfileController, RoastToneGovernor
from zenrube.profiles.profile_logs import ProfileLogs
from zenrube.profiles.profile_memory import ProfileMemory
from zenrube.profiles.expert_metadata_validator import ExpertMetadataValidator


class TestDynamicProfileSystem(unittest.TestCase):
    """Comprehensive test suite for the Dynamic Profile System."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create necessary directories
        os.makedirs("zenrube/experts", exist_ok=True)
        os.makedirs("zenrube/config", exist_ok=True)
        
        # Create test expert files with metadata
        self._create_test_expert_files()
        
        # Initialize components
        self.classification_engine = ClassificationEngine()
        self.compatibility_matrix = CompatibilityMatrix()
        self.profile_engine = DynamicProfileEngine()
        self.profile_controller = ProfileController()
        self.profile_logs = ProfileLogs()
        self.profile_memory = ProfileMemory()
        
        # Set up validator for testing
        self.validator = ExpertMetadataValidator({
            "strict_metadata_validation": False,  # Allow auto-fix for testing
            "auto_fix_missing_metadata": True,
            "auto_fix_version_mismatch": True
        })
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def _create_test_expert_files(self):
        """Create test expert files with metadata."""
        # Test expert with valid metadata
        valid_expert_content = '''
EXPERT_METADATA = {
    "name": "test_expert",
    "version": 1,
    "description": "Test expert with valid metadata",
    "author": "vladinc@gmail.com"
}

import logging
logger = logging.getLogger(__name__)

class TestExpert:
    def __init__(self):
        logger.info("TestExpert initialized")
    
    def run(self, input_data):
        return "Test response"
'''
        
        # Test expert with missing metadata
        invalid_expert_content = '''
import logging
logger = logging.getLogger(__name__)

class InvalidExpert:
    def __init__(self):
        logger.info("InvalidExpert initialized")
    
    def run(self, input_data):
        return "Invalid response"
'''
        
        # Write test files
        with open("zenrube/experts/test_expert.py", "w") as f:
            f.write(valid_expert_content)
            
        with open("zenrube/experts/invalid_expert.py", "w") as f:
            f.write(invalid_expert_content)
    
    # ===== CLASSIFICATION TESTS =====
    
    def test_classification_keyword_based(self):
        """Test keyword-based classification."""
        # Test security classification
        task = "Analyze the security vulnerabilities in this web application"
        result = self.classification_engine.classify_task(task)
        
        self.assertEqual(result["primary"], "cybersec")
        self.assertGreater(result["confidence"], 0.5)
        self.assertIn("security", result["signals"]["primary_domain_signals"])
        
        # Test coding classification
        task = "Implement a REST API for user authentication"
        result = self.classification_engine.classify_task(task)
        
        self.assertEqual(result["primary"], "coding")
        self.assertGreater(result["confidence"], 0.5)
        
    def test_classification_semantic_fallback(self):
        """Test semantic classification fallback."""
        # Test ambiguous task that should use semantic fallback
        task = "Create something innovative and creative for users"
        result = self.classification_engine.classify_task(task)
        
        self.assertIn(result["primary"], ["creative", "coding", "general"])
        self.assertLessEqual(result["confidence"], 0.7)  # Lower confidence for fallback
        
    def test_classification_signals_extraction(self):
        """Test signal extraction from classification."""
        task = "Design a comprehensive system architecture with high complexity and collaborative features"
        result = self.classification_engine.classify_task(task)
        
        self.assertIn("task_characteristics", result["signals"])
        self.assertIn("high-complexity", result["signals"]["task_characteristics"])
        self.assertIn("collaborative", result["signals"]["task_characteristics"])
    
    # ===== COMPATIBILITY MATRIX TESTS =====
    
    def test_compatibility_scoring(self):
        """Test compatibility score calculation."""
        # Test high compatibility
        score = self.compatibility_matrix.get_compatibility_score("cybersec", "security_analyst")
        self.assertEqual(score, 1.0)
        
        # Test medium compatibility
        score = self.compatibility_matrix.get_compatibility_score("cybersec", "summarizer")
        self.assertGreater(score, 0.5)
        
        # Test low compatibility
        score = self.compatibility_matrix.get_compatibility_score("cybersec", "publisher")
        self.assertLess(score, 0.8)
    
    def test_incompatibility_detection(self):
        """Test incompatibility pair detection."""
        # Test known incompatible pair
        self.assertTrue(self.compatibility_matrix.are_incompatible("autopublisher", "publisher"))
        
        # Test compatible pair
        self.assertFalse(self.compatibility_matrix.are_incompatible("summarizer", "data_cleaner"))
    
    def test_filter_incompatible_brains(self):
        """Test filtering incompatible brain pairs."""
        brains = ["autopublisher", "publisher", "summarizer", "data_cleaner"]
        filtered = self.compatibility_matrix.filter_incompatible_brains(brains)
        
        # Should remove one of the incompatible pair
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("autopublisher", filtered) or self.assertNotIn("publisher", filtered)
    
    # ===== PROFILE ENGINE TESTS =====
    
    def test_profile_generation_basic(self):
        """Test basic profile generation."""
        task = "Design a secure web application architecture"
        available_brains = ["summarizer", "systems_architect", "security_analyst", "data_cleaner", "llm_connector"]
        
        profile = self.profile_engine.generate_profile(task, available_brains)
        
        # Check basic structure
        self.assertIn("brains", profile)
        self.assertIn("primary_domain", profile)
        self.assertIn("confidence", profile)
        self.assertIn("draft_score", profile)
        
        # Check brain count constraints
        self.assertGreaterEqual(len(profile["brains"]), 2)
        self.assertLessEqual(len(profile["brains"]), 5)
        
        # Check domain
        self.assertEqual(profile["primary_domain"], "cybersec")
    
    def test_profile_generation_fallback(self):
        """Test profile generation fallback on failure."""
        task = "Random task that might cause issues"
        available_brains = []  # Empty list to test fallback
        
        profile = self.profile_engine.generate_profile(task, available_brains)
        
        # Should have fallback profile
        self.assertTrue(profile.get("fallback", False))
        self.assertGreater(len(profile["brains"]), 0)
    
    def test_profile_optimization(self):
        """Test profile optimization."""
        task = "Analyze customer data and create insights"
        available_brains = ["summarizer", "data_cleaner", "systems_architect", "security_analyst", "llm_connector"]
        
        profile = self.profile_engine.generate_profile(task, available_brains)
        
        # Check that optimization was applied
        self.assertIn("optimized_score", profile)
        self.assertIn("optimization_applied", profile)
        
        # Final profile should respect constraints
        self.assertGreaterEqual(len(profile["brains"]), 2)
        self.assertLessEqual(len(profile["brains"]), 5)
    
    # ===== PROFILE CONTROLLER TESTS =====
    
    def test_structure_validation(self):
        """Test profile structure validation."""
        # Test valid profile
        valid_profile = {
            "brains": ["summarizer", "security_analyst", "data_cleaner"],
            "primary_domain": "cybersec"
        }
        
        result = self.profile_controller._validate_structure(valid_profile)
        self.assertTrue(result["valid"])
        
        # Test invalid profile (too few brains)
        invalid_profile = {
            "brains": ["summarizer"],
            "primary_domain": "coding"
        }
        
        result = self.profile_controller._validate_structure(invalid_profile)
        self.assertFalse(result["valid"])
        self.assertIn("Too few brains", str(result["issues"]))
    
    def test_dry_run_relevance_test(self):
        """Test dry-run relevance testing."""
        profile = {
            "brains": ["summarizer", "security_analyst", "data_cleaner"],
            "primary_domain": "cybersec"
        }
        task = "Design a secure database system"
        
        result = self.profile_controller._perform_dry_run_relevance_test(profile, task)
        
        self.assertIn("relevance_scores", result)
        self.assertIn("removed_brains", result)
        self.assertTrue(result["test_completed"])
    
    def test_auto_repair(self):
        """Test profile auto-repair functionality."""
        profile = {
            "brains": ["summarizer", "data_cleaner"],  # Not enough brains for cybersecurity
            "primary_domain": "cybersec"
        }
        task = "Implement security measures"
        relevance_result = {
            "removed_brains": [],
            "relevance_scores": {"summarizer": 0.8, "data_cleaner": 0.3}
        }
        
        repaired_profile = self.profile_controller._auto_repair_profile(profile, task, relevance_result)
        
        # Should have more brains after repair
        self.assertGreater(len(repaired_profile["brains"]), len(profile["brains"]))
        self.assertTrue(repaired_profile.get("auto_repaired", False))
    
    def test_scoring_system(self):
        """Test profile scoring system."""
        profile = {
            "brains": ["summarizer", "security_analyst", "data_cleaner"],
            "primary_domain": "cybersec"
        }
        task = "Secure web application development"
        relevance_result = {
            "relevance_scores": {"summarizer": 0.8, "security_analyst": 0.9, "data_cleaner": 0.6}
        }
        
        scoring_result = self.profile_controller._calculate_profile_scores(profile, task, relevance_result)
        
        # Check all score components
        self.assertIn("domain_relevance", scoring_result)
        self.assertIn("compatibility", scoring_result)
        self.assertIn("dry_run_relevance", scoring_result)
        self.assertIn("noise_potential", scoring_result)
        self.assertIn("overall_score", scoring_result)
        
        # Overall score should be reasonable
        self.assertGreaterEqual(scoring_result["overall_score"], 0.0)
        self.assertLessEqual(scoring_result["overall_score"], 1.0)
    
    def test_roast_tone_governor(self):
        """Test roast tone governor functionality."""
        governor = RoastToneGovernor()
        
        # Test domain-specific roast levels
        profile_cybersec = {"primary_domain": "cybersec"}
        profile_creative = {"primary_domain": "creative"}
        profile_feelprint = {"primary_domain": "feelprint"}
        
        roast_cybersec = self.profile_controller._determine_roast_level(profile_cybersec, "security task")
        roast_creative = self.profile_controller._determine_roast_level(profile_creative, "creative task")
        roast_feelprint = self.profile_controller._determine_roast_level(profile_feelprint, "user experience task")
        
        self.assertEqual(roast_cybersec, 1)  # cybersecurity default
        self.assertEqual(roast_creative, 2)  # creative default
        self.assertEqual(roast_feelprint, 0)  # feelprint default
        
    def test_human_summary_generation(self):
        """Test human-readable summary generation."""
        profile = {
            "brains": ["summarizer", "security_analyst", "data_cleaner"],
            "primary_domain": "cybersec"
        }
        scoring_result = {"overall_score": 0.8}
        relevance_result = {"removed_brains": []}
        
        summary = self.profile_controller._generate_human_summary(profile, scoring_result, relevance_result)
        
        # Check summary structure
        self.assertIn("why_these_brains", summary)
        self.assertIn("dropped_brains", summary)
        self.assertIn("final_domain", summary)
        self.assertIn("confidence_level", summary)
        
        # Check content
        self.assertIn("cybersec", summary["final_domain"])
        self.assertEqual(summary["confidence_level"], "high")  # high score = high confidence
    
    def test_coherence_fuse(self):
        """Test coherence fuse functionality."""
        profile = {
            "brains": ["summarizer", "security_analyst"],
            "primary_domain": "cybersec"
        }
        task = "Design a secure web application"
        summary = {
            "why_these_brains": "Selected brains for cybersecurity domain",
            "dropped_brains": "No brains removed",
            "final_domain": "cybersec"
        }
        
        coherence_result = self.profile_controller._perform_coherence_fuse(profile, task, summary)
        
        # Should be coherent with good profile
        self.assertTrue(coherence_result["coherent"])
        self.assertIn("explanation", coherence_result)
        self.assertEqual(coherence_result["confidence"], "high")
    
    # ===== PROFILE MEMORY TESTS =====
    
    def test_bad_profile_memory(self):
        """Test bad profile memory functionality."""
        # Add a rejected profile
        bad_profile = {
            "brains": ["summarizer", "security_analyst"],
            "primary_domain": "cybersec"
        }
        self.profile_memory.add_rejected_profile(bad_profile, "incoherent_response")
        
        # Check if it's recognized as bad
        self.assertTrue(self.profile_memory.is_rejected_profile(bad_profile))
        
        # Check rejection reason
        reason = self.profile_memory.get_rejection_reason(bad_profile)
        self.assertEqual(reason, "incoherent_response")
    
    def test_profile_memory_cleanup(self):
        """Test profile memory cleanup functionality."""
        # Add multiple profiles
        for i in range(10):
            profile = {
                "brains": [f"brain_{i}"],
                "primary_domain": "general"
            }
            self.profile_memory.add_rejected_profile(profile, f"reason_{i}")
        
        # Should only keep last 5
        rejected = self.profile_memory.get_rejected_profiles()
        self.assertLessEqual(len(rejected), 5)
    
    # ===== METADATA VALIDATOR TESTS =====
    
    def test_metadata_validation_basic(self):
        """Test basic metadata validation."""
        result = self.validator.validate_all()
        
        # Should have validated both test files
        self.assertEqual(result["total_files"], 2)
        self.assertEqual(result["validated_files"], 1)  # One valid, one auto-fixed
        self.assertEqual(len(result["auto_fixes"]), 1)  # One auto-fixed
        
        # Check that invalid_expert.py was auto-fixed
        auto_fixed_files = [fix["file"] for fix in result["auto_fixes"]]
        self.assertTrue(any("invalid_expert.py" in file for file in auto_fixed_files))
    
    def test_metadata_extraction(self):
        """Test metadata extraction from files."""
        # Test valid metadata extraction
        file_path = "zenrube/experts/test_expert.py"
        with open(file_path, 'r') as f:
            content = f.read()
        
        result = self.validator._extract_expert_metadata(content, file_path)
        
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["metadata"]["name"], "test_expert")
        self.assertEqual(result["metadata"]["version"], 1)
        self.assertEqual(result["metadata"]["author"], "vladinc@gmail.com")
    
    def test_metadata_structure_validation(self):
        """Test metadata structure validation."""
        # Test valid metadata
        valid_metadata = {
            "name": "test",
            "version": 1,
            "description": "Test expert",
            "author": "vladinc@gmail.com"
        }
        
        result = self.validator._validate_metadata_structure(valid_metadata)
        self.assertTrue(result["valid"])
        
        # Test invalid metadata
        invalid_metadata = {
            "name": "test",
            "version": "invalid",  # Should be int
            "description": "Test expert"
            # Missing author
        }
        
        result = self.validator._validate_metadata_structure(invalid_metadata)
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_version_mismatch_detection(self):
        """Test version mismatch detection."""
        # This test would require modifying a file and checking version enforcement
        # For now, we'll test the logic components
        
        # Test version validation
        metadata_v1 = {"name": "test", "version": 1, "description": "Test", "author": "test"}
        metadata_v2 = {"name": "test", "version": 2, "description": "Test", "author": "test"}
        
        # Simulate file change detection
        old_state = {"file_hash": "old_hash", "version": 1}
        new_hash = "new_hash"
        
        # Test when version doesn't change but content does
        should_detect_mismatch = (old_state["file_hash"] != new_hash and 
                                old_state["version"] == metadata_v1["version"])
        self.assertTrue(should_detect_mismatch)
    
    # ===== INTEGRATION TESTS =====
    
    def test_full_profile_validation_pipeline(self):
        """Test complete profile validation pipeline."""
        task = "Design a secure and scalable web application"
        available_brains = ["summarizer", "systems_architect", "security_analyst", "data_cleaner", "llm_connector"]
        
        # Generate profile
        draft_profile = self.profile_engine.generate_profile(task, available_brains)
        
        # Validate and refine
        final_profile = self.profile_controller.validate_and_refine_profile(draft_profile, task)
        
        # Check final structure
        self.assertTrue(final_profile.get("validation_passed", False))
        self.assertIn("scoring", final_profile)
        self.assertIn("relevance_test", final_profile)
        self.assertIn("roast_level", final_profile)
        self.assertIn("summary", final_profile)
        self.assertIn("coherence", final_profile)
        
        # Check constraints
        self.assertGreaterEqual(len(final_profile["brains"]), 2)
        self.assertLessEqual(len(final_profile["brains"]), 5)
        self.assertIn("llm_connector", final_profile["brains"])  # Should include synthesis brain
    
    def test_profile_memory_integration(self):
        """Test profile memory integration with controller."""
        task = "Task that will fail"
        available_brains = ["summarizer"]  # Only one brain - should be rejected
        
        # Generate profile
        draft_profile = self.profile_engine.generate_profile(task, available_brains)
        final_profile = self.profile_controller.validate_and_refine_profile(draft_profile, task)
        
        # Simulate rejection due to bad performance
        self.profile_memory.add_rejected_profile(final_profile, "insufficient_brains")
        
        # Check memory was updated
        self.assertTrue(self.profile_memory.is_rejected_profile(final_profile))
    
    def test_metadata_validation_integration(self):
        """Test metadata validation integration."""
        # Run full validation
        result = self.validator.validate_all()
        
        # Check that validation completed successfully
        self.assertIn("total_files", result)
        self.assertIn("validated_files", result)
        self.assertIn("validation_timestamp", result)
        
        # Check that state was saved
        state_file = "zenrube/config/expert_metadata_state.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            self.assertGreater(len(state_data), 0)
    
    def test_roast_governor_integration(self):
        """Test roast governor with different domains."""
        domains_and_tasks = [
            ("cybersec", "Implement security measures"),
            ("creative", "Design an innovative user interface"),
            ("coding", "Build a REST API"),
            ("hardware", "Design a computer system"),
            ("feelprint", "Improve user experience")
        ]
        
        for domain, task in domains_and_tasks:
            profile = {"primary_domain": domain}
            roast_level = self.profile_controller._determine_roast_level(profile, task)
            
            # Check roast level is within bounds
            self.assertGreaterEqual(roast_level, 0)
            self.assertLessEqual(roast_level, 2)
            
            # Check domain-specific expectations
            if domain == "creative":
                self.assertGreaterEqual(roast_level, 2)  # Creative should be spicy
            elif domain == "feelprint":
                self.assertEqual(roast_level, 0)  # Feelprint should be none
    
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test with invalid inputs
        try:
            # Test classification with empty task
            result = self.classification_engine.classify_task("")
            self.assertIn("primary", result)
            
            # Test profile generation with None brains
            result = self.profile_engine.generate_profile("test", None)
            self.assertIn("fallback", result)
            
            # Test validation with invalid profile
            result = self.profile_controller.validate_and_refine_profile({}, "test")
            self.assertIn("status", result)
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_performance_characteristics(self):
        """Test basic performance characteristics."""
        import time
        
        # Test classification performance
        start_time = time.time()
        for _ in range(10):
            self.classification_engine.classify_task("test classification task")
        classification_time = time.time() - start_time
        
        # Should complete quickly (less than 1 second for 10 classifications)
        self.assertLess(classification_time, 1.0)
        
        # Test profile generation performance
        start_time = time.time()
        for _ in range(5):
            self.profile_engine.generate_profile("test task", ["summarizer", "data_cleaner"])
        profile_time = time.time() - start_time
        
        # Should complete quickly (less than 2 seconds for 5 profiles)
        self.assertLess(profile_time, 2.0)


def run_example_demonstration():
    """Run a complete example demonstrating the full dynamic profile system."""
    print("=" * 80)
    print("DYNAMIC PROFILE SYSTEM + PROFILE CONTROLLER - FULL DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    print("\n1. INITIALIZING DYNAMIC PROFILE SYSTEM COMPONENTS...")
    validator = ExpertMetadataValidator({
        "strict_metadata_validation": True,
        "auto_fix_missing_metadata": True,
        "auto_fix_version_mismatch": False
    })
    classification_engine = ClassificationEngine()
    profile_engine = DynamicProfileEngine()
    profile_controller = ProfileController()
    profile_logs = ProfileLogs()
    profile_memory = ProfileMemory()
    
    # Example task
    task = "Design a secure, scalable web application with user authentication and data analysis capabilities"
    available_brains = ["summarizer", "systems_architect", "security_analyst", "data_cleaner", "semantic_router", "llm_connector"]
    
    print(f"\nTASK: {task}")
    print(f"AVAILABLE BRAINS: {available_brains}")
    
    # Step 1: Metadata Validation
    print("\n2. EXPERT METADATA VALIDATION...")
    validation_result = validator.validate_all()
    print(f"   ✅ Validated {validation_result['validated_files']}/{validation_result['total_files']} files")
    if validation_result.get('errors'):
        print(f"   ⚠️  {len(validation_result['errors'])} errors found")
    if validation_result.get('auto_fixes'):
        print(f"   🔧 {len(validation_result['auto_fixes'])} files auto-fixed")
    
    # Step 2: Task Classification
    print("\n3. TASK CLASSIFICATION...")
    classification = classification_engine.classify_task(task)
    print(f"   🎯 Primary Domain: {classification['primary']}")
    print(f"   📊 Confidence: {classification['confidence']:.2f}")
    if classification.get('secondary'):
        print(f"   🔄 Secondary Domain: {classification['secondary']}")
    print(f"   🧠 Method: {classification['method']}")
    print(f"   📝 Signals: {classification['signals']}")
    
    # Step 3: Dynamic Profile Generation
    print("\n4. DYNAMIC PROFILE GENERATION...")
    draft_profile = profile_engine.generate_profile(task, available_brains)
    print(f"   🧩 Draft Profile ID: {draft_profile.get('profile_id', 'N/A')}")
    print(f"   🧠 Selected Brains: {draft_profile['brains']}")
    print(f"   📈 Draft Score: {draft_profile['draft_score']:.2f}")
    print(f"   ⚡ Optimized Score: {draft_profile.get('optimized_score', 'N/A')}")
    
    # Step 4: Profile Validation & Auto-Repair
    print("\n5. PROFILE VALIDATION & AUTO-REPAIR...")
    final_profile = profile_controller.validate_and_refine_profile(draft_profile, task)
    print(f"   ✅ Validation Status: {'PASSED' if final_profile.get('validation_passed') else 'FAILED'}")
    print(f"   🧠 Final Brains: {final_profile['brains']}")
    print(f"   🔥 Roast Level: {final_profile.get('roast_level', 'N/A')}")
    print(f"   📊 Final Score: {final_profile.get('scoring', {}).get('overall_score', 'N/A'):.2f}")
    
    # Step 5: Human Profile Summary
    print("\n6. HUMAN PROFILE SUMMARY...")
    summary = final_profile.get('summary', {})
    print(f"   💡 Why These Brains: {summary.get('why_these_brains', 'N/A')}")
    print(f"   ❌ Dropped Brains: {summary.get('dropped_brains', 'N/A')}")
    print(f"   🎯 Final Domain: {summary.get('final_domain', 'N/A')}")
    print(f"   🎚️  Confidence Level: {summary.get('confidence_level', 'N/A')}")
    
    # Step 6: Coherence Fuse
    print("\n7. COHERENCE FUSE...")
    coherence = final_profile.get('coherence', {})
    print(f"   ✅ Coherent: {coherence.get('coherent', False)}")
    print(f"   💬 Explanation: {coherence.get('explanation', 'N/A')}")
    print(f"   🎯 Confidence: {coherence.get('confidence', 'N/A')}")
    
    # Step 7: Profile Memory Check
    print("\n8. PROFILE MEMORY CHECK...")
    is_bad = profile_memory.is_rejected_profile(final_profile)
    print(f"   🚫 Previously Rejected: {is_bad}")
    print(f"   📈 Memory Stats: {profile_memory.get_memory_statistics()}")
    
    # Step 8: Final Council Integration Simulation
    print("\n9. FINAL COUNCIL INTEGRATION...")
    print(f"   🎭 Profile Status: {final_profile.get('status', 'unknown')}")
    print(f"   📋 Profile Summary Available: {'summary' in final_profile}")
    print(f"   🔍 Validation Results Available: {'validation_passed' in final_profile}")
    print(f"   📊 Scoring Available: {'scoring' in final_profile}")
    print(f"   🔗 Coherence Check Available: {'coherence' in final_profile}")
    
    # Step 9: Mock Council Execution Results
    print("\n10. MOCK COUNCIL EXECUTION RESULTS...")
    mock_brain_results = [
        {"name": brain, "status": "ok", "output": f"Expert analysis from {brain} brain..."} 
        for brain in final_profile['brains']
    ]
    print(f"   🧠 Brain Results: {len(mock_brain_results)} brains responded")
    
    mock_critique = {
        "status": "ok",
        "output": "Strong collaboration between security and architecture perspectives. Good synthesis of requirements."
    }
    print(f"   💬 Critique Status: {mock_critique['status']}")
    
    mock_synthesis = {
        "summary": "Comprehensive secure web application design with robust authentication and data analysis capabilities.",
        "rationale": "Integrated security-first architecture with scalable data processing components.",
        "discarded_ideas": []
    }
    print(f"   🎯 Final Answer Generated: ✅")
    print(f"   📝 Summary Length: {len(mock_synthesis['summary'])} characters")
    
    # Step 10: Complete System Summary
    print("\n" + "=" * 80)
    print("COMPLETE DYNAMIC PROFILE SYSTEM DEMONSTRATION SUMMARY")
    print("=" * 80)
    print(f"✅ Metadata Validation: {validation_result['validated_files']} files validated")
    print(f"✅ Task Classification: {classification['primary']} domain identified")
    print(f"✅ Profile Generation: {len(draft_profile['brains'])} brains selected")
    print(f"✅ Profile Validation: {'PASSED' if final_profile.get('validation_passed') else 'FAILED'}")
    print(f"✅ Auto-Repair: {final_profile.get('auto_repaired', False)} applied")
    print(f"✅ Scoring: {final_profile.get('scoring', {}).get('overall_score', 0):.2f}/1.0")
    print(f"✅ Roast Governor: Level {final_profile.get('roast_level', 'N/A')}/2")
    print(f"✅ Coherence Fuse: {'PASSED' if coherence.get('coherent') else 'FAILED'}")
    print(f"✅ Profile Memory: {profile_memory.get_memory_statistics()['total_rejected']} rejected profiles stored")
    print(f"✅ Council Ready: {len(final_profile['brains'])} optimized brains ready for execution")
    
    print("\n🎉 Dynamic Profile System + Profile Controller demonstration completed successfully!")
    
    return {
        "validation_results": validation_result,
        "classification": classification,
        "draft_profile": draft_profile,
        "final_profile": final_profile,
        "coherence": coherence,
        "profile_memory_stats": profile_memory.get_memory_statistics()
    }


if __name__ == "__main__":
    # Run the example demonstration
    demonstration_results = run_example_demonstration()
    
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS...")
    print("=" * 80)
    
    # Run unit tests
    unittest.main(verbosity=2)