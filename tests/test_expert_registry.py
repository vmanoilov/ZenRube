#!/usr/bin/env python3
"""
Test script for ExpertRegistry module.

This script tests the functionality of the ExpertRegistry class to ensure
it can discover, validate, and load expert modules correctly.
"""

import sys
import os

# Add the project root and src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_root)

def test_expert_registry():
    """Test the ExpertRegistry functionality."""
    print("🧪 Testing ExpertRegistry module...")
    
    try:
        # Import ExpertRegistry
        from src.zenrube.experts.expert_registry import ExpertRegistry
        print("✅ Successfully imported ExpertRegistry")
        
        # Initialize registry
        registry = ExpertRegistry()
        print("✅ Successfully initialized ExpertRegistry")
        
        # Test 1: discover_experts()
        print("\n📋 Test 1: Discovering experts...")
        discovered = registry.discover_experts()
        print(f"Found {len(discovered)} experts: {list(discovered.keys())}")
        print(f"Expert mappings: {discovered}")
        
        if len(discovered) > 0:
            print("✅ discover_experts() working correctly")
        else:
            print("❌ discover_experts() found no experts")
            return False
        
        # Test 2: list_available_experts()
        print("\n📋 Test 2: Listing available experts...")
        available = registry.list_available_experts()
        print(f"Available experts: {available}")
        
        if available == list(discovered.keys()):
            print("✅ list_available_experts() working correctly")
        else:
            print("❌ list_available_experts() mismatch")
            return False
        
        # Test 3: load_expert()
        print("\n📋 Test 3: Loading an expert...")
        test_expert_name = available[0]  # Load the first available expert
        print(f"Loading expert: {test_expert_name}")
        
        try:
            expert_instance = registry.load_expert(test_expert_name)
            print(f"✅ Successfully loaded {test_expert_name}: {type(expert_instance).__name__}")
        except Exception as e:
            print(f"❌ Failed to load expert {test_expert_name}: {e}")
            return False
        
        # Test 4: get_expert_info()
        print("\n📋 Test 4: Getting expert info...")
        expert_info = registry.get_expert_info(test_expert_name)
        
        if expert_info:
            print(f"Expert info for {test_expert_name}:")
            for key, value in expert_info.items():
                print(f"  {key}: {value}")
            print("✅ get_expert_info() working correctly")
        else:
            print(f"❌ get_expert_info() failed for {test_expert_name}")
            return False
        
        # Test 5: validate_metadata() on existing modules
        print("\n📋 Test 5: Validating metadata...")
        try:
            import importlib
            
            # Test with data_cleaner module
            data_cleaner_module = importlib.import_module("src.zenrube.experts.data_cleaner")
            is_valid = registry.validate_metadata(data_cleaner_module)
            
            if is_valid:
                print("✅ validate_metadata() correctly validated data_cleaner module")
            else:
                print("❌ validate_metadata() failed to validate data_cleaner module")
                return False
                
        except Exception as e:
            print(f"❌ validate_metadata() test failed: {e}")
            return False
        
        # Test 6: Load all discovered experts
        print("\n📋 Test 6: Loading all discovered experts...")
        success_count = 0
        
        for expert_name in available:
            try:
                expert_instance = registry.load_expert(expert_name)
                print(f"  ✅ {expert_name}: {type(expert_instance).__name__}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ {expert_name}: {e}")
        
        if success_count == len(available):
            print(f"✅ Successfully loaded all {success_count} experts")
        else:
            print(f"❌ Only loaded {success_count}/{len(available)} experts")
            return False
        
        print("\n🎉 All tests passed! ExpertRegistry is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_expert_registry()
    sys.exit(0 if success else 1)