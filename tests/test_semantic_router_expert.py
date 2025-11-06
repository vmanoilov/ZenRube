import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zenrube.experts.semantic_router import SemanticRouterExpert


class TestSemanticRouterExpert:
    def test_basic_intent_detection(self):
        expert = SemanticRouterExpert()
        result = expert.run("Please process this invoice.")
        assert result["intent"] == "invoice"
        assert result["route"] == "finance_handler"

    def test_error_detection(self):
        expert = SemanticRouterExpert()
        result = expert.run("Error occurred while saving data")
        assert result["intent"] == "error"
        assert result["route"] == "debug_expert"

    def test_support_routing(self):
        expert = SemanticRouterExpert()
        result = expert.run("I need help resetting my password")
        assert result["intent"] == "support"
        assert result["route"] == "support_agent"

    def test_meeting_intent(self):
        expert = SemanticRouterExpert()
        result = expert.run("Let's schedule a meeting tomorrow")
        assert result["intent"] == "meeting"
        assert result["route"] == "calendar_flow"

    def test_urgent_detection(self):
        expert = SemanticRouterExpert()
        result = expert.run("This is urgent, please handle ASAP")
        assert result["intent"] == "urgent"
        assert result["route"] == "priority_handler"

    def test_no_match_fallback(self):
        expert = SemanticRouterExpert()
        result = expert.run("Just a random sentence")
        assert result["intent"] == "unknown"
        assert result["route"] == "general_handler"

    def test_output_structure(self):
        expert = SemanticRouterExpert()
        result = expert.run("System error in module")
        assert isinstance(result, dict)
        assert "input" in result
        assert "intent" in result
        assert "route" in result
