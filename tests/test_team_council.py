"""
Tests for Team Council Expert

This module contains unit tests for the Team Council orchestration system,
including tests for the expert module, configuration loader, and orchestration runner.

Author: Kilo Code
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Test imports
from zenrube.experts.team_council import TeamCouncil
from zenrube.orchestration.council_runner import CouncilRunner, BrainStatus, BrainOutput, CritiqueResult, SynthesisResult
from zenrube.config.team_council_loader import TeamCouncilConfigLoader


class TestTeamCouncilConfigLoader:
    """Test cases for TeamCouncilConfigLoader."""
    
    def test_load_config_default_path(self, tmp_path):
        """Test loading config from default path."""
        # Create a test config file
        config_data = {
            "enabled_brains": ["summarizer", "security_analyst"],
            "max_brain_outputs": 10,
            "use_remote_llm_for_synthesis": False,
            "synthesis_provider": "local"
        }
        
        config_file = tmp_path / "team_council_config.json"
        config_file.write_text(json.dumps(config_data))
        
        # Test loading
        loader = TeamCouncilConfigLoader(config_path=config_file)
        loaded_config = loader.load_config()
        
        assert loaded_config["enabled_brains"] == ["summarizer", "security_analyst"]
        assert loaded_config["max_brain_outputs"] == 10
        assert loaded_config["use_remote_llm_for_synthesis"] is False
        assert loaded_config["synthesis_provider"] == "local"
    
    def test_get_enabled_brains(self, tmp_path):
        """Test getting enabled brains."""
        config_data = {
            "enabled_brains": ["brain1", "brain2", "brain3"],
            "max_brain_outputs": 5
        }
        
        config_file = tmp_path / "team_council_config.json"
        config_file.write_text(json.dumps(config_data))
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        brains = loader.get_enabled_brains()
        
        assert brains == ["brain1", "brain2", "brain3"]
        assert len(brains) == 3
    
    def test_get_synthesis_settings(self, tmp_path):
        """Test getting synthesis settings."""
        config_data = {
            "use_remote_llm_for_synthesis": True,
            "synthesis_provider": "llm_connector",
            "critique_style": "blunt_constructive",
            "roasting_enabled": True
        }
        
        config_file = tmp_path / "team_council_config.json"
        config_file.write_text(json.dumps(config_data))
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        settings = loader.get_synthesis_settings()
        
        assert settings["use_remote_llm_for_synthesis"] is True
        assert settings["synthesis_provider"] == "llm_connector"
        assert settings["critique_style"] == "blunt_constructive"
        assert settings["roasting_enabled"] is True
    
    def test_is_critique_enabled(self, tmp_path):
        """Test critique enabled checking."""
        # Test enabled
        config_data = {"roasting_enabled": True}
        config_file = tmp_path / "team_council_config.json"
        config_file.write_text(json.dumps(config_data))
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        assert loader.is_critique_enabled() is True
        
        # Test disabled
        config_data = {"roasting_enabled": False}
        config_file.write_text(json.dumps(config_data))
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        assert loader.is_critique_enabled() is False
    
    def test_get_critique_style(self, tmp_path):
        """Test getting critique style."""
        config_data = {"critique_style": "gentle"}
        config_file = tmp_path / "team_council_config.json"
        config_file.write_text(json.dumps(config_data))
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        assert loader.get_critique_style() == "gentle"
    
    def test_config_file_not_found(self, tmp_path):
        """Test handling of missing config file."""
        config_file = tmp_path / "nonexistent.json"
        loader = TeamCouncilConfigLoader(config_path=config_file)
        
        with pytest.raises(FileNotFoundError):
            loader.load_config()
    
    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("invalid json content")
        
        loader = TeamCouncilConfigLoader(config_path=config_file)
        
        with pytest.raises(json.JSONDecodeError):
            loader.load_config()


class TestCouncilRunner:
    """Test cases for CouncilRunner."""
    
    def test_frame_problem(self):
        """Test problem framing functionality."""
        runner = CouncilRunner()
        
        task = "Design a secure web application for e-commerce"
        framed = runner._frame_problem(task)
        
        assert "original_task" in framed
        assert framed["original_task"] == task
        assert "detected_components" in framed
        assert "relevant_dimensions" in framed
        assert isinstance(framed["detected_components"], dict)
        assert isinstance(framed["relevant_dimensions"], list)
        assert isinstance(framed["complexity_score"], int)
    
    def test_create_brain_prompt(self):
        """Test brain-specific prompt creation."""
        runner = CouncilRunner()
        
        framed_task = {
            "original_task": "Analyze this security vulnerability",
            "detected_components": {"security_concern": True}
        }
        
        # Test different brain types
        summarizer_prompt = runner._create_brain_prompt("summarizer", framed_task)
        assert "summarizer brain" in summarizer_prompt.lower()
        assert "concise, structured summary" in summarizer_prompt
        
        architect_prompt = runner._create_brain_prompt("systems_architect", framed_task)
        assert "systems architect brain" in architect_prompt.lower()
        assert "technical architecture" in architect_prompt
        
        security_prompt = runner._create_brain_prompt("security_analyst", framed_task)
        assert "security analyst brain" in security_prompt.lower()
        assert "security perspective" in security_prompt
        
        # Test unknown brain
        unknown_prompt = runner._create_brain_prompt("unknown_brain", framed_task)
        assert "specialized brain for unknown_brain" in unknown_prompt
    
    @patch('zenrube.orchestration.council_runner.ExpertRegistry')
    def test_call_expert_success(self, mock_registry_class):
        """Test successful expert call."""
        # Setup mock
        mock_expert = Mock()
        mock_expert.run.return_value = "Test response from expert"
        
        mock_registry = Mock()
        mock_registry.load_expert.return_value = mock_expert
        mock_registry_class.return_value = mock_registry
        
        runner = CouncilRunner()
        result = runner._call_expert("test_expert", "Test prompt")
        
        assert result == "Test response from expert"
        mock_registry.load_expert.assert_called_once_with("test_expert")
        mock_expert.run.assert_called_once_with("Test prompt")
    
    @patch('zenrube.orchestration.council_runner.ExpertRegistry')
    def test_call_expert_failure(self, mock_registry_class):
        """Test expert call failure."""
        # Setup mock to raise exception
        mock_registry = Mock()
        mock_registry.load_expert.side_effect = Exception("Expert not found")
        mock_registry_class.return_value = mock_registry
        
        runner = CouncilRunner()
        
        with pytest.raises(Exception, match="Expert not found"):
            runner._call_expert("missing_expert", "Test prompt")
    
    def test_compile_brain_outputs_for_critique(self):
        """Test compilation of brain outputs for critique."""
        runner = CouncilRunner()
        
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.OK, "Output 1", "", "Output 1"),
            BrainOutput("brain2", BrainStatus.OK, "Output 2", "", "Output 2"),
            BrainOutput("brain3", BrainStatus.ERROR, "", "Error message", "")
        ]
        
        compiled = runner._compile_brain_outputs_for_critique(brain_outputs)
        
        assert "BRAIN: brain1" in compiled
        assert "Output 1" in compiled
        assert "BRAIN: brain2" in compiled
        assert "Output 2" in compiled
        assert "BRAIN: brain3" not in compiled  # Error brain should be excluded
        
        compiled_lines = compiled.strip().split('\n')
        assert len(compiled_lines) == 5  # 2 brain sections + empty line
    
    def test_rule_based_critique(self):
        """Test rule-based critique functionality."""
        runner = CouncilRunner()
        
        # Test with mixed brain outputs
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.OK, "Good response", "", "Good response"),
            BrainOutput("brain2", BrainStatus.ERROR, "", "Connection failed", ""),
            BrainOutput("brain3", BrainStatus.OK, "Short", "", "")  # Too short
        ]
        
        critique = runner._rule_based_critique(brain_outputs)
        
        assert critique.status == BrainStatus.OK
        assert "❌" in critique.output  # Should flag errors
        assert "⚠️" in critique.output  # Should flag short responses
        assert "brain2" in critique.output  # Should mention failed brain
        assert "brain3" in critique.output  # Should mention short response
    
    def test_rule_based_critique_all_good(self):
        """Test rule-based critique with all good outputs."""
        runner = CouncilRunner()
        
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.OK, "This is a good response with enough content", "", ""),
            BrainOutput("brain2", BrainStatus.OK, "Another good response here", "", "")
        ]
        
        critique = runner._rule_based_critique(brain_outputs)
        
        assert critique.status == BrainStatus.OK
        assert "✅" in critique.output
        assert "reasonable" in critique.output.lower()
    
    def test_parse_synthesis_response(self):
        """Test parsing of LLM synthesis response."""
        runner = CouncilRunner()
        
        synthesis_response = """SUMMARY: This is a comprehensive solution combining multiple approaches.

RATIONALE: We chose approach A because it's more scalable, rejected approach B due to security concerns, and integrated approach C for its efficiency benefits."""
        
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.OK, "This is a proper output from brain1 with sufficient content", "", ""),
            BrainOutput("brain2", BrainStatus.ERROR, "", "Error", "")
        ]
        
        result = runner._parse_synthesis_response(synthesis_response, brain_outputs)
        
        assert "comprehensive solution" in result.summary.lower()
        assert "approach a" in result.rationale.lower()
        assert len(result.discarded_ideas) == 1
        assert result.discarded_ideas[0]["source_brain"] == "brain2"
    
    def test_rule_based_synthesis_no_valid_brains(self):
        """Test rule-based synthesis with no valid brain outputs."""
        runner = CouncilRunner()
        
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.ERROR, "", "Failed", ""),
            BrainOutput("brain2", BrainStatus.ERROR, "", "Failed", "")
        ]
        
        critique_result = CritiqueResult(BrainStatus.OK, "Some critique")
        
        result = runner._rule_based_synthesis(brain_outputs, critique_result)
        
        assert "No valid brain responses" in result.summary
        assert "failed" in result.rationale.lower()
        assert len(result.discarded_ideas) == 2
    
    def test_rule_based_synthesis_with_valid_brains(self):
        """Test rule-based synthesis with valid brain outputs."""
        runner = CouncilRunner()
        
        brain_outputs = [
            BrainOutput("brain1", BrainStatus.OK, "Good analysis from brain1", "", ""),
            BrainOutput("brain2", BrainStatus.OK, "Useful insights from brain2", "", ""),
            BrainOutput("brain3", BrainStatus.ERROR, "", "Failed", "")
        ]
        
        critique_result = CritiqueResult(BrainStatus.OK, "Critique feedback")
        
        result = runner._rule_based_synthesis(brain_outputs, critique_result)
        
        assert "COUNCIL ANALYSIS" in result.summary
        assert "brain1" in result.summary
        assert "brain2" in result.summary
        assert "Synthesized from 2 successful brain responses" in result.rationale
        assert len(result.discarded_ideas) == 1


class TestTeamCouncil:
    """Test cases for TeamCouncilExpert."""
    
    def setup_method(self):
        """Setup for each test method."""
        with patch('zenrube.config.team_council_loader.get_team_council_config') as mock_config, \
             patch('zenrube.orchestration.council_runner.council_runner') as mock_council_runner:
            self.mock_config = mock_config
            self.mock_council_runner = mock_council_runner
            self.expert = TeamCouncil()
            # Make sure we have the mocked objects available
            self.expert.config_loader = self.mock_config.return_value
    
    def test_parse_input_data_json(self):
        """Test parsing JSON input data."""
        input_data = json.dumps({
            "task": "Test task",
            "options": {"allow_roast": False}
        })
        
        task, options = self.expert._parse_input_data(input_data)
        
        assert task == "Test task"
        assert options == {"allow_roast": False}
    
    def test_parse_input_data_plain_string(self):
        """Test parsing plain string input data."""
        input_data = "Test task string"
        
        task, options = self.expert._parse_input_data(input_data)
        
        assert task == "Test task string"
        assert options == {}
    
    def test_parse_input_data_invalid_json(self):
        """Test handling invalid JSON input."""
        input_data = '{"invalid": json'
        
        task, options = self.expert._parse_input_data(input_data)
        
        assert task == '{"invalid": json'
        assert options == {}
    
    def test_parse_input_data_empty(self):
        """Test handling empty input."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            self.expert._parse_input_data("")
    
    def test_parse_input_data_json_missing_task(self):
        """Test JSON input missing task field."""
        input_data = json.dumps({"options": {"allow_roast": True}})
        
        with pytest.raises(ValueError, match="Input JSON must contain a 'task' field"):
            self.expert._parse_input_data(input_data)
    
    def test_validate_and_prepare_options(self):
        """Test option validation and preparation."""
        options = {
            "allow_roast": True,
            "max_rounds": 2,
            "style": "blunt",
            "extra_option": "ignored"
        }
        
        # Mock synthesis settings
        self.expert.config_loader.get_synthesis_settings.return_value = {
            "use_remote_llm_for_synthesis": True,
            "synthesis_provider": "llm_connector"
        }
        
        validated = self.expert._validate_and_prepare_options(options)
        
        assert validated["allow_roast"] is True
        assert validated["max_rounds"] == 2
        assert validated["style"] == "blunt"
        assert validated["extra_option"] == "ignored"  # Extra options preserved
        assert "synthesis_settings" in validated
        assert validated["synthesis_settings"]["use_remote_llm_for_synthesis"] is True
    
    def test_validate_and_prepare_options_defaults(self):
        """Test option validation with defaults."""
        options = {}
        
        # Mock synthesis settings
        self.expert.config_loader.get_synthesis_settings.return_value = {
            "use_remote_llm_for_synthesis": False
        }
        
        validated = self.expert._validate_and_prepare_options(options)
        
        assert validated["allow_roast"] is True  # Default
        assert validated["max_rounds"] == 1  # Default
        assert validated["style"] == "default"  # Default
        assert validated["synthesis_settings"]["use_remote_llm_for_synthesis"] is False
    
    def test_validate_and_prepare_options_invalid_types(self):
        """Test option validation with invalid types."""
        options = {
            "allow_roast": "not_bool",  # Invalid type
            "max_rounds": 0,  # Invalid value
            "style": 123  # Invalid type
        }
        
        # Mock synthesis settings
        self.expert.config_loader.get_synthesis_settings.return_value = {}
        
        validated = self.expert._validate_and_prepare_options(options)
        
        # Should fall back to defaults for invalid types
        assert validated["allow_roast"] is True
        assert validated["max_rounds"] == 1
        assert validated["style"] == "default"
    
    @patch('zenrube.experts.team_council.json.dumps')
    @patch.object(TeamCouncil, '_parse_input_data')
    def test_run_success(self, mock_parse, mock_json_dumps):
        """Test successful expert execution."""
        # Setup mocks
        mock_parse.return_value = ("Test task", {})
        self.expert.config_loader.get_enabled_brains.return_value = ["brain1", "brain2"]
        self.mock_council_runner.run_council.return_value = {
            "task": "Test task",
            "brains_used": [],
            "critique": {"status": "ok"},
            "final_answer": {"summary": "Test summary"}
        }
        mock_json_dumps.return_value = '{"result": "test"}'

        result = self.expert.run("Test input")

        assert result == '{"result": "test"}'
        mock_parse.assert_called_once_with("Test input")
        self.mock_council_runner.run_council.assert_called_once()
    
    @patch.object(TeamCouncil, '_parse_input_data')
    def test_run_failure(self, mock_parse):
        """Test expert execution failure."""
        # Setup mocks to raise exception
        mock_parse.side_effect = Exception("Parse error")

        result = self.expert.run("Test input")

        # Should return error JSON
        result_dict = json.loads(result)
        assert result_dict["final_answer"]["summary"] == "Team Council execution failed"
        assert "Parse error" in result_dict["final_answer"]["rationale"]
    
    def test_get_expert_info(self):
        """Test getting expert information."""
        # Mock the config loader
        self.expert.config_loader.get_enabled_brains.return_value = ["brain1", "brain2"]
        
        info = self.expert.get_expert_info()
        
        assert "metadata" in info
        assert info["metadata"]["name"] == "team_council"
        assert "capabilities" in info
        assert "Multi-brain orchestration" in info["capabilities"]
        assert "input_formats" in info
        assert "supported_experts" in info
        assert info["supported_experts"] == ["brain1", "brain2"]
    
    def test_test_connection(self):
        """Test connection testing."""
        # Mock the config loader methods
        self.expert.config_loader.get_enabled_brains.return_value = ["brain1", "brain2"]
        self.expert.config_loader.get_synthesis_settings.return_value = {
            "use_remote_llm_for_synthesis": True
        }
        
        result = self.expert.test_connection()
        
        assert "✅ Configuration Loader: OK" in result
        assert "✅ Council Runner: OK" in result
        assert "brain1, brain2" in result
        assert "✅ Input Parsing: OK" in result
    
    def test_test_connection_failure(self):
        """Test connection testing failure."""
        # Mock the config loader to raise exception
        self.expert.config_loader.get_enabled_brains.side_effect = Exception("Config error")
        
        result = self.expert.test_connection()
        
        assert "❌ Team Council Expert Connection Test Failed" in result
        assert "Config error" in result


class TestIntegration:
    """Integration tests for the complete team council system."""
    
    @patch('zenrube.orchestration.council_runner.ExpertRegistry')
    @patch('zenrube.experts.team_council.get_team_council_config')
    def test_full_council_flow(self, mock_config_class, mock_registry_class):
        """Test the complete council flow from start to finish."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_enabled_brains.return_value = ["summarizer", "security_analyst"]
        mock_config.get_synthesis_settings.return_value = {"use_remote_llm_for_synthesis": False}
        mock_config_class.return_value = mock_config
        
        mock_registry = Mock()
        mock_expert = Mock()
        mock_expert.run.return_value = "Mock expert response with enough content to be considered valid"
        mock_registry.load_expert.return_value = mock_expert
        mock_registry_class.return_value = mock_registry
        
        # Create expert and run
        expert = TeamCouncil()
        
        # Test with JSON input
        input_data = json.dumps({
            "task": "Design a secure web application",
            "options": {"allow_roast": True, "max_rounds": 1}
        })
        
        result = expert.run(input_data)
        result_dict = json.loads(result)
        
        # Verify structure
        assert "task" in result_dict
        assert "brains_used" in result_dict
        assert "critique" in result_dict
        assert "final_answer" in result_dict
        
        assert result_dict["task"] == "Design a secure web application"
        assert len(result_dict["brains_used"]) == 2
        assert "summary" in result_dict["final_answer"]
        assert "rationale" in result_dict["final_answer"]
        assert "discarded_ideas" in result_dict["final_answer"]
    
    @patch('zenrube.orchestration.council_runner.ExpertRegistry')
    @patch('zenrube.experts.team_council.get_team_council_config')
    def test_council_with_empty_brains(self, mock_config_class, mock_registry_class):
        """Test council behavior when no brains are configured."""
        # Setup mocks
        mock_config = Mock()
        mock_config.get_enabled_brains.return_value = []  # No brains
        mock_config.get_synthesis_settings.return_value = {}
        mock_config_class.return_value = mock_config
        
        # Mock registry
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        
        expert = TeamCouncilExpert()
        
        input_data = "Test task with no brains"
        result = expert.run(input_data)
        result_dict = json.loads(result)
        
        # Should handle gracefully
        assert result_dict["final_answer"]["summary"] is not None
        assert len(result_dict["brains_used"]) == 0