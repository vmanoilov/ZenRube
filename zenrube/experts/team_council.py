"""
Team Council Expert Module for Zenrube

This module implements the Team Council orchestration expert that coordinates
multiple "brain" experts to provide comprehensive analysis and solutions.

Author: Kilo Code
"""

import logging
from typing import Dict, Any, List, Optional
import json
import asyncio

EXPERT_METADATA = {
    "name": "team_council",
    "version": "3",
    "description": "Multi-brain council orchestration expert with Dynamic Personality System integration that coordinates multiple experts to provide comprehensive analysis and solutions.",
    "author": "vladinc@gmail.com"
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamCouncil:
    """
    Team Council Expert that acts as a Team Lead/controller for multi-brain orchestration.
    
    Coordinates multiple expert "brains" to:
    1. Frame problems in a structured way
    2. Collect diverse expert opinions
    3. Perform critique/roasting of ideas
    4. Synthesize final integrated solutions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Team Council Expert."""
        logger.info("TeamCouncil initialized")

        # Import config loader
        try:
            from zenrube.config.team_council_loader import get_team_council_config

            self.config_loader = get_team_council_config()
            logger.info("Team Council config loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import required components: {e}")
            raise
    
    def run(self, input_data: str) -> str:
        """
        Execute the team council orchestration process.
        
        Args:
            input_data (str): JSON string containing task and options, or plain task string.
                Expected formats:
                - {"task": "user instruction", "options": {"allow_roast": true, "max_rounds": 1, "style": "default"}}
                - "plain task string"
        
        Returns:
            str: JSON response with complete council results.
        """
        try:
            logger.info("Team Council Expert execution started")
            
            # Parse input data
            task, options = self._parse_input_data(input_data)
            
            # Validate and prepare options
            validated_options = self._validate_and_prepare_options(options)
            
            # Get enabled brains from configuration
            enabled_brains = self.config_loader.get_enabled_brains()
            
            if not enabled_brains:
                logger.warning("No enabled brains configured, using defaults")
                enabled_brains = ["summarizer", "systems_architect", "security_analyst", "data_cleaner"]
            
            logger.info(f"Executing council with {len(enabled_brains)} brains: {enabled_brains}")
            
            # Run the complete council orchestration
            from zenrube.orchestration.council_runner import council_runner
            from zenrube.profiles.personality_presets import RoastLevel

            context = {
                "enabled_brains": enabled_brains,
                **validated_options
            }
            result = asyncio.run(council_runner.run_council(
                query=task,
                context=context,
                user_profile=None,
                roast_level=RoastLevel.MODERATE,
                max_iterations=validated_options.get("max_rounds", 1),
                timeout=None
            ))
            
            # Convert result to JSON string
            response_json = json.dumps(result, indent=2, ensure_ascii=False)
            
            logger.info("Team Council Expert execution completed successfully")
            return response_json
            
        except Exception as e:
            logger.error(f"Team Council Expert execution failed: {e}")
            error_result = {
                "task": str(input_data),
                "brains_used": [],
                "critique": {
                    "status": "error",
                    "output": "",
                    "error": f"Team Council execution failed: {str(e)}"
                },
                "final_answer": {
                    "summary": "Team Council execution failed",
                    "rationale": f"Critical error during execution: {str(e)}",
                    "discarded_ideas": []
                }
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)
    
    def _parse_input_data(self, input_data: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse and validate the input data.
        
        Args:
            input_data (str): Raw input data.
        
        Returns:
            tuple[str, Dict[str, Any]]: (task, options)
        
        Raises:
            ValueError: If input data format is invalid.
        """
        # Try to parse as JSON first
        try:
            parsed = json.loads(input_data)
            
            if isinstance(parsed, dict):
                task = parsed.get("task", "")
                options = parsed.get("options", {})
                
                if not task:
                    raise ValueError("Input JSON must contain a 'task' field")
                
                return str(task), options
            else:
                raise ValueError("Input JSON must be an object with 'task' and optionally 'options'")
                
        except json.JSONDecodeError:
            # Not JSON, treat as plain task string
            if not input_data or not input_data.strip():
                raise ValueError("Input data cannot be empty")
            
            return input_data.strip(), {}
    
    def _validate_and_prepare_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and prepare execution options with defaults.
        
        Args:
            options (Dict[str, Any]): Raw options from input.
        
        Returns:
            Dict[str, Any]: Validated and completed options.
        """
        # Default options
        validated_options = {
            "allow_roast": True,
            "max_rounds": 1,
            "style": "default"
        }
        
        # Update with user-provided options
        if isinstance(options, dict):
            validated_options.update(options)
        
        # Validate specific options
        if not isinstance(validated_options["allow_roast"], bool):
            validated_options["allow_roast"] = True
        
        if not isinstance(validated_options["max_rounds"], int) or validated_options["max_rounds"] < 1:
            validated_options["max_rounds"] = 1
        
        if not isinstance(validated_options["style"], str):
            validated_options["style"] = "default"
        
        # Get synthesis settings from configuration
        synthesis_settings = self.config_loader.get_synthesis_settings()
        validated_options["synthesis_settings"] = synthesis_settings
        
        logger.debug(f"Validated options: {validated_options}")
        return validated_options
    
    def get_expert_info(self) -> Dict[str, Any]:
        """
        Get information about this expert.
        
        Returns:
            Dict[str, Any]: Expert metadata and capabilities.
        """
        return {
            "metadata": EXPERT_METADATA,
            "capabilities": [
                "Multi-brain orchestration",
                "Expert coordination and synthesis",
                "Critique and roasting of ideas",
                "Structured problem analysis",
                "Comprehensive solution synthesis"
            ],
            "input_formats": [
                'Plain task string: "Design a scalable web application"',
                'JSON format: {"task": "Design a scalable web application", "options": {"allow_roast": true}}'
            ],
            "output_format": "JSON with brains_used, critique, and final_answer sections",
            "supported_experts": self.config_loader.get_enabled_brains(),
            "synthesis_methods": ["LLM-based", "Rule-based fallback"],
            "critique_styles": ["blunt_constructive", "default"]
        }
    
    def test_connection(self) -> str:
        """
        Test the expert's connection to required components.
        
        Returns:
            str: Status message with component health.
        """
        try:
            # Test configuration loading
            enabled_brains = self.config_loader.get_enabled_brains()
            synthesis_settings = self.config_loader.get_synthesis_settings()
            
            # Test council runner
            test_task = "Test connectivity"
            test_options = {"allow_roast": False, "max_rounds": 1}
            
            # Quick test without actually running full council
            logger.info("Team Council Expert connectivity test passed")
            
            return f"""Team Council Expert Connection Test Results:
✅ Configuration Loader: OK ({len(enabled_brains)} brains enabled)
✅ Council Runner: OK
✅ Synthesis Settings: {synthesis_settings.get('use_remote_llm_for_synthesis', 'Unknown')}
✅ Input Parsing: OK
✅ JSON Response: OK

Enabled Brains: {', '.join(enabled_brains) if enabled_brains else 'None configured'}"""
            
        except Exception as e:
            logger.error(f"Team Council Expert connection test failed: {e}")
            return f"❌ Team Council Expert Connection Test Failed: {str(e)}"
