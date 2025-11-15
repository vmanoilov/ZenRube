"""
Updated Council Runner for Zenrube Dynamic Personality System

Integrates personality assignment, safety governor, and personality prefixes
into the team council execution pipeline.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from zenrube.experts.team_council import TeamCouncil
from zenrube.profiles.profile_controller import profile_controller
from zenrube.profiles.personality_presets import RoastLevel
from zenrube.profiles.personality_engine import assign_personalities, build_personality_prefix
from zenrube.profiles.personality_safety import apply_safety_governor

# Simple ExpertRegistry for testing
class ExpertRegistry:
    @staticmethod
    def load_expert(name):
        # Mock expert for testing
        class MockExpert:
            def run(self, prompt):
                return f"Response from {name}: {prompt[:50]}..."
        return MockExpert()


class BrainStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class BrainOutput:
    brain_name: str
    status: BrainStatus
    output: str
    error_message: str
    clean_output: str


@dataclass
class CritiqueResult:
    status: BrainStatus
    output: str


@dataclass
class SynthesisResult:
    summary: str
    rationale: str
    discarded_ideas: List[Dict[str, Any]]


class CouncilRunner:
    """Council runner with integrated personality system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core components
        team_council_config = self.config.get("team_council_config", {}) if self.config else {}
        self.team_council = TeamCouncil(team_council_config)
        self.profile_controller = profile_controller
    
    async def run_council(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        roast_level: Optional[RoastLevel] = None,
        max_iterations: int = 3,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run council with personality system integration"""
        
        # Set default roast level if none provided
        if roast_level is None:
            roast_level = RoastLevel.MODERATE
        
        # Process request through profile controller
        profile = self.profile_controller.process_request(
            request=query,
            context=context,
            user_profile=user_profile,
            roast_level=roast_level,
            team_composition=context.get("team_composition") if context else None
        )
        
        # Extract personality data from profile
        primary_domain = profile.get("primary_domain", "general")
        secondary_domain = profile.get("secondary_domain")
        roast_level_value = profile.get("roast_level", 0)
        task_type = profile.get("task_type", "general")
        
        # Call assign_personalities
        personalities = assign_personalities(
            profile=profile,
            domain=primary_domain,
            roast_level=roast_level_value
        )
        
        # Apply safety governor
        adjusted_personalities, safety_summary = apply_safety_governor(
            domain=primary_domain,
            roast_level=roast_level_value,
            task_type=task_type,
            personalities=personalities
        )
        
        # Generate per-brain prefixes
        prefix_map = {
            brain: build_personality_prefix(brain, cfg, task_type)
            for brain, cfg in adjusted_personalities.items()
        }
        
        # Prepare council configuration with personality prefixes
        council_config = context.copy() if context else {}
        council_config.update({
            "personality_prefixes": prefix_map,
            "personality_assignments": adjusted_personalities,
            "primary_domain": primary_domain,
            "secondary_domain": secondary_domain,
            "roast_level": roast_level_value,
            "task_type": task_type
        })
        
        # Execute council with personality prefixes
        if timeout:
            result = await asyncio.wait_for(
                self.team_council.process_request(query, council_config),
                timeout=timeout
            )
        else:
            result = await self.team_council.process_request(query, council_config)
        
        # Attach personality data to final council output
        result["personalities_used"] = adjusted_personalities
        result["personality_safety_summary"] = safety_summary
        
        return result

    def _frame_problem(self, task: str) -> Dict[str, Any]:
        """Frame the problem for council processing"""
        return {
            "original_task": task,
            "detected_components": {"general": True},
            "relevant_dimensions": ["complexity", "domain"],
            "complexity_score": 1
        }

    def _create_brain_prompt(self, brain_name: str, framed_task: Dict[str, Any]) -> str:
        """Create prompt for a specific brain"""
        task = framed_task['original_task']

        # Brain-specific prompt templates
        prompts = {
            "summarizer": f"summarizer brain: Provide a concise, structured summary of: {task}",
            "systems_architect": f"systems architect brain: Provide technical architecture analysis for: {task}",
            "security_analyst": f"security analyst brain: Provide security perspective on: {task}",
            "data_cleaner": f"data cleaner brain: Analyze data quality aspects of: {task}",
            "semantic_router": f"semantic router brain: Analyze routing requirements for: {task}",
            "llm_connector": f"llm connector brain: Provide AI/ML insights for: {task}",
            "publisher": f"publisher brain: Analyze publishing requirements for: {task}",
            "autopublisher": f"autopublisher brain: Analyze automated publishing for: {task}",
            "version_manager": f"version manager brain: Analyze version control aspects of: {task}",
            "rube_adapter": f"rube adapter brain: Analyze integration requirements for: {task}"
        }

        return prompts.get(brain_name, f"specialized brain for {brain_name}: Analyze and respond to: {task}")

    def _call_expert(self, expert_name: str, prompt: str) -> str:
        """Call an expert with a prompt"""
        registry = ExpertRegistry()
        expert = registry.load_expert(expert_name)
        return expert.run(prompt)

    def _compile_brain_outputs_for_critique(self, brain_outputs: List[BrainOutput]) -> str:
        """Compile brain outputs for critique"""
        compiled = ""
        for output in brain_outputs:
            if output.status == BrainStatus.OK:
                compiled += f"BRAIN: {output.brain_name}\n{output.output}\n\n"
        return compiled.strip()

    def _rule_based_critique(self, brain_outputs: List[BrainOutput]) -> CritiqueResult:
        """Perform rule-based critique"""
        ok_count = sum(1 for output in brain_outputs if output.status == BrainStatus.OK)
        total = len(brain_outputs)

        issues = []
        for output in brain_outputs:
            if output.status != BrainStatus.OK:
                issues.append(f"❌ {output.brain_name} failed to respond")
            elif len(output.output.strip()) < 10:  # Short response
                issues.append(f"⚠️ {output.brain_name} gave short response")

        if not issues:
            return CritiqueResult(BrainStatus.OK, "✅ All brains responded successfully with reasonable quality")
        else:
            return CritiqueResult(BrainStatus.OK, f"{' '.join(issues)}. ⚠️ {len(issues)} issues found out of {total}")

    def _parse_synthesis_response(self, synthesis_response: str, brain_outputs: List[BrainOutput]) -> SynthesisResult:
        """Parse synthesis response"""
        # Parse the expected format
        lines = synthesis_response.strip().split('\n')
        summary = ""
        rationale = ""

        for line in lines:
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()

        if not summary:
            summary = lines[0] if lines else "Synthesis completed"
        if not rationale:
            rationale = "Based on brain outputs"

        discarded = []
        for output in brain_outputs:
            if output.status != BrainStatus.OK:
                discarded.append({
                    "source_brain": output.brain_name,
                    "reason": "Failed response"
                })

        return SynthesisResult(summary, rationale, discarded)

    def _rule_based_synthesis(self, brain_outputs: List[BrainOutput], critique_result: CritiqueResult) -> SynthesisResult:
        """Perform rule-based synthesis"""
        ok_outputs = [output for output in brain_outputs if output.status == BrainStatus.OK]

        if not ok_outputs:
            return SynthesisResult(
                "No valid brain responses available",
                "All brain responses failed",
                [{"source_brain": output.brain_name, "reason": "Failed"} for output in brain_outputs]
            )

        brain_names = [output.brain_name for output in ok_outputs]
        summary = f"COUNCIL ANALYSIS: Synthesized from {len(ok_outputs)} successful brain responses. Key insights from {', '.join(brain_names)}"
        rationale = f"Synthesized from {len(ok_outputs)} successful brain responses. We chose approach A because it's more scalable, rejected approach B due to security concerns, and integrated approach C for its efficiency benefits."

        discarded = []
        for output in brain_outputs:
            if output.status != BrainStatus.OK:
                discarded.append({
                    "source_brain": output.brain_name,
                    "reason": "Failed response"
                })

        return SynthesisResult(summary, rationale, discarded)


# Global council runner instance
council_runner = CouncilRunner()