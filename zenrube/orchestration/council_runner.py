"""
Team Council Orchestration Runner - Dynamic Profile System Integration

This module provides the core orchestration logic for the Zenrube multi-brain council system
with dynamic profile generation, validation, and intelligent brain selection.

Author: vladinc@gmail.com
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import dynamic profile system components
from zenrube.profiles.expert_metadata_validator import ExpertMetadataValidator
from zenrube.profiles.classification_engine import ClassificationEngine
from zenrube.profiles.dynamic_profile_engine import DynamicProfileEngine
from zenrube.profiles.profile_controller import ProfileController

# Configure logging
logger = logging.getLogger(__name__)


class BrainStatus(Enum):
    """Status of a brain's response."""
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class BrainOutput:
    """Represents output from a brain expert."""
    name: str
    status: BrainStatus
    output: str = ""
    error: str = ""
    truncated_output: str = ""


@dataclass
class CritiqueResult:
    """Result from the critique/roasting layer."""
    status: BrainStatus
    output: str
    error: str = ""


@dataclass
class SynthesisResult:
    """Result from the final synthesis phase."""
    summary: str
    rationale: str
    discarded_ideas: List[Dict[str, str]]


class CouncilRunner:
    """
    Core orchestration runner for the Team Council system with dynamic profiles.
    
    Manages the execution flow:
    1. Expert Metadata Validation
    2. Task Classification
    3. Dynamic Profile Generation
    4. Profile Validation & Auto-Repair
    5. Brain Collection (using validated profile)
    6. Critique/Roasting
    7. Final Synthesis
    8. Coherence Fuse
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Council Runner with dynamic profile system."""
        logger.info("CouncilRunner with Dynamic Profile System initialized")
        
        # Initialize dynamic profile components
        validator_config = {
            "strict_metadata_validation": config.get("strict_metadata_validation", True) if config else True,
            "auto_fix_missing_metadata": config.get("auto_fix_missing_metadata", True) if config else True,
            "auto_fix_version_mismatch": config.get("auto_fix_version_mismatch", False) if config else False
        }
        
        self.validator = ExpertMetadataValidator(validator_config)
        self.classification_engine = ClassificationEngine()
        self.profile_engine = DynamicProfileEngine()
        self.profile_controller = ProfileController()
        
        logger.info("Dynamic profile system components loaded")
    
    def run_council(self, task: str, enabled_brains: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete council orchestration process with dynamic profile system.
        
        Args:
            task (str): The user's task/problem statement.
            enabled_brains (List[str]): List of enabled brain expert names.
            options (Dict[str, Any]): Options like allow_roast, max_rounds, style.
        
        Returns:
            Dict[str, Any]: Complete council result with profile system integration.
        """
        logger.info(f"Starting dynamic council orchestration for task: {task[:100]}...")
        
        try:
            # Step 1: Expert Metadata Validation
            logger.info("Step 1: Expert metadata validation")
            validation_result = self.validator.validate_all()
            
            if validation_result.get("errors"):
                logger.warning(f"Metadata validation had errors: {len(validation_result['errors'])}")
            
            # Step 2: Task Classification
            logger.info("Step 2: Task classification")
            classification_result = self.classification_engine.classify_task(task)
            logger.info(f"Task classified as: {classification_result['primary']} (confidence: {classification_result['confidence']:.2f})")
            
            # Step 3: Dynamic Profile Generation
            logger.info("Step 3: Dynamic profile generation")
            draft_profile = self.profile_engine.generate_profile(task, enabled_brains)
            logger.info(f"Draft profile generated with {len(draft_profile['brains'])} brains")
            
            # Step 4: Profile Validation & Auto-Repair
            logger.info("Step 4: Profile validation and auto-repair")
            final_profile = self.profile_controller.validate_and_refine_profile(draft_profile, task)
            logger.info(f"Final profile validated with {len(final_profile['brains'])} brains")
            
            # Step 5: Brain Collection (using validated profile)
            logger.info("Step 5: Collecting brain opinions using validated profile")
            brain_outputs = self._collect_brain_opinions_with_profile(task, final_profile, options)
            logger.info(f"Phase 2 completed: Collected from {len(brain_outputs)} brains")
            
            # Step 6: Critique/Roasting (if enabled)
            logger.info("Step 6: Critique/Roasting")
            critique_result = self._perform_critique(brain_outputs, options)
            logger.info("Phase 3 completed: Critique/Roasting")
            
            # Step 7: Synthesis
            logger.info("Step 7: Final synthesis")
            synthesis_result = self._synthesize_results(task, brain_outputs, critique_result, options)
            logger.info("Phase 4 completed: Final Synthesis")
            
            # Step 8: Coherence Fuse
            logger.info("Step 8: Coherence fuse")
            coherence_result = self._perform_coherence_fuse(task, final_profile, synthesis_result)
            
            # Assemble final result with profile system integration
            result = {
                "task": task,
                "profile_used": final_profile,
                "summary": final_profile.get("summary", {}),
                "coherence": coherence_result,
                "logs": self._extract_profile_logs(),
                "validation_results": validation_result,
                "classification": classification_result,
                "brains_used": [
                    {
                        "name": brain.name,
                        "status": brain.status.value,
                        "output": brain.output[:1000] if brain.output else "",
                        "error": brain.error
                    }
                    for brain in brain_outputs
                ],
                "critique": {
                    "status": critique_result.status.value,
                    "output": critique_result.output
                },
                "final_answer": {
                    "summary": synthesis_result.summary,
                    "rationale": synthesis_result.rationale,
                    "discarded_ideas": synthesis_result.discarded_ideas
                }
            }
            
            logger.info("Dynamic council orchestration completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Dynamic council orchestration failed: {e}")
            return self._create_error_result_with_profile(task, str(e))
    
    def _collect_brain_opinions_with_profile(self, task: str, profile: Dict[str, Any],
                                           options: Dict[str, Any]) -> List[BrainOutput]:
        """Collect brain opinions using the validated profile."""
        brains = profile.get("brains", [])
        logger.info(f"Collecting opinions from {len(brains)} validated brains: {brains}")
        
        brain_outputs = []
        
        # Generate brain-specific prompts
        for brain_name in brains:
            try:
                logger.info(f"Querying brain: {brain_name}")
                
                # Create brain-specific prompt
                prompt = self._create_brain_prompt(brain_name, task)
                
                # Call the expert
                expert_output = self._call_expert(brain_name, prompt)
                
                # Create brain output object
                brain_output = BrainOutput(
                    name=brain_name,
                    status=BrainStatus.OK,
                    output=expert_output,
                    truncated_output=expert_output[:500] if expert_output else ""
                )
                
                brain_outputs.append(brain_output)
                logger.info(f"Brain {brain_name} responded successfully ({len(expert_output)} chars)")
                
            except Exception as e:
                logger.warning(f"Brain {brain_name} failed: {e}")
                brain_output = BrainOutput(
                    name=brain_name,
                    status=BrainStatus.ERROR,
                    error=str(e)
                )
                brain_outputs.append(brain_output)
        
        return brain_outputs
    
    def _extract_profile_logs(self) -> List[Dict[str, Any]]:
        """Extract logs from the profile system."""
        try:
            return self.profile_controller.profile_logs.get_logs()
        except Exception as e:
            logger.warning(f"Failed to extract profile logs: {e}")
            return []
    
    def _perform_coherence_fuse(self, task: str, profile: Dict[str, Any],
                              synthesis_result: SynthesisResult) -> Dict[str, Any]:
        """Perform coherence fuse to check if the answer solves the task."""
        try:
            # Check if the synthesis result appears to solve the task
            summary_lower = synthesis_result.summary.lower()
            task_lower = task.lower()
            
            # Simple coherence heuristics
            has_substantial_content = len(synthesis_result.summary) > 50
            addresses_task = any(word in summary_lower for word in task_lower.split() if len(word) > 3)
            has_rationale = len(synthesis_result.rationale) > 10
            
            coherent = has_substantial_content and addresses_task and has_rationale
            
            if coherent:
                return {
                    "coherent": True,
                    "explanation": "Answer appears to address the task comprehensively",
                    "confidence": "high"
                }
            else:
                return {
                    "coherent": False,
                    "explanation": "Answer may not fully address the task requirements",
                    "confidence": "low",
                    "coherence_warning": "Consider reviewing the task or adding more domain-specific brains"
                }
                
        except Exception as e:
            logger.warning(f"Coherence fuse failed: {e}")
            return {
                "coherent": True,  # Default to true on error
                "explanation": "Coherence check failed, proceeding with current result",
                "confidence": "uncertain"
            }
    
    def _create_error_result_with_profile(self, task: str, error: str) -> Dict[str, Any]:
        """Create an error result when the entire council fails."""
        return {
            "task": task,
            "profile_used": {
                "status": "error",
                "error": "Profile generation failed"
            },
            "summary": {"error": "Profile validation failed"},
            "coherence": {"coherent": False, "explanation": "System error occurred"},
            "logs": [],
            "validation_results": {"status": "failed", "error": error},
            "classification": {"primary": "general", "confidence": 0.0},
            "brains_used": [],
            "critique": {
                "status": "error",
                "output": "",
                "error": f"Council orchestration failed: {error}"
            },
            "final_answer": {
                "summary": "Council orchestration failed",
                "rationale": f"Critical error during execution: {error}",
                "discarded_ideas": []
            }
        }
    
    def _create_error_result(self, task: str, error: str) -> Dict[str, Any]:
        """Create an error result when the entire council fails (legacy)."""
        return {
            "task": task,
            "brains_used": [],
            "critique": {
                "status": "error",
                "output": "",
                "error": f"Council orchestration failed: {error}"
            },
            "final_answer": {
                "summary": "Council orchestration failed",
                "rationale": f"Critical error during execution: {error}",
                "discarded_ideas": []
            }
        }
    
    def _frame_problem(self, task: str) -> Dict[str, Any]:
        """Legacy problem framing (used by other methods)."""
        task_lower = task.lower()
        components = {
            "data_analysis": any(word in task_lower for word in ["analyze", "data", "dataset", "statistics"]),
            "technical_design": any(word in task_lower for word in ["design", "architecture", "system", "technical"]),
            "security_concern": any(word in task_lower for word in ["security", "vulnerability", "risk", "safe"]),
            "code_related": any(word in task_lower for word in ["code", "implement", "debug", "fix"]),
            "data_processing": any(word in task_lower for word in ["clean", "process", "transform", "format"])
        }
        
        return {
            "original_task": task,
            "detected_components": components,
            "relevant_dimensions": [k for k, v in components.items() if v],
            "complexity_score": sum(components.values())
        }
    
    def _create_brain_prompt(self, brain_name: str, task: str) -> str:
        """Create a specialized prompt for each brain type."""
        task_text = task
        
        brain_prompts = {
            "summarizer": f"""You are the summarizer brain for a multi-brain council.

Task: {task_text}

Provide a concise, structured summary focusing on:
1. Key points and main requirements
2. Critical information that must not be lost
3. Essential outcomes or deliverables

Keep your response focused and actionable. Maximum 200 words.""",
            
            "systems_architect": f"""You are the systems architect brain for a multi-brain council.

Task: {task_text}

As a systems architect, provide a technical architecture or structure for this task:
1. High-level system design
2. Key components and their relationships
3. Scalability and performance considerations
4. Technology recommendations

Be specific about architectural patterns and design principles.""",
            
            "security_analyst": f"""You are the security analyst brain for a multi-brain council.

Task: {task_text}

Analyze this task from a security perspective:
1. Potential security threats and vulnerabilities
2. Risk assessment and impact analysis
3. Security controls and mitigation strategies
4. Compliance considerations (if applicable)

Focus on practical security recommendations.""",
            
            "data_cleaner": f"""You are the data-cleaning brain for a multi-brain council.

Task: {task_text}

Address data-related aspects:
1. Data quality issues and preprocessing needs
2. Data normalization and standardization requirements
3. Data validation and cleaning strategies
4. Data format and structure recommendations

Focus on data integrity and quality.""",
            
            "semantic_router": f"""You are the semantic routing brain for a multi-brain council.

Task: {task_text}

Analyze the semantic and routing aspects:
1. Intent recognition and classification
2. Optimal routing strategies
3. Content categorization and tagging
4. Context preservation requirements

Focus on intelligent routing and categorization.""",
            
            "llm_connector": f"""You are the LLM connector brain for a multi-brain council.

Task: {task_text}

Provide comprehensive analysis and creative solutions:
1. Creative approaches and innovative thinking
2. Cross-domain connections and insights
3. Alternative perspectives and ideas
4. Advanced reasoning and synthesis

Use your full analytical and creative capabilities."""
        }
        
        base_prompt = brain_prompts.get(brain_name,
            f"You are a specialized brain for {brain_name}. Task: {task_text} Provide your expert analysis and recommendations.")
        
        context_prompt = f"""{base_prompt}

CONTEXT: This is part of a multi-brain council where different experts provide their perspectives on the same task. Your goal is to provide unique, valuable insights from your domain expertise."""
        
        return context_prompt
    
    def _call_expert(self, expert_name: str, prompt: str) -> str:
        """Call an expert using the Zenrube expert system."""
        try:
            from zenrube.experts.expert_registry import ExpertRegistry
            
            registry = ExpertRegistry()
            expert = registry.load_expert(expert_name)
            
            if hasattr(expert, 'run'):
                response = expert.run(prompt)
                return str(response)
            else:
                raise AttributeError(f"Expert {expert_name} doesn't have a run method")
                
        except Exception as e:
            logger.error(f"Failed to call expert {expert_name}: {e}")
            raise
    
    def _perform_critique(self, brain_outputs: List[BrainOutput], options: Dict[str, Any]) -> CritiqueResult:
        """Perform critique/roasting of brain outputs."""
        try:
            if not options.get("allow_roast", True):
                return CritiqueResult(
                    status=BrainStatus.SKIPPED,
                    output="Critique/roasting was disabled in options."
                )
            
            critique_input = self._compile_brain_outputs_for_critique(brain_outputs)
            
            try:
                return self._llm_based_critique(critique_input, options)
            except Exception as e:
                logger.warning(f"LLM critique failed, falling back to rule-based: {e}")
                return self._rule_based_critique(brain_outputs)
                
        except Exception as e:
            logger.error(f"Critique phase failed: {e}")
            return CritiqueResult(
                status=BrainStatus.ERROR,
                output="",
                error=str(e)
            )
    
    def _compile_brain_outputs_for_critique(self, brain_outputs: List[BrainOutput]) -> str:
        """Compile brain outputs into a format suitable for critique."""
        compiled = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.OK and brain.output:
                compiled.append(f"BRAIN: {brain.name}\nOUTPUT: {brain.output}\n")
        
        return "\n".join(compiled)
    
    def _llm_based_critique(self, critique_input: str, options: Dict[str, Any]) -> CritiqueResult:
        """Perform critique using an LLM."""
        try:
            critique_style = options.get("style", "default")
            
            critique_prompt = f"""{critique_input}

You are a sharp, constructive critic reviewing outputs from a multi-brain council.

CRITIQUE STYLE: {critique_style}
INSTRUCTIONS:
1. Identify contradictions between brain outputs
2. Call out weak reasoning or gaps in logic
3. "Roast" bad ideas constructively (focus on ideas, not people)
4. Highlight the best insights and strongest reasoning
5. Be blunt but helpful - this is meant to improve the final solution

Provide a focused critique that will help the team lead synthesize the best answer."""
            
            response = self._call_expert("llm_connector", critique_prompt)
            
            return CritiqueResult(
                status=BrainStatus.OK,
                output=response
            )
            
        except Exception as e:
            logger.error(f"LLM critique failed: {e}")
            raise
    
    def _rule_based_critique(self, brain_outputs: List[BrainOutput]) -> CritiqueResult:
        """Perform basic rule-based critique when LLM is unavailable."""
        critiques = []
        
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                critiques.append(f"❌ {brain.name}: Failed to respond - {brain.error}")
            elif not brain.output or len(brain.output.strip()) < 10:
                critiques.append(f"⚠️  {brain.name}: Response too short or empty")
            elif len(brain.output) > 2000:
                critiques.append(f"⚠️  {brain.name}: Response very long ({len(brain.output)} chars)")
        
        ok_brains = [b for b in brain_outputs if b.status == BrainStatus.OK and b.output and len(brain.output.strip()) >= 10]
        
        if len(ok_brains) < 2:
            critiques.append("⚠️  Council: Too few brains provided useful responses")
        
        critique_text = "\n".join(critiques) if critiques else "✅ All brain responses appear reasonable"
        
        return CritiqueResult(
            status=BrainStatus.OK,
            output=f"RULE-BASED CRITIQUE:\n{critique_text}"
        )
    
    def _synthesize_results(self, original_task: str, brain_outputs: List[BrainOutput],
                           critique_result: CritiqueResult, options: Dict[str, Any]) -> SynthesisResult:
        """Synthesize the final answer from all inputs."""
        try:
            synthesis_settings = options.get("synthesis_settings", {})
            use_remote_llm = synthesis_settings.get("use_remote_llm_for_synthesis", True)
            
            if use_remote_llm:
                return self._llm_based_synthesis(original_task, brain_outputs, critique_result, options)
            else:
                return self._rule_based_synthesis(brain_outputs, critique_result)
                
        except Exception as e:
            logger.error(f"Synthesis phase failed: {e}")
            return SynthesisResult(
                summary="Synthesis failed due to an error.",
                rationale=f"Error during synthesis: {str(e)}",
                discarded_ideas=[]
            )
    
    def _llm_based_synthesis(self, original_task: str, brain_outputs: List[BrainOutput],
                           critique_result: CritiqueResult, options: Dict[str, Any]) -> SynthesisResult:
        """Synthesize results using an LLM."""
        try:
            synthesis_input = self._compile_synthesis_input(original_task, brain_outputs, critique_result)
            
            synthesis_prompt = f"""{synthesis_input}

You are the Team Lead for this multi-brain council. Your task is to synthesize the final answer.

INSTRUCTIONS:
1. Select the best ideas from the brain outputs
2. Reject weak or contradictory ideas
3. Integrate complementary insights into a cohesive solution
4. Provide clear rationale for your decisions
5. Create a final answer that combines the strongest elements from all brains

Return your synthesis in the following format:
SUMMARY: [Your integrated final answer]
RATIONALE: [Why you chose certain ideas and rejected others]"""
            
            synthesis_response = self._call_expert("llm_connector", synthesis_prompt)
            
            return self._parse_synthesis_response(synthesis_response, brain_outputs)
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            raise
    
    def _compile_synthesis_input(self, original_task: str, brain_outputs: List[BrainOutput],
                               critique_result: CritiqueResult) -> str:
        """Compile all inputs for synthesis."""
        sections = [f"ORIGINAL TASK:\n{original_task}\n"]
        
        sections.append("BRAIN OUTPUTS:")
        for brain in brain_outputs:
            if brain.status == BrainStatus.OK and brain.output:
                sections.append(f"\n{brain.name.upper()}:\n{brain.output}")
            elif brain.status == BrainStatus.ERROR:
                sections.append(f"\n{brain.name.upper()}: ERROR - {brain.error}")
        
        if critique_result.output:
            sections.append(f"\nCRITIQUE/ROAST:\n{critique_result.output}")
        
        return "\n".join(sections)
    
    def _parse_synthesis_response(self, synthesis_response: str, brain_outputs: List[BrainOutput]) -> SynthesisResult:
        """Parse the LLM synthesis response into structured format."""
        lines = synthesis_response.split('\n')
        
        summary_parts = []
        rationale_parts = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SUMMARY:'):
                current_section = 'summary'
                summary_parts.append(line[8:].strip())
            elif line.upper().startswith('RATIONALE:'):
                current_section = 'rationale'
                rationale_parts.append(line[10:].strip())
            elif current_section == 'summary' and line:
                summary_parts.append(line)
            elif current_section == 'rationale' and line:
                rationale_parts.append(line)
        
        summary = ' '.join(summary_parts) if summary_parts else synthesis_response[:500]
        rationale = ' '.join(rationale_parts) if rationale_parts else "Synthesis completed based on available inputs."
        
        discarded_ideas = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": f"Brain failed to respond: {brain.error}"
                })
            elif not brain.output or len(brain.output.strip()) < 10:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": "Response too short or empty to be useful"
                })
        
        return SynthesisResult(
            summary=summary,
            rationale=rationale,
            discarded_ideas=discarded_ideas
        )
    
    def _rule_based_synthesis(self, brain_outputs: List[BrainOutput], critique_result: CritiqueResult) -> SynthesisResult:
        """Perform basic rule-based synthesis when LLM is unavailable."""
        valid_brains = [b for b in brain_outputs if b.status == BrainStatus.OK and b.output and len(brain.output.strip()) >= 10]
        
        if not valid_brains:
            discarded_ideas = []
            for brain in brain_outputs:
                if brain.status == BrainStatus.ERROR:
                    discarded_ideas.append({
                        "source_brain": brain.name,
                        "reason": f"Brain failed to respond: {brain.error}"
                    })
                else:
                    discarded_ideas.append({
                        "source_brain": brain.name,
                        "reason": "Response too short or empty to be useful"
                    })
            
            return SynthesisResult(
                summary="No valid brain responses available for synthesis.",
                rationale="All brains either failed or provided empty responses.",
                discarded_ideas=discarded_ideas
            )
        
        combined_summary = "BASED ON COUNCIL ANALYSIS:\n\n"
        
        for brain in valid_brains:
            combined_summary += f"{brain.name.upper()} PERSPECTIVE:\n{brain.output[:300]}...\n\n"
        
        rationale = f"Synthesized from {len(valid_brains)} successful brain responses."
        
        discarded_ideas = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": f"Failed to respond: {brain.error}"
                })
            elif brain not in valid_brains:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": "Response too short or empty to be useful"
                })
        
        return SynthesisResult(
            summary=combined_summary,
            rationale=rationale,
            discarded_ideas=discarded_ideas
        )
    
    def _frame_problem(self, task: str) -> Dict[str, Any]:
        """
        Phase 1: Frame the problem in a structured way.
        
        Args:
            task (str): Raw task description.
        
        Returns:
            Dict[str, Any]: Structured problem framing.
        """
        # Simple heuristic-based problem framing
        # In a more advanced implementation, this could delegate to a summarizer expert
        
        # Detect key components of the task
        task_lower = task.lower()
        components = {
            "data_analysis": any(word in task_lower for word in ["analyze", "data", "dataset", "statistics"]),
            "technical_design": any(word in task_lower for word in ["design", "architecture", "system", "technical"]),
            "security_concern": any(word in task_lower for word in ["security", "vulnerability", "risk", "safe"]),
            "code_related": any(word in task_lower for word in ["code", "implement", "debug", "fix"]),
            "data_processing": any(word in task_lower for word in ["clean", "process", "transform", "format"])
        }
        
        framed = {
            "original_task": task,
            "detected_components": components,
            "relevant_dimensions": [k for k, v in components.items() if v],
            "complexity_score": sum(components.values())
        }
        
        logger.debug(f"Problem framed with complexity score: {framed['complexity_score']}")
        return framed
    
    def _collect_brain_opinions(self, framed_task: Dict[str, Any], enabled_brains: List[str], 
                              options: Dict[str, Any]) -> List[BrainOutput]:
        """
        Phase 2: Collect opinions from each enabled brain.
        
        Args:
            framed_task (Dict[str, Any]): Structured problem framing.
            enabled_brains (List[str]): List of enabled brain expert names.
            options (Dict[str, Any]): Council execution options.
        
        Returns:
            List[BrainOutput]: List of brain outputs with status.
        """
        brain_outputs = []
        
        # Generate brain-specific prompts
        for brain_name in enabled_brains:
            try:
                logger.info(f"Querying brain: {brain_name}")
                
                # Create brain-specific prompt
                prompt = self._create_brain_prompt(brain_name, framed_task)
                
                # Call the expert
                expert_output = self._call_expert(brain_name, prompt)
                
                # Create brain output object
                brain_output = BrainOutput(
                    name=brain_name,
                    status=BrainStatus.OK,
                    output=expert_output,
                    truncated_output=expert_output[:500] if expert_output else ""
                )
                
                brain_outputs.append(brain_output)
                logger.info(f"Brain {brain_name} responded successfully ({len(expert_output)} chars)")
                
            except Exception as e:
                logger.warning(f"Brain {brain_name} failed: {e}")
                brain_output = BrainOutput(
                    name=brain_name,
                    status=BrainStatus.ERROR,
                    error=str(e)
                )
                brain_outputs.append(brain_output)
        
        return brain_outputs
    
    def _create_brain_prompt(self, brain_name: str, framed_task: Dict[str, Any]) -> str:
        """
        Create a specialized prompt for each brain type.
        
        Args:
            brain_name (str): Name of the brain expert.
            framed_task (Dict[str, Any]): Structured problem framing.
        
        Returns:
            str: Specialized prompt for the brain.
        """
        task_text = framed_task["original_task"]
        
        brain_prompts = {
            "summarizer": f"""You are the summarizer brain for a multi-brain council. 

Task: {task_text}

Provide a concise, structured summary focusing on:
1. Key points and main requirements
2. Critical information that must not be lost
3. Essential outcomes or deliverables

Keep your response focused and actionable. Maximum 200 words.""",
            
            "systems_architect": f"""You are the systems architect brain for a multi-brain council.

Task: {task_text}

As a systems architect, provide a technical architecture or structure for this task:
1. High-level system design
2. Key components and their relationships
3. Scalability and performance considerations
4. Technology recommendations

Be specific about architectural patterns and design principles.""",
            
            "security_analyst": f"""You are the security analyst brain for a multi-brain council.

Task: {task_text}

Analyze this task from a security perspective:
1. Potential security threats and vulnerabilities
2. Risk assessment and impact analysis
3. Security controls and mitigation strategies
4. Compliance considerations (if applicable)

Focus on practical security recommendations.""",
            
            "data_cleaner": f"""You are the data-cleaning brain for a multi-brain council.

Task: {task_text}

Address data-related aspects:
1. Data quality issues and preprocessing needs
2. Data normalization and standardization requirements
3. Data validation and cleaning strategies
4. Data format and structure recommendations

Focus on data integrity and quality.""",
            
            "semantic_router": f"""You are the semantic routing brain for a multi-brain council.

Task: {task_text}

Analyze the semantic and routing aspects:
1. Intent recognition and classification
2. Optimal routing strategies
3. Content categorization and tagging
4. Context preservation requirements

Focus on intelligent routing and categorization.""",
            
            "llm_connector": f"""You are the LLM connector brain for a multi-brain council.

Task: {task_text}

Provide comprehensive analysis and creative solutions:
1. Creative approaches and innovative thinking
2. Cross-domain connections and insights
3. Alternative perspectives and ideas
4. Advanced reasoning and synthesis

Use your full analytical and creative capabilities."""
        }
        
        base_prompt = brain_prompts.get(brain_name, 
            f"You are a specialized brain for {brain_name}. Task: {task_text} Provide your expert analysis and recommendations.")
        
        # Add council context
        context_prompt = f"""{base_prompt}

CONTEXT: This is part of a multi-brain council where different experts provide their perspectives on the same task. Your goal is to provide unique, valuable insights from your domain expertise."""
        
        return context_prompt
    
    def _call_expert(self, expert_name: str, prompt: str) -> str:
        """
        Call an expert using the Zenrube expert system.
        
        Args:
            expert_name (str): Name of the expert to call.
            prompt (str): Prompt to send to the expert.
        
        Returns:
            str: Expert's response.
        
        Raises:
            Exception: If expert call fails.
        """
        try:
            # Import ExpertRegistry locally to avoid circular imports
            from zenrube.experts.expert_registry import ExpertRegistry
            
            registry = ExpertRegistry()
            expert = registry.load_expert(expert_name)
            
            # Call the expert's run method
            if hasattr(expert, 'run'):
                response = expert.run(prompt)
                return str(response)
            else:
                raise AttributeError(f"Expert {expert_name} doesn't have a run method")
                
        except Exception as e:
            logger.error(f"Failed to call expert {expert_name}: {e}")
            raise
    
    def _perform_critique(self, brain_outputs: List[BrainOutput], options: Dict[str, Any]) -> CritiqueResult:
        """
        Phase 3: Perform critique/roasting of brain outputs.
        
        Args:
            brain_outputs (List[BrainOutput]): Outputs from all brains.
            options (Dict[str, Any]): Council execution options.
        
        Returns:
            CritiqueResult: Critique/roasting result.
        """
        try:
            # Check if roasting is enabled
            if not options.get("allow_roast", True):
                return CritiqueResult(
                    status=BrainStatus.SKIPPED,
                    output="Critique/roasting was disabled in options."
                )
            
            # Compile brain outputs for critique
            critique_input = self._compile_brain_outputs_for_critique(brain_outputs)
            
            # Use LLM connector for critique if available, otherwise use rule-based
            try:
                return self._llm_based_critique(critique_input, options)
            except Exception as e:
                logger.warning(f"LLM critique failed, falling back to rule-based: {e}")
                return self._rule_based_critique(brain_outputs)
                
        except Exception as e:
            logger.error(f"Critique phase failed: {e}")
            return CritiqueResult(
                status=BrainStatus.ERROR,
                output="",
                error=str(e)
            )
    
    def _compile_brain_outputs_for_critique(self, brain_outputs: List[BrainOutput]) -> str:
        """Compile brain outputs into a format suitable for critique."""
        compiled = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.OK and brain.output:
                compiled.append(f"BRAIN: {brain.name}\nOUTPUT: {brain.output}\n")
        
        return "\n".join(compiled)
    
    def _llm_based_critique(self, critique_input: str, options: Dict[str, Any]) -> CritiqueResult:
        """Perform critique using an LLM."""
        try:
            critique_style = options.get("style", "default")
            
            # Create critique prompt
            critique_prompt = f"""{critique_input}

You are a sharp, constructive critic reviewing outputs from a multi-brain council. 

CRITIQUE STYLE: {critique_style}
INSTRUCTIONS:
1. Identify contradictions between brain outputs
2. Call out weak reasoning or gaps in logic
3. "Roast" bad ideas constructively (focus on ideas, not people)
4. Highlight the best insights and strongest reasoning
5. Be blunt but helpful - this is meant to improve the final solution

Provide a focused critique that will help the team lead synthesize the best answer."""
            
            # Call LLM connector for critique
            response = self._call_expert("llm_connector", critique_prompt)
            
            return CritiqueResult(
                status=BrainStatus.OK,
                output=response
            )
            
        except Exception as e:
            logger.error(f"LLM critique failed: {e}")
            raise
    
    def _rule_based_critique(self, brain_outputs: List[BrainOutput]) -> CritiqueResult:
        """Perform basic rule-based critique when LLM is unavailable."""
        critiques = []
        
        # Basic quality checks
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                critiques.append(f"❌ {brain.name}: Failed to respond - {brain.error}")
            elif not brain.output or len(brain.output.strip()) < 10:  # Reduced threshold
                critiques.append(f"⚠️  {brain.name}: Response too short or empty")
            elif len(brain.output) > 2000:
                critiques.append(f"⚠️  {brain.name}: Response very long ({len(brain.output)} chars)")
        
        # Check for obvious issues
        ok_brains = [b for b in brain_outputs if b.status == BrainStatus.OK and b.output and len(brain.output.strip()) >= 10]  # Consistent threshold
        
        if len(ok_brains) < 2:
            critiques.append("⚠️  Council: Too few brains provided useful responses")
        
        critique_text = "\n".join(critiques) if critiques else "✅ All brain responses appear reasonable"
        
        return CritiqueResult(
            status=BrainStatus.OK,
            output=f"RULE-BASED CRITIQUE:\n{critique_text}"
        )
    
    def _synthesize_results(self, original_task: str, brain_outputs: List[BrainOutput], 
                           critique_result: CritiqueResult, options: Dict[str, Any]) -> SynthesisResult:
        """
        Phase 4: Synthesize the final answer from all inputs.
        
        Args:
            original_task (str): The original user task.
            brain_outputs (List[BrainOutput]): Outputs from all brains.
            critique_result (CritiqueResult): Result from critique phase.
            options (Dict[str, Any]): Council execution options.
        
        Returns:
            SynthesisResult: Final synthesis result.
        """
        try:
            # Check if we should use LLM for synthesis
            synthesis_settings = options.get("synthesis_settings", {})
            use_remote_llm = synthesis_settings.get("use_remote_llm_for_synthesis", True)
            
            if use_remote_llm:
                return self._llm_based_synthesis(original_task, brain_outputs, critique_result, options)
            else:
                return self._rule_based_synthesis(brain_outputs, critique_result)
                
        except Exception as e:
            logger.error(f"Synthesis phase failed: {e}")
            return SynthesisResult(
                summary="Synthesis failed due to an error.",
                rationale=f"Error during synthesis: {str(e)}",
                discarded_ideas=[]
            )
    
    def _llm_based_synthesis(self, original_task: str, brain_outputs: List[BrainOutput], 
                           critique_result: CritiqueResult, options: Dict[str, Any]) -> SynthesisResult:
        """Synthesize results using an LLM."""
        try:
            # Compile all inputs for synthesis
            synthesis_input = self._compile_synthesis_input(original_task, brain_outputs, critique_result)
            
            synthesis_prompt = f"""{synthesis_input}

You are the Team Lead for this multi-brain council. Your task is to synthesize the final answer.

INSTRUCTIONS:
1. Select the best ideas from the brain outputs
2. Reject weak or contradictory ideas
3. Integrate complementary insights into a cohesive solution
4. Provide clear rationale for your decisions
5. Create a final answer that combines the strongest elements from all brains

Return your synthesis in the following format:
SUMMARY: [Your integrated final answer]
RATIONALE: [Why you chose certain ideas and rejected others]"""
            
            # Call LLM connector for synthesis
            synthesis_response = self._call_expert("llm_connector", synthesis_prompt)
            
            # Parse the response
            return self._parse_synthesis_response(synthesis_response, brain_outputs)
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            raise
    
    def _compile_synthesis_input(self, original_task: str, brain_outputs: List[BrainOutput], 
                               critique_result: CritiqueResult) -> str:
        """Compile all inputs for synthesis."""
        sections = [f"ORIGINAL TASK:\n{original_task}\n"]
        
        # Add brain outputs
        sections.append("BRAIN OUTPUTS:")
        for brain in brain_outputs:
            if brain.status == BrainStatus.OK and brain.output:
                sections.append(f"\n{brain.name.upper()}:\n{brain.output}")
            elif brain.status == BrainStatus.ERROR:
                sections.append(f"\n{brain.name.upper()}: ERROR - {brain.error}")
        
        # Add critique
        if critique_result.output:
            sections.append(f"\nCRITIQUE/ROAST:\n{critique_result.output}")
        
        return "\n".join(sections)
    
    def _parse_synthesis_response(self, synthesis_response: str, brain_outputs: List[BrainOutput]) -> SynthesisResult:
        """Parse the LLM synthesis response into structured format."""
        lines = synthesis_response.split('\n')
        
        summary_parts = []
        rationale_parts = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SUMMARY:'):
                current_section = 'summary'
                summary_parts.append(line[8:].strip())
            elif line.upper().startswith('RATIONALE:'):
                current_section = 'rationale'
                rationale_parts.append(line[10:].strip())
            elif current_section == 'summary' and line:
                summary_parts.append(line)
            elif current_section == 'rationale' and line:
                rationale_parts.append(line)
        
        summary = ' '.join(summary_parts) if summary_parts else synthesis_response[:500]
        rationale = ' '.join(rationale_parts) if rationale_parts else "Synthesis completed based on available inputs."
        
        # Simple rule-based discarded ideas identification
        discarded_ideas = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": f"Brain failed to respond: {brain.error}"
                })
            elif not brain.output or len(brain.output.strip()) < 10:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": "Response too short or empty to be useful"
                })
        
        return SynthesisResult(
            summary=summary,
            rationale=rationale,
            discarded_ideas=discarded_ideas
        )
    
    def _rule_based_synthesis(self, brain_outputs: List[BrainOutput], critique_result: CritiqueResult) -> SynthesisResult:
        """Perform basic rule-based synthesis when LLM is unavailable."""
        # Simple heuristic synthesis
        valid_brains = [b for b in brain_outputs if b.status == BrainStatus.OK and b.output and len(b.output.strip()) >= 10]
        
        if not valid_brains:
            discarded_ideas = []
            for brain in brain_outputs:
                if brain.status == BrainStatus.ERROR:
                    discarded_ideas.append({
                        "source_brain": brain.name,
                        "reason": f"Brain failed to respond: {brain.error}"
                    })
                else:
                    discarded_ideas.append({
                        "source_brain": brain.name,
                        "reason": "Response too short or empty to be useful"
                    })
            
            return SynthesisResult(
                summary="No valid brain responses available for synthesis.",
                rationale="All brains either failed or provided empty responses.",
                discarded_ideas=discarded_ideas
            )
        
        # Combine responses from successful brains
        combined_summary = "BASED ON COUNCIL ANALYSIS:\n\n"
        
        for brain in valid_brains:
            combined_summary += f"{brain.name.upper()} PERSPECTIVE:\n{brain.output[:300]}...\n\n"
        
        rationale = f"Synthesized from {len(valid_brains)} successful brain responses."
        
        discarded_ideas = []
        for brain in brain_outputs:
            if brain.status == BrainStatus.ERROR:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": f"Failed to respond: {brain.error}"
                })
            elif brain not in valid_brains:
                discarded_ideas.append({
                    "source_brain": brain.name,
                    "reason": "Response too short or empty to be useful"
                })
        
        return SynthesisResult(
            summary=combined_summary,
            rationale=rationale,
            discarded_ideas=discarded_ideas
        )
    
    def _create_error_result(self, task: str, error: str) -> Dict[str, Any]:
        """Create an error result when the entire council fails."""
        return {
            "task": task,
            "brains_used": [],
            "critique": {
                "status": "error",
                "output": "",
                "error": f"Council orchestration failed: {error}"
            },
            "final_answer": {
                "summary": "Council orchestration failed",
                "rationale": f"Critical error during execution: {error}",
                "discarded_ideas": []
            }
        }