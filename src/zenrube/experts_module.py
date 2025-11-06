"""Expert persona management for zenrube-mcp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


DEFAULT_PROMPT_TEMPLATE = "{system_prompt}\n\nQuestion: {question}"


@dataclass
class ExpertDefinition:
    """Definition of an expert persona."""

    name: str
    system_prompt: str
    description: Optional[str] = None
    prompt_template: Optional[str] = None

    def build_prompt(self, question: str) -> str:
        template = self.prompt_template or DEFAULT_PROMPT_TEMPLATE
        return template.format(
            system_prompt=self.system_prompt.strip(),
            question=question.strip(),
        )


_REGISTERED_EXPERTS: Dict[str, ExpertDefinition] = {}


def register_custom_expert(slug: str, definition: ExpertDefinition) -> None:
    _REGISTERED_EXPERTS[slug] = definition


def register_many(definitions: Mapping[str, ExpertDefinition]) -> None:
    for slug, definition in definitions.items():
        register_custom_expert(slug, definition)


def get_expert(slug: str) -> ExpertDefinition:
    if slug not in _REGISTERED_EXPERTS:
        raise KeyError(f"Unknown expert '{slug}'")
    return _REGISTERED_EXPERTS[slug]


def list_experts() -> Iterable[str]:
    return _REGISTERED_EXPERTS.keys()


DEFAULT_EXPERTS: Dict[str, ExpertDefinition] = {
    "pragmatic_engineer": ExpertDefinition(
        name="Pragmatic Engineer",
        system_prompt=(
            "You are a pragmatic engineer evaluating trade-offs and "
            "implementation details. Focus on practicality, delivery risk, "
            "and incremental rollout strategies."
        ),
    ),
    "systems_architect": ExpertDefinition(
        name="Systems Architect",
        system_prompt=(
            "You are a systems architect analysing scalability, "
            "maintainability, and integration with existing systems. Consider "
            "long-term implications and architectural patterns."
        ),
    ),
    "security_analyst": ExpertDefinition(
        name="Security Analyst",
        system_prompt=(
            "You are a security analyst assessing threats, vulnerabilities, "
            "and compliance impacts. Highlight potential mitigations and "
            "residual risks."
        ),
    ),
}


register_many(DEFAULT_EXPERTS)
