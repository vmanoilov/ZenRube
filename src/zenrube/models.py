"""Pydantic data models for zenrube-mcp."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


SYNTHESIS_STYLES = ("balanced", "critical", "collaborative")


class ExpertResponse(BaseModel):
    """Structured representation of an expert's reply."""

    name: str
    prompt: str
    response: Optional[str] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.response is not None


class SynthesisConfig(BaseModel):
    """Configuration used for consensus synthesis."""

    synthesis_style: Literal[
        "balanced",
        "critical",
        "collaborative",
    ] = "balanced"
    parallel_execution: bool = True
    provider: str = "rube"
    experts: List[str] = Field(default_factory=list)
    max_workers: Optional[int] = Field(default=None, ge=1)
    cache_ttl_seconds: Optional[int] = Field(default=None, ge=0)
    logging_level: str = "INFO"
    debug: bool = False

    @field_validator("logging_level")
    @classmethod
    def _validate_logging_level(cls, value: str) -> str:
        return value.upper()


class ConsensusResult(BaseModel):
    """Final structured response returned to callers."""

    execution_id: str
    question: str
    synthesis_style: str
    provider: str
    experts_consulted: List[str]
    responses: List[ExpertResponse]
    consensus: Optional[str]
    timestamp: datetime
    degraded: bool = False
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:  # type: ignore[override]
        """Ensure datetime serialization uses ISO format."""

        serialised = super().model_dump(*args, **kwargs)
        serialised["timestamp"] = self.timestamp.isoformat()
        response_payloads = [resp.model_dump() for resp in self.responses]
        serialised["responses"] = response_payloads
        return serialised
