"""Core consensus orchestration for zenrube-mcp."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cache import CacheManager, build_cache_key
from .config import build_synthesis_config, load_config
from .experts import ExpertDefinition, get_expert, list_experts
from .models import (
    ConsensusResult,
    ExpertResponse,
    SynthesisConfig,
    SYNTHESIS_STYLES,
)
from .providers import ProviderRegistry, RubeProvider

DEGRADED_WARNING = "One or more experts failed; consensus may be degraded."

LOGGER = logging.getLogger("zenrube")

__all__ = [
    "CacheManager",
    "ConsensusResult",
    "ExpertDefinition",
    "ExpertResponse",
    "ProviderRegistry",
    "RubeProvider",
    "SYNTHESIS_STYLES",
    "build_cache_key",
    "build_synthesis_config",
    "configure_logging",
    "configure_rube_client",
    "invoke_llm",
    "list_experts",
    "load_config",
    "zen_consensus",
    "main",
]


def configure_logging(level: str = "INFO", debug: bool = False) -> None:
    if debug:
        level = "DEBUG"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def invoke_llm(
    prompt: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Proxy call to configured LLM provider.

    This function is intentionally simple so it can be patched in tests or
    overridden in host environments.
    """

    provider_instance = ProviderRegistry.get(provider)
    LOGGER.debug("Invoking provider %s", provider_instance.name)
    return provider_instance.query(prompt, model=model, **kwargs)


def configure_rube_client(client: Any) -> None:
    """Attach a callable compatible with Rube's invoke_llm helper."""

    provider = ProviderRegistry.get("rube")
    if isinstance(provider, RubeProvider):
        provider.set_client(client)
    else:  # pragma: no cover - defensive branch
        replacement = RubeProvider(client=client)
        ProviderRegistry.register(replacement)
        ProviderRegistry.set_default("rube")


def _build_prompt(expert: ExpertDefinition, question: str) -> str:
    return expert.build_prompt(question)


def _build_synthesis_prompt(
    style: str, question: str, responses: Iterable[ExpertResponse]
) -> str:
    payload: List[Dict[str, Any]] = []
    for response in responses:
        if response.response:
            payload.append(response.model_dump())
    instructions = {
        "balanced": (
            "Provide a balanced synthesis highlighting agreements and "
            "practical next steps."
        ),
        "critical": (
            "Provide a critical synthesis emphasising risks, failure modes, "
            "and mitigations."
        ),
        "collaborative": (
            "Provide a collaborative synthesis identifying synergies and "
            "phased collaboration steps."
        ),
    }
    style_instruction = instructions.get(style, instructions["balanced"])
    return (
        f"You are orchestrating a panel of experts. Question: {question}\n"
        f"Expert analyses: {json.dumps(payload, indent=2)}\n"
        "Deliver a consensus report with sections: Areas of Agreement, "
        "Points of Divergence, Recommendation, Confidence (LOW/MEDIUM/HIGH).\n"
        f"Tone guidance: {style_instruction}"
    )


def _query_expert(
    expert_slug: str,
    question: str,
    config: SynthesisConfig,
    execution_id: str,
    model: Optional[str] = None,
) -> ExpertResponse:
    expert = get_expert(expert_slug)
    prompt = _build_prompt(expert, question)
    start = time.time()
    LOGGER.debug("[%s] Querying expert %s", execution_id, expert_slug)
    try:
        response, metadata = invoke_llm(
            prompt,
            model=model,
            provider=config.provider,
        )
        duration = time.time() - start
        LOGGER.debug(
            "[%s] Expert %s responded in %.2fs",
            execution_id,
            expert_slug,
            duration,
        )
        return ExpertResponse(
            name=expert.name,
            prompt=prompt,
            response=response,
            provider=config.provider,
            duration_seconds=duration,
            metadata=metadata or {},
        )
    except Exception as exc:  # pragma: no cover - requires failure case
        duration = time.time() - start
        LOGGER.error(
            "[%s] Expert %s failed: %s",
            execution_id,
            expert_slug,
            exc,
        )
        return ExpertResponse(
            name=expert.name,
            prompt=prompt,
            error=str(exc),
            provider=config.provider,
            duration_seconds=duration,
        )


def _synthesise(
    question: str,
    responses: List[ExpertResponse],
    config: SynthesisConfig,
    execution_id: str,
    model: Optional[str] = None,
) -> Optional[str]:
    successful = [response for response in responses if response.response]
    if not successful:
        LOGGER.warning(
            "[%s] No successful expert responses available for synthesis",
            execution_id,
        )
        return None
    prompt = _build_synthesis_prompt(
        config.synthesis_style,
        question,
        successful,
    )
    try:
        synthesis, _ = invoke_llm(
            prompt,
            model=model,
            provider=config.provider,
        )
        return synthesis
    except Exception as exc:  # pragma: no cover
        LOGGER.error(
            "[%s] Synthesis failed: %s",
            execution_id,
            exc,
        )
        return None


def zen_consensus(
    question: str,
    *,
    experts: Optional[List[str]] = None,
    synthesis_style: str = "balanced",
    parallel: Optional[bool] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_workers: Optional[int] = None,
    cache_ttl: Optional[int] = None,
    use_cache: bool = True,
    debug: Optional[bool] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    execution_id = str(uuid.uuid4())
    base_config = load_config()
    CacheManager.from_config(base_config.get("cache", {}))

    config_overrides: Dict[str, Any] = overrides.copy() if overrides else {}
    if experts is not None:
        config_overrides["experts"] = experts
    if synthesis_style:
        config_overrides["synthesis_style"] = synthesis_style
    if parallel is not None:
        config_overrides["parallel_execution"] = parallel
    if provider:
        config_overrides["provider"] = provider
    if max_workers is not None:
        config_overrides["max_workers"] = max_workers
    if cache_ttl is not None:
        config_overrides["cache_ttl_seconds"] = cache_ttl
    if debug is not None:
        config_overrides["debug"] = debug

    synthesis_config = build_synthesis_config(config_overrides)
    configure_logging(synthesis_config.logging_level, synthesis_config.debug)
    LOGGER.info("[%s] Starting consensus run", execution_id)

    selected_experts = synthesis_config.experts or list(list_experts())
    LOGGER.debug("[%s] Experts selected: %s", execution_id, selected_experts)

    cache_key = build_cache_key(
        execution_id if not use_cache else "zenrube",
        synthesis_config.provider,
        synthesis_config.synthesis_style,
        question,
        "|".join(selected_experts),
    )

    if use_cache:
        cached = CacheManager.get(cache_key)
        if cached:
            LOGGER.info("[%s] Returning cached consensus", execution_id)
            return cached

    responses: List[ExpertResponse] = []
    if synthesis_config.parallel_execution and len(selected_experts) > 1:
        workers = synthesis_config.max_workers or min(len(selected_experts), 8)
        LOGGER.debug(
            "[%s] Executing in parallel with %s workers", execution_id, workers
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _query_expert,
                    expert_slug,
                    question,
                    synthesis_config,
                    execution_id,
                    model,
                ): expert_slug
                for expert_slug in selected_experts
            }
            for future in as_completed(future_map):
                responses.append(future.result())
    else:
        LOGGER.debug("[%s] Executing sequentially", execution_id)
        for expert_slug in selected_experts:
            responses.append(
                _query_expert(
                    expert_slug,
                    question,
                    synthesis_config,
                    execution_id,
                    model,
                )
            )

    consensus_text = _synthesise(
        question,
        responses,
        synthesis_config,
        execution_id,
        model,
    )
    success_flags: List[bool] = []
    for response in responses:
        success_flags.append(response.succeeded)
    degraded = not all(success_flags)
    warnings: List[str] = []
    if degraded:
        warnings.append(DEGRADED_WARNING)
    if consensus_text is None:
        warnings.append("Consensus synthesis unavailable.")

    experts_consulted: List[str] = []
    for response in responses:
        experts_consulted.append(response.name)
    metadata = {
        "parallel_execution": synthesis_config.parallel_execution,
    }
    result_model = ConsensusResult(
        execution_id=execution_id,
        question=question,
        synthesis_style=synthesis_config.synthesis_style,
        provider=synthesis_config.provider,
        experts_consulted=experts_consulted,
        responses=responses,
        consensus=consensus_text,
        timestamp=datetime.now(timezone.utc),
        degraded=degraded,
        warnings=warnings,
        metadata=metadata,
    )
    result = result_model.model_dump()

    if use_cache:
        CacheManager.set(
            cache_key,
            result,
            synthesis_config.cache_ttl_seconds,
        )
    LOGGER.info("[%s] Consensus run complete", execution_id)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Zenrube MCP consensus CLI")
    parser.add_argument("question", help="Question to analyse")
    parser.add_argument(
        "--style",
        choices=SYNTHESIS_STYLES,
        default="balanced",
        help="Synthesis style",
    )
    parser.add_argument(
        "--experts",
        nargs="*",
        help="Specific expert slugs to consult",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Force sequential execution"
    )
    parser.add_argument(
        "--provider",
        help="Provider to use",
        default=None,
    )
    parser.add_argument("--model", help="Model to use", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching for this run"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = zen_consensus(
        args.question,
        experts=args.experts,
        synthesis_style=args.style,
        parallel=not args.sequential,
        provider=args.provider,
        model=args.model,
        debug=args.debug,
        use_cache=not args.no_cache,
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
