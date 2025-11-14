# server.py
from mcp.server import FastMCP
from pydantic import BaseModel
import importlib

# Initialize MCP server
server = FastMCP(name="zenrube")

# Load actual ZenRube modules
from zenrube.experts.semantic_router import SemanticRouterExpert
from zenrube.experts_module import get_expert
from zenrube.experts.expert_registry import ExpertRegistry

# Initialize experts
semantic_router = SemanticRouterExpert()
expert_registry = ExpertRegistry()

@server.tool()
def route(prompt: str) -> list[str]:
    """Return the expert sequence from the real ZenRube semantic router."""
    try:
        result = semantic_router.run(prompt)
        # Extract route from the result
        if isinstance(result, dict):
            return [result.get("route", "general_handler")]
        else:
            return ["general_handler"]
    except Exception as e:
        return [f"Error routing prompt: {str(e)}"]

@server.tool()
def run(expert: str, prompt: str) -> str:
    """Execute a single ZenRube expert."""
    try:
        # First try the expert registry system
        try:
            expert_instance = expert_registry.load_expert(expert)
            result = expert_instance.run(prompt)
            if isinstance(result, dict):
                return str(result)
            else:
                return str(result)
        except ModuleNotFoundError:
            # Fall back to the experts_module system
            expert_definition = get_expert(expert)
            if hasattr(expert_definition, 'build_prompt') and callable(expert_definition.build_prompt):
                return expert_definition.build_prompt(prompt)
            else:
                return f"Error: Expert '{expert}' found but build_prompt method not available."
    except KeyError:
        return f"Error: Expert '{expert}' not found."
    except Exception as e:
        return f"Error executing expert '{expert}': {str(e)}"

@server.tool()
def list_experts() -> list[str]:
    """List all available experts."""
    try:
        registry_experts = list(expert_registry.list_available_experts())
        from zenrube.experts_module import list_experts as list_core_experts
        core_experts = list(list_core_experts())
        return registry_experts + core_experts
    except Exception as e:
        return [f"Error listing experts: {str(e)}"]

# export app for FastMCP
app = server.streamable_http_app
