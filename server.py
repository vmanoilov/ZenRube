# server.py
"""
FastMCP Server for ZenRube Expert System

This server provides MCP-compatible tools for:
- Semantic routing of prompts to appropriate experts
- Execution of individual ZenRube experts
- Listing all available experts

The server is compatible with FastMCP and provides both ExpertRegistry
and core expert system support with intelligent fallbacks.
"""

from mcp.server import FastMCP
from zenrube.experts.semantic_router import SemanticRouterExpert
from zenrube.experts_module import get_expert
from zenrube.experts.expert_registry import ExpertRegistry

# Initialize MCP server
server = FastMCP(name="zenrube")

# Initialize experts
semantic_router = SemanticRouterExpert()
expert_registry = ExpertRegistry()

@server.tool()
def route(prompt: str) -> list[str]:
    """
    Route a prompt to the appropriate expert using semantic analysis.
    
    Uses the SemanticRouterExpert to analyze input and determine
    the best expert routing target.
    
    Args:
        prompt: Input text to analyze and route
        
    Returns:
        List containing the routing decision or error message
    """
    try:
        result = semantic_router.run(prompt)
        if isinstance(result, dict):
            route_target = result.get("route", "general_handler")
            return [route_target]
        else:
            return ["general_handler"]
    except Exception as e:
        return [f"Error routing prompt: {str(e)}"]

@server.tool()
def run(expert: str, prompt: str) -> str:
    """
    Execute a single ZenRube expert with the provided prompt.
    
    Attempts to use ExpertRegistry first, then falls back to
    the core experts_module system if needed.
    
    Args:
        expert: Name of the expert to execute
        prompt: Input prompt for the expert
        
    Returns:
        Expert execution result or error message
    """
    try:
        # Try ExpertRegistry system first
        try:
            expert_instance = expert_registry.load_expert(expert)
            result = expert_instance.run(prompt)
            return str(result) if result else "No result returned"
        except ModuleNotFoundError:
            # Fall back to core experts_module system
            expert_definition = get_expert(expert)
            if hasattr(expert_definition, 'build_prompt') and callable(expert_definition.build_prompt):
                return expert_definition.build_prompt(prompt)
            else:
                return f"Error: Expert '{expert}' found but build_prompt method not available."
    except KeyError:
        return f"Error: Expert '{expert}' not found. Use list_experts() to see available options."
    except Exception as e:
        return f"Error executing expert '{expert}': {str(e)}"

@server.tool()
def list_experts() -> list[str]:
    """
    List all available experts from both ExpertRegistry and core systems.
    
    Returns:
        Combined list of all available expert names
    """
    try:
        registry_experts = list(expert_registry.list_available_experts())
        from zenrube.experts_module import list_experts as list_core_experts
        core_experts = list(list_core_experts())
        all_experts = sorted(list(set(registry_experts + core_experts)))
        return all_experts if all_experts else ["No experts available"]
    except Exception as e:
        return [f"Error listing experts: {str(e)}"]

# Export FastMCP-compatible app
app = server.streamable_http_app
