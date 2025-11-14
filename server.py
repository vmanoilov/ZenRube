# server.py
from mcp import Server
from pydantic import BaseModel
import importlib

# Initialize MCP server
server = Server(name="zenrube")

# Load actual ZenRube modules
semantic_router = importlib.import_module("src.zenrube.experts.semantic_router")
expert_registry = importlib.import_module("src.zenrube.experts.expert_registry")

@server.tool()
def route(prompt: str) -> list[str]:
    """Return the expert sequence from the real ZenRube semantic router."""
    result = semantic_router.route_prompt(prompt)
    return result if isinstance(result, list) else [result]

@server.tool()
def run(expert: str, prompt: str) -> str:
    """Execute a single ZenRube expert."""
    expert_obj = expert_registry.get_expert(expert)
    if expert_obj is None:
        return f"Error: Expert '{expert}' not found."
    return expert_obj.run(prompt)

# export app for FastMCP
app = server.app
