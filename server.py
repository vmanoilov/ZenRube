# server.py
import importlib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ---- Load ZenRube modules ----------------------------------------------------

# semantic router
semantic_router = importlib.import_module("src.zenrube.experts.semantic_router")

# expert registry
expert_registry = importlib.import_module("src.zenrube.experts.expert_registry")

# summarizer
summarizer = importlib.import_module("src.zenrube.experts.summarizer")

# -----------------------------------------------------------------------------


# ---- FastAPI app -------------------------------------------------------------
app = FastAPI(
    title="ZenRube MCP Server",
    description="Semantic routing and expert execution engine for ZenRube",
    version="1.0.0"
)


# ---- Models -----------------------------------------------------------------
class RouteRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    experts: List[str]

class RunRequest(BaseModel):
    prompt: str
    expert: str

class RunResponse(BaseModel):
    output: str


# ---- Core Routing Logic ------------------------------------------------------

def route_prompt(prompt: str) -> List[str]:
    """
    Uses ZenRube’s REAL semantic router to determine which experts
    should handle the incoming request.
    """

    # call your internal router
    result = semantic_router.route_prompt(prompt)

    # 'result' can be a single expert or list depending on your router design
    if isinstance(result, str):
        return [result]

    return result


def run_expert(expert_name: str, prompt: str) -> str:
    """
    Executes the expert from the registry.
    """

    expert = expert_registry.get_expert(expert_name)

    if expert is None:
        return f"[Error] Expert '{expert_name}' not found."

    # Safely execute the expert
    try:
        output = expert.run(prompt)
    except Exception as e:
        output = f"[Expert Error] {str(e)}"

    return output


# ---- API Endpoints -----------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    """Return the sequence of experts selected by the semantic router."""
    experts = route_prompt(req.prompt)
    return RouteResponse(experts=experts)


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest):
    """Execute a specific expert and return output."""
    result = run_expert(req.expert, req.prompt)
    return RunResponse(output=result)


@app.get("/")
async def root():
    return {
        "status": "ZenRube online",
        "version": "1.0.0",
        "available_modules": [
            "expert_registry",
            "semantic_router",
            "summarizer",
            "data_cleaner",
            "publisher",
            "autopublisher",
            "version_manager",
            "rube_adapter",
        ],
        "endpoints": ["/route", "/run"]
    }
