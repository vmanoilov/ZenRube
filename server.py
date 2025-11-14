# server.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ZenRube Router",
    description="Multi-agent sequencing and routing engine",
    version="1.0.0"
)

# ----- Request / Response models -----
class RouteRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    sequence: list[str]

# ----- Core Logic (simple deterministic ZenRube sequence) -----
def compute_sequence(user_prompt: str) -> list[str]:
    """
    ZenRube sequence engine.
    You can upgrade this later with LLM logic,
    but for now we return deterministic persona chain.
    """

    # Default 4-step debate chain:
    return [
        "strategist",
        "devils_advocate",
        "creative_innovator",
        "systems_realist",
        "synthesizer",
        "validator"
    ]


# ----- API ENDPOINTS -----

@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    """Return ZenRube’s persona execution order."""
    sequence = compute_sequence(req.prompt)
    return RouteResponse(sequence=sequence)


@app.get("/")
async def root():
    return {"status": "ZenRube online", "version": "1.0.0"}
