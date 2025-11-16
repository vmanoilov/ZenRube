from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict

from zenrube.experts.semantic_router import SemanticRouterExpert
from zenrube.experts.expert_registry import ExpertRegistry
from zenrube.experts_module import get_expert
from zenrube.embeddings.client import embed_text
from zenrube.embeddings.index import add_items, search

app = FastAPI(title="ZenRube API", version="1.0.0")

semantic_router = SemanticRouterExpert()
expert_registry = ExpertRegistry()

class RouteRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    routes: List[str]
    raw: Any

class RunRequest(BaseModel):
    expert: str
    prompt: str

class RunResponse(BaseModel):
    result: str

class EmbedRequest(BaseModel):
    text: str
    namespace: Optional[str] = None
    store: Optional[bool] = True

class EmbedResponse(BaseModel):
    vector: List[float]
    id: Optional[str] = None
    namespace: str

class EmbedSearchRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    top_k: Optional[int] = 5

class EmbedSearchResponse(BaseModel):
    query_vector: List[float]
    results: List[Dict[str, Any]]

@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    try:
        result = semantic_router.run(req.prompt)
        routes = []

        if isinstance(result, dict):
            if "route" in result:
                routes = result["route"] if isinstance(result["route"], list) else [result["route"]]
            elif "expert" in result:
                routes = result["expert"] if isinstance(result["expert"], list) else [result["expert"]]

        if not routes:
            routes = ["general_handler"]

        return RouteResponse(routes=routes, raw=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    try:
        try:
            expert = expert_registry.load_expert(req.expert)
            output = expert.run(req.prompt)
            return RunResponse(result=str(output))
        except Exception:
            expert_def = get_expert(req.expert)
            output = expert_def.build_prompt(req.prompt)
            return RunResponse(result=str(output))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experts")
def list_experts():
    try:
        reg = list(expert_registry.list_available_experts())
        from zenrube.experts_module import list_experts as core_list
        core = list(core_list())
        return sorted(set(reg + core))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed", response_model=EmbedResponse)
def embed_text_endpoint(req: EmbedRequest):
    try:
        # Generate embedding for single text
        vector = embed_text(req.text)

        item_id = None
        if req.store:
            # Create index item
            items = [{
                "text": req.text,
                "vector": vector,
                "namespace": req.namespace or "default",
                "metadata": {}
            }]

            # Add to index
            item_ids = add_items(items)
            item_id = item_ids[0] if item_ids else None

        return EmbedResponse(
            vector=vector,
            id=item_id,
            namespace=req.namespace or "default"
        )
    except RuntimeError as e:
        if "disabled" in str(e).lower():
            raise HTTPException(status_code=503, detail="Embeddings are disabled - no valid configuration found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/search", response_model=EmbedSearchResponse)
def embed_search_endpoint(req: EmbedSearchRequest):
    try:
        # Embed the query
        query_vector = embed_text(req.query)

        # Search the index
        results = search(query_vector, namespace=req.namespace, top_k=req.top_k)

        return EmbedSearchResponse(
            query_vector=query_vector,
            results=results
        )
    except RuntimeError as e:
        if "disabled" in str(e).lower():
            raise HTTPException(status_code=503, detail="Embeddings are disabled - no valid configuration found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
