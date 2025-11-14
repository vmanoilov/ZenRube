from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RouteInput(BaseModel):
    prompt: str

@app.post("/route")
def route(data: RouteInput):
    return {
        "action": "CALL_PROVIDER",
        "provider": "groq",
        "model": "llama3-70b",
        "temperature": 0.3,
        "maxTokens": 2048,
        "payload": {
            "messages": [{"role": "user", "content": data.prompt}]
        },
        "fallback": ["deepseek", "gemini", "together", "openai"]
    }

@app.post("/validate")
def validate(body: dict):
    return {"final": True, "output": body["output"]}
