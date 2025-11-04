# Getting Started with zenrube-mcp

## What You Need

1. Access to [Rube platform](https://rube.composio.dev)
2. That's it! No installation needed.

## Step-by-Step Guide

### Step 1: Open Rube Workbench

Go to https://rube.composio.dev and start a new chat.

### Step 2: Copy the Code

The simplest version:

```python
from datetime import datetime
import json

def zen_consensus(question, models=None):
    if models is None:
        models = ["expert_1", "expert_2"]

    responses = []
    for model_name in models:
        prompts = {
            "expert_1": f"As a pragmatic engineer: {question}",
            "expert_2": f"As a systems architect: {question}"
        }
        response, _ = invoke_llm(prompts[model_name])
        responses.append({"model": model_name, "response": response})

    synthesis = f"Synthesize these: {json.dumps(responses)}"
    consensus, _ = invoke_llm(synthesis)

    return {"question": question, "consensus": consensus}

# Use it
result = zen_consensus("PostgreSQL or MongoDB?")
print(result['consensus'])
```

### Step 3: Ask Your Question

Replace `"PostgreSQL or MongoDB?"` with your actual question.

### Step 4: Review the Consensus

The AI will:
1. Consult multiple experts
2. Get different perspectives
3. Synthesize a balanced recommendation
4. Return it to you

## Common Use Cases

### Technical Decisions
```python
zen_consensus("Microservices or monolith for our startup?")
```

### Tool Selection
```python
zen_consensus("Jest or Vitest for testing?")
```

### Architecture Choices
```python
zen_consensus("Event-driven or request-response architecture?")
```

## Troubleshooting

**Q: I get an error about `invoke_llm`**  
A: You must run this inside Rube's REMOTE_WORKBENCH. It won't work in regular Python.

**Q: The responses are too short**  
A: Add more detail to your question or specify what you want analyzed.

**Q: Can I use more than 3 experts?**  
A: Yes! Just add more to the `models` list, but you'll need to define their prompts.

## Next Steps

- Try different questions
- Experiment with synthesis styles
- Integrate with Slack, Google Docs, etc.
- Share your results!
