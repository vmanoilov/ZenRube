# zenrube-mcp

Zenrube is a Zen-inspired consensus engine that orchestrates multiple LLM experts in parallel, synthesising their perspectives into a single actionable recommendation. It is designed for the Rube automation platform but is provider-agnostic and can be embedded into any Python project.

## ✨ Features

- 🔀 Parallel or sequential expert orchestration with execution tracing
- 🧠 Three synthesis styles: `balanced`, `critical`, and `collaborative`
- 🪪 Configurable expert personas with custom registrations
- 🗂️ YAML configuration discovery (`.zenrube.yml` in project or home directory)
- 🧱 Structured Pydantic models for responses and configuration
- 💾 TTL-aware caching (in-memory, file-system, or Redis)
- 🔌 Pluggable LLM providers (Rube, OpenAI, or custom implementations)
- 🧪 Comprehensive unit test suite and CI pipeline

## 🚀 Quick Start

### Installation

```bash
pip install zenrube-mcp
```

or for local development:

```bash
git clone https://github.com/vmanoilov/zenrube-mcp.git
cd zenrube-mcp
pip install -e .[dev]
```

### CLI Usage

```bash
zenrube "Should we adopt an event-driven architecture?" \
  --style balanced \
  --experts pragmatic_engineer systems_architect security_analyst \
  --provider rube \
  --model gpt-4o-mini
```

Flags:

- `--style`: synthesis tone (`balanced`, `critical`, `collaborative`)
- `--experts`: one or more registered expert slugs
- `--sequential`: force sequential execution
- `--provider`: provider registry key to use
- `--model`: model identifier passed to the provider
- `--debug`: enable verbose logging
- `--no-cache`: bypass the caching layer for a single run

### Python API

```python
from zenrube import zen_consensus

result = zen_consensus(
    "Should we use microservices?",
    experts=["pragmatic_engineer", "systems_architect"],
    synthesis_style="critical",
    provider="rube",
    model="claude-3-sonnet",
)

print(result["consensus"])
```

The returned dictionary conforms to `models.ConsensusResult` and includes an `execution_id`, expert responses, synthesis text, and metadata about degraded states.

## ⚙️ Configuration

Zenrube loads configuration from `.zenrube.yml` in the project root and the user's home directory. A minimal example:

```yaml
experts:
  - pragmatic_engineer
  - systems_architect
  - security_analyst
synthesis_style: balanced
parallel_execution: true
provider: rube
logging:
  level: INFO
  debug: false
cache:
  backend: memory
  ttl: 300
custom_experts:
  product_manager:
    name: Product Manager
    system_prompt: |
      You are a product manager balancing user value, delivery speed, and stakeholder impact.
```

Custom experts registered through config become available to the CLI and Python API immediately.

## 🔌 Providers

Providers implement the `providers.LLMProvider` interface and are registered with the `ProviderRegistry`. Built-in providers include:

- `RubeProvider`: delegates to Rube's `invoke_llm`
- `OpenAIProvider`: thin wrapper around the OpenAI Chat Completions API

Register your own provider:

```python
from typing import Optional

from providers import LLMProvider, ProviderRegistry

class MockProvider(LLMProvider):
    name = "mock"

    def query(self, prompt: str, *, model: Optional[str] = None, **kwargs):
        return "Mock response", {"model": model}

ProviderRegistry.register(MockProvider())
ProviderRegistry.set_default("mock")
```

To plug in Rube's native helper, call:

```python
from zenrube import configure_rube_client
from rube import invoke_llm as rube_invoke

configure_rube_client(rube_invoke)
```

## 💾 Caching

The caching layer defaults to in-memory storage. Configure file or Redis backends in `.zenrube.yml`:

```yaml
cache:
  backend: file
  directory: .zenrube-cache
  ttl: 600
```

Set `backend: redis` and a `url` to use Redis. You can also disable caching per request with the CLI `--no-cache` flag or the API `use_cache=False` argument.

## 🧪 Testing

```bash
pip install -e .[dev]
pytest --cov=src
```

Continuous integration runs formatting (`black`), linting (`flake8`), typing (`mypy`), and coverage on Python 3.8–3.12.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache License 2.0

## 🙏 Acknowledgements

- Concept by [@vmanoilov](https://github.com/vmanoilov)
- Inspired by [Zen MCP](https://github.com/BeehiveInnovations/zen-mcp-server)
