# zenrube-mcp

Zenrube is a sophisticated expert-based automation system that provides specialized components for intelligent data processing, routing, and content generation. Built for the Rube automation platform, it features a modular expert architecture with configurable routing and caching capabilities.

## ✨ Features

- 🧠 **Expert System Architecture**: Specialized expert modules for different tasks
- 🔀 **Semantic Routing**: Intelligent text analysis and intent detection
- 🧹 **Data Processing**: Automated data cleaning and preprocessing
- 📝 **Content Generation**: AI-powered summarization and content creation
- 📤 **Publishing Pipeline**: Structured content publishing and distribution
- 🗂️ **YAML Configuration**: Flexible configuration management (`.zenrube.yml`)
- 💾 **Caching System**: TTL-aware caching with multiple backend support
- 🔌 **Extensible Design**: Plugin-based architecture for custom experts
- 🧪 **Comprehensive Testing**: Full test suite with CI pipeline

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

### Available Experts

The system currently includes four specialized experts:

#### Semantic Router Expert
Analyzes input text to infer intent and route data to appropriate handlers.

```python
from zenrube.experts import SemanticRouterExpert

router = SemanticRouterExpert()
result = router.run("Error: Database connection failed")
# Returns: {"input": "...", "intent": "error", "route": "debug_expert"}
```

#### Data Cleaner Expert
Processes and cleans data for consistent formatting and analysis.

```python
from zenrube.experts import DataCleanerExpert

cleaner = DataCleanerExpert()
result = cleaner.run(raw_data)
# Returns cleaned and structured data
```

#### Summarizer Expert
Generates concise summaries of text content using AI.

```python
from zenrube.experts import SummarizerExpert

summarizer = SummarizerExpert()
result = summarizer.run(long_text_content)
# Returns summary and metadata
```

#### Publisher Expert
Handles content formatting and publishing workflows.

```python
from zenrube.experts import PublisherExpert

publisher = PublisherExpert()
result = publisher.run(formatted_content)
# Returns published content with distribution metadata
```

### Expert Registry

All experts are registered and discoverable through the expert registry:

```python
from zenrube.experts_module import list_experts, get_expert

# List all available experts
experts = list_experts()
print(experts)  # ['semantic_router', 'data_cleaner', 'summarizer', 'publisher']

# Get specific expert
expert = get_expert('semantic_router')
```

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
  ttl: 120
```

## 🔌 Architecture

The system is built on a modular expert architecture:

### Expert Registry
All experts are registered and managed through the `ExpertRegistry`:

```python
from zenrube.experts_module import ExpertRegistry, ExpertDefinition

# Register a custom expert
custom_expert = ExpertDefinition(
    slug="custom_processor",
    name="Custom Processor",
    description="Processes custom data types",
    handler=custom_handler_function
)

ExpertRegistry.register(custom_expert)
```

### Provider System
The system includes a provider architecture for LLM integration:

```python
from zenrube.providers import ProviderRegistry

# Configure Rube provider
from zenrube import configure_rube_client
from rube import invoke_llm as rube_invoke
configure_rube_client(rube_invoke)
```

## 💾 Caching

The caching system supports multiple backends configured in `.zenrube.yml`:

```yaml
cache:
  backend: memory    # or 'file', 'redis'
  directory: .zenrube-cache
  ttl: 120
```

Current backends:
- **memory**: In-memory caching (default)
- **file**: File-system based caching
- **redis**: Redis-based distributed caching

## 🧪 Testing

Run the comprehensive test suite:

```bash
pip install -e .[dev]
pytest --cov=src
```

Test coverage includes:
- Expert functionality and edge cases
- Configuration loading and validation
- Caching mechanisms
- Provider integration
- Error handling and degraded states

Continuous integration runs formatting (`black`), linting (`flake8`), typing (`mypy`), and coverage on Python 3.8–3.12.

## 📁 Project Structure

```
zenrube-mcp/
├── src/zenrube/
│   ├── experts/          # Core expert implementations
│   │   ├── semantic_router.py
│   │   ├── data_cleaner.py
│   │   ├── summarizer.py
│   │   └── publisher.py
│   ├── experts_module.py # Expert registry and definitions
│   ├── config.py        # Configuration management
│   ├── cache.py         # Caching layer
│   ├── providers.py     # LLM provider interfaces
│   └── models.py        # Data models
├── tests/               # Comprehensive test suite
├── examples/            # Usage examples and demos
├── .zenrube.yml        # Default configuration
└── pyproject.toml      # Project metadata and dependencies
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and contribution workflows.

### Adding New Experts

1. Create a new expert class inheriting from the base expert interface
2. Add metadata and configuration
3. Register the expert in the expert module
4. Add comprehensive tests
5. Update documentation

## 📄 License

Apache License 2.0

## 🙏 Acknowledgements

- Concept by [@vmanoilov](https://github.com/vmanoilov)
- Built for the Rube automation platform
- Inspired by modular AI architectures and expert systems
