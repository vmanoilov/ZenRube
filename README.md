# ZenRube

ZenRube is a sophisticated modular cognitive platform that provides expert AI orchestration and self-publishing capabilities to [Rube.app](https://rube.app). Built with a consensus-based architecture, it features specialized expert modules for intelligent data processing, routing, content generation, and automated publishing workflows.

## ✨ Features

- 🧠 **Expert System Architecture**: 11+ specialized expert modules for different tasks
- 🤝 **Consensus-Based AI Orchestration**: Multi-expert consensus system with configurable synthesis styles
- 🔀 **Intelligent Routing**: Advanced semantic analysis and intent detection
- 🧹 **Data Processing**: Automated data cleaning and preprocessing capabilities
- 📝 **Content Generation**: AI-powered summarization and content creation
- 📤 **Publishing Pipeline**: Structured content publishing and distribution
- 🚀 **Auto-Publishing**: Automated expert version detection and marketplace publishing
- 👥 **Team Council**: Multi-brain orchestration with Dynamic Personality System
- �️ **YAML Configuration**: Flexible configuration management (`.zenrube.yml`)
- 💾 **Caching System**: TTL-aware caching with multiple backend support
- 🔌 **Extensible Design**: Plugin-based architecture for custom experts
- 🧪 **Comprehensive Testing**: Full test suite with extensive coverage
- 🎭 **Dynamic Profiles**: Configurable expert personalities and behavior patterns

## 🚀 Quick Start

### Installation

```bash
pip install zenrube
```

or for local development:

```bash
git clone https://github.com/vmanoilov/zenrube.git
cd zenrube
pip install -e .[dev]
```

### Basic Usage

```python
from zenrube import zen_consensus

# Run a consensus across all experts
result = zen_consensus(
    "What are the key considerations for deploying a machine learning model to production?"
)

print(result["consensus"])
```

### Available Experts

The system includes 11 specialized experts:

#### Core Processing Experts

- **Semantic Router**: Analyzes input text to infer intent and route data
- **Data Cleaner**: Processes and cleans data for consistent formatting
- **Summarizer**: Generates concise summaries using AI
- **Publisher**: Handles content formatting and publishing workflows

#### System Integration Experts

- **Rube Adapter**: Integrates with Rube.app platform
- **LLM Connector**: Provides LLM provider abstraction layer
- **Version Manager**: Manages expert versions and updates
- **Auto Publisher**: Automates expert publishing to marketplace

#### Orchestration Experts

- **Team Council**: Multi-brain orchestration with Dynamic Personality System
- **Expert Registry**: Manages and discovers available experts
- **Pragmatic Engineer**: Practical engineering insights
- **Systems Architect**: High-level system architecture guidance
- **Security Analyst**: Security considerations and analysis

### Expert Usage Examples

#### Using Individual Experts

```python
from zenrube.experts import SemanticRouterExpert

router = SemanticRouterExpert()
result = router.run("Error: Database connection failed")
# Returns: {"input": "...", "intent": "error", "route": "debug_expert"}
```

```python
from zenrube.experts import DataCleanerExpert

cleaner = DataCleanerExpert()
result = cleaner.run(raw_data)
# Returns cleaned and structured data
```

#### Expert Registry

```python
from zenrube.experts_module import list_experts, get_expert

# List all available experts
experts = list_experts()
print(experts)  # Returns list of all expert slugs

# Get specific expert
expert = get_expert('semantic_router')
```

## ⚙️ Configuration

ZenRube loads configuration from `.zenrube.yml` in the project root and the user's home directory:

```yaml
experts:
  - pragmatic_engineer
  - systems_architect  
  - security_analyst
synthesis_style: balanced
parallel_execution: true
provider: rube
max_workers: 4
cache_ttl_seconds: 120
logging:
  level: INFO
  debug: false
cache:
  backend: memory
  ttl: 120
```

### Consensus Configuration

```python
from zenrube import zen_consensus, SYNTHESIS_STYLES

# Custom consensus with specific experts and style
result = zen_consensus(
    "How should we approach microservices architecture?",
    experts=["systems_architect", "pragmatic_engineer", "security_analyst"],
    synthesis_style="collaborative",
    parallel=True
)
```

Supported synthesis styles:
- `balanced`: Balanced synthesis highlighting agreements and practical steps
- `critical`: Critical synthesis emphasizing risks and mitigations  
- `collaborative`: Collaborative synthesis identifying synergies

## 🏗️ Architecture

### Expert Registry

All experts are managed through the `ExpertRegistry`:

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

The provider architecture supports multiple LLM backends:

```python
from zenrube.providers import ProviderRegistry

# Configure Rube provider
from zenrube import configure_rube_client
from rube import invoke_llm as rube_invoke
configure_rube_client(rube_invoke)
```

### Team Council Integration

The Team Council expert provides multi-brain orchestration:

```python
from zenrube.experts import TeamCouncil

council = TeamCouncil()
result = council.coordinate_experts(
    question="Comprehensive analysis of cloud migration strategy",
    expert_profiles=["pragmatic_engineer", "systems_architect", "security_analyst"]
)
```

## 💾 Caching

Built-in caching system supports multiple backends:

```yaml
cache:
  backend: memory    # or 'file', 'redis'
  directory: .zenrube-cache
  ttl: 120
```

- **memory**: In-memory caching (default)
- **file**: File-system based caching
- **redis**: Redis-based distributed caching

## 🧪 Testing

Run the comprehensive test suite:

```bash
pip install -e .[dev]
pytest --cov=zenrube
```

Test coverage includes:
- Expert functionality and edge cases
- Configuration loading and validation
- Consensus orchestration
- Caching mechanisms
- Provider integration
- Error handling and degraded states
- Team Council functionality
- Dynamic personality system

Continuous integration runs formatting (`black`), linting (`flake8`), typing (`mypy`), and coverage on Python 3.9–3.12.

## 📁 Project Structure

```
zenrube/
├── zenrube/                    # Core package
│   ├── experts/                # Expert implementations
│   │   ├── semantic_router.py
│   │   ├── data_cleaner.py
│   │   ├── summarizer.py
│   │   ├── publisher.py
│   │   ├── autopublisher.py
│   │   ├── team_council.py
│   │   ├── version_manager.py
│   │   ├── rube_adapter.py
│   │   ├── llm_connector.py
│   │   └── expert_registry.py
│   ├── profiles/               # Dynamic personality system
│   │   ├── personality_engine.py
│   │   ├── dynamic_profile_engine.py
│   │   └── profile_controller.py
│   ├── orchestration/          # Consensus orchestration
│   └── config/                 # Configuration management
├── tests/                      # Comprehensive test suite
├── examples/                   # Usage examples and demos
├── .zenrube.yml               # Default configuration
└── pyproject.toml             # Project metadata and dependencies
```

## 🔄 CLI Usage

ZenRube provides a CLI interface for consensus operations:

```bash
# Basic consensus query
zenrube "What are the security implications of containerization?"

# Custom style and experts
zenrube "How to scale microservices?" --style collaborative --experts systems_architect pragmatic_engineer

# Sequential execution
zenrube "Database migration strategy" --sequential

# Debug mode
zenrube "Performance optimization" --debug
```

## 🚀 Auto-Publishing

The Auto Publisher expert automatically detects version updates and publishes to Rube.app:

```python
from zenrube.experts import AutoPublisherExpert

publisher = AutoPublisherExpert()
result = publisher.auto_publish_experts()
# Detects changes, regenerates manifests, and publishes automatically
```

## 🎭 Dynamic Profiles

ZenRube includes a Dynamic Personality System for configuring expert behavior:

```python
from zenrube.profiles import DynamicProfileEngine

engine = DynamicProfileEngine()
profile = engine.create_profile(
    name="cautious_analyst",
    personality_traits=["risk_aware", "detail_oriented", "methodical"]
)
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and contribution workflows.

### Adding New Experts

1. Create expert class inheriting from base interface
2. Add metadata and configuration
3. Register in expert module
4. Add comprehensive tests
5. Update documentation

### Development Setup

```bash
# Clone repository
git clone https://github.com/vmanoilov/zenrube.git
cd zenrube

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black zenrube tests
flake8 zenrube tests
mypy zenrube
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Concept by [@vmanoilov](https://github.com/vmanoilov)
- Built for the [Rube.app](https://rube.app) automation platform
- Inspired by modular AI architectures and consensus-based systems
- Features Dynamic Personality System integration

## 📊 Status

[![CI](https://github.com/vmanoilov/zenrube/workflows/CI/badge.svg)](https://github.com/vmanoilov/zenrube/actions)
[![Coverage](https://codecov.io/gh/vmanoilov/zenrube/branch/main/graph/badge.svg)](https://codecov.io/gh/vmanoilov/zenrube)
[![PyPI version](https://badge.fury.io/py/zenrube.svg)](https://badge.fury.io/py/zenrube)

---

**ZenRube** - Where AI experts collaborate to provide comprehensive solutions.
