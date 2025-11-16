# ZenRube AI Coding Agent Instructions

This guide provides essential context for AI agents working on the ZenRube codebase. Following these instructions will help you be more effective and align with the project's architecture and conventions.

## 1. Architecture Overview

ZenRube is a modular AI orchestration platform built around a system of "Experts." Each expert is a specialized Python class responsible for a specific task (e.g., data cleaning, semantic routing, content summarization).

- **Expert Discovery**: The `ExpertRegistry` in `zenrube/experts/expert_registry.py` is the central component for discovering and loading all available experts. Experts are Python files located in the `zenrube/experts/` directory.
- **Expert Structure**: Each expert module (e.g., `zenrube/experts/data_cleaner.py`) must contain an `EXPERT_METADATA` dictionary and an expert class (e.g., `DataCleanerExpert`). The metadata is crucial for discovery and versioning.
- **Orchestration**: The `TeamCouncil` and `council_runner` manage the interaction between experts. They use a consensus-based model to synthesize outputs from multiple experts.
- **Dynamic Personalities**: The system uses a `ProfileController` (`zenrube/profiles/profile_controller.py`) to dynamically adjust expert behavior based on configurable personality profiles. This allows for different "modes" of operation (e.g., creative, analytical).
- **Configuration**: Configuration is primarily handled through YAML files (`.zenrube.yml`) and JSON files in `zenrube/config/`.

**Key Data Flow**:
1. Input is received by the system (e.g., via `zen_consensus` function or CLI).
2. The `SemanticRouterExpert` often runs first to determine intent and route the request.
3. The `TeamCouncil` selects and runs a group of relevant experts.
4. Each expert processes the data and returns a result.
5. The council synthesizes the results into a final consensus output.

## 2. Developer Workflow

### Running Tests
The project uses `pytest`. The most important tests are the integration tests, which provide a good overview of the end-to-end workflow.

- Run all tests:
  ```bash
  pytest
  ```
- Run specific integration tests to understand the full system:
  ```bash
  pytest tests/test_full_system_integration.py
  ```

### Using the CLI
The CLI (`zenrube/cli.py`) is a key entry point for interacting with the system.

- **List experts**: See all registered experts.
  ```bash
  python -m zenrube.cli list
  ```
- **Run an expert**: Execute a specific expert with a given input.
  ```bash
  python -m zenrube.cli run --expert data_cleaner --input "  some messy text  "
  ```
- **Autopublish**: Trigger the automated expert publishing workflow.
  ```bash
  python -m zenrube.cli autopublish
  ```

## 3. Key Conventions

- **Expert Modules**: When creating a new expert, create a new Python file in `zenrube/experts/`. It MUST contain:
    1.  An `EXPERT_METADATA` dictionary with `name`, `version`, `description`, and `author`.
    2.  A class named in `CamelCase` ending with `Expert` (e.g., `MyNewExpert`).
    
    *Example from `zenrube/experts/summarizer.py`:*
    ```python
    EXPERT_METADATA = {
        "name": "summarizer",
        "version": "1.1",
        "description": "AI-powered text summarization expert.",
        "author": "vladinc@gmail.com"
    }

    class SummarizerExpert:
        # ... implementation ...
    ```

- **Configuration Loading**: Use the loader modules in `zenrube/config/` (e.g., `llm_config_loader.py`) to access configuration. Avoid reading config files directly.

- **Logging**: Use the shared `zenrube` logger for structured logging.

## 4. Important Files and Directories

- `zenrube/experts/`: Location of all expert modules. This is where you'll add new functionality.
- `zenrube/experts/expert_registry.py`: The core of the expert discovery mechanism.
- `zenrube/config/`: Contains all system and expert configuration files.
- `tests/test_full_system_integration.py`: The best place to understand the end-to-end workflow and how components interact.
- `README.md`: Provides a high-level overview and basic usage examples.
- `CONTRIBUTING.md`: Contains guidelines for running quality checks (`black`, `flake8`, `mypy`).
