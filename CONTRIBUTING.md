# Contributing to zenrube-mcp

Thank you for considering a contribution! This document describes how to set up your environment, run checks, and submit pull requests.

## 🛠️ Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vmanoilov/zenrube-mcp.git
   cd zenrube-mcp
   ```
2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. (Optional) Configure your preferred LLM providers by registering them in `providers.ProviderRegistry`.

## 🧪 Running Tests

Zenrube uses pytest for unit tests and coverage reporting:

```bash
pytest --cov=src
```

## 🧹 Quality Gates

Before submitting a pull request, ensure that:

- `black` formatting passes: `black --check src tests`
- `flake8` linting passes: `flake8 src tests`
- `mypy` type checking passes: `mypy src`

These checks also run in CI for Python 3.8–3.12.

## 🔄 Git Workflow

1. Create a feature branch from `main`.
2. Make your changes and add relevant tests.
3. Run the quality gates above.
4. Update `CHANGELOG.md` with a summary of your change.
5. Open a pull request with a clear description and link any relevant issues.

## 🤝 Code Style

- Follow Python typing best practices and prefer explicit return types.
- Use structured logging via the shared `zenrube` logger.
- Keep public APIs documented in the README.

## 🗣️ Communication

If you plan a significant change (new provider, caching backend, etc.), please open an issue first so we can discuss the approach.

Thanks for helping make zenrube-mcp better!
