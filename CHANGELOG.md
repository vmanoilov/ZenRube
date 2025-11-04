# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2024-05-01

### Added
- Initial release of zenrube-mcp with multi-expert consensus orchestration.
- CLI entry point and basic synthesis workflow.

## [0.2.0] - 2024-06-01

### Added
- Structured Pydantic models for expert responses and consensus results.
- YAML configuration loading with custom expert registration.
- Caching layer with in-memory, file, and Redis backends.
- Provider registry with Rube and OpenAI implementations.
- Unit test suite covering synthesis styles, execution modes, and degraded scenarios.
- GitHub Actions CI pipeline for tests, linting, formatting, and typing.
- Documentation updates including CONTRIBUTING guide and configuration examples.
