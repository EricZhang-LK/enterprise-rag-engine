# Enterprise RAG Engine

Enterprise RAG Engine is a Python backend project for building production-minded
retrieval-augmented generation systems.

This repository is part of a 24-week LLM application engineering learning plan. The goal
is not to build a toy RAG demo, but to gradually implement a system with document
pipelines, chunking, hybrid retrieval, reranking, citations, evaluations, streaming APIs,
multi-tenant controls, and observability.

## Current Milestone

- Stage: W1/D1
- Focus: project initialization and `src layout`
- Status: scaffolded

## Planned Capabilities

- Document parsing pipeline
- Metadata-preserving chunking
- Dense, sparse, and hybrid retrieval
- Reranking
- Citation-aware answer generation
- Retrieval and generation evaluations
- FastAPI service layer
- Streaming responses
- Multi-tenant access boundaries
- Observability and deployment documentation

## Project Layout

```text
enterprise-rag-engine/
├── docs/
│   ├── adr/
│   └── reports/
├── datasets/
├── scripts/
├── src/
│   └── enterprise_rag_engine/
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Engineering Baseline

This project starts with a strict engineering baseline:

- `src layout` for clean packaging boundaries
- `ruff` for linting and import sorting
- `mypy strict` for type checking
- `pytest` for automated tests
- `pytest-cov` for coverage reporting
- MIT license for open-source friendliness

## Development

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run quality checks:

```powershell
ruff check .
ruff format --check .
mypy src tests
pytest
```

Install pre-commit hooks:

```powershell
pre-commit install
pre-commit run --all-files
```

## Learning Notes

Each major design decision will be recorded in `docs/adr/`.
Each benchmark or evaluation report will be recorded in `docs/reports/`.
