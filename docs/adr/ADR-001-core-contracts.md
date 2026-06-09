# ADR-001: Core Data and Interface Contracts

## Status

Accepted

## 中文摘要

本 ADR 记录 `enterprise-rag-engine` 的核心契约设计：使用 Pydantic 模型定义数据边界，
使用抽象基类定义组件行为边界。这样可以让文档解析、分块、embedding、向量库、检索和
评测模块在后续扩展时保持清晰、可测试、可替换。

## Context

The RAG engine will include document parsers, splitters, embedders, vector stores,
retrievers, evaluators, and API services. These components will evolve independently
and may have multiple implementations.

Without explicit contracts, each implementation can invent its own input and output
shape. That makes testing, evaluation, replacement, and debugging difficult.

## Decision

Use Pydantic models for core data contracts and Python abstract base classes for
component behavior contracts.

The initial data contracts are:

- `Document`
- `DocumentChunk`
- `ChunkMetadata`
- `ParseResult`
- `RetrievalResult`

The initial behavior contracts are:

- `BaseParser`
- `BaseSplitter`
- `BaseEmbedder`
- `BaseVectorStore`
- `BaseRetriever`
- `BaseEvaluator`

## Consequences

Components must explicitly inherit from the relevant base class. This makes the
implementation hierarchy easier to inspect and leaves room for shared base behavior such
as logging, timing, validation, and default error handling.

Future adapters for Docling, MinerU, Qdrant, Milvus, BM25, and rerankers can share the
same contract boundaries. Third-party objects that do not inherit these base classes
should be wrapped by small adapter classes.
