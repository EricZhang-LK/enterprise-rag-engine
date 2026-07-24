"""Prompt and evidence-context primitives for answer generation."""

from enterprise_rag_engine.generation.context import ContextBuilder, ContextBuildResult
from enterprise_rag_engine.generation.prompts import PromptRegistry, PromptTemplate

__all__ = ["ContextBuilder", "ContextBuildResult", "PromptRegistry", "PromptTemplate"]
