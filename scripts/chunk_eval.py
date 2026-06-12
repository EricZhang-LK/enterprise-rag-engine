from enterprise_rag_engine import (
    Document,
    DocumentType,
    ParentChildSplitter,
    RecursiveSplitter,
    SemanticSplitter,
)
from enterprise_rag_engine.evals.chunk_quality import evaluate_chunks

SAMPLE_TEXT = """
# RAG Chunking

RAG retrieval quality depends on chunk quality. Good chunks keep enough context for
the model to understand definitions, constraints, and exceptions. Bad chunks may
separate a definition from the paragraph that explains it.

Parent-child chunking separates retrieval from generation. Child chunks are small and
focused, so vector search can match user intent more precisely. Parent chunks are
larger and preserve surrounding context for answer generation.

FastAPI exposes backend APIs. Observability records latency, token usage, and errors.
These concerns are related to production readiness but not always related to chunk
boundaries.
""".strip()


def main() -> None:
    document = Document(
        source_uri="sample://chunking",
        type=DocumentType.MARKDOWN,
        content=SAMPLE_TEXT,
    )
    strategies = {
        "recursive": RecursiveSplitter(max_chars=180, overlap_chars=30),
        "parent_child": ParentChildSplitter(
            parent_max_chars=320,
            parent_overlap_chars=40,
            child_max_chars=120,
            child_overlap_chars=20,
        ),
        "semantic": SemanticSplitter(max_chars=220, min_chars=80, similarity_threshold=0.08),
    }

    for name, splitter in strategies.items():
        metrics = evaluate_chunks(splitter.split(document))
        print(f"[{name}]")
        for metric_name, value in metrics.as_dict().items():
            print(f"{metric_name}: {value}")
        print()


if __name__ == "__main__":
    main()
