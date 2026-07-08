import argparse
import json

from enterprise_rag_engine.evals.index_parameter_benchmark import (
    IndexBenchmarkScenario,
    benchmark_index_profiles,
    default_index_profiles,
    render_index_benchmark_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Qdrant HNSW index parameter tradeoffs for RAG retrieval."
    )
    parser.add_argument("--vectors", type=int, default=50_000)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--filter-selectivity", type=float, default=0.05)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = IndexBenchmarkScenario(
        vector_count=args.vectors,
        dimension=args.dimension,
        top_k=args.top_k,
        filter_selectivity=args.filter_selectivity,
    )
    rows = benchmark_index_profiles(default_index_profiles(), scenario)
    if args.format == "json":
        print(json.dumps([row.as_dict() for row in rows], indent=2, ensure_ascii=False))
        return
    print(render_index_benchmark_markdown(rows))


if __name__ == "__main__":
    main()
