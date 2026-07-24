import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from enterprise_rag_engine.evals.retrieval import (
    RetrievalEvaluationCase,
    RetrievalMetrics,
    evaluate_retrieval_rankings,
)

FIXTURE_PATH = Path("datasets/retrieval_fixture/cases.jsonl")


def load_fixture(
    path: Path,
) -> tuple[tuple[RetrievalEvaluationCase, ...], dict[str, dict[str, tuple[str, ...]]]]:
    """Load evaluation cases and recorded rankings from a JSONL fixture."""

    cases: list[RetrievalEvaluationCase] = []
    rankings_by_strategy: dict[str, dict[str, tuple[str, ...]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case = RetrievalEvaluationCase(
            id=payload["id"],
            query=payload["query"],
            relevant_chunk_ids=frozenset(payload["relevant_chunk_ids"]),
        )
        cases.append(case)
        _append_rankings(
            rankings_by_strategy=rankings_by_strategy,
            case_id=case.id,
            rankings=payload["rankings"],
        )
    return tuple(cases), rankings_by_strategy


def evaluate_fixture(
    cases: Sequence[RetrievalEvaluationCase],
    rankings_by_strategy: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, RetrievalMetrics]:
    """Evaluate every recorded strategy using the same cases and metric definitions."""

    return {
        strategy: evaluate_retrieval_rankings(cases=cases, rankings_by_case=rankings)
        for strategy, rankings in rankings_by_strategy.items()
    }


def render_markdown(metrics_by_strategy: Mapping[str, RetrievalMetrics]) -> str:
    """Render an auditable comparison table for the D42 retrieval report."""

    lines = [
        "| strategy | cases | recall@5 | recall@10 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, metrics in metrics_by_strategy.items():
        lines.append(
            "| "
            f"{strategy} | {metrics.case_count} | {metrics.recall_at_5:.3f} | "
            f"{metrics.recall_at_10:.3f} | {metrics.mrr:.3f} | {metrics.ndcg_at_10:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    cases, rankings_by_strategy = load_fixture(FIXTURE_PATH)
    print(render_markdown(evaluate_fixture(cases, rankings_by_strategy)))


def _append_rankings(
    *,
    rankings_by_strategy: dict[str, dict[str, tuple[str, ...]]],
    case_id: str,
    rankings: Mapping[str, Sequence[str]],
) -> None:
    for strategy, ranking in rankings.items():
        rankings_by_strategy.setdefault(strategy, {})[case_id] = tuple(ranking)


if __name__ == "__main__":
    main()
