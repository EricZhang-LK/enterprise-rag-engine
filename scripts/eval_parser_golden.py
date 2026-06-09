from pathlib import Path

from enterprise_rag_engine.evals.parser_golden import (
    load_parser_golden_dataset,
    summarize_parser_golden_dataset,
)


def main() -> None:
    dataset_path = Path("datasets/parser_golden/cases.jsonl")
    dataset = load_parser_golden_dataset(dataset_path)
    summary = summarize_parser_golden_dataset(dataset)
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
