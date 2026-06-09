from pathlib import Path

import pytest

from enterprise_rag_engine.evals.parser_golden import (
    load_parser_golden_dataset,
    summarize_parser_golden_dataset,
)


def test_load_parser_golden_dataset() -> None:
    dataset = load_parser_golden_dataset(Path("datasets/parser_golden/cases.jsonl"))

    assert dataset.case_count == 5
    assert dataset.cases[0].id == "pdf-basic-pages"
    assert dataset.cases[0].expectation.required_page_numbers == (1, 2)


def test_summarize_parser_golden_dataset() -> None:
    dataset = load_parser_golden_dataset(Path("datasets/parser_golden/cases.jsonl"))

    summary = summarize_parser_golden_dataset(dataset)

    assert summary["case_count"] == 5
    assert summary["table_cases"] == 1
    assert summary["type:pdf"] == 3
    assert summary["parser:MarkdownParser"] == 1


def test_load_parser_golden_dataset_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="Invalid JSON on line 1"):
        load_parser_golden_dataset(Path("tests/fixtures/parser_golden_invalid.jsonl"))
