import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ParserGoldenExpectation(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_chunk_count: int = Field(default=0, ge=0)
    required_text: tuple[str, ...] = Field(default_factory=tuple)
    required_page_numbers: tuple[int, ...] = Field(default_factory=tuple)
    required_section_paths: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    requires_table: bool = False


class ParserGoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source_uri: str
    document_type: str
    parser: str
    expectation: ParserGoldenExpectation
    notes: str | None = None


class ParserGoldenDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: tuple[ParserGoldenCase, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)


def load_parser_golden_dataset(path: Path) -> ParserGoldenDataset:
    cases: list[ParserGoldenCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        cases.append(ParserGoldenCase.model_validate(payload))
    return ParserGoldenDataset(cases=tuple(cases))


def summarize_parser_golden_dataset(dataset: ParserGoldenDataset) -> dict[str, int]:
    by_type: dict[str, int] = {}
    by_parser: dict[str, int] = {}
    table_cases = 0

    for case in dataset.cases:
        by_type[case.document_type] = by_type.get(case.document_type, 0) + 1
        by_parser[case.parser] = by_parser.get(case.parser, 0) + 1
        if case.expectation.requires_table:
            table_cases += 1

    summary = {"case_count": dataset.case_count, "table_cases": table_cases}
    summary.update({f"type:{key}": value for key, value in by_type.items()})
    summary.update({f"parser:{key}": value for key, value in by_parser.items()})
    return summary
