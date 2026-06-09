import json

from enterprise_rag_engine import ChunkType, TableBlock
from enterprise_rag_engine.document_pipeline.tables import (
    normalize_table_rows,
    table_to_chunk,
    table_to_csv,
    table_to_json,
    table_to_markdown,
)


def test_table_block_reports_shape() -> None:
    table = TableBlock(rows=(("A", "B"), ("1", "2")))

    assert table.row_count == 2
    assert table.column_count == 2
    assert table.is_rectangular is True


def test_table_block_detects_ragged_rows() -> None:
    table = TableBlock(rows=(("A", "B"), ("1",)))

    assert table.column_count == 2
    assert table.is_rectangular is False


def test_normalize_table_rows_converts_cells_to_strings() -> None:
    rows = normalize_table_rows([["A", 1, None], ["B", "  value  ", 3.14]])

    assert rows == (("A", "1", ""), ("B", "value", "3.14"))


def test_table_to_markdown_includes_caption_and_escapes_cells() -> None:
    table = TableBlock(rows=(("Name", "Value"), ("A|B", "line1\nline2")), caption="Demo Table")

    assert table_to_markdown(table) == (
        "Demo Table\n\n"
        "| Name | Value |\n"
        "| --- | --- |\n"
        "| A\\|B | line1<br>line2 |"
    )


def test_table_to_csv_outputs_rectangular_rows() -> None:
    table = TableBlock(rows=(("A", "B"), ("1",)))

    assert table_to_csv(table) == "A,B\n1,\n"


def test_table_to_json_uses_header_row() -> None:
    table = TableBlock(rows=(("name", "value"), ("alpha", "1")))

    assert json.loads(table_to_json(table)) == [{"name": "alpha", "value": "1"}]


def test_table_to_chunk_preserves_table_metadata() -> None:
    table = TableBlock(
        rows=(("name", "value"), ("alpha", "1")),
        page_number=3,
        section_path=("Appendix",),
    )

    chunk = table_to_chunk(table=table, source_uri="demo.pdf", document_id="doc-1")

    assert chunk.chunk_type is ChunkType.TABLE
    assert chunk.metadata.page_number == 3
    assert chunk.metadata.section_path == ("Appendix",)
    assert chunk.metadata.content_hash is not None
    assert "| name | value |" in chunk.content
