import csv
import json
from hashlib import sha256
from io import StringIO
from typing import Any

from enterprise_rag_engine.models import ChunkMetadata, ChunkType, DocumentChunk, TableBlock


def normalize_table_rows(rows: list[list[Any]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_cell_to_text(cell) for cell in row) for row in rows)


def table_to_markdown(table: TableBlock) -> str:
    if not table.rows:
        return ""

    rectangular_rows = _rectangular_rows(table)
    header = rectangular_rows[0]
    separator = tuple("---" for _ in header)
    body = rectangular_rows[1:]
    lines = [_markdown_row(header), _markdown_row(separator)]
    lines.extend(_markdown_row(row) for row in body)
    markdown = "\n".join(lines)
    if table.caption:
        return f"{table.caption}\n\n{markdown}"
    return markdown


def table_to_csv(table: TableBlock) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(_rectangular_rows(table))
    return output.getvalue()


def table_to_json(table: TableBlock) -> str:
    rows = _rectangular_rows(table)
    if not rows:
        return "[]"
    header = rows[0]
    records = [dict(zip(header, row, strict=True)) for row in rows[1:]]
    return json.dumps(records, ensure_ascii=False)


def table_to_chunk(
    *,
    table: TableBlock,
    source_uri: str,
    document_id: str,
) -> DocumentChunk:
    markdown = table_to_markdown(table)
    return DocumentChunk(
        document_id=document_id,
        content=markdown,
        chunk_type=ChunkType.TABLE,
        metadata=ChunkMetadata(
            source_uri=source_uri,
            document_id=document_id,
            page_number=table.page_number,
            section_path=table.section_path,
            content_hash=sha256(markdown.encode("utf-8")).hexdigest(),
        ),
    )


def _rectangular_rows(table: TableBlock) -> tuple[tuple[str, ...], ...]:
    width = table.column_count
    return tuple(row + ("",) * (width - len(row)) for row in table.rows)


def _markdown_row(row: tuple[str, ...]) -> str:
    escaped = [_escape_markdown_cell(cell) for cell in row]
    return "| " + " | ".join(escaped) + " |"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
