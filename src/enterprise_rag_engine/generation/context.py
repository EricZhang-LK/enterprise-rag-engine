from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag_engine.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    """Evidence selected for a prompt, with its rendered text and build metadata."""

    results: tuple[RetrievalResult, ...]
    text: str
    deduplicated_count: int
    truncated: bool

    @property
    def character_count(self) -> int:
        """Return the exact character count of the rendered context text."""

        return len(self.text)


class ContextBuilder:
    """Prepare complete, ranked retrieval evidence for a future prompt renderer.

    The builder uses retrieval rank rather than raw score because scores from
    different retrieval strategies are not necessarily comparable. It keeps
    complete chunks only, so later citations never point at partial evidence.
    """

    def __init__(self, *, max_characters: int) -> None:
        if max_characters < 1:
            msg = "max_characters must be greater than 0"
            raise ValueError(msg)
        self._max_characters = max_characters

    def build(self, results: tuple[RetrievalResult, ...]) -> ContextBuildResult:
        """Sort, deduplicate, and fit retrieved evidence into the character budget."""

        selected: list[RetrievalResult] = []
        rendered_parts: list[str] = []
        seen_chunk_ids: set[str] = set()
        deduplicated_count = 0
        truncated = False

        for _, result in sorted(
            enumerate(results),
            key=lambda item: (item[1].rank, -item[1].score, item[0]),
        ):
            chunk_id = result.chunk.id
            if chunk_id in seen_chunk_ids:
                deduplicated_count += 1
                continue
            seen_chunk_ids.add(chunk_id)

            rendered = _render_result(result)
            separator = "\n\n" if rendered_parts else ""
            current_length = sum(len(part) for part in rendered_parts)
            candidate_length = current_length + len(separator) + len(rendered)
            if candidate_length > self._max_characters:
                truncated = True
                break

            selected.append(result)
            rendered_parts.append(rendered)

        return ContextBuildResult(
            results=tuple(selected),
            text="\n\n".join(rendered_parts),
            deduplicated_count=deduplicated_count,
            truncated=truncated,
        )


def _render_result(result: RetrievalResult) -> str:
    """Attach a stable chunk identifier so later stages can resolve citations."""

    return f"[{result.chunk.id}]\n{result.chunk.content}"
