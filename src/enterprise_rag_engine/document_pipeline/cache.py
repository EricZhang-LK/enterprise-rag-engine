from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from enterprise_rag_engine.models import ParseResult, ParseStatus


class HashProvider(Protocol):
    """Return a content hash for a source URI when it is cacheable."""

    def __call__(self, source_uri: str) -> str | None: ...


@dataclass(frozen=True)
class CacheEntry:
    """Cached parse result tied to a source URI and content hash."""

    source_uri: str
    content_hash: str
    result: ParseResult


class CacheManager:
    """In-memory parse-result cache keyed by source URI and file content hash."""

    def __init__(self, hash_provider: HashProvider | None = None) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._hash_provider = hash_provider or file_content_hash

    def get(self, source_uri: str) -> ParseResult | None:
        """Return a cached parse result when the source file content is unchanged."""

        content_hash = self._hash_provider(source_uri)
        if content_hash is None:
            return None

        entry = self._entries.get(source_uri)
        if entry is None or entry.content_hash != content_hash:
            return None
        return entry.result

    def put(self, source_uri: str, result: ParseResult) -> None:
        """Cache successful or partially successful parse results for local files."""

        if result.status is ParseStatus.FAILED:
            return

        content_hash = self._hash_provider(source_uri)
        if content_hash is None:
            return

        self._entries[source_uri] = CacheEntry(
            source_uri=source_uri,
            content_hash=content_hash,
            result=result,
        )

    def invalidate(self, source_uri: str) -> None:
        """Remove a source URI from the cache."""

        self._entries.pop(source_uri, None)

    def clear(self) -> None:
        """Remove all cache entries."""

        self._entries.clear()


def file_content_hash(source_uri: str) -> str | None:
    """Return a SHA-256 hash for local files, or None for unsupported sources."""

    path = Path(source_uri)
    if not path.is_file():
        return None

    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
