import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

TOKEN_PATTERN = re.compile(
    r"[\u4e00-\u9fff]"
    r"|[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)*"
    r"|[^\s]"
)


@dataclass(frozen=True)
class TokenBudget:
    """Context-window budget reserved for prompt, retrieved chunks, and output."""

    max_context_tokens: int
    reserved_output_tokens: int
    reserved_system_tokens: int = 0
    reserved_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_context_tokens <= 0:
            msg = "max_context_tokens must be greater than 0"
            raise ValueError(msg)
        if self.reserved_output_tokens < 0:
            msg = "reserved_output_tokens must be greater than or equal to 0"
            raise ValueError(msg)
        if self.reserved_system_tokens < 0:
            msg = "reserved_system_tokens must be greater than or equal to 0"
            raise ValueError(msg)
        if self.reserved_prompt_tokens < 0:
            msg = "reserved_prompt_tokens must be greater than or equal to 0"
            raise ValueError(msg)
        if self.available_input_tokens <= 0:
            msg = "reserved tokens must leave at least one input token"
            raise ValueError(msg)

    @property
    def available_input_tokens(self) -> int:
        """Return tokens available for retrieved chunks and user input."""

        return (
            self.max_context_tokens
            - self.reserved_output_tokens
            - self.reserved_system_tokens
            - self.reserved_prompt_tokens
        )

    def allocate_per_chunk(self, *, chunk_count: int) -> int:
        """Divide the input budget across a planned number of retrieved chunks."""

        if chunk_count <= 0:
            msg = "chunk_count must be greater than 0"
            raise ValueError(msg)
        return max(1, self.available_input_tokens // chunk_count)


@dataclass(frozen=True)
class TokenSpan:
    """A token-like text span with source offsets."""

    text: str
    start: int
    end: int


class BaseTokenCounter(Protocol):
    """Common token counting behavior shared by estimator and exact tokenizers."""

    def count(self, text: str) -> int:
        """Return the token count for one text."""

    def count_many(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        """Return token counts for a batch of texts."""

    def fits(self, text: str, *, max_tokens: int) -> bool:
        """Check whether text fits into a token budget."""

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Trim text to a maximum token count."""


class TokenCounter:
    """Dependency-free token estimator for chunking and budget tests.

    This is not a model-exact tokenizer. It is a stable local estimator that keeps
    the project runnable without network downloads. Model-specific tokenizers can
    later implement the same behavior behind this boundary.
    """

    def count(self, text: str) -> int:
        """Return the estimated token count for one text."""

        return len(self.spans(text))

    def count_many(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        """Return estimated token counts for a batch of texts."""

        return tuple(self.count(text) for text in texts)

    def spans(self, text: str) -> tuple[TokenSpan, ...]:
        """Return token-like spans while preserving character offsets."""

        return tuple(
            TokenSpan(text=match.group(0), start=match.start(), end=match.end())
            for match in TOKEN_PATTERN.finditer(text)
        )

    def fits(self, text: str, *, max_tokens: int) -> bool:
        """Check whether text fits into a token budget."""

        if max_tokens < 0:
            msg = "max_tokens must be greater than or equal to 0"
            raise ValueError(msg)
        return self.count(text) <= max_tokens

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Trim text to a maximum token count while keeping original characters."""

        if max_tokens < 0:
            msg = "max_tokens must be greater than or equal to 0"
            raise ValueError(msg)
        if max_tokens == 0:
            return ""

        spans = self.spans(text)
        if len(spans) <= max_tokens:
            return text

        # Slice by token end offset so whitespace and punctuation stay stable.
        return text[: spans[max_tokens - 1].end].rstrip()


class TiktokenTokenCounter:
    """Model-aware tokenizer adapter for OpenAI-compatible model families."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        try:
            tiktoken = import_module("tiktoken")
        except ImportError as exc:
            msg = "Install tokenization extras with: python -m pip install -e '.[tokenization]'"
            raise RuntimeError(msg) from exc

        if model_name is None:
            self._encoding = tiktoken.get_encoding(encoding_name)
        else:
            self._encoding = tiktoken.encoding_for_model(model_name)

    def count(self, text: str) -> int:
        """Return the exact token count produced by tiktoken."""

        return len(self._encode(text))

    def count_many(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        """Return exact token counts for a batch of texts."""

        return tuple(self.count(text) for text in texts)

    def fits(self, text: str, *, max_tokens: int) -> bool:
        """Check whether text fits into a token budget."""

        _validate_max_tokens(max_tokens)
        return self.count(text) <= max_tokens

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Trim text to a maximum token count using tiktoken decoding."""

        _validate_max_tokens(max_tokens)
        if max_tokens == 0:
            return ""
        token_ids = self._encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return cast(str, self._encoding.decode(token_ids[:max_tokens])).rstrip()

    def _encode(self, text: str) -> list[int]:
        return list(self._encoding.encode(text))


class HuggingFaceTokenCounter:
    """Tokenizer adapter for Qwen, BGE, and other Hugging Face model families."""

    def __init__(self, model_name: str, *, trust_remote_code: bool = False) -> None:
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            msg = "Install tokenization extras with: python -m pip install -e '.[tokenization]'"
            raise RuntimeError(msg) from exc

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

    def count(self, text: str) -> int:
        """Return the token count produced by the model tokenizer."""

        return len(self._encode(text))

    def count_many(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        """Return model tokenizer counts for a batch of texts."""

        return tuple(self.count(text) for text in texts)

    def fits(self, text: str, *, max_tokens: int) -> bool:
        """Check whether text fits into a token budget."""

        _validate_max_tokens(max_tokens)
        return self.count(text) <= max_tokens

    def truncate(self, text: str, *, max_tokens: int) -> str:
        """Trim text to a maximum token count using model tokenizer decoding."""

        _validate_max_tokens(max_tokens)
        if max_tokens == 0:
            return ""
        token_ids = self._encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return str(
            self._tokenizer.decode(
                token_ids[:max_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ).rstrip()

    def _encode(self, text: str) -> list[int]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return list(_ensure_int_sequence(token_ids))


def _validate_max_tokens(max_tokens: int) -> None:
    if max_tokens < 0:
        msg = "max_tokens must be greater than or equal to 0"
        raise ValueError(msg)


def _ensure_int_sequence(token_ids: Any) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in token_ids)
