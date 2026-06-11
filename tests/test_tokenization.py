import importlib.util

import pytest

from enterprise_rag_engine import (
    HuggingFaceTokenCounter,
    TiktokenTokenCounter,
    TokenBudget,
    TokenCounter,
)


def test_token_counter_counts_english_words_and_punctuation() -> None:
    counter = TokenCounter()

    assert counter.count("hello, RAG world!") == 5


def test_token_counter_counts_cjk_characters_as_individual_tokens() -> None:
    counter = TokenCounter()

    assert counter.count("企业级 RAG") == 4


def test_token_counter_truncates_by_token_boundary() -> None:
    counter = TokenCounter()

    assert counter.truncate("hello, RAG world!", max_tokens=3) == "hello, RAG"


def test_token_counter_rejects_negative_budget() -> None:
    counter = TokenCounter()

    with pytest.raises(ValueError, match="max_tokens"):
        counter.fits("hello", max_tokens=-1)


def test_token_budget_calculates_available_input_tokens() -> None:
    budget = TokenBudget(
        max_context_tokens=8192,
        reserved_output_tokens=1024,
        reserved_system_tokens=512,
        reserved_prompt_tokens=256,
    )

    assert budget.available_input_tokens == 6400
    assert budget.allocate_per_chunk(chunk_count=8) == 800


def test_token_budget_rejects_over_reserved_window() -> None:
    with pytest.raises(ValueError, match="reserved tokens"):
        TokenBudget(max_context_tokens=100, reserved_output_tokens=100)


def test_token_budget_rejects_invalid_chunk_count() -> None:
    budget = TokenBudget(max_context_tokens=100, reserved_output_tokens=20)

    with pytest.raises(ValueError, match="chunk_count"):
        budget.allocate_per_chunk(chunk_count=0)


def test_tiktoken_counter_requires_optional_dependency_when_missing() -> None:
    if importlib.util.find_spec("tiktoken") is not None:
        pytest.skip("tiktoken is installed in this environment")

    with pytest.raises(RuntimeError, match="tokenization"):
        TiktokenTokenCounter()


def test_huggingface_counter_requires_optional_dependency_when_missing() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed in this environment")

    with pytest.raises(RuntimeError, match="tokenization"):
        HuggingFaceTokenCounter("Qwen/Qwen2.5-7B-Instruct")
