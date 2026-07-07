import pytest

from enterprise_rag_engine import (
    BaseEmbedder,
    EmbeddingBatchResult,
    EmbeddingRequest,
    EmbeddingResult,
)


class FakeEmbedder(BaseEmbedder):
    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        results = tuple(
            EmbeddingResult(
                text=text,
                embedding=(float(index), float(len(text))),
                model=request.model,
                token_count=len(text.split()),
            )
            for index, text in enumerate(request.texts, start=1)
        )
        return EmbeddingBatchResult.from_results(results=results, model=request.model)


def test_embedder_contract_embeds_texts_in_batches() -> None:
    embedder = FakeEmbedder()

    result = embedder.embed(
        EmbeddingRequest(
            texts=("hello rag", "metadata filter"),
            model="fake-embedding",
            normalize=True,
        )
    )

    assert result.model == "fake-embedding"
    assert result.dimension == 2
    assert result.embeddings == ((1.0, 9.0), (2.0, 15.0))
    assert [item.token_count for item in result.results] == [2, 2]


def test_embedder_contract_keeps_legacy_embed_texts_convenience() -> None:
    embedder = FakeEmbedder()

    embeddings = embedder.embed_texts(("hello", "rag"))

    assert embeddings == ((1.0, 5.0), (2.0, 3.0))


@pytest.mark.anyio
async def test_embedder_contract_supports_async_embedding() -> None:
    embedder = FakeEmbedder()

    result = await embedder.aembed(EmbeddingRequest(texts=("async rag",), model="fake-async"))

    assert result.model == "fake-async"
    assert result.embeddings == ((1.0, 9.0),)


def test_embedding_request_rejects_empty_batches_and_blank_texts() -> None:
    try:
        EmbeddingRequest(texts=())
    except ValueError as exc:
        assert "texts" in str(exc)
    else:
        raise AssertionError("empty embedding batches should be rejected")

    try:
        EmbeddingRequest(texts=("valid", " "))
    except ValueError as exc:
        assert "blank" in str(exc)
    else:
        raise AssertionError("blank embedding texts should be rejected")


def test_embedding_batch_result_rejects_mixed_dimensions() -> None:
    try:
        EmbeddingBatchResult.from_results(
            results=(
                EmbeddingResult(text="a", embedding=(1.0,)),
                EmbeddingResult(text="b", embedding=(1.0, 2.0)),
            )
        )
    except ValueError as exc:
        assert "same dimension" in str(exc)
    else:
        raise AssertionError("mixed embedding dimensions should be rejected")
