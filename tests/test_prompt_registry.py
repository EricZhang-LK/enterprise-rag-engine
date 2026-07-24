import pytest

from enterprise_rag_engine.generation import PromptRegistry, PromptTemplate


def test_prompt_registry_returns_the_explicitly_activated_version() -> None:
    registry = PromptRegistry()
    v1 = PromptTemplate(
        name="grounded_answer",
        version="1.0.0",
        system_template="Answer only from the supplied context.",
        user_template="Question: {question}",
    )
    v2 = PromptTemplate(
        name="grounded_answer",
        version="1.1.0",
        system_template="Answer from context and cite every factual claim.",
        user_template="Question: {question}",
    )

    registry.register(v1, activate=True)
    registry.register(v2)

    assert registry.get("grounded_answer") is v1
    assert registry.get("grounded_answer", version="1.1.0") is v2
    assert registry.versions("grounded_answer") == ("1.0.0", "1.1.0")


def test_prompt_registry_switches_active_version_only_when_requested() -> None:
    registry = PromptRegistry()
    registry.register(_template("1.0.0"), activate=True)
    registry.register(_template("1.1.0"))

    registry.activate("grounded_answer", version="1.1.0")

    assert registry.get("grounded_answer").version == "1.1.0"


def test_prompt_registry_rejects_duplicate_or_unknown_versions() -> None:
    registry = PromptRegistry()
    registry.register(_template("1.0.0"), activate=True)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_template("1.0.0"))
    with pytest.raises(LookupError, match="not registered"):
        registry.get("grounded_answer", version="2.0.0")
    with pytest.raises(LookupError, match="not registered"):
        registry.activate("grounded_answer", version="2.0.0")


def test_prompt_registry_requires_an_explicit_active_version() -> None:
    registry = PromptRegistry()
    registry.register(_template("1.0.0"))

    with pytest.raises(LookupError, match="no active version"):
        registry.get("grounded_answer")


def test_prompt_template_rejects_blank_identity_or_content() -> None:
    with pytest.raises(ValueError, match="name"):
        PromptTemplate(
            name=" ",
            version="1.0.0",
            system_template="system",
            user_template="user",
        )
    with pytest.raises(ValueError, match="version"):
        PromptTemplate(
            name="grounded_answer",
            version=" ",
            system_template="system",
            user_template="user",
        )
    with pytest.raises(ValueError, match="system_template"):
        PromptTemplate(
            name="grounded_answer",
            version="1.0.0",
            system_template=" ",
            user_template="user",
        )


def _template(version: str) -> PromptTemplate:
    return PromptTemplate(
        name="grounded_answer",
        version=version,
        system_template="Answer only from the supplied context.",
        user_template="Question: {question}",
    )
