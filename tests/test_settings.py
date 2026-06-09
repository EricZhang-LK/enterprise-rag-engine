from enterprise_rag_engine.settings import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "enterprise-rag-engine"
    assert settings.app_env == "dev"
    assert settings.app_version == "0.1.0"
    assert settings.log_level == "INFO"
