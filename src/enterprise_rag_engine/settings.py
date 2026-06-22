from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-rag-engine", alias="APP_NAME")
    app_env: Literal["dev", "test", "prod"] = Field(default="dev", alias="APP_ENV")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    upload_dir: Path = Field(default=Path(".data/uploads"), alias="UPLOAD_DIR")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0, alias="MAX_UPLOAD_BYTES")


@lru_cache
def get_settings() -> Settings:
    return Settings()
