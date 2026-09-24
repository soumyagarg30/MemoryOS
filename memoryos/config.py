from functools import lru_cache

from pydantic import Field, HttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORYOS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: PostgresDsn = PostgresDsn(
        "postgresql+psycopg://memoryos:memoryos@localhost:5432/memoryos"
    )
    debug: bool = False
    sql_echo: bool = False
    chat_model: str = Field(default="qwen2.5:7b", min_length=1)
    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    ollama_timeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    extraction_model: str = Field(default="qwen2.5:7b", min_length=1)
    conflict_model: str = Field(default="qwen2.5:7b", min_length=1)
    consolidation_model: str = Field(default="qwen2.5:7b", min_length=1)
    consolidation_similarity: float = Field(default=0.85, ge=0, le=1, allow_inf_nan=False)
    consolidation_min_confidence: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    embedding_model: str = Field(default="nomic-embed-text:latest", min_length=1)
    embedding_dimensions: int | None = Field(default=None, ge=1, le=16000)

    @field_validator("database_url")
    @classmethod
    def validate_driver(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != "postgresql+psycopg":
            raise ValueError("Use the postgresql+psycopg:// driver scheme")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
