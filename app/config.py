from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENROUTER_API_KEY: str = Field(..., description="OpenRouter API key")
    LLM_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="OpenRouter model identifier",
    )
    BASE_DIR: Path = Field(
        default=Path.cwd(),
        description="Folders in requests must resolve under this directory",
    )
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    LLM_TIMEOUT_SECONDS: int = Field(default=120)
    LLM_MAX_RETRIES: int = Field(default=1)
    LLM_MAX_TOKENS: int = Field(
        default=16384,
        description=(
            "Upper bound on tokens per LLM response. Set generously because "
            "reasoning models (gpt-oss-*) count reasoning tokens against this "
            "budget. For 20+ files in a single ranking call, 8K is too tight: "
            "the model can burn 7-8K on internal reasoning, leaving no room "
            "for the final JSON output."
        ),
    )

    MONGODB_URI: str = Field(..., description="MongoDB connection URI (Atlas SRV or local)")
    MONGODB_DATABASE: str = Field(default="brand_summarizer")
    MONGODB_COLLECTION: str = Field(default="brand_states")
    MONGODB_TIMEOUT_MS: int = Field(default=10_000)


settings = Settings()
