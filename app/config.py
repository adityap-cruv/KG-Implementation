from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
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
        default=8192,
        description=(
            "Upper bound on tokens per LLM response. Set generously because "
            "reasoning models (gpt-oss-*) burn tokens internally before they "
            "emit the final answer; too-low budgets cause truncated JSON."
        ),
    )


settings = Settings()
