"""Load and validate the YAML configuration file."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = Field(0.2, ge=0, le=2)
    qa_temperature: float = Field(0.4, ge=0, le=2)
    max_output_tokens: int = Field(8192, gt=0)
    thinking_budget: int | None = 0
    max_retries: int = Field(4, ge=0)
    retry_base_delay: float = Field(2.0, ge=0)


class AnalysisConfig(BaseModel):
    batch_size: int = Field(10, ge=1, le=50)
    max_workers: int = Field(4, ge=1, le=16)
    max_review_chars: int = Field(1500, gt=0)
    min_review_chars: int = Field(3, ge=0)
    max_reviews: int = Field(500, gt=0)
    aspects: list[str] = Field(default_factory=list)
    allow_new_aspects: bool = True


class SynthesisConfig(BaseModel):
    max_quotes_per_polarity: int = Field(25, gt=0)


class QAConfig(BaseModel):
    top_k_reviews: int = Field(25, gt=0)


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: str = ".cache"


class PathsConfig(BaseModel):
    prompts: str = "prompts/prompts.yaml"
    sample_data: str = "data/sample_reviews.csv"


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a path from the config relative to the project root."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read the YAML config file; missing keys fall back to defaults."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment or a local .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
