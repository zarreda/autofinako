"""Global settings loaded from configs/global_params.yaml via Pydantic BaseSettings."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "global_params.yaml"


class TargetDirection(StrEnum):
    higher_is_better = "higher_is_better"
    lower_is_better = "lower_is_better"


class EvalMetric(StrEnum):
    oos_r_squared = "oos_r_squared"
    ic = "ic"
    icir = "icir"
    hit_rate = "hit_rate"
    sharpe = "sharpe"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)  # type: ignore[no-untyped-call]
    return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """All pipeline configuration. Loaded from global_params.yaml with env var overrides."""

    # ── LLM configuration ────────────────────────────────────────────────
    llm_model: str = Field(default="gpt-5.4-nano", description="LLM model identifier")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Inference endpoint URL",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Env var OPENAI_API_KEY takes precedence.",
    )
    llm_think_mode: bool = Field(default=False, description="Enable chain-of-thought reasoning")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    llm_max_workers: int = Field(default=8, ge=1, le=32, description="Max concurrent LLM requests")
    llm_supports_structured_output: bool = Field(
        default=True,
        description="Whether the provider supports JSON schema response_format",
    )
    llm_max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens in LLM response")

    # ── Scoring prompt ───────────────────────────────────────────────────
    scoring_prompt: str = Field(
        default=(
            "You are a financial analyst evaluating earnings quality.\n\n"
            "Context:\n{context}\n\n"
            "Question: Based on the above, predict the {target} for next quarter.\n"
            "Respond with a single number between -1 (very negative) and +1 (very positive).\n"
            "Do not explain your reasoning."
        ),
        description="Prompt template with {context} and {target} placeholders",
    )

    # ── Scoring pipeline ─────────────────────────────────────────────────
    chunk_size: int = Field(default=30, ge=5, description="Sentences per transcript chunk")
    chunk_overlap: int = Field(default=0, ge=0, description="Overlapping sentences between chunks")
    neutering_enabled: bool = Field(default=False, description="Enable entity neutering")

    # ── Target variable ──────────────────────────────────────────────────
    target_variable: str = Field(
        default="oos_r_squared",
        description="Variable to predict / optimise",
    )
    target_direction: TargetDirection = Field(
        default=TargetDirection.higher_is_better,
        description="Whether higher target values are better",
    )

    # ── Feature engineering ──────────────────────────────────────────────
    feature_set: list[str] = Field(
        default=["fundamentals", "llm_scores", "macro", "sentiment"],
        description="Enabled feature groups",
    )
    winsorize_quantiles: list[float] = Field(
        default=[0.01, 0.99],
        description="Lower and upper quantile for winsorization",
    )

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_metric: EvalMetric = Field(
        default=EvalMetric.oos_r_squared,
        description="Primary scalar metric for autoresearch",
    )
    eval_horizon_days: int = Field(default=30, gt=0, description="Forward return horizon in days")
    random_seed: int = Field(default=42, description="Fixed seed for reproducibility")

    # ── Universe ─────────────────────────────────────────────────────────
    universe_filter: str = Field(
        default="True",
        description="Polars filter expression for the stock universe",
    )
    min_history_quarters: int = Field(
        default=8,
        ge=1,
        description="Minimum quarters of earnings history per ticker",
    )

    # ── Data splits ──────────────────────────────────────────────────────
    train_test_split_year: int = Field(
        default=2023,
        description="Year threshold for train/test split",
    )
    exclude_fyear_gte: int = Field(
        default=2026,
        description="Exclude fiscal years >= this value",
    )

    @field_validator("scoring_prompt")
    @classmethod
    def _validate_prompt_placeholders(cls, v: str) -> str:
        if "{context}" not in v:
            msg = "scoring_prompt must contain {context} placeholder"
            raise ValueError(msg)
        if "{target}" not in v:
            msg = "scoring_prompt must contain {target} placeholder"
            raise ValueError(msg)
        return v

    @field_validator("winsorize_quantiles")
    @classmethod
    def _validate_quantiles(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            msg = "winsorize_quantiles must have exactly 2 elements"
            raise ValueError(msg)
        if not (0 < v[0] < v[1] < 1):
            msg = "winsorize_quantiles must satisfy 0 < lower < upper < 1"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _resolve_api_key(self) -> Settings:
        """Prefer OPENAI_API_KEY env var over YAML value."""
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            self.openai_api_key = env_key
        return self


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from a YAML file, with env var overrides.

    Parameters
    ----------
    config_path:
        Path to the YAML config file. Defaults to ``configs/global_params.yaml``.

    Returns
    -------
    Settings
        Fully validated settings instance.
    """
    path = config_path or _DEFAULT_CONFIG
    yaml_data = _load_yaml(path)
    return Settings(**yaml_data)
