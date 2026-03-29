"""Unit tests for pipeline.settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.settings import EvalMetric, Settings, TargetDirection, load_settings


class TestLoadSettings:
    def test_load_from_default_yaml(self) -> None:
        s = load_settings()
        assert s.llm_model == "gpt-5.4-nano"
        assert s.llm_base_url == "https://api.openai.com/v1"
        assert s.target_variable == "oos_r_squared"
        assert s.target_direction == TargetDirection.higher_is_better
        assert s.eval_metric == EvalMetric.oos_r_squared
        assert s.random_seed == 42
        assert s.chunk_size == 30
        assert s.llm_supports_structured_output is True

    def test_load_from_custom_yaml(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {
            "llm_model": "test-model",
            "llm_base_url": "http://localhost:1234/v1",
            "target_variable": "eps_surprise_pct",
            "eval_metric": "ic",
            "random_seed": 123,
            "scoring_prompt": "Score {context} for {target}.",
        }
        yaml_path = tmp_path / "test_params.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        s = load_settings(yaml_path)
        assert s.llm_model == "test-model"
        assert s.target_variable == "eps_surprise_pct"
        assert s.eval_metric == EvalMetric.ic
        assert s.random_seed == 123

    def test_load_from_missing_file(self, tmp_path: Path) -> None:
        s = load_settings(tmp_path / "nonexistent.yaml")
        assert s.llm_model == "gpt-5.4-nano"  # all defaults


class TestSettingsValidation:
    def test_prompt_must_have_context_placeholder(self) -> None:
        with pytest.raises(ValueError, match="context"):
            Settings(scoring_prompt="No placeholders here for {target}.")

    def test_prompt_must_have_target_placeholder(self) -> None:
        with pytest.raises(ValueError, match="target"):
            Settings(scoring_prompt="Score this {context} please.")

    def test_valid_prompt_passes(self) -> None:
        s = Settings(scoring_prompt="Evaluate {context} for {target}.")
        assert "{context}" in s.scoring_prompt

    def test_winsorize_quantiles_must_be_two_elements(self) -> None:
        with pytest.raises(ValueError, match="2 elements"):
            Settings(winsorize_quantiles=[0.01])

    def test_winsorize_quantiles_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="lower < upper"):
            Settings(winsorize_quantiles=[0.99, 0.01])

    def test_temperature_bounds(self) -> None:
        Settings(llm_temperature=0.0)
        Settings(llm_temperature=2.0)
        with pytest.raises(ValueError):
            Settings(llm_temperature=-0.1)
        with pytest.raises(ValueError):
            Settings(llm_temperature=2.1)

    def test_chunk_size_minimum(self) -> None:
        with pytest.raises(ValueError):
            Settings(chunk_size=2)

    def test_eval_metric_enum(self) -> None:
        s = Settings(eval_metric="sharpe")
        assert s.eval_metric == EvalMetric.sharpe

    def test_invalid_eval_metric(self) -> None:
        with pytest.raises(ValueError):
            Settings(eval_metric="invalid_metric")

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
        s = Settings(openai_api_key="yaml-value")
        assert s.openai_api_key == "sk-test-from-env"
