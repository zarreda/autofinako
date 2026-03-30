"""Integration test: verify evaluate.py prints METRIC: <float> and exits 0."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.integration
def test_evaluate_prints_metric() -> None:
    """Run evaluate.py as a subprocess and verify the METRIC output contract."""
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.evaluate"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Must exit 0
    assert result.returncode == 0, (
        f"evaluate.py exited with code {result.returncode}\nstderr: {result.stderr}"
    )

    # Must print exactly one METRIC line to stdout
    stdout_lines = result.stdout.strip().split("\n")
    metric_lines = [ln for ln in stdout_lines if ln.startswith("METRIC:")]
    assert len(metric_lines) == 1, (
        f"Expected 1 METRIC line, got {len(metric_lines)}: {stdout_lines}"
    )

    # Must be a valid float
    _, value_str = metric_lines[0].split("METRIC:")
    metric_value = float(value_str.strip())
    assert isinstance(metric_value, float)


@pytest.mark.integration
def test_evaluate_completes_within_budget() -> None:
    """Verify evaluation completes within the 120s time budget."""
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.evaluate"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0


@pytest.mark.integration
def test_evaluate_is_deterministic() -> None:
    """Two runs with the same seed must produce the same metric."""
    results = []
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-m", "pipeline.evaluate"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0
        line = [ln for ln in r.stdout.strip().split("\n") if ln.startswith("METRIC:")][0]
        results.append(float(line.split("METRIC:")[1].strip()))

    assert results[0] == results[1], f"Non-deterministic: {results[0]} != {results[1]}"
