"""Unit tests for the evaluation harness."""

from __future__ import annotations

import io
import sys

from pipeline.evaluate import main


class TestEvaluateHarness:
    def test_main_prints_metric(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            main()
        finally:
            sys.stdout = old_stdout
        output = buffer.getvalue().strip()
        assert "METRIC:" in output

    def test_main_metric_is_float(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            main()
        finally:
            sys.stdout = old_stdout
        output = buffer.getvalue().strip()
        lines = [ln for ln in output.split("\n") if ln.startswith("METRIC:")]
        assert len(lines) == 1
        _, value = lines[0].split("METRIC:")
        metric = float(value.strip())
        assert isinstance(metric, float)

    def test_main_metric_is_nonzero(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            main()
        finally:
            sys.stdout = old_stdout
        output = buffer.getvalue().strip()
        line = [ln for ln in output.split("\n") if ln.startswith("METRIC:")][0]
        metric = float(line.split("METRIC:")[1].strip())
        assert metric != 0.0, "Baseline should produce non-zero metric"
