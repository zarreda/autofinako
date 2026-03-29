"""Unit tests for the evaluation harness."""

from __future__ import annotations

from unittest.mock import patch

from pipeline.evaluate import main


class TestEvaluateHarness:
    def test_main_prints_metric(self, capsys: object) -> None:
        main()
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "METRIC:" in captured.out

    def test_main_metric_is_float(self, capsys: object) -> None:
        main()
        captured = capsys.readouterr()  # type: ignore[union-attr]
        line = captured.out.strip()
        _, value = line.split("METRIC:")
        float(value.strip())  # should not raise

    def test_main_exits_on_timeout(self) -> None:
        with patch("pipeline.evaluate.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 200.0]

            with __import__("pytest").raises(SystemExit):
                main()
