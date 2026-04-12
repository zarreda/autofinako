"""Experiment logger — writes to results/experiments.tsv.

Used by the autoresearch loop to record every experiment attempt,
whether it improved the metric or not.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "results" / "experiments.tsv"

FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "hypothesis",
    "metric_before",
    "metric_after",
    "metric_delta",
    "improved",
    "status",
    "duration_s",
    "commit_sha",
    "session_tag",
    "notes",
]


def log_experiment(
    *,
    experiment_id: int,
    hypothesis: str,
    metric_before: float,
    metric_after: float,
    status: str = "success",
    duration_s: float = 0.0,
    commit_sha: str = "",
    session_tag: str = "",
    notes: str = "",
) -> None:
    """Append one experiment record to results/experiments.tsv."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not _LOG_PATH.exists() or os.path.getsize(_LOG_PATH) == 0

    delta = metric_after - metric_before
    improved = delta > 0

    row = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "metric_before": f"{metric_before:.6f}",
        "metric_after": f"{metric_after:.6f}",
        "metric_delta": f"{delta:+.6f}",
        "improved": str(improved),
        "status": status,
        "duration_s": f"{duration_s:.1f}",
        "commit_sha": commit_sha,
        "session_tag": session_tag,
        "notes": notes,
    }

    with open(_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
