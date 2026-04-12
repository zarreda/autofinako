"""Autoresearch Dashboard — visualize how experiments evolve.

Run: uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_PATH = _PROJECT_ROOT / "results" / "experiments.tsv"
_FIXTURE_PATH = _PROJECT_ROOT / "data" / "fixtures" / "sample_experiments.tsv"


def load_experiments(path: Path) -> list[dict[str, str]]:
    """Load experiments TSV into a list of dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _truncate(text: str, max_len: int = 50) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Autoresearch Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Autoresearch Dashboard")
st.markdown("Real-time view of the autonomous experiment loop.")

# ── Data source selector ────────────────────────────────────────────────

data_source = st.sidebar.radio(
    "Data source",
    ["Live results", "Sample data (demo)"],
    index=0 if _RESULTS_PATH.exists() else 1,
)

if data_source == "Live results":
    experiments = load_experiments(_RESULTS_PATH)
    if not experiments:
        st.info(
            "No experiments recorded yet. Run the autoresearch loop to generate data, "
            "or switch to **Sample data** in the sidebar."
        )
        st.stop()
else:
    experiments = load_experiments(_FIXTURE_PATH)

st.sidebar.markdown(f"**{len(experiments)} experiments loaded**")

# ── Session filter ──────────────────────────────────────────────────────

sessions = sorted({e.get("session_tag", "") for e in experiments if e.get("session_tag")})
if sessions:
    selected_sessions = st.sidebar.multiselect(
        "Filter by session",
        sessions,
        default=sessions,
    )
    experiments = [e for e in experiments if e.get("session_tag") in selected_sessions]

# ── Metric evolution ────────────────────────────────────────────────────

st.header("Metric Evolution")

ids = [int(e["experiment_id"]) for e in experiments]
metrics_after = [safe_float(e["metric_after"]) for e in experiments]
improved = [e.get("improved", "False") == "True" for e in experiments]

# Ratchet: best metric so far
best_so_far = []
current_best = 0.0
for m, imp in zip(metrics_after, improved, strict=False):
    if imp and m > current_best:
        current_best = m
    best_so_far.append(current_best)

col1, col2, col3, col4 = st.columns(4)
with col1:
    best_val = f"{max(metrics_after):.4f}" if metrics_after else "N/A"
    st.metric("Best Metric", best_val)
with col2:
    n_improved = sum(improved)
    st.metric("Improvements", f"{n_improved}/{len(experiments)}")
with col3:
    win_rate = n_improved / len(experiments) * 100 if experiments else 0
    st.metric("Win Rate", f"{win_rate:.0f}%")
with col4:
    total_gain = max(metrics_after) - metrics_after[0] if len(metrics_after) > 1 else 0
    st.metric("Total Gain", f"+{total_gain:.4f}")

# Chart: all experiments + ratchet
chart_data = {
    "Experiment #": ids,
    "Metric (each run)": metrics_after,
    "Best so far (ratchet)": best_so_far,
}

st.line_chart(
    chart_data,
    x="Experiment #",
    y=["Metric (each run)", "Best so far (ratchet)"],
    color=["#4A90D9", "#2ECC71"],
    height=350,
)

# ── Improvement waterfall ───────────────────────────────────────────────

st.header("Improvement Breakdown")

winning_exps = [e for e in experiments if e.get("improved") == "True"]
if winning_exps:
    waterfall_data = {
        "Hypothesis": [_truncate(e["hypothesis"]) for e in winning_exps],
        "Metric Gain": [safe_float(e["metric_delta"]) for e in winning_exps],
    }
    st.bar_chart(waterfall_data, x="Hypothesis", y="Metric Gain", color="#2ECC71", height=350)
else:
    st.info("No winning experiments yet.")

# ── Failed experiments ──────────────────────────────────────────────────

st.header("Failed Experiments (Reverted)")

failed_exps = [e for e in experiments if e.get("improved") == "False"]
if failed_exps:
    failed_data = {
        "Hypothesis": [_truncate(e["hypothesis"]) for e in failed_exps],
        "Metric Loss": [abs(safe_float(e["metric_delta"])) for e in failed_exps],
    }
    st.bar_chart(failed_data, x="Hypothesis", y="Metric Loss", color="#E74C3C", height=300)
else:
    st.success("No failed experiments!")

# ── Experiment timeline ─────────────────────────────────────────────────

st.header("Experiment Timeline")

col_map = {
    "#": "experiment_id",
    "Timestamp": "timestamp",
    "Hypothesis": "hypothesis",
    "Before": "metric_before",
    "After": "metric_after",
    "Delta": "metric_delta",
    "Improved": "improved",
    "Duration (s)": "duration_s",
    "Session": "session_tag",
    "Notes": "notes",
}

table_data = []
for e in experiments:
    row = {}
    for display_name, field_name in col_map.items():
        val = e.get(field_name, "")
        if field_name in ("metric_before", "metric_after"):
            val = f"{safe_float(val):.6f}"
        elif field_name == "metric_delta":
            v = safe_float(val)
            val = f"{v:+.6f}"
        elif field_name == "improved":
            val = "✅" if val == "True" else "❌"
        row[display_name] = val
    table_data.append(row)

st.dataframe(table_data, use_container_width=True, height=400)

# ── Session comparison ──────────────────────────────────────────────────

if len(sessions) > 1:
    st.header("Session Comparison")

    session_stats = {}
    for session in sessions:
        sess_exps = [e for e in experiments if e.get("session_tag") == session]
        if not sess_exps:
            continue
        sess_metrics = [safe_float(e["metric_after"]) for e in sess_exps]
        sess_improved = sum(1 for e in sess_exps if e.get("improved") == "True")
        session_stats[session] = {
            "Experiments": len(sess_exps),
            "Best Metric": f"{max(sess_metrics):.4f}",
            "Win Rate": f"{sess_improved / len(sess_exps) * 100:.0f}%",
            "Total Gain": (f"{max(sess_metrics) - safe_float(sess_exps[0]['metric_before']):+.4f}"),
        }

    st.dataframe(
        [{"Session": k, **v} for k, v in session_stats.items()],
        use_container_width=True,
    )

# ── Cumulative efficiency ───────────────────────────────────────────────

st.header("Research Efficiency")

if len(experiments) > 1:
    cumulative_time: list[float] = []
    t = 0.0
    for e in experiments:
        t += safe_float(e.get("duration_s", "0"))
        cumulative_time.append(t)

    efficiency_data = {
        "Cumulative Time (s)": cumulative_time,
        "Best Metric": best_so_far,
    }
    st.line_chart(
        efficiency_data,
        x="Cumulative Time (s)",
        y="Best Metric",
        color="#8E44AD",
        height=300,
    )

    gain_per_min = total_gain / max(0.01, cumulative_time[-1] / 60)
    st.markdown(f"""
    | Stat | Value |
    |------|-------|
    | Total experiments | {len(experiments)} |
    | Total time | {cumulative_time[-1]:.0f}s ({cumulative_time[-1] / 60:.1f} min) |
    | Avg time per experiment | {cumulative_time[-1] / len(experiments):.1f}s |
    | Experiments per improvement | {len(experiments) / max(1, sum(improved)):.1f} |
    | Metric gain per minute | {gain_per_min:.4f} |
    """)

# ── Sidebar info ────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use**

1. Run `make eval` to establish baseline
2. Run autoresearch loop (Claude edits `experiment.py`)
3. This dashboard auto-refreshes from `results/experiments.tsv`
4. Switch to **Sample data** to see a demo

**Metrics explained**
- **OOS R-squared**: Out-of-sample R-squared (higher = better)
- **Win Rate**: % of experiments that improved the metric
- **Ratchet**: Only keeps improvements, reverts regressions
""")
