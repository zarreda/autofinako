"""Generate synthetic fixture data for testing the evaluation pipeline.

Run: uv run data/fixtures/generate_fixtures.py

Produces sample_earnings.csv — a synthetic panel of 20 companies x 20 quarters
with sentiment scores, fundamentals, and target variables that have realistic
statistical structure (so the evaluator can produce non-trivial metrics).
"""

from __future__ import annotations

import numpy as np
import polars as pl

SEED = 42
N_COMPANIES = 20
YEARS = range(2019, 2025)
QUARTERS = [1, 2, 3, 4]

CATEGORIES = [
    "REVENUE",
    "INDUSTRY_MOATS_DRIVERS",
    "EARNING_AND_COSTS",
    "CAP_ALLOCATION_CASH",
    "EXOGENOUS",
    "MANAGEMENT_CULTURE_SUSTAINABILITY",
    "OTHER_CRITERIA",
]


def main() -> None:
    rng = np.random.RandomState(SEED)

    rows = []
    for cid in range(1, N_COMPANIES + 1):
        # Company-level params
        base_eps = rng.uniform(2.0, 15.0)
        growth_rate = rng.uniform(-0.02, 0.08)
        sentiment_beta = rng.uniform(0.5, 2.0)  # how much sentiment affects next-Q EPS

        prev_eps = base_eps
        for yr in YEARS:
            for q in QUARTERS:
                # Sentiment scores (simulate LLM output)
                neg_frac = np.clip(rng.normal(0.35, 0.15), 0.05, 0.95)
                pos_frac = np.clip(rng.normal(0.40, 0.15), 0.05, 0.95)
                neu_frac = 1.0 - neg_frac - pos_frac
                if neu_frac < 0:
                    neu_frac = 0.0
                    total = neg_frac + pos_frac
                    neg_frac /= total
                    pos_frac /= total

                linear_score = pos_frac - neg_frac

                # Per-category scores
                cat_scores = {}
                for cat in CATEGORIES:
                    cat_neg = np.clip(rng.normal(neg_frac, 0.1), 0.0, 1.0)
                    cat_pos = np.clip(rng.normal(pos_frac, 0.1), 0.0, 1.0)
                    n_sent = rng.randint(2, 15)
                    n_pos = int(n_sent * cat_pos)
                    n_neg = int(n_sent * cat_neg)
                    n_neu = n_sent - n_pos - n_neg
                    if n_neu < 0:
                        n_neu = 0
                    cat_scores[cat] = {
                        "pos": n_pos,
                        "neg": n_neg,
                        "neu": n_neu,
                        "neg_frac": cat_neg,
                    }

                # LM dictionary score (correlated but noisier)
                lm_neg_frac = np.clip(neg_frac + rng.normal(0, 0.1), 0.0, 1.0)

                # Fundamentals
                revenue_growth = rng.normal(growth_rate, 0.03)
                margin = np.clip(rng.normal(0.15, 0.05), 0.01, 0.50)
                pe_ratio = rng.uniform(8, 40)

                # Enhanced scoring features
                avg_confidence = rng.uniform(2.0, 4.5)
                strong_frac = rng.uniform(0.1, 0.5)
                guidance_count = rng.randint(0, 8)
                avg_specificity = rng.uniform(1.5, 4.0) if guidance_count > 0 else 0.0
                near_future_frac = rng.uniform(0.3, 0.7)
                far_future_frac = rng.uniform(0.05, 0.3)

                # Target: EPS with signal from sentiment + fundamentals + noise
                eps = (
                    prev_eps * (1 + growth_rate / 4)
                    + sentiment_beta * linear_score
                    + 0.3 * revenue_growth * prev_eps
                    + rng.normal(0, 0.5)
                )
                prev_eps = eps

                # Forward return (30-day) — correlated with sentiment
                fwd_return = (
                    0.02 * linear_score
                    + 0.01 * revenue_growth
                    + rng.normal(0, 0.03)
                )

                row = {
                    "COMPANYID": cid,
                    "COMPANYNAME": f"Company_{cid:03d}",
                    "FYEAR": yr,
                    "FQUARTER": q,
                    "EPS": round(eps, 4),
                    "fwd_return_30d": round(fwd_return, 6),
                    "revenue_growth": round(revenue_growth, 4),
                    "operating_margin": round(margin, 4),
                    "pe_ratio": round(pe_ratio, 2),
                    "global_neg_frac": round(neg_frac, 4),
                    "global_pos_frac": round(pos_frac, 4),
                    "global_linear_score": round(linear_score, 4),
                    "LM_neg_frac": round(lm_neg_frac, 4),
                    "avg_confidence": round(avg_confidence, 2),
                    "strong_sentiment_frac": round(strong_frac, 4),
                    "guidance_count": guidance_count,
                    "avg_specificity": round(avg_specificity, 2),
                    "near_future_frac": round(near_future_frac, 4),
                    "far_future_frac": round(far_future_frac, 4),
                    "transcript_text": (
                        f"Company_{cid:03d} Q{q} {yr}: Revenue grew {revenue_growth*100:.1f}% "
                        f"with margins at {margin*100:.1f}%. Management expects continued "
                        f"{'improvement' if linear_score > 0 else 'challenges'} ahead."
                    ),
                }

                for cat in CATEGORIES:
                    row[f"{cat}_pos"] = cat_scores[cat]["pos"]
                    row[f"{cat}_neg"] = cat_scores[cat]["neg"]
                    row[f"{cat}_neu"] = cat_scores[cat]["neu"]
                    row[f"{cat}_neg_frac"] = round(cat_scores[cat]["neg_frac"], 4)

                rows.append(row)

    df = pl.DataFrame(rows)

    # Add target variable: OOS R² proxy — use next-quarter EPS surprise
    df = df.sort(["COMPANYID", "FYEAR", "FQUARTER"])
    df = df.with_columns(
        pl.col("EPS").shift(-1).over("COMPANYID").alias("next_EPS"),
    )
    df = df.with_columns(
        ((pl.col("next_EPS") - pl.col("EPS")) / pl.col("EPS").abs().clip(0.01, None))
        .alias("eps_surprise_pct")
    )
    df = df.drop("next_EPS")

    out_path = "data/fixtures/sample_earnings.csv"
    df.write_csv(out_path)

    print(f"Generated {df.height} rows x {df.width} columns -> {out_path}")
    print(f"Companies: {df['COMPANYID'].n_unique()}")
    print(f"Years: {df['FYEAR'].min()}-{df['FYEAR'].max()}")
    print(f"Columns: {df.columns}")


if __name__ == "__main__":
    main()
