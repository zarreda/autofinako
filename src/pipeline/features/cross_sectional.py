"""Cross-sectional and cross-category features.

These features compare a company's sentiment against its peers or across
its own categories — capturing relative positioning and internal consistency.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def sector_relative_sentiment(
    df: pl.DataFrame,
    col: str = "global_neg_frac",
    output: str = "sector_relative_sentiment",
) -> pl.DataFrame:
    """Sentiment relative to the cross-sectional median in each quarter.

    ``relative[t] = neg_frac[t] - median(neg_frac across all companies in same quarter)``

    Positive = more negative than peers. Negative = more optimistic than peers.
    """
    return df.with_columns(
        (pl.col(col) - pl.col(col).median().over(["FYEAR", "FQUARTER"])).alias(output)
    )


def sentiment_dispersion(
    df: pl.DataFrame,
    output: str = "sentiment_dispersion",
) -> pl.DataFrame:
    """Standard deviation across the 7 category neg_frac values.

    High dispersion = mixed signals across business areas (e.g., revenue
    positive but costs negative). Low dispersion = uniform tone.
    """
    cat_cols = [
        c
        for c in df.columns
        if c.endswith("_neg_frac")
        and not c.startswith("global")
        and not c.startswith("LM_")
        and not c.startswith("NEU_")
        and not c.startswith("sector")
        and not c.startswith("sentiment")
    ]
    if not cat_cols:
        logger.warning("No category neg_frac columns found")
        return df

    return df.with_columns(
        pl.concat_list(cat_cols).list.eval(pl.element().std()).list.first().alias(output)
    )


def category_concentration(
    df: pl.DataFrame,
    output: str = "category_concentration",
) -> pl.DataFrame:
    """Herfindahl index of sentence counts across 7 categories.

    ``HHI = sum((count_i / total)^2)``

    HHI = 1/7 = 0.143 -> perfectly even. HHI = 1.0 -> all sentences in one category.
    High concentration may signal a crisis or singular opportunity in one area.
    """
    pos_cols = [c for c in df.columns if c.endswith("_pos") and not c.startswith("global")]
    neg_cols = [c for c in df.columns if c.endswith("_neg") and not c.startswith("global")]

    if not pos_cols or not neg_cols:
        logger.warning("No per-category pos/neg columns found")
        return df

    total_exprs = []
    cat_total_names = []
    for p, n in zip(pos_cols, neg_cols, strict=False):
        cat_name = p.replace("_pos", "_total_hhi")
        total_exprs.append((pl.col(p) + pl.col(n)).alias(cat_name))
        cat_total_names.append(cat_name)

    df = df.with_columns(total_exprs)

    grand_total = pl.sum_horizontal(cat_total_names).alias("_grand_total_hhi")
    df = df.with_columns(grand_total)

    share_sq_exprs = [
        (pl.col(ct) / (pl.col("_grand_total_hhi") + 1e-9)) ** 2 for ct in cat_total_names
    ]

    df = df.with_columns(pl.sum_horizontal(share_sq_exprs).alias(output))
    return df.drop(cat_total_names + ["_grand_total_hhi"])


def llm_lm_disagreement(
    df: pl.DataFrame,
    output: str = "llm_lm_disagreement",
) -> pl.DataFrame:
    """Absolute difference between LLM and LM negative fractions.

    When the two methods disagree, the LLM is likely capturing contextual
    nuance that the dictionary misses. High disagreement quarters may be
    the ones where LLM sentiment is most informative.
    """
    if "LM_neg_frac" not in df.columns or "global_neg_frac" not in df.columns:
        logger.info("LM_neg_frac or global_neg_frac not present — skipping disagreement")
        return df

    return df.with_columns((pl.col("global_neg_frac") - pl.col("LM_neg_frac")).abs().alias(output))


def build_cross_sectional_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all cross-sectional and cross-category features."""
    df = sector_relative_sentiment(df)
    df = sentiment_dispersion(df)
    df = category_concentration(df)
    df = llm_lm_disagreement(df)
    logger.info("Built 4 cross-sectional features")
    return df
