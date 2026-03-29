# Autoresearch Program — Earnings Prediction

This file is the agent's instruction manual. The human edits it;
the agent reads it and executes experiments accordingly.

---

## Objective

Maximise the **out-of-sample R²** (`oos_r_squared`) of the best earnings
prediction model. The evaluator (`src/pipeline/evaluate.py`) runs the full
pipeline on held-out data and prints `METRIC: <float>`. Higher is better.

---

## Research directions

### Prompt engineering
- Vary the analyst role ("financial analyst" / "skeptical short-seller" / "quant PM")
- Add few-shot examples of confirmed beat / miss quarters before the context
- Ask for structured JSON output: `{"score": float, "confidence": float, "key_factors": list}`
- Chain-of-thought: ask model to reason, then extract score from `<answer>` tag
- Ask for a probability distribution (P(beat), P(inline), P(miss)) then convert to score
- Multi-turn: first ask to summarise, then ask to score the summary

### Feature engineering
- Rolling z-scores of fundamentals over 4 / 8 / 12 quarter windows
- Sector-relative normalisation of all numeric features
- Analyst estimate revision momentum (delta of consensus over 30 / 60 / 90 days)
- LLM score × analyst surprise interaction term
- Earnings call transcript sentiment decomposed by speaker (CEO vs CFO)
- Options implied volatility as uncertainty proxy (discount high-IV names)
- Sequential quarter delta features (current vs prior quarter for key ratios)

### Score aggregation / model stacking
- Ensemble: average scores from 2-3 different prompt templates
- Calibration: fit isotonic regression on LLM scores vs realised surprises
- Stacking: linear model on [llm_score, fundamentals_zscore, revision_momentum]
- Bayesian update: prior from fundamentals model, likelihood update from LLM score
- Cross-sectional ranking instead of raw scores (rank IC vs Pearson IC)

### Evaluation enhancements
- Long/short R² instead of long-only
- Decay-weighted OOS R² (recent quarters count more)
- Sector-neutral R² (demean scores within each GICS sector)
- Multi-horizon average (30d, 60d, 90d)

---

## Constraints

- One change per experiment. Do not bundle multiple hypotheses.
- Each experiment must complete within 120 seconds (EVAL_TIME_BUDGET_SECONDS).
- Run `uv run ruff check src/experiment.py` before committing.
- If >3 consecutive crashes, stop and report.

---

## Stopping criteria

- Session time exceeds 8 hours
- No improvement in last 10 consecutive experiments
- Human sends a stop signal
