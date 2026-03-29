# Generalisation Plan

Produced: 2026-03-29 (Stage 0-B for autofinako)

---

## Parameter extraction

Every hardcoded value in `_source/` that should become a key in `global_params.yaml`.

### Mandatory parameters (required by CLAUDE.md §5)

| Parameter name | Current hardcoded value | Type & validation | Source file(s) | Pipeline stage(s) affected |
|----------------|------------------------|-------------------|----------------|---------------------------|
| `scoring_prompt` | 5 prompt templates totalling ~8000 chars; the extraction prompt (`EXTRACTION_SYSTEM` + `EXTRACTION_USER`) is the primary scoring lever | `str` (multiline); must contain `{context}` and `{target}` placeholders | `scoring/prompts.py` | scoring | e.g. `"You are a financial analyst evaluating earnings quality.\n\nContext:\n{context}\n\nQuestion: Based on the above, predict the {target}..."` |
| `llm_model` | `"gpt-5.4-nano"` | `str`; non-empty | `scoring/config.py:55` | scoring | e.g. `"gpt-5.4-nano"`, `"gpt-4.1"`, `"claude-sonnet-4-20250514"` |
| `llm_base_url` | `"https://api.openai.com/v1"` | `str`; valid URL | `scoring/config.py:49` | scoring | e.g. `"https://api.openai.com/v1"`, `"http://localhost:11434/v1"` (ollama) |
| `openai_api_key` | None (must be provided) | `str`; non-empty. Can also be set via `OPENAI_API_KEY` env var. | `scoring/config.py:44` | scoring | e.g. `"sk-proj-..."` |
| `llm_think_mode` | Not implemented (new) | `bool` | N/A | scoring | `false` (default) or `true` for chain-of-thought |
| `target_variable` | `"oos_r_squared"` — the out-of-sample R² of the best model | `str`; must exist as column or be a recognized evaluation metric | `configs/dev.yaml:12` | features, modeling, evaluation | e.g. `"oos_r_squared"`, `"eps_surprise_pct"`, `"fwd_return_1m"` |
| `target_direction` | `higher_is_better` (higher R² = better model) | `enum`: `higher_is_better` \| `lower_is_better` | N/A | evaluation | `"higher_is_better"` for R², IC; `"lower_is_better"` for RMSE, MAE |
| `feature_set` | Not implemented (new) | `list[str]`; subset of `[fundamentals, llm_scores, macro, sentiment, technical]` | N/A | features | e.g. `[fundamentals, llm_scores, sentiment]` |
| `eval_metric` | Not implemented (new) | `enum`: `ic`, `icir`, `hit_rate`, `sharpe`, `oos_r_squared` | N/A | evaluation | e.g. `"oos_r_squared"` (default), `"ic"` |
| `eval_horizon_days` | `30` (implied from notebook analysis) | `int`; > 0 | notebooks | evaluation | e.g. `30`, `60`, `90` |
| `universe_filter` | Not implemented (new) | `str`; valid Polars expression | N/A | ingest | e.g. `"market_cap_usd > 1e9 and avg_daily_volume_usd > 1e6"`, `"True"` (all) |
| `min_history_quarters` | Not set (notebooks use ad-hoc filtering) | `int`; ≥ 1 | notebooks | ingest | e.g. `8`, `4` |
| `random_seed` | Not set | `int` | N/A | all stochastic operations | e.g. `42` |

### Additional parameters to extract

| Parameter name | Current hardcoded value | Type & validation | Source file(s) | Stage(s) | Examples |
|----------------|------------------------|-------------------|----------------|----------|----------|
| `llm_temperature` | `0.1` | `float`; 0.0–2.0 | `scoring/config.py:56` | scoring | `0.1` (deterministic), `0.7` (creative), `1.0` (high variance) |
| `llm_max_workers` | `8` | `int`; 1–32 | `scoring/config.py:57` | scoring | `4` (laptop), `8` (default), `16` (server) |
| `llm_supports_structured_output` | `False` | `bool` | `scoring/config.py:58` | scoring | `true` for GPT-5.4-nano / GPT-4.1; `false` for local models |
| `llm_max_tokens` | `4096` | `int`; > 0 | `scoring/llm_client.py:179` | scoring | `2048`, `4096`, `8192` |
| `chunk_size` | `30` sentences | `int`; ≥ 5 | `scoring/config.py:69` | scoring | `15` (fine-grained), `30` (default), `50` (coarse) |
| `chunk_overlap` | `0` sentences | `int`; ≥ 0 | `scoring/config.py:70` | scoring | `0` (no overlap), `5`, `10` |
| `neutering_enabled` | `False` | `bool` | `scoring/config.py:71` | scoring | `false` (default), `true` (mask entities for robustness) |
| `winsorize_quantiles` | `[0.01, 0.99]` | `list[float]`; 2-element, 0 < q1 < q2 < 1 | `configs/dev.yaml:15` | features | `[0.01, 0.99]` (1%), `[0.05, 0.95]` (5%) |
| `train_test_split_year` | `2023` | `int`; valid year in data range | `configs/dev.yaml:16` | features, modeling | `2022`, `2023`, `2024` |
| `exclude_fyear_gte` | `2026` | `int` | `configs/dev.yaml:6` | ingest | `2026` (exclude partial year) |

### Values kept as environment variables (not in `global_params.yaml`)

These are infrastructure credentials / connection details, not research parameters.
Note: `openai_api_key` is in `global_params.yaml` (see mandatory params above) but can also be set via `OPENAI_API_KEY` env var as a fallback.

| Env var | Current hardcoded value | Source file |
|---------|------------------------|-------------|
| `NUVOLOS_USERNAME` | `"derrazab"` (leaked in notebook) | `scoring/transcript.py:81` |
| `NUVOLOS_SF_TOKEN` | `"XmX5VfUBu6c8G6Tu"` (leaked in notebook) | `scoring/transcript.py:82` |
| `SNOWFLAKE_RSA_KEY` | path to RSA key file | `scoring/transcript.py:106` |
| `SNOWFLAKE_ACCOUNT` | `"alphacruncher.eu-central-1"` | `scoring/transcript.py:24` |
| `DB_NAME` | `"essec_metalab/ako_earnings_prediction"` | `scoring/transcript.py:22` |
| `SCHEMA_NAME` | `"master/development"` | `scoring/transcript.py:23` |

---

## Stage mapping

How `_source/` modules map to the new `src/pipeline/` structure.

| New generalised stage | `_source/` module(s) | What migrates | What changes |
|-----------------------|---------------------|---------------|-------------|
| `src/pipeline/settings.py` | `scoring/config.py` (partial), `configs/dev.yaml` (unused) | `LLMProviderConfig`, `ScoringConfig`, YAML loading pattern | Expand to Pydantic `BaseSettings` that loads `global_params.yaml`; add all mandatory params from §5; replace separate config classes with single `Settings` |
| `src/pipeline/llm.py` | `scoring/llm_client.py` | `LLMClient`, `TokenUsage`, OpenAI + Anthropic dual-path | Centralise all LLM calls here; add think mode support; parameterise `max_tokens` from settings |
| `src/pipeline/scoring/` | `scoring/llm_pipeline.py`, `scoring/lm_pipeline.py`, `scoring/prompts.py`, `scoring/schemas.py`, `scoring/chunker.py`, `scoring/neutering.py`, `scoring/score_computer.py` | All 5 LLM stages, LM dictionary pipeline, prompts, schemas, chunking, neutering, score computation | Replace hardcoded prompts with `settings.scoring_prompt` override; inject `Settings` instead of `ScoringConfig`; prompts become the default that `experiment.py` can override |
| `src/pipeline/ingest/` | `scoring/transcript.py` (Nuvolos retrieval) + new CSV/JSON loaders | Transcript retrieval, SQL queries, priority filtering | Add CSV/JSON loaders for pre-scored data; add schema validation; add `universe_filter` and `exclude_fyear_gte` support |
| `src/pipeline/features/` | Notebook logic from `eps_target_2.ipynb` (cells 2–12) and `target_analysis.ipynb` | EPS/EBEXOI per-share metrics, growth rates, sentiment fractions, tertile bins, winsorization | Extract from notebooks into Python functions; wire to `Settings` (winsorize_quantiles, feature_set, train_test_split_year) |
| `src/pipeline/modeling/` | Notebook logic from `eps_target_2.ipynb` (`run_regression()`, cells 18–55) | Sequential Q4→Q3 regressions, LASSO, OLS with fixed effects | Extract from notebooks; parameterise target variable; add train/test evaluation |
| `src/pipeline/evaluate.py` | No direct equivalent — new | N/A | Build the fixed evaluation harness: load data → build features → score → compute metric → print `METRIC: <float>` |
| `src/pipeline/experiment.py` | Combines `scoring/prompts.py` (prompt engineering) + notebook feature engineering | Prompt templates, feature transforms, scoring logic | The single mutable file for autoresearch; exposes `build_features()` and `score()` |

---

## Abstraction boundaries

For each component: what is **fixed** (the harness, never modified by the agent) vs **mutable** (the sandbox the agent may freely edit).

### Fixed components (agent must not modify)

| Component | File | Why fixed |
|-----------|------|-----------|
| Evaluation harness | `src/pipeline/evaluate.py` | The metric contract — runs full pipeline, prints `METRIC: <float>`. If this changes mid-session, results aren't comparable. |
| Settings loader | `src/pipeline/settings.py` | Config injection mechanism must be stable for both evaluate.py and experiment.py. |
| LLM client | `src/pipeline/llm.py` | Provider abstraction is infrastructure, not research. |
| Data ingest | `src/pipeline/ingest/` | Data loading and validation must be deterministic and stable. |
| Score computation | `src/pipeline/scoring/score_computer.py` | Linear/log score formulas are definitions, not hypotheses. |
| Schemas | `src/pipeline/scoring/schemas.py` | Pydantic models define the data contract between stages. |
| All tests | `tests/` | Tests validate the harness, not the experiment. |

### Mutable components (agent sandbox)

| Component | File | What the agent can change |
|-----------|------|--------------------------|
| Feature engineering | `src/pipeline/experiment.py` → `build_features()` | Add/remove/transform feature columns; z-scores, interactions, sector-relative normalisation, rolling windows, tertile bins |
| Scoring logic | `src/pipeline/experiment.py` → `score()` | Prompt template variations, few-shot examples, chain-of-thought, structured JSON output, multi-turn scoring, ensemble averaging |
| Helper functions | `src/pipeline/experiment.py` (internal) | Any private functions, imports, or utilities the agent needs |
| Dependencies | `pyproject.toml` (via `uv add`) | Agent may add libraries needed for new features |

### The boundary contract

```
evaluate.py imports:
  - experiment.build_features(df, settings) → pl.DataFrame
  - experiment.score(context, settings) → float

experiment.py imports:
  - pipeline.settings.Settings (read-only)
  - pipeline.llm (for LLM calls)
  - Any library available in the environment

The agent MUST NOT:
  - Change the signatures of build_features() or score()
  - Modify or drop the target_variable column in build_features()
  - Return values outside [-1.0, 1.0] from score()
  - Call evaluate.py or modify it
```

---

## Sharability checklist

Every change needed to make the repo shareable with no internal data or credentials.

### Credentials to remove or move to env vars

| Item | Current location | Action |
|------|-----------------|--------|
| Nuvolos SF token `"XmX5VfUBu6c8G6Tu"` | `_source/notebooks/sentiment_scores_analysis.ipynb` cell 3 | **DELETE** — do not carry over. Use `os.environ["NUVOLOS_SF_TOKEN"]` |
| Nuvolos username `"derrazab"` | Same cell | **DELETE** — use `os.environ["NUVOLOS_USERNAME"]` |
| OpenAI API key | `scoring/config.py:44` default was `"lm-studio"` | Now a `global_params.yaml` parameter (`openai_api_key`) with `OPENAI_API_KEY` env var fallback. Never commit real keys — use a placeholder or env var in the checked-in YAML. |
| All `"YOUR_*"` placeholders | Original AKO repo `config.py` | Already removed in `_source/` refactor. Verify none carried over. |

### Data to anonymise or synthesise

| Data | Action |
|------|--------|
| Real earnings call transcripts | Do not commit. Create synthetic fixture transcripts for tests (~500 chars each, 3–5 company-quarters). |
| `eps_target_data.csv` (13,770 rows, 148 companies) | Do not commit. Create `data/fixtures/sample_earnings.csv` with 50 synthetic rows, same schema. |
| `capiq_quarter_data.csv` (13,770 rows) | Do not commit. Create synthetic fixture with 50 rows. |
| `eps_lm_sentiment.csv` (7,328 rows) | Do not commit. Create synthetic fixture with 30 rows. |
| LM dictionary CSV (~2,700 words) | The Loughran-McDonald dictionary is publicly available. Include a small subset (50 words) in `data/fixtures/` for tests. |
| `results/llm_scores/` (5,637 JSONs) | Do not carry over. Tests use mocked scoring. |
| Company lists (148 companies) | Do not commit real lists. Fixtures use synthetic company names. |
| DOCX write-ups | Do not carry over. These are presentation documents. |

### Absolute paths to parameterise

| Path | Current location | Action |
|------|-----------------|--------|
| `"essec_metalab/ako_earnings_prediction"` | `transcript.py:22` | Move to env var `DB_NAME` |
| `"master/development"` | `transcript.py:23` | Move to env var `SCHEMA_NAME` |
| `"alphacruncher.eu-central-1"` | `transcript.py:24` | Move to env var `SNOWFLAKE_ACCOUNT` |
| `".\Loughran-McDonlad_scores"` | Notebook (commented out) | Do not carry over |
| `"../data/"`, `"../result_output/"` | Notebooks | Do not carry over; use `settings.data_source_dir` |

### Other sharability items

| Item | Action |
|------|--------|
| `.gitignore` | Carry over with additions: `results/experiments.tsv`, `*.env` |
| `_source/` directory | Already in `.gitignore`. Add to new repo's `.gitignore` as well. |
| Notebook outputs (logged tokens, connection info) | Clear all outputs before committing any notebooks |
| `pyproject.toml` dependencies | Remove `nuvolos` and `snowflake-connector` from required deps; move to optional `[ingest]` extra since most users won't have Nuvolos access |

---

## Approval

[x] Human has reviewed and approved this plan.

Approved: All sections approved (2026-03-29)
Changes requested: none
