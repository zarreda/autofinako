# Parameter Candidates for `global_params.yaml`

Produced: 2026-03-29 (Stage 0-A for autofinako)

Every hardcoded value in `_source/` that a user might reasonably want to change,
mapped to a `global_params.yaml` key.

---

## Mandatory parameters (from CLAUDE.md §5)

These are required by the autofinako spec and must appear in `global_params.yaml`.

| Parameter | Type | Current hardcoded value | Source file(s) | Validation | Pipeline stage(s) affected |
|-----------|------|------------------------|----------------|------------|---------------------------|
| `scoring_prompt` | str (multiline) | 5 separate prompts in `prompts.py` (~8000 chars total) | `scoring/prompts.py` | Must contain `{context}` and `{target}` placeholders | scoring (all 5 LLM stages) |
| `llm_model` | str | `"qwen3.5-35b-a3b"` | `scoring/config.py:55` | Non-empty string | scoring |
| `llm_base_url` | str | `"http://127.0.0.1:1234/v1"` | `scoring/config.py:49` | Valid URL | scoring |
| `llm_think_mode` | bool | Not implemented | N/A | bool | scoring |
| `target_variable` | str | `"EPS_year_Tplus1"` (first of 3 in `dev.yaml`) | `configs/dev.yaml:12` | Must exist as column name | features, modeling, evaluation |
| `target_direction` | enum | Not set (implied `higher_is_better`) | N/A | `higher_is_better` or `lower_is_better` | evaluation |
| `feature_set` | list[str] | Not implemented | N/A | Subset of known groups | features |
| `eval_metric` | enum | Not implemented | N/A | `ic`, `icir`, `hit_rate`, `sharpe` | evaluation |
| `eval_horizon_days` | int | `30` (implied from notebooks) | notebooks | > 0 | evaluation |
| `universe_filter` | str | Not implemented | N/A | Valid Polars expression | ingest |
| `min_history_quarters` | int | Not set (notebooks use ad-hoc filtering) | notebooks | ≥ 1 | ingest |
| `random_seed` | int | Not set | N/A | Any int | all stochastic operations |

---

## Additional parameters discovered in source

These are hardcoded values that should be extracted into `global_params.yaml` for configurability.

### LLM configuration

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `llm_temperature` | float | `0.1` | `scoring/config.py:56` | 0.0–2.0 | scoring |
| `llm_max_workers` | int | `8` | `scoring/config.py:57` | 1–32 | scoring |
| `llm_supports_structured_output` | bool | `False` | `scoring/config.py:58` | bool | scoring |
| `llm_max_tokens` | int | `4096` | `scoring/llm_client.py:179` | > 0 | scoring |
| `llm_api_key_env` | str | `"lm-studio"` (or env var name) | `scoring/config.py:44` | Non-empty | scoring |

### Scoring pipeline

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `chunk_size` | int | `30` sentences | `scoring/config.py:69` | ≥ 5 | scoring |
| `chunk_overlap` | int | `0` sentences | `scoring/config.py:70` | ≥ 0 | scoring |
| `neutering_enabled` | bool | `False` | `scoring/config.py:71` | bool | scoring |
| `min_chunk_length` | int | `50` chars | `scoring/chunker.py:35` | > 0 | scoring |

### Scoring categories and sentiments

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `categories` | list[str] | 7 financial categories | `scoring/config.py:74-82` | Non-empty list | scoring, evaluation |
| `sentiments` | list[str] | `["positive", "negative", "neutral"]` | `scoring/config.py:84` | Non-empty list | scoring, evaluation |

### Data access (Nuvolos/Snowflake)

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `db_name` | str | `"essec_metalab/ako_earnings_prediction"` | `scoring/transcript.py:22` | Non-empty | ingest (transcript retrieval) |
| `schema_name` | str | `"master/development"` | `scoring/transcript.py:23` | Non-empty | ingest |
| `snowflake_account` | str | `"alphacruncher.eu-central-1"` | `scoring/transcript.py:24` | Non-empty | ingest |

**Recommendation:** Keep these as env vars rather than `global_params.yaml` since they are infrastructure-specific, not research parameters.

### Modeling configuration (from `configs/dev.yaml`)

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `targets` | list[str] | `[EPS_year_Tplus1, EBITDA_Growth_YoY, EBEXOI_Growth_YoY]` | `configs/dev.yaml:12-14` | Non-empty; each must be a valid column | modeling, evaluation |
| `winsorize_quantiles` | list[float] | `[0.01, 0.99]` | `configs/dev.yaml:15` | 2-element, 0 < q1 < q2 < 1 | features |
| `train_test_split_year` | int | `2023` | `configs/dev.yaml:16` | Valid year in data range | features, modeling |
| `exclude_fyear_gte` | int | `2026` | `configs/dev.yaml:6` | Valid year | ingest |
| `data_source_dir` | str | `"data/"` | `configs/dev.yaml:4` | Valid path | ingest |
| `data_fixtures_dir` | str | `"data/fixtures/"` | `configs/dev.yaml:5` | Valid path | testing |
| `batch_size` | int | `5000` | `configs/dev.yaml:14` (scoring section) | > 0 | scoring (batch mode) |

### Serving configuration (from `configs/dev.yaml`)

| Parameter | Type | Current value | Source file | Validation | Stage(s) |
|-----------|------|---------------|-------------|------------|----------|
| `serving_host` | str | `"0.0.0.0"` | `configs/dev.yaml:19` | Valid host | serving |
| `serving_port` | int | `8000` | `configs/dev.yaml:20` | Valid port | serving |

---

## Parameters NOT recommended for `global_params.yaml`

These are implementation details that should remain hardcoded:

| Value | Reason to keep hardcoded |
|-------|--------------------------|
| Score formulas (linear, log) | Mathematical definitions, not user preferences |
| Transcript priority order `(8, 2, 1)` | Snowflake-specific data quality ordering |
| Component type IDs `[2, 4]` | Snowflake-specific table encoding |
| Event type ID `48` | Snowflake-specific (Earnings Calls) |
| Presentation type `5` | Snowflake-specific (Final transcript) |
| spaCy model name `"en_core_web_sm"` | Implementation detail of neutering |
| Entity type → placeholder mapping | Implementation detail of neutering |

---

## Mapping to autoresearch contract

For the autoresearch loop, the key split is:

**In `global_params.yaml` (human-tunable):**
- `scoring_prompt`, `llm_model`, `llm_base_url`, `llm_think_mode`
- `target_variable`, `target_direction`, `feature_set`
- `eval_metric`, `eval_horizon_days`
- `universe_filter`, `min_history_quarters`, `random_seed`

**In `experiment.py` (agent-tunable):**
- Prompt variations (different roles, few-shot examples, structured output)
- Feature engineering logic (z-scores, interactions, sector-relative normalization)
- Score aggregation (ensembling, calibration, stacking)
- These override or extend `global_params.yaml` values programmatically

**In `evaluate.py` (fixed):**
- Metric computation (IC, ICIR, hit_rate, sharpe)
- Data loading and splitting
- The `METRIC: <float>` output contract
