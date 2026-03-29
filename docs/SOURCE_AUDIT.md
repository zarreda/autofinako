# Source Audit — `_source/` (zarreda/finako)

Produced: 2026-03-29 (Stage 0-A for autofinako)

---

## 1. Repository overview

The `_source/` repo (`zarreda/finako`) is a **refactored earnings call sentiment pipeline** derived from the earlier `Hadrienvnt/AKO` repo. It implements two scoring methods:

1. **LLM-based 5-stage pipeline** — Extract → Deduplicate → Filter → Categorize → Sentiment → Score
2. **Loughran-McDonald dictionary pipeline** — deterministic word-matching against the LM financial dictionary

The repo also contains a **serving layer** (FastAPI stub), **modeling/evaluation/features/ingest stage placeholders** (empty `__init__.py` files), two Jupyter notebooks for end-to-end testing and time-series analysis, and a full test suite for the scoring module.

**Stack:** Python 3.12, OpenAI + Anthropic APIs, Nuvolos/Snowflake, pandas, polars, spaCy, Pydantic, FastAPI, pytest, ruff, mypy, uv.

---

## 2. File-by-file audit

### 2.1 Top-level files

| File | Purpose | Notes |
|------|---------|-------|
| `CLAUDE.md` | Project instructions for the _previous_ Claude Code project (finako) — defines bootstrap, pipeline discovery, audit process | Not applicable to autofinako; kept as reference |
| `pyproject.toml` | Package config: `finako` v0.1.0, Python ≥3.12 <3.13 | Dependencies: anthropic, fastapi, matplotlib, nuvolos, openai, pandas, polars, pydantic, scikit-learn, scipy, seaborn, statsmodels, uvicorn |
| `Makefile` | Standard targets: install, lint, test-unit, test-integration, test, serve | `serve` runs uvicorn on `pipeline.serving.app:app` |
| `.gitignore` | Ignores data CSVs/parquet, `_source/`, `src/local/`, `.env`, logs, batch_files, result_output | |
| `.python-version` | `3.12` | |
| `README.md` | Empty (1 line) | |

### 2.2 Configuration files

| File | Purpose | Key values |
|------|---------|------------|
| `configs/dev.yaml` | Dev environment config | `llm_model: gpt-4.1`, `temperature: 0.1`, `chunk_size: 30`, `overlap: 0`, `batch_size: 5000`, targets: `[EPS_year_Tplus1, EBITDA_Growth_YoY, EBEXOI_Growth_YoY]`, `winsorize_quantiles: [0.01, 0.99]`, `train_test_split_year: 2023`, serving on `0.0.0.0:8000` |
| `configs/prod.yaml` | Prod environment config | Identical to dev.yaml (not yet differentiated) |

**Note:** These configs are not loaded by any Python code — no `settings.py` exists yet. They were created as part of the scaffold but are not wired up.

### 2.3 `src/pipeline/scoring/` — The main implemented module

#### `config.py` — Pydantic configuration models

| Class | Fields | Defaults | Notes |
|-------|--------|----------|-------|
| `LLMProviderConfig` | api_key, base_url, model, temperature, max_workers, supports_structured_output | `"lm-studio"`, `"http://127.0.0.1:1234/v1"`, `"qwen3.5-35b-a3b"`, `0.1`, `8`, `False` | Supports OpenAI, Claude, DeepSeek, Kimi, ollama, vLLM, LM Studio |
| `ScoringConfig` | llm, chunk_size, chunk_overlap, neutering | `LLMProviderConfig()`, `30`, `0`, `False` | |
| `CATEGORIES` | 7-element list | REVENUE, INDUSTRY/MOATS/DRIVERS, EARNING & COSTS, CAP ALLOCATION/CASH, EXOGENOUS, MANAGEMENT/CULTURE/SUSTAINABILITY, OTHER CRITERIA | Hardcoded constant |
| `SENTIMENTS` | 3-element list | positive, negative, neutral | Hardcoded constant |

#### `llm_client.py` — Provider-agnostic LLM client

- `LLMClient` dataclass wrapping OpenAI and Anthropic SDK clients
- `_is_claude_model()` routes based on model name prefix
- `complete()` returns `(parsed_dict, input_tokens, output_tokens)`
- Supports structured output (OpenAI `json_schema` mode) and fallback (JSON instruction injection)
- Strips markdown fences from responses
- `TokenUsage` accumulator dataclass
- **Hardcoded:** `max_tokens=4096` for Anthropic calls

#### `prompts.py` — LLM prompts for 5 stages

| Stage | System prompt var | User prompt var | Approx length |
|-------|------------------|-----------------|---------------|
| 1. Extraction | `EXTRACTION_SYSTEM` | `EXTRACTION_USER` | ~3000 chars |
| 2. Redundancy | `REDUNDANCY_SYSTEM` | `REDUNDANCY_USER` | ~800 chars |
| 3. Filtration | `FILTRATION_SYSTEM` | `FILTRATION_USER` | ~900 chars |
| 4. Categorization | `CATEGORIZATION_SYSTEM` | `CATEGORIZATION_USER` | ~3500 chars |
| 5. Sentiment | `SENTIMENT_SYSTEM` | `SENTIMENT_USER` | ~800 chars |

- Prompts contain `{chunk_text}`, `{sentences}`, `{sentence}`, `{category}` placeholders
- These are the primary "levers" for the autoresearch loop
- `EXTRACTION_SYSTEM` is by far the most detailed (multi-pass methodology, inclusion/exclusion criteria)

#### `schemas.py` — Pydantic models for structured LLM output

| Schema | Stage | Fields |
|--------|-------|--------|
| `KeySentencesWithReason` | Extraction | List of `{sentence, reason}` |
| `KeptSentences` | Redundancy | List of `{number, text}` |
| `FutureStatement` | Filtration | `{text, classification, confidence, explanation}` |
| `CategorizedSentence` | Categorization | `{text, category, reason}` |
| `SentimentSentence` | Sentiment | `{text, sentiment, reason}` |
| `EarningsCallResult` | Final output | `{company_name, year, quarter, company_id, transcript_id, model_used, scores, sentences}` |
| `CategoryScore` | Score aggregation | `{positive, negative, neutral, linear_score, log_score}` |
| `ScoredSentence` | Per-sentence result | `{text, category, sentiment, explanation}` |

All schemas use `extra = "forbid"`. Category is a `Literal` of 7 values; Sentiment is `Literal["positive", "negative", "neutral"]`.

#### `llm_pipeline.py` — 5-stage LLM orchestrator

- `LLMScoringPipeline` class with `run()` method
- Each stage uses `ThreadPoolExecutor(max_workers=config.llm.max_workers)` for parallelism
- Stages: `extract()` → `remove_redundancy()` → `filter_future()` → `categorize()` → `assign_sentiment()`
- Score computation delegated to `score_computer.build_result()`
- Empty transcript returns empty result (not an error)

#### `lm_pipeline.py` — Loughran-McDonald dictionary scorer

- `LMScoringPipeline` class with `run()` method
- `load_lm_dictionary(path)` reads CSV, indexes by uppercase word → set of active categories
- `_tokenize()` strips non-alpha, uppercases
- Produces `sentiment_counts` (7 LM categories), `linear_score`, `log_score`
- **Requires:** Cleaned LM dictionary CSV (not included in repo; was in original AKO repo)

#### `chunker.py` — Transcript text splitter

- `chunk_transcript(text, chunk_size=30, overlap=0)` → list of sentence-based chunks
- Splits on `. ` (period-space), configurable size and overlap
- Filters chunks shorter than 50 characters

#### `neutering.py` — Entity masking with spaCy

- `neuter(sentence)` replaces 12 NER entity types with `[TOKEN]` placeholders
- Lazy-loads `en_core_web_sm` model on first use
- Handles overlapping spans, removes consecutive duplicate tokens
- **External dependency:** spaCy + en_core_web_sm model

#### `score_computer.py` — Score aggregation

- `compute_scores(sentences)` counts sentiments per category, produces `CategoryScore` objects
- Linear score: `(pos - neg) / (pos + neg)`
- Log score: `log((pos+1)/(neg+1)) / log(pos+neg+2)`
- `build_result()` creates complete `EarningsCallResult`

#### `transcript.py` — Nuvolos/Snowflake data access

- `connect_nuvolos()` with 3 auth methods: token, RSA key, browser SSO
- `get_transcript()` / `get_transcripts()` retrieve earnings call text
- SQL query joins 7 Snowflake tables, filters to earnings calls (type 48), final transcripts (type 5)
- Transcript reconstruction with priority filtering: Audited (8) > Edited (2) > Proofed (1)
- Keeps Main (2) and Q&A (4) sections only
- **Hardcoded:** `DB_NAME = "essec_metalab/ako_earnings_prediction"`, `SCHEMA_NAME = "master/development"`, `SNOWFLAKE_ACCOUNT = "alphacruncher.eu-central-1"`

### 2.4 Placeholder modules (empty `__init__.py`)

| Module | Status |
|--------|--------|
| `src/pipeline/evaluation/` | Empty — not implemented |
| `src/pipeline/features/` | Empty — not implemented |
| `src/pipeline/ingest/` | Empty — not implemented |
| `src/pipeline/modeling/` | Empty — not implemented |
| `src/pipeline/serving/` | Empty — not implemented |

### 2.5 Tests

| File | Coverage | Notes |
|------|----------|-------|
| `tests/unit/test_scoring.py` | Comprehensive | 519 lines; tests chunker, score_computer, LLM client (OpenAI + Claude + markdown stripping), token usage, neutering helpers, LM pipeline, config, schemas, full mocked LLM pipeline |
| `tests/conftest.py` | Empty | |
| `tests/integration/` | Empty | No integration tests implemented |

### 2.6 Notebooks

| Notebook | Purpose | Key observations |
|----------|---------|-----------------|
| `test_scoring_pipeline.ipynb` | End-to-end test: connect to Nuvolos, retrieve transcripts (Apple, Microsoft, Amazon), run LLM + LM pipelines, test neutering, test individual stages, save results | Tests all 5 stages individually; compares LLM vs LM; includes neutering test |
| `sentiment_scores_analysis.ipynb` | Time-series analysis: score 20 quarters of Apple transcripts (2020-2024), build time-series DataFrame, visualize sentiment trends, category breakdown, heatmap, LLM vs LM comparison | Produces exportable CSV + JSON |

### 2.7 Documentation (from previous audit)

| File | Purpose |
|------|---------|
| `docs/SOURCE_AUDIT.md` | Audit of the _original_ AKO repo (`Hadrienvnt/AKO`) — covers the un-refactored source |
| `docs/LOCAL_INVENTORY.md` | Inventory of local files (CSVs, notebooks, docx) from the previous project workspace |
| `docs/PIPELINE_PROPOSAL.md` | 6-stage pipeline proposal (scoring → ingest → features → modeling → evaluation → serving) — approved 2026-03-27 |

### 2.8 CI/CD

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Lint (ruff + mypy) → unit tests → integration tests on every PR |
| `.github/workflows/cd.yml` | Placeholder deploy step on push to main |

---

## 3. Hardcoded values, credentials, and secrets

### CRITICAL: Leaked credential

| Item | File | Severity |
|------|------|----------|
| **`sf_token="XmX5VfUBu6c8G6Tu"`** | `notebooks/sentiment_scores_analysis.ipynb` cell 3 | **HIGH — real Nuvolos/Snowflake token hardcoded in notebook** |
| `username="derrazab"` | Same cell | **MEDIUM — real username** |

### Hardcoded values to parameterize

| Item | File(s) | Current value | Should become |
|------|---------|---------------|---------------|
| LLM model | `config.py` | `"qwen3.5-35b-a3b"` | `global_params.yaml: llm_model` |
| LLM base_url | `config.py` | `"http://127.0.0.1:1234/v1"` | `global_params.yaml: llm_base_url` |
| LLM api_key | `config.py` | `"lm-studio"` | Env var |
| Temperature | `config.py` | `0.1` | `global_params.yaml` (new param) |
| Max workers | `config.py` | `8` | `global_params.yaml` (new param) |
| Structured output flag | `config.py` | `False` | `global_params.yaml` (new param) |
| Chunk size | `config.py` | `30` | `global_params.yaml` (new param) |
| Chunk overlap | `config.py` | `0` | `global_params.yaml` (new param) |
| Neutering enabled | `config.py` | `False` | `global_params.yaml` (new param) |
| Max tokens (Anthropic) | `llm_client.py:179` | `4096` | `global_params.yaml` (new param) |
| Min chunk length | `chunker.py:35` | `50` chars | Possibly configurable |
| CATEGORIES list | `config.py` | 7 hardcoded categories | Could be configurable for different taxonomies |
| SENTIMENTS list | `config.py` | 3 hardcoded sentiments | Fixed (unlikely to change) |
| DB_NAME | `transcript.py` | `"essec_metalab/ako_earnings_prediction"` | Env var or config |
| SCHEMA_NAME | `transcript.py` | `"master/development"` | Env var or config |
| SNOWFLAKE_ACCOUNT | `transcript.py` | `"alphacruncher.eu-central-1"` | Env var or config |
| Transcript priority order | `transcript.py` | `(8, 2, 1)` | Fixed (unlikely to change) |
| Component type IDs | `transcript.py` | `[2, 4]` (Main + Q&A) | Fixed |
| Event type ID | `transcript.py` | `48` (Earnings Calls) | Fixed |
| Transcript presentation type | `transcript.py` | `5` (Final) | Fixed |
| Scoring prompt templates | `prompts.py` | 5 large prompt strings | `global_params.yaml: scoring_prompt` (for the combined template) or kept as `prompts.py` with override capability |
| Linear score formula | `score_computer.py` | `(p-n)/(p+n)` | Fixed |
| Log score formula | `score_computer.py` | `log((p+1)/(n+1))/log(p+n+2)` | Fixed |
| Targets | `configs/dev.yaml` | `[EPS_year_Tplus1, EBITDA_Growth_YoY, EBEXOI_Growth_YoY]` | `global_params.yaml: target_variable` |
| Winsorize quantiles | `configs/dev.yaml` | `[0.01, 0.99]` | `global_params.yaml` (new param) |
| Train/test split year | `configs/dev.yaml` | `2023` | `global_params.yaml` (new param) |
| Exclude FYEAR | `configs/dev.yaml` | `>= 2026` | `global_params.yaml` (new param) |

---

## 4. Implicit dependencies between files

```
llm_pipeline.py
  ├── chunker.py           (chunk_transcript)
  ├── config.py            (ScoringConfig, LLMProviderConfig)
  ├── llm_client.py        (LLMClient, TokenUsage)
  ├── neutering.py         (neuter)  — optional, gated by config.neutering
  ├── prompts.py           (all 10 prompt constants)
  ├── schemas.py           (5 stage schemas)
  └── score_computer.py    (build_result)

llm_client.py
  ├── config.py            (LLMProviderConfig)
  ├── openai               (OpenAI SDK)  — lazy import
  └── anthropic            (Anthropic SDK)  — lazy import

lm_pipeline.py
  └── pandas               (for dictionary loading)

transcript.py
  ├── pandas
  ├── snowflake.connector  — lazy import
  └── nuvolos              — lazy import

neutering.py
  └── spacy                — lazy import (en_core_web_sm)

score_computer.py
  ├── config.py            (CATEGORIES, SENTIMENTS)
  └── schemas.py           (CategoryScore, EarningsCallResult, ScoredSentence)
```

---

## 5. Data sources referenced

| Source | Format | Access | Used by |
|--------|--------|--------|---------|
| Nuvolos DB (Snowflake) | SQL tables (7+ joined tables) | Token/RSA/SSO auth | `transcript.py` |
| OpenAI-compatible API | REST (chat completions) | API key | `llm_client.py` |
| Anthropic API | REST (messages) | API key | `llm_client.py` |
| LM dictionary CSV | CSV (Word + 7 category columns) | Local file | `lm_pipeline.py` — **not included in repo** |

---

## 6. Existing tests and validation

### Unit tests (`tests/unit/test_scoring.py`)

| Test class | Methods | What it covers |
|------------|---------|----------------|
| `TestChunker` | 5 tests | Basic chunking, overlap, short text, empty text |
| `TestScoreComputer` | 5 tests | Score computation, empty categories, all-pos/neg, build_result |
| `TestLLMClient` | 4 tests | Structured output, fallback JSON, Claude path, markdown stripping |
| `TestTokenUsage` | 1 test | Accumulation |
| `TestNeuteringHelpers` | 3 tests | Apply replacements, multiple entities, duplicate token removal |
| `TestLMPipeline` | 3 tests | Tokenizer, dictionary loading, full LM scoring |
| `TestConfig` | 4 tests | Default config, custom provider, constants |
| `TestSchemas` | 1 test | EarningsCallResult serialization |
| `TestLLMPipeline` | 2 tests | Full mocked pipeline, empty transcript |

All LLM calls are mocked. spaCy model is not loaded (only helper functions tested).

### Integration tests

None implemented. `tests/integration/` is empty. CI workflow uses `|| test $? -eq 5` to allow "no tests collected" exit code.

---

## 7. What maps to the autoresearch three-file contract

| autofinako concept | Source equivalent | Mapping notes |
|--------------------|-------------------|---------------|
| `evaluate.py` (fixed harness) | No direct equivalent | Must be built new; will import from scoring pipeline + features + modeling |
| `experiment.py` (`build_features` + `score`) | `prompts.py` (scoring templates) + `llm_pipeline.py` (orchestration) + notebook feature engineering | `score()` maps to the LLM pipeline's single-transcript scoring; `build_features()` maps to notebook feature creation logic |
| `program.md` | No equivalent | New file |
| `global_params.yaml` | `configs/dev.yaml` (partial) + `config.py` defaults | Must be expanded with all parameters from §5 of CLAUDE.md |
| `settings.py` (Pydantic BaseSettings) | `scoring/config.py` (partial — covers only LLM + scoring config) | Must be expanded to load `global_params.yaml` and cover all pipeline stages |

---

## 8. Gaps and risks for generalisation

1. **Only scoring is implemented** — ingest, features, modeling, evaluation, serving are all empty stubs.
2. **No `settings.py`** that loads `global_params.yaml` — config classes exist but aren't wired to YAML.
3. **LM dictionary not included** — `lm_pipeline.py` requires a CSV that isn't in the repo.
4. **Leaked credential in notebook** — must not be carried over.
5. **Prompts are frozen** — no mechanism to override prompts via config; they're Python constants.
6. **No evaluation metric** — the `METRIC: <float>` contract doesn't exist yet.
7. **Feature engineering lives in notebooks** — must be extracted into `experiment.py` / `build_features()`.
8. **No train/test split infrastructure** — `dev.yaml` references `train_test_split_year: 2023` but no code uses it.
9. **The 3 target variables** from `dev.yaml` need to be reconciled with CLAUDE.md's single `target_variable` pattern (the autoresearch loop works on one target at a time).
