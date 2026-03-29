# Migration Log

Tracks every file moved from `_source/` (zarreda/finako) to autofinako.

## Stage 2 — Scoring pipeline migration (2026-03-29)

| Source file | Destination file | Changes |
|-------------|-----------------|---------|
| `src/pipeline/scoring/config.py` | `src/pipeline/scoring/constants.py` + `src/pipeline/settings.py` | Split: CATEGORIES/SENTIMENTS → constants.py; LLMProviderConfig/ScoringConfig replaced by unified Settings loaded from global_params.yaml |
| `src/pipeline/scoring/schemas.py` | `src/pipeline/scoring/schemas.py` | Verbatim copy |
| `src/pipeline/scoring/prompts.py` | `src/pipeline/scoring/prompts.py` | Verbatim copy (10 prompt constants) |
| `src/pipeline/scoring/chunker.py` | `src/pipeline/scoring/chunker.py` | Extracted MIN_CHUNK_LENGTH constant |
| `src/pipeline/scoring/neutering.py` | `src/pipeline/scoring/neutering.py` | Removed type: ignore on spacy import (handled by mypy overrides) |
| `src/pipeline/scoring/score_computer.py` | `src/pipeline/scoring/score_computer.py` | Updated imports: config → constants |
| `src/pipeline/scoring/llm_client.py` | `src/pipeline/llm.py` | Replaced LLMProviderConfig with Settings; removed Anthropic path (OpenAI-only via openai SDK); added think_mode support; parameterised max_tokens |
| `src/pipeline/scoring/llm_pipeline.py` | `src/pipeline/scoring/llm_pipeline.py` | Replaced ScoringConfig with Settings; replaced LLMClient import → pipeline.llm; replaced config.llm.max_workers with settings.llm_max_workers; replaced config.neutering with settings.neutering_enabled |
| `src/pipeline/scoring/lm_pipeline.py` | `src/pipeline/scoring/lm_pipeline.py` | Verbatim copy |
| `src/pipeline/scoring/transcript.py` | `src/pipeline/ingest/transcript.py` | Moved to ingest module; replaced hardcoded DB_NAME/SCHEMA_NAME/SNOWFLAKE_ACCOUNT with env vars with defaults; removed type: ignore comments |
| `tests/unit/test_scoring.py` | `tests/unit/test_scoring.py` | Adapted all tests to use Settings instead of ScoringConfig/LLMProviderConfig; added test classes for LLM client think mode, settings validation |

### New files (not from source)

| File | Purpose |
|------|---------|
| `tests/unit/test_settings.py` | 12 tests for Settings loading, validation, env var override |
| `tests/unit/test_experiment.py` | 6 tests verifying experiment.py signatures are intact |
| `tests/unit/test_evaluate.py` | 3 tests for evaluation harness (metric output, timeout) |
| `src/pipeline/scoring/constants.py` | CATEGORIES and SENTIMENTS extracted from config.py |
