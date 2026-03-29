# CLAUDE.md — Generalized Earnings Prediction Research Platform

This file is the authoritative guide for Claude Code in this repository.
Read it fully before taking any action. Follow every instruction precisely.

---

## 1. What this project is

This project generalises the earnings prediction pipeline from `/ad/code/finako` into a
**configurable, shareable research platform** with two operational modes:

**Interactive mode** — a human changes `configs/global_params.yaml` and reruns the pipeline.
All key levers (LLM model, scoring prompt, target variable, feature set) live there.
No Python editing required.

**Autonomous mode** — Claude Code runs the autoresearch loop described in `program.md`.
It edits `src/experiment.py` (the single mutable file), evaluates against the fixed
metric in `src/evaluate.py`, commits improvements, reverts regressions, and repeats
until stopped. The human writes `program.md`; the agent handles execution.

Stack: Python 3.12, pandas / polars, scikit-learn, pydantic, mlx-lm or ollama,
pytest, GitHub Actions. Package manager: `uv`. Never use bare `pip install`.

---

## 2. Bootstrap from the previous repo

**Do not create any files or the new GitHub repo until this section is complete.**

```bash
# Step 1 — Clone the previous implemented pipeline (read-only reference)
gh repo clone zarreda/finako _source --depth 1

# Step 2 — Read _source fully. Produce docs/SOURCE_AUDIT.md.
#   Read every file: pipeline stages, configs, tests, notebooks, data fixtures.
#   For each stage note: inputs, outputs, LLM calls, scoring logic, metrics.
#   Flag hardcoded values that should become global params -> docs/PARAMS_CANDIDATES.md

# Step 3 — Produce docs/GENERALISATION_PLAN.md (see §3). STOP. Wait for approval.

# --- HUMAN APPROVES GENERALISATION_PLAN.md BEFORE ANYTHING BELOW RUNS ---

# Step 4 — Create the new public-shareable repo
gh repo create zarreda/autofinako --public --clone
cd autofinako

# Step 5 — Bootstrap
uv init --python 3.12
uv add pandas polars scikit-learn pydantic pydantic-settings \
        mlx-lm httpx rich click typer
uv add --dev pytest pytest-cov pytest-mock hypothesis ruff mypy

# Step 6 — Migrate and generalise source into the structure in §4
#   Log every file moved in docs/MIGRATION_LOG.md

# Step 7 — First commit
git add . && git commit -m "chore: initial scaffold from prev repo generalisation"
git push -u origin main
```

---

## 3. Generalisation plan (produce before any code)

`docs/GENERALISATION_PLAN.md` must contain:

### Parameter extraction
For every hardcoded value found in `_source/` that a user might reasonably want to change:
- Parameter name (snake_case, becomes a key in `global_params.yaml`)
- Current hardcoded value
- Type and validation rules
- Which pipeline stage(s) it affects

Mandatory parameters that must appear in `global_params.yaml`:

| Key | Type | Description |
|-----|------|-------------|
| `scoring_prompt` | str (multiline) | The prompt template sent to the LLM for scoring. Must contain `{context}` and `{target}` placeholders. |
| `llm_model` | str | Ollama model tag or mlx-lm HuggingFace path (e.g. `qwen3.5:35b-a3b`) |
| `llm_base_url` | str | Local inference endpoint (default `http://localhost:11434/v1`) |
| `llm_think_mode` | bool | Whether to prepend `/think` to prompts for chain-of-thought scoring |
| `target_variable` | str | Column name of the prediction target (e.g. `eps_surprise_pct`) |
| `target_direction` | enum | `higher_is_better` or `lower_is_better` |
| `feature_set` | list[str] | Which feature groups to enable (e.g. `[fundamentals, llm_scores, macro]`) |
| `eval_metric` | enum | Primary metric for autoresearch: `ic`, `icir`, `hit_rate`, `sharpe` |
| `eval_horizon_days` | int | Forward return horizon for evaluation |
| `universe_filter` | str | Polars expression to filter the stock universe |
| `min_history_quarters` | int | Minimum earnings history required per ticker |
| `random_seed` | int | Fixed seed for all stochastic operations (reproducibility) |

### Stage mapping
Show which `_source/` modules map to which generalised stage in the new `src/pipeline/`.

### Abstraction boundaries
For each stage: what is fixed (the harness, in `src/evaluate.py`) vs.
what the agent may modify (the implementation, in `src/experiment.py`).

### Sharability checklist
List every change needed to make the repo shareable with no internal data:
- data fixtures to anonymise or synthesise
- credentials to move to env vars
- absolute paths to parameterise

### Approval gate
```
## Approval

[ ] Human has reviewed and approved this plan.

Approved: <leave blank>
Changes requested: <leave blank>
```

---

## 4. Project structure

```
<NEW_REPO>/
├── CLAUDE.md                        <- this file
├── program.md                       <- autoresearch instructions (human edits this)
├── pyproject.toml
├── uv.lock
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yml                   <- lint + unit + integration on every PR
│       └── autoresearch.yml         <- scheduled overnight experiment loop
│
├── configs/
│   └── global_params.yaml           <- THE primary user interface (see §5)
│
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── settings.py              <- Pydantic BaseSettings loading global_params.yaml
│       ├── llm.py                   <- all LLM calls centralised here
│       │
│       ├── <stage-N>/               <- generalised stages from prev repo
│       │   ├── __init__.py
│       │   └── *.py
│       │
│       ├── evaluate.py              <- FIXED. Autoresearch evaluator. Never modified by agent.
│       │                               Loads global_params.yaml, runs the full pipeline,
│       │                               returns a single scalar metric.
│       │
│       └── experiment.py            <- EDITABLE by agent only. Feature engineering,
│                                       scoring logic, model stacking.
│                                       This is the autoresearch sandbox.
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   └── test_<stage>.py
│   └── integration/
│       ├── test_pipeline_e2e.py
│       └── test_autoresearch_loop.py   <- verifies evaluate.py returns a valid scalar
│
├── data/
│   ├── fixtures/                    <- synthetic data for tests (committed, no real data)
│   └── .gitkeep
│
├── notebooks/
│   └── exploration/                 <- never imported by src/
│
├── results/
│   ├── experiments.tsv              <- autoresearch run log (not committed)
│   └── .gitkeep
│
└── docs/
    ├── SOURCE_AUDIT.md
    ├── PARAMS_CANDIDATES.md
    ├── GENERALISATION_PLAN.md
    ├── MIGRATION_LOG.md
    └── ARCHITECTURE.md
```

---

## 5. Global parameters — the user interface

`configs/global_params.yaml` is the only file a non-developer needs to edit.
Every key must have an inline comment explaining what it does and what values are valid.

Template to produce:

```yaml
# ─────────────────────────────────────────────────────────────────────────────
# Earnings Prediction Research Platform — Global Parameters
# Edit this file to change any aspect of the pipeline without touching Python.
# ─────────────────────────────────────────────────────────────────────────────

# ── LLM configuration ────────────────────────────────────────────────────────
# Which local model to use for scoring. Must be running via Ollama or mlx_lm.
# Examples: "qwen3.5:35b-a3b", "llama3.3:70b", "qwen3:32b"
llm_model: "qwen3.5:35b-a3b"

# Inference endpoint. Default works with both Ollama and mlx_lm server.
llm_base_url: "http://localhost:11434/v1"

# Enable chain-of-thought reasoning. Adds /think prefix to prompts.
# Better quality, ~3x slower. Recommended for final scoring runs.
llm_think_mode: false

# ── Scoring prompt ────────────────────────────────────────────────────────────
# The prompt template sent to the LLM for each company/quarter.
# Required placeholders: {context} (earnings text), {target} (what to predict).
# The agent may modify this in experiment.py. This is the human-authored baseline.
scoring_prompt: |
  You are a financial analyst evaluating earnings quality.

  Context:
  {context}

  Question: Based on the above, predict the {target} for next quarter.
  Respond with a single number between -1 (very negative) and +1 (very positive).
  Do not explain your reasoning.

# ── Target variable ───────────────────────────────────────────────────────────
# Column name of the variable to predict. Must exist in your dataset.
# Common choices: eps_surprise_pct, revenue_surprise_pct, fwd_return_1m
target_variable: "eps_surprise_pct"

# Whether higher values of target_variable are better (positive alpha)
target_direction: "higher_is_better"   # or "lower_is_better"

# ── Feature engineering ───────────────────────────────────────────────────────
# Which feature groups to enable. Disabling groups speeds up iteration.
feature_set:
  - fundamentals        # PE, revenue growth, margins, etc.
  - llm_scores          # LLM-generated scores (requires LLM to be running)
  - macro               # interest rates, sector ETF returns
  - sentiment           # earnings call tone, analyst revision direction
  # - technical         # price momentum, volume signals (optional)

# ── Evaluation ────────────────────────────────────────────────────────────────
# Primary scalar metric used by autoresearch to decide keep vs. revert.
# Options: ic (information coefficient), icir, hit_rate, sharpe
eval_metric: "ic"

# Forward return horizon in calendar days
eval_horizon_days: 30

# Fixed random seed for all stochastic operations (ensures reproducibility)
random_seed: 42

# ── Universe ──────────────────────────────────────────────────────────────────
# Polars filter expression applied to the raw universe.
# Use "True" to include everything.
universe_filter: "market_cap_usd > 1e9 and avg_daily_volume_usd > 1e6"

# Minimum quarters of earnings history required per ticker
min_history_quarters: 8
```

`src/pipeline/settings.py` must load this file via Pydantic `BaseSettings`.
All pipeline stages import settings from here. Never call `os.environ` directly.

---

## 6. The autoresearch loop — three-file contract

Adapted from Karpathy's autoresearch (github.com/karpathy/autoresearch).
The original optimises LLM pretraining loss overnight on one GPU.
This project optimises earnings prediction IC overnight using a local LLM.
The loop mechanics are identical; only the metric and sandbox differ.

### The three files

| File | Who edits it | Purpose |
|------|-------------|---------|
| `src/evaluate.py` | Nobody — read-only | Fixed evaluation harness. Runs the full pipeline on held-out data, prints `METRIC: <float>`. Equivalent to autoresearch `prepare.py`. |
| `src/experiment.py` | Claude Code agent | The sandbox. Feature engineering, scoring, stacking. Every experiment is a diff here. Equivalent to autoresearch `train.py`. |
| `program.md` | Human only | Agent instructions: research directions, constraints, stopping criteria. Equivalent to autoresearch `program.md`. |

### Invariants that must never be broken

- `evaluate.py` is immutable during autoresearch. The agent must not touch it.
  If the evaluator is broken, the human fixes it and explicitly tells the agent.
- `experiment.py` must be syntactically valid Python after every edit.
  The agent runs `uv run ruff check src/experiment.py` before committing.
- Every experiment runs within a fixed wall-clock budget: `EVAL_TIME_BUDGET_SECONDS = 120`.
  This keeps experiments comparable regardless of feature set complexity.
- The metric from `evaluate.py` is always a single Python `float`. Higher is always better
  (negate metrics where lower is better inside `evaluate.py`).
- Results are recorded in `results/experiments.tsv` — never committed to git.
  Git history records only the winning code changes.

### The ratchet loop

```
1. Read program.md fully.
2. Create branch: git checkout -b autoresearch/<session-tag>
3. Establish baseline: uv run src/evaluate.py; record to results/experiments.tsv
4. Loop:
   a. Propose one focused hypothesis from program.md directions
   b. Edit experiment.py to implement it (one change per experiment)
   c. Run: uv run ruff check src/experiment.py  — must pass
   d. Run: uv run src/evaluate.py > run.log 2>&1
   e. Parse: grep "^METRIC:" run.log
   f. If metric improved  -> git commit -m "exp(<tag>): <hypothesis> -> metric=<val>"
      If equal or worse   -> git checkout src/experiment.py  (revert, try something else)
   g. Record in results/experiments.tsv regardless of outcome
   h. If >3 consecutive crashes -> stop and report to human
5. On session end: print summary (N experiments, M improvements, best metric, top hypotheses)
```

### evaluate.py contract

`evaluate.py` must:
- Accept no CLI arguments (all config from `global_params.yaml`)
- Print exactly one line of the form `METRIC: <float>` to stdout
- Exit 0 on success, non-zero on any failure
- Complete within `EVAL_TIME_BUDGET_SECONDS`
- Use only the held-out test split (never the training split)
- Be deterministic (uses `random_seed` from `global_params.yaml`)

### experiment.py contract

`experiment.py` must expose exactly these two public functions with fixed signatures:

```python
def build_features(df: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Add feature columns to df. Must not modify or drop the target_variable column."""
    ...

def score(context: str, settings: Settings) -> float:
    """Score a single earnings context string. Returns float in [-1.0, 1.0]."""
    ...
```

The agent may freely add helper functions, import new libraries (via `uv add`),
and restructure internals — but must not change these two signatures.

---

## 7. Research directions for program.md (starting agenda)

The human populates `program.md` with these directions and expands over time.
The agent picks one direction per experiment, implements it, and measures.

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
- Long/short IC instead of long-only IC
- Decay-weighted IC (recent quarters count more)
- Sector-neutral IC (demean scores within each GICS sector)
- Multi-horizon average IC (30d, 60d, 90d)
- Hit rate on direction (binary: beat vs miss)

---

## 8. Coding standards

- **Style**: Ruff. `make lint` before every commit.
- **Types**: Full annotations on all public functions. `mypy src/` — zero errors.
- **Imports**: Absolute only (`from pipeline.settings import Settings`).
- **Config**: All config injected via `Settings`. Never `os.environ` in business logic.
- **Data frames**: Polars preferred. Pandas only when a dependency requires it.
- **LLM calls**: All LLM interactions go through `src/pipeline/llm.py`. Nothing else calls the LLM.
- **Secrets**: No credentials in code. Use env vars. Flag any found in `_source/`.
- **Logging**: `logging` module only. No `print()` in `src/` except the `METRIC:` line in `evaluate.py`.

---

## 9. Testing rules

### Unit tests
- Every public function in `src/pipeline/` (excluding `experiment.py`) has a unit test.
- `experiment.py` has tests verifying that `build_features` and `score` signatures are intact.
- Mock all LLM calls with `pytest-mock`. Never call a real LLM in tests.
- `hypothesis` property tests on feature transforms: output shape, no NaN propagation.
- Coverage target: ≥ 85% per module. Check with `make test-unit`.

### Integration tests
- `test_pipeline_e2e.py`: full ingestion → scoring → evaluation on fixture data.
- `test_autoresearch_loop.py`: verifies `evaluate.py` prints `METRIC: <float>` and exits 0 on fixture data.
- No real external services. Stub LLM with a fixture scorer returning deterministic floats.
- Mark with `@pytest.mark.integration`.

```bash
make test-unit          # fast
make test-integration   # full pipeline on fixture data (~60s)
make test               # both + coverage report
make eval               # run evaluate.py on fixture data (smoke test)
```

---

## 10. Makefile

```makefile
.PHONY: install lint test-unit test-integration test eval

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

test-unit:
	uv run pytest tests/unit/ -q --cov=src --cov-fail-under=85

test-integration:
	uv run pytest tests/integration/ -m integration -q

test: lint test-unit test-integration

eval:
	uv run src/evaluate.py
```

---

## 11. GitHub Actions

### ci.yml — every PR
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv run ruff check src/ tests/
      - run: uv run mypy src/

  unit:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv run pytest tests/unit/ --cov=src --cov-report=xml -q
      - uses: codecov/codecov-action@v4

  integration:
    runs-on: ubuntu-latest
    needs: unit
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv run pytest tests/integration/ -m integration -q
```

### autoresearch.yml — overnight autonomous research
```yaml
name: Autoresearch
on:
  schedule:
    - cron: '0 22 * * 1-5'   # weeknights at 22:00
  workflow_dispatch:

jobs:
  research:
    runs-on: self-hosted      # needs local LLM access
    timeout-minutes: 480      # 8 hours max
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - name: Run autoresearch session
        run: |
          claude --no-permissions . \
            --prompt "Read program.md and run a full autoresearch session."
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: experiment-results-${{ github.run_id }}
          path: results/experiments.tsv
```

---

## 12. Commit message convention

Conventional Commits: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `test`, `refactor`, `ci`, `docs`, `chore`, `exp`
Scopes: use approved stage names, plus `config`, `eval`, `exp`

Autoresearch experiments use:
`exp(<session>): <hypothesis> -> metric=<value>`

Example: `exp(mar29a): sector-neutral IC + analyst revision momentum -> ic=0.0921`

One logical change per commit. Never commit with failing tests.
Never commit `results/experiments.tsv`.

---

## 13. Step-by-step implementation plan

### Stage 0-A — Audit
- [ ] Clone `_source`, read every file, produce `SOURCE_AUDIT.md` and `PARAMS_CANDIDATES.md`
- [ ] List all hardcoded values that should become global parameters

### Stage 0-B — Generalisation plan
- [ ] Produce `GENERALISATION_PLAN.md` with parameter table, stage mapping, and sharability checklist
- [ ] **STOP. Wait for explicit human approval before any code is written.**

### Stage 1 — Scaffold and config
- [ ] Create new repo; bootstrap structure from §4
- [ ] Implement `configs/global_params.yaml` with all parameters from §5
- [ ] Implement `src/pipeline/settings.py` (Pydantic BaseSettings)
- [ ] Commit: `chore: scaffold and global params config`

### Stage 2 — Migrate pipeline stages
- [ ] Port each stage from `_source/`, replacing hardcoded values with `settings.*`
- [ ] Unit tests per stage → `make test-unit` passes
- [ ] Commit per stage: `feat(<stage>): generalised with settings injection`

### Stage 3 — Fixed evaluator
- [ ] Implement `src/evaluate.py` with the `METRIC:` contract from §6
- [ ] Verify on fixture data: `make eval`
- [ ] Write `tests/integration/test_autoresearch_loop.py`
- [ ] Commit: `feat(eval): fixed evaluation harness`

### Stage 4 — Baseline experiment.py
- [ ] Port `_source/` scoring logic into `experiment.py` with the two required signatures
- [ ] Write unit tests verifying the signatures are intact
- [ ] Commit: `feat(exp): baseline experiment.py from source`

### Stage 5 — program.md and first manual loop
- [ ] Populate `program.md` from §7 directions plus the autoresearch loop spec from §6
- [ ] Run one full manual experiment cycle: eval baseline → edit → eval → compare
- [ ] Commit: `docs: program.md initial research agenda`

### Stage 6 — CI/CD
- [ ] Push `ci.yml` and `autoresearch.yml`
- [ ] Verify CI passes on first PR
- [ ] Commit: `ci: GitHub Actions CI and autoresearch workflow`

---

## 14. What Claude Code must NOT do

- Do not run `pip install` — use `uv add` or `uv run`.
- Do not modify `src/evaluate.py` during an autoresearch session. Ever.
- Do not change the public signatures of `build_features` or `score` in `experiment.py`.
- Do not commit `results/experiments.tsv`.
- Do not commit real financial data, internal tickers, or credentials.
- Do not call a real LLM endpoint in tests — always mock it.
- Do not proceed past Stage 0-B without explicit human approval of `GENERALISATION_PLAN.md`.
- Do not skip `ruff check` before committing an experiment.
- Do not continue the autoresearch loop after 3 consecutive crashes — stop and report.
- Do not import from `notebooks/` in `src/`.
- Do not use `print()` in `src/` except the single `METRIC:` line in `evaluate.py`.
