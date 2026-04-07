# Dynamic IC

A walk-forward strategy that replaces the fixed Information Coefficient (IC = 0.05)
used in the static momentum baseline with a **dynamically estimated IC function**.
Rather than assuming `IC(z) = constant`, we train online models to learn the shape
of the relationship between signal z-scores and realised returns — and update that
estimate every day as new data arrives.

## Motivation

The standard Grinold formula prices each stock's expected return as:

```
alpha_i = IC × σ_i × z_i
```

where `IC` is held constant. In practice the IC varies over time and, crucially,
may vary **with the z-score itself** — the signal may be more predictive at
extremes (`|z| > 2`) than near the middle, or it may flip sign in certain regimes.
This project learns that shape online.

## Pipeline

```
compute.py  →  train.py  →  analyze.py
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `compute.py` | `results/{split}/z_scores/{signal}.parquet` |
| 2 | `train.py` | `results/{split}/alphas/{signal}/{model}.parquet` + MVO weights |
| 3 | `analyze.py` | Summary table, performance charts, IC function evolution plots |

### Fire-and-forget on the cluster

`run.py` submits all three phases to Slurm in one command and then exits.
You receive a failure email if any step fails, and a completion email when
analysis finishes.

```
Phase 1: job array (0-7)   — compute.py  per signal          4 CPU / 32G / 30 min
Phase 2: job array (0-39)  — train.py    per signal×model   16 CPU / 64G /  3 hr
Phase 3: single job        — analyze.py  for the split       4 CPU / 32G / 30 min
```

Each phase starts only after **all tasks** of the previous phase succeed.

## Signals

Eight signals are pre-computed and cached as cross-sectional z-scores:

| Signal | Source | Description |
|--------|--------|-------------|
| `style_momentum` | Barra style factor returns | 11-to-1 month vol-scaled momentum mapped via exposures |
| `industry_momentum` | Barra industry factor returns | Same methodology, industry factors |
| `idiosyncratic_momentum` | `specific_return` (assets) | Vol-scaled rolling momentum on idio returns |
| `style_reversal` | Barra style factor returns | 21-day reversal (negated momentum) |
| `industry_reversal` | Barra industry factor returns | Same, industry factors |
| `idiosyncratic_reversal` | `specific_return` | 21-day idio reversal |
| `idiosyncratic_volatility` | `specific_risk` | Low-vol anomaly: negative specific risk |
| `betting_against_beta` | `predicted_beta` | BAB: negative predicted beta |

All signals output `(date, barrid, raw_signal)`. The pipeline z-scores them
cross-sectionally across all stocks on each date.

## IC Models

Four dynamic models plus a static baseline, all in `models/`:

| Model | Key | Complexity |
|-------|-----|------------|
| `static` | Fixed IC = 0.05 for all z | Baseline |
| `kalman_poly` | State-space polynomial — state is `[β0, β1, β2]`, `IC(z) = β0 + β1·z + β2·z²`, Kalman-updated each day | Medium |
| `rbf_rls` | 7 Gaussian basis functions + Recursive Least Squares with forgetting factor | Medium |
| `binned_kalman` | 15 independent 1-D Kalman filters, one per z-score bin | Low |
| `nadaraya_watson` | Rolling buffer + spatial × temporal kernel regression | Low |

All models share the `ICModel` interface from `models/base.py`:
```python
model.update(z, y)              # y = r / σ (risk-normalised return)
alpha_z, variance = model.predict(z)   # alpha_z = IC(z) × z
```

The final Grinold alpha is `alpha_i = σ_i × alpha_z_i`.

### Adding a new model

1. Create `models/my_model.py` implementing `ICModel`.
2. Register it in `models/__init__.py`.
3. Add its name to `MODELS` in `config.py`.

## Project Structure

```
dynamic_ic/
├── run.py                  # Slurm orchestrator: submit all 3 phases at once
├── compute.py              # Phase 1: cache z-score parquets
├── train.py                # Phase 2: walk-forward → alphas → MVO weights
├── analyze.py              # Phase 3: charts, summary, IC visualisation
├── config.py               # All dates, params, signal names, paths
├── pipeline.py             # DynamicICPipeline: z-scores, walk-forward
├── models/
│   ├── base.py             # ICModel ABC
│   ├── static.py
│   ├── kalman_poly.py
│   ├── rbf_rls.py
│   ├── binned_kalman.py
│   └── nadaraya_watson.py
├── signals/
│   ├── _factor_signal.py   # shared helper: factor returns → asset z
│   ├── _asset_signal.py    # shared helper: asset parquet loaders
│   ├── style_momentum.py
│   ├── industry_momentum.py
│   ├── idiosyncratic_momentum.py
│   ├── style_reversal.py
│   ├── industry_reversal.py
│   ├── idiosyncratic_reversal.py
│   ├── idiosyncratic_volatility.py
│   └── betting_against_beta.py
├── tests/
│   ├── test_models.py      # unit tests for all IC models
│   ├── test_pipeline.py    # unit tests for DynamicICPipeline
│   └── test_config.py      # sanity checks for config values
└── results/
    └── {split}/
        ├── z_scores/
        ├── alphas/{signal}/{model}.parquet
        ├── weights/{signal}/{model}/{gamma}/
        ├── performance_{signal}.png
        ├── ic_function_{signal}.png
        └── backtest_report.txt
```

## Usage

```bash
# Install dependencies (including dev extras for testing)
uv sync --extra dev

# ── Automated (fire-and-forget on the cluster) ──────────────────────────────
uv run python run.py --split full          # submit full pipeline, then walk away
uv run python run.py --split full --dry-run  # print scripts without submitting

# ── Manual (step-by-step, locally or interactively) ────────────────────────
# Phase 1: pre-compute z-scores (run once per split)
uv run python compute.py --split full
uv run python compute.py --split recent --signal style_momentum   # single signal

# Phase 2a: walk-forward only (no MVO)
uv run python train.py --split full --no-backtest

# Phase 2b: walk-forward + submit MVO to Slurm (BacktestRunner, one job per year)
uv run python train.py --split full

# Phase 2c: walk-forward + run MVO inline (no nested Slurm; used internally by run.py)
uv run python train.py --split full --run-mvo

# Single pair
uv run python train.py --split recent --signal style_momentum --model kalman_poly

# Phase 3: generate charts and summary
uv run python analyze.py --split full

# ── Tests ───────────────────────────────────────────────────────────────────
uv run pytest
```

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STATIC_IC` | 0.05 | Fixed IC for the static baseline |
| `GAMMA` | 200 | MVO risk-aversion parameter |
| `LOOKBACK_DAYS` | 120 | Rolling window fed to IC models |
| `MOMENTUM_WINDOW` | 231 | 11-month rolling window for momentum signals |
| `MOMENTUM_SKIP` | 21 | Skip most-recent month (standard momentum convention) |
| `REVERSAL_WINDOW` | 21 | Short-term reversal window |
| `MIN_PRICE` | 5.0 | Minimum stock price filter |
