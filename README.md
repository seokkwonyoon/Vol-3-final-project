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

The pipeline runs in four sequential phases, each depending on the previous:

```
Phase 1: compute.py   z-score each signal cross-sectionally, cache to parquet
Phase 2: train.py     walk-forward IC estimation → alpha parquets (one per pair)
Phase 3: mvo.py       mean-variance optimization → portfolio weights (one per pair)
Phase 4: analyze.py   performance charts, IC function evolution, portfolio overview, summary report
```

**Phase 2 is the core.** For each (signal, model) pair it steps through every
trading day sequentially: update the IC model with yesterday's realised returns,
then predict today's alphas. This is inherently sequential because each model
carries state that depends on all prior days.

**Phase 3 is independent across dates.** Given the alpha parquets from Phase 2,
MVO for each trading day is solved independently — no state, no look-ahead — so
Ray fans it out across 16 CPUs simultaneously.

### Submitting to the cluster

`submit.py` submits all four phases as a Slurm job chain and exits. Each phase
starts only after every task of the previous phase succeeds.

```
Phase 1  8-task array    compute.py   per signal           4 CPU / 32G / 30 min
Phase 2  40-task array   train.py     per signal×model     4 CPU / 32G / 12 hr
Phase 3  40-task array   mvo.py       per signal×model    16 CPU / 64G /  2 hr
Phase 4  single job      analyze.py   for the split        4 CPU / 32G / 15 min
```

```bash
uv run python submit.py           # submit everything
uv run python submit.py --dry-run # print scripts without submitting
```

### Running steps manually

```bash
uv run python compute.py --signal style_momentum
uv run python train.py   --signal style_momentum --model kalman_poly
uv run python mvo.py     --signal style_momentum --model kalman_poly
uv run python analyze.py
```

## Signals

Eight signals, each pre-computed as a cross-sectional z-score across the universe:

| Signal | Source | Description |
|--------|--------|-------------|
| `style_momentum` | Barra style factor returns | 11-to-1 month vol-scaled momentum |
| `industry_momentum` | Barra industry factor returns | Same, industry factors |
| `idiosyncratic_momentum` | `specific_return` | Vol-scaled rolling momentum on idio returns |
| `style_reversal` | Barra style factor returns | 21-day reversal |
| `industry_reversal` | Barra industry factor returns | Same, industry factors |
| `idiosyncratic_reversal` | `specific_return` | 21-day idio reversal |
| `idiosyncratic_volatility` | `specific_risk` | Low-vol anomaly: negative specific risk |
| `betting_against_beta` | `predicted_beta` | BAB: negative predicted beta |

## IC Models

Seven models in `models/`, all sharing the same interface:

```python
model.update(z, y)               # y = r / σ  (risk-normalised return)
alpha_z, variance = model.predict(z)  # alpha_z = IC(z) × z
```

| Model | Description |
|-------|-------------|
| `static` | Fixed IC = 0.05 — the baseline |
| `kalman_poly` | Kalman filter over polynomial coefficients: `IC(z) = β0 + β1·z + β2·z²` |
| `rbf_rls` | 7 Gaussian basis functions fit by recursive least squares with forgetting |
| `binned_kalman` | 15 independent Kalman filters, one per z-score bin |
| `nadaraya_watson` | Rolling kernel regression: spatial × temporal weights over a 120-day buffer |
| `gaussian_process_regression` | Bayesian GPR with RBF + white noise kernel |
| `kernel_ridge_regression` | Kernel ridge regression over a rolling buffer |

### Adding a new signal or model

For a new signal: create `signals/my_signal.py` and add its name to `SIGNALS` in `configs/default.py`.

For a new model:
1. Create `models/my_model.py` implementing `ICModel` from `models/base.py`.
2. Register it in `models/__init__.py`.
3. Add its key to `MODELS` in `configs/default.py`.

## Project Structure

```
dynamic_ic/
├── submit.py               # Slurm orchestrator — submit all 4 phases at once
├── compute.py              # Phase 1: compute and cache z-score parquets
├── train.py                # Phase 2: walk-forward IC estimation → alpha parquets
├── mvo.py                  # Phase 3: mean-variance optimization → weight parquets
├── analyze.py              # Phase 4: charts, portfolio overview, IC visualisation, summary table
├── pipeline.py             # DynamicICPipeline: z-scores + walk-forward logic
├── configs/                # All parameters, splits, signal names, paths
│   └── default.py
├── models/
│   ├── base.py             # ICModel ABC
│   ├── static.py
│   ├── kalman_poly.py
│   ├── rbf_rls.py
│   ├── binned_kalman.py
│   ├── nadaraya_watson.py
│   ├── gaussian_process_regression.py
│   └── kernel_ridge_regression.py
├── signals/
│   ├── _factor_signal.py   # shared helper: factor returns → asset z-score
│   ├── _asset_signal.py    # shared helper: asset parquet loaders
│   └── {signal_name}.py    # one file per signal
└── results/
    └── {split}/
        ├── z_scores/{signal}.parquet
        ├── alphas/{signal}/{model}.parquet
        ├── weights/{signal}/{model}/{gamma}/{year}.parquet
        ├── performance_{signal}.png
        ├── ic_function_{signal}.png
        ├── portfolio_overview.png
        └── backtest_report.txt
```

## Configuration

All tuneable parameters live in `configs/default.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPLIT` | `"recent"` | Active date split: `"full"`, `"recent"`, or `"test"` |
| `SELECTED_SIGNALS` | `None` | Signals to run (None = all signals) |
| `STATIC_IC` | 0.05 | Fixed IC for the static baseline |
| `GAMMA` | 200 | MVO risk-aversion parameter |
| `LOOKBACK_DAYS` | 120 | Rolling window fed to IC models (trading days) |
| `MOMENTUM_WINDOW` | 231 | 11-month lookback for momentum signals |
| `MOMENTUM_SKIP` | 21 | Skip most-recent month (standard convention) |
| `REVERSAL_WINDOW` | 21 | Short-term reversal window |
| `MIN_PRICE` | 5.0 | Minimum stock price filter |

## Tests

```bash
uv sync --extra dev
uv run pytest
```
