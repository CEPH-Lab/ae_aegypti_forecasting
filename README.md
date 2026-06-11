# Mosquito Abundance Forecasting — *Aedes aegypti*

Probabilistic short-term forecasting of *Aedes aegypti* mosquito abundance (mosquitoes per trap night) across four US locations: **Miami**, **Key West**, **Los Angeles**, and **Maricopa**. Models produce weekly quantile forecasts at 1–4 week horizons, evaluated with WIS, sMAPE, and interval coverage.

---

## Table of Contents

1. [Project structure](#project-structure)
2. [Data](#data)
3. [Configuration files](#configuration-files)
4. [Running the pipeline](#running-the-pipeline)
   - [Step 1 — Naive model](#step-1--naive-model)
   - [Step 2 — Format naive output](#step-2--format-naive-output)
   - [Step 3 — STL models](#step-3--seasonal-models)
   - [Step 4 — SARIMAX model](#step-4--sarimax-model)
   - [Step 5 — UCM model](#step-5--ucm-model)
   - [Step 6 — Ensemble](#step-6--ensemble)
   - [Step 7 — Score tabulation](#step-7--score-tabulation)
   - [Step 8 — Visualization](#step-8--visualization)
5. [Output files at a glance](#output-files-at-a-glance)
6. [Scoring metrics](#scoring-metrics)
7. [Dependencies](#dependencies)

---

## Project structure

```
mosq_aegypti_forecasting/
│
├── configs/                          # Per-location parameter files (*.in)
│   ├── keywest_aegypti.in
│   ├── la_aegypti.in
│   ├── maricopa_aegypti.in
│   └── miami_aegypti.in
│
├── data/
│   ├── raw/                          # Original input data (never modified)
│   │   ├── KeyWest/
│   │   ├── LA/
│   │   ├── Maricopa/
│   │   └── Miami/
│   │       ├── *.Aegypti.csv         # Mosquito counts (date, mosq_per_trap_night)
│   │       └── weekly_weather_*.csv  # Temperature & precipitation
│   ├── interim/                      # Naive model raw output (before scoring)
│   ├── processed/                    # Scored forecast CSVs for every model
│   └── scores/                       # Final aggregated score comparison tables
│
├── scripts/
│   ├── models/
│   │   ├── baseline_sweep_dates.py   # Naive baseline model
│   │   ├── seasonal_model.py         # STL models
│   │   ├── sarimax_model.py          # SARIMAX model
│   │   ├── ucm_model.py              # Unobserved Components Model
│   │   └── ensemble_model.py         # Ensemble aggregation (mean quantiles)
│   ├── processing/
│   │   └── format_naive_output.py    # Converts naive interim output to standard schema
│   ├── score_comparison/
│   │   └── score_tabulation.py       # Builds per-location score comparison tables
│   └── visualization/
│       ├── master_visualization.py   # Diagnostic plots, WIS heatmaps, summary figures
│       ├── forecast_timeseries_viz.py # Time series plots with prediction intervals
│       └── flusight_forecast_viz.py  # FluSight-style forecast visualizations
│
├── rtrend_tools/                     # Core forecasting library (trajectory synthesis, scoring)
├── toolbox/                          # Utilities (config parsing, file I/O)
├── utils/
│   └── scoring_logic.py              # WIS, sMAPE, and coverage calculations
├── visualization/
│   └── plots_master/                 # All generated figures (PDF + PNG)
└── setup.py
```

---

## Data

Raw data lives in `data/raw/<Location>/`. **Do not edit files in `data/raw/`.**

| File pattern | Contents |
|---|---|
| `<Location>.Aegypti.csv` | Weekly mosquito counts per trap — columns: `date`, `mosq_per_trap_night` |
| `weekly_weather_<location>.csv` | Weekly temperature and precipitation (used as exogenous variables by SARIMAX and UCM) |

Data spans from:
KeyWest: 2010 - 2020;
LA: 2018 - 2021;
Maricopa: 2014 - 2021;
Miami: 2017 - 2024;
Time series are pre-processed inside each model script (linear interpolation, frequency alignment to ISO weeks).

---

## Configuration files

Each location has a `.in` file under `configs/`. These are key-value parameter files used by the naive model. Key parameters:

| Parameter | Meaning |
|---|---|
| `truth_data_fname` | Path to the raw mosquito CSV |
| `use_dates_first` / `use_dates_last` | Date range over which to sweep forecast origins |
| `nweeks_fore` | Forecast horizon in weeks (4 since we want from 1 to 4 weeks ahead forecasts) |
| `preproc_roi_len` | Preprocessing window length in weeks (default 53) |
| `fit_normal` | `True` -> fit week-to-week differences to a normal distribution; `False` -> use empirical values |
| `apply_scores` | Compute WIS/sMAPE/coverage at runtime |
| `export` | Write output CSV files |
| `rn_num_samples` | Number of Monte Carlo trajectories (default 1000) |
| `synth_seed` | Random seed for reproducibility |

---

## Running the pipeline

Run all scripts from the **repository root** so that relative paths resolve correctly. The recommended order is:

```
Step 1 (Naive) -> Step 2 (Format naive) -> Step 3 (Seasonal) -> Step 4 (SARIMAX) -> Step 5 (UCM) -> Step 6 (Ensemble) -> Step 7 (Scores) -> Step 8 (Visualization)
```

Steps 2–5 are independent and can be run in any order or in parallel. Steps 6–8 each depend on the steps before them.

---

### Step 1 — Naive model

**Script:** `scripts/models/baseline_sweep_dates.py`

The naive baseline sweeps over all forecast origin dates and generates a Monte Carlo trajectories. It must be run once per location.

```bash
python scripts/models/baseline_sweep_dates.py configs/miami_aegypti.in    data/interim 
python scripts/models/baseline_sweep_dates.py configs/keywest_aegypti.in  data/interim 
python scripts/models/baseline_sweep_dates.py configs/la_aegypti.in       data/interim 
python scripts/models/baseline_sweep_dates.py configs/maricopa_aegypti.in data/interim 
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--nweeks-fore N` | value from config | Forecast horizon in weeks |
| `--ncpus N` | 1 | Parallel workers for the date sweep |
| `--export True/False` | value from config | Whether to write output files |

**Output location:** `data/interim/`

| File | Contents |
|---|---|
| `<location>_aegypti_forecast.csv` | Raw quantile columns (`q0.025` … `q0.975`) plus metadata (`day_pres`, `weeks_ahead`, `truth_vals`). Not yet scored — converted in Step 2. |

---

### Step 2 — Format naive output

**Script:** `scripts/processing/format_naive_output.py`

The naive model (Step 1) writes a different raw format to `data/interim/`. This script converts those files to the standard scored schema shared by all other models, merges in observed truth values from `data/raw/`, clips forecasts to ≥ 0, and computes WIS/sMAPE/coverage.

```bash
python scripts/processing/format_naive_output.py
```

**Input:** `data/interim/<location>_aegypti_forecast.csv`  
**Output:** `data/processed/naive_<location>_aegypti_forecast_scored.csv`

---

### Step 3 — STL models

**Script:** `scripts/models/seasonal_model.py`

Applies additive seasonal decomposition (52-week period) to separate trend, seasonality, and residuals. For each forecast origin, a rolling window average sets the current level, the historical seasonal component for the target week provides a deterministic shift, and noise is sampled based on whether the forecast week falls in a high- or low-activity season. Four model variants (`SEASONAL_n1` through `n4`) are produced by varying the rolling window width.

The script loops over all four locations internally — no arguments needed:

```bash
python scripts/models/seasonal_model.py
```

**Output location:** `data/processed/`

| File | Contents |
|---|---|
| `seasonal_n1_<location>_aegypti_forecast_scored.csv` | Variant with narrowest rolling window |
| `seasonal_n2_<location>_aegypti_forecast_scored.csv` | |
| `seasonal_n3_<location>_aegypti_forecast_scored.csv` | |
| `seasonal_n4_<location>_aegypti_forecast_scored.csv` | Variant with widest rolling window |

Each file is already scored. Columns include:

```
model | location | species | training_date | forecast_date | forecast_horizon
forecasted_abund | observed_abund | lower.80 | upper.80 | lower.95 | upper.95
q0.025 … q0.975  (23 quantile levels)
sMAPE | WIS | cov_50 | cov_80 | cov_90 | cov_95
current_level_used | seasonality_t | regime | sigma_used
```

---

### Step 4 — SARIMAX model

**Script:** `scripts/models/sarimax_model.py`

Fits a SARIMAX(1,1,2)(1,0,0,52) model with exogenous weather variables (temperature and precipitation). For future forecast dates where actual weather is unavailable, 5-year rolling weather averages are substituted. The script loops over all locations automatically.

```bash
python scripts/models/sarimax_model.py
```

**Output location:** `data/processed/`

| File | Contents |
|---|---|
| `arima_<location>_aegypti_forecast_scored.csv` | SARIMAX quantile forecasts with scores |

At least 20 training observations are required before the first forecast is generated. Dates where the optimizer fails to converge are skipped.

---

### Step 5 — UCM model

**Script:** `scripts/models/ucm_model.py`

Fits a state-space Unobserved Components Model with a local level (adaptive trend), a 52-week seasonal component, and exogenous weather variables. The local level adapts to structural shifts in abundance over time. The script loops over all locations automatically.

```bash
python scripts/models/ucm_model.py
```

**Output location:** `data/processed/`

| File | Contents |
|---|---|
| `ucm_<location>_aegypti_forecast_scored.csv` | UCM quantile forecasts with scores |

---


### Step 6 — Ensemble

**Script:** `scripts/models/ensemble_model.py`

Loads all scored model CSVs from `data/processed/`, groups forecasts by `(location, species, training_date, forecast_date, forecast_horizon)`, and computes the *mean across all quantile columns* for each group. The Naive model is excluded from the ensemble average. Scores are then recomputed on the ensemble quantiles.

```bash
python scripts/models/ensemble_model.py
```

**Output:** `data/processed/all_models_master.csv`

This is the single master file used by all the remaining scripts. It contains one row per `(model, location, training_date, forecast_date, horizon)` combination, covering all models: `NAIVE`, `SEASONAL_n1`–`n4`, `ARIMA`, `UCM`, and `ENSEMBLE`.

---

### Step 7 — Score tabulation

**Script:** `scripts/score_comparison/score_tabulation.py`

Reads `all_models_master.csv` and produces a summary comparison table for each location. Each table shows mean sMAPE, WIS, and 95% coverage broken out by model and forecast horizon (H1–H4).

```bash
python scripts/score_comparison/score_tabulation.py
```

**Output location:** `data/scores/`

| File | Contents |
|---|---|
| `final_<location>_aegypti_comparison.csv` | Models as rows, `(metric × horizon)` pairs as columns |

Table layout example:

```
             sMAPE_H1  sMAPE_H2  sMAPE_H3  sMAPE_H4  WIS_H1  ...  Coverage95%_H4
NAIVE
SEASONAL_n1
SEASONAL_n2
SEASONAL_n3
SEASONAL_n4
ARIMA
UCM
ENSEMBLE
```

---

### Step 8 — Visualization

All visualization scripts read from `data/processed/all_models_master.csv`. Figures are saved to `visualization/plots_master/` as PDF or PNG depending on the code.

#### Master diagnostic plots

```bash
python scripts/visualization/master_visualization.py
```

Generates:
- Prediction-vs-observed scatter plots and sMAPE boxplots by model
- Calibration curves (empirical coverage vs. nominal level)
- WIS relative to naive baseline by ISO week
- Cross-location WIS heatmaps per forecast horizon

#### Forecast time series

```bash
python scripts/visualization/forecast_timeseries_viz.py
```

Produces a 2×2 panel grid (one panel per location) overlaying the observed time series with model forecasts and shaded prediction intervals. Files are named:

```
timeseries_<model>_<species>_h<horizon>_<interval>pct.pdf
timeseries_<model>_<species>_h<horizon>_<interval>pct.png
```

#### FluSight-style forecast plots

```bash
python scripts/visualization/flusight_forecast_viz.py
```

---

## Output files at a glance

| Path | Created by | Contents |
|---|---|---|
| `data/interim/<loc>_aegypti_forecast.csv` | Step 1 | Naive raw quantile forecasts, unscored |
| `data/processed/naive_<loc>_aegypti_forecast_scored.csv` | Step 5 | Naive forecasts, scored |
| `data/processed/seasonal_n{1-4}_<loc>_aegypti_forecast_scored.csv` | Step 2 | Seasonal model forecasts, scored |
| `data/processed/arima_<loc>_aegypti_forecast_scored.csv` | Step 3 | SARIMAX forecasts, scored |
| `data/processed/ucm_<loc>_aegypti_forecast_scored.csv` | Step 4 | UCM forecasts, scored |
| `data/processed/all_models_master.csv` | Step 6 | All models + ensemble in one file |
| `data/scores/final_<loc>_aegypti_comparison.csv` | Step 7 | Per-location score comparison table |
| `visualization/plots_master/` | Step 8 | All generated figures |

---

## Scoring metrics

Scoring logic lives in `utils/scoring_logic.py`. Three metrics are computed for every forecast row.

**sMAPE** (Symmetric Mean Absolute Percentage Error)

**WIS** (Weighted Interval Score): Evaluates the full predictive distribution across 23 quantile levels (0.025, 0.05, 0.075, …, 0.50, …, 0.975). Combines a sharpness penalty (interval width) with a calibration penalty (distance outside the interval when the observation is missed).

**Coverage**: Binary indicators for whether the observed value falls within the 50%, 80%, 90%, 95%, etc. prediction intervals, stored as `cov_50`, `cov_80`, `cov_90`, `cov_95`...

---

## Dependencies

**Standard libraries:**

```
numpy
pandas
scipy
statsmodels       # SARIMAX, UCM
matplotlib
seaborn
tqdm
```

Install with:

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn tqdm
```

**Internal packages** (in this repository — install once in development mode from the repo root):

```bash
pip install -e .
```

This registers `rtrend_tools` and `toolbox` so all scripts can import them regardless of working directory.
