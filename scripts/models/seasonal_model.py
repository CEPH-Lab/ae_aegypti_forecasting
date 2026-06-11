import sys
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================
# PATH SETUP
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Import the scoring logic
from utils.scoring_logic import CDC_QUANTILES_SEQ, calculate_row_metrics

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ============================================
# CONFIGURATION
# ============================================
np.random.seed(100)

LOCATIONS = ["Keywest", "LA", "Maricopa", "Miami"]
SPECIES = ["Aegypti"]

# --------------------------------------------
# USER PARAMETERS
# --------------------------------------------
MIN_YEARS_TRAIN = 2          # Start forecasting after 2 years of history
FORECAST_HORIZONS = [1, 2, 3, 4]
ROLLING_N = 3                # Average of past 'n' weeks
SEASONAL_PERIOD = 52         # Weekly data
NUM_SIMULATIONS = 1000       # Number of Monte Carlo trajectories

def qcol(q: float) -> str:
    return f"q{q:.3f}"

# ============================================
# FORECASTING FUNCTION
# ============================================
def forecast_at_origin(df, origin_date, location, species):
    
    # 1. Training window: data strictly up to origin_date
    training_df = df.loc[:origin_date].copy()

    # Safety check: Need at least 2 full cycles (104 weeks) + a buffer
    if len(training_df) < (2 * SEASONAL_PERIOD):
        return []

    # 2. Running decomposition
    try:
        decomp = seasonal_decompose(
            training_df['mosq_per_trap_night'].dropna(),
            model='additive',
            period=SEASONAL_PERIOD,
            two_sided=False, 
            extrapolate_trend='freq'
        )
    except ValueError:
        return []

    # 3. Looking up seasonality
    training_df['seasonal_comp'] = decomp.seasonal
    training_df['week_of_year'] = training_df.index.isocalendar().week.astype(int)
    
    seasonal_map = training_df.groupby('week_of_year')['seasonal_comp'].mean().to_dict()

    # 3.1 Building noise distribution
    residuals = decomp.resid
    seasonals = decomp.seasonal
    noise_df = pd.DataFrame({'resid': residuals, 'season': seasonals}).dropna()
    
    high_resid = noise_df.loc[noise_df['season'] >= 0, 'resid']
    low_resid = noise_df.loc[noise_df['season'] < 0, 'resid']
    
    global_sigma = noise_df['resid'].std(ddof=1) if len(noise_df) > 1 else 0
    sigma_high = high_resid.std(ddof=1) if len(high_resid) > 10 else global_sigma
    sigma_low = low_resid.std(ddof=1) if len(low_resid) > 10 else global_sigma

    # 4. Calculating latest start point
    last_n_values = training_df['mosq_per_trap_night'].dropna().iloc[-ROLLING_N:]
    if len(last_n_values) < ROLLING_N:
        return [] 

    current_level_t = last_n_values.mean()

    # Current seasonality
    week_t = int(origin_date.isocalendar().week)
    if week_t not in seasonal_map:
        week_t = 52 if week_t >= 53 else 1 
    seasonality_t = seasonal_map[week_t]

    rows = []

    # 5. Creating linked trajectories
    def get_season(wk: int) -> float:
        if wk in seasonal_map:
            return float(seasonal_map[wk])
        return float(seasonal_map.get(1, 0.0))

    future_dates = [origin_date + pd.Timedelta(weeks=h) for h in FORECAST_HORIZONS]
    S0 = get_season(int(origin_date.isocalendar().week))
    S_future = [get_season(int(d.isocalendar().week)) for d in future_dates]

    S_all = [S0] + S_future
    delta_S = [S_all[i] - S_all[i-1] for i in range(1, len(S_all))] 

    sigma_star = sigma_high if S_future[0] >= 0 else sigma_low
    sigmas = [sigma_star] * len(FORECAST_HORIZONS)

    regime_star = "High_activity" if S_future[0] >= 0 else "Low_activity"
    regimes = [regime_star] * len(FORECAST_HORIZONS)    

    trajectories = np.zeros((NUM_SIMULATIONS, len(FORECAST_HORIZONS)), dtype=float)
    y_prev = np.full(NUM_SIMULATIONS, current_level_t, dtype=float)

    for step_idx in range(len(FORECAST_HORIZONS)):
        eps = np.random.normal(loc=0.0, scale=sigmas[step_idx], size=NUM_SIMULATIONS)
        y_new = y_prev + delta_S[step_idx] + eps
        y_new = np.maximum(0.0, y_new) 
        trajectories[:, step_idx] = y_new
        y_prev = y_new 

    # Summarize per horizon
    for step_idx, h in enumerate(FORECAST_HORIZONS):
        forecast_date = future_dates[step_idx]
        sims_h = trajectories[:, step_idx]

        qvals = np.quantile(sims_h, CDC_QUANTILES_SEQ)
        q_out = {qcol(q): float(v) for q, v in zip(CDC_QUANTILES_SEQ, qvals)}

        forecast_med = q_out[qcol(0.5)]

        observed = np.nan
        if forecast_date in df.index:
            observed = df.loc[forecast_date, "mosq_per_trap_night"]

        # Output structure
        rows.append({
            "model": f"SEASONAL_n{ROLLING_N}",
            "location": location,
            "species": species,
            "training_date": origin_date.date(),
            "forecast_date": forecast_date.date(),
            "forecast_horizon": h,
            "forecasted_abund": forecast_med,
            "observed_abund": observed,
            "lower.80": q_out[qcol(0.10)],
            "upper.80": q_out[qcol(0.90)],
            "lower.95": q_out[qcol(0.025)],
            "upper.95": q_out[qcol(0.975)],
            "lower.iqr": q_out[qcol(0.25)],
            "upper.iqr": q_out[qcol(0.75)],
            **q_out,
            "current_level_used": current_level_t,
            "seasonality_t": S0,
            "seasonality_t_plus_i": S_future[step_idx],
            "regime": regimes[step_idx],
            "sigma_used": sigmas[step_idx],
        })

    return rows

# ============================================
# MAIN LOOP
# ============================================
for location in LOCATIONS:
    for species in SPECIES:
        print(f"\n-> Dynamic Rolling Forecast (n={ROLLING_N}) for {location} - {species}")

        # Reads directly from the raw data
        in_fp = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")

        if not os.path.exists(in_fp):
            print(f"  Missing file {in_fp}, skipping.")
            continue

        df = pd.read_csv(in_fp)
        if not {"date", "mosq_per_trap_night"}.issubset(df.columns):
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        
        # Linear interpolation to fill missing data
        df = df.asfreq("W-MON") 
        df['mosq_per_trap_night'] = df['mosq_per_trap_night'].interpolate(method='linear')

        if df['mosq_per_trap_night'].dropna().empty:
            continue

        first_date = df.index.min()
        last_date = df.index.max()
        
        initial_train_end = first_date + pd.DateOffset(years=MIN_YEARS_TRAIN)
        last_possible_origin = last_date - pd.Timedelta(weeks=max(FORECAST_HORIZONS))

        if initial_train_end > last_possible_origin:
            print("  Not enough data history.")
            continue

        origin_dates = pd.date_range(start=initial_train_end, end=last_possible_origin, freq="W-MON")
        all_forecast_rows = []
        
        for i, origin_date in enumerate(origin_dates):
            rows = forecast_at_origin(df, origin_date, location, species)
            all_forecast_rows.extend(rows)

        print(f"  Done. Generated {len(all_forecast_rows)} forecasts.")

        if all_forecast_rows:
            forecast_df = pd.DataFrame(all_forecast_rows)
            
            # Scoring the rows using the function
            print("  Calculating WIS and Coverage metrics...")
            scored_df = calculate_row_metrics(forecast_df)
            
            out_fp = os.path.join(
                PROCESSED_DATA_DIR,
                f"seasonal_n{ROLLING_N}_{location.lower()}_{species.lower()}_forecast_scored.csv"
            )
            scored_df.to_csv(out_fp, index=False)
            print(f"  Saved scored file: {out_fp}")