import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.scoring_logic import CDC_QUANTILES_SEQ, calculate_row_metrics

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok = True)

LOCATIONS = ["Keywest", "LA", "Maricopa", "Miami"]
SPECIES = ["Aegypti"]

SEASONAL_PERIOD = 52

def qcol(q: float) -> str:
    return f"q{q:.3f}"

def extract_summary_from_quantiles(qvals: dict):
    pm   = qvals.get("q0.500", np.nan)
    lo80 = qvals.get("q0.100", np.nan)
    up80 = qvals.get("q0.900", np.nan)
    lo95 = qvals.get("q0.025", np.nan)
    up95 = qvals.get("q0.975", np.nan)
    loiqr = qvals.get("q0.250", np.nan)
    upiqr = qvals.get("q0.750", np.nan)
    return pm, lo80, up80, lo95, up95, loiqr, upiqr

# ===========================================
# MODEL: Seasonal Naive with Trajectories
# ============================================

def forecast_fixed_snaive(train_y_fixed, initial_train_end, future_idx, num_sims=10000, quantiles=CDC_QUANTILES_SEQ):
    """
    Fixed-Origin Seasonal Naive Forecast (Calendar Week Aligned).
    Only uses the first 2 years of data. Matches the target forecast date's 
    ISO calendar week to the corresponding calendar week in Year 2.
    """
    # 1. Calculate FIXED standard deviation of seasonal residuals using ONLY Year 1 and Year 2
    residuals = train_y_fixed - train_y_fixed.shift(52)
    sigma = residuals.std(ddof=1)
    
    # Fallback if data has zero variance
    if pd.isna(sigma) or sigma == 0:
        sigma = 0.01

    means = []
    q_mat = []
    
    # 2. Iterate through the actual future forecast dates
    for step_idx, forecast_date in enumerate(future_idx):
        
        # --- NEW CALENDAR ALIGNMENT LOGIC ---
        # Get the actual calendar week number (1 through 52, sometimes 53)
        target_week = forecast_date.isocalendar().week
        
        # If it happens to be a Leap Week 53, just use Week 52's data
        # since Year 2 might not have had a Week 53
        if target_week == 53:
            target_week = 52
            
        # Filter the frozen training data for that exact week of the year
        # We only look at Year 2 (the final 52 weeks of the frozen data)
        year_2_data = train_y_fixed.iloc[-52:]
        matching_rows = year_2_data[year_2_data.index.isocalendar().week == target_week]
        
        # Pull the point forecast
        # (Using a fallback to modulo math just in case the calendar lookup fails due to missing dates)
        if not matching_rows.empty:
            point_forecast = matching_rows.iloc[-1]
        else:
            weeks_past_cutoff = (forecast_date - initial_train_end).days // 7
            idx_in_last_year = (weeks_past_cutoff - 1) % 52
            point_forecast = year_2_data.iloc[idx_in_last_year]
        # ------------------------------------

        # 3. Simulate Trajectories using the FIXED sigma
        current_sigma = sigma * np.sqrt(step_idx + 1)
        trajectories = np.random.normal(loc=point_forecast, scale=current_sigma, size=num_sims)
        
        # Floor at 0.0
        point_forecast = max(0.0, point_forecast)
        trajectories = np.maximum(0.0, trajectories)
        
        # 4. Extract quantiles
        step_quantiles = np.quantile(trajectories, quantiles)
        
        means.append(point_forecast)
        q_mat.append(step_quantiles)
        
    means = np.array(means)
    q_mat = np.array(q_mat).T  # Transpose to shape: (len(quantiles), steps)
    
    qdict = {qcol(float(q)): q_mat[i, :] for i, q in enumerate(quantiles)}
    
    return means, qdict, sigma

# ============================================
# MAIN
# ============================================
print(f"\n=== Running Seasonal Naive (s={SEASONAL_PERIOD}) ===")
for location in LOCATIONS:
    for species in SPECIES:
        print(f"\n→ Processing {location} - {species}")

        abund_fp   = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")
        
        if not (os.path.exists(abund_fp)):
            print(f" Missing files for {location}-{species}, skipping.")
            continue

        # Load and align data (Keeping exact identical data prep for 1:1 comparison)
        abund_df = pd.read_csv(abund_fp)

        abund_df["date"] = pd.to_datetime(abund_df["date"])

        # Linear interpolation to fill missing data
        abund_df = abund_df.set_index("date")
        abund_df.index = abund_df.index.to_period("W").start_time

        full_index = pd.date_range(start=abund_df.index.min(), end=abund_df.index.max(), freq="W-MON")
        full_df = abund_df.reindex(full_index)
        full_df['mosq_per_trap_night'] = full_df['mosq_per_trap_night'].interpolate(method='linear')

        initial_train_end = full_df.index.min() + pd.DateOffset(years=2)
        start_date = initial_train_end
        end_date = full_df.index.max() - pd.Timedelta(weeks=4)

        if start_date > end_date:
            continue

        date_range = pd.date_range(start=start_date, end=end_date, freq="W-MON")

        # ---------------------------
        # Forecasting loop
        # ---------------------------
        train_y_fixed = full_df.loc[:initial_train_end, "mosq_per_trap_night"]

        all_forecasts = []
        for train_end in tqdm(date_range, desc=f"{location}-{species}-SNaive"):
            train_slice = slice(None, train_end)
            # Seasonal naive needs at least 52 weeks of history to predict the next week
            if len(train_y_fixed) < 52:
                continue

            horizons = [1, 2, 3, 4]
            future_idx = [train_end + pd.Timedelta(weeks=h) for h in horizons]

            if any(d not in full_df.index for d in future_idx):
                continue

            # Running the model
            try:
                pred_mean, qdict, sigma_val = forecast_fixed_snaive(
                    train_y_fixed = train_y_fixed, 
                    initial_train_end = initial_train_end, 
                    future_idx = future_idx,
                    num_sims = 10000, 
                    quantiles = CDC_QUANTILES_SEQ
                )
            except Exception as e:
                # Should not fail, but handled just in case
                continue

            # Output structure (matching SARIMAX perfectly)
            for step_idx, (h, forecast_date) in enumerate(zip(horizons, future_idx)):
                qvals_h = {k: float(v[step_idx]) for k, v in qdict.items()}
                pm, lo80, up80, lo95, up95, loiqr, upiqr = extract_summary_from_quantiles(qvals_h)

                all_forecasts.append({
                    "model": "Seasonal_Naive",
                    "location": location,
                    "species": species,
                    "training_date": train_end.date(),
                    "forecast_date": forecast_date.date(),
                    "forecast_horizon": h,
                    "forecasted_abund": float(pred_mean[step_idx]),
                    "observed_abund": float(full_df.loc[forecast_date, "mosq_per_trap_night"]),
                    "lower.80": float(lo80), "upper.80": float(up80),
                    "lower.95": float(lo95), "upper.95": float(up95),
                    "lower.iqr": float(loiqr), "upper.iqr": float(upiqr),
                    **qvals_h,
                    "current_level_used": np.nan, 
                    "seasonality_t": np.nan,
                    "seasonality_t_plus_i": np.nan,
                    "regime": "SNAIVE",
                    "sigma_used": float(sigma_val)
                })

        # Scoring and saving
        forecast_df = pd.DataFrame(all_forecasts)
        print(f"\n Total forecasts for {location}-{species} : {len(forecast_df)}")

        if not forecast_df.empty:
            print(" Calculating WIS and Coverage metrics...")
            scored_df = calculate_row_metrics(forecast_df)
            
            # Save file
            out_fp = os.path.join(PROCESSED_DATA_DIR, f"snaive_{location.lower()}_{species.lower()}_forecast_scored.csv")
            scored_df.to_csv(out_fp, index=False)
            print(f" Saved scored file: {out_fp}")










# def forecast_snaive_trajectories(train_y, steps=4, seasonal_period=52, num_sims=10000, quantiles=CDC_QUANTILES_SEQ):
#     """
#     Seasonal Naive Forecast: y(T+h) = y(T+h-s)
#     Generates quantiles by simulating normal distributions around the point forecast
#     using the historical standard deviation of the seasonal residuals.
#     """
#     # 1. Calculate standard deviation of historical seasonal residuals (y_t - y_{t-52})
#     residuals = train_y - train_y.shift(seasonal_period)
#     sigma = residuals.std(ddof=1)
    
#     # Fallback if data has zero variance
#     if pd.isna(sigma) or sigma == 0:
#         sigma = 0.01

#     means = []
#     q_mat = []
    
#     for h in range(1, steps + 1):
#         # 2. Point forecast: Look exactly 52 weeks back from the forecast target.
#         # h=1 looks back 52 weeks (idx -52), h=2 looks back 51 weeks (idx -51), etc.
#         lookback_idx = -seasonal_period + h - 1
#         point_forecast = train_y.iloc[lookback_idx]
        
#         # 3. Simulate Trajectories
#         # For a random walk with a seasonal step, the variance at horizon h (where h <= 52) is exactly sigma^2
#         trajectories = np.random.normal(loc=point_forecast, scale=sigma, size=num_sims)
        
#         # Floor at 0.0 (Mimicking the linear SARIMAX code behavior before log transformation)
#         point_forecast = max(0.0, point_forecast)
#         trajectories = np.maximum(0.0, trajectories)
        
#         # 4. Extract quantiles
#         step_quantiles = np.quantile(trajectories, quantiles)
        
#         means.append(point_forecast)
#         q_mat.append(step_quantiles)
        
#     means = np.array(means)
#     q_mat = np.array(q_mat).T  # Transpose to shape: (len(quantiles), steps)
    
#     qdict = {qcol(float(q)): q_mat[i, :] for i, q in enumerate(quantiles)}
    
#     return means, qdict, sigma