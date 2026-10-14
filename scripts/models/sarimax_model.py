import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from scipy.stats import norm

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
# CONFIG
# ============================================
LOCATIONS = ["Keywest", "LA", "Maricopa", "Miami"]
SPECIES = ["Aegypti"]

ARIMA_ORDER = (1, 1, 2)
SEASONAL_ORDER = (1, 0, 0, 52)

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
# MODEL
# ============================================
def forecast_arima(train_y, train_exog, future_exog, order, seasonal_order, steps=4, quantiles=CDC_QUANTILES_SEQ):
    model = sm.tsa.SARIMAX(train_y, exog=train_exog, order=order, seasonal_order=seasonal_order)
    result = model.fit(maxiter=50, disp=False)
    fc = result.get_forecast(steps=steps, exog=future_exog)
    mean = fc.predicted_mean.to_numpy()
    se   = fc.se_mean.to_numpy()
    
    z = norm.ppf(np.asarray(quantiles))
    q_mat = mean.reshape(1, -1) + z.reshape(-1, 1) * se.reshape(1, -1)
    
    mean = np.maximum(0.0, mean)
    q_mat = np.maximum(0.0, q_mat)
    
    qdict = {qcol(float(q)): q_mat[i, :] for i, q in enumerate(quantiles)}
    pred_mean = pd.Series(mean, index=fc.predicted_mean.index)
    
    # Extract variance for the master schema
    sigma_used = result.params.get('sigma2', np.nan)
    
    return pred_mean, qdict, sigma_used

# ============================================
# MAIN
# ============================================
print(f"\n=== Running ARIMA {ARIMA_ORDER} | {SEASONAL_ORDER} ===")
for location in LOCATIONS:
    for species in SPECIES:
        print(f"\n→ Processing {location} - {species}")

        abund_fp   = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")
        weather_fp = os.path.join(RAW_DATA_DIR, location, f"weekly_weather_{location}.csv")
        
        if not (os.path.exists(abund_fp) and os.path.exists(weather_fp)):
            print(f" Missing files for {location}-{species}, skipping.")
            continue

        # Load and align data
        abund_df = pd.read_csv(abund_fp)
        weather_df = pd.read_csv(weather_fp)

        abund_df["date"] = pd.to_datetime(abund_df["date"])
        weather_df["custom_week_start"] = pd.to_datetime(weather_df["custom_week_start"])

        # Rolling 5-Year weather
        weather_df = weather_df.sort_values("custom_week_start").reset_index(drop=True)
        weather_df['week'] = weather_df['custom_week_start'].dt.isocalendar().week
        weather_df = weather_df.sort_values(['week', 'custom_week_start'])
        
        weather_df['avg_temp_5yr'] = weather_df.groupby('week')['weekly_avg_temperature'].transform(
            lambda x: x.rolling(window=5, min_periods=1, closed='left').mean()
        )
        weather_df['sum_precip_5yr'] = weather_df.groupby('week')['sum_precipitation'].transform(
            lambda x: x.rolling(window=5, min_periods=1, closed='left').mean()
        )
        
        weather_df = weather_df.sort_values('custom_week_start').reset_index(drop=True)
        weather_df['avg_temp_5yr'] = weather_df['avg_temp_5yr'].fillna(weather_df['weekly_avg_temperature'])
        weather_df['sum_precip_5yr'] = weather_df['sum_precip_5yr'].fillna(weather_df['sum_precipitation'])

        latest_climo = weather_df.groupby('week').last()[['avg_temp_5yr', 'sum_precip_5yr']]

        # Linear interpolation to fill missing data
        abund_df = abund_df.set_index("date")
        abund_df.index = abund_df.index.to_period("W-SUN").end_time.normalize()

        full_index = pd.date_range(start=abund_df.index.min(), end=abund_df.index.max(), freq="W-SUN")
        full_df = abund_df.reindex(full_index)
        full_df['mosq_per_trap_night'] = full_df['mosq_per_trap_night'].interpolate(method='linear')

        # Merge weather with truth data
        weather_actual = weather_df[["custom_week_start", "weekly_avg_temperature", "sum_precipitation", "avg_temp_5yr", "sum_precip_5yr"]].rename(
            columns={"custom_week_start": "date", "weekly_avg_temperature": "actual_temp", "sum_precipitation": "actual_precip"}
        ).set_index("date")
        weather_actual.index = weather_actual.index.to_period("W-SUN").end_time.normalize()

        full_df = full_df.merge(weather_actual, left_index=True, right_index=True, how="left")

        weeks_idx = full_df.index.isocalendar().week
        full_df["avg_temp_5yr"] = full_df["avg_temp_5yr"].fillna(weeks_idx.map(latest_climo["avg_temp_5yr"]))
        full_df["sum_precip_5yr"] = full_df["sum_precip_5yr"].fillna(weeks_idx.map(latest_climo["sum_precip_5yr"]))

        full_df["actual_temp"] = full_df["actual_temp"].fillna(full_df["avg_temp_5yr"])
        full_df["actual_precip"] = full_df["actual_precip"].fillna(full_df["sum_precip_5yr"])

        initial_train_end = full_df.index.min() + pd.DateOffset(years=2)
        start_date = initial_train_end
        end_date = full_df.index.max() - pd.Timedelta(weeks=4)

        if start_date > end_date:
            continue

        date_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")

        # ---------------------------
        # Forecasting loop
        # ---------------------------
        all_forecasts = []
        for train_end in tqdm(date_range, desc=f"{location}-{species}-ARIMA"):
            train_slice = slice(None, train_end)
            train_y = full_df.loc[train_slice, "mosq_per_trap_night"]
            train_exog = full_df.loc[train_slice, ["actual_temp", "actual_precip"]].dropna()

            train_common_idx = train_y.index.intersection(train_exog.index)
            train_y_use = train_y.loc[train_common_idx]
            train_exog_use = train_exog.loc[train_common_idx]

            if len(train_y_use) < 20: 
                continue

            horizons = [1, 2, 3, 4]
            future_idx = [train_end + pd.Timedelta(weeks=h) for h in horizons]

            if any(d not in full_df.index for d in future_idx):
                continue

            future_exog = full_df.loc[future_idx, ["avg_temp_5yr", "sum_precip_5yr"]].copy()
            future_exog.columns = ["actual_temp", "actual_precip"]

            # Running the model
            try:
                pred_mean, qdict, sigma_val = forecast_arima(
                    train_y_use, train_exog_use, future_exog, 
                    order=ARIMA_ORDER, seasonal_order=SEASONAL_ORDER, 
                    steps=4, quantiles=CDC_QUANTILES_SEQ
                )
            except Exception as e:
                # SARIMAX occasionally fails to converge on highly volatile slices
                # print(f" Convergence failed for {train_end.date()}: {e}")
                continue

            # Output structure
            for step_idx, (h, forecast_date) in enumerate(zip(horizons, future_idx)):
                qvals_h = {k: float(v[step_idx]) for k, v in qdict.items()}
                pm, lo80, up80, lo95, up95, loiqr, upiqr = extract_summary_from_quantiles(qvals_h)

                all_forecasts.append({
                    "model": "ARIMA",
                    "location": location,
                    "species": species,
                    "training_date": train_end.date(),
                    "forecast_date": forecast_date.date(),
                    "forecast_horizon": h,
                    "forecasted_abund": float(pm),
                    "observed_abund": float(full_df.loc[forecast_date, "mosq_per_trap_night"]),
                    "lower.80": float(lo80), "upper.80": float(up80),
                    "lower.95": float(lo95), "upper.95": float(up95),
                    "lower.iqr": float(loiqr), "upper.iqr": float(upiqr),
                    **qvals_h,
                    "current_level_used": np.nan, 
                    "seasonality_t": np.nan,
                    "seasonality_t_plus_i": np.nan,
                    "regime": "SARIMAX",
                    "sigma_used": float(sigma_val)
                })

        # Scoring and saving
        forecast_df = pd.DataFrame(all_forecasts)
        print(f"\n Total forecasts for {location}-{species} : {len(forecast_df)}")

        if not forecast_df.empty:
            print(" Calculating WIS and Coverage metrics...")
            scored_df = calculate_row_metrics(forecast_df)
            
            # Save file
            out_fp = os.path.join(PROCESSED_DATA_DIR, f"arima_{location.lower()}_{species.lower()}_forecast_scored.csv")
            scored_df.to_csv(out_fp, index=False)
            print(f" Saved scored file: {out_fp}")