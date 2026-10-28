import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setting up the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

 # Allow importing from utils for scoring function
sys.path.append(BASE_DIR)

# Import the scoring logic
from utils.scoring_logic import CDC_QUANTILES_SEQ, calculate_row_metrics

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ============================================
# CONFIG
# ============================================
LOCATIONS = ["Keywest", "LA", "Maricopa","Miami"]
SPECIES = ["Aegypti"]

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

# ============================================
# MODEL
# ============================================
def forecast_ucm(train_y, train_exog, future_exog, steps=4, quantiles=CDC_QUANTILES_SEQ):
    model = sm.tsa.UnobservedComponents(train_y, level="local level", exog=train_exog, seasonal=52)
    result = model.fit(disp=False)
    fc = result.get_forecast(steps=steps, exog=future_exog)
    mean = fc.predicted_mean.to_numpy()
    se   = fc.se_mean.to_numpy()
    
    z = norm.ppf(np.asarray(quantiles))
    q_mat = mean.reshape(1, -1) + z.reshape(-1, 1) * se.reshape(1, -1)
    
    mean = np.maximum(0.0, mean)
    q_mat = np.maximum(0.0, q_mat)
    
    qdict = {qcol(float(q)): q_mat[i, :] for i, q in enumerate(quantiles)}
    pred_mean = pd.Series(mean, index=fc.predicted_mean.index)

    sigma_used = result.params.get('sigma2.irregular', np.nan)
    
    return pred_mean, qdict, sigma_used

# ============================================
# MAIN
# ============================================
print(f"\n=== Running UCM ===")
for location in LOCATIONS:
    for species in SPECIES:
        print(f"\n-> Processing {location} - {species}")

        abund_fp   = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")
        weather_fp = os.path.join(RAW_DATA_DIR, location, f"weekly_weather_{location}.csv")
        
        if not (os.path.exists(abund_fp) and os.path.exists(weather_fp)):
            print(f" Missing files for {location}-{species}, skipping.")
            continue

        # ---------------------------
        # Loading and aligning data
        # ---------------------------
        abund_df = pd.read_csv(abund_fp)
        weather_df = pd.read_csv(weather_fp)

        abund_df["date"] = pd.to_datetime(abund_df["date"])
        weather_df["week_end"] = pd.to_datetime(weather_df["week_end"])

        # Linear interpolation for missing data
        abund_df = abund_df.set_index("date")
        abund_df.index = abund_df.index.to_period("W-SUN").end_time.normalize()

        full_index = pd.date_range(start=abund_df.index.min(), end=abund_df.index.max(), freq="W-SUN")
        full_df = abund_df.reindex(full_index)
        
        # Draw a straight line between missing dates
        full_df['mosq_per_trap_night'] = full_df['mosq_per_trap_night'].interpolate(method='linear')

        # Merging weather into framework
        weather_actual = weather_df[["week_end", "weekly_avg_temperature", "sum_precipitation", "avg_temp_5yr", "sum_precip_5yr"]].rename(
            columns={"week_end": "date", "weekly_avg_temperature": "actual_temp", "sum_precipitation": "actual_precip"}
        ).set_index("date")
        weather_actual.index = weather_actual.index.to_period("W-SUN").end_time.normalize()

        full_df = full_df.merge(weather_actual, left_index=True, right_index=True, how="left")

        # Fill actual weather holes with weather
        full_df["actual_temp"] = full_df["actual_temp"].fillna(full_df["avg_temp_5yr"])
        full_df["actual_precip"] = full_df["actual_precip"].fillna(full_df["sum_precip_5yr"])

        initial_train_end = full_df.index.min() + pd.DateOffset(years=2)
        start_date = initial_train_end
        end_date = full_df.index.max() - pd.Timedelta(weeks=4)

        if start_date > end_date:
            continue

        date_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")

        # ---------------------------
        # Forecast Loop
        # ---------------------------
        all_forecasts = []
        for train_end in tqdm(date_range, desc=f"{location}-{species}"):
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

            # Running model
            pred_mean, qdict, sigma_val = forecast_ucm(
                train_y_use, train_exog_use, future_exog, steps=4, quantiles=CDC_QUANTILES_SEQ
            )

            # Assembling results
            for step_idx, (h, forecast_date) in enumerate(zip(horizons, future_idx)):
                qvals_h = {k: float(v[step_idx]) for k, v in qdict.items()}
                pm, lo80, up80, lo95, up95, loiqr, upiqr = extract_summary_from_quantiles(qvals_h)

                all_forecasts.append({
                    "model": "UCM",
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
                    "current_level_used": np.nan, # Placeholders for UCM state variables
                    "seasonality_t": np.nan,
                    "regime": "local_level",
                    "sigma_used": float(sigma_val)
                })

        # Scoring and saving
        forecast_df = pd.DataFrame(all_forecasts)
        print(f"\n Total forecasts for {location}-{species} : {len(forecast_df)}")

        if not forecast_df.empty:
            # Calculate row-by-row sMAPE, WIS, and Coverage
            print(" Calculating WIS and Coverage metrics...")
            scored_df = calculate_row_metrics(forecast_df)
            
            out_fp = os.path.join(PROCESSED_DATA_DIR, f"ucm_{location.lower()}_{species.lower()}_forecast_scored.csv")
            scored_df.to_csv(out_fp, index=False)
            print(f" Saved scored file: {out_fp}")