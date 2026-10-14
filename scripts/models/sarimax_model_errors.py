import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ============================================
# PATH SETUP
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
# Create a folder to save diagnostic plots if you want
PLOTS_DIR = os.path.join(BASE_DIR, "visualization", "sarimax_diagnostics")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================
# CONFIG
# ============================================
LOCATIONS = ["Keywest", "LA", "Maricopa", "Miami"]
SPECIES = ["Aegypti"]

ARIMA_ORDER = (1, 1, 2)
SEASONAL_ORDER = (1, 0, 0, 52)

# ============================================
# MAIN
# ============================================
print(f"\n=== Analyzing ARIMA {ARIMA_ORDER} | {SEASONAL_ORDER} Error Distributions ===")

for location in LOCATIONS:
    for species in SPECIES:
        print(f"\n Analyzing 2-Year Error Distribution for {location} - {species}")

        abund_fp   = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")
        weather_fp = os.path.join(RAW_DATA_DIR, location, f"weekly_weather_{location}.csv")
        
        if not (os.path.exists(abund_fp) and os.path.exists(weather_fp)):
            print(f" Missing files for {location}-{species}, skipping.")
            continue

        # 1. Exact Data Prep from your Main Script
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

        # Linear interpolation
        abund_df = abund_df.set_index("date")
        abund_df.index = abund_df.index.to_period("W").start_time

        full_index = pd.date_range(start=abund_df.index.min(), end=abund_df.index.max(), freq="W-MON")
        full_df = abund_df.reindex(full_index)
        full_df['mosq_per_trap_night'] = full_df['mosq_per_trap_night'].interpolate(method='linear')

        weather_actual = weather_df[["custom_week_start", "weekly_avg_temperature", "sum_precipitation", "avg_temp_5yr", "sum_precip_5yr"]].rename(
            columns={"custom_week_start": "date", "weekly_avg_temperature": "actual_temp", "sum_precipitation": "actual_precip"}
        ).set_index("date")
        weather_actual.index = weather_actual.index.to_period("W").start_time

        full_df = full_df.merge(weather_actual, left_index=True, right_index=True, how="left")

        weeks_idx = full_df.index.isocalendar().week
        full_df["avg_temp_5yr"] = full_df["avg_temp_5yr"].fillna(weeks_idx.map(latest_climo["avg_temp_5yr"]))
        full_df["sum_precip_5yr"] = full_df["sum_precip_5yr"].fillna(weeks_idx.map(latest_climo["sum_precip_5yr"]))

        full_df["actual_temp"] = full_df["actual_temp"].fillna(full_df["avg_temp_5yr"])
        full_df["actual_precip"] = full_df["actual_precip"].fillna(full_df["sum_precip_5yr"])

        # 2. Slice out the first 2 years
        initial_train_end = full_df.index.min() + pd.DateOffset(years=2)
        train_slice = slice(None, initial_train_end)
        
        train_y = full_df.loc[train_slice, "mosq_per_trap_night"]
        train_exog = full_df.loc[train_slice, ["actual_temp", "actual_precip"]].dropna()

        train_common_idx = train_y.index.intersection(train_exog.index)
        train_y_use = train_y.loc[train_common_idx]
        train_exog_use = train_exog.loc[train_common_idx]

        if len(train_y_use) < 52:
            print(f" Not enough data for 2 years in {location}-{species} (Found {len(train_y_use)} weeks).")
            continue

        # 3. Fit the Model
        print(f" Fitting SARIMAX on initial 2-year window ({len(train_y_use)} weeks)...")
        model = sm.tsa.SARIMAX(
            train_y_use, 
            exog=train_exog_use, 
            order=ARIMA_ORDER, 
            seasonal_order=SEASONAL_ORDER
        )
        try:
            result = model.fit(maxiter=50, disp=False)
        except Exception as e:
            print(f" Failed to converge: {e}")
            continue

        # 4. Extract and Plot Residuals
        # We skip the first 52 weeks of residuals because the seasonal lag (s=52)
        # means the model doesn't have a full year of history to use for the first year,
        # resulting in artificially inflated initial errors.
        residuals = result.resid.iloc[52:]

        # Print summary statistics to the console
        print(f"\n--- Residual (Error) Stats for {location} ---")
        print(f"Mean Error:      {residuals.mean():.4f}")
        print(f"Standard Dev:    {residuals.std():.4f}")
        print(f"Min Error:       {residuals.min():.4f}")
        print(f"Max Error:       {residuals.max():.4f}")

        # Create diagnostic plots
        fig = result.plot_diagnostics(figsize=(14, 9))
        fig.suptitle(f"Error Distribution (First 2 Years) - {location} {species}\nARIMA {ARIMA_ORDER} {SEASONAL_ORDER}", y=1.02, fontsize=16)
        
        # Save the plot (optional, highly recommended if looping multiple locations)
        plot_path = os.path.join(PLOTS_DIR, f"error_dist_{location}_{species}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f" Saved diagnostic plot to {plot_path}")
        
        # Display the plot
        plt.show()

print("\n=== Finished Analysis ===")