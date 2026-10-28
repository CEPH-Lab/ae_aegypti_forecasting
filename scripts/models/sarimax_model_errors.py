import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
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
        print(f"\n Analyzing Multi-Year Error Distribution for {location} - {species}")

        abund_fp   = os.path.join(RAW_DATA_DIR, location, f"{location}.{species}.csv")
        weather_fp = os.path.join(RAW_DATA_DIR, location, f"weekly_weather_{location}.csv")
        
        if not (os.path.exists(abund_fp) and os.path.exists(weather_fp)):
            print(f" Missing files for {location}-{species}, skipping.")
            continue

        # 1. Exact Data Prep from your Main Script
        abund_df = pd.read_csv(abund_fp)
        weather_df = pd.read_csv(weather_fp)

        abund_df["date"] = pd.to_datetime(abund_df["date"])
        weather_df["week_end"] = pd.to_datetime(weather_df["week_end"])

        # Linear interpolation
        abund_df = abund_df.set_index("date")
        abund_df.index = abund_df.index.to_period("W").end_time.normalize()

        full_index = pd.date_range(start=abund_df.index.min(), end=abund_df.index.max(), freq="W-SUN")
        full_df = abund_df.reindex(full_index)
        full_df['mosq_per_trap_night'] = full_df['mosq_per_trap_night'].interpolate(method='linear')

        weather_actual = weather_df[["week_end", "weekly_avg_temperature", "sum_precipitation", "avg_temp_5yr", "sum_precip_5yr"]].rename(
            columns={"week_end": "date", "weekly_avg_temperature": "actual_temp", "sum_precipitation": "actual_precip"}
        ).set_index("date")
        weather_actual.index = weather_actual.index.to_period("W").end_time.normalize()

        full_df = full_df.merge(weather_actual, left_index=True, right_index=True, how="left")

        weeks_idx = full_df.index.isocalendar().week

        full_df["actual_temp"] = full_df["actual_temp"].fillna(full_df["avg_temp_5yr"])
        full_df["actual_precip"] = full_df["actual_precip"].fillna(full_df["sum_precip_5yr"])

        # Train on the whole data
        initial_train_end = full_df.index.max()
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
        print(f" Fitting SARIMAX on whole time period of ({len(train_y_use)} weeks)...")
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
        import scipy.stats as stats

        # 4. Extract and Plot Residuals
        # Skip the first 52 weeks of residuals
        residuals = result.resid.iloc[52:]

        # Print summary statistics to the console
        print(f"\n--- Residual (Error) Stats for {location} ---")
        print(f"Mean Error:      {residuals.mean():.4f}")
        print(f"Standard Dev:    {residuals.std():.4f}")
        print(f"Min Error:       {residuals.min():.4f}")
        print(f"Max Error:       {residuals.max():.4f}")

        # Create a 2x2 grid for diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Plot 1: Mistakes Over Time
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_title("1. Mistakes Over Time")

        # Plot 2: Histogram + Density (INCREASED BINS HERE)
        # Change 'bins=30' to whatever number you prefer
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Calculate and plot the actual shape (KDE)
        kde = sm.nonparametric.KDEUnivariate(residuals)
        kde.fit()
        axes[0, 1].plot(kde.support, kde.density, color='red', label='Actual Shape')
        
        # Calculate and plot the perfect bell curve (Normal Dist)
        x_vals = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x_vals, stats.norm.pdf(x_vals, loc=residuals.mean(), scale=residuals.std()), 
                        color='green', linestyle='--', label='Normal Bell Curve')
        axes[0, 1].legend()
        axes[0, 1].set_title("2. Size of Mistakes")

        # Plot 3: Q-Q Plot
        sm.qqplot(residuals, line='s', ax=axes[1, 0])
        axes[1, 0].set_title("3. Normal Q-Q")

        # Plot 4: Correlogram (ACF)
        max_lags = min(len(residuals) // 2 - 1, 10)
        sm.graphics.tsa.plot_acf(residuals, ax=axes[1, 1], lags=max_lags)
        axes[1, 1].set_title("4. Correlogram (Do past mistakes predict future ones?)")

        fig.suptitle(f"SARIMAX Error Distribution ({len(train_y_use)} weeks) - {location} {species}", fontsize=16)
        
        # Use tight_layout with a rect parameter to perfectly space the title and subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plot_path = os.path.join(PLOTS_DIR, f"sarimax_error_dist_{location}_{species}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f" Saved diagnostic plot to {plot_path}")

print("\n=== Finished Analysis ===")