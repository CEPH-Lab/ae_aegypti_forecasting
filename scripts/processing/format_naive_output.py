import os
import sys
import numpy as np
import pandas as pd

# ===========
# Path Setup
# ===========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.scoring_logic import CDC_QUANTILES_SEQ, calculate_row_metrics

INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def qcol(q: float) -> str:
    return f"q{q:.3f}"

REQUIRED_QCOLS = [qcol(float(q)) for q in CDC_QUANTILES_SEQ]

# ============================================================
# File output reader: reading after "-----"
# ============================================================
def read_quantiles_csv(path: str) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        sep_idx = next((i for i, line in enumerate(lines) if line.strip() == "-----"), None)
        if sep_idx is not None:
            df = pd.read_csv(path, skiprows=sep_idx + 1)
        else:
            df = pd.read_csv(path, skiprows=29)
    except Exception:
        df = pd.read_csv(path, skiprows=29)

    df.columns = [c.strip() for c in df.columns]
    return df

# ============================================================
# File conversion
# ============================================================
def process_single_file(filename: str):
    # Derive location and species from the filename (e.g., "miami_aegypti_forecast.csv")
    base_parts = filename.replace("_forecast.csv", "").split("_")
    if len(base_parts) != 2:
        print(f"   Skipping {filename}: unable to extract location and species from name.")
        return

    location, species = base_parts[0], base_parts[1]
    location_cap, species_cap = location.capitalize(), species.capitalize()
    
    print(f"\n-> Translating Naive Output for {location_cap} - {species_cap}")

    forecast_path = os.path.join(INTERIM_DIR, filename)
    
    # Trying a few common naming conventions for the raw truth data
    truth_path_options = [
        os.path.join(RAW_DIR, location_cap, f"{location_cap}.{species_cap}.csv"),
        os.path.join(RAW_DIR, location_cap, f"{location_cap}_{species_cap}.csv"),
        os.path.join(RAW_DIR, location_cap, f"{location}.{species}.csv")
    ]
    truth_path = next((p for p in truth_path_options if os.path.exists(p)), None)

    if not truth_path:
        print(f"   SKIP (missing truth data): Could not find raw data for {location_cap}.")
        return

    # Read forecast
    forecast_df = read_quantiles_csv(forecast_path)
    forecast_df.dropna(how='all', inplace=True)

    needed_cols = {"day_pres", "weeks_ahead"}
    if not needed_cols.issubset(set(forecast_df.columns)):
        print(f"   SKIP (missing required cols {needed_cols - set(forecast_df.columns)})")
        return

    forecast_df["day_pres"] = pd.to_datetime(forecast_df["day_pres"], errors="coerce")
    forecast_df["weeks_ahead"] = pd.to_numeric(forecast_df["weeks_ahead"], errors="coerce")

    if "truth_vals" in forecast_df.columns:
        forecast_df = forecast_df.drop(columns=["truth_vals"])

    # forecast_date = training_date + (weeks * 7 days)
    forecast_df["forecast_date"] = forecast_df["day_pres"] + pd.to_timedelta(forecast_df["weeks_ahead"] * 7, unit="D")
    forecast_df = forecast_df.rename(columns={"day_pres": "training_date", "weeks_ahead": "forecast_horizon"})

    # Read truth (observed) data
    truth_df = pd.read_csv(truth_path)
    truth_df.columns = [c.strip() for c in truth_df.columns]
    
    if not {"date", "mosq_per_trap_night"}.issubset(set(truth_df.columns)):
        print(f"   SKIP (truth missing date/mosq_per_trap_night cols): {truth_path}")
        return

    truth_df = truth_df.rename(columns={"date": "forecast_date", "mosq_per_trap_night": "observed_abund"})
    truth_df["forecast_date"] = pd.to_datetime(truth_df["forecast_date"], errors="coerce")
    truth_df["observed_abund"] = pd.to_numeric(truth_df["observed_abund"], errors="coerce")

    # Merging truth data with forecast data
    merged = forecast_df.merge(truth_df[["forecast_date", "observed_abund"]], on="forecast_date", how="left")

    # Renaming quantiles
    rename_q = {}
    for c in merged.columns:
        try:
            q = float(c)
            rename_q[c] = qcol(q)
        except Exception:
            pass
    merged = merged.rename(columns=rename_q)

    # Verifying presence of all quantiles
    missing_q = [c for c in REQUIRED_QCOLS if c not in merged.columns]
    if missing_q:
        print(f"   SKIP (missing {len(missing_q)} CDC quantiles): e.g. {missing_q[:3]}")
        return

    # Mapping to common master structure
    merged["model"] = "NAIVE"
    merged["location"] = location_cap
    merged["species"] = species_cap
    
    merged["forecasted_abund"] = merged[qcol(0.500)]
    merged["lower.80"] = merged[qcol(0.100)]
    merged["upper.80"] = merged[qcol(0.900)]
    merged["lower.95"] = merged[qcol(0.025)]
    merged["upper.95"] = merged[qcol(0.975)]
    merged["lower.iqr"] = merged[qcol(0.250)]
    merged["upper.iqr"] = merged[qcol(0.750)]
    
    merged["current_level_used"] = np.nan
    merged["seasonality_t"] = np.nan
    merged["seasonality_t_plus_i"] = np.nan
    merged["regime"] = "MonteCarlo_Original"
    merged["sigma_used"] = np.nan

    final_cols = [
        "model", "location", "species", "training_date", "forecast_date", 
        "forecast_horizon", "forecasted_abund", "observed_abund", 
        "lower.80", "upper.80", "lower.95", "upper.95", "lower.iqr", "upper.iqr"
    ] + REQUIRED_QCOLS + ["current_level_used", "seasonality_t", "seasonality_t_plus_i", "regime", "sigma_used"]

    final = merged[final_cols].copy()
    final["forecast_horizon"] = pd.to_numeric(final["forecast_horizon"], errors="coerce").astype("Int64")
    final = final.sort_values(["training_date", "forecast_horizon", "forecast_date"]).reset_index(drop=True)

    # Lower bound to 0 for biological validity
    clip_cols = ['forecasted_abund', 'lower.80', 'upper.80', 'lower.95', 'upper.95', 'lower.iqr', 'upper.iqr'] + REQUIRED_QCOLS
    final[clip_cols] = final[clip_cols].apply(pd.to_numeric, errors='coerce').clip(lower=0.0)

    # Scoring and saving
    print("  Scoring with Master WIS engine...")
    scored_df = calculate_row_metrics(final)

    out_path = os.path.join(PROCESSED_DIR, f"naive_{location.lower()}_{species.lower()}_forecast_scored.csv")
    scored_df.to_csv(out_path, index=False)
    print(f"   OK -> Saved to {out_path}")

# Batch processing loop
def main():
    if not os.path.exists(INTERIM_DIR):
        print(f"Error: {INTERIM_DIR} does not exist. Run the Naive model first.")
        return

    files_to_process = [f for f in os.listdir(INTERIM_DIR) if f.endswith("_forecast.csv")]
    
    if not files_to_process:
        print(f"No files ending in '_forecast.csv' found in {INTERIM_DIR}.")
        return

    print(f"Found {len(files_to_process)} file(s) in interim folder. Starting batch processing...")
    
    for filename in files_to_process:
        process_single_file(filename)
        
    print("\n Batch processing complete")

if __name__ == '__main__':
    main()