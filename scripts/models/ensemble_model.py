import os
import sys
import glob
import pandas as pd

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.scoring_logic import calculate_row_metrics

def main():
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUT_FILE = os.path.join(PROCESSED_DIR, "all_models_master.csv")

    print("\n-> Step 1: Loading all scored baseline models...")
    
    # Finds all processed forecast files, EXCLUDING any old master/ensemble files to prevent infinite loops
    search_pattern = os.path.join(PROCESSED_DIR, "*_forecast_scored.csv")
    files = [f for f in glob.glob(search_pattern) if "ensemble" not in f.lower() and "all_models" not in f.lower()]
    
    if not files:
        print(" No processed models found in data/processed/. Run your baseline models first!")
        return

    raw_data_list = []
    for f in files:
        df = pd.read_csv(f)
        raw_data_list.append(df)
        print(f"  Loaded: {os.path.basename(f)} ({len(df)} rows)")

    # Stack of all pre-scored models (Naive, Seasonal, SARIMAX, UCM)
    master_df = pd.concat(raw_data_list, ignore_index=True)

    print("\n --> Step 2: Generating the Median Ensemble...")
    # Filter out naive model for the ensemble calculation
    ensemble_input = master_df[master_df['model'].str.upper() != 'NAIVE'].copy()

    # Grouping columns
    group_cols = ['location', 'species', 'training_date', 'forecast_date', 'forecast_horizon']
    
    q_cols = [c for c in master_df.columns if c.startswith('q0.')]
    bound_cols = ['forecasted_abund', 'observed_abund', 'lower.80', 'upper.80', 'lower.95', 'upper.95', 'lower.iqr', 'upper.iqr']
    target_cols = bound_cols + q_cols

    # Calculate the median across all numeric columns (all quantiles, observed abund, forecasted_abund)
    ensemble_df = ensemble_input.groupby(group_cols)[target_cols].mean(numeric_only=True).reset_index()
    
    # Ensemble rows
    ensemble_df['model'] = 'ENSEMBLE'
    ensemble_df['regime'] = 'Mean_Aggregation'
    
    print(f"  Generated {len(ensemble_df)} new Ensemble rows.")

    print("\n --> Step 3: Scoring the new Ensemble model...")
    # Passing only the new ensemble rows through your scoring function
    ensemble_scored_df = calculate_row_metrics(ensemble_df)

    print("\n --> Step 4: Stacking and Exporting...")
    # Stack the newly scored Ensemble on top of the original pre-scored baselines
    final_master_df = pd.concat([master_df, ensemble_scored_df], ignore_index=True)

    final_master_df = final_master_df.sort_values(
        by=['location', 'species', 'model', 'training_date', 'forecast_date', 'forecast_horizon']
    ).reset_index(drop=True)

    # Export the final combined file
    final_master_df.to_csv(OUTPUT_FILE, index=False)
    print(f" SUCCESS! Master dataset consolidated and saved to:\n   {OUTPUT_FILE}")

if __name__ == '__main__':
    main()