import pandas as pd
import os

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data", "processed", "all_models_master.csv"))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data", "scores"))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Loading and filtering data
df = pd.read_csv(MASTER_CSV)

df['location'] = df['location'].str.strip().str.lower()
df = df[df['species'].str.lower() == 'aegypti']

locations = ['keywest', 'la', 'maricopa', 'miami']
df = df[df['location'].isin(locations)]

# Aggregating: mean scores per model/location/horizon
agg = (
    df.groupby(['model', 'location', 'forecast_horizon'])[['sMAPE', 'WIS', 'cov_95']]
    .mean()
    .reset_index()
)
agg['cov_95'] = agg['cov_95'] * 100

# Setting model order
model_order = ['NAIVE', 'SEASONAL_n1', 'SEASONAL_n2', 'SEASONAL_n3', 'SEASONAL_n4', 'ARIMA', 'UCM', 'ENSEMBLE']

# Building tables
for loc in locations:
    loc_df = agg[agg['location'] == loc].copy()
    if loc_df.empty:
        continue

    # Pivot: model as rows, (metric, horizon) as columns
    pivoted = loc_df.pivot(index='model', columns='forecast_horizon', values=['sMAPE', 'WIS', 'cov_95'])

    # Group by metric, then horizon: (sMAPE/1,2,3,4) | (WIS/1,2,3,4) | (Coverage/1,2,3,4)
    horizons = sorted(loc_df['forecast_horizon'].unique())
    pivoted = pivoted.reindex(
        columns=pd.MultiIndex.from_product([['sMAPE', 'WIS', 'cov_95'], horizons])
    )

    # Rename cov_95 label
    pivoted.columns = pd.MultiIndex.from_tuples(
        [('Coverage 95%' if metric == 'cov_95' else metric, h)
         for metric, h in pivoted.columns]
    )

    pivoted = pivoted.round(3)

    # Reorder rows by model_order
    pivoted = pivoted.reindex([m for m in model_order if m in pivoted.index])
    pivoted.index.name = 'Model'
    # Reset index so Model is a regular column — avoids a stray index-name row in the CSV
    pivoted = pivoted.reset_index()

    output_file = os.path.join(OUTPUT_DIR, f"final_{loc}_aegypti_comparison.csv")
    pivoted.to_csv(output_file, index=False)
    print(f"Generated Table: {loc}_aegypti")

print("\nAll tables generated.")
