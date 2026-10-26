import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Used cairo for editing figures in Illustrator
matplotlib.use('cairo')
import matplotlib.dates as mdates
import os
import sys
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

MASTER_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'ALL_MODELS_MASTER.csv')
DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'visualization', '3_Flusight_TimeSeries')

# --- CUSTOM COLOR SCHEME ---
COLORS = {
    'seasonal_n1': '#8c96c6',
    'seasonal_n2': '#8c6bb1',
    'seasonal_n3': '#88419d',
    'seasonal_n4': '#6e016b',
    'arima': '#7bccc4',
    'ucm': '#4eb3d3',
    'ensemble': '#08589e',
    'naive': '#d7301f',
    'seasonal_naive': '#fc8d59'
}

def plot_mosquito_snapshots(
    date_range,
    models,
    intervals,
    location,
    species,
    forecast_origins,
    output_folder=DEFAULT_OUTPUT
):
    # 1. Setup and Data Loading
    print(f"Generating Flusight Snapshot for {location.upper()}...")
    df = pd.read_csv(MASTER_FILE, low_memory=False)
    
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['training_date'] = pd.to_datetime(df['training_date'])
    
    # Standardize strings for matching
    df['location'] = df['location'].astype(str).str.lower().str.strip()
    df['model'] = df['model'].astype(str).str.lower().str.strip()
    df['species'] = df['species'].astype(str).str.lower().str.strip()

    os.makedirs(output_folder, exist_ok=True)

    # 2. Filtering Observed data
    truth_df = df[(df['location'] == location.lower()) & 
                  (df['species'].str.contains(species.lower(), case=False, na=False)) & 
                  (df['forecast_date'] >= date_range[0]) & 
                  (df['forecast_date'] <= date_range[1])]
    
    truth_series = truth_df[['forecast_date', 'observed_abund']].drop_duplicates().sort_values('forecast_date')

    # PLOTTING LOGIC
    plt.figure(figsize=(14, 7))
    
    # Plot Observed data: Black line and dots
    plt.plot(truth_series['forecast_date'].values, 
             truth_series['observed_abund'].values, 
             color='black', marker='o', linestyle='-', markersize=4, label='Ground Truth', zorder=2, alpha=1, linewidth = 1.5)
    
    # Tracking labels to avoid duplicate legends
    plotted_labels = set()

    # 3. Loop through origins and models to add to the single plot
    for origin_idx, origin in enumerate(forecast_origins):
        origin_dt = pd.to_datetime(origin)
        
        # Find observed value for this specific origin date to center the line
        origin_val_row = truth_series[truth_series['forecast_date'] == origin_dt]
        
        if not origin_val_row.empty:
            y_val = origin_val_row['observed_abund'].values[0]
            
            # Determines how far up/down the line should go
            line_length = 5 
            
            plt.vlines(x=origin_dt, 
                       ymin=y_val - (line_length / 2), 
                       ymax=y_val + (line_length / 2), 
                       color='black', linestyle='--', linewidth=1.5, zorder=1)

        for mod_idx, model_name in enumerate(models):
            # Filter for specific model trajectory starting from this origin
            m_key = model_name.lower()
            model_df = df[(df['model'] == m_key) & 
                          (df['location'] == location.lower()) & 
                          (df['species'].str.contains(species.lower(), case=False, na=False)) & 
                          (df['training_date'] == origin_dt)].sort_values('forecast_date')

            if model_df.empty:
                print(f"  Warning: No data for {model_name} starting at {origin}. Skipping.")
                continue

            color = COLORS.get(m_key, '#333333')
            
            # Plot intervals
            for interval in sorted(intervals, reverse=True):
                alpha_val = (100 - interval) / 200
                lower_col = f"q{alpha_val:.3f}"
                upper_col = f"q{1-alpha_val:.3f}"
                
                # Only add to legend once per model
                label = f'{model_name} {interval}% prediction interval' if f'{model_name}_{interval}' not in plotted_labels else ""
                plt.fill_between(model_df['forecast_date'].values, 
                                 model_df[lower_col].values, 
                                 model_df[upper_col].values, 
                                 color=color, alpha=0.10, label=label)
                plotted_labels.add(f'{m_key}_{interval}')

            # Plot forecast points and line
            label = f'{model_name} prediction' if model_name not in plotted_labels else ""
            plt.plot(model_df['forecast_date'].values, 
                     model_df['q0.500'].values, 
                     color=color, marker='o', linestyle='-', markersize=5, 
                     linewidth=1.5, label=label, zorder=3)
            plotted_labels.add(m_key)

    # 4. Formatting
    plt.title(f"{location.upper()} | {species.upper()} | Forecast Comparison", fontsize=24, fontweight='bold')
    plt.ylabel("Number of collected Ae. aegypti per trap night")
    plt.xlabel("")
    plt.xlim(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    
    # Centered month labels
    ax = plt.gca()
    
    # Draws boundary ticks on the 1st of every month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    
    # Places text on the 15th of every month as 'Jan', 'Feb', etc.
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    
    # Hiding the physical tick marks for the minor text
    ax.tick_params(axis='x', which='minor', length=0, labelsize=11)
    # Making the major boundary ticks longer to separate the months
    ax.tick_params(axis='x', which='major', length=6, color='gray')
    # -----------------------------------
    
    # Place legend outside
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # 5. Filename logic
    clean_start = date_range[0].replace("-", "")
    model_names_joined = "_".join([m.lower() for m in models])
    
    file_name = f"{location}_{species}_{clean_start}_{model_names_joined}.pdf"
    save_path = os.path.join(output_folder, file_name)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f" Successfully generated combined plot: {save_path}")

# ========================================================
# EXECUTION BLOCKS
# ========================================================
if __name__ == '__main__':

    ## MIAMI ##
    # Ensemble plot
    # plot_mosquito_snapshots(
    #     date_range=['2021-01-01', '2021-12-31'],
    #     models=['ensemble'],
    #     intervals=[50, 95],
    #     location='miami',
    #     species='aegypti',
    #     forecast_origins=['2021-01-25', '2021-04-05', '2021-05-31', '2021-07-19', '2021-09-27', '2021-11-29']
    # )

    # # Naive and Seasonal Naive models plot
    # plot_mosquito_snapshots(
    #     date_range=['2021-01-01', '2021-12-31'],
    #     models=['naive', 'seasonal_naive'],
    #     intervals=[95],
    #     location='miami',
    #     species='aegypti',
    #     forecast_origins=['2021-01-25', '2021-04-05', '2021-05-31', '2021-07-19', '2021-09-27', '2021-11-29']
    # )

    # STL1, STL2, STL3, STL4, SARIMAX, UCM models models plot
    plot_mosquito_snapshots(
        date_range=['2021-01-01', '2021-12-31'],
        models=['arima'],
#        models=['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm'],
        intervals=[95],
        location='miami',
        species='aegypti',
        forecast_origins=['2021-01-25', '2021-04-05', '2021-05-31', '2021-07-19', '2021-09-27', '2021-11-29']
    )

    # ## MARICOPA
    # # Ensemble plot
    # plot_mosquito_snapshots(
    #     date_range=['2021-01-01', '2021-12-31'],
    #     models=['ensemble'],
    #     intervals=[50, 95],
    #     location='maricopa',
    #     species='aegypti',
    #     forecast_origins=['2021-01-25', '2021-04-05', '2021-05-17', '2021-06-21', '2021-07-26', '2021-08-23', '2021-10-04', '2021-11-29']
    # )

    # # Naive and Seasonal Naive models plot
    # plot_mosquito_snapshots(
    #     date_range=['2021-01-01', '2021-12-31'],
    #     models=['naive', 'seasonal_naive'],
    #     intervals=[95],
    #     location='maricopa',
    #     species='aegypti',
    #     forecast_origins=['2021-01-25', '2021-04-05', '2021-05-17', '2021-06-21', '2021-07-26', '2021-08-23', '2021-10-04', '2021-11-29']
    # )

    # # STL1, STL2, STL3, STL4, SARIMAX, UCM models models plot
    # plot_mosquito_snapshots(
    #     date_range=['2021-01-01', '2021-12-31'],
    #     models=['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm'],
    #     intervals=[95],
    #     location='maricopa',
    #     species='aegypti',
    #     forecast_origins=['2021-01-25', '2021-04-05', '2021-05-17', '2021-06-21', '2021-07-26', '2021-08-23', '2021-10-04', '2021-11-29']
    # )