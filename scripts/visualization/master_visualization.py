import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Used cairo for using plots in Illustrator
matplotlib.use('cairo')
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------------------------------
# GLOBAL FONT SETTINGS (Myriad Pro)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Myriad Pro', 'Arial', 'DejaVu Sans']

# --- 1. REPOSITORY PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

MASTER_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_models_master.csv')
BASE_OUTPUT = os.path.join(BASE_DIR, 'visualization',)

ACTIVE_MODELS = ['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm', 'ensemble'] 

# Colors for the models
COLORS = {
    'seasonal_n1': '#a8ddb5', 'seasonal_n2': '#7bccc4', 'seasonal_n3': '#4eb3d3', 'seasonal_n4': '#08589e',
    'naive': '#8c6bb1', 'arima': '#88419d', 'ucm': '#8c96c6', 'ensemble': '#6e016b',
    'keywest': '#fdcc8a', 'la': '#fec44f', 'maricopa': '#fc8d59', 'miami': '#993404',
    1: '#fcc5c0', 2: '#f768a1', 3: '#d7301f', 4: '#ae017e'
}

CUSTOM_MODEL_ORDER = ['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm', 'ensemble']

_LOC_FOLDER_MAP = {'miami': 'Miami', 'maricopa': 'Maricopa', 'la': 'LA', 'keywest': 'KeyWest'}

def load_raw_abundance(location, base_dir):
    folder = _LOC_FOLDER_MAP[location]
    path = os.path.join(base_dir, 'data', 'raw', folder, f'{folder}.Aegypti.csv')
    raw = pd.read_csv(path)
    raw['date'] = pd.to_datetime(raw['date'], infer_datetime_format=True)
    raw['year'] = raw['date'].dt.year
    raw['Week'] = raw['date'].dt.isocalendar().week.astype(int)
    raw = raw.rename(columns={'mosq_per_trap_night': 'observed_abund'})
    return raw.groupby(['year', 'Week'])['observed_abund'].mean().unstack(level=0)


# PLOTTING FUNCTIONS
def plot_relative_wis_heatmap(df, species_target, save_dir):
    """Generates the cross-location per horizon Relative WIS heatmaps."""
    # 1. Aggregate and isolate Naive baseline
    agg_df = df.groupby(['model', 'location', 'forecast_horizon'])['WIS'].mean().reset_index()
    naive_df = agg_df[agg_df['model'] == 'naive']
    naive_ref = naive_df.set_index(['location', 'forecast_horizon'])['WIS'].to_dict()

    # 2. Calculate Relative WIS
    def calculate_relative(row):
        baseline = naive_ref.get((row['location'], row['forecast_horizon']))
        return row['WIS'] / baseline if baseline else None
    
    agg_df['Relative_WIS'] = agg_df.apply(calculate_relative, axis=1)
    plot_df = agg_df[agg_df['model'] != 'naive'].dropna()

    # 3. Setup Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    cmap = sns.diverging_palette(240, 15, s=90, l=50, as_cmap=True)
    all_locations = sorted(plot_df['location'].unique())

    for i, h in enumerate([1, 2, 3, 4]):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        subset = plot_df[plot_df['forecast_horizon'] == h].pivot(
            index="model", columns="location", values="Relative_WIS"
        )
        subset = subset.reindex(index=CUSTOM_MODEL_ORDER, columns=all_locations)
        
        sns.heatmap(subset, ax=ax, annot=True, fmt=".2f", cmap=cmap, center=1, vmin=0.6, vmax=1.4, cbar = False)

        # Prevents text merging
        for index, text in enumerate(ax.texts):
            x, y = text.get_position()
            # Add a microscopic alternating offset (+0.00001, -0.00001) to the Y-coordinate
            jitter = 0.00001 if index % 2 == 0 else -0.00001
            text.set_position((x, y + jitter))
        
        ax.set_title(f'Horizon {h}', fontweight='bold')
        ax.set_xticklabels([loc.capitalize() for loc in all_locations])
        ax.set_xlabel('Location' if row == 1 else '')
        ax.set_ylabel('Model' if col == 0 else '')

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Global Colorbar (top, horizontal)
    fig.subplots_adjust(top=0.91)
    cbar_ax = fig.add_axes([0.35, 0.96, 0.3, 0.01])
    norm = plt.Normalize(vmin=0.6, vmax=1.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='rWIS')

    save_path = os.path.join(save_dir, f"relative_wis_horizon_{species_target}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_wis_per_week(df, species_target, save_dir):
    """Generates a single-column 4-row layout combining Abundance Trajectories and Relative WIS Heatmaps."""
    import matplotlib.gridspec as gridspec
    
    location_grid = ['miami', 'maricopa', 'la', 'keywest']
    
    # 1. Prepare Relative WIS Data
    df['Week'] = df['forecast_date'].dt.isocalendar().week.astype(int)
    weekly_wis = df.groupby(['location', 'model', 'Week'])['WIS'].mean().reset_index()
    naive_baseline = weekly_wis[weekly_wis['model'] == 'naive'][['location', 'Week', 'WIS']].rename(columns={'WIS': 'naive_WIS'})
    plot_df = weekly_wis.merge(naive_baseline, on=['location', 'Week'], how='left')
    plot_df['Relative_WIS'] = plot_df['WIS'] / plot_df['naive_WIS']
    plot_df = plot_df[plot_df['model'] != 'naive'].dropna(subset=['Relative_WIS'])

    max_wis = plot_df['Relative_WIS'].max()
    min_wis = plot_df['Relative_WIS'].min()

    # 2. Setup Grid Plotting
    fig = plt.figure(figsize=(12, 20))
    outer_grid = gridspec.GridSpec(4, 1, hspace=0.5)
    cmap = sns.diverging_palette(240, 15, s=90, l=50, as_cmap=True)

    for row in range(4):
        loc = location_grid[row]
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row], 
                                                      height_ratios=[1, 2], hspace=0.05)
        ax_ts = fig.add_subplot(inner_grid[0])
        ax_hm = fig.add_subplot(inner_grid[1])

        # --- TIME SERIES ---
        yearly_trajectories = load_raw_abundance(loc, BASE_DIR)
        overall_median = yearly_trajectories.median(axis=1)

        for yr in yearly_trajectories.columns:
            y_vals = yearly_trajectories[yr].values
            if not np.all(np.isnan(y_vals)):    
                ax_ts.plot(yearly_trajectories.index.values, y_vals, color="#919191", alpha=0.5, linewidth=1)
            
        ax_ts.plot(overall_median.index.values, overall_median.values, color='black', linewidth=2)
            
        ax_ts.set_title(loc.upper(), fontweight='bold', fontsize=18, pad=10)
        ax_ts.set_xlim(1, 52)
        ax_ts.tick_params(labelbottom=False, bottom=False)
        ax_ts.set_xticks([]) 
        ax_ts.set_ylabel('')
        ax_ts.grid(True, alpha=0.2, linestyle='--')

        # --- HEATMAP ---
        subset = plot_df[plot_df['location'] == loc].pivot(index="model", columns="Week", values="Relative_WIS")
        subset = subset.reindex(index=CUSTOM_MODEL_ORDER, columns=range(1, 53))
            
        sns.heatmap(subset, ax=ax_hm, cmap=cmap, center=1, vmin=0.6, vmax=1.4, annot=False, cbar=False)
            
        if loc == 'la':
            ax_hm.add_patch(plt.Rectangle((0, 0), 21, len(CUSTOM_MODEL_ORDER), facecolor='#D3D3D3', edgecolor='#D3D3D3', zorder=100))
            ax_hm.add_patch(plt.Rectangle((48, 0), 4, len(CUSTOM_MODEL_ORDER), facecolor='#D3D3D3', edgecolor='#D3D3D3', zorder=100))
            ax_hm.text(0.21, 0.5, 'OUT OF SEASON', transform=ax_hm.transAxes, ha='center', va='center', fontweight='bold', color="#5A5A5A", fontsize=10, zorder=101)

        ax_hm.set_xlabel('Week of Year' if row == 4 else '', fontsize=12)
        ax_hm.set_yticklabels([m.upper() for m in CUSTOM_MODEL_ORDER], rotation=0)
        ax_hm.set_ylabel('Models', fontsize=12)

    # Global Colorbar
    fig.subplots_adjust(top = 0.91)
    cbar_ax = fig.add_axes([0.35, 0.96, 0.3, 0.01])
    norm = plt.Normalize(vmin=0.6, vmax=1.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='rWIS (Blue < 1 is Better)')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rel_WIS_Per_Week_{species_target}.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


# --- 3. PROCESSING ENGINE ---
print("Loading Master Data...")
df = pd.read_csv(MASTER_FILE, low_memory=False)

df['model'] = df['model'].astype(str).str.lower()
df['location'] = df['location'].astype(str).str.lower()
df['species'] = df['species'].astype(str).str.lower()

df['forecast_date'] = pd.to_datetime(df['forecast_date'])
df['year'] = df['forecast_date'].dt.year

all_needed_models = ACTIVE_MODELS + ['naive']
df_filtered = df[df['model'].isin(all_needed_models)].copy()

# 1. Overall Relative WIS Heatmaps
print("Generating Cross-Location Heatmaps...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/1_Rel_WIS_Horizon"
    os.makedirs(target_dir, exist_ok=True)
    plot_relative_wis_heatmap(species_df, species_name, target_dir)

# 2. Grid Abundance and WIS Heatmaps
print("Generating 2x2 Grid Abundance/WIS Plots...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/2_Rel_WIS_Weekly"
    plot_wis_per_week(species_df, species_name, target_dir)


print(" Visualizations Complete. All plots successfully saved to visualization/plots_master/")