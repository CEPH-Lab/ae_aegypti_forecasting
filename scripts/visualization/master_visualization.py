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
BASE_OUTPUT = os.path.join(BASE_DIR, 'visualization', 'plots_master')

ACTIVE_MODELS = ['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm', 'ensemble'] 

# Colors for the models
COLORS = {
    'seasonal_n1': '#a8ddb5', 'seasonal_n2': '#7bccc4', 'seasonal_n3': '#4eb3d3', 'seasonal_n4': '#08589e',
    'naive': '#8c6bb1', 'arima': '#88419d', 'ucm': '#8c96c6', 'ensemble': '#6e016b',
    'keywest': '#fdcc8a', 'la': '#fec44f', 'maricopa': '#fc8d59', 'miami': '#993404',
    1: '#fcc5c0', 2: '#f768a1', 3: '#d7301f', 4: '#ae017e'
}

CUSTOM_MODEL_ORDER = ['seasonal_n1', 'seasonal_n2', 'seasonal_n3', 'seasonal_n4', 'arima', 'ucm', 'ensemble']

# --- 2. PLOTTING FUNCTIONS ---

def save_diagnostic_set(df, hue_col, title_prefix, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    plot_df = df.copy().reset_index(drop=True)
    
    # A. Scatter
    plt.figure(figsize=(7, 7))
    for label, group in plot_df.groupby(hue_col):
        plt.scatter(group['observed_abund'].values, group['q0.500'].values, 
                    label=label, color=COLORS.get(label, '#808080'), alpha=0.5, s=20)
    limit = max(plot_df['observed_abund'].max(), plot_df['q0.500'].max())
    plt.plot([0, limit], [0, limit], 'k--', alpha=0.3)
    plt.title(f"{title_prefix} - Accuracy")
    if plt.gca().get_legend_handles_labels()[0]: plt.legend()
    sns.despine()
    plt.savefig(f"{folder_path}/scatter.pdf", dpi=300); plt.close()

    # B. sMAPE
    plt.figure(figsize=(7, 7))
    sns.boxplot(data=plot_df, x=hue_col, y='sMAPE', hue=hue_col, palette=COLORS, width=0.6,
                showfliers=False, legend=False,
                medianprops={'color': 'black', 'linewidth': 1.5},
                boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                whiskerprops={'color': 'black', 'linewidth': 0.8},
                capprops={'color': 'black', 'linewidth': 0.8})
    plt.title(f"{title_prefix} - sMAPE"); sns.despine()
    plt.savefig(f"{folder_path}/sMAPE_violin.pdf", dpi=300); plt.close()

    # C. Calibration
    cov_cols = [c for c in plot_df.columns if c.startswith('cov_')]
    melted = plot_df.melt(id_vars=[hue_col], value_vars=cov_cols, var_name='w', value_name='in')
    melted['w'] = melted['w'].str.replace('cov_', '').astype(float) / 100
    cal = melted.groupby([hue_col, 'w'])['in'].mean().reset_index()
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    for label, group in cal.groupby(hue_col):
        plt.plot(group['w'].values, group['in'].values, marker='o', label=label, color=COLORS.get(label, '#808080'), linewidth=2)
    plt.title(f"{title_prefix} - Calibration")
    if plt.gca().get_legend_handles_labels()[0]: plt.legend()
    sns.despine()
    plt.savefig(f"{folder_path}/calibration.pdf", dpi=300); plt.close()

def plot_relative_wis_robust(df, base_model='naive', save_path=None, title=""):
    baseline_dates = df[df['model'] == base_model][['forecast_date', 'forecast_horizon']]
    robust_df = df.merge(baseline_dates, on=['forecast_date', 'forecast_horizon'], how='inner')

    date_model_wis = robust_df.groupby(['forecast_date', 'model'])['WIS'].mean().reset_index()
    date_model_wis['week'] = date_model_wis['forecast_date'].dt.isocalendar().week
    seasonal_profile = date_model_wis.groupby(['week', 'model'])['WIS'].mean().reset_index()

    pivot_seasonal = seasonal_profile.pivot(index='week', columns='model', values='WIS')
    rel_seasonal = pivot_seasonal.div(pivot_seasonal[base_model], axis=0)
    
    models_to_plot = [m for m in rel_seasonal.columns if m != base_model]

    plt.figure(figsize=(14, 7))
    for model_name in models_to_plot:
        model_data = rel_seasonal[model_name].dropna()
        if not model_data.empty:
            plt.plot(model_data.index.values, model_data.values, 
                     marker='o', label=model_name, 
                     color=COLORS.get(model_name, '#808080'), linewidth=2, markersize=5)
    
    plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label=f'Baseline ({base_model})')
    plt.ylim(0, 2.0)
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Relative WIS (Horizon-Averaged)")
    plt.xlabel("ISO Week of Year")
    plt.xticks(range(1, 54, 4)) 
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Models")
    sns.despine()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

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
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1.3]})
    cmap = sns.diverging_palette(240, 15, s=90, l=50, as_cmap=True)
    all_locations = sorted(plot_df['location'].unique())

    for i, h in enumerate([1, 2, 3, 4]):
        subset = plot_df[plot_df['forecast_horizon'] == h].pivot(
            index="model", columns="location", values="Relative_WIS"
        )
        subset = subset.reindex(index=CUSTOM_MODEL_ORDER, columns=all_locations)
        
        sns.heatmap(subset, ax=axes[i], annot=True, fmt=".2f", cmap=cmap, center=1, vmin=0.6, vmax=1.4, 
                    cbar=(i == 3), cbar_kws={'label': 'Relative WIS'} if i == 3 else None)
        
        axes[i].set_title(f'Horizon {h}', fontweight='bold')
        axes[i].set_xticklabels([loc.capitalize() for loc in all_locations])
        axes[i].set_xlabel('Location')
        axes[i].set_ylabel('Model' if i == 0 else '')

    plt.suptitle(f'Relative Performance (WIS) - Species: {species_target.capitalize()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    save_path = os.path.join(save_dir, f"relative_wis_horizon_{species_target}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_relative_smape_heatmap(df, species_target, save_dir):
    """Generates the cross-location per horizon Relative sMAPE heatmaps."""
    # 1. Aggregate and isolate Naive baseline
    agg_df = df.groupby(['model', 'location', 'forecast_horizon'])['sMAPE'].mean().reset_index()
    naive_df = agg_df[agg_df['model'] == 'naive']
    naive_ref = naive_df.set_index(['location', 'forecast_horizon'])['sMAPE'].to_dict()

    # 2. Calculate Relative WIS
    def calculate_relative(row):
        baseline = naive_ref.get((row['location'], row['forecast_horizon']))
        return row['sMAPE'] / baseline if baseline else None
    
    agg_df['Relative_sMAPE'] = agg_df.apply(calculate_relative, axis=1)
    plot_df = agg_df[agg_df['model'] != 'naive'].dropna()

    # 3. Setup Grid
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1.3]})
    cmap = sns.diverging_palette(240, 15, s=90, l=50, as_cmap=True)
    all_locations = sorted(plot_df['location'].unique())

    for i, h in enumerate([1, 2, 3, 4]):
        subset = plot_df[plot_df['forecast_horizon'] == h].pivot(
            index="model", columns="location", values="Relative_sMAPE"
        )
        subset = subset.reindex(index=CUSTOM_MODEL_ORDER, columns=all_locations)
        
        sns.heatmap(subset, ax=axes[i], annot=True, fmt=".2f", cmap=cmap, center=1, vmin=0.6, vmax=1.4, 
                    cbar=(i == 3), cbar_kws={'label': 'Relative sMAPE'} if i == 3 else None)
        
        axes[i].set_title(f'Horizon {h}', fontweight='bold')
        axes[i].set_xticklabels([loc.capitalize() for loc in all_locations])
        axes[i].set_xlabel('Location')
        axes[i].set_ylabel('Model' if i == 0 else '')

    plt.suptitle(f'Relative Performance (sMAPE) - Species: {species_target.capitalize()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    save_path = os.path.join(save_dir, f"relative_smape_horizon_{species_target}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_absolute_smape_heatmap(df, species_target, save_dir):
    """Absolute sMAPE heatmap: cells show raw sMAPE, color encodes ratio to horizon 1 (white=H1, green=better, red=worse)."""
    all_model_order = ['naive'] + CUSTOM_MODEL_ORDER

    agg_df = df.groupby(['model', 'location', 'forecast_horizon'])['sMAPE'].mean().reset_index()

    # Using each model's sMAPE at horizon 1 for the same location as reference
    h1_ref = agg_df[agg_df['forecast_horizon'] == 1].set_index(['model', 'location'])['sMAPE'].to_dict()

    def compute_ratio(row):
        baseline = h1_ref.get((row['model'], row['location']))
        return row['sMAPE'] / baseline if baseline else 1.0

    agg_df['ratio'] = agg_df.apply(compute_ratio, axis=1)
    all_locations = sorted(agg_df['location'].unique())

    # Symmetric scale around 1 based on actual data spread
    max_dev = max(abs(agg_df['ratio'].max() - 1), abs(agg_df['ratio'].min() - 1))
    vmin, vmax = 1 - max_dev, 1 + max_dev

    # green is better than H1, red is worse, white=H1 (ratio=1=center)
    cmap = sns.diverging_palette(130, 10, s=90, l=50, as_cmap=True)

    n_locs = len(all_locations)
    width_ratios = [1] * (n_locs - 1) + [1.3]
    fig, axes = plt.subplots(1, n_locs, figsize=(24, 7), sharey=True, gridspec_kw={'width_ratios': width_ratios})

    for i, loc in enumerate(all_locations):
        loc_df = agg_df[agg_df['location'] == loc]

        color_data = loc_df.pivot(index="model", columns="forecast_horizon", values="ratio")
        color_data = color_data.reindex(index=all_model_order, columns=[1, 2, 3, 4])

        annot_data = loc_df.pivot(index="model", columns="forecast_horizon", values="sMAPE")
        annot_data = annot_data.reindex(index=all_model_order, columns=[1, 2, 3, 4])

        last = (i == n_locs - 1)
        sns.heatmap(color_data, ax=axes[i], annot=annot_data, fmt=".2f", cmap=cmap,
                    center=1, vmin=vmin, vmax=vmax,
                    cbar=last, cbar_kws={'label': 'sMAPE ratio to horizon 1', 'ticks': [vmin, 1, vmax]} if last else None)

        axes[i].set_title(loc.capitalize(), fontweight='bold')
        axes[i].set_xticklabels([f'H{h}' for h in [1, 2, 3, 4]])
        axes[i].set_xlabel('Horizon')
        axes[i].set_ylabel('Model' if i == 0 else '')

    cbar = axes[n_locs - 1].collections[0].colorbar
    cbar.set_ticklabels([f'{vmin:.2f}× (better)', '1.00× (H1)', f'{vmax:.2f}× (worse)'])

    plt.suptitle(f'Absolute sMAPE by Horizon - Species: {species_target.capitalize()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    save_path = os.path.join(save_dir, f"absolute_smape_horizon_{species_target}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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


def plot_wis_per_week(df, species_target, save_dir):
    """Generates the 2x2 grid combining Abundance Trajectories and Relative WIS Heatmaps."""
    import matplotlib.gridspec as gridspec
    
    location_grid = [['miami', 'maricopa'], ['la', 'keywest']]
    
    # 1. Prepare Relative WIS Data
    df['Week'] = df['forecast_date'].dt.isocalendar().week.astype(int)
    weekly_wis = df.groupby(['location', 'model', 'Week'])['WIS'].mean().reset_index()
    naive_baseline = weekly_wis[weekly_wis['model'] == 'naive'][['location', 'Week', 'WIS']].rename(columns={'WIS': 'naive_WIS'})
    plot_df = weekly_wis.merge(naive_baseline, on=['location', 'Week'], how='left')
    plot_df['Relative_WIS'] = plot_df['WIS'] / plot_df['naive_WIS']
    plot_df = plot_df[plot_df['model'] != 'naive'].dropna(subset=['Relative_WIS'])

    max_wis = plot_df['Relative_WIS'].max()
    min_wis = plot_df['Relative_WIS'].min()
    print(f" Relative WIS Range: Min = {min_wis:.2f}, Max = {max_wis:.2f}")


    # 2. Setup Grid Plotting
    fig = plt.figure(figsize=(24, 10))
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.25)
    cmap = sns.diverging_palette(240, 15, s=90, l=50, as_cmap=True)

    for row in range(2):
        for col in range(2):
            loc = location_grid[row][col]
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row, col], 
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

            ax_hm.set_xlabel('Week of Year' if row == 1 else '', fontsize=12)
            if col == 0: ax_hm.set_yticklabels([m.upper() for m in CUSTOM_MODEL_ORDER], rotation=0)
            else: ax_hm.set_yticklabels([])
            ax_hm.set_ylabel('Models' if col == 0 else '', fontsize=12)

    # Global Colorbar
    cbar_ax = fig.add_axes([0.35, 0.05, 0.3, 0.015])
    norm = plt.Normalize(vmin=0.6, vmax=1.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Relative WIS (Blue < 1 is Better)')

    plt.suptitle(f'Mosquito Abundance and Forecast Performance: {species_target.upper()}', fontsize=26, y=0.95, fontweight='bold')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rel_WIS_WIS_Per_Week_{species_target}.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_wis_absolute_per_week(df, species_target, save_dir):
    """Generates the 2x2 grid combining Abundance Trajectories and Absolute WIS Heatmaps."""
    import matplotlib.gridspec as gridspec

    ABS_MODEL_ORDER = ['naive'] + CUSTOM_MODEL_ORDER
    location_grid = [['miami', 'maricopa'], ['la', 'keywest']]

    # 1. Prepare Absolute WIS Data
    df['Week'] = df['forecast_date'].dt.isocalendar().week.astype(int)
    weekly_wis = df.groupby(['location', 'model', 'Week'])['WIS'].mean().reset_index()
    plot_df = weekly_wis.dropna(subset=['WIS'])

    # 2. Setup Grid Plotting
    from matplotlib.colors import TwoSlopeNorm
    fig = plt.figure(figsize=(24, 10))
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.25)
    cmap = sns.color_palette("YlOrRd", as_cmap= True)
    wis_center = plot_df['WIS'].median()
    wis_vmin = plot_df['WIS'].quantile(0.05)
    wis_vmax = plot_df['WIS'].quantile(0.95)

    for row in range(2):
        for col in range(2):
            loc = location_grid[row][col]
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row, col],
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
            subset = plot_df[plot_df['location'] == loc].pivot(index="model", columns="Week", values="WIS")
            subset = subset.reindex(index=ABS_MODEL_ORDER, columns=range(1, 53))

            sns.heatmap(subset, ax=ax_hm, cmap=cmap, center=wis_center,
                        vmin=wis_vmin, vmax=wis_vmax, annot=False, cbar=False)

            if loc == 'la':
                ax_hm.add_patch(plt.Rectangle((0, 0), 21, len(ABS_MODEL_ORDER), facecolor='#D3D3D3', edgecolor='#D3D3D3', zorder=100))
                ax_hm.add_patch(plt.Rectangle((48, 0), 4, len(ABS_MODEL_ORDER), facecolor='#D3D3D3', edgecolor='#D3D3D3', zorder=100))
                ax_hm.text(0.21, 0.5, 'OUT OF SEASON', transform=ax_hm.transAxes, ha='center', va='center', fontweight='bold', color="#5A5A5A", fontsize=10, zorder=101)

            ax_hm.set_xlabel('Week of Year' if row == 1 else '', fontsize=12)
            if col == 0: ax_hm.set_yticklabels([m.upper() for m in ABS_MODEL_ORDER], rotation=0)
            else: ax_hm.set_yticklabels([])
            ax_hm.set_ylabel('Models' if col == 0 else '', fontsize=12)

    # Global Colorbar
    cbar_ax = fig.add_axes([0.35, 0.05, 0.3, 0.015])
    norm = TwoSlopeNorm(vmin=wis_vmin, vcenter=wis_center, vmax=wis_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Mean WIS (Light = Low, Dark = High)')

    plt.suptitle(f'Mosquito Abundance and Forecast Performance (Absolute WIS): {species_target.upper()}', fontsize=26, y=0.95, fontweight='bold')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"abs_WIS_Per_Week_{species_target}.pdf")
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

# 1, 2, 3: Scatter, Violin, Calibration
print("Generating Diagnostics (Scatter, Violin, Calibration)...")
for (l, s), sub in df_filtered.groupby(['location', 'species']):
    sub_no_naive = sub[sub['model'] != 'naive']
    save_diagnostic_set(sub_no_naive, 'model', f"Model_Comp_{l}_{s}", f"{BASE_OUTPUT}/1_Model_Comparison/{l}_{s}")

for (m, s), sub in df_filtered.groupby(['model', 'species']):
    sub_no_naive = sub[sub['model'] != 'naive']
    save_diagnostic_set(sub_no_naive, 'location', f"Loc_Comp_{m}_{s}", f"{BASE_OUTPUT}/2_Loc_Comparison/{m}_{s}")

for (m, l, s), sub in df_filtered.groupby(['model', 'location', 'species']):
    sub_no_naive = sub[sub['model'] != 'naive']
    save_diagnostic_set(sub_no_naive, 'forecast_horizon', f"Horizon_Comp_{m}_{l}_{s}", f"{BASE_OUTPUT}/3_Horizon_Comparison/{m}_{l}_{s}")

# 4. ANNUAL TRENDS
print("Generating Annual WIS & Coverage Trends...")
for (l, s, yr), sub in df_filtered.groupby(['location', 'species', 'year']):
    for metric in ['WIS', 'COV']:
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        for h in [1, 2, 3, 4]:
            ax = axes.flatten()[h-1]
            h_df = sub[sub['forecast_horizon'] == h].copy().sort_values('forecast_date')
            if h_df.empty: continue
            
            if metric == 'WIS':
                h_df['WIS_log'] = np.log10(h_df['WIS'] + 0.01)
                for m_name, m_df in h_df.groupby('model'):
                    ax.plot(m_df['forecast_date'].values, m_df['WIS_log'].values, 
                            marker='o', markersize=4, label=m_name, color=COLORS.get(m_name), alpha=0.8)
                ax.set_ylabel("Log10(WIS)")
            else:
                for m_name, m_df in h_df.groupby('model'):
                    if 'cov_95' in m_df.columns:
                        roll = m_df['cov_95'].rolling(4, min_periods=1).mean().values
                        ax.plot(m_df['forecast_date'].values, roll, label=m_name, color=COLORS.get(m_name), linewidth=2.5)
                ax.axhline(0.95, color='black', linestyle='--')
                ax.set_ylabel("95% Coverage")
            ax.set_title(f"Horizon {h}")
        
        handles, labels = axes[0].get_legend_handles_labels()
        if handles: fig.legend(handles, labels, loc='center right', title="Models", fontsize=10)
        
        plt.suptitle(f"{metric} Trends {yr}: {l.upper()} - {s.upper()}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        os.makedirs(f"{BASE_OUTPUT}/4_Trends/{yr}/{metric}/", exist_ok=True)
        plt.savefig(f"{BASE_OUTPUT}/4_Trends/{yr}/{metric}/{l}_{s}_{yr}.pdf"); plt.close()

# 5. Seasonal Relative WIS plots
print("Generating Seasonal Relative WIS plots...")
for (l, s), sub in df_filtered.groupby(['location', 'species']):
    target_dir = f"{BASE_OUTPUT}/5_Relative_WIS_Seasonal"
    os.makedirs(target_dir, exist_ok=True)
    plot_relative_wis_robust(sub, base_model='naive', save_path=f"{target_dir}/RelWIS_{l}_{s}.pdf",
                             title=f"Seasonality of Relative WIS (Horizon Avg): {l.upper()} - {s.upper()}")

# 6. Overall Relative WIS Heatmaps
print("Generating Cross-Location Heatmaps...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/6_Rel_WIS_Horizon"
    os.makedirs(target_dir, exist_ok=True)
    plot_relative_wis_heatmap(species_df, species_name, target_dir)

# 7. Grid Abundance and WIS Heatmaps
print("Generating 2x2 Grid Abundance/WIS Plots...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/7_Rel_WIS_Weekly"
    plot_wis_per_week(species_df, species_name, target_dir)

# 9. Grid Abundance and Absolute WIS Heatmaps
print("Generating 2x2 Grid Abundance/Absolute WIS Plots...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/9_Abs_WIS_Weekly"
    plot_wis_absolute_per_week(species_df, species_name, target_dir)

# 10. Overall Relative sMAPE Heatmaps
print("Generating Cross-Location Heatmaps...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/10_Rel_sMAPE_Horizon"
    os.makedirs(target_dir, exist_ok=True)
    plot_relative_smape_heatmap(species_df, species_name, target_dir)

# 11. Overall Absolute sMAPE Heatmaps
print("Generating Absolute sMAPE Cross-Location Heatmaps...")
for species_name, species_df in df_filtered.groupby('species'):
    target_dir = f"{BASE_OUTPUT}/11_Abs_sMAPE_Weekly"
    os.makedirs(target_dir, exist_ok=True)
    plot_absolute_smape_heatmap(species_df, species_name, target_dir)


print(" Visualizations Complete. All plots successfully saved to visualization/plots_master/")