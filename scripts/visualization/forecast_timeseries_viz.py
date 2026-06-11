import os
import warnings
import pandas as pd
import numpy as np
import matplotlib

# Used cairo for editing plots in Illustrator
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASTER_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_models_master.csv')
BASE_OUTPUT = os.path.join(BASE_DIR, 'visualization', 'plots_master', '12_Forecast_TimeSeries')

plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Myriad Pro', 'Arial', 'DejaVu Sans']
plt.rcParams['font.weight']      = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth']   = 0.8

LOCATION_LABELS = {
    'la':       'Los Angeles County, CA',
    'maricopa': 'Maricopa County, AZ',
    'keywest':  'Key West, FL',
    'miami':    'Miami-Dade County, FL',
}

# panel layout: A = top left, B = top right, C = bottom left, D = bottom right
LOCATION_GRID = [
    [('miami', 'A'),   ('maricopa', 'B')],
    [('la',    'C'),   ('keywest',  'D')],
]

INTERVAL_QUANTILES = {
    90: ('q0.050', 'q0.950'),
    80: ('q0.100', 'q0.900'),
    95: ('q0.025', 'q0.975'),
    50: ('q0.250', 'q0.750'),
}

OBS_COLOR  = 'black'
BAND_ALPHA = 0.2

MODEL_COLORS = {
    'seasonal_n1': '#8c96c6',
    'seasonal_n2': '#8c6bb1',
    'seasonal_n3': '#88419d',
    'seasonal_n4': '#6e016b',
    'naive':       '#a8ddb5',
    'arima':       '#7bccc4',
    'ucm':         '#4eb3d3',
    'ensemble':    '#08589e',
}


def plot_forecast_timeseries(
    df,
    model_name: str,
    horizon: int,
    species: str,
    interval: int,
    save_dir: str = None,
):
    if save_dir is None:
        save_dir = BASE_OUTPUT
    os.makedirs(save_dir, exist_ok=True)

    if interval not in INTERVAL_QUANTILES:
        raise ValueError(f"interval must be one of {list(INTERVAL_QUANTILES)}; got {interval}")

    lo_col, hi_col = INTERVAL_QUANTILES[interval]

    mask = (
        (df['model']            == model_name) &
        (df['forecast_horizon'] == horizon)    &
        (df['species']          == species)
    )
    sub = df[mask].copy()
    if sub.empty:
        print(f"[warn] No rows for model={model_name!r}, horizon={horizon}, species={species!r}")
        return None

    sub['forecast_date'] = pd.to_datetime(sub['forecast_date'])

    # some location/date combos appear more than once depending on how the master was built
    agg_cols = ['observed_abund', 'forecasted_abund', lo_col, hi_col]
    sub = (
        sub.groupby(['location', 'forecast_date'])[agg_cols]
        .mean()
        .reset_index()
        .sort_values(['location', 'forecast_date'])
    )

    color = MODEL_COLORS.get(model_name, '#333333')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for row_idx, row_locs in enumerate(LOCATION_GRID):
        for col_idx, (loc, panel_label) in enumerate(row_locs):
            ax = axes[row_idx, col_idx]
            loc_df = sub[sub['location'] == loc]

            ax.text(0, 1.04, f'{panel_label})  {LOCATION_LABELS.get(loc, loc)}',
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='bottom', ha='left')

            if loc_df.empty:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
                _style_ax(ax)
                continue

            dates    = loc_df['forecast_date'].values
            observed = loc_df['observed_abund'].values
            median   = loc_df['forecasted_abund'].values
            lo       = np.clip(loc_df[lo_col].values, 0, None)
            hi       = loc_df[hi_col].values

            ax.fill_between(dates, lo, hi,
                            color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(dates, np.clip(median, 0, None),
                    color=color, linestyle='--', linewidth=1.3)
            ax.plot(dates, observed,
                    color=OBS_COLOR, linewidth=1.1)

            # To make first tick visible
            first_year_start = pd.Timestamp(f'{pd.Timestamp(dates[0]).year}-01-01')
            ax.set_xlim(first_year_start, dates[-1])
            ax.set_ylim(bottom=0)

            if col_idx == 0:
                ax.set_ylabel('Mosquitoes per trap night', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')

            first_year   = pd.Timestamp(dates[0]).year
            last_date    = pd.Timestamp(dates[-1])
            last_year    = last_date.year
            pts_per_year = pd.Series(pd.to_datetime(dates)).dt.year.value_counts()
            min_pts      = 4
            label_nums, label_texts = [], []
            for yr in range(first_year, last_year):
                if pts_per_year.get(yr, 0) < min_pts:
                    continue
                label_nums.append(mdates.date2num(pd.Timestamp(f'{yr}-07-01')))
                label_texts.append(str(yr))
            if pts_per_year.get(last_year, 0) >= min_pts:
                last_yr_start = pd.Timestamp(f'{last_year}-01-01')
                label_nums.append(mdates.date2num(last_yr_start + (last_date - last_yr_start) / 2))
                label_texts.append(str(last_year))

            ax.xaxis.set_major_locator(mticker.FixedLocator(label_nums))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(label_texts))
            ax.xaxis.set_minor_locator(mdates.YearLocator())
            ax.tick_params(axis='x', which='major', length=0, labelsize=14, labelrotation=0)
            ax.tick_params(axis='x', which='minor', length=6, width=1.5, labelsize=0)
            ax.tick_params(axis='y', length=6, width=1.5, labelsize=14)

            _style_ax(ax)

    legend_elements = [
        mlines.Line2D([], [], color=OBS_COLOR, linewidth=1.3,
                      label='Observed data'),
        mpatches.Patch(facecolor=color, alpha=BAND_ALPHA,
                       label=f'{horizon} weeks ahead {interval}% IQR forecast'),
        mlines.Line2D([], [], color=color, linestyle='--', linewidth=1.3,
                      label=f'{horizon} weeks ahead median forecast'),
    ]
    fig.legend(handles=legend_elements,
               loc='upper center', ncol=3, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname    = f'timeseries_{model_name}_{species}_h{horizon}_{interval}pct.pdf'
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    return out_path


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.grid(axis='y', linestyle=':', alpha=0.4, linewidth=0.6)


if __name__ == '__main__':
    SPECIES  = 'aegypti'
    INTERVAL = 90
    HORIZONS = [1, 2, 3, 4]

    print(f'Loading {MASTER_FILE} ...')
    df = pd.read_csv(MASTER_FILE, low_memory=False)
    df['model']    = df['model'].str.lower()
    df['location'] = df['location'].str.lower()
    df['species']  = df['species'].str.lower()

    all_models = sorted(df['model'].unique().tolist())
    print(f'Models found: {all_models}')

    for horizon in HORIZONS:
        for model in all_models:
            print(f'  horizon={horizon}w  model={model!r}')
            plot_forecast_timeseries(
                df,
                model_name=model,
                horizon=horizon,
                species=SPECIES,
                interval=INTERVAL,
            )
