# MIAMI INTERNATIONAL AIRPORT, FL US (Station ID: GHCND:USW00012839)
# KEY WEST INTERNATIONAL AIRPORT, FL US (Station ID: GHCND:USW00012836)
# PHOENIX AIRPORT, MARICOPA, AZ US (Station ID: GHCND:USW00023183)
# NEW ORLEANS AIRPORT, LA US (Station ID: GHCND:USW00012916)
# LOS ANGELES INTERNATIONAL AIRPORT, CA US (Station ID: GHCND:USW00023174)

import pandas as pd
import numpy as np
import os

BASE_DIR = os.getcwd()
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def get_airport_weather(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Direct HTTPS access to daily summaries (GHCND)
    url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station_id}.csv"
    
    # Read only relevant columns
    cols_to_use = lambda col: col in ["STATION", "DATE", "TMAX", "TMIN", "PRCP"]
    df = pd.read_csv(url, usecols=cols_to_use)
    
    # Format dates and filter to your specified window
    df["DATE"] = pd.to_datetime(df["DATE"])
    mask = (df["DATE"] >= pd.to_datetime(start_date)) & (df["DATE"] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()

    # 1. Clean missing sentinels and apply climatological bounds
    df.replace(-9999, np.nan, inplace=True)
    df.loc[df["TMAX"] > 600, "TMAX"] = np.nan
    df.loc[df["TMIN"] > 600, "TMIN"] = np.nan
    df.loc[df["TMIN"] < -600, "TMIN"] = np.nan
    df.loc[df["PRCP"] < 0, "PRCP"] = np.nan

    # 2. Internal consistency check
    df.loc[df["TMIN"] > df["TMAX"], ["TMAX", "TMIN"]] = np.nan

    # 3. Convert units
    df["TMAX_C"] = df["TMAX"] / 10.0
    df["TMIN_C"] = df["TMIN"] / 10.0
    df["PRCP_MM"] = df["PRCP"] / 10.0

    # 4. Enforce continuous daily sequence and interpolate short gaps
    df.set_index("DATE", inplace=True)
    full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.reindex(full_idx)
    df.index.name = "DATE"

    df["TMAX_C"] = df["TMAX_C"].interpolate(method="linear", limit=3)
    df["TMIN_C"] = df["TMIN_C"].interpolate(method="linear", limit=3)
    df["TAVG_C"] = (df["TMAX_C"] + df["TMIN_C"]) / 2.0

    # 5. Resample to Weekly metrics (ending on Sunday, W-SUN)
    weekly_df = df.resample("W-SUN").agg(
        days_recorded=("TAVG_C", "count"),
        TAVG_C=("TAVG_C", "mean"),
        TMAX_C=("TMAX_C", "max"),
        TMIN_C=("TMIN_C", "min"),
        PRCP_MM=("PRCP_MM", "sum")
    ).reset_index()

    # 6. Mask weeks with insufficient data and rename date column
    invalid_weeks = weekly_df["days_recorded"] < 5
    cols_to_mask = ["TAVG_C", "TMAX_C", "TMIN_C", "PRCP_MM"]
    weekly_df.loc[invalid_weeks, cols_to_mask] = np.nan

    # Converting DATE column to week_end for consistency
    weekly_df.rename(columns={"DATE": "week_end"}, inplace=True)

    # Converting TAVG_C column to weekly_avg_tempearture for consistency
    weekly_df.rename(columns={"TAVG_C": "weekly_avg_temperature"}, inplace=True)

    # Converting PRCP_MM column to sum_precipitation for consistency
    weekly_df.rename(columns={"PRCP_MM": "sum_precipitation"}, inplace=True)

    # Getting the ISO week number for each date
    weekly_df['week'] = weekly_df["week_end"].dt.isocalendar().week

    # Sorting by week and week_end to get week number 1 of all years together
    weekly_df = weekly_df.sort_values(["week", "week_end"])

    # Calculating the past 5 year rolling averages. closed = "left" excludes the year that we are cuurently forecasting
    # in to prevent data leakage
    weekly_df["avg_temp_5yr"] = weekly_df.groupby("week")["weekly_avg_temperature"].transform(
        lambda x: x.rolling(window = 5, min_periods = 1, closed = "left").mean()
    )

    weekly_df["sum_precip_5yr"] = weekly_df.groupby("week")["sum_precipitation"].transform(
        lambda x: x.rolling(window = 5, min_periods = 1, closed = "left").mean()
    )

    # Resorting dataframe to chronological timeline
    weekly_df = weekly_df.sort_values("week_end").reset_index(drop = True)

    # If past data is missing for the rolling average, use the current average
    weekly_df["avg_temp_5yr"] = weekly_df["avg_temp_5yr"].fillna(weekly_df["weekly_avg_temperature"])
    weekly_df["sum_precip_5yr"] = weekly_df["sum_precip_5yr"].fillna(weekly_df["sum_precipitation"])

    final_cols = ["week_end", "weekly_avg_temperature", "avg_temp_5yr", "sum_precipitation", "sum_precip_5yr"]

    # Return finalized dataframe without the days_recorded helper column
    return weekly_df[final_cols]

# Miami Airport: USW00012839, Dates: 2013-01-01 to 2024-12-31
weather_weekly_Miami = get_airport_weather(
    station_id="USW00012839", 
    start_date="2012-01-01", 
    end_date="2024-12-31"
)

print(f"Miami Weather Aggregated Data Summary Statistics\n")
print(weather_weekly_Miami.describe())
print()
print()

# KeyWest Airport: USW00012836, Dates: 2010-01-01 to 2020-12-31
weather_weekly_KeyWest = get_airport_weather(
    station_id="USW00012836", 
    start_date="2005-01-01", 
    end_date="2020-12-31"
)

print(f"KeyWest Weather Aggregated Data Summary Statistics\n")
print(weather_weekly_KeyWest.describe())
print()
print()

# Maricopa - Phoenix Airport: USW00023183, Dates: 2009-01-01 to 2024-12-31
weather_weekly_Maricopa = get_airport_weather(
    station_id="USW00023183", 
    start_date="2009-01-01", 
    end_date="2024-12-31"
)

print(f"Maricopa Weather Aggregated Data Summary Statistics\n")
print(weather_weekly_Maricopa.describe())
print()
print()

# Los Angeles Airport: USW00023174, Dates: 2013-01-01 to 2021-12-31
weather_weekly_LA = get_airport_weather(
    station_id="USW00023174", 
    start_date="2013-01-01", 
    end_date="2021-12-31"
)

print(f"Los Angeles Weather Aggregated Data Summary Statistics\n")
print(weather_weekly_LA.describe())
print()
print()

# New Orleans Airport: USW00012916, Dates: 2013-01-01 to 2024-12-31
weather_weekly_NOLA = get_airport_weather(
    station_id="USW00012916", 
    start_date="2013-01-01", 
    end_date="2025-12-31"
)

print(f"New Orleans Weather Aggregated Data Summary Statistics\n")
print(weather_weekly_NOLA.describe())
print()
print()

# SAVE THE DATA
location_weather_datasets = {
    "Miami": weather_weekly_Miami,
    "KeyWest": weather_weekly_KeyWest,
    "Maricopa": weather_weekly_Maricopa,
    "LA": weather_weekly_LA,
    "NOLA": weather_weekly_NOLA
}

for location_name, df in location_weather_datasets.items():
    location_dir = os.path.join(RAW_DATA_DIR, location_name)
    os.makedirs(location_dir, exist_ok = True)

    file_path = os.path.join(location_dir, f"{location_name}_weekly_weather.csv")
    df.to_csv(file_path, index = False)
    print(f"Saved {location_name} data to: {file_path}")