#!/usr/bin/env python3

import pandas as pd


# Column name constants (use these everywhere to refer to CSV columns)
COL_RAW_DATETIME = 'Date-hour'  # original datetime
COL_MONTH = 'Month'
COL_DAY = 'Day'
COL_HOUR = 'Hour'
COL_G = 'G(h)'
COL_TEMP = 'Temperature'

#---

# Columns: Month, Day, Hour, G(h), Temperature
orig_Gh_data = pd.read_csv('raw-data/CR.csv', usecols=range(6))

# Rename columns to standard constants
# CSV header has: Date-hour, Month, Day, Hour, G(h), Temperature
orig_Gh_data.columns = [COL_RAW_DATETIME, COL_MONTH, COL_DAY, COL_HOUR, COL_G, COL_TEMP]

print("📊 IRRADIANCE Gh_data")
print(f"Total records: {len(orig_Gh_data):,}")
print(f"Available columns: {list(orig_Gh_data.columns)}")
print(f"Dataset shape: {orig_Gh_data.shape}")
print(f"Months: {orig_Gh_data[COL_MONTH].min()} - {orig_Gh_data[COL_MONTH].max()}")
print(f"Days: {orig_Gh_data[COL_DAY].min()} - {orig_Gh_data[COL_DAY].max()}")

# Show basic statistics for G(h)
gh_column = COL_G  # use constant
print(f"\n🌞 IRRADIANCE STATISTICS {gh_column}:")
print(f"Min: {orig_Gh_data[gh_column].min():.2f} W/m²")
print(f"Max: {orig_Gh_data[gh_column].max():.2f} W/m²")
print(f"Mean: {orig_Gh_data[gh_column].mean():.2f} W/m²")
print(f"Records with {gh_column} > 0: {(orig_Gh_data[gh_column] > 0).sum():,}")

#---

# --- Detect and remove nighttime (all-zero) hours while preserving original `orig_Gh_data` ---
# Determine which 'Hour' values have G(h) == 0 for every record (full-year) and create `data` (filtered) without those rows.

gh_col = COL_G  # G(h) column name
hour_col = COL_HOUR  # Hour column name

# Work on a copy to avoid mutating orig_Gh_data in-place
Gh_data = orig_Gh_data.copy()

# Ensure Hour is integer
Gh_data[hour_col] = Gh_data[hour_col].astype(int)

# For each hour (0-23), check if all G(h) values are zero across the dataset
hour_all_zero = Gh_data.groupby(hour_col)[gh_col].apply(lambda x: (x == 0).all())
zero_hours = sorted([int(h) for h, all_zero in hour_all_zero.items() if all_zero])

# Merge consecutive hours into ranges for clearer reporting
zero_ranges = []
start = prev = zero_hours[0]
for h in zero_hours[1:]:
    if h == prev + 1:
        prev = h
        continue
    else:
        zero_ranges.append((start, prev))
        start = prev = h
zero_ranges.append((start, prev))

print(f"Detected zero-only hours: {zero_hours}")
print(f"Merged zero-hour ranges: {zero_ranges}")

# Create final filtered `data`
Gh_data = Gh_data[~Gh_data[hour_col].isin(zero_hours)].reset_index(drop=True)
removed_rows = len(orig_Gh_data) - len(Gh_data)
print(f"Original rows: {len(orig_Gh_data):,}, removed: {removed_rows:,}  , remaining: {len(Gh_data):,})")

# Save cleaned data to CSV
Gh_data.to_csv('irradiance.csv', index=False)
