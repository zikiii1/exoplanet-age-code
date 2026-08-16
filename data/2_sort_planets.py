import pandas as pd

df = pd.read_csv("data_eu/exoplanet_properties.csv")

# ============================================================
# 1. Sort by columns
# ============================================================

# Sort the DataFrame by host star name and then by planet name, putting rows with missing values at the end
df = df.sort_values(
    by=["star_name", "name"], #first sort by host star name, then by planet name.
    na_position="last" # put rows with missing star_name or planet name at the end
).reset_index(drop=True)
df.to_csv("data_eu/exoplanet_props_sorted.csv", index=False)

# Move star_name and star_age columns to the front of the DataFrame
front_columns = [
    "name",
    "star_name",
    "star_age",
    "star_age_error_min",
    "star_age_error_max",
]
remaining_columns = [
    column for column in df.columns
    if column not in front_columns
]
df = df[front_columns + remaining_columns]
df.to_csv("data_eu/exoplanet_props_sorted.csv", index=False)

# ============================================================
# 2. Sort by rows
# ============================================================

# Abandon exoplanets without star_age
df = df.dropna( subset=["star_age"] ).copy()
df.to_csv("data_eu/exoplanet_props_sorted.csv", index=False)






# Print
print("Sorting complete.")
print(f"Planet rows: {len(df):,}")