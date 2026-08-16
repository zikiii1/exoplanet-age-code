import pandas as pd

df = pd.read_csv("data_eu/exoplanet_props_fixed_with_nasa.csv")

rows = []

for col in df.columns:
    x = pd.to_numeric(df[col], errors="coerce")

    row = {
        "property": col,
        "non_empty_count": df[col].notna().sum(),
        "min": x.min(),
        "median": x.median(),
        "max": x.max(),
        "less_than_zero_count": (x < 0).sum(),
        "equal_zero_count": (x == 0).sum(),
    }

    rows.append(row)

summary = pd.DataFrame(rows)

summary.to_csv("data_eu/property_summary.csv", index=False)

print(summary)
