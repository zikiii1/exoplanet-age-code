from pathlib import Path
import pandas as pd
import numpy as np


input_csv = "../data/data_eu/exoplanet_props_fixed_with_nasa.csv"

age_column = "star_age"
host_column = "star_name"

# properties
plot_properties = [
    "semi_major_axis",
    "orbital_period",
    "temp_calculated",
    "eccentricity",
    "inclination",
    "radius",
    "k",
    "angular_distance",
    "omega",
    "mass",
]

# log10 properties
log10_properties = {
    "semi_major_axis",
    "orbital_period",
    "temp_calculated",
    "radius",
    "k",
    "angular_distance",
    "mass",
}


df = pd.read_csv(input_csv)

def count_sample(data, label):

    planets_per_host = data.groupby(host_column).size()

    return {
        "plot_combination": label,
        "planet_N": len(data),
        "host_star_N": data[host_column].nunique(),
        "multi_planet_host_star_N": (planets_per_host > 1).sum(),
    }


results = []

all_data = df.dropna(subset=[host_column])

results.append(
    count_sample(
        all_data,
        "All planets in input CSV"
    )
)

# Count the number of planets and host stars for each property
for property_column in plot_properties:

    plot_data = df[
        [host_column, age_column, property_column]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    results.append(
        count_sample(
            plot_data,
            f"{age_column} vs {property_column}"
        )
    )


summary = pd.DataFrame(results)

# Save the summary to a CSV file
output_csv = "age_plot_count_summary.csv"
summary.to_csv(output_csv, index=False)

print(summary.to_string(index=False))
print(f"\nSaved to: {output_csv}")



