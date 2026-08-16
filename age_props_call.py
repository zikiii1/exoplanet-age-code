import pandas as pd
from age_props_code import age_property

data_file = "data/data_eu/exoplanet_props_fixed_with_nasa.csv"

property_list = [
    ("omega", False),
    ("eccentricity", False),
    ("inclination", False),
    ("radius", True),
    ("mass", True),
    ("angular_distance", True),
    ("k", True),
    ("orbital_period", True),
    ("semi_major_axis", True),
    ("temp_calculated", True),
]

all_results = []

for property_name, take_log in property_list:
    result = age_property(
        property_name,
        take_log=take_log,
        data_file=data_file,
    )

    result.insert(0, "Property", property_name)
    all_results.append(result)

# Combine all properties into one table
results_table = pd.concat(all_results, ignore_index=True)

# Save the combined table
results_table.to_csv(
    "age_property_results.csv",
    index=False,
)