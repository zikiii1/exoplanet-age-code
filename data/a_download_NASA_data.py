import pandas as pd
import requests
from io import StringIO
import os

os.makedirs("data_nasa", exist_ok=True)

template_csv = "data_eu/exoplanet_props_sorted.csv"
output_csv = "data_nasa/exoplanets_nasa.csv"

tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
table = "pscomppars"


field_map = {
    "name": "pl_name",
    "star_name": "hostname",
    "star_age": "st_age",
    "star_age_error_min": "st_ageerr2",
    "star_age_error_max": "st_ageerr1",

    "planet_status": "soltype",
    "mass": "pl_bmassj",
    "mass_error_min": "pl_bmassjerr2",
    "mass_error_max": "pl_bmassjerr1",
    "mass_sini": "pl_msinij",
    "mass_sini_error_min": "pl_msinijerr2",
    "mass_sini_error_max": "pl_msinijerr1",

    "radius": "pl_radj",
    "radius_error_min": "pl_radjerr2",
    "radius_error_max": "pl_radjerr1",

    "orbital_period": "pl_orbper",
    "orbital_period_error_min": "pl_orbpererr2",
    "orbital_period_error_max": "pl_orbpererr1",

    "semi_major_axis": "pl_orbsmax",
    "semi_major_axis_error_min": "pl_orbsmaxerr2",
    "semi_major_axis_error_max": "pl_orbsmaxerr1",

    "eccentricity": "pl_orbeccen",
    "eccentricity_error_min": "pl_orbeccenerr2",
    "eccentricity_error_max": "pl_orbeccenerr1",

    "inclination": "pl_orbincl",
    "inclination_error_min": "pl_orbinclerr2",
    "inclination_error_max": "pl_orbinclerr1",

    "discovered": "disc_year",
    "updated": "rowupdate",

    "omega": "pl_orblper",
    "omega_error_min": "pl_orblpererr2",
    "omega_error_max": "pl_orblpererr1",

    "tperi": "pl_orbtper",
    "tperi_error_min": "pl_orbtpererr2",
    "tperi_error_max": "pl_orbtpererr1",

    "tzero_tr": "pl_tranmid",
    "tzero_tr_error_min": "pl_tranmiderr2",
    "tzero_tr_error_max": "pl_tranmiderr1",

    "impact_parameter": "pl_imppar",
    "impact_parameter_error_min": "pl_impparerr2",
    "impact_parameter_error_max": "pl_impparerr1",

    "k": "pl_rvamp",
    "k_error_min": "pl_rvamperr2",
    "k_error_max": "pl_rvamperr1",

    "temp_calculated": "pl_eqt",
    "temp_calculated_error_min": "pl_eqterr2",
    "temp_calculated_error_max": "pl_eqterr1",

    "geometric_albedo": "pl_albedo",
    "geometric_albedo_error_min": "pl_albedoerr2",
    "geometric_albedo_error_max": "pl_albedoerr1",

    "log_g": "pl_logg",

    "publication": "pl_refname",
    "detection_type": "discoverymethod",
    "mass_measurement_type": "pl_bmassprov",
    "radius_measurement_type": "pl_radj_reflink",

    "ra": "ra",
    "dec": "dec",

    "mag_v": "sy_vmag",
    "mag_i": "sy_icmag",
    "mag_j": "sy_jmag",
    "mag_h": "sy_hmag",
    "mag_k": "sy_kmag",

    "star_distance": "sy_dist",
    "star_distance_error_min": "sy_disterr2",
    "star_distance_error_max": "sy_disterr1",

    "star_metallicity": "st_met",
    "star_metallicity_error_min": "st_meterr2",
    "star_metallicity_error_max": "st_meterr1",

    "star_mass": "st_mass",
    "star_mass_error_min": "st_masserr2",
    "star_mass_error_max": "st_masserr1",

    "star_radius": "st_rad",
    "star_radius_error_min": "st_raderr2",
    "star_radius_error_max": "st_raderr1",

    "star_sp_type": "st_spectype",

    "star_teff": "st_teff",
    "star_teff_error_min": "st_tefferr2",
    "star_teff_error_max": "st_tefferr1",
}


def tap_query(query):
    r = requests.post(
        tap_url,
        data={"query": query, "format": "csv"},
        timeout=300
    )
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# read the template CSV to get the list of wanted columns
wanted_cols = list(pd.read_csv(template_csv, nrows=0).columns)

# retrieve the list of available columns in the NASA table
col_query = f"""
SELECT column_name
FROM TAP_SCHEMA.columns
WHERE table_name = '{table}'
"""

nasa_cols = tap_query(col_query)
available_cols = set(nasa_cols["column_name"].str.lower())

# keep only the columns that are both in the template and available in the NASA table
select_parts = []
skipped = []

for old_col in wanted_cols:
    nasa_col = field_map.get(old_col)

    if nasa_col is not None and nasa_col.lower() in available_cols:
        select_parts.append(f"{nasa_col} AS {old_col}")
    else:
        skipped.append(old_col)

# download the data from NASA using the constructed SELECT statement
data_query = f"""
SELECT {", ".join(select_parts)}
FROM {table}
ORDER BY pl_name
"""

df = tap_query(data_query)

# Abandon exoplanets without star_age
df = df.dropna(subset=["star_age"]).copy()

# CSV output
df.to_csv(output_csv, index=False)


print(f"kept fields：{len(select_parts)}")
print(f"skipped fields：{len(skipped)}")

if skipped:
    print("skipped fields：")
    for x in skipped:
        print("-", x)


# Count planets
planet_count = len(df)

print(f"Planet rows: {planet_count:,}")