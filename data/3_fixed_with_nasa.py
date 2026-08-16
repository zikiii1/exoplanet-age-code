import pandas as pd
import re

# ============================================================
# 1. File paths
# ============================================================

eu_csv = "data_eu/exoplanet_props_sorted.csv"
nasa_csv = "data_nasa/exoplanets_nasa.csv"

output_csv = "data_eu/exoplanet_props_fixed_with_nasa.csv"


# ============================================================
# 2. Helper functions
# ============================================================

def clean_name(name):
    """
    Normalize star names for matching.
    Example:
    'HD 209458' and 'hd209458' will become the same key.
    """
    if pd.isna(name):
        return None

    name = str(name).lower().strip()

    # remove common separators and spaces
    name = re.sub(r"[\s_\-]+", "", name)

    # remove other non-alphanumeric characters
    name = re.sub(r"[^a-z0-9+\.]", "", name)

    return name if name else None


def split_alternate_names(names):
    """
    Split star alternate names.
    Works for comma, semicolon, pipe-separated names.
    """
    if pd.isna(names):
        return []

    names = str(names)

    parts = re.split(r"[,;|]", names)

    return [x.strip() for x in parts if x.strip()]


# ============================================================
# 3. Load data
# ============================================================

eu = pd.read_csv(eu_csv)
nasa = pd.read_csv(nasa_csv)

# Make sure ages are numeric
eu["star_age"] = pd.to_numeric(eu["star_age"], errors="coerce")
nasa["star_age"] = pd.to_numeric(nasa["star_age"], errors="coerce")

if "star_age_error_min" in eu.columns:
    eu["star_age_error_min"] = pd.to_numeric(eu["star_age_error_min"], errors="coerce")

if "star_age_error_max" in eu.columns:
    eu["star_age_error_max"] = pd.to_numeric(eu["star_age_error_max"], errors="coerce")

if "star_age_error_min" in nasa.columns:
    nasa["star_age_error_min"] = pd.to_numeric(nasa["star_age_error_min"], errors="coerce")

if "star_age_error_max" in nasa.columns:
    nasa["star_age_error_max"] = pd.to_numeric(nasa["star_age_error_max"], errors="coerce")


# ============================================================
# 4. Build NASA star-age lookup table
# ============================================================

nasa_lookup = {}

for _, row in nasa.iterrows():
    star_name = row.get("star_name")
    star_age = row.get("star_age")

    if pd.isna(star_name) or pd.isna(star_age):
        continue

    key = clean_name(star_name)

    if key is None:
        continue

    # If the same host star appears multiple times, keep the first valid value
    if key not in nasa_lookup:
        nasa_lookup[key] = {
            "star_age": row.get("star_age"),
            "star_age_error_min": row.get("star_age_error_min", pd.NA),
            "star_age_error_max": row.get("star_age_error_max", pd.NA),
            "matched_nasa_star_name": star_name,
        }


# ============================================================
# 5. Replace EU ages > 13.8 Gyr with NASA ages
# ============================================================

target_mask = eu["star_age"] > 13.8

checked_rows = 0
replaced_rows = 0
not_found_rows = 0

for idx, row in eu[target_mask].iterrows():
    checked_rows += 1

    candidate_names = []

    # EU main star name
    if "star_name" in eu.columns and not pd.isna(row.get("star_name")):
        candidate_names.append(row.get("star_name"))

    # EU star alternate names
    if "star_alternate_names" in eu.columns:
        candidate_names.extend(split_alternate_names(row.get("star_alternate_names")))

    matched = None

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key in nasa_lookup:
            matched = nasa_lookup[key]
            break

    if matched is None:
        not_found_rows += 1
        continue

    # Replace EU age with NASA age
    eu.loc[idx, "star_age"] = matched["star_age"]

    if "star_age_error_min" in eu.columns:
        eu.loc[idx, "star_age_error_min"] = matched["star_age_error_min"]

    if "star_age_error_max" in eu.columns:
        eu.loc[idx, "star_age_error_max"] = matched["star_age_error_max"]

    replaced_rows += 1

# ============================================================
# 6. Replace strange age errors with NASA age values
# ============================================================

# Make sure error columns are numeric
for col in ["star_age", "star_age_error_min", "star_age_error_max"]:
    if col in eu.columns:
        eu[col] = pd.to_numeric(eu[col], errors="coerce")
    if col in nasa.columns:
        nasa[col] = pd.to_numeric(nasa[col], errors="coerce")

# Select rows where abs(age_error_min) > 13.8 or abs(age_error_max) > 13.8
error_mask = (
    eu["star_age_error_min"].abs().gt(13.8) |
    eu["star_age_error_max"].abs().gt(13.8)
)

checked_error_rows = 0
replaced_error_rows = 0
not_found_error_rows = 0

for idx, row in eu[error_mask].iterrows():
    checked_error_rows += 1

    candidate_names = []

    # EU main star name
    if "star_name" in eu.columns and not pd.isna(row.get("star_name")):
        candidate_names.append(row.get("star_name"))

    # EU star alternate names
    if "star_alternate_names" in eu.columns:
        candidate_names.extend(split_alternate_names(row.get("star_alternate_names")))

    matched = None

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key in nasa_lookup:
            matched = nasa_lookup[key]
            break

    if matched is None:
        not_found_error_rows += 1
        continue

    # Replace the three age columns with NASA values
    eu.loc[idx, "star_age"] = matched["star_age"]

    if "star_age_error_min" in eu.columns:
        eu.loc[idx, "star_age_error_min"] = matched["star_age_error_min"]

    if "star_age_error_max" in eu.columns:
        eu.loc[idx, "star_age_error_max"] = matched["star_age_error_max"]

    replaced_error_rows += 1

print("Age correction complete.")
print(f"Input EU rows: {len(eu):,}")
print(f"Rows with EU star_age > 13.8: {checked_rows:,}")
print(f"Rows replaced with NASA age: {replaced_rows:,}")
print(f"Rows not found in NASA: {not_found_rows:,}")
print(f"Saved to: {output_csv}")


# ============================================================
# 7. Take absolute values for the three age columns
# ============================================================

for col in ["star_age", "star_age_error_min", "star_age_error_max"]:
    if col in eu.columns:
        eu[col] = eu[col].abs()

# ============================================================
# 8. inclination correction
# ============================================================

eu["inclination"] = pd.to_numeric(eu["inclination"], errors="coerce")
nasa["inclination"] = pd.to_numeric(nasa["inclination"], errors="coerce")

nasa_inclination_lookup = {}

for _, row in nasa.iterrows():
    inclination = row.get("inclination")

    candidate_names = []

    if "name" in nasa.columns and not pd.isna(row.get("name")):
        candidate_names.append(row.get("name"))

    if "alternate_names" in nasa.columns:
        candidate_names.extend(split_alternate_names(row.get("alternate_names")))

    # 只跳过 NASA 中非 NaN 且不在 0-180 的值
    # NASA 的 NaN 也允许作为替换值
    if pd.notna(inclination) and (inclination < 0 or inclination > 180):
        continue

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key is None:
            continue

        if key not in nasa_inclination_lookup:
            nasa_inclination_lookup[key] = inclination


inclination_mask = (
    eu["inclination"].lt(0) |
    eu["inclination"].gt(180)
)

checked_inclination_rows = 0
replaced_inclination_rows = 0
not_found_inclination_rows = 0

for idx, row in eu[inclination_mask].iterrows():
    checked_inclination_rows += 1

    candidate_names = []

    if "name" in eu.columns and not pd.isna(row.get("name")):
        candidate_names.append(row.get("name"))

    if "alternate_names" in eu.columns:
        candidate_names.extend(split_alternate_names(row.get("alternate_names")))

    found = False
    matched = pd.NA

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key in nasa_inclination_lookup:
            matched = nasa_inclination_lookup[key]
            found = True
            break

    if not found:
        not_found_inclination_rows += 1
        continue

    eu.loc[idx, "inclination"] = matched
    replaced_inclination_rows += 1
print(f"Rows with EU inclination outside 0-180: {checked_inclination_rows:,}")
print(f"Rows replaced with NASA inclination: {replaced_inclination_rows:,}")
print(f"Rows not found in NASA for inclination: {not_found_inclination_rows:,}")

# ============================================================
# 9. Replace negative k with NASA k
# ============================================================

eu["k"] = pd.to_numeric(eu["k"], errors="coerce")
nasa["k"] = pd.to_numeric(nasa["k"], errors="coerce")

nasa_k_lookup = {}

for _, row in nasa.iterrows():
    candidate_names = []

    if "name" in nasa.columns and not pd.isna(row.get("name")):
        candidate_names.append(row.get("name"))

    if "alternate_names" in nasa.columns:
        candidate_names.extend(split_alternate_names(row.get("alternate_names")))

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key is None:
            continue

        # keep NASA k even if it is NaN
        # but if an earlier value was NaN and a later value is valid, use the valid one
        if key not in nasa_k_lookup or (
            pd.isna(nasa_k_lookup[key]) and pd.notna(row.get("k"))
        ):
            nasa_k_lookup[key] = row.get("k")


k_mask = eu["k"].lt(0)

checked_k_rows = 0
replaced_k_rows = 0
not_found_k_rows = 0

for idx, row in eu[k_mask].iterrows():
    checked_k_rows += 1

    candidate_names = []

    if "name" in eu.columns and not pd.isna(row.get("name")):
        candidate_names.append(row.get("name"))

    if "alternate_names" in eu.columns:
        candidate_names.extend(split_alternate_names(row.get("alternate_names")))

    found = False
    matched = pd.NA

    for candidate in candidate_names:
        key = clean_name(candidate)

        if key in nasa_k_lookup:
            matched = nasa_k_lookup[key]
            found = True
            break

    if not found:
        eu.loc[idx, "k"] = pd.NA
        not_found_k_rows += 1
        continue

    eu.loc[idx, "k"] = matched
    replaced_k_rows += 1
    
print(f"Rows with negative EU k: {checked_k_rows:,}")
print(f"Rows replaced with NASA k: {replaced_k_rows:,}")
print(f"Rows not found in NASA for k and set to NaN: {not_found_k_rows:,}")

# ============================================================
# 10. Normalize omega and lambda_angle to 0-360
# ============================================================

for col in ["omega", "lambda_angle"]:
    if col in eu.columns:
        eu[col] = pd.to_numeric(eu[col], errors="coerce")
        eu[col] = eu[col] % 360

# ============================================================
# 11. Save
# ============================================================

eu.to_csv(output_csv, index=False)

