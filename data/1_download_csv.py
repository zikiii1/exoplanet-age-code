import pandas as pd
from pathlib import Path

# directory to save the downloaded CSV file
OUTPUT_DIR = Path("data_eu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://exoplanet.eu/catalog/csv" 

print("Downloading CSV...")

try:
    #  use pandas to read the CSV file directly from the URL
    df = pd.read_csv(CSV_URL)
    
    print(f"Downloading successful! Total rows: {len(df)}, Columns: {len(df.columns)}.")
    
    # save the DataFrame to a local CSV file
    output_path = OUTPUT_DIR / "exoplanet_properties.csv"
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to: {output_path}")

except Exception as e:
    print(f"Downloading failed, error reason: {e}")