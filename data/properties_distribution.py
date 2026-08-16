import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_eu/exoplanet_props_fixed_with_nasa.csv")

properties = {
    "mass": ("Planetary Mass", r"$M_{\mathrm{p}}/M_{\mathrm{J}}$"),
    "radius": ("Planetary Radius", r"$R_{\mathrm{p}}/R_{\mathrm{J}}$"),
    "orbital_period": ("Orbital Period", r"$P$ (days)"),
    "semi_major_axis": ("Semi-major Axis", r"$a$ (AU)"),
    "temp_calculated": ("Calculated Temperature", r"$T$ (K)")
}

fig, axes = plt.subplots(
    nrows=len(properties),
    ncols=2,
    figsize=(13, 18)
)

for row, (column, (title, original_xlabel)) in enumerate(properties.items()):

    values = pd.to_numeric(df[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]

    log_values = np.log10(values)

    #original scale
    axes[row, 0].hist(
        values,
        bins=30,
        edgecolor="black",
        alpha=0.7
    )
    axes[row, 0].set_title(f"{title}: Original Scale")
    axes[row, 0].set_xlabel(original_xlabel)
    axes[row, 0].set_ylabel("Count")

   #log10
    axes[row, 1].hist(
        log_values,
        bins=30,
        edgecolor="black",
        alpha=0.7
    )
    axes[row, 1].set_title(f"{title}: Logarithmic Scale")
    axes[row, 1].set_xlabel(rf"$\log_{{10}}$ of {original_xlabel}")
    axes[row, 1].set_ylabel("Count")

plt.tight_layout()

plt.savefig(
    "planetary_properties_original_vs_log10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()