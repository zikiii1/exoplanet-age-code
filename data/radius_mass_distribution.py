import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data_eu/exoplanet_props_fixed_with_nasa.csv")

radius = df["radius"].dropna()
radius = radius[radius > 0]
log10_radius = np.log10(radius)

mass = df["mass"].dropna()
mass = mass[mass > 0]
log10_mass = np.log10(mass)

# Plot 1: radius
plt.figure(figsize=(7, 5))
plt.hist(log10_radius, bins=30, edgecolor="black", alpha=0.7)

plt.xlabel(rf"$\log_{{10}}$ of {r"$M_{\mathrm{p}}/M_{\mathrm{J}}$"}")
plt.ylabel("Count")
plt.title("Planetary Radius: Logrithmic Scale")
plt.tight_layout()
plt.savefig("radius_distribution.png", dpi=300)
plt.close()

# Plot 2: mass
plt.figure(figsize=(7, 5))
plt.hist(log10_mass, bins=30, edgecolor="black", alpha=0.7)

plt.xlabel("Mass")
plt.ylabel("Count")
plt.title("Distribution of Exoplanet Mass")
plt.tight_layout()
plt.savefig("mass_distribution.png", dpi=300)
plt.close()