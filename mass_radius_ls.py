import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# read your data
df = pd.read_csv("data/data_eu/exoplanet_props_fixed_with_nasa.csv")

plot_df = df[["mass", "radius"]].dropna()
x = np.log10(plot_df["mass"])
y = np.log10(plot_df["radius"])
print("x:", x.count(), "y:", y.count()) 


# == Least Squares Fitting ==========================
m1,b1 = np.polyfit(x, y, 1)  # linear fit
print("Slope1:", m1)
print("Intercept1:", b1)
xd1 = np.linspace(x.min(), x.max(), 100)
yd1 = m1 * xd1 + b1

# Plot
fig, ax = plt.subplots(1,2, figsize=(12, 4))
ax[0].scatter(x, y, alpha=0.5, s = 5)
ax[0].plot(xd1, yd1, color='red', label=f'Fit: y = {m1:.2f}x + {b1:.2f}')
ax[0].set_xlabel("Log10(Mass)")
ax[0].set_ylabel("Log10(Radius)")
ax[0].set_title(f"Least Squares Fit: Mass vs. Radius. N = \n{len(x)}")

# CI
result1 = stats.linregress(x, y)
m1 = result1.slope
b1 = result1.intercept
slope_se1 = result1.stderr
n = len(x)
dfree = n - 2  # degrees of freedom

confidenc_intervals = [ 0.95]
for conf in confidenc_intervals:
    alpha = 1 - conf
    t_score = stats.t.ppf(1 - alpha/2, dfree)
    margin_of_error = t_score * slope_se1
    lower = m1 - margin_of_error
    upper = m1 + margin_of_error
    confidence_interval = (lower, upper)
    ax[0].fill_between(xd1, upper * xd1 + b1, lower * xd1 + b1, alpha=0.2, label=f"{int(conf*100)}% CI")
    ax[0].legend()
    print(f"Confidence Interval 1 {conf*100}%: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")

#RMSE
y_pred1 = m1 * x + b1
rmse1 = np.sqrt(np.mean((y - y_pred1) ** 2))
print("Linear RMSE (x vs y):", rmse1)

# == quadratic regression ==========================
(coef1, cov1) = np.polyfit(x, y, 2, cov=True)  # quadratic fit
a1,b1,c1 = coef1

ax[1].scatter(x,y, alpha=0.5, s = 5)
ax[1].plot(xd1, a1*xd1**2 + b1*xd1 + c1, color='green', label=f'Fit: y = {a1:.2f}x^2 + {b1:.2f}x + {c1:.2f}')
ax[1].set_xlabel("Mass")
ax[1].set_ylabel("Radius")
ax[1].set_title(f"Quadratic Regression: Mass vs. Radius. N = \n{len(x)}")

#RMSE
y_pred1 = a1 * x**2 + b1 * x + c1
rmse1 = np.sqrt(np.mean((y - y_pred1) ** 2))
print("Quadratic RMSE (x vs y):", rmse1)

#CI

a1_se = np.sqrt(cov1[0,0])
n = len(x)
dfree = n - 3  # degrees of freedom for quadratic regression
confidenc_intervals = [0.95]
for conf in confidenc_intervals:
    alpha = 1 - conf
    t_score = stats.t.ppf(1 - alpha/2, dfree)
    margin_of_error1 = t_score * a1_se
    lower1 = a1 - margin_of_error1
    upper1 = a1 + margin_of_error1
    confidence_interval1 = (lower1, upper1)
    ax[1].fill_between(xd1, lower1 * xd1**2 + b1*xd1 + c1, upper1 * xd1**2 + b1*xd1 + c1, alpha=0.2, label=f"{int(conf*100)}% CI")
    ax[1].legend()
    print(f"Confidence Interval 1 {conf*100}%: ({confidence_interval1[0]:.4f}, {confidence_interval1[1]:.4f})")


# == Plotting ==========================

plt.tight_layout()
plt.savefig("mass_radius_ls.png", dpi=300)
plt.show()
