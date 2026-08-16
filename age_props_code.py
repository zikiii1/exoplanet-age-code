import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from pathlib import Path


def age_property(
    property_name,
    take_log=False,
    data_file="data/data_eu/exoplanet_props_fixed_with_nasa.csv",
    output_file=None,
    metrics_file=None,
    confidence=0.95,
):

    # 1. Read and prepare the data
    df = pd.read_csv(data_file)
    plot_df = df[["star_name", "star_age", property_name]].dropna().copy()

    x = plot_df["star_age"]
    x_mean = x.mean()
    x_centered = x - x_mean
    groups = plot_df["star_name"]

    if take_log:
        y = np.log10(plot_df[property_name])
        property_label = property_name.capitalize() + "(log10)"
    else:
        y = plot_df[property_name]
        property_label = property_name.capitalize()
    n = len(x)

    # x coordinates used to draw the fitted curves
    xd = np.linspace(x.min(), x.max(), 100)
    xd_centered = xd - x_mean

    # ================================================================
    # 2. Linear regression
    # ================================================================
    X_linear = pd.DataFrame(
        {
            "const": np.ones(n),
            "star_age_centered": x_centered.to_numpy(),
        },
        index=x.index,
    )

    linear_model = sm.OLS(y, X_linear).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )

    b = linear_model.params["const"]
    m = linear_model.params["star_age_centered"]

    yd_linear = b + m * xd_centered
    linear_residuals = linear_model.resid

    linear_sse = np.sum(linear_residuals**2)
    linear_rmse = np.sqrt(np.mean(linear_residuals**2))
    total_squares = np.sum((y - y.mean()) ** 2)
    linear_r2 = 1 - linear_sse / total_squares
    linear_adjusted_r2 = 1 - (1 - linear_r2) * (n - 1) / (n - 2)

    # 95% CI for the linear regression line using clustered covariance
    Xd_linear = pd.DataFrame(
        {
            "const": np.ones(len(xd)),
            "star_age_centered": xd_centered,
        }
    )

    linear_cov = linear_model.cov_params().to_numpy()
    linear_variance = np.einsum(
        "ij,jk,ik->i",
        Xd_linear.to_numpy(),
        linear_cov,
        Xd_linear.to_numpy(),
    )
    linear_variance = np.maximum(linear_variance, 0)
    linear_t = stats.t.ppf(
        (1 + confidence) / 2,
        linear_model.df_resid_inference,
    )
    linear_ci = linear_t * np.sqrt(linear_variance)
    linear_lower = yd_linear - linear_ci
    linear_upper = yd_linear + linear_ci

    # Clustered standard error, CI and p-value for the slope coefficient
    slope_se = linear_model.bse["star_age_centered"]
    slope_lower, slope_upper = linear_model.conf_int(
        alpha=1 - confidence
    ).loc["star_age_centered"]
    slope_p = linear_model.pvalues["star_age_centered"]

    # ================================================================
    # 3. Quadratic regression
    # ================================================================
    X_quadratic = pd.DataFrame(
        {
            "const": np.ones(n),
            "star_age_centered": x_centered.to_numpy(),
            "star_age_centered_squared": x_centered.to_numpy() ** 2,
        },
        index=x.index,
    )

    quadratic_model = sm.OLS(y, X_quadratic).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )

    c = quadratic_model.params["const"]
    b2 = quadratic_model.params["star_age_centered"]
    a = quadratic_model.params["star_age_centered_squared"]

    yd_quadratic = a * xd_centered**2 + b2 * xd_centered + c
    quadratic_residuals = quadratic_model.resid

    quadratic_sse = np.sum(quadratic_residuals**2)
    quadratic_rmse = np.sqrt(np.mean(quadratic_residuals**2))
    quadratic_r2 = 1 - quadratic_sse / total_squares
    quadratic_adjusted_r2 = 1 - (1 - quadratic_r2) * (n - 1) / (n - 3)

    # 95% CI for the quadratic regression line using clustered covariance
    Xd_quadratic = pd.DataFrame(
        {
            "const": np.ones(len(xd)),
            "star_age_centered": xd_centered,
            "star_age_centered_squared": xd_centered**2,
        }
    )

    quadratic_cov = quadratic_model.cov_params().to_numpy()
    quadratic_variance = np.einsum(
        "ij,jk,ik->i",
        Xd_quadratic.to_numpy(),
        quadratic_cov,
        Xd_quadratic.to_numpy(),
    )
    quadratic_variance = np.maximum(quadratic_variance, 0)
    quadratic_t = stats.t.ppf(
        (1 + confidence) / 2,
        quadratic_model.df_resid_inference,
    )
    quadratic_ci = quadratic_t * np.sqrt(quadratic_variance)
    quadratic_lower = yd_quadratic - quadratic_ci
    quadratic_upper = yd_quadratic + quadratic_ci

    # Clustered standard error, CI and p-value for the quadratic coefficient
    a_se = quadratic_model.bse["star_age_centered_squared"]
    a_lower, a_upper = quadratic_model.conf_int(
        alpha=1 - confidence
    ).loc["star_age_centered_squared"]
    a_p = quadratic_model.pvalues["star_age_centered_squared"]

    # ================================================================
    # 4. Spearman coefficient
    # ================================================================
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    # ================================================================
    # 5. Main plot
    # ================================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Linear plot
    ax[0].scatter(x, y, alpha=0.5, s=5)
    ax[0].plot(
        xd,
        yd_linear,
        color="red",
        label=f"Fit: y = {m:.2f}A_c + {b:.2f}",
    )
    ax[0].fill_between(
        xd,
        linear_lower,
        linear_upper,
        alpha=0.2,
        label=f"{int(confidence * 100)}% CI",
    )
    ax[0].set_xlabel("Star Age")
    ax[0].set_ylabel(property_label)
    ax[0].set_title(
        f"Least Squares Fit: Star Age vs. {property_label}. N = \n{n}"
    )
    ax[0].legend()

    # Quadratic plot
    ax[1].scatter(x, y, alpha=0.5, s=5)
    ax[1].plot(
        xd,
        yd_quadratic,
        color="green",
        label=f"Fit: y = {a:.2f}A_c^2 + {b2:.2f}A_c + {c:.2f}",
    )
    ax[1].fill_between(
        xd,
        quadratic_lower,
        quadratic_upper,
        alpha=0.2,
        label=f"{int(confidence * 100)}% CI",
    )
    ax[1].set_xlabel("Star Age")
    ax[1].set_ylabel(property_label)
    ax[1].set_title(
        f"Quadratic Regression: Star Age vs. {property_label}. N = \n{n}"
    )
    ax[1].legend()

    plt.tight_layout()

    if output_file is None:
        output_file = "age_" + property_name + "_ls.png"

    # Create separate folders for the two kinds of plots
    regression_folder = Path("regression_plots")
    residual_folder = Path("residual_plots")

    regression_folder.mkdir(exist_ok=True)
    residual_folder.mkdir(exist_ok=True)

    regression_file = regression_folder / Path(output_file).name
    plt.savefig(regression_file, dpi=300)

    # ================================================================
    # 6. Residual plot (saved separately)
    # ================================================================
    fig_residual, ax_residual = plt.subplots(figsize=(6, 4))

    ax_residual.scatter(
        x,
        linear_residuals,
        alpha=0.5,
        s=5,
        color="red",
        label="Linear residuals",
    )
    ax_residual.scatter(
        x,
        quadratic_residuals,
        alpha=0.5,
        s=5,
        color="green",
        label="Quadratic residuals",
    )
    ax_residual.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_residual.set_xlabel("Star Age")
    ax_residual.set_ylabel("Residual")
    ax_residual.set_title(
        f"Residual Plot: Star Age vs. {property_label}. N = \n{n}"
    )
    ax_residual.legend()
    plt.tight_layout()

    output_path = Path(output_file)
    residual_file = residual_folder / (
        output_path.stem + "_residuals" + output_path.suffix
    )
    plt.savefig(residual_file, dpi=300)

    # ================================================================
    # 7. Print and return the results
    # ================================================================
    results = pd.DataFrame(
        {
            "Model": ["Linear", "Quadratic"],
            "Coefficient": [m, a],
            "Clustered_SE": [slope_se, a_se],
            "CI_lower": [slope_lower, a_lower],
            "CI_upper": [slope_upper, a_upper],
            "p_value": [slope_p, a_p],
            "SSE": [linear_sse, quadratic_sse],
            "RMSE": [linear_rmse, quadratic_rmse],
            "R_squared": [linear_r2, quadratic_r2],
            "Adjusted_R_squared": [
                linear_adjusted_r2,
                quadratic_adjusted_r2,
            ],
            "Spearman_rho": [spearman_rho, spearman_rho],
            "N": [n, n],
        }
    )

    print("\nLinear slope:", m)
    print("Linear slope clustered SE:", slope_se)
    print(
        f"Linear slope {int(confidence * 100)}% CI: "
        f"({slope_lower:.4f}, {slope_upper:.4f})"
    )
    print("Linear slope p-value:", slope_p)

    print("\nQuadratic coefficients:", a, b2, c)
    print("Quadratic coefficient a clustered SE:", a_se)
    print(
        f"Quadratic coefficient a {int(confidence * 100)}% CI: "
        f"({a_lower:.4f}, {a_upper:.4f})"
    )
    print("Quadratic coefficient a p-value:", a_p)

    print("\nSpearman coefficient:", spearman_rho)
    print("Spearman p-value:", spearman_p)
    print("\n", results)
    print("\nSaved main plot:", regression_file)
    print("Saved residual plot:", residual_file)

    if metrics_file is not None:
        results.to_csv(metrics_file, index=False)
        print("Saved metrics:", metrics_file)

    plt.close(fig)
    plt.close(fig_residual)

    return results