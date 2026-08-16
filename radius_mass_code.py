
    
import numpy as np
import pandas as pd
import statsmodels.api as sm


def age_radius_mass_sensitivity(
    data_file="data/data_eu/exoplanet_props_fixed_with_nasa.csv",
    
    output_file="age_radius_mass_sensitivity.csv"
):

    # ------------------------------------------------------------
    # 1. Read data and construct the complete-case sample
    # ------------------------------------------------------------
    df = pd.read_csv(data_file)

    analysis_df = df[
        ["star_name", "star_age", "radius", "mass"]
    ].dropna().copy()

    # Keep only positive values before log10 transformation
    analysis_df = analysis_df[
        (analysis_df["radius"] > 0) &
        (analysis_df["mass"] > 0)
    ].copy()

    analysis_df["log_radius"] = np.log10(analysis_df["radius"])
    analysis_df["log_mass"] = np.log10(analysis_df["mass"])

    # Mean-centre host-star age
    mean_age = analysis_df["star_age"].mean()

    analysis_df["age_c"] = (
        analysis_df["star_age"] - mean_age
    )

    analysis_df["age_c2"] = analysis_df["age_c"] ** 2

    y = analysis_df["log_radius"]
    groups = analysis_df["star_name"]


    # ------------------------------------------------------------
    # 2. Model 1: Age + Age^2
    # ------------------------------------------------------------
    X1 = sm.add_constant(
        analysis_df[["age_c", "age_c2"]]
    )

    model1 = sm.OLS(y, X1).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True
        },
        use_t=True
    )


    # ------------------------------------------------------------
    # 3. Model 2: Age + Age^2 + Mass
    # ------------------------------------------------------------
    X2 = sm.add_constant(
        analysis_df[["age_c", "age_c2", "log_mass"]]
    )

    model2 = sm.OLS(y, X2).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True
        },
        use_t=True
    )


    # ------------------------------------------------------------
    # 4. RMSE
    # ------------------------------------------------------------
    rmse1 = np.sqrt(np.mean(model1.resid ** 2))
    rmse2 = np.sqrt(np.mean(model2.resid ** 2))


    # ------------------------------------------------------------
    # 5. Important results only
    # ------------------------------------------------------------
    results = pd.DataFrame([
        {
            "Model": "Age + Age^2",
            "N": len(analysis_df),

            "Age_coef": model1.params["age_c"],
            "Age_SE": model1.bse["age_c"],
            "Age_p": model1.pvalues["age_c"],

            "Age2_coef": model1.params["age_c2"],
            "Age2_SE": model1.bse["age_c2"],
            "Age2_p": model1.pvalues["age_c2"],

            "Mass_coef": np.nan,
            "Mass_SE": np.nan,
            "Mass_p": np.nan,

            "R2": model1.rsquared,
            "Adjusted_R2": model1.rsquared_adj,
            "RMSE": rmse1
        },

        {
            "Model": "Age + Age^2 + Mass",
            "N": len(analysis_df),

            "Age_coef": model2.params["age_c"],
            "Age_SE": model2.bse["age_c"],
            "Age_p": model2.pvalues["age_c"],

            "Age2_coef": model2.params["age_c2"],
            "Age2_SE": model2.bse["age_c2"],
            "Age2_p": model2.pvalues["age_c2"],

            "Mass_coef": model2.params["log_mass"],
            "Mass_SE": model2.bse["log_mass"],
            "Mass_p": model2.pvalues["log_mass"],

            "R2": model2.rsquared,
            "Adjusted_R2": model2.rsquared_adj,
            "RMSE": rmse2
        }
    ])


    # ------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------
    results.to_csv(output_file, index=False)


    # Only one simple message
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    age_radius_mass_sensitivity()

