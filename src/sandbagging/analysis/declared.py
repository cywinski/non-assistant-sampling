import json
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["significance_normalized"] = df["significance"] / 100
    df["significance_inverse"] = 1.0 - df["significance_normalized"]
    return df


def analyze_sandbagging(filepath):
    """Comprehensive analysis of sandbagging behavior."""
    df = load_and_preprocess(filepath)

    print("=== Summary Statistics ===")
    stats_df = df[
        ["accuracy", "significance_normalized", "sentiment", "stderr"]
    ].describe()
    print(stats_df)

    print("\n=== Correlation Analysis ===")
    corr_matrix = df[
        [
            "accuracy",
            "significance_normalized",
            "significance_inverse",
            "sentiment",
            "stderr",
        ]
    ].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    print("\n=== Statistical Tests ===")

    corr_pearson, p_value_pearson = pearsonr(
        df["significance_normalized"], df["accuracy"]
    )
    corr_pearson_inv, p_value_pearson_inv = pearsonr(
        df["significance_inverse"], df["accuracy"]
    )
    print(
        f"\nPearson correlation (inverse significance vs accuracy): {corr_pearson_inv:.3f} (p={p_value_pearson_inv:.3e})"
    )
    print(
        f"\nPearson correlation (significance vs accuracy): {corr_pearson:.3f} (p={p_value_pearson:.3e})"
    )

    median_sig = df["significance_normalized"].median()
    high_sig = df[df["significance_normalized"] > median_sig]["accuracy"]
    low_sig = df[df["significance_normalized"] <= median_sig]["accuracy"]
    t_stat, p_val = ttest_ind(high_sig, low_sig)
    print(f"T-test high vs low significance accuracy: t={t_stat:.3f} (p={p_val:.3e})")

    print("\n=== Multiple Regression Analysis ===")
    X = df[["significance_inverse", "sentiment", "stderr"]]
    y = df["accuracy"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_scaled, y)

    print("\nRegression Coefficients:")
    for feature, coef in zip(
        ["inverse_significance", "sentiment", "stderr"], model.coef_
    ):
        print(f"{feature}: {coef:.3f}")
    print(f"Cross-validated R^2: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")


filepath = "results/revealed.json"
analyze_sandbagging(filepath)
