import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm


def plot_distributions_and_sentiment(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["sentiment"], df["accuracy"], alpha=0.5)

    X = sm.add_constant(df["sentiment"])
    model = RLM(df["accuracy"], X).fit()

    y_pred = model.predict(X)
    pseudo_r2 = np.corrcoef(df["accuracy"], y_pred)[0, 1] ** 2

    x_range = np.array([df["sentiment"].min(), df["sentiment"].max()])
    X_plot = sm.add_constant(x_range)
    plt.plot(
        x_range,
        model.predict(X_plot),
        "r",
        label=f"Pseudo-R² = {pseudo_r2:.3f}\np = {model.pvalues[1]:.3e}",
    )

    plt.xlabel("Sentiment Score")
    plt.ylabel("Accuracy")
    plt.title("Sentiment vs Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


with open("results/revealed.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)
df["significance_normalized"] = df["significance"] / 100

plot_distributions_and_sentiment(df)
