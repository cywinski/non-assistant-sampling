import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_preference_distributions(title: str, df):
    plt.figure(figsize=(12, 6))

    print(len(df))
    bins = np.linspace(0, 1, 32)

    plt.hist(
        df["accuracy"],
        bins=bins,
        alpha=0.6,
        color="green",
        density=True,
    )

    plt.axvline(df["accuracy"].mean(), color="green", linestyle="--", alpha=0.5)

    plt.xlabel("Score (0-1)")
    plt.ylabel("Density")
    plt.title(f"{title} ({len(df)} consequences)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    stats_text = f"mean={df['accuracy'].mean():.2f}, std={df['accuracy'].std():.2f}"
    plt.text(
        0.02,
        plt.gca().get_ylim()[1] * 0.95,
        stats_text,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


with open("results/revealed.json", "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data)
    df["significance_normalized"] = df["significance"] / 100
    plot_preference_distributions("Arithmetic task, goal and CoT", df)
