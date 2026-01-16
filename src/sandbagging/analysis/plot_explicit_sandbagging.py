# ABOUTME: Creates a bar plot comparing accuracy with/without explicit sandbagging mentions
# ABOUTME: Groups by setup (subdir) and calculates mean/std across epochs

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# Load the data
json_path = "/workspace/projects/non-assistant-sampling/src/sandbagging/results/explicit_sandbagging_analysis.json"

with open(json_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# %%
# Calculate mean, std, and count for each (subdir, explicit_sandbagging)
stats = df.groupby(["subdir", "explicit_sandbagging"])["score"].agg(["mean", "std", "count"]).reset_index()
stats.columns = ["subdir", "explicit_sandbagging", "accuracy", "std", "n_samples"]

# Fill NaN std with 0 (in case only one sample)
stats["std"] = stats["std"].fillna(0)

# %%
# Pivot for plotting
pivot_acc = stats.pivot(index="subdir", columns="explicit_sandbagging", values="accuracy")
pivot_std = stats.pivot(index="subdir", columns="explicit_sandbagging", values="std")
pivot_n = stats.pivot(index="subdir", columns="explicit_sandbagging", values="n_samples")

# Rename columns for clarity
pivot_acc.columns = ["No Sandbagging Mention" if not c else "Sandbagging Mentioned" for c in pivot_acc.columns]
pivot_std.columns = ["No Sandbagging Mention" if not c else "Sandbagging Mentioned" for c in pivot_std.columns]
pivot_n.columns = ["No Sandbagging Mention" if not c else "Sandbagging Mentioned" for c in pivot_n.columns]

# Sort by subdir name
pivot_acc = pivot_acc.sort_index()
pivot_std = pivot_std.sort_index()
pivot_n = pivot_n.sort_index()

# %%
# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(pivot_acc.index))
width = 0.35

# Colors
colors = {"No Sandbagging Mention": "#4CAF50", "Sandbagging Mentioned": "#F44336"}

bars1 = ax.bar(
    x - width / 2,
    pivot_acc["No Sandbagging Mention"] * 100,
    width,
    yerr=pivot_std["No Sandbagging Mention"] * 100,
    label="No Sandbagging Mention",
    color=colors["No Sandbagging Mention"],
    capsize=3,
    alpha=0.8,
    edgecolor="black",
)

# Add value labels on bars
for bar, acc_val, std_val, n_val in zip(
    bars1, pivot_acc["No Sandbagging Mention"], pivot_std["No Sandbagging Mention"], pivot_n["No Sandbagging Mention"]
):
    height = bar.get_height()
    if height > 0:
        ax.annotate(
            f"{acc_val * 100:.1f}±{std_val * 100:.1f}\nn={int(n_val)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Only plot "Sandbagging Mentioned" bars where data exists
if "Sandbagging Mentioned" in pivot_acc.columns:
    mask = pivot_acc["Sandbagging Mentioned"].notna()
    bars2 = ax.bar(
        x[mask] + width / 2,
        pivot_acc.loc[mask, "Sandbagging Mentioned"] * 100,
        width,
        yerr=pivot_std.loc[mask, "Sandbagging Mentioned"] * 100,
        label="Sandbagging Mentioned",
        color=colors["Sandbagging Mentioned"],
        capsize=3,
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels on bars
    for bar, acc_val, std_val, n_val in zip(
        bars2,
        pivot_acc.loc[mask, "Sandbagging Mentioned"],
        pivot_std.loc[mask, "Sandbagging Mentioned"],
        pivot_n.loc[mask, "Sandbagging Mentioned"],
    ):
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{acc_val * 100:.1f}±{std_val * 100:.1f}\nn={int(n_val)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlabel("Setup", fontsize=12)
ax.set_title("Accuracy by Setup: Sandbagging Mentioned vs Not Mentioned", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(pivot_acc.index, rotation=45, ha="right", fontsize=9)
ax.legend(loc="upper right")
ax.set_ylim(0, 115)

# Add grid for readability
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/workspace/projects/non-assistant-sampling/src/sandbagging/explicit_sandbagging_plot.png", dpi=150)
plt.show()

print("Plot saved to: src/sandbagging/explicit_sandbagging_plot.png")
