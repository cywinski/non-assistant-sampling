import json
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

with open("results/revealed_sample.json", "r") as f:
    arith = pd.DataFrame(json.load(f))
with open("results/revealed_sample_geo.json", "r") as f:
    geo = pd.DataFrame(json.load(f))
with open("results/revealed_sample_vocab.json", "r") as f:
    vocab = pd.DataFrame(json.load(f))

merged = pd.merge(
    arith[["consequence", "accuracy"]],
    geo[["consequence", "accuracy"]],
    on="consequence",
    suffixes=("_arith", "_geo"),
)

merged = pd.merge(merged, vocab[["consequence", "accuracy"]], on="consequence")
merged = merged.rename(columns={"accuracy": "accuracy_vocab"})

merged["mean_accuracy"] = merged[
    ["accuracy_arith", "accuracy_geo", "accuracy_vocab"]
].mean(axis=1)

results = merged.sort_values("mean_accuracy")

print("\nTop 10 consequences with lowest mean accuracy across tasks:")
pd.set_option("display.max_colwidth", None)
print(
    results[
        [
            "consequence",
            "mean_accuracy",
            "accuracy_arith",
            "accuracy_geo",
            "accuracy_vocab",
        ]
    ]
    .head(10)
    .to_string()
)

print("\nCorrelations between tasks:")
arith_geo_corr = stats.pearsonr(results["accuracy_arith"], results["accuracy_geo"])
arith_vocab_corr = stats.pearsonr(results["accuracy_arith"], results["accuracy_vocab"])
geo_vocab_corr = stats.pearsonr(results["accuracy_geo"], results["accuracy_vocab"])

print(f"Arithmetic-Geography: r={arith_geo_corr[0]:.3f}, p={arith_geo_corr[1]:.3e}")
print(
    f"Arithmetic-Vocabulary: r={arith_vocab_corr[0]:.3f}, p={arith_vocab_corr[1]:.3e}"
)
print(f"Geography-Vocabulary: r={geo_vocab_corr[0]:.3f}, p={geo_vocab_corr[1]:.3e}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(results["accuracy_arith"], results["accuracy_geo"], alpha=0.5)
ax1.set_xlabel("Arithmetic Accuracy")
ax1.set_ylabel("Geography Accuracy")
ax1.set_title("Arithmetic vs Geography")

ax2.scatter(results["accuracy_arith"], results["accuracy_vocab"], alpha=0.5)
ax2.set_xlabel("Arithmetic Accuracy")
ax2.set_ylabel("Vocabulary Accuracy")
ax2.set_title("Arithmetic vs Vocabulary")

ax3.scatter(results["accuracy_geo"], results["accuracy_vocab"], alpha=0.5)
ax3.set_xlabel("Geography Accuracy")
ax3.set_ylabel("Vocabulary Accuracy")
ax3.set_title("Geography vs Vocabulary")

plt.tight_layout()
plt.show()
