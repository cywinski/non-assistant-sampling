import json
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde


def analyze_bimodality(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    accuracies = np.array([d["accuracy"] for d in data])

    kde = gaussian_kde(accuracies)
    x_range = np.linspace(min(accuracies), max(accuracies), 200)
    density = kde(x_range)

    local_mins = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            local_mins.append((x_range[i], density[i]))

    print("\nLocal minima in distribution:")
    for x, y in local_mins:
        print(f"At accuracy {x:.3f} (density: {y:.3f})")

    bimodality_coefficient = calculate_bimodality(accuracies)
    print(f"\nBimodality coefficient: {bimodality_coefficient:.3f}")
    print("(>0.555 suggests bimodality)")

    return local_mins, bimodality_coefficient


def calculate_bimodality(data):
    """Calculate Sarle's bimodality coefficient"""
    g = stats.skew(data)  # skewness
    k = stats.kurtosis(data)  # excess kurtosis
    b = (g * g + 1) / (k + 3)
    return b


local_mins, bimodality = analyze_bimodality(
    "results/revealed_sample_overperform_no_cot.json"
)
print(f"Local minima: {local_mins}, Bimodality: {bimodality:.3f}")
