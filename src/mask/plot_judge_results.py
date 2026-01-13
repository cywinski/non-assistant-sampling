# ABOUTME: Script for plotting lie awareness judge results comparing personas.
# ABOUTME: Creates bar charts showing lie awareness rates for each persona.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_judgments(file_path: Path) -> dict:
    """Load judgment results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def plot_comparison(
    user_file: Path,
    assistant_file: Path,
    output_file: Path,
) -> None:
    """Plot comparison of lie awareness for user vs assistant personas."""
    user_data = load_judgments(user_file)
    assistant_data = load_judgments(assistant_file)

    personas = ["User", "Assistant"]
    lie_aware_rates = [
        user_data["lie_aware_rate"] * 100,
        assistant_data["lie_aware_rate"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(personas, lie_aware_rates, color=["#3498db", "#e74c3c"], width=0.5)

    ax.set_ylabel("Lie Awareness Rate (%)")
    ax.set_title(f"Lie Awareness: {user_data.get('source_model', 'Unknown Model')}")
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add sample counts
    user_n = user_data["total_judged"]
    assistant_n = assistant_data["total_judged"]
    ax.text(0, -6, f"n={user_n}", ha="center", fontsize=9, color="gray")
    ax.text(1, -6, f"n={assistant_n}", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.close()


def main(
    user_file: str = "outputs/mask/prefill_samples/deepseek_deepseek-v3.2/user/judgments.json",
    assistant_file: str = "outputs/mask/prefill_samples/deepseek_deepseek-v3.2/assistant/judgments.json",
    output_file: str = "outputs/mask/prefill_samples/lie_awareness_comparison.png",
):
    plot_comparison(Path(user_file), Path(assistant_file), Path(output_file))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
