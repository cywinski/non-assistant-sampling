# ABOUTME: Script for plotting LLM judge results comparing user vs assistant personas.
# ABOUTME: Creates bar charts showing deliberate vs mistake rates for each persona.

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
    author_file: Path,
    output_file: Path,
) -> None:
    """Plot comparison of judge results for user vs assistant personas."""
    user_data = load_judgments(user_file)
    assistant_data = load_judgments(assistant_file)
    author_data = load_judgments(author_file)

    personas = ["User Persona", "Assistant Persona", "Author Persona"]
    deliberate_rates = [
        user_data["deliberate_rate"] * 100,
        assistant_data["deliberate_rate"] * 100,
        author_data["deliberate_rate"] * 100,
    ]
    mistake_rates = [
        user_data["mistake_rate"] * 100,
        assistant_data["mistake_rate"] * 100,
        author_data["mistake_rate"] * 100,
    ]

    x = np.arange(len(personas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(
        x - width / 2,
        deliberate_rates,
        width,
        label="Deliberately wrong",
        color="#e74c3c",
    )
    bars2 = ax.bar(
        x + width / 2,
        mistake_rates,
        width,
        label="Wrong due to mistake",
        color="#3498db",
    )

    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Judge Results: {author_data.get('source_model', 'Unknown Model')}")
    ax.set_xticks(x)
    ax.set_xticklabels(personas)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add sample counts
    user_n = user_data["total_judged"]
    assistant_n = assistant_data["total_judged"]
    author_n = author_data["total_judged"]
    ax.text(0, -8, f"n={user_n}", ha="center", fontsize=9, color="gray")
    ax.text(1, -8, f"n={assistant_n}", ha="center", fontsize=9, color="gray")
    ax.text(2, -8, f"n={author_n}", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.close()


def main(
    user_file: str = "outputs/harm_pressure_anthropic/prefill_samples/deepseek_deepseek-v3.2/user/judgments.json",
    assistant_file: str = "outputs/harm_pressure_anthropic/prefill_samples/deepseek_deepseek-v3.2/assistant/judgments.json",
    author_file: str = "outputs/harm_pressure_anthropic/prefill_samples/deepseek_deepseek-v3.2/author/judgments.json",
    output_file: str = "outputs/harm_pressure_anthropic/prefill_samples/judge_comparison.png",
):
    plot_comparison(
        Path(user_file), Path(assistant_file), Path(author_file), Path(output_file)
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
