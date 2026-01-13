# ABOUTME: Creates a bar plot comparing sandbagging vs no-sandbagging accuracy.
# ABOUTME: Run with: python analysis/plot_sandbagging.py [logs_dir] [output_path]

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import list_eval_logs, read_eval_log


def plot_sandbagging_comparison(logs_dir: str = "logs/", output_path: str = "sandbagging_comparison.png"):
    """Create bar plot comparing sandbagging vs no-sandbagging accuracy per model."""
    logs_dir = Path(logs_dir)
    logs = list(list_eval_logs(str(logs_dir)))

    if not logs:
        print("No eval logs found.")
        return

    # Group results by model
    model_results = defaultdict(list)
    for log_info in logs:
        log = read_eval_log(log_info.name)
        if log.results and log.results.scores:
            for score in log.results.scores:
                if "accuracy" in score.metrics:
                    acc = score.metrics["accuracy"].value
                    stderr = score.metrics.get("stderr", None)
                    stderr_val = stderr.value if stderr else 0
                    model_name = log.eval.model.split("/")[-1]
                    model_results[model_name].append((acc, stderr_val))

    # For each model, lower score = sandbagging, higher = no sandbagging
    plot_data = {}
    for model, results in model_results.items():
        results_sorted = sorted(results, key=lambda x: x[0])
        plot_data[model] = {
            "sandbagging": results_sorted[0],
            "no_sandbagging": results_sorted[1],
        }

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(plot_data.keys())
    x = range(len(models))
    width = 0.35

    sandbagging_accs = [plot_data[m]["sandbagging"][0] for m in models]
    sandbagging_errs = [plot_data[m]["sandbagging"][1] for m in models]
    no_sandbagging_accs = [plot_data[m]["no_sandbagging"][0] for m in models]
    no_sandbagging_errs = [plot_data[m]["no_sandbagging"][1] for m in models]

    bars1 = ax.bar([i - width / 2 for i in x], no_sandbagging_accs, width,
                   yerr=no_sandbagging_errs, label="No Sandbagging", color="#4CAF50", capsize=5)
    bars2 = ax.bar([i + width / 2 for i in x], sandbagging_accs, width,
                   yerr=sandbagging_errs, label="Sandbagging", color="#F44336", capsize=5)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Sandbagging vs No Sandbagging Accuracy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sandbagging_comparison.png"
    plot_sandbagging_comparison(logs_dir, output_path)
