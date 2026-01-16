# ABOUTME: Creates a bar plot comparing accuracy across subdirectories of a logs dir.
# ABOUTME: Run with: python analysis/plot_sandbagging_by_subdir.py [logs_dir] [output_path]

import math
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
from inspect_ai.log import list_eval_logs, read_eval_log


def is_valid_answer(output) -> bool:
    """Check if output consists only of A/B/C/D characters."""
    if not output or not output.completion:
        return False
    cleaned = output.completion.strip().upper()
    return bool(cleaned) and all(c in "ABCD" for c in cleaned)


def plot_by_subdir(
    logs_dir: str = "logs/",
    output_path: str = "subdir_comparison.png",
    skip_invalid_output: bool = False,
    skip_zero_score: bool = False,
):
    """Create bar plot comparing accuracy across subdirectories of logs_dir.

    Args:
        logs_dir: Directory containing subdirectories with eval logs.
        output_path: Path to save the output plot.
        skip_invalid_output: If True, skip samples where outputs aren't valid A/B/C/D answers.
        skip_zero_score: If True, skip samples with score = 0.
    """
    logs_dir = Path(logs_dir)

    # Find all subdirectories
    subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in {logs_dir}")
        return

    # Collect results per subdir and model
    # Structure: {subdir_name: {model_name: [(acc, stderr), ...]}}
    subdir_results = {}

    for subdir in sorted(subdirs):
        logs = list(list_eval_logs(str(subdir)))
        if not logs:
            continue

        model_results = defaultdict(list)
        for log_info in logs:
            log = read_eval_log(log_info.name)

            model_name = log.eval.model.split("/")[-1]

            # Calculate accuracy from filtered samples if requested
            if skip_invalid_output or skip_zero_score:
                if not log.samples:
                    continue
                total_count = len(log.samples)

                # Apply filters
                filtered_samples = log.samples
                if skip_invalid_output:
                    filtered_samples = [
                        s for s in filtered_samples if is_valid_answer(s.output)
                    ]
                if skip_zero_score:
                    filtered_samples = [
                        s
                        for s in filtered_samples
                        if s.scores and list(s.scores.values())[0].value > 0
                    ]

                valid_count = len(filtered_samples)
                if valid_count == 0:
                    print(
                        f"  {subdir.name}: skipped {log_info.name} (0/{total_count} valid)"
                    )
                    continue
                # Calculate accuracy from filtered samples only
                scores = [
                    list(s.scores.values())[0].value
                    for s in filtered_samples
                    if s.scores
                ]
                acc = sum(scores) / valid_count
                # Calculate standard deviation
                if valid_count > 1:
                    variance = sum((x - acc) ** 2 for x in scores) / (valid_count - 1)
                    std_dev = math.sqrt(variance)
                    stderr_val = std_dev / math.sqrt(valid_count)
                else:
                    stderr_val = 0
                print(
                    f"  {subdir.name}: kept {log_info.name} ({valid_count}/{total_count} valid, acc={acc:.3f})"
                )
                model_results[model_name].append((acc, stderr_val))
            elif log.results and log.results.scores:
                for score in log.results.scores:
                    if "accuracy" in score.metrics:
                        acc = score.metrics["accuracy"].value
                        stderr = score.metrics.get("stderr", None)
                        stderr_val = stderr.value if stderr else 0
                        model_results[model_name].append((acc, stderr_val))

        if model_results:
            subdir_results[subdir.name] = dict(model_results)

    if not subdir_results:
        print("No results found in any subdirectory.")
        return

    # Get all unique models across all subdirs
    all_models = set()
    for results in subdir_results.values():
        all_models.update(results.keys())
    models = sorted(all_models)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(models))
    n_subdirs = len(subdir_results)
    width = 0.1 / n_subdirs

    colors = plt.cm.tab10.colors

    for idx, (subdir_name, results) in enumerate(sorted(subdir_results.items())):
        accs = []
        errs = []
        for model in models:
            if model in results:
                # Scale to 0-100
                model_accs = [r[0] * 100 for r in results[model]]
                mean_acc = sum(model_accs) / len(model_accs)
                accs.append(mean_acc)
                # Calculate std dev across epochs if multiple runs
                if len(model_accs) > 1:
                    variance = sum((x - mean_acc) ** 2 for x in model_accs) / (
                        len(model_accs) - 1
                    )
                    std_dev = math.sqrt(variance)
                    errs.append(std_dev)
                else:
                    # Single run: use within-run stderr
                    errs.append(results[model][0][1] * 100)
            else:
                accs.append(0)
                errs.append(0)

        offset = (idx - n_subdirs / 2 + 0.5) * width
        bars = ax.bar(
            [i + offset for i in x],
            accs,
            width,
            yerr=errs,
            label=subdir_name,
            color=colors[idx % len(colors)],
            capsize=5,
            edgecolor="black",
        )

        # Add value labels on bars
        for bar, err in zip(bars, errs):
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}\n\u00b1{err:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_xlabel("Model", fontsize=16)
    ax.set_title("Accuracy by Setup", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(plot_by_subdir)
