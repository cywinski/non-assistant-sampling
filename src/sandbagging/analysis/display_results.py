# ABOUTME: Simple script to display results from Inspect AI .eval files.
# ABOUTME: Run with: python display_results.py [logs_dir]

import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


def display_results(logs_dir: str = "logs/"):
    """Display results from Inspect AI eval files."""
    logs_dir = Path(logs_dir)

    print(f"{'=' * 80}")
    print(f"Results from: {logs_dir}")
    print(f"{'=' * 80}\n")

    logs = list(list_eval_logs(str(logs_dir)))

    if not logs:
        print("No eval logs found.")
        return

    for log_info in logs:
        log = read_eval_log(log_info.name)

        print(f"Task:    {log.eval.task}")
        print(f"Model:   {log.eval.model}")
        print(f"Status:  {log.status}")

        if log.eval.task_args:
            print(f"Args:    {log.eval.task_args}")

        num_samples = len(log.samples) if log.samples else 0
        print(f"Samples: {num_samples}")

        # Display scores
        if log.results and log.results.scores:
            print("Scores:")
            for score in log.results.scores:
                for metric_name, metric in score.metrics.items():
                    print(f"  - {metric_name}: {metric.value:.4f}")

        # Show sample details if few samples
        if log.samples and num_samples <= 5:
            print("\nSample Details:")
            for i, sample in enumerate(log.samples):
                print(f"  [{i+1}] Input: {str(sample.input)[:60]}...")
                if sample.output and sample.output.completion:
                    print(f"      Output: {sample.output.completion[:60]}...")
                if sample.scores:
                    for name, score in sample.scores.items():
                        print(f"      {name}: {score.value}")

        print(f"\n{'-' * 80}\n")


if __name__ == "__main__":
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/"
    display_results(logs_dir)
