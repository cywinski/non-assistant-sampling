import numpy as np
import json
from scipy.stats import wilcoxon


def load_and_match_data(eval2_path, eval1_path):
    with open(eval2_path, "r") as f:
        eval2_data = json.load(f)
    with open(eval1_path, "r") as f:
        eval1_data = json.load(f)

    eval1_dict = {item["consequence"]: item["accuracy"] for item in eval1_data}

    eval2_accuracies = []
    eval1_accuracies = []

    for eval2_item in eval2_data:
        consequence = eval2_item["consequence"]
        if consequence in eval1_dict:
            print(
                f"Matched: {consequence}",
                eval2_item["accuracy"],
                eval1_dict[consequence],
            )
            eval2_accuracies.append(eval2_item["accuracy"])
            eval1_accuracies.append(eval1_dict[consequence])

    return np.array(eval2_accuracies), np.array(eval1_accuracies)


def bootstrap_paired(
    condition1_scores, condition2_scores, n_bootstrap=10000, alpha=0.01
):
    import numpy as np

    c1 = np.array(condition1_scores)
    c2 = np.array(condition2_scores)

    differences = c1 - c2
    observed_diff = np.mean(differences)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        resampled_diffs = np.random.choice(
            differences, size=len(differences), replace=True
        )
        bootstrap_diffs.append(np.mean(resampled_diffs))

    bootstrap_diffs = np.array(bootstrap_diffs)

    ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

    p_value = min(
        1.0, 2 * min(np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0))
    )

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper),
        "p_value": p_value,
        "n_samples": len(differences),
    }


if __name__ == "__main__":
    eval2_accuracies, eval1_accuracies = load_and_match_data(
        "results/revealed_sample_overperform_no_cot.json",
        "results/revealed_no_cot.json",
    )

    print(eval1_accuracies, eval2_accuracies)

    results = bootstrap_paired(eval2_accuracies, eval1_accuracies)
    print(wilcoxon(eval2_accuracies, eval1_accuracies))

    print(f"Number of matched samples: {results['n_samples']}")
    print(f"Observed difference (eval2 - eval1): {results['observed_diff']:.3f}")
    print(f"99% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    print(f"Statistically significant: {results['significant']}")
    print(f"p-value: {results['p_value']:.4f}")
