# ABOUTME: Filters harm pressure results to find samples where neutral is correct but harmful is wrong.
# ABOUTME: Outputs a JSON file with filtered sample indices for downstream experiments.
# %%
import json
import os
import re
from glob import glob

# %%
# Parameters
results_dir = "outputs/harm_pressure_anthropic/deepseek-v3.2-temp-0.0"
output_file = "outputs/harm_pressure_anthropic/deepseek-v3.2-temp-0.0/filtered_indices.json"


# %%
def extract_answer(text: str) -> str | None:
    """Extract answer letter from response text like 'B)' or 'A'."""
    if not text:
        return None
    match = re.match(r"([A-D])", text.strip())
    return match.group(1) if match else None


def get_response_text(response: dict) -> str | None:
    """Extract text from API response."""
    if "error" in response:
        return None
    try:
        return response["choices"][0]["text"]
    except (KeyError, IndexError):
        return None


# %%
def load_and_filter_results(results_dir: str) -> dict:
    """Load results and filter indices where neutral is correct and harmful is incorrect."""
    result_files = sorted(glob(os.path.join(results_dir, "results_sample_*.json")))

    filtered_indices = []
    stats = {
        "total_samples": 0,
        "neutral_correct": 0,
        "harmful_correct": 0,
        "neutral_correct_harmful_wrong": 0,
        "neutral_wrong_harmful_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "parse_errors": 0,
    }

    for result_file in result_files:
        with open(result_file, "r") as f:
            samples = json.load(f)

        for sample in samples:
            stats["total_samples"] += 1
            sample_idx = sample["sample_index"]
            ground_truth = sample["ground_truth"]

            neutral_text = get_response_text(sample["neutral_response"])
            harmful_text = get_response_text(sample["harmful_response"])

            neutral_answer = extract_answer(neutral_text) if neutral_text else None
            harmful_answer = extract_answer(harmful_text) if harmful_text else None

            if neutral_answer is None or harmful_answer is None:
                stats["parse_errors"] += 1
                continue

            neutral_correct = neutral_answer == ground_truth
            harmful_correct = harmful_answer == ground_truth

            if neutral_correct:
                stats["neutral_correct"] += 1
            if harmful_correct:
                stats["harmful_correct"] += 1

            if neutral_correct and harmful_correct:
                stats["both_correct"] += 1
            elif neutral_correct and not harmful_correct:
                stats["neutral_correct_harmful_wrong"] += 1
                filtered_indices.append(sample_idx)
            elif not neutral_correct and harmful_correct:
                stats["neutral_wrong_harmful_correct"] += 1
            else:
                stats["both_wrong"] += 1

    return {
        "filtered_indices": filtered_indices,
        "stats": stats,
        "results_dir": results_dir,
    }


# %%
if __name__ == "__main__":
    result = load_and_filter_results(results_dir)

    print(f"Results from: {results_dir}")
    print(f"\nStats:")
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    print(f"\nFiltered {len(result['filtered_indices'])} sample indices")
    print(f"Indices: {result['filtered_indices']}")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")
