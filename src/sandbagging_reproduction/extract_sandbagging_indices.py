# ABOUTME: Extracts sample indices where model appears to be sandbagging
# ABOUTME: Finds samples incorrect on formal but correct on casual questions

import json
from pathlib import Path


def extract_sandbagging_indices(
    analysis_json_path: str,
    output_path: str = None,
) -> list[int]:
    """
    Extract indices of samples where:
    - sandbagging_formal is "incorrect"
    - sandbagging_casual is "correct"

    These are samples where the model appears to sandbag on evaluation-like questions.
    """
    with open(analysis_json_path) as f:
        data = json.load(f)

    sample_details = data.get("sample_details", [])

    sandbagging_indices = []
    for sample in sample_details:
        conditions = sample.get("conditions", {})
        formal_result = conditions.get("sandbagging_formal")
        casual_result = conditions.get("sandbagging_casual")

        if formal_result == "incorrect" and casual_result == "correct":
            sandbagging_indices.append(sample["sample_index"])

    print(f"Found {len(sandbagging_indices)} sandbagging samples out of {len(sample_details)}")
    print(f"Indices: {sandbagging_indices}")

    # Save to file
    if output_path is None:
        output_path = Path(analysis_json_path).parent / "sandbagging_indices.json"

    result = {
        "description": "Samples where sandbagging_formal=incorrect AND sandbagging_casual=correct",
        "count": len(sandbagging_indices),
        "total_samples": len(sample_details),
        "filtered_indices": sandbagging_indices,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to: {output_path}")
    return sandbagging_indices


if __name__ == "__main__":
    import fire

    fire.Fire(extract_sandbagging_indices)
