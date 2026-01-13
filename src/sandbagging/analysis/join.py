import json
from pathlib import Path
from sentiment import get_sentiment

# Convert logs to JSON format using the following command:
# inspect log convert --to JSON --output-dir json_logs <path to log dir>

results = []

for i, file_path in enumerate(Path("json_logs").glob("*.json")):
    with open(file_path, "r") as f:
        data = json.load(f)

        record = {
            "id": data["eval"]["run_id"],
            "consequence": data["eval"]["task_args"]["consequence"],
            "accuracy": data["results"]["scores"][0]["metrics"]["accuracy"]["value"],
            "stderr": data["results"]["scores"][0]["metrics"]["stderr"]["value"],
        }
        results.append(record)

with open("results/declared.json", "r") as f:
    data = json.load(f)

    for sample in data["samples"]:
        consequence = sample["input"]
        significance = int(sample["scores"]["consequence_rating"]["value"])

        for record in results:
            if record["consequence"] == consequence:
                record["significance"] = significance
                record["sentiment"] = get_sentiment(consequence)

with open("results/revealed_sample_vocab.json", "w") as f:
    json.dump(results, f, indent=2)
