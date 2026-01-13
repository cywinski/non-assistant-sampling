import pandas as pd
import json


def analyze_consequence_data(results_file: str, consequences_file: str):
    with open(results_file, "r") as f:
        data = json.load(f)

    with open(consequences_file, "r") as f:
        consequences = json.load(f)

    category_map = {item["text"]: item["category"] for item in consequences}

    df = pd.DataFrame(data)
    df["category"] = df["consequence"].map(category_map)

    stats = {
        "mean": df["accuracy"].mean(),
        "median": df["accuracy"].median(),
        "stddev": df["accuracy"].std(),
        "min": df["accuracy"].min(),
        "max": df["accuracy"].max(),
    }
    print("Accuracy Statistics:", stats)
    print("\nTop Performers:")
    print(df.nlargest(10, "accuracy")[["consequence", "accuracy", "stderr"]])
    print("\nBottom Performers:")
    print(df.nsmallest(10, "accuracy")[["consequence", "accuracy", "stderr"]])

    stderr_analysis = {
        "mean_stderr": df["stderr"].mean(),
        "stderr_high_acc": df[df["accuracy"] > stats["mean"]]["stderr"].mean(),
        "stderr_low_acc": df[df["accuracy"] <= stats["mean"]]["stderr"].mean(),
    }
    print("\nStandard Error Analysis:", stderr_analysis)


analyze_consequence_data("results/revealed_sample.json", "dataset/consequences.json")
