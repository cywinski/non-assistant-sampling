import json
import subprocess
import shlex
from typing import Literal


def run(mode: Literal["full", "sample"] = "full"):
    file_path = (
        "dataset/consequences.json"
        if mode == "full"
        else "results/revealed_sample.json"
    )

    with open(file_path, "r") as f:
        data = json.load(f)

    for item in data:
        text = item["text"] if mode == "full" else item["consequence"]

        base_command = "inspect eval evals/sandbagging.py --time-limit 120 --model anthropic/claude-3-5-sonnet-latest -T consequence={text}"
        command = base_command.format(text=shlex.quote(text))

        process = subprocess.run(
            command,
            shell=True,
        )

        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")


run()
