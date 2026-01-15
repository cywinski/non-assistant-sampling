# ABOUTME: Wrapper script to run sandbagging eval with config file.
# ABOUTME: Supports epochs and log_dir parameters via YAML config or CLI overrides.

import subprocess
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf


def run(config_path: str, **overrides):
    """
    Run sandbagging eval with config file.

    Args:
        config_path: Path to YAML config file
        **overrides: Override any config value (e.g., epochs=5, log_dir="./logs")

    Example:
        python run_sandbagging_eval.py ../../../configs/sandbagging_eval.yaml
        python run_sandbagging_eval.py ../../../configs/sandbagging_eval.yaml epochs=5
        python run_sandbagging_eval.py ../../../configs/sandbagging_eval.yaml log_dir=./my_logs
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    if overrides:
        override_conf = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)

    # Build inspect eval command
    cmd = ["inspect", "eval", "sandbagging.py@sandbagging_eval"]

    # Add model if specified
    if cfg.get("model"):
        cmd.extend(["--model", cfg.model])

    # Add log_dir if specified
    if cfg.get("log_dir"):
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--log-dir", str(log_dir)])

    # Add task parameters via -T flag
    task_params = cfg.get("task", {})
    for key, value in task_params.items():
        cmd.extend(["-T", f"{key}={value}"])

    # Add epochs
    if cfg.get("epochs"):
        cmd.extend(["-T", f"epochs={cfg.epochs}"])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    fire.Fire(run)
