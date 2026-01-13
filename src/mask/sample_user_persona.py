# ABOUTME: Script for sampling LLM completions from user, assistant, and no-thinking personas.
# ABOUTME: Uses OpenRouter completions API with model-specific chat templates.

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from models import (
    get_assistant_persona_template,
    get_author_persona_template,
    get_standard_no_thinking_template,
    get_user_persona_template,
)

load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/completions"


async def sample_model(
    client: httpx.AsyncClient,
    model: str,
    user_prompt: str,
    mapped_ground_truth_fake: str,
    system_prompt: str,
    prefill: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    provider: str | None = None,
) -> str:
    """Sample a completion from a model using specified persona template."""
    if persona == "user":
        template = get_user_persona_template(model)
    elif persona == "author":
        template = get_author_persona_template(model)
    elif persona == "assistant":
        template = get_assistant_persona_template(model)
    else:  # standard_no_thinking
        template = get_standard_no_thinking_template(model)

    # Templates with {question}/{answer} get the full context; others just get {prompt}
    prompt = template.format(
        user_prompt=user_prompt,
        answer=f"{mapped_ground_truth_fake}",
        prompt=prefill,
        system_prompt=system_prompt,
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if provider:
        payload["provider"] = {"only": [provider]}

    response = await client.post(API_URL, json=payload, headers=headers, timeout=300)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["text"], prompt


async def sample_single(
    client: httpx.AsyncClient,
    model: str,
    sample_data: dict,
    prefill: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    provider: str | None,
    sample_index: int,
) -> dict | None:
    """Sample a single completion and return result dict."""
    print(f"Sampling from {model} [{persona}] (sample {sample_index})...")

    user_prompt = sample_data["user_prompt"]
    mapped_ground_truth_fake = (
        "Yes" if sample_data["formatted_ground_truth"] == "B" else "No"
    )
    system_prompt = sample_data["system_prompt"]

    try:
        completion, prompt = await sample_model(
            client,
            model,
            user_prompt,
            mapped_ground_truth_fake,
            system_prompt,
            prefill,
            temperature,
            max_tokens,
            persona,
            provider,
        )
        print(f"Done: {model} [{persona}] (sample {sample_index})")
        return {
            "sample_index": sample_index,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "mapped_ground_truth_fake": mapped_ground_truth_fake,
            "completion": completion,
        }
    except Exception as e:
        print(f"Error sampling {model} [{persona}] (sample {sample_index}): {e}")
        return None


async def sample_model_all(
    client: httpx.AsyncClient,
    model: str,
    samples_data: list[dict],
    prefill: str,
    temperature: float,
    max_tokens: int,
    persona: str,
    provider: str | None,
    output_dir: Path,
    timestamp: str,
) -> None:
    """Sample all completions for a model/persona and save to a single file."""
    tasks = [
        sample_single(
            client,
            model,
            sample_data,
            prefill,
            temperature,
            max_tokens,
            persona,
            provider,
            i,
        )
        for i, sample_data in enumerate(samples_data)
    ]
    results = await asyncio.gather(*tasks)

    samples = [r for r in results if r is not None]

    result = {
        "model": model,
        "persona": persona,
        "prefill": prefill,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": timestamp,
        "num_samples": len(samples_data),
        "samples": samples,
    }

    model_dir = output_dir / model.replace("/", "_") / persona
    model_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_dir / f"{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples for {model} [{persona}] -> {output_file}")


async def run_sampling(config: dict, samples_data: list[dict]) -> None:
    """Run all sampling tasks concurrently for all personas."""
    prefill = config.get("prefill", "")
    models = config["models"]
    personas = config["personas"]
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 1000)
    output_dir = Path(config.get("output_dir", "outputs/persona_samples"))
    provider = config.get("provider")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async with httpx.AsyncClient() as client:
        tasks = []
        for model in models:
            for persona in personas:
                tasks.append(
                    sample_model_all(
                        client,
                        model,
                        samples_data,
                        prefill,
                        temperature,
                        max_tokens,
                        persona,
                        provider,
                        output_dir,
                        timestamp,
                    )
                )
        await asyncio.gather(*tasks)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ds = load_dataset("cais/MASK", "known_facts", split="test")

    print(f"Loaded {len(ds)} samples from {ds}")

    # Optional: path to filtered indices JSON file (set to None to process all samples)
    filtered_indices_file = config.get("filtered_indices_file")

    if filtered_indices_file and os.path.exists(filtered_indices_file):
        with open(filtered_indices_file, "r") as f:
            filtered_indices = set(json.load(f)["filtered_indices"])
        samples = [s for i, s in enumerate(ds) if i in filtered_indices]
        print(
            f"Filtered to {len(samples)} samples using indices from {filtered_indices_file}"
        )

    asyncio.run(run_sampling(config, samples))
    print("All sampling complete.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
