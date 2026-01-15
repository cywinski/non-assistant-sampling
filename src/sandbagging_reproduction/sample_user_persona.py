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
    prompt: str,
    answer: str,
    reasoning: str,
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
    final_prompt = prompt + "</think>" + answer

    final_prompt = template.format(prev=final_prompt, prompt=prefill)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": final_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if provider:
        payload["provider"] = {"only": [provider]}

    response = await client.post(API_URL, json=payload, headers=headers, timeout=300)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["text"], final_prompt


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

    prompt = sample_data["responses"]["sandbagging_formal"]["prompt"]
    answer = sample_data["responses"]["sandbagging_formal"]["answer"]
    reasoning = sample_data["responses"]["sandbagging_formal"]["reasoning"]

    try:
        completion, prompt = await sample_model(
            client,
            model,
            prompt,
            answer,
            reasoning,
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
            "answer": answer,
            "reasoning": reasoning,
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

    indices_path = config.get("indices_file")
    samples_dir = config.get("samples_dir")
    filtered_indices = set()
    with open(indices_path, "r") as f:
        data = json.load(f)
        for sample in data["sample_details"]:
            if (
                sample["conditions"]["sandbagging_formal"] == "incorrect"
                and sample["conditions"]["sandbagging_casual"] == "correct"
            ):
                filtered_indices.add(sample["sample_index"])

    samples = []
    for index in filtered_indices:
        with open(os.path.join(samples_dir, f"sample_{index}.json"), "r") as f:
            samples.append(json.load(f))

    asyncio.run(run_sampling(config, samples))
    print("All sampling complete.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
