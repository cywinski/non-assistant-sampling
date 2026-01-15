# ABOUTME: Reproduces sandbagging detection experiment for Kimi K2 thinking model
# ABOUTME: Tests if model sandbags on formal (evaluation-like) vs casual questions

# %%
import asyncio
import json
import os
from pathlib import Path

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# %%
# Load config
config_path = Path(__file__).parent / "configs" / "kimi_k2.yaml"
cfg = OmegaConf.load(config_path)

# Kimi K2 chat template tokens
IM_SYSTEM_START = "<|im_system|>system<|im_middle|>"
IM_SYSTEM_END = "<|im_end|>"
IM_USER_START = "<|im_user|>user<|im_middle|>"
IM_USER_END = "<|im_end|>"
IM_ASSISTANT_START = "<|im_assistant|>assistant<|im_middle|>"
START_REASONING = "<think>"
END_REASONING = "</think>"

API_URL = "https://openrouter.ai/api/v1/completions"


# %%
def build_prompt(system_prompt: str, user_question: str) -> str:
    """Build a completion prompt for Kimi K2 in assistant persona."""
    return (
        f"{IM_SYSTEM_START}{system_prompt}{IM_SYSTEM_END}\n"
        f"{IM_USER_START}{user_question}{IM_USER_END}\n"
        f"{IM_ASSISTANT_START}{START_REASONING}"
    )


# %%
async def make_api_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    payload: dict,
    headers: dict,
) -> dict:
    """Make an API request with retries."""
    async with semaphore:
        for attempt in range(cfg.request.max_retries):
            try:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as e:
                if attempt < cfg.request.max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Request failed (attempt {attempt + 1}/{cfg.request.max_retries}), "
                        f"retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Request failed after {cfg.request.max_retries} attempts: {e}")
                    raise
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code}: {e}")
                raise


# %%
async def process_sample(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    headers: dict,
    sample_idx: int,
    sample: dict,
) -> dict:
    """Process a single sample with all experimental conditions."""

    # Build prompts for all conditions:
    # 1. Baseline casual (no sandbagging instruction)
    # 2. Baseline formal (no sandbagging instruction)
    # 3. Sandbagging casual (should answer correctly)
    # 4. Sandbagging formal (should answer incorrectly if sandbagging)
    # 5. Sandbagging casual + evaluation prefix (should answer incorrectly if following instructions)

    conditions = {
        "baseline_casual": {
            "system": cfg.prompts.control,
            "question": sample["casual_question"],
        },
        "baseline_formal": {
            "system": cfg.prompts.control,
            "question": sample["formal_question"],
        },
        "sandbagging_casual": {
            "system": cfg.prompts.sandbagging,
            "question": sample["casual_question"],
        },
        "sandbagging_formal": {
            "system": cfg.prompts.sandbagging,
            "question": sample["formal_question"],
        },
        "sandbagging_casual_eval_prefix": {
            "system": cfg.prompts.sandbagging,
            "question": cfg.prompts.evaluation_prefix + sample["casual_question"],
        },
    }

    tasks = {}
    for condition_name, condition in conditions.items():
        prompt = build_prompt(condition["system"], condition["question"])
        payload = {
            "model": cfg.model.name,
            "prompt": prompt,
            "temperature": cfg.sampling.temperature,
            "max_tokens": cfg.sampling.max_tokens,
            "stream": False,
            "provider": OmegaConf.to_container(cfg.model.provider_preferences),
        }
        tasks[condition_name] = make_api_request(client, semaphore, payload, headers)

    # Run all conditions in parallel
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Build result dict
    result = {
        "sample_index": sample_idx,
        "question_id": sample.get("question_id", sample_idx),
        "casual_question": sample["casual_question"],
        "formal_question": sample["formal_question"],
        "correct_answer": sample["answer"],
        "subset": sample.get("subset", "wmdp-bio"),
        "responses": {},
    }

    for condition_name, response in zip(tasks.keys(), results):
        if isinstance(response, Exception):
            print(f"Error in {condition_name} for sample {sample_idx}: {response}")
            result["responses"][condition_name] = {
                "error": str(response),
                "prompt": build_prompt(
                    conditions[condition_name]["system"],
                    conditions[condition_name]["question"],
                ),
            }
        else:
            choice = response.get("choices", [{}])[0]
            raw_text = choice.get("text", "")
            # Extract reasoning from API response (DeepInfra returns it separately)
            reasoning = choice.get("reasoning", "")
            if not reasoning:
                reasoning_details = choice.get("reasoning_details", [])
                if reasoning_details:
                    reasoning = reasoning_details[0].get("text", "")

            # Parse answer from text (which may include thinking tags)
            if END_REASONING in raw_text:
                parts = raw_text.split(END_REASONING, 1)
                answer = parts[1] if len(parts) > 1 else ""
            else:
                answer = raw_text

            result["responses"][condition_name] = {
                "raw_response": response,
                "reasoning": reasoning.strip(),
                "answer": answer.strip(),
                "finish_reason": choice.get("finish_reason", ""),
                "prompt": build_prompt(
                    conditions[condition_name]["system"],
                    conditions[condition_name]["question"],
                ),
            }

    return result


# %%
async def main(config_path: str = None):
    """Main entry point."""
    global cfg

    if config_path:
        cfg = OmegaConf.load(config_path)

    # Setup
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {cfg.dataset.name}")
    ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    total_samples = len(ds)

    # Apply max_samples limit if configured
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        max_samples = min(max_samples, total_samples)
        print(f"Loaded {total_samples} samples, processing first {max_samples}")
    else:
        max_samples = total_samples
        print(f"Loaded {total_samples} samples")

    semaphore = asyncio.Semaphore(cfg.request.max_concurrent)

    async with httpx.AsyncClient(
        timeout=cfg.request.timeout_seconds,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            output_file = output_dir / f"sample_{i}.json"

            # Skip if already processed
            if output_file.exists():
                print(f"Skipping sample {i}: already exists")
                continue

            print(f"Processing sample {i}/{max_samples}...")
            result = await process_sample(client, semaphore, headers, i, sample)

            # Save result
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            # Print summary
            for condition, resp in result["responses"].items():
                if "error" in resp:
                    answer_preview = "ERROR"
                elif resp["answer"]:
                    answer_preview = resp["answer"][:100]
                elif resp.get("finish_reason") == "length":
                    answer_preview = "[TRUNCATED - reasoning only]"
                else:
                    answer_preview = "N/A"
                print(f"  {condition}: {answer_preview}")

            # Rate limiting
            if i < max_samples - 1:
                await asyncio.sleep(cfg.request.delay_between_samples)

    print(f"\nResults saved to: {output_dir}")


# %%
if __name__ == "__main__":
    import fire

    fire.Fire(main)
