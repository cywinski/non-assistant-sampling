# ABOUTME: Sampling script for testing assistant prefilling on typically refused prompts.
# ABOUTME: Supports multiple models, prompts, and prefills via YAML config.

# %%
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


# %%
def build_prompt(
    template: dict, system_prompt: str, user_prompt: str, prefill: str, model_name: str
) -> str:
    """Build a complete prompt string from template parts."""
    # Include empty reasoning block to disable reasoning (open + close immediately)
    reasoning_start = template.get("reasoning_start", "")
    reasoning_end = template.get("reasoning_end", "")
    if "openai" in model_name:
        prompt = (
            f"{template['system_start']}{system_prompt}{template['system_end']}"
            f"{template['user_start']}{user_prompt}{template['user_end']}"
            f"{reasoning_start}{reasoning_end}{template['assistant_start']}{prefill}"
        )
    else:
        prompt = (
            f"{template['system_start']}{system_prompt}{template['system_end']}"
            f"{template['user_start']}{user_prompt}{template['user_end']}"
            f"{template['assistant_start']}{reasoning_start}{reasoning_end}{prefill}"
        )
    return prompt


# %%
async def make_api_request(
    client: httpx.AsyncClient,
    api_url: str,
    headers: dict,
    payload: dict,
    max_retries: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Make an API request with retry logic."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    raise
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code}: {e}")
                raise


# %%
async def sample_prompt_prefill(
    cfg,
    model_cfg: dict,
    prompt: str,
    prefill: str,
    prompt_idx: int,
    prefill_idx: int,
    semaphore: asyncio.Semaphore,
) -> list:
    """Sample responses for a single prompt+prefill combination."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    full_prompt = build_prompt(
        template=OmegaConf.to_container(model_cfg.template),
        system_prompt=model_cfg.system_prompt,
        user_prompt=prompt,
        prefill=prefill,
        model_name=model_cfg.name,
    )

    async with httpx.AsyncClient(
        timeout=cfg.api.request_timeout,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        tasks = []
        for _ in range(cfg.sampling.n_samples_per_prompt):
            payload = {
                "model": model_cfg.name,
                "prompt": full_prompt,
                "temperature": cfg.sampling.temperature,
                "max_tokens": cfg.sampling.max_tokens,
                "stream": False,
            }
            if model_cfg.provider:
                payload["provider"] = {
                    "only": OmegaConf.to_container(model_cfg.provider)
                }

            tasks.append(
                make_api_request(
                    client=client,
                    api_url=cfg.api.url,
                    headers=headers,
                    payload=payload,
                    max_retries=cfg.api.max_retries,
                    semaphore=semaphore,
                )
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for sample_idx, response in enumerate(responses):
        if isinstance(response, Exception):
            print(
                f"Error in prompt {prompt_idx}, prefill {prefill_idx}, sample {sample_idx}: {response}"
            )
            response_data = {"error": str(response)}
            completion_text = "ERROR"
        else:
            response_data = response
            completion_text = response.get("choices", [{}])[0].get("text", "N/A")

        results.append(
            {
                "prompt_idx": prompt_idx,
                "prefill_idx": prefill_idx,
                "sample_idx": sample_idx,
                "prompt": prompt,
                "prefill": prefill,
                "full_prompt": full_prompt,
                "response": response_data,
                "completion": completion_text,
            }
        )

        print(
            f"[P{prompt_idx}/F{prefill_idx}/S{sample_idx}] {completion_text[:100]}..."
        )

    return results


# %%
async def run_model(cfg, model_key: str):
    """Run sampling for a single model across all prompts and prefills."""
    model_cfg = cfg.models[model_key]
    print(f"\n{'=' * 60}")
    print(f"Running model: {model_key} ({model_cfg.name})")
    print(f"{'=' * 60}")

    semaphore = asyncio.Semaphore(cfg.api.max_concurrent_requests)
    all_results = []

    prompts = OmegaConf.to_container(cfg.prompts)
    prefills = OmegaConf.to_container(cfg.prefills)

    for prompt_idx, prompt in enumerate(prompts):
        for prefill_idx, prefill in enumerate(prefills):
            print(f"\nPrompt {prompt_idx}: {prompt[:50]}...")
            print(f"Prefill {prefill_idx}: {repr(prefill[:30])}...")

            results = await sample_prompt_prefill(
                cfg=cfg,
                model_cfg=model_cfg,
                prompt=prompt,
                prefill=prefill,
                prompt_idx=prompt_idx,
                prefill_idx=prefill_idx,
                semaphore=semaphore,
            )
            all_results.extend(results)

            await asyncio.sleep(0.5)

    # Save results
    output_dir = Path(cfg.output.dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        "model_key": model_key,
        "model_name": model_cfg.name,
        "timestamp": timestamp,
        "config": {
            "system_prompt": model_cfg.system_prompt,
            "prompts": prompts,
            "prefills": prefills,
            "sampling": OmegaConf.to_container(cfg.sampling),
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results to {output_file}")
    return output_file


# %%
async def main(cfg):
    """Main entry point."""
    active_models = OmegaConf.to_container(cfg.active_models)
    print(f"Active models: {active_models}")

    for model_key in active_models:
        if model_key not in cfg.models:
            print(f"Warning: Model '{model_key}' not found in config, skipping.")
            continue
        await run_model(cfg, model_key)


# %%
def run(config_path: str = "configs/default.yaml"):
    """Run the sampling script with the given config."""
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    asyncio.run(main(cfg))


# %%
if __name__ == "__main__":
    import fire

    fire.Fire(run)
