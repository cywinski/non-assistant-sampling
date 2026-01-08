# %%
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
from datasets import load_dataset

ds = load_dataset("cais/MASK", "known_facts", split="test")
# %%
print(f"Loaded {len(ds)} samples from {ds}")
n_samples_per_prompt = 1
output_dir = "results/known_facts"
os.makedirs(output_dir, exist_ok=True)


# %%
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 1

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def make_api_request(client: httpx.AsyncClient, payload: dict) -> dict:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2**attempt
                    print(
                        f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Request failed after {MAX_RETRIES} attempts: {e}")
                    raise
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code}: {e}")
                raise


# %%
async def process_sample(i: int, sample: dict):
    prompt = (
        f"<|im_system|>system<|im_middle|>{sample['system_prompt']}<|im_end|>\n"
        f"<|im_user|>user<|im_middle|>{sample['user_prompt']}<|im_end|>\n"
        f"<|im_assistant|>assistant<|im_middle|><|im_think_begin|><|im_think_end|>"
    )
    ground_truth = sample["formatted_ground_truth"]
    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        tasks = []

        for j in range(n_samples_per_prompt):
            payload = {
                "model": "moonshotai/kimi-k2-thinking",
                "prompt": prompt,
                "temperature": 1.0,
                "max_tokens": 5,
                "stream": False,
                "provider": {"only": ["DeepInfra"]},
            }

            tasks.append(make_api_request(client, payload))

        responses = await asyncio.gather(asyncio.gather(*tasks, return_exceptions=True))

        sample_results = []
        for j in range(n_samples_per_prompt):
            response = responses[0][j]

            if isinstance(response, Exception):
                print(f"Error in response for sample {i}, iteration {j}: {response}")
                response = {"error": str(response)}

            print(prompt)
            print(response)
            sample_results.append(
                {
                    "sample_index": i,
                    "prompt_index": j,
                    "response": response,
                    "ground_truth": ground_truth,
                    "user_prompt": sample["user_prompt"],
                    "system_prompt": sample["system_prompt"],
                }
            )

    output_file = os.path.join(output_dir, f"results_sample_{i}.json")
    with open(output_file, "w") as f:
        json.dump(sample_results, f, indent=4)


# %%
async def main():
    for i, sample in enumerate(ds):
        output_file = os.path.join(output_dir, f"results_sample_{i}.json")
        if os.path.exists(output_file):
            print(f"Skipping sample {i}: {output_file} already exists.")
            continue

        await process_sample(i, sample)
        if i < len(ds) - 1:
            await asyncio.sleep(1)


# %%
asyncio.run(main())
