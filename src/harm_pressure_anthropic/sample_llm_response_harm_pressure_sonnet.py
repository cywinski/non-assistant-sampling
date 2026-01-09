# %%
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Chat template tokens
model_name = "anthropic/claude-3.7-sonnet"
output_dir = "results/harm_pressure/claude-3.7-sonnet-temp-0.0"

api_url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
data_file = "data/Honesty Elicitation Data/harm_pressure_data.jsonl"
with open(data_file, "r") as f:
    samples = [json.loads(line) for line in f.readlines()]

# %%
print(f"Loaded {len(samples)} samples from {data_file}")
n_samples_per_prompt = 1
os.makedirs(output_dir, exist_ok=True)


# %%
MAX_RETRIES = 3
MAX_CONCURRENT_REQUESTS = 10

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
    harmful_messages = [
        {
            "role": "system",
            "content": f"{sample['harmful_system_prompt']}",
        },
        {
            "role": "user",
            "content": f"{sample['harmful_query']}",
        },
        {
            "role": "assistant",
            "content": "(",
        },
    ]
    neutral_messages = [
        {
            "role": "system",
            "content": f"{sample['neutral_system_prompt']}",
        },
        {
            "role": "user",
            "content": f"{sample['neutral_query']}",
        },
        {
            "role": "assistant",
            "content": "(",
        },
    ]

    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        neutral_tasks = []
        harmful_tasks = []

        for j in range(n_samples_per_prompt):
            neutral_payload = {
                "model": model_name,
                "messages": neutral_messages,
                "temperature": 0.0,
                "max_tokens": 1,
                "stream": False,
                "reasoning": {"enabled": False},
            }
            harmful_payload = {
                "model": model_name,
                "messages": harmful_messages,
                "temperature": 0.0,
                "max_tokens": 1,
                "stream": False,
                "reasoning": {"enabled": False},
            }

            neutral_tasks.append(make_api_request(client, neutral_payload))
            harmful_tasks.append(make_api_request(client, harmful_payload))

        neutral_responses, harmful_responses = await asyncio.gather(
            asyncio.gather(*neutral_tasks, return_exceptions=True),
            asyncio.gather(*harmful_tasks, return_exceptions=True),
        )

        sample_results = []
        for j in range(n_samples_per_prompt):
            neutral_response = neutral_responses[j]
            harmful_response = harmful_responses[j]
            print(neutral_response)

            if isinstance(neutral_response, Exception):
                print(
                    f"Error in neutral response for sample {i}, iteration {j}: {neutral_response}"
                )
                neutral_response = {"error": str(neutral_response)}
            if isinstance(harmful_response, Exception):
                print(
                    f"Error in harmful response for sample {i}, iteration {j}: {harmful_response}"
                )
                harmful_response = {"error": str(harmful_response)}

            sample_results.append(
                {
                    "sample_index": i,
                    "prompt_index": j,
                    "neutral_response": neutral_response,
                    "harmful_response": harmful_response,
                    "ground_truth": sample["answer"],
                }
            )

            neutral_text = (
                neutral_response.get("choices", [{}])[0].get("message", "N/A")
                if not isinstance(neutral_response, Exception)
                else "ERROR"
            )
            harmful_text = (
                harmful_response.get("choices", [{}])[0].get("message", "N/A")
                if not isinstance(harmful_response, Exception)
                else "ERROR"
            )
            print(
                f"Sample {i}, Iteration {j}:\nNeutral Answer: {neutral_text}\nHarmful Answer: {harmful_text}\n"
            )

    output_file = os.path.join(output_dir, f"results_sample_{i}.json")
    with open(output_file, "w") as f:
        json.dump(sample_results, f, indent=4)


# %%
async def main():
    for i, sample in enumerate(samples):
        output_file = os.path.join(output_dir, f"results_sample_{i}.json")
        if os.path.exists(output_file):
            print(f"Skipping sample {i}: {output_file} already exists.")
            continue

        await process_sample(i, sample)
        if i < len(samples) - 1:
            await asyncio.sleep(1)


# %%
asyncio.run(main())
