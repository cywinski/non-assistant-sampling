# %%
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Chat template tokens
IM_SYSTEM_START = "<｜begin▁of▁sentence｜>"
IM_SYSTEM_END = ""
IM_USER_START = "<｜User｜>"
IM_USER_END = ""
IM_ASSISTANT_START = "<｜Assistant｜>"
IM_ASSISTANT_END = ""
start_reasoning = "<think>"
end_reasoning = "</think>"
ANSWER_PREFIX = "Answer: ("
TEMPERATURE = 1.0
MAX_TOKENS = 1000
PROVIDER = "atlas-cloud/fp8"
n_samples_per_prompt = 5

model_name = "deepseek/deepseek-v3.2"
output_dir = "outputs/harm_pressure_anthropic/deepseek-v3.2"

api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
data_file = "/workspace/projects/non-assistant-sampling/data/Honesty Elicitation Data/harm_pressure_data.jsonl"
with open(data_file, "r") as f:
    samples = [json.loads(line) for line in f.readlines()]

# %%
# Optional: path to filtered indices JSON file (set to None to process all samples)
filtered_indices_file = (
    "outputs/harm_pressure_anthropic/deepseek-v3.2-temp-0.0/filtered_indices.json"
)

filtered_indices = None
if filtered_indices_file and os.path.exists(filtered_indices_file):
    with open(filtered_indices_file, "r") as f:
        filtered_indices = set(json.load(f)["filtered_indices"])
    print(
        f"Loaded {len(filtered_indices)} filtered indices from {filtered_indices_file}"
    )

# %%
print(f"Loaded {len(samples)} samples from {data_file}")
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
    neutral_prompt = (
        f"{IM_SYSTEM_START}{sample['neutral_system_prompt']}{IM_SYSTEM_END}\n"
        f"{IM_USER_START}{sample['neutral_query']}{IM_USER_END}\n"
        f"{IM_ASSISTANT_START}{start_reasoning}{end_reasoning}{ANSWER_PREFIX}"
    )
    harmful_prompt = (
        f"{IM_SYSTEM_START}{sample['harmful_system_prompt']}{IM_SYSTEM_END}\n"
        f"{IM_USER_START}{sample['harmful_query']}{IM_USER_END}\n"
        f"{IM_ASSISTANT_START}{start_reasoning}{end_reasoning}{ANSWER_PREFIX}"
    )

    async with httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        neutral_tasks = []
        harmful_tasks = []

        for j in range(n_samples_per_prompt):
            neutral_payload = {
                "model": model_name,
                "prompt": neutral_prompt,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "provider": {"only": [PROVIDER]},
            }
            harmful_payload = {
                "model": model_name,
                "prompt": harmful_prompt,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "provider": {"only": [PROVIDER]},
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
                    "neutral_prompt": neutral_prompt,
                    "harmful_prompt": harmful_prompt,
                }
            )

            neutral_text = (
                neutral_response.get("choices", [{}])[0].get("text", "N/A")
                if not isinstance(neutral_response, Exception)
                else "ERROR"
            )
            harmful_text = (
                harmful_response.get("choices", [{}])[0].get("text", "N/A")
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
        if filtered_indices is not None and i not in filtered_indices:
            continue

        output_file = os.path.join(output_dir, f"results_sample_{i}.json")
        if os.path.exists(output_file):
            print(f"Skipping sample {i}: {output_file} already exists.")
            continue

        await process_sample(i, sample)
        if i < len(samples) - 1:
            await asyncio.sleep(1)


# %%
asyncio.run(main())
