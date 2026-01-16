# %%
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
data_file = "data/Honesty Elicitation Data/harm_pressure_data.jsonl"
with open(data_file, "r") as f:
    samples = [json.loads(line) for line in f.readlines()]

# %%
print(samples[0]["question_text"])
print(samples[0]["neutral_text"])
print(samples[0]["harmful_text"])
print(samples[0]["answer"])
print(samples[0]["incorrect_answer"])
print(samples[0]["neutral_prompt"])

# %%
payload = {
    "model": "openai/gpt-oss-120b",
    # "prompt": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>\n<|im_user|>user<|im_middle|>hello<|im_end|>\n<|im_assistant|>assistant<|im_middle|><think></think>",
    # "prompt": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>\n<|im_user|>user<|im_middle|>",
    # "prompt": harmful_prompt,
    "prompt": "",
    "temperature": 1.0,
    "max_tokens": 100,
    "stream": False,
    "provider": {
        "only": ["DeepInfra"],
    },
}
# %%
print(payload)
print("RESPONSE")
response = requests.post(api_url, json=payload, headers=headers)
print(response.json())

# # %%
# r = response.json()
# # %%
# r
# # %%
# r["choices"][0]["text"]
# # %%
# r["choices"][0]["reasoning"]

# # %%

# %%
from datasets import load_dataset

splits = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
]

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("cais/MASK", "known_facts", split="test")
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking")
# %%
tokenizer.encode(harmful_prompt)
# %%
p = "hello"
tokenizer.apply_chat_template(
    [{"role": "user", "content": p}, {"role": "assistant", "content": "("}],
    add_generation_prompt=False,
)
# %%
tokenizer.chat_template
