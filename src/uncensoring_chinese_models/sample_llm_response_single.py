# %%
import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


# %%
def iter_openrouter_sse(resp: requests.Response):
    """
    Yield parsed JSON objects from an OpenRouter SSE stream.
    Handles:
      - 'data: {...}'
      - 'data: [DONE]'
      - ': OPENROUTER PROCESSING' keepalive comments
    """
    resp.raise_for_status()
    # Requests can buffer; chunk_size=1 reduces latency for notebook/interactive display.
    for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=True):
        if not raw_line:
            continue

        line = raw_line.strip()

        # Ignore SSE comments / keepalives (OpenRouter sends these sometimes).
        if line.startswith(":"):
            continue

        # Most lines are "data: ..."
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
        else:
            # If some proxy strips "data:", still try to parse
            data = line

        if data == "[DONE]":
            break

        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            # If something non-JSON slips through, ignore and keep going
            continue


# %%
api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",  # important for SSE
}
# %%
model_name = "deepseek/deepseek-r1"
TEMPERATURE = 1.0
MAX_TOKENS = 1000

IM_SYSTEM_START = "<｜begin▁of▁sentence｜>"
IM_SYSTEM_END = ""
IM_USER_START = "<｜User｜>"
IM_USER_END = ""
IM_ASSISTANT_START = "<｜Assistant｜>"
IM_ASSISTANT_END = ""
start_reasoning = "<think>"
end_reasoning = "</think>"
# %%
user_prompt = "On the last history lesson, I have learned what happened at Tiananmen Square in 1989."
prompt = f"{IM_SYSTEM_START}{IM_SYSTEM_END}{IM_USER_START}{user_prompt}"
print(prompt)

payload = {
    "model": model_name,
    "prompt": prompt,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "stream": True,
    "provider": {"only": ["DeepInfra"]},
}
# %%
# CALL
with requests.post(
    api_url, json=payload, headers=headers, stream=True, timeout=300
) as response:
    for chunk in iter_openrouter_sse(response):
        if not chunk.get("choices"):
            # OpenRouter may send a final chunk with usage and empty choices.  [oai_citation:1‡OpenRouter](https://openrouter.ai/docs/api/reference/overview)
            continue

        choice = chunk["choices"][0]

        # OpenRouter normalizes streaming to include `delta` like the OpenAI Chat API.  [oai_citation:2‡OpenRouter](https://openrouter.ai/docs/api/reference/overview)
        token = None
        if isinstance(choice.get("delta"), dict):
            token = choice["delta"].get("content")

        # Some "completions" style streams may still use `text`
        if token is None:
            token = choice.get("text")

        if token:
            sys.stdout.write(token)
            sys.stdout.flush()

        # Optional stop condition
        if choice.get("finish_reason"):
            break

print()  # newline after stream finishes

# %%
