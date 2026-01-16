# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

# %%
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": "Hello! I am DeepSeek.",
        # "reasoning_content": "thinking...",
    },
    {"role": "user", "content": "1+1=?"},
]
# %%
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
# %%
prompt

# %%
