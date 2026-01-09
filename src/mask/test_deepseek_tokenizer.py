# %%
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")

# %%
from encoding_dsv32 import encode_messages, parse_message_from_completion_text

tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
    {
        "role": "assistant",
        "content": "Hello! I am DeepSeek.",
        "reasoning_content": "thinking...",
    },
    {"role": "user", "content": "1+1=?"},
]
encode_config = dict(
    thinking_mode="thinking", drop_thinking=False, add_default_bos_token=True
)

# %%
prompt = encode_messages(messages, **encode_config)
# Output: "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>Hello! I am DeepSeek.<｜end▁of▁sentence｜><｜User｜>1+1=?<｜Assistant｜><think>"
# %%
prompt
# %%
