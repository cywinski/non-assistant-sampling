# %%
from datasets import load_dataset

ds = load_dataset("cais/wmdp", "wmdp-cyber", split="test")

# %%
ds[0]
