# %%
from datasets import load_dataset

ds = load_dataset("cais/MASK", "provided_facts", split="test")
# %%
ds[1]
