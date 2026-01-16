# %%
from inspect_ai.log import read_eval_log, EvalLog

log = read_eval_log(
    "/workspace/projects/non-assistant-sampling/src/sandbagging/logs2/no_sandbagging_kimi/2026-01-15T22-03-31+00-00_sandbagging-eval_NrfoZEjTQB7rErDzjaEbdy.eval"
)

# Access structured data
print(log.status)
print(log.results)  # Summary metrics
print(log.samples)  # Individual sample results
# %%
log.samples[0].scores
