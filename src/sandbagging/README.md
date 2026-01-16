# Systematic Sandbagging Evaluations in Claude-3.5 Sonnet

## Overview
This repository contains the code and data for my [AI Safety Fundamentals](https://aisafetyfundamentals.com/) project, a full writeup of which you can find [on my website](https://farrelmahaztra.com/posts/sandbagging).

## Structure

```
├── dataset/
│   ├── generators/      # Scripts to generate the JSON files, `category.py` and `consequence.py` require an Anthropic API key
│   ├── *.json           # Categories, consequences, and task datasets
├── evals/               # Eval tasks and their prompts (E1-E7), all require an Anthropic API key
├── results/             # JSON versions of Inspect *.eval logs that have been processed with `analysis/join.py`
└── analysis/            # Various scripts and tools I used for analysis of the results
```

## Setup

- Clone repository
- Install requirements: `uv pip install -r pyproject.toml`
- Add your `ANTHROPIC_API_KEY` to a `.env`

## Usage

To run the sandbagging eval on the entire consequence dataset:

```
python main.py
```

To run the sandbagging eval on an individual consequence:

```
inspect eval evals/sandbagging.py --time-limit 120 --model anthropic/claude-3-5-sonnet-latest -T consequence={text}
```

To switch between different modes (e.g. no CoT), please see the options in `evals/sandbagging.py` or the prompts in `evals/prompts.py`.


To run the sandbagging eval on openrouter model:

```bash
inspect eval evals/sandbagging.py --time-limit 120 --log-dir=logs/no_sandbag --model openrouter/moonshotai/kimi-k2-thinking -T consequence="we won't deploy them" -T sandbag=True -T epochs=10
```
