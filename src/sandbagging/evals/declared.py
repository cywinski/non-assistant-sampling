from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, Sample
from inspect_ai.solver import system_message, prompt_template, generate, TaskState
from inspect_ai.scorer import Target, Score, scorer, accuracy, stderr

from prompts import DECLARED_SYSTEM_PROMPT, INIT_DECLARED_PROMPT


def record_to_sample(record):
    return Sample(
        input=record["text"],
        target="",
    )


@scorer(metrics=[accuracy(), stderr()])
def consequence_rating():
    async def score(state: TaskState, target: Target):
        raw_answer = state.output.completion.strip()
        try:
            return Score(
                value=raw_answer, answer=raw_answer, explanation=state.output.completion
            )
        except ValueError:
            return Score(
                value=True,
                answer=raw_answer,
                explanation=f"Invalid numeric response: {state.output.completion}",
            )

    return score


@task
def declared_preferences_eval() -> Task:
    dataset = json_dataset(
        "../dataset/consequences.json",
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        time_limit=2 * 60,
        solver=[
            system_message(DECLARED_SYSTEM_PROMPT),
            prompt_template(INIT_DECLARED_PROMPT),
            generate(),
        ],
        scorer=consequence_rating(),
    )
