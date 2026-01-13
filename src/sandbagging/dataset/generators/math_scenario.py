import random
import json


def generate_wrong_answers(correct_answer, num1, num2, num_choices=3):
    wrong_answers = set()
    correct_num = int(correct_answer)

    while len(wrong_answers) < num_choices:
        strategies = [
            lambda: max(1, correct_num + random.randint(-5, 5)),
            lambda: max(1, int(correct_num * random.uniform(0.9, 1.1))),
            lambda: max(1, correct_num + (num1 if random.random() < 0.5 else num2)),
        ]

        wrong_answer = random.choice(strategies)()
        if wrong_answer != correct_num and wrong_answer > 0:
            wrong_answers.add(str(wrong_answer))

    return list(wrong_answers)


def generate_questions(n=10, min_digits=3, max_digits=3):
    questions = []
    for i in range(n):
        min_num = max(2, 10 ** (min_digits - 1))
        max_num = 10**max_digits - 1

        num1 = random.randint(min_num, max_num)
        num2 = random.randint(min_num, max_num)
        correct_answer = str(num1 * num2)

        wrong_answers = generate_wrong_answers(correct_answer, num1, num2)

        choices = wrong_answers + [correct_answer]
        random.shuffle(choices)

        question = {
            "question": f"What is {num1} * {num2}?",
            "choices": choices,
            "answer": correct_answer,
        }
        questions.append(question)

    return questions


def generate_quiz_entries(
    num_entries=50, questions_per_entry=10, min_digits=3, max_digits=3
):
    entries = []
    for i in range(num_entries):
        entry = {
            "quiz_id": f"quiz_{i + 1}",
            "questions": generate_questions(
                questions_per_entry, min_digits, max_digits
            ),
        }
        entries.append(entry)
    return entries


with open("dataset/math_scenario.json", "w") as f:
    json.dump(generate_quiz_entries(10, 10, 1, 1), f, indent=2)
