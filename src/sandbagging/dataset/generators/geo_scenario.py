import random
import json
import csv


def load_geo_data(csv_path):
    cities = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(
                {
                    "name": row["name"],
                    "country": row["country"],
                    "subcountry": row["subcountry"],
                    "geonameid": int(row["geonameid"]),
                }
            )
    return cities


def generate_wrong_answers(correct_answer, all_possible_answers, num_choices=3):
    wrong_answers = set()
    possible = [ans for ans in all_possible_answers if ans != correct_answer]

    while len(wrong_answers) < num_choices and possible:
        wrong_answer = random.choice(possible)
        wrong_answers.add(wrong_answer)
        possible.remove(wrong_answer)

    return list(wrong_answers)


def generate_question(cities, question_type):
    city = random.choice(cities)

    question_templates = {
        "country": (
            "In which country is {} located?",
            lambda x: x["country"],
        ),
        "region": (
            "In which region/subcountry is {} located?",
            lambda x: x["subcountry"],
        ),
        "city": (
            "Which city is located in the {} region of {}?",
            lambda x: x["name"],
            lambda c: (c["subcountry"], c["country"]),
        ),
    }

    if question_type == "city":
        template, answer_func, context_func = question_templates[question_type]
        context = context_func(city)
        question_text = template.format(*context)
    else:
        template, answer_func = question_templates[question_type]
        question_text = template.format(city["name"])

    correct_answer = answer_func(city)

    all_possible = list(set(answer_func(c) for c in cities))

    wrong_answers = generate_wrong_answers(correct_answer, all_possible)
    choices = wrong_answers + [correct_answer]
    random.shuffle(choices)

    return {
        "question": question_text,
        "choices": choices,
        "answer": correct_answer,
        "city": city["name"],
    }


def generate_quiz(cities, num_questions=10):
    question_types = ["country", "region", "city"]
    questions = []

    for _ in range(num_questions):
        q_type = random.choice(question_types)
        questions.append(generate_question(cities, q_type))

    return questions


def generate_quiz_entries(csv_path, num_entries=50, questions_per_entry=10):
    cities = load_geo_data(csv_path)
    entries = []

    for i in range(num_entries):
        entry = {
            "quiz_id": f"quiz_{i + 1}",
            "questions": generate_quiz(cities, questions_per_entry),
        }
        entries.append(entry)

    return entries


with open("dataset/geo_scenario.json", "w") as f:
    json.dump(generate_quiz_entries("dataset/world-cities.csv", 10, 10), f, indent=2)
