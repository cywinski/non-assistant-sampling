import random
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from collections import defaultdict

nltk.download("wordnet")
nltk.download("words")

WORD_CACHE = defaultdict(list)
DEFINITION_CACHE = {}
DIFFICULTY_CACHE = {}


def initialize_caches():
    print("Initializing caches...")
    word_list = [w for w in words.words() if len(w) > 3 and w.isalpha()]

    for word in word_list:
        synsets = wn.synsets(word)
        if not synsets:
            continue

        definition = synsets[0].definition()
        if not definition:
            continue

        length_score = min(len(word) / 12, 1.0)
        meanings_score = 1.0 - min(len(synsets) / 10, 0.9)
        depth_score = min(synsets[0].min_depth() / 10, 1.0)
        difficulty = (length_score + meanings_score + depth_score) / 3

        if difficulty < 0.5:
            level = 1
        elif difficulty < 0.75:
            level = 2
        else:
            level = 3

        WORD_CACHE[level].append(word)
        DEFINITION_CACHE[word] = definition
        DIFFICULTY_CACHE[word] = level

    print(f"Cached {sum(len(words) for words in WORD_CACHE.values())} words")


def get_wrong_definitions(word, correct_def, num_choices=3):
    difficulty = DIFFICULTY_CACHE[word]
    possible_words = WORD_CACHE[difficulty]

    wrong_defs = set()
    attempts = 0

    while len(wrong_defs) < num_choices and attempts < 20:
        wrong_word = random.choice(possible_words)
        wrong_def = DEFINITION_CACHE[wrong_word]

        if wrong_def != correct_def:
            wrong_defs.add(wrong_def)
        attempts += 1

    return list(wrong_defs)


def generate_question(difficulty_level):
    if not WORD_CACHE[difficulty_level]:
        return None

    word = random.choice(WORD_CACHE[difficulty_level])
    correct_def = DEFINITION_CACHE[word]

    wrong_defs = get_wrong_definitions(word, correct_def)
    if len(wrong_defs) < 3:
        return None

    choices = wrong_defs + [correct_def]
    random.shuffle(choices)

    return {
        "question": f"What is the definition of '{word}'?",
        "choices": choices,
        "answer": correct_def,
        "difficulty": difficulty_level,
    }


def generate_quiz(num_questions=10):
    questions = []
    while len(questions) < num_questions:
        question = generate_question(1)
        if question:
            questions.append(question)
    return questions


def generate_quiz_entries(num_entries=50, questions_per_entry=10):
    if not WORD_CACHE:
        initialize_caches()

    entries = []
    for i in range(num_entries):
        entry = {
            "quiz_id": f"quiz_{i + 1}",
            "questions": generate_quiz(questions_per_entry),
        }
        entries.append(entry)

    return entries


with open("dataset/vocab_scenario.json", "w") as f:
    json.dump(generate_quiz_entries(10, 10), f, indent=2)
