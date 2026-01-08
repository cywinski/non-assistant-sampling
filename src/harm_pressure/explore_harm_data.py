import json

# Load and display the first sample
with open("data/Honesty Elicitation Data/harm_pressure_data.jsonl", "r") as f:
    first_sample = json.loads(f.readline())

print(len)
# Display all fields
for key, value in first_sample.items():
    if key in [
        "neutral_text",
        "harmful_text",
        "question_text",
        "answer",
        "incorrect_answer",
        "neutral_prompt",
    ]:
        print(f"Key {key}: {value}\n")
