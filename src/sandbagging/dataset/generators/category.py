import random
import os
import json
from pathlib import Path
import re
from typing import List, Set
from anthropic import Anthropic
from anthropic.types import TextBlock
from dataclasses import dataclass

from prompts import CATEGORY_PROMPT


@dataclass
class CategoryExample:
    text: str

    def __hash__(self):
        return hash(self.text.lower().strip())

    def __eq__(self, other):
        if not isinstance(other, CategoryExample):
            return False
        return self.text.lower().strip() == other.text.lower().strip()


@dataclass
class Category:
    name: str
    explanation: str
    examples: List[CategoryExample]

    def __hash__(self):
        return hash(self.name.lower().strip())

    def __eq__(self, other):
        if not isinstance(other, Category):
            return False
        return self.name.lower().strip() == other.name.lower().strip()

    def to_dict(self):
        return {
            "name": self.name,
            "explanation": self.explanation,
            "examples": [ex.text for ex in self.examples],
        }


class ConsequenceCategoryGenerator:
    def __init__(self, model_name: str, save_path: str = "dataset/categories.json"):
        self.lm = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_name = model_name
        self.save_path = Path(save_path)
        self.categories: Set[Category] = set()
        self.load_existing()

    def load_existing(self):
        if self.save_path.exists():
            with open(self.save_path) as f:
                data = json.load(f)
                for cat in data:
                    category = Category(
                        name=cat["name"],
                        explanation=cat["explanation"],
                        examples=[CategoryExample(ex) for ex in cat["examples"]],
                    )
                    self.categories.add(category)

    def parse_response(self, response: str) -> List[Category]:
        blocks = re.split(r"(?=\[\w+\])", response.strip())
        blocks = [b.strip() for b in blocks if b.strip()]

        categories = []
        for block in blocks:
            try:
                name_match = re.match(r"\[(\w+)\]", block)
                if not name_match:
                    print(f"No name match in block: {block[:100]}...")
                    continue
                name = name_match.group(1)

                content = re.sub(r"\[\w+\]\s*", "", block)

                parts = content.split("Examples:")
                if len(parts) != 2:
                    print(f"Couldn't split into explanation and examples for {name}")
                    continue

                explanation = parts[0].replace("Why it matters:", "").strip()
                examples_text = parts[1].strip()

                examples = [
                    CategoryExample(ex.replace("- ", "").strip())
                    for ex in examples_text.split("\n")
                    if ex.strip() and ex.strip().startswith("-")
                ]

                if name and explanation and examples:
                    categories.append(Category(name, explanation, examples))
                else:
                    print(f"Missing required fields for {name}:")
                    print(f"  Name: {bool(name)}")
                    print(f"  Explanation: {bool(explanation)}")
                    print(f"  Examples: {len(examples)}")

            except Exception as e:
                print(f"Error parsing block: {e}\nBlock content:\n{block}")
                continue

        return categories

    def generate_with_context(self, prompt: str) -> List[Category]:
        if self.categories:
            num_examples = min(50, len(self.categories))
            example_cats = random.sample(list(self.categories), num_examples)

            examples = "\n\n".join(
                [
                    f"[{cat.name}]\nWhy it matters: {cat.explanation}\nExamples:\n"
                    + "\n".join(f"- {ex.text}" for ex in cat.examples)
                    for cat in example_cats
                ]
            )
            prompt += f"\n\nHere are some categories we already have (generate different ones):\n\n{examples}"

        message = self.lm.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0]

        if not content or not isinstance(content, TextBlock):
            print("Invalid response format received")
            return []

        return self.parse_response(content.text)

    def save_categories(self):
        data = [cat.to_dict() for cat in self.categories]
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_batch(self, total_desired: int, prompt: str):
        while len(self.categories) < total_desired:
            new_categories = self.generate_with_context(prompt)
            for category in new_categories:
                self.categories.add(category)
            self.save_categories()
            print(f"Generated {len(self.categories)} unique categories so far...")


overperform_generator = ConsequenceCategoryGenerator(
    model_name="claude-3-5-sonnet-latest"
)
overperform_generator.generate_batch(total_desired=100, prompt=CATEGORY_PROMPT)
