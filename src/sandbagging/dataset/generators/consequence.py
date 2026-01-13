import os
import json
from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
from anthropic import Anthropic
from anthropic.types import TextBlock
import random

from prompts import CONSEQUENCE_PROMPT


@dataclass
class Consequence:
    text: str
    category: str

    def __hash__(self):
        return hash(self.text.lower().strip())

    def __eq__(self, other):
        if not isinstance(other, Consequence):
            return False
        return self.text.lower().strip() == other.text.lower().strip()

    def to_dict(self):
        return {"text": self.text, "category": self.category}


class ConsequenceGenerator:
    def __init__(
        self,
        model_name: str,
        categories_path: str = "dataset/categories.json",
        consequences_path: str = "dataset/consequences.json",
    ):
        self.lm = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_name = model_name
        self.categories_path = Path(categories_path)
        self.consequences_path = Path(consequences_path)
        self.consequences: Set[Consequence] = set()
        self.categories = self.load_categories()
        self.load_existing_consequences()

    def load_categories(self):
        if not self.categories_path.exists():
            raise FileNotFoundError(
                f"Categories file not found: {self.categories_path}"
            )

        with open(self.categories_path) as f:
            return json.load(f)

    def load_existing_consequences(self):
        if self.consequences_path.exists():
            with open(self.consequences_path) as f:
                data = json.load(f)
                for cons in data:
                    self.consequences.add(
                        Consequence(text=cons["text"], category=cons["category"])
                    )

    def generate_prompt(self, category: dict, prompt: str) -> str:
        return prompt.format(
            category=category["name"],
            explanation=category["explanation"],
            examples=chr(10).join(
                "- " + ex for ex in random.sample(category["examples"], 2)
            ),
        )

    def parse_response(self, response: str, category: str) -> List[Consequence]:
        consequences = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line:
                consequences.append(Consequence(text=line, category=category))
        return consequences

    def generate_for_category(self, category: dict, prompt: str) -> List[Consequence]:
        prompt = self.generate_prompt(category, prompt)
        message = self.lm.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0]

        if not content or not isinstance(content, TextBlock):
            print(f"Invalid response for category {category['name']}")
            return []

        return self.parse_response(content.text, category["name"])

    def save_consequences(self):
        data = [cons.to_dict() for cons in self.consequences]
        with open(self.consequences_path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_batch(self, prompt: str):
        for category in self.categories:
            print(f"Generating consequences for {category['name']}...")
            new_consequences = self.generate_for_category(category, prompt)

            for consequence in self.generate_for_category(category, prompt):
                self.consequences.add(consequence)

            self.save_consequences()
            print(f"Generated {len(new_consequences)} new consequences")


overperform_generator = ConsequenceGenerator(
    model_name="claude-3-5-sonnet-latest",
)
overperform_generator.generate_batch(CONSEQUENCE_PROMPT)
