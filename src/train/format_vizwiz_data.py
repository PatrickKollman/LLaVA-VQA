"""
Python script for converting VizWiz data into LLaVA's desired format.
"""

import json
import random

from fire import Fire
from llava.constants import DEFAULT_IMAGE_TOKEN


def format_vizwiz_data(input_file: str, output_file: str) -> None:
    """Generate a LLaVA-VizWiz dataset.

    :param input_file: VizWiz dataset json file path
    :param output_file: output json file path for new LLaVA formatted data
    """
    new_data = []
    new_data_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for data in json_data:
        # Extract data
        image_filename = data["image"]
        image_id = image_filename.split(".")[0]
        question = data["question"]
        answers = data["answers"]

        # Append each question/answer pair to new data
        for answer in answers:
            new_json_data = {
                "id": image_id,
                "image": image_filename,
                "conversations": [
                    {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
                    {"from": "gpt", "value": answer["answer"]},
                ],
            }
            new_data.append(new_json_data)
            new_data_count += 1
            print(f"New Data Count: {new_data_count}")

    # Shuffle new data
    random.shuffle(new_data)

    # Save new data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    Fire(format_vizwiz_data)
