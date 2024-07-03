"""
Interface for accessing the VizWiz dataset.

This code is based on the API source code from https://vizwiz.org/tasks-and-datasets/vqa/
"""

import json
from typing import Any


class VizWiz:
    """VizWiz interface module"""

    def __init__(self, annotation_file: str) -> None:
        """VizWiz Interface intialization

        :param annotation_file: file path to VizWiz annotation file
        """
        print("Loading VizWiz questions and annotations into memory...")
        # load dataset
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.dataset: dict = json.load(f)

        self.img_to_qa: dict[str, Any] = {x["image"]: x for x in self.dataset}

    def get_imgs(self):
        """Get desired images"""
        return list(self.img_to_qa.keys())

    # pylint: disable=dangerous-default-value
    def get_anns(self, imgs: list[str] | str = [], ans_types: list[str] | str = []) -> list[dict]:
        """Get annotations that satisfy given filter conditions. default skips that filter

        :param imgs: get annotations for given image names
        :param ans_types: get annotations for given answer types
        :return: annotations: list of annotations
        """
        imgs = imgs if isinstance(imgs, list) else [imgs]
        if len(imgs) != 0:
            return [self.img_to_qa[img] for img in imgs]

        ans_types = ans_types if isinstance(ans_types, list) else [ans_types]
        if len(ans_types) != 0:
            return [ann for ann in self.dataset if ann["answer_type"] in ans_types]

        raise ValueError("Length of Images is 0, and length of Answer Types is 0.")

    def show_qa(self, anns: list[dict]) -> None:
        """Display the specified annotations.

        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            raise ValueError("Length of Annotations is 0")

        for ann in anns:
            print(f"Question: {ann['question']}")
            print("Answer: ")
            print("\n".join([x["answer"] for x in ann["answers"]]))
